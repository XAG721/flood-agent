from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.frc_rag.conflicts_evaluation import (
    approximate_token_count,
    bm25_scores,
    local_model_snapshot,
    minmax,
    rank_desc,
    read_jsonl,
    rrf_scores,
    stable_hash,
    write_jsonl,
)
from research.frc_rag.public_evidence import paired_bootstrap


HOUSING_ABLATION_SCHEMA = "frc-housing-field-applicability-ablation-v1"
HOUSING_SOURCE_REPOSITORY = "reglab/housing_qa"
HOUSING_SOURCE_REVISION = "761550cc974fa1d9141ffd39014db89efa2a7230"
HOUSING_SOURCE_URL = (
    "https://huggingface.co/datasets/reglab/housing_qa/resolve/"
    f"{HOUSING_SOURCE_REVISION}/data/questions.json.zip?download=true"
)
HOUSING_SOURCE_ZIP_SHA256 = "7e7722c267d44ecc1f9f52d69109116d8a0a352b89f002438585872f67de9985"
HOUSING_SOURCE_JSON_SHA256 = "4f6eb35e1b865a6c5cfd0cf73a0b514a86bd1d8306bf6719f2b4fb94793c7e4b"
HOUSING_SOURCE_LICENSE = "CC-BY-SA-4.0"
HOUSING_SNAPSHOT_YEAR = 2021
DEFAULT_CASE_COUNT = 40
DEFAULT_FIELDS_PER_CASE = 4
DEFAULT_SAME_JURISDICTION_DISTRACTORS = 2
DEFAULT_SEED = 20260713
DEFAULT_TOP_K = 4
DEFAULT_TOKEN_BUDGET = 1200
DEFAULT_FIELD_THRESHOLD = 0.55
DEFAULT_ROLE_THRESHOLD = 0.55
DEFAULT_FIELD_WEIGHT = 2.0
DEFAULT_ROLE_WEIGHT = 1.0

HOUSING_ROLES = {
    "governing_rule": "Find the governing statutory rule that directly answers the requested fields.",
    "condition_exception": (
        "Find statutory conditions, exceptions, qualifications, caveats, or limitations relevant "
        "to the requested fields."
    ),
    "source_attribution": "Find the statute citation or legal authority supporting the answer.",
}

HOUSING_METHODS = (
    "bm25_topk",
    "dense_topk",
    "hybrid_topk",
    "cross_encoder_topk",
    "field_decomposition_topk",
    "frc_full",
    "w/o_field",
    "w/o_applicability",
)

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _tokens(value: str) -> set[str]:
    return set(_WORD_RE.findall(value.lower()))


def _lexical_overlap(query: str, passage: str) -> float:
    left = _tokens(query)
    right = _tokens(passage)
    return len(left & right) / max(1, len(left | right))


def _normalized_answer(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    if text.startswith("unknown") or text.startswith("insufficient"):
        return "Unknown"
    return None


def _single_statute_record(row: dict[str, Any], *, max_tokens: int) -> dict[str, Any] | None:
    statutes = row.get("statutes", [])
    answer = _normalized_answer(row.get("answer"))
    if len(statutes) != 1 or answer not in {"Yes", "No"}:
        return None
    statute = statutes[0]
    excerpt = str(statute.get("excerpt") or "").strip()
    citation = str(statute.get("citation") or "").strip()
    question = str(row.get("question") or "").strip()
    state = str(row.get("state") or "").strip()
    if not excerpt or not citation or not question or not state:
        return None
    token_count = approximate_token_count(f"{citation} {excerpt}")
    if token_count < 8 or token_count > max_tokens:
        return None
    try:
        record = {
            "idx": int(row["idx"]),
            "state": state,
            "question": question,
            "answer": answer,
            "question_group": int(row["question_group"]),
            "statute_idx": int(statute["statute_idx"]),
            "citation": citation,
            "excerpt": excerpt,
            "token_count": token_count,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid HousingQA row: {row!r}") from exc
    return record


def _candidate(record: dict[str, Any], origin: str) -> dict[str, Any]:
    return {
        "id": f"housing-statute-{record['statute_idx']}",
        "text": f"{record['citation']}\n{record['excerpt']}",
        "citation": record["citation"],
        "token_count": record["token_count"],
        "metadata": {
            "jurisdiction": record["state"],
            "snapshot_year": HOUSING_SNAPSHOT_YEAR,
            "source_question_idx": record["idx"],
            "source_question_group": record["question_group"],
            "candidate_origin": [origin],
            "source_dataset": HOUSING_SOURCE_REPOSITORY,
        },
    }


def _merge_candidate(
    candidates: dict[str, dict[str, Any]], record: dict[str, Any], origin: str
) -> None:
    value = _candidate(record, origin)
    existing = candidates.get(value["id"])
    if existing is None:
        candidates[value["id"]] = value
        return
    origins = set(existing["metadata"].get("candidate_origin", []))
    origins.add(origin)
    existing["metadata"]["candidate_origin"] = sorted(origins)


def build_housing_composite_cases(
    rows: Iterable[dict[str, Any]],
    *,
    case_count: int = DEFAULT_CASE_COUNT,
    fields_per_case: int = DEFAULT_FIELDS_PER_CASE,
    same_jurisdiction_distractors: int = DEFAULT_SAME_JURISDICTION_DISTRACTORS,
    seed: int = DEFAULT_SEED,
    max_candidate_tokens: int = 220,
) -> list[dict[str, Any]]:
    """Build a frozen multi-field retrieval test from public expert HousingQA annotations.

    Each original question becomes one task field. Its single annotated statute is the field's
    gold evidence. Wrong-jurisdiction hard negatives use the same original question group with
    the opposite annotated answer. Gold answers and evidence maps are retained for scoring only.
    """

    if case_count <= 0 or fields_per_case <= 0 or same_jurisdiction_distractors < 0:
        raise ValueError("case and field counts must be positive")
    records = [
        parsed
        for row in rows
        if (parsed := _single_statute_record(row, max_tokens=max_candidate_tokens)) is not None
    ]
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_state[record["state"]].append(record)
        by_group[record["question_group"]].append(record)

    def opposite_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            other
            for other in by_group[record["question_group"]]
            if other["state"] != record["state"] and other["answer"] != record["answer"]
        ]

    eligible_by_state: dict[str, list[dict[str, Any]]] = {
        state: [record for record in state_records if opposite_candidates(record)]
        for state, state_records in by_state.items()
    }
    eligible_states = [
        state
        for state, state_records in eligible_by_state.items()
        if len({record["question_group"] for record in state_records}) >= fields_per_case
    ]
    eligible_states.sort(key=lambda state: (_stable_order(seed, state), state))

    cases: list[dict[str, Any]] = []
    previously_used_groups: dict[str, set[int]] = defaultdict(set)
    max_rounds = max(
        (
            len({record["question_group"] for record in records_for_state})
            // fields_per_case
            for records_for_state in eligible_by_state.values()
        ),
        default=0,
    )
    for round_index in range(max_rounds):
        cases_before_round = len(cases)
        for state in eligible_states:
            ranked = sorted(
                eligible_by_state[state],
                key=lambda record: (
                    _stable_order(
                        seed,
                        f"target|{round_index}|{state}|{record['question_group']}|{record['idx']}",
                    ),
                    record["idx"],
                ),
            )
            targets: list[dict[str, Any]] = []
            used_groups: set[int] = set()
            used_statutes: set[int] = set()
            for record in ranked:
                if (
                    record["question_group"] in previously_used_groups[state]
                    or record["question_group"] in used_groups
                    or record["statute_idx"] in used_statutes
                ):
                    continue
                targets.append(record)
                used_groups.add(record["question_group"])
                used_statutes.add(record["statute_idx"])
                if len(targets) == fields_per_case:
                    break
            if len(targets) != fields_per_case:
                continue

            case_id = f"housing-{len(cases) + 1:03d}"
            candidates: dict[str, dict[str, Any]] = {}
            fields = []
            field_evidence_map: dict[str, list[str]] = {}
            expected_answers: dict[str, str] = {}
            for index, record in enumerate(targets, start=1):
                field_id = f"field-{index:02d}"
                gold_id = f"housing-statute-{record['statute_idx']}"
                fields.append(
                    {
                        "field_id": field_id,
                        "question": record["question"],
                        "source_question_idx": record["idx"],
                        "source_question_group": record["question_group"],
                    }
                )
                field_evidence_map[field_id] = [gold_id]
                expected_answers[field_id] = record["answer"]
                _merge_candidate(candidates, record, "gold_field_evidence")

                negatives = sorted(
                    opposite_candidates(record),
                    key=lambda other: (
                        _stable_order(
                            seed,
                            f"wrong-state|{case_id}|{record['question_group']}|"
                            f"{other['state']}|{other['idx']}",
                        ),
                        other["idx"],
                    ),
                )
                negative = next(
                    (item for item in negatives if item["statute_idx"] not in used_statutes),
                    None,
                )
                if negative is None:
                    break
                used_statutes.add(negative["statute_idx"])
                _merge_candidate(
                    candidates,
                    negative,
                    "wrong_jurisdiction_same_field_opposite_answer",
                )
            else:
                composite_query = " ".join(field["question"] for field in fields)
                distractors = [
                    record
                    for record in by_state[state]
                    if record["question_group"] not in used_groups
                    and record["statute_idx"] not in used_statutes
                ]
                distractors.sort(
                    key=lambda record: (
                        -_lexical_overlap(
                            composite_query, f"{record['citation']} {record['excerpt']}"
                        ),
                        _stable_order(seed, f"same-state|{case_id}|{record['idx']}"),
                        record["idx"],
                    )
                )
                selected_distractors = 0
                for record in distractors:
                    if record["statute_idx"] in used_statutes:
                        continue
                    used_statutes.add(record["statute_idx"])
                    _merge_candidate(candidates, record, "same_jurisdiction_other_field")
                    selected_distractors += 1
                    if selected_distractors == same_jurisdiction_distractors:
                        break
                if selected_distractors < same_jurisdiction_distractors:
                    continue

                query_lines = [
                    f"Consider statutory housing law for {state} in the year "
                    f"{HOUSING_SNAPSHOT_YEAR}.",
                    "Retrieve evidence that answers every field:",
                    *[f"{field['field_id']}: {field['question']}" for field in fields],
                ]
                cases.append(
                    {
                        "id": case_id,
                        "dataset": "housing_qa",
                        "jurisdiction": state,
                        "as_of_year": HOUSING_SNAPSHOT_YEAR,
                        "question": "\n".join(query_lines),
                        "fields": fields,
                        "field_evidence_map": field_evidence_map,
                        "expected_answers": expected_answers,
                        "gold_evidence_ids": sorted(
                            {item for values in field_evidence_map.values() for item in values}
                        ),
                        "required_roles": list(HOUSING_ROLES),
                        "candidates": sorted(candidates.values(), key=lambda item: item["id"]),
                    }
                )
                previously_used_groups[state].update(used_groups)
                if len(cases) == case_count:
                    break
        if len(cases) == case_count or len(cases) == cases_before_round:
            break
    if len(cases) != case_count:
        raise ValueError(
            f"HousingQA source can only build {len(cases)} of {case_count} requested cases "
            "under the frozen identifiability constraints"
        )
    return cases


class HousingRealScorer:
    def __init__(
        self,
        *,
        embedding_model: str,
        reranker_model: str,
        device: str,
        hf_home: Path,
        embedding_batch_size: int = 32,
        rerank_batch_size: int = 16,
        field_relevance_mix: float = 0.15,
        role_relevance_mix: float = 0.15,
    ) -> None:
        import os

        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedding_batch_size = embedding_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.field_relevance_mix = field_relevance_mix
        self.role_relevance_mix = role_relevance_mix
        self.embedder = SentenceTransformer(
            str(local_model_snapshot(hf_home, embedding_model)), device=device
        )
        self.reranker = CrossEncoder(
            str(local_model_snapshot(hf_home, reranker_model)), device=device
        )

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        return np.asarray(
            self.embedder.encode(
                texts,
                batch_size=self.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        import numpy as np

        return np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=self.rerank_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )

    def score_case(self, case: dict[str, Any]) -> dict[str, Any]:
        candidates = case["candidates"]
        texts = [str(candidate["text"]) for candidate in candidates]
        if not texts:
            return case
        question = str(case["question"])
        vectors = self.encode(texts)
        query_vector = self.encode([question])[0]
        bm25 = bm25_scores(question, texts)
        dense = minmax(vectors @ query_vector)
        hybrid = rrf_scores([rank_desc(bm25), rank_desc(dense)], len(texts))
        cross = minmax(self.predict([(question, text) for text in texts]))

        field_values: dict[str, Any] = {}
        for field in case["fields"]:
            field_id = str(field["field_id"])
            field_query = (
                f"Find statutory evidence that directly answers this field for "
                f"{case['jurisdiction']} in {case['as_of_year']}: {field['question']}"
            )
            calibrated = minmax(self.predict([(field_query, text) for text in texts]))
            field_values[field_id] = (
                (1.0 - self.field_relevance_mix) * calibrated
                + self.field_relevance_mix * cross
            )

        role_values: dict[str, Any] = {}
        for role, instruction in HOUSING_ROLES.items():
            role_query = f"{instruction} Task: {question}"
            calibrated = minmax(self.predict([(role_query, text) for text in texts]))
            role_values[role] = (
                (1.0 - self.role_relevance_mix) * calibrated
                + self.role_relevance_mix * cross
            )

        scored = []
        for index, candidate in enumerate(candidates):
            scored.append(
                {
                    **candidate,
                    "scores": {
                        "bm25": float(bm25[index]),
                        "dense": float(dense[index]),
                        "hybrid": float(hybrid[index]),
                        "cross_encoder": float(cross[index]),
                    },
                    "field_scores": {
                        field_id: float(values[index])
                        for field_id, values in field_values.items()
                    },
                    "role_scores": {
                        role: float(values[index]) for role, values in role_values.items()
                    },
                }
            )
        return {**case, "candidates": scored}

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score every candidate/query pair in three GPU batches.

        Per-case calibration remains identical to :meth:`score_case`; batching only removes
        repeated model-dispatch overhead and does not alter the frozen experiment protocol.
        """

        unique_texts = list(
            dict.fromkeys(
                str(candidate["text"])
                for case in cases
                for candidate in case.get("candidates", [])
            )
        )
        text_vectors = self.encode(unique_texts)
        vector_by_text = {
            text: text_vectors[index] for index, text in enumerate(unique_texts)
        }
        query_vectors = self.encode([str(case["question"]) for case in cases])
        print(
            f"encoded {len(unique_texts)} unique statutes and {len(cases)} composite queries",
            flush=True,
        )

        base_pairs: list[tuple[str, str]] = []
        base_spans: list[tuple[int, int]] = []
        for case in cases:
            start = len(base_pairs)
            base_pairs.extend(
                (str(case["question"]), str(candidate["text"]))
                for candidate in case["candidates"]
            )
            base_spans.append((start, len(base_pairs)))
        base_raw = self.predict(base_pairs)
        print(f"reranked {len(base_pairs)} composite-query pairs", flush=True)

        field_pairs: list[tuple[str, str]] = []
        field_spans: dict[tuple[int, str], tuple[int, int]] = {}
        for case_index, case in enumerate(cases):
            for field in case["fields"]:
                field_id = str(field["field_id"])
                field_query = (
                    f"Find statutory evidence that directly answers this field for "
                    f"{case['jurisdiction']} in {case['as_of_year']}: {field['question']}"
                )
                start = len(field_pairs)
                field_pairs.extend(
                    (field_query, str(candidate["text"]))
                    for candidate in case["candidates"]
                )
                field_spans[(case_index, field_id)] = (start, len(field_pairs))
        field_raw = self.predict(field_pairs)
        print(f"reranked {len(field_pairs)} task-field pairs", flush=True)

        role_pairs: list[tuple[str, str]] = []
        role_spans: dict[tuple[int, str], tuple[int, int]] = {}
        for case_index, case in enumerate(cases):
            for role, instruction in HOUSING_ROLES.items():
                role_query = f"{instruction} Task: {case['question']}"
                start = len(role_pairs)
                role_pairs.extend(
                    (role_query, str(candidate["text"]))
                    for candidate in case["candidates"]
                )
                role_spans[(case_index, role)] = (start, len(role_pairs))
        role_raw = self.predict(role_pairs)
        print(f"reranked {len(role_pairs)} functional-role pairs", flush=True)

        output = []
        for case_index, case in enumerate(cases):
            candidates = case["candidates"]
            texts = [str(candidate["text"]) for candidate in candidates]
            passage_vectors = [vector_by_text[text] for text in texts]
            import numpy as np

            passage_matrix = np.asarray(passage_vectors, dtype=np.float32)
            bm25 = bm25_scores(str(case["question"]), texts)
            dense = minmax(passage_matrix @ query_vectors[case_index])
            hybrid = rrf_scores([rank_desc(bm25), rank_desc(dense)], len(texts))
            base_start, base_end = base_spans[case_index]
            cross = minmax(base_raw[base_start:base_end])
            field_values = {}
            for field in case["fields"]:
                field_id = str(field["field_id"])
                start, end = field_spans[(case_index, field_id)]
                calibrated = minmax(field_raw[start:end])
                field_values[field_id] = (
                    (1.0 - self.field_relevance_mix) * calibrated
                    + self.field_relevance_mix * cross
                )
            role_values = {}
            for role in HOUSING_ROLES:
                start, end = role_spans[(case_index, role)]
                calibrated = minmax(role_raw[start:end])
                role_values[role] = (
                    (1.0 - self.role_relevance_mix) * calibrated
                    + self.role_relevance_mix * cross
                )
            scored_candidates = []
            for candidate_index, candidate in enumerate(candidates):
                scored_candidates.append(
                    {
                        **candidate,
                        "scores": {
                            "bm25": float(bm25[candidate_index]),
                            "dense": float(dense[candidate_index]),
                            "hybrid": float(hybrid[candidate_index]),
                            "cross_encoder": float(cross[candidate_index]),
                        },
                        "field_scores": {
                            field_id: float(values[candidate_index])
                            for field_id, values in field_values.items()
                        },
                        "role_scores": {
                            role: float(values[candidate_index])
                            for role, values in role_values.items()
                        },
                    }
                )
            output.append({**case, "candidates": scored_candidates})
        return output


def score_housing_cases(
    cases: list[dict[str, Any]],
    *,
    output_path: Path,
    metadata_path: Path,
    source_hash: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    case_signature = stable_hash(
        [
            {
                "id": case["id"],
                "jurisdiction": case["jurisdiction"],
                "fields": case["fields"],
                "candidate_ids": [candidate["id"] for candidate in case["candidates"]],
            }
            for case in cases
        ]
    )
    expected = {
        "source_sha256": source_hash,
        "case_signature": case_signature,
        "config": config,
        "cases": len(cases),
    }
    if output_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == expected:
            cached = list(read_jsonl(output_path))
            if len(cached) == len(cases):
                return cached
    scorer = HousingRealScorer(
        embedding_model=config["embedding_model"],
        reranker_model=config["reranker_model"],
        device=config["device"],
        hf_home=Path(config["hf_home"]),
        embedding_batch_size=int(config["embedding_batch_size"]),
        rerank_batch_size=int(config["rerank_batch_size"]),
        field_relevance_mix=float(config["field_relevance_mix"]),
        role_relevance_mix=float(config["role_relevance_mix"]),
    )
    scored = scorer.score_cases(cases)
    write_jsonl(output_path, scored)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return scored


def _is_applicable(case: dict[str, Any], candidate: dict[str, Any]) -> bool:
    metadata = candidate.get("metadata", {})
    return (
        metadata.get("jurisdiction") == case.get("jurisdiction")
        and int(metadata.get("snapshot_year", -1)) == int(case.get("as_of_year", -2))
    )


def _take_with_budget(
    candidates: Iterable[dict[str, Any]], *, top_k: int, token_budget: int
) -> list[dict[str, Any]]:
    selected = []
    cost = 0
    for candidate in candidates:
        candidate_cost = int(candidate.get("token_count", 1))
        if cost + candidate_cost > token_budget:
            continue
        selected.append(candidate)
        cost += candidate_cost
        if len(selected) == top_k:
            break
    return selected


def _coverage_gain(old: float, score: float, threshold: float) -> float:
    if old < threshold <= max(old, score):
        return 1.0
    return max(0.0, score - old) * 0.25


def _coverage_select(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
    field_weight: float,
    role_weight: float,
    field_threshold: float,
    role_threshold: float,
) -> list[dict[str, Any]]:
    fields = [str(field["field_id"]) for field in case.get("fields", [])]
    roles = [str(role) for role in case.get("required_roles", [])]
    field_best = {field: 0.0 for field in fields}
    role_best = {role: 0.0 for role in roles}
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    cost = 0
    while remaining and len(selected) < top_k:
        scored = []
        for candidate in remaining:
            candidate_cost = int(candidate.get("token_count", 1))
            if cost + candidate_cost > token_budget:
                continue
            field_gain = sum(
                _coverage_gain(
                    field_best[field],
                    float(candidate.get("field_scores", {}).get(field, 0.0)),
                    field_threshold,
                )
                for field in fields
            ) / max(1, len(fields))
            role_gain = sum(
                _coverage_gain(
                    role_best[role],
                    float(candidate.get("role_scores", {}).get(role, 0.0)),
                    role_threshold,
                )
                for role in roles
            ) / max(1, len(roles))
            relevance = float(candidate.get("scores", {}).get("cross_encoder", 0.0))
            objective = field_weight * field_gain + role_weight * role_gain + relevance
            scored.append((objective, candidate["id"], candidate))
        if not scored:
            break
        _, _, best = max(scored, key=lambda item: (item[0], item[1]))
        selected.append(best)
        remaining = [candidate for candidate in remaining if candidate["id"] != best["id"]]
        cost += int(best.get("token_count", 1))
        for field in fields:
            field_best[field] = max(
                field_best[field], float(best.get("field_scores", {}).get(field, 0.0))
            )
        for role in roles:
            role_best[role] = max(
                role_best[role], float(best.get("role_scores", {}).get(role, 0.0))
            )
    return selected


def select_housing_evidence(
    case: dict[str, Any],
    method: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    field_threshold: float = DEFAULT_FIELD_THRESHOLD,
    role_threshold: float = DEFAULT_ROLE_THRESHOLD,
    field_weight: float = DEFAULT_FIELD_WEIGHT,
    role_weight: float = DEFAULT_ROLE_WEIGHT,
) -> list[dict[str, Any]]:
    if method not in HOUSING_METHODS:
        raise ValueError(f"unsupported HousingQA selector: {method}")
    apply_filter = method != "w/o_applicability"
    candidates = [
        candidate
        for candidate in case.get("candidates", [])
        if not apply_filter or _is_applicable(case, candidate)
    ]
    score_name = {
        "bm25_topk": "bm25",
        "dense_topk": "dense",
        "hybrid_topk": "hybrid",
        "cross_encoder_topk": "cross_encoder",
    }.get(method)
    if score_name:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.get("scores", {}).get(score_name, 0.0)),
                candidate["id"],
            ),
        )
        return _take_with_budget(ordered, top_k=top_k, token_budget=token_budget)
    return _coverage_select(
        case,
        candidates,
        top_k=top_k,
        token_budget=token_budget,
        field_weight=field_weight if method != "w/o_field" else 0.0,
        role_weight=0.0 if method == "field_decomposition_topk" else role_weight,
        field_threshold=field_threshold,
        role_threshold=role_threshold,
    )


def select_housing_methods(
    cases: list[dict[str, Any]],
    *,
    methods: Iterable[str] = HOUSING_METHODS,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    field_threshold: float = DEFAULT_FIELD_THRESHOLD,
    role_threshold: float = DEFAULT_ROLE_THRESHOLD,
    field_weight: float = DEFAULT_FIELD_WEIGHT,
    role_weight: float = DEFAULT_ROLE_WEIGHT,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for method in methods:
            selected = select_housing_evidence(
                case,
                method,
                top_k=top_k,
                token_budget=token_budget,
                field_threshold=field_threshold,
                role_threshold=role_threshold,
                field_weight=field_weight,
                role_weight=role_weight,
            )
            rows.append(
                {
                    "case_id": case["id"],
                    "method": method,
                    "jurisdiction": case["jurisdiction"],
                    "as_of_year": case["as_of_year"],
                    "fields": case["fields"],
                    "field_evidence_map": case["field_evidence_map"],
                    "expected_answers": case["expected_answers"],
                    "gold_evidence_ids": case["gold_evidence_ids"],
                    "candidate_count": len(case["candidates"]),
                    "selected_ids": [candidate["id"] for candidate in selected],
                    "selected_evidence": selected,
                }
            )
    return rows


def housing_selection_signature(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["case_id"]), tuple(str(value) for value in row["selected_ids"])


def housing_answer_prompt(row: dict[str, Any]) -> str:
    fields = "\n".join(
        f"- {field['field_id']}: {field['question']}" for field in row.get("fields", [])
    )
    evidence = "\n\n".join(
        f"[{index}] {item.get('citation', '')}\n{item.get('text', '')}"
        for index, item in enumerate(row.get("selected_evidence", []), start=1)
    )
    return (
        f"Consider statutory housing law for {row['jurisdiction']} in {row['as_of_year']}.\n"
        "Answer every field using only the supplied statute excerpts.\n"
        "For each field return exactly Yes, No, or Unknown. Use Unknown when the excerpts are "
        "insufficient. Return one JSON object only, mapping field_id to the label.\n\n"
        f"Fields:\n{fields}\n\nStatute excerpts:\n{evidence}\n\nJSON:"
    )


def parse_housing_answers(text: str, field_ids: Iterable[str]) -> dict[str, str]:
    expected = [str(field_id) for field_id in field_ids]
    stripped = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    payload: Any = {}
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}
    if isinstance(payload, dict) and isinstance(payload.get("answers"), dict):
        payload = payload["answers"]
    if not isinstance(payload, dict):
        payload = {}
    return {
        field_id: _normalized_answer(payload.get(field_id)) or "Invalid" for field_id in expected
    }


class LocalHousingAnswerer:
    def __init__(self, *, model_path: Path, batch_size: int = 16, max_new_tokens: int = 160) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def _chat_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def answer(
        self, rows: list[dict[str, Any]], *, output_path: Path, cache_key: str
    ) -> list[dict[str, Any]]:
        completed: dict[tuple[str, str], dict[str, Any]] = {}
        if output_path.is_file():
            for row in read_jsonl(output_path):
                if row.get("cache_key") == cache_key:
                    completed[(str(row["case_id"]), str(row["method"]))] = row

        signature_predictions: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for source in rows:
            prediction = completed.get((source["case_id"], source["method"]))
            if prediction is not None:
                signature_predictions.setdefault(housing_selection_signature(source), prediction)

        pending_by_signature: dict[
            tuple[str, tuple[str, ...]], list[dict[str, Any]]
        ] = defaultdict(list)
        for source in rows:
            key = (source["case_id"], source["method"])
            if key in completed:
                continue
            signature = housing_selection_signature(source)
            equivalent = signature_predictions.get(signature)
            if equivalent is not None:
                completed[key] = {
                    **equivalent,
                    "method": source["method"],
                    "reused_from_equivalent_selection": True,
                }
            else:
                pending_by_signature[signature].append(source)

        representatives = [values[0] for values in pending_by_signature.values()]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(representatives), self.batch_size):
            batch = representatives[start : start + self.batch_size]
            prompts = [self._chat_prompt(housing_answer_prompt(row)) for row in batch]
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
            for source, raw_response in zip(batch, decoded, strict=True):
                prediction = {
                    "case_id": source["case_id"],
                    "method": source["method"],
                    "selected_ids": source["selected_ids"],
                    "answers": parse_housing_answers(
                        raw_response, [field["field_id"] for field in source["fields"]]
                    ),
                    "raw_response": raw_response,
                    "cache_key": cache_key,
                    "reused_from_equivalent_selection": False,
                }
                signature = housing_selection_signature(source)
                for equivalent_source in pending_by_signature[signature]:
                    equivalent_key = (
                        equivalent_source["case_id"],
                        equivalent_source["method"],
                    )
                    completed[equivalent_key] = {
                        **prediction,
                        "method": equivalent_source["method"],
                        "reused_from_equivalent_selection": equivalent_source is not source,
                    }
            print(
                f"generated {min(start + len(batch), len(representatives))}/"
                f"{len(representatives)} unique HousingQA selections",
                flush=True,
            )
            ordered_partial = [
                completed[(row["case_id"], row["method"])]
                for row in rows
                if (row["case_id"], row["method"]) in completed
            ]
            write_jsonl(output_path, ordered_partial)

        output = [completed[(row["case_id"], row["method"])] for row in rows]
        write_jsonl(output_path, output)
        return output


def _score_housing_row(
    row: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, Any]:
    selected_ids = set(row["selected_ids"])
    gold_ids = set(row["gold_evidence_ids"])
    true_positive = len(selected_ids & gold_ids)
    recall = true_positive / max(1, len(gold_ids))
    precision = true_positive / max(1, len(selected_ids))
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    field_coverage = sum(
        bool(selected_ids & set(evidence_ids))
        for evidence_ids in row["field_evidence_map"].values()
    ) / max(1, len(row["field_evidence_map"]))
    answers = prediction.get("answers", {})
    answer_accuracy = sum(
        answers.get(field_id) == expected
        for field_id, expected in row["expected_answers"].items()
    ) / max(1, len(row["expected_answers"]))
    valid_answer_rate = sum(
        answers.get(field_id) in {"Yes", "No", "Unknown"}
        for field_id in row["expected_answers"]
    ) / max(1, len(row["expected_answers"]))
    wrong_jurisdiction = sum(
        item.get("metadata", {}).get("jurisdiction") != row["jurisdiction"]
        for item in row["selected_evidence"]
    )
    metrics = {
        "evidence_recall": round(recall, 6),
        "evidence_precision": round(precision, 6),
        "evidence_f1": round(f1, 6),
        "field_coverage": round(field_coverage, 6),
        "core_field_coverage": round(field_coverage, 6),
        "citation_support_precision": round(precision, 6),
        "unsupported_field_rate": round(1.0 - field_coverage, 6),
        "wrong_jurisdiction_rate": round(wrong_jurisdiction / max(1, len(selected_ids)), 6),
        "wrong_jurisdiction_case": float(wrong_jurisdiction > 0),
        "answer_accuracy": round(answer_accuracy, 6),
        "valid_answer_rate": round(valid_answer_rate, 6),
        "case_accuracy": float(answer_accuracy == 1.0 and field_coverage == 1.0),
        "token_cost": sum(int(item.get("token_count", 1)) for item in row["selected_evidence"]),
    }
    return {
        "case_id": row["case_id"],
        "method": row["method"],
        "jurisdiction": row["jurisdiction"],
        "selected_ids": row["selected_ids"],
        "gold_evidence_ids": row["gold_evidence_ids"],
        "answers": answers,
        "expected_answers": row["expected_answers"],
        "metrics": metrics,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    metric_names = tuple(rows[0]["metrics"]) if rows else ()
    return {
        "cases": len(rows),
        **{
            name: round(sum(float(row["metrics"][name]) for row in rows) / len(rows), 6)
            for name in metric_names
        },
    }


def _paired_comparison(
    by_method: dict[str, list[dict[str, Any]]], left: str, right: str
) -> dict[str, Any]:
    left_rows = {row["case_id"]: row for row in by_method[left]}
    right_rows = {row["case_id"]: row for row in by_method[right]}
    if set(left_rows) != set(right_rows):
        raise ValueError(f"paired HousingQA cases differ for {left} and {right}")
    metrics = (
        "evidence_f1",
        "field_coverage",
        "citation_support_precision",
        "answer_accuracy",
        "case_accuracy",
        "wrong_jurisdiction_rate",
    )
    return {
        "left": left,
        "right": right,
        "direction": "left_minus_right",
        "metrics": {
            metric: paired_bootstrap(
                [
                    float(left_rows[case_id]["metrics"][metric])
                    - float(right_rows[case_id]["metrics"][metric])
                    for case_id in sorted(left_rows)
                ],
                seed=DEFAULT_SEED,
            )
            for metric in metrics
        },
    }


def build_housing_ablation_report(
    *,
    cases: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
    source_path: Path,
    source_zip_path: Path | None,
    scored_path: Path,
) -> dict[str, Any]:
    prediction_map = {
        (str(row["case_id"]), str(row["method"])): row for row in predictions
    }
    if len(prediction_map) != len(selected_rows):
        raise ValueError("HousingQA prediction coverage is incomplete")
    results = [
        _score_housing_row(row, prediction_map[(row["case_id"], row["method"])])
        for row in selected_rows
    ]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_method[row["method"]].append(row)
    if set(by_method) != set(HOUSING_METHODS):
        raise ValueError("HousingQA report must cover every frozen method")
    aggregate = {method: _aggregate(by_method[method]) for method in HOUSING_METHODS}
    baseline_methods = (
        "bm25_topk",
        "dense_topk",
        "hybrid_topk",
        "cross_encoder_topk",
        "field_decomposition_topk",
    )
    strongest_baseline = max(
        baseline_methods,
        key=lambda method: (
            float(aggregate[method]["field_coverage"]),
            float(aggregate[method]["answer_accuracy"]),
            method,
        ),
    )
    comparisons = {
        "full_minus_w_o_field": _paired_comparison(by_method, "frc_full", "w/o_field"),
        "full_minus_w_o_applicability": _paired_comparison(
            by_method, "frc_full", "w/o_applicability"
        ),
        "full_minus_strongest_baseline": _paired_comparison(
            by_method, "frc_full", strongest_baseline
        ),
    }
    selection_changes = {
        variant: sum(
            left["selected_ids"] != right["selected_ids"]
            for left, right in zip(
                sorted(by_method["frc_full"], key=lambda row: row["case_id"]),
                sorted(by_method[variant], key=lambda row: row["case_id"]),
                strict=True,
            )
        )
        for variant in ("w/o_field", "w/o_applicability")
    }
    field_difference = comparisons["full_minus_w_o_field"]["metrics"]["field_coverage"]
    baseline_difference = comparisons["full_minus_strongest_baseline"]["metrics"][
        "field_coverage"
    ]
    return {
        "schema_version": HOUSING_ABLATION_SCHEMA,
        "dataset": "reglab/housing_qa/questions",
        "metadata": {
            "public_dataset": True,
            "expert_annotated_supporting_statutes": True,
            "source_repository": HOUSING_SOURCE_REPOSITORY,
            "source_revision": HOUSING_SOURCE_REVISION,
            "source_url": HOUSING_SOURCE_URL,
            "source_license": HOUSING_SOURCE_LICENSE,
            "source_json_sha256": sha256(source_path),
            "source_zip_sha256": sha256(source_zip_path) if source_zip_path else None,
            "scored_cases_sha256": sha256(scored_path),
            "case_count": len(cases),
            "field_count": sum(len(case["fields"]) for case in cases),
            "jurisdiction_count": len({case["jurisdiction"] for case in cases}),
            "snapshot_year": HOUSING_SNAPSHOT_YEAR,
            "candidate_occurrences": sum(len(case["candidates"]) for case in cases),
            "methods": list(HOUSING_METHODS),
            "models": {
                "embedding": config["embedding_model"],
                "reranker": config["reranker_model"],
                "generator": config["generator_model"],
            },
            "real_model_scores": True,
            "generator": "local_qwen",
            "selection_parameters": {
                "top_k": config["top_k"],
                "token_budget": config["token_budget"],
                "field_threshold": config["field_threshold"],
                "role_threshold": config["role_threshold"],
                "field_weight": config["field_weight"],
                "role_weight": config["role_weight"],
            },
            "frozen_protocol": (
                "deterministically selected jurisdiction-rounds form composite cases; four "
                "original expert questions become task fields; each field has one annotated "
                "statute; same-field opposite-answer statutes from other jurisdictions are hard "
                "negatives"
            ),
            "gold_usage": (
                "expected answers and field_evidence_map are used only after selection and "
                "generation for scoring; neither is included in scorer queries or generator prompts"
            ),
        },
        "coverage": {
            "w/o_field": "RUN_PUBLIC_EXPERT_REAL_MODEL",
            "w/o_applicability": "RUN_PUBLIC_EXPERT_REAL_MODEL_JURISDICTION_2021",
            "jurisdiction": "IDENTIFIABLE",
            "snapshot_year": "FROZEN_2021",
            "version_or_expiry_variation": "NOT_IDENTIFIABLE_SINGLE_SNAPSHOT",
        },
        "aggregates": aggregate,
        "strongest_baseline_by_field_coverage": strongest_baseline,
        "paired_comparisons": comparisons,
        "selection_changed_cases": selection_changes,
        "decision": {
            "full_strictly_better_than_w_o_field": float(field_difference["mean_difference"]) > 0.0,
            "full_w_o_field_ci_excludes_zero": float(field_difference["ci_low"]) > 0.0,
            "full_field_coverage_gain_over_strongest_baseline_at_least_0_05": float(
                baseline_difference["mean_difference"]
            )
            >= 0.05,
            "gate_2": "NO-GO",
            "reason": (
                "This public expert benchmark can test field coverage and jurisdiction filtering, "
                "but a single 2021 snapshot cannot identify version/expiry behavior and the "
                "cross-domain result cannot override the existing failed public comparisons."
            ),
        },
        "limitations": [
            "HousingQA is a public housing-law benchmark, not a flood-response or district-government benchmark.",
            "The public corpus is explicitly accurate as of 2021; it has no paired historical/current statute versions, so version replacement and expiry remain untested.",
            "Composite four-field cases and hard-negative pools are deterministic transformations of expert single-question annotations, not separately expert-reviewed composite tasks.",
            "Public model pretraining contamination cannot be excluded.",
            "The frozen test configuration was not tuned after observing these results.",
        ],
        "case_results": results,
    }


def render_housing_ablation_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# HousingQA real-model field and applicability ablation",
        "",
        f"- Source: `{metadata['source_repository']}@{metadata['source_revision']}` "
        f"({metadata['source_license']})",
        f"- Cases / fields / jurisdictions: {metadata['case_count']} / "
        f"{metadata['field_count']} / {metadata['jurisdiction_count']}",
        f"- Snapshot: {metadata['snapshot_year']}; real-model scores: "
        f"`{metadata['real_model_scores']}`; generator: `{metadata['generator']}`",
        f"- Gold usage: {metadata['gold_usage']}",
        "",
        "| Method | Evidence F1 | Field coverage | Citation precision | Answer accuracy | "
        "Case accuracy | Wrong jurisdiction | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in HOUSING_METHODS:
        row = report["aggregates"][method]
        lines.append(
            f"| {method} | {row['evidence_f1']:.6f} | {row['field_coverage']:.6f} | "
            f"{row['citation_support_precision']:.6f} | {row['answer_accuracy']:.6f} | "
            f"{row['case_accuracy']:.6f} | {row['wrong_jurisdiction_rate']:.6f} | "
            f"{row['token_cost']:.2f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, comparison in report["paired_comparisons"].items():
        field = comparison["metrics"]["field_coverage"]
        answer = comparison["metrics"]["answer_accuracy"]
        lines.append(
            f"- `{name}`: field coverage {field['mean_difference']:+.6f} "
            f"(95% CI [{field['ci_low']:+.6f}, {field['ci_high']:+.6f}]); "
            f"answer accuracy {answer['mean_difference']:+.6f} "
            f"(95% CI [{answer['ci_low']:+.6f}, {answer['ci_high']:+.6f}])."
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"- Gate 2: `{report['decision']['gate_2']}` — {report['decision']['reason']}",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)
