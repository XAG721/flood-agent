"""Prospective CUAD development and confirmation for v53 role closure."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import research.frc_rag.contractnli_native_zero_consensus_abstention as v50
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    _round_for_display,
    select_v49,
)
from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    _bm25,
)
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (
    BUDGETS,
    NON_FRC_BASELINES,
)


SCHEMA_VERSION = "frc-cuad-top3-rank-concurrence-role-closure-v53"
EXPERIMENT_ID = "FRC-CUAD-TOP3-RANK-CONCURRENCE-ROLE-CLOSURE-V53"
DATASET_ID = "cuad_official_v1_top3_role_closure_v53"
CAPABILITY = "parameter_free_top3_role_concurrence_contract_evidence_selection"
PROTOCOL_SHA256 = "9390f8f5642fa7be0de265b59e9aef03d76e303e2fa615bd498a93c5a3e6544c"
SOURCE_ARCHIVE_SHA256 = (
    "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
)

STAGES = ("development", "confirmation")
SOURCE_MEMBERS = {
    "development": "train_separate_questions.json",
    "confirmation": "test.json",
}
SAMPLE_SALTS = {
    "development": "FRC-CUAD-V53-DEVELOPMENT|",
    "confirmation": "FRC-CUAD-V53-CONFIRMATION|",
}
TARGET_CASES = 400
TARGET_PER_GROUP = 200
MAX_CASES_PER_CONTRACT = 4
MINIMUM_CONTRACTS = 90
MINIMUM_STRATUM_CASES = 40
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260813
RETRIEVAL_TOP_K_PER_METHOD = 48
RETRIEVAL_POOL_MAXIMUM = 96
TOP3 = 3

GATED_PREFIX = "top3_rank_concurrence_"
GATED_NON_FRC_METHODS = tuple(
    f"{GATED_PREFIX}{method}_v53" for method in NON_FRC_BASELINES
)
SAME_GATE_FRC_ABLATION = "top3_rank_concurrence_low_core_only_frc_v53"
CANDIDATE = "top3_rank_concurrence_role_closure_low_core_frc_v53"
METHODS = (
    *V49_METHODS,
    *GATED_NON_FRC_METHODS,
    SAME_GATE_FRC_ABLATION,
    CANDIDATE,
)
UNGATED_FRC_CONTROLS = (DYNAMIC_RANK, ADAPTIVE_ARGMAX, LOW_CORE_DIVERGENCE_V49)

QUERY_TEMPLATES = {
    "anchor": "{question}",
    "first_fact": (
        "Find contract language that directly states the following reviewed "
        "clause or term: {question}"
    ),
    "second_fact_or_bridge": (
        "Find exceptions, limitations, conditions, scope, dates, parties, "
        "amounts, or triggers relevant to the following reviewed clause or "
        "term: {question}"
    ),
    "counterevidence": (
        "Find contract language that narrows, contradicts, excludes, or makes "
        "inapplicable the following reviewed clause or term: {question}"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _opaque(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"cuad{digest[:22]}"


def read_source_member(source_archive: Path, stage: str) -> dict[str, Any]:
    """Open exactly one registered CUAD split member."""

    if stage not in STAGES:
        raise ValueError(f"Unsupported CUAD v53 stage: {stage}")
    expected = SOURCE_MEMBERS[stage]
    with zipfile.ZipFile(source_archive) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == expected]
        if len(matches) != 1:
            raise ValueError(f"CUAD v53 archive must contain one {expected}")
        raw = archive.read(matches[0])
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        raise ValueError("CUAD v53 source must be a SQuAD-style object")
    return value


def _iter_cases(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    seen_question_ids: set[str] = set()
    for contract_index, raw_contract in enumerate(source.get("data", [])):
        if not isinstance(raw_contract, dict):
            raise ValueError("CUAD v53 contract must be an object")
        title = _normalise(raw_contract.get("title")) or f"contract-{contract_index}"
        paragraphs = raw_contract.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ValueError("CUAD v53 contract lacks paragraphs")
        for paragraph_index, raw_paragraph in enumerate(paragraphs):
            if not isinstance(raw_paragraph, dict):
                raise ValueError("CUAD v53 paragraph must be an object")
            context = str(raw_paragraph.get("context", ""))
            qas = raw_paragraph.get("qas")
            if not context or not isinstance(qas, list):
                raise ValueError("CUAD v53 paragraph lacks context or qas")
            contract_key = f"{title}|{paragraph_index}"
            for raw_qa in qas:
                if not isinstance(raw_qa, dict):
                    raise ValueError("CUAD v53 question must be an object")
                question_id = str(raw_qa.get("id", ""))
                question = _normalise(raw_qa.get("question"))
                answers = raw_qa.get("answers")
                if not question_id or not question or not isinstance(answers, list):
                    raise ValueError("CUAD v53 question schema is incomplete")
                if question_id in seen_question_ids:
                    raise ValueError("CUAD v53 question ids must be unique")
                seen_question_ids.add(question_id)
                clean_answers: list[dict[str, Any]] = []
                for raw_answer in answers:
                    if not isinstance(raw_answer, dict):
                        raise ValueError("CUAD v53 answer must be an object")
                    text = str(raw_answer.get("text", ""))
                    start = int(raw_answer.get("answer_start", -1))
                    if (
                        not text
                        or start < 0
                        or context[start : start + len(text)] != text
                    ):
                        raise ValueError("CUAD v53 answer offset is invalid")
                    clean_answers.append({"text": text, "answer_start": start})
                impossible = raw_qa.get("is_impossible")
                if impossible is not None and bool(impossible) == bool(clean_answers):
                    raise ValueError("CUAD v53 impossible flag conflicts with answers")
                yield {
                    "contract_key": contract_key,
                    "question_id": question_id,
                    "question": question,
                    "context": context,
                    "answers": clean_answers,
                    "answer_state": "answer_bearing" if clean_answers else "no_answer",
                }


def _sample_key(row: dict[str, Any], stage: str) -> tuple[str, str, str]:
    digest = hashlib.sha256(
        (
            SAMPLE_SALTS[stage]
            + str(row["answer_state"])
            + "|"
            + str(row["contract_key"])
            + "|"
            + str(row["question_id"])
        ).encode()
    ).hexdigest()
    return digest, str(row["contract_key"]), str(row["question_id"])


def select_balanced_sample(
    source: dict[str, Any],
    *,
    stage: str,
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_contract: int = MAX_CASES_PER_CONTRACT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported CUAD v53 stage: {stage}")
    rows = list(_iter_cases(source))
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in rows if row["answer_state"] == group],
            key=lambda row: _sample_key(row, stage),
        )
        for group in groups
    }
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("CUAD v53 source lacks the registered answer-state quota")

    selected: list[dict[str, Any]] = []
    contract_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    cursors = {group: 0 for group in groups}
    while any(group_counts[group] < target_per_group for group in groups):
        progress = False
        for group in groups:
            if group_counts[group] >= target_per_group:
                continue
            values = by_group[group]
            while cursors[group] < len(values):
                row = values[cursors[group]]
                cursors[group] += 1
                contract = str(row["contract_key"])
                if contract_counts[contract] >= maximum_cases_per_contract:
                    continue
                selected.append(row)
                contract_counts[contract] += 1
                group_counts[group] += 1
                progress = True
                break
        if not progress:
            break
    expected = target_per_group * len(groups)
    if len(selected) != expected:
        raise ValueError("CUAD v53 could not form its balanced capped sample")
    selected = sorted(selected, key=lambda row: _sample_key(row, stage))
    return selected, {
        "stage": stage,
        "eligible_cases": len(rows),
        "eligible_answer_state_counts": {
            group: len(by_group[group]) for group in groups
        },
        "selected_cases": len(selected),
        "selected_answer_state_counts": {
            group: group_counts[group] for group in groups
        },
        "selected_contracts": len(contract_counts),
        "maximum_cases_per_contract": max(contract_counts.values(), default=0),
        "target_per_group": target_per_group,
        "contract_cap": maximum_cases_per_contract,
        "sample_salt": SAMPLE_SALTS[stage],
    }


def _word_matches(
    text: str, start: int = 0, end: int | None = None
) -> list[re.Match[str]]:
    stop = len(text) if end is None else end
    return list(re.finditer(r"\S+", text[start:stop]))


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _window_spans(
    text: str,
    *,
    width: int,
    stride: int,
    start: int = 0,
    end: int | None = None,
) -> list[tuple[int, int]]:
    stop = len(text) if end is None else end
    matches = list(re.finditer(r"\S+", text[start:stop]))
    if not matches:
        return []
    result: list[tuple[int, int]] = []
    positions = list(range(0, max(1, len(matches) - width + 1), stride))
    suffix = max(0, len(matches) - width)
    if suffix not in positions:
        positions.append(suffix)
    for index in sorted(set(positions)):
        last = min(len(matches), index + width) - 1
        result.append((start + matches[index].start(), start + matches[last].end()))
    return result


def build_source_candidates(context: str, tokenizer: Any) -> list[dict[str, Any]]:
    proposals: list[tuple[int, int, str]] = []
    source_order = {
        "line": 0,
        "clause": 1,
        "window_24": 2,
        "window_48": 3,
        "window_96": 4,
    }

    def add_span(start: int, end: int, kind: str) -> None:
        trimmed = _trim_span(context, start, end)
        if trimmed is None:
            return
        begin, finish = trimmed
        words = _word_matches(context, begin, finish)
        if len(words) <= 128:
            proposals.append((begin, finish, kind))
            return
        for left, right in _window_spans(
            context, width=96, stride=48, start=begin, end=finish
        ):
            proposals.append((left, right, kind))

    for match in re.finditer(r"[^\r\n]+", context):
        add_span(match.start(), match.end(), "line")
        line = match.group(0)
        offset = match.start()
        for clause in re.finditer(r".+?(?:[.;:?!](?=\s|$)|$)", line):
            add_span(offset + clause.start(), offset + clause.end(), "clause")
    for width, stride in ((24, 12), (48, 24), (96, 48)):
        for start, end in _window_spans(context, width=width, stride=stride):
            add_span(start, end, f"window_{width}")

    by_offsets: dict[tuple[int, int], tuple[int, int, str]] = {}
    for proposal in proposals:
        key = (proposal[0], proposal[1])
        current = by_offsets.get(key)
        if current is None or source_order[proposal[2]] < source_order[current[2]]:
            by_offsets[key] = proposal
    ordered = sorted(
        by_offsets.values(),
        key=lambda item: (item[0], item[1] - item[0], source_order[item[2]]),
    )
    by_text: dict[str, tuple[int, int, str]] = {}
    for proposal in ordered:
        normalised = _normalise(context[proposal[0] : proposal[1]])
        if normalised and normalised not in by_text:
            by_text[normalised] = proposal

    result: list[dict[str, Any]] = []
    for start, end, kind in sorted(
        by_text.values(),
        key=lambda item: (item[0], item[1] - item[0], source_order[item[2]]),
    ):
        text = context[start:end]
        try:
            token_count = max(1, len(tokenizer.encode(text, add_special_tokens=False)))
        except (AttributeError, TypeError):
            token_count = max(1, len(text.split()))
        canonical_id = f"span_{start:08d}_{end:08d}"
        result.append(
            {
                "id": canonical_id,
                "source_id": canonical_id,
                "source_kind": kind,
                "text": text,
                "token_count": token_count,
                "start": start,
                "end": end,
            }
        )
    if not result:
        raise ValueError("CUAD v53 contract produced no source candidates")
    return result


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "mean": 0.0, "maximum": 0}
    return {
        "minimum": min(values),
        "mean": round(float(np.mean(values)), 6),
        "maximum": max(values),
    }


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]], tokenizer: Any, *, stage: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    contract_lengths: list[int] = []
    for row in selected:
        contract_key = str(row["contract_key"])
        question_id = str(row["question_id"])
        case_id = _opaque(stage, contract_key, question_id)
        units = build_source_candidates(str(row["context"]), tokenizer)
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "stage": stage,
            "id": case_id,
            "query": str(row["question"]),
            "candidates": [
                {
                    "id": str(unit["id"]),
                    "source_id": str(unit["source_id"]),
                    "source_kind": str(unit["source_kind"]),
                    "text": str(unit["text"]),
                    "token_count": int(unit["token_count"]),
                }
                for unit in units
            ],
            "gold_fields_visible_to_scorer": False,
        }
        if v50._contains_forbidden_blind_key(blind):
            raise AssertionError("CUAD v53 gold leaked into blind cache")
        prepared.append(blind)
        maps.append(
            {
                "schema_version": "frc-cuad-v53-candidate-map-v1",
                "id": case_id,
                "contract_id_sha256": hashlib.sha256(contract_key.encode()).hexdigest(),
                "question_id_sha256": hashlib.sha256(question_id.encode()).hexdigest(),
                "units": [
                    {
                        "candidate_id": str(unit["id"]),
                        "source_kind": str(unit["source_kind"]),
                        "start": int(unit["start"]),
                        "end": int(unit["end"]),
                        "token_count": int(unit["token_count"]),
                        "text_sha256": hashlib.sha256(
                            str(unit["text"]).encode()
                        ).hexdigest(),
                    }
                    for unit in units
                ],
            }
        )
        pool_sizes.append(len(units))
        token_costs.extend(int(unit["token_count"]) for unit in units)
        contract_lengths.append(len(str(row["context"])))
    return (
        prepared,
        maps,
        {
            "stage": stage,
            "prepared_cases": len(prepared),
            "source_candidate_pool_size": _distribution(pool_sizes),
            "source_candidate_token_cost": _distribution(token_costs),
            "contract_length_codepoints": _distribution(contract_lengths),
            "contract_length_quartile_boundaries": [
                round(float(value), 6)
                for value in np.quantile(contract_lengths, [0.25, 0.5, 0.75])
            ],
            "gold_fields_exported_to_blind_cache": False,
            "ready_for_query_generation": len(prepared) == len(selected),
        },
    )


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in prepared_rows:
        if v50._contains_forbidden_blind_key(row):
            raise ValueError("CUAD v53 query builder received a forbidden field")
        question = _normalise(row["query"])
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "stage": str(row["stage"]),
                "id": str(row["id"]),
                "atomic_queries": {
                    role: QUERY_TEMPLATES[role].format(question=question)
                    for role in DYNAMIC_ROLES
                },
                "fallback_used": False,
                "fallback_reason": None,
                "gold_fields_visible_to_generator": False,
            }
        )
    return result


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if [str(row["id"]) for row in prepared_rows] != [
        str(row["id"]) for row in query_rows
    ]:
        raise ValueError("CUAD v53 query cache is incomplete or out of order")
    expected = build_deterministic_queries(prepared_rows)
    if list(query_rows) != expected:
        raise ValueError("CUAD v53 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


def _ranking(
    values: Sequence[float], candidates: Sequence[dict[str, Any]]
) -> list[str]:
    return [
        str(candidates[index]["id"])
        for index in sorted(
            range(len(candidates)),
            key=lambda index: (-float(values[index]), str(candidates[index]["id"])),
        )
    ]


def build_retrieval_pool(
    candidates: Sequence[dict[str, Any]],
    *,
    query: str,
    dense_scores: Sequence[float],
) -> list[dict[str, Any]]:
    values = [dict(candidate) for candidate in candidates]
    if len(values) != len(dense_scores) or not values:
        raise ValueError("CUAD v53 retrieval inputs are incomplete")
    bm25_scores = _bm25(query, [str(candidate["text"]) for candidate in values])
    bm25_order = _ranking(bm25_scores, values)
    dense_order = _ranking(dense_scores, values)
    limit = min(RETRIEVAL_TOP_K_PER_METHOD, len(values))
    selected_ids = set(bm25_order[:limit]) | set(dense_order[:limit])
    missing_rank = len(values) + 1
    bm25_rank = {identifier: index for index, identifier in enumerate(bm25_order)}
    dense_rank = {identifier: index for index, identifier in enumerate(dense_order)}
    by_id = {str(candidate["id"]): candidate for candidate in values}
    ordered_ids = sorted(
        selected_ids,
        key=lambda identifier: (
            min(
                bm25_rank.get(identifier, missing_rank),
                dense_rank.get(identifier, missing_rank),
            ),
            bm25_rank.get(identifier, missing_rank)
            + dense_rank.get(identifier, missing_rank),
            identifier,
        ),
    )
    if len(ordered_ids) > RETRIEVAL_POOL_MAXIMUM:
        raise AssertionError("CUAD v53 retrieval pool exceeded its frozen maximum")
    return [by_id[identifier] for identifier in ordered_ids]


class FrozenCuadScorer(v50.FrozenContractNliScorer):
    """Frozen BGE scorer with a gold-free BM25+dense retrieval pool."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._source_embedding_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _source_signature(candidates: Sequence[dict[str, Any]]) -> str:
        commitment = [
            {
                "id": str(candidate["id"]),
                "text_sha256": hashlib.sha256(
                    str(candidate["text"]).encode()
                ).hexdigest(),
            }
            for candidate in candidates
        ]
        payload = json.dumps(
            commitment,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reduced: list[dict[str, Any]] = []
        retrieval_meta: dict[str, dict[str, int]] = {}
        for case in cases:
            candidates = [dict(value) for value in case["candidates"]]
            texts = [str(value["text"]) for value in candidates]
            signature = self._source_signature(candidates)
            contexts = self._source_embedding_cache.get(signature)
            if contexts is None:
                contexts = self._encode(texts)
                self._source_embedding_cache[signature] = contexts
            query_vector = self._encode([str(case["query"])])[0]
            dense = contexts @ query_vector
            pool = build_retrieval_pool(
                candidates, query=str(case["query"]), dense_scores=dense
            )
            clone = dict(case)
            clone["candidates"] = pool
            reduced.append(clone)
            retrieval_meta[str(case["id"])] = {
                "source_candidate_count": len(candidates),
                "retrieval_pool_count": len(pool),
            }
        rows = super().score_cases(reduced)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
            row["retrieval_pool"] = retrieval_meta[str(row["id"])]
        return rows


def _quartile(value: int, boundaries: Sequence[float]) -> str:
    if value <= float(boundaries[0]):
        return "q1"
    if value <= float(boundaries[1]):
        return "q2"
    if value <= float(boundaries[2]):
        return "q3"
    return "q4"


def _official_words(value: str) -> set[str]:
    text = value
    for token in (".", ",", ";", ":"):
        text = text.replace(token, "")
    text = text.lower().replace("/", " ")
    return {word for word in text.split(" ") if word}


def official_answer_match(answer: str, prediction: str, *, parties: bool) -> bool:
    answer_words = _official_words(answer)
    prediction_words = _official_words(prediction)
    union = answer_words | prediction_words
    jaccard = len(answer_words & prediction_words) / len(union) if union else 0.0
    return jaccard >= 0.5 or (parties and answer in prediction)


def _answer_count_group(count: int) -> str:
    if count == 0:
        return "none"
    if count == 1:
        return "one"
    return "multiple"


def _category(question_id: str, question: str) -> str:
    if "__" in question_id:
        value = _normalise(question_id.rsplit("__", 1)[-1])
        if value:
            return value
    return f"question-{hashlib.sha256(question.encode()).hexdigest()[:12]}"


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    contract_length_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    source_by_question: dict[str, dict[str, Any]] = {}
    for row in _iter_cases(source):
        commitment = hashlib.sha256(str(row["question_id"]).encode()).hexdigest()
        if commitment in source_by_question:
            raise ValueError("CUAD v53 question commitment is not unique")
        source_by_question[commitment] = row
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    pool_sizes = [len(row.get("candidates", [])) for row in scored_rows]
    pool_boundaries = [
        round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])
    ]
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        source_row = source_by_question.get(str(candidate_map["question_id_sha256"]))
        scored = scored_by_id.get(str(candidate_map["id"]))
        if source_row is None or scored is None:
            raise ValueError("CUAD v53 gold join is incomplete")
        context = str(source_row["context"])
        unit_map = {str(unit["candidate_id"]): unit for unit in candidate_map["units"]}
        pool_ids = [str(item["id"]) for item in scored.get("candidates", [])]
        predictions: dict[str, str] = {}
        for identifier in pool_ids:
            unit = unit_map.get(identifier)
            if unit is None:
                raise ValueError("CUAD v53 retrieval candidate lacks an offset map")
            start, end = int(unit["start"]), int(unit["end"])
            text = context[start:end]
            if hashlib.sha256(text.encode()).hexdigest() != unit["text_sha256"]:
                raise ValueError("CUAD v53 candidate text commitment changed")
            predictions[identifier] = text
        answers = list(source_row["answers"])
        parties = "Parties" in str(source_row["question_id"])
        answer_groups: list[list[str]] = []
        seen_answers: set[str] = set()
        for answer in answers:
            answer_text = str(answer["text"])
            key = _normalise(answer_text).lower()
            if key in seen_answers:
                continue
            seen_answers.add(key)
            answer_groups.append(
                sorted(
                    identifier
                    for identifier, prediction in predictions.items()
                    if official_answer_match(answer_text, prediction, parties=parties)
                )
            )
        pool_size = len(pool_ids)
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "contract_cluster": str(candidate_map["contract_id_sha256"]),
                "answer_state": str(source_row["answer_state"]),
                "answer_groups": answer_groups,
                "answer_count": len(answer_groups),
                "answer_count_group": _answer_count_group(len(answer_groups)),
                "category": _category(
                    str(source_row["question_id"]), str(source_row["question"])
                ),
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _quartile(pool_size, pool_boundaries),
                "contract_length_quartile": _quartile(
                    len(context), contract_length_quartile_boundaries
                ),
                "candidate_ceiling_complete": (not answer_groups or all(answer_groups)),
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    answer_rows = [row for row in gold_rows if row["answer_state"] == "answer_bearing"]
    no_answer_rows = [row for row in gold_rows if row["answer_state"] == "no_answer"]
    ceiling = float(np.mean([row["candidate_ceiling_complete"] for row in answer_rows]))
    contracts = len({str(row["contract_cluster"]) for row in gold_rows})
    checks = {
        "exact_target_cases_met": len(gold_rows) == TARGET_CASES,
        "exact_answer_state_balance_met": len(answer_rows) == TARGET_PER_GROUP
        and len(no_answer_rows) == TARGET_PER_GROUP,
        "minimum_contracts_met": contracts >= MINIMUM_CONTRACTS,
        "contract_cap_met": int(
            sampling.get("maximum_cases_per_contract", MAX_CASES_PER_CONTRACT + 1)
        )
        <= MAX_CASES_PER_CONTRACT,
        "candidate_ceiling_complete_rate_at_least_0_9": ceiling >= 0.9,
    }
    return {
        "schema_version": "frc-cuad-v53-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": sampling.get("stage"),
        "cases": len(gold_rows),
        "contracts": contracts,
        "answer_bearing_cases": len(answer_rows),
        "no_answer_cases": len(no_answer_rows),
        "candidate_ceiling_complete_rate_on_answer_bearing_cases": round(ceiling, 6),
        "candidate_pool_quartiles": dict(
            Counter(row["candidate_pool_quartile"] for row in gold_rows)
        ),
        "contract_length_quartiles": dict(
            Counter(row["contract_length_quartile"] for row in gold_rows)
        ),
        "answer_count_groups": dict(
            Counter(row["answer_count_group"] for row in gold_rows)
        ),
        "sampling": sampling,
        "checks": checks,
        "development_open_checks_passed": all(checks.values()),
    }


def _role_order(candidates: Sequence[dict[str, Any]], role: str) -> list[str]:
    result: list[tuple[float, str]] = []
    for candidate in candidates:
        raw = candidate.get("raw_dynamic_role_scores")
        if not isinstance(raw, dict) or role not in raw:
            raise ValueError("CUAD v53 raw role score is missing")
        score = float(raw[role])
        if not math.isfinite(score):
            raise ValueError("CUAD v53 raw role score must be finite")
        result.append((-score, str(candidate["id"])))
    return [identifier for _, identifier in sorted(result)]


def top3_rank_concurrence_details(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rankings = {
        role: _role_order(candidates, role)
        for role in ("anchor", "first_fact", "second_fact_or_bridge")
    }
    anchor = rankings["anchor"][:TOP3]
    first = rankings["first_fact"][:TOP3]
    second = rankings["second_fact_or_bridge"][:TOP3]
    polarity = set(first) | set(second)
    nucleus = set(anchor) & polarity
    missing = len(candidates) + 1
    positions = {
        role: {identifier: index + 1 for index, identifier in enumerate(values)}
        for role, values in rankings.items()
    }
    ordered = sorted(
        nucleus,
        key=lambda identifier: (
            max(
                positions["anchor"].get(identifier, missing),
                min(
                    positions["first_fact"].get(identifier, missing),
                    positions["second_fact_or_bridge"].get(identifier, missing),
                ),
            ),
            positions["anchor"].get(identifier, missing)
            + min(
                positions["first_fact"].get(identifier, missing),
                positions["second_fact_or_bridge"].get(identifier, missing),
            ),
            identifier,
        ),
    )
    return {
        "anchor_frontier_ids": anchor,
        "first_fact_frontier_ids": first,
        "second_fact_or_bridge_frontier_ids": second,
        "concurrence_nucleus_ids": ordered,
        "concurrence_nucleus_size": len(ordered),
        "top3_rank_concurrence_passed": bool(ordered),
    }


def _role_closure_select(
    candidates: Sequence[dict[str, Any]], *, token_budget: int
) -> list[dict[str, Any]]:
    values = list(candidates)
    details = top3_rank_concurrence_details(values)
    if not details["top3_rank_concurrence_passed"]:
        return []
    by_id = {str(candidate["id"]): candidate for candidate in values}
    base = select_v49(values, LOW_CORE_DIVERGENCE_V49, token_budget=token_budget)
    ordered_ids = list(details["concurrence_nucleus_ids"])
    ordered_ids.extend(str(candidate["id"]) for candidate in base)
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_text_sources: set[str] = set()
    total = 0
    for identifier in ordered_ids:
        if identifier in seen_ids:
            continue
        candidate = by_id[identifier]
        source = str(candidate["source_id"])
        if source in seen_text_sources:
            continue
        cost = int(candidate["token_count"])
        if total + cost > token_budget:
            if not result:
                result.append(candidate)
            continue
        result.append(candidate)
        seen_ids.add(identifier)
        seen_text_sources.add(source)
        total += cost
    return result


def select_v53(
    candidates: Sequence[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    values = list(candidates)
    if method in V49_METHODS:
        return select_v49(values, method, token_budget=token_budget)
    gate = top3_rank_concurrence_details(values)
    if not gate["top3_rank_concurrence_passed"]:
        return []
    if method == CANDIDATE:
        return _role_closure_select(values, token_budget=token_budget)
    if method == SAME_GATE_FRC_ABLATION:
        return select_v49(values, LOW_CORE_DIVERGENCE_V49, token_budget=token_budget)
    if method in GATED_NON_FRC_METHODS:
        base = method[len(GATED_PREFIX) : -len("_v53")]
        if base not in NON_FRC_BASELINES:
            raise ValueError("CUAD v53 gated baseline mapping is invalid")
        return select_v49(values, base, token_budget=token_budget)
    raise ValueError(f"Unsupported CUAD v53 method: {method}")


def _selection_metrics(
    selected: Sequence[dict[str, Any]], answer_groups: Sequence[Sequence[str]]
) -> dict[str, Any]:
    selected_ids = [str(candidate["id"]) for candidate in selected]
    cost = sum(int(candidate["token_count"]) for candidate in selected)
    if not answer_groups:
        correct = not selected_ids
        return {
            "precision": None,
            "recall": None,
            "answer_f1": None,
            "utility_f1": 1.0 if correct else 0.0,
            "complete_recall": None,
            "abstained": not selected_ids,
            "matched_answer_count": 0,
            "selected_unit_count": len(selected_ids),
            "selected_token_cost": cost,
        }
    group_sets = [set(group) for group in answer_groups]
    selected_set = set(selected_ids)
    matched_answers = sum(bool(group & selected_set) for group in group_sets)
    gold_union = set().union(*group_sets) if group_sets else set()
    matched_predictions = sum(identifier in gold_union for identifier in selected_ids)
    precision = matched_predictions / len(selected_ids) if selected_ids else 0.0
    recall = matched_answers / len(group_sets)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "answer_f1": f1,
        "utility_f1": f1,
        "complete_recall": matched_answers == len(group_sets),
        "abstained": not selected_ids,
        "matched_answer_count": matched_answers,
        "selected_unit_count": len(selected_ids),
        "selected_token_cost": cost,
    }


def _aggregate(rows: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    metrics = [
        configuration["methods"][method]["metrics"]
        for row in rows
        for configuration in row["configurations"].values()
    ]
    answer = [metric for metric in metrics if metric["answer_f1"] is not None]
    no_answer = [metric for metric in metrics if metric["answer_f1"] is None]
    return {
        "answer_or_abstention_macro_f1": float(
            np.mean([metric["utility_f1"] for metric in metrics])
        ),
        "answer_bearing_macro_f1": float(
            np.mean([metric["answer_f1"] for metric in answer])
        ),
        "answer_macro_precision": float(
            np.mean([metric["precision"] for metric in answer])
        ),
        "answer_macro_recall": float(np.mean([metric["recall"] for metric in answer])),
        "complete_answer_recall": float(
            np.mean([metric["complete_recall"] for metric in answer])
        ),
        "no_answer_abstention_accuracy": float(
            np.mean([metric["abstained"] for metric in no_answer])
        ),
        "abstention_rate": float(np.mean([metric["abstained"] for metric in metrics])),
        "mean_selected_unit_count": float(
            np.mean([metric["selected_unit_count"] for metric in metrics])
        ),
        "mean_selected_token_cost": float(
            np.mean([metric["selected_token_cost"] for metric in metrics])
        ),
    }


def _case_utility(row: dict[str, Any], method: str) -> float:
    return float(
        np.mean(
            [
                configuration["methods"][method]["metrics"]["utility_f1"]
                for configuration in row["configurations"].values()
            ]
        )
    )


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]], candidate: str, baseline: str
) -> dict[str, float | int]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["contract_cluster"])].append(
            _case_utility(row, candidate) - _case_utility(row, baseline)
        )
    clusters = sorted(by_cluster)
    arrays = [np.asarray(by_cluster[cluster], dtype=float) for cluster in clusters]
    point = float(np.mean(np.concatenate(arrays)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        picked = rng.integers(0, len(arrays), size=len(arrays))
        samples[index] = float(
            np.mean(np.concatenate([arrays[item] for item in picked]))
        )
    return {
        "point": round(point, 6),
        "ci_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_high": round(float(np.quantile(samples, 0.975)), 6),
        "clusters": len(clusters),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _strongest(rows: Sequence[dict[str, Any]], methods: Sequence[str]) -> str:
    return sorted(
        methods,
        key=lambda method: (
            -float(_aggregate(rows, method)["answer_or_abstention_macro_f1"]),
            method,
        ),
    )[0]


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = _strongest(rows, GATED_NON_FRC_METHODS)
    candidate_value = float(np.mean([_case_utility(row, CANDIDATE) for row in rows]))
    baseline_value = float(np.mean([_case_utility(row, strongest) for row in rows]))
    return strongest, round(candidate_value - baseline_value, 6)


def evaluate_stage(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stage not in STAGES or len(gold_rows) != len(scored_rows) or not gold_rows:
        raise ValueError("CUAD v53 gold and score caches differ")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("CUAD v53 gold and score ids differ")
        candidates = v50.merge_v50_scored_candidates(dict(scored))
        gate = top3_rank_concurrence_details(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v53(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(selected, gold["answer_groups"]),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "contract_cluster": str(gold["contract_cluster"]),
                "answer_state": str(gold["answer_state"]),
                "answer_count_group": str(gold["answer_count_group"]),
                "category": str(gold["category"]),
                "candidate_unit_count": int(gold["candidate_unit_count"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "contract_length_quartile": str(gold["contract_length_quartile"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "top3_gate": gate,
                "configurations": configurations,
            }
        )

    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = _strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_frc = _strongest(evidence, UNGATED_FRC_CONTROLS)
    comparisons = {
        "candidate_minus_strongest_shared_gate_non_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_non_frc
        ),
        "candidate_minus_same_gate_frc_ablation": _cluster_bootstrap(
            evidence, CANDIDATE, SAME_GATE_FRC_ABLATION
        ),
        "candidate_minus_ungated_v49": _cluster_bootstrap(
            evidence, CANDIDATE, LOW_CORE_DIVERGENCE_V49
        ),
        "candidate_minus_strongest_ungated_frozen_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_frc
        ),
    }
    candidate = aggregates[CANDIDATE]
    ungated = aggregates[LOW_CORE_DIVERGENCE_V49]
    recall_drop = round(
        ungated["answer_macro_recall"] - candidate["answer_macro_recall"], 6
    )

    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        subset = []
        for row in evidence:
            clone = dict(row)
            clone["configurations"] = {str(budget): row["configurations"][str(budget)]}
            subset.append(clone)
        _, delta = _stratum_delta(subset)
        budget_deltas[str(budget)] = delta

    dimensions = {
        "answer_state": lambda row: row["answer_state"],
        "candidate_pool_quartile": lambda row: row["candidate_pool_quartile"],
        "contract_length_quartile": lambda row: row["contract_length_quartile"],
        "category": lambda row: row["category"],
        "gate_outcome": lambda row: (
            "passed"
            if row["top3_gate"]["top3_rank_concurrence_passed"]
            else "abstained"
        ),
        "answer_count_group": lambda row: row["answer_count_group"],
    }
    strata: dict[str, dict[str, Any]] = {}
    for dimension, getter in dimensions.items():
        for value in sorted({str(getter(row)) for row in evidence}):
            subset = [row for row in evidence if str(getter(row)) == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(subset)
            strata[f"{dimension}:{value}"] = {
                "cases": len(subset),
                "strongest_shared_gate_non_frc": strongest,
                "delta": delta,
            }
    minimum_delta = min(
        [*budget_deltas.values(), *(row["delta"] for row in strata.values())],
        default=-math.inf,
    )
    ceiling = float(
        np.mean(
            [
                row["candidate_ceiling_complete"]
                for row in evidence
                if row["answer_state"] == "answer_bearing"
            ]
        )
    )
    answer_state_counts = Counter(row["answer_state"] for row in evidence)
    contracts = len({row["contract_cluster"] for row in evidence})
    checks = {
        "exact_cases_equals_400": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": answer_state_counts
        == {"answer_bearing": TARGET_PER_GROUP, "no_answer": TARGET_PER_GROUP},
        "minimum_contracts_at_least_90": contracts >= MINIMUM_CONTRACTS,
        "candidate_ceiling_complete_rate_at_least_0_9": ceiling >= 0.9,
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_01": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        >= 0.01,
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_same_gate_frc_ablation_point_at_least_0_005": comparisons[
            "candidate_minus_same_gate_frc_ablation"
        ]["point"]
        >= 0.005,
        "candidate_minus_same_gate_frc_ablation_ci_low_above_0": comparisons[
            "candidate_minus_same_gate_frc_ablation"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_ungated_v49_point_at_least_0_01": comparisons[
            "candidate_minus_ungated_v49"
        ]["point"]
        >= 0.01,
        "candidate_minus_ungated_v49_ci_low_above_0": comparisons[
            "candidate_minus_ungated_v49"
        ]["ci_low"]
        > 0.0,
        "answer_macro_recall_drop_vs_ungated_v49_at_most_0_03": recall_drop <= 0.03,
        "no_answer_abstention_accuracy_at_least_0_2": candidate[
            "no_answer_abstention_accuracy"
        ]
        >= 0.2,
        "candidate_abstention_rate_at_least_0_05": candidate["abstention_rate"] >= 0.05,
        "candidate_abstention_rate_at_most_0_8": candidate["abstention_rate"] <= 0.8,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_03": minimum_delta
        >= -0.03,
        "deterministic_query_fallback_rate_equals_0": query_summary["fallback_rate"]
        == 0.0,
        "score_or_prediction_fallback_rate_equals_0": float(
            source_artifacts.get("score_fallback_rate", 0.0)
        )
        == 0.0,
    }
    supported = all(checks.values())
    if stage == "development":
        status = (
            "CUAD_V53_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_TEST"
            if supported
            else "CUAD_V53_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TEST"
        )
    else:
        status = (
            "CUAD_V53_TOP3_ROLE_CLOSURE_SUPPORT_ESTABLISHED"
            if supported
            else "CUAD_V53_TOP3_ROLE_CLOSURE_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "schema_version": "frc-cuad-top3-role-closure-report-v53",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "stage": stage,
            "split": SOURCE_MEMBERS[stage],
            "cases": len(evidence),
            "contracts": contracts,
            "answer_state_counts": dict(answer_state_counts),
            "budgets": list(BUDGETS),
            "official_leaderboard_result": False,
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "gold_joined_after_complete_score_cache": True,
            "contractnli_case_level_artifact_reused": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "gate_formula": "top3(anchor) intersect (top3(first_fact) union top3(second_fact_or_bridge)) is nonempty",
            "selector_formula": "ordered concurrence nucleus union unchanged v49 low-core selection under token budget",
            "aggregates": aggregates,
            "strongest_shared_gate_non_frc": strongest_non_frc,
            "strongest_ungated_frozen_frc": strongest_frc,
            "family_comparison": comparisons,
            "answer_macro_recall_drop_vs_ungated_v49": recall_drop,
            "candidate_ceiling_complete_rate": round(ceiling, 6),
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "minimum_budget_or_supported_stratum_delta": round(minimum_delta, 6),
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_established": supported,
                "test_open_authorized": stage == "development" and supported,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_stage_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    return report, evidence


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    rounded = _round_for_display(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    outcome = analysis["outcome"]
    candidate = analysis["aggregates"][CANDIDATE]
    lines = [
        f"# CUAD Top-3 rank-concurrence role closure ({rounded['metadata']['stage']}, v53)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/contracts: {rounded['metadata']['cases']}/{rounded['metadata']['contracts']}",
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        f"- Candidate answer F1: {candidate['answer_bearing_macro_f1']:.6f}",
        f"- Candidate answer recall: {candidate['answer_macro_recall']:.6f}",
        f"- No-answer abstention accuracy: {candidate['no_answer_abstention_accuracy']:.6f}",
        f"- Abstention rate: {candidate['abstention_rate']:.6f}",
        "- Selector adoption: `false`",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Family comparisons",
        "",
    ]
    for name, value in analysis["family_comparison"].items():
        lines.append(
            f"- `{name}`: {value['point']:+.6f} "
            f"(95% CI [{value['ci_low']:+.6f}, {value['ci_high']:+.6f}])"
        )
    lines.extend(["", "## Support checks", ""])
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a balanced, full-given-contract CUAD clause-selection mechanism experiment. It is not an official CUAD leaderboard result, open-corpus retrieval, legal advice, SetR reproduction, selector adoption, or flood-domain expert validation.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            _round_for_display(row),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                )


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("CUAD v53 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = value["source_access_boundary_at_registration"]
    if boundary["cuad_data_archive_downloaded_locally"]:
        raise ValueError("CUAD v53 archive was downloaded before protocol registration")
    if boundary["cuad_train_content_opened"] or boundary["cuad_test_content_opened"]:
        raise ValueError("CUAD v53 source content was opened before registration")
    if (
        value["development_and_confirmation_scope"]["development_target_cases"]
        != TARGET_CASES
    ):
        raise ValueError("CUAD v53 development size changed")
    if value["evaluation"]["token_budgets"] != list(BUDGETS):
        raise ValueError("CUAD v53 budgets changed")
    if not value["top3_rank_concurrence_gate"][
        "top_k_equals_number_of_semantic_roles_and_is_not_dataset_fitted"
    ]:
        raise ValueError("CUAD v53 parameter-free Top-3 boundary changed")
    return value


def validate_source_registration(
    registration_path: Path, *, protocol_path: Path, source_archive: Path
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value["protocol_sha256_before_download"] != _sha256(protocol_path):
        raise ValueError("CUAD v53 source registration protocol hash mismatch")
    if value["data_archive"]["sha256"] != _sha256(source_archive):
        raise ValueError("CUAD v53 source archive hash mismatch")
    if value["data_archive"]["sha256"] != SOURCE_ARCHIVE_SHA256:
        raise ValueError("CUAD v53 source archive changed")
    if any(
        value["data_archive"][key]
        for key in (
            "cuadv1_content_opened",
            "train_content_opened",
            "test_content_opened",
        )
    ):
        raise ValueError("CUAD v53 data content was opened before source registration")
    return value


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    source_registration_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "source_registration_sha256": _sha256(source_registration_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("CUAD v53 implementation hash mismatch")
    if value.get("train_or_test_content_opened_before_registration") is not False:
        raise ValueError("CUAD v53 implementation was registered too late")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("CUAD v53 synthetic invariants are incomplete")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE",
    "CAPABILITY",
    "DATASET_ID",
    "EXPERIMENT_ID",
    "FrozenCuadScorer",
    "GATED_NON_FRC_METHODS",
    "MAX_CASES_PER_CONTRACT",
    "METHODS",
    "PROTOCOL_SHA256",
    "SAME_GATE_FRC_ABLATION",
    "SCHEMA_VERSION",
    "SOURCE_ARCHIVE_SHA256",
    "SOURCE_MEMBERS",
    "STAGES",
    "TARGET_CASES",
    "TARGET_PER_GROUP",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "build_retrieval_pool",
    "build_source_candidates",
    "evaluate_stage",
    "official_answer_match",
    "prepare_blind_cases",
    "read_source_member",
    "select_balanced_sample",
    "select_v53",
    "top3_rank_concurrence_details",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_registration",
    "write_report",
]
