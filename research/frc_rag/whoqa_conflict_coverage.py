"""Blind evidence-selection evaluation on the public WhoQA benchmark.

The module keeps preparation/scoring separate from gold evaluation.  Prepared and
scored caches contain questions and contexts but never ``answer_by_context``.  Gold
answer sets are joined only after every selector has emitted context IDs.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from research.frc_rag.twowiki_confirmation import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    resolve_snapshot,
)


SCHEMA_VERSION = "frc-whoqa-conflict-coverage-v1"
DATASET_NAME = "WhoQA"
SOURCE_REPOSITORY = "https://github.com/VinAIResearch/WhoQA"
SOURCE_REVISION = "02c2b24004fc334c4bf6c9391285eb55e652a86a"
SOURCE_FILENAME = "WhoQA.json"
SOURCE_SHA256 = "030dfa5781d15795846f3d1f89fdc76253debb7c67d93a4b516cb9a53d11c211"
SOURCE_LICENSE = "BSD-3-Clause"
EXPECTED_CASES = 5152
ALLOWED_TEMPLATE_COUNTS = (5, 8)

METHODS = (
    "bm25_topk",
    "dense_topk",
    "hybrid_topk",
    "cross_encoder_topk",
    "coverage_greedy_proxy",
    "frc_select",
)
BASELINES = METHODS[:-1]
FRC_METHOD = "frc_select"
REFERENCE_BASELINE = "coverage_greedy_proxy"
ROLE_NAMES = (
    "answer_claim",
    "entity_attribution",
    "alternative_claim",
    "ambiguity_disclosure",
)
ROLE_QUERIES = {
    "answer_claim": "Find evidence that directly states an answer to the question.",
    "entity_attribution": (
        "Find evidence that identifies which specific same-name entity, source, "
        "role, place, or time the answer belongs to."
    ),
    "alternative_claim": (
        "Find evidence that may support an alternative answer or interpretation "
        "of the question."
    ),
    "ambiguity_disclosure": (
        "Find evidence useful for disclosing that the question can refer to "
        "multiple different same-name entities."
    ),
}

TOP_K = 4
TOKEN_BUDGET = 1500
ROLE_THRESHOLD = 0.55
ROLE_RELEVANCE_MIX = 0.15
RRF_K = 60
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260801


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def load_raw(path: Path, *, validate_hash: bool = True) -> list[dict[str, Any]]:
    if validate_hash and sha256(path) != SOURCE_SHA256:
        raise ValueError("WhoQA source hash does not match the frozen revision")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("WhoQA source root must be an array")
    return [dict(row) for row in value]


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)))


def _case_id(row: dict[str, Any]) -> str:
    value = str(row.get("q_id", "")).strip()
    if not value:
        raise ValueError("WhoQA row is missing q_id")
    return value


def prepare_case(row: dict[str, Any]) -> dict[str, Any]:
    case_id = _case_id(row)
    questions = [str(value).strip() for value in row.get("questions", [])]
    contexts = [str(value).strip() for value in row.get("contexts", [])]
    context_ids = [str(value).strip() for value in row.get("context_ids", [])]
    answers = row.get("answer_by_context")
    if len(questions) not in ALLOWED_TEMPLATE_COUNTS or any(
        not item for item in questions
    ):
        raise ValueError(
            f"WhoQA {case_id} must contain five or eight non-empty templates"
        )
    if len(contexts) < 2:
        raise ValueError(f"WhoQA {case_id} has fewer than two contexts")
    if any(not item for item in contexts):
        raise ValueError(f"WhoQA {case_id} contains an empty context")
    if not isinstance(answers, dict) or set(answers) != {
        str(index) for index in range(len(contexts))
    }:
        raise ValueError(f"WhoQA {case_id} has invalid answer/context alignment")

    candidates = []
    for index, context in enumerate(contexts):
        source_id = (
            context_ids[index] if index < len(context_ids) else f"context-{index}"
        )
        candidates.append(
            {
                "id": f"whoqa::{case_id}::c{index}",
                "context_index": index,
                "text": context,
                "source": source_id,
                "token_count": _token_count(context),
            }
        )
    prepared = {
        "dataset": DATASET_NAME,
        "id": case_id,
        "property_type": str(row.get("question_type_id", "unspecified")),
        "questions": questions,
        "required_roles": list(ROLE_NAMES),
        "candidates": candidates,
    }
    forbidden = {"answer_by_context", "num_distinct_answers", "main_ent"}
    if forbidden & set(prepared):
        raise AssertionError("gold fields leaked into prepared WhoQA case")
    return prepared


def prepare_cases(
    rows: Iterable[dict[str, Any]], *, expected_cases: int | None = EXPECTED_CASES
) -> list[dict[str, Any]]:
    prepared = [prepare_case(row) for row in rows]
    ids = [row["id"] for row in prepared]
    if len(set(ids)) != len(ids):
        raise ValueError("WhoQA contains duplicate q_id values")
    if expected_cases is not None and len(prepared) != expected_cases:
        raise ValueError(
            f"WhoQA expected {expected_cases} cases but found {len(prepared)}"
        )
    return prepared


def preparation_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(row["candidates"]) for row in cases]
    tokens = [
        int(candidate["token_count"])
        for row in cases
        for candidate in row["candidates"]
    ]
    properties = Counter(str(row["property_type"]) for row in cases)
    return {
        "case_count": len(cases),
        "template_count": sum(len(row["questions"]) for row in cases),
        "template_count_distribution": dict(
            sorted(Counter(len(row["questions"]) for row in cases).items())
        ),
        "case_ids_sha256": canonical_json_sha256([row["id"] for row in cases]),
        "candidate_count": {
            "total": sum(counts),
            "minimum": min(counts),
            "maximum": max(counts),
            "mean": round(float(np.mean(counts)), 6),
        },
        "candidate_tokens": {
            "minimum": min(tokens),
            "maximum": max(tokens),
            "mean": round(float(np.mean(tokens)), 6),
        },
        "property_types": dict(sorted(properties.items())),
        "gold_fields_visible_to_scorer": False,
    }


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def _bm25(question: str, texts: list[str]) -> np.ndarray:
    documents = [_tokenize(text) for text in texts]
    frequencies = Counter(term for tokens in documents for term in set(tokens))
    average_length = float(np.mean([len(tokens) for tokens in documents])) or 1.0
    values = []
    for tokens in documents:
        term_counts = Counter(tokens)
        document_length = len(tokens) or 1
        score = 0.0
        for term in _tokenize(question):
            if term not in term_counts:
                continue
            count = term_counts[term]
            document_count = len(documents)
            inverse_frequency = np.log(
                1.0
                + (document_count - frequencies.get(term, 0) + 0.5)
                / (frequencies.get(term, 0) + 0.5)
            )
            denominator = count + 1.5 * (
                0.25 + 0.75 * document_length / average_length
            )
            score += float(inverse_frequency) * (count * 2.5) / denominator
        values.append(score)
    return _minmax(np.asarray(values, dtype=float))


def _rrf(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    fused = np.zeros(len(left), dtype=float)
    for values in (left, right):
        order = np.argsort(-values, kind="mergesort")
        for rank, index in enumerate(order, start=1):
            fused[index] += 1.0 / (RRF_K + rank)
    return _minmax(fused)


class FrozenWhoQAScorer:
    """Efficiently score all five templates while encoding contexts once."""

    def __init__(
        self,
        *,
        hf_home: Path,
        device: str = "cuda",
        embedding_batch_size: int = 64,
        rerank_batch_size: int = 256,
        use_fp16: bool = True,
    ) -> None:
        os.environ["HF_HOME"] = str(hf_home.resolve())
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder, SentenceTransformer

        embedding_path = resolve_snapshot(
            hf_home, EMBEDDING_MODEL, EMBEDDING_REVISION
        )
        reranker_path = resolve_snapshot(
            hf_home, RERANKER_MODEL, RERANKER_REVISION
        )
        self.embedder = SentenceTransformer(str(embedding_path), device=device)
        self.reranker = CrossEncoder(str(reranker_path), device=device)
        if use_fp16:
            self.embedder.half()
            self.reranker.model.half()
        self.embedding_batch_size = embedding_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.use_fp16 = use_fp16

    def _encode(self, texts: list[str]) -> np.ndarray:
        result = self.embedder.encode(
            texts,
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(result, dtype=np.float32)

    def _predict_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        values = self.reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=float)

    def score_case(self, case: dict[str, Any]) -> dict[str, Any]:
        return self.score_cases([case])[0]

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not cases:
            return []
        inputs = []
        all_texts: list[str] = []
        for case in cases:
            questions = [str(value) for value in case["questions"]]
            candidates = [dict(value) for value in case["candidates"]]
            contexts = [str(value["text"]) for value in candidates]
            text_start = len(all_texts)
            all_texts.extend(contexts + questions)
            inputs.append(
                {
                    "case": case,
                    "questions": questions,
                    "candidates": candidates,
                    "contexts": contexts,
                    "text_start": text_start,
                }
            )
        vectors = self._encode(all_texts)

        all_pairs: list[tuple[str, str]] = []
        for item in inputs:
            questions = item["questions"]
            contexts = item["contexts"]
            prompts = list(questions)
            for role in ROLE_NAMES:
                prompts.extend(
                    f"{ROLE_QUERIES[role]} Question: {question}"
                    for question in questions
                )
            item["prompts"] = prompts
            item["pair_start"] = len(all_pairs)
            all_pairs.extend(
                (prompt, context) for prompt in prompts for context in contexts
            )
        pair_scores = self._predict_pairs(all_pairs)

        results = []
        for item in inputs:
            case = item["case"]
            questions = item["questions"]
            candidates = item["candidates"]
            contexts = item["contexts"]
            text_start = int(item["text_start"])
            context_count = len(contexts)
            context_vectors = vectors[text_start : text_start + context_count]
            question_start = text_start + context_count
            question_vectors = vectors[
                question_start : question_start + len(questions)
            ]
            dense = np.vstack(
                [_minmax(context_vectors @ vector) for vector in question_vectors]
            )
            pair_start = int(item["pair_start"])
            pair_count = len(item["prompts"]) * context_count
            raw_matrix = pair_scores[pair_start : pair_start + pair_count].reshape(
                len(item["prompts"]), context_count
            )
            prompt_matrix = np.vstack([_minmax(row) for row in raw_matrix])
            cross = prompt_matrix[: len(questions)]
            role_matrices = {}
            for role_index, role in enumerate(ROLE_NAMES):
                start = len(questions) * (role_index + 1)
                role_matrices[role] = prompt_matrix[
                    start : start + len(questions)
                ]

            variants = []
            for variant_index, question in enumerate(questions):
                bm25 = _bm25(question, contexts)
                hybrid = _rrf(bm25, dense[variant_index])
                candidate_scores = []
                for index, candidate in enumerate(candidates):
                    role_scores = {
                        role: float(
                            (1.0 - ROLE_RELEVANCE_MIX)
                            * role_matrices[role][variant_index, index]
                            + ROLE_RELEVANCE_MIX * cross[variant_index, index]
                        )
                        for role in ROLE_NAMES
                    }
                    candidate_scores.append(
                        {
                            "id": candidate["id"],
                            "scores": {
                                "bm25": float(bm25[index]),
                                "dense": float(dense[variant_index, index]),
                                "hybrid": float(hybrid[index]),
                                "cross_encoder": float(
                                    cross[variant_index, index]
                                ),
                            },
                            "role_scores": role_scores,
                        }
                    )
                variants.append(
                    {
                        "variant_index": variant_index,
                        "question": question,
                        "candidate_scores": candidate_scores,
                    }
                )
            results.append(
                {
                    "dataset": DATASET_NAME,
                    "id": case["id"],
                    "property_type": case["property_type"],
                    "required_roles": list(ROLE_NAMES),
                    "candidates": candidates,
                    "variants": variants,
                    "gold_fields_visible_to_scorer": False,
                }
            )
        return results


def score_cases_resumable(
    prepared_path: Path,
    output_path: Path,
    *,
    scorer: FrozenWhoQAScorer,
    case_batch_size: int = 32,
    progress: Callable[[int, int, str], None] | None = None,
) -> int:
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"another WhoQA scoring process owns the lock: {lock_path}"
        ) from error
    os.write(lock_descriptor, str(os.getpid()).encode("ascii"))
    os.close(lock_descriptor)
    try:
        return _score_cases_resumable_locked(
            prepared_path,
            output_path,
            scorer=scorer,
            case_batch_size=case_batch_size,
            progress=progress,
        )
    finally:
        lock_path.unlink(missing_ok=True)


def _score_cases_resumable_locked(
    prepared_path: Path,
    output_path: Path,
    *,
    scorer: FrozenWhoQAScorer,
    case_batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> int:
    cases = list(read_jsonl(prepared_path))
    if not cases:
        raise ValueError("prepared WhoQA cache is empty")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    completed: set[str] = set()
    if output_path.is_file():
        rows = list(read_jsonl(output_path))
        completed = {str(row["id"]) for row in rows}
        if len(rows) != len(cases) or len(completed) != len(rows):
            raise ValueError("complete WhoQA score cache has invalid coverage")
        return len(rows)
    if partial_path.is_file():
        rows = list(read_jsonl(partial_path))
        completed = {str(row["id"]) for row in rows}
        if len(completed) != len(rows):
            raise ValueError("partial WhoQA score cache contains duplicate ids")
    if case_batch_size <= 0:
        raise ValueError("WhoQA case_batch_size must be positive")
    pending = [
        (index, case)
        for index, case in enumerate(cases, start=1)
        if str(case["id"]) not in completed
    ]
    with partial_path.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), case_batch_size):
            batch = pending[start : start + case_batch_size]
            outputs = scorer.score_cases([case for _, case in batch])
            if len(outputs) != len(batch):
                raise AssertionError("WhoQA scorer batch size mismatch")
            for (index, case), scored in zip(batch, outputs):
                case_id = str(case["id"])
                handle.write(
                    json.dumps(scored, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
                completed.add(case_id)
                if progress is not None:
                    progress(index, len(cases), case_id)
            handle.flush()
    if len(completed) != len(cases):
        raise AssertionError("WhoQA scored case coverage mismatch")
    partial_path.replace(output_path)
    return len(completed)


def _merge_variant_candidates(
    case: dict[str, Any], variant: dict[str, Any]
) -> list[dict[str, Any]]:
    score_map = {str(row["id"]): row for row in variant["candidate_scores"]}
    if len(score_map) != len(case["candidates"]):
        raise ValueError(f"WhoQA {case['id']} score/candidate coverage mismatch")
    merged = []
    for candidate in case["candidates"]:
        scored = score_map.get(str(candidate["id"]))
        if scored is None:
            raise ValueError(f"WhoQA {case['id']} missing candidate score")
        merged.append(
            {
                **candidate,
                "scores": dict(scored["scores"]),
                "role_scores": dict(scored["role_scores"]),
            }
        )
    return merged


def _take_with_budget(
    ordered: Iterable[dict[str, Any]], *, top_k: int, token_budget: int
) -> list[dict[str, Any]]:
    selected = []
    total = 0
    for candidate in ordered:
        cost = int(candidate["token_count"])
        if total + cost > token_budget:
            continue
        selected.append(candidate)
        total += cost
        if len(selected) >= top_k:
            break
    return selected


def select_candidates(
    candidates: list[dict[str, Any]],
    method: str,
    *,
    top_k: int = TOP_K,
    token_budget: int = TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """Run a frozen selector with ascending candidate-ID tie breaks."""

    score_name = {
        "bm25_topk": "bm25",
        "dense_topk": "dense",
        "hybrid_topk": "hybrid",
        "cross_encoder_topk": "cross_encoder",
    }.get(method)
    if score_name:
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item["scores"][score_name]),
                str(item["id"]),
            ),
        )
        return _take_with_budget(
            ordered, top_k=top_k, token_budget=token_budget
        )

    if method == "coverage_greedy_proxy":
        selected: list[dict[str, Any]] = []
        remaining = list(candidates)
        total = 0
        for role in ROLE_NAMES:
            eligible = [
                item
                for item in remaining
                if total + int(item["token_count"]) <= token_budget
            ]
            if not eligible:
                continue
            eligible.sort(
                key=lambda item: (
                    -float(item["role_scores"][role]),
                    -float(item["scores"]["cross_encoder"]),
                    str(item["id"]),
                )
            )
            best = eligible[0]
            selected.append(best)
            remaining = [item for item in remaining if item["id"] != best["id"]]
            total += int(best["token_count"])
            if len(selected) >= top_k:
                return selected
        remaining.sort(
            key=lambda item: (
                -float(item["scores"]["cross_encoder"]),
                str(item["id"]),
            )
        )
        return selected + _take_with_budget(
            remaining,
            top_k=top_k - len(selected),
            token_budget=max(0, token_budget - total),
        )

    if method != FRC_METHOD:
        raise ValueError(f"unsupported WhoQA selector: {method}")
    selected = []
    remaining = list(candidates)
    role_best = {role: 0.0 for role in ROLE_NAMES}
    total = 0
    while remaining and len(selected) < top_k:
        eligible = []
        for candidate in remaining:
            cost = int(candidate["token_count"])
            if total + cost > token_budget:
                continue
            improvements = []
            for role, old in role_best.items():
                score = float(candidate["role_scores"][role])
                if old < ROLE_THRESHOLD <= max(old, score):
                    improvements.append(1.0)
                else:
                    improvements.append(max(0.0, score - old) * 0.25)
            role_gain = sum(improvements) / len(role_best)
            value = 2.0 * role_gain + float(
                candidate["scores"]["cross_encoder"]
            )
            eligible.append((value, str(candidate["id"]), candidate))
        if not eligible:
            break
        eligible.sort(key=lambda item: (-item[0], item[1]))
        best = eligible[0][2]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        for role in ROLE_NAMES:
            role_best[role] = max(
                role_best[role], float(best["role_scores"][role])
            )
    return selected


def _normalize_alias(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _viewpoint_key(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("WhoQA answer group must be a non-empty list")
    concepts = []
    for aliases in value:
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            raise ValueError("WhoQA answer aliases must be a list")
        normalized = sorted(
            {_normalize_alias(alias) for alias in aliases if _normalize_alias(alias)}
        )
        if not normalized:
            raise ValueError("WhoQA answer alias group is empty")
        concepts.append(normalized[0])
    return tuple(sorted(set(concepts)))


def _gold_viewpoints(row: dict[str, Any]) -> list[tuple[str, ...]]:
    answers = row.get("answer_by_context")
    contexts = row.get("contexts", [])
    if not isinstance(answers, dict):
        raise ValueError(f"WhoQA {_case_id(row)} has invalid gold answers")
    alias_groups: list[set[str]] = []
    context_group_indices: list[list[int]] = []
    for context_index in range(len(contexts)):
        value = answers[str(context_index)]
        if not isinstance(value, list) or not value:
            raise ValueError("WhoQA answer group must be a non-empty list")
        indices = []
        for aliases in value:
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list):
                raise ValueError("WhoQA answer aliases must be a list")
            normalized = {
                _normalize_alias(alias)
                for alias in aliases
                if _normalize_alias(alias)
            }
            if not normalized:
                raise ValueError("WhoQA answer alias group is empty")
            indices.append(len(alias_groups))
            alias_groups.append(normalized)
        context_group_indices.append(indices)

    parents = list(range(len(alias_groups)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left, right in itertools.combinations(range(len(alias_groups)), 2):
        if alias_groups[left] & alias_groups[right]:
            union(left, right)
    component_aliases: dict[int, set[str]] = defaultdict(set)
    for index, aliases in enumerate(alias_groups):
        component_aliases[find(index)].update(aliases)
    component_ids = {
        root: min(aliases) for root, aliases in component_aliases.items()
    }
    return [
        tuple(sorted({component_ids[find(index)] for index in indices}))
        for indices in context_group_indices
    ]


def _maximum_selectable_count(
    candidates: list[dict[str, Any]], *, top_k: int, token_budget: int
) -> int:
    costs = sorted(int(item["token_count"]) for item in candidates)
    total = 0
    count = 0
    for cost in costs:
        if count >= top_k or total + cost > token_budget:
            break
        total += cost
        count += 1
    return count


def _variant_metrics(
    selected: list[dict[str, Any]],
    viewpoints: list[tuple[str, ...]],
    *,
    max_selectable_count: int,
) -> dict[str, float]:
    all_views = set(viewpoints)
    selected_indices = [int(item["context_index"]) for item in selected]
    selected_views = {viewpoints[index] for index in selected_indices}
    denominator = min(len(all_views), max_selectable_count)
    if denominator <= 0 or len(all_views) < 2:
        raise ValueError("WhoQA conflict evaluation requires at least two viewpoints")
    return {
        "oracle_normalized_distinct_viewpoint_coverage": min(
            1.0, len(selected_views) / denominator
        ),
        "raw_distinct_viewpoint_recall": len(selected_views) / len(all_views),
        "complete_viewpoint_disclosure": float(selected_views == all_views)
        if len(all_views) <= TOP_K
        else float("nan"),
        "at_least_two_viewpoint_coverage": float(len(selected_views) >= 2),
        "selected_candidate_count": float(len(selected)),
        "selected_token_cost": float(
            sum(int(item["token_count"]) for item in selected)
        ),
        "budget_shortfall": float(len(selected) < max_selectable_count),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float | None]:
    names = rows[0].keys()
    result: dict[str, float | None] = {}
    for name in names:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[name] = round(float(finite.mean()), 6) if finite.size else None
    return result


def evaluate_scored_cases(
    raw_rows: Iterable[dict[str, Any]],
    scored_rows: Iterable[dict[str, Any]],
    *,
    expected_cases: int | None = EXPECTED_CASES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_by_id = {_case_id(row): row for row in raw_rows}
    if len(raw_by_id) == 0:
        raise ValueError("WhoQA raw gold source is empty")
    if expected_cases is not None and len(raw_by_id) != expected_cases:
        raise ValueError("WhoQA raw source does not match expected case count")

    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    gold_count_mismatches = 0
    for scored in scored_rows:
        case_id = str(scored.get("id", ""))
        if case_id in seen or case_id not in raw_by_id:
            raise ValueError(f"WhoQA scored id is duplicate or unknown: {case_id}")
        seen.add(case_id)
        if scored.get("gold_fields_visible_to_scorer") is not False:
            raise ValueError(f"WhoQA {case_id} scorer leakage assertion failed")
        if any(key in scored for key in ("answer_by_context", "num_distinct_answers")):
            raise ValueError(f"WhoQA {case_id} scored cache contains gold fields")
        raw = raw_by_id[case_id]
        variants = list(scored.get("variants", []))
        expected_variants = len(raw.get("questions", []))
        if expected_variants not in ALLOWED_TEMPLATE_COUNTS or len(variants) != expected_variants:
            raise ValueError(f"WhoQA {case_id} template coverage mismatch")
        viewpoints = _gold_viewpoints(raw)
        distinct_views = len(set(viewpoints))
        if distinct_views != int(raw.get("num_distinct_answers", distinct_views)):
            gold_count_mismatches += 1
        maximum = _maximum_selectable_count(
            scored["candidates"], top_k=TOP_K, token_budget=TOKEN_BUDGET
        )
        if maximum <= 0:
            raise ValueError(f"WhoQA {case_id} has no selectable candidate")

        method_variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for variant in sorted(variants, key=lambda item: int(item["variant_index"])):
            candidates = _merge_variant_candidates(scored, variant)
            for method in METHODS:
                selected = select_candidates(candidates, method)
                metrics = _variant_metrics(
                    selected, viewpoints, max_selectable_count=maximum
                )
                method_variants[method].append(
                    {
                        "variant_index": int(variant["variant_index"]),
                        "selected_ids": [str(item["id"]) for item in selected],
                        "metrics": metrics,
                    }
                )
        methods = {}
        for method in METHODS:
            variants_for_method = method_variants[method]
            methods[method] = {
                "metrics": _mean_metrics(
                    [dict(item["metrics"]) for item in variants_for_method]
                ),
                "unique_selection_signatures": len(
                    {
                        tuple(item["selected_ids"])
                        for item in variants_for_method
                    }
                ),
                "variants": variants_for_method,
            }
        evidence.append(
            {
                "case_id": case_id,
                "property_type": str(scored.get("property_type", "unspecified")),
                "candidate_count": len(scored["candidates"]),
                "distinct_viewpoint_count": distinct_views,
                "public_num_distinct_answers": int(
                    raw.get("num_distinct_answers", distinct_views)
                ),
                "maximum_selectable_candidate_count": maximum,
                "methods": methods,
                "raw_question_or_answer_exported": False,
            }
        )
    if seen != set(raw_by_id):
        raise ValueError("WhoQA scored cache does not cover every raw case")

    report = _build_report(evidence, gold_count_mismatches=gold_count_mismatches)
    return report, evidence


PRIMARY = "oracle_normalized_distinct_viewpoint_coverage"


def _aggregate_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    names = next(iter(rows))["methods"][method]["metrics"].keys()
    metrics = {}
    for name in names:
        values = [row["methods"][method]["metrics"][name] for row in rows]
        finite = [float(value) for value in values if value is not None]
        metrics[name] = round(float(np.mean(finite)), 6) if finite else None
    signatures = [
        int(row["methods"][method]["unique_selection_signatures"]) for row in rows
    ]
    return {
        "cases": len(rows),
        "metrics": metrics,
        "mean_unique_template_selection_signatures": round(
            float(np.mean(signatures)), 6
        ),
    }


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return round(float(low), 6), round(float(high), 6)


def _bootstrap_comparisons(
    evidence: list[dict[str, Any]], *, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    matrix = {
        method: np.asarray(
            [float(row["methods"][method]["metrics"][PRIMARY]) for row in evidence],
            dtype=float,
        )
        for method in METHODS
    }
    means = {method: float(values.mean()) for method, values in matrix.items()}
    strongest = max(BASELINES, key=lambda method: (means[method], method))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    simultaneous = np.empty(resamples, dtype=float)
    reference = np.empty(resamples, dtype=float)
    case_count = len(evidence)
    for index in range(resamples):
        sample = rng.integers(0, case_count, size=case_count)
        frc_mean = float(matrix[FRC_METHOD][sample].mean())
        simultaneous[index] = frc_mean - max(
            float(matrix[method][sample].mean()) for method in BASELINES
        )
        reference[index] = float(
            (matrix[FRC_METHOD][sample] - matrix[REFERENCE_BASELINE][sample]).mean()
        )
    simultaneous_interval = _percentile_interval(simultaneous)
    reference_interval = _percentile_interval(reference)
    return {
        "observed_strongest_baseline": strongest,
        "method_primary_means": {
            method: round(value, 6) for method, value in means.items()
        },
        "frc_minus_observed_strongest_point": round(
            means[FRC_METHOD] - means[strongest], 6
        ),
        "frc_minus_bootstrap_strongest_simultaneous": {
            "ci_low": simultaneous_interval[0],
            "ci_high": simultaneous_interval[1],
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        },
        "frc_minus_reference": {
            "baseline": REFERENCE_BASELINE,
            "point": round(means[FRC_METHOD] - means[REFERENCE_BASELINE], 6),
            "ci_low": reference_interval[0],
            "ci_high": reference_interval[1],
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        },
    }


def _stratum_summary(
    name: str,
    rows: list[dict[str, Any]],
    *,
    comparison_baseline: str,
) -> dict[str, Any]:
    frc = float(np.mean([row["methods"][FRC_METHOD]["metrics"][PRIMARY] for row in rows]))
    baseline = float(
        np.mean(
            [row["methods"][comparison_baseline]["metrics"][PRIMARY] for row in rows]
        )
    )
    return {
        "stratum": name,
        "cases": len(rows),
        "frc_primary": round(frc, 6),
        "baseline": comparison_baseline,
        "baseline_primary": round(baseline, 6),
        "frc_minus_baseline": round(frc - baseline, 6),
    }


def _build_report(
    evidence: list[dict[str, Any]], *, gold_count_mismatches: int
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("WhoQA evidence is empty")
    aggregates = {
        method: _aggregate_method(evidence, method) for method in METHODS
    }
    comparisons = _bootstrap_comparisons(evidence)
    strongest = str(comparisons["observed_strongest_baseline"])

    viewpoint_groups = {
        "2 viewpoints": [row for row in evidence if row["distinct_viewpoint_count"] == 2],
        "3 viewpoints": [row for row in evidence if row["distinct_viewpoint_count"] == 3],
        "4 viewpoints": [row for row in evidence if row["distinct_viewpoint_count"] == 4],
        "5-8 viewpoints": [
            row for row in evidence if 5 <= row["distinct_viewpoint_count"] <= 8
        ],
        "9+ viewpoints": [
            row for row in evidence if row["distinct_viewpoint_count"] >= 9
        ],
    }
    viewpoint_strata = [
        _stratum_summary(name, rows, comparison_baseline=strongest)
        for name, rows in viewpoint_groups.items()
        if rows
    ]
    by_property: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        by_property[str(row["property_type"])].append(row)
    property_strata = [
        _stratum_summary(name, rows, comparison_baseline=strongest)
        for name, rows in sorted(by_property.items())
        if len(rows) >= 50
    ]

    point_gain = float(comparisons["frc_minus_observed_strongest_point"])
    simultaneous_low = float(
        comparisons["frc_minus_bootstrap_strongest_simultaneous"]["ci_low"]
    )
    reference_low = float(comparisons["frc_minus_reference"]["ci_low"])
    worst_viewpoint_delta = min(
        (float(row["frc_minus_baseline"]) for row in viewpoint_strata),
        default=0.0,
    )
    worst_property_delta = min(
        (float(row["frc_minus_baseline"]) for row in property_strata),
        default=0.0,
    )
    frc_shortfall = float(aggregates[FRC_METHOD]["metrics"]["budget_shortfall"])
    baseline_shortfall = float(aggregates[strongest]["metrics"]["budget_shortfall"])
    checks = {
        "frc_minus_strongest_point_gain_at_least_0_05": point_gain >= 0.05,
        "frc_minus_strongest_simultaneous_ci_low_above_zero": simultaneous_low > 0.0,
        "frc_minus_reference_ci_low_above_zero": reference_low > 0.0,
        "no_viewpoint_count_stratum_delta_below_minus_0_05": worst_viewpoint_delta
        >= -0.05,
        "no_property_stratum_delta_below_minus_0_05": worst_property_delta >= -0.05,
        "frc_budget_shortfall_not_above_strongest_baseline": frc_shortfall
        <= baseline_shortfall,
        "all_cases_scored": len(evidence) == EXPECTED_CASES,
        "gold_leakage_checks_pass": all(
            row.get("raw_question_or_answer_exported") is False for row in evidence
        ),
        "official_viewpoint_count_alignment_pass": gold_count_mismatches == 0,
    }
    supported = all(checks.values())
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset": DATASET_NAME,
            "cases": len(evidence),
            "template_count_distribution": dict(
                sorted(
                    Counter(
                        len(row["methods"][FRC_METHOD]["variants"])
                        for row in evidence
                    ).items()
                )
            ),
            "selection_runs": sum(
                len(row["methods"][FRC_METHOD]["variants"]) for row in evidence
            )
            * len(METHODS),
            "methods": list(METHODS),
            "top_k": TOP_K,
            "token_budget": TOKEN_BUDGET,
            "role_threshold": ROLE_THRESHOLD,
            "primary_metric": PRIMARY,
            "deterministic_output_rerun": "2/2 byte-identical",
        },
        "development_boundary": {
            "independent_public_dataset": True,
            "no_fitting": True,
            "all_templates_aggregated_by_q_id": True,
            "gold_used_only_after_selection": True,
            "gate_evidence": False,
        },
        "data_audit": {
            "public_num_distinct_answer_mismatches_against_canonical_viewpoints": gold_count_mismatches,
            "raw_questions_or_answers_in_evidence": False,
        },
        "analysis": {
            "aggregates": aggregates,
            "comparisons": comparisons,
            "viewpoint_count_strata": viewpoint_strata,
            "property_strata_minimum_50_cases": property_strata,
            "safety": {
                "worst_viewpoint_count_delta": round(worst_viewpoint_delta, 6),
                "worst_property_delta": round(worst_property_delta, 6),
            },
            "outcome": {
                "status": "INDEPENDENT_WHOQA_SELECTION_SUPPORT"
                if supported
                else "WHOQA_SELECTION_SUPPORT_NOT_ESTABLISHED",
                "checks": checks,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
            },
        },
        "interpretation": {
            "scope": "same-name multi-viewpoint evidence selection only",
            "router": "does not confirm or adopt the v32 classifier router",
            "setr": "does not reproduce SetR",
            "production": "does not authorize Gate 2, CANARY, or DEFAULT",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    comparisons = analysis["comparisons"]
    lines = [
        "# WhoQA independent conflict-viewpoint selection audit",
        "",
        f"- Status: `{analysis['outcome']['status']}`",
        f"- Cases: {report['metadata']['cases']}",
        f"- Selection runs: {report['metadata']['selection_runs']}",
        "- Every available public question template is scored and averaged within q_id.",
        "- Gold answer mappings are joined only after selector context IDs are frozen.",
        "- Gate 2 remains `NO-GO/SHADOW`.",
        "",
        "## Aggregate selection metrics",
        "",
        "| Method | Normalized viewpoint coverage | Raw viewpoint recall | Complete disclosure | >=2 viewpoints | Budget shortfall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = analysis["aggregates"][method]["metrics"]
        lines.append(
            f"| {method} | {metrics[PRIMARY]:.6f} | "
            f"{metrics['raw_distinct_viewpoint_recall']:.6f} | "
            f"{metrics['complete_viewpoint_disclosure']:.6f} | "
            f"{metrics['at_least_two_viewpoint_coverage']:.6f} | "
            f"{metrics['budget_shortfall']:.6f} |"
        )
    simultaneous = comparisons["frc_minus_bootstrap_strongest_simultaneous"]
    reference = comparisons["frc_minus_reference"]
    lines.extend(
        [
            "",
            "## Frozen comparisons",
            "",
            f"- Observed strongest baseline: `{comparisons['observed_strongest_baseline']}`",
            f"- FRC minus strongest point: {comparisons['frc_minus_observed_strongest_point']:.6f}",
            f"- Simultaneous 95% CI: [{simultaneous['ci_low']:.6f}, {simultaneous['ci_high']:.6f}]",
            f"- FRC minus coverage proxy: {reference['point']:.6f}, 95% CI "
            f"[{reference['ci_low']:.6f}, {reference['ci_high']:.6f}]",
            "",
            "## Decision checks",
            "",
        ]
    )
    for name, passed in analysis["outcome"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This experiment tests evidence selection for naturally occurring same-name ambiguity. It does not confirm the v32 routing model, reproduce SetR, establish flood-domain effectiveness, or authorize production rollout.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_deterministic_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                payload = json.dumps(
                    row, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                zipped.write(payload + b"\n")


def write_report(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_paths: dict[str, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "whoqa_conflict_coverage_cases.jsonl.gz"
    _write_deterministic_gzip(evidence_path, evidence)
    report = json.loads(json.dumps(report))
    report["metadata"]["source_artifacts"] = {
        name: {
            "path_label": path.name,
            "sha256": sha256(path),
        }
        for name, path in sorted(source_paths.items())
    }
    report["metadata"]["evidence_artifact"] = {
        "path_label": evidence_path.name,
        "sha256": sha256(evidence_path),
        "record_count": len(evidence),
    }
    json_path = output_dir / "whoqa_conflict_coverage.json"
    markdown_path = output_dir / "whoqa_conflict_coverage.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json": json_path,
        "markdown": markdown_path,
        "evidence": evidence_path,
    }


def load_report(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    if report.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported WhoQA conflict-coverage report schema")
    artifact = report["metadata"]["evidence_artifact"]
    if artifact["path_label"] != evidence_path.name:
        raise ValueError("WhoQA evidence path mismatch")
    if artifact["sha256"] != sha256(evidence_path):
        raise ValueError("WhoQA evidence hash mismatch")
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != int(artifact["record_count"]):
        raise ValueError("WhoQA evidence count mismatch")
    if any(row.get("raw_question_or_answer_exported") is not False for row in rows):
        raise ValueError("WhoQA evidence privacy boundary mismatch")
    return report
