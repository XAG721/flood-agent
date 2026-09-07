"""Gold-blind global batching for QASC scoring without changing frozen scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from research.frc_rag.public_evidence import select_precomputed
from research.frc_rag.qasc_confirmation import _gold_blind_view
from research.frc_rag.twowiki_confirmation import (
    ROLE_NAMES,
    ROLE_QUERIES,
    FrozenBgeScorer,
    _bm25,
    _minmax,
    _rrf,
    canonical_json_sha256,
)


SCHEMA_VERSION = "frc-qasc-global-batch-scoring-benchmark-v1"
SCORE_NAMES = ("bm25", "dense", "hybrid", "cross_encoder")


class BatchCaseScorer(Protocol):
    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class GlobalBatchedFrozenBgeScorer(FrozenBgeScorer):
    """Batch all QASC cases while preserving per-case score calibration."""

    def _predict_raw(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        values = self.reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=float)

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not cases:
            return []
        if any(not case.get("candidates") for case in cases):
            raise ValueError("QASC global batching requires non-empty candidates")

        unique_texts = list(
            dict.fromkeys(
                str(candidate["text"])
                for case in cases
                for candidate in case["candidates"]
            )
        )
        text_vectors = self._encode(unique_texts)
        vector_by_text = {
            text: text_vectors[index] for index, text in enumerate(unique_texts)
        }
        questions = [str(case["question"]) for case in cases]
        question_vectors = self._encode(questions)

        relevance_pairs: list[tuple[str, str]] = []
        relevance_spans: list[tuple[int, int]] = []
        for question, case in zip(questions, cases, strict=True):
            start = len(relevance_pairs)
            relevance_pairs.extend(
                (question, str(candidate["text"]))
                for candidate in case["candidates"]
            )
            relevance_spans.append((start, len(relevance_pairs)))
        relevance_raw = self._predict_raw(relevance_pairs)

        role_pairs: list[tuple[str, str]] = []
        role_spans: dict[tuple[int, str], tuple[int, int]] = {}
        for case_index, (question, case) in enumerate(
            zip(questions, cases, strict=True)
        ):
            for role in ROLE_NAMES:
                role_question = f"{ROLE_QUERIES[role]} Question: {question}"
                start = len(role_pairs)
                role_pairs.extend(
                    (role_question, str(candidate["text"]))
                    for candidate in case["candidates"]
                )
                role_spans[(case_index, role)] = (start, len(role_pairs))
        role_raw = self._predict_raw(role_pairs)

        output = []
        for case_index, case in enumerate(cases):
            candidates = list(case["candidates"])
            texts = [str(candidate["text"]) for candidate in candidates]
            candidate_vectors = np.asarray(
                [vector_by_text[text] for text in texts], dtype=np.float32
            )
            question = questions[case_index]
            bm25 = _bm25(question, texts)
            dense = _minmax(candidate_vectors @ question_vectors[case_index])
            hybrid = _rrf(bm25, dense, rrf_k=self.rrf_k)
            start, end = relevance_spans[case_index]
            cross = _minmax(relevance_raw[start:end])
            role_values: dict[str, np.ndarray] = {}
            for role in ROLE_NAMES:
                start, end = role_spans[(case_index, role)]
                calibrated = _minmax(role_raw[start:end])
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
                        "role_scores": {
                            role: float(values[candidate_index])
                            for role, values in role_values.items()
                        },
                    }
                )
            output.append({**case, "candidates": scored_candidates})
        return output


def _restore_labels(
    source: dict[str, Any], scored: dict[str, Any]
) -> dict[str, Any]:
    labels = {
        str(candidate["id"]): {
            "gold": bool(candidate["gold"]),
            "gold_roles": list(candidate["gold_roles"]),
        }
        for candidate in source["candidates"]
    }
    scored_candidates = []
    for candidate in scored["candidates"]:
        candidate_id = str(candidate["id"])
        if candidate_id not in labels:
            raise ValueError("QASC batch scorer changed candidate ids")
        scored_candidates.append({**candidate, **labels[candidate_id]})
    if {str(item["id"]) for item in scored_candidates} != set(labels):
        raise ValueError("QASC batch scorer dropped candidates")
    return {
        **scored,
        "answer": source["answer"],
        "gold_evidence_ids": list(source["gold_evidence_ids"]),
        "candidates": scored_candidates,
    }


class QascGoldBlindBatchScorer:
    """Strip all evaluation labels before one global batch scoring call."""

    def __init__(self, delegate: BatchCaseScorer) -> None:
        self.delegate = delegate

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        blind = [_gold_blind_view(case) for case in cases]
        scored = self.delegate.score_cases(blind)
        if len(scored) != len(cases):
            raise ValueError("QASC batch scorer changed case count")
        scored_by_id = {str(case["id"]): case for case in scored}
        if len(scored_by_id) != len(scored):
            raise ValueError("QASC batch scorer returned duplicate case ids")
        source_ids = {str(case["id"]) for case in cases}
        if set(scored_by_id) != source_ids:
            raise ValueError("QASC batch scorer changed case ids")
        return [
            _restore_labels(case, scored_by_id[str(case["id"])]) for case in cases
        ]


def select_benchmark_cases(
    cases: list[dict[str, Any]], *, count: int, seed: str
) -> list[dict[str, Any]]:
    if count < 1 or count > len(cases):
        raise ValueError("QASC benchmark count is outside source size")
    ranked = sorted(
        cases,
        key=lambda case: (
            canonical_json_sha256([seed, str(case["id"])]),
            str(case["id"]),
        ),
    )
    return ranked[:count]


def _rank_ids(case: dict[str, Any], namespace: str, name: str) -> list[str]:
    return [
        str(candidate["id"])
        for candidate in sorted(
            case["candidates"],
            key=lambda candidate: (
                -float(candidate[namespace][name]),
                str(candidate["id"]),
            ),
        )
    ]


def compare_scored_cases(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    if [str(case["id"]) for case in reference] != [
        str(case["id"]) for case in candidate
    ]:
        raise ValueError("QASC score comparison case ids differ")
    max_score_delta = 0.0
    max_role_delta = 0.0
    score_rank_mismatches = 0
    role_rank_mismatches = 0
    selector_mismatches = 0
    candidate_count = 0
    for expected, actual in zip(reference, candidate, strict=True):
        expected_ids = [str(item["id"]) for item in expected["candidates"]]
        actual_ids = [str(item["id"]) for item in actual["candidates"]]
        if expected_ids != actual_ids:
            raise ValueError("QASC score comparison candidate ids differ")
        candidate_count += len(expected_ids)
        for left, right in zip(
            expected["candidates"], actual["candidates"], strict=True
        ):
            for name in SCORE_NAMES:
                max_score_delta = max(
                    max_score_delta,
                    abs(float(left["scores"][name]) - float(right["scores"][name])),
                )
            for role in ROLE_NAMES:
                max_role_delta = max(
                    max_role_delta,
                    abs(
                        float(left["role_scores"][role])
                        - float(right["role_scores"][role])
                    ),
                )
        score_rank_mismatches += sum(
            _rank_ids(expected, "scores", name)
            != _rank_ids(actual, "scores", name)
            for name in SCORE_NAMES
        )
        role_rank_mismatches += sum(
            _rank_ids(expected, "role_scores", role)
            != _rank_ids(actual, "role_scores", role)
            for role in ROLE_NAMES
        )
        expected_selected = [
            str(item["id"])
            for item in select_precomputed(expected, "frc_select")
        ]
        actual_selected = [
            str(item["id"])
            for item in select_precomputed(actual, "frc_select")
        ]
        selector_mismatches += expected_selected != actual_selected
    passed = (
        max_score_delta <= tolerance
        and max_role_delta <= tolerance
        and score_rank_mismatches == 0
        and role_rank_mismatches == 0
        and selector_mismatches == 0
    )
    return {
        "case_count": len(reference),
        "candidate_count": candidate_count,
        "tolerance": tolerance,
        "max_absolute_score_delta": max_score_delta,
        "max_absolute_role_score_delta": max_role_delta,
        "score_ranking_mismatches": score_rank_mismatches,
        "role_ranking_mismatches": role_rank_mismatches,
        "frc_selection_mismatches": selector_mismatches,
        "passed": passed,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
