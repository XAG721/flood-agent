from __future__ import annotations

import hashlib

import numpy as np

from research.frc_rag.qasc_confirmation import QascGoldBlindScorer
from research.frc_rag.qasc_scoring_optimization import (
    GlobalBatchedFrozenBgeScorer,
    QascGoldBlindBatchScorer,
    compare_scored_cases,
    select_benchmark_cases,
)


class _FakeEmbedder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.asarray([value + 1 for value in digest[:8]], dtype=np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32)


class _FakeReranker:
    def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
        return np.asarray(
            [
                int.from_bytes(
                    hashlib.sha256(f"{left}\0{right}".encode()).digest()[:8],
                    "big",
                )
                / 2**64
                for left, right in pairs
            ],
            dtype=float,
        )


def _scorer() -> GlobalBatchedFrozenBgeScorer:
    scorer = object.__new__(GlobalBatchedFrozenBgeScorer)
    scorer.embedder = _FakeEmbedder()
    scorer.reranker = _FakeReranker()
    scorer.embedding_batch_size = 8
    scorer.rerank_batch_size = 8
    scorer.role_relevance_mix = 0.15
    scorer.rrf_k = 60
    return scorer


def _cases() -> list[dict]:
    cases = []
    for case_index in range(3):
        candidates = []
        for candidate_index in range(6):
            candidates.append(
                {
                    "id": f"fact-{candidate_index}",
                    "text": f"shared science fact {candidate_index}",
                    "token_count": 5 + candidate_index,
                    "gold": candidate_index < 2,
                    "gold_roles": (
                        ["procedure"]
                        if candidate_index == 0
                        else ["answer"] if candidate_index == 1 else []
                    ),
                }
            )
        cases.append(
            {
                "dataset": "QASC",
                "id": f"qasc-{case_index}",
                "question": f"What connects process {case_index} to result?",
                "question_type": "qasc_what",
                "required_roles": ["procedure", "answer"],
                "answer": f"result {case_index}",
                "gold_evidence_ids": ["fact-0", "fact-1"],
                "candidates": candidates,
            }
        )
    return cases


def test_global_batch_scores_match_per_case_pipeline_exactly() -> None:
    scorer = _scorer()
    sequential_wrapper = QascGoldBlindScorer(scorer)
    sequential = [sequential_wrapper.score_case(case) for case in _cases()]
    batched = QascGoldBlindBatchScorer(scorer).score_cases(_cases())
    comparison = compare_scored_cases(sequential, batched, tolerance=0.0)
    assert comparison["passed"] is True
    assert comparison["max_absolute_score_delta"] == 0.0
    assert comparison["max_absolute_role_score_delta"] == 0.0
    assert comparison["frc_selection_mismatches"] == 0


def test_global_batch_wrapper_is_gold_blind_and_restores_labels() -> None:
    class Spy:
        def score_cases(self, cases: list[dict]) -> list[dict]:
            assert all("answer" not in case for case in cases)
            assert all("gold_evidence_ids" not in case for case in cases)
            assert all(
                "gold" not in candidate and "gold_roles" not in candidate
                for case in cases
                for candidate in case["candidates"]
            )
            return [
                {
                    **case,
                    "candidates": [
                        {
                            **candidate,
                            "scores": {name: 0.0 for name in (
                                "bm25",
                                "dense",
                                "hybrid",
                                "cross_encoder",
                            )},
                            "role_scores": {
                                name: 0.0
                                for name in (
                                    "condition",
                                    "attribution",
                                    "procedure",
                                    "answer",
                                    "exception",
                                )
                            },
                        }
                        for candidate in case["candidates"]
                    ],
                }
                for case in cases
            ]

    source = _cases()
    scored = QascGoldBlindBatchScorer(Spy()).score_cases(source)
    assert [case["answer"] for case in scored] == [case["answer"] for case in source]
    assert all(
        candidate["gold"] == source[case_index]["candidates"][candidate_index]["gold"]
        for case_index, case in enumerate(scored)
        for candidate_index, candidate in enumerate(case["candidates"])
    )


def test_benchmark_selection_and_comparison_fail_closed() -> None:
    cases = _cases()
    assert select_benchmark_cases(cases, count=2, seed="fixed") == select_benchmark_cases(
        list(reversed(cases)), count=2, seed="fixed"
    )
    scored = QascGoldBlindBatchScorer(_scorer()).score_cases(cases)
    altered = QascGoldBlindBatchScorer(_scorer()).score_cases(cases)
    altered[0]["candidates"][0]["scores"]["dense"] += 1e-3
    comparison = compare_scored_cases(scored, altered, tolerance=1e-6)
    assert comparison["passed"] is False
    assert comparison["max_absolute_score_delta"] > 1e-6
