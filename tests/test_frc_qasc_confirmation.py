from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag import qasc_confirmation
from research.frc_rag.qasc_confirmation import (
    CANDIDATE_COUNT,
    QascGoldBlindScorer,
    load_and_prepare_parquet,
    preparation_summary,
    prepare_cases,
)


def _rows(count: int = 45) -> list[dict]:
    rows = []
    for index in range(count):
        labels = list("ABCDEFGH")
        texts = [f"choice {letter} for {index}" for letter in labels]
        rows.append(
            {
                "id": f"qasc-{index:03d}",
                "question": (
                    f"What process connects science concept {index} to result?"
                ),
                "choices": {"text": texts, "label": labels},
                "answerKey": labels[index % len(labels)],
                "fact1": f"Science concept {index} starts intermediate process {index}.",
                "fact2": f"Intermediate process {index} produces final result {index}.",
                "combinedfact": f"Science concept {index} produces result {index}.",
                "formatted_question": (
                    f"What process connects science concept {index} to result? "
                    + " ".join(
                        f"({label}) {text}"
                        for label, text in zip(labels, texts, strict=True)
                    )
                ),
            }
        )
    return rows


class _SpyScorer:
    def __init__(self) -> None:
        self.received: list[dict] = []

    def score_case(self, case: dict) -> dict:
        assert "answer" not in case
        assert "gold_evidence_ids" not in case
        assert all("gold" not in item for item in case["candidates"])
        assert all("gold_roles" not in item for item in case["candidates"])
        self.received.append(json.loads(json.dumps(case)))
        candidates = []
        for index, candidate in enumerate(case["candidates"]):
            candidates.append(
                {
                    **candidate,
                    "scores": {
                        "bm25": index / 100,
                        "dense": index / 90,
                        "hybrid": index / 80,
                        "cross_encoder": index / 70,
                    },
                    "role_scores": {
                        role: index / 60
                        for role in (
                            "condition",
                            "attribution",
                            "procedure",
                            "answer",
                            "exception",
                        )
                    },
                }
            )
        return {**case, "candidates": candidates}


def test_qasc_hard_pool_is_deterministic_and_gold_blind(monkeypatch) -> None:
    monkeypatch.setattr(qasc_confirmation, "MINIMUM_ELIGIBLE_CASES", 40)
    rows = _rows()
    first = prepare_cases(rows)
    second = prepare_cases(reversed(rows))
    assert first == second
    assert len(first) == 45
    assert all(len(case["candidates"]) == CANDIDATE_COUNT for case in first)
    assert all(len(case["gold_evidence_ids"]) == 2 for case in first)
    assert all(case["question_type"] == "qasc_what" for case in first)
    assert all(
        sorted(
            role for candidate in case["candidates"] for role in candidate["gold_roles"]
        )
        == ["answer", "procedure"]
        for case in first
    )
    summary = preparation_summary(first)
    assert summary["candidate_count"] == {
        "total": 1800,
        "minimum": 40,
        "maximum": 40,
        "mean": 40.0,
    }
    assert summary["gold_evidence_count"]["mean"] == 2.0

    source_case = first[0]
    spy = _SpyScorer()
    scored = QascGoldBlindScorer(spy).score_case(source_case)
    assert spy.received
    assert scored["answer"] == source_case["answer"]
    assert scored["gold_evidence_ids"] == source_case["gold_evidence_ids"]
    assert all("scores" in item for item in scored["candidates"])

    mutated = json.loads(json.dumps(source_case))
    mutated["answer"] = "invented answer"
    mutated["gold_evidence_ids"] = [mutated["candidates"][0]["id"]]
    for candidate in mutated["candidates"]:
        candidate["gold"] = not candidate["gold"]
        candidate["gold_roles"] = ["exception"]
    mutated_spy = _SpyScorer()
    mutated_scored = QascGoldBlindScorer(mutated_spy).score_case(mutated)
    assert spy.received == mutated_spy.received
    assert [item["scores"] for item in scored["candidates"]] == [
        item["scores"] for item in mutated_scored["candidates"]
    ]


def test_qasc_source_hash_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "validation.parquet"
    source.write_bytes(b"not the frozen QASC artifact")
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_and_prepare_parquet(source)
