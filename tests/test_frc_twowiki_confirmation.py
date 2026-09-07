from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.twowiki_confirmation import (
    load_and_prepare_parquet,
    prepare_case,
    stable_sample_rows,
)


def _row(case_id: str) -> dict:
    return {
        "_id": case_id,
        "question": f"Question {case_id}?",
        "answer": "answer",
        "type": "inference",
        "supporting_facts": [["Alpha", 0], ["Beta", 1]],
        "context": [
            ["Alpha", ["First supporting sentence.", "Distractor."]],
            ["Beta", ["Another distractor.", "Second supporting sentence."]],
        ],
    }


def test_stable_sample_is_independent_of_input_order() -> None:
    rows = [_row("case-a"), _row("case-b"), _row("case-c")]
    forward = stable_sample_rows(rows, sample_size=2, seed=42)
    backward = stable_sample_rows(reversed(rows), sample_size=2, seed=42)
    assert [row["_id"] for row in forward] == [row["_id"] for row in backward]


def test_prepare_case_preserves_sentence_level_gold_evidence() -> None:
    case = prepare_case(_row("case-a"))
    assert case["dataset"] == "2WikiMultiHopQA"
    assert case["question_type"] == "inference"
    assert case["required_roles"] == ["procedure", "answer"]
    assert len(case["candidates"]) == 4
    assert len(case["gold_evidence_ids"]) == 2
    gold = [candidate for candidate in case["candidates"] if candidate["gold"]]
    assert [(item["source"], item["metadata"]["sentence_index"]) for item in gold] == [
        ("Alpha", 0),
        ("Beta", 1),
    ]


def test_prepare_case_rejects_missing_supporting_fact() -> None:
    row = _row("case-a")
    row["supporting_facts"].append(["Missing", 0])
    with pytest.raises(ValueError, match="missing supporting facts"):
        prepare_case(row)


def test_parquet_loader_requires_registered_hash(tmp_path: Path) -> None:
    source = tmp_path / "dev.parquet"
    source.write_text(json.dumps(_row("case-a")), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_and_prepare_parquet(source, sample_size=1)
