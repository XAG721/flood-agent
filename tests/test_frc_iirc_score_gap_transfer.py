from __future__ import annotations

import hashlib
import json
from pathlib import Path

import research.frc_rag.iirc_cardinality_transfer as v86
import research.frc_rag.iirc_score_gap_transfer as v87


def _commit(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def test_v87_partitions_only_use_v86_remaining(monkeypatch: object) -> None:
    ids = [f"q{index:04d}" for index in range(2400)]
    rows = [{"questions": [{"qid": value} for value in ids]}]
    monkeypatch.setitem(v86.SOURCE_COUNTS, "train_articles", 1)
    monkeypatch.setitem(v86.SOURCE_COUNTS, "train_questions", len(ids))
    v86_ordered = sorted(
        ids,
        key=lambda value: (
            hashlib.sha256(f"{v86.PARTITION_SALT}{value}".encode()).hexdigest(),
            value,
        ),
    )
    v86_parts = {
        "all": sorted(ids),
        "development": v86_ordered[:400],
        "confirmation": v86_ordered[400:1200],
        "remaining": v86_ordered[1200:],
    }
    for name, values in v86_parts.items():
        monkeypatch.setitem(v86.PARTITION_COMMITMENTS, name, _commit(values))
    eligible = sorted(v86_parts["remaining"])
    ordered = sorted(
        eligible,
        key=lambda value: (
            hashlib.sha256(f"{v87.PARTITION_SALT}{value}".encode()).hexdigest(),
            value,
        ),
    )
    expected = {
        "eligible": eligible,
        "development": ordered[:400],
        "confirmation": ordered[400:1200],
        "remaining": [],
    }
    for name, values in expected.items():
        monkeypatch.setitem(v87.PARTITION_COMMITMENTS, name, _commit(values))
    actual = v87.build_partitions(rows)
    assert actual == expected
    assert not (set(actual["eligible"]) & set(v86_parts["development"]))
    assert not (set(actual["eligible"]) & set(v86_parts["confirmation"]))


def test_method_registry_keeps_v86_candidate_as_control() -> None:
    assert v87.V86_CANDIDATE_METHOD in v87.CONTROL_METHODS
    assert v87.CANDIDATE_METHOD not in v87.CONTROL_METHODS
    assert len(v87.METHODS) == len(set(v87.METHODS))


def test_evaluation_rejects_incomplete_method_registry(
    tmp_path: Path, monkeypatch: object
) -> None:
    selection = tmp_path / "selection.jsonl"
    gold = tmp_path / "gold.jsonl"
    selection.write_text(
        json.dumps({"case_id": "iirc::q1", "methods": {}}, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            {
                "id": "iirc::q1",
                "answer_type": "span",
                "gold_evidence_sources": ["s1"],
                "candidate_ceiling_complete": True,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(v87.STAGE_CASES, "development", 1)
    try:
        v87.evaluate_stage(selection, gold, stage="development")
    except ValueError as error:
        assert "selection methods changed" in str(error)
    else:
        raise AssertionError("v87 accepted an incomplete method registry")
