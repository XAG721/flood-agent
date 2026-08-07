from __future__ import annotations

import hashlib
import json
from pathlib import Path

import research.frc_rag.iirc_cardinality_transfer as v86
import research.frc_rag.iirc_pooled_score_gap_transfer as v88
import research.frc_rag.iirc_score_gap_transfer as v87


def _commit(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _ordered(values: list[str], salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{salt}{value}".encode()).hexdigest(),
            value,
        ),
    )


def test_v88_partitions_only_use_v87_remaining(monkeypatch: object) -> None:
    ids = [f"q{index:04d}" for index in range(4400)]
    rows = [{"questions": [{"qid": value} for value in ids]}]
    monkeypatch.setitem(v86.SOURCE_COUNTS, "train_articles", 1)
    monkeypatch.setitem(v86.SOURCE_COUNTS, "train_questions", len(ids))
    order86 = _ordered(ids, v86.PARTITION_SALT)
    parts86 = {
        "all": sorted(ids),
        "development": order86[:400],
        "confirmation": order86[400:1200],
        "remaining": order86[1200:],
    }
    for name, values in parts86.items():
        monkeypatch.setitem(v86.PARTITION_COMMITMENTS, name, _commit(values))
    eligible87 = sorted(parts86["remaining"])
    order87 = _ordered(eligible87, v87.PARTITION_SALT)
    parts87 = {
        "eligible": eligible87,
        "development": order87[:400],
        "confirmation": order87[400:1200],
        "remaining": order87[1200:],
    }
    for name, values in parts87.items():
        monkeypatch.setitem(v87.PARTITION_COMMITMENTS, name, _commit(values))
    eligible88 = sorted(parts87["remaining"])
    order88 = _ordered(eligible88, v88.PARTITION_SALT)
    expected = {
        "eligible": eligible88,
        "development": order88[:800],
        "confirmation": order88[800:2000],
        "remaining": order88[2000:],
    }
    for name, values in expected.items():
        monkeypatch.setitem(v88.PARTITION_COMMITMENTS, name, _commit(values))
    actual = v88.build_partitions(rows)
    assert actual == expected
    assert not (set(actual["eligible"]) & set(parts87["development"]))
    assert not (set(actual["eligible"]) & set(parts87["confirmation"]))


def test_v88_method_registry_keeps_prior_candidates_as_controls() -> None:
    assert v87.V86_CANDIDATE_METHOD in v88.CONTROL_METHODS
    assert v87.CANDIDATE_METHOD in v88.CONTROL_METHODS
    assert v88.CANDIDATE_METHOD not in v88.CONTROL_METHODS
    assert len(v88.METHODS) == len(set(v88.METHODS))


def test_v88_evaluation_rejects_incomplete_methods(
    tmp_path: Path, monkeypatch: object
) -> None:
    selection = tmp_path / "selection.jsonl"
    gold = tmp_path / "gold.jsonl"
    selection.write_text(
        json.dumps({"case_id": "iirc::q1", "methods": {}}) + "\n",
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            {
                "id": "iirc::q1",
                "answer_type": "span",
                "gold_evidence_sources": ["s1"],
                "candidate_ceiling_complete": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(v88.STAGE_CASES, "development", 1)
    try:
        v88.evaluate_stage(selection, gold, stage="development")
    except ValueError as error:
        assert "selection methods changed" in str(error)
    else:
        raise AssertionError("v88 accepted an incomplete method registry")
