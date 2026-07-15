from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from research.frc_rag.housing_ablation import HOUSING_SOURCE_JSON_SHA256
from research.frc_rag.housing_weight_sensitivity import (
    STATUS,
    evaluate_housing_weight_sensitivity,
)
from research.frc_rag.public_evidence import load_housing_public_weight_sensitivity


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_REPORT = (
    REPO_ROOT
    / "output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.json"
)


def _candidate(
    identifier: str,
    jurisdiction: str,
    *,
    cross_encoder: float,
    f1: float = 0.0,
    f2: float = 0.0,
    r1: float = 0.0,
    r2: float = 0.0,
    r3: float = 0.0,
) -> dict:
    return {
        "id": identifier,
        "token_count": 10,
        "scores": {"cross_encoder": cross_encoder},
        "field_scores": {"f1": f1, "f2": f2},
        "role_scores": {"r1": r1, "r2": r2, "r3": r3},
        "metadata": {"jurisdiction": jurisdiction, "snapshot_year": 2021},
    }


def _case(index: int) -> dict:
    jurisdiction = f"J{index}"
    first = f"gold-{index}-1"
    second = f"gold-{index}-2"
    return {
        "id": f"case-{index}",
        "jurisdiction": jurisdiction,
        "as_of_year": 2021,
        "required_roles": ["r1", "r2", "r3"],
        "fields": [{"field_id": "f1"}, {"field_id": "f2"}],
        "field_evidence_map": {"f1": [first], "f2": [second]},
        "gold_evidence_ids": [first, second],
        "candidates": [
            _candidate(
                first,
                jurisdiction,
                cross_encoder=0.70,
                f1=1.0,
                r1=1.0,
            ),
            _candidate(
                second,
                jurisdiction,
                cross_encoder=0.65,
                f2=1.0,
                r2=1.0,
            ),
            _candidate(
                f"role-{index}",
                jurisdiction,
                cross_encoder=0.80,
                r1=1.0,
                r2=1.0,
                r3=1.0,
            ),
        ],
    }


def _metadata(case_count: int) -> dict:
    return {
        "source_sha256": HOUSING_SOURCE_JSON_SHA256,
        "case_signature": "a" * 64,
        "cases": case_count,
        "config": {
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "reranker_model": "BAAI/bge-reranker-large",
            "score_calibration": "per_case_minmax",
            "top_k": 2,
            "token_budget": 100,
            "field_threshold": 0.55,
            "role_threshold": 0.55,
            "field_weight": 2.0,
            "role_weight": 1.0,
        },
    }


def test_public_weight_sensitivity_is_deterministic_and_identifiable(
    tmp_path: Path,
) -> None:
    cases = [_case(1), _case(2)]
    scored_path = tmp_path / "scored.jsonl"
    metadata_path = tmp_path / "metadata.json"
    scored_path.write_text("fixture\n", encoding="utf-8")
    metadata_path.write_text("{}\n", encoding="utf-8")

    first, first_rows = evaluate_housing_weight_sensitivity(
        cases,
        scored_metadata=_metadata(len(cases)),
        scored_cases_path=scored_path,
        scored_metadata_path=metadata_path,
    )
    second, second_rows = evaluate_housing_weight_sensitivity(
        copy.deepcopy(cases),
        scored_metadata=_metadata(len(cases)),
        scored_cases_path=scored_path,
        scored_metadata_path=metadata_path,
    )

    assert first == second
    assert first_rows == second_rows
    assert first["status"] == STATUS
    assert first["decision"] == {
        "gate_2": "NO-GO",
        "public_real_model_weight_sensitivity_complete": True,
        "frozen_parameters_changed": False,
        "reason": (
            "Both field and role weights are behaviorally identifiable on frozen public "
            "real-model scores, but this post-hoc cross-domain sensitivity does not prove "
            "FRC superiority or flood-domain validity."
        ),
    }
    assert all(
        item["status"] == "IDENTIFIABLE" for item in first["identifiability"].values()
    )


def test_public_loader_reaggregates_committed_weight_sensitivity() -> None:
    audited = load_housing_public_weight_sensitivity(COMMITTED_REPORT)

    assert audited["status"] == STATUS
    assert audited["metadata"]["case_count"] == 40
    assert audited["metadata"]["field_count"] == 160
    assert audited["dimensions"]["field_weight"] == [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert audited["dimensions"]["role_weight"] == [0.0, 0.5, 1.0, 2.0, 4.0]


def test_public_loader_rejects_tampered_weight_aggregate(tmp_path: Path) -> None:
    payload = json.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))
    payload["results"][0]["aggregate"]["evidence_f1"] += 0.1
    copied_case_artifact = tmp_path / payload["case_results_artifact"]["file"]
    shutil.copy2(
        COMMITTED_REPORT.parent / copied_case_artifact.name, copied_case_artifact
    )
    tampered = tmp_path / "housing_weight_sensitivity.json"
    tampered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate mismatch"):
        load_housing_public_weight_sensitivity(tampered)
