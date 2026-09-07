from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.twowiki_bridge_aware_precision_trim_router import (
    BASE_RETAINED_COUNT,
    BRIDGE_THRESHOLD_GRID,
    BRIDGE_TRIM_ACTION,
    CLASSIFIER_CONFIGURATIONS,
    HASH_DIMENSION_GRID,
    KEEP_ACTION,
    LOGISTIC_L2_GRID,
    POLICY_CONFIGURATIONS,
    TRIMMED_RETAINED_COUNT,
    classifier_configurations,
    load_router_artifact,
    precision_trim,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "docs/progressive_upgrade/"
    "twowiki_bridge_aware_precision_trim_router_model_development_v83.json"
)


def test_v83_finite_search_contract_is_complete_and_unique() -> None:
    configurations = classifier_configurations()
    assert len(configurations) == CLASSIFIER_CONFIGURATIONS == 16
    assert POLICY_CONFIGURATIONS == 112
    assert HASH_DIMENSION_GRID == (64, 128, 256, 512)
    assert LOGISTIC_L2_GRID == (0.5, 2.0, 8.0, 32.0)
    assert BRIDGE_THRESHOLD_GRID == (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert len({json.dumps(row, sort_keys=True) for row in configurations}) == 16


def test_v83_precision_trim_is_order_preserving_and_bounded() -> None:
    selected = [{"id": str(index)} for index in range(BASE_RETAINED_COUNT)]
    trimmed, action = precision_trim(selected, 0.8, threshold=0.3)
    assert action == BRIDGE_TRIM_ACTION
    assert trimmed == selected[:TRIMMED_RETAINED_COUNT]
    retained, action = precision_trim(selected, 0.2, threshold=0.3)
    assert action == KEEP_ACTION
    assert retained == selected
    with pytest.raises(ValueError, match="exceeds the registered cap"):
        precision_trim([*selected, {"id": "extra"}], 0.8, threshold=0.3)


def test_real_v83_model_reproduces_registered_crossfit_diagnostic() -> None:
    artifact, _ = load_router_artifact(MODEL_PATH)
    selected = artifact["selected_configuration"]
    diagnostic = artifact["crossfit_model_selection_diagnostic"]
    assert selected == {
        "classifier_configuration_index": 3,
        "hash_dimension": 64,
        "l2": 32.0,
        "bridge_threshold": 0.3,
        "base_retained_count": 5,
        "trimmed_retained_count": 4,
        "bridge_label": "official training type equals bridge_comparison",
        "runtime_classifier_input": "question text only",
        "official_question_type_answer_or_gold_used_at_runtime": False,
        "policy": (
            "run the frozen v82 selector first; retain its first four items when "
            "the frozen bridge probability reaches the threshold, otherwise keep "
            "all five"
        ),
    }
    assert diagnostic["candidate"]["evidence_macro_f1"] == 0.554477
    assert diagnostic["candidate"]["complete_evidence_recall"] == 0.6225
    assert diagnostic["candidate_minus_v82"] == {
        "point": 0.009652,
        "ci_low": 0.007831,
        "ci_high": 0.011496,
        "resamples": 10000,
        "seed": 20261203,
    }
    assert diagnostic["bridge_balanced_accuracy"] == 0.989583
    assert (
        min(
            row["candidate_minus_v82"]
            for row in diagnostic["per_training_cohort"].values()
        )
        >= 0.005
    )
    assert artifact["prospective_boundary"]["v83_target_training_or_tuning_cases"] == 0


def test_v83_model_loader_rejects_classifier_tampering(tmp_path: Path) -> None:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    artifact["bridge_classifier"]["intercept"] += 1.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="classifier hash mismatch"):
        load_router_artifact(path)
