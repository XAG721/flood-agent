from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.hotpot_question_type_cardinality_router import (
    BASE_RETAINED_COUNT,
    BRIDGE_ACTION,
    BRIDGE_RETAINED_COUNT,
    CLASSIFIER_CONFIGURATIONS,
    COMPARISON_ACTION,
    COMPARISON_RETAINED_COUNT,
    COMPARISON_THRESHOLD_GRID,
    HASH_DIMENSION_GRID,
    LOGISTIC_L2_GRID,
    POLICY_CONFIGURATIONS,
    cardinality_control,
    classifier_configurations,
    load_router_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "docs/progressive_upgrade/"
    "hotpot_question_type_cardinality_router_model_development_v84.json"
)


def test_v84_finite_search_contract_is_complete_and_unique() -> None:
    configurations = classifier_configurations()
    assert len(configurations) == CLASSIFIER_CONFIGURATIONS == 16
    assert POLICY_CONFIGURATIONS == 112
    assert HASH_DIMENSION_GRID == (64, 128, 256, 512)
    assert LOGISTIC_L2_GRID == (0.5, 2.0, 8.0, 32.0)
    assert COMPARISON_THRESHOLD_GRID == (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert len({json.dumps(row, sort_keys=True) for row in configurations}) == 16


def test_v84_cardinality_control_preserves_order_and_bounds() -> None:
    selected = [{"id": str(index)} for index in range(BASE_RETAINED_COUNT)]
    retained, action = cardinality_control(selected, 0.8, threshold=0.3)
    assert action == COMPARISON_ACTION
    assert retained == selected[:COMPARISON_RETAINED_COUNT]
    retained, action = cardinality_control(selected, 0.2, threshold=0.3)
    assert action == BRIDGE_ACTION
    assert retained == selected[:BRIDGE_RETAINED_COUNT]
    with pytest.raises(ValueError, match="exceeds the registered cap"):
        cardinality_control([*selected, {"id": "extra"}], 0.8, threshold=0.3)


def test_real_v84_model_reproduces_registered_crossfit_diagnostic() -> None:
    artifact, _ = load_router_artifact(MODEL_PATH)
    selected = artifact["selected_configuration"]
    diagnostic = artifact["crossfit_model_selection_diagnostic"]
    assert selected == {
        "classifier_configuration_index": 15,
        "hash_dimension": 512,
        "l2": 32.0,
        "comparison_threshold": 0.3,
        "base_retained_count": 5,
        "comparison_retained_count": 3,
        "bridge_retained_count": 4,
        "comparison_label": "official exposed training type equals comparison",
        "runtime_classifier_input": "question text only",
        "official_question_type_answer_or_gold_used_at_runtime": False,
        "policy": (
            "run the frozen v76 selector first; retain its first three items for "
            "predicted comparison questions and first four items otherwise"
        ),
    }
    assert diagnostic["candidate"]["evidence_macro_f1"] == 0.657218
    assert diagnostic["candidate"]["complete_evidence_recall"] == 0.645385
    assert diagnostic["candidate_minus_fixed_prefix4"] == {
        "point": 0.027613,
        "ci_low": 0.024012,
        "ci_high": 0.031077,
        "resamples": 10000,
        "seed": 20261206,
    }
    assert diagnostic["comparison_balanced_accuracy"] == 0.911056
    assert min(
        row["candidate_minus_fixed_prefix4"]
        for row in diagnostic["per_training_cohort"].values()
    ) >= 0.005
    assert artifact["prospective_boundary"]["v84_target_training_or_tuning_cases"] == 0


def test_v84_model_loader_rejects_classifier_tampering(tmp_path: Path) -> None:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    artifact["comparison_classifier"]["intercept"] += 1.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="classifier hash mismatch"):
        load_router_artifact(path)
