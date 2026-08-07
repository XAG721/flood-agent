from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.frc_rag.twowiki_cascaded_style_residual_router import (
    CLASSIFIER_CONFIGURATIONS,
    DEFAULT_ACTION,
    DIRECT_CROSS_ACTION,
    DIRECT_THRESHOLD_GRID,
    FLIP_THRESHOLD_GRID,
    HASH_DIMENSION_GRID,
    LOGISTIC_L2_GRID,
    ROUTE_POLICY_CONFIGURATIONS,
    SAFE_FLIP_ACTION,
    classifier_configurations,
    load_router_artifact,
    route_actions,
    routes_for_actions,
)
from research.frc_rag.twowiki_question_router import ANCHOR_METHOD, SOFT_METHOD
from research.frc_rag.twowiki_residual_three_route_router import CROSS_METHOD


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "docs/progressive_upgrade/"
    "twowiki_cascaded_style_residual_router_model_development_v82.json"
)


def test_v82_finite_search_contract_is_complete_and_unique() -> None:
    configurations = classifier_configurations()
    assert len(configurations) == CLASSIFIER_CONFIGURATIONS == 16
    assert ROUTE_POLICY_CONFIGURATIONS == 560
    assert HASH_DIMENSION_GRID == (64, 128, 256, 512)
    assert LOGISTIC_L2_GRID == (0.5, 2.0, 8.0, 32.0)
    assert len(DIRECT_THRESHOLD_GRID) == 5
    assert len(FLIP_THRESHOLD_GRID) == 7
    assert len({json.dumps(row, sort_keys=True) for row in configurations}) == 16


def test_v82_cascade_prioritizes_direct_comparison_then_safe_flip() -> None:
    actions = route_actions(
        np.asarray([0.95, 0.20, 0.20, 0.20]),
        np.asarray([0.90, 0.01, 0.03, 0.01]),
        np.asarray([0.80, 0.02, 0.02, 0.01]),
        direct_threshold=0.8,
        flip_threshold=0.015,
    )
    assert actions.tolist() == [
        DIRECT_CROSS_ACTION,
        SAFE_FLIP_ACTION,
        DEFAULT_ACTION,
        DEFAULT_ACTION,
    ]
    assert routes_for_actions(
        actions, [ANCHOR_METHOD, ANCHOR_METHOD, SOFT_METHOD, SOFT_METHOD]
    ).tolist() == [CROSS_METHOD, SOFT_METHOD, SOFT_METHOD, SOFT_METHOD]
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        route_actions(
            np.asarray([0.2]),
            np.asarray([0.1, 0.2]),
            np.asarray([0.1]),
            direct_threshold=0.8,
            flip_threshold=0.015,
        )


def test_real_v82_model_reproduces_registered_oof_diagnostic() -> None:
    artifact, _ = load_router_artifact(MODEL_PATH)
    selected = artifact["selected_configuration"]
    diagnostic = artifact["crossfit_model_selection_diagnostic"]
    assert selected["hash_dimension"] in HASH_DIMENSION_GRID
    assert selected["l2"] in LOGISTIC_L2_GRID
    assert selected["direct_threshold"] in DIRECT_THRESHOLD_GRID
    assert selected["flip_threshold"] in FLIP_THRESHOLD_GRID
    assert diagnostic["candidate_minus_v75"]["point"] >= 0.005
    assert diagnostic["candidate_minus_v75"]["ci_low"] > 0.0
    assert diagnostic["direct_comparison_balanced_accuracy"] >= 0.9
    assert all(
        row["candidate_minus_v75"] >= -0.005
        for row in diagnostic["per_question_type"].values()
    )
    assert all(
        row["candidate_minus_v75"] >= 0.0
        for row in diagnostic["per_training_cohort"].values()
    )
    assert artifact["prospective_boundary"]["v82_target_training_or_tuning_cases"] == 0


def test_v82_model_loader_rejects_classifier_tampering(tmp_path: Path) -> None:
    artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    artifact["direct_comparison_classifier"]["intercept"] += 1.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="classifier hash mismatch"):
        load_router_artifact(path)
