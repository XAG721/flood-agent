from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from research.frc_rag.iirc_score_gap_router import (
    FORBIDDEN_RUNTIME_FIELDS,
    blind_features,
    predict_cardinality,
    predict_serialized_forest,
    serialize_forest,
)


def _candidate(index: int) -> dict[str, object]:
    score = 1.0 - index * 0.1
    return {
        "id": f"c{index}",
        "source": f"s{index}",
        "text": f"candidate {index}",
        "scores": {
            "bm25": score,
            "dense": score,
            "hybrid": score,
            "cross_encoder": score,
        },
        "role_scores": {
            "answer": score,
            "attribution": score,
            "condition": score,
            "exception": score,
            "procedure": score,
        },
    }


def _row() -> dict[str, object]:
    return {
        "id": "iirc::synthetic",
        "question": "Which two sources answer the question?",
        "candidates": [_candidate(index) for index in range(6)],
    }


def test_blind_feature_schema_is_deterministic_and_gold_free() -> None:
    row = _row()
    first = blind_features(row)
    second = blind_features(json.loads(json.dumps(row)))
    assert first.shape == (154,)
    assert np.array_equal(first, second)
    assert not (FORBIDDEN_RUNTIME_FIELDS & set(row))


def test_blind_features_reject_gold_fields() -> None:
    row = _row()
    row["answer_type"] = "span"
    with pytest.raises(ValueError, match="forbidden gold field"):
        blind_features(row)


def test_serialized_forest_matches_sklearn_prediction() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
    )
    targets = np.asarray(
        [[0.1, 0.4, 0.2, 0.0], [0.2, 0.5, 0.3, 0.1], [0.6, 0.3, 0.2, 0.1], [0.7, 0.2, 0.1, 0.0]],
        dtype=float,
    )
    model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=87,
        n_jobs=1,
    ).fit(features, targets)
    serialized = serialize_forest(model)
    assert np.allclose(
        predict_serialized_forest(serialized, features), model.predict(features)
    )


def test_predict_cardinality_uses_frozen_biases() -> None:
    row = _row()
    training = np.vstack([blind_features(row), blind_features(row) + 0.01])
    targets = np.asarray([[0.1, 0.4, 0.2, 0.0], [0.1, 0.4, 0.2, 0.0]])
    model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=87,
        n_jobs=1,
    ).fit(training, targets)
    artifact = {
        "forest": serialize_forest(model),
        "model_selection": {"selected_biases": [-0.07, 0.0, 0.0, 0.0]},
    }
    assert predict_cardinality(row, artifact)["cardinality"] == 2


def test_model_runner_defaults_to_ignored_training_inputs() -> None:
    runner = Path("scripts/run_iirc_score_gap_router_model.py").read_text(
        encoding="utf-8"
    )
    assert ".cache/benchmarks/iirc/v86/development/scored_blind.jsonl" in runner
    assert ".cache/benchmarks/iirc/v86/development/sealed_gold.jsonl" in runner
