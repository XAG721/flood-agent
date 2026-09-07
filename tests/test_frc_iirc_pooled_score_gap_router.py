from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from research.frc_rag.iirc_pooled_score_gap_router import predict_cardinality
from research.frc_rag.iirc_score_gap_router import blind_features, serialize_forest


def _row() -> dict[str, object]:
    candidates = []
    for index in range(5):
        score = 1.0 - index * 0.1
        candidates.append(
            {
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
        )
    return {"id": "iirc::synthetic", "question": "Which source?", "candidates": candidates}


def test_pooled_router_uses_the_frozen_blind_feature_schema() -> None:
    row = _row()
    features = blind_features(row)
    training = np.vstack([features, features + 0.01])
    targets = np.asarray([[0.2, 0.5, 0.3, 0.1], [0.2, 0.5, 0.3, 0.1]])
    model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=88,
        n_jobs=1,
    ).fit(training, targets)
    artifact = {
        "forest": serialize_forest(model),
        "model_selection": {"selected_biases": [-0.07, 0.0, 0.0, 0.0]},
    }
    result = predict_cardinality(row, artifact)
    assert result["cardinality"] == 2
    assert set(result["predicted_utilities"]) == {"1", "2", "3", "4"}
