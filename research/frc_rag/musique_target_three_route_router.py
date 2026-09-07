"""Target-domain three-route router trained on frozen v78 MuSiQue evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.hotpot_graph_router import FEATURE_NAMES, graph_features
from research.frc_rag.hotpot_three_route_router import (
    DEFAULT_ROUTE,
    LEARNED_ROUTES,
    _crossfit_predictions,
    _export_model,
    _fit_model,
    _predict_model,
    route_predictions,
)
from research.frc_rag.musique_graph_router_transfer import _paired_bootstrap
from research.frc_rag.twowiki_question_router import (
    ANCHOR_METHOD,
    SOFT_METHOD,
    load_model_artifact as load_v75_model_artifact,
    select_method as select_v75_method,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    read_jsonl_gzip,
    sha256,
)


EXPERIMENT_ID = "FRC-MUSIQUE-TARGET-TRAINED-THREE-ROUTE-ROUTER-V79"
SCHEMA_VERSION = "frc-musique-target-trained-three-route-router-v79"
CANDIDATE_METHOD = "musique_target_trained_three_route_router_v79"
TRAINING_CASES = 600
FOLD_COUNT = 5
FOLD_SALT = "FRC-MUSIQUE-V79-TARGET-ROUTER-FOLD|"
RANDOM_STATE = 29
ESTIMATOR_GRID = (10, 25, 50)
DEPTH_GRID = (1, 2, 3)
MIN_LEAF_GRID = (15, 30, 60)
LEARNING_RATE_GRID = (0.05, 0.1)
LOSS_GRID = ("huber", "squared_error")
THRESHOLD_GRID = (0.0, 0.0025, 0.005, 0.01)
MODEL_CONFIGURATIONS = (
    len(ESTIMATOR_GRID)
    * len(DEPTH_GRID)
    * len(MIN_LEAF_GRID)
    * len(LEARNING_RATE_GRID)
    * len(LOSS_GRID)
)
ROUTE_POLICY_CONFIGURATIONS = MODEL_CONFIGURATIONS * len(THRESHOLD_GRID)


def model_configurations() -> list[dict[str, Any]]:
    return [
        {
            "estimators": estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_leaf,
            "learning_rate": learning_rate,
            "loss": loss,
            "random_state": RANDOM_STATE,
        }
        for estimators, max_depth, min_leaf, learning_rate, loss in product(
            ESTIMATOR_GRID,
            DEPTH_GRID,
            MIN_LEAF_GRID,
            LEARNING_RATE_GRID,
            LOSS_GRID,
        )
    ]


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % FOLD_COUNT


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("v79 cannot aggregate empty training evidence")
    return {
        "evidence_macro_f1": round(
            sum(float(row["evidence_f1"]) for row in rows) / len(rows), 6
        ),
        "evidence_macro_recall": round(
            sum(float(row["evidence_recall"]) for row in rows) / len(rows), 6
        ),
        "complete_evidence_recall": round(
            sum(float(row["complete_evidence"]) for row in rows) / len(rows), 6
        ),
        "mean_selected_tokens": round(
            sum(float(row["selected_tokens"]) for row in rows) / len(rows), 6
        ),
    }


def _rows_for_routes(
    routes: Sequence[str], metrics: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    return [metrics[str(route)][index] for index, route in enumerate(routes)]


def develop_model(
    scored_path: Path,
    cases_path: Path,
    v75_model_path: Path,
) -> dict[str, Any]:
    scored = read_jsonl(scored_path)
    cases = read_jsonl_gzip(cases_path)
    if len(scored) != TRAINING_CASES or len(cases) != TRAINING_CASES:
        raise ValueError("v79 requires exactly 600 frozen v78 training cases")
    scored_by_id = {str(row["id"]): row for row in scored}
    cases_by_id = {str(row["case_id"]): row for row in cases}
    if (
        len(scored_by_id) != TRAINING_CASES
        or len(cases_by_id) != TRAINING_CASES
        or set(scored_by_id) != set(cases_by_id)
    ):
        raise ValueError("v79 scored rows and training evidence do not align")
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    case_ids = sorted(scored_by_id)
    features = np.vstack(
        [graph_features(scored_by_id[case_id], v75_model) for case_id in case_ids]
    )
    method_names = (DEFAULT_ROUTE, SOFT_METHOD, ANCHOR_METHOD)
    metrics = {
        method: [cases_by_id[case_id]["methods"][method] for case_id in case_ids]
        for method in method_names
    }
    f1 = {
        method: np.asarray(
            [float(row["evidence_f1"]) for row in rows], dtype=np.float64
        )
        for method, rows in metrics.items()
    }
    targets = {
        route: f1[route] - f1[DEFAULT_ROUTE] for route in LEARNED_ROUTES
    }
    folds = np.asarray([_fold(case_id) for case_id in case_ids])
    if set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("v79 deterministic training folds are incomplete")
    configurations = model_configurations()
    if len(configurations) != MODEL_CONFIGURATIONS:
        raise AssertionError("v79 model grid changed")
    search_rows: list[dict[str, Any]] = []
    for index, configuration in enumerate(configurations):
        predictions, _ = _crossfit_predictions(
            features,
            targets,
            folds,
            configuration,
            verify_export=False,
        )
        for threshold in THRESHOLD_GRID:
            routes, _ = route_predictions(
                predictions[SOFT_METHOD],
                predictions[ANCHOR_METHOD],
                threshold=threshold,
            )
            aggregate = _aggregate(_rows_for_routes(routes, metrics))
            search_rows.append(
                {
                    "model_configuration_index": index,
                    "threshold": threshold,
                    **aggregate,
                    "route_counts": dict(sorted(Counter(routes).items())),
                }
            )
    if len(search_rows) != ROUTE_POLICY_CONFIGURATIONS:
        raise AssertionError("v79 route-policy search count changed")
    ranked = sorted(
        search_rows,
        key=lambda row: (
            -float(row["evidence_macro_f1"]),
            -float(row["complete_evidence_recall"]),
            float(row["mean_selected_tokens"]),
            int(row["model_configuration_index"]),
            float(row["threshold"]),
        ),
    )
    selected = ranked[0]
    selected_index = int(selected["model_configuration_index"])
    selected_configuration = configurations[selected_index]
    predictions, fold_models = _crossfit_predictions(
        features, targets, folds, selected_configuration
    )
    routes, winning_gain = route_predictions(
        predictions[SOFT_METHOD],
        predictions[ANCHOR_METHOD],
        threshold=float(selected["threshold"]),
    )
    routed_rows = _rows_for_routes(routes, metrics)
    routed_f1 = np.asarray(
        [float(row["evidence_f1"]) for row in routed_rows], dtype=np.float64
    )
    if round(float(routed_f1.mean()), 6) != selected["evidence_macro_f1"]:
        raise AssertionError("v79 selected result is not reproducible")
    full_models: dict[str, Any] = {}
    for route in LEARNED_ROUTES:
        model = _fit_model(features, targets[route], selected_configuration)
        payload = _export_model(model, features.shape[1], selected_configuration)
        if not np.allclose(
            _predict_model(payload, features), model.predict(features), atol=1e-12
        ):
            raise ValueError("v79 exported full model differs from sklearn")
        full_models[route] = {
            "payload": payload,
            "sha256": canonical_json_sha256(payload),
        }
    anchor_f1 = f1[ANCHOR_METHOD]
    cross_f1 = f1[DEFAULT_ROUTE]
    soft_f1 = f1[SOFT_METHOD]
    oracle_f1 = np.maximum.reduce((cross_f1, soft_f1, anchor_f1))
    v78_f1 = np.asarray(
        [
            float(
                cases_by_id[case_id]["methods"][
                    "mean_calibrated_three_route_router_v78"
                ]["evidence_f1"]
            )
            for case_id in case_ids
        ]
    )
    hop_values = np.asarray([int(cases_by_id[case_id]["hop_count"]) for case_id in case_ids])
    per_hop: dict[str, Any] = {}
    for hop in (2, 3, 4):
        mask = hop_values == hop
        per_hop[str(hop)] = {
            "cases": int(mask.sum()),
            "candidate_evidence_macro_f1": round(float(routed_f1[mask].mean()), 6),
            "anchor_evidence_macro_f1": round(float(anchor_f1[mask].mean()), 6),
            "candidate_minus_anchor": round(
                float((routed_f1[mask] - anchor_f1[mask]).mean()), 6
            ),
            "oracle_evidence_macro_f1": round(float(oracle_f1[mask].mean()), 6),
            "route_counts": dict(sorted(Counter(routes[mask]).items())),
        }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training": {
            "dataset": "MuSiQue",
            "source_experiment": "FRC-MUSIQUE-MEAN-CALIBRATED-THREE-ROUTE-TRANSFER-V78",
            "cases": len(case_ids),
            "case_ids_sha256": canonical_json_sha256(case_ids),
            "scored_path": scored_path.as_posix(),
            "scored_sha256": sha256(scored_path),
            "cases_path": cases_path.as_posix(),
            "cases_sha256": sha256(cases_path),
            "eligible_for_v79_target_stages": False,
            "gold_evidence_metrics_used_for_model_selection": True,
        },
        "search": {
            "model_configurations": MODEL_CONFIGURATIONS,
            "thresholds_per_model": len(THRESHOLD_GRID),
            "route_policy_configurations": ROUTE_POLICY_CONFIGURATIONS,
            "configuration_space": {
                "estimators": list(ESTIMATOR_GRID),
                "max_depth": list(DEPTH_GRID),
                "min_samples_leaf": list(MIN_LEAF_GRID),
                "learning_rate": list(LEARNING_RATE_GRID),
                "loss": list(LOSS_GRID),
                "threshold": list(THRESHOLD_GRID),
                "random_state": RANDOM_STATE,
            },
            "all_configuration_results_sha256": canonical_json_sha256(search_rows),
            "top_10": ranked[:10],
            "all_model_and_threshold_selection_used_only_v78_training_cases": True,
            "oof_score_is_model_selection_evidence_not_unbiased_confirmation": True,
        },
        "selected_configuration": {
            **selected_configuration,
            "model_configuration_index": selected_index,
            "threshold": float(selected["threshold"]),
            "features": "71 gold-free graph, score and selection-overlap features",
            "feature_names": list(FEATURE_NAMES),
            "targets": {
                SOFT_METHOD: "soft_f1_minus_cross_encoder_f1",
                ANCHOR_METHOD: "anchor_f1_minus_cross_encoder_f1",
            },
            "default_route": DEFAULT_ROUTE,
            "learned_routes": list(LEARNED_ROUTES),
            "official_hop_answer_decomposition_or_gold_used_at_runtime": False,
        },
        "crossfit_model_selection_diagnostic": {
            "folds": FOLD_COUNT,
            "fold_models": fold_models,
            "candidate": _aggregate(routed_rows),
            "route_counts": dict(sorted(Counter(routes).items())),
            "mean_winning_predicted_gain": round(float(winning_gain.mean()), 6),
            "controls": {method: _aggregate(rows) for method, rows in metrics.items()},
            "v78_candidate": {
                "evidence_macro_f1": round(float(v78_f1.mean()), 6)
            },
            "oracle_three_route_evidence_macro_f1": round(float(oracle_f1.mean()), 6),
            "candidate_minus_anchor": _paired_bootstrap(
                routed_f1, anchor_f1, seed=20261131
            ),
            "candidate_minus_v78": _paired_bootstrap(
                routed_f1, v78_f1, seed=20261132
            ),
            "per_hop": per_hop,
        },
        "models": full_models,
        "models_sha256": canonical_json_sha256(
            {route: value["sha256"] for route, value in full_models.items()}
        ),
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "prospective_boundary": {
            "all_v78_training_ids_excluded_from_v79_target_stages": True,
            "v79_target_ids_rows_features_scores_or_metrics_seen": False,
            "v79_target_training_or_tuning_cases": 0,
            "v79_can_provide_same_dataset_case_disjoint_confirmation_if_gates_pass": True,
        },
    }
    return artifact


def load_router_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v79 router artifact experiment id")
    selected = artifact.get("selected_configuration", {})
    if selected.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("v79 router feature contract drifted")
    if selected.get("default_route") != DEFAULT_ROUTE or tuple(
        selected.get("learned_routes", ())
    ) != LEARNED_ROUTES:
        raise ValueError("v79 router route contract drifted")
    if float(selected.get("threshold", -1.0)) not in THRESHOLD_GRID:
        raise ValueError("v79 router threshold is outside the registered grid")
    models = artifact.get("models", {})
    for route in LEARNED_ROUTES:
        contract = models.get(route, {})
        payload = contract.get("payload")
        if not isinstance(payload, dict) or canonical_json_sha256(payload) != contract.get(
            "sha256"
        ):
            raise ValueError(f"v79 router model payload hash mismatch: {route}")
        if payload.get("feature_count") != len(FEATURE_NAMES):
            raise ValueError("v79 router feature dimension drifted")
    expected_hash = canonical_json_sha256(
        {route: models[route]["sha256"] for route in LEARNED_ROUTES}
    )
    if expected_hash != artifact.get("models_sha256"):
        raise ValueError("v79 router model collection hash mismatch")
    return artifact


def predict_route_gains(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
) -> dict[str, float]:
    features = graph_features(row, v75_model)
    return {
        route: float(
            _predict_model(router_artifact["models"][route]["payload"], features)[0]
        )
        for route in LEARNED_ROUTES
    }


def select_method(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gains = predict_route_gains(row, router_artifact, v75_model)
    routes, winning_gain = route_predictions(
        np.asarray([gains[SOFT_METHOD]]),
        np.asarray([gains[ANCHOR_METHOD]]),
        threshold=float(router_artifact["selected_configuration"]["threshold"]),
    )
    route = str(routes[0])
    selected, _ = select_v75_method(row, route, v75_model)
    return selected, {
        "route": route,
        "predicted_soft_gain": gains[SOFT_METHOD],
        "predicted_anchor_gain": gains[ANCHOR_METHOD],
        "predicted_winning_gain": float(winning_gain[0]),
    }


__all__ = [
    "CANDIDATE_METHOD",
    "EXPERIMENT_ID",
    "MODEL_CONFIGURATIONS",
    "ROUTE_POLICY_CONFIGURATIONS",
    "SCHEMA_VERSION",
    "THRESHOLD_GRID",
    "develop_model",
    "load_router_artifact",
    "model_configurations",
    "predict_route_gains",
    "select_method",
]
