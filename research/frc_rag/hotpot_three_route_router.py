"""History-only three-route graph router developed for prospective v78 transfer."""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from research.frc_rag.hotpot_graph_router import (
    FEATURE_NAMES,
    FOLD_COUNT,
    _history_fold,
    _paired_bootstrap,
    _selected_metrics,
    blind_history_row,
    graph_features,
)
from research.frc_rag.twowiki_question_router import (
    ANCHOR_METHOD,
    CANDIDATE_METHOD as V75_METHOD,
    LEXICAL_METHOD,
    SOFT_METHOD,
    load_model_artifact as load_v75_model_artifact,
    select_method as select_v75_method,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
)


EXPERIMENT_ID = "FRC-HOTPOT-HISTORY-THREE-ROUTE-GRAPH-ROUTER-V78"
SCHEMA_VERSION = "frc-hotpot-history-three-route-graph-router-v78"
CANDIDATE_METHOD = "history_three_route_graph_router_v78"
DEFAULT_ROUTE = "cross_encoder_topk"
LEARNED_ROUTES = (SOFT_METHOD, ANCHOR_METHOD)
ROUTES = (DEFAULT_ROUTE, *LEARNED_ROUTES)
HISTORY_CASES = 1000
RANDOM_STATE = 19
ESTIMATOR_GRID = (10, 25, 50)
DEPTH_GRID = (1, 2, 3)
MIN_LEAF_GRID = (20, 40, 80)
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
    """Return the complete deterministic finite model grid."""
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


def _fit_model(
    features: np.ndarray, targets: np.ndarray, configuration: dict[str, Any]
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=int(configuration["estimators"]),
        max_depth=int(configuration["max_depth"]),
        min_samples_leaf=int(configuration["min_samples_leaf"]),
        learning_rate=float(configuration["learning_rate"]),
        loss=str(configuration["loss"]),
        random_state=int(configuration["random_state"]),
    )
    model.fit(features, targets)
    return model


def _export_model(
    model: GradientBoostingRegressor,
    feature_count: int,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    init_prediction = float(model.init_.predict(np.zeros((1, feature_count)))[0])

    def export_tree(tree: Any) -> dict[str, Any]:
        structure = tree.tree_
        return {
            "children_left": [int(value) for value in structure.children_left],
            "children_right": [int(value) for value in structure.children_right],
            "features": [int(value) for value in structure.feature],
            "thresholds": [float(value) for value in structure.threshold],
            "values": [float(value[0][0]) for value in structure.value],
        }

    payload = {
        "model_type": "gradient_boosting_regressor_export",
        "sklearn_version": "1.7.2",
        "feature_count": feature_count,
        **configuration,
        "init_prediction": init_prediction,
        "trees": [export_tree(estimator[0]) for estimator in model.estimators_],
    }
    if len(payload["trees"]) != int(configuration["estimators"]):
        raise ValueError("v78 exported model has the wrong tree count")
    return payload


def _predict_model(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != int(payload["feature_count"]):
        raise ValueError("v78 model feature dimension mismatch")
    predictions = np.full(
        len(matrix), float(payload["init_prediction"]), dtype=np.float64
    )
    for tree in payload["trees"]:
        tree_predictions = np.empty(len(matrix), dtype=np.float64)
        for row_index, row in enumerate(matrix):
            node = 0
            while int(tree["children_left"][node]) != -1:
                feature = int(tree["features"][node])
                if row[feature] <= float(tree["thresholds"][node]):
                    node = int(tree["children_left"][node])
                else:
                    node = int(tree["children_right"][node])
            tree_predictions[row_index] = float(tree["values"][node])
        predictions += float(payload["learning_rate"]) * tree_predictions
    return predictions


def route_predictions(
    soft_gain: np.ndarray,
    anchor_gain: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the highest positive predicted gain, otherwise keep cross-encoder."""
    soft = np.asarray(soft_gain, dtype=np.float64)
    anchor = np.asarray(anchor_gain, dtype=np.float64)
    if soft.shape != anchor.shape or soft.ndim != 1:
        raise ValueError("v78 route predictions require aligned one-dimensional gains")
    if not np.isfinite(soft).all() or not np.isfinite(anchor).all():
        raise ValueError("v78 route predictions contain a non-finite gain")
    gains = np.column_stack((soft, anchor))
    best = gains.argmax(axis=1)
    best_gain = gains[np.arange(len(gains)), best]
    routes = np.full(len(gains), DEFAULT_ROUTE, dtype=object)
    eligible = best_gain > float(threshold)
    route_names = np.asarray(LEARNED_ROUTES, dtype=object)
    routes[eligible] = route_names[best[eligible]]
    return routes.astype(str), best_gain


def _aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("v78 cannot aggregate an empty metric collection")
    return {
        "evidence_macro_f1": round(
            sum(row["evidence_f1"] for row in rows) / len(rows), 6
        ),
        "evidence_macro_recall": round(
            sum(row["evidence_recall"] for row in rows) / len(rows), 6
        ),
        "complete_evidence_recall": round(
            sum(row["complete_evidence"] for row in rows) / len(rows), 6
        ),
        "mean_selected_tokens": round(
            sum(row["selected_tokens"] for row in rows) / len(rows), 6
        ),
    }


def _rows_for_routes(
    routes: Sequence[str], metrics: dict[str, list[dict[str, float]]]
) -> list[dict[str, float]]:
    if len(routes) != len(next(iter(metrics.values()))):
        raise ValueError("v78 route and metric lengths differ")
    return [metrics[str(route)][index] for index, route in enumerate(routes)]


def _crossfit_predictions(
    feature_matrix: np.ndarray,
    targets: dict[str, np.ndarray],
    folds: np.ndarray,
    configuration: dict[str, Any],
    *,
    verify_export: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions = {
        route: np.zeros(len(feature_matrix), dtype=np.float64)
        for route in LEARNED_ROUTES
    }
    fold_models: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        training = folds != fold
        held_out = ~training
        local: dict[str, Any] = {
            "fold": fold,
            "training_cases": int(training.sum()),
            "held_out_cases": int(held_out.sum()),
            "models": {},
        }
        for route in LEARNED_ROUTES:
            model = _fit_model(
                feature_matrix[training], targets[route][training], configuration
            )
            native = model.predict(feature_matrix[held_out])
            predictions[route][held_out] = native
            if verify_export:
                payload = _export_model(model, feature_matrix.shape[1], configuration)
                exported = _predict_model(payload, feature_matrix[held_out])
                if not np.allclose(exported, native, atol=1e-12):
                    difference = float(np.max(np.abs(exported - native)))
                    raise ValueError(
                        "v78 exported fold model differs from sklearn: "
                        f"route={route}, fold={fold}, "
                        f"max_abs_diff={difference:.17g}, "
                        f"configuration={configuration}"
                    )
                local["models"][route] = canonical_json_sha256(payload)
        fold_models.append(local)
    return predictions, fold_models


def _v76_crossfit_routes(
    feature_matrix: np.ndarray, soft_targets: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    configuration = {
        "estimators": 25,
        "max_depth": 2,
        "min_samples_leaf": 40,
        "learning_rate": 0.1,
        "loss": "huber",
        "random_state": 7,
    }
    predictions = np.zeros(len(feature_matrix), dtype=np.float64)
    for fold in range(FOLD_COUNT):
        training = folds != fold
        held_out = ~training
        model = _fit_model(feature_matrix[training], soft_targets[training], configuration)
        predictions[held_out] = model.predict(feature_matrix[held_out])
    return np.where(predictions > 0.0, SOFT_METHOD, DEFAULT_ROUTE).astype(str)


def develop_model(
    history_path: Path,
    v75_model_path: Path,
    *,
    v76_model_path: Path | None = None,
) -> dict[str, Any]:
    """Select and fit a two-gain router using only permanently excluded history."""
    history = read_jsonl(history_path)
    if len(history) != HISTORY_CASES or len({str(row["id"]) for row in history}) != HISTORY_CASES:
        raise ValueError("v78 history requires exactly 1000 unique HotpotQA cases")
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    case_ids: list[str] = []
    question_types: list[str] = []
    features: list[np.ndarray] = []
    method_names = (DEFAULT_ROUTE, SOFT_METHOD, ANCHOR_METHOD, LEXICAL_METHOD, V75_METHOD)
    metrics: dict[str, list[dict[str, float]]] = {name: [] for name in method_names}
    for original in history:
        if str(original.get("dataset", "")).lower() != "hotpotqa":
            raise ValueError("v78 history contains a non-HotpotQA row")
        blind = blind_history_row(original)
        case_ids.append(str(original["id"]))
        question_types.append(str(original["question_type"]))
        features.append(graph_features(blind, v75_model))
        for method in method_names:
            metrics[method].append(
                _selected_metrics(
                    blind,
                    original["gold_evidence_ids"],
                    method,
                    v75_model,
                )
            )
    feature_matrix = np.vstack(features)
    f1 = {
        method: np.asarray([row["evidence_f1"] for row in rows], dtype=np.float64)
        for method, rows in metrics.items()
    }
    targets = {
        route: f1[route] - f1[DEFAULT_ROUTE] for route in LEARNED_ROUTES
    }
    folds = np.asarray([_history_fold(case_id) for case_id in case_ids])
    configurations = model_configurations()
    if len(configurations) != MODEL_CONFIGURATIONS:
        raise AssertionError("v78 model configuration grid changed")

    search_rows: list[dict[str, Any]] = []
    cached_predictions: dict[int, dict[str, np.ndarray]] = {}
    for index, configuration in enumerate(configurations):
        predictions, _ = _crossfit_predictions(
            feature_matrix,
            targets,
            folds,
            configuration,
            verify_export=False,
        )
        cached_predictions[index] = predictions
        for threshold in THRESHOLD_GRID:
            routes, _ = route_predictions(
                predictions[SOFT_METHOD],
                predictions[ANCHOR_METHOD],
                threshold=threshold,
            )
            routed_rows = _rows_for_routes(routes, metrics)
            aggregate = _aggregate(routed_rows)
            search_rows.append(
                {
                    "model_configuration_index": index,
                    "threshold": threshold,
                    **aggregate,
                    "route_counts": dict(sorted(Counter(routes).items())),
                }
            )
    if len(search_rows) != ROUTE_POLICY_CONFIGURATIONS:
        raise AssertionError("v78 route-policy search count changed")
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
    selected_predictions, fold_models = _crossfit_predictions(
        feature_matrix, targets, folds, selected_configuration
    )
    routes, predicted_gain = route_predictions(
        selected_predictions[SOFT_METHOD],
        selected_predictions[ANCHOR_METHOD],
        threshold=float(selected["threshold"]),
    )
    routed_rows = _rows_for_routes(routes, metrics)
    routed_f1 = np.asarray(
        [row["evidence_f1"] for row in routed_rows], dtype=np.float64
    )
    if round(float(routed_f1.mean()), 6) != selected["evidence_macro_f1"]:
        raise AssertionError("v78 selected search result is not reproducible")

    v76_routes = _v76_crossfit_routes(feature_matrix, targets[SOFT_METHOD], folds)
    v76_rows = _rows_for_routes(v76_routes, metrics)
    v76_f1 = np.asarray([row["evidence_f1"] for row in v76_rows], dtype=np.float64)
    if round(float(v76_f1.mean()), 6) != 0.562448:
        raise ValueError("v78 could not reproduce the frozen v76 history score")
    if v76_model_path is not None:
        registered_v76 = json.loads(v76_model_path.read_text(encoding="utf-8"))
        if registered_v76["crossfit"]["candidate"]["evidence_macro_f1"] != 0.562448:
            raise ValueError("v78 registered v76 artifact changed")

    full_models: dict[str, Any] = {}
    for route in LEARNED_ROUTES:
        model = _fit_model(feature_matrix, targets[route], selected_configuration)
        payload = _export_model(model, feature_matrix.shape[1], selected_configuration)
        if not np.allclose(
            _predict_model(payload, feature_matrix),
            model.predict(feature_matrix),
            atol=1e-12,
        ):
            raise ValueError("v78 exported full model differs from sklearn")
        full_models[route] = {
            "payload": payload,
            "sha256": canonical_json_sha256(payload),
        }

    per_type: dict[str, Any] = {}
    for question_type in sorted(set(question_types)):
        mask = np.asarray([value == question_type for value in question_types])
        best_control = np.maximum.reduce(
            [f1[DEFAULT_ROUTE][mask], f1[SOFT_METHOD][mask], f1[ANCHOR_METHOD][mask]]
        )
        per_type[question_type] = {
            "cases": int(mask.sum()),
            "candidate_evidence_macro_f1": round(float(routed_f1[mask].mean()), 6),
            "v76_candidate_evidence_macro_f1": round(float(v76_f1[mask].mean()), 6),
            "candidate_minus_v76": round(float((routed_f1[mask] - v76_f1[mask]).mean()), 6),
            "oracle_headroom": round(float((best_control - routed_f1[mask]).mean()), 6),
            "route_counts": dict(sorted(Counter(routes[mask]).items())),
        }

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "history": {
            "cases": len(history),
            "unique_ids": len(set(case_ids)),
            "case_ids_sha256": canonical_json_sha256(case_ids),
            "question_type_counts": dict(sorted(Counter(question_types).items())),
            "source_sha256": sha256(history_path),
            "eligible_for_v78_target_stages": False,
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
            "all_exploration_used_only_permanently_excluded_history": True,
            "v77_target_rows_features_scores_gold_or_metrics_used_for_fit_or_selection": False,
        },
        "selected_configuration": {
            **selected_configuration,
            "model_configuration_index": selected_index,
            "threshold": float(selected["threshold"]),
            "features": "the unchanged 71 gold-free v76 graph/score/overlap features",
            "feature_names": list(FEATURE_NAMES),
            "targets": {
                SOFT_METHOD: "soft_f1_minus_cross_encoder_f1",
                ANCHOR_METHOD: "anchor_f1_minus_cross_encoder_f1",
            },
            "default_route": DEFAULT_ROUTE,
            "learned_routes": list(LEARNED_ROUTES),
            "official_type_answer_or_gold_used_at_runtime": False,
        },
        "crossfit": {
            "folds": FOLD_COUNT,
            "fold_models": fold_models,
            "candidate": _aggregate(routed_rows),
            "route_counts": dict(sorted(Counter(routes).items())),
            "mean_winning_predicted_gain": round(float(predicted_gain.mean()), 6),
            "oof_prediction_mean_by_learned_route": {
                route: round(float(selected_predictions[route].mean()), 12)
                for route in LEARNED_ROUTES
            },
            "history_target_mean_by_learned_route": {
                route: round(float(targets[route].mean()), 12)
                for route in LEARNED_ROUTES
            },
            "controls": {name: _aggregate(rows) for name, rows in metrics.items()},
            "v76_candidate": _aggregate(v76_rows),
            "candidate_minus_v76": _paired_bootstrap(
                routed_f1, v76_f1, seed=20261111
            ),
            "candidate_minus_cross_encoder": _paired_bootstrap(
                routed_f1, f1[DEFAULT_ROUTE], seed=20261112
            ),
            "per_question_type": per_type,
        },
        "models": full_models,
        "models_sha256": canonical_json_sha256(
            {route: value["sha256"] for route, value in full_models.items()}
        ),
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "prospective_boundary": {
            "all_1000_history_ids_excluded_from_v78_target_stages": True,
            "v77_used_only_to_identify_missing_route_family_and_oracle_headroom": True,
            "v77_per_case_features_scores_gold_or_metrics_used_for_training": False,
            "v78_target_ids_rows_features_scores_or_metrics_seen": False,
        },
    }
    return artifact


def load_router_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v78 router artifact experiment id")
    selected = artifact.get("selected_configuration", {})
    if selected.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("v78 router feature contract drifted")
    if selected.get("default_route") != DEFAULT_ROUTE or tuple(
        selected.get("learned_routes", ())
    ) != LEARNED_ROUTES:
        raise ValueError("v78 router route contract drifted")
    if float(selected.get("threshold", -1.0)) not in THRESHOLD_GRID:
        raise ValueError("v78 router threshold is outside the registered grid")
    models = artifact.get("models", {})
    for route in LEARNED_ROUTES:
        contract = models.get(route, {})
        payload = contract.get("payload")
        if not isinstance(payload, dict) or canonical_json_sha256(payload) != contract.get(
            "sha256"
        ):
            raise ValueError(f"v78 router model payload hash mismatch: {route}")
        if payload.get("feature_count") != len(FEATURE_NAMES):
            raise ValueError("v78 router model feature dimension drifted")
    expected_models_hash = canonical_json_sha256(
        {route: models[route]["sha256"] for route in LEARNED_ROUTES}
    )
    if expected_models_hash != artifact.get("models_sha256"):
        raise ValueError("v78 router model collection hash mismatch")
    return artifact


def select_method(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predicted = predict_route_gains(row, router_artifact, v75_model)
    routes, best_gain = route_predictions(
        np.asarray([predicted[SOFT_METHOD]]),
        np.asarray([predicted[ANCHOR_METHOD]]),
        threshold=float(router_artifact["selected_configuration"]["threshold"]),
    )
    route = str(routes[0])
    selected, _ = select_v75_method(row, route, v75_model)
    return selected, {
        "route": route,
        "predicted_soft_gain": predicted[SOFT_METHOD],
        "predicted_anchor_gain": predicted[ANCHOR_METHOD],
        "predicted_winning_gain": float(best_gain[0]),
    }


def predict_route_gains(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
) -> dict[str, float]:
    """Predict both history-relative gains without applying a route policy."""
    features = graph_features(row, v75_model)
    return {
        route: float(
            _predict_model(router_artifact["models"][route]["payload"], features)[0]
        )
        for route in LEARNED_ROUTES
    }


__all__ = [
    "CANDIDATE_METHOD",
    "DEFAULT_ROUTE",
    "EXPERIMENT_ID",
    "LEARNED_ROUTES",
    "MODEL_CONFIGURATIONS",
    "ROUTES",
    "ROUTE_POLICY_CONFIGURATIONS",
    "SCHEMA_VERSION",
    "THRESHOLD_GRID",
    "develop_model",
    "load_router_artifact",
    "model_configurations",
    "predict_route_gains",
    "route_predictions",
    "select_method",
]
