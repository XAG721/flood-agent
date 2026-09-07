"""Anchor-default three-route router developed on frozen v78+v79 evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from research.frc_rag.hotpot_graph_router import FEATURE_NAMES, graph_features
from research.frc_rag.hotpot_three_route_router import (
    _export_model,
    _predict_model,
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


EXPERIMENT_ID = "FRC-MUSIQUE-ANCHOR-DEFAULT-THREE-ROUTE-ROUTER-V80"
SCHEMA_VERSION = "frc-musique-anchor-default-three-route-router-v80"
CANDIDATE_METHOD = "musique_anchor_default_three_route_router_v80"
DEFAULT_ROUTE = ANCHOR_METHOD
CROSS_METHOD = "cross_encoder_topk"
LEARNED_ROUTES = (CROSS_METHOD, SOFT_METHOD)
ROUTES = (*LEARNED_ROUTES, DEFAULT_ROUTE)
TRAINING_CASES = 1070
FOLD_COUNT = 5
FOLD_SALT = "FRC-MUSIQUE-V80-ANCHOR-DEFAULT-FOLD|"
RANDOM_STATE = 31
ESTIMATOR_GRID = (25, 50, 100)
DEPTH_GRID = (1, 2)
MIN_LEAF_GRID = (25, 50, 100)
LEARNING_RATE_GRID = (0.05, 0.1)
LOSS_GRID = ("huber", "squared_error")
WEIGHT_MODE_GRID = ("uniform", "nonzero_x4", "magnitude_x20")
THRESHOLD_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
MODEL_CONFIGURATIONS = (
    len(ESTIMATOR_GRID)
    * len(DEPTH_GRID)
    * len(MIN_LEAF_GRID)
    * len(LEARNING_RATE_GRID)
    * len(LOSS_GRID)
    * len(WEIGHT_MODE_GRID)
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
            "weight_mode": weight_mode,
            "random_state": RANDOM_STATE,
        }
        for estimators, max_depth, min_leaf, learning_rate, loss, weight_mode in product(
            ESTIMATOR_GRID,
            DEPTH_GRID,
            MIN_LEAF_GRID,
            LEARNING_RATE_GRID,
            LOSS_GRID,
            WEIGHT_MODE_GRID,
        )
    ]


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % FOLD_COUNT


def _sample_weights(targets: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    if mode == "uniform":
        weights = np.ones(len(values), dtype=np.float64)
    elif mode == "nonzero_x4":
        weights = np.where(np.abs(values) > 1e-12, 4.0, 1.0)
    elif mode == "magnitude_x20":
        weights = 1.0 + 20.0 * np.abs(values)
    else:
        raise ValueError(f"unsupported v80 sample weight mode: {mode}")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("v80 sample weights are invalid")
    return weights


def _fit_model(
    features: np.ndarray,
    targets: np.ndarray,
    configuration: dict[str, Any],
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=int(configuration["estimators"]),
        max_depth=int(configuration["max_depth"]),
        min_samples_leaf=int(configuration["min_samples_leaf"]),
        learning_rate=float(configuration["learning_rate"]),
        loss=str(configuration["loss"]),
        random_state=int(configuration["random_state"]),
    )
    model.fit(
        features,
        targets,
        sample_weight=_sample_weights(targets, str(configuration["weight_mode"])),
    )
    return model


def route_predictions(
    cross_gain: np.ndarray,
    soft_gain: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    cross = np.asarray(cross_gain, dtype=np.float64)
    soft = np.asarray(soft_gain, dtype=np.float64)
    if cross.shape != soft.shape or cross.ndim != 1:
        raise ValueError("v80 route gains must be aligned one-dimensional arrays")
    if not np.isfinite(cross).all() or not np.isfinite(soft).all():
        raise ValueError("v80 route gains contain a non-finite value")
    gains = np.column_stack((cross, soft))
    best = gains.argmax(axis=1)
    best_gain = gains[np.arange(len(gains)), best]
    routes = np.full(len(gains), DEFAULT_ROUTE, dtype=object)
    eligible = best_gain > float(threshold)
    route_names = np.asarray(LEARNED_ROUTES, dtype=object)
    routes[eligible] = route_names[best[eligible]]
    return routes.astype(str), best_gain


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("v80 cannot aggregate empty evidence")
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


def _crossfit_predictions(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    folds: np.ndarray,
    configuration: dict[str, Any],
    *,
    verify_export: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions = {
        route: np.zeros(len(features), dtype=np.float64) for route in LEARNED_ROUTES
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
                features[training], targets[route][training], configuration
            )
            native = model.predict(features[held_out])
            predictions[route][held_out] = native
            if verify_export:
                payload = _export_model(model, features.shape[1], configuration)
                exported = _predict_model(payload, features[held_out])
                if not np.allclose(exported, native, atol=1e-12):
                    raise ValueError(
                        f"v80 portable fold model differs from sklearn: {route}/{fold}"
                    )
                local["models"][route] = {
                    "payload": payload,
                    "sha256": canonical_json_sha256(payload),
                }
        fold_models.append(local)
    return predictions, fold_models


def develop_model(
    scored_paths: Sequence[Path],
    cases_paths: Sequence[Path],
    v75_model_path: Path,
) -> dict[str, Any]:
    if len(scored_paths) != 2 or len(cases_paths) != 2:
        raise ValueError("v80 requires exactly the frozen v78 and v79 sources")
    scored_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    cohort_by_id: dict[str, str] = {}
    source_contracts: list[dict[str, Any]] = []
    for cohort, scored_path, cases_path in zip(
        ("v78", "v79"), scored_paths, cases_paths, strict=True
    ):
        local_scored = read_jsonl(scored_path)
        local_cases = read_jsonl_gzip(cases_path)
        if len(local_scored) != len(local_cases):
            raise ValueError(f"v80 {cohort} scored and case evidence differ")
        scored_rows.extend(local_scored)
        case_rows.extend(local_cases)
        for row in local_scored:
            case_id = str(row["id"])
            if case_id in cohort_by_id:
                raise ValueError("v80 training cohorts overlap")
            cohort_by_id[case_id] = cohort
        source_contracts.append(
            {
                "cohort": cohort,
                "cases": len(local_scored),
                "scored_path": scored_path.as_posix(),
                "scored_sha256": sha256(scored_path),
                "cases_path": cases_path.as_posix(),
                "cases_sha256": sha256(cases_path),
            }
        )
    if len(scored_rows) != TRAINING_CASES or len(case_rows) != TRAINING_CASES:
        raise ValueError("v80 requires exactly 1,070 frozen training cases")
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    cases_by_id = {str(row["case_id"]): row for row in case_rows}
    if (
        len(scored_by_id) != TRAINING_CASES
        or len(cases_by_id) != TRAINING_CASES
        or set(scored_by_id) != set(cases_by_id)
    ):
        raise ValueError("v80 training evidence does not align")
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    case_ids = sorted(scored_by_id)
    features = np.vstack(
        [graph_features(scored_by_id[case_id], v75_model) for case_id in case_ids]
    )
    metrics = {
        method: [cases_by_id[case_id]["methods"][method] for case_id in case_ids]
        for method in ROUTES
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
        raise ValueError("v80 deterministic folds are incomplete")
    configurations = model_configurations()
    if len(configurations) != MODEL_CONFIGURATIONS:
        raise AssertionError("v80 model grid changed")
    hop_values = np.asarray(
        [int(cases_by_id[case_id]["hop_count"]) for case_id in case_ids]
    )
    search_rows: list[dict[str, Any]] = []
    for index, configuration in enumerate(configurations):
        predictions, _ = _crossfit_predictions(
            features, targets, folds, configuration
        )
        for threshold in THRESHOLD_GRID:
            routes, _ = route_predictions(
                predictions[CROSS_METHOD],
                predictions[SOFT_METHOD],
                threshold=threshold,
            )
            routed_rows = _rows_for_routes(routes, metrics)
            aggregate = _aggregate(routed_rows)
            per_hop_delta: dict[str, float] = {}
            for hop in (2, 3, 4):
                mask = hop_values == hop
                routed_f1 = np.asarray(
                    [float(row["evidence_f1"]) for row in routed_rows]
                )
                per_hop_delta[str(hop)] = round(
                    float((routed_f1[mask] - f1[DEFAULT_ROUTE][mask]).mean()), 6
                )
            counts = Counter(routes)
            search_rows.append(
                {
                    "model_configuration_index": index,
                    "threshold": threshold,
                    **aggregate,
                    "candidate_minus_anchor": round(
                        aggregate["evidence_macro_f1"]
                        - _aggregate(metrics[DEFAULT_ROUTE])["evidence_macro_f1"],
                        6,
                    ),
                    "per_hop_delta_vs_anchor": per_hop_delta,
                    "route_counts": dict(sorted(counts.items())),
                }
            )
    if len(search_rows) != ROUTE_POLICY_CONFIGURATIONS:
        raise AssertionError("v80 route-policy search count changed")
    safe_rows = [
        row
        for row in search_rows
        if min(row["per_hop_delta_vs_anchor"].values()) >= -0.005
        and int(row["route_counts"].get(DEFAULT_ROUTE, 0)) < TRAINING_CASES
        and sum(count >= 0.05 * TRAINING_CASES for count in row["route_counts"].values())
        >= 2
    ]
    if not safe_rows:
        raise ValueError("v80 finite search has no OOF-safe policy")
    ranked = sorted(
        safe_rows,
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
        features,
        targets,
        folds,
        selected_configuration,
        verify_export=True,
    )
    routes, winning_gain = route_predictions(
        predictions[CROSS_METHOD],
        predictions[SOFT_METHOD],
        threshold=float(selected["threshold"]),
    )
    routed_rows = _rows_for_routes(routes, metrics)
    routed_f1 = np.asarray(
        [float(row["evidence_f1"]) for row in routed_rows], dtype=np.float64
    )
    if round(float(routed_f1.mean()), 6) != selected["evidence_macro_f1"]:
        raise AssertionError("v80 selected OOF result is not reproducible")
    full_models: dict[str, Any] = {}
    for route in LEARNED_ROUTES:
        model = _fit_model(features, targets[route], selected_configuration)
        payload = _export_model(model, features.shape[1], selected_configuration)
        if not np.allclose(
            _predict_model(payload, features), model.predict(features), atol=1e-12
        ):
            raise ValueError("v80 exported full model differs from sklearn")
        full_models[route] = {
            "payload": payload,
            "sha256": canonical_json_sha256(payload),
        }
    oracle_f1 = np.maximum.reduce([f1[method] for method in ROUTES])
    per_hop: dict[str, Any] = {}
    for hop in (2, 3, 4):
        mask = hop_values == hop
        per_hop[str(hop)] = {
            "cases": int(mask.sum()),
            "candidate_evidence_macro_f1": round(float(routed_f1[mask].mean()), 6),
            "anchor_evidence_macro_f1": round(
                float(f1[DEFAULT_ROUTE][mask].mean()), 6
            ),
            "candidate_minus_anchor": round(
                float((routed_f1[mask] - f1[DEFAULT_ROUTE][mask]).mean()), 6
            ),
            "oracle_evidence_macro_f1": round(float(oracle_f1[mask].mean()), 6),
            "route_counts": dict(sorted(Counter(routes[mask]).items())),
        }
    per_cohort: dict[str, Any] = {}
    cohort_values = np.asarray([cohort_by_id[case_id] for case_id in case_ids])
    for cohort in ("v78", "v79"):
        mask = cohort_values == cohort
        per_cohort[cohort] = {
            "cases": int(mask.sum()),
            "candidate_evidence_macro_f1": round(float(routed_f1[mask].mean()), 6),
            "anchor_evidence_macro_f1": round(
                float(f1[DEFAULT_ROUTE][mask].mean()), 6
            ),
            "candidate_minus_anchor": round(
                float((routed_f1[mask] - f1[DEFAULT_ROUTE][mask]).mean()), 6
            ),
            "route_counts": dict(sorted(Counter(routes[mask]).items())),
        }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training": {
            "dataset": "MuSiQue",
            "cases": len(case_ids),
            "case_ids_sha256": canonical_json_sha256(case_ids),
            "sources": source_contracts,
            "gold_evidence_metrics_used_for_model_selection": True,
            "eligible_for_v80_target_stage": False,
        },
        "search": {
            "model_configurations": MODEL_CONFIGURATIONS,
            "thresholds_per_model": len(THRESHOLD_GRID),
            "route_policy_configurations": ROUTE_POLICY_CONFIGURATIONS,
            "oof_safe_policy_configurations": len(safe_rows),
            "configuration_space": {
                "estimators": list(ESTIMATOR_GRID),
                "max_depth": list(DEPTH_GRID),
                "min_samples_leaf": list(MIN_LEAF_GRID),
                "learning_rate": list(LEARNING_RATE_GRID),
                "loss": list(LOSS_GRID),
                "weight_mode": list(WEIGHT_MODE_GRID),
                "threshold": list(THRESHOLD_GRID),
                "random_state": RANDOM_STATE,
            },
            "all_configuration_results_sha256": canonical_json_sha256(search_rows),
            "top_10_safe": ranked[:10],
            "all_selection_used_only_permanently_excluded_v78_v79_cases": True,
            "oof_score_is_model_selection_evidence_not_unbiased_confirmation": True,
        },
        "selected_configuration": {
            **selected_configuration,
            "model_configuration_index": selected_index,
            "threshold": float(selected["threshold"]),
            "features": "71 gold-free graph, score and selection-overlap features",
            "feature_names": list(FEATURE_NAMES),
            "targets": {
                CROSS_METHOD: "cross_encoder_f1_minus_anchor_f1",
                SOFT_METHOD: "soft_f1_minus_anchor_f1",
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
            "oracle_three_route_evidence_macro_f1": round(float(oracle_f1.mean()), 6),
            "candidate_minus_anchor": _paired_bootstrap(
                routed_f1, f1[DEFAULT_ROUTE], seed=20261151
            ),
            "per_hop": per_hop,
            "per_training_cohort": per_cohort,
        },
        "models": full_models,
        "models_sha256": canonical_json_sha256(
            {route: value["sha256"] for route, value in full_models.items()}
        ),
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "prospective_boundary": {
            "all_v78_v79_training_ids_excluded_from_v80_target_stage": True,
            "v80_target_ids_rows_features_scores_or_metrics_seen": False,
            "v80_target_training_or_tuning_cases": 0,
            "v80_is_terminal_same_dataset_holdout_due_four_hop_capacity": True,
        },
    }
    return artifact


def load_router_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v80 router artifact experiment id")
    selected = artifact.get("selected_configuration", {})
    if selected.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("v80 router feature contract drifted")
    if selected.get("default_route") != DEFAULT_ROUTE or tuple(
        selected.get("learned_routes", ())
    ) != LEARNED_ROUTES:
        raise ValueError("v80 router route contract drifted")
    if float(selected.get("threshold", -1.0)) not in THRESHOLD_GRID:
        raise ValueError("v80 router threshold is outside the registered grid")
    if selected.get("weight_mode") not in WEIGHT_MODE_GRID:
        raise ValueError("v80 router sample-weight contract drifted")
    models = artifact.get("models", {})
    for route in LEARNED_ROUTES:
        contract = models.get(route, {})
        payload = contract.get("payload")
        if not isinstance(payload, dict) or canonical_json_sha256(payload) != contract.get(
            "sha256"
        ):
            raise ValueError(f"v80 router model payload hash mismatch: {route}")
        if payload.get("feature_count") != len(FEATURE_NAMES):
            raise ValueError("v80 router feature dimension drifted")
    expected_hash = canonical_json_sha256(
        {route: models[route]["sha256"] for route in LEARNED_ROUTES}
    )
    if expected_hash != artifact.get("models_sha256"):
        raise ValueError("v80 router model collection hash mismatch")
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
        np.asarray([gains[CROSS_METHOD]]),
        np.asarray([gains[SOFT_METHOD]]),
        threshold=float(router_artifact["selected_configuration"]["threshold"]),
    )
    route = str(routes[0])
    selected, _ = select_v75_method(row, route, v75_model)
    return selected, {
        "route": route,
        "predicted_cross_gain_vs_anchor": gains[CROSS_METHOD],
        "predicted_soft_gain_vs_anchor": gains[SOFT_METHOD],
        "predicted_winning_gain_vs_anchor": float(winning_gain[0]),
    }


__all__ = [
    "CANDIDATE_METHOD",
    "CROSS_METHOD",
    "DEFAULT_ROUTE",
    "EXPERIMENT_ID",
    "LEARNED_ROUTES",
    "MODEL_CONFIGURATIONS",
    "ROUTES",
    "ROUTE_POLICY_CONFIGURATIONS",
    "SCHEMA_VERSION",
    "THRESHOLD_GRID",
    "WEIGHT_MODE_GRID",
    "develop_model",
    "load_router_artifact",
    "model_configurations",
    "predict_route_gains",
    "route_predictions",
    "select_method",
]
