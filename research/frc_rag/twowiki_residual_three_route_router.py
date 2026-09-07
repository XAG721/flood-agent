"""Residual three-route router developed on frozen v75 target-domain evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from research.frc_rag.hotpot_graph_router import (
    FEATURE_NAMES as GRAPH_FEATURE_NAMES,
    graph_features,
)
from research.frc_rag.hotpot_three_route_router import _export_model
from research.frc_rag.musique_graph_router_transfer import _paired_bootstrap
from research.frc_rag.twowiki_question_router import (
    ANCHOR_METHOD,
    CANDIDATE_METHOD as V75_CANDIDATE_METHOD,
    ROUTER_THRESHOLD,
    SOFT_METHOD,
    load_model_artifact as load_v75_model_artifact,
    route_probability,
    select_method as select_v75_method,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    read_jsonl_gzip,
    sha256,
)


EXPERIMENT_ID = "FRC-2WIKI-RESIDUAL-THREE-ROUTE-ROUTER-V81"
SCHEMA_VERSION = "frc-twowiki-residual-three-route-router-v81"
CANDIDATE_METHOD = "twowiki_residual_three_route_router_v81"
CROSS_METHOD = "cross_encoder_topk"
ROUTES = (CROSS_METHOD, SOFT_METHOD, ANCHOR_METHOD)
DEFAULT_ACTION = "v75_default"
CROSS_ACTION = "cross_override"
FLIP_ACTION = "soft_anchor_flip_override"
LEARNED_ACTIONS = (CROSS_ACTION, FLIP_ACTION)
ACTIONS = (*LEARNED_ACTIONS, DEFAULT_ACTION)
FEATURE_NAMES = (
    *GRAPH_FEATURE_NAMES,
    "v75_default_anchor_indicator",
    "v75_probability_margin",
)
TRAINING_CASES = 1600
FOLD_COUNT = 5
FOLD_SALT = "FRC-2WIKI-V81-RESIDUAL-FOLD|"
RANDOM_STATE = 37
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
PORTABLE_PREDICTION_ATOL = 1e-10
PORTABLE_PREDICTION_RTOL = 1e-10


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


def _predict_model(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Predict with the portable tree payload using sklearn branch semantics."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != int(payload["feature_count"]):
        raise ValueError("v81 model feature dimension mismatch")
    predictions = np.full(
        len(matrix), float(payload["init_prediction"]), dtype=np.float64
    )
    for tree in payload["trees"]:
        tree_predictions = np.empty(len(matrix), dtype=np.float64)
        for row_index, row in enumerate(matrix):
            node = 0
            while int(tree["children_left"][node]) != -1:
                feature = int(tree["features"][node])
                # sklearn casts input matrices to float32 but promotes the scalar
                # before comparing it with its float64 tree threshold. Comparing a
                # numpy.float32 scalar directly would instead round the threshold
                # back to float32 and can select the wrong child at boundary values.
                if float(row[feature]) <= float(tree["thresholds"][node]):
                    node = int(tree["children_left"][node])
                else:
                    node = int(tree["children_right"][node])
            tree_predictions[row_index] = float(tree["values"][node])
        predictions += float(payload["learning_rate"]) * tree_predictions
    return predictions


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
        raise ValueError(f"unsupported v81 sample weight mode: {mode}")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("v81 sample weights are invalid")
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


def residual_features(
    row: dict[str, Any], v75_model: dict[str, Any]
) -> np.ndarray:
    base = graph_features(row, v75_model)
    probability = route_probability(str(row["question"]), v75_model)
    extra = np.asarray(
        [
            float(probability >= ROUTER_THRESHOLD),
            abs(probability - ROUTER_THRESHOLD) * 2.0,
        ],
        dtype=np.float64,
    )
    values = np.concatenate((base, extra))
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("v81 residual features are invalid")
    return values


def route_actions(
    cross_gain: np.ndarray,
    flip_gain: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    cross = np.asarray(cross_gain, dtype=np.float64)
    flip = np.asarray(flip_gain, dtype=np.float64)
    if cross.shape != flip.shape or cross.ndim != 1:
        raise ValueError("v81 residual gains must be aligned one-dimensional arrays")
    if not np.isfinite(cross).all() or not np.isfinite(flip).all():
        raise ValueError("v81 residual gains contain a non-finite value")
    gains = np.column_stack((cross, flip))
    best = gains.argmax(axis=1)
    best_gain = gains[np.arange(len(gains)), best]
    actions = np.full(len(gains), DEFAULT_ACTION, dtype=object)
    eligible = best_gain > float(threshold)
    names = np.asarray(LEARNED_ACTIONS, dtype=object)
    actions[eligible] = names[best[eligible]]
    return actions.astype(str), best_gain


def routes_for_actions(
    actions: Sequence[str], default_routes: Sequence[str]
) -> np.ndarray:
    if len(actions) != len(default_routes):
        raise ValueError("v81 actions and default routes are not aligned")
    routes: list[str] = []
    for action, default_route in zip(actions, default_routes, strict=True):
        if default_route not in {SOFT_METHOD, ANCHOR_METHOD}:
            raise ValueError("v81 default route is outside the v75 route family")
        if action == DEFAULT_ACTION:
            route = default_route
        elif action == CROSS_ACTION:
            route = CROSS_METHOD
        elif action == FLIP_ACTION:
            route = SOFT_METHOD if default_route == ANCHOR_METHOD else ANCHOR_METHOD
        else:
            raise ValueError(f"unsupported v81 residual action: {action}")
        routes.append(route)
    return np.asarray(routes, dtype=str)


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("v81 cannot aggregate empty evidence")
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
        action: np.zeros(len(features), dtype=np.float64)
        for action in LEARNED_ACTIONS
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
        for action in LEARNED_ACTIONS:
            model = _fit_model(
                features[training], targets[action][training], configuration
            )
            native = model.predict(features[held_out])
            predictions[action][held_out] = native
            if verify_export:
                payload = _export_model(model, features.shape[1], configuration)
                exported = _predict_model(payload, features[held_out])
                if not np.allclose(
                    exported,
                    native,
                    atol=PORTABLE_PREDICTION_ATOL,
                    rtol=PORTABLE_PREDICTION_RTOL,
                ):
                    max_absolute_error = float(np.max(np.abs(exported - native)))
                    raise ValueError(
                        "v81 portable fold model differs from sklearn: "
                        f"{action}/{fold}; max_absolute_error={max_absolute_error:.3e}"
                    )
                local["models"][action] = {
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
        raise ValueError("v81 requires exactly v75 development and confirmation")
    scored_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    cohort_by_id: dict[str, str] = {}
    source_contracts: list[dict[str, Any]] = []
    for cohort, scored_path, cases_path in zip(
        ("v75_development", "v75_confirmation"),
        scored_paths,
        cases_paths,
        strict=True,
    ):
        local_scored = read_jsonl(scored_path)
        local_cases = read_jsonl_gzip(cases_path)
        if len(local_scored) != len(local_cases) or len(local_scored) != 800:
            raise ValueError(f"v81 {cohort} evidence count changed")
        scored_rows.extend(local_scored)
        case_rows.extend(local_cases)
        for row in local_scored:
            case_id = str(row["id"])
            if case_id in cohort_by_id:
                raise ValueError("v81 training cohorts overlap")
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
        raise ValueError("v81 requires exactly 1,600 frozen training cases")
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    cases_by_id = {str(row["case_id"]): row for row in case_rows}
    if (
        len(scored_by_id) != TRAINING_CASES
        or len(cases_by_id) != TRAINING_CASES
        or set(scored_by_id) != set(cases_by_id)
    ):
        raise ValueError("v81 training evidence does not align")
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    case_ids = sorted(scored_by_id)
    features = np.vstack(
        [residual_features(scored_by_id[case_id], v75_model) for case_id in case_ids]
    )
    probabilities = np.asarray(
        [
            route_probability(str(scored_by_id[case_id]["question"]), v75_model)
            for case_id in case_ids
        ]
    )
    default_routes = np.where(
        probabilities >= ROUTER_THRESHOLD, ANCHOR_METHOD, SOFT_METHOD
    ).astype(str)
    metrics = {
        route: [cases_by_id[case_id]["methods"][route] for case_id in case_ids]
        for route in ROUTES
    }
    default_rows = _rows_for_routes(default_routes, metrics)
    registered_v75_rows = [
        cases_by_id[case_id]["methods"][V75_CANDIDATE_METHOD]
        for case_id in case_ids
    ]
    if any(
        abs(float(default["evidence_f1"]) - float(registered["evidence_f1"]))
        > 1e-12
        or int(default["selected_tokens"]) != int(registered["selected_tokens"])
        for default, registered in zip(
            default_rows, registered_v75_rows, strict=True
        )
    ):
        raise ValueError("v81 recomputed v75 default differs from audited evidence")
    f1 = {
        route: np.asarray(
            [float(row["evidence_f1"]) for row in rows], dtype=np.float64
        )
        for route, rows in metrics.items()
    }
    default_f1 = np.asarray(
        [float(row["evidence_f1"]) for row in default_rows], dtype=np.float64
    )
    flip_routes = np.where(
        default_routes == ANCHOR_METHOD, SOFT_METHOD, ANCHOR_METHOD
    ).astype(str)
    flip_f1 = np.asarray(
        [f1[route][index] for index, route in enumerate(flip_routes)],
        dtype=np.float64,
    )
    targets = {
        CROSS_ACTION: f1[CROSS_METHOD] - default_f1,
        FLIP_ACTION: flip_f1 - default_f1,
    }
    folds = np.asarray([_fold(case_id) for case_id in case_ids])
    if set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("v81 deterministic folds are incomplete")
    configurations = model_configurations()
    if len(configurations) != MODEL_CONFIGURATIONS:
        raise AssertionError("v81 model grid changed")
    type_values = np.asarray(
        [str(cases_by_id[case_id]["question_type"]) for case_id in case_ids]
    )
    question_types = tuple(sorted(set(type_values.tolist())))
    if question_types != (
        "bridge_comparison",
        "comparison",
        "compositional",
        "inference",
    ):
        raise ValueError("v81 training question-type balance changed")
    default_aggregate = _aggregate(default_rows)
    search_rows: list[dict[str, Any]] = []
    for index, configuration in enumerate(configurations):
        predictions, _ = _crossfit_predictions(
            features, targets, folds, configuration
        )
        for threshold in THRESHOLD_GRID:
            actions, _ = route_actions(
                predictions[CROSS_ACTION],
                predictions[FLIP_ACTION],
                threshold=threshold,
            )
            routes = routes_for_actions(actions, default_routes)
            routed_rows = _rows_for_routes(routes, metrics)
            aggregate = _aggregate(routed_rows)
            routed_f1 = np.asarray(
                [float(row["evidence_f1"]) for row in routed_rows]
            )
            per_type_delta: dict[str, float] = {}
            for question_type in question_types:
                mask = type_values == question_type
                per_type_delta[question_type] = round(
                    float((routed_f1[mask] - default_f1[mask]).mean()), 6
                )
            search_rows.append(
                {
                    "model_configuration_index": index,
                    "threshold": threshold,
                    **aggregate,
                    "candidate_minus_v75": round(
                        aggregate["evidence_macro_f1"]
                        - default_aggregate["evidence_macro_f1"],
                        6,
                    ),
                    "per_question_type_delta_vs_v75": per_type_delta,
                    "action_counts": dict(sorted(Counter(actions).items())),
                    "route_counts": dict(sorted(Counter(routes).items())),
                }
            )
    if len(search_rows) != ROUTE_POLICY_CONFIGURATIONS:
        raise AssertionError("v81 route-policy search count changed")
    safe_rows = [
        row
        for row in search_rows
        if min(row["per_question_type_delta_vs_v75"].values()) >= -0.005
        and sum(
            int(row["action_counts"].get(action, 0))
            for action in LEARNED_ACTIONS
        )
        >= 0.05 * TRAINING_CASES
        and int(row["action_counts"].get(DEFAULT_ACTION, 0))
        >= 0.1 * TRAINING_CASES
        and float(row["candidate_minus_v75"]) > 0.0
    ]
    if not safe_rows:
        raise ValueError("v81 finite search has no OOF-safe residual policy")
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
    actions, winning_gain = route_actions(
        predictions[CROSS_ACTION],
        predictions[FLIP_ACTION],
        threshold=float(selected["threshold"]),
    )
    routes = routes_for_actions(actions, default_routes)
    routed_rows = _rows_for_routes(routes, metrics)
    routed_f1 = np.asarray(
        [float(row["evidence_f1"]) for row in routed_rows], dtype=np.float64
    )
    if round(float(routed_f1.mean()), 6) != selected["evidence_macro_f1"]:
        raise AssertionError("v81 selected OOF result is not reproducible")
    full_models: dict[str, Any] = {}
    for action in LEARNED_ACTIONS:
        model = _fit_model(features, targets[action], selected_configuration)
        payload = _export_model(model, features.shape[1], selected_configuration)
        if not np.allclose(
            _predict_model(payload, features),
            model.predict(features),
            atol=PORTABLE_PREDICTION_ATOL,
            rtol=PORTABLE_PREDICTION_RTOL,
        ):
            raise ValueError("v81 exported full model differs from sklearn")
        full_models[action] = {
            "payload": payload,
            "sha256": canonical_json_sha256(payload),
        }
    oracle_f1 = np.maximum.reduce([f1[route] for route in ROUTES])
    per_type: dict[str, Any] = {}
    for question_type in question_types:
        mask = type_values == question_type
        per_type[question_type] = {
            "cases": int(mask.sum()),
            "candidate_evidence_macro_f1": round(float(routed_f1[mask].mean()), 6),
            "v75_evidence_macro_f1": round(float(default_f1[mask].mean()), 6),
            "candidate_minus_v75": round(
                float((routed_f1[mask] - default_f1[mask]).mean()), 6
            ),
            "oracle_evidence_macro_f1": round(float(oracle_f1[mask].mean()), 6),
            "action_counts": dict(sorted(Counter(actions[mask]).items())),
            "route_counts": dict(sorted(Counter(routes[mask]).items())),
        }
    per_cohort: dict[str, Any] = {}
    cohort_values = np.asarray([cohort_by_id[case_id] for case_id in case_ids])
    for cohort in ("v75_development", "v75_confirmation"):
        mask = cohort_values == cohort
        per_cohort[cohort] = {
            "cases": int(mask.sum()),
            "candidate_evidence_macro_f1": round(float(routed_f1[mask].mean()), 6),
            "v75_evidence_macro_f1": round(float(default_f1[mask].mean()), 6),
            "candidate_minus_v75": round(
                float((routed_f1[mask] - default_f1[mask]).mean()), 6
            ),
            "action_counts": dict(sorted(Counter(actions[mask]).items())),
            "route_counts": dict(sorted(Counter(routes[mask]).items())),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training": {
            "dataset": "2WikiMultiHopQA",
            "cases": len(case_ids),
            "case_ids_sha256": canonical_json_sha256(case_ids),
            "sources": source_contracts,
            "v75_default_was_out_of_sample_for_both_training_cohorts": True,
            "gold_evidence_metrics_used_for_model_selection": True,
            "eligible_for_v81_target_stages": False,
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
            "all_selection_used_only_permanently_excluded_v75_cases": True,
            "oof_score_is_model_selection_evidence_not_unbiased_confirmation": True,
        },
        "selected_configuration": {
            **selected_configuration,
            "model_configuration_index": selected_index,
            "threshold": float(selected["threshold"]),
            "features": "73 gold-free graph, score, overlap and v75 route-confidence features",
            "feature_names": list(FEATURE_NAMES),
            "targets": {
                CROSS_ACTION: "cross_encoder_f1_minus_v75_default_f1",
                FLIP_ACTION: "opposite_soft_or_anchor_f1_minus_v75_default_f1",
            },
            "default_action": DEFAULT_ACTION,
            "learned_actions": list(LEARNED_ACTIONS),
            "official_question_type_answer_or_gold_used_at_runtime": False,
        },
        "crossfit_model_selection_diagnostic": {
            "folds": FOLD_COUNT,
            "fold_models": fold_models,
            "candidate": _aggregate(routed_rows),
            "v75_default": _aggregate(default_rows),
            "candidate_minus_v75": _paired_bootstrap(
                routed_f1, default_f1, seed=20261171
            ),
            "action_counts": dict(sorted(Counter(actions).items())),
            "route_counts": dict(sorted(Counter(routes).items())),
            "mean_winning_predicted_gain": round(float(winning_gain.mean()), 6),
            "oracle_three_route_evidence_macro_f1": round(float(oracle_f1.mean()), 6),
            "per_question_type": per_type,
            "per_training_cohort": per_cohort,
        },
        "models": full_models,
        "models_sha256": canonical_json_sha256(
            {action: value["sha256"] for action, value in full_models.items()}
        ),
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "prospective_boundary": {
            "all_3400_prior_2wiki_ids_excluded_from_v81_target_stages": True,
            "v81_target_ids_rows_features_scores_or_metrics_seen": False,
            "v81_target_training_or_tuning_cases": 0,
            "v80_holdout_rows_scores_gold_or_metrics_used": False,
        },
    }


def load_router_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v81 router artifact experiment id")
    selected = artifact.get("selected_configuration", {})
    if selected.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("v81 router feature contract drifted")
    if selected.get("default_action") != DEFAULT_ACTION or tuple(
        selected.get("learned_actions", ())
    ) != LEARNED_ACTIONS:
        raise ValueError("v81 router action contract drifted")
    if float(selected.get("threshold", -1.0)) not in THRESHOLD_GRID:
        raise ValueError("v81 router threshold is outside the registered grid")
    if selected.get("weight_mode") not in WEIGHT_MODE_GRID:
        raise ValueError("v81 router sample-weight contract drifted")
    models = artifact.get("models", {})
    for action in LEARNED_ACTIONS:
        contract = models.get(action, {})
        payload = contract.get("payload")
        if not isinstance(payload, dict) or canonical_json_sha256(payload) != contract.get(
            "sha256"
        ):
            raise ValueError(f"v81 router model payload hash mismatch: {action}")
        if payload.get("feature_count") != len(FEATURE_NAMES):
            raise ValueError("v81 router feature dimension drifted")
    expected_hash = canonical_json_sha256(
        {action: models[action]["sha256"] for action in LEARNED_ACTIONS}
    )
    if expected_hash != artifact.get("models_sha256"):
        raise ValueError("v81 router model collection hash mismatch")
    return artifact


def predict_residual_gains(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
) -> dict[str, float]:
    features = residual_features(row, v75_model)
    return {
        action: float(
            _predict_model(router_artifact["models"][action]["payload"], features)[0]
        )
        for action in LEARNED_ACTIONS
    }


def select_method(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    v75_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, default_metadata = select_v75_method(
        row, V75_CANDIDATE_METHOD, v75_model
    )
    default_route = str(default_metadata["route"])
    gains = predict_residual_gains(row, router_artifact, v75_model)
    actions, winning_gain = route_actions(
        np.asarray([gains[CROSS_ACTION]]),
        np.asarray([gains[FLIP_ACTION]]),
        threshold=float(router_artifact["selected_configuration"]["threshold"]),
    )
    action = str(actions[0])
    route = str(routes_for_actions([action], [default_route])[0])
    selected, _ = select_v75_method(row, route, v75_model)
    return selected, {
        "route": route,
        "action": action,
        "v75_default_route": default_route,
        "v75_comparison_probability": float(
            default_metadata["comparison_probability"]
        ),
        "predicted_cross_override_gain": gains[CROSS_ACTION],
        "predicted_flip_override_gain": gains[FLIP_ACTION],
        "predicted_winning_residual_gain": float(winning_gain[0]),
    }


__all__ = [
    "ACTIONS",
    "CANDIDATE_METHOD",
    "CROSS_ACTION",
    "CROSS_METHOD",
    "DEFAULT_ACTION",
    "EXPERIMENT_ID",
    "FEATURE_NAMES",
    "FLIP_ACTION",
    "LEARNED_ACTIONS",
    "MODEL_CONFIGURATIONS",
    "ROUTES",
    "ROUTE_POLICY_CONFIGURATIONS",
    "SCHEMA_VERSION",
    "THRESHOLD_GRID",
    "WEIGHT_MODE_GRID",
    "develop_model",
    "load_router_artifact",
    "model_configurations",
    "predict_residual_gains",
    "residual_features",
    "route_actions",
    "routes_for_actions",
    "select_method",
]
