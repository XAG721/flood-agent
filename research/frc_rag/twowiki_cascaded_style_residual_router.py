"""Cascaded direct-comparison and safe-flip router for 2Wiki (v82)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.musique_graph_router_transfer import _paired_bootstrap
from research.frc_rag.twowiki_question_router import (
    ANCHOR_METHOD,
    CANDIDATE_METHOD as V75_CANDIDATE_METHOD,
    SOFT_METHOD,
    deserialize_model,
    fit_logistic,
    load_model_artifact as load_v75_model_artifact,
    predict_logistic,
    question_features,
    serialize_model,
    select_method as select_v75_method,
)
from research.frc_rag.twowiki_residual_three_route_router import (
    CROSS_ACTION as V81_CROSS_ACTION,
    CROSS_METHOD,
    EXPERIMENT_ID as V81_EXPERIMENT_ID,
    FLIP_ACTION as V81_FLIP_ACTION,
    _fold as v81_fold,
    _predict_model as predict_v81_portable_model,
    load_router_artifact as load_v81_router_artifact,
    predict_residual_gains,
    residual_features,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    read_jsonl_gzip,
    sha256,
)


EXPERIMENT_ID = "FRC-2WIKI-CASCADED-STYLE-RESIDUAL-ROUTER-V82"
SCHEMA_VERSION = "frc-twowiki-cascaded-style-residual-router-v82"
CANDIDATE_METHOD = "twowiki_cascaded_style_residual_router_v82"
DIRECT_CROSS_ACTION = "direct_comparison_cross_override"
SAFE_FLIP_ACTION = "v81_safe_flip_override"
DEFAULT_ACTION = "v75_default"
LEARNED_ACTIONS = (DIRECT_CROSS_ACTION, SAFE_FLIP_ACTION)
ACTIONS = (*LEARNED_ACTIONS, DEFAULT_ACTION)
ROUTES = (CROSS_METHOD, SOFT_METHOD, ANCHOR_METHOD)
TRAINING_CASES = 2400
COHORT_CASES = 800
FOLD_COUNT = 5
FOLD_SALT = "FRC-2WIKI-V82-DIRECT-COMPARISON-FOLD|"
HASH_DIMENSION_GRID = (64, 128, 256, 512)
LOGISTIC_L2_GRID = (0.5, 2.0, 8.0, 32.0)
DIRECT_THRESHOLD_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
FLIP_THRESHOLD_GRID = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
CLASSIFIER_CONFIGURATIONS = len(HASH_DIMENSION_GRID) * len(LOGISTIC_L2_GRID)
ROUTE_POLICY_CONFIGURATIONS = (
    CLASSIFIER_CONFIGURATIONS
    * len(DIRECT_THRESHOLD_GRID)
    * len(FLIP_THRESHOLD_GRID)
)


def classifier_configurations() -> list[dict[str, float | int]]:
    return [
        {"hash_dimension": dimension, "l2": l2}
        for dimension, l2 in product(HASH_DIMENSION_GRID, LOGISTIC_L2_GRID)
    ]


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % FOLD_COUNT


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("v82 direct-comparison labels require both classes")
    return float(
        ((predictions[positive] == 1).mean() + (predictions[negative] == 0).mean())
        / 2.0
    )


def _crossfit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    l2: float,
    include_models: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probabilities = np.zeros(len(features), dtype=np.float64)
    fold_models: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        training = folds != fold
        held_out = ~training
        model = fit_logistic(features[training], labels[training], l2=l2)
        if not model["converged"]:
            raise ValueError("v82 direct-comparison classifier did not converge")
        probabilities[held_out] = predict_logistic(model, features[held_out])
        local: dict[str, Any] = {
            "fold": fold,
            "training_cases": int(training.sum()),
            "held_out_cases": int(held_out.sum()),
        }
        if include_models:
            payload = serialize_model(model)
            local["model"] = payload
            local["model_sha256"] = canonical_json_sha256(payload)
        fold_models.append(local)
    return probabilities, fold_models


def route_actions(
    direct_probability: np.ndarray,
    cross_gain: np.ndarray,
    flip_gain: np.ndarray,
    *,
    direct_threshold: float,
    flip_threshold: float,
) -> np.ndarray:
    direct = np.asarray(direct_probability, dtype=np.float64)
    cross = np.asarray(cross_gain, dtype=np.float64)
    flip = np.asarray(flip_gain, dtype=np.float64)
    if direct.ndim != 1 or direct.shape != cross.shape or direct.shape != flip.shape:
        raise ValueError("v82 policy inputs must be aligned one-dimensional arrays")
    if not np.isfinite(np.column_stack((direct, cross, flip))).all():
        raise ValueError("v82 policy inputs contain a non-finite value")
    if np.any((direct < 0.0) | (direct > 1.0)):
        raise ValueError("v82 direct-comparison probabilities are outside [0, 1]")
    actions = np.full(len(direct), DEFAULT_ACTION, dtype=object)
    direct_mask = direct >= float(direct_threshold)
    actions[direct_mask] = DIRECT_CROSS_ACTION
    flip_mask = (
        ~direct_mask
        & (flip > cross)
        & (flip > float(flip_threshold))
    )
    actions[flip_mask] = SAFE_FLIP_ACTION
    return actions.astype(str)


def routes_for_actions(
    actions: Sequence[str], default_routes: Sequence[str]
) -> np.ndarray:
    if len(actions) != len(default_routes):
        raise ValueError("v82 actions and default routes are not aligned")
    routes: list[str] = []
    for action, default_route in zip(actions, default_routes, strict=True):
        if default_route not in {SOFT_METHOD, ANCHOR_METHOD}:
            raise ValueError("v82 default route is outside the v75 route family")
        if action == DEFAULT_ACTION:
            route = default_route
        elif action == DIRECT_CROSS_ACTION:
            route = CROSS_METHOD
        elif action == SAFE_FLIP_ACTION:
            route = SOFT_METHOD if default_route == ANCHOR_METHOD else ANCHOR_METHOD
        else:
            raise ValueError(f"unsupported v82 action: {action}")
        routes.append(route)
    return np.asarray(routes, dtype=str)


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("v82 cannot aggregate empty evidence")
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


def _load_training_evidence(
    scored_paths: Sequence[Path], cases_paths: Sequence[Path]
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    if len(scored_paths) != 3 or len(cases_paths) != 3:
        raise ValueError("v82 requires v75 development/confirmation and v81 development")
    cohorts = ("v75_development", "v75_confirmation", "v81_development")
    scored_by_id: dict[str, dict[str, Any]] = {}
    cases_by_id: dict[str, dict[str, Any]] = {}
    cohort_by_id: dict[str, str] = {}
    contracts: list[dict[str, Any]] = []
    for cohort, scored_path, cases_path in zip(
        cohorts, scored_paths, cases_paths, strict=True
    ):
        scored = read_jsonl(scored_path)
        cases = read_jsonl_gzip(cases_path)
        if len(scored) != COHORT_CASES or len(cases) != COHORT_CASES:
            raise ValueError(f"v82 {cohort} evidence count changed")
        local_scored = {str(row["id"]): row for row in scored}
        local_cases = {str(row["case_id"]): row for row in cases}
        if (
            len(local_scored) != COHORT_CASES
            or len(local_cases) != COHORT_CASES
            or set(local_scored) != set(local_cases)
            or set(local_scored) & set(scored_by_id)
        ):
            raise ValueError(f"v82 {cohort} evidence is duplicated or misaligned")
        scored_by_id.update(local_scored)
        cases_by_id.update(local_cases)
        cohort_by_id.update({case_id: cohort for case_id in local_scored})
        contracts.append(
            {
                "cohort": cohort,
                "cases": COHORT_CASES,
                "scored_path": scored_path.as_posix(),
                "scored_sha256": sha256(scored_path),
                "cases_path": cases_path.as_posix(),
                "cases_sha256": sha256(cases_path),
            }
        )
    if len(scored_by_id) != TRAINING_CASES:
        raise ValueError("v82 requires exactly 2,400 exposed training cases")
    return scored_by_id, cases_by_id, cohort_by_id, contracts


def _out_of_sample_v81_gains(
    case_ids: Sequence[str],
    scored_by_id: dict[str, dict[str, Any]],
    cohort_by_id: dict[str, str],
    v81_router: dict[str, Any],
    v75_model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    fold_models = {
        int(row["fold"]): row["models"]
        for row in v81_router["crossfit_model_selection_diagnostic"]["fold_models"]
    }
    if set(fold_models) != set(range(FOLD_COUNT)):
        raise ValueError("v82 cannot reconstruct all v81 fold models")
    cross_values: list[float] = []
    flip_values: list[float] = []
    provenance = Counter()
    for case_id in case_ids:
        row = scored_by_id[case_id]
        if cohort_by_id[case_id] == "v81_development":
            gains = predict_residual_gains(row, v81_router, v75_model)
            cross = float(gains[V81_CROSS_ACTION])
            flip = float(gains[V81_FLIP_ACTION])
            provenance["v81_full_model_on_unseen_v81_development"] += 1
        else:
            features = residual_features(row, v75_model)
            models = fold_models[v81_fold(case_id)]
            cross = float(
                predict_v81_portable_model(
                    models[V81_CROSS_ACTION]["payload"], features
                )[0]
            )
            flip = float(
                predict_v81_portable_model(
                    models[V81_FLIP_ACTION]["payload"], features
                )[0]
            )
            provenance["v81_fold_model_on_held_out_v75_case"] += 1
        cross_values.append(cross)
        flip_values.append(flip)
    return (
        np.asarray(cross_values, dtype=np.float64),
        np.asarray(flip_values, dtype=np.float64),
        dict(sorted(provenance.items())),
    )


def develop_model(
    scored_paths: Sequence[Path],
    cases_paths: Sequence[Path],
    v75_model_path: Path,
    v81_router_path: Path,
) -> dict[str, Any]:
    scored_by_id, cases_by_id, cohort_by_id, source_contracts = (
        _load_training_evidence(scored_paths, cases_paths)
    )
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    v81_router = load_v81_router_artifact(v81_router_path)
    if (
        v81_router.get("experiment_id") != V81_EXPERIMENT_ID
        or v81_router.get("v75_router_model_payload_sha256")
        != v75_artifact["model_sha256"]
    ):
        raise ValueError("v82 frozen v75/v81 dependency chain is inconsistent")
    case_ids = sorted(scored_by_id)
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
        raise ValueError("v82 training question-type contract changed")
    labels = (type_values == "comparison").astype(int)
    folds = np.asarray([_fold(case_id) for case_id in case_ids], dtype=int)
    if set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("v82 deterministic folds are incomplete")
    cohort_values = np.asarray([cohort_by_id[case_id] for case_id in case_ids])
    cohorts = ("v75_development", "v75_confirmation", "v81_development")
    cross_gain, flip_gain, gain_provenance = _out_of_sample_v81_gains(
        case_ids, scored_by_id, cohort_by_id, v81_router, v75_model
    )
    probabilities = np.asarray(
        [
            float(
                select_v75_method(
                    scored_by_id[case_id], V75_CANDIDATE_METHOD, v75_model
                )[1]["comparison_probability"]
            )
            for case_id in case_ids
        ]
    )
    default_routes = np.where(
        probabilities >= 0.5, ANCHOR_METHOD, SOFT_METHOD
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
        for default, registered in zip(default_rows, registered_v75_rows, strict=True)
    ):
        raise ValueError("v82 recomputed v75 default differs from audited evidence")
    default_f1 = np.asarray(
        [float(row["evidence_f1"]) for row in default_rows], dtype=np.float64
    )
    default_aggregate = _aggregate(default_rows)
    configurations = classifier_configurations()
    if len(configurations) != CLASSIFIER_CONFIGURATIONS:
        raise AssertionError("v82 classifier grid changed")
    questions = [str(scored_by_id[case_id]["question"]) for case_id in case_ids]
    search_rows: list[dict[str, Any]] = []
    for configuration_index, configuration in enumerate(configurations):
        features = np.vstack(
            [
                question_features(
                    question, dimension=int(configuration["hash_dimension"])
                )
                for question in questions
            ]
        )
        direct_probability, _ = _crossfit_classifier(
            features, labels, folds, l2=float(configuration["l2"])
        )
        for direct_threshold in DIRECT_THRESHOLD_GRID:
            direct_predictions = direct_probability >= direct_threshold
            balanced = _balanced_accuracy(labels, direct_predictions.astype(int))
            for flip_threshold in FLIP_THRESHOLD_GRID:
                actions = route_actions(
                    direct_probability,
                    cross_gain,
                    flip_gain,
                    direct_threshold=direct_threshold,
                    flip_threshold=flip_threshold,
                )
                routes = routes_for_actions(actions, default_routes)
                routed_rows = _rows_for_routes(routes, metrics)
                aggregate = _aggregate(routed_rows)
                routed_f1 = np.asarray(
                    [float(row["evidence_f1"]) for row in routed_rows]
                )
                per_type_delta = {
                    question_type: round(
                        float(
                            (
                                routed_f1[type_values == question_type]
                                - default_f1[type_values == question_type]
                            ).mean()
                        ),
                        6,
                    )
                    for question_type in question_types
                }
                per_cohort_delta = {
                    cohort: round(
                        float(
                            (
                                routed_f1[cohort_values == cohort]
                                - default_f1[cohort_values == cohort]
                            ).mean()
                        ),
                        6,
                    )
                    for cohort in cohorts
                }
                search_rows.append(
                    {
                        "classifier_configuration_index": configuration_index,
                        "direct_threshold": direct_threshold,
                        "flip_threshold": flip_threshold,
                        **aggregate,
                        "candidate_minus_v75": round(
                            aggregate["evidence_macro_f1"]
                            - default_aggregate["evidence_macro_f1"],
                            6,
                        ),
                        "direct_comparison_balanced_accuracy": round(balanced, 6),
                        "per_question_type_delta_vs_v75": per_type_delta,
                        "per_training_cohort_delta_vs_v75": per_cohort_delta,
                        "action_counts": dict(sorted(Counter(actions).items())),
                        "route_counts": dict(sorted(Counter(routes).items())),
                    }
                )
    if len(search_rows) != ROUTE_POLICY_CONFIGURATIONS:
        raise AssertionError("v82 route-policy search count changed")
    safe_rows = [
        row
        for row in search_rows
        if min(row["per_question_type_delta_vs_v75"].values()) >= -0.005
        and min(row["per_training_cohort_delta_vs_v75"].values()) >= 0.0
        and float(row["direct_comparison_balanced_accuracy"]) >= 0.9
        and sum(int(row["action_counts"].get(action, 0)) for action in LEARNED_ACTIONS)
        >= 0.05 * TRAINING_CASES
        and int(row["action_counts"].get(DEFAULT_ACTION, 0)) >= 0.1 * TRAINING_CASES
        and float(row["candidate_minus_v75"]) > 0.0
    ]
    if not safe_rows:
        raise ValueError("v82 finite search has no cohort-robust safe policy")

    def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
        configuration = configurations[int(row["classifier_configuration_index"])]
        return (
            -min(row["per_training_cohort_delta_vs_v75"].values()),
            -float(row["evidence_macro_f1"]),
            -min(row["per_question_type_delta_vs_v75"].values()),
            -float(row["direct_comparison_balanced_accuracy"]),
            float(configuration["hash_dimension"]),
            -float(configuration["l2"]),
            abs(float(row["direct_threshold"]) - 0.8),
            -float(row["flip_threshold"]),
        )

    ranked = sorted(safe_rows, key=rank_key)
    selected = ranked[0]
    selected_index = int(selected["classifier_configuration_index"])
    selected_configuration = configurations[selected_index]
    selected_features = np.vstack(
        [
            question_features(
                question, dimension=int(selected_configuration["hash_dimension"])
            )
            for question in questions
        ]
    )
    direct_probability, fold_models = _crossfit_classifier(
        selected_features,
        labels,
        folds,
        l2=float(selected_configuration["l2"]),
        include_models=True,
    )
    actions = route_actions(
        direct_probability,
        cross_gain,
        flip_gain,
        direct_threshold=float(selected["direct_threshold"]),
        flip_threshold=float(selected["flip_threshold"]),
    )
    routes = routes_for_actions(actions, default_routes)
    routed_rows = _rows_for_routes(routes, metrics)
    routed_f1 = np.asarray(
        [float(row["evidence_f1"]) for row in routed_rows], dtype=np.float64
    )
    if round(float(routed_f1.mean()), 6) != selected["evidence_macro_f1"]:
        raise AssertionError("v82 selected OOF result is not reproducible")
    full_model = fit_logistic(
        selected_features, labels, l2=float(selected_configuration["l2"])
    )
    if not full_model["converged"]:
        raise ValueError("v82 full direct-comparison classifier did not converge")
    classifier_payload = serialize_model(full_model)
    classifier_sha256 = canonical_json_sha256(classifier_payload)
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
            "action_counts": dict(sorted(Counter(actions[mask]).items())),
            "route_counts": dict(sorted(Counter(routes[mask]).items())),
        }
    per_cohort: dict[str, Any] = {}
    for cohort in cohorts:
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
            "official_type_used_only_as_exposed_training_label": True,
            "runtime_uses_official_type_answer_or_gold": False,
            "all_training_cases_permanently_excluded_from_v82_targets": True,
        },
        "search": {
            "classifier_configurations": CLASSIFIER_CONFIGURATIONS,
            "route_policy_configurations": ROUTE_POLICY_CONFIGURATIONS,
            "cohort_robust_safe_policy_configurations": len(safe_rows),
            "configuration_space": {
                "hash_dimension": list(HASH_DIMENSION_GRID),
                "l2": list(LOGISTIC_L2_GRID),
                "direct_threshold": list(DIRECT_THRESHOLD_GRID),
                "flip_threshold": list(FLIP_THRESHOLD_GRID),
                "folds": FOLD_COUNT,
            },
            "selection_order": (
                "largest worst-cohort delta, overall F1, worst-type delta, "
                "balanced accuracy, smaller dimension, larger L2, direct threshold "
                "closest to 0.8, larger safe-flip threshold"
            ),
            "all_configuration_results_sha256": canonical_json_sha256(search_rows),
            "top_10_safe": ranked[:10],
            "all_selection_used_only_permanently_excluded_cases": True,
            "oof_score_is_model_selection_evidence_not_unbiased_confirmation": True,
        },
        "selected_configuration": {
            **selected_configuration,
            "classifier_configuration_index": selected_index,
            "direct_threshold": float(selected["direct_threshold"]),
            "flip_threshold": float(selected["flip_threshold"]),
            "direct_label": "official training type equals comparison",
            "runtime_direct_classifier_input": "question text only",
            "policy": (
                "direct comparison probability first routes to cross; otherwise only "
                "the v81 flip action may override v75 when flip gain exceeds both "
                "cross gain and the frozen threshold"
            ),
            "official_question_type_answer_or_gold_used_at_runtime": False,
        },
        "crossfit_model_selection_diagnostic": {
            "folds": FOLD_COUNT,
            "fold_models": fold_models,
            "candidate": _aggregate(routed_rows),
            "v75_default": _aggregate(default_rows),
            "candidate_minus_v75": _paired_bootstrap(
                routed_f1, default_f1, seed=20261191
            ),
            "direct_comparison_balanced_accuracy": round(
                _balanced_accuracy(
                    labels,
                    (direct_probability >= float(selected["direct_threshold"])).astype(
                        int
                    ),
                ),
                6,
            ),
            "action_counts": dict(sorted(Counter(actions).items())),
            "route_counts": dict(sorted(Counter(routes).items())),
            "per_question_type": per_type,
            "per_training_cohort": per_cohort,
            "v81_gain_prediction_provenance": gain_provenance,
        },
        "direct_comparison_classifier": classifier_payload,
        "direct_comparison_classifier_sha256": classifier_sha256,
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "v81_router_file_sha256": sha256(v81_router_path),
        "v81_router_experiment_id": V81_EXPERIMENT_ID,
        "prospective_boundary": {
            "all_4200_prior_2wiki_ids_excluded_from_v82_target_stages": True,
            "v82_target_ids_rows_features_scores_or_metrics_seen": False,
            "v82_target_training_or_tuning_cases": 0,
            "v81_development_is_training_not_confirmation_evidence": True,
        },
    }


def load_router_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v82 router artifact experiment id")
    selected = artifact.get("selected_configuration", {})
    if (
        selected.get("hash_dimension") not in HASH_DIMENSION_GRID
        or selected.get("l2") not in LOGISTIC_L2_GRID
        or selected.get("direct_threshold") not in DIRECT_THRESHOLD_GRID
        or selected.get("flip_threshold") not in FLIP_THRESHOLD_GRID
        or selected.get("official_question_type_answer_or_gold_used_at_runtime")
        is not False
    ):
        raise ValueError("v82 selected configuration drifted")
    payload = artifact.get("direct_comparison_classifier")
    if not isinstance(payload, dict) or canonical_json_sha256(payload) != artifact.get(
        "direct_comparison_classifier_sha256"
    ):
        raise ValueError("v82 direct-comparison classifier hash mismatch")
    fixed_count = int(payload.get("total_feature_count", 0)) - int(
        payload.get("feature_dimension", -1)
    )
    if (
        payload.get("feature_dimension") != selected["hash_dimension"]
        or fixed_count <= 0
        or len(payload.get("weights", [])) != payload.get("total_feature_count")
        or len(payload.get("means", [])) != payload.get("total_feature_count")
        or len(payload.get("scales", [])) != payload.get("total_feature_count")
        or not payload.get("converged")
    ):
        raise ValueError("v82 direct-comparison classifier payload is invalid")
    return artifact, deserialize_model(payload)


def direct_comparison_probability(
    question: str, router_artifact: dict[str, Any], classifier: dict[str, Any]
) -> float:
    dimension = int(router_artifact["selected_configuration"]["hash_dimension"])
    return float(
        predict_logistic(classifier, question_features(question, dimension=dimension))[0]
    )


def select_method(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    classifier: dict[str, Any],
    v81_router: dict[str, Any],
    v75_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, default_metadata = select_v75_method(
        row, V75_CANDIDATE_METHOD, v75_model
    )
    default_route = str(default_metadata["route"])
    probability = direct_comparison_probability(
        str(row["question"]), router_artifact, classifier
    )
    gains = predict_residual_gains(row, v81_router, v75_model)
    actions = route_actions(
        np.asarray([probability]),
        np.asarray([gains[V81_CROSS_ACTION]]),
        np.asarray([gains[V81_FLIP_ACTION]]),
        direct_threshold=float(
            router_artifact["selected_configuration"]["direct_threshold"]
        ),
        flip_threshold=float(
            router_artifact["selected_configuration"]["flip_threshold"]
        ),
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
        "direct_comparison_probability": probability,
        "predicted_cross_override_gain": float(gains[V81_CROSS_ACTION]),
        "predicted_flip_override_gain": float(gains[V81_FLIP_ACTION]),
    }


__all__ = [
    "ACTIONS",
    "CANDIDATE_METHOD",
    "CLASSIFIER_CONFIGURATIONS",
    "DEFAULT_ACTION",
    "DIRECT_CROSS_ACTION",
    "DIRECT_THRESHOLD_GRID",
    "EXPERIMENT_ID",
    "FLIP_THRESHOLD_GRID",
    "HASH_DIMENSION_GRID",
    "LEARNED_ACTIONS",
    "LOGISTIC_L2_GRID",
    "ROUTES",
    "ROUTE_POLICY_CONFIGURATIONS",
    "SAFE_FLIP_ACTION",
    "SCHEMA_VERSION",
    "classifier_configurations",
    "develop_model",
    "direct_comparison_probability",
    "load_router_artifact",
    "route_actions",
    "routes_for_actions",
    "select_method",
]
