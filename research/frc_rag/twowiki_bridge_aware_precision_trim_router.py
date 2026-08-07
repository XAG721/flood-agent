"""Bridge-aware precision trimming layered on the frozen v82 2Wiki router."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.musique_graph_router_transfer import _paired_bootstrap
from research.frc_rag.public_evidence import evidence_metrics
from research.frc_rag.twowiki_cascaded_style_residual_router import (
    CANDIDATE_METHOD as V82_CANDIDATE_METHOD,
    EXPERIMENT_ID as V82_EXPERIMENT_ID,
    load_router_artifact as load_v82_router_artifact,
    select_method as select_v82_method,
)
from research.frc_rag.twowiki_question_router import (
    deserialize_model,
    fit_logistic,
    load_model_artifact as load_v75_model_artifact,
    predict_logistic,
    question_features,
    serialize_model,
)
from research.frc_rag.twowiki_residual_three_route_router import (
    load_router_artifact as load_v81_router_artifact,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
)


EXPERIMENT_ID = "FRC-2WIKI-BRIDGE-AWARE-PRECISION-TRIM-ROUTER-V83"
SCHEMA_VERSION = "frc-twowiki-bridge-aware-precision-trim-router-v83"
CANDIDATE_METHOD = "twowiki_bridge_aware_precision_trim_v83"
BRIDGE_TRIM_ACTION = "bridge_aware_trim_to_four"
KEEP_ACTION = "keep_frozen_v82_selection"
ACTIONS = (BRIDGE_TRIM_ACTION, KEEP_ACTION)
TRAINING_CASES = 3200
COHORT_CASES = 800
FOLD_COUNT = 5
FOLD_SALT = "FRC-2WIKI-V83-BRIDGE-FOLD|"
HASH_DIMENSION_GRID = (64, 128, 256, 512)
LOGISTIC_L2_GRID = (0.5, 2.0, 8.0, 32.0)
BRIDGE_THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CLASSIFIER_CONFIGURATIONS = len(HASH_DIMENSION_GRID) * len(LOGISTIC_L2_GRID)
POLICY_CONFIGURATIONS = CLASSIFIER_CONFIGURATIONS * len(BRIDGE_THRESHOLD_GRID)
BASE_RETAINED_COUNT = 5
TRIMMED_RETAINED_COUNT = 4
QUESTION_TYPES = (
    "bridge_comparison",
    "comparison",
    "compositional",
    "inference",
)
COHORTS = (
    "v75_development",
    "v75_confirmation",
    "v81_development",
    "v82_development",
)


def classifier_configurations() -> list[dict[str, float | int]]:
    return [
        {"hash_dimension": dimension, "l2": l2}
        for dimension, l2 in product(HASH_DIMENSION_GRID, LOGISTIC_L2_GRID)
    ]


def _fold(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}{case_id}".encode()).hexdigest()
    return int(digest[:16], 16) % FOLD_COUNT


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("v83 bridge labels require both classes")
    return float(
        ((predictions[positive] == 1).mean() + (predictions[negative] == 0).mean()) / 2
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
            raise ValueError("v83 bridge classifier did not converge")
        probabilities[held_out] = predict_logistic(model, features[held_out])
        row: dict[str, Any] = {
            "fold": fold,
            "training_cases": int(training.sum()),
            "held_out_cases": int(held_out.sum()),
        }
        if include_models:
            row["model"] = serialize_model(model)
        fold_models.append(row)
    if not np.isfinite(probabilities).all():
        raise ValueError("v83 bridge probabilities contain a non-finite value")
    return probabilities, fold_models


def _load_training_evidence(
    scored_paths: Sequence[Path], gold_paths: Sequence[Path]
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    if len(scored_paths) != len(COHORTS) or len(gold_paths) != len(COHORTS):
        raise ValueError("v83 requires four exposed 2Wiki training cohorts")
    scored_by_id: dict[str, dict[str, Any]] = {}
    gold_by_id: dict[str, dict[str, Any]] = {}
    cohort_by_id: dict[str, str] = {}
    contracts: list[dict[str, Any]] = []
    for cohort, scored_path, gold_path in zip(
        COHORTS, scored_paths, gold_paths, strict=True
    ):
        scored = read_jsonl(scored_path)
        gold = read_jsonl(gold_path)
        if len(scored) != COHORT_CASES or len(gold) != COHORT_CASES:
            raise ValueError(f"v83 {cohort} evidence count changed")
        local_scored = {str(row["id"]): row for row in scored}
        local_gold = {str(row["id"]): row for row in gold}
        if (
            len(local_scored) != COHORT_CASES
            or len(local_gold) != COHORT_CASES
            or set(local_scored) != set(local_gold)
            or set(local_scored) & set(scored_by_id)
        ):
            raise ValueError(f"v83 {cohort} evidence is duplicated or misaligned")
        scored_by_id.update(local_scored)
        gold_by_id.update(local_gold)
        cohort_by_id.update({case_id: cohort for case_id in local_scored})
        contracts.append(
            {
                "cohort": cohort,
                "cases": COHORT_CASES,
                "scored_path": scored_path.as_posix(),
                "scored_sha256": sha256(scored_path),
                "gold_path": gold_path.as_posix(),
                "gold_sha256": sha256(gold_path),
            }
        )
    if len(scored_by_id) != TRAINING_CASES:
        raise ValueError("v83 requires exactly 3,200 exposed training cases")
    return scored_by_id, gold_by_id, cohort_by_id, contracts


def _selection_metrics(
    gold_ids: Sequence[str], selected: Sequence[dict[str, Any]]
) -> dict[str, float]:
    selected_ids = [str(row["id"]) for row in selected]
    metrics = evidence_metrics(gold_ids, selected_ids)
    return {
        **metrics,
        "complete_evidence": float(set(gold_ids) <= set(selected_ids)),
        "selected_tokens": float(
            sum(int(row.get("token_count", 1)) for row in selected)
        ),
    }


def _aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("v83 cannot aggregate empty evidence")
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


def _diagnostic_rows(
    trim_mask: np.ndarray,
    base_rows: Sequence[dict[str, float]],
    trimmed_rows: Sequence[dict[str, float]],
) -> list[dict[str, float]]:
    if len(trim_mask) != len(base_rows) or len(base_rows) != len(trimmed_rows):
        raise ValueError("v83 diagnostic rows are not aligned")
    return [
        trimmed_rows[index] if bool(trim_mask[index]) else base_rows[index]
        for index in range(len(base_rows))
    ]


def develop_model(
    scored_paths: Sequence[Path],
    gold_paths: Sequence[Path],
    v75_model_path: Path,
    v81_router_path: Path,
    v82_router_path: Path,
) -> dict[str, Any]:
    scored_by_id, gold_by_id, cohort_by_id, source_contracts = _load_training_evidence(
        scored_paths, gold_paths
    )
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    v81_router = load_v81_router_artifact(v81_router_path)
    v82_router, v82_classifier = load_v82_router_artifact(v82_router_path)
    if (
        v82_router.get("experiment_id") != V82_EXPERIMENT_ID
        or v82_router.get("v75_router_model_payload_sha256")
        != v75_artifact["model_sha256"]
        or v82_router.get("v81_router_file_sha256") != sha256(v81_router_path)
    ):
        raise ValueError("v83 frozen v75/v81/v82 dependency chain is inconsistent")
    case_ids = sorted(scored_by_id)
    questions = [str(scored_by_id[case_id]["question"]) for case_id in case_ids]
    types = np.asarray(
        [str(gold_by_id[case_id]["question_type"]) for case_id in case_ids]
    )
    if tuple(sorted(set(types.tolist()))) != QUESTION_TYPES:
        raise ValueError("v83 training question-type contract changed")
    labels = (types == "bridge_comparison").astype(int)
    folds = np.asarray([_fold(case_id) for case_id in case_ids], dtype=int)
    if set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("v83 deterministic folds are incomplete")
    cohorts = np.asarray([cohort_by_id[case_id] for case_id in case_ids])
    base_selections: list[list[dict[str, Any]]] = []
    base_metadata: list[dict[str, Any]] = []
    for case_id in case_ids:
        selected, metadata = select_v82_method(
            scored_by_id[case_id],
            v82_router,
            v82_classifier,
            v81_router,
            v75_model,
        )
        if len(selected) != BASE_RETAINED_COUNT:
            raise ValueError("v83 requires the frozen v82 base to select five items")
        base_selections.append(list(selected))
        base_metadata.append(metadata)
    base_rows = [
        _selection_metrics(gold_by_id[case_id]["gold_evidence_ids"], selected)
        for case_id, selected in zip(case_ids, base_selections, strict=True)
    ]
    trimmed_rows = [
        _selection_metrics(
            gold_by_id[case_id]["gold_evidence_ids"],
            selected[:TRIMMED_RETAINED_COUNT],
        )
        for case_id, selected in zip(case_ids, base_selections, strict=True)
    ]
    base_aggregate = _aggregate(base_rows)
    base_f1 = np.asarray([row["evidence_f1"] for row in base_rows])
    search_rows: list[dict[str, Any]] = []
    configurations = classifier_configurations()
    if len(configurations) != CLASSIFIER_CONFIGURATIONS:
        raise AssertionError("v83 classifier grid changed")
    for configuration_index, configuration in enumerate(configurations):
        features = np.vstack(
            [
                question_features(
                    question, dimension=int(configuration["hash_dimension"])
                )
                for question in questions
            ]
        )
        probabilities, _ = _crossfit_classifier(
            features, labels, folds, l2=float(configuration["l2"])
        )
        for threshold in BRIDGE_THRESHOLD_GRID:
            trim_mask = probabilities >= threshold
            candidate_rows = _diagnostic_rows(trim_mask, base_rows, trimmed_rows)
            candidate = _aggregate(candidate_rows)
            candidate_f1 = np.asarray([row["evidence_f1"] for row in candidate_rows])
            per_cohort = {
                cohort: round(
                    float(
                        (
                            candidate_f1[cohorts == cohort] - base_f1[cohorts == cohort]
                        ).mean()
                    ),
                    6,
                )
                for cohort in COHORTS
            }
            per_type = {
                question_type: round(
                    float(
                        (
                            candidate_f1[types == question_type]
                            - base_f1[types == question_type]
                        ).mean()
                    ),
                    6,
                )
                for question_type in QUESTION_TYPES
            }
            search_rows.append(
                {
                    "classifier_configuration_index": configuration_index,
                    **configuration,
                    "bridge_threshold": threshold,
                    "candidate": candidate,
                    "candidate_minus_v82": round(
                        candidate["evidence_macro_f1"]
                        - base_aggregate["evidence_macro_f1"],
                        6,
                    ),
                    "bridge_balanced_accuracy": round(
                        _balanced_accuracy(labels, trim_mask.astype(int)), 6
                    ),
                    "action_counts": {
                        BRIDGE_TRIM_ACTION: int(trim_mask.sum()),
                        KEEP_ACTION: int((~trim_mask).sum()),
                    },
                    "per_training_cohort_delta_vs_v82": per_cohort,
                    "per_question_type_delta_vs_v82": per_type,
                }
            )
    if len(search_rows) != POLICY_CONFIGURATIONS:
        raise AssertionError("v83 finite policy grid changed")
    safe = [
        row
        for row in search_rows
        if row["candidate_minus_v82"] >= 0.005
        and min(row["per_training_cohort_delta_vs_v82"].values()) >= 0.005
        and min(row["per_question_type_delta_vs_v82"].values()) >= -0.001
        and row["bridge_balanced_accuracy"] >= 0.95
        and row["candidate"]["complete_evidence_recall"] >= 0.58
        and 0.15 <= row["action_counts"][BRIDGE_TRIM_ACTION] / TRAINING_CASES <= 0.35
    ]
    if not safe:
        raise ValueError("v83 has no cohort-robust safe policy")
    selected = max(
        safe,
        key=lambda row: (
            min(row["per_training_cohort_delta_vs_v82"].values()),
            row["candidate_minus_v82"],
            row["bridge_balanced_accuracy"],
            -int(row["hash_dimension"]),
            float(row["l2"]),
            -abs(float(row["bridge_threshold"]) - 0.3),
        ),
    )
    selected_configuration = {
        "classifier_configuration_index": int(
            selected["classifier_configuration_index"]
        ),
        "hash_dimension": int(selected["hash_dimension"]),
        "l2": float(selected["l2"]),
        "bridge_threshold": float(selected["bridge_threshold"]),
        "base_retained_count": BASE_RETAINED_COUNT,
        "trimmed_retained_count": TRIMMED_RETAINED_COUNT,
        "bridge_label": "official training type equals bridge_comparison",
        "runtime_classifier_input": "question text only",
        "official_question_type_answer_or_gold_used_at_runtime": False,
        "policy": (
            "run the frozen v82 selector first; retain its first four items when "
            "the frozen bridge probability reaches the threshold, otherwise keep "
            "all five"
        ),
    }
    dimension = selected_configuration["hash_dimension"]
    features = np.vstack(
        [question_features(question, dimension=dimension) for question in questions]
    )
    probabilities, fold_models = _crossfit_classifier(
        features,
        labels,
        folds,
        l2=selected_configuration["l2"],
        include_models=True,
    )
    trim_mask = probabilities >= selected_configuration["bridge_threshold"]
    diagnostic_rows = _diagnostic_rows(trim_mask, base_rows, trimmed_rows)
    diagnostic_f1 = np.asarray([row["evidence_f1"] for row in diagnostic_rows])
    final_model = fit_logistic(features, labels, l2=selected_configuration["l2"])
    if not final_model["converged"]:
        raise ValueError("v83 final bridge classifier did not converge")
    classifier_payload = serialize_model(final_model)
    candidate_aggregate = _aggregate(diagnostic_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training": {
            "dataset": "2WikiMultiHopQA",
            "cases": TRAINING_CASES,
            "sources": source_contracts,
            "case_ids_sha256": canonical_json_sha256(case_ids),
            "official_type_used_only_as_exposed_training_label": True,
            "runtime_uses_official_type_answer_or_gold": False,
            "all_training_cases_permanently_excluded_from_v83_targets": True,
        },
        "search": {
            "classifier_configurations": CLASSIFIER_CONFIGURATIONS,
            "policy_configurations": POLICY_CONFIGURATIONS,
            "configuration_space": {
                "hash_dimension": list(HASH_DIMENSION_GRID),
                "l2": list(LOGISTIC_L2_GRID),
                "bridge_threshold": list(BRIDGE_THRESHOLD_GRID),
                "folds": FOLD_COUNT,
                "base_retained_count": BASE_RETAINED_COUNT,
                "trimmed_retained_count": TRIMMED_RETAINED_COUNT,
            },
            "cohort_robust_safe_policy_configurations": len(safe),
            "all_selection_used_only_permanently_excluded_cases": True,
            "oof_score_is_model_selection_evidence_not_unbiased_confirmation": True,
            "selection_order": (
                "largest worst-cohort delta, overall delta, bridge balanced "
                "accuracy, smaller dimension, larger L2, threshold closest to 0.3"
            ),
            "top_10_safe": sorted(
                safe,
                key=lambda row: (
                    min(row["per_training_cohort_delta_vs_v82"].values()),
                    row["candidate_minus_v82"],
                    row["bridge_balanced_accuracy"],
                ),
                reverse=True,
            )[:10],
            "all_configuration_results_sha256": canonical_json_sha256(search_rows),
        },
        "selected_configuration": selected_configuration,
        "bridge_classifier": classifier_payload,
        "bridge_classifier_sha256": canonical_json_sha256(classifier_payload),
        "crossfit_model_selection_diagnostic": {
            "folds": FOLD_COUNT,
            "fold_models": fold_models,
            "v82_base": base_aggregate,
            "candidate": candidate_aggregate,
            "candidate_minus_v82": _paired_bootstrap(
                diagnostic_f1, base_f1, seed=20261203
            ),
            "bridge_balanced_accuracy": round(
                _balanced_accuracy(labels, trim_mask.astype(int)), 6
            ),
            "action_counts": dict(
                sorted(
                    Counter(
                        np.where(trim_mask, BRIDGE_TRIM_ACTION, KEEP_ACTION).tolist()
                    ).items()
                )
            ),
            "retained_count_distribution": {
                str(TRIMMED_RETAINED_COUNT): int(trim_mask.sum()),
                str(BASE_RETAINED_COUNT): int((~trim_mask).sum()),
            },
            "per_training_cohort": {
                cohort: {
                    "cases": int((cohorts == cohort).sum()),
                    "candidate_minus_v82": round(
                        float(
                            (
                                diagnostic_f1[cohorts == cohort]
                                - base_f1[cohorts == cohort]
                            ).mean()
                        ),
                        6,
                    ),
                }
                for cohort in COHORTS
            },
            "per_question_type": {
                question_type: {
                    "cases": int((types == question_type).sum()),
                    "candidate_minus_v82": round(
                        float(
                            (
                                diagnostic_f1[types == question_type]
                                - base_f1[types == question_type]
                            ).mean()
                        ),
                        6,
                    ),
                }
                for question_type in QUESTION_TYPES
            },
        },
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "v81_router_file_sha256": sha256(v81_router_path),
        "v82_router_file_sha256": sha256(v82_router_path),
        "v82_router_experiment_id": v82_router["experiment_id"],
        "base_method": V82_CANDIDATE_METHOD,
        "prospective_boundary": {
            "all_5000_prior_2wiki_ids_excluded_from_v83_target_stages": True,
            "v82_development_is_training_not_confirmation_evidence": True,
            "v83_target_ids_rows_features_scores_or_metrics_seen": False,
            "v83_target_training_or_tuning_cases": 0,
        },
    }


def load_router_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v83 router artifact experiment id")
    selected = artifact.get("selected_configuration", {})
    if (
        selected.get("hash_dimension") not in HASH_DIMENSION_GRID
        or selected.get("l2") not in LOGISTIC_L2_GRID
        or selected.get("bridge_threshold") not in BRIDGE_THRESHOLD_GRID
        or selected.get("base_retained_count") != BASE_RETAINED_COUNT
        or selected.get("trimmed_retained_count") != TRIMMED_RETAINED_COUNT
        or selected.get("official_question_type_answer_or_gold_used_at_runtime")
        is not False
    ):
        raise ValueError("v83 selected configuration drifted")
    payload = artifact.get("bridge_classifier")
    if not isinstance(payload, dict) or canonical_json_sha256(payload) != artifact.get(
        "bridge_classifier_sha256"
    ):
        raise ValueError("v83 bridge classifier hash mismatch")
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
        raise ValueError("v83 bridge classifier payload is invalid")
    return artifact, deserialize_model(payload)


def bridge_probability(
    question: str, router_artifact: dict[str, Any], classifier: dict[str, Any]
) -> float:
    dimension = int(router_artifact["selected_configuration"]["hash_dimension"])
    return float(
        predict_logistic(classifier, question_features(question, dimension=dimension))[
            0
        ]
    )


def precision_trim(
    selected: Sequence[dict[str, Any]],
    probability: float,
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], str]:
    values = list(selected)
    if len(values) > BASE_RETAINED_COUNT:
        raise ValueError("v83 frozen base selection exceeds the registered cap")
    if float(probability) >= float(threshold) and len(values) > TRIMMED_RETAINED_COUNT:
        return values[:TRIMMED_RETAINED_COUNT], BRIDGE_TRIM_ACTION
    return values, KEEP_ACTION


def select_method(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    classifier: dict[str, Any],
    v82_router: dict[str, Any],
    v82_classifier: dict[str, Any],
    v81_router: dict[str, Any],
    v75_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, v82_metadata = select_v82_method(
        row, v82_router, v82_classifier, v81_router, v75_model
    )
    probability = bridge_probability(str(row["question"]), router_artifact, classifier)
    retained, action = precision_trim(
        selected,
        probability,
        threshold=float(router_artifact["selected_configuration"]["bridge_threshold"]),
    )
    return retained, {
        "route": v82_metadata["route"],
        "action": action,
        "v82_action": v82_metadata["action"],
        "v82_route": v82_metadata["route"],
        "bridge_probability": probability,
        "base_retained_count": len(selected),
        "retained_count": len(retained),
        "v75_default_route": v82_metadata.get("v75_default_route"),
        "v75_comparison_probability": v82_metadata.get("v75_comparison_probability"),
        "direct_comparison_probability": v82_metadata.get(
            "direct_comparison_probability"
        ),
        "predicted_cross_override_gain": v82_metadata.get(
            "predicted_cross_override_gain"
        ),
        "predicted_flip_override_gain": v82_metadata.get(
            "predicted_flip_override_gain"
        ),
    }


__all__ = [
    "ACTIONS",
    "BASE_RETAINED_COUNT",
    "BRIDGE_THRESHOLD_GRID",
    "BRIDGE_TRIM_ACTION",
    "CANDIDATE_METHOD",
    "CLASSIFIER_CONFIGURATIONS",
    "COHORTS",
    "EXPERIMENT_ID",
    "HASH_DIMENSION_GRID",
    "KEEP_ACTION",
    "LOGISTIC_L2_GRID",
    "POLICY_CONFIGURATIONS",
    "SCHEMA_VERSION",
    "TRAINING_CASES",
    "TRIMMED_RETAINED_COUNT",
    "bridge_probability",
    "classifier_configurations",
    "develop_model",
    "load_router_artifact",
    "precision_trim",
    "select_method",
]
