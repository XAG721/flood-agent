"""Question-text cardinality control layered on the frozen HotpotQA v76 router."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.hotpot_graph_router import (
    CANDIDATE_METHOD as V76_CANDIDATE_METHOD,
    EXPERIMENT_ID as V76_EXPERIMENT_ID,
    _paired_bootstrap,
    blind_history_row,
    load_router_artifact as load_v76_router_artifact,
    select_method as select_v76_method,
)
from research.frc_rag.public_evidence import evidence_metrics
from research.frc_rag.twowiki_question_router import (
    deserialize_model,
    fit_logistic,
    load_model_artifact as load_v75_model_artifact,
    predict_logistic,
    question_features,
    serialize_model,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
)


EXPERIMENT_ID = "FRC-HOTPOT-QUESTION-TYPE-CARDINALITY-ROUTER-V84"
SCHEMA_VERSION = "frc-hotpot-question-type-cardinality-router-v84"
CANDIDATE_METHOD = "hotpot_question_type_cardinality_router_v84"
COMPARISON_ACTION = "predicted_comparison_retain_three"
BRIDGE_ACTION = "predicted_bridge_retain_four"
ACTIONS = (COMPARISON_ACTION, BRIDGE_ACTION)
COHORTS = ("history", "v76_development", "v76_confirmation")
COHORT_CASES = {"history": 1000, "v76_development": 800, "v76_confirmation": 800}
TRAINING_CASES = sum(COHORT_CASES.values())
QUESTION_TYPES = ("bridge", "comparison")
FOLD_COUNT = 5
FOLD_SALT = "FRC-HOTPOT-V84-QUESTION-TYPE-FOLD|"
HASH_DIMENSION_GRID = (64, 128, 256, 512)
LOGISTIC_L2_GRID = (0.5, 2.0, 8.0, 32.0)
COMPARISON_THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CLASSIFIER_CONFIGURATIONS = len(HASH_DIMENSION_GRID) * len(LOGISTIC_L2_GRID)
POLICY_CONFIGURATIONS = CLASSIFIER_CONFIGURATIONS * len(COMPARISON_THRESHOLD_GRID)
BASE_RETAINED_COUNT = 5
COMPARISON_RETAINED_COUNT = 3
BRIDGE_RETAINED_COUNT = 4


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
        raise ValueError("v84 comparison labels require both classes")
    return float(
        ((predictions[positive] == 1).mean() + (predictions[negative] == 0).mean())
        / 2
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
            raise ValueError("v84 comparison classifier did not converge")
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
        raise ValueError("v84 comparison probabilities contain a non-finite value")
    return probabilities, fold_models


def _selection_metrics(
    gold_ids: Sequence[str], selected: Sequence[dict[str, Any]]
) -> dict[str, float]:
    ids = [str(row["id"]) for row in selected]
    metrics = evidence_metrics(gold_ids, ids)
    return {
        **metrics,
        "complete_evidence": float(set(gold_ids) <= set(ids)),
        "selected_tokens": float(
            sum(int(row.get("token_count", 1)) for row in selected)
        ),
    }


def _aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("v84 cannot aggregate empty evidence")
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


def _load_training_evidence(
    history_path: Path,
    scored_paths: Sequence[Path],
    gold_paths: Sequence[Path],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    if len(scored_paths) != 2 or len(gold_paths) != 2:
        raise ValueError("v84 requires the two exposed v76 target cohorts")
    history = read_jsonl(history_path)
    if len(history) != COHORT_CASES["history"]:
        raise ValueError("v84 history count changed")
    scored_by_id: dict[str, dict[str, Any]] = {}
    gold_by_id: dict[str, dict[str, Any]] = {}
    cohort_by_id: dict[str, str] = {}
    for row in history:
        case_id = str(row["id"])
        if case_id in scored_by_id:
            raise ValueError("v84 history ids are duplicated")
        scored_by_id[case_id] = blind_history_row(row)
        gold_by_id[case_id] = {
            "id": case_id,
            "question_type": str(row["question_type"]),
            "gold_evidence_ids": list(row["gold_evidence_ids"]),
        }
        cohort_by_id[case_id] = "history"
    contracts = [
        {
            "cohort": "history",
            "cases": COHORT_CASES["history"],
            "scored_path": history_path.as_posix(),
            "scored_sha256": sha256(history_path),
            "gold_path": history_path.as_posix(),
            "gold_sha256": sha256(history_path),
        }
    ]
    for cohort, scored_path, gold_path in zip(
        COHORTS[1:], scored_paths, gold_paths, strict=True
    ):
        scored = read_jsonl(scored_path)
        gold = read_jsonl(gold_path)
        expected = COHORT_CASES[cohort]
        local_scored = {str(row["id"]): row for row in scored}
        local_gold = {str(row["id"]): row for row in gold}
        if (
            len(scored) != expected
            or len(gold) != expected
            or len(local_scored) != expected
            or len(local_gold) != expected
            or set(local_scored) != set(local_gold)
            or set(local_scored) & set(scored_by_id)
        ):
            raise ValueError(f"v84 {cohort} evidence is duplicated or misaligned")
        scored_by_id.update(local_scored)
        gold_by_id.update(local_gold)
        cohort_by_id.update({case_id: cohort for case_id in local_scored})
        contracts.append(
            {
                "cohort": cohort,
                "cases": expected,
                "scored_path": scored_path.as_posix(),
                "scored_sha256": sha256(scored_path),
                "gold_path": gold_path.as_posix(),
                "gold_sha256": sha256(gold_path),
            }
        )
    if len(scored_by_id) != TRAINING_CASES:
        raise ValueError("v84 requires exactly 2,600 exposed training cases")
    return scored_by_id, gold_by_id, cohort_by_id, contracts


def develop_model(
    history_path: Path,
    scored_paths: Sequence[Path],
    gold_paths: Sequence[Path],
    v75_model_path: Path,
    v76_router_path: Path,
) -> dict[str, Any]:
    scored_by_id, gold_by_id, cohort_by_id, contracts = _load_training_evidence(
        history_path, scored_paths, gold_paths
    )
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    v76_router = load_v76_router_artifact(v76_router_path)
    if v76_router.get("experiment_id") != V76_EXPERIMENT_ID:
        raise ValueError("v84 frozen v76 dependency changed")
    case_ids = sorted(scored_by_id)
    questions = [str(scored_by_id[case_id]["question"]) for case_id in case_ids]
    types = np.asarray(
        [str(gold_by_id[case_id]["question_type"]) for case_id in case_ids]
    )
    if tuple(sorted(set(types.tolist()))) != QUESTION_TYPES:
        raise ValueError("v84 training question-type contract changed")
    labels = (types == "comparison").astype(int)
    folds = np.asarray([_fold(case_id) for case_id in case_ids], dtype=int)
    cohorts = np.asarray([cohort_by_id[case_id] for case_id in case_ids])
    if set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("v84 deterministic folds are incomplete")
    base_selections: list[list[dict[str, Any]]] = []
    for case_id in case_ids:
        selected, _ = select_v76_method(
            scored_by_id[case_id], V76_CANDIDATE_METHOD, v76_router, v75_model
        )
        if not COMPARISON_RETAINED_COUNT <= len(selected) <= BASE_RETAINED_COUNT:
            raise ValueError(
                "v84 frozen v76 base selection falls outside the registered bounds"
            )
        base_selections.append(list(selected))
    rows3 = [
        _selection_metrics(gold_by_id[case_id]["gold_evidence_ids"], selected[:3])
        for case_id, selected in zip(case_ids, base_selections, strict=True)
    ]
    rows4 = [
        _selection_metrics(gold_by_id[case_id]["gold_evidence_ids"], selected[:4])
        for case_id, selected in zip(case_ids, base_selections, strict=True)
    ]
    rows5 = [
        _selection_metrics(gold_by_id[case_id]["gold_evidence_ids"], selected)
        for case_id, selected in zip(case_ids, base_selections, strict=True)
    ]
    fixed4_f1 = np.asarray([row["evidence_f1"] for row in rows4])
    search_rows: list[dict[str, Any]] = []
    configurations = classifier_configurations()
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
        for threshold in COMPARISON_THRESHOLD_GRID:
            comparison_mask = probabilities >= threshold
            candidate_rows = [
                rows3[index] if comparison_mask[index] else rows4[index]
                for index in range(TRAINING_CASES)
            ]
            candidate = _aggregate(candidate_rows)
            candidate_f1 = np.asarray(
                [row["evidence_f1"] for row in candidate_rows]
            )
            per_cohort = {
                cohort: {
                    "cases": int((cohorts == cohort).sum()),
                    "candidate_evidence_macro_f1": round(
                        float(candidate_f1[cohorts == cohort].mean()), 6
                    ),
                    "candidate_minus_fixed_prefix4": round(
                        float(
                            (
                                candidate_f1[cohorts == cohort]
                                - fixed4_f1[cohorts == cohort]
                            ).mean()
                        ),
                        6,
                    ),
                    "candidate_complete_evidence_recall": round(
                        float(
                            np.asarray(
                                [row["complete_evidence"] for row in candidate_rows]
                            )[cohorts == cohort].mean()
                        ),
                        6,
                    ),
                }
                for cohort in COHORTS
            }
            search_rows.append(
                {
                    "classifier_configuration_index": configuration_index,
                    **configuration,
                    "comparison_threshold": threshold,
                    "candidate": candidate,
                    "candidate_minus_fixed_prefix4": round(
                        candidate["evidence_macro_f1"]
                        - _aggregate(rows4)["evidence_macro_f1"],
                        6,
                    ),
                    "comparison_balanced_accuracy": round(
                        _balanced_accuracy(labels, comparison_mask.astype(int)), 6
                    ),
                    "action_counts": {
                        COMPARISON_ACTION: int(comparison_mask.sum()),
                        BRIDGE_ACTION: int((~comparison_mask).sum()),
                    },
                    "per_training_cohort": per_cohort,
                }
            )
    if len(search_rows) != POLICY_CONFIGURATIONS:
        raise AssertionError("v84 finite policy grid changed")
    safe = [
        row
        for row in search_rows
        if min(
            item["candidate_minus_fixed_prefix4"]
            for item in row["per_training_cohort"].values()
        )
        >= 0.005
        and min(
            item["candidate_complete_evidence_recall"]
            for item in row["per_training_cohort"].values()
        )
        >= 0.62
        and row["comparison_balanced_accuracy"] >= 0.9
        and 0.2
        <= row["action_counts"][COMPARISON_ACTION] / TRAINING_CASES
        <= 0.5
    ]
    if not safe:
        raise ValueError("v84 has no cohort-robust safe policy")
    selected = max(
        safe,
        key=lambda row: (
            min(
                item["candidate_minus_fixed_prefix4"]
                for item in row["per_training_cohort"].values()
            ),
            min(
                item["candidate_evidence_macro_f1"]
                for item in row["per_training_cohort"].values()
            ),
            row["candidate_minus_fixed_prefix4"],
            row["comparison_balanced_accuracy"],
            -int(row["hash_dimension"]),
            float(row["l2"]),
            -abs(float(row["comparison_threshold"]) - 0.3),
        ),
    )
    selected_configuration = {
        "classifier_configuration_index": int(
            selected["classifier_configuration_index"]
        ),
        "hash_dimension": int(selected["hash_dimension"]),
        "l2": float(selected["l2"]),
        "comparison_threshold": float(selected["comparison_threshold"]),
        "base_retained_count": BASE_RETAINED_COUNT,
        "comparison_retained_count": COMPARISON_RETAINED_COUNT,
        "bridge_retained_count": BRIDGE_RETAINED_COUNT,
        "comparison_label": "official exposed training type equals comparison",
        "runtime_classifier_input": "question text only",
        "official_question_type_answer_or_gold_used_at_runtime": False,
        "policy": (
            "run the frozen v76 selector first; retain its first three items for "
            "predicted comparison questions and first four items otherwise"
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
    comparison_mask = probabilities >= selected_configuration["comparison_threshold"]
    diagnostic_rows = [
        rows3[index] if comparison_mask[index] else rows4[index]
        for index in range(TRAINING_CASES)
    ]
    diagnostic_f1 = np.asarray([row["evidence_f1"] for row in diagnostic_rows])
    final_model = fit_logistic(features, labels, l2=selected_configuration["l2"])
    if not final_model["converged"]:
        raise ValueError("v84 final comparison classifier did not converge")
    classifier_payload = serialize_model(final_model)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training": {
            "dataset": "HotpotQA distractor validation",
            "cases": TRAINING_CASES,
            "sources": contracts,
            "case_ids_sha256": canonical_json_sha256(case_ids),
            "official_type_used_only_as_exposed_training_label": True,
            "runtime_uses_official_type_answer_or_gold": False,
            "all_training_cases_permanently_excluded_from_v84_targets": True,
        },
        "search": {
            "classifier_configurations": CLASSIFIER_CONFIGURATIONS,
            "policy_configurations": POLICY_CONFIGURATIONS,
            "configuration_space": {
                "hash_dimension": list(HASH_DIMENSION_GRID),
                "l2": list(LOGISTIC_L2_GRID),
                "comparison_threshold": list(COMPARISON_THRESHOLD_GRID),
                "folds": FOLD_COUNT,
                "base_retained_count": BASE_RETAINED_COUNT,
                "comparison_retained_count": COMPARISON_RETAINED_COUNT,
                "bridge_retained_count": BRIDGE_RETAINED_COUNT,
            },
            "cohort_robust_safe_policy_configurations": len(safe),
            "all_selection_used_only_permanently_excluded_cases": True,
            "oof_score_is_model_selection_evidence_not_unbiased_confirmation": True,
            "selection_order": (
                "largest worst-cohort delta versus fixed prefix4, worst-cohort F1, "
                "overall delta, balanced accuracy, smaller dimension, larger L2"
            ),
            "top_10_safe": sorted(
                safe,
                key=lambda row: (
                    min(
                        item["candidate_minus_fixed_prefix4"]
                        for item in row["per_training_cohort"].values()
                    ),
                    min(
                        item["candidate_evidence_macro_f1"]
                        for item in row["per_training_cohort"].values()
                    ),
                ),
                reverse=True,
            )[:10],
            "all_configuration_results_sha256": canonical_json_sha256(search_rows),
        },
        "selected_configuration": selected_configuration,
        "comparison_classifier": classifier_payload,
        "comparison_classifier_sha256": canonical_json_sha256(classifier_payload),
        "crossfit_model_selection_diagnostic": {
            "folds": FOLD_COUNT,
            "fold_models": fold_models,
            "fixed_prefix3": _aggregate(rows3),
            "fixed_prefix4": _aggregate(rows4),
            "frozen_v76": _aggregate(rows5),
            "candidate": _aggregate(diagnostic_rows),
            "candidate_minus_fixed_prefix4": _paired_bootstrap(
                diagnostic_f1, fixed4_f1, seed=20261206
            ),
            "comparison_balanced_accuracy": round(
                _balanced_accuracy(labels, comparison_mask.astype(int)), 6
            ),
            "action_counts": dict(
                sorted(
                    Counter(
                        np.where(
                            comparison_mask, COMPARISON_ACTION, BRIDGE_ACTION
                        ).tolist()
                    ).items()
                )
            ),
            "retained_count_distribution": {
                str(COMPARISON_RETAINED_COUNT): int(comparison_mask.sum()),
                str(BRIDGE_RETAINED_COUNT): int((~comparison_mask).sum()),
            },
            "per_training_cohort": {
                cohort: {
                    "cases": int((cohorts == cohort).sum()),
                    "candidate": _aggregate(
                        [
                            diagnostic_rows[index]
                            for index in range(TRAINING_CASES)
                            if cohorts[index] == cohort
                        ]
                    ),
                    "candidate_minus_fixed_prefix4": round(
                        float(
                            (
                                diagnostic_f1[cohorts == cohort]
                                - fixed4_f1[cohorts == cohort]
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
                    "candidate": _aggregate(
                        [
                            diagnostic_rows[index]
                            for index in range(TRAINING_CASES)
                            if types[index] == question_type
                        ]
                    ),
                    "fixed_prefix4": _aggregate(
                        [
                            rows4[index]
                            for index in range(TRAINING_CASES)
                            if types[index] == question_type
                        ]
                    ),
                    "candidate_minus_fixed_prefix4": round(
                        float(
                            (
                                diagnostic_f1[types == question_type]
                                - fixed4_f1[types == question_type]
                            ).mean()
                        ),
                        6,
                    ),
                }
                for question_type in QUESTION_TYPES
            },
        },
        "v75_router_model_payload_sha256": v75_artifact["model_sha256"],
        "v76_router_file_sha256": sha256(v76_router_path),
        "v76_router_experiment_id": v76_router["experiment_id"],
        "base_method": V76_CANDIDATE_METHOD,
        "prospective_boundary": {
            "all_2600_prior_hotpot_ids_excluded_from_v84_target_stages": True,
            "v76_development_and_confirmation_are_training_not_v84_confirmation_evidence": True,
            "v84_target_ids_rows_features_scores_or_metrics_seen": False,
            "v84_target_training_or_tuning_cases": 0,
        },
    }


def load_router_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v84 router artifact experiment id")
    selected = artifact.get("selected_configuration", {})
    if (
        selected.get("hash_dimension") not in HASH_DIMENSION_GRID
        or selected.get("l2") not in LOGISTIC_L2_GRID
        or selected.get("comparison_threshold") not in COMPARISON_THRESHOLD_GRID
        or selected.get("base_retained_count") != BASE_RETAINED_COUNT
        or selected.get("comparison_retained_count") != COMPARISON_RETAINED_COUNT
        or selected.get("bridge_retained_count") != BRIDGE_RETAINED_COUNT
        or selected.get("official_question_type_answer_or_gold_used_at_runtime")
        is not False
    ):
        raise ValueError("v84 selected configuration drifted")
    payload = artifact.get("comparison_classifier")
    if not isinstance(payload, dict) or canonical_json_sha256(payload) != artifact.get(
        "comparison_classifier_sha256"
    ):
        raise ValueError("v84 comparison classifier hash mismatch")
    if (
        payload.get("feature_dimension") != selected["hash_dimension"]
        or len(payload.get("weights", [])) != payload.get("total_feature_count")
        or len(payload.get("means", [])) != payload.get("total_feature_count")
        or len(payload.get("scales", [])) != payload.get("total_feature_count")
        or not payload.get("converged")
    ):
        raise ValueError("v84 comparison classifier payload is invalid")
    return artifact, deserialize_model(payload)


def comparison_probability(
    question: str, router_artifact: dict[str, Any], classifier: dict[str, Any]
) -> float:
    dimension = int(router_artifact["selected_configuration"]["hash_dimension"])
    return float(
        predict_logistic(classifier, question_features(question, dimension=dimension))[
            0
        ]
    )


def cardinality_control(
    selected: Sequence[dict[str, Any]],
    probability: float,
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], str]:
    values = list(selected)
    if len(values) > BASE_RETAINED_COUNT:
        raise ValueError("v84 frozen base selection exceeds the registered cap")
    if float(probability) >= float(threshold):
        return values[:COMPARISON_RETAINED_COUNT], COMPARISON_ACTION
    return values[:BRIDGE_RETAINED_COUNT], BRIDGE_ACTION


def select_method(
    row: dict[str, Any],
    router_artifact: dict[str, Any],
    classifier: dict[str, Any],
    v76_router: dict[str, Any],
    v75_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, v76_metadata = select_v76_method(
        row, V76_CANDIDATE_METHOD, v76_router, v75_model
    )
    probability = comparison_probability(
        str(row["question"]), router_artifact, classifier
    )
    retained, action = cardinality_control(
        selected,
        probability,
        threshold=float(
            router_artifact["selected_configuration"]["comparison_threshold"]
        ),
    )
    return retained, {
        "route": v76_metadata["route"],
        "action": action,
        "comparison_probability": probability,
        "base_retained_count": len(selected),
        "retained_count": len(retained),
        "predicted_soft_gain": v76_metadata.get("predicted_soft_gain"),
    }


__all__ = [
    "ACTIONS",
    "BASE_RETAINED_COUNT",
    "BRIDGE_ACTION",
    "BRIDGE_RETAINED_COUNT",
    "CANDIDATE_METHOD",
    "CLASSIFIER_CONFIGURATIONS",
    "COMPARISON_ACTION",
    "COMPARISON_RETAINED_COUNT",
    "COMPARISON_THRESHOLD_GRID",
    "EXPERIMENT_ID",
    "HASH_DIMENSION_GRID",
    "LOGISTIC_L2_GRID",
    "POLICY_CONFIGURATIONS",
    "SCHEMA_VERSION",
    "TRAINING_CASES",
    "cardinality_control",
    "classifier_configurations",
    "comparison_probability",
    "develop_model",
    "load_router_artifact",
    "select_method",
]
