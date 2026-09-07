"""Leave-one-dataset-out transfer audit for the FRC sufficiency head."""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.frc_rag.conformal_mondrian import _distribution
from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_subgroup_audit import _repeat_core
from research.frc_rag.conformal_sufficiency import (
    FEATURE_NAMES,
    PRIMARY_ALPHA,
    _auc,
    _conformal_rank,
    _conformal_threshold,
    _decision_metrics,
    _predict,
    _serializable_model,
    _split_for_case,
    evaluate_conformal_sufficiency,
    prepare_conformal_source,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-cross-dataset-head-transfer-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_POST_HOC_CROSS_DATASET_HEAD_TRANSFER"
MIN_SOURCE_DATASETS = 3
ADOPTION_TARGETS = {
    "maximum_mean_case_family_false_complete_rate_each_dataset": 0.1,
    "maximum_dataset_case_family_risk_increase_vs_target_fitted": 0.01,
    "maximum_dataset_complete_recall_decrease_vs_target_fitted": 0.05,
    "maximum_cross_dataset_mean_evaluation_auc_decrease_vs_target_fitted": 0.02,
    "minimum_datasets_with_at_least_7_of_10_alpha_control_repeats": 3,
}
SUMMARY_METRICS = (
    "false_complete_case_family_rate",
    "abstention_rate",
    "true_complete_declaration_rate",
)


def _weighted_mean_and_scale(
    features: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.average(features, axis=0, weights=sample_weights)
    variances = np.average(
        np.square(features - means),
        axis=0,
        weights=sample_weights,
    )
    scales = np.sqrt(np.maximum(variances, 0.0))
    scales = np.where(scales < 1e-12, 1.0, scales)
    return means, scales


def _fit_balanced_pooled_head(
    training_rows: list[dict[str, Any]],
    *,
    l2: float,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not training_rows:
        raise ValueError("pooled transfer training rows are empty")
    datasets = sorted({str(item["dataset"]) for item in training_rows})
    if len(datasets) < MIN_SOURCE_DATASETS:
        raise ValueError("transferred head requires at least three source datasets")

    features = np.vstack([item["features"] for item in training_rows])
    labels = np.asarray(
        [int(item["complete"]) for item in training_rows],
        dtype=np.float64,
    )
    dataset_array = np.asarray(
        [str(item["dataset"]) for item in training_rows],
        dtype=object,
    )
    total = len(training_rows)
    sample_weights = np.zeros(total, dtype=np.float64)
    per_dataset = {}
    for dataset in datasets:
        dataset_mask = dataset_array == dataset
        complete_mask = dataset_mask & (labels == 1.0)
        incomplete_mask = dataset_mask & (labels == 0.0)
        complete_count = int(complete_mask.sum())
        incomplete_count = int(incomplete_mask.sum())
        if not complete_count or not incomplete_count:
            raise ValueError(
                f"source train split lacks both classes for {dataset}"
            )
        class_total_weight = total / (2.0 * len(datasets))
        sample_weights[complete_mask] = class_total_weight / complete_count
        sample_weights[incomplete_mask] = class_total_weight / incomplete_count
        per_dataset[dataset] = {
            "train_variants": int(dataset_mask.sum()),
            "complete_variants": complete_count,
            "incomplete_variants": incomplete_count,
            "total_sample_weight": round(
                float(sample_weights[dataset_mask].sum()),
                12,
            ),
            "complete_sample_weight": round(
                float(sample_weights[complete_mask].sum()),
                12,
            ),
            "incomplete_sample_weight": round(
                float(sample_weights[incomplete_mask].sum()),
                12,
            ),
        }

    means, scales = _weighted_mean_and_scale(features, sample_weights)
    standardized = (features - means) / scales
    design = np.column_stack([np.ones(total), standardized])
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    completed_iterations = 0
    for completed_iterations in range(1, iterations + 1):
        logits = np.clip(design @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weights * (probabilities - labels))
        gradient += penalty @ coefficients
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        hessian += np.eye(design.shape[1]) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    model = {
        "means": means,
        "scales": scales,
        "intercept": float(coefficients[0]),
        "weights": coefficients[1:],
        "l2": l2,
        "iterations": iterations,
    }
    audit = {
        "source_datasets": datasets,
        "source_dataset_count": len(datasets),
        "source_split": "train",
        "training_variants": total,
        "feature_count": features.shape[1],
        "feature_names": list(FEATURE_NAMES),
        "pool_weighting": "equal_dataset_equal_within_dataset_class",
        "sample_weight_sum": round(float(sample_weights.sum()), 12),
        "weighted_standardization": True,
        "completed_irls_iterations": completed_iterations,
        "per_dataset": per_dataset,
    }
    return model, audit


def _variants_for_split(
    prepared_source: dict[str, Any],
    split_version: str,
) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "split": _split_for_case(
                str(item["case_id"]),
                split_version=split_version,
            ),
        }
        for item in prepared_source["variants"]
    ]


def fit_transferred_head(
    prepared_sources: dict[str, dict[str, Any]],
    *,
    target_dataset: str,
    split_version: str,
    l2: float = 4.0,
    iterations: int = 80,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit only on other datasets' train variants for one target and split."""
    if target_dataset not in prepared_sources:
        raise ValueError("target dataset is not present in prepared sources")
    source_datasets = sorted(set(prepared_sources) - {target_dataset})
    if len(source_datasets) < MIN_SOURCE_DATASETS:
        raise ValueError("leave-one-dataset-out transfer requires four datasets")
    rows = []
    for dataset in source_datasets:
        rows.extend(
            {
                **item,
                "dataset": dataset,
            }
            for item in _variants_for_split(
                prepared_sources[dataset],
                split_version,
            )
            if item["split"] == "train"
        )
    model, audit = _fit_balanced_pooled_head(
        rows,
        l2=l2,
        iterations=iterations,
    )
    audit.update(
        {
            "target_dataset": target_dataset,
            "split_version": split_version,
            "target_train_variants_used": 0,
            "target_train_labels_used": False,
            "target_train_features_used": False,
        }
    )
    return model, audit


def _calibrate_target(
    variants: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    rounded_scores = np.asarray(
        [round(float(value), 12) for value in scores],
        dtype=np.float64,
    )
    calibration_maxima: dict[str, float] = {}
    for item, score in zip(variants, rounded_scores, strict=True):
        if item["split"] != "calibration" or bool(item["complete"]):
            continue
        case_id = str(item["case_id"])
        calibration_maxima[case_id] = max(
            calibration_maxima.get(case_id, -math.inf),
            float(score),
        )
    units = np.asarray(list(calibration_maxima.values()), dtype=np.float64)
    threshold = round(_conformal_threshold(units, alpha), 12)
    rank = _conformal_rank(len(units), alpha)
    nominal_bound = round(
        (len(units) + 1 - rank) / (len(units) + 1),
        12,
    )
    return {
        "scores": rounded_scores,
        "threshold": threshold,
        "calibration_case_level_units": len(units),
        "threshold_order_statistic_rank": rank,
        "finite_sample_nominal_case_error_upper_bound": nominal_bound,
    }


def _aggregate_metrics(
    per_repeat: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    return {
        metric: _distribution(repeat[method][metric] for repeat in per_repeat)
        for metric in SUMMARY_METRICS
    }


def _analyze_case_records(
    records: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    alpha: float,
) -> list[dict[str, Any]]:
    analyses = []
    for dataset in datasets:
        per_repeat = []
        for split_version in split_versions:
            current = [
                item
                for item in records
                if item["dataset"] == dataset
                and item["split_version"] == split_version
            ]
            if not current:
                raise ValueError(
                    f"missing transfer records for {dataset}/{split_version}"
                )
            calibration = [
                item
                for item in current
                if item["split"] == "calibration" and not bool(item["complete"])
            ]
            case_maxima: dict[str, float] = {}
            for item in calibration:
                case_id = str(item["case_id"])
                case_maxima[case_id] = max(
                    case_maxima.get(case_id, -math.inf),
                    float(item["transferred_score"]),
                )
            units = np.asarray(list(case_maxima.values()), dtype=np.float64)
            threshold = round(_conformal_threshold(units, alpha), 12)
            rank = _conformal_rank(len(units), alpha)
            nominal_bound = round(
                (len(units) + 1 - rank) / (len(units) + 1),
                12,
            )
            evaluation = [item for item in current if item["split"] == "evaluation"]
            transferred_decisions = np.asarray(
                [float(item["transferred_score"]) > threshold for item in evaluation],
                dtype=bool,
            )
            stored_decisions = np.asarray(
                [bool(item["transferred_declared_complete"]) for item in evaluation],
                dtype=bool,
            )
            if not np.array_equal(transferred_decisions, stored_decisions):
                raise ValueError("stored transferred decisions do not match threshold")
            target_fitted_decisions = np.asarray(
                [bool(item["target_fitted_declared_complete"]) for item in evaluation],
                dtype=bool,
            )
            labels = np.asarray(
                [bool(item["complete"]) for item in evaluation],
                dtype=bool,
            )
            target_metrics = _decision_metrics(
                evaluation,
                target_fitted_decisions,
            )
            transferred_metrics = _decision_metrics(
                evaluation,
                transferred_decisions,
            )
            target_auc = round(
                _auc(
                    labels,
                    np.asarray(
                        [item["target_fitted_score"] for item in evaluation],
                        dtype=np.float64,
                    ),
                ),
                6,
            )
            transferred_auc = round(
                _auc(
                    labels,
                    np.asarray(
                        [item["transferred_score"] for item in evaluation],
                        dtype=np.float64,
                    ),
                ),
                6,
            )
            per_repeat.append(
                {
                    "split_version": split_version,
                    "calibration_case_level_units": len(units),
                    "threshold": threshold,
                    "threshold_order_statistic_rank": rank,
                    "finite_sample_nominal_case_error_upper_bound": nominal_bound,
                    "target_fitted_auc": target_auc,
                    "transferred_auc": transferred_auc,
                    "target_fitted": target_metrics,
                    "transferred": transferred_metrics,
                    "delta_transferred_minus_target_fitted": {
                        metric: round(
                            float(transferred_metrics[metric])
                            - float(target_metrics[metric]),
                            6,
                        )
                        for metric in SUMMARY_METRICS
                    }
                    | {"evaluation_auc": round(transferred_auc - target_auc, 6)},
                }
            )
        analyses.append(
            {
                "dataset": dataset,
                "per_repeat": per_repeat,
                "aggregate": {
                    "target_fitted": _aggregate_metrics(
                        per_repeat,
                        "target_fitted",
                    ),
                    "transferred": _aggregate_metrics(
                        per_repeat,
                        "transferred",
                    ),
                    "target_fitted_auc": _distribution(
                        repeat["target_fitted_auc"] for repeat in per_repeat
                    ),
                    "transferred_auc": _distribution(
                        repeat["transferred_auc"] for repeat in per_repeat
                    ),
                    "delta_transferred_minus_target_fitted": {
                        metric: _distribution(
                            repeat["delta_transferred_minus_target_fitted"][metric]
                            for repeat in per_repeat
                        )
                        for metric in (*SUMMARY_METRICS, "evaluation_auc")
                    },
                    "case_family_rate_at_or_below_alpha_repeats": sum(
                        repeat["transferred"][
                            "false_complete_case_family_rate"
                        ]
                        <= alpha
                        for repeat in per_repeat
                    ),
                    "all_finite_sample_nominal_bounds_at_or_below_alpha": all(
                        repeat[
                            "finite_sample_nominal_case_error_upper_bound"
                        ]
                        <= alpha
                        for repeat in per_repeat
                    ),
                },
            }
        )
    return analyses


def _check(measured: Any, target: Any, passed: bool) -> dict[str, Any]:
    return {"measured": measured, "target": target, "passed": bool(passed)}


def _validate_training_audit(
    training_audit: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
) -> bool:
    expected_pairs = {
        (dataset, split_version)
        for dataset in datasets
        for split_version in split_versions
    }
    observed_pairs = {
        (str(item["target_dataset"]), str(item["split_version"]))
        for item in training_audit
    }
    return observed_pairs == expected_pairs and all(
        item["source_datasets"]
        == sorted(set(datasets) - {item["target_dataset"]})
        and int(item["source_dataset_count"]) == len(datasets) - 1
        and item["source_split"] == "train"
        and int(item["target_train_variants_used"]) == 0
        and item["target_train_labels_used"] is False
        and item["target_train_features_used"] is False
        and item["weighted_standardization"] is True
        and item["pool_weighting"]
        == "equal_dataset_equal_within_dataset_class"
        for item in training_audit
    )


def _build_outcome(
    analyses: list[dict[str, Any]],
    training_audit: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    alpha: float,
    adoption_targets: dict[str, float | int],
) -> dict[str, Any]:
    training_valid = _validate_training_audit(
        training_audit,
        datasets=datasets,
        split_versions=split_versions,
    )
    mean_risks = {
        item["dataset"]: item["aggregate"]["transferred"][
            "false_complete_case_family_rate"
        ]["mean"]
        for item in analyses
    }
    risk_increases = {
        item["dataset"]: item["aggregate"][
            "delta_transferred_minus_target_fitted"
        ]["false_complete_case_family_rate"]["mean"]
        for item in analyses
    }
    recall_decreases = {
        item["dataset"]: round(
            -float(
                item["aggregate"]["delta_transferred_minus_target_fitted"][
                    "true_complete_declaration_rate"
                ]["mean"]
            ),
            6,
        )
        for item in analyses
    }
    mean_auc_decrease = round(
        statistics.fmean(
            -float(
                item["aggregate"]["delta_transferred_minus_target_fitted"][
                    "evaluation_auc"
                ]["mean"]
            )
            for item in analyses
        ),
        6,
    )
    alpha_control_repeats = {
        item["dataset"]: int(
            item["aggregate"][
                "case_family_rate_at_or_below_alpha_repeats"
            ]
        )
        for item in analyses
    }
    datasets_meeting_repeat_target = sum(
        value >= 7 for value in alpha_control_repeats.values()
    )
    all_bounds = all(
        item["aggregate"][
            "all_finite_sample_nominal_bounds_at_or_below_alpha"
        ]
        for item in analyses
    )
    checks = {
        "target_train_exclusion_and_source_train_only": _check(
            training_valid,
            True,
            training_valid,
        ),
        "target_threshold_uses_calibration_only": _check(True, True, True),
        "no_evaluation_selection": _check(True, True, True),
        "maximum_mean_case_family_false_complete_rate_each_dataset": _check(
            mean_risks,
            "<="
            + str(
                adoption_targets[
                    "maximum_mean_case_family_false_complete_rate_each_dataset"
                ]
            ),
            all(
                value
                <= adoption_targets[
                    "maximum_mean_case_family_false_complete_rate_each_dataset"
                ]
                for value in mean_risks.values()
            ),
        ),
        "maximum_dataset_case_family_risk_increase_vs_target_fitted": _check(
            risk_increases,
            "<="
            + str(
                adoption_targets[
                    "maximum_dataset_case_family_risk_increase_vs_target_fitted"
                ]
            ),
            all(
                value
                <= adoption_targets[
                    "maximum_dataset_case_family_risk_increase_vs_target_fitted"
                ]
                for value in risk_increases.values()
            ),
        ),
        "maximum_dataset_complete_recall_decrease_vs_target_fitted": _check(
            recall_decreases,
            "<="
            + str(
                adoption_targets[
                    "maximum_dataset_complete_recall_decrease_vs_target_fitted"
                ]
            ),
            all(
                value
                <= adoption_targets[
                    "maximum_dataset_complete_recall_decrease_vs_target_fitted"
                ]
                for value in recall_decreases.values()
            ),
        ),
        "cross_dataset_mean_evaluation_auc_decrease_vs_target_fitted": _check(
            mean_auc_decrease,
            "<="
            + str(
                adoption_targets[
                    "maximum_cross_dataset_mean_evaluation_auc_decrease_vs_target_fitted"
                ]
            ),
            mean_auc_decrease
            <= adoption_targets[
                "maximum_cross_dataset_mean_evaluation_auc_decrease_vs_target_fitted"
            ],
        ),
        "datasets_with_at_least_7_of_10_alpha_control_repeats": _check(
            {
                "count": datasets_meeting_repeat_target,
                "per_dataset": alpha_control_repeats,
            },
            ">="
            + str(
                adoption_targets[
                    "minimum_datasets_with_at_least_7_of_10_alpha_control_repeats"
                ]
            ),
            datasets_meeting_repeat_target
            >= adoption_targets[
                "minimum_datasets_with_at_least_7_of_10_alpha_control_repeats"
            ],
        ),
        "all_finite_sample_nominal_bounds_at_or_below_alpha": _check(
            all_bounds,
            True,
            all_bounds,
        ),
    }
    all_passed = all(item["passed"] for item in checks.values())
    return {
        "status": (
            "POST_HOC_TRANSFER_CANDIDATE" if all_passed else "DO_NOT_ADOPT"
        ),
        "post_hoc_method_development": True,
        "independent_confirmation": False,
        "target_train_labels_used_for_transferred_head": False,
        "alpha": alpha,
        "checks": checks,
        "all_pre_registered_adoption_checks_passed": all_passed,
        "conditional_subgroup_guarantee_claimed": False,
        "zero_shot_claimed": False,
        "gate_2": "NO-GO/SHADOW",
    }


def evaluate_cross_dataset_head_transfer(
    dataset_sources: dict[str, Path],
    expected_robustness: dict[str, tuple[Path, dict[str, Any]]],
    *,
    reference_run_report: Path | None = None,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    alpha: float = PRIMARY_ALPHA,
    l2: float = 4.0,
    iterations: int = 80,
    adoption_targets: dict[str, float | int] = ADOPTION_TARGETS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(dataset_sources) != set(expected_robustness):
        raise ValueError("dataset sources and robustness reports must match")
    if len(dataset_sources) < MIN_SOURCE_DATASETS + 1:
        raise ValueError("head transfer audit requires at least four datasets")
    if len(set(split_versions)) != len(split_versions):
        raise ValueError("split versions must be unique")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    prepared_sources: dict[str, dict[str, Any]] = {}
    provenance = []
    for dataset in sorted(dataset_sources):
        source_path = dataset_sources[dataset]
        expected_path, expected_report = expected_robustness[dataset]
        metadata = expected_report["metadata"]
        if tuple(metadata["split_versions"]) != split_versions:
            raise ValueError("expected robustness split versions do not match")
        if float(metadata["primary_alpha"]) != alpha:
            raise ValueError("expected robustness alpha does not match")
        selector = metadata["selector"]
        prepared = prepare_conformal_source(
            source_path,
            k=int(selector["top_k"]),
            token_budget=int(selector["token_budget"]),
            role_threshold=float(selector["role_threshold"]),
        )
        if prepared["dataset"] != dataset:
            raise ValueError("prepared dataset name does not match source key")
        if metadata["source_sha256"] != prepared["source_sha256"]:
            raise ValueError("expected robustness source hash does not match")
        prepared_sources[dataset] = prepared
        provenance.append(
            {
                "dataset": dataset,
                "source_path_label": source_path.name,
                "source_sha256": prepared["source_sha256"],
                "eligible_cases": len(
                    {item["case_id"] for item in prepared["variants"]}
                ),
                "reference_report_path_label": expected_path.name,
                "reference_report_sha256": sha256(expected_path),
                "reference_repeats_exactly_reproduced": True,
            }
        )

    case_records: list[dict[str, Any]] = []
    training_audit: list[dict[str, Any]] = []
    transferred_models: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target_dataset in sorted(dataset_sources):
        source_path = dataset_sources[target_dataset]
        _, expected_report = expected_robustness[target_dataset]
        selector = expected_report["metadata"]["selector"]
        reference_alphas = tuple(
            float(value) for value in expected_report["metadata"]["alphas"]
        )
        generated_reference_repeats = []
        prepared_target = prepared_sources[target_dataset]
        for split_version in split_versions:
            target_fitted, target_fitted_records = evaluate_conformal_sufficiency(
                source_path,
                reference_run_report=reference_run_report,
                split_version=split_version,
                alphas=reference_alphas,
                primary_alpha=alpha,
                k=int(selector["top_k"]),
                token_budget=int(selector["token_budget"]),
                role_threshold=float(selector["role_threshold"]),
                l2=l2,
                iterations=iterations,
                prepared_source=prepared_target,
            )
            generated_reference_repeats.append(
                _repeat_core(target_fitted, split_version=split_version)
            )
            model, audit = fit_transferred_head(
                prepared_sources,
                target_dataset=target_dataset,
                split_version=split_version,
                l2=l2,
                iterations=iterations,
            )
            training_audit.append(audit)
            transferred_models[target_dataset].append(
                {
                    "split_version": split_version,
                    "model": _serializable_model(model),
                }
            )
            target_variants = _variants_for_split(
                prepared_target,
                split_version,
            )
            transferred_scores = _predict(
                model,
                np.vstack([item["features"] for item in target_variants]),
            )
            calibration = _calibrate_target(
                target_variants,
                transferred_scores,
                alpha=alpha,
            )
            target_fitted_by_key = {
                (str(item["case_id"]), float(item["target_missing_ratio"])): item
                for item in target_fitted_records
            }
            for variant, transferred_score in zip(
                target_variants,
                calibration["scores"],
                strict=True,
            ):
                key = (
                    str(variant["case_id"]),
                    float(variant["target_missing_ratio"]),
                )
                fitted = target_fitted_by_key[key]
                if (
                    fitted["split"] != variant["split"]
                    or bool(fitted["complete"]) != bool(variant["complete"])
                ):
                    raise AssertionError("target reference variant does not match")
                is_evaluation = variant["split"] == "evaluation"
                transferred_declared = (
                    float(transferred_score) > calibration["threshold"]
                    if is_evaluation
                    else None
                )
                case_records.append(
                    {
                        "dataset": target_dataset,
                        "split_version": split_version,
                        "case_id": variant["case_id"],
                        "split": variant["split"],
                        "question_type": variant["question_type"],
                        "required_role_count": str(
                            variant["required_role_count"]
                        ),
                        "candidate_count_bucket": variant[
                            "candidate_count_bucket"
                        ],
                        "target_missing_ratio": variant[
                            "target_missing_ratio"
                        ],
                        "complete": variant["complete"],
                        "source_missing": variant["source_missing"],
                        "target_fitted_score": fitted["sufficiency_score"],
                        "transferred_score": float(transferred_score),
                        "target_fitted_declared_complete": (
                            fitted["conformal_declared_complete"]
                            if is_evaluation
                            else None
                        ),
                        "transferred_declared_complete": transferred_declared,
                        "transferred_threshold": calibration["threshold"],
                        "threshold_source": "target_calibration_only",
                        "target_train_used_for_head": False,
                    }
                )
        if generated_reference_repeats != expected_report["repeats"]:
            raise ValueError(
                f"generated repeats do not match {target_dataset} reference report"
            )

    datasets = sorted(dataset_sources)
    analyses = _analyze_case_records(
        case_records,
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
    )
    outcome = _build_outcome(
        analyses,
        training_audit,
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
        adoption_targets=adoption_targets,
    )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "datasets": datasets,
            "dataset_count": len(datasets),
            "split_versions": list(split_versions),
            "repeat_count_per_dataset": len(split_versions),
            "primary_alpha": alpha,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "l2": l2,
            "iterations": iterations,
            "case_artifact": {},
        },
        "provenance": provenance,
        "development_protocol": {
            "post_hoc": True,
            "independent_confirmation": False,
            "target_train_labels_or_features_used_for_transferred_head": False,
            "source_training_uses_train_splits_only": True,
            "source_datasets_equal_weight": True,
            "within_source_classes_equal_weight": True,
            "weighted_standardization_uses_source_train_only": True,
            "target_threshold_uses_calibration_only": True,
            "evaluation_used_for_model_threshold_or_selection": False,
            "target_calibration_labels_used": True,
            "zero_shot_claimed": False,
            "adoption_targets": adoption_targets,
        },
        "training_audit": training_audit,
        "transferred_models": dict(transferred_models),
        "datasets": analyses,
        "outcome": outcome,
        "theoretical_scope": {
            "head_parameter_transfer_tested": True,
            "zero_shot_transfer_tested": False,
            "reason_not_zero_shot": (
                "Target calibration labels set the target conformal threshold."
            ),
            "target_marginal_statement": (
                "The recorded finite-sample marginal bound requires target "
                "calibration and evaluation case-family exchangeability."
            ),
            "conditional_subgroup_guarantee_claimed": False,
        },
        "decision": {
            "finding": (
                "This post-hoc audit isolates cross-dataset head-parameter "
                "transfer from the existing per-target refit results."
            ),
            "next_step": (
                "FREEZE_BEFORE_NEW_UNTOUCHED_CONFIRMATION"
                if outcome["all_pre_registered_adoption_checks_passed"]
                else "STOP_TRANSFER_METHOD_SELECTION_WITHOUT_NEW_DATA"
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "KEEP_TARGET_FITTED_GLOBAL_METHOD_IN_SHADOW",
            "limitations": [
                "All four public datasets had already been inspected before this method-development audit.",
                "Target calibration labels are required, so the experiment is not zero-shot.",
                "Public QA datasets are not district flood-response corpora.",
                "No post-hoc result authorizes CANARY or DEFAULT.",
            ],
        },
    }
    return report, case_records


def render_cross_dataset_head_transfer_markdown(
    report: dict[str, Any],
) -> str:
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG 跨数据集充分性头迁移审计",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 数据集：{', '.join(metadata['datasets'])}",
        f"- 重复分组：{metadata['repeat_count_per_dataset']}",
        f"- alpha：{metadata['primary_alpha']}",
        "- 目标 train 标签/特征用于迁移头：`False`",
        "- 目标 calibration 标签用于阈值：`True`",
        "- 独立确认：`False`；零样本：`False`",
        f"- Gate 2：`{outcome['gate_2']}`",
        "",
        "## 与目标集重新拟合头比较",
        "",
        (
            "| 数据集 | 目标拟合风险 | 迁移风险 | 目标拟合完整召回 | "
            "迁移完整召回 | 目标拟合 AUC | 迁移 AUC | alpha 达标次数 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        aggregate = item["aggregate"]
        lines.append(
            f"| {item['dataset']} | "
            f"{aggregate['target_fitted']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['transferred']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['target_fitted']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{aggregate['transferred']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{aggregate['target_fitted_auc']['mean']:.6f} | "
            f"{aggregate['transferred_auc']['mean']:.6f} | "
            f"{aggregate['case_family_rate_at_or_below_alpha_repeats']}/"
            f"{metadata['repeat_count_per_dataset']} |"
        )
    lines.extend(
        [
            "",
            "## 预注册采用检查",
            "",
            "| 检查 | 实测 | 目标 | 通过 |",
            "|---|---|---|---|",
        ]
    )
    for name, check in outcome["checks"].items():
        measured = json.dumps(check["measured"], ensure_ascii=False)
        lines.append(
            f"| {name} | `{measured}` | `{check['target']}` | "
            f"`{check['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            (
                "该审计只检验目标 train 标签隔离后的模型参数迁移；目标 calibration "
                "仍使用标签设阈值，因此不是零样本。四个数据集结果均已在方法设计前可见，"
                "故结果属于事后开发，不能改变 Gate 2。"
            ),
            "",
            f"- 下一步：`{report['decision']['next_step']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_cross_dataset_head_transfer(
    report: dict[str, Any],
    case_records: list[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    cases_path: Path,
) -> tuple[Path, Path, Path]:
    for path in (json_path, markdown_path, cases_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in case_records
    ).encode("utf-8")
    with cases_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as archive:
            archive.write(payload)
    report["metadata"]["case_artifact"] = {
        "path_label": cases_path.name,
        "sha256": sha256(cases_path),
        "record_count": len(case_records),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_cross_dataset_head_transfer_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path, cases_path


def load_cross_dataset_head_transfer(
    json_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported cross-dataset head transfer schema")
    artifact = metadata["case_artifact"]
    if artifact["path_label"] != cases_path.name:
        raise ValueError("head transfer case artifact path label does not match")
    if artifact["sha256"] != sha256(cases_path):
        raise ValueError("head transfer case artifact hash does not match")
    records = []
    with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != artifact["record_count"]:
        raise ValueError("head transfer case artifact record count does not match")
    datasets = list(metadata["datasets"])
    split_versions = tuple(metadata["split_versions"])
    alpha = float(metadata["primary_alpha"])
    analyses = _analyze_case_records(
        records,
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
    )
    outcome = _build_outcome(
        analyses,
        report["training_audit"],
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
        adoption_targets={
            key: value
            for key, value in report["development_protocol"][
                "adoption_targets"
            ].items()
        },
    )
    if analyses != report.get("datasets") or outcome != report.get("outcome"):
        raise ValueError("head transfer aggregates do not match case artifact")
    return report
