"""Train-only observable-context development for the FRC sufficiency head."""

from __future__ import annotations

import gzip
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_subgroup_audit import (
    DIMENSIONS as REPORTING_DIMENSIONS,
    _aggregate_subgroups,
    _repeat_core,
    _repeat_subgroups,
)
from research.frc_rag.conformal_sufficiency import (
    FEATURE_NAMES,
    PRIMARY_ALPHA,
    _conformal_rank,
    _conformal_threshold,
    _decision_metrics,
    _fit_weighted_logistic,
    _predict,
    _split_for_case,
    evaluate_conformal_sufficiency,
    prepare_conformal_source,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-contextual-head-development-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_POST_HOC_CONTEXTUAL_HEAD_DEVELOPMENT"
MIN_TRAIN_CATEGORY_CASES = 10
MIN_INCOMPLETE_CASE_FAMILIES = 30
MIN_ELIGIBLE_REPEATS = 7
CONTEXT_DIMENSIONS = (
    "question_type",
    "required_role_count",
    "question_type_x_required_role_count",
)
SUMMARY_METRICS = (
    "false_complete_case_family_rate",
    "abstention_rate",
    "true_complete_declaration_rate",
)
ADOPTION_TARGETS = {
    "minimum_mean_complete_recall_gain": 0.02,
    "maximum_dataset_complete_recall_decrease": 0.01,
    "maximum_dataset_overall_case_risk_increase": 0.01,
    "minimum_mean_worst_eligible_subgroup_risk_reduction": 0.02,
}


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "sample_std": (
            round(statistics.stdev(ordered), 6)
            if len(ordered) > 1
            else 0.0
        ),
    }


def _context_values(record: dict[str, Any]) -> dict[str, str]:
    question_type = str(record["question_type"])
    required_role_count = str(record["required_role_count"])
    return {
        "question_type": question_type,
        "required_role_count": required_role_count,
        "question_type_x_required_role_count": (
            f"question_type={question_type}|"
            f"required_role_count={required_role_count}"
        ),
    }


def _fit_context_encoder(
    train_records: list[dict[str, Any]],
    *,
    min_train_category_cases: int,
) -> dict[str, Any]:
    by_case: dict[str, dict[str, str]] = {}
    for record in train_records:
        case_id = str(record["case_id"])
        values = _context_values(record)
        previous = by_case.setdefault(case_id, values)
        if previous != values:
            raise ValueError("context attributes vary within a train case")
    if not by_case:
        raise ValueError("train split has no case families")
    vocabularies: dict[str, list[str]] = {}
    category_case_counts: dict[str, dict[str, int]] = {}
    for dimension in CONTEXT_DIMENSIONS:
        counts = Counter(values[dimension] for values in by_case.values())
        category_case_counts[dimension] = {
            key: counts[key] for key in sorted(counts)
        }
        vocabularies[dimension] = [
            key
            for key in sorted(counts)
            if counts[key] >= min_train_category_cases
        ]
    feature_names = []
    for dimension in CONTEXT_DIMENSIONS:
        feature_names.extend(
            f"context::{dimension}::{value}"
            for value in vocabularies[dimension]
        )
        feature_names.append(f"context::{dimension}::__other__")
    return {
        "source_split": "train",
        "train_case_families": len(by_case),
        "minimum_train_category_case_families": min_train_category_cases,
        "vocabularies": vocabularies,
        "category_train_case_counts": category_case_counts,
        "feature_names": feature_names,
    }


def _encode_context(
    records: list[dict[str, Any]],
    encoder: dict[str, Any],
) -> np.ndarray:
    encoded = np.zeros(
        (len(records), len(encoder["feature_names"])),
        dtype=np.float64,
    )
    offsets: dict[str, int] = {}
    offset = 0
    for dimension in CONTEXT_DIMENSIONS:
        offsets[dimension] = offset
        offset += len(encoder["vocabularies"][dimension]) + 1
    for row_index, record in enumerate(records):
        values = _context_values(record)
        for dimension in CONTEXT_DIMENSIONS:
            vocabulary = encoder["vocabularies"][dimension]
            try:
                category_index = vocabulary.index(values[dimension])
            except ValueError:
                category_index = len(vocabulary)
            encoded[row_index, offsets[dimension] + category_index] = 1.0
    return encoded


def _feature_matrix(
    records: list[dict[str, Any]],
    encoder: dict[str, Any],
) -> np.ndarray:
    base = np.vstack([record["features"] for record in records])
    return np.column_stack([base, _encode_context(records, encoder)])


def _serializable_model(
    model: dict[str, Any],
    encoder: dict[str, Any],
) -> dict[str, Any]:
    return {
        "feature_names": [*FEATURE_NAMES, *encoder["feature_names"]],
        "base_feature_count": len(FEATURE_NAMES),
        "context_feature_count": len(encoder["feature_names"]),
        "scaler_mean": [round(float(value), 12) for value in model["means"]],
        "scaler_scale": [round(float(value), 12) for value in model["scales"]],
        "intercept": round(float(model["intercept"]), 12),
        "weights": [round(float(value), 12) for value in model["weights"]],
        "l2": model["l2"],
        "iterations": model["iterations"],
        "optimizer": "deterministic class-balanced IRLS",
    }


def _calibration_case_maxima(
    records: list[dict[str, Any]],
) -> np.ndarray:
    maxima: dict[str, float] = {}
    for record in records:
        if bool(record["complete"]):
            continue
        case_id = str(record["case_id"])
        maxima[case_id] = max(
            maxima.get(case_id, -np.inf),
            float(record["contextual_score"]),
        )
    if not maxima:
        raise ValueError("calibration split has no incomplete case families")
    return np.asarray(list(maxima.values()), dtype=np.float64)


def _nominal_bound(calibration_units: int, alpha: float) -> float:
    rank = _conformal_rank(calibration_units, alpha)
    return round(
        (calibration_units + 1 - rank) / (calibration_units + 1),
        12,
    )


def _method_metrics(
    records: list[dict[str, Any]],
    decision_field: str,
) -> dict[str, Any]:
    decisions = np.asarray(
        [bool(record[decision_field]) for record in records],
        dtype=bool,
    )
    return _decision_metrics(records, decisions)


def _method_subgroups(
    records: list[dict[str, Any]],
    decision_field: str,
    *,
    min_incomplete_case_families: int,
) -> list[dict[str, Any]]:
    projected = [
        {
            **record,
            "conformal_declared_complete": bool(record[decision_field]),
        }
        for record in records
    ]
    return _repeat_subgroups(
        projected,
        min_incomplete_case_families=min_incomplete_case_families,
    )


def _aggregate_metric_rows(
    rows: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    return {
        metric: _distribution(row[method][metric] for row in rows)
        for metric in SUMMARY_METRICS
    }


def _validate_case_records(case_records: list[dict[str, Any]]) -> bool:
    thresholds: dict[tuple[str, str], float] = {}
    for record in case_records:
        if not bool(record["context_vocabulary_train_only"]):
            return False
        if not bool(record["threshold_calibration_only"]):
            return False
        key = (str(record["dataset"]), str(record["split_version"]))
        threshold = float(record["contextual_threshold"])
        previous = thresholds.setdefault(key, threshold)
        if previous != threshold:
            return False
    return bool(thresholds)


def _analyze_case_records(
    case_records: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    alpha: float,
    min_incomplete_case_families: int,
    min_eligible_repeats: int,
) -> list[dict[str, Any]]:
    analyses = []
    for dataset in datasets:
        per_repeat = []
        for split_version in split_versions:
            records = [
                record
                for record in case_records
                if record["dataset"] == dataset
                and record["split_version"] == split_version
            ]
            if not records:
                raise ValueError(
                    f"missing contextual case records for {dataset}/{split_version}"
                )
            baseline = _method_metrics(
                records,
                "baseline_global_declared_complete",
            )
            contextual = _method_metrics(
                records,
                "contextual_declared_complete",
            )
            per_repeat.append(
                {
                    "split_version": split_version,
                    "baseline_global": baseline,
                    "contextual_global": contextual,
                    "delta_contextual_minus_baseline": {
                        metric: round(
                            float(contextual[metric])
                            - float(baseline[metric]),
                            6,
                        )
                        for metric in SUMMARY_METRICS
                    },
                    "baseline_subgroups": _method_subgroups(
                        records,
                        "baseline_global_declared_complete",
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                    "contextual_subgroups": _method_subgroups(
                        records,
                        "contextual_declared_complete",
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                }
            )
        baseline_subgroups = _aggregate_subgroups(
            [
                {"subgroups": repeat["baseline_subgroups"]}
                for repeat in per_repeat
            ],
            alpha=alpha,
            min_eligible_repeats=min_eligible_repeats,
        )
        contextual_subgroups = _aggregate_subgroups(
            [
                {"subgroups": repeat["contextual_subgroups"]}
                for repeat in per_repeat
            ],
            alpha=alpha,
            min_eligible_repeats=min_eligible_repeats,
        )
        baseline_worst = baseline_subgroups["worst_eligible_subgroup"]
        contextual_worst = contextual_subgroups["worst_eligible_subgroup"]
        worst_reduction = (
            round(
                float(
                    baseline_worst["mean_false_complete_case_family_rate"]
                )
                - float(
                    contextual_worst[
                        "mean_false_complete_case_family_rate"
                    ]
                ),
                6,
            )
            if baseline_worst is not None and contextual_worst is not None
            else None
        )
        analyses.append(
            {
                "dataset": dataset,
                "per_repeat": per_repeat,
                "aggregate": {
                    "baseline_global": _aggregate_metric_rows(
                        per_repeat,
                        "baseline_global",
                    ),
                    "contextual_global": _aggregate_metric_rows(
                        per_repeat,
                        "contextual_global",
                    ),
                    "delta_contextual_minus_baseline": {
                        metric: _distribution(
                            repeat["delta_contextual_minus_baseline"][metric]
                            for repeat in per_repeat
                        )
                        for metric in SUMMARY_METRICS
                    },
                    "baseline_subgroups": baseline_subgroups,
                    "contextual_subgroups": contextual_subgroups,
                    "worst_eligible_subgroup_risk_reduction": worst_reduction,
                },
            }
        )
    if not _validate_case_records(case_records):
        raise ValueError("contextual case records violate train/calibration scope")
    return analyses


def _check(measured: Any, target: Any, passed: bool) -> dict[str, Any]:
    return {
        "measured": measured,
        "target": target,
        "passed": bool(passed),
    }


def _build_outcome(
    analyses: list[dict[str, Any]],
    case_records: list[dict[str, Any]],
    *,
    alpha: float,
    adoption_targets: dict[str, float],
) -> dict[str, Any]:
    mean_recall_gain = round(
        statistics.fmean(
            float(
                item["aggregate"]["delta_contextual_minus_baseline"][
                    "true_complete_declaration_rate"
                ]["mean"]
            )
            for item in analyses
        ),
        6,
    )
    recall_decreases = {
        item["dataset"]: round(
            -float(
                item["aggregate"]["delta_contextual_minus_baseline"][
                    "true_complete_declaration_rate"
                ]["mean"]
            ),
            6,
        )
        for item in analyses
    }
    risk_increases = {
        item["dataset"]: float(
            item["aggregate"]["delta_contextual_minus_baseline"][
                "false_complete_case_family_rate"
            ]["mean"]
        )
        for item in analyses
    }
    worst_reductions = [
        float(item["aggregate"]["worst_eligible_subgroup_risk_reduction"])
        for item in analyses
        if item["aggregate"]["worst_eligible_subgroup_risk_reduction"]
        is not None
    ]
    mean_worst_reduction = (
        round(statistics.fmean(worst_reductions), 6)
        if len(worst_reductions) == len(analyses) and worst_reductions
        else None
    )
    eligible_subgroups = sum(
        item["aggregate"]["contextual_subgroups"]["eligible_subgroup_count"]
        for item in analyses
    )
    all_subgroup_means = bool(eligible_subgroups) and all(
        item["aggregate"]["contextual_subgroups"][
            "all_eligible_subgroup_mean_risks_at_or_below_alpha"
        ]
        for item in analyses
    )
    all_subgroup_repeats = bool(eligible_subgroups) and all(
        item["aggregate"]["contextual_subgroups"][
            "all_eligible_subgroup_repeat_consistency_targets_met"
        ]
        for item in analyses
    )
    scope_valid = _validate_case_records(case_records)
    checks = {
        "no_evaluation_selection": _check(True, True, True),
        "context_vocabulary_train_only": _check(
            scope_valid,
            True,
            scope_valid,
        ),
        "threshold_calibration_only": _check(
            scope_valid,
            True,
            scope_valid,
        ),
        "mean_complete_recall_gain": _check(
            mean_recall_gain,
            (
                ">="
                + str(adoption_targets["minimum_mean_complete_recall_gain"])
            ),
            mean_recall_gain
            >= adoption_targets["minimum_mean_complete_recall_gain"],
        ),
        "maximum_dataset_complete_recall_decrease": _check(
            recall_decreases,
            (
                "<="
                + str(
                    adoption_targets[
                        "maximum_dataset_complete_recall_decrease"
                    ]
                )
            ),
            all(
                value
                <= adoption_targets[
                    "maximum_dataset_complete_recall_decrease"
                ]
                for value in recall_decreases.values()
            ),
        ),
        "maximum_dataset_overall_case_risk_increase": _check(
            risk_increases,
            (
                "<="
                + str(
                    adoption_targets[
                        "maximum_dataset_overall_case_risk_increase"
                    ]
                )
            ),
            all(
                value
                <= adoption_targets[
                    "maximum_dataset_overall_case_risk_increase"
                ]
                for value in risk_increases.values()
            ),
        ),
        "mean_worst_eligible_subgroup_risk_reduction": _check(
            mean_worst_reduction,
            (
                ">="
                + str(
                    adoption_targets[
                        "minimum_mean_worst_eligible_subgroup_risk_reduction"
                    ]
                )
            ),
            mean_worst_reduction is not None
            and mean_worst_reduction
            >= adoption_targets[
                "minimum_mean_worst_eligible_subgroup_risk_reduction"
            ],
        ),
        "all_eligible_subgroup_mean_risks_at_or_below_alpha": _check(
            all_subgroup_means,
            True,
            all_subgroup_means,
        ),
        "all_eligible_subgroup_repeat_consistency_targets_met": _check(
            all_subgroup_repeats,
            True,
            all_subgroup_repeats,
        ),
    }
    all_passed = all(check["passed"] for check in checks.values())
    return {
        "status": (
            "CONTEXTUAL_HEAD_CANDIDATE" if all_passed else "DO_NOT_ADOPT"
        ),
        "post_hoc_method_development": True,
        "independent_confirmation": False,
        "eligible_subgroup_count": eligible_subgroups,
        "alpha": alpha,
        "checks": checks,
        "all_pre_registered_adoption_checks_passed": all_passed,
        "conditional_subgroup_guarantee_claimed": False,
        "gate_2": "NO-GO/SHADOW",
    }


def evaluate_contextual_conformal(
    dataset_sources: dict[str, Path],
    expected_robustness: dict[str, tuple[Path, dict[str, Any]]],
    *,
    reference_run_report: Path | None = None,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    alpha: float = PRIMARY_ALPHA,
    min_train_category_cases: int = MIN_TRAIN_CATEGORY_CASES,
    min_incomplete_case_families: int = MIN_INCOMPLETE_CASE_FAMILIES,
    min_eligible_repeats: int = MIN_ELIGIBLE_REPEATS,
    adoption_targets: dict[str, float] = ADOPTION_TARGETS,
    l2: float = 4.0,
    iterations: int = 80,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(dataset_sources) != set(expected_robustness):
        raise ValueError("dataset sources and robustness reports must match")
    if len(dataset_sources) < 2:
        raise ValueError("contextual development requires at least two datasets")
    if len(set(split_versions)) != len(split_versions):
        raise ValueError("split versions must be unique")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if min_train_category_cases < 1:
        raise ValueError("minimum train category cases must be positive")
    if min_incomplete_case_families < 1:
        raise ValueError("minimum incomplete case families must be positive")
    if not 1 <= min_eligible_repeats <= len(split_versions):
        raise ValueError("minimum eligible repeats is outside split count")

    case_records: list[dict[str, Any]] = []
    training_repeats: dict[str, list[dict[str, Any]]] = {}
    provenance = []
    for dataset in sorted(dataset_sources):
        source_path = dataset_sources[dataset]
        expected_path, expected_report = expected_robustness[dataset]
        if tuple(expected_report["metadata"]["split_versions"]) != split_versions:
            raise ValueError("expected robustness split versions do not match")
        if float(expected_report["metadata"]["primary_alpha"]) != alpha:
            raise ValueError("expected robustness alpha does not match")
        selector = expected_report["metadata"]["selector"]
        top_k = int(selector["top_k"])
        token_budget = int(selector["token_budget"])
        role_threshold = float(selector["role_threshold"])
        prepared = prepare_conformal_source(
            source_path,
            k=top_k,
            token_budget=token_budget,
            role_threshold=role_threshold,
        )
        if prepared["dataset"] != dataset:
            raise ValueError(
                f"source dataset {prepared['dataset']} does not match {dataset}"
            )
        if expected_report["metadata"]["dataset"] != dataset:
            raise ValueError("expected robustness dataset does not match source")
        if (
            expected_report["metadata"]["source_sha256"]
            != prepared["source_sha256"]
        ):
            raise ValueError("expected robustness source hash does not match")

        generated_repeats = []
        dataset_training = []
        for split_version in split_versions:
            baseline_report, baseline_records = evaluate_conformal_sufficiency(
                source_path,
                reference_run_report=reference_run_report,
                split_version=split_version,
                k=top_k,
                token_budget=token_budget,
                role_threshold=role_threshold,
                prepared_source=prepared,
            )
            generated_repeats.append(
                _repeat_core(baseline_report, split_version=split_version)
            )
            variants = [
                {
                    **record,
                    "split": _split_for_case(
                        str(record["case_id"]),
                        split_version=split_version,
                    ),
                }
                for record in prepared["variants"]
            ]
            by_split = {
                split: [record for record in variants if record["split"] == split]
                for split in ("train", "calibration", "evaluation")
            }
            if any(not records for records in by_split.values()):
                raise ValueError("all contextual splits must be non-empty")
            encoder = _fit_context_encoder(
                by_split["train"],
                min_train_category_cases=min_train_category_cases,
            )
            train_labels = np.asarray(
                [int(record["complete"]) for record in by_split["train"]],
                dtype=np.float64,
            )
            model = _fit_weighted_logistic(
                _feature_matrix(by_split["train"], encoder),
                train_labels,
                l2=l2,
                iterations=iterations,
            )
            for split_records in by_split.values():
                scores = _predict(
                    model,
                    _feature_matrix(split_records, encoder),
                )
                for record, score in zip(split_records, scores, strict=True):
                    record["contextual_score"] = float(score)
            calibration_units = _calibration_case_maxima(
                by_split["calibration"]
            )
            threshold = _conformal_threshold(calibration_units, alpha)
            evaluation = by_split["evaluation"]
            contextual_decisions = np.asarray(
                [float(record["contextual_score"]) > threshold for record in evaluation]
            )
            baseline_by_key = {
                (str(record["case_id"]), float(record["target_missing_ratio"])): record
                for record in baseline_records
                if record["split"] == "evaluation"
            }
            if len(baseline_by_key) != len(evaluation):
                raise AssertionError("baseline/contextual evaluation cases differ")
            for record, declared in zip(
                evaluation,
                contextual_decisions,
                strict=True,
            ):
                key = (
                    str(record["case_id"]),
                    float(record["target_missing_ratio"]),
                )
                baseline = baseline_by_key[key]
                case_records.append(
                    {
                        "dataset": dataset,
                        "split_version": split_version,
                        "case_id": record["case_id"],
                        "question_type": record["question_type"],
                        "required_role_count": str(
                            record["required_role_count"]
                        ),
                        "candidate_count_bucket": record[
                            "candidate_count_bucket"
                        ],
                        "target_missing_ratio": record[
                            "target_missing_ratio"
                        ],
                        "complete": record["complete"],
                        "source_missing": record["source_missing"],
                        "baseline_global_score": baseline[
                            "sufficiency_score"
                        ],
                        "contextual_score": round(
                            float(record["contextual_score"]),
                            12,
                        ),
                        "baseline_global_declared_complete": baseline[
                            "conformal_declared_complete"
                        ],
                        "contextual_declared_complete": bool(declared),
                        "contextual_threshold": round(threshold, 12),
                        "contextual_calibration_case_level_units": len(
                            calibration_units
                        ),
                        "contextual_threshold_order_statistic_rank": (
                            _conformal_rank(len(calibration_units), alpha)
                        ),
                        "contextual_finite_sample_nominal_case_error_upper_bound": (
                            _nominal_bound(len(calibration_units), alpha)
                        ),
                        "context_vocabulary_train_only": True,
                        "threshold_calibration_only": True,
                    }
                )
            dataset_training.append(
                {
                    "split_version": split_version,
                    "context_encoder": encoder,
                    "model": _serializable_model(model, encoder),
                    "calibration": {
                        "scope": "global",
                        "case_level_units": len(calibration_units),
                        "threshold": round(threshold, 12),
                        "threshold_order_statistic_rank": _conformal_rank(
                            len(calibration_units), alpha
                        ),
                        "finite_sample_nominal_case_error_upper_bound": (
                            _nominal_bound(len(calibration_units), alpha)
                        ),
                        "threshold_uses_calibration_only": True,
                    },
                }
            )
        if generated_repeats != expected_report["repeats"]:
            raise ValueError(
                f"generated repeats do not match {dataset} robustness report"
            )
        training_repeats[dataset] = dataset_training
        provenance.append(
            {
                "dataset": dataset,
                "source_path_label": source_path.name,
                "source_sha256": prepared["source_sha256"],
                "eligible_cases": len(
                    {record["case_id"] for record in prepared["variants"]}
                ),
                "robustness_report_path_label": expected_path.name,
                "robustness_report_sha256": sha256(expected_path),
                "baseline_robustness_repeats_exactly_reproduced": True,
            }
        )

    datasets = sorted(dataset_sources)
    analyses = _analyze_case_records(
        case_records,
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
        min_incomplete_case_families=min_incomplete_case_families,
        min_eligible_repeats=min_eligible_repeats,
    )
    outcome = _build_outcome(
        analyses,
        case_records,
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
            "base_feature_count": len(FEATURE_NAMES),
            "context_dimensions": list(CONTEXT_DIMENSIONS),
            "min_train_category_case_families": min_train_category_cases,
            "subgroup_dimensions": list(REPORTING_DIMENSIONS),
            "min_incomplete_case_families_per_repeat": (
                min_incomplete_case_families
            ),
            "min_eligible_repeats": min_eligible_repeats,
            "case_artifact": {},
        },
        "provenance": provenance,
        "development_protocol": {
            "post_hoc": True,
            "prompted_by_subgroup_and_mondrian_results": True,
            "independent_confirmation": False,
            "context_vocabulary_source": "train_case_families_only",
            "context_feature_or_model_selection_on_evaluation": False,
            "global_threshold_uses_calibration_only": True,
            "group_specific_thresholds": False,
            "inference_context_uses_gold_fields": False,
            "l2": l2,
            "iterations": iterations,
            "adoption_targets": adoption_targets,
        },
        "training_repeats": training_repeats,
        "datasets": analyses,
        "outcome": outcome,
        "theoretical_scope": {
            "global_marginal_statement": (
                "With the score function fixed from train only, the global "
                "calibration order statistic retains the same marginal "
                "finite-sample interpretation under case exchangeability."
            ),
            "conditional_subgroup_guarantee_claimed": False,
        },
        "decision": {
            "finding": (
                "This is post-hoc contextual-head development and requires a "
                "new untouched confirmation even if all adoption checks pass."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "KEEP_FROZEN_GLOBAL_METHOD_IN_SHADOW",
            "limitations": [
                "The method was designed after prior subgroup results were visible.",
                "Public QA context labels are not flood-domain operational strata.",
                "Global calibration does not provide conditional subgroup guarantees.",
                "No result from reused data authorizes CANARY or DEFAULT.",
            ],
        },
    }
    return report, case_records


def render_contextual_conformal_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG train-only 上下文充分性头事后方法开发",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 数据集：{', '.join(metadata['datasets'])}",
        f"- alpha：{metadata['primary_alpha']}",
        (
            "- 上下文："
            + ", ".join(metadata["context_dimensions"])
            + f"；每类最低训练 case：{metadata['min_train_category_case_families']}"
        ),
        "- 阈值：仅 calibration 的全局阈值",
        "- 独立确认：`False`",
        f"- Gate 2：`{outcome['gate_2']}`",
        "",
        "## 与冻结全局充分性头的比较",
        "",
        (
            "| 数据集 | 基线 case 风险 | 上下文 case 风险 | "
            "基线完整召回 | 上下文完整召回 | 最坏子群风险改善 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        aggregate = item["aggregate"]
        reduction = aggregate["worst_eligible_subgroup_risk_reduction"]
        reduction_text = "-" if reduction is None else f"{reduction:.6f}"
        lines.append(
            f"| {item['dataset']} | "
            f"{aggregate['baseline_global']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['contextual_global']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['baseline_global']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{aggregate['contextual_global']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{reduction_text} |"
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
                "该实验只检验 train-only 可观察上下文是否能改善充分性打分。"
                "它是在既有结果可见后开展的方法开发，不是独立确认；"
                "当前 Gate 2 继续保持 `NO-GO/SHADOW`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_contextual_conformal(
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
        render_contextual_conformal_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path, cases_path


def load_contextual_conformal(
    json_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported contextual conformal schema")
    artifact = metadata["case_artifact"]
    if artifact["path_label"] != cases_path.name:
        raise ValueError("contextual case artifact path label does not match")
    if artifact["sha256"] != sha256(cases_path):
        raise ValueError("contextual case artifact hash does not match")
    records = []
    with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != artifact["record_count"]:
        raise ValueError("contextual case artifact record count does not match")
    analyses = _analyze_case_records(
        records,
        datasets=list(metadata["datasets"]),
        split_versions=tuple(metadata["split_versions"]),
        alpha=float(metadata["primary_alpha"]),
        min_incomplete_case_families=int(
            metadata["min_incomplete_case_families_per_repeat"]
        ),
        min_eligible_repeats=int(metadata["min_eligible_repeats"]),
    )
    outcome = _build_outcome(
        analyses,
        records,
        alpha=float(metadata["primary_alpha"]),
        adoption_targets={
            key: float(value)
            for key, value in report["development_protocol"][
                "adoption_targets"
            ].items()
        },
    )
    if analyses != report.get("datasets") or outcome != report.get("outcome"):
        raise ValueError("contextual aggregates do not match case artifact")
    return report
