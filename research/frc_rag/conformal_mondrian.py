"""Post-hoc hierarchical Mondrian development for the FRC sufficiency head."""

from __future__ import annotations

import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_subgroup_audit import (
    DIMENSIONS,
    _aggregate_subgroups,
    _repeat_core,
    _repeat_subgroups,
)
from research.frc_rag.conformal_sufficiency import (
    PRIMARY_ALPHA,
    _conformal_rank,
    _conformal_threshold,
    _decision_metrics,
    evaluate_conformal_sufficiency,
    prepare_conformal_source,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-hierarchical-mondrian-development-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_POST_HOC_MONDRIAN_DEVELOPMENT"
MIN_GROUP_CALIBRATION_UNITS = 30
MIN_INCOMPLETE_CASE_FAMILIES = 30
MIN_ELIGIBLE_REPEATS = 7
FALLBACK_ORDER = (
    "joint",
    "question_type",
    "required_role_count",
    "global",
)
GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "joint": ("question_type", "required_role_count"),
    "question_type": ("question_type",),
    "required_role_count": ("required_role_count",),
    "global": (),
}
ADOPTION_TARGETS = {
    "minimum_mean_worst_eligible_subgroup_risk_reduction": 0.02,
    "maximum_dataset_overall_case_risk_increase": 0.01,
    "maximum_dataset_complete_recall_decrease": 0.05,
    "minimum_group_specific_threshold_case_fraction": 0.5,
}
SUMMARY_METRICS = (
    "false_complete_case_family_rate",
    "abstention_rate",
    "true_complete_declaration_rate",
)


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


def _nominal_bound(calibration_units: int, alpha: float) -> float:
    rank = _conformal_rank(calibration_units, alpha)
    return round(
        (calibration_units + 1 - rank) / (calibration_units + 1),
        12,
    )


def _attributes(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question_type": str(record["question_type"]),
        "required_role_count": str(record["required_role_count"]),
        "candidate_count_bucket": str(record["candidate_count_bucket"]),
    }


def _validate_group_configuration(
    fallback_order: tuple[str, ...],
    group_fields: dict[str, tuple[str, ...]],
) -> None:
    if not fallback_order or fallback_order[-1] != "global":
        raise ValueError("Mondrian fallback order must end with global")
    if len(set(fallback_order)) != len(fallback_order):
        raise ValueError("Mondrian fallback sources must be unique")
    if set(fallback_order) != set(group_fields):
        raise ValueError("Mondrian fallback sources and group fields must match")
    if group_fields["global"]:
        raise ValueError("global Mondrian fallback cannot use group fields")
    allowed_fields = set(_attributes({
        "question_type": "probe",
        "required_role_count": "1",
        "candidate_count_bucket": "lt_20",
    }))
    configured_fields = {
        field for fields in group_fields.values() for field in fields
    }
    if not configured_fields <= allowed_fields:
        raise ValueError("unsupported observable Mondrian group field")


def _group_key(
    source: str,
    record: dict[str, Any],
    *,
    group_fields: dict[str, tuple[str, ...]] = GROUP_FIELDS,
) -> str:
    attributes = _attributes(record)
    fields = group_fields.get(source)
    if fields is None:
        raise ValueError(f"unsupported Mondrian threshold source: {source}")
    if not fields:
        return "__all__"
    return "|".join(f"{field}={attributes[field]}" for field in fields)


def _calibration_case_units(
    records: list[dict[str, Any]],
    *,
    group_fields: dict[str, tuple[str, ...]] = GROUP_FIELDS,
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["split"] == "calibration":
            by_case[str(record["case_id"])].append(record)
    units = []
    for case_id, case_records in sorted(by_case.items()):
        observable_fields = tuple(
            dict.fromkeys(
                field for fields in group_fields.values() for field in fields
            )
        )
        attribute_rows = {
            tuple(_attributes(item)[field] for field in observable_fields)
            for item in case_records
        }
        if len(attribute_rows) != 1:
            raise ValueError("observable Mondrian attributes vary within a case")
        incomplete = [
            item for item in case_records if not bool(item["complete"])
        ]
        if not incomplete:
            continue
        units.append(
            {
                "case_id": case_id,
                **_attributes(case_records[0]),
                "maximum_incomplete_score": max(
                    float(item["sufficiency_score"]) for item in incomplete
                ),
            }
        )
    if not units:
        raise ValueError("calibration split has no incomplete case families")
    return units


def _threshold_tables(
    calibration_units: list[dict[str, Any]],
    *,
    alpha: float,
    min_group_calibration_units: int,
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
    group_fields: dict[str, tuple[str, ...]] = GROUP_FIELDS,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    tables = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for source in fallback_order:
        grouped: dict[str, list[float]] = defaultdict(list)
        for unit in calibration_units:
            grouped[_group_key(source, unit, group_fields=group_fields)].append(
                float(unit["maximum_incomplete_score"])
            )
        entries = []
        for key, scores in sorted(grouped.items()):
            count = len(scores)
            eligible = source == "global" or count >= min_group_calibration_units
            entry: dict[str, Any] = {
                "source": source,
                "key": key,
                "calibration_case_level_units": count,
                "eligible": eligible,
            }
            if eligible:
                score_array = np.asarray(scores, dtype=np.float64)
                entry.update(
                    {
                        "threshold": round(
                            _conformal_threshold(score_array, alpha),
                            12,
                        ),
                        "threshold_order_statistic_rank": _conformal_rank(
                            count,
                            alpha,
                        ),
                        "finite_sample_nominal_case_error_upper_bound": (
                            _nominal_bound(count, alpha)
                        ),
                    }
                )
                lookup[(source, key)] = entry
            else:
                entry.update(
                    {
                        "threshold": None,
                        "threshold_order_statistic_rank": None,
                        "finite_sample_nominal_case_error_upper_bound": None,
                    }
                )
            entries.append(entry)
        tables.append({"source": source, "groups": entries})
    if ("global", "__all__") not in lookup:
        raise AssertionError("global Mondrian fallback threshold is missing")
    return tables, lookup


def _select_threshold(
    record: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
    group_fields: dict[str, tuple[str, ...]] = GROUP_FIELDS,
) -> dict[str, Any]:
    for source in fallback_order:
        entry = lookup.get(
            (source, _group_key(source, record, group_fields=group_fields))
        )
        if entry is not None:
            return entry
    raise AssertionError("global Mondrian fallback threshold was not selected")


def _source_usage(
    records: list[dict[str, Any]],
    *,
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
) -> dict[str, Any]:
    by_case: dict[str, str] = {}
    for record in records:
        case_id = str(record["case_id"])
        source = str(record["mondrian_threshold_source"])
        previous = by_case.setdefault(case_id, source)
        if previous != source:
            raise ValueError("Mondrian threshold source varies within a case")
    counts = {
        source: sum(value == source for value in by_case.values())
        for source in fallback_order
    }
    total = len(by_case)
    group_specific = total - counts["global"]
    return {
        "evaluation_case_families": total,
        "case_families_by_source": counts,
        "case_fraction_by_source": {
            source: round(count / max(1, total), 6)
            for source, count in counts.items()
        },
        "group_specific_case_families": group_specific,
        "group_specific_threshold_case_fraction": round(
            group_specific / max(1, total),
            6,
        ),
    }


def _method_metrics(
    records: list[dict[str, Any]],
    decision_field: str,
) -> dict[str, Any]:
    decisions = np.asarray(
        [bool(item[decision_field]) for item in records],
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


def _combined_source_usage(
    per_repeat: list[dict[str, Any]],
    *,
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
) -> dict[str, Any]:
    counts = {
        source: sum(
            int(repeat["threshold_source_usage"]["case_families_by_source"][source])
            for repeat in per_repeat
        )
        for source in fallback_order
    }
    total = sum(counts.values())
    group_specific = total - counts["global"]
    return {
        "evaluation_case_family_repeat_units": total,
        "case_family_repeat_units_by_source": counts,
        "case_fraction_by_source": {
            source: round(count / max(1, total), 6)
            for source, count in counts.items()
        },
        "group_specific_case_family_repeat_units": group_specific,
        "group_specific_threshold_case_fraction": round(
            group_specific / max(1, total),
            6,
        ),
    }


def _validate_record_thresholds(
    records: list[dict[str, Any]],
    *,
    min_group_calibration_units: int,
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
) -> bool:
    allowed = set(fallback_order)
    for record in records:
        source = str(record["mondrian_threshold_source"])
        units = int(record["mondrian_calibration_case_level_units"])
        if source not in allowed or units < 1:
            return False
        if source != "global" and units < min_group_calibration_units:
            return False
        if not bool(record["threshold_source_calibration_only"]):
            return False
    return True


def _analyze_case_records(
    case_records: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    alpha: float,
    min_group_calibration_units: int,
    min_incomplete_case_families: int,
    min_eligible_repeats: int,
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
) -> list[dict[str, Any]]:
    analyses = []
    for dataset in datasets:
        per_repeat = []
        for split_version in split_versions:
            records = [
                item
                for item in case_records
                if item["dataset"] == dataset
                and item["split_version"] == split_version
            ]
            if not records:
                raise ValueError(
                    f"missing Mondrian case records for {dataset}/{split_version}"
                )
            global_metrics = _method_metrics(
                records,
                "global_declared_complete",
            )
            mondrian_metrics = _method_metrics(
                records,
                "mondrian_declared_complete",
            )
            per_repeat.append(
                {
                    "split_version": split_version,
                    "global": global_metrics,
                    "mondrian": mondrian_metrics,
                    "delta_mondrian_minus_global": {
                        metric: round(
                            float(mondrian_metrics[metric])
                            - float(global_metrics[metric]),
                            6,
                        )
                        for metric in SUMMARY_METRICS
                    },
                    "threshold_source_usage": _source_usage(
                        records,
                        fallback_order=fallback_order,
                    ),
                    "global_subgroups": _method_subgroups(
                        records,
                        "global_declared_complete",
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                    "mondrian_subgroups": _method_subgroups(
                        records,
                        "mondrian_declared_complete",
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                }
            )
        global_subgroups = _aggregate_subgroups(
            [
                {"subgroups": repeat["global_subgroups"]}
                for repeat in per_repeat
            ],
            alpha=alpha,
            min_eligible_repeats=min_eligible_repeats,
        )
        mondrian_subgroups = _aggregate_subgroups(
            [
                {"subgroups": repeat["mondrian_subgroups"]}
                for repeat in per_repeat
            ],
            alpha=alpha,
            min_eligible_repeats=min_eligible_repeats,
        )
        global_worst = global_subgroups["worst_eligible_subgroup"]
        mondrian_worst = mondrian_subgroups["worst_eligible_subgroup"]
        worst_risk_reduction = (
            round(
                float(global_worst["mean_false_complete_case_family_rate"])
                - float(
                    mondrian_worst["mean_false_complete_case_family_rate"]
                ),
                6,
            )
            if global_worst is not None and mondrian_worst is not None
            else None
        )
        analyses.append(
            {
                "dataset": dataset,
                "per_repeat": per_repeat,
                "aggregate": {
                    "global": _aggregate_metric_rows(per_repeat, "global"),
                    "mondrian": _aggregate_metric_rows(
                        per_repeat,
                        "mondrian",
                    ),
                    "delta_mondrian_minus_global": {
                        metric: _distribution(
                            repeat["delta_mondrian_minus_global"][metric]
                            for repeat in per_repeat
                        )
                        for metric in SUMMARY_METRICS
                    },
                    "threshold_source_usage": _combined_source_usage(
                        per_repeat,
                        fallback_order=fallback_order,
                    ),
                    "global_subgroups": global_subgroups,
                    "mondrian_subgroups": mondrian_subgroups,
                    "worst_eligible_subgroup_risk_reduction": (
                        worst_risk_reduction
                    ),
                },
            }
        )
    if not _validate_record_thresholds(
        case_records,
        min_group_calibration_units=min_group_calibration_units,
        fallback_order=fallback_order,
    ):
        raise ValueError("Mondrian case records contain an invalid threshold source")
    return analyses


def _check(
    measured: Any,
    target: Any,
    passed: bool,
) -> dict[str, Any]:
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
    min_group_calibration_units: int,
    adoption_targets: dict[str, float],
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
) -> dict[str, Any]:
    risk_reductions = [
        float(item["aggregate"]["worst_eligible_subgroup_risk_reduction"])
        for item in analyses
        if item["aggregate"]["worst_eligible_subgroup_risk_reduction"]
        is not None
    ]
    mean_worst_reduction = (
        round(statistics.fmean(risk_reductions), 6)
        if len(risk_reductions) == len(analyses) and risk_reductions
        else None
    )
    overall_risk_increases = {
        item["dataset"]: round(
            float(
                item["aggregate"]["mondrian"][
                    "false_complete_case_family_rate"
                ]["mean"]
            )
            - float(
                item["aggregate"]["global"][
                    "false_complete_case_family_rate"
                ]["mean"]
            ),
            6,
        )
        for item in analyses
    }
    recall_decreases = {
        item["dataset"]: round(
            float(
                item["aggregate"]["global"][
                    "true_complete_declaration_rate"
                ]["mean"]
            )
            - float(
                item["aggregate"]["mondrian"][
                    "true_complete_declaration_rate"
                ]["mean"]
            ),
            6,
        )
        for item in analyses
    }
    total_usage = _source_usage(
        [
            {
                **record,
                "case_id": (
                    f"{record['dataset']}::{record['split_version']}::"
                    f"{record['case_id']}"
                ),
            }
            for record in case_records
        ],
        fallback_order=fallback_order,
    )
    mondrian_eligible_count = sum(
        item["aggregate"]["mondrian_subgroups"]["eligible_subgroup_count"]
        for item in analyses
    )
    all_subgroup_means = bool(mondrian_eligible_count) and all(
        item["aggregate"]["mondrian_subgroups"][
            "all_eligible_subgroup_mean_risks_at_or_below_alpha"
        ]
        for item in analyses
    )
    all_subgroup_repeats = bool(mondrian_eligible_count) and all(
        item["aggregate"]["mondrian_subgroups"][
            "all_eligible_subgroup_repeat_consistency_targets_met"
        ]
        for item in analyses
    )
    threshold_sources_valid = _validate_record_thresholds(
        case_records,
        min_group_calibration_units=min_group_calibration_units,
        fallback_order=fallback_order,
    )
    checks = {
        "no_evaluation_selection": _check(True, True, True),
        "all_threshold_sources_calibration_only": _check(
            threshold_sources_valid,
            True,
            threshold_sources_valid,
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
        "maximum_dataset_overall_case_risk_increase": _check(
            overall_risk_increases,
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
                for value in overall_risk_increases.values()
            ),
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
        "group_specific_threshold_case_fraction": _check(
            total_usage["group_specific_threshold_case_fraction"],
            (
                ">="
                + str(
                    adoption_targets[
                        "minimum_group_specific_threshold_case_fraction"
                    ]
                )
            ),
            total_usage["group_specific_threshold_case_fraction"]
            >= adoption_targets[
                "minimum_group_specific_threshold_case_fraction"
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
    all_passed = all(item["passed"] for item in checks.values())
    return {
        "status": "DEVELOPMENT_CANDIDATE" if all_passed else "DO_NOT_ADOPT",
        "post_hoc_method_development": True,
        "independent_confirmation": False,
        "eligible_subgroup_count": mondrian_eligible_count,
        "alpha": alpha,
        "checks": checks,
        "all_pre_registered_adoption_checks_passed": all_passed,
        "threshold_source_usage": total_usage,
        "conditional_subgroup_guarantee_claimed": False,
        "gate_2": "NO-GO/SHADOW",
    }


def evaluate_hierarchical_mondrian(
    dataset_sources: dict[str, Path],
    expected_robustness: dict[str, tuple[Path, dict[str, Any]]],
    *,
    reference_run_report: Path | None = None,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    alpha: float = PRIMARY_ALPHA,
    min_group_calibration_units: int = MIN_GROUP_CALIBRATION_UNITS,
    min_incomplete_case_families: int = MIN_INCOMPLETE_CASE_FAMILIES,
    min_eligible_repeats: int = MIN_ELIGIBLE_REPEATS,
    adoption_targets: dict[str, float] = ADOPTION_TARGETS,
    fallback_order: tuple[str, ...] = FALLBACK_ORDER,
    group_fields: dict[str, tuple[str, ...]] = GROUP_FIELDS,
    schema_version: str = SCHEMA_VERSION,
    status: str = STATUS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_group_configuration(fallback_order, group_fields)
    if set(dataset_sources) != set(expected_robustness):
        raise ValueError("dataset sources and robustness reports must match")
    if len(dataset_sources) < 2:
        raise ValueError("Mondrian development requires at least two datasets")
    if len(set(split_versions)) != len(split_versions):
        raise ValueError("split versions must be unique")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if min_group_calibration_units < 1:
        raise ValueError("minimum group calibration units must be positive")
    if min_incomplete_case_families < 1:
        raise ValueError("minimum incomplete case families must be positive")
    if not 1 <= min_eligible_repeats <= len(split_versions):
        raise ValueError("minimum eligible repeats is outside split count")

    case_records: list[dict[str, Any]] = []
    threshold_calibration: dict[str, list[dict[str, Any]]] = {}
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
        calibration_repeats = []
        for split_version in split_versions:
            sufficiency, records = evaluate_conformal_sufficiency(
                source_path,
                reference_run_report=reference_run_report,
                split_version=split_version,
                k=top_k,
                token_budget=token_budget,
                role_threshold=role_threshold,
                prepared_source=prepared,
            )
            generated_repeats.append(
                _repeat_core(sufficiency, split_version=split_version)
            )
            units = _calibration_case_units(
                records,
                group_fields=group_fields,
            )
            tables, lookup = _threshold_tables(
                units,
                alpha=alpha,
                min_group_calibration_units=min_group_calibration_units,
                fallback_order=fallback_order,
                group_fields=group_fields,
            )
            global_entry = lookup[("global", "__all__")]
            frozen_global = sufficiency["calibration"]["alphas"][str(alpha)]
            if (
                global_entry["threshold"] != frozen_global["threshold"]
                or global_entry["threshold_order_statistic_rank"]
                != frozen_global["threshold_order_statistic_rank"]
                or global_entry[
                    "finite_sample_nominal_case_error_upper_bound"
                ]
                != frozen_global[
                    "finite_sample_nominal_case_error_upper_bound"
                ]
            ):
                raise AssertionError(
                    "recomputed global threshold does not match frozen report"
                )
            calibration_repeats.append(
                {
                    "split_version": split_version,
                    "threshold_sources": tables,
                    "global_threshold_exactly_reproduced": True,
                }
            )
            for record in records:
                if record["split"] != "evaluation":
                    continue
                threshold = _select_threshold(
                    record,
                    lookup,
                    fallback_order=fallback_order,
                    group_fields=group_fields,
                )
                source = str(threshold["source"])
                mondrian_declared = (
                    bool(record["conformal_declared_complete"])
                    if source == "global"
                    else float(record["sufficiency_score"])
                    > float(threshold["threshold"])
                )
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
                        "sufficiency_score": record["sufficiency_score"],
                        "global_declared_complete": record[
                            "conformal_declared_complete"
                        ],
                        "mondrian_declared_complete": mondrian_declared,
                        "mondrian_threshold_source": source,
                        "mondrian_threshold_key": threshold["key"],
                        "mondrian_threshold": threshold["threshold"],
                        "mondrian_calibration_case_level_units": threshold[
                            "calibration_case_level_units"
                        ],
                        "mondrian_threshold_order_statistic_rank": threshold[
                            "threshold_order_statistic_rank"
                        ],
                        "mondrian_finite_sample_nominal_case_error_upper_bound": (
                            threshold[
                                "finite_sample_nominal_case_error_upper_bound"
                            ]
                        ),
                        "threshold_source_calibration_only": True,
                    }
                )
        if generated_repeats != expected_report["repeats"]:
            raise ValueError(
                f"generated repeats do not match {dataset} robustness report"
            )
        threshold_calibration[dataset] = calibration_repeats
        provenance.append(
            {
                "dataset": dataset,
                "source_path_label": source_path.name,
                "source_sha256": prepared["source_sha256"],
                "eligible_cases": len(
                    {item["case_id"] for item in prepared["variants"]}
                ),
                "robustness_report_path_label": expected_path.name,
                "robustness_report_sha256": sha256(expected_path),
                "robustness_repeats_exactly_reproduced": True,
            }
        )

    datasets = sorted(dataset_sources)
    analyses = _analyze_case_records(
        case_records,
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
        min_group_calibration_units=min_group_calibration_units,
        min_incomplete_case_families=min_incomplete_case_families,
        min_eligible_repeats=min_eligible_repeats,
        fallback_order=fallback_order,
    )
    outcome = _build_outcome(
        analyses,
        case_records,
        alpha=alpha,
        min_group_calibration_units=min_group_calibration_units,
        adoption_targets=adoption_targets,
        fallback_order=fallback_order,
    )
    report = {
        "metadata": {
            "schema_version": schema_version,
            "status": status,
            "datasets": datasets,
            "dataset_count": len(datasets),
            "split_versions": list(split_versions),
            "repeat_count_per_dataset": len(split_versions),
            "primary_alpha": alpha,
            "min_group_calibration_case_families": (
                min_group_calibration_units
            ),
            "fallback_order": list(fallback_order),
            "subgroup_dimensions": list(DIMENSIONS),
            "min_incomplete_case_families_per_repeat": (
                min_incomplete_case_families
            ),
            "min_eligible_repeats": min_eligible_repeats,
            "score_decision_precision": (
                "Stored 12-decimal sufficiency scores are used for non-global "
                "Mondrian decisions; the exact frozen decision is retained on "
                "global fallback."
            ),
            "case_artifact": {},
        },
        "provenance": provenance,
        "development_protocol": {
            "post_hoc": True,
            "prompted_by_observed_subgroup_instability": True,
            "independent_confirmation": False,
            "single_global_sufficiency_head_per_train_split": True,
            "thresholds_use_calibration_only": True,
            "evaluation_used_for_threshold_or_fallback_selection": False,
            "group_assignment_uses_gold_fields": False,
            "adoption_targets": adoption_targets,
        },
        "threshold_calibration": threshold_calibration,
        "datasets": analyses,
        "outcome": outcome,
        "theoretical_scope": {
            "groupwise_marginal_statement": (
                "A finite-sample groupwise marginal statement requires "
                "exchangeability within the exact calibration group whose "
                "threshold is used."
            ),
            "fallback_statement": (
                "Fallback does not create a conditional guarantee for a more "
                "specific subgroup that lacks calibration support."
            ),
            "conditional_subgroup_guarantee_claimed": False,
        },
        "decision": {
            "finding": (
                "This is post-hoc method development. Even a development "
                "candidate requires a new untouched confirmation set."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "KEEP_FROZEN_GLOBAL_METHOD_IN_SHADOW",
            "limitations": [
                "The method was designed after the subgroup audit was inspected.",
                "Public QA groups are not flood-domain operational strata.",
                "Fallback thresholds do not guarantee unsupported finer groups.",
                "No result from these reused data authorizes CANARY or DEFAULT.",
            ],
        },
    }
    if group_fields != GROUP_FIELDS:
        report["metadata"]["group_fields"] = {
            source: list(group_fields[source]) for source in fallback_order
        }
    return report, case_records


def render_hierarchical_mondrian_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG 分层 Mondrian conformal 事后方法开发",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 数据集：{', '.join(metadata['datasets'])}",
        f"- alpha：{metadata['primary_alpha']}",
        (
            "- 回退顺序："
            + " → ".join(metadata["fallback_order"])
            + f"；分组阈值最低校准 case 数："
            f"{metadata['min_group_calibration_case_families']}"
        ),
        "- 独立确认：`False`",
        f"- Gate 2：`{outcome['gate_2']}`",
        "",
        "## 与全局阈值的冻结比较",
        "",
        (
            "| 数据集 | 全局 case 风险 | Mondrian case 风险 | "
            "全局完整召回 | Mondrian 完整召回 | 最坏子群风险改善 | 分组阈值占比 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        aggregate = item["aggregate"]
        reduction = aggregate["worst_eligible_subgroup_risk_reduction"]
        reduction_text = "-" if reduction is None else f"{reduction:.6f}"
        lines.append(
            f"| {item['dataset']} | "
            f"{aggregate['global']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['mondrian']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['global']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{aggregate['mondrian']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{reduction_text} | "
            f"{aggregate['threshold_source_usage']['group_specific_threshold_case_fraction']:.6f} |"
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
                "该实验是在已查看子群不稳定结果后开展的方法开发，不是独立确认。"
                "只有冻结方法后在全新未触碰数据或独立防汛专家数据上复现，"
                "才可重新评审；当前继续保持 `NO-GO/SHADOW`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_hierarchical_mondrian(
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
        render_hierarchical_mondrian_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path, cases_path


def load_hierarchical_mondrian(
    json_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported hierarchical Mondrian schema")
    artifact = metadata["case_artifact"]
    if artifact["path_label"] != cases_path.name:
        raise ValueError("Mondrian case artifact path label does not match")
    if artifact["sha256"] != sha256(cases_path):
        raise ValueError("Mondrian case artifact hash does not match")
    records = []
    with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != artifact["record_count"]:
        raise ValueError("Mondrian case artifact record count does not match")
    analyses = _analyze_case_records(
        records,
        datasets=list(metadata["datasets"]),
        split_versions=tuple(metadata["split_versions"]),
        alpha=float(metadata["primary_alpha"]),
        min_group_calibration_units=int(
            metadata["min_group_calibration_case_families"]
        ),
        min_incomplete_case_families=int(
            metadata["min_incomplete_case_families_per_repeat"]
        ),
        min_eligible_repeats=int(metadata["min_eligible_repeats"]),
    )
    outcome = _build_outcome(
        analyses,
        records,
        alpha=float(metadata["primary_alpha"]),
        min_group_calibration_units=int(
            metadata["min_group_calibration_case_families"]
        ),
        adoption_targets={
            key: float(value)
            for key, value in report["development_protocol"][
                "adoption_targets"
            ].items()
        },
    )
    if analyses != report.get("datasets") or outcome != report.get("outcome"):
        raise ValueError("Mondrian aggregates do not match case artifact")
    return report
