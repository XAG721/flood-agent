"""Review-only ranking audit over frozen 37- and 64-feature sufficiency scores."""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from research.frc_rag.conformal_contextual import _distribution
from research.frc_rag.conformal_score_stability import (
    SCHEMA_VERSION as SCORE_STABILITY_SCHEMA_VERSION,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-review-ranking-v1"
STATUS = "RUN_PUBLIC_POST_HOC_REVIEW_ONLY_SCORE_STABILITY_RANKING"
REVIEW_BUDGET_FRACTIONS = (0.05, 0.1, 0.2)
SUBGROUP_DIMENSIONS = (
    "question_type",
    "required_role_count",
    "candidate_count_bucket",
)
SUBGROUP_BUDGET = 0.1
MIN_SUBGROUP_POOL_VARIANTS = 30
MIN_SUBGROUP_COMPLETE_VARIANTS = 5
MIN_SUBGROUP_SUPPORTED_REPEATS = 7
ADOPTION_TARGETS: dict[str, float | int | bool] = {
    "automatic_decisions_exactly_unchanged": True,
    "identical_reviewed_count_at_every_budget": True,
    "minimum_cross_dataset_mean_average_precision_gain": 0.01,
    "minimum_datasets_with_positive_average_precision_gain": 3,
    "minimum_cross_dataset_mean_complete_capture_gain_at_0_05": 0.02,
    "minimum_cross_dataset_mean_complete_capture_gain_at_0_10": 0.02,
    "minimum_cross_dataset_mean_complete_capture_gain_at_0_20": 0.01,
    "minimum_datasets_with_positive_complete_capture_gain_each_budget": 3,
    "maximum_dataset_complete_capture_decrease_each_budget": 0.01,
    "maximum_dataset_review_precision_decrease_at_0_10": 0.01,
    "maximum_dataset_worst_eligible_subgroup_capture_decrease_at_0_10": 0.05,
}


def _budget_key(value: float) -> str:
    return str(float(value))


def _target_budget_suffix(value: float) -> str:
    return f"0_{round(float(value) * 100):02d}"


def _record_key(record: dict[str, Any]) -> tuple[str, float]:
    return str(record["case_id"]), float(record["target_missing_ratio"])


def _rank_records(
    records: list[dict[str, Any]], score_field: str
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            -float(record[score_field]),
            str(record["case_id"]),
            float(record["target_missing_ratio"]),
        ),
    )


def _average_precision(ordered: list[dict[str, Any]]) -> float:
    positives = sum(bool(record["complete"]) for record in ordered)
    if not positives:
        return 0.0
    found = 0
    precision_sum = 0.0
    for rank, record in enumerate(ordered, start=1):
        if bool(record["complete"]):
            found += 1
            precision_sum += found / rank
    return precision_sum / positives


def _review_metrics(
    records: list[dict[str, Any]],
    reviewed_field: tuple[str, str],
) -> dict[str, float | int]:
    method, budget = reviewed_field
    reviewed = [
        record for record in records if bool(record["reviewed"][budget][method])
    ]
    complete = sum(bool(record["complete"]) for record in records)
    reviewed_complete = sum(bool(record["complete"]) for record in reviewed)
    return {
        "review_pool_variants": len(records),
        "review_pool_complete_variants": complete,
        "reviewed_count": len(reviewed),
        "reviewed_complete_count": reviewed_complete,
        "review_precision": round(reviewed_complete / max(1, len(reviewed)), 6),
        "complete_capture_rate": round(reviewed_complete / max(1, complete), 6),
    }


def _build_evidence(
    source_records: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    budgets: tuple[float, ...],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for dataset in datasets:
        for split_version in split_versions:
            records = [
                record
                for record in source_records
                if record["dataset"] == dataset
                and record["split_version"] == split_version
            ]
            if not records:
                raise ValueError(
                    f"missing source records for {dataset}/{split_version}"
                )
            keys = [_record_key(record) for record in records]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate review-ranking record key")
            pool = [
                record
                for record in records
                if not bool(record["baseline_declared_complete"])
            ]
            if not pool:
                raise ValueError("review pool is empty")
            baseline_order = _rank_records(pool, "baseline_score")
            stability_order = _rank_records(pool, "score_stability_score")
            baseline_rank = {
                _record_key(record): rank
                for rank, record in enumerate(baseline_order, start=1)
            }
            stability_rank = {
                _record_key(record): rank
                for rank, record in enumerate(stability_order, start=1)
            }
            selected: dict[str, dict[str, set[tuple[str, float]]]] = {}
            for budget in budgets:
                key = _budget_key(budget)
                reviewed_count = math.ceil(len(pool) * budget)
                selected[key] = {
                    "baseline": {
                        _record_key(record)
                        for record in baseline_order[:reviewed_count]
                    },
                    "score_stability": {
                        _record_key(record)
                        for record in stability_order[:reviewed_count]
                    },
                }
            for record in records:
                key = _record_key(record)
                automatic = bool(record["baseline_declared_complete"])
                in_pool = not automatic
                evidence.append(
                    {
                        "dataset": dataset,
                        "split_version": split_version,
                        "case_id": record["case_id"],
                        "question_type": record["question_type"],
                        "required_role_count": str(record["required_role_count"]),
                        "candidate_count_bucket": record["candidate_count_bucket"],
                        "target_missing_ratio": record["target_missing_ratio"],
                        "complete": bool(record["complete"]),
                        "source_missing": bool(record["source_missing"]),
                        "baseline_score": float(record["baseline_score"]),
                        "score_stability_score": float(record["score_stability_score"]),
                        "automatic_decision_before": automatic,
                        "automatic_decision_after": automatic,
                        "review_pool": in_pool,
                        "baseline_review_rank": (
                            baseline_rank[key] if in_pool else None
                        ),
                        "score_stability_review_rank": (
                            stability_rank[key] if in_pool else None
                        ),
                        "reviewed": {
                            budget_key: {
                                method: in_pool and key in method_selected
                                for method, method_selected in methods.items()
                            }
                            for budget_key, methods in selected.items()
                        },
                        "automatic_threshold_or_score_changed": False,
                        "ranking_or_budget_selected_on_evaluation": False,
                    }
                )
    return evidence


def _subgroup_rows(
    records: list[dict[str, Any]],
    *,
    budget: str,
    min_pool_variants: int,
    min_complete_variants: int,
) -> list[dict[str, Any]]:
    rows = []
    for dimension in SUBGROUP_DIMENSIONS:
        values = sorted({str(record[dimension]) for record in records})
        for value in values:
            group = [record for record in records if str(record[dimension]) == value]
            complete = sum(bool(record["complete"]) for record in group)
            supported = (
                len(group) >= min_pool_variants and complete >= min_complete_variants
            )
            baseline_complete = sum(
                bool(record["complete"])
                and bool(record["reviewed"][budget]["baseline"])
                for record in group
            )
            stability_complete = sum(
                bool(record["complete"])
                and bool(record["reviewed"][budget]["score_stability"])
                for record in group
            )
            baseline_capture = baseline_complete / max(1, complete)
            stability_capture = stability_complete / max(1, complete)
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "review_pool_variants": len(group),
                    "complete_variants": complete,
                    "supported": supported,
                    "baseline_complete_capture_rate": round(baseline_capture, 6),
                    "score_stability_complete_capture_rate": round(
                        stability_capture, 6
                    ),
                    "delta_score_stability_minus_baseline": round(
                        stability_capture - baseline_capture, 6
                    ),
                }
            )
    return rows


def _aggregate_subgroups(
    repeats: list[dict[str, Any]],
    *,
    min_supported_repeats: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for repeat in repeats:
        for row in repeat["subgroups"]:
            grouped[(row["dimension"], row["value"])].append(row)
    aggregates = []
    for (dimension, value), rows in sorted(grouped.items()):
        supported = [row for row in rows if row["supported"]]
        eligible = len(supported) >= min_supported_repeats
        aggregates.append(
            {
                "dimension": dimension,
                "value": value,
                "supported_repeats": len(supported),
                "eligible": eligible,
                "mean_complete_capture_delta": (
                    round(
                        statistics.fmean(
                            float(row["delta_score_stability_minus_baseline"])
                            for row in supported
                        ),
                        6,
                    )
                    if supported
                    else None
                ),
            }
        )
    eligible = [row for row in aggregates if row["eligible"]]
    worst = (
        min(
            eligible,
            key=lambda row: (
                float(row["mean_complete_capture_delta"]),
                row["dimension"],
                row["value"],
            ),
        )
        if eligible
        else None
    )
    return {
        "minimum_supported_repeats": min_supported_repeats,
        "eligible_subgroup_count": len(eligible),
        "subgroups": aggregates,
        "worst_eligible_subgroup": worst,
    }


def _validate_evidence(
    evidence: list[dict[str, Any]], budgets: tuple[float, ...]
) -> bool:
    budget_keys = {_budget_key(value) for value in budgets}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in evidence:
        before = bool(record["automatic_decision_before"])
        if bool(record["automatic_decision_after"]) != before:
            return False
        if bool(record["automatic_threshold_or_score_changed"]):
            return False
        if bool(record["ranking_or_budget_selected_on_evaluation"]):
            return False
        if bool(record["review_pool"]) == before:
            return False
        if set(record["reviewed"]) != budget_keys:
            return False
        if before:
            if record["baseline_review_rank"] is not None:
                return False
            if record["score_stability_review_rank"] is not None:
                return False
            if any(any(methods.values()) for methods in record["reviewed"].values()):
                return False
        groups[(record["dataset"], record["split_version"])].append(record)
    for records in groups.values():
        keys = [_record_key(record) for record in records]
        if len(keys) != len(set(keys)):
            return False
        pool = [record for record in records if record["review_pool"]]
        if not pool:
            return False
        for method, score_field, rank_field in (
            ("baseline", "baseline_score", "baseline_review_rank"),
            (
                "score_stability",
                "score_stability_score",
                "score_stability_review_rank",
            ),
        ):
            expected_order = _rank_records(pool, score_field)
            expected_ranks = {
                _record_key(record): rank
                for rank, record in enumerate(expected_order, start=1)
            }
            if any(
                record[rank_field] != expected_ranks[_record_key(record)]
                for record in pool
            ):
                return False
            for budget in budgets:
                key = _budget_key(budget)
                reviewed_count = math.ceil(len(pool) * budget)
                if any(
                    bool(record["reviewed"][key][method])
                    != (int(record[rank_field]) <= reviewed_count)
                    for record in pool
                ):
                    return False
    return bool(evidence)


def _analyze_evidence(
    evidence: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    budgets: tuple[float, ...],
    subgroup_budget: float,
    min_subgroup_pool_variants: int,
    min_subgroup_complete_variants: int,
    min_subgroup_supported_repeats: int,
) -> list[dict[str, Any]]:
    analyses = []
    subgroup_budget_key = _budget_key(subgroup_budget)
    for dataset in datasets:
        repeats = []
        for split_version in split_versions:
            records = [
                record
                for record in evidence
                if record["dataset"] == dataset
                and record["split_version"] == split_version
            ]
            pool = [record for record in records if record["review_pool"]]
            if not pool:
                raise ValueError(f"missing review pool for {dataset}/{split_version}")
            baseline_order = sorted(
                pool, key=lambda record: int(record["baseline_review_rank"])
            )
            stability_order = sorted(
                pool,
                key=lambda record: int(record["score_stability_review_rank"]),
            )
            budget_metrics = {}
            for budget in budgets:
                key = _budget_key(budget)
                baseline = _review_metrics(pool, ("baseline", key))
                stability = _review_metrics(pool, ("score_stability", key))
                if baseline["reviewed_count"] != stability["reviewed_count"]:
                    raise ValueError("review workloads differ")
                budget_metrics[key] = {
                    "budget_fraction": budget,
                    "baseline": baseline,
                    "score_stability": stability,
                    "delta_score_stability_minus_baseline": {
                        "review_precision": round(
                            float(stability["review_precision"])
                            - float(baseline["review_precision"]),
                            6,
                        ),
                        "complete_capture_rate": round(
                            float(stability["complete_capture_rate"])
                            - float(baseline["complete_capture_rate"]),
                            6,
                        ),
                    },
                }
            repeats.append(
                {
                    "split_version": split_version,
                    "evaluation_variants": len(records),
                    "automatic_candidate_count": sum(
                        bool(record["automatic_decision_before"]) for record in records
                    ),
                    "review_pool_variants": len(pool),
                    "review_pool_complete_variants": sum(
                        bool(record["complete"]) for record in pool
                    ),
                    "average_precision": {
                        "baseline": round(_average_precision(baseline_order), 6),
                        "score_stability": round(
                            _average_precision(stability_order), 6
                        ),
                        "delta": round(
                            _average_precision(stability_order)
                            - _average_precision(baseline_order),
                            6,
                        ),
                    },
                    "budgets": budget_metrics,
                    "subgroups": _subgroup_rows(
                        pool,
                        budget=subgroup_budget_key,
                        min_pool_variants=min_subgroup_pool_variants,
                        min_complete_variants=min_subgroup_complete_variants,
                    ),
                }
            )
        subgroup_aggregate = _aggregate_subgroups(
            repeats,
            min_supported_repeats=min_subgroup_supported_repeats,
        )
        budget_aggregates = {}
        for budget in budgets:
            key = _budget_key(budget)
            budget_aggregates[key] = {
                "budget_fraction": budget,
            }
            for method in ("baseline", "score_stability"):
                budget_aggregates[key][method] = {
                    metric: _distribution(
                        repeat["budgets"][key][method][metric] for repeat in repeats
                    )
                    for metric in (
                        "reviewed_count",
                        "review_precision",
                        "complete_capture_rate",
                    )
                }
            budget_aggregates[key]["delta_score_stability_minus_baseline"] = {
                metric: _distribution(
                    repeat["budgets"][key]["delta_score_stability_minus_baseline"][
                        metric
                    ]
                    for repeat in repeats
                )
                for metric in ("review_precision", "complete_capture_rate")
            }
        analyses.append(
            {
                "dataset": dataset,
                "per_repeat": repeats,
                "aggregate": {
                    "average_precision": {
                        method: _distribution(
                            repeat["average_precision"][method] for repeat in repeats
                        )
                        for method in (
                            "baseline",
                            "score_stability",
                            "delta",
                        )
                    },
                    "budgets": budget_aggregates,
                    "subgroups_at_0_10": subgroup_aggregate,
                },
            }
        )
    if not _validate_evidence(evidence, budgets):
        raise ValueError("review-ranking evidence violates frozen boundaries")
    return analyses


def _check(measured: Any, target: Any, passed: bool) -> dict[str, Any]:
    return {"measured": measured, "target": target, "passed": bool(passed)}


def _build_outcome(
    analyses: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    budgets: tuple[float, ...],
    adoption_targets: dict[str, float | int | bool],
) -> dict[str, Any]:
    scope_valid = _validate_evidence(evidence, budgets)
    ap_gains = {
        item["dataset"]: float(item["aggregate"]["average_precision"]["delta"]["mean"])
        for item in analyses
    }
    mean_ap_gain = round(statistics.fmean(ap_gains.values()), 6)
    capture_gains: dict[str, dict[str, float]] = {}
    precision_gains: dict[str, dict[str, float]] = {}
    workload_equal = True
    for budget in budgets:
        key = _budget_key(budget)
        capture_gains[key] = {
            item["dataset"]: float(
                item["aggregate"]["budgets"][key][
                    "delta_score_stability_minus_baseline"
                ]["complete_capture_rate"]["mean"]
            )
            for item in analyses
        }
        precision_gains[key] = {
            item["dataset"]: float(
                item["aggregate"]["budgets"][key][
                    "delta_score_stability_minus_baseline"
                ]["review_precision"]["mean"]
            )
            for item in analyses
        }
        workload_equal = workload_equal and all(
            all(
                repeat["budgets"][key]["baseline"]["reviewed_count"]
                == repeat["budgets"][key]["score_stability"]["reviewed_count"]
                for repeat in item["per_repeat"]
            )
            for item in analyses
        )
    mean_capture_gains = {
        key: round(statistics.fmean(values.values()), 6)
        for key, values in capture_gains.items()
    }
    positive_capture_counts = {
        key: sum(value > 0.0 for value in values.values())
        for key, values in capture_gains.items()
    }
    capture_decreases = {
        key: {dataset: round(max(0.0, -value), 6) for dataset, value in values.items()}
        for key, values in capture_gains.items()
    }
    subgroup_worst = {
        item["dataset"]: item["aggregate"]["subgroups_at_0_10"][
            "worst_eligible_subgroup"
        ]
        for item in analyses
    }
    subgroup_decreases = {
        dataset: (
            None
            if row is None
            else round(max(0.0, -float(row["mean_complete_capture_delta"])), 6)
        )
        for dataset, row in subgroup_worst.items()
    }
    checks = {
        "automatic_decisions_exactly_unchanged": _check(scope_valid, True, scope_valid),
        "automatic_thresholds_and_scores_unchanged": _check(
            scope_valid, True, scope_valid
        ),
        "ranking_and_budgets_not_selected_on_evaluation": _check(
            scope_valid, True, scope_valid
        ),
        "identical_reviewed_count_at_every_budget": _check(
            workload_equal, True, workload_equal
        ),
        "minimum_cross_dataset_mean_average_precision_gain": _check(
            mean_ap_gain,
            f">={adoption_targets['minimum_cross_dataset_mean_average_precision_gain']}",
            mean_ap_gain
            >= float(
                adoption_targets["minimum_cross_dataset_mean_average_precision_gain"]
            ),
        ),
        "minimum_datasets_with_positive_average_precision_gain": _check(
            {
                "count": sum(value > 0.0 for value in ap_gains.values()),
                "per_dataset": ap_gains,
            },
            f">={adoption_targets['minimum_datasets_with_positive_average_precision_gain']}",
            sum(value > 0.0 for value in ap_gains.values())
            >= int(
                adoption_targets[
                    "minimum_datasets_with_positive_average_precision_gain"
                ]
            ),
        ),
    }
    for budget in budgets:
        key = _budget_key(budget)
        suffix = _target_budget_suffix(budget)
        target_key = "minimum_cross_dataset_mean_complete_capture_gain_at_" + suffix
        checks[target_key] = _check(
            mean_capture_gains[key],
            f">={adoption_targets[target_key]}",
            mean_capture_gains[key] >= float(adoption_targets[target_key]),
        )
    checks["minimum_datasets_with_positive_complete_capture_gain_each_budget"] = _check(
        positive_capture_counts,
        f">={adoption_targets['minimum_datasets_with_positive_complete_capture_gain_each_budget']}",
        all(
            count
            >= int(
                adoption_targets[
                    "minimum_datasets_with_positive_complete_capture_gain_each_budget"
                ]
            )
            for count in positive_capture_counts.values()
        ),
    )
    checks["maximum_dataset_complete_capture_decrease_each_budget"] = _check(
        capture_decreases,
        f"<={adoption_targets['maximum_dataset_complete_capture_decrease_each_budget']}",
        all(
            decrease
            <= float(
                adoption_targets[
                    "maximum_dataset_complete_capture_decrease_each_budget"
                ]
            )
            for values in capture_decreases.values()
            for decrease in values.values()
        ),
    )
    precision_decreases_010 = {
        dataset: round(max(0.0, -value), 6)
        for dataset, value in precision_gains["0.1"].items()
    }
    checks["maximum_dataset_review_precision_decrease_at_0_10"] = _check(
        precision_decreases_010,
        f"<={adoption_targets['maximum_dataset_review_precision_decrease_at_0_10']}",
        all(
            value
            <= float(
                adoption_targets["maximum_dataset_review_precision_decrease_at_0_10"]
            )
            for value in precision_decreases_010.values()
        ),
    )
    checks["maximum_dataset_worst_eligible_subgroup_capture_decrease_at_0_10"] = _check(
        subgroup_decreases,
        (
            "<="
            + str(
                adoption_targets[
                    "maximum_dataset_worst_eligible_subgroup_capture_decrease_at_0_10"
                ]
            )
        ),
        all(
            value is not None
            and value
            <= float(
                adoption_targets[
                    "maximum_dataset_worst_eligible_subgroup_capture_decrease_at_0_10"
                ]
            )
            for value in subgroup_decreases.values()
        ),
    )
    all_passed = all(check["passed"] for check in checks.values())
    return {
        "status": (
            "POST_HOC_REVIEW_RANKER_CANDIDATE"
            if all_passed
            else "KEEP_BASE_REVIEW_ORDER"
        ),
        "post_hoc_method_development": True,
        "review_only": True,
        "automatic_decisions_changed": False,
        "gate_2": "NO-GO/SHADOW",
        "checks": checks,
        "all_pre_registered_adoption_checks_passed": all_passed,
    }


def evaluate_review_ranking(
    source_report: dict[str, Any],
    source_records: list[dict[str, Any]],
    *,
    source_artifact: dict[str, Any],
    budgets: tuple[float, ...] = REVIEW_BUDGET_FRACTIONS,
    subgroup_budget: float = SUBGROUP_BUDGET,
    min_subgroup_pool_variants: int = MIN_SUBGROUP_POOL_VARIANTS,
    min_subgroup_complete_variants: int = MIN_SUBGROUP_COMPLETE_VARIANTS,
    min_subgroup_supported_repeats: int = MIN_SUBGROUP_SUPPORTED_REPEATS,
    adoption_targets: dict[str, float | int | bool] = ADOPTION_TARGETS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = source_report.get("metadata", {})
    if metadata.get("schema_version") != SCORE_STABILITY_SCHEMA_VERSION:
        raise ValueError("review ranking requires score-stability source report")
    datasets = list(metadata["datasets"])
    split_versions = tuple(metadata["split_versions"])
    if sorted(budgets) != list(budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("review budgets must be unique and sorted")
    if subgroup_budget not in budgets:
        raise ValueError("subgroup budget must be one of the review budgets")
    evidence = _build_evidence(
        source_records,
        datasets=datasets,
        split_versions=split_versions,
        budgets=budgets,
    )
    analyses = _analyze_evidence(
        evidence,
        datasets=datasets,
        split_versions=split_versions,
        budgets=budgets,
        subgroup_budget=subgroup_budget,
        min_subgroup_pool_variants=min_subgroup_pool_variants,
        min_subgroup_complete_variants=min_subgroup_complete_variants,
        min_subgroup_supported_repeats=min_subgroup_supported_repeats,
    )
    outcome = _build_outcome(
        analyses,
        evidence,
        budgets=budgets,
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
            "review_budget_fractions": list(budgets),
            "review_pool_definition": "baseline_declared_complete is false",
            "subgroup_dimensions": list(SUBGROUP_DIMENSIONS),
            "subgroup_budget": subgroup_budget,
            "minimum_subgroup_pool_variants_per_repeat": (min_subgroup_pool_variants),
            "minimum_subgroup_complete_variants_per_repeat": (
                min_subgroup_complete_variants
            ),
            "minimum_subgroup_supported_repeats": (min_subgroup_supported_repeats),
            "source_artifact": source_artifact,
            "evidence_artifact": {},
        },
        "development_protocol": {
            "post_hoc": True,
            "independent_confirmation": False,
            "review_only": True,
            "automatic_decision_field": "baseline_declared_complete",
            "automatic_decisions_or_thresholds_modified": False,
            "baseline_ranking": "baseline_score descending",
            "candidate_ranking": "score_stability_score descending",
            "tie_break": "case_id ascending, target_missing_ratio ascending",
            "review_count_rule": "ceil(review_pool_size * budget_fraction)",
            "ranking_or_budget_selection_on_evaluation": False,
            "adoption_targets": adoption_targets,
        },
        "datasets": analyses,
        "outcome": outcome,
        "decision": {
            "next_step": (
                "FREEZE_REVIEW_ONLY_RANKER_FOR_UNTOUCHED_VALIDATION"
                if outcome["status"] == "POST_HOC_REVIEW_RANKER_CANDIDATE"
                else "STOP_REVIEW_RANKER_DEVELOPMENT_ON_REVEALED_EVALUATIONS"
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "HUMAN_REVIEW_ONLY_NO_AUTOMATIC_BOUNDARY_CHANGE",
            "limitations": [
                "Benchmark completeness labels simulate reviewer utility.",
                "No real reviewer time, behavior or flood-domain judgment is measured.",
                "The ranking was evaluated on already revealed public QA records.",
                "No result authorizes CANARY, DEFAULT or automatic completion.",
            ],
        },
    }
    return report, evidence


def render_review_ranking_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG review-only ranking audit",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Datasets: {', '.join(metadata['datasets'])}",
        "- Automatic decision or threshold changed: `False`",
        f"- Review budgets: {metadata['review_budget_fractions']}",
        f"- Gate 2: `{outcome['gate_2']}`",
        "",
        "## Ranking comparison at fixed review workload",
        "",
        (
            "| Dataset | Baseline AP | Stability AP | AP gain | "
            "5% capture gain | 10% capture gain | 20% capture gain |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        aggregate = item["aggregate"]
        lines.append(
            f"| {item['dataset']} | "
            f"{aggregate['average_precision']['baseline']['mean']:.6f} | "
            f"{aggregate['average_precision']['score_stability']['mean']:.6f} | "
            f"{aggregate['average_precision']['delta']['mean']:.6f} | "
            f"{aggregate['budgets']['0.05']['delta_score_stability_minus_baseline']['complete_capture_rate']['mean']:.6f} | "
            f"{aggregate['budgets']['0.1']['delta_score_stability_minus_baseline']['complete_capture_rate']['mean']:.6f} | "
            f"{aggregate['budgets']['0.2']['delta_score_stability_minus_baseline']['complete_capture_rate']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered adoption checks",
            "",
            "| Check | Measured | Target | Passed |",
            "|---|---|---|---|",
        ]
    )
    for name, check in outcome["checks"].items():
        measured = json.dumps(check["measured"], ensure_ascii=False)
        lines.append(
            f"| {name} | `{measured}` | `{check['target']}` | `{check['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This audit only compares review order at fixed human-review "
                "workload. The automatic candidate set, frozen 37-feature "
                "score and conformal threshold remain unchanged. Benchmark "
                "labels simulate review utility and do not represent real "
                "flood-response reviewer behavior or production authorization."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_review_ranking(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> tuple[Path, Path, Path]:
    for path in (json_path, markdown_path, evidence_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in evidence
    ).encode("utf-8")
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            archive.write(payload)
    report["metadata"]["evidence_artifact"] = {
        "path_label": evidence_path.name,
        "sha256": sha256(evidence_path),
        "record_count": len(evidence),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_review_ranking_markdown(report), encoding="utf-8")
    return json_path, markdown_path, evidence_path


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_review_ranking(
    json_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported review-ranking schema")
    artifact = metadata["evidence_artifact"]
    if artifact["path_label"] != evidence_path.name:
        raise ValueError("review-ranking evidence path label does not match")
    if artifact["sha256"] != sha256(evidence_path):
        raise ValueError("review-ranking evidence hash does not match")
    evidence = read_gzip_jsonl(evidence_path)
    if len(evidence) != artifact["record_count"]:
        raise ValueError("review-ranking evidence count does not match")
    budgets = tuple(float(value) for value in metadata["review_budget_fractions"])
    analyses = _analyze_evidence(
        evidence,
        datasets=list(metadata["datasets"]),
        split_versions=tuple(metadata["split_versions"]),
        budgets=budgets,
        subgroup_budget=float(metadata["subgroup_budget"]),
        min_subgroup_pool_variants=int(
            metadata["minimum_subgroup_pool_variants_per_repeat"]
        ),
        min_subgroup_complete_variants=int(
            metadata["minimum_subgroup_complete_variants_per_repeat"]
        ),
        min_subgroup_supported_repeats=int(
            metadata["minimum_subgroup_supported_repeats"]
        ),
    )
    outcome = _build_outcome(
        analyses,
        evidence,
        budgets=budgets,
        adoption_targets=report["development_protocol"]["adoption_targets"],
    )
    if analyses != report.get("datasets") or outcome != report.get("outcome"):
        raise ValueError("review-ranking aggregates do not match evidence")
    return report
