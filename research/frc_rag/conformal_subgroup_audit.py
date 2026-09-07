"""Frozen observable-subgroup stress audit for the conformal sufficiency head."""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_sufficiency import (
    PRIMARY_ALPHA,
    evaluate_conformal_sufficiency,
    prepare_conformal_source,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-observable-subgroup-audit-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_OBSERVABLE_SUBGROUP_STRESS_AUDIT"
DIMENSIONS = (
    "question_type",
    "required_role_count",
    "candidate_count_bucket",
)
MIN_INCOMPLETE_CASE_FAMILIES = 30
MIN_ELIGIBLE_REPEATS = 7


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    numeric = sorted(float(value) for value in values)
    if not numeric:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": len(numeric),
        "mean": round(statistics.fmean(numeric), 6),
        "median": round(statistics.median(numeric), 6),
        "min": round(numeric[0], 6),
        "max": round(numeric[-1], 6),
        "sample_std": (
            round(statistics.stdev(numeric), 6)
            if len(numeric) > 1
            else 0.0
        ),
    }


def _wilson_interval(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [
        round(max(0.0, centre - radius), 6),
        round(min(1.0, centre + radius), 6),
    ]


def _nominal_bound(calibration_units: int, alpha: float) -> tuple[int, float]:
    rank = min(
        calibration_units,
        math.ceil((calibration_units + 1) * (1.0 - alpha)),
    )
    bound = (calibration_units + 1 - rank) / (calibration_units + 1)
    return rank, round(bound, 12)


def _repeat_core(
    report: dict[str, Any],
    *,
    split_version: str,
) -> dict[str, Any]:
    alphas = tuple(
        float(value) for value in report["calibration"]["alphas"]
    )
    return {
        "split_version": split_version,
        "split_summary": report["split_summary"],
        "evaluation_auc": report["evaluation"]["auc"],
        "calibration_case_level_units": report["calibration"][
            "case_level_calibration_units"
        ],
        "finite_sample_nominal_bounds": {
            str(alpha): report["calibration"]["alphas"][str(alpha)][
                "finite_sample_nominal_case_error_upper_bound"
            ]
            for alpha in alphas
        },
        "baseline": report["evaluation"]["baseline_role_coverage_heuristic"],
        "alphas": {
            str(alpha): report["calibration"]["alphas"][str(alpha)][
                "evaluation"
            ]
            for alpha in alphas
        },
    }


def _group_metrics(
    records: list[dict[str, Any]],
    *,
    min_incomplete_case_families: int,
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case[str(record["case_id"])].append(record)
    incomplete_case_ids = {
        case_id
        for case_id, case_records in by_case.items()
        if any(not bool(item["complete"]) for item in case_records)
    }
    false_complete_case_ids = {
        case_id
        for case_id, case_records in by_case.items()
        if any(
            not bool(item["complete"])
            and bool(item["conformal_declared_complete"])
            for item in case_records
        )
    }
    complete_records = [item for item in records if bool(item["complete"])]
    declarations = sum(
        bool(item["conformal_declared_complete"]) for item in records
    )
    complete_declarations = sum(
        bool(item["conformal_declared_complete"])
        for item in complete_records
    )
    incomplete_count = len(incomplete_case_ids)
    false_count = len(false_complete_case_ids)
    return {
        "case_families": len(by_case),
        "variants": len(records),
        "incomplete_case_families": incomplete_count,
        "false_complete_case_families": false_count,
        "false_complete_case_family_rate": round(
            false_count / max(1, incomplete_count),
            6,
        ),
        "false_complete_case_family_rate_wilson_95": _wilson_interval(
            false_count,
            incomplete_count,
        ),
        "abstention_rate": round(
            1.0 - declarations / max(1, len(records)),
            6,
        ),
        "complete_variants": len(complete_records),
        "true_complete_declaration_rate": round(
            complete_declarations / max(1, len(complete_records)),
            6,
        ),
        "eligible_for_subgroup_conclusion": (
            incomplete_count >= min_incomplete_case_families
        ),
    }


def _repeat_subgroups(
    records: list[dict[str, Any]],
    *,
    min_incomplete_case_families: int,
) -> list[dict[str, Any]]:
    rows = []
    for dimension in DIMENSIONS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record[dimension])].append(record)
        for value, group_records in sorted(grouped.items()):
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    **_group_metrics(
                        group_records,
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                }
            )
    return rows


def _aggregate_subgroups(
    per_repeat: list[dict[str, Any]],
    *,
    alpha: float,
    min_eligible_repeats: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for repeat in per_repeat:
        for subgroup in repeat["subgroups"]:
            grouped[(subgroup["dimension"], subgroup["value"])].append(
                subgroup
            )
    results = []
    for (dimension, value), rows in sorted(grouped.items()):
        eligible = [
            row for row in rows if row["eligible_for_subgroup_conclusion"]
        ]
        aggregate_eligible = len(eligible) >= min_eligible_repeats
        result: dict[str, Any] = {
            "dimension": dimension,
            "value": value,
            "observed_repeats": len(rows),
            "eligible_repeats": len(eligible),
            "eligible_for_cross_repeat_conclusion": aggregate_eligible,
        }
        if aggregate_eligible:
            risk = _distribution(
                row["false_complete_case_family_rate"] for row in eligible
            )
            repeats_at_or_below = sum(
                row["false_complete_case_family_rate"] <= alpha
                for row in eligible
            )
            result.update(
                {
                    "false_complete_case_family_rate": risk,
                    "wilson_upper_95": _distribution(
                        row[
                            "false_complete_case_family_rate_wilson_95"
                        ][1]
                        for row in eligible
                    ),
                    "abstention_rate": _distribution(
                        row["abstention_rate"] for row in eligible
                    ),
                    "true_complete_declaration_rate": _distribution(
                        row["true_complete_declaration_rate"]
                        for row in eligible
                    ),
                    "risk_at_or_below_alpha_repeats": repeats_at_or_below,
                    "mean_risk_at_or_below_alpha": risk["mean"] <= alpha,
                    "repeat_consistency_target_met": (
                        repeats_at_or_below >= min_eligible_repeats
                    ),
                }
            )
        results.append(result)
    eligible_results = [
        item
        for item in results
        if item["eligible_for_cross_repeat_conclusion"]
    ]
    worst = (
        max(
            eligible_results,
            key=lambda item: (
                item["false_complete_case_family_rate"]["mean"],
                item["dimension"],
                item["value"],
            ),
        )
        if eligible_results
        else None
    )
    return {
        "subgroups": results,
        "eligible_subgroup_count": len(eligible_results),
        "insufficient_support_subgroup_count": (
            len(results) - len(eligible_results)
        ),
        "all_eligible_subgroup_mean_risks_at_or_below_alpha": (
            bool(eligible_results)
            and all(
                item["mean_risk_at_or_below_alpha"]
                for item in eligible_results
            )
        ),
        "all_eligible_subgroup_repeat_consistency_targets_met": (
            bool(eligible_results)
            and all(
                item["repeat_consistency_target_met"]
                for item in eligible_results
            )
        ),
        "worst_eligible_subgroup": (
            {
                "dimension": worst["dimension"],
                "value": worst["value"],
                "mean_false_complete_case_family_rate": worst[
                    "false_complete_case_family_rate"
                ]["mean"],
                "max_false_complete_case_family_rate": worst[
                    "false_complete_case_family_rate"
                ]["max"],
                "risk_at_or_below_alpha_repeats": worst[
                    "risk_at_or_below_alpha_repeats"
                ],
            }
            if worst is not None
            else None
        ),
    }


def _analyze_case_records(
    case_records: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    theoretical_bounds: dict[str, list[dict[str, Any]]],
    alpha: float,
    min_incomplete_case_families: int,
    min_eligible_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analyses = []
    for dataset in datasets:
        dataset_records = [
            item for item in case_records if item["dataset"] == dataset
        ]
        per_repeat = []
        bounds_by_split = {
            item["split_version"]: item
            for item in theoretical_bounds[dataset]
        }
        for split_version in split_versions:
            repeat_records = [
                item
                for item in dataset_records
                if item["split_version"] == split_version
            ]
            if not repeat_records:
                raise ValueError(
                    f"missing case records for {dataset}/{split_version}"
                )
            bound = bounds_by_split[split_version]
            per_repeat.append(
                {
                    "split_version": split_version,
                    "evaluation_case_families": len(
                        {item["case_id"] for item in repeat_records}
                    ),
                    "calibration_case_level_units": bound[
                        "calibration_case_level_units"
                    ],
                    "threshold_order_statistic_rank": bound[
                        "threshold_order_statistic_rank"
                    ],
                    "finite_sample_nominal_case_error_upper_bound": bound[
                        "finite_sample_nominal_case_error_upper_bound"
                    ],
                    "subgroups": _repeat_subgroups(
                        repeat_records,
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                }
            )
        aggregate = _aggregate_subgroups(
            per_repeat,
            alpha=alpha,
            min_eligible_repeats=min_eligible_repeats,
        )
        analyses.append(
            {
                "dataset": dataset,
                "per_repeat": per_repeat,
                "aggregate": aggregate,
            }
        )
    eligible_count = sum(
        item["aggregate"]["eligible_subgroup_count"] for item in analyses
    )
    all_nominal = all(
        repeat["finite_sample_nominal_case_error_upper_bound"] <= alpha
        for item in analyses
        for repeat in item["per_repeat"]
    )
    all_mean = bool(eligible_count) and all(
        item["aggregate"][
            "all_eligible_subgroup_mean_risks_at_or_below_alpha"
        ]
        for item in analyses
    )
    all_repeat_consistency = bool(eligible_count) and all(
        item["aggregate"][
            "all_eligible_subgroup_repeat_consistency_targets_met"
        ]
        for item in analyses
    )
    if not eligible_count:
        status = "INSUFFICIENT_SUBGROUP_SUPPORT"
    elif all_nominal and all_mean and all_repeat_consistency:
        status = "SUBGROUP_STABLE"
    else:
        status = "SUBGROUP_INSTABILITY_DETECTED"
    outcome = {
        "status": status,
        "dataset_count": len(analyses),
        "eligible_subgroup_count": eligible_count,
        "all_global_finite_sample_nominal_bounds_at_or_below_alpha": (
            all_nominal
        ),
        "all_eligible_subgroup_mean_risks_at_or_below_alpha": all_mean,
        "all_eligible_subgroup_repeat_consistency_targets_met": (
            all_repeat_consistency
        ),
        "conditional_subgroup_guarantee_claimed": False,
        "gate_2": "NO-GO/SHADOW",
    }
    return analyses, outcome


def evaluate_conformal_subgroup_audit(
    dataset_sources: dict[str, Path],
    expected_robustness: dict[str, tuple[Path, dict[str, Any]]],
    *,
    reference_run_report: Path | None = None,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    alpha: float = PRIMARY_ALPHA,
    min_incomplete_case_families: int = MIN_INCOMPLETE_CASE_FAMILIES,
    min_eligible_repeats: int = MIN_ELIGIBLE_REPEATS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(dataset_sources) != set(expected_robustness):
        raise ValueError("dataset sources and robustness reports must match")
    if len(dataset_sources) < 2:
        raise ValueError("subgroup audit requires at least two datasets")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if min_incomplete_case_families < 1:
        raise ValueError("minimum incomplete case families must be positive")
    if not 1 <= min_eligible_repeats <= len(split_versions):
        raise ValueError("minimum eligible repeats is outside split count")

    case_records: list[dict[str, Any]] = []
    theoretical_bounds: dict[str, list[dict[str, Any]]] = {}
    provenance = []
    for dataset in sorted(dataset_sources):
        source_path = dataset_sources[dataset]
        expected_path, expected_report = expected_robustness[dataset]
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
        dataset_bounds = []
        for split_version in split_versions:
            report, records = evaluate_conformal_sufficiency(
                source_path,
                reference_run_report=reference_run_report,
                split_version=split_version,
                k=top_k,
                token_budget=token_budget,
                role_threshold=role_threshold,
                prepared_source=prepared,
            )
            generated_repeats.append(
                _repeat_core(report, split_version=split_version)
            )
            alpha_result = report["calibration"]["alphas"][str(alpha)]
            units = report["calibration"]["case_level_calibration_units"]
            expected_rank, expected_bound = _nominal_bound(units, alpha)
            if (
                alpha_result["threshold_order_statistic_rank"]
                != expected_rank
                or alpha_result[
                    "finite_sample_nominal_case_error_upper_bound"
                ]
                != expected_bound
            ):
                raise AssertionError("finite-sample nominal bound is inconsistent")
            dataset_bounds.append(
                {
                    "split_version": split_version,
                    "calibration_case_level_units": units,
                    "threshold_order_statistic_rank": expected_rank,
                    "finite_sample_nominal_case_error_upper_bound": (
                        expected_bound
                    ),
                }
            )
            for record in records:
                if record["split"] != "evaluation":
                    continue
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
                        "complete": record["complete"],
                        "conformal_declared_complete": record[
                            "conformal_declared_complete"
                        ],
                    }
                )
        if generated_repeats != expected_report["repeats"]:
            raise ValueError(
                f"generated repeats do not match {dataset} robustness report"
            )
        theoretical_bounds[dataset] = dataset_bounds
        provenance.append(
            {
                "dataset": dataset,
                "source_path_label": source_path.name,
                "source_sha256": prepared["source_sha256"],
                "eligible_cases": len(
                    {
                        item["case_id"]
                        for item in prepared["variants"]
                    }
                ),
                "robustness_report_path_label": expected_path.name,
                "robustness_report_sha256": sha256(expected_path),
                "robustness_repeats_exactly_reproduced": True,
            }
        )

    datasets = sorted(dataset_sources)
    analyses, outcome = _analyze_case_records(
        case_records,
        datasets=datasets,
        split_versions=split_versions,
        theoretical_bounds=theoretical_bounds,
        alpha=alpha,
        min_incomplete_case_families=min_incomplete_case_families,
        min_eligible_repeats=min_eligible_repeats,
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
            "subgroup_dimensions": list(DIMENSIONS),
            "candidate_count_buckets": [
                "lt_20",
                "20_to_39",
                "40_to_59",
                "ge_60",
            ],
            "min_incomplete_case_families_per_repeat": (
                min_incomplete_case_families
            ),
            "min_eligible_repeats": min_eligible_repeats,
            "no_subgroup_threshold_or_model_selection": True,
            "case_artifact": {},
        },
        "provenance": provenance,
        "theoretical_scope": {
            "global_finite_sample_claim": (
                "Under case exchangeability, strict declaration above the "
                "split-conformal order statistic has the recorded marginal "
                "case-error upper bound."
            ),
            "conditional_subgroup_claim": (
                "No distribution-free conditional guarantee is claimed for "
                "observable subgroups because thresholds are calibrated globally."
            ),
            "gold_fields_used_for_subgroup_assignment": False,
        },
        "theoretical_global_bounds": theoretical_bounds,
        "datasets": analyses,
        "outcome": outcome,
        "decision": {
            "finding": (
                "Observable subgroup stress tests diagnose heterogeneity; they do "
                "not select a new threshold and do not authorize deployment."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
            "limitations": [
                "Subgroup repeats share cases and are descriptive correlated checks.",
                "Only subgroups with pre-registered minimum support enter conclusions.",
                "Public QA strata are not substitutes for flood-domain expert strata.",
                "Wilson intervals are descriptive and are not conformal guarantees.",
            ],
        },
    }
    return report, case_records


def render_conformal_subgroup_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG 可观察子群充分性压力审计",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 数据集：{', '.join(metadata['datasets'])}",
        f"- 主 alpha：{metadata['primary_alpha']}",
        f"- 每数据集重复分组：{metadata['repeat_count_per_dataset']}",
        (
            "- 子群最低支持：每次 "
            f"{metadata['min_incomplete_case_families_per_repeat']} 个不完整 "
            f"case，至少 {metadata['min_eligible_repeats']} 次可评"
        ),
        f"- 总体结论：`{outcome['status']}`",
        f"- Gate 2：`{outcome['gate_2']}`",
        "",
        "## 数据集最坏可评子群",
        "",
        "| 数据集 | 可评子群数 | 最坏维度 | 最坏取值 | 平均 case 风险 | 最大 case 风险 | alpha 内重复 |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        aggregate = item["aggregate"]
        worst = aggregate["worst_eligible_subgroup"]
        if worst is None:
            lines.append(
                f"| {item['dataset']} | 0 | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {item['dataset']} | {aggregate['eligible_subgroup_count']} | "
            f"{worst['dimension']} | {worst['value']} | "
            f"{worst['mean_false_complete_case_family_rate']:.6f} | "
            f"{worst['max_false_complete_case_family_rate']:.6f} | "
            f"{worst['risk_at_or_below_alpha_repeats']}/"
            f"{metadata['repeat_count_per_dataset']} |"
        )
    lines.extend(
        [
            "",
            "## 理论边界",
            "",
            (
                "- 全局有限样本名义上界全部不高于 alpha："
                f"`{outcome['all_global_finite_sample_nominal_bounds_at_or_below_alpha']}`"
            ),
            (
                "- 所有可评子群平均风险不高于 alpha："
                f"`{outcome['all_eligible_subgroup_mean_risks_at_or_below_alpha']}`"
            ),
            (
                "- 所有可评子群达到重复一致性目标："
                f"`{outcome['all_eligible_subgroup_repeat_consistency_targets_met']}`"
            ),
            "- 不声明分布无关的条件子群保证。",
            "",
            (
                "该审计只暴露全局校准在可观察子群上的异质性，不使用 gold 字段"
                "分组、不改变阈值，也不改变 `NO-GO/SHADOW`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_conformal_subgroup_audit(
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
        render_conformal_subgroup_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path, cases_path


def load_conformal_subgroup_audit(
    json_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported conformal subgroup audit schema")
    artifact = metadata["case_artifact"]
    if artifact["path_label"] != cases_path.name:
        raise ValueError("subgroup case artifact path label does not match")
    if artifact["sha256"] != sha256(cases_path):
        raise ValueError("subgroup case artifact hash does not match")
    records = []
    with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != artifact["record_count"]:
        raise ValueError("subgroup case artifact record count does not match")
    analyses, outcome = _analyze_case_records(
        records,
        datasets=list(metadata["datasets"]),
        split_versions=tuple(metadata["split_versions"]),
        theoretical_bounds=report["theoretical_global_bounds"],
        alpha=float(metadata["primary_alpha"]),
        min_incomplete_case_families=int(
            metadata["min_incomplete_case_families_per_repeat"]
        ),
        min_eligible_repeats=int(metadata["min_eligible_repeats"]),
    )
    if analyses != report.get("datasets") or outcome != report.get("outcome"):
        raise ValueError("subgroup audit aggregates do not match case artifact")
    return report
