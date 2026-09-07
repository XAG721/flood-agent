"""Repeated grouped-split robustness audit for the FRC sufficiency head."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from research.frc_rag.conformal_sufficiency import (
    DEFAULT_ALPHAS,
    PRIMARY_ALPHA,
    SCHEMA_VERSION as SUFFICIENCY_SCHEMA_VERSION,
    SPLIT_VERSION,
    evaluate_conformal_sufficiency,
    prepare_conformal_source,
)


SCHEMA_VERSION = "frc-conformal-robustness-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_REPEATED_GROUP_SPLIT_ROBUSTNESS"
DEFAULT_SPLIT_VERSIONS = (SPLIT_VERSION,) + tuple(
    f"{SPLIT_VERSION}-repeat-{index:02d}" for index in range(1, 10)
)
ROBUSTNESS_METRICS = (
    "false_complete_case_family_rate",
    "false_complete_rate_all_incomplete",
    "false_complete_rate_source_missing",
    "declaration_rate",
    "abstention_rate",
    "true_complete_declaration_rate",
    "complete_declaration_precision",
)


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot summarize an empty metric distribution")
    ordered = sorted(numeric)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "sample_std": round(statistics.stdev(ordered), 6)
        if len(ordered) > 1
        else 0.0,
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: _distribution(row[metric] for row in rows)
        for metric in ROBUSTNESS_METRICS
    }


def _triage_metrics(
    auto_complete: dict[str, Any], review_eligible: dict[str, Any]
) -> dict[str, float | int]:
    auto_true = (
        int(auto_complete["declarations"])
        - int(auto_complete["false_complete_count_all_incomplete"])
    )
    eligible_true = (
        int(review_eligible["declarations"])
        - int(review_eligible["false_complete_count_all_incomplete"])
    )
    review_band_count = int(review_eligible["declarations"]) - int(
        auto_complete["declarations"]
    )
    review_band_true = eligible_true - auto_true
    review_band_incomplete = (
        int(review_eligible["false_complete_count_all_incomplete"])
        - int(auto_complete["false_complete_count_all_incomplete"])
    )
    if min(review_band_count, review_band_true, review_band_incomplete) < 0:
        raise ValueError("review alpha must create a superset of auto declarations")
    variants = int(auto_complete["variants"])
    complete = int(auto_complete["complete_variants"])
    incomplete = int(auto_complete["incomplete_variants"])
    return {
        "variants": variants,
        "complete_variants": complete,
        "incomplete_variants": incomplete,
        "auto_complete_count": int(auto_complete["declarations"]),
        "auto_complete_rate": round(
            int(auto_complete["declarations"]) / max(1, variants), 6
        ),
        "auto_complete_precision": round(
            auto_true / max(1, int(auto_complete["declarations"])), 6
        ),
        "auto_complete_false_rate_on_incomplete": round(
            int(auto_complete["false_complete_count_all_incomplete"])
            / max(1, incomplete),
            6,
        ),
        "review_band_count": review_band_count,
        "review_band_rate": round(review_band_count / max(1, variants), 6),
        "review_band_precision": round(
            review_band_true / max(1, review_band_count), 6
        ),
        "review_band_complete_capture_rate": round(
            review_band_true / max(1, complete), 6
        ),
        "review_band_incomplete_routing_rate": round(
            review_band_incomplete / max(1, incomplete), 6
        ),
        "auto_or_review_complete_coverage": round(
            eligible_true / max(1, complete), 6
        ),
        "insufficient_count": variants - int(review_eligible["declarations"]),
        "insufficient_rate": round(
            1.0 - int(review_eligible["declarations"]) / max(1, variants), 6
        ),
    }


def _build_aggregate(
    repeats: list[dict[str, Any]],
    *,
    alphas: tuple[float, ...],
    primary_alpha: float,
    review_alpha: float,
) -> dict[str, Any]:
    baseline_rows = [item["baseline"] for item in repeats]
    alpha_results: dict[str, Any] = {}
    for alpha in alphas:
        key = str(alpha)
        rows = [item["alphas"][key] for item in repeats]
        alpha_results[key] = {
            "alpha": alpha,
            "metrics": _aggregate_metrics(rows),
            "finite_sample_nominal_case_error_upper_bound": _distribution(
                item["finite_sample_nominal_bounds"][key]
                for item in repeats
            ),
            "nominal_bound_at_or_below_alpha_repeats": sum(
                item["finite_sample_nominal_bounds"][key] <= alpha
                for item in repeats
            ),
            "case_family_rate_at_or_below_alpha_repeats": sum(
                row["false_complete_case_family_rate"] <= alpha for row in rows
            ),
            "case_family_rate_at_or_below_alpha_fraction": round(
                sum(
                    row["false_complete_case_family_rate"] <= alpha for row in rows
                )
                / len(rows),
                6,
            ),
        }
    primary_key = str(primary_alpha)
    primary_rows = [item["alphas"][primary_key] for item in repeats]
    false_complete_reductions = [
        baseline["false_complete_case_family_rate"]
        - conformal["false_complete_case_family_rate"]
        for baseline, conformal in zip(baseline_rows, primary_rows, strict=True)
    ]
    recall_changes = [
        conformal["true_complete_declaration_rate"]
        - baseline["true_complete_declaration_rate"]
        for baseline, conformal in zip(baseline_rows, primary_rows, strict=True)
    ]
    triage_rows = [
        _triage_metrics(
            item["alphas"][primary_key],
            item["alphas"][str(review_alpha)],
        )
        for item in repeats
    ]
    return {
        "baseline_role_coverage_heuristic": _aggregate_metrics(baseline_rows),
        "alphas": alpha_results,
        "primary_alpha_paired_differences": {
            "baseline_minus_conformal_false_complete_case_family_rate": (
                _distribution(false_complete_reductions)
            ),
            "conformal_minus_baseline_true_complete_declaration_rate": (
                _distribution(recall_changes)
            ),
            "repeats_with_lower_case_family_false_complete_than_baseline": sum(
                reduction > 0 for reduction in false_complete_reductions
            ),
            "repeat_count": len(repeats),
        },
        "three_state_triage": {
            "auto_complete_alpha": primary_alpha,
            "review_eligible_alpha": review_alpha,
            "policy": (
                "HIGH_CONFIDENCE_CANDIDATE uses the primary threshold; "
                "REVIEW_PRIORITY is the additional lower-score band admitted only by "
                "review alpha; all remaining variants are INSUFFICIENT. Gate 2 blocks "
                "automatic completion, and both candidate bands remain human-reviewed."
            ),
            "metrics": {
                metric: _distribution(row[metric] for row in triage_rows)
                for metric in (
                    "auto_complete_rate",
                    "auto_complete_precision",
                    "auto_complete_false_rate_on_incomplete",
                    "review_band_rate",
                    "review_band_precision",
                    "review_band_complete_capture_rate",
                    "review_band_incomplete_routing_rate",
                    "auto_or_review_complete_coverage",
                    "insufficient_rate",
                )
            },
            "per_repeat": [
                {
                    "split_version": repeat["split_version"],
                    **triage,
                }
                for repeat, triage in zip(repeats, triage_rows, strict=True)
            ],
        },
    }


def evaluate_conformal_robustness(
    source_path: Path,
    *,
    reference_run_report: Path | None = None,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    primary_alpha: float = PRIMARY_ALPHA,
    review_alpha: float = 0.2,
    k: int = 5,
    token_budget: int = 1500,
    role_threshold: float = 0.55,
    l2: float = 4.0,
    iterations: int = 80,
) -> dict[str, Any]:
    if len(split_versions) < 2:
        raise ValueError("robustness audit requires at least two split versions")
    if len(split_versions) != len(set(split_versions)):
        raise ValueError("split versions must be unique")
    if primary_alpha not in alphas:
        raise ValueError("primary alpha must be present in alphas")
    if review_alpha not in alphas or review_alpha <= primary_alpha:
        raise ValueError(
            "review alpha must be present in alphas and exceed primary alpha"
        )

    prepared_source = prepare_conformal_source(
        source_path,
        k=k,
        token_budget=token_budget,
        role_threshold=role_threshold,
    )
    repeats: list[dict[str, Any]] = []
    source_metadata: dict[str, Any] | None = None
    for split_version in split_versions:
        report, _ = evaluate_conformal_sufficiency(
            source_path,
            reference_run_report=reference_run_report,
            split_version=split_version,
            alphas=alphas,
            primary_alpha=primary_alpha,
            k=k,
            token_budget=token_budget,
            role_threshold=role_threshold,
            l2=l2,
            iterations=iterations,
            prepared_source=prepared_source,
        )
        if source_metadata is None:
            source_metadata = report["metadata"]
        elif (
            report["metadata"]["source_sha256"] != source_metadata["source_sha256"]
            or report["metadata"]["reference_run"] != source_metadata["reference_run"]
        ):
            raise AssertionError("source provenance changed during robustness run")
        repeats.append(
            {
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
                "baseline": report["evaluation"][
                    "baseline_role_coverage_heuristic"
                ],
                "alphas": {
                    str(alpha): report["calibration"]["alphas"][str(alpha)][
                        "evaluation"
                    ]
                    for alpha in alphas
                },
            }
        )

    if source_metadata is None:
        raise AssertionError("robustness audit did not execute")
    aggregate = _build_aggregate(
        repeats,
        alphas=alphas,
        primary_alpha=primary_alpha,
        review_alpha=review_alpha,
    )
    primary_summary = aggregate["alphas"][str(primary_alpha)]
    paired = aggregate["primary_alpha_paired_differences"]
    repeat_count = len(repeats)
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "sufficiency_schema_version": SUFFICIENCY_SCHEMA_VERSION,
            "dataset": source_metadata["dataset"],
            "source_path_label": source_metadata["source_path_label"],
            "source_sha256": source_metadata["source_sha256"],
            "reference_run": source_metadata["reference_run"],
            "split_versions": list(split_versions),
            "repeat_count": repeat_count,
            "split_protocol": (
                "Each pre-registered SHA-256 split version independently assigns whole "
                "case families to 40% train, 30% calibration and 30% evaluation. "
                "No evaluation result selects a split, alpha or model."
            ),
            "source_preparation": {
                "jsonl_parse_passes": 1,
                "split_independent_features_reused": True,
                "case_split_reassigned_in_memory_per_repeat": True,
            },
            "alphas": list(alphas),
            "primary_alpha": primary_alpha,
            "review_alpha": review_alpha,
            "selector": source_metadata["selector"],
            "label_definition": source_metadata["label_definition"],
            "feature_leakage_control": source_metadata["feature_leakage_control"],
        },
        "repeats": repeats,
        "aggregate": aggregate,
        "decision": {
            "safety_reduction_replicated_in_every_split": (
                paired[
                    "repeats_with_lower_case_family_false_complete_than_baseline"
                ]
                == repeat_count
            ),
            "primary_alpha_case_family_rate_at_or_below_alpha_in_every_split": (
                primary_summary["case_family_rate_at_or_below_alpha_repeats"]
                == repeat_count
            ),
            "primary_alpha_mean_abstention_rate": primary_summary["metrics"][
                "abstention_rate"
            ]["mean"],
            "primary_alpha_mean_complete_declaration_recall": primary_summary[
                "metrics"
            ]["true_complete_declaration_rate"]["mean"],
            "three_state_triage_preserves_primary_auto_boundary": True,
            "three_state_mean_complete_auto_or_review_coverage": aggregate[
                "three_state_triage"
            ]["metrics"]["auto_or_review_complete_coverage"]["mean"],
            "three_state_mean_review_band_complete_capture": aggregate[
                "three_state_triage"
            ]["metrics"]["review_band_complete_capture_rate"]["mean"],
            "finding": (
                "Repeated grouped splits test whether the safety-utility finding is "
                "stable; they do not tune the head or create independent datasets."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
            "limitations": [
                "Repeated splits reuse the same ConditionalQA cases and are correlated descriptive checks, not confidence intervals.",
                "ConditionalQA is cross-domain and missing evidence is removed synthetically.",
                "No repeated-split result replaces double-expert flood-domain completeness labels.",
                "Evaluation outcomes do not select alpha, regularization, features or a preferred split.",
            ],
        },
    }


def render_conformal_robustness_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    aggregate = report["aggregate"]
    decision = report["decision"]
    baseline = aggregate["baseline_role_coverage_heuristic"]
    lines = [
        "# FRC-RAG 证据充分性重复分组稳健性审计",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 数据：{metadata['dataset']}",
        f"- 预注册重复分组：{metadata['repeat_count']} 组",
        f"- 冻结分数 SHA-256：`{metadata['source_sha256']}`",
        f"- 主分析 alpha：{metadata['primary_alpha']}",
        f"- Gate 2：`{decision['gate_2']}`",
        "",
        "## 风险—效用跨分组分布",
        "",
        "| 判定器 | case 家族误放行率均值 [min, max] | 变体误放行率均值 | 拒答率均值 | 完整召回率均值 | 放行精度均值 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| 角色覆盖启发式 | "
            f"{baseline['false_complete_case_family_rate']['mean']:.6f} "
            f"[{baseline['false_complete_case_family_rate']['min']:.6f}, "
            f"{baseline['false_complete_case_family_rate']['max']:.6f}] | "
            f"{baseline['false_complete_rate_all_incomplete']['mean']:.6f} | "
            f"{baseline['abstention_rate']['mean']:.6f} | "
            f"{baseline['true_complete_declaration_rate']['mean']:.6f} | "
            f"{baseline['complete_declaration_precision']['mean']:.6f} |"
        ),
    ]
    for alpha, item in aggregate["alphas"].items():
        metrics = item["metrics"]
        lines.append(
            f"| Split-conformal alpha={float(alpha):.2f} | "
            f"{metrics['false_complete_case_family_rate']['mean']:.6f} "
            f"[{metrics['false_complete_case_family_rate']['min']:.6f}, "
            f"{metrics['false_complete_case_family_rate']['max']:.6f}] | "
            f"{metrics['false_complete_rate_all_incomplete']['mean']:.6f} | "
            f"{metrics['abstention_rate']['mean']:.6f} | "
            f"{metrics['true_complete_declaration_rate']['mean']:.6f} | "
            f"{metrics['complete_declaration_precision']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 主分析逐分组结果",
            "",
            "| 分组版本 | evaluation case | AUC | case 家族误放行率 | 拒答率 | 完整召回率 | 放行精度 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    primary_key = str(metadata["primary_alpha"])
    for repeat in report["repeats"]:
        metrics = repeat["alphas"][primary_key]
        lines.append(
            f"| `{repeat['split_version']}` | "
            f"{repeat['split_summary']['evaluation']['cases']} | "
            f"{repeat['evaluation_auc']:.6f} | "
            f"{metrics['false_complete_case_family_rate']:.6f} | "
            f"{metrics['abstention_rate']:.6f} | "
            f"{metrics['true_complete_declaration_rate']:.6f} | "
            f"{metrics['complete_declaration_precision']:.6f} |"
        )
    paired = aggregate["primary_alpha_paired_differences"]
    primary = aggregate["alphas"][primary_key]
    triage = aggregate["three_state_triage"]
    triage_metrics = triage["metrics"]
    lines.extend(
        [
            "",
            "## 判定",
            "",
            (
                f"- 主分析在 {paired['repeats_with_lower_case_family_false_complete_than_baseline']}/"
                f"{paired['repeat_count']} 组中低于角色覆盖启发式。"
            ),
            (
                f"- case 家族误放行率在 {primary['case_family_rate_at_or_below_alpha_repeats']}/"
                f"{metadata['repeat_count']} 组中不高于预注册 alpha。"
            ),
            (
                f"- 平均拒答率为 {decision['primary_alpha_mean_abstention_rate']:.6f}，"
                "平均完整召回率为 "
                f"{decision['primary_alpha_mean_complete_declaration_recall']:.6f}。"
            ),
            "",
            "## 三态人工复核分层",
            "",
            "| 状态 | 分数边界 | 平均占比 | 解释 |",
            "|---|---|---:|---|",
            (
                f"| `HIGH_CONFIDENCE_CANDIDATE` | alpha={triage['auto_complete_alpha']:.2f} "
                f"严格阈值以上 | {triage_metrics['auto_complete_rate']['mean']:.6f} | "
                "保持主安全边界；Gate 2 阻断自动完成，仍需人工复核 |"
            ),
            (
                f"| `REVIEW_PRIORITY` | alpha={triage['review_eligible_alpha']:.2f} "
                "与主阈值之间 | "
                f"{triage_metrics['review_band_rate']['mean']:.6f} | "
                "只提高人工复核优先级，不自动放行 |"
            ),
            (
                "| `INSUFFICIENT` | 低于人工复核带 | "
                f"{triage_metrics['insufficient_rate']['mean']:.6f} | "
                "维持证据不足/继续检索 |"
            ),
            "",
            (
                "人工复核优先带平均额外覆盖完整变体 "
                f"`{triage_metrics['review_band_complete_capture_rate']['mean']:.6f}`，"
                "HIGH_CONFIDENCE_CANDIDATE 与 REVIEW_PRIORITY 合计覆盖完整变体 "
                f"`{triage_metrics['auto_or_review_complete_coverage']['mean']:.6f}`；"
                "复核带本身精度为 "
                f"`{triage_metrics['review_band_precision']['mean']:.6f}`。"
            ),
            "",
            "重复分组用于检查结论是否依赖单次划分，不用于挑选最有利分组、alpha 或模型。"
            "这些分组复用同一公开数据，彼此相关，不能解释为独立重复试验或置信区间。",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in decision["limitations"])
    lines.extend(
        [
            "",
            "系统继续保持 `NO-GO/SHADOW`；真实防汛双专家标注、独立行为评判和"
            "可接受的安全—效用折中仍未完成。",
            "",
        ]
    )
    return "\n".join(lines)


def write_conformal_robustness(
    report: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> tuple[Path, Path]:
    for path in (json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(
        render_conformal_robustness_markdown(report), encoding="utf-8"
    )
    return json_path, markdown_path


def validate_conformal_robustness(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported conformal robustness schema")
    repeats = report.get("repeats", [])
    if len(repeats) != int(metadata.get("repeat_count", -1)):
        raise ValueError("repeat count does not match robustness metadata")
    split_versions = [item["split_version"] for item in repeats]
    if split_versions != metadata.get("split_versions"):
        raise ValueError("split versions do not match robustness metadata")
    alphas = tuple(float(value) for value in metadata["alphas"])
    recomputed = _build_aggregate(
        repeats,
        alphas=alphas,
        primary_alpha=float(metadata["primary_alpha"]),
        review_alpha=float(metadata["review_alpha"]),
    )
    if recomputed != report.get("aggregate"):
        raise ValueError("robustness aggregate does not match repeat records")
    return report


def load_conformal_robustness(path: Path) -> dict[str, Any]:
    return validate_conformal_robustness(
        json.loads(path.read_text(encoding="utf-8"))
    )
