"""Untuned cross-dataset confirmation of the sufficiency safety finding."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from research.frc_rag.conformal_robustness import (
    DEFAULT_SPLIT_VERSIONS,
    evaluate_conformal_robustness,
    load_conformal_robustness,
    validate_conformal_robustness,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-cross-dataset-confirmation-v1"
SERIES_SCHEMA_VERSION = "frc-conformal-cross-dataset-series-v1"
STATUS_PREFIX = "RUN_PUBLIC_REAL_MODEL"
SERIES_STATUS = "AGGREGATE_PUBLIC_REAL_MODEL_CROSS_DATASET_CONFIRMATIONS"


def dataset_slug(dataset: str) -> str:
    """Return a stable file/status-safe dataset identifier."""

    slug = re.sub(r"[^a-z0-9]+", "", dataset.lower())
    if not slug:
        raise ValueError("confirmation dataset requires an alphanumeric name")
    return slug


def _status_for_dataset(dataset: str) -> str:
    return f"{STATUS_PREFIX}_{dataset_slug(dataset).upper()}_CONFIRMATION"


def _reference_snapshot(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    primary_key = str(report["metadata"]["primary_alpha"])
    alpha = report["aggregate"]["alphas"][primary_key]
    return {
        "report_path_label": path.name,
        "report_sha256": sha256(path),
        "dataset": report["metadata"]["dataset"],
        "source_sha256": report["metadata"]["source_sha256"],
        "repeat_count": report["metadata"]["repeat_count"],
        "split_versions": report["metadata"]["split_versions"],
        "primary_alpha": report["metadata"]["primary_alpha"],
        "baseline_case_family_false_complete_mean": report["aggregate"][
            "baseline_role_coverage_heuristic"
        ]["false_complete_case_family_rate"]["mean"],
        "conformal_case_family_false_complete_mean": alpha["metrics"][
            "false_complete_case_family_rate"
        ]["mean"],
        "conformal_case_family_false_complete_range": [
            alpha["metrics"]["false_complete_case_family_rate"]["min"],
            alpha["metrics"]["false_complete_case_family_rate"]["max"],
        ],
        "conformal_abstention_mean": alpha["metrics"]["abstention_rate"]["mean"],
        "conformal_complete_recall_mean": alpha["metrics"][
            "true_complete_declaration_rate"
        ]["mean"],
        "safety_reduction_repeats": report["aggregate"][
            "primary_alpha_paired_differences"
        ]["repeats_with_lower_case_family_false_complete_than_baseline"],
        "case_family_rate_at_or_below_alpha_repeats": alpha[
            "case_family_rate_at_or_below_alpha_repeats"
        ],
    }


def _confirmation_outcome(
    reference: dict[str, Any],
    confirmation: dict[str, Any],
    *,
    required_safety_reduction_repeats: int,
    required_alpha_control_repeats: int,
    max_mean_case_family_rate: float,
) -> dict[str, Any]:
    primary_key = str(confirmation["metadata"]["primary_alpha"])
    alpha = confirmation["aggregate"]["alphas"][primary_key]
    baseline = confirmation["aggregate"]["baseline_role_coverage_heuristic"]
    paired = confirmation["aggregate"]["primary_alpha_paired_differences"]
    confirmation_metrics = {
        "baseline_case_family_false_complete_mean": baseline[
            "false_complete_case_family_rate"
        ]["mean"],
        "conformal_case_family_false_complete_mean": alpha["metrics"][
            "false_complete_case_family_rate"
        ]["mean"],
        "conformal_case_family_false_complete_range": [
            alpha["metrics"]["false_complete_case_family_rate"]["min"],
            alpha["metrics"]["false_complete_case_family_rate"]["max"],
        ],
        "conformal_abstention_mean": alpha["metrics"]["abstention_rate"]["mean"],
        "conformal_complete_recall_mean": alpha["metrics"][
            "true_complete_declaration_rate"
        ]["mean"],
        "safety_reduction_repeats": paired[
            "repeats_with_lower_case_family_false_complete_than_baseline"
        ],
        "case_family_rate_at_or_below_alpha_repeats": alpha[
            "case_family_rate_at_or_below_alpha_repeats"
        ],
    }
    checks = {
        "safety_reduction_repeats_meet_pre_registered_minimum": (
            confirmation_metrics["safety_reduction_repeats"]
            >= required_safety_reduction_repeats
        ),
        "alpha_control_repeats_meet_pre_registered_minimum": (
            confirmation_metrics["case_family_rate_at_or_below_alpha_repeats"]
            >= required_alpha_control_repeats
        ),
        "mean_case_family_false_complete_rate_at_or_below_alpha": (
            confirmation_metrics["conformal_case_family_false_complete_mean"]
            <= max_mean_case_family_rate
        ),
        "confirmation_dataset_differs_from_reference": (
            confirmation["metadata"]["dataset"] != reference["dataset"]
        ),
        "split_versions_frozen_from_reference": (
            confirmation["metadata"]["split_versions"]
            == reference["split_versions"]
        ),
    }
    comparison = {
        "confirmation_minus_reference_case_family_false_complete_mean": round(
            confirmation_metrics["conformal_case_family_false_complete_mean"]
            - reference["conformal_case_family_false_complete_mean"],
            6,
        ),
        "confirmation_minus_reference_abstention_mean": round(
            confirmation_metrics["conformal_abstention_mean"]
            - reference["conformal_abstention_mean"],
            6,
        ),
        "confirmation_minus_reference_complete_recall_mean": round(
            confirmation_metrics["conformal_complete_recall_mean"]
            - reference["conformal_complete_recall_mean"],
            6,
        ),
    }
    fully_confirmed = all(checks.values())
    partial_confirmed = (
        checks["safety_reduction_repeats_meet_pre_registered_minimum"]
        and checks["mean_case_family_false_complete_rate_at_or_below_alpha"]
        and checks["confirmation_dataset_differs_from_reference"]
        and checks["split_versions_frozen_from_reference"]
    )
    return {
        "confirmation_metrics": confirmation_metrics,
        "checks": checks,
        "comparison": comparison,
        "confirmation_status": (
            "FULL_CONFIRMATION"
            if fully_confirmed
            else "PARTIAL_CONFIRMATION"
            if partial_confirmed
            else "NOT_CONFIRMED"
        ),
        "mean_safety_signal_generalized": partial_confirmed,
        "repeat_consistency_target_met": checks[
            "alpha_control_repeats_meet_pre_registered_minimum"
        ],
        "cross_dataset_safety_signal_confirmed": fully_confirmed,
    }


def evaluate_cross_dataset_confirmation(
    reference_robustness_path: Path,
    confirmation_source_path: Path,
    *,
    reference_run_report: Path | None = None,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    safety_reduction_fraction: float = 1.0,
    alpha_control_fraction: float = 0.7,
    max_mean_case_family_rate: float = 0.1,
) -> dict[str, Any]:
    if not 0.0 < safety_reduction_fraction <= 1.0:
        raise ValueError("safety reduction fraction must be in (0, 1]")
    if not 0.0 < alpha_control_fraction <= 1.0:
        raise ValueError("alpha control fraction must be in (0, 1]")
    reference_report = load_conformal_robustness(reference_robustness_path)
    reference = _reference_snapshot(reference_robustness_path, reference_report)
    if tuple(reference["split_versions"]) != split_versions:
        raise ValueError("confirmation split versions must match the reference")
    confirmation = evaluate_conformal_robustness(
        confirmation_source_path,
        reference_run_report=reference_run_report,
        split_versions=split_versions,
    )
    confirmation_dataset = confirmation["metadata"]["dataset"]
    repeat_count = len(split_versions)
    required_safety = math.ceil(repeat_count * safety_reduction_fraction)
    required_alpha = math.ceil(repeat_count * alpha_control_fraction)
    hypotheses = {
        "declared_before_confirmation_run": True,
        "required_safety_reduction_repeats": required_safety,
        "required_alpha_control_repeats": required_alpha,
        "max_mean_case_family_false_complete_rate": max_mean_case_family_rate,
        "no_confirmation_feature_or_model_selection": True,
    }
    outcome = _confirmation_outcome(
        reference,
        confirmation,
        required_safety_reduction_repeats=required_safety,
        required_alpha_control_repeats=required_alpha,
        max_mean_case_family_rate=max_mean_case_family_rate,
    )
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": _status_for_dataset(confirmation_dataset),
            "reference_dataset": reference["dataset"],
            "confirmation_dataset": confirmation_dataset,
            "confirmation_source_path_label": confirmation_source_path.name,
            "confirmation_source_sha256": confirmation["metadata"]["source_sha256"],
            "repeat_count": repeat_count,
            "split_versions": list(split_versions),
            "primary_alpha": confirmation["metadata"]["primary_alpha"],
            "protocol": (
                "Reuse the frozen linear L2=4 sufficiency protocol, alpha and split "
                "versions on an untouched public dataset. No confirmation result "
                "selects a feature, regularization value, alpha or split."
            ),
        },
        "pre_registered_hypotheses": hypotheses,
        "reference": reference,
        "confirmation_robustness": confirmation,
        "outcome": outcome,
        "decision": {
            "cross_dataset_safety_signal_confirmed": outcome[
                "cross_dataset_safety_signal_confirmed"
            ],
            "confirmation_status": outcome["confirmation_status"],
            "finding": (
                "Confirmation concerns the conservative sufficiency/abstention "
                "signal, not FRC retrieval superiority or production readiness."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
            "limitations": [
                (
                    f"{confirmation_dataset} is a public QA corpus, not a district "
                    "flood corpus."
                ),
                (
                    "The model is refit and calibrated within the confirmation "
                    "dataset grouped splits; this is protocol transfer, not "
                    "zero-shot weight transfer."
                ),
                (
                    "Missing evidence remains deterministic synthetic removal from "
                    "frozen candidates."
                ),
                (
                    "Cross-dataset confirmation does not replace flood-domain "
                    "double-expert labels or independent behavior review."
                ),
            ],
        },
    }


def render_cross_dataset_confirmation_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    reference = report["reference"]
    outcome = report["outcome"]
    confirmation = outcome["confirmation_metrics"]
    reference_dataset = metadata["reference_dataset"]
    confirmation_dataset = metadata["confirmation_dataset"]
    lines = [
        f"# FRC-RAG 证据充分性 {confirmation_dataset} 跨数据集确认",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 参考数据：{reference_dataset}",
        f"- 确认数据：{confirmation_dataset}",
        f"- 预注册重复分组：{metadata['repeat_count']} 组",
        f"- alpha：{metadata['primary_alpha']}",
        f"- Gate 2：`{report['decision']['gate_2']}`",
        "",
        "## 跨数据集结果",
        "",
        (
            f"| 指标 | {reference_dataset} 参考 | "
            f"{confirmation_dataset} 确认 | 差值 |"
        ),
        "|---|---:|---:|---:|",
        (
            "| case 家族误放行率均值 | "
            f"{reference['conformal_case_family_false_complete_mean']:.6f} | "
            f"{confirmation['conformal_case_family_false_complete_mean']:.6f} | "
            f"{outcome['comparison']['confirmation_minus_reference_case_family_false_complete_mean']:+.6f} |"
        ),
        (
            "| 拒答率均值 | "
            f"{reference['conformal_abstention_mean']:.6f} | "
            f"{confirmation['conformal_abstention_mean']:.6f} | "
            f"{outcome['comparison']['confirmation_minus_reference_abstention_mean']:+.6f} |"
        ),
        (
            "| 完整召回率均值 | "
            f"{reference['conformal_complete_recall_mean']:.6f} | "
            f"{confirmation['conformal_complete_recall_mean']:.6f} | "
            f"{outcome['comparison']['confirmation_minus_reference_complete_recall_mean']:+.6f} |"
        ),
        "",
        "## 预注册确认检查",
        "",
    ]
    for name, passed in outcome["checks"].items():
        lines.append(f"- `{name}`：`{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            (
                f"确认状态：`{outcome['confirmation_status']}`；"
                "完整跨数据集确认："
                f"`{outcome['cross_dataset_safety_signal_confirmed']}`。"
            ),
            "",
            (
                f"{confirmation_dataset} 上相对角色覆盖启发式的安全改善出现在 "
                f"{confirmation['safety_reduction_repeats']}/"
                f"{metadata['repeat_count']} 组；实测 case 风险不高于 alpha "
                f"的分组为 "
                f"{confirmation['case_family_rate_at_or_below_alpha_repeats']}/"
                f"{metadata['repeat_count']}。"
            ),
            "",
            (
                "该确认只检验固定安全协议是否跨公开数据复现，不进行候选选型，"
                "也不证明 FRC 检索优于公平基线。系统继续保持 "
                "`NO-GO/SHADOW`。"
            ),
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["decision"]["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_cross_dataset_confirmation(
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
        render_cross_dataset_confirmation_markdown(report), encoding="utf-8"
    )
    return json_path, markdown_path


def load_cross_dataset_confirmation(
    path: Path, *, reference_robustness_path: Path | None = None
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported cross-dataset confirmation schema")
    confirmation = validate_conformal_robustness(
        report["confirmation_robustness"]
    )
    expected_status = _status_for_dataset(confirmation["metadata"]["dataset"])
    if report["metadata"].get("status") != expected_status:
        raise ValueError("cross-dataset status does not match confirmation dataset")
    reference = report["reference"]
    if reference_robustness_path is not None:
        reference_report = load_conformal_robustness(reference_robustness_path)
        expected_reference = _reference_snapshot(
            reference_robustness_path, reference_report
        )
        if reference != expected_reference:
            raise ValueError("reference robustness snapshot does not match source")
    hypotheses = report["pre_registered_hypotheses"]
    recomputed = _confirmation_outcome(
        reference,
        confirmation,
        required_safety_reduction_repeats=int(
            hypotheses["required_safety_reduction_repeats"]
        ),
        required_alpha_control_repeats=int(
            hypotheses["required_alpha_control_repeats"]
        ),
        max_mean_case_family_rate=float(
            hypotheses["max_mean_case_family_false_complete_rate"]
        ),
    )
    if recomputed != report.get("outcome"):
        raise ValueError("cross-dataset outcome does not match embedded evidence")
    if report["decision"].get("confirmation_status") != recomputed[
        "confirmation_status"
    ]:
        raise ValueError("cross-dataset decision does not match outcome")
    if report["decision"].get(
        "cross_dataset_safety_signal_confirmed"
    ) != recomputed["cross_dataset_safety_signal_confirmed"]:
        raise ValueError("cross-dataset decision confirmation flag does not match")
    return report


def evaluate_cross_dataset_series(
    confirmation_paths: tuple[Path, ...],
    *,
    reference_robustness_path: Path,
) -> dict[str, Any]:
    """Aggregate independent confirmations without changing their decision rules."""

    if len(confirmation_paths) < 2:
        raise ValueError("cross-dataset series requires at least two confirmations")
    loaded = [
        (
            path,
            load_cross_dataset_confirmation(
                path,
                reference_robustness_path=reference_robustness_path,
            ),
        )
        for path in confirmation_paths
    ]
    datasets = [report["metadata"]["confirmation_dataset"] for _, report in loaded]
    if len(datasets) != len(set(datasets)):
        raise ValueError("cross-dataset series requires unique confirmation datasets")
    references = [report["reference"] for _, report in loaded]
    if any(reference != references[0] for reference in references[1:]):
        raise ValueError("cross-dataset series reports must share one reference")
    hypotheses = [
        report["pre_registered_hypotheses"] for _, report in loaded
    ]
    if any(hypothesis != hypotheses[0] for hypothesis in hypotheses[1:]):
        raise ValueError("cross-dataset series reports must share one decision rule")

    entries = []
    for path, report in loaded:
        outcome = report["outcome"]
        metrics = outcome["confirmation_metrics"]
        entries.append(
            {
                "dataset": report["metadata"]["confirmation_dataset"],
                "report_path_label": path.name,
                "report_sha256": sha256(path),
                "source_sha256": report["metadata"][
                    "confirmation_source_sha256"
                ],
                "confirmation_status": outcome["confirmation_status"],
                "full_confirmation": outcome[
                    "cross_dataset_safety_signal_confirmed"
                ],
                "mean_safety_signal_generalized": outcome[
                    "mean_safety_signal_generalized"
                ],
                "conformal_case_family_false_complete_mean": metrics[
                    "conformal_case_family_false_complete_mean"
                ],
                "conformal_abstention_mean": metrics[
                    "conformal_abstention_mean"
                ],
                "conformal_complete_recall_mean": metrics[
                    "conformal_complete_recall_mean"
                ],
                "safety_reduction_repeats": metrics[
                    "safety_reduction_repeats"
                ],
                "case_family_rate_at_or_below_alpha_repeats": metrics[
                    "case_family_rate_at_or_below_alpha_repeats"
                ],
                "checks": outcome["checks"],
            }
        )
    entries.sort(key=lambda item: item["dataset"])

    statuses = [entry["confirmation_status"] for entry in entries]
    if all(status == "FULL_CONFIRMATION" for status in statuses):
        series_status = "FULL_CONFIRMATION"
    elif all(
        status in {"FULL_CONFIRMATION", "PARTIAL_CONFIRMATION"}
        for status in statuses
    ):
        series_status = "PARTIAL_CONFIRMATION"
    else:
        series_status = "NOT_CONFIRMED"
    full_count = sum(entry["full_confirmation"] for entry in entries)
    partial_count = statuses.count("PARTIAL_CONFIRMATION")
    not_confirmed_count = statuses.count("NOT_CONFIRMED")
    return {
        "metadata": {
            "schema_version": SERIES_SCHEMA_VERSION,
            "status": SERIES_STATUS,
            "reference_dataset": references[0]["dataset"],
            "confirmation_dataset_count": len(entries),
            "primary_alpha": references[0]["primary_alpha"],
            "repeat_count_per_dataset": references[0]["repeat_count"],
            "aggregation_policy": (
                "FULL only when every registered confirmation is FULL; PARTIAL "
                "when every result is FULL or PARTIAL but at least one is not "
                "FULL; otherwise NOT_CONFIRMED."
            ),
        },
        "per_dataset_decision_rule": hypotheses[0],
        "confirmations": entries,
        "outcome": {
            "confirmation_status": series_status,
            "full_confirmation_count": full_count,
            "partial_confirmation_count": partial_count,
            "not_confirmed_count": not_confirmed_count,
            "all_confirmation_datasets_full": full_count == len(entries),
            "all_mean_safety_signals_generalized": all(
                entry["mean_safety_signal_generalized"] for entry in entries
            ),
            "cross_dataset_series_confirmed": series_status
            == "FULL_CONFIRMATION",
        },
        "decision": {
            "finding": (
                "The series summarizes frozen public-dataset safety protocol "
                "transfer. It does not establish retrieval superiority, flood-domain "
                "validity or production readiness."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
        },
    }


def render_cross_dataset_series_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG 证据充分性跨数据集确认系列",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 参考数据：{metadata['reference_dataset']}",
        f"- 确认数据集数：{metadata['confirmation_dataset_count']}",
        f"- alpha：{metadata['primary_alpha']}",
        f"- 系列结论：`{outcome['confirmation_status']}`",
        f"- Gate 2：`{report['decision']['gate_2']}`",
        "",
        "## 独立确认结果",
        "",
        (
            "| 数据集 | 结论 | case 风险均值 | 拒答率 | 完整召回率 | "
            "安全改善分组 | 风险不高于 alpha 分组 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    repeat_count = metadata["repeat_count_per_dataset"]
    for item in report["confirmations"]:
        lines.append(
            f"| {item['dataset']} | `{item['confirmation_status']}` | "
            f"{item['conformal_case_family_false_complete_mean']:.6f} | "
            f"{item['conformal_abstention_mean']:.6f} | "
            f"{item['conformal_complete_recall_mean']:.6f} | "
            f"{item['safety_reduction_repeats']}/{repeat_count} | "
            f"{item['case_family_rate_at_or_below_alpha_repeats']}/"
            f"{repeat_count} |"
        )
    lines.extend(
        [
            "",
            (
                f"完整确认 {outcome['full_confirmation_count']}/"
                f"{metadata['confirmation_dataset_count']}；部分确认 "
                f"{outcome['partial_confirmation_count']}/"
                f"{metadata['confirmation_dataset_count']}。"
            ),
            "",
            (
                "系列规则不允许用一个数据集的通过覆盖另一个数据集的失败。"
                "本报告不改变 `NO-GO/SHADOW`，也不能替代真实防汛领域双专家标注。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_cross_dataset_series(
    report: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> tuple[Path, Path]:
    for path in (json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_cross_dataset_series_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path


def load_cross_dataset_series(
    path: Path,
    *,
    confirmation_paths: tuple[Path, ...],
    reference_robustness_path: Path,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("metadata", {}).get("schema_version") != SERIES_SCHEMA_VERSION:
        raise ValueError("unsupported cross-dataset series schema")
    recomputed = evaluate_cross_dataset_series(
        confirmation_paths,
        reference_robustness_path=reference_robustness_path,
    )
    if recomputed != report:
        raise ValueError("cross-dataset series does not match source reports")
    return report
