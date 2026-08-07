"""Candidate-size-aware multi-axis Mondrian development for FRC sufficiency."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from research.frc_rag.conformal_mondrian import (
    ADOPTION_TARGETS,
    MIN_ELIGIBLE_REPEATS,
    MIN_GROUP_CALIBRATION_UNITS,
    MIN_INCOMPLETE_CASE_FAMILIES,
    _analyze_case_records,
    _build_outcome,
    evaluate_hierarchical_mondrian,
)
from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_sufficiency import PRIMARY_ALPHA
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-multi-axis-mondrian-development-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_POST_HOC_MULTI_AXIS_MONDRIAN_DEVELOPMENT"
FALLBACK_ORDER = (
    "question_type_candidate_count_bucket_required_role_count",
    "question_type_candidate_count_bucket",
    "candidate_count_bucket_required_role_count",
    "question_type_required_role_count",
    "candidate_count_bucket",
    "question_type",
    "required_role_count",
    "global",
)
GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "question_type_candidate_count_bucket_required_role_count": (
        "question_type",
        "candidate_count_bucket",
        "required_role_count",
    ),
    "question_type_candidate_count_bucket": (
        "question_type",
        "candidate_count_bucket",
    ),
    "candidate_count_bucket_required_role_count": (
        "candidate_count_bucket",
        "required_role_count",
    ),
    "question_type_required_role_count": (
        "question_type",
        "required_role_count",
    ),
    "candidate_count_bucket": ("candidate_count_bucket",),
    "question_type": ("question_type",),
    "required_role_count": ("required_role_count",),
    "global": (),
}


def evaluate_multi_axis_mondrian(
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the frozen post-hoc eight-level observable-group hierarchy."""
    report, records = evaluate_hierarchical_mondrian(
        dataset_sources,
        expected_robustness,
        reference_run_report=reference_run_report,
        split_versions=split_versions,
        alpha=alpha,
        min_group_calibration_units=min_group_calibration_units,
        min_incomplete_case_families=min_incomplete_case_families,
        min_eligible_repeats=min_eligible_repeats,
        adoption_targets=adoption_targets,
        fallback_order=FALLBACK_ORDER,
        group_fields=GROUP_FIELDS,
        schema_version=SCHEMA_VERSION,
        status=STATUS,
    )
    report["development_protocol"].update(
        {
            "candidate_size_aware": True,
            "prompted_by_candidate_count_and_question_type_instability": True,
            "new_untouched_dataset_inspected_before_adoption_decision": False,
        }
    )
    report["decision"]["finding"] = (
        "This candidate-size-aware hierarchy is post-hoc method development. "
        "Only an all-checks development candidate may proceed to one untouched "
        "MuSiQue confirmation; failure stops this method-selection cycle."
    )
    report["decision"]["limitations"].insert(
        1,
        "The candidate-count axis was added after 2WikiMultiHopQA subgroup "
        "instability was observed.",
    )
    report["decision"]["next_step"] = (
        "FREEZE_AND_CONFIRM_ON_UNTOUCHED_MUSIQUE"
        if report["outcome"]["all_pre_registered_adoption_checks_passed"]
        else "STOP_WITHOUT_MUSIQUE_DOWNLOAD"
    )
    return report, records


def render_multi_axis_mondrian_markdown(report: dict[str, Any]) -> str:
    """Render the compact human-readable development report."""
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG 多轴 Mondrian conformal 事后方法开发",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 数据集：{', '.join(metadata['datasets'])}",
        f"- alpha：{metadata['primary_alpha']}",
        "- 回退顺序：" + " → ".join(metadata["fallback_order"]),
        (
            "- 分组阈值最低校准 case 数："
            f"{metadata['min_group_calibration_case_families']}"
        ),
        "- 事后方法开发：`True`；独立确认：`False`",
        f"- Gate 2：`{outcome['gate_2']}`",
        "",
        "## 与全局阈值的冻结比较",
        "",
        (
            "| 数据集 | 全局 case 风险 | 多轴 case 风险 | 全局完整召回 | "
            "多轴完整召回 | 最坏子群风险改善 | 分组阈值占比 |"
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
            "## 决策边界",
            "",
            (
                "该层级是在查看四数据集子群结果后冻结的事后开发规则，不能作为独立确认。"
                "只有全部采用检查通过才会下载并运行未触碰的 MuSiQue；否则保留负结果并停止扩张。"
            ),
            "",
            f"- 下一步：`{report['decision']['next_step']}`",
            "- 当前仍保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_multi_axis_mondrian(
    report: dict[str, Any],
    case_records: list[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    cases_path: Path,
) -> tuple[Path, Path, Path]:
    """Write deterministic JSON, Markdown and compressed case evidence."""
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
        render_multi_axis_mondrian_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path, cases_path


def load_multi_axis_mondrian(
    json_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    """Validate provenance and recompute all aggregates from case evidence."""
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported multi-axis Mondrian schema")
    expected_fields = {
        source: list(GROUP_FIELDS[source]) for source in FALLBACK_ORDER
    }
    if metadata.get("fallback_order") != list(FALLBACK_ORDER):
        raise ValueError("multi-axis Mondrian fallback order does not match")
    if metadata.get("group_fields") != expected_fields:
        raise ValueError("multi-axis Mondrian group fields do not match")
    artifact = metadata["case_artifact"]
    if artifact["path_label"] != cases_path.name:
        raise ValueError("multi-axis Mondrian case path label does not match")
    if artifact["sha256"] != sha256(cases_path):
        raise ValueError("multi-axis Mondrian case artifact hash does not match")
    records = []
    with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != artifact["record_count"]:
        raise ValueError("multi-axis Mondrian record count does not match")
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
        fallback_order=FALLBACK_ORDER,
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
        fallback_order=FALLBACK_ORDER,
    )
    if analyses != report.get("datasets") or outcome != report.get("outcome"):
        raise ValueError("multi-axis Mondrian aggregates do not match case artifact")
    return report
