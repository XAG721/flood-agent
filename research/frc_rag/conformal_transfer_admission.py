"""Calibration-only admission guard for cross-dataset sufficiency heads."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from research.frc_rag.conformal_head_transfer import (
    load_cross_dataset_head_transfer,
)
from research.frc_rag.conformal_sufficiency import (
    _auc,
    _conformal_threshold,
    _decision_metrics,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-transfer-admission-guard-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_POST_HOC_TRANSFER_ADMISSION_GUARD"
CERTIFICATE_TARGETS: dict[str, float | int] = {
    "minimum_incomplete_calibration_case_families": 30,
    "minimum_complete_calibration_variants": 30,
    "minimum_calibration_auc": 0.75,
    "minimum_calibration_complete_recall_at_frozen_threshold": 0.15,
}
ADOPTION_TARGETS: dict[str, float | int] = {
    "minimum_total_certified_repeats": 8,
    "minimum_datasets_with_at_least_one_certificate": 2,
    "maximum_unsafe_certificate_count": 0,
    "maximum_low_utility_certificate_count": 0,
    "minimum_safe_and_useful_certificate_precision": 1.0,
}
MINIMUM_USEFUL_EVALUATION_RECALL = 0.1


def _check(measured: Any, target: Any, passed: bool) -> dict[str, Any]:
    return {"measured": measured, "target": target, "passed": bool(passed)}


def issue_transfer_certificate(
    calibration_evidence: dict[str, float | int],
    *,
    certificate_targets: dict[str, float | int] = CERTIFICATE_TARGETS,
) -> dict[str, Any]:
    """Issue a certificate from calibration summaries only."""
    checks = {
        "minimum_incomplete_calibration_case_families": _check(
            int(calibration_evidence["incomplete_case_families"]),
            ">="
            + str(
                certificate_targets[
                    "minimum_incomplete_calibration_case_families"
                ]
            ),
            int(calibration_evidence["incomplete_case_families"])
            >= certificate_targets[
                "minimum_incomplete_calibration_case_families"
            ],
        ),
        "minimum_complete_calibration_variants": _check(
            int(calibration_evidence["complete_variants"]),
            ">="
            + str(
                certificate_targets["minimum_complete_calibration_variants"]
            ),
            int(calibration_evidence["complete_variants"])
            >= certificate_targets["minimum_complete_calibration_variants"],
        ),
        "minimum_calibration_auc": _check(
            float(calibration_evidence["auc"]),
            ">=" + str(certificate_targets["minimum_calibration_auc"]),
            float(calibration_evidence["auc"])
            >= certificate_targets["minimum_calibration_auc"],
        ),
        "minimum_calibration_complete_recall_at_frozen_threshold": _check(
            float(calibration_evidence["complete_recall_at_threshold"]),
            ">="
            + str(
                certificate_targets[
                    "minimum_calibration_complete_recall_at_frozen_threshold"
                ]
            ),
            float(calibration_evidence["complete_recall_at_threshold"])
            >= certificate_targets[
                "minimum_calibration_complete_recall_at_frozen_threshold"
            ],
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": (
            "CALIBRATION_CERTIFIED"
            if passed
            else "REJECT_TRANSFER_REQUIRE_TARGET_FIT"
        ),
        "certified": passed,
        "input_fields": sorted(calibration_evidence),
        "evaluation_fields_used": False,
        "target_train_fields_used": False,
        "checks": checks,
    }


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _calibration_evidence(
    records: list[dict[str, Any]],
    *,
    alpha: float,
) -> dict[str, float | int]:
    calibration = [item for item in records if item["split"] == "calibration"]
    if not calibration:
        raise ValueError("transfer admission input has no calibration records")
    labels = np.asarray(
        [bool(item["complete"]) for item in calibration],
        dtype=bool,
    )
    scores = np.asarray(
        [float(item["transferred_score"]) for item in calibration],
        dtype=np.float64,
    )
    case_maxima: dict[str, float] = {}
    for item in calibration:
        if bool(item["complete"]):
            continue
        case_id = str(item["case_id"])
        case_maxima[case_id] = max(
            case_maxima.get(case_id, -math.inf),
            float(item["transferred_score"]),
        )
    units = np.asarray(list(case_maxima.values()), dtype=np.float64)
    threshold = round(_conformal_threshold(units, alpha), 12)
    stored_thresholds = {
        round(float(item["transferred_threshold"]), 12) for item in calibration
    }
    if stored_thresholds != {threshold}:
        raise ValueError("frozen transferred threshold does not reaggregate")
    complete_count = int(labels.sum())
    complete_recall = round(
        int(((scores > threshold) & labels).sum()) / max(1, complete_count),
        6,
    )
    return {
        "incomplete_case_families": len(units),
        "complete_variants": complete_count,
        "incomplete_variants": int((~labels).sum()),
        "auc": round(_auc(labels, scores), 6),
        "threshold": threshold,
        "complete_recall_at_threshold": complete_recall,
    }


def _evaluation_evidence(
    records: list[dict[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    evaluation = [item for item in records if item["split"] == "evaluation"]
    if not evaluation:
        raise ValueError("transfer admission input has no evaluation records")
    decisions = np.asarray(
        [bool(item["transferred_declared_complete"]) for item in evaluation],
        dtype=bool,
    )
    recomputed = np.asarray(
        [
            float(item["transferred_score"])
            > float(item["transferred_threshold"])
            for item in evaluation
        ],
        dtype=bool,
    )
    if not np.array_equal(decisions, recomputed):
        raise ValueError("transferred evaluation decisions do not match threshold")
    metrics = _decision_metrics(evaluation, decisions)
    case_risk = float(metrics["false_complete_case_family_rate"])
    recall = float(metrics["true_complete_declaration_rate"])
    return {
        "false_complete_case_family_rate": case_risk,
        "true_complete_declaration_rate": recall,
        "safe_at_alpha": case_risk <= alpha,
        "useful_at_minimum_recall": recall
        >= MINIMUM_USEFUL_EVALUATION_RECALL,
    }


def _build_evidence_records(
    transfer_records: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    alpha: float,
    certificate_targets: dict[str, float | int],
) -> list[dict[str, Any]]:
    evidence = []
    for dataset in datasets:
        for split_version in split_versions:
            current = [
                item
                for item in transfer_records
                if item["dataset"] == dataset
                and item["split_version"] == split_version
            ]
            if not current:
                raise ValueError(
                    f"missing transfer admission input for {dataset}/{split_version}"
                )
            calibration = _calibration_evidence(current, alpha=alpha)
            certificate = issue_transfer_certificate(
                calibration,
                certificate_targets=certificate_targets,
            )
            evaluation = _evaluation_evidence(current, alpha=alpha)
            evidence.append(
                {
                    "dataset": dataset,
                    "split_version": split_version,
                    "calibration_evidence": calibration,
                    "certificate": certificate,
                    "retrospective_evaluation": evaluation,
                }
            )
    return evidence


def _build_outcome(
    evidence: list[dict[str, Any]],
    *,
    datasets: list[str],
    adoption_targets: dict[str, float | int],
) -> dict[str, Any]:
    certified = [item for item in evidence if item["certificate"]["certified"]]
    counts_by_dataset = {
        dataset: sum(item["dataset"] == dataset for item in certified)
        for dataset in datasets
    }
    datasets_with_certificate = sum(value > 0 for value in counts_by_dataset.values())
    unsafe = [
        item
        for item in certified
        if not item["retrospective_evaluation"]["safe_at_alpha"]
    ]
    low_utility = [
        item
        for item in certified
        if not item["retrospective_evaluation"]["useful_at_minimum_recall"]
    ]
    safe_and_useful = [
        item
        for item in certified
        if item["retrospective_evaluation"]["safe_at_alpha"]
        and item["retrospective_evaluation"]["useful_at_minimum_recall"]
    ]
    precision = round(len(safe_and_useful) / max(1, len(certified)), 6)
    certificate_inputs_valid = all(
        item["certificate"]["evaluation_fields_used"] is False
        and item["certificate"]["target_train_fields_used"] is False
        and not {
            "false_complete_case_family_rate",
            "true_complete_declaration_rate",
            "safe_at_alpha",
            "useful_at_minimum_recall",
        }
        & set(item["certificate"]["input_fields"])
        for item in evidence
    )
    checks = {
        "certificate_uses_calibration_only": _check(
            certificate_inputs_valid,
            True,
            certificate_inputs_valid,
        ),
        "minimum_total_certified_repeats": _check(
            len(certified),
            ">="
            + str(adoption_targets["minimum_total_certified_repeats"]),
            len(certified)
            >= adoption_targets["minimum_total_certified_repeats"],
        ),
        "minimum_datasets_with_at_least_one_certificate": _check(
            {
                "count": datasets_with_certificate,
                "per_dataset": counts_by_dataset,
            },
            ">="
            + str(
                adoption_targets[
                    "minimum_datasets_with_at_least_one_certificate"
                ]
            ),
            datasets_with_certificate
            >= adoption_targets[
                "minimum_datasets_with_at_least_one_certificate"
            ],
        ),
        "maximum_unsafe_certificate_count": _check(
            len(unsafe),
            "<="
            + str(adoption_targets["maximum_unsafe_certificate_count"]),
            len(unsafe)
            <= adoption_targets["maximum_unsafe_certificate_count"],
        ),
        "maximum_low_utility_certificate_count": _check(
            len(low_utility),
            "<="
            + str(adoption_targets["maximum_low_utility_certificate_count"]),
            len(low_utility)
            <= adoption_targets["maximum_low_utility_certificate_count"],
        ),
        "minimum_safe_and_useful_certificate_precision": _check(
            precision,
            ">="
            + str(
                adoption_targets[
                    "minimum_safe_and_useful_certificate_precision"
                ]
            ),
            bool(certified)
            and precision
            >= adoption_targets[
                "minimum_safe_and_useful_certificate_precision"
            ],
        ),
    }
    all_passed = all(item["passed"] for item in checks.values())
    return {
        "status": (
            "POST_HOC_ADMISSION_GUARD_CANDIDATE"
            if all_passed
            else "DO_NOT_ADOPT"
        ),
        "post_hoc_method_development": True,
        "independent_confirmation": False,
        "certificate_count": len(certified),
        "certificate_count_by_dataset": counts_by_dataset,
        "unsafe_certificate_count": len(unsafe),
        "low_utility_certificate_count": len(low_utility),
        "safe_and_useful_certificate_count": len(safe_and_useful),
        "safe_and_useful_certificate_precision": precision,
        "checks": checks,
        "all_pre_registered_adoption_checks_passed": all_passed,
        "gate_2": "NO-GO/SHADOW",
    }


def evaluate_transfer_admission_guard(
    transfer_report_path: Path,
    transfer_cases_path: Path,
    *,
    certificate_targets: dict[str, float | int] = CERTIFICATE_TARGETS,
    adoption_targets: dict[str, float | int] = ADOPTION_TARGETS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transfer = load_cross_dataset_head_transfer(
        transfer_report_path,
        transfer_cases_path,
    )
    metadata = transfer["metadata"]
    datasets = list(metadata["datasets"])
    split_versions = tuple(metadata["split_versions"])
    alpha = float(metadata["primary_alpha"])
    transfer_records = _read_gzip_jsonl(transfer_cases_path)
    evidence = _build_evidence_records(
        transfer_records,
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
        certificate_targets=certificate_targets,
    )
    outcome = _build_outcome(
        evidence,
        datasets=datasets,
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
            "minimum_useful_evaluation_recall": (
                MINIMUM_USEFUL_EVALUATION_RECALL
            ),
            "evidence_artifact": {},
        },
        "input_provenance": {
            "transfer_report_path_label": transfer_report_path.name,
            "transfer_report_sha256": sha256(transfer_report_path),
            "transfer_case_artifact_path_label": transfer_cases_path.name,
            "transfer_case_artifact_sha256": sha256(transfer_cases_path),
            "transfer_case_record_count": metadata["case_artifact"][
                "record_count"
            ],
            "transfer_report_validated_before_guard": True,
        },
        "development_protocol": {
            "post_hoc": True,
            "independent_confirmation": False,
            "admission_only_no_score_threshold_or_declaration_change": True,
            "certificate_uses_calibration_only": True,
            "evaluation_used_to_issue_certificate": False,
            "target_train_used_to_issue_certificate": False,
            "certificate_targets": certificate_targets,
            "adoption_targets": adoption_targets,
        },
        "evidence": evidence,
        "outcome": outcome,
        "decision": {
            "finding": (
                "The guard is evaluated as a fail-closed admission mechanism for "
                "an already frozen transferred head."
            ),
            "next_step": (
                "FREEZE_BEFORE_NEW_UNTOUCHED_GUARD_CONFIRMATION"
                if outcome["all_pre_registered_adoption_checks_passed"]
                else "REQUIRE_TARGET_FITTED_HEAD_STOP_GUARD_TUNING"
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "REQUIRE_TARGET_FITTED_HEAD",
            "limitations": [
                "The guard was designed after transfer evaluation failures were known.",
                "Calibration AUC and recall are not independent validation estimates.",
                "The guard cannot improve the transferred head's score ranking.",
                "No result authorizes CANARY or DEFAULT.",
            ],
        },
    }
    return report, evidence


def render_transfer_admission_markdown(report: dict[str, Any]) -> str:
    outcome = report["outcome"]
    metadata = report["metadata"]
    lines = [
        "# FRC-RAG 跨域充分性头 calibration-only 准入守卫",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 数据集：{', '.join(metadata['datasets'])}",
        f"- 重复总数：{len(report['evidence'])}",
        "- 证书使用 evaluation：`False`",
        "- 证书使用目标 train：`False`",
        "- 修改模型/阈值/声明：`False`",
        f"- Gate 2：`{outcome['gate_2']}`",
        "",
        "## 证书结果",
        "",
        f"- 通过证书：{outcome['certificate_count']}",
        f"- 数据集分布：`{json.dumps(outcome['certificate_count_by_dataset'], ensure_ascii=False)}`",
        f"- 误准入（evaluation 风险 > alpha）：{outcome['unsafe_certificate_count']}",
        f"- 低效用准入（完整召回 < 0.10）：{outcome['low_utility_certificate_count']}",
        (
            "- 安全且有效证书精度："
            f"{outcome['safe_and_useful_certificate_precision']:.6f}"
        ),
        "",
        "## 预注册采用检查",
        "",
        "| 检查 | 实测 | 目标 | 通过 |",
        "|---|---|---|---|",
    ]
    for name, check in outcome["checks"].items():
        measured = json.dumps(check["measured"], ensure_ascii=False)
        lines.append(
            f"| {name} | `{measured}` | `{check['target']}` | "
            f"`{check['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            (
                "该守卫只能拒绝跨域头，不能改善其排序能力或加强 conformal 保证。"
                "证书在已知迁移失败后设计，属于事后方法开发。"
            ),
            "",
            f"- 下一步：`{report['decision']['next_step']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_transfer_admission_guard(
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
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as archive:
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
    markdown_path.write_text(
        render_transfer_admission_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path, evidence_path


def load_transfer_admission_guard(
    json_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported transfer admission guard schema")
    artifact = metadata["evidence_artifact"]
    if artifact["path_label"] != evidence_path.name:
        raise ValueError("transfer admission evidence path label does not match")
    if artifact["sha256"] != sha256(evidence_path):
        raise ValueError("transfer admission evidence hash does not match")
    evidence = _read_gzip_jsonl(evidence_path)
    if len(evidence) != artifact["record_count"]:
        raise ValueError("transfer admission evidence count does not match")
    outcome = _build_outcome(
        evidence,
        datasets=list(metadata["datasets"]),
        adoption_targets={
            key: value
            for key, value in report["development_protocol"][
                "adoption_targets"
            ].items()
        },
    )
    if evidence != report.get("evidence") or outcome != report.get("outcome"):
        raise ValueError("transfer admission aggregates do not match evidence")
    return report
