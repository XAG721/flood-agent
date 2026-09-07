"""Frozen, descriptive QASC question-type diagnostic for conformal sufficiency."""

from __future__ import annotations

import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.frc_rag.conformal_sufficiency import (
    evaluate_conformal_sufficiency,
    prepare_conformal_source,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-qasc-question-type-diagnostic-v1"
STATUS = "RUN_QASC_POST_CONFIRMATION_DESCRIPTIVE_DIAGNOSTIC"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float:
    return round(statistics.fmean(values), 6)


def _repeat_core(report: dict[str, Any], split_version: str) -> dict[str, Any]:
    alphas = tuple(float(value) for value in report["calibration"]["alphas"])
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
            str(alpha): report["calibration"]["alphas"][str(alpha)]["evaluation"]
            for alpha in alphas
        },
    }


def _group_metrics(records: list[dict[str, Any]], minimum: int) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case[str(record["case_id"])].append(record)
    incomplete = {
        case_id
        for case_id, rows in by_case.items()
        if any(not bool(row["complete"]) for row in rows)
    }
    false_complete = {
        case_id
        for case_id, rows in by_case.items()
        if any(
            not bool(row["complete"])
            and bool(row["conformal_declared_complete"])
            for row in rows
        )
    }
    complete_rows = [row for row in records if bool(row["complete"])]
    declarations = sum(bool(row["conformal_declared_complete"]) for row in records)
    complete_declarations = sum(
        bool(row["conformal_declared_complete"]) for row in complete_rows
    )
    return {
        "case_families": len(by_case),
        "variants": len(records),
        "incomplete_case_families": len(incomplete),
        "false_complete_case_families": len(false_complete),
        "false_complete_case_family_rate": round(
            len(false_complete) / max(1, len(incomplete)), 6
        ),
        "abstention_rate": round(1.0 - declarations / max(1, len(records)), 6),
        "complete_variants": len(complete_rows),
        "true_complete_declaration_rate": round(
            complete_declarations / max(1, len(complete_rows)), 6
        ),
        "eligible": len(incomplete) >= minimum,
    }


def analyze_qasc_question_types(
    case_records: list[dict[str, Any]],
    *,
    split_versions: tuple[str, ...],
    alpha: float = 0.1,
    minimum_incomplete_case_families: int = 30,
    minimum_eligible_repeats: int = 7,
) -> dict[str, Any]:
    if not case_records:
        raise ValueError("QASC diagnostic requires case records")
    if not 1 <= minimum_eligible_repeats <= len(split_versions):
        raise ValueError("minimum eligible repeats is outside split count")
    per_repeat = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split_version in split_versions:
        rows = [
            row for row in case_records if row["split_version"] == split_version
        ]
        if not rows:
            raise ValueError(f"missing QASC records for {split_version}")
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_type[str(row["question_type"])].append(row)
        subgroups = []
        for value, group_rows in sorted(by_type.items()):
            metrics = {
                "question_type": value,
                **_group_metrics(group_rows, minimum_incomplete_case_families),
            }
            subgroups.append(metrics)
            grouped[value].append(metrics)
        per_repeat.append(
            {
                "split_version": split_version,
                "evaluation_case_families": len({row["case_id"] for row in rows}),
                "question_types": subgroups,
            }
        )
    aggregates = []
    for value, rows in sorted(grouped.items()):
        eligible = [row for row in rows if row["eligible"]]
        aggregate_eligible = len(eligible) >= minimum_eligible_repeats
        item: dict[str, Any] = {
            "question_type": value,
            "observed_repeats": len(rows),
            "eligible_repeats": len(eligible),
            "eligible_for_conclusion": aggregate_eligible,
        }
        if aggregate_eligible:
            risks = [row["false_complete_case_family_rate"] for row in eligible]
            item.update(
                {
                    "false_complete_case_family_rate_mean": _mean(risks),
                    "false_complete_case_family_rate_min": min(risks),
                    "false_complete_case_family_rate_max": max(risks),
                    "risk_at_or_below_alpha_repeats": sum(
                        risk <= alpha for risk in risks
                    ),
                    "abstention_rate_mean": _mean(
                        row["abstention_rate"] for row in eligible
                    ),
                    "true_complete_declaration_rate_mean": _mean(
                        row["true_complete_declaration_rate"] for row in eligible
                    ),
                }
            )
        aggregates.append(item)
    eligible = [item for item in aggregates if item["eligible_for_conclusion"]]
    stable = bool(eligible) and all(
        item["false_complete_case_family_rate_mean"] <= alpha
        and item["risk_at_or_below_alpha_repeats"] >= minimum_eligible_repeats
        for item in eligible
    )
    if not eligible:
        diagnostic_status = "INSUFFICIENT_QUESTION_TYPE_SUPPORT"
    elif stable:
        diagnostic_status = "DESCRIPTIVE_QUESTION_TYPE_STABLE"
    else:
        diagnostic_status = "DESCRIPTIVE_HETEROGENEITY_DETECTED"
    worst = max(
        eligible,
        key=lambda item: (
            item["false_complete_case_family_rate_mean"],
            item["question_type"],
        ),
        default=None,
    )
    return {
        "per_repeat": per_repeat,
        "question_types": aggregates,
        "outcome": {
            "status": diagnostic_status,
            "eligible_question_type_count": len(eligible),
            "all_eligible_mean_risks_at_or_below_alpha": bool(eligible)
            and all(
                item["false_complete_case_family_rate_mean"] <= alpha
                for item in eligible
            ),
            "all_eligible_repeat_consistency_targets_met": bool(eligible)
            and all(
                item["risk_at_or_below_alpha_repeats"]
                >= minimum_eligible_repeats
                for item in eligible
            ),
            "worst_eligible_question_type": worst,
        },
    }


def evaluate_qasc_subgroup_diagnostic(
    source_path: Path,
    confirmation_path: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _load_json(protocol_path)
    confirmation = _load_json(confirmation_path)
    if protocol.get("schema_version") != "frc-qasc-subgroup-diagnostic-protocol-v1":
        raise ValueError("unsupported QASC subgroup diagnostic protocol")
    if sha256(source_path) != protocol["frozen_inputs"]["score_source_sha256"]:
        raise ValueError("QASC subgroup score source hash does not match protocol")
    if sha256(confirmation_path) != protocol["frozen_inputs"][
        "confirmation_report_sha256"
    ]:
        raise ValueError("QASC confirmation report hash does not match protocol")
    if confirmation["outcome"]["confirmation_status"] != "NOT_CONFIRMED":
        raise ValueError("QASC diagnostic requires the frozen negative confirmation")
    robustness = confirmation["confirmation_robustness"]
    selector = robustness["metadata"]["selector"]
    prepared = prepare_conformal_source(
        source_path,
        k=int(selector["top_k"]),
        token_budget=int(selector["token_budget"]),
        role_threshold=float(selector["role_threshold"]),
    )
    split_versions = tuple(protocol["frozen_analysis"]["split_versions"])
    alpha = float(protocol["frozen_analysis"]["alpha"])
    case_records = []
    generated_repeats = []
    for split_version in split_versions:
        report, records = evaluate_conformal_sufficiency(
            source_path,
            split_version=split_version,
            k=int(selector["top_k"]),
            token_budget=int(selector["token_budget"]),
            role_threshold=float(selector["role_threshold"]),
            prepared_source=prepared,
        )
        generated_repeats.append(_repeat_core(report, split_version))
        case_records.extend(
            {
                "split_version": split_version,
                "case_id": record["case_id"],
                "question_type": record["question_type"],
                "complete": record["complete"],
                "conformal_declared_complete": record[
                    "conformal_declared_complete"
                ],
            }
            for record in records
            if record["split"] == "evaluation"
        )
    if generated_repeats != robustness["repeats"]:
        raise ValueError("QASC frozen confirmation repeats were not reproduced")
    analysis = analyze_qasc_question_types(
        case_records,
        split_versions=split_versions,
        alpha=alpha,
        minimum_incomplete_case_families=int(
            protocol["frozen_analysis"]["minimum_incomplete_case_families"]
        ),
        minimum_eligible_repeats=int(
            protocol["frozen_analysis"]["minimum_eligible_repeats"]
        ),
    )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "dataset": "QASC",
            "split": "validation",
            "dimension": "question_type",
            "alpha": alpha,
            "split_versions": list(split_versions),
            "minimum_incomplete_case_families": protocol["frozen_analysis"][
                "minimum_incomplete_case_families"
            ],
            "minimum_eligible_repeats": protocol["frozen_analysis"][
                "minimum_eligible_repeats"
            ],
            "post_confirmation_descriptive_only": True,
            "case_artifact": {},
        },
        "provenance": {
            "protocol_path_label": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "score_source_path_label": source_path.name,
            "score_source_sha256": sha256(source_path),
            "confirmation_report_path_label": confirmation_path.name,
            "confirmation_report_sha256": sha256(confirmation_path),
            "confirmation_repeats_exactly_reproduced": True,
        },
        "analysis": analysis,
        "decision": {
            "confirmation_status_changed": False,
            "method_or_threshold_selection_allowed": False,
            "review_ranking_run": False,
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
            "claim_boundary": (
                "Post-confirmation question-type results are correlated descriptive "
                "diagnostics, not conditional conformal guarantees."
            ),
        },
    }
    return report, case_records


def render_qasc_subgroup_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    analysis = report["analysis"]
    lines = [
        "# QASC Post-confirmation Question-type Diagnostic",
        "",
        f"- Status: `{analysis['outcome']['status']}`",
        "- Scope: descriptive only; no threshold or method selection",
        f"- Gate 2: `{report['decision']['gate_2']}`",
        "",
        "| Question type | Eligible repeats | Mean risk | Max risk | <= alpha repeats |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in analysis["question_types"]:
        if item["eligible_for_conclusion"]:
            lines.append(
                f"| {item['question_type']} | {item['eligible_repeats']} | "
                f"{item['false_complete_case_family_rate_mean']:.6f} | "
                f"{item['false_complete_case_family_rate_max']:.6f} | "
                f"{item['risk_at_or_below_alpha_repeats']}/{len(metadata['split_versions'])} |"
            )
        else:
            lines.append(
                f"| {item['question_type']} | {item['eligible_repeats']} | "
                "insufficient | insufficient | insufficient |"
            )
    lines.extend(
        [
            "",
            "This diagnostic cannot change the frozen QASC `NOT_CONFIRMED` result, "
            "reopen failed methods, or authorize deployment.",
            "",
        ]
    )
    return "\n".join(lines)


def write_qasc_subgroup_diagnostic(
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
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(payload)
    report["metadata"]["case_artifact"] = {
        "path_label": cases_path.name,
        "sha256": sha256(cases_path),
        "record_count": len(case_records),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_qasc_subgroup_markdown(report), encoding="utf-8")
    return json_path, markdown_path, cases_path


def load_qasc_subgroup_diagnostic(
    json_path: Path, cases_path: Path
) -> dict[str, Any]:
    report = _load_json(json_path)
    if report.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported QASC subgroup diagnostic schema")
    artifact = report["metadata"]["case_artifact"]
    if artifact["path_label"] != cases_path.name or artifact["sha256"] != sha256(
        cases_path
    ):
        raise ValueError("QASC subgroup case artifact does not match")
    with gzip.open(cases_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != artifact["record_count"]:
        raise ValueError("QASC subgroup case record count does not match")
    metadata = report["metadata"]
    rebuilt = analyze_qasc_question_types(
        records,
        split_versions=tuple(metadata["split_versions"]),
        alpha=float(metadata["alpha"]),
        minimum_incomplete_case_families=int(
            metadata["minimum_incomplete_case_families"]
        ),
        minimum_eligible_repeats=int(metadata["minimum_eligible_repeats"]),
    )
    if rebuilt != report["analysis"]:
        raise ValueError("QASC subgroup diagnostic does not reaggregate")
    return report
