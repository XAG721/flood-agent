"""Integrity audit for the frozen supplemental QASC confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-qasc-supplemental-confirmation-audit-v1"
STATUS = "QASC_SUPPLEMENTAL_EXTERNAL_CONFIRMATION_COMPLETE"
PROTOCOL_SCHEMA_VERSION = "frc-qasc-confirmation-protocol-v1"
PROVENANCE_SCHEMA_VERSION = "frc-qasc-score-source-provenance-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path_label": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _written_protocol_status(confirmation: dict[str, Any]) -> str:
    checks = confirmation["outcome"]["checks"]
    safety = checks["safety_reduction_repeats_meet_pre_registered_minimum"]
    alpha = checks["alpha_control_repeats_meet_pre_registered_minimum"]
    mean = checks["mean_case_family_false_complete_rate_at_or_below_alpha"]
    if safety and alpha and mean:
        return "FULL_CONFIRMATION"
    if safety:
        return "PARTIAL_CONFIRMATION"
    return "NOT_CONFIRMED"


def _build_report(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    confirmation: dict[str, Any],
    *,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    evaluator_status = confirmation["outcome"]["confirmation_status"]
    written_status = _written_protocol_status(confirmation)
    deviation_present = evaluator_status != written_status
    metrics = confirmation["outcome"]["confirmation_metrics"]
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "dataset": "QASC",
            "split": "validation",
            "post_registration_execution": True,
            "independent_of_method_selection": True,
            "protocol_schema_version": protocol["schema_version"],
            "gate_2": "NO-GO/SHADOW",
        },
        "inputs": inputs,
        "score_provenance": manifest,
        "confirmation": confirmation,
        "protocol_deviation": {
            "present": deviation_present,
            "id": "PARTIAL_STATUS_DEFINITION_MISMATCH",
            "written_protocol_status": written_status,
            "frozen_evaluator_status": evaluator_status,
            "description": (
                "The registered prose classified any 10/10 safety-reduction result "
                "with a failed absolute-risk check as partial confirmation. The "
                "pre-existing frozen evaluator requires the mean-risk check to pass "
                "before partial confirmation."
            ),
            "resolution": (
                "Use the stricter frozen evaluator status without changing code, "
                "thresholds, metrics or the registered protocol."
            ),
            "adoption_impact": "NONE_MORE_PERMISSIVE; NOT_CONFIRMED_IS_RETAINED",
        },
        "outcome": {
            "strict_confirmation_status": evaluator_status,
            "checks": confirmation["outcome"]["checks"],
            "baseline_case_family_false_complete_mean": metrics[
                "baseline_case_family_false_complete_mean"
            ],
            "conformal_case_family_false_complete_mean": metrics[
                "conformal_case_family_false_complete_mean"
            ],
            "conformal_case_family_false_complete_range": metrics[
                "conformal_case_family_false_complete_range"
            ],
            "safety_reduction_repeats": metrics["safety_reduction_repeats"],
            "case_family_rate_at_or_below_alpha_repeats": metrics[
                "case_family_rate_at_or_below_alpha_repeats"
            ],
            "conformal_abstention_mean": metrics["conformal_abstention_mean"],
            "conformal_complete_recall_mean": metrics[
                "conformal_complete_recall_mean"
            ],
            "all_pre_registered_full_confirmation_checks_passed": all(
                confirmation["outcome"]["checks"].values()
            ),
        },
        "decision": {
            "next_step": "STOP_QASC_METHOD_CHANGES_AND_RETAIN_NEGATIVE_CONFIRMATION",
            "existing_three_confirmation_series_changed": False,
            "failed_post_hoc_methods_reopened": False,
            "review_ranking_run_on_qasc": False,
            "musique_downloaded_or_inspected": False,
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
            "claim_boundary": (
                "QASC shows strong mean safety reduction in a controlled evidence "
                "pool but does not meet the frozen absolute-risk consistency checks."
            ),
        },
    }


def evaluate_qasc_confirmation_audit(
    protocol_path: Path,
    manifest_path: Path,
    raw_source_path: Path,
    prepared_path: Path,
    score_path: Path,
    confirmation_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    manifest = _load_json(manifest_path)
    confirmation = _load_json(confirmation_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported QASC protocol schema")
    if manifest.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported QASC score provenance schema")
    if manifest["protocol"]["sha256"] != sha256(protocol_path):
        raise ValueError("QASC manifest protocol hash does not match")
    if manifest["source"]["sha256"] != sha256(raw_source_path):
        raise ValueError("QASC raw source hash does not match manifest")
    if manifest["source"]["sha256"] != protocol["frozen_source"]["sha256"]:
        raise ValueError("QASC raw source hash does not match protocol")
    if manifest["preparation"]["sha256"] != sha256(prepared_path):
        raise ValueError("QASC prepared source hash does not match manifest")
    if manifest["scoring"]["sha256"] != sha256(score_path):
        raise ValueError("QASC score source hash does not match manifest")
    if manifest["scoring"]["status"] != "COMPLETE":
        raise ValueError("QASC scoring is not complete")
    if manifest["scoring"]["case_count"] != 926:
        raise ValueError("QASC scored case count is not 926")
    if manifest["preparation"]["case_count"] != 926:
        raise ValueError("QASC prepared case count is not 926")
    if manifest["preparation"]["candidate_count"] != {
        "total": 37040,
        "minimum": 40,
        "maximum": 40,
        "mean": 40.0,
    }:
        raise ValueError("QASC candidate pool does not match frozen shape")
    if manifest["preparation"]["gold_fields_visible_to_scorer"] is not False:
        raise ValueError("QASC preparation is not gold blind")
    if manifest["scoring"]["gold_fields_visible_to_scorer"] is not False:
        raise ValueError("QASC scoring is not gold blind")
    if confirmation["metadata"]["confirmation_dataset"] != "QASC":
        raise ValueError("confirmation report is not QASC")
    if confirmation["metadata"]["confirmation_source_sha256"] != sha256(score_path):
        raise ValueError("QASC confirmation source hash does not match scores")
    inputs = {
        "protocol": _artifact(protocol_path),
        "manifest": _artifact(manifest_path),
        "raw_source": _artifact(raw_source_path),
        "prepared_source": _artifact(prepared_path),
        "score_source": _artifact(score_path),
        "confirmation_report": _artifact(confirmation_path),
    }
    return _build_report(protocol, manifest, confirmation, inputs=inputs)


def render_qasc_confirmation_markdown(report: dict[str, Any]) -> str:
    outcome = report["outcome"]
    deviation = report["protocol_deviation"]
    lines = [
        "# FRC-RAG QASC Supplemental External Confirmation Audit",
        "",
        f"- Strict status: `{outcome['strict_confirmation_status']}`",
        "- Data: QASC validation, 926 controlled candidate pools",
        f"- Gate 2: `{report['decision']['gate_2']}`",
        "- Method, thresholds, and existing confirmation series changed: `False`",
        "",
        "## Frozen checks",
        "",
        "| Metric | Observed | Frozen target | Pass |",
        "|---|---:|---:|---|",
        (
            "| Repeats reducing case-family false completion | "
            f"{outcome['safety_reduction_repeats']}/10 | 10/10 | "
            f"`{outcome['checks']['safety_reduction_repeats_meet_pre_registered_minimum']}` |"
        ),
        (
            "| Repeats with risk at or below alpha | "
            f"{outcome['case_family_rate_at_or_below_alpha_repeats']}/10 | "
            "at least 7/10 | "
            f"`{outcome['checks']['alpha_control_repeats_meet_pre_registered_minimum']}` |"
        ),
        (
            "| Mean case-family false-completion risk | "
            f"{outcome['conformal_case_family_false_complete_mean']:.6f} | "
            "at most 0.10 | "
            f"`{outcome['checks']['mean_case_family_false_complete_rate_at_or_below_alpha']}` |"
        ),
        "",
        "## Effect and scope boundary",
        "",
        (
            "Baseline case-family risk was "
            f"{outcome['baseline_case_family_false_complete_mean']:.6f}; "
            "conformal risk was "
            f"{outcome['conformal_case_family_false_complete_mean']:.6f}; "
            "mean abstention was "
            f"{outcome['conformal_abstention_mean']:.6f}; complete recall was "
            f"{outcome['conformal_complete_recall_mean']:.6f}."
        ),
        "",
        "Each case contains two official facts and 38 frozen BM25 distractors. "
        "This is not an evaluation of QASC's original 17M-sentence open-corpus "
        "retrieval task and is not flood-domain expert validation.",
        "",
        "## Protocol deviation",
        "",
        f"- Present: `{deviation['present']}`",
        f"- Written-protocol status: `{deviation['written_protocol_status']}`",
        f"- Frozen-evaluator status: `{deviation['frozen_evaluator_status']}`",
        f"- Resolution: {deviation['resolution']}",
        "",
        "The stricter `NOT_CONFIRMED` status is retained. No post-hoc "
        "reclassification or tuning is allowed, the failed 64-dimensional review "
        "ranking is not run, and MuSiQue remains neither downloaded nor inspected.",
        "",
    ]
    return "\n".join(lines)


def write_qasc_confirmation_audit(
    report: dict[str, Any], *, json_path: Path, markdown_path: Path
) -> tuple[Path, Path]:
    for path in (json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_qasc_confirmation_markdown(report), encoding="utf-8"
    )
    return json_path, markdown_path


def load_qasc_confirmation_audit(
    json_path: Path, protocol_path: Path, confirmation_path: Path
) -> dict[str, Any]:
    report = _load_json(json_path)
    if report.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported QASC confirmation audit schema")
    if report["inputs"]["protocol"]["path_label"] != protocol_path.name:
        raise ValueError("QASC protocol path label does not match")
    if report["inputs"]["protocol"]["sha256"] != sha256(protocol_path):
        raise ValueError("QASC protocol hash does not match")
    if (
        report["inputs"]["confirmation_report"]["path_label"]
        != confirmation_path.name
    ):
        raise ValueError("QASC confirmation path label does not match")
    if report["inputs"]["confirmation_report"]["sha256"] != sha256(
        confirmation_path
    ):
        raise ValueError("QASC confirmation hash does not match")
    protocol = _load_json(protocol_path)
    confirmation = _load_json(confirmation_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported QASC protocol schema")
    manifest = report["score_provenance"]
    if manifest.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported QASC score provenance schema")
    if manifest["protocol"]["sha256"] != sha256(protocol_path):
        raise ValueError("embedded QASC protocol hash does not match")
    if manifest["source"]["sha256"] != protocol["frozen_source"]["sha256"]:
        raise ValueError("embedded QASC source hash does not match protocol")
    rebuilt = _build_report(protocol, manifest, confirmation, inputs=report["inputs"])
    if rebuilt != report:
        raise ValueError("QASC confirmation audit does not reaggregate")
    return report
