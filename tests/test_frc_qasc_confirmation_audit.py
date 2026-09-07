from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.frc_rag.qasc_confirmation_audit import (
    load_qasc_confirmation_audit,
    render_qasc_confirmation_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs"
    / "progressive_upgrade"
    / "conformal_qasc_confirmation_protocol.json"
)
CONFIRMATION_PATH = (
    REPO_ROOT
    / "output"
    / "rag_evaluation"
    / "conformal_cross_dataset"
    / "qasc_confirmation.json"
)
AUDIT_PATH = (
    REPO_ROOT
    / "output"
    / "rag_evaluation"
    / "conformal_qasc_confirmation"
    / "conformal_qasc_confirmation.json"
)


def test_committed_qasc_audit_reaggregates_without_private_cache() -> None:
    report = load_qasc_confirmation_audit(
        AUDIT_PATH, PROTOCOL_PATH, CONFIRMATION_PATH
    )
    assert report["outcome"]["strict_confirmation_status"] == "NOT_CONFIRMED"
    assert report["protocol_deviation"] == {
        "present": True,
        "id": "PARTIAL_STATUS_DEFINITION_MISMATCH",
        "written_protocol_status": "PARTIAL_CONFIRMATION",
        "frozen_evaluator_status": "NOT_CONFIRMED",
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
    }
    assert report["decision"]["review_ranking_run_on_qasc"] is False
    assert report["decision"]["musique_downloaded_or_inspected"] is False
    markdown = render_qasc_confirmation_markdown(report)
    assert "Strict status: `NOT_CONFIRMED`" in markdown
    assert "Frozen-evaluator status: `NOT_CONFIRMED`" in markdown


def test_qasc_audit_rejects_tampered_aggregate(tmp_path: Path) -> None:
    report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    report["outcome"]["strict_confirmation_status"] = "FULL_CONFIRMATION"
    tampered = tmp_path / "qasc-audit.json"
    tampered.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not reaggregate"):
        load_qasc_confirmation_audit(tampered, PROTOCOL_PATH, CONFIRMATION_PATH)
