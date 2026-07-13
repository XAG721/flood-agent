from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flood_system.completion_audit import (
    _canonical_text_sha256,
    _evidence_sha256,
    build_progressive_completion_audit,
    check_artifact_groups,
    check_gate_one,
    write_progressive_completion_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_completion_audit_proves_controlled_scope_without_overstating_production() -> (
    None
):
    report = build_progressive_completion_audit(REPO_ROOT)

    assert report["summary"] == {
        "controlled_first_iteration": "PASS",
        "passed_local_requirements": 12,
        "local_requirement_count": 12,
        "frc_gate_2": "NO-GO",
        "production_readiness": "NO-GO_EXTERNAL",
        "overall": "CONTROLLED_SCOPE_COMPLETE_PRODUCTION_NO_GO",
    }
    assert all(item["status"] == "PASS" for item in report["requirements"])
    assert len(report["external_no_go"]) == 9
    assert all(item["status"] == "NO-GO_EXTERNAL" for item in report["external_no_go"])


def test_artifact_check_reports_the_exact_missing_deliverable(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("evidence", encoding="utf-8")

    result = check_artifact_groups(
        tmp_path,
        "demo",
        "demo artifacts",
        {"group": ("present.txt", "missing.txt")},
    )

    assert result["status"] == "FAIL"
    assert result["details"]["missing"] == {"group": ["missing.txt"]}


def test_gate_one_check_rejects_threshold_regression() -> None:
    report = {
        "baseline": {"recall_at_k": 0.94, "critical_miss_rate": 0.06, "ece": 0.11},
        "calibration_improvement": -0.01,
        "missing_feature_stress": {"recall_drop_percentage_points": 11.0},
    }

    result = check_gate_one(report)

    assert result["status"] == "FAIL"
    assert result["details"]["recall_at_5"] == 0.94


def test_completion_audit_output_is_deterministic(tmp_path: Path) -> None:
    first_json = tmp_path / "first.json"
    first_md = tmp_path / "first.md"
    second_json = tmp_path / "second.json"
    second_md = tmp_path / "second.md"

    write_progressive_completion_audit(REPO_ROOT, first_json, first_md)
    write_progressive_completion_audit(REPO_ROOT, second_json, second_md)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()
    report = json.loads(first_json.read_text(encoding="utf-8"))
    assert report["metadata"]["audit_version"] == "progressive-completion-audit-v4"


def test_evidence_hash_is_independent_of_platform_newlines(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes("第一行\n第二行\n".encode())
    crlf.write_bytes("第一行\r\n第二行\r\n".encode())

    assert _canonical_text_sha256(lf) == _canonical_text_sha256(crlf)


def test_binary_evidence_hash_uses_exact_archive_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "cases.jsonl.gz"
    payload = b"\x1f\x8b\x08\x00binary-evidence"
    archive.write_bytes(payload)

    assert _evidence_sha256(archive) == hashlib.sha256(payload).hexdigest()
