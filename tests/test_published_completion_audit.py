from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_published_completion_audit import verify_published_completion_audit


ROOT = Path(__file__).resolve().parents[1]


def test_published_completion_audit_is_portable_without_local_cache() -> None:
    report = verify_published_completion_audit()
    assert report["summary"]["frc_gate_2"] == "NO-GO"
    assert report["summary"]["production_readiness"] == "NO-GO_EXTERNAL"


def test_published_completion_audit_rejects_more_permissive_status(
    tmp_path: Path,
) -> None:
    report = json.loads(
        (ROOT / "output/acceptance/progressive_completion_audit.json").read_text(
            encoding="utf-8"
        )
    )
    report["summary"]["frc_gate_2"] = "GO"
    json_path = tmp_path / "audit.json"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match release state"):
        verify_published_completion_audit(
            json_path,
            ROOT / "output/acceptance/progressive_completion_audit.md",
        )
