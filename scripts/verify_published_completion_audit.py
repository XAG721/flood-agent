from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "output/acceptance/progressive_completion_audit.json"
MARKDOWN_PATH = ROOT / "output/acceptance/progressive_completion_audit.md"


def verify_published_completion_audit(
    json_path: Path = JSON_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    expected = {
        "controlled_first_iteration": "PASS",
        "passed_local_requirements": 67,
        "local_requirement_count": 67,
        "frc_gate_2": "NO-GO",
        "production_readiness": "NO-GO_EXTERNAL",
        "overall": "CONTROLLED_SCOPE_COMPLETE_PRODUCTION_NO_GO",
    }
    if summary != expected:
        raise ValueError("published completion audit summary does not match release state")
    metadata = report.get("metadata", {})
    if metadata.get("audit_version") != "progressive-completion-audit-v84":
        raise ValueError("published completion audit version is not v84")
    if metadata.get("scope") != "single-district controlled simulation":
        raise ValueError("published completion audit scope is not controlled simulation")
    markdown = markdown_path.read_text(encoding="utf-8")
    for statement in (
        "CONTROLLED_SCOPE_COMPLETE_PRODUCTION_NO_GO",
        "NO-GO_EXTERNAL",
        "FRC-RAG Gate 2",
    ):
        if statement not in markdown:
            raise ValueError(f"published completion audit markdown omits {statement}")
    return report


def main() -> int:
    report = verify_published_completion_audit()
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
