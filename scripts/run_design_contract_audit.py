from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.design_contract_audit import write_design_contract_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every explicit design-contract clause with fail-closed local, controlled, "
            "gate, or external evidence boundaries."
        )
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("output/acceptance/design_contract_audit.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("output/acceptance/design_contract_audit.md"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    json_target, markdown_target = write_design_contract_audit(
        repo_root,
        (repo_root / args.json_output).resolve(),
        (repo_root / args.markdown_output).resolve(),
    )
    print(json_target)
    print(markdown_target)
    report = json.loads(json_target.read_text(encoding="utf-8"))
    summary = report["summary"]
    return (
        0
        if summary["all_items_accounted"] and summary["controlled_scope_complete"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
