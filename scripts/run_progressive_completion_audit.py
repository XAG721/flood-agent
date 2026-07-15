from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.completion_audit import write_progressive_completion_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the progressive-upgrade controlled completion contract and preserve external No-Go boundaries."
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("output/acceptance/progressive_completion_audit.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("output/acceptance/progressive_completion_audit.md"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    json_target, markdown_target = write_progressive_completion_audit(
        repo_root,
        (repo_root / args.json_output).resolve(),
        (repo_root / args.markdown_output).resolve(),
    )
    print(json_target)
    print(markdown_target)
    report = json.loads(json_target.read_text(encoding="utf-8"))
    return 0 if report["summary"]["controlled_first_iteration"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
