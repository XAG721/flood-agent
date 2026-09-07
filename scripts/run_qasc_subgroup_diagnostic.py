from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.public_evidence import sha256
from research.frc_rag.qasc_subgroup_diagnostic import (
    evaluate_qasc_subgroup_diagnostic,
    write_qasc_subgroup_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen descriptive QASC question-type diagnostic."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/role_scores_qasc.jsonl"
        ),
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_cross_dataset/qasc_confirmation.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conformal_qasc_subgroup_diagnostic_protocol.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_qasc_subgroup_diagnostic"),
    )
    args = parser.parse_args()
    for path in (args.source, args.confirmation, args.protocol):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    import json

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected = protocol["implementation"]
    implementation_path = REPOSITORY_ROOT / expected["module_path"]
    runner_path = REPOSITORY_ROOT / expected["runner_path"]
    if sha256(implementation_path) != expected["module_sha256"]:
        parser.error("QASC diagnostic implementation hash does not match protocol")
    if sha256(runner_path) != expected["runner_sha256"]:
        parser.error("QASC diagnostic runner hash does not match protocol")
    report, records = evaluate_qasc_subgroup_diagnostic(
        args.source, args.confirmation, args.protocol
    )
    paths = write_qasc_subgroup_diagnostic(
        report,
        records,
        json_path=args.output_dir / "conformal_qasc_subgroup_diagnostic.json",
        markdown_path=args.output_dir / "conformal_qasc_subgroup_diagnostic.md",
        cases_path=args.output_dir / "conformal_qasc_subgroup_cases.jsonl.gz",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
