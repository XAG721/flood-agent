from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_robustness import (
    evaluate_conformal_robustness,
    write_conformal_robustness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run pre-registered repeated grouped splits for the FRC evidence-"
            "sufficiency split-conformal audit."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/role_scores_conditionalqa.jsonl"
        ),
        help="Frozen real-model ConditionalQA role-score JSONL.",
    )
    parser.add_argument(
        "--reference-run-report",
        type=Path,
        help="Optional hashed run_report.md declaring frozen scorer configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_robustness"),
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input score file does not exist: {args.input}")
    report = evaluate_conformal_robustness(
        args.input,
        reference_run_report=args.reference_run_report,
    )
    paths = write_conformal_robustness(
        report,
        json_path=args.output_dir / "conformal_robustness.json",
        markdown_path=args.output_dir / "conformal_robustness.md",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
