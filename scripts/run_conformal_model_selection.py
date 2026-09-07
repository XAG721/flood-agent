from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_model_selection import (
    evaluate_nested_model_selection,
    write_nested_model_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run nested train-only feature/regularization selection for the FRC "
            "evidence-sufficiency head."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/role_scores_conditionalqa.jsonl"
        ),
    )
    parser.add_argument("--reference-run-report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_model_selection"),
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input score file does not exist: {args.input}")
    report = evaluate_nested_model_selection(
        args.input,
        reference_run_report=args.reference_run_report,
    )
    paths = write_nested_model_selection(
        report,
        json_path=args.output_dir / "conformal_model_selection.json",
        markdown_path=args.output_dir / "conformal_model_selection.md",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
