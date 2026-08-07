from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_sufficiency import (
    evaluate_conformal_sufficiency,
    write_conformal_sufficiency,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a split-conformal evidence-sufficiency abstention head over "
            "frozen ConditionalQA FRC role scores."
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
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_sufficiency"),
    )
    parser.add_argument(
        "--reference-run-report",
        type=Path,
        help=(
            "Optional hashed run_report.md that declares the frozen scorer model "
            "and calibration configuration."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--role-threshold", type=float, default=0.55)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input score file does not exist: {args.input}")
    report, case_records = evaluate_conformal_sufficiency(
        args.input,
        reference_run_report=args.reference_run_report,
        k=args.top_k,
        token_budget=args.token_budget,
        role_threshold=args.role_threshold,
    )
    paths = write_conformal_sufficiency(
        report,
        case_records,
        json_path=args.output_dir / "conformal_sufficiency.json",
        markdown_path=args.output_dir / "conformal_sufficiency.md",
        cases_path=args.output_dir / "conformal_sufficiency_cases.jsonl.gz",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
