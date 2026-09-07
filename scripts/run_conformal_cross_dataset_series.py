from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_cross_dataset import (
    evaluate_cross_dataset_series,
    write_cross_dataset_series,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate two or more frozen cross-dataset confirmation reports "
            "without changing their per-dataset decision rules."
        )
    )
    parser.add_argument(
        "--reference-robustness",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_robustness/conformal_robustness.json"
        ),
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        action="append",
        required=True,
        help="Confirmation JSON path; repeat for each independent dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_cross_dataset"),
    )
    args = parser.parse_args()

    required = (args.reference_robustness, *args.confirmation)
    for path in required:
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    report = evaluate_cross_dataset_series(
        tuple(args.confirmation),
        reference_robustness_path=args.reference_robustness,
    )
    paths = write_cross_dataset_series(
        report,
        json_path=args.output_dir / "cross_dataset_confirmation_series.json",
        markdown_path=args.output_dir / "cross_dataset_confirmation_series.md",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
