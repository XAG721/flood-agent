from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_cross_dataset import (
    dataset_slug,
    evaluate_cross_dataset_confirmation,
    write_cross_dataset_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm the frozen ConditionalQA sufficiency protocol on an untouched "
            "public role-score dataset without model or threshold selection."
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
        "--confirmation-input",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/role_scores_hotpotqa.jsonl"
        ),
    )
    parser.add_argument("--reference-run-report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_cross_dataset"),
    )
    parser.add_argument(
        "--output-stem",
        help=(
            "Optional output basename without extension. Defaults to a slug derived "
            "from the dataset name in the score file."
        ),
    )
    args = parser.parse_args()

    for path in (args.reference_robustness, args.confirmation_input):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    report = evaluate_cross_dataset_confirmation(
        args.reference_robustness,
        args.confirmation_input,
        reference_run_report=args.reference_run_report,
    )
    stem = args.output_stem or (
        f"{dataset_slug(report['metadata']['confirmation_dataset'])}_confirmation"
    )
    paths = write_cross_dataset_confirmation(
        report,
        json_path=args.output_dir / f"{stem}.json",
        markdown_path=args.output_dir / f"{stem}.md",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
