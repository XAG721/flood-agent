from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_transfer_admission import (
    evaluate_transfer_admission_guard,
    write_transfer_admission_guard,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen calibration-only fail-closed admission guard over "
            "the cross-dataset head-transfer evidence."
        )
    )
    parser.add_argument(
        "--transfer-report",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_head_transfer/"
            "conformal_head_transfer.json"
        ),
    )
    parser.add_argument(
        "--transfer-cases",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_head_transfer/"
            "conformal_head_transfer_cases.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_transfer_admission"),
    )
    args = parser.parse_args()
    for path in (args.transfer_report, args.transfer_cases):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")

    report, evidence = evaluate_transfer_admission_guard(
        args.transfer_report,
        args.transfer_cases,
    )
    paths = write_transfer_admission_guard(
        report,
        evidence,
        json_path=args.output_dir / "conformal_transfer_admission.json",
        markdown_path=args.output_dir / "conformal_transfer_admission.md",
        evidence_path=(
            args.output_dir / "conformal_transfer_admission_evidence.jsonl.gz"
        ),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
