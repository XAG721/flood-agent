from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.qasc_confirmation_audit import (
    evaluate_qasc_confirmation_audit,
    write_qasc_confirmation_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the frozen supplemental QASC confirmation and provenance."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conformal_qasc_confirmation_protocol.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/qasc_provenance.json"
        ),
    )
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=Path(".cache/benchmarks/qasc/validation.parquet"),
    )
    parser.add_argument(
        "--prepared-source",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/candidate_pool_qasc.jsonl"
        ),
    )
    parser.add_argument(
        "--score-source",
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
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_qasc_confirmation"),
    )
    args = parser.parse_args()
    for path in (
        args.protocol,
        args.manifest,
        args.raw_source,
        args.prepared_source,
        args.score_source,
        args.confirmation,
    ):
        if not path.is_file():
            parser.error(f"required QASC artifact does not exist: {path}")
    report = evaluate_qasc_confirmation_audit(
        args.protocol,
        args.manifest,
        args.raw_source,
        args.prepared_source,
        args.score_source,
        args.confirmation,
    )
    paths = write_qasc_confirmation_audit(
        report,
        json_path=args.output_dir / "conformal_qasc_confirmation.json",
        markdown_path=args.output_dir / "conformal_qasc_confirmation.md",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
