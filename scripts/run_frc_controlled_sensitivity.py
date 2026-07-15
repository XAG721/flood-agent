from __future__ import annotations
# ruff: noqa: E402 -- research stays outside the runtime wheel.

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.controlled_sensitivity import (
    evaluate_controlled_sensitivity,
    write_controlled_sensitivity,
)
from flood_system.rag_evaluation import RAGBaselineEvaluator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run controlled-domain FRC role/field-weight and conflict-threshold sensitivity."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("flood_system/rag_benchmarks/controlled_frc_sensitivity_benchmark.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/controlled_domain_sensitivity"),
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=520)
    args = parser.parse_args()

    evaluator, cases, _ = RAGBaselineEvaluator.from_benchmark(args.benchmark)
    report = evaluate_controlled_sensitivity(
        evaluator.documents,
        cases,
        benchmark_path=args.benchmark,
        top_k=args.top_k,
        token_budget=args.token_budget,
    )
    json_path, markdown_path = write_controlled_sensitivity(
        report,
        json_path=args.output_dir / "controlled_domain_sensitivity.json",
        markdown_path=args.output_dir / "controlled_domain_sensitivity.md",
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
