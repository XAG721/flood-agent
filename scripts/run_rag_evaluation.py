from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.rag_evaluation import RAGBaselineEvaluator, evaluate_frc_gate, render_rag_evaluation_markdown
from flood_system.frc_evaluation import default_conflict_benchmark, evaluate_conflict_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V3 local RAG baseline and FRC-RAG evaluation.")
    parser.add_argument("--benchmark", type=Path, default=Path("flood_system/rag_benchmarks/district_policy_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/rag_evaluation"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=520)
    args = parser.parse_args()

    evaluator, cases, metadata = RAGBaselineEvaluator.from_benchmark(args.benchmark)
    report = evaluator.evaluate(cases, top_k=args.top_k, token_budget=args.token_budget)
    ablations = evaluator.evaluate_ablations(cases, top_k=args.top_k, token_budget=args.token_budget)
    conflict_report = evaluate_conflict_cases(default_conflict_benchmark())
    gate_report = evaluate_frc_gate(report, ablations, conflict_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rag_evaluation_report.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "comparison": report,
                "ablations": ablations,
                "conflict_detection": conflict_report,
                "gate_2": gate_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "rag_evaluation_report.md").write_text(
        render_rag_evaluation_markdown(report, ablations, metadata, conflict_report, gate_report),
        encoding="utf-8",
    )
    print(f"RAG evaluation complete: {len(cases)} cases, {len(report['methods'])} methods")
    print(f"Gate 2: {gate_report['status']} (coverage gain {gate_report['coverage_gain_percentage_points']:.4f} pp)")
    print(args.output_dir / "rag_evaluation_report.md")


if __name__ == "__main__":
    main()
