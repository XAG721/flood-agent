from __future__ import annotations

from pathlib import Path

from flood_system.rag_evaluation import RAGBaselineEvaluator, RAG_METHODS, render_rag_evaluation_markdown


BENCHMARK = Path("flood_system/rag_benchmarks/district_policy_benchmark.json")


def test_rag_evaluation_runs_all_v3_baselines_and_frc_select() -> None:
    evaluator, cases, metadata = RAGBaselineEvaluator.from_benchmark(BENCHMARK)

    report = evaluator.evaluate(cases, top_k=4, token_budget=520)

    assert report["methods"] == list(RAG_METHODS)
    assert report["case_count"] == 3
    assert set(report["aggregates"]) == set(RAG_METHODS)
    assert report["aggregates"]["frc_select"]["evidence_recall"] == 1.0
    assert report["aggregates"]["frc_select"]["role_coverage"] == 1.0
    assert report["aggregates"]["frc_select"]["unsupported_evidence_ratio"] < report["aggregates"]["hybrid"]["unsupported_evidence_ratio"]
    assert metadata["annotation_scope"].startswith("仓库内人工构造")


def test_frc_select_ablation_and_markdown_report_are_reproducible() -> None:
    evaluator, cases, metadata = RAGBaselineEvaluator.from_benchmark(BENCHMARK)
    report = evaluator.evaluate(cases, top_k=4, token_budget=520)
    ablations = evaluator.evaluate_ablations(cases, top_k=4, token_budget=520)

    assert set(ablations["variants"]) == {
        "full_frc_select",
        "without_budget",
        "without_trust_conflict",
        "without_set_objective",
    }
    assert ablations["variants"]["full_frc_select"]["role_coverage"] >= ablations["variants"]["without_set_objective"]["role_coverage"]
    markdown = render_rag_evaluation_markdown(report, ablations, metadata)
    assert "| bm25 |" in markdown
    assert "| dense_top_k |" in markdown
    assert "| frc_select |" in markdown
    assert "正式论文结论仍需真实神经向量模型" in markdown
