from __future__ import annotations

from pathlib import Path

from flood_system.frc_evaluation import default_conflict_benchmark, evaluate_conflict_cases
from flood_system.rag_evaluation import RAGBaselineEvaluator, RAG_METHODS, evaluate_frc_gate, render_rag_evaluation_markdown


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
        "without_role",
        "without_field",
        "without_budget",
        "without_trust_conflict",
        "without_set_objective",
    }
    assert ablations["variants"]["full_frc_select"]["role_coverage"] >= ablations["variants"]["without_role"]["role_coverage"]
    assert ablations["variants"]["full_frc_select"]["evidence_recall"] >= ablations["variants"]["without_field"]["evidence_recall"]
    assert ablations["variants"]["full_frc_select"]["role_coverage"] >= ablations["variants"]["without_set_objective"]["role_coverage"]
    markdown = render_rag_evaluation_markdown(report, ablations, metadata)
    assert "| bm25 |" in markdown
    assert "| dense_top_k |" in markdown
    assert "| coverage_greedy_proxy |" in markdown
    assert "| frc_select |" in markdown
    assert "不是 SetR 复现" in markdown
    assert "正式论文结论仍需真实神经向量模型" in markdown


def test_gate_two_report_keeps_failed_coverage_threshold_as_no_go() -> None:
    evaluator, cases, _ = RAGBaselineEvaluator.from_benchmark(BENCHMARK)
    report = evaluator.evaluate(cases, top_k=4, token_budget=520)
    ablations = evaluator.evaluate_ablations(cases, top_k=4, token_budget=520)
    conflict_report = evaluate_conflict_cases(default_conflict_benchmark())

    gate = evaluate_frc_gate(report, ablations, conflict_report)

    assert gate["status"] == "NO-GO"
    assert gate["criteria"]["相同预算下字段覆盖率相对最强基线提高至少5个百分点"] is False
    assert gate["criteria"]["Full引用正确率严格优于w/o Role和w/o Field"] is True
    assert gate["strongest_reproducible_baseline_method"] == "coverage_greedy_proxy"
    assert gate["undisclosed_critical_conflicts"] == 0
