from research.frc_rag.evaluation import default_conflict_benchmark, evaluate_conflict_cases


def test_conflict_detection_benchmark_reports_gate_two_metrics():
    report = evaluate_conflict_cases(default_conflict_benchmark())
    assert report["case_count"] == 4
    assert report["f1"] >= 0.85
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["is_simulated"] is True
