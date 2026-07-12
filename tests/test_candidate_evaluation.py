from __future__ import annotations

from flood_system.candidate_evaluation import compare_missing_feature_robustness, evaluate_candidate_cases
from flood_system.simulation_dataset import build_candidate_evaluation_payload, build_floodagent_bench


def test_candidate_metrics_cover_recall_critical_miss_and_calibration():
    cases = [
        {
            "gold_object_ids": ["A", "B"],
            "critical_object_ids": ["A"],
            "candidates": [
                {"object_id": "A", "confidence": 0.9},
                {"object_id": "C", "confidence": 0.6},
                {"object_id": "B", "confidence": 0.5},
            ],
        }
    ]
    report = evaluate_candidate_cases(cases, top_k=2)
    assert report["recall_at_k"] == 0.5
    assert report["critical_miss_rate"] == 0.0
    assert 0 <= report["ece"] <= 1


def test_missing_feature_report_preserves_separate_full_and_degraded_results():
    full = [{"gold_object_ids": ["A"], "critical_object_ids": ["A"], "candidates": [{"object_id": "A", "confidence": 0.9}]}]
    degraded = [{"gold_object_ids": ["A"], "critical_object_ids": ["A"], "candidates": [{"object_id": "B", "confidence": 0.8}]}]
    report = compare_missing_feature_robustness(full, degraded, top_k=1)
    assert report["full"]["recall_at_k"] == 1.0
    assert report["degraded"]["recall_at_k"] == 0.0
    assert report["recall_drop_percentage_points"] == 100.0


def test_simulated_gate_one_fixture_meets_declared_candidate_thresholds():
    payload = build_candidate_evaluation_payload(build_floodagent_bench(seed=42))
    calibrated = evaluate_candidate_cases(payload["full_cases"])
    raw = evaluate_candidate_cases(payload["full_cases"], confidence_key="raw_confidence")
    degraded = compare_missing_feature_robustness(payload["full_cases"], payload["degraded_cases"])
    assert calibrated["recall_at_k"] >= 0.95
    assert calibrated["critical_miss_rate"] <= 0.05
    assert calibrated["ece"] <= 0.10
    assert calibrated["ece"] < raw["ece"]
    assert degraded["recall_drop_percentage_points"] <= 10
