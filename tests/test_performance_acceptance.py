from __future__ import annotations

import pytest

from flood_system.performance_acceptance import evaluate_budget, percentile, summarize_samples


def test_percentile_uses_nearest_rank_and_rejects_invalid_quantile() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.50) == 3
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5
    with pytest.raises(ValueError, match="quantile"):
        percentile([1], 1.1)


def test_performance_summary_and_budget_fail_closed() -> None:
    results = {
        "healthy": summarize_samples([10, 20, 30], [200, 200, 200]),
        "slow": summarize_samples([10, 20, 500], [200, 200, 503]),
    }
    budget = {
        "endpoints": {
            "healthy": {"max_p95_ms": 100, "max_error_rate": 0},
            "slow": {"max_p95_ms": 200, "max_error_rate": 0},
            "missing": {"max_p95_ms": 200, "max_error_rate": 0},
        }
    }

    decision = evaluate_budget(results, budget)

    assert decision["status"] == "FAIL"
    assert next(item for item in decision["checks"] if item["endpoint_id"] == "healthy")["passed"] is True
    assert next(item for item in decision["checks"] if item["endpoint_id"] == "slow")["passed"] is False
    assert next(item for item in decision["checks"] if item["endpoint_id"] == "missing")["reason"] == "missing benchmark result"
