from __future__ import annotations

import math
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 3)


def summarize_samples(durations_ms: list[float], status_codes: list[int]) -> dict[str, Any]:
    if len(durations_ms) != len(status_codes):
        raise ValueError("durations and status codes must have the same length")
    failures = sum(not 200 <= status < 300 for status in status_codes)
    return {
        "sample_count": len(durations_ms),
        "p50_ms": percentile(durations_ms, 0.50),
        "p95_ms": percentile(durations_ms, 0.95),
        "max_ms": round(max(durations_ms), 3) if durations_ms else 0.0,
        "error_count": failures,
        "error_rate": round(failures / len(status_codes), 6) if status_codes else 0.0,
    }


def evaluate_budget(results: dict[str, dict[str, Any]], budget: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    endpoint_budgets = budget.get("endpoints", {})
    for endpoint_id, limits in endpoint_budgets.items():
        if endpoint_id not in results:
            checks.append(
                {
                    "endpoint_id": endpoint_id,
                    "passed": False,
                    "reason": "missing benchmark result",
                }
            )
            continue
        metrics = results[endpoint_id]
        passed = (
            metrics["p95_ms"] <= float(limits["max_p95_ms"])
            and metrics["error_rate"] <= float(limits.get("max_error_rate", 0.0))
        )
        checks.append(
            {
                "endpoint_id": endpoint_id,
                "passed": passed,
                "actual_p95_ms": metrics["p95_ms"],
                "max_p95_ms": float(limits["max_p95_ms"]),
                "actual_error_rate": metrics["error_rate"],
                "max_error_rate": float(limits.get("max_error_rate", 0.0)),
            }
        )
    return {
        "status": "PASS" if checks and all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
    }
