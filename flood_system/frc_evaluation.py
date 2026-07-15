from __future__ import annotations

from typing import Any


def detect_conflict_keys(evidence: list[dict[str, Any]]) -> set[str]:
    values: dict[str, set[str]] = {}
    for item in evidence:
        key = str(item.get("conflict_key", "")).strip()
        value = str(item.get("conflict_value", "")).strip()
        if key and value:
            values.setdefault(key, set()).add(value)
    return {key for key, variants in values.items() if len(variants) > 1}


def evaluate_conflict_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    rows = []
    for case in cases:
        expected = set(case["gold_conflict_keys"])
        predicted = detect_conflict_keys(case["evidence"])
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        rows.append(
            {
                "case_id": case["case_id"],
                "expected": sorted(expected),
                "predicted": sorted(predicted),
            }
        )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "case_count": len(cases),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "rows": rows,
        "is_simulated": True,
    }


def default_conflict_benchmark() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "action-conflict",
            "evidence": [
                {"conflict_key": "action", "conflict_value": "close"},
                {"conflict_key": "action", "conflict_value": "keep_open"},
            ],
            "gold_conflict_keys": ["action"],
        },
        {
            "case_id": "deadline-conflict",
            "evidence": [
                {"conflict_key": "deadline", "conflict_value": "30m"},
                {"conflict_key": "deadline", "conflict_value": "60m"},
            ],
            "gold_conflict_keys": ["deadline"],
        },
        {
            "case_id": "duplicate-support",
            "evidence": [
                {"conflict_key": "responsible_party", "conflict_value": "housing"},
                {"conflict_key": "responsible_party", "conflict_value": "housing"},
            ],
            "gold_conflict_keys": [],
        },
        {
            "case_id": "different-fields",
            "evidence": [
                {"conflict_key": "action", "conflict_value": "close"},
                {"conflict_key": "deadline", "conflict_value": "30m"},
            ],
            "gold_conflict_keys": [],
        },
    ]
