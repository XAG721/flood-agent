from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_candidate_cases(
    cases: list[dict[str, Any]], *, top_k: int = 5, bins: int = 10, confidence_key: str = "confidence"
) -> dict[str, Any]:
    if not cases:
        raise ValueError("candidate evaluation requires at least one case")
    recalls: list[float] = []
    critical_total = 0
    critical_missed = 0
    calibration_rows: list[tuple[float, int]] = []
    for case in cases:
        gold = set(case.get("gold_object_ids", []))
        critical = set(case.get("critical_object_ids", []))
        ranked = sorted(
            case.get("candidates", []),
            key=lambda item: (-float(item[confidence_key]), str(item["object_id"])),
        )
        predicted = {str(item["object_id"]) for item in ranked[:top_k]}
        recalls.append(len(predicted & gold) / len(gold) if gold else 1.0)
        critical_total += len(critical)
        critical_missed += len(critical - predicted)
        for item in ranked:
            confidence = min(1.0, max(0.0, float(item[confidence_key])))
            calibration_rows.append((confidence, int(str(item["object_id"]) in gold)))
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [row for row in calibration_rows if lower <= row[0] <= upper if index == bins - 1 or row[0] < upper]
        if not bucket:
            continue
        confidence = sum(row[0] for row in bucket) / len(bucket)
        accuracy = sum(row[1] for row in bucket) / len(bucket)
        ece += len(bucket) / len(calibration_rows) * abs(confidence - accuracy)
    return {
        "case_count": len(cases),
        "top_k": top_k,
        "recall_at_k": round(sum(recalls) / len(recalls), 6),
        "critical_miss_rate": round(critical_missed / critical_total, 6) if critical_total else 0.0,
        "ece": round(ece, 6),
        "calibration_sample_count": len(calibration_rows),
        "confidence_key": confidence_key,
    }


def compare_missing_feature_robustness(
    full_cases: list[dict[str, Any]], degraded_cases: list[dict[str, Any]], *, top_k: int = 5
) -> dict[str, Any]:
    full = evaluate_candidate_cases(full_cases, top_k=top_k)
    degraded = evaluate_candidate_cases(degraded_cases, top_k=top_k)
    return {
        "full": full,
        "degraded": degraded,
        "recall_drop_percentage_points": round((full["recall_at_k"] - degraded["recall_at_k"]) * 100, 4),
    }


def run_candidate_evaluation(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    calibrated = evaluate_candidate_cases(payload["full_cases"])
    raw = evaluate_candidate_cases(payload["full_cases"], confidence_key="raw_confidence")
    report = {
        "metadata": payload.get("metadata", {}),
        "baseline": calibrated,
        "uncalibrated": raw,
        "calibration_improvement": round(raw["ece"] - calibrated["ece"], 6),
        "missing_feature_stress": compare_missing_feature_robustness(
            payload["full_cases"], payload["degraded_cases"]
        ),
        "source": str(input_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
