from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from flood_system.frc_conflicts_evaluation import (
    CONFLICT_LABELS,
    aggregate_diagnostics,
    classification_metrics,
    selection_diagnostics,
    selection_signature,
    sha256,
)
from flood_system.frc_public_evidence import paired_bootstrap, select_precomputed


FULL_METHOD = "frc_full"
WITHOUT_CONFLICT_METHOD = "wo_conflict"
CONFLICT_DISCLOSURE_ROLES = ("alternative_claim", "temporal_validity")
CORE_ROLES = ("answer_claim", "source_attribution")
DEFAULT_CONFLICT_THRESHOLDS = (0.25, 0.4, 0.55, 0.7, 0.85)
FROZEN_ROLE_THRESHOLD = 0.55


def _threshold_method(value: float) -> str:
    return f"conflict_threshold_{value:.2f}"


def _validated_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("at least one conflict threshold is required")
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("conflict thresholds must be finite and within (0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("conflict thresholds must be unique")
    if FROZEN_ROLE_THRESHOLD not in thresholds:
        raise ValueError("conflict thresholds must include the frozen 0.55 value")
    return thresholds


def _selection_row(
    case: dict[str, Any],
    method: str,
    selected: list[dict[str, Any]],
    *,
    conflict_threshold: float | None,
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "source": case.get("source", ""),
        "question": case["question"],
        "conflict_type": case["conflict_type"],
        "correct_answer": case.get("correct_answer", ""),
        "method": method,
        "conflict_threshold": conflict_threshold,
        "candidate_count": len(case.get("candidates", [])),
        "selected_ids": [item["id"] for item in selected],
        "selected_evidence": selected,
    }


def select_conflict_ablation_evidence(
    cases: Iterable[dict[str, Any]],
    *,
    top_k: int = 5,
    token_budget: int = 1500,
    conflict_thresholds: Iterable[float] = DEFAULT_CONFLICT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Build full, w/o Conflict and conflict-threshold selections.

    ``w/o Conflict`` removes only the two conflict-disclosure roles. The
    sensitivity scan changes only their coverage threshold; answer and source
    roles remain fixed at 0.55. All variants reuse the same candidates and
    saved real-model scores.
    """

    if top_k <= 0 or token_budget <= 0:
        raise ValueError("top_k and token_budget must be positive")
    thresholds = _validated_thresholds(conflict_thresholds)
    rows: list[dict[str, Any]] = []
    for case in cases:
        full = select_precomputed(
            case,
            "frc_select",
            k=top_k,
            budget=token_budget,
            threshold=FROZEN_ROLE_THRESHOLD,
        )
        rows.append(
            _selection_row(
                case,
                FULL_METHOD,
                full,
                conflict_threshold=FROZEN_ROLE_THRESHOLD,
            )
        )

        without_conflict_case = {**case, "required_roles": list(CORE_ROLES)}
        without_conflict = select_precomputed(
            without_conflict_case,
            "frc_select",
            k=top_k,
            budget=token_budget,
            threshold=FROZEN_ROLE_THRESHOLD,
        )
        rows.append(
            _selection_row(
                case,
                WITHOUT_CONFLICT_METHOD,
                without_conflict,
                conflict_threshold=None,
            )
        )

        for threshold in thresholds:
            selected = select_precomputed(
                case,
                "frc_select",
                k=top_k,
                budget=token_budget,
                threshold=FROZEN_ROLE_THRESHOLD,
                role_thresholds={role: threshold for role in CONFLICT_DISCLOSURE_ROLES},
            )
            rows.append(
                _selection_row(
                    case,
                    _threshold_method(threshold),
                    selected,
                    conflict_threshold=threshold,
                )
            )
    return rows


def seed_equivalent_predictions(
    selected_rows: Iterable[dict[str, Any]],
    *,
    reference_selected_rows: Iterable[dict[str, Any]],
    reference_predictions: Iterable[dict[str, Any]],
    cache_key: str,
) -> list[dict[str, Any]]:
    """Reuse a prediction only when case and ordered evidence IDs are identical."""

    prediction_by_method = {
        (row["case_id"], row["method"]): row for row in reference_predictions
    }
    prediction_by_signature: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for reference in reference_selected_rows:
        prediction = prediction_by_method.get((reference["case_id"], reference["method"]))
        if prediction is None:
            continue
        signature = selection_signature(reference)
        previous = prediction_by_signature.get(signature)
        if previous is not None and (
            previous.get("raw_prediction") != prediction.get("raw_prediction")
            or previous.get("predicted_label") != prediction.get("predicted_label")
        ):
            raise ValueError(f"inconsistent reference predictions for exact selection: {signature[0]}")
        prediction_by_signature.setdefault(signature, prediction)

    seeds = []
    for row in selected_rows:
        prediction = prediction_by_signature.get(selection_signature(row))
        if prediction is None:
            continue
        seeds.append(
            {
                "cache_key": cache_key,
                "case_id": row["case_id"],
                "method": row["method"],
                "gold_label": row["conflict_type"],
                "raw_prediction": prediction.get("raw_prediction", ""),
                "predicted_label": prediction.get("predicted_label"),
                "reused_from_exact_reference_selection": True,
            }
        )
    return seeds


def _role_coverage_proxy(selected: list[dict[str, Any]], role: str) -> float:
    best = max(
        (float(item.get("role_scores", {}).get(role, 0.0)) for item in selected),
        default=0.0,
    )
    return float(best >= FROZEN_ROLE_THRESHOLD)


def _case_results(
    cases: dict[str, dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prediction_map = {(row["case_id"], row["method"]): row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise ValueError("duplicate conflict-ablation predictions")
    results = []
    for selected in selected_rows:
        key = (selected["case_id"], selected["method"])
        prediction = prediction_map.get(key)
        if prediction is None:
            raise ValueError(f"missing conflict-ablation prediction: {key}")
        case = cases[selected["case_id"]]
        metrics = selection_diagnostics(selected, case)
        alternative = _role_coverage_proxy(selected["selected_evidence"], "alternative_claim")
        temporal = _role_coverage_proxy(selected["selected_evidence"], "temporal_validity")
        predicted_label = prediction.get("predicted_label")
        results.append(
            {
                "case_id": selected["case_id"],
                "conflict_type": selected["conflict_type"],
                "method": selected["method"],
                "conflict_threshold": selected.get("conflict_threshold"),
                "answer_annotated": bool(selected.get("correct_answer", "").strip()),
                "selected_ids": selected["selected_ids"],
                "predicted_label": predicted_label,
                "metrics": {
                    **metrics,
                    "prediction_parsed": float(predicted_label in CONFLICT_LABELS),
                    "classification_correct": float(predicted_label == selected["conflict_type"]),
                    "alternative_claim_coverage_proxy": alternative,
                    "temporal_validity_coverage_proxy": temporal,
                    "conflict_role_coverage_proxy": (alternative + temporal) / 2.0,
                },
            }
        )
    return results


def _aggregate_proxy_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "prediction_parsed",
        "classification_correct",
        "alternative_claim_coverage_proxy",
        "temporal_validity_coverage_proxy",
        "conflict_role_coverage_proxy",
    )
    return {
        name: round(sum(float(row["metrics"][name]) for row in rows) / max(1, len(rows)), 6)
        for name in names
    }


def _paired_metric(
    full: dict[str, dict[str, Any]],
    ablated: dict[str, dict[str, Any]],
    metric: str,
    *,
    include: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, float | int]:
    case_ids = sorted(set(full) & set(ablated))
    if include is not None:
        case_ids = [case_id for case_id in case_ids if include(full[case_id])]
    differences = [
        float(full[case_id]["metrics"][metric]) - float(ablated[case_id]["metrics"][metric])
        for case_id in case_ids
    ]
    return paired_bootstrap(differences, resamples=2000)


def build_conflict_ablation_report(
    *,
    cases: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
    conflict_thresholds: Iterable[float],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    thresholds = _validated_thresholds(conflict_thresholds)
    cases_by_id = {case["id"]: case for case in cases}
    if len(cases_by_id) != len(cases):
        raise ValueError("duplicate CONFLICTS case IDs")
    expected_methods = {
        FULL_METHOD,
        WITHOUT_CONFLICT_METHOD,
        *(_threshold_method(value) for value in thresholds),
    }
    grouped_selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped_selected[row["method"]].append(row)
    if set(grouped_selected) != expected_methods:
        raise ValueError("conflict-ablation method coverage mismatch")
    if any(len(rows) != len(cases) for rows in grouped_selected.values()):
        raise ValueError("every conflict-ablation method must cover every case")

    case_results = _case_results(cases_by_id, selected_rows, predictions)
    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_results:
        grouped_results[row["method"]].append(row)
    for row in predictions:
        grouped_predictions[row["method"]].append(row)
    diagnostics = aggregate_diagnostics(selected_rows, cases_by_id)

    aggregates = []
    for method in sorted(expected_methods):
        result_rows = grouped_results[method]
        aggregates.append(
            {
                "method": method,
                "cases": len(result_rows),
                "classification": classification_metrics(grouped_predictions[method]),
                "selection": diagnostics[method],
                "coverage_proxies": _aggregate_proxy_metrics(result_rows),
            }
        )
    aggregate_by_method = {row["method"]: row for row in aggregates}

    full_by_case = {row["case_id"]: row for row in grouped_results[FULL_METHOD]}
    ablated_by_case = {
        row["case_id"]: row for row in grouped_results[WITHOUT_CONFLICT_METHOD]
    }
    changed_cases = sum(
        full_by_case[case_id]["selected_ids"] != ablated_by_case[case_id]["selected_ids"]
        for case_id in full_by_case
    )
    comparisons = {
        "selection_changed_cases": changed_cases,
        "selection_unchanged_cases": len(cases) - changed_cases,
        "full_minus_wo_conflict": {
            "classification_accuracy": _paired_metric(
                full_by_case, ablated_by_case, "classification_correct"
            ),
            "conflict_role_coverage_proxy": _paired_metric(
                full_by_case,
                ablated_by_case,
                "conflict_role_coverage_proxy",
                include=lambda row: row["conflict_type"] != "No conflict",
            ),
            "exact_answer_support": _paired_metric(
                full_by_case,
                ablated_by_case,
                "exact_answer_support",
                include=lambda row: bool(cases_by_id[row["case_id"]].get("correct_answer")),
            ),
            "newest_date_retention": _paired_metric(
                full_by_case,
                ablated_by_case,
                "newest_date_retention",
                include=lambda row: row["conflict_type"] == "Conflict due to outdated information",
            ),
            "temporal_endpoint_coverage": _paired_metric(
                full_by_case,
                ablated_by_case,
                "temporal_endpoint_coverage",
                include=lambda row: row["conflict_type"] == "Conflict due to outdated information",
            ),
        },
    }
    sensitivity = [
        {
            "conflict_threshold": threshold,
            **aggregate_by_method[_threshold_method(threshold)],
        }
        for threshold in thresholds
    ]
    classification_ci = comparisons["full_minus_wo_conflict"]["classification_accuracy"]
    return {
        "schema_version": "frc-conflicts-real-model-ablation-v1",
        "metadata": {
            "dataset": "Google CONFLICTS",
            "data_origin": "PUBLIC",
            "is_simulated": False,
            "real_model_scores": True,
            "generator": "local_qwen",
            "case_count": len(cases),
            "top_k": int(config["top_k"]),
            "token_budget": int(config["token_budget"]),
            "frozen_role_threshold": FROZEN_ROLE_THRESHOLD,
            "core_roles": list(CORE_ROLES),
            "conflict_disclosure_roles": list(CONFLICT_DISCLOSURE_ROLES),
            "conflict_thresholds": list(thresholds),
            "protocol": "post_hoc_one_factor_at_a_time_same_saved_real_model_candidate_pool",
            "config": config,
            "source_sha256": {name: sha256(path) for name, path in sorted(source_paths.items())},
        },
        "ablation": {
            "status": "RUN_REAL_MODEL_CONFLICTS",
            "full": aggregate_by_method[FULL_METHOD],
            "without_conflict": aggregate_by_method[WITHOUT_CONFLICT_METHOD],
            "comparison": comparisons,
        },
        "conflict_threshold_sensitivity": {
            "status": "RUN_REAL_MODEL_CONFLICTS",
            "results": sensitivity,
        },
        "aggregates": aggregates,
        "case_results": case_results,
        "decision": {
            "full_superiority_over_wo_conflict_proven": bool(
                float(classification_ci["ci_low"]) > 0.0
            ),
            "gate_2": "NO-GO",
            "interpretation": (
                "This public real-model ablation measures the conflict-disclosure component on "
                "CONFLICTS. Gate 2 remains NO-GO unless all predeclared comparisons and independent "
                "expected-behavior judging pass."
            ),
        },
        "limitations": [
            "The experiment was added after earlier aggregate results were inspected and is post hoc, not preregistered confirmatory evidence.",
            "CONFLICTS supplies case-level conflict types but not candidate-level gold conflict spans; conflict-role coverage is therefore a reranker-score proxy.",
            "Exact Qwen predictions are reused only when the case and ordered selected evidence IDs are identical; changed selections are regenerated locally.",
            "Conflict-type classification is not the CONFLICTS paper's independent expected-behavior adherence judgment.",
            "The experiment does not supply task-field or applicability annotations and cannot satisfy w/o Field or w/o Applicability.",
        ],
    }


def render_conflict_ablation_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    ablation = report["ablation"]
    comparison = ablation["comparison"]
    lines = [
        "# CONFLICTS real-model FRC conflict ablation",
        "",
        f"- Cases: {metadata['case_count']}",
        f"- Data origin: {metadata['data_origin']}",
        f"- Real-model scores: {metadata['real_model_scores']}",
        f"- Generator: {metadata['generator']}",
        f"- Protocol: `{metadata['protocol']}`",
        f"- Gate 2: **{report['decision']['gate_2']}**",
        "",
        "`w/o Conflict` removes only `alternative_claim` and `temporal_validity`; all other "
        "candidate, model, K and token-budget conditions remain fixed.",
        "",
        "## Ablation",
        "",
        "| Variant | Accuracy | Macro F1 | Conflict-role coverage proxy | Exact answer support | Newest-date retention | Changed selections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("full", "without_conflict"):
        row = ablation[key]
        lines.append(
            f"| {row['method']} | {row['classification']['accuracy']:.6f} | "
            f"{row['classification']['macro_f1']:.6f} | "
            f"{row['coverage_proxies']['conflict_role_coverage_proxy']:.6f} | "
            f"{row['selection']['exact_answer_support']:.6f} | "
            f"{row['selection']['newest_date_retention']:.6f} | "
            f"{comparison['selection_changed_cases'] if key == 'without_conflict' else 0} |"
        )
    paired = comparison["full_minus_wo_conflict"]["classification_accuracy"]
    lines.extend(
        [
            "",
            "Full minus `w/o Conflict` classification accuracy: "
            f"{paired['mean_difference']:+.6f}, paired 95% CI "
            f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}].",
            "",
            "## Conflict-threshold sensitivity",
            "",
            "Only the two conflict-disclosure role thresholds change; core-role thresholds remain 0.55.",
            "",
            "| Threshold | Accuracy | Macro F1 | Conflict-role coverage proxy | Exact answer support | Newest-date retention |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["conflict_threshold_sensitivity"]["results"]:
        lines.append(
            f"| {row['conflict_threshold']:.2f} | {row['classification']['accuracy']:.6f} | "
            f"{row['classification']['macro_f1']:.6f} | "
            f"{row['coverage_proxies']['conflict_role_coverage_proxy']:.6f} | "
            f"{row['selection']['exact_answer_support']:.6f} | "
            f"{row['selection']['newest_date_retention']:.6f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", report["decision"]["interpretation"], ""])
    return "\n".join(lines)
