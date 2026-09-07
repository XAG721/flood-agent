"""Cross-fitted selector-routing discovery audit for Google CONFLICTS.

The router is deliberately a retrospective discovery experiment.  It uses only
runtime-observable retrieval diagnostics and already generated classifier labels
as features.  Gold conflict labels are isolated to the training target and to
reporting.  A grouped hash fold prevents rows from the same case from appearing
in both train and evaluation data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.conflicts_evaluation import (
    CONFLICT_LABELS,
    CONFLICT_METHODS,
    domain,
    lexical_diversity,
)
from research.frc_rag.public_evidence import paired_bootstrap, sha256


SCHEMA_VERSION = "frc-conflicts-selector-routing-discovery-v1"
STATUS = "RUN_PUBLIC_RETROSPECTIVE_CROSSFIT_DISCOVERY"
FOLD_NAMESPACE = "frc-router-v1"
FOLD_COUNT = 5
LOGISTIC_L2 = 1.0
LOGISTIC_ITERATIONS = 100
BOOTSTRAP_RESAMPLES = 10_000
STATIC_BASELINE = "coverage_greedy_proxy"
FRC_METHOD = "frc_select"
ALL_METHODS = tuple(CONFLICT_METHODS)
WITHOUT_FRC_METHODS = tuple(
    method for method in CONFLICT_METHODS if method != FRC_METHOD
)
SCORE_FIELDS = ("bm25", "dense", "hybrid", "cross_encoder")
ROLE_FIELDS = (
    "answer_claim",
    "source_attribution",
    "alternative_claim",
    "temporal_validity",
)
ROUTER_VARIANTS = {
    "all_methods": ALL_METHODS,
    "without_frc": WITHOUT_FRC_METHODS,
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _fold_for_case(case_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_NAMESPACE}|{case_id}".encode()).hexdigest()
    return int(digest[:8], 16) % FOLD_COUNT


def _feature_names(methods: tuple[str, ...]) -> list[str]:
    names = [f"method={method}" for method in methods]
    names.extend(f"predicted_label={label}" for label in CONFLICT_LABELS)
    names.extend(
        [
            "own_prediction_vote_share",
            "distinct_prediction_share",
            "selected_count",
            "selected_token_cost",
            "selected_domain_coverage",
            "selected_lexical_diversity",
        ]
    )
    for field in SCORE_FIELDS:
        names.extend(
            [
                f"{field}_mean",
                f"{field}_std",
                f"{field}_max",
                f"{field}_top_margin",
            ]
        )
    for role in ROLE_FIELDS:
        names.extend([f"{role}_mean", f"{role}_max"])
    names.extend(f"selection_jaccard={method}" for method in methods)
    return names


def _index_rows(
    rows: Iterable[dict[str, Any]],
    *,
    kind: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("case_id", "")), str(row.get("method", ""))
        if not all(key):
            raise ValueError(f"{kind} row is missing case_id or method")
        if key in indexed:
            raise ValueError(f"duplicate {kind} row: {key}")
        indexed[key] = row
    return indexed


def _validate_sources(
    scored_cases: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    case_ids = [str(row.get("id", "")) for row in scored_cases]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError("scored cases require non-empty IDs")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate scored case ID")
    case_ids = sorted(case_ids)
    selected = _index_rows(selected_rows, kind="selection")
    predicted = _index_rows(predictions, kind="prediction")
    expected = {(case_id, method) for case_id in case_ids for method in ALL_METHODS}
    if set(selected) != expected:
        raise ValueError("selection rows do not cover every case and frozen method")
    if set(predicted) != expected:
        raise ValueError("prediction rows do not cover every case and frozen method")
    for case_id in case_ids:
        gold_labels = {
            str(predicted[(case_id, method)].get("gold_label", ""))
            for method in ALL_METHODS
        }
        if len(gold_labels) != 1 or next(iter(gold_labels)) not in CONFLICT_LABELS:
            raise ValueError(f"inconsistent or unsupported gold label: {case_id}")
        for method in ALL_METHODS:
            selection = selected[(case_id, method)]
            identifiers = [str(value) for value in selection.get("selected_ids", [])]
            evidence = list(selection.get("selected_evidence", []))
            evidence_ids = [str(item.get("id", "")) for item in evidence]
            if not identifiers or identifiers != evidence_ids:
                raise ValueError(f"selected evidence mismatch: {case_id}/{method}")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate selected evidence: {case_id}/{method}")
            label = predicted[(case_id, method)].get("predicted_label")
            if label is not None and str(label) not in CONFLICT_LABELS:
                raise ValueError(f"unsupported predicted label: {case_id}/{method}")
    return case_ids, selected, predicted


def _feature_vector(
    case_id: str,
    method: str,
    methods: tuple[str, ...],
    selected: dict[tuple[str, str], dict[str, Any]],
    predicted: dict[tuple[str, str], dict[str, Any]],
) -> list[float]:
    row = selected[(case_id, method)]
    evidence = list(row["selected_evidence"])
    prediction = predicted[(case_id, method)].get("predicted_label")
    votes = Counter(predicted[(case_id, item)].get("predicted_label") for item in methods)
    identifiers = set(str(value) for value in row["selected_ids"])
    values: list[float] = [float(method == item) for item in methods]
    values.extend(float(prediction == label) for label in CONFLICT_LABELS)
    values.extend(
        [
            votes[prediction] / len(methods),
            len(votes) / len(methods),
            float(len(evidence)),
            sum(_safe_float(item.get("token_count")) for item in evidence),
            len(
                {
                    domain(str(item.get("url", "")))
                    for item in evidence
                    if domain(str(item.get("url", "")))
                }
            )
            / max(1, len(evidence)),
            lexical_diversity(evidence),
        ]
    )
    for field in SCORE_FIELDS:
        scores = np.asarray(
            [_safe_float(item.get("scores", {}).get(field)) for item in evidence],
            dtype=np.float64,
        )
        ordered = np.sort(scores)
        values.extend(
            [
                float(scores.mean()),
                float(scores.std()),
                float(scores.max()),
                float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0,
            ]
        )
    for role in ROLE_FIELDS:
        scores = np.asarray(
            [
                _safe_float(item.get("role_scores", {}).get(role))
                for item in evidence
            ],
            dtype=np.float64,
        )
        values.extend([float(scores.mean()), float(scores.max())])
    for other_method in methods:
        other = set(
            str(value) for value in selected[(case_id, other_method)]["selected_ids"]
        )
        values.append(len(identifiers & other) / max(1, len(identifiers | other)))
    expected = len(_feature_names(methods))
    if len(values) != expected or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid router feature vector: {case_id}/{method}")
    return values


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = LOGISTIC_L2,
    iterations: int = LOGISTIC_ITERATIONS,
) -> dict[str, Any]:
    if features.ndim != 2 or len(features) != len(labels) or not len(features):
        raise ValueError("router training matrix is invalid")
    if set(np.asarray(labels, dtype=int)) != {0, 1}:
        raise ValueError("router training fold requires both target classes")
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    design = np.column_stack([np.ones(len(features)), (features - means) / scales])
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    converged = False
    completed_iterations = 0
    for iteration in range(iterations):
        logits = np.clip(design @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) + penalty @ coefficients
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        hessian += np.eye(design.shape[1], dtype=np.float64) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        completed_iterations = iteration + 1
        if float(np.max(np.abs(step))) < 1e-9:
            converged = True
            break
    return {
        "means": means,
        "scales": scales,
        "intercept": float(coefficients[0]),
        "weights": coefficients[1:],
        "iterations": completed_iterations,
        "converged": converged,
    }


def _predict_logistic(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["means"]) / model["scales"]
    logits = np.clip(
        model["intercept"] + standardized @ model["weights"],
        -35.0,
        35.0,
    )
    return 1.0 / (1.0 + np.exp(-logits))


def _crossfit_variant(
    evidence: list[dict[str, Any]],
    *,
    variant: str,
    methods: tuple[str, ...],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    choices: dict[str, str] = {}
    model_audit: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        train_units = [
            (row, method)
            for row in evidence
            if int(row["fold"]) != fold
            for method in methods
        ]
        evaluation_rows = [row for row in evidence if int(row["fold"]) == fold]
        if not train_units or not evaluation_rows:
            raise ValueError(f"empty router train or evaluation fold: {fold}")
        features = np.asarray(
            [row["features"][variant][method] for row, method in train_units],
            dtype=np.float64,
        )
        labels = np.asarray(
            [float(row["correctness"][method]) for row, method in train_units],
            dtype=np.float64,
        )
        model = _fit_logistic(features, labels)
        if not model["converged"]:
            raise ValueError(f"router model did not converge: {variant}/{fold}")
        model_payload = {
            "intercept": round(float(model["intercept"]), 12),
            "weights": [round(float(value), 12) for value in model["weights"]],
            "means": [round(float(value), 12) for value in model["means"]],
            "scales": [round(float(value), 12) for value in model["scales"]],
        }
        model_audit.append(
            {
                "fold": fold,
                "training_cases": len({row["case_id"] for row, _ in train_units}),
                "training_rows": len(train_units),
                "positive_rows": int(labels.sum()),
                "evaluation_cases": len(evaluation_rows),
                "feature_count": features.shape[1],
                "iterations": model["iterations"],
                "model_sha256": _canonical_sha256(model_payload),
            }
        )
        for row in evaluation_rows:
            matrix = np.asarray(
                [row["features"][variant][method] for method in methods],
                dtype=np.float64,
            )
            probabilities = _predict_logistic(model, matrix)
            choices[row["case_id"]] = methods[int(np.argmax(probabilities))]
    if len(choices) != len(evidence):
        raise ValueError(f"router crossfit coverage mismatch: {variant}")
    return choices, model_audit


def _classification_metrics(
    evidence: list[dict[str, Any]],
    predictions: dict[str, str | None],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, float | int]] = {}
    parsed = sum(predictions[row["case_id"]] in CONFLICT_LABELS for row in evidence)
    correct = sum(
        predictions[row["case_id"]] == row["gold_label"] for row in evidence
    )
    for label in CONFLICT_LABELS:
        true_positive = sum(
            predictions[row["case_id"]] == label and row["gold_label"] == label
            for row in evidence
        )
        predicted_positive = sum(
            predictions[row["case_id"]] == label for row in evidence
        )
        support = sum(row["gold_label"] == label for row in evidence)
        precision = true_positive / max(1, predicted_positive)
        recall = true_positive / max(1, support)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(
                2.0 * precision * recall / max(1e-12, precision + recall),
                6,
            ),
            "support": support,
        }
    return {
        "cases": len(evidence),
        "parsed_predictions": parsed,
        "parse_rate": round(parsed / len(evidence), 6),
        "correct": correct,
        "accuracy": round(correct / len(evidence), 6),
        "macro_f1": round(
            sum(float(item["f1"]) for item in per_label.values())
            / len(per_label),
            6,
        ),
        "per_label": per_label,
    }


def _majority_prediction(row: dict[str, Any]) -> str | None:
    votes = Counter(row["predictions"][method] for method in ALL_METHODS)
    maximum = max(votes.values())
    winners = {label for label, count in votes.items() if count == maximum}
    tie_order = (STATIC_BASELINE,) + tuple(
        method for method in ALL_METHODS if method != STATIC_BASELINE
    )
    return next(
        row["predictions"][method]
        for method in tie_order
        if row["predictions"][method] in winners
    )


def _paired_summary(differences: list[float]) -> dict[str, float | int]:
    summary = paired_bootstrap(
        differences,
        resamples=BOOTSTRAP_RESAMPLES,
    )
    summary.update(
        {
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
        }
    )
    return summary


def _analyze_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not evidence:
        raise ValueError("router evidence is empty")
    case_ids = [str(row["case_id"]) for row in evidence]
    if case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("router evidence case IDs must be unique and sorted")
    for row in evidence:
        if int(row["fold"]) != _fold_for_case(row["case_id"]):
            raise ValueError("router evidence fold mismatch")
        if set(row["predictions"]) != set(ALL_METHODS):
            raise ValueError("router evidence prediction coverage mismatch")
        if set(row["correctness"]) != set(ALL_METHODS):
            raise ValueError("router evidence target coverage mismatch")
        for variant, methods in ROUTER_VARIANTS.items():
            if set(row["features"][variant]) != set(methods):
                raise ValueError(f"router evidence feature coverage mismatch: {variant}")
            expected = len(_feature_names(methods))
            if any(
                len(row["features"][variant][method]) != expected
                for method in methods
            ):
                raise ValueError(f"router evidence feature length mismatch: {variant}")
    choices: dict[str, dict[str, str]] = {}
    models: dict[str, list[dict[str, Any]]] = {}
    for variant, methods in ROUTER_VARIANTS.items():
        choices[variant], models[variant] = _crossfit_variant(
            evidence,
            variant=variant,
            methods=methods,
        )
    prediction_sets: dict[str, dict[str, str | None]] = {
        STATIC_BASELINE: {
            row["case_id"]: row["predictions"][STATIC_BASELINE]
            for row in evidence
        },
        "majority_vote": {
            row["case_id"]: _majority_prediction(row) for row in evidence
        },
    }
    for variant in ROUTER_VARIANTS:
        prediction_sets[variant] = {
            row["case_id"]: row["predictions"][choices[variant][row["case_id"]]]
            for row in evidence
        }
    metrics = {
        name: _classification_metrics(evidence, predictions)
        for name, predictions in prediction_sets.items()
    }
    oracle_all_correct = sum(any(row["correctness"].values()) for row in evidence)
    oracle_without_frc_correct = sum(
        any(row["correctness"][method] for method in WITHOUT_FRC_METHODS)
        for row in evidence
    )
    frc_unique = sum(
        row["correctness"][FRC_METHOD]
        and not any(
            row["correctness"][method] for method in WITHOUT_FRC_METHODS
        )
        for row in evidence
    )
    differences = [
        float(
            row["correctness"][choices["all_methods"][row["case_id"]]]
        )
        - float(row["correctness"][STATIC_BASELINE])
        for row in evidence
    ]
    frc_contribution = [
        float(
            row["correctness"][choices["all_methods"][row["case_id"]]]
        )
        - float(
            row["correctness"][choices["without_frc"][row["case_id"]]]
        )
        for row in evidence
    ]
    folds = []
    for fold in range(FOLD_COUNT):
        rows = [row for row in evidence if int(row["fold"]) == fold]
        item: dict[str, Any] = {"fold": fold, "cases": len(rows)}
        for name in (STATIC_BASELINE, "all_methods", "without_frc"):
            item[f"{name}_accuracy"] = round(
                sum(
                    (
                        row["correctness"][name]
                        if name == STATIC_BASELINE
                        else row["correctness"][choices[name][row["case_id"]]]
                    )
                    for row in rows
                )
                / len(rows),
                6,
            )
        item["all_minus_static"] = round(
            item["all_methods_accuracy"] - item[f"{STATIC_BASELINE}_accuracy"],
            6,
        )
        folds.append(item)
    choice_counts = {
        variant: dict(Counter(variant_choices.values()))
        for variant, variant_choices in choices.items()
    }
    recall_deltas = {
        label: round(
            float(metrics["all_methods"]["per_label"][label]["recall"])
            - float(metrics[STATIC_BASELINE]["per_label"][label]["recall"]),
            6,
        )
        for label in CONFLICT_LABELS
    }
    worst_recall_label = min(
        CONFLICT_LABELS,
        key=lambda label: (recall_deltas[label], label),
    )
    unique_signatures = [
        len(set(row["selection_signatures"].values())) for row in evidence
    ]
    analysis = {
        "classification": metrics,
        "paired_all_router_minus_static": _paired_summary(differences),
        "paired_all_router_minus_without_frc": _paired_summary(frc_contribution),
        "oracle": {
            "all_methods_correct": oracle_all_correct,
            "all_methods_accuracy": round(oracle_all_correct / len(evidence), 6),
            "without_frc_correct": oracle_without_frc_correct,
            "without_frc_accuracy": round(
                oracle_without_frc_correct / len(evidence), 6
            ),
            "frc_unique_correct_cases": frc_unique,
            "frc_unique_accuracy_gain": round(frc_unique / len(evidence), 6),
        },
        "folds": folds,
        "choice_counts": choice_counts,
        "safety_and_stability": {
            "per_label_recall_delta_all_router_minus_static": recall_deltas,
            "worst_recall_delta": {
                "label": worst_recall_label,
                "delta": recall_deltas[worst_recall_label],
            },
            "folds_with_positive_accuracy_gain": sum(
                float(item["all_minus_static"]) > 0.0 for item in folds
            ),
            "folds_with_nonnegative_accuracy_gain": sum(
                float(item["all_minus_static"]) >= 0.0 for item in folds
            ),
        },
        "execution_cost": {
            "static_classifier_inputs_per_case": 1,
            "router_method_inputs_per_case": len(ALL_METHODS),
            "mean_unique_selection_signatures_per_case": round(
                sum(unique_signatures) / len(unique_signatures), 6
            ),
            "maximum_unique_selection_signatures_per_case": max(unique_signatures),
            "requires_all_method_predictions_at_inference": True,
        },
        "model_audit": models,
    }
    point_gain = float(
        analysis["paired_all_router_minus_static"]["mean_difference"]
    )
    confidence_low = float(
        analysis["paired_all_router_minus_static"]["ci_low"]
    )
    frc_gain = float(
        analysis["paired_all_router_minus_without_frc"]["mean_difference"]
    )
    analysis["outcome"] = {
        "status": (
            "DISCOVERY_ROUTING_SIGNAL_NOT_ADOPTED_"
            "FRC_CONTRIBUTION_NOT_ESTABLISHED"
        ),
        "checks": {
            "all_router_point_gain_at_least_0_05": point_gain >= 0.05,
            "all_router_paired_ci_low_above_zero": confidence_low > 0.0,
            "all_router_paired_ci_low_at_least_0_05": confidence_low >= 0.05,
            "frc_crossfit_contribution_at_least_0_01": frc_gain >= 0.01,
            "frc_unique_oracle_gain_at_least_0_05": (
                float(analysis["oracle"]["frc_unique_accuracy_gain"]) >= 0.05
            ),
            "all_folds_nonnegative_accuracy_gain": all(
                float(item["all_minus_static"]) >= 0.0 for item in folds
            ),
            "outdated_conflict_recall_not_lower_than_static": (
                recall_deltas["Conflict due to outdated information"] >= 0.0
            ),
            "misinformation_recall_above_zero": (
                float(
                    metrics["all_methods"]["per_label"][
                        "Conflict due to misinformation"
                    ]["recall"]
                )
                > 0.0
            ),
        },
        "replace_frc_selector": False,
        "router_adopted": False,
        "independent_confirmation_required": True,
        "gate_2": "NO-GO/SHADOW",
    }
    return analysis


def evaluate_selector_router(
    scored_cases: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    source_artifacts: dict[str, Any],
    prior_exposure: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_ids, selected, predicted = _validate_sources(
        scored_cases,
        selected_rows,
        predictions,
    )
    evidence = []
    for case_id in case_ids:
        gold_label = str(predicted[(case_id, ALL_METHODS[0])]["gold_label"])
        features: dict[str, dict[str, list[float]]] = {}
        for variant, methods in ROUTER_VARIANTS.items():
            features[variant] = {
                method: _feature_vector(
                    case_id,
                    method,
                    methods,
                    selected,
                    predicted,
                )
                for method in methods
            }
        evidence.append(
            {
                "case_id": case_id,
                "fold": _fold_for_case(case_id),
                "gold_label": gold_label,
                "predictions": {
                    method: predicted[(case_id, method)].get("predicted_label")
                    for method in ALL_METHODS
                },
                "correctness": {
                    method: (
                        predicted[(case_id, method)].get("predicted_label")
                        == gold_label
                    )
                    for method in ALL_METHODS
                },
                "selection_signatures": {
                    method: _canonical_sha256(
                        selected[(case_id, method)]["selected_ids"]
                    )
                    for method in ALL_METHODS
                },
                "features": features,
                "feature_inputs_include_gold": False,
            }
        )
    analysis = _analyze_evidence(evidence)
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "dataset": "Google CONFLICTS",
            "cases": len(evidence),
            "methods": list(ALL_METHODS),
            "fold_namespace": FOLD_NAMESPACE,
            "fold_count": FOLD_COUNT,
            "logistic_l2": LOGISTIC_L2,
            "logistic_max_iterations": LOGISTIC_ITERATIONS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "source_artifacts": source_artifacts,
            "evidence_artifact": {},
        },
        "development_boundary": {
            "retrospective_discovery": True,
            "pre_registered_confirmation": False,
            "prior_aggregate_probe_disclosed": True,
            "prior_exposure": prior_exposure,
            "feature_whitelist_only": True,
            "gold_used_only_as_train_target_and_reporting_label": True,
            "case_grouped_crossfit": True,
            "independent_confirmation": False,
            "gate_evidence": False,
        },
        "feature_contract": {
            "all_methods": _feature_names(ALL_METHODS),
            "without_frc": _feature_names(WITHOUT_FRC_METHODS),
            "banned_runtime_features": [
                "gold_label",
                "correct_answer",
                "classification_correct",
                "conflict_type",
                "answer_token_recall",
                "exact_answer_support",
            ],
            "raw_question_or_answer_exported_to_evidence": False,
        },
        "analysis": analysis,
        "interpretation": {
            "system_routing_headroom": (
                "Cross-fitted observable routing improves the point estimate over "
                "the frozen strongest static baseline, but the gain is not stable "
                "across every fold and safety-critical recall regresses."
            ),
            "frc_contribution": (
                "The strict without-FRC router is nearly identical and the FRC "
                "selector contributes only a small number of unique oracle wins; "
                "FRC-specific superiority is not established."
            ),
            "production": (
                "The router multiplies inference work, is not adopted, and does not "
                "authorize CANARY, DEFAULT, or any Gate 2 change."
            ),
        },
    }
    return report, evidence


def render_selector_router_markdown(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    classification = analysis["classification"]
    paired = analysis["paired_all_router_minus_static"]
    contribution = analysis["paired_all_router_minus_without_frc"]
    safety = analysis["safety_and_stability"]
    outcome = analysis["outcome"]
    lines = [
        "# CONFLICTS selector-routing discovery audit",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases: {report['metadata']['cases']}",
        "- Boundary: retrospective discovery; not an independent confirmation.",
        "- Runtime features contain no gold label, correct answer, or correctness metric.",
        f"- Gate 2: `{outcome['gate_2']}`",
        "",
        "## Cross-fitted comparison",
        "",
        "| Method | Accuracy | Macro-F1 | Correct |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        STATIC_BASELINE,
        "majority_vote",
        "without_frc",
        "all_methods",
    ):
        metrics = classification[name]
        lines.append(
            f"| {name} | {metrics['accuracy']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['correct']} |"
        )
    lines.extend(
        [
            "",
            "## Paired discovery signal",
            "",
            f"- All-method router - static baseline: {paired['mean_difference']:+.6f}",
            f"- 95% paired bootstrap CI: [{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}]",
            f"- Wins/ties/losses: {paired['wins']}/{paired['ties']}/{paired['losses']}",
            f"- All-method router - strict no-FRC router: {contribution['mean_difference']:+.6f}",
            f"- FRC-only unique oracle cases: {analysis['oracle']['frc_unique_correct_cases']}",
            (
                "- Outdated-conflict recall delta vs static: "
                f"{safety['per_label_recall_delta_all_router_minus_static']['Conflict due to outdated information']:+.6f}"
            ),
            (
                "- Folds with nonnegative accuracy gain: "
                f"{safety['folds_with_nonnegative_accuracy_gain']}/{FOLD_COUNT}"
            ),
            "",
            "## Interpretation",
            "",
            (
                "The six frozen selectors contain learnable complementary behavior, "
                "but the strict no-FRC ablation reaches nearly the same result. "
                "The large outdated-conflict recall loss and two negative folds "
                "block adoption. Therefore this audit exposes selector-routing "
                "headroom, but supports neither a safe router nor FRC superiority. "
                "The result was developed after aggregate CONFLICTS results were "
                "visible and requires untouched confirmation."
            ),
            "",
            "The router remains disabled and Gate 2 remains `NO-GO/SHADOW`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_selector_router(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> tuple[Path, Path, Path]:
    for path in (json_path, markdown_path, evidence_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for row in evidence
    ).encode("utf-8")
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            archive.write(payload)
    report["metadata"]["evidence_artifact"] = {
        "path_label": evidence_path.name,
        "sha256": sha256(evidence_path),
        "record_count": len(evidence),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_selector_router_markdown(report), encoding="utf-8")
    return json_path, markdown_path, evidence_path


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_selector_router(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported selector-router schema")
    artifact = metadata.get("evidence_artifact", {})
    if artifact.get("path_label") != evidence_path.name:
        raise ValueError("selector-router evidence path mismatch")
    if artifact.get("sha256") != sha256(evidence_path):
        raise ValueError("selector-router evidence hash mismatch")
    evidence = read_gzip_jsonl(evidence_path)
    if int(artifact.get("record_count", -1)) != len(evidence):
        raise ValueError("selector-router evidence count mismatch")
    if metadata.get("cases") != len(evidence):
        raise ValueError("selector-router report case count mismatch")
    if metadata.get("methods") != list(ALL_METHODS):
        raise ValueError("selector-router method contract mismatch")
    if report.get("feature_contract", {}).get("all_methods") != _feature_names(
        ALL_METHODS
    ):
        raise ValueError("selector-router all-method feature contract mismatch")
    if report.get("feature_contract", {}).get("without_frc") != _feature_names(
        WITHOUT_FRC_METHODS
    ):
        raise ValueError("selector-router no-FRC feature contract mismatch")
    recomputed = _analyze_evidence(evidence)
    if recomputed != report.get("analysis"):
        raise ValueError("selector-router aggregates do not match evidence")
    return report
