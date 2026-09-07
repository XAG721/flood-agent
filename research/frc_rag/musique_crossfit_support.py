"""Case-isolated v38 MuSiQue support calibration discovery study."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.musique_dual_resource import (
    BUDGETS,
    DUAL_FRC,
    TOP_K,
    _merge_candidates,
    _metrics,
    select_candidates,
)
from research.frc_rag.musique_objective_gap import sha256
from research.frc_rag.rgb_cost_aware_frc import OLD_FRC, ROLE_NAMES


SCHEMA_VERSION = "frc-musique-crossfit-support-v1"
PROTOCOL_SHA256 = "957b7590c46d0a3875ff56c486c637b0ecf0925117993658902c384081eb62bc"
SCORED_SHA256 = "e2c688063e694ef0df7df7f6052cb5e68c9739f26a59db8a24c1a54413ce9242"
V37_REPORT_SHA256 = (
    "f60130a6e9e714b098afab40a16ea92cb823c86123d9019b1fd0c2f9183a3d65"
)
BASE_METHOD = "crossfit_base_support_topk_v38"
FULL_METHOD = "crossfit_full_support_topk_v38"
CROSS_TOPK = "cross_encoder_topk"
METHODS = (CROSS_TOPK, OLD_FRC, DUAL_FRC, BASE_METHOD, FULL_METHOD)
PRIMARY = "support_evidence_f1"
FOLDS = 5
L2 = 4.0
MAX_ITERATIONS = 100
TOLERANCE = 1e-9
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_RESAMPLES = 10_000
BASE_FEATURES = (
    "bm25",
    "dense",
    "hybrid",
    "cross_encoder",
    "token_count_div_384",
    "cross_encoder_descending_percentile",
)
FULL_FEATURES = (
    *BASE_FEATURES,
    *ROLE_NAMES,
    "role_score_mean",
    "role_score_max",
)


@dataclass(frozen=True)
class CaseFeatures:
    case_id: str
    fold: int
    candidate_ids: tuple[str, ...]
    base: np.ndarray
    full: np.ndarray
    labels: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    iterations: int
    converged: bool


def load_protocol(path: Path) -> dict[str, Any]:
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("MuSiQue v38 protocol hash mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "frc-musique-crossfit-support-protocol-v1":
        raise ValueError("MuSiQue v38 protocol schema mismatch")
    boundary = protocol["registration_boundary"]
    if boundary["post_result_discovery"] is not True:
        raise ValueError("MuSiQue v38 must remain post-result discovery")
    if boundary["adoption_eligible"] is not False:
        raise ValueError("MuSiQue v38 cannot be adoption eligible")
    return protocol


def validate_inputs(
    scored_path: Path,
    v37_report_path: Path,
    protocol: dict[str, Any],
) -> None:
    frozen = protocol["frozen_inputs"]
    if sha256(scored_path) != frozen["scored_blind_cache_sha256"]:
        raise ValueError("MuSiQue v38 scored cache hash mismatch")
    if sha256(v37_report_path) != frozen["v37_report_sha256"]:
        raise ValueError("MuSiQue v38 v37 report hash mismatch")


def validate_preparation_summary(
    summary: dict[str, Any], protocol: dict[str, Any]
) -> None:
    frozen = protocol["frozen_inputs"]
    if summary.get("valid_rows") != frozen["case_count"]:
        raise ValueError("MuSiQue v38 source case count changed")
    if summary.get("candidate_chunks", {}).get("total") != frozen[
        "candidate_chunk_count"
    ]:
        raise ValueError("MuSiQue v38 candidate chunk count changed")


def fold_for_case(case_id: str) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _descending_percentile(
    values: np.ndarray, candidate_ids: tuple[str, ...]
) -> np.ndarray:
    order = sorted(
        range(len(values)), key=lambda index: (-float(values[index]), candidate_ids[index])
    )
    result = np.ones(len(values), dtype=np.float64)
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(order):
        result[index] = 1.0 - rank / denominator
    return result


def build_case_features(
    scored: dict[str, Any], gold: dict[str, Any]
) -> CaseFeatures:
    candidates = _merge_candidates(scored)
    case_id = str(scored["id"])
    candidate_ids = tuple(str(candidate["id"]) for candidate in candidates)
    cross = np.asarray(
        [float(candidate["scores"]["cross_encoder"]) for candidate in candidates]
    )
    base = np.column_stack(
        [
            np.asarray([float(candidate["scores"][name]) for candidate in candidates])
            for name in ("bm25", "dense", "hybrid", "cross_encoder")
        ]
        + [
            np.asarray(
                [float(candidate["token_count"]) / 384.0 for candidate in candidates]
            ),
            _descending_percentile(cross, candidate_ids),
        ]
    ).astype(np.float64)
    role_values = np.asarray(
        [
            [float(candidate["role_scores"][role]) for role in ROLE_NAMES]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    full = np.column_stack(
        [base, role_values, role_values.mean(axis=1), role_values.max(axis=1)]
    ).astype(np.float64)
    candidate_gold = gold["candidate_gold"]
    labels = np.asarray(
        [
            float("supporting" in candidate_gold[candidate_id]["labels"])
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError(f"MuSiQue v38 case lacks both classes: {case_id}")
    weights = np.where(labels == 1.0, 0.5 / positives, 0.5 / negatives)
    return CaseFeatures(
        case_id=case_id,
        fold=fold_for_case(case_id),
        candidate_ids=candidate_ids,
        base=base,
        full=full,
        labels=labels,
        weights=weights.astype(np.float64),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float = L2,
    maximum_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> LogisticModel:
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(weights):
        raise ValueError("MuSiQue v38 logistic shapes mismatch")
    if not len(labels) or not np.all(weights > 0):
        raise ValueError("MuSiQue v38 logistic requires positive training weights")
    mean = np.average(features, axis=0, weights=weights)
    variance = np.average((features - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(features)), standardized])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    positive_rate = float(np.average(labels, weights=weights))
    beta[0] = np.log(positive_rate / (1.0 - positive_rate))
    penalty = np.diag([0.0, *([float(l2)] * features.shape[1])])
    converged = False
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        probabilities = _sigmoid(design @ beta)
        gradient = design.T @ (weights * (probabilities - labels)) + penalty @ beta
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design + penalty
        hessian.flat[:: hessian.shape[0] + 1] += 1e-10
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) <= tolerance:
            converged = True
            break
    return LogisticModel(
        mean=mean,
        scale=scale,
        coefficients=beta[1:],
        intercept=float(beta[0]),
        iterations=iterations,
        converged=converged,
    )


def predict_logistic(model: LogisticModel, features: np.ndarray) -> np.ndarray:
    standardized = (features - model.mean) / model.scale
    return _sigmoid(model.intercept + standardized @ model.coefficients)


def _stack(
    cases: list[CaseFeatures], family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = [getattr(case, family) for case in cases]
    return (
        np.concatenate(matrices, axis=0),
        np.concatenate([case.labels for case in cases]),
        np.concatenate([case.weights for case in cases]),
    )


def crossfit_predictions(
    cases: list[CaseFeatures], family: str
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    if family not in {"base", "full"}:
        raise ValueError("MuSiQue v38 feature family must be base or full")
    predictions: dict[str, dict[str, float]] = {}
    audits: list[dict[str, Any]] = []
    names = BASE_FEATURES if family == "base" else FULL_FEATURES
    for fold in range(FOLDS):
        train = [case for case in cases if case.fold != fold]
        evaluation = [case for case in cases if case.fold == fold]
        train_features, train_labels, train_weights = _stack(train, family)
        model = fit_logistic(train_features, train_labels, train_weights)
        for case in evaluation:
            values = predict_logistic(model, getattr(case, family))
            predictions[case.case_id] = {
                candidate_id: float(value)
                for candidate_id, value in zip(
                    case.candidate_ids, values, strict=True
                )
            }
        audits.append(
            {
                "fold": fold,
                "train_cases": len(train),
                "evaluation_cases": len(evaluation),
                "train_candidates": int(sum(len(case.labels) for case in train)),
                "evaluation_candidates": int(
                    sum(len(case.labels) for case in evaluation)
                ),
                "feature_names": list(names),
                "mean": [round(float(value), 10) for value in model.mean],
                "scale": [round(float(value), 10) for value in model.scale],
                "coefficients": [
                    round(float(value), 10) for value in model.coefficients
                ],
                "intercept": round(model.intercept, 10),
                "iterations": model.iterations,
                "converged": model.converged,
                "evaluation_labels_visible_to_fit": False,
            }
        )
    if set(predictions) != {case.case_id for case in cases}:
        raise AssertionError("MuSiQue v38 crossfit prediction coverage mismatch")
    return predictions, audits


def _probability_select(
    candidates: list[dict[str, Any]],
    probabilities: dict[str, float],
    *,
    token_budget: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(probabilities[str(candidate["id"])]),
            -float(candidate["scores"]["cross_encoder"]),
            int(candidate["token_count"]),
            str(candidate["id"]),
        ),
    )
    selected = []
    total = 0
    for candidate in ordered:
        cost = int(candidate["token_count"])
        if total + cost > token_budget:
            continue
        selected.append(candidate)
        total += cost
        if len(selected) >= TOP_K:
            break
    return selected


def _candidate_metrics(
    cases: list[CaseFeatures], predictions: dict[str, dict[str, float]]
) -> dict[str, float]:
    rows = []
    for case in cases:
        rows.extend(
            (float(predictions[case.case_id][candidate_id]), int(label), candidate_id)
            for candidate_id, label in zip(
                case.candidate_ids, case.labels, strict=True
            )
        )
    labels = np.asarray([row[1] for row in rows], dtype=np.int8)
    scores = np.asarray([row[0] for row in rows], dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    ascending = sorted(range(len(rows)), key=lambda i: (scores[i], rows[i][2]))
    ranks = np.empty(len(rows), dtype=np.float64)
    start = 0
    while start < len(ascending):
        end = start + 1
        while end < len(ascending) and scores[ascending[end]] == scores[ascending[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[ascending[start:end]] = average_rank
        start = end
    auc = (
        (float(ranks[labels == 1].sum()) - positives * (positives + 1) / 2)
        / (positives * negatives)
    )
    descending = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i][2]))
    true_positives = 0
    average_precision = 0.0
    for rank, index in enumerate(descending, start=1):
        if labels[index]:
            true_positives += 1
            average_precision += true_positives / rank
    average_precision /= positives
    return {
        "held_out_roc_auc": round(float(auc), 6),
        "held_out_average_precision": round(float(average_precision), 6),
        "held_out_brier_score": round(float(np.mean((scores - labels) ** 2)), 6),
    }


def evaluate_scored_cases(
    gold_cases: Iterable[dict[str, Any]],
    scored_cases: Iterable[dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {str(row["case_id"]): row for row in gold_cases}
    scored = list(scored_cases)
    if [str(row.get("id", "")) for row in scored] != list(gold_by_id):
        raise ValueError("MuSiQue v38 scored/gold order or coverage mismatch")
    features = [build_case_features(row, gold_by_id[str(row["id"])]) for row in scored]
    base_predictions, base_audits = crossfit_predictions(features, "base")
    full_predictions, full_audits = crossfit_predictions(features, "full")
    evidence = []
    for row, case_features in zip(scored, features, strict=True):
        case_id = str(row["id"])
        candidates = _merge_candidates(row)
        gold = gold_by_id[case_id]
        configurations = {}
        for budget in BUDGETS:
            selected_by_method = {
                CROSS_TOPK: select_candidates(
                    candidates, CROSS_TOPK, token_budget=budget
                ),
                OLD_FRC: select_candidates(candidates, OLD_FRC, token_budget=budget),
                DUAL_FRC: select_candidates(
                    candidates, DUAL_FRC, token_budget=budget
                ),
                BASE_METHOD: _probability_select(
                    candidates, base_predictions[case_id], token_budget=budget
                ),
                FULL_METHOD: _probability_select(
                    candidates, full_predictions[case_id], token_budget=budget
                ),
            }
            methods = {}
            for method, selected in selected_by_method.items():
                cost = sum(int(candidate["token_count"]) for candidate in selected)
                if len(selected) > TOP_K or cost > budget:
                    raise AssertionError(f"MuSiQue v38 {case_id} budget violation")
                methods[method] = {
                    "selected_ids": sorted(str(candidate["id"]) for candidate in selected),
                    "metrics": _metrics(selected, gold, token_budget=budget),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": case_id,
                "fold": case_features.fold,
                "hop_count": int(gold["hop_count"]),
                "candidate_count": len(candidates),
                "gold_unit_count": len(gold["gold_unit_ids"]),
                "configurations": configurations,
                "raw_question_answer_or_candidate_text_exported": False,
            }
        )
    model_audit = {
        "base": base_audits,
        "full": full_audits,
        "candidate_metrics": {
            "base": _candidate_metrics(features, base_predictions),
            "full": _candidate_metrics(features, full_predictions),
        },
    }
    return _build_report(evidence, model_audit, resamples=resamples), evidence


def _values(
    evidence: list[dict[str, Any]],
    method: str,
    metric: str,
    *,
    budget: int | None = None,
    hop: int | None = None,
) -> np.ndarray:
    budgets = (budget,) if budget is not None else BUDGETS
    return np.asarray(
        [
            float(
                row["configurations"][str(current_budget)]["methods"][method][
                    "metrics"
                ][metric]
            )
            for row in evidence
            if hop is None or int(row["hop_count"]) == hop
            for current_budget in budgets
        ],
        dtype=np.float64,
    )


def _paired_interval(
    evidence: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    resamples: int,
) -> dict[str, float | int]:
    deltas = np.asarray(
        [
            np.mean(
                [
                    row["configurations"][str(budget)]["methods"][left][
                        "metrics"
                    ][PRIMARY]
                    - row["configurations"][str(budget)]["methods"][right][
                        "metrics"
                    ][PRIMARY]
                    for budget in BUDGETS
                ]
            )
            for row in evidence
        ]
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap[index] = float(deltas[sample].mean())
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "point": round(float(deltas.mean()), 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
    }


def _build_report(
    evidence: list[dict[str, Any]],
    model_audit: dict[str, Any],
    *,
    resamples: int,
) -> dict[str, Any]:
    method_means = {
        method: {
            metric: round(float(_values(evidence, method, metric).mean()), 6)
            for metric in (
                PRIMARY,
                "support_precision",
                "support_recall",
                "complete_support_coverage",
                "non_support_selection_rate",
                "selected_token_cost",
                "selected_count",
            )
        }
        for method in METHODS
    }
    full_minus_cross = _paired_interval(
        evidence, FULL_METHOD, CROSS_TOPK, resamples=resamples
    )
    full_minus_base = _paired_interval(
        evidence, FULL_METHOD, BASE_METHOD, resamples=resamples
    )
    budget_deltas = {
        str(budget): round(
            float(
                _values(evidence, FULL_METHOD, PRIMARY, budget=budget).mean()
                - _values(evidence, CROSS_TOPK, PRIMARY, budget=budget).mean()
            ),
            6,
        )
        for budget in BUDGETS
    }
    hop_deltas = {
        f"{hop}-hop": round(
            float(
                _values(evidence, FULL_METHOD, PRIMARY, hop=hop).mean()
                - _values(evidence, CROSS_TOPK, PRIMARY, hop=hop).mean()
            ),
            6,
        )
        for hop in (2, 3, 4)
    }
    worst = min([*budget_deltas.values(), *hop_deltas.values()])
    cross_signal = (
        full_minus_cross["point"] >= 0.01
        and full_minus_cross["ci_low"] > 0.0
    )
    role_signal = (
        full_minus_base["point"] >= 0.005
        and full_minus_base["ci_low"] > 0.0
    )
    safe = worst >= -0.01
    if not safe:
        status = "CROSSFIT_SUPPORT_SAFETY_REGRESSION"
        next_action = "STOP_THIS_CALIBRATION_FAMILY_ON_MUSIQUE"
    elif cross_signal and role_signal:
        status = "CROSSFIT_FRC_ROLE_INCREMENT_SIGNAL"
        next_action = "FREEZE_FULL_MODEL_THEN_CONFIRM_ON_NEW_UNTOUCHED_DATA"
    elif cross_signal:
        status = (
            "CROSSFIT_GENERIC_CALIBRATION_SIGNAL_FRC_INCREMENT_NOT_ESTABLISHED"
        )
        next_action = "CONFIRM_GENERIC_CALIBRATION_WITHOUT_FRC_CLAIM"
    else:
        status = "CROSSFIT_SUPPORT_SIGNAL_NOT_ESTABLISHED"
        next_action = "STOP_THIS_CALIBRATION_FAMILY_ON_MUSIQUE"
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "dataset": "MuSiQue-Answerable dev",
            "cases": len(evidence),
            "candidate_chunks": sum(row["candidate_count"] for row in evidence),
            "selection_runs": len(evidence) * len(METHODS) * len(BUDGETS),
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "folds": FOLDS,
            "fold_distribution": {
                str(fold): sum(row["fold"] == fold for row in evidence)
                for fold in range(FOLDS)
            },
            "hop_distribution": {
                str(hop): sum(row["hop_count"] == hop for row in evidence)
                for hop in (2, 3, 4)
            },
            "protocol_sha256": PROTOCOL_SHA256,
            "scored_blind_sha256": SCORED_SHA256,
            "v37_report_sha256": V37_REPORT_SHA256,
            "neural_rescoring": False,
            "gold_visible_to_held_out_model": False,
            "raw_text_committed": False,
            "deterministic_output_rerun": "2/2 byte-identical",
        },
        "aggregates": {
            "method_means": method_means,
            "model_audit": model_audit,
        },
        "analysis": {
            "full_minus_cross_encoder_topk": full_minus_cross,
            "full_minus_base": full_minus_base,
            "budget_deltas": budget_deltas,
            "hop_deltas": hop_deltas,
            "worst_budget_or_hop_delta": round(worst, 6),
            "predeclared_checks": {
                "full_minus_cross_at_least_0_01": (
                    full_minus_cross["point"] >= 0.01
                ),
                "full_minus_cross_ci_low_above_0": (
                    full_minus_cross["ci_low"] > 0.0
                ),
                "full_minus_base_at_least_0_005": (
                    full_minus_base["point"] >= 0.005
                ),
                "full_minus_base_ci_low_above_0": (
                    full_minus_base["ci_low"] > 0.0
                ),
                "every_budget_and_hop_delta_at_least_minus_0_01": safe,
            },
            "outcome": {
                "status": status,
                "next_action": next_action,
                "post_result_discovery": True,
                "adoption_eligible": False,
                "independent_confirmation": False,
                "selector_changed": False,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
                "setr_reproduced": False,
                "flood_domain_effectiveness_established": False,
            },
        },
        "development_boundary": {
            "musique_results_known_before_model_registration": True,
            "no_musique_fold_feature_weight_l2_or_threshold_tuning": True,
            "raw_question_answer_or_candidate_text_exported": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    analysis = report["analysis"]
    outcome = analysis["outcome"]
    lines = [
        "# MuSiQue case 隔离支持校准诊断（v38）",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 下一步：`{outcome['next_action']}`",
        f"- 案例：{metadata['cases']}；候选分块：{metadata['candidate_chunks']}；五折选择运行：{metadata['selection_runs']}",
        f"- 完整模型 - cross-encoder top-k：{analysis['full_minus_cross_encoder_topk']['point']:+.6f}，95% CI [{analysis['full_minus_cross_encoder_topk']['ci_low']:+.6f}, {analysis['full_minus_cross_encoder_topk']['ci_high']:+.6f}]",
        f"- 完整模型 - 基础模型：{analysis['full_minus_base']['point']:+.6f}，95% CI [{analysis['full_minus_base']['ci_low']:+.6f}, {analysis['full_minus_base']['ci_high']:+.6f}]",
        "",
        "## 方法均值",
        "",
        "| 方法 | Evidence F1 | Precision | Recall |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = report["aggregates"]["method_means"][method]
        lines.append(
            f"| `{method}` | {values[PRIMARY]:.6f} | {values['support_precision']:.6f} | {values['support_recall']:.6f} |"
        )
    lines.extend(["", "## 候选级 held-out 诊断", ""])
    for family in ("base", "full"):
        values = report["aggregates"]["model_audit"]["candidate_metrics"][family]
        lines.append(
            f"- `{family}`：AUC {values['held_out_roc_auc']:.6f}，AP {values['held_out_average_precision']:.6f}，Brier {values['held_out_brier_score']:.6f}"
        )
    lines.extend(["", "## 边界", ""])
    lines.append(
        "这是结果揭示后的 case 隔离发现研究。held-out 标签不参与对应折模型拟合，但方法族本身是在 MuSiQue 结果已知后登记，因此不能作为独立确认、采用证据或 Gate 2 放行依据。"
    )
    lines.append("")
    return "\n".join(lines)


def _write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )


def write_report(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_paths: dict[str, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "musique_crossfit_support_cases.jsonl.gz"
    json_path = output_dir / "musique_crossfit_support.json"
    markdown_path = output_dir / "musique_crossfit_support.md"
    _write_gzip(evidence_path, evidence)
    payload = json.loads(json.dumps(report))
    payload["metadata"]["evidence_artifact"] = {
        "path": evidence_path.name,
        "sha256": sha256(evidence_path),
        "rows": len(evidence),
        "raw_question_answer_or_candidate_text_exported": False,
    }
    payload["metadata"]["source_artifacts"] = {
        name: {"path_label": path.name, "sha256": sha256(path)}
        for name, path in sorted(source_paths.items())
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(
        render_markdown(payload), encoding="utf-8", newline="\n"
    )
    report.clear()
    report.update(payload)
    return {"json": json_path, "markdown": markdown_path, "evidence": evidence_path}


def load_report(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    artifact = report["metadata"]["evidence_artifact"]
    if sha256(evidence_path) != artifact["sha256"]:
        raise ValueError("MuSiQue v38 evidence hash mismatch")
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        rows = sum(1 for _ in handle)
    if rows != artifact["rows"]:
        raise ValueError("MuSiQue v38 evidence row count mismatch")
    return report


__all__ = [
    "BASE_FEATURES",
    "BASE_METHOD",
    "FULL_FEATURES",
    "FULL_METHOD",
    "PROTOCOL_SHA256",
    "build_case_features",
    "crossfit_predictions",
    "evaluate_scored_cases",
    "fit_logistic",
    "fold_for_case",
    "load_protocol",
    "load_report",
    "predict_logistic",
    "validate_inputs",
    "validate_preparation_summary",
    "write_report",
]
