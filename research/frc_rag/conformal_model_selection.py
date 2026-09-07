"""Nested grouped model selection for FRC evidence-sufficiency utility."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_sufficiency import (
    BASELINE_ROLE_THRESHOLD,
    DEFAULT_RATIOS,
    FEATURE_NAMES,
    PRIMARY_ALPHA,
    _auc,
    _conformal_threshold,
    _decision_metrics,
    _display_dataset_name,
    _fit_weighted_logistic,
    _predict,
    _reference_run_provenance,
    build_variants,
)
from research.frc_rag.public_evidence import read_jsonl, sha256


SCHEMA_VERSION = "frc-conformal-nested-model-selection-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_NESTED_GROUP_SELECTION"
INNER_SPLIT_VERSION = "frc-conformal-inner-selection-v1"
DEFAULT_INNER_REPEATS = 3
DEFAULT_CANDIDATES = tuple(
    {
        "candidate_id": f"{transform}-l2-{l2:g}",
        "feature_transform": transform,
        "l2": float(l2),
    }
    for transform in ("linear", "sqrt", "log1p", "quadratic")
    for l2 in (1.0, 4.0, 16.0)
)
SUPPORTED_TRANSFORMS = {"linear", "sqrt", "log1p", "quadratic"}
TRANSFORM_COMPLEXITY = {
    "linear": 0,
    "sqrt": 1,
    "log1p": 1,
    "quadratic": 2,
}
SELECTION_METRICS = (
    "false_complete_case_family_rate",
    "false_complete_rate_all_incomplete",
    "abstention_rate",
    "true_complete_declaration_rate",
    "complete_declaration_precision",
)
OUTER_METRICS = (
    *SELECTION_METRICS,
    "declaration_rate",
    "false_complete_rate_source_missing",
)


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(numeric)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "sample_std": round(statistics.stdev(ordered), 6)
        if len(ordered) > 1
        else 0.0,
    }


def _transformed_feature_names(transform: str) -> list[str]:
    if transform == "linear":
        return list(FEATURE_NAMES)
    suffix = {
        "sqrt": "sqrt",
        "log1p": "log1p",
        "quadratic": "squared",
    }.get(transform)
    if suffix is None:
        raise ValueError(f"unsupported feature transform: {transform}")
    return [*FEATURE_NAMES, *(f"{name}__{suffix}" for name in FEATURE_NAMES)]


def _transform_features(features: np.ndarray, transform: str) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if transform == "linear":
        return array
    nonnegative = np.maximum(array, 0.0)
    if transform == "sqrt":
        additional = np.sqrt(nonnegative)
    elif transform == "log1p":
        additional = np.log1p(nonnegative)
    elif transform == "quadratic":
        additional = np.square(array)
    else:
        raise ValueError(f"unsupported feature transform: {transform}")
    transformed = np.column_stack([array, additional])
    if not np.isfinite(transformed).all():
        raise ValueError("non-finite transformed feature")
    return transformed


def _inner_partition(
    case_id: str, *, outer_split_version: str, repeat_index: int
) -> str:
    digest = hashlib.sha256(
        (
            f"{INNER_SPLIT_VERSION}\0{outer_split_version}\0"
            f"{repeat_index}\0{case_id}"
        ).encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < 0.5:
        return "fit"
    if unit < 0.75:
        return "calibration"
    return "validation"


def _case_max_incomplete_scores(
    variants: list[dict[str, Any]], scores: np.ndarray
) -> np.ndarray:
    maxima: dict[str, float] = {}
    for item, score in zip(variants, scores, strict=True):
        if item["complete"]:
            continue
        maxima[item["case_id"]] = max(
            maxima.get(item["case_id"], -math.inf), float(score)
        )
    if not maxima:
        raise ValueError("calibration partition has no incomplete case families")
    return np.asarray(list(maxima.values()), dtype=np.float64)


def _fit_and_evaluate(
    fit_variants: list[dict[str, Any]],
    calibration_variants: list[dict[str, Any]],
    evaluation_variants: list[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    alpha: float,
    iterations: int,
) -> dict[str, Any]:
    transform = str(candidate["feature_transform"])
    fit_features = _transform_features(
        np.vstack([item["features"] for item in fit_variants]), transform
    )
    fit_labels = np.asarray(
        [int(item["complete"]) for item in fit_variants], dtype=np.float64
    )
    model = _fit_weighted_logistic(
        fit_features,
        fit_labels,
        l2=float(candidate["l2"]),
        iterations=iterations,
    )
    calibration_features = _transform_features(
        np.vstack([item["features"] for item in calibration_variants]), transform
    )
    calibration_scores = _predict(model, calibration_features)
    threshold = _conformal_threshold(
        _case_max_incomplete_scores(calibration_variants, calibration_scores),
        alpha,
    )
    evaluation_features = _transform_features(
        np.vstack([item["features"] for item in evaluation_variants]), transform
    )
    evaluation_scores = _predict(model, evaluation_features)
    decisions = evaluation_scores > threshold
    evaluation_labels = np.asarray(
        [bool(item["complete"]) for item in evaluation_variants]
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "feature_transform": transform,
        "feature_count": len(_transformed_feature_names(transform)),
        "l2": float(candidate["l2"]),
        "threshold": round(threshold, 12),
        "auc": round(_auc(evaluation_labels, evaluation_scores), 6),
        "metrics": _decision_metrics(evaluation_variants, decisions),
    }


def _validate_candidates(
    candidates: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    if not candidates:
        raise ValueError("model candidate grid cannot be empty")
    candidate_ids = [str(item.get("candidate_id", "")) for item in candidates]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("every model candidate requires candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("model candidate ids must be unique")
    for candidate in candidates:
        transform = str(candidate.get("feature_transform", ""))
        if transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(f"unsupported feature transform: {transform}")
        if float(candidate.get("l2", 0.0)) <= 0.0:
            raise ValueError("candidate l2 must be positive")
    return candidates


def _aggregate_inner_candidate(
    candidate: dict[str, Any], repeats: list[dict[str, Any]], *, alpha: float
) -> dict[str, Any]:
    metrics = {
        metric: _distribution(item["metrics"][metric] for item in repeats)
        for metric in SELECTION_METRICS
    }
    return {
        **candidate,
        "feature_count": len(
            _transformed_feature_names(str(candidate["feature_transform"]))
        ),
        "inner_repeats": repeats,
        "aggregate": metrics,
        "safe_in_all_inner_repeats": all(
            item["metrics"]["false_complete_case_family_rate"] <= alpha
            for item in repeats
        ),
        "safe_inner_repeat_count": sum(
            item["metrics"]["false_complete_case_family_rate"] <= alpha
            for item in repeats
        ),
    }


def _candidate_selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = item["aggregate"]
    return (
        -int(item["safe_inner_repeat_count"]),
        float(aggregate["false_complete_case_family_rate"]["mean"]),
        -float(aggregate["true_complete_declaration_rate"]["mean"]),
        -float(aggregate["complete_declaration_precision"]["mean"]),
        float(aggregate["abstention_rate"]["mean"]),
        TRANSFORM_COMPLEXITY[str(item["feature_transform"])],
        float(item["l2"]),
        str(item["candidate_id"]),
    )


def _select_candidate_within_outer_train(
    train_variants: list[dict[str, Any]],
    *,
    outer_split_version: str,
    candidates: tuple[dict[str, Any], ...],
    alpha: float,
    inner_repeats: int,
    iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        repeat_reports = []
        for repeat_index in range(inner_repeats):
            fit_variants = [
                item
                for item in train_variants
                if _inner_partition(
                    item["case_id"],
                    outer_split_version=outer_split_version,
                    repeat_index=repeat_index,
                )
                == "fit"
            ]
            calibration_variants = [
                item
                for item in train_variants
                if _inner_partition(
                    item["case_id"],
                    outer_split_version=outer_split_version,
                    repeat_index=repeat_index,
                )
                == "calibration"
            ]
            validation_variants = [
                item
                for item in train_variants
                if _inner_partition(
                    item["case_id"],
                    outer_split_version=outer_split_version,
                    repeat_index=repeat_index,
                )
                == "validation"
            ]
            if not fit_variants or not calibration_variants or not validation_variants:
                raise ValueError("inner grouped split produced an empty partition")
            result = _fit_and_evaluate(
                fit_variants,
                calibration_variants,
                validation_variants,
                candidate=candidate,
                alpha=alpha,
                iterations=iterations,
            )
            result["repeat_index"] = repeat_index
            result["fit_cases"] = len(
                {item["case_id"] for item in fit_variants}
            )
            result["calibration_cases"] = len(
                {item["case_id"] for item in calibration_variants}
            )
            result["validation_cases"] = len(
                {item["case_id"] for item in validation_variants}
            )
            repeat_reports.append(result)
        candidate_reports.append(
            _aggregate_inner_candidate(candidate, repeat_reports, alpha=alpha)
        )
    selected = min(candidate_reports, key=_candidate_selection_key)
    return selected, candidate_reports


def _outer_aggregate(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    fixed_rows = [item["fixed_linear"]["metrics"] for item in repeats]
    optimized_rows = [item["nested_selected"]["metrics"] for item in repeats]
    selected_counts = Counter(item["selected_candidate_id"] for item in repeats)
    paired = {
        "complete_recall_gain": [
            optimized["true_complete_declaration_rate"]
            - fixed["true_complete_declaration_rate"]
            for fixed, optimized in zip(fixed_rows, optimized_rows, strict=True)
        ],
        "abstention_reduction": [
            fixed["abstention_rate"] - optimized["abstention_rate"]
            for fixed, optimized in zip(fixed_rows, optimized_rows, strict=True)
        ],
        "case_family_false_complete_change": [
            optimized["false_complete_case_family_rate"]
            - fixed["false_complete_case_family_rate"]
            for fixed, optimized in zip(fixed_rows, optimized_rows, strict=True)
        ],
        "precision_change": [
            optimized["complete_declaration_precision"]
            - fixed["complete_declaration_precision"]
            for fixed, optimized in zip(fixed_rows, optimized_rows, strict=True)
        ],
    }
    return {
        "selected_candidate_counts": dict(sorted(selected_counts.items())),
        "fixed_linear": {
            metric: _distribution(row[metric] for row in fixed_rows)
            for metric in OUTER_METRICS
        },
        "nested_selected": {
            metric: _distribution(row[metric] for row in optimized_rows)
            for metric in OUTER_METRICS
        },
        "paired_differences": {
            metric: _distribution(values) for metric, values in paired.items()
        },
        "repeats_with_complete_recall_gain": sum(
            value > 0 for value in paired["complete_recall_gain"]
        ),
        "repeats_with_lower_abstention": sum(
            value > 0 for value in paired["abstention_reduction"]
        ),
        "repeats_with_case_family_risk_not_increased": sum(
            value <= 0 for value in paired["case_family_false_complete_change"]
        ),
        "repeat_count": len(repeats),
    }


def evaluate_nested_model_selection(
    source_path: Path,
    *,
    reference_run_report: Path | None = None,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    candidates: tuple[dict[str, Any], ...] = DEFAULT_CANDIDATES,
    inner_repeats: int = DEFAULT_INNER_REPEATS,
    alpha: float = PRIMARY_ALPHA,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    k: int = 5,
    token_budget: int = 1500,
    role_threshold: float = BASELINE_ROLE_THRESHOLD,
    iterations: int = 80,
) -> dict[str, Any]:
    candidates = _validate_candidates(candidates)
    if len(split_versions) < 2 or len(split_versions) != len(set(split_versions)):
        raise ValueError("at least two unique outer split versions are required")
    if inner_repeats < 2:
        raise ValueError("at least two inner repeats are required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    source_rows = list(read_jsonl(source_path))
    datasets = {
        str(row.get("dataset", "")).strip() for row in source_rows
    }
    if len(datasets) != 1 or not next(iter(datasets), ""):
        raise ValueError("source score rows require one consistent dataset name")
    dataset = _display_dataset_name(next(iter(datasets)))
    outer_repeats: list[dict[str, Any]] = []
    fixed_candidate = {
        "candidate_id": "fixed-linear-l2-4",
        "feature_transform": "linear",
        "l2": 4.0,
    }
    for outer_split_version in split_versions:
        variants = build_variants(
            source_rows,
            split_version=outer_split_version,
            ratios=ratios,
            k=k,
            token_budget=token_budget,
            role_threshold=role_threshold,
        )
        by_split = {
            split: [item for item in variants if item["split"] == split]
            for split in ("train", "calibration", "evaluation")
        }
        selected, candidate_reports = _select_candidate_within_outer_train(
            by_split["train"],
            outer_split_version=outer_split_version,
            candidates=candidates,
            alpha=alpha,
            inner_repeats=inner_repeats,
            iterations=iterations,
        )
        selected_candidate = {
            "candidate_id": selected["candidate_id"],
            "feature_transform": selected["feature_transform"],
            "l2": selected["l2"],
        }
        fixed_result = _fit_and_evaluate(
            by_split["train"],
            by_split["calibration"],
            by_split["evaluation"],
            candidate=fixed_candidate,
            alpha=alpha,
            iterations=iterations,
        )
        selected_result = _fit_and_evaluate(
            by_split["train"],
            by_split["calibration"],
            by_split["evaluation"],
            candidate=selected_candidate,
            alpha=alpha,
            iterations=iterations,
        )
        outer_repeats.append(
            {
                "outer_split_version": outer_split_version,
                "outer_case_counts": {
                    split: len({item["case_id"] for item in split_variants})
                    for split, split_variants in by_split.items()
                },
                "selected_candidate_id": selected["candidate_id"],
                "inner_selection": {
                    "selected_candidate": selected,
                    "candidate_reports": candidate_reports,
                },
                "fixed_linear": fixed_result,
                "nested_selected": selected_result,
            }
        )

    aggregate = _outer_aggregate(outer_repeats)
    recall_gain = aggregate["paired_differences"]["complete_recall_gain"]["mean"]
    risk_change = aggregate["paired_differences"][
        "case_family_false_complete_change"
    ]["mean"]
    abstention_reduction = aggregate["paired_differences"][
        "abstention_reduction"
    ]["mean"]
    adoption_checks = {
        "mean_complete_recall_gain_at_least_0_05": recall_gain >= 0.05,
        "mean_abstention_reduction_at_least_0_05": abstention_reduction >= 0.05,
        "mean_case_family_false_complete_increase_at_most_0_02": (
            risk_change <= 0.02
        ),
        "evaluation_does_not_select_candidate": True,
    }
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "dataset": dataset,
            "source_path_label": source_path.name,
            "source_sha256": sha256(source_path),
            "reference_run": _reference_run_provenance(reference_run_report),
            "outer_split_versions": list(split_versions),
            "outer_repeat_count": len(split_versions),
            "inner_split_version": INNER_SPLIT_VERSION,
            "inner_repeats": inner_repeats,
            "alpha": alpha,
            "candidate_grid": list(candidates),
            "selection_rule": (
                "Within each outer train only, every repeat uses disjoint 50% fit, 25% "
                "calibration and 25% validation case families. Rank candidates by the "
                "number of validation repeats at or below alpha, mean validation risk, "
                "complete recall, precision, abstention, transform complexity, "
                "regularization and stable candidate id. Outer calibration sets the "
                "final threshold; outer evaluation is used once for reporting."
            ),
            "selector": {
                "method": "frc_select",
                "top_k": k,
                "token_budget": token_budget,
                "role_threshold": role_threshold,
            },
            "feature_leakage_control": (
                "Transforms use only the existing observable candidate-score, role, "
                "token and selected-set features. Gold evidence is a target only."
            ),
        },
        "outer_repeats": outer_repeats,
        "aggregate": aggregate,
        "decision": {
            "adoption_checks": adoption_checks,
            "adopt_nested_selected_experimental_default": all(
                adoption_checks.values()
            ),
            "finding": (
                "Nested train-only selection is adopted only if it improves mean recall "
                "and abstention by at least five points without increasing mean case-"
                "family false-complete risk by more than two points."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
            "limitations": [
            "All outer repeats reuse the same cross-domain ConditionalQA cases.",
            "The candidate grid and three-way inner split are pre-registered; no outer evaluation result adds or removes a candidate.",
                "Synthetic missingness and benchmark evidence ids do not replace flood-domain double-expert labels.",
                "Passing the experimental adoption checks would not authorize CANARY or DEFAULT.",
            ],
        },
    }


def render_nested_model_selection_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    aggregate = report["aggregate"]
    fixed = aggregate["fixed_linear"]
    selected = aggregate["nested_selected"]
    paired = aggregate["paired_differences"]
    decision = report["decision"]
    lines = [
        "# FRC-RAG 证据充分性嵌套选型实验",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 外层预注册分组：{metadata['outer_repeat_count']} 组",
        f"- 每组内层重复：{metadata['inner_repeats']} 次",
        f"- 候选配置：{len(metadata['candidate_grid'])} 个",
        f"- alpha：{metadata['alpha']}",
        f"- Gate 2：`{decision['gate_2']}`",
        "",
        "## Outer evaluation 汇总",
        "",
        "| 指标 | 固定线性 L2=4 | train 内嵌套选型 | 配对变化 |",
        "|---|---:|---:|---:|",
        (
            "| case 家族误放行率均值 | "
            f"{fixed['false_complete_case_family_rate']['mean']:.6f} | "
            f"{selected['false_complete_case_family_rate']['mean']:.6f} | "
            f"{paired['case_family_false_complete_change']['mean']:+.6f} |"
        ),
        (
            "| 拒答率均值 | "
            f"{fixed['abstention_rate']['mean']:.6f} | "
            f"{selected['abstention_rate']['mean']:.6f} | "
            f"{-paired['abstention_reduction']['mean']:+.6f} |"
        ),
        (
            "| 完整召回率均值 | "
            f"{fixed['true_complete_declaration_rate']['mean']:.6f} | "
            f"{selected['true_complete_declaration_rate']['mean']:.6f} | "
            f"{paired['complete_recall_gain']['mean']:+.6f} |"
        ),
        (
            "| 放行精度均值 | "
            f"{fixed['complete_declaration_precision']['mean']:.6f} | "
            f"{selected['complete_declaration_precision']['mean']:.6f} | "
            f"{paired['precision_change']['mean']:+.6f} |"
        ),
        "",
        "## 每个 outer 分组的选型",
        "",
        "| Outer 分组 | 选中配置 | 固定召回 | 选中召回 | 固定风险 | 选中风险 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for repeat in report["outer_repeats"]:
        fixed_metrics = repeat["fixed_linear"]["metrics"]
        selected_metrics = repeat["nested_selected"]["metrics"]
        lines.append(
            f"| `{repeat['outer_split_version']}` | "
            f"`{repeat['selected_candidate_id']}` | "
            f"{fixed_metrics['true_complete_declaration_rate']:.6f} | "
            f"{selected_metrics['true_complete_declaration_rate']:.6f} | "
            f"{fixed_metrics['false_complete_case_family_rate']:.6f} | "
            f"{selected_metrics['false_complete_case_family_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 预注册采用判据",
            "",
        ]
    )
    for name, passed in decision["adoption_checks"].items():
        lines.append(f"- `{name}`：`{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            (
                "实验默认切换："
                f"`{decision['adopt_nested_selected_experimental_default']}`。"
            ),
            "",
            "所有候选配置都只在 outer train 的三次内层 fit/calibration/validation "
            "case 分组上比较；内层 validation 不参与定阈值，outer calibration "
            "只定最终阈值，outer evaluation 不参与选型。"
            "即使采用判据通过，系统仍保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in decision["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_nested_model_selection(
    report: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> tuple[Path, Path]:
    for path in (json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(
        render_nested_model_selection_markdown(report), encoding="utf-8"
    )
    return json_path, markdown_path


def load_nested_model_selection(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported nested model-selection schema")
    repeats = report.get("outer_repeats", [])
    if len(repeats) != int(metadata.get("outer_repeat_count", -1)):
        raise ValueError("outer repeat count does not match metadata")
    if [item["outer_split_version"] for item in repeats] != metadata.get(
        "outer_split_versions"
    ):
        raise ValueError("outer split versions do not match metadata")
    recomputed = _outer_aggregate(repeats)
    if recomputed != report.get("aggregate"):
        raise ValueError("nested model-selection aggregate does not match repeats")
    return report
