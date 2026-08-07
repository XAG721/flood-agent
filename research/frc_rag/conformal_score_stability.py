"""Target-fitted sufficiency-head development with retrieval-score stability signals."""

from __future__ import annotations

import gzip
import json
import statistics
from collections import Counter
from itertools import chain, combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.frc_rag.conformal_contextual import (
    _aggregate_metric_rows,
    _distribution,
    _method_metrics,
    _method_subgroups,
    _nominal_bound,
)
from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_subgroup_audit import (
    DIMENSIONS as REPORTING_DIMENSIONS,
    _aggregate_subgroups,
    _repeat_core,
)
from research.frc_rag.conformal_sufficiency import (
    BASELINE_ROLE_THRESHOLD,
    DEFAULT_RATIOS,
    FEATURE_NAMES,
    PRIMARY_ALPHA,
    _auc,
    _conformal_rank,
    _conformal_threshold,
    _display_dataset_name,
    _fit_weighted_logistic,
    _predict,
    _split_for_case,
    build_variants,
    evaluate_conformal_sufficiency,
    extract_features,
)
from research.frc_rag.public_evidence import (
    read_jsonl,
    sha256,
)


SCHEMA_VERSION = "frc-conformal-score-stability-development-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_POST_HOC_SCORE_STABILITY_HEAD_DEVELOPMENT"
RETRIEVAL_STAGES = ("bm25", "dense", "hybrid", "cross_encoder")
PER_STAGE_FEATURES = (
    "top1_top2_margin",
    "top1_top5_mean_margin",
    "top5_range",
    "selected_topk_overlap",
    "selected_mean_reciprocal_rank",
)
STAGE_PAIRS = tuple(combinations(RETRIEVAL_STAGES, 2))
SCORE_STABILITY_FEATURE_NAMES = tuple(
    f"score_stability::{stage}::{feature}"
    for stage in RETRIEVAL_STAGES
    for feature in PER_STAGE_FEATURES
) + tuple(
    f"score_stability::{left}::{right}::topk_jaccard"
    for left, right in STAGE_PAIRS
) + ("score_stability::stage_top1_consensus_fraction",)
AUGMENTED_FEATURE_NAMES = (*FEATURE_NAMES, *SCORE_STABILITY_FEATURE_NAMES)
MIN_INCOMPLETE_CASE_FAMILIES = 30
MIN_ELIGIBLE_REPEATS = 7
SUMMARY_METRICS = (
    "false_complete_case_family_rate",
    "abstention_rate",
    "true_complete_declaration_rate",
)
ADOPTION_TARGETS: dict[str, float | int | bool] = {
    "minimum_mean_complete_recall_gain": 0.02,
    "maximum_dataset_complete_recall_decrease": 0.01,
    "maximum_dataset_case_risk_increase": 0.01,
    "maximum_each_dataset_mean_case_risk": 0.1,
    "minimum_mean_evaluation_auc_gain": 0.01,
    "minimum_datasets_with_positive_auc_gain": 3,
    "minimum_datasets_with_at_least_7_of_10_alpha_control_repeats": 4,
    "minimum_mean_worst_eligible_subgroup_risk_reduction": 0.01,
    "maximum_dataset_worst_eligible_subgroup_risk_increase": 0.02,
    "all_finite_sample_nominal_bounds_at_or_below_alpha": True,
}


def _score(candidate: dict[str, Any], stage: str) -> float:
    return float(candidate.get("scores", {}).get(stage, 0.0))


def _ordered_candidates(
    candidates: list[dict[str, Any]], stage: str
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (-_score(candidate, stage), str(candidate["id"])),
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def extract_score_stability_features(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    k: int,
    token_budget: int,
) -> np.ndarray:
    """Append inference-observable multi-stage score and rank stability signals."""

    base = extract_features(
        row,
        selected,
        k=k,
        token_budget=token_budget,
    )
    candidates = list(row.get("candidates", []))
    selected_ids = {str(candidate["id"]) for candidate in selected}
    stage_topk_ids: dict[str, set[str]] = {}
    stage_top1_ids: list[str] = []
    values: list[float] = []
    for stage in RETRIEVAL_STAGES:
        ordered = _ordered_candidates(candidates, stage)
        top = ordered[:k]
        top_scores = [_score(candidate, stage) for candidate in top]
        top1 = top_scores[0] if top_scores else 0.0
        top2 = top_scores[1] if len(top_scores) > 1 else top1
        top_mean = statistics.fmean(top_scores) if top_scores else 0.0
        top_range = max(top_scores) - min(top_scores) if top_scores else 0.0
        top_ids = {str(candidate["id"]) for candidate in top}
        stage_topk_ids[stage] = top_ids
        if ordered:
            stage_top1_ids.append(str(ordered[0]["id"]))
        rank_by_id = {
            str(candidate["id"]): index + 1
            for index, candidate in enumerate(ordered)
        }
        selected_reciprocal_ranks = [
            1.0 / rank_by_id[candidate_id]
            for candidate_id in selected_ids
            if candidate_id in rank_by_id
        ]
        values.extend(
            (
                top1 - top2,
                top1 - top_mean,
                top_range,
                len(selected_ids & top_ids) / max(1, len(selected_ids)),
                (
                    statistics.fmean(selected_reciprocal_ranks)
                    if selected_reciprocal_ranks
                    else 0.0
                ),
            )
        )
    values.extend(
        _jaccard(stage_topk_ids[left], stage_topk_ids[right])
        for left, right in STAGE_PAIRS
    )
    top1_counts = Counter(stage_top1_ids)
    values.append(
        max(top1_counts.values(), default=0) / max(1, len(RETRIEVAL_STAGES))
    )
    augmented = np.concatenate([base, np.asarray(values, dtype=np.float64)])
    if len(augmented) != len(AUGMENTED_FEATURE_NAMES):
        raise AssertionError(
            "score-stability feature schema mismatch: "
            f"{len(augmented)} != {len(AUGMENTED_FEATURE_NAMES)}"
        )
    if not np.isfinite(augmented).all():
        raise ValueError("non-finite score-stability feature")
    return augmented


def prepare_score_stability_source(
    source_path: Path,
    *,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    k: int = 5,
    token_budget: int = 1500,
    role_threshold: float = BASELINE_ROLE_THRESHOLD,
) -> dict[str, Any]:
    rows = iter(read_jsonl(source_path))
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("source score file is empty")
    dataset = _display_dataset_name(str(first_row.get("dataset", "")).strip())
    if not dataset:
        raise ValueError("source score rows require a dataset name")
    row_count = 0

    def counted_rows() -> Iterable[dict[str, Any]]:
        nonlocal row_count
        for row in chain((first_row,), rows):
            row_count += 1
            yield row

    variants = build_variants(
        counted_rows(),
        ratios=ratios,
        k=k,
        token_budget=token_budget,
        role_threshold=role_threshold,
        feature_extractor=extract_score_stability_features,
    )
    return {
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256(source_path),
        "source_record_count": row_count,
        "dataset": dataset,
        "ratios": tuple(ratios),
        "k": k,
        "token_budget": token_budget,
        "role_threshold": role_threshold,
        "variants": variants,
    }


def _serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": list(AUGMENTED_FEATURE_NAMES),
        "base_feature_count": len(FEATURE_NAMES),
        "score_stability_feature_count": len(SCORE_STABILITY_FEATURE_NAMES),
        "scaler_source": "target train split only",
        "scaler_mean": [round(float(value), 12) for value in model["means"]],
        "scaler_scale": [round(float(value), 12) for value in model["scales"]],
        "intercept": round(float(model["intercept"]), 12),
        "weights": [round(float(value), 12) for value in model["weights"]],
        "l2": model["l2"],
        "iterations": model["iterations"],
        "optimizer": "deterministic class-balanced IRLS",
    }


def _calibration_case_maxima(records: list[dict[str, Any]]) -> np.ndarray:
    maxima: dict[str, float] = {}
    for record in records:
        if bool(record["complete"]):
            continue
        case_id = str(record["case_id"])
        maxima[case_id] = max(
            maxima.get(case_id, -np.inf),
            float(record["score_stability_score"]),
        )
    if not maxima:
        raise ValueError("calibration split has no incomplete case families")
    return np.asarray(list(maxima.values()), dtype=np.float64)


def _validate_case_records(case_records: list[dict[str, Any]]) -> bool:
    thresholds: dict[tuple[str, str], float] = {}
    forbidden = {"gold", "gold_roles", "gold_evidence_ids", "removed_gold_count"}
    for record in case_records:
        if forbidden & record.keys():
            return False
        if not bool(record["feature_schema_frozen_before_run"]):
            return False
        if not bool(record["target_train_fitted"]):
            return False
        if not bool(record["feature_standardization_train_only"]):
            return False
        if not bool(record["threshold_calibration_only"]):
            return False
        if bool(record["evaluation_used_for_selection"]):
            return False
        key = (str(record["dataset"]), str(record["split_version"]))
        threshold = float(record["score_stability_threshold"])
        previous = thresholds.setdefault(key, threshold)
        if previous != threshold:
            return False
    return bool(thresholds)


def _analyze_case_records(
    case_records: list[dict[str, Any]],
    *,
    datasets: list[str],
    split_versions: tuple[str, ...],
    alpha: float,
    min_incomplete_case_families: int,
    min_eligible_repeats: int,
) -> list[dict[str, Any]]:
    analyses = []
    for dataset in datasets:
        per_repeat = []
        for split_version in split_versions:
            records = [
                record
                for record in case_records
                if record["dataset"] == dataset
                and record["split_version"] == split_version
            ]
            if not records:
                raise ValueError(
                    f"missing score-stability records for {dataset}/{split_version}"
                )
            baseline = _method_metrics(records, "baseline_declared_complete")
            stability = _method_metrics(
                records, "score_stability_declared_complete"
            )
            labels = np.asarray(
                [bool(record["complete"]) for record in records], dtype=bool
            )
            baseline_auc = _auc(
                labels,
                np.asarray(
                    [float(record["baseline_score"]) for record in records]
                ),
            )
            stability_auc = _auc(
                labels,
                np.asarray(
                    [
                        float(record["score_stability_score"])
                        for record in records
                    ]
                ),
            )
            per_repeat.append(
                {
                    "split_version": split_version,
                    "baseline": baseline,
                    "score_stability": stability,
                    "delta_score_stability_minus_baseline": {
                        metric: round(
                            float(stability[metric]) - float(baseline[metric]),
                            6,
                        )
                        for metric in SUMMARY_METRICS
                    },
                    "auc": {
                        "baseline": round(baseline_auc, 6),
                        "score_stability": round(stability_auc, 6),
                        "delta": round(stability_auc - baseline_auc, 6),
                    },
                    "finite_sample_nominal_case_error_upper_bound": float(
                        records[0][
                            "score_stability_finite_sample_nominal_case_error_upper_bound"
                        ]
                    ),
                    "baseline_subgroups": _method_subgroups(
                        records,
                        "baseline_declared_complete",
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                    "score_stability_subgroups": _method_subgroups(
                        records,
                        "score_stability_declared_complete",
                        min_incomplete_case_families=(
                            min_incomplete_case_families
                        ),
                    ),
                }
            )
        baseline_subgroups = _aggregate_subgroups(
            [
                {"subgroups": repeat["baseline_subgroups"]}
                for repeat in per_repeat
            ],
            alpha=alpha,
            min_eligible_repeats=min_eligible_repeats,
        )
        stability_subgroups = _aggregate_subgroups(
            [
                {"subgroups": repeat["score_stability_subgroups"]}
                for repeat in per_repeat
            ],
            alpha=alpha,
            min_eligible_repeats=min_eligible_repeats,
        )
        baseline_worst = baseline_subgroups["worst_eligible_subgroup"]
        stability_worst = stability_subgroups["worst_eligible_subgroup"]
        worst_reduction = (
            round(
                float(
                    baseline_worst["mean_false_complete_case_family_rate"]
                )
                - float(
                    stability_worst["mean_false_complete_case_family_rate"]
                ),
                6,
            )
            if baseline_worst is not None and stability_worst is not None
            else None
        )
        analyses.append(
            {
                "dataset": dataset,
                "per_repeat": per_repeat,
                "aggregate": {
                    "baseline": _aggregate_metric_rows(per_repeat, "baseline"),
                    "score_stability": _aggregate_metric_rows(
                        per_repeat, "score_stability"
                    ),
                    "delta_score_stability_minus_baseline": {
                        metric: _distribution(
                            repeat["delta_score_stability_minus_baseline"][metric]
                            for repeat in per_repeat
                        )
                        for metric in SUMMARY_METRICS
                    },
                    "auc": {
                        "baseline": _distribution(
                            repeat["auc"]["baseline"] for repeat in per_repeat
                        ),
                        "score_stability": _distribution(
                            repeat["auc"]["score_stability"]
                            for repeat in per_repeat
                        ),
                        "delta": _distribution(
                            repeat["auc"]["delta"] for repeat in per_repeat
                        ),
                    },
                    "alpha_control_repeat_count": sum(
                        float(
                            repeat["score_stability"][
                                "false_complete_case_family_rate"
                            ]
                        )
                        <= alpha
                        for repeat in per_repeat
                    ),
                    "baseline_subgroups": baseline_subgroups,
                    "score_stability_subgroups": stability_subgroups,
                    "worst_eligible_subgroup_risk_reduction": worst_reduction,
                },
            }
        )
    if not _validate_case_records(case_records):
        raise ValueError("score-stability records violate frozen data scopes")
    return analyses


def _check(measured: Any, target: Any, passed: bool) -> dict[str, Any]:
    return {"measured": measured, "target": target, "passed": bool(passed)}


def _build_outcome(
    analyses: list[dict[str, Any]],
    case_records: list[dict[str, Any]],
    *,
    alpha: float,
    adoption_targets: dict[str, float | int | bool],
) -> dict[str, Any]:
    recall_gains = {
        item["dataset"]: float(
            item["aggregate"]["delta_score_stability_minus_baseline"][
                "true_complete_declaration_rate"
            ]["mean"]
        )
        for item in analyses
    }
    risk_increases = {
        item["dataset"]: float(
            item["aggregate"]["delta_score_stability_minus_baseline"][
                "false_complete_case_family_rate"
            ]["mean"]
        )
        for item in analyses
    }
    stability_risks = {
        item["dataset"]: float(
            item["aggregate"]["score_stability"][
                "false_complete_case_family_rate"
            ]["mean"]
        )
        for item in analyses
    }
    auc_gains = {
        item["dataset"]: float(item["aggregate"]["auc"]["delta"]["mean"])
        for item in analyses
    }
    alpha_control = {
        item["dataset"]: int(
            item["aggregate"]["alpha_control_repeat_count"]
        )
        for item in analyses
    }
    worst_reductions = {
        item["dataset"]: item["aggregate"][
            "worst_eligible_subgroup_risk_reduction"
        ]
        for item in analyses
    }
    finite_bounds = [
        float(repeat["finite_sample_nominal_case_error_upper_bound"])
        for item in analyses
        for repeat in item["per_repeat"]
    ]
    mean_recall_gain = round(statistics.fmean(recall_gains.values()), 6)
    mean_auc_gain = round(statistics.fmean(auc_gains.values()), 6)
    numeric_worst_reductions = [
        float(value) for value in worst_reductions.values() if value is not None
    ]
    mean_worst_reduction = (
        round(statistics.fmean(numeric_worst_reductions), 6)
        if len(numeric_worst_reductions) == len(analyses)
        else None
    )
    scope_valid = _validate_case_records(case_records)
    checks = {
        "feature_schema_frozen_before_run": _check(scope_valid, True, scope_valid),
        "target_train_fitted_and_standardized": _check(
            scope_valid, True, scope_valid
        ),
        "evaluation_not_used_for_selection": _check(
            scope_valid, True, scope_valid
        ),
        "threshold_calibration_only": _check(scope_valid, True, scope_valid),
        "inference_features_exclude_gold_fields": _check(
            scope_valid, True, scope_valid
        ),
        "minimum_mean_complete_recall_gain": _check(
            mean_recall_gain,
            f">={adoption_targets['minimum_mean_complete_recall_gain']}",
            mean_recall_gain
            >= float(adoption_targets["minimum_mean_complete_recall_gain"]),
        ),
        "maximum_dataset_complete_recall_decrease": _check(
            {key: round(-value, 6) for key, value in recall_gains.items()},
            f"<={adoption_targets['maximum_dataset_complete_recall_decrease']}",
            all(
                -value
                <= float(
                    adoption_targets["maximum_dataset_complete_recall_decrease"]
                )
                for value in recall_gains.values()
            ),
        ),
        "maximum_dataset_case_risk_increase": _check(
            risk_increases,
            f"<={adoption_targets['maximum_dataset_case_risk_increase']}",
            all(
                value
                <= float(adoption_targets["maximum_dataset_case_risk_increase"])
                for value in risk_increases.values()
            ),
        ),
        "maximum_each_dataset_mean_case_risk": _check(
            stability_risks,
            f"<={adoption_targets['maximum_each_dataset_mean_case_risk']}",
            all(
                value
                <= float(adoption_targets["maximum_each_dataset_mean_case_risk"])
                for value in stability_risks.values()
            ),
        ),
        "minimum_mean_evaluation_auc_gain": _check(
            mean_auc_gain,
            f">={adoption_targets['minimum_mean_evaluation_auc_gain']}",
            mean_auc_gain
            >= float(adoption_targets["minimum_mean_evaluation_auc_gain"]),
        ),
        "minimum_datasets_with_positive_auc_gain": _check(
            {"count": sum(value > 0.0 for value in auc_gains.values()), "per_dataset": auc_gains},
            f">={adoption_targets['minimum_datasets_with_positive_auc_gain']}",
            sum(value > 0.0 for value in auc_gains.values())
            >= int(adoption_targets["minimum_datasets_with_positive_auc_gain"]),
        ),
        "minimum_datasets_with_at_least_7_of_10_alpha_control_repeats": _check(
            {
                "count": sum(value >= 7 for value in alpha_control.values()),
                "per_dataset": alpha_control,
            },
            (
                ">="
                + str(
                    adoption_targets[
                        "minimum_datasets_with_at_least_7_of_10_alpha_control_repeats"
                    ]
                )
            ),
            sum(value >= 7 for value in alpha_control.values())
            >= int(
                adoption_targets[
                    "minimum_datasets_with_at_least_7_of_10_alpha_control_repeats"
                ]
            ),
        ),
        "minimum_mean_worst_eligible_subgroup_risk_reduction": _check(
            {"mean": mean_worst_reduction, "per_dataset": worst_reductions},
            (
                ">="
                + str(
                    adoption_targets[
                        "minimum_mean_worst_eligible_subgroup_risk_reduction"
                    ]
                )
            ),
            mean_worst_reduction is not None
            and mean_worst_reduction
            >= float(
                adoption_targets[
                    "minimum_mean_worst_eligible_subgroup_risk_reduction"
                ]
            ),
        ),
        "maximum_dataset_worst_eligible_subgroup_risk_increase": _check(
            {
                key: (None if value is None else round(-float(value), 6))
                for key, value in worst_reductions.items()
            },
            (
                "<="
                + str(
                    adoption_targets[
                        "maximum_dataset_worst_eligible_subgroup_risk_increase"
                    ]
                )
            ),
            all(
                value is not None
                and -float(value)
                <= float(
                    adoption_targets[
                        "maximum_dataset_worst_eligible_subgroup_risk_increase"
                    ]
                )
                for value in worst_reductions.values()
            ),
        ),
        "all_finite_sample_nominal_bounds_at_or_below_alpha": _check(
            {
                "count": len(finite_bounds),
                "maximum": round(max(finite_bounds), 12),
            },
            True,
            bool(finite_bounds) and all(value <= alpha for value in finite_bounds),
        ),
    }
    all_passed = all(check["passed"] for check in checks.values())
    return {
        "status": (
            "POST_HOC_SCORE_STABILITY_CANDIDATE"
            if all_passed
            else "DO_NOT_ADOPT"
        ),
        "post_hoc_method_development": True,
        "independent_confirmation": False,
        "alpha": alpha,
        "checks": checks,
        "all_pre_registered_adoption_checks_passed": all_passed,
        "conditional_subgroup_guarantee_claimed": False,
        "gate_2": "NO-GO/SHADOW",
    }


def evaluate_score_stability_conformal(
    dataset_sources: dict[str, Path],
    expected_robustness: dict[str, tuple[Path, dict[str, Any]]],
    *,
    split_versions: tuple[str, ...] = DEFAULT_SPLIT_VERSIONS,
    alpha: float = PRIMARY_ALPHA,
    min_incomplete_case_families: int = MIN_INCOMPLETE_CASE_FAMILIES,
    min_eligible_repeats: int = MIN_ELIGIBLE_REPEATS,
    adoption_targets: dict[str, float | int | bool] = ADOPTION_TARGETS,
    l2: float = 4.0,
    iterations: int = 80,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(dataset_sources) != set(expected_robustness):
        raise ValueError("dataset sources and robustness reports must match")
    if len(dataset_sources) < 2:
        raise ValueError("score-stability development requires multiple datasets")
    if len(set(split_versions)) != len(split_versions):
        raise ValueError("split versions must be unique")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    case_records: list[dict[str, Any]] = []
    training_repeats: dict[str, list[dict[str, Any]]] = {}
    provenance = []
    for dataset in sorted(dataset_sources):
        source_path = dataset_sources[dataset]
        expected_path, expected_report = expected_robustness[dataset]
        if tuple(expected_report["metadata"]["split_versions"]) != split_versions:
            raise ValueError("expected robustness split versions do not match")
        if float(expected_report["metadata"]["primary_alpha"]) != alpha:
            raise ValueError("expected robustness alpha does not match")
        selector = expected_report["metadata"]["selector"]
        top_k = int(selector["top_k"])
        token_budget = int(selector["token_budget"])
        role_threshold = float(selector["role_threshold"])
        prepared = prepare_score_stability_source(
            source_path,
            k=top_k,
            token_budget=token_budget,
            role_threshold=role_threshold,
        )
        if prepared["dataset"] != dataset:
            raise ValueError(
                f"source dataset {prepared['dataset']} does not match {dataset}"
            )
        if expected_report["metadata"]["dataset"] != dataset:
            raise ValueError("expected robustness dataset does not match source")
        if expected_report["metadata"]["source_sha256"] != prepared["source_sha256"]:
            raise ValueError("expected robustness source hash does not match")
        base_prepared = {
            **prepared,
            "variants": [
                {
                    **record,
                    "features": record["features"][: len(FEATURE_NAMES)],
                }
                for record in prepared["variants"]
            ],
        }

        generated_repeats = []
        dataset_training = []
        for split_version in split_versions:
            baseline_report, baseline_records = evaluate_conformal_sufficiency(
                source_path,
                split_version=split_version,
                k=top_k,
                token_budget=token_budget,
                role_threshold=role_threshold,
                prepared_source=base_prepared,
            )
            generated_repeats.append(
                _repeat_core(baseline_report, split_version=split_version)
            )
            variants = [
                {
                    **record,
                    "split": _split_for_case(
                        str(record["case_id"]), split_version=split_version
                    ),
                }
                for record in prepared["variants"]
            ]
            by_split = {
                split: [record for record in variants if record["split"] == split]
                for split in ("train", "calibration", "evaluation")
            }
            if any(not records for records in by_split.values()):
                raise ValueError("all score-stability splits must be non-empty")
            train_labels = np.asarray(
                [int(record["complete"]) for record in by_split["train"]],
                dtype=np.float64,
            )
            model = _fit_weighted_logistic(
                np.vstack([record["features"] for record in by_split["train"]]),
                train_labels,
                l2=l2,
                iterations=iterations,
            )
            for split_records in by_split.values():
                scores = _predict(
                    model,
                    np.vstack([record["features"] for record in split_records]),
                )
                for record, score in zip(split_records, scores, strict=True):
                    record["score_stability_score"] = float(score)
            calibration_units = _calibration_case_maxima(by_split["calibration"])
            threshold = _conformal_threshold(calibration_units, alpha)
            evaluation = by_split["evaluation"]
            decisions = np.asarray(
                [
                    float(record["score_stability_score"]) > threshold
                    for record in evaluation
                ],
                dtype=bool,
            )
            baseline_by_key = {
                (str(record["case_id"]), float(record["target_missing_ratio"])): record
                for record in baseline_records
                if record["split"] == "evaluation"
            }
            if len(baseline_by_key) != len(evaluation):
                raise AssertionError("baseline/stability evaluation cases differ")
            nominal_bound = _nominal_bound(len(calibration_units), alpha)
            for record, declared in zip(evaluation, decisions, strict=True):
                key = (
                    str(record["case_id"]),
                    float(record["target_missing_ratio"]),
                )
                baseline = baseline_by_key[key]
                case_records.append(
                    {
                        "dataset": dataset,
                        "split_version": split_version,
                        "case_id": record["case_id"],
                        "question_type": record["question_type"],
                        "required_role_count": str(record["required_role_count"]),
                        "candidate_count_bucket": record[
                            "candidate_count_bucket"
                        ],
                        "target_missing_ratio": record["target_missing_ratio"],
                        "complete": record["complete"],
                        "source_missing": record["source_missing"],
                        "baseline_score": baseline["sufficiency_score"],
                        "score_stability_score": round(
                            float(record["score_stability_score"]), 12
                        ),
                        "baseline_declared_complete": baseline[
                            "conformal_declared_complete"
                        ],
                        "score_stability_declared_complete": bool(declared),
                        "score_stability_threshold": round(threshold, 12),
                        "score_stability_calibration_case_level_units": len(
                            calibration_units
                        ),
                        "score_stability_threshold_order_statistic_rank": (
                            _conformal_rank(len(calibration_units), alpha)
                        ),
                        "score_stability_finite_sample_nominal_case_error_upper_bound": (
                            nominal_bound
                        ),
                        "feature_schema_frozen_before_run": True,
                        "target_train_fitted": True,
                        "feature_standardization_train_only": True,
                        "threshold_calibration_only": True,
                        "evaluation_used_for_selection": False,
                    }
                )
            dataset_training.append(
                {
                    "split_version": split_version,
                    "model": _serializable_model(model),
                    "calibration": {
                        "scope": "global",
                        "case_level_units": len(calibration_units),
                        "threshold": round(threshold, 12),
                        "threshold_order_statistic_rank": _conformal_rank(
                            len(calibration_units), alpha
                        ),
                        "finite_sample_nominal_case_error_upper_bound": (
                            nominal_bound
                        ),
                        "threshold_uses_calibration_only": True,
                    },
                }
            )
        if generated_repeats != expected_report["repeats"]:
            raise ValueError(
                f"generated repeats do not match {dataset} robustness report"
            )
        training_repeats[dataset] = dataset_training
        provenance.append(
            {
                "dataset": dataset,
                "source_path_label": source_path.name,
                "source_sha256": prepared["source_sha256"],
                "source_record_count": prepared["source_record_count"],
                "eligible_cases": len(
                    {record["case_id"] for record in prepared["variants"]}
                ),
                "baseline_report_path_label": expected_path.name,
                "baseline_report_sha256": sha256(expected_path),
                "baseline_robustness_repeats_exactly_reproduced": True,
            }
        )

    datasets = sorted(dataset_sources)
    analyses = _analyze_case_records(
        case_records,
        datasets=datasets,
        split_versions=split_versions,
        alpha=alpha,
        min_incomplete_case_families=min_incomplete_case_families,
        min_eligible_repeats=min_eligible_repeats,
    )
    outcome = _build_outcome(
        analyses,
        case_records,
        alpha=alpha,
        adoption_targets=adoption_targets,
    )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "datasets": datasets,
            "dataset_count": len(datasets),
            "split_versions": list(split_versions),
            "repeat_count_per_dataset": len(split_versions),
            "primary_alpha": alpha,
            "base_feature_count": len(FEATURE_NAMES),
            "score_stability_feature_count": len(SCORE_STABILITY_FEATURE_NAMES),
            "augmented_feature_count": len(AUGMENTED_FEATURE_NAMES),
            "retrieval_stages": list(RETRIEVAL_STAGES),
            "subgroup_dimensions": list(REPORTING_DIMENSIONS),
            "min_incomplete_case_families_per_repeat": (
                min_incomplete_case_families
            ),
            "min_eligible_repeats": min_eligible_repeats,
            "case_artifact": {},
        },
        "provenance": provenance,
        "development_protocol": {
            "post_hoc": True,
            "independent_confirmation": False,
            "target_fitted_head_required": True,
            "feature_schema_frozen_before_run": True,
            "feature_names": list(AUGMENTED_FEATURE_NAMES),
            "inference_features_use_gold_fields": False,
            "feature_or_hyperparameter_selection_on_evaluation": False,
            "feature_standardization_source": "target train split only",
            "global_threshold_uses_calibration_only": True,
            "group_specific_thresholds": False,
            "l2": l2,
            "iterations": iterations,
            "adoption_targets": adoption_targets,
        },
        "training_repeats": training_repeats,
        "datasets": analyses,
        "outcome": outcome,
        "theoretical_scope": {
            "global_marginal_statement": (
                "With the target-fitted score function fixed from train only, "
                "the global calibration order statistic retains its marginal "
                "finite-sample interpretation under case exchangeability."
            ),
            "conditional_subgroup_guarantee_claimed": False,
        },
        "decision": {
            "finding": (
                "This post-hoc experiment tests a new observable signal family; "
                "it is not an independent confirmation."
            ),
            "next_step": (
                "FREEZE_FOR_UNTOUCHED_CONFIRMATION"
                if outcome["status"] == "POST_HOC_SCORE_STABILITY_CANDIDATE"
                else "STOP_SCORE_STABILITY_EXPANSION_ON_REVEALED_EVALUATIONS"
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "KEEP_FROZEN_37_FEATURE_TARGET_FITTED_HEAD_IN_SHADOW",
            "limitations": [
                "The signal family was designed after earlier evaluations were visible.",
                "All labels are public QA evidence ids rather than flood-domain expert judgments.",
                "Global calibration does not establish conditional subgroup guarantees.",
                "No result from reused data authorizes CANARY or DEFAULT.",
            ],
        },
    }
    return report, case_records


def render_score_stability_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    outcome = report["outcome"]
    lines = [
        "# FRC-RAG 多阶段检索分数稳定性充分性头",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 数据集：{', '.join(metadata['datasets'])}",
        (
            f"- 特征：{metadata['base_feature_count']} + "
            f"{metadata['score_stability_feature_count']} = "
            f"{metadata['augmented_feature_count']}"
        ),
        "- 头拟合：每个目标数据集 train-only",
        "- 阈值：每个目标数据集 calibration-only 全局阈值",
        "- 独立确认：`False`",
        f"- Gate 2：`{outcome['gate_2']}`",
        "",
        "## 与冻结 37 维充分性头比较",
        "",
        "| 数据集 | 基线风险 | 稳定性风险 | 基线召回 | 稳定性召回 | AUC 增益 | 最坏子群风险改善 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["datasets"]:
        aggregate = item["aggregate"]
        reduction = aggregate["worst_eligible_subgroup_risk_reduction"]
        reduction_text = "-" if reduction is None else f"{reduction:.6f}"
        lines.append(
            f"| {item['dataset']} | "
            f"{aggregate['baseline']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['score_stability']['false_complete_case_family_rate']['mean']:.6f} | "
            f"{aggregate['baseline']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{aggregate['score_stability']['true_complete_declaration_rate']['mean']:.6f} | "
            f"{aggregate['auc']['delta']['mean']:.6f} | {reduction_text} |"
        )
    lines.extend(
        [
            "",
            "## 预注册采用检查",
            "",
            "| 检查 | 实测 | 目标 | 通过 |",
            "|---|---|---|---|",
        ]
    )
    for name, check in outcome["checks"].items():
        measured = json.dumps(check["measured"], ensure_ascii=False)
        lines.append(
            f"| {name} | `{measured}` | `{check['target']}` | "
            f"`{check['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            (
                "该实验只检验冻结多阶段检索分数的一致性是否为新的可观察充分性信号。"
                "它使用目标 train 拟合并在既有公开 evaluation 上事后开发，不是独立确认；"
                "无论结果如何，Gate 2 均保持 `NO-GO/SHADOW`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_score_stability_conformal(
    report: dict[str, Any],
    case_records: list[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    cases_path: Path,
) -> tuple[Path, Path, Path]:
    for path in (json_path, markdown_path, cases_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in case_records
    ).encode("utf-8")
    with cases_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as archive:
            archive.write(payload)
    report["metadata"]["case_artifact"] = {
        "path_label": cases_path.name,
        "sha256": sha256(cases_path),
        "record_count": len(case_records),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_score_stability_markdown(report), encoding="utf-8"
    )
    return json_path, markdown_path, cases_path


def load_score_stability_conformal(
    json_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported score-stability conformal schema")
    artifact = metadata["case_artifact"]
    if artifact["path_label"] != cases_path.name:
        raise ValueError("score-stability case artifact path label does not match")
    if artifact["sha256"] != sha256(cases_path):
        raise ValueError("score-stability case artifact hash does not match")
    records = []
    with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if len(records) != artifact["record_count"]:
        raise ValueError("score-stability case artifact count does not match")
    analyses = _analyze_case_records(
        records,
        datasets=list(metadata["datasets"]),
        split_versions=tuple(metadata["split_versions"]),
        alpha=float(metadata["primary_alpha"]),
        min_incomplete_case_families=int(
            metadata["min_incomplete_case_families_per_repeat"]
        ),
        min_eligible_repeats=int(metadata["min_eligible_repeats"]),
    )
    outcome = _build_outcome(
        analyses,
        records,
        alpha=float(metadata["primary_alpha"]),
        adoption_targets=report["development_protocol"]["adoption_targets"],
    )
    if analyses != report.get("datasets") or outcome != report.get("outcome"):
        raise ValueError("score-stability aggregates do not match case artifact")
    return report
