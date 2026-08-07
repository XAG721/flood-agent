"""Split-conformal evidence-sufficiency audit over frozen FRC candidate scores.

The experiment intentionally learns only an abstention head.  It does not retrain the
retriever/reranker, inspect gold flags while extracting features, or change Gate 2.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from research.frc_rag.public_evidence import (
    _stable_missing_evidence_ids,
    read_jsonl,
    select_precomputed,
    sha256,
)


SCHEMA_VERSION = "frc-conformal-sufficiency-v1"
SPLIT_VERSION = "frc-conformal-sufficiency-split-v1"
STATUS = "RUN_PUBLIC_REAL_MODEL_SPLIT_CONFORMAL_ABSTENTION"
DEFAULT_RATIOS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_ALPHAS = (0.05, 0.1, 0.2)
PRIMARY_ALPHA = 0.1
BASELINE_ROLE_THRESHOLD = 0.55
DATASET_DISPLAY_NAMES = {
    "conditionalqa": "ConditionalQA",
    "hotpotqa": "HotpotQA",
    "multihoprag": "MultiHop-RAG",
}

FEATURE_NAMES = (
    "candidate_count",
    "selected_count",
    "selected_token_budget_ratio",
    "selected_candidate_fraction",
    "required_role_count",
    "role_top_mean",
    "role_top_min",
    "role_top_max",
    "role_top_std",
    "role_second_mean",
    "role_second_min",
    "role_second_max",
    "role_second_std",
    "role_gap_mean",
    "role_gap_min",
    "role_gap_max",
    "role_gap_std",
    "role_top_relevance_mean",
    "role_top_relevance_min",
    "role_top_relevance_max",
    "selected_role_top_mean",
    "selected_role_top_min",
    "selected_role_top_max",
    "selected_role_top_std",
    "selected_relevance_mean",
    "selected_relevance_min",
    "selected_relevance_max",
    "selected_relevance_std",
    "candidate_top5_relevance_mean",
    "candidate_top5_relevance_max",
    "candidate_top5_relevance_std",
    "role_top_selected_fraction",
    "unique_role_top_fraction",
    "selected_overlap_cross_topk",
    "selected_role_redundancy",
    "selected_relevance_mass_fraction",
    "candidate_token_budget_ratio",
)


def _display_dataset_name(value: str) -> str:
    normalized = value.strip()
    return DATASET_DISPLAY_NAMES.get(normalized.lower(), normalized)


def _summary(values: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(array.mean()),
        float(array.min()),
        float(array.max()),
        float(array.std()),
    )


def _cross_score(candidate: dict[str, Any]) -> float:
    return float(candidate.get("scores", {}).get("cross_encoder", 0.0))


def _role_score(candidate: dict[str, Any], role: str) -> float:
    return float(candidate.get("role_scores", {}).get(role, 0.0))


def _mean_pairwise_role_similarity(
    selected: list[dict[str, Any]], roles: list[str]
) -> float:
    if len(selected) < 2 or not roles:
        return 0.0
    vectors = [
        np.asarray([_role_score(candidate, role) for role in roles], dtype=np.float64)
        for candidate in selected
    ]
    similarities: list[float] = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            similarities.append(float(left @ right) / denominator if denominator else 0.0)
    return float(np.mean(similarities)) if similarities else 0.0


def extract_features(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    k: int,
    token_budget: int,
) -> np.ndarray:
    """Extract observable score/selection features without gold-label fields."""

    candidates = list(row.get("candidates", []))
    roles = [str(role) for role in row.get("required_roles", [])]
    selected_ids = {str(candidate["id"]) for candidate in selected}

    role_top: list[float] = []
    role_second: list[float] = []
    role_gap: list[float] = []
    role_top_relevance: list[float] = []
    role_top_ids: list[str] = []
    selected_role_top: list[float] = []
    for role in roles:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -_role_score(candidate, role),
                -_cross_score(candidate),
                str(candidate["id"]),
            ),
        )
        top = ordered[0] if ordered else None
        first = _role_score(top, role) if top else 0.0
        second = _role_score(ordered[1], role) if len(ordered) > 1 else 0.0
        role_top.append(first)
        role_second.append(second)
        role_gap.append(first - second)
        role_top_relevance.append(_cross_score(top) if top else 0.0)
        if top:
            role_top_ids.append(str(top["id"]))
        selected_role_top.append(
            max((_role_score(candidate, role) for candidate in selected), default=0.0)
        )

    candidate_relevance = sorted(
        (_cross_score(candidate) for candidate in candidates), reverse=True
    )
    selected_relevance = [_cross_score(candidate) for candidate in selected]
    cross_top_ids = {
        str(candidate["id"])
        for candidate in sorted(
            candidates,
            key=lambda candidate: (-_cross_score(candidate), str(candidate["id"])),
        )[:k]
    }
    selected_tokens = sum(int(candidate.get("token_count", 1)) for candidate in selected)
    candidate_tokens = sum(
        int(candidate.get("token_count", 1)) for candidate in candidates
    )
    positive_relevance_mass = sum(max(0.0, value) for value in candidate_relevance)
    selected_positive_mass = sum(max(0.0, value) for value in selected_relevance)

    role_top_stats = _summary(role_top)
    role_second_stats = _summary(role_second)
    role_gap_stats = _summary(role_gap)
    role_top_relevance_stats = _summary(role_top_relevance)
    selected_role_stats = _summary(selected_role_top)
    selected_relevance_stats = _summary(selected_relevance)
    candidate_top5_stats = _summary(candidate_relevance[:5])

    values = (
        float(len(candidates)),
        float(len(selected)),
        selected_tokens / max(1, token_budget),
        len(selected) / max(1, len(candidates)),
        float(len(roles)),
        *role_top_stats,
        *role_second_stats,
        *role_gap_stats,
        role_top_relevance_stats[0],
        role_top_relevance_stats[1],
        role_top_relevance_stats[2],
        *selected_role_stats,
        *selected_relevance_stats,
        candidate_top5_stats[0],
        candidate_top5_stats[2],
        candidate_top5_stats[3],
        sum(role_id in selected_ids for role_id in role_top_ids)
        / max(1, len(role_top_ids)),
        len(set(role_top_ids)) / max(1, len(role_top_ids)),
        len(selected_ids & cross_top_ids) / max(1, len(selected_ids)),
        _mean_pairwise_role_similarity(selected, roles),
        selected_positive_mass / max(1e-12, positive_relevance_mass),
        candidate_tokens / max(1, token_budget),
    )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(
            f"feature schema mismatch: {len(values)} != {len(FEATURE_NAMES)}"
        )
    features = np.asarray(values, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("non-finite sufficiency feature")
    return features


def _split_for_case(case_id: str, *, split_version: str = SPLIT_VERSION) -> str:
    digest = hashlib.sha256(
        f"{split_version}\0{case_id}".encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < 0.4:
        return "train"
    if unit < 0.7:
        return "calibration"
    return "evaluation"


def _candidate_count_bucket(count: int) -> str:
    if count < 20:
        return "lt_20"
    if count < 40:
        return "20_to_39"
    if count < 60:
        return "40_to_59"
    return "ge_60"


def build_variants(
    rows: Iterable[dict[str, Any]],
    *,
    split_version: str = SPLIT_VERSION,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    k: int = 5,
    token_budget: int = 1500,
    role_threshold: float = BASELINE_ROLE_THRESHOLD,
    feature_extractor: Callable[..., np.ndarray] = extract_features,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in rows:
        case_id = str(source.get("id", ""))
        gold_ids = {str(value) for value in source.get("gold_evidence_ids", [])}
        if len(gold_ids) < 2:
            continue
        if not case_id or case_id in seen_ids:
            raise ValueError(f"missing or duplicate case id: {case_id!r}")
        seen_ids.add(case_id)
        split = _split_for_case(case_id, split_version=split_version)
        question_type = str(source.get("question_type") or "unspecified")
        required_role_count = len(source.get("required_roles", []))
        original_candidate_count = len(source.get("candidates", []))
        for ratio in ratios:
            removed_ids = _stable_missing_evidence_ids(case_id, gold_ids, ratio)
            row = {
                **source,
                "candidates": [
                    candidate
                    for candidate in source.get("candidates", [])
                    if str(candidate["id"]) not in removed_ids
                ],
            }
            selected = select_precomputed(
                row,
                "frc_select",
                k=k,
                budget=token_budget,
                threshold=role_threshold,
            )
            selected_ids = {str(candidate["id"]) for candidate in selected}
            selected_role_min = min(
                (
                    max(
                        (_role_score(candidate, str(role)) for candidate in selected),
                        default=0.0,
                    )
                    for role in row.get("required_roles", [])
                ),
                default=0.0,
            )
            variants.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "question_type": question_type,
                    "required_role_count": required_role_count,
                    "candidate_count_bucket": _candidate_count_bucket(
                        original_candidate_count
                    ),
                    "target_missing_ratio": float(ratio),
                    "removed_gold_count": len(removed_ids),
                    "source_missing": bool(removed_ids),
                    "complete": not removed_ids and gold_ids <= selected_ids,
                    "baseline_declared_complete": selected_role_min >= role_threshold,
                    "selected_ids": sorted(selected_ids),
                    "features": feature_extractor(
                        row, selected, k=k, token_budget=token_budget
                    ),
                }
            )
    if not variants:
        raise ValueError("no cases with at least two gold evidence ids")
    split_case_ids: dict[str, set[str]] = defaultdict(set)
    for variant in variants:
        split_case_ids[variant["split"]].add(variant["case_id"])
    if any(
        split_case_ids[left] & split_case_ids[right]
        for left, right in (
            ("train", "calibration"),
            ("train", "evaluation"),
            ("calibration", "evaluation"),
        )
    ):
        raise AssertionError("case leakage across train/calibration/evaluation")
    return variants


def prepare_conformal_source(
    source_path: Path,
    *,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    k: int = 5,
    token_budget: int = 1500,
    role_threshold: float = BASELINE_ROLE_THRESHOLD,
) -> dict[str, Any]:
    """Build split-independent variants once for repeated grouped evaluations."""

    rows = iter(read_jsonl(source_path))
    first_row = next(rows, None)
    if first_row is None:
        raise ValueError("source score file is empty")
    dataset = _display_dataset_name(str(first_row.get("dataset", "")).strip())
    if not dataset:
        raise ValueError("source score rows require a dataset name")
    variants = build_variants(
        chain((first_row,), rows),
        ratios=ratios,
        k=k,
        token_budget=token_budget,
        role_threshold=role_threshold,
    )
    return {
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256(source_path),
        "dataset": dataset,
        "ratios": tuple(ratios),
        "k": k,
        "token_budget": token_budget,
        "role_threshold": role_threshold,
        "variants": variants,
    }


def _fit_weighted_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    iterations: int,
) -> dict[str, Any]:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    standardized = (features - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("training split must contain complete and incomplete variants")
    sample_weight = np.where(
        labels == 1,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        logits = np.clip(design @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weight * (probabilities - labels))
        gradient += penalty @ coefficients
        curvature = sample_weight * probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        hessian += np.eye(design.shape[1]) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return {
        "means": means,
        "scales": scales,
        "intercept": float(coefficients[0]),
        "weights": coefficients[1:],
        "l2": l2,
        "iterations": iterations,
    }


def _predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["means"]) / model["scales"]
    logits = np.clip(
        model["intercept"] + standardized @ model["weights"], -35.0, 35.0
    )
    return 1.0 / (1.0 + np.exp(-logits))


def _conformal_rank(calibration_units: int, alpha: float) -> int:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if calibration_units < 1:
        raise ValueError("calibration split has no incomplete variants")
    rank = math.ceil((calibration_units + 1) * (1.0 - alpha))
    return min(calibration_units, rank)


def _conformal_threshold(negative_scores: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(negative_scores, dtype=np.float64))
    if not len(ordered):
        raise ValueError("calibration split has no incomplete variants")
    rank = _conformal_rank(len(ordered), alpha)
    return float(ordered[rank - 1])


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return 0.0
    comparisons = 0.0
    for score in positives:
        comparisons += float(np.sum(score > negatives))
        comparisons += 0.5 * float(np.sum(score == negatives))
    return comparisons / (len(positives) * len(negatives))


def _wilson_interval(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)]


def _decision_metrics(
    variants: list[dict[str, Any]], decisions: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray([bool(item["complete"]) for item in variants])
    source_missing = np.asarray([bool(item["source_missing"]) for item in variants])
    incomplete = ~labels
    true_complete = labels
    false_complete_count = int(np.sum(decisions & incomplete))
    source_missing_false_count = int(np.sum(decisions & source_missing))
    true_declaration_count = int(np.sum(decisions & true_complete))
    declared_count = int(decisions.sum())
    incomplete_case_ids = {
        item["case_id"]
        for item, is_incomplete in zip(variants, incomplete, strict=True)
        if is_incomplete
    }
    false_complete_case_ids = {
        item["case_id"]
        for item, declared, is_incomplete in zip(
            variants, decisions, incomplete, strict=True
        )
        if declared and is_incomplete
    }
    source_missing_case_ids = {
        item["case_id"]
        for item, is_missing in zip(variants, source_missing, strict=True)
        if is_missing
    }
    source_missing_false_case_ids = {
        item["case_id"]
        for item, declared, is_missing in zip(
            variants, decisions, source_missing, strict=True
        )
        if declared and is_missing
    }
    return {
        "variants": len(variants),
        "complete_variants": int(true_complete.sum()),
        "incomplete_variants": int(incomplete.sum()),
        "source_missing_variants": int(source_missing.sum()),
        "declarations": declared_count,
        "declaration_rate": round(declared_count / max(1, len(variants)), 6),
        "abstention_rate": round(1.0 - declared_count / max(1, len(variants)), 6),
        "false_complete_count_all_incomplete": false_complete_count,
        "false_complete_rate_all_incomplete": round(
            false_complete_count / max(1, int(incomplete.sum())), 6
        ),
        "false_complete_rate_all_incomplete_wilson_95": _wilson_interval(
            false_complete_count, int(incomplete.sum())
        ),
        "false_complete_count_source_missing": source_missing_false_count,
        "false_complete_rate_source_missing": round(
            source_missing_false_count / max(1, int(source_missing.sum())), 6
        ),
        "false_complete_rate_source_missing_wilson_95": _wilson_interval(
            source_missing_false_count, int(source_missing.sum())
        ),
        "incomplete_case_families": len(incomplete_case_ids),
        "false_complete_case_families": len(false_complete_case_ids),
        "false_complete_case_family_rate": round(
            len(false_complete_case_ids) / max(1, len(incomplete_case_ids)), 6
        ),
        "false_complete_case_family_rate_wilson_95": _wilson_interval(
            len(false_complete_case_ids), len(incomplete_case_ids)
        ),
        "source_missing_case_families": len(source_missing_case_ids),
        "source_missing_false_case_families": len(source_missing_false_case_ids),
        "source_missing_false_case_family_rate": round(
            len(source_missing_false_case_ids)
            / max(1, len(source_missing_case_ids)),
            6,
        ),
        "true_complete_declaration_rate": round(
            true_declaration_count / max(1, int(true_complete.sum())), 6
        ),
        "complete_declaration_precision": round(
            true_declaration_count / max(1, declared_count), 6
        ),
    }


def _mcnemar_exact(
    variants: list[dict[str, Any]],
    baseline: np.ndarray,
    conformal: np.ndarray,
) -> dict[str, Any]:
    incomplete = np.asarray([not bool(item["complete"]) for item in variants])
    baseline_error = baseline & incomplete
    conformal_error = conformal & incomplete
    baseline_only = int(np.sum(baseline_error & ~conformal_error))
    conformal_only = int(np.sum(~baseline_error & conformal_error))
    discordant = baseline_only + conformal_only
    if not discordant:
        p_value = 1.0
    else:
        lower = min(baseline_only, conformal_only)
        log_probabilities = [
            (
                math.lgamma(discordant + 1)
                - math.lgamma(value + 1)
                - math.lgamma(discordant - value + 1)
                - discordant * math.log(2.0)
            )
            for value in range(lower + 1)
        ]
        maximum = max(log_probabilities)
        log_tail = maximum + math.log(
            sum(math.exp(value - maximum) for value in log_probabilities)
        )
        log_two_sided = math.log(2.0) + log_tail
        p_value = min(1.0, math.exp(log_two_sided))
    return {
        "scope": "evaluation variants whose selected set is actually incomplete",
        "baseline_error_conformal_correct": baseline_only,
        "baseline_correct_conformal_error": conformal_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p_value": p_value,
    }


def _serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": [round(float(value), 12) for value in model["means"]],
        "scaler_scale": [round(float(value), 12) for value in model["scales"]],
        "intercept": round(float(model["intercept"]), 12),
        "weights": [round(float(value), 12) for value in model["weights"]],
        "l2": model["l2"],
        "iterations": model["iterations"],
        "optimizer": "deterministic class-balanced IRLS",
    }


def _reference_run_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "NOT_PROVIDED",
            "generation_outputs_used": False,
        }
    if not path.is_file():
        raise ValueError(f"reference run report does not exist: {path}")
    wanted = {
        "scoring backend",
        "embedding model",
        "reranker model",
        "score calibration",
        "role relevance mix",
        "FRC alpha/beta/gamma",
    }
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        key, value = line.removeprefix("- ").split(": ", 1)
        if key in wanted:
            values[key] = value
    missing = sorted(wanted - values.keys())
    if missing:
        raise ValueError(
            "reference run report is missing configuration keys: "
            + ", ".join(missing)
        )
    return {
        "status": "VERIFIED_BY_HASHED_REFERENCE_REPORT",
        "report_path_label": path.name,
        "report_sha256": sha256(path),
        "scoring_backend": values["scoring backend"],
        "embedding_model": values["embedding model"],
        "reranker_model": values["reranker model"],
        "score_calibration": values["score calibration"],
        "role_relevance_mix": float(values["role relevance mix"]),
        "frc_alpha_beta_gamma": [
            float(value) for value in values["FRC alpha/beta/gamma"].split("/")
        ],
        "generation_outputs_used": False,
    }


def evaluate_conformal_sufficiency(
    source_path: Path,
    *,
    reference_run_report: Path | None = None,
    split_version: str = SPLIT_VERSION,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    primary_alpha: float = PRIMARY_ALPHA,
    k: int = 5,
    token_budget: int = 1500,
    role_threshold: float = BASELINE_ROLE_THRESHOLD,
    l2: float = 4.0,
    iterations: int = 80,
    prepared_source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if primary_alpha not in alphas:
        raise ValueError("primary alpha must be present in alphas")
    if prepared_source is None:
        prepared_source = prepare_conformal_source(
            source_path,
            ratios=ratios,
            k=k,
            token_budget=token_budget,
            role_threshold=role_threshold,
        )
    expected_preparation = {
        "source_path": str(source_path.resolve()),
        "ratios": tuple(ratios),
        "k": k,
        "token_budget": token_budget,
        "role_threshold": role_threshold,
    }
    for key, expected in expected_preparation.items():
        if prepared_source.get(key) != expected:
            raise ValueError(
                f"prepared conformal source {key} does not match evaluation"
            )
    dataset = str(prepared_source["dataset"])
    source_digest = str(prepared_source["source_sha256"])
    variants = [
        {
            **item,
            "split": _split_for_case(
                str(item["case_id"]),
                split_version=split_version,
            ),
        }
        for item in prepared_source["variants"]
    ]
    by_split = {
        split: [item for item in variants if item["split"] == split]
        for split in ("train", "calibration", "evaluation")
    }
    if any(not values for values in by_split.values()):
        raise ValueError("all three grouped splits must be non-empty")

    train_features = np.vstack([item["features"] for item in by_split["train"]])
    train_labels = np.asarray(
        [int(item["complete"]) for item in by_split["train"]], dtype=np.float64
    )
    model = _fit_weighted_logistic(
        train_features, train_labels, l2=l2, iterations=iterations
    )
    for split_variants in by_split.values():
        scores = _predict(
            model, np.vstack([item["features"] for item in split_variants])
        )
        for variant, score in zip(split_variants, scores, strict=True):
            variant["sufficiency_score"] = float(score)

    calibration = by_split["calibration"]
    calibration_labels = np.asarray([bool(item["complete"]) for item in calibration])
    calibration_case_maxima: dict[str, float] = {}
    for item in calibration:
        if item["complete"]:
            continue
        calibration_case_maxima[item["case_id"]] = max(
            calibration_case_maxima.get(item["case_id"], -math.inf),
            item["sufficiency_score"],
        )
    calibration_units = np.asarray(
        list(calibration_case_maxima.values()), dtype=np.float64
    )
    thresholds = {
        alpha: _conformal_threshold(calibration_units, alpha)
        for alpha in alphas
    }
    evaluation = by_split["evaluation"]
    evaluation_scores = np.asarray([item["sufficiency_score"] for item in evaluation])
    evaluation_labels = np.asarray([bool(item["complete"]) for item in evaluation])
    baseline_decisions = np.asarray(
        [bool(item["baseline_declared_complete"]) for item in evaluation]
    )
    decisions_by_alpha = {
        alpha: evaluation_scores > threshold
        for alpha, threshold in thresholds.items()
    }
    primary_decisions = decisions_by_alpha[primary_alpha]
    for variant, declared in zip(evaluation, primary_decisions, strict=True):
        variant["conformal_declared_complete"] = bool(declared)

    alpha_results = {
        str(alpha): {
            "alpha": alpha,
            "threshold": round(thresholds[alpha], 12),
            "threshold_order_statistic_rank": _conformal_rank(
                len(calibration_units), alpha
            ),
            "finite_sample_nominal_case_error_upper_bound": round(
                (
                    len(calibration_units)
                    + 1
                    - _conformal_rank(len(calibration_units), alpha)
                )
                / (len(calibration_units) + 1),
                12,
            ),
            "evaluation": _decision_metrics(evaluation, decisions_by_alpha[alpha]),
        }
        for alpha in alphas
    }
    per_ratio = {}
    for ratio in ratios:
        indices = [
            index
            for index, item in enumerate(evaluation)
            if item["target_missing_ratio"] == ratio
        ]
        ratio_variants = [evaluation[index] for index in indices]
        per_ratio[str(ratio)] = {
            "target_missing_ratio": ratio,
            "baseline_role_coverage_heuristic": _decision_metrics(
                ratio_variants, baseline_decisions[indices]
            ),
            "split_conformal": _decision_metrics(
                ratio_variants, primary_decisions[indices]
            ),
        }

    split_summary = {}
    for split, split_variants in by_split.items():
        labels = np.asarray([bool(item["complete"]) for item in split_variants])
        scores = np.asarray([item["sufficiency_score"] for item in split_variants])
        split_summary[split] = {
            "cases": len({item["case_id"] for item in split_variants}),
            "variants": len(split_variants),
            "complete_variants": int(labels.sum()),
            "incomplete_variants": int((~labels).sum()),
            "auc": round(_auc(labels, scores), 6),
        }

    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "dataset": dataset,
            "source_path_label": source_path.name,
            "source_sha256": source_digest,
            "reference_run": _reference_run_provenance(reference_run_report),
            "eligible_cases": len({item["case_id"] for item in variants}),
            "variant_count": len(variants),
            "split_version": split_version,
            "split_rule": "SHA-256(case_id): train [0,.4), calibration [.4,.7), evaluation [.7,1)",
            "missing_ratios": list(ratios),
            "selector": {
                "method": "frc_select",
                "top_k": k,
                "token_budget": token_budget,
                "role_threshold": role_threshold,
            },
            "label_definition": (
                "complete only when no gold source evidence was removed and the frozen "
                "FRC selector selected every original gold evidence id"
            ),
            "feature_leakage_control": (
                "features use candidate scores, required roles, token counts and selected "
                "ids only; gold, gold_roles and gold_evidence_ids are excluded"
            ),
            "paired_test_implementation": (
                "two-sided exact McNemar binomial tail computed with log-sum-exp "
                "to avoid large-discordance integer-to-float overflow"
            ),
            "primary_alpha": primary_alpha,
        },
        "split_summary": split_summary,
        "model": _serializable_model(model),
        "calibration": {
            "method": (
                "one-sided split-conformal upper quantile over each calibration case's "
                "maximum incomplete-variant score; declarations require score strictly "
                "above the threshold"
            ),
            "case_level_calibration_units": len(calibration_units),
            "incomplete_calibration_variants": int((~calibration_labels).sum()),
            "thresholds": {
                str(alpha): round(threshold, 12)
                for alpha, threshold in thresholds.items()
            },
            "alphas": alpha_results,
        },
        "evaluation": {
            "auc": round(_auc(evaluation_labels, evaluation_scores), 6),
            "baseline_role_coverage_heuristic": _decision_metrics(
                evaluation, baseline_decisions
            ),
            "split_conformal": _decision_metrics(evaluation, primary_decisions),
            "per_missing_ratio": per_ratio,
            "paired_false_complete_test": _mcnemar_exact(
                evaluation, baseline_decisions, primary_decisions
            ),
        },
        "decision": {
            "finding": (
                "The calibrated head is evaluated as a safety abstention mechanism, not "
                "as a retrieval-quality improvement."
            ),
            "gate_2": "NO-GO/SHADOW",
            "production_policy": "SHADOW_OR_HUMAN_REVIEW_ONLY",
            "limitations": [
                "ConditionalQA is cross-domain rather than a real district flood corpus.",
                "Missingness is deterministic synthetic removal from a frozen candidate pool.",
                "Completeness labels are derived from benchmark evidence ids, not double-expert flood-domain judgments.",
                "The evaluation is a single grouped split and does not authorize CANARY or DEFAULT.",
            ],
        },
    }

    case_records = []
    for item in variants:
        case_records.append(
            {
                "case_id": item["case_id"],
                "split": item["split"],
                "question_type": item["question_type"],
                "required_role_count": item["required_role_count"],
                "candidate_count_bucket": item["candidate_count_bucket"],
                "target_missing_ratio": item["target_missing_ratio"],
                "removed_gold_count": item["removed_gold_count"],
                "source_missing": item["source_missing"],
                "complete": item["complete"],
                "selected_ids": item["selected_ids"],
                "baseline_declared_complete": item["baseline_declared_complete"],
                "sufficiency_score": round(item["sufficiency_score"], 12),
                "conformal_declared_complete": (
                    item.get("conformal_declared_complete")
                    if item["split"] == "evaluation"
                    else None
                ),
            }
        )
    return report, case_records


def render_conformal_sufficiency_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    reference_run = metadata["reference_run"]
    evaluation = report["evaluation"]
    baseline = evaluation["baseline_role_coverage_heuristic"]
    conformal = evaluation["split_conformal"]
    primary_alpha = metadata["primary_alpha"]
    threshold = report["calibration"]["thresholds"][str(primary_alpha)]
    lines = [
        "# FRC-RAG 证据充分性 Split-Conformal 校准实验",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 数据：{metadata['dataset']}，{metadata['eligible_cases']} 个分组样本，"
        f"{metadata['variant_count']} 个缺失变体",
        f"- 冻结分数来源 SHA-256：`{metadata['source_sha256']}`",
        f"- 主分析：alpha={primary_alpha}，严格放行阈值 `{threshold:.6f}`",
        f"- Gate 2：`{report['decision']['gate_2']}`",
        "",
        "## 分组切分",
        "",
        "| 切分 | 样本 | 变体 | 完整 | 不完整 | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if reference_run["status"] == "VERIFIED_BY_HASHED_REFERENCE_REPORT":
        lines[4:4] = [
            (
                f"- 冻结评分器：`{reference_run['embedding_model']}` + "
                f"`{reference_run['reranker_model']}`，"
                f"校准 `{reference_run['score_calibration']}`"
            ),
            f"- 参考运行报告 SHA-256：`{reference_run['report_sha256']}`",
        ]
    for split in ("train", "calibration", "evaluation"):
        item = report["split_summary"][split]
        lines.append(
            f"| {split} | {item['cases']} | {item['variants']} | "
            f"{item['complete_variants']} | {item['incomplete_variants']} | {item['auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 主评估结果",
            "",
            "| 判定器 | 放行率 | 不完整误放行率 | 缺源证据误放行率 | 完整召回率 | 放行精度 |",
            "|---|---:|---:|---:|---:|---:|",
            (
                "| 角色覆盖阈值启发式 | "
                f"{baseline['declaration_rate']:.6f} | "
                f"{baseline['false_complete_rate_all_incomplete']:.6f} | "
                f"{baseline['false_complete_rate_source_missing']:.6f} | "
                f"{baseline['true_complete_declaration_rate']:.6f} | "
                f"{baseline['complete_declaration_precision']:.6f} |"
            ),
            (
                f"| Split-conformal (alpha={primary_alpha}) | "
                f"{conformal['declaration_rate']:.6f} | "
                f"{conformal['false_complete_rate_all_incomplete']:.6f} | "
                f"{conformal['false_complete_rate_source_missing']:.6f} | "
                f"{conformal['true_complete_declaration_rate']:.6f} | "
                f"{conformal['complete_declaration_precision']:.6f} |"
            ),
            "",
            (
                "不完整误放行率的 split-conformal Wilson 95% 区间为 "
                f"`[{conformal['false_complete_rate_all_incomplete_wilson_95'][0]:.6f}, "
                f"{conformal['false_complete_rate_all_incomplete_wilson_95'][1]:.6f}]`。"
            ),
            (
                "以 case 家族计，任一不完整变体被误放行的比例为 "
                f"`{conformal['false_complete_case_family_rate']:.6f}` "
                f"（{conformal['false_complete_case_families']}/"
                f"{conformal['incomplete_case_families']}）。"
            ),
            (
                "配对 McNemar 精确检验：基线错误而校准正确 "
                f"{evaluation['paired_false_complete_test']['baseline_error_conformal_correct']} 个，"
                "基线正确而校准错误 "
                f"{evaluation['paired_false_complete_test']['baseline_correct_conformal_error']} 个，"
                f"`p={evaluation['paired_false_complete_test']['two_sided_exact_p_value']:.3e}`。"
            ),
            "",
            "## Alpha 风险—效用敏感性",
            "",
            "| Alpha | 阈值 | case 家族误放行率 | 变体误放行率 | 放行率 | 完整召回率 | 放行精度 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for alpha, item in report["calibration"]["alphas"].items():
        alpha_metrics = item["evaluation"]
        lines.append(
            f"| {float(alpha):.2f} | {item['threshold']:.6f} | "
            f"{alpha_metrics['false_complete_case_family_rate']:.6f} | "
            f"{alpha_metrics['false_complete_rate_all_incomplete']:.6f} | "
            f"{alpha_metrics['declaration_rate']:.6f} | "
            f"{alpha_metrics['true_complete_declaration_rate']:.6f} | "
            f"{alpha_metrics['complete_declaration_precision']:.6f} |"
        )
    lines.extend(
        [
            "",
            "alpha=0.05/0.10/0.20 均在 evaluation 结果揭示前预先固定；"
            "该表用于显示安全与效用权衡，不据 evaluation 重新选择主阈值。",
            "",
            "## 不同缺失比例",
            "",
            "| 目标缺失比例 | 基线误放行率 | 校准误放行率 | 基线放行率 | 校准放行率 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio, item in evaluation["per_missing_ratio"].items():
        ratio_baseline = item["baseline_role_coverage_heuristic"]
        ratio_conformal = item["split_conformal"]
        lines.append(
            f"| {float(ratio):.2f} | "
            f"{ratio_baseline['false_complete_rate_all_incomplete']:.6f} | "
            f"{ratio_conformal['false_complete_rate_all_incomplete']:.6f} | "
            f"{ratio_baseline['declaration_rate']:.6f} | "
            f"{ratio_conformal['declaration_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该实验只评估“是否应当拒答/转人工”的安全判定头，不证明检索质量提高。"
            "同一 case 的四个缺失变体始终位于同一分组，特征不读取 gold 字段；"
            "阈值只由 calibration 中每个 case 的最不利不完整变体确定，"
            "evaluation 不参与训练或选阈值。",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["decision"]["limitations"])
    lines.extend(
        [
            "",
            "因此系统继续保持 `NO-GO/SHADOW`；本结果不能替代真实防汛领域的"
            "双专家完整性标注、独立 expected-behavior 评判或生产验收。",
            "",
        ]
    )
    return "\n".join(lines)


def write_conformal_sufficiency(
    report: dict[str, Any],
    case_records: list[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    cases_path: Path,
) -> tuple[Path, Path, Path]:
    for path in (json_path, markdown_path, cases_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(
        render_conformal_sufficiency_markdown(report), encoding="utf-8"
    )
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in case_records
    ).encode("utf-8")
    with cases_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            archive.write(payload)
    return json_path, markdown_path, cases_path


def load_conformal_sufficiency(
    json_path: Path, cases_path: Path
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    if report.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported conformal sufficiency schema")
    with gzip.open(cases_path, "rt", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if len(cases) != int(report["metadata"]["variant_count"]):
        raise ValueError("conformal case artifact count does not match report")
    split_counts = defaultdict(int)
    for item in cases:
        split_counts[item["split"]] += 1
    for split, summary in report["split_summary"].items():
        if split_counts[split] != int(summary["variants"]):
            raise ValueError(f"{split} variant count does not match report")
    evaluation_cases = [item for item in cases if item["split"] == "evaluation"]
    decisions = np.asarray(
        [bool(item["conformal_declared_complete"]) for item in evaluation_cases]
    )
    recomputed = _decision_metrics(evaluation_cases, decisions)
    if recomputed != report["evaluation"]["split_conformal"]:
        raise ValueError("conformal aggregate metrics do not match case artifact")
    return report
