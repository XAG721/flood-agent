"""Monotone nonlinear MuSiQue chain-support experiment (v69)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag import musique_multisignal_chain_support as v68
from research.frc_rag import musique_sequential_chain_support as v66

EXPERIMENT_ID = "FRC-MUSIQUE-MONOTONE-INTERACTION-SUPPORT-V69"
PROTOCOL_SCHEMA = "frc-musique-monotone-interaction-support-protocol-v69"
SOURCE_SCHEMA = (
    "frc-musique-monotone-interaction-support-source-registration-v69"
)
FIXED_V66_THRESHOLD = v68.FIXED_V66_THRESHOLD
STAGE_SALTS = {
    "calibration": "FRC-MUSIQUE-V69-CALIBRATION|",
    "development": "FRC-MUSIQUE-V69-DEVELOPMENT|",
    "confirmation": "FRC-MUSIQUE-V69-CONFIRMATION|",
}
STAGE_PREFIXES = {
    "calibration": "m69k",
    "development": "m69d",
    "confirmation": "m69c",
}
STAGE_TARGET_PER_GROUP = {
    "calibration": 800,
    "development": 400,
    "confirmation": 400,
}
STAGE_BOOTSTRAP_SEEDS = {
    "development": 20260829,
    "confirmation": 20260830,
}
PAIR_ASSIGNMENT_SALT = "FRC-MUSIQUE-V69-PAIR-ASSIGNMENT|"
CROSSFIT_SALT = "FRC-MUSIQUE-V69-CROSSFIT|"
BASE_FEATURE_NAMES = v68.FEATURE_NAMES
MARGIN_FEATURES = (
    "direct_margin_scaled",
    "chain_min_margin_scaled",
    "chain_mean_margin_scaled",
)
MARGIN_KNOTS = (-0.5, 0.0, 0.5)
FRACTION_FEATURES = (
    "high_margin_fraction",
    "transition_entity_match_fraction",
    "distinct_selected_paragraph_ratio",
)
FRACTION_KNOTS = (0.25, 0.5, 0.75)
HINGE_FEATURE_NAMES = (
    "direct_margin_scaled_hinge_m0_5",
    "direct_margin_scaled_hinge_0_0",
    "direct_margin_scaled_hinge_p0_5",
    "chain_min_margin_scaled_hinge_m0_5",
    "chain_min_margin_scaled_hinge_0_0",
    "chain_min_margin_scaled_hinge_p0_5",
    "chain_mean_margin_scaled_hinge_m0_5",
    "chain_mean_margin_scaled_hinge_0_0",
    "chain_mean_margin_scaled_hinge_p0_5",
    "high_margin_fraction_hinge_p0_25",
    "high_margin_fraction_hinge_p0_5",
    "high_margin_fraction_hinge_p0_75",
    "transition_entity_match_fraction_hinge_p0_25",
    "transition_entity_match_fraction_hinge_p0_5",
    "transition_entity_match_fraction_hinge_p0_75",
    "distinct_selected_paragraph_ratio_hinge_p0_25",
    "distinct_selected_paragraph_ratio_hinge_p0_5",
    "distinct_selected_paragraph_ratio_hinge_p0_75",
)
INTERACTION_FEATURE_NAMES = (
    "weakest_link_x_transition_consistency",
    "mean_confidence_x_transition_consistency",
    "strong_hop_x_paragraph_coverage",
    "transition_x_paragraph_coverage",
    "direct_confidence_x_final_alignment",
)
LINEAR_FEATURE_NAMES = BASE_FEATURE_NAMES
ADDITIVE_FEATURE_NAMES = BASE_FEATURE_NAMES + HINGE_FEATURE_NAMES
CANDIDATE_FEATURE_NAMES = ADDITIVE_FEATURE_NAMES + INTERACTION_FEATURE_NAMES
UNCONSTRAINED_FEATURE_NAMES = {"hop_count_is_3", "hop_count_is_4"}
NONNEGATIVE_FEATURE_NAMES = tuple(
    name for name in CANDIDATE_FEATURE_NAMES if name not in UNCONSTRAINED_FEATURE_NAMES
)
OPTIMIZER_ITERATIONS = 5000
OPTIMIZER_LEARNING_RATE = 0.1
OPTIMIZER_L2_WEIGHT = 0.01
FAIL_CLOSED_RAW_SCORE = -1_000_000.0

sha256 = v68.sha256
normalize_question = v68.normalize_question
nested_keys = v68.nested_keys
extract_squad2_questions = v68.extract_squad2_questions
load_source_commitments = v68.load_source_commitments
resolve_placeholders = v68.resolve_placeholders


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("MuSiQue v69 protocol schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v69 experiment id changed")
    for stage, target in STAGE_TARGET_PER_GROUP.items():
        if int(value["frozen_stages"][stage]["target_per_answer_state"]) != target:
            raise ValueError(f"MuSiQue v69 {stage} size changed")
    if tuple(value["base_feature_contract"]["fixed_order"]) != BASE_FEATURE_NAMES:
        raise ValueError("MuSiQue v69 base feature order changed")
    basis = value["nonlinear_basis_contract"]
    if (
        tuple(basis["margin_features"]) != MARGIN_FEATURES
        or tuple(float(item) for item in basis["margin_knots"]) != MARGIN_KNOTS
        or tuple(basis["bounded_fraction_features"]) != FRACTION_FEATURES
        or tuple(float(item) for item in basis["bounded_fraction_knots"])
        != FRACTION_KNOTS
        or tuple(basis["fixed_hinge_order"]) != HINGE_FEATURE_NAMES
    ):
        raise ValueError("MuSiQue v69 nonlinear basis changed")
    if (
        tuple(value["interaction_contract"]["fixed_order"])
        != INTERACTION_FEATURE_NAMES
    ):
        raise ValueError("MuSiQue v69 interaction order changed")
    model = value["model_contract"]
    if (
        int(model["iterations"]) != OPTIMIZER_ITERATIONS
        or float(model["learning_rate"]) != OPTIMIZER_LEARNING_RATE
        or float(model["l2_weight"]) != OPTIMIZER_L2_WEIGHT
        or int(model["total_learned_scalars_across_all_reported_methods"])
        != 76
    ):
        raise ValueError("MuSiQue v69 optimizer or capacity changed")
    if value["stopping_and_outcomes"]["gate_2"] != "NO-GO/SHADOW":
        raise ValueError("MuSiQue v69 Gate 2 boundary changed")
    return value


def validate_source_registration(
    path: Path,
    *,
    protocol: dict[str, Any],
    train_path: Path,
    dev_path: Path,
    squad2_path: Path,
    prior_paths: Sequence[Path],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("MuSiQue v69 source registration schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v69 source registration experiment changed")
    if len(prior_paths) != 7:
        raise ValueError("MuSiQue v69 requires exactly seven prior maps")
    expected = {
        train_path: protocol["sources"]["calibration_and_development_sha256"],
        dev_path: protocol["sources"]["optional_confirmation_sha256"],
        squad2_path: value["leakage_guard_source"]["sha256"],
    }
    registered_prior = list(value["prior_exclusion_sources"].values())[:7]
    expected.update(
        {
            prior_path: registered["sha256"]
            for prior_path, registered in zip(
                prior_paths, registered_prior, strict=True
            )
        }
    )
    for source_path, expected_hash in expected.items():
        if not source_path.exists() or sha256(source_path) != expected_hash:
            raise ValueError(f"MuSiQue v69 registered source changed: {source_path}")
    if int(value["prior_exclusion_sources"]["expected_union_source_commitments"]) != 4900:
        raise ValueError("MuSiQue v69 prior commitment count changed")
    return value


def select_stage_sample(
    rows: Iterable[dict[str, Any]],
    *,
    stage: str,
    source_split: str,
    squad_questions: set[str],
    excluded_source_commitments: set[str],
    target_per_group: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGE_SALTS:
        raise ValueError(f"Unsupported MuSiQue v69 stage: {stage}")
    expected_split = "dev" if stage == "confirmation" else "train"
    if source_split != expected_split:
        raise ValueError("MuSiQue v69 stage/source split mismatch")
    target = target_per_group or STAGE_TARGET_PER_GROUP[stage]
    retained: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
        "answerable": [],
        "unanswerable": [],
    }
    source_rows = 0
    schema_reasons: dict[str, int] = {}
    squad_overlap = 0
    excluded_commitment_rows = 0
    pair_assignment_excluded = 0
    duplicate_assigned_rows = 0
    seen_source_ids: set[str] = set()
    eligible_counts = {"answerable": 0, "unanswerable": 0}
    for row in rows:
        source_rows += 1
        valid, reason = v66._valid_row(row)
        if not valid:
            schema_reasons[reason] = schema_reasons.get(reason, 0) + 1
            continue
        if v66._has_squad_overlap(row, squad_questions):
            squad_overlap += 1
            continue
        source_id = str(row["id"])
        commitment = _hash(source_id)
        if commitment in excluded_source_commitments:
            excluded_commitment_rows += 1
            continue
        group = "answerable" if row["answerable"] else "unanswerable"
        assigned = (
            "answerable"
            if int(_hash(PAIR_ASSIGNMENT_SALT + source_split + "|" + source_id), 16)
            % 2
            == 0
            else "unanswerable"
        )
        if group != assigned:
            pair_assignment_excluded += 1
            continue
        if source_id in seen_source_ids:
            duplicate_assigned_rows += 1
            continue
        seen_source_ids.add(source_id)
        eligible_counts[group] += 1
        retained[group].append(
            (_hash(STAGE_SALTS[stage] + source_id), source_id, row)
        )
        retained[group].sort(key=lambda item: (item[0], item[1]))
        if len(retained[group]) > target:
            retained[group].pop()
    if any(len(values) != target for values in retained.values()):
        raise ValueError("MuSiQue v69 has insufficient eligible balanced cases")
    selected_items = sorted(
        [item for values in retained.values() for item in values],
        key=lambda item: (item[0], item[1]),
    )
    selected = [item[2] for item in selected_items]
    selected_ids = {str(row["id"]) for row in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("MuSiQue v69 sample contains duplicate source ids")
    if any(_hash(value) in excluded_source_commitments for value in selected_ids):
        raise ValueError("MuSiQue v69 sample overlaps an excluded commitment")
    schema_excluded = sum(schema_reasons.values())
    return selected, {
        "stage": stage,
        "source_split": source_split,
        "source_rows": source_rows,
        "target_cases": target * 2,
        "selected_cases": len(selected),
        "selected_answer_state_counts": {
            "answerable": sum(bool(row["answerable"]) for row in selected),
            "unanswerable": sum(not bool(row["answerable"]) for row in selected),
        },
        "eligible_answer_state_counts_after_exclusion": eligible_counts,
        "schema_excluded_cases": schema_excluded,
        "schema_exclusion_rate": schema_excluded / source_rows if source_rows else 1.0,
        "schema_exclusion_reasons": schema_reasons,
        "squad2_exact_question_overlap_excluded": squad_overlap,
        "selected_squad2_exact_question_overlap": 0,
        "excluded_source_commitment_rows": excluded_commitment_rows,
        "selected_excluded_source_commitment_overlap": 0,
        "pair_assignment_excluded_rows": pair_assignment_excluded,
        "duplicate_assigned_source_rows_excluded": duplicate_assigned_rows,
        "selected_source_id_commitment": _hash("\n".join(sorted(selected_ids))),
    }


def prepare_blind_stage(
    selected: Sequence[dict[str, Any]], *, stage: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prefix = STAGE_PREFIXES[stage]
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    direct_inputs: list[dict[str, Any]] = []
    paragraph_counts: list[int] = []
    hop_counts: list[int] = []
    for source in selected:
        source_id = str(source["id"])
        case_id = f"{prefix}-{_hash(source_id)[:20]}"
        contexts: list[dict[str, Any]] = []
        mapping_contexts: list[dict[str, Any]] = []
        for paragraph in sorted(source["paragraphs"], key=lambda item: int(item["idx"])):
            paragraph_idx = int(paragraph["idx"])
            title = str(paragraph.get("title", "")).strip()
            text = str(paragraph["paragraph_text"]).strip()
            context = f"{title}\n{text}" if title else text
            context_row = {
                "paragraph_idx": paragraph_idx,
                "context": context,
                "context_sha256": _hash(context),
            }
            contexts.append(context_row)
            mapping_contexts.append(
                {
                    "paragraph_idx": paragraph_idx,
                    "context_sha256": context_row["context_sha256"],
                }
            )
            direct_inputs.append(
                {
                    "schema_version": "frc-musique-v69-direct-qa-input-v1",
                    "id": f"{case_id}::direct::p{paragraph_idx}",
                    "case_id": case_id,
                    "paragraph_idx": paragraph_idx,
                    "question": str(source["question"]),
                    "context": context,
                    "gold_fields_visible_to_verifier": False,
                }
            )
        templates = [str(item["question"]) for item in source["question_decomposition"]]
        prepared.append(
            {
                "schema_version": "frc-musique-v69-prepared-blind-v1",
                "id": case_id,
                "composed_question": str(source["question"]),
                "oracle_hop_templates": templates,
                "contexts": contexts,
                "gold_fields_visible_to_feature_or_chain_executor": False,
            }
        )
        maps.append(
            {
                "schema_version": "frc-musique-v69-candidate-map-v1",
                "id": case_id,
                "source_id_commitment": _hash(source_id),
                "contexts": mapping_contexts,
            }
        )
        paragraph_counts.append(len(contexts))
        hop_counts.append(len(templates))
    forbidden = {
        "answerable",
        "answer",
        "answer_aliases",
        "paragraph_support_idx",
        "is_supporting",
    }
    if forbidden & nested_keys([prepared, maps, direct_inputs]):
        raise ValueError("MuSiQue v69 blind caches expose forbidden gold fields")
    return prepared, maps, direct_inputs, {
        "prepared_cases": len(prepared),
        "direct_qa_input_rows": len(direct_inputs),
        "minimum_paragraphs": min(paragraph_counts, default=0),
        "maximum_paragraphs": max(paragraph_counts, default=0),
        "mean_paragraphs": sum(paragraph_counts) / len(paragraph_counts)
        if paragraph_counts
        else 0.0,
        "hop_count_distribution": {
            str(hop): hop_counts.count(hop) for hop in sorted(set(hop_counts))
        },
        "gold_fields_exported_to_blind_caches": False,
    }


class LocalRobertaQASupportVerifier(v68.LocalRobertaQASupportVerifier):
    def verify(
        self, rows: Sequence[dict[str, Any]], *, cache_key: str
    ) -> list[dict[str, Any]]:
        result = super().verify(rows, cache_key=cache_key)
        for row in result:
            row["schema_version"] = "frc-musique-v69-raw-qa-v1"
        return result


def aggregate_qa_rows(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> list[dict[str, Any]]:
    result = v68.aggregate_qa_rows(qa_inputs, decisions, cache_key=cache_key)
    for row in result:
        row["schema_version"] = "frc-musique-v69-aggregated-qa-v1"
    return result


initial_chain_states = v68.initial_chain_states
normalize_entity = v68.normalize_entity
entity_appears_in_context = v68.entity_appears_in_context


def build_hop_qa_inputs(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    hop_index: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    inputs, blocked = v68.build_hop_qa_inputs(
        prepared, states, hop_index=hop_index
    )
    for row in inputs:
        row["schema_version"] = "frc-musique-v69-hop-qa-input-v1"
    return inputs, blocked


def apply_hop_results(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    aggregated: Sequence[dict[str, Any]],
    blocked: dict[str, str],
    *,
    hop_index: int,
) -> list[dict[str, Any]]:
    result = v68.apply_hop_results(
        prepared, states, aggregated, blocked, hop_index=hop_index
    )
    for row in result:
        row["schema_version"] = "frc-musique-v69-hop-feature-decision-v1"
    return result


def _hinge(value: float, knot: float) -> float:
    return max(float(value) - knot, 0.0) / (1.0 - knot)


def expand_feature_vector(base: dict[str, float]) -> dict[str, float]:
    if tuple(base) != BASE_FEATURE_NAMES:
        raise ValueError("MuSiQue v69 base feature order changed")
    result = {name: float(base[name]) for name in BASE_FEATURE_NAMES}
    hinge_values: list[float] = []
    for name in MARGIN_FEATURES:
        hinge_values.extend(_hinge(base[name], knot) for knot in MARGIN_KNOTS)
    for name in FRACTION_FEATURES:
        hinge_values.extend(_hinge(base[name], knot) for knot in FRACTION_KNOTS)
    result.update(dict(zip(HINGE_FEATURE_NAMES, hinge_values, strict=True)))
    direct_unit = (float(base["direct_margin_scaled"]) + 1.0) / 2.0
    minimum_unit = (float(base["chain_min_margin_scaled"]) + 1.0) / 2.0
    mean_unit = (float(base["chain_mean_margin_scaled"]) + 1.0) / 2.0
    transition = float(base["transition_entity_match_fraction"])
    distinct = float(base["distinct_selected_paragraph_ratio"])
    high = float(base["high_margin_fraction"])
    alignment = float(base["direct_final_paragraph_agreement"])
    interactions = (
        minimum_unit * transition,
        mean_unit * transition,
        high * distinct,
        transition * distinct,
        direct_unit * alignment,
    )
    result.update(
        dict(zip(INTERACTION_FEATURE_NAMES, interactions, strict=True))
    )
    if tuple(result) != CANDIDATE_FEATURE_NAMES:
        raise ValueError("MuSiQue v69 expanded feature order changed")
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("MuSiQue v69 feature range changed")
    return result


def finalize_feature_decisions(
    prepared: Sequence[dict[str, Any]],
    direct_decisions: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    hop_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_rows = v68.finalize_feature_decisions(
        prepared, direct_decisions, states, hop_decisions
    )
    result: list[dict[str, Any]] = []
    for base_row in base_rows:
        if base_row["feature_complete"]:
            features = expand_feature_vector(base_row["features"])
        else:
            features = {name: None for name in CANDIDATE_FEATURE_NAMES}
        result.append(
            {
                "schema_version": "frc-musique-v69-feature-decision-v1",
                "case_id": base_row["case_id"],
                "hop_count": base_row["hop_count"],
                "executed_hops": base_row["executed_hops"],
                "valid_hops": base_row["valid_hops"],
                "feature_complete": base_row["feature_complete"],
                "direct_score_margin": base_row["direct_score_margin"],
                "chain_bottleneck_score": base_row["chain_bottleneck_score"],
                "chain_mean_score": base_row["chain_mean_score"],
                "features": features,
                "feature_sha256": _hash(
                    json.dumps(features, sort_keys=True, separators=(",", ":"))
                ),
                "invalid_fail_closed_used": base_row[
                    "invalid_fail_closed_used"
                ],
                "hop_decision_sha256": base_row["hop_decision_sha256"],
            }
        )
    return result


def build_gold_evidence(
    selected: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    feature_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(selected) == len(prepared) == len(feature_decisions)):
        raise ValueError("MuSiQue v69 evidence inputs differ in length")
    evidence: list[dict[str, Any]] = []
    for source, blind, feature in zip(
        selected, prepared, feature_decisions, strict=True
    ):
        case_id = str(blind["id"])
        if case_id != str(feature["case_id"]):
            raise ValueError("MuSiQue v69 evidence order changed")
        hop_count = len(source["question_decomposition"])
        if feature["feature_complete"]:
            scoring_features = feature["features"]
            direct_score = feature["direct_score_margin"]
            chain_bottleneck_score = feature["chain_bottleneck_score"]
            chain_mean_score = feature["chain_mean_score"]
        else:
            base = {
                "direct_margin_scaled": -1.0,
                "chain_min_margin_scaled": -1.0,
                "chain_mean_margin_scaled": -1.0,
                "high_margin_fraction": 0.0,
                "transition_entity_match_fraction": 0.0,
                "distinct_selected_paragraph_ratio": 0.0,
                "direct_final_paragraph_agreement": 0.0,
                "hop_count_is_3": float(hop_count == 3),
                "hop_count_is_4": float(hop_count == 4),
            }
            scoring_features = expand_feature_vector(base)
            direct_score = FAIL_CLOSED_RAW_SCORE
            chain_bottleneck_score = FAIL_CLOSED_RAW_SCORE
            chain_mean_score = FAIL_CLOSED_RAW_SCORE
        scoring_feature_sha256 = _hash(
            json.dumps(scoring_features, sort_keys=True, separators=(",", ":"))
        )
        evidence.append(
            {
                "schema_version": "frc-musique-v69-feature-evidence-v2",
                "case_id": case_id,
                "answer_state": "answerable"
                if source["answerable"]
                else "unanswerable",
                "hop_count": hop_count,
                "paragraph_count": len(source["paragraphs"]),
                "direct_score_margin": direct_score,
                "chain_bottleneck_score": chain_bottleneck_score,
                "chain_mean_score": chain_mean_score,
                "features": scoring_features,
                "feature_sha256": scoring_feature_sha256,
                "blind_feature_sha256": feature["feature_sha256"],
                "feature_complete": feature["feature_complete"],
                "chain_executed_hops": feature["executed_hops"],
                "chain_valid_hops": feature["valid_hops"],
                "chain_hop_decision_sha256": feature["hop_decision_sha256"],
                "invalid_feature_output": feature["invalid_fail_closed_used"],
                "invalid_fail_closed_raw_score": FAIL_CLOSED_RAW_SCORE
                if not feature["feature_complete"]
                else None,
            }
        )
    return evidence


def feature_matrix(
    evidence: Sequence[dict[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row["features"][name]) for name in feature_names]
            for row in evidence
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("MuSiQue v69 feature matrix is non-finite")
    return matrix


def _labels(evidence: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [float(row["answer_state"] == "answerable") for row in evidence],
        dtype=np.float64,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_projected_logistic(
    evidence: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    names = tuple(feature_names)
    matrix = feature_matrix(evidence, names)
    labels = _labels(evidence)
    weights = np.zeros(len(names), dtype=np.float64)
    intercept = 0.0
    monotone_indices = [
        index for index, name in enumerate(names) if name in NONNEGATIVE_FEATURE_NAMES
    ]
    for _ in range(OPTIMIZER_ITERATIONS):
        probabilities = _sigmoid(matrix @ weights + intercept)
        residual = probabilities - labels
        gradient = matrix.T @ residual / len(labels) + OPTIMIZER_L2_WEIGHT * weights
        intercept_gradient = float(np.mean(residual))
        weights -= OPTIMIZER_LEARNING_RATE * gradient
        intercept -= OPTIMIZER_LEARNING_RATE * intercept_gradient
        if monotone_indices:
            weights[monotone_indices] = np.maximum(weights[monotone_indices], 0.0)
    probabilities = _sigmoid(matrix @ weights + intercept)
    epsilon = 1e-12
    loss = float(
        -np.mean(
            labels * np.log(np.clip(probabilities, epsilon, 1.0))
            + (1.0 - labels)
            * np.log(np.clip(1.0 - probabilities, epsilon, 1.0))
        )
        + 0.5 * OPTIMIZER_L2_WEIGHT * float(weights @ weights)
    )
    return {
        "feature_names": list(names),
        "coefficients": [float(value) for value in weights],
        "intercept": float(intercept),
        "iterations": OPTIMIZER_ITERATIONS,
        "learning_rate": OPTIMIZER_LEARNING_RATE,
        "l2_weight": OPTIMIZER_L2_WEIGHT,
        "training_objective": loss,
        "monotone_nonnegative_feature_names": [
            names[index] for index in monotone_indices
        ],
    }


def predict_projected_logistic(
    evidence: Sequence[dict[str, Any]], model: dict[str, Any]
) -> np.ndarray:
    matrix = feature_matrix(evidence, model["feature_names"])
    weights = np.asarray(model["coefficients"], dtype=np.float64)
    return _sigmoid(matrix @ weights + float(model["intercept"]))


def score_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    decisions = scores > threshold
    answer_mask = labels == 1.0
    noanswer_mask = labels == 0.0
    answer_pass = float(np.mean(decisions[answer_mask]))
    rejection = float(np.mean(~decisions[noanswer_mask]))
    return {
        "rows": int(len(labels)),
        "answerable_rows": int(np.sum(answer_mask)),
        "unanswerable_rows": int(np.sum(noanswer_mask)),
        "threshold_exact": float(threshold),
        "answerable_pass_rate": round(answer_pass, 6),
        "unanswerable_rejection_rate": round(rejection, 6),
        "balanced_accuracy": round((answer_pass + rejection) / 2.0, 6),
    }


def select_threshold_from_scores(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    finite_scores = sorted(
        {float(value) for value in scores if math.isfinite(float(value))}
    )
    if not finite_scores:
        raise ValueError("MuSiQue v69 calibration score set is empty")
    candidates = [math.nextafter(finite_scores[0], -math.inf), *finite_scores]
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for threshold in candidates:
        metrics = score_metrics(labels, scores, threshold)
        answer_pass = float(metrics["answerable_pass_rate"])
        rejection = float(metrics["unanswerable_rejection_rate"])
        feasible = answer_pass >= 0.60 and rejection >= 0.80
        shortfall = max(0.0, 0.60 - answer_pass) + max(0.0, 0.80 - rejection)
        if feasible:
            key = (1, metrics["balanced_accuracy"], answer_pass, rejection, threshold)
        else:
            key = (0, -shortfall, metrics["balanced_accuracy"], rejection, threshold)
        ranked.append((key, metrics))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[0][1]
    selected["safety_constraints_met"] = bool(
        selected["answerable_pass_rate"] >= 0.60
        and selected["unanswerable_rejection_rate"] >= 0.80
    )
    selected["candidate_threshold_count"] = len(candidates)
    return selected


def _raw_scores(evidence: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in evidence], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"MuSiQue v69 non-finite raw score: {field}")
    return values


def _crossfit_raw(
    evidence: Sequence[dict[str, Any]], *, score_field: str, folds: int = 5
) -> dict[str, Any]:
    held_labels: list[float] = []
    held_scores: list[float] = []
    held_thresholds: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds != fold
        ]
        held = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds == fold
        ]
        selected = select_threshold_from_scores(
            _labels(train), _raw_scores(train, score_field)
        )
        threshold = float(selected["threshold_exact"])
        held_labels.extend(_labels(held).tolist())
        held_scores.extend(_raw_scores(held, score_field).tolist())
        held_thresholds.extend([threshold] * len(held))
        fold_rows.append(
            {
                "fold": fold,
                "train_cases": len(train),
                "held_out_cases": len(held),
                "threshold_exact": threshold,
                "training_safety_constraints_met": selected[
                    "safety_constraints_met"
                ],
            }
        )
    labels = np.asarray(held_labels, dtype=np.float64)
    scores = np.asarray(held_scores, dtype=np.float64)
    thresholds = np.asarray(held_thresholds, dtype=np.float64)
    decisions = scores > thresholds
    answer_mask = labels == 1.0
    noanswer_mask = labels == 0.0
    answer_pass = float(np.mean(decisions[answer_mask]))
    rejection = float(np.mean(~decisions[noanswer_mask]))
    return {
        "folds": fold_rows,
        "held_out_cases": len(labels),
        "answerable_pass_rate": round(answer_pass, 6),
        "unanswerable_rejection_rate": round(rejection, 6),
        "balanced_accuracy": round((answer_pass + rejection) / 2.0, 6),
    }


def _crossfit_model(
    evidence: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str],
    folds: int = 5,
) -> dict[str, Any]:
    held_labels: list[float] = []
    held_scores: list[float] = []
    held_thresholds: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds != fold
        ]
        held = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds == fold
        ]
        model = fit_projected_logistic(train, feature_names=feature_names)
        train_scores = predict_projected_logistic(train, model)
        selected = select_threshold_from_scores(_labels(train), train_scores)
        threshold = float(selected["threshold_exact"])
        scores = predict_projected_logistic(held, model)
        held_labels.extend(_labels(held).tolist())
        held_scores.extend(scores.tolist())
        held_thresholds.extend([threshold] * len(held))
        fold_rows.append(
            {
                "fold": fold,
                "train_cases": len(train),
                "held_out_cases": len(held),
                "threshold_exact": threshold,
                "training_safety_constraints_met": selected[
                    "safety_constraints_met"
                ],
                "model_sha256": _hash(
                    json.dumps(model, sort_keys=True, separators=(",", ":"))
                ),
            }
        )
    labels = np.asarray(held_labels, dtype=np.float64)
    scores = np.asarray(held_scores, dtype=np.float64)
    thresholds = np.asarray(held_thresholds, dtype=np.float64)
    decisions = scores > thresholds
    answer_mask = labels == 1.0
    noanswer_mask = labels == 0.0
    answer_pass = float(np.mean(decisions[answer_mask]))
    rejection = float(np.mean(~decisions[noanswer_mask]))
    return {
        "folds": fold_rows,
        "held_out_cases": len(labels),
        "answerable_pass_rate": round(answer_pass, 6),
        "unanswerable_rejection_rate": round(rejection, 6),
        "balanced_accuracy": round((answer_pass + rejection) / 2.0, 6),
    }


def fit_calibration(evidence: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = _labels(evidence)
    direct_scores = _raw_scores(evidence, "direct_score_margin")
    chain_scores = _raw_scores(evidence, "chain_bottleneck_score")
    models = {
        "linear_nine_signal_control": fit_projected_logistic(
            evidence, feature_names=LINEAR_FEATURE_NAMES
        ),
        "monotone_additive_spline_control": fit_projected_logistic(
            evidence, feature_names=ADDITIVE_FEATURE_NAMES
        ),
        "monotone_interaction_candidate": fit_projected_logistic(
            evidence, feature_names=CANDIDATE_FEATURE_NAMES
        ),
    }
    result: dict[str, Any] = {
        "calibrated_direct_composed_question": {
            "pooled": select_threshold_from_scores(labels, direct_scores),
            "crossfit": _crossfit_raw(evidence, score_field="direct_score_margin"),
        },
        "calibrated_chain_bottleneck": {
            "pooled": select_threshold_from_scores(labels, chain_scores),
            "crossfit": _crossfit_raw(
                evidence, score_field="chain_bottleneck_score"
            ),
        },
    }
    feature_contracts = {
        "linear_nine_signal_control": LINEAR_FEATURE_NAMES,
        "monotone_additive_spline_control": ADDITIVE_FEATURE_NAMES,
        "monotone_interaction_candidate": CANDIDATE_FEATURE_NAMES,
    }
    for name, feature_names in feature_contracts.items():
        scores = predict_projected_logistic(evidence, models[name])
        result[name] = {
            "model": models[name],
            "pooled": select_threshold_from_scores(labels, scores),
            "crossfit": _crossfit_model(evidence, feature_names=feature_names),
        }
    return result


def paired_correctness_interval(
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_threshold: float,
    baseline_scores: np.ndarray,
    baseline_threshold: float,
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    return v68.paired_correctness_interval(
        labels,
        candidate_scores,
        candidate_threshold,
        baseline_scores,
        baseline_threshold,
        seed=seed,
        resamples=resamples,
    )


def evaluate_stage(
    evidence: Sequence[dict[str, Any]],
    *,
    stage: str,
    calibration: dict[str, Any],
    sampling: dict[str, Any],
    structural_census: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> dict[str, Any]:
    if stage not in STAGE_BOOTSTRAP_SEEDS:
        raise ValueError("MuSiQue v69 evaluation stage changed")
    labels = _labels(evidence)
    method_scores = {
        "calibrated_direct_composed_question": _raw_scores(
            evidence, "direct_score_margin"
        ),
        "fixed_v66_full_chain": _raw_scores(evidence, "chain_bottleneck_score"),
        "calibrated_chain_bottleneck": _raw_scores(
            evidence, "chain_bottleneck_score"
        ),
        "linear_nine_signal_control": predict_projected_logistic(
            evidence, calibration["linear_nine_signal_control"]["model"]
        ),
        "monotone_additive_spline_control": predict_projected_logistic(
            evidence, calibration["monotone_additive_spline_control"]["model"]
        ),
        "monotone_interaction_candidate": predict_projected_logistic(
            evidence, calibration["monotone_interaction_candidate"]["model"]
        ),
    }
    method_thresholds = {
        "calibrated_direct_composed_question": float(
            calibration["calibrated_direct_composed_question"]["pooled"][
                "threshold_exact"
            ]
        ),
        "fixed_v66_full_chain": FIXED_V66_THRESHOLD,
        "calibrated_chain_bottleneck": float(
            calibration["calibrated_chain_bottleneck"]["pooled"][
                "threshold_exact"
            ]
        ),
        "linear_nine_signal_control": float(
            calibration["linear_nine_signal_control"]["pooled"][
                "threshold_exact"
            ]
        ),
        "monotone_additive_spline_control": float(
            calibration["monotone_additive_spline_control"]["pooled"][
                "threshold_exact"
            ]
        ),
        "monotone_interaction_candidate": float(
            calibration["monotone_interaction_candidate"]["pooled"][
                "threshold_exact"
            ]
        ),
    }
    methods = {
        name: score_metrics(labels, scores, method_thresholds[name])
        for name, scores in method_scores.items()
    }
    comparator_order = (
        "calibrated_direct_composed_question",
        "fixed_v66_full_chain",
        "calibrated_chain_bottleneck",
        "linear_nine_signal_control",
        "monotone_additive_spline_control",
    )
    baseline_name = max(
        comparator_order,
        key=lambda name: (
            methods[name]["balanced_accuracy"],
            -comparator_order.index(name),
        ),
    )
    candidate_name = "monotone_interaction_candidate"
    paired = paired_correctness_interval(
        labels,
        method_scores[candidate_name],
        method_thresholds[candidate_name],
        method_scores[baseline_name],
        method_thresholds[baseline_name],
        seed=STAGE_BOOTSTRAP_SEEDS[stage],
    )
    interaction_paired = paired_correctness_interval(
        labels,
        method_scores[candidate_name],
        method_thresholds[candidate_name],
        method_scores["monotone_additive_spline_control"],
        method_thresholds["monotone_additive_spline_control"],
        seed=STAGE_BOOTSTRAP_SEEDS[stage] + 1000,
    )
    candidate_decisions = (
        method_scores[candidate_name] > method_thresholds[candidate_name]
    )
    strata: dict[str, Any] = {}
    for hop in sorted(
        {
            int(row["hop_count"])
            for row in evidence
            if row["answer_state"] == "unanswerable"
        }
    ):
        indices = [
            index
            for index, row in enumerate(evidence)
            if row["answer_state"] == "unanswerable"
            and int(row["hop_count"]) == hop
        ]
        strata[str(hop)] = {
            "cases": len(indices),
            "rejection_rate": round(
                float(np.mean([not candidate_decisions[index] for index in indices])),
                6,
            ),
        }
    invalid_count = sum(bool(row["invalid_feature_output"]) for row in evidence)
    candidate = methods[candidate_name]
    checks = {
        "exact_cases_equals_800": len(evidence) == 800,
        "exact_answer_state_balance": int(np.sum(labels == 1.0)) == 400,
        "schema_exclusion_rate_at_most_0_01": float(
            sampling["schema_exclusion_rate"]
        )
        <= 0.01,
        "selected_prior_or_calibration_source_overlap_equals_0": int(
            sampling["selected_excluded_source_commitment_overlap"]
        )
        == 0,
        "selected_squad2_exact_question_overlap_equals_0": int(
            sampling["selected_squad2_exact_question_overlap"]
        )
        == 0,
        "invalid_feature_output_rate_equals_0": invalid_count == 0,
        "candidate_balanced_accuracy_at_least_0_72": candidate[
            "balanced_accuracy"
        ]
        >= 0.72,
        "candidate_answerable_pass_rate_at_least_0_60": candidate[
            "answerable_pass_rate"
        ]
        >= 0.60,
        "candidate_unanswerable_rejection_rate_at_least_0_80": candidate[
            "unanswerable_rejection_rate"
        ]
        >= 0.80,
        "candidate_minus_strongest_fair_baseline_at_least_0_05": paired["point"]
        >= 0.05,
        "strongest_baseline_paired_correctness_ci_low_above_0": paired["ci_low"]
        > 0.0,
        "candidate_minus_additive_control_at_least_0_02": interaction_paired[
            "point"
        ]
        >= 0.02,
        "additive_control_paired_correctness_ci_low_above_0": interaction_paired[
            "ci_low"
        ]
        > 0.0,
        "every_observed_unanswerable_hop_stratum_rejection_rate_at_least_0_75": all(
            value["rejection_rate"] >= 0.75 for value in strata.values()
        ),
    }
    passed = all(checks.values())
    if stage == "development":
        status = (
            "MUSIQUE_V69_MONOTONE_INTERACTION_DEVELOPMENT_FEASIBILITY_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "MUSIQUE_V69_MONOTONE_INTERACTION_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "MUSIQUE_V69_ORACLE_PLAN_MONOTONE_INTERACTION_COMPONENT_FEASIBILITY_ESTABLISHED"
            if passed
            else "MUSIQUE_V69_MONOTONE_INTERACTION_CONFIRMATION_NOT_ESTABLISHED"
        )
    return {
        "schema_version": "frc-musique-v69-stage-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "stage": stage,
            "cases": len(evidence),
            "answer_state_counts": {
                "answerable": int(np.sum(labels == 1.0)),
                "unanswerable": int(np.sum(labels == 0.0)),
            },
            "sampling": sampling,
            "structural_census": structural_census,
            "source_artifacts": source_artifacts,
            "automatic_decomposer_result": False,
            "strict_independent_model_training_confirmation": False,
            "official_musique_leaderboard_result": False,
        },
        "analysis": {
            "methods": methods,
            "strongest_fair_baseline": {
                "name": baseline_name,
                **methods[baseline_name],
            },
            "paired_correctness_delta": paired,
            "interaction_mechanism_delta_vs_additive": interaction_paired,
            "unanswerable_hop_strata": strata,
            "invalid_feature_output_count": invalid_count,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": stage == "development" and passed,
                "selector_or_retrieval_scoring_part_of_v69": False,
                "reuse_failed_stage_for_basis_interaction_model_threshold_rule_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_evidence(path: Path, evidence: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


def write_calibration_report(
    calibration: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    result_path: Path,
    report_path: Path,
    evidence_path: Path,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "frc-musique-v69-calibration-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": metadata,
        "analysis": calibration,
        "outcome": {
            "status": "MUSIQUE_V69_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT",
            "development_open_authorized": True,
            "calibration_alone_authorizes_adoption": False,
            "gate_2": "NO-GO/SHADOW",
        },
    }
    write_json(result_path, payload)
    write_evidence(evidence_path, evidence)
    lines = [
        "# MuSiQue v69 单调交互链校准",
        "",
        f"- 案例：{len(evidence)}",
        "- 状态：`MUSIQUE_V69_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`",
        "",
        "| 方法 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in calibration.items():
        crossfit = value["crossfit"]
        lines.append(
            f"| `{name}` | {crossfit['balanced_accuracy']:.6f} | "
            f"{crossfit['answerable_pass_rate']:.6f} | "
            f"{crossfit['unanswerable_rejection_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "校准只冻结基函数、交互、模型与阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_stage_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    result_path: Path,
    report_path: Path,
    evidence_path: Path,
) -> None:
    write_json(result_path, report)
    write_evidence(evidence_path, evidence)
    analysis = report["analysis"]
    lines = [
        f"# MuSiQue v69 单调交互链支持门（{report['metadata']['stage']}）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 案例：{report['metadata']['cases']}",
        f"- 最强公平基线：`{analysis['strongest_fair_baseline']['name']}`",
        "",
        "| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in analysis["methods"].items():
        lines.append(
            f"| `{name}` | {value['balanced_accuracy']:.6f} | "
            f"{value['answerable_pass_rate']:.6f} | "
            f"{value['unanswerable_rejection_rate']:.6f} |"
        )
    paired = analysis["paired_correctness_delta"]
    interaction = analysis["interaction_mechanism_delta_vs_additive"]
    lines.extend(
        [
            "",
            f"候选相对最强公平基线差值 {paired['point']:+.6f}，95% CI "
            f"[{paired['ci_low']:+.6f},{paired['ci_high']:+.6f}]。",
            f"候选相对无交互分段控制差值 {interaction['point']:+.6f}，95% CI "
            f"[{interaction['ci_low']:+.6f},{interaction['ci_high']:+.6f}]。",
            "",
            "该实验只检验 oracle 计划单调交互支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
