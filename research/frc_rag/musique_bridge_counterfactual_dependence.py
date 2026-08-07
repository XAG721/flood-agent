"""Bridge-counterfactual MuSiQue chain-support experiment (v71)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag import musique_paragraph_competition_support as v70

EXPERIMENT_ID = "FRC-MUSIQUE-BRIDGE-COUNTERFACTUAL-DEPENDENCE-V71"
PROTOCOL_SCHEMA = "frc-musique-bridge-counterfactual-dependence-protocol-v71"
SOURCE_SCHEMA = (
    "frc-musique-bridge-counterfactual-dependence-source-registration-v71"
)
FIXED_V66_THRESHOLD = v70.FIXED_V66_THRESHOLD
FAIL_CLOSED_RAW_SCORE = v70.FAIL_CLOSED_RAW_SCORE
STAGE_SALTS = {
    "calibration": "FRC-MUSIQUE-V71-CALIBRATION|",
    "development": "FRC-MUSIQUE-V71-DEVELOPMENT|",
    "confirmation": "FRC-MUSIQUE-V71-CONFIRMATION|",
}
STAGE_PREFIXES = {
    "calibration": "m71k",
    "development": "m71d",
    "confirmation": "m71c",
}
STAGE_TARGET_PER_GROUP = {
    "calibration": 800,
    "development": 400,
    "confirmation": 400,
}
STAGE_BOOTSTRAP_SEEDS = {
    "development": 20260903,
    "confirmation": 20260904,
}
PAIR_ASSIGNMENT_SALT = "FRC-MUSIQUE-V71-PAIR-ASSIGNMENT|"
CROSSFIT_SALT = "FRC-MUSIQUE-V71-CROSSFIT|"
UNKNOWN_BRIDGE_SENTINEL = "unknown bridge entity"
MARGIN_DROP_CLIP_MAXIMUM = 10.0
BASE_FEATURE_NAMES = v70.BASE_FEATURE_NAMES
ADDITIVE_FEATURE_NAMES = v70.ADDITIVE_FEATURE_NAMES
INTERACTION_FEATURE_NAMES = v70.INTERACTION_FEATURE_NAMES
COMPETITION_FEATURE_NAMES = v70.COMPETITION_FEATURE_NAMES
PARAGRAPH_COMPETITION_FEATURE_NAMES = v70.CANDIDATE_FEATURE_NAMES
COUNTERFACTUAL_FEATURE_NAMES = (
    "bridge_margin_drop_min",
    "bridge_margin_drop_mean",
    "bridge_paragraph_change_min",
    "bridge_paragraph_change_mean",
    "bridge_span_change_min",
    "bridge_span_change_mean",
)
CANDIDATE_FEATURE_NAMES = BASE_FEATURE_NAMES + COUNTERFACTUAL_FEATURE_NAMES
ALL_FEATURE_NAMES = v70.ALL_FEATURE_NAMES + COUNTERFACTUAL_FEATURE_NAMES
UNCONSTRAINED_FEATURE_NAMES = {"hop_count_is_3", "hop_count_is_4"}
NONNEGATIVE_FEATURE_NAMES = tuple(
    name for name in ALL_FEATURE_NAMES if name not in UNCONSTRAINED_FEATURE_NAMES
)
OPTIMIZER_ITERATIONS = 5000
OPTIMIZER_LEARNING_RATE = 0.1
OPTIMIZER_L2_WEIGHT = 0.01

sha256 = v70.sha256
nested_keys = v70.nested_keys
extract_squad2_questions = v70.extract_squad2_questions
load_source_commitments = v70.load_source_commitments


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("MuSiQue v71 protocol schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v71 experiment id changed")
    for stage, target in STAGE_TARGET_PER_GROUP.items():
        if int(value["frozen_stages"][stage]["target_per_answer_state"]) != target:
            raise ValueError(f"MuSiQue v71 {stage} size changed")
    mechanism = value["mechanism_hypothesis"]
    if (
        mechanism["fixed_unknown_entity_sentinel"] != UNKNOWN_BRIDGE_SENTINEL
        or float(mechanism["positive_margin_drop_clip_maximum"])
        != MARGIN_DROP_CLIP_MAXIMUM
    ):
        raise ValueError("MuSiQue v71 counterfactual construction changed")
    contract = value["counterfactual_feature_contract"]
    if (
        tuple(contract["fixed_order"]) != COUNTERFACTUAL_FEATURE_NAMES
        or int(contract["feature_count"]) != len(COUNTERFACTUAL_FEATURE_NAMES)
        or int(contract["candidate_feature_count"])
        != len(CANDIDATE_FEATURE_NAMES)
    ):
        raise ValueError("MuSiQue v71 feature contract changed")
    model = value["model_contract"]
    if (
        int(model["iterations"]) != OPTIMIZER_ITERATIONS
        or float(model["learning_rate"]) != OPTIMIZER_LEARNING_RATE
        or float(model["l2_weight"]) != OPTIMIZER_L2_WEIGHT
        or int(model["total_learned_scalars_across_all_reported_methods"])
        != 118
    ):
        raise ValueError("MuSiQue v71 optimizer or capacity changed")
    if value["sampling"]["expected_prior_source_commitment_union_count"] != 9700:
        raise ValueError("MuSiQue v71 prior exclusion boundary changed")
    if value["stopping_and_outcomes"]["gate_2"] != "NO-GO/SHADOW":
        raise ValueError("MuSiQue v71 Gate 2 boundary changed")
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
        raise ValueError("MuSiQue v71 source registration schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v71 source registration experiment changed")
    if len(prior_paths) != 11:
        raise ValueError("MuSiQue v71 requires exactly eleven prior maps")
    expected = {
        train_path: protocol["sources"]["calibration_and_development_sha256"],
        dev_path: protocol["sources"]["optional_confirmation_sha256"],
        squad2_path: value["leakage_guard_source"]["sha256"],
    }
    registered_prior = [
        item
        for item in value["prior_exclusion_sources"].values()
        if isinstance(item, dict)
    ]
    if len(registered_prior) != 11:
        raise ValueError("MuSiQue v71 prior source registration changed")
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
            raise ValueError(f"MuSiQue v71 registered source changed: {source_path}")
    if (
        int(
            value["prior_exclusion_sources"][
                "expected_union_source_commitments"
            ]
        )
        != 9700
    ):
        raise ValueError("MuSiQue v71 prior commitment count changed")
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
        raise ValueError(f"Unsupported MuSiQue v71 stage: {stage}")
    expected_split = "dev" if stage == "confirmation" else "train"
    if source_split != expected_split:
        raise ValueError("MuSiQue v71 stage/source split mismatch")
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
        valid, reason = v70.v66._valid_row(row)
        if not valid:
            schema_reasons[reason] = schema_reasons.get(reason, 0) + 1
            continue
        if v70.v66._has_squad_overlap(row, squad_questions):
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
        raise ValueError("MuSiQue v71 has insufficient eligible balanced cases")
    selected_items = sorted(
        [item for values in retained.values() for item in values],
        key=lambda item: (item[0], item[1]),
    )
    selected = [item[2] for item in selected_items]
    selected_ids = {str(row["id"]) for row in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("MuSiQue v71 sample contains duplicate source ids")
    if any(_hash(value) in excluded_source_commitments for value in selected_ids):
        raise ValueError("MuSiQue v71 sample overlaps an excluded commitment")
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
    prepared, maps, direct_inputs, census = v70.prepare_blind_stage(
        selected, stage=stage
    )
    old_prefix = v70.STAGE_PREFIXES[stage]
    new_prefix = STAGE_PREFIXES[stage]
    for row in prepared:
        row["schema_version"] = "frc-musique-v71-prepared-blind-v1"
        row["id"] = str(row["id"]).replace(old_prefix + "-", new_prefix + "-", 1)
    for row in maps:
        row["schema_version"] = "frc-musique-v71-candidate-map-v1"
        row["id"] = str(row["id"]).replace(old_prefix + "-", new_prefix + "-", 1)
    for row in direct_inputs:
        row["schema_version"] = "frc-musique-v71-direct-qa-input-v1"
        row["id"] = str(row["id"]).replace(old_prefix + "-", new_prefix + "-", 1)
        row["case_id"] = str(row["case_id"]).replace(
            old_prefix + "-", new_prefix + "-", 1
        )
    return prepared, maps, direct_inputs, census


class LocalRobertaQASupportVerifier(v70.LocalRobertaQASupportVerifier):
    def verify(
        self, rows: Sequence[dict[str, Any]], *, cache_key: str
    ) -> list[dict[str, Any]]:
        result = super().verify(rows, cache_key=cache_key)
        for row in result:
            row["schema_version"] = "frc-musique-v71-raw-qa-v1"
        return result


def aggregate_qa_rows(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> list[dict[str, Any]]:
    result = v70.aggregate_qa_rows(qa_inputs, decisions, cache_key=cache_key)
    for row in result:
        row["schema_version"] = "frc-musique-v71-aggregated-qa-v1"
    return result


def initial_chain_states(
    prepared: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return v70.initial_chain_states(prepared)


def build_hop_qa_inputs(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    hop_index: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    inputs, blocked = v70.build_hop_qa_inputs(
        prepared, states, hop_index=hop_index
    )
    for row in inputs:
        row["schema_version"] = "frc-musique-v71-hop-qa-input-v1"
    return inputs, blocked


def apply_hop_results(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    aggregated: Sequence[dict[str, Any]],
    blocked: dict[str, str],
    *,
    hop_index: int,
) -> list[dict[str, Any]]:
    result = v70.apply_hop_results(
        prepared, states, aggregated, blocked, hop_index=hop_index
    )
    for row in result:
        row["schema_version"] = "frc-musique-v71-hop-feature-decision-v1"
    return result


def build_counterfactual_qa_inputs(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in prepared:
        case_id = str(row["id"])
        state = states[case_id]
        expected_hops = len(row["oracle_hop_templates"])
        if not state["alive"] or int(state["completed_hops"]) != expected_hops:
            continue
        for hop_index in range(2, expected_hops + 1):
            template = str(row["oracle_hop_templates"][hop_index - 1])
            question, substitutions = re.subn(
                r"#\d+", UNKNOWN_BRIDGE_SENTINEL, template
            )
            transition_id = f"{case_id}::bridge_cf::h{hop_index}"
            for context in row["contexts"]:
                paragraph_idx = int(context["paragraph_idx"])
                result.append(
                    {
                        "schema_version": "frc-musique-v71-counterfactual-qa-input-v1",
                        "id": f"{transition_id}::p{paragraph_idx}",
                        "case_id": transition_id,
                        "root_case_id": case_id,
                        "hop_index": hop_index,
                        "placeholder_substitutions": substitutions,
                        "paragraph_idx": paragraph_idx,
                        "question": question,
                        "context": str(context["context"]),
                        "gold_fields_visible_to_verifier": False,
                    }
                )
    forbidden = {
        "answerable",
        "answer",
        "answer_aliases",
        "paragraph_support_idx",
        "is_supporting",
    }
    if forbidden & nested_keys(result):
        raise ValueError("MuSiQue v71 counterfactual inputs expose gold")
    return result


def aggregate_counterfactual_qa_rows(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> list[dict[str, Any]]:
    aggregated = aggregate_qa_rows(qa_inputs, decisions, cache_key=cache_key)
    metadata: dict[str, tuple[str, int]] = {}
    for row in qa_inputs:
        transition_id = str(row["case_id"])
        value = (str(row["root_case_id"]), int(row["hop_index"]))
        if transition_id in metadata and metadata[transition_id] != value:
            raise ValueError("MuSiQue v71 counterfactual transition changed")
        metadata[transition_id] = value
    result: list[dict[str, Any]] = []
    for row in aggregated:
        transition_id = str(row["case_id"])
        root_case_id, hop_index = metadata[transition_id]
        result.append(
            {
                **row,
                "schema_version": "frc-musique-v71-counterfactual-decision-v1",
                "transition_id": transition_id,
                "root_case_id": root_case_id,
                "hop_index": hop_index,
            }
        )
    return result


def _normalize_span(value: str) -> str:
    return re.sub(
        r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", value.casefold())
    ).strip()


def finalize_feature_decisions(
    prepared: Sequence[dict[str, Any]],
    direct_decisions: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    hop_decisions: Sequence[dict[str, Any]],
    counterfactual_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    factual = v70.finalize_feature_decisions(
        prepared, direct_decisions, states, hop_decisions
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for decision in counterfactual_decisions:
        grouped.setdefault(str(decision["root_case_id"]), []).append(decision)
    result: list[dict[str, Any]] = []
    for row, base in zip(prepared, factual, strict=True):
        case_id = str(row["id"])
        state = states[case_id]
        expected_transitions = len(row["oracle_hop_templates"]) - 1
        transitions = sorted(
            grouped.get(case_id, []), key=lambda item: int(item["hop_index"])
        )
        complete = bool(
            base["feature_complete"]
            and len(transitions) == expected_transitions
            and [int(item["hop_index"]) for item in transitions]
            == list(range(2, expected_transitions + 2))
        )
        summaries: list[dict[str, Any]] = []
        if complete:
            margin_drops: list[float] = []
            paragraph_changes: list[float] = []
            span_changes: list[float] = []
            for decision in transitions:
                hop_index = int(decision["hop_index"])
                factual_margin = float(state["margins"][hop_index - 1])
                factual_paragraph = int(
                    state["selected_paragraphs"][hop_index - 1]
                )
                factual_span = str(state["predictions"][hop_index])
                ablated_margin = decision.get("score_margin")
                ablated_span = str(decision.get("predicted_span", "")).strip()
                ablated_valid = bool(
                    ablated_margin is not None
                    and math.isfinite(float(ablated_margin))
                    and ablated_span
                    and not decision["invalid_fail_closed_used"]
                )
                if ablated_valid:
                    margin_drop = min(
                        max(factual_margin - float(ablated_margin), 0.0),
                        MARGIN_DROP_CLIP_MAXIMUM,
                    ) / MARGIN_DROP_CLIP_MAXIMUM
                    paragraph_change = float(
                        factual_paragraph
                        != int(decision["selected_paragraph_idx"])
                    )
                    span_change = float(
                        _normalize_span(factual_span)
                        != _normalize_span(ablated_span)
                    )
                else:
                    margin_drop = 1.0
                    paragraph_change = 1.0
                    span_change = 1.0
                margin_drops.append(margin_drop)
                paragraph_changes.append(paragraph_change)
                span_changes.append(span_change)
                summaries.append(
                    {
                        "hop_index": hop_index,
                        "counterfactual_valid_span": ablated_valid,
                        "counterfactual_score_margin": float(ablated_margin)
                        if ablated_valid
                        else None,
                        "counterfactual_selected_paragraph_idx": int(
                            decision["selected_paragraph_idx"]
                        )
                        if ablated_valid
                        else None,
                        "counterfactual_predicted_span_sha256": str(
                            decision["predicted_span_sha256"]
                        ),
                        "positive_margin_drop_scaled": margin_drop,
                        "selected_paragraph_changed": paragraph_change,
                        "normalized_span_changed": span_change,
                    }
                )
            counterfactual_features = {
                "bridge_margin_drop_min": min(margin_drops),
                "bridge_margin_drop_mean": float(np.mean(margin_drops)),
                "bridge_paragraph_change_min": min(paragraph_changes),
                "bridge_paragraph_change_mean": float(
                    np.mean(paragraph_changes)
                ),
                "bridge_span_change_min": min(span_changes),
                "bridge_span_change_mean": float(np.mean(span_changes)),
            }
            features = {**base["features"], **counterfactual_features}
        else:
            features = {name: None for name in ALL_FEATURE_NAMES}
        if tuple(features) != ALL_FEATURE_NAMES:
            raise ValueError("MuSiQue v71 full feature order changed")
        result.append(
            {
                "schema_version": "frc-musique-v71-feature-decision-v1",
                "case_id": case_id,
                "hop_count": base["hop_count"],
                "executed_hops": base["executed_hops"],
                "valid_hops": base["valid_hops"],
                "counterfactual_transitions": len(transitions),
                "feature_complete": complete,
                "direct_score_margin": base["direct_score_margin"]
                if complete
                else None,
                "chain_bottleneck_score": base["chain_bottleneck_score"]
                if complete
                else None,
                "chain_mean_score": base["chain_mean_score"]
                if complete
                else None,
                "features": features,
                "feature_sha256": _hash(
                    json.dumps(features, sort_keys=True, separators=(",", ":"))
                ),
                "factual_feature_sha256": base["feature_sha256"],
                "counterfactual_decision_sha256": _hash(
                    json.dumps(summaries, sort_keys=True, separators=(",", ":"))
                ),
                "invalid_fail_closed_used": bool(
                    base["invalid_fail_closed_used"] or not complete
                ),
                "hop_decision_sha256": base["hop_decision_sha256"],
            }
        )
    return result


def _fail_closed_features(hop_count: int) -> dict[str, float]:
    prior = v70._fail_closed_features(hop_count)
    counterfactual = {name: 0.0 for name in COUNTERFACTUAL_FEATURE_NAMES}
    return {**prior, **counterfactual}


def build_gold_evidence(
    selected: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    feature_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(selected) == len(prepared) == len(feature_decisions)):
        raise ValueError("MuSiQue v71 evidence inputs differ in length")
    evidence: list[dict[str, Any]] = []
    for source, blind, feature in zip(
        selected, prepared, feature_decisions, strict=True
    ):
        case_id = str(blind["id"])
        if case_id != str(feature["case_id"]):
            raise ValueError("MuSiQue v71 evidence order changed")
        hop_count = len(source["question_decomposition"])
        if feature["feature_complete"]:
            scoring_features = feature["features"]
            direct_score = feature["direct_score_margin"]
            chain_score = feature["chain_bottleneck_score"]
            chain_mean = feature["chain_mean_score"]
        else:
            scoring_features = _fail_closed_features(hop_count)
            direct_score = FAIL_CLOSED_RAW_SCORE
            chain_score = FAIL_CLOSED_RAW_SCORE
            chain_mean = FAIL_CLOSED_RAW_SCORE
        evidence.append(
            {
                "schema_version": "frc-musique-v71-feature-evidence-v1",
                "case_id": case_id,
                "answer_state": "answerable"
                if source["answerable"]
                else "unanswerable",
                "hop_count": hop_count,
                "paragraph_count": len(source["paragraphs"]),
                "direct_score_margin": direct_score,
                "chain_bottleneck_score": chain_score,
                "chain_mean_score": chain_mean,
                "features": scoring_features,
                "feature_sha256": _hash(
                    json.dumps(
                        scoring_features, sort_keys=True, separators=(",", ":")
                    )
                ),
                "blind_feature_sha256": feature["feature_sha256"],
                "feature_complete": feature["feature_complete"],
                "chain_executed_hops": feature["executed_hops"],
                "chain_valid_hops": feature["valid_hops"],
                "counterfactual_transitions": feature[
                    "counterfactual_transitions"
                ],
                "counterfactual_decision_sha256": feature[
                    "counterfactual_decision_sha256"
                ],
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
        raise ValueError("MuSiQue v71 feature matrix is non-finite")
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


score_metrics = v70.score_metrics
select_threshold_from_scores = v70.select_threshold_from_scores
paired_correctness_interval = v70.paired_correctness_interval


def _raw_scores(evidence: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in evidence], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"MuSiQue v71 non-finite raw score: {field}")
    return values


def _crossfit_metrics(
    held_labels: Sequence[float],
    held_scores: Sequence[float],
    held_thresholds: Sequence[float],
    fold_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray(held_labels, dtype=np.float64)
    scores = np.asarray(held_scores, dtype=np.float64)
    thresholds = np.asarray(held_thresholds, dtype=np.float64)
    decisions = scores > thresholds
    answer_mask = labels == 1.0
    noanswer_mask = labels == 0.0
    answer_pass = float(np.mean(decisions[answer_mask]))
    rejection = float(np.mean(~decisions[noanswer_mask]))
    return {
        "folds": list(fold_rows),
        "held_out_cases": len(labels),
        "answerable_pass_rate": round(answer_pass, 6),
        "unanswerable_rejection_rate": round(rejection, 6),
        "balanced_accuracy": round((answer_pass + rejection) / 2.0, 6),
    }


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
    return _crossfit_metrics(held_labels, held_scores, held_thresholds, fold_rows)


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
    return _crossfit_metrics(held_labels, held_scores, held_thresholds, fold_rows)


MODEL_FEATURE_CONTRACTS = {
    "linear_nine_signal_control": BASE_FEATURE_NAMES,
    "monotone_additive_spline_control": ADDITIVE_FEATURE_NAMES,
    "monotone_interaction_control": INTERACTION_FEATURE_NAMES,
    "paragraph_competition_control": PARAGRAPH_COMPETITION_FEATURE_NAMES,
    "bridge_counterfactual_only_control": COUNTERFACTUAL_FEATURE_NAMES,
    "bridge_counterfactual_candidate": CANDIDATE_FEATURE_NAMES,
}


def fit_calibration(evidence: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = _labels(evidence)
    direct_scores = _raw_scores(evidence, "direct_score_margin")
    chain_scores = _raw_scores(evidence, "chain_bottleneck_score")
    models = {
        name: fit_projected_logistic(evidence, feature_names=feature_names)
        for name, feature_names in MODEL_FEATURE_CONTRACTS.items()
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
    for name, feature_names in MODEL_FEATURE_CONTRACTS.items():
        scores = predict_projected_logistic(evidence, models[name])
        result[name] = {
            "model": models[name],
            "pooled": select_threshold_from_scores(labels, scores),
            "crossfit": _crossfit_model(evidence, feature_names=feature_names),
        }
    return result


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
        raise ValueError("MuSiQue v71 evaluation stage changed")
    labels = _labels(evidence)
    method_scores = {
        "calibrated_direct_composed_question": _raw_scores(
            evidence, "direct_score_margin"
        ),
        "fixed_v66_full_chain": _raw_scores(evidence, "chain_bottleneck_score"),
        "calibrated_chain_bottleneck": _raw_scores(
            evidence, "chain_bottleneck_score"
        ),
        **{
            name: predict_projected_logistic(evidence, calibration[name]["model"])
            for name in MODEL_FEATURE_CONTRACTS
        },
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
        **{
            name: float(calibration[name]["pooled"]["threshold_exact"])
            for name in MODEL_FEATURE_CONTRACTS
        },
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
        "monotone_interaction_control",
        "paragraph_competition_control",
        "bridge_counterfactual_only_control",
    )
    prior_control_order = (
        "linear_nine_signal_control",
        "monotone_additive_spline_control",
        "monotone_interaction_control",
        "paragraph_competition_control",
    )

    def _strongest(names: Sequence[str]) -> str:
        return max(
            names,
            key=lambda name: (
                methods[name]["balanced_accuracy"],
                -names.index(name),
            ),
        )

    baseline_name = _strongest(comparator_order)
    prior_name = _strongest(prior_control_order)
    candidate_name = "bridge_counterfactual_candidate"

    def _paired(baseline: str, seed_offset: int) -> dict[str, Any]:
        return paired_correctness_interval(
            labels,
            method_scores[candidate_name],
            method_thresholds[candidate_name],
            method_scores[baseline],
            method_thresholds[baseline],
            seed=STAGE_BOOTSTRAP_SEEDS[stage] + seed_offset,
        )

    paired = _paired(baseline_name, 0)
    prior_paired = _paired(prior_name, 1000)
    counterfactual_only_paired = _paired(
        "bridge_counterfactual_only_control", 2000
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
        "candidate_minus_strongest_prior_feature_control_at_least_0_02": prior_paired[
            "point"
        ]
        >= 0.02,
        "prior_feature_control_paired_correctness_ci_low_above_0": prior_paired[
            "ci_low"
        ]
        > 0.0,
        "candidate_minus_counterfactual_only_control_at_least_0_02": counterfactual_only_paired[
            "point"
        ]
        >= 0.02,
        "counterfactual_only_control_paired_correctness_ci_low_above_0": counterfactual_only_paired[
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
            "MUSIQUE_V71_BRIDGE_COUNTERFACTUAL_DEVELOPMENT_FEASIBILITY_"
            "ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "MUSIQUE_V71_BRIDGE_COUNTERFACTUAL_DEVELOPMENT_SUPPORT_NOT_"
            "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "MUSIQUE_V71_ORACLE_PLAN_BRIDGE_COUNTERFACTUAL_COMPONENT_"
            "FEASIBILITY_ESTABLISHED"
            if passed
            else "MUSIQUE_V71_BRIDGE_COUNTERFACTUAL_CONFIRMATION_NOT_ESTABLISHED"
        )
    return {
        "schema_version": "frc-musique-v71-stage-result-v1",
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
            "official_musique_leaderboard_result": False,
            "automatic_decomposer_result": False,
            "strict_independent_model_training_confirmation": False,
        },
        "analysis": {
            "methods": methods,
            "strongest_fair_baseline": {
                "name": baseline_name,
                **methods[baseline_name],
            },
            "strongest_prior_feature_control": {
                "name": prior_name,
                **methods[prior_name],
            },
            "paired_correctness_delta": paired,
            "incremental_delta_vs_strongest_prior_feature_control": prior_paired,
            "incremental_delta_vs_counterfactual_only_control": counterfactual_only_paired,
            "unanswerable_hop_strata": strata,
            "invalid_feature_output_count": invalid_count,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": stage == "development" and passed,
                "selector_or_retrieval_scoring_part_of_v71": False,
                "reuse_failed_stage_for_feature_model_threshold_rule_gate_or_selection": False,
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
    status = "MUSIQUE_V71_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT"
    payload = {
        "schema_version": "frc-musique-v71-calibration-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": metadata,
        "analysis": calibration,
        "outcome": {
            "status": status,
            "development_open_authorized": True,
            "calibration_alone_authorizes_adoption": False,
            "gate_2": "NO-GO/SHADOW",
        },
    }
    write_json(result_path, payload)
    write_evidence(evidence_path, evidence)
    lines = [
        "# MuSiQue v71 桥接实体反事实依赖校准",
        "",
        f"- 案例：{len(evidence)}",
        f"- 状态：`{status}`",
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
            "校准只冻结模型与阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。",
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
        f"# MuSiQue v71 桥接实体反事实支持门（{report['metadata']['stage']}）",
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
    prior = analysis["incremental_delta_vs_strongest_prior_feature_control"]
    only = analysis["incremental_delta_vs_counterfactual_only_control"]
    lines.extend(
        [
            "",
            f"候选相对最强公平基线差值 {paired['point']:+.6f}，95% CI "
            f"[{paired['ci_low']:+.6f},{paired['ci_high']:+.6f}]。",
            f"候选相对最强既有特征控制差值 {prior['point']:+.6f}，95% CI "
            f"[{prior['ci_low']:+.6f},{prior['ci_high']:+.6f}]。",
            f"候选相对反事实信号单独控制差值 {only['point']:+.6f}，95% CI "
            f"[{only['ci_low']:+.6f},{only['ci_high']:+.6f}]。",
            "",
            "该实验只检验 oracle 计划桥接实体反事实支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
