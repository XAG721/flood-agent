"""In-domain rival-bridge MuSiQue support experiment (v72)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag import musique_bridge_counterfactual_dependence as v71

EXPERIMENT_ID = "FRC-MUSIQUE-IN-DOMAIN-RIVAL-BRIDGE-V72"
PROTOCOL_SCHEMA = "frc-musique-in-domain-rival-bridge-protocol-v72"
SOURCE_SCHEMA = "frc-musique-in-domain-rival-bridge-source-registration-v72"
FIXED_V66_THRESHOLD = v71.FIXED_V66_THRESHOLD
FAIL_CLOSED_RAW_SCORE = v71.FAIL_CLOSED_RAW_SCORE
STAGE_SALTS = {
    "calibration": "FRC-MUSIQUE-V72-CALIBRATION|",
    "development": "FRC-MUSIQUE-V72-DEVELOPMENT|",
    "confirmation": "FRC-MUSIQUE-V72-CONFIRMATION|",
}
STAGE_PREFIXES = {
    "calibration": "m72k",
    "development": "m72d",
    "confirmation": "m72c",
}
STAGE_TARGET_PER_GROUP = {
    "calibration": 800,
    "development": 400,
    "confirmation": 400,
}
STAGE_BOOTSTRAP_SEEDS = {
    "development": 20260906,
    "confirmation": 20260907,
}
PAIR_ASSIGNMENT_SALT = "FRC-MUSIQUE-V72-PAIR-ASSIGNMENT|"
CROSSFIT_SALT = "FRC-MUSIQUE-V72-CROSSFIT|"
UNKNOWN_BRIDGE_SENTINEL = v71.UNKNOWN_BRIDGE_SENTINEL
MARGIN_ADVANTAGE_CLIP_MAXIMUM = 10.0
BASE_FEATURE_NAMES = v71.BASE_FEATURE_NAMES
ADDITIVE_FEATURE_NAMES = v71.ADDITIVE_FEATURE_NAMES
INTERACTION_FEATURE_NAMES = v71.INTERACTION_FEATURE_NAMES
COMPETITION_FEATURE_NAMES = v71.COMPETITION_FEATURE_NAMES
PARAGRAPH_COMPETITION_FEATURE_NAMES = v71.PARAGRAPH_COMPETITION_FEATURE_NAMES
FIXED_SENTINEL_FEATURE_NAMES = v71.CANDIDATE_FEATURE_NAMES
SENTINEL_ONLY_FEATURE_NAMES = v71.COUNTERFACTUAL_FEATURE_NAMES
RIVAL_FEATURE_NAMES = (
    "rival_dependency_coverage_fraction",
    "rival_margin_advantage_min",
    "rival_margin_advantage_mean",
    "rival_factual_win_fraction",
    "rival_paragraph_change_fraction",
    "rival_span_change_fraction",
)
CANDIDATE_FEATURE_NAMES = BASE_FEATURE_NAMES + RIVAL_FEATURE_NAMES
ALL_FEATURE_NAMES = v71.ALL_FEATURE_NAMES + RIVAL_FEATURE_NAMES
UNCONSTRAINED_FEATURE_NAMES = {"hop_count_is_3", "hop_count_is_4"}
NONNEGATIVE_FEATURE_NAMES = tuple(
    name for name in ALL_FEATURE_NAMES if name not in UNCONSTRAINED_FEATURE_NAMES
)
OPTIMIZER_ITERATIONS = 5000
OPTIMIZER_LEARNING_RATE = 0.1
OPTIMIZER_L2_WEIGHT = 0.01

sha256 = v71.sha256
nested_keys = v71.nested_keys
extract_squad2_questions = v71.extract_squad2_questions
load_source_commitments = v71.load_source_commitments
score_metrics = v71.score_metrics
select_threshold_from_scores = v71.select_threshold_from_scores
paired_correctness_interval = v71.paired_correctness_interval


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_span(value: str) -> str:
    return re.sub(
        r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", value.casefold())
    ).strip()


def validate_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("MuSiQue v72 protocol schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v72 experiment id changed")
    for stage, target in STAGE_TARGET_PER_GROUP.items():
        if int(value["frozen_stages"][stage]["target_per_answer_state"]) != target:
            raise ValueError(f"MuSiQue v72 {stage} size changed")
    if (
        float(
            value["mechanism_hypothesis"][
                "positive_margin_advantage_clip_maximum"
            ]
        )
        != MARGIN_ADVANTAGE_CLIP_MAXIMUM
    ):
        raise ValueError("MuSiQue v72 rival margin construction changed")
    contract = value["feature_contract"]
    if (
        tuple(contract["fixed_order"]) != RIVAL_FEATURE_NAMES
        or int(contract["feature_count"]) != len(RIVAL_FEATURE_NAMES)
        or int(contract["candidate_feature_count"])
        != len(CANDIDATE_FEATURE_NAMES)
    ):
        raise ValueError("MuSiQue v72 feature contract changed")
    model = value["model_contract"]
    if (
        int(model["iterations"]) != OPTIMIZER_ITERATIONS
        or float(model["learning_rate"]) != OPTIMIZER_LEARNING_RATE
        or float(model["l2_weight"]) != OPTIMIZER_L2_WEIGHT
        or int(model["total_learned_scalars_across_all_reported_methods"])
        != 135
    ):
        raise ValueError("MuSiQue v72 optimizer or capacity changed")
    if value["sampling"]["expected_prior_source_commitment_union_count"] != 12100:
        raise ValueError("MuSiQue v72 prior exclusion boundary changed")
    if value["stopping_and_outcomes"]["gate_2"] != "NO-GO/SHADOW":
        raise ValueError("MuSiQue v72 Gate 2 boundary changed")
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
        raise ValueError("MuSiQue v72 source registration schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v72 source registration experiment changed")
    if len(prior_paths) != 13:
        raise ValueError("MuSiQue v72 requires exactly thirteen prior maps")
    registered_prior = [
        item
        for item in value["prior_exclusion_sources"].values()
        if isinstance(item, dict)
    ]
    if len(registered_prior) != 13:
        raise ValueError("MuSiQue v72 prior source registration changed")
    expected = {
        train_path: protocol["sources"]["calibration_and_development_sha256"],
        dev_path: protocol["sources"]["optional_confirmation_sha256"],
        squad2_path: value["leakage_guard_source"]["sha256"],
        **{
            prior_path: registered["sha256"]
            for prior_path, registered in zip(
                prior_paths, registered_prior, strict=True
            )
        },
    }
    for source_path, expected_hash in expected.items():
        if not source_path.exists() or sha256(source_path) != expected_hash:
            raise ValueError(f"MuSiQue v72 registered source changed: {source_path}")
    if (
        int(value["prior_exclusion_sources"]["expected_union_source_commitments"])
        != 12100
    ):
        raise ValueError("MuSiQue v72 prior commitment count changed")
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
        raise ValueError(f"Unsupported MuSiQue v72 stage: {stage}")
    expected_split = "dev" if stage == "confirmation" else "train"
    if source_split != expected_split:
        raise ValueError("MuSiQue v72 stage/source split mismatch")
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
        valid, reason = v71.v70.v66._valid_row(row)
        if not valid:
            schema_reasons[reason] = schema_reasons.get(reason, 0) + 1
            continue
        if v71.v70.v66._has_squad_overlap(row, squad_questions):
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
        raise ValueError("MuSiQue v72 has insufficient eligible balanced cases")
    selected_items = sorted(
        [item for values in retained.values() for item in values],
        key=lambda item: (item[0], item[1]),
    )
    selected = [item[2] for item in selected_items]
    selected_ids = {str(row["id"]) for row in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("MuSiQue v72 sample contains duplicate source ids")
    if any(_hash(value) in excluded_source_commitments for value in selected_ids):
        raise ValueError("MuSiQue v72 sample overlaps an excluded commitment")
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
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    prepared, maps, direct_inputs, census = v71.v70.prepare_blind_stage(
        selected, stage=stage
    )
    old_prefix = v71.v70.STAGE_PREFIXES[stage]
    new_prefix = STAGE_PREFIXES[stage]
    for row in prepared:
        row["schema_version"] = "frc-musique-v72-prepared-blind-v1"
        row["id"] = str(row["id"]).replace(old_prefix + "-", new_prefix + "-", 1)
    for row in maps:
        row["schema_version"] = "frc-musique-v72-candidate-map-v1"
        row["id"] = str(row["id"]).replace(old_prefix + "-", new_prefix + "-", 1)
    for row in direct_inputs:
        row["schema_version"] = "frc-musique-v72-direct-qa-input-v1"
        row["id"] = str(row["id"]).replace(old_prefix + "-", new_prefix + "-", 1)
        row["case_id"] = str(row["case_id"]).replace(
            old_prefix + "-", new_prefix + "-", 1
        )
    return prepared, maps, direct_inputs, census


class LocalRobertaQASupportVerifier(v71.LocalRobertaQASupportVerifier):
    def verify(
        self, rows: Sequence[dict[str, Any]], *, cache_key: str
    ) -> list[dict[str, Any]]:
        result = super().verify(rows, cache_key=cache_key)
        for row in result:
            row["schema_version"] = "frc-musique-v72-raw-qa-v1"
        return result


def aggregate_qa_rows(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> list[dict[str, Any]]:
    result = v71.v70.aggregate_qa_rows(qa_inputs, decisions, cache_key=cache_key)
    for row in result:
        row["schema_version"] = "frc-musique-v72-aggregated-qa-v1"
    return result


def initial_chain_states(
    prepared: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return v71.initial_chain_states(prepared)


def build_hop_qa_inputs(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    hop_index: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    inputs, blocked = v71.v70.build_hop_qa_inputs(
        prepared, states, hop_index=hop_index
    )
    for row in inputs:
        row["schema_version"] = "frc-musique-v72-hop-qa-input-v1"
    return inputs, blocked


def apply_hop_results(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    aggregated: Sequence[dict[str, Any]],
    blocked: dict[str, str],
    *,
    hop_index: int,
) -> list[dict[str, Any]]:
    result = v71.v70.apply_hop_results(
        prepared, states, aggregated, blocked, hop_index=hop_index
    )
    for row in result:
        row["schema_version"] = "frc-musique-v72-hop-feature-decision-v1"
    return result


def build_rival_catalog(
    qa_inputs: Sequence[dict[str, Any]],
    raw_rows: Sequence[dict[str, Any]],
    hop_decisions: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    hop_index: int,
) -> list[dict[str, Any]]:
    if len(qa_inputs) != len(raw_rows):
        raise ValueError("MuSiQue v72 rival catalog QA rows differ")
    selected = {str(row["case_id"]): row for row in hop_decisions}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for qa_input, raw in zip(qa_inputs, raw_rows, strict=True):
        if qa_input["id"] != raw["id"]:
            raise ValueError("MuSiQue v72 rival catalog QA order changed")
        case_id = str(qa_input["case_id"])
        decision = selected.get(case_id)
        if decision is None or not decision["valid_span"]:
            continue
        paragraph_idx = int(qa_input["paragraph_idx"])
        if paragraph_idx == int(decision["selected_paragraph_idx"]):
            continue
        margin = raw.get("score_margin")
        start = raw.get("span_start")
        end = raw.get("span_end")
        if (
            margin is None
            or start is None
            or end is None
            or not math.isfinite(float(margin))
            or bool(raw.get("invalid_fail_closed_used"))
        ):
            continue
        context = str(qa_input["context"])
        raw_span = context[int(start) : int(end)]
        if _hash(raw_span) != str(raw["span_sha256"]):
            raise ValueError("MuSiQue v72 reconstructed rival span changed")
        span = raw_span.strip()
        factual_span = str(states[case_id]["predictions"][hop_index])
        if not span or not _normalize_span(span):
            continue
        if _normalize_span(span) == _normalize_span(factual_span):
            continue
        candidates.setdefault(case_id, []).append(
            {
                "case_id": case_id,
                "hop_index": hop_index,
                "paragraph_idx": paragraph_idx,
                "score_margin": float(margin),
                "rival_span": span,
                "rival_span_sha256": _hash(span),
            }
        )
    result: list[dict[str, Any]] = []
    for case_id, values in sorted(candidates.items()):
        winner = sorted(
            values,
            key=lambda item: (-float(item["score_margin"]), int(item["paragraph_idx"])),
        )[0]
        result.append(
            {
                "schema_version": "frc-musique-v72-rival-catalog-v1",
                **winner,
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
        raise ValueError("MuSiQue v72 rival catalog exposes gold")
    return result


def build_sentinel_qa_inputs(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = v71.build_counterfactual_qa_inputs(prepared, states)
    for row in result:
        row["schema_version"] = "frc-musique-v72-sentinel-qa-input-v1"
        row["id"] = str(row["id"]).replace("::bridge_cf::", "::sentinel_cf::")
        row["case_id"] = str(row["case_id"]).replace(
            "::bridge_cf::", "::sentinel_cf::"
        )
    return result


def _placeholder_indices(template: str, hop_index: int) -> list[int]:
    return sorted(
        {
            int(value)
            for value in re.findall(r"#(\d+)", template)
            if 1 <= int(value) < hop_index
        }
    )


def build_rival_qa_inputs(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    rival_catalog: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    catalog = {
        (str(row["case_id"]), int(row["hop_index"])): row
        for row in rival_catalog
    }
    result: list[dict[str, Any]] = []
    for row in prepared:
        case_id = str(row["id"])
        state = states[case_id]
        expected_hops = len(row["oracle_hop_templates"])
        if not state["alive"] or int(state["completed_hops"]) != expected_hops:
            continue
        for hop_index in range(2, expected_hops + 1):
            template = str(row["oracle_hop_templates"][hop_index - 1])
            dependencies = _placeholder_indices(template, hop_index)
            for dependency in dependencies:
                rival = catalog.get((case_id, dependency))
                if rival is None:
                    continue
                question = template
                for prior in dependencies:
                    replacement = (
                        str(rival["rival_span"])
                        if prior == dependency
                        else str(state["predictions"][prior])
                    )
                    question = question.replace(f"#{prior}", replacement)
                transition_id = (
                    f"{case_id}::rival_cf::h{hop_index}::d{dependency}"
                )
                for context in row["contexts"]:
                    paragraph_idx = int(context["paragraph_idx"])
                    result.append(
                        {
                            "schema_version": "frc-musique-v72-rival-qa-input-v1",
                            "id": f"{transition_id}::p{paragraph_idx}",
                            "case_id": transition_id,
                            "root_case_id": case_id,
                            "hop_index": hop_index,
                            "dependency_hop_index": dependency,
                            "rival_source_paragraph_idx": int(
                                rival["paragraph_idx"]
                            ),
                            "rival_span_sha256": str(rival["rival_span_sha256"]),
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
        raise ValueError("MuSiQue v72 rival inputs expose gold")
    return result


def aggregate_counterfactual_qa_rows(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
    family: str,
) -> list[dict[str, Any]]:
    if family not in {"sentinel", "rival"}:
        raise ValueError("MuSiQue v72 counterfactual family changed")
    aggregated = aggregate_qa_rows(qa_inputs, decisions, cache_key=cache_key)
    metadata: dict[str, dict[str, Any]] = {}
    for row in qa_inputs:
        transition_id = str(row["case_id"])
        value = {
            "root_case_id": str(row["root_case_id"]),
            "hop_index": int(row["hop_index"]),
            "dependency_hop_index": int(row["dependency_hop_index"])
            if family == "rival"
            else None,
        }
        if transition_id in metadata and metadata[transition_id] != value:
            raise ValueError("MuSiQue v72 counterfactual transition changed")
        metadata[transition_id] = value
    result: list[dict[str, Any]] = []
    for row in aggregated:
        transition_id = str(row["case_id"])
        result.append(
            {
                **row,
                "schema_version": f"frc-musique-v72-{family}-decision-v1",
                "transition_id": transition_id,
                **metadata[transition_id],
            }
        )
    return result


def _counterfactual_statistics(
    decision: dict[str, Any],
    *,
    factual_margin: float,
    factual_paragraph: int,
    factual_span: str,
) -> tuple[float, float, float, float, bool]:
    rival_margin = decision.get("score_margin")
    rival_span = str(decision.get("predicted_span", "")).strip()
    valid = bool(
        rival_margin is not None
        and math.isfinite(float(rival_margin))
        and rival_span
        and not decision["invalid_fail_closed_used"]
    )
    if not valid:
        return 1.0, 1.0, 1.0, 1.0, False
    margin = float(rival_margin)
    advantage = min(
        max(factual_margin - margin, 0.0), MARGIN_ADVANTAGE_CLIP_MAXIMUM
    ) / MARGIN_ADVANTAGE_CLIP_MAXIMUM
    return (
        advantage,
        float(factual_margin > margin),
        float(factual_paragraph != int(decision["selected_paragraph_idx"])),
        float(_normalize_span(factual_span) != _normalize_span(rival_span)),
        True,
    )


def finalize_feature_decisions(
    prepared: Sequence[dict[str, Any]],
    direct_decisions: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    hop_decisions: Sequence[dict[str, Any]],
    sentinel_decisions: Sequence[dict[str, Any]],
    rival_decisions: Sequence[dict[str, Any]],
    rival_catalog: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    factual = v71.v70.finalize_feature_decisions(
        prepared, direct_decisions, states, hop_decisions
    )
    sentinel_grouped: dict[str, list[dict[str, Any]]] = {}
    for decision in sentinel_decisions:
        sentinel_grouped.setdefault(str(decision["root_case_id"]), []).append(
            decision
        )
    rival_grouped: dict[str, list[dict[str, Any]]] = {}
    for decision in rival_decisions:
        rival_grouped.setdefault(str(decision["root_case_id"]), []).append(decision)
    catalog_keys = {
        (str(row["case_id"]), int(row["hop_index"])) for row in rival_catalog
    }
    result: list[dict[str, Any]] = []
    for row, base in zip(prepared, factual, strict=True):
        case_id = str(row["id"])
        state = states[case_id]
        expected_hops = len(row["oracle_hop_templates"])
        sentinels = sorted(
            sentinel_grouped.get(case_id, []),
            key=lambda item: int(item["hop_index"]),
        )
        expected_sentinel_hops = list(range(2, expected_hops + 1))
        dependency_pairs: list[tuple[int, int]] = []
        eligible_pairs: list[tuple[int, int]] = []
        for hop_index in range(2, expected_hops + 1):
            dependencies = _placeholder_indices(
                str(row["oracle_hop_templates"][hop_index - 1]), hop_index
            )
            for dependency in dependencies:
                dependency_pairs.append((hop_index, dependency))
                if (case_id, dependency) in catalog_keys:
                    eligible_pairs.append((hop_index, dependency))
        rivals = sorted(
            rival_grouped.get(case_id, []),
            key=lambda item: (
                int(item["hop_index"]), int(item["dependency_hop_index"])
            ),
        )
        observed_pairs = [
            (int(item["hop_index"]), int(item["dependency_hop_index"]))
            for item in rivals
        ]
        complete = bool(
            base["feature_complete"]
            and [int(item["hop_index"]) for item in sentinels]
            == expected_sentinel_hops
            and observed_pairs == eligible_pairs
        )
        sentinel_summaries: list[dict[str, Any]] = []
        rival_summaries: list[dict[str, Any]] = []
        if complete:
            sentinel_drops: list[float] = []
            sentinel_paragraph_changes: list[float] = []
            sentinel_span_changes: list[float] = []
            for decision in sentinels:
                hop_index = int(decision["hop_index"])
                statistics = _counterfactual_statistics(
                    decision,
                    factual_margin=float(state["margins"][hop_index - 1]),
                    factual_paragraph=int(
                        state["selected_paragraphs"][hop_index - 1]
                    ),
                    factual_span=str(state["predictions"][hop_index]),
                )
                sentinel_drops.append(statistics[0])
                sentinel_paragraph_changes.append(statistics[2])
                sentinel_span_changes.append(statistics[3])
                sentinel_summaries.append(
                    {
                        "hop_index": hop_index,
                        "valid_span": statistics[4],
                        "positive_margin_drop_scaled": statistics[0],
                        "selected_paragraph_changed": statistics[2],
                        "normalized_span_changed": statistics[3],
                        "predicted_span_sha256": str(
                            decision["predicted_span_sha256"]
                        ),
                    }
                )
            sentinel_features = {
                "bridge_margin_drop_min": min(sentinel_drops),
                "bridge_margin_drop_mean": float(np.mean(sentinel_drops)),
                "bridge_paragraph_change_min": min(sentinel_paragraph_changes),
                "bridge_paragraph_change_mean": float(
                    np.mean(sentinel_paragraph_changes)
                ),
                "bridge_span_change_min": min(sentinel_span_changes),
                "bridge_span_change_mean": float(np.mean(sentinel_span_changes)),
            }
            advantages: list[float] = []
            wins: list[float] = []
            paragraph_changes: list[float] = []
            span_changes: list[float] = []
            for decision in rivals:
                hop_index = int(decision["hop_index"])
                statistics = _counterfactual_statistics(
                    decision,
                    factual_margin=float(state["margins"][hop_index - 1]),
                    factual_paragraph=int(
                        state["selected_paragraphs"][hop_index - 1]
                    ),
                    factual_span=str(state["predictions"][hop_index]),
                )
                advantages.append(statistics[0])
                wins.append(statistics[1])
                paragraph_changes.append(statistics[2])
                span_changes.append(statistics[3])
                rival_summaries.append(
                    {
                        "hop_index": hop_index,
                        "dependency_hop_index": int(
                            decision["dependency_hop_index"]
                        ),
                        "valid_span": statistics[4],
                        "positive_factual_margin_advantage_scaled": statistics[0],
                        "factual_margin_wins": statistics[1],
                        "selected_paragraph_changed": statistics[2],
                        "normalized_span_changed": statistics[3],
                        "predicted_span_sha256": str(
                            decision["predicted_span_sha256"]
                        ),
                    }
                )
            coverage = (
                len(eligible_pairs) / len(dependency_pairs)
                if dependency_pairs
                else 0.0
            )
            rival_features = {
                "rival_dependency_coverage_fraction": coverage,
                "rival_margin_advantage_min": min(advantages)
                if advantages
                else 0.0,
                "rival_margin_advantage_mean": float(np.mean(advantages))
                if advantages
                else 0.0,
                "rival_factual_win_fraction": float(np.mean(wins))
                if wins
                else 0.0,
                "rival_paragraph_change_fraction": float(
                    np.mean(paragraph_changes)
                )
                if paragraph_changes
                else 0.0,
                "rival_span_change_fraction": float(np.mean(span_changes))
                if span_changes
                else 0.0,
            }
            features = {**base["features"], **sentinel_features, **rival_features}
        else:
            features = {name: None for name in ALL_FEATURE_NAMES}
        if tuple(features) != ALL_FEATURE_NAMES:
            raise ValueError("MuSiQue v72 full feature order changed")
        result.append(
            {
                "schema_version": "frc-musique-v72-feature-decision-v1",
                "case_id": case_id,
                "hop_count": base["hop_count"],
                "executed_hops": base["executed_hops"],
                "valid_hops": base["valid_hops"],
                "sentinel_transitions": len(sentinels),
                "rival_dependency_count": len(dependency_pairs),
                "rival_eligible_count": len(eligible_pairs),
                "rival_transitions": len(rivals),
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
                "sentinel_decision_sha256": _hash(
                    json.dumps(
                        sentinel_summaries, sort_keys=True, separators=(",", ":")
                    )
                ),
                "rival_decision_sha256": _hash(
                    json.dumps(
                        rival_summaries, sort_keys=True, separators=(",", ":")
                    )
                ),
                "invalid_fail_closed_used": bool(
                    base["invalid_fail_closed_used"] or not complete
                ),
                "hop_decision_sha256": base["hop_decision_sha256"],
            }
        )
    return result


def _fail_closed_features(hop_count: int) -> dict[str, float]:
    prior = v71._fail_closed_features(hop_count)
    rival = {name: 0.0 for name in RIVAL_FEATURE_NAMES}
    return {**prior, **rival}


def build_gold_evidence(
    selected: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    feature_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(selected) == len(prepared) == len(feature_decisions)):
        raise ValueError("MuSiQue v72 evidence inputs differ in length")
    evidence: list[dict[str, Any]] = []
    for source, blind, feature in zip(
        selected, prepared, feature_decisions, strict=True
    ):
        case_id = str(blind["id"])
        if case_id != str(feature["case_id"]):
            raise ValueError("MuSiQue v72 evidence order changed")
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
                "schema_version": "frc-musique-v72-feature-evidence-v1",
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
                "sentinel_transitions": feature["sentinel_transitions"],
                "rival_dependency_count": feature["rival_dependency_count"],
                "rival_eligible_count": feature["rival_eligible_count"],
                "rival_transitions": feature["rival_transitions"],
                "sentinel_decision_sha256": feature[
                    "sentinel_decision_sha256"
                ],
                "rival_decision_sha256": feature["rival_decision_sha256"],
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
        raise ValueError("MuSiQue v72 feature matrix is non-finite")
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
        index
        for index, name in enumerate(names)
        if name in NONNEGATIVE_FEATURE_NAMES
    ]
    for _ in range(OPTIMIZER_ITERATIONS):
        probabilities = _sigmoid(matrix @ weights + intercept)
        residual = probabilities - labels
        gradient = (
            matrix.T @ residual / len(labels) + OPTIMIZER_L2_WEIGHT * weights
        )
        intercept_gradient = float(np.mean(residual))
        weights -= OPTIMIZER_LEARNING_RATE * gradient
        intercept -= OPTIMIZER_LEARNING_RATE * intercept_gradient
        if monotone_indices:
            weights[monotone_indices] = np.maximum(
                weights[monotone_indices], 0.0
            )
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


def _raw_scores(evidence: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in evidence], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"MuSiQue v72 non-finite raw score: {field}")
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
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds
            != fold
        ]
        held = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds
            == fold
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
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds
            != fold
        ]
        held = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds
            == fold
        ]
        model = fit_projected_logistic(train, feature_names=feature_names)
        selected = select_threshold_from_scores(
            _labels(train), predict_projected_logistic(train, model)
        )
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
    "fixed_sentinel_control": FIXED_SENTINEL_FEATURE_NAMES,
    "in_domain_rival_only_control": RIVAL_FEATURE_NAMES,
    "in_domain_rival_candidate": CANDIDATE_FEATURE_NAMES,
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
        raise ValueError("MuSiQue v72 evaluation stage changed")
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
        "fixed_sentinel_control",
        "in_domain_rival_only_control",
    )
    prior_control_order = (
        "linear_nine_signal_control",
        "monotone_additive_spline_control",
        "monotone_interaction_control",
        "paragraph_competition_control",
        "fixed_sentinel_control",
    )

    def _strongest(names: Sequence[str]) -> str:
        return max(
            names,
            key=lambda name: (
                methods[name]["balanced_accuracy"], -names.index(name)
            ),
        )

    baseline_name = _strongest(comparator_order)
    prior_name = _strongest(prior_control_order)
    candidate_name = "in_domain_rival_candidate"

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
    rival_only_paired = _paired("in_domain_rival_only_control", 2000)
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
                float(
                    np.mean([not candidate_decisions[index] for index in indices])
                ),
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
        "candidate_minus_strongest_fair_baseline_at_least_0_05": paired[
            "point"
        ]
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
        "candidate_minus_rival_only_control_at_least_0_02": rival_only_paired[
            "point"
        ]
        >= 0.02,
        "rival_only_control_paired_correctness_ci_low_above_0": rival_only_paired[
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
            "MUSIQUE_V72_IN_DOMAIN_RIVAL_DEVELOPMENT_FEASIBILITY_"
            "ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "MUSIQUE_V72_IN_DOMAIN_RIVAL_DEVELOPMENT_SUPPORT_NOT_"
            "ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "MUSIQUE_V72_ORACLE_PLAN_IN_DOMAIN_RIVAL_COMPONENT_"
            "FEASIBILITY_ESTABLISHED"
            if passed
            else "MUSIQUE_V72_IN_DOMAIN_RIVAL_CONFIRMATION_NOT_ESTABLISHED"
        )
    return {
        "schema_version": "frc-musique-v72-stage-result-v1",
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
            "incremental_delta_vs_rival_only_control": rival_only_paired,
            "unanswerable_hop_strata": strata,
            "invalid_feature_output_count": invalid_count,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": stage == "development" and passed,
                "selector_or_retrieval_scoring_part_of_v72": False,
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
    status = "MUSIQUE_V72_MODELS_AND_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT"
    payload = {
        "schema_version": "frc-musique-v72-calibration-result-v1",
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
        "# MuSiQue v72 域内竞争桥接实体校准",
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
            "校准只冻结模型和阈值并开放互斥开发集，不授权采用、检索评分或 Gate 2 放行。",
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
        f"# MuSiQue v72 域内竞争桥接实体支持门（{report['metadata']['stage']}）",
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
    only = analysis["incremental_delta_vs_rival_only_control"]
    lines.extend(
        [
            "",
            f"候选相对最强公平基线差值 {paired['point']:+.6f}，95% CI "
            f"[{paired['ci_low']:+.6f},{paired['ci_high']:+.6f}]。",
            f"候选相对最强既有特征控制差值 {prior['point']:+.6f}，95% CI "
            f"[{prior['ci_low']:+.6f},{prior['ci_high']:+.6f}]。",
            f"候选相对域内竞争信号单独控制差值 {only['point']:+.6f}，95% CI "
            f"[{only['ci_low']:+.6f},{only['ci_high']:+.6f}]。",
            "",
            "该实验只检验 oracle 计划域内竞争桥接实体支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
