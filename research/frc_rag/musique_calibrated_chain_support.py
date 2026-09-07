"""Case-level calibrated MuSiQue chain-support experiment (v67)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag import musique_sequential_chain_support as v66

EXPERIMENT_ID = "FRC-MUSIQUE-CALIBRATED-CHAIN-SUPPORT-V67"
PROTOCOL_SCHEMA = "frc-musique-calibrated-chain-support-protocol-v67"
SOURCE_SCHEMA = "frc-musique-calibrated-chain-support-source-registration-v67"
FIXED_V66_THRESHOLD = 0.974609375
STAGE_SALTS = {
    "calibration": "FRC-MUSIQUE-V67-CALIBRATION|",
    "development": "FRC-MUSIQUE-V67-DEVELOPMENT|",
    "confirmation": "FRC-MUSIQUE-V67-CONFIRMATION|",
}
STAGE_PREFIXES = {
    "calibration": "m67k",
    "development": "m67d",
    "confirmation": "m67c",
}
STAGE_TARGET_PER_GROUP = {
    "calibration": 500,
    "development": 300,
    "confirmation": 300,
}
STAGE_BOOTSTRAP_SEEDS = {
    "development": 20260823,
    "confirmation": 20260824,
}
PAIR_ASSIGNMENT_SALT = "FRC-MUSIQUE-V67-PAIR-ASSIGNMENT|"
CROSSFIT_SALT = "FRC-MUSIQUE-V67-CROSSFIT|"

sha256 = v66.sha256
normalize_question = v66.normalize_question
nested_keys = v66.nested_keys
extract_squad2_questions = v66.extract_squad2_questions
load_source_commitments = v66.load_source_commitments
resolve_placeholders = v66.resolve_placeholders


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("MuSiQue v67 protocol schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v67 experiment id changed")
    stages = value.get("frozen_stages", {})
    for stage, per_group in STAGE_TARGET_PER_GROUP.items():
        if int(stages[stage]["target_per_answer_state"]) != per_group:
            raise ValueError(f"MuSiQue v67 {stage} size changed")
    if int(value["threshold_selection"]["learned_parameter_count"]) != 2:
        raise ValueError("MuSiQue v67 learned parameter count changed")
    if value["stopping_and_outcomes"]["gate_2"] != "NO-GO/SHADOW":
        raise ValueError("MuSiQue v67 Gate 2 boundary changed")
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
        raise ValueError("MuSiQue v67 source registration schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MuSiQue v67 source registration experiment changed")
    expected = {
        train_path: protocol["sources"]["calibration_and_development_sha256"],
        dev_path: protocol["sources"]["optional_confirmation_sha256"],
        squad2_path: value["leakage_guard_source"]["sha256"],
    }
    prior_values = list(value["prior_exclusion_sources"].values())[:3]
    if len(prior_paths) != 3:
        raise ValueError("MuSiQue v67 requires exactly three prior maps")
    expected.update(
        {path_value: item["sha256"] for path_value, item in zip(prior_paths, prior_values, strict=True)}
    )
    for source_path, expected_hash in expected.items():
        if not source_path.exists() or sha256(source_path) != expected_hash:
            raise ValueError(f"MuSiQue v67 registered source changed: {source_path}")
    if int(value["prior_exclusion_sources"]["expected_union_source_commitments"]) != 1500:
        raise ValueError("MuSiQue v67 prior commitment count changed")
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
        raise ValueError(f"Unsupported MuSiQue v67 stage: {stage}")
    expected_split = "dev" if stage == "confirmation" else "train"
    if source_split != expected_split:
        raise ValueError("MuSiQue v67 stage/source split mismatch")
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
            if int(_hash(PAIR_ASSIGNMENT_SALT + source_split + "|" + source_id), 16) % 2 == 0
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
        retained[group].append((_hash(STAGE_SALTS[stage] + source_id), source_id, row))
        retained[group].sort(key=lambda item: (item[0], item[1]))
        if len(retained[group]) > target:
            retained[group].pop()
    if any(len(values) != target for values in retained.values()):
        raise ValueError("MuSiQue v67 has insufficient eligible balanced cases")
    selected_items = sorted(
        [item for values in retained.values() for item in values],
        key=lambda item: (item[0], item[1]),
    )
    selected = [item[2] for item in selected_items]
    selected_ids = {str(row["id"]) for row in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("MuSiQue v67 sample contains duplicate source ids")
    if any(_hash(value) in excluded_source_commitments for value in selected_ids):
        raise ValueError("MuSiQue v67 sample overlaps an excluded commitment")
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
                {"paragraph_idx": paragraph_idx, "context_sha256": context_row["context_sha256"]}
            )
            direct_inputs.append(
                {
                    "schema_version": "frc-musique-v67-direct-qa-input-v1",
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
                "schema_version": "frc-musique-v67-prepared-blind-v1",
                "id": case_id,
                "composed_question": str(source["question"]),
                "oracle_hop_templates": templates,
                "contexts": contexts,
                "gold_fields_visible_to_chain_executor": False,
            }
        )
        maps.append(
            {
                "schema_version": "frc-musique-v67-candidate-map-v1",
                "id": case_id,
                "source_id_commitment": _hash(source_id),
                "contexts": mapping_contexts,
            }
        )
        paragraph_counts.append(len(contexts))
        hop_counts.append(len(templates))
    forbidden = {"answerable", "answer", "answer_aliases", "paragraph_support_idx", "is_supporting"}
    if forbidden & nested_keys([prepared, maps, direct_inputs]):
        raise ValueError("MuSiQue v67 blind caches expose forbidden gold fields")
    return prepared, maps, direct_inputs, {
        "prepared_cases": len(prepared),
        "direct_qa_input_rows": len(direct_inputs),
        "minimum_paragraphs": min(paragraph_counts, default=0),
        "maximum_paragraphs": max(paragraph_counts, default=0),
        "mean_paragraphs": sum(paragraph_counts) / len(paragraph_counts) if paragraph_counts else 0.0,
        "hop_count_distribution": {str(hop): hop_counts.count(hop) for hop in sorted(set(hop_counts))},
        "gold_fields_exported_to_blind_caches": False,
    }


class LocalRobertaQASupportVerifier(v66.LocalRobertaQASupportVerifier):
    def verify(self, rows: Sequence[dict[str, Any]], *, cache_key: str) -> list[dict[str, Any]]:
        result = super().verify(rows, cache_key=cache_key)
        for row in result:
            row["schema_version"] = "frc-musique-v67-raw-qa-v1"
        return result


def aggregate_qa_rows(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> list[dict[str, Any]]:
    base = v66.aggregate_qa_rows(qa_inputs, decisions, cache_key=cache_key)
    result: list[dict[str, Any]] = []
    for row in base:
        result.append(
            {
                "schema_version": "frc-musique-v67-aggregated-qa-v1",
                "case_id": row["case_id"],
                "paragraph_rows": row["paragraph_rows"],
                "invalid_paragraph_rows": row["invalid_paragraph_rows"],
                "invalid_fail_closed_used": row["invalid_fail_closed_used"],
                "score_margin": row["score_margin"],
                "selected_paragraph_idx": row["selected_paragraph_idx"],
                "predicted_span": row["predicted_span"],
                "predicted_span_sha256": row["predicted_span_sha256"],
                "decision_sha256": row["decision_sha256"],
            }
        )
    return result


def initial_chain_states(prepared: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): {
            "alive": True,
            "predictions": {},
            "margins": [],
            "completed_hops": 0,
            "invalid_output": False,
        }
        for row in prepared
    }


def build_hop_qa_inputs(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    hop_index: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    qa_inputs: list[dict[str, Any]] = []
    blocked: dict[str, str] = {}
    for row in prepared:
        case_id = str(row["id"])
        templates = list(row["oracle_hop_templates"])
        if hop_index > len(templates):
            continue
        state = states[case_id]
        if not state["alive"]:
            blocked[case_id] = "invalid_predecessor_output"
            continue
        question, missing = resolve_placeholders(
            str(templates[hop_index - 1]),
            {int(key): str(value) for key, value in state["predictions"].items()},
            hop_index=hop_index,
        )
        if question is None:
            blocked[case_id] = "unresolved_placeholder:" + ",".join(map(str, missing))
            continue
        for context in row["contexts"]:
            paragraph_idx = int(context["paragraph_idx"])
            qa_inputs.append(
                {
                    "schema_version": "frc-musique-v67-hop-qa-input-v1",
                    "id": f"{case_id}::hop{hop_index}::p{paragraph_idx}",
                    "case_id": case_id,
                    "hop_index": hop_index,
                    "paragraph_idx": paragraph_idx,
                    "question": question,
                    "context": str(context["context"]),
                    "gold_fields_visible_to_verifier": False,
                }
            )
    return qa_inputs, blocked


def apply_hop_results(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    aggregated: Sequence[dict[str, Any]],
    blocked: dict[str, str],
    *,
    hop_index: int,
) -> list[dict[str, Any]]:
    by_case = {str(row["case_id"]): row for row in aggregated}
    decisions: list[dict[str, Any]] = []
    for row in prepared:
        case_id = str(row["id"])
        if hop_index > len(row["oracle_hop_templates"]):
            continue
        state = states[case_id]
        aggregate = by_case.get(case_id)
        reason = blocked.get(case_id)
        if reason is not None:
            state["alive"] = False
            executed = False
            valid = False
            margin = None
            paragraph_idx = None
            span_hash = _hash("")
            invalid = False
        elif aggregate is None:
            raise ValueError("MuSiQue v67 active hop case lacks an aggregate")
        else:
            executed = True
            margin = aggregate["score_margin"]
            predicted_span = str(aggregate["predicted_span"]).strip()
            valid = bool(
                margin is not None
                and math.isfinite(float(margin))
                and predicted_span
                and not aggregate["invalid_fail_closed_used"]
            )
            paragraph_idx = aggregate["selected_paragraph_idx"]
            span_hash = str(aggregate["predicted_span_sha256"])
            invalid = not valid
            if valid:
                state["predictions"][hop_index] = predicted_span
                state["margins"].append(float(margin))
                state["completed_hops"] += 1
                reason = None
            else:
                state["alive"] = False
                state["invalid_output"] = True
                reason = "invalid_or_nonfinite_span"
        decisions.append(
            {
                "schema_version": "frc-musique-v67-hop-score-decision-v1",
                "case_id": case_id,
                "hop_index": hop_index,
                "executed": executed,
                "valid_span": valid,
                "failure_reason": reason,
                "score_margin": margin,
                "selected_paragraph_idx": paragraph_idx,
                "predicted_span_sha256": span_hash,
                "invalid_fail_closed_used": invalid,
            }
        )
    return decisions


def finalize_chain_scores(
    prepared: Sequence[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    hop_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for decision in hop_decisions:
        grouped.setdefault(str(decision["case_id"]), []).append(decision)
    result: list[dict[str, Any]] = []
    for row in prepared:
        case_id = str(row["id"])
        expected_hops = len(row["oracle_hop_templates"])
        decisions = sorted(grouped.get(case_id, []), key=lambda item: item["hop_index"])
        if len(decisions) != expected_hops:
            raise ValueError("MuSiQue v67 chain decision count changed")
        state = states[case_id]
        complete = bool(state["alive"] and int(state["completed_hops"]) == expected_hops)
        score = min(state["margins"]) if complete else None
        result.append(
            {
                "schema_version": "frc-musique-v67-chain-score-decision-v1",
                "case_id": case_id,
                "hop_count": expected_hops,
                "executed_hops": sum(bool(item["executed"]) for item in decisions),
                "valid_hops": sum(bool(item["valid_span"]) for item in decisions),
                "chain_complete": complete,
                "chain_bottleneck_score": score,
                "invalid_fail_closed_used": bool(state["invalid_output"] or not complete),
                "hop_decision_sha256": _hash(json.dumps(decisions, sort_keys=True, separators=(",", ":"))),
            }
        )
    return result


def build_gold_evidence(
    selected: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    direct_decisions: Sequence[dict[str, Any]],
    chain_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(selected) == len(prepared) == len(direct_decisions) == len(chain_decisions)):
        raise ValueError("MuSiQue v67 evidence inputs differ in length")
    evidence: list[dict[str, Any]] = []
    for source, blind, direct, chain in zip(
        selected, prepared, direct_decisions, chain_decisions, strict=True
    ):
        case_id = str(blind["id"])
        if case_id != str(direct["case_id"]) or case_id != str(chain["case_id"]):
            raise ValueError("MuSiQue v67 evidence order changed")
        evidence.append(
            {
                "schema_version": "frc-musique-v67-score-evidence-v1",
                "case_id": case_id,
                "answer_state": "answerable" if source["answerable"] else "unanswerable",
                "hop_count": len(source["question_decomposition"]),
                "paragraph_count": len(source["paragraphs"]),
                "direct_score_margin": direct["score_margin"],
                "direct_prediction_sha256": direct["predicted_span_sha256"],
                "chain_bottleneck_score": chain["chain_bottleneck_score"],
                "chain_complete": chain["chain_complete"],
                "chain_executed_hops": chain["executed_hops"],
                "chain_valid_hops": chain["valid_hops"],
                "chain_hop_decision_sha256": chain["hop_decision_sha256"],
                "invalid_chain_output": chain["invalid_fail_closed_used"],
            }
        )
    return evidence


def _decision(score: Any, threshold: float) -> bool:
    return bool(score is not None and math.isfinite(float(score)) and float(score) > threshold)


def method_metrics(
    evidence: Sequence[dict[str, Any]], *, score_field: str, threshold: float
) -> dict[str, Any]:
    answerable = [row for row in evidence if row["answer_state"] == "answerable"]
    unanswerable = [row for row in evidence if row["answer_state"] == "unanswerable"]
    if not answerable or not unanswerable:
        raise ValueError("MuSiQue v67 metrics require both answer states")
    answer_pass = float(np.mean([_decision(row[score_field], threshold) for row in answerable]))
    rejection = float(np.mean([not _decision(row[score_field], threshold) for row in unanswerable]))
    return {
        "rows": len(evidence),
        "answerable_rows": len(answerable),
        "unanswerable_rows": len(unanswerable),
        "threshold_exact": float(threshold),
        "answerable_pass_rate": round(answer_pass, 6),
        "unanswerable_rejection_rate": round(rejection, 6),
        "balanced_accuracy": round((answer_pass + rejection) / 2.0, 6),
    }


def select_threshold(
    evidence: Sequence[dict[str, Any]], *, score_field: str
) -> dict[str, Any]:
    scores = sorted(
        {
            float(row[score_field])
            for row in evidence
            if row[score_field] is not None and math.isfinite(float(row[score_field]))
        }
    )
    if not scores:
        raise ValueError("MuSiQue v67 calibration score set is empty")
    candidates = [math.nextafter(scores[0], -math.inf), *scores]
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for threshold in candidates:
        metrics = method_metrics(evidence, score_field=score_field, threshold=threshold)
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


def crossfit_threshold_diagnostic(
    evidence: Sequence[dict[str, Any]], *, score_field: str, folds: int = 5
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    predictions: list[tuple[dict[str, Any], bool]] = []
    for fold in range(folds):
        train = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds != fold
        ]
        held_out = [
            row
            for row in evidence
            if int(_hash(CROSSFIT_SALT + str(row["case_id"])), 16) % folds == fold
        ]
        threshold_result = select_threshold(train, score_field=score_field)
        threshold = float(threshold_result["threshold_exact"])
        predictions.extend((row, _decision(row[score_field], threshold)) for row in held_out)
        fold_rows.append(
            {
                "fold": fold,
                "train_cases": len(train),
                "held_out_cases": len(held_out),
                "threshold_exact": threshold,
                "training_safety_constraints_met": threshold_result["safety_constraints_met"],
            }
        )
    if len(predictions) != len(evidence):
        raise ValueError("MuSiQue v67 crossfit coverage changed")
    answerable = [pred for row, pred in predictions if row["answer_state"] == "answerable"]
    unanswerable = [pred for row, pred in predictions if row["answer_state"] == "unanswerable"]
    answer_pass = float(np.mean(answerable))
    rejection = float(np.mean([not value for value in unanswerable]))
    return {
        "folds": fold_rows,
        "held_out_cases": len(predictions),
        "answerable_pass_rate": round(answer_pass, 6),
        "unanswerable_rejection_rate": round(rejection, 6),
        "balanced_accuracy": round((answer_pass + rejection) / 2.0, 6),
    }


def fit_calibration(evidence: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "direct_composed_question": {
            "pooled": select_threshold(evidence, score_field="direct_score_margin"),
            "crossfit": crossfit_threshold_diagnostic(evidence, score_field="direct_score_margin"),
        },
        "oracle_plan_chain_bottleneck": {
            "pooled": select_threshold(evidence, score_field="chain_bottleneck_score"),
            "crossfit": crossfit_threshold_diagnostic(evidence, score_field="chain_bottleneck_score"),
        },
    }


def _correct(row: dict[str, Any], decision: bool) -> bool:
    return decision if row["answer_state"] == "answerable" else not decision


def paired_correctness_interval(
    evidence: Sequence[dict[str, Any]],
    *,
    candidate_threshold: float,
    baseline_score_field: str,
    baseline_threshold: float,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    deltas = np.asarray(
        [
            float(_correct(row, _decision(row["chain_bottleneck_score"], candidate_threshold)))
            - float(_correct(row, _decision(row[baseline_score_field], baseline_threshold)))
            for row in evidence
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        samples[index] = float(np.mean(rng.choice(deltas, size=len(deltas), replace=True)))
    return {
        "point": round(float(np.mean(deltas)), 6),
        "ci_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_high": round(float(np.quantile(samples, 0.975)), 6),
        "resamples": resamples,
        "seed": seed,
    }


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
        raise ValueError("MuSiQue v67 evaluation stage changed")
    direct_threshold = float(calibration["direct_composed_question"]["pooled"]["threshold_exact"])
    chain_threshold = float(calibration["oracle_plan_chain_bottleneck"]["pooled"]["threshold_exact"])
    direct = method_metrics(evidence, score_field="direct_score_margin", threshold=direct_threshold)
    fixed = method_metrics(evidence, score_field="chain_bottleneck_score", threshold=FIXED_V66_THRESHOLD)
    candidate = method_metrics(evidence, score_field="chain_bottleneck_score", threshold=chain_threshold)
    baseline_name, baseline_metrics, baseline_field, baseline_threshold = (
        ("calibrated_direct_composed_question", direct, "direct_score_margin", direct_threshold)
        if direct["balanced_accuracy"] >= fixed["balanced_accuracy"]
        else ("fixed_v66_full_chain", fixed, "chain_bottleneck_score", FIXED_V66_THRESHOLD)
    )
    paired = paired_correctness_interval(
        evidence,
        candidate_threshold=chain_threshold,
        baseline_score_field=baseline_field,
        baseline_threshold=baseline_threshold,
        seed=STAGE_BOOTSTRAP_SEEDS[stage],
    )
    strata: dict[str, Any] = {}
    for hop in sorted({int(row["hop_count"]) for row in evidence if row["answer_state"] == "unanswerable"}):
        rows = [row for row in evidence if row["answer_state"] == "unanswerable" and int(row["hop_count"]) == hop]
        strata[str(hop)] = {
            "cases": len(rows),
            "rejection_rate": round(float(np.mean([not _decision(row["chain_bottleneck_score"], chain_threshold) for row in rows])), 6),
        }
    invalid_count = sum(bool(row["invalid_chain_output"]) for row in evidence)
    checks = {
        "exact_cases_equals_600": len(evidence) == 600,
        "exact_answer_state_balance": sum(row["answer_state"] == "answerable" for row in evidence) == 300,
        "schema_exclusion_rate_at_most_0_01": float(sampling["schema_exclusion_rate"]) <= 0.01,
        "selected_prior_or_calibration_source_overlap_equals_0": int(sampling["selected_excluded_source_commitment_overlap"]) == 0,
        "selected_squad2_exact_question_overlap_equals_0": int(sampling["selected_squad2_exact_question_overlap"]) == 0,
        "invalid_chain_output_rate_equals_0": invalid_count == 0,
        "calibrated_chain_balanced_accuracy_at_least_0_72": candidate["balanced_accuracy"] >= 0.72,
        "calibrated_chain_answerable_pass_rate_at_least_0_60": candidate["answerable_pass_rate"] >= 0.60,
        "calibrated_chain_unanswerable_rejection_rate_at_least_0_80": candidate["unanswerable_rejection_rate"] >= 0.80,
        "calibrated_chain_minus_strongest_fair_baseline_at_least_0_05": paired["point"] >= 0.05,
        "paired_correctness_ci_low_above_0": paired["ci_low"] > 0.0,
        "every_observed_unanswerable_hop_stratum_rejection_rate_at_least_0_75": all(value["rejection_rate"] >= 0.75 for value in strata.values()),
    }
    passed = all(checks.values())
    if stage == "development":
        status = (
            "MUSIQUE_V67_CALIBRATED_CHAIN_DEVELOPMENT_FEASIBILITY_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "MUSIQUE_V67_CALIBRATED_CHAIN_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "MUSIQUE_V67_ORACLE_PLAN_CALIBRATED_CHAIN_COMPONENT_FEASIBILITY_ESTABLISHED"
            if passed
            else "MUSIQUE_V67_CALIBRATED_CHAIN_CONFIRMATION_NOT_ESTABLISHED"
        )
    return {
        "schema_version": "frc-musique-v67-stage-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "stage": stage,
            "cases": len(evidence),
            "answer_state_counts": {
                "answerable": sum(row["answer_state"] == "answerable" for row in evidence),
                "unanswerable": sum(row["answer_state"] == "unanswerable" for row in evidence),
            },
            "sampling": sampling,
            "structural_census": structural_census,
            "source_artifacts": source_artifacts,
            "automatic_decomposer_result": False,
            "strict_independent_model_training_confirmation": False,
            "official_musique_leaderboard_result": False,
        },
        "analysis": {
            "calibrated_direct_composed_question": direct,
            "fixed_v66_full_chain": fixed,
            "calibrated_oracle_plan_chain_bottleneck": candidate,
            "strongest_fair_baseline": {"name": baseline_name, **baseline_metrics},
            "paired_correctness_delta": paired,
            "unanswerable_hop_strata": strata,
            "invalid_chain_output_count": invalid_count,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": stage == "development" and passed,
                "selector_or_retrieval_scoring_part_of_v67": False,
                "reuse_failed_stage_for_tuning_threshold_score_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_evidence(path: Path, evidence: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in evidence:
                compressed.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


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
        "schema_version": "frc-musique-v67-calibration-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": metadata,
        "analysis": calibration,
        "outcome": {
            "status": "MUSIQUE_V67_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT",
            "development_open_authorized": True,
            "calibration_alone_authorizes_adoption": False,
            "gate_2": "NO-GO/SHADOW",
        },
    }
    write_json(result_path, payload)
    write_evidence(evidence_path, evidence)
    lines = [
        "# MuSiQue v67 链级阈值校准",
        "",
        f"- 案例：{len(evidence)}",
        "- 状态：`MUSIQUE_V67_THRESHOLDS_FROZEN_OPEN_DEVELOPMENT`",
        "",
        "| 方法 | 阈值 | OOF 平衡准确率 | OOF 答案通过率 | OOF 无答案拒绝率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("direct_composed_question", "oracle_plan_chain_bottleneck"):
        value = calibration[name]
        lines.append(
            f"| `{name}` | {value['pooled']['threshold_exact']:.9f} | {value['crossfit']['balanced_accuracy']:.6f} | {value['crossfit']['answerable_pass_rate']:.6f} | {value['crossfit']['unanswerable_rejection_rate']:.6f} |"
        )
    lines.extend(["", "校准结果只冻结阈值并开放互斥开发集，不授权方法采用、检索评分或 Gate 2 放行。", ""])
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
    outcome = analysis["outcome"]
    lines = [
        f"# MuSiQue v67 校准链支持门（{report['metadata']['stage']}）",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 案例：{report['metadata']['cases']}",
        f"- 最强公平基线：`{analysis['strongest_fair_baseline']['name']}`",
        "",
        "| 方法 | 平衡准确率 | 答案通过率 | 无答案拒绝率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, name in (
        ("校准直接门", "calibrated_direct_composed_question"),
        ("固定 v66 完整链", "fixed_v66_full_chain"),
        ("校准链瓶颈门", "calibrated_oracle_plan_chain_bottleneck"),
    ):
        value = analysis[name]
        lines.append(
            f"| {label} | {value['balanced_accuracy']:.6f} | {value['answerable_pass_rate']:.6f} | {value['unanswerable_rejection_rate']:.6f} |"
        )
    paired = analysis["paired_correctness_delta"]
    lines.extend(
        [
            "",
            f"候选相对最强公平基线的配对正确性差值为 {paired['point']:+.6f}，95% CI [{paired['ci_low']:+.6f},{paired['ci_high']:+.6f}]。",
            "",
            "该实验只检验 oracle 计划下的链级校准支持门，不是自动分解、FRC 检索、真实 SetR、洪水领域或生产结果。Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
