"""Oracle-plan sequential MuSiQue chain-support experiment (v66)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.musique_full_roberta_transfer import (
    LocalRobertaQASupportVerifier as V65LocalRobertaQASupportVerifier,
)


EXPERIMENT_ID = "FRC-MUSIQUE-SEQUENTIAL-CHAIN-SUPPORT-V66"
DATASET_ID = "musique_full_v1.0_train_v66_case_disjoint_successor"
LOCKED_THRESHOLD = 0.974609375
TARGET_CASES = 600
TARGET_PER_GROUP = 300
PAIR_ASSIGNMENT_SALT = "FRC-MUSIQUE-FULL-V66-PAIR-ASSIGNMENT|"
STAGE_SALTS = {
    "development": "FRC-MUSIQUE-FULL-V66-DEVELOPMENT|",
    "confirmation": "FRC-MUSIQUE-FULL-V66-CONFIRMATION|",
}
BOOTSTRAP_SEEDS = {"development": 20260820, "confirmation": 20260821}
PROTOCOL_SHA256 = "518bb128745521677e3e5b33e42cfd7484aab8163c1fb3441fb4192c1aa70456"
SOURCE_REGISTRATION_SHA256 = (
    "0febdc4c8ae7e495c7f79d826444633a154feeac64017d5d22095dbd6494dd3f"
)
MODEL_REGISTRATION_SHA256 = (
    "b80ee8b127713e1440c25eee58974fe5699a808fdec37980f9729d239d048325"
)
TARGET_SOURCE_SHA256 = (
    "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034"
)
SQUAD2_TRAIN_SHA256 = (
    "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002"
)
V65_INVALID_MAP_SHA256 = (
    "bd26514fc68009b4f39ca94991d03c8cd980b84df363024d4b9a1f9c434dcbba"
)
V65_CORRECTED_MAP_SHA256 = (
    "d5095960ffcf38f4491830e19d5af0ac3c877edb5cd4ca5d5fd73d0f7179a965"
)
DEVELOPMENT_FAILURE = (
    "MUSIQUE_V66_SEQUENTIAL_CHAIN_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_"
    "STOP_BEFORE_CONFIRMATION"
)
DEVELOPMENT_PASS = (
    "MUSIQUE_V66_SEQUENTIAL_CHAIN_DEVELOPMENT_FEASIBILITY_ESTABLISHED_"
    "OPEN_CONFIRMATION"
)
CONFIRMATION_FAILURE = "MUSIQUE_V66_SEQUENTIAL_CHAIN_CONFIRMATION_NOT_ESTABLISHED"
CONFIRMATION_PASS = (
    "MUSIQUE_V66_ORACLE_PLAN_SEQUENTIAL_CHAIN_COMPONENT_FEASIBILITY_ESTABLISHED"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())


def nested_keys(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        return set(payload) | {
            key for value in payload.values() for key in nested_keys(value)
        }
    if isinstance(payload, list):
        return {key for value in payload for key in nested_keys(value)}
    return set()


def validate_protocol(path: Path) -> dict[str, Any]:
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("MuSiQue v66 protocol hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("MuSiQue v66 protocol experiment changed")
    if value["prior_boundary"]["v66_case_ids_selected_before_this_protocol"]:
        raise ValueError("MuSiQue v66 cases were selected before protocol freeze")
    if value["blind_execution"]["locked_threshold_exact"] != LOCKED_THRESHOLD:
        raise ValueError("MuSiQue v66 threshold changed")
    return value


def validate_source_registration(
    path: Path,
    *,
    target_path: Path,
    squad2_train_path: Path,
    invalid_map_path: Path,
    corrected_map_path: Path,
) -> dict[str, Any]:
    if sha256(path) != SOURCE_REGISTRATION_SHA256:
        raise ValueError("MuSiQue v66 source registration changed")
    expected = {
        target_path: TARGET_SOURCE_SHA256,
        squad2_train_path: SQUAD2_TRAIN_SHA256,
        invalid_map_path: V65_INVALID_MAP_SHA256,
        corrected_map_path: V65_CORRECTED_MAP_SHA256,
    }
    for source, expected_hash in expected.items():
        if sha256(source) != expected_hash:
            raise ValueError(f"MuSiQue v66 source changed: {source}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["target"]["v66_source_commitments_selected"] != 0:
        raise ValueError("MuSiQue v66 source commitments predate registration")
    return value


def extract_squad2_questions(source: dict[str, Any]) -> set[str]:
    result = {
        normalize_question(str(qa.get("question", "")))
        for article in source.get("data", [])
        for paragraph in article.get("paragraphs", [])
        for qa in paragraph.get("qas", [])
        if str(qa.get("question", "")).strip()
    }
    if not result:
        raise ValueError("MuSiQue v66 SQuAD2 question guard is empty")
    return result


def load_source_commitments(rows: Iterable[dict[str, Any]]) -> set[str]:
    result = {str(row.get("source_id_commitment", "")) for row in rows}
    result.discard("")
    if not result:
        raise ValueError("MuSiQue v66 source commitment input is empty")
    return result


def _valid_row(row: dict[str, Any]) -> tuple[bool, str]:
    if not str(row.get("id", "")).strip():
        return False, "missing_id"
    if not str(row.get("question", "")).strip():
        return False, "missing_question"
    if not isinstance(row.get("answerable"), bool):
        return False, "invalid_answerable"
    decomposition = row.get("question_decomposition")
    if not isinstance(decomposition, list) or not 2 <= len(decomposition) <= 4:
        return False, "invalid_hop_count"
    if any(not str(item.get("question", "")).strip() for item in decomposition):
        return False, "invalid_decomposition_question"
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or len(paragraphs) < 3:
        return False, "insufficient_paragraphs"
    indices: set[int] = set()
    for paragraph in paragraphs:
        try:
            index = int(paragraph["idx"])
        except (KeyError, TypeError, ValueError):
            return False, "invalid_paragraph_idx"
        if index in indices:
            return False, "duplicate_paragraph_idx"
        indices.add(index)
        if not str(paragraph.get("paragraph_text", "")).strip():
            return False, "empty_paragraph"
    return True, "eligible"


def _has_squad_overlap(row: dict[str, Any], squad_questions: set[str]) -> bool:
    questions = [str(row["question"])] + [
        str(item["question"]) for item in row["question_decomposition"]
    ]
    return any(normalize_question(question) in squad_questions for question in questions)


def select_stage_sample(
    rows: Iterable[dict[str, Any]],
    *,
    stage: str,
    squad_questions: set[str],
    excluded_source_commitments: set[str],
    target_per_group: int = TARGET_PER_GROUP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGE_SALTS:
        raise ValueError(f"Unsupported MuSiQue v66 stage: {stage}")
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
        valid, reason = _valid_row(row)
        if not valid:
            schema_reasons[reason] = schema_reasons.get(reason, 0) + 1
            continue
        if _has_squad_overlap(row, squad_questions):
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
            if int(_hash(PAIR_ASSIGNMENT_SALT + source_id), 16) % 2 == 0
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
        if len(retained[group]) > target_per_group:
            retained[group].pop()
    if any(len(values) != target_per_group for values in retained.values()):
        raise ValueError("MuSiQue v66 has insufficient eligible balanced cases")
    selected_items = sorted(
        [item for values in retained.values() for item in values],
        key=lambda item: (item[0], item[1]),
    )
    selected = [item[2] for item in selected_items]
    selected_ids = {str(row["id"]) for row in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("MuSiQue v66 sample contains duplicate source ids")
    if any(_hash(value) in excluded_source_commitments for value in selected_ids):
        raise ValueError("MuSiQue v66 sample overlaps an excluded commitment")
    schema_excluded = sum(schema_reasons.values())
    return selected, {
        "stage": stage,
        "source_rows": source_rows,
        "target_cases": target_per_group * 2,
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
    prefix = "m66d" if stage == "development" else "m66c"
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
                    "schema_version": "frc-musique-v66-direct-qa-input-v1",
                    "id": f"{case_id}::direct::p{paragraph_idx}",
                    "case_id": case_id,
                    "paragraph_idx": paragraph_idx,
                    "question": str(source["question"]),
                    "context": context,
                    "gold_fields_visible_to_verifier": False,
                }
            )
        templates = [
            str(item["question"]) for item in source["question_decomposition"]
        ]
        prepared.append(
            {
                "schema_version": "frc-musique-v66-prepared-blind-v1",
                "id": case_id,
                "composed_question": str(source["question"]),
                "oracle_hop_templates": templates,
                "contexts": contexts,
                "gold_fields_visible_to_chain_executor": False,
            }
        )
        maps.append(
            {
                "schema_version": "frc-musique-v66-candidate-map-v1",
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
        raise ValueError("MuSiQue v66 blind caches expose forbidden gold fields")
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


class LocalRobertaQASupportVerifier(V65LocalRobertaQASupportVerifier):
    def verify(
        self, rows: Sequence[dict[str, Any]], *, cache_key: str
    ) -> list[dict[str, Any]]:
        result = super().verify(rows, cache_key=cache_key)
        for row in result:
            row["schema_version"] = "frc-musique-v66-raw-qa-v1"
        return result


def aggregate_qa_rows(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> list[dict[str, Any]]:
    if [str(row["id"]) for row in qa_inputs] != [str(row["id"]) for row in decisions]:
        raise ValueError("MuSiQue v66 QA cache is incomplete or out of order")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for qa_input, decision in zip(qa_inputs, decisions, strict=True):
        if decision.get("cache_key") != cache_key:
            raise ValueError("MuSiQue v66 QA cache key changed")
        grouped.setdefault(str(qa_input["case_id"]), []).append((qa_input, decision))
    result: list[dict[str, Any]] = []
    for case_id, rows in grouped.items():
        valid = [
            pair
            for pair in rows
            if not pair[1].get("invalid_fail_closed_used")
            and pair[1].get("score_margin") is not None
            and math.isfinite(float(pair[1]["score_margin"]))
        ]
        valid.sort(
            key=lambda pair: (
                -float(pair[1]["score_margin"]),
                int(pair[0]["paragraph_idx"]),
            )
        )
        winner = valid[0] if valid else None
        margin = float(winner[1]["score_margin"]) if winner else None
        predicted_span = ""
        if winner:
            start = winner[1].get("span_start")
            end = winner[1].get("span_end")
            context = str(winner[0]["context"])
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(context):
                predicted_span = context[start:end].strip()
        support_passed = bool(
            winner
            and margin is not None
            and margin > LOCKED_THRESHOLD
            and predicted_span
        )
        payload = {
            "case_id": case_id,
            "margin": margin,
            "paragraph_idx": int(winner[0]["paragraph_idx"]) if winner else None,
            "predicted_span_sha256": _hash(predicted_span),
            "support_passed": support_passed,
        }
        result.append(
            {
                "schema_version": "frc-musique-v66-aggregated-qa-v1",
                "case_id": case_id,
                "paragraph_rows": len(rows),
                "invalid_paragraph_rows": len(rows) - len(valid),
                "invalid_fail_closed_used": winner is None,
                "score_margin": margin,
                "selected_paragraph_idx": payload["paragraph_idx"],
                "predicted_span": predicted_span,
                "predicted_span_sha256": payload["predicted_span_sha256"],
                "support_passed": support_passed,
                "decision_sha256": _hash(
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                ),
            }
        )
    return result


_PLACEHOLDER = re.compile(r"#([1-9][0-9]*)")


def resolve_placeholders(
    template: str, predictions: dict[int, str], *, hop_index: int
) -> tuple[str | None, list[int]]:
    referenced = sorted({int(value) for value in _PLACEHOLDER.findall(template)})
    missing = [
        value
        for value in referenced
        if value >= hop_index or not str(predictions.get(value, "")).strip()
    ]
    if missing:
        return None, missing
    resolved = _PLACEHOLDER.sub(
        lambda match: str(predictions[int(match.group(1))]).strip(), template
    )
    resolved = re.sub(r"\s+", " ", resolved).strip()
    return (resolved or None), []


def initial_chain_states(
    prepared: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): {
            "alive": True,
            "predictions": {},
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
            blocked[case_id] = "predecessor_hop_unsupported"
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
                    "schema_version": "frc-musique-v66-hop-qa-input-v1",
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
        result = by_case.get(case_id)
        reason = blocked.get(case_id)
        if reason is not None:
            state["alive"] = False
            passed = False
            executed = False
            margin = None
            paragraph_idx = None
            span_hash = _hash("")
            invalid = False
        elif result is None:
            raise ValueError("MuSiQue v66 active hop case lacks an aggregate")
        else:
            executed = True
            passed = bool(result["support_passed"])
            margin = result["score_margin"]
            paragraph_idx = result["selected_paragraph_idx"]
            span_hash = str(result["predicted_span_sha256"])
            invalid = bool(result["invalid_fail_closed_used"])
            if invalid:
                state["invalid_output"] = True
            if passed:
                state["predictions"][hop_index] = str(result["predicted_span"])
                state["completed_hops"] += 1
            else:
                state["alive"] = False
            reason = None if passed else "hop_margin_not_above_threshold"
        decisions.append(
            {
                "schema_version": "frc-musique-v66-hop-decision-v1",
                "case_id": case_id,
                "hop_index": hop_index,
                "executed": executed,
                "support_passed": passed,
                "failure_reason": reason,
                "score_margin": margin,
                "selected_paragraph_idx": paragraph_idx,
                "predicted_span_sha256": span_hash,
                "invalid_fail_closed_used": invalid,
            }
        )
    return decisions


def finalize_chain_decisions(
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
            raise ValueError("MuSiQue v66 chain decision count changed")
        state = states[case_id]
        passed = bool(
            state["alive"] and int(state["completed_hops"]) == expected_hops
        )
        result.append(
            {
                "schema_version": "frc-musique-v66-chain-decision-v1",
                "case_id": case_id,
                "hop_count": expected_hops,
                "executed_hops": sum(bool(item["executed"]) for item in decisions),
                "passed_hops": sum(bool(item["support_passed"]) for item in decisions),
                "support_passed": passed,
                "invalid_fail_closed_used": bool(state["invalid_output"]),
                "hop_decision_sha256": _hash(
                    json.dumps(decisions, sort_keys=True, separators=(",", ":"))
                ),
            }
        )
    return result


def build_gold_evidence(
    selected: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    direct_decisions: Sequence[dict[str, Any]],
    chain_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (
        len(selected)
        == len(prepared)
        == len(direct_decisions)
        == len(chain_decisions)
    ):
        raise ValueError("MuSiQue v66 gold join inputs are incomplete")
    evidence: list[dict[str, Any]] = []
    for source, blind, direct, chain in zip(
        selected, prepared, direct_decisions, chain_decisions, strict=True
    ):
        case_id = str(blind["id"])
        if case_id != str(direct["case_id"]) or case_id != str(chain["case_id"]):
            raise ValueError("MuSiQue v66 gold join order changed")
        answerable = bool(source["answerable"])
        evidence.append(
            {
                "schema_version": "frc-musique-v66-support-evidence-v1",
                "case_id": case_id,
                "answer_state": "answerable" if answerable else "unanswerable",
                "hop_count": len(source["question_decomposition"]),
                "paragraph_count": len(source["paragraphs"]),
                "direct_support_passed": bool(direct["support_passed"]),
                "direct_score_margin": direct["score_margin"],
                "direct_prediction_sha256": str(direct["decision_sha256"]),
                "sequential_chain_support_passed": bool(chain["support_passed"]),
                "sequential_executed_hops": int(chain["executed_hops"]),
                "sequential_passed_hops": int(chain["passed_hops"]),
                "sequential_hop_decision_sha256": str(
                    chain["hop_decision_sha256"]
                ),
                "invalid_chain_output": bool(
                    direct["invalid_fail_closed_used"]
                    or chain["invalid_fail_closed_used"]
                ),
            }
        )
    return evidence


def _method_metrics(
    evidence: Sequence[dict[str, Any]], *, decision_field: str
) -> tuple[dict[str, float | int], np.ndarray]:
    answerable = [row for row in evidence if row["answer_state"] == "answerable"]
    unanswerable = [row for row in evidence if row["answer_state"] == "unanswerable"]
    answer_pass = np.mean([bool(row[decision_field]) for row in answerable])
    noanswer_reject = np.mean([not bool(row[decision_field]) for row in unanswerable])
    correctness = np.asarray(
        [
            bool(row[decision_field])
            if row["answer_state"] == "answerable"
            else not bool(row[decision_field])
            for row in evidence
        ],
        dtype=float,
    )
    return {
        "rows": len(evidence),
        "answerable_rows": len(answerable),
        "unanswerable_rows": len(unanswerable),
        "answerable_support_pass_rate": round(float(answer_pass), 6),
        "unanswerable_rejection_rate": round(float(noanswer_reject), 6),
        "balanced_accuracy": round(float((answer_pass + noanswer_reject) / 2), 6),
    }, correctness


def paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    resamples: int = 10000,
) -> dict[str, float | int]:
    if candidate.shape != baseline.shape or candidate.ndim != 1 or not len(candidate):
        raise ValueError("MuSiQue v66 paired bootstrap inputs are invalid")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        draws[start:stop] = delta[indices].mean(axis=1)
    return {
        "point": round(float(delta.mean()), 6),
        "ci_low": round(float(np.quantile(draws, 0.025)), 6),
        "ci_high": round(float(np.quantile(draws, 0.975)), 6),
        "resamples": resamples,
        "seed": seed,
    }


def evaluate_stage(
    evidence: Sequence[dict[str, Any]],
    *,
    stage: str,
    sampling: dict[str, Any],
    structural_census: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> dict[str, Any]:
    direct, direct_correct = _method_metrics(
        evidence, decision_field="direct_support_passed"
    )
    chain, chain_correct = _method_metrics(
        evidence, decision_field="sequential_chain_support_passed"
    )
    comparison = paired_bootstrap(
        chain_correct, direct_correct, seed=BOOTSTRAP_SEEDS[stage]
    )
    unanswerable = [row for row in evidence if row["answer_state"] == "unanswerable"]
    traps = [row for row in unanswerable if row["direct_support_passed"]]
    trap_rejection = (
        np.mean([not row["sequential_chain_support_passed"] for row in traps])
        if traps
        else 0.0
    )
    hop_strata: dict[str, dict[str, float | int]] = {}
    for hop in sorted({int(row["hop_count"]) for row in unanswerable}):
        rows = [row for row in unanswerable if int(row["hop_count"]) == hop]
        hop_strata[str(hop)] = {
            "cases": len(rows),
            "rejection_rate": round(
                float(
                    np.mean(
                        [not row["sequential_chain_support_passed"] for row in rows]
                    )
                ),
                6,
            ),
        }
    invalid_count = sum(bool(row["invalid_chain_output"]) for row in evidence)
    checks = {
        "exact_cases_equals_600": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": (
            direct["answerable_rows"] == TARGET_PER_GROUP
            and direct["unanswerable_rows"] == TARGET_PER_GROUP
        ),
        "schema_exclusion_rate_at_most_0_01": sampling["schema_exclusion_rate"]
        <= 0.01,
        "selected_v65_source_overlap_equals_0": sampling[
            "selected_excluded_source_commitment_overlap"
        ]
        == 0,
        "selected_squad2_exact_question_overlap_equals_0": sampling[
            "selected_squad2_exact_question_overlap"
        ]
        == 0,
        "invalid_chain_output_rate_equals_0": invalid_count == 0,
        "sequential_chain_balanced_accuracy_at_least_0_72": chain[
            "balanced_accuracy"
        ]
        >= 0.72,
        "sequential_chain_answerable_pass_rate_at_least_0_60": chain[
            "answerable_support_pass_rate"
        ]
        >= 0.60,
        "sequential_chain_unanswerable_rejection_rate_at_least_0_80": chain[
            "unanswerable_rejection_rate"
        ]
        >= 0.80,
        "sequential_minus_direct_balanced_accuracy_at_least_0_10": comparison[
            "point"
        ]
        >= 0.10,
        "sequential_minus_direct_correctness_ci_low_above_0_05": comparison[
            "ci_low"
        ]
        > 0.05,
        "direct_passed_unanswerable_trap_cases_at_least_100": len(traps) >= 100,
        "sequential_chain_trap_rejection_rate_at_least_0_65": float(
            trap_rejection
        )
        >= 0.65,
        "every_observed_unanswerable_hop_stratum_rejection_rate_at_least_0_75": all(
            float(value["rejection_rate"]) >= 0.75 for value in hop_strata.values()
        ),
    }
    passed = all(checks.values())
    if stage == "development":
        status = DEVELOPMENT_PASS if passed else DEVELOPMENT_FAILURE
    else:
        status = CONFIRMATION_PASS if passed else CONFIRMATION_FAILURE
    return {
        "schema_version": "frc-musique-v66-stage-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "stage": stage,
            "dataset_id": DATASET_ID,
            "cases": len(evidence),
            "answer_state_counts": {
                "answerable": direct["answerable_rows"],
                "unanswerable": direct["unanswerable_rows"],
            },
            "official_musique_leaderboard_result": False,
            "automatic_decomposer_result": False,
            "strict_independent_model_training_confirmation": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "direct_composed_question": direct,
            "oracle_plan_sequential_chain": chain,
            "paired_correctness_delta": comparison,
            "direct_passed_unanswerable_traps": {
                "cases": len(traps),
                "sequential_rejection_rate": round(float(trap_rejection), 6),
            },
            "unanswerable_hop_strata": hop_strata,
            "invalid_chain_output_count": invalid_count,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": bool(
                    stage == "development" and passed
                ),
                "reuse_failed_stage_for_tuning_threshold_rule_gate_or_selection": False,
                "selector_or_retrieval_scoring_part_of_v66": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }


def write_stage_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    result_path: Path,
    report_path: Path,
    evidence_path: Path,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = report["analysis"]
    outcome = analysis["outcome"]
    lines = [
        f"# MuSiQue sequential chain support ({report['metadata']['stage']}, v66)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases: {report['metadata']['cases']}",
        f"- Direct balanced accuracy: {analysis['direct_composed_question']['balanced_accuracy']:.6f}",
        f"- Sequential balanced accuracy: {analysis['oracle_plan_sequential_chain']['balanced_accuracy']:.6f}",
        f"- Paired correctness delta: {analysis['paired_correctness_delta']['point']:+.6f} "
        f"(95% CI [{analysis['paired_correctness_delta']['ci_low']:+.6f}, "
        f"{analysis['paired_correctness_delta']['ci_high']:+.6f}])",
        f"- Direct-passed unanswerable traps: {analysis['direct_passed_unanswerable_traps']['cases']}",
        f"- Sequential trap rejection: {analysis['direct_passed_unanswerable_traps']['sequential_rejection_rate']:.6f}",
        "- Official decomposition questions are an oracle plan; gold decomposition answers and supporting labels were not visible to execution.",
        "- This is not an official leaderboard, automatic decomposer, FRC retrieval, SetR, flood-domain, or production result.",
        "- Gate 2 remains `NO-GO/SHADOW`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    with gzip.GzipFile(filename=str(evidence_path), mode="wb", mtime=0) as raw:
        for row in evidence:
            raw.write(
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
