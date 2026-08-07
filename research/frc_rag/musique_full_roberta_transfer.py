"""Prospective MuSiQue-Full RoBERTa support-transfer experiment (v65)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from research.frc_rag.squad2_calibrated_roberta_support import (
    LocalRobertaQASupportVerifier as V64LocalRobertaQASupportVerifier,
)


EXPERIMENT_ID = "FRC-MUSIQUE-FULL-ROBERTA-TRANSFER-V65"
DATASET_ID = "musique_full_v1.0_train_v65_confirmation_role"
SAMPLE_SALT = "FRC-MUSIQUE-FULL-V65|"
CORRECTED_SAMPLE_SALT = "FRC-MUSIQUE-FULL-V65-CORRECTED|"
PAIR_ASSIGNMENT_SALT = "FRC-MUSIQUE-FULL-V65-PAIR-ASSIGNMENT|"
TARGET_CASES = 600
TARGET_PER_GROUP = 300
LOCKED_THRESHOLD = 0.974609375
MODEL_REPOSITORY = "deepset/roberta-base-squad2"
MODEL_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
PROTOCOL_SHA256 = "578bd51f20cdec041da034c87e5f3890721c9f317de511574474e07692514a38"
SOURCE_REGISTRATION_SHA256 = (
    "c29637ad608af431bed6bb0992cb0b77fc9da8aa96a20b684fb33176837a36f7"
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
V36_PREPARED_SHA256 = (
    "b9a987875dbc7aeaef4720c6df02265d0368b49f344b8a24b933a10ca0cfc28b"
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
    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_surface(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def validate_protocol(path: Path) -> dict[str, Any]:
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("MuSiQue v65 protocol hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("MuSiQue v65 protocol experiment changed")
    if value["prior_boundary"]["v65_target_content_opened_or_parsed_before_this_protocol"]:
        raise ValueError("MuSiQue v65 target was opened before protocol freeze")
    if value["support_transfer"]["locked_threshold_exact"] != LOCKED_THRESHOLD:
        raise ValueError("MuSiQue v65 threshold changed")
    return value


def validate_source_registration(
    path: Path,
    *,
    target_path: Path,
    squad2_train_path: Path,
    v36_prepared_path: Path,
) -> dict[str, Any]:
    if sha256(path) != SOURCE_REGISTRATION_SHA256:
        raise ValueError("MuSiQue v65 source registration changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        target_path: TARGET_SOURCE_SHA256,
        squad2_train_path: SQUAD2_TRAIN_SHA256,
        v36_prepared_path: V36_PREPARED_SHA256,
    }
    for source_path, expected_hash in expected.items():
        if sha256(source_path) != expected_hash:
            raise ValueError(f"MuSiQue v65 source changed: {source_path}")
    if value["target"]["content_opened_or_parsed_before_registration"] is not False:
        raise ValueError("MuSiQue v65 source boundary changed")
    return value


def extract_squad2_train_questions(source: dict[str, Any]) -> set[str]:
    questions: set[str] = set()
    for article in source.get("data", []):
        for paragraph in article.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                value = normalize_question(str(qa.get("question", "")))
                if value:
                    questions.add(value)
    if not questions:
        raise ValueError("MuSiQue v65 SQuAD2 leakage guard is empty")
    return questions


def load_v36_source_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        value = str(row.get("id", ""))
        if value.startswith("musique::"):
            value = value.removeprefix("musique::")
        if value:
            result.add(value)
    if not result:
        raise ValueError("MuSiQue v65 v36 exclusion ids are empty")
    return result


def _valid_row(row: dict[str, Any]) -> tuple[bool, str]:
    if not str(row.get("id", "")).strip():
        return False, "missing_id"
    if not str(row.get("question", "")).strip():
        return False, "missing_question"
    if not isinstance(row.get("answerable"), bool):
        return False, "invalid_answerable"
    decomposition = row.get("question_decomposition")
    if not isinstance(decomposition, list) or not decomposition:
        return False, "missing_decomposition"
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
    values = [str(row["question"])]
    values.extend(str(item["question"]) for item in row["question_decomposition"])
    return any(normalize_question(value) in squad_questions for value in values)


def select_balanced_sample(
    rows: Iterable[dict[str, Any]],
    *,
    squad_questions: set[str],
    v36_source_ids: set[str],
    excluded_source_commitments: set[str] | None = None,
    target_per_group: int = TARGET_PER_GROUP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_source_commitments = excluded_source_commitments or set()
    retained: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
        "answerable": [],
        "unanswerable": [],
    }
    source_rows = 0
    eligible_counts = {"answerable": 0, "unanswerable": 0}
    schema_exclusion_reasons: dict[str, int] = {}
    squad_overlap = 0
    v36_overlap = 0
    invalid_run_overlap = 0
    pair_assignment_excluded = 0
    duplicate_assigned_source_ids = 0
    seen_assigned_source_ids: set[str] = set()
    for row in rows:
        source_rows += 1
        valid, reason = _valid_row(row)
        if not valid:
            schema_exclusion_reasons[reason] = schema_exclusion_reasons.get(reason, 0) + 1
            continue
        source_id = str(row["id"])
        if source_id in v36_source_ids:
            v36_overlap += 1
            continue
        if _has_squad_overlap(row, squad_questions):
            squad_overlap += 1
            continue
        group = "answerable" if row["answerable"] else "unanswerable"
        source_commitment = _hash(source_id)
        if source_commitment in excluded_source_commitments:
            invalid_run_overlap += 1
            continue
        assigned_group = (
            "answerable"
            if int(_hash(PAIR_ASSIGNMENT_SALT + source_id), 16) % 2 == 0
            else "unanswerable"
        )
        if group != assigned_group:
            pair_assignment_excluded += 1
            continue
        if source_id in seen_assigned_source_ids:
            duplicate_assigned_source_ids += 1
            continue
        seen_assigned_source_ids.add(source_id)
        eligible_counts[group] += 1
        key = _hash(CORRECTED_SAMPLE_SALT + source_id)
        retained[group].append((key, source_id, row))
        retained[group].sort(key=lambda item: (item[0], item[1]))
        if len(retained[group]) > target_per_group:
            retained[group].pop()
    if any(len(retained[group]) != target_per_group for group in retained):
        raise ValueError("MuSiQue v65 has insufficient eligible balanced cases")
    selected_items = sorted(
        [item for values in retained.values() for item in values],
        key=lambda item: (item[0], item[1]),
    )
    selected = [item[2] for item in selected_items]
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise ValueError("MuSiQue v65 corrected sample still has duplicate source ids")
    schema_excluded = sum(schema_exclusion_reasons.values())
    return selected, {
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
        "schema_exclusion_reasons": schema_exclusion_reasons,
        "squad2_exact_question_overlap_excluded": squad_overlap,
        "selected_squad2_exact_question_overlap": 0,
        "v36_source_id_overlap_excluded": v36_overlap,
        "selected_v36_source_id_overlap": 0,
        "invalid_run_source_rows_excluded": invalid_run_overlap,
        "selected_invalid_run_source_overlap": 0,
        "pair_assignment_excluded_rows": pair_assignment_excluded,
        "duplicate_assigned_source_rows_excluded": duplicate_assigned_source_ids,
        "pair_assignment_rule": "SHA-256 parity assigns each source id to one answer state",
        "selected_source_id_commitment": _hash(
            "\n".join(sorted(str(row["id"]) for row in selected))
        ),
    }


def prepare_blind_qa(
    selected: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    qa_inputs: list[dict[str, Any]] = []
    paragraph_counts: list[int] = []
    for row in selected:
        source_id = str(row["id"])
        case_id = f"m65-{_hash(source_id)[:20]}"
        candidates = []
        mapping_paragraphs = []
        for paragraph in sorted(row["paragraphs"], key=lambda item: int(item["idx"])):
            index = int(paragraph["idx"])
            candidate_id = f"p{index}"
            title = str(paragraph.get("title", "")).strip()
            text = str(paragraph["paragraph_text"]).strip()
            context = f"{title}\n{text}" if title else text
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "paragraph_idx": index,
                    "context": context,
                    "context_sha256": _hash(context),
                }
            )
            mapping_paragraphs.append(
                {
                    "candidate_id": candidate_id,
                    "paragraph_idx": index,
                    "context_sha256": _hash(context),
                }
            )
            qa_inputs.append(
                {
                    "schema_version": "frc-musique-v65-qa-input-v1",
                    "id": f"{case_id}::{candidate_id}",
                    "case_id": case_id,
                    "paragraph_idx": index,
                    "question": str(row["question"]),
                    "context": context,
                    "gold_fields_visible_to_verifier": False,
                }
            )
        paragraph_counts.append(len(candidates))
        prepared.append(
            {
                "schema_version": "frc-musique-v65-prepared-blind-v1",
                "id": case_id,
                "question": str(row["question"]),
                "candidates": candidates,
                "gold_fields_visible_to_verifier": False,
            }
        )
        maps.append(
            {
                "schema_version": "frc-musique-v65-candidate-map-v1",
                "id": case_id,
                "source_id_commitment": _hash(source_id),
                "paragraphs": mapping_paragraphs,
            }
        )
    forbidden = {
        "answerable",
        "answer",
        "answer_aliases",
        "question_decomposition",
        "is_supporting",
        "paragraph_support_idx",
    }
    if forbidden & nested_keys([prepared, maps, qa_inputs]):
        raise ValueError("MuSiQue v65 blind cache exposes gold fields")
    return prepared, maps, qa_inputs, {
        "prepared_cases": len(prepared),
        "qa_input_rows": len(qa_inputs),
        "minimum_paragraphs": min(paragraph_counts, default=0),
        "maximum_paragraphs": max(paragraph_counts, default=0),
        "mean_paragraphs": sum(paragraph_counts) / len(paragraph_counts)
        if paragraph_counts
        else 0.0,
        "gold_fields_exported_to_blind_caches": False,
    }


def nested_keys(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        return set(payload) | {
            key for value in payload.values() for key in nested_keys(value)
        }
    if isinstance(payload, list):
        return {key for value in payload for key in nested_keys(value)}
    return set()


class LocalRobertaQASupportVerifier(V64LocalRobertaQASupportVerifier):
    def verify(
        self, rows: Sequence[dict[str, Any]], *, cache_key: str
    ) -> list[dict[str, Any]]:
        result = super().verify(rows, cache_key=cache_key)
        for row in result:
            row["schema_version"] = "frc-musique-v65-raw-paragraph-qa-v1"
        return result


def aggregate_paragraph_decisions(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> list[dict[str, Any]]:
    if [str(row["id"]) for row in qa_inputs] != [str(row["id"]) for row in decisions]:
        raise ValueError("MuSiQue v65 paragraph QA cache is incomplete or out of order")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for qa_input, decision in zip(qa_inputs, decisions, strict=True):
        if decision.get("cache_key") != cache_key:
            raise ValueError("MuSiQue v65 paragraph QA cache key changed")
        grouped.setdefault(str(qa_input["case_id"]), []).append((qa_input, decision))
    result: list[dict[str, Any]] = []
    for case_id, rows in grouped.items():
        valid = [
            (qa_input, decision)
            for qa_input, decision in rows
            if not decision.get("invalid_fail_closed_used")
            and decision.get("score_margin") is not None
            and math.isfinite(float(decision["score_margin"]))
        ]
        valid.sort(
            key=lambda item: (
                -float(item[1]["score_margin"]),
                int(item[0]["paragraph_idx"]),
            )
        )
        winner = valid[0] if valid else None
        margin = float(winner[1]["score_margin"]) if winner else None
        support_passed = bool(winner and margin is not None and margin > LOCKED_THRESHOLD)
        digest_payload = {
            "case_id": case_id,
            "score_margin": margin,
            "selected_paragraph_idx": int(winner[0]["paragraph_idx"])
            if winner
            else None,
            "span_sha256": winner[1].get("span_sha256") if winner else None,
            "threshold": LOCKED_THRESHOLD,
            "support_passed": support_passed,
        }
        result.append(
            {
                "schema_version": "frc-musique-v65-case-support-decision-v1",
                "case_id": case_id,
                "paragraph_rows": len(rows),
                "invalid_paragraph_rows": len(rows) - len(valid),
                "invalid_fail_closed_used": winner is None,
                "score_margin": margin,
                "selected_paragraph_idx": digest_payload["selected_paragraph_idx"],
                "support_passed": support_passed,
                "locked_threshold": LOCKED_THRESHOLD,
                "raw_prediction_sha256": _hash(
                    json.dumps(digest_payload, sort_keys=True, separators=(",", ":"))
                ),
            }
        )
    return result


def _surface_present(row: dict[str, Any]) -> bool:
    surfaces = [str(row.get("answer", ""))]
    surfaces.extend(str(value) for value in row.get("answer_aliases", []))
    normalized = [normalize_surface(value) for value in surfaces]
    normalized = [value for value in normalized if len(value) >= 2]
    pool = normalize_surface(
        " ".join(str(paragraph.get("paragraph_text", "")) for paragraph in row["paragraphs"])
    )
    return any(value in pool for value in normalized)


def build_gold_evidence(
    selected: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(selected) == len(prepared) == len(decisions)):
        raise ValueError("MuSiQue v65 gold join inputs are incomplete")
    evidence: list[dict[str, Any]] = []
    for source, blind, decision in zip(selected, prepared, decisions, strict=True):
        if str(blind["id"]) != str(decision["case_id"]):
            raise ValueError("MuSiQue v65 gold join order changed")
        answerable = bool(source["answerable"])
        evidence.append(
            {
                "schema_version": "frc-musique-v65-support-evidence-v1",
                "case_id": str(blind["id"]),
                "answer_state": "answerable" if answerable else "unanswerable",
                "hop_count": len(source["question_decomposition"]),
                "paragraph_count": len(source["paragraphs"]),
                "local_span_trap": bool(not answerable and _surface_present(source)),
                "score_margin": decision["score_margin"],
                "selected_paragraph_idx": decision["selected_paragraph_idx"],
                "support_passed": bool(decision["support_passed"]),
                "invalid_fail_closed_used": bool(
                    decision["invalid_fail_closed_used"]
                ),
                "raw_prediction_sha256": str(decision["raw_prediction_sha256"]),
            }
        )
    return evidence


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_support_gate(
    evidence: Sequence[dict[str, Any]],
    *,
    sampling: dict[str, Any],
    structural_census: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> dict[str, Any]:
    answerable = [row for row in evidence if row["answer_state"] == "answerable"]
    unanswerable = [row for row in evidence if row["answer_state"] == "unanswerable"]
    answer_pass = _rate([bool(row["support_passed"]) for row in answerable])
    noanswer_reject = _rate([not bool(row["support_passed"]) for row in unanswerable])
    balanced_accuracy = (answer_pass + noanswer_reject) / 2
    traps = [row for row in unanswerable if row["local_span_trap"]]
    trap_reject = _rate([not bool(row["support_passed"]) for row in traps])
    hop_strata: dict[str, dict[str, Any]] = {}
    for hop in sorted({int(row["hop_count"]) for row in unanswerable}):
        rows = [row for row in unanswerable if int(row["hop_count"]) == hop]
        hop_strata[str(hop)] = {
            "cases": len(rows),
            "rejection_rate": _rate(
                [not bool(row["support_passed"]) for row in rows]
            ),
        }
    invalid_cases = sum(bool(row["invalid_fail_closed_used"]) for row in evidence)
    checks = {
        "exact_cases_equals_600": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": len(answerable) == len(unanswerable) == TARGET_PER_GROUP,
        "schema_exclusion_rate_at_most_0_01": sampling["schema_exclusion_rate"] <= 0.01,
        "selected_squad2_exact_question_overlap_equals_0": sampling[
            "selected_squad2_exact_question_overlap"
        ]
        == 0,
        "selected_v36_source_id_overlap_equals_0": sampling[
            "selected_v36_source_id_overlap"
        ]
        == 0,
        "selected_invalid_run_source_overlap_equals_0": sampling[
            "selected_invalid_run_source_overlap"
        ]
        == 0,
        "minimum_paragraphs_at_least_3": structural_census["minimum_paragraphs"]
        >= 3,
        "qa_invalid_case_rate_equals_0": invalid_cases == 0,
        "balanced_accuracy_at_least_0_8": balanced_accuracy >= 0.8,
        "answerable_support_pass_rate_at_least_0_8": answer_pass >= 0.8,
        "unanswerable_rejection_rate_at_least_0_7": noanswer_reject >= 0.7,
        "local_span_trap_cases_at_least_100": len(traps) >= 100,
        "local_span_trap_rejection_rate_at_least_0_65": trap_reject >= 0.65,
        "every_unanswerable_hop_stratum_rejection_rate_at_least_0_6": bool(
            hop_strata
        )
        and all(row["rejection_rate"] >= 0.6 for row in hop_strata.values()),
    }
    support_established = all(checks.values())
    status = (
        "MUSIQUE_FULL_V65_CROSS_DOMAIN_SUPPORT_ESTABLISHED_OPEN_RETRIEVAL"
        if support_established
        else "MUSIQUE_FULL_V65_ROBERTA_TRANSFER_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_RETRIEVAL"
    )
    return {
        "schema_version": "frc-musique-v65-support-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "stage": "pre_retrieval_support_gate",
            "cases": len(evidence),
            "answer_state_counts": {
                "answerable": len(answerable),
                "unanswerable": len(unanswerable),
            },
            "source_artifacts": source_artifacts,
            "strict_independent_model_training_confirmation": False,
            "official_musique_leaderboard_result": False,
        },
        "analysis": {
            "support_verifier": {
                "rows": len(evidence),
                "answer_bearing_support_pass_rate": answer_pass,
                "no_answer_rejection_rate": noanswer_reject,
                "balanced_accuracy": balanced_accuracy,
                "invalid_output_count": invalid_cases,
                "invalid_output_rate": invalid_cases / len(evidence)
                if evidence
                else 1.0,
                "locked_threshold": LOCKED_THRESHOLD,
                "gold_fields_visible_to_verifier": False,
            },
            "local_span_trap": {
                "cases": len(traps),
                "unanswerable_rejection_rate": trap_reject,
            },
            "unanswerable_hop_strata": hop_strata,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_gate_passed": support_established,
                "retrieval_scoring_open_authorized": support_established,
                "reuse_v65_for_tuning_threshold_or_selection": False,
                "strict_independent_model_training_confirmation_claimed": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }


def _round_for_display(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _round_for_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_for_display(item) for item in value]
    return value


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    rounded = _round_for_display(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    verifier = analysis["support_verifier"]
    outcome = analysis["outcome"]
    lines = [
        "# MuSiQue-Full calibrated RoBERTa transfer support gate (v65)",
        "",
        f"- status: `{outcome['status']}`",
        f"- cases: {rounded['metadata']['cases']}",
        f"- balanced accuracy: {verifier['balanced_accuracy']:.6f}",
        f"- answerable pass rate: {verifier['answer_bearing_support_pass_rate']:.6f}",
        f"- unanswerable rejection rate: {verifier['no_answer_rejection_rate']:.6f}",
        f"- local-span-trap rejection rate: {analysis['local_span_trap']['unanswerable_rejection_rate']:.6f}",
        f"- retrieval scoring authorized: `{outcome['retrieval_scoring_open_authorized']}`",
        f"- selector adoption authorized: `{outcome['selector_adoption_authorized']}`",
        f"- Gate 2: `{outcome['gate_2']}`",
        "",
        "> This is a balanced MuSiQue-Full cross-domain mechanism test with an exact SQuAD-question leakage guard. It is not an official leaderboard result, strict independent model-training confirmation, proof of complete multi-hop reasoning, open-corpus retrieval, SetR reproduction, flood-domain validation or production evidence.",
        "",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            _round_for_display(row),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


__all__ = [
    "DATASET_ID",
    "EXPERIMENT_ID",
    "LOCKED_THRESHOLD",
    "LocalRobertaQASupportVerifier",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "TARGET_CASES",
    "TARGET_PER_GROUP",
    "aggregate_paragraph_decisions",
    "build_gold_evidence",
    "evaluate_support_gate",
    "extract_squad2_train_questions",
    "load_v36_source_ids",
    "normalize_question",
    "prepare_blind_qa",
    "select_balanced_sample",
    "sha256",
    "validate_protocol",
    "validate_source_registration",
    "write_report",
]
