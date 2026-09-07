"""Disjoint QuAC transfer of a public SQuAD2 QA support gate (v62)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research.frc_rag.quac_anchor_safe_consensus_slot as v57
import research.frc_rag.squad2_extractive_support_gate as v59
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    _round_for_display,
)
from research.frc_rag.rgb_cost_aware_frc import read_jsonl


SCHEMA_VERSION = "frc-quac-roberta-qa-support-transfer-v62"
EXPERIMENT_ID = "FRC-QUAC-ROBERTA-QA-SUPPORT-TRANSFER-V62"
DATASET_ID = "quac_v0_2_disjoint_roberta_qa_support_transfer_v62"
CAPABILITY = "out_of_domain_extractive_qa_support_and_evidence_selection"
PROTOCOL_SHA256 = "130b703c7b69aa5556f7ed85d759d8865ed2fa40206ce91dabfbd6d12d80b007"
MODEL_REPOSITORY = "deepset/roberta-base-squad2"
MODEL_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"

STAGES = v57.STAGES
SOURCE_FILES = v57.SOURCE_FILES
TARGET_CASES = 600
TARGET_PER_GROUP = 300
MAX_CASES_PER_DIALOGUE = 2
MAX_CASES_PER_DOCUMENT_PER_STATE = 4
MINIMUM_DIALOGUES = 250
MINIMUM_DOCUMENTS = 200
MINIMUM_STRATUM_CASES = 50
BUDGETS = v57.BUDGETS
SAMPLE_SALTS = {
    "development": "FRC-QUAC-V62-DISJOINT-DEVELOPMENT|",
    "confirmation": "FRC-QUAC-V62-CONFIRMATION|",
}

GATED_NON_FRC_BY_BASE = {
    base: f"roberta_supported_{base}_v62"
    for base in v59.GATED_NON_FRC_BY_BASE
}
GATED_FRC_BY_BASE = {
    base: f"roberta_supported_{base}_v62" for base in v59.GATED_FRC_BY_BASE
}
GATED_NON_FRC_METHODS = tuple(GATED_NON_FRC_BY_BASE.values())
GATED_FRC_CONTROLS = tuple(GATED_FRC_BY_BASE.values())
CANDIDATE = "roberta_supported_adaptive_argmax_cardinality_frc_v43_v62"
EXACT_ANCHOR_GATED = GATED_NON_FRC_BY_BASE["cross_encoder_topk"]
EXACT_ANCHOR_UNGATED = v59.EXACT_ANCHOR_UNGATED
METHOD_RENAMES = {
    **{
        old: GATED_NON_FRC_BY_BASE[base]
        for base, old in v59.GATED_NON_FRC_BY_BASE.items()
    },
    **{
        old: GATED_FRC_BY_BASE[base]
        for base, old in v59.GATED_FRC_BY_BASE.items()
    },
    v59.CANDIDATE: CANDIDATE,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _opaque(prefix: str, *parts: str) -> str:
    return f"{prefix}{_hash(*parts)[:22]}"


def read_stage_source(source_root: Path, stage: str) -> dict[str, Any]:
    return v57.read_stage_source(source_root, stage)


def load_v57_excluded_commitments(path: Path) -> set[str]:
    rows = list(read_jsonl(path))
    commitments = {str(row["source_case_commitment"]) for row in rows}
    if len(rows) != TARGET_CASES or len(commitments) != TARGET_CASES:
        raise ValueError("QuAC v62 requires 600 unique v57 case commitments")
    return commitments


def _sample_key(row: dict[str, Any], stage: str) -> tuple[str, str, str]:
    return (
        _hash(
            SAMPLE_SALTS[stage],
            str(row["answer_state"]),
            str(row["dialogue_id"]),
            str(row["turn_id"]),
        ),
        str(row["dialogue_id"]),
        str(row["turn_id"]),
    )


def select_disjoint_balanced_sample(
    source: dict[str, Any],
    *,
    stage: str,
    excluded_commitments: set[str],
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_dialogue: int = MAX_CASES_PER_DIALOGUE,
    maximum_cases_per_document_per_state: int = MAX_CASES_PER_DOCUMENT_PER_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported QuAC v62 stage: {stage}")
    rows, schema = v57.extract_cases(source)
    available = [
        row
        for row in rows
        if _hash(str(row["dialogue_id"]), str(row["turn_id"]))
        not in excluded_commitments
    ]
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in available if row["answer_state"] == group],
            key=lambda row: _sample_key(row, stage),
        )
        for group in groups
    }
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("QuAC v62 source lacks the disjoint answer-state quota")
    selected: list[dict[str, Any]] = []
    selected_by_group: Counter[str] = Counter()
    dialogue_counts: Counter[str] = Counter()
    document_state_counts: Counter[tuple[str, str]] = Counter()
    cursors = {group: 0 for group in groups}
    while any(selected_by_group[group] < target_per_group for group in groups):
        progressed = False
        for group in groups:
            if selected_by_group[group] >= target_per_group:
                continue
            values = by_group[group]
            while cursors[group] < len(values):
                row = values[cursors[group]]
                cursors[group] += 1
                dialogue_id = str(row["dialogue_id"])
                document_state = (str(row["document_id"]), group)
                if dialogue_counts[dialogue_id] >= maximum_cases_per_dialogue:
                    continue
                if (
                    document_state_counts[document_state]
                    >= maximum_cases_per_document_per_state
                ):
                    continue
                selected.append(row)
                selected_by_group[group] += 1
                dialogue_counts[dialogue_id] += 1
                document_state_counts[document_state] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError("QuAC v62 caps prevent the registered balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    overlap = sum(
        _hash(str(row["dialogue_id"]), str(row["turn_id"]))
        in excluded_commitments
        for row in selected
    )
    return selected, {
        "stage": stage,
        "target_cases": target_per_group * 2,
        "selected_cases": len(selected),
        "eligible_answer_state_counts_after_exclusion": dict(
            Counter(row["answer_state"] for row in available)
        ),
        "selected_answer_state_counts": dict(selected_by_group),
        "selected_dialogues": len(dialogue_counts),
        "selected_documents": len({str(row["document_id"]) for row in selected}),
        "maximum_cases_per_dialogue": max(dialogue_counts.values(), default=0),
        "maximum_cases_per_document_per_answer_state": max(
            document_state_counts.values(), default=0
        ),
        "v57_excluded_commitment_count": len(excluded_commitments),
        "selected_v57_commitment_overlap": overlap,
        **schema,
    }


def _current_question(query: str) -> str:
    first = query.splitlines()[0].strip()
    prefix = "Current question:"
    if not first.startswith(prefix) or not first[len(prefix) :].strip():
        raise ValueError("QuAC v62 cannot recover the current question")
    return first[len(prefix) :].strip()


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]],
    source: dict[str, Any],
    tokenizer: Any,
    *,
    stage: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    prepared, maps, documents, structural = v57.prepare_blind_cases(
        selected, source, tokenizer, stage=stage
    )
    selected_by_commitment = {
        _hash(str(row["dialogue_id"]), str(row["turn_id"])): row
        for row in selected
    }
    qa_inputs: list[dict[str, Any]] = []
    for blind, mapping in zip(prepared, maps, strict=True):
        commitment = str(mapping["source_case_commitment"])
        source_row = selected_by_commitment[commitment]
        case_id = _opaque("q62c", stage, commitment)
        blind["schema_version"] = SCHEMA_VERSION
        blind["dataset_id"] = DATASET_ID
        blind["capability"] = CAPABILITY
        blind["id"] = case_id
        mapping["schema_version"] = "frc-quac-v62-candidate-map-v1"
        mapping["id"] = case_id
        qa_inputs.append(
            {
                "schema_version": "frc-quac-v62-qa-input-v1",
                "id": case_id,
                "question": _current_question(str(source_row["query"])),
                "context": str(source_row["context"]),
                "gold_fields_visible_to_verifier": False,
            }
        )
    for document in documents:
        document["schema_version"] = "frc-quac-v62-blind-document-v1"
    structural = {
        **structural,
        "schema_version": "frc-quac-v62-structural-census-v1",
        "qa_input_rows": len(qa_inputs),
        "qa_gold_fields_exported": False,
    }
    return prepared, maps, documents, qa_inputs, structural


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = v57.build_deterministic_queries(prepared_rows)
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
    return rows


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("QuAC v62 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenQuacV62Scorer(v57.FrozenQuacScorer):
    """Frozen BGE stack with v62 schema labels."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def best_support_decision(
    features: Sequence[dict[str, Any]],
    context: str,
    *,
    maximum_answer_tokens: int = 30,
    top_k: int = 20,
) -> dict[str, Any]:
    best_span_score = -math.inf
    best_interval: tuple[int, int] | None = None
    null_score = math.inf
    invalid = False
    for feature in features:
        starts = np.asarray(feature["start_logits"], dtype=float)
        ends = np.asarray(feature["end_logits"], dtype=float)
        offsets = list(feature["offsets"])
        sequence_ids = list(feature["sequence_ids"])
        cls_index = int(feature["cls_index"])
        if (
            starts.shape != ends.shape
            or starts.ndim != 1
            or len(offsets) != len(starts)
            or len(sequence_ids) != len(starts)
            or cls_index < 0
            or cls_index >= len(starts)
            or not np.all(np.isfinite(starts))
            or not np.all(np.isfinite(ends))
        ):
            invalid = True
            continue
        null_score = min(null_score, float(starts[cls_index] + ends[cls_index]))
        start_indices = np.argsort(-starts)[:top_k]
        end_indices = np.argsort(-ends)[:top_k]
        for start in start_indices:
            for end in end_indices:
                start_i = int(start)
                end_i = int(end)
                if (
                    sequence_ids[start_i] != 1
                    or sequence_ids[end_i] != 1
                    or end_i < start_i
                    or end_i - start_i + 1 > maximum_answer_tokens
                ):
                    continue
                start_offset = offsets[start_i]
                end_offset = offsets[end_i]
                if start_offset is None or end_offset is None:
                    continue
                char_start = int(start_offset[0])
                char_end = int(end_offset[1])
                if char_start < 0 or char_end <= char_start or char_end > len(context):
                    continue
                score = float(starts[start_i] + ends[end_i])
                if score > best_span_score:
                    best_span_score = score
                    best_interval = (char_start, char_end)
    if not math.isfinite(null_score) or not math.isfinite(best_span_score):
        invalid = True
    support_passed = bool(
        not invalid and best_interval is not None and best_span_score > null_score
    )
    span = context[slice(*best_interval)] if best_interval is not None else ""
    return {
        "decision": "SUPPORTED" if support_passed else "UNSUPPORTED",
        "support_passed": support_passed,
        "best_span_score": best_span_score if math.isfinite(best_span_score) else None,
        "null_score": null_score if math.isfinite(null_score) else None,
        "score_margin": (
            best_span_score - null_score
            if math.isfinite(best_span_score) and math.isfinite(null_score)
            else None
        ),
        "span_start": best_interval[0] if best_interval is not None else None,
        "span_end": best_interval[1] if best_interval is not None else None,
        "span_sha256": hashlib.sha256(span.encode("utf-8")).hexdigest(),
        "invalid_fail_closed_used": invalid,
    }


class LocalRobertaQASupportVerifier:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 384,
        doc_stride: int = 128,
        maximum_answer_tokens: int = 30,
        fp16: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.maximum_answer_tokens = maximum_answer_tokens
        self.fp16 = fp16 and self.device.type == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        if not self.tokenizer.is_fast:
            raise ValueError("QuAC v62 requires a fast tokenizer for offset mapping")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def verify(self, rows: Sequence[dict[str, Any]], *, cache_key: str) -> list[dict[str, Any]]:
        questions = [str(row["question"]) for row in rows]
        contexts = [str(row["context"]) for row in rows]
        encoded = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        sample_map = encoded["overflow_to_sample_mapping"].tolist()
        offsets = encoded["offset_mapping"].tolist()
        sequence_ids = [encoded.sequence_ids(index) for index in range(len(sample_map))]
        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key not in {"overflow_to_sample_mapping", "offset_mapping"}
        }
        grouped: list[list[dict[str, Any]]] = [[] for _ in rows]
        for start in range(0, len(sample_map), self.batch_size):
            stop = min(start + self.batch_size, len(sample_map))
            batch = {
                key: value[start:stop].to(self.device)
                for key, value in model_inputs.items()
            }
            with self.torch.inference_mode(), self.torch.autocast(
                device_type=self.device.type,
                dtype=self.torch.float16,
                enabled=self.fp16,
            ):
                output = self.model(**batch)
            start_logits = output.start_logits.detach().float().cpu().numpy()
            end_logits = output.end_logits.detach().float().cpu().numpy()
            input_ids = batch["input_ids"].detach().cpu().numpy()
            for relative, feature_index in enumerate(range(start, stop)):
                cls_positions = np.flatnonzero(
                    input_ids[relative] == self.tokenizer.cls_token_id
                )
                cls_index = int(cls_positions[0]) if len(cls_positions) else 0
                grouped[sample_map[feature_index]].append(
                    {
                        "start_logits": start_logits[relative],
                        "end_logits": end_logits[relative],
                        "offsets": offsets[feature_index],
                        "sequence_ids": sequence_ids[feature_index],
                        "cls_index": cls_index,
                    }
                )
        predictions: list[dict[str, Any]] = []
        for row, context, features in zip(rows, contexts, grouped, strict=True):
            decision = best_support_decision(
                features,
                context,
                maximum_answer_tokens=self.maximum_answer_tokens,
            )
            predictions.append(
                {
                    "schema_version": "frc-quac-v62-qa-decision-v1",
                    "id": str(row["id"]),
                    "cache_key": cache_key,
                    **decision,
                    "feature_count": len(features),
                    "model_repository": MODEL_REPOSITORY,
                    "model_revision": MODEL_REVISION,
                    "gold_fields_visible_to_verifier": False,
                }
            )
        return predictions


def validate_support_cache(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> dict[str, Any]:
    if [str(row["id"]) for row in qa_inputs] != [str(row["id"]) for row in decisions]:
        raise ValueError("QuAC v62 QA cache is incomplete or out of order")
    if any(row.get("cache_key") != cache_key for row in decisions):
        raise ValueError("QuAC v62 QA cache key changed")
    invalid = sum(bool(row.get("invalid_fail_closed_used")) for row in decisions)
    return {
        "rows": len(decisions),
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / len(decisions) if decisions else 1.0,
        "gold_fields_visible_to_verifier": False,
    }


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    document_length_quartile_boundaries: Sequence[float],
    turn_position_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    return v57.build_gold_rows(
        source,
        candidate_maps,
        scored_rows,
        document_length_quartile_boundaries=document_length_quartile_boundaries,
        turn_position_quartile_boundaries=turn_position_quartile_boundaries,
    )


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    value = v57.build_candidate_coverage(gold_rows, sampling)
    value["schema_version"] = "frc-quac-v62-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    value["selected_v57_commitment_overlap"] = sampling[
        "selected_v57_commitment_overlap"
    ]
    return value


def _rename_methods(report: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
    aggregates = report["analysis"]["aggregates"]
    report["analysis"]["aggregates"] = {
        METHOD_RENAMES.get(name, name): value for name, value in aggregates.items()
    }
    for key in ("strongest_shared_gate_non_frc", "strongest_same_gate_frc"):
        value = report["analysis"][key]
        report["analysis"][key] = METHOD_RENAMES.get(value, value)
    for row in evidence:
        for configuration in row["configurations"].values():
            methods = configuration["methods"]
            configuration["methods"] = {
                METHOD_RENAMES.get(name, name): value for name, value in methods.items()
            }


def evaluate_stage(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    verifier_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapted_gold = [
        {
            **row,
            "article_cluster": row["document_cluster"],
            "paragraph_cluster": row["dialogue_cluster"],
            "paragraph_length_quartile": row["document_length_quartile"],
            "article_case_count_quartile": row[
                "dialogue_turn_position_quartile"
            ],
        }
        for row in gold_rows
    ]
    adapted_support = [
        {
            **row,
            "raw_prediction_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "decision": row["decision"],
                        "score_margin": row["score_margin"],
                        "span_sha256": row["span_sha256"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "raw_prediction_codepoints": 0,
        }
        for row in support_rows
    ]
    sampling = source_artifacts["sampling"]
    compatible_source = {
        **source_artifacts,
        "sampling": {
            **sampling,
            "selected_v58_commitment_overlap": sampling[
                "selected_v57_commitment_overlap"
            ],
            "maximum_cases_per_paragraph": sampling["maximum_cases_per_dialogue"],
            "maximum_cases_per_article_per_answer_state": sampling[
                "maximum_cases_per_document_per_answer_state"
            ],
        },
    }
    report, evidence = v59.evaluate_stage(
        adapted_gold,
        scored_rows,
        adapted_support,
        query_summary,
        verifier_summary,
        compatible_source,
        stage=stage,
    )
    _rename_methods(report, evidence)
    analysis = report["analysis"]
    metadata = report["metadata"]
    metadata.update(
        {
            "dataset_id": DATASET_ID,
            "split": SOURCE_FILES[stage],
            "documents": metadata.pop("articles"),
            "dialogues": metadata.pop("paragraphs"),
            "official_quac_answer_result": False,
            "official_squad2_answer_string_result": False,
            "v57_case_level_artifact_reused": False,
        }
    )
    verifier = analysis["support_verifier"]
    verifier["answer_bearing_support_pass_rate"] = verifier.pop(
        "answer_bearing_span_pass_rate"
    )
    candidate = analysis["aggregates"][CANDIDATE]
    comparisons = analysis["family_comparison"]
    minimum_delta = analysis["minimum_budget_or_supported_stratum_delta"]
    state_counts = Counter(row["answer_state"] for row in evidence)
    checks = {
        "exact_cases_equals_600": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": state_counts
        == {"answer_bearing": TARGET_PER_GROUP, "no_answer": TARGET_PER_GROUP},
        "selected_v57_commitment_overlap_equals_0": sampling[
            "selected_v57_commitment_overlap"
        ]
        == 0,
        "minimum_dialogues_at_least_250": metadata["dialogues"]
        >= MINIMUM_DIALOGUES,
        "minimum_documents_at_least_200": metadata["documents"]
        >= MINIMUM_DOCUMENTS,
        "dialogue_cap_at_most_2": sampling["maximum_cases_per_dialogue"]
        <= MAX_CASES_PER_DIALOGUE,
        "document_state_cap_at_most_4": sampling[
            "maximum_cases_per_document_per_answer_state"
        ]
        <= MAX_CASES_PER_DOCUMENT_PER_STATE,
        "schema_exclusion_rate_at_most_0_01": sampling["schema_exclusion_rate"]
        <= 0.01,
        "candidate_ceiling_complete_rate_at_least_0_99": analysis[
            "candidate_ceiling_complete_rate"
        ]
        >= 0.99,
        "qa_invalid_rate_equals_0": verifier["invalid_output_rate"] == 0.0,
        "support_verifier_balanced_accuracy_at_least_0_75": verifier[
            "balanced_accuracy"
        ]
        >= 0.75,
        "answer_bearing_support_pass_rate_at_least_0_75": verifier[
            "answer_bearing_support_pass_rate"
        ]
        >= 0.75,
        "no_answer_rejection_rate_at_least_0_65": verifier[
            "no_answer_rejection_rate"
        ]
        >= 0.65,
        "candidate_answer_recall_at_least_0_70": candidate["answer_macro_recall"]
        >= 0.70,
        "candidate_no_answer_abstention_accuracy_at_least_0_65": candidate[
            "no_answer_abstention_accuracy"
        ]
        >= 0.65,
        "candidate_abstention_rate_at_least_0_15": candidate["abstention_rate"]
        >= 0.15,
        "candidate_abstention_rate_at_most_0_70": candidate["abstention_rate"]
        <= 0.70,
        "candidate_minus_gated_exact_anchor_point_at_least_0_005": comparisons[
            "candidate_minus_gated_exact_anchor"
        ]["point"]
        >= 0.005,
        "candidate_minus_gated_exact_anchor_ci_low_above_0": comparisons[
            "candidate_minus_gated_exact_anchor"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_005": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        >= 0.005,
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_strongest_same_gate_frc_point_at_least_minus_0_005": comparisons[
            "candidate_minus_strongest_same_gate_frc"
        ]["point"]
        >= -0.005,
        "gated_exact_anchor_minus_ungated_exact_anchor_point_at_least_0_1": comparisons[
            "gated_exact_anchor_minus_ungated_exact_anchor"
        ]["point"]
        >= 0.1,
        "gated_exact_anchor_minus_ungated_exact_anchor_ci_low_above_0": comparisons[
            "gated_exact_anchor_minus_ungated_exact_anchor"
        ]["ci_low"]
        > 0.0,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_02": minimum_delta
        >= -0.02,
        "deterministic_query_fallback_rate_equals_0": query_summary[
            "fallback_rate"
        ]
        == 0.0,
        "score_fallback_rate_equals_0": source_artifacts["score_fallback_rate"]
        == 0.0,
    }
    supported = all(checks.values())
    status = (
        "QUAC_V62_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_VALIDATION"
        if stage == "development" and supported
        else (
            "QUAC_V62_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION"
            if stage == "development"
            else (
                "QUAC_V62_TRANSFER_SUPPORT_ESTABLISHED"
                if supported
                else "QUAC_V62_TRANSFER_SUPPORT_NOT_ESTABLISHED"
            )
        )
    )
    report["schema_version"] = "frc-quac-roberta-qa-support-report-v62"
    report["experiment_id"] = EXPERIMENT_ID
    analysis["mechanism"] = (
        "parameter-free model-native null-versus-span RoBERTa QA decision, "
        "shared across frozen selectors on disjoint QuAC cases"
    )
    analysis["support_checks"] = checks
    analysis["outcome"].update(
        {
            "status": status,
            "support_established": supported,
            "confirmation_open_authorized": stage == "development" and supported,
            "selector_adoption_authorized": False,
            "canary_or_default_authorized": False,
            "reuse_stage_for_tuning_or_selection": False,
            "gate_2": "NO-GO/SHADOW",
        }
    )
    for row in evidence:
        row["document_cluster"] = row.pop("article_cluster")
        row["dialogue_cluster"] = row.pop("paragraph_cluster")
        row["document_length_quartile"] = row.pop("paragraph_length_quartile")
        row["dialogue_turn_position_quartile"] = row.pop(
            "article_case_count_quartile"
        )
    analysis["supported_stratum_deltas"] = {
        key.replace("paragraph_length_quartile", "document_length_quartile").replace(
            "article_case_count_quartile", "dialogue_turn_position_quartile"
        ): value
        for key, value in analysis["supported_stratum_deltas"].items()
    }
    return report, evidence


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
    candidate = analysis["aggregates"][CANDIDATE]
    lines = [
        f"# QuAC RoBERTa QA support transfer ({rounded['metadata']['stage']}, v62)",
        "",
        f"- Status: `{analysis['outcome']['status']}`",
        f"- Cases/documents/dialogues: {rounded['metadata']['cases']}/{rounded['metadata']['documents']}/{rounded['metadata']['dialogues']}",
        f"- Verifier balanced accuracy: {verifier['balanced_accuracy']:.6f}",
        f"- Answer-bearing support pass rate: {verifier['answer_bearing_support_pass_rate']:.6f}",
        f"- No-answer rejection rate: {verifier['no_answer_rejection_rate']:.6f}",
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        f"- Candidate answer recall: {candidate['answer_macro_recall']:.6f}",
        "- Selector adoption: `false`",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Family comparisons",
        "",
    ]
    for name, value in analysis["family_comparison"].items():
        lines.append(
            f"- `{name}`: {value['point']:+.6f} "
            f"(95% CI [{value['ci_low']:+.6f}, {value['ci_high']:+.6f}])"
        )
    lines.extend(["", "## Support checks", ""])
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is an out-of-domain QuAC component-transfer and evidence-selection mechanism experiment. The QA model was trained on SQuAD2. This is not official QuAC answer extraction, official SQuAD2 evaluation, SetR reproduction, open-domain retrieval, selector adoption, or flood-domain validation.",
            "",
        ]
    )
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


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("QuAC v62 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value["prior_boundary"]["v57_quac_validation_content_opened_or_parsed"] is not False:
        raise ValueError("QuAC v62 validation boundary changed")
    if value["public_support_model"]["revision"] != MODEL_REVISION:
        raise ValueError("QuAC v62 model revision changed")
    if value["scope"]["development_cases"] != TARGET_CASES:
        raise ValueError("QuAC v62 development size changed")
    return value


def validate_source_contract(
    source_registration_path: Path,
    *,
    protocol_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    if _sha256(source_registration_path) != protocol["source_and_split"][
        "source_registration_sha256"
    ]:
        raise ValueError("QuAC v62 source registration changed")
    return v57.validate_source_registration(
        source_registration_path,
        protocol_path=(
            protocol_path.resolve().parents[2]
            / "docs/progressive_upgrade/quac_anchor_safe_consensus_slot_protocol_v57.json"
        ),
        source_root=source_root,
    )


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    source_registration_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": _sha256(protocol_path),
        "source_registration": _sha256(source_registration_path),
        "module": _sha256(module_path),
        "runner": _sha256(runner_path),
        "tests": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("QuAC v62 implementation registration changed")
    if value.get("model_files_downloaded_before_registration") is not False:
        raise ValueError("QuAC v62 model was downloaded before implementation freeze")
    if value.get("quac_validation_content_opened_or_parsed") is not False:
        raise ValueError("QuAC v62 validation was opened before implementation freeze")
    return value


def validate_model_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    model_root: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("QuAC v62 model registration protocol hash changed")
    if value["repository"] != MODEL_REPOSITORY or value["revision"] != MODEL_REVISION:
        raise ValueError("QuAC v62 registered model identity changed")
    expected = {
        path.name: _sha256(path)
        for path in sorted(model_root.iterdir())
        if path.is_file()
    }
    if value["files"] != expected:
        raise ValueError("QuAC v62 local model files changed")
    return value


__all__ = [
    "BUDGETS",
    "CANDIDATE",
    "EXPERIMENT_ID",
    "FrozenQuacV62Scorer",
    "LocalRobertaQASupportVerifier",
    "MODEL_REVISION",
    "SOURCE_FILES",
    "TARGET_CASES",
    "best_support_decision",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "evaluate_stage",
    "load_v57_excluded_commitments",
    "prepare_blind_cases",
    "read_stage_source",
    "select_disjoint_balanced_sample",
    "validate_implementation_registration",
    "validate_model_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_contract",
    "validate_support_cache",
    "write_report",
]
