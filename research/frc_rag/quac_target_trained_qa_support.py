"""Target-trained public QA support gate on sealed QuAC validation (v63)."""

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
import research.frc_rag.quac_roberta_qa_support_transfer as v62


SCHEMA_VERSION = "frc-quac-target-trained-qa-support-v63"
EXPERIMENT_ID = "FRC-QUAC-TARGET-TRAINED-QA-SUPPORT-V63"
DATASET_ID = "quac_v0_2_target_trained_qa_support_validation_v63"
PROTOCOL_SHA256 = "40549911361d112aeac4bd6d1d432fb4068e96106afe53c7727bbb41481f3294"
MODEL_REPOSITORY = "ixa-ehu/SciBERT-SQuAD-QuAC"
MODEL_REVISION = "8d44c186b6f1662c65d48603ce8e3dedb0093953"
SOURCE_FILE = "val_v0.2.json"
SOURCE_SHA256 = "09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378"
SAMPLE_SALT = "FRC-QUAC-V63-TARGET-TRAINED-VALIDATION|"
TARGET_CASES = 600
TARGET_PER_GROUP = 300
MAX_CASES_PER_DIALOGUE = 2
MAX_CASES_PER_DOCUMENT_PER_STATE = 2
MINIMUM_DIALOGUES = 250
MINIMUM_DOCUMENTS = 200
BUDGETS = v62.BUDGETS
CANDIDATE = "scibert_quac_supported_adaptive_argmax_cardinality_frc_v43_v63"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _opaque(prefix: str, *parts: str) -> str:
    return prefix + _hash(*parts)[:24]


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("QuAC v63 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("QuAC v63 experiment identity changed")
    if value["prior_boundary"]["v62_validation_content_opened_or_parsed"] is not False:
        raise ValueError("QuAC v63 was not registered before validation opened")
    if value["public_support_model"]["revision"] != MODEL_REVISION:
        raise ValueError("QuAC v63 model revision changed")
    if value["scope"]["cases"] != TARGET_CASES:
        raise ValueError("QuAC v63 sample size changed")
    if value["stopping_and_claim_boundary"]["strict_independent_confirmation_claimed"]:
        raise ValueError("QuAC v63 cannot claim strict independent confirmation")
    return value


def validate_source_contract(
    source_registration_path: Path,
    *,
    protocol_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    if (
        _sha256(source_registration_path)
        != protocol["source"]["source_registration_sha256"]
    ):
        raise ValueError("QuAC v63 source registration changed")
    value = v57.validate_source_registration(
        source_registration_path,
        protocol_path=(
            protocol_path.resolve().parents[2]
            / "docs/progressive_upgrade/quac_anchor_safe_consensus_slot_protocol_v57.json"
        ),
        source_root=source_root,
    )
    if _sha256(source_root / SOURCE_FILE) != SOURCE_SHA256:
        raise ValueError("QuAC v63 validation source hash changed")
    return value


def read_validation_source(source_root: Path) -> dict[str, Any]:
    path = source_root / SOURCE_FILE
    if _sha256(path) != SOURCE_SHA256:
        raise ValueError("QuAC v63 validation source hash changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _hash(
            SAMPLE_SALT,
            str(row["answer_state"]),
            str(row["dialogue_id"]),
            str(row["turn_id"]),
        ),
        str(row["dialogue_id"]),
        str(row["turn_id"]),
    )


def select_balanced_validation_sample(
    source: dict[str, Any],
    *,
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_dialogue: int = MAX_CASES_PER_DIALOGUE,
    maximum_cases_per_document_per_state: int = MAX_CASES_PER_DOCUMENT_PER_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, schema = v57.extract_cases(source)
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in rows if row["answer_state"] == group],
            key=_sample_key,
        )
        for group in groups
    }
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("QuAC v63 source lacks the balanced answer-state quota")
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
            raise ValueError("QuAC v63 caps prevent the registered balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row)))
    return selected, {
        "stage": "validation",
        "target_cases": target_per_group * 2,
        "selected_cases": len(selected),
        "eligible_answer_state_counts": dict(
            Counter(row["answer_state"] for row in rows)
        ),
        "selected_answer_state_counts": dict(selected_by_group),
        "selected_dialogues": len(dialogue_counts),
        "selected_documents": len({str(row["document_id"]) for row in selected}),
        "maximum_cases_per_dialogue": max(dialogue_counts.values(), default=0),
        "maximum_cases_per_document_per_answer_state": max(
            document_state_counts.values(), default=0
        ),
        "v57_excluded_commitment_count": 0,
        "selected_v57_commitment_overlap": 0,
        **schema,
    }


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]],
    source: dict[str, Any],
    tokenizer: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    prepared, maps, documents, qa_inputs, structural = v62.prepare_blind_cases(
        selected,
        source,
        tokenizer,
        stage="confirmation",
    )
    id_map: dict[str, str] = {}
    for mapping in maps:
        old_id = str(mapping["id"])
        new_id = _opaque("q63c", str(mapping["source_case_commitment"]))
        id_map[old_id] = new_id
        mapping["id"] = new_id
        mapping["schema_version"] = "frc-quac-v63-candidate-map-v1"
    for row in prepared:
        row["id"] = id_map[str(row["id"])]
        row["schema_version"] = "frc-quac-v63-blind-case-v1"
    for row in qa_inputs:
        row["id"] = id_map[str(row["id"])]
        row["schema_version"] = "frc-quac-v63-qa-input-v1"
    for row in documents:
        row["schema_version"] = "frc-quac-v63-blind-document-v1"
    structural = {
        **structural,
        "schema_version": "frc-quac-v63-structural-census-v1",
        "qa_input_rows": len(qa_inputs),
        "qa_gold_fields_exported": False,
    }
    return prepared, maps, documents, qa_inputs, structural


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = v62.build_deterministic_queries(prepared_rows)
    for row in rows:
        row["schema_version"] = "frc-quac-v63-query-v1"
    return rows


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]],
    query_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("QuAC v63 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
    }


class FrozenQuacV63Scorer(v62.FrozenQuacV62Scorer):
    """Frozen BGE stack with v63 schema labels."""

    def score_batch(
        self, cases: Sequence[dict[str, Any]], start_index: int
    ) -> list[dict[str, Any]]:
        rows = super().score_batch(cases, start_index)
        for row in rows:
            row["schema_version"] = "frc-quac-v63-scored-case-v1"
        return rows


class LocalSciBertQuacSupportVerifier:
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
            raise ValueError("QuAC v63 requires a fast tokenizer for offset mapping")
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def verify(
        self, rows: Sequence[dict[str, Any]], *, cache_key: str
    ) -> list[dict[str, Any]]:
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
            with (
                self.torch.inference_mode(),
                self.torch.autocast(
                    device_type=self.device.type,
                    dtype=self.torch.float16,
                    enabled=self.fp16,
                ),
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
            decision = v62.best_support_decision(
                features,
                context,
                maximum_answer_tokens=self.maximum_answer_tokens,
            )
            predictions.append(
                {
                    "schema_version": "frc-quac-v63-qa-decision-v1",
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
        raise ValueError("QuAC v63 QA cache is incomplete or out of order")
    if any(row.get("cache_key") != cache_key for row in decisions):
        raise ValueError("QuAC v63 QA cache key changed")
    invalid = sum(bool(row.get("invalid_fail_closed_used")) for row in decisions)
    return {
        "rows": len(decisions),
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / len(decisions) if decisions else 1.0,
        "gold_fields_visible_to_verifier": False,
    }


def build_support_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_rows, _ = v57.extract_cases(source)
    by_commitment = {
        v62._hash(str(row["dialogue_id"]), str(row["turn_id"])): row
        for row in source_rows
    }
    support_by_id = {str(row["id"]): row for row in support_rows}
    result: list[dict[str, Any]] = []
    for mapping in candidate_maps:
        case_id = str(mapping["id"])
        source_row = by_commitment[str(mapping["source_case_commitment"])]
        support = support_by_id[case_id]
        result.append(
            {
                "case_id": case_id,
                "answer_state": source_row["answer_state"],
                "support_passed": bool(support["support_passed"]),
                "score_margin": support["score_margin"],
                "invalid_fail_closed_used": bool(support["invalid_fail_closed_used"]),
                "dialogue_cluster": mapping["dialogue_commitment"],
                "document_cluster": mapping["document_commitment"],
            }
        )
    return result


def evaluate_support_open_gate(
    support_gold_rows: Sequence[dict[str, Any]],
    verifier_summary: dict[str, Any],
    sampling: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter(row["answer_state"] for row in support_gold_rows)
    answer_rows = [
        row for row in support_gold_rows if row["answer_state"] == "answer_bearing"
    ]
    no_answer_rows = [
        row for row in support_gold_rows if row["answer_state"] == "no_answer"
    ]
    answer_pass = sum(row["support_passed"] for row in answer_rows) / max(
        len(answer_rows), 1
    )
    no_answer_reject = sum(not row["support_passed"] for row in no_answer_rows) / max(
        len(no_answer_rows), 1
    )
    balanced = (answer_pass + no_answer_reject) / 2
    checks = {
        "exact_cases_equals_600": len(support_gold_rows) == TARGET_CASES,
        "exact_answer_state_balance": counts
        == {"answer_bearing": TARGET_PER_GROUP, "no_answer": TARGET_PER_GROUP},
        "schema_exclusion_rate_at_most_0_01": sampling["schema_exclusion_rate"] <= 0.01,
        "qa_invalid_rate_equals_0": verifier_summary["invalid_output_rate"] == 0.0,
        "support_verifier_balanced_accuracy_at_least_0_75": balanced >= 0.75,
        "answer_bearing_support_pass_rate_at_least_0_75": answer_pass >= 0.75,
        "no_answer_rejection_rate_at_least_0_65": no_answer_reject >= 0.65,
        "minimum_dialogues_at_least_250": sampling["selected_dialogues"]
        >= MINIMUM_DIALOGUES,
        "minimum_documents_at_least_200": sampling["selected_documents"]
        >= MINIMUM_DOCUMENTS,
    }
    passed = all(checks.values())
    verifier = {
        **verifier_summary,
        "answer_bearing_support_pass_rate": answer_pass,
        "no_answer_rejection_rate": no_answer_reject,
        "balanced_accuracy": balanced,
    }
    status = (
        "QUAC_V63_SUPPORT_GATE_PASSED_OPEN_RETRIEVAL_SCORING"
        if passed
        else "QUAC_V63_VALIDATION_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_RETRIEVAL_SCORING"
    )
    report = {
        "schema_version": "frc-quac-v63-support-open-gate-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "stage": "validation_support_gate",
            "dataset_id": DATASET_ID,
            "cases": len(support_gold_rows),
            "dialogues": sampling["selected_dialogues"],
            "documents": sampling["selected_documents"],
            "answer_state_counts": dict(counts),
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "support_verifier": verifier,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_gate_passed": passed,
                "retrieval_scoring_open_authorized": passed,
                "validation_reuse_for_tuning_or_selection": False,
                "strict_independent_confirmation_claimed": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    evidence = [dict(row) for row in support_gold_rows]
    return _round_floats(report), _round_floats(evidence)


def _v63_method_name(name: str) -> str:
    if name.startswith("roberta_supported_") and name.endswith("_v62"):
        return "scibert_quac_supported_" + name[len("roberta_supported_") : -4] + "_v63"
    return name


def _rename_v63_methods(report: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
    analysis = report["analysis"]
    analysis["aggregates"] = {
        _v63_method_name(name): value for name, value in analysis["aggregates"].items()
    }
    for key in ("strongest_shared_gate_non_frc", "strongest_same_gate_frc"):
        analysis[key] = _v63_method_name(analysis[key])
    for row in evidence:
        for configuration in row["configurations"].values():
            configuration["methods"] = {
                _v63_method_name(name): value
                for name, value in configuration["methods"].items()
            }


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    document_length_quartile_boundaries: Sequence[float],
    turn_position_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    return v62.build_gold_rows(
        source,
        candidate_maps,
        scored_rows,
        document_length_quartile_boundaries=document_length_quartile_boundaries,
        turn_position_quartile_boundaries=turn_position_quartile_boundaries,
    )


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    value = v62.build_candidate_coverage(gold_rows, sampling)
    value["schema_version"] = "frc-quac-v63-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    return value


def evaluate_final_method(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    verifier_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report, evidence = v62.evaluate_stage(
        gold_rows,
        scored_rows,
        support_rows,
        query_summary,
        verifier_summary,
        source_artifacts,
        stage="confirmation",
    )
    _rename_v63_methods(report, evidence)
    analysis = report["analysis"]
    metadata = report["metadata"]
    checks = analysis["support_checks"]
    supported = all(checks.values())
    status = (
        "QUAC_V63_TARGET_TRAINED_QA_METHOD_FEASIBILITY_SIGNAL_PROVENANCE_LIMITED"
        if supported
        else "QUAC_V63_VALIDATION_SELECTOR_SUPPORT_NOT_ESTABLISHED"
    )
    report["schema_version"] = "frc-quac-v63-final-method-result-v1"
    report["experiment_id"] = EXPERIMENT_ID
    metadata.update(
        {
            "dataset_id": DATASET_ID,
            "stage": "validation_final_method",
            "public_support_model_repository": MODEL_REPOSITORY,
            "public_support_model_revision": MODEL_REVISION,
            "support_model_exact_training_split_verified": False,
            "strict_independent_confirmation": False,
        }
    )
    analysis["mechanism"] = (
        "fixed public SciBERT QA model disclosed as SQuAD2.0+QuAC fine-tuned; "
        "model-native null-versus-span gate shared across frozen selectors"
    )
    analysis["outcome"].update(
        {
            "status": status,
            "support_established": supported,
            "training_provenance_limited": True,
            "strict_independent_confirmation_claimed": False,
            "validation_reuse_for_tuning_or_selection": False,
            "selector_adoption_authorized": False,
            "canary_or_default_authorized": False,
            "gate_2": "NO-GO/SHADOW",
        }
    )
    return _round_floats(report), _round_floats(evidence)


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
    *,
    title: str,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
    outcome = report["analysis"]["outcome"]
    verifier = report["analysis"]["support_verifier"]
    lines = [
        f"# {title}",
        "",
        f"- status: `{outcome['status']}`",
        f"- cases: {report['metadata']['cases']}",
        f"- balanced accuracy: {verifier['balanced_accuracy']:.6f}",
        (
            "- answer-bearing pass / no-answer rejection: "
            f"{verifier['answer_bearing_support_pass_rate']:.6f} / "
            f"{verifier['no_answer_rejection_rate']:.6f}"
        ),
        f"- selector adoption authorized: `{outcome['selector_adoption_authorized']}`",
        f"- Gate 2: `{outcome['gate_2']}`",
        "",
        "> The public model is disclosed as QuAC-trained, but its exact training split and checkpoint-selection provenance are not fully documented. This result is not a strict independent confirmation or an official QuAC answer score.",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


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
        raise ValueError("QuAC v63 implementation registration changed")
    if value.get("model_files_downloaded_before_registration") is not False:
        raise ValueError("QuAC v63 model was downloaded before implementation freeze")
    if value.get("quac_validation_content_opened_or_parsed") is not False:
        raise ValueError("QuAC v63 validation was opened before implementation freeze")
    return value


def validate_model_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    model_root: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("QuAC v63 model registration protocol hash changed")
    if value["repository"] != MODEL_REPOSITORY or value["revision"] != MODEL_REVISION:
        raise ValueError("QuAC v63 registered model identity changed")
    expected = {
        path.name: _sha256(path)
        for path in sorted(model_root.iterdir())
        if path.is_file()
    }
    if value["files"] != expected:
        raise ValueError("QuAC v63 local model files changed")
    return value


__all__ = [
    "BUDGETS",
    "CANDIDATE",
    "EXPERIMENT_ID",
    "FrozenQuacV63Scorer",
    "LocalSciBertQuacSupportVerifier",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "SOURCE_FILE",
    "TARGET_CASES",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "build_support_gold_rows",
    "evaluate_final_method",
    "evaluate_support_open_gate",
    "prepare_blind_cases",
    "read_validation_source",
    "select_balanced_validation_sample",
    "validate_implementation_registration",
    "validate_model_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_contract",
    "validate_support_cache",
    "write_report",
]
