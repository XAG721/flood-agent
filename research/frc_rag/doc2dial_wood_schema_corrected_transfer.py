"""Outcome-prospective Doc2Dial wOOD v56 schema-corrected transfer."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research.frc_rag.contractnli_native_zero_consensus_abstention as v50
import research.frc_rag.doc2dial_document_contrastive_role_closure as v54
import research.frc_rag.doc2dial_wood_document_contrastive_transfer as v55
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    _round_for_display,
)


SCHEMA_VERSION = "frc-doc2dial-wood-schema-corrected-transfer-v56"
EXPERIMENT_ID = "FRC-DOC2DIAL-WOOD-SCHEMA-CORRECTED-TRANSFER-V56"
DATASET_ID = "doc2dial_v0_9_wood_schema_corrected_transfer_v56"
CAPABILITY = "wood_dialogue_span_selection_with_schema_corrected_abstention"
PROTOCOL_SHA256 = "f3a5282189b5a8bc4e730d880dd17fb9f296bde21ddfa8e0fea0c85e7fa27c90"
SOURCE_ARCHIVE_SHA256 = v55.SOURCE_ARCHIVE_SHA256
SOURCE_MEMBERS = dict(v55.SOURCE_MEMBERS)
STAGES = v55.STAGES
SAMPLE_SALTS = {
    "development": "FRC-DOC2DIAL-WOOD-V56-DEVELOPMENT|",
    "confirmation": "FRC-DOC2DIAL-WOOD-V56-CONFIRMATION|",
}
DECOY_SALT = "FRC-DOC2DIAL-WOOD-V56-DECOY|"
TARGET_CASES = v55.TARGET_CASES
TARGET_PER_GROUP = v55.TARGET_PER_GROUP
MAX_CASES_PER_DIALOGUE = v55.MAX_CASES_PER_DIALOGUE
MAX_CASES_PER_DOCUMENT_PER_STATE = v55.MAX_CASES_PER_DOCUMENT_PER_STATE
MINIMUM_DIALOGUES = v55.MINIMUM_DIALOGUES
MINIMUM_DOCUMENTS = v55.MINIMUM_DOCUMENTS
DECOY_COUNT = v55.DECOY_COUNT
MINIMUM_STRATUM_CASES = v55.MINIMUM_STRATUM_CASES
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260816
BUDGETS = v55.BUDGETS
CANDIDATE = v55.CANDIDATE
SAME_GATE_FRC_ABLATION = v55.SAME_GATE_FRC_ABLATION
GATED_NON_FRC_METHODS = v55.GATED_NON_FRC_METHODS
METHODS = v55.METHODS


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


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _ordered_turns(raw_turns: Any) -> list[Any]:
    if isinstance(raw_turns, list):
        return list(raw_turns)
    if not isinstance(raw_turns, dict):
        raise ValueError("Doc2Dial wOOD v56 turns must be a list or keyed object")
    numeric: list[tuple[int, Any]] = []
    for key, value in raw_turns.items():
        try:
            integer = int(str(key))
        except ValueError as exc:
            raise ValueError(
                "Doc2Dial wOOD v56 keyed turn id must be an integer"
            ) from exc
        numeric.append((integer, value))
    numeric.sort(key=lambda row: row[0])
    if len({value for value, _ in numeric}) != len(numeric):
        raise ValueError("Doc2Dial wOOD v56 keyed turn ids are not unique")
    return [value for _, value in numeric]


def _normalise_dialogue_source(source: dict[str, Any]) -> dict[str, Any]:
    root = source.get("dial_data")
    if not isinstance(root, dict):
        raise ValueError("Doc2Dial wOOD v56 dialogue root is absent")
    output: dict[str, Any] = {"dial_data": {}}
    for domain, documents in root.items():
        if not isinstance(documents, dict):
            raise ValueError("Doc2Dial wOOD v56 document-dialogue map is invalid")
        output_documents: dict[str, list[dict[str, Any]]] = {}
        for doc_id, dialogues in documents.items():
            if not isinstance(dialogues, list):
                raise ValueError("Doc2Dial wOOD v56 dialogues must be a list")
            output_dialogues = []
            for raw_dialogue in dialogues:
                if not isinstance(raw_dialogue, dict):
                    raise ValueError("Doc2Dial wOOD v56 dialogue must be an object")
                dialogue = dict(raw_dialogue)
                dialogue["turns"] = [
                    v55._normalise_turn(turn)
                    for turn in _ordered_turns(raw_dialogue.get("turns"))
                ]
                output_dialogues.append(dialogue)
            output_documents[str(doc_id)] = output_dialogues
        output["dial_data"][str(domain)] = output_documents
    return output


def read_stage_sources(
    archive_path: Path, stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported Doc2Dial wOOD v56 stage: {stage}")
    documents = v55._read_member(archive_path, SOURCE_MEMBERS["documents"])
    dialogues = _normalise_dialogue_source(
        v55._read_member(archive_path, SOURCE_MEMBERS[stage])
    )
    return documents, dialogues


def _subtype(user: dict[str, Any], target: dict[str, Any], reference_ids: list[str]) -> str:
    if reference_ids:
        return "answer_bearing"
    return (
        "ood_act"
        if v55._is_ood(user.get("da")) or v55._is_ood(target.get("da"))
        else "other_empty"
    )


def extract_cases(
    document_source: dict[str, Any], dialogue_source: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents = {
        (str(row["domain"]), str(row["doc_id"])): row
        for row in v54.iter_documents(document_source)
    }
    aliases = {key: v54._span_aliases(value) for key, value in documents.items()}
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    candidate_agent_turns = 0
    seen_cases: set[tuple[str, str]] = set()
    for dialogue in v54.iter_dialogues(dialogue_source):
        document_key = (str(dialogue["domain"]), str(dialogue["doc_id"]))
        if document_key not in documents:
            raise ValueError("Doc2Dial wOOD v56 dialogue document is missing")
        turns = list(dialogue["turns"])
        for position, target in enumerate(turns):
            if position == 0 or not isinstance(target, dict):
                continue
            user = turns[position - 1]
            if not isinstance(user, dict):
                continue
            if _normalise(target.get("role")).lower() != "agent":
                continue
            if _normalise(user.get("role")).lower() != "user":
                continue
            candidate_agent_turns += 1
            turn_id = _normalise(target.get("turn_id"))
            query = _normalise(user.get("utterance"))
            references = target.get("references")
            if not turn_id:
                exclusions["empty_target_turn_id"] += 1
                continue
            if not query:
                exclusions["empty_immediately_preceding_user_utterance"] += 1
                continue
            if not isinstance(references, list):
                exclusions["references_not_list"] += 1
                continue
            reference_ids: list[str] = []
            invalid = False
            for reference in references:
                if not isinstance(reference, dict):
                    invalid = True
                    break
                sp_id = _normalise(reference.get("sp_id"))
                if not sp_id or sp_id not in aliases[document_key]:
                    invalid = True
                    break
                reference_ids.append(sp_id)
            if invalid:
                exclusions["unmapped_nonempty_reference"] += 1
                continue
            case_key = (str(dialogue["dial_id"]), turn_id)
            if case_key in seen_cases:
                raise ValueError("Doc2Dial wOOD v56 case is not unique")
            seen_cases.add(case_key)
            references_unique = sorted(set(reference_ids))
            rows.append(
                {
                    "domain": str(dialogue["domain"]),
                    "doc_id": str(dialogue["doc_id"]),
                    "dial_id": str(dialogue["dial_id"]),
                    "target_turn_id": turn_id,
                    "turn_position": position,
                    "query": query,
                    "reference_ids": references_unique,
                    "answer_state": (
                        "answer_bearing" if references_unique else "no_answer"
                    ),
                    "no_answer_subtype": _subtype(user, target, references_unique),
                }
            )
    excluded = sum(exclusions.values())
    return rows, {
        "candidate_agent_turns": candidate_agent_turns,
        "eligible_cases": len(rows),
        "schema_excluded_cases": excluded,
        "schema_exclusion_rate": (
            excluded / candidate_agent_turns if candidate_agent_turns else 1.0
        ),
        "schema_exclusion_reasons": dict(sorted(exclusions.items())),
        "eligible_no_answer_subtypes": dict(
            Counter(
                row["no_answer_subtype"]
                for row in rows
                if row["answer_state"] == "no_answer"
            )
        ),
    }


def _sample_key(row: dict[str, Any], stage: str) -> tuple[str, str, str]:
    return (
        _hash(
            SAMPLE_SALTS[stage],
            str(row["answer_state"]),
            str(row["dial_id"]),
            str(row["target_turn_id"]),
        ),
        str(row["dial_id"]),
        str(row["target_turn_id"]),
    )


def select_balanced_sample(
    document_source: dict[str, Any],
    dialogue_source: dict[str, Any],
    *,
    stage: str,
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_dialogue: int = MAX_CASES_PER_DIALOGUE,
    maximum_cases_per_document_per_state: int = MAX_CASES_PER_DOCUMENT_PER_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported Doc2Dial wOOD v56 stage: {stage}")
    rows, schema = extract_cases(document_source, dialogue_source)
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in rows if row["answer_state"] == group],
            key=lambda row: _sample_key(row, stage),
        )
        for group in groups
    }
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("Doc2Dial wOOD v56 lacks the registered answer-state quota")
    selected: list[dict[str, Any]] = []
    selected_by_group: Counter[str] = Counter()
    dialogue_counts: Counter[str] = Counter()
    document_state_counts: Counter[tuple[str, str, str]] = Counter()
    cursors = {group: 0 for group in groups}
    while any(selected_by_group[group] < target_per_group for group in groups):
        progressed = False
        for group in groups:
            if selected_by_group[group] >= target_per_group:
                continue
            while cursors[group] < len(by_group[group]):
                row = by_group[group][cursors[group]]
                cursors[group] += 1
                dial_id = str(row["dial_id"])
                doc_state = (
                    str(row["domain"]),
                    str(row["doc_id"]),
                    group,
                )
                if dialogue_counts[dial_id] >= maximum_cases_per_dialogue:
                    continue
                if document_state_counts[doc_state] >= maximum_cases_per_document_per_state:
                    continue
                selected.append(row)
                selected_by_group[group] += 1
                dialogue_counts[dial_id] += 1
                document_state_counts[doc_state] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError("Doc2Dial wOOD v56 caps prevent the balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    documents = {(row["domain"], row["doc_id"]) for row in selected}
    return selected, {
        "stage": stage,
        "target_cases": target_per_group * 2,
        "selected_cases": len(selected),
        "eligible_answer_state_counts": dict(
            Counter(row["answer_state"] for row in rows)
        ),
        "selected_answer_state_counts": dict(selected_by_group),
        "selected_no_answer_subtypes": dict(
            Counter(
                row["no_answer_subtype"]
                for row in selected
                if row["answer_state"] == "no_answer"
            )
        ),
        "selected_dialogues": len(dialogue_counts),
        "selected_documents": len(documents),
        "maximum_cases_per_dialogue": max(dialogue_counts.values(), default=0),
        "maximum_cases_per_document_per_answer_state": max(
            document_state_counts.values(), default=0
        ),
        **schema,
    }


def _select_probe_documents(
    own_key: str,
    case_id: str,
    metadata: dict[str, dict[str, Any]],
) -> tuple[list[str], bool]:
    own = metadata[own_key]
    same_quartile: list[str] = []
    same_domain: list[str] = []
    fallback: list[str] = []
    for key, row in metadata.items():
        if key == own_key:
            continue
        if row["domain_commitment"] == own["domain_commitment"]:
            target = (
                same_quartile
                if row["length_quartile"] == own["length_quartile"]
                else same_domain
            )
            target.append(key)
        else:
            fallback.append(key)

    def ordered(values: Sequence[str]) -> list[str]:
        return sorted(values, key=lambda key: (_hash(DECOY_SALT, case_id, key), key))

    selected = (ordered(same_quartile) + ordered(same_domain))[:DECOY_COUNT]
    used_fallback = len(selected) < DECOY_COUNT
    if used_fallback:
        selected.extend(ordered(fallback)[: DECOY_COUNT - len(selected)])
    if len(selected) != DECOY_COUNT or len(set(selected)) != DECOY_COUNT:
        raise ValueError("Doc2Dial wOOD v56 cannot construct seven unique decoys")
    return selected, used_fallback


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]],
    document_source: dict[str, Any],
    tokenizer: Any,
    *,
    stage: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    documents = list(v54.iter_documents(document_source))
    lengths = [len(str(row["doc_text"])) for row in documents]
    length_boundaries = [
        round(float(value), 6) for value in np.quantile(lengths, [0.25, 0.5, 0.75])
    ]
    store: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    aliases_by_key: dict[str, list[dict[str, str]]] = {}
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    for document in documents:
        raw_key = (str(document["domain"]), str(document["doc_id"]))
        doc_key = _opaque("w56d", *raw_key)
        candidates, aliases = v54._canonical_span_rows(document, tokenizer)
        row = {
            "schema_version": "frc-doc2dial-wood-v56-blind-document-v1",
            "id": doc_key,
            "candidates": [
                {
                    "id": str(candidate["id"]),
                    "source_id": str(candidate["source_id"]),
                    "text": str(candidate["text"]),
                    "token_count": int(candidate["token_count"]),
                }
                for candidate in candidates
            ],
            "gold_fields_visible_to_scorer": False,
        }
        if not row["candidates"] or v50._contains_forbidden_blind_key(row):
            raise AssertionError("Doc2Dial wOOD v56 blind document is invalid")
        store.append(row)
        metadata[doc_key] = {
            "domain_commitment": _hash(str(document["domain"])),
            "length_quartile": v54._quartile(
                len(str(document["doc_text"])), length_boundaries
            ),
        }
        aliases_by_key[doc_key] = aliases
        pool_sizes.append(len(candidates))
        token_costs.extend(int(candidate["token_count"]) for candidate in candidates)
    store.sort(key=lambda row: str(row["id"]))
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    positions: list[int] = []
    fallback_count = 0
    for source in selected:
        own_key = _opaque("w56d", str(source["domain"]), str(source["doc_id"]))
        case_id = _opaque(
            "w56c",
            stage,
            str(source["dial_id"]),
            str(source["target_turn_id"]),
        )
        decoys, fallback_used = _select_probe_documents(own_key, case_id, metadata)
        fallback_count += int(fallback_used)
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "stage": stage,
            "id": case_id,
            "query": str(source["query"]),
            "own_doc_key": own_key,
            "probe_doc_keys": [own_key, *decoys],
            "decoy_fallback_used": fallback_used,
            "gold_fields_visible_to_scorer": False,
        }
        if v50._contains_forbidden_blind_key(blind):
            raise AssertionError("Doc2Dial wOOD v56 gold leaked into blind case")
        prepared.append(blind)
        maps.append(
            {
                "schema_version": "frc-doc2dial-wood-v56-candidate-map-v1",
                "id": case_id,
                "domain_commitment": _hash(str(source["domain"])),
                "document_commitment": _hash(str(source["doc_id"])),
                "dialogue_commitment": _hash(str(source["dial_id"])),
                "target_turn_commitment": _hash(str(source["target_turn_id"])),
                "turn_position": int(source["turn_position"]),
                "span_aliases": aliases_by_key[own_key],
            }
        )
        positions.append(int(source["turn_position"]))
    position_boundaries = [
        round(float(value), 6)
        for value in np.quantile(positions, [0.25, 0.5, 0.75])
    ]
    return prepared, maps, store, {
        "stage": stage,
        "prepared_cases": len(prepared),
        "blind_documents": len(store),
        "source_candidate_pool_size": v54._distribution(pool_sizes),
        "source_candidate_token_cost": v54._distribution(token_costs),
        "document_length_codepoints": v54._distribution(lengths),
        "document_length_quartile_boundaries": length_boundaries,
        "dialogue_turn_position": v54._distribution(positions),
        "dialogue_turn_position_quartile_boundaries": position_boundaries,
        "decoy_fallback_count": fallback_count,
        "decoy_fallback_rate": fallback_count / len(prepared) if prepared else 1.0,
        "gold_fields_exported_to_blind_cache": False,
        "ready_for_query_generation": len(prepared) == len(selected),
    }


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = v54.build_deterministic_queries(prepared_rows)
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
    return rows


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("Doc2Dial wOOD v56 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenDoc2DialWoodSchemaScorer(v55.FrozenDoc2DialWoodScorer):
    """Unchanged v54 scorer and gate with v56 schema-correction metadata."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
            row["transferred_retrieval_gate_selector_unchanged"] = True
        return rows


build_retrieval_pool = v55.build_retrieval_pool
rank_concurrent_role_closure_details = v55.rank_concurrent_role_closure_details
select_v56 = v55.select_v55


def _source_index(
    document_source: dict[str, Any], dialogue_source: dict[str, Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows, _ = extract_cases(document_source, dialogue_source)
    result = {}
    for row in rows:
        key = (
            _hash(str(row["doc_id"])),
            _hash(str(row["dial_id"])),
            _hash(str(row["target_turn_id"])),
        )
        if key in result:
            raise ValueError("Doc2Dial wOOD v56 source commitment is not unique")
        result[key] = row
    return result


def build_gold_rows(
    document_source: dict[str, Any],
    dialogue_source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    document_length_quartile_boundaries: Sequence[float],
    turn_position_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    source_cases = _source_index(document_source, dialogue_source)
    documents = {
        (str(row["domain"]), str(row["doc_id"])): row
        for row in v54.iter_documents(document_source)
    }
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    pool_sizes = [len(row.get("candidates", [])) for row in scored_rows]
    pool_boundaries = [
        round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])
    ]
    result = []
    for candidate_map in candidate_maps:
        key = (
            str(candidate_map["document_commitment"]),
            str(candidate_map["dialogue_commitment"]),
            str(candidate_map["target_turn_commitment"]),
        )
        source = source_cases.get(key)
        scored = scored_by_id.get(str(candidate_map["id"]))
        if source is None or scored is None:
            raise ValueError("Doc2Dial wOOD v56 gold join is incomplete")
        alias_map = {
            str(row["sp_commitment"]): str(row["candidate_id"])
            for row in candidate_map["span_aliases"]
        }
        gold_ids = {
            alias_map[_hash(str(sp_id))] for sp_id in source["reference_ids"]
        }
        available = {str(row["id"]) for row in scored.get("candidates", [])}
        document = documents[(str(source["domain"]), str(source["doc_id"]))]
        pool_size = len(available)
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "document_cluster": str(candidate_map["document_commitment"]),
                "dialogue_cluster": str(candidate_map["dialogue_commitment"]),
                "answer_state": str(source["answer_state"]),
                "no_answer_subtype": str(source["no_answer_subtype"]),
                "gold_candidate_ids": sorted(gold_ids),
                "reference_count": len(gold_ids),
                "reference_count_group": v54._reference_count_group(len(gold_ids)),
                "domain": str(source["domain"]),
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": v54._quartile(
                    pool_size, pool_boundaries
                ),
                "document_length_quartile": v54._quartile(
                    len(str(document["doc_text"])),
                    document_length_quartile_boundaries,
                ),
                "dialogue_turn_position_quartile": v54._quartile(
                    int(candidate_map["turn_position"]),
                    turn_position_quartile_boundaries,
                ),
                "candidate_ceiling_complete": gold_ids <= available,
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    value = v55.build_candidate_coverage(gold_rows, sampling)
    value["schema_version"] = "frc-doc2dial-wood-v56-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    value["no_answer_subtypes"] = dict(
        Counter(
            row["no_answer_subtype"]
            for row in gold_rows
            if row["answer_state"] == "no_answer"
        )
    )
    return value


def evaluate_stage(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old_seed = v55.BOOTSTRAP_SEED
    old_resamples = v55.BOOTSTRAP_RESAMPLES
    v55.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    v55.BOOTSTRAP_RESAMPLES = BOOTSTRAP_RESAMPLES
    try:
        report, evidence = v55.evaluate_stage(
            gold_rows,
            scored_rows,
            query_summary,
            source_artifacts,
            stage=stage,
        )
    finally:
        v55.BOOTSTRAP_SEED = old_seed
        v55.BOOTSTRAP_RESAMPLES = old_resamples
    for gold, row in zip(gold_rows, evidence, strict=True):
        row["no_answer_subtype"] = str(gold["no_answer_subtype"])
    subtype_strata = {}
    for subtype in sorted({row["no_answer_subtype"] for row in evidence}):
        subset = [row for row in evidence if row["no_answer_subtype"] == subtype]
        if len(subset) < MINIMUM_STRATUM_CASES:
            continue
        strongest, delta = v55._stratum_delta(subset)
        subtype_strata[f"no_answer_subtype:{subtype}"] = {
            "cases": len(subset),
            "strongest_shared_gate_non_frc": strongest,
            "delta": delta,
        }
    analysis = report["analysis"]
    analysis["supported_stratum_deltas"].update(subtype_strata)
    minimum_delta = min(
        [
            *analysis["budget_deltas"].values(),
            *(
                row["delta"]
                for row in analysis["supported_stratum_deltas"].values()
            ),
        ],
        default=-math.inf,
    )
    analysis["minimum_budget_or_supported_stratum_delta"] = round(minimum_delta, 6)
    analysis["support_checks"][
        "every_budget_and_supported_stratum_delta_at_least_minus_0_03"
    ] = minimum_delta >= -0.03
    supported = all(analysis["support_checks"].values())
    if stage == "development":
        status = (
            "DOC2DIAL_WOOD_V56_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if supported
            else "DOC2DIAL_WOOD_V56_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "DOC2DIAL_WOOD_V56_SCHEMA_CORRECTED_TRANSFER_SUPPORT_ESTABLISHED"
            if supported
            else "DOC2DIAL_WOOD_V56_SCHEMA_CORRECTED_TRANSFER_SUPPORT_NOT_ESTABLISHED"
        )
    report["schema_version"] = "frc-doc2dial-wood-schema-corrected-report-v56"
    report["experiment_id"] = EXPERIMENT_ID
    report["metadata"]["dataset_id"] = DATASET_ID
    report["metadata"]["pre_outcome_schema_access_disclosed"] = True
    report["metadata"]["no_answer_subtypes"] = dict(
        Counter(
            row["no_answer_subtype"]
            for row in evidence
            if row["answer_state"] == "no_answer"
        )
    )
    analysis["schema_corrections"] = {
        "keyed_turn_objects_sorted_by_integer_key": True,
        "all_empty_references_are_no_answer": True,
        "ood_act_is_descriptive_subtype_only": True,
        "retrieval_gate_selector_changed": False,
    }
    analysis["outcome"]["status"] = status
    analysis["outcome"]["support_established"] = supported
    analysis["outcome"]["confirmation_open_authorized"] = (
        stage == "development" and supported
    )
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
    outcome = analysis["outcome"]
    candidate = analysis["aggregates"][CANDIDATE]
    lines = [
        f"# Doc2Dial wOOD schema-corrected transfer ({rounded['metadata']['stage']}, v56)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/documents/dialogues: {rounded['metadata']['cases']}/{rounded['metadata']['documents']}/{rounded['metadata']['dialogues']}",
        f"- No-answer subtypes: {json.dumps(rounded['metadata']['no_answer_subtypes'], sort_keys=True)}",
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        f"- Candidate answer F1: {candidate['answer_bearing_macro_f1']:.6f}",
        f"- Candidate answer recall: {candidate['answer_macro_recall']:.6f}",
        f"- No-answer abstention accuracy: {candidate['no_answer_abstention_accuracy']:.6f}",
        f"- Answer-bearing gate pass rate: {analysis['gate_diagnostics']['answer_bearing_pass_rate']:.6f}",
        f"- No-answer gate pass rate: {analysis['gate_diagnostics']['no_answer_pass_rate']:.6f}",
        "- Retrieval/gate/selector changed from v54: `false`",
        "- Confirmation opened before gates: `false`",
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
            "v56 disclosed pre-outcome train schema access caused by v55. It changed only schema normalization and the official empty-reference interpretation; it is not an official shared-task result, SetR reproduction, flood-domain validation or rollout authorization.",
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
        raise ValueError("Doc2Dial wOOD v56 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    disclosure = value["pre_registration_source_access_disclosure"]
    if disclosure["wood_dev_confirmation_member_opened"]:
        raise ValueError("Doc2Dial wOOD v56 confirmation was opened too early")
    if disclosure[
        "utterance_span_text_reference_value_selected_case_query_candidate_neural_score_or_metric_inspected"
    ]:
        raise ValueError("Doc2Dial wOOD v56 outcome-bearing access occurred too early")
    if value["evaluation"]["budgets"] != list(BUDGETS):
        raise ValueError("Doc2Dial wOOD v56 budgets changed")
    return value


def validate_source_registration(
    registration_path: Path, *, protocol_path: Path, source_archive: Path
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("Doc2Dial wOOD v56 source registration protocol mismatch")
    if value["source"]["sha256"] != _sha256(source_archive):
        raise ValueError("Doc2Dial wOOD v56 archive changed")
    access = value["pre_registration_access"]
    if access["confirmation_member_parsed"] or access[
        "utterance_span_text_reference_value_selected_case_query_candidate_score_or_metric_printed_or_persisted"
    ]:
        raise ValueError("Doc2Dial wOOD v56 source boundary changed")
    return value


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
        "protocol_sha256": _sha256(protocol_path),
        "source_registration_sha256": _sha256(source_registration_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("Doc2Dial wOOD v56 implementation hash mismatch")
    if value.get("outcome_bearing_preparation_started_before_registration") is not False:
        raise ValueError("Doc2Dial wOOD v56 implementation was registered too late")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("Doc2Dial wOOD v56 synthetic invariants are incomplete")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BUDGETS",
    "CANDIDATE",
    "CAPABILITY",
    "DATASET_ID",
    "EXPERIMENT_ID",
    "FrozenDoc2DialWoodSchemaScorer",
    "GATED_NON_FRC_METHODS",
    "METHODS",
    "PROTOCOL_SHA256",
    "SAME_GATE_FRC_ABLATION",
    "SCHEMA_VERSION",
    "SOURCE_ARCHIVE_SHA256",
    "SOURCE_MEMBERS",
    "STAGES",
    "TARGET_CASES",
    "TARGET_PER_GROUP",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "build_retrieval_pool",
    "evaluate_stage",
    "extract_cases",
    "prepare_blind_cases",
    "rank_concurrent_role_closure_details",
    "read_stage_sources",
    "select_balanced_sample",
    "select_v56",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_registration",
    "write_report",
]
