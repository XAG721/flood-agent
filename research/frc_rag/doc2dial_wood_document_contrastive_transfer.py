"""Prospective Doc2Dial v0.9 wOOD transfer of the frozen v54 mechanism."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research.frc_rag.contractnli_native_zero_consensus_abstention as v50
import research.frc_rag.doc2dial_document_contrastive_role_closure as v54
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    _round_for_display,
)
from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_RANK
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import NON_FRC_BASELINES


SCHEMA_VERSION = "frc-doc2dial-wood-document-contrastive-transfer-v55"
EXPERIMENT_ID = "FRC-DOC2DIAL-WOOD-DOCUMENT-CONTRASTIVE-TRANSFER-V55"
DATASET_ID = "doc2dial_v0_9_wood_document_contrastive_transfer_v55"
CAPABILITY = "wood_dialogue_span_selection_with_contrastive_abstention"
PROTOCOL_SHA256 = "6765c57d535968272fc5b06d75295f876d9eb7e9d202fa884d0e22452e2d9fc5"
SOURCE_ARCHIVE_SHA256 = (
    "9143efd9d12ca30b1c772f65102b1c3e77d625fca03e69d316773acea5406786"
)
SOURCE_MEMBERS = {
    "documents": "doc2dial/v0.9/data/doc2dial_doc.json",
    "development": "doc2dial/v0.9/data/wOOD/doc2dial_dial_train.json",
    "confirmation": "doc2dial/v0.9/data/wOOD/doc2dial_dial_dev.json",
}
STAGES = ("development", "confirmation")
SAMPLE_SALTS = {
    "development": "FRC-DOC2DIAL-WOOD-V55-DEVELOPMENT|",
    "confirmation": "FRC-DOC2DIAL-WOOD-V55-CONFIRMATION|",
}
DECOY_SALT = "FRC-DOC2DIAL-WOOD-V55-DECOY|"
TARGET_CASES = 400
TARGET_PER_GROUP = 200
MAX_CASES_PER_DIALOGUE = 2
MAX_CASES_PER_DOCUMENT_PER_STATE = 4
MINIMUM_DIALOGUES = 160
MINIMUM_DOCUMENTS = 80
DECOY_COUNT = 7
MINIMUM_STRATUM_CASES = 40
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260815
BUDGETS = (128, 256, 512)

GATED_PREFIX = "document_contrastive_"
GATED_NON_FRC_METHODS = tuple(
    f"{GATED_PREFIX}{method}_v55" for method in NON_FRC_BASELINES
)
SAME_GATE_FRC_ABLATION = (
    "document_contrastive_low_core_only_frc_v54_transferred_v55"
)
CANDIDATE = (
    "document_contrastive_rank_concurrent_atomic_role_closure_frc_v54_transferred_v55"
)
METHODS = (
    *V49_METHODS,
    *GATED_NON_FRC_METHODS,
    SAME_GATE_FRC_ABLATION,
    CANDIDATE,
)
UNGATED_FRC_CONTROLS = (DYNAMIC_RANK, ADAPTIVE_ARGMAX, LOW_CORE_DIVERGENCE_V49)
V55_TO_V54 = {
    CANDIDATE: v54.CANDIDATE,
    SAME_GATE_FRC_ABLATION: v54.SAME_GATE_FRC_ABLATION,
    **{
        v55_method: v54_method
        for v55_method, v54_method in zip(
            GATED_NON_FRC_METHODS, v54.GATED_NON_FRC_METHODS, strict=True
        )
    },
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


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _read_member(archive_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if names.count(member) != 1:
            raise ValueError(f"Doc2Dial wOOD v55 archive lacks one {member}")
        raw = archive.read(member)
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Doc2Dial wOOD v55 JSON member must be an object")
    return value


def _normalise_reference(raw: Any) -> dict[str, str] | None:
    if isinstance(raw, str):
        value = _normalise(raw)
        return {"sp_id": value, "label": ""} if value else None
    if not isinstance(raw, dict):
        return None
    sp_id = _normalise(raw.get("sp_id") or raw.get("id_sp"))
    if not sp_id and len(raw) == 1:
        key, value = next(iter(raw.items()))
        if isinstance(value, str):
            sp_id = _normalise(key)
            return {"sp_id": sp_id, "label": _normalise(value)} if sp_id else None
    return (
        {"sp_id": sp_id, "label": _normalise(raw.get("label"))}
        if sp_id
        else None
    )


def _normalise_turn(raw_turn: Any) -> dict[str, Any]:
    if not isinstance(raw_turn, dict):
        raise ValueError("Doc2Dial wOOD v55 turn must be an object")
    if "references" in raw_turn:
        raw_references = raw_turn["references"]
    else:
        raw_references = raw_turn.get("reference", [])
    if raw_references is None:
        raw_references = []
    if isinstance(raw_references, dict):
        if any(key in raw_references for key in ("sp_id", "id_sp")):
            raw_references = [raw_references]
        else:
            raw_references = [
                {"sp_id": key, "label": value}
                for key, value in raw_references.items()
            ]
    if not isinstance(raw_references, list):
        raise ValueError("Doc2Dial wOOD v55 reference container is invalid")
    references = []
    for raw_reference in raw_references:
        reference = _normalise_reference(raw_reference)
        if reference is None:
            raise ValueError("Doc2Dial wOOD v55 reference is invalid")
        references.append(reference)
    result = dict(raw_turn)
    result.pop("reference", None)
    result["references"] = references
    return result


def _normalise_dialogue_source(source: dict[str, Any]) -> dict[str, Any]:
    root = source.get("dial_data")
    if not isinstance(root, dict):
        raise ValueError("Doc2Dial wOOD v55 dialogue root is absent")
    output: dict[str, Any] = {"dial_data": {}}
    for domain, documents in root.items():
        if not isinstance(documents, dict):
            raise ValueError("Doc2Dial wOOD v55 document-dialogue map is invalid")
        output_documents: dict[str, list[dict[str, Any]]] = {}
        for doc_id, dialogues in documents.items():
            if not isinstance(dialogues, list):
                raise ValueError("Doc2Dial wOOD v55 dialogues must be a list")
            output_dialogues = []
            for raw_dialogue in dialogues:
                if not isinstance(raw_dialogue, dict):
                    raise ValueError("Doc2Dial wOOD v55 dialogue must be an object")
                dialogue = dict(raw_dialogue)
                turns = raw_dialogue.get("turns")
                if not isinstance(turns, list):
                    raise ValueError("Doc2Dial wOOD v55 turns are absent")
                dialogue["turns"] = [_normalise_turn(turn) for turn in turns]
                output_dialogues.append(dialogue)
            output_documents[str(doc_id)] = output_dialogues
        output["dial_data"][str(domain)] = output_documents
    return output


def read_stage_sources(
    archive_path: Path, stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open the shared document member and exactly one wOOD dialogue member."""

    if stage not in STAGES:
        raise ValueError(f"Unsupported Doc2Dial wOOD v55 stage: {stage}")
    documents = _read_member(archive_path, SOURCE_MEMBERS["documents"])
    dialogues = _normalise_dialogue_source(
        _read_member(archive_path, SOURCE_MEMBERS[stage])
    )
    return documents, dialogues


def _is_ood(value: Any) -> bool:
    return _normalise(value).lower().endswith("/ood")


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
            raise ValueError("Doc2Dial wOOD v55 dialogue document is missing")
        turns = list(dialogue["turns"])
        for position, raw_target in enumerate(turns):
            if position == 0 or not isinstance(raw_target, dict):
                continue
            raw_user = turns[position - 1]
            if not isinstance(raw_user, dict):
                continue
            if _normalise(raw_target.get("role")).lower() != "agent":
                continue
            if _normalise(raw_user.get("role")).lower() != "user":
                continue
            candidate_agent_turns += 1
            turn_id = _normalise(raw_target.get("turn_id"))
            query = _normalise(raw_user.get("utterance"))
            references = raw_target.get("references")
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
            adjacent_ood = _is_ood(raw_user.get("da")) or _is_ood(
                raw_target.get("da")
            )
            if not reference_ids and not adjacent_ood:
                exclusions["empty_reference_without_adjacent_ood_act"] += 1
                continue
            case_key = (str(dialogue["dial_id"]), turn_id)
            if case_key in seen_cases:
                raise ValueError("Doc2Dial wOOD v55 case is not unique")
            seen_cases.add(case_key)
            rows.append(
                {
                    "domain": str(dialogue["domain"]),
                    "doc_id": str(dialogue["doc_id"]),
                    "dial_id": str(dialogue["dial_id"]),
                    "target_turn_id": turn_id,
                    "turn_position": position,
                    "query": query,
                    "reference_ids": sorted(set(reference_ids)),
                    "answer_state": (
                        "answer_bearing" if reference_ids else "no_answer"
                    ),
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
    }


def _sample_key(row: dict[str, Any], stage: str) -> tuple[str, str, str]:
    digest = _hash(
        SAMPLE_SALTS[stage],
        str(row["answer_state"]),
        str(row["dial_id"]),
        str(row["target_turn_id"]),
    )
    return digest, str(row["dial_id"]), str(row["target_turn_id"])


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
        raise ValueError(f"Unsupported Doc2Dial wOOD v55 stage: {stage}")
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
        raise ValueError("Doc2Dial wOOD v55 lacks the registered answer-state quota")
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
            values = by_group[group]
            while cursors[group] < len(values):
                row = values[cursors[group]]
                cursors[group] += 1
                dial_id = str(row["dial_id"])
                document_state = (
                    str(row["domain"]),
                    str(row["doc_id"]),
                    group,
                )
                if dialogue_counts[dial_id] >= maximum_cases_per_dialogue:
                    continue
                if (
                    document_state_counts[document_state]
                    >= maximum_cases_per_document_per_state
                ):
                    continue
                selected.append(row)
                selected_by_group[group] += 1
                dialogue_counts[dial_id] += 1
                document_state_counts[document_state] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError("Doc2Dial wOOD v55 caps prevent the balanced sample")
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
    same_quartile = []
    same_domain = []
    fallback = []
    for key, row in metadata.items():
        if key == own_key:
            continue
        if row["domain_commitment"] == own["domain_commitment"]:
            (same_quartile if row["length_quartile"] == own["length_quartile"] else same_domain).append(key)
        else:
            fallback.append(key)

    def ordered(values: Sequence[str]) -> list[str]:
        return sorted(values, key=lambda key: (_hash(DECOY_SALT, case_id, key), key))

    primary = ordered(same_quartile) + ordered(same_domain)
    selected = primary[:DECOY_COUNT]
    used_fallback = len(selected) < DECOY_COUNT
    if used_fallback:
        selected.extend(ordered(fallback)[: DECOY_COUNT - len(selected)])
    if len(selected) != DECOY_COUNT or len(set(selected)) != DECOY_COUNT:
        raise ValueError("Doc2Dial wOOD v55 cannot construct seven unique decoys")
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
        doc_key = _opaque("w55d", *raw_key)
        candidates, aliases = v54._canonical_span_rows(document, tokenizer)
        row = {
            "schema_version": "frc-doc2dial-wood-v55-blind-document-v1",
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
            raise AssertionError("Doc2Dial wOOD v55 blind document is invalid")
        store.append(row)
        metadata[doc_key] = {
            "domain_commitment": _hash(str(document["domain"])),
            "document_commitment": _hash(str(document["doc_id"])),
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
    turn_positions: list[int] = []
    fallback_count = 0
    for source in selected:
        own_key = _opaque("w55d", str(source["domain"]), str(source["doc_id"]))
        case_id = _opaque(
            "w55c",
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
            raise AssertionError("Doc2Dial wOOD v55 gold leaked into blind case")
        prepared.append(blind)
        maps.append(
            {
                "schema_version": "frc-doc2dial-wood-v55-candidate-map-v1",
                "id": case_id,
                "domain_commitment": _hash(str(source["domain"])),
                "document_commitment": _hash(str(source["doc_id"])),
                "dialogue_commitment": _hash(str(source["dial_id"])),
                "target_turn_commitment": _hash(str(source["target_turn_id"])),
                "turn_position": int(source["turn_position"]),
                "span_aliases": aliases_by_key[own_key],
            }
        )
        turn_positions.append(int(source["turn_position"]))
    turn_boundaries = [
        round(float(value), 6)
        for value in np.quantile(turn_positions, [0.25, 0.5, 0.75])
    ]
    return prepared, maps, store, {
        "stage": stage,
        "prepared_cases": len(prepared),
        "blind_documents": len(store),
        "source_candidate_pool_size": v54._distribution(pool_sizes),
        "source_candidate_token_cost": v54._distribution(token_costs),
        "document_length_codepoints": v54._distribution(lengths),
        "document_length_quartile_boundaries": length_boundaries,
        "dialogue_turn_position": v54._distribution(turn_positions),
        "dialogue_turn_position_quartile_boundaries": turn_boundaries,
        "decoy_fallback_count": fallback_count,
        "decoy_fallback_rate": fallback_count / len(prepared) if prepared else 1.0,
        "gold_fields_exported_to_blind_cache": False,
        "ready_for_query_generation": len(prepared) == len(selected),
    }


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = v54.build_deterministic_queries(prepared_rows)
    for row in result:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
    return result


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected = build_deterministic_queries(prepared_rows)
    if list(query_rows) != expected:
        raise ValueError("Doc2Dial wOOD v55 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenDoc2DialWoodScorer(v54.FrozenDoc2DialScorer):
    """Exact v54 neural scorer and gate with v55 transfer metadata."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
            row["transferred_v54_formula_unchanged"] = True
        return rows


build_retrieval_pool = v54.build_retrieval_pool
rank_concurrent_role_closure_details = v54.rank_concurrent_role_closure_details


def select_v55(
    candidates: Sequence[dict[str, Any]],
    method: str,
    *,
    gate_passed: bool,
    token_budget: int,
) -> list[dict[str, Any]]:
    if method in V49_METHODS:
        mapped = method
    else:
        mapped = V55_TO_V54.get(method)
        if mapped is None:
            raise ValueError(f"Unsupported Doc2Dial wOOD v55 method: {method}")
    return v54.select_v54(
        candidates,
        mapped,
        gate_passed=gate_passed,
        token_budget=token_budget,
    )


def _source_case_index(
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
            raise ValueError("Doc2Dial wOOD v55 source commitment is not unique")
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
    source_cases = _source_case_index(document_source, dialogue_source)
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
            raise ValueError("Doc2Dial wOOD v55 gold join is incomplete")
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
        position = int(candidate_map["turn_position"])
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "document_cluster": str(candidate_map["document_commitment"]),
                "dialogue_cluster": str(candidate_map["dialogue_commitment"]),
                "answer_state": str(source["answer_state"]),
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
                    position, turn_position_quartile_boundaries
                ),
                "candidate_ceiling_complete": gold_ids <= available,
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    value = v54.build_candidate_coverage(gold_rows, sampling)
    value["schema_version"] = "frc-doc2dial-wood-v55-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    return value


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]], candidate: str, baseline: str
) -> dict[str, float | int]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["document_cluster"])].append(
            v54._case_utility(row, candidate) - v54._case_utility(row, baseline)
        )
    clusters = sorted(by_cluster)
    arrays = [np.asarray(by_cluster[cluster], dtype=float) for cluster in clusters]
    point = float(np.mean(np.concatenate(arrays)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        picked = rng.integers(0, len(arrays), size=len(arrays))
        samples[index] = float(
            np.mean(np.concatenate([arrays[item] for item in picked]))
        )
    return {
        "point": round(point, 6),
        "ci_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_high": round(float(np.quantile(samples, 0.975)), 6),
        "clusters": len(clusters),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = v54._strongest(rows, GATED_NON_FRC_METHODS)
    candidate_value = float(np.mean([v54._case_utility(row, CANDIDATE) for row in rows]))
    baseline_value = float(
        np.mean([v54._case_utility(row, strongest) for row in rows])
    )
    return strongest, round(candidate_value - baseline_value, 6)


def evaluate_stage(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stage not in STAGES or len(gold_rows) != len(scored_rows) or not gold_rows:
        raise ValueError("Doc2Dial wOOD v55 gold and score caches differ")
    evidence = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("Doc2Dial wOOD v55 gold and score ids differ")
        candidates = v50.merge_v50_scored_candidates(dict(scored))
        gate = dict(scored["document_contrastive_gate"])
        gate_passed = bool(gate["assigned_document_rank_one"])
        configurations = {}
        for budget in BUDGETS:
            methods = {}
            for method in METHODS:
                selected = select_v55(
                    candidates,
                    method,
                    gate_passed=gate_passed,
                    token_budget=budget,
                )
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": v54._selection_metrics(
                        selected,
                        str(gold["answer_state"]),
                        gold["gold_candidate_ids"],
                    ),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "document_cluster": str(gold["document_cluster"]),
                "dialogue_cluster": str(gold["dialogue_cluster"]),
                "answer_state": str(gold["answer_state"]),
                "reference_count_group": str(gold["reference_count_group"]),
                "domain": str(gold["domain"]),
                "candidate_unit_count": int(gold["candidate_unit_count"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "document_length_quartile": str(
                    gold["document_length_quartile"]
                ),
                "dialogue_turn_position_quartile": str(
                    gold["dialogue_turn_position_quartile"]
                ),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "document_contrastive_gate": gate,
                "role_closure": rank_concurrent_role_closure_details(candidates),
                "configurations": configurations,
            }
        )
    aggregates = {method: v54._aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = v54._strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_frc = v54._strongest(evidence, UNGATED_FRC_CONTROLS)
    comparisons = {
        "candidate_minus_strongest_shared_gate_non_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_non_frc
        ),
        "candidate_minus_same_gate_frc_ablation": _cluster_bootstrap(
            evidence, CANDIDATE, SAME_GATE_FRC_ABLATION
        ),
        "candidate_minus_ungated_v49": _cluster_bootstrap(
            evidence, CANDIDATE, LOW_CORE_DIVERGENCE_V49
        ),
        "candidate_minus_strongest_ungated_frozen_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_frc
        ),
    }
    candidate = aggregates[CANDIDATE]
    ungated = aggregates[LOW_CORE_DIVERGENCE_V49]
    recall_drop = round(
        ungated["answer_macro_recall"] - candidate["answer_macro_recall"], 6
    )
    answer_pass = float(
        np.mean(
            [
                row["document_contrastive_gate"]["assigned_document_rank_one"]
                for row in evidence
                if row["answer_state"] == "answer_bearing"
            ]
        )
    )
    no_answer_pass = float(
        np.mean(
            [
                row["document_contrastive_gate"]["assigned_document_rank_one"]
                for row in evidence
                if row["answer_state"] == "no_answer"
            ]
        )
    )
    budget_deltas = {}
    for budget in BUDGETS:
        subset = []
        for row in evidence:
            clone = dict(row)
            clone["configurations"] = {str(budget): row["configurations"][str(budget)]}
            subset.append(clone)
        _, delta = _stratum_delta(subset)
        budget_deltas[str(budget)] = delta
    dimensions = {
        "answer_state": lambda row: row["answer_state"],
        "domain": lambda row: row["domain"],
        "candidate_pool_quartile": lambda row: row["candidate_pool_quartile"],
        "document_length_quartile": lambda row: row["document_length_quartile"],
        "dialogue_turn_position_quartile": lambda row: row[
            "dialogue_turn_position_quartile"
        ],
        "reference_count_group": lambda row: row["reference_count_group"],
        "gate_outcome": lambda row: (
            "passed"
            if row["document_contrastive_gate"]["assigned_document_rank_one"]
            else "abstained"
        ),
    }
    strata = {}
    for dimension, getter in dimensions.items():
        for value in sorted({str(getter(row)) for row in evidence}):
            subset = [row for row in evidence if str(getter(row)) == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(subset)
            strata[f"{dimension}:{value}"] = {
                "cases": len(subset),
                "strongest_shared_gate_non_frc": strongest,
                "delta": delta,
            }
    minimum_delta = min(
        [*budget_deltas.values(), *(row["delta"] for row in strata.values())],
        default=-math.inf,
    )
    answer_rows = [row for row in evidence if row["answer_state"] == "answer_bearing"]
    ceiling = float(
        np.mean([row["candidate_ceiling_complete"] for row in answer_rows])
    )
    state_counts = Counter(row["answer_state"] for row in evidence)
    documents = len({row["document_cluster"] for row in evidence})
    dialogues = len({row["dialogue_cluster"] for row in evidence})
    sampling = source_artifacts["sampling"]
    structural = source_artifacts["structural_census"]
    checks = {
        "exact_cases_equals_400": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": state_counts
        == {"answer_bearing": TARGET_PER_GROUP, "no_answer": TARGET_PER_GROUP},
        "minimum_dialogues_at_least_160": dialogues >= MINIMUM_DIALOGUES,
        "minimum_documents_at_least_80": documents >= MINIMUM_DOCUMENTS,
        "dialogue_cap_at_most_2": int(sampling["maximum_cases_per_dialogue"])
        <= MAX_CASES_PER_DIALOGUE,
        "document_state_cap_at_most_4": int(
            sampling["maximum_cases_per_document_per_answer_state"]
        )
        <= MAX_CASES_PER_DOCUMENT_PER_STATE,
        "schema_exclusion_rate_at_most_0_01": float(
            sampling["schema_exclusion_rate"]
        )
        <= 0.01,
        "candidate_ceiling_complete_rate_at_least_0_99": ceiling >= 0.99,
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_01": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        >= 0.01,
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_same_gate_frc_ablation_point_at_least_0_005": comparisons[
            "candidate_minus_same_gate_frc_ablation"
        ]["point"]
        >= 0.005,
        "candidate_minus_same_gate_frc_ablation_ci_low_above_0": comparisons[
            "candidate_minus_same_gate_frc_ablation"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_ungated_v49_point_at_least_0_01": comparisons[
            "candidate_minus_ungated_v49"
        ]["point"]
        >= 0.01,
        "candidate_minus_ungated_v49_ci_low_above_0": comparisons[
            "candidate_minus_ungated_v49"
        ]["ci_low"]
        > 0.0,
        "answer_macro_recall_drop_vs_ungated_v49_at_most_0_03": recall_drop <= 0.03,
        "no_answer_abstention_accuracy_at_least_0_5": candidate[
            "no_answer_abstention_accuracy"
        ]
        >= 0.5,
        "answer_bearing_gate_pass_rate_at_least_0_5": answer_pass >= 0.5,
        "candidate_abstention_rate_at_least_0_1": candidate["abstention_rate"]
        >= 0.1,
        "candidate_abstention_rate_at_most_0_8": candidate["abstention_rate"] <= 0.8,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_03": minimum_delta
        >= -0.03,
        "deterministic_decoy_fallback_rate_equals_0": float(
            structural["decoy_fallback_rate"]
        )
        == 0.0,
        "deterministic_query_fallback_rate_equals_0": query_summary["fallback_rate"]
        == 0.0,
        "score_or_prediction_fallback_rate_equals_0": float(
            source_artifacts.get("score_fallback_rate", 0.0)
        )
        == 0.0,
    }
    supported = all(checks.values())
    if stage == "development":
        status = (
            "DOC2DIAL_WOOD_V55_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if supported
            else "DOC2DIAL_WOOD_V55_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "DOC2DIAL_WOOD_V55_TRANSFER_SUPPORT_ESTABLISHED"
            if supported
            else "DOC2DIAL_WOOD_V55_TRANSFER_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "schema_version": "frc-doc2dial-wood-transfer-report-v55",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "stage": stage,
            "split": SOURCE_MEMBERS[stage],
            "cases": len(evidence),
            "documents": documents,
            "dialogues": dialogues,
            "answer_state_counts": dict(state_counts),
            "budgets": list(BUDGETS),
            "official_shared_task_result": False,
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "gold_joined_after_complete_score_cache": True,
            "v54_case_level_artifact_reused": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "transferred_v54_formula_unchanged": True,
            "gate_formula": "assigned document ranks first by maximum anchor reranker logit among assigned plus seven deterministic decoys",
            "selector_formula": "anchor Top-1 plus each atomic-role Top-1 only when also in anchor Top-3, under token budget",
            "aggregates": aggregates,
            "strongest_shared_gate_non_frc": strongest_non_frc,
            "strongest_ungated_frozen_frc": strongest_frc,
            "family_comparison": comparisons,
            "answer_macro_recall_drop_vs_ungated_v49": recall_drop,
            "candidate_ceiling_complete_rate": round(ceiling, 6),
            "gate_diagnostics": {
                "answer_bearing_pass_rate": round(answer_pass, 6),
                "no_answer_pass_rate": round(no_answer_pass, 6),
                "no_answer_rejection_rate": round(1.0 - no_answer_pass, 6),
                "probe_documents_per_case": DECOY_COUNT + 1,
                "randomization_p_value_on_rank_one": 1.0 / (DECOY_COUNT + 1),
            },
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "minimum_budget_or_supported_stratum_delta": round(minimum_delta, 6),
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_established": supported,
                "confirmation_open_authorized": stage == "development" and supported,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_stage_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
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
    outcome = analysis["outcome"]
    candidate = analysis["aggregates"][CANDIDATE]
    lines = [
        f"# Doc2Dial wOOD document-contrastive transfer ({rounded['metadata']['stage']}, v55)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/documents/dialogues: {rounded['metadata']['cases']}/{rounded['metadata']['documents']}/{rounded['metadata']['dialogues']}",
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        f"- Candidate answer F1: {candidate['answer_bearing_macro_f1']:.6f}",
        f"- Candidate answer recall: {candidate['answer_macro_recall']:.6f}",
        f"- No-answer abstention accuracy: {candidate['no_answer_abstention_accuracy']:.6f}",
        f"- Answer-bearing gate pass rate: {analysis['gate_diagnostics']['answer_bearing_pass_rate']:.6f}",
        f"- No-answer gate pass rate: {analysis['gate_diagnostics']['no_answer_pass_rate']:.6f}",
        f"- Abstention rate: {candidate['abstention_rate']:.6f}",
        "- Transferred v54 formula changed: `false`",
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
            "This is a balanced Doc2Dial v0.9 wOOD span-selection transfer experiment. It is not an official shared-task submission, response-generation result, SetR reproduction, selector adoption, or flood-domain expert validation.",
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
        raise ValueError("Doc2Dial wOOD v55 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = value["source_access_boundary_at_registration"]
    if boundary["v0_9_archive_downloaded"] or boundary[
        "v0_9_document_or_dialogue_content_opened"
    ]:
        raise ValueError("Doc2Dial wOOD v55 source was accessed before protocol")
    if value["evaluation"]["token_budgets"] != list(BUDGETS):
        raise ValueError("Doc2Dial wOOD v55 budgets changed")
    if value["transferred_deterministic_decoys"]["count"] != DECOY_COUNT:
        raise ValueError("Doc2Dial wOOD v55 decoy count changed")
    return value


def validate_source_registration(
    registration_path: Path, *, protocol_path: Path, source_archive: Path
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value["protocol_sha256_before_download"] != _sha256(protocol_path):
        raise ValueError("Doc2Dial wOOD v55 source registration protocol mismatch")
    if _sha256(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("Doc2Dial wOOD v55 archive changed")
    if value["source"]["sha256"] != SOURCE_ARCHIVE_SHA256:
        raise ValueError("Doc2Dial wOOD v55 registered archive hash changed")
    if any(
        value[key]
        for key in (
            "archive_member_content_read_before_registration",
            "document_content_opened",
            "development_content_opened",
            "confirmation_content_opened",
        )
    ):
        raise ValueError("Doc2Dial wOOD v55 member content was opened too early")
    registered = {
        key: str(row["path"]) for key, row in value["resolved_members"].items()
    }
    if registered != SOURCE_MEMBERS:
        raise ValueError("Doc2Dial wOOD v55 member resolution changed")
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
        raise ValueError("Doc2Dial wOOD v55 implementation hash mismatch")
    if value.get("archive_member_content_opened_before_registration") is not False:
        raise ValueError("Doc2Dial wOOD v55 implementation was registered too late")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("Doc2Dial wOOD v55 synthetic invariants are incomplete")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BUDGETS",
    "CANDIDATE",
    "CAPABILITY",
    "DATASET_ID",
    "EXPERIMENT_ID",
    "FrozenDoc2DialWoodScorer",
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
    "select_v55",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_registration",
    "write_report",
]
