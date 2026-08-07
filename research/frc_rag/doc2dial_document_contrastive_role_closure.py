"""Prospective Doc2Dial v54 document-contrastive role-closure experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import research.frc_rag.contractnli_native_zero_consensus_abstention as v50
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    _round_for_display,
    select_v49,
)
from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    _bm25,
)
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import NON_FRC_BASELINES


SCHEMA_VERSION = "frc-doc2dial-document-contrastive-role-closure-v54"
EXPERIMENT_ID = "FRC-DOC2DIAL-DOCUMENT-CONTRASTIVE-ROLE-CLOSURE-V54"
DATASET_ID = "doc2dial_v1_0_1_document_contrastive_v54"
CAPABILITY = "document_grounded_dialogue_evidence_selection_with_abstention"
PROTOCOL_SHA256 = "4e5b348ab823d16f60e2d92b4dcf92dcca02e3eb9d11ffba3e8228138d68e23c"

STAGES = ("development", "confirmation")
SOURCE_FILES = {
    "documents": "doc2dial_doc.json",
    "development": "doc2dial_dial_train.json",
    "confirmation": "doc2dial_dial_validation.json",
}
SOURCE_SHA256 = {
    "DATA_README.md": "d38cfe10afaa4337b87877e4b03e14c321bdc8ec8452f518266e669279499ed7",
    "doc2dial_doc.json": "547ed8ee5bfd7babdadfb6246b1b3fc5fa107a4abdef972bc63c9b717000f56f",
    "doc2dial_dial_train.json": "c5e66816b28137ae4ba82e310cdc691b7608b123eda23f1f8f7021f3df8a4dff",
    "doc2dial_dial_validation.json": "9b4118e345af47451c84e8e27c1dba2844fcaafeacb059380b4ac8fbb9b62a16",
}
SAMPLE_SALTS = {
    "development": "FRC-DOC2DIAL-V54-DEVELOPMENT|",
    "confirmation": "FRC-DOC2DIAL-V54-CONFIRMATION|",
}
DECOY_SALT = "FRC-DOC2DIAL-V54-DECOY|"
TARGET_CASES = 400
TARGET_PER_GROUP = 200
MAX_CASES_PER_DIALOGUE = 2
MAX_CASES_PER_DOCUMENT_PER_STATE = 4
MINIMUM_DIALOGUES = 160
MINIMUM_DOCUMENTS = 80
DECOY_COUNT = 7
MINIMUM_STRATUM_CASES = 40
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260814
RETRIEVAL_TOP_K_PER_METHOD = 48
RETRIEVAL_POOL_MAXIMUM = 96
PROBE_TOP_K_PER_METHOD = 8
PROBE_POOL_MAXIMUM = 16
ANCHOR_FRONTIER = 3
BUDGETS = (128, 256, 512)

GATED_PREFIX = "document_contrastive_"
GATED_NON_FRC_METHODS = tuple(
    f"{GATED_PREFIX}{method}_v54" for method in NON_FRC_BASELINES
)
SAME_GATE_FRC_ABLATION = "document_contrastive_low_core_only_frc_v54"
CANDIDATE = "document_contrastive_rank_concurrent_atomic_role_closure_frc_v54"
METHODS = (
    *V49_METHODS,
    *GATED_NON_FRC_METHODS,
    SAME_GATE_FRC_ABLATION,
    CANDIDATE,
)
UNGATED_FRC_CONTROLS = (DYNAMIC_RANK, ADAPTIVE_ARGMAX, LOW_CORE_DIVERGENCE_V49)

ROLE_ALIASES = {
    "anchor": "anchor",
    "condition": "first_fact",
    "solution": "second_fact_or_bridge",
    "exception": "counterevidence",
}
QUERY_TEMPLATES = {
    "anchor": "{query}",
    "first_fact": (
        "Find prerequisites, eligibility requirements, conditions, or definitions "
        "needed to answer: {query}"
    ),
    "second_fact_or_bridge": (
        "Find steps, benefits, outcomes, or direct answers that resolve: {query}"
    ),
    "counterevidence": (
        "Find exceptions, exclusions, deadlines, limitations, or cases where this "
        "does not apply: {query}"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _opaque(prefix: str, *parts: str) -> str:
    return f"{prefix}{_hash(*parts)[:22]}"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Doc2Dial v54 source must be an object: {path.name}")
    return value


def read_stage_sources(
    source_root: Path, stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open the shared document source and exactly one registered dialogue split."""

    if stage not in STAGES:
        raise ValueError(f"Unsupported Doc2Dial v54 stage: {stage}")
    documents = _read_json_object(source_root / SOURCE_FILES["documents"])
    dialogues = _read_json_object(source_root / SOURCE_FILES[stage])
    return documents, dialogues


def _document_root(source: dict[str, Any]) -> dict[str, Any]:
    root = source.get("doc_data", source)
    if not isinstance(root, dict):
        raise ValueError("Doc2Dial v54 document root must be an object")
    return root


def iter_documents(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    for raw_domain, payload in _document_root(source).items():
        domain = _normalise(raw_domain)
        if not domain or not isinstance(payload, dict):
            raise ValueError("Doc2Dial v54 domain document map is invalid")
        for raw_key, raw_document in payload.items():
            if not isinstance(raw_document, dict):
                raise ValueError("Doc2Dial v54 document must be an object")
            doc_id = _normalise(raw_document.get("doc_id") or raw_key)
            actual_domain = _normalise(raw_document.get("domain") or domain)
            text = str(raw_document.get("doc_text", ""))
            spans = raw_document.get("spans")
            if (
                not doc_id
                or not actual_domain
                or not text
                or not isinstance(spans, dict)
                or not spans
            ):
                raise ValueError("Doc2Dial v54 document schema is incomplete")
            key = (actual_domain, doc_id)
            if key in seen:
                raise ValueError("Doc2Dial v54 document key is not unique")
            seen.add(key)
            yield {
                "domain": actual_domain,
                "doc_id": doc_id,
                "title": _normalise(raw_document.get("title")),
                "doc_text": text,
                "spans": spans,
            }


def _dialogue_root(source: dict[str, Any]) -> dict[str, Any]:
    root = source.get("dial_data", source)
    if not isinstance(root, dict):
        raise ValueError("Doc2Dial v54 dialogue root must be an object")
    return root


def _walk_dialogues(
    value: Any, *, domain: str, document_hint: str | None = None
) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _walk_dialogues(
                item, domain=domain, document_hint=document_hint
            )
        return
    if not isinstance(value, dict):
        raise ValueError("Doc2Dial v54 dialogue container is invalid")
    if "turns" in value:
        result = dict(value)
        result.setdefault("domain", domain)
        if document_hint:
            result.setdefault("doc_id", document_hint)
        yield result
        return
    for raw_key, item in value.items():
        hint = document_hint or _normalise(raw_key)
        yield from _walk_dialogues(item, domain=domain, document_hint=hint)


def iter_dialogues(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for raw_domain, payload in _dialogue_root(source).items():
        domain = _normalise(raw_domain)
        if not domain:
            raise ValueError("Doc2Dial v54 dialogue domain is empty")
        for raw_dialogue in _walk_dialogues(payload, domain=domain):
            dial_id = _normalise(raw_dialogue.get("dial_id"))
            doc_id = _normalise(raw_dialogue.get("doc_id"))
            actual_domain = _normalise(raw_dialogue.get("domain") or domain)
            turns = raw_dialogue.get("turns")
            if not dial_id or not doc_id or not isinstance(turns, list):
                raise ValueError("Doc2Dial v54 dialogue schema is incomplete")
            if dial_id in seen:
                raise ValueError("Doc2Dial v54 dialogue id is not unique")
            seen.add(dial_id)
            yield {
                "domain": actual_domain,
                "doc_id": doc_id,
                "dial_id": dial_id,
                "turns": turns,
            }


def _span_aliases(document: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for raw_key, raw_span in document["spans"].items():
        if not isinstance(raw_span, dict):
            raise ValueError("Doc2Dial v54 span must be an object")
        sp_id = _normalise(raw_span.get("sp_id") or raw_key)
        if not sp_id:
            raise ValueError("Doc2Dial v54 span id is empty")
        aliases.add(sp_id)
    return aliases


def extract_cases(
    document_source: dict[str, Any], dialogue_source: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents = {
        (str(row["domain"]), str(row["doc_id"])): row
        for row in iter_documents(document_source)
    }
    aliases = {key: _span_aliases(value) for key, value in documents.items()}
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    candidate_agent_turns = 0
    seen_cases: set[tuple[str, str]] = set()
    for dialogue in iter_dialogues(dialogue_source):
        document_key = (str(dialogue["domain"]), str(dialogue["doc_id"]))
        if document_key not in documents:
            raise ValueError("Doc2Dial v54 dialogue document is missing")
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
            for raw_reference in references:
                if not isinstance(raw_reference, dict):
                    invalid = True
                    break
                sp_id = _normalise(raw_reference.get("sp_id"))
                if not sp_id or sp_id not in aliases[document_key]:
                    invalid = True
                    break
                reference_ids.append(sp_id)
            if invalid:
                exclusions["unmapped_nonempty_reference"] += 1
                continue
            case_key = (str(dialogue["dial_id"]), turn_id)
            if case_key in seen_cases:
                raise ValueError("Doc2Dial v54 dialogue-turn case is not unique")
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
        raise ValueError(f"Unsupported Doc2Dial v54 stage: {stage}")
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
        raise ValueError("Doc2Dial v54 source lacks the registered answer-state quota")

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
            raise ValueError("Doc2Dial v54 caps prevent the registered balanced sample")

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


def _token_count(tokenizer: Any, text: str) -> int:
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def _canonical_span_rows(
    document: dict[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    text = str(document["doc_text"])
    raw_rows: list[dict[str, Any]] = []
    for raw_key, raw_span in document["spans"].items():
        if not isinstance(raw_span, dict):
            raise ValueError("Doc2Dial v54 span must be an object")
        sp_id = _normalise(raw_span.get("sp_id") or raw_key)
        span_text = str(raw_span.get("text_sp", ""))
        try:
            start = int(raw_span.get("start_sp"))
            raw_end = int(raw_span.get("end_sp"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Doc2Dial v54 span offsets are invalid") from exc
        if not sp_id or not span_text or start < 0 or raw_end < start:
            raise ValueError("Doc2Dial v54 span schema is incomplete")
        if text[start:raw_end] == span_text:
            end = raw_end
        elif text[start : raw_end + 1] == span_text:
            end = raw_end + 1
        else:
            raise ValueError("Doc2Dial v54 span text and offsets disagree")
        raw_rows.append(
            {"sp_id": sp_id, "start": start, "end": end, "text": span_text}
        )
    raw_rows.sort(key=lambda row: (row["start"], row["end"], row["sp_id"]))
    canonical_by_content: dict[tuple[int, int, str], dict[str, Any]] = {}
    aliases: list[dict[str, str]] = []
    for row in raw_rows:
        content_key = (int(row["start"]), int(row["end"]), str(row["text"]))
        candidate = canonical_by_content.get(content_key)
        if candidate is None:
            candidate_id = _opaque(
                "d2s",
                str(document["domain"]),
                str(document["doc_id"]),
                str(row["sp_id"]),
            )
            candidate = {
                "id": candidate_id,
                "source_id": candidate_id,
                "text": str(row["text"]),
                "token_count": _token_count(tokenizer, str(row["text"])),
                "start": int(row["start"]),
                "end": int(row["end"]),
            }
            canonical_by_content[content_key] = candidate
        aliases.append(
            {
                "sp_commitment": _hash(str(row["sp_id"])),
                "candidate_id": str(candidate["id"]),
            }
        )
    candidates = sorted(
        canonical_by_content.values(),
        key=lambda row: (row["start"], row["end"], row["id"]),
    )
    return candidates, sorted(aliases, key=lambda row: row["sp_commitment"])


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "mean": 0.0, "maximum": 0}
    return {
        "minimum": min(values),
        "mean": round(float(np.mean(values)), 6),
        "maximum": max(values),
    }


def _quartile(value: int, boundaries: Sequence[float]) -> str:
    if value <= float(boundaries[0]):
        return "q1"
    if value <= float(boundaries[1]):
        return "q2"
    if value <= float(boundaries[2]):
        return "q3"
    return "q4"


def _select_probe_documents(
    own_key: str,
    case_id: str,
    metadata: dict[str, dict[str, Any]],
    *,
    count: int = DECOY_COUNT,
) -> tuple[list[str], bool]:
    own = metadata[own_key]
    same_quartile = []
    same_domain = []
    fallback = []
    for key, row in metadata.items():
        if key == own_key:
            continue
        if row["domain_commitment"] == own["domain_commitment"]:
            if row["length_quartile"] == own["length_quartile"]:
                same_quartile.append(key)
            else:
                same_domain.append(key)
        else:
            fallback.append(key)

    def ordered(values: Sequence[str]) -> list[str]:
        return sorted(values, key=lambda key: (_hash(DECOY_SALT, case_id, key), key))

    primary = ordered(same_quartile) + ordered(same_domain)
    selected = primary[:count]
    used_fallback = len(selected) < count
    if used_fallback:
        selected.extend(ordered(fallback)[: count - len(selected)])
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError("Doc2Dial v54 cannot construct seven unique decoys")
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
    documents = list(iter_documents(document_source))
    lengths = [len(str(row["doc_text"])) for row in documents]
    length_boundaries = [
        round(float(value), 6) for value in np.quantile(lengths, [0.25, 0.5, 0.75])
    ]
    document_store: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    raw_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    aliases_by_key: dict[str, list[dict[str, str]]] = {}
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    for document in documents:
        raw_key = (str(document["domain"]), str(document["doc_id"]))
        doc_key = _opaque("d2d", *raw_key)
        candidates, aliases = _canonical_span_rows(document, tokenizer)
        if not candidates:
            raise ValueError("Doc2Dial v54 document has no canonical candidates")
        row = {
            "schema_version": "frc-doc2dial-v54-blind-document-v1",
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
        if v50._contains_forbidden_blind_key(row):
            raise AssertionError("Doc2Dial v54 gold leaked into blind document store")
        document_store.append(row)
        metadata[doc_key] = {
            "domain_commitment": _hash(str(document["domain"])),
            "document_commitment": _hash(str(document["doc_id"])),
            "length": len(str(document["doc_text"])),
            "length_quartile": _quartile(
                len(str(document["doc_text"])), length_boundaries
            ),
        }
        raw_by_key[raw_key] = document
        aliases_by_key[doc_key] = aliases
        pool_sizes.append(len(candidates))
        token_costs.extend(int(candidate["token_count"]) for candidate in candidates)
    document_store.sort(key=lambda row: str(row["id"]))

    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    turn_positions: list[int] = []
    fallback_count = 0
    for source_row in selected:
        raw_key = (str(source_row["domain"]), str(source_row["doc_id"]))
        if raw_key not in raw_by_key:
            raise ValueError("Doc2Dial v54 selected document is absent")
        own_key = _opaque("d2d", *raw_key)
        case_id = _opaque(
            "d2c",
            stage,
            str(source_row["dial_id"]),
            str(source_row["target_turn_id"]),
        )
        decoys, fallback_used = _select_probe_documents(
            own_key, case_id, metadata
        )
        fallback_count += int(fallback_used)
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "stage": stage,
            "id": case_id,
            "query": str(source_row["query"]),
            "own_doc_key": own_key,
            "probe_doc_keys": [own_key, *decoys],
            "decoy_fallback_used": fallback_used,
            "gold_fields_visible_to_scorer": False,
        }
        if v50._contains_forbidden_blind_key(blind):
            raise AssertionError("Doc2Dial v54 gold leaked into blind case cache")
        prepared.append(blind)
        maps.append(
            {
                "schema_version": "frc-doc2dial-v54-candidate-map-v1",
                "id": case_id,
                "domain_commitment": _hash(str(source_row["domain"])),
                "document_commitment": _hash(str(source_row["doc_id"])),
                "dialogue_commitment": _hash(str(source_row["dial_id"])),
                "target_turn_commitment": _hash(
                    str(source_row["target_turn_id"])
                ),
                "turn_position": int(source_row["turn_position"]),
                "span_aliases": aliases_by_key[own_key],
            }
        )
        turn_positions.append(int(source_row["turn_position"]))
    turn_boundaries = [
        round(float(value), 6)
        for value in np.quantile(turn_positions, [0.25, 0.5, 0.75])
    ]
    return prepared, maps, document_store, {
        "stage": stage,
        "prepared_cases": len(prepared),
        "blind_documents": len(document_store),
        "source_candidate_pool_size": _distribution(pool_sizes),
        "source_candidate_token_cost": _distribution(token_costs),
        "document_length_codepoints": _distribution(lengths),
        "document_length_quartile_boundaries": length_boundaries,
        "dialogue_turn_position": _distribution(turn_positions),
        "dialogue_turn_position_quartile_boundaries": turn_boundaries,
        "decoy_fallback_count": fallback_count,
        "decoy_fallback_rate": fallback_count / len(prepared) if prepared else 1.0,
        "gold_fields_exported_to_blind_cache": False,
        "ready_for_query_generation": len(prepared) == len(selected),
    }


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in prepared_rows:
        if v50._contains_forbidden_blind_key(row):
            raise ValueError("Doc2Dial v54 query builder received a forbidden field")
        query = _normalise(row["query"])
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "stage": str(row["stage"]),
                "id": str(row["id"]),
                "atomic_queries": {
                    role: QUERY_TEMPLATES[role].format(query=query)
                    for role in DYNAMIC_ROLES
                },
                "fallback_used": False,
                "fallback_reason": None,
                "gold_fields_visible_to_generator": False,
            }
        )
    return result


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if [str(row["id"]) for row in prepared_rows] != [
        str(row["id"]) for row in query_rows
    ]:
        raise ValueError("Doc2Dial v54 query cache is incomplete or out of order")
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("Doc2Dial v54 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


def _ranking(
    values: Sequence[float], candidates: Sequence[dict[str, Any]]
) -> list[str]:
    return [
        str(candidates[index]["id"])
        for index in sorted(
            range(len(candidates)),
            key=lambda index: (-float(values[index]), str(candidates[index]["id"])),
        )
    ]


def build_retrieval_pool(
    candidates: Sequence[dict[str, Any]],
    *,
    query: str,
    dense_scores: Sequence[float],
    top_k_per_method: int = RETRIEVAL_TOP_K_PER_METHOD,
    maximum: int = RETRIEVAL_POOL_MAXIMUM,
) -> list[dict[str, Any]]:
    values = [dict(candidate) for candidate in candidates]
    if len(values) != len(dense_scores) or not values:
        raise ValueError("Doc2Dial v54 retrieval inputs are incomplete")
    bm25_scores = _bm25(query, [str(candidate["text"]) for candidate in values])
    bm25_order = _ranking(bm25_scores, values)
    dense_order = _ranking(dense_scores, values)
    limit = min(top_k_per_method, len(values))
    selected_ids = set(bm25_order[:limit]) | set(dense_order[:limit])
    missing_rank = len(values) + 1
    bm25_rank = {identifier: index for index, identifier in enumerate(bm25_order)}
    dense_rank = {identifier: index for index, identifier in enumerate(dense_order)}
    by_id = {str(candidate["id"]): candidate for candidate in values}
    ordered_ids = sorted(
        selected_ids,
        key=lambda identifier: (
            min(
                bm25_rank.get(identifier, missing_rank),
                dense_rank.get(identifier, missing_rank),
            ),
            bm25_rank.get(identifier, missing_rank)
            + dense_rank.get(identifier, missing_rank),
            identifier,
        ),
    )
    if len(ordered_ids) > maximum:
        raise AssertionError("Doc2Dial v54 retrieval pool exceeded its maximum")
    return [by_id[identifier] for identifier in ordered_ids]


class FrozenDoc2DialScorer(v50.FrozenContractNliScorer):
    """Frozen BGE scorer with shared document store and contrastive probes."""

    def __init__(
        self,
        *args: Any,
        document_store: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._documents = {str(row["id"]): dict(row) for row in document_store}
        if len(self._documents) != len(document_store):
            raise ValueError("Doc2Dial v54 blind document ids are not unique")
        self._source_embedding_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _source_signature(candidates: Sequence[dict[str, Any]]) -> str:
        commitment = [
            {
                "id": str(candidate["id"]),
                "text_sha256": _hash(str(candidate["text"])),
            }
            for candidate in candidates
        ]
        payload = json.dumps(
            commitment,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _source_vectors(self, candidates: Sequence[dict[str, Any]]) -> np.ndarray:
        signature = self._source_signature(candidates)
        cached = self._source_embedding_cache.get(signature)
        if cached is None:
            cached = self._encode([str(row["text"]) for row in candidates])
            self._source_embedding_cache[signature] = cached
        return cached

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reduced: list[dict[str, Any]] = []
        retrieval_meta: dict[str, dict[str, int]] = {}
        gate_work: list[dict[str, Any]] = []
        gate_pairs: list[tuple[str, str]] = []
        for case in cases:
            if case.get("gold_fields_visible_to_scorer") is not False:
                raise ValueError("Doc2Dial v54 blind assertion failed")
            own_key = str(case["own_doc_key"])
            probe_keys = [str(value) for value in case["probe_doc_keys"]]
            if (
                len(probe_keys) != DECOY_COUNT + 1
                or probe_keys[0] != own_key
                or len(set(probe_keys)) != len(probe_keys)
            ):
                raise ValueError("Doc2Dial v54 probe document set changed")
            own = self._documents.get(own_key)
            if own is None:
                raise ValueError("Doc2Dial v54 own blind document is missing")
            own_candidates = [dict(value) for value in own["candidates"]]
            query = str(case["query"])
            query_vector = self._encode([query])[0]
            own_dense = self._source_vectors(own_candidates) @ query_vector
            own_pool = build_retrieval_pool(
                own_candidates, query=query, dense_scores=own_dense
            )
            clone = dict(case)
            clone["candidates"] = own_pool
            reduced.append(clone)
            retrieval_meta[str(case["id"])] = {
                "source_candidate_count": len(own_candidates),
                "retrieval_pool_count": len(own_pool),
            }
            documents: list[dict[str, Any]] = []
            for doc_key in probe_keys:
                document = self._documents.get(doc_key)
                if document is None:
                    raise ValueError("Doc2Dial v54 probe blind document is missing")
                candidates = [dict(value) for value in document["candidates"]]
                dense = self._source_vectors(candidates) @ query_vector
                pool = build_retrieval_pool(
                    candidates,
                    query=query,
                    dense_scores=dense,
                    top_k_per_method=PROBE_TOP_K_PER_METHOD,
                    maximum=PROBE_POOL_MAXIMUM,
                )
                start = len(gate_pairs)
                gate_pairs.extend((query, str(row["text"])) for row in pool)
                documents.append(
                    {"id": doc_key, "pair_start": start, "pair_count": len(pool)}
                )
            gate_work.append(
                {
                    "case_id": str(case["id"]),
                    "own_doc_key": own_key,
                    "documents": documents,
                    "decoy_fallback_used": bool(case["decoy_fallback_used"]),
                }
            )

        rows = super().score_cases(reduced)
        gate_scores = self._predict(gate_pairs).reshape(-1)
        gate_by_case: dict[str, dict[str, Any]] = {}
        for item in gate_work:
            document_scores: dict[str, float] = {}
            for document in item["documents"]:
                start = int(document["pair_start"])
                count = int(document["pair_count"])
                values = gate_scores[start : start + count]
                if count <= 0 or len(values) != count:
                    raise ValueError("Doc2Dial v54 probe score is incomplete")
                document_scores[str(document["id"])] = float(np.max(values))
            ranking = sorted(
                document_scores,
                key=lambda key: (-document_scores[key], key),
            )
            own_key = str(item["own_doc_key"])
            gate_by_case[str(item["case_id"])] = {
                "assigned_document_rank": ranking.index(own_key) + 1,
                "assigned_document_rank_one": ranking[0] == own_key,
                "probe_document_count": len(ranking),
                "randomization_p_value_if_passed": (
                    1.0 / len(ranking) if ranking[0] == own_key else None
                ),
                "document_ranking": ranking,
                "document_scores": document_scores,
                "decoy_fallback_used": bool(item["decoy_fallback_used"]),
            }
        for row in rows:
            case_id = str(row["id"])
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
            row["retrieval_pool"] = retrieval_meta[case_id]
            row["document_contrastive_gate"] = gate_by_case[case_id]
        return rows


def _source_case_index(
    document_source: dict[str, Any], dialogue_source: dict[str, Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows, _ = extract_cases(document_source, dialogue_source)
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            _hash(str(row["doc_id"])),
            _hash(str(row["dial_id"])),
            _hash(str(row["target_turn_id"])),
        )
        if key in result:
            raise ValueError("Doc2Dial v54 source commitment is not unique")
        result[key] = row
    return result


def _reference_count_group(count: int) -> str:
    if count == 0:
        return "none"
    if count == 1:
        return "one"
    return "multiple"


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
        for row in iter_documents(document_source)
    }
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    pool_sizes = [len(row.get("candidates", [])) for row in scored_rows]
    pool_boundaries = [
        round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])
    ]
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        key = (
            str(candidate_map["document_commitment"]),
            str(candidate_map["dialogue_commitment"]),
            str(candidate_map["target_turn_commitment"]),
        )
        source = source_cases.get(key)
        scored = scored_by_id.get(str(candidate_map["id"]))
        if source is None or scored is None:
            raise ValueError("Doc2Dial v54 gold join is incomplete")
        alias_map = {
            str(row["sp_commitment"]): str(row["candidate_id"])
            for row in candidate_map["span_aliases"]
        }
        gold_ids: set[str] = set()
        for sp_id in source["reference_ids"]:
            candidate_id = alias_map.get(_hash(str(sp_id)))
            if candidate_id is None:
                raise ValueError("Doc2Dial v54 gold reference lacks candidate map")
            gold_ids.add(candidate_id)
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
                "reference_count_group": _reference_count_group(len(gold_ids)),
                "domain": str(source["domain"]),
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _quartile(pool_size, pool_boundaries),
                "document_length_quartile": _quartile(
                    len(str(document["doc_text"])),
                    document_length_quartile_boundaries,
                ),
                "dialogue_turn_position_quartile": _quartile(
                    position, turn_position_quartile_boundaries
                ),
                "candidate_ceiling_complete": gold_ids <= available,
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    answer = [row for row in gold_rows if row["answer_state"] == "answer_bearing"]
    no_answer = [row for row in gold_rows if row["answer_state"] == "no_answer"]
    ceiling = (
        float(np.mean([row["candidate_ceiling_complete"] for row in answer]))
        if answer
        else 0.0
    )
    documents = len({str(row["document_cluster"]) for row in gold_rows})
    dialogues = len({str(row["dialogue_cluster"]) for row in gold_rows})
    checks = {
        "exact_target_cases_met": len(gold_rows) == TARGET_CASES,
        "exact_answer_state_balance_met": len(answer) == TARGET_PER_GROUP
        and len(no_answer) == TARGET_PER_GROUP,
        "minimum_dialogues_met": dialogues >= MINIMUM_DIALOGUES,
        "minimum_documents_met": documents >= MINIMUM_DOCUMENTS,
        "dialogue_cap_met": int(sampling["maximum_cases_per_dialogue"])
        <= MAX_CASES_PER_DIALOGUE,
        "document_state_cap_met": int(
            sampling["maximum_cases_per_document_per_answer_state"]
        )
        <= MAX_CASES_PER_DOCUMENT_PER_STATE,
        "schema_exclusion_rate_at_most_0_01": float(
            sampling["schema_exclusion_rate"]
        )
        <= 0.01,
        "candidate_ceiling_complete_rate_at_least_0_99": ceiling >= 0.99,
    }
    return {
        "schema_version": "frc-doc2dial-v54-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": sampling.get("stage"),
        "cases": len(gold_rows),
        "documents": documents,
        "dialogues": dialogues,
        "answer_bearing_cases": len(answer),
        "no_answer_cases": len(no_answer),
        "candidate_ceiling_complete_rate_on_answer_bearing_cases": round(
            ceiling, 6
        ),
        "candidate_pool_quartiles": dict(
            Counter(row["candidate_pool_quartile"] for row in gold_rows)
        ),
        "document_length_quartiles": dict(
            Counter(row["document_length_quartile"] for row in gold_rows)
        ),
        "reference_count_groups": dict(
            Counter(row["reference_count_group"] for row in gold_rows)
        ),
        "sampling": sampling,
        "checks": checks,
        "development_open_checks_passed": all(checks.values()),
    }


def _role_order(candidates: Sequence[dict[str, Any]], role: str) -> list[str]:
    values: list[tuple[float, str]] = []
    for candidate in candidates:
        raw = candidate.get("raw_dynamic_role_scores")
        if not isinstance(raw, dict) or role not in raw:
            raise ValueError("Doc2Dial v54 raw role score is missing")
        score = float(raw[role])
        if not math.isfinite(score):
            raise ValueError("Doc2Dial v54 raw role score must be finite")
        values.append((-score, str(candidate["id"])))
    return [identifier for _, identifier in sorted(values)]


def rank_concurrent_role_closure_details(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rankings = {role: _role_order(candidates, role) for role in DYNAMIC_ROLES}
    anchor = rankings["anchor"][:ANCHOR_FRONTIER]
    ordered = list(anchor[:1])
    role_winners: dict[str, str | None] = {}
    for public_role in ("condition", "solution", "exception"):
        internal = ROLE_ALIASES[public_role]
        winner = rankings[internal][0] if rankings[internal] else None
        role_winners[public_role] = winner
        if winner is not None and winner in anchor and winner not in ordered:
            ordered.append(winner)
    return {
        "anchor_frontier_ids": anchor,
        "role_winner_ids": role_winners,
        "closure_ids": ordered,
        "closure_size": len(ordered),
    }


def _role_closure_select(
    candidates: Sequence[dict[str, Any]], *, token_budget: int
) -> list[dict[str, Any]]:
    values = list(candidates)
    details = rank_concurrent_role_closure_details(values)
    by_id = {str(candidate["id"]): candidate for candidate in values}
    result: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    total = 0
    for identifier in details["closure_ids"]:
        candidate = by_id[str(identifier)]
        source_id = str(candidate["source_id"])
        if source_id in seen_sources:
            continue
        cost = int(candidate["token_count"])
        if total + cost > token_budget:
            if not result:
                result.append(candidate)
            continue
        result.append(candidate)
        seen_sources.add(source_id)
        total += cost
    return result


def select_v54(
    candidates: Sequence[dict[str, Any]],
    method: str,
    *,
    gate_passed: bool,
    token_budget: int,
) -> list[dict[str, Any]]:
    values = list(candidates)
    if method in V49_METHODS:
        return select_v49(values, method, token_budget=token_budget)
    if not gate_passed:
        return []
    if method == CANDIDATE:
        return _role_closure_select(values, token_budget=token_budget)
    if method == SAME_GATE_FRC_ABLATION:
        return select_v49(values, LOW_CORE_DIVERGENCE_V49, token_budget=token_budget)
    if method in GATED_NON_FRC_METHODS:
        base = method[len(GATED_PREFIX) : -len("_v54")]
        if base not in NON_FRC_BASELINES:
            raise ValueError("Doc2Dial v54 gated baseline mapping is invalid")
        return select_v49(values, base, token_budget=token_budget)
    raise ValueError(f"Unsupported Doc2Dial v54 method: {method}")


def _selection_metrics(
    selected: Sequence[dict[str, Any]],
    answer_state: str,
    gold_ids: Sequence[str],
) -> dict[str, Any]:
    selected_ids = {str(candidate["id"]) for candidate in selected}
    cost = sum(int(candidate["token_count"]) for candidate in selected)
    if answer_state == "no_answer":
        correct = not selected_ids
        return {
            "precision": None,
            "recall": None,
            "answer_f1": None,
            "utility_f1": 1.0 if correct else 0.0,
            "complete_recall": None,
            "abstained": not selected_ids,
            "gold_unit_hits": 0,
            "selected_unit_count": len(selected_ids),
            "selected_token_cost": cost,
        }
    gold = set(gold_ids)
    hits = len(selected_ids & gold)
    precision = hits / len(selected_ids) if selected_ids else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "answer_f1": f1,
        "utility_f1": f1,
        "complete_recall": bool(gold and gold <= selected_ids),
        "abstained": not selected_ids,
        "gold_unit_hits": hits,
        "selected_unit_count": len(selected_ids),
        "selected_token_cost": cost,
    }


def _aggregate(rows: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    metrics = [
        configuration["methods"][method]["metrics"]
        for row in rows
        for configuration in row["configurations"].values()
    ]
    answer = [metric for metric in metrics if metric["answer_f1"] is not None]
    no_answer = [metric for metric in metrics if metric["answer_f1"] is None]
    return {
        "answer_or_abstention_macro_f1": float(
            np.mean([metric["utility_f1"] for metric in metrics])
        ),
        "answer_bearing_macro_f1": float(
            np.mean([metric["answer_f1"] for metric in answer])
        ),
        "answer_macro_precision": float(
            np.mean([metric["precision"] for metric in answer])
        ),
        "answer_macro_recall": float(
            np.mean([metric["recall"] for metric in answer])
        ),
        "complete_reference_recall": float(
            np.mean([metric["complete_recall"] for metric in answer])
        ),
        "no_answer_abstention_accuracy": float(
            np.mean([metric["abstained"] for metric in no_answer])
        ),
        "abstention_rate": float(np.mean([metric["abstained"] for metric in metrics])),
        "mean_selected_unit_count": float(
            np.mean([metric["selected_unit_count"] for metric in metrics])
        ),
        "mean_selected_token_cost": float(
            np.mean([metric["selected_token_cost"] for metric in metrics])
        ),
    }


def _case_utility(row: dict[str, Any], method: str) -> float:
    return float(
        np.mean(
            [
                configuration["methods"][method]["metrics"]["utility_f1"]
                for configuration in row["configurations"].values()
            ]
        )
    )


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]], candidate: str, baseline: str
) -> dict[str, float | int]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["document_cluster"])].append(
            _case_utility(row, candidate) - _case_utility(row, baseline)
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


def _strongest(rows: Sequence[dict[str, Any]], methods: Sequence[str]) -> str:
    return sorted(
        methods,
        key=lambda method: (
            -float(_aggregate(rows, method)["answer_or_abstention_macro_f1"]),
            method,
        ),
    )[0]


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = _strongest(rows, GATED_NON_FRC_METHODS)
    candidate_value = float(np.mean([_case_utility(row, CANDIDATE) for row in rows]))
    baseline_value = float(np.mean([_case_utility(row, strongest) for row in rows]))
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
        raise ValueError("Doc2Dial v54 gold and score caches differ")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("Doc2Dial v54 gold and score ids differ")
        candidates = v50.merge_v50_scored_candidates(dict(scored))
        gate = dict(scored["document_contrastive_gate"])
        gate_passed = bool(gate["assigned_document_rank_one"])
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v54(
                    candidates,
                    method,
                    gate_passed=gate_passed,
                    token_budget=budget,
                )
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(
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

    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = _strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_frc = _strongest(evidence, UNGATED_FRC_CONTROLS)
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
    answer_gate_pass_rate = float(
        np.mean(
            [
                row["document_contrastive_gate"]["assigned_document_rank_one"]
                for row in evidence
                if row["answer_state"] == "answer_bearing"
            ]
        )
    )
    no_answer_gate_pass_rate = float(
        np.mean(
            [
                row["document_contrastive_gate"]["assigned_document_rank_one"]
                for row in evidence
                if row["answer_state"] == "no_answer"
            ]
        )
    )

    budget_deltas: dict[str, float] = {}
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
    strata: dict[str, dict[str, Any]] = {}
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
        "answer_bearing_gate_pass_rate_at_least_0_5": answer_gate_pass_rate >= 0.5,
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
            "DOC2DIAL_V54_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_VALIDATION"
            if supported
            else "DOC2DIAL_V54_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION"
        )
    else:
        status = (
            "DOC2DIAL_V54_DOCUMENT_CONTRASTIVE_ROLE_CLOSURE_SUPPORT_ESTABLISHED"
            if supported
            else "DOC2DIAL_V54_DOCUMENT_CONTRASTIVE_ROLE_CLOSURE_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "schema_version": "frc-doc2dial-document-contrastive-report-v54",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "stage": stage,
            "split": SOURCE_FILES[stage],
            "cases": len(evidence),
            "documents": documents,
            "dialogues": dialogues,
            "answer_state_counts": dict(state_counts),
            "budgets": list(BUDGETS),
            "official_shared_task_result": False,
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "gold_joined_after_complete_score_cache": True,
            "cuad_case_level_artifact_reused": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "gate_formula": "assigned document ranks first by maximum anchor reranker logit among assigned plus seven deterministic decoys",
            "selector_formula": "anchor Top-1 plus each atomic-role Top-1 only when also in anchor Top-3, under token budget",
            "aggregates": aggregates,
            "strongest_shared_gate_non_frc": strongest_non_frc,
            "strongest_ungated_frozen_frc": strongest_frc,
            "family_comparison": comparisons,
            "answer_macro_recall_drop_vs_ungated_v49": recall_drop,
            "candidate_ceiling_complete_rate": round(ceiling, 6),
            "gate_diagnostics": {
                "answer_bearing_pass_rate": round(answer_gate_pass_rate, 6),
                "no_answer_pass_rate": round(no_answer_gate_pass_rate, 6),
                "no_answer_rejection_rate": round(1.0 - no_answer_gate_pass_rate, 6),
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
                "validation_open_authorized": stage == "development" and supported,
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
        f"# Doc2Dial document-contrastive role closure ({rounded['metadata']['stage']}, v54)",
        "",
        f"- Status: `{outcome['status']}`",
        (
            f"- Cases/documents/dialogues: {rounded['metadata']['cases']}/"
            f"{rounded['metadata']['documents']}/{rounded['metadata']['dialogues']}"
        ),
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        f"- Candidate answer F1: {candidate['answer_bearing_macro_f1']:.6f}",
        f"- Candidate answer recall: {candidate['answer_macro_recall']:.6f}",
        f"- No-answer abstention accuracy: {candidate['no_answer_abstention_accuracy']:.6f}",
        f"- Answer-bearing gate pass rate: {analysis['gate_diagnostics']['answer_bearing_pass_rate']:.6f}",
        f"- No-answer gate pass rate: {analysis['gate_diagnostics']['no_answer_pass_rate']:.6f}",
        f"- Abstention rate: {candidate['abstention_rate']:.6f}",
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
            "This is a balanced Doc2Dial v1.0.1 document-grounded span-selection mechanism experiment. It is not an official shared-task submission, response-generation result, SetR reproduction, selector adoption, or flood-domain expert validation.",
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
        raise ValueError("Doc2Dial v54 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = value["source_access_boundary_at_registration"]
    if boundary["dialogue_json_content_opened"] or boundary[
        "document_json_content_opened"
    ]:
        raise ValueError("Doc2Dial v54 source content was opened before registration")
    if value["development_and_confirmation_scope"]["development_target_cases"] != TARGET_CASES:
        raise ValueError("Doc2Dial v54 development size changed")
    if value["evaluation"]["token_budgets"] != list(BUDGETS):
        raise ValueError("Doc2Dial v54 budgets changed")
    if value["deterministic_decoy_documents"]["count"] != DECOY_COUNT:
        raise ValueError("Doc2Dial v54 decoy count changed")
    return value


def validate_source_registration(
    registration_path: Path, *, protocol_path: Path, source_root: Path
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value["protocol_sha256_before_download"] != _sha256(protocol_path):
        raise ValueError("Doc2Dial v54 source registration protocol hash mismatch")
    registered = {str(row["path"]): str(row["sha256"]) for row in value["files"]}
    for name, expected in SOURCE_SHA256.items():
        path = source_root / name
        if registered.get(name) != expected or _sha256(path) != expected:
            raise ValueError(f"Doc2Dial v54 source changed: {name}")
    if value["dialogue_or_document_json_content_read_before_registration"] is not False:
        raise ValueError("Doc2Dial v54 source was opened before registration")
    if value["validation_content_read"] is not False:
        raise ValueError("Doc2Dial v54 validation was opened before registration")
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
        raise ValueError("Doc2Dial v54 implementation hash mismatch")
    if value.get("dialogue_or_document_json_opened_before_registration") is not False:
        raise ValueError("Doc2Dial v54 implementation was registered too late")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("Doc2Dial v54 synthetic invariants are incomplete")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BUDGETS",
    "CANDIDATE",
    "CAPABILITY",
    "DATASET_ID",
    "DECOY_COUNT",
    "EXPERIMENT_ID",
    "FrozenDoc2DialScorer",
    "GATED_NON_FRC_METHODS",
    "METHODS",
    "PROTOCOL_SHA256",
    "SAME_GATE_FRC_ABLATION",
    "SCHEMA_VERSION",
    "SOURCE_FILES",
    "SOURCE_SHA256",
    "STAGES",
    "TARGET_CASES",
    "TARGET_PER_GROUP",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "build_retrieval_pool",
    "evaluate_stage",
    "extract_cases",
    "iter_dialogues",
    "iter_documents",
    "prepare_blind_cases",
    "rank_concurrent_role_closure_details",
    "read_stage_sources",
    "select_balanced_sample",
    "select_v54",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_registration",
    "write_report",
]
