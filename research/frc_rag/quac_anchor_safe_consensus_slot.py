"""Prospective QuAC v57 anchor-safe single-slot consensus experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import research.frc_rag.contractnli_native_zero_consensus_abstention as v50
import research.frc_rag.doc2dial_document_contrastive_role_closure as v54
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    _round_for_display,
    select_v49,
)
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_ROLES
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import NON_FRC_BASELINES


SCHEMA_VERSION = "frc-quac-anchor-safe-consensus-slot-v57"
EXPERIMENT_ID = "FRC-QUAC-ANCHOR-SAFE-CONSENSUS-SLOT-V57"
DATASET_ID = "quac_v0_2_anchor_safe_consensus_slot_v57"
CAPABILITY = "conversational_evidence_selection_with_document_abstention"
PROTOCOL_SHA256 = "bd5dab2f9166cfc2ac9e256ee772ec13d87a29ed00093adcec91f491ab44f810"

STAGES = ("development", "confirmation")
SOURCE_FILES = {
    "development": "train_v0.2.json",
    "confirmation": "val_v0.2.json",
}
SOURCE_URLS = {
    "development": "https://s3.amazonaws.com/my89public/quac/train_v0.2.json",
    "confirmation": "https://s3.amazonaws.com/my89public/quac/val_v0.2.json",
}
SAMPLE_SALTS = {
    "development": "FRC-QUAC-V57-DEVELOPMENT|",
    "confirmation": "FRC-QUAC-V57-CONFIRMATION|",
}
DECOY_SALT = "FRC-QUAC-V57-DECOY|"
TARGET_CASES = 600
TARGET_PER_GROUP = 300
MAX_CASES_PER_DIALOGUE = 2
MAX_CASES_PER_DOCUMENT_PER_STATE = 4
MINIMUM_DIALOGUES = 250
MINIMUM_DOCUMENTS = 200
DECOY_COUNT = 7
MINIMUM_STRATUM_CASES = 50
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260817
BUDGETS = (128, 256, 512)
MAX_SENTENCE_TOKENS = 96
WINDOW_TOKENS = 80

GATED_PREFIX = "document_contrastive_"
GATED_SUFFIX = "_v57"
GATED_NON_FRC_METHODS = tuple(
    f"{GATED_PREFIX}{method}{GATED_SUFFIX}" for method in NON_FRC_BASELINES
)
EXACT_ANCHOR_ABLATION = "document_contrastive_cross_encoder_topk_v57"
SAME_GATE_ROLE_CLOSURE = (
    "document_contrastive_rank_concurrent_atomic_role_closure_frc_v54_transferred_v57"
)
SAME_GATE_LOW_CORE = (
    "document_contrastive_low_core_divergence_frc_v49_transferred_v57"
)
SAME_GATE_FRC_CONTROLS = (SAME_GATE_ROLE_CLOSURE, SAME_GATE_LOW_CORE)
CANDIDATE = "anchor_safe_single_slot_consensus_frc_v57"
METHODS = (
    *V49_METHODS,
    *GATED_NON_FRC_METHODS,
    *SAME_GATE_FRC_CONTROLS,
    CANDIDATE,
)

QUERY_TEMPLATES = {
    "anchor": "{query}",
    "first_fact": (
        "Find the passage that directly answers the current question in this "
        "conversation: {query}"
    ),
    "second_fact_or_bridge": (
        "Find antecedents, people, events, or earlier context needed to resolve "
        "the current question: {query}"
    ),
    "counterevidence": (
        "Find negations, contrasts, qualifications, dates, or limitations relevant "
        "to the current question: {query}"
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


def read_stage_source(source_root: Path, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported QuAC v57 stage: {stage}")
    path = source_root / SOURCE_FILES[stage]
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("QuAC v57 source root must be an object")
    return value


def iter_dialogues(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    data = source.get("data")
    if not isinstance(data, list):
        raise ValueError("QuAC v57 data must be a list")
    seen_dialogues: set[str] = set()
    for section_index, section in enumerate(data):
        if not isinstance(section, dict):
            raise ValueError("QuAC v57 section must be an object")
        title = _normalise(section.get("title"))
        section_title = _normalise(section.get("section_title"))
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise ValueError("QuAC v57 paragraphs must be a list")
        for dialogue_index, dialogue in enumerate(paragraphs):
            if not isinstance(dialogue, dict):
                raise ValueError("QuAC v57 dialogue must be an object")
            context = str(dialogue.get("context", ""))
            dialogue_id = _normalise(dialogue.get("id"))
            turns = dialogue.get("qas")
            if not dialogue_id:
                dialogue_id = _hash(
                    str(section_index), str(dialogue_index), title, section_title, context
                )
            if dialogue_id in seen_dialogues:
                raise ValueError("QuAC v57 dialogue id is not unique")
            seen_dialogues.add(dialogue_id)
            yield {
                "title": title,
                "section_title": section_title,
                "context": context,
                "dialogue_id": dialogue_id,
                "document_id": _hash(title, section_title, context),
                "turns": turns,
            }


def _conversation_query(turns: Sequence[dict[str, Any]], position: int) -> tuple[str, int]:
    question = _normalise(turns[position].get("question"))
    history: list[str] = []
    for prior in turns[max(0, position - 2) : position]:
        if not isinstance(prior, dict):
            continue
        prior_question = _normalise(prior.get("question"))
        answer = prior.get("orig_answer")
        prior_answer = (
            _normalise(answer.get("text")) if isinstance(answer, dict) else ""
        )
        if prior_question and prior_answer:
            history.append(f"Question: {prior_question}\nAnswer: {prior_answer}")
    parts = [f"Current question: {question}"]
    if history:
        parts.append("Conversation history:\n" + "\n".join(history))
    return "\n".join(parts), len(history)


def extract_cases(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    candidate_turns = 0
    seen_cases: set[tuple[str, str]] = set()
    for dialogue in iter_dialogues(source):
        context = str(dialogue["context"])
        turns = dialogue["turns"]
        if not context or not isinstance(turns, list):
            count = len(turns) if isinstance(turns, list) else 1
            candidate_turns += count
            exclusions["missing_context_or_turn_list"] += count
            continue
        for position, turn in enumerate(turns):
            candidate_turns += 1
            if not isinstance(turn, dict):
                exclusions["turn_not_object"] += 1
                continue
            question = _normalise(turn.get("question"))
            turn_id = _normalise(turn.get("id"))
            answer = turn.get("orig_answer")
            if not question:
                exclusions["empty_question"] += 1
                continue
            if not turn_id:
                exclusions["empty_turn_id"] += 1
                continue
            if not isinstance(answer, dict):
                exclusions["orig_answer_not_object"] += 1
                continue
            answer_text = str(answer.get("text", ""))
            canonical_answer = _normalise(answer_text)
            if not canonical_answer:
                exclusions["empty_orig_answer"] += 1
                continue
            answer_state = (
                "no_answer"
                if canonical_answer.upper() == "CANNOTANSWER"
                else "answer_bearing"
            )
            start: int | None = None
            end: int | None = None
            if answer_state == "answer_bearing":
                try:
                    start = int(answer.get("answer_start"))
                except (TypeError, ValueError):
                    exclusions["invalid_answer_start"] += 1
                    continue
                end = start + len(answer_text)
                if start < 0 or end > len(context) or context[start:end] != answer_text:
                    exclusions["answer_interval_mismatch"] += 1
                    continue
            case_key = (str(dialogue["dialogue_id"]), turn_id)
            if case_key in seen_cases:
                raise ValueError("QuAC v57 dialogue-turn case is not unique")
            seen_cases.add(case_key)
            query, history_depth = _conversation_query(turns, position)
            rows.append(
                {
                    "dialogue_id": str(dialogue["dialogue_id"]),
                    "document_id": str(dialogue["document_id"]),
                    "turn_id": turn_id,
                    "turn_position": position,
                    "history_depth": history_depth,
                    "query": query,
                    "context": context,
                    "answer_state": answer_state,
                    "answer_text": answer_text,
                    "answer_start": start,
                    "answer_end": end,
                }
            )
    excluded = sum(exclusions.values())
    return rows, {
        "candidate_turns": candidate_turns,
        "eligible_cases": len(rows),
        "schema_excluded_cases": excluded,
        "schema_exclusion_rate": excluded / candidate_turns if candidate_turns else 1.0,
        "schema_exclusion_reasons": dict(sorted(exclusions.items())),
    }


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


def select_balanced_sample(
    source: dict[str, Any],
    *,
    stage: str,
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_dialogue: int = MAX_CASES_PER_DIALOGUE,
    maximum_cases_per_document_per_state: int = MAX_CASES_PER_DOCUMENT_PER_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported QuAC v57 stage: {stage}")
    rows, schema = extract_cases(source)
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in rows if row["answer_state"] == group],
            key=lambda row: _sample_key(row, stage),
        )
        for group in groups
    }
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("QuAC v57 source lacks the registered answer-state quota")
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
                if document_state_counts[document_state] >= maximum_cases_per_document_per_state:
                    continue
                selected.append(row)
                selected_by_group[group] += 1
                dialogue_counts[dialogue_id] += 1
                document_state_counts[document_state] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError("QuAC v57 caps prevent the registered balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    return selected, {
        "stage": stage,
        "target_cases": target_per_group * 2,
        "selected_cases": len(selected),
        "eligible_answer_state_counts": dict(Counter(row["answer_state"] for row in rows)),
        "selected_answer_state_counts": dict(selected_by_group),
        "selected_dialogues": len(dialogue_counts),
        "selected_documents": len({str(row["document_id"]) for row in selected}),
        "maximum_cases_per_dialogue": max(dialogue_counts.values(), default=0),
        "maximum_cases_per_document_per_answer_state": max(document_state_counts.values(), default=0),
        **schema,
    }


def _token_count(tokenizer: Any, text: str) -> int:
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def _trim_interval(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_intervals(text: str) -> list[tuple[int, int]]:
    boundaries = [0]
    for match in re.finditer(r"(?<=[.!?])[\"')\]]*\s+|\n+", text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    result: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        start, end = _trim_interval(text, left, right)
        if start < end:
            result.append((start, end))
    return result


def _window_interval(
    text: str, start: int, end: int, tokenizer: Any
) -> list[tuple[int, int]]:
    if _token_count(tokenizer, text[start:end]) <= MAX_SENTENCE_TOKENS:
        return [(start, end)]
    words = list(re.finditer(r"\S+", text[start:end]))
    if not words:
        return []
    result: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(words):
        window_end = cursor + 1
        while window_end <= len(words):
            candidate_start = start + words[cursor].start()
            candidate_end = start + words[window_end - 1].end()
            if _token_count(tokenizer, text[candidate_start:candidate_end]) > WINDOW_TOKENS:
                if window_end == cursor + 1:
                    result.append((candidate_start, candidate_end))
                    cursor = window_end
                else:
                    previous_end = start + words[window_end - 2].end()
                    result.append((candidate_start, previous_end))
                    cursor = window_end - 1
                break
            window_end += 1
        else:
            result.append((start + words[cursor].start(), start + words[-1].end()))
            cursor = len(words)
    return result


def canonical_candidate_rows(
    context: str, document_id: str, tokenizer: Any
) -> list[dict[str, Any]]:
    effective_end = len(context)
    sentinel = "CANNOTANSWER"
    stripped = context.rstrip()
    if stripped.endswith(sentinel):
        sentinel_start = len(stripped) - len(sentinel)
        if sentinel_start == 0 or stripped[sentinel_start - 1].isspace():
            effective_end = sentinel_start
    text = context[:effective_end]
    rows: list[dict[str, Any]] = []
    for sentence_start, sentence_end in _sentence_intervals(text):
        for start, end in _window_interval(text, sentence_start, sentence_end, tokenizer):
            candidate_text = text[start:end]
            if not candidate_text:
                continue
            candidate_id = _opaque("q57u", document_id, str(start), str(end))
            rows.append(
                {
                    "id": candidate_id,
                    "source_id": candidate_id,
                    "text": candidate_text,
                    "token_count": _token_count(tokenizer, candidate_text),
                    "start": start,
                    "end": end,
                }
            )
    return rows


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


def _document_catalog(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dialogue in iter_dialogues(source):
        document_id = str(dialogue["document_id"])
        row = {"document_id": document_id, "context": str(dialogue["context"])}
        existing = result.get(document_id)
        if existing is not None and existing != row:
            raise ValueError("QuAC v57 document commitment collision")
        result[document_id] = row
    return result


def _select_probe_documents(
    own_key: str,
    case_id: str,
    metadata: dict[str, dict[str, Any]],
) -> tuple[list[str], bool]:
    own = metadata[own_key]
    same = [
        key
        for key, row in metadata.items()
        if key != own_key and row["length_quartile"] == own["length_quartile"]
    ]
    other = [
        key
        for key, row in metadata.items()
        if key != own_key and row["length_quartile"] != own["length_quartile"]
    ]
    ordered = lambda values: sorted(  # noqa: E731
        values, key=lambda key: (_hash(DECOY_SALT, case_id, key), key)
    )
    selected = ordered(same)[:DECOY_COUNT]
    fallback = len(selected) < DECOY_COUNT
    if fallback:
        selected.extend(ordered(other)[: DECOY_COUNT - len(selected)])
    if len(selected) != DECOY_COUNT or len(set(selected)) != DECOY_COUNT:
        raise ValueError("QuAC v57 cannot construct seven unique decoys")
    return selected, fallback


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]],
    source: dict[str, Any],
    tokenizer: Any,
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    catalog = _document_catalog(source)
    lengths = [len(row["context"]) for row in catalog.values()]
    length_boundaries = [
        round(float(value), 6) for value in np.quantile(lengths, [0.25, 0.5, 0.75])
    ]
    raw_to_opaque = {
        key: _opaque("q57d", key) for key in sorted(catalog)
    }
    metadata = {
        raw_to_opaque[key]: {
            "document_commitment": _hash(key),
            "length": len(row["context"]),
            "length_quartile": _quartile(len(row["context"]), length_boundaries),
        }
        for key, row in catalog.items()
    }
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    required_docs: set[str] = set()
    fallback_count = 0
    positions: list[int] = []
    for source_row in selected:
        raw_doc = str(source_row["document_id"])
        own_key = raw_to_opaque[raw_doc]
        case_id = _opaque(
            "q57c", stage, str(source_row["dialogue_id"]), str(source_row["turn_id"])
        )
        decoys, fallback = _select_probe_documents(own_key, case_id, metadata)
        fallback_count += int(fallback)
        required_docs.update([own_key, *decoys])
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "stage": stage,
            "id": case_id,
            "query": str(source_row["query"]),
            "own_doc_key": own_key,
            "probe_doc_keys": [own_key, *decoys],
            "decoy_fallback_used": fallback,
            "gold_fields_visible_to_scorer": False,
        }
        if v50._contains_forbidden_blind_key(blind):
            raise AssertionError("QuAC v57 gold leaked into blind case cache")
        prepared.append(blind)
        maps.append(
            {
                "schema_version": "frc-quac-v57-candidate-map-v1",
                "id": case_id,
                "source_case_commitment": _hash(
                    str(source_row["dialogue_id"]), str(source_row["turn_id"])
                ),
                "document_commitment": _hash(raw_doc),
                "dialogue_commitment": _hash(str(source_row["dialogue_id"])),
                "turn_position": int(source_row["turn_position"]),
                "history_depth": int(source_row["history_depth"]),
            }
        )
        positions.append(int(source_row["turn_position"]))
    documents: list[dict[str, Any]] = []
    interval_maps: dict[str, list[dict[str, Any]]] = {}
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    opaque_to_raw = {value: key for key, value in raw_to_opaque.items()}
    for doc_key in sorted(required_docs):
        raw_doc = opaque_to_raw[doc_key]
        candidates = canonical_candidate_rows(catalog[raw_doc]["context"], raw_doc, tokenizer)
        if not candidates:
            raise ValueError("QuAC v57 document has no candidate units")
        document = {
            "schema_version": "frc-quac-v57-blind-document-v1",
            "id": doc_key,
            "candidates": [
                {
                    "id": str(row["id"]),
                    "source_id": str(row["source_id"]),
                    "text": str(row["text"]),
                    "token_count": int(row["token_count"]),
                }
                for row in candidates
            ],
            "gold_fields_visible_to_scorer": False,
        }
        if v50._contains_forbidden_blind_key(document):
            raise AssertionError("QuAC v57 gold leaked into blind document store")
        documents.append(document)
        interval_maps[doc_key] = [
            {"candidate_id": str(row["id"]), "start": int(row["start"]), "end": int(row["end"])}
            for row in candidates
        ]
        pool_sizes.append(len(candidates))
        token_costs.extend(int(row["token_count"]) for row in candidates)
    for row, blind in zip(maps, prepared, strict=True):
        row["candidate_intervals"] = interval_maps[str(blind["own_doc_key"])]
    turn_boundaries = [
        round(float(value), 6) for value in np.quantile(positions, [0.25, 0.5, 0.75])
    ]
    return prepared, maps, documents, {
        "stage": stage,
        "prepared_cases": len(prepared),
        "source_documents": len(catalog),
        "blind_documents": len(documents),
        "source_candidate_pool_size": _distribution(pool_sizes),
        "source_candidate_token_cost": _distribution(token_costs),
        "document_length_codepoints": _distribution(lengths),
        "document_length_quartile_boundaries": length_boundaries,
        "dialogue_turn_position": _distribution(positions),
        "dialogue_turn_position_quartile_boundaries": turn_boundaries,
        "decoy_fallback_count": fallback_count,
        "decoy_fallback_rate": fallback_count / len(prepared) if prepared else 1.0,
        "gold_fields_exported_to_blind_cache": False,
        "ready_for_query_generation": len(prepared) == len(selected),
    }


def build_deterministic_queries(prepared_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in prepared_rows:
        if v50._contains_forbidden_blind_key(row):
            raise ValueError("QuAC v57 query builder received a forbidden field")
        query = str(row["query"])
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
    if [str(row["id"]) for row in prepared_rows] != [str(row["id"]) for row in query_rows]:
        raise ValueError("QuAC v57 query cache is incomplete or out of order")
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("QuAC v57 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenQuacScorer(v54.FrozenDoc2DialScorer):
    """Reuse the frozen BGE scoring stack with v57 schema labels."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def _source_case_index(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows, _ = extract_cases(source)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _hash(str(row["dialogue_id"]), str(row["turn_id"]))
        if key in result:
            raise ValueError("QuAC v57 source commitment is not unique")
        result[key] = row
    return result


def _history_group(depth: int) -> str:
    return {0: "none", 1: "one", 2: "two"}.get(depth, "other")


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    document_length_quartile_boundaries: Sequence[float],
    turn_position_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    source_cases = _source_case_index(source)
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    pool_sizes = [len(row.get("candidates", [])) for row in scored_rows]
    pool_boundaries = [
        round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])
    ]
    answer_lengths = [
        len(str(row["answer_text"]))
        for row in source_cases.values()
        if row["answer_state"] == "answer_bearing"
    ]
    answer_boundaries = [
        round(float(value), 6)
        for value in np.quantile(answer_lengths, [0.25, 0.5, 0.75])
    ]
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        source_row = source_cases.get(str(candidate_map["source_case_commitment"]))
        scored = scored_by_id.get(str(candidate_map["id"]))
        if source_row is None or scored is None:
            raise ValueError("QuAC v57 gold join is incomplete")
        gold_ids: set[str] = set()
        if source_row["answer_state"] == "answer_bearing":
            start = int(source_row["answer_start"])
            end = int(source_row["answer_end"])
            for interval in candidate_map["candidate_intervals"]:
                if int(interval["start"]) < end and int(interval["end"]) > start:
                    gold_ids.add(str(interval["candidate_id"]))
        available = {str(row["id"]) for row in scored.get("candidates", [])}
        pool_size = len(available)
        answer_length = len(str(source_row["answer_text"])) if gold_ids else 0
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "document_cluster": str(candidate_map["document_commitment"]),
                "dialogue_cluster": str(candidate_map["dialogue_commitment"]),
                "answer_state": str(source_row["answer_state"]),
                "gold_candidate_ids": sorted(gold_ids),
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _quartile(pool_size, pool_boundaries),
                "document_length_quartile": _quartile(
                    len(str(source_row["context"])), document_length_quartile_boundaries
                ),
                "dialogue_turn_position_quartile": _quartile(
                    int(candidate_map["turn_position"]), turn_position_quartile_boundaries
                ),
                "history_depth_group": _history_group(int(candidate_map["history_depth"])),
                "answer_length_quartile": (
                    _quartile(answer_length, answer_boundaries) if gold_ids else "not_applicable"
                ),
                "candidate_ceiling_complete": (
                    bool(gold_ids and gold_ids <= available)
                    if source_row["answer_state"] == "answer_bearing"
                    else True
                ),
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    answer = [row for row in gold_rows if row["answer_state"] == "answer_bearing"]
    ceiling = float(np.mean([row["candidate_ceiling_complete"] for row in answer])) if answer else 0.0
    return {
        "schema_version": "frc-quac-v57-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": sampling.get("stage"),
        "cases": len(gold_rows),
        "documents": len({row["document_cluster"] for row in gold_rows}),
        "dialogues": len({row["dialogue_cluster"] for row in gold_rows}),
        "answer_bearing_cases": len(answer),
        "no_answer_cases": len(gold_rows) - len(answer),
        "candidate_ceiling_complete_rate_on_answer_bearing_cases": round(ceiling, 6),
        "sampling": sampling,
    }


def _role_order(candidates: Sequence[dict[str, Any]], role: str) -> list[str]:
    values: list[tuple[float, str]] = []
    for candidate in candidates:
        raw = candidate.get("raw_dynamic_role_scores")
        if not isinstance(raw, dict) or role not in raw:
            raise ValueError("QuAC v57 raw role score is missing")
        score = float(raw[role])
        if not math.isfinite(score):
            raise ValueError("QuAC v57 raw role score must be finite")
        values.append((-score, str(candidate["id"])))
    return [identifier for _, identifier in sorted(values)]


def single_slot_consensus_details(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rankings = {role: _role_order(candidates, role) for role in DYNAMIC_ROLES}
    anchor = rankings["anchor"]
    role_ranks = {
        role: {identifier: index + 1 for index, identifier in enumerate(rankings[role])}
        for role in DYNAMIC_ROLES
        if role != "anchor"
    }
    contenders: list[dict[str, Any]] = []
    for anchor_index, identifier in enumerate(anchor[5:12], start=6):
        supporting = sorted(
            role for role, ranks in role_ranks.items() if ranks.get(identifier, math.inf) <= 3
        )
        if len(supporting) < 2:
            continue
        rank_sum = sum(role_ranks[role][identifier] for role in supporting)
        contenders.append(
            {
                "id": identifier,
                "anchor_rank": anchor_index,
                "supporting_roles": supporting,
                "supporting_role_count": len(supporting),
                "supporting_rank_sum": rank_sum,
            }
        )
    contenders.sort(
        key=lambda row: (
            -int(row["supporting_role_count"]),
            int(row["supporting_rank_sum"]),
            int(row["anchor_rank"]),
            str(row["id"]),
        )
    )
    return {
        "anchor_rank_1_to_12": anchor[:12],
        "eligible_contenders": contenders,
        "winner_id": str(contenders[0]["id"]) if contenders else None,
    }


def _anchor_safe_select(
    candidates: Sequence[dict[str, Any]], *, token_budget: int
) -> list[dict[str, Any]]:
    values = list(candidates)
    baseline = select_v49(values, "cross_encoder_topk", token_budget=token_budget)
    if len(baseline) != 5:
        return baseline
    details = single_slot_consensus_details(values)
    winner_id = details["winner_id"]
    if winner_id is None or winner_id in {str(row["id"]) for row in baseline}:
        return baseline
    by_id = {str(row["id"]): row for row in values}
    winner = by_id[winner_id]
    preserved = baseline[:4]
    if sum(int(row["token_count"]) for row in preserved) + int(winner["token_count"]) > token_budget:
        return baseline
    return [*preserved, winner]


def select_v57(
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
        return _anchor_safe_select(values, token_budget=token_budget)
    if method == SAME_GATE_ROLE_CLOSURE:
        return v54._role_closure_select(values, token_budget=token_budget)
    if method == SAME_GATE_LOW_CORE:
        return select_v49(values, LOW_CORE_DIVERGENCE_V49, token_budget=token_budget)
    if method in GATED_NON_FRC_METHODS:
        base = method[len(GATED_PREFIX) : -len(GATED_SUFFIX)]
        if base not in NON_FRC_BASELINES:
            raise ValueError("QuAC v57 gated baseline mapping is invalid")
        return select_v49(values, base, token_budget=token_budget)
    raise ValueError(f"Unsupported QuAC v57 method: {method}")


def _selection_metrics(
    selected: Sequence[dict[str, Any]], answer_state: str, gold_ids: Sequence[str]
) -> dict[str, Any]:
    return v54._selection_metrics(selected, answer_state, gold_ids)


def _aggregate(rows: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    return v54._aggregate(rows, method)


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
        samples[index] = float(np.mean(np.concatenate([arrays[item] for item in picked])))
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
        key=lambda method: (-_aggregate(rows, method)["answer_or_abstention_macro_f1"], method),
    )[0]


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = _strongest(rows, GATED_NON_FRC_METHODS)
    delta = float(np.mean([_case_utility(row, CANDIDATE) for row in rows])) - float(
        np.mean([_case_utility(row, strongest) for row in rows])
    )
    return strongest, round(delta, 6)


def evaluate_stage(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stage not in STAGES or len(gold_rows) != len(scored_rows) or not gold_rows:
        raise ValueError("QuAC v57 gold and score caches differ")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("QuAC v57 gold and score ids differ")
        candidates = v50.merge_v50_scored_candidates(dict(scored))
        gate = dict(scored["document_contrastive_gate"])
        gate_passed = bool(gate["assigned_document_rank_one"])
        configurations: dict[str, Any] = {}
        case_triggered = False
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v57(
                    candidates, method, gate_passed=gate_passed, token_budget=budget
                )
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(
                        selected, str(gold["answer_state"]), gold["gold_candidate_ids"]
                    ),
                }
            triggered = methods[CANDIDATE]["selected_ids"] != methods[EXACT_ANCHOR_ABLATION]["selected_ids"]
            case_triggered = case_triggered or triggered
            configurations[str(budget)] = {
                "single_slot_triggered": triggered,
                "methods": methods,
            }
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "document_cluster": str(gold["document_cluster"]),
                "dialogue_cluster": str(gold["dialogue_cluster"]),
                "answer_state": str(gold["answer_state"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "document_length_quartile": str(gold["document_length_quartile"]),
                "dialogue_turn_position_quartile": str(gold["dialogue_turn_position_quartile"]),
                "history_depth_group": str(gold["history_depth_group"]),
                "answer_length_quartile": str(gold["answer_length_quartile"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "document_contrastive_gate": gate,
                "single_slot": {
                    **single_slot_consensus_details(candidates),
                    "triggered_in_any_budget": case_triggered,
                },
                "configurations": configurations,
            }
        )
    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = _strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_same_gate_frc = _strongest(evidence, SAME_GATE_FRC_CONTROLS)
    comparisons = {
        "candidate_minus_exact_anchor_ablation": _cluster_bootstrap(
            evidence, CANDIDATE, EXACT_ANCHOR_ABLATION
        ),
        "candidate_minus_strongest_shared_gate_non_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_non_frc
        ),
        "candidate_minus_strongest_same_gate_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_same_gate_frc
        ),
    }
    candidate = aggregates[CANDIDATE]
    exact_anchor = aggregates[EXACT_ANCHOR_ABLATION]
    recall_drop = round(exact_anchor["answer_macro_recall"] - candidate["answer_macro_recall"], 6)
    trigger_rate = float(np.mean([row["single_slot"]["triggered_in_any_budget"] for row in evidence]))
    answer_gate_pass_rate = float(
        np.mean(
            [
                row["document_contrastive_gate"]["assigned_document_rank_one"]
                for row in evidence
                if row["answer_state"] == "answer_bearing"
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
        _, budget_deltas[str(budget)] = _stratum_delta(subset)
    dimensions = {
        "answer_state": lambda row: row["answer_state"],
        "dialogue_turn_position_quartile": lambda row: row["dialogue_turn_position_quartile"],
        "document_length_quartile": lambda row: row["document_length_quartile"],
        "candidate_pool_quartile": lambda row: row["candidate_pool_quartile"],
        "answer_length_quartile": lambda row: row["answer_length_quartile"],
        "history_depth_group": lambda row: row["history_depth_group"],
        "gate_outcome": lambda row: "passed" if row["document_contrastive_gate"]["assigned_document_rank_one"] else "abstained",
        "single_slot_trigger_outcome": lambda row: "triggered" if row["single_slot"]["triggered_in_any_budget"] else "not_triggered",
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
    ceiling = float(np.mean([row["candidate_ceiling_complete"] for row in answer_rows]))
    state_counts = Counter(row["answer_state"] for row in evidence)
    documents = len({row["document_cluster"] for row in evidence})
    dialogues = len({row["dialogue_cluster"] for row in evidence})
    sampling = source_artifacts["sampling"]
    structural = source_artifacts["structural_census"]
    checks = {
        "exact_cases_equals_600": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": state_counts == {"answer_bearing": TARGET_PER_GROUP, "no_answer": TARGET_PER_GROUP},
        "minimum_dialogues_at_least_250": dialogues >= MINIMUM_DIALOGUES,
        "minimum_documents_at_least_200": documents >= MINIMUM_DOCUMENTS,
        "dialogue_cap_at_most_2": int(sampling["maximum_cases_per_dialogue"]) <= MAX_CASES_PER_DIALOGUE,
        "document_state_cap_at_most_4": int(sampling["maximum_cases_per_document_per_answer_state"]) <= MAX_CASES_PER_DOCUMENT_PER_STATE,
        "schema_exclusion_rate_at_most_0_01": float(sampling["schema_exclusion_rate"]) <= 0.01,
        "candidate_ceiling_complete_rate_at_least_0_99": ceiling >= 0.99,
        "single_slot_trigger_rate_at_least_0_1": trigger_rate >= 0.1,
        "single_slot_trigger_rate_at_most_0_8": trigger_rate <= 0.8,
        "candidate_minus_exact_anchor_ablation_point_at_least_0_005": comparisons["candidate_minus_exact_anchor_ablation"]["point"] >= 0.005,
        "candidate_minus_exact_anchor_ablation_ci_low_above_0": comparisons["candidate_minus_exact_anchor_ablation"]["ci_low"] > 0.0,
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_005": comparisons["candidate_minus_strongest_shared_gate_non_frc"]["point"] >= 0.005,
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": comparisons["candidate_minus_strongest_shared_gate_non_frc"]["ci_low"] > 0.0,
        "candidate_minus_strongest_same_gate_frc_point_at_least_0_005": comparisons["candidate_minus_strongest_same_gate_frc"]["point"] >= 0.005,
        "candidate_minus_strongest_same_gate_frc_ci_low_above_0": comparisons["candidate_minus_strongest_same_gate_frc"]["ci_low"] > 0.0,
        "answer_recall_drop_vs_exact_anchor_ablation_at_most_0_01": recall_drop <= 0.01,
        "no_answer_abstention_accuracy_at_least_0_5": candidate["no_answer_abstention_accuracy"] >= 0.5,
        "answer_bearing_gate_pass_rate_at_least_0_5": answer_gate_pass_rate >= 0.5,
        "candidate_abstention_rate_at_least_0_1": candidate["abstention_rate"] >= 0.1,
        "candidate_abstention_rate_at_most_0_8": candidate["abstention_rate"] <= 0.8,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_02": minimum_delta >= -0.02,
        "deterministic_decoy_fallback_rate_equals_0": float(structural["decoy_fallback_rate"]) == 0.0,
        "deterministic_query_fallback_rate_equals_0": query_summary["fallback_rate"] == 0.0,
        "score_or_prediction_fallback_rate_equals_0": float(source_artifacts.get("score_fallback_rate", 0.0)) == 0.0,
    }
    supported = all(checks.values())
    if stage == "development":
        status = (
            "QUAC_V57_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_VALIDATION"
            if supported
            else "QUAC_V57_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION"
        )
    else:
        status = (
            "QUAC_V57_ANCHOR_SAFE_CONSENSUS_SLOT_SUPPORT_ESTABLISHED"
            if supported
            else "QUAC_V57_ANCHOR_SAFE_CONSENSUS_SLOT_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "schema_version": "frc-quac-anchor-safe-consensus-report-v57",
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
            "official_quac_answer_result": False,
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "gold_joined_after_complete_score_cache": True,
            "v56_case_level_artifact_reused": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "selector_formula": "shared-gate cross-encoder Top-5 with first four preserved and at most one fifth-slot role-consensus replacement",
            "aggregates": aggregates,
            "strongest_shared_gate_non_frc": strongest_non_frc,
            "strongest_same_gate_frc": strongest_same_gate_frc,
            "family_comparison": comparisons,
            "answer_recall_drop_vs_exact_anchor_ablation": recall_drop,
            "candidate_ceiling_complete_rate": round(ceiling, 6),
            "single_slot_trigger_rate": round(trigger_rate, 6),
            "gate_diagnostics": {
                "answer_bearing_pass_rate": round(answer_gate_pass_rate, 6),
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
        f"# QuAC anchor-safe consensus slot ({rounded['metadata']['stage']}, v57)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/documents/dialogues: {rounded['metadata']['cases']}/{rounded['metadata']['documents']}/{rounded['metadata']['dialogues']}",
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        f"- Candidate answer F1: {candidate['answer_bearing_macro_f1']:.6f}",
        f"- Candidate answer recall: {candidate['answer_macro_recall']:.6f}",
        f"- No-answer abstention accuracy: {candidate['no_answer_abstention_accuracy']:.6f}",
        f"- Single-slot trigger rate: {analysis['single_slot_trigger_rate']:.6f}",
        "- Selector adoption: `false`",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Family comparisons",
        "",
    ]
    for name, value in analysis["family_comparison"].items():
        lines.append(
            f"- `{name}`: {value['point']:+.6f} (95% CI [{value['ci_low']:+.6f}, {value['ci_high']:+.6f}])"
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
            "This is a balanced QuAC evidence-unit selection mechanism experiment. It is not official QuAC answer extraction, hidden-test evaluation, SetR reproduction, selector adoption, or flood-domain validation.",
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
        raise ValueError("QuAC v57 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value["prior_result_boundary"]["quac_train_or_validation_downloaded_or_opened_before_protocol_registration"] is not False:
        raise ValueError("QuAC v57 source was opened before protocol registration")
    if value["scope"]["development_cases"] != TARGET_CASES:
        raise ValueError("QuAC v57 development size changed")
    if value["methods"]["budgets"] != list(BUDGETS):
        raise ValueError("QuAC v57 budgets changed")
    return value


def validate_source_registration(
    registration_path: Path, *, protocol_path: Path, source_root: Path
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value.get("protocol_sha256_before_download") != _sha256(protocol_path):
        raise ValueError("QuAC v57 source registration protocol hash mismatch")
    registered = {str(row["path"]): str(row["sha256"]) for row in value["files"]}
    for name in SOURCE_FILES.values():
        path = source_root / name
        if registered.get(name) != _sha256(path):
            raise ValueError(f"QuAC v57 source changed: {name}")
    if value.get("json_content_parsed_before_registration") is not False:
        raise ValueError("QuAC v57 source was parsed before registration")
    if value.get("validation_content_read") is not False:
        raise ValueError("QuAC v57 validation was opened before registration")
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
        raise ValueError("QuAC v57 implementation hash mismatch")
    if value.get("quac_json_opened_before_implementation_registration") is not False:
        raise ValueError("QuAC v57 implementation was registered too late")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("QuAC v57 synthetic invariants are incomplete")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BUDGETS",
    "CANDIDATE",
    "CAPABILITY",
    "DATASET_ID",
    "DECOY_COUNT",
    "EXACT_ANCHOR_ABLATION",
    "EXPERIMENT_ID",
    "FrozenQuacScorer",
    "GATED_NON_FRC_METHODS",
    "METHODS",
    "PROTOCOL_SHA256",
    "SAME_GATE_FRC_CONTROLS",
    "SCHEMA_VERSION",
    "SOURCE_FILES",
    "SOURCE_URLS",
    "STAGES",
    "TARGET_CASES",
    "TARGET_PER_GROUP",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "canonical_candidate_rows",
    "evaluate_stage",
    "extract_cases",
    "iter_dialogues",
    "prepare_blind_cases",
    "read_stage_source",
    "select_balanced_sample",
    "select_v57",
    "single_slot_consensus_details",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_registration",
    "write_report",
]
