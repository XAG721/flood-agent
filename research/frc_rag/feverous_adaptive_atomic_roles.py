"""Prospective FEVEROUS v43 bounded-pool selector experiment.

The official FEVEROUS baseline predictions define candidate pages without using
gold evidence.  Gold annotations are joined only after the complete atomic-query
and neural-score caches have passed ordering and blindness checks.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.hover_dynamic_atomic_roles import (
    BASELINES,
    BUDGETS,
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    METHODS as V41_METHODS,
    RANK_COVERAGE_WEIGHT,
    FrozenDynamicHoVerScorer,
    merge_scored_candidates,
    select_candidates,
)
from research.frc_rag.rgb_cost_aware_frc import sha256


SCHEMA_VERSION = "frc-feverous-adaptive-atomic-roles-v43"
EXPERIMENT_ID = "FRC-FEVEROUS-ADAPTIVE-ATOMIC-ROLES-V43"
DATASET_ID = "feverous_official_dev_bounded_pool_v43"
CAPABILITY = "structured_unstructured_element_evidence_selection"
PROTOCOL_SHA256 = (
    "2591f23a9edc498c800a4dfcad680c5e2df36e8178cb30df7214ac7ed7264b57"
)
OFFICIAL_REPOSITORY_REVISION = "32b68ce4e33c53f34ae2e6d88b51cd073ab85ab6"
EXCLUDED_DOCUMENTATION_EXAMPLES = frozenset({33670, 35206})
CHALLENGES = (
    "Numerical Reasoning",
    "Multi-hop Reasoning",
    "Entity Disambiguation",
    "Combining Tables and Text",
    "Search terms not in claim",
    "Other",
)
SAMPLE_SALT = "FRC-FEVEROUS-V43|"
CASES_PER_CHALLENGE = 40
MINIMUM_TOTAL_CASES = 180
MINIMUM_CHALLENGE_CASES = 20
LEXICAL_POOL_SIZE = 80
MAXIMUM_POOL_SIZE = 189
TOP_K = 5
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260803

OFFICIAL_BASELINE = "official_baseline_topk"
ADAPTIVE_ARGMAX = "adaptive_argmax_cardinality_frc_v43"
NON_FRC_BASELINES = (OFFICIAL_BASELINE, *BASELINES)
METHODS = (OFFICIAL_BASELINE, *V41_METHODS, ADAPTIVE_ARGMAX)

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_WHITESPACE = re.compile(r"\s+")
_LINK = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
_EVIDENCE_ID = re.compile(
    r"^(.*)_(header_cell|table_caption|sentence|cell|item)_(.+)$"
)
_FORBIDDEN_BLIND_KEYS = {
    "challenge",
    "evidence",
    "gold",
    "gold_element_ids",
    "gold_pages",
    "label",
    "supporting_facts",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _normalize(value: Any) -> str:
    return _WHITESPACE.sub(" ", str(value or "")).strip()


def _clean_wiki_text(value: Any) -> str:
    text = _LINK.sub(r"\1", str(value or ""))
    return _normalize(text)


def _tokens(value: Any) -> list[str]:
    return [token.lower() for token in _WORD.findall(str(value or ""))]


def _integer_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a FEVEROUS id")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid FEVEROUS id: {value!r}") from exc
    return result


def _optional_row_id(row: dict[str, Any]) -> int | None:
    if "id" not in row or not _normalize(row.get("id")):
        return None
    return _integer_id(row["id"])


def _contains_forbidden_blind_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_BLIND_KEYS:
                return True
            if _contains_forbidden_blind_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_blind_key(item) for item in value)
    return False


def _evidence_groups(row: dict[str, Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in row.get("evidence", []):
        if not isinstance(group, dict):
            continue
        content = [
            _normalize(item)
            for item in group.get("content", [])
            if _normalize(item)
        ]
        if content:
            groups.append(content)
    return groups


def _predicted_triples(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for item in row.get("predicted_evidence", []):
        if isinstance(item, str):
            try:
                result.append(parse_evidence_id(item))
            except ValueError:
                continue
            continue
        if not isinstance(item, list) or len(item) != 3:
            continue
        page, kind, position = (_normalize(value) for value in item)
        if page and kind in {
            "sentence",
            "cell",
            "header_cell",
            "table_caption",
            "item",
        } and position:
            result.append((page, kind, position))
    return result


def canonical_evidence_id(page: str, kind: str, position: str) -> str:
    return f"{page}_{kind}_{position}"


def parse_evidence_id(value: str) -> tuple[str, str, str]:
    match = _EVIDENCE_ID.fullmatch(_normalize(value))
    if match is None:
        raise ValueError(f"unsupported FEVEROUS evidence id: {value!r}")
    return match.group(1), match.group(2), match.group(3)


def _eligible_rows(
    development_rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    baseline_by_id: dict[int, dict[str, Any]] = {}
    for row in baseline_rows:
        row_id = _optional_row_id(row)
        if row_id is not None:
            baseline_by_id[row_id] = row
    census: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in development_rows:
        case_id = _optional_row_id(row)
        if case_id is None:
            census["header_or_non_case_rows"] += 1
            continue
        if case_id in seen:
            raise ValueError(f"duplicate FEVEROUS development id: {case_id}")
        seen.add(case_id)
        if case_id in EXCLUDED_DOCUMENTATION_EXAMPLES:
            census["documentation_examples_excluded"] += 1
            continue
        if not _normalize(row.get("claim")):
            census["empty_claim_excluded"] += 1
            continue
        if not _evidence_groups(row):
            census["empty_evidence_excluded"] += 1
            continue
        challenge = _normalize(row.get("challenge"))
        if challenge not in CHALLENGES:
            census["unknown_challenge_excluded"] += 1
            continue
        baseline = baseline_by_id.get(case_id)
        if baseline is None:
            census["missing_baseline_row_excluded"] += 1
            continue
        triples = _predicted_triples(baseline)
        if not triples:
            census["empty_baseline_prediction_excluded"] += 1
            continue
        eligible.append(
            {
                "id": case_id,
                "claim": _normalize(row["claim"]),
                "challenge": challenge,
                "development": row,
                "baseline": baseline,
            }
        )
    census["eligible"] = len(eligible)
    return eligible, dict(sorted(census.items()))


def select_sample(
    development_rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
    *,
    cases_per_challenge: int = CASES_PER_CHALLENGE,
    minimum_total_cases: int = MINIMUM_TOTAL_CASES,
    minimum_challenge_cases: int = MINIMUM_CHALLENGE_CASES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible, census = _eligible_rows(development_rows, baseline_rows)
    by_challenge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_challenge[str(row["challenge"])].append(row)
    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for challenge in CHALLENGES:
        rows = sorted(
            by_challenge.get(challenge, []),
            key=lambda row: (
                hashlib.sha256(
                    f"{SAMPLE_SALT}{int(row['id'])}".encode("utf-8")
                ).hexdigest(),
                int(row["id"]),
            ),
        )
        if rows and len(rows) < minimum_challenge_cases:
            raise ValueError(
                f"FEVEROUS challenge {challenge!r} has only {len(rows)} eligible cases"
            )
        chosen = rows[:cases_per_challenge]
        selected.extend(chosen)
        counts[challenge] = {"eligible": len(rows), "selected": len(chosen)}
    if len(selected) < minimum_total_cases:
        raise ValueError(
            f"FEVEROUS selected {len(selected)} cases, below {minimum_total_cases}"
        )
    return selected, {
        "eligibility_census": census,
        "challenge_counts": counts,
        "selected": len(selected),
        "selected_ids_sha256": canonical_json_sha256(
            [int(row["id"]) for row in selected]
        ),
        "sampling_salt": SAMPLE_SALT,
        "cases_per_challenge": cases_per_challenge,
    }


class FeverousDatabase:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(str(path))

    def close(self) -> None:
        self.connection.close()

    def get_page(self, page: str) -> dict[str, Any] | None:
        candidates = [page, unicodedata.normalize("NFD", page).strip()]
        cursor = self.connection.cursor()
        try:
            for candidate in dict.fromkeys(candidates):
                cursor.execute("SELECT data FROM wiki WHERE id = ?", (candidate,))
                result = cursor.fetchone()
                if result is not None:
                    value = json.loads(result[0])
                    if isinstance(value, dict):
                        return value
        finally:
            cursor.close()
        return None


def _section_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(value for _, value in stack if value)


def _unit_text(
    page: str,
    sections: list[tuple[int, str]],
    kind: str,
    value: str,
    local_context: str = "",
) -> str:
    pieces = [f"Page: {page}"]
    section = _section_path(sections)
    if section:
        pieces.append(f"Section: {section}")
    if local_context:
        pieces.append(local_context)
    pieces.append(f"{kind}: {value}")
    return " | ".join(pieces)


def extract_page_units(page: str, value: dict[str, Any]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    sections: list[tuple[int, str]] = []
    for entry in value.get("order", []):
        data = value.get(entry)
        if entry.startswith("section_") and isinstance(data, dict):
            level = int(data.get("level", 1))
            text = _clean_wiki_text(data.get("value"))
            while sections and sections[-1][0] >= level:
                sections.pop()
            if text:
                sections.append((level, text))
            continue
        if entry.startswith("sentence_"):
            text = _clean_wiki_text(data)
            if text:
                position = entry.removeprefix("sentence_")
                units.append(
                    {
                        "evidence_id": canonical_evidence_id(
                            page, "sentence", position
                        ),
                        "page": page,
                        "kind": "sentence",
                        "position": position,
                        "text": _unit_text(page, sections, "Sentence", text),
                    }
                )
            continue
        if entry.startswith("table_") and isinstance(data, dict):
            table_number = entry.removeprefix("table_")
            caption = _clean_wiki_text(data.get("caption"))
            if caption:
                units.append(
                    {
                        "evidence_id": canonical_evidence_id(
                            page, "table_caption", table_number
                        ),
                        "page": page,
                        "kind": "table_caption",
                        "position": table_number,
                        "text": _unit_text(
                            page, sections, "Table caption", caption
                        ),
                    }
                )
            rows = data.get("table", [])
            header_values = [
                _clean_wiki_text(cell.get("value"))
                for row in rows
                for cell in row
                if isinstance(cell, dict) and bool(cell.get("is_header"))
            ][:12]
            header_context = "Headers: " + " | ".join(
                value for value in header_values if value
            )
            for row in rows:
                if not isinstance(row, list):
                    continue
                row_values = [
                    _clean_wiki_text(cell.get("value"))
                    for cell in row
                    if isinstance(cell, dict)
                ]
                row_context = "Row: " + " | ".join(
                    value for value in row_values if value
                )
                context = " ; ".join(
                    value for value in (caption, header_context, row_context) if value
                )
                for cell in row:
                    if not isinstance(cell, dict):
                        continue
                    local_id = _normalize(cell.get("id"))
                    text = _clean_wiki_text(cell.get("value"))
                    if not local_id or not text:
                        continue
                    if local_id.startswith("header_cell_"):
                        kind = "header_cell"
                        position = local_id.removeprefix("header_cell_")
                    elif local_id.startswith("cell_"):
                        kind = "cell"
                        position = local_id.removeprefix("cell_")
                    else:
                        continue
                    units.append(
                        {
                            "evidence_id": canonical_evidence_id(
                                page, kind, position
                            ),
                            "page": page,
                            "kind": kind,
                            "position": position,
                            "text": _unit_text(
                                page, sections, "Cell", text, context
                            ),
                        }
                    )
            continue
        if entry.startswith("list_") and isinstance(data, dict):
            for item in data.get("list", []):
                if not isinstance(item, dict):
                    continue
                local_id = _normalize(item.get("id"))
                text = _clean_wiki_text(item.get("value"))
                if not local_id.startswith("item_") or not text:
                    continue
                position = local_id.removeprefix("item_")
                units.append(
                    {
                        "evidence_id": canonical_evidence_id(
                            page, "item", position
                        ),
                        "page": page,
                        "kind": "item",
                        "position": position,
                        "text": _unit_text(page, sections, "List item", text),
                    }
                )
    deduplicated: dict[str, dict[str, str]] = {}
    for unit in units:
        deduplicated.setdefault(str(unit["evidence_id"]), unit)
    return list(deduplicated.values())


def _bm25_scores(query: str, texts: Sequence[str]) -> list[float]:
    documents = [_tokens(text) for text in texts]
    if not documents:
        return []
    query_tokens = _tokens(query)
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    total_documents = len(documents)
    average_length = sum(len(document) for document in documents) / max(
        1, total_documents
    )
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            df = document_frequency[token]
            inverse = math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
            )
            score += inverse * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def _truncate_with_tokenizer(text: str, tokenizer: Any) -> tuple[str, int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return text, 1
    token_ids = token_ids[:256]
    if len(tokenizer.encode(text, add_special_tokens=False)) > 256:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return _normalize(text), len(token_ids)


def prepare_blind_cases(
    development_rows: Iterable[dict[str, Any]],
    baseline_rows: Iterable[dict[str, Any]],
    database_path: Path,
    tokenizer: Any,
    *,
    cases_per_challenge: int = CASES_PER_CHALLENGE,
    minimum_total_cases: int = MINIMUM_TOTAL_CASES,
    minimum_challenge_cases: int = MINIMUM_CHALLENGE_CASES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, selection_census = select_sample(
        development_rows,
        baseline_rows,
        cases_per_challenge=cases_per_challenge,
        minimum_total_cases=minimum_total_cases,
        minimum_challenge_cases=minimum_challenge_cases,
    )
    database = FeverousDatabase(database_path)
    prepared: list[dict[str, Any]] = []
    candidate_maps: list[dict[str, Any]] = []
    missing_pages = 0
    invalid_predicted_units = 0
    page_counts: list[int] = []
    raw_unit_counts: list[int] = []
    pool_counts: list[int] = []
    token_counts: list[int] = []
    try:
        for selected_row in selected:
            numeric_id = int(selected_row["id"])
            case_id = f"feverous-v43::{numeric_id}"
            triples = _predicted_triples(dict(selected_row["baseline"]))
            official_ids = [
                canonical_evidence_id(page, kind, position)
                for page, kind, position in triples
            ]
            official_rank = {
                evidence_id: rank
                for rank, evidence_id in enumerate(dict.fromkeys(official_ids))
            }
            pages = list(dict.fromkeys(page for page, _, _ in triples))
            all_units: list[dict[str, str]] = []
            valid_pages: list[str] = []
            for page in pages:
                page_json = database.get_page(page)
                if page_json is None:
                    missing_pages += 1
                    continue
                valid_pages.append(page)
                all_units.extend(extract_page_units(page, page_json))
            units_by_id = {
                str(unit["evidence_id"]): unit for unit in all_units
            }
            valid_official = [
                evidence_id
                for evidence_id in official_rank
                if evidence_id in units_by_id
            ]
            invalid_predicted_units += len(official_rank) - len(valid_official)
            if not units_by_id or not valid_official:
                raise ValueError(
                    f"FEVEROUS case {numeric_id} has no valid official predicted unit"
                )
            units = list(units_by_id.values())
            scores = _bm25_scores(
                str(selected_row["claim"]),
                [str(unit["text"]) for unit in units],
            )
            ordered = sorted(
                zip(units, scores, strict=True),
                key=lambda item: (-float(item[1]), str(item[0]["evidence_id"])),
            )
            pool_ids = [
                str(unit["evidence_id"])
                for unit, _ in ordered[:LEXICAL_POOL_SIZE]
            ]
            pool_ids.extend(valid_official)
            pool_ids = list(dict.fromkeys(pool_ids))
            if len(pool_ids) > MAXIMUM_POOL_SIZE:
                raise AssertionError("FEVEROUS candidate pool exceeds frozen maximum")
            rank_by_id = {
                str(unit["evidence_id"]): rank for rank, (unit, _) in enumerate(ordered)
            }
            pool_ids.sort(key=lambda value: (rank_by_id[value], value))
            candidates: list[dict[str, Any]] = []
            mappings: list[dict[str, Any]] = []
            for index, evidence_id in enumerate(pool_ids):
                unit = units_by_id[evidence_id]
                text, token_count = _truncate_with_tokenizer(
                    str(unit["text"]), tokenizer
                )
                opaque_id = f"u{index:04d}"
                candidates.append(
                    {
                        "id": opaque_id,
                        "source_id": opaque_id,
                        "text": text,
                        "token_count": token_count,
                        "official_rank": official_rank.get(evidence_id),
                    }
                )
                mappings.append(
                    {
                        "candidate_id": opaque_id,
                        "evidence_id": evidence_id,
                        "page": str(unit["page"]),
                        "kind": str(unit["kind"]),
                        "position": str(unit["position"]),
                        "official_rank": official_rank.get(evidence_id),
                    }
                )
                token_counts.append(token_count)
            case = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "capability": CAPABILITY,
                "id": case_id,
                "query": str(selected_row["claim"]),
                "candidates": candidates,
                "gold_fields_visible_to_scorer": False,
            }
            if _contains_forbidden_blind_key(case):
                raise AssertionError("FEVEROUS gold field leaked into blind cache")
            prepared.append(case)
            candidate_maps.append(
                {
                    "schema_version": "frc-feverous-v43-candidate-map-v1",
                    "id": case_id,
                    "dataset_numeric_id": numeric_id,
                    "units": mappings,
                }
            )
            page_counts.append(len(valid_pages))
            raw_unit_counts.append(len(units))
            pool_counts.append(len(candidates))
    finally:
        database.close()

    def distribution(values: Sequence[int]) -> dict[str, float | int]:
        return {
            "minimum": int(min(values)),
            "mean": round(float(np.mean(values)), 6),
            "maximum": int(max(values)),
        }

    summary = {
        **selection_census,
        "valid_rows": len(prepared),
        "case_ids_sha256": canonical_json_sha256(
            [str(case["id"]) for case in prepared]
        ),
        "candidate_pages": distribution(page_counts),
        "raw_candidate_units": distribution(raw_unit_counts),
        "blind_pool_units": distribution(pool_counts),
        "unit_token_cost": distribution(token_counts),
        "missing_candidate_pages": missing_pages,
        "invalid_official_predicted_units": invalid_predicted_units,
        "gold_identifiers_or_labels_exported": False,
        "challenge_exported_to_blind_cache": False,
        "gold_fields_visible_to_scorer": False,
    }
    return prepared, candidate_maps, summary


class FrozenFeverousScorer:
    def __init__(self, **kwargs: Any) -> None:
        self.delegate = FrozenDynamicHoVerScorer(**kwargs)

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = self.delegate.score_cases(cases)
        case_by_id = {str(case["id"]): case for case in cases}
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
            metadata = {
                str(candidate["id"]): candidate.get("official_rank")
                for candidate in case_by_id[str(row["id"])]["candidates"]
            }
            for candidate in row["candidates"]:
                candidate["official_rank"] = metadata[str(candidate["id"])]
        return rows


def _merge_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = merge_scored_candidates(row)
    ranks = {
        str(candidate["id"]): candidate.get("official_rank")
        for candidate in row.get("candidates", [])
    }
    for candidate in candidates:
        candidate["official_rank"] = ranks[str(candidate["id"])]
    return candidates


def _take_with_budget(
    ordered: Sequence[dict[str, Any]], *, token_budget: int, top_k: int = TOP_K
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0
    for candidate in ordered:
        cost = int(candidate["token_count"])
        if total + cost > token_budget:
            continue
        selected.append(candidate)
        total += cost
        if len(selected) >= top_k:
            break
    return selected


def _rank_utilities(
    candidates: Sequence[dict[str, Any]], score_key: str
) -> dict[str, dict[str, float]]:
    result = {str(candidate["id"]): {} for candidate in candidates}
    for role in DYNAMIC_ROLES:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate[score_key][role]),
                str(candidate["id"]),
            ),
        )
        for rank, candidate in enumerate(ordered):
            result[str(candidate["id"])][role] = 1.0 / math.log2(2 + rank)
    return result


def adaptive_target_cardinality(candidates: Sequence[dict[str, Any]]) -> int:
    if not candidates:
        return 0
    argmax_ids = {
        str(
            min(
                candidates,
                key=lambda candidate: (
                    -float(candidate["dynamic_role_scores"][role]),
                    str(candidate["id"]),
                ),
            )["id"]
        )
        for role in DYNAMIC_ROLES
    }
    return min(TOP_K, max(1, len(argmax_ids)))


def _adaptive_select(
    candidates: list[dict[str, Any]], *, token_budget: int
) -> list[dict[str, Any]]:
    target = adaptive_target_cardinality(candidates)
    utilities = _rank_utilities(candidates, "dynamic_role_scores")
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    best_by_role = {role: 0.0 for role in DYNAMIC_ROLES}
    total = 0
    while remaining and len(selected) < target:
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in remaining:
            cost = int(candidate["token_count"])
            if total + cost > token_budget:
                continue
            role_gain = sum(
                max(
                    0.0,
                    utilities[str(candidate["id"])][role] - best_by_role[role],
                )
                for role in DYNAMIC_ROLES
            )
            gain = float(candidate["scores"]["cross_encoder"]) + (
                RANK_COVERAGE_WEIGHT * role_gain
            )
            eligible.append((gain, str(candidate["id"]), candidate))
        if not eligible:
            break
        eligible.sort(key=lambda value: (-value[0], value[1]))
        best = eligible[0][2]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        for role in DYNAMIC_ROLES:
            best_by_role[role] = max(
                best_by_role[role], utilities[str(best["id"])][role]
            )
    return selected


def select_v43(
    candidates: list[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    if method == OFFICIAL_BASELINE:
        ordered = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.get("official_rank") is not None
            ),
            key=lambda candidate: (
                int(candidate["official_rank"]),
                str(candidate["id"]),
            ),
        )
        return _take_with_budget(ordered, token_budget=token_budget)
    if method == ADAPTIVE_ARGMAX:
        return _adaptive_select(candidates, token_budget=token_budget)
    return select_candidates(candidates, method, token_budget=token_budget)


def build_gold_rows(
    development_rows: Iterable[dict[str, Any]],
    candidate_maps: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    development_by_id: dict[int, dict[str, Any]] = {}
    for row in development_rows:
        row_id = _optional_row_id(row)
        if row_id is not None:
            development_by_id[row_id] = row
    gold_rows: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        numeric_id = _integer_id(candidate_map["dataset_numeric_id"])
        row = development_by_id.get(numeric_id)
        if row is None:
            raise ValueError(f"FEVEROUS gold row missing for {numeric_id}")
        evidence_groups = _evidence_groups(row)
        candidate_by_evidence = {
            str(unit["evidence_id"]): str(unit["candidate_id"])
            for unit in candidate_map.get("units", [])
        }
        groups: list[dict[str, Any]] = []
        union_candidate_ids: set[str] = set()
        for group in evidence_groups:
            mapped = [
                candidate_by_evidence[evidence_id]
                for evidence_id in group
                if evidence_id in candidate_by_evidence
            ]
            union_candidate_ids.update(mapped)
            pages = {parse_evidence_id(evidence_id)[0] for evidence_id in group}
            groups.append(
                {
                    "gold_element_count": len(group),
                    "gold_page_count": len(pages),
                    "candidate_ids": mapped,
                    "complete_in_pool": len(mapped) == len(group),
                }
            )
        gold_rows.append(
            {
                "case_id": str(candidate_map["id"]),
                "dataset_numeric_id": numeric_id,
                "label": _normalize(row.get("label")),
                "challenge": _normalize(row.get("challenge")),
                "groups": groups,
                "union_candidate_ids": sorted(union_candidate_ids),
                "candidate_ceiling_complete": any(
                    bool(group["complete_in_pool"]) for group in groups
                ),
                "minimum_gold_element_count": min(
                    int(group["gold_element_count"]) for group in groups
                ),
                "minimum_gold_page_count": min(
                    int(group["gold_page_count"]) for group in groups
                ),
            }
        )
    return gold_rows


def _selection_metrics(
    selected: Sequence[dict[str, Any]], gold: dict[str, Any]
) -> dict[str, float | int | bool]:
    selected_ids = {str(candidate["id"]) for candidate in selected}
    union_gold = set(str(value) for value in gold["union_candidate_ids"])
    hits = len(selected_ids & union_gold)
    precision = hits / len(selected_ids) if selected_ids else 1.0
    complete_recall = any(
        bool(group["complete_in_pool"])
        and set(str(value) for value in group["candidate_ids"]).issubset(selected_ids)
        for group in gold["groups"]
    )
    return {
        "precision": precision,
        "complete_evidence_recall": float(complete_recall),
        "selected_unit_count": len(selected),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "gold_unit_hits": hits,
    }


def _aggregate(
    rows: Sequence[dict[str, Any]], method: str, budgets: Sequence[int] = BUDGETS
) -> dict[str, float]:
    precision_values: list[float] = []
    recall_values: list[float] = []
    count_values: list[float] = []
    cost_values: list[float] = []
    f1_by_budget: list[float] = []
    for budget in budgets:
        metrics = [
            row["configurations"][str(budget)]["methods"][method]["metrics"]
            for row in rows
        ]
        precision = float(np.mean([float(item["precision"]) for item in metrics]))
        recall = float(
            np.mean([float(item["complete_evidence_recall"]) for item in metrics])
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_by_budget.append(f1)
        precision_values.extend(float(item["precision"]) for item in metrics)
        recall_values.extend(
            float(item["complete_evidence_recall"]) for item in metrics
        )
        count_values.extend(float(item["selected_unit_count"]) for item in metrics)
        cost_values.extend(float(item["selected_token_cost"]) for item in metrics)
    return {
        "evidence_macro_f1": float(np.mean(f1_by_budget)),
        "macro_precision": float(np.mean(precision_values)),
        "complete_evidence_recall": float(np.mean(recall_values)),
        "mean_selected_unit_count": float(np.mean(count_values)),
        "mean_selected_token_cost": float(np.mean(cost_values)),
    }


def _comparison(point: float, values: np.ndarray) -> dict[str, float]:
    return {
        "point": round(float(point), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def _bootstrap(
    rows: Sequence[dict[str, Any]], *, resamples: int
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    adaptive_dynamic = np.empty(resamples, dtype=float)
    adaptive_strongest = np.empty(resamples, dtype=float)
    for index in range(resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(item)] for item in indices]
        adaptive = _aggregate(sample, ADAPTIVE_ARGMAX)["evidence_macro_f1"]
        dynamic = _aggregate(sample, DYNAMIC_RANK)["evidence_macro_f1"]
        strongest = max(
            _aggregate(sample, method)["evidence_macro_f1"]
            for method in NON_FRC_BASELINES
        )
        adaptive_dynamic[index] = adaptive - dynamic
        adaptive_strongest[index] = adaptive - strongest
    adaptive_point = _aggregate(rows, ADAPTIVE_ARGMAX)["evidence_macro_f1"]
    dynamic_point = _aggregate(rows, DYNAMIC_RANK)["evidence_macro_f1"]
    strongest_point = max(
        _aggregate(rows, method)["evidence_macro_f1"]
        for method in NON_FRC_BASELINES
    )
    return {
        "adaptive_minus_dynamic": _comparison(
            adaptive_point - dynamic_point, adaptive_dynamic
        ),
        "adaptive_minus_bootstrap_strongest_non_frc": _comparison(
            adaptive_point - strongest_point, adaptive_strongest
        ),
    }


def evaluate_feverous(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if [str(row["case_id"]) for row in gold_rows] != [
        str(row["id"]) for row in scored_rows
    ]:
        raise ValueError("FEVEROUS gold and score rows are incomplete or misordered")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        candidates = _merge_candidates(dict(scored))
        configurations: dict[str, Any] = {}
        target = adaptive_target_cardinality(candidates)
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v43(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(selected, dict(gold)),
                }
                if method == ADAPTIVE_ARGMAX:
                    methods[method]["target_cardinality"] = target
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "dataset_numeric_id": int(gold["dataset_numeric_id"]),
                "label": str(gold["label"]),
                "challenge": str(gold["challenge"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "minimum_gold_element_count": int(
                    gold["minimum_gold_element_count"]
                ),
                "minimum_gold_page_count": int(gold["minimum_gold_page_count"]),
                "candidate_unit_count": len(candidates),
                "adaptive_target_cardinality": target,
                "configurations": configurations,
            }
        )

    aggregates = {
        method: {key: round(value, 6) for key, value in _aggregate(evidence, method).items()}
        for method in METHODS
    }
    strongest_non_frc = max(
        NON_FRC_BASELINES,
        key=lambda method: float(aggregates[method]["evidence_macro_f1"]),
    )
    bootstrap = _bootstrap(evidence, resamples=resamples)
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: _aggregate(evidence, method, (budget,))[
                "evidence_macro_f1"
            ],
        )
        budget_deltas[str(budget)] = round(
            _aggregate(evidence, ADAPTIVE_ARGMAX, (budget,))["evidence_macro_f1"]
            - _aggregate(evidence, strongest, (budget,))["evidence_macro_f1"],
            6,
        )

    supported_challenge_deltas: dict[str, dict[str, Any]] = {}
    for challenge in CHALLENGES:
        subset = [row for row in evidence if row["challenge"] == challenge]
        if len(subset) < MINIMUM_CHALLENGE_CASES:
            continue
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: _aggregate(subset, method)["evidence_macro_f1"],
        )
        supported_challenge_deltas[challenge] = {
            "cases": len(subset),
            "strongest_non_frc": strongest,
            "adaptive_evidence_macro_f1": round(
                _aggregate(subset, ADAPTIVE_ARGMAX)["evidence_macro_f1"], 6
            ),
            "delta": round(
                _aggregate(subset, ADAPTIVE_ARGMAX)["evidence_macro_f1"]
                - _aggregate(subset, strongest)["evidence_macro_f1"],
                6,
            ),
        }

    ceiling_rate = float(
        np.mean([bool(row["candidate_ceiling_complete"]) for row in evidence])
    )
    count_reduction = (
        float(aggregates[DYNAMIC_RANK]["mean_selected_unit_count"])
        - float(aggregates[ADAPTIVE_ARGMAX]["mean_selected_unit_count"])
    )
    recall_drop = (
        float(aggregates[DYNAMIC_RANK]["complete_evidence_recall"])
        - float(aggregates[ADAPTIVE_ARGMAX]["complete_evidence_recall"])
    )
    adaptive_dynamic = bootstrap["adaptive_minus_dynamic"]
    adaptive_strongest = bootstrap[
        "adaptive_minus_bootstrap_strongest_non_frc"
    ]
    safety_values = [
        *budget_deltas.values(),
        *(
            float(value["delta"])
            for value in supported_challenge_deltas.values()
        ),
    ]
    checks = {
        "minimum_total_cases_met": len(evidence) >= MINIMUM_TOTAL_CASES,
        "candidate_ceiling_complete_rate_at_least_0_60": ceiling_rate >= 0.60,
        "query_parser_fallback_rate_at_most_0_05": (
            float(query_summary["fallback_rate"]) <= 0.05
        ),
        "adaptive_minus_dynamic_point_at_least_0_01": (
            float(adaptive_dynamic["point"]) >= 0.01
        ),
        "adaptive_minus_dynamic_ci_low_above_0": (
            float(adaptive_dynamic["ci_low"]) > 0.0
        ),
        "adaptive_minus_strongest_point_at_least_0_01": (
            float(adaptive_strongest["point"]) >= 0.01
        ),
        "adaptive_minus_strongest_ci_low_above_0": (
            float(adaptive_strongest["ci_low"]) > 0.0
        ),
        "every_budget_and_supported_challenge_delta_at_least_minus_0_02": (
            min(safety_values) >= -0.02 if safety_values else False
        ),
        "mean_selected_unit_reduction_at_least_0_50": count_reduction >= 0.50,
        "complete_evidence_recall_drop_at_most_0_02": recall_drop <= 0.02,
    }
    informative = checks["minimum_total_cases_met"] and checks[
        "candidate_ceiling_complete_rate_at_least_0_60"
    ]
    if not informative:
        status = "FEVEROUS_BOUNDED_POOL_INCONCLUSIVE"
    elif all(checks.values()):
        status = "FEVEROUS_ADAPTIVE_ATOMIC_ROLE_SUPPORT_ESTABLISHED_ON_BOUNDED_POOL"
    else:
        status = "FEVEROUS_ADAPTIVE_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED"

    target_distribution = Counter(
        int(row["adaptive_target_cardinality"]) for row in evidence
    )
    report = {
        "schema_version": "frc-feverous-adaptive-atomic-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "protocol_sha256": PROTOCOL_SHA256,
            "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "methods": list(METHODS),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "gold_joined_after_complete_score_cache": True,
            "official_leaderboard_result": False,
        },
        "analysis": {
            "aggregates": aggregates,
            "strongest_non_frc": strongest_non_frc,
            "family_comparison": bootstrap,
            "budget_deltas": budget_deltas,
            "supported_challenge_deltas": supported_challenge_deltas,
            "candidate_ceiling_complete_rate": round(ceiling_rate, 6),
            "mean_selected_unit_reduction_vs_dynamic": round(count_reduction, 6),
            "complete_evidence_recall_drop_vs_dynamic": round(recall_drop, 6),
            "target_cardinality_distribution": {
                str(key): target_distribution[key]
                for key in sorted(target_distribution)
            },
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "gate_2": "NO-GO/SHADOW",
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
            },
        },
        "limitations": [
            "Candidate pages come from the official FEVEROUS baseline output rather than a new full-corpus retriever.",
            "This is a bounded-pool evidence selector experiment, not an official leaderboard run.",
            "The method was developed after observing the v42 SciFact result; SciFact was not reused for evaluation or parameter selection.",
            "The result cannot establish real SetR reproduction, flood-domain validity, or production readiness.",
        ],
    }
    return report, evidence


def write_report(
    report: dict[str, Any], evidence: Sequence[dict[str, Any]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "feverous_adaptive_atomic_roles.json"
    markdown_path = output_dir / "feverous_adaptive_atomic_roles.md"
    evidence_path = output_dir / "feverous_adaptive_atomic_roles_cases.jsonl.gz"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = report["analysis"]
    comparison = analysis["family_comparison"]
    lines = [
        "# FEVEROUS 自适应原子角色选择器（v43）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 可评测样本：{report['metadata']['cases']}",
        f"- 候选池完整证据组上限率：{analysis['candidate_ceiling_complete_rate']:.6f}",
        f"- 最强非 FRC 基线：`{analysis['strongest_non_frc']}`",
        f"- 自适应减动态角色：{comparison['adaptive_minus_dynamic']['point']:+.6f}，95% CI [{comparison['adaptive_minus_dynamic']['ci_low']:+.6f}, {comparison['adaptive_minus_dynamic']['ci_high']:+.6f}]",
        f"- 自适应减重采样最强非 FRC：{comparison['adaptive_minus_bootstrap_strongest_non_frc']['point']:+.6f}，95% CI [{comparison['adaptive_minus_bootstrap_strongest_non_frc']['ci_low']:+.6f}, {comparison['adaptive_minus_bootstrap_strongest_non_frc']['ci_high']:+.6f}]",
        f"- 相对动态方法平均少选证据单元：{analysis['mean_selected_unit_reduction_vs_dynamic']:.6f}",
        f"- 相对动态方法完整证据召回下降：{analysis['complete_evidence_recall_drop_vs_dynamic']:+.6f}",
        "- Gate 2：`NO-GO/SHADOW`；本实验不授权 CANARY/DEFAULT。",
        "",
        "## 方法汇总",
        "",
        "| 方法 | Evidence F1 | Precision | Complete recall | 平均单元数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = analysis["aggregates"][method]
        lines.append(
            f"| `{method}` | {value['evidence_macro_f1']:.6f} | {value['macro_precision']:.6f} | {value['complete_evidence_recall']:.6f} | {value['mean_selected_unit_count']:.6f} |"
        )
    lines.extend(["", "## 支持检查", ""])
    for name, passed in analysis["support_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "该结果只比较官方基线候选页内的证据单元选择，不是 FEVEROUS 官方榜单结果，也不证明全库检索、真实 SetR、防汛领域效果或生产可用性。",
            "",
        ]
    )
    markdown_path.write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in evidence:
                handle.write(
                    (
                        json.dumps(row, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    ).encode("utf-8")
                )
    return {"json": json_path, "markdown": markdown_path, "cases": evidence_path}


def validate_protocol(
    path: Path, *, erratum_path: Path | None = None
) -> dict[str, Any]:
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("FEVEROUS v43 protocol hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 experiment id mismatch")
    if value["methods"]["candidate_method"] != ADAPTIVE_ARGMAX:
        raise ValueError("FEVEROUS v43 candidate method mismatch")
    if value["methods"]["token_budgets"] != list(BUDGETS):
        raise ValueError("FEVEROUS v43 budget registration mismatch")
    if erratum_path is None:
        return value
    erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
    if erratum.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 protocol erratum experiment mismatch")
    if erratum.get("base_protocol_sha256") != sha256(path):
        raise ValueError("FEVEROUS v43 protocol erratum base hash mismatch")
    correction = erratum.get("correction", {})
    if correction.get("field") != (
        "blind_candidate_construction.maximum_pool_size"
    ):
        raise ValueError("FEVEROUS v43 protocol erratum field mismatch")
    if correction.get("original") != 110:
        raise ValueError("FEVEROUS v43 protocol erratum original mismatch")
    if correction.get("effective") != MAXIMUM_POOL_SIZE:
        raise ValueError("FEVEROUS v43 protocol erratum capacity mismatch")
    if correction.get("selection_rule_changed") is not False:
        raise ValueError("FEVEROUS v43 protocol erratum changed selection")
    if erratum.get("method_or_threshold_changed") is not False:
        raise ValueError("FEVEROUS v43 protocol erratum changed the method")
    return {"base_protocol": value, "effective_erratum": erratum}


def validate_implementation_registration(
    path: Path,
    *,
    protocol_path: Path,
    protocol_erratum_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
    inherited_module_path: Path,
    erratum_path: Path | None = None,
    erratum2_path: Path | None = None,
    erratum3_path: Path | None = None,
    erratum4_path: Path | None = None,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(protocol_path, erratum_path=protocol_erratum_path)
    expected = {
        "protocol_sha256": sha256(protocol_path),
        "protocol_erratum_sha256": sha256(protocol_erratum_path),
        "module_sha256": sha256(module_path),
        "runner_sha256": sha256(runner_path),
        "test_sha256": sha256(test_path),
        "inherited_v41_module_sha256": sha256(inherited_module_path),
    }
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 implementation experiment mismatch")
    if value.get("data_content_accessed_before_registration") is not False:
        raise ValueError("FEVEROUS v43 implementation blindness not registered")
    if value.get("hashes") == expected:
        return value
    if erratum_path is None:
        raise ValueError("FEVEROUS v43 implementation registration mismatch")
    erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
    if erratum.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 implementation erratum experiment mismatch")
    if erratum.get("superseded_registration_sha256") != sha256(path):
        raise ValueError("FEVEROUS v43 implementation erratum base hash mismatch")
    if erratum.get("original_hashes") != value.get("hashes"):
        raise ValueError("FEVEROUS v43 implementation erratum history mismatch")
    if erratum.get("method_or_threshold_changed") is not False:
        raise ValueError("FEVEROUS v43 erratum cannot change the registered method")
    if erratum.get("effective_hashes") == expected:
        return {"base_registration": value, "effective_erratum": erratum}
    if erratum2_path is None:
        raise ValueError("FEVEROUS v43 implementation erratum hash mismatch")
    erratum2 = json.loads(erratum2_path.read_text(encoding="utf-8"))
    if erratum2.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 implementation erratum2 experiment mismatch")
    if erratum2.get("superseded_erratum_sha256") != sha256(erratum_path):
        raise ValueError("FEVEROUS v43 implementation erratum2 base hash mismatch")
    if erratum2.get("previous_effective_hashes") != erratum.get(
        "effective_hashes"
    ):
        raise ValueError("FEVEROUS v43 implementation erratum2 history mismatch")
    if erratum2.get("method_or_threshold_changed") is not False:
        raise ValueError("FEVEROUS v43 erratum2 cannot change the registered method")
    if erratum2.get("effective_hashes") == expected:
        return {
            "base_registration": value,
            "first_erratum": erratum,
            "effective_erratum": erratum2,
        }
    if erratum3_path is None:
        raise ValueError("FEVEROUS v43 implementation erratum2 hash mismatch")
    erratum3 = json.loads(erratum3_path.read_text(encoding="utf-8"))
    if erratum3.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 implementation erratum3 experiment mismatch")
    if erratum3.get("superseded_erratum_sha256") != sha256(erratum2_path):
        raise ValueError("FEVEROUS v43 implementation erratum3 base hash mismatch")
    if erratum3.get("previous_effective_hashes") != erratum2.get(
        "effective_hashes"
    ):
        raise ValueError("FEVEROUS v43 implementation erratum3 history mismatch")
    if erratum3.get("protocol_erratum_sha256") != sha256(
        protocol_erratum_path
    ):
        raise ValueError("FEVEROUS v43 implementation erratum3 protocol mismatch")
    if erratum3.get("method_or_threshold_changed") is not False:
        raise ValueError("FEVEROUS v43 erratum3 cannot change the registered method")
    if erratum3.get("effective_hashes") == expected:
        return {
            "base_registration": value,
            "first_erratum": erratum,
            "second_erratum": erratum2,
            "effective_erratum": erratum3,
        }
    if erratum4_path is None:
        raise ValueError("FEVEROUS v43 implementation erratum3 hash mismatch")
    erratum4 = json.loads(erratum4_path.read_text(encoding="utf-8"))
    if erratum4.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 implementation erratum4 experiment mismatch")
    if erratum4.get("superseded_erratum_sha256") != sha256(erratum3_path):
        raise ValueError("FEVEROUS v43 implementation erratum4 base hash mismatch")
    if erratum4.get("previous_effective_hashes") != erratum3.get(
        "effective_hashes"
    ):
        raise ValueError("FEVEROUS v43 implementation erratum4 history mismatch")
    if erratum4.get("effective_hashes") != expected:
        raise ValueError("FEVEROUS v43 implementation erratum4 hash mismatch")
    if erratum4.get("method_or_threshold_changed") is not False:
        raise ValueError("FEVEROUS v43 erratum4 cannot change the registered method")
    if erratum4.get("trigger", {}).get("gold_metrics_computed_in_memory") is not True:
        raise ValueError("FEVEROUS v43 erratum4 must disclose metric computation")
    return {
        "base_registration": value,
        "first_erratum": erratum,
        "second_erratum": erratum2,
        "third_erratum": erratum3,
        "effective_erratum": erratum4,
    }


def validate_execution_registration(
    path: Path,
    *,
    implementation_path: Path,
    implementation_erratum_path: Path,
    implementation_erratum2_path: Path,
    implementation_erratum3_path: Path,
    implementation_erratum4_path: Path | None,
    protocol_path: Path,
    protocol_erratum_path: Path,
    development_path: Path,
    baseline_path: Path,
    database_archive_path: Path,
    database_path: Path,
    prepared_path: Path,
    candidate_map_path: Path,
    census_path: Path,
    query_path: Path | None,
    scored_path: Path | None,
    execution_erratum_path: Path | None,
    runtime_parameters: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": sha256(implementation_path),
        "implementation_erratum_sha256": sha256(implementation_erratum_path),
        "implementation_erratum2_sha256": sha256(implementation_erratum2_path),
        "implementation_erratum3_sha256": sha256(implementation_erratum3_path),
        "protocol_sha256": sha256(protocol_path),
        "protocol_erratum_sha256": sha256(protocol_erratum_path),
        "development_sha256": sha256(development_path),
        "baseline_prediction_sha256": sha256(baseline_path),
        "database_archive_sha256": sha256(database_archive_path),
        "database_sha256": sha256(database_path),
        "prepared_blind_sha256": sha256(prepared_path),
        "candidate_map_sha256": sha256(candidate_map_path),
        "census_sha256": sha256(census_path),
    }
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 execution experiment mismatch")
    if value.get("hashes") != expected:
        raise ValueError("FEVEROUS v43 execution hash mismatch")
    if value.get("runtime_parameters") != runtime_parameters:
        raise ValueError("FEVEROUS v43 execution runtime mismatch")
    if value.get("query_generation_started_before_registration") is not False:
        raise ValueError("FEVEROUS v43 execution order mismatch")
    if execution_erratum_path is None:
        return value
    if implementation_erratum4_path is None or query_path is None or scored_path is None:
        raise ValueError("FEVEROUS v43 execution erratum inputs are incomplete")
    erratum = json.loads(execution_erratum_path.read_text(encoding="utf-8"))
    if erratum.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("FEVEROUS v43 execution erratum experiment mismatch")
    if erratum.get("base_execution_sha256") != sha256(path):
        raise ValueError("FEVEROUS v43 execution erratum base hash mismatch")
    if erratum.get("previous_execution_hashes") != value.get("hashes"):
        raise ValueError("FEVEROUS v43 execution erratum history mismatch")
    expected_erratum = {
        "implementation_erratum4_sha256": sha256(implementation_erratum4_path),
        "queries_sha256": sha256(query_path),
        "scored_sha256": sha256(scored_path),
    }
    if erratum.get("effective_hashes") != expected_erratum:
        raise ValueError("FEVEROUS v43 execution erratum hash mismatch")
    if erratum.get("method_or_threshold_changed") is not False:
        raise ValueError("FEVEROUS v43 execution erratum changed the method")
    trigger = erratum.get("trigger", {})
    if trigger.get("gold_metrics_computed_in_memory") is not True:
        raise ValueError("FEVEROUS v43 execution erratum omits metric disclosure")
    if trigger.get("report_or_result_artifact_written") is not False:
        raise ValueError("FEVEROUS v43 execution erratum report boundary mismatch")
    return {"base_execution": value, "effective_erratum": erratum}


__all__ = [
    "ADAPTIVE_ARGMAX",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CHALLENGES",
    "EXPERIMENT_ID",
    "FrozenFeverousScorer",
    "METHODS",
    "MAXIMUM_POOL_SIZE",
    "OFFICIAL_BASELINE",
    "PROTOCOL_SHA256",
    "adaptive_target_cardinality",
    "build_gold_rows",
    "canonical_evidence_id",
    "evaluate_feverous",
    "extract_page_units",
    "parse_evidence_id",
    "prepare_blind_cases",
    "read_jsonl",
    "select_sample",
    "select_v43",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "write_report",
]
