"""Prospective OTT-QA guarded adaptive-cardinality experiment (v44).

The module keeps dataset preparation, blind neural scoring, and gold joining
separate.  OTT-QA answer fields are consumed only by deterministic sampling and
the post-score gold join; they are never exported to query or score caches.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from contextlib import nullcontext
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


SCHEMA_VERSION = "frc-ottqa-guarded-adaptive-atomic-roles-v44"
EXPERIMENT_ID = "FRC-OTTQA-GUARDED-ADAPTIVE-ATOMIC-ROLES-V44"
DATASET_ID = "ottqa_dev_oracle_table_linked_evidence_v44"
CAPABILITY = "oracle_table_linked_answer_evidence_selection"
PROTOCOL_SHA256 = "95116d2af455d1f8792e81cc1ed1f0f48db5fd7ef1274d257f73cec8e5c1e072"
PROTOCOL_ERRATUM_SHA256 = (
    "e7341f3ce645f27ea1fbf3f8bc79691b14a8c4b5655035f83bdb8112ff906019"
)
PROTOCOL_ERRATUM2_SHA256 = (
    "1bcbb7ee99e974b80624bdc24392d1c6eb403e2dfdd86b4681afbab55a9b6fbc"
)
PROTOCOL_ERRATUM3_SHA256 = (
    "aa237afc9062211cc090247285fb8cc3f2f4fb92e6a8f8c190e1c4a77923ee38"
)
OFFICIAL_REPOSITORY_REVISION = "b289bcca691db21a5259c3420e6b9819a9be9ba3"

SAMPLE_SALT = "FRC-OTTQA-V44|"
TARGET_CASES_PER_MODE = 120
MINIMUM_TOTAL_CASES = 180
MINIMUM_CASES_PER_MODE = 80
SOURCE_MODES = ("table", "passage")

CHUNK_WINDOW = 192
CHUNK_STRIDE = 160
LEXICAL_POOL_SIZE = 96
MAXIMUM_POOL_SIZE = 128
TOP_K = 5
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260804

OFFICIAL_ANCHOR = "official_anchor_topk"
ADAPTIVE_V43 = "adaptive_argmax_cardinality_frc_v43"
GUARDED_V44 = "guarded_adaptive_cardinality_frc_v44"
NON_FRC_BASELINES = (OFFICIAL_ANCHOR, *BASELINES)
METHODS = (OFFICIAL_ANCHOR, *V41_METHODS, ADAPTIVE_V43, GUARDED_V44)

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_SPACE = re.compile(r"\s+")
_FORBIDDEN_BLIND_KEYS = {
    "answer",
    "answer-node",
    "answer-text",
    "candidate_ceiling",
    "gold",
    "gold_candidate_ids",
    "gold_coordinates",
    "label",
    "source_mode",
    "supporting_facts",
    "table_id",
    "where",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob_sha1_bytes(value: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_archive_member(archive: tarfile.TarFile, member_path: str) -> bytes:
    member = archive.getmember(member_path)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"OTT-QA archive member is not a file: {member_path}")
    return handle.read()


def read_archive_json(path: Path, member_path: str) -> Any:
    with tarfile.open(path, mode="r:") as archive:
        return json.loads(_read_archive_member(archive, member_path).decode("utf-8"))


def archive_git_blob_sha1(path: Path, member_path: str) -> str:
    with tarfile.open(path, mode="r:") as archive:
        return _git_blob_sha1_bytes(_read_archive_member(archive, member_path))


def archive_directory_git_manifest(
    path: Path, repository_prefix: str
) -> dict[str, Any]:
    prefix = repository_prefix.rstrip("/") + "/"
    entries: list[str] = []
    total = 0
    with tarfile.open(path, mode="r:") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.startswith(prefix)
            ),
            key=lambda member: member.name,
        )
        for member in members:
            value = _read_archive_member(archive, member.name)
            entries.append(f"{member.name} {_git_blob_sha1_bytes(value)} {len(value)}")
            total += len(value)
    payload = "\n".join(entries).encode("utf-8")
    return {
        "files": len(entries),
        "bytes": total,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def directory_git_manifest(directory: Path, repository_prefix: str) -> dict[str, Any]:
    entries: list[str] = []
    total = 0
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        size = path.stat().st_size
        relative = path.relative_to(directory).as_posix()
        entries.append(
            f"{repository_prefix.rstrip('/')}/{relative} {git_blob_sha1(path)} {size}"
        )
        total += size
    payload = "\n".join(entries).encode("utf-8")
    return {
        "files": len(files),
        "bytes": total,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _normalise(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _match_text(value: Any) -> str:
    return " ".join(_WORD.findall(_normalise(value).lower()))


def _tokens(value: Any) -> list[str]:
    return _WORD.findall(_normalise(value).lower())


def _contains_forbidden_blind_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_BLIND_KEYS:
                return True
            if _contains_forbidden_blind_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_blind_key(item) for item in value)
    return False


def _answer_nodes(row: dict[str, Any]) -> list[list[Any]]:
    result: list[list[Any]] = []
    for value in row.get("answer-node", []):
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            continue
        coordinate = value[1]
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            continue
        try:
            row_index = int(coordinate[0])
            column_index = int(coordinate[1])
        except (TypeError, ValueError):
            continue
        kind = _normalise(value[-1]).lower()
        if kind not in SOURCE_MODES:
            continue
        result.append(
            [
                _normalise(value[0]),
                [row_index, column_index],
                None if value[2] is None else str(value[2]),
                kind,
            ]
        )
    return result


def _anchor_nodes(row: dict[str, Any]) -> list[tuple[int, int, int, str | None]]:
    result: list[tuple[int, int, int, str | None]] = []
    for family_index, family in enumerate(("tf-idf", "string-overlap", "links")):
        values = row.get(family, [])
        if not isinstance(values, list):
            continue
        for node_index, value in enumerate(values):
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                continue
            coordinate = value[1]
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                continue
            try:
                row_index = int(coordinate[0])
            except (TypeError, ValueError):
                continue
            url = None if value[2] is None else str(value[2])
            result.append((family_index, node_index, row_index, url))
    return result


def _eligible(row: dict[str, Any]) -> bool:
    required = ("question_id", "question", "table_id", "answer-text")
    if any(not _normalise(row.get(key)) for key in required):
        return False
    if not _answer_nodes(row):
        return False
    return bool(_anchor_nodes(row))


def _source_mode(row: dict[str, Any]) -> str:
    nodes = _answer_nodes(row)
    return str(nodes[0][-1]) if nodes else ""


def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
    question_id = str(row["question_id"])
    digest = hashlib.sha256((SAMPLE_SALT + question_id).encode("utf-8")).hexdigest()
    return digest, question_id


def select_sample(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    census = Counter()
    for raw in rows:
        row = dict(raw)
        census["source_rows"] += 1
        question_id = _normalise(row.get("question_id"))
        if not question_id:
            census["missing_question_id"] += 1
            continue
        if question_id in seen:
            census["duplicate_question_id"] += 1
            continue
        seen.add(question_id)
        if not _eligible(row):
            census["ineligible"] += 1
            continue
        mode = _source_mode(row)
        if mode not in SOURCE_MODES:
            census["invalid_source_mode"] += 1
            continue
        eligible[mode].append(row)

    for mode in SOURCE_MODES:
        eligible[mode].sort(key=_sample_key)
    selected_by_mode = {
        mode: list(eligible[mode][:TARGET_CASES_PER_MODE]) for mode in SOURCE_MODES
    }
    selected_ids = {
        str(row["question_id"])
        for values in selected_by_mode.values()
        for row in values
    }
    target_total = TARGET_CASES_PER_MODE * len(SOURCE_MODES)
    shortfall = target_total - sum(len(values) for values in selected_by_mode.values())
    if shortfall > 0:
        remaining = sorted(
            (
                row
                for mode in SOURCE_MODES
                for row in eligible[mode]
                if str(row["question_id"]) not in selected_ids
            ),
            key=_sample_key,
        )
        for row in remaining[:shortfall]:
            mode = _source_mode(row)
            selected_by_mode[mode].append(row)
            selected_ids.add(str(row["question_id"]))

    selected = [row for mode in SOURCE_MODES for row in selected_by_mode[mode]]
    selected.sort(key=_sample_key)
    summary = {
        "source_rows": census["source_rows"],
        "eligible_by_source_mode": {mode: len(eligible[mode]) for mode in SOURCE_MODES},
        "selected_by_source_mode": {
            mode: sum(_source_mode(row) == mode for row in selected)
            for mode in SOURCE_MODES
        },
        "selected_total": len(selected),
        "exclusions": {
            key: census[key]
            for key in sorted(census)
            if key != "source_rows" and census[key]
        },
        "selection_uses_gold_source_mode_only": True,
        "selection_exports_source_mode_to_blind_cache": False,
    }
    return selected, summary


def _surface(cell: Any) -> str:
    if isinstance(cell, (list, tuple)) and cell:
        return _normalise(cell[0])
    return _normalise(cell)


def _links(cell: Any) -> list[str]:
    if not isinstance(cell, (list, tuple)) or len(cell) < 2:
        return []
    value = cell[1]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if _normalise(item)]


def _table_prefix(table: dict[str, Any], table_id: str) -> str:
    values = [
        table.get("title"),
        table.get("section_title"),
        table.get("section"),
        table_id.replace("_", " "),
    ]
    return " | ".join(
        dict.fromkeys(_normalise(value) for value in values if _normalise(value))
    )


def _chunk_text(text: str, tokenizer: Any) -> list[dict[str, Any]]:
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not token_ids:
        return []
    result: list[dict[str, Any]] = []
    start = 0
    while start < len(token_ids):
        ids = token_ids[start : start + CHUNK_WINDOW]
        decoded = _normalise(tokenizer.decode(ids, skip_special_tokens=True))
        if decoded:
            result.append(
                {
                    "chunk_index": len(result),
                    "token_start": start,
                    "token_count": len(ids),
                    "text": decoded,
                }
            )
        if start + CHUNK_WINDOW >= len(token_ids):
            break
        start += CHUNK_STRIDE
    return result


def _bm25_scores(query: str, texts: Sequence[str]) -> list[float]:
    query_tokens = _tokens(query)
    documents = [_tokens(text) for text in texts]
    if not documents:
        return []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(len(document) for document in documents) / len(documents)
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        length = max(1, len(document))
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            df = document_frequency[token]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * length / max(1.0, average_length)
            )
            score += inverse * frequency * 2.2 / denominator
        scores.append(score)
    return scores


def _opaque(prefix: str, *values: str) -> str:
    payload = "|".join(values).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:20]


def _build_units(
    table: dict[str, Any],
    passages: dict[str, Any],
    table_id: str,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
    units: list[dict[str, Any]] = []
    row_units: dict[str, list[str]] = defaultdict(list)
    passage_units: dict[str, list[str]] = defaultdict(list)
    prefix = _table_prefix(table, table_id)
    headers = [_surface(cell) for cell in table.get("header", [])]
    linked_urls: set[str] = set()
    for row_index, row in enumerate(table.get("data", [])):
        cells = [_surface(cell) for cell in row]
        fields = [
            f"{headers[index]}: {value}"
            if index < len(headers) and headers[index]
            else value
            for index, value in enumerate(cells)
            if value
        ]
        text = _normalise(f"Table: {prefix}. Row: " + " | ".join(fields))
        source_key = f"table-row:{row_index}"
        for chunk in _chunk_text(text, tokenizer):
            canonical = f"{source_key}:chunk:{chunk['chunk_index']}"
            row_units[str(row_index)].append(canonical)
            units.append(
                {
                    "canonical_unit_id": canonical,
                    "canonical_source_id": source_key,
                    "source_kind": "table",
                    "row_index": row_index,
                    "passage_url": None,
                    **chunk,
                }
            )
        for cell in row:
            linked_urls.update(_links(cell))

    for url in sorted(linked_urls):
        passage = passages.get(url)
        if not isinstance(passage, str) or not _normalise(passage):
            continue
        title = url.replace("/wiki/", "").replace("_", " ")
        text = _normalise(f"Passage title: {title}. {passage}")
        source_key = f"passage:{url}"
        for chunk in _chunk_text(text, tokenizer):
            canonical = f"{source_key}:chunk:{chunk['chunk_index']}"
            passage_units[url].append(canonical)
            units.append(
                {
                    "canonical_unit_id": canonical,
                    "canonical_source_id": source_key,
                    "source_kind": "passage",
                    "row_index": None,
                    "passage_url": url,
                    **chunk,
                }
            )
    return units, row_units, passage_units


def prepare_blind_cases(
    linked_rows: Iterable[dict[str, Any]],
    table_directory: Path | None,
    passage_directory: Path | None,
    tokenizer: Any,
    *,
    source_archive: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, sampling = select_sample(linked_rows)
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    pool_counts: list[int] = []
    anchor_counts: list[int] = []
    all_unit_counts: list[int] = []
    missing_sources = Counter()

    archive_context = (
        tarfile.open(source_archive, mode="r:")
        if source_archive is not None
        else nullcontext(None)
    )
    with archive_context as archive:
        for row in selected:
            question_id = str(row["question_id"])
            table_id = str(row["table_id"])
            if archive is not None:
                try:
                    table = json.loads(
                        _read_archive_member(
                            archive, f"data/traindev_tables_tok/{table_id}.json"
                        ).decode("utf-8")
                    )
                    passages = json.loads(
                        _read_archive_member(
                            archive, f"data/traindev_request_tok/{table_id}.json"
                        ).decode("utf-8")
                    )
                except KeyError:
                    missing_sources["missing_table_or_passage_file"] += 1
                    continue
            else:
                if table_directory is None or passage_directory is None:
                    raise ValueError("OTT-QA directory source paths are required")
                table_path = table_directory / f"{table_id}.json"
                passage_path = passage_directory / f"{table_id}.json"
                if not table_path.is_file() or not passage_path.is_file():
                    missing_sources["missing_table_or_passage_file"] += 1
                    continue
                table = read_json(table_path)
                passages = read_json(passage_path)
            if not isinstance(table, dict) or not isinstance(passages, dict):
                missing_sources["invalid_table_or_passage_schema"] += 1
                continue
            all_units, row_units, passage_units = _build_units(
                table, passages, table_id, tokenizer
            )
            if not all_units:
                missing_sources["empty_atomic_unit_set"] += 1
                continue
            unit_by_id = {str(unit["canonical_unit_id"]): unit for unit in all_units}
            anchored: list[str] = []
            for _, _, row_index, url in _anchor_nodes(row):
                anchored.extend(row_units.get(str(row_index), []))
                if url is not None:
                    anchored.extend(passage_units.get(url, []))
            anchored = list(
                dict.fromkeys(value for value in anchored if value in unit_by_id)
            )
            scores = _bm25_scores(
                str(row["question"]), [str(unit["text"]) for unit in all_units]
            )
            lexical = [
                str(all_units[index]["canonical_unit_id"])
                for index in sorted(
                    range(len(all_units)),
                    key=lambda index: (
                        -float(scores[index]),
                        str(all_units[index]["canonical_unit_id"]),
                    ),
                )[:LEXICAL_POOL_SIZE]
            ]
            ordered = list(dict.fromkeys([*anchored, *lexical]))[:MAXIMUM_POOL_SIZE]
            case_id = _opaque("o", SAMPLE_SALT, question_id)
            official_rank = {value: index for index, value in enumerate(anchored)}
            candidates: list[dict[str, Any]] = []
            map_units: list[dict[str, Any]] = []
            for canonical in ordered:
                unit = unit_by_id[canonical]
                candidate_id = _opaque("u", case_id, canonical)
                source_id = _opaque("s", case_id, str(unit["canonical_source_id"]))
                candidates.append(
                    {
                        "id": candidate_id,
                        "source_id": source_id,
                        "text": str(unit["text"]),
                        "token_count": int(unit["token_count"]),
                        "official_rank": official_rank.get(canonical),
                    }
                )
                map_units.append(
                    {
                        "candidate_id": candidate_id,
                        "canonical_unit_id": canonical,
                        "canonical_source_id": str(unit["canonical_source_id"]),
                        "source_kind": str(unit["source_kind"]),
                        "row_index": unit["row_index"],
                        "passage_url": unit["passage_url"],
                        "chunk_index": int(unit["chunk_index"]),
                        "text": str(unit["text"]),
                        "text_sha256": hashlib.sha256(
                            str(unit["text"]).encode("utf-8")
                        ).hexdigest(),
                        "official_rank": official_rank.get(canonical),
                    }
                )
            blind = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "capability": CAPABILITY,
                "id": case_id,
                "query": str(row["question"]),
                "candidates": candidates,
                "gold_fields_visible_to_generator": False,
                "gold_fields_visible_to_scorer": False,
            }
            if _contains_forbidden_blind_key(blind):
                raise AssertionError("OTT-QA gold field leaked into blind cache")
            prepared.append(blind)
            maps.append(
                {
                    "schema_version": "frc-ottqa-v44-candidate-map-v1",
                    "id": case_id,
                    "question_id": question_id,
                    "table_id": table_id,
                    "units": map_units,
                }
            )
            pool_counts.append(len(candidates))
            anchor_counts.append(len(anchored))
            all_unit_counts.append(len(all_units))

    def distribution(values: Sequence[int]) -> dict[str, float | int]:
        if not values:
            return {"minimum": 0, "mean": 0.0, "maximum": 0}
        return {
            "minimum": min(values),
            "mean": round(float(np.mean(values)), 6),
            "maximum": max(values),
        }

    summary = {
        "sampling": sampling,
        "prepared_cases": len(prepared),
        "all_atomic_units": distribution(all_unit_counts),
        "official_anchor_units": distribution(anchor_counts),
        "retained_candidate_units": distribution(pool_counts),
        "missing_sources": dict(sorted(missing_sources.items())),
        "maximum_pool_size": MAXIMUM_POOL_SIZE,
        "gold_fields_exported_to_blind_cache": False,
    }
    return prepared, maps, summary


class FrozenOttqaScorer(FrozenDynamicHoVerScorer):
    """v41 neural scorer with official anchor ranks restored after scoring."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
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


def _distinct_role_argmax(candidates: Sequence[dict[str, Any]]) -> int:
    if not candidates:
        return 0
    identifiers = {
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
    return len(identifiers)


def adaptive_v43_target_cardinality(candidates: Sequence[dict[str, Any]]) -> int:
    distinct = _distinct_role_argmax(candidates)
    return min(TOP_K, max(1, distinct)) if candidates else 0


def guarded_target_cardinality(candidates: Sequence[dict[str, Any]]) -> int:
    distinct = _distinct_role_argmax(candidates)
    return min(TOP_K, max(3, distinct + 1)) if candidates else 0


def _rank_utilities(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result = {str(candidate["id"]): {} for candidate in candidates}
    for role in DYNAMIC_ROLES:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate["dynamic_role_scores"][role]),
                str(candidate["id"]),
            ),
        )
        for rank, candidate in enumerate(ordered):
            result[str(candidate["id"])][role] = 1.0 / math.log2(2 + rank)
    return result


def _adaptive_select(
    candidates: list[dict[str, Any]], *, token_budget: int, target: int
) -> list[dict[str, Any]]:
    utilities = _rank_utilities(candidates)
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    best_by_role = {role: 0.0 for role in DYNAMIC_ROLES}
    selected_sources: set[str] = set()
    total = 0
    while remaining and len(selected) < target:
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in remaining:
            cost = int(candidate["token_count"])
            source_id = str(candidate["source_id"])
            if source_id in selected_sources or total + cost > token_budget:
                continue
            role_gain = sum(
                max(0.0, utilities[str(candidate["id"])][role] - best_by_role[role])
                for role in DYNAMIC_ROLES
            )
            gain = (
                float(candidate["scores"]["cross_encoder"])
                + RANK_COVERAGE_WEIGHT * role_gain
            )
            eligible.append((gain, str(candidate["id"]), candidate))
        if not eligible:
            break
        eligible.sort(key=lambda value: (-value[0], value[1]))
        best = eligible[0][2]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        selected_sources.add(str(best["source_id"]))
        for role in DYNAMIC_ROLES:
            best_by_role[role] = max(
                best_by_role[role], utilities[str(best["id"])][role]
            )
    return selected


def select_v44(
    candidates: list[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    if method == OFFICIAL_ANCHOR:
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
    if method == ADAPTIVE_V43:
        return _adaptive_select(
            candidates,
            token_budget=token_budget,
            target=adaptive_v43_target_cardinality(candidates),
        )
    if method == GUARDED_V44:
        return _adaptive_select(
            candidates,
            token_budget=token_budget,
            target=guarded_target_cardinality(candidates),
        )
    return select_candidates(candidates, method, token_budget=token_budget)


def build_gold_rows(
    linked_rows: Iterable[dict[str, Any]],
    candidate_maps: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    linked_by_id = {str(row.get("question_id")): dict(row) for row in linked_rows}
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        question_id = str(candidate_map["question_id"])
        row = linked_by_id.get(question_id)
        if row is None:
            raise ValueError(f"OTT-QA gold row missing for {question_id}")
        answer_text = _match_text(row.get("answer-text"))
        nodes = _answer_nodes(row)
        mapped: set[str] = set()
        represented_nodes = 0
        for node in nodes:
            surface, coordinate, url, kind = node
            expected = _match_text(surface if kind == "table" else answer_text)
            node_ids: set[str] = set()
            for unit in candidate_map.get("units", []):
                if kind == "table":
                    same_source = unit.get("source_kind") == "table" and int(
                        unit.get("row_index", -1)
                    ) == int(coordinate[0])
                else:
                    same_source = unit.get("source_kind") == "passage" and str(
                        unit.get("passage_url")
                    ) == str(url)
                if (
                    same_source
                    and expected
                    and expected in _match_text(unit.get("text"))
                ):
                    node_ids.add(str(unit["candidate_id"]))
            if node_ids:
                represented_nodes += 1
                mapped.update(node_ids)
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "question_id": question_id,
                "source_mode": _source_mode(row),
                "answer_node_count": len(nodes),
                "represented_answer_node_count": represented_nodes,
                "gold_candidate_ids": sorted(mapped),
                "candidate_ceiling_complete": bool(mapped),
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], structural_census: dict[str, Any]
) -> dict[str, Any]:
    counts = Counter(str(row["source_mode"]) for row in gold_rows)
    complete = [bool(row["candidate_ceiling_complete"]) for row in gold_rows]
    by_mode: dict[str, Any] = {}
    for mode in SOURCE_MODES:
        subset = [row for row in gold_rows if row["source_mode"] == mode]
        by_mode[mode] = {
            "cases": len(subset),
            "candidate_ceiling_complete_rate": round(
                float(
                    np.mean([bool(row["candidate_ceiling_complete"]) for row in subset])
                ),
                6,
            )
            if subset
            else 0.0,
        }
    checks = {
        "minimum_total_cases_met": len(gold_rows) >= MINIMUM_TOTAL_CASES,
        "minimum_cases_per_source_mode_met": all(
            counts[mode] >= MINIMUM_CASES_PER_MODE for mode in SOURCE_MODES
        ),
        "candidate_ceiling_complete_rate_at_least_0_60": (
            float(np.mean(complete)) >= 0.60 if complete else False
        ),
    }
    return {
        "schema_version": "frc-ottqa-v44-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "candidate_ceiling_complete_rate": round(float(np.mean(complete)), 6)
        if complete
        else 0.0,
        "by_source_mode": by_mode,
        "structural_census_sha256": canonical_json_sha256(structural_census),
        "checks": checks,
        "candidate_method_or_threshold_changed": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }


def _selection_metrics(
    selected: Sequence[dict[str, Any]], gold: dict[str, Any]
) -> dict[str, float | int]:
    selected_ids = {str(candidate["id"]) for candidate in selected}
    gold_ids = {str(value) for value in gold["gold_candidate_ids"]}
    hits = len(selected_ids & gold_ids)
    precision = hits / len(selected_ids) if selected_ids else 1.0
    recall = float(bool(selected_ids & gold_ids))
    return {
        "precision": precision,
        "answer_evidence_recall": recall,
        "selected_unit_count": len(selected),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "gold_unit_hits": hits,
    }


def _aggregate(
    rows: Sequence[dict[str, Any]], method: str, budgets: Sequence[int] = BUDGETS
) -> dict[str, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    counts: list[float] = []
    costs: list[float] = []
    f1_by_budget: list[float] = []
    for budget in budgets:
        metrics = [
            row["configurations"][str(budget)]["methods"][method]["metrics"]
            for row in rows
        ]
        precision = float(np.mean([float(item["precision"]) for item in metrics]))
        recall = float(
            np.mean([float(item["answer_evidence_recall"]) for item in metrics])
        )
        f1_by_budget.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precisions.extend(float(item["precision"]) for item in metrics)
        recalls.extend(float(item["answer_evidence_recall"]) for item in metrics)
        counts.extend(float(item["selected_unit_count"]) for item in metrics)
        costs.extend(float(item["selected_token_cost"]) for item in metrics)
    return {
        "answer_evidence_macro_f1": float(np.mean(f1_by_budget)),
        "macro_precision": float(np.mean(precisions)),
        "answer_evidence_recall": float(np.mean(recalls)),
        "mean_selected_unit_count": float(np.mean(counts)),
        "mean_selected_token_cost": float(np.mean(costs)),
    }


def _comparison(point: float, values: np.ndarray) -> dict[str, float]:
    return {
        "point": round(float(point), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def _bootstrap(rows: Sequence[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    guarded_dynamic = np.empty(resamples, dtype=float)
    guarded_strongest = np.empty(resamples, dtype=float)
    guarded_v43 = np.empty(resamples, dtype=float)
    for index in range(resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(item)] for item in indices]
        guarded = _aggregate(sample, GUARDED_V44)["answer_evidence_macro_f1"]
        dynamic = _aggregate(sample, DYNAMIC_RANK)["answer_evidence_macro_f1"]
        strongest = max(
            _aggregate(sample, method)["answer_evidence_macro_f1"]
            for method in NON_FRC_BASELINES
        )
        replay = _aggregate(sample, ADAPTIVE_V43)["answer_evidence_macro_f1"]
        guarded_dynamic[index] = guarded - dynamic
        guarded_strongest[index] = guarded - strongest
        guarded_v43[index] = guarded - replay
    guarded = _aggregate(rows, GUARDED_V44)["answer_evidence_macro_f1"]
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "guarded_minus_dynamic": _comparison(
            guarded - _aggregate(rows, DYNAMIC_RANK)["answer_evidence_macro_f1"],
            guarded_dynamic,
        ),
        "guarded_minus_bootstrap_strongest_non_frc": _comparison(
            guarded
            - max(
                _aggregate(rows, method)["answer_evidence_macro_f1"]
                for method in NON_FRC_BASELINES
            ),
            guarded_strongest,
        ),
        "guarded_minus_v43": _comparison(
            guarded - _aggregate(rows, ADAPTIVE_V43)["answer_evidence_macro_f1"],
            guarded_v43,
        ),
    }


def evaluate_ottqa(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if [str(row["case_id"]) for row in gold_rows] != [
        str(row["id"]) for row in scored_rows
    ]:
        raise ValueError("OTT-QA gold and score rows are incomplete or misordered")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        candidates = _merge_candidates(dict(scored))
        configurations: dict[str, Any] = {}
        v43_target = adaptive_v43_target_cardinality(candidates)
        v44_target = guarded_target_cardinality(candidates)
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v44(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(selected, dict(gold)),
                }
                if method == ADAPTIVE_V43:
                    methods[method]["target_cardinality"] = v43_target
                if method == GUARDED_V44:
                    methods[method]["target_cardinality"] = v44_target
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "question_id_sha256": hashlib.sha256(
                    str(gold["question_id"]).encode("utf-8")
                ).hexdigest(),
                "source_mode": str(gold["source_mode"]),
                "answer_node_count": int(gold["answer_node_count"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "candidate_unit_count": len(candidates),
                "v43_target_cardinality": v43_target,
                "v44_target_cardinality": v44_target,
                "configurations": configurations,
            }
        )

    aggregates = {
        method: {
            key: round(value, 6) for key, value in _aggregate(evidence, method).items()
        }
        for method in METHODS
    }
    strongest_non_frc = max(
        NON_FRC_BASELINES,
        key=lambda method: float(aggregates[method]["answer_evidence_macro_f1"]),
    )
    bootstrap = _bootstrap(evidence, resamples=resamples)
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: _aggregate(evidence, method, (budget,))[
                "answer_evidence_macro_f1"
            ],
        )
        budget_deltas[str(budget)] = round(
            _aggregate(evidence, GUARDED_V44, (budget,))["answer_evidence_macro_f1"]
            - _aggregate(evidence, strongest, (budget,))["answer_evidence_macro_f1"],
            6,
        )
    source_deltas: dict[str, Any] = {}
    for mode in SOURCE_MODES:
        subset = [row for row in evidence if row["source_mode"] == mode]
        if len(subset) < 40:
            continue
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: _aggregate(subset, method)["answer_evidence_macro_f1"],
        )
        source_deltas[mode] = {
            "cases": len(subset),
            "strongest_non_frc": strongest,
            "delta": round(
                _aggregate(subset, GUARDED_V44)["answer_evidence_macro_f1"]
                - _aggregate(subset, strongest)["answer_evidence_macro_f1"],
                6,
            ),
        }
    source_counts = Counter(str(row["source_mode"]) for row in evidence)
    ceiling_rate = float(
        np.mean([bool(row["candidate_ceiling_complete"]) for row in evidence])
    )
    count_reduction = float(
        aggregates[DYNAMIC_RANK]["mean_selected_unit_count"]
    ) - float(aggregates[GUARDED_V44]["mean_selected_unit_count"])
    recall_drop = float(aggregates[DYNAMIC_RANK]["answer_evidence_recall"]) - float(
        aggregates[GUARDED_V44]["answer_evidence_recall"]
    )
    recall_gain_v43 = float(aggregates[GUARDED_V44]["answer_evidence_recall"]) - float(
        aggregates[ADAPTIVE_V43]["answer_evidence_recall"]
    )
    safety = [
        *budget_deltas.values(),
        *(float(value["delta"]) for value in source_deltas.values()),
    ]
    guarded_dynamic = bootstrap["guarded_minus_dynamic"]
    guarded_strongest = bootstrap["guarded_minus_bootstrap_strongest_non_frc"]
    guarded_v43 = bootstrap["guarded_minus_v43"]
    checks = {
        "minimum_total_cases_met": len(evidence) >= MINIMUM_TOTAL_CASES,
        "minimum_cases_per_source_mode_met": all(
            source_counts[mode] >= MINIMUM_CASES_PER_MODE for mode in SOURCE_MODES
        ),
        "candidate_ceiling_complete_rate_at_least_0_60": ceiling_rate >= 0.60,
        "query_parser_fallback_rate_at_most_0_05": float(query_summary["fallback_rate"])
        <= 0.05,
        "guarded_minus_dynamic_point_at_least_0_005": float(guarded_dynamic["point"])
        >= 0.005,
        "guarded_minus_dynamic_ci_low_above_0": float(guarded_dynamic["ci_low"]) > 0.0,
        "guarded_minus_strongest_point_at_least_0_01": float(guarded_strongest["point"])
        >= 0.01,
        "guarded_minus_strongest_ci_low_above_0": float(guarded_strongest["ci_low"])
        > 0.0,
        "every_budget_and_supported_source_mode_delta_at_least_minus_0_02": min(safety)
        >= -0.02
        if safety
        else False,
        "mean_selected_unit_reduction_at_least_0_50": count_reduction >= 0.50,
        "answer_evidence_recall_drop_at_most_0_02": recall_drop <= 0.02,
        "guarded_minus_v43_recall_at_least_0_02": recall_gain_v43 >= 0.02,
        "guarded_minus_v43_f1_at_least_minus_0_005": float(guarded_v43["point"])
        >= -0.005,
    }
    informative = (
        checks["minimum_total_cases_met"]
        and checks["minimum_cases_per_source_mode_met"]
        and checks["candidate_ceiling_complete_rate_at_least_0_60"]
    )
    if not informative:
        status = "OTTQA_ORACLE_TABLE_BOUNDED_POOL_INCONCLUSIVE"
    elif all(checks.values()):
        status = (
            "OTTQA_GUARDED_ADAPTIVE_SUPPORT_ESTABLISHED_ON_ORACLE_TABLE_BOUNDED_POOL"
        )
    else:
        status = "OTTQA_GUARDED_ADAPTIVE_SUPPORT_NOT_ESTABLISHED"
    targets = Counter(int(row["v44_target_cardinality"]) for row in evidence)
    report = {
        "schema_version": "frc-ottqa-guarded-adaptive-report-v1",
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
            "oracle_table_bounded_pool": True,
        },
        "analysis": {
            "aggregates": aggregates,
            "strongest_non_frc": strongest_non_frc,
            "family_comparison": bootstrap,
            "budget_deltas": budget_deltas,
            "supported_source_mode_deltas": source_deltas,
            "candidate_ceiling_complete_rate": round(ceiling_rate, 6),
            "mean_selected_unit_reduction_vs_dynamic": round(count_reduction, 6),
            "answer_evidence_recall_drop_vs_dynamic": round(recall_drop, 6),
            "answer_evidence_recall_gain_vs_v43": round(recall_gain_v43, 6),
            "target_cardinality_distribution": {
                str(key): targets[key] for key in sorted(targets)
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
            "The official table_id is an oracle boundary; this experiment does not evaluate open-domain table retrieval.",
            "Answer-node traces cover answer-bearing cells or passages, not every intermediate multi-hop fact.",
            "The unavailable official full-corpus S3 artifacts prevent an official OTT-QA leaderboard reproduction.",
            "The result cannot establish real SetR reproduction, flood-domain validity, or production readiness.",
        ],
    }
    return report, evidence


def write_report(
    report: dict[str, Any], evidence: Sequence[dict[str, Any]], output_directory: Path
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "ottqa_guarded_adaptive_atomic_roles.json"
    markdown_path = output_directory / "ottqa_guarded_adaptive_atomic_roles.md"
    cases_path = output_directory / "ottqa_guarded_adaptive_atomic_roles_cases.jsonl.gz"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = report["analysis"]
    comparison = analysis["family_comparison"]
    lines = [
        "# OTT-QA 有保护自适应证据基数实验（v44）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 可评测样本：{report['metadata']['cases']}",
        f"- 候选池答案证据上限率：{analysis['candidate_ceiling_complete_rate']:.6f}",
        f"- 最强非 FRC 基线：`{analysis['strongest_non_frc']}`",
        f"- v44 - v41 动态：{comparison['guarded_minus_dynamic']['point']:+.6f}，95% CI [{comparison['guarded_minus_dynamic']['ci_low']:+.6f}, {comparison['guarded_minus_dynamic']['ci_high']:+.6f}]",
        f"- v44 - 最强非 FRC：{comparison['guarded_minus_bootstrap_strongest_non_frc']['point']:+.6f}，95% CI [{comparison['guarded_minus_bootstrap_strongest_non_frc']['ci_low']:+.6f}, {comparison['guarded_minus_bootstrap_strongest_non_frc']['ci_high']:+.6f}]",
        f"- v44 - v43：{comparison['guarded_minus_v43']['point']:+.6f}，95% CI [{comparison['guarded_minus_v43']['ci_low']:+.6f}, {comparison['guarded_minus_v43']['ci_high']:+.6f}]",
        f"- 相对动态方法平均少选证据单元：{analysis['mean_selected_unit_reduction_vs_dynamic']:.6f}",
        f"- 相对动态方法答案证据召回下降：{analysis['answer_evidence_recall_drop_vs_dynamic']:+.6f}",
        "- Gate 2：`NO-GO/SHADOW`；该有界候选池实验不授权 CANARY/DEFAULT。",
        "",
        "## 方法汇总",
        "",
        "| 方法 | Evidence F1 | Precision | Answer recall | 平均单元数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = analysis["aggregates"][method]
        lines.append(
            f"| `{method}` | {value['answer_evidence_macro_f1']:.6f} | {value['macro_precision']:.6f} | {value['answer_evidence_recall']:.6f} | {value['mean_selected_unit_count']:.6f} |"
        )
    lines.extend(["", "## 支持检查", ""])
    for name, passed in analysis["support_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "该结果只比较官方 oracle table 及其 linked passages 内的答案承载证据选择，不是 OTT-QA 官方榜单、完整开放域检索或完整多跳证据链评测。",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    with cases_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in evidence:
                handle.write(
                    (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                )
    return {"json": json_path, "markdown": markdown_path, "cases": cases_path}


def validate_protocol(
    path: Path,
    *,
    erratum_path: Path | None = None,
    erratum2_path: Path | None = None,
    erratum3_path: Path | None = None,
) -> dict[str, Any]:
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("OTT-QA v44 protocol hash mismatch")
    value = read_json(path)
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("OTT-QA v44 experiment id mismatch")
    if value.get("methods", {}).get("candidate_method") != GUARDED_V44:
        raise ValueError("OTT-QA v44 candidate method mismatch")
    if value.get("methods", {}).get("token_budgets") != list(BUDGETS):
        raise ValueError("OTT-QA v44 budget registration mismatch")
    if erratum_path is None:
        return value
    if sha256(erratum_path) != PROTOCOL_ERRATUM_SHA256:
        raise ValueError("OTT-QA v44 protocol erratum hash mismatch")
    erratum = read_json(erratum_path)
    if erratum.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("OTT-QA v44 protocol erratum experiment mismatch")
    if erratum.get("base_protocol_sha256") != sha256(path):
        raise ValueError("OTT-QA v44 protocol erratum base mismatch")
    correction = erratum.get("correction", {})
    if correction.get("scientific_effect") not in (
        None,
        "none; storage portability only",
    ):
        raise ValueError("OTT-QA v44 protocol erratum changed scientific scope")
    if erratum.get("method_or_threshold_changed") is not False:
        raise ValueError("OTT-QA v44 protocol erratum changed the method")
    for key in (
        "selection_rule_changed",
        "candidate_construction_changed",
        "candidate_pool_size_changed",
        "model_or_revision_changed",
        "method_or_formula_changed",
        "budget_or_threshold_changed",
        "metric_or_decision_rule_changed",
    ):
        if correction.get(key) is not False:
            raise ValueError(f"OTT-QA v44 protocol erratum changed {key}")
    if erratum2_path is None or erratum3_path is None:
        return {"base_protocol": value, "effective_erratum": erratum}
    if sha256(erratum2_path) != PROTOCOL_ERRATUM2_SHA256:
        raise ValueError("OTT-QA v44 protocol erratum2 hash mismatch")
    erratum2 = read_json(erratum2_path)
    if erratum2.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("OTT-QA v44 protocol erratum2 experiment mismatch")
    if erratum2.get("base_protocol_sha256") != sha256(path):
        raise ValueError("OTT-QA v44 protocol erratum2 base mismatch")
    if erratum2.get("prior_protocol_erratum_sha256") != sha256(erratum_path):
        raise ValueError("OTT-QA v44 protocol erratum2 chain mismatch")
    if erratum2.get("method_or_threshold_changed") is not False:
        raise ValueError("OTT-QA v44 protocol erratum2 changed the method")
    if sha256(erratum3_path) != PROTOCOL_ERRATUM3_SHA256:
        raise ValueError("OTT-QA v44 protocol erratum3 hash mismatch")
    erratum3 = read_json(erratum3_path)
    if erratum3.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("OTT-QA v44 protocol erratum3 experiment mismatch")
    if erratum3.get("base_protocol_sha256") != sha256(path):
        raise ValueError("OTT-QA v44 protocol erratum3 base mismatch")
    if erratum3.get("prior_protocol_erratum_sha256") != sha256(erratum_path):
        raise ValueError("OTT-QA v44 protocol erratum3 first-chain mismatch")
    if erratum3.get("prior_protocol_erratum2_sha256") != sha256(erratum2_path):
        raise ValueError("OTT-QA v44 protocol erratum3 second-chain mismatch")
    if erratum3.get("method_or_threshold_changed") is not False:
        raise ValueError("OTT-QA v44 protocol erratum3 changed the method")
    return {
        "base_protocol": value,
        "storage_errata": [erratum, erratum2, erratum3],
    }


def validate_implementation_registration(
    path: Path,
    *,
    protocol_path: Path,
    protocol_erratum_path: Path,
    protocol_erratum2_path: Path,
    protocol_erratum3_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
    inherited_module_path: Path,
    erratum_path: Path,
    erratum2_path: Path,
) -> dict[str, Any]:
    validate_protocol(
        protocol_path,
        erratum_path=protocol_erratum_path,
        erratum2_path=protocol_erratum2_path,
        erratum3_path=protocol_erratum3_path,
    )
    value = read_json(path)
    if (
        sha256(path)
        != "797aeb50d6fabaf931ba41325c9f9723d44bc6b5b3e66aca04cb5b73aba3df87"
    ):
        raise ValueError("OTT-QA v44 base implementation registration hash mismatch")
    if value.get("hashes", {}).get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("OTT-QA v44 base implementation protocol mismatch")
    if (
        sha256(erratum_path)
        != "282622aece3610da4e98868c954c7203fbf42a4df5ce75593fe6589e83b38aa4"
    ):
        raise ValueError("OTT-QA v44 first implementation erratum hash mismatch")
    first_erratum = read_json(erratum_path)
    if first_erratum.get("base_implementation_registration_sha256") != sha256(path):
        raise ValueError("OTT-QA v44 first implementation erratum base mismatch")
    if first_erratum.get("method_or_threshold_changed") is not False:
        raise ValueError("OTT-QA v44 first implementation erratum changed method")
    expected = {
        "protocol_sha256": sha256(protocol_path),
        "protocol_erratum_sha256": sha256(protocol_erratum_path),
        "protocol_erratum2_sha256": sha256(protocol_erratum2_path),
        "protocol_erratum3_sha256": sha256(protocol_erratum3_path),
        "module_sha256": sha256(module_path),
        "runner_sha256": sha256(runner_path),
        "test_sha256": sha256(test_path),
        "inherited_v41_module_sha256": sha256(inherited_module_path),
    }
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("OTT-QA v44 implementation experiment mismatch")
    if value.get("data_content_accessed_before_registration") is not False:
        raise ValueError("OTT-QA v44 implementation blindness was not registered")
    erratum2 = read_json(erratum2_path)
    if erratum2.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("OTT-QA v44 implementation erratum2 experiment mismatch")
    if erratum2.get("prior_implementation_erratum_sha256") != sha256(erratum_path):
        raise ValueError("OTT-QA v44 implementation erratum2 chain mismatch")
    if erratum2.get("hashes") != expected:
        raise ValueError("OTT-QA v44 effective implementation hash mismatch")
    if erratum2.get("method_or_threshold_changed") is not False:
        raise ValueError("OTT-QA v44 implementation erratum2 changed the method")
    return {
        "base_registration": value,
        "implementation_errata": [first_erratum, erratum2],
    }


def validate_execution_registration(
    path: Path,
    *,
    implementation_path: Path,
    implementation_erratum_path: Path,
    implementation_erratum2_path: Path,
    protocol_path: Path,
    protocol_erratum_path: Path,
    protocol_erratum2_path: Path,
    protocol_erratum3_path: Path,
    source_archive_path: Path,
    prepared_path: Path,
    candidate_map_path: Path,
    census_path: Path,
    coverage_path: Path,
    runtime_parameters: dict[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    expected = {
        "implementation_registration_sha256": sha256(implementation_path),
        "implementation_erratum_sha256": sha256(implementation_erratum_path),
        "implementation_erratum2_sha256": sha256(implementation_erratum2_path),
        "protocol_sha256": sha256(protocol_path),
        "protocol_erratum_sha256": sha256(protocol_erratum_path),
        "protocol_erratum2_sha256": sha256(protocol_erratum2_path),
        "protocol_erratum3_sha256": sha256(protocol_erratum3_path),
        "source_archive_sha256": sha256(source_archive_path),
        "prepared_blind_sha256": sha256(prepared_path),
        "candidate_map_sha256": sha256(candidate_map_path),
        "blind_census_sha256": sha256(census_path),
        "candidate_coverage_sha256": sha256(coverage_path),
    }
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("OTT-QA v44 execution experiment mismatch")
    if value.get("hashes") != expected:
        raise ValueError("OTT-QA v44 execution registration mismatch")
    if value.get("runtime_parameters") != runtime_parameters:
        raise ValueError("OTT-QA v44 runtime registration mismatch")
    sources = value.get("source_representation", {})
    if (
        Path(str(sources.get("archive_path", ""))).resolve()
        != source_archive_path.resolve()
    ):
        raise ValueError("OTT-QA v44 source-archive registration mismatch")
    coverage = read_json(coverage_path)
    if coverage.get("candidate_method_or_threshold_changed") is not False:
        raise ValueError("OTT-QA v44 coverage changed the candidate method")
    return value


__all__ = [
    "ADAPTIVE_V43",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EXPERIMENT_ID",
    "GUARDED_V44",
    "MAXIMUM_POOL_SIZE",
    "FrozenOttqaScorer",
    "adaptive_v43_target_cardinality",
    "archive_directory_git_manifest",
    "archive_git_blob_sha1",
    "build_candidate_coverage",
    "build_gold_rows",
    "evaluate_ottqa",
    "directory_git_manifest",
    "git_blob_sha1",
    "guarded_target_cardinality",
    "prepare_blind_cases",
    "read_archive_json",
    "select_sample",
    "select_v44",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "write_report",
]
