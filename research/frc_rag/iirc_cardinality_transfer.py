"""Prospective IIRC evidence-cardinality transfer experiment (v86)."""

from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote

import numpy as np

from research.frc_rag.iirc_cardinality_transfer_router import (
    load_model_artifact,
    rank_evidence_sources,
    select_method as select_adaptive_method,
)
from research.frc_rag.twowiki_confirmation import _bm25
from research.frc_rag.twowiki_support_path_closure import (
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-IIRC-CARDINALITY-TRANSFER-PROSPECTIVE-V86"
SCHEMA_VERSION = "frc-iirc-cardinality-transfer-v86"
PARTITION_SALT = "FRC-IIRC-V86-TRAIN-PARTITION|"
STAGES = ("development", "confirmation")
STAGE_CASES = {"development": 400, "confirmation": 800}
STAGE_OFFSETS = {"development": 0, "confirmation": 400}
CONTENT_TOKENS = 384
OVERLAP_TOKENS = 64
STRIDE_TOKENS = CONTENT_TOKENS - OVERLAP_TOKENS
MAX_SOURCES = 4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"development": 20260806, "confirmation": 20260807}

SOURCE_HASHES = {
    "train": "00a88b97dbbda56c32c620254eb7f0ac0a41507152c2de21f21164052ffcdaa5",
    "dev": "49a109c43c902e52f9a8b150f6b9d383261c308384f614ccd2b9990735cb12b1",
    "train_dev_archive": "adfc34c8180337467105b2f534410c34e4fe43a82e6da03922440387802ca441",
    "context_archive": "d239390a27ed3f4aa868afe126187e6d108990a7bcc2a3efb6cae5de917264c7",
    "context_json": "d1b848328f4cef1dbb46f7fbf02eb980f3ad97df67199464f6f55f883bb08bae",
}
SOURCE_SIZES = {
    "train": 25_735_742,
    "dev": 2_679_112,
    "train_dev_archive": 5_713_428,
    "context_archive": 385_263_479,
    "context_json": 1_151_197_025,
}
SOURCE_COUNTS = {"train_articles": 4754, "train_questions": 10839, "context_articles": 56550}
PARTITION_COMMITMENTS = {
    "all": "87e23bb192797d8dfe6a7e51d8c921efb1775ceeb4e46ef72df59ee1c7d49fc5",
    "development": "e5cf8c245933581c2386c4aefbcff439c4429a66bf679509f2310346fd2224a0",
    "confirmation": "ca5f157b14ba0dd035fb5efbfa9c6c44d25d0878ce41cf498648d69b434e5c70",
    "remaining": "d9e86cd22b2ffe43a48357ac4b1a226e026fb4ab742719b9f1743da6ee15d7f4",
}

BM25_FIXED2 = "bm25_fixed2"
DENSE_FIXED2 = "dense_fixed2"
HYBRID_FIXED2 = "hybrid_fixed2"
CROSS_FIXED = tuple(f"cross_encoder_fixed{value}" for value in range(1, 5))
FRC_FIXED = tuple(f"frc_fixed{value}" for value in range(1, 5))
CANDIDATE_METHOD = "cross_dataset_question_cardinality_frc_v86"
CONTROL_METHODS = (BM25_FIXED2, DENSE_FIXED2, HYBRID_FIXED2, *CROSS_FIXED, *FRC_FIXED)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
ROLE_NAMES = ("answer", "attribution", "condition", "exception", "procedure")

STRICT_GATES = {
    "invalid_selector_output_rate_at_most": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.35,
    "candidate_complete_evidence_recall_at_least": 0.30,
    "candidate_minus_strongest_fixed_f1_at_least": 0.005,
    "candidate_minus_strongest_fixed_ci_low_above": 0.0,
    "candidate_minus_frc_fixed2_f1_at_least": 0.005,
    "candidate_minus_frc_fixed2_ci_low_above": 0.0,
    "candidate_complete_recall_delta_vs_strongest_at_least": -0.02,
    "candidate_evidence_recall_delta_vs_strongest_at_least": -0.02,
    "candidate_mean_selected_sources_above_strongest_at_most": 0.5,
    "distinct_candidate_actions_at_least": 2,
    "second_largest_action_fraction_at_least": 0.05,
    "supported_stratum_delta_vs_strongest_at_least": -0.03,
    "supported_stratum_min_cases": 40,
}

_WHITESPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")


def normalize_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", unquote(value).replace("_", " ")).strip().lower()


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", html.unescape(_HTML_TAG.sub(" ", value))).strip()


def source_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"{kind}|{value}".encode("utf-8")).hexdigest()
    return f"s{digest[:16]}"


def validate_sources(
    *,
    train_path: Path,
    dev_path: Path,
    train_dev_archive: Path,
    context_archive: Path,
    context_json: Path,
) -> None:
    paths = {
        "train": train_path,
        "dev": dev_path,
        "train_dev_archive": train_dev_archive,
        "context_archive": context_archive,
        "context_json": context_json,
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size != SOURCE_SIZES[name]:
            raise ValueError(f"v86 source size mismatch: {name}")
        if sha256(path) != SOURCE_HASHES[name]:
            raise ValueError(f"v86 source hash mismatch: {name}")


def _read_iirc(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"v86 expected IIRC article array: {path}")
    return value


def build_partitions(train_rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    ids = [
        str(question.get("qid") or "").strip()
        for article in train_rows
        for question in article.get("questions", [])
    ]
    if (
        len(train_rows) != SOURCE_COUNTS["train_articles"]
        or len(ids) != SOURCE_COUNTS["train_questions"]
        or any(not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("v86 IIRC train IDs or counts changed")
    ordered = sorted(
        ids,
        key=lambda value: (
            hashlib.sha256(f"{PARTITION_SALT}{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    partitions = {
        "all": sorted(ids),
        "development": ordered[: STAGE_CASES["development"]],
        "confirmation": ordered[
            STAGE_OFFSETS["confirmation"] : STAGE_OFFSETS["confirmation"]
            + STAGE_CASES["confirmation"]
        ],
        "remaining": ordered[sum(STAGE_CASES.values()) :],
    }
    for name, values in partitions.items():
        digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
        if digest != PARTITION_COMMITMENTS[name]:
            raise ValueError(f"v86 partition commitment changed: {name}")
    if set(partitions["development"]) & set(partitions["confirmation"]):
        raise ValueError("v86 development and confirmation overlap")
    return partitions


def build_context_offset_index(source_path: Path, index_path: Path) -> dict[str, Any]:
    """Build a small title-to-byte-range SQLite index without duplicating article text."""

    if sha256(source_path) != SOURCE_HASHES["context_json"]:
        raise ValueError("v86 context source changed before indexing")
    if index_path.exists():
        with closing(sqlite3.connect(index_path)) as connection:
            meta = dict(connection.execute("SELECT key, value FROM metadata"))
            count = int(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
            quick = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if (
            meta.get("source_sha256") == SOURCE_HASHES["context_json"]
            and int(meta.get("source_size", 0)) == SOURCE_SIZES["context_json"]
            and count == SOURCE_COUNTS["context_articles"]
            and quick == "ok"
        ):
            return {"articles": count, "quick_check": quick, "reused": True}
        raise ValueError("v86 existing context offset index is stale")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    decoder = json.JSONDecoder()
    with closing(sqlite3.connect(temporary)) as connection:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute(
            "CREATE TABLE articles (title TEXT PRIMARY KEY, offset INTEGER NOT NULL, length INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        batch: list[tuple[str, int, int]] = []
        with source_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                stripped = raw.lstrip()
                if not stripped.startswith(b'"'):
                    continue
                decoded = stripped.decode("utf-8")
                title, _ = decoder.raw_decode(decoded)
                if not isinstance(title, str) or normalize_title(title) != title:
                    raise ValueError("v86 context title normalization changed")
                batch.append((title, offset, len(raw)))
                if len(batch) >= 1000:
                    connection.executemany(
                        "INSERT INTO articles(title, offset, length) VALUES (?, ?, ?)",
                        batch,
                    )
                    batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO articles(title, offset, length) VALUES (?, ?, ?)", batch
            )
        count = int(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
        if count != SOURCE_COUNTS["context_articles"]:
            raise ValueError(f"v86 context article count changed: {count}")
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("source_sha256", SOURCE_HASHES["context_json"]),
                ("source_size", str(SOURCE_SIZES["context_json"])),
                ("article_count", str(count)),
            ),
        )
        connection.commit()
    temporary.replace(index_path)
    return {"articles": count, "quick_check": "ok", "reused": False}


def load_context_articles(
    source_path: Path, index_path: Path, titles: Iterable[str]
) -> dict[str, str]:
    requested = sorted({normalize_title(title) for title in titles if normalize_title(title)})
    result: dict[str, str] = {}
    with closing(
        sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    ) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows: list[tuple[str, int, int]] = []
        for start in range(0, len(requested), 500):
            batch = requested[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                connection.execute(
                    f"SELECT title, offset, length FROM articles WHERE title IN ({placeholders})",
                    batch,
                ).fetchall()
            )
    with source_path.open("rb") as stream:
        for title, offset, length in rows:
            stream.seek(int(offset))
            raw = stream.read(int(length)).strip()
            if raw.endswith(b","):
                raw = raw[:-1]
            parsed = json.loads(b"{" + raw + b"}")
            if list(parsed) != [title]:
                raise ValueError("v86 indexed context title is invalid")
            value = parsed[title]
            if not isinstance(value, str):
                raise ValueError("v86 indexed context record is invalid")
            result[title] = value
    return result


def _article_question_map(
    train_rows: Sequence[dict[str, Any]], selected_ids: Sequence[str]
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    wanted = set(selected_ids)
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for article in train_rows:
        for question in article.get("questions", []):
            qid = str(question.get("qid") or "").strip()
            if qid in wanted:
                if qid in result:
                    raise ValueError("v86 selected question ID is duplicated")
                result[qid] = (article, question)
    if set(result) != wanted:
        raise ValueError("v86 selected question IDs are missing")
    return result


def requested_titles_for_stage(
    train_rows: Sequence[dict[str, Any]], selected_ids: Sequence[str]
) -> set[str]:
    mapping = _article_question_map(train_rows, selected_ids)
    return {
        normalize_title(link.get("target"))
        for article, _ in mapping.values()
        for link in article.get("links", [])
        if normalize_title(link.get("target"))
    }


def _best_chunk(
    question: str, title: str, raw_text: str, tokenizer: Any
) -> tuple[str, int] | None:
    chunks: list[tuple[str, int, int]] = []
    for paragraph_index, raw_paragraph in enumerate(raw_text.split("\n\n")):
        paragraph = clean_text(raw_paragraph)
        if not paragraph:
            continue
        token_ids = list(tokenizer.encode(paragraph, add_special_tokens=False))
        for chunk_index, start in enumerate(range(0, len(token_ids), STRIDE_TOKENS)):
            local = token_ids[start : start + CONTENT_TOKENS]
            if not local:
                break
            text = clean_text(
                tokenizer.decode(
                    local,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            if text:
                chunks.append((text, len(local), paragraph_index * 10000 + chunk_index))
            if start + CONTENT_TOKENS >= len(token_ids):
                break
    if not chunks:
        return None
    texts = [f"{title}. {text}" for text, _, _ in chunks]
    scores = _bm25(question, texts)
    index = min(
        range(len(chunks)),
        key=lambda value: (-float(scores[value]), chunks[value][2]),
    )
    text, _, _ = chunks[index]
    final_ids = list(
        tokenizer.encode(f"{title}. {text}", add_special_tokens=False)
    )[:CONTENT_TOKENS]
    final_text = clean_text(
        tokenizer.decode(
            final_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    return (final_text, len(final_ids)) if final_text else None


def prepare_case(
    article: dict[str, Any],
    question_row: dict[str, Any],
    articles: dict[str, str],
    tokenizer: Any,
    *,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    qid = str(question_row.get("qid") or "").strip()
    question = clean_text(question_row.get("question"))
    pid = str(article.get("pid") or "").strip()
    if not qid or not question or not pid:
        raise ValueError("v86 selected row has an empty ID, question, or passage ID")
    main_source = source_id("main", pid)
    source_by_passage = {"main": main_source}
    candidates: list[dict[str, Any]] = []
    main_chunk = _best_chunk(question, clean_text(article.get("title")), str(article.get("text") or ""), tokenizer)
    if main_chunk is None:
        raise ValueError("v86 selected row has no usable main passage")
    candidates.append(
        {
            "id": f"iirc::{qid}::{main_source}",
            "source": main_source,
            "text": main_chunk[0],
            "token_count": main_chunk[1],
            "metadata": {"source_kind": "main"},
        }
    )
    requested = 0
    unresolved = 0
    seen_titles: set[str] = set()
    for link in article.get("links", []):
        title = normalize_title(link.get("target"))
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        requested += 1
        local_source = source_id("link", title)
        source_by_passage[title] = local_source
        raw_text = articles.get(title)
        if raw_text is None:
            unresolved += 1
            continue
        chunk = _best_chunk(question, title, raw_text, tokenizer)
        if chunk is None:
            unresolved += 1
            continue
        candidates.append(
            {
                "id": f"iirc::{qid}::{local_source}",
                "source": local_source,
                "text": chunk[0],
                "token_count": chunk[1],
                "metadata": {"source_kind": "linked"},
            }
        )
    candidates.sort(key=lambda item: (str(item["source"]), str(item["id"])))
    if len(candidates) < MAX_SOURCES:
        raise ValueError("v86 selected row has fewer than four blind candidate sources")
    contexts = question_row.get("context")
    answer = question_row.get("answer")
    if not isinstance(contexts, list) or not isinstance(answer, dict):
        raise ValueError("v86 selected row gold fields are invalid")
    gold_passages = {
        normalize_title(context.get("passage"))
        for context in contexts
        if isinstance(context, dict)
    }
    if "main" in {
        str(context.get("passage") or "").strip().lower()
        for context in contexts
        if isinstance(context, dict)
    }:
        gold_passages.add("main")
    if not gold_passages or "" in gold_passages:
        raise ValueError("v86 selected row has empty gold evidence")
    if any(passage not in source_by_passage for passage in gold_passages):
        raise ValueError("v86 gold passage is outside the original article links")
    gold_sources = sorted(source_by_passage[passage] for passage in gold_passages)
    candidate_sources = {str(candidate["source"]) for candidate in candidates}
    case_id = f"iirc::{qid}"
    blind = {
        "id": case_id,
        "dataset": "iirc",
        "source": f"iirc_train_v86_{stage}",
        "question": question,
        "required_roles": list(ROLE_NAMES),
        "candidates": candidates,
    }
    gold = {
        "id": case_id,
        "public_id": qid,
        "answer_type": str(answer.get("type") or ""),
        "gold_evidence_sources": gold_sources,
        "candidate_ceiling_complete": set(gold_sources) <= candidate_sources,
    }
    census = {
        "requested_link_sources": requested,
        "unresolved_link_sources": unresolved,
        "candidate_sources": len(candidates),
        "gold_sources": len(gold_sources),
        "gold_sources_in_candidates": len(set(gold_sources) & candidate_sources),
    }
    return blind, gold, census


def prepare_stage(
    train_path: Path,
    context_json: Path,
    context_index: Path,
    tokenizer: Any,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v86 stage: {stage}")
    train_rows = _read_iirc(train_path)
    partitions = build_partitions(train_rows)
    selected_ids = partitions[stage]
    mapping = _article_question_map(train_rows, selected_ids)
    requested_titles = requested_titles_for_stage(train_rows, selected_ids)
    articles = load_context_articles(context_json, context_index, requested_titles)
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    census_rows: list[dict[str, int]] = []
    for qid in selected_ids:
        blind, gold, census = prepare_case(
            *mapping[qid], articles, tokenizer, stage=stage
        )
        blind_rows.append(blind)
        gold_rows.append(gold)
        census_rows.append(census)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "selected_ids_sha256": PARTITION_COMMITMENTS[stage],
        "blind_sha256": sha256(blind_path),
        "sealed_gold_sha256": sha256(gold_path),
        "requested_unique_context_titles": len(requested_titles),
        "resolved_unique_context_titles": len(articles),
        "candidate_sources": {
            "minimum": min(row["candidate_sources"] for row in census_rows),
            "mean": round(float(np.mean([row["candidate_sources"] for row in census_rows])), 6),
            "maximum": max(row["candidate_sources"] for row in census_rows),
        },
        "unresolved_link_sources": sum(row["unresolved_link_sources"] for row in census_rows),
        "gold_statistics_sealed_not_reported": True,
        "confirmation_overlap": 0,
    }


def _rank_by_score(row: dict[str, Any], score: str) -> list[dict[str, Any]]:
    return sorted(
        row.get("candidates", []),
        key=lambda item: (
            -float(item.get("scores", {}).get(score, float("-inf"))),
            str(item.get("id") or ""),
        ),
    )


def select_all_methods(
    row: dict[str, Any], model: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    for method, score in (
        (BM25_FIXED2, "bm25"),
        (DENSE_FIXED2, "dense"),
        (HYBRID_FIXED2, "hybrid"),
    ):
        selected = _rank_by_score(row, score)[:2]
        methods[method] = {
            "selected_ids": [str(item["id"]) for item in selected],
            "selected_sources": [str(item["source"]) for item in selected],
            "cardinality": len(selected),
        }
    cross_order = _rank_by_score(row, "cross_encoder")
    frc_order = rank_evidence_sources(row, limit=MAX_SOURCES)
    for cardinality in range(1, 5):
        cross = cross_order[:cardinality]
        methods[f"cross_encoder_fixed{cardinality}"] = {
            "selected_ids": [str(item["id"]) for item in cross],
            "selected_sources": [str(item["source"]) for item in cross],
            "cardinality": len(cross),
        }
        frc = frc_order[:cardinality]
        methods[f"frc_fixed{cardinality}"] = {
            "selected_ids": [str(item["candidate_id"]) for item in frc],
            "selected_sources": [str(item["source"]) for item in frc],
            "cardinality": len(frc),
        }
    methods[CANDIDATE_METHOD] = select_adaptive_method(row, model)
    return methods


def write_selection_outputs(
    scored_path: Path, model_path: Path, output_path: Path
) -> list[dict[str, Any]]:
    model = load_model_artifact(model_path)
    rows = read_jsonl(scored_path)
    forbidden = {
        "answer",
        "answer_type",
        "context",
        "gold_evidence_sources",
        "question_links",
    }
    if any(forbidden & set(row) for row in rows):
        raise ValueError("v86 scored cache contains a forbidden gold field")
    outputs = [
        {
            "case_id": str(row["id"]),
            "candidate_sources": len({str(c["source"]) for c in row["candidates"]}),
            "methods": select_all_methods(row, model),
        }
        for row in rows
    ]
    write_jsonl(output_path, outputs)
    return outputs


def _case_metrics(gold: Sequence[str], selected: Sequence[str]) -> dict[str, float]:
    gold_set = set(gold)
    selected_set = set(selected)
    overlap = len(gold_set & selected_set)
    precision = overlap / len(selected_set) if selected_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "complete_evidence": float(gold_set <= selected_set),
    }


def _aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        "evidence_macro_precision": round(float(np.mean([row["evidence_precision"] for row in rows])), 6),
        "evidence_macro_recall": round(float(np.mean([row["evidence_recall"] for row in rows])), 6),
        "evidence_macro_f1": round(float(np.mean([row["evidence_f1"] for row in rows])), 6),
        "complete_evidence_recall": round(float(np.mean([row["complete_evidence"] for row in rows])), 6),
    }


def _paired_bootstrap(
    candidate: np.ndarray,
    controls: dict[str, np.ndarray],
    *,
    seed: int,
    strongest_inside_resample: bool,
) -> dict[str, Any]:
    if not controls:
        raise ValueError("v86 bootstrap requires controls")
    point_control = max(controls, key=lambda name: (float(np.mean(controls[name])), name))
    point = float(np.mean(candidate) - np.mean(controls[point_control]))
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    names = sorted(controls)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(candidate), len(candidate))
        control_mean = (
            max(float(np.mean(controls[name][sample])) for name in names)
            if strongest_inside_resample
            else float(np.mean(controls[point_control][sample]))
        )
        values[index] = float(np.mean(candidate[sample])) - control_mean
    return {
        "point": round(point, 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
        "point_strongest_control": point_control,
        "strongest_selected_inside_each_resample": strongest_inside_resample,
    }


def _stratum_name(gold: dict[str, Any]) -> list[str]:
    cardinality = len(gold["gold_evidence_sources"])
    return [
        f"answer_type={gold['answer_type']}",
        f"gold_cardinality={cardinality if cardinality < 4 else '4_plus'}",
        f"candidate_ceiling={'complete' if gold['candidate_ceiling_complete'] else 'incomplete'}",
    ]


def evaluate_stage(
    selection_path: Path,
    gold_path: Path,
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stage not in STAGES:
        raise ValueError(f"unsupported v86 stage: {stage}")
    selections = read_jsonl(selection_path)
    gold_rows = read_jsonl(gold_path)
    selection_by_id = {str(row["case_id"]): row for row in selections}
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    expected = STAGE_CASES[stage]
    if (
        len(selections) != expected
        or len(gold_rows) != expected
        or len(selection_by_id) != expected
        or len(gold_by_id) != expected
        or set(selection_by_id) != set(gold_by_id)
    ):
        raise ValueError("v86 selection/gold coverage changed")
    case_rows: list[dict[str, Any]] = []
    invalid = 0
    for case_id in sorted(gold_by_id):
        gold = gold_by_id[case_id]
        methods = selection_by_id[case_id].get("methods", {})
        if set(methods) != set(METHODS):
            raise ValueError("v86 selection methods changed")
        metrics: dict[str, Any] = {}
        for method in METHODS:
            selected = [str(value) for value in methods[method].get("selected_sources", [])]
            cardinality = int(methods[method].get("cardinality", 0))
            if (
                not 1 <= cardinality <= MAX_SOURCES
                or cardinality != len(selected)
                or len(selected) != len(set(selected))
            ):
                invalid += 1
            metrics[method] = {
                **_case_metrics(gold["gold_evidence_sources"], selected),
                "selected_sources": cardinality,
            }
        case_rows.append(
            {
                "case_id": case_id,
                "answer_type": gold["answer_type"],
                "gold_cardinality": len(gold["gold_evidence_sources"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "strata": _stratum_name(gold),
                "methods": metrics,
                "candidate_action": int(methods[CANDIDATE_METHOD]["cardinality"]),
                "candidate_probabilities": methods[CANDIDATE_METHOD].get("probabilities", {}),
            }
        )
    aggregates = {
        method: {
            **_aggregate([row["methods"][method] for row in case_rows]),
            "mean_selected_sources": round(
                float(np.mean([row["methods"][method]["selected_sources"] for row in case_rows])), 6
            ),
        }
        for method in METHODS
    }
    f1_arrays = {
        method: np.asarray([row["methods"][method]["evidence_f1"] for row in case_rows])
        for method in METHODS
    }
    strongest = max(
        CONTROL_METHODS,
        key=lambda method: (aggregates[method]["evidence_macro_f1"], method),
    )
    simultaneous = _paired_bootstrap(
        f1_arrays[CANDIDATE_METHOD],
        {method: f1_arrays[method] for method in CONTROL_METHODS},
        seed=BOOTSTRAP_SEEDS[stage],
        strongest_inside_resample=True,
    )
    frc2 = _paired_bootstrap(
        f1_arrays[CANDIDATE_METHOD],
        {"frc_fixed2": f1_arrays["frc_fixed2"]},
        seed=BOOTSTRAP_SEEDS[stage] + 100,
        strongest_inside_resample=False,
    )
    strata: dict[str, Any] = {}
    for name in sorted({name for row in case_rows for name in row["strata"]}):
        local = [row for row in case_rows if name in row["strata"]]
        candidate_value = float(np.mean([row["methods"][CANDIDATE_METHOD]["evidence_f1"] for row in local]))
        strongest_value = float(np.mean([row["methods"][strongest]["evidence_f1"] for row in local]))
        strata[name] = {
            "cases": len(local),
            "candidate_f1": round(candidate_value, 6),
            "strongest_fixed_f1": round(strongest_value, 6),
            "delta": round(candidate_value - strongest_value, 6),
        }
    actions = Counter(row["candidate_action"] for row in case_rows)
    action_fractions = {str(key): value / expected for key, value in sorted(actions.items())}
    ordered_action_counts = sorted(actions.values(), reverse=True)
    second_fraction = (
        ordered_action_counts[1] / expected if len(ordered_action_counts) >= 2 else 0.0
    )
    candidate = aggregates[CANDIDATE_METHOD]
    strongest_metrics = aggregates[strongest]
    supported_deltas = [
        value["delta"]
        for value in strata.values()
        if value["cases"] >= STRICT_GATES["supported_stratum_min_cases"]
    ]
    checks = {
        "exact_cases": len(case_rows) == expected,
        "invalid_selector_output_rate_at_most_0": invalid / (expected * len(METHODS)) <= 0,
        "candidate_evidence_macro_f1_at_least_0_35": candidate["evidence_macro_f1"] >= 0.35,
        "candidate_complete_evidence_recall_at_least_0_30": candidate["complete_evidence_recall"] >= 0.30,
        "candidate_minus_strongest_fixed_f1_at_least_0_005": simultaneous["point"] >= 0.005,
        "candidate_minus_strongest_fixed_ci_low_above_0": simultaneous["ci_low"] > 0.0,
        "candidate_minus_frc_fixed2_f1_at_least_0_005": frc2["point"] >= 0.005,
        "candidate_minus_frc_fixed2_ci_low_above_0": frc2["ci_low"] > 0.0,
        "candidate_complete_recall_delta_vs_strongest_at_least_minus_0_02": candidate["complete_evidence_recall"] - strongest_metrics["complete_evidence_recall"] >= -0.02,
        "candidate_evidence_recall_delta_vs_strongest_at_least_minus_0_02": candidate["evidence_macro_recall"] - strongest_metrics["evidence_macro_recall"] >= -0.02,
        "candidate_mean_selected_sources_above_strongest_at_most_0_5": candidate["mean_selected_sources"] - strongest_metrics["mean_selected_sources"] <= 0.5,
        "distinct_candidate_actions_at_least_2": len(actions) >= 2,
        "second_largest_action_fraction_at_least_0_05": second_fraction >= 0.05,
        "supported_stratum_delta_vs_strongest_at_least_minus_0_03": bool(supported_deltas) and min(supported_deltas) >= -0.03,
        "development_confirmation_overlap_0": True,
    }
    all_pass = all(checks.values())
    status = (
        "IIRC_V86_INDEPENDENT_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
        if stage == "development" and all_pass
        else "IIRC_V86_INDEPENDENT_TRANSFER_SUPPORT_ESTABLISHED"
        if stage == "confirmation" and all_pass
        else f"IIRC_V86_{stage.upper()}_SUPPORT_NOT_ESTABLISHED"
    )
    report = {
        "schema_version": "frc-iirc-cardinality-transfer-result-v86",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "cases": expected,
        "aggregates": aggregates,
        "strongest_fixed_control": strongest,
        "candidate_minus_strongest_fixed": simultaneous,
        "candidate_minus_frc_fixed2": frc2,
        "candidate_action_counts": dict(sorted((str(k), v) for k, v in actions.items())),
        "candidate_action_fractions": action_fractions,
        "strata": strata,
        "checks": checks,
        "all_strict_gates_pass": all_pass,
        "status": status,
        "gate_2": "NO-GO/SHADOW",
        "claim_limits": {
            "independent_iirc_train_partition": True,
            "answer_generation_evaluated": False,
            "setr_reproduced": False,
            "flood_domain_effectiveness_established": False,
            "selector_adoption_authorized": False,
            "canary_or_default_authorized": False,
        },
    }
    return report, case_rows


def write_report(
    report: dict[str, Any], case_rows: Sequence[dict[str, Any]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iirc_cardinality_transfer_{report['stage']}_v86"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    cases_path = output_dir / f"{stem}_cases.jsonl.gz"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        f"# IIRC v86 {report['stage']} result",
        "",
        f"- Status: `{report['status']}`",
        f"- Cases: {report['cases']}",
        f"- Candidate F1: {report['aggregates'][CANDIDATE_METHOD]['evidence_macro_f1']:.6f}",
        f"- Strongest fixed control: `{report['strongest_fixed_control']}`",
        f"- Delta: {report['candidate_minus_strongest_fixed']['point']:.6f} "
        f"(95% CI {report['candidate_minus_strongest_fixed']['ci_low']:.6f}, "
        f"{report['candidate_minus_strongest_fixed']['ci_high']:.6f})",
        f"- Strict gates: {'PASS' if report['all_strict_gates_pass'] else 'FAIL'}",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "This is an independent public evidence-selection transfer test, not SetR reproduction, answer-generation evidence, flood-domain validation, or production authorization.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    write_jsonl_gzip(cases_path, list(case_rows))
    return {"json": json_path, "markdown": markdown_path, "cases": cases_path}
