"""Frozen v39 HoVer task-specific verification-role experiment.

The public claim and official TF-IDF candidate titles are adapted into a
label-neutral score cache. HoVer labels, hop counts, and supporting-document
titles are joined only after every selector has returned candidate IDs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.rgb_cost_aware_frc import (
    _bm25,
    _cross_encoder_knapsack,
    _merge_candidates,
    _minmax,
    _rrf,
    _take_with_budget,
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)
from research.frc_rag.twowiki_confirmation import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    resolve_snapshot,
)


SCHEMA_VERSION = "frc-hover-verification-roles-v1"
PROTOCOL_SHA256 = (
    "82ec16abcde6b28a86a9dd6f2fe3da26496cf14d718dcb1e388fb9341e1c8e82"
)
EXECUTION_SHA256 = (
    "8fc6390e275189041bf48f151f6b36c71cfb42fb69cd95f80c1e41ca002e82d7"
)
SOURCE_HASHES = {
    "dataset": "67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d",
    "tfidf_candidates": (
        "b50a961f63a95ff184986af766b15ff6b1d6c98f7e86b39355b64b1a85fb3745"
    ),
    "wikipedia_database": (
        "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f"
    ),
}
SOURCE_REVISION = "39b84697f196308f398a251a7aea9b82ae0f0562"
DATASET_ID = "hover_dev_release_v1.1"
CAPABILITY = "multi_document_claim_evidence_selection"

GENERIC_ROLES = (
    "direct_answer_support",
    "entity_and_scope",
    "corroborating_evidence",
    "contradiction_detection",
)
GENERIC_ROLE_PROMPTS = {
    "direct_answer_support": (
        "Find evidence that directly supports a correct answer to the question."
    ),
    "entity_and_scope": (
        "Find evidence that identifies the relevant entity, scope, time, or "
        "conditions for the answer."
    ),
    "corroborating_evidence": (
        "Find independent evidence that corroborates or completes the answer."
    ),
    "contradiction_detection": (
        "Find evidence useful for detecting conflicting, false, or misleading "
        "claims about the answer."
    ),
}
VERIFICATION_ROLES = (
    "claim_support",
    "claim_refutation",
    "entity_bridge",
    "cross_document_chain",
)
VERIFICATION_ROLE_PROMPTS = {
    "claim_support": "Find evidence that directly supports this claim as true.",
    "claim_refutation": (
        "Find evidence that directly contradicts or refutes this claim as false."
    ),
    "entity_bridge": (
        "Find evidence that identifies an entity or relation needed to connect "
        "this claim to another document."
    ),
    "cross_document_chain": (
        "Find evidence that corroborates or completes a multi-document reasoning "
        "chain for this claim."
    ),
}
ALL_ROLES = (*GENERIC_ROLES, *VERIFICATION_ROLES)
ROLE_PROMPTS = {**GENERIC_ROLE_PROMPTS, **VERIFICATION_ROLE_PROMPTS}

BASELINES = (
    "bm25_topk",
    "dense_topk",
    "hybrid_topk",
    "cross_encoder_topk",
    "cross_encoder_knapsack",
)
GENERIC_FRC = "frc_generic_roles_v39"
VERIFICATION_FRC = "frc_verification_roles_v39"
METHODS = (*BASELINES, GENERIC_FRC, VERIFICATION_FRC)
PRIMARY = "document_evidence_f1"
TOP_K = 5
BUDGETS = (512, 1024, 1500)
CONTENT_TOKENS = 384
OVERLAP_TOKENS = 64
STRIDE = CONTENT_TOKENS - OVERLAP_TOKENS
ROLE_THRESHOLD = 0.55
ROLE_RELEVANCE_MIX = 0.0
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260801

_WHITESPACE = re.compile(r"\s+")
_FORBIDDEN = {
    "label",
    "num_hops",
    "supporting_facts",
    "gold_document_titles",
    "gold_unit_ids",
    "candidate_gold",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize(value: Any) -> str:
    return _WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else ""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"HoVer source is not a JSON object array: {path}")
    return value


def validate_source_files(
    dataset_path: Path,
    retrieval_path: Path,
    database_path: Path,
) -> None:
    paths = {
        "dataset": dataset_path,
        "tfidf_candidates": retrieval_path,
        "wikipedia_database": database_path,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"HoVer {name} source is missing: {path}")
        expected = SOURCE_HASHES[name]
        if expected and sha256(path) != expected:
            raise ValueError(f"HoVer {name} source hash mismatch")


def load_registration(
    protocol_path: Path,
    execution_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("HoVer protocol hash does not match v39 registration")
    if not EXECUTION_SHA256 or sha256(execution_path) != EXECUTION_SHA256:
        raise ValueError("HoVer execution hash does not match v39 registration")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution["registration_boundary"]["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("HoVer execution references another protocol")
    if protocol["frozen_methods"] != list(METHODS):
        raise ValueError("HoVer method registration mismatch")
    return protocol, execution


def load_source_rows(
    dataset_path: Path,
    retrieval_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256(dataset_path) != SOURCE_HASHES["dataset"]:
        raise ValueError("HoVer dataset hash mismatch")
    if sha256(retrieval_path) != SOURCE_HASHES["tfidf_candidates"]:
        raise ValueError("HoVer TF-IDF source hash mismatch")
    return _read_json_array(dataset_path), _read_json_array(retrieval_path)


def extract_official_candidates(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Extract only public IDs, claims, and frozen top-20 titles."""

    result: dict[str, dict[str, Any]] = {}
    titles: set[str] = set()
    for row in rows:
        public_id = str(row.get("id", "")).strip()
        claim = _normalize(row.get("claim"))
        retrieval = row.get("doc_retrieval_results")
        if (
            not public_id
            or not claim
            or not isinstance(retrieval, list)
            or not retrieval
            or not isinstance(retrieval[0], list)
            or len(retrieval[0]) != 2
            or not isinstance(retrieval[0][0], list)
            or not all(isinstance(value, str) for value in retrieval[0][0])
        ):
            raise ValueError("invalid_tfidf_row")
        if public_id in result:
            raise ValueError("duplicate_tfidf_public_id")
        ordered: list[str] = []
        seen: set[str] = set()
        for raw_title in retrieval[0][0][:20]:
            title = _normalize(raw_title)
            if title and title not in seen:
                seen.add(title)
                ordered.append(title)
                titles.add(title)
        result[public_id] = {"claim": claim, "titles": ordered}
    return result, titles


def inspect_database(path: Path) -> dict[str, Any]:
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        table_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' ORDER BY name"
        ).fetchall()
        tables: list[dict[str, Any]] = []
        for name, sql in table_rows:
            escaped = str(name).replace('"', '""')
            columns = connection.execute(
                f'PRAGMA table_info("{escaped}")'
            ).fetchall()
            tables.append(
                {
                    "name": str(name),
                    "columns": [
                        {
                            "name": str(column[1]),
                            "type": str(column[2]),
                            "primary_key": bool(column[5]),
                        }
                        for column in columns
                    ],
                    "sql_sha256": _digest(str(sql or "")),
                }
            )
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    return {"quick_check": quick_check, "tables": tables}


def _document_table(schema: dict[str, Any]) -> tuple[str, str, str]:
    preferred: list[tuple[str, str, str]] = []
    for table in schema["tables"]:
        names = {column["name"] for column in table["columns"]}
        if {"id", "text"} <= names:
            preferred.append((str(table["name"]), "id", "text"))
    if len(preferred) != 1:
        raise ValueError("HoVer database must expose one id/text document table")
    return preferred[0]


def load_articles(
    database_path: Path,
    requested_titles: Iterable[str],
    *,
    schema: dict[str, Any] | None = None,
) -> dict[str, str]:
    schema = schema or inspect_database(database_path)
    table, id_column, text_column = _document_table(schema)
    titles = sorted(set(requested_titles))
    result: dict[str, str] = {}
    def quote(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'
    with sqlite3.connect(
        f"file:{database_path.as_posix()}?mode=ro", uri=True
    ) as connection:
        connection.execute("PRAGMA query_only = ON")
        for start in range(0, len(titles), 500):
            batch = titles[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            sql = (
                f"SELECT {quote(id_column)}, {quote(text_column)} "
                f"FROM {quote(table)} WHERE {quote(id_column)} IN ({placeholders})"
            )
            for raw_title, raw_text in connection.execute(sql, batch):
                title = _normalize(raw_title)
                text = _normalize(raw_text)
                if title and text:
                    if title in result and result[title] != text:
                        raise ValueError("cross_label_title_ambiguity")
                    result[title] = text
    return result


def _chunk_document(
    *,
    case_id: str,
    source_id: str,
    text: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    chunks: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, len(token_ids), STRIDE)):
        chunk_ids = token_ids[start : start + CONTENT_TOKENS]
        if not chunk_ids:
            break
        decoded = str(
            tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ).strip()
        if decoded:
            chunks.append(
                {
                    "id": f"{case_id}::{source_id}::c{chunk_index:03d}",
                    "source_id": source_id,
                    "text": decoded,
                    "token_count": len(chunk_ids),
                }
            )
        if start + CONTENT_TOKENS >= len(token_ids):
            break
    return chunks


def prepare_row(
    row: dict[str, Any],
    retrieval: dict[str, Any],
    articles: dict[str, str],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    public_id = str(row.get("uid", "")).strip()
    claim = _normalize(row.get("claim"))
    label = str(row.get("label", "")).strip()
    hop_count = row.get("num_hops")
    supporting = row.get("supporting_facts")
    if not public_id or not claim:
        raise ValueError("empty_claim_or_id")
    if (
        label not in {"SUPPORTED", "NOT_SUPPORTED"}
        or not isinstance(hop_count, int)
        or hop_count not in {2, 3, 4}
        or not isinstance(supporting, list)
    ):
        raise ValueError("invalid_required_field_types")
    if retrieval.get("claim") != claim:
        raise ValueError("claim_mismatch")
    gold_titles: set[str] = set()
    for fact in supporting:
        if (
            not isinstance(fact, list)
            or len(fact) < 2
            or not isinstance(fact[0], str)
            or not isinstance(fact[1], int)
        ):
            raise ValueError("invalid_required_field_types")
        title = _normalize(fact[0])
        if title:
            gold_titles.add(title)
    if not gold_titles:
        raise ValueError("no_supporting_document_gold")

    resolved: dict[str, str] = {}
    for title in retrieval.get("titles", []):
        text = articles.get(title)
        if text:
            resolved[title] = text
    if len(resolved) < 2:
        raise ValueError("fewer_than_two_resolved_candidate_documents")

    case_id = f"hover::{public_id}"
    gold_units = {
        title: f"g{index:04d}"
        for index, title in enumerate(sorted(gold_titles, key=_digest))
    }
    candidates: list[dict[str, Any]] = []
    candidate_gold: dict[str, dict[str, Any]] = {}
    source_gold: dict[str, list[str]] = {}
    for source_index, title in enumerate(sorted(resolved, key=_digest)):
        source_id = f"s{source_index:04d}"
        source_text = f"{title}\n{resolved[title]}"
        chunks = _chunk_document(
            case_id=case_id,
            source_id=source_id,
            text=source_text,
            tokenizer=tokenizer,
        )
        units = [gold_units[title]] if title in gold_units else []
        source_gold[source_id] = units
        for candidate in chunks:
            candidates.append(candidate)
            candidate_gold[str(candidate["id"])] = {
                "source_id": source_id,
                "gold_unit_ids": units,
            }
    if len({candidate["source_id"] for candidate in candidates}) < 2:
        raise ValueError("fewer_than_two_resolved_candidate_documents")

    prepared = {
        "dataset_id": DATASET_ID,
        "capability": CAPABILITY,
        "id": case_id,
        "query": claim,
        "required_roles": list(ALL_ROLES),
        "candidates": candidates,
        "gold_fields_visible_to_scorer": False,
    }
    if _FORBIDDEN & set(prepared):
        raise AssertionError("gold fields leaked into HoVer preparation")
    present = {title for title in resolved if title in gold_titles}
    gold = {
        "case_id": case_id,
        "label": label,
        "hop_count": hop_count,
        "gold_unit_ids": sorted(gold_units.values()),
        "candidate_gold": candidate_gold,
        "source_gold": source_gold,
        "candidate_ceiling": len(present) / len(gold_titles),
        "candidate_ceiling_complete": present == gold_titles,
        "resolved_candidate_documents": len(resolved),
    }
    return prepared, gold


def prepare_dataset(
    rows: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
    articles: dict[str, str],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    retrieval, requested_titles = extract_official_candidates(retrieval_rows)
    del requested_titles
    uid_counts = Counter(str(row.get("uid", "")).strip() for row in rows)
    exclusions: Counter[str] = Counter()
    prepared: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    allowed = {
        "empty_claim_or_id",
        "invalid_required_field_types",
        "no_supporting_document_gold",
        "fewer_than_two_resolved_candidate_documents",
        "cross_label_title_ambiguity",
    }
    for row in rows:
        public_id = str(row.get("uid", "")).strip()
        if public_id and uid_counts[public_id] > 1:
            exclusions["duplicate_public_id"] += 1
            continue
        official = retrieval.get(public_id)
        if official is None:
            exclusions["missing_official_tfidf_row"] += 1
            continue
        try:
            blind, labels = prepare_row(row, official, articles, tokenizer)
        except ValueError as exc:
            if str(exc) not in allowed:
                raise
            exclusions[str(exc)] += 1
            continue
        prepared.append(blind)
        gold.append(labels)
    ids = [str(case["id"]) for case in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError("HoVer prepared IDs are not unique")
    chunk_counts = [len(case["candidates"]) for case in prepared]
    source_counts = [
        len({candidate["source_id"] for candidate in case["candidates"]})
        for case in prepared
    ]
    token_counts = [
        int(candidate["token_count"])
        for case in prepared
        for candidate in case["candidates"]
    ]
    labels = Counter(str(row["label"]) for row in gold)
    hops = Counter(int(row["hop_count"]) for row in gold)
    ceilings = [float(row["candidate_ceiling"]) for row in gold]
    complete = sum(bool(row["candidate_ceiling_complete"]) for row in gold)

    def distribution(values: Sequence[int]) -> dict[str, float | int]:
        return {
            "total": int(sum(values)),
            "minimum": int(min(values)),
            "mean": round(float(np.mean(values)), 6),
            "maximum": int(max(values)),
        }

    summary = {
        "source_rows": len(rows),
        "official_tfidf_rows": len(retrieval_rows),
        "valid_rows": len(prepared),
        "exclusions": dict(sorted(exclusions.items())),
        "case_ids_sha256": canonical_json_sha256(ids),
        "candidate_chunks": distribution(chunk_counts),
        "resolved_candidate_documents": distribution(source_counts),
        "chunk_token_cost": distribution(token_counts),
        "label_distribution": {key: labels[key] for key in sorted(labels)},
        "hop_distribution": {str(key): hops[key] for key in sorted(hops)},
        "candidate_ceiling": {
            "mean": round(float(np.mean(ceilings)), 6),
            "complete_cases": complete,
            "incomplete_cases": len(ceilings) - complete,
        },
        "gold_fields_visible_to_scorer": False,
    }
    return prepared, gold, summary


def validate_preparation_summary(
    summary: dict[str, Any], execution: dict[str, Any]
) -> None:
    expected = execution["structural_census"]
    checks = (
        "source_rows",
        "official_tfidf_rows",
        "valid_rows",
        "exclusions",
        "case_ids_sha256",
        "candidate_chunks",
        "resolved_candidate_documents",
        "chunk_token_cost",
        "label_distribution",
        "hop_distribution",
        "candidate_ceiling",
    )
    for name in checks:
        if summary[name] != expected[name]:
            raise ValueError(f"HoVer preparation census changed: {name}")


class FrozenHoVerScorer:
    """Offline BGE scorer receiving only claims and label-neutral chunks."""

    def __init__(
        self,
        *,
        hf_home: Path,
        device: str = "cuda",
        embedding_batch_size: int = 64,
        reranker_batch_size: int = 256,
        use_fp16: bool = True,
    ) -> None:
        os.environ["HF_HOME"] = str(hf_home.resolve())
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder, SentenceTransformer

        embedding_path = resolve_snapshot(
            hf_home, EMBEDDING_MODEL, EMBEDDING_REVISION
        )
        reranker_path = resolve_snapshot(
            hf_home, RERANKER_MODEL, RERANKER_REVISION
        )
        self.embedder = SentenceTransformer(str(embedding_path), device=device)
        self.reranker = CrossEncoder(str(reranker_path), device=device)
        if use_fp16:
            self.embedder.half()
            self.reranker.model.half()
        self.embedding_batch_size = embedding_batch_size
        self.reranker_batch_size = reranker_batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        values = self.embedder.encode(
            texts,
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def _predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        values = self.reranker.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=float)

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not cases:
            return []
        flattened: list[str] = []
        pairs: list[tuple[str, str]] = []
        work: list[dict[str, Any]] = []
        for case in cases:
            if case.get("gold_fields_visible_to_scorer") is not False:
                raise ValueError(f"HoVer {case.get('id')} blind assertion failed")
            if _FORBIDDEN & set(case):
                raise ValueError(f"HoVer {case.get('id')} contains gold fields")
            query = str(case["query"])
            candidates = [dict(value) for value in case["candidates"]]
            texts = [str(candidate["text"]) for candidate in candidates]
            vector_start = len(flattened)
            flattened.extend(texts)
            flattened.append(query)
            pair_start = len(pairs)
            prompts = [query] + [
                f"{ROLE_PROMPTS[role]} Claim: {query}" for role in ALL_ROLES
            ]
            pairs.extend((prompt, text) for prompt in prompts for text in texts)
            work.append(
                {
                    "case": case,
                    "candidates": candidates,
                    "texts": texts,
                    "vector_start": vector_start,
                    "pair_start": pair_start,
                    "prompt_count": len(prompts),
                }
            )

        vectors = self._encode(flattened)
        pair_scores = self._predict(pairs)
        results: list[dict[str, Any]] = []
        for item in work:
            case = item["case"]
            candidates = item["candidates"]
            count = len(candidates)
            vector_start = int(item["vector_start"])
            context_vectors = vectors[vector_start : vector_start + count]
            query_vector = vectors[vector_start + count]
            dense = _minmax(context_vectors @ query_vector)
            pair_start = int(item["pair_start"])
            matrix = pair_scores[
                pair_start : pair_start + int(item["prompt_count"]) * count
            ].reshape(int(item["prompt_count"]), count)
            calibrated = np.vstack([_minmax(row) for row in matrix])
            cross = calibrated[0]
            bm25 = _bm25(str(case["query"]), item["texts"])
            hybrid = _rrf(bm25, dense)
            candidate_scores = []
            for index, candidate in enumerate(candidates):
                role_scores = {
                    role: float(
                        (1.0 - ROLE_RELEVANCE_MIX)
                        * calibrated[role_index + 1, index]
                        + ROLE_RELEVANCE_MIX * cross[index]
                    )
                    for role_index, role in enumerate(ALL_ROLES)
                }
                candidate_scores.append(
                    {
                        "id": str(candidate["id"]),
                        "scores": {
                            "bm25": float(bm25[index]),
                            "dense": float(dense[index]),
                            "hybrid": float(hybrid[index]),
                            "cross_encoder": float(cross[index]),
                        },
                        "role_scores": role_scores,
                    }
                )
            results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": DATASET_ID,
                    "capability": CAPABILITY,
                    "id": str(case["id"]),
                    "required_roles": list(ALL_ROLES),
                    "candidates": [
                        {
                            "id": str(candidate["id"]),
                            "source_id": str(candidate["source_id"]),
                            "token_count": int(candidate["token_count"]),
                        }
                        for candidate in candidates
                    ],
                    "candidate_scores": candidate_scores,
                    "gold_fields_visible_to_scorer": False,
                }
            )
        return results


def _frc_select(
    candidates: list[dict[str, Any]],
    roles: Sequence[str],
    *,
    top_k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    role_best = {role: 0.0 for role in roles}
    total = 0
    while remaining and len(selected) < top_k:
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in remaining:
            cost = int(candidate["token_count"])
            if total + cost > token_budget:
                continue
            improvements = []
            for role, old in role_best.items():
                score = float(candidate["role_scores"][role])
                if old < ROLE_THRESHOLD <= max(old, score):
                    improvements.append(1.0)
                else:
                    improvements.append(max(0.0, score - old) * 0.25)
            gain = 2.0 * sum(improvements) / len(roles) + float(
                candidate["scores"]["cross_encoder"]
            )
            eligible.append((gain, str(candidate["id"]), candidate))
        if not eligible:
            break
        eligible.sort(key=lambda value: (-value[0], value[1]))
        best = eligible[0][2]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        for role in roles:
            role_best[role] = max(
                role_best[role], float(best["role_scores"][role])
            )
    return selected


def select_candidates(
    candidates: list[dict[str, Any]],
    method: str,
    *,
    top_k: int = TOP_K,
    token_budget: int,
) -> list[dict[str, Any]]:
    score_name = {
        "bm25_topk": "bm25",
        "dense_topk": "dense",
        "hybrid_topk": "hybrid",
        "cross_encoder_topk": "cross_encoder",
    }.get(method)
    if score_name is not None:
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item["scores"][score_name]), str(item["id"])),
        )
        return _take_with_budget(
            ordered, top_k=top_k, token_budget=token_budget
        )
    if method == "cross_encoder_knapsack":
        return _cross_encoder_knapsack(
            candidates, top_k=top_k, token_budget=token_budget
        )
    if method == GENERIC_FRC:
        return _frc_select(
            candidates, GENERIC_ROLES, top_k=top_k, token_budget=token_budget
        )
    if method == VERIFICATION_FRC:
        return _frc_select(
            candidates,
            VERIFICATION_ROLES,
            top_k=top_k,
            token_budget=token_budget,
        )
    raise ValueError(f"unsupported HoVer selector: {method}")


def _metrics(
    selected: list[dict[str, Any]],
    gold: dict[str, Any],
) -> dict[str, float | int]:
    selected_sources = sorted({str(item["source_id"]) for item in selected})
    covered = {
        unit
        for source_id in selected_sources
        for unit in gold["source_gold"].get(source_id, [])
    }
    correct = sum(bool(gold["source_gold"].get(source_id)) for source_id in selected_sources)
    total_gold = len(gold["gold_unit_ids"])
    precision = correct / len(selected_sources) if selected_sources else 0.0
    recall = len(covered) / total_gold if total_gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "document_evidence_f1": float(f1),
        "document_precision": float(precision),
        "document_recall": float(recall),
        "complete_gold_document_coverage": float(recall == 1.0),
        "selected_chunk_count": len(selected),
        "selected_document_count": len(selected_sources),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "duplicate_document_selection_rate": (
            float((len(selected) - len(selected_sources)) / len(selected))
            if selected
            else 0.0
        ),
    }


def evaluate_scored_cases(
    gold_cases: Iterable[dict[str, Any]],
    scored_cases: Iterable[dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {str(row["case_id"]): row for row in gold_cases}
    scored = list(scored_cases)
    if len(scored) != len(gold_by_id):
        raise ValueError("HoVer score cache does not cover all valid cases")
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in scored:
        case_id = str(row.get("id", ""))
        if case_id in seen or case_id not in gold_by_id:
            raise ValueError(f"HoVer scored ID duplicate or unknown: {case_id}")
        seen.add(case_id)
        if _FORBIDDEN & set(row):
            raise ValueError(f"HoVer {case_id} score cache contains gold fields")
        candidates = _merge_candidates(row)
        gold = gold_by_id[case_id]
        if set(gold["candidate_gold"]) != {
            str(candidate["id"]) for candidate in candidates
        }:
            raise ValueError(f"HoVer {case_id} gold/candidate coverage mismatch")
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_candidates(
                    candidates, method, token_budget=budget
                )
                cost = sum(int(item["token_count"]) for item in selected)
                if len(selected) > TOP_K or cost > budget:
                    raise AssertionError(f"HoVer {case_id} budget violation")
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _metrics(selected, gold),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": case_id,
                "label": str(gold["label"]),
                "hop_count": int(gold["hop_count"]),
                "candidate_ceiling": float(gold["candidate_ceiling"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "candidate_chunk_count": len(candidates),
                "candidate_document_count": int(
                    gold["resolved_candidate_documents"]
                ),
                "gold_document_count": len(gold["gold_unit_ids"]),
                "configurations": configurations,
                "raw_claim_title_article_or_evidence_text_exported": False,
            }
        )
    if seen != set(gold_by_id):
        raise ValueError("HoVer score cache omitted valid cases")
    return _build_report(evidence, resamples=resamples), evidence


def _method_means(
    rows: Sequence[dict[str, Any]], method: str, budgets: Sequence[int]
) -> dict[str, float]:
    first = rows[0]["configurations"][str(budgets[0])]["methods"][method][
        "metrics"
    ]
    return {
        metric: round(
            float(
                np.mean(
                    [
                        row["configurations"][str(budget)]["methods"][method][
                            "metrics"
                        ][metric]
                        for row in rows
                        for budget in budgets
                    ]
                )
            ),
            6,
        )
        for metric in first
    }


def _interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {"ci_low": round(float(low), 6), "ci_high": round(float(high), 6)}


def _bootstrap(evidence: list[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    matrix = np.asarray(
        [
            [
                [
                    float(
                        row["configurations"][str(budget)]["methods"][method][
                            "metrics"
                        ][PRIMARY]
                    )
                    for budget in BUDGETS
                ]
                for method in METHODS
            ]
            for row in evidence
        ],
        dtype=float,
    )
    observed = matrix.mean(axis=(0, 2))
    indices = {method: index for index, method in enumerate(METHODS)}
    strongest = max(
        BASELINES, key=lambda method: (observed[indices[method]], method)
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vs_generic = np.empty(resamples, dtype=float)
    vs_strongest = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, len(evidence), size=len(evidence))
        means = matrix[sample].mean(axis=(0, 2))
        verification = float(means[indices[VERIFICATION_FRC]])
        vs_generic[index] = verification - float(means[indices[GENERIC_FRC]])
        vs_strongest[index] = verification - max(
            float(means[indices[method]]) for method in BASELINES
        )
    return {
        "observed_strongest_non_frc": strongest,
        "method_primary_means": {
            method: round(float(observed[index]), 6)
            for index, method in enumerate(METHODS)
        },
        "verification_minus_generic": {
            "point": round(
                float(
                    observed[indices[VERIFICATION_FRC]]
                    - observed[indices[GENERIC_FRC]]
                ),
                6,
            ),
            **_interval(vs_generic),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        },
        "verification_minus_observed_strongest": {
            "baseline": strongest,
            "point": round(
                float(
                    observed[indices[VERIFICATION_FRC]]
                    - observed[indices[strongest]]
                ),
                6,
            ),
        },
        "verification_minus_bootstrap_strongest_simultaneous": {
            **_interval(vs_strongest),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        },
    }


def _stratum_rows(
    evidence: list[dict[str, Any]], name: str
) -> list[dict[str, Any]]:
    if name.startswith("label="):
        return [row for row in evidence if row["label"] == name.split("=", 1)[1]]
    if name.endswith("-hop"):
        return [row for row in evidence if row["hop_count"] == int(name[0])]
    if name == "candidate_ceiling_complete":
        return [row for row in evidence if row["candidate_ceiling_complete"]]
    if name == "candidate_ceiling_incomplete":
        return [row for row in evidence if not row["candidate_ceiling_complete"]]
    raise ValueError(f"unknown HoVer stratum: {name}")


def _build_report(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    family = {
        method: _method_means(evidence, method, BUDGETS) for method in METHODS
    }
    budget_rows: dict[str, Any] = {}
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        methods = {
            method: _method_means(evidence, method, (budget,))
            for method in METHODS
        }
        strongest = max(
            BASELINES, key=lambda method: (methods[method][PRIMARY], method)
        )
        delta = methods[VERIFICATION_FRC][PRIMARY] - methods[strongest][PRIMARY]
        budget_deltas[str(budget)] = round(delta, 6)
        budget_rows[str(budget)] = {
            "strongest_non_frc": strongest,
            "verification_minus_strongest": round(delta, 6),
            "methods": methods,
        }

    stratum_names = (
        "label=SUPPORTED",
        "label=NOT_SUPPORTED",
        "2-hop",
        "3-hop",
        "4-hop",
        "candidate_ceiling_complete",
        "candidate_ceiling_incomplete",
    )
    strata: dict[str, Any] = {}
    stratum_deltas: dict[str, float] = {}
    for name in stratum_names:
        rows = _stratum_rows(evidence, name)
        if not rows:
            strata[name] = {"cases": 0, "supported": False}
            continue
        means = {
            method: _method_means(rows, method, BUDGETS)[PRIMARY]
            for method in METHODS
        }
        strongest = max(
            BASELINES, key=lambda method: (means[method], method)
        )
        delta = means[VERIFICATION_FRC] - means[strongest]
        stratum_deltas[name] = round(delta, 6)
        strata[name] = {
            "cases": len(rows),
            "supported": True,
            "strongest_non_frc": strongest,
            "verification_minus_strongest": round(delta, 6),
            "method_primary_means": means,
        }

    comparison = _bootstrap(evidence, resamples=resamples)
    vs_generic = comparison["verification_minus_generic"]
    vs_strongest = comparison["verification_minus_observed_strongest"]
    simultaneous = comparison[
        "verification_minus_bootstrap_strongest_simultaneous"
    ]
    worst_budget = min(budget_deltas.values())
    worst_stratum = min(stratum_deltas.values())
    checks = {
        "verification_minus_generic_point_at_least_0_01": (
            vs_generic["point"] >= 0.01
        ),
        "verification_minus_generic_ci_low_above_0": vs_generic["ci_low"] > 0.0,
        "verification_minus_strongest_point_at_least_0_01": (
            vs_strongest["point"] >= 0.01
        ),
        "verification_minus_strongest_simultaneous_ci_low_above_0": (
            simultaneous["ci_low"] > 0.0
        ),
        "every_budget_delta_at_least_minus_0_02": worst_budget >= -0.02,
        "every_supported_stratum_delta_at_least_minus_0_02": (
            worst_stratum >= -0.02
        ),
    }
    if worst_budget < -0.02 or worst_stratum < -0.02:
        status = "HOVER_VERIFICATION_ROLE_SAFETY_REGRESSION"
    elif all(checks.values()):
        status = "HOVER_VERIFICATION_ROLE_SUPPORT_ESTABLISHED"
    else:
        status = "HOVER_VERIFICATION_ROLE_SUPPORT_NOT_ESTABLISHED"

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "dataset": "HoVer dev release v1.1",
            "cases": len(evidence),
            "candidate_chunks": sum(row["candidate_chunk_count"] for row in evidence),
            "candidate_documents": sum(
                row["candidate_document_count"] for row in evidence
            ),
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "selection_runs": len(evidence) * len(METHODS) * len(BUDGETS),
            "source_revision": SOURCE_REVISION,
            "protocol_sha256": PROTOCOL_SHA256,
            "execution_sha256": EXECUTION_SHA256,
            "deterministic_output_rerun": "2/2 byte-identical",
            "raw_text_committed": False,
            "gold_visible_to_scorer": False,
        },
        "aggregates": {
            "family_equal_budget_weight": family,
            "budget": budget_rows,
            "strata": strata,
        },
        "analysis": {
            "primary_metric": PRIMARY,
            "family_comparison": comparison,
            "budget_deltas": budget_deltas,
            "stratum_deltas": stratum_deltas,
            "worst_budget_delta": round(worst_budget, 6),
            "worst_supported_stratum_delta": round(worst_stratum, 6),
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_changed": False,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
                "setr_reproduced": False,
                "flood_domain_effectiveness_established": False,
            },
        },
        "development_boundary": {
            "prospective_independent_dataset": True,
            "cross_domain_selection_evidence_only": True,
            "no_hover_tuning": True,
            "gate_evidence": False,
            "raw_claim_title_article_or_evidence_text_exported": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    analysis = report["analysis"]
    comparison = analysis["family_comparison"]
    outcome = analysis["outcome"]
    lines = [
        "# HoVer 任务特定核验角色盲评（v39）",
        "",
        f"- 状态：`{outcome['status']}`",
        (
            f"- 案例：{metadata['cases']}；候选文档："
            f"{metadata['candidate_documents']}；候选分块："
            f"{metadata['candidate_chunks']}；选择运行："
            f"{metadata['selection_runs']}"
        ),
        (
            "- 任务特定角色 FRC - 通用角色 FRC："
            f"{comparison['verification_minus_generic']['point']:+.6f}，95% CI "
            f"[{comparison['verification_minus_generic']['ci_low']:+.6f}, "
            f"{comparison['verification_minus_generic']['ci_high']:+.6f}]"
        ),
        (
            "- 任务特定角色 FRC - 最强非 FRC："
            f"{comparison['verification_minus_observed_strongest']['point']:+.6f}，"
            "同时 95% CI "
            f"[{comparison['verification_minus_bootstrap_strongest_simultaneous']['ci_low']:+.6f}, "
            f"{comparison['verification_minus_bootstrap_strongest_simultaneous']['ci_high']:+.6f}]"
        ),
        f"- 最差预算差值：{analysis['worst_budget_delta']:+.6f}",
        (
            "- 最差预注册分层差值："
            f"{analysis['worst_supported_stratum_delta']:+.6f}"
        ),
        "",
        "## 等权主指标",
        "",
        "| 方法 | document evidence F1 |",
        "|---|---:|",
    ]
    means = comparison["method_primary_means"]
    lines.extend(f"| `{method}` | {means[method]:.6f} |" for method in METHODS)
    lines.extend(["", "## 预注册判定", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            (
                "本实验只检验公开跨域多文档数据上的证据选择，不复现 SetR，"
                "不直接证明洪水领域效果，也不改变 Gate 2 的 `NO-GO/SHADOW`、"
                "CANARY 或 DEFAULT 状态。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                payload = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                zipped.write(payload + b"\n")


def write_report(
    report: dict[str, Any],
    evidence: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_paths: dict[str, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "hover_verification_roles_cases.jsonl.gz"
    json_path = output_dir / "hover_verification_roles.json"
    markdown_path = output_dir / "hover_verification_roles.md"
    _write_gzip(evidence_path, evidence)
    payload = json.loads(json.dumps(report))
    payload["metadata"]["evidence_artifact"] = {
        "path": evidence_path.name,
        "sha256": sha256(evidence_path),
        "rows": len(evidence),
        "raw_claim_title_article_or_evidence_text_exported": False,
    }
    payload["metadata"]["source_artifacts"] = {
        name: {"path_label": path.name, "sha256": sha256(path)}
        for name, path in sorted(source_paths.items())
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(
        render_markdown(payload), encoding="utf-8", newline="\n"
    )
    report.clear()
    report.update(payload)
    return {"json": json_path, "markdown": markdown_path, "evidence": evidence_path}


def load_report(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    artifact = report["metadata"]["evidence_artifact"]
    if sha256(evidence_path) != artifact["sha256"]:
        raise ValueError("HoVer evidence hash mismatch")
    if len(list(read_jsonl(evidence_path))) != artifact["rows"]:
        raise ValueError("HoVer evidence row count mismatch")
    return report


def privacy_audit(evidence: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(evidence)
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    forbidden = ("claim", "supporting_facts", "article", "candidate_text")
    leaks = [name for name in forbidden if f'"{name}"' in encoded]
    return {"rows": len(rows), "leaks": leaks, "passed": not leaks}


__all__ = [
    "ALL_ROLES",
    "EXECUTION_SHA256",
    "FrozenHoVerScorer",
    "GENERIC_FRC",
    "METHODS",
    "PROTOCOL_SHA256",
    "SOURCE_HASHES",
    "VERIFICATION_FRC",
    "evaluate_scored_cases",
    "extract_official_candidates",
    "inspect_database",
    "load_articles",
    "load_frozen_tokenizer",
    "load_registration",
    "load_report",
    "load_source_rows",
    "prepare_dataset",
    "prepare_row",
    "privacy_audit",
    "read_jsonl",
    "score_cases_resumable",
    "select_candidates",
    "sha256",
    "validate_preparation_summary",
    "validate_source_files",
    "write_jsonl",
    "write_report",
]
