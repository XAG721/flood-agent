"""Frozen blind RGB evaluation for the v35 cost-aware FRC selector.

Preparation and neural scoring operate on label-free candidates.  Raw RGB labels
are reconstructed in memory only after every selector has returned candidate IDs.
Raw text and answers remain in the ignored benchmark cache.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from research.frc_rag.twowiki_confirmation import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    RERANKER_MODEL,
    RERANKER_REVISION,
    resolve_snapshot,
)


SCHEMA_VERSION = "frc-rgb-cost-aware-v1"
PROTOCOL_SHA256 = "c370f83ada7df1c16b8249e05aad929d0a0dc439efa48594ca03ec04e7e420a4"
EXECUTION_SHA256 = "f22441b13950b033e0f2d029887f77df5be9e3cabd5a1506ef8ee8e9eee50e71"
POST_RESULT_CORRECTION_SHA256 = (
    "10b85a8077dd4613dab091998116520aa5bdf5b2dfcc5229e182486280216772"
)

DATASETS = (
    ("rgb_en_refine", "noise_robustness", "data/en_refine.json"),
    ("rgb_en_int", "information_integration", "data/en_int.json"),
    ("rgb_en_fact", "counterfactual_robustness", "data/en_fact.json"),
)
SOURCE_HASHES = {
    "rgb_en_refine": "ad6468eb2d14496a6a2d9961d35b60924739e21a0c6e7811552f1ac31e3423f3",
    "rgb_en_int": "797a6fc39a75d0cfb88d9c1b57f27277a959870eba3d56f26cf2dad482cd7725",
    "rgb_en_fact": "27b5a3ae2d3cd0c5c12282b3bbab659708dbd46c58e5890fd787f86f5530c28e",
}

ROLE_NAMES = (
    "direct_answer_support",
    "entity_and_scope",
    "corroborating_evidence",
    "contradiction_detection",
)
ROLE_QUERIES = {
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

METHODS = (
    "bm25_topk",
    "dense_topk",
    "hybrid_topk",
    "cross_encoder_topk",
    "cross_encoder_density",
    "cross_encoder_knapsack",
    "coverage_greedy_proxy",
    "frc_select_v34",
    "frc_cost_aware_v35",
)
BASELINES = METHODS[:7]
OLD_FRC = "frc_select_v34"
NEW_FRC = "frc_cost_aware_v35"
PRIMARY = "evidence_f1"

TOP_K = 5
BUDGETS = (512, 1024, 1500)
CONTENT_TOKENS = 384
OVERLAP_TOKENS = 64
STRIDE = CONTENT_TOKENS - OVERLAP_TOKENS
ROLE_RELEVANCE_MIX = 0.15
ROLE_THRESHOLD = 0.55
RRF_K = 60
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260801

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_FORBIDDEN_SCORE_KEYS = {
    "answer",
    "answers",
    "source_label",
    "gold_unit_ids",
    "is_positive",
    "is_negative",
    "is_positive_wrong",
    "candidate_gold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row in {path} is not an object")
                yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def load_registration(
    protocol_path: Path,
    execution_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("RGB protocol hash does not match the frozen v35 registration")
    if sha256(execution_path) != EXECUTION_SHA256:
        raise ValueError("RGB execution hash does not match the frozen v35 registration")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution["registration_boundary"]["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("RGB execution registration references a different protocol")
    if protocol["frozen_methods"] != list(METHODS):
        raise ValueError("RGB registered methods do not match the implementation")
    if protocol["frozen_cost_aware_selector"]["budgets"] != list(BUDGETS):
        raise ValueError("RGB registered budgets do not match the implementation")
    return protocol, execution


def validate_source_files(source_root: Path) -> None:
    for dataset_id, _, relative in DATASETS:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"RGB source file is missing: {path}")
        if sha256(path) != SOURCE_HASHES[dataset_id]:
            raise ValueError(f"RGB source hash mismatch: {dataset_id}")


def load_source_rows(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    validate_source_files(source_root)
    result: dict[str, list[dict[str, Any]]] = {}
    for dataset_id, _, relative in DATASETS:
        result[dataset_id] = list(read_jsonl(source_root / relative))
    return result


def load_frozen_tokenizer(hf_home: Path) -> Any:
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    snapshot = resolve_snapshot(hf_home, RERANKER_MODEL, RERANKER_REVISION)
    return AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        use_fast=True,
    )


def _normalize_text(value: Any) -> str:
    return _WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else ""


def _source_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_case_id(dataset_id: str, row: dict[str, Any]) -> str:
    public_id = str(row.get("id", "")).strip()
    if not public_id:
        raise ValueError("empty_public_id")
    return f"{dataset_id}::{public_id}"


def _source_records(
    dataset_id: str,
    row: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    positive = row.get("positive")
    negative = row.get("negative")
    if not isinstance(positive, list) or not isinstance(negative, list):
        raise ValueError("invalid_field_types")
    if not all(isinstance(value, str) for value in negative):
        raise ValueError("invalid_field_types")

    positive_items: list[tuple[str, str]] = []
    if dataset_id == "rgb_en_int":
        for group_index, group in enumerate(positive):
            if not isinstance(group, list) or not all(
                isinstance(value, str) for value in group
            ):
                raise ValueError("invalid_field_types")
            positive_items.extend(
                (text, f"group-{group_index}") for text in group
            )
    else:
        if not all(isinstance(value, str) for value in positive):
            raise ValueError("invalid_field_types")
        positive_items = [
            (text, f"positive-{_source_digest(_normalize_text(text))}")
            for text in positive
            if _normalize_text(text)
        ]

    wrong = row.get("positive_wrong", []) if dataset_id == "rgb_en_fact" else []
    if not isinstance(wrong, list) or not all(
        isinstance(value, str) for value in wrong
    ):
        raise ValueError("invalid_field_types")

    records: dict[str, dict[str, Any]] = {}

    def add(text: str, label: str, gold_unit: str | None = None) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return
        digest = _source_digest(normalized)
        record = records.setdefault(
            digest,
            {"text": normalized, "labels": set(), "gold_units": set()},
        )
        record["labels"].add(label)
        if gold_unit is not None:
            record["gold_units"].add(gold_unit)

    for text, gold_unit in positive_items:
        add(text, "positive", gold_unit)
    for text in negative:
        add(text, "negative")
    for text in wrong:
        add(text, "positive_wrong")

    gold_units = {
        str(unit)
        for record in records.values()
        for unit in record["gold_units"]
    }
    return records, gold_units


def prepare_row(
    dataset_id: str,
    capability: str,
    row: dict[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = _raw_case_id(dataset_id, row)
    query = _normalize_text(row.get("query"))
    if not query:
        raise ValueError("empty_query")
    records, gold_units = _source_records(dataset_id, row)
    if not gold_units:
        raise ValueError("no_correct_positive_gold")
    if len(records) < 2:
        raise ValueError("fewer_than_two_unique_candidates")
    for record in records.values():
        labels = set(record["labels"])
        if "positive" in labels and labels & {"negative", "positive_wrong"}:
            raise ValueError("cross_label_duplicate_conflict")

    candidates: list[dict[str, Any]] = []
    candidate_gold: dict[str, dict[str, Any]] = {}
    for source_index, (_, record) in enumerate(sorted(records.items())):
        source_id = f"s{source_index:04d}"
        token_ids = list(tokenizer.encode(record["text"], add_special_tokens=False))
        if not token_ids:
            continue
        chunk_index = 0
        for start in range(0, len(token_ids), STRIDE):
            chunk_ids = token_ids[start : start + CONTENT_TOKENS]
            if not chunk_ids:
                break
            candidate_id = f"{case_id}::{source_id}::c{chunk_index:03d}"
            text = str(
                tokenizer.decode(
                    chunk_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ).strip()
            candidates.append(
                {
                    "id": candidate_id,
                    "source_id": source_id,
                    "text": text,
                    "token_count": len(chunk_ids),
                }
            )
            candidate_gold[candidate_id] = {
                "labels": sorted(record["labels"]),
                "gold_unit_ids": sorted(record["gold_units"]),
            }
            chunk_index += 1
            if start + CONTENT_TOKENS >= len(token_ids):
                break
    if len(candidates) < 2:
        raise ValueError("fewer_than_two_unique_candidates")

    prepared = {
        "dataset_id": dataset_id,
        "capability": capability,
        "id": case_id,
        "query": query,
        "required_roles": list(ROLE_NAMES),
        "candidates": candidates,
        "gold_fields_visible_to_scorer": False,
    }
    if _FORBIDDEN_SCORE_KEYS & set(prepared):
        raise AssertionError("gold fields leaked into a prepared RGB case")
    gold = {
        "dataset_id": dataset_id,
        "capability": capability,
        "case_id": case_id,
        "gold_unit_ids": sorted(gold_units),
        "candidate_gold": candidate_gold,
    }
    return prepared, gold


def prepare_datasets(
    source_rows: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    exclusions: dict[str, Counter[str]] = defaultdict(Counter)
    capability_by_dataset = {
        dataset_id: capability for dataset_id, capability, _ in DATASETS
    }
    for dataset_id, _, _ in DATASETS:
        for row in source_rows[dataset_id]:
            try:
                blind, labels = prepare_row(
                    dataset_id,
                    capability_by_dataset[dataset_id],
                    row,
                    tokenizer,
                )
            except ValueError as exc:
                reason = str(exc)
                if reason not in {
                    "empty_public_id",
                    "empty_query",
                    "invalid_field_types",
                    "no_correct_positive_gold",
                    "fewer_than_two_unique_candidates",
                    "cross_label_duplicate_conflict",
                }:
                    raise
                exclusions[dataset_id][reason] += 1
                continue
            prepared.append(blind)
            gold.append(labels)

    ids = [str(case["id"]) for case in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError("RGB prepared cases contain duplicate IDs")
    summary = preparation_summary(prepared, source_rows, exclusions)
    return prepared, gold, summary


def _distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"total": 0, "minimum": 0, "mean": 0.0, "maximum": 0}
    return {
        "total": int(sum(ordered)),
        "minimum": int(ordered[0]),
        "mean": round(float(np.mean(ordered)), 6),
        "maximum": int(ordered[-1]),
    }


def preparation_summary(
    prepared: list[dict[str, Any]],
    source_rows: dict[str, list[dict[str, Any]]],
    exclusions: dict[str, Counter[str]],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_id, _, _ in DATASETS:
        cases = [case for case in prepared if case["dataset_id"] == dataset_id]
        chunk_counts = [len(case["candidates"]) for case in cases]
        token_costs = [
            int(candidate["token_count"])
            for case in cases
            for candidate in case["candidates"]
        ]
        datasets[dataset_id] = {
            "source_rows": len(source_rows[dataset_id]),
            "valid_rows": len(cases),
            "exclusions": dict(sorted(exclusions[dataset_id].items())),
            "case_ids_sha256": canonical_json_sha256(
                [str(case["id"]) for case in cases]
            ),
            "candidate_chunks_per_case": _distribution(chunk_counts),
            "chunk_token_cost": _distribution(token_costs),
        }
    return {
        "valid_cases": len(prepared),
        "candidate_chunks": sum(
            len(case["candidates"]) for case in prepared
        ),
        "datasets": datasets,
        "gold_fields_visible_to_scorer": False,
    }


def validate_preparation_summary(
    summary: dict[str, Any], execution: dict[str, Any]
) -> None:
    census = execution["structural_census"]
    if summary["valid_cases"] != census["total_valid_rows"]:
        raise ValueError("RGB prepared case count does not match registration")
    for dataset_id, expected in census["datasets"].items():
        actual = summary["datasets"][dataset_id]
        if actual["valid_rows"] != next(
            item["valid_rows"]
            for item in execution["source_checkout"]["files"]
            if item["dataset_id"] == dataset_id
        ):
            raise ValueError(f"RGB valid row count mismatch: {dataset_id}")
        if actual["case_ids_sha256"] != expected["case_ids_sha256"]:
            raise ValueError(f"RGB case ID fingerprint mismatch: {dataset_id}")
        if actual["candidate_chunks_per_case"]["total"] != expected[
            "candidate_chunks"
        ]["total"]:
            raise ValueError(f"RGB chunk count mismatch: {dataset_id}")


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bm25(query: str, texts: list[str]) -> np.ndarray:
    documents = [_tokens(text) for text in texts]
    frequencies = Counter(term for tokens in documents for term in set(tokens))
    average_length = float(np.mean([len(tokens) for tokens in documents])) or 1.0
    values: list[float] = []
    query_tokens = _tokens(query)
    for tokens in documents:
        counts = Counter(tokens)
        length = len(tokens) or 1
        score = 0.0
        for term in query_tokens:
            count = counts.get(term, 0)
            if not count:
                continue
            document_count = len(documents)
            document_frequency = frequencies.get(term, 0)
            inverse_frequency = np.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = count + 1.5 * (0.25 + 0.75 * length / average_length)
            score += float(inverse_frequency) * (count * 2.5) / denominator
        values.append(score)
    return _minmax(np.asarray(values, dtype=float))


def _rrf(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    fused = np.zeros(len(left), dtype=float)
    for values in (left, right):
        order = np.argsort(-values, kind="mergesort")
        for rank, index in enumerate(order, start=1):
            fused[index] += 1.0 / (RRF_K + rank)
    return _minmax(fused)


class FrozenRGBScorer:
    """Offline BGE scorer that never receives RGB answer or label fields."""

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
        work: list[dict[str, Any]] = []
        pairs: list[tuple[str, str]] = []
        for case in cases:
            if case.get("gold_fields_visible_to_scorer") is not False:
                raise ValueError(f"RGB {case.get('id')} blind assertion failed")
            if _FORBIDDEN_SCORE_KEYS & set(case):
                raise ValueError(f"RGB {case.get('id')} contains gold fields")
            query = str(case["query"])
            candidates = [dict(value) for value in case["candidates"]]
            texts = [str(candidate["text"]) for candidate in candidates]
            vector_start = len(flattened)
            flattened.extend(texts)
            flattened.append(query)
            pair_start = len(pairs)
            prompts = [query] + [
                f"{ROLE_QUERIES[role]} Question: {query}" for role in ROLE_NAMES
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
                        (1.0 - ROLE_RELEVANCE_MIX) * calibrated[role_index + 1, index]
                        + ROLE_RELEVANCE_MIX * cross[index]
                    )
                    for role_index, role in enumerate(ROLE_NAMES)
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
            scored_candidates = [
                {
                    "id": str(candidate["id"]),
                    "source_id": str(candidate["source_id"]),
                    "token_count": int(candidate["token_count"]),
                }
                for candidate in candidates
            ]
            results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": str(case["dataset_id"]),
                    "capability": str(case["capability"]),
                    "id": str(case["id"]),
                    "required_roles": list(ROLE_NAMES),
                    "candidates": scored_candidates,
                    "candidate_scores": candidate_scores,
                    "gold_fields_visible_to_scorer": False,
                }
            )
        return results


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"RGB scoring lock already exists: {path}") from exc


def score_cases_resumable(
    cases: list[dict[str, Any]],
    scorer: Any,
    output_path: Path,
    *,
    case_batch_size: int = 8,
) -> list[dict[str, Any]]:
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    descriptor = _acquire_lock(lock_path)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        existing = list(read_jsonl(output_path)) if output_path.exists() else []
        expected_ids = [str(case["id"]) for case in cases]
        existing_ids = [str(row.get("id", "")) for row in existing]
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise ValueError("RGB resumable score cache is not a valid case prefix")
        if len(existing) > len(cases):
            raise ValueError("RGB score cache contains too many rows")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for start in range(len(existing), len(cases), case_batch_size):
                batch = cases[start : start + case_batch_size]
                scored = scorer.score_cases(batch)
                if [row.get("id") for row in scored] != [
                    row.get("id") for row in batch
                ]:
                    raise ValueError("RGB scorer changed case order or coverage")
                for row in scored:
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                existing.extend(scored)
        return existing
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def _merge_candidates(scored: dict[str, Any]) -> list[dict[str, Any]]:
    if scored.get("gold_fields_visible_to_scorer") is not False:
        raise ValueError(f"RGB {scored.get('id')} scorer leakage assertion failed")
    if _FORBIDDEN_SCORE_KEYS & set(scored):
        raise ValueError(f"RGB {scored.get('id')} score cache contains gold fields")
    score_map = {
        str(value["id"]): value for value in scored.get("candidate_scores", [])
    }
    candidates = list(scored.get("candidates", []))
    if len(score_map) != len(candidates):
        raise ValueError(f"RGB {scored.get('id')} candidate score mismatch")
    merged = []
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        value = score_map.get(candidate_id)
        if value is None:
            raise ValueError(f"RGB {scored.get('id')} candidate score missing")
        merged.append(
            {
                "id": candidate_id,
                "source_id": str(candidate["source_id"]),
                "token_count": int(candidate["token_count"]),
                "scores": dict(value["scores"]),
                "role_scores": dict(value["role_scores"]),
            }
        )
    return merged


def _take_with_budget(
    ordered: Iterable[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
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


def _cross_encoder_knapsack(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: str(item["id"]))
    states: dict[tuple[int, int], tuple[float, tuple[str, ...]]] = {
        (0, 0): (0.0, ())
    }
    for candidate in ordered:
        candidate_id = str(candidate["id"])
        cost = int(candidate["token_count"])
        value = float(candidate["scores"]["cross_encoder"])
        updates: dict[tuple[int, int], tuple[float, tuple[str, ...]]] = {}
        for (count, used), (old_value, old_ids) in list(states.items()):
            key = (count + 1, used + cost)
            if key[0] > top_k or key[1] > token_budget:
                continue
            proposed = (old_value + value, old_ids + (candidate_id,))
            current = states.get(key) or updates.get(key)
            if current is None or proposed[0] > current[0] + 1e-12 or (
                abs(proposed[0] - current[0]) <= 1e-12
                and proposed[1] < current[1]
            ):
                updates[key] = proposed
        states.update(updates)
    best_key, best = min(
        states.items(),
        key=lambda item: (
            -item[1][0],
            item[0][1],
            item[1][1],
        ),
    )
    del best_key
    by_id = {str(candidate["id"]): candidate for candidate in candidates}
    return [by_id[candidate_id] for candidate_id in best[1]]


def _coverage_proxy(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    total = 0
    for role in ROLE_NAMES:
        eligible = [
            candidate
            for candidate in remaining
            if total + int(candidate["token_count"]) <= token_budget
        ]
        if not eligible:
            continue
        eligible.sort(
            key=lambda item: (
                -float(item["role_scores"][role]),
                -float(item["scores"]["cross_encoder"]),
                str(item["id"]),
            )
        )
        best = eligible[0]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        if len(selected) >= top_k:
            return selected
    remaining.sort(
        key=lambda item: (
            -float(item["scores"]["cross_encoder"]),
            str(item["id"]),
        )
    )
    return selected + _take_with_budget(
        remaining,
        top_k=top_k - len(selected),
        token_budget=max(0, token_budget - total),
    )


def _old_frc(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    role_best = {role: 0.0 for role in ROLE_NAMES}
    total = 0
    while remaining and len(selected) < top_k:
        eligible = []
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
            gain = 2.0 * sum(improvements) / len(role_best) + float(
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
        for role in ROLE_NAMES:
            role_best[role] = max(
                role_best[role], float(best["role_scores"][role])
            )
    return selected


def _set_objective(selected: Iterable[dict[str, Any]]) -> float:
    rows = list(selected)
    if not rows:
        return 0.0
    role_coverage = np.mean(
        [
            max(float(candidate["role_scores"][role]) for candidate in rows)
            for role in ROLE_NAMES
        ]
    )
    relevance = sum(
        float(candidate["scores"]["cross_encoder"]) for candidate in rows
    )
    return 2.0 * float(role_coverage) + relevance


def _new_cost_aware_frc(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    total = 0
    current = 0.0
    while remaining and len(selected) < top_k:
        eligible = []
        for candidate in remaining:
            cost = int(candidate["token_count"])
            if total + cost > token_budget:
                continue
            marginal = _set_objective([*selected, candidate]) - current
            eligible.append(
                (
                    marginal / cost,
                    marginal,
                    float(candidate["scores"]["cross_encoder"]),
                    cost,
                    str(candidate["id"]),
                    candidate,
                )
            )
        if not eligible:
            break
        eligible.sort(
            key=lambda value: (
                -value[0],
                -value[1],
                -value[2],
                value[3],
                value[4],
            )
        )
        best = eligible[0][5]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        current = _set_objective(selected)

    alternatives: list[list[dict[str, Any]]] = [selected]
    alternatives.extend(
        [candidate]
        for candidate in candidates
        if int(candidate["token_count"]) <= token_budget
    )
    alternatives.sort(
        key=lambda rows: (
            -_set_objective(rows),
            sum(int(item["token_count"]) for item in rows),
            tuple(sorted(str(item["id"]) for item in rows)),
        )
    )
    return alternatives[0]


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
            key=lambda item: (
                -float(item["scores"][score_name]),
                str(item["id"]),
            ),
        )
        return _take_with_budget(
            ordered, top_k=top_k, token_budget=token_budget
        )
    if method == "cross_encoder_density":
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item["scores"]["cross_encoder"])
                / int(item["token_count"]),
                -float(item["scores"]["cross_encoder"]),
                int(item["token_count"]),
                str(item["id"]),
            ),
        )
        return _take_with_budget(
            ordered, top_k=top_k, token_budget=token_budget
        )
    if method == "cross_encoder_knapsack":
        return _cross_encoder_knapsack(
            candidates, top_k=top_k, token_budget=token_budget
        )
    if method == "coverage_greedy_proxy":
        return _coverage_proxy(
            candidates, top_k=top_k, token_budget=token_budget
        )
    if method == OLD_FRC:
        return _old_frc(candidates, top_k=top_k, token_budget=token_budget)
    if method == NEW_FRC:
        return _new_cost_aware_frc(
            candidates, top_k=top_k, token_budget=token_budget
        )
    raise ValueError(f"unsupported RGB selector: {method}")


def _metrics(
    selected: list[dict[str, Any]],
    gold: dict[str, Any],
    *,
    token_budget: int,
) -> dict[str, float | int]:
    candidate_gold = gold["candidate_gold"]
    selected_labels = [candidate_gold[str(item["id"])] for item in selected]
    selected_count = len(selected)
    positive_count = sum(
        "positive" in value["labels"] for value in selected_labels
    )
    negative_count = sum(
        "negative" in value["labels"] for value in selected_labels
    )
    wrong_count = sum(
        "positive_wrong" in value["labels"] for value in selected_labels
    )
    covered = {
        unit
        for value in selected_labels
        for unit in value["gold_unit_ids"]
    }
    total_gold = len(gold["gold_unit_ids"])
    precision = positive_count / selected_count if selected_count else 0.0
    recall = len(covered) / total_gold if total_gold else 0.0
    evidence_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    token_cost = sum(int(item["token_count"]) for item in selected)
    return {
        "evidence_f1": float(evidence_f1),
        "positive_precision": float(precision),
        "gold_unit_recall": float(recall),
        "complete_gold_unit_coverage": float(recall == 1.0),
        "negative_selection_rate": (
            float(negative_count / selected_count) if selected_count else 0.0
        ),
        "positive_wrong_selection_rate": (
            float(wrong_count / selected_count) if selected_count else 0.0
        ),
        "selected_token_cost": token_cost,
        "budget_utilization": float(token_cost / token_budget),
        "budget_shortfall": token_budget - token_cost,
        "selected_count": selected_count,
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
        raise ValueError("RGB scored cache does not cover every valid case")
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in scored:
        case_id = str(row.get("id", ""))
        if case_id in seen or case_id not in gold_by_id:
            raise ValueError(f"RGB scored ID is duplicate or unknown: {case_id}")
        seen.add(case_id)
        candidates = _merge_candidates(row)
        gold = gold_by_id[case_id]
        if set(gold["candidate_gold"]) != {
            str(candidate["id"]) for candidate in candidates
        }:
            raise ValueError(f"RGB {case_id} gold/candidate coverage mismatch")
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_candidates(
                    candidates,
                    method,
                    token_budget=budget,
                )
                cost = sum(int(item["token_count"]) for item in selected)
                if len(selected) > TOP_K or cost > budget:
                    raise AssertionError(f"RGB {case_id} selector violated budget")
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _metrics(selected, gold, token_budget=budget),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": case_id,
                "dataset_id": str(row["dataset_id"]),
                "capability": str(row["capability"]),
                "candidate_count": len(candidates),
                "gold_unit_count": len(gold["gold_unit_ids"]),
                "configurations": configurations,
                "raw_question_answer_or_candidate_text_exported": False,
            }
        )
    if seen != set(gold_by_id):
        raise ValueError("RGB scored cache omitted one or more valid cases")
    return _build_report(evidence, resamples=resamples), evidence


def _mean_method_metrics(
    rows: list[dict[str, Any]], method: str, budget: int
) -> dict[str, float]:
    first = rows[0]["configurations"][str(budget)]["methods"][method]["metrics"]
    return {
        metric: round(
            float(
                np.mean(
                    [
                        float(
                            row["configurations"][str(budget)]["methods"][method][
                                "metrics"
                            ][metric]
                        )
                        for row in rows
                    ]
                )
            ),
            6,
        )
        for metric in first
    }


def _family_means(evidence: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = list(
        evidence[0]["configurations"][str(BUDGETS[0])]["methods"][METHODS[0]][
            "metrics"
        ]
    )
    result: dict[str, dict[str, float]] = {}
    for method in METHODS:
        result[method] = {}
        for metric in metrics:
            dataset_means = []
            for dataset_id, _, _ in DATASETS:
                rows = [row for row in evidence if row["dataset_id"] == dataset_id]
                budget_means = [
                    float(
                        np.mean(
                            [
                                row["configurations"][str(budget)]["methods"][method][
                                    "metrics"
                                ][metric]
                                for row in rows
                            ]
                        )
                    )
                    for budget in BUDGETS
                ]
                dataset_means.append(float(np.mean(budget_means)))
            result[method][metric] = round(float(np.mean(dataset_means)), 6)
    return result


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {"ci_low": round(float(low), 6), "ci_high": round(float(high), 6)}


def _bootstrap(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    matrices: dict[str, np.ndarray] = {}
    for dataset_id, _, _ in DATASETS:
        rows = [row for row in evidence if row["dataset_id"] == dataset_id]
        matrices[dataset_id] = np.asarray(
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
                for row in rows
            ],
            dtype=float,
        )
    observed = np.mean(
        [matrix.mean(axis=(0, 2)) for matrix in matrices.values()], axis=0
    )
    method_index = {method: index for index, method in enumerate(METHODS)}
    strongest_index = max(
        (method_index[method] for method in BASELINES),
        key=lambda index: (observed[index], METHODS[index]),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    old_deltas = np.empty(resamples, dtype=float)
    simultaneous = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_dataset_means = []
        for matrix in matrices.values():
            sample = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            sampled_dataset_means.append(matrix[sample].mean(axis=(0, 2)))
        method_means = np.mean(sampled_dataset_means, axis=0)
        new_mean = float(method_means[method_index[NEW_FRC]])
        old_deltas[index] = new_mean - float(method_means[method_index[OLD_FRC]])
        simultaneous[index] = new_mean - max(
            float(method_means[method_index[method]]) for method in BASELINES
        )
    return {
        "observed_strongest_baseline": METHODS[strongest_index],
        "method_primary_means": {
            method: round(float(observed[index]), 6)
            for index, method in enumerate(METHODS)
        },
        "cost_aware_minus_old_frc": {
            "point": round(
                float(
                    observed[method_index[NEW_FRC]]
                    - observed[method_index[OLD_FRC]]
                ),
                6,
            ),
            **_percentile_interval(old_deltas),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        },
        "cost_aware_minus_observed_strongest": {
            "baseline": METHODS[strongest_index],
            "point": round(
                float(
                    observed[method_index[NEW_FRC]] - observed[strongest_index]
                ),
                6,
            ),
        },
        "cost_aware_minus_bootstrap_strongest_simultaneous": {
            **_percentile_interval(simultaneous),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        },
    }


def _build_report(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("RGB evidence is empty")
    by_dataset_budget: dict[str, Any] = {}
    dataset_budget_deltas: dict[str, float] = {}
    for dataset_id, capability, _ in DATASETS:
        rows = [row for row in evidence if row["dataset_id"] == dataset_id]
        by_dataset_budget[dataset_id] = {
            "capability": capability,
            "cases": len(rows),
            "budgets": {},
        }
        for budget in BUDGETS:
            aggregates = {
                method: _mean_method_metrics(rows, method, budget)
                for method in METHODS
            }
            strongest = max(
                BASELINES,
                key=lambda method: (aggregates[method][PRIMARY], method),
            )
            delta = (
                aggregates[NEW_FRC][PRIMARY] - aggregates[strongest][PRIMARY]
            )
            dataset_budget_deltas[f"{dataset_id}::{budget}"] = round(delta, 6)
            by_dataset_budget[dataset_id]["budgets"][str(budget)] = {
                "strongest_baseline": strongest,
                "cost_aware_minus_strongest": round(delta, 6),
                "methods": aggregates,
            }

    family = _family_means(evidence)
    comparison = _bootstrap(evidence, resamples=resamples)
    fact_budgets = by_dataset_budget["rgb_en_fact"]["budgets"]
    wrong_delta = float(
        np.mean(
            [
                fact_budgets[str(budget)]["methods"][NEW_FRC][
                    "positive_wrong_selection_rate"
                ]
                - fact_budgets[str(budget)]["methods"][OLD_FRC][
                    "positive_wrong_selection_rate"
                ]
                for budget in BUDGETS
            ]
        )
    )
    worst_dataset_budget = min(dataset_budget_deltas.values())
    old = comparison["cost_aware_minus_old_frc"]
    strongest = comparison["cost_aware_minus_observed_strongest"]
    simultaneous = comparison[
        "cost_aware_minus_bootstrap_strongest_simultaneous"
    ]
    checks = {
        "old_frc_point_at_least_0_01": old["point"] >= 0.01,
        "old_frc_ci_low_above_0": old["ci_low"] > 0.0,
        "strongest_point_at_least_0_01": strongest["point"] >= 0.01,
        "strongest_simultaneous_ci_low_above_0": simultaneous["ci_low"] > 0.0,
        "every_dataset_budget_delta_at_least_minus_0_02": (
            worst_dataset_budget >= -0.02
        ),
        "positive_wrong_rate_increase_at_most_0_02": wrong_delta <= 0.02,
    }
    safety_regression = (
        worst_dataset_budget < -0.02 or wrong_delta > 0.02
    )
    if safety_regression:
        status = "RGB_COST_AWARE_FRC_SAFETY_REGRESSION"
    elif all(checks.values()):
        status = "RGB_COST_AWARE_FRC_SUPPORT_ESTABLISHED"
    else:
        status = "RGB_COST_AWARE_FRC_SUPPORT_NOT_ESTABLISHED"

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "cases": len(evidence),
            "candidate_chunks": sum(row["candidate_count"] for row in evidence),
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "selection_runs": len(evidence) * len(METHODS) * len(BUDGETS),
            "source_revision": "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615",
            "protocol_sha256": PROTOCOL_SHA256,
            "execution_sha256": EXECUTION_SHA256,
            "post_result_correction_sha256": POST_RESULT_CORRECTION_SHA256,
            "deterministic_output_rerun": "2/2 byte-identical",
            "raw_text_committed": False,
            "gold_visible_to_scorer": False,
        },
        "aggregates": {
            "family_equal_dataset_budget_weight": family,
            "dataset_budget": by_dataset_budget,
        },
        "analysis": {
            "primary_metric": PRIMARY,
            "family_comparison": comparison,
            "dataset_budget_deltas": dataset_budget_deltas,
            "worst_dataset_budget_delta": round(worst_dataset_budget, 6),
            "positive_wrong_rate_increase_over_old_frc": round(wrong_delta, 6),
            "positive_wrong_rate_population": (
                "rgb_en_fact valid rows, equal weight across three budgets"
            ),
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
            "independent_confirmation": True,
            "cross_domain_selection_evidence_only": True,
            "gate_evidence": False,
            "raw_question_answer_or_candidate_text_exported": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    analysis = report["analysis"]
    comparison = analysis["family_comparison"]
    outcome = analysis["outcome"]
    lines = [
        "# RGB 成本感知 FRC 盲评报告（v35）",
        "",
        f"- 状态：`{outcome['status']}`",
        f"- 有效案例：{metadata['cases']}；候选分块：{metadata['candidate_chunks']}；选择运行：{metadata['selection_runs']}",
        f"- 最强非 FRC 基线：`{comparison['observed_strongest_baseline']}`",
        f"- 成本感知 FRC - 旧 FRC：{comparison['cost_aware_minus_old_frc']['point']:+.6f}，95% CI [{comparison['cost_aware_minus_old_frc']['ci_low']:+.6f}, {comparison['cost_aware_minus_old_frc']['ci_high']:+.6f}]",
        f"- 成本感知 FRC - 最强基线：{comparison['cost_aware_minus_observed_strongest']['point']:+.6f}，同时 bootstrap 95% CI [{comparison['cost_aware_minus_bootstrap_strongest_simultaneous']['ci_low']:+.6f}, {comparison['cost_aware_minus_bootstrap_strongest_simultaneous']['ci_high']:+.6f}]",
        f"- 最差数据集/预算差值：{analysis['worst_dataset_budget_delta']:+.6f}",
        f"- 错误正例选择率相对旧 FRC 变化：{analysis['positive_wrong_rate_increase_over_old_frc']:+.6f}",
        "",
        "## 等权主指标",
        "",
        "| 方法 | evidence F1 |",
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
            "本实验只检验公开跨域数据上的证据选择可行性，不复现 SetR，不直接证明洪水领域效果，也不改变 Gate 2 的 `NO-GO/SHADOW`、CANARY 或 DEFAULT 状态。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_deterministic_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
    evidence_path = output_dir / "rgb_cost_aware_frc_cases.jsonl.gz"
    json_path = output_dir / "rgb_cost_aware_frc.json"
    markdown_path = output_dir / "rgb_cost_aware_frc.md"
    _write_deterministic_gzip(evidence_path, evidence)
    payload = json.loads(json.dumps(report))
    payload["metadata"]["evidence_artifact"] = {
        "path": evidence_path.name,
        "sha256": sha256(evidence_path),
        "rows": len(evidence),
        "raw_question_answer_or_candidate_text_exported": False,
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
        render_markdown(payload),
        encoding="utf-8",
        newline="\n",
    )
    report.clear()
    report.update(payload)
    return {"json": json_path, "markdown": markdown_path, "evidence": evidence_path}


def load_report(json_path: Path, evidence_path: Path) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    artifact = report["metadata"]["evidence_artifact"]
    if sha256(evidence_path) != artifact["sha256"]:
        raise ValueError("RGB evidence hash mismatch")
    rows = list(read_jsonl(evidence_path))
    if len(rows) != artifact["rows"]:
        raise ValueError("RGB evidence row count mismatch")
    return report


__all__ = [
    "BASELINES",
    "BUDGETS",
    "EXECUTION_SHA256",
    "FrozenRGBScorer",
    "METHODS",
    "NEW_FRC",
    "OLD_FRC",
    "PRIMARY",
    "PROTOCOL_SHA256",
    "evaluate_scored_cases",
    "load_frozen_tokenizer",
    "load_registration",
    "load_report",
    "load_source_rows",
    "prepare_datasets",
    "prepare_row",
    "read_jsonl",
    "score_cases_resumable",
    "select_candidates",
    "sha256",
    "validate_preparation_summary",
    "write_jsonl",
    "write_report",
]
