"""Prospective HoVer dynamic atomic-role experiment (v41).

The module deliberately separates a gold-free mechanism pilot from the
confirmation partition.  The confirmation adapter refuses to run until a
committed pilot report passes every preregistered mechanism check.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.frc_rag.hover_verification_roles import (
    VERIFICATION_ROLE_PROMPTS,
    _bm25,
    _chunk_document,
    _cross_encoder_knapsack,
    _digest,
    _minmax,
    _normalize,
    _rrf,
    _take_with_budget,
    extract_official_candidates,
    inspect_database,
    load_articles,
)
from research.frc_rag.rgb_cost_aware_frc import (
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


SCHEMA_VERSION = "frc-hover-dynamic-atomic-roles-v41"
PROTOCOL_SHA256 = (
    "42486a98c01fd0b9663d3eb9bd0de63634d91605eed06060ac1236ad26997e86"
)
V40_CLOSURE_SHA256 = (
    "ec98358d72453f85014f8f87e4712e06fc4a1cb8ea63415160e9c27f3cbfac73"
)
SOURCE_HASHES = {
    "dataset": "1f1cd57abd616fa00c70bdc575ce77c16fc6cf1a6cffd5ff87c208030a336bb6",
    "tfidf": "45e03a5e47ce8076981eefc86f28d764f4c2eddf1b94b69d9263675260108462",
    "database": "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f",
}
DATASET_ID = "hover_train_release_v1.1_v41"
CAPABILITY = "multi_document_claim_evidence_selection"

V40_SALT = "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V40|"
V41_SALT = "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V41|"
V40_EXCLUDED = 2512
PILOT_CASES = 512
CONFIRMATION_CASES = 2000

STATIC_ROLES = tuple(VERIFICATION_ROLE_PROMPTS)
DYNAMIC_ROLES = (
    "anchor",
    "first_fact",
    "second_fact_or_bridge",
    "counterevidence",
)
DYNAMIC_FALLBACK_PROMPTS = {
    "anchor": "Find evidence identifying the entities and attributes in this claim.",
    "first_fact": "Find evidence for the first verifiable proposition in this claim.",
    "second_fact_or_bridge": (
        "Find evidence for another proposition or an intermediate relation in "
        "this claim."
    ),
    "counterevidence": (
        "Find evidence that could contradict the key relation asserted by this "
        "claim."
    ),
}

BASELINES = (
    "bm25_topk",
    "dense_topk",
    "hybrid_topk",
    "cross_encoder_topk",
    "cross_encoder_knapsack",
)
STATIC_THRESHOLD = "static_threshold_frc_v39_replay"
STATIC_RANK = "static_rank_coverage_frc_v41"
DYNAMIC_RANK = "dynamic_rank_coverage_frc_v41"
METHODS = (*BASELINES, STATIC_THRESHOLD, STATIC_RANK, DYNAMIC_RANK)
PRIMARY = "document_evidence_f1"
TOP_K = 5
BUDGETS = (512, 1024, 1500)
ROLE_THRESHOLD = 0.55
RANK_COVERAGE_WEIGHT = 0.5
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260801

_WHITESPACE = re.compile(r"\s+")
_FORBIDDEN = {
    "article",
    "candidate_ceiling",
    "candidate_gold",
    "evidence",
    "gold",
    "gold_document_titles",
    "gold_fields",
    "gold_unit_ids",
    "hop_count",
    "label",
    "num_hops",
    "source_gold",
    "supporting_facts",
    "title",
}
_BOUNDARY_ASSERTIONS = {
    "gold_fields_joined",
    "gold_fields_visible_to_generator",
    "gold_fields_visible_to_scorer",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON object array: {path}")
    return value


def validate_registration(
    protocol_path: Path,
    closure_path: Path,
    dataset_path: Path,
    retrieval_path: Path,
    database_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("v41 protocol changed after registration")
    if sha256(closure_path) != V40_CLOSURE_SHA256:
        raise ValueError("v40 integrity closure changed after registration")
    actual = {
        "dataset": sha256(dataset_path),
        "tfidf": sha256(retrieval_path),
        "database": sha256(database_path),
    }
    if actual != SOURCE_HASHES:
        raise ValueError(f"v41 source hash mismatch: {actual}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if closure.get("status") != "REGISTRATION_BOUNDARY_VIOLATED_BEFORE_SCORING":
        raise ValueError("v40 closure status changed")
    if protocol["integrity_lineage"]["v40_algorithm_inherited_without_change"] is not True:
        raise ValueError("v41 algorithm inheritance is not frozen")
    return protocol, closure


def build_partitions(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    uids = [str(row.get("uid", "")).strip() for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("HoVer train UIDs must be non-empty and unique")
    v40_order = sorted(
        uids,
        key=lambda uid: (
            hashlib.sha256((V40_SALT + uid).encode("utf-8")).hexdigest(),
            uid,
        ),
    )
    excluded = v40_order[:V40_EXCLUDED]
    excluded_set = set(excluded)
    eligible = sorted(
        (uid for uid in uids if uid not in excluded_set),
        key=lambda uid: (
            hashlib.sha256((V41_SALT + uid).encode("utf-8")).hexdigest(),
            uid,
        ),
    )
    if len(eligible) < PILOT_CASES + CONFIRMATION_CASES:
        raise ValueError("insufficient unexposed HoVer v41 IDs")
    return {
        "v40_excluded": excluded,
        "eligible": eligible,
        "pilot": eligible[:PILOT_CASES],
        "confirmation": eligible[PILOT_CASES : PILOT_CASES + CONFIRMATION_CASES],
    }


def partition_commitments(partitions: dict[str, list[str]]) -> dict[str, Any]:
    return {
        name: {
            "count": len(ids),
            "sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        }
        for name, ids in partitions.items()
    }


def _selected_rows(
    rows: Sequence[dict[str, Any]],
    *,
    public_key: str,
    ordered_ids: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = set(ordered_ids)
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        public_id = str(row.get(public_key, "")).strip()
        if public_id in wanted:
            if public_id in selected:
                raise ValueError(f"duplicate selected public id: {public_id}")
            selected[public_id] = row
    if set(selected) != wanted:
        missing = sorted(wanted - set(selected))
        raise ValueError(f"selected public IDs missing: {missing[:3]}")
    return [selected[public_id] for public_id in ordered_ids]


def prepare_blind_partition(
    dataset_rows: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
    ordered_ids: Sequence[str],
    articles: dict[str, str],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_dataset = _selected_rows(
        dataset_rows,
        public_key="uid",
        ordered_ids=ordered_ids,
    )
    selected_retrieval = _selected_rows(
        retrieval_rows,
        public_key="id",
        ordered_ids=ordered_ids,
    )
    retrieval, _ = extract_official_candidates(selected_retrieval)
    prepared: list[dict[str, Any]] = []
    requested = 0
    resolved = 0
    chunk_counts: list[int] = []
    source_counts: list[int] = []
    token_counts: list[int] = []
    exclusions: Counter[str] = Counter()
    for row in selected_dataset:
        public_id = str(row.get("uid", "")).strip()
        claim = _normalize(row.get("claim"))
        official = retrieval.get(public_id)
        if not public_id or not claim or official is None:
            exclusions["missing_id_claim_or_tfidf"] += 1
            continue
        if official["claim"] != claim:
            raise ValueError(f"claim mismatch for v41 {public_id}")
        requested += len(official["titles"])
        resolved_rows = {
            title: articles[title]
            for title in official["titles"]
            if title in articles
        }
        resolved += len(resolved_rows)
        if len(resolved_rows) < 2:
            exclusions["fewer_than_two_resolved_documents"] += 1
            continue
        case_id = f"hover-v41::{public_id}"
        candidates: list[dict[str, Any]] = []
        for source_index, title in enumerate(sorted(resolved_rows, key=_digest)):
            source_id = f"s{source_index:04d}"
            candidates.extend(
                _chunk_document(
                    case_id=case_id,
                    source_id=source_id,
                    text=f"{title}\n{resolved_rows[title]}",
                    tokenizer=tokenizer,
                )
            )
        sources = {str(candidate["source_id"]) for candidate in candidates}
        if len(sources) < 2:
            exclusions["fewer_than_two_resolved_documents"] += 1
            continue
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "id": case_id,
            "query": claim,
            "candidates": candidates,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_key(blind):
            raise AssertionError("gold or source identity leaked into v41 blind cache")
        prepared.append(blind)
        chunk_counts.append(len(candidates))
        source_counts.append(len(sources))
        token_counts.extend(int(candidate["token_count"]) for candidate in candidates)
    if len(prepared) != len(ordered_ids):
        raise ValueError(
            "v41 partition preparation excludes rows; substitution is forbidden: "
            f"{dict(exclusions)}"
        )

    def distribution(values: Sequence[int]) -> dict[str, float | int]:
        return {
            "total": int(sum(values)),
            "minimum": int(min(values)),
            "mean": round(float(np.mean(values)), 6),
            "maximum": int(max(values)),
        }

    summary = {
        "requested_documents": requested,
        "resolved_documents": resolved,
        "valid_rows": len(prepared),
        "exclusions": dict(sorted(exclusions.items())),
        "case_ids_sha256": canonical_json_sha256(
            [str(case["id"]) for case in prepared]
        ),
        "candidate_chunks": distribution(chunk_counts),
        "candidate_documents": distribution(source_counts),
        "chunk_token_cost": distribution(token_counts),
        "gold_fields_visible_to_scorer": False,
    }
    return prepared, summary


def requested_titles_for_partition(
    retrieval_rows: Sequence[dict[str, Any]], ordered_ids: Sequence[str]
) -> set[str]:
    selected = _selected_rows(
        retrieval_rows,
        public_key="id",
        ordered_ids=ordered_ids,
    )
    _, titles = extract_official_candidates(selected)
    return titles


def build_atomic_prompt(claim: str) -> str:
    return (
        "Decompose the claim below into four short evidence-retrieval queries. "
        "Use only entities and factual assertions already present in the claim. "
        "Do not decide whether the claim is true. Return exactly one JSON object "
        "with string keys anchor, first_fact, second_fact_or_bridge, and "
        "counterevidence. The anchor query identifies entities and attributes; "
        "first_fact retrieves the first independently verifiable proposition; "
        "second_fact_or_bridge retrieves another proposition or intermediate "
        "relation; counterevidence retrieves information that could contradict "
        "the key asserted relation. No markdown or explanation.\n\nClaim: "
        + _normalize(claim)
    )


def parse_atomic_queries(response: str, claim: str) -> tuple[dict[str, str], bool, str]:
    reason = ""
    try:
        value = json.loads(response)
        if not isinstance(value, dict) or set(value) != set(DYNAMIC_ROLES):
            raise ValueError("schema")
        parsed: dict[str, str] = {}
        for role in DYNAMIC_ROLES:
            raw = value[role]
            if not isinstance(raw, str):
                raise ValueError("non_string")
            normalized = _WHITESPACE.sub(" ", raw).strip()[:256]
            if not normalized:
                raise ValueError("empty")
            parsed[role] = normalized
        return parsed, False, reason
    except (json.JSONDecodeError, ValueError) as exc:
        reason = str(exc) or exc.__class__.__name__
    normalized_claim = _normalize(claim)
    return (
        {
            role: f"{DYNAMIC_FALLBACK_PROMPTS[role]} Claim: {normalized_claim}"
            for role in DYNAMIC_ROLES
        },
        True,
        reason,
    )


class LocalQwenAtomicQueryGenerator:
    def __init__(
        self,
        *,
        model_path: Path,
        batch_size: int = 8,
        max_input_tokens: int = 512,
        max_new_tokens: int = 160,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, claims: list[str]) -> list[str]:
        prompts = [build_atomic_prompt(claim) for claim in claims]
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        responses: list[str] = []
        for start in range(0, len(chat_prompts), self.batch_size):
            batch = chat_prompts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            prompt_length = encoded.input_ids.shape[-1]
            responses.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
            )
        return responses


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"v41 query-generation lock exists: {path}") from exc


def generate_queries_resumable(
    cases: Sequence[dict[str, Any]],
    generator: Any,
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
            raise ValueError("v41 query cache is not a valid case prefix")
        if len(existing) > len(cases):
            raise ValueError("v41 query cache has excess rows")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for start in range(len(existing), len(cases), case_batch_size):
                batch = list(cases[start : start + case_batch_size])
                responses = generator.generate([str(case["query"]) for case in batch])
                if len(responses) != len(batch):
                    raise ValueError("v41 generator changed case coverage")
                for case, response in zip(batch, responses, strict=True):
                    atomic, fallback, reason = parse_atomic_queries(
                        response, str(case["query"])
                    )
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "id": str(case["id"]),
                        "claim_sha256": hashlib.sha256(
                            str(case["query"]).encode("utf-8")
                        ).hexdigest(),
                        "prompt_sha256": hashlib.sha256(
                            build_atomic_prompt(str(case["query"])).encode("utf-8")
                        ).hexdigest(),
                        "atomic_queries": atomic,
                        "fallback": fallback,
                        "fallback_reason": reason,
                        "gold_fields_visible_to_generator": False,
                    }
                    if _contains_forbidden_key(row):
                        raise AssertionError("gold leaked into v41 query cache")
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    existing.append(row)
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"generated {len(existing)}/{len(cases)} v41 atomic-query rows",
                    flush=True,
                )
        return existing
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def validate_query_cache(
    cases: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if [row.get("id") for row in rows] != [case.get("id") for case in cases]:
        raise ValueError("v41 query cache coverage/order mismatch")
    fallbacks: Counter[str] = Counter()
    lengths: list[int] = []
    for case, row in zip(cases, rows, strict=True):
        if _contains_forbidden_key(row):
            raise ValueError("v41 query cache contains forbidden field")
        expected = hashlib.sha256(str(case["query"]).encode("utf-8")).hexdigest()
        if row.get("claim_sha256") != expected:
            raise ValueError("v41 query cache claim commitment mismatch")
        atomic = row.get("atomic_queries")
        if not isinstance(atomic, dict) or tuple(atomic) != tuple(sorted(DYNAMIC_ROLES)):
            if not isinstance(atomic, dict) or set(atomic) != set(DYNAMIC_ROLES):
                raise ValueError("v41 query cache schema mismatch")
        for role in DYNAMIC_ROLES:
            query = atomic.get(role)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("v41 query cache contains empty query")
            lengths.append(len(query))
        if row.get("fallback"):
            fallbacks[str(row.get("fallback_reason", "unknown"))] += 1
    fallback_count = sum(fallbacks.values())
    return {
        "rows": len(rows),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(rows), 6) if rows else 0.0,
        "fallback_reasons": dict(sorted(fallbacks.items())),
        "query_length_codepoints": {
            "minimum": min(lengths),
            "mean": round(float(np.mean(lengths)), 6),
            "maximum": max(lengths),
        },
        "gold_fields_visible_to_generator": False,
    }


class FrozenDynamicHoVerScorer:
    def __init__(
        self,
        *,
        query_rows: Sequence[dict[str, Any]],
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
        self.query_by_id = {
            str(row["id"]): dict(row["atomic_queries"]) for row in query_rows
        }

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
        flattened: list[str] = []
        pairs: list[tuple[str, str]] = []
        work: list[dict[str, Any]] = []
        for case in cases:
            if case.get("gold_fields_visible_to_scorer") is not False:
                raise ValueError("v41 blind scorer assertion failed")
            if _contains_forbidden_key(case):
                raise ValueError("v41 scorer received a forbidden field")
            case_id = str(case["id"])
            atomic = self.query_by_id.get(case_id)
            if atomic is None:
                raise ValueError(f"v41 atomic queries missing for {case_id}")
            query = str(case["query"])
            candidates = [dict(value) for value in case["candidates"]]
            texts = [str(candidate["text"]) for candidate in candidates]
            vector_start = len(flattened)
            flattened.extend(texts)
            flattened.append(query)
            prompts = [query]
            prompts.extend(
                f"{VERIFICATION_ROLE_PROMPTS[role]} Claim: {query}"
                for role in STATIC_ROLES
            )
            prompts.extend(str(atomic[role]) for role in DYNAMIC_ROLES)
            pair_start = len(pairs)
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
            start = int(item["vector_start"])
            contexts = vectors[start : start + count]
            query_vector = vectors[start + count]
            dense = _minmax(contexts @ query_vector)
            pair_start = int(item["pair_start"])
            matrix = pair_scores[
                pair_start : pair_start + int(item["prompt_count"]) * count
            ].reshape(int(item["prompt_count"]), count)
            calibrated = np.vstack([_minmax(row) for row in matrix])
            direct = calibrated[0]
            bm25 = _bm25(str(case["query"]), item["texts"])
            hybrid = _rrf(bm25, dense)
            candidate_scores: list[dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                static_scores = {
                    role: float(calibrated[role_index + 1, index])
                    for role_index, role in enumerate(STATIC_ROLES)
                }
                dynamic_offset = 1 + len(STATIC_ROLES)
                dynamic_scores = {
                    role: float(calibrated[dynamic_offset + role_index, index])
                    for role_index, role in enumerate(DYNAMIC_ROLES)
                }
                candidate_scores.append(
                    {
                        "id": str(candidate["id"]),
                        "scores": {
                            "bm25": float(bm25[index]),
                            "dense": float(dense[index]),
                            "hybrid": float(hybrid[index]),
                            "cross_encoder": float(direct[index]),
                        },
                        "static_role_scores": static_scores,
                        "dynamic_role_scores": dynamic_scores,
                    }
                )
            results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": DATASET_ID,
                    "capability": CAPABILITY,
                    "id": str(case["id"]),
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


def merge_scored_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("gold_fields_visible_to_scorer") is not False:
        raise ValueError("v41 scored cache blindness assertion failed")
    if _contains_forbidden_key(row):
        raise ValueError("v41 scored cache contains forbidden field")
    score_map = {
        str(value["id"]): value for value in row.get("candidate_scores", [])
    }
    candidates: list[dict[str, Any]] = []
    for candidate in row.get("candidates", []):
        candidate_id = str(candidate["id"])
        score = score_map.get(candidate_id)
        if score is None:
            raise ValueError("v41 candidate score missing")
        candidates.append(
            {
                "id": candidate_id,
                "source_id": str(candidate["source_id"]),
                "token_count": int(candidate["token_count"]),
                "scores": dict(score["scores"]),
                "static_role_scores": dict(score["static_role_scores"]),
                "dynamic_role_scores": dict(score["dynamic_role_scores"]),
            }
        )
    if len(candidates) != len(score_map):
        raise ValueError("v41 candidate-score cardinality mismatch")
    return candidates


def _static_threshold_select(
    candidates: list[dict[str, Any]], *, token_budget: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    role_best = {role: 0.0 for role in STATIC_ROLES}
    total = 0
    while remaining and len(selected) < TOP_K:
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in remaining:
            cost = int(candidate["token_count"])
            if total + cost > token_budget:
                continue
            improvements: list[float] = []
            for role, old in role_best.items():
                score = float(candidate["static_role_scores"][role])
                if old < ROLE_THRESHOLD <= max(old, score):
                    improvements.append(1.0)
                else:
                    improvements.append(max(0.0, score - old) * 0.25)
            gain = 2.0 * sum(improvements) / len(STATIC_ROLES) + float(
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
        for role in STATIC_ROLES:
            role_best[role] = max(
                role_best[role], float(best["static_role_scores"][role])
            )
    return selected


def _rank_utilities(
    candidates: Sequence[dict[str, Any]], score_key: str, roles: Sequence[str]
) -> dict[str, dict[str, float]]:
    result = {str(candidate["id"]): {} for candidate in candidates}
    for role in roles:
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


def _rank_coverage_select(
    candidates: list[dict[str, Any]],
    *,
    score_key: str,
    roles: Sequence[str],
    token_budget: int,
) -> list[dict[str, Any]]:
    utilities = _rank_utilities(candidates, score_key, roles)
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    best_by_role = {role: 0.0 for role in roles}
    selected_sources: set[str] = set()
    total = 0
    while remaining and len(selected) < TOP_K:
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in remaining:
            source_id = str(candidate["source_id"])
            cost = int(candidate["token_count"])
            if source_id in selected_sources or total + cost > token_budget:
                continue
            role_gain = sum(
                max(
                    0.0,
                    utilities[str(candidate["id"])][role] - best_by_role[role],
                )
                for role in roles
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
        selected_sources.add(str(best["source_id"]))
        for role in roles:
            best_by_role[role] = max(
                best_by_role[role], utilities[str(best["id"])][role]
            )
    return selected


def select_candidates(
    candidates: list[dict[str, Any]], method: str, *, token_budget: int
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
            key=lambda candidate: (
                -float(candidate["scores"][score_name]),
                str(candidate["id"]),
            ),
        )
        return _take_with_budget(
            ordered, top_k=TOP_K, token_budget=token_budget
        )
    if method == "cross_encoder_knapsack":
        return _cross_encoder_knapsack(
            candidates, top_k=TOP_K, token_budget=token_budget
        )
    if method == STATIC_THRESHOLD:
        return _static_threshold_select(candidates, token_budget=token_budget)
    if method == STATIC_RANK:
        return _rank_coverage_select(
            candidates,
            score_key="static_role_scores",
            roles=STATIC_ROLES,
            token_budget=token_budget,
        )
    if method == DYNAMIC_RANK:
        return _rank_coverage_select(
            candidates,
            score_key="dynamic_role_scores",
            roles=DYNAMIC_ROLES,
            token_budget=token_budget,
        )
    raise ValueError(f"unsupported v41 method: {method}")


def _ordinal_ranks(
    candidates: Sequence[dict[str, Any]], score_key: str, role: str
) -> np.ndarray:
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(candidates[index][score_key][role]),
            str(candidates[index]["id"]),
        ),
    )
    ranks = np.empty(len(candidates), dtype=float)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _mean_pairwise_spearman(
    candidates: Sequence[dict[str, Any]], score_key: str, roles: Sequence[str]
) -> float:
    rank_rows = [_ordinal_ranks(candidates, score_key, role) for role in roles]
    values: list[float] = []
    for first in range(len(roles)):
        for second in range(first + 1, len(roles)):
            values.append(float(np.corrcoef(rank_rows[first], rank_rows[second])[0, 1]))
    return float(np.mean(values))


def _distinct_argmax(
    candidates: Sequence[dict[str, Any]], score_key: str, roles: Sequence[str]
) -> int:
    winners = {
        min(
            candidates,
            key=lambda candidate: (
                -float(candidate[score_key][role]),
                str(candidate["id"]),
            ),
        )["id"]
        for role in roles
    }
    return len(winners)


def evaluate_mechanism_gate(
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
) -> dict[str, Any]:
    static_corr: list[float] = []
    dynamic_corr: list[float] = []
    static_distinct: list[int] = []
    dynamic_distinct: list[int] = []
    identities: list[bool] = []
    for row in scored_rows:
        candidates = merge_scored_candidates(row)
        static_corr.append(
            _mean_pairwise_spearman(candidates, "static_role_scores", STATIC_ROLES)
        )
        dynamic_corr.append(
            _mean_pairwise_spearman(
                candidates, "dynamic_role_scores", DYNAMIC_ROLES
            )
        )
        static_distinct.append(
            _distinct_argmax(candidates, "static_role_scores", STATIC_ROLES)
        )
        dynamic_distinct.append(
            _distinct_argmax(candidates, "dynamic_role_scores", DYNAMIC_ROLES)
        )
        for budget in BUDGETS:
            static_ids = [
                str(candidate["id"])
                for candidate in select_candidates(
                    candidates, STATIC_RANK, token_budget=budget
                )
            ]
            dynamic_ids = [
                str(candidate["id"])
                for candidate in select_candidates(
                    candidates, DYNAMIC_RANK, token_budget=budget
                )
            ]
            identities.append(static_ids == dynamic_ids)
    static_mean = float(np.mean(static_corr))
    dynamic_mean = float(np.mean(dynamic_corr))
    static_argmax = float(np.mean(static_distinct))
    dynamic_argmax = float(np.mean(dynamic_distinct))
    identity = float(np.mean(identities))
    checks = {
        "generation_fallback_rate_at_most_0_05": (
            float(query_summary["fallback_rate"]) <= 0.05
        ),
        "dynamic_spearman_reduction_at_least_0_05": (
            static_mean - dynamic_mean >= 0.05
        ),
        "dynamic_distinct_argmax_gain_at_least_0_25": (
            dynamic_argmax - static_argmax >= 0.25
        ),
        "ordered_selection_identity_at_most_0_90": identity <= 0.90,
        "no_forbidden_field_leak": not any(
            _contains_forbidden_key(row) for row in scored_rows
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "frc-hover-dynamic-atomic-mechanism-report-v1",
        "metadata": {
            "experiment_id": "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V41",
            "protocol_sha256": PROTOCOL_SHA256,
            "cases": len(scored_rows),
            "budgets": list(BUDGETS),
            "gold_fields_joined": false_value(),
        },
        "query_generation": query_summary,
        "mechanism": {
            "static_within_family_mean_spearman": round(static_mean, 6),
            "dynamic_within_family_mean_spearman": round(dynamic_mean, 6),
            "spearman_reduction": round(static_mean - dynamic_mean, 6),
            "static_mean_distinct_role_argmax": round(static_argmax, 6),
            "dynamic_mean_distinct_role_argmax": round(dynamic_argmax, 6),
            "distinct_argmax_gain": round(dynamic_argmax - static_argmax, 6),
            "static_dynamic_ordered_selection_identity": round(identity, 6),
        },
        "checks": checks,
        "outcome": {
            "all_mechanism_checks_pass": passed,
            "confirmation_open_authorized": passed,
            "status": (
                "MECHANISM_ESTABLISHED_OPEN_CONFIRMATION"
                if passed
                else "MECHANISM_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
            ),
            "gate_2": "NO-GO/SHADOW",
            "selector_adoption_authorized": false_value(),
        },
    }


def false_value() -> bool:
    """Return False while keeping boundary fields visually explicit."""

    return False


def render_mechanism_report(report: dict[str, Any]) -> str:
    mechanism = report["mechanism"]
    lines = [
        "# HoVer 动态原子核验角色机制门槛（v41）",
        "",
        f"- 状态：`{report['outcome']['status']}`",
        f"- 样例：{report['metadata']['cases']}（未连接 label、hop 或 gold）",
        "- 静态/动态角色内部平均 Spearman："
        f"{mechanism['static_within_family_mean_spearman']:.6f} / "
        f"{mechanism['dynamic_within_family_mean_spearman']:.6f}",
        f"- 相关性下降：{mechanism['spearman_reduction']:+.6f}",
        "- 静态/动态平均不同角色首选数："
        f"{mechanism['static_mean_distinct_role_argmax']:.6f} / "
        f"{mechanism['dynamic_mean_distinct_role_argmax']:.6f}",
        "- 静态与动态有序选择完全相同率："
        f"{mechanism['static_dynamic_ordered_selection_identity']:.6f}",
        "",
        "## 预注册检查",
        "",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`：{'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本报告只验证角色信号是否真正改变排序结构，不使用任何正确性标签，"
            "也不构成检索性能、真实防汛有效性、SetR 复现或 Gate 2 放行证据。",
            "",
        ]
    )
    return "\n".join(lines)


def write_mechanism_report(
    report: dict[str, Any], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hover_dynamic_atomic_roles_pilot.json"
    md_path = output_dir / "hover_dynamic_atomic_roles_pilot.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(
        render_mechanism_report(report), encoding="utf-8", newline="\n"
    )
    return {"json": json_path, "markdown": md_path}


def require_pilot_pass(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outcome = report.get("outcome", {})
    if outcome.get("confirmation_open_authorized") is not True:
        raise RuntimeError("v41 confirmation is closed by the pilot mechanism gate")
    if report.get("metadata", {}).get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("v41 pilot report references another protocol")
    return report


def build_confirmation_gold(
    dataset_rows: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
    ordered_ids: Sequence[str],
    prepared_rows: Sequence[dict[str, Any]],
    resolved_titles: set[str],
) -> list[dict[str, Any]]:
    selected_dataset = _selected_rows(
        dataset_rows, public_key="uid", ordered_ids=ordered_ids
    )
    selected_retrieval = _selected_rows(
        retrieval_rows, public_key="id", ordered_ids=ordered_ids
    )
    retrieval, _ = extract_official_candidates(selected_retrieval)
    prepared_by_id = {str(row["id"]): row for row in prepared_rows}
    gold_rows: list[dict[str, Any]] = []
    for row in selected_dataset:
        public_id = str(row["uid"])
        case_id = f"hover-v41::{public_id}"
        prepared = prepared_by_id.get(case_id)
        if prepared is None:
            raise ValueError("v41 prepared confirmation case missing")
        label = str(row.get("label", ""))
        hop_count = row.get("num_hops")
        supporting = row.get("supporting_facts")
        if (
            label not in {"SUPPORTED", "NOT_SUPPORTED"}
            or hop_count not in {2, 3, 4}
            or not isinstance(supporting, list)
        ):
            raise ValueError("v41 invalid confirmation gold fields")
        gold_titles = {
            _normalize(fact[0])
            for fact in supporting
            if isinstance(fact, list)
            and len(fact) >= 2
            and isinstance(fact[0], str)
            and isinstance(fact[1], int)
            and _normalize(fact[0])
        }
        if not gold_titles:
            raise ValueError("v41 confirmation has no supporting document gold")
        official_titles = [
            title
            for title in retrieval[public_id]["titles"]
            if title in resolved_titles
        ]
        sorted_titles = sorted(official_titles, key=_digest)
        title_to_source = {
            title: f"s{index:04d}" for index, title in enumerate(sorted_titles)
        }
        present = gold_titles & set(sorted_titles)
        gold_rows.append(
            {
                "case_id": case_id,
                "label": label,
                "hop_count": int(hop_count),
                "gold_document_count": len(gold_titles),
                "gold_sources": sorted(
                    title_to_source[title] for title in present
                ),
                "candidate_ceiling": len(present) / len(gold_titles),
                "candidate_ceiling_complete": present == gold_titles,
            }
        )
    return gold_rows


def _selection_metrics(
    selected: Sequence[dict[str, Any]], gold: dict[str, Any]
) -> dict[str, float | int]:
    selected_sources = sorted({str(item["source_id"]) for item in selected})
    gold_sources = set(gold["gold_sources"])
    correct = len(set(selected_sources) & gold_sources)
    precision = correct / len(selected_sources) if selected_sources else 0.0
    recall = correct / int(gold["gold_document_count"])
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "document_precision": precision,
        "document_recall": recall,
        "document_evidence_f1": f1,
        "complete_gold_document_coverage": float(recall == 1.0),
        "selected_chunk_count": len(selected),
        "selected_document_count": len(selected_sources),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "duplicate_document_selection_rate": (
            (len(selected) - len(selected_sources)) / len(selected)
            if selected
            else 0.0
        ),
    }


def evaluate_confirmation(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {str(row["case_id"]): row for row in gold_rows}
    if len(scored_rows) != len(gold_by_id):
        raise ValueError("v41 confirmation score/gold cardinality mismatch")
    evidence: list[dict[str, Any]] = []
    for row in scored_rows:
        case_id = str(row["id"])
        gold = gold_by_id.get(case_id)
        if gold is None:
            raise ValueError("v41 confirmation scored unknown case")
        candidates = merge_scored_candidates(row)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_candidates(
                    candidates, method, token_budget=budget
                )
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(selected, gold),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": case_id,
                "label": str(gold["label"]),
                "hop_count": int(gold["hop_count"]),
                "gold_document_count": int(gold["gold_document_count"]),
                "candidate_ceiling": float(gold["candidate_ceiling"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "configurations": configurations,
                "raw_claim_title_article_or_evidence_text_exported": False,
            }
        )
    return _build_confirmation_report(evidence, resamples=resamples), evidence


def _method_mean(
    rows: Sequence[dict[str, Any]], method: str, budgets: Sequence[int]
) -> float:
    return float(
        np.mean(
            [
                row["configurations"][str(budget)]["methods"][method]["metrics"][
                    PRIMARY
                ]
                for row in rows
                for budget in budgets
            ]
        )
    )


def _bootstrap_confirmation(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    matrix = np.asarray(
        [
            [
                [
                    row["configurations"][str(budget)]["methods"][method][
                        "metrics"
                    ][PRIMARY]
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
    strongest = max(BASELINES, key=lambda method: observed[indices[method]])
    dynamic_static = np.empty(resamples, dtype=float)
    dynamic_strongest = np.empty(resamples, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for index in range(resamples):
        sample = rng.integers(0, len(evidence), size=len(evidence))
        means = matrix[sample].mean(axis=(0, 2))
        dynamic = float(means[indices[DYNAMIC_RANK]])
        dynamic_static[index] = dynamic - float(means[indices[STATIC_RANK]])
        dynamic_strongest[index] = dynamic - max(
            float(means[indices[method]]) for method in BASELINES
        )

    def comparison(values: np.ndarray, point: float) -> dict[str, Any]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {
            "point": round(point, 6),
            "ci_low": round(float(low), 6),
            "ci_high": round(float(high), 6),
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
        }

    return {
        "method_primary_means": {
            method: round(float(observed[index]), 6)
            for index, method in enumerate(METHODS)
        },
        "observed_strongest_non_frc": strongest,
        "dynamic_minus_static_rank": comparison(
            dynamic_static,
            float(observed[indices[DYNAMIC_RANK]] - observed[indices[STATIC_RANK]]),
        ),
        "dynamic_minus_bootstrap_strongest_non_frc": comparison(
            dynamic_strongest,
            float(observed[indices[DYNAMIC_RANK]] - observed[indices[strongest]]),
        ),
    }


def _stratum_rows(
    evidence: Sequence[dict[str, Any]], name: str
) -> list[dict[str, Any]]:
    if name.startswith("label="):
        return [row for row in evidence if row["label"] == name.split("=", 1)[1]]
    if name.endswith("-hop"):
        return [row for row in evidence if row["hop_count"] == int(name[0])]
    if name == "candidate_ceiling_complete":
        return [row for row in evidence if row["candidate_ceiling_complete"]]
    if name == "candidate_ceiling_incomplete":
        return [row for row in evidence if not row["candidate_ceiling_complete"]]
    raise ValueError(f"unknown v41 stratum: {name}")


def _build_confirmation_report(
    evidence: list[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    bootstrap = _bootstrap_confirmation(evidence, resamples=resamples)
    means = bootstrap["method_primary_means"]
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(
            BASELINES,
            key=lambda method: _method_mean(evidence, method, (budget,)),
        )
        budget_deltas[str(budget)] = round(
            _method_mean(evidence, DYNAMIC_RANK, (budget,))
            - _method_mean(evidence, strongest, (budget,)),
            6,
        )
    strata = (
        "label=SUPPORTED",
        "label=NOT_SUPPORTED",
        "2-hop",
        "3-hop",
        "4-hop",
        "candidate_ceiling_complete",
        "candidate_ceiling_incomplete",
    )
    stratum_deltas: dict[str, float | None] = {}
    for name in strata:
        rows = _stratum_rows(evidence, name)
        if not rows:
            stratum_deltas[name] = None
            continue
        strongest = max(
            BASELINES, key=lambda method: _method_mean(rows, method, BUDGETS)
        )
        stratum_deltas[name] = round(
            _method_mean(rows, DYNAMIC_RANK, BUDGETS)
            - _method_mean(rows, strongest, BUDGETS),
            6,
        )
    dynamic_static = bootstrap["dynamic_minus_static_rank"]
    dynamic_strongest = bootstrap[
        "dynamic_minus_bootstrap_strongest_non_frc"
    ]
    checks = {
        "dynamic_minus_static_point_at_least_0_005": (
            dynamic_static["point"] >= 0.005
        ),
        "dynamic_minus_static_ci_low_above_0": dynamic_static["ci_low"] > 0.0,
        "dynamic_minus_strongest_point_at_least_0_01": (
            dynamic_strongest["point"] >= 0.01
        ),
        "dynamic_minus_strongest_ci_low_above_0": (
            dynamic_strongest["ci_low"] > 0.0
        ),
        "every_budget_and_stratum_delta_at_least_minus_0_02": min(
            [
                *budget_deltas.values(),
                *(value for value in stratum_deltas.values() if value is not None),
            ]
        )
        >= -0.02,
    }
    supported = all(checks.values())
    return {
        "schema_version": "frc-hover-dynamic-atomic-confirmation-report-v1",
        "metadata": {
            "experiment_id": "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V41",
            "protocol_sha256": PROTOCOL_SHA256,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "methods": list(METHODS),
            "primary_metric": PRIMARY,
            "raw_text_committed": False,
        },
        "analysis": {
            "family_comparison": bootstrap,
            "budget_deltas": budget_deltas,
            "stratum_deltas": stratum_deltas,
            "support_checks": checks,
            "outcome": {
                "status": (
                    "DYNAMIC_ATOMIC_ROLE_SUPPORT_ESTABLISHED_SAME_FAMILY"
                    if supported
                    else "DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED"
                ),
                "support_established": supported,
                "same_dataset_family_confirmation_only": True,
                "selector_adoption_authorized": False,
                "gate_2": "NO-GO/SHADOW",
                "canary_or_default_authorized": False,
                "setr_reproduced": False,
                "flood_domain_effectiveness_established": False,
            },
        },
        "aggregates": {"method_primary_means": means},
    }


def write_confirmation_report(
    report: dict[str, Any], evidence: list[dict[str, Any]], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hover_dynamic_atomic_roles.json"
    md_path = output_dir / "hover_dynamic_atomic_roles.md"
    evidence_path = output_dir / "hover_dynamic_atomic_roles_cases.jsonl.gz"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    comparison = report["analysis"]["family_comparison"]
    lines = [
        "# HoVer 动态原子核验角色确认（v41）",
        "",
        f"- 状态：`{report['analysis']['outcome']['status']}`",
        f"- 确认样例：{report['metadata']['cases']}",
        "- 动态 - 静态秩覆盖："
        f"{comparison['dynamic_minus_static_rank']['point']:+.6f}，95% CI "
        f"[{comparison['dynamic_minus_static_rank']['ci_low']:+.6f}, "
        f"{comparison['dynamic_minus_static_rank']['ci_high']:+.6f}]",
        "- 动态 - bootstrap 最强非 FRC："
        f"{comparison['dynamic_minus_bootstrap_strongest_non_frc']['point']:+.6f}，"
        "95% CI "
        f"[{comparison['dynamic_minus_bootstrap_strongest_non_frc']['ci_low']:+.6f}, "
        f"{comparison['dynamic_minus_bootstrap_strongest_non_frc']['ci_high']:+.6f}]",
        "",
        "## 方法主指标",
        "",
        "| 方法 | Document Evidence F1 |",
        "|---|---:|",
    ]
    for method, value in report["aggregates"]["method_primary_means"].items():
        lines.append(f"| `{method}` | {value:.6f} |")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本轮只是在 HoVer 同数据集家族的未分析 train 分区上确认。无论结果"
            "是否通过，都不复现 SetR、不证明真实防汛领域效果、不改变 Gate 2，"
            "也不授权 CANARY、DEFAULT 或选择器替换。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    with evidence_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
    return {"json": json_path, "markdown": md_path, "evidence": evidence_path}


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _BOUNDARY_ASSERTIONS:
                if child is not False:
                    return True
                continue
            if normalized in _FORBIDDEN or normalized.startswith("gold_"):
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


__all__ = [
    "BASELINES",
    "BUDGETS",
    "CONFIRMATION_CASES",
    "DYNAMIC_RANK",
    "DYNAMIC_ROLES",
    "FrozenDynamicHoVerScorer",
    "LocalQwenAtomicQueryGenerator",
    "METHODS",
    "PILOT_CASES",
    "PROTOCOL_SHA256",
    "STATIC_RANK",
    "STATIC_THRESHOLD",
    "build_atomic_prompt",
    "build_confirmation_gold",
    "build_partitions",
    "canonical_json_sha256",
    "evaluate_confirmation",
    "evaluate_mechanism_gate",
    "generate_queries_resumable",
    "inspect_database",
    "load_articles",
    "load_frozen_tokenizer",
    "merge_scored_candidates",
    "parse_atomic_queries",
    "partition_commitments",
    "prepare_blind_partition",
    "read_jsonl",
    "requested_titles_for_partition",
    "require_pilot_pass",
    "score_cases_resumable",
    "select_candidates",
    "sha256",
    "validate_query_cache",
    "validate_registration",
    "write_confirmation_report",
    "write_jsonl",
    "write_mechanism_report",
]
