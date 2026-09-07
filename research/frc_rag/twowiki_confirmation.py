"""Frozen 2WikiMultiHopQA preparation and real-model role scoring.

The module is deliberately separate from the confirmation evaluator: it creates the
previously unseen public score source, while the existing conformal code consumes that
source without changing the model, threshold, or decision rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


DATASET_NAME = "2WikiMultiHopQA"
RAW_DATASET_NAME = "2wikimultihopqa"
ROLE_NAMES = ("condition", "attribution", "procedure", "answer", "exception")
REQUIRED_ROLES = ("procedure", "answer")
ROLE_QUERIES = {
    "condition": (
        "Find evidence stating conditions, thresholds, requirements, limitations, "
        "or triggers."
    ),
    "attribution": (
        "Find evidence stating the responsible entity, authority, source, or actor."
    ),
    "procedure": (
        "Find evidence describing intermediate reasoning, procedure, process, steps, "
        "or bridge entities."
    ),
    "answer": "Find evidence directly supporting the final answer.",
    "exception": (
        "Find evidence stating exceptions, exclusions, insufficiency, not applicable "
        "cases, or unanswerability."
    ),
}
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANKER_REVISION = "55611d7bca2a7133960a6d3b71e083071bbfc312"
EXPECTED_SOURCE_SHA256 = (
    "c0d8b60b9026b728fb07ad74c5252a0f188f6942e8ba5c02df4dfa369502ea8d"
)
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 42
DEFAULT_ROLE_RELEVANCE_MIX = 0.15
DEFAULT_RRF_K = 60


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


def _nested(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (dict, list, tuple)):
        converted = value.tolist()
        if converted is not value:
            return converted
    return value


def _sample_key(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{case_id}".encode("utf-8")).hexdigest()


def stable_sample_rows(
    rows: Iterable[dict[str, Any]],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    seen: set[str] = set()
    keyed: list[tuple[str, str, dict[str, Any]]] = []
    for row in materialized:
        case_id = str(row.get("_id") or "").strip()
        if not case_id or case_id in seen:
            raise ValueError(f"missing or duplicate 2Wiki case id: {case_id!r}")
        seen.add(case_id)
        keyed.append((_sample_key(case_id, seed), case_id, row))
    if len(keyed) < sample_size:
        raise ValueError(
            f"2Wiki dev contains {len(keyed)} rows, below sample_size={sample_size}"
        )
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in keyed[:sample_size]]


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)))


def prepare_case(row: dict[str, Any]) -> dict[str, Any]:
    case_id = str(row.get("_id") or "").strip()
    if not case_id:
        raise ValueError("2Wiki row is missing _id")
    context = _nested(row.get("context")) or []
    supporting_facts = _nested(row.get("supporting_facts")) or []
    gold_pairs = {(str(title), int(sentence_index)) for title, sentence_index in supporting_facts}
    if len(gold_pairs) < 2:
        raise ValueError(f"2Wiki case {case_id} has fewer than two supporting facts")

    candidates: list[dict[str, Any]] = []
    matched_pairs: set[tuple[str, int]] = set()
    for paragraph_index, paragraph in enumerate(context):
        title, sentences = paragraph
        title = str(title)
        sentences = _nested(sentences) or []
        for sentence_index, sentence in enumerate(sentences):
            pair = (title, sentence_index)
            is_gold = pair in gold_pairs
            if is_gold:
                matched_pairs.add(pair)
            text = f"{title}. {str(sentence).strip()}".strip()
            candidates.append(
                {
                    "id": (
                        f"2wiki::{case_id}::p{paragraph_index}::s{sentence_index}"
                    ),
                    "text": text,
                    "source": title,
                    "metadata": {
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "sentence_index": sentence_index,
                    },
                    "gold": is_gold,
                    "gold_roles": list(REQUIRED_ROLES) if is_gold else [],
                    "token_count": _token_count(text),
                }
            )
    if matched_pairs != gold_pairs:
        missing = sorted(gold_pairs - matched_pairs)
        raise ValueError(f"2Wiki case {case_id} is missing supporting facts: {missing}")
    if not candidates:
        raise ValueError(f"2Wiki case {case_id} has no context sentences")
    return {
        "dataset": DATASET_NAME,
        "source": "official_dev",
        "id": case_id,
        "question": str(row.get("question") or ""),
        "answer": str(row.get("answer") or ""),
        "question_type": str(row.get("type") or "unspecified"),
        "not_answerable": False,
        "required_roles": list(REQUIRED_ROLES),
        "gold_evidence_ids": [
            candidate["id"] for candidate in candidates if candidate["gold"]
        ],
        "candidates": candidates,
    }


def prepare_cases(
    rows: Iterable[dict[str, Any]],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    return [
        prepare_case(row)
        for row in stable_sample_rows(rows, sample_size=sample_size, seed=seed)
    ]


def load_and_prepare_parquet(
    path: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
) -> list[dict[str, Any]]:
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "2Wiki source hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    import pandas as pd

    frame = pd.read_parquet(path)
    return prepare_cases(
        frame.to_dict(orient="records"),
        sample_size=sample_size,
        seed=seed,
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _minmax(values: np.ndarray) -> np.ndarray:
    if not values.size:
        return values.astype(float)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values.astype(float) - low) / (high - low)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def _bm25(question: str, texts: list[str]) -> np.ndarray:
    documents = [_tokenize(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    average_length = float(np.mean([len(tokens) for tokens in documents])) or 1.0
    scores: list[float] = []
    for tokens in documents:
        frequencies = Counter(tokens)
        document_length = len(tokens) or 1
        score = 0.0
        for term in _tokenize(question):
            if term not in frequencies:
                continue
            frequency = frequencies[term]
            document_count = len(documents)
            term_document_count = document_frequency.get(term, 0)
            inverse_frequency = math.log(
                1
                + (document_count - term_document_count + 0.5)
                / (term_document_count + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * document_length / average_length
            )
            score += inverse_frequency * (frequency * 2.5) / denominator
        scores.append(score)
    return _minmax(np.asarray(scores, dtype=float))


def _rank_desc(values: np.ndarray) -> list[int]:
    return list(np.argsort(-values, kind="mergesort"))


def _rrf(left: np.ndarray, right: np.ndarray, *, rrf_k: int) -> np.ndarray:
    fused = np.zeros(len(left), dtype=float)
    for ranks in (_rank_desc(left), _rank_desc(right)):
        for rank, index in enumerate(ranks, start=1):
            fused[index] += 1.0 / (rrf_k + rank)
    return _minmax(fused)


def resolve_snapshot(hf_home: Path, model_name: str, revision: str) -> Path:
    model_directory = "models--" + model_name.replace("/", "--")
    snapshot = hf_home / "hub" / model_directory / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"frozen model snapshot is missing: {snapshot}")
    return snapshot


class FrozenBgeScorer:
    def __init__(
        self,
        *,
        hf_home: Path,
        device: str = "cuda",
        embedding_batch_size: int = 32,
        rerank_batch_size: int = 8,
        role_relevance_mix: float = DEFAULT_ROLE_RELEVANCE_MIX,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        os.environ["HF_HOME"] = str(hf_home.resolve())
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder, SentenceTransformer

        embedding_path = resolve_snapshot(
            hf_home, EMBEDDING_MODEL, EMBEDDING_REVISION
        )
        reranker_path = resolve_snapshot(hf_home, RERANKER_MODEL, RERANKER_REVISION)
        self.embedder = SentenceTransformer(str(embedding_path), device=device)
        self.reranker = CrossEncoder(str(reranker_path), device=device)
        self.embedding_batch_size = embedding_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.role_relevance_mix = role_relevance_mix
        self.rrf_k = rrf_k

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
            batch_size=self.rerank_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _minmax(np.asarray(values, dtype=float))

    def score_case(self, case: dict[str, Any]) -> dict[str, Any]:
        candidates = list(case["candidates"])
        texts = [str(candidate["text"]) for candidate in candidates]
        question = str(case["question"])
        bm25 = _bm25(question, texts)
        candidate_vectors = self._encode(texts)
        question_vector = self._encode([question])[0]
        dense = _minmax(candidate_vectors @ question_vector)
        hybrid = _rrf(bm25, dense, rrf_k=self.rrf_k)
        cross = self._predict([(question, text) for text in texts])

        role_scores = [{role: 0.0 for role in ROLE_NAMES} for _ in texts]
        for role in ROLE_NAMES:
            role_question = f"{ROLE_QUERIES[role]} Question: {question}"
            role_values = self._predict(
                [(role_question, text) for text in texts]
            )
            for index, value in enumerate(role_values):
                role_scores[index][role] = float(
                    (1.0 - self.role_relevance_mix) * value
                    + self.role_relevance_mix * cross[index]
                )

        scored_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            scored = dict(candidate)
            scored["scores"] = {
                "bm25": float(bm25[index]),
                "dense": float(dense[index]),
                "hybrid": float(hybrid[index]),
                "cross_encoder": float(cross[index]),
            }
            scored["role_scores"] = role_scores[index]
            scored_candidates.append(scored)
        return {**case, "candidates": scored_candidates}


def score_cases_resumable(
    prepared_path: Path,
    output_path: Path,
    *,
    scorer: FrozenBgeScorer,
    progress: Callable[[int, int, str], None] | None = None,
) -> int:
    cases = read_jsonl(prepared_path)
    if not cases:
        raise ValueError("prepared 2Wiki source is empty")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    completed_ids: set[str] = set()
    if partial_path.is_file():
        partial_rows = read_jsonl(partial_path)
        completed_ids = {str(row["id"]) for row in partial_rows}
        if len(completed_ids) != len(partial_rows):
            raise ValueError("partial score file contains duplicate ids")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            if case_id not in completed_ids:
                scored = scorer.score_case(case)
                handle.write(
                    json.dumps(
                        scored,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
                handle.flush()
                completed_ids.add(case_id)
            if progress is not None:
                progress(index, len(cases), case_id)
    if len(completed_ids) != len(cases):
        raise AssertionError("scored case count does not match prepared case count")
    partial_path.replace(output_path)
    return len(cases)


def preparation_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [len(case["candidates"]) for case in cases]
    gold_counts = [len(case["gold_evidence_ids"]) for case in cases]
    question_types = Counter(case["question_type"] for case in cases)
    return {
        "case_count": len(cases),
        "selected_case_ids_sha256": canonical_json_sha256(
            [case["id"] for case in cases]
        ),
        "candidate_count": {
            "total": sum(candidate_counts),
            "minimum": min(candidate_counts),
            "maximum": max(candidate_counts),
            "mean": round(float(np.mean(candidate_counts)), 6),
        },
        "gold_evidence_count": {
            "total": sum(gold_counts),
            "minimum": min(gold_counts),
            "maximum": max(gold_counts),
            "mean": round(float(np.mean(gold_counts)), 6),
        },
        "question_types": dict(sorted(question_types.items())),
    }

