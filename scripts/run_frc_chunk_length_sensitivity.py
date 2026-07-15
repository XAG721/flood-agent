from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from flood_system.frc_public_evidence import (
    evidence_metrics,
    read_jsonl,
    select_precomputed,
    selected_role_coverage,
    sha256,
)


ROLE_QUERIES = {
    "condition": "Find evidence stating conditions, thresholds, requirements, limitations, or triggers.",
    "attribution": "Find evidence stating the responsible entity, authority, source, or actor.",
    "procedure": "Find evidence describing intermediate reasoning, procedure, process, steps, or bridge entities.",
    "answer": "Find evidence directly supporting the final answer.",
    "exception": "Find evidence stating exceptions, exclusions, insufficiency, not applicable cases, or unanswerability.",
}
METHODS = ("cross_encoder_topk", "coverage_greedy_proxy", "frc_select")
METRIC_NAMES = (
    "evidence_recall",
    "evidence_precision",
    "evidence_f1",
    "role_coverage",
    "token_cost",
    "selected_chunk_count",
    "unique_parent_count",
    "duplicate_parent_ratio",
)


def minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def chunk_text(
    text: str,
    tokenizer: Any,
    *,
    chunk_length: int,
    overlap_ratio: float,
) -> list[tuple[str, int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return [(text, 0)]
    overlap = int(round(chunk_length * overlap_ratio))
    step = max(1, chunk_length - overlap)
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start : start + chunk_length]
        decoded = tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        actual_count = len(tokenizer.encode(decoded, add_special_tokens=False))
        if actual_count > chunk_length:
            decoded_ids = tokenizer.encode(decoded, add_special_tokens=False)[:chunk_length]
            decoded = tokenizer.decode(
                decoded_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            actual_count = len(tokenizer.encode(decoded, add_special_tokens=False))
        chunks.append((decoded, actual_count))
        if start + chunk_length >= len(token_ids):
            break
        start += step
    return chunks


def build_chunked_cases(
    source_rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    chunk_length: int,
    overlap_ratio: float,
) -> list[dict[str, Any]]:
    output = []
    for source in source_rows:
        candidates = []
        for parent in source.get("candidates", []):
            chunks = chunk_text(
                str(parent.get("text", "")),
                tokenizer,
                chunk_length=chunk_length,
                overlap_ratio=overlap_ratio,
            )
            for index, (text, token_count) in enumerate(chunks):
                candidates.append(
                    {
                        "id": f"{parent['id']}::chunk-{chunk_length}-{index}",
                        "parent_id": parent["id"],
                        "text": text,
                        "token_count": token_count,
                        "scores": {},
                        "role_scores": {},
                    }
                )
        output.append(
            {
                "dataset": source["dataset"],
                "id": source["id"],
                "question": source["question"],
                "required_roles": list(source.get("required_roles", [])),
                "gold_evidence_ids": list(source.get("gold_evidence_ids", [])),
                "candidates": candidates,
            }
        )
    return output


def _predict(reranker: Any, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray:
    return np.asarray(
        reranker.predict(
            pairs,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=float,
    )


def score_chunked_cases(
    cases: list[dict[str, Any]],
    reranker: Any,
    *,
    batch_size: int,
    role_relevance_mix: float,
) -> None:
    flat_candidates: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    relevance_pairs: list[tuple[str, str]] = []
    for case in cases:
        start = len(flat_candidates)
        flat_candidates.extend(case["candidates"])
        spans.append((start, len(flat_candidates)))
        relevance_pairs.extend((case["question"], candidate["text"]) for candidate in case["candidates"])
    relevance_raw = _predict(reranker, relevance_pairs, batch_size=batch_size)
    role_raw: dict[str, np.ndarray] = {}
    for role, query in ROLE_QUERIES.items():
        pairs = []
        for case in cases:
            role_query = f"{query} Question: {case['question']}"
            pairs.extend((role_query, candidate["text"]) for candidate in case["candidates"])
        role_raw[role] = _predict(reranker, pairs, batch_size=batch_size)
    for case, (start, end) in zip(cases, spans, strict=True):
        relevance = minmax(relevance_raw[start:end])
        roles = {role: minmax(values[start:end]) for role, values in role_raw.items()}
        for index, candidate in enumerate(case["candidates"]):
            candidate["scores"]["cross_encoder"] = float(relevance[index])
            candidate["role_scores"] = {
                role: float(
                    (1.0 - role_relevance_mix) * role_values[index]
                    + role_relevance_mix * relevance[index]
                )
                for role, role_values in roles.items()
            }


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    chunk_length: int,
    top_k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        for method in METHODS:
            selected = select_precomputed(case, method, k=top_k, budget=token_budget)
            parent_ids = list(dict.fromkeys(item["parent_id"] for item in selected))
            metrics = evidence_metrics(case["gold_evidence_ids"], parent_ids)
            metrics["role_coverage"] = selected_role_coverage(
                {**case, "selected_evidence": selected}
            )
            metrics["token_cost"] = sum(int(item["token_count"]) for item in selected)
            metrics["selected_chunk_count"] = len(selected)
            metrics["unique_parent_count"] = len(parent_ids)
            metrics["duplicate_parent_ratio"] = (
                1.0 - len(parent_ids) / len(selected) if selected else 0.0
            )
            if metrics["token_cost"] > token_budget:
                raise ValueError(f"token budget exceeded for {case['id']} {method}")
            output.append(
                {
                    "case_id": case["id"],
                    "chunk_length": chunk_length,
                    "method": method,
                    "gold_evidence_ids": case["gold_evidence_ids"],
                    "selected_chunk_ids": [item["id"] for item in selected],
                    "selected_parent_ids": parent_ids,
                    "metrics": metrics,
                }
            )
    return output


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["chunk_length"]), str(row["method"]))].append(row)
    output = []
    for (chunk_length, method), group in sorted(grouped.items()):
        output.append(
            {
                "chunk_length": chunk_length,
                "method": method,
                "metrics": {
                    **{
                        name: round(
                            sum(float(row["metrics"][name]) for row in group) / len(group),
                            6,
                        )
                        for name in METRIC_NAMES
                    },
                    "cases": len(group),
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-reranker FRC chunk-length sensitivity on ConditionalQA."
    )
    parser.add_argument("--source-role-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reranker-model-id", default="BAAI/bge-reranker-large")
    parser.add_argument("--reranker-model-path", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-lengths", default="64,128,256")
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--role-relevance-mix", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, __version__ as sentence_transformers_version

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    chunk_lengths = [int(value.strip()) for value in args.chunk_lengths.split(",") if value.strip()]
    if len(chunk_lengths) < 2 or any(value <= 0 for value in chunk_lengths):
        raise ValueError("at least two positive chunk lengths are required")
    if not 0 <= args.overlap_ratio < 1:
        raise ValueError("overlap ratio must be in [0, 1)")
    source_rows = list(read_jsonl(args.source_role_scores))
    if not source_rows or any(row.get("dataset") != "conditionalqa" for row in source_rows):
        raise ValueError("chunk-length sensitivity currently requires ConditionalQA role-score rows")
    reranker = CrossEncoder(
        args.reranker_model_path,
        device=args.device,
        max_length=512,
    )
    case_results = []
    chunk_inventory = []
    for chunk_length in chunk_lengths:
        cases = build_chunked_cases(
            source_rows,
            reranker.tokenizer,
            chunk_length=chunk_length,
            overlap_ratio=args.overlap_ratio,
        )
        chunk_inventory.append(
            {
                "chunk_length": chunk_length,
                "cases": len(cases),
                "chunks": sum(len(case["candidates"]) for case in cases),
            }
        )
        score_chunked_cases(
            cases,
            reranker,
            batch_size=args.batch_size,
            role_relevance_mix=args.role_relevance_mix,
        )
        length_results = evaluate_cases(
            cases,
            chunk_length=chunk_length,
            top_k=args.top_k,
            token_budget=args.token_budget,
        )
        case_results.extend(length_results)
        print(json.dumps(aggregate(length_results), ensure_ascii=False))
    payload = {
        "schema_version": "frc-chunk-length-sensitivity-v1",
        "dataset": "conditionalqa",
        "metadata": {
            "reranker_model": args.reranker_model_id,
            "reranker_revision": Path(args.reranker_model_path).name,
            "sentence_transformers_version": sentence_transformers_version,
            "source_role_scores_sha256": sha256(args.source_role_scores),
            "score_calibration": "per_case_minmax",
            "role_relevance_mix": args.role_relevance_mix,
            "selection_parameters": {"alpha": 2.0, "beta": 1.0, "gamma": 0.0},
            "chunk_lengths": chunk_lengths,
            "overlap_ratio": args.overlap_ratio,
            "chunk_tokenizer": args.reranker_model_id,
            "methods": list(METHODS),
            "top_k": args.top_k,
            "token_budget": args.token_budget,
            "seed": args.seed,
            "parent_evidence_protocol": "one or more selected chunks recall their unique parent evidence ID",
            "dense_embedding_status": "not used because frozen gamma=0.0 and all compared selectors use reranker scores",
            "chunk_inventory": chunk_inventory,
        },
        "aggregates": aggregate(case_results),
        "case_results": case_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
