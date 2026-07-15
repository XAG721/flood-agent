from __future__ import annotations
# ruff: noqa: E402 -- research stays outside the runtime wheel.

import argparse
import sys
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.public_evidence import (
    evidence_metrics,
    paired_bootstrap,
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


def minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    names = ("evidence_recall", "evidence_precision", "evidence_f1", "role_coverage", "token_cost")
    return {
        **{
            name: round(sum(float(row["metrics"][name]) for row in rows) / len(rows), 6)
            for name in names
        },
        "condition_coverage": None,
        "redundancy": None,
        "cases": len(rows),
    }


def build_biencoder_rows(
    source_rows: list[dict[str, Any]],
    model: Any,
    *,
    batch_size: int,
    role_relevance_mix: float,
) -> list[dict[str, Any]]:
    texts: list[str] = []
    spans: list[tuple[int, int, int, int]] = []
    for row in source_rows:
        candidate_start = len(texts)
        texts.extend(str(candidate.get("text", "")) for candidate in row.get("candidates", []))
        candidate_end = len(texts)
        role_start = len(texts)
        texts.extend(f"{query} Question: {row['question']}" for query in ROLE_QUERIES.values())
        role_end = len(texts)
        spans.append((candidate_start, candidate_end, role_start, role_end))
    vectors = np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    output: list[dict[str, Any]] = []
    for source, span in zip(source_rows, spans, strict=True):
        candidate_start, candidate_end, role_start, role_end = span
        passage_vectors = vectors[candidate_start:candidate_end]
        role_vectors = vectors[role_start:role_end]
        role_matrix = passage_vectors @ role_vectors.T
        calibrated = np.column_stack(
            [minmax(role_matrix[:, index]) for index in range(len(ROLE_QUERIES))]
        )
        row = copy.deepcopy(source)
        for index, candidate in enumerate(row.get("candidates", [])):
            dense_relevance = float(candidate.get("scores", {}).get("dense", 0.0))
            candidate["scores"]["cross_encoder"] = dense_relevance
            candidate["role_scores"] = {
                role: float(
                    (1.0 - role_relevance_mix) * calibrated[index, role_index]
                    + role_relevance_mix * dense_relevance
                )
                for role_index, role in enumerate(ROLE_QUERIES)
            }
        output.append(row)
    return output


def evaluate(
    source_rows: list[dict[str, Any]],
    biencoder_rows: list[dict[str, Any]],
    *,
    top_k: int,
    token_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    f1_differences: list[float] = []
    for source, ablated in zip(source_rows, biencoder_rows, strict=True):
        full_selected = select_precomputed(source, "frc_select", k=top_k, budget=token_budget)
        selected = select_precomputed(ablated, "frc_select", k=top_k, budget=token_budget)
        gold = source.get("gold_evidence_ids", [])
        full_metrics = evidence_metrics(gold, [item["id"] for item in full_selected])
        metrics = evidence_metrics(gold, [item["id"] for item in selected])
        full_metrics["role_coverage"] = selected_role_coverage(
            {**source, "selected_evidence": full_selected}
        )
        metrics["role_coverage"] = selected_role_coverage(
            {**ablated, "selected_evidence": selected}
        )
        full_metrics["token_cost"] = sum(int(item.get("token_count", 1)) for item in full_selected)
        metrics["token_cost"] = sum(int(item.get("token_count", 1)) for item in selected)
        f1_differences.append(float(metrics["evidence_f1"] - full_metrics["evidence_f1"]))
        results.append(
            {
                "case_id": source["id"],
                "gold_evidence_ids": gold,
                "selected_ids": [item["id"] for item in selected],
                "full_selected_ids": [item["id"] for item in full_selected],
                "metrics": metrics,
                "full_metrics": full_metrics,
            }
        )
    return results, paired_bootstrap(f1_differences)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a true FRC w/o Reranker ablation using BGE bi-encoder relevance and role scores."
    )
    parser.add_argument("--source-role-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--cache-folder", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--role-relevance-mix", type=float, default=0.15)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer, __version__ as sentence_transformers_version

    source_rows = list(read_jsonl(args.source_role_scores))
    if not source_rows or any(row.get("dataset") != "conditionalqa" for row in source_rows):
        raise ValueError("w/o Reranker ablation currently requires ConditionalQA role-score rows")
    model = SentenceTransformer(
        args.model_name,
        device=args.device,
        cache_folder=str(args.cache_folder) if args.cache_folder else None,
    )
    ablated_rows = build_biencoder_rows(
        source_rows,
        model,
        batch_size=args.batch_size,
        role_relevance_mix=args.role_relevance_mix,
    )
    case_results, paired = evaluate(
        source_rows,
        ablated_rows,
        top_k=args.top_k,
        token_budget=args.token_budget,
    )
    payload = {
        "schema_version": "frc-real-model-ablation-v1",
        "variant": "w/o_reranker",
        "dataset": "conditionalqa",
        "metadata": {
            "model": args.model_name,
            "sentence_transformers_version": sentence_transformers_version,
            "source_role_scores_sha256": sha256(args.source_role_scores),
            "scoring_backend": "bge_biencoder_no_cross_encoder",
            "cross_encoder_used": False,
            "score_calibration": "per_case_minmax",
            "role_relevance_mix": args.role_relevance_mix,
            "selection_parameters": {"alpha": 2.0, "beta": 1.0, "gamma": 0.0},
            "top_k": args.top_k,
            "token_budget": args.token_budget,
            "protocol": (
                "replace Cross-Encoder relevance with saved normalized BGE dense relevance and "
                "recompute every role score from normalized BGE passage/role-query embeddings"
            ),
        },
        "aggregate": mean_metrics(case_results),
        "paired_w_o_reranker_minus_full_evidence_f1": paired,
        "case_results": case_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(json.dumps(payload["aggregate"], ensure_ascii=False))
    print(json.dumps(paired, ensure_ascii=False))


if __name__ == "__main__":
    main()
