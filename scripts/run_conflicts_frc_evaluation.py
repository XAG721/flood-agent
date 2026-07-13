from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from flood_system.frc_conflicts_evaluation import (
    LocalConflictClassifier,
    build_conflicts_report,
    load_conflicts_cases,
    render_conflicts_markdown,
    score_conflicts_cases,
    select_conflicts_evidence,
    sha256,
    stable_hash,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real-model FRC retrieval and conflict-type audit on Google CONFLICTS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cache/benchmarks/rag_conflicts/conflicts.jsonl"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/benchmarks/rag_conflicts/evaluation_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conflicts_frc"),
    )
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--generator-model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--generator-batch-size", type=int, default=16)
    args = parser.parse_args()

    input_hash = sha256(args.input)
    config = {
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "reranker_model": "BAAI/bge-reranker-large",
        "generator_model": "Qwen2.5-7B-Instruct-GPTQ-Int4",
        "device": args.device,
        "hf_home": str(args.hf_home.resolve()),
        "embedding_batch_size": args.embedding_batch_size,
        "rerank_batch_size": args.rerank_batch_size,
        "generator_batch_size": args.generator_batch_size,
        "score_calibration": "per_case_minmax",
        "role_relevance_mix": 0.15,
        "alpha": 2.0,
        "beta": 1.0,
        "gamma": 0.0,
        "top_k": 5,
        "token_budget": 1500,
        "sample_size": args.sample_size,
    }
    cases = load_conflicts_cases(args.input, sample_size=args.sample_size)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.cache_dir / "scored_cases.jsonl"
    scored_metadata = args.cache_dir / "scored_cases.metadata.json"
    scored = score_conflicts_cases(
        cases,
        output_path=scored_path,
        metadata_path=scored_metadata,
        input_hash=input_hash,
        config=config,
    )
    selected = select_conflicts_evidence(
        scored,
        top_k=config["top_k"],
        token_budget=config["token_budget"],
    )
    selected_path = args.cache_dir / "selected_evidence.jsonl"
    write_jsonl(selected_path, selected)

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    classifier_key = stable_hash(
        {
            "input_sha256": input_hash,
            "selected_sha256": sha256(selected_path),
            "model_path": str(args.generator_model_path.resolve()),
            "labels": "CONFLICTS-v1-five-labels",
        }
    )
    classifier = LocalConflictClassifier(
        model_path=args.generator_model_path,
        batch_size=args.generator_batch_size,
    )
    predictions_path = args.cache_dir / "conflict_predictions.jsonl"
    predictions = classifier.classify(
        selected,
        output_path=predictions_path,
        cache_key=classifier_key,
    )
    # A resumed run may observe duplicate append-only checkpoints after an external
    # process interruption. Rewrite the completed key set once, in canonical order.
    write_jsonl(predictions_path, predictions)
    report = build_conflicts_report(
        dataset_path=args.input,
        scored_path=scored_path,
        selected_rows=selected,
        predictions=predictions,
        config=config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "conflicts_frc_report.json"
    markdown_path = args.output_dir / "conflicts_frc_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_conflicts_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    print(report["decision"]["gate_2"])


if __name__ == "__main__":
    main()
