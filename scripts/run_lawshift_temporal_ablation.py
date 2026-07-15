from __future__ import annotations
# ruff: noqa: E402 -- research stays outside the runtime wheel.

import argparse
import sys
import gzip
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.lawshift_temporal_ablation import (
    DEFAULT_DISTRACTOR_ARTICLES,
    DEFAULT_PAIRS_PER_REVISION,
    DEFAULT_SEED,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_TOP_K,
    build_lawshift_ablation_report,
    build_lawshift_temporal_cases,
    build_source_manifest,
    render_lawshift_ablation_markdown,
    score_lawshift_cases,
    select_lawshift_methods,
    sha256,
)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_deterministic_jsonl_gzip(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                compressed.write(line.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pinned LawShift before/after version-applicability retrieval ablation "
            "with the local multilingual BGE reranker."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(".cache/benchmarks/lawshift"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/benchmarks/lawshift/evaluation_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/lawshift_temporal_ablation"),
    )
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--pairs-per-revision", type=int, default=DEFAULT_PAIRS_PER_REVISION)
    parser.add_argument("--distractor-articles", type=int, default=DEFAULT_DISTRACTOR_ARTICLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    args = parser.parse_args()

    source_manifest = build_source_manifest(args.input_root)
    cases = build_lawshift_temporal_cases(
        args.input_root,
        pairs_per_revision=args.pairs_per_revision,
        distractor_articles=args.distractor_articles,
        seed=args.seed,
    )
    config = {
        "reranker_model": args.reranker_model,
        "device": args.device,
        "hf_home": str(args.hf_home.resolve()),
        "rerank_batch_size": args.rerank_batch_size,
        "pairs_per_revision": args.pairs_per_revision,
        "distractor_articles": args.distractor_articles,
        "seed": args.seed,
        "top_k": args.top_k,
        "token_budget": args.token_budget,
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = args.cache_dir / "source_manifest.json"
    _write_text(source_manifest_path, json.dumps(source_manifest, ensure_ascii=False, indent=2))
    scored_path = args.cache_dir / "scored_cases.jsonl"
    scored_metadata_path = args.cache_dir / "scored_cases.metadata.json"
    scored = score_lawshift_cases(
        cases,
        output_path=scored_path,
        metadata_path=scored_metadata_path,
        source_manifest_sha256=source_manifest["sha256"],
        config=config,
    )
    selected = select_lawshift_methods(
        scored,
        top_k=args.top_k,
        token_budget=args.token_budget,
    )
    report = build_lawshift_ablation_report(
        cases=cases,
        selected_rows=selected,
        config=config,
        source_manifest=source_manifest,
        scored_path=scored_path,
    )
    report["metadata"]["case_manifest_sha256"] = sha256(scored_metadata_path)
    case_results = report.pop("case_results")
    case_results_path = args.output_dir / "lawshift_temporal_ablation_cases.jsonl.gz"
    _write_deterministic_jsonl_gzip(case_results_path, case_results)
    report["case_results_artifact"] = {
        "file": case_results_path.name,
        "format": "gzip-jsonl",
        "rows": len(case_results),
        "sha256": sha256(case_results_path),
    }
    json_path = args.output_dir / "lawshift_temporal_ablation.json"
    markdown_path = args.output_dir / "lawshift_temporal_ablation.md"
    _write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _write_text(markdown_path, render_lawshift_ablation_markdown(report))
    print(json_path)
    print(markdown_path)
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
