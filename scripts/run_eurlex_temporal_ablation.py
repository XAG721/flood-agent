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

from research.frc_rag.eurlex_temporal_ablation import (
    DEFAULT_DISTRACTOR_PAIRS,
    DEFAULT_SEED,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_TOP_K,
    build_eurlex_ablation_report,
    build_eurlex_temporal_cases,
    load_eurlex_source,
    render_eurlex_ablation_markdown,
    score_eurlex_cases,
    select_eurlex_methods,
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
            "Run a frozen EUR-Lex effective/expiry applicability retrieval ablation "
            "with the local multilingual BGE reranker."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/eurlex_temporal_selection.json"),
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=Path(".cache/benchmarks/eurlex_temporal/documents"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/benchmarks/eurlex_temporal/evaluation_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/eurlex_temporal_ablation"),
    )
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument(
        "--distractor-pairs", type=int, default=DEFAULT_DISTRACTOR_PAIRS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    args = parser.parse_args()

    source_manifest, source_pairs = load_eurlex_source(
        args.manifest, args.documents_dir
    )
    cases = build_eurlex_temporal_cases(
        source_pairs,
        distractor_pairs=args.distractor_pairs,
        seed=args.seed,
    )
    config = {
        "reranker_model": args.reranker_model,
        "device": args.device,
        "hf_home": str(args.hf_home.resolve()),
        "rerank_batch_size": args.rerank_batch_size,
        "distractor_pairs": args.distractor_pairs,
        "seed": args.seed,
        "top_k": args.top_k,
        "token_budget": args.token_budget,
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.cache_dir / "scored_cases.jsonl"
    scored_metadata_path = args.cache_dir / "scored_cases.metadata.json"
    scored = score_eurlex_cases(
        cases,
        output_path=scored_path,
        metadata_path=scored_metadata_path,
        source_manifest_sha256=sha256(args.manifest),
        config=config,
    )
    selected = select_eurlex_methods(
        scored,
        top_k=args.top_k,
        token_budget=args.token_budget,
    )
    report = build_eurlex_ablation_report(
        cases=cases,
        selected_rows=selected,
        config=config,
        source_manifest=source_manifest,
        source_manifest_path=args.manifest,
        scored_path=scored_path,
    )
    report["metadata"]["case_manifest_sha256"] = sha256(scored_metadata_path)
    case_results = report.pop("case_results")
    case_results_path = args.output_dir / "eurlex_temporal_ablation_cases.jsonl.gz"
    _write_deterministic_jsonl_gzip(case_results_path, case_results)
    report["case_results_artifact"] = {
        "file": case_results_path.name,
        "format": "gzip-jsonl",
        "rows": len(case_results),
        "sha256": sha256(case_results_path),
    }
    json_path = args.output_dir / "eurlex_temporal_ablation.json"
    markdown_path = args.output_dir / "eurlex_temporal_ablation.md"
    _write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _write_text(markdown_path, render_eurlex_ablation_markdown(report))
    print(json_path)
    print(markdown_path)
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
