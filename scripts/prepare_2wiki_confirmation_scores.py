from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.twowiki_confirmation import (
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    EXPECTED_SOURCE_SHA256,
    RERANKER_MODEL,
    RERANKER_REVISION,
    FrozenBgeScorer,
    load_and_prepare_parquet,
    preparation_summary,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the pre-registered 2WikiMultiHopQA dev sample and score it "
            "with the frozen local BGE embedder/reranker snapshots."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cache/benchmarks/2wikimultihopqa/dev.parquet"),
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/"
            "candidate_pool_2wikimultihopqa.jsonl"
        ),
    )
    parser.add_argument(
        "--role-scores-output",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/"
            "role_scores_2wikimultihopqa.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/"
            "2wikimultihopqa_provenance.json"
        ),
    )
    parser.add_argument(
        "--hf-home", type=Path, default=Path("D:/RAG_test/.hf_cache")
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"2Wiki dev artifact does not exist: {args.input}")
    if args.force_prepare or not args.prepared_output.is_file():
        cases = load_and_prepare_parquet(
            args.input,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        write_jsonl(args.prepared_output, cases)
    else:
        if sha256(args.input) != EXPECTED_SOURCE_SHA256:
            parser.error("2Wiki dev artifact hash does not match the protocol")
        cases = read_jsonl(args.prepared_output)
    summary = preparation_summary(cases)
    manifest = {
        "schema_version": "frc-2wiki-score-source-provenance-v1",
        "dataset": "2WikiMultiHopQA",
        "split": "dev",
        "source": {
            "path_label": args.input.name,
            "sha256": sha256(args.input),
            "expected_sha256": EXPECTED_SOURCE_SHA256,
        },
        "sampling": {
            "sample_size": args.sample_size,
            "seed": args.seed,
            "algorithm": "sha256 seeded id ordering",
        },
        "preparation": {
            **summary,
            "path_label": args.prepared_output.name,
            "sha256": sha256(args.prepared_output),
        },
        "scoring": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "reranker_model": RERANKER_MODEL,
            "reranker_revision": RERANKER_REVISION,
            "score_calibration": "per_case_minmax",
            "role_relevance_mix": 0.15,
            "device_requested": args.device,
            "embedding_batch_size": args.embedding_batch_size,
            "rerank_batch_size": args.rerank_batch_size,
            "status": "NOT_RUN" if args.prepare_only else "RUNNING",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args.prepare_only:
        print(args.prepared_output)
        print(args.manifest)
        return 0

    started = time.perf_counter()

    def report_progress(index: int, total: int, case_id: str) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            print(f"scored {index}/{total}: {case_id}", flush=True)

    scorer = FrozenBgeScorer(
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        rerank_batch_size=args.rerank_batch_size,
    )
    scored_count = score_cases_resumable(
        args.prepared_output,
        args.role_scores_output,
        scorer=scorer,
        progress=report_progress,
    )
    elapsed = time.perf_counter() - started
    manifest["scoring"].update(
        {
            "status": "COMPLETE",
            "case_count": scored_count,
            "elapsed_seconds": round(elapsed, 6),
            "path_label": args.role_scores_output.name,
            "sha256": sha256(args.role_scores_output),
        }
    )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.role_scores_output)
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
