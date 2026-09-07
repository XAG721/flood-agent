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

from research.frc_rag.qasc_confirmation import (
    CANDIDATE_COUNT,
    DATASET_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    EXPECTED_SOURCE_SHA256,
    RERANKER_MODEL,
    RERANKER_REVISION,
    SOURCE_FILENAME,
    SOURCE_LICENSE,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    FrozenBgeScorer,
    load_and_prepare_parquet,
    preparation_summary,
    read_jsonl,
    score_prepared_cases,
    sha256,
    write_jsonl,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_protocol(protocol: dict) -> None:
    if protocol.get("schema_version") != "frc-qasc-confirmation-protocol-v1":
        raise ValueError("unsupported QASC confirmation protocol")
    source = protocol["frozen_source"]
    if (
        source["repository"] != SOURCE_REPOSITORY
        or source["revision"] != SOURCE_REVISION
        or source["path_label"] != SOURCE_FILENAME
        or source["sha256"] != EXPECTED_SOURCE_SHA256
        or source["license"] != SOURCE_LICENSE
    ):
        raise ValueError("QASC frozen source does not match code")
    pool = protocol["frozen_controlled_candidate_pool"]
    if pool["candidate_count_per_case"] != CANDIDATE_COUNT:
        raise ValueError("QASC candidate count does not match code")
    scoring = protocol["frozen_real_model_scoring"]
    expected_model = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "reranker_model": RERANKER_MODEL,
        "reranker_revision": RERANKER_REVISION,
    }
    if any(scoring[key] != value for key, value in expected_model.items()):
        raise ValueError("QASC frozen model snapshots do not match code")
    if scoring["gold_fields_visible_to_scorer"] is not False:
        raise ValueError("QASC scorer must be gold blind")
    for key in ("adapter", "download_runner", "score_runner"):
        registered_path = (
            REPOSITORY_ROOT / protocol["frozen_implementation"][f"{key}_path"]
        )
        if (
            sha256(registered_path)
            != protocol["frozen_implementation"][f"{key}_sha256"]
        ):
            raise ValueError(f"QASC frozen implementation hash mismatch: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and score the frozen QASC validation controlled evidence pool."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cache/benchmarks/qasc/validation.parquet"),
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/candidate_pool_qasc.jsonl"
        ),
    )
    parser.add_argument(
        "--role-scores-output",
        type=Path,
        default=Path(".cache/benchmarks/frc_public_reference/role_scores_qasc.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".cache/benchmarks/frc_public_reference/qasc_provenance.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conformal_qasc_confirmation_protocol.json"
        ),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("D:/RAG_test/.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    for path in (args.input, args.protocol):
        if not path.is_file():
            parser.error(f"required frozen input does not exist: {path}")
    protocol = _load_json(args.protocol)
    _validate_protocol(protocol)
    if args.force_prepare or not args.prepared_output.is_file():
        cases = load_and_prepare_parquet(args.input)
        write_jsonl(args.prepared_output, cases)
    else:
        if sha256(args.input) != EXPECTED_SOURCE_SHA256:
            parser.error("QASC validation hash does not match the protocol")
        cases = read_jsonl(args.prepared_output)
    summary = preparation_summary(cases)
    manifest = {
        "schema_version": "frc-qasc-score-source-provenance-v1",
        "dataset": DATASET_NAME,
        "split": "validation",
        "protocol": {
            "path_label": args.protocol.name,
            "sha256": sha256(args.protocol),
        },
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path_label": args.input.name,
            "sha256": sha256(args.input),
            "expected_sha256": EXPECTED_SOURCE_SHA256,
            "license": SOURCE_LICENSE,
        },
        "preparation": {
            **summary,
            "path_label": args.prepared_output.name,
            "sha256": sha256(args.prepared_output),
            "gold_fields_visible_to_scorer": False,
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
            "gold_fields_visible_to_scorer": False,
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
    scored_count = score_prepared_cases(
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
