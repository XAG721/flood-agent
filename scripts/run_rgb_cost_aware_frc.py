from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    FrozenRGBScorer,
    evaluate_scored_cases,
    load_frozen_tokenizer,
    load_registration,
    load_report,
    load_source_rows,
    prepare_datasets,
    read_jsonl,
    score_cases_resumable,
    sha256,
    validate_preparation_summary,
    write_jsonl,
    write_report,
)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen blind RGB cost-aware FRC experiment registered for v35."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/progressive_upgrade/rgb_cost_aware_frc_protocol.json"),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path("docs/progressive_upgrade/rgb_cost_aware_frc_execution.json"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".cache/benchmarks/rgb"),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(r"D:\RAG_test\.hf_cache"),
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path(".cache/benchmarks/rgb/prepared_blind.jsonl"),
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path(".cache/benchmarks/rgb/scored_blind.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/rgb_cost_aware_frc"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after deterministic blind preparation and census validation.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Use an already complete blind score cache without loading neural models.",
    )
    args = parser.parse_args()
    if args.prepare_only and args.evaluate_only:
        parser.error("--prepare-only and --evaluate-only are mutually exclusive")
    if args.resamples <= 0:
        parser.error("--resamples must be positive")

    protocol_path = _resolve(args.protocol)
    execution_path = _resolve(args.execution)
    source_root = _resolve(args.source_root)
    hf_home = _resolve(args.hf_home)
    prepared_path = _resolve(args.prepared)
    scored_path = _resolve(args.scored)
    output_dir = _resolve(args.output_dir)
    _, execution = load_registration(protocol_path, execution_path)
    source_rows = load_source_rows(source_root)
    tokenizer = load_frozen_tokenizer(hf_home)
    prepared, gold, summary = prepare_datasets(source_rows, tokenizer)
    validate_preparation_summary(summary, execution)
    write_jsonl(prepared_path, prepared)

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "RGB_BLIND_PREPARATION_COMPLETE",
                    "summary": summary,
                    "prepared_sha256": sha256(prepared_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.evaluate_only:
        scored = list(read_jsonl(scored_path))
        if [row.get("id") for row in scored] != [
            row.get("id") for row in prepared
        ]:
            raise ValueError("RGB evaluate-only score cache is incomplete or reordered")
    else:
        scorer = FrozenRGBScorer(
            hf_home=hf_home,
            device=args.device,
            embedding_batch_size=args.embedding_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            use_fp16=args.device.startswith("cuda"),
        )
        scored = score_cases_resumable(
            prepared,
            scorer,
            scored_path,
            case_batch_size=args.case_batch_size,
        )

    report, evidence = evaluate_scored_cases(
        gold,
        scored,
        resamples=args.resamples,
    )
    source_paths = {
        "execution": execution_path,
        "implementation": REPO_ROOT
        / "research/frc_rag/rgb_cost_aware_frc.py",
        "post_result_correction": REPO_ROOT
        / "docs/progressive_upgrade/rgb_cost_aware_frc_post_result_correction.json",
        "prepared_blind": prepared_path,
        "protocol": protocol_path,
        "rgb_en_fact": source_root / "data/en_fact.json",
        "rgb_en_int": source_root / "data/en_int.json",
        "rgb_en_refine": source_root / "data/en_refine.json",
        "scored_blind": scored_path,
    }
    paths = write_report(
        report,
        evidence,
        output_dir,
        source_paths=source_paths,
    )
    first_hashes = {name: sha256(path) for name, path in paths.items()}
    paths = write_report(
        report,
        evidence,
        output_dir,
        source_paths=source_paths,
    )
    second_hashes = {name: sha256(path) for name, path in paths.items()}
    if first_hashes != second_hashes:
        raise AssertionError("RGB report outputs are not byte deterministic")
    loaded = load_report(paths["json"], paths["evidence"])
    comparison = loaded["analysis"]["family_comparison"]
    print(
        json.dumps(
            {
                "status": loaded["analysis"]["outcome"]["status"],
                "cases": loaded["metadata"]["cases"],
                "candidate_chunks": loaded["metadata"]["candidate_chunks"],
                "strongest_baseline": comparison["observed_strongest_baseline"],
                "cost_aware_minus_strongest": comparison[
                    "cost_aware_minus_observed_strongest"
                ]["point"],
                "simultaneous_ci": comparison[
                    "cost_aware_minus_bootstrap_strongest_simultaneous"
                ],
                "output_hashes": second_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
