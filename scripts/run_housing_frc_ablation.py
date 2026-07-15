from __future__ import annotations
# ruff: noqa: E402 -- research stays outside the runtime wheel.

import argparse
import sys
import gc
import gzip
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conflicts_evaluation import read_jsonl, stable_hash, write_jsonl
from research.frc_rag.housing_ablation import (
    DEFAULT_CASE_COUNT,
    DEFAULT_FIELD_THRESHOLD,
    DEFAULT_FIELD_WEIGHT,
    DEFAULT_FIELDS_PER_CASE,
    DEFAULT_ROLE_THRESHOLD,
    DEFAULT_ROLE_WEIGHT,
    DEFAULT_SAME_JURISDICTION_DISTRACTORS,
    DEFAULT_SEED,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_TOP_K,
    HOUSING_SOURCE_JSON_SHA256,
    HOUSING_SOURCE_REVISION,
    HOUSING_SOURCE_ZIP_SHA256,
    LocalHousingAnswerer,
    build_housing_ablation_report,
    build_housing_composite_cases,
    housing_selection_signature,
    render_housing_ablation_markdown,
    score_housing_cases,
    select_housing_methods,
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
            "Run HousingQA public expert-data w/o Field and w/o Applicability ablations "
            "using frozen BGE, Cross-Encoder, and local Qwen models."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cache/benchmarks/housing_qa/questions.json"),
    )
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=Path(".cache/benchmarks/housing_qa/questions.json.zip"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/benchmarks/housing_qa/evaluation_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/housing_frc_ablation"),
    )
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--generator-model-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--fields-per-case", type=int, default=DEFAULT_FIELDS_PER_CASE)
    parser.add_argument(
        "--same-jurisdiction-distractors",
        type=int,
        default=DEFAULT_SAME_JURISDICTION_DISTRACTORS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--field-threshold", type=float, default=DEFAULT_FIELD_THRESHOLD)
    parser.add_argument("--role-threshold", type=float, default=DEFAULT_ROLE_THRESHOLD)
    parser.add_argument("--field-weight", type=float, default=DEFAULT_FIELD_WEIGHT)
    parser.add_argument("--role-weight", type=float, default=DEFAULT_ROLE_WEIGHT)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--generator-batch-size", type=int, default=16)
    args = parser.parse_args()

    if sha256(args.input) != HOUSING_SOURCE_JSON_SHA256:
        raise ValueError(
            "HousingQA questions.json hash mismatch; use the pinned public source revision "
            f"{HOUSING_SOURCE_REVISION}"
        )
    if not args.source_zip.is_file() or sha256(args.source_zip) != HOUSING_SOURCE_ZIP_SHA256:
        raise ValueError("HousingQA source zip is missing or does not match the pinned artifact")

    source_rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("HousingQA questions source must be a JSON array")
    cases = build_housing_composite_cases(
        source_rows,
        case_count=args.case_count,
        fields_per_case=args.fields_per_case,
        same_jurisdiction_distractors=args.same_jurisdiction_distractors,
        seed=args.seed,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "benchmark_manifest.json"
    manifest = {
        "source_revision": HOUSING_SOURCE_REVISION,
        "source_json_sha256": sha256(args.input),
        "seed": args.seed,
        "case_count": len(cases),
        "fields_per_case": args.fields_per_case,
        "same_jurisdiction_distractors": args.same_jurisdiction_distractors,
        "cases": cases,
    }
    _write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    config = {
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "reranker_model": "BAAI/bge-reranker-large",
        "generator_model": "Qwen2.5-7B-Instruct-GPTQ-Int4",
        "device": args.device,
        "hf_home": str(args.hf_home.resolve()),
        "embedding_batch_size": args.embedding_batch_size,
        "rerank_batch_size": args.rerank_batch_size,
        "generator_batch_size": args.generator_batch_size,
        "field_relevance_mix": 0.15,
        "role_relevance_mix": 0.15,
        "score_calibration": "per_case_minmax",
        "top_k": args.top_k,
        "token_budget": args.token_budget,
        "field_threshold": args.field_threshold,
        "role_threshold": args.role_threshold,
        "field_weight": args.field_weight,
        "role_weight": args.role_weight,
        "seed": args.seed,
    }
    scored_path = args.cache_dir / "scored_cases.jsonl"
    scored_metadata_path = args.cache_dir / "scored_cases.metadata.json"
    scored = score_housing_cases(
        cases,
        output_path=scored_path,
        metadata_path=scored_metadata_path,
        source_hash=sha256(args.input),
        config=config,
    )
    selected = select_housing_methods(
        scored,
        top_k=args.top_k,
        token_budget=args.token_budget,
        field_threshold=args.field_threshold,
        role_threshold=args.role_threshold,
        field_weight=args.field_weight,
        role_weight=args.role_weight,
    )
    selected_path = args.cache_dir / "selected_evidence.jsonl"
    write_jsonl(selected_path, selected)

    prediction_key = stable_hash(
        {
            "source_json_sha256": sha256(args.input),
            "manifest_sha256": sha256(manifest_path),
            "scored_sha256": sha256(scored_path),
            "selected_sha256": sha256(selected_path),
            "generator_model": config["generator_model"],
            "protocol": "frc-housing-field-applicability-ablation-v1",
        }
    )
    predictions_path = args.cache_dir / "answer_predictions.jsonl"
    cached = []
    if predictions_path.is_file():
        cached = [
            row for row in read_jsonl(predictions_path) if row.get("cache_key") == prediction_key
        ]
    cached_keys = {(row["case_id"], row["method"]) for row in cached}
    expected_keys = {(row["case_id"], row["method"]) for row in selected}
    if cached_keys == expected_keys:
        cached_by_key = {(row["case_id"], row["method"]): row for row in cached}
        predictions = [
            cached_by_key[(source["case_id"], source["method"])]
            for source in selected
        ]
    else:
        if args.generator_model_path is None:
            raise RuntimeError(
                f"{len(expected_keys - cached_keys)} HousingQA prediction rows remain; pass "
                "--generator-model-path to generate them"
            )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        answerer = LocalHousingAnswerer(
            model_path=args.generator_model_path,
            batch_size=args.generator_batch_size,
        )
        predictions = answerer.answer(
            selected, output_path=predictions_path, cache_key=prediction_key
        )
    write_jsonl(predictions_path, predictions)

    unique_selections = {housing_selection_signature(row) for row in selected}
    config["prediction_provenance"] = {
        "total_method_rows": len(selected),
        "unique_case_selection_prompts": len(unique_selections),
        "equivalent_selection_reuse_rows": len(selected) - len(unique_selections),
    }
    report = build_housing_ablation_report(
        cases=cases,
        selected_rows=selected,
        predictions=predictions,
        config=config,
        source_path=args.input,
        source_zip_path=args.source_zip,
        scored_path=scored_path,
    )
    report["metadata"]["manifest_sha256"] = sha256(manifest_path)
    report["metadata"]["prediction_provenance"] = config["prediction_provenance"]
    case_results = report.pop("case_results")
    case_results_path = args.output_dir / "housing_frc_ablation_cases.jsonl.gz"
    _write_deterministic_jsonl_gzip(case_results_path, case_results)
    report["case_results_artifact"] = {
        "file": case_results_path.name,
        "format": "gzip-jsonl",
        "rows": len(case_results),
        "sha256": sha256(case_results_path),
    }
    json_path = args.output_dir / "housing_frc_ablation.json"
    markdown_path = args.output_dir / "housing_frc_ablation.md"
    _write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _write_text(markdown_path, render_housing_ablation_markdown(report))
    print(json_path)
    print(markdown_path)
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
