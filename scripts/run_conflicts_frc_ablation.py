from __future__ import annotations

import argparse
import gc
import gzip
import json
from pathlib import Path

from flood_system.frc_conflicts_ablation import (
    DEFAULT_CONFLICT_THRESHOLDS,
    build_conflict_ablation_report,
    render_conflict_ablation_markdown,
    seed_equivalent_predictions,
    select_conflict_ablation_evidence,
)
from flood_system.frc_conflicts_evaluation import (
    LocalConflictClassifier,
    read_jsonl,
    selection_signature,
    sha256,
    stable_hash,
    write_jsonl,
)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc


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
            "Run public real-model CONFLICTS w/o Conflict ablation and conflict-role "
            "threshold sensitivity from the frozen saved candidate pool."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cache/benchmarks/rag_conflicts/conflicts.jsonl"),
    )
    parser.add_argument(
        "--scored-cache",
        type=Path,
        default=Path(".cache/benchmarks/rag_conflicts/evaluation_cache_full/scored_cases.jsonl"),
    )
    parser.add_argument(
        "--scored-metadata",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/evaluation_cache_full/scored_cases.metadata.json"
        ),
    )
    parser.add_argument(
        "--reference-selected",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/evaluation_cache_full/selected_evidence.jsonl"
        ),
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/evaluation_cache_full/conflict_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/benchmarks/rag_conflicts/conflict_ablation_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conflicts_frc_ablation"),
    )
    parser.add_argument("--generator-model-path", type=Path, default=None)
    parser.add_argument("--generator-batch-size", type=int, default=16)
    parser.add_argument(
        "--conflict-thresholds",
        type=_parse_thresholds,
        default=DEFAULT_CONFLICT_THRESHOLDS,
    )
    args = parser.parse_args()

    metadata = json.loads(args.scored_metadata.read_text(encoding="utf-8"))
    if metadata.get("input_sha256") != sha256(args.input):
        raise ValueError("scored CONFLICTS cache does not match the input dataset")
    config = dict(metadata["config"])
    if int(metadata.get("cases", 0)) <= 0:
        raise ValueError("scored CONFLICTS cache has no cases")
    config["generator_batch_size"] = args.generator_batch_size

    cases = list(read_jsonl(args.scored_cache))
    if len(cases) != int(metadata["cases"]):
        raise ValueError("scored CONFLICTS cache case count mismatch")
    selected = select_conflict_ablation_evidence(
        cases,
        top_k=int(config["top_k"]),
        token_budget=int(config["token_budget"]),
        conflict_thresholds=args.conflict_thresholds,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.cache_dir / "selected_evidence.jsonl"
    write_jsonl(selected_path, selected)

    classifier_key = stable_hash(
        {
            "input_sha256": metadata["input_sha256"],
            "scored_cache_sha256": sha256(args.scored_cache),
            "selected_sha256": sha256(selected_path),
            "generator_model": config["generator_model"],
            "labels": "CONFLICTS-v1-five-labels",
            "protocol": "frc-conflicts-real-model-ablation-v1",
        }
    )
    reference_selected = list(read_jsonl(args.reference_selected))
    reference_predictions = list(read_jsonl(args.reference_predictions))
    seeds = seed_equivalent_predictions(
        selected,
        reference_selected_rows=reference_selected,
        reference_predictions=reference_predictions,
        cache_key=classifier_key,
    )
    predictions_path = args.cache_dir / "conflict_predictions.jsonl"
    cached = []
    if predictions_path.is_file():
        cached = [
            row
            for row in read_jsonl(predictions_path)
            if row.get("cache_key") == classifier_key
        ]
    completed = {(row["case_id"], row["method"]): row for row in seeds}
    completed.update({(row["case_id"], row["method"]): row for row in cached})
    partial = [
        completed[(row["case_id"], row["method"])]
        for row in selected
        if (row["case_id"], row["method"]) in completed
    ]
    write_jsonl(predictions_path, partial)
    pending_rows = [
        row
        for row in selected
        if (row["case_id"], row["method"]) not in completed
    ]
    reference_signatures = {selection_signature(row) for row in reference_selected}
    pending_unique_signatures = {
        selection_signature(row)
        for row in selected
        if selection_signature(row) not in reference_signatures
    }

    if pending_rows:
        if args.generator_model_path is None:
            raise RuntimeError(
                f"{len(pending_rows)} prediction rows remain; pass --generator-model-path "
                "to generate changed selections"
            )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        classifier = LocalConflictClassifier(
            model_path=args.generator_model_path,
            batch_size=args.generator_batch_size,
        )
        predictions = classifier.classify(
            selected,
            output_path=predictions_path,
            cache_key=classifier_key,
        )
    else:
        predictions = partial
    write_jsonl(predictions_path, predictions)

    config["prediction_provenance"] = {
        "total_variant_rows": len(selected),
        "reused_exact_reference_rows": len(seeds),
        "changed_selection_rows": len(selected) - len(seeds),
        "changed_unique_prompts": len(pending_unique_signatures),
        "reuse_rule": "same case ID and identical ordered selected evidence IDs",
    }
    report = build_conflict_ablation_report(
        cases=cases,
        selected_rows=selected,
        predictions=predictions,
        config=config,
        conflict_thresholds=args.conflict_thresholds,
        source_paths={
            "dataset": args.input,
            "reference_predictions": args.reference_predictions,
            "reference_selected": args.reference_selected,
            "scored_cache": args.scored_cache,
            "scored_metadata": args.scored_metadata,
        },
    )
    case_results = report.pop("case_results")
    case_results_path = args.output_dir / "conflicts_frc_ablation_cases.jsonl.gz"
    _write_deterministic_jsonl_gzip(case_results_path, case_results)
    report["case_results_artifact"] = {
        "file": case_results_path.name,
        "format": "gzip-jsonl",
        "rows": len(case_results),
        "sha256": sha256(case_results_path),
    }
    json_path = args.output_dir / "conflicts_frc_ablation.json"
    markdown_path = args.output_dir / "conflicts_frc_ablation.md"
    _write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _write_text(markdown_path, render_conflict_ablation_markdown(report))
    print(json_path)
    print(markdown_path)
    print(report["decision"]["gate_2"])


if __name__ == "__main__":
    main()
