from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.musique_dual_resource import (  # noqa: E402
    load_frozen_tokenizer,
    load_source_rows,
    prepare_dataset,
    read_jsonl,
)
from research.frc_rag.musique_objective_gap import (  # noqa: E402
    _validate_source_inputs,
    evaluate_scored_cases,
    load_protocol,
    load_report,
    sha256,
    validate_preparation_summary,
    write_report,
)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen post-result MuSiQue v37 objective-gap diagnostic."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/musique_objective_gap_diagnostic_protocol.json"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique/data/musique_ans_v1.0_dev.jsonl"
        ),
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique/scored_dual_resource_blind.jsonl"
        ),
    )
    parser.add_argument(
        "--v36-report",
        type=Path,
        default=Path(
            "output/rag_evaluation/musique_dual_resource/musique_dual_resource.json"
        ),
    )
    parser.add_argument(
        "--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/musique_objective_gap"),
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("--resamples must be positive")

    protocol_path = _resolve(args.protocol)
    input_path = _resolve(args.input)
    scored_path = _resolve(args.scored)
    v36_report_path = _resolve(args.v36_report)
    hf_home = _resolve(args.hf_home)
    output_dir = _resolve(args.output_dir)
    protocol = load_protocol(protocol_path)
    _validate_source_inputs(scored_path, v36_report_path, protocol)
    source_rows = load_source_rows(input_path)
    tokenizer = load_frozen_tokenizer(hf_home)
    prepared, gold, summary = prepare_dataset(source_rows, tokenizer)
    validate_preparation_summary(summary, protocol)
    scored = list(read_jsonl(scored_path))
    if [row["id"] for row in prepared] != [row.get("id") for row in scored]:
        raise ValueError("MuSiQue v37 blind score cache is incomplete or reordered")
    report, evidence = evaluate_scored_cases(
        gold, scored, resamples=args.resamples
    )
    source_paths = {
        "implementation": REPO_ROOT
        / "research/frc_rag/musique_objective_gap.py",
        "protocol": protocol_path,
        "raw_source": input_path,
        "scored_blind": scored_path,
        "v36_report": v36_report_path,
    }
    paths = write_report(report, evidence, output_dir, source_paths=source_paths)
    first = {name: sha256(path) for name, path in paths.items()}
    paths = write_report(report, evidence, output_dir, source_paths=source_paths)
    second = {name: sha256(path) for name, path in paths.items()}
    if first != second:
        raise AssertionError("MuSiQue v37 outputs are not byte deterministic")
    loaded = load_report(paths["json"], paths["evidence"])
    print(
        json.dumps(
            {
                "status": loaded["analysis"]["outcome"]["status"],
                "next_action": loaded["analysis"]["outcome"]["next_action"],
                "cases": loaded["metadata"]["cases"],
                "objective_gap": loaded["aggregates"]["objective_gap"],
                "exact_minus_v36": loaded["analysis"][
                    "exact_minus_v36_support_f1"
                ],
                "exact_minus_cross_encoder_topk": loaded["analysis"][
                    "exact_minus_cross_encoder_topk_support_f1"
                ],
                "output_hashes": second,
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
