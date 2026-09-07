from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.whoqa_budget_stress import (  # noqa: E402
    evaluate_cases,
    load_protocol,
    load_report,
    read_jsonl,
    sha256,
    validate_frozen_inputs,
    write_report,
)
from research.frc_rag.whoqa_conflict_coverage import load_raw  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen post-v33 WhoQA slot/token budget stress diagnostic."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/progressive_upgrade/whoqa_budget_stress_protocol.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cache/benchmarks/whoqa/WhoQA.json"),
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path(".cache/benchmarks/whoqa/scored_blind.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/whoqa_budget_stress"),
    )
    args = parser.parse_args()

    protocol_path = (REPO_ROOT / args.protocol).resolve()
    input_path = (REPO_ROOT / args.input).resolve()
    scored_path = (REPO_ROOT / args.scored).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    protocol = load_protocol(protocol_path)
    validate_frozen_inputs(REPO_ROOT, protocol)
    raw = load_raw(input_path)
    report, evidence = evaluate_cases(raw, read_jsonl(scored_path), protocol)
    frozen = protocol["frozen_inputs"]
    source_paths = {
        "protocol": protocol_path,
        "raw_source": input_path,
        "scored_blind": scored_path,
        "parent_protocol": REPO_ROOT / frozen["parent_protocol"]["path"],
        "post_result_alias_correction": (
            REPO_ROOT / frozen["post_result_alias_correction"]["path"]
        ),
        "frozen_selector_implementation": (
            REPO_ROOT / frozen["frozen_selector_implementation"]["path"]
        ),
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
        raise AssertionError("WhoQA budget-stress outputs are not byte deterministic")
    loaded = load_report(paths["json"], paths["evidence"])
    outcome = loaded["analysis"]["outcome"]
    family = loaded["analysis"]["family_comparison"]
    print(
        json.dumps(
            {
                "status": outcome["status"],
                "cases": loaded["metadata"]["cases"],
                "selection_runs": loaded["metadata"]["selection_runs"],
                "strongest_baseline": family["observed_strongest_baseline"],
                "frc_minus_strongest": family[
                    "frc_minus_observed_strongest_point"
                ],
                "simultaneous_ci": family[
                    "frc_minus_bootstrap_strongest_simultaneous"
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
