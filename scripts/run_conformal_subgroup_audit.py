from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_subgroup_audit import (
    evaluate_conformal_subgroup_audit,
    write_conformal_subgroup_audit,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen observable-subgroup stress audit across the registered "
            "public datasets without threshold selection."
        )
    )
    parser.add_argument("--conditionalqa-input", type=Path, required=True)
    parser.add_argument("--hotpotqa-input", type=Path, required=True)
    parser.add_argument("--multihoprag-input", type=Path, required=True)
    parser.add_argument("--twowiki-input", type=Path)
    parser.add_argument("--reference-run-report", type=Path, required=True)
    parser.add_argument(
        "--conditionalqa-robustness",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_robustness/"
            "conformal_robustness.json"
        ),
    )
    parser.add_argument(
        "--hotpotqa-confirmation",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_cross_dataset/"
            "hotpotqa_confirmation.json"
        ),
    )
    parser.add_argument(
        "--multihoprag-confirmation",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_cross_dataset/"
            "multihoprag_confirmation.json"
        ),
    )
    parser.add_argument(
        "--twowiki-confirmation",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_cross_dataset/"
            "2wikimultihopqa_confirmation.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_subgroup_audit"),
    )
    args = parser.parse_args()
    required_paths = [
        args.conditionalqa_input,
        args.hotpotqa_input,
        args.multihoprag_input,
        args.reference_run_report,
        args.conditionalqa_robustness,
        args.hotpotqa_confirmation,
        args.multihoprag_confirmation,
    ]
    if args.twowiki_input is not None:
        required_paths.extend((args.twowiki_input, args.twowiki_confirmation))
    for path in required_paths:
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")

    conditional = _load_json(args.conditionalqa_robustness)
    hotpot_confirmation = _load_json(args.hotpotqa_confirmation)
    multihop_confirmation = _load_json(args.multihoprag_confirmation)
    dataset_sources = {
        "ConditionalQA": args.conditionalqa_input,
        "HotpotQA": args.hotpotqa_input,
        "MultiHop-RAG": args.multihoprag_input,
    }
    expected_robustness = {
        "ConditionalQA": (
            args.conditionalqa_robustness,
            conditional,
        ),
        "HotpotQA": (
            args.hotpotqa_confirmation,
            hotpot_confirmation["confirmation_robustness"],
        ),
        "MultiHop-RAG": (
            args.multihoprag_confirmation,
            multihop_confirmation["confirmation_robustness"],
        ),
    }
    if args.twowiki_input is not None:
        twowiki_confirmation = _load_json(args.twowiki_confirmation)
        dataset_sources["2WikiMultiHopQA"] = args.twowiki_input
        expected_robustness["2WikiMultiHopQA"] = (
            args.twowiki_confirmation,
            twowiki_confirmation["confirmation_robustness"],
        )
    report, case_records = evaluate_conformal_subgroup_audit(
        dataset_sources,
        expected_robustness,
        reference_run_report=args.reference_run_report,
    )
    paths = write_conformal_subgroup_audit(
        report,
        case_records,
        json_path=args.output_dir / "conformal_subgroup_audit.json",
        markdown_path=args.output_dir / "conformal_subgroup_audit.md",
        cases_path=args.output_dir / "conformal_subgroup_cases.jsonl.gz",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
