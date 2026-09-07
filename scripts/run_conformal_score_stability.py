from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_robustness import DEFAULT_SPLIT_VERSIONS
from research.frc_rag.conformal_score_stability import (
    ADOPTION_TARGETS,
    AUGMENTED_FEATURE_NAMES,
    RETRIEVAL_STAGES,
    SCORE_STABILITY_FEATURE_NAMES,
    evaluate_score_stability_conformal,
    write_score_stability_conformal,
)
from research.frc_rag.conformal_sufficiency import FEATURE_NAMES
from research.frc_rag.public_evidence import sha256


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol_targets(protocol: dict) -> dict[str, float | int | bool]:
    registered = protocol["pre_registered_adoption_checks"]
    return {key: registered[key] for key in ADOPTION_TARGETS}


def _validate_protocol(
    protocol: dict,
    sources: dict[str, Path],
    baseline_reports: dict[str, Path],
) -> None:
    if protocol.get("schema_version") != "frc-conformal-score-stability-protocol-v1":
        raise ValueError("unsupported score-stability protocol schema")
    model = protocol["frozen_model_protocol"]
    if model["base_feature_count"] != len(FEATURE_NAMES):
        raise ValueError("protocol base feature count does not match code")
    if model["additional_feature_count"] != len(SCORE_STABILITY_FEATURE_NAMES):
        raise ValueError("protocol stability feature count does not match code")
    if model["augmented_feature_count"] != len(AUGMENTED_FEATURE_NAMES):
        raise ValueError("protocol augmented feature count does not match code")
    if model["retrieval_stages"] != list(RETRIEVAL_STAGES):
        raise ValueError("protocol retrieval stages do not match code")
    if tuple(protocol["frozen_splits_and_calibration"]["split_versions"]) != (
        DEFAULT_SPLIT_VERSIONS
    ):
        raise ValueError("protocol split versions do not match code")
    if _protocol_targets(protocol) != ADOPTION_TARGETS:
        raise ValueError("protocol adoption targets do not match code")
    registered = {
        item["dataset"]: item for item in protocol["frozen_inputs"]
    }
    if set(registered) != set(sources) or set(registered) != set(baseline_reports):
        raise ValueError("protocol datasets do not match requested inputs")
    for dataset, item in registered.items():
        source_path = sources[dataset]
        baseline_path = baseline_reports[dataset]
        if item["source_path_label"] != source_path.name:
            raise ValueError(f"protocol source label mismatch for {dataset}")
        if item["source_sha256"] != sha256(source_path):
            raise ValueError(f"protocol source hash mismatch for {dataset}")
        if item["baseline_report_path_label"] != baseline_path.name:
            raise ValueError(f"protocol baseline label mismatch for {dataset}")
        if item["baseline_report_sha256"] != sha256(baseline_path):
            raise ValueError(f"protocol baseline hash mismatch for {dataset}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen target-fitted FRC multi-stage score-stability "
            "sufficiency-head comparison."
        )
    )
    parser.add_argument("--conditionalqa-input", type=Path, required=True)
    parser.add_argument("--hotpotqa-input", type=Path, required=True)
    parser.add_argument("--multihoprag-input", type=Path, required=True)
    parser.add_argument("--twowiki-input", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conformal_score_stability_protocol.json"
        ),
    )
    parser.add_argument(
        "--conditionalqa-robustness",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_robustness/conformal_robustness.json"
        ),
    )
    parser.add_argument(
        "--hotpotqa-confirmation",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_cross_dataset/hotpotqa_confirmation.json"
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
        default=Path("output/rag_evaluation/conformal_score_stability"),
    )
    args = parser.parse_args()

    sources = {
        "2WikiMultiHopQA": args.twowiki_input,
        "ConditionalQA": args.conditionalqa_input,
        "HotpotQA": args.hotpotqa_input,
        "MultiHop-RAG": args.multihoprag_input,
    }
    baseline_paths = {
        "2WikiMultiHopQA": args.twowiki_confirmation,
        "ConditionalQA": args.conditionalqa_robustness,
        "HotpotQA": args.hotpotqa_confirmation,
        "MultiHop-RAG": args.multihoprag_confirmation,
    }
    for path in (*sources.values(), *baseline_paths.values(), args.protocol):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")

    protocol = _load_json(args.protocol)
    _validate_protocol(protocol, sources, baseline_paths)
    conditional = _load_json(args.conditionalqa_robustness)
    hotpot = _load_json(args.hotpotqa_confirmation)
    multihop = _load_json(args.multihoprag_confirmation)
    twowiki = _load_json(args.twowiki_confirmation)
    expected = {
        "2WikiMultiHopQA": (
            args.twowiki_confirmation,
            twowiki["confirmation_robustness"],
        ),
        "ConditionalQA": (args.conditionalqa_robustness, conditional),
        "HotpotQA": (
            args.hotpotqa_confirmation,
            hotpot["confirmation_robustness"],
        ),
        "MultiHop-RAG": (
            args.multihoprag_confirmation,
            multihop["confirmation_robustness"],
        ),
    }
    report, case_records = evaluate_score_stability_conformal(
        sources,
        expected,
        adoption_targets=_protocol_targets(protocol),
    )
    registered = {
        item["dataset"]: item for item in protocol["frozen_inputs"]
    }
    for item in report["provenance"]:
        expected_count = int(registered[item["dataset"]]["record_count"])
        if item["source_record_count"] != expected_count:
            raise ValueError(
                f"protocol source record count mismatch for {item['dataset']}"
            )
    paths = write_score_stability_conformal(
        report,
        case_records,
        json_path=args.output_dir / "conformal_score_stability.json",
        markdown_path=args.output_dir / "conformal_score_stability.md",
        cases_path=(
            args.output_dir / "conformal_score_stability_cases.jsonl.gz"
        ),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
