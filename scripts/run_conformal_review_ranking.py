from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_review_ranking import (
    ADOPTION_TARGETS,
    MIN_SUBGROUP_COMPLETE_VARIANTS,
    MIN_SUBGROUP_POOL_VARIANTS,
    MIN_SUBGROUP_SUPPORTED_REPEATS,
    REVIEW_BUDGET_FRACTIONS,
    SUBGROUP_BUDGET,
    evaluate_review_ranking,
    read_gzip_jsonl,
    write_review_ranking,
)
from research.frc_rag.conformal_score_stability import (
    load_score_stability_conformal,
)
from research.frc_rag.public_evidence import sha256


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol_targets(protocol: dict) -> dict[str, float | int | bool]:
    registered = protocol["pre_registered_adoption_checks"]
    return {key: registered[key] for key in ADOPTION_TARGETS}


def _validate_protocol(
    protocol: dict,
    source_report_path: Path,
    source_cases_path: Path,
    source_report: dict,
    source_records: list[dict],
) -> None:
    if protocol.get("schema_version") != ("frc-conformal-review-ranking-protocol-v1"):
        raise ValueError("unsupported review-ranking protocol schema")
    boundary = protocol["development_boundary"]
    required_boundary = {
        "post_hoc": True,
        "independent_confirmation": False,
        "gate_evidence": False,
        "evaluation_used_to_choose_ranking_or_budget": False,
        "automatic_decisions_may_change": False,
        "automatic_thresholds_may_change": False,
        "review_only": True,
    }
    for key, expected in required_boundary.items():
        if boundary.get(key) is not expected:
            raise ValueError(f"protocol boundary mismatch: {key}")
    frozen = protocol["frozen_input"]
    if frozen["source_report_path_label"] != source_report_path.name:
        raise ValueError("source report label does not match protocol")
    if frozen["source_report_sha256"] != sha256(source_report_path):
        raise ValueError("source report hash does not match protocol")
    if frozen["source_cases_path_label"] != source_cases_path.name:
        raise ValueError("source cases label does not match protocol")
    if frozen["source_cases_sha256"] != sha256(source_cases_path):
        raise ValueError("source cases hash does not match protocol")
    if int(frozen["source_case_record_count"]) != len(source_records):
        raise ValueError("source case count does not match protocol")
    metadata = source_report["metadata"]
    if frozen["datasets"] != metadata["datasets"]:
        raise ValueError("source datasets do not match protocol")
    if int(frozen["repeat_count_per_dataset"]) != int(
        metadata["repeat_count_per_dataset"]
    ):
        raise ValueError("source repeat count does not match protocol")
    rankings = protocol["frozen_rankings"]
    if tuple(rankings["review_budget_fractions"]) != REVIEW_BUDGET_FRACTIONS:
        raise ValueError("review budgets do not match code")
    metrics = protocol["frozen_metrics"]
    if float(metrics["subgroup_budget"]) != SUBGROUP_BUDGET:
        raise ValueError("subgroup budget does not match code")
    if int(metrics["minimum_subgroup_pool_variants_per_repeat"]) != (
        MIN_SUBGROUP_POOL_VARIANTS
    ):
        raise ValueError("subgroup pool support does not match code")
    if int(metrics["minimum_subgroup_complete_variants_per_repeat"]) != (
        MIN_SUBGROUP_COMPLETE_VARIANTS
    ):
        raise ValueError("subgroup complete support does not match code")
    if int(metrics["minimum_supported_repeats_per_subgroup"]) != (
        MIN_SUBGROUP_SUPPORTED_REPEATS
    ):
        raise ValueError("subgroup repeat support does not match code")
    if _protocol_targets(protocol) != ADOPTION_TARGETS:
        raise ValueError("adoption targets do not match code")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen 64-feature score as a review-only ordering "
            "without changing the 37-feature automatic boundary."
        )
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_score_stability/"
            "conformal_score_stability.json"
        ),
    )
    parser.add_argument(
        "--source-cases",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_score_stability/"
            "conformal_score_stability_cases.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/progressive_upgrade/conformal_review_ranking_protocol.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conformal_review_ranking"),
    )
    args = parser.parse_args()
    for path in (args.source_report, args.source_cases, args.protocol):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")

    source_report = load_score_stability_conformal(
        args.source_report, args.source_cases
    )
    source_records = read_gzip_jsonl(args.source_cases)
    protocol = _load_json(args.protocol)
    _validate_protocol(
        protocol,
        args.source_report,
        args.source_cases,
        source_report,
        source_records,
    )
    source_artifact = {
        "report_path_label": args.source_report.name,
        "report_sha256": sha256(args.source_report),
        "cases_path_label": args.source_cases.name,
        "cases_sha256": sha256(args.source_cases),
        "case_record_count": len(source_records),
    }
    report, evidence = evaluate_review_ranking(
        source_report,
        source_records,
        source_artifact=source_artifact,
        budgets=tuple(protocol["frozen_rankings"]["review_budget_fractions"]),
        subgroup_budget=float(protocol["frozen_metrics"]["subgroup_budget"]),
        min_subgroup_pool_variants=int(
            protocol["frozen_metrics"]["minimum_subgroup_pool_variants_per_repeat"]
        ),
        min_subgroup_complete_variants=int(
            protocol["frozen_metrics"]["minimum_subgroup_complete_variants_per_repeat"]
        ),
        min_subgroup_supported_repeats=int(
            protocol["frozen_metrics"]["minimum_supported_repeats_per_subgroup"]
        ),
        adoption_targets=_protocol_targets(protocol),
    )
    paths = write_review_ranking(
        report,
        evidence,
        json_path=args.output_dir / "conformal_review_ranking.json",
        markdown_path=args.output_dir / "conformal_review_ranking.md",
        evidence_path=(args.output_dir / "conformal_review_ranking_evidence.jsonl.gz"),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
