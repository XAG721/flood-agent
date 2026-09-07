from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conflicts_evaluation import read_jsonl
from research.frc_rag.conflicts_selector_router import (
    ALL_METHODS,
    BOOTSTRAP_RESAMPLES,
    FOLD_COUNT,
    FOLD_NAMESPACE,
    LOGISTIC_ITERATIONS,
    LOGISTIC_L2,
    STATIC_BASELINE,
    evaluate_selector_router,
    write_selector_router,
)
from research.frc_rag.public_evidence import sha256


PROTOCOL_SCHEMA_VERSION = "frc-conflicts-selector-routing-protocol-v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, rows: list[dict]) -> dict[str, object]:
    return {
        "path_label": path.name,
        "sha256": sha256(path),
        "record_count": len(rows),
    }


def _validate_artifact(
    frozen: dict,
    key: str,
    path: Path,
    rows: list[dict],
) -> None:
    artifact = frozen[key]
    if artifact.get("path_label") != path.name:
        raise ValueError(f"protocol {key} path label mismatch")
    if artifact.get("sha256") != sha256(path):
        raise ValueError(f"protocol {key} hash mismatch")
    if int(artifact.get("record_count", -1)) != len(rows):
        raise ValueError(f"protocol {key} record count mismatch")


def _validate_protocol(
    protocol: dict,
    *,
    scored_path: Path,
    scored_rows: list[dict],
    selected_path: Path,
    selected_rows: list[dict],
    predictions_path: Path,
    prediction_rows: list[dict],
) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported selector-routing protocol schema")
    boundary = protocol.get("development_boundary", {})
    required_boundary = {
        "retrospective_discovery": True,
        "pre_registered_confirmation": False,
        "prior_aggregate_probe_disclosed": True,
        "independent_confirmation": False,
        "gate_evidence": False,
        "production_authorization": False,
    }
    for key, expected in required_boundary.items():
        if boundary.get(key) is not expected:
            raise ValueError(f"selector-routing boundary mismatch: {key}")
    frozen = protocol["frozen_inputs"]
    _validate_artifact(frozen, "scored_cases", scored_path, scored_rows)
    _validate_artifact(frozen, "selected_evidence", selected_path, selected_rows)
    _validate_artifact(
        frozen,
        "conflict_predictions",
        predictions_path,
        prediction_rows,
    )
    router = protocol["frozen_router"]
    if tuple(router.get("methods", [])) != ALL_METHODS:
        raise ValueError("selector-routing method order mismatch")
    expected = {
        "static_baseline": STATIC_BASELINE,
        "fold_namespace": FOLD_NAMESPACE,
        "fold_count": FOLD_COUNT,
        "logistic_l2": LOGISTIC_L2,
        "logistic_max_iterations": LOGISTIC_ITERATIONS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    for key, value in expected.items():
        if router.get(key) != value:
            raise ValueError(f"selector-routing frozen parameter mismatch: {key}")
    prior = protocol.get("prior_exposure", {})
    required_prior = {
        "aggregate_oracle_accuracy_seen": 0.545852,
        "sklearn_probe_all_router_accuracy_seen": 0.395197,
        "sklearn_probe_all_router_gain_seen": 0.050218,
        "strict_no_frc_router_accuracy_seen": 0.393013,
    }
    for key, value in required_prior.items():
        if prior.get(key) != value:
            raise ValueError(f"selector-routing prior exposure mismatch: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen retrospective cross-fitted selector-routing discovery "
            "audit on existing CONFLICTS selections and predictions."
        )
    )
    parser.add_argument(
        "--scored-cases",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/evaluation_cache_full/scored_cases.jsonl"
        ),
    )
    parser.add_argument(
        "--selected-evidence",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/evaluation_cache_full/"
            "selected_evidence.jsonl"
        ),
    )
    parser.add_argument(
        "--conflict-predictions",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/evaluation_cache_full/"
            "conflict_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conflicts_selector_router_protocol.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conflicts_selector_router"),
    )
    args = parser.parse_args()
    for path in (
        args.scored_cases,
        args.selected_evidence,
        args.conflict_predictions,
        args.protocol,
    ):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    scored_rows = list(read_jsonl(args.scored_cases))
    selected_rows = list(read_jsonl(args.selected_evidence))
    prediction_rows = list(read_jsonl(args.conflict_predictions))
    protocol = _read_json(args.protocol)
    _validate_protocol(
        protocol,
        scored_path=args.scored_cases,
        scored_rows=scored_rows,
        selected_path=args.selected_evidence,
        selected_rows=selected_rows,
        predictions_path=args.conflict_predictions,
        prediction_rows=prediction_rows,
    )
    source_artifacts = {
        "scored_cases": _artifact(args.scored_cases, scored_rows),
        "selected_evidence": _artifact(args.selected_evidence, selected_rows),
        "conflict_predictions": _artifact(
            args.conflict_predictions,
            prediction_rows,
        ),
    }
    report, evidence = evaluate_selector_router(
        scored_rows,
        selected_rows,
        prediction_rows,
        source_artifacts=source_artifacts,
        prior_exposure=protocol["prior_exposure"],
    )
    paths = write_selector_router(
        report,
        evidence,
        json_path=args.output_dir / "conflicts_selector_router.json",
        markdown_path=args.output_dir / "conflicts_selector_router.md",
        evidence_path=args.output_dir / "conflicts_selector_router_cases.jsonl.gz",
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
