from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conformal_robustness import (
    DEFAULT_SPLIT_VERSIONS,
    evaluate_conformal_robustness,
)
from research.frc_rag.conformal_sufficiency import (
    DEFAULT_ALPHAS,
    evaluate_conformal_sufficiency,
)
from research.frc_rag.public_evidence import sha256


SCHEMA_VERSION = "frc-conformal-source-preparation-benchmark-v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_repeated_parse(
    source_path: Path,
    *,
    reference_run_report: Path | None,
) -> list[dict[str, Any]]:
    repeats = []
    for split_version in DEFAULT_SPLIT_VERSIONS:
        report, _ = evaluate_conformal_sufficiency(
            source_path,
            reference_run_report=reference_run_report,
            split_version=split_version,
        )
        repeats.append(
            {
                "split_version": split_version,
                "split_summary": report["split_summary"],
                "evaluation_auc": report["evaluation"]["auc"],
                "calibration_case_level_units": report["calibration"][
                    "case_level_calibration_units"
                ],
                "finite_sample_nominal_bounds": {
                    str(alpha): report["calibration"]["alphas"][str(alpha)][
                        "finite_sample_nominal_case_error_upper_bound"
                    ]
                    for alpha in DEFAULT_ALPHAS
                },
                "baseline": report["evaluation"][
                    "baseline_role_coverage_heuristic"
                ],
                "alphas": {
                    str(alpha): report["calibration"]["alphas"][str(alpha)][
                        "evaluation"
                    ]
                    for alpha in DEFAULT_ALPHAS
                },
            }
        )
    return repeats


def benchmark(
    source_path: Path,
    *,
    reference_run_report: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    legacy_repeats = _legacy_repeated_parse(
        source_path,
        reference_run_report=reference_run_report,
    )
    legacy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    optimized = evaluate_conformal_robustness(
        source_path,
        reference_run_report=reference_run_report,
    )
    optimized_seconds = time.perf_counter() - started
    optimized_repeats = optimized["repeats"]
    results_identical = legacy_repeats == optimized_repeats
    if not results_identical:
        raise AssertionError(
            "single-parse optimization changed repeated-split results"
        )
    legacy_hash = _canonical_sha256(legacy_repeats)
    optimized_hash = _canonical_sha256(optimized_repeats)
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": "RUN_LOCAL_CONFORMAL_SOURCE_PREPARATION_BENCHMARK",
            "dataset": optimized["metadata"]["dataset"],
            "source_path_label": source_path.name,
            "source_sha256": sha256(source_path),
            "repeat_count": len(DEFAULT_SPLIT_VERSIONS),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        },
        "comparison": {
            "legacy": {
                "jsonl_parse_passes": len(DEFAULT_SPLIT_VERSIONS),
                "elapsed_seconds": round(legacy_seconds, 6),
                "canonical_repeat_sha256": legacy_hash,
            },
            "single_parse": {
                "jsonl_parse_passes": 1,
                "elapsed_seconds": round(optimized_seconds, 6),
                "canonical_repeat_sha256": optimized_hash,
            },
            "results_identical": results_identical,
            "speedup": round(
                legacy_seconds / max(optimized_seconds, 1e-12),
                6,
            ),
        },
        "interpretation": {
            "scope": (
                "Measures local repeated-split experiment orchestration only; "
                "retriever, reranker and generator inference are not rerun."
            ),
            "limitations": [
                "Single-machine wall time is descriptive, not a production SLA.",
                "Legacy runs first and the optimized run may benefit from OS file cache.",
                "Canonical repeat hashes, rather than elapsed time, prove equivalence.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark legacy repeated JSONL parsing against single-parse feature "
            "reuse and require exact repeated-result equivalence."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-run-report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/rag_evaluation/conformal_cross_dataset/"
            "source_preparation_benchmark.json"
        ),
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input score file does not exist: {args.input}")
    if (
        args.reference_run_report is not None
        and not args.reference_run_report.is_file()
    ):
        parser.error(
            f"reference run report does not exist: {args.reference_run_report}"
        )
    report = benchmark(
        args.input,
        reference_run_report=args.reference_run_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
