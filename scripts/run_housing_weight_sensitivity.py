from __future__ import annotations
# ruff: noqa: E402 -- research stays outside the runtime wheel.

import argparse
import sys
import gzip
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.housing_weight_sensitivity import (
    evaluate_housing_weight_sensitivity,
    read_scored_cases,
    render_housing_weight_sensitivity_markdown,
)
from research.frc_rag.housing_ablation import sha256


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_deterministic_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-factor HousingQA field/role-weight sensitivity over the frozen public "
            "real-model candidate cache without retuning or rerunning generation."
        )
    )
    parser.add_argument(
        "--scored-cases",
        type=Path,
        default=Path(
            ".cache/benchmarks/housing_qa/evaluation_cache/scored_cases.jsonl"
        ),
    )
    parser.add_argument(
        "--scored-metadata",
        type=Path,
        default=Path(
            ".cache/benchmarks/housing_qa/evaluation_cache/scored_cases.metadata.json"
        ),
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        default=Path(".cache/benchmarks/housing_qa/questions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/housing_weight_sensitivity"),
    )
    args = parser.parse_args()

    cases = read_scored_cases(args.scored_cases)
    scored_metadata = json.loads(args.scored_metadata.read_text(encoding="utf-8"))
    report, case_results = evaluate_housing_weight_sensitivity(
        cases,
        scored_metadata=scored_metadata,
        scored_cases_path=args.scored_cases,
        scored_metadata_path=args.scored_metadata,
        source_json_path=args.source_json,
    )
    metadata = report["metadata"]
    if (
        metadata["case_count"] != 40
        or metadata["field_count"] != 160
        or metadata["jurisdiction_count"] < 20
    ):
        raise ValueError(
            "HousingQA public weight sensitivity must cover 40 cases, 160 fields, and at "
            "least 20 jurisdictions"
        )

    case_path = args.output_dir / "housing_weight_sensitivity_cases.jsonl.gz"
    _write_deterministic_gzip_jsonl(case_path, case_results)
    report["case_results_artifact"] = {
        "file": case_path.name,
        "format": "gzip-jsonl",
        "rows": len(case_results),
        "sha256": sha256(case_path),
    }
    json_path = args.output_dir / "housing_weight_sensitivity.json"
    markdown_path = args.output_dir / "housing_weight_sensitivity.md"
    _write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_text(markdown_path, render_housing_weight_sensitivity_markdown(report))
    print(json_path)
    print(markdown_path)
    print(json.dumps(report["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
