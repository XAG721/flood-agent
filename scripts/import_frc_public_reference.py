from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.frc_public_evidence import build_public_reference_report, render_public_reference_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import and independently audit real-model FRC public-dataset artifacts."
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--conflicts-path", type=Path, default=None)
    parser.add_argument(
        "--supplemental-ablation",
        type=Path,
        action="append",
        default=[],
        help="Machine-readable real-model ablation artifact; may be repeated.",
    )
    parser.add_argument(
        "--chunk-length-sensitivity",
        type=Path,
        default=None,
        help="Machine-readable real-model chunk-length sensitivity artifact.",
    )
    parser.add_argument(
        "--controlled-domain-sensitivity",
        type=Path,
        default=None,
        help="Machine-readable synthetic controlled-domain field/role/conflict sensitivity artifact.",
    )
    parser.add_argument(
        "--conflicts-ablation",
        type=Path,
        default=None,
        help="Machine-readable public real-model CONFLICTS w/o Conflict and threshold artifact.",
    )
    parser.add_argument(
        "--housing-ablation",
        type=Path,
        default=None,
        help=(
            "Machine-readable public expert HousingQA w/o Field and jurisdiction-only "
            "w/o Applicability artifact."
        ),
    )
    parser.add_argument(
        "--lawshift-ablation",
        type=Path,
        default=None,
        help=(
            "Machine-readable public expert-reviewed LawShift before/after version "
            "applicability artifact."
        ),
    )
    parser.add_argument(
        "--eurlex-ablation",
        type=Path,
        default=None,
        help=(
            "Machine-readable public official EUR-Lex/CELLAR effective and expiry "
            "date applicability artifact."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/public_frc_reference"),
    )
    args = parser.parse_args()

    report = build_public_reference_report(
        args.reference_root,
        conflicts_path=args.conflicts_path,
        supplemental_ablation_paths=args.supplemental_ablation,
        chunk_length_sensitivity_path=args.chunk_length_sensitivity,
        controlled_domain_sensitivity_path=args.controlled_domain_sensitivity,
        conflicts_ablation_path=args.conflicts_ablation,
        housing_ablation_path=args.housing_ablation,
        lawshift_ablation_path=args.lawshift_ablation,
        eurlex_ablation_path=args.eurlex_ablation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "public_frc_reference_report.json"
    markdown_path = args.output_dir / "public_frc_reference_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_public_reference_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    print(report["decision"]["status"])


if __name__ == "__main__":
    main()
