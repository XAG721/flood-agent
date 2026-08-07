from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.finqa_anchor_equivalence_diagnostic import (  # noqa: E402
    analyze_anchor_equivalence,
    validate_protocol,
    write_report,
)
from research.frc_rag.rgb_cost_aware_frc import read_jsonl, sha256  # noqa: E402


EXPECTED_SCORED_SHA256 = (
    "7ecba0f941c26f332caf042a692f4370012b77d60a0fa623aae8895b7b5136dc"
)
EXPECTED_RESULT_SHA256 = (
    "53f2e6ca555184bb0d5e473f7c4d4df19e102a9bceed42f930ae85ee20a6ed84"
)
MODULE_PATH = REPO_ROOT / "research/frc_rag/finqa_anchor_equivalence_diagnostic.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_finqa_anchor_equivalence_diagnostic.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_finqa_anchor_equivalence_diagnostic.py"


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the registered gold-free FinQA v45 anchor diagnostic."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/finqa_anchor_equivalence_diagnostic_protocol_v45.json"
        ),
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path(".cache/benchmarks/finqa/v45/scored.jsonl"),
    )
    parser.add_argument(
        "--locked-result",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/finqa_anchor_guarded_atomic_roles_result_v45.json"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/finqa_anchor_equivalence_diagnostic_result_v45.json"
        ),
    )
    output = Path("output/rag_evaluation/finqa_anchor_guarded_atomic_roles")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=output / "anchor_equivalence_diagnostic.md",
    )
    parser.add_argument(
        "--evidence-output",
        type=Path,
        default=output / "anchor_equivalence_cases.jsonl.gz",
    )
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, _resolve(value))

    protocol = validate_protocol(args.protocol)
    if sha256(args.scored) != EXPECTED_SCORED_SHA256:
        raise ValueError("FinQA anchor diagnostic score-cache hash mismatch")
    if sha256(args.locked_result) != EXPECTED_RESULT_SHA256:
        raise ValueError("FinQA anchor diagnostic locked-result hash mismatch")
    report, evidence = analyze_anchor_equivalence(read_jsonl(args.scored))
    report["metadata"]["integrity"] = {
        "protocol_sha256": sha256(args.protocol),
        "scored_blind_cache_sha256": sha256(args.scored),
        "locked_result_sha256": sha256(args.locked_result),
        "diagnostic_module_sha256": sha256(MODULE_PATH),
        "diagnostic_runner_sha256": sha256(RUNNER_PATH),
        "diagnostic_test_sha256": sha256(TEST_PATH),
        "protocol": protocol,
    }
    write_report(
        report,
        evidence,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        evidence_path=args.evidence_output,
    )
    first = {
        "json": sha256(args.json_output),
        "markdown": sha256(args.markdown_output),
        "evidence": sha256(args.evidence_output),
    }
    write_report(
        report,
        evidence,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        evidence_path=args.evidence_output,
    )
    second = {
        "json": sha256(args.json_output),
        "markdown": sha256(args.markdown_output),
        "evidence": sha256(args.evidence_output),
    }
    if first != second:
        raise AssertionError("FinQA anchor diagnostic outputs are not deterministic")
    print(
        json.dumps(
            {
                "status": report["analysis"]["outcome"]["status"],
                "overall": report["analysis"]["overall"],
                "output_hashes": second,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
