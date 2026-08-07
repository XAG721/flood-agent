from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.iirc_score_gap_router import train_router  # noqa: E402


DEFAULT_SCORED = (
    REPO_ROOT / ".cache/benchmarks/iirc/v86/development/scored_blind.jsonl"
)
DEFAULT_GOLD = REPO_ROOT / ".cache/benchmarks/iirc/v86/development/sealed_gold.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs/progressive_upgrade/iirc_score_gap_router_model_development_v87.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the frozen IIRC v87 router.")
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = train_router(args.scored, args.gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(artifact["crossfit_diagnostic"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
