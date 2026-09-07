from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.twowiki_bridge_aware_precision_trim_router import (
    develop_model,
)


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
COHORT_ROOTS = (
    REPOSITORY_ROOT / ".cache/benchmarks/twowiki_question_router_v75/development",
    REPOSITORY_ROOT / ".cache/benchmarks/twowiki_question_router_v75/confirmation",
    REPOSITORY_ROOT / ".cache/benchmarks/twowiki_residual_router_v81/development",
    REPOSITORY_ROOT / ".cache/benchmarks/twowiki_cascaded_router_v82/development",
)
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V81_MODEL_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_router_model_development_v81.json"
)
V82_MODEL_PATH = (
    DOCS_ROOT / "twowiki_cascaded_style_residual_router_model_development_v82.json"
)
DEFAULT_OUTPUT = (
    DOCS_ROOT / "twowiki_bridge_aware_precision_trim_router_model_development_v83.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Develop the frozen v83 bridge-aware precision-trim model."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = develop_model(
        [root / "scored_blind.jsonl" for root in COHORT_ROOTS],
        [root / "sealed_gold.jsonl" for root in COHORT_ROOTS],
        V75_MODEL_PATH,
        V81_MODEL_PATH,
        V82_MODEL_PATH,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostic = artifact["crossfit_model_selection_diagnostic"]
    print(args.output)
    print(json.dumps(artifact["selected_configuration"], sort_keys=True))
    print(json.dumps(diagnostic["candidate"], sort_keys=True))
    print(json.dumps(diagnostic["candidate_minus_v82"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
