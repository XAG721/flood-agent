from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.hotpot_question_type_cardinality_router import develop_model


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
HISTORY_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/frc_public_reference/role_scores_hotpotqa.jsonl"
)
COHORT_ROOTS = (
    REPOSITORY_ROOT / ".cache/benchmarks/hotpot_graph_router_v76/development",
    REPOSITORY_ROOT / ".cache/benchmarks/hotpot_graph_router_v76/confirmation",
)
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
DEFAULT_OUTPUT = (
    DOCS_ROOT / "hotpot_question_type_cardinality_router_model_development_v84.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Develop the frozen HotpotQA v84 cardinality router."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = develop_model(
        HISTORY_PATH,
        [root / "scored_blind.jsonl" for root in COHORT_ROOTS],
        [root / "sealed_gold.jsonl" for root in COHORT_ROOTS],
        V75_MODEL_PATH,
        V76_MODEL_PATH,
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
    print(json.dumps(diagnostic["per_training_cohort"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
