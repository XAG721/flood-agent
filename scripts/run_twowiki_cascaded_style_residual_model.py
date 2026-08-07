from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.twowiki_cascaded_style_residual_router import develop_model


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
SCORED_PATHS = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_question_router_v75/development/scored_blind.jsonl",
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_question_router_v75/confirmation/scored_blind.jsonl",
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_residual_router_v81/development/scored_blind.jsonl",
)
CASES_PATHS = (
    REPOSITORY_ROOT
    / "output/rag_evaluation/twowiki_question_router_v75/development/cases.jsonl.gz",
    REPOSITORY_ROOT
    / "output/rag_evaluation/twowiki_question_router_v75/confirmation/cases.jsonl.gz",
    REPOSITORY_ROOT
    / "output/rag_evaluation/twowiki_residual_router_v81/development/cases.jsonl.gz",
)
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V81_MODEL_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_router_model_development_v81.json"
)
OUTPUT_PATH = (
    DOCS_ROOT
    / "twowiki_cascaded_style_residual_router_model_development_v82.json"
)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Develop the v82 cascaded router on exposed 2Wiki evidence."
    )
    parser.parse_args()
    artifact = develop_model(
        SCORED_PATHS, CASES_PATHS, V75_MODEL_PATH, V81_MODEL_PATH
    )
    write_json(OUTPUT_PATH, artifact)
    print(OUTPUT_PATH)
    print(
        json.dumps(
            {
                "selected_configuration": artifact["selected_configuration"],
                "diagnostic": artifact["crossfit_model_selection_diagnostic"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
