from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.twowiki_residual_three_route_router import develop_model


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
SCORED_PATHS = (
    Path(
        ".cache/benchmarks/twowiki_question_router_v75/"
        "development/scored_blind.jsonl"
    ),
    Path(
        ".cache/benchmarks/twowiki_question_router_v75/"
        "confirmation/scored_blind.jsonl"
    ),
)
CASES_PATHS = (
    Path(
        "output/rag_evaluation/twowiki_question_router_v75/"
        "development/cases.jsonl.gz"
    ),
    Path(
        "output/rag_evaluation/twowiki_question_router_v75/"
        "confirmation/cases.jsonl.gz"
    ),
)
V75_MODEL_PATH = Path(
    "docs/progressive_upgrade/twowiki_question_router_model_development_v75.json"
)
OUTPUT_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_router_model_development_v81.json"
)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Develop the v81 residual router on frozen v75 evidence."
    )
    parser.parse_args()
    artifact = develop_model(SCORED_PATHS, CASES_PATHS, V75_MODEL_PATH)
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
