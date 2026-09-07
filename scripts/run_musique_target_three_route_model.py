from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.musique_target_three_route_router import develop_model


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
SCORED_PATH = Path(
    ".cache/benchmarks/musique_mean_calibrated_three_route_v78/development/scored_blind.jsonl"
)
CASES_PATH = Path(
    "output/rag_evaluation/musique_mean_calibrated_three_route_v78/development/cases.jsonl.gz"
)
V75_MODEL_PATH = Path(
    "docs/progressive_upgrade/twowiki_question_router_model_development_v75.json"
)
OUTPUT_PATH = DOCS_ROOT / "musique_target_three_route_router_model_development_v79.json"


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Develop the v79 target-domain router on frozen v78 evidence."
    )
    parser.parse_args()
    artifact = develop_model(SCORED_PATH, CASES_PATH, V75_MODEL_PATH)
    write_json(OUTPUT_PATH, artifact)
    print(OUTPUT_PATH)
    print(
        json.dumps(
            artifact["crossfit_model_selection_diagnostic"],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
