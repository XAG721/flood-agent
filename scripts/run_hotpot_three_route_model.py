from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.hotpot_three_route_router import develop_model


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
HISTORY_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/frc_public_reference/role_scores_hotpotqa.jsonl"
)
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
OUTPUT_PATH = DOCS_ROOT / "hotpot_three_route_router_model_development_v78.json"


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Develop the v78 three-route router on excluded HotpotQA history."
    )
    parser.parse_args()
    artifact = develop_model(
        HISTORY_PATH,
        V75_MODEL_PATH,
        v76_model_path=V76_MODEL_PATH,
    )
    artifact["history_artifacts"] = {
        "path": HISTORY_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": HISTORY_PATH.stat().st_size,
        "sha256": artifact["history"]["source_sha256"],
        "v75_model_path": V75_MODEL_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "v76_model_path": V76_MODEL_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
    }
    write_json(OUTPUT_PATH, artifact)
    print(OUTPUT_PATH)
    print(json.dumps(artifact["crossfit"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
