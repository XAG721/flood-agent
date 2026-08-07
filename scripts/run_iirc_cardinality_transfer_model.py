from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.iirc_cardinality_transfer_router import (  # noqa: E402
    develop_model,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs/progressive_upgrade/iirc_cardinality_transfer_router_model_development_v86.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the exposed-history cardinality router for IIRC v86."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    artifact = develop_model(REPO_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "training_cases": artifact["training_cases"],
                "selected": artifact["model_selection"],
                "diagnostic": artifact["crossfit_diagnostic"],
                "model_payload_sha256": artifact["model_payload_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
