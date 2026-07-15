from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.api import app


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the current FastAPI OpenAPI contract.")
    parser.add_argument("--output", type=Path, default=Path("docs/openapi.json"))
    args = parser.parse_args()
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
