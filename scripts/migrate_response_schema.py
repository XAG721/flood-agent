from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.repository import SQLiteRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply idempotent response schema migrations and print the ledger.")
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    repository = SQLiteRepository(args.db.expanduser().resolve())
    print(json.dumps(repository.list_schema_migrations(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
