from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.postgis_migration import DEFAULT_MAPPING_VERSION, migrate_to_postgis


def main() -> int:
    parser = argparse.ArgumentParser(description="Project the encrypted SQLite response domain into PostGIS shadow tables.")
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument(
        "--ddl",
        type=Path,
        default=Path("infra/postgis/001_shadow_projection.sql"),
    )
    parser.add_argument("--mapping-version", default=DEFAULT_MAPPING_VERSION)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = migrate_to_postgis(
        args.source_db,
        args.dsn,
        ddl_path=args.ddl.expanduser().resolve(),
        mapping_version=args.mapping_version,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        target = args.report.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
