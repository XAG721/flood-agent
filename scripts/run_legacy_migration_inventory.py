from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_system.response_workflow.models import LegacyMigrationRequest, OperatorRole
from flood_system.system import FloodWarningSystem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory legacy V2 events and quarantine records that cannot be mapped without fabricating facts."
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mapping-version", default="legacy-v2-to-response-v1")
    parser.add_argument("--apply", action="store_true", help="Record a non-dry-run inventory batch; formal state is still not fabricated.")
    args = parser.parse_args()
    system = FloodWarningSystem(args.db.expanduser().resolve())
    record = system.response_workflow.run_legacy_migration_inventory(
        LegacyMigrationRequest(
            mapping_version=args.mapping_version,
            dry_run=not args.apply,
            operator_id="migration-cli-admin",
            operator_role=OperatorRole.ADMIN,
            terminal_id="migration-cli",
            note="CLI legacy migration inventory",
        )
    )
    print(record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
