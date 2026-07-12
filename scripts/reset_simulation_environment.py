from __future__ import annotations

import argparse
import os
from pathlib import Path

from flood_system.simulation_dataset import write_floodagent_bench
from flood_system.system import FloodWarningSystem


CONFIRMATION = "RESET-SIMULATION"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Destructively rebuild a local simulated environment. Production is always refused."
    )
    parser.add_argument("--db", type=Path, default=Path("data/flood_warning_system_simulation.db"))
    parser.add_argument("--confirm", required=True, help=f"Must equal {CONFIRMATION}")
    args = parser.parse_args()
    environment = os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
    if environment in {"production", "prod"}:
        raise SystemExit("refusing to reset a production environment")
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"confirmation mismatch; pass --confirm {CONFIRMATION}")

    workspace = Path.cwd().resolve()
    db_path = args.db.expanduser().resolve()
    if workspace not in db_path.parents:
        raise SystemExit(f"database must remain inside workspace: {workspace}")
    if db_path.suffix.lower() != ".db":
        raise SystemExit("database path must end with .db")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (db_path, db_path.with_suffix(db_path.suffix + ".key")):
        if candidate.exists():
            candidate.unlink()
    runtime_rag = db_path.parent / "rag_documents.runtime.json"
    if runtime_rag.exists():
        runtime_rag.unlink()

    system = FloodWarningSystem(db_path)
    dashboard = system.response_workflow.bootstrap_demo()
    bench = write_floodagent_bench(Path("output/floodagent_bench/floodagent_bench.json"))
    print(f"simulation database: {db_path}")
    print(f"bootstrap event: {dashboard.event.event_id}")
    print(f"dataset: {bench}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
