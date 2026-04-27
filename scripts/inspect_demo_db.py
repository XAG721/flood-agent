from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flood_system.system import FloodWarningSystem

DEFAULT_DB_PATH = ROOT / "data" / "flood_warning_system_demo.db"
DEFAULT_EVENT_ID = "event_demo_beilin_primary"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = [
        "v2_events",
        "v2_hazard_states",
        "v2_entity_profiles",
        "v2_agent_tasks",
        "v2_agent_results",
        "v2_event_shared_memory",
        "v2_supervisor_runs",
        "v2_audit_records",
        "v2_action_proposals",
        "v2_notification_drafts",
        "v2_execution_logs",
        "v3_audience_warnings",
    ]
    return {table: int(scalar(conn, f"SELECT COUNT(*) FROM {table}") or 0) for table in tables}


def inspect_db(db_path: Path, *, event_id: str, freshness_hours: int) -> dict[str, Any]:
    if not db_path.exists():
        raise SystemExit(f"Database does not exist: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    counts = table_counts(conn)
    active_events = int(
        scalar(conn, "SELECT COUNT(*) FROM v2_events WHERE json_extract(payload, '$.status') = 'active'") or 0
    )
    pending = int(
        scalar(
            conn,
            "SELECT COUNT(*) FROM v2_action_proposals WHERE event_id = ? AND status = 'pending'",
            (event_id,),
        )
        or 0
    )
    approved = int(
        scalar(
            conn,
            "SELECT COUNT(*) FROM v2_action_proposals WHERE event_id = ? AND status = 'approved'",
            (event_id,),
        )
        or 0
    )
    warnings = int(scalar(conn, "SELECT COUNT(*) FROM v3_audience_warnings WHERE event_id = ?", (event_id,)) or 0)
    execution_logs = int(scalar(conn, "SELECT COUNT(*) FROM v2_execution_logs WHERE event_id = ?", (event_id,)) or 0)
    latest_event_update = scalar(conn, "SELECT updated_at FROM v2_events WHERE event_id = ?", (event_id,))
    latest_hazard_update = scalar(conn, "SELECT generated_at FROM v2_hazard_states WHERE event_id = ?", (event_id,))
    latest_event_dt = parse_dt(str(latest_event_update) if latest_event_update else None)
    latest_hazard_dt = parse_dt(str(latest_hazard_update) if latest_hazard_update else None)
    conn.close()

    freshness_cutoff = datetime.now(timezone.utc) - timedelta(hours=freshness_hours)
    checks = [
        Check("single_active_event", active_events == 1, f"active event count = {active_events}"),
        Check("target_event_exists", latest_event_dt is not None, f"target event = {event_id}"),
        Check("hazard_exists", latest_hazard_dt is not None, f"hazard generated_at = {latest_hazard_update}"),
        Check("event_is_fresh", bool(latest_event_dt and latest_event_dt >= freshness_cutoff), f"event updated_at = {latest_event_update}"),
        Check(
            "hazard_is_fresh",
            bool(latest_hazard_dt and latest_hazard_dt >= freshness_cutoff),
            f"hazard generated_at = {latest_hazard_update}",
        ),
        Check("profiles_available", counts["v2_entity_profiles"] >= 8, f"profiles = {counts['v2_entity_profiles']}"),
        Check("pending_proposal_available", pending >= 1, f"pending proposals = {pending}"),
        Check("approved_proposal_available", approved >= 1, f"approved proposals = {approved}"),
        Check("warnings_available", warnings >= 2, f"audience warnings = {warnings}"),
        Check("execution_logs_available", execution_logs >= 1, f"execution logs = {execution_logs}"),
        Check("agent_results_available", counts["v2_agent_results"] >= 6, f"agent results = {counts['v2_agent_results']}"),
        Check("audit_records_available", counts["v2_audit_records"] >= 1, f"audit records = {counts['v2_audit_records']}"),
    ]

    service_checks: list[Check] = []
    try:
        system = FloodWarningSystem(db_path)
        overview = system.agent_twin.get_twin_overview(event_id)
        council = system.agent_twin.get_agent_council(event_id)
        focus_id = overview.lead_object_id
        focus = system.agent_twin.get_focus_object(event_id, focus_id) if focus_id else None
        service_checks.extend(
            [
                Check("twin_overview_loads", len(overview.focus_objects) >= 5, f"focus objects = {len(overview.focus_objects)}"),
                Check("map_layers_load", len(overview.map_layers) >= 5, f"map layers = {len(overview.map_layers)}"),
                Check("agent_council_loads", len(council.roles) >= 4, f"council roles = {len(council.roles)}"),
                Check("focus_object_loads", focus is not None, f"lead object = {focus_id}"),
            ]
        )
    except Exception as exc:  # pragma: no cover - CLI guardrail
        service_checks.append(Check("service_contracts_load", False, f"{type(exc).__name__}: {exc}"))

    all_checks = checks + service_checks
    return {
        "db_path": str(db_path),
        "event_id": event_id,
        "counts": counts,
        "checks": [check.__dict__ for check in all_checks],
        "passed": all(check.passed for check in all_checks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an AgentTwin demo database.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Demo database path.")
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID, help="Primary demo event id.")
    parser.add_argument("--freshness-hours", type=int, default=48, help="Freshness threshold for demo data.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_db(Path(args.db_path).resolve(), event_id=args.event_id, freshness_hours=args.freshness_hours)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Demo DB: {report['db_path']}")
        print(f"Primary event: {report['event_id']}")
        for check in report["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"[{mark}] {check['name']}: {check['detail']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
