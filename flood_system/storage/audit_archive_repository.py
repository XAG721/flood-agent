from __future__ import annotations

import json
from datetime import datetime, timezone

from ..v2.models import (
    ArchiveRunRecord,
    ArchiveStatusView,
    AuditRecord,
    OperationalAlert,
    SupervisorHealthState,
)


class AuditArchiveRepositoryMixin:
    def save_supervisor_health_state(self, state: SupervisorHealthState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_supervisor_health_state (component_key, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(component_key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (state.component_key, state.updated_at.isoformat(), state.model_dump_json()),
            )

    def get_supervisor_health_state(self, component_key: str = "supervisor_loop") -> SupervisorHealthState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_supervisor_health_state WHERE component_key = ?",
                (component_key,),
            ).fetchone()
        return SupervisorHealthState.model_validate_json(row["payload"]) if row else None

    def save_operational_alert(self, alert: OperationalAlert) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_operational_alerts (
                    alert_id, source_type, severity, status, event_id, first_seen_at, last_seen_at, resolved_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alert_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    severity = excluded.severity,
                    status = excluded.status,
                    event_id = excluded.event_id,
                    first_seen_at = excluded.first_seen_at,
                    last_seen_at = excluded.last_seen_at,
                    resolved_at = excluded.resolved_at,
                    payload = excluded.payload
                """,
                (
                    alert.alert_id,
                    alert.source_type,
                    alert.severity.value,
                    alert.status.value,
                    alert.event_id,
                    alert.first_seen_at.isoformat(),
                    alert.last_seen_at.isoformat(),
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.model_dump_json(),
                ),
            )

    def list_operational_alerts(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        status: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 50,
    ) -> list[OperationalAlert]:
        query = "SELECT payload FROM v2_operational_alerts WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        if severity is not None:
            query += " AND severity = ?"
            parameters.append(severity)
        if source_type is not None:
            query += " AND source_type = ?"
            parameters.append(source_type)
        if status is not None:
            query += " AND status = ?"
            parameters.append(status)
        if from_ts is not None:
            query += " AND last_seen_at >= ?"
            parameters.append(from_ts)
        if to_ts is not None:
            query += " AND last_seen_at <= ?"
            parameters.append(to_ts)
        query += " ORDER BY last_seen_at DESC, alert_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [OperationalAlert.model_validate_json(row["payload"]) for row in rows]

    def add_audit_record(self, record: AuditRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_audit_records (audit_id, source_type, severity, event_id, session_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(audit_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    severity = excluded.severity,
                    event_id = excluded.event_id,
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.audit_id,
                    record.source_type,
                    record.severity.value,
                    record.event_id,
                    record.session_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_audit_records(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        query = "SELECT payload FROM v2_audit_records WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        if severity is not None:
            query += " AND severity = ?"
            parameters.append(severity)
        if source_type is not None:
            query += " AND source_type = ?"
            parameters.append(source_type)
        if from_ts is not None:
            query += " AND created_at >= ?"
            parameters.append(from_ts)
        if to_ts is not None:
            query += " AND created_at <= ?"
            parameters.append(to_ts)
        query += " ORDER BY created_at DESC, audit_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [AuditRecord.model_validate_json(row["payload"]) for row in rows]

    def save_archive_run(self, record: ArchiveRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_archive_runs (archive_run_id, started_at, completed_at, status, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(archive_run_id) DO UPDATE SET
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    record.archive_run_id,
                    record.started_at.isoformat(),
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.status,
                    record.model_dump_json(),
                ),
            )

    def list_archive_runs(self, *, limit: int = 10) -> list[ArchiveRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_archive_runs
                ORDER BY started_at DESC, archive_run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [ArchiveRunRecord.model_validate_json(row["payload"]) for row in rows]

    def archive_operational_records(
        self,
        *,
        now: datetime | None = None,
        hot_retention_days: int = 14,
        archive_retention_days: int = 180,
    ) -> ArchiveStatusView:
        now_dt = now or datetime.now(timezone.utc)
        hot_cutoff = now_dt.timestamp() - (hot_retention_days * 86400)
        archive_cutoff = now_dt.timestamp() - (archive_retention_days * 86400)
        archive_id_seed = int(now_dt.timestamp())
        archived_count = 0
        deleted_count = 0
        touched_tables: list[str] = []
        hot_tables = {
            "v2_agent_tasks": ("task_id", None, None, "agent_task"),
            "v2_agent_results": ("result_id", None, None, "agent_result"),
            "v2_copilot_memory_events": ("memory_event_id", None, "session_id", "memory_event"),
            "v2_copilot_tool_executions": ("execution_id", None, "session_id", "tool_execution"),
            "v2_supervisor_runs": ("supervisor_run_id", "event_id", None, "supervisor_run"),
            "v2_trigger_events": ("trigger_id", "event_id", None, "trigger_event"),
            "v2_agent_task_events": ("task_event_id", "event_id", None, "agent_task_event"),
        }

        with self._connect() as conn:
            for table_name, (id_column, event_column, session_column, record_kind) in hot_tables.items():
                rows = conn.execute(f"SELECT payload FROM {table_name}").fetchall()
                stale_payloads = []
                for row in rows:
                    payload = json.loads(row["payload"])
                    created_at = payload.get("created_at")
                    if not created_at:
                        continue
                    try:
                        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if created_dt.timestamp() >= hot_cutoff:
                        continue
                    stale_payloads.append(payload)
                if not stale_payloads:
                    continue
                touched_tables.append(table_name)
                for index, payload in enumerate(stale_payloads, start=1):
                    source_id = str(payload.get(id_column))
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO v2_archived_records (
                            archive_id, source_table, source_id, event_id, session_id, record_kind, created_at, archived_at, payload
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"arch_{archive_id_seed}_{table_name}_{index}",
                            table_name,
                            source_id,
                            payload.get(event_column) if event_column else None,
                            payload.get(session_column) if session_column else None,
                            record_kind,
                            payload["created_at"],
                            now_dt.isoformat(),
                            json.dumps(payload),
                        ),
                    )
                    conn.execute(f"DELETE FROM {table_name} WHERE {id_column} = ?", (source_id,))
                    archived_count += 1

            archived_rows = conn.execute("SELECT archive_id, created_at FROM v2_archived_records").fetchall()
            for row in archived_rows:
                try:
                    created_dt = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if created_dt.timestamp() < archive_cutoff:
                    conn.execute("DELETE FROM v2_archived_records WHERE archive_id = ?", (row["archive_id"],))
                    deleted_count += 1

            hot_record_count = 0
            for table_name in hot_tables:
                count_row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
                hot_record_count += int(count_row["count"]) if count_row else 0
            archived_record_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM v2_archived_records").fetchone()["count"]
            )

        run = ArchiveRunRecord(
            archive_run_id=f"archive_{archive_id_seed}",
            status="completed",
            hot_records_archived=archived_count,
            expired_archives_deleted=deleted_count,
            tables_touched=touched_tables,
            started_at=now_dt,
            completed_at=datetime.now(timezone.utc),
        )
        self.save_archive_run(run)
        return ArchiveStatusView(
            hot_retention_days=hot_retention_days,
            archive_retention_days=archive_retention_days,
            hot_record_count=hot_record_count,
            archived_record_count=archived_record_count,
            last_archive_run=run,
            latest_archive_runs=self.list_archive_runs(limit=5),
        )

    def get_archive_status(
        self,
        *,
        hot_retention_days: int = 14,
        archive_retention_days: int = 180,
    ) -> ArchiveStatusView:
        hot_tables = [
            "v2_agent_tasks",
            "v2_agent_results",
            "v2_copilot_memory_events",
            "v2_copilot_tool_executions",
            "v2_supervisor_runs",
            "v2_trigger_events",
            "v2_agent_task_events",
        ]
        with self._connect() as conn:
            hot_record_count = 0
            for table_name in hot_tables:
                count_row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
                hot_record_count += int(count_row["count"]) if count_row else 0
            archived_record_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM v2_archived_records").fetchone()["count"]
            )
        latest_runs = self.list_archive_runs(limit=5)
        return ArchiveStatusView(
            hot_retention_days=hot_retention_days,
            archive_retention_days=archive_retention_days,
            hot_record_count=hot_record_count,
            archived_record_count=archived_record_count,
            last_archive_run=latest_runs[0] if latest_runs else None,
            latest_archive_runs=latest_runs,
        )
