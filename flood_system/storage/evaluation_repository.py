from __future__ import annotations

from ..v2.models import AgentTimelineEntry, EvaluationReport, SupervisorRunRecord


class EvaluationRepositoryMixin:
    def save_v2_evaluation_report(self, report: EvaluationReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_evaluation_reports (report_id, created_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (report.report_id, report.created_at.isoformat(), report.model_dump_json()),
            )

    def get_v2_evaluation_report(self, report_id: str) -> EvaluationReport | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_evaluation_reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        return EvaluationReport.model_validate_json(row["payload"]) if row else None

    def list_v2_evaluation_reports(self, *, limit: int = 20) -> list[EvaluationReport]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_evaluation_reports
                ORDER BY created_at DESC, report_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [EvaluationReport.model_validate_json(row["payload"]) for row in rows]

    def save_v2_supervisor_run(self, record: SupervisorRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_supervisor_runs (supervisor_run_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(supervisor_run_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.supervisor_run_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_supervisor_runs(self, event_id: str, *, limit: int = 20) -> list[SupervisorRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_supervisor_runs
                WHERE event_id = ?
                ORDER BY created_at DESC, supervisor_run_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [SupervisorRunRecord.model_validate_json(row["payload"]) for row in rows]

    def list_v2_agent_timeline(self, event_id: str, *, limit: int = 120) -> list[AgentTimelineEntry]:
        task_events = self.list_v2_agent_task_events(event_id, limit=limit)
        trigger_events = self.list_v2_trigger_events(event_id, limit=limit)
        entries: list[AgentTimelineEntry] = []
        for trigger in trigger_events:
            summary = f"Trigger {trigger.trigger_type.value} is {trigger.status.value}."
            entries.append(
                AgentTimelineEntry(
                    entry_id=f"trigger_{trigger.trigger_id}",
                    event_id=event_id,
                    entry_type="trigger",
                    trigger_id=trigger.trigger_id,
                    trigger_type=trigger.trigger_type,
                    summary=summary,
                    payload=trigger.payload | {"status": trigger.status.value, "error_message": trigger.error_message},
                    created_at=trigger.created_at,
                )
            )
        for task_event in task_events:
            summary = task_event.payload.get("summary") if isinstance(task_event.payload.get("summary"), str) else task_event.event_type.value
            entries.append(
                AgentTimelineEntry(
                    entry_id=f"task_{task_event.task_event_id}",
                    event_id=event_id,
                    entry_type="task_event",
                    task_id=task_event.task_id,
                    task_event_type=task_event.event_type,
                    trigger_id=task_event.trigger_id,
                    agent_name=task_event.agent_name,
                    summary=summary,
                    payload=task_event.payload,
                    created_at=task_event.created_at,
                )
            )
        entries.sort(key=lambda item: item.created_at, reverse=True)
        return entries[:limit]
