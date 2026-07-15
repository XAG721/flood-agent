from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel

from ..response_workflow.models import (
    ApprovalRecord,
    DistrictScenarioReport,
    EscalationRecord,
    EventReviewDraft,
    ResponseTask,
    TaskFeedback,
    TaskAssignmentRecord,
    TaskVersionSnapshot,
    TimelineEntry,
    OutboxMessage,
)


class ResponseTaskRepositoryMixin:
    def save_response_task(self, task: ResponseTask) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_tasks(task_id, event_id, object_id, status, version, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET status=excluded.status, version=excluded.version,
                    updated_at=excluded.updated_at, payload=excluded.payload""",
                (
                    task.task_id,
                    task.event_id,
                    task.object_id,
                    task.status.value,
                    task.version,
                    task.updated_at.isoformat(),
                    self._secure_dump(task),
                ),
            )

    def get_response_task(self, task_id: str) -> ResponseTask | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._secure_load(ResponseTask, row["payload"]) if row else None

    def list_response_tasks(self, event_id: str) -> list[ResponseTask]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_tasks WHERE event_id = ? ORDER BY updated_at DESC",
                (event_id,),
            ).fetchall()
        return [self._secure_load(ResponseTask, row["payload"]) for row in rows]

    def save_task_version_snapshot(self, snapshot: TaskVersionSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_task_versions(snapshot_id, event_id, task_id, version, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot.snapshot_id,
                    snapshot.event_id,
                    snapshot.task_id,
                    snapshot.version,
                    snapshot.created_at.isoformat(),
                    self._secure_dump(snapshot),
                ),
            )

    def list_task_version_snapshots(self, task_id: str) -> list[TaskVersionSnapshot]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_task_versions WHERE task_id = ? ORDER BY version",
                (task_id,),
            ).fetchall()
        return [self._secure_load(TaskVersionSnapshot, row["payload"]) for row in rows]

    def backfill_task_version_snapshots(self) -> int:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tasks.payload FROM response_tasks AS tasks "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM response_task_versions AS versions WHERE versions.task_id = tasks.task_id"
                ")"
            ).fetchall()
        tasks = [self._secure_load(ResponseTask, row["payload"]) for row in rows]
        for task in tasks:
            self.save_task_version_snapshot(
                TaskVersionSnapshot(
                    snapshot_id=f"TASKVER-LEGACY-{uuid4().hex[:12]}",
                    event_id=task.event_id,
                    task_id=task.task_id,
                    version=task.version,
                    task=task,
                    change_type="legacy_baseline",
                    changed_fields=[],
                    note="升级时保存的当前版本基线；升级前更早版本无法重建",
                    created_by="system-migration",
                    terminal_id="repository-startup",
                    created_at=datetime.now(timezone.utc),
                )
            )
        return len(tasks)

    def save_task_assignment(self, assignment: TaskAssignmentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_task_assignments("
                "assignment_id, event_id, task_id, assignment_version, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    assignment.assignment_id,
                    assignment.event_id,
                    assignment.task_id,
                    assignment.assignment_version,
                    assignment.created_at.isoformat(),
                    self._secure_dump(assignment),
                ),
            )

    def list_task_assignments(self, task_id: str) -> list[TaskAssignmentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_task_assignments WHERE task_id = ? ORDER BY assignment_version",
                (task_id,),
            ).fetchall()
        return [self._secure_load(TaskAssignmentRecord, row["payload"]) for row in rows]

    def list_event_task_assignments(self, event_id: str) -> list[TaskAssignmentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_task_assignments WHERE event_id = ? "
                "ORDER BY created_at, assignment_version",
                (event_id,),
            ).fetchall()
        return [self._secure_load(TaskAssignmentRecord, row["payload"]) for row in rows]

    def _insert_response_record(
        self,
        table: str,
        id_column: str,
        record: BaseModel,
        *,
        event_id: str,
        task_id: str | None,
        created_at: str,
    ) -> None:
        record_id = getattr(record, id_column)
        columns = [id_column, "event_id"]
        values: list[object] = [record_id, event_id]
        if task_id is not None:
            columns.append("task_id")
            values.append(task_id)
        if table == "response_escalations":
            columns.append("resolved")
            values.append(int(getattr(record, "resolved")))
        columns.extend(["created_at", "payload"])
        values.extend([created_at, self._secure_dump(record)])
        placeholders = ", ".join("?" for _ in values)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )

    def save_approval_record(self, record: ApprovalRecord) -> None:
        self._insert_response_record(
            "response_approvals",
            "approval_id",
            record,
            event_id=record.event_id,
            task_id=record.task_id,
            created_at=record.created_at.isoformat(),
        )

    def commit_task_decision(
        self,
        approval: ApprovalRecord,
        task: ResponseTask,
        outbox: OutboxMessage | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_approvals(approval_id, event_id, task_id, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    approval.approval_id,
                    approval.event_id,
                    approval.task_id,
                    approval.created_at.isoformat(),
                    self._secure_dump(approval),
                ),
            )
            if outbox is not None:
                conn.execute(
                    "INSERT INTO response_outbox("
                    "message_id, event_id, task_id, status, idempotency_key, attempts, updated_at, created_at, payload"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        outbox.message_id,
                        outbox.event_id,
                        outbox.task_id,
                        outbox.status.value,
                        outbox.idempotency_key,
                        outbox.attempts,
                        outbox.updated_at.isoformat(),
                        outbox.created_at.isoformat(),
                        self._secure_dump(outbox),
                    ),
                )
            conn.execute(
                """INSERT INTO response_tasks(task_id, event_id, object_id, status, version, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET status=excluded.status, version=excluded.version,
                    updated_at=excluded.updated_at, payload=excluded.payload""",
                (
                    task.task_id,
                    task.event_id,
                    task.object_id,
                    task.status.value,
                    task.version,
                    task.updated_at.isoformat(),
                    self._secure_dump(task),
                ),
            )

    def save_task_feedback(self, record: TaskFeedback) -> None:
        self._insert_response_record(
            "response_feedback",
            "feedback_id",
            record,
            event_id=record.event_id,
            task_id=record.task_id,
            created_at=record.created_at.isoformat(),
        )

    def save_escalation_record(self, record: EscalationRecord) -> None:
        self._insert_response_record(
            "response_escalations",
            "escalation_id",
            record,
            event_id=record.event_id,
            task_id=record.task_id,
            created_at=record.created_at.isoformat(),
        )

    def save_timeline_entry(self, record: TimelineEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_timeline(entry_id, event_id, task_id, object_id, entry_type, created_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.entry_id,
                    record.event_id,
                    record.task_id,
                    record.object_id,
                    record.entry_type,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_timeline_entries(self, event_id: str) -> list[TimelineEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_timeline WHERE event_id = ? ORDER BY created_at, rowid",
                (event_id,),
            ).fetchall()
        return [self._secure_load(TimelineEntry, row["payload"]) for row in rows]

    def list_task_feedback(self, task_id: str) -> list[TaskFeedback]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_feedback WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            ).fetchall()
        return [self._secure_load(TaskFeedback, row["payload"]) for row in rows]

    def list_event_feedback(self, event_id: str) -> list[TaskFeedback]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_feedback WHERE event_id = ? ORDER BY created_at",
                (event_id,),
            ).fetchall()
        return [self._secure_load(TaskFeedback, row["payload"]) for row in rows]

    def list_escalations(self, event_id: str) -> list[EscalationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_escalations WHERE event_id = ? ORDER BY created_at",
                (event_id,),
            ).fetchall()
        return [self._secure_load(EscalationRecord, row["payload"]) for row in rows]

    def save_event_review_draft(self, record: EventReviewDraft) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_review_drafts(review_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET payload=excluded.payload""",
                (
                    record.review_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def get_latest_event_review_draft(self, event_id: str) -> EventReviewDraft | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_review_drafts WHERE event_id = ? ORDER BY created_at DESC LIMIT 1",
                (event_id,),
            ).fetchone()
        return self._secure_load(EventReviewDraft, row["payload"]) if row else None

    def save_district_scenario_report(self, record: DistrictScenarioReport) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_scenario_reports(report_id, event_id, created_at, payload) VALUES (?, ?, ?, ?)",
                (
                    record.report_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_district_scenario_reports(
        self, event_id: str
    ) -> list[DistrictScenarioReport]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_scenario_reports WHERE event_id = ? ORDER BY created_at DESC, rowid DESC",
                (event_id,),
            ).fetchall()
        return [
            self._secure_load(DistrictScenarioReport, row["payload"]) for row in rows
        ]

    def get_latest_district_scenario_report(
        self, event_id: str
    ) -> DistrictScenarioReport | None:
        reports = self.list_district_scenario_reports(event_id)
        return reports[0] if reports else None
