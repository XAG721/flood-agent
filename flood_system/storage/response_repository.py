from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel

from ..response_workflow.models import (
    AlertSnapshot,
    CandidateRunRecord,
    AuditArchiveRecord,
    AuditArchiveVerificationResult,
    ApprovalRecord,
    BackupRestoreResult,
    BackupImportResult,
    BackupRetentionResult,
    DatabaseBackupRecord,
    KeyRotationResult,
    DistrictScenarioReport,
    EscalationRecord,
    EventReviewDraft,
    EventRiskObject,
    RiskObjectVersionSnapshot,
    EvidencePackageVersion,
    ResponseEvent,
    ResponseTask,
    TaskFeedback,
    TaskAssignmentRecord,
    TaskVersionSnapshot,
    TimelineEntry,
    FeatureFlagSetting,
    OutboxMessage,
    OutboxStatus,
    RuleEvaluationRecord,
    DeadlineExtensionRecord,
    MigrationBatchRecord,
    LegacyAdapterCallRecord,
    DocumentVersionRecord,
)


T = TypeVar("T", bound=BaseModel)


def _load(model_type: type[T], payload: str) -> T:
    return model_type.model_validate(json.loads(payload))


class ResponseRepositoryMixin:
    def _secure_dump(self, model: BaseModel) -> str:
        return self.data_protector.encrypt_payload(model.model_dump_json())

    def _secure_load(self, model_type: type[T], payload: str) -> T:
        return _load(model_type, self.data_protector.decrypt_payload(payload))

    def save_response_event(self, event: ResponseEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_events(event_id, status, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    status=excluded.status, updated_at=excluded.updated_at, payload=excluded.payload""",
                (event.event_id, event.status.value, event.updated_at.isoformat(), self._secure_dump(event)),
            )

    def get_response_event(self, event_id: str) -> ResponseEvent | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM response_events WHERE event_id = ?", (event_id,)).fetchone()
        return self._secure_load(ResponseEvent, row["payload"]) if row else None

    def list_response_events(self) -> list[ResponseEvent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM response_events ORDER BY updated_at DESC").fetchall()
        return [self._secure_load(ResponseEvent, row["payload"]) for row in rows]

    def save_alert_snapshot(self, snapshot: AlertSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_alert_snapshots(snapshot_id, event_id, version, created_at, payload) VALUES (?, ?, ?, ?, ?)",
                (snapshot.snapshot_id, snapshot.event_id, snapshot.version, snapshot.created_at.isoformat(), self._secure_dump(snapshot)),
            )

    def list_alert_snapshots(self, event_id: str) -> list[AlertSnapshot]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_alert_snapshots WHERE event_id = ? ORDER BY version", (event_id,)
            ).fetchall()
        return [self._secure_load(AlertSnapshot, row["payload"]) for row in rows]

    def save_event_risk_object(self, item: EventRiskObject) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_event_objects(event_id, object_id, verification_status, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(event_id, object_id) DO UPDATE SET
                    verification_status=excluded.verification_status, payload=excluded.payload""",
                (item.event_id, item.object_id, item.verification_status.value, item.created_at.isoformat(), self._secure_dump(item)),
            )

    def get_event_risk_object(self, event_id: str, object_id: str) -> EventRiskObject | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_event_objects WHERE event_id = ? AND object_id = ?", (event_id, object_id)
            ).fetchone()
        return self._secure_load(EventRiskObject, row["payload"]) if row else None

    def list_event_risk_objects(self, event_id: str) -> list[EventRiskObject]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_event_objects WHERE event_id = ? ORDER BY created_at", (event_id,)
            ).fetchall()
        return [self._secure_load(EventRiskObject, row["payload"]) for row in rows]

    def save_risk_object_version(self, snapshot: RiskObjectVersionSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_risk_object_versions("
                "snapshot_id, event_id, object_id, version, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    snapshot.snapshot_id,
                    snapshot.event_id,
                    snapshot.object_id,
                    snapshot.version,
                    snapshot.created_at.isoformat(),
                    self._secure_dump(snapshot),
                ),
            )

    def list_risk_object_versions(
        self, event_id: str, *, object_id: str | None = None
    ) -> list[RiskObjectVersionSnapshot]:
        query = "SELECT payload FROM response_risk_object_versions WHERE event_id = ?"
        values: list[object] = [event_id]
        if object_id:
            query += " AND object_id = ?"
            values.append(object_id)
        query += " ORDER BY object_id, version"
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [self._secure_load(RiskObjectVersionSnapshot, row["payload"]) for row in rows]

    def save_response_task(self, task: ResponseTask) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_tasks(task_id, event_id, object_id, status, version, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET status=excluded.status, version=excluded.version,
                    updated_at=excluded.updated_at, payload=excluded.payload""",
                (task.task_id, task.event_id, task.object_id, task.status.value, task.version, task.updated_at.isoformat(), self._secure_dump(task)),
            )

    def get_response_task(self, task_id: str) -> ResponseTask | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM response_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._secure_load(ResponseTask, row["payload"]) if row else None

    def list_response_tasks(self, event_id: str) -> list[ResponseTask]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_tasks WHERE event_id = ? ORDER BY updated_at DESC", (event_id,)
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

    def _insert_response_record(self, table: str, id_column: str, record: BaseModel, *, event_id: str, task_id: str | None, created_at: str) -> None:
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
            conn.execute(f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders})", values)

    def save_approval_record(self, record: ApprovalRecord) -> None:
        self._insert_response_record("response_approvals", "approval_id", record, event_id=record.event_id, task_id=record.task_id, created_at=record.created_at.isoformat())

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

    @staticmethod
    def _feature_scope_key(setting: FeatureFlagSetting) -> str:
        return "|".join(
            (
                setting.environment,
                setting.event_id or "*",
                setting.role.value if setting.role else "*",
                setting.scenario or "*",
            )
        )

    def save_feature_flag(self, setting: FeatureFlagSetting) -> None:
        scope_key = self._feature_scope_key(setting)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_feature_flags(flag_key, scope_key, enabled, version, updated_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(flag_key, scope_key) DO UPDATE SET enabled=excluded.enabled,
                    version=excluded.version, updated_at=excluded.updated_at, payload=excluded.payload""",
                (
                    setting.flag_key,
                    scope_key,
                    int(setting.enabled),
                    setting.version,
                    setting.updated_at.isoformat(),
                    self._secure_dump(setting),
                ),
            )

    def list_feature_flags(self) -> list[FeatureFlagSetting]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_feature_flags ORDER BY flag_key, scope_key"
            ).fetchall()
        return [self._secure_load(FeatureFlagSetting, row["payload"]) for row in rows]

    def save_outbox_message(self, message: OutboxMessage) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_outbox(
                    message_id, event_id, task_id, status, idempotency_key, attempts, updated_at, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET status=excluded.status, attempts=excluded.attempts,
                    updated_at=excluded.updated_at, payload=excluded.payload""",
                (
                    message.message_id,
                    message.event_id,
                    message.task_id,
                    message.status.value,
                    message.idempotency_key,
                    message.attempts,
                    message.updated_at.isoformat(),
                    message.created_at.isoformat(),
                    self._secure_dump(message),
                ),
            )

    def get_outbox_message(self, message_id: str) -> OutboxMessage | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM response_outbox WHERE message_id = ?", (message_id,)).fetchone()
        return self._secure_load(OutboxMessage, row["payload"]) if row else None

    def get_outbox_message_by_idempotency_key(self, idempotency_key: str) -> OutboxMessage | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_outbox WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return self._secure_load(OutboxMessage, row["payload"]) if row else None

    def list_outbox_messages(
        self, *, event_id: str | None = None, status: OutboxStatus | None = None, limit: int = 200
    ) -> list[OutboxMessage]:
        clauses: list[str] = []
        values: list[object] = []
        if event_id:
            clauses.append("event_id = ?")
            values.append(event_id)
        if status:
            clauses.append("status = ?")
            values.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload FROM response_outbox{where} ORDER BY created_at LIMIT ?", values
            ).fetchall()
        return [self._secure_load(OutboxMessage, row["payload"]) for row in rows]

    def save_candidate_run(self, run: CandidateRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_candidate_runs(run_id, event_id, created_at, payload) VALUES (?, ?, ?, ?)",
                (run.run_id, run.event_id, run.created_at.isoformat(), self._secure_dump(run)),
            )

    def list_candidate_runs(self, event_id: str) -> list[CandidateRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_candidate_runs WHERE event_id = ? ORDER BY created_at", (event_id,)
            ).fetchall()
        return [self._secure_load(CandidateRunRecord, row["payload"]) for row in rows]

    def save_evidence_package(self, package: EvidencePackageVersion) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_evidence_packages(
                    package_id, event_id, object_id, version, status, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    package.package_id,
                    package.event_id,
                    package.object_id,
                    package.version,
                    package.status,
                    package.created_at.isoformat(),
                    self._secure_dump(package),
                ),
            )

    def list_evidence_packages(
        self, event_id: str, *, object_id: str | None = None
    ) -> list[EvidencePackageVersion]:
        query = "SELECT payload FROM response_evidence_packages WHERE event_id = ?"
        values: list[object] = [event_id]
        if object_id:
            query += " AND object_id = ?"
            values.append(object_id)
        query += " ORDER BY created_at, version"
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [self._secure_load(EvidencePackageVersion, row["payload"]) for row in rows]

    def get_latest_evidence_package(self, package_id: str) -> EvidencePackageVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_evidence_packages WHERE package_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (package_id,),
            ).fetchone()
        return self._secure_load(EvidencePackageVersion, row["payload"]) if row else None

    def save_rule_evaluation(self, record: RuleEvaluationRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_rule_evaluations("
                "evaluation_id, event_id, task_id, task_version, outcome, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.evaluation_id,
                    record.event_id,
                    record.task_id,
                    record.task_version,
                    record.overall_outcome.value,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_rule_evaluations(
        self, *, event_id: str | None = None, task_id: str | None = None
    ) -> list[RuleEvaluationRecord]:
        clauses: list[str] = []
        values: list[object] = []
        if event_id:
            clauses.append("event_id = ?")
            values.append(event_id)
        if task_id:
            clauses.append("task_id = ?")
            values.append(task_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload FROM response_rule_evaluations{where} ORDER BY created_at", values
            ).fetchall()
        return [self._secure_load(RuleEvaluationRecord, row["payload"]) for row in rows]

    def save_deadline_extension(self, record: DeadlineExtensionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_deadline_extensions(extension_id, event_id, task_id, status, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(extension_id) DO UPDATE SET status=excluded.status, payload=excluded.payload""",
                (
                    record.extension_id,
                    record.event_id,
                    record.task_id,
                    record.status.value,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def get_deadline_extension(self, extension_id: str) -> DeadlineExtensionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_deadline_extensions WHERE extension_id = ?", (extension_id,)
            ).fetchone()
        return self._secure_load(DeadlineExtensionRecord, row["payload"]) if row else None

    def list_deadline_extensions(self, event_id: str) -> list[DeadlineExtensionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_deadline_extensions WHERE event_id = ? ORDER BY created_at", (event_id,)
            ).fetchall()
        return [self._secure_load(DeadlineExtensionRecord, row["payload"]) for row in rows]

    def list_schema_migrations(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version, checksum, applied_at FROM response_schema_migrations ORDER BY applied_at, version"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_migration_batch(self, record: MigrationBatchRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_migration_batches(batch_id, mapping_version, status, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record.batch_id,
                    record.mapping_version,
                    record.status,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_migration_batches(self) -> list[MigrationBatchRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM response_migration_batches ORDER BY created_at").fetchall()
        return [self._secure_load(MigrationBatchRecord, row["payload"]) for row in rows]

    def save_legacy_adapter_call(self, record: LegacyAdapterCallRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_legacy_adapter_calls("
                "call_id, legacy_endpoint, mapping_version, trace_id, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.call_id,
                    record.legacy_endpoint,
                    record.mapping_version,
                    record.trace_id,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_legacy_adapter_calls(self) -> list[LegacyAdapterCallRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM response_legacy_adapter_calls ORDER BY created_at").fetchall()
        return [self._secure_load(LegacyAdapterCallRecord, row["payload"]) for row in rows]

    def save_document_version(self, record: DocumentVersionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_document_versions("
                "version_id, document_id, version_number, source_hash, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.version_id,
                    record.document_id,
                    record.version_number,
                    record.source_hash,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def get_document_version(self, version_id: str) -> DocumentVersionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_document_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
        return self._secure_load(DocumentVersionRecord, row["payload"]) if row else None

    def list_document_versions(self, document_id: str | None = None) -> list[DocumentVersionRecord]:
        query = "SELECT payload FROM response_document_versions"
        values: list[object] = []
        if document_id:
            query += " WHERE document_id = ?"
            values.append(document_id)
        query += " ORDER BY document_id, version_number"
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [self._secure_load(DocumentVersionRecord, row["payload"]) for row in rows]

    def save_task_feedback(self, record: TaskFeedback) -> None:
        self._insert_response_record("response_feedback", "feedback_id", record, event_id=record.event_id, task_id=record.task_id, created_at=record.created_at.isoformat())

    def save_escalation_record(self, record: EscalationRecord) -> None:
        self._insert_response_record("response_escalations", "escalation_id", record, event_id=record.event_id, task_id=record.task_id, created_at=record.created_at.isoformat())

    def save_timeline_entry(self, record: TimelineEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_timeline(entry_id, event_id, task_id, object_id, entry_type, created_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.entry_id, record.event_id, record.task_id, record.object_id, record.entry_type, record.created_at.isoformat(), self._secure_dump(record)),
            )

    def list_timeline_entries(self, event_id: str) -> list[TimelineEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_timeline WHERE event_id = ? ORDER BY created_at, rowid", (event_id,)
            ).fetchall()
        return [self._secure_load(TimelineEntry, row["payload"]) for row in rows]

    def list_task_feedback(self, task_id: str) -> list[TaskFeedback]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_feedback WHERE task_id = ? ORDER BY created_at", (task_id,)
            ).fetchall()
        return [self._secure_load(TaskFeedback, row["payload"]) for row in rows]

    def list_event_feedback(self, event_id: str) -> list[TaskFeedback]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_feedback WHERE event_id = ? ORDER BY created_at", (event_id,)
            ).fetchall()
        return [self._secure_load(TaskFeedback, row["payload"]) for row in rows]

    def list_escalations(self, event_id: str) -> list[EscalationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_escalations WHERE event_id = ? ORDER BY created_at", (event_id,)
            ).fetchall()
        return [self._secure_load(EscalationRecord, row["payload"]) for row in rows]

    def save_event_review_draft(self, record: EventReviewDraft) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_review_drafts(review_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET payload=excluded.payload""",
                (record.review_id, record.event_id, record.created_at.isoformat(), self._secure_dump(record)),
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
                (record.report_id, record.event_id, record.created_at.isoformat(), self._secure_dump(record)),
            )

    def list_district_scenario_reports(self, event_id: str) -> list[DistrictScenarioReport]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_scenario_reports WHERE event_id = ? ORDER BY created_at DESC, rowid DESC",
                (event_id,),
            ).fetchall()
        return [self._secure_load(DistrictScenarioReport, row["payload"]) for row in rows]

    def get_latest_district_scenario_report(self, event_id: str) -> DistrictScenarioReport | None:
        reports = self.list_district_scenario_reports(event_id)
        return reports[0] if reports else None

    def create_database_backup(self, *, label: str, operator_id: str, terminal_id: str) -> DatabaseBackupRecord:
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")[:40] or "manual"
        created_at = datetime.now(timezone.utc)
        backup_id = f"BACKUP-{created_at:%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
        backup_dir = self.db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{backup_id}-{safe_label}.db"
        target = backup_dir / filename
        temporary = target.with_suffix(".tmp")
        source = sqlite3.connect(self.db_path)
        destination = sqlite3.connect(temporary)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        integrity = self._database_integrity_check(temporary)
        if integrity != "ok":
            temporary.unlink(missing_ok=True)
            raise ValueError(f"backup integrity check failed: {integrity}")
        temporary.replace(target)
        digest = self._file_sha256(target)
        manifest_filename = f"{backup_id}.manifest.json"
        manifest_path = backup_dir / manifest_filename
        record = DatabaseBackupRecord(
            backup_id=backup_id,
            label=label,
            database_name=self.db_path.name,
            backup_filename=filename,
            manifest_filename=manifest_filename,
            sha256=digest,
            size_bytes=target.stat().st_size,
            integrity_check=integrity,
            encryption_key_id=self.data_protector.key_id,
            source_instance_id=self.instance_id,
            created_by=operator_id,
            terminal_id=terminal_id,
            created_at=created_at,
        )
        signature_payload = record.model_dump(mode="json", exclude={"manifest_signature"})
        record = record.model_copy(update={"manifest_signature": self.backup_manifest_signer.sign(signature_payload)})
        manifest_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_database_backups(backup_id, created_at, payload) VALUES (?, ?, ?)",
                (record.backup_id, record.created_at.isoformat(), self._secure_dump(record)),
            )
        return record

    def list_database_backups(self) -> list[DatabaseBackupRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_database_backups ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [self._secure_load(DatabaseBackupRecord, row["payload"]) for row in rows]

    def get_database_backup(self, backup_id: str) -> DatabaseBackupRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_database_backups WHERE backup_id = ?", (backup_id,)
            ).fetchone()
        return self._secure_load(DatabaseBackupRecord, row["payload"]) if row else None

    def apply_backup_retention(
        self,
        *,
        keep_latest: int,
        max_age_days: int | None,
        dry_run: bool,
        operator_id: str,
        terminal_id: str,
    ) -> BackupRetentionResult:
        backups = self.list_database_backups()
        executed_at = datetime.now(timezone.utc)
        protected = {item.backup_id for item in backups[:keep_latest]}
        with self._connect() as conn:
            protected.update(
                str(row["backup_id"])
                for row in conn.execute("SELECT backup_id FROM response_key_rotation_state").fetchall()
            )
        policy_matches = []
        for record in backups[keep_latest:]:
            age_days = (executed_at - record.created_at).total_seconds() / 86400
            if record.backup_id in protected:
                continue
            if max_age_days is None or age_days >= max_age_days:
                policy_matches.append(record)
        backup_dir = (self.db_path.parent / "backups").resolve()

        def retained_files(record: DatabaseBackupRecord) -> list[Path]:
            paths = []
            for filename in (record.backup_filename, record.manifest_filename):
                if Path(filename).name != filename:
                    raise ValueError("stored backup filename contains path components")
                target = (backup_dir / filename).resolve()
                if target.parent != backup_dir:
                    raise ValueError("backup retention target escaped the backup directory")
                if target.is_file():
                    paths.append(target)
            return paths

        candidates = [(record, retained_files(record)) for record in policy_matches]
        already_pruned = [record.backup_id for record, paths in candidates if not paths]
        candidates = [(record, paths) for record, paths in candidates if paths]
        deleted_files: list[str] = []
        pruned_ids: list[str] = []
        if not dry_run:
            for record, paths in candidates:
                for target in paths:
                    target.unlink()
                    deleted_files.append(target.name)
                pruned_ids.append(record.backup_id)
        result = BackupRetentionResult(
            run_id=f"RETENTION-{executed_at:%Y%m%d%H%M%S}-{uuid4().hex[:8]}",
            keep_latest=keep_latest,
            max_age_days=max_age_days,
            dry_run=dry_run,
            protected_backup_ids=sorted(protected),
            candidate_backup_ids=[item.backup_id for item, _ in candidates],
            already_pruned_backup_ids=already_pruned,
            pruned_backup_ids=pruned_ids,
            deleted_files=deleted_files,
            executed_by=operator_id,
            terminal_id=terminal_id,
            executed_at=executed_at,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_backup_retention_runs(run_id, executed_at, payload) VALUES (?, ?, ?)",
                (result.run_id, result.executed_at.isoformat(), self._secure_dump(result)),
            )
        return result

    def create_audit_archive(
        self,
        *,
        event_id: str,
        label: str,
        bundle: dict,
        timeline_entries: int,
        timeline_head_hash: str | None,
        retention_days: int,
        operator_id: str,
        terminal_id: str,
    ) -> AuditArchiveRecord:
        created_at = datetime.now(timezone.utc)
        archive_id = f"AUDIT-{created_at:%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")[:40] or "audit"
        archive_dir = self.db_path.parent / "backups" / "audit-archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_filename = f"{archive_id}-{safe_label}.audit.json.enc"
        manifest_filename = f"{archive_id}.manifest.json"
        archive_path = archive_dir / archive_filename
        temporary = archive_path.with_suffix(".tmp")
        protected_payload = self.data_protector.encrypt_payload(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        temporary.write_text(protected_payload, encoding="utf-8")
        temporary.replace(archive_path)
        record = AuditArchiveRecord(
            archive_id=archive_id,
            event_id=event_id,
            label=label,
            archive_filename=archive_filename,
            manifest_filename=manifest_filename,
            sha256=self._file_sha256(archive_path),
            size_bytes=archive_path.stat().st_size,
            encryption_key_id=self.data_protector.key_id,
            timeline_entries=timeline_entries,
            timeline_head_hash=timeline_head_hash,
            retention_until=created_at + timedelta(days=retention_days),
            source_instance_id=self.instance_id,
            created_by=operator_id,
            terminal_id=terminal_id,
            created_at=created_at,
        )
        signature_payload = record.model_dump(mode="json", exclude={"manifest_signature"})
        record = record.model_copy(update={"manifest_signature": self.backup_manifest_signer.sign(signature_payload)})
        (archive_dir / manifest_filename).write_text(record.model_dump_json(indent=2), encoding="utf-8")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_audit_archives(archive_id, event_id, created_at, payload) VALUES (?, ?, ?, ?)",
                (record.archive_id, record.event_id, record.created_at.isoformat(), self._secure_dump(record)),
            )
        return record

    def list_audit_archives(self, event_id: str) -> list[AuditArchiveRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_audit_archives WHERE event_id = ? ORDER BY created_at DESC, rowid DESC",
                (event_id,),
            ).fetchall()
        return [self._secure_load(AuditArchiveRecord, row["payload"]) for row in rows]

    def verify_audit_archive(
        self, *, archive_id: str, operator_id: str, terminal_id: str
    ) -> AuditArchiveVerificationResult:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_audit_archives WHERE archive_id = ?", (archive_id,)
            ).fetchone()
        if row is None:
            raise LookupError(f"audit archive not found: {archive_id}")
        record = self._secure_load(AuditArchiveRecord, row["payload"])
        self.backup_manifest_signer.verify(
            record.model_dump(mode="json", exclude={"manifest_signature"}), record.manifest_signature
        )
        archive_dir = self.db_path.parent / "backups" / "audit-archives"
        archive_path = archive_dir / record.archive_filename
        manifest_path = archive_dir / record.manifest_filename
        if not archive_path.is_file() or not manifest_path.is_file():
            raise LookupError("audit archive requires both encrypted payload and manifest files")
        disk_record = AuditArchiveRecord.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if disk_record.model_dump() != record.model_dump():
            raise ValueError("audit archive manifest does not match the database record")
        sha_verified = self._file_sha256(archive_path) == record.sha256
        if not sha_verified:
            raise ValueError("audit archive SHA-256 verification failed")
        payload = json.loads(self.data_protector.decrypt_payload(archive_path.read_text(encoding="utf-8")))
        timeline = payload.get("dashboard", {}).get("timeline", [])
        head_hash = timeline[-1].get("record_hash") if timeline else None
        head_verified = head_hash == record.timeline_head_hash and len(timeline) == record.timeline_entries
        result = AuditArchiveVerificationResult(
            archive_id=record.archive_id,
            event_id=record.event_id,
            signature_verified=True,
            sha256_verified=True,
            decryption_verified=True,
            timeline_head_verified=head_verified,
            status="verified" if head_verified else "failed",
            verified_by=operator_id,
            terminal_id=terminal_id,
            verified_at=datetime.now(timezone.utc),
        )
        return result

    def restore_database_backup(
        self, *, backup_id: str, operator_id: str, terminal_id: str
    ) -> BackupRestoreResult:
        record = self.get_database_backup(backup_id)
        if record is None:
            raise LookupError(f"database backup not found: {backup_id}")
        if record.manifest_signature:
            self.backup_manifest_signer.verify(
                record.model_dump(mode="json", exclude={"manifest_signature"}),
                record.manifest_signature,
            )
        backup_path = self.db_path.parent / "backups" / record.backup_filename
        if not backup_path.is_file():
            raise LookupError(f"database backup file not found: {record.backup_filename}")
        digest_matches = self._file_sha256(backup_path) == record.sha256
        if not digest_matches:
            raise ValueError("backup SHA-256 verification failed")
        if not self.data_protector.has_key(record.encryption_key_id):
            raise ValueError("backup encryption key identifier is not present in the active key ring")
        source_integrity = self._database_integrity_check(backup_path)
        if source_integrity != "ok":
            raise ValueError(f"backup integrity check failed: {source_integrity}")
        self._verify_backup_payload_decryption(backup_path)
        restored_at = datetime.now(timezone.utc)
        restore_id = f"RESTORE-{restored_at:%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
        restore_dir = self.db_path.parent / "backups" / "restores"
        restore_dir.mkdir(parents=True, exist_ok=True)
        restored_filename = f"{restore_id}.db"
        restored_path = restore_dir / restored_filename
        source = sqlite3.connect(backup_path)
        destination = sqlite3.connect(restored_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        restored_integrity = self._database_integrity_check(restored_path)
        status = "verified" if restored_integrity == "ok" else "failed"
        result = BackupRestoreResult(
            restore_id=restore_id,
            backup_id=backup_id,
            restored_filename=str(Path("restores") / restored_filename),
            sha256_verified=digest_matches,
            integrity_check=restored_integrity,
            status=status,
            restored_by=operator_id,
            terminal_id=terminal_id,
            restored_at=restored_at,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_backup_restores(restore_id, backup_id, restored_at, payload) VALUES (?, ?, ?, ?)",
                (result.restore_id, result.backup_id, result.restored_at.isoformat(), self._secure_dump(result)),
            )
        return result

    def import_database_backup(
        self,
        *,
        manifest_filename: str,
        backup_filename: str,
        operator_id: str,
        terminal_id: str,
    ) -> BackupImportResult:
        for filename in (manifest_filename, backup_filename):
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise ValueError("backup import filenames must not contain path components")
        backup_dir = self.db_path.parent / "backups"
        incoming_dir = backup_dir / "incoming"
        manifest_path = incoming_dir / manifest_filename
        backup_path = incoming_dir / backup_filename
        if not manifest_path.is_file() or not backup_path.is_file():
            raise LookupError("backup import requires both manifest and database files in backups/incoming")
        try:
            record = DatabaseBackupRecord.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError("backup manifest is invalid") from exc
        if record.manifest_filename != manifest_filename or record.backup_filename != backup_filename:
            raise ValueError("backup manifest filenames do not match the supplied files")
        self.backup_manifest_signer.verify(
            record.model_dump(mode="json", exclude={"manifest_signature"}),
            record.manifest_signature,
        )
        if not self.data_protector.has_key(record.encryption_key_id):
            raise ValueError("backup encryption key identifier is not present in the active key ring")
        sha_verified = self._file_sha256(backup_path) == record.sha256
        if not sha_verified or backup_path.stat().st_size != record.size_bytes:
            raise ValueError("backup file digest or size verification failed")
        integrity = self._database_integrity_check(backup_path)
        if integrity != "ok":
            raise ValueError(f"backup integrity check failed: {integrity}")
        self._verify_backup_payload_decryption(backup_path)
        existing = self.get_database_backup(record.backup_id)
        if existing is not None and existing.model_dump() != record.model_dump():
            raise ValueError("a different backup record already uses this backup_id")
        backup_dir.mkdir(parents=True, exist_ok=True)
        canonical_backup = backup_dir / record.backup_filename
        canonical_manifest = backup_dir / record.manifest_filename
        if not canonical_backup.exists():
            shutil.copy2(backup_path, canonical_backup)
        if not canonical_manifest.exists():
            shutil.copy2(manifest_path, canonical_manifest)
        if existing is None:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO response_database_backups(backup_id, created_at, payload) VALUES (?, ?, ?)",
                    (record.backup_id, record.created_at.isoformat(), self._secure_dump(record)),
                )
        imported_at = datetime.now(timezone.utc)
        result = BackupImportResult(
            backup_id=record.backup_id,
            source_instance_id=record.source_instance_id,
            manifest_signature_verified=True,
            sha256_verified=True,
            integrity_check=integrity,
            decryption_verified=True,
            status="verified",
            imported_by=operator_id,
            terminal_id=terminal_id,
            imported_at=imported_at,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_backup_imports(backup_id, imported_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(backup_id) DO UPDATE SET imported_at=excluded.imported_at, payload=excluded.payload",
                (record.backup_id, imported_at.isoformat(), self._secure_dump(result)),
            )
        return result

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _database_integrity_check(path: Path) -> str:
        try:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                row = conn.execute("PRAGMA integrity_check").fetchone()
            finally:
                conn.close()
            return str(row[0]) if row else "no-result"
        except sqlite3.DatabaseError as exc:
            return f"error: {exc}"

    def _verify_backup_payload_decryption(self, path: Path) -> None:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT payload FROM response_events LIMIT 1").fetchone()
        finally:
            conn.close()
        if row:
            self.data_protector.decrypt_payload(str(row[0]))

    def consume_identity_nonce(self, nonce: str, *, expires_at: datetime) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM response_identity_nonces WHERE expires_at < ?", (now,))
                conn.execute(
                    "INSERT INTO response_identity_nonces(nonce, expires_at) VALUES (?, ?)",
                    (nonce, expires_at.isoformat()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def rotate_data_encryption_key(
        self,
        *,
        new_key: bytes,
        backup_id: str,
        operator_id: str,
        terminal_id: str,
    ) -> KeyRotationResult:
        old_protector = self.data_protector
        new_protector = old_protector.with_active_key(new_key)
        if new_protector.key_id == old_protector.key_id:
            raise ValueError("new encryption key must differ from the active key")
        with self._connect() as conn:
            pending = conn.execute(
                "SELECT rotation_id FROM response_key_rotation_state WHERE status = 'pending_activation' LIMIT 1"
            ).fetchone()
        if pending is not None:
            raise ValueError("another encryption key rotation is pending activation")
        started_at = datetime.now(timezone.utc)
        rotation_id = f"ROTATE-{started_at:%Y%m%d%H%M%S}-{uuid4().hex[:8]}"
        tables = (
            "response_events", "response_alert_snapshots", "response_event_objects", "response_risk_object_versions", "response_tasks",
            "response_task_versions", "response_task_assignments",
            "response_approvals", "response_feedback", "response_escalations", "response_timeline",
            "response_review_drafts", "response_scenario_reports", "response_database_backups",
            "response_backup_restores", "response_backup_imports", "response_backup_retention_runs",
            "response_audit_archives",
            "response_feature_flags",
            "response_outbox",
            "response_candidate_runs",
            "response_evidence_packages",
            "response_rule_evaluations",
            "response_deadline_extensions",
            "response_migration_batches",
            "response_legacy_adapter_calls",
            "response_document_versions",
        )
        records_reencrypted = 0
        env_managed = bool(os.getenv("FLOOD_DATA_ENCRYPTION_KEY", "").strip())
        self._rotation_in_progress = True
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO response_key_rotation_state("
                    "rotation_id, backup_id, old_key_id, new_key_id, status, started_at"
                    ") VALUES (?, ?, ?, ?, 'pending_activation', ?)",
                    (rotation_id, backup_id, old_protector.key_id, new_protector.key_id, started_at.isoformat()),
                )
                for table in tables:
                    rows = conn.execute(f"SELECT rowid, payload FROM {table}").fetchall()
                    for row in rows:
                        raw_payload = old_protector.decrypt_payload(str(row["payload"]))
                        conn.execute(
                            f"UPDATE {table} SET payload = ? WHERE rowid = ?",
                            (new_protector.encrypt_payload(raw_payload), row["rowid"]),
                        )
                        records_reencrypted += 1
        finally:
            self._rotation_in_progress = False
        self.data_protector = new_protector
        status = "pending_activation"
        if not env_managed:
            key_path = self.db_path.with_suffix(f"{self.db_path.suffix}.key")
            temporary = key_path.with_suffix(f"{key_path.suffix}.next")
            temporary.write_bytes(new_key + b"\n")
            temporary.replace(key_path)
            with self._connect() as conn:
                conn.execute(
                    "UPDATE response_key_rotation_state SET status = 'activated', completed_at = ? WHERE rotation_id = ?",
                    (datetime.now(timezone.utc).isoformat(), rotation_id),
                )
            status = "activated"
        completed_at = datetime.now(timezone.utc)
        return KeyRotationResult(
            rotation_id=rotation_id,
            backup_id=backup_id,
            old_key_id=old_protector.key_id,
            new_key_id=new_protector.key_id,
            records_reencrypted=records_reencrypted,
            status=status,
            activation_required=status == "pending_activation",
            rotated_by=operator_id,
            terminal_id=terminal_id,
            started_at=started_at,
            completed_at=completed_at,
        )
