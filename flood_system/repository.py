from __future__ import annotations

import sqlite3
import hashlib
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .storage.agent_repository import AgentRepositoryMixin
from .storage.audit_archive_repository import AuditArchiveRepositoryMixin
from .storage.copilot_repository import CopilotRepositoryMixin
from .storage.evaluation_repository import EvaluationRepositoryMixin
from .storage.event_repository import EventRepositoryMixin
from .storage.memory_repository import MemoryRepositoryMixin
from .storage.notification_repository import NotificationRepositoryMixin
from .storage.proposal_repository import ProposalRepositoryMixin
from .storage.runtime_repository import RuntimeRepositoryMixin
from .storage.response_repository import ResponseRepositoryMixin
from .storage.schema import REPOSITORY_SCHEMA_SQL
from .storage.trigger_repository import TriggerRepositoryMixin
from .security import BackupManifestSigner, DataProtectionError, DataProtector


class SQLiteRepository(
    EventRepositoryMixin,
    RuntimeRepositoryMixin,
    ProposalRepositoryMixin,
    CopilotRepositoryMixin,
    TriggerRepositoryMixin,
    AgentRepositoryMixin,
    MemoryRepositoryMixin,
    EvaluationRepositoryMixin,
    AuditArchiveRepositoryMixin,
    NotificationRepositoryMixin,
    ResponseRepositoryMixin,
):
    """SQLite-backed repository composed from focused storage mixins."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._rotation_in_progress = False
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_protector = DataProtector.for_database(self.db_path)
        self.backup_manifest_signer = BackupManifestSigner.from_environment()
        self.instance_id = (
            os.getenv("FLOOD_INSTANCE_ID", "").strip()
            or f"{socket.gethostname()}-{self.db_path.stem}"
        )
        self._initialize()
        self._check_pending_key_rotation()
        self._migrate_response_payload_encryption()
        self.backfill_task_version_snapshots()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "response_key_rotation_authorized",
            0,
            lambda: 1 if self._rotation_in_progress else 0,
        )
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(REPOSITORY_SCHEMA_SQL)
            checksum = hashlib.sha256(REPOSITORY_SCHEMA_SQL.encode("utf-8")).hexdigest()
            version = f"schema-{checksum[:16]}"
            existing = conn.execute(
                "SELECT checksum FROM response_schema_migrations WHERE version = ?",
                (version,),
            ).fetchone()
            if existing is not None and existing["checksum"] != checksum:
                raise RuntimeError("repository schema migration checksum mismatch")
            conn.execute(
                "INSERT OR IGNORE INTO response_schema_migrations(version, checksum, applied_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                (version, checksum),
            )

    def _migrate_response_payload_encryption(self) -> None:
        tables = (
            "response_events",
            "response_alert_snapshots",
            "response_event_objects",
            "response_risk_object_versions",
            "response_risk_object_registry",
            "response_risk_object_registry_versions",
            "response_risk_object_imports",
            "response_tasks",
            "response_task_versions",
            "response_task_assignments",
            "response_approvals",
            "response_feedback",
            "response_escalations",
            "response_timeline",
            "response_review_drafts",
            "response_scenario_reports",
            "response_database_backups",
            "response_backup_restores",
            "response_backup_imports",
            "response_backup_retention_runs",
            "response_audit_archives",
            "response_feature_flags",
            "response_idempotency_records",
            "response_outbox",
            "response_dispatch_callbacks",
            "response_candidate_runs",
            "response_candidate_object_lists",
            "response_evidence_packages",
            "response_rule_evaluations",
            "response_deadline_extensions",
            "response_migration_batches",
            "response_legacy_adapter_calls",
            "response_document_versions",
        )
        self._rotation_in_progress = True
        try:
            with self._connect() as conn:
                for table in tables:
                    rows = conn.execute(
                        f"SELECT rowid, payload FROM {table}"
                    ).fetchall()
                    for row in rows:
                        payload = str(row["payload"])
                        if (
                            self.data_protector.payload_key_id(payload)
                            == self.data_protector.key_id
                        ):
                            continue
                        raw_payload = self.data_protector.decrypt_payload(payload)
                        conn.execute(
                            f"UPDATE {table} SET payload = ? WHERE rowid = ?",
                            (
                                self.data_protector.encrypt_payload(raw_payload),
                                row["rowid"],
                            ),
                        )
        finally:
            self._rotation_in_progress = False

    def _check_pending_key_rotation(self) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rotation_id, new_key_id FROM response_key_rotation_state "
                "WHERE status = 'pending_activation' ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return
            if self.data_protector.key_id != row["new_key_id"]:
                raise DataProtectionError(
                    "database encryption key rotation is pending activation; promote the new key before restart"
                )
            conn.execute(
                "UPDATE response_key_rotation_state SET status = 'activated', completed_at = CURRENT_TIMESTAMP "
                "WHERE rotation_id = ?",
                (row["rotation_id"],),
            )
