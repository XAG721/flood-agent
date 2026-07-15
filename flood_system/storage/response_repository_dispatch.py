from __future__ import annotations



from ..response_workflow.models import (
    CandidateObjectListVersion,
    CandidateRunRecord,
    DispatchCallbackRecord,
    EvidencePackageVersion,
    FeatureFlagSetting,
    OutboxMessage,
    OutboxStatus,
    RuleEvaluationRecord,
    DeadlineExtensionRecord,
    MigrationBatchRecord,
    LegacyAdapterCallRecord,
)


class ResponseDispatchRepositoryMixin:
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
            row = conn.execute(
                "SELECT payload FROM response_outbox WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return self._secure_load(OutboxMessage, row["payload"]) if row else None

    def get_outbox_message_by_idempotency_key(
        self, idempotency_key: str
    ) -> OutboxMessage | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_outbox WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return self._secure_load(OutboxMessage, row["payload"]) if row else None

    def list_outbox_messages(
        self,
        *,
        event_id: str | None = None,
        status: OutboxStatus | None = None,
        limit: int = 200,
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
                f"SELECT payload FROM response_outbox{where} ORDER BY created_at LIMIT ?",
                values,
            ).fetchall()
        return [self._secure_load(OutboxMessage, row["payload"]) for row in rows]

    def save_dispatch_callback(
        self, record: DispatchCallbackRecord
    ) -> DispatchCallbackRecord:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_dispatch_callbacks(
                    callback_id, message_id, event_id, task_id, external_id, version, status,
                    sequence_state, idempotency_key, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING""",
                (
                    record.callback_id,
                    record.message_id,
                    record.event_id,
                    record.task_id,
                    record.external_id,
                    record.version,
                    record.status.value,
                    record.sequence_state.value,
                    record.idempotency_key,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )
            row = conn.execute(
                "SELECT payload FROM response_dispatch_callbacks WHERE idempotency_key = ?",
                (record.idempotency_key,),
            ).fetchone()
        return self._secure_load(DispatchCallbackRecord, row["payload"])

    def get_dispatch_callback_by_idempotency_key(
        self, idempotency_key: str
    ) -> DispatchCallbackRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_dispatch_callbacks WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return (
            self._secure_load(DispatchCallbackRecord, row["payload"]) if row else None
        )

    def list_dispatch_callbacks(self, message_id: str) -> list[DispatchCallbackRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_dispatch_callbacks WHERE message_id = ? "
                "ORDER BY created_at, callback_id",
                (message_id,),
            ).fetchall()
        return [
            self._secure_load(DispatchCallbackRecord, row["payload"]) for row in rows
        ]

    def save_candidate_run(self, run: CandidateRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_candidate_runs(run_id, event_id, created_at, payload) VALUES (?, ?, ?, ?)",
                (
                    run.run_id,
                    run.event_id,
                    run.created_at.isoformat(),
                    self._secure_dump(run),
                ),
            )

    def save_candidate_object_list(
        self, record: CandidateObjectListVersion
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_candidate_object_lists("
                "list_id, event_id, version, content_hash, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.list_id,
                    record.event_id,
                    record.version,
                    record.content_hash,
                    record.frozen_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_candidate_object_lists(
        self, event_id: str
    ) -> list[CandidateObjectListVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_candidate_object_lists "
                "WHERE event_id = ? ORDER BY version",
                (event_id,),
            ).fetchall()
        return [
            self._secure_load(CandidateObjectListVersion, row["payload"])
            for row in rows
        ]

    def list_candidate_runs(self, event_id: str) -> list[CandidateRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_candidate_runs WHERE event_id = ? ORDER BY created_at",
                (event_id,),
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
        return [
            self._secure_load(EvidencePackageVersion, row["payload"]) for row in rows
        ]

    def get_latest_evidence_package(
        self, package_id: str
    ) -> EvidencePackageVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_evidence_packages WHERE package_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (package_id,),
            ).fetchone()
        return (
            self._secure_load(EvidencePackageVersion, row["payload"]) if row else None
        )

    def list_evidence_package_versions(
        self, package_id: str
    ) -> list[EvidencePackageVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_evidence_packages WHERE package_id = ? "
                "ORDER BY version",
                (package_id,),
            ).fetchall()
        return [
            self._secure_load(EvidencePackageVersion, row["payload"]) for row in rows
        ]

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
                f"SELECT payload FROM response_rule_evaluations{where} ORDER BY created_at",
                values,
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

    def get_deadline_extension(
        self, extension_id: str
    ) -> DeadlineExtensionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_deadline_extensions WHERE extension_id = ?",
                (extension_id,),
            ).fetchone()
        return (
            self._secure_load(DeadlineExtensionRecord, row["payload"]) if row else None
        )

    def list_deadline_extensions(self, event_id: str) -> list[DeadlineExtensionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_deadline_extensions WHERE event_id = ? ORDER BY created_at",
                (event_id,),
            ).fetchall()
        return [
            self._secure_load(DeadlineExtensionRecord, row["payload"]) for row in rows
        ]

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
            rows = conn.execute(
                "SELECT payload FROM response_migration_batches ORDER BY created_at"
            ).fetchall()
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
            rows = conn.execute(
                "SELECT payload FROM response_legacy_adapter_calls ORDER BY created_at"
            ).fetchall()
        return [
            self._secure_load(LegacyAdapterCallRecord, row["payload"]) for row in rows
        ]
