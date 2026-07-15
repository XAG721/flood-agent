from __future__ import annotations



from ..response_workflow.models import (
    ContractVersionRecord,
    DocumentLifecycleEvent,
    DocumentParseRecord,
    DocumentSourceRecord,
    DocumentVersionRecord,
    IndexBuildRecord,
)


class ResponseDocumentRepositoryMixin:
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

    def save_document_version_bundle(
        self,
        record: DocumentVersionRecord,
        source: DocumentSourceRecord,
        lifecycle_event: DocumentLifecycleEvent,
    ) -> None:
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
            conn.execute(
                "INSERT INTO response_document_sources("
                "version_id, source_hash, created_at, payload"
                ") VALUES (?, ?, ?, ?)",
                (
                    source.version_id,
                    source.source_hash,
                    source.created_at.isoformat(),
                    self._secure_dump(source),
                ),
            )
            conn.execute(
                "INSERT INTO response_document_lifecycle_events("
                "lifecycle_event_id, version_id, status, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    lifecycle_event.lifecycle_event_id,
                    lifecycle_event.version_id,
                    lifecycle_event.status.value,
                    lifecycle_event.created_at.isoformat(),
                    self._secure_dump(lifecycle_event),
                ),
            )

    def get_document_version(self, version_id: str) -> DocumentVersionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_document_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        return self._secure_load(DocumentVersionRecord, row["payload"]) if row else None

    def list_document_versions(
        self, document_id: str | None = None
    ) -> list[DocumentVersionRecord]:
        query = "SELECT payload FROM response_document_versions"
        values: list[object] = []
        if document_id:
            query += " WHERE document_id = ?"
            values.append(document_id)
        query += " ORDER BY document_id, version_number"
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            self._secure_load(DocumentVersionRecord, row["payload"]) for row in rows
        ]

    def get_document_source(self, version_id: str) -> DocumentSourceRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_document_sources WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        return self._secure_load(DocumentSourceRecord, row["payload"]) if row else None

    def save_document_parse(self, record: DocumentParseRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO response_document_parses("
                "parse_id, version_id, parser_version, source_hash, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.parse_id,
                    record.version_id,
                    record.parser_version,
                    record.source_hash,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_document_parses(self, version_id: str) -> list[DocumentParseRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_document_parses "
                "WHERE version_id = ? ORDER BY created_at, rowid",
                (version_id,),
            ).fetchall()
        return [self._secure_load(DocumentParseRecord, row["payload"]) for row in rows]

    def save_document_lifecycle_event(self, record: DocumentLifecycleEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_document_lifecycle_events("
                "lifecycle_event_id, version_id, status, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    record.lifecycle_event_id,
                    record.version_id,
                    record.status.value,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_document_lifecycle_events(
        self, version_id: str
    ) -> list[DocumentLifecycleEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_document_lifecycle_events "
                "WHERE version_id = ? ORDER BY created_at, rowid",
                (version_id,),
            ).fetchall()
        return [
            self._secure_load(DocumentLifecycleEvent, row["payload"]) for row in rows
        ]

    def save_index_build(self, record: IndexBuildRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_index_builds("
                "build_id, document_version_id, status, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    record.build_id,
                    record.document_version_id,
                    record.status.value,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def get_index_build(self, build_id: str) -> IndexBuildRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_index_builds WHERE build_id = ?",
                (build_id,),
            ).fetchone()
        return self._secure_load(IndexBuildRecord, row["payload"]) if row else None

    def list_index_builds(self, version_id: str) -> list[IndexBuildRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_index_builds "
                "WHERE document_version_id = ? ORDER BY created_at, rowid",
                (version_id,),
            ).fetchall()
        return [self._secure_load(IndexBuildRecord, row["payload"]) for row in rows]

    def save_contract_version(self, record: ContractVersionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO response_contract_versions("
                "contract_type, version_id, content_hash, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    record.contract_type,
                    record.version_id,
                    record.content_hash,
                    record.created_at.isoformat(),
                    self._secure_dump(record),
                ),
            )

    def list_contract_versions(self, contract_type: str) -> list[ContractVersionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_contract_versions "
                "WHERE contract_type = ? ORDER BY created_at, rowid",
                (contract_type,),
            ).fetchall()
        return [self._secure_load(ContractVersionRecord, row["payload"]) for row in rows]
