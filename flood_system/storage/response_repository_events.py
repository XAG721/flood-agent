from __future__ import annotations



from ..response_workflow.models import (
    AlertSnapshot,
    CandidateRunRecord,
    EventRiskObject,
    RiskObjectVersionSnapshot,
    RiskObjectRegistryImportResult,
    RiskObjectRegistryRecord,
    RiskObjectRegistryVersionSnapshot,
    EvidencePackageVersion,
    ResponseEvent,
)


class ResponseEventRepositoryMixin:
    def save_response_event(self, event: ResponseEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_events(event_id, status, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    status=excluded.status, updated_at=excluded.updated_at, payload=excluded.payload""",
                (
                    event.event_id,
                    event.status.value,
                    event.updated_at.isoformat(),
                    self._secure_dump(event),
                ),
            )

    def get_response_event(self, event_id: str) -> ResponseEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._secure_load(ResponseEvent, row["payload"]) if row else None

    def list_response_events(self) -> list[ResponseEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_events ORDER BY updated_at DESC"
            ).fetchall()
        return [self._secure_load(ResponseEvent, row["payload"]) for row in rows]

    def save_alert_snapshot(self, snapshot: AlertSnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO response_alert_snapshots(snapshot_id, event_id, version, created_at, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    snapshot.snapshot_id,
                    snapshot.event_id,
                    snapshot.version,
                    snapshot.created_at.isoformat(),
                    self._secure_dump(snapshot),
                ),
            )

    def list_alert_snapshots(self, event_id: str) -> list[AlertSnapshot]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_alert_snapshots WHERE event_id = ? ORDER BY version",
                (event_id,),
            ).fetchall()
        return [self._secure_load(AlertSnapshot, row["payload"]) for row in rows]

    def save_event_risk_object(self, item: EventRiskObject) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO response_event_objects(event_id, object_id, verification_status, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(event_id, object_id) DO UPDATE SET
                    verification_status=excluded.verification_status, payload=excluded.payload""",
                (
                    item.event_id,
                    item.object_id,
                    item.verification_status.value,
                    item.created_at.isoformat(),
                    self._secure_dump(item),
                ),
            )

    def get_event_risk_object(
        self, event_id: str, object_id: str
    ) -> EventRiskObject | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_event_objects WHERE event_id = ? AND object_id = ?",
                (event_id, object_id),
            ).fetchone()
        return self._secure_load(EventRiskObject, row["payload"]) if row else None

    def list_event_risk_objects(self, event_id: str) -> list[EventRiskObject]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_event_objects WHERE event_id = ? ORDER BY created_at",
                (event_id,),
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
        return [
            self._secure_load(RiskObjectVersionSnapshot, row["payload"]) for row in rows
        ]

    def get_risk_object_registry_record(
        self, area_id: str, object_id: str
    ) -> RiskObjectRegistryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM response_risk_object_registry "
                "WHERE area_id = ? AND object_id = ?",
                (area_id, object_id),
            ).fetchone()
        return (
            self._secure_load(RiskObjectRegistryRecord, row["payload"]) if row else None
        )

    def list_risk_object_registry(
        self,
        *,
        area_id: str,
        include_inactive: bool = False,
    ) -> list[RiskObjectRegistryRecord]:
        query = "SELECT payload FROM response_risk_object_registry WHERE area_id = ?"
        values: list[object] = [area_id]
        if not include_inactive:
            query += " AND registry_status = 'active'"
        query += " ORDER BY object_id"
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            self._secure_load(RiskObjectRegistryRecord, row["payload"]) for row in rows
        ]

    def list_risk_object_registry_versions(
        self,
        *,
        area_id: str,
        object_id: str | None = None,
    ) -> list[RiskObjectRegistryVersionSnapshot]:
        query = "SELECT payload FROM response_risk_object_registry_versions WHERE area_id = ?"
        values: list[object] = [area_id]
        if object_id:
            query += " AND object_id = ?"
            values.append(object_id)
        query += " ORDER BY object_id, registry_version"
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            self._secure_load(RiskObjectRegistryVersionSnapshot, row["payload"])
            for row in rows
        ]

    def list_risk_object_imports(
        self, *, area_id: str
    ) -> list[RiskObjectRegistryImportResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_risk_object_imports "
                "WHERE area_id = ? ORDER BY created_at DESC",
                (area_id,),
            ).fetchall()
        return [
            self._secure_load(RiskObjectRegistryImportResult, row["payload"])
            for row in rows
        ]

    def risk_object_registry_metrics(self) -> dict[str, int]:
        with self._connect() as conn:
            status_rows = conn.execute(
                "SELECT registry_status, COUNT(*) AS count "
                "FROM response_risk_object_registry GROUP BY registry_status"
            ).fetchall()
            run_rows = conn.execute(
                "SELECT payload FROM response_candidate_runs"
            ).fetchall()
            import_rows = conn.execute(
                "SELECT payload FROM response_risk_object_imports"
            ).fetchall()
            candidate_list_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM response_candidate_object_lists"
                ).fetchone()["count"]
            )
        metrics = {
            "total": sum(int(row["count"]) for row in status_rows),
            "active": 0,
            "inactive": 0,
            "stale_candidate_runs": 0,
            "quarantined_rows": 0,
            "frozen_candidate_lists": candidate_list_count,
        }
        for row in status_rows:
            status = str(row["registry_status"])
            if status == "active":
                metrics["active"] += int(row["count"])
            else:
                metrics["inactive"] += int(row["count"])
        metrics["stale_candidate_runs"] = sum(
            self._secure_load(CandidateRunRecord, row["payload"]).status == "stale"
            for row in run_rows
        )
        metrics["quarantined_rows"] = sum(
            self._secure_load(
                RiskObjectRegistryImportResult,
                row["payload"],
            ).quarantined_count
            for row in import_rows
        )
        return metrics

    def document_governance_metrics(self) -> dict[str, int]:
        with self._connect() as conn:
            document_versions = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM response_document_versions"
                ).fetchone()["count"]
            )
            parse_records = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM response_document_parses"
                ).fetchone()["count"]
            )
            contract_versions = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM response_contract_versions"
                ).fetchone()["count"]
            )
            build_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM response_index_builds GROUP BY status"
            ).fetchall()
        metrics = {
            "document_versions": document_versions,
            "parse_records": parse_records,
            "index_builds_completed": 0,
            "index_builds_failed": 0,
            "contract_versions": contract_versions,
        }
        for row in build_rows:
            key = f"index_builds_{row['status']}"
            if key in metrics:
                metrics[key] = int(row["count"])
        return metrics

    def evidence_governance_metrics(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM response_evidence_packages ORDER BY package_id, version"
            ).fetchall()
        versions = [
            self._secure_load(EvidencePackageVersion, row["payload"]) for row in rows
        ]
        latest: dict[str, EvidencePackageVersion] = {}
        for package in versions:
            latest[package.package_id] = package
        metrics = {
            "package_versions": len(versions),
            "packages": len(latest),
            "unresolved_conflicts": 0,
            "missing_fields": 0,
            "blocking_missing_fields": 0,
            "nli_unavailable": 0,
            "nli_partial": 0,
            "nli_error": 0,
        }
        for package in latest.values():
            metrics["unresolved_conflicts"] += sum(
                item.resolution_status != "resolved" for item in package.conflicts
            )
            metrics["missing_fields"] += len(package.missing_fields)
            metrics["blocking_missing_fields"] += len(
                package.blocking_missing_fields
            )
            key = f"nli_{package.nli_status}"
            if key in metrics:
                metrics[key] += 1
        return metrics

    def commit_risk_object_registry_import(
        self,
        *,
        result: RiskObjectRegistryImportResult,
        records: list[RiskObjectRegistryRecord],
        registry_snapshots: list[RiskObjectRegistryVersionSnapshot],
        stale_candidate_runs: list[CandidateRunRecord],
        stale_event_objects: list[EventRiskObject],
        event_object_snapshots: list[RiskObjectVersionSnapshot],
    ) -> None:
        with self._connect() as conn:
            for record in records:
                conn.execute(
                    "INSERT INTO response_risk_object_registry("
                    "area_id, object_id, registry_version, registry_status, updated_at, payload"
                    ") VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(area_id, object_id) DO UPDATE SET "
                    "registry_version=excluded.registry_version, "
                    "registry_status=excluded.registry_status, "
                    "updated_at=excluded.updated_at, payload=excluded.payload",
                    (
                        record.area_id,
                        record.object_id,
                        record.registry_version,
                        record.registry_status,
                        record.updated_at.isoformat(),
                        self._secure_dump(record),
                    ),
                )
            for snapshot in registry_snapshots:
                conn.execute(
                    "INSERT INTO response_risk_object_registry_versions("
                    "snapshot_id, area_id, object_id, registry_version, created_at, payload"
                    ") VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.snapshot_id,
                        snapshot.area_id,
                        snapshot.object_id,
                        snapshot.registry_version,
                        snapshot.created_at.isoformat(),
                        self._secure_dump(snapshot),
                    ),
                )
            for run in stale_candidate_runs:
                conn.execute(
                    "UPDATE response_candidate_runs SET payload = ? WHERE run_id = ?",
                    (self._secure_dump(run), run.run_id),
                )
            for item in stale_event_objects:
                conn.execute(
                    "UPDATE response_event_objects SET verification_status = ?, payload = ? "
                    "WHERE event_id = ? AND object_id = ?",
                    (
                        item.verification_status.value,
                        self._secure_dump(item),
                        item.event_id,
                        item.object_id,
                    ),
                )
            for snapshot in event_object_snapshots:
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
            conn.execute(
                "INSERT INTO response_risk_object_imports("
                "import_id, area_id, source_hash, source_format, created_at, payload"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    result.import_id,
                    result.area_id,
                    result.source_hash,
                    result.source_format,
                    result.created_at.isoformat(),
                    self._secure_dump(result),
                ),
            )
