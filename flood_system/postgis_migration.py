from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .response_workflow.models import (
    AlertSnapshot,
    ApprovalRecord,
    EventRiskObject,
    EvidencePackageVersion,
    ResponseEvent,
    ResponseTask,
    RiskObjectVersionSnapshot,
    TaskFeedback,
    TaskVersionSnapshot,
    TimelineEntry,
)
from .security import DataProtectionError, DataProtector


POSTGIS_SCHEMA = "flood_simulation"
DEFAULT_MAPPING_VERSION = "sqlite-response-to-postgis-shadow-v1"


@dataclass(frozen=True)
class ProjectionSpec:
    source_table: str
    record_type: str
    id_columns: tuple[str, ...]
    model_type: type[BaseModel]


PROJECTION_SPECS = (
    ProjectionSpec("response_events", "response_event", ("event_id",), ResponseEvent),
    ProjectionSpec("response_alert_snapshots", "alert_snapshot", ("snapshot_id",), AlertSnapshot),
    ProjectionSpec("response_event_objects", "risk_object", ("event_id", "object_id"), EventRiskObject),
    ProjectionSpec(
        "response_risk_object_versions", "risk_object_version", ("snapshot_id",), RiskObjectVersionSnapshot
    ),
    ProjectionSpec("response_tasks", "task", ("task_id",), ResponseTask),
    ProjectionSpec("response_task_versions", "task_version", ("snapshot_id",), TaskVersionSnapshot),
    ProjectionSpec("response_approvals", "approval", ("approval_id",), ApprovalRecord),
    ProjectionSpec("response_feedback", "feedback", ("feedback_id",), TaskFeedback),
    ProjectionSpec(
        "response_evidence_packages", "evidence_package", ("package_id", "version"), EvidencePackageVersion
    ),
    ProjectionSpec("response_timeline", "timeline", ("entry_id",), TimelineEntry),
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_payload(raw_payload: str) -> str:
    return json.dumps(json.loads(raw_payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_id(row: sqlite3.Row, columns: tuple[str, ...]) -> str:
    return "::".join(str(row[column]) for column in columns)


def _timestamp(row: sqlite3.Row) -> str | None:
    for column in ("created_at", "updated_at"):
        if column in row.keys() and row[column]:
            return str(row[column])
    return None


def _polygon_wkt(snapshot: AlertSnapshot) -> str | None:
    if snapshot.affected_geometry is None:
        return None
    points = ", ".join(f"{longitude:.12g} {latitude:.12g}" for longitude, latitude in snapshot.affected_geometry.coordinates)
    return f"POLYGON(({points}))"


def build_projection_records(source_db: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_db = source_db.expanduser().resolve()
    if not source_db.is_file():
        raise FileNotFoundError(f"SQLite source database does not exist: {source_db}")
    key_path = source_db.with_suffix(f"{source_db.suffix}.key")
    if not os.getenv("FLOOD_DATA_ENCRYPTION_KEY", "").strip() and not key_path.is_file():
        raise DataProtectionError(
            "source database key is required for migration; refusing to create a replacement key"
        )
    protector = DataProtector.for_database(source_db)
    connection = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    records: list[dict[str, Any]] = []
    quarantined: list[dict[str, str]] = []
    event_simulation: dict[str, bool] = {}
    try:
        connection.execute("BEGIN")
        for spec in PROJECTION_SPECS:
            rows = connection.execute(f"SELECT * FROM {spec.source_table}").fetchall()
            for row in rows:
                source_id = _source_id(row, spec.id_columns)
                ciphertext = str(row["payload"])
                ciphertext_hash = _sha256(ciphertext)
                try:
                    raw_payload = protector.decrypt_payload(ciphertext)
                    model = spec.model_type.model_validate_json(raw_payload)
                    canonical_hash = _sha256(_canonical_payload(raw_payload))
                    event_id = str(row["event_id"]) if "event_id" in row.keys() and row["event_id"] else None
                    explicit_simulated = getattr(model, "is_simulated", None)
                    if explicit_simulated is None and hasattr(model, "object"):
                        explicit_simulated = getattr(model.object, "is_simulated", None)
                    is_simulated = bool(
                        explicit_simulated
                        if explicit_simulated is not None
                        else event_simulation.get(str(event_id), False)
                    )
                    if spec.record_type == "response_event":
                        event_simulation[str(model.event_id)] = bool(model.is_simulated)
                        is_simulated = bool(model.is_simulated)
                    if not is_simulated:
                        raise ValueError("non-simulated record cannot enter flood_simulation schema")
                    records.append(
                        {
                            "record_type": spec.record_type,
                            "source_id": source_id,
                            "source_table": spec.source_table,
                            "event_id": event_id,
                            "object_id": str(row["object_id"])
                            if "object_id" in row.keys() and row["object_id"]
                            else None,
                            "task_id": str(row["task_id"])
                            if "task_id" in row.keys() and row["task_id"]
                            else None,
                            "version": int(row["version"])
                            if "version" in row.keys() and row["version"] is not None
                            else None,
                            "source_status": str(row["status"])
                            if "status" in row.keys() and row["status"]
                            else None,
                            "source_created_at": _timestamp(row),
                            "payload_ciphertext": ciphertext,
                            "payload_sha256": ciphertext_hash,
                            "canonical_payload_sha256": canonical_hash,
                            "is_simulated": True,
                            "affected_geometry_wkt": _polygon_wkt(model)
                            if isinstance(model, AlertSnapshot)
                            else None,
                        }
                    )
                except Exception as exc:
                    quarantined.append(
                        {
                            "source_table": spec.source_table,
                            "source_id": source_id,
                            "payload_sha256": ciphertext_hash,
                            "error_code": type(exc).__name__,
                            "error_detail": str(exc)[:1000],
                        }
                    )
    finally:
        connection.rollback()
        connection.close()
    return records, quarantined


def _group_digest(records: list[dict[str, Any]], record_type: str) -> str:
    values = sorted(
        f"{record['source_id']}:{record['canonical_payload_sha256']}"
        for record in records
        if record["record_type"] == record_type
    )
    return _sha256("\n".join(values))


def migrate_to_postgis(
    source_db: Path,
    dsn: str,
    *,
    ddl_path: Path,
    mapping_version: str = DEFAULT_MAPPING_VERSION,
) -> dict[str, Any]:
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover - exercised by CLI installation guidance
        raise RuntimeError('PostGIS migration requires: pip install ".[postgres]"') from exc

    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,80}", mapping_version):
        raise ValueError("mapping version contains unsupported characters")
    records, quarantined = build_projection_records(source_db)
    logical_rows = sorted(
        f"{record['record_type']}:{record['source_id']}:{record['canonical_payload_sha256']}" for record in records
    )
    source_logical_hash = _sha256("\n".join(logical_rows))
    batch_id = f"PGSIM-{mapping_version[:24]}-{source_logical_hash[:16]}"
    ddl = ddl_path.read_text(encoding="utf-8")
    schema_hash = _sha256(ddl)
    started_at = datetime.now(timezone.utc)
    source_counts = {
        spec.record_type: sum(record["record_type"] == spec.record_type for record in records)
        for spec in PROJECTION_SPECS
    }
    source_hashes = {spec.record_type: _group_digest(records, spec.record_type) for spec in PROJECTION_SPECS}
    target_counts: dict[str, int] = {}
    target_hashes: dict[str, str] = {}
    reconciliations: list[dict[str, Any]] = []
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(ddl)
            cursor.execute("SELECT PostGIS_Version()")
            postgis_version = str(cursor.fetchone()[0])
            cursor.execute(
                "INSERT INTO flood_simulation.schema_migrations(version, checksum) VALUES (%s, %s) "
                "ON CONFLICT(version) DO UPDATE SET checksum = EXCLUDED.checksum",
                (f"shadow-{schema_hash[:16]}", schema_hash),
            )
            cursor.execute(
                """INSERT INTO flood_simulation.migration_batches(
                    batch_id, mapping_version, source_instance_id, source_logical_sha256, schema_sha256,
                    status, source_counts, target_counts, table_hashes, quarantine_count, started_at
                ) VALUES (%s, %s, %s, %s, %s, 'running', %s, %s, %s, %s, %s)
                ON CONFLICT(batch_id) DO UPDATE SET
                    status='running', source_counts=EXCLUDED.source_counts, target_counts=EXCLUDED.target_counts,
                    table_hashes=EXCLUDED.table_hashes, quarantine_count=EXCLUDED.quarantine_count,
                    started_at=EXCLUDED.started_at, completed_at=NULL""",
                (
                    batch_id,
                    mapping_version,
                    source_db.expanduser().resolve().name,
                    source_logical_hash,
                    schema_hash,
                    Jsonb(source_counts),
                    Jsonb({}),
                    Jsonb(source_hashes),
                    len(quarantined),
                    started_at,
                ),
            )
            cursor.execute("DELETE FROM flood_simulation.migration_quarantine WHERE batch_id = %s", (batch_id,))
            cursor.execute("DELETE FROM flood_simulation.reconciliation_runs WHERE batch_id = %s", (batch_id,))
            for item in quarantined:
                cursor.execute(
                    """INSERT INTO flood_simulation.migration_quarantine(
                        batch_id, source_table, source_id, payload_sha256, error_code, error_detail
                    ) VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        batch_id,
                        item["source_table"],
                        item["source_id"],
                        item["payload_sha256"],
                        item["error_code"],
                        item["error_detail"],
                    ),
                )
            for record in records:
                cursor.execute(
                    """INSERT INTO flood_simulation.shadow_records(
                        record_type, source_id, source_table, event_id, object_id, task_id, version,
                        source_status, source_created_at, payload_ciphertext, payload_sha256,
                        canonical_payload_sha256, is_simulated, affected_geometry, migration_batch,
                        mapping_version
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE,
                        ST_GeomFromText(%s::text, 4326), %s, %s
                    ) ON CONFLICT(record_type, source_id) DO UPDATE SET
                        source_table=EXCLUDED.source_table, event_id=EXCLUDED.event_id,
                        object_id=EXCLUDED.object_id, task_id=EXCLUDED.task_id, version=EXCLUDED.version,
                        source_status=EXCLUDED.source_status, source_created_at=EXCLUDED.source_created_at,
                        payload_ciphertext=EXCLUDED.payload_ciphertext, payload_sha256=EXCLUDED.payload_sha256,
                        canonical_payload_sha256=EXCLUDED.canonical_payload_sha256,
                        affected_geometry=EXCLUDED.affected_geometry, migration_batch=EXCLUDED.migration_batch,
                        mapping_version=EXCLUDED.mapping_version, migrated_at=CURRENT_TIMESTAMP""",
                    (
                        record["record_type"],
                        record["source_id"],
                        record["source_table"],
                        record["event_id"],
                        record["object_id"],
                        record["task_id"],
                        record["version"],
                        record["source_status"],
                        record["source_created_at"],
                        record["payload_ciphertext"],
                        record["payload_sha256"],
                        record["canonical_payload_sha256"],
                        record["affected_geometry_wkt"],
                        batch_id,
                        mapping_version,
                    ),
                )
            for spec in PROJECTION_SPECS:
                cursor.execute(
                    """SELECT source_id, canonical_payload_sha256
                    FROM flood_simulation.shadow_records
                    WHERE migration_batch = %s AND record_type = %s
                    ORDER BY source_id""",
                    (batch_id, spec.record_type),
                )
                rows = cursor.fetchall()
                target_counts[spec.record_type] = len(rows)
                target_hashes[spec.record_type] = _sha256(
                    "\n".join(f"{row[0]}:{row[1]}" for row in rows)
                )
                passed = (
                    source_counts[spec.record_type] == target_counts[spec.record_type]
                    and source_hashes[spec.record_type] == target_hashes[spec.record_type]
                )
                reconciliation = {
                    "source_table": spec.source_table,
                    "record_type": spec.record_type,
                    "source_count": source_counts[spec.record_type],
                    "target_count": target_counts[spec.record_type],
                    "source_hash": source_hashes[spec.record_type],
                    "target_hash": target_hashes[spec.record_type],
                    "passed": passed,
                }
                reconciliations.append(reconciliation)
                cursor.execute(
                    """INSERT INTO flood_simulation.reconciliation_runs(
                        batch_id, source_table, record_type, source_count, target_count,
                        source_hash, target_hash, passed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        batch_id,
                        spec.source_table,
                        spec.record_type,
                        source_counts[spec.record_type],
                        target_counts[spec.record_type],
                        source_hashes[spec.record_type],
                        target_hashes[spec.record_type],
                        passed,
                    ),
                )
            status = "completed" if not quarantined and all(item["passed"] for item in reconciliations) else "failed"
            cursor.execute(
                """UPDATE flood_simulation.migration_batches
                SET status=%s, target_counts=%s, table_hashes=%s, completed_at=%s
                WHERE batch_id=%s""",
                (
                    status,
                    Jsonb(target_counts),
                    Jsonb(
                        {
                            record_type: {"source": source_hashes[record_type], "target": target_hashes[record_type]}
                            for record_type in source_hashes
                        }
                    ),
                    datetime.now(timezone.utc),
                    batch_id,
                ),
            )
    return {
        "batch_id": batch_id,
        "mapping_version": mapping_version,
        "target_schema": POSTGIS_SCHEMA,
        "postgis_version": postgis_version,
        "status": status,
        "source_logical_sha256": source_logical_hash,
        "schema_sha256": schema_hash,
        "source_counts": source_counts,
        "target_counts": target_counts,
        "quarantine_count": len(quarantined),
        "reconciliations": reconciliations,
    }
