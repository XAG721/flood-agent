from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3

from .schema import REPOSITORY_SCHEMA_SQL


MIGRATION_LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS response_schema_migrations (
    version TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class RepositoryMigration:
    version: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


REPOSITORY_MIGRATIONS = (
    RepositoryMigration("0001_runtime_schema_baseline", REPOSITORY_SCHEMA_SQL),
)


def apply_repository_migrations(
    connection: sqlite3.Connection,
    migrations: tuple[RepositoryMigration, ...] = REPOSITORY_MIGRATIONS,
) -> list[str]:
    """Apply ordered, checksum-locked migrations and return newly applied versions."""

    connection.executescript(MIGRATION_LEDGER_SQL)
    applied: list[str] = []
    for migration in migrations:
        existing = connection.execute(
            "SELECT checksum FROM response_schema_migrations WHERE version = ?",
            (migration.version,),
        ).fetchone()
        if existing is not None:
            if str(existing["checksum"]) != migration.checksum:
                raise RuntimeError(
                    f"repository migration checksum mismatch: {migration.version}"
                )
            continue
        connection.executescript(migration.sql)
        connection.execute(
            "INSERT INTO response_schema_migrations(version, checksum, applied_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (migration.version, migration.checksum),
        )
        applied.append(migration.version)
    return applied
