from __future__ import annotations

import sqlite3

import pytest

from flood_system.repository import SQLiteRepository
from flood_system.storage.migrations import (
    REPOSITORY_MIGRATIONS,
    RepositoryMigration,
    apply_repository_migrations,
)


def test_repository_applies_ordered_checksum_locked_baseline(tmp_path):
    repository = SQLiteRepository(tmp_path / "migration-baseline.db")

    migrations = repository.list_schema_migrations()

    assert migrations[-1]["version"] == "0001_runtime_schema_baseline"
    assert migrations[-1]["checksum"] == REPOSITORY_MIGRATIONS[0].checksum


def test_repository_migrations_are_idempotent(tmp_path):
    repository = SQLiteRepository(tmp_path / "migration-idempotent.db")

    with repository._connect() as connection:
        assert apply_repository_migrations(connection) == []


def test_repository_rejects_changed_sql_for_applied_version(tmp_path):
    db_path = tmp_path / "migration-checksum.db"
    SQLiteRepository(db_path)
    changed = RepositoryMigration(
        version="0001_runtime_schema_baseline",
        sql="CREATE TABLE checksum_drift(id TEXT PRIMARY KEY);",
    )

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            apply_repository_migrations(connection, (changed,))
    finally:
        connection.close()
