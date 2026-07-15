from __future__ import annotations

import json
import shutil
import sqlite3

import pytest
from cryptography.fernet import Fernet

from flood_system.response_workflow.models import (
    BackupCreateRequest,
    BackupImportRequest,
    BackupRestoreRequest,
    BackupRetentionRequest,
    KeyRotationRequest,
    OperatorRole,
)
from flood_system.security import BackupManifestError, DataProtectionError, DataProtector
from flood_system.system import FloodWarningSystem


def admin_request_kwargs() -> dict[str, object]:
    return {
        "operator_id": "admin-dr",
        "operator_role": OperatorRole.ADMIN,
        "terminal_id": "dr-admin-terminal",
    }


def test_signed_backup_can_be_imported_and_restored_on_another_instance(tmp_path, monkeypatch):
    encryption_key = Fernet.generate_key().decode("ascii")
    manifest_secret = "pytest-cross-host-backup-manifest-secret-with-32-bytes"
    monkeypatch.setenv("FLOOD_DATA_ENCRYPTION_KEY", encryption_key)
    monkeypatch.setenv("FLOOD_BACKUP_MANIFEST_SECRET", manifest_secret)
    monkeypatch.setenv("FLOOD_INSTANCE_ID", "district-primary-a")
    source = FloodWarningSystem(tmp_path / "host-a" / "source.db")
    source_dashboard = source.response_workflow.bootstrap_demo()
    backup = source.response_workflow.create_database_backup(
        BackupCreateRequest(label="cross-host-dr", **admin_request_kwargs())
    )

    monkeypatch.setenv("FLOOD_INSTANCE_ID", "district-standby-b")
    standby = FloodWarningSystem(tmp_path / "host-b" / "standby.db")
    incoming = standby.repository.db_path.parent / "backups" / "incoming"
    incoming.mkdir(parents=True)
    source_backups = source.repository.db_path.parent / "backups"
    shutil.copy2(source_backups / backup.backup_filename, incoming / backup.backup_filename)
    shutil.copy2(source_backups / backup.manifest_filename, incoming / backup.manifest_filename)

    imported = standby.response_workflow.import_database_backup(
        BackupImportRequest(
            manifest_filename=backup.manifest_filename,
            backup_filename=backup.backup_filename,
            **admin_request_kwargs(),
        )
    )
    restored = standby.response_workflow.restore_database_backup(
        BackupRestoreRequest(backup_id=backup.backup_id, **admin_request_kwargs())
    )

    assert imported.status == "verified"
    assert imported.source_instance_id == "district-primary-a"
    assert imported.manifest_signature_verified is True
    assert imported.decryption_verified is True
    assert restored.status == "verified"
    restored_path = standby.repository.db_path.parent / "backups" / restored.restored_filename
    recovered = FloodWarningSystem(restored_path)
    recovered_events = recovered.response_workflow.list_events()
    assert any(item.event_id == source_dashboard.event.event_id for item in recovered_events)


def test_cross_host_import_rejects_tampered_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOOD_DATA_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("FLOOD_BACKUP_MANIFEST_SECRET", "pytest-tamper-backup-manifest-secret-with-32-bytes")
    source = FloodWarningSystem(tmp_path / "source" / "source.db")
    source.response_workflow.bootstrap_demo()
    backup = source.response_workflow.create_database_backup(
        BackupCreateRequest(label="tamper-test", **admin_request_kwargs())
    )
    standby = FloodWarningSystem(tmp_path / "standby" / "standby.db")
    incoming = standby.repository.db_path.parent / "backups" / "incoming"
    incoming.mkdir(parents=True)
    source_backups = source.repository.db_path.parent / "backups"
    shutil.copy2(source_backups / backup.backup_filename, incoming / backup.backup_filename)
    manifest = json.loads((source_backups / backup.manifest_filename).read_text(encoding="utf-8"))
    manifest["source_instance_id"] = "attacker-controlled-instance"
    (incoming / backup.manifest_filename).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BackupManifestError, match="signature verification failed"):
        standby.response_workflow.import_database_backup(
            BackupImportRequest(
                manifest_filename=backup.manifest_filename,
                backup_filename=backup.backup_filename,
                **admin_request_kwargs(),
            )
        )


def test_production_key_rotation_requires_activation_before_restart_and_keeps_old_backup_recoverable(tmp_path, monkeypatch):
    old_key = Fernet.generate_key().decode("ascii")
    new_key = Fernet.generate_key().decode("ascii")
    db_path = tmp_path / "rotation" / "production.db"
    monkeypatch.setenv("FLOOD_DATA_ENCRYPTION_KEY", old_key)
    monkeypatch.setenv("FLOOD_DATA_ENCRYPTION_KEY_NEXT", new_key)
    monkeypatch.setenv("FLOOD_BACKUP_MANIFEST_SECRET", "pytest-key-rotation-manifest-secret-with-32-bytes")
    monkeypatch.delenv("FLOOD_DATA_DECRYPTION_KEYS", raising=False)
    system = FloodWarningSystem(db_path)
    event_id = system.response_workflow.bootstrap_demo().event.event_id

    result = system.response_workflow.rotate_data_encryption_key(
        KeyRotationRequest(label="annual", **admin_request_kwargs())
    )

    assert result.status == "pending_activation"
    assert result.activation_required is True
    assert result.old_key_id == DataProtector.key_identifier(old_key.encode("ascii"))
    assert result.new_key_id == DataProtector.key_identifier(new_key.encode("ascii"))
    assert result.records_reencrypted > 0
    assert system.response_workflow.get_dashboard(event_id).event.event_id == event_id
    conn = sqlite3.connect(db_path)
    try:
        key_ids = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT json_extract(payload, '$.key_id') FROM response_events"
            ).fetchall()
        }
    finally:
        conn.close()
    assert key_ids == {result.new_key_id}

    with pytest.raises(DataProtectionError, match="pending activation"):
        FloodWarningSystem(db_path)

    monkeypatch.setenv("FLOOD_DATA_ENCRYPTION_KEY", new_key)
    monkeypatch.setenv("FLOOD_DATA_DECRYPTION_KEYS", old_key)
    monkeypatch.delenv("FLOOD_DATA_ENCRYPTION_KEY_NEXT", raising=False)
    restarted = FloodWarningSystem(db_path)
    assert restarted.response_workflow.get_dashboard(event_id).event.event_id == event_id
    restored = restarted.response_workflow.restore_database_backup(
        BackupRestoreRequest(backup_id=result.backup_id, **admin_request_kwargs())
    )
    assert restored.status == "verified"
    restarted.response_workflow.create_database_backup(
        BackupCreateRequest(label="routine-1", **admin_request_kwargs())
    )
    restarted.response_workflow.create_database_backup(
        BackupCreateRequest(label="routine-2", **admin_request_kwargs())
    )
    retention = restarted.response_workflow.apply_backup_retention(
        BackupRetentionRequest(keep_latest=1, max_age_days=None, dry_run=False, **admin_request_kwargs())
    )
    rotation_backup = restarted.response_workflow.repository.get_database_backup(result.backup_id)
    assert result.backup_id in retention.protected_backup_ids
    assert result.backup_id not in retention.pruned_backup_ids
    assert rotation_backup is not None
    assert (db_path.parent / "backups" / rotation_backup.backup_filename).is_file()


def test_local_key_rotation_atomically_promotes_sidecar_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOOD_DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("FLOOD_DATA_DECRYPTION_KEYS", raising=False)
    monkeypatch.setenv("FLOOD_BACKUP_MANIFEST_SECRET", "pytest-local-rotation-manifest-secret-with-32-bytes")
    next_key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("FLOOD_DATA_ENCRYPTION_KEY_NEXT", next_key)
    db_path = tmp_path / "local" / "local.db"
    system = FloodWarningSystem(db_path)
    old_key_id = system.repository.data_protector.key_id
    event_id = system.response_workflow.bootstrap_demo().event.event_id

    result = system.response_workflow.rotate_data_encryption_key(
        KeyRotationRequest(label="local", **admin_request_kwargs())
    )

    assert result.status == "activated"
    assert result.old_key_id == old_key_id
    assert db_path.with_suffix(".db.key").read_text(encoding="ascii").strip() == next_key
    restarted = FloodWarningSystem(db_path)
    assert restarted.repository.data_protector.key_id == result.new_key_id
    assert restarted.response_workflow.get_dashboard(event_id).event.event_id == event_id
