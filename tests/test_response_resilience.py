from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone

import pytest

from flood_system.response_workflow.models import (
    AlertSnapshot,
    AuditArchiveRequest,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    BackupCreateRequest,
    BackupRestoreRequest,
    BackupRetentionRequest,
    OperatorRole,
    TaskActionRequest,
    TaskAssignmentRequest,
    TaskStatus,
    TaskUpdateRequest,
)
from flood_system.system import FloodWarningSystem
from flood_system.security import DataProtectionError
from flood_system.storage.schema import REPOSITORY_SCHEMA_SQL

from tests.support.response_workflow import (
    seed_workflow,
    assign_field_operator,
)


def test_liaison_assignment_and_reassignment_control_the_field_executor(tmp_path):
    _, workflow, event_id, task = seed_workflow(
        tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED
    )
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
    )
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    workflow.acknowledge_task(
        task.task_id,
        TaskActionRequest(operator_id="liaison-1", operator_role=OperatorRole.LIAISON),
    )

    with pytest.raises(ValueError, match="must be assigned"):
        workflow.start_task(
            task.task_id,
            TaskActionRequest(
                operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR
            ),
        )
    with pytest.raises(PermissionError):
        workflow.assign_task(
            task.task_id,
            TaskAssignmentRequest(
                operator_id="duty-1",
                operator_role=OperatorRole.DUTY_OFFICER,
                assignee_id="field-1",
                assignee_name="执行员一",
            ),
        )

    first = assign_field_operator(workflow, task.task_id, assignee_id="field-1")
    reassigned = workflow.assign_task(
        task.task_id,
        TaskAssignmentRequest(
            operator_id="liaison-1",
            operator_role=OperatorRole.LIAISON,
            assignee_id="field-2",
            assignee_name="执行员二",
            reason="原执行员设备故障，重新分派",
        ),
    )

    assert first.assignment_version == 1
    assert reassigned.assignment_version == 2
    assert reassigned.assignee_id == "field-2"
    with pytest.raises(PermissionError, match="assigned field operator"):
        workflow.start_task(
            task.task_id,
            TaskActionRequest(
                operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR
            ),
        )
    started = workflow.start_task(
        task.task_id,
        TaskActionRequest(
            operator_id="field-2", operator_role=OperatorRole.FIELD_OPERATOR
        ),
    )
    assert started.status == TaskStatus.IN_PROGRESS
    assignments = workflow.repository.list_task_assignments(task.task_id)
    assert [item.assignee_id for item in assignments] == ["field-1", "field-2"]
    assert assignments[1].previous_assignee_id == "field-1"
    assert len(workflow.get_dashboard(event_id).assignments) == 2
    assert {
        item.action for item in workflow.repository.list_timeline_entries(event_id)
    } >= {"task_assigned", "task_reassigned", "task_started"}


def test_task_versions_keep_complete_immutable_snapshots_and_approval_history(tmp_path):
    _, workflow, _, task = seed_workflow(
        tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED
    )
    original_title = task.title
    revised = workflow.update_task(
        task.task_id,
        TaskUpdateRequest(
            title="修订后的现场核查任务",
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="district-console",
            note="补充现场核查范围",
        ),
    )
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
    )
    rejected = workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.REJECTED,
            note="需要进一步明确交通管控边界",
        ),
    )

    versions = workflow.list_task_versions(task.task_id)
    assert [item.version for item in versions] == [1, 2, 3]
    assert versions[0].task.title == original_title
    assert versions[1].task.title == revised.title
    assert versions[1].change_type == "revised"
    assert versions[2].change_type == "approval_rejected"
    assert versions[2].task.status == TaskStatus.DRAFT
    assert rejected.version == 3
    with pytest.raises(sqlite3.IntegrityError, match="task versions cannot be deleted"):
        with workflow.repository._connect() as conn:
            conn.execute(
                "DELETE FROM response_task_versions WHERE task_id = ?", (task.task_id,)
            )


def test_database_backup_and_isolated_restore_drill_are_hash_verified(tmp_path):
    _, workflow, _, _ = seed_workflow(tmp_path)

    backup = workflow.create_database_backup(
        BackupCreateRequest(
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="secure-admin-terminal",
            label="pre-storm-drill",
        )
    )

    backup_dir = workflow.repository.db_path.parent / "backups"
    assert backup.integrity_check == "ok"
    assert len(backup.sha256) == 64
    assert (backup_dir / backup.backup_filename).is_file()
    assert (backup_dir / backup.manifest_filename).is_file()
    assert (
        workflow.list_database_backups(OperatorRole.AUDITOR)[0].backup_id
        == backup.backup_id
    )

    restored = workflow.restore_database_backup(
        BackupRestoreRequest(
            backup_id=backup.backup_id,
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="secure-admin-terminal",
        )
    )

    assert restored.status == "verified"
    assert restored.sha256_verified is True
    assert restored.integrity_check == "ok"
    assert (backup_dir / restored.restored_filename).is_file()
    with pytest.raises(PermissionError):
        workflow.create_database_backup(
            BackupCreateRequest(
                operator_id="reviewer-1",
                operator_role=OperatorRole.REVIEWER,
                label="unauthorized",
            )
        )


def test_backup_retention_supports_dry_run_and_preserves_latest_recovery_point(
    tmp_path,
):
    system, workflow, _, _ = seed_workflow(tmp_path)
    created = [
        workflow.create_database_backup(
            BackupCreateRequest(
                label=f"retention-{index}",
                operator_id="admin-1",
                operator_role=OperatorRole.ADMIN,
                terminal_id="admin-terminal",
            )
        )
        for index in range(3)
    ]
    backup_dir = workflow.repository.db_path.parent / "backups"

    preview = workflow.apply_backup_retention(
        BackupRetentionRequest(
            keep_latest=1,
            max_age_days=None,
            dry_run=True,
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
    )
    assert len(preview.candidate_backup_ids) == 2
    assert preview.pruned_backup_ids == []
    assert all((backup_dir / item.backup_filename).is_file() for item in created)

    applied = workflow.apply_backup_retention(
        BackupRetentionRequest(
            keep_latest=1,
            max_age_days=None,
            dry_run=False,
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
    )
    latest = workflow.list_database_backups(OperatorRole.ADMIN)[0]
    assert set(applied.pruned_backup_ids) == set(preview.candidate_backup_ids)
    assert (backup_dir / latest.backup_filename).is_file()
    assert len(workflow.list_database_backups(OperatorRole.AUDITOR)) == 3
    assert (
        workflow.restore_database_backup(
            BackupRestoreRequest(
                backup_id=latest.backup_id,
                operator_id="admin-1",
                operator_role=OperatorRole.ADMIN,
                terminal_id="admin-terminal",
            )
        ).status
        == "verified"
    )
    with pytest.raises(
        sqlite3.IntegrityError, match="retention runs cannot be deleted"
    ):
        with workflow.repository._connect() as conn:
            conn.execute(
                "DELETE FROM response_backup_retention_runs WHERE run_id = ?",
                (applied.run_id,),
            )
    repeated = workflow.apply_backup_retention(
        BackupRetentionRequest(
            keep_latest=1,
            max_age_days=None,
            dry_run=False,
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
    )
    assert repeated.candidate_backup_ids == []
    assert set(repeated.already_pruned_backup_ids) == set(applied.pruned_backup_ids)
    assert repeated.pruned_backup_ids == []


def test_encrypted_signed_audit_archive_is_verifiable_and_detects_tampering(tmp_path):
    system, workflow, event_id, _ = seed_workflow(tmp_path)
    archive = workflow.create_audit_archive(
        event_id,
        AuditArchiveRequest(
            label="seven-year-audit",
            retention_days=2555,
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        ),
    )
    archive_dir = workflow.repository.db_path.parent / "backups" / "audit-archives"
    archive_path = archive_dir / archive.archive_filename
    stored = archive_path.read_text(encoding="utf-8")

    assert '"protected":true' in stored
    assert "区县防办测试事件" not in stored
    assert (archive_dir / archive.manifest_filename).is_file()
    assert (
        workflow.list_audit_archives(event_id, OperatorRole.AUDITOR)[0].archive_id
        == archive.archive_id
    )
    verified = workflow.verify_audit_archive(
        archive.archive_id,
        TaskActionRequest(
            operator_id="auditor-1",
            operator_role=OperatorRole.AUDITOR,
            terminal_id="audit-terminal",
        ),
    )
    assert verified.status == "verified"
    assert verified.signature_verified and verified.sha256_verified
    assert verified.decryption_verified and verified.timeline_head_verified
    with pytest.raises(
        sqlite3.IntegrityError, match="audit archives cannot be deleted"
    ):
        with workflow.repository._connect() as conn:
            conn.execute(
                "DELETE FROM response_audit_archives WHERE archive_id = ?",
                (archive.archive_id,),
            )

    archive_path.write_text(stored + "tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        workflow.verify_audit_archive(
            archive.archive_id,
            TaskActionRequest(
                operator_id="admin-1",
                operator_role=OperatorRole.ADMIN,
                terminal_id="admin-terminal",
            ),
        )


def test_response_payloads_are_encrypted_at_rest_and_tampering_is_rejected(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path)

    with workflow.repository._connect() as conn:
        object_payload = conn.execute(
            "SELECT payload FROM response_event_objects LIMIT 1"
        ).fetchone()["payload"]
        task_payload = conn.execute(
            "SELECT payload FROM response_tasks WHERE task_id = ?", (task.task_id,)
        ).fetchone()["payload"]

    object_envelope = json.loads(object_payload)
    task_envelope = json.loads(task_payload)
    assert object_envelope["protected"] is True
    assert object_envelope["algorithm"] == "fernet-aes128-cbc-hmac-sha256"
    assert "测试路口" not in object_payload
    assert "现场核查" not in task_payload

    ciphertext = task_envelope["ciphertext"]
    task_envelope["ciphertext"] = f"{ciphertext[:-2]}AA"
    with workflow.repository._connect() as conn:
        conn.execute(
            "UPDATE response_tasks SET payload = ? WHERE task_id = ?",
            (json.dumps(task_envelope), task.task_id),
        )

    with pytest.raises(DataProtectionError, match="authentication failed"):
        workflow.repository.get_response_task(task.task_id)


def test_legacy_plaintext_immutable_payload_is_encrypted_on_repository_startup(
    tmp_path,
):
    db_path = tmp_path / "legacy.db"
    snapshot = AlertSnapshot(
        snapshot_id="ALT-LEGACY",
        event_id="FLOOD-LEGACY",
        alert_id="ALERT-LEGACY",
        source_department="气象部门",
        disaster_type="暴雨",
        level="橙色",
        issued_at=datetime.now(timezone.utc),
        affected_area="碑林区",
        raw_content="旧版明文预警",
        version=1,
        created_at=datetime.now(timezone.utc),
    )
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(REPOSITORY_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO response_alert_snapshots(snapshot_id, event_id, version, created_at, payload) VALUES (?, ?, ?, ?, ?)",
            (
                snapshot.snapshot_id,
                snapshot.event_id,
                snapshot.version,
                snapshot.created_at.isoformat(),
                snapshot.model_dump_json(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    system = FloodWarningSystem(db_path)

    with system.repository._connect() as migrated:
        payload = migrated.execute(
            "SELECT payload FROM response_alert_snapshots WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        ).fetchone()["payload"]
    assert json.loads(payload)["protected"] is True
    assert "旧版明文预警" not in payload
    assert (
        system.repository.list_alert_snapshots(snapshot.event_id)[0].raw_content
        == "旧版明文预警"
    )


def test_high_risk_task_cannot_be_self_approved_or_approved_by_reviewer(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path)
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
    )
    with pytest.raises(PermissionError, match="commander"):
        workflow.decide_task(
            task.task_id,
            ApprovalRequest(
                operator_id="reviewer-1",
                operator_role=OperatorRole.REVIEWER,
                decision=ApprovalDecision.APPROVED,
            ),
        )
    with pytest.raises(PermissionError, match="drafter"):
        workflow.decide_task(
            task.task_id,
            ApprovalRequest(
                operator_id="duty-1",
                operator_role=OperatorRole.COMMANDER,
                decision=ApprovalDecision.APPROVED,
            ),
        )
