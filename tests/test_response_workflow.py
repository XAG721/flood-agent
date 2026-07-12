from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timedelta, timezone

import pytest

from flood_system.response_workflow.models import (
    AlertInput,
    AlertAppendRequest,
    AlertSnapshot,
    AuditArchiveRequest,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    BackupCreateRequest,
    BackupRestoreRequest,
    BackupRetentionRequest,
    CandidateDiscoveryRequest,
    DeadlineExtensionDecisionRequest,
    DeadlineExtensionRequest,
    DeadlineExtensionStatus,
    DocumentImportRequest,
    EventCloseRequest,
    EventCreateRequest,
    EvidenceConflictResolutionRequest,
    EvidenceFreezeRequest,
    EvidenceManualSupplementRequest,
    EvidenceRole,
    FeedbackRequest,
    FeatureFlagUpdateRequest,
    GeoPolygon,
    LegacyMigrationRequest,
    ObjectVerificationStatus,
    OperatorRole,
    OutboxProcessRequest,
    OutboxStatus,
    PlanBasis,
    RiskObjectBatchRequest,
    RiskObjectInput,
    RiskObjectVerificationRequest,
    RetrievalMode,
    RuleOutcome,
    ScenarioEvaluationRequest,
    TaskActionRequest,
    TaskAssignmentRequest,
    TaskCreateRequest,
    TaskDraftGenerationRequest,
    TaskEvidenceRef,
    TaskStatus,
    TaskUpdateRequest,
)
from flood_system.system import FloodWarningSystem
from flood_system.models import CorpusType, RAGDocument, RiskLevel, Stage
from flood_system.v2.models import EventRecord, EventStatus as V2EventStatus
from flood_system.security import DataProtectionError
from flood_system.storage.schema import REPOSITORY_SCHEMA_SQL
from flood_system.response_workflow.state_machine import (
    ALLOWED_TASK_TRANSITIONS,
    TERMINAL_TASK_STATUSES,
    can_transition,
)


def seed_workflow(tmp_path, *, approval_policy=ApprovalPolicy.COMMANDER_REQUIRED):
    system = FloodWarningSystem(tmp_path / "workflow.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="区县防办测试事件",
            area_id="district-demo",
            alert=AlertInput(
                alert_id="ALERT-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                valid_until=now + timedelta(hours=3),
                affected_area="测试片区",
                raw_content="测试预警原文",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )
    event_id = dashboard.event.event_id
    workflow.add_risk_objects(
        event_id,
        RiskObjectBatchRequest(
            objects=[
                RiskObjectInput(
                    object_id="TUNNEL-001",
                    name="测试下穿通道",
                    object_type="下穿通道",
                    location="测试路口",
                    responsible_organization="住建局",
                    responsible_role="排水值班负责人",
                    trigger_reasons=["预警覆盖"],
                    source_refs=["ALERT-001"],
                    vulnerability="低洼",
                    risk_score=90,
                    system_explanation="候选对象，待人工核验",
                )
            ],
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )
    workflow.verify_risk_object(
        event_id,
        "TUNNEL-001",
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="台账核验通过",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
        ),
    )
    task = workflow.create_task(
        event_id,
        TaskCreateRequest(
            object_id="TUNNEL-001",
            title="现场值守和管控准备",
            action="到场核查并准备交通管控",
            responsible_organization="住建局",
            responsible_role="排水值班负责人",
            cooperate_roles=["交警联络员"],
            deadline_at=now + timedelta(hours=1),
            acknowledge_deadline_at=now + timedelta(minutes=10),
            required_evidence=["现场照片", "积水深度"],
            plan_basis=[PlanBasis(document="区防汛预案", version="2026", clause="4.2")],
            approval_policy=approval_policy,
            escalation_rule="超时上报防办",
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            generated_by_ai=False,
        ),
    )
    return system, workflow, event_id, task


def test_area_registry_candidate_discovery_is_explainable_and_requires_manual_verification(tmp_path):
    system = FloodWarningSystem(tmp_path / "candidate-discovery.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="碑林区候选风险对象筛查",
            area_id="beilin_10km2",
            alert=AlertInput(
                alert_id="ALERT-DISCOVERY-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                valid_until=now + timedelta(hours=3),
                affected_area="碑林区重点片区",
                raw_content="橙色暴雨预警",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    result = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            entity_types=["school", "hospital", "underground_space"],
            min_risk_score=45,
            max_candidates=10,
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="duty-terminal",
        ),
    )

    assert result.association_mode == "area_registry"
    assert result.scanned_profiles == 3
    assert result.matched_profiles == 3
    assert result.limitations and "GIS" in result.limitations[0]
    assert all(item.verification_status == ObjectVerificationStatus.PENDING for item in result.candidates)
    assert all(result.alert_snapshot_id in item.source_refs for item in result.candidates)
    assert all(f"entity_profile:{item.object_id}" in item.source_refs for item in result.candidates)
    assert all("不替代水动力风险分析" in item.system_explanation for item in result.candidates)
    assert all(item.raw_candidate_score == item.risk_score for item in result.candidates)
    assert all(item.calibrated_confidence is not None for item in result.candidates)
    assert all(item.calibration_version == "candidate-logistic-v1" for item in result.candidates)

    verified = workflow.verify_risk_object(
        dashboard.event.event_id,
        result.candidates[0].object_id,
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="台账与属地电话复核通过",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
        ),
    )
    assert verified.verification_status == ObjectVerificationStatus.CONFIRMED


def test_candidate_discovery_returns_empty_result_when_area_has_no_registered_profiles(tmp_path):
    system = FloodWarningSystem(tmp_path / "candidate-discovery-empty.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="无台账片区筛查",
            area_id="area-without-profile",
            alert=AlertInput(
                alert_id="ALERT-DISCOVERY-EMPTY",
                source_department="气象部门",
                disaster_type="暴雨",
                level="黄色",
                issued_at=now,
                affected_area="无台账片区",
                raw_content="黄色暴雨预警",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    result = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )

    assert result.scanned_profiles == 0
    assert result.matched_profiles == 0
    assert result.candidates == []


def test_candidate_discovery_uses_epsg4326_point_in_polygon_when_alert_geometry_is_available(tmp_path):
    system = FloodWarningSystem(tmp_path / "candidate-discovery-gis.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="学校周边精确空间筛查",
            area_id="beilin_10km2",
            alert=AlertInput(
                alert_id="ALERT-GIS-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                affected_area="文艺路小学周边",
                affected_geometry=GeoPolygon(
                    coordinates=[
                        (108.9565, 34.2425),
                        (108.9595, 34.2425),
                        (108.9595, 34.2455),
                        (108.9565, 34.2455),
                        (108.9565, 34.2425),
                    ]
                ),
                raw_content="学校周边短时强降雨预警范围",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    result = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            entity_types=["school", "hospital"],
            min_risk_score=45,
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )

    assert result.association_mode == "gis_point_in_polygon"
    assert result.scanned_profiles == 2
    assert result.spatially_evaluated_profiles == 2
    assert result.spatially_matched_profiles == 1
    assert result.excluded_unlocated_profiles == 0
    assert [item.object_id for item in result.candidates] == ["school_wyl_primary"]
    assert any(ref.startswith("alert_geometry:") for ref in result.candidates[0].source_refs)
    assert result.candidates[0].trigger_reasons[0] == "对象登记点落入橙色暴雨预警多边形"
    assert "108." not in result.candidates[0].trigger_reasons[0]
    assert result.limitations == []


def assign_field_operator(workflow, task_id: str, *, assignee_id: str = "field-1"):
    return workflow.assign_task(
        task_id,
        TaskAssignmentRequest(
            operator_id="liaison-1",
            operator_role=OperatorRole.LIAISON,
            assignee_id=assignee_id,
            assignee_name="现场执行员",
            assignee_role=OperatorRole.FIELD_OPERATOR,
            reason="按当前班次分派",
        ),
    )


def test_feature_flags_are_scoped_versioned_and_admin_only(tmp_path):
    _, workflow, event_id, _ = seed_workflow(tmp_path)
    with pytest.raises(PermissionError):
        workflow.update_feature_flag(
            FeatureFlagUpdateRequest(
                flag_key="feature.simulated_dispatch",
                enabled=False,
                environment="development",
                event_id=event_id,
                reason="非管理员不得更改开关",
                operator_id="duty-1",
                operator_role=OperatorRole.DUTY_OFFICER,
            )
        )

    first = workflow.update_feature_flag(
        FeatureFlagUpdateRequest(
            flag_key="feature.simulated_dispatch",
            enabled=False,
            environment="development",
            event_id=event_id,
            reason="演练模拟下发关闭路径",
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
    )
    second = workflow.update_feature_flag(
        FeatureFlagUpdateRequest(
            flag_key="feature.simulated_dispatch",
            enabled=True,
            environment="development",
            event_id=event_id,
            reason="演练后恢复模拟下发",
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
    )

    assert first.version == 1
    assert second.version == 2
    assert workflow.is_feature_enabled("feature.simulated_dispatch", event_id=event_id) is True
    assert any(item.action == "feature_flag_changed" for item in workflow.get_dashboard(event_id).timeline)


def test_approved_dispatch_uses_idempotent_outbox_and_payload_hash(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path)
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER),
    )
    issued = workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            decision=ApprovalDecision.APPROVED,
            note="同意模拟下发",
        ),
    )

    messages = workflow.list_outbox_messages(event_id=issued.event_id)
    assert len(messages) == 1
    assert messages[0].status == OutboxStatus.SENT
    assert messages[0].payload_hash == issued.approval_payload_hash
    rerun = workflow.process_outbox(
        OutboxProcessRequest(
            message_id=messages[0].message_id,
            operator_id="liaison-1",
            operator_role=OperatorRole.LIAISON,
        )
    )
    assert rerun[0].status == OutboxStatus.SENT
    assert rerun[0].attempts == 1


def test_outbox_fails_closed_when_approved_payload_is_tampered(tmp_path, monkeypatch):
    system, workflow, _, task = seed_workflow(tmp_path)
    monkeypatch.setenv("FLOOD_ENVIRONMENT", "production")
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER),
    )
    issued = workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            decision=ApprovalDecision.APPROVED,
            note="批准后等待生产下发",
        ),
    )
    message = workflow.list_outbox_messages(event_id=issued.event_id)[0]
    assert message.status == OutboxStatus.PENDING

    system.repository.save_response_task(issued.model_copy(update={"action": "未经重新审批的篡改动作"}))
    result = workflow.process_outbox(
        OutboxProcessRequest(
            message_id=message.message_id,
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
        )
    )
    assert result[0].status == OutboxStatus.FAILED
    assert result[0].last_error == "approved payload hash mismatch"


def test_full_deterministic_response_loop_and_close(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    issued = workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            decision=ApprovalDecision.APPROVED,
            note="同意执行",
        ),
    )
    assert issued.status == TaskStatus.ISSUED
    workflow.acknowledge_task(
        task.task_id, TaskActionRequest(operator_id="liaison-1", operator_role=OperatorRole.LIAISON)
    )
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(
        task.task_id, TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR)
    )
    pending = workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场处置完成",
            evidence=[{"type": "现场照片", "url": "object://photo-1"}, {"type": "积水深度", "value": "8cm"}],
        ),
    )
    assert pending.status == TaskStatus.PENDING_VERIFICATION
    completed = workflow.verify_completion(
        task.task_id,
        True,
        TaskActionRequest(operator_id="reviewer-1", operator_role=OperatorRole.REVIEWER, note="证据完整"),
    )
    assert completed.status == TaskStatus.COMPLETED
    closed = workflow.close_event(
        event_id,
        EventCloseRequest(operator_id="commander-1", operator_role=OperatorRole.COMMANDER, note="闭环完成"),
    )
    assert closed.event.status.value == "closed"
    assert closed.metrics["completion_rate"] == 1.0
    assert closed.review_draft is not None
    assert closed.review_draft.process_metrics["completed_tasks"] == 1
    assert closed.review_draft.evidence_findings
    assert closed.review_draft.improvement_recommendations
    assert {item.action for item in closed.timeline} >= {
        "event_created",
        "task_approved_and_issued",
        "completion_verified",
        "event_closed",
    }


def test_closed_event_generates_persisted_district_scenario_report(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    workflow.acknowledge_task(task.task_id, TaskActionRequest(operator_id="liaison-1", operator_role=OperatorRole.LIAISON))
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(task.task_id, TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR))
    workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场处置完成",
            evidence=[{"type": "现场照片", "url": "object://photo-1"}, {"type": "积水深度", "value": "8cm"}],
        ),
    )
    workflow.verify_completion(
        task.task_id,
        True,
        TaskActionRequest(operator_id="reviewer-1", operator_role=OperatorRole.REVIEWER),
    )
    workflow.close_event(
        event_id,
        EventCloseRequest(operator_id="commander-1", operator_role=OperatorRole.COMMANDER),
    )

    report = workflow.run_scenario_evaluation(
        event_id,
        ScenarioEvaluationRequest(
            operator_id="auditor-1",
            operator_role=OperatorRole.AUDITOR,
            expected_object_ids=["TUNNEL-001"],
        ),
    )

    assert report.overall_status in {"passed", "passed_with_observations"}
    assert len(report.acceptance_checks) == 10
    assert all(check.passed for check in report.acceptance_checks)
    assert len(report.metrics) == 16
    assert next(item for item in report.metrics if item.metric_id == "object_omission_rate").system_value == 0.0
    assert next(item for item in report.metrics if item.metric_id == "approval_duration").manual_value == 20.0
    assert workflow.list_scenario_reports(event_id)[0].report_id == report.report_id
    assert workflow.get_dashboard(event_id).scenario_report.report_id == report.report_id


def test_incomplete_scenario_report_exposes_blocking_acceptance_failures(tmp_path):
    _, workflow, event_id, _ = seed_workflow(tmp_path)

    report = workflow.run_scenario_evaluation(
        event_id,
        ScenarioEvaluationRequest(operator_id="auditor-1", operator_role=OperatorRole.AUDITOR),
    )

    assert report.overall_status == "failed"
    assert any(not check.passed for check in report.acceptance_checks)
    assert any(item.severity == "blocking" for item in report.failure_cases)

    rerun = workflow.run_scenario_evaluation(
        event_id,
        ScenarioEvaluationRequest(operator_id="auditor-1", operator_role=OperatorRole.AUDITOR),
    )
    assert rerun.overall_status == "failed"
    assert "auditor" not in rerun.scope["operator_roles"]


def test_timeline_hash_chain_is_verified_and_immutable_records_cannot_be_deleted(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="district-console-07",
        ),
    )

    report = workflow.verify_timeline_integrity(event_id, OperatorRole.AUDITOR)
    timeline = workflow.repository.list_timeline_entries(event_id)

    assert report.status == "verified"
    assert report.verified_entries == report.total_entries == len(timeline)
    assert timeline[0].previous_hash == "GENESIS"
    assert all(item.record_hash for item in timeline)
    assert all(timeline[index].previous_hash == timeline[index - 1].record_hash for index in range(1, len(timeline)))
    submitted = next(item for item in timeline if item.action == "task_submitted")
    assert submitted.terminal_id == "district-console-07"
    assert submitted.before_state == {"status": "draft"}
    assert submitted.after_state["status"] == "pending_approval"

    with pytest.raises(sqlite3.IntegrityError, match="timeline cannot be deleted"):
        with workflow.repository._connect() as conn:
            conn.execute("DELETE FROM response_timeline WHERE event_id = ?", (event_id,))
    with pytest.raises(sqlite3.IntegrityError, match="alert snapshots cannot be deleted"):
        with workflow.repository._connect() as conn:
            conn.execute("DELETE FROM response_alert_snapshots WHERE event_id = ?", (event_id,))


def test_liaison_assignment_and_reassignment_control_the_field_executor(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
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
            TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR),
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
            TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR),
        )
    started = workflow.start_task(
        task.task_id,
        TaskActionRequest(operator_id="field-2", operator_role=OperatorRole.FIELD_OPERATOR),
    )
    assert started.status == TaskStatus.IN_PROGRESS
    assignments = workflow.repository.list_task_assignments(task.task_id)
    assert [item.assignee_id for item in assignments] == ["field-1", "field-2"]
    assert assignments[1].previous_assignee_id == "field-1"
    assert len(workflow.get_dashboard(event_id).assignments) == 2
    assert {item.action for item in workflow.repository.list_timeline_entries(event_id)} >= {
        "task_assigned", "task_reassigned", "task_started"
    }


def test_task_versions_keep_complete_immutable_snapshots_and_approval_history(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
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
        TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER),
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
            conn.execute("DELETE FROM response_task_versions WHERE task_id = ?", (task.task_id,))


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
    assert workflow.list_database_backups(OperatorRole.AUDITOR)[0].backup_id == backup.backup_id

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


def test_backup_retention_supports_dry_run_and_preserves_latest_recovery_point(tmp_path):
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
    assert workflow.restore_database_backup(
        BackupRestoreRequest(
            backup_id=latest.backup_id,
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
    ).status == "verified"
    with pytest.raises(sqlite3.IntegrityError, match="retention runs cannot be deleted"):
        with workflow.repository._connect() as conn:
            conn.execute("DELETE FROM response_backup_retention_runs WHERE run_id = ?", (applied.run_id,))
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
    assert workflow.list_audit_archives(event_id, OperatorRole.AUDITOR)[0].archive_id == archive.archive_id
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
    with pytest.raises(sqlite3.IntegrityError, match="audit archives cannot be deleted"):
        with workflow.repository._connect() as conn:
            conn.execute("DELETE FROM response_audit_archives WHERE archive_id = ?", (archive.archive_id,))

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
        object_payload = conn.execute("SELECT payload FROM response_event_objects LIMIT 1").fetchone()["payload"]
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


def test_legacy_plaintext_immutable_payload_is_encrypted_on_repository_startup(tmp_path):
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
            (snapshot.snapshot_id, snapshot.event_id, snapshot.version, snapshot.created_at.isoformat(), snapshot.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()

    system = FloodWarningSystem(db_path)

    with system.repository._connect() as migrated:
        payload = migrated.execute(
            "SELECT payload FROM response_alert_snapshots WHERE snapshot_id = ?", (snapshot.snapshot_id,)
        ).fetchone()["payload"]
    assert json.loads(payload)["protected"] is True
    assert "旧版明文预警" not in payload
    assert system.repository.list_alert_snapshots(snapshot.event_id)[0].raw_content == "旧版明文预警"


def test_high_risk_task_cannot_be_self_approved_or_approved_by_reviewer(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
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


def test_overdue_unacknowledged_task_is_escalated(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    escalated = workflow.run_deadline_sweep(
        event_id,
        TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER),
        now=task.deadline_at + timedelta(minutes=1),
    )
    assert len(escalated) == 1
    assert escalated[0].status == TaskStatus.ESCALATED
    assert workflow.repository.list_escalations(event_id)[0].reason == "任务超过确认时限仍未确认"


def test_grounded_task_draft_covers_all_roles_and_keeps_plan_attribution(tmp_path):
    system = FloodWarningSystem(tmp_path / "grounded-draft.db")
    dashboard = system.response_workflow.bootstrap_demo()

    assert len(dashboard.tasks) == 1
    task = dashboard.tasks[0]
    assert task.generated_by_ai is True
    assert task.generation_version == "response-rag-task-v1"
    assert task.evidence_role_coverage == {
        "condition": True,
        "object": True,
        "responsibility": True,
        "procedure": True,
        "exception": True,
        "attribution": True,
    }
    assert task.source_evidence
    assert any(item.source_type == "alert_snapshot" for item in task.source_evidence)
    assert any(item.source_type == "risk_object_registry" for item in task.source_evidence)
    assert any(item.source_type == "plan_document" and item.clause for item in task.source_evidence)
    assert task.plan_basis
    assert all(item.version == "2026 演示有效版" for item in task.plan_basis)
    assert "6/6" in task.grounding_summary


def test_grounded_task_draft_is_blocked_when_policy_evidence_is_missing(tmp_path):
    _, workflow, event_id, _ = seed_workflow(tmp_path)

    class EmptyRag:
        def query_evidence_set(self, *args, **kwargs):
            return []

    workflow.rag_service = EmptyRag()
    result = workflow.generate_task_draft(
        event_id,
        "TUNNEL-001",
        TaskDraftGenerationRequest(
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )

    assert result.status == "insufficient_evidence"
    assert result.task is None
    assert {"procedure", "exception", "attribution"}.issubset(result.missing_roles)
    assert result.validation_errors
    timeline = workflow.repository.list_timeline_entries(event_id)
    assert timeline[-1].action == "task_draft_blocked_by_evidence_gate"


def test_manual_evidence_supplement_creates_version_and_can_be_frozen(tmp_path):
    _, workflow, event_id, _ = seed_workflow(tmp_path)

    class EmptyRag:
        def query_evidence_set(self, *args, **kwargs):
            return []

    workflow.rag_service = EmptyRag()
    blocked = workflow.generate_task_draft(
        event_id,
        "TUNNEL-001",
        TaskDraftGenerationRequest(
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )
    package = blocked.evidence_package
    assert package is not None and package.status == "needs_review"
    additions = [
        TaskEvidenceRef(
            source_type="manual_verified_document",
            source_id="MANUAL-PROCEDURE",
            title="Reviewed procedure",
            excerpt="Execute the verified closure and drainage procedure.",
            roles=[EvidenceRole.PROCEDURE],
            document_version="2026-A",
            clause="4.1",
        ),
        TaskEvidenceRef(
            source_type="manual_verified_document",
            source_id="MANUAL-EXCEPTION",
            title="Reviewed exception",
            excerpt="Escalate when equipment is insufficient.",
            roles=[EvidenceRole.EXCEPTION],
            document_version="2026-A",
            clause="4.2",
        ),
        TaskEvidenceRef(
            source_type="manual_verified_document",
            source_id="MANUAL-ATTRIBUTION",
            title="Reviewed plan attribution",
            excerpt="Current district plan version and clause verified.",
            roles=[EvidenceRole.ATTRIBUTION],
            document_version="2026-A",
            clause="4.3",
        ),
    ]
    revised = workflow.supplement_evidence_package(
        package.package_id,
        EvidenceManualSupplementRequest(
            evidence=additions,
            reason="Reviewer verified missing clauses against the source document",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="review-terminal",
        ),
    )
    assert revised.version == 2
    assert revised.status == "needs_review"
    assert revised.missing_fields == []
    frozen = workflow.freeze_evidence_package(
        package.package_id,
        EvidenceFreezeRequest(
            reason="All required fields and sources were independently checked",
            operator_id="reviewer-2",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="review-terminal-2",
        ),
    )
    assert frozen.version == 3
    assert frozen.status == "frozen"
    assert frozen.content_hash != package.content_hash


def test_retrieval_modes_compare_baseline_and_frc_without_shadow_side_effects(tmp_path):
    _, workflow, _, _ = seed_workflow(tmp_path)

    baseline = RAGDocument(
        doc_id="BASELINE-1",
        corpus=CorpusType.POLICY,
        title="Baseline plan",
        content="Baseline response procedure",
        metadata={"status": "active"},
    )
    frc = RAGDocument(
        doc_id="FRC-1",
        corpus=CorpusType.POLICY,
        title="FRC plan",
        content="FRC response procedure",
        metadata={"status": "active"},
    )

    class ModeAwareRag:
        def query(self, *args, **kwargs):
            return [baseline]

        def query_evidence_set(self, *args, **kwargs):
            return [frc]

    workflow.rag_service = ModeAwareRag()

    shadow_documents, shadow_trace = workflow._run_retrieval_strategy("response", RetrievalMode.SHADOW)
    default_documents, default_trace = workflow._run_retrieval_strategy("response", RetrievalMode.DEFAULT)
    review_documents, _ = workflow._run_retrieval_strategy("response", RetrievalMode.REVIEW)

    assert [item.doc_id for item in shadow_documents] == ["BASELINE-1"]
    assert [item.doc_id for item in default_documents] == ["FRC-1"]
    assert [item.doc_id for item in review_documents] == ["BASELINE-1", "FRC-1"]
    assert shadow_trace == default_trace
    assert shadow_trace["comparison"] == {
        "overlap": [],
        "baseline_only": ["BASELINE-1"],
        "frc_only": ["FRC-1"],
    }


def test_frc_formal_mode_is_blocked_until_gate_two_feature_flag_is_enabled(tmp_path):
    _, workflow, event_id, _ = seed_workflow(tmp_path)
    with pytest.raises(ValueError, match="Gate 2"):
        workflow.generate_task_draft(
            event_id,
            "TUNNEL-001",
            TaskDraftGenerationRequest(
                operator_id="duty-1",
                operator_role=OperatorRole.DUTY_OFFICER,
                retrieval_mode=RetrievalMode.DEFAULT,
            ),
        )

    workflow.update_feature_flag(
        FeatureFlagUpdateRequest(
            flag_key="feature.frc_rag_formal_enabled",
            enabled=True,
            environment="development",
            event_id=event_id,
            reason="Gate 2 review approved for this simulated event",
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
    )
    result = workflow.generate_task_draft(
        event_id,
        "TUNNEL-001",
        TaskDraftGenerationRequest(
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            retrieval_mode=RetrievalMode.DEFAULT,
        ),
    )
    assert result.evidence_package is not None
    assert result.evidence_package.retrieval_mode == RetrievalMode.DEFAULT


def test_duplicate_completion_feedback_is_merged_without_duplicate_record(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    workflow.acknowledge_task(task.task_id, TaskActionRequest(operator_id="liaison-1", operator_role=OperatorRole.LIAISON))
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(task.task_id, TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR))
    request = FeedbackRequest(
        operator_id="field-1",
        operator_role=OperatorRole.FIELD_OPERATOR,
        summary="现场处置完成",
        evidence=[{"type": "现场照片", "url": "object://photo-1"}, {"type": "积水深度", "value": "8cm"}],
    )

    first = workflow.submit_feedback(task.task_id, request)
    second = workflow.submit_feedback(task.task_id, request)

    assert first.status == TaskStatus.PENDING_VERIFICATION
    assert second.status == TaskStatus.PENDING_VERIFICATION
    feedback = workflow.repository.list_task_feedback(task.task_id)
    assert len(feedback) == 1
    assert feedback[0].category.value == "completion"
    assert feedback[0].dedupe_key
    assert workflow.repository.list_timeline_entries(task.event_id)[-1].action == "duplicate_feedback_merged"


def test_resource_shortage_feedback_creates_escalation_with_alternatives(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    workflow.acknowledge_task(task.task_id, TaskActionRequest(operator_id="liaison-1", operator_role=OperatorRole.LIAISON))
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(task.task_id, TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR))

    escalated = workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场排涝设备不足，请求支援",
            resource_gap="缺少一台移动排水泵",
        ),
    )

    assert escalated.status == TaskStatus.BLOCKED
    feedback = workflow.repository.list_task_feedback(task.task_id)[0]
    assert feedback.category.value == "resource_shortage"
    escalation = workflow.repository.list_escalations(event_id)[0]
    assert escalation.recommended_actions
    assert any("资源" in item for item in escalation.recommended_actions)


def test_situation_update_is_recorded_without_forcing_completion(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    workflow.acknowledge_task(task.task_id, TaskActionRequest(operator_id="liaison-1", operator_role=OperatorRole.LIAISON))
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(task.task_id, TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR))

    unchanged = workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场积水正在缓慢回落，继续观察。",
        ),
    )

    assert unchanged.status == TaskStatus.IN_PROGRESS
    feedback = workflow.repository.list_task_feedback(task.task_id)[0]
    assert feedback.category.value == "situation_update"
    assert workflow.repository.list_timeline_entries(task.event_id)[-1].action == "situation_feedback_recorded"


def test_rule_engine_hard_blocks_ungrounded_ai_draft_and_optimistic_lock_conflict(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    with pytest.raises(ValueError, match="version conflict"):
        workflow.update_task(
            task.task_id,
            TaskUpdateRequest(
                title="并发覆盖标题",
                expected_version=task.version + 1,
                operator_id="duty-1",
                operator_role=OperatorRole.DUTY_OFFICER,
            ),
        )

    now = datetime.now(timezone.utc)
    ai_task = workflow.create_task(
        event_id,
        TaskCreateRequest(
            object_id="TUNNEL-001",
            title="无冻结证据的 AI 草案",
            action="执行未经证据绑定的动作",
            responsible_organization="住建局",
            responsible_role="排水值班负责人",
            cooperate_roles=["交警联络员"],
            deadline_at=now + timedelta(hours=1),
            acknowledge_deadline_at=now + timedelta(minutes=10),
            required_evidence=["现场照片"],
            plan_basis=[PlanBasis(document="区防汛预案", version="2026", clause="4.2")],
            approval_policy=ApprovalPolicy.COMMANDER_REQUIRED,
            escalation_rule="超时升级",
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            generated_by_ai=True,
            generation_version="unsafe-ai-v1",
        ),
    )
    with pytest.raises(ValueError, match="blocked by rules"):
        workflow.submit_task(
            ai_task.task_id,
            TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER),
        )
    evaluations = workflow.repository.list_rule_evaluations(task_id=ai_task.task_id)
    assert evaluations[-1].overall_outcome == RuleOutcome.HARD_BLOCK
    assert "RULE-EVIDENCE-PACKAGE-FROZEN" in {
        item.rule_id for item in evaluations[-1].checks if item.outcome == RuleOutcome.HARD_BLOCK
    }


def test_partial_completion_can_continue_to_full_completion(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(operator_id="reviewer-1", operator_role=OperatorRole.REVIEWER, decision=ApprovalDecision.APPROVED),
    )
    workflow.acknowledge_task(task.task_id, TaskActionRequest(operator_id="liaison-1", operator_role=OperatorRole.LIAISON))
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(task.task_id, TaskActionRequest(operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR))
    partial = workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="已完成一半现场处置",
            completion_percent=50,
        ),
    )
    assert partial.status == TaskStatus.PARTIALLY_COMPLETED
    completed = workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场处置全部完成",
            completion_percent=100,
            evidence=[{"type": "现场照片", "url": "object://partial-photo"}, {"type": "积水深度", "value": "5cm"}],
        ),
    )
    assert completed.status == TaskStatus.PENDING_VERIFICATION


def test_deadline_extension_is_independently_approved_without_mutating_approved_payload(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    issued = workflow.decide_task(
        task.task_id,
        ApprovalRequest(operator_id="reviewer-1", operator_role=OperatorRole.REVIEWER, decision=ApprovalDecision.APPROVED),
    )
    original_hash = issued.approval_payload_hash
    extension = workflow.request_deadline_extension(
        task.task_id,
        DeadlineExtensionRequest(
            proposed_completion_deadline_at=task.deadline_at + timedelta(hours=1),
            proposed_verification_deadline_at=task.verification_deadline_at + timedelta(hours=1),
            reason="现场道路临时封闭，需要延后处置",
            operator_id="liaison-1",
            operator_role=OperatorRole.LIAISON,
        ),
    )
    decided = workflow.decide_deadline_extension(
        extension.extension_id,
        DeadlineExtensionDecisionRequest(
            approved=True,
            reason="已核实现场交通条件，同意延期",
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
        ),
    )
    current = workflow.repository.get_response_task(task.task_id)
    assert decided.status == DeadlineExtensionStatus.APPROVED
    assert current.effective_completion_deadline_at == task.deadline_at + timedelta(hours=1)
    assert current.approval_payload_hash == original_hash
    assert workflow.get_dashboard(event_id).deadline_extensions[-1].extension_id == extension.extension_id


def test_manual_takeover_and_commander_cancellation_are_audited_terminal_branches(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED)
    workflow.submit_task(task.task_id, TaskActionRequest(operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER))
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(operator_id="reviewer-1", operator_role=OperatorRole.REVIEWER, decision=ApprovalDecision.APPROVED),
    )
    taken = workflow.take_over_task(
        task.task_id,
        TaskActionRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            note="现场通信异常，由防办人工接管",
        ),
    )
    assert taken.status == TaskStatus.TAKEN_OVER
    cancelled = workflow.cancel_task(
        task.task_id,
        TaskActionRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            note="预警已撤销，经确认终止该模拟任务",
        ),
    )
    assert cancelled.status == TaskStatus.CANCELLED
    actions = [item.action for item in workflow.get_dashboard(event_id).timeline]
    assert "task_taken_over" in actions
    assert "task_cancelled" in actions


def test_schema_ledger_migration_quarantine_and_legacy_adapter_are_traceable(tmp_path):
    system = FloodWarningSystem(tmp_path / "migration.db")
    now = datetime.now(timezone.utc)
    system.repository.save_v2_event(
        EventRecord(
            event_id="LEGACY-EVENT-001",
            area_id="district-demo",
            title="旧演示事件",
            trigger_reason="旧页面模拟触发",
            current_stage=Stage.MONITORING,
            current_risk_level=RiskLevel.ORANGE,
            status=V2EventStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
    )
    workflow = system.response_workflow
    batch = workflow.run_legacy_migration_inventory(
        LegacyMigrationRequest(
            mapping_version="legacy-v2-to-response-v1",
            dry_run=False,
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="migration-terminal",
        )
    )
    assert system.repository.list_schema_migrations()
    assert batch.source_count == 1
    assert batch.migrated_count == 0
    assert batch.quarantined_count == 1
    assert batch.quarantine[0].reason_code == "MISSING_AUTHORITATIVE_WARNING_REVISION"

    mapped = workflow.read_legacy_event(
        "LEGACY-EVENT-001",
        operator_id="auditor-1",
        terminal_id="audit-terminal",
        trace_id="trace-legacy-001",
    )
    assert mapped["read_only"] is True
    assert mapped["migration_eligibility"] == "quarantine_pending_authoritative_warning"
    calls = system.repository.list_legacy_adapter_calls()
    assert calls[-1].trace_id == "trace-legacy-001"
    assert calls[-1].read_only is True


def test_warning_revision_marks_objects_stale_and_requires_reverification(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    before = workflow.repository.get_event_risk_object(event_id, "TUNNEL-001")
    now = datetime.now(timezone.utc)
    workflow.append_alert(
        event_id,
        AlertAppendRequest(
            alert=AlertInput(
                alert_id="ALERT-001-UPDATE",
                source_department="气象部门",
                disaster_type="暴雨",
                level="红色",
                issued_at=now,
                valid_until=now + timedelta(hours=2),
                affected_area="测试片区",
                raw_content="红色暴雨预警更新",
                lifecycle_status="updated",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="duty-terminal",
        ),
    )
    stale = workflow.repository.get_event_risk_object(event_id, "TUNNEL-001")
    assert stale.stale is True
    assert stale.version == before.version + 1
    with pytest.raises(ValueError, match="confirmed event risk objects"):
        workflow.create_task(
            event_id,
            TaskCreateRequest(
                object_id="TUNNEL-001",
                title=task.title,
                action=task.action,
                responsible_organization=task.responsible_organization,
                responsible_role=task.responsible_role,
                deadline_at=task.deadline_at,
                acknowledge_deadline_at=task.acknowledge_deadline_at,
                required_evidence=task.required_evidence,
                plan_basis=task.plan_basis,
                escalation_rule=task.escalation_rule,
                operator_id="duty-1",
                operator_role=OperatorRole.DUTY_OFFICER,
            ),
        )
    refreshed = workflow.verify_risk_object(
        event_id,
        "TUNNEL-001",
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="按新预警版本重新核验",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
        ),
    )
    assert refreshed.stale is False
    assert len(workflow.repository.list_risk_object_versions(event_id, object_id="TUNNEL-001")) >= 4


def test_evidence_conflicts_are_separate_from_applicability_and_require_human_resolution(tmp_path):
    _, workflow, event_id, _ = seed_workflow(tmp_path)
    evidence = [
        TaskEvidenceRef(
            source_type="plan_document",
            source_id="DOC-A",
            title="现行预案 A",
            excerpt="达到阈值后应封闭道路",
            roles=[EvidenceRole.PROCEDURE, EvidenceRole.ATTRIBUTION],
            document_version="2026-A",
            clause="4.1",
            conflict_key="action",
            conflict_value="close_road",
        ),
        TaskEvidenceRef(
            source_type="plan_document",
            source_id="DOC-B",
            title="专项规程 B",
            excerpt="达到阈值后应保持道路通行并设置引导",
            roles=[EvidenceRole.PROCEDURE, EvidenceRole.ATTRIBUTION],
            document_version="2026-B",
            clause="2.3",
            conflict_key="action",
            conflict_value="keep_open",
        ),
    ]
    conflicts = workflow._detect_evidence_conflicts(evidence)
    assert len(conflicts) == 1
    package = workflow._build_evidence_package(
        event_id=event_id,
        object_id="TUNNEL-001",
        evidence=evidence,
        role_coverage={role.value: True for role in EvidenceRole},
        validation_errors=["存在未决动作冲突"],
        conflicts=conflicts,
        action="待人工裁决",
        plan_basis=[PlanBasis(document="现行预案 A", version="2026-A", clause="4.1")],
        operator_id="reviewer-1",
    )
    workflow.repository.save_evidence_package(package)
    assert package.status == "needs_review"
    assert package.field_states["action"].value == "CONFLICTED"

    resolved = workflow.resolve_evidence_conflict(
        package.package_id,
        conflicts[0].conflict_id,
        EvidenceConflictResolutionRequest(
            selected_source_ids=["DOC-A"],
            reason="经辖区与文件效力核验，采用现行预案 A",
            operator_id="reviewer-2",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="review-terminal",
        ),
    )
    assert resolved.version == 2
    assert resolved.status == "frozen"
    assert resolved.conflicts[0].resolution_status == "resolved"
    assert resolved.field_states["action"].value == "SUPPORTED"


def test_task_transition_table_is_exhaustive_and_terminal_states_have_no_outgoing_edges():
    for source in TaskStatus:
        for target in TaskStatus:
            assert can_transition(source, target) is ((source, target) in ALLOWED_TASK_TRANSITIONS)
    for terminal in TERMINAL_TASK_STATUSES:
        assert not any(source == terminal for source, _ in ALLOWED_TASK_TRANSITIONS)
    assert all(source != target for source, target in ALLOWED_TASK_TRANSITIONS)


def test_read_only_event_replay_requires_integrity_and_returns_traceable_snapshot(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    replay = workflow.replay_event(
        event_id,
        TaskActionRequest(
            operator_id="auditor-1",
            operator_role=OperatorRole.AUDITOR,
            terminal_id="audit-terminal",
        ),
    )
    assert replay["mode"] == "read_only_replay"
    assert replay["integrity"].status == "verified"
    assert replay["dashboard"].event.event_id == event_id
    assert any(item.task_id == task.task_id for item in workflow.list_task_transitions(task.task_id))


def test_document_versions_are_clause_indexed_and_superseded_versions_leave_current_retrieval(tmp_path):
    system = FloodWarningSystem(tmp_path / "documents.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    first = workflow.register_document(
        DocumentImportRequest(
            document_id="DISTRICT-PLAN-001",
            title="区级下穿通道响应规程",
            version_label="2026-A",
            issuer="模拟区防办",
            jurisdiction="district-simulation",
            effective_at=now,
            content="第一条 发生橙色暴雨预警时，住建部门应核查下穿通道。\n\n第二条 若设备不足，应报告防办协调支援。",
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="document-terminal",
        )
    )
    assert first.version_number == 1
    assert len(first.clauses) == 2
    assert first.index_status == "indexed"
    duplicate = workflow.register_document(
        DocumentImportRequest(
            document_id="DISTRICT-PLAN-001",
            title="区级下穿通道响应规程",
            version_label="2026-A",
            issuer="模拟区防办",
            jurisdiction="district-simulation",
            effective_at=now,
            content="第一条 发生橙色暴雨预警时，住建部门应核查下穿通道。\n\n第二条 若设备不足，应报告防办协调支援。",
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="document-terminal",
        )
    )
    assert duplicate.version_id == first.version_id

    second = workflow.register_document(
        DocumentImportRequest(
            document_id="DISTRICT-PLAN-001",
            title="区级下穿通道响应规程",
            version_label="2026-B",
            issuer="模拟区防办",
            jurisdiction="district-simulation",
            effective_at=now + timedelta(days=1),
            replaces_version_id=first.version_id,
            content="第一条 发生红色暴雨预警时，住建部门应立即核查并组织封控下穿通道。\n\n第二条 若现场受阻，应升级防办人工接管。",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="document-terminal",
        )
    )
    assert second.version_number == 2
    documents = system.rag_service.list_documents()
    old = [item for item in documents if item.metadata.get("document_version_id") == first.version_id]
    new = [item for item in documents if item.metadata.get("document_version_id") == second.version_id]
    assert old and all(item.metadata["status"] == "superseded" for item in old)
    assert new and all(item.metadata["status"] == "active" for item in new)
    current = workflow._query_task_evidence("红色暴雨 下穿通道 封控")
    assert all(item.metadata.get("document_version_id") != first.version_id for item in current)


def test_task_creation_idempotency_key_prevents_duplicate_formal_tasks(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    request = TaskCreateRequest(
        object_id="TUNNEL-001",
        title="第二项幂等模拟任务",
        action="核查备用排水设施",
        responsible_organization="住建局",
        responsible_role="排水值班负责人",
        deadline_at=task.deadline_at,
        acknowledge_deadline_at=task.acknowledge_deadline_at,
        required_evidence=["设施照片"],
        plan_basis=task.plan_basis,
        escalation_rule="超时升级",
        idempotency_key="demo-task:TUNNEL-001:backup-drainage:v1",
        operator_id="duty-1",
        operator_role=OperatorRole.DUTY_OFFICER,
    )
    first = workflow.create_task(event_id, request)
    second = workflow.create_task(event_id, request)
    assert first.task_id == second.task_id
    matching = [
        item
        for item in workflow.repository.list_response_tasks(event_id)
        if item.creation_idempotency_key == request.idempotency_key
    ]
    assert len(matching) == 1
