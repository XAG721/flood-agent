from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flood_system.response_workflow.models import (
    AlertInput,
    AlertAppendRequest,
    DocumentImportRequest,
    EvidenceConflictResolutionRequest,
    EvidenceRole,
    LegacyMigrationRequest,
    ObjectVerificationStatus,
    OperatorRole,
    PlanBasis,
    RiskObjectVerificationRequest,
    TaskActionRequest,
    TaskCreateRequest,
    TaskEvidenceRef,
    TaskStatus,
)
from flood_system.system import FloodWarningSystem
from flood_system.models import RiskLevel, Stage
from flood_system.compat.legacy_platform.models import (
    EventRecord,
    EventStatus as V2EventStatus,
)
from flood_system.response_workflow.state_machine import (
    ALLOWED_TASK_TRANSITIONS,
    TERMINAL_TASK_STATUSES,
    can_transition,
)
from flood_system.response_workflow.evidence_governance import NliResult

from tests.support.response_workflow import (
    seed_workflow,
)


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
    assert (
        len(
            workflow.repository.list_risk_object_versions(
                event_id, object_id="TUNNEL-001"
            )
        )
        >= 4
    )


def test_evidence_conflicts_are_separate_from_applicability_and_require_human_resolution(
    tmp_path,
):
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
            field_support={
                "trigger_condition": 0.8,
                "risk_object": 0.8,
                "responsible_party": 0.8,
                "action": 0.9,
                "deadline": 0.8,
                "feedback_requirement": 0.8,
                "escalation_condition": 0.8,
            },
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
    versions = workflow.list_evidence_package_versions(package.package_id)
    assert [item.version for item in versions] == [1, 2]
    comparison = workflow.compare_evidence_package_versions(package.package_id)
    assert comparison["from_version"] == 1
    assert comparison["to_version"] == 2
    assert comparison["conflict_changes"][conflicts[0].conflict_id] == {
        "before": "unresolved",
        "after": "resolved",
    }


def test_versioned_nli_adapter_creates_auditable_semantic_conflict(tmp_path):
    _, workflow, _, _ = seed_workflow(tmp_path)

    class ContradictionNli:
        model_version = "test-nli-v1"

        def classify(self, premise: str, hypothesis: str) -> NliResult:
            return NliResult(
                relation="contradiction",
                confidence=0.94,
                model_version=self.model_version,
            )

    workflow.nli_adapter = ContradictionNli()
    evidence = [
        TaskEvidenceRef(
            source_type="plan_document",
            source_id="DOC-CLOSE",
            title="道路封控规程",
            excerpt="达到橙色阈值后必须立即封闭下穿道路。",
            roles=[EvidenceRole.PROCEDURE],
            source_locator="document://DOC-CLOSE#4.1",
            field_support={"action": 0.92},
        ),
        TaskEvidenceRef(
            source_type="plan_document",
            source_id="DOC-OPEN",
            title="道路保通规程",
            excerpt="达到橙色阈值后禁止封闭下穿道路，应保持通行。",
            roles=[EvidenceRole.PROCEDURE],
            source_locator="document://DOC-OPEN#2.3",
            field_support={"action": 0.9},
        ),
    ]

    conflicts, assessments = workflow._analyze_evidence_conflicts(evidence)
    repeated, repeated_assessments = workflow._analyze_evidence_conflicts(evidence)

    assert len(assessments) == 1
    assert assessments[0].relation.value == "contradiction"
    assert assessments[0].model_version == "test-nli-v1"
    assert conflicts[0].conflict_type == "SEMANTIC_CONTRADICTION"
    assert conflicts[0].nli_assessment_id == assessments[0].assessment_id
    assert conflicts[0].detection_methods == ["versioned_nli"]
    assert repeated[0].conflict_id == conflicts[0].conflict_id
    assert repeated_assessments[0].assessment_id == assessments[0].assessment_id


def test_task_transition_table_is_exhaustive_and_terminal_states_have_no_outgoing_edges():
    for source in TaskStatus:
        for target in TaskStatus:
            assert can_transition(source, target) is (
                (source, target) in ALLOWED_TASK_TRANSITIONS
            )
    for terminal in TERMINAL_TASK_STATUSES:
        assert not any(source == terminal for source, _ in ALLOWED_TASK_TRANSITIONS)
    assert all(source != target for source, target in ALLOWED_TASK_TRANSITIONS)


def test_read_only_event_replay_requires_integrity_and_returns_traceable_snapshot(
    tmp_path,
):
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
    assert any(
        item.task_id == task.task_id
        for item in workflow.list_task_transitions(task.task_id)
    )


def test_document_versions_are_clause_indexed_and_superseded_versions_leave_current_retrieval(
    tmp_path,
):
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
    old = [
        item
        for item in documents
        if item.metadata.get("document_version_id") == first.version_id
    ]
    new = [
        item
        for item in documents
        if item.metadata.get("document_version_id") == second.version_id
    ]
    assert old and all(item.metadata["status"] == "superseded" for item in old)
    assert new and all(item.metadata["status"] == "active" for item in new)
    current = workflow._query_task_evidence("红色暴雨 下穿通道 封控")
    assert all(
        item.metadata.get("document_version_id") != first.version_id for item in current
    )


def test_document_parse_and_index_dependency_failures_are_isolated(tmp_path):
    system = FloodWarningSystem(tmp_path / "document-failure-isolation.db")
    workflow = system.response_workflow
    request = DocumentImportRequest(
        document_id="PLAN-FAILURE-001",
        title="不可解析模拟文档",
        version_label="2026-failure",
        issuer="模拟区防办",
        jurisdiction="district-demo",
        effective_at=datetime.now(timezone.utc),
        content="        ",
        operator_id="admin-1",
        operator_role=OperatorRole.ADMIN,
    )

    with pytest.raises(ValueError, match="no indexable clauses"):
        workflow.register_document(request)
    assert workflow.list_document_versions("PLAN-FAILURE-001") == []

    workflow.rag_service = None
    with pytest.raises(ValueError, match="indexing service is unavailable"):
        workflow.register_document(
            request.model_copy(
                update={"content": "有效条款：达到橙色预警时应核查下穿通道。"}
            )
        )
    assert workflow.list_document_versions("PLAN-FAILURE-001") == []


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
