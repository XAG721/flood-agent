from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flood_system.response_workflow.models import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    DeadlineExtensionDecisionRequest,
    DeadlineExtensionRequest,
    DeadlineExtensionStatus,
    EvidenceFreezeRequest,
    EvidenceManualSupplementRequest,
    EvidenceRole,
    FeedbackRequest,
    FeatureFlagUpdateRequest,
    OperatorRole,
    PlanBasis,
    RetrievalMode,
    RuleOutcome,
    TaskActionRequest,
    TaskCreateRequest,
    TaskDraftGenerationRequest,
    TaskEvidenceRef,
    TaskStatus,
    TaskUpdateRequest,
)
from flood_system.system import FloodWarningSystem
from flood_system.models import CorpusType, RAGDocument

from tests.support.response_workflow import (
    seed_workflow,
    assign_field_operator,
)


def test_overdue_unacknowledged_task_is_escalated(tmp_path):
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
    escalated = workflow.run_deadline_sweep(
        event_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
        now=task.deadline_at + timedelta(minutes=1),
    )
    assert len(escalated) == 1
    assert escalated[0].status == TaskStatus.ESCALATED
    assert (
        workflow.repository.list_escalations(event_id)[0].reason
        == "任务超过确认时限仍未确认"
    )


@pytest.mark.parametrize(
    ("status", "deadline_field", "deadline_type", "expected_reason"),
    [
        (
            TaskStatus.ISSUED,
            "acknowledge_deadline_at",
            "acknowledge",
            "任务超过确认时限仍未确认",
        ),
        (
            TaskStatus.ACKNOWLEDGED,
            "start_deadline_at",
            "start",
            "任务超过开始时限仍未开始",
        ),
        (TaskStatus.IN_PROGRESS, "deadline_at", "completion", "任务超过完成时限"),
        (
            TaskStatus.PENDING_VERIFICATION,
            "verification_deadline_at",
            "verification",
            "任务超过核验时限",
        ),
    ],
)
def test_all_four_deadline_types_escalate_with_explicit_audit(
    tmp_path, status, deadline_field, deadline_type, expected_reason
):
    _, workflow, event_id, task = seed_workflow(
        tmp_path,
        approval_policy=ApprovalPolicy.REVIEWER_REQUIRED,
    )
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
    )
    issued = workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    now = datetime.now(timezone.utc)
    prepared = issued.model_copy(
        update={
            "status": status,
            deadline_field: now - timedelta(minutes=1),
            "effective_start_deadline_at": None,
            "effective_completion_deadline_at": None,
            "effective_verification_deadline_at": None,
        }
    )
    workflow.repository.save_response_task(prepared)

    escalated = workflow.run_deadline_sweep(
        event_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
        now=now,
    )

    assert [item.task_id for item in escalated] == [task.task_id]
    assert escalated[0].status == TaskStatus.ESCALATED
    assert workflow.repository.list_escalations(event_id)[-1].reason == expected_reason
    entry = workflow.repository.list_timeline_entries(event_id)[-1]
    assert entry.action == "deadline_escalated"
    assert entry.detail["deadline_type"] == deadline_type


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
    assert any(
        item.source_type == "risk_object_registry" for item in task.source_evidence
    )
    assert any(
        item.source_type == "plan_document" and item.clause
        for item in task.source_evidence
    )
    assert task.plan_basis
    assert all(item.version == "2026 演示有效版" for item in task.plan_basis)
    assert "6/6" in task.grounding_summary
    package = dashboard.evidence_packages[-1]
    assert len(package.field_states) == 9
    assert set(package.required_fields) == {
        "trigger_condition",
        "risk_object",
        "responsible_party",
        "action",
        "deadline",
        "feedback_requirement",
        "escalation_condition",
    }
    assert not package.blocking_missing_fields
    assert package.nli_status == "unavailable"
    assert package.nli_model_version == "nli-unavailable"
    assert all(item.source_locator for item in package.evidence)
    assert all(item.field_support for item in package.evidence)


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
    assert result.evidence_package is not None
    assert len(result.evidence_package.field_states) == 9
    assert result.evidence_package.missing_reasons
    assert set(result.evidence_package.missing_reasons.values()) == {"NOT_RETRIEVED"}
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

    shadow_documents, shadow_trace = workflow._run_retrieval_strategy(
        "response", RetrievalMode.SHADOW
    )
    default_documents, default_trace = workflow._run_retrieval_strategy(
        "response", RetrievalMode.DEFAULT
    )
    review_documents, _ = workflow._run_retrieval_strategy(
        "response", RetrievalMode.REVIEW
    )

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
    _, workflow, _, task = seed_workflow(
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
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(
        task.task_id,
        TaskActionRequest(
            operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR
        ),
    )
    request = FeedbackRequest(
        operator_id="field-1",
        operator_role=OperatorRole.FIELD_OPERATOR,
        summary="现场处置完成",
        evidence=[
            {"type": "现场照片", "url": "object://photo-1"},
            {"type": "积水深度", "value": "8cm"},
        ],
    )

    first = workflow.submit_feedback(task.task_id, request)
    second = workflow.submit_feedback(task.task_id, request)

    assert first.status == TaskStatus.PENDING_VERIFICATION
    assert second.status == TaskStatus.PENDING_VERIFICATION
    feedback = workflow.repository.list_task_feedback(task.task_id)
    assert len(feedback) == 1
    assert feedback[0].category.value == "completion"
    assert feedback[0].dedupe_key
    assert (
        workflow.repository.list_timeline_entries(task.event_id)[-1].action
        == "duplicate_feedback_merged"
    )


def test_resource_shortage_feedback_creates_escalation_with_alternatives(tmp_path):
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
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(
        task.task_id,
        TaskActionRequest(
            operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR
        ),
    )

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
    _, workflow, _, task = seed_workflow(
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
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(
        task.task_id,
        TaskActionRequest(
            operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR
        ),
    )

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
    assert (
        workflow.repository.list_timeline_entries(task.event_id)[-1].action
        == "situation_feedback_recorded"
    )


def test_rule_engine_hard_blocks_ungrounded_ai_draft_and_optimistic_lock_conflict(
    tmp_path,
):
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
            TaskActionRequest(
                operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
            ),
        )
    evaluations = workflow.repository.list_rule_evaluations(task_id=ai_task.task_id)
    assert evaluations[-1].overall_outcome == RuleOutcome.HARD_BLOCK
    assert "RULE-EVIDENCE-PACKAGE-FROZEN" in {
        item.rule_id
        for item in evaluations[-1].checks
        if item.outcome == RuleOutcome.HARD_BLOCK
    }


def test_partial_completion_can_continue_to_full_completion(tmp_path):
    _, workflow, _, task = seed_workflow(
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
    assign_field_operator(workflow, task.task_id)
    workflow.start_task(
        task.task_id,
        TaskActionRequest(
            operator_id="field-1", operator_role=OperatorRole.FIELD_OPERATOR
        ),
    )
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
            evidence=[
                {"type": "现场照片", "url": "object://partial-photo"},
                {"type": "积水深度", "value": "5cm"},
            ],
        ),
    )
    assert completed.status == TaskStatus.PENDING_VERIFICATION


def test_deadline_extension_is_independently_approved_without_mutating_approved_payload(
    tmp_path,
):
    _, workflow, event_id, task = seed_workflow(
        tmp_path, approval_policy=ApprovalPolicy.REVIEWER_REQUIRED
    )
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
    )
    issued = workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            decision=ApprovalDecision.APPROVED,
        ),
    )
    original_hash = issued.approval_payload_hash
    extension = workflow.request_deadline_extension(
        task.task_id,
        DeadlineExtensionRequest(
            proposed_completion_deadline_at=task.deadline_at + timedelta(hours=1),
            proposed_verification_deadline_at=task.verification_deadline_at
            + timedelta(hours=1),
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
    assert current.effective_completion_deadline_at == task.deadline_at + timedelta(
        hours=1
    )
    assert current.approval_payload_hash == original_hash
    assert (
        workflow.get_dashboard(event_id).deadline_extensions[-1].extension_id
        == extension.extension_id
    )


def test_manual_takeover_and_commander_cancellation_are_audited_terminal_branches(
    tmp_path,
):
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
