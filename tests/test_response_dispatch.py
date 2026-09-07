from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from flood_system.response_workflow.models import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    CallbackSequenceState,
    DispatchCallbackRequest,
    DispatchCallbackStatus,
    EventCloseRequest,
    FeedbackRequest,
    FeatureFlagUpdateRequest,
    OperatorRole,
    OutboxProcessRequest,
    OutboxStatus,
    ScenarioEvaluationRequest,
    TaskActionRequest,
    TaskStatus,
)

from tests.support.response_workflow import (
    seed_workflow,
    assign_field_operator,
    approve_task_with_pending_outbox,
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
    assert (
        workflow.is_feature_enabled("feature.simulated_dispatch", event_id=event_id)
        is True
    )
    assert any(
        item.action == "feature_flag_changed"
        for item in workflow.get_dashboard(event_id).timeline
    )


def test_approved_dispatch_uses_idempotent_outbox_and_payload_hash(tmp_path):
    _, workflow, _, task = seed_workflow(tmp_path)
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
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
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
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

    system.repository.save_response_task(
        issued.model_copy(update={"action": "未经重新审批的篡改动作"})
    )
    result = workflow.process_outbox(
        OutboxProcessRequest(
            message_id=message.message_id,
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
        )
    )
    assert result[0].status == OutboxStatus.FAILED
    assert result[0].last_error == "approved payload hash mismatch"


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_callbacks"),
    [
        ("normal", OutboxStatus.SENT, 2),
        ("reject", OutboxStatus.FAILED, 1),
        ("partial_success", OutboxStatus.PARTIALLY_SENT, 1),
        ("duplicate_callback", OutboxStatus.SENT, 2),
        ("out_of_order_callback", OutboxStatus.SENT, 2),
    ],
)
def test_simulated_dispatch_fault_matrix_is_audited_without_mutating_task_state(
    tmp_path, scenario, expected_status, expected_callbacks
):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    issued, message = approve_task_with_pending_outbox(workflow, event_id, task)

    result = workflow.process_outbox(
        OutboxProcessRequest(
            message_id=message.message_id,
            simulation_scenario=scenario,
            operator_id="liaison-1",
            operator_role=OperatorRole.LIAISON,
        )
    )[0]

    assert result.status == expected_status
    assert result.simulation_scenario == scenario
    assert result.callback_count == expected_callbacks
    assert (
        workflow.get_dashboard(event_id).tasks[0].status
        == issued.status
        == TaskStatus.ISSUED
    )
    callbacks = workflow.list_dispatch_callbacks(message.message_id)
    assert len(callbacks) == expected_callbacks
    if scenario == "duplicate_callback":
        assert any(
            item.action == "duplicate_dispatch_callback_ignored"
            for item in workflow.get_dashboard(event_id).timeline
        )
    if scenario == "out_of_order_callback":
        assert [item.version for item in callbacks] == [2, 1]
        assert callbacks[1].sequence_state == CallbackSequenceState.OUT_OF_ORDER


def test_simulated_dispatch_timeout_stays_pending_and_recovers_idempotently(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    issued, message = approve_task_with_pending_outbox(workflow, event_id, task)

    timed_out = workflow.process_outbox(
        OutboxProcessRequest(
            message_id=message.message_id,
            simulation_scenario="timeout",
            operator_id="liaison-1",
            operator_role=OperatorRole.LIAISON,
        )
    )[0]
    recovered = workflow.process_outbox(
        OutboxProcessRequest(
            message_id=message.message_id,
            simulation_scenario="normal",
            operator_id="liaison-1",
            operator_role=OperatorRole.LIAISON,
        )
    )[0]

    assert timed_out.status == OutboxStatus.PENDING
    assert timed_out.gateway_status == "timeout"
    assert timed_out.attempts == 1
    assert recovered.status == OutboxStatus.SENT
    assert recovered.attempts == 2
    assert recovered.callback_count == 2
    assert (
        workflow.get_dashboard(event_id).tasks[0].status
        == issued.status
        == TaskStatus.ISSUED
    )


def test_external_service_callback_is_token_bound_idempotent_and_state_safe(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    issued, message = approve_task_with_pending_outbox(workflow, event_id, task)
    request = DispatchCallbackRequest(
        message_id=message.message_id,
        external_id="SIMDISPATCH-EXTERNAL-1",
        source="deterministic_simulation_gateway",
        version=1,
        event_time=datetime.now(timezone.utc),
        request_id="SIMREQ-external-1",
        trace_id="SIMTRACE-external-1",
        idempotency_key="callback-external-1",
        status=DispatchCallbackStatus.DELIVERED,
        mock_token="simulation-only-change-me",
        operator_id="simulation-gateway",
        operator_role=OperatorRole.EXTERNAL_SERVICE,
        terminal_id="simulation-gateway-terminal",
    )

    first = workflow.ingest_simulated_dispatch_callback(request)
    duplicate = workflow.ingest_simulated_dispatch_callback(request)

    assert duplicate.callback_id == first.callback_id
    assert len(workflow.list_dispatch_callbacks(message.message_id)) == 1
    assert (
        workflow.get_dashboard(event_id).tasks[0].status
        == issued.status
        == TaskStatus.ISSUED
    )
    assert any(
        item.action == "duplicate_dispatch_callback_ignored"
        for item in workflow.get_dashboard(event_id).timeline
    )
    with pytest.raises(PermissionError, match="invalid simulation callback token"):
        workflow.ingest_simulated_dispatch_callback(
            request.model_copy(update={"mock_token": "wrong-token"})
        )


def test_full_deterministic_response_loop_and_close(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
    )
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
    pending = workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场处置完成",
            evidence=[
                {"type": "现场照片", "url": "object://photo-1"},
                {"type": "积水深度", "value": "8cm"},
            ],
        ),
    )
    assert pending.status == TaskStatus.PENDING_VERIFICATION
    completed = workflow.verify_completion(
        task.task_id,
        True,
        TaskActionRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            note="证据完整",
        ),
    )
    assert completed.status == TaskStatus.COMPLETED
    closed = workflow.close_event(
        event_id,
        EventCloseRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            note="闭环完成",
        ),
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


def test_verification_can_return_completion_for_rework_without_losing_feedback(
    tmp_path,
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
    workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场处置完成，提交初次核验",
            evidence=[
                {"type": "现场照片", "url": "object://photo-rework"},
                {"type": "积水深度", "value": "8cm"},
            ],
        ),
    )

    returned = workflow.verify_completion(
        task.task_id,
        False,
        TaskActionRequest(
            operator_id="reviewer-2",
            operator_role=OperatorRole.REVIEWER,
            note="照片缺少时间标记，退回整改",
        ),
    )

    assert returned.status == TaskStatus.IN_PROGRESS
    assert len(workflow.repository.list_task_feedback(task.task_id)) == 1
    assert (
        workflow.repository.list_timeline_entries(event_id)[-1].action
        == "completion_returned"
    )


def test_closed_event_generates_persisted_district_scenario_report(tmp_path):
    _, workflow, event_id, task = seed_workflow(tmp_path)
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="duty-1", operator_role=OperatorRole.DUTY_OFFICER
        ),
    )
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
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
    workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="field-1",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="现场处置完成",
            evidence=[
                {"type": "现场照片", "url": "object://photo-1"},
                {"type": "积水深度", "value": "8cm"},
            ],
        ),
    )
    workflow.verify_completion(
        task.task_id,
        True,
        TaskActionRequest(
            operator_id="reviewer-1", operator_role=OperatorRole.REVIEWER
        ),
    )
    workflow.close_event(
        event_id,
        EventCloseRequest(
            operator_id="commander-1", operator_role=OperatorRole.COMMANDER
        ),
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
    assert (
        next(
            item for item in report.metrics if item.metric_id == "object_omission_rate"
        ).system_value
        == 0.0
    )
    assert (
        next(
            item for item in report.metrics if item.metric_id == "approval_duration"
        ).manual_value
        == 20.0
    )
    assert workflow.list_scenario_reports(event_id)[0].report_id == report.report_id
    assert (
        workflow.get_dashboard(event_id).scenario_report.report_id == report.report_id
    )


def test_incomplete_scenario_report_exposes_blocking_acceptance_failures(tmp_path):
    _, workflow, event_id, _ = seed_workflow(tmp_path)

    report = workflow.run_scenario_evaluation(
        event_id,
        ScenarioEvaluationRequest(
            operator_id="auditor-1", operator_role=OperatorRole.AUDITOR
        ),
    )

    assert report.overall_status == "failed"
    assert any(not check.passed for check in report.acceptance_checks)
    assert any(item.severity == "blocking" for item in report.failure_cases)

    rerun = workflow.run_scenario_evaluation(
        event_id,
        ScenarioEvaluationRequest(
            operator_id="auditor-1", operator_role=OperatorRole.AUDITOR
        ),
    )
    assert rerun.overall_status == "failed"
    assert "auditor" not in rerun.scope["operator_roles"]


def test_timeline_hash_chain_is_verified_and_immutable_records_cannot_be_deleted(
    tmp_path,
):
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
    assert all(
        timeline[index].previous_hash == timeline[index - 1].record_hash
        for index in range(1, len(timeline))
    )
    submitted = next(item for item in timeline if item.action == "task_submitted")
    assert submitted.terminal_id == "district-console-07"
    assert submitted.before_state == {"status": "draft"}
    assert submitted.after_state["status"] == "pending_approval"

    with pytest.raises(sqlite3.IntegrityError, match="timeline cannot be deleted"):
        with workflow.repository._connect() as conn:
            conn.execute(
                "DELETE FROM response_timeline WHERE event_id = ?", (event_id,)
            )
    with pytest.raises(
        sqlite3.IntegrityError, match="alert snapshots cannot be deleted"
    ):
        with workflow.repository._connect() as conn:
            conn.execute(
                "DELETE FROM response_alert_snapshots WHERE event_id = ?", (event_id,)
            )
