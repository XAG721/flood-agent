from __future__ import annotations

from datetime import datetime, timedelta, timezone


from flood_system.response_workflow.models import (
    AlertInput,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    EventCreateRequest,
    FeatureFlagUpdateRequest,
    ObjectVerificationStatus,
    OperatorRole,
    OutboxStatus,
    PlanBasis,
    RiskObjectBatchRequest,
    RiskObjectInput,
    RiskObjectVerificationRequest,
    TaskActionRequest,
    TaskAssignmentRequest,
    TaskCreateRequest,
)
from flood_system.system import FloodWarningSystem


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


def approve_task_with_pending_outbox(workflow, event_id: str, task):
    workflow.update_feature_flag(
        FeatureFlagUpdateRequest(
            flag_key="feature.simulated_dispatch",
            enabled=False,
            environment="development",
            event_id=event_id,
            reason="保留下发消息以注入模拟网关故障",
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="admin-terminal",
        )
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
            operator_id="commander-1",
            operator_role=OperatorRole.COMMANDER,
            decision=ApprovalDecision.APPROVED,
            note="批准后执行模拟接口故障矩阵",
        ),
    )
    message = workflow.list_outbox_messages(event_id=event_id)[0]
    assert message.status == OutboxStatus.PENDING
    return issued, message
