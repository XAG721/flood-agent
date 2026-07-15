from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flood_system.response_workflow.models import (
    AlertInput,
    ApprovalDecision,
    ApprovalRequest,
    EvidenceFieldState,
    EvidencePackageVersion,
    EvidenceRole,
    EventCreateRequest,
    FeedbackRequest,
    GeoPolygon,
    ObjectVerificationStatus,
    OperatorRole,
    PlanBasis,
    RiskObjectBatchRequest,
    RiskObjectInput,
    RiskObjectVerificationRequest,
    TaskActionRequest,
    TaskAssignmentRequest,
    TaskCreateRequest,
    TaskEvidenceRef,
)
from flood_system.system import FloodWarningSystem


def seed_fixture(db_path: Path) -> dict[str, str]:
    db_path = db_path.expanduser().resolve()
    if db_path.exists():
        raise FileExistsError(f"fixture database already exists: {db_path}")
    system = FloodWarningSystem(db_path)
    workflow = system.response_workflow
    now = datetime(2026, 7, 12, 6, 0, tzinfo=timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="PostGIS 迁移演练事件",
            area_id="district-simulation",
            alert=AlertInput(
                alert_id="SIM-POSTGIS-ALERT-001",
                source_department="模拟专业预警发布端",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                valid_until=now + timedelta(hours=3),
                affected_area="模拟空间迁移片区",
                affected_geometry=GeoPolygon(
                    coordinates=[
                        (108.9500, 34.2400),
                        (108.9650, 34.2400),
                        (108.9650, 34.2550),
                        (108.9500, 34.2550),
                        (108.9500, 34.2400),
                    ]
                ),
                raw_content="用于验证 SQLite 到 PostGIS 单向影子迁移的模拟预警。",
                source_type="simulation",
                source_version="postgis-pilot-v1",
                is_simulated=True,
            ),
            operator_id="migration-fixture-duty",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="migration-fixture",
        )
    )
    event_id = dashboard.event.event_id
    workflow.add_risk_objects(
        event_id,
        RiskObjectBatchRequest(
            objects=[
                RiskObjectInput(
                    object_id="SIM-POSTGIS-TUNNEL-001",
                    name="模拟下穿通道",
                    object_type="underpass_tunnel",
                    location="受限位置字段",
                    responsible_organization="模拟区住建部门",
                    responsible_role="排水值守岗",
                    trigger_reasons=["模拟预警多边形覆盖"],
                    source_refs=["SIM-POSTGIS-ALERT-001"],
                    vulnerability="低洼且历史积水",
                    risk_score=88,
                    system_explanation="仅用于迁移与空间索引演练",
                    source_type="simulation",
                    source_version="postgis-pilot-v1",
                    is_simulated=True,
                )
            ],
            operator_id="migration-fixture-duty",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="migration-fixture",
        ),
    )
    workflow.verify_risk_object(
        event_id,
        "SIM-POSTGIS-TUNNEL-001",
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="模拟台账核验通过",
            operator_id="migration-fixture-reviewer",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="migration-fixture",
        ),
    )
    task = workflow.create_task(
        event_id,
        TaskCreateRequest(
            object_id="SIM-POSTGIS-TUNNEL-001",
            title="模拟现场值守",
            action="到场核查并准备交通管控",
            responsible_organization="模拟区住建部门",
            responsible_role="排水值守岗",
            cooperate_roles=["模拟交警联络岗"],
            deadline_at=now + timedelta(days=3650),
            acknowledge_deadline_at=now + timedelta(days=3649, hours=23),
            required_evidence=["模拟现场照片", "模拟积水深度"],
            plan_basis=[PlanBasis(document="模拟区防汛预案", version="2026-sim", clause="4.2")],
            escalation_rule="模拟超时上报",
            operator_id="migration-fixture-duty",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="migration-fixture",
            generated_by_ai=False,
        ),
    )
    evidence = TaskEvidenceRef(
        source_type="simulated_verified_document",
        source_id="SIM-POSTGIS-PLAN-001",
        title="模拟区防汛预案",
        excerpt="下穿通道达到警戒条件时应现场值守并准备交通管控。",
        roles=[EvidenceRole.PROCEDURE, EvidenceRole.ATTRIBUTION],
        document_version="2026-sim",
        clause="4.2",
    )
    evidence_hash = hashlib.sha256(evidence.model_dump_json().encode("utf-8")).hexdigest()
    system.repository.save_evidence_package(
        EvidencePackageVersion(
            package_id="SIM-POSTGIS-EVIDENCE-001",
            event_id=event_id,
            object_id="SIM-POSTGIS-TUNNEL-001",
            status="frozen",
            retrieval_run_id="SIM-POSTGIS-RETRIEVAL-001",
            field_states={"action": EvidenceFieldState.SUPPORTED},
            role_coverage={"procedure": True, "attribution": True},
            evidence=[evidence],
            content_hash=evidence_hash,
            created_by="migration-fixture-reviewer",
            reviewed_by="migration-fixture-reviewer",
            frozen_at=now,
            created_at=now,
        )
    )
    workflow.submit_task(
        task.task_id,
        TaskActionRequest(
            operator_id="migration-fixture-duty",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="migration-fixture",
        ),
    )
    workflow.decide_task(
        task.task_id,
        ApprovalRequest(
            decision=ApprovalDecision.APPROVED,
            operator_id="migration-fixture-reviewer",
            operator_role=OperatorRole.REVIEWER,
            note="模拟迁移演练批准",
            terminal_id="migration-fixture",
        ),
    )
    workflow.acknowledge_task(
        task.task_id,
        TaskActionRequest(
            operator_id="migration-fixture-liaison",
            operator_role=OperatorRole.LIAISON,
            terminal_id="migration-fixture",
        ),
    )
    workflow.assign_task(
        task.task_id,
        TaskAssignmentRequest(
            operator_id="migration-fixture-liaison",
            operator_role=OperatorRole.LIAISON,
            assignee_id="migration-fixture-field",
            assignee_name="模拟现场执行员",
            assignee_role=OperatorRole.FIELD_OPERATOR,
            reason="模拟迁移演练分派",
            terminal_id="migration-fixture",
        ),
    )
    workflow.start_task(
        task.task_id,
        TaskActionRequest(
            operator_id="migration-fixture-field",
            operator_role=OperatorRole.FIELD_OPERATOR,
            terminal_id="migration-fixture",
        ),
    )
    workflow.submit_feedback(
        task.task_id,
        FeedbackRequest(
            operator_id="migration-fixture-field",
            operator_role=OperatorRole.FIELD_OPERATOR,
            summary="模拟现场处置完成，等待独立核验",
            evidence=[
                {"type": "模拟现场照片", "url": "simulated://postgis/photo-001"},
                {"type": "模拟积水深度", "value": "8cm"},
            ],
            completion_percent=100,
            terminal_id="migration-fixture",
        ),
    )
    return {"db_path": str(db_path), "event_id": event_id, "task_id": task.task_id}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic simulated response fixture for PostGIS migration.")
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seed_fixture(args.db), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
