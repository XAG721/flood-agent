from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flood_system.response_workflow.models import (
    AlertInput,
    EventCreateRequest,
    GeoPolygon,
    ObjectVerificationStatus,
    OperatorRole,
    PlanBasis,
    RiskObjectBatchRequest,
    RiskObjectInput,
    RiskObjectVerificationRequest,
    TaskCreateRequest,
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
    return {"db_path": str(db_path), "event_id": event_id, "task_id": task.task_id}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic simulated response fixture for PostGIS migration.")
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seed_fixture(args.db), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
