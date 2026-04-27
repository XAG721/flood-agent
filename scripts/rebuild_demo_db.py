from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flood_system.models import ResourceStatus, RiskLevel, Stage
from flood_system.system import FloodWarningSystem
from flood_system.v2.models import (
    ActionProposal,
    AgentName,
    AgentResult,
    AgentTask,
    AgentTaskEvent,
    AgentTaskEventType,
    AgentTaskStatus,
    AlertSeverity,
    AuditRecord,
    AutonomyLevel,
    EventRecord,
    EventStatus,
    ExecutionLogEntry,
    ExecutionMode,
    GenerationSource,
    HazardState,
    HazardTile,
    MonitoringPointState,
    NotificationDraft,
    ProposalStatus,
    RoadReachability,
    SharedMemorySnapshot,
    SupervisorHealthState,
    SupervisorRunRecord,
    SupervisorRunStatus,
)
from flood_system.v3.models import AudienceWarningDraft

DEFAULT_DB_PATH = ROOT / "data" / "flood_warning_system_demo.db"
DEMO_EVENT_ID = "event_demo_beilin_primary"
HISTORY_EVENT_ID = "event_demo_beilin_closed"
AREA_ID = "beilin_10km2"


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def prepare_db_file(db_path: Path, *, force: bool) -> Path | None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return None
    if not force:
        raise SystemExit(f"{db_path} already exists. Re-run with --force to rebuild it.")
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{db_path.stem}_{utc_now().strftime('%Y%m%d_%H%M%S')}{db_path.suffix}"
    shutil.move(str(db_path), str(backup_path))
    return backup_path


def build_primary_event(now: datetime) -> EventRecord:
    return EventRecord(
        event_id=DEMO_EVENT_ID,
        area_id=AREA_ID,
        title="碑林区强降雨内涝风险演示事件",
        trigger_reason="production_demo_seed",
        current_stage=Stage.RESPONSE,
        current_risk_level=RiskLevel.RED,
        status=EventStatus.ACTIVE,
        metadata={
            "source": "scripts/rebuild_demo_db.py",
            "demo_mode": "production_grade_client_demo",
            "scenario": "school_hospital_community_closure",
        },
        created_at=now - timedelta(minutes=35),
        updated_at=now,
    )


def build_history_event(now: datetime) -> EventRecord:
    return EventRecord(
        event_id=HISTORY_EVENT_ID,
        area_id=AREA_ID,
        title="碑林区内涝处置复盘样例",
        trigger_reason="demo_history_seed",
        current_stage=Stage.POSTMORTEM,
        current_risk_level=RiskLevel.BLUE,
        status=EventStatus.CLOSED,
        metadata={"source": "scripts/rebuild_demo_db.py", "demo_history": True},
        created_at=now - timedelta(days=5, hours=2),
        updated_at=now - timedelta(days=5),
    )


def build_hazard_state(event_id: str, now: datetime, *, closed: bool = False) -> HazardState:
    risk_level = RiskLevel.BLUE if closed else RiskLevel.RED
    score_base = 42.0 if closed else 92.0
    depth_base = 18.0 if closed else 74.0
    trend = "falling" if closed else "rapidly_rising"
    generated_at = now - timedelta(days=5) if closed else now - timedelta(minutes=2)
    villages = [
        ("文艺路街道", "文艺路至南门连接线"),
        ("南院门街道", "南门地铁至体育学院转移通道"),
        ("东关南街街道", "东大街至李家村通道"),
        ("太乙路街道", "和平门医养中心转移通道"),
        ("建国路片区", "建国路至和平门路线"),
        ("柏树林街道", "柏树林至东大街排水走廊"),
    ]
    tiles: list[HazardTile] = []
    for area_name, road_name in villages:
        for horizon, offset in [(10, 0), (30, 6), (60, 14)]:
            tiles.append(
                HazardTile(
                    tile_id=f"{area_name}_{horizon}",
                    area_name=area_name,
                    horizon_minutes=horizon,
                    risk_level=risk_level,
                    risk_score=min(100.0, score_base + offset),
                    predicted_water_depth_cm=depth_base + offset,
                    trend=trend,
                    uncertainty=0.24 if not closed else 0.16,
                    affected_roads=[] if closed else [road_name],
                )
            )
    roads = [
        RoadReachability(
            road_id=f"beilin_demo_road_{index}",
            name=road_name,
            from_village=area_name,
            to_location=destination,
            accessible=closed or index in {4, 6},
            travel_time_minutes=12 + index,
            depth_limit_cm=28.0,
            failure_reason="" if closed or index in {4, 6} else f"{road_name} 已超过通行水深阈值，需要绕行。",
        )
        for index, (area_name, road_name, destination) in enumerate(
            [
                ("文艺路街道", "文艺路至南门连接线", "南门地铁入口"),
                ("南院门街道", "南门地铁至体育学院转移通道", "西安体育学院避难点"),
                ("东关南街街道", "东大街至李家村通道", "李家村商圈高地"),
                ("太乙路街道", "和平门医养中心转移通道", "和平门避难点"),
                ("建国路片区", "建国路至和平门路线", "和平门避难点"),
                ("柏树林街道", "柏树林至东大街排水走廊", "东大街集结点"),
            ],
            start=1,
        )
    ]
    points = [
        MonitoringPointState(
            point_name="南门地铁泵站",
            latest_water_level_m=4.62 if not closed else 2.1,
            latest_rainfall_mm=32.5 if not closed else 4.0,
            status="严重" if not closed else "恢复",
            updated_at=generated_at,
        ),
        MonitoringPointState(
            point_name="文艺路下穿监测点",
            latest_water_level_m=4.71 if not closed else 2.0,
            latest_rainfall_mm=31.2 if not closed else 3.6,
            status="严重" if not closed else "恢复",
            updated_at=generated_at,
        ),
        MonitoringPointState(
            point_name="李家村集水井",
            latest_water_level_m=4.54 if not closed else 1.9,
            latest_rainfall_mm=28.8 if not closed else 3.1,
            status="警戒" if not closed else "恢复",
            updated_at=generated_at,
        ),
        MonitoringPointState(
            point_name="和平门下穿监测点",
            latest_water_level_m=4.48 if not closed else 2.2,
            latest_rainfall_mm=27.9 if not closed else 2.8,
            status="警戒" if not closed else "恢复",
            updated_at=generated_at,
        ),
    ]
    return HazardState(
        event_id=event_id,
        area_id=AREA_ID,
        generated_at=generated_at,
        overall_risk_level=risk_level,
        overall_score=100.0 if not closed else 38.0,
        trend=trend,
        uncertainty=0.24 if not closed else 0.16,
        freshness_seconds=120 if not closed else 3600,
        hazard_tiles=tiles,
        road_reachability=roads,
        monitoring_points=points,
    )


def build_proposals(now: datetime) -> list[ActionProposal]:
    return [
        ActionProposal(
            proposal_id="proposal_demo_pending_school",
            event_id=DEMO_EVENT_ID,
            entity_id="community_jsl_grid",
            area_id=AREA_ID,
            proposal_scope="regional",
            action_type="evacuation_prepare",
            execution_mode=ExecutionMode.EVACUATION_TASK,
            action_display_name="社区网格协同转移准备",
            action_display_tagline="先排涝封控，后分户协助",
            action_display_category="community",
            title="建设路低洼网格地下室回流协助与排涝封控",
            summary="建议立即对建设路低洼网格三组启动排涝封控和分户摸排，优先协助老人家庭、地下室住户和行动不便居民离开低洼院落。",
            trigger_reason="建设路低洼网格已进入 Red 风险，社区上报地下室和老院落出现回流水，三维主屏需要直接标记待审批处置点。",
            recommendation="启动社区网格协同转移准备，优先封控低洼巷道、布置临时排水泵，并同步通知网格员逐户确认。",
            evidence_summary="监测点水位升高、建设路社区网格上报回流、低洼院落与老人家庭风险叠加。",
            severity="critical",
            requires_confirmation=True,
            required_operator_roles=["commander"],
            payload={"decision_confidence": 0.86, "demo_role": "pending_main_screen"},
            high_risk_object_ids=["community_jsl_grid", "resident_elderly_ls1"],
            action_scope={"roads": ["建国路至和平门路线"], "audiences": ["community", "elderly_residents", "grid_workers"]},
            risk_stage_key="community_grid_backflow",
            system_version_hash="demo-v3",
            generation_source=GenerationSource.SYSTEM,
            grounding_summary="社区网格、监测点和脆弱人群画像共同触发人工审批。",
            chat_follow_up_prompt="请说明建设路低洼网格 proposal 的证据与执行边界。",
            status=ProposalStatus.PENDING,
            created_at=now - timedelta(minutes=18),
        ),
        ActionProposal(
            proposal_id="proposal_demo_approved_hospital",
            event_id=DEMO_EVENT_ID,
            entity_id="hospital_bl_central",
            area_id=AREA_ID,
            proposal_scope="regional",
            action_type="resource_dispatch",
            execution_mode=ExecutionMode.RESOURCE_DISPATCH,
            action_display_name="医院通道保障",
            action_display_tagline="保急诊、保电力、保救护入口",
            action_display_category="hospital",
            title="碑林中心医院急诊与后勤入口保障",
            summary="已批准调度排水泵、救护车和交警协同，优先保障医院急诊通道与后备电力入口。",
            trigger_reason="医院处于东关南街高风险片区，急诊通道和后勤入口存在积水阻断风险。",
            recommendation="保障医院连续服务能力，并保持救护车辆通行。",
            evidence_summary="医院画像、道路可达性、监测点水位和资源状态共同支持该动作。",
            severity="critical",
            requires_confirmation=True,
            required_operator_roles=["commander"],
            payload={"decision_confidence": 0.91, "demo_role": "approved_warning_ready"},
            high_risk_object_ids=["hospital_bl_central", "nursing_home_hpm"],
            action_scope={"resources": ["portable_pumps", "ambulance", "traffic_police"], "priority": "critical_service"},
            risk_stage_key="hospital_continuity",
            system_version_hash="demo-v3",
            generation_source=GenerationSource.SYSTEM,
            grounding_summary="关键服务连续性和交通阻断证据支持该动作。",
            status=ProposalStatus.APPROVED,
            resolved_at=now - timedelta(minutes=8),
            resolved_by="demo_commander",
            resolution_note="同意执行，先保障急诊入口和后勤供电入口。",
            created_at=now - timedelta(minutes=22),
        ),
    ]


def seed_agent_council(system: FloodWarningSystem, now: datetime) -> list[str]:
    repo = system.repository
    roles = [
        (AgentName.HAZARD, "assess_hazard", "风险态势已达到 Red，水位与道路阻断信号正在共同上升。", 0.92),
        (AgentName.EXPOSURE, "assess_exposure", "重点影响对象包括学校、医院、医养中心和地下交通空间。", 0.88),
        (AgentName.RESOURCE, "assess_resources", "排水泵、救护车和交警资源需要优先绑定医院与学校场景。", 0.82),
        (AgentName.PLANNING, "draft_plan", "建议先放行医院保障动作，社区网格协同转移动作保留人工审批闸门。", 0.86),
        (AgentName.POLICY, "audit_constraints", "高风险人群转移和交通管制动作需要指挥员审批。", 0.84),
        (AgentName.COMMS, "draft_warning", "已批准动作可生成领导版、部门版、社区版与公众版 warning。", 0.8),
    ]
    result_ids: list[str] = []
    active_agents: list[AgentName] = []
    for index, (agent_name, task_type, summary, confidence) in enumerate(roles, start=1):
        created_at = now - timedelta(minutes=30 - index)
        task_id = f"task_demo_{agent_name.value}"
        result_id = f"result_demo_{agent_name.value}"
        active_agents.append(agent_name)
        repo.save_v2_agent_task(
            AgentTask(
                task_id=task_id,
                event_id=DEMO_EVENT_ID,
                agent_name=agent_name,
                task_type=task_type,
                status=AgentTaskStatus.COMPLETED,
                input_payload={"scenario": "production_demo"},
                output_payload={"summary": summary},
                priority=index,
                source_trigger_id="trigger_demo_initial",
                created_at=created_at,
                started_at=created_at + timedelta(seconds=5),
                completed_at=created_at + timedelta(seconds=18),
            )
        )
        repo.add_v2_agent_task_event(
            AgentTaskEvent(
                task_event_id=f"tke_demo_{agent_name.value}_queued",
                event_id=DEMO_EVENT_ID,
                task_id=task_id,
                agent_name=agent_name,
                event_type=AgentTaskEventType.TASK_ENQUEUED,
                trigger_id="trigger_demo_initial",
                payload={"summary": f"{agent_name.value} 已进入会商队列。"},
                created_at=created_at,
            )
        )
        repo.add_v2_agent_task_event(
            AgentTaskEvent(
                task_event_id=f"tke_demo_{agent_name.value}_result",
                event_id=DEMO_EVENT_ID,
                task_id=task_id,
                agent_name=agent_name,
                event_type=AgentTaskEventType.AGENT_RESULT_SAVED,
                trigger_id="trigger_demo_initial",
                payload={"summary": summary, "result_id": result_id},
                created_at=created_at + timedelta(seconds=20),
            )
        )
        repo.save_v2_agent_result(
            AgentResult(
                result_id=result_id,
                task_id=task_id,
                event_id=DEMO_EVENT_ID,
                agent_name=agent_name,
                summary=summary,
                structured_output={
                    "role": agent_name.value,
                    "recommended_next_tasks": ["human_gate", "warning_drafting"],
                    "scenario": "production_demo",
                },
                confidence=confidence,
                decision_confidence=confidence - 0.04,
                evidence_refs=["hazard_demo_primary", "profile_community_jsl_grid", "profile_hospital_bl_central"],
                missing_slots=[] if agent_name != AgentName.RESOURCE else ["现场排水泵到场 ETA 仍需值班员确认"],
                handoff_recommendations=["进入 Orchestrator 合成", "保持人工审批闸门"],
                recommended_next_tasks=["proposal_review", "warning_generation"],
                created_at=created_at + timedelta(seconds=22),
            )
        )
        result_ids.append(result_id)
    repo.save_v2_event_shared_memory(
        SharedMemorySnapshot(
            event_id=DEMO_EVENT_ID,
            autonomy_level=AutonomyLevel.HUMAN_GATE_REQUIRED,
            active_agents=active_agents,
            focus_entity_ids=["community_jsl_grid", "hospital_bl_central", "nursing_home_hpm", "metro_nsm_hub"],
            focus_entity_names=["建设路低洼网格三组", "碑林中心医院", "和平门医养中心", "南门地铁换乘枢纽"],
            top_risks=[
                "建设路低洼网格地下室和老院落 10 分钟内可能出现明显回流水。",
                "碑林中心医院急诊与后勤入口需要保持通行和供电连续性。",
                "医养中心卧床老人转移需要提前组织辅助车辆。",
            ],
            recommended_actions=[
                "批准医院急诊与后勤入口保障动作。",
                "审核学校北门分流 proposal，并同步交警与社区通知。",
                "面向社区、部门和公众生成分众 warning。",
            ],
            pending_proposal_ids=["proposal_demo_pending_school"],
            recent_result_ids=result_ids,
            unresolved_items=["学校家长接送车辆绕行路线仍需现场确认"],
            active_decision_path=["ImpactAgent", "ActionAgent", "AuditAgent", "Orchestrator", "Human Gate"],
            open_questions=["学校北侧道路是否允许单向临时管制？"],
            blocked_by=["社区网格协同转移动作必须经过指挥员审批"],
            latest_hazard_level=RiskLevel.RED,
            latest_summary="Agent council 已完成影响研判、动作建议和审计边界合成。",
            last_trigger="production_demo_seed",
            updated_at=now - timedelta(minutes=4),
        )
    )
    repo.save_v2_supervisor_run(
        SupervisorRunRecord(
            supervisor_run_id="supervisor_demo_primary",
            event_id=DEMO_EVENT_ID,
            trigger_type="manual_run",
            autonomy_level=AutonomyLevel.HUMAN_GATE_REQUIRED,
            status=SupervisorRunStatus.COMPLETED,
            summary="演示主事件会商已完成，等待人工审批与 warning 生成。",
            created_tasks=[f"task_demo_{agent.value}" for agent in active_agents],
            completed_task_ids=[f"task_demo_{agent.value}" for agent in active_agents],
            created_at=now - timedelta(minutes=31),
            completed_at=now - timedelta(minutes=24),
        )
    )
    repo.save_supervisor_health_state(
        SupervisorHealthState(
            running=False,
            interval_seconds=60.0,
            consecutive_failures=0,
            retries_used_in_last_cycle=0,
            skipped_sweeps=0,
            last_started_at=now - timedelta(minutes=31),
            last_success_at=now - timedelta(minutes=24),
            last_completed_at=now - timedelta(minutes=24),
            pending_trigger_count=0,
            last_trigger_processed_at=now - timedelta(minutes=24),
            updated_at=now - timedelta(minutes=2),
        )
    )
    return result_ids


def seed_closure(system: FloodWarningSystem, now: datetime) -> None:
    repo = system.repository
    proposals = build_proposals(now)
    for proposal in proposals:
        repo.save_v2_action_proposal(proposal)
    approved = proposals[1]
    drafts = [
        NotificationDraft(
            draft_id="draft_demo_leadership_hospital",
            event_id=DEMO_EVENT_ID,
            proposal_id=approved.proposal_id,
            entity_id=approved.entity_id,
            area_id=AREA_ID,
            audience="leadership",
            channel="briefing",
            content="建议批准医院急诊和后勤入口保障动作，优先调度排水泵、救护车与交警协同。",
            grounding_summary=approved.grounding_summary,
            created_at=now - timedelta(minutes=6),
        ),
        NotificationDraft(
            draft_id="draft_demo_department_hospital",
            event_id=DEMO_EVENT_ID,
            proposal_id=approved.proposal_id,
            entity_id=approved.entity_id,
            area_id=AREA_ID,
            audience="department",
            channel="dashboard",
            content="请城管、交警、卫健部门按医院保障方案到位，保持急诊入口和后勤入口通行。",
            grounding_summary=approved.grounding_summary,
            created_at=now - timedelta(minutes=5),
        ),
        NotificationDraft(
            draft_id="draft_demo_community_hospital",
            event_id=DEMO_EVENT_ID,
            proposal_id=approved.proposal_id,
            entity_id=approved.entity_id,
            area_id=AREA_ID,
            audience="community",
            channel="sms",
            content="医院周边道路实施临时保障，请居民避开急诊通道，服从现场疏导。",
            grounding_summary=approved.grounding_summary,
            created_at=now - timedelta(minutes=4),
        ),
        NotificationDraft(
            draft_id="draft_demo_public_hospital",
            event_id=DEMO_EVENT_ID,
            proposal_id=approved.proposal_id,
            entity_id=approved.entity_id,
            area_id=AREA_ID,
            audience="public",
            channel="public_notice",
            content="受强降雨影响，碑林中心医院周边部分道路实施临时交通组织，请提前绕行。",
            grounding_summary=approved.grounding_summary,
            created_at=now - timedelta(minutes=3),
        ),
    ]
    for draft in drafts:
        repo.save_v2_notification_draft(draft)
        repo.save_v3_audience_warning(AudienceWarningDraft.from_notification_draft(draft))
    logs = [
        ExecutionLogEntry(
            log_id="exec_demo_hospital_pump",
            event_id=DEMO_EVENT_ID,
            proposal_id=approved.proposal_id,
            entity_id=approved.entity_id,
            area_id=AREA_ID,
            action_type="resource_assigned",
            summary="2 台移动排水泵已调度至医院后勤入口。",
            operator_id="demo_operations",
            details={"resource": "portable_pumps", "count": 2, "eta_minutes": 12},
            grounding_summary=approved.grounding_summary,
            created_at=now - timedelta(minutes=3),
        ),
        ExecutionLogEntry(
            log_id="exec_demo_hospital_route",
            event_id=DEMO_EVENT_ID,
            proposal_id=approved.proposal_id,
            entity_id=approved.entity_id,
            area_id=AREA_ID,
            action_type="route_confirmed",
            summary="急诊入口通行保障路线已确认，交警已接单。",
            operator_id="demo_traffic",
            details={"route": "东大街至李家村通道", "status": "protected"},
            grounding_summary=approved.grounding_summary,
            created_at=now - timedelta(minutes=2),
        ),
    ]
    for log in logs:
        repo.save_v2_execution_log(log)
    audits = [
        AuditRecord(
            audit_id="audit_demo_human_gate",
            source_type="v3_audit_agent",
            action="human_gate_required",
            summary="建设路低洼网格协同转移 proposal 保持人工审批闸门。",
            details={"proposal_id": "proposal_demo_pending_school", "reason": "community_grid_backflow_control"},
            severity=AlertSeverity.WARNING,
            event_id=DEMO_EVENT_ID,
            created_at=now - timedelta(minutes=10),
        ),
        AuditRecord(
            audit_id="audit_demo_warning_generated",
            source_type="v3_warning_center",
            action="warnings_generated",
            summary="已为医院保障 proposal 生成 4 条分众 warning。",
            details={"proposal_id": approved.proposal_id, "warning_count": 4},
            severity=AlertSeverity.INFO,
            event_id=DEMO_EVENT_ID,
            created_at=now - timedelta(minutes=2),
        ),
    ]
    for audit in audits:
        repo.add_audit_record(audit)
    for event_type, payload in [
        ("hazard_updated", {"source": "demo_seed", "risk_level": "Red"}),
        ("plan_proposed", {"proposal_ids": [item.proposal_id for item in proposals]}),
        ("approval_resolved", {"proposal_id": approved.proposal_id, "status": "approved"}),
        ("notification_sent", {"proposal_id": approved.proposal_id, "warning_count": 4}),
    ]:
        repo.add_v2_stream_record_for_payload(DEMO_EVENT_ID, event_type, payload)


def rebuild_demo_db(db_path: Path, *, force: bool) -> dict[str, object]:
    backup_path = prepare_db_file(db_path, force=force)
    now = utc_now()
    system = FloodWarningSystem(db_path)
    repo = system.repository
    repo.save_v2_event(build_primary_event(now))
    repo.save_v2_event(build_history_event(now))
    repo.save_v2_hazard_state(build_hazard_state(DEMO_EVENT_ID, now))
    repo.save_v2_hazard_state(build_hazard_state(HISTORY_EVENT_ID, now, closed=True))
    repo.save_event_resource_status(
        DEMO_EVENT_ID,
        ResourceStatus(
            area_id=AREA_ID,
            vehicle_count=22,
            staff_count=76,
            supply_kits=520,
            rescue_boats=3,
            ambulance_count=8,
            drone_count=5,
            portable_pumps=12,
            power_generators=8,
            medical_staff_count=34,
            volunteer_count=138,
            satellite_phones=14,
            notes="演示主事件资源状态：医院、学校和社区闭环场景可直接使用。",
        ),
    )
    seed_agent_council(system, now)
    seed_closure(system, now)

    overview = system.agent_twin.get_twin_overview(DEMO_EVENT_ID)
    return {
        "db_path": str(db_path),
        "backup_path": str(backup_path) if backup_path else None,
        "event_id": DEMO_EVENT_ID,
        "focus_object_count": len(overview.focus_objects),
        "map_layer_count": len(overview.map_layers),
        "pending_proposal_count": overview.pending_proposal_count,
        "approved_proposal_count": overview.approved_proposal_count,
        "warning_draft_count": overview.warning_draft_count,
        "generated_at": now.isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a clean AgentTwin demo database.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Target demo database path.")
    parser.add_argument("--force", action="store_true", help="Backup and replace an existing demo database.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = rebuild_demo_db(Path(args.db_path).resolve(), force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
