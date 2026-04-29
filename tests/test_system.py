from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from fastapi.testclient import TestClient

from flood_system import api as api_module
from flood_system.models import ResourceStatus, RiskLevel
from flood_system.system import FloodWarningSystem
from flood_system.v2.exposure_engine import ExposureEngine
from flood_system.v2.llm_gateway import LLMGenerationError, MockLLMGateway
from flood_system.v2.models import (
    AdvisoryRequest,
    EntityProfile,
    EntityType,
    EventCreateRequest,
    HazardState,
    HazardTile,
    LLMErrorCode,
    LongTermMemoryRecord,
    ObservationBatchRequest,
    ObservationIngestItem,
    ProposalResolutionRequest,
    RegionalAnalysisPackageStatus,
    SimulationCell,
    SimulationUpdateRequest,
    ProposalStatus,
    ReplayRequest,
    ToolFailureMode,
    TriggerEventType,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
)
from flood_system.v2.reporting import format_daily_report_message
from flood_system.v2.routing import RoutePlanningService


def build_system(tmp_path: Path) -> FloodWarningSystem:
    return FloodWarningSystem(tmp_path / "system.db", llm_gateway=MockLLMGateway())


class AlwaysFailLLMGateway:
    model_name = "failing-llm"

    @staticmethod
    def _fail(stage: str):
        raise LLMGenerationError(LLMErrorCode.UNAVAILABLE, f"{stage} unavailable in test gateway")

    def generate_object_advisory(self, payload):
        self._fail("object_advisory")

    def generate_copilot_chat(self, payload):
        self._fail("copilot_chat")

    def generate_regional_decision(self, payload):
        self._fail("regional_decision")

    def generate_proposal_draft(self, payload):
        self._fail("proposal_draft")

    def generate_regional_analysis_package(self, payload):
        self._fail("regional_analysis_package")

    def generate_execution_bundle(self, payload):
        self._fail("execution_bundle")

    def generate_execution_summary(self, payload):
        self._fail("execution_summary")

    def generate_daily_operations_summary(self, payload):
        self._fail("daily_operations_summary")

    def generate_high_risk_postmortem_summary(self, payload):
        self._fail("high_risk_postmortem")


def sample_observations(now: datetime | None = None) -> list[ObservationIngestItem]:
    now = now or datetime.now(timezone.utc)
    return [
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=22),
            source_type="monitoring_point",
            source_name="West flood point",
            village="Lianshi Village",
            rainfall_mm=26,
            water_level_m=3.7,
            citizen_reports=2,
        ),
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=12),
            source_type="water_level_sensor",
            source_name="School gate sensor",
            village="Wuyuanli Village",
            rainfall_mm=34,
            water_level_m=4.2,
            road_blocked=True,
            citizen_reports=4,
            notes="School access road is starting to pond.",
        ),
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=6),
            source_type="camera_alert",
            source_name="Factory loading bay camera",
            village="Wuyuanli Village",
            rainfall_mm=31,
            water_level_m=4.5,
            road_blocked=True,
            citizen_reports=5,
            notes="Loading bay runoff is increasing.",
        ),
    ]


def sample_simulation_update(now: datetime | None = None) -> SimulationUpdateRequest:
    now = now or datetime.now(timezone.utc)
    return SimulationUpdateRequest(
        generated_at=now,
        depth_threshold_m=0.45,
        flow_threshold_mps=1.2,
        cells=[
            SimulationCell(cell_id="grid_01", label="School cluster", water_depth_m=1.1, flow_velocity_mps=1.5),
            SimulationCell(cell_id="grid_02", label="Factory edge", water_depth_m=0.9, flow_velocity_mps=1.4),
            SimulationCell(cell_id="grid_03", label="Residential lowland", water_depth_m=1.3, flow_velocity_mps=1.7),
            SimulationCell(cell_id="grid_04", label="Transit corridor", water_depth_m=0.7, flow_velocity_mps=1.3),
        ],
    )


def wait_for_agent_processing(system: FloodWarningSystem, event_id: str, timeout: float = 2.5) -> None:
    system.supervisor_loop.trigger_poll_seconds = 0.02
    system.start_background_services()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            triggers = system.production_platform.list_trigger_events(event_id)
            tasks = system.production_platform.list_agent_tasks(event_id)
            if tasks and all(item.status.value in {"completed", "failed"} for item in tasks[:2]):
                return
            if triggers and all(item.status.value in {"processed", "failed"} for item in triggers):
                return
            time.sleep(0.05)
        raise AssertionError("agent processing did not complete in time.")
    finally:
        system.stop_background_services()


def seed_event(system: FloodWarningSystem) -> str:
    production = system.production_platform
    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="Beilin severe rainfall event",
            trigger_reason="test_seed",
            operator="pytest",
        )
    )
    production.ingest_observations(
        event.event_id,
        ObservationBatchRequest(operator="pytest", observations=sample_observations()),
    )
    wait_for_agent_processing(system, event.event_id)
    return event.event_id


@contextmanager
def bound_test_client(system: FloodWarningSystem):
    original_system = api_module.system
    original_production = api_module.production
    api_module.system = system
    api_module.production = system.production_platform
    try:
        with TestClient(api_module.app) as client:
            yield client
    finally:
        api_module.system = original_system
        api_module.production = original_production


def test_platform_event_ingestion_recomputes_hazard_and_exposure(tmp_path: Path):
    system = build_system(tmp_path)
    production = system.production_platform

    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="Beilin severe rainfall event",
            trigger_reason="test_seed",
            operator="pytest",
        )
    )
    snapshot = production.ingest_observations(
        event.event_id,
        ObservationBatchRequest(operator="pytest", observations=sample_observations()),
    )

    assert snapshot.event.event_id == event.event_id
    assert snapshot.latest_hazard_state is not None
    assert snapshot.latest_hazard_state.overall_risk_level.value in {"Yellow", "Orange", "Red"}
    assert snapshot.latest_exposure_summary is not None
    assert snapshot.latest_exposure_summary.affected_entities
    assert {record.event_type.value for record in snapshot.recent_stream} >= {
        "observation_ingested",
        "hazard_updated",
        "impact_recomputed",
    }
    triggers = production.list_trigger_events(event.event_id)
    assert triggers
    assert triggers[0].trigger_type.value == "observation_ingested"
    wait_for_agent_processing(system, event.event_id)
    shared_memory = production.get_shared_memory_snapshot(event.event_id)
    assert shared_memory.active_agents
    assert shared_memory.latest_summary
    assert production.list_agent_tasks(event.event_id)
    assert production.list_supervisor_runs(event.event_id)


def test_platform_entity_impact_and_advisory_require_confirmation_for_school(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    impact = production.get_entity_impact("school_wyl_primary", event_id=event_id)
    advisory = production.generate_advisory(
        AdvisoryRequest(
            event_id=event_id,
            area_id="beilin_10km2",
            entity_id="school_wyl_primary",
            operator_role="commander",
        )
    )

    assert impact.entity.entity_id == "school_wyl_primary"
    assert impact.entity.entity_type.value == "school"
    assert impact.risk_reason
    assert advisory.answer
    assert advisory.evidence
    assert advisory.proposal is None
    assert advisory.requires_human_confirmation is True
    assert advisory.generation_source.value == "llm"
    assert advisory.model_name == "mock-llm"
    assert advisory.grounding_summary


def test_exposure_uses_selected_tile_name_instead_of_entity_village(tmp_path: Path):
    system = build_system(tmp_path)
    area_profile = system.production_platform.area_profiles["beilin_10km2"]
    exposure = ExposureEngine(RoutePlanningService())
    entity = EntityProfile(
        entity_id="entity_beilin_name_guard",
        area_id="beilin_10km2",
        entity_type=EntityType.RESIDENT,
        name="李家村测试住户",
        village="柏树林街道",
        location_hint="李家村北侧院落",
        vulnerability_tags=[],
        mobility_constraints=[],
        key_assets=[],
        notification_preferences=[],
        emergency_contacts=[],
        custom_attributes={},
    )
    hazard_state = HazardState(
        event_id="event_test",
        area_id="beilin_10km2",
        generated_at=datetime.now(timezone.utc),
        overall_risk_level=RiskLevel.RED,
        overall_score=92.0,
        trend="rising",
        uncertainty=0.12,
        freshness_seconds=0,
        hazard_tiles=[
            HazardTile(
                tile_id="tile_beilin",
                area_name="碑林区",
                horizon_minutes=10,
                risk_level=RiskLevel.RED,
                risk_score=92.0,
                predicted_water_depth_cm=74.0,
                trend="rising",
                uncertainty=0.12,
                affected_roads=[],
            )
        ],
        road_reachability=[],
        monitoring_points=[],
    )
    resource_status = ResourceStatus(
        area_id="beilin_10km2",
        vehicle_count=10,
        staff_count=20,
        supply_kits=50,
        rescue_boats=1,
        ambulance_count=1,
        drone_count=1,
        portable_pumps=2,
        power_generators=4,
        medical_staff_count=20,
        volunteer_count=30,
        satellite_phones=4,
        notes="pytest",
    )

    impact = exposure.assess_entity(
        "event_test",
        entity,
        area_profile,
        hazard_state,
        resource_status,
        evidence=[],
    )

    assert impact.risk_reason[0] == "碑林区 10分钟风险栅格评分为 92，预测积水深度约 74 厘米。"
    assert impact.evidence[0].title == "碑林区 实时风险栅格"


def test_platform_location_based_advisory_supports_profile_overrides(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    advisory = production.generate_advisory(
        AdvisoryRequest(
            event_id=event_id,
            area_id="beilin_10km2",
            location_hint="North lane courtyard",
            village="Lianshi Village",
            operator_role="district_operator",
            profile_overrides={
                "entity_type": "resident",
                "name": "Resident Wang",
                "resident_count": 3,
                "vulnerability_tags": ["elderly", "limited_mobility"],
                "mobility_constraints": ["needs_assistance"],
            },
        )
    )

    assert advisory.entity_id == "ad_hoc_location"
    assert advisory.answer
    assert advisory.impact_summary
    assert advisory.recommended_actions
    assert advisory.generation_source.value == "llm"
    assert advisory.model_name == "mock-llm"


def test_platform_tool_registry_exposes_schema_timeout_and_failure_modes(tmp_path: Path):
    system = build_system(tmp_path)
    registry = system.production_platform.tools

    specs = {spec.tool_name: spec for spec in registry.list_specs()}
    assert "get_hazard_tiles" in specs
    assert "create_action_proposal" in specs
    assert specs["get_hazard_tiles"].timeout_ms > 0
    assert specs["get_entity_profile"].input_schema["required"] == ["entity_id"]
    assert ToolFailureMode.INVALID_INPUT in specs["get_route_options"].failure_modes

    invalid = registry.execute("get_entity_profile")
    assert invalid.status.value == "failed"
    assert invalid.failure_reason is not None
    assert invalid.failure_reason.startswith(ToolFailureMode.INVALID_INPUT.value)


def test_platform_copilot_returns_explainable_plan_execution_and_proposal(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event_id, operator_role="commander")
    )
    view = production.send_copilot_message(
        session.session_id,
        V2CopilotMessageRequest(content="What does this mean for the school right now?"),
    )

    assert view.latest_answer is not None
    assert view.latest_answer.evidence
    assert view.latest_answer.tool_trace
    assert view.latest_answer.planner_summary
    assert view.latest_answer.tool_selection_reasoning
    assert view.latest_answer.skipped_tools
    assert view.latest_answer.tool_executions
    assert view.latest_answer.data_freshness.hazard_state_freshness_seconds is not None
    assert isinstance(view.latest_answer.evidence_gaps, list)
    assert view.latest_answer.plan_runs
    assert view.latest_answer.memory_snapshot is not None
    assert view.latest_answer.generation_source.value == "llm"
    assert view.latest_answer.model_name == "mock-llm"
    assert view.latest_answer.grounding_summary
    assert view.latest_answer.follow_up_prompts
    assert view.memory_snapshot is not None
    assert view.plan_runs
    assert view.recent_tool_executions

    tool_names = [item.tool_name for item in view.latest_answer.tool_executions]
    assert "resolve_target_entity" in tool_names
    assert "get_hazard_tiles" in tool_names
    assert "synthesize_entity_impact" in tool_names
    assert "get_policy_constraints" in tool_names
    assert any(
        item.tool_name in {"draft_action_proposal", "create_action_proposal"}
        for item in view.latest_answer.tool_executions
    )
    assert any(item.tool_name == "get_route_options" for item in view.latest_answer.skipped_tools)

    school_proposal = next((item for item in view.proposals if item.entity_id == "school_wyl_primary"), None)
    assert school_proposal is None
    assert view.latest_answer.proposal is None


def test_platform_route_question_uses_route_traffic_and_shelter_tools(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event_id, operator_role="commander")
    )
    view = production.send_copilot_message(
        session.session_id,
        V2CopilotMessageRequest(content="Which route should the nursing home use to reach the safest shelter?"),
    )

    assert view.latest_answer is not None
    execution_names = [item.tool_name for item in view.latest_answer.tool_executions]
    assert "resolve_target_entity" in execution_names
    assert "get_route_options" in execution_names
    assert "get_live_traffic" in execution_names
    assert "get_shelter_capacity" in execution_names


def test_platform_regional_approval_generates_notification_drafts_and_execution_logs(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    result = production.ingest_simulation_update(event_id, sample_simulation_update())
    assert result["overall_risk_level"].value in {"Orange", "Red"}

    pending = production.list_regional_proposals(event_id, statuses=[ProposalStatus.PENDING.value])
    proposal = next(item.proposal for item in pending if item.proposal.action_type == "regional_notification")
    assert proposal.action_display_name
    assert proposal.action_display_tagline
    assert proposal.action_display_category
    assert proposal.chat_follow_up_prompt

    approved = production.approve_regional_proposal(
        proposal.proposal_id,
        ProposalResolutionRequest(
            operator_id="shift_commander",
            operator_role="commander",
            note="Push the district-wide notification draft first.",
        ),
    )

    assert approved.proposal.status == ProposalStatus.APPROVED
    assert approved.proposal.resolved_by == "shift_commander"
    assert approved.proposal.generation_source.value == "llm"
    drafts = production.repository.list_v2_notification_drafts(event_id)
    logs = production.repository.list_v2_execution_logs(event_id)
    assert drafts
    assert logs
    assert all(item.generation_source.value == "llm" for item in drafts)
    assert all(item.generation_source.value == "llm" for item in logs)


def test_platform_red_simulation_can_generate_generic_llm_action(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    production.ingest_simulation_update(
        event_id,
        SimulationUpdateRequest(
            generated_at=datetime.now(timezone.utc),
            depth_threshold_m=0.35,
            flow_threshold_mps=0.8,
            cells=[
                SimulationCell(cell_id="grid_a", label="North corridor", water_depth_m=1.9, flow_velocity_mps=2.1),
                SimulationCell(cell_id="grid_b", label="School edge", water_depth_m=1.8, flow_velocity_mps=2.0),
                SimulationCell(cell_id="grid_c", label="Transit hub", water_depth_m=1.7, flow_velocity_mps=1.9),
                SimulationCell(cell_id="grid_d", label="Residential lowland", water_depth_m=2.0, flow_velocity_mps=2.2),
            ],
        ),
    )

    pending = production.list_regional_proposals(event_id, statuses=[ProposalStatus.PENDING.value])
    generic = next((item for item in pending if item.proposal.action_type == "traffic_control"), None)
    assert generic is not None
    assert generic.proposal.execution_mode.value == "generic_task"
    assert generic.proposal.generation_source.value == "llm"
    assert generic.proposal.action_display_name
    assert generic.proposal.action_display_tagline
    assert generic.proposal.action_display_category


def test_platform_simulation_update_builds_regional_analysis_package(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    result = production.ingest_simulation_update(event_id, sample_simulation_update())
    package = production.get_pending_regional_analysis_package(event_id)

    assert package is not None
    assert package.package_id == result["risk_stage_key"]
    assert package.trigger_type == "simulation_updated"
    assert package.status == RegionalAnalysisPackageStatus.PENDING
    assert package.proposal_count == len(package.proposal_ids)
    assert package.analysis_message
    assert package.risk_assessment
    assert package.rescue_plan
    assert package.resource_dispatch_plan

    pending = production.list_regional_proposals(event_id, statuses=[ProposalStatus.PENDING.value])
    assert set(package.proposal_ids) == {item.proposal.proposal_id for item in pending}


def test_platform_regional_analysis_package_approve_and_reject(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    production.ingest_simulation_update(event_id, sample_simulation_update())
    package = production.get_pending_regional_analysis_package(event_id)
    assert package is not None

    approved = production.approve_regional_analysis_package(
        package.package_id,
        ProposalResolutionRequest(
            operator_id="shift_commander",
            operator_role="commander",
            note="Approve the full regional package.",
        ),
    )
    assert approved.status == RegionalAnalysisPackageStatus.APPROVED
    approved_items = [
        item
        for item in production.list_regional_proposals(event_id)
        if item.proposal.risk_stage_key == package.package_id and item.proposal.proposal_id in approved.proposal_ids
    ]
    assert approved_items
    assert all(item.proposal.status == ProposalStatus.APPROVED for item in approved_items)

    event_id_reject = seed_event(system)
    production.ingest_simulation_update(event_id_reject, sample_simulation_update())
    reject_package = production.get_pending_regional_analysis_package(event_id_reject)
    assert reject_package is not None

    rejected = production.reject_regional_analysis_package(
        reject_package.package_id,
        ProposalResolutionRequest(
            operator_id="shift_commander",
            operator_role="commander",
            note="Reject the full regional package.",
        ),
    )
    assert rejected.status == RegionalAnalysisPackageStatus.REJECTED
    rejected_items = [
        item
        for item in production.list_regional_proposals(event_id_reject)
        if item.proposal.risk_stage_key == reject_package.package_id and item.proposal.proposal_id in rejected.proposal_ids
    ]
    assert rejected_items
    assert all(item.proposal.status == ProposalStatus.REJECTED for item in rejected_items)


def test_platform_new_risk_stage_creates_new_package_and_preserves_history(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    production.ingest_simulation_update(event_id, sample_simulation_update())
    first_package = production.get_pending_regional_analysis_package(event_id)
    assert first_package is not None
    production.approve_regional_analysis_package(
        first_package.package_id,
        ProposalResolutionRequest(operator_id="shift_commander", operator_role="commander", note="Approve stage one."),
    )

    production.ingest_simulation_update(
        event_id,
        SimulationUpdateRequest(
            generated_at=datetime.now(timezone.utc),
            depth_threshold_m=0.35,
            flow_threshold_mps=0.8,
            cells=[
                SimulationCell(cell_id="grid_a", label="North corridor", water_depth_m=1.9, flow_velocity_mps=2.1),
                SimulationCell(cell_id="grid_b", label="School edge", water_depth_m=1.8, flow_velocity_mps=2.0),
                SimulationCell(cell_id="grid_c", label="Transit hub", water_depth_m=1.7, flow_velocity_mps=1.9),
                SimulationCell(cell_id="grid_d", label="Residential lowland", water_depth_m=2.0, flow_velocity_mps=2.2),
            ],
        ),
    )
    current_package = production.get_pending_regional_analysis_package(event_id)
    history = production.list_regional_analysis_packages(event_id, include_pending=False)

    assert current_package is not None
    assert current_package.package_id != first_package.package_id
    assert any(item.package_id == first_package.package_id for item in history)


def test_daily_summary_service_generates_deduped_reports_and_delivers_to_sessions(tmp_path: Path):
    system = build_system(tmp_path)
    production = system.production_platform
    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="Daily summary delivery event",
            trigger_reason="test_daily_summary",
            operator="pytest",
        )
    )
    observation_time = datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)
    production.ingest_observations(
        event.event_id,
        ObservationBatchRequest(operator="pytest", observations=sample_observations(observation_time)),
    )
    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event.event_id, operator_role="commander")
    )

    reports = system.daily_summary_service.run_once(now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc))
    assert len(reports) == 1
    report = reports[0]
    assert report.event_id == event.event_id
    assert session.session_id in report.delivered_session_ids
    assert system.repository.get_v2_daily_report_run(event.event_id, "2026-04-14") is not None
    assert len(production.list_daily_reports(event.event_id)) == 1

    second = system.daily_summary_service.run_once(now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc))
    assert second == []
    session_view = production.get_copilot_session(session.session_id)
    assert session_view.daily_reports
    assert session_view.daily_reports[0].report_id == report.report_id
    assert any(report.headline in item.content for item in session_view.messages)


def test_daily_summary_service_skips_events_without_previous_day_activity(tmp_path: Path):
    system = build_system(tmp_path)
    production = system.production_platform
    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="No activity event",
            trigger_reason="test_daily_summary_idle",
            operator="pytest",
        )
    )

    reports = system.daily_summary_service.run_once(now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc))
    assert reports == []
    assert production.list_daily_reports(event.event_id) == []


def test_postmortem_service_creates_single_summary_and_long_term_memory(tmp_path: Path):
    system = build_system(tmp_path)
    production = system.production_platform
    event_id = seed_event(system)
    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event_id, operator_role="commander")
    )

    production.ingest_simulation_update(
        event_id,
        sample_simulation_update(datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)),
    )
    open_episodes = system.repository.list_open_v2_high_risk_episodes(event_id)
    assert len(open_episodes) == 1

    system.event_postmortem_service.sync_risk_transition(
        event=production.get_event(event_id),
        previous_risk_level=open_episodes[0].peak_risk_level,
        current_risk_level=RiskLevel.YELLOW,
        trigger_source="pytest_close",
        observed_at=datetime(2026, 4, 15, 2, 0, tzinfo=timezone.utc),
    )
    summaries = system.event_postmortem_service.run_once()
    assert len(summaries) == 1
    summary = summaries[0]

    all_episodes = system.repository.list_v2_high_risk_episodes(event_id)
    assert len(all_episodes) == 1
    assert all_episodes[0].status.value == "summarized"
    assert len(production.list_episode_summaries(event_id)) == 1

    memories = production.list_long_term_memories(event_id)
    assert len(memories) == 1
    assert memories[0].source_summary_id == summary.summary_id
    cross_event_memories = production.long_term_memory_store.query_memories(
        area_id="beilin_10km2",
        risk_level=memories[0].risk_level,
    )
    assert any(item.event_id == event_id for item in cross_event_memories)

    session_view = production.get_copilot_session(session.session_id)
    assert session_view.episode_summaries
    assert session_view.episode_summaries[0].summary_id == summary.summary_id
    assert any(summary.headline in item.content for item in session_view.messages)
    assert any("一、风险升级路径" in item.content for item in session_view.messages)


def test_long_term_memory_query_prioritizes_area_risk_and_entity_matches(tmp_path: Path):
    system = build_system(tmp_path)
    now = datetime.now(timezone.utc)

    stronger = LongTermMemoryRecord(
        memory_id="ltm_stronger",
        event_id="event_stronger",
        source_summary_id="summary_stronger",
        memory_type="postmortem",
        area_id="beilin_10km2",
        risk_level=RiskLevel.RED,
        entity_types=["school"],
        action_types=["evacuation"],
        tags=["school", "night_shift"],
        headline="School evacuation pattern",
        summary="Prioritize school evacuation approval path.",
        retrieval_text="School evacuation pattern with commander approval path.",
        lessons=["Escalate early for school evacuation."],
        pitfalls=["Avoid delaying transport dispatch."],
        recommendations=["Lock vehicle allocation before notification."],
        created_at=now - timedelta(days=2),
    )
    weaker = LongTermMemoryRecord(
        memory_id="ltm_weaker",
        event_id="event_weaker",
        source_summary_id="summary_weaker",
        memory_type="postmortem",
        area_id="other_area",
        risk_level=RiskLevel.YELLOW,
        entity_types=["factory"],
        action_types=["traffic_control"],
        tags=["traffic"],
        headline="Traffic control pattern",
        summary="Use traffic control for industrial corridors.",
        retrieval_text="Traffic control pattern for factory corridor.",
        lessons=["Coordinate road closure with police."],
        pitfalls=["Do not delay public notice."],
        recommendations=["Re-check detour capacity."],
        created_at=now - timedelta(hours=4),
    )

    system.production_platform.long_term_memory_store.save_memory(stronger)
    system.production_platform.long_term_memory_store.save_memory(weaker)

    ranked = system.production_platform.long_term_memory_store.query_memories(
        area_id="beilin_10km2",
        entity_type="school",
        risk_level=RiskLevel.RED,
        action_type="evacuation",
        tags=["night_shift"],
        top_k=2,
    )

    assert [item.memory_id for item in ranked][:1] == ["ltm_stronger"]


def test_daily_summary_and_postmortem_fallback_without_llm(tmp_path: Path):
    system = FloodWarningSystem(tmp_path / "system.db", llm_gateway=AlwaysFailLLMGateway())
    production = system.production_platform
    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="Fallback summaries event",
            trigger_reason="test_summary_fallback",
            operator="pytest",
        )
    )
    production.ingest_observations(
        event.event_id,
        ObservationBatchRequest(
            operator="pytest",
            observations=sample_observations(datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)),
        ),
    )

    reports = system.daily_summary_service.run_once(now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc))
    assert len(reports) == 1
    assert reports[0].generation_source.value == "system"
    assert "一、态势概述" in format_daily_report_message(reports[0])

    system.event_postmortem_service.sync_risk_transition(
        event=production.get_event(event.event_id),
        previous_risk_level=RiskLevel.YELLOW,
        current_risk_level=RiskLevel.ORANGE,
        trigger_source="pytest_open",
        observed_at=datetime(2026, 4, 15, 3, 0, tzinfo=timezone.utc),
    )
    system.event_postmortem_service.sync_risk_transition(
        event=production.get_event(event.event_id),
        previous_risk_level=RiskLevel.ORANGE,
        current_risk_level=RiskLevel.YELLOW,
        trigger_source="pytest_close",
        observed_at=datetime(2026, 4, 15, 4, 0, tzinfo=timezone.utc),
    )
    summaries = system.event_postmortem_service.run_once()
    assert len(summaries) == 1
    assert summaries[0].headline
    assert production.list_long_term_memories(event.event_id)


def test_platform_api_exposes_reports_postmortems_and_long_term_memory(tmp_path: Path):
    system = build_system(tmp_path)
    production = system.production_platform
    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="API reporting event",
            trigger_reason="test_reporting_api",
            operator="pytest",
        )
    )
    production.ingest_observations(
        event.event_id,
        ObservationBatchRequest(
            operator="pytest",
            observations=sample_observations(datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)),
        ),
    )
    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event.event_id, operator_role="commander")
    )
    system.daily_summary_service.run_once(now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc))
    production.ingest_simulation_update(
        event.event_id,
        sample_simulation_update(datetime(2026, 4, 15, 0, 30, tzinfo=timezone.utc)),
    )
    open_episodes = system.repository.list_open_v2_high_risk_episodes(event.event_id)
    assert open_episodes
    system.event_postmortem_service.sync_risk_transition(
        event=production.get_event(event.event_id),
        previous_risk_level=open_episodes[0].peak_risk_level,
        current_risk_level=RiskLevel.YELLOW,
        trigger_source="pytest_close",
        observed_at=datetime(2026, 4, 15, 2, 0, tzinfo=timezone.utc),
    )
    system.event_postmortem_service.run_once()

    with bound_test_client(system) as client:
        session_response = client.get(f"/platform/copilot/sessions/{session.session_id}")
        assert session_response.status_code == 200
        assert session_response.json()["daily_reports"]
        assert session_response.json()["episode_summaries"]

        daily_reports = client.get(f"/platform/events/{event.event_id}/daily-reports")
        assert daily_reports.status_code == 200
        assert len(daily_reports.json()) == 1

        episode_summaries = client.get(f"/platform/events/{event.event_id}/episode-summaries")
        assert episode_summaries.status_code == 200
        assert len(episode_summaries.json()) == 1

        long_term_memory = client.get(f"/platform/events/{event.event_id}/long-term-memory")
        assert long_term_memory.status_code == 200
        assert len(long_term_memory.json()) == 1

        experience_context = client.get(f"/platform/events/{event.event_id}/experience-context")
        assert experience_context.status_code == 200
        assert "long_term_memories" in experience_context.json()


def test_platform_api_only_exposes_new_flow_and_supports_regional_proposals(tmp_path: Path):
    system = build_system(tmp_path)

    with bound_test_client(system) as client:
        status_response = client.get("/platform/supervisor/status")
        assert status_response.status_code == 200
        assert status_response.json()["running"] is True
        assert "pending_trigger_count" in status_response.json()

        legacy = client.post("/incidents/ingest", json={})
        assert legacy.status_code == 404

        event_response = client.post(
            "/platform/events",
            json={
                "area_id": "beilin_10km2",
                "title": "API event",
                "trigger_reason": "api_test",
                "operator": "pytest",
            },
        )
        assert event_response.status_code == 200
        event_id = event_response.json()["event_id"]

        ingest_response = client.post(
            f"/platform/events/{event_id}/observations",
            json={
                "operator": "pytest",
                "observations": [item.model_dump(mode="json") for item in sample_observations()],
            },
        )
        assert ingest_response.status_code == 200
        wait_for_agent_processing(system, event_id)

        session_response = client.post(
            "/platform/copilot/sessions/bootstrap",
            json={"event_id": event_id, "operator_role": "commander"},
        )
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        first_reply = client.post(
            f"/platform/copilot/sessions/{session_id}/messages",
            json={"content": "What does this mean for the school right now?"},
        )
        assert first_reply.status_code == 200
        assert first_reply.json()["latest_answer"]["planner_summary"]
        assert first_reply.json()["latest_answer"]["tool_executions"]
        assert first_reply.json()["latest_answer"]["plan_runs"]
        assert first_reply.json()["memory_snapshot"]
        assert first_reply.json()["recent_tool_executions"]
        assert first_reply.json()["shared_memory_snapshot"]
        assert first_reply.json()["active_agents"]
        assert first_reply.json()["recent_agent_results"]
        assert first_reply.json()["autonomy_level"]

        agent_status = client.get(f"/platform/events/{event_id}/agent-status")
        assert agent_status.status_code == 200
        assert agent_status.json()["completed_task_count"] >= 1
        assert "active_decision_path" in agent_status.json()
        assert "blocked_by" in agent_status.json()

        agent_tasks = client.get(f"/platform/events/{event_id}/agent-tasks")
        assert agent_tasks.status_code == 200
        assert agent_tasks.json()

        shared_memory = client.get(f"/platform/events/{event_id}/shared-memory")
        assert shared_memory.status_code == 200
        assert shared_memory.json()["active_agents"]
        assert "active_decision_path" in shared_memory.json()
        assert "open_questions" in shared_memory.json()

        session_memory = client.get(f"/platform/copilot/sessions/{session_id}/memory")
        assert session_memory.status_code == 200
        assert session_memory.json()["session_memory"]["session_id"] == session_id

        trigger_feed = client.get(f"/platform/events/{event_id}/trigger-events")
        assert trigger_feed.status_code == 200
        assert trigger_feed.json()

        timeline = client.get(f"/platform/events/{event_id}/agent-timeline")
        assert timeline.status_code == 200
        assert timeline.json()

        supervisor_runs = client.get(f"/platform/events/{event_id}/supervisor-runs")
        assert supervisor_runs.status_code == 200
        assert supervisor_runs.json()

        manual_run = client.post(f"/platform/events/{event_id}/supervisor/run")
        assert manual_run.status_code == 200
        assert manual_run.json()["trigger_type"] == "manual_run"

        tick = client.post(f"/platform/supervisor/tick?event_id={event_id}")
        assert tick.status_code == 200
        assert len(tick.json()) >= 1

        replay = client.post(
            f"/platform/agent-tasks/{agent_tasks.json()[0]['task_id']}/replay",
            json={"replay_reason": "pytest replay"},
        )
        assert replay.status_code == 200
        assert replay.json()["replayed_from_task_id"] == agent_tasks.json()[0]["task_id"]

        second_reply = client.post(
            f"/platform/copilot/sessions/{session_id}/messages",
            json={"content": "What does this mean for the factory right now?"},
        )
        assert second_reply.status_code == 200
        assert second_reply.json()["proposals"] == []

        simulation_response = client.post(
            f"/platform/events/{event_id}/simulation-updates",
            json={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "depth_threshold_m": 0.45,
                "flow_threshold_mps": 1.2,
                "cells": [item.model_dump(mode="json") for item in sample_simulation_update().cells],
            },
        )
        assert simulation_response.status_code == 200
        pending_snapshot = client.get("/platform/proposals/pending")
        assert pending_snapshot.status_code == 200
        pending_items = pending_snapshot.json()["items"]
        assert pending_items
        proposal_id = pending_items[0]["proposal"]["proposal_id"]
        pending_package_response = client.get(f"/platform/events/{event_id}/regional-analysis-packages/pending")
        assert pending_package_response.status_code == 200
        assert pending_package_response.json()["package_id"]
        package_id = pending_package_response.json()["package_id"]
        package_history_response = client.get(f"/platform/events/{event_id}/regional-analysis-packages?include_pending=false")
        assert package_history_response.status_code == 200

        draft_update = client.patch(
            f"/platform/proposals/{proposal_id}/draft",
            json={
                "operator_id": "shift_commander",
                "operator_role": "commander",
                "action_scope": {"message": "Use the commander-edited district notification."},
            },
        )
        assert draft_update.status_code == 200
        assert draft_update.json()["proposal"]["edited_by_commander"] is True

        approve_response = client.post(
            f"/platform/regional-analysis-packages/{package_id}/approve",
            json={
                "operator_id": "shift_commander",
                "operator_role": "commander",
                "note": "Approve the current regional action.",
            },
        )
        assert approve_response.status_code == 200
        assert approve_response.json()["status"] == "approved"

        regional_history = client.get(f"/platform/events/{event_id}/regional-proposals")
        assert regional_history.status_code == 200
        assert regional_history.json()

        experience_context = client.get(f"/platform/events/{event_id}/experience-context")
        assert experience_context.status_code == 200
        assert "relevant_records" in experience_context.json()
        assert "strategy_patterns" in experience_context.json()

        strategy_history = client.get("/platform/entities/factory_wyr_bio/strategy-history")
        assert strategy_history.status_code == 200
        assert strategy_history.json()["entity_id"] == "factory_wyr_bio"

        decision_report = client.get(f"/platform/events/{event_id}/decision-report")
        assert decision_report.status_code == 200
        assert decision_report.json()["event_id"] == event_id
        assert "active_decision_path" in decision_report.json()

        metrics = client.get("/platform/agent-metrics")
        assert metrics.status_code == 200
        assert "fanout_count" in metrics.json()

        benchmarks = client.get("/platform/evaluation/benchmarks")
        assert benchmarks.status_code == 200
        assert len(benchmarks.json()) >= 3

        capabilities = client.get("/platform/security/capabilities?operator_role=observer")
        assert capabilities.status_code == 200
        assert capabilities.json()["operator_role"] == "observer"
        assert capabilities.json()["capabilities"]["archive_run"] is False
        assert capabilities.json()["capabilities"]["evaluation_run"] is False

        report = client.post("/platform/evaluation/run")
        assert report.status_code == 200
        report_id = report.json()["report_id"]
        assert report.json()["scenario_results"]
        assert "hallucination_rate" in report.json()

        report_detail = client.get(f"/platform/evaluation/reports/{report_id}")
        assert report_detail.status_code == 200
        assert report_detail.json()["report_id"] == report_id
        assert report_detail.json()["scenario_results"]

        replay = client.post(f"/platform/evaluation/reports/{report_id}/replay")
        assert replay.status_code == 200
        assert replay.json()["report_id"] != report_id
        assert replay.json()["notes"][0].startswith(f"已重放评测报告 {report_id}")


def test_platform_api_returns_explicit_llm_errors_without_rule_fallback(tmp_path: Path):
    system = FloodWarningSystem(tmp_path / "system.db", llm_gateway=AlwaysFailLLMGateway())
    event_id = seed_event(system)

    with bound_test_client(system) as client:
        advisory_response = client.post(
            "/platform/advisories/generate",
            json={
                "event_id": event_id,
                "area_id": "beilin_10km2",
                "entity_id": "school_wyl_primary",
                "operator_role": "commander",
            },
        )
        assert advisory_response.status_code == 503
        assert "llm_unavailable" in advisory_response.json()["detail"]

        session_response = client.post(
            "/platform/copilot/sessions/bootstrap",
            json={"event_id": event_id, "operator_role": "commander"},
        )
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        message_response = client.post(
            f"/platform/copilot/sessions/{session_id}/messages",
            json={"content": "What does this mean for the school right now?"},
        )
        assert message_response.status_code == 503
        assert "llm_unavailable" in message_response.json()["detail"]


def test_event_resource_override_affects_only_target_event(tmp_path: Path):
    system = build_system(tmp_path)
    production = system.production_platform
    first_event_id = seed_event(system)
    second_event_id = seed_event(system)

    override = ResourceStatus(
        area_id="beilin_10km2",
        vehicle_count=1,
        staff_count=2,
        supply_kits=5,
        rescue_boats=0,
        ambulance_count=0,
        drone_count=0,
        portable_pumps=0,
        power_generators=1,
        medical_staff_count=2,
        volunteer_count=2,
        satellite_phones=1,
        notes="pytest override",
    )
    production.save_event_resource_status(first_event_id, override)

    first_impact = production.get_entity_impact("resident_elderly_ls1", event_id=first_event_id)
    second_impact = production.get_entity_impact("resident_elderly_ls1", event_id=second_event_id)

    assert "协助转运车辆储备不足" in first_impact.resource_gap
    assert "医疗支援覆盖不足" in first_impact.resource_gap
    assert "协助转运车辆储备不足" not in second_impact.resource_gap

    production.delete_event_resource_status(first_event_id)
    reset_impact = production.get_entity_impact("resident_elderly_ls1", event_id=first_event_id)
    assert "协助转运车辆储备不足" not in reset_impact.resource_gap


def test_admin_api_runtime_updates_take_effect_without_restart(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    region = system.production_platform.area_profiles["beilin_10km2"].region

    with bound_test_client(system) as client:
        list_response = client.get("/platform/admin/entity-profiles")
        assert list_response.status_code == 200
        assert any(item["entity_id"] == "school_wyl_primary" for item in list_response.json())

        profile_payload = {
            "profile": {
                "entity_id": "school_wyl_primary",
                "area_id": "beilin_10km2",
                "entity_type": "school",
                "name": "WYL Primary School Updated",
                "village": "Wuyuanli Village",
                "location_hint": "North gate higher ground",
                "resident_count": 860,
                "current_occupancy": 820,
                "vulnerability_tags": ["children", "dismissal_peak"],
                "mobility_constraints": [],
                "key_assets": [],
                "inventory_summary": "",
                "continuity_requirement": "",
                "preferred_transport_mode": "walk",
                "notification_preferences": ["dashboard", "sms"],
                "emergency_contacts": [{"name": "Duty lead", "phone": "13800000002", "role": "lead"}],
                "custom_attributes": {"school_bus_count": 8},
            },
            "operator_id": "pytest_admin",
            "operator_role": "commander",
        }
        update_profile = client.put("/platform/admin/entity-profiles/school_wyl_primary", json=profile_payload)
        assert update_profile.status_code == 200
        impact_response = client.get(f"/platform/entities/school_wyl_primary/impact?event_id={event_id}")
        assert impact_response.status_code == 200
        assert impact_response.json()["entity"]["name"] == "WYL Primary School Updated"

        area_resource_response = client.put(
            "/platform/admin/areas/beilin_10km2/resource-status",
            json={
                "resource_status": {
                    "area_id": "beilin_10km2",
                    "vehicle_count": 9,
                    "staff_count": 20,
                    "supply_kits": 80,
                    "rescue_boats": 1,
                    "ambulance_count": 2,
                    "drone_count": 1,
                    "portable_pumps": 3,
                    "power_generators": 4,
                    "medical_staff_count": 14,
                    "volunteer_count": 30,
                    "satellite_phones": 6,
                    "notes": "pytest area default",
                },
                "operator_id": "pytest_admin",
                "operator_role": "commander",
            },
        )
        assert area_resource_response.status_code == 200
        assert area_resource_response.json()["scope"] == "area_default"

        event_resource_response = client.put(
            f"/platform/admin/events/{event_id}/resource-status",
            json={
                "resource_status": {
                    "area_id": "beilin_10km2",
                    "vehicle_count": 1,
                    "staff_count": 2,
                    "supply_kits": 5,
                    "rescue_boats": 0,
                    "ambulance_count": 0,
                    "drone_count": 0,
                    "portable_pumps": 0,
                    "power_generators": 1,
                    "medical_staff_count": 2,
                    "volunteer_count": 2,
                    "satellite_phones": 1,
                    "notes": "pytest event override",
                },
                "operator_id": "pytest_admin",
                "operator_role": "commander",
            },
        )
        assert event_resource_response.status_code == 200
        assert event_resource_response.json()["scope"] == "event_override"

        imported = client.post(
            "/platform/admin/rag-documents/import",
            json={
                "documents": [
                    {
                        "doc_id": "policy_school_runtime",
                        "corpus": "policy",
                        "title": "Orange response evacuation for school runtime override",
                        "content": "Orange response evacuation for school should prioritize guardian notification and gate transfer.",
                        "metadata": {
                            "region": region,
                            "updated_at": "2026-04-02T08:00:00+00:00",
                        },
                    }
                ],
                "operator_id": "pytest_admin",
                "operator_role": "commander",
            },
        )
        assert imported.status_code == 200
        assert any(item["doc_id"] == "policy_school_runtime" for item in imported.json()["documents"])

        advisory_response = client.post(
            "/platform/advisories/generate",
            json={
                "event_id": event_id,
                "area_id": "beilin_10km2",
                "entity_id": "school_wyl_primary",
                "operator_role": "commander",
            },
        )
        assert advisory_response.status_code == 200
        evidence_ids = [item["source_id"] for item in advisory_response.json()["evidence"]]
        assert "policy_school_runtime" in evidence_ids


def test_platform_copilot_carries_focus_memory_across_turns(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event_id, operator_role="commander")
    )
    first = production.send_copilot_message(
        session.session_id,
        V2CopilotMessageRequest(content="What does this mean for the factory right now?"),
    )
    second = production.send_copilot_message(
        session.session_id,
        V2CopilotMessageRequest(content="Continue with that target and tell me the stock and shutdown advice."),
    )

    assert first.memory_snapshot is not None
    assert first.memory_snapshot.focus_entity_id == "factory_wyr_bio"
    assert second.latest_answer is not None
    assert second.latest_answer.memory_snapshot is not None
    assert second.latest_answer.memory_snapshot.focus_entity_id == "factory_wyr_bio"
    assert any("previous turn" in item.lower() for item in second.latest_answer.carried_context_notes)


def test_supervisor_creates_shared_memory_and_agent_results(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    shared_memory = production.get_shared_memory_snapshot(event_id)
    agent_tasks = production.list_agent_tasks(event_id)
    agent_results = production.list_agent_results(event_id)
    supervisor_runs = production.list_supervisor_runs(event_id)
    agent_status = production.get_agent_status(event_id)

    assert shared_memory.active_agents
    assert shared_memory.latest_summary
    assert agent_tasks
    assert all(task.status.value == "completed" for task in agent_tasks[:2])
    assert agent_results
    assert supervisor_runs
    assert agent_status["completed_task_count"] >= 1


def test_trigger_deduping_and_task_replay_timeline(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    first = production.publish_trigger(event_id, trigger_type=TriggerEventType.FRESHNESS_EXPIRED, payload={"scope": "hazard"})
    second = production.publish_trigger(event_id, trigger_type=TriggerEventType.FRESHNESS_EXPIRED, payload={"scope": "hazard"})
    assert first.trigger_id == second.trigger_id

    task = production.list_agent_tasks(event_id)[0]
    replay = production.replay_agent_task(task.task_id, ReplayRequest(replay_reason="pytest replay"))
    assert replay.replayed_from_task_id == task.task_id

    timeline = production.list_agent_timeline(event_id)
    assert any(entry.entry_type == "trigger" for entry in timeline)
    assert any(entry.task_event_type == "replay_completed" for entry in timeline if entry.task_event_type)


def test_background_supervisor_loop_runs_periodically(tmp_path: Path):
    system = build_system(tmp_path)
    system.supervisor_loop.interval_seconds = 0.1
    event_id = seed_event(system)
    initial_count = len(system.production_platform.list_supervisor_runs(event_id))

    system.start_background_services()
    try:
        deadline = time.monotonic() + 1.5
        status = None
        while time.monotonic() < deadline:
            current_count = len(system.production_platform.list_supervisor_runs(event_id))
            status = system.background_services_status()
            if current_count > initial_count and status["last_completed_at"] is not None:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("background supervisor loop did not create a new run in time.")

        assert status is not None
        assert status["running"] is True
        assert status["last_completed_at"] is not None
        assert status["last_error"] is None
        assert status["circuit_state"] == "closed"
        assert status["consecutive_failures"] == 0
    finally:
        system.stop_background_services()

    assert system.background_services_status()["running"] is False


def test_supervisor_loop_retries_and_records_warning(tmp_path: Path):
    system = build_system(tmp_path)
    seed_event(system)
    system.supervisor_loop.interval_seconds = 0.05
    system.supervisor_loop.max_retries = 1
    system.supervisor_loop.retry_backoffs_seconds = (0.01, 0.01)
    original_tick = system.production_platform.agent_supervisor.tick
    state = {"calls": 0}

    def flaky_tick(event_id: str | None = None):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("synthetic sweep failure")
        return original_tick(event_id)

    system.production_platform.agent_supervisor.tick = flaky_tick
    system.start_background_services()
    try:
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            status = system.background_services_status()
            warnings = system.production_platform.list_operational_alerts(limit=10)
            if status["last_retry_at"] is not None and warnings:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("supervisor loop did not retry and emit a warning alert.")

        assert status["last_retry_at"] is not None
        assert any(item.severity.value == "warning" for item in warnings)
    finally:
        system.stop_background_services()
        system.production_platform.agent_supervisor.tick = original_tick


def test_archive_status_and_audit_endpoints(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform
    old_time = datetime.now(timezone.utc) - timedelta(days=30)

    task = production.list_agent_tasks(event_id)[0].model_copy(update={"created_at": old_time})
    result = production.list_agent_results(event_id)[0].model_copy(update={"created_at": old_time})
    run = production.list_supervisor_runs(event_id)[0].model_copy(update={"created_at": old_time})
    system.repository.save_v2_agent_task(task)
    system.repository.save_v2_agent_result(result)
    system.repository.save_v2_supervisor_run(run)

    with bound_test_client(system) as client:
        archive_response = client.post("/platform/archive/run")
        assert archive_response.status_code == 200
        archive_payload = archive_response.json()
        assert archive_payload["archived_record_count"] >= 1
        assert archive_payload["last_archive_run"] is not None

        audit_response = client.get("/platform/audit/records")
        assert audit_response.status_code == 200
        assert any(item["source_type"] == "housekeeping" for item in audit_response.json())

        status_response = client.get("/platform/archive/status")
        assert status_response.status_code == 200
        assert status_response.json()["last_archive_run"]["hot_records_archived"] >= 1


def test_rbac_blocks_low_privilege_control_actions(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)

    with bound_test_client(system) as client:
        archive_denied = client.post("/platform/archive/run", headers={"X-Operator-Role": "observer"})
        assert archive_denied.status_code == 403

        evaluation_denied = client.post("/platform/evaluation/run", headers={"X-Operator-Role": "street_operator"})
        assert evaluation_denied.status_code == 403

        supervisor_denied = client.post(
            f"/platform/events/{event_id}/supervisor/run",
            headers={"X-Operator-Role": "district_operator"},
        )
        assert supervisor_denied.status_code == 403


def test_agent_twin_api_exposes_twin_overview_dialog_and_stream(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform
    production.ingest_simulation_update(event_id, sample_simulation_update())

    with bound_test_client(system) as client:
        overview_response = client.get(f"/agent-twin/events/{event_id}/twin-overview")
        assert overview_response.status_code == 200
        overview_payload = overview_response.json()
        assert overview_payload["event_id"] == event_id
        assert overview_payload["focus_objects"]
        assert "pending_proposal_count" in overview_payload
        assert overview_payload["map_layers"]

        object_id = overview_payload["focus_objects"][0]["object_id"]
        focus_response = client.get(f"/agent-twin/events/{event_id}/objects/{object_id}")
        assert focus_response.status_code == 200
        assert focus_response.json()["object_id"] == object_id
        assert focus_response.json()["recommended_actions"]

        council_response = client.get(f"/agent-twin/events/{event_id}/agent-council")
        assert council_response.status_code == 200
        council_payload = council_response.json()
        assert council_payload["roles"]
        assert council_payload["audit_decision"]["status"] in {"blocked", "approved_for_review"}

        dialog_response = client.post(
            f"/agent-twin/events/{event_id}/dialog",
            json={"object_id": object_id, "message": "请解释当前对象的影响链并给出处置建议。"},
        )
        assert dialog_response.status_code == 200
        dialog_payload = dialog_response.json()
        assert dialog_payload["object_id"] == object_id
        assert dialog_payload["answer"]
        assert dialog_payload["recommended_actions"]
        stream_events = system.agent_twin.build_stream_events(event_id, focus_object_id=object_id)
        assert {item.event_type for item in stream_events} >= {
            "twin_overview_updated",
            "focus_object_updated",
            "agent_council_updated",
            "proposal_status_changed",
            "warnings_generated",
            "proposal_generated",
        }


def test_agent_twin_proposal_generation_and_warning_bridge_reuses_platform_closure(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform
    production.ingest_simulation_update(event_id, sample_simulation_update())

    with bound_test_client(system) as client:
        proposal_response = client.post(
            f"/agent-twin/events/{event_id}/proposals/generate",
            json={"object_ids": []},
        )
        assert proposal_response.status_code == 200
        proposal_payload = proposal_response.json()
        assert proposal_payload["blocked"] is False
        assert proposal_payload["proposals"]

        proposal_id = proposal_payload["proposals"][0]["proposal"]["proposal"]["proposal_id"]
        approve_response = client.post(
            f"/platform/proposals/{proposal_id}/approve",
            json={
                "operator_id": "shift_commander",
                "operator_role": "commander",
                "note": "approve from AgentTwin contract test",
            },
        )
        assert approve_response.status_code == 200
        assert approve_response.json()["proposal"]["status"] == "approved"

        warnings_response = client.post(f"/agent-twin/proposals/{proposal_id}/warnings/generate")
        assert warnings_response.status_code == 200
        warnings_payload = warnings_response.json()
        assert warnings_payload["proposal_id"] == proposal_id
        assert warnings_payload["warnings"]

        notification_drafts = production.repository.list_v2_notification_drafts(event_id)
        assert any(item.proposal_id == proposal_id for item in notification_drafts)
        agent_twin_warning_rows = production.repository.list_v3_audience_warnings(event_id, proposal_id=proposal_id)
        assert agent_twin_warning_rows
