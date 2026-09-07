from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


from flood_system.models import ResourceStatus, RiskLevel
from flood_system.compat.legacy_platform.exposure_engine import ExposureEngine
from flood_system.compat.legacy_platform.models import (
    AdvisoryRequest,
    EntityProfile,
    EntityType,
    EventCreateRequest,
    HazardState,
    HazardTile,
    ObservationBatchRequest,
    ProposalResolutionRequest,
    RegionalAnalysisPackageStatus,
    SimulationCell,
    SimulationUpdateRequest,
    ProposalStatus,
    ToolFailureMode,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
)
from flood_system.compat.legacy_platform.routing import RoutePlanningService

from tests.support.system import (
    build_system,
    sample_observations,
    sample_simulation_update,
    wait_for_agent_processing,
    seed_event,
)


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
    assert snapshot.latest_hazard_state.overall_risk_level.value in {
        "Yellow",
        "Orange",
        "Red",
    }
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


def test_platform_entity_impact_and_advisory_require_confirmation_for_school(
    tmp_path: Path,
):
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

    assert (
        impact.risk_reason[0]
        == "碑林区 10分钟风险栅格评分为 92，预测积水深度约 74 厘米。"
    )
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


def test_platform_tool_registry_exposes_schema_timeout_and_failure_modes(
    tmp_path: Path,
):
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


def test_platform_copilot_returns_explainable_plan_execution_and_proposal(
    tmp_path: Path,
):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event_id, operator_role="commander")
    )
    view = production.send_copilot_message(
        session.session_id,
        V2CopilotMessageRequest(
            content="What does this mean for the school right now?"
        ),
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
    assert any(
        item.tool_name == "get_route_options"
        for item in view.latest_answer.skipped_tools
    )

    school_proposal = next(
        (item for item in view.proposals if item.entity_id == "school_wyl_primary"),
        None,
    )
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
        V2CopilotMessageRequest(
            content="Which route should the nursing home use to reach the safest shelter?"
        ),
    )

    assert view.latest_answer is not None
    execution_names = [item.tool_name for item in view.latest_answer.tool_executions]
    assert "resolve_target_entity" in execution_names
    assert "get_route_options" in execution_names
    assert "get_live_traffic" in execution_names
    assert "get_shelter_capacity" in execution_names


def test_platform_regional_approval_generates_notification_drafts_and_execution_logs(
    tmp_path: Path,
):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    result = production.ingest_simulation_update(event_id, sample_simulation_update())
    assert result["overall_risk_level"].value in {"Orange", "Red"}

    pending = production.list_regional_proposals(
        event_id, statuses=[ProposalStatus.PENDING.value]
    )
    proposal = next(
        item.proposal
        for item in pending
        if item.proposal.action_type == "regional_notification"
    )
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
                SimulationCell(
                    cell_id="grid_a",
                    label="North corridor",
                    water_depth_m=1.9,
                    flow_velocity_mps=2.1,
                ),
                SimulationCell(
                    cell_id="grid_b",
                    label="School edge",
                    water_depth_m=1.8,
                    flow_velocity_mps=2.0,
                ),
                SimulationCell(
                    cell_id="grid_c",
                    label="Transit hub",
                    water_depth_m=1.7,
                    flow_velocity_mps=1.9,
                ),
                SimulationCell(
                    cell_id="grid_d",
                    label="Residential lowland",
                    water_depth_m=2.0,
                    flow_velocity_mps=2.2,
                ),
            ],
        ),
    )

    pending = production.list_regional_proposals(
        event_id, statuses=[ProposalStatus.PENDING.value]
    )
    generic = next(
        (item for item in pending if item.proposal.action_type == "traffic_control"),
        None,
    )
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

    pending = production.list_regional_proposals(
        event_id, statuses=[ProposalStatus.PENDING.value]
    )
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
        if item.proposal.risk_stage_key == package.package_id
        and item.proposal.proposal_id in approved.proposal_ids
    ]
    assert approved_items
    assert all(
        item.proposal.status == ProposalStatus.APPROVED for item in approved_items
    )

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
        if item.proposal.risk_stage_key == reject_package.package_id
        and item.proposal.proposal_id in rejected.proposal_ids
    ]
    assert rejected_items
    assert all(
        item.proposal.status == ProposalStatus.REJECTED for item in rejected_items
    )


def test_platform_new_risk_stage_creates_new_package_and_preserves_history(
    tmp_path: Path,
):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    production = system.production_platform

    production.ingest_simulation_update(event_id, sample_simulation_update())
    first_package = production.get_pending_regional_analysis_package(event_id)
    assert first_package is not None
    production.approve_regional_analysis_package(
        first_package.package_id,
        ProposalResolutionRequest(
            operator_id="shift_commander",
            operator_role="commander",
            note="Approve stage one.",
        ),
    )

    production.ingest_simulation_update(
        event_id,
        SimulationUpdateRequest(
            generated_at=datetime.now(timezone.utc),
            depth_threshold_m=0.35,
            flow_threshold_mps=0.8,
            cells=[
                SimulationCell(
                    cell_id="grid_a",
                    label="North corridor",
                    water_depth_m=1.9,
                    flow_velocity_mps=2.1,
                ),
                SimulationCell(
                    cell_id="grid_b",
                    label="School edge",
                    water_depth_m=1.8,
                    flow_velocity_mps=2.0,
                ),
                SimulationCell(
                    cell_id="grid_c",
                    label="Transit hub",
                    water_depth_m=1.7,
                    flow_velocity_mps=1.9,
                ),
                SimulationCell(
                    cell_id="grid_d",
                    label="Residential lowland",
                    water_depth_m=2.0,
                    flow_velocity_mps=2.2,
                ),
            ],
        ),
    )
    current_package = production.get_pending_regional_analysis_package(event_id)
    history = production.list_regional_analysis_packages(
        event_id, include_pending=False
    )

    assert current_package is not None
    assert current_package.package_id != first_package.package_id
    assert any(item.package_id == first_package.package_id for item in history)
