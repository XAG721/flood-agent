from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


from flood_system.models import ResourceStatus
from flood_system.system import FloodWarningSystem
from flood_system.compat.legacy_platform.models import (
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
)

from tests.support.system import (
    build_system,
    AlwaysFailLLMGateway,
    sample_observations,
    sample_simulation_update,
    wait_for_agent_processing,
    seed_event,
    bound_test_client,
)


def test_platform_api_only_exposes_new_flow_and_supports_regional_proposals(
    tmp_path: Path,
):
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
                "observations": [
                    item.model_dump(mode="json") for item in sample_observations()
                ],
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
        assert (
            replay.json()["replayed_from_task_id"] == agent_tasks.json()[0]["task_id"]
        )

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
                "cells": [
                    item.model_dump(mode="json")
                    for item in sample_simulation_update().cells
                ],
            },
        )
        assert simulation_response.status_code == 200
        pending_snapshot = client.get("/platform/proposals/pending")
        assert pending_snapshot.status_code == 200
        pending_items = pending_snapshot.json()["items"]
        assert pending_items
        proposal_id = pending_items[0]["proposal"]["proposal_id"]
        pending_package_response = client.get(
            f"/platform/events/{event_id}/regional-analysis-packages/pending"
        )
        assert pending_package_response.status_code == 200
        assert pending_package_response.json()["package_id"]
        package_id = pending_package_response.json()["package_id"]
        package_history_response = client.get(
            f"/platform/events/{event_id}/regional-analysis-packages?include_pending=false"
        )
        assert package_history_response.status_code == 200

        draft_update = client.patch(
            f"/platform/proposals/{proposal_id}/draft",
            json={
                "operator_id": "shift_commander",
                "operator_role": "commander",
                "action_scope": {
                    "message": "Use the commander-edited district notification."
                },
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

        experience_context = client.get(
            f"/platform/events/{event_id}/experience-context"
        )
        assert experience_context.status_code == 200
        assert "relevant_records" in experience_context.json()
        assert "strategy_patterns" in experience_context.json()

        strategy_history = client.get(
            "/platform/entities/factory_wyr_bio/strategy-history"
        )
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

        capabilities = client.get(
            "/platform/security/capabilities?operator_role=observer"
        )
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
    system = FloodWarningSystem(
        tmp_path / "system.db", llm_gateway=AlwaysFailLLMGateway()
    )
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

    first_impact = production.get_entity_impact(
        "resident_elderly_ls1", event_id=first_event_id
    )
    second_impact = production.get_entity_impact(
        "resident_elderly_ls1", event_id=second_event_id
    )

    assert "协助转运车辆储备不足" in first_impact.resource_gap
    assert "医疗支援覆盖不足" in first_impact.resource_gap
    assert "协助转运车辆储备不足" not in second_impact.resource_gap

    production.delete_event_resource_status(first_event_id)
    reset_impact = production.get_entity_impact(
        "resident_elderly_ls1", event_id=first_event_id
    )
    assert "协助转运车辆储备不足" not in reset_impact.resource_gap


def test_admin_api_runtime_updates_take_effect_without_restart(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)
    region = system.production_platform.area_profiles["beilin_10km2"].region

    with bound_test_client(system) as client:
        list_response = client.get("/platform/admin/entity-profiles")
        assert list_response.status_code == 200
        assert any(
            item["entity_id"] == "school_wyl_primary" for item in list_response.json()
        )

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
                "emergency_contacts": [
                    {"name": "Duty lead", "phone": "13800000002", "role": "lead"}
                ],
                "custom_attributes": {"school_bus_count": 8},
            },
            "operator_id": "pytest_admin",
            "operator_role": "commander",
        }
        update_profile = client.put(
            "/platform/admin/entity-profiles/school_wyl_primary", json=profile_payload
        )
        assert update_profile.status_code == 200
        impact_response = client.get(
            f"/platform/entities/school_wyl_primary/impact?event_id={event_id}"
        )
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
        assert any(
            item["doc_id"] == "policy_school_runtime"
            for item in imported.json()["documents"]
        )

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
        evidence_ids = [
            item["source_id"] for item in advisory_response.json()["evidence"]
        ]
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
        V2CopilotMessageRequest(
            content="What does this mean for the factory right now?"
        ),
    )
    second = production.send_copilot_message(
        session.session_id,
        V2CopilotMessageRequest(
            content="Continue with that target and tell me the stock and shutdown advice."
        ),
    )

    assert first.memory_snapshot is not None
    assert first.memory_snapshot.focus_entity_id == "factory_wyr_bio"
    assert second.latest_answer is not None
    assert second.latest_answer.memory_snapshot is not None
    assert second.latest_answer.memory_snapshot.focus_entity_id == "factory_wyr_bio"
    assert any(
        "previous turn" in item.lower()
        for item in second.latest_answer.carried_context_notes
    )
