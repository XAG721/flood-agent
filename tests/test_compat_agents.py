from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time


from flood_system.compat.legacy_platform.models import (
    ReplayRequest,
    TriggerEventType,
)

from tests.support.system import (
    build_system,
    sample_simulation_update,
    seed_event,
    bound_test_client,
)


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

    first = production.publish_trigger(
        event_id,
        trigger_type=TriggerEventType.FRESHNESS_EXPIRED,
        payload={"scope": "hazard"},
    )
    second = production.publish_trigger(
        event_id,
        trigger_type=TriggerEventType.FRESHNESS_EXPIRED,
        payload={"scope": "hazard"},
    )
    assert first.trigger_id == second.trigger_id

    task = production.list_agent_tasks(event_id)[0]
    replay = production.replay_agent_task(
        task.task_id, ReplayRequest(replay_reason="pytest replay")
    )
    assert replay.replayed_from_task_id == task.task_id

    timeline = production.list_agent_timeline(event_id)
    assert any(entry.entry_type == "trigger" for entry in timeline)
    assert any(
        entry.task_event_type == "replay_completed"
        for entry in timeline
        if entry.task_event_type
    )


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
            current_count = len(
                system.production_platform.list_supervisor_runs(event_id)
            )
            status = system.background_services_status()
            if (
                current_count > initial_count
                and status["last_completed_at"] is not None
            ):
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                "background supervisor loop did not create a new run in time."
            )

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
            raise AssertionError(
                "supervisor loop did not retry and emit a warning alert."
            )

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

    task = production.list_agent_tasks(event_id)[0].model_copy(
        update={"created_at": old_time}
    )
    result = production.list_agent_results(event_id)[0].model_copy(
        update={"created_at": old_time}
    )
    run = production.list_supervisor_runs(event_id)[0].model_copy(
        update={"created_at": old_time}
    )
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
        assert any(
            item["source_type"] == "housekeeping" for item in audit_response.json()
        )

        status_response = client.get("/platform/archive/status")
        assert status_response.status_code == 200
        assert status_response.json()["last_archive_run"]["hot_records_archived"] >= 1


def test_rbac_blocks_low_privilege_control_actions(tmp_path: Path):
    system = build_system(tmp_path)
    event_id = seed_event(system)

    with bound_test_client(system) as client:
        archive_denied = client.post(
            "/platform/archive/run", headers={"X-Operator-Role": "observer"}
        )
        assert archive_denied.status_code == 403

        evaluation_denied = client.post(
            "/platform/evaluation/run", headers={"X-Operator-Role": "street_operator"}
        )
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
        focus_response = client.get(
            f"/agent-twin/events/{event_id}/objects/{object_id}"
        )
        assert focus_response.status_code == 200
        assert focus_response.json()["object_id"] == object_id
        assert focus_response.json()["recommended_actions"]

        council_response = client.get(f"/agent-twin/events/{event_id}/agent-council")
        assert council_response.status_code == 200
        council_payload = council_response.json()
        assert council_payload["roles"]
        assert council_payload["audit_decision"]["status"] in {
            "blocked",
            "approved_for_review",
        }

        dialog_response = client.post(
            f"/agent-twin/events/{event_id}/dialog",
            json={
                "object_id": object_id,
                "message": "请解释当前对象的影响链并给出处置建议。",
            },
        )
        assert dialog_response.status_code == 200
        dialog_payload = dialog_response.json()
        assert dialog_payload["object_id"] == object_id
        assert dialog_payload["answer"]
        assert dialog_payload["recommended_actions"]
        stream_events = system.agent_twin.build_stream_events(
            event_id, focus_object_id=object_id
        )
        assert {item.event_type for item in stream_events} >= {
            "twin_overview_updated",
            "focus_object_updated",
            "agent_council_updated",
            "proposal_status_changed",
            "warnings_generated",
            "proposal_generated",
        }


def test_agent_twin_proposal_generation_and_warning_bridge_reuses_platform_closure(
    tmp_path: Path,
):
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

        proposal_id = proposal_payload["proposals"][0]["proposal"]["proposal"][
            "proposal_id"
        ]
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

        warnings_response = client.post(
            f"/agent-twin/proposals/{proposal_id}/warnings/generate"
        )
        assert warnings_response.status_code == 200
        warnings_payload = warnings_response.json()
        assert warnings_payload["proposal_id"] == proposal_id
        assert warnings_payload["warnings"]

        notification_drafts = production.repository.list_v2_notification_drafts(
            event_id
        )
        assert any(item.proposal_id == proposal_id for item in notification_drafts)
        agent_twin_warning_rows = production.repository.list_v3_audience_warnings(
            event_id, proposal_id=proposal_id
        )
        assert agent_twin_warning_rows
