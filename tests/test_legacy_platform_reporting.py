from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


from flood_system.models import RiskLevel
from flood_system.system import FloodWarningSystem
from flood_system.compat.legacy_platform.models import (
    EventCreateRequest,
    LongTermMemoryRecord,
    ObservationBatchRequest,
    V2CopilotSessionRequest,
)
from flood_system.compat.legacy_platform.reporting import format_daily_report_message

from tests.support.system import (
    build_system,
    AlwaysFailLLMGateway,
    sample_observations,
    sample_simulation_update,
    seed_event,
    bound_test_client,
)


def test_daily_summary_service_generates_deduped_reports_and_delivers_to_sessions(
    tmp_path: Path,
):
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
        ObservationBatchRequest(
            operator="pytest", observations=sample_observations(observation_time)
        ),
    )
    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event.event_id, operator_role="commander")
    )

    reports = system.daily_summary_service.run_once(
        now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc)
    )
    assert len(reports) == 1
    report = reports[0]
    assert report.event_id == event.event_id
    assert session.session_id in report.delivered_session_ids
    assert (
        system.repository.get_v2_daily_report_run(event.event_id, "2026-04-14")
        is not None
    )
    assert len(production.list_daily_reports(event.event_id)) == 1

    second = system.daily_summary_service.run_once(
        now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc)
    )
    assert second == []
    session_view = production.get_copilot_session(session.session_id)
    assert session_view.daily_reports
    assert session_view.daily_reports[0].report_id == report.report_id
    assert any(report.headline in item.content for item in session_view.messages)


def test_daily_summary_service_skips_events_without_previous_day_activity(
    tmp_path: Path,
):
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

    reports = system.daily_summary_service.run_once(
        now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc)
    )
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


def test_long_term_memory_query_prioritizes_area_risk_and_entity_matches(
    tmp_path: Path,
):
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
    system = FloodWarningSystem(
        tmp_path / "system.db", llm_gateway=AlwaysFailLLMGateway()
    )
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
            observations=sample_observations(
                datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)
            ),
        ),
    )

    reports = system.daily_summary_service.run_once(
        now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc)
    )
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
            observations=sample_observations(
                datetime(2026, 4, 14, 10, 0, tzinfo=timezone.utc)
            ),
        ),
    )
    session = production.bootstrap_copilot_session(
        V2CopilotSessionRequest(event_id=event.event_id, operator_role="commander")
    )
    system.daily_summary_service.run_once(
        now=datetime(2026, 4, 15, 1, 0, tzinfo=timezone.utc)
    )
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
        session_response = client.get(
            f"/platform/copilot/sessions/{session.session_id}"
        )
        assert session_response.status_code == 200
        assert session_response.json()["daily_reports"]
        assert session_response.json()["episode_summaries"]

        daily_reports = client.get(f"/platform/events/{event.event_id}/daily-reports")
        assert daily_reports.status_code == 200
        assert len(daily_reports.json()) == 1

        episode_summaries = client.get(
            f"/platform/events/{event.event_id}/episode-summaries"
        )
        assert episode_summaries.status_code == 200
        assert len(episode_summaries.json()) == 1

        long_term_memory = client.get(
            f"/platform/events/{event.event_id}/long-term-memory"
        )
        assert long_term_memory.status_code == 200
        assert len(long_term_memory.json()) == 1

        experience_context = client.get(
            f"/platform/events/{event.event_id}/experience-context"
        )
        assert experience_context.status_code == 200
        assert "long_term_memories" in experience_context.json()
