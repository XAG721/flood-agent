from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import ResourceStatus
from .v2.models import (
    AgentResult,
    AgentMetricsView,
    AgentTask,
    AgentTaskEvent,
    AgentTimelineEntry,
    ActionProposal,
    Advisory,
    AlertSeverity,
    ArchiveRunRecord,
    ArchiveStatusView,
    AuditRecord,
    EntityProfile,
    EventRecord,
    EventEpisodeSummaryRecord,
    EventStreamRecord,
    EvaluationBenchmark,
    EvaluationReport,
    ExperienceRecord,
    ExperienceContextView,
    ExecutionLogEntry,
    HazardState,
    HighRiskEpisodeRecord,
    MemoryEventRecord,
    MemorySnapshot,
    NotificationDraft,
    ObservationIngestItem,
    OperationalAlert,
    PlanRunRecord,
    ProposalStatus,
    ReplayRequest,
    SharedMemorySnapshot,
    SimulationUpdateRecord,
    StrategyHistoryView,
    StrategyPattern,
    SupervisorHealthState,
    SupervisorRunRecord,
    ToolExecutionAuditRecord,
    TriggerEvent,
    TriggerEventStatus,
    TriggerEventType,
    V2CopilotMessage,
    DailyReportRecord,
    DailyReportRunRecord,
    LongTermMemoryRecord,
)


class SQLiteRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_events (
                    event_id TEXT PRIMARY KEY,
                    area_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_simulation_updates (
                    simulation_update_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_hazard_states (
                    event_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_stream_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_entity_profiles (
                    entity_id TEXT PRIMARY KEY,
                    area_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_area_resource_status (
                    area_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_event_resource_status (
                    event_id TEXT PRIMARY KEY,
                    area_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_advisories (
                    advisory_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_action_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    entity_id TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_copilot_sessions (
                    session_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_copilot_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_copilot_memory_state (
                    session_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_copilot_memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_event_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_copilot_plan_runs (
                    plan_run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_copilot_tool_executions (
                    execution_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_agent_results (
                    result_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_event_shared_memory (
                    event_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_supervisor_runs (
                    supervisor_run_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_trigger_events (
                    trigger_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dedupe_key TEXT,
                    created_at TEXT NOT NULL,
                    leased_at TEXT,
                    processed_at TEXT,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_agent_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_event_id TEXT UNIQUE NOT NULL,
                    event_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_supervisor_health_state (
                    component_key TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_operational_alerts (
                    alert_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    event_id TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    resolved_at TEXT,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_audit_records (
                    audit_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    event_id TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_archived_records (
                    archive_id TEXT PRIMARY KEY,
                    source_table TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    event_id TEXT,
                    session_id TEXT,
                    record_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_archive_runs (
                    archive_run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_notification_drafts (
                    draft_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_execution_logs (
                    log_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_experience_records (
                    experience_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    entity_id TEXT,
                    entity_type TEXT,
                    risk_level TEXT,
                    action_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_daily_reports (
                    report_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    report_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_daily_report_runs (
                    run_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    report_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_high_risk_episodes (
                    episode_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_event_episode_summaries (
                    summary_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_long_term_memories (
                    memory_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS v2_evaluation_reports (
                    report_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_v2_observations_event_id ON v2_observations(event_id, observed_at);
                CREATE INDEX IF NOT EXISTS idx_v2_simulation_updates_event_id ON v2_simulation_updates(event_id, generated_at);
                CREATE INDEX IF NOT EXISTS idx_v2_stream_records_event_id ON v2_stream_records(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_proposals_event_id ON v2_action_proposals(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_proposals_status ON v2_action_proposals(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_messages_session_id ON v2_copilot_messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_memory_events_session_id ON v2_copilot_memory_events(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_plan_runs_session_id ON v2_copilot_plan_runs(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_tool_executions_session_id ON v2_copilot_tool_executions(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_agent_tasks_event_id ON v2_agent_tasks(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_agent_tasks_agent_name ON v2_agent_tasks(agent_name, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_agent_results_event_id ON v2_agent_results(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_supervisor_runs_event_id ON v2_supervisor_runs(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_trigger_events_event_id ON v2_trigger_events(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_trigger_events_status ON v2_trigger_events(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_trigger_events_dedupe ON v2_trigger_events(dedupe_key, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_agent_task_events_event_id ON v2_agent_task_events(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_agent_task_events_task_id ON v2_agent_task_events(task_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_alerts_status ON v2_operational_alerts(status, severity, last_seen_at);
                CREATE INDEX IF NOT EXISTS idx_v2_alerts_event_id ON v2_operational_alerts(event_id, last_seen_at);
                CREATE INDEX IF NOT EXISTS idx_v2_audit_created_at ON v2_audit_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_audit_source_type ON v2_audit_records(source_type, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_archived_records_created_at ON v2_archived_records(created_at, archived_at);
                CREATE INDEX IF NOT EXISTS idx_v2_archived_records_source ON v2_archived_records(source_table, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_notification_event_id ON v2_notification_drafts(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_execution_event_id ON v2_execution_logs(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_entity_profiles_area_id ON v2_entity_profiles(area_id);
                CREATE INDEX IF NOT EXISTS idx_v2_event_resource_area_id ON v2_event_resource_status(area_id);
                CREATE INDEX IF NOT EXISTS idx_v2_experience_event_id ON v2_experience_records(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_experience_entity_id ON v2_experience_records(entity_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_experience_entity_type ON v2_experience_records(entity_type, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_daily_reports_event_date ON v2_daily_reports(event_id, report_date, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_daily_report_runs_event_date ON v2_daily_report_runs(event_id, report_date, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_high_risk_episodes_event_id ON v2_high_risk_episodes(event_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_v2_event_episode_summaries_event_id ON v2_event_episode_summaries(event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_v2_long_term_memories_event_id ON v2_long_term_memories(event_id, created_at);
                """
            )

    def save_v2_event(self, event: EventRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_events (event_id, area_id, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    area_id = excluded.area_id,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (event.event_id, event.area_id, event.updated_at.isoformat(), event.model_dump_json()),
            )

    def get_v2_event(self, event_id: str) -> EventRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return EventRecord.model_validate_json(row["payload"]) if row else None

    def list_v2_events(self, *, limit: int = 200) -> list[EventRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_events
                ORDER BY updated_at DESC, event_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [EventRecord.model_validate_json(row["payload"]) for row in rows]

    def get_latest_v2_event_id(self, area_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id FROM v2_events
                WHERE area_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (area_id,),
            ).fetchone()
        return str(row["event_id"]) if row else None

    def add_v2_observations(self, event_id: str, observations: list[ObservationIngestItem]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO v2_observations (event_id, observed_at, payload)
                VALUES (?, ?, ?)
                """,
                [
                    (event_id, observation.observed_at.isoformat(), observation.model_dump_json())
                    for observation in observations
                ],
            )

    def list_v2_observations(self, event_id: str) -> list[ObservationIngestItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_observations
                WHERE event_id = ?
                ORDER BY observed_at ASC, id ASC
                """,
                (event_id,),
            ).fetchall()
        return [ObservationIngestItem.model_validate_json(row["payload"]) for row in rows]

    def save_v2_simulation_update(self, update: SimulationUpdateRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_simulation_updates (simulation_update_id, event_id, generated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(simulation_update_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    generated_at = excluded.generated_at,
                    payload = excluded.payload
                """,
                (
                    update.simulation_update_id,
                    update.event_id,
                    update.generated_at.isoformat(),
                    update.model_dump_json(),
                ),
            )

    def get_latest_v2_simulation_update(self, event_id: str) -> SimulationUpdateRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_simulation_updates
                WHERE event_id = ?
                ORDER BY generated_at DESC, simulation_update_id DESC
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
        return SimulationUpdateRecord.model_validate_json(row["payload"]) if row else None

    def list_v2_simulation_updates(self, event_id: str, *, limit: int = 20) -> list[SimulationUpdateRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_simulation_updates
                WHERE event_id = ?
                ORDER BY generated_at DESC, simulation_update_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [SimulationUpdateRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_hazard_state(self, hazard_state: HazardState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_hazard_states (event_id, generated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    generated_at = excluded.generated_at,
                    payload = excluded.payload
                """,
                (hazard_state.event_id, hazard_state.generated_at.isoformat(), hazard_state.model_dump_json()),
            )

    def get_v2_hazard_state(self, event_id: str) -> HazardState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_hazard_states WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return HazardState.model_validate_json(row["payload"]) if row else None

    def add_v2_stream_record(self, record: EventStreamRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_stream_records (event_id, created_at, payload)
                VALUES (?, ?, ?)
                """,
                (record.event_id, record.created_at.isoformat(), record.model_dump_json()),
            )

    def add_v2_stream_record_for_payload(self, event_id: str, event_type: str, payload: dict) -> None:
        self.add_v2_stream_record(
            EventStreamRecord(
                event_id=event_id,
                event_type=event_type,
                payload=payload,
                created_at=datetime.now(timezone.utc),
            )
        )

    def list_v2_stream_records(self, event_id: str, *, limit: int = 20) -> list[EventStreamRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_stream_records
                WHERE event_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [EventStreamRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_entity_profile(self, entity: EntityProfile) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_entity_profiles (entity_id, area_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    area_id = excluded.area_id,
                    payload = excluded.payload
                """,
                (entity.entity_id, entity.area_id, entity.model_dump_json()),
            )

    def get_v2_entity_profile(self, entity_id: str) -> EntityProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_entity_profiles WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
        return EntityProfile.model_validate_json(row["payload"]) if row else None

    def list_v2_entity_profiles(
        self,
        *,
        area_id: str | None = None,
        entity_type: str | None = None,
    ) -> list[EntityProfile]:
        query = "SELECT payload FROM v2_entity_profiles"
        clauses: list[str] = []
        parameters: list[object] = []
        if area_id is not None:
            clauses.append("area_id = ?")
            parameters.append(area_id)
        with self._connect() as conn:
            if clauses:
                query = f"{query} WHERE {' AND '.join(clauses)}"
            query = f"{query} ORDER BY entity_id ASC"
            rows = conn.execute(query, parameters).fetchall()
        profiles = [EntityProfile.model_validate_json(row["payload"]) for row in rows]
        if entity_type is not None:
            profiles = [item for item in profiles if item.entity_type.value == entity_type]
        return profiles

    def delete_v2_entity_profile(self, entity_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM v2_entity_profiles WHERE entity_id = ?",
                (entity_id,),
            )
        return cursor.rowcount > 0

    def has_v2_entity_profiles(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM v2_entity_profiles LIMIT 1").fetchone()
        return row is not None

    def save_area_resource_status(self, resource_status: ResourceStatus) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_area_resource_status (area_id, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(area_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (resource_status.area_id, updated_at, resource_status.model_dump_json()),
            )

    def get_area_resource_status(self, area_id: str) -> ResourceStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_area_resource_status WHERE area_id = ?",
                (area_id,),
            ).fetchone()
        return ResourceStatus.model_validate_json(row["payload"]) if row else None

    def has_area_resource_statuses(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM v2_area_resource_status LIMIT 1").fetchone()
        return row is not None

    def save_event_resource_status(self, event_id: str, resource_status: ResourceStatus) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_event_resource_status (event_id, area_id, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    area_id = excluded.area_id,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (event_id, resource_status.area_id, updated_at, resource_status.model_dump_json()),
            )

    def get_event_resource_status(self, event_id: str) -> ResourceStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_event_resource_status WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return ResourceStatus.model_validate_json(row["payload"]) if row else None

    def delete_event_resource_status(self, event_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM v2_event_resource_status WHERE event_id = ?",
                (event_id,),
            )
        return cursor.rowcount > 0

    def save_v2_advisory(self, advisory: Advisory) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_advisories (advisory_id, event_id, generated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(advisory_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    generated_at = excluded.generated_at,
                    payload = excluded.payload
                """,
                (advisory.advisory_id, advisory.event_id, advisory.generated_at.isoformat(), advisory.model_dump_json()),
            )

    def save_v2_action_proposal(self, proposal: ActionProposal) -> None:
        resolved_at = proposal.resolved_at.isoformat() if proposal.resolved_at else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_action_proposals (
                    proposal_id, event_id, entity_id, created_at, resolved_at, status, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    entity_id = excluded.entity_id,
                    created_at = excluded.created_at,
                    resolved_at = excluded.resolved_at,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    proposal.proposal_id,
                    proposal.event_id,
                    proposal.entity_id,
                    proposal.created_at.isoformat(),
                    resolved_at,
                    proposal.status.value,
                    proposal.model_dump_json(),
                ),
            )

    def get_v2_action_proposal(self, proposal_id: str) -> ActionProposal | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_action_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        return ActionProposal.model_validate_json(row["payload"]) if row else None

    def list_v2_action_proposals(
        self,
        event_id: str | None = None,
        *,
        proposal_scope: str | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ActionProposal]:
        query = "SELECT payload FROM v2_action_proposals WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        query += " ORDER BY created_at DESC, proposal_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        proposals = [ActionProposal.model_validate_json(row["payload"]) for row in rows]
        if proposal_scope is not None:
            proposals = [item for item in proposals if item.proposal_scope == proposal_scope]
        if statuses is not None:
            allowed = set(statuses)
            proposals = [item for item in proposals if item.status.value in allowed]
        return proposals

    def list_v2_pending_regional_proposals(self) -> list[ActionProposal]:
        return self.list_v2_action_proposals(
            proposal_scope="regional",
            statuses=[ProposalStatus.PENDING.value],
        )

    def save_v2_copilot_session(self, session_id: str, event_id: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_sessions (session_id, event_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    payload = excluded.payload
                """,
                (session_id, event_id, json.dumps(payload)),
            )

    def get_v2_copilot_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, event_id, payload FROM v2_copilot_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        payload["session_id"] = row["session_id"]
        payload["event_id"] = row["event_id"]
        return payload

    def list_recent_v2_copilot_sessions(self, *, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, event_id, payload FROM v2_copilot_sessions
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        sessions: list[dict] = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["session_id"] = row["session_id"]
            payload["event_id"] = row["event_id"]
            sessions.append(payload)
        return sessions

    def list_v2_copilot_sessions_by_event(self, event_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, event_id, payload FROM v2_copilot_sessions
                WHERE event_id = ?
                ORDER BY rowid ASC
                """,
                (event_id,),
            ).fetchall()
        sessions: list[dict] = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["session_id"] = row["session_id"]
            payload["event_id"] = row["event_id"]
            sessions.append(payload)
        return sessions

    def save_v2_copilot_message(self, session_id: str, message: V2CopilotMessage) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_messages (message_id, session_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (message.message_id, session_id, message.created_at.isoformat(), message.model_dump_json()),
            )

    def list_v2_copilot_messages(self, session_id: str) -> list[V2CopilotMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_copilot_messages
                WHERE session_id = ?
                ORDER BY created_at ASC, message_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [V2CopilotMessage.model_validate_json(row["payload"]) for row in rows]

    def save_v2_copilot_memory_state(self, snapshot: MemorySnapshot) -> None:
        updated_dt = snapshot.updated_at or datetime.now(timezone.utc)
        payload = snapshot.model_copy(update={"updated_at": updated_dt}).model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_memory_state (session_id, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (snapshot.session_id, updated_dt.isoformat(), payload),
            )

    def get_v2_copilot_memory_state(self, session_id: str) -> MemorySnapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_copilot_memory_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return MemorySnapshot.model_validate_json(row["payload"]) if row else None

    def add_v2_copilot_memory_event(self, record: MemoryEventRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_memory_events (memory_event_id, session_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_event_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.memory_event_id,
                    record.session_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_copilot_memory_events(self, session_id: str, *, limit: int = 20) -> list[MemoryEventRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_copilot_memory_events
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [MemoryEventRecord.model_validate_json(row["payload"]) for row in rows]

    def publish_v2_trigger_event(
        self,
        trigger: TriggerEvent,
        *,
        dedupe_window_seconds: int = 15,
    ) -> TriggerEvent:
        with self._connect() as conn:
            if trigger.dedupe_key:
                cutoff = datetime.fromtimestamp(
                    datetime.now(timezone.utc).timestamp() - dedupe_window_seconds,
                    tz=timezone.utc,
                ).isoformat()
                row = conn.execute(
                    """
                    SELECT payload FROM v2_trigger_events
                    WHERE dedupe_key = ?
                      AND status IN (?, ?)
                      AND created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (
                        trigger.dedupe_key,
                        TriggerEventStatus.PENDING.value,
                        TriggerEventStatus.LEASED.value,
                        cutoff,
                    ),
                ).fetchone()
                if row:
                    return TriggerEvent.model_validate_json(row["payload"])
            conn.execute(
                """
                INSERT INTO v2_trigger_events (
                    trigger_id, event_id, trigger_type, status, dedupe_key, created_at, leased_at, processed_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trigger_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    trigger_type = excluded.trigger_type,
                    status = excluded.status,
                    dedupe_key = excluded.dedupe_key,
                    created_at = excluded.created_at,
                    leased_at = excluded.leased_at,
                    processed_at = excluded.processed_at,
                    payload = excluded.payload
                """,
                (
                    trigger.trigger_id,
                    trigger.event_id,
                    trigger.trigger_type.value,
                    trigger.status.value,
                    trigger.dedupe_key,
                    trigger.created_at.isoformat(),
                    trigger.leased_at.isoformat() if trigger.leased_at else None,
                    trigger.processed_at.isoformat() if trigger.processed_at else None,
                    trigger.model_dump_json(),
                ),
            )
        return trigger

    def lease_next_v2_trigger_event(self) -> TriggerEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_trigger_events
                WHERE status = ?
                ORDER BY created_at ASC, trigger_id ASC
                LIMIT 1
                """,
                (TriggerEventStatus.PENDING.value,),
            ).fetchone()
            if row is None:
                return None
            trigger = TriggerEvent.model_validate_json(row["payload"])
            leased = trigger.model_copy(
                update={
                    "status": TriggerEventStatus.LEASED,
                    "leased_at": datetime.now(timezone.utc),
                }
            )
            conn.execute(
                """
                UPDATE v2_trigger_events
                SET status = ?, leased_at = ?, payload = ?
                WHERE trigger_id = ?
                """,
                (
                    leased.status.value,
                    leased.leased_at.isoformat() if leased.leased_at else None,
                    leased.model_dump_json(),
                    leased.trigger_id,
                ),
            )
            return leased

    def mark_v2_trigger_event_processed(self, trigger_id: str) -> TriggerEvent | None:
        trigger = self.get_v2_trigger_event(trigger_id)
        if trigger is None:
            return None
        processed = trigger.model_copy(
            update={
                "status": TriggerEventStatus.PROCESSED,
                "processed_at": datetime.now(timezone.utc),
                "error_message": None,
            }
        )
        self._save_v2_trigger_event(processed)
        return processed

    def mark_v2_trigger_event_failed(self, trigger_id: str, *, error_message: str) -> TriggerEvent | None:
        trigger = self.get_v2_trigger_event(trigger_id)
        if trigger is None:
            return None
        failed = trigger.model_copy(
            update={
                "status": TriggerEventStatus.FAILED,
                "processed_at": datetime.now(timezone.utc),
                "error_message": error_message,
            }
        )
        self._save_v2_trigger_event(failed)
        return failed

    def _save_v2_trigger_event(self, trigger: TriggerEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_trigger_events (
                    trigger_id, event_id, trigger_type, status, dedupe_key, created_at, leased_at, processed_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trigger_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    trigger_type = excluded.trigger_type,
                    status = excluded.status,
                    dedupe_key = excluded.dedupe_key,
                    created_at = excluded.created_at,
                    leased_at = excluded.leased_at,
                    processed_at = excluded.processed_at,
                    payload = excluded.payload
                """,
                (
                    trigger.trigger_id,
                    trigger.event_id,
                    trigger.trigger_type.value,
                    trigger.status.value,
                    trigger.dedupe_key,
                    trigger.created_at.isoformat(),
                    trigger.leased_at.isoformat() if trigger.leased_at else None,
                    trigger.processed_at.isoformat() if trigger.processed_at else None,
                    trigger.model_dump_json(),
                ),
            )

    def get_v2_trigger_event(self, trigger_id: str) -> TriggerEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_trigger_events WHERE trigger_id = ?",
                (trigger_id,),
            ).fetchone()
        return TriggerEvent.model_validate_json(row["payload"]) if row else None

    def list_v2_trigger_events(self, event_id: str, *, limit: int = 40) -> list[TriggerEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_trigger_events
                WHERE event_id = ?
                ORDER BY created_at DESC, trigger_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [TriggerEvent.model_validate_json(row["payload"]) for row in rows]

    def count_pending_v2_trigger_events(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM v2_trigger_events WHERE status = ?",
                (TriggerEventStatus.PENDING.value,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def save_v2_copilot_plan_run(self, record: PlanRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_plan_runs (plan_run_id, session_id, message_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plan_run_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    message_id = excluded.message_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.plan_run_id,
                    record.session_id,
                    record.message_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_copilot_plan_runs(
        self,
        session_id: str,
        *,
        message_id: str | None = None,
        limit: int = 20,
    ) -> list[PlanRunRecord]:
        query = """
            SELECT payload FROM v2_copilot_plan_runs
            WHERE session_id = ?
        """
        parameters: list[object] = [session_id]
        if message_id is not None:
            query += " AND message_id = ?"
            parameters.append(message_id)
        query += " ORDER BY created_at DESC, plan_run_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [PlanRunRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_copilot_tool_execution(self, record: ToolExecutionAuditRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_tool_executions (execution_id, session_id, message_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    message_id = excluded.message_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.execution_id,
                    record.session_id,
                    record.message_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_copilot_tool_executions(
        self,
        session_id: str,
        *,
        message_id: str | None = None,
        limit: int = 40,
    ) -> list[ToolExecutionAuditRecord]:
        query = """
            SELECT payload FROM v2_copilot_tool_executions
            WHERE session_id = ?
        """
        parameters: list[object] = [session_id]
        if message_id is not None:
            query += " AND message_id = ?"
            parameters.append(message_id)
        query += " ORDER BY created_at DESC, execution_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [ToolExecutionAuditRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_agent_task(self, task: AgentTask) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_agent_tasks (task_id, event_id, agent_name, status, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    agent_name = excluded.agent_name,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    task.task_id,
                    task.event_id,
                    task.agent_name.value,
                    task.status.value,
                    task.created_at.isoformat(),
                    task.model_dump_json(),
                ),
            )

    def get_v2_agent_task(self, task_id: str) -> AgentTask | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM v2_agent_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return AgentTask.model_validate_json(row["payload"]) if row else None

    def list_v2_agent_tasks(self, event_id: str, *, limit: int = 50) -> list[AgentTask]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_agent_tasks
                WHERE event_id = ?
                ORDER BY created_at DESC, task_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [AgentTask.model_validate_json(row["payload"]) for row in rows]

    def add_v2_agent_task_event(self, record: AgentTaskEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_agent_task_events (
                    task_event_id, event_id, task_id, agent_name, event_type, created_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_event_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    task_id = excluded.task_id,
                    agent_name = excluded.agent_name,
                    event_type = excluded.event_type,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.task_event_id,
                    record.event_id,
                    record.task_id,
                    record.agent_name.value,
                    record.event_type.value,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_agent_task_events(
        self,
        event_id: str,
        *,
        limit: int = 120,
    ) -> list[AgentTaskEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_agent_task_events
                WHERE event_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [AgentTaskEvent.model_validate_json(row["payload"]) for row in rows]

    def count_v2_agent_task_events(self, *, event_type: str | None = None) -> int:
        query = "SELECT COUNT(*) AS count FROM v2_agent_task_events WHERE 1 = 1"
        parameters: list[object] = []
        if event_type is not None:
            query += " AND event_type = ?"
            parameters.append(event_type)
        with self._connect() as conn:
            row = conn.execute(query, tuple(parameters)).fetchone()
        return int(row["count"]) if row else 0

    def save_v2_agent_result(self, result: AgentResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_agent_results (result_id, event_id, agent_name, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(result_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    agent_name = excluded.agent_name,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    result.result_id,
                    result.event_id,
                    result.agent_name.value,
                    result.created_at.isoformat(),
                    result.model_dump_json(),
                ),
            )

    def list_v2_agent_results(self, event_id: str, *, limit: int = 30) -> list[AgentResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_agent_results
                WHERE event_id = ?
                ORDER BY created_at DESC, result_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [AgentResult.model_validate_json(row["payload"]) for row in rows]

    def save_v2_event_shared_memory(self, snapshot: SharedMemorySnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_event_shared_memory (event_id, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    snapshot.event_id,
                    snapshot.updated_at.isoformat(),
                    snapshot.model_dump_json(),
                ),
            )

    def get_v2_event_shared_memory(self, event_id: str) -> SharedMemorySnapshot | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM v2_event_shared_memory WHERE event_id = ?", (event_id,)).fetchone()
        return SharedMemorySnapshot.model_validate_json(row["payload"]) if row else None

    def save_v2_experience_record(self, record: ExperienceRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_experience_records (
                    experience_id, event_id, entity_id, entity_type, risk_level, action_type, created_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experience_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    entity_id = excluded.entity_id,
                    entity_type = excluded.entity_type,
                    risk_level = excluded.risk_level,
                    action_type = excluded.action_type,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.experience_id,
                    record.event_id,
                    record.entity_id,
                    record.entity_type,
                    record.risk_level.value if record.risk_level else None,
                    record.action_type,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def save_v2_daily_report(self, record: DailyReportRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_daily_reports (report_id, event_id, report_date, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    report_date = excluded.report_date,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.report_id,
                    record.event_id,
                    record.report_date.isoformat(),
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_daily_reports(self, event_id: str, *, limit: int = 30) -> list[DailyReportRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_daily_reports
                WHERE event_id = ?
                ORDER BY report_date DESC, created_at DESC, report_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [DailyReportRecord.model_validate_json(row["payload"]) for row in rows]

    def get_v2_daily_report_by_event_and_date(self, event_id: str, report_date: str) -> DailyReportRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_daily_reports
                WHERE event_id = ? AND report_date = ?
                ORDER BY created_at DESC, report_id DESC
                LIMIT 1
                """,
                (event_id, report_date),
            ).fetchone()
        return DailyReportRecord.model_validate_json(row["payload"]) if row else None

    def save_v2_daily_report_run(self, record: DailyReportRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_daily_report_runs (run_id, event_id, report_date, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    report_date = excluded.report_date,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.run_id,
                    record.event_id,
                    record.report_date.isoformat(),
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def get_v2_daily_report_run(self, event_id: str, report_date: str) -> DailyReportRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_daily_report_runs
                WHERE event_id = ? AND report_date = ?
                ORDER BY created_at DESC, run_id DESC
                LIMIT 1
                """,
                (event_id, report_date),
            ).fetchone()
        return DailyReportRunRecord.model_validate_json(row["payload"]) if row else None

    def save_v2_high_risk_episode(self, record: HighRiskEpisodeRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_high_risk_episodes (episode_id, event_id, started_at, status, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    started_at = excluded.started_at,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    record.episode_id,
                    record.event_id,
                    record.started_at.isoformat(),
                    record.status.value,
                    record.model_dump_json(),
                ),
            )

    def list_v2_high_risk_episodes(self, event_id: str, *, limit: int = 20) -> list[HighRiskEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_high_risk_episodes
                WHERE event_id = ?
                ORDER BY started_at DESC, episode_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [HighRiskEpisodeRecord.model_validate_json(row["payload"]) for row in rows]

    def list_open_v2_high_risk_episodes(self, event_id: str) -> list[HighRiskEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_high_risk_episodes
                WHERE event_id = ? AND status = 'open'
                ORDER BY started_at DESC, episode_id DESC
                """,
                (event_id,),
            ).fetchall()
        return [HighRiskEpisodeRecord.model_validate_json(row["payload"]) for row in rows]

    def list_pending_v2_episode_summaries(self, *, limit: int = 50) -> list[HighRiskEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.payload
                FROM v2_high_risk_episodes e
                LEFT JOIN v2_event_episode_summaries s ON s.episode_id = e.episode_id
                WHERE e.status = 'closed' AND s.summary_id IS NULL
                ORDER BY e.started_at ASC, e.episode_id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [HighRiskEpisodeRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_event_episode_summary(self, record: EventEpisodeSummaryRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_event_episode_summaries (summary_id, episode_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(summary_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    event_id = excluded.event_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.summary_id,
                    record.episode_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_event_episode_summaries(self, event_id: str, *, limit: int = 20) -> list[EventEpisodeSummaryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_event_episode_summaries
                WHERE event_id = ?
                ORDER BY created_at DESC, summary_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [EventEpisodeSummaryRecord.model_validate_json(row["payload"]) for row in rows]

    def get_v2_event_episode_summary_by_episode(self, episode_id: str) -> EventEpisodeSummaryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_event_episode_summaries
                WHERE episode_id = ?
                ORDER BY created_at DESC, summary_id DESC
                LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
        return EventEpisodeSummaryRecord.model_validate_json(row["payload"]) if row else None

    def save_v2_long_term_memory(self, record: LongTermMemoryRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_long_term_memories (memory_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.memory_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_long_term_memories(
        self,
        *,
        event_id: str | None = None,
        limit: int = 100,
    ) -> list[LongTermMemoryRecord]:
        query = "SELECT payload FROM v2_long_term_memories WHERE 1 = 1"
        params: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            params.append(event_id)
        query += " ORDER BY created_at DESC, memory_id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [LongTermMemoryRecord.model_validate_json(row["payload"]) for row in rows]

    def list_v2_experience_records(
        self,
        *,
        event_id: str | None = None,
        entity_id: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        query = "SELECT payload FROM v2_experience_records WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        if entity_id is not None:
            query += " AND entity_id = ?"
            parameters.append(entity_id)
        if entity_type is not None:
            query += " AND entity_type = ?"
            parameters.append(entity_type)
        query += " ORDER BY created_at DESC, experience_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(parameters)).fetchall()
        return [ExperienceRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_evaluation_report(self, report: EvaluationReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_evaluation_reports (report_id, created_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (report.report_id, report.created_at.isoformat(), report.model_dump_json()),
            )

    def get_v2_evaluation_report(self, report_id: str) -> EvaluationReport | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_evaluation_reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        return EvaluationReport.model_validate_json(row["payload"]) if row else None

    def list_v2_evaluation_reports(self, *, limit: int = 20) -> list[EvaluationReport]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_evaluation_reports
                ORDER BY created_at DESC, report_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [EvaluationReport.model_validate_json(row["payload"]) for row in rows]

    def save_v2_supervisor_run(self, record: SupervisorRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_supervisor_runs (supervisor_run_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(supervisor_run_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.supervisor_run_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_supervisor_runs(self, event_id: str, *, limit: int = 20) -> list[SupervisorRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_supervisor_runs
                WHERE event_id = ?
                ORDER BY created_at DESC, supervisor_run_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [SupervisorRunRecord.model_validate_json(row["payload"]) for row in rows]

    def list_v2_agent_timeline(self, event_id: str, *, limit: int = 120) -> list[AgentTimelineEntry]:
        task_events = self.list_v2_agent_task_events(event_id, limit=limit)
        trigger_events = self.list_v2_trigger_events(event_id, limit=limit)
        entries: list[AgentTimelineEntry] = []
        for trigger in trigger_events:
            summary = f"Trigger {trigger.trigger_type.value} is {trigger.status.value}."
            entries.append(
                AgentTimelineEntry(
                    entry_id=f"trigger_{trigger.trigger_id}",
                    event_id=event_id,
                    entry_type="trigger",
                    trigger_id=trigger.trigger_id,
                    trigger_type=trigger.trigger_type,
                    summary=summary,
                    payload=trigger.payload | {"status": trigger.status.value, "error_message": trigger.error_message},
                    created_at=trigger.created_at,
                )
            )
        for task_event in task_events:
            summary = task_event.payload.get("summary") if isinstance(task_event.payload.get("summary"), str) else task_event.event_type.value
            entries.append(
                AgentTimelineEntry(
                    entry_id=f"task_{task_event.task_event_id}",
                    event_id=event_id,
                    entry_type="task_event",
                    task_id=task_event.task_id,
                    task_event_type=task_event.event_type,
                    trigger_id=task_event.trigger_id,
                    agent_name=task_event.agent_name,
                    summary=summary,
                    payload=task_event.payload,
                    created_at=task_event.created_at,
                )
            )
        entries.sort(key=lambda item: item.created_at, reverse=True)
        return entries[:limit]

    def save_supervisor_health_state(self, state: SupervisorHealthState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_supervisor_health_state (component_key, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(component_key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (state.component_key, state.updated_at.isoformat(), state.model_dump_json()),
            )

    def get_supervisor_health_state(self, component_key: str = "supervisor_loop") -> SupervisorHealthState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_supervisor_health_state WHERE component_key = ?",
                (component_key,),
            ).fetchone()
        return SupervisorHealthState.model_validate_json(row["payload"]) if row else None

    def save_operational_alert(self, alert: OperationalAlert) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_operational_alerts (
                    alert_id, source_type, severity, status, event_id, first_seen_at, last_seen_at, resolved_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alert_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    severity = excluded.severity,
                    status = excluded.status,
                    event_id = excluded.event_id,
                    first_seen_at = excluded.first_seen_at,
                    last_seen_at = excluded.last_seen_at,
                    resolved_at = excluded.resolved_at,
                    payload = excluded.payload
                """,
                (
                    alert.alert_id,
                    alert.source_type,
                    alert.severity.value,
                    alert.status.value,
                    alert.event_id,
                    alert.first_seen_at.isoformat(),
                    alert.last_seen_at.isoformat(),
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.model_dump_json(),
                ),
            )

    def list_operational_alerts(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        status: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 50,
    ) -> list[OperationalAlert]:
        query = "SELECT payload FROM v2_operational_alerts WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        if severity is not None:
            query += " AND severity = ?"
            parameters.append(severity)
        if source_type is not None:
            query += " AND source_type = ?"
            parameters.append(source_type)
        if status is not None:
            query += " AND status = ?"
            parameters.append(status)
        if from_ts is not None:
            query += " AND last_seen_at >= ?"
            parameters.append(from_ts)
        if to_ts is not None:
            query += " AND last_seen_at <= ?"
            parameters.append(to_ts)
        query += " ORDER BY last_seen_at DESC, alert_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [OperationalAlert.model_validate_json(row["payload"]) for row in rows]

    def add_audit_record(self, record: AuditRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_audit_records (audit_id, source_type, severity, event_id, session_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(audit_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    severity = excluded.severity,
                    event_id = excluded.event_id,
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.audit_id,
                    record.source_type,
                    record.severity.value,
                    record.event_id,
                    record.session_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_audit_records(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        query = "SELECT payload FROM v2_audit_records WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        if severity is not None:
            query += " AND severity = ?"
            parameters.append(severity)
        if source_type is not None:
            query += " AND source_type = ?"
            parameters.append(source_type)
        if from_ts is not None:
            query += " AND created_at >= ?"
            parameters.append(from_ts)
        if to_ts is not None:
            query += " AND created_at <= ?"
            parameters.append(to_ts)
        query += " ORDER BY created_at DESC, audit_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [AuditRecord.model_validate_json(row["payload"]) for row in rows]

    def save_archive_run(self, record: ArchiveRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_archive_runs (archive_run_id, started_at, completed_at, status, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(archive_run_id) DO UPDATE SET
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    record.archive_run_id,
                    record.started_at.isoformat(),
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.status,
                    record.model_dump_json(),
                ),
            )

    def list_archive_runs(self, *, limit: int = 10) -> list[ArchiveRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_archive_runs
                ORDER BY started_at DESC, archive_run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [ArchiveRunRecord.model_validate_json(row["payload"]) for row in rows]

    def archive_operational_records(
        self,
        *,
        now: datetime | None = None,
        hot_retention_days: int = 14,
        archive_retention_days: int = 180,
    ) -> ArchiveStatusView:
        now_dt = now or datetime.now(timezone.utc)
        hot_cutoff = now_dt.timestamp() - (hot_retention_days * 86400)
        archive_cutoff = now_dt.timestamp() - (archive_retention_days * 86400)
        archive_id_seed = int(now_dt.timestamp())
        archived_count = 0
        deleted_count = 0
        touched_tables: list[str] = []
        hot_tables = {
            "v2_agent_tasks": ("task_id", None, None, "agent_task"),
            "v2_agent_results": ("result_id", None, None, "agent_result"),
            "v2_copilot_memory_events": ("memory_event_id", None, "session_id", "memory_event"),
            "v2_copilot_tool_executions": ("execution_id", None, "session_id", "tool_execution"),
            "v2_supervisor_runs": ("supervisor_run_id", "event_id", None, "supervisor_run"),
            "v2_trigger_events": ("trigger_id", "event_id", None, "trigger_event"),
            "v2_agent_task_events": ("task_event_id", "event_id", None, "agent_task_event"),
        }

        with self._connect() as conn:
            for table_name, (id_column, event_column, session_column, record_kind) in hot_tables.items():
                rows = conn.execute(f"SELECT payload FROM {table_name}").fetchall()
                stale_payloads = []
                for row in rows:
                    payload = json.loads(row["payload"])
                    created_at = payload.get("created_at")
                    if not created_at:
                        continue
                    try:
                        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if created_dt.timestamp() >= hot_cutoff:
                        continue
                    stale_payloads.append(payload)
                if not stale_payloads:
                    continue
                touched_tables.append(table_name)
                for index, payload in enumerate(stale_payloads, start=1):
                    source_id = str(payload.get(id_column))
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO v2_archived_records (
                            archive_id, source_table, source_id, event_id, session_id, record_kind, created_at, archived_at, payload
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"arch_{archive_id_seed}_{table_name}_{index}",
                            table_name,
                            source_id,
                            payload.get(event_column) if event_column else None,
                            payload.get(session_column) if session_column else None,
                            record_kind,
                            payload["created_at"],
                            now_dt.isoformat(),
                            json.dumps(payload),
                        ),
                    )
                    conn.execute(f"DELETE FROM {table_name} WHERE {id_column} = ?", (source_id,))
                    archived_count += 1

            archived_rows = conn.execute(
                "SELECT archive_id, created_at FROM v2_archived_records"
            ).fetchall()
            for row in archived_rows:
                try:
                    created_dt = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if created_dt.timestamp() < archive_cutoff:
                    conn.execute("DELETE FROM v2_archived_records WHERE archive_id = ?", (row["archive_id"],))
                    deleted_count += 1

            hot_record_count = 0
            for table_name in hot_tables:
                count_row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
                hot_record_count += int(count_row["count"]) if count_row else 0
            archived_record_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM v2_archived_records").fetchone()["count"]
            )

        run = ArchiveRunRecord(
            archive_run_id=f"archive_{archive_id_seed}",
            status="completed",
            hot_records_archived=archived_count,
            expired_archives_deleted=deleted_count,
            tables_touched=touched_tables,
            started_at=now_dt,
            completed_at=datetime.now(timezone.utc),
        )
        self.save_archive_run(run)
        return ArchiveStatusView(
            hot_retention_days=hot_retention_days,
            archive_retention_days=archive_retention_days,
            hot_record_count=hot_record_count,
            archived_record_count=archived_record_count,
            last_archive_run=run,
            latest_archive_runs=self.list_archive_runs(limit=5),
        )

    def get_archive_status(
        self,
        *,
        hot_retention_days: int = 14,
        archive_retention_days: int = 180,
    ) -> ArchiveStatusView:
        hot_tables = [
            "v2_agent_tasks",
            "v2_agent_results",
            "v2_copilot_memory_events",
            "v2_copilot_tool_executions",
            "v2_supervisor_runs",
            "v2_trigger_events",
            "v2_agent_task_events",
        ]
        with self._connect() as conn:
            hot_record_count = 0
            for table_name in hot_tables:
                count_row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
                hot_record_count += int(count_row["count"]) if count_row else 0
            archived_record_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM v2_archived_records").fetchone()["count"]
            )
        latest_runs = self.list_archive_runs(limit=5)
        return ArchiveStatusView(
            hot_retention_days=hot_retention_days,
            archive_retention_days=archive_retention_days,
            hot_record_count=hot_record_count,
            archived_record_count=archived_record_count,
            last_archive_run=latest_runs[0] if latest_runs else None,
            latest_archive_runs=latest_runs,
        )

    def save_v2_notification_draft(self, draft: NotificationDraft) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_notification_drafts (draft_id, event_id, proposal_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(draft_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    proposal_id = excluded.proposal_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (draft.draft_id, draft.event_id, draft.proposal_id, draft.created_at.isoformat(), draft.model_dump_json()),
            )

    def list_v2_notification_drafts(self, event_id: str) -> list[NotificationDraft]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_notification_drafts
                WHERE event_id = ?
                ORDER BY created_at DESC, draft_id DESC
                """,
                (event_id,),
            ).fetchall()
        return [NotificationDraft.model_validate_json(row["payload"]) for row in rows]

    def save_v2_execution_log(self, entry: ExecutionLogEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_execution_logs (log_id, event_id, proposal_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(log_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    proposal_id = excluded.proposal_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (entry.log_id, entry.event_id, entry.proposal_id, entry.created_at.isoformat(), entry.model_dump_json()),
            )

    def list_v2_execution_logs(self, event_id: str) -> list[ExecutionLogEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_execution_logs
                WHERE event_id = ?
                ORDER BY created_at DESC, log_id DESC
                """,
                (event_id,),
            ).fetchall()
        return [ExecutionLogEntry.model_validate_json(row["payload"]) for row in rows]
