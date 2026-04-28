from __future__ import annotations

REPOSITORY_SCHEMA_SQL = """
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

CREATE TABLE IF NOT EXISTS v3_audience_warnings (
    warning_id TEXT PRIMARY KEY,
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
CREATE INDEX IF NOT EXISTS idx_v3_audience_warnings_event_id ON v3_audience_warnings(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_v3_audience_warnings_proposal_id ON v3_audience_warnings(proposal_id, created_at);
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
