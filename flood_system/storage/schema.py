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

CREATE TABLE IF NOT EXISTS response_events (
    event_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_alert_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(event_id, version)
);

CREATE TABLE IF NOT EXISTS response_event_objects (
    event_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(event_id, object_id)
);

CREATE TABLE IF NOT EXISTS response_risk_object_versions (
    snapshot_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(event_id, object_id, version)
);

CREATE TABLE IF NOT EXISTS response_risk_object_registry (
    area_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    registry_version INTEGER NOT NULL,
    registry_status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(area_id, object_id)
);

CREATE TABLE IF NOT EXISTS response_risk_object_registry_versions (
    snapshot_id TEXT PRIMARY KEY,
    area_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    registry_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(area_id, object_id, registry_version)
);

CREATE TABLE IF NOT EXISTS response_risk_object_imports (
    import_id TEXT PRIMARY KEY,
    area_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_format TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_tasks (
    task_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_task_versions (
    snapshot_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(task_id, version)
);

CREATE TABLE IF NOT EXISTS response_task_assignments (
    assignment_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    assignment_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(task_id, assignment_version)
);

CREATE TABLE IF NOT EXISTS response_approvals (
    approval_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_feedback (
    feedback_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_escalations (
    escalation_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_timeline (
    entry_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT,
    object_id TEXT,
    entry_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_review_drafts (
    review_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_scenario_reports (
    report_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_database_backups (
    backup_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_backup_restores (
    restore_id TEXT PRIMARY KEY,
    backup_id TEXT NOT NULL,
    restored_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_backup_imports (
    backup_id TEXT PRIMARY KEY,
    imported_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_backup_retention_runs (
    run_id TEXT PRIMARY KEY,
    executed_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_audit_archives (
    archive_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_identity_nonces (
    nonce TEXT PRIMARY KEY,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_idempotency_records (
    record_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(scope, idempotency_key)
);

CREATE TABLE IF NOT EXISTS response_key_rotation_state (
    rotation_id TEXT PRIMARY KEY,
    backup_id TEXT NOT NULL,
    old_key_id TEXT NOT NULL,
    new_key_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS response_feature_flags (
    flag_key TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(flag_key, scope_key)
);

CREATE TABLE IF NOT EXISTS response_outbox (
    message_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_dispatch_callbacks (
    callback_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    sequence_state TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY(message_id) REFERENCES response_outbox(message_id)
);

CREATE TABLE IF NOT EXISTS response_candidate_runs (
    run_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_evidence_packages (
    package_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(package_id, version)
);

CREATE TABLE IF NOT EXISTS response_rule_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_version INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_deadline_extensions (
    extension_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_schema_migrations (
    version TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_migration_batches (
    batch_id TEXT PRIMARY KEY,
    mapping_version TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_legacy_adapter_calls (
    call_id TEXT PRIMARY KEY,
    legacy_endpoint TEXT NOT NULL,
    mapping_version TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_document_versions (
    version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    source_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(document_id, version_number),
    UNIQUE(document_id, source_hash)
);

DROP TRIGGER IF EXISTS protect_response_registry_versions_update;
CREATE TRIGGER protect_response_registry_versions_update
BEFORE UPDATE ON response_risk_object_registry_versions
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'risk-object registry versions are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_registry_versions_delete;
CREATE TRIGGER protect_response_registry_versions_delete
BEFORE DELETE ON response_risk_object_registry_versions BEGIN
    SELECT RAISE(ABORT, 'risk-object registry versions cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_registry_imports_update;
CREATE TRIGGER protect_response_registry_imports_update
BEFORE UPDATE ON response_risk_object_imports
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'risk-object registry imports are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_registry_imports_delete;
CREATE TRIGGER protect_response_registry_imports_delete
BEFORE DELETE ON response_risk_object_imports BEGIN
    SELECT RAISE(ABORT, 'risk-object registry imports cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_alert_snapshots_update;
CREATE TRIGGER protect_response_alert_snapshots_update
BEFORE UPDATE ON response_alert_snapshots
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'response alert snapshots are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_alert_snapshots_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_alert_snapshots_delete
BEFORE DELETE ON response_alert_snapshots BEGIN
    SELECT RAISE(ABORT, 'response alert snapshots cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_approvals_update;
CREATE TRIGGER protect_response_approvals_update
BEFORE UPDATE ON response_approvals
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'response approvals are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_approvals_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_approvals_delete
BEFORE DELETE ON response_approvals BEGIN
    SELECT RAISE(ABORT, 'response approvals cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_feedback_update;
CREATE TRIGGER protect_response_feedback_update
BEFORE UPDATE ON response_feedback
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'response evidence feedback is immutable');
END;
DROP TRIGGER IF EXISTS protect_response_feedback_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_feedback_delete
BEFORE DELETE ON response_feedback BEGIN
    SELECT RAISE(ABORT, 'response evidence feedback cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_timeline_update;
CREATE TRIGGER protect_response_timeline_update
BEFORE UPDATE ON response_timeline
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'response timeline is append-only');
END;
DROP TRIGGER IF EXISTS protect_response_timeline_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_timeline_delete
BEFORE DELETE ON response_timeline BEGIN
    SELECT RAISE(ABORT, 'response timeline cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_task_versions_update;
CREATE TRIGGER protect_response_task_versions_update
BEFORE UPDATE ON response_task_versions
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'response task versions are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_task_versions_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_task_versions_delete
BEFORE DELETE ON response_task_versions BEGIN
    SELECT RAISE(ABORT, 'response task versions cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_task_assignments_update;
CREATE TRIGGER protect_response_task_assignments_update
BEFORE UPDATE ON response_task_assignments
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'response task assignments are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_task_assignments_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_task_assignments_delete
BEFORE DELETE ON response_task_assignments BEGIN
    SELECT RAISE(ABORT, 'response task assignments cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_audit_archives_update;
CREATE TRIGGER protect_response_audit_archives_update
BEFORE UPDATE ON response_audit_archives
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'response audit archives are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_audit_archives_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_audit_archives_delete
BEFORE DELETE ON response_audit_archives BEGIN
    SELECT RAISE(ABORT, 'response audit archives cannot be deleted');
END;
DROP TRIGGER IF EXISTS protect_response_backup_retention_runs_update;
CREATE TRIGGER protect_response_backup_retention_runs_update
BEFORE UPDATE ON response_backup_retention_runs
WHEN NOT (
    COALESCE(json_extract(OLD.payload, '$.protected'), 0) = 0
    AND json_extract(NEW.payload, '$.protected') = 1
    OR response_key_rotation_authorized() = 1
)
BEGIN
    SELECT RAISE(ABORT, 'backup retention runs are immutable');
END;
DROP TRIGGER IF EXISTS protect_response_backup_retention_runs_delete;
CREATE TRIGGER IF NOT EXISTS protect_response_backup_retention_runs_delete
BEFORE DELETE ON response_backup_retention_runs BEGIN
    SELECT RAISE(ABORT, 'backup retention runs cannot be deleted');
END;

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
CREATE INDEX IF NOT EXISTS idx_response_alerts_event_id ON response_alert_snapshots(event_id, version);
CREATE INDEX IF NOT EXISTS idx_response_objects_event_id ON response_event_objects(event_id, verification_status);
CREATE INDEX IF NOT EXISTS idx_response_object_versions ON response_risk_object_versions(event_id, object_id, version);
CREATE INDEX IF NOT EXISTS idx_response_registry_area_status ON response_risk_object_registry(area_id, registry_status, updated_at);
CREATE INDEX IF NOT EXISTS idx_response_registry_versions ON response_risk_object_registry_versions(area_id, object_id, registry_version);
CREATE INDEX IF NOT EXISTS idx_response_risk_imports ON response_risk_object_imports(area_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_tasks_event_id ON response_tasks(event_id, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_response_task_versions_task_id ON response_task_versions(task_id, version);
CREATE INDEX IF NOT EXISTS idx_response_task_assignments_task_id ON response_task_assignments(task_id, assignment_version);
CREATE INDEX IF NOT EXISTS idx_response_approvals_task_id ON response_approvals(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_feedback_task_id ON response_feedback(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_escalations_event_id ON response_escalations(event_id, resolved, created_at);
CREATE INDEX IF NOT EXISTS idx_response_timeline_event_id ON response_timeline(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_review_event_id ON response_review_drafts(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_scenario_report_event_id ON response_scenario_reports(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_backup_created_at ON response_database_backups(created_at);
CREATE INDEX IF NOT EXISTS idx_response_audit_archives_event ON response_audit_archives(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_restore_backup_id ON response_backup_restores(backup_id, restored_at);
CREATE INDEX IF NOT EXISTS idx_response_backup_imported_at ON response_backup_imports(imported_at);
CREATE INDEX IF NOT EXISTS idx_response_identity_nonce_expiry ON response_identity_nonces(expires_at);
CREATE INDEX IF NOT EXISTS idx_response_idempotency_expiry ON response_idempotency_records(expires_at);
CREATE INDEX IF NOT EXISTS idx_response_idempotency_status ON response_idempotency_records(status, created_at);
CREATE INDEX IF NOT EXISTS idx_response_feature_flags_scope ON response_feature_flags(scope_key, flag_key);
CREATE INDEX IF NOT EXISTS idx_response_outbox_status ON response_outbox(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_response_outbox_event ON response_outbox(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_candidate_runs_event ON response_candidate_runs(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_response_evidence_packages_event ON response_evidence_packages(event_id, object_id, version);
CREATE INDEX IF NOT EXISTS idx_response_rule_evaluations_task ON response_rule_evaluations(task_id, task_version, created_at);
CREATE INDEX IF NOT EXISTS idx_response_deadline_extensions_task ON response_deadline_extensions(task_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_response_migration_batches_version ON response_migration_batches(mapping_version, created_at);
CREATE INDEX IF NOT EXISTS idx_response_legacy_calls_endpoint ON response_legacy_adapter_calls(legacy_endpoint, created_at);
CREATE INDEX IF NOT EXISTS idx_response_document_versions ON response_document_versions(document_id, version_number);
CREATE INDEX IF NOT EXISTS idx_response_dispatch_callbacks
    ON response_dispatch_callbacks(message_id, external_id, version, created_at);
"""
