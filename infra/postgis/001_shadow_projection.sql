CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS flood_simulation;

CREATE TABLE IF NOT EXISTS flood_simulation.schema_migrations (
    version TEXT PRIMARY KEY,
    checksum CHAR(64) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flood_simulation.migration_batches (
    batch_id TEXT PRIMARY KEY,
    mapping_version TEXT NOT NULL,
    source_instance_id TEXT NOT NULL,
    source_logical_sha256 CHAR(64) NOT NULL,
    schema_sha256 CHAR(64) NOT NULL,
    status TEXT NOT NULL,
    source_counts JSONB NOT NULL,
    target_counts JSONB NOT NULL,
    table_hashes JSONB NOT NULL,
    quarantine_count INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS flood_simulation.shadow_records (
    record_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_table TEXT NOT NULL,
    event_id TEXT,
    area_id TEXT,
    object_id TEXT,
    task_id TEXT,
    version INTEGER,
    source_status TEXT,
    source_created_at TIMESTAMPTZ,
    payload_ciphertext TEXT NOT NULL,
    payload_sha256 CHAR(64) NOT NULL,
    canonical_payload_sha256 CHAR(64) NOT NULL,
    is_simulated BOOLEAN NOT NULL CHECK (is_simulated),
    affected_geometry geometry(Polygon, 4326),
    object_location geometry(Point, 4326),
    migration_batch TEXT NOT NULL REFERENCES flood_simulation.migration_batches(batch_id),
    mapping_version TEXT NOT NULL,
    migrated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (record_type, source_id)
);

ALTER TABLE flood_simulation.shadow_records
    ADD COLUMN IF NOT EXISTS area_id TEXT;
ALTER TABLE flood_simulation.shadow_records
    ADD COLUMN IF NOT EXISTS object_location geometry(Point, 4326);

CREATE INDEX IF NOT EXISTS idx_shadow_records_event
    ON flood_simulation.shadow_records(event_id, record_type);
CREATE INDEX IF NOT EXISTS idx_shadow_records_object
    ON flood_simulation.shadow_records(object_id, record_type);
CREATE INDEX IF NOT EXISTS idx_shadow_records_area
    ON flood_simulation.shadow_records(area_id, record_type);
CREATE INDEX IF NOT EXISTS idx_shadow_records_task
    ON flood_simulation.shadow_records(task_id, record_type);
CREATE INDEX IF NOT EXISTS idx_shadow_records_geometry
    ON flood_simulation.shadow_records USING GIST(affected_geometry);
CREATE INDEX IF NOT EXISTS idx_shadow_records_object_location
    ON flood_simulation.shadow_records USING GIST(object_location);

CREATE TABLE IF NOT EXISTS flood_simulation.migration_quarantine (
    quarantine_id BIGSERIAL PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES flood_simulation.migration_batches(batch_id),
    source_table TEXT NOT NULL,
    source_id TEXT NOT NULL,
    payload_sha256 CHAR(64) NOT NULL,
    error_code TEXT NOT NULL,
    error_detail TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(batch_id, source_table, source_id)
);

CREATE TABLE IF NOT EXISTS flood_simulation.reconciliation_runs (
    reconciliation_id BIGSERIAL PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES flood_simulation.migration_batches(batch_id),
    source_table TEXT NOT NULL,
    record_type TEXT NOT NULL,
    source_count INTEGER NOT NULL,
    target_count INTEGER NOT NULL,
    source_hash CHAR(64) NOT NULL,
    target_hash CHAR(64) NOT NULL,
    passed BOOLEAN NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(batch_id, record_type)
);

CREATE OR REPLACE VIEW flood_simulation.response_events AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'response_event';
CREATE OR REPLACE VIEW flood_simulation.alert_snapshots AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'alert_snapshot';
CREATE OR REPLACE VIEW flood_simulation.risk_objects AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'risk_object';
CREATE OR REPLACE VIEW flood_simulation.risk_object_versions AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'risk_object_version';
CREATE OR REPLACE VIEW flood_simulation.risk_object_registry AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'risk_object_registry';
CREATE OR REPLACE VIEW flood_simulation.risk_object_registry_versions AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'risk_object_registry_version';
CREATE OR REPLACE VIEW flood_simulation.risk_object_registry_imports AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'risk_object_registry_import';
CREATE OR REPLACE VIEW flood_simulation.tasks AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'task';
CREATE OR REPLACE VIEW flood_simulation.task_versions AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'task_version';
CREATE OR REPLACE VIEW flood_simulation.approvals AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'approval';
CREATE OR REPLACE VIEW flood_simulation.feedback AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'feedback';
CREATE OR REPLACE VIEW flood_simulation.evidence_packages AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'evidence_package';
CREATE OR REPLACE VIEW flood_simulation.timeline AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'timeline';
CREATE OR REPLACE VIEW flood_simulation.outbox_messages AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'outbox';
CREATE OR REPLACE VIEW flood_simulation.dispatch_callbacks AS
    SELECT * FROM flood_simulation.shadow_records WHERE record_type = 'dispatch_callback';
