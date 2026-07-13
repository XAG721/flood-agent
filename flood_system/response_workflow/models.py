from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkflowModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class OperatorRole(StrEnum):
    DUTY_OFFICER = "duty_officer"
    REVIEWER = "reviewer"
    COMMANDER = "commander"
    LIAISON = "liaison"
    FIELD_OPERATOR = "field_operator"
    AUDITOR = "auditor"
    ADMIN = "admin"
    EXTERNAL_SERVICE = "external_service"


class EventStatus(StrEnum):
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


class AlertLifecycleStatus(StrEnum):
    ACTIVE = "active"
    UPDATED = "updated"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ObjectVerificationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXCLUDED = "excluded"


class TaskStatus(StrEnum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    ISSUED = "issued"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    PENDING_VERIFICATION = "pending_verification"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    WAIVED = "waived"
    BLOCKED = "blocked"
    PARTIALLY_COMPLETED = "partially_completed"
    CANCELLED = "cancelled"
    TAKEN_OVER = "taken_over"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    PARTIALLY_SENT = "partially_sent"
    FAILED = "failed"
    MANUAL_TAKEOVER = "manual_takeover"


class IdempotencyStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    INDETERMINATE = "indeterminate"


class DispatchCallbackStatus(StrEnum):
    ACCEPTED = "accepted"
    DELIVERED = "delivered"
    PARTIAL_SUCCESS = "partial_success"
    REJECTED = "rejected"


class CallbackSequenceState(StrEnum):
    IN_ORDER = "in_order"
    OUT_OF_ORDER = "out_of_order"


class ApprovalPolicy(StrEnum):
    REVIEWER_REQUIRED = "reviewer_required"
    COMMANDER_REQUIRED = "commander_required"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class EvidenceRole(StrEnum):
    CONDITION = "condition"
    OBJECT = "object"
    RESPONSIBILITY = "responsibility"
    PROCEDURE = "procedure"
    EXCEPTION = "exception"
    ATTRIBUTION = "attribution"


class EvidenceFieldState(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONFLICTED = "CONFLICTED"
    MISSING = "MISSING"


class RetrievalMode(StrEnum):
    BASELINE_ONLY = "BASELINE_ONLY"
    SHADOW = "SHADOW"
    REVIEW = "REVIEW"
    CANARY = "CANARY"
    DEFAULT = "DEFAULT"


class FeedbackCategory(StrEnum):
    COMPLETION = "completion"
    PARTIAL_COMPLETION = "partial_completion"
    BLOCKED = "blocked"
    RESOURCE_SHORTAGE = "resource_shortage"
    COORDINATION_REQUEST = "coordination_request"
    SITUATION_UPDATE = "situation_update"


class DataClassification(StrEnum):
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    HIGHLY_SENSITIVE = "highly_sensitive"


class RuleOutcome(StrEnum):
    PASS = "PASS"
    SOFT_WARNING = "SOFT_WARNING"
    HARD_BLOCK = "HARD_BLOCK"


class DeadlineExtensionStatus(StrEnum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"


class SensitiveContact(WorkflowModel):
    name: str
    role: str
    phone: str
    classification: DataClassification = DataClassification.HIGHLY_SENSITIVE


class GeoPolygon(WorkflowModel):
    coordinates: list[tuple[float, float]] = Field(min_length=4)
    crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def validate_exterior_ring(self):
        if self.crs != "EPSG:4326":
            raise ValueError("only EPSG:4326 alert geometry is supported")
        if self.coordinates[0] != self.coordinates[-1]:
            raise ValueError("polygon exterior ring must be closed")
        for longitude, latitude in self.coordinates:
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError("polygon coordinate is outside EPSG:4326 bounds")
        if len(set(self.coordinates[:-1])) < 3:
            raise ValueError(
                "polygon exterior ring requires at least three distinct vertices"
            )
        return self


class AlertInput(WorkflowModel):
    alert_id: str
    source_department: str
    disaster_type: str
    level: str
    issued_at: datetime
    valid_until: datetime | None = None
    affected_area: str
    affected_geometry: GeoPolygon | None = None
    raw_content: str
    source_type: str = "simulation"
    source_version: str = "v1"
    is_simulated: bool = True
    lifecycle_status: AlertLifecycleStatus = AlertLifecycleStatus.ACTIVE


class EventCreateRequest(WorkflowModel):
    title: str
    area_id: str
    alert: AlertInput
    operator_id: str
    operator_role: OperatorRole
    terminal_id: str = "unknown-terminal"


class AlertAppendRequest(WorkflowModel):
    alert: AlertInput
    operator_id: str
    operator_role: OperatorRole
    terminal_id: str = "unknown-terminal"


class AlertSnapshot(AlertInput):
    snapshot_id: str
    event_id: str
    version: int
    created_at: datetime
    raw_payload_hash: str = ""
    data_version: str = "warning-v1"


class ResponseEvent(WorkflowModel):
    event_id: str
    title: str
    area_id: str
    status: EventStatus = EventStatus.ACTIVE
    current_alert_version: int = 1
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    closed_by: str | None = None
    workflow_engine_version: str = "response-workflow-v1"
    data_version: str = "response-schema-v1"
    source_type: str = "simulation"
    is_simulated: bool = True


class RiskObjectInput(WorkflowModel):
    object_id: str
    canonical_object_id: str | None = None
    aliases: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    name: str
    object_type: str
    location: str
    longitude: float | None = None
    latitude: float | None = None
    location_classification: DataClassification = DataClassification.RESTRICTED
    responsible_organization: str
    responsible_role: str
    trigger_reasons: list[str] = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    vulnerability: str
    historical_risk: str = ""
    risk_score: float = Field(ge=0, le=100)
    system_explanation: str
    sensitive_contacts: list[SensitiveContact] = Field(default_factory=list)
    special_population_notes: str = ""
    special_population_classification: DataClassification = (
        DataClassification.HIGHLY_SENSITIVE
    )
    source_type: str = "simulation"
    source_version: str = "registry-v1"
    data_version: str = "risk-object-v1"
    is_simulated: bool = True
    missing_fields: list[str] = Field(default_factory=list)
    raw_candidate_score: float | None = Field(default=None, ge=0, le=100)
    calibrated_confidence: float | None = Field(default=None, ge=0, le=1)
    calibration_version: str = "candidate-logistic-v1"
    association_mode: str = "manual"
    registry_status: str = "active"
    registry_valid_from: datetime | None = None
    registry_valid_until: datetime | None = None

    @model_validator(mode="after")
    def validate_registry_coordinates(self):
        if (self.longitude is None) != (self.latitude is None):
            raise ValueError("longitude and latitude must be provided together")
        if self.longitude is not None and not -180 <= self.longitude <= 180:
            raise ValueError("longitude is outside EPSG:4326 bounds")
        if self.latitude is not None and not -90 <= self.latitude <= 90:
            raise ValueError("latitude is outside EPSG:4326 bounds")
        return self


class RiskObjectBatchRequest(WorkflowModel):
    objects: list[RiskObjectInput] = Field(min_length=1)
    operator_id: str
    operator_role: OperatorRole
    terminal_id: str = "unknown-terminal"


class CandidateDiscoveryRequest(WorkflowModel):
    entity_types: list[str] = Field(default_factory=list)
    min_risk_score: float = Field(default=45, ge=0, le=100)
    max_candidates: int = Field(default=20, ge=1, le=200)
    operator_id: str
    operator_role: OperatorRole
    terminal_id: str = "unknown-terminal"


class CandidateDiscoveryResult(WorkflowModel):
    run_id: str = "legacy-candidate-run"
    event_id: str
    alert_snapshot_id: str
    association_mode: str = "area_registry"
    scanned_profiles: int
    matched_profiles: int
    spatially_evaluated_profiles: int = 0
    spatially_matched_profiles: int = 0
    excluded_unlocated_profiles: int = 0
    candidates: list[EventRiskObject]
    limitations: list[str] = Field(default_factory=list)
    algorithm_version: str = "candidate-rules-v1"
    feature_version: str = "candidate-features-v1"
    data_version: str = "risk-object-v1"
    run_status: str = "completed"
    calibration_version: str = "candidate-logistic-v1"


class CandidateRunRecord(WorkflowModel):
    run_id: str
    event_id: str
    alert_snapshot_id: str
    risk_object_data_version: str
    algorithm_version: str
    feature_version: str
    association_mode: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    candidate_object_ids: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    status: str = "completed"
    stale_at: datetime | None = None
    stale_reason: str = ""
    created_by: str
    terminal_id: str
    created_at: datetime


class RiskObjectVerificationRequest(WorkflowModel):
    decision: ObjectVerificationStatus
    note: str = ""
    operator_id: str
    operator_role: OperatorRole
    terminal_id: str = "unknown-terminal"


class EventRiskObject(RiskObjectInput):
    event_id: str
    verification_status: ObjectVerificationStatus = ObjectVerificationStatus.PENDING
    verified_by: str | None = None
    verified_at: datetime | None = None
    verification_note: str = ""
    created_at: datetime
    stale: bool = False
    candidate_run_id: str | None = None
    version: int = 1
    updated_at: datetime | None = None


class RiskObjectVersionSnapshot(WorkflowModel):
    snapshot_id: str
    event_id: str
    object_id: str
    version: int
    object: EventRiskObject
    change_type: str
    changed_by: str
    terminal_id: str
    created_at: datetime


class RiskObjectRegistryRecord(RiskObjectInput):
    area_id: str
    registry_version: int = Field(ge=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_filename: str
    created_by: str
    terminal_id: str
    created_at: datetime
    updated_at: datetime


class RiskObjectRegistryVersionSnapshot(WorkflowModel):
    snapshot_id: str
    area_id: str
    object_id: str
    registry_version: int = Field(ge=1)
    record: RiskObjectRegistryRecord
    change_type: str
    changed_by: str
    terminal_id: str
    created_at: datetime


class PlanBasis(WorkflowModel):
    document: str
    version: str
    clause: str


class TaskEvidenceRef(WorkflowModel):
    source_type: str
    source_id: str
    title: str
    excerpt: str
    roles: list[EvidenceRole] = Field(min_length=1)
    document_version: str | None = None
    clause: str | None = None
    trust_score: float | None = Field(default=None, ge=0, le=1)
    conflict_key: str | None = None
    conflict_value: str | None = None
    jurisdiction: str | None = None
    superseded: bool = False


class EvidenceConflict(WorkflowModel):
    conflict_id: str
    field_name: str
    conflict_type: str
    severity: str
    evidence_source_ids: list[str] = Field(min_length=2)
    resolution_status: str = "unresolved"
    resolution_reason: str = ""
    selected_source_ids: list[str] = Field(default_factory=list)


class EvidencePackageVersion(WorkflowModel):
    package_id: str
    event_id: str
    object_id: str
    version: int = 1
    status: str
    task_schema_version: str = "response-task-schema-v1"
    retrieval_strategy: str = "FRC-RAG"
    retrieval_run_id: str
    field_states: dict[str, EvidenceFieldState]
    role_coverage: dict[str, bool]
    evidence: list[TaskEvidenceRef] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    content_hash: str
    created_by: str
    reviewed_by: str | None = None
    frozen_at: datetime | None = None
    created_at: datetime
    retrieval_mode: RetrievalMode = RetrievalMode.SHADOW
    baseline_source_ids: list[str] = Field(default_factory=list)
    frc_source_ids: list[str] = Field(default_factory=list)
    shadow_comparison: dict[str, Any] = Field(default_factory=dict)


class TaskCreateRequest(WorkflowModel):
    object_id: str
    title: str
    action: str
    responsible_organization: str
    responsible_role: str
    cooperate_roles: list[str] = Field(default_factory=list)
    deadline_at: datetime
    acknowledge_deadline_at: datetime
    start_deadline_at: datetime | None = None
    verification_deadline_at: datetime | None = None
    required_evidence: list[str] = Field(min_length=1)
    plan_basis: list[PlanBasis] = Field(min_length=1)
    approval_policy: ApprovalPolicy = ApprovalPolicy.REVIEWER_REQUIRED
    escalation_rule: str
    dependencies: list[str] = Field(default_factory=list)
    operator_id: str
    operator_role: OperatorRole
    terminal_id: str = "unknown-terminal"
    generated_by_ai: bool = False
    generation_version: str | None = None
    grounding_summary: str = ""
    evidence_role_coverage: dict[str, bool] = Field(default_factory=dict)
    source_evidence: list[TaskEvidenceRef] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    evidence_package_id: str | None = None
    evidence_package_version: int | None = None
    evidence_package_hash: str | None = None
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def validate_deadlines(self):
        if self.start_deadline_at is None:
            self.start_deadline_at = (
                self.acknowledge_deadline_at
                + (self.deadline_at - self.acknowledge_deadline_at) / 3
            )
        if self.verification_deadline_at is None:
            self.verification_deadline_at = (
                self.deadline_at + (self.deadline_at - self.acknowledge_deadline_at) / 4
            )
        if self.acknowledge_deadline_at > self.deadline_at:
            raise ValueError(
                "acknowledge_deadline_at must not be later than deadline_at"
            )
        if (
            not self.acknowledge_deadline_at
            <= self.start_deadline_at
            <= self.deadline_at
        ):
            raise ValueError(
                "start_deadline_at must be between acknowledge and completion deadlines"
            )
        if self.verification_deadline_at < self.deadline_at:
            raise ValueError(
                "verification_deadline_at must not be earlier than completion deadline"
            )
        if self.generated_by_ai and not self.generation_version:
            raise ValueError("AI-generated drafts must include generation_version")
        return self


class TaskUpdateRequest(WorkflowModel):
    title: str | None = None
    action: str | None = None
    responsible_organization: str | None = None
    responsible_role: str | None = None
    cooperate_roles: list[str] | None = None
    deadline_at: datetime | None = None
    acknowledge_deadline_at: datetime | None = None
    required_evidence: list[str] | None = None
    plan_basis: list[PlanBasis] | None = None
    approval_policy: ApprovalPolicy | None = None
    escalation_rule: str | None = None
    dependencies: list[str] | None = None
    grounding_summary: str | None = None
    evidence_role_coverage: dict[str, bool] | None = None
    source_evidence: list[TaskEvidenceRef] | None = None
    validation_warnings: list[str] | None = None
    operator_id: str
    operator_role: OperatorRole
    note: str = ""
    terminal_id: str = "unknown-terminal"
    expected_version: int | None = Field(default=None, ge=1)


class TaskActionRequest(WorkflowModel):
    operator_id: str
    operator_role: OperatorRole
    note: str = ""
    terminal_id: str = "unknown-terminal"
    expected_version: int | None = Field(default=None, ge=1)
    idempotency_key: str | None = None


class IngestionFileEnvelope(WorkflowModel):
    filename: str = Field(min_length=1, max_length=180)
    media_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, repr=False)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RiskObjectRegistryBatchImportRequest(TaskActionRequest):
    area_id: str = Field(min_length=1, max_length=120)
    source_version: str = Field(min_length=1, max_length=120)
    source_filename: str = Field(default="api.json", min_length=1, max_length=180)
    objects: list[RiskObjectInput] = Field(min_length=1, max_length=5000)


class RiskObjectRegistryFileImportRequest(TaskActionRequest):
    area_id: str = Field(min_length=1, max_length=120)
    source_version: str = Field(min_length=1, max_length=120)
    file: IngestionFileEnvelope


class RiskObjectImportQuarantineItem(WorkflowModel):
    source_row: int | None = None
    source_id: str = ""
    reason_code: str
    reason: str


class RiskObjectRegistryImportResult(WorkflowModel):
    import_id: str
    area_id: str
    source_filename: str
    source_format: str
    source_version: str
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_bytes: int = Field(ge=0)
    imported_count: int = Field(ge=0)
    created_count: int = Field(ge=0)
    updated_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    quarantined_count: int = Field(ge=0)
    changed_object_ids: list[str] = Field(default_factory=list)
    affected_event_ids: list[str] = Field(default_factory=list)
    stale_candidate_run_count: int = Field(ge=0)
    quarantine: list[RiskObjectImportQuarantineItem] = Field(default_factory=list)
    is_simulated: bool = False
    imported_by: str
    terminal_id: str
    created_at: datetime


class TaskAssignmentRequest(TaskActionRequest):
    assignee_id: str
    assignee_name: str
    assignee_role: OperatorRole = OperatorRole.FIELD_OPERATOR
    reason: str = "现场执行分派"


class ApprovalRequest(TaskActionRequest):
    decision: ApprovalDecision


class FeedbackRequest(TaskActionRequest):
    summary: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    blocked_reason: str | None = None
    resource_gap: str | None = None
    completion_percent: int | None = Field(default=None, ge=0, le=100)


class ResponseTask(WorkflowModel):
    task_id: str
    event_id: str
    object_id: str
    version: int = 1
    title: str
    action: str
    responsible_organization: str
    responsible_role: str
    cooperate_roles: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.DRAFT
    deadline_at: datetime
    acknowledge_deadline_at: datetime
    start_deadline_at: datetime | None = None
    verification_deadline_at: datetime | None = None
    required_evidence: list[str]
    plan_basis: list[PlanBasis]
    approval_policy: ApprovalPolicy
    escalation_rule: str
    dependencies: list[str] = Field(default_factory=list)
    generated_by_ai: bool = False
    generation_version: str | None = None
    grounding_summary: str = ""
    evidence_role_coverage: dict[str, bool] = Field(default_factory=dict)
    source_evidence: list[TaskEvidenceRef] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    drafted_by: str
    created_at: datetime
    updated_at: datetime
    approved_version: int | None = None
    assignee_id: str | None = None
    assignee_name: str | None = None
    assignee_role: OperatorRole | None = None
    assignment_version: int = 0
    approval_payload_hash: str | None = None
    evidence_package_hash: str | None = None
    evidence_package_id: str | None = None
    evidence_package_version: int | None = None
    rule_set_version: str = "response-rules-v1"
    dispatch_message_id: str | None = None
    data_version: str = "response-schema-v1"
    is_simulated: bool = True
    last_rule_evaluation_id: str | None = None
    effective_start_deadline_at: datetime | None = None
    effective_completion_deadline_at: datetime | None = None
    effective_verification_deadline_at: datetime | None = None
    active_extension_id: str | None = None
    creation_idempotency_key: str | None = None


class RuleCheckResult(WorkflowModel):
    rule_id: str
    outcome: RuleOutcome
    message: str
    field_name: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class RuleEvaluationRecord(WorkflowModel):
    evaluation_id: str
    event_id: str
    task_id: str
    task_version: int
    rule_set_version: str
    overall_outcome: RuleOutcome
    checks: list[RuleCheckResult]
    evaluated_by: str
    terminal_id: str
    created_at: datetime


class DeadlineExtensionRequest(TaskActionRequest):
    proposed_start_deadline_at: datetime | None = None
    proposed_completion_deadline_at: datetime
    proposed_verification_deadline_at: datetime | None = None
    reason: str = Field(min_length=4)


class DeadlineExtensionDecisionRequest(TaskActionRequest):
    approved: bool
    reason: str = Field(min_length=4)


class EvidenceConflictResolutionRequest(TaskActionRequest):
    selected_source_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=4)


class EvidenceManualSupplementRequest(TaskActionRequest):
    evidence: list[TaskEvidenceRef] = Field(min_length=1)
    reason: str = Field(min_length=4)
    freeze_when_complete: bool = False


class EvidenceFreezeRequest(TaskActionRequest):
    reason: str = Field(min_length=4)


class DeadlineExtensionRecord(WorkflowModel):
    extension_id: str
    event_id: str
    task_id: str
    task_version: int
    status: DeadlineExtensionStatus
    original_start_deadline_at: datetime | None = None
    original_completion_deadline_at: datetime
    original_verification_deadline_at: datetime | None = None
    proposed_start_deadline_at: datetime | None = None
    proposed_completion_deadline_at: datetime
    proposed_verification_deadline_at: datetime | None = None
    request_reason: str
    requested_by: str
    requested_by_role: OperatorRole
    decided_by: str | None = None
    decision_reason: str = ""
    created_at: datetime
    decided_at: datetime | None = None


class LegacyMigrationRequest(TaskActionRequest):
    mapping_version: str = "legacy-v2-to-response-v1"
    dry_run: bool = True


class MigrationQuarantineItem(WorkflowModel):
    source_id: str
    reason_code: str
    reason: str
    source_hash: str


class MigrationBatchRecord(WorkflowModel):
    batch_id: str
    source_table: str
    target_domain: str
    mapping_version: str
    dry_run: bool
    source_count: int
    migrated_count: int
    quarantined_count: int
    ignored_count: int
    source_hash: str
    target_hash: str | None = None
    status: str
    quarantine: list[MigrationQuarantineItem] = Field(default_factory=list)
    executed_by: str
    terminal_id: str
    created_at: datetime


class LegacyAdapterCallRecord(WorkflowModel):
    call_id: str
    legacy_endpoint: str
    legacy_resource_id: str | None = None
    mapping_version: str
    trace_id: str
    read_only: bool = True
    result_status: str
    operator_id: str
    terminal_id: str
    created_at: datetime


class DocumentClause(WorkflowModel):
    clause_id: str
    heading: str
    text: str
    page_number: int | None = None


class DocumentImportRequest(TaskActionRequest):
    document_id: str
    title: str
    version_label: str
    issuer: str
    jurisdiction: str
    effective_at: datetime
    expires_at: datetime | None = None
    content: str = Field(min_length=8)
    replaces_version_id: str | None = None


class DocumentVersionRecord(WorkflowModel):
    version_id: str
    document_id: str
    version_number: int
    version_label: str
    title: str
    issuer: str
    jurisdiction: str
    effective_at: datetime
    expires_at: datetime | None = None
    replaces_version_id: str | None = None
    lifecycle_status: str = "active"
    source_hash: str
    clauses: list[DocumentClause]
    index_status: str
    index_version: str
    is_simulated: bool = True
    created_by: str
    terminal_id: str
    created_at: datetime


class TaskVersionSnapshot(WorkflowModel):
    snapshot_id: str
    event_id: str
    task_id: str
    version: int
    task: ResponseTask
    change_type: str
    changed_fields: list[str] = Field(default_factory=list)
    note: str = ""
    created_by: str
    terminal_id: str
    created_at: datetime


class TaskAssignmentRecord(WorkflowModel):
    assignment_id: str
    event_id: str
    task_id: str
    assignment_version: int
    previous_assignee_id: str | None = None
    assignee_id: str
    assignee_name: str
    assignee_role: OperatorRole
    reason: str
    assigned_by: str
    assigned_by_role: OperatorRole
    terminal_id: str
    created_at: datetime


class TimelineEntry(WorkflowModel):
    entry_id: str
    event_id: str
    task_id: str | None = None
    object_id: str | None = None
    entry_type: str
    action: str
    actor_id: str
    actor_role: str
    terminal_id: str = "unknown-terminal"
    before_state: dict[str, Any] = Field(default_factory=dict)
    after_state: dict[str, Any] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str = "GENESIS"
    record_hash: str = ""
    created_at: datetime


class TimelineIntegrityIssue(WorkflowModel):
    entry_id: str
    issue: str


class TimelineIntegrityReport(WorkflowModel):
    event_id: str
    status: str
    total_entries: int
    verified_entries: int
    head_hash: str | None = None
    issues: list[TimelineIntegrityIssue] = Field(default_factory=list)
    verified_at: datetime


class BackupCreateRequest(TaskActionRequest):
    label: str = "manual"


class BackupRetentionRequest(TaskActionRequest):
    keep_latest: int = Field(default=14, ge=1, le=1000)
    max_age_days: int | None = Field(default=90, ge=1, le=36500)
    dry_run: bool = True


class BackupRetentionResult(WorkflowModel):
    run_id: str
    keep_latest: int
    max_age_days: int | None
    dry_run: bool
    protected_backup_ids: list[str] = Field(default_factory=list)
    candidate_backup_ids: list[str] = Field(default_factory=list)
    already_pruned_backup_ids: list[str] = Field(default_factory=list)
    pruned_backup_ids: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    executed_by: str
    terminal_id: str
    executed_at: datetime


class AuditArchiveRequest(TaskActionRequest):
    label: str = "scheduled-audit-archive"
    retention_days: int = Field(default=2555, ge=365, le=36500)


class AuditArchiveRecord(WorkflowModel):
    archive_id: str
    event_id: str
    label: str
    archive_filename: str
    manifest_filename: str
    sha256: str
    size_bytes: int
    encryption_key_id: str
    timeline_entries: int
    timeline_head_hash: str | None = None
    retention_until: datetime
    source_instance_id: str
    signature_algorithm: str = "hmac-sha256"
    manifest_signature: str = ""
    created_by: str
    terminal_id: str
    created_at: datetime


class AuditArchiveVerificationResult(WorkflowModel):
    archive_id: str
    event_id: str
    signature_verified: bool
    sha256_verified: bool
    decryption_verified: bool
    timeline_head_verified: bool
    status: str
    verified_by: str
    terminal_id: str
    verified_at: datetime


class DatabaseBackupRecord(WorkflowModel):
    backup_id: str
    label: str
    database_name: str
    backup_filename: str
    manifest_filename: str
    sha256: str
    size_bytes: int
    integrity_check: str
    encryption_key_id: str
    source_instance_id: str = "legacy-unknown"
    signature_algorithm: str = "hmac-sha256"
    manifest_signature: str = ""
    created_by: str
    terminal_id: str
    created_at: datetime


class BackupRestoreRequest(TaskActionRequest):
    backup_id: str


class BackupImportRequest(TaskActionRequest):
    manifest_filename: str
    backup_filename: str


class BackupImportResult(WorkflowModel):
    backup_id: str
    source_instance_id: str
    manifest_signature_verified: bool
    sha256_verified: bool
    integrity_check: str
    decryption_verified: bool
    status: str
    imported_by: str
    terminal_id: str
    imported_at: datetime


class KeyRotationRequest(TaskActionRequest):
    label: str = "scheduled-rotation"


class KeyRotationResult(WorkflowModel):
    rotation_id: str
    backup_id: str
    old_key_id: str
    new_key_id: str
    records_reencrypted: int
    status: str
    activation_required: bool
    rotated_by: str
    terminal_id: str
    started_at: datetime
    completed_at: datetime


class BackupRestoreResult(WorkflowModel):
    restore_id: str
    backup_id: str
    restored_filename: str
    sha256_verified: bool
    integrity_check: str
    status: str
    restored_by: str
    terminal_id: str
    restored_at: datetime


class ApprovalRecord(WorkflowModel):
    approval_id: str
    event_id: str
    task_id: str
    task_version: int
    decision: ApprovalDecision
    operator_id: str
    operator_role: OperatorRole
    note: str
    created_at: datetime
    task_payload_hash: str = ""
    evidence_package_hash: str = ""
    rule_set_version: str = "response-rules-v1"
    rule_evaluation_id: str | None = None


class FeatureFlagSetting(WorkflowModel):
    flag_key: str
    enabled: bool
    environment: str = "development"
    event_id: str | None = None
    role: OperatorRole | None = None
    scenario: str | None = None
    version: int = 1
    reason: str
    updated_by: str
    terminal_id: str
    updated_at: datetime


class FeatureFlagUpdateRequest(TaskActionRequest):
    flag_key: str
    enabled: bool
    environment: str = "development"
    event_id: str | None = None
    role: OperatorRole | None = None
    scenario: str | None = None
    reason: str = Field(min_length=4)


class IdempotencyRecord(WorkflowModel):
    record_id: str
    scope: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1, max_length=200)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: IdempotencyStatus = IdempotencyStatus.PROCESSING
    response_status: int | None = Field(default=None, ge=100, le=599)
    response_content_type: str | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_body_base64: str = ""
    response_ref: str | None = None
    replayable: bool = True
    expires_at: datetime
    created_at: datetime
    completed_at: datetime | None = None


class OutboxMessage(WorkflowModel):
    message_id: str
    event_id: str
    task_id: str
    destination: str = "simulated://member-unit"
    idempotency_key: str
    payload_hash: str
    approval_id: str
    task_version: int
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    last_error: str | None = None
    simulation_scenario: str = "normal"
    gateway_status: str | None = None
    external_request_id: str | None = None
    trace_id: str | None = None
    callback_count: int = 0
    created_at: datetime
    updated_at: datetime
    sent_at: datetime | None = None


class OutboxProcessRequest(TaskActionRequest):
    message_id: str | None = None
    max_messages: int = Field(default=50, ge=1, le=500)
    simulation_scenario: str = "normal"


class DispatchCallbackRequest(TaskActionRequest):
    message_id: str
    external_id: str
    source: str = "deterministic_simulation_gateway"
    version: int = Field(ge=1)
    event_time: datetime
    received_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str
    trace_id: str
    idempotency_key: str
    status: DispatchCallbackStatus
    error_code: str | None = None
    mock_token: str
    is_simulated: bool = True


class DispatchCallbackRecord(WorkflowModel):
    callback_id: str
    message_id: str
    event_id: str
    task_id: str
    external_id: str
    source: str
    version: int
    event_time: datetime
    received_time: datetime
    request_id: str
    trace_id: str
    idempotency_key: str
    status: DispatchCallbackStatus
    error_code: str | None = None
    sequence_state: CallbackSequenceState = CallbackSequenceState.IN_ORDER
    is_simulated: bool = True
    created_at: datetime


class TaskFeedback(WorkflowModel):
    feedback_id: str
    event_id: str
    task_id: str
    object_id: str
    summary: str
    evidence: list[dict[str, Any]]
    blocked_reason: str | None = None
    resource_gap: str | None = None
    category: FeedbackCategory = FeedbackCategory.SITUATION_UPDATE
    classification_reason: str = ""
    dedupe_key: str = ""
    duplicate_of: str | None = None
    operator_id: str
    operator_role: OperatorRole
    created_at: datetime


class EscalationRecord(WorkflowModel):
    escalation_id: str
    event_id: str
    task_id: str
    reason: str
    previous_status: TaskStatus
    target_role: str
    recommended_actions: list[str] = Field(default_factory=list)
    resolved: bool = False
    created_at: datetime


class EventReviewDraft(WorkflowModel):
    review_id: str
    event_id: str
    status: str = "draft"
    headline: str
    executive_summary: str
    process_metrics: dict[str, int | float]
    key_decisions: list[str] = Field(default_factory=list)
    delays_and_escalations: list[str] = Field(default_factory=list)
    evidence_findings: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    improvement_recommendations: list[str] = Field(default_factory=list)
    source_timeline_entry_ids: list[str] = Field(default_factory=list)
    generated_by: str
    generation_source: str = "system"
    created_at: datetime


class ScenarioEvaluationRequest(TaskActionRequest):
    scenario_name: str = "区县下穿通道最小响应场景"
    scenario_type: str = "simulation"
    manual_baseline: dict[str, float] = Field(
        default_factory=lambda: {
            "object_list_generation_time": 25.0,
            "task_generation_time": 35.0,
            "approval_duration": 20.0,
            "acknowledgement_duration": 15.0,
            "report_generation_time": 60.0,
        }
    )
    expected_object_ids: list[str] = Field(default_factory=list)
    note: str = ""


class ScenarioMetricResult(WorkflowModel):
    metric_id: str
    category: str
    label: str
    unit: str
    manual_value: float | None = None
    system_value: float | None = None
    status: str = "measured"
    interpretation: str
    evidence_refs: list[str] = Field(default_factory=list)


class ScenarioAcceptanceCheck(WorkflowModel):
    check_id: str
    requirement: str
    passed: bool
    detail: str
    evidence_refs: list[str] = Field(default_factory=list)


class ScenarioFailureCase(WorkflowModel):
    failure_id: str
    stage: str
    severity: str
    trigger: str
    observed: str
    expected: str
    recommendation: str
    reproducible: bool = True


class DistrictScenarioReport(WorkflowModel):
    report_id: str
    event_id: str
    scenario_name: str
    scenario_type: str
    scope: dict[str, Any]
    baseline_note: str
    metrics: list[ScenarioMetricResult]
    acceptance_checks: list[ScenarioAcceptanceCheck]
    failure_cases: list[ScenarioFailureCase] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    overall_status: str
    generated_by: str
    created_at: datetime


class EventDashboard(WorkflowModel):
    event: ResponseEvent
    alert_snapshots: list[AlertSnapshot]
    risk_objects: list[EventRiskObject]
    tasks: list[ResponseTask]
    assignments: list[TaskAssignmentRecord] = Field(default_factory=list)
    feedback: list[TaskFeedback] = Field(default_factory=list)
    escalations: list[EscalationRecord] = Field(default_factory=list)
    review_draft: EventReviewDraft | None = None
    scenario_report: DistrictScenarioReport | None = None
    candidate_runs: list[CandidateRunRecord] = Field(default_factory=list)
    evidence_packages: list[EvidencePackageVersion] = Field(default_factory=list)
    outbox: list[OutboxMessage] = Field(default_factory=list)
    rule_evaluations: list[RuleEvaluationRecord] = Field(default_factory=list)
    deadline_extensions: list[DeadlineExtensionRecord] = Field(default_factory=list)
    risk_object_versions: list[RiskObjectVersionSnapshot] = Field(default_factory=list)
    timeline: list[TimelineEntry]
    metrics: dict[str, int | float]


class EventCloseRequest(TaskActionRequest):
    pass


class ReviewDraftRequest(TaskActionRequest):
    pass


class TaskDraftGenerationRequest(TaskActionRequest):
    query: str = ""
    acknowledge_minutes: int = Field(default=15, ge=5, le=240)
    deadline_minutes: int = Field(default=120, ge=15, le=1440)
    retrieval_mode: RetrievalMode = RetrievalMode.SHADOW

    @model_validator(mode="after")
    def validate_generation_deadlines(self):
        if self.acknowledge_minutes >= self.deadline_minutes:
            raise ValueError(
                "acknowledge_minutes must be earlier than deadline_minutes"
            )
        return self


class TaskDraftGenerationResult(WorkflowModel):
    event_id: str
    object_id: str
    status: str
    task: ResponseTask | None = None
    evidence: list[TaskEvidenceRef] = Field(default_factory=list)
    role_coverage: dict[str, bool] = Field(default_factory=dict)
    missing_roles: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    grounding_summary: str
    generation_version: str
    evidence_package: EvidencePackageVersion | None = None
