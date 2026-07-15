export type WorkflowRole =
  | "duty_officer"
  | "reviewer"
  | "commander"
  | "liaison"
  | "field_operator"
  | "auditor"
  | "admin";

export type ResponseTaskStatus =
  | "draft"
  | "pending_approval"
  | "issued"
  | "acknowledged"
  | "in_progress"
  | "pending_verification"
  | "completed"
  | "escalated"
  | "waived"
  | "blocked"
  | "partially_completed"
  | "cancelled"
  | "taken_over";

export interface ResponseEvent {
  event_id: string;
  title: string;
  area_id: string;
  status: "active" | "closing" | "closed";
  current_alert_version: number;
  created_at: string;
  updated_at: string;
  closed_at?: string | null;
  workflow_engine_version: string;
  data_version: string;
  source_type: string;
  is_simulated: boolean;
}

export interface AlertSnapshot {
  snapshot_id: string;
  event_id: string;
  alert_id: string;
  source_department: string;
  disaster_type: string;
  level: string;
  issued_at: string;
  valid_until?: string | null;
  affected_area: string;
  affected_geometry?: {
    coordinates: Array<[number, number]>;
    crs: "EPSG:4326";
  } | null;
  raw_content: string;
  version: number;
  created_at: string;
  raw_payload_hash: string;
  data_version: string;
  source_type: string;
  source_version: string;
  is_simulated: boolean;
  lifecycle_status: "active" | "updated" | "revoked" | "expired";
}

export interface EventRiskObject {
  event_id: string;
  object_id: string;
  canonical_object_id?: string | null;
  aliases?: string[];
  duplicate_of?: string | null;
  name: string;
  object_type: string;
  location: string;
  longitude?: number | null;
  latitude?: number | null;
  location_classification?: "internal" | "restricted" | "highly_sensitive";
  responsible_organization: string;
  responsible_role: string;
  trigger_reasons: string[];
  source_refs: string[];
  vulnerability: string;
  historical_risk: string;
  risk_score: number;
  system_explanation: string;
  verification_status: "pending" | "confirmed" | "excluded";
  verification_note: string;
  sensitive_contacts?: Array<{
    name: string;
    role: string;
    phone: string;
    classification: "internal" | "restricted" | "highly_sensitive";
  }>;
  special_population_notes?: string;
  special_population_classification?: "internal" | "restricted" | "highly_sensitive";
  source_type: string;
  source_version: string;
  data_version: string;
  is_simulated: boolean;
  missing_fields: string[];
  stale: boolean;
  candidate_run_id?: string | null;
  version: number;
  updated_at?: string | null;
  raw_candidate_score?: number | null;
  calibrated_confidence?: number | null;
  calibration_version: string;
  association_mode: string;
  registry_status?: string;
  registry_valid_from?: string | null;
  registry_valid_until?: string | null;
}

export interface RiskObjectVersionSnapshot {
  snapshot_id: string;
  event_id: string;
  object_id: string;
  version: number;
  object: EventRiskObject;
  change_type: string;
  changed_by: string;
  terminal_id: string;
  created_at: string;
}

export interface PlanBasis {
  document: string;
  version: string;
  clause: string;
}

export type EvidenceRole = "condition" | "object" | "responsibility" | "procedure" | "exception" | "attribution";

export interface TaskEvidenceRef {
  source_type: string;
  source_id: string;
  title: string;
  excerpt: string;
  roles: EvidenceRole[];
  document_version?: string | null;
  clause?: string | null;
  trust_score?: number | null;
  conflict_key?: string | null;
  conflict_value?: string | null;
  jurisdiction?: string | null;
  superseded?: boolean;
  document_version_id?: string | null;
  source_locator: string;
  page_number?: number | null;
  section_path: string[];
  table_name?: string | null;
  row_start?: number | null;
  row_end?: number | null;
  field_support: Record<string, number>;
  conflicts_with: string[];
}

export interface ResponseTask {
  task_id: string;
  event_id: string;
  object_id: string;
  version: number;
  title: string;
  action: string;
  responsible_organization: string;
  responsible_role: string;
  cooperate_roles: string[];
  status: ResponseTaskStatus;
  deadline_at: string;
  acknowledge_deadline_at: string;
  start_deadline_at?: string | null;
  verification_deadline_at?: string | null;
  required_evidence: string[];
  plan_basis: PlanBasis[];
  approval_policy: "reviewer_required" | "commander_required";
  escalation_rule: string;
  dependencies: string[];
  generated_by_ai: boolean;
  generation_version?: string | null;
  drafted_by: string;
  approved_version?: number | null;
  assignee_id?: string | null;
  assignee_name?: string | null;
  assignee_role?: WorkflowRole | null;
  assignment_version: number;
  grounding_summary: string;
  evidence_role_coverage: Partial<Record<EvidenceRole, boolean>>;
  source_evidence: TaskEvidenceRef[];
  validation_warnings: string[];
  approval_payload_hash?: string | null;
  evidence_package_hash?: string | null;
  evidence_package_id?: string | null;
  evidence_package_version?: number | null;
  task_schema_version: string;
  rule_set_version: string;
  dispatch_message_id?: string | null;
  data_version: string;
  is_simulated: boolean;
  last_rule_evaluation_id?: string | null;
  effective_start_deadline_at?: string | null;
  effective_completion_deadline_at?: string | null;
  effective_verification_deadline_at?: string | null;
  active_extension_id?: string | null;
  creation_idempotency_key?: string | null;
  candidate_object_list_id?: string | null;
  candidate_object_list_version?: number | null;
  candidate_object_list_hash?: string | null;
}

export interface CandidateFeatureSnapshot {
  object_id: string;
  rank: number;
  spatial_score: number;
  temporal_score: number;
  attribute_score: number;
  semantic_score: number;
  data_quality_score: number;
  raw_score: number;
  calibrated_confidence: number;
  missing_features: string[];
  explanations: Record<string, string>;
  feature_version: string;
}

export interface CandidateRunRecord {
  run_id: string;
  event_id: string;
  alert_snapshot_id: string;
  risk_object_data_version: string;
  algorithm_version: string;
  feature_version: string;
  association_mode: string;
  parameters: Record<string, unknown>;
  candidate_object_ids: string[];
  candidate_features: CandidateFeatureSnapshot[];
  missing_features: string[];
  limitations: string[];
  status: string;
  stale_at?: string | null;
  stale_reason?: string;
  created_by: string;
  terminal_id: string;
  created_at: string;
}

export interface CandidateObjectReference {
  object_id: string;
  object_version: number;
  candidate_run_id?: string | null;
  data_version: string;
  verification_hash: string;
}

export interface CandidateObjectListVersion {
  list_id: string;
  event_id: string;
  version: number;
  status: "frozen";
  alert_snapshot_id: string;
  source_run_ids: string[];
  objects: CandidateObjectReference[];
  content_hash: string;
  note: string;
  is_simulated: boolean;
  frozen_by: string;
  frozen_role: WorkflowRole;
  terminal_id: string;
  frozen_at: string;
}

export interface RiskObjectRegistryRecord {
  area_id: string;
  object_id: string;
  canonical_object_id?: string | null;
  aliases: string[];
  duplicate_of?: string | null;
  name: string;
  object_type: string;
  location: string;
  longitude?: number | null;
  latitude?: number | null;
  responsible_organization: string;
  responsible_role: string;
  trigger_reasons: string[];
  source_refs: string[];
  vulnerability: string;
  historical_risk: string;
  risk_score: number;
  system_explanation: string;
  sensitive_contacts: Array<{
    name: string;
    role: string;
    phone: string;
    classification: "internal" | "restricted" | "highly_sensitive";
  }>;
  special_population_notes: string;
  source_type: string;
  source_version: string;
  data_version: string;
  is_simulated: boolean;
  missing_fields: string[];
  registry_status: string;
  registry_valid_from?: string | null;
  registry_valid_until?: string | null;
  registry_version: number;
  source_hash: string;
  content_hash: string;
  source_filename: string;
  created_by: string;
  terminal_id: string;
  created_at: string;
  updated_at: string;
}

export interface RiskObjectRegistryImportResult {
  import_id: string;
  area_id: string;
  source_filename: string;
  source_format: string;
  source_version: string;
  source_hash: string;
  source_bytes: number;
  imported_count: number;
  created_count: number;
  updated_count: number;
  unchanged_count: number;
  quarantined_count: number;
  changed_object_ids: string[];
  affected_event_ids: string[];
  stale_candidate_run_count: number;
  quarantine: Array<{
    source_row?: number | null;
    source_id: string;
    reason_code: string;
    reason: string;
  }>;
  is_simulated: boolean;
  imported_by: string;
  terminal_id: string;
  created_at: string;
}

export type EvidenceFieldState = "SUPPORTED" | "CONFLICTED" | "MISSING";
export type EvidenceMissingReason = "SOURCE_ABSENT_CONFIRMED" | "NOT_RETRIEVED" | "INDEX_INCOMPLETE" | "SOURCE_UNAVAILABLE";
export type NliRelation = "entailment" | "contradiction" | "neutral" | "unavailable" | "error";
export type RetrievalMode = "BASELINE_ONLY" | "SHADOW" | "REVIEW" | "CANARY" | "DEFAULT";

export interface EvidenceNliAssessment {
  assessment_id: string;
  left_source_id: string;
  right_source_id: string;
  shared_fields: string[];
  relation: NliRelation;
  confidence: number;
  model_version: string;
  status: string;
  error: string;
}

export interface EvidencePackageVersion {
  package_id: string;
  event_id: string;
  object_id: string;
  version: number;
  status: string;
  task_schema_version: string;
  retrieval_strategy: string;
  retrieval_run_id: string;
  retrieval_mode: RetrievalMode;
  baseline_source_ids: string[];
  frc_source_ids: string[];
  shadow_comparison: {
    overlap?: string[];
    baseline_only?: string[];
    frc_only?: string[];
  };
  field_states: Record<string, EvidenceFieldState>;
  role_coverage: Record<string, boolean>;
  evidence: TaskEvidenceRef[];
  conflicts: Array<{
    conflict_id: string;
    field_name: string;
    conflict_type: string;
    severity: string;
    evidence_source_ids: string[];
    resolution_status: string;
    resolution_reason: string;
    selected_source_ids: string[];
    conflict_dimensions: string[];
    detection_methods: string[];
    detector_version: string;
    nli_assessment_id?: string | null;
    nli_relation?: NliRelation | null;
    nli_confidence?: number | null;
  }>;
  missing_fields: string[];
  required_fields: string[];
  blocking_missing_fields: string[];
  missing_reasons: Partial<Record<string, EvidenceMissingReason>>;
  field_evidence_map: Record<string, string[]>;
  nli_status: string;
  nli_model_version: string;
  nli_assessments: EvidenceNliAssessment[];
  content_hash: string;
  created_by: string;
  reviewed_by?: string | null;
  frozen_at?: string | null;
  created_at: string;
}

export interface DocumentVersionRecord {
  version_id: string;
  document_id: string;
  version_number: number;
  version_label: string;
  title: string;
  issuer: string;
  jurisdiction: string;
  effective_at: string;
  expires_at?: string | null;
  replaces_version_id?: string | null;
  lifecycle_status: "active" | "draft" | "parsed" | "published" | "superseded" | "retired" | "parse_failed" | "index_failed";
  source_hash: string;
  source_filename: string;
  media_type: string;
  source_size: number;
  clauses: Array<{
    clause_id: string;
    heading: string;
    text: string;
    page_number?: number | null;
    section_path: string[];
    clause_number?: string | null;
    table_name?: string | null;
    row_start?: number | null;
    row_end?: number | null;
    source_locator: string;
    content_type: string;
    text_hash: string;
  }>;
  parser_version?: string | null;
  parse_method?: string | null;
  parse_hash?: string | null;
  parsed_at?: string | null;
  index_status: string;
  index_version: string;
  index_build_id?: string | null;
  published_at?: string | null;
  retired_at?: string | null;
  superseded_by_version_id?: string | null;
  is_simulated: boolean;
  created_by: string;
  created_at: string;
}

export interface IndexBuildRecord {
  build_id: string;
  document_version_id: string;
  corpus_version: string;
  parser_version: string;
  embedding_version: string;
  index_version: string;
  status: "completed" | "failed";
  clause_count: number;
  error: string;
  source_hash: string;
  created_by: string;
  created_at: string;
  finished_at: string;
}

export interface OutboxMessage {
  message_id: string;
  event_id: string;
  task_id: string;
  destination: string;
  idempotency_key: string;
  payload_hash: string;
  approval_id: string;
  task_version: number;
  status: "pending" | "sent" | "partially_sent" | "failed" | "manual_takeover";
  attempts: number;
  last_error?: string | null;
  simulation_scenario: string;
  gateway_status?: string | null;
  external_request_id?: string | null;
  trace_id?: string | null;
  callback_count: number;
  created_at: string;
  updated_at: string;
  sent_at?: string | null;
}

export type SimulationDispatchScenario =
  | "normal"
  | "timeout"
  | "reject"
  | "partial_success"
  | "duplicate_callback"
  | "out_of_order_callback";

export interface DispatchCallbackRecord {
  callback_id: string;
  message_id: string;
  event_id: string;
  task_id: string;
  external_id: string;
  source: string;
  version: number;
  event_time: string;
  received_time: string;
  request_id: string;
  trace_id: string;
  idempotency_key: string;
  status: "accepted" | "delivered" | "partial_success" | "rejected";
  error_code?: string | null;
  sequence_state: "in_order" | "out_of_order";
  is_simulated: boolean;
  created_at: string;
}

export interface RuleEvaluationRecord {
  evaluation_id: string;
  event_id: string;
  task_id: string;
  task_version: number;
  rule_set_version: string;
  overall_outcome: "PASS" | "SOFT_WARNING" | "HARD_BLOCK";
  checks: Array<{
    rule_id: string;
    outcome: "PASS" | "SOFT_WARNING" | "HARD_BLOCK";
    message: string;
    field_name?: string | null;
    evidence_refs: string[];
  }>;
  evaluated_by: string;
  terminal_id: string;
  created_at: string;
}

export interface DeadlineExtensionRecord {
  extension_id: string;
  event_id: string;
  task_id: string;
  task_version: number;
  status: "requested" | "approved" | "rejected";
  original_completion_deadline_at: string;
  proposed_completion_deadline_at: string;
  proposed_verification_deadline_at?: string | null;
  request_reason: string;
  requested_by: string;
  requested_by_role: WorkflowRole;
  decided_by?: string | null;
  decision_reason: string;
  created_at: string;
  decided_at?: string | null;
}

export interface TaskDraftGenerationResult {
  event_id: string;
  object_id: string;
  status: "ready" | "insufficient_evidence";
  task?: ResponseTask | null;
  evidence: TaskEvidenceRef[];
  role_coverage: Record<EvidenceRole, boolean>;
  missing_roles: EvidenceRole[];
  validation_errors: string[];
  grounding_summary: string;
  generation_version: string;
  evidence_package?: EvidencePackageVersion | null;
}

export interface TimelineEntry {
  entry_id: string;
  event_id: string;
  task_id?: string | null;
  object_id?: string | null;
  entry_type: string;
  action: string;
  actor_id: string;
  actor_role: string;
  detail: Record<string, unknown>;
  created_at: string;
}

export type FeedbackCategory = "completion" | "partial_completion" | "blocked" | "resource_shortage" | "coordination_request" | "situation_update";

export interface TaskFeedback {
  feedback_id: string;
  event_id: string;
  task_id: string;
  object_id: string;
  summary: string;
  evidence: Array<Record<string, unknown>>;
  blocked_reason?: string | null;
  resource_gap?: string | null;
  category: FeedbackCategory;
  classification_reason: string;
  dedupe_key: string;
  duplicate_of?: string | null;
  operator_id: string;
  operator_role: WorkflowRole;
  created_at: string;
}

export interface EscalationRecord {
  escalation_id: string;
  event_id: string;
  task_id: string;
  reason: string;
  previous_status: ResponseTaskStatus;
  target_role: string;
  recommended_actions: string[];
  resolved: boolean;
  created_at: string;
}

export interface EventReviewDraft {
  review_id: string;
  event_id: string;
  status: "draft";
  headline: string;
  executive_summary: string;
  process_metrics: Record<string, number>;
  key_decisions: string[];
  delays_and_escalations: string[];
  evidence_findings: string[];
  unresolved_issues: string[];
  improvement_recommendations: string[];
  source_timeline_entry_ids: string[];
  generated_by: string;
  generation_source: string;
  created_at: string;
}

export interface ScenarioMetricResult {
  metric_id: string;
  category: string;
  label: string;
  unit: string;
  manual_value?: number | null;
  system_value?: number | null;
  status: "measured" | "not_evaluable";
  interpretation: string;
  evidence_refs: string[];
}

export interface ScenarioAcceptanceCheck {
  check_id: string;
  requirement: string;
  passed: boolean;
  detail: string;
  evidence_refs: string[];
}

export interface ScenarioFailureCase {
  failure_id: string;
  stage: string;
  severity: "blocking" | "handled" | string;
  trigger: string;
  observed: string;
  expected: string;
  recommendation: string;
  reproducible: boolean;
}

export interface DistrictScenarioReport {
  report_id: string;
  event_id: string;
  scenario_name: string;
  scenario_type: string;
  scope: {
    area_id: string;
    alert_versions: number;
    risk_object_instances: number;
    risk_object_types: string[];
    task_count: number;
    operator_roles: string[];
  };
  baseline_note: string;
  metrics: ScenarioMetricResult[];
  acceptance_checks: ScenarioAcceptanceCheck[];
  failure_cases: ScenarioFailureCase[];
  observations: string[];
  overall_status: "passed" | "passed_with_observations" | "failed";
  generated_by: string;
  created_at: string;
}

export interface EventDashboard {
  event: ResponseEvent;
  alert_snapshots: AlertSnapshot[];
  risk_objects: EventRiskObject[];
  tasks: ResponseTask[];
  assignments: Array<{
    assignment_id: string;
    event_id: string;
    task_id: string;
    assignment_version: number;
    previous_assignee_id?: string | null;
    assignee_id: string;
    assignee_name: string;
    assignee_role: WorkflowRole;
    reason: string;
    assigned_by: string;
    assigned_by_role: WorkflowRole;
    terminal_id: string;
    created_at: string;
  }>;
  feedback: TaskFeedback[];
  escalations: EscalationRecord[];
  review_draft?: EventReviewDraft | null;
  scenario_report?: DistrictScenarioReport | null;
  candidate_runs: CandidateRunRecord[];
  candidate_object_lists: CandidateObjectListVersion[];
  evidence_packages: EvidencePackageVersion[];
  outbox: OutboxMessage[];
  rule_evaluations: RuleEvaluationRecord[];
  deadline_extensions: DeadlineExtensionRecord[];
  risk_object_versions: RiskObjectVersionSnapshot[];
  timeline: TimelineEntry[];
  metrics: {
    alert_versions: number;
    candidate_objects: number;
    confirmed_objects: number;
    frozen_candidate_lists?: number;
    task_count: number;
    completed_tasks: number;
    escalated_tasks: number;
    feedback_count?: number;
    duplicate_feedback_merged?: number;
    timeline_entries: number;
    completion_rate: number;
  };
}
