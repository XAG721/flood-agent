export type RiskLevel = "None" | "Blue" | "Yellow" | "Orange" | "Red";
export type EntityType =
  | "resident"
  | "school"
  | "factory"
  | "hospital"
  | "nursing_home"
  | "metro_station"
  | "underground_space"
  | "community";

export type TravelMode = "walk" | "vehicle" | "assisted";
export type CorpusType = "policy" | "case" | "profile";
export type OperatorRole = "observer" | "street_operator" | "district_operator" | "commander";
export type GenerationSource = "system" | "llm";
export type ExecutionMode = "notification" | "evacuation_task" | "resource_dispatch" | "generic_task";

export interface ObservationIngestItem {
  observed_at: string;
  source_type: string;
  source_name: string;
  village?: string | null;
  rainfall_mm: number;
  water_level_m: number;
  road_blocked?: boolean;
  citizen_reports?: number;
  notes?: string;
}

export interface SimulationCell {
  cell_id: string;
  label?: string;
  water_depth_m: number;
  flow_velocity_mps: number;
}

export interface SimulationUpdateRequest {
  simulation_update_id?: string | null;
  area_id?: string | null;
  generated_at: string;
  depth_threshold_m: number;
  flow_threshold_mps: number;
  cells: SimulationCell[];
}

export interface V2EventRecord {
  event_id: string;
  area_id: string;
  title: string;
  trigger_reason: string;
  current_stage: string;
  current_risk_level: RiskLevel;
  status: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export type AgentName =
  | "hazard_agent"
  | "exposure_agent"
  | "resource_agent"
  | "planning_agent"
  | "policy_agent"
  | "comms_agent";

export type AgentTaskStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "canceled"
  | "superseded";

export type AutonomyLevel =
  | "auto_observe"
  | "auto_recommend"
  | "human_gate_required";

export interface EvidenceItem {
  evidence_type: string;
  title: string;
  source_id: string;
  excerpt: string;
  timestamp?: string | null;
  priority: number;
  retrieval_explain?: Record<string, unknown>;
}

export interface RouteOption {
  route_id: string;
  summary: string;
  destination_name: string;
  destination_type: string;
  travel_mode: string;
  eta_minutes: number;
  risk_score: number;
  segments: string[];
  risk_segments: string[];
  blocked_reason: string;
  available: boolean;
}

export interface EmergencyContact {
  name: string;
  phone: string;
  role: string;
}

export interface EntityProfile {
  entity_id: string;
  area_id: string;
  entity_type: EntityType;
  name: string;
  village: string;
  location_hint: string;
  resident_count: number;
  current_occupancy: number;
  vulnerability_tags: string[];
  mobility_constraints: string[];
  key_assets: string[];
  inventory_summary: string;
  continuity_requirement: string;
  preferred_transport_mode: TravelMode;
  notification_preferences: string[];
  emergency_contacts: EmergencyContact[];
  custom_attributes: Record<string, unknown>;
}

export interface EntityImpactView {
  event_id: string;
  entity: EntityProfile;
  risk_level: RiskLevel;
  time_to_impact_minutes: number;
  risk_reason: string[];
  safe_routes: RouteOption[];
  blocked_routes: RouteOption[];
  nearest_shelters: string[];
  resource_gap: string[];
  evidence: EvidenceItem[];
}

export interface HazardTile {
  tile_id: string;
  area_name: string;
  horizon_minutes: number;
  risk_level: RiskLevel;
  risk_score: number;
  predicted_water_depth_cm: number;
  trend: string;
  uncertainty: number;
  affected_roads: string[];
}

export interface RoadReachability {
  road_id: string;
  name: string;
  from_village: string;
  to_location: string;
  accessible: boolean;
  travel_time_minutes: number;
  depth_limit_cm: number;
  failure_reason: string;
}

export interface MonitoringPointState {
  point_name: string;
  latest_water_level_m: number;
  latest_rainfall_mm: number;
  status: string;
  updated_at: string;
}

export interface HazardStateV2 {
  event_id: string;
  area_id: string;
  generated_at: string;
  overall_risk_level: RiskLevel;
  overall_score: number;
  trend: string;
  uncertainty: number;
  freshness_seconds: number;
  hazard_tiles: HazardTile[];
  road_reachability: RoadReachability[];
  monitoring_points: MonitoringPointState[];
}

export interface ActionProposalV2 {
  proposal_id: string;
  event_id: string;
  entity_id: string | null;
  area_id?: string | null;
  proposal_scope?: string;
  action_type?: string | null;
  execution_mode?: ExecutionMode;
  action_display_name?: string;
  action_display_tagline?: string;
  action_display_category?: string;
  title: string;
  summary: string;
  trigger_reason?: string;
  recommendation?: string;
  evidence_summary?: string;
  severity: string;
  requires_confirmation: boolean;
  required_operator_roles: string[];
  payload: Record<string, unknown>;
  high_risk_object_ids?: string[];
  action_scope?: Record<string, unknown>;
  risk_stage_key?: string | null;
  system_version_hash?: string;
  generation_source?: GenerationSource;
  model_name?: string;
  prompt_profile?: string;
  grounding_summary?: string;
  chat_follow_up_prompt?: string;
  source_session_id: string | null;
  status: "pending" | "approved" | "rejected" | "withdrawn" | "superseded";
  updated_at?: string | null;
  edited_by_commander?: boolean;
  last_editor?: string;
  has_new_system_suggestion?: boolean;
  superseded_by_proposal_id?: string | null;
  withdrawn_reason?: string;
  resolved_at: string | null;
  resolved_by: string | null;
  resolution_note: string;
  created_at: string;
}

export interface ProposalDraftUpdateRequest {
  operator_id: string;
  operator_role: string;
  action_scope: Record<string, unknown>;
}

export interface RegionalProposalView {
  proposal: ActionProposalV2;
  event_title: string;
  current_risk_level: RiskLevel;
  high_risk_object_names: string[];
}

export interface RegionalProposalQueueSnapshot {
  queue_version: string;
  generated_at: string;
  items: RegionalProposalView[];
}

export type RegionalAnalysisPackageStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "withdrawn"
  | "superseded"
  | "partially_resolved";

export interface RegionalAnalysisPackageView {
  package_id: string;
  event_id: string;
  current_risk_level: RiskLevel;
  trigger_type: string;
  focus_object_ids: string[];
  focus_object_names: string[];
  proposal_ids: string[];
  proposal_titles: string[];
  proposal_count: number;
  analysis_message: string;
  risk_assessment: string;
  rescue_plan: string;
  resource_dispatch_plan: string;
  status: RegionalAnalysisPackageStatus;
  created_at: string;
  updated_at: string;
}

export interface TwinFocusObjectSummary {
  object_id: string;
  name: string;
  entity_type: EntityType;
  village: string;
  risk_level: RiskLevel;
  time_to_impact_minutes: number;
  summary: string;
  recommended_action: string;
  pending_proposal_ids: string[];
  canvas_position: Record<string, number>;
}

export interface TwinObjectMapLayer {
  object_id: string;
  name: string;
  risk_level: RiskLevel;
  entity_type: EntityType;
  east_offset_m: number;
  north_offset_m: number;
  height_offset_m: number;
  proposal_state: string;
  is_lead: boolean;
}

export interface TwinSignalView {
  signal_id: string;
  title: string;
  detail: string;
  severity: string;
  created_at: string;
}

export interface TwinOverviewView {
  event_id: string;
  area_id: string;
  event_title: string;
  generated_at: string;
  overall_risk_level: RiskLevel;
  trend: string;
  summary: string;
  lead_object_id: string | null;
  lead_object_name: string | null;
  focus_objects: TwinFocusObjectSummary[];
  map_layers: TwinObjectMapLayer[];
  pending_proposal_count: number;
  approved_proposal_count: number;
  warning_draft_count: number;
  active_alert_count: number;
  recommended_actions: string[];
  signals: TwinSignalView[];
  recent_warning_drafts: AudienceWarningDraft[];
}

export interface FocusObjectView {
  event_id: string;
  object_id: string;
  object_name: string;
  entity_type: EntityType;
  village: string;
  risk_level: RiskLevel;
  time_to_impact_minutes: number;
  summary: string;
  risk_reasons: string[];
  recommended_actions: string[];
  risk_reminders: string[];
  evidence: EvidenceItem[];
  related_proposals: RegionalProposalView[];
}

export interface AgentDialogRequest {
  object_id?: string | null;
  message: string;
}

export interface V3ProposalDraft {
  blocked: boolean;
  block_reason?: string | null;
  proposal: RegionalProposalView | null;
}

export interface AgentDialogResponse {
  event_id: string;
  object_id: string;
  object_name: string;
  message: string;
  answer: string;
  impact_summary: string[];
  evidence: EvidenceItem[];
  recommended_actions: string[];
  risk_reminders: string[];
  follow_up_prompts: string[];
  grounding_summary: string;
  proposal_entry: V3ProposalDraft | null;
  response_source: string;
  generated_at: string;
}

export interface AgentCouncilRoleView {
  role: string;
  label: string;
  status: string;
  summary: string;
  confidence?: number | null;
  evidence_count: number;
  recommended_action?: string | null;
}

export interface AuditDecisionView {
  status: string;
  summary: string;
  rationale: string;
  risk_flags: string[];
  approval_required: boolean;
}

export interface AgentCouncilView {
  event_id: string;
  generated_at: string;
  overall_summary: string;
  decision_path: string[];
  open_questions: string[];
  blocked_by: string[];
  roles: AgentCouncilRoleView[];
  audit_decision: AuditDecisionView;
  recent_result_ids: string[];
}

export interface ProposalGenerationRequest {
  object_ids: string[];
}

export interface ProposalGenerationResponse {
  event_id: string;
  queue_version: string;
  generated_at: string;
  blocked: boolean;
  block_reason?: string | null;
  proposals: V3ProposalDraft[];
}

export interface AudienceWarningDraft {
  warning_id: string;
  event_id: string;
  proposal_id: string;
  audience: string;
  channel: string;
  content: string;
  grounding_summary: string;
  created_at: string;
  source_draft_id?: string | null;
}

export interface WarningGenerationResponse {
  event_id: string;
  proposal_id: string;
  generated_at: string;
  warnings: AudienceWarningDraft[];
}

export interface TwinStreamEvent {
  event_type: string;
  version: string;
  created_at: string;
  payload: Record<string, unknown>;
}

export interface AgentDialogTranscriptEntry {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  response?: AgentDialogResponse;
}

export interface ToolTraceStep {
  tool_name: string;
  summary: string;
}

export interface SkippedToolReason {
  tool_name: string;
  reason: string;
}

export interface MemorySnapshot {
  session_id: string;
  focus_entity_id?: string | null;
  focus_entity_name?: string | null;
  focus_area_id?: string | null;
  current_goal?: string | null;
  pending_proposal_ids: string[];
  executed_proposal_ids: string[];
  unresolved_slots: string[];
  last_completion_status?: "direct_answer" | "conservative_answer" | "human_escalation" | null;
  updated_at?: string | null;
}

export interface PlanRunRecord {
  plan_run_id: string;
  session_id: string;
  event_id: string;
  message_id?: string | null;
  plan_id: string;
  planning_layer: "rule" | "llm" | "merged" | "replan";
  replan_round: number;
  parent_plan_id?: string | null;
  intent: string;
  target_entity_id?: string | null;
  target_entity_name?: string | null;
  selected_tools: string[];
  tool_selection_reasoning: string[];
  skipped_tools: SkippedToolReason[];
  plan_notes: string[];
  created_at: string;
}

export interface AgentTask {
  task_id: string;
  event_id: string;
  agent_name: AgentName;
  task_type: string;
  status: AgentTaskStatus;
  input_payload: Record<string, unknown>;
  output_payload: Record<string, unknown>;
  priority: number;
  session_id?: string | null;
  parent_task_id?: string | null;
  replayed_from_task_id?: string | null;
  replay_reason?: string | null;
  source_trigger_id?: string | null;
  failure_reason?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface AgentResult {
  result_id: string;
  task_id: string;
  event_id: string;
  agent_name: AgentName;
  summary: string;
  structured_output: Record<string, unknown>;
  confidence: number;
  decision_confidence?: number;
  evidence_refs: string[];
  missing_slots: string[];
  handoff_recommendations: string[];
  recommended_next_tasks?: string[];
  stop_reason?: string | null;
  supersedes_task_ids?: string[];
  created_at: string;
}

export interface SharedMemorySnapshot {
  event_id: string;
  autonomy_level: AutonomyLevel;
  active_agents: AgentName[];
  focus_entity_ids: string[];
  focus_entity_names: string[];
  top_risks: string[];
  recommended_actions: string[];
  pending_proposal_ids: string[];
  recent_result_ids: string[];
  unresolved_items: string[];
  active_decision_path: string[];
  open_questions: string[];
  blocked_by: string[];
  latest_hazard_level?: RiskLevel | null;
  latest_summary: string;
  last_trigger: string;
  updated_at: string;
}

export interface MemoryEventRecord {
  memory_event_id: string;
  session_id: string;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface SessionMemoryView {
  session_id: string;
  memory_snapshot: MemorySnapshot;
  recent_events: MemoryEventRecord[];
}

export interface MemoryBundleView {
  session_memory?: SessionMemoryView | null;
  event_shared_memory?: SharedMemorySnapshot | null;
}

export interface TriggerEvent {
  trigger_id: string;
  event_id: string;
  trigger_type: string;
  status: "pending" | "leased" | "processed" | "failed";
  payload: Record<string, unknown>;
  dedupe_key?: string | null;
  error_message?: string | null;
  created_at: string;
  leased_at?: string | null;
  processed_at?: string | null;
}

export interface AgentTimelineEntry {
  entry_id: string;
  event_id: string;
  entry_type: string;
  task_id?: string | null;
  task_event_type?: string | null;
  trigger_id?: string | null;
  trigger_type?: string | null;
  agent_name?: AgentName | null;
  summary: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface SupervisorRunRecord {
  supervisor_run_id: string;
  event_id: string;
  trigger_type: string;
  autonomy_level: AutonomyLevel;
  status: "running" | "completed" | "failed";
  session_id?: string | null;
  summary: string;
  created_tasks: string[];
  completed_task_ids: string[];
  created_at: string;
  completed_at?: string | null;
}

export interface SupervisorLoopStatus {
  running: boolean;
  interval_seconds: number;
  consecutive_failures?: number;
  retries_used_in_last_cycle?: number;
  skipped_sweeps?: number;
  circuit_state?: "closed" | "open" | "half_open";
  last_started_at?: string | null;
  last_success_at?: string | null;
  last_failure_at?: string | null;
  last_retry_at?: string | null;
  last_completed_at?: string | null;
  last_error?: string | null;
  circuit_opened_at?: string | null;
  circuit_expires_at?: string | null;
  pending_trigger_count?: number;
  last_trigger_processed_at?: string | null;
  recent_replay_count?: number;
  recent_timeline_failure_count?: number;
}

export interface OperationalAlert {
  alert_id: string;
  source_type: string;
  severity: "info" | "warning" | "critical";
  status: "open" | "resolved";
  summary: string;
  details: string;
  event_id?: string | null;
  first_seen_at: string;
  last_seen_at: string;
  resolved_at?: string | null;
}

export interface AuditRecord {
  audit_id: string;
  source_type: string;
  action: string;
  summary: string;
  details: Record<string, unknown>;
  severity: "info" | "warning" | "critical";
  event_id?: string | null;
  session_id?: string | null;
  created_at: string;
}

export interface ArchiveRunRecord {
  archive_run_id: string;
  status: string;
  hot_records_archived: number;
  expired_archives_deleted: number;
  tables_touched: string[];
  error_message?: string | null;
  started_at: string;
  completed_at?: string | null;
}

export interface ArchiveStatusView {
  hot_retention_days: number;
  archive_retention_days: number;
  hot_record_count: number;
  archived_record_count: number;
  last_archive_run?: ArchiveRunRecord | null;
  latest_archive_runs: ArchiveRunRecord[];
}

export interface DatasetSourceStatusView {
  source_id: string;
  title: string;
  category: string;
  source_type: string;
  source_ref: string;
  download_url: string;
  cache_status: string;
  cached_files: string[];
  notes: string;
  last_fetched_at?: string | null;
  parser_kind: string;
  artifact_count: number;
  downloaded_artifact_count: number;
  failed_artifact_count: number;
  progress_percent: number;
  retryable: boolean;
  last_error?: string | null;
  parsed_summary: Record<string, unknown>;
  latest_fetch_details: Record<string, unknown>[];
  completeness_status: string;
  artifacts_manifest_path?: string | null;
  versions_manifest_path?: string | null;
  parsed: boolean;
  required: boolean;
  missing_artifact_types: string[];
  raw_file_count: number;
  artifacts: Record<string, unknown>[];
}

export interface RawCacheHealthView {
  source_id: string;
  title: string;
  completeness_status: string;
  cache_status: string;
  parsed: boolean;
  required: boolean;
  raw_file_count: number;
  downloaded_artifact_count: number;
  failed_artifact_count: number;
  last_fetched_at?: string | null;
  missing_artifact_types: string[];
  last_error?: string | null;
}

export interface DatasetPipelineStatusView {
  area_id: string;
  raw_dir: string;
  normalized_dir: string;
  bootstrap_dir: string;
  runtime_rag_path: string;
  source_count: number;
  cached_source_count: number;
  failed_source_count: number;
  cached_file_count: number;
  raw_ready: boolean;
  raw_completeness_percent: number;
  missing_required_sources: string[];
  stale_sources: string[];
  sources: DatasetSourceStatusView[];
  raw_cache_health: RawCacheHealthView[];
  latest_download_log: Record<string, unknown>[];
  latest_fetch_summary: Record<string, unknown>;
  latest_build_summary: Record<string, unknown>;
  latest_validation: Record<string, unknown>;
  normalized_files: string[];
  bootstrap_files: string[];
  active_job?: DatasetJobView | null;
  recent_jobs?: DatasetJobView[];
}

export interface DatasetJobView {
  job_id: string;
  action: string;
  status: string;
  progress_percent: number;
  current_step: string;
  message: string;
  source_ids: string[];
  request_payload?: Record<string, unknown>;
  attempt_count?: number;
  max_attempts?: number;
  auto_retry_enabled?: boolean;
  retry_count?: number;
  retry_of_job_id?: string | null;
  cancel_requested?: boolean;
  started_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
  cancel_requested_at?: string | null;
  canceled_at?: string | null;
  error?: string | null;
  result_summary?: string | null;
}

export interface AgentStatusView {
  event_id: string;
  active_agents: AgentName[];
  autonomy_level: AutonomyLevel;
  latest_hazard_level?: RiskLevel | null;
  pending_task_count: number;
  running_task_count: number;
  completed_task_count: number;
  superseded_task_count?: number;
  active_decision_path?: string[];
  open_questions?: string[];
  blocked_by?: string[];
  latest_summary: string;
  updated_at: string;
}

export interface ExperienceRecord {
  experience_id: string;
  event_id: string;
  entity_id?: string | null;
  entity_type?: string | null;
  risk_level?: RiskLevel | null;
  action_type: string;
  action_summary: string;
  outcome: string;
  confidence: number;
  tags: string[];
  payload: Record<string, unknown>;
  created_at: string;
}

export interface DailyReportView {
  report_id: string;
  event_id: string;
  report_date: string;
  timezone: string;
  headline: string;
  situation_summary: string;
  decisions_summary: string;
  action_summary: string;
  unresolved_risks: string[];
  next_day_recommendations: string[];
  generation_source?: GenerationSource;
  model_name?: string;
  grounding_summary?: string;
  delivered_session_ids: string[];
  created_at: string;
}

export interface EventEpisodeSummaryView {
  summary_id: string;
  episode_id: string;
  event_id: string;
  headline: string;
  escalation_path: string[];
  key_decisions: string[];
  successful_actions: string[];
  failed_actions: string[];
  coordination_gaps: string[];
  reusable_rules: string[];
  memory_tags: string[];
  created_at: string;
}

export interface LongTermMemoryView {
  memory_id: string;
  event_id: string;
  source_summary_id: string;
  memory_type: string;
  area_id: string;
  risk_level?: RiskLevel | null;
  entity_types: string[];
  action_types: string[];
  tags: string[];
  headline: string;
  summary: string;
  retrieval_text: string;
  lessons: string[];
  pitfalls: string[];
  recommendations: string[];
  created_at: string;
}

export interface StrategyPattern {
  pattern_id: string;
  entity_type?: string | null;
  risk_level?: RiskLevel | null;
  action_type: string;
  sample_size: number;
  approval_rate: number;
  execution_success_rate: number;
  recommended_summary: string;
  common_failures: string[];
  supporting_experience_ids: string[];
}

export interface ExperienceContextView {
  event_id: string;
  relevant_records: ExperienceRecord[];
  strategy_patterns: StrategyPattern[];
  outcome_risk_notes: string[];
  long_term_memories: LongTermMemoryView[];
}

export interface StrategyHistoryView {
  entity_id: string;
  records: ExperienceRecord[];
  patterns: StrategyPattern[];
}

export interface EvaluationBenchmark {
  benchmark_id: string;
  title: string;
  question: string;
  scenario_type: string;
  expected_tools: string[];
  expected_completion_status?: "direct_answer" | "conservative_answer" | "human_escalation" | null;
  expected_human_confirmation: boolean;
}

export interface EvaluationReport {
  report_id: string;
  created_at: string;
  benchmark_count: number;
  tool_selection_correctness: number;
  dynamic_dispatch_correctness: number;
  shared_memory_reuse_rate: number;
  evidence_coverage_rate: number;
  human_escalation_correctness: number;
  hallucination_rate?: number;
  scenario_results?: Array<{
    benchmark_id: string;
    title: string;
    passed: boolean;
    event_id: string;
    session_id: string;
    used_tools: string[];
    expected_tools: string[];
    completion_status?: "direct_answer" | "conservative_answer" | "human_escalation" | null;
    expected_completion_status?: "direct_answer" | "conservative_answer" | "human_escalation" | null;
    human_confirmation: boolean;
    expected_human_confirmation: boolean;
    evidence_count: number;
    shared_memory_reused: boolean;
    notes: string[];
  }>;
  notes: string[];
}

export interface OperatorCapabilitiesView {
  operator_role: OperatorRole;
  role_rank: number;
  capabilities: Record<string, boolean>;
  action_labels: Record<string, string>;
}

export interface AgentMetricsView {
  generated_at: string;
  task_graph_latency_ms: number;
  agent_failure_heatmap: Record<string, number>;
  stale_data_frequency: number;
  auto_retry_success_rate: number;
  superseded_task_ratio: number;
  fanout_count: number;
  stale_data_replan_count: number;
}

export interface DecisionReportView {
  event_id: string;
  latest_summary: string;
  active_decision_path: string[];
  blocked_by: string[];
  open_questions: string[];
  recent_result_ids: string[];
}

export interface ToolExecutionResultView {
  execution_id?: string | null;
  tool_name: string;
  status: "success" | "failed" | "skipped" | "timeout";
  input: Record<string, unknown>;
  output_summary: string;
  raw_output?: unknown;
  failure_reason?: string | null;
  duration_ms: number;
  timed_out: boolean;
  data_freshness_seconds?: number | null;
  attempt?: number;
  retry_of_execution_id?: string | null;
  fallback_from_tool?: string | null;
  dependency_tools?: string[];
  stale?: boolean;
  cache_hit?: boolean;
  parallel_group?: string | null;
  session_id?: string;
  event_id?: string;
  message_id?: string | null;
  plan_id?: string | null;
  replan_round?: number;
  created_at?: string;
}

export interface DataFreshnessSummary {
  hazard_state_freshness_seconds?: number | null;
  traffic_freshness_seconds?: number | null;
  profile_freshness_label?: string | null;
  rag_document_recency_summary?: string | null;
}

export interface StructuredAnswer {
  answer: string;
  evidence: EvidenceItem[];
  impact_summary: string[];
  recommended_actions: string[];
  follow_up_prompts?: string[];
  confidence: number;
  confidence_explanation?: string;
  requires_human_confirmation: boolean;
  missing_data: string[];
  proposal: ActionProposalV2 | null;
  generation_source?: GenerationSource;
  model_name?: string;
  grounding_summary?: string;
  planner_summary?: string;
  tool_selection_reasoning?: string[];
  skipped_tools?: SkippedToolReason[];
  tool_executions?: ToolExecutionResultView[];
  data_freshness?: DataFreshnessSummary;
  evidence_gaps?: string[];
  tool_trace: ToolTraceStep[];
  planning_layers_summary?: string[];
  plan_runs?: PlanRunRecord[];
  completion_status?: "direct_answer" | "conservative_answer" | "human_escalation";
  termination_reason?: string;
  memory_snapshot?: MemorySnapshot | null;
  replan_count?: number;
  used_fallbacks?: string[];
  carried_context_notes?: string[];
}

export interface Advisory {
  advisory_id: string;
  event_id: string;
  entity_id: string | null;
  answer: string;
  impact_summary: string[];
  recommended_actions: string[];
  route_options: RouteOption[];
  evidence: EvidenceItem[];
  confidence: number;
  confidence_explanation?: string;
  requires_human_confirmation: boolean;
  missing_data: string[];
  proposal: ActionProposalV2 | null;
  generation_source?: GenerationSource;
  model_name?: string;
  grounding_summary?: string;
  generated_at: string;
}

export interface V2CopilotMessage {
  message_id: string;
  role: "assistant" | "user";
  content: string;
  created_at: string;
  structured_answer: StructuredAnswer | null;
}

export interface NotificationDraftV2 {
  draft_id: string;
  event_id: string;
  proposal_id: string;
  entity_id: string | null;
  area_id?: string | null;
  audience: string;
  channel: string;
  content: string;
  status: string;
  generation_source?: GenerationSource;
  model_name?: string;
  grounding_summary?: string;
  created_at: string;
}

export interface ExecutionLogEntryV2 {
  log_id: string;
  event_id: string;
  proposal_id: string;
  entity_id: string | null;
  area_id?: string | null;
  action_type: string;
  summary: string;
  operator_id: string;
  details: Record<string, unknown>;
  generation_source?: GenerationSource;
  model_name?: string;
  grounding_summary?: string;
  created_at: string;
}

export interface V2CopilotSessionView {
  session_id: string;
  event: V2EventRecord;
  messages: V2CopilotMessage[];
  latest_answer: StructuredAnswer | null;
  proposals: ActionProposalV2[];
  notification_drafts: NotificationDraftV2[];
  execution_logs: ExecutionLogEntryV2[];
  daily_reports?: DailyReportView[];
  episode_summaries?: EventEpisodeSummaryView[];
  memory_snapshot?: MemorySnapshot | null;
  session_memory_snapshot?: MemorySnapshot | null;
  plan_runs?: PlanRunRecord[];
  recent_tool_executions?: ToolExecutionResultView[];
  shared_memory_snapshot?: SharedMemorySnapshot | null;
  active_agents?: AgentName[];
  recent_agent_results?: AgentResult[];
  autonomy_level?: AutonomyLevel;
  pending_regional_analysis_package?: RegionalAnalysisPackageView | null;
  regional_analysis_package_history?: RegionalAnalysisPackageView[];
}

export interface ExposureSummaryV2 {
  event_id: string;
  area_id: string;
  generated_at: string;
  affected_entities: EntityImpactView[];
  top_risks: string[];
}

export interface V2StreamRecord {
  event_id: string;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface V2EventSnapshot {
  event: V2EventRecord;
  latest_hazard_state: HazardStateV2 | null;
  latest_exposure_summary: ExposureSummaryV2 | null;
  recent_stream: V2StreamRecord[];
}

export interface ResourceStatus {
  area_id: string;
  vehicle_count: number;
  staff_count: number;
  supply_kits: number;
  rescue_boats: number;
  ambulance_count: number;
  drone_count: number;
  portable_pumps: number;
  power_generators: number;
  medical_staff_count: number;
  volunteer_count: number;
  satellite_phones: number;
  notes: string;
}

export interface ResourceStatusView {
  scope: string;
  area_id: string;
  event_id?: string | null;
  resource_status: ResourceStatus;
}

export interface RAGDocument {
  doc_id: string;
  corpus: CorpusType;
  title: string;
  content: string;
  metadata: Record<string, unknown>;
}
