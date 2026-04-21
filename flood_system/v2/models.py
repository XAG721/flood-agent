from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import RAGDocument, ResourceStatus, RiskLevel, Stage


class V2Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class EventType(StrEnum):
    OBSERVATION_INGESTED = "observation_ingested"
    HAZARD_UPDATED = "hazard_updated"
    IMPACT_RECOMPUTED = "impact_recomputed"
    PLAN_PROPOSED = "plan_proposed"
    APPROVAL_RESOLVED = "approval_resolved"
    NOTIFICATION_SENT = "notification_sent"
    ADVISORY_GENERATED = "advisory_generated"


class EventStatus(StrEnum):
    ACTIVE = "active"
    STABILIZING = "stabilizing"
    CLOSED = "closed"


class EntityType(StrEnum):
    RESIDENT = "resident"
    SCHOOL = "school"
    FACTORY = "factory"
    HOSPITAL = "hospital"
    NURSING_HOME = "nursing_home"
    METRO_STATION = "metro_station"
    UNDERGROUND_SPACE = "underground_space"
    COMMUNITY = "community"


class TravelMode(StrEnum):
    WALK = "walk"
    VEHICLE = "vehicle"
    ASSISTED = "assisted"


class EvidenceType(StrEnum):
    REALTIME = "realtime"
    APPROVED_PLAN = "approved_plan"
    POLICY = "policy"
    CASE = "case"
    PROFILE = "profile"
    OPERATIONS_MANUAL = "operations_manual"
    MEMORY = "memory"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    SUPERSEDED = "superseded"


class RegionalAnalysisPackageStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    SUPERSEDED = "superseded"
    PARTIALLY_RESOLVED = "partially_resolved"


class GenerationSource(StrEnum):
    SYSTEM = "system"
    LLM = "llm"


class ExecutionMode(StrEnum):
    NOTIFICATION = "notification"
    EVACUATION_TASK = "evacuation_task"
    RESOURCE_DISPATCH = "resource_dispatch"
    GENERIC_TASK = "generic_task"


class LLMErrorCode(StrEnum):
    UNAVAILABLE = "llm_unavailable"
    INVALID_OUTPUT = "llm_invalid_output"


class ToolFailureMode(StrEnum):
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    STALE_DATA = "stale_data"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    TIMEOUT = "timeout"
    POLICY_BLOCKED = "policy_blocked"
    UNKNOWN = "unknown"


class ToolExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class PlanningLayer(StrEnum):
    RULE = "rule"
    LLM = "llm"
    MERGED = "merged"
    REPLAN = "replan"


class CompletionStatus(StrEnum):
    DIRECT_ANSWER = "direct_answer"
    CONSERVATIVE_ANSWER = "conservative_answer"
    HUMAN_ESCALATION = "human_escalation"


class MemoryEventType(StrEnum):
    USER_QUESTION = "user_question"
    PLANNER_SELECTED_FOCUS = "planner_selected_focus"
    TOOL_MEMORY_UPDATE = "tool_memory_update"
    PROPOSAL_APPROVED = "proposal_approved"
    PROPOSAL_REJECTED = "proposal_rejected"
    REVIEWER_UNRESOLVED_SLOT = "reviewer_unresolved_slot"


class AgentName(StrEnum):
    HAZARD = "hazard_agent"
    EXPOSURE = "exposure_agent"
    RESOURCE = "resource_agent"
    PLANNING = "planning_agent"
    POLICY = "policy_agent"
    COMMS = "comms_agent"


class AgentTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    SUPERSEDED = "superseded"


class SupervisorRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TriggerEventType(StrEnum):
    OBSERVATION_INGESTED = "observation_ingested"
    SIMULATION_UPDATED = "simulation_updated"
    RESOURCE_OVERRIDE_UPDATED = "resource_override_updated"
    RESOURCE_OVERRIDE_DELETED = "resource_override_deleted"
    PROPOSAL_RESOLVED = "proposal_resolved"
    COPILOT_ESCALATION_REQUESTED = "copilot_escalation_requested"
    FRESHNESS_EXPIRED = "freshness_expired"
    MANUAL_TICK = "manual_tick"
    MANUAL_RUN = "manual_run"


class TriggerEventStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    PROCESSED = "processed"
    FAILED = "failed"


class AgentTaskEventType(StrEnum):
    TASK_ENQUEUED = "task_enqueued"
    TASK_CLAIMED = "task_claimed"
    AGENT_RESULT_SAVED = "agent_result_saved"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    REPLAY_REQUESTED = "replay_requested"
    REPLAY_COMPLETED = "replay_completed"
    TRIGGER_PROCESSED = "trigger_processed"


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class AutonomyLevel(StrEnum):
    AUTO_OBSERVE = "auto_observe"
    AUTO_RECOMMEND = "auto_recommend"
    HUMAN_GATE_REQUIRED = "human_gate_required"


class HighRiskEpisodeStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    SUMMARIZED = "summarized"


class ObservationIngestItem(V2Model):
    observed_at: datetime
    source_type: str = "monitoring_point"
    source_name: str = ""
    village: str | None = None
    rainfall_mm: float = 0.0
    water_level_m: float = 0.0
    road_blocked: bool = False
    citizen_reports: int = 0
    notes: str = ""


class SimulationCell(V2Model):
    cell_id: str
    label: str = ""
    water_depth_m: float = 0.0
    flow_velocity_mps: float = 0.0


class SimulationUpdateRequest(V2Model):
    simulation_update_id: str | None = None
    area_id: str | None = None
    generated_at: datetime
    depth_threshold_m: float = 0.5
    flow_threshold_mps: float = 1.5
    cells: list[SimulationCell] = Field(default_factory=list)


class SimulationUpdateRecord(V2Model):
    simulation_update_id: str
    event_id: str
    area_id: str
    generated_at: datetime
    depth_threshold_m: float
    flow_threshold_mps: float
    overall_risk_level: RiskLevel
    overall_score: float
    exceeded_cell_count: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


class EventCreateRequest(V2Model):
    area_id: str
    title: str
    trigger_reason: str = "manual_creation"
    operator: str = "system_operator"
    stage: Stage = Stage.MONITORING
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationBatchRequest(V2Model):
    operator: str = "field_ingestion"
    observations: list[ObservationIngestItem] = Field(default_factory=list)


class EventRecord(V2Model):
    event_id: str
    area_id: str
    title: str
    trigger_reason: str
    current_stage: Stage
    current_risk_level: RiskLevel
    status: EventStatus = EventStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class EventStreamRecord(V2Model):
    event_id: str
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class HazardTile(V2Model):
    tile_id: str
    area_name: str
    horizon_minutes: int
    risk_level: RiskLevel
    risk_score: float
    predicted_water_depth_cm: float
    trend: str
    uncertainty: float
    affected_roads: list[str] = Field(default_factory=list)


class RoadReachability(V2Model):
    road_id: str
    name: str
    from_village: str
    to_location: str
    accessible: bool
    travel_time_minutes: int
    depth_limit_cm: float
    failure_reason: str = ""


class MonitoringPointState(V2Model):
    point_name: str
    latest_water_level_m: float
    latest_rainfall_mm: float
    status: str
    updated_at: datetime


class HazardState(V2Model):
    event_id: str
    area_id: str
    generated_at: datetime
    overall_risk_level: RiskLevel
    overall_score: float
    trend: str
    uncertainty: float
    freshness_seconds: int
    hazard_tiles: list[HazardTile] = Field(default_factory=list)
    road_reachability: list[RoadReachability] = Field(default_factory=list)
    monitoring_points: list[MonitoringPointState] = Field(default_factory=list)


class ContactProfile(V2Model):
    name: str
    phone: str = ""
    role: str = ""


class EntityProfile(V2Model):
    entity_id: str
    area_id: str
    entity_type: EntityType
    name: str
    village: str
    location_hint: str
    resident_count: int = 0
    current_occupancy: int = 0
    vulnerability_tags: list[str] = Field(default_factory=list)
    mobility_constraints: list[str] = Field(default_factory=list)
    key_assets: list[str] = Field(default_factory=list)
    inventory_summary: str = ""
    continuity_requirement: str = ""
    preferred_transport_mode: TravelMode = TravelMode.WALK
    notification_preferences: list[str] = Field(default_factory=list)
    emergency_contacts: list[ContactProfile] = Field(default_factory=list)
    custom_attributes: dict[str, Any] = Field(default_factory=dict)


class RouteOption(V2Model):
    route_id: str
    summary: str
    destination_name: str
    destination_type: str
    travel_mode: TravelMode
    eta_minutes: int
    risk_score: float
    segments: list[str] = Field(default_factory=list)
    risk_segments: list[str] = Field(default_factory=list)
    blocked_reason: str = ""
    available: bool = True


class EvidenceItem(V2Model):
    evidence_type: EvidenceType
    title: str
    source_id: str
    excerpt: str
    timestamp: datetime | None = None
    priority: int = 0
    retrieval_explain: dict[str, Any] = Field(default_factory=dict)


class PolicyConstraint(V2Model):
    entity_type: EntityType
    risk_level: RiskLevel
    requires_confirmation: bool
    actions_requiring_approval: list[str] = Field(default_factory=list)
    operator_roles: list[str] = Field(default_factory=list)


class EntityImpactView(V2Model):
    event_id: str
    entity: EntityProfile
    risk_level: RiskLevel
    time_to_impact_minutes: int
    risk_reason: list[str] = Field(default_factory=list)
    safe_routes: list[RouteOption] = Field(default_factory=list)
    blocked_routes: list[RouteOption] = Field(default_factory=list)
    nearest_shelters: list[str] = Field(default_factory=list)
    resource_gap: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ExposureSummary(V2Model):
    event_id: str
    area_id: str
    generated_at: datetime
    affected_entities: list[EntityImpactView] = Field(default_factory=list)
    top_risks: list[str] = Field(default_factory=list)


class ActionProposal(V2Model):
    proposal_id: str
    event_id: str
    entity_id: str | None = None
    area_id: str | None = None
    proposal_scope: str = "entity"
    action_type: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.GENERIC_TASK
    action_display_name: str = ""
    action_display_tagline: str = ""
    action_display_category: str = ""
    title: str
    summary: str
    trigger_reason: str = ""
    recommendation: str = ""
    evidence_summary: str = ""
    severity: str
    requires_confirmation: bool
    required_operator_roles: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    high_risk_object_ids: list[str] = Field(default_factory=list)
    action_scope: dict[str, Any] = Field(default_factory=dict)
    risk_stage_key: str | None = None
    system_version_hash: str = ""
    generation_source: GenerationSource = GenerationSource.SYSTEM
    model_name: str = ""
    prompt_profile: str = ""
    grounding_summary: str = ""
    chat_follow_up_prompt: str = ""
    source_session_id: str | None = None
    status: ProposalStatus = ProposalStatus.PENDING
    updated_at: datetime | None = None
    edited_by_commander: bool = False
    last_editor: str = "system"
    has_new_system_suggestion: bool = False
    superseded_by_proposal_id: str | None = None
    withdrawn_reason: str = ""
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolution_note: str = ""
    created_at: datetime


class ProposalResolutionRequest(V2Model):
    operator_id: str = "console_operator"
    operator_role: str = "commander"
    note: str = ""


class ProposalDraftUpdateRequest(V2Model):
    operator_id: str = "console_operator"
    operator_role: str = "commander"
    action_scope: dict[str, Any] = Field(default_factory=dict)


class BatchProposalResolutionRequest(ProposalResolutionRequest):
    proposal_ids: list[str] = Field(default_factory=list)


class AdvisoryRequest(V2Model):
    event_id: str | None = None
    entity_id: str | None = None
    area_id: str = "beilin_10km2"
    location_hint: str | None = None
    village: str | None = None
    profile_overrides: dict[str, Any] = Field(default_factory=dict)
    operator_role: str = "commander"


class Advisory(V2Model):
    advisory_id: str
    event_id: str
    entity_id: str | None = None
    answer: str
    impact_summary: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    route_options: list[RouteOption] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float
    confidence_explanation: str = ""
    requires_human_confirmation: bool = False
    missing_data: list[str] = Field(default_factory=list)
    proposal: ActionProposal | None = None
    generation_source: GenerationSource = GenerationSource.SYSTEM
    model_name: str = ""
    grounding_summary: str = ""
    generated_at: datetime


class ToolRetryPolicy(V2Model):
    max_attempts: int = 1
    retryable_failure_modes: list[ToolFailureMode] = Field(default_factory=list)


class ToolSpec(V2Model):
    tool_name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 0
    failure_modes: list[ToolFailureMode] = Field(default_factory=list)
    retry_policy: ToolRetryPolicy = Field(default_factory=ToolRetryPolicy)
    fallback_tools: list[str] = Field(default_factory=list)
    parallel_group: str | None = None
    staleness_budget_seconds: int | None = None
    produces_memory_updates: bool = False


class SkippedToolReason(V2Model):
    tool_name: str
    reason: str


class ToolExecutionResult(V2Model):
    execution_id: str | None = None
    tool_name: str
    status: ToolExecutionStatus
    input: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    raw_output: Any = None
    failure_reason: str | None = None
    duration_ms: int = 0
    timed_out: bool = False
    data_freshness_seconds: int | None = None
    attempt: int = 1
    retry_of_execution_id: str | None = None
    fallback_from_tool: str | None = None
    dependency_tools: list[str] = Field(default_factory=list)
    stale: bool = False
    cache_hit: bool = False
    parallel_group: str | None = None


class PlannerRequestContext(V2Model):
    question: str
    event_id: str
    memory_snapshot: "MemorySnapshot | None" = None
    available_tools: list[str] = Field(default_factory=list)
    tool_whitelist: list[str] = Field(default_factory=list)
    recent_failures: list[str] = Field(default_factory=list)


class PlannerSuggestion(V2Model):
    planning_layer: PlanningLayer = PlanningLayer.LLM
    selected_tools: list[str] = Field(default_factory=list)
    tool_selection_reasoning: list[str] = Field(default_factory=list)
    skipped_tools: list[SkippedToolReason] = Field(default_factory=list)
    plan_notes: list[str] = Field(default_factory=list)
    invalid_reason: str | None = None


class MergedPlanDecision(V2Model):
    selected_tools: list[str] = Field(default_factory=list)
    tool_selection_reasoning: list[str] = Field(default_factory=list)
    skipped_tools: list[SkippedToolReason] = Field(default_factory=list)
    plan_notes: list[str] = Field(default_factory=list)
    llm_applied: bool = False
    llm_status: str = "rule_only"


class CopilotExecutionPlan(V2Model):
    plan_id: str = ""
    planning_layer: PlanningLayer = PlanningLayer.RULE
    intent: str
    target_entity_resolution: str = ""
    target_resolution_mode: str = ""
    target_entity_id: str | None = None
    target_entity_name: str | None = None
    target_entity_type: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    optional_tools: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    tool_selection_reasoning: list[str] = Field(default_factory=list)
    skipped_tools: list[SkippedToolReason] = Field(default_factory=list)
    plan_notes: list[str] = Field(default_factory=list)
    replan_round: int = 0
    parent_plan_id: str | None = None


class DataFreshnessSummary(V2Model):
    hazard_state_freshness_seconds: int | None = None
    traffic_freshness_seconds: int | None = None
    profile_freshness_label: str | None = None
    rag_document_recency_summary: str | None = None


class ToolTraceStep(V2Model):
    tool_name: str
    summary: str


class CompletionAssessment(V2Model):
    status: CompletionStatus = CompletionStatus.DIRECT_ANSWER
    termination_reason: str = ""
    should_replan: bool = False
    evidence_gaps: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)


class MemorySnapshot(V2Model):
    session_id: str
    focus_entity_id: str | None = None
    focus_entity_name: str | None = None
    focus_area_id: str | None = None
    current_goal: str | None = None
    pending_proposal_ids: list[str] = Field(default_factory=list)
    executed_proposal_ids: list[str] = Field(default_factory=list)
    unresolved_slots: list[str] = Field(default_factory=list)
    last_completion_status: CompletionStatus | None = None
    updated_at: datetime | None = None


class MemoryEventRecord(V2Model):
    memory_event_id: str
    session_id: str
    event_type: MemoryEventType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class PlanRunRecord(V2Model):
    plan_run_id: str
    session_id: str
    event_id: str
    message_id: str | None = None
    plan_id: str
    planning_layer: PlanningLayer
    replan_round: int = 0
    parent_plan_id: str | None = None
    intent: str
    target_entity_id: str | None = None
    target_entity_name: str | None = None
    selected_tools: list[str] = Field(default_factory=list)
    tool_selection_reasoning: list[str] = Field(default_factory=list)
    skipped_tools: list[SkippedToolReason] = Field(default_factory=list)
    plan_notes: list[str] = Field(default_factory=list)
    created_at: datetime


class AgentTask(V2Model):
    task_id: str
    event_id: str
    agent_name: AgentName
    task_type: str
    status: AgentTaskStatus = AgentTaskStatus.PENDING
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    session_id: str | None = None
    parent_task_id: str | None = None
    replayed_from_task_id: str | None = None
    replay_reason: str | None = None
    source_trigger_id: str | None = None
    failure_reason: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AgentResult(V2Model):
    result_id: str
    task_id: str
    event_id: str
    agent_name: AgentName
    summary: str
    structured_output: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    decision_confidence: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    handoff_recommendations: list[str] = Field(default_factory=list)
    recommended_next_tasks: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    supersedes_task_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class SharedMemorySnapshot(V2Model):
    event_id: str
    autonomy_level: AutonomyLevel = AutonomyLevel.AUTO_OBSERVE
    active_agents: list[AgentName] = Field(default_factory=list)
    focus_entity_ids: list[str] = Field(default_factory=list)
    focus_entity_names: list[str] = Field(default_factory=list)
    top_risks: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    pending_proposal_ids: list[str] = Field(default_factory=list)
    recent_result_ids: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    active_decision_path: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    latest_hazard_level: RiskLevel | None = None
    latest_summary: str = ""
    last_trigger: str = ""
    updated_at: datetime


class OutcomeSignal(V2Model):
    signal_id: str
    signal_type: str
    event_id: str | None = None
    entity_id: str | None = None
    proposal_id: str | None = None
    severity: str = "info"
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ExperienceRecord(V2Model):
    experience_id: str
    event_id: str
    entity_id: str | None = None
    entity_type: str | None = None
    risk_level: RiskLevel | None = None
    action_type: str
    action_summary: str
    outcome: str
    confidence: float = 0.0
    tags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DailyReportRecord(V2Model):
    report_id: str
    event_id: str
    report_date: date
    timezone: str = "Asia/Shanghai"
    headline: str
    situation_summary: str = ""
    decisions_summary: str = ""
    action_summary: str = ""
    unresolved_risks: list[str] = Field(default_factory=list)
    next_day_recommendations: list[str] = Field(default_factory=list)
    generation_source: GenerationSource = GenerationSource.SYSTEM
    model_name: str = ""
    grounding_summary: str = ""
    delivered_session_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class DailyReportRunRecord(V2Model):
    run_id: str
    event_id: str
    report_date: date
    timezone: str = "Asia/Shanghai"
    status: str = "completed"
    report_id: str | None = None
    created_at: datetime


class HighRiskEpisodeRecord(V2Model):
    episode_id: str
    event_id: str
    started_at: datetime
    ended_at: datetime | None = None
    peak_risk_level: RiskLevel
    trigger_source: str = ""
    status: HighRiskEpisodeStatus = HighRiskEpisodeStatus.OPEN
    created_at: datetime
    updated_at: datetime


class EventEpisodeSummaryRecord(V2Model):
    summary_id: str
    episode_id: str
    event_id: str
    headline: str
    escalation_path: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    successful_actions: list[str] = Field(default_factory=list)
    failed_actions: list[str] = Field(default_factory=list)
    coordination_gaps: list[str] = Field(default_factory=list)
    reusable_rules: list[str] = Field(default_factory=list)
    memory_tags: list[str] = Field(default_factory=list)
    created_at: datetime


class LongTermMemoryRecord(V2Model):
    memory_id: str
    event_id: str
    source_summary_id: str
    memory_type: str = "high_risk_postmortem"
    area_id: str
    risk_level: RiskLevel | None = None
    entity_types: list[str] = Field(default_factory=list)
    action_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    headline: str
    summary: str
    retrieval_text: str
    lessons: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    created_at: datetime


class StrategyPattern(V2Model):
    pattern_id: str
    entity_type: str | None = None
    risk_level: RiskLevel | None = None
    action_type: str
    sample_size: int = 0
    approval_rate: float = 0.0
    execution_success_rate: float = 0.0
    recommended_summary: str = ""
    common_failures: list[str] = Field(default_factory=list)
    supporting_experience_ids: list[str] = Field(default_factory=list)


class ExperienceContextView(V2Model):
    event_id: str
    relevant_records: list[ExperienceRecord] = Field(default_factory=list)
    strategy_patterns: list[StrategyPattern] = Field(default_factory=list)
    outcome_risk_notes: list[str] = Field(default_factory=list)
    long_term_memories: list[LongTermMemoryRecord] = Field(default_factory=list)


class StrategyHistoryView(V2Model):
    entity_id: str
    records: list[ExperienceRecord] = Field(default_factory=list)
    patterns: list[StrategyPattern] = Field(default_factory=list)


class EvaluationBenchmark(V2Model):
    benchmark_id: str
    title: str
    question: str
    scenario_type: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_completion_status: CompletionStatus | None = None
    expected_human_confirmation: bool = False


class BenchmarkScenarioResult(V2Model):
    benchmark_id: str
    title: str
    passed: bool = False
    event_id: str
    session_id: str
    used_tools: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    completion_status: CompletionStatus | None = None
    expected_completion_status: CompletionStatus | None = None
    human_confirmation: bool = False
    expected_human_confirmation: bool = False
    evidence_count: int = 0
    shared_memory_reused: bool = False
    notes: list[str] = Field(default_factory=list)


class EvaluationReport(V2Model):
    report_id: str
    created_at: datetime
    benchmark_count: int = 0
    tool_selection_correctness: float = 0.0
    dynamic_dispatch_correctness: float = 0.0
    shared_memory_reuse_rate: float = 0.0
    evidence_coverage_rate: float = 0.0
    human_escalation_correctness: float = 0.0
    hallucination_rate: float = 0.0
    scenario_results: list[BenchmarkScenarioResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OperatorCapabilitiesView(V2Model):
    operator_role: str
    role_rank: int
    capabilities: dict[str, bool] = Field(default_factory=dict)
    action_labels: dict[str, str] = Field(default_factory=dict)


class AgentMetricsView(V2Model):
    generated_at: datetime
    task_graph_latency_ms: float = 0.0
    agent_failure_heatmap: dict[str, int] = Field(default_factory=dict)
    stale_data_frequency: int = 0
    auto_retry_success_rate: float = 0.0
    superseded_task_ratio: float = 0.0
    fanout_count: int = 0
    stale_data_replan_count: int = 0


class DecisionReportView(V2Model):
    event_id: str
    latest_summary: str = ""
    active_decision_path: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recent_result_ids: list[str] = Field(default_factory=list)


class TriggerEvent(V2Model):
    trigger_id: str
    event_id: str
    trigger_type: TriggerEventType
    status: TriggerEventStatus = TriggerEventStatus.PENDING
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str | None = None
    error_message: str | None = None
    created_at: datetime
    leased_at: datetime | None = None
    processed_at: datetime | None = None


class AgentTaskEvent(V2Model):
    task_event_id: str
    event_id: str
    task_id: str
    agent_name: AgentName
    event_type: AgentTaskEventType
    trigger_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AgentTimelineEntry(V2Model):
    entry_id: str
    event_id: str
    entry_type: str
    task_id: str | None = None
    task_event_type: AgentTaskEventType | None = None
    trigger_id: str | None = None
    trigger_type: TriggerEventType | None = None
    agent_name: AgentName | None = None
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class SupervisorRunRecord(V2Model):
    supervisor_run_id: str
    event_id: str
    trigger_type: str
    autonomy_level: AutonomyLevel = AutonomyLevel.AUTO_OBSERVE
    status: SupervisorRunStatus = SupervisorRunStatus.RUNNING
    session_id: str | None = None
    summary: str = ""
    created_tasks: list[str] = Field(default_factory=list)
    completed_task_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None


class SupervisorHealthState(V2Model):
    component_key: str = "supervisor_loop"
    running: bool = False
    interval_seconds: float = 60.0
    consecutive_failures: int = 0
    retries_used_in_last_cycle: int = 0
    skipped_sweeps: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    last_started_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_retry_at: datetime | None = None
    last_completed_at: datetime | None = None
    last_error: str | None = None
    circuit_opened_at: datetime | None = None
    circuit_expires_at: datetime | None = None
    pending_trigger_count: int = 0
    last_trigger_processed_at: datetime | None = None
    recent_replay_count: int = 0
    recent_timeline_failure_count: int = 0
    updated_at: datetime


class OperationalAlert(V2Model):
    alert_id: str
    source_type: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.OPEN
    summary: str
    details: str = ""
    event_id: str | None = None
    first_seen_at: datetime
    last_seen_at: datetime
    resolved_at: datetime | None = None


class AuditRecord(V2Model):
    audit_id: str
    source_type: str
    action: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.INFO
    event_id: str | None = None
    session_id: str | None = None
    created_at: datetime


class ArchiveRunRecord(V2Model):
    archive_run_id: str
    status: str
    hot_records_archived: int = 0
    expired_archives_deleted: int = 0
    tables_touched: list[str] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


class ArchiveStatusView(V2Model):
    hot_retention_days: int = 14
    archive_retention_days: int = 180
    hot_record_count: int = 0
    archived_record_count: int = 0
    last_archive_run: ArchiveRunRecord | None = None
    latest_archive_runs: list[ArchiveRunRecord] = Field(default_factory=list)


class AutonomyDecision(V2Model):
    autonomy_level: AutonomyLevel
    reason: str


class ToolExecutionAuditRecord(ToolExecutionResult):
    session_id: str
    event_id: str
    message_id: str | None = None
    plan_id: str | None = None
    replan_round: int = 0
    created_at: datetime


class CopilotStructuredAnswer(V2Model):
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    impact_summary: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    follow_up_prompts: list[str] = Field(default_factory=list)
    confidence: float
    confidence_explanation: str = ""
    requires_human_confirmation: bool = False
    missing_data: list[str] = Field(default_factory=list)
    proposal: ActionProposal | None = None
    generation_source: GenerationSource = GenerationSource.SYSTEM
    model_name: str = ""
    grounding_summary: str = ""
    planner_summary: str = ""
    tool_selection_reasoning: list[str] = Field(default_factory=list)
    skipped_tools: list[SkippedToolReason] = Field(default_factory=list)
    tool_executions: list[ToolExecutionResult] = Field(default_factory=list)
    data_freshness: DataFreshnessSummary = Field(default_factory=DataFreshnessSummary)
    evidence_gaps: list[str] = Field(default_factory=list)
    tool_trace: list[ToolTraceStep] = Field(default_factory=list)
    planning_layers_summary: list[str] = Field(default_factory=list)
    plan_runs: list[PlanRunRecord] = Field(default_factory=list)
    completion_status: CompletionStatus = CompletionStatus.DIRECT_ANSWER
    termination_reason: str = ""
    memory_snapshot: MemorySnapshot | None = None
    replan_count: int = 0
    used_fallbacks: list[str] = Field(default_factory=list)
    carried_context_notes: list[str] = Field(default_factory=list)
    experience_summary: str = ""
    historical_pattern_refs: list[str] = Field(default_factory=list)
    outcome_risk_notes: list[str] = Field(default_factory=list)


class NotificationDraft(V2Model):
    draft_id: str
    event_id: str
    proposal_id: str
    entity_id: str | None = None
    area_id: str | None = None
    audience: str
    channel: str
    content: str
    status: str = "draft"
    generation_source: GenerationSource = GenerationSource.SYSTEM
    model_name: str = ""
    grounding_summary: str = ""
    created_at: datetime


class ExecutionLogEntry(V2Model):
    log_id: str
    event_id: str
    proposal_id: str
    entity_id: str | None = None
    area_id: str | None = None
    action_type: str
    summary: str
    operator_id: str
    details: dict[str, Any] = Field(default_factory=dict)
    generation_source: GenerationSource = GenerationSource.SYSTEM
    model_name: str = ""
    grounding_summary: str = ""
    created_at: datetime


class RegionalProposalView(V2Model):
    proposal: ActionProposal
    event_title: str
    current_risk_level: RiskLevel
    high_risk_object_names: list[str] = Field(default_factory=list)


class RegionalProposalQueueSnapshot(V2Model):
    queue_version: str
    generated_at: datetime
    items: list[RegionalProposalView] = Field(default_factory=list)


class RegionalAnalysisPackageView(V2Model):
    package_id: str
    event_id: str
    current_risk_level: RiskLevel
    trigger_type: str = "simulation_updated"
    focus_object_ids: list[str] = Field(default_factory=list)
    focus_object_names: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    proposal_titles: list[str] = Field(default_factory=list)
    proposal_count: int = 0
    analysis_message: str = ""
    risk_assessment: str = ""
    rescue_plan: str = ""
    resource_dispatch_plan: str = ""
    status: RegionalAnalysisPackageStatus = RegionalAnalysisPackageStatus.PENDING
    created_at: datetime
    updated_at: datetime


class V2CopilotSessionRequest(V2Model):
    event_id: str
    operator_role: str = "commander"


class V2CopilotMessageRequest(V2Model):
    content: str


class V2CopilotMessage(V2Model):
    message_id: str
    role: str
    content: str
    created_at: datetime
    structured_answer: CopilotStructuredAnswer | None = None


class V2CopilotSessionView(V2Model):
    session_id: str
    event: EventRecord
    messages: list[V2CopilotMessage] = Field(default_factory=list)
    latest_answer: CopilotStructuredAnswer | None = None
    proposals: list[ActionProposal] = Field(default_factory=list)
    notification_drafts: list[NotificationDraft] = Field(default_factory=list)
    execution_logs: list[ExecutionLogEntry] = Field(default_factory=list)
    daily_reports: list[DailyReportRecord] = Field(default_factory=list)
    episode_summaries: list[EventEpisodeSummaryRecord] = Field(default_factory=list)
    memory_snapshot: MemorySnapshot | None = None
    plan_runs: list[PlanRunRecord] = Field(default_factory=list)
    recent_tool_executions: list[ToolExecutionAuditRecord] = Field(default_factory=list)
    shared_memory_snapshot: SharedMemorySnapshot | None = None
    session_memory_snapshot: MemorySnapshot | None = None
    active_agents: list[AgentName] = Field(default_factory=list)
    recent_agent_results: list[AgentResult] = Field(default_factory=list)
    autonomy_level: AutonomyLevel = AutonomyLevel.AUTO_OBSERVE
    pending_regional_analysis_package: RegionalAnalysisPackageView | None = None
    regional_analysis_package_history: list[RegionalAnalysisPackageView] = Field(default_factory=list)


class SessionMemoryView(V2Model):
    session_id: str
    memory_snapshot: MemorySnapshot
    recent_events: list[MemoryEventRecord] = Field(default_factory=list)


class MemoryBundleView(V2Model):
    session_memory: SessionMemoryView | None = None
    event_shared_memory: SharedMemorySnapshot | None = None


class ReplayRequest(V2Model):
    replay_reason: str = ""


class EventSnapshot(V2Model):
    event: EventRecord
    latest_hazard_state: HazardState | None = None
    latest_exposure_summary: ExposureSummary | None = None
    recent_stream: list[EventStreamRecord] = Field(default_factory=list)


class EntityProfileUpsertRequest(V2Model):
    profile: EntityProfile
    operator_id: str = "console_operator"
    operator_role: str = "commander"


class ResourceStatusUpdateRequest(V2Model):
    resource_status: ResourceStatus
    operator_id: str = "console_operator"
    operator_role: str = "commander"


class ResourceStatusView(V2Model):
    scope: str
    area_id: str
    event_id: str | None = None
    resource_status: ResourceStatus


class RAGDocumentImportItem(V2Model):
    doc_id: str
    corpus: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_document(self) -> RAGDocument:
        return RAGDocument(
            doc_id=self.doc_id,
            corpus=self.corpus,
            title=self.title,
            content=self.content,
            metadata=self.metadata,
        )


class RAGDocumentImportRequest(V2Model):
    documents: list[RAGDocumentImportItem] = Field(default_factory=list)
    operator_id: str = "console_operator"
    operator_role: str = "commander"


class DatasetFetchRequest(V2Model):
    download: bool = True
    source_ids: list[str] = Field(default_factory=list)
    force_refresh: bool = False


class DatasetBuildRequest(V2Model):
    download: bool = False
    sync_demo_db: bool = True


class DatasetSyncRequest(V2Model):
    db_path: str | None = None


class DatasetJobView(V2Model):
    job_id: str
    action: str
    status: str
    progress_percent: int = 0
    current_step: str = ""
    message: str = ""
    source_ids: list[str] = Field(default_factory=list)
    request_payload: dict[str, Any] = Field(default_factory=dict)
    attempt_count: int = 0
    max_attempts: int = 1
    auto_retry_enabled: bool = True
    retry_count: int = 0
    retry_of_job_id: str | None = None
    cancel_requested: bool = False
    started_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    canceled_at: datetime | None = None
    error: str | None = None
    result_summary: str | None = None


class DatasetSourceStatusView(V2Model):
    source_id: str
    title: str
    category: str
    source_type: str
    source_ref: str
    download_url: str = ""
    cache_status: str
    cached_files: list[str] = Field(default_factory=list)
    notes: str = ""
    last_fetched_at: datetime | None = None
    parser_kind: str = ""
    artifact_count: int = 0
    downloaded_artifact_count: int = 0
    failed_artifact_count: int = 0
    progress_percent: int = 0
    retryable: bool = False
    last_error: str | None = None
    parsed_summary: dict[str, Any] = Field(default_factory=dict)
    latest_fetch_details: list[dict[str, Any]] = Field(default_factory=list)
    completeness_status: str = "missing"
    artifacts_manifest_path: str | None = None
    versions_manifest_path: str | None = None
    parsed: bool = False
    required: bool = True
    missing_artifact_types: list[str] = Field(default_factory=list)
    raw_file_count: int = 0
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class RawCacheHealthView(V2Model):
    source_id: str
    title: str
    completeness_status: str
    cache_status: str
    parsed: bool = False
    required: bool = True
    raw_file_count: int = 0
    downloaded_artifact_count: int = 0
    failed_artifact_count: int = 0
    last_fetched_at: datetime | None = None
    missing_artifact_types: list[str] = Field(default_factory=list)
    last_error: str | None = None


class DatasetPipelineStatusView(V2Model):
    area_id: str
    raw_dir: str
    normalized_dir: str
    bootstrap_dir: str
    runtime_rag_path: str
    source_count: int = 0
    cached_source_count: int = 0
    failed_source_count: int = 0
    cached_file_count: int = 0
    raw_ready: bool = False
    raw_completeness_percent: int = 0
    missing_required_sources: list[str] = Field(default_factory=list)
    stale_sources: list[str] = Field(default_factory=list)
    sources: list[DatasetSourceStatusView] = Field(default_factory=list)
    raw_cache_health: list[RawCacheHealthView] = Field(default_factory=list)
    latest_download_log: list[dict[str, Any]] = Field(default_factory=list)
    latest_fetch_summary: dict[str, Any] = Field(default_factory=dict)
    latest_build_summary: dict[str, Any] = Field(default_factory=dict)
    latest_validation: dict[str, Any] = Field(default_factory=dict)
    normalized_files: list[str] = Field(default_factory=list)
    bootstrap_files: list[str] = Field(default_factory=list)
    active_job: DatasetJobView | None = None
    recent_jobs: list[DatasetJobView] = Field(default_factory=list)


class DatasetOperationResult(V2Model):
    action: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    status: DatasetPipelineStatusView
