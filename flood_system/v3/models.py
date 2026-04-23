from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ..models import RiskLevel
from ..v2.models import EvidenceItem, NotificationDraft, RegionalProposalView


class V3Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class TwinFocusObjectSummary(V3Model):
    object_id: str
    name: str
    entity_type: str
    village: str
    risk_level: RiskLevel
    time_to_impact_minutes: int
    summary: str
    recommended_action: str
    pending_proposal_ids: list[str] = Field(default_factory=list)
    canvas_position: dict[str, float] = Field(default_factory=dict)


class TwinObjectMapLayer(V3Model):
    object_id: str
    name: str
    risk_level: RiskLevel
    entity_type: str
    east_offset_m: float = 0.0
    north_offset_m: float = 0.0
    height_offset_m: float = 0.0
    proposal_state: str = "idle"
    is_lead: bool = False


class TwinSignalView(V3Model):
    signal_id: str
    title: str
    detail: str
    severity: str
    created_at: datetime


class TwinOverviewView(V3Model):
    event_id: str
    area_id: str
    event_title: str
    generated_at: datetime
    overall_risk_level: RiskLevel
    trend: str
    summary: str
    lead_object_id: str | None = None
    lead_object_name: str | None = None
    focus_objects: list[TwinFocusObjectSummary] = Field(default_factory=list)
    map_layers: list[TwinObjectMapLayer] = Field(default_factory=list)
    pending_proposal_count: int = 0
    approved_proposal_count: int = 0
    warning_draft_count: int = 0
    active_alert_count: int = 0
    recommended_actions: list[str] = Field(default_factory=list)
    signals: list[TwinSignalView] = Field(default_factory=list)
    recent_warning_drafts: list["AudienceWarningDraft"] = Field(default_factory=list)


class FocusObjectView(V3Model):
    event_id: str
    object_id: str
    object_name: str
    entity_type: str
    village: str
    risk_level: RiskLevel
    time_to_impact_minutes: int
    summary: str
    risk_reasons: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    risk_reminders: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    related_proposals: list[RegionalProposalView] = Field(default_factory=list)


class AgentDialogRequest(V3Model):
    object_id: str | None = None
    message: str


class V3ProposalDraft(V3Model):
    blocked: bool = False
    block_reason: str | None = None
    proposal: RegionalProposalView | None = None


class AgentDialogResponse(V3Model):
    event_id: str
    object_id: str
    object_name: str
    message: str
    answer: str
    impact_summary: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    risk_reminders: list[str] = Field(default_factory=list)
    follow_up_prompts: list[str] = Field(default_factory=list)
    grounding_summary: str = ""
    proposal_entry: V3ProposalDraft | None = None
    response_source: str = "llm"
    generated_at: datetime


class AgentCouncilRoleView(V3Model):
    role: str
    label: str
    status: str
    summary: str
    confidence: float | None = None
    evidence_count: int = 0
    recommended_action: str | None = None


class AuditDecisionView(V3Model):
    status: str
    summary: str
    rationale: str
    risk_flags: list[str] = Field(default_factory=list)
    approval_required: bool = True


class AgentCouncilView(V3Model):
    event_id: str
    generated_at: datetime
    overall_summary: str
    decision_path: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    roles: list[AgentCouncilRoleView] = Field(default_factory=list)
    audit_decision: AuditDecisionView
    recent_result_ids: list[str] = Field(default_factory=list)


class ProposalGenerationRequest(V3Model):
    object_ids: list[str] = Field(default_factory=list)


class ProposalGenerationResponse(V3Model):
    event_id: str
    queue_version: str
    generated_at: datetime
    blocked: bool = False
    block_reason: str | None = None
    proposals: list[V3ProposalDraft] = Field(default_factory=list)


class AudienceWarningDraft(V3Model):
    warning_id: str
    event_id: str
    proposal_id: str
    audience: str
    channel: str
    content: str
    grounding_summary: str = ""
    created_at: datetime
    source_draft_id: str | None = None

    @classmethod
    def from_notification_draft(cls, draft: NotificationDraft) -> "AudienceWarningDraft":
        return cls(
            warning_id=f"warn_{draft.draft_id}",
            event_id=draft.event_id,
            proposal_id=draft.proposal_id,
            audience=draft.audience,
            channel=draft.channel,
            content=draft.content,
            grounding_summary=draft.grounding_summary,
            created_at=draft.created_at,
            source_draft_id=draft.draft_id,
        )


class WarningGenerationResponse(V3Model):
    event_id: str
    proposal_id: str
    generated_at: datetime
    warnings: list[AudienceWarningDraft] = Field(default_factory=list)


class TwinStreamEvent(V3Model):
    event_type: str
    version: str
    created_at: datetime
    payload: dict = Field(default_factory=dict)
