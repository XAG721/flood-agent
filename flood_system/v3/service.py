from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any
from uuid import uuid4

from ..v2.llm_gateway import LLMGenerationError
from ..v2.models import (
    AdvisoryRequest,
    AlertSeverity,
    GenerationSource,
    NotificationDraft,
    ProposalStatus,
)
from .models import (
    AgentCouncilRoleView,
    AgentCouncilView,
    AgentDialogRequest,
    AgentDialogResponse,
    AuditDecisionView,
    AudienceWarningDraft,
    FocusObjectView,
    ProposalGenerationRequest,
    ProposalGenerationResponse,
    TwinFocusObjectSummary,
    TwinObjectMapLayer,
    TwinOverviewView,
    TwinSignalView,
    TwinStreamEvent,
    V3ProposalDraft,
    WarningGenerationResponse,
)


@dataclass
class AgentTwinService:
    platform: Any
    repository: Any

    def get_twin_overview(self, event_id: str) -> TwinOverviewView:
        event = self.platform.get_event(event_id)
        hazard_state = self.platform.get_hazard_state(event_id)
        exposure = self.platform.get_exposure_summary(event_id, top_k=6)
        proposals = self.repository.list_v2_action_proposals(event_id, proposal_scope="regional")
        proposal_views = self.platform.list_regional_proposals(event_id)
        pending_proposals = [item for item in proposal_views if item.proposal.status == ProposalStatus.PENDING]
        approved_proposals = [item for item in proposal_views if item.proposal.status == ProposalStatus.APPROVED]
        alerts = self.platform.list_operational_alerts(event_id=event_id, limit=4)
        warnings = self.list_audience_warnings(event_id)

        focus_objects = [
            self._focus_summary_from_impact(
                impact=item,
                proposal_views=proposal_views,
                index=index,
            )
            for index, item in enumerate(exposure.affected_entities)
        ]
        lead = focus_objects[0] if focus_objects else None
        recommended_actions = [
            item.proposal.recommendation or item.proposal.summary
            for item in pending_proposals[:3]
        ]
        if not recommended_actions and focus_objects:
            recommended_actions = [item.recommended_action for item in focus_objects[:3]]
        if not recommended_actions:
            recommended_actions = [
                "继续监测重点对象",
                "准备处置资源并保持人工审批",
            ]

        signals = [
            TwinSignalView(
                signal_id=alert.alert_id,
                title=alert.summary,
                detail=str(alert.details or ""),
                severity=alert.severity.value,
                created_at=alert.last_seen_at or alert.first_seen_at,
            )
            for alert in alerts
        ]
        if not signals:
            signals = [
                TwinSignalView(
                    signal_id=point.point_name,
                    title=f"{point.point_name} 监测点",
                    detail=f"水位 {point.latest_water_level_m:.2f} m，雨量 {point.latest_rainfall_mm:.1f} mm",
                    severity="info",
                    created_at=point.updated_at,
                )
                for point in hazard_state.monitoring_points[:4]
            ]

        summary = (
            f"{event.title} 当前处于 {hazard_state.overall_risk_level.value} 风险，"
            f"系统已识别 {len(focus_objects)} 个优先关注对象，"
            f"待审批 proposal {len(pending_proposals)} 条。"
        )

        return TwinOverviewView(
            event_id=event_id,
            area_id=event.area_id,
            event_title=event.title,
            generated_at=datetime.now(timezone.utc),
            overall_risk_level=hazard_state.overall_risk_level,
            trend=hazard_state.trend,
            summary=summary,
            lead_object_id=lead.object_id if lead else None,
            lead_object_name=lead.name if lead else None,
            focus_objects=focus_objects,
            map_layers=self._build_map_layers(
                focus_objects,
                proposal_views=proposal_views,
                warnings=warnings,
            ),
            pending_proposal_count=len(pending_proposals),
            approved_proposal_count=len(approved_proposals),
            warning_draft_count=len(warnings),
            active_alert_count=len(alerts),
            recommended_actions=recommended_actions,
            signals=signals,
            recent_warning_drafts=warnings[:4],
        )

    def get_focus_object(self, event_id: str, object_id: str) -> FocusObjectView:
        impact = self.platform.get_entity_impact(object_id, event_id=event_id)
        proposal_views = self.platform.list_regional_proposals(event_id)
        related_proposals = [
            item
            for item in proposal_views
            if item.proposal.entity_id == object_id or object_id in item.proposal.high_risk_object_ids
        ]
        try:
            advisory = self.platform.generate_advisory(
                AdvisoryRequest(
                    event_id=event_id,
                    area_id=impact.entity.area_id,
                    entity_id=object_id,
                    operator_role="commander",
                )
            )
            recommended_actions = advisory.recommended_actions
            summary = advisory.answer
        except LLMGenerationError:
            recommended_actions = self._fallback_actions_for_impact(impact)
            summary = impact.risk_reason[0] if impact.risk_reason else f"{impact.entity.name} 当前需要持续关注。"

        return FocusObjectView(
            event_id=event_id,
            object_id=object_id,
            object_name=impact.entity.name,
            entity_type=impact.entity.entity_type.value,
            village=impact.entity.village,
            risk_level=impact.risk_level,
            time_to_impact_minutes=impact.time_to_impact_minutes,
            summary=summary,
            risk_reasons=impact.risk_reason,
            recommended_actions=recommended_actions,
            risk_reminders=impact.resource_gap or ["高风险动作仍需人工审批。"],
            evidence=impact.evidence,
            related_proposals=related_proposals,
        )

    def run_dialog(self, event_id: str, request: AgentDialogRequest) -> AgentDialogResponse:
        overview = self.get_twin_overview(event_id)
        object_id = request.object_id or overview.lead_object_id
        if not object_id:
            raise ValueError("No focus object is available for the current event.")

        focus = self.get_focus_object(event_id, object_id)
        shared_memory = self.platform.get_shared_memory_snapshot(event_id)

        try:
            llm_output = self.platform.llm_gateway.generate_copilot_chat(
                {
                    "question": request.message,
                    "impact": self.platform.get_entity_impact(object_id, event_id=event_id).model_dump(mode="json"),
                    "top_risks": focus.risk_reasons,
                    "pending_proposals": [item.proposal.model_dump(mode="json") for item in focus.related_proposals[:3]],
                    "shared_memory_open_questions": list(shared_memory.open_questions),
                }
            )
            answer = llm_output.answer
            impact_summary = llm_output.impact_summary
            recommended_actions = llm_output.recommended_actions
            follow_up_prompts = llm_output.follow_up_prompts
            grounding_summary = llm_output.grounding_summary
            response_source = "llm"
        except LLMGenerationError as exc:
            answer = (
                f"当前模型暂时不可用，系统改为返回保守研判。"
                f"{focus.object_name} 目前最需要关注的是：{focus.risk_reasons[0] if focus.risk_reasons else focus.summary}"
            )
            impact_summary = focus.risk_reasons[:3]
            recommended_actions = focus.recommended_actions[:3]
            follow_up_prompts = [
                "请解释该对象当前的影响链。",
                "请生成可审批的行动 proposal。",
            ]
            grounding_summary = str(exc)
            response_source = "fallback"

        proposal_entry = None
        if focus.related_proposals:
            proposal_entry = V3ProposalDraft(proposal=focus.related_proposals[0])

        return AgentDialogResponse(
            event_id=event_id,
            object_id=focus.object_id,
            object_name=focus.object_name,
            message=request.message,
            answer=answer,
            impact_summary=impact_summary,
            evidence=focus.evidence,
            recommended_actions=recommended_actions,
            risk_reminders=focus.risk_reminders,
            follow_up_prompts=follow_up_prompts,
            grounding_summary=grounding_summary,
            proposal_entry=proposal_entry,
            response_source=response_source,
            generated_at=datetime.now(timezone.utc),
        )

    def get_agent_council(self, event_id: str) -> AgentCouncilView:
        shared_memory = self.platform.get_shared_memory_snapshot(event_id)
        agent_status = self.platform.get_agent_status(event_id)
        recent_results = self.platform.list_agent_results(event_id)[:6]
        proposal_views = self.platform.list_regional_proposals(event_id)
        pending_views = [item for item in proposal_views if item.proposal.status == ProposalStatus.PENDING]
        warning_drafts = self.list_audience_warnings(event_id)

        roles = [
            AgentCouncilRoleView(
                role="impact_agent",
                label="Impact Agent",
                status="active" if shared_memory.focus_entity_ids else "standby",
                summary=shared_memory.top_risks[0] if shared_memory.top_risks else "Waiting for more impact evidence.",
                confidence=recent_results[0].confidence if recent_results else None,
                evidence_count=sum(len(item.evidence_refs) for item in recent_results[:2]),
                recommended_action=shared_memory.recommended_actions[0] if shared_memory.recommended_actions else None,
            ),
            AgentCouncilRoleView(
                role="action_agent",
                label="Action Agent",
                status="active" if pending_views else "standby",
                summary=pending_views[0].proposal.recommendation if pending_views else "No pending action package has been generated yet.",
                confidence=float(pending_views[0].proposal.payload.get("decision_confidence", 0.0)) if pending_views else None,
                evidence_count=len(pending_views[0].high_risk_object_names) if pending_views else 0,
                recommended_action=pending_views[0].proposal.title if pending_views else None,
            ),
            AgentCouncilRoleView(
                role="warning_agent",
                label="Warning Agent",
                status="ready" if pending_views else "waiting",
                summary="Audience-specific warning drafts will be generated after approval." if pending_views else "Waiting for approved proposal to unlock warning drafting.",
                confidence=None,
                evidence_count=len(warning_drafts),
                recommended_action="Generate leadership, department, community, and public warning drafts.",
            ),
            AgentCouncilRoleView(
                role="audit_agent",
                label="Audit Agent",
                status="blocked" if shared_memory.blocked_by else "clear",
                summary=shared_memory.blocked_by[0] if shared_memory.blocked_by else "No blocking audit rule is currently active.",
                confidence=None,
                evidence_count=len(shared_memory.open_questions),
                recommended_action="Keep high-risk actions under commander approval.",
            ),
        ]

        return AgentCouncilView(
            event_id=event_id,
            generated_at=datetime.now(timezone.utc),
            overall_summary=(
                agent_status.get("latest_summary")
                if isinstance(agent_status, dict)
                else getattr(agent_status, "latest_summary", None)
            )
            or shared_memory.latest_summary,
            decision_path=list(shared_memory.active_decision_path),
            open_questions=list(shared_memory.open_questions),
            blocked_by=list(shared_memory.blocked_by),
            roles=roles,
            audit_decision=AuditDecisionView(
                status="blocked" if shared_memory.blocked_by else "approved_for_review",
                summary=(
                    "AuditAgent blocked automatic escalation until missing constraints are resolved."
                    if shared_memory.blocked_by
                    else "AuditAgent allows proposals to enter the human approval queue."
                ),
                rationale=" | ".join(shared_memory.blocked_by) if shared_memory.blocked_by else "Evidence and constraints are sufficient for commander review.",
                risk_flags=list(shared_memory.blocked_by or shared_memory.open_questions[:2]),
                approval_required=True,
            ),
            recent_result_ids=list(shared_memory.recent_result_ids),
        )

    def generate_proposals(self, event_id: str, request: ProposalGenerationRequest | None = None) -> ProposalGenerationResponse:
        event = self.platform.get_event(event_id)
        hazard_state = self.platform.get_hazard_state(event_id)
        exposure = self.platform.get_exposure_summary(event_id)
        evidence = self.platform.get_knowledge_evidence(event_id=event_id, area_id=event.area_id)

        if hazard_state.overall_risk_level.value not in {"Orange", "Red"}:
            reason = "当前总体风险未达到生成高优先级 proposal 的阈值。"
            self.platform.add_audit_record(
                source_type="v3_audit_agent",
                action="proposal_blocked",
                summary=reason,
                details={"event_id": event_id, "risk_level": hazard_state.overall_risk_level.value},
                severity=AlertSeverity.WARNING,
                event_id=event_id,
            )
            return ProposalGenerationResponse(
                event_id=event_id,
                queue_version=self.platform.get_pending_regional_proposals_snapshot().queue_version,
                generated_at=datetime.now(timezone.utc),
                blocked=True,
                block_reason=reason,
            )

        if not exposure.affected_entities or not evidence:
            reason = "当前证据不足，AuditAgent 阻断 proposal 进入待审批队列。"
            self.platform.add_audit_record(
                source_type="v3_audit_agent",
                action="proposal_blocked",
                summary=reason,
                details={"event_id": event_id, "has_exposure": bool(exposure.affected_entities), "has_evidence": bool(evidence)},
                severity=AlertSeverity.WARNING,
                event_id=event_id,
            )
            return ProposalGenerationResponse(
                event_id=event_id,
                queue_version=self.platform.get_pending_regional_proposals_snapshot().queue_version,
                generated_at=datetime.now(timezone.utc),
                blocked=True,
                block_reason=reason,
            )

        result = self.platform.reconcile_regional_proposals(event_id)
        views = [
            item
            for item in self.platform.get_pending_regional_proposals_snapshot().items
            if item.proposal.event_id == event_id
        ]
        drafts = [V3ProposalDraft(proposal=item) for item in views]

        self.platform.repository.add_v2_stream_record_for_payload(
            event_id,
            "plan_proposed",
            {
                "source": "v3_proposal_generation",
                "proposal_ids": [item.proposal.proposal_id for item in views],
                "queue_version": self.platform.get_pending_regional_proposals_snapshot().queue_version,
            },
        )
        self.platform.add_audit_record(
            source_type="v3_orchestrator",
            action="proposal_generated",
            summary=f"V3 已为事件 {event_id} 生成 {len(drafts)} 条待审批 proposal。",
            details={"event_id": event_id, "llm_status": result.get("llm_status", "ok")},
            event_id=event_id,
        )

        return ProposalGenerationResponse(
            event_id=event_id,
            queue_version=self.platform.get_pending_regional_proposals_snapshot().queue_version,
            generated_at=datetime.now(timezone.utc),
            blocked=False,
            proposals=drafts,
        )

    def generate_warnings(self, proposal_id: str) -> WarningGenerationResponse:
        proposal = self.repository.get_v2_action_proposal(proposal_id)
        if proposal is None:
            raise ValueError("proposal not found.")
        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError("warnings can only be generated for approved proposals.")

        existing = self.repository.list_v3_audience_warnings(proposal.event_id, proposal_id=proposal_id)
        if not existing:
            mirrored_existing = [
                AudienceWarningDraft.from_notification_draft(item)
                for item in self.repository.list_v2_notification_drafts(proposal.event_id)
                if item.proposal_id == proposal_id
            ]
            if mirrored_existing:
                existing = mirrored_existing
                for item in existing:
                    self.repository.save_v3_audience_warning(item)
            else:
                event = self.platform.get_event(proposal.event_id)
                drafts, logs = self.platform.notification_gateway.build_regional_execution_bundle(
                    event_id=proposal.event_id,
                    area_id=event.area_id,
                    proposal=proposal,
                    operator_id="v3_warning_center",
                    event_title=event.title,
                )
                for draft in drafts:
                    self.repository.save_v2_notification_draft(draft)
                for log in logs:
                    self.repository.save_v2_execution_log(log)
                existing = [AudienceWarningDraft.from_notification_draft(item) for item in drafts]
                for item in existing:
                    self.repository.save_v3_audience_warning(item)
        if not existing:
            proposal_title = proposal.title or "当前区域动作"
            fallback_drafts = [
                NotificationDraft(
                    draft_id=f"draft_{uuid4().hex[:10]}",
                    event_id=proposal.event_id,
                    proposal_id=proposal.proposal_id,
                    entity_id=proposal.entity_id,
                    area_id=proposal.area_id,
                    audience="district_public",
                    channel="console",
                    content=f"请关注：{proposal_title} 已进入执行准备阶段，相关部门正在核对影响对象和处置范围。",
                    generation_source=GenerationSource.SYSTEM,
                    grounding_summary=proposal.grounding_summary or proposal.summary,
                    created_at=datetime.now(timezone.utc),
                ),
                NotificationDraft(
                    draft_id=f"draft_{uuid4().hex[:10]}",
                    event_id=proposal.event_id,
                    proposal_id=proposal.proposal_id,
                    entity_id=proposal.entity_id,
                    area_id=proposal.area_id,
                    audience="operations_desk",
                    channel="briefing",
                    content=f"值守提醒：围绕《{proposal_title}》准备执行材料，并复核对象、路线、资源和审批边界。",
                    generation_source=GenerationSource.SYSTEM,
                    grounding_summary=proposal.grounding_summary or proposal.summary,
                    created_at=datetime.now(timezone.utc),
                ),
            ]
            for draft in fallback_drafts:
                self.repository.save_v2_notification_draft(draft)
            existing = [AudienceWarningDraft.from_notification_draft(item) for item in fallback_drafts]
            for item in existing:
                self.repository.save_v3_audience_warning(item)

        self.platform.repository.add_v2_stream_record_for_payload(
            proposal.event_id,
            "notification_sent",
            {
                "source": "v3_warning_generation",
                "proposal_id": proposal_id,
                "warning_count": len(existing),
            },
        )
        self.platform.add_audit_record(
            source_type="v3_warning_center",
            action="warnings_generated",
            summary=f"已为 proposal {proposal_id} 生成 {len(existing)} 份分众预警草稿。",
            details={"proposal_id": proposal_id, "warning_count": len(existing)},
            event_id=proposal.event_id,
        )

        return WarningGenerationResponse(
            event_id=proposal.event_id,
            proposal_id=proposal_id,
            generated_at=datetime.now(timezone.utc),
            warnings=existing,
        )

    def list_audience_warnings(self, event_id: str) -> list[AudienceWarningDraft]:
        warnings = self.repository.list_v3_audience_warnings(event_id)
        if warnings:
            return warnings
        mirrored = [
            AudienceWarningDraft.from_notification_draft(item)
            for item in self.repository.list_v2_notification_drafts(event_id)
        ]
        for item in mirrored:
            self.repository.save_v3_audience_warning(item)
        return mirrored

    def build_stream_events(self, event_id: str, *, focus_object_id: str | None = None) -> list[TwinStreamEvent]:
        overview = self.get_twin_overview(event_id)
        council = self.get_agent_council(event_id)
        proposals = [
            item
            for item in self.platform.get_pending_regional_proposals_snapshot().items
            if item.proposal.event_id == event_id
        ]
        warnings = self.list_audience_warnings(event_id)
        focus_object = None
        if focus_object_id or overview.lead_object_id:
            resolved_focus_id = focus_object_id or overview.lead_object_id
            if resolved_focus_id:
                focus_object = self.get_focus_object(event_id, resolved_focus_id)

        return [
            self._stream_event(
                "twin_overview_updated",
                {"overview": overview.model_dump(mode="json")},
            ),
            self._stream_event(
                "focus_object_updated",
                {"focus_object": focus_object.model_dump(mode="json") if focus_object else None},
            ),
            self._stream_event(
                "agent_council_updated",
                {"council": council.model_dump(mode="json")},
            ),
            self._stream_event(
                "proposal_status_changed",
                {"proposals": [item.model_dump(mode="json") for item in proposals]},
            ),
            self._stream_event(
                "warnings_generated",
                {"warnings": [item.model_dump(mode="json") for item in warnings]},
            ),
            self._stream_event(
                "proposal_generated",
                {
                    "proposal_ids": [item.proposal.proposal_id for item in proposals],
                    "count": len(proposals),
                },
            ),
        ]

    def _stream_event(self, event_type: str, payload: dict[str, Any]) -> TwinStreamEvent:
        version = sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return TwinStreamEvent(
            event_type=event_type,
            version=version,
            created_at=datetime.now(timezone.utc),
            payload=payload,
        )

    def _focus_summary_from_impact(self, *, impact: Any, proposal_views: list[Any], index: int) -> TwinFocusObjectSummary:
        pending_ids = [
            item.proposal.proposal_id
            for item in proposal_views
            if item.proposal.status == ProposalStatus.PENDING
            and (item.proposal.entity_id == impact.entity.entity_id or impact.entity.entity_id in item.proposal.high_risk_object_ids)
        ]
        return TwinFocusObjectSummary(
            object_id=impact.entity.entity_id,
            name=impact.entity.name,
            entity_type=impact.entity.entity_type.value,
            village=impact.entity.village,
            risk_level=impact.risk_level,
            time_to_impact_minutes=impact.time_to_impact_minutes,
            summary=impact.risk_reason[0] if impact.risk_reason else f"{impact.entity.name} 需要持续关注。",
            recommended_action=impact.resource_gap[0] if impact.resource_gap else "保持监测并准备行动 proposal。",
            pending_proposal_ids=pending_ids,
            canvas_position=self._canvas_position(index),
        )

    def _build_map_layers(
        self,
        focus_objects: list[TwinFocusObjectSummary],
        proposal_views: list[Any],
        warnings: list[AudienceWarningDraft],
    ) -> list[TwinObjectMapLayer]:
        layers: list[TwinObjectMapLayer] = []
        warning_proposal_ids = {item.proposal_id for item in warnings}
        for index, item in enumerate(focus_objects):
            column = index % 3
            row = index // 3
            linked_proposals = [
                proposal
                for proposal in proposal_views
                if proposal.proposal.entity_id == item.object_id or item.object_id in proposal.proposal.high_risk_object_ids
            ]
            has_pending = any(proposal.proposal.status == ProposalStatus.PENDING for proposal in linked_proposals)
            has_approved = any(proposal.proposal.status == ProposalStatus.APPROVED for proposal in linked_proposals)
            has_warning = any(proposal.proposal.proposal_id in warning_proposal_ids for proposal in linked_proposals)
            proposal_state = "warning_generated" if has_warning else "approved" if has_approved else "pending" if has_pending else "monitoring"
            layers.append(
                TwinObjectMapLayer(
                    object_id=item.object_id,
                    name=item.name,
                    risk_level=item.risk_level,
                    entity_type=item.entity_type,
                    east_offset_m=-340 + column * 290,
                    north_offset_m=260 - row * 220,
                    height_offset_m=0.0,
                    proposal_state=proposal_state,
                    is_lead=index == 0,
                )
            )
        return layers

    @staticmethod
    def _canvas_position(index: int) -> dict[str, float]:
        presets = [
            {"left": 18.0, "top": 24.0},
            {"left": 62.0, "top": 28.0},
            {"left": 34.0, "top": 58.0},
            {"left": 74.0, "top": 62.0},
            {"left": 48.0, "top": 16.0},
            {"left": 22.0, "top": 72.0},
        ]
        return presets[index % len(presets)]

    @staticmethod
    def _fallback_actions_for_impact(impact: Any) -> list[str]:
        actions = []
        if impact.blocked_routes:
            actions.append("优先避开当前受阻路段并重规划转移路线。")
        if impact.resource_gap:
            actions.append(impact.resource_gap[0])
        actions.append("保持人工审批与现场联动。")
        return actions[:3]
