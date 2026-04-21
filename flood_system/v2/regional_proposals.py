from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any
from uuid import uuid4

from ..models import RiskLevel
from .models import (
    ActionProposal,
    EventRecord,
    ExecutionMode,
    ExposureSummary,
    GenerationSource,
    HazardState,
    ProposalStatus,
    SimulationUpdateRecord,
)


HIGH_RISK_LEVELS = {RiskLevel.ORANGE, RiskLevel.RED}
PACKAGE_PAYLOAD_KEY = "regional_analysis_package"


@dataclass
class RegionalProposalReconcileResult:
    risk_stage_key: str | None
    pending_proposal_ids: list[str]
    recommended_actions: list[str]
    queue_changed: bool
    should_interrupt: bool


@dataclass
class _DesiredAction:
    action_type: str
    execution_mode: ExecutionMode
    action_display_name: str
    action_display_tagline: str
    action_display_category: str
    title: str
    summary: str
    trigger_reason: str
    recommendation: str
    evidence_summary: str
    high_risk_object_ids: list[str]
    action_scope: dict
    grounding_summary: str
    chat_follow_up_prompt: str
    system_version_hash: str


class RegionalProposalManager:
    def __init__(self, repository, llm_gateway) -> None:
        self.repository = repository
        self.llm_gateway = llm_gateway

    def reconcile(
        self,
        *,
        event: EventRecord,
        hazard_state: HazardState,
        exposure_summary: ExposureSummary,
        simulation_update: SimulationUpdateRecord,
        previous_risk_level: RiskLevel | None,
        knowledge_evidence: list,
    ) -> RegionalProposalReconcileResult:
        now = datetime.now(timezone.utc)
        proposals = self.repository.list_v2_action_proposals(event.event_id, proposal_scope="regional")
        pending = [item for item in proposals if item.status == ProposalStatus.PENDING]
        queue_changed = False
        should_interrupt = False

        if hazard_state.overall_risk_level not in HIGH_RISK_LEVELS:
            for proposal in pending:
                updated = proposal.model_copy(
                    update={
                        "status": ProposalStatus.WITHDRAWN,
                        "withdrawn_reason": "overall_risk_deescalated",
                        "updated_at": now,
                    }
                )
                self.repository.save_v2_action_proposal(updated)
                queue_changed = True
            return RegionalProposalReconcileResult(
                risk_stage_key=None,
                pending_proposal_ids=[],
                recommended_actions=[],
                queue_changed=queue_changed,
                should_interrupt=False,
            )

        current_stage_key = self._resolve_risk_stage_key(
            event_id=event.event_id,
            current_level=hazard_state.overall_risk_level,
            previous_level=previous_risk_level,
            proposals=proposals,
            generated_at=simulation_update.generated_at,
        )
        desired_actions = {
            item.action_type: item
            for item in self._build_desired_actions(
                event=event,
                hazard_state=hazard_state,
                exposure_summary=exposure_summary,
                simulation_update=simulation_update,
                knowledge_evidence=knowledge_evidence,
            )
        }
        latest_by_type = self._latest_by_action_type(proposals)
        pending_by_type = self._latest_by_action_type(pending)

        for action_type, desired in desired_actions.items():
            current_pending = pending_by_type.get(action_type)
            latest = latest_by_type.get(action_type)

            if current_pending and current_pending.risk_stage_key == current_stage_key:
                if current_pending.system_version_hash == desired.system_version_hash:
                    continue
                if current_pending.edited_by_commander:
                    flagged = current_pending.model_copy(
                        update={
                            "has_new_system_suggestion": True,
                            "updated_at": now,
                        }
                    )
                    self.repository.save_v2_action_proposal(flagged)
                    queue_changed = True
                    should_interrupt = True
                    continue
                replacement = self._create_proposal(
                    event=event,
                    hazard_state=hazard_state,
                    desired=desired,
                    risk_stage_key=current_stage_key,
                    created_at=now,
                )
                superseded = current_pending.model_copy(
                    update={
                        "status": ProposalStatus.SUPERSEDED,
                        "superseded_by_proposal_id": replacement.proposal_id,
                        "updated_at": now,
                    }
                )
                self.repository.save_v2_action_proposal(superseded)
                self.repository.save_v2_action_proposal(replacement)
                queue_changed = True
                should_interrupt = True
                continue

            if latest and latest.status in {ProposalStatus.REJECTED, ProposalStatus.APPROVED}:
                same_stage = latest.risk_stage_key == current_stage_key
                same_hash = latest.system_version_hash == desired.system_version_hash
                if same_stage and same_hash:
                    continue

            if current_pending and current_pending.risk_stage_key != current_stage_key:
                replacement = self._create_proposal(
                    event=event,
                    hazard_state=hazard_state,
                    desired=desired,
                    risk_stage_key=current_stage_key,
                    created_at=now,
                )
                superseded = current_pending.model_copy(
                    update={
                        "status": ProposalStatus.SUPERSEDED,
                        "superseded_by_proposal_id": replacement.proposal_id,
                        "updated_at": now,
                    }
                )
                self.repository.save_v2_action_proposal(superseded)
                self.repository.save_v2_action_proposal(replacement)
                queue_changed = True
                should_interrupt = True
                continue

            created = self._create_proposal(
                event=event,
                hazard_state=hazard_state,
                desired=desired,
                risk_stage_key=current_stage_key,
                created_at=now,
            )
            self.repository.save_v2_action_proposal(created)
            queue_changed = True
            should_interrupt = True

        for action_type, current_pending in pending_by_type.items():
            if action_type in desired_actions:
                continue
            if current_pending.edited_by_commander:
                flagged = current_pending.model_copy(
                    update={
                        "has_new_system_suggestion": True,
                        "updated_at": now,
                    }
                )
                self.repository.save_v2_action_proposal(flagged)
            else:
                superseded = current_pending.model_copy(
                    update={
                        "status": ProposalStatus.SUPERSEDED,
                        "updated_at": now,
                        "withdrawn_reason": "action_removed_by_replan",
                    }
                )
                self.repository.save_v2_action_proposal(superseded)
            queue_changed = True
            should_interrupt = True

        refreshed_pending = self.repository.list_v2_action_proposals(
            event.event_id,
            proposal_scope="regional",
            statuses=[ProposalStatus.PENDING.value],
        )
        refreshed_pending = self._attach_package_summary(
            event=event,
            hazard_state=hazard_state,
            exposure_summary=exposure_summary,
            pending_proposals=refreshed_pending,
            knowledge_evidence=knowledge_evidence,
            risk_stage_key=current_stage_key,
        )
        return RegionalProposalReconcileResult(
            risk_stage_key=current_stage_key,
            pending_proposal_ids=[item.proposal_id for item in refreshed_pending],
            recommended_actions=[item.summary for item in refreshed_pending],
            queue_changed=queue_changed,
            should_interrupt=should_interrupt,
        )

    def _resolve_risk_stage_key(
        self,
        *,
        event_id: str,
        current_level: RiskLevel,
        previous_level: RiskLevel | None,
        proposals: list[ActionProposal],
        generated_at: datetime,
    ) -> str:
        latest_stage = next((item.risk_stage_key for item in proposals if item.risk_stage_key), None)
        latest_stage_has_pending = any(
            item.risk_stage_key == latest_stage and item.status == ProposalStatus.PENDING
            for item in proposals
        )
        entered_high_risk = previous_level not in HIGH_RISK_LEVELS
        escalated_to_red = previous_level == RiskLevel.ORANGE and current_level == RiskLevel.RED
        if latest_stage is None or entered_high_risk or escalated_to_red or not latest_stage_has_pending:
            timestamp = generated_at.strftime("%Y%m%d%H%M%S%f")
            return f"risk_stage_{event_id}_{current_level.value.lower()}_{timestamp}"
        return latest_stage

    def _build_desired_actions(
        self,
        *,
        event: EventRecord,
        hazard_state: HazardState,
        exposure_summary: ExposureSummary,
        simulation_update: SimulationUpdateRecord,
        knowledge_evidence: list,
    ) -> list[_DesiredAction]:
        decision = self.llm_gateway.generate_regional_decision(
            {
                "event": event.model_dump(mode="json"),
                "hazard_state": hazard_state.model_dump(mode="json"),
                "simulation_update": simulation_update.model_dump(mode="json"),
                "exposure_summary": exposure_summary.model_dump(mode="json"),
                "knowledge_evidence": [item.model_dump(mode="json") for item in knowledge_evidence[:6]],
            }
        )
        desired: list[_DesiredAction] = []
        for action in decision.actions:
            draft = self.llm_gateway.generate_proposal_draft(
                {
                    "event": event.model_dump(mode="json"),
                    "hazard_state": hazard_state.model_dump(mode="json"),
                    "simulation_update": simulation_update.model_dump(mode="json"),
                    "exposure_summary": exposure_summary.model_dump(mode="json"),
                    "knowledge_evidence": [item.model_dump(mode="json") for item in knowledge_evidence[:6]],
                    "action": action.model_dump(mode="json"),
                    "regional_decision_summary": decision.summary,
                }
            )
            desired.append(
                _DesiredAction(
                    action_type=action.action_type,
                    execution_mode=action.execution_mode,
                    action_display_name=draft.action_display_name,
                    action_display_tagline=draft.action_display_tagline,
                    action_display_category=draft.action_display_category,
                    title=draft.title,
                    summary=draft.summary,
                    trigger_reason=draft.trigger_reason,
                    recommendation=draft.recommendation,
                    evidence_summary=draft.evidence_summary,
                    high_risk_object_ids=draft.high_risk_object_ids,
                    action_scope=draft.action_scope,
                    grounding_summary=draft.grounding_summary or decision.grounding_summary,
                    chat_follow_up_prompt=draft.chat_follow_up_prompt,
                    system_version_hash=self._build_version_hash(
                        action.action_type,
                        draft.high_risk_object_ids,
                        draft.action_scope,
                    ),
                )
            )
        return desired

    def _attach_package_summary(
        self,
        *,
        event: EventRecord,
        hazard_state: HazardState,
        exposure_summary: ExposureSummary,
        pending_proposals: list[ActionProposal],
        knowledge_evidence: list,
        risk_stage_key: str,
    ) -> list[ActionProposal]:
        stage_pending = [item for item in pending_proposals if item.risk_stage_key == risk_stage_key]
        if not stage_pending:
            return pending_proposals

        summary_payload = self._build_package_summary_payload(
            event=event,
            hazard_state=hazard_state,
            exposure_summary=exposure_summary,
            pending_proposals=stage_pending,
            knowledge_evidence=knowledge_evidence,
            risk_stage_key=risk_stage_key,
        )

        updated_pending: list[ActionProposal] = []
        for proposal in pending_proposals:
            if proposal.risk_stage_key != risk_stage_key:
                updated_pending.append(proposal)
                continue
            current_payload = dict(proposal.payload)
            if current_payload.get(PACKAGE_PAYLOAD_KEY) == summary_payload:
                updated_pending.append(proposal)
                continue
            updated = proposal.model_copy(
                update={
                    "payload": {
                        **current_payload,
                        PACKAGE_PAYLOAD_KEY: summary_payload,
                    },
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self.repository.save_v2_action_proposal(updated)
            updated_pending.append(updated)
        return updated_pending

    def _build_package_summary_payload(
        self,
        *,
        event: EventRecord,
        hazard_state: HazardState,
        exposure_summary: ExposureSummary,
        pending_proposals: list[ActionProposal],
        knowledge_evidence: list,
        risk_stage_key: str,
    ) -> dict[str, Any]:
        focus_object_ids = self._collect_focus_object_ids(pending_proposals)
        generated_at = datetime.now(timezone.utc).isoformat()
        try:
            package_output = self.llm_gateway.generate_regional_analysis_package(
                {
                    "event": event.model_dump(mode="json"),
                    "hazard_state": hazard_state.model_dump(mode="json"),
                    "exposure_summary": exposure_summary.model_dump(mode="json"),
                    "pending_proposals": [item.model_dump(mode="json") for item in pending_proposals],
                    "recommended_actions": [item.recommendation or item.summary for item in pending_proposals],
                    "knowledge_evidence": [item.model_dump(mode="json") for item in knowledge_evidence[:6]],
                }
            )
            analysis_message = package_output.analysis_message
            risk_assessment = package_output.risk_assessment
            rescue_plan = package_output.rescue_plan
            resource_dispatch_plan = package_output.resource_dispatch_plan
        except Exception:
            analysis_message = (
                f"{event.title} has a new regional analysis package with {len(pending_proposals)} coordinated actions."
            )
            risk_assessment = self._fallback_risk_assessment(hazard_state, exposure_summary)
            rescue_plan = self._fallback_rescue_plan(pending_proposals, focus_object_ids)
            resource_dispatch_plan = self._fallback_resource_plan(focus_object_ids)

        return {
            "package_id": risk_stage_key,
            "event_id": event.event_id,
            "current_risk_level": hazard_state.overall_risk_level.value,
            "trigger_type": "simulation_updated",
            "focus_object_ids": focus_object_ids,
            "analysis_message": analysis_message,
            "risk_assessment": risk_assessment,
            "rescue_plan": rescue_plan,
            "resource_dispatch_plan": resource_dispatch_plan,
            "generated_at": generated_at,
        }

    @staticmethod
    def _collect_focus_object_ids(proposals: list[ActionProposal]) -> list[str]:
        focus_object_ids: list[str] = []
        seen: set[str] = set()
        for proposal in proposals:
            for entity_id in proposal.high_risk_object_ids:
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                focus_object_ids.append(entity_id)
        return focus_object_ids

    @staticmethod
    def _fallback_risk_assessment(hazard_state: HazardState, exposure_summary: ExposureSummary) -> str:
        lead_risk = exposure_summary.top_risks[0] if exposure_summary.top_risks else "Flood impacts remain elevated."
        return (
            f"Current district risk level is {hazard_state.overall_risk_level.value}. "
            f"Primary assessment: {lead_risk}"
        )

    @staticmethod
    def _fallback_rescue_plan(pending_proposals: list[ActionProposal], focus_object_ids: list[str]) -> str:
        focus_count = len(focus_object_ids)
        action_titles = ", ".join(item.title for item in pending_proposals[:2])
        if not action_titles:
            action_titles = "regional notification and rescue coordination"
        return f"Coordinate rescue preparation for {focus_count or 'all identified'} priority targets and advance {action_titles}."

    @staticmethod
    def _fallback_resource_plan(focus_object_ids: list[str]) -> str:
        if focus_object_ids:
            return (
                f"Pre-position pumps, transport support, and warning resources around {len(focus_object_ids)} focus objects."
            )
        return "Pre-position pumps, transport support, and public-warning resources in the highest-risk corridors."

    @staticmethod
    def _build_version_hash(action_type: str, high_risk_object_ids: list[str], action_scope: dict) -> str:
        payload = {
            "action_type": action_type,
            "high_risk_object_ids": high_risk_object_ids,
            "action_scope": action_scope,
        }
        return sha1(str(payload).encode("utf-8")).hexdigest()[:16]

    def _create_proposal(
        self,
        *,
        event: EventRecord,
        hazard_state: HazardState,
        desired: _DesiredAction,
        risk_stage_key: str,
        created_at: datetime,
    ) -> ActionProposal:
        return ActionProposal(
            proposal_id=f"regional_{uuid4().hex[:12]}",
            event_id=event.event_id,
            area_id=event.area_id,
            proposal_scope="regional",
            action_type=desired.action_type,
            execution_mode=desired.execution_mode,
            action_display_name=desired.action_display_name,
            action_display_tagline=desired.action_display_tagline,
            action_display_category=desired.action_display_category,
            title=desired.title,
            summary=desired.summary,
            trigger_reason=desired.trigger_reason,
            recommendation=desired.recommendation,
            evidence_summary=desired.evidence_summary,
            severity=hazard_state.overall_risk_level.value,
            requires_confirmation=True,
            required_operator_roles=["commander"],
            payload={
                "action_scope": desired.action_scope,
                "evidence_summary": desired.evidence_summary,
            },
            high_risk_object_ids=desired.high_risk_object_ids,
            action_scope=desired.action_scope,
            risk_stage_key=risk_stage_key,
            system_version_hash=desired.system_version_hash,
            generation_source=GenerationSource.LLM,
            model_name=self.llm_gateway.model_name,
            prompt_profile="proposal_draft",
            grounding_summary=desired.grounding_summary,
            chat_follow_up_prompt=desired.chat_follow_up_prompt,
            updated_at=created_at,
            created_at=created_at,
        )

    @staticmethod
    def _latest_by_action_type(proposals: list[ActionProposal]) -> dict[str, ActionProposal]:
        latest: dict[str, ActionProposal] = {}
        for proposal in proposals:
            if not proposal.action_type or proposal.action_type in latest:
                continue
            latest[proposal.action_type] = proposal
        return latest
