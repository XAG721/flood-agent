from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from ..models import RiskLevel
from .models import (
    ActionProposal,
    Advisory,
    AdvisoryRequest,
    EntityImpactView,
    EntityType,
    EvidenceItem,
    ExecutionMode,
    GenerationSource,
    PolicyConstraint,
)


class PolicyGuard:
    def constraints_for(self, entity_type: EntityType, risk_level: RiskLevel) -> PolicyConstraint:
        requires_confirmation = risk_level in {RiskLevel.ORANGE, RiskLevel.RED}
        approval_actions = ["区域通知", "资源调度", "转移处置"]
        if entity_type == EntityType.RESIDENT and risk_level in {RiskLevel.BLUE, RiskLevel.YELLOW}:
            requires_confirmation = False
            approval_actions = ["电话确认", "社区触达"]
        if entity_type in {
            EntityType.SCHOOL,
            EntityType.FACTORY,
            EntityType.HOSPITAL,
            EntityType.NURSING_HOME,
        }:
            requires_confirmation = True
        return PolicyConstraint(
            entity_type=entity_type,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            actions_requiring_approval=approval_actions,
            operator_roles=["commander", "district_operator"],
        )


class DecisionEngine:
    def __init__(self, policy_guard: PolicyGuard, llm_gateway) -> None:
        self.policy_guard = policy_guard
        self.llm_gateway = llm_gateway

    def generate_advisory(
        self,
        *,
        request: AdvisoryRequest,
        impact: EntityImpactView,
        additional_evidence: list[EvidenceItem],
        allow_proposal: bool = False,
    ) -> Advisory:
        entity = impact.entity
        constraint = self.policy_guard.constraints_for(entity.entity_type, impact.risk_level)
        evidence = [*impact.evidence, *additional_evidence]
        llm_result = self.llm_gateway.generate_object_advisory(
            {
                "request": request.model_dump(mode="json"),
                "impact": impact.model_dump(mode="json"),
                "additional_evidence": [item.model_dump(mode="json") for item in additional_evidence],
                "policy_constraint": constraint.model_dump(mode="json"),
            }
        )

        proposal = None
        if allow_proposal and constraint.requires_confirmation:
            proposal = ActionProposal(
                proposal_id=f"v2_proposal_{uuid4().hex[:10]}",
                event_id=impact.event_id,
                entity_id=entity.entity_id,
                area_id=entity.area_id,
                proposal_scope="entity",
                action_type="entity_advisory",
                execution_mode=ExecutionMode.GENERIC_TASK,
                title=f"{entity.name} 处置建议",
                summary=llm_result.answer,
                trigger_reason=impact.risk_reason[0] if impact.risk_reason else f"{entity.name} 需要人工确认。",
                recommendation=llm_result.answer,
                evidence_summary=llm_result.grounding_summary,
                severity=impact.risk_level.value,
                requires_confirmation=True,
                required_operator_roles=constraint.operator_roles,
                payload={
                    "recommended_actions": llm_result.recommended_actions,
                    "approval_actions": constraint.actions_requiring_approval,
                },
                high_risk_object_ids=[entity.entity_id],
                action_scope={"entity_id": entity.entity_id},
                generation_source=GenerationSource.LLM,
                model_name=self.llm_gateway.model_name,
                prompt_profile="object_advisory",
                grounding_summary=llm_result.grounding_summary,
                created_at=datetime.now(timezone.utc),
            )

        return Advisory(
            advisory_id=f"advisory_{uuid4().hex[:10]}",
            event_id=impact.event_id,
            entity_id=impact.entity.entity_id,
            answer=llm_result.answer,
            impact_summary=llm_result.impact_summary,
            recommended_actions=llm_result.recommended_actions,
            route_options=impact.safe_routes[:2] or impact.blocked_routes[:1],
            evidence=evidence[:8],
            confidence=round(llm_result.confidence, 2),
            confidence_explanation=llm_result.confidence_explanation,
            requires_human_confirmation=constraint.requires_confirmation,
            missing_data=list(llm_result.missing_data),
            proposal=proposal,
            generation_source=GenerationSource.LLM,
            model_name=self.llm_gateway.model_name,
            grounding_summary=llm_result.grounding_summary,
            generated_at=datetime.now(timezone.utc),
        )

    def get_policy_constraints(self, entity_type: EntityType, risk_level: RiskLevel) -> PolicyConstraint:
        return self.policy_guard.constraints_for(entity_type, risk_level)
