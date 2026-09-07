from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .models import (
    ActionProposal,
    ExecutionLogEntry,
    GenerationSource,
    NotificationDraft,
)


class NotificationGateway:
    def __init__(self, llm_gateway) -> None:
        self.llm_gateway = llm_gateway

    def build_regional_execution_bundle(
        self,
        *,
        event_id: str,
        area_id: str,
        proposal: ActionProposal,
        operator_id: str,
        event_title: str,
    ) -> tuple[list[NotificationDraft], list[ExecutionLogEntry]]:
        created_at = datetime.now(timezone.utc)
        bundle = self.llm_gateway.generate_execution_bundle(
            {
                "event_id": event_id,
                "area_id": area_id,
                "event_title": event_title,
                "proposal": proposal.model_dump(mode="json"),
                "operator_id": operator_id,
            }
        )
        summary = self.llm_gateway.generate_execution_summary(
            {
                "event_id": event_id,
                "area_id": area_id,
                "event_title": event_title,
                "proposal": proposal.model_dump(mode="json"),
                "bundle": bundle.model_dump(mode="json"),
                "operator_id": operator_id,
            }
        )

        drafts = [
            NotificationDraft(
                draft_id=f"draft_{uuid4().hex[:10]}",
                event_id=event_id,
                proposal_id=proposal.proposal_id,
                area_id=area_id,
                audience=item.audience,
                channel=item.channel,
                content=item.content,
                generation_source=GenerationSource.LLM,
                model_name=self.llm_gateway.model_name,
                grounding_summary=bundle.grounding_summary,
                created_at=created_at,
            )
            for item in bundle.drafts
        ]

        logs = [
            ExecutionLogEntry(
                log_id=f"log_{uuid4().hex[:10]}",
                event_id=event_id,
                proposal_id=proposal.proposal_id,
                area_id=area_id,
                action_type=proposal.action_type or "regional_action",
                summary=summary.approval_summary,
                operator_id=operator_id,
                details={
                    "event_title": event_title,
                    "execution_mode": proposal.execution_mode.value,
                    "action_scope": proposal.action_scope,
                    "task_instructions": bundle.task_instructions,
                    "high_risk_object_ids": proposal.high_risk_object_ids,
                },
                generation_source=GenerationSource.LLM,
                model_name=self.llm_gateway.model_name,
                grounding_summary=bundle.grounding_summary,
                created_at=created_at,
            ),
            ExecutionLogEntry(
                log_id=f"log_{uuid4().hex[:10]}",
                event_id=event_id,
                proposal_id=proposal.proposal_id,
                area_id=area_id,
                action_type="audit",
                summary=summary.audit_summary,
                operator_id=operator_id,
                details={
                    "resolution_note": proposal.resolution_note,
                    "last_editor": proposal.last_editor,
                    "draft_count": len(drafts),
                    "task_summary": bundle.task_summary,
                },
                generation_source=GenerationSource.LLM,
                model_name=self.llm_gateway.model_name,
                grounding_summary=summary.audit_summary,
                created_at=created_at,
            ),
        ]
        return drafts, logs
