from __future__ import annotations

from datetime import datetime, timezone
from collections import defaultdict
from uuid import uuid4

from .models import (
    CompletionStatus,
    ExperienceRecord,
    MemoryEventRecord,
    MemoryEventType,
    MemorySnapshot,
    RiskLevel,
    StrategyPattern,
)


class SessionMemoryStore:
    def __init__(self, repository) -> None:
        self.repository = repository

    def load_snapshot(self, session_id: str, *, area_id: str | None = None) -> MemorySnapshot:
        snapshot = self.repository.get_v2_copilot_memory_state(session_id)
        if snapshot is not None:
            return snapshot
        snapshot = MemorySnapshot(
            session_id=session_id,
            focus_area_id=area_id,
            updated_at=datetime.now(timezone.utc),
        )
        self.save_snapshot(snapshot)
        return snapshot

    def save_snapshot(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        normalized = snapshot.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        self.repository.save_v2_copilot_memory_state(normalized)
        return normalized

    def append_event(
        self,
        session_id: str,
        event_type: MemoryEventType,
        payload: dict,
    ) -> MemoryEventRecord:
        record = MemoryEventRecord(
            memory_event_id=f"mem_{uuid4().hex[:10]}",
            session_id=session_id,
            event_type=event_type,
            payload=payload,
            created_at=datetime.now(timezone.utc),
        )
        self.repository.add_v2_copilot_memory_event(record)
        return record

    def note_user_question(self, session_id: str, question: str) -> None:
        self.append_event(
            session_id,
            MemoryEventType.USER_QUESTION,
            {"question": question},
        )

    def apply_answer(
        self,
        session_id: str,
        *,
        area_id: str,
        target_entity_id: str | None,
        target_entity_name: str | None,
        intent: str,
        unresolved_slots: list[str],
        completion_status: CompletionStatus,
        carried_context_notes: list[str],
        pending_proposal_id: str | None,
    ) -> MemorySnapshot:
        snapshot = self.load_snapshot(session_id, area_id=area_id)
        pending = list(snapshot.pending_proposal_ids)
        if pending_proposal_id and pending_proposal_id not in pending:
            pending.append(pending_proposal_id)

        updated = snapshot.model_copy(
            update={
                "focus_entity_id": target_entity_id or snapshot.focus_entity_id,
                "focus_entity_name": target_entity_name or snapshot.focus_entity_name,
                "focus_area_id": area_id,
                "current_goal": intent,
                "pending_proposal_ids": pending,
                "unresolved_slots": list(dict.fromkeys(unresolved_slots)),
                "last_completion_status": completion_status,
            }
        )
        updated = self.save_snapshot(updated)
        self.append_event(
            session_id,
            MemoryEventType.PLANNER_SELECTED_FOCUS,
            {
                "focus_entity_id": updated.focus_entity_id,
                "focus_entity_name": updated.focus_entity_name,
                "current_goal": updated.current_goal,
                "carried_context_notes": carried_context_notes,
            },
        )
        if updated.unresolved_slots:
            self.append_event(
                session_id,
                MemoryEventType.REVIEWER_UNRESOLVED_SLOT,
                {"unresolved_slots": updated.unresolved_slots},
            )
        return updated

    def apply_proposal_resolution(
        self,
        session_id: str,
        proposal_id: str,
        *,
        approved: bool,
    ) -> MemorySnapshot:
        snapshot = self.load_snapshot(session_id)
        pending = [item for item in snapshot.pending_proposal_ids if item != proposal_id]
        executed = list(snapshot.executed_proposal_ids)
        if approved and proposal_id not in executed:
            executed.append(proposal_id)
        updated = self.save_snapshot(
            snapshot.model_copy(
                update={
                    "pending_proposal_ids": pending,
                    "executed_proposal_ids": executed,
                }
            )
        )
        self.append_event(
            session_id,
            MemoryEventType.PROPOSAL_APPROVED if approved else MemoryEventType.PROPOSAL_REJECTED,
            {"proposal_id": proposal_id},
        )
        return updated


class OperationalExperienceStore:
    def __init__(self, repository) -> None:
        self.repository = repository

    def record_outcome(
        self,
        *,
        event_id: str,
        action_type: str,
        action_summary: str,
        outcome: str,
        entity_id: str | None = None,
        entity_type: str | None = None,
        risk_level: RiskLevel | None = None,
        confidence: float = 0.0,
        tags: list[str] | None = None,
        payload: dict | None = None,
    ) -> ExperienceRecord:
        record = ExperienceRecord(
            experience_id=f"exp_{uuid4().hex[:12]}",
            event_id=event_id,
            entity_id=entity_id,
            entity_type=entity_type,
            risk_level=risk_level,
            action_type=action_type,
            action_summary=action_summary,
            outcome=outcome,
            confidence=round(confidence, 2),
            tags=tags or [],
            payload=payload or {},
            created_at=datetime.now(timezone.utc),
        )
        self.repository.save_v2_experience_record(record)
        return record

    def query_similar_cases(
        self,
        *,
        event_id: str | None = None,
        entity_id: str | None = None,
        entity_type: str | None = None,
        risk_level: RiskLevel | None = None,
        limit: int = 8,
    ) -> list[ExperienceRecord]:
        records = self.repository.list_v2_experience_records(
            event_id=event_id,
            entity_id=entity_id,
            entity_type=entity_type,
            limit=max(limit * 3, limit),
        )
        if risk_level is not None:
            filtered = [record for record in records if record.risk_level == risk_level]
            if filtered:
                records = filtered
        return records[:limit]

    def rank_strategy_patterns(
        self,
        *,
        entity_type: str | None = None,
        risk_level: RiskLevel | None = None,
        limit: int = 5,
    ) -> list[StrategyPattern]:
        records = self.repository.list_v2_experience_records(
            entity_type=entity_type,
            limit=200,
        )
        if risk_level is not None:
            records = [record for record in records if record.risk_level == risk_level]
        grouped: dict[tuple[str | None, RiskLevel | None, str], list[ExperienceRecord]] = defaultdict(list)
        for record in records:
            grouped[(record.entity_type, record.risk_level, record.action_type)].append(record)

        patterns: list[StrategyPattern] = []
        for index, ((group_entity_type, group_risk_level, action_type), items) in enumerate(grouped.items(), start=1):
            sample_size = len(items)
            approvals = sum(1 for item in items if item.outcome in {"approved", "executed", "success"})
            executions = sum(1 for item in items if item.outcome in {"executed", "success"})
            failures = [item.action_summary for item in items if item.outcome in {"rejected", "failed", "execution_failed"}]
            patterns.append(
                StrategyPattern(
                    pattern_id=f"pattern_{index}",
                    entity_type=group_entity_type,
                    risk_level=group_risk_level,
                    action_type=action_type,
                    sample_size=sample_size,
                    approval_rate=round(approvals / sample_size, 2) if sample_size else 0.0,
                    execution_success_rate=round(executions / sample_size, 2) if sample_size else 0.0,
                    recommended_summary=items[0].action_summary if items else "",
                    common_failures=failures[:3],
                    supporting_experience_ids=[item.experience_id for item in items[:5]],
                )
            )
        patterns.sort(key=lambda item: (item.sample_size, item.approval_rate, item.execution_success_rate), reverse=True)
        return patterns[:limit]
