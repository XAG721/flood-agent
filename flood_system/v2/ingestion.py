from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from ..models import RiskLevel
from .models import EventCreateRequest, EventRecord, EventStatus, EventStreamRecord, EventType, ObservationIngestItem


class IngestionService:
    def __init__(self, repository) -> None:
        self.repository = repository

    def create_event(self, request: EventCreateRequest) -> EventRecord:
        now = datetime.now(timezone.utc)
        event = EventRecord(
            event_id=f"event_{uuid4().hex[:10]}",
            area_id=request.area_id,
            title=request.title,
            trigger_reason=request.trigger_reason,
            current_stage=request.stage,
            current_risk_level=RiskLevel.NONE,
            status=EventStatus.ACTIVE,
            metadata=request.metadata,
            created_at=now,
            updated_at=now,
        )
        self.repository.save_v2_event(event)
        self.repository.add_v2_stream_record(
            EventStreamRecord(
                event_id=event.event_id,
                event_type=EventType.PLAN_PROPOSED,
                payload={"operator": request.operator, "title": request.title},
                created_at=now,
            )
        )
        return event

    def add_observations(self, event_id: str, observations: list[ObservationIngestItem], operator: str) -> None:
        self.repository.add_v2_observations(event_id, observations)
        self.repository.add_v2_stream_record(
            EventStreamRecord(
                event_id=event_id,
                event_type=EventType.OBSERVATION_INGESTED,
                payload={"operator": operator, "count": len(observations)},
                created_at=datetime.now(timezone.utc),
            )
        )
