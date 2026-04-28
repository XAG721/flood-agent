from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from ..models import RiskLevel
from .models import (
    EventCreateRequest,
    EventRecord,
    EventSnapshot,
    HazardState,
    ObservationBatchRequest,
    ProposalStatus,
    SimulationUpdateRecord,
    SimulationUpdateRequest,
    TriggerEventType,
)


class PlatformEventOpsMixin:
    def create_event(self, request: EventCreateRequest):
        return self.ingestion.create_event(request)

    def ingest_observations(self, event_id: str, request: ObservationBatchRequest) -> EventSnapshot:
        self.ingestion.add_observations(event_id, request.observations, request.operator)
        previous_hazard = self.repository.get_v2_hazard_state(event_id)
        previous_risk_level = previous_hazard.overall_risk_level if previous_hazard else None
        hazard_state = self._recompute_hazard(event_id)
        exposure = self.get_exposure_summary(event_id)
        event = self.get_event(event_id).model_copy(
            update={
                "current_risk_level": hazard_state.overall_risk_level,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.repository.save_v2_event(event)
        self.repository.add_v2_stream_record_for_payload(
            event_id,
            "impact_recomputed",
            {
                "high_risk_entities": [item.entity.entity_id for item in exposure.affected_entities[:5]],
                "overall_risk_level": hazard_state.overall_risk_level.value,
            },
        )
        self.publish_trigger(
            event_id,
            trigger_type=TriggerEventType.OBSERVATION_INGESTED,
            payload={"operator": request.operator},
        )
        self._sync_high_risk_transition(
            event=event,
            previous_risk_level=previous_risk_level,
            current_risk_level=hazard_state.overall_risk_level,
            trigger_source=TriggerEventType.OBSERVATION_INGESTED.value,
            observed_at=hazard_state.generated_at,
        )
        return self.get_event_snapshot(event_id)

    def ingest_simulation_update(self, event_id: str, request: SimulationUpdateRequest) -> dict:
        event = self.get_event(event_id)
        previous = self.repository.get_latest_v2_simulation_update(event_id)
        previous_hazard = self.repository.get_v2_hazard_state(event_id)
        hazard_state, simulation_record = self._build_simulation_hazard_state(event, request)
        self.repository.save_v2_simulation_update(simulation_record)
        self.repository.save_v2_hazard_state(hazard_state)
        updated_event = event.model_copy(
            update={
                "current_risk_level": hazard_state.overall_risk_level,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.repository.save_v2_event(updated_event)
        self.repository.add_v2_stream_record_for_payload(
            event_id,
            "hazard_updated",
            {
                "source": "simulation_update",
                "overall_score": hazard_state.overall_score,
                "overall_risk_level": hazard_state.overall_risk_level.value,
                "simulation_update_id": simulation_record.simulation_update_id,
            },
        )
        trigger = self.publish_trigger(
            event_id,
            trigger_type=TriggerEventType.SIMULATION_UPDATED,
            payload={
                "simulation_update_id": simulation_record.simulation_update_id,
                "previous_risk_level": previous.overall_risk_level.value if previous else None,
                "current_risk_level": simulation_record.overall_risk_level.value,
            },
            dedupe=False,
        )
        run = self.agent_supervisor.process_trigger(trigger)
        self._sync_high_risk_transition(
            event=updated_event,
            previous_risk_level=previous_hazard.overall_risk_level if previous_hazard else None,
            current_risk_level=hazard_state.overall_risk_level,
            trigger_source=TriggerEventType.SIMULATION_UPDATED.value,
            observed_at=hazard_state.generated_at,
        )
        latest_pending = self.list_regional_proposals(event_id, statuses=[ProposalStatus.PENDING.value])
        latest_stage_key = latest_pending[0].proposal.risk_stage_key if latest_pending else None
        latest_planning_result = next(
            (item for item in self.list_agent_results(event_id) if item.agent_name.value == "planning_agent"),
            None,
        )
        llm_status = "ok"
        llm_error = None
        if latest_planning_result is not None:
            llm_status = str(latest_planning_result.structured_output.get("llm_status") or "ok")
            llm_error = latest_planning_result.structured_output.get("llm_error")
        return {
            "event_id": event_id,
            "overall_risk_level": hazard_state.overall_risk_level,
            "risk_stage_key": latest_stage_key,
            "trigger_id": trigger.trigger_id,
            "supervisor_run_id": run.supervisor_run_id,
            "queue_version": self.get_pending_regional_proposals_snapshot().queue_version,
            "llm_status": llm_status,
            "llm_error": llm_error,
        }

    def get_event(self, event_id: str):
        event = self.repository.get_v2_event(event_id)
        if event is None:
            raise ValueError(f"v2 event {event_id} does not exist.")
        return event

    def get_event_snapshot(self, event_id: str) -> EventSnapshot:
        return EventSnapshot(
            event=self.get_event(event_id),
            latest_hazard_state=self.repository.get_v2_hazard_state(event_id),
            latest_exposure_summary=self.get_exposure_summary(event_id),
            recent_stream=self.repository.list_v2_stream_records(event_id, limit=12),
        )

    def get_hazard_state(self, event_id: str) -> HazardState:
        hazard_state = self.repository.get_v2_hazard_state(event_id)
        if hazard_state is None:
            hazard_state = self._recompute_hazard(event_id)
        return hazard_state

    def _build_simulation_hazard_state(
        self,
        event: EventRecord,
        request: SimulationUpdateRequest,
    ) -> tuple[HazardState, SimulationUpdateRecord]:
        depth_threshold = max(request.depth_threshold_m, 0.01)
        flow_threshold = max(request.flow_threshold_mps, 0.01)
        cells = request.cells
        exceed_count = 0
        max_depth = 0.0
        max_flow = 0.0
        total_score = 0.0
        tiles = []
        for index, cell in enumerate(cells, start=1):
            max_depth = max(max_depth, cell.water_depth_m)
            max_flow = max(max_flow, cell.flow_velocity_mps)
            depth_ratio = cell.water_depth_m / depth_threshold
            flow_ratio = cell.flow_velocity_mps / flow_threshold
            combined = max(depth_ratio, flow_ratio)
            if combined >= 1.0:
                exceed_count += 1
            total_score += min(combined / 2.0, 1.0)
            tile_level = self._risk_level_from_score(min(combined / 2.0, 1.0))
            tiles.append(
                {
                    "tile_id": cell.cell_id,
                    "area_name": cell.label or f"Grid {index}",
                    "horizon_minutes": 60,
                    "risk_level": tile_level,
                    "risk_score": round(min(combined / 2.0, 1.0), 3),
                    "predicted_water_depth_cm": round(cell.water_depth_m * 100, 1),
                    "trend": "rising" if combined >= 1.0 else "stable",
                    "uncertainty": 0.12,
                    "affected_roads": [],
                }
            )
        exceed_share = exceed_count / len(cells) if cells else 0.0
        mean_score = total_score / len(cells) if cells else 0.0
        overall_score = round(min((mean_score * 0.65) + (exceed_share * 0.35), 1.0), 3)
        overall_risk_level = self._risk_level_from_score(overall_score)
        hazard_state = HazardState(
            event_id=event.event_id,
            area_id=event.area_id,
            generated_at=request.generated_at,
            overall_risk_level=overall_risk_level,
            overall_score=overall_score,
            trend="rising" if overall_risk_level in {RiskLevel.ORANGE, RiskLevel.RED} else "stable",
            uncertainty=0.12,
            freshness_seconds=0,
            hazard_tiles=tiles,
            road_reachability=[],
            monitoring_points=[],
        )
        simulation_update_id = request.simulation_update_id or f"sim_{uuid4().hex[:12]}"
        record = SimulationUpdateRecord(
            simulation_update_id=simulation_update_id,
            event_id=event.event_id,
            area_id=event.area_id,
            generated_at=request.generated_at,
            depth_threshold_m=depth_threshold,
            flow_threshold_mps=flow_threshold,
            overall_risk_level=overall_risk_level,
            overall_score=overall_score,
            exceeded_cell_count=exceed_count,
            payload=request.model_dump(mode="json"),
        )
        return hazard_state, record

    @staticmethod
    def _risk_level_from_score(score: float) -> RiskLevel:
        if score >= 0.78:
            return RiskLevel.RED
        if score >= 0.52:
            return RiskLevel.ORANGE
        if score >= 0.26:
            return RiskLevel.YELLOW
        if score > 0:
            return RiskLevel.BLUE
        return RiskLevel.NONE

    def _recompute_hazard(self, event_id: str) -> HazardState:
        event = self.get_event(event_id)
        observations = self.repository.list_v2_observations(event_id)
        hazard_state = self.hazard_engine.compute(event_id, self.area_profiles[event.area_id], observations)
        self.repository.save_v2_hazard_state(hazard_state)
        self.repository.add_v2_stream_record_for_payload(
            event_id,
            "hazard_updated",
            {
                "overall_score": hazard_state.overall_score,
                "overall_risk_level": hazard_state.overall_risk_level.value,
            },
        )
        return hazard_state
