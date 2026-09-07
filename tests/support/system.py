from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from fastapi.testclient import TestClient

from flood_system import api as api_module
from flood_system.system import FloodWarningSystem
from flood_system.compat.legacy_platform.llm_gateway import (
    LLMGenerationError,
    MockLLMGateway,
)
from flood_system.compat.legacy_platform.models import (
    EventCreateRequest,
    LLMErrorCode,
    ObservationBatchRequest,
    ObservationIngestItem,
    SimulationCell,
    SimulationUpdateRequest,
)


def build_system(tmp_path: Path) -> FloodWarningSystem:
    return FloodWarningSystem(tmp_path / "system.db", llm_gateway=MockLLMGateway())


class AlwaysFailLLMGateway:
    model_name = "failing-llm"

    @staticmethod
    def _fail(stage: str):
        raise LLMGenerationError(
            LLMErrorCode.UNAVAILABLE, f"{stage} unavailable in test gateway"
        )

    def generate_object_advisory(self, payload):
        self._fail("object_advisory")

    def generate_copilot_chat(self, payload):
        self._fail("copilot_chat")

    def generate_regional_decision(self, payload):
        self._fail("regional_decision")

    def generate_proposal_draft(self, payload):
        self._fail("proposal_draft")

    def generate_regional_analysis_package(self, payload):
        self._fail("regional_analysis_package")

    def generate_execution_bundle(self, payload):
        self._fail("execution_bundle")

    def generate_execution_summary(self, payload):
        self._fail("execution_summary")

    def generate_daily_operations_summary(self, payload):
        self._fail("daily_operations_summary")

    def generate_high_risk_postmortem_summary(self, payload):
        self._fail("high_risk_postmortem")


def sample_observations(now: datetime | None = None) -> list[ObservationIngestItem]:
    now = now or datetime.now(timezone.utc)
    return [
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=22),
            source_type="monitoring_point",
            source_name="West flood point",
            village="Lianshi Village",
            rainfall_mm=26,
            water_level_m=3.7,
            citizen_reports=2,
        ),
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=12),
            source_type="water_level_sensor",
            source_name="School gate sensor",
            village="Wuyuanli Village",
            rainfall_mm=34,
            water_level_m=4.2,
            road_blocked=True,
            citizen_reports=4,
            notes="School access road is starting to pond.",
        ),
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=6),
            source_type="camera_alert",
            source_name="Factory loading bay camera",
            village="Wuyuanli Village",
            rainfall_mm=31,
            water_level_m=4.5,
            road_blocked=True,
            citizen_reports=5,
            notes="Loading bay runoff is increasing.",
        ),
    ]


def sample_simulation_update(now: datetime | None = None) -> SimulationUpdateRequest:
    now = now or datetime.now(timezone.utc)
    return SimulationUpdateRequest(
        generated_at=now,
        depth_threshold_m=0.45,
        flow_threshold_mps=1.2,
        cells=[
            SimulationCell(
                cell_id="grid_01",
                label="School cluster",
                water_depth_m=1.1,
                flow_velocity_mps=1.5,
            ),
            SimulationCell(
                cell_id="grid_02",
                label="Factory edge",
                water_depth_m=0.9,
                flow_velocity_mps=1.4,
            ),
            SimulationCell(
                cell_id="grid_03",
                label="Residential lowland",
                water_depth_m=1.3,
                flow_velocity_mps=1.7,
            ),
            SimulationCell(
                cell_id="grid_04",
                label="Transit corridor",
                water_depth_m=0.7,
                flow_velocity_mps=1.3,
            ),
        ],
    )


def wait_for_agent_processing(
    system: FloodWarningSystem, event_id: str, timeout: float = 2.5
) -> None:
    system.supervisor_loop.trigger_poll_seconds = 0.02
    system.start_background_services()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            triggers = system.production_platform.list_trigger_events(event_id)
            tasks = system.production_platform.list_agent_tasks(event_id)
            if tasks and all(
                item.status.value in {"completed", "failed"} for item in tasks[:2]
            ):
                return
            if triggers and all(
                item.status.value in {"processed", "failed"} for item in triggers
            ):
                return
            time.sleep(0.05)
        raise AssertionError("agent processing did not complete in time.")
    finally:
        system.stop_background_services()


def seed_event(system: FloodWarningSystem) -> str:
    production = system.production_platform
    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="Beilin severe rainfall event",
            trigger_reason="test_seed",
            operator="pytest",
        )
    )
    production.ingest_observations(
        event.event_id,
        ObservationBatchRequest(operator="pytest", observations=sample_observations()),
    )
    wait_for_agent_processing(system, event.event_id)
    return event.event_id


@contextmanager
def bound_test_client(system: FloodWarningSystem):
    with TestClient(api_module.create_app(system_override=system)) as client:
        yield client
