from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .data_pipeline.service import BeilinDatasetService
from .rag_runtime import RAGService, RuntimeRAGDocumentProvider
from .repository import SQLiteRepository
from .sample_data import build_area_profiles, build_rag_documents, build_resource_status
from .v3.service import AgentTwinService
from .v2.bootstrap import build_entity_profiles
from .v2.multi_agent import HousekeepingService, SupervisorLoopService
from .v2.platform import ProductionPlatform
from .v2.reporting import DailySummaryService, EventPostmortemService


class FloodWarningSystem:
    """V2-only application container."""

    def __init__(self, db_path: str | Path, *, llm_gateway=None) -> None:
        self.db_path = Path(db_path)
        self.repository = SQLiteRepository(self.db_path)
        self.area_profiles = build_area_profiles()
        self.bootstrap_resource_status = build_resource_status()
        self.rag_runtime_path = self.db_path.parent / "rag_documents.runtime.json"
        self.rag_provider = RuntimeRAGDocumentProvider(self.rag_runtime_path, build_rag_documents())
        self.rag_service = RAGService(self.rag_provider)
        self._seed_runtime_data()
        self.production_platform = ProductionPlatform(
            repository=self.repository,
            rag_service=self.rag_service,
            area_profiles=self.area_profiles,
            bootstrap_resource_status=self.bootstrap_resource_status,
            llm_gateway=llm_gateway,
        )
        self.agent_twin = AgentTwinService(
            platform=self.production_platform,
            repository=self.repository,
        )
        self.dataset_service = BeilinDatasetService(
            repo_root=self.db_path.parent.parent,
            db_path=self.db_path,
            add_audit_record=self.production_platform.add_audit_record,
        )
        self.supervisor_loop = SupervisorLoopService(
            self.production_platform.agent_supervisor,
            interval_seconds=self._supervisor_loop_interval_seconds(),
        )
        self.housekeeping_service = HousekeepingService(
            repository=self.repository,
            platform=self.production_platform,
            interval_seconds=self._housekeeping_interval_seconds(),
        )
        self.daily_summary_service = DailySummaryService(
            repository=self.repository,
            platform=self.production_platform,
        )
        self.event_postmortem_service = EventPostmortemService(
            repository=self.repository,
            platform=self.production_platform,
            long_term_memory_store=self.production_platform.long_term_memory_store,
        )
        self.production_platform.event_postmortem_service = self.event_postmortem_service

    def _seed_runtime_data(self) -> None:
        if not self.repository.has_v2_entity_profiles():
            for area_profile in self.area_profiles.values():
                for entity in build_entity_profiles(area_profile).values():
                    self.repository.save_v2_entity_profile(entity)
        if not self.repository.has_area_resource_statuses():
            for resource_status in self.bootstrap_resource_status.values():
                self.repository.save_area_resource_status(resource_status)

    @staticmethod
    def _supervisor_loop_interval_seconds() -> float:
        raw = os.getenv("FLOOD_SUPERVISOR_LOOP_INTERVAL_SECONDS", "60")
        try:
            value = float(raw)
        except ValueError:
            return 60.0
        return max(1.0, value)

    @staticmethod
    def _housekeeping_interval_seconds() -> float:
        raw = os.getenv("FLOOD_HOUSEKEEPING_INTERVAL_SECONDS", "21600")
        try:
            value = float(raw)
        except ValueError:
            return 21600.0
        return max(60.0, value)

    def start_background_services(self) -> None:
        if os.getenv("FLOOD_SUPERVISOR_LOOP_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
            return
        self.supervisor_loop.start()
        self.housekeeping_service.start()
        self.daily_summary_service.start()
        self.event_postmortem_service.start()

    def stop_background_services(self) -> None:
        self.supervisor_loop.stop()
        self.housekeeping_service.stop()
        self.daily_summary_service.stop()
        self.event_postmortem_service.stop()

    def background_services_status(self) -> dict:
        return self.supervisor_loop.status()


def create_default_system(db_path: str | Path | None = None, *, llm_gateway=None) -> FloodWarningSystem:
    path = Path(db_path) if db_path else Path(tempfile.gettempdir()) / "flood_warning_system_v2.db"
    return FloodWarningSystem(path, llm_gateway=llm_gateway)
