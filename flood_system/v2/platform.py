from __future__ import annotations

from ..rag_runtime import RAGService
from ..sample_data import build_area_profiles, build_resource_status
from .copilot_orchestrator import CopilotOrchestrator
from .decision_engine import DecisionEngine, PolicyGuard
from .exposure_engine import ExposureEngine
from .hazard_engine import HazardEngine
from .ingestion import IngestionService
from .llm_gateway import ResponsesLLMGateway
from .memory_store import OperationalExperienceStore, SessionMemoryStore
from .multi_agent import AgentSupervisor
from .notification_gateway import NotificationGateway
from .platform_agent_ops import PlatformAgentOpsMixin
from .platform_audit import AuditOperationsMixin
from .platform_evaluation_ops import PlatformEvaluationOpsMixin
from .platform_event_ops import PlatformEventOpsMixin
from .platform_governance_ops import PlatformGovernanceOpsMixin
from .platform_impact_ops import PlatformImpactOpsMixin
from .platform_regional_ops import PlatformRegionalOpsMixin
from .platform_runtime_admin import RuntimeAdminMixin
from .regional_proposals import RegionalProposalManager
from .reporting import LongTermMemoryStore
from .routing import RoutePlanningService
from .tools import build_v2_tools


class ProductionPlatform(
    RuntimeAdminMixin,
    PlatformEventOpsMixin,
    PlatformImpactOpsMixin,
    PlatformAgentOpsMixin,
    PlatformEvaluationOpsMixin,
    PlatformGovernanceOpsMixin,
    PlatformRegionalOpsMixin,
    AuditOperationsMixin,
):
    """Production V2 platform assembled from focused operation mixins."""

    def __init__(
        self,
        *,
        repository,
        rag_service: RAGService,
        area_profiles: dict,
        bootstrap_resource_status: dict,
        llm_gateway=None,
    ) -> None:
        self.repository = repository
        self.rag_service = rag_service
        self.area_profiles = area_profiles or build_area_profiles()
        self.bootstrap_resource_status = bootstrap_resource_status or build_resource_status()
        self.llm_gateway = llm_gateway or ResponsesLLMGateway()
        self.route_planner = RoutePlanningService()
        self.hazard_engine = HazardEngine()
        self.exposure_engine = ExposureEngine(self.route_planner)
        self.decision_engine = DecisionEngine(PolicyGuard(), self.llm_gateway)
        self.notification_gateway = NotificationGateway(self.llm_gateway)
        self.regional_proposals = RegionalProposalManager(self.repository, self.llm_gateway)
        self.ingestion = IngestionService(self.repository)
        self.tools = build_v2_tools(self)
        self.copilot = CopilotOrchestrator(self)
        self.agent_supervisor = AgentSupervisor(self, self.repository)
        self.session_memory_store = SessionMemoryStore(self.repository)
        self.operational_experience_store = OperationalExperienceStore(self.repository)
        self.long_term_memory_store = LongTermMemoryStore(self.repository, self.rag_service)
        self.event_postmortem_service = None
