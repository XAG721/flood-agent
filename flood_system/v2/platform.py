from __future__ import annotations

import csv
from hashlib import sha1
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..models import CorpusType, RAGDocument, ResourceStatus, RiskLevel
from ..rag_runtime import RAGService, validate_rag_document_payload
from ..sample_data import build_area_profiles, build_resource_status
from .copilot_orchestrator import CopilotOrchestrator
from .decision_engine import DecisionEngine, PolicyGuard
from .exposure_engine import ExposureEngine
from .hazard_engine import HazardEngine
from .ingestion import IngestionService
from .llm_gateway import LLMGenerationError, ResponsesLLMGateway
from .memory_store import OperationalExperienceStore, SessionMemoryStore
from .multi_agent import AgentSupervisor
from .models import (
    ActionProposal,
    Advisory,
    AdvisoryRequest,
    AgentResult,
    AgentMetricsView,
    AgentTask,
    AgentTimelineEntry,
    AlertSeverity,
    ArchiveStatusView,
    AuditRecord,
    BatchProposalResolutionRequest,
    DailyReportRecord,
    EntityImpactView,
    EntityProfile,
    EntityType,
    EventEpisodeSummaryRecord,
    EventCreateRequest,
    EventSnapshot,
    EvaluationBenchmark,
    EvaluationReport,
    BenchmarkScenarioResult,
    EvidenceItem,
    HazardState,
    ExperienceContextView,
    ObservationBatchRequest,
    ObservationIngestItem,
    PolicyConstraint,
    ProposalResolutionRequest,
    RAGDocumentImportRequest,
    ReplayRequest,
    ResourceStatusView,
    SessionMemoryView,
    SharedMemorySnapshot,
    StrategyHistoryView,
    SupervisorRunRecord,
    TriggerEvent,
    TriggerEventType,
    OperationalAlert,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
    V2CopilotSessionView,
    MemoryBundleView,
    DecisionReportView,
    CompletionStatus,
    EventRecord,
    ExecutionLogEntry,
    ExposureSummary,
    LongTermMemoryRecord,
    NotificationDraft,
    ProposalDraftUpdateRequest,
    RegionalAnalysisPackageStatus,
    RegionalAnalysisPackageView,
    RegionalProposalQueueSnapshot,
    RegionalProposalView,
    ProposalStatus,
    SimulationUpdateRecord,
    SimulationUpdateRequest,
)
from .notification_gateway import NotificationGateway
from .regional_proposals import PACKAGE_PAYLOAD_KEY, RegionalProposalManager
from .reporting import LongTermMemoryStore
from .routing import RoutePlanningService
from .tools import build_v2_tools


class ProductionPlatform:
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

    def add_audit_record(
        self,
        *,
        source_type: str,
        action: str,
        summary: str,
        details: dict | None = None,
        severity: AlertSeverity = AlertSeverity.INFO,
        event_id: str | None = None,
        session_id: str | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            audit_id=f"audit_{uuid4().hex[:12]}",
            source_type=source_type,
            action=action,
            summary=summary,
            details=details or {},
            severity=severity,
            event_id=event_id,
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
        )
        self.repository.add_audit_record(record)
        return record

    def save_operational_alert(
        self,
        *,
        source_type: str,
        severity: AlertSeverity,
        summary: str,
        details: str = "",
        event_id: str | None = None,
    ) -> OperationalAlert:
        now = datetime.now(timezone.utc)
        alert = OperationalAlert(
            alert_id=f"alert_{uuid4().hex[:12]}",
            source_type=source_type,
            severity=severity,
            summary=summary,
            details=details,
            event_id=event_id,
            first_seen_at=now,
            last_seen_at=now,
        )
        self.repository.save_operational_alert(alert)
        return alert

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

    def list_regional_proposals(
        self,
        event_id: str,
        *,
        statuses: list[str] | None = None,
    ) -> list[RegionalProposalView]:
        proposals = self.repository.list_v2_action_proposals(
            event_id,
            proposal_scope="regional",
            statuses=statuses,
        )
        event = self.get_event(event_id)
        hazard_state = self.get_hazard_state(event_id)
        return [self._regional_proposal_view(item, event, hazard_state) for item in proposals]

    def get_pending_regional_proposals_snapshot(self) -> RegionalProposalQueueSnapshot:
        pending = self.repository.list_v2_pending_regional_proposals()
        items = [self._regional_proposal_view_for_any_event(item) for item in pending]
        payload = [
            {
                "proposal_id": item.proposal.proposal_id,
                "updated_at": (item.proposal.updated_at or item.proposal.created_at).isoformat(),
                "status": item.proposal.status.value,
            }
            for item in items
        ]
        queue_version = sha1(str(payload).encode("utf-8")).hexdigest()[:16]
        return RegionalProposalQueueSnapshot(
            queue_version=queue_version,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )

    def list_regional_analysis_packages(
        self,
        event_id: str,
        *,
        include_pending: bool = True,
    ) -> list[RegionalAnalysisPackageView]:
        proposals = self.repository.list_v2_action_proposals(event_id, proposal_scope="regional")
        packages_by_stage: dict[str, list[ActionProposal]] = {}
        for proposal in proposals:
            if not proposal.risk_stage_key:
                continue
            packages_by_stage.setdefault(proposal.risk_stage_key, []).append(proposal)

        packages = [
            self._regional_analysis_package_view(stage_proposals)
            for stage_proposals in packages_by_stage.values()
        ]
        packages.sort(key=lambda item: item.created_at, reverse=True)
        if include_pending:
            return packages
        return [item for item in packages if item.status != RegionalAnalysisPackageStatus.PENDING]

    def get_pending_regional_analysis_package(self, event_id: str) -> RegionalAnalysisPackageView | None:
        return next(
            (
                item
                for item in self.list_regional_analysis_packages(event_id)
                if item.status == RegionalAnalysisPackageStatus.PENDING
            ),
            None,
        )

    def get_regional_analysis_package(self, package_id: str) -> RegionalAnalysisPackageView:
        proposals = self.repository.list_v2_action_proposals(proposal_scope="regional")
        grouped = [item for item in proposals if item.risk_stage_key == package_id]
        if not grouped:
            raise ValueError(f"Unknown regional analysis package: {package_id}")
        return self._regional_analysis_package_view(grouped)

    def update_regional_proposal_draft(
        self,
        proposal_id: str,
        request: ProposalDraftUpdateRequest,
    ) -> RegionalProposalView:
        proposal = self._get_regional_proposal_or_raise(proposal_id)
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError("Only pending regional proposals can be edited.")
        updated = proposal.model_copy(
            update={
                "action_scope": request.action_scope,
                "payload": {
                    **proposal.payload,
                    "action_scope": request.action_scope,
                },
                "edited_by_commander": True,
                "last_editor": "commander",
                "updated_at": datetime.now(timezone.utc),
                "has_new_system_suggestion": False,
            }
        )
        self.repository.save_v2_action_proposal(updated)
        self.add_audit_record(
            source_type="regional_proposal",
            action="proposal_draft_updated",
            summary=f"指挥长已更新区域请示草稿 {proposal_id}。",
            details={"proposal_id": proposal_id, "event_id": proposal.event_id},
            event_id=proposal.event_id,
        )
        return self._regional_proposal_view_for_any_event(updated)

    def approve_regional_proposal(
        self,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> RegionalProposalView:
        proposal = self._get_regional_proposal_or_raise(proposal_id)
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError("Only pending regional proposals can be approved.")
        event = self.get_event(proposal.event_id)
        drafts, logs = self.notification_gateway.build_regional_execution_bundle(
            event_id=proposal.event_id,
            area_id=event.area_id,
            proposal=proposal,
            operator_id=request.operator_id,
            event_title=event.title,
        )
        now = datetime.now(timezone.utc)
        approved = proposal.model_copy(
            update={
                "status": ProposalStatus.APPROVED,
                "resolved_at": now,
                "resolved_by": request.operator_id,
                "resolution_note": request.note,
                "updated_at": now,
                "last_editor": "commander" if proposal.edited_by_commander else proposal.last_editor,
            }
        )
        self.repository.save_v2_action_proposal(approved)
        for draft in drafts:
            self.repository.save_v2_notification_draft(draft)
        for log in logs:
            self.repository.save_v2_execution_log(log)
        self.repository.add_v2_stream_record_for_payload(
            approved.event_id,
            "approval_resolved",
            {"proposal_id": approved.proposal_id, "status": approved.status.value},
        )
        self.add_audit_record(
            source_type="regional_proposal",
            action="proposal_approved",
            summary=f"区域请示《{approved.title}》已批准。",
            details={"proposal_id": approved.proposal_id, "action_type": approved.action_type},
            event_id=approved.event_id,
        )
        return self._regional_proposal_view_for_any_event(approved)

    def reject_regional_proposal(
        self,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> RegionalProposalView:
        proposal = self._get_regional_proposal_or_raise(proposal_id)
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError("Only pending regional proposals can be rejected.")
        rejected = proposal.model_copy(
            update={
                "status": ProposalStatus.REJECTED,
                "resolved_at": datetime.now(timezone.utc),
                "resolved_by": request.operator_id,
                "resolution_note": request.note,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.repository.save_v2_action_proposal(rejected)
        self.repository.add_v2_stream_record_for_payload(
            rejected.event_id,
            "approval_resolved",
            {"proposal_id": rejected.proposal_id, "status": rejected.status.value},
        )
        self.add_audit_record(
            source_type="regional_proposal",
            action="proposal_rejected",
            summary=f"区域请示《{rejected.title}》已驳回。",
            details={"proposal_id": rejected.proposal_id, "action_type": rejected.action_type},
            event_id=rejected.event_id,
        )
        return self._regional_proposal_view_for_any_event(rejected)

    def approve_regional_analysis_package(
        self,
        package_id: str,
        request: ProposalResolutionRequest,
    ) -> RegionalAnalysisPackageView:
        pending = self._pending_package_proposals(package_id)
        if not pending:
            package = self.get_regional_analysis_package(package_id)
            if package.status != RegionalAnalysisPackageStatus.PENDING:
                raise ValueError("Only pending regional analysis packages can be approved.")
            raise ValueError("No pending proposals found for this regional analysis package.")
        for proposal in pending:
            self.approve_regional_proposal(proposal.proposal_id, request)
        return self.get_regional_analysis_package(package_id)

    def reject_regional_analysis_package(
        self,
        package_id: str,
        request: ProposalResolutionRequest,
    ) -> RegionalAnalysisPackageView:
        pending = self._pending_package_proposals(package_id)
        if not pending:
            package = self.get_regional_analysis_package(package_id)
            if package.status != RegionalAnalysisPackageStatus.PENDING:
                raise ValueError("Only pending regional analysis packages can be rejected.")
            raise ValueError("No pending proposals found for this regional analysis package.")
        for proposal in pending:
            self.reject_regional_proposal(proposal.proposal_id, request)
        return self.get_regional_analysis_package(package_id)

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

    def list_entity_profiles(
        self,
        *,
        area_id: str | None = None,
        entity_type: str | None = None,
    ) -> list[EntityProfile]:
        return self.repository.list_v2_entity_profiles(area_id=area_id, entity_type=entity_type)

    def get_entity_profile(self, entity_id: str) -> EntityProfile:
        entity = self.repository.get_v2_entity_profile(entity_id)
        if entity is None:
            raise ValueError(f"Unknown entity_id: {entity_id}")
        return entity

    def save_entity_profile(self, entity: EntityProfile) -> EntityProfile:
        if entity.area_id not in self.area_profiles:
            raise ValueError(f"Unknown area_id: {entity.area_id}")
        self.repository.save_v2_entity_profile(entity)
        self.add_audit_record(
            source_type="runtime_admin",
            action="entity_profile_saved",
            summary=f"已保存运行期对象画像 {entity.name}。",
            details={"entity_id": entity.entity_id, "area_id": entity.area_id},
        )
        return entity

    def delete_entity_profile(self, entity_id: str) -> None:
        if not self.repository.delete_v2_entity_profile(entity_id):
            raise ValueError(f"Unknown entity_id: {entity_id}")
        self.add_audit_record(
            source_type="runtime_admin",
            action="entity_profile_deleted",
            summary=f"已删除运行期对象画像 {entity_id}。",
            details={"entity_id": entity_id},
        )

    def get_entity_impact(self, entity_id: str, *, event_id: str | None = None) -> EntityImpactView:
        entity = self.get_entity_profile(entity_id)
        resolved_event_id = event_id or self.repository.get_latest_v2_event_id(entity.area_id)
        if resolved_event_id is None:
            raise ValueError(f"No active v2 event found for area {entity.area_id}.")
        hazard_state = self.get_hazard_state(resolved_event_id)
        area_profile = self.area_profiles[entity.area_id]
        resource_status = self.get_resource_status(entity.area_id, event_id=resolved_event_id)
        evidence = self.get_knowledge_evidence(
            event_id=resolved_event_id,
            area_id=entity.area_id,
            entity_id=entity.entity_id,
        )
        return self.exposure_engine.assess_entity(
            resolved_event_id,
            entity,
            area_profile,
            hazard_state,
            resource_status,
            evidence=evidence,
        )

    def get_exposure_summary(self, event_id: str, *, entity_type: str | None = None, top_k: int = 5):
        event = self.get_event(event_id)
        hazard_state = self.get_hazard_state(event_id)
        area_profile = self.area_profiles[event.area_id]
        resource_status = self.get_resource_status(event.area_id, event_id=event_id)
        profiles = self.list_entity_profiles(area_id=event.area_id, entity_type=entity_type)
        filtered = {item.entity_id: item for item in profiles}
        summary = self.exposure_engine.summarize(
            event_id,
            area_profile,
            hazard_state,
            filtered,
            resource_status,
            evidence=self.get_knowledge_evidence(event_id=event_id, area_id=event.area_id, entity_id=None),
        )
        summary.affected_entities = summary.affected_entities[:top_k]
        return summary

    def generate_advisory(self, request: AdvisoryRequest) -> Advisory:
        if request.entity_id:
            impact = self.get_entity_impact(request.entity_id, event_id=request.event_id)
        else:
            impact = self._impact_from_location_request(request)
        advisory = self.generate_advisory_for_impact(impact, request=request)
        self.repository.save_v2_advisory(advisory)
        self.repository.add_v2_stream_record_for_payload(
            advisory.event_id,
            "advisory_generated",
            {
                "entity_id": advisory.entity_id,
                "requires_human_confirmation": advisory.requires_human_confirmation,
                "confidence": advisory.confidence,
            },
        )
        return advisory

    def generate_advisory_for_impact(
        self,
        impact: EntityImpactView,
        *,
        request: AdvisoryRequest | None = None,
    ) -> Advisory:
        request = request or AdvisoryRequest(
            event_id=impact.event_id,
            entity_id=impact.entity.entity_id,
            area_id=impact.entity.area_id,
        )
        advisory = self.decision_engine.generate_advisory(
            request=request,
            impact=impact,
            additional_evidence=self.get_knowledge_evidence(
                event_id=impact.event_id,
                area_id=impact.entity.area_id,
                entity_id=impact.entity.entity_id,
            ),
            allow_proposal=False,
        )
        advisory.proposal = None
        return advisory

    def draft_action_proposal(self, event_id: str, entity_id: str):
        return None

    def get_area_resource_status(self, area_id: str) -> ResourceStatus:
        resource_status = self.repository.get_area_resource_status(area_id)
        if resource_status is None:
            fallback = self.bootstrap_resource_status.get(area_id)
            if fallback is None:
                raise ValueError(f"Unknown area_id: {area_id}")
            self.repository.save_area_resource_status(fallback)
            resource_status = fallback
        return resource_status

    def save_area_resource_status(self, resource_status: ResourceStatus) -> ResourceStatus:
        if resource_status.area_id not in self.area_profiles:
            raise ValueError(f"Unknown area_id: {resource_status.area_id}")
        self.repository.save_area_resource_status(resource_status)
        self.add_audit_record(
            source_type="runtime_admin",
            action="area_resource_updated",
            summary=f"已更新区域 {resource_status.area_id} 的默认资源。",
            details={"area_id": resource_status.area_id},
        )
        return resource_status

    def get_event_resource_status(self, event_id: str) -> ResourceStatus | None:
        return self.repository.get_event_resource_status(event_id)

    def save_event_resource_status(self, event_id: str, resource_status: ResourceStatus) -> ResourceStatus:
        event = self.get_event(event_id)
        if resource_status.area_id != event.area_id:
            raise ValueError("event resource override area_id must match the event area_id.")
        self.repository.save_event_resource_status(event_id, resource_status)
        self.repository.add_v2_stream_record_for_payload(
            event_id,
            "impact_recomputed",
            {"resource_override_updated": True, "area_id": resource_status.area_id},
        )
        self.add_audit_record(
            source_type="runtime_admin",
            action="event_resource_override_updated",
            summary=f"已更新事件 {event_id} 的资源覆盖。",
            details={"event_id": event_id, "area_id": resource_status.area_id},
            event_id=event_id,
        )
        self.publish_trigger(
            event_id,
            trigger_type=TriggerEventType.RESOURCE_OVERRIDE_UPDATED,
            payload={"area_id": resource_status.area_id},
        )
        return resource_status

    def delete_event_resource_status(self, event_id: str) -> None:
        if not self.repository.delete_event_resource_status(event_id):
            raise ValueError(f"No event resource override found for event {event_id}.")
        self.add_audit_record(
            source_type="runtime_admin",
            action="event_resource_override_deleted",
            summary=f"已清除事件 {event_id} 的资源覆盖。",
            details={"event_id": event_id},
            event_id=event_id,
        )
        self.publish_trigger(
            event_id,
            trigger_type=TriggerEventType.RESOURCE_OVERRIDE_DELETED,
            payload={"event_id": event_id},
        )

    def get_resource_status(self, area_id: str, *, event_id: str | None = None) -> ResourceStatus:
        if event_id is not None:
            override = self.repository.get_event_resource_status(event_id)
            if override is not None:
                return override
        return self.get_area_resource_status(area_id)

    def get_area_resource_status_view(self, area_id: str) -> ResourceStatusView:
        resource_status = self.get_area_resource_status(area_id)
        return ResourceStatusView(scope="area_default", area_id=area_id, resource_status=resource_status)

    def get_event_resource_status_view(self, event_id: str) -> ResourceStatusView:
        event = self.get_event(event_id)
        override = self.repository.get_event_resource_status(event_id)
        if override is not None:
            return ResourceStatusView(
                scope="event_override",
                area_id=event.area_id,
                event_id=event_id,
                resource_status=override,
            )
        return ResourceStatusView(
            scope="area_default",
            area_id=event.area_id,
            event_id=event_id,
            resource_status=self.get_area_resource_status(event.area_id),
        )

    def list_rag_documents(self) -> list[RAGDocument]:
        return self.rag_service.list_documents()

    def import_rag_documents(self, request: RAGDocumentImportRequest) -> list[RAGDocument]:
        documents = [item.to_document() for item in request.documents]
        validate_rag_document_payload(documents)
        imported = self.rag_service.import_documents(documents)
        self.add_audit_record(
            source_type="runtime_admin",
            action="rag_documents_imported",
            summary=f"已导入 {len(imported)} 份运行期知识文档。",
            details={"document_ids": [item.doc_id for item in imported]},
        )
        return imported

    def reload_rag_documents(self) -> list[RAGDocument]:
        documents = self.rag_service.reload_rag_store()
        self.add_audit_record(
            source_type="runtime_admin",
            action="rag_documents_reloaded",
            summary=f"已重载 {len(documents)} 份运行期知识文档。",
            details={"document_ids": [item.doc_id for item in documents]},
        )
        return documents

    def get_shelter_capacity(self, area_id: str) -> list[dict]:
        return [
            {
                "shelter_id": item.shelter_id,
                "name": item.name,
                "village": item.village,
                "capacity": item.capacity,
                "available_capacity": item.available_capacity,
                "accessible": item.accessible,
            }
            for item in self.area_profiles[area_id].shelters
        ]

    def get_live_traffic(self, event_id: str) -> list[dict]:
        event = self.get_event(event_id)
        traffic = self.route_planner.build_live_traffic(
            self.area_profiles[event.area_id],
            self.get_hazard_state(event_id),
        )
        return [
            {"road_id": item.road_id, "congestion_index": item.congestion_index, "note": item.note}
            for item in traffic
        ]

    def get_policy_constraints(self, entity_type: str, risk_level: str) -> PolicyConstraint:
        return self.decision_engine.get_policy_constraints(EntityType(entity_type), RiskLevel(risk_level))

    def resolve_target_entity(
        self,
        event_id: str,
        question: str,
        *,
        preferred_entity_id: str | None = None,
        entity_type: str | None = None,
    ) -> dict:
        if preferred_entity_id:
            entity = self.get_entity_profile(preferred_entity_id)
            return {
                "entity_id": entity.entity_id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type.value,
                "reason": f"Resolved {entity.name} from the current plan context.",
            }
        exposure = self.get_exposure_summary(event_id, entity_type=entity_type, top_k=1)
        if exposure.affected_entities:
            entity = exposure.affected_entities[0].entity
            return {
                "entity_id": entity.entity_id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type.value,
                "reason": f"Resolved {entity.name} from exposure ranking for: {question}",
            }
        raise ValueError("当前问题没有可用的暴露对象。")

    def get_knowledge_evidence(
        self,
        *,
        event_id: str,
        area_id: str,
        entity_id: str | None = None,
    ) -> list[EvidenceItem]:
        entity = None
        if entity_id:
            try:
                entity = self.get_entity_profile(entity_id)
            except ValueError:
                entity = None
        risk_level = self.get_hazard_state(event_id).overall_risk_level
        return self._knowledge_evidence(entity, risk_level, area_id=area_id)

    def reconcile_regional_proposals(self, event_id: str, *, previous_risk_level: str | None = None) -> dict:
        event = self.get_event(event_id)
        hazard_state = self.get_hazard_state(event_id)
        exposure = self.get_exposure_summary(event_id)
        simulation_update = self.repository.get_latest_v2_simulation_update(event_id)
        if simulation_update is None:
            raise ValueError(f"No simulation update found for event {event_id}.")
        previous = RiskLevel(previous_risk_level) if previous_risk_level else None
        knowledge_evidence = self.get_knowledge_evidence(event_id=event_id, area_id=event.area_id)
        try:
            result = self.regional_proposals.reconcile(
                event=event,
                hazard_state=hazard_state,
                exposure_summary=exposure,
                simulation_update=simulation_update,
                previous_risk_level=previous,
                knowledge_evidence=knowledge_evidence,
            )
            llm_status = "ok"
            llm_error = None
        except LLMGenerationError as exc:
            self.save_operational_alert(
                source_type="regional_proposal_llm",
                severity=AlertSeverity.CRITICAL,
                summary="区域级主动决策模型调用失败。",
                details=str(exc),
                event_id=event_id,
            )
            self.add_audit_record(
                source_type="regional_proposal",
                action="regional_llm_failed",
                summary=f"区域级主动决策模型调用失败：{exc}",
                details={"event_id": event_id, "llm_error_code": exc.code},
                severity=AlertSeverity.CRITICAL,
                event_id=event_id,
            )
            return {
                "risk_stage_key": None,
                "pending_proposal_ids": [],
                "recommended_actions": [],
                "queue_changed": False,
                "should_interrupt": False,
                "llm_status": "failed",
                "llm_error": str(exc),
                "llm_error_code": exc.code,
            }
        return {
            "risk_stage_key": result.risk_stage_key,
            "pending_proposal_ids": result.pending_proposal_ids,
            "recommended_actions": result.recommended_actions,
            "queue_changed": result.queue_changed,
            "should_interrupt": result.should_interrupt,
            "llm_status": llm_status,
            "llm_error": llm_error,
        }

    def bootstrap_copilot_session(self, request: V2CopilotSessionRequest) -> V2CopilotSessionView:
        return self.copilot.bootstrap_session(request)

    def get_copilot_session(self, session_id: str) -> V2CopilotSessionView:
        return self.copilot.get_session_view(session_id)

    def send_copilot_message(self, session_id: str, request: V2CopilotMessageRequest) -> V2CopilotSessionView:
        return self.copilot.answer(session_id, request.content)

    def approve_copilot_proposal(
        self,
        session_id: str,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.approve_proposal(session_id, proposal_id, request)

    def reject_copilot_proposal(
        self,
        session_id: str,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.reject_proposal(session_id, proposal_id, request)

    def batch_approve_copilot_proposals(
        self,
        session_id: str,
        request: BatchProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.batch_approve_proposals(session_id, request)

    def batch_reject_copilot_proposals(
        self,
        session_id: str,
        request: BatchProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.batch_reject_proposals(session_id, request)

    def run_supervisor(self, event_id: str, *, trigger_type: str = "manual_run", session_id: str | None = None) -> SupervisorRunRecord:
        return self.agent_supervisor.run_for_event(event_id, trigger_type=trigger_type, session_id=session_id)

    def tick_supervisor(self, event_id: str | None = None) -> list[SupervisorRunRecord]:
        return self.agent_supervisor.tick(event_id)

    def publish_trigger(
        self,
        event_id: str,
        *,
        trigger_type: TriggerEventType,
        payload: dict | None = None,
        dedupe: bool = True,
    ) -> TriggerEvent:
        trigger = self.agent_supervisor.publish_trigger(
            event_id,
            trigger_type=trigger_type,
            payload=payload,
            dedupe=dedupe,
        )
        self.add_audit_record(
            source_type="trigger_event",
            action="trigger_published",
            summary=f"已为事件 {event_id} 发布触发事件 {trigger_type.value}。",
            details={"trigger_id": trigger.trigger_id, "event_id": event_id},
            event_id=event_id,
        )
        return trigger

    def get_session_memory(self, session_id: str) -> SessionMemoryView:
        session = self.repository.get_v2_copilot_session(session_id)
        if session is None:
            raise ValueError("v2 copilot session not found.")
        event = self.get_event(session["event_id"])
        snapshot = self.session_memory_store.load_snapshot(session_id, area_id=event.area_id)
        events = self.repository.list_v2_copilot_memory_events(session_id, limit=20)
        return SessionMemoryView(session_id=session_id, memory_snapshot=snapshot, recent_events=events)

    def get_event_shared_memory(self, event_id: str) -> SharedMemorySnapshot:
        return self.agent_supervisor.shared_memory.load(event_id)

    def get_memory_bundle(self, session_id: str, event_id: str) -> MemoryBundleView:
        return MemoryBundleView(
            session_memory=self.get_session_memory(session_id),
            event_shared_memory=self.get_event_shared_memory(event_id),
        )

    def list_daily_reports(self, event_id: str) -> list[DailyReportRecord]:
        return self.repository.list_v2_daily_reports(event_id, limit=20)

    def list_episode_summaries(self, event_id: str) -> list[EventEpisodeSummaryRecord]:
        return self.repository.list_v2_event_episode_summaries(event_id, limit=20)

    def list_long_term_memories(self, event_id: str) -> list[LongTermMemoryRecord]:
        return self.repository.list_v2_long_term_memories(event_id=event_id, limit=20)

    def get_experience_context(self, event_id: str) -> ExperienceContextView:
        event = self.get_event(event_id)
        shared = self.get_event_shared_memory(event_id)
        records = self.operational_experience_store.query_similar_cases(
            entity_type=None,
            risk_level=shared.latest_hazard_level,
            limit=8,
        )
        entity_type: str | None = None
        if records and records[0].entity_type:
            entity_type = records[0].entity_type
        elif shared.focus_entity_ids:
            try:
                entity_type = self.get_entity_profile(shared.focus_entity_ids[0]).entity_type.value
            except ValueError:
                entity_type = None
        patterns = self.operational_experience_store.rank_strategy_patterns(
            entity_type=entity_type,
            risk_level=shared.latest_hazard_level,
            limit=5,
        )
        notes: list[str] = []
        if patterns:
            top = patterns[0]
            action_label = {
                "regional_notification": "区域通知",
                "regional_evacuation": "区域转移建议",
                "regional_resource_dispatch": "区域资源调度",
            }.get(top.action_type, top.action_type)
            notes.append(
                f"区域 {event.area_id} 的历史主导模式为 {action_label}，在 {top.sample_size} 个样本中的批准率为 {top.approval_rate:.0%}。"
            )
        if not records:
            notes.append("当前可用的历史经验仍较少，建议主要依赖实时态势与策略规则。")
        long_term_memories = self.long_term_memory_store.query_memories(
            area_id=event.area_id,
            entity_type=entity_type,
            risk_level=shared.latest_hazard_level,
            top_k=5,
        )
        return ExperienceContextView(
            event_id=event_id,
            relevant_records=records,
            strategy_patterns=patterns,
            outcome_risk_notes=notes,
            long_term_memories=long_term_memories,
        )

    def get_strategy_history(self, entity_id: str) -> StrategyHistoryView:
        entity = self.get_entity_profile(entity_id)
        records = self.operational_experience_store.query_similar_cases(
            entity_id=entity_id,
            entity_type=entity.entity_type.value,
            limit=20,
        )
        patterns = self.operational_experience_store.rank_strategy_patterns(
            entity_type=entity.entity_type.value,
            limit=5,
        )
        return StrategyHistoryView(entity_id=entity_id, records=records, patterns=patterns)

    def get_shared_memory_snapshot(self, event_id: str) -> SharedMemorySnapshot:
        return self.agent_supervisor.shared_memory.load(event_id)

    def list_agent_tasks(self, event_id: str) -> list[AgentTask]:
        return self.repository.list_v2_agent_tasks(event_id, limit=80)

    def list_agent_results(self, event_id: str) -> list[AgentResult]:
        return self.repository.list_v2_agent_results(event_id, limit=40)

    def list_supervisor_runs(self, event_id: str) -> list[SupervisorRunRecord]:
        return self.repository.list_v2_supervisor_runs(event_id, limit=30)

    def list_trigger_events(self, event_id: str) -> list[TriggerEvent]:
        return self.repository.list_v2_trigger_events(event_id, limit=40)

    def list_agent_timeline(self, event_id: str) -> list[AgentTimelineEntry]:
        return self.repository.list_v2_agent_timeline(event_id, limit=120)

    def get_decision_report(self, event_id: str) -> DecisionReportView:
        shared = self.get_shared_memory_snapshot(event_id)
        return DecisionReportView(
            event_id=event_id,
            latest_summary=shared.latest_summary,
            active_decision_path=shared.active_decision_path,
            blocked_by=shared.blocked_by,
            open_questions=shared.open_questions,
            recent_result_ids=shared.recent_result_ids,
        )

    def replay_agent_task(self, task_id: str, request: ReplayRequest) -> AgentTask:
        task = self.agent_supervisor.replay_task(task_id, replay_reason=request.replay_reason)
        self.add_audit_record(
            source_type="agent_task",
            action="agent_task_replayed",
            summary=f"Replayed task {task_id}.",
            details={"new_task_id": task.task_id, "replay_reason": request.replay_reason},
            event_id=task.event_id,
            session_id=task.session_id,
        )
        return task

    def get_agent_status(self, event_id: str) -> dict:
        return self.agent_supervisor.get_agent_status(event_id)

    def get_agent_metrics(self) -> AgentMetricsView:
        recent_reports = self.repository.list_v2_evaluation_reports(limit=1)
        with self.repository._connect() as conn:
            failure_rows = conn.execute(
                """
                SELECT agent_name, COUNT(*) AS count
                FROM v2_agent_task_events
                WHERE event_type = 'task_failed'
                GROUP BY agent_name
                """
            ).fetchall()
            task_rows = conn.execute("SELECT payload FROM v2_agent_tasks").fetchall()
            result_rows = conn.execute("SELECT payload FROM v2_agent_results").fetchall()
        total_tasks = len(task_rows)
        superseded = 0
        fanout_count = 0
        for row in task_rows:
            task = AgentTask.model_validate_json(row["payload"])
            if task.status.value == "superseded":
                superseded += 1
        stale_data_frequency = 0
        for row in result_rows:
            result = AgentResult.model_validate_json(row["payload"])
            if any("stale" in slot.lower() for slot in result.missing_slots):
                stale_data_frequency += 1
            fanout_count += len(result.recommended_next_tasks)
        latency_ms = 0.0
        recent_runs = []
        for area_id in self.area_profiles:
            event_id = self.repository.get_latest_v2_event_id(area_id)
            if event_id:
                recent_runs.extend(self.repository.list_v2_supervisor_runs(event_id, limit=5))
        durations = [
            max((run.completed_at - run.created_at).total_seconds() * 1000, 0.0)
            for run in recent_runs
            if run.completed_at is not None
        ]
        if durations:
            latency_ms = round(sum(durations) / len(durations), 1)
        return AgentMetricsView(
            generated_at=datetime.now(timezone.utc),
            task_graph_latency_ms=latency_ms,
            agent_failure_heatmap={str(row["agent_name"]): int(row["count"]) for row in failure_rows},
            stale_data_frequency=stale_data_frequency,
            auto_retry_success_rate=recent_reports[0].tool_selection_correctness if recent_reports else 0.0,
            superseded_task_ratio=round(superseded / total_tasks, 2) if total_tasks else 0.0,
            fanout_count=fanout_count,
            stale_data_replan_count=stale_data_frequency,
        )

    def get_evaluation_report(self, report_id: str) -> EvaluationReport:
        report = self.repository.get_v2_evaluation_report(report_id)
        if report is None:
            raise ValueError(f"Unknown evaluation report: {report_id}")
        return report

    def list_evaluation_benchmarks(self) -> list[EvaluationBenchmark]:
        return [
            EvaluationBenchmark(
                benchmark_id="elderly-lowland",
                title="低洼老人家庭影响判断",
                question="当前水位对低洼老人家庭意味着什么？",
                scenario_type="elderly",
                expected_tools=["resolve_target_entity", "get_hazard_tiles", "synthesize_entity_impact", "get_knowledge_evidence"],
                expected_completion_status=CompletionStatus.CONSERVATIVE_ANSWER,
                expected_human_confirmation=False,
            ),
            EvaluationBenchmark(
                benchmark_id="school-escalation",
                title="小学风险升级判断",
                question="当前洪水对附近小学意味着什么？",
                scenario_type="school",
                expected_tools=["resolve_target_entity", "get_hazard_tiles", "synthesize_entity_impact", "get_policy_constraints"],
                expected_completion_status=CompletionStatus.HUMAN_ESCALATION,
                expected_human_confirmation=True,
            ),
            EvaluationBenchmark(
                benchmark_id="factory-proposal",
                title="工厂库存与停工建议",
                question="当前洪水对附近工厂的库存和停工安排意味着什么？",
                scenario_type="factory",
                expected_tools=["resolve_target_entity", "synthesize_entity_impact", "draft_action_proposal"],
                expected_completion_status=CompletionStatus.HUMAN_ESCALATION,
                expected_human_confirmation=True,
            ),
            EvaluationBenchmark(
                benchmark_id="route-guidance",
                title="养老机构转移路线建议",
                question="养老机构应通过哪条路线前往最安全的避难点？",
                scenario_type="route",
                expected_tools=["resolve_target_entity", "get_route_options", "get_live_traffic", "get_shelter_capacity"],
                expected_completion_status=CompletionStatus.HUMAN_ESCALATION,
                expected_human_confirmation=True,
            ),
        ]

    def run_evaluation(self) -> EvaluationReport:
        report = self._run_evaluation_for_benchmarks(self.list_evaluation_benchmarks())
        self.repository.save_v2_evaluation_report(report)
        return report

    def replay_evaluation_report(self, report_id: str) -> EvaluationReport:
        source_report = self.get_evaluation_report(report_id)
        benchmark_ids = [item.benchmark_id for item in source_report.scenario_results]
        benchmarks = [item for item in self.list_evaluation_benchmarks() if item.benchmark_id in benchmark_ids]
        if not benchmarks:
            raise ValueError(f"Evaluation report {report_id} does not contain replayable benchmark scenarios.")
        report = self._run_evaluation_for_benchmarks(benchmarks)
        report.notes = [f"已重放评测报告 {report_id}。", *report.notes]
        self.repository.save_v2_evaluation_report(report)
        return report

    def _run_evaluation_for_benchmarks(
        self,
        benchmarks: list[EvaluationBenchmark],
    ) -> EvaluationReport:
        scenario_results: list[BenchmarkScenarioResult] = []
        tool_scores: list[float] = []
        dispatch_scores: list[float] = []
        reuse_scores: list[float] = []
        evidence_scores: list[float] = []
        escalation_scores: list[float] = []
        hallucination_scores: list[float] = []

        for benchmark in benchmarks:
            event_id = self._create_evaluation_event(benchmark)
            session = self.bootstrap_copilot_session(
                V2CopilotSessionRequest(event_id=event_id, operator_role="commander")
            )
            first_view = self.send_copilot_message(
                session.session_id,
                V2CopilotMessageRequest(content=benchmark.question),
            )
            follow_up = self.send_copilot_message(
                session.session_id,
                V2CopilotMessageRequest(content="Continue with that target and restate the highest-priority action."),
            )
            answer = first_view.latest_answer
            if answer is None:
                raise ValueError("evaluation benchmark did not produce a structured answer.")

            used_tools = [item.tool_name for item in answer.tool_executions]
            missing_expected = [tool for tool in benchmark.expected_tools if tool not in used_tools]
            tool_score = 1.0 if not benchmark.expected_tools else round(
                (len(benchmark.expected_tools) - len(missing_expected)) / len(benchmark.expected_tools),
                2,
            )
            dispatch_ok = bool(answer.plan_runs and answer.plan_runs[0].selected_tools)
            memory_reused = bool(follow_up.latest_answer and follow_up.latest_answer.carried_context_notes)
            evidence_ok = bool(answer.evidence)
            escalation_ok = answer.requires_human_confirmation == benchmark.expected_human_confirmation
            hallucination = 1.0 if (not answer.evidence and answer.confidence >= 0.6) else 0.0
            status_ok = benchmark.expected_completion_status is None or answer.completion_status == benchmark.expected_completion_status
            passed = tool_score >= 0.75 and evidence_ok and escalation_ok and status_ok

            notes: list[str] = []
            if missing_expected:
                notes.append(f"Missing expected tools: {', '.join(missing_expected)}.")
            if not evidence_ok:
                notes.append("当前回答没有附带证据条目。")
            if not status_ok and benchmark.expected_completion_status is not None:
                notes.append(
                    f"Expected completion status {benchmark.expected_completion_status.value}, got {answer.completion_status.value}."
                )
            if not escalation_ok:
                notes.append("Human escalation behavior did not match the benchmark expectation.")
            if not memory_reused:
                notes.append("Follow-up turn did not reuse carried context notes.")

            scenario_results.append(
                BenchmarkScenarioResult(
                    benchmark_id=benchmark.benchmark_id,
                    title=benchmark.title,
                    passed=passed,
                    event_id=event_id,
                    session_id=session.session_id,
                    used_tools=used_tools,
                    expected_tools=benchmark.expected_tools,
                    completion_status=answer.completion_status,
                    expected_completion_status=benchmark.expected_completion_status,
                    human_confirmation=answer.requires_human_confirmation,
                    expected_human_confirmation=benchmark.expected_human_confirmation,
                    evidence_count=len(answer.evidence),
                    shared_memory_reused=memory_reused,
                    notes=notes,
                )
            )
            tool_scores.append(tool_score)
            dispatch_scores.append(1.0 if dispatch_ok else 0.0)
            reuse_scores.append(1.0 if memory_reused else 0.0)
            evidence_scores.append(1.0 if evidence_ok else 0.0)
            escalation_scores.append(1.0 if escalation_ok else 0.0)
            hallucination_scores.append(hallucination)

        return EvaluationReport(
            report_id=f"eval_{uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc),
            benchmark_count=len(benchmarks),
            tool_selection_correctness=round(sum(tool_scores) / len(tool_scores), 2) if tool_scores else 0.0,
            dynamic_dispatch_correctness=round(sum(dispatch_scores) / len(dispatch_scores), 2) if dispatch_scores else 0.0,
            shared_memory_reuse_rate=round(sum(reuse_scores) / len(reuse_scores), 2) if reuse_scores else 0.0,
            evidence_coverage_rate=round(sum(evidence_scores) / len(evidence_scores), 2) if evidence_scores else 0.0,
            human_escalation_correctness=round(sum(escalation_scores) / len(escalation_scores), 2) if escalation_scores else 0.0,
            hallucination_rate=round(sum(hallucination_scores) / len(hallucination_scores), 2) if hallucination_scores else 0.0,
            scenario_results=scenario_results,
            notes=[
                "This report was generated from isolated benchmark runs rather than recent live-message sampling.",
                "A second follow-up turn is executed for every benchmark to verify session-memory reuse.",
            ],
        )

    def _create_evaluation_event(self, benchmark: EvaluationBenchmark) -> str:
        event = self.create_event(
            EventCreateRequest(
                area_id="beilin_10km2",
                title=f"Evaluation benchmark: {benchmark.title}",
                trigger_reason=f"evaluation_{benchmark.benchmark_id}",
                operator="evaluation_runner",
            )
        )
        observations = self._load_evaluation_observations(benchmark.scenario_type)
        self.ingest_observations(
            event.event_id,
            ObservationBatchRequest(operator="evaluation_runner", observations=observations),
        )
        self.agent_supervisor.tick(event.event_id)
        return event.event_id

    def _load_evaluation_observations(self, scenario_type: str) -> list[ObservationIngestItem]:
        filename = {
            "elderly": "observations_beilin_warning.csv",
            "school": "observations_beilin_extreme.csv",
            "factory": "observations_beilin_extreme.csv",
            "route": "observations_beilin_warning.csv",
        }.get(scenario_type, "observations_beilin_warning.csv")
        csv_path = Path(__file__).resolve().parent.parent / "bootstrap_data" / filename
        observations: list[ObservationIngestItem] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                observations.append(
                    ObservationIngestItem(
                        observed_at=datetime.fromisoformat(str(row["observed_at"]).replace("Z", "+00:00")),
                        source_type=str(row.get("source_type", "monitoring_point")),
                        source_name=str(row.get("source_name", "")),
                        village=str(row.get("village") or "") or None,
                        rainfall_mm=float(row.get("rainfall_mm") or 0.0),
                        water_level_m=float(row.get("water_level_m") or 0.0),
                        road_blocked=str(row.get("road_blocked", "")).strip().lower() in {"1", "true", "yes"},
                        citizen_reports=int(float(row.get("citizen_reports") or 0)),
                        notes=str(row.get("notes", "")),
                    )
                )
        if not observations:
            raise ValueError(f"Evaluation observations are missing from {csv_path.name}.")
        return observations

    def list_operational_alerts(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 50,
    ) -> list[OperationalAlert]:
        return self.repository.list_operational_alerts(
            event_id=event_id,
            severity=severity,
            source_type=source_type,
            status="open",
            from_ts=from_ts,
            to_ts=to_ts,
            limit=limit,
        )

    def list_audit_records(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        return self.repository.list_audit_records(
            event_id=event_id,
            severity=severity,
            source_type=source_type,
            from_ts=from_ts,
            to_ts=to_ts,
            limit=limit,
        )

    def get_archive_status(self) -> ArchiveStatusView:
        return self.repository.get_archive_status()

    def run_archive_cycle(self) -> ArchiveStatusView:
        status = self.repository.archive_operational_records()
        self.add_audit_record(
            source_type="housekeeping",
            action="archive_run_completed",
            summary="Manual archive cycle completed.",
            details={
                "hot_records_archived": status.last_archive_run.hot_records_archived if status.last_archive_run else 0,
                "expired_archives_deleted": status.last_archive_run.expired_archives_deleted if status.last_archive_run else 0,
            },
        )
        return status

    def _regional_analysis_package_view(self, proposals: list[ActionProposal]) -> RegionalAnalysisPackageView:
        latest_proposals = self._latest_package_proposals(proposals)
        if not latest_proposals:
            raise ValueError("Regional analysis package requires at least one proposal.")

        seed = latest_proposals[0]
        event = self.get_event(seed.event_id)
        package_meta = self._package_payload(seed)
        focus_object_ids = package_meta.get("focus_object_ids") or self._collect_focus_object_ids(latest_proposals)
        focus_object_names: list[str] = []
        for entity_id in focus_object_ids:
            try:
                focus_object_names.append(self.get_entity_profile(entity_id).name)
            except ValueError:
                continue

        created_at = min(item.created_at for item in latest_proposals)
        updated_at = max((item.updated_at or item.resolved_at or item.created_at) for item in latest_proposals)
        current_risk_level = package_meta.get("current_risk_level") or event.current_risk_level.value
        return RegionalAnalysisPackageView(
            package_id=seed.risk_stage_key or seed.proposal_id,
            event_id=seed.event_id,
            current_risk_level=RiskLevel(current_risk_level),
            trigger_type=str(package_meta.get("trigger_type") or "simulation_updated"),
            focus_object_ids=focus_object_ids,
            focus_object_names=focus_object_names,
            proposal_ids=[item.proposal_id for item in latest_proposals],
            proposal_titles=[item.title for item in latest_proposals],
            proposal_count=len(latest_proposals),
            analysis_message=str(package_meta.get("analysis_message") or self._default_package_analysis_message(event, latest_proposals)),
            risk_assessment=str(package_meta.get("risk_assessment") or self._default_package_risk_assessment(event, latest_proposals)),
            rescue_plan=str(package_meta.get("rescue_plan") or self._default_package_rescue_plan(latest_proposals)),
            resource_dispatch_plan=str(
                package_meta.get("resource_dispatch_plan") or self._default_package_resource_plan(latest_proposals)
            ),
            status=self._derive_package_status(latest_proposals),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _regional_proposal_view(
        self,
        proposal: ActionProposal,
        event: EventRecord,
        hazard_state: HazardState,
    ) -> RegionalProposalView:
        names = []
        for entity_id in proposal.high_risk_object_ids:
            try:
                names.append(self.get_entity_profile(entity_id).name)
            except ValueError:
                continue
        return RegionalProposalView(
            proposal=proposal,
            event_title=event.title,
            current_risk_level=hazard_state.overall_risk_level,
            high_risk_object_names=names,
        )

    def _regional_proposal_view_for_any_event(self, proposal: ActionProposal) -> RegionalProposalView:
        event = self.get_event(proposal.event_id)
        hazard_state = self.get_hazard_state(proposal.event_id)
        return self._regional_proposal_view(proposal, event, hazard_state)

    def _pending_package_proposals(self, package_id: str) -> list[ActionProposal]:
        return [
            item
            for item in self.repository.list_v2_action_proposals(proposal_scope="regional")
            if item.risk_stage_key == package_id and item.status == ProposalStatus.PENDING
        ]

    def _get_regional_proposal_or_raise(self, proposal_id: str) -> ActionProposal:
        proposal = self.repository.get_v2_action_proposal(proposal_id)
        if proposal is None or proposal.proposal_scope != "regional":
            raise ValueError(f"Unknown regional proposal: {proposal_id}")
        return proposal

    @staticmethod
    def _latest_package_proposals(proposals: list[ActionProposal]) -> list[ActionProposal]:
        latest: dict[str, ActionProposal] = {}
        ordered = sorted(proposals, key=lambda item: (item.created_at, item.proposal_id), reverse=True)
        for proposal in ordered:
            key = proposal.action_type or proposal.proposal_id
            if key in latest:
                continue
            latest[key] = proposal
        return sorted(latest.values(), key=lambda item: (item.created_at, item.proposal_id))

    @staticmethod
    def _package_payload(proposal: ActionProposal) -> dict:
        payload = proposal.payload.get(PACKAGE_PAYLOAD_KEY)
        return payload if isinstance(payload, dict) else {}

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
    def _derive_package_status(proposals: list[ActionProposal]) -> RegionalAnalysisPackageStatus:
        statuses = {item.status for item in proposals}
        if ProposalStatus.PENDING in statuses:
            return RegionalAnalysisPackageStatus.PENDING
        if statuses == {ProposalStatus.APPROVED}:
            return RegionalAnalysisPackageStatus.APPROVED
        if statuses == {ProposalStatus.REJECTED}:
            return RegionalAnalysisPackageStatus.REJECTED
        if statuses == {ProposalStatus.WITHDRAWN}:
            return RegionalAnalysisPackageStatus.WITHDRAWN
        if statuses == {ProposalStatus.SUPERSEDED}:
            return RegionalAnalysisPackageStatus.SUPERSEDED
        return RegionalAnalysisPackageStatus.PARTIALLY_RESOLVED

    @staticmethod
    def _default_package_analysis_message(event: EventRecord, proposals: list[ActionProposal]) -> str:
        return f"{event.title} has a regional analysis package covering {len(proposals)} coordinated actions."

    @staticmethod
    def _default_package_risk_assessment(event: EventRecord, proposals: list[ActionProposal]) -> str:
        risk_level = proposals[0].severity if proposals else event.current_risk_level.value
        return f"The current package was generated under {risk_level} risk conditions for {event.title}."

    @staticmethod
    def _default_package_rescue_plan(proposals: list[ActionProposal]) -> str:
        if not proposals:
            return "No rescue actions were generated."
        return " | ".join(item.recommendation or item.summary for item in proposals[:2])

    @staticmethod
    def _default_package_resource_plan(proposals: list[ActionProposal]) -> str:
        if not proposals:
            return "No resource dispatch actions were generated."
        resource_actions = [
            item.recommendation or item.summary
            for item in proposals
            if item.execution_mode in {ExecutionMode.RESOURCE_DISPATCH, ExecutionMode.GENERIC_TASK}
        ]
        return " | ".join(resource_actions[:2] or [proposals[0].recommendation or proposals[0].summary])

    def _sync_high_risk_transition(
        self,
        *,
        event: EventRecord,
        previous_risk_level: RiskLevel | None,
        current_risk_level: RiskLevel,
        trigger_source: str,
        observed_at: datetime | None = None,
    ) -> None:
        if self.event_postmortem_service is None:
            return
        self.event_postmortem_service.sync_risk_transition(
            event=event,
            previous_risk_level=previous_risk_level,
            current_risk_level=current_risk_level,
            trigger_source=trigger_source,
            observed_at=observed_at,
        )

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

    def _impact_from_location_request(self, request: AdvisoryRequest) -> EntityImpactView:
        area_id = request.area_id
        village = request.village or request.profile_overrides.get("village") or self.area_profiles[area_id].villages[0]
        synthetic = EntityProfile(
            entity_id="ad_hoc_location",
            area_id=area_id,
            entity_type=EntityType(request.profile_overrides.get("entity_type", "resident")),
            name=request.profile_overrides.get("name", request.location_hint or "临时建议对象"),
            village=village,
            location_hint=request.location_hint or village,
            resident_count=int(request.profile_overrides.get("resident_count", 1)),
            current_occupancy=int(
                request.profile_overrides.get(
                    "current_occupancy",
                    request.profile_overrides.get("resident_count", 1),
                )
            ),
            vulnerability_tags=list(request.profile_overrides.get("vulnerability_tags", [])),
            mobility_constraints=list(request.profile_overrides.get("mobility_constraints", [])),
            key_assets=list(request.profile_overrides.get("key_assets", [])),
            inventory_summary=str(request.profile_overrides.get("inventory_summary", "")),
            continuity_requirement=str(request.profile_overrides.get("continuity_requirement", "")),
            preferred_transport_mode=request.profile_overrides.get("preferred_transport_mode", "walk"),
            notification_preferences=list(request.profile_overrides.get("notification_preferences", [])),
            emergency_contacts=[],
            custom_attributes=dict(request.profile_overrides),
        )
        resolved_event_id = request.event_id or self.repository.get_latest_v2_event_id(area_id)
        if resolved_event_id is None:
            raise ValueError(f"No active v2 event found for area {area_id}.")
        hazard_state = self.get_hazard_state(resolved_event_id)
        return self.exposure_engine.assess_entity(
            hazard_state.event_id,
            synthetic,
            self.area_profiles[area_id],
            hazard_state,
            self.get_resource_status(area_id, event_id=resolved_event_id),
            evidence=self.get_knowledge_evidence(
                event_id=hazard_state.event_id,
                area_id=area_id,
                entity_id=None,
            ),
        )

    def _knowledge_evidence(
        self,
        entity: EntityProfile | None,
        risk_level: RiskLevel,
        *,
        area_id: str | None = None,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        resolved_area_id = entity.area_id if entity is not None else (area_id or "beilin_10km2")
        filters = {"region": self.area_profiles[resolved_area_id].region}
        policy_docs = self.rag_service.query(
            CorpusType.POLICY,
            f"{risk_level.value} response evacuation",
            filters=filters,
            top_k=2,
        )
        case_docs = self.rag_service.query(
            CorpusType.CASE,
            "vulnerable school factory flood",
            filters=filters,
            top_k=2,
        )
        profile_docs = self.rag_service.query(
            CorpusType.PROFILE,
            "shelter vulnerable medical",
            filters=filters,
            top_k=2,
        )
        memory_records = self.long_term_memory_store.query_memories(
            area_id=resolved_area_id,
            entity_type=entity.entity_type.value if entity is not None else None,
            risk_level=risk_level,
            top_k=2,
        )
        for priority, docs, evidence_type in (
            (2, policy_docs, "policy"),
            (3, case_docs, "case"),
            (4, profile_docs, "profile"),
        ):
            for doc in docs:
                evidence.append(
                    EvidenceItem(
                        evidence_type=evidence_type,
                        title=doc.title,
                        source_id=doc.doc_id,
                        excerpt=doc.content[:120],
                        priority=priority,
                        retrieval_explain=self.rag_service.explain(doc),
                    )
                )
        for record in memory_records:
            evidence.append(
                EvidenceItem(
                    evidence_type="memory",
                    title=record.headline,
                    source_id=record.memory_id,
                    excerpt=record.summary[:120],
                    priority=5,
                    retrieval_explain={
                        "tags": record.tags,
                        "entity_types": record.entity_types,
                        "action_types": record.action_types,
                        "source_summary_id": record.source_summary_id,
                    },
                )
            )
        if entity is not None:
            evidence.insert(
                0,
                EvidenceItem(
                    evidence_type="profile",
                    title=f"{entity.name}画像快照",
                    source_id=entity.entity_id,
                    excerpt=(
                        f"{entity.location_hint}; type={entity.entity_type.value}; "
                        f"key vulnerability={','.join(entity.vulnerability_tags[:3]) or 'none'}"
                    ),
                    priority=1,
                )
            )
        return evidence

