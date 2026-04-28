from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1

from ..models import RiskLevel
from .llm_gateway import LLMGenerationError
from .models import (
    ActionProposal,
    AlertSeverity,
    EventRecord,
    ExecutionMode,
    HazardState,
    ProposalDraftUpdateRequest,
    ProposalResolutionRequest,
    ProposalStatus,
    RegionalAnalysisPackageStatus,
    RegionalAnalysisPackageView,
    RegionalProposalQueueSnapshot,
    RegionalProposalView,
)
from .regional_proposals import PACKAGE_PAYLOAD_KEY


class PlatformRegionalOpsMixin:
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
