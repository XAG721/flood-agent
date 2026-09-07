from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from ..models import CorpusType, RAGDocument
from ..simulation_dataset import SimulatedExternalGateway
from .feedback_policy import (
    recommend_deadline_alternatives,
)
from .evidence_governance import (
    UnavailableNliAdapter,
    task_field_slots,
)
from .state_machine import ensure_transition

from .models import (
    AlertAppendRequest,
    AlertSnapshot,
    EscalationRecord,
    EventCreateRequest,
    EventDashboard,
    EventRiskObject,
    EventStatus,
    EvidenceRole,
    ObjectVerificationStatus,
    OperatorRole,
    ResponseEvent,
    ResponseTask,
    RiskObjectBatchRequest,
    RiskObjectInput,
    RiskObjectVerificationRequest,
    RiskObjectVersionSnapshot,
    TaskActionRequest,
    TaskDraftGenerationRequest,
    TaskStatus,
    TaskVersionSnapshot,
    TimelineEntry,
)

from .service_shared import EDIT_ROLES, EXECUTION_ROLES
from .ports import RagServicePort, ResponseWorkflowRepositoryPort, SimulationGatewayPort


class ResponseWorkflowCoreMixin:
    def __init__(
        self,
        repository: ResponseWorkflowRepositoryPort,
        rag_service: RagServicePort | None = None,
        simulation_gateway: SimulationGatewayPort | None = None,
        nli_adapter=None,
    ) -> None:
        self.repository = repository
        self.rag_service = rag_service
        self.simulation_gateway = simulation_gateway or SimulatedExternalGateway()
        self.nli_adapter = nli_adapter or UnavailableNliAdapter()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex[:12]}"

    @staticmethod
    def _require_role(
        role: OperatorRole, allowed: set[OperatorRole], action: str
    ) -> None:
        if role not in allowed:
            raise PermissionError(f"{role.value} is not authorized to {action}")

    def _event(self, event_id: str, *, active: bool = False) -> ResponseEvent:
        event = self.repository.get_response_event(event_id)
        if event is None:
            raise LookupError(f"response event not found: {event_id}")
        if active and event.status == EventStatus.CLOSED:
            raise ValueError("closed events are read-only")
        return event

    def _task(self, task_id: str) -> ResponseTask:
        task = self.repository.get_response_task(task_id)
        if task is None:
            raise LookupError(f"response task not found: {task_id}")
        return task

    def _record(
        self,
        *,
        event_id: str,
        entry_type: str,
        action: str,
        actor_id: str,
        actor_role: str,
        terminal_id: str = "unknown-terminal",
        task_id: str | None = None,
        object_id: str | None = None,
        before_state: dict | None = None,
        after_state: dict | None = None,
        detail: dict | None = None,
    ) -> TimelineEntry:
        previous_entries = self.repository.list_timeline_entries(event_id)
        previous_hash = (
            previous_entries[-1].record_hash
            if previous_entries and previous_entries[-1].record_hash
            else "GENESIS"
        )
        entry = TimelineEntry(
            entry_id=self._id("LOG"),
            event_id=event_id,
            task_id=task_id,
            object_id=object_id,
            entry_type=entry_type,
            action=action,
            actor_id=actor_id,
            actor_role=actor_role,
            terminal_id=terminal_id,
            before_state=before_state or {},
            after_state=after_state or {},
            detail=detail or {},
            previous_hash=previous_hash,
            created_at=self._now(),
        )
        entry = entry.model_copy(
            update={"record_hash": self._timeline_entry_hash(entry)}
        )
        self.repository.save_timeline_entry(entry)
        return entry

    @staticmethod
    def _timeline_entry_hash(entry: TimelineEntry) -> str:
        canonical = json.dumps(
            entry.model_dump(mode="json", exclude={"record_hash"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def create_event(self, request: EventCreateRequest) -> EventDashboard:
        self._require_role(request.operator_role, EDIT_ROLES, "create an event")
        now = self._now()
        event_id = f"FLOOD-{now:%Y%m%d}-{uuid4().hex[:6].upper()}"
        event = ResponseEvent(
            event_id=event_id,
            title=request.title,
            area_id=request.area_id,
            created_at=now,
            updated_at=now,
            workflow_engine_version=self.WORKFLOW_ENGINE_VERSION,
            source_type=request.alert.source_type,
            is_simulated=request.alert.is_simulated,
        )
        snapshot = AlertSnapshot(
            **request.alert.model_dump(),
            snapshot_id=self._id("ALT"),
            event_id=event_id,
            version=1,
            created_at=now,
            raw_payload_hash=hashlib.sha256(
                request.alert.raw_content.encode("utf-8")
            ).hexdigest(),
        )
        self.repository.save_response_event(event)
        self.repository.save_alert_snapshot(snapshot)
        self._record(
            event_id=event_id,
            entry_type="event",
            action="event_created",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            after_state={"event_status": event.status.value, "alert_version": 1},
            detail={"alert_id": request.alert.alert_id, "alert_version": 1},
        )
        return self.get_dashboard(event_id)

    def append_alert(self, event_id: str, request: AlertAppendRequest) -> AlertSnapshot:
        self._require_role(
            request.operator_role, EDIT_ROLES, "append an alert snapshot"
        )
        event = self._event(event_id, active=True)
        snapshots = self.repository.list_alert_snapshots(event_id)
        if (
            snapshots
            and snapshots[-1].alert_id == request.alert.alert_id
            and snapshots[-1].raw_content == request.alert.raw_content
        ):
            return snapshots[-1]
        version = len(snapshots) + 1
        snapshot = AlertSnapshot(
            **request.alert.model_dump(),
            snapshot_id=self._id("ALT"),
            event_id=event_id,
            version=version,
            created_at=self._now(),
            raw_payload_hash=hashlib.sha256(
                request.alert.raw_content.encode("utf-8")
            ).hexdigest(),
        )
        self.repository.save_alert_snapshot(snapshot)
        event = event.model_copy(
            update={"current_alert_version": version, "updated_at": self._now()}
        )
        self.repository.save_response_event(event)
        for item in self.repository.list_event_risk_objects(event_id):
            if item.stale:
                continue
            stale = item.model_copy(
                update={
                    "stale": True,
                    "version": item.version + 1,
                    "updated_at": self._now(),
                }
            )
            self.repository.save_event_risk_object(stale)
            self._save_risk_object_version(
                stale,
                change_type="warning_revision_marked_stale",
                operator_id=request.operator_id,
                terminal_id=request.terminal_id,
            )
        self._record(
            event_id=event_id,
            entry_type="alert",
            action="alert_version_appended",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state={"alert_version": version - 1},
            after_state={"alert_version": version},
            detail={"alert_id": request.alert.alert_id, "version": version},
        )
        return snapshot

    def list_events(self) -> list[ResponseEvent]:
        return self.repository.list_response_events()

    def list_task_versions(self, task_id: str) -> list[TaskVersionSnapshot]:
        self._task(task_id)
        return self.repository.list_task_version_snapshots(task_id)

    def get_dashboard(self, event_id: str) -> EventDashboard:
        event = self._event(event_id)
        objects = self.repository.list_event_risk_objects(event_id)
        tasks = self.repository.list_response_tasks(event_id)
        timeline = self.repository.list_timeline_entries(event_id)
        feedback = self.repository.list_event_feedback(event_id)
        escalations = self.repository.list_escalations(event_id)
        candidate_object_lists = self.repository.list_candidate_object_lists(event_id)
        completed = sum(
            task.status
            in {TaskStatus.COMPLETED, TaskStatus.WAIVED, TaskStatus.CANCELLED}
            for task in tasks
        )
        escalated = sum(
            task.status in {TaskStatus.ESCALATED, TaskStatus.BLOCKED} for task in tasks
        )
        confirmed = sum(
            item.verification_status == ObjectVerificationStatus.CONFIRMED
            for item in objects
        )
        return EventDashboard(
            event=event,
            alert_snapshots=self.repository.list_alert_snapshots(event_id),
            risk_objects=objects,
            tasks=tasks,
            assignments=self.repository.list_event_task_assignments(event_id),
            feedback=feedback,
            escalations=escalations,
            review_draft=self.repository.get_latest_event_review_draft(event_id),
            scenario_report=self.repository.get_latest_district_scenario_report(
                event_id
            ),
            candidate_runs=self.repository.list_candidate_runs(event_id),
            candidate_object_lists=candidate_object_lists,
            evidence_packages=self.repository.list_evidence_packages(event_id),
            outbox=self.repository.list_outbox_messages(event_id=event_id),
            rule_evaluations=self.repository.list_rule_evaluations(event_id=event_id),
            deadline_extensions=self.repository.list_deadline_extensions(event_id),
            risk_object_versions=self.repository.list_risk_object_versions(event_id),
            timeline=timeline,
            metrics={
                "alert_versions": event.current_alert_version,
                "candidate_objects": len(objects),
                "confirmed_objects": confirmed,
                "frozen_candidate_lists": len(candidate_object_lists),
                "task_count": len(tasks),
                "completed_tasks": completed,
                "escalated_tasks": escalated,
                "feedback_count": len(feedback),
                "duplicate_feedback_merged": sum(
                    entry.action == "duplicate_feedback_merged" for entry in timeline
                ),
                "timeline_entries": len(timeline),
                "completion_rate": round(completed / len(tasks), 4) if tasks else 0.0,
            },
        )

    def _save_task_version_snapshot(
        self,
        task: ResponseTask,
        *,
        change_type: str,
        changed_fields: list[str],
        operator_id: str,
        terminal_id: str,
        note: str,
    ) -> TaskVersionSnapshot:
        snapshot = TaskVersionSnapshot(
            snapshot_id=self._id("TASKVER"),
            event_id=task.event_id,
            task_id=task.task_id,
            version=task.version,
            task=task,
            change_type=change_type,
            changed_fields=changed_fields,
            note=note,
            created_by=operator_id,
            terminal_id=terminal_id,
            created_at=self._now(),
        )
        self.repository.save_task_version_snapshot(snapshot)
        return snapshot

    def _save_risk_object_version(
        self,
        item: EventRiskObject,
        *,
        change_type: str,
        operator_id: str,
        terminal_id: str,
    ) -> RiskObjectVersionSnapshot:
        snapshot = RiskObjectVersionSnapshot(
            snapshot_id=self._id("OBJVER"),
            event_id=item.event_id,
            object_id=item.object_id,
            version=item.version,
            object=item,
            change_type=change_type,
            changed_by=operator_id,
            terminal_id=terminal_id,
            created_at=self._now(),
        )
        self.repository.save_risk_object_version(snapshot)
        return snapshot

    def get_dashboard_for_identity(self, event_id: str, identity) -> EventDashboard:
        dashboard = self.get_dashboard(event_id)
        full_sensitive_roles = {
            OperatorRole.DUTY_OFFICER,
            OperatorRole.REVIEWER,
            OperatorRole.COMMANDER,
            OperatorRole.ADMIN,
        }
        precise_location_roles = full_sensitive_roles | {
            OperatorRole.LIAISON,
            OperatorRole.FIELD_OPERATOR,
        }
        can_view_sensitive = identity.operator_role in full_sensitive_roles
        can_view_precise_location = identity.operator_role in precise_location_roles
        redacted_objects: list[EventRiskObject] = []
        for item in dashboard.risk_objects:
            updates: dict[str, object] = {}
            if not can_view_precise_location:
                updates["location"] = "[精确位置已按岗位权限脱敏]"
            if not can_view_sensitive:
                updates["sensitive_contacts"] = []
                if item.special_population_notes:
                    updates["special_population_notes"] = (
                        "[特殊人群信息已按岗位权限脱敏]"
                    )
            redacted_objects.append(item.model_copy(update=updates))
        redacted_alerts = dashboard.alert_snapshots
        if not can_view_precise_location:
            redacted_alerts = [
                item.model_copy(update={"affected_geometry": None})
                for item in dashboard.alert_snapshots
            ]
        return dashboard.model_copy(
            update={
                "risk_objects": redacted_objects,
                "alert_snapshots": redacted_alerts,
            }
        )

    def _transition(
        self,
        task_id: str,
        request: TaskActionRequest,
        allowed_from: set[TaskStatus],
        target: TaskStatus,
        action: str,
    ) -> ResponseTask:
        self._require_role(request.operator_role, EXECUTION_ROLES, action)
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        self._assert_expected_version(task, request.expected_version)
        if task.status not in allowed_from:
            raise ValueError(
                f"cannot transition task from {task.status.value} to {target.value}"
            )
        previous_status = task.status
        ensure_transition(task.status, target)
        task = task.model_copy(update={"status": target, "updated_at": self._now()})
        self.repository.save_response_task(task)
        self._task_event(
            task,
            action,
            request,
            {"note": request.note, "previous_status": previous_status.value},
        )
        return task

    def _task_event(
        self, task: ResponseTask, action: str, request: TaskActionRequest, detail: dict
    ) -> None:
        self._record(
            event_id=task.event_id,
            task_id=task.task_id,
            object_id=task.object_id,
            entry_type="task_status",
            action=action,
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state={"status": detail["previous_status"]}
            if detail.get("previous_status")
            else {},
            after_state={"status": task.status.value, "version": task.version},
            detail={"status": task.status.value, **detail},
        )

    def _save_escalation(
        self,
        task: ResponseTask,
        previous: TaskStatus,
        reason: str,
        target_role: str,
        recommended_actions: list[str] | None = None,
    ) -> EscalationRecord:
        record = EscalationRecord(
            escalation_id=self._id("ESC"),
            event_id=task.event_id,
            task_id=task.task_id,
            reason=reason,
            previous_status=previous,
            target_role=target_role,
            recommended_actions=recommended_actions
            or recommend_deadline_alternatives(reason),
            created_at=self._now(),
        )
        self.repository.save_escalation_record(record)
        return record

    def _query_task_evidence(self, query: str) -> list[RAGDocument]:
        if self.rag_service is None:
            return []
        documents = self.rag_service.query_evidence_set(
            CorpusType.POLICY,
            query,
            top_k=6,
            candidate_k=20,
            token_budget=1800,
            slots=task_field_slots(),
            required_roles=[role.value for role in EvidenceRole],
        )
        return self._filter_current_policy_documents(documents)

    def bootstrap_demo(self) -> EventDashboard:
        existing = self.list_events()
        if existing:
            return self.get_dashboard(existing[0].event_id)
        now = self._now()
        from .models import AlertInput

        dashboard = self.create_event(
            EventCreateRequest(
                title="碑林区下穿通道暴雨响应事件",
                area_id="beilin_10km2",
                alert=AlertInput(
                    alert_id="XA-BL-RAIN-20260711-001",
                    source_department="西安市气象部门",
                    disaster_type="暴雨",
                    level="橙色",
                    issued_at=now,
                    valid_until=now + timedelta(hours=6),
                    affected_area="碑林区重点低洼片区",
                    raw_content="预计未来六小时短时强降雨持续，请加强下穿通道和低洼点巡查。",
                ),
                operator_id="demo_duty_officer",
                operator_role=OperatorRole.DUTY_OFFICER,
            )
        )
        event_id = dashboard.event.event_id
        self.add_risk_objects(
            event_id,
            RiskObjectBatchRequest(
                objects=[
                    RiskObjectInput(
                        object_id="TUNNEL-017",
                        name="长安北路下穿通道",
                        object_type="下穿通道",
                        location="长安北路与友谊路交汇处",
                        responsible_organization="区住建局",
                        responsible_role="排水值班负责人",
                        trigger_reasons=["橙色暴雨预警覆盖", "历史积水点"],
                        source_refs=["XA-BL-RAIN-20260711-001", "风险对象台账-2026"],
                        vulnerability="低洼、车流量高，强降雨时积水增长快",
                        historical_risk="近三年有两次临时交通管控记录",
                        risk_score=87,
                        system_explanation="预警影响范围与对象位置重叠，且历史风险和脆弱性均较高；需人工核验。",
                    )
                ],
                operator_id="demo_duty_officer",
                operator_role=OperatorRole.DUTY_OFFICER,
            ),
        )
        self.verify_risk_object(
            event_id,
            "TUNNEL-017",
            RiskObjectVerificationRequest(
                decision=ObjectVerificationStatus.CONFIRMED,
                note="已通过对象台账和电话值守信息核验",
                operator_id="demo_reviewer",
                operator_role=OperatorRole.REVIEWER,
            ),
        )
        generated = self.generate_task_draft(
            event_id,
            "TUNNEL-017",
            TaskDraftGenerationRequest(
                operator_id="demo_duty_officer",
                operator_role=OperatorRole.DUTY_OFFICER,
                acknowledge_minutes=15,
                deadline_minutes=120,
            ),
        )
        if generated.task is None:
            raise ValueError(
                f"demo task evidence gate failed: {', '.join(generated.validation_errors)}"
            )
        task = generated.task
        self.submit_task(
            task.task_id,
            TaskActionRequest(
                operator_id="demo_duty_officer", operator_role=OperatorRole.DUTY_OFFICER
            ),
        )
        return self.get_dashboard(event_id)
