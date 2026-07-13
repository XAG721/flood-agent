from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from ..models import CorpusType, RAGDocument
from ..simulation_dataset import SimulatedExternalGateway, SimulatedGatewayTimeout
from .candidate_discovery import (
    build_registry_candidate,
    build_risk_object_candidate,
    calculate_profile_risk_score,
    calculate_registry_risk_score,
    point_in_polygon,
)
from .feedback_policy import (
    build_review_recommendations,
    classify_feedback,
    feedback_dedupe_key,
    missing_evidence_types,
    recommend_deadline_alternatives,
    recommend_feedback_alternatives,
    review_timeline_line,
)
from .state_machine import ensure_transition
from .risk_object_ingestion import parse_risk_object_file

from .models import (
    AlertAppendRequest,
    AlertLifecycleStatus,
    AlertSnapshot,
    AuditArchiveRecord,
    AuditArchiveRequest,
    AuditArchiveVerificationResult,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovalRequest,
    BackupCreateRequest,
    BackupImportRequest,
    BackupImportResult,
    BackupRestoreRequest,
    BackupRestoreResult,
    BackupRetentionRequest,
    BackupRetentionResult,
    CandidateDiscoveryRequest,
    CandidateDiscoveryResult,
    CandidateRunRecord,
    CallbackSequenceState,
    DatabaseBackupRecord,
    DeadlineExtensionDecisionRequest,
    DeadlineExtensionRecord,
    DeadlineExtensionRequest,
    DeadlineExtensionStatus,
    DispatchCallbackRecord,
    DispatchCallbackRequest,
    DispatchCallbackStatus,
    KeyRotationRequest,
    KeyRotationResult,
    LegacyAdapterCallRecord,
    LegacyMigrationRequest,
    DistrictScenarioReport,
    DocumentClause,
    DocumentImportRequest,
    DocumentVersionRecord,
    EscalationRecord,
    EventReviewDraft,
    EventCloseRequest,
    EventCreateRequest,
    EventDashboard,
    EventRiskObject,
    EventStatus,
    EvidenceRole,
    EvidenceFieldState,
    EvidenceConflict,
    EvidenceConflictResolutionRequest,
    EvidenceFreezeRequest,
    EvidenceManualSupplementRequest,
    EvidencePackageVersion,
    FeedbackCategory,
    FeedbackRequest,
    FeatureFlagSetting,
    FeatureFlagUpdateRequest,
    ObjectVerificationStatus,
    OperatorRole,
    OutboxMessage,
    OutboxProcessRequest,
    OutboxStatus,
    MigrationBatchRecord,
    MigrationQuarantineItem,
    ResponseEvent,
    ResponseTask,
    RiskObjectBatchRequest,
    RiskObjectImportQuarantineItem,
    RiskObjectInput,
    RiskObjectRegistryBatchImportRequest,
    RiskObjectRegistryFileImportRequest,
    RiskObjectRegistryImportResult,
    RiskObjectRegistryRecord,
    RiskObjectRegistryVersionSnapshot,
    RiskObjectVerificationRequest,
    RiskObjectVersionSnapshot,
    ReviewDraftRequest,
    RetrievalMode,
    RuleCheckResult,
    RuleEvaluationRecord,
    RuleOutcome,
    ScenarioAcceptanceCheck,
    ScenarioEvaluationRequest,
    ScenarioFailureCase,
    ScenarioMetricResult,
    TaskActionRequest,
    TaskAssignmentRecord,
    TaskAssignmentRequest,
    TaskCreateRequest,
    TaskFeedback,
    TaskDraftGenerationRequest,
    TaskDraftGenerationResult,
    TaskEvidenceRef,
    TaskStatus,
    TaskUpdateRequest,
    TaskVersionSnapshot,
    TimelineEntry,
    TimelineIntegrityIssue,
    TimelineIntegrityReport,
)


EDIT_ROLES = {
    OperatorRole.DUTY_OFFICER,
    OperatorRole.REVIEWER,
    OperatorRole.COMMANDER,
    OperatorRole.ADMIN,
}
APPROVAL_ROLES = {OperatorRole.REVIEWER, OperatorRole.COMMANDER}
EXECUTION_ROLES = {
    OperatorRole.LIAISON,
    OperatorRole.FIELD_OPERATOR,
    OperatorRole.DUTY_OFFICER,
    OperatorRole.REVIEWER,
    OperatorRole.COMMANDER,
}


class ResponseWorkflowService:
    """Deterministic district flood-response workflow; no model call is required."""

    GENERATION_VERSION = "response-rag-task-v1"
    WORKFLOW_ENGINE_VERSION = "response-workflow-v2"
    RULE_SET_VERSION = "response-rules-v2"

    def __init__(self, repository, rag_service=None, simulation_gateway=None) -> None:
        self.repository = repository
        self.rag_service = rag_service
        self.simulation_gateway = simulation_gateway or SimulatedExternalGateway()

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

    def add_risk_objects(
        self, event_id: str, request: RiskObjectBatchRequest
    ) -> list[EventRiskObject]:
        self._require_role(
            request.operator_role, EDIT_ROLES, "add candidate risk objects"
        )
        self._event(event_id, active=True)
        created: list[EventRiskObject] = []
        for payload in request.objects:
            change_type = "candidate_added"
            duplicate_source_id: str | None = None
            if payload.duplicate_of:
                canonical = self.repository.get_event_risk_object(
                    event_id, payload.duplicate_of
                )
                if canonical is None:
                    raise ValueError(
                        f"duplicate risk object references unknown canonical object: {payload.duplicate_of}"
                    )
                duplicate_source_id = payload.object_id
                merged = canonical.model_dump(include=set(RiskObjectInput.model_fields))
                merged.update(
                    {
                        "canonical_object_id": canonical.canonical_object_id
                        or canonical.object_id,
                        "aliases": sorted(
                            set(
                                canonical.aliases
                                + payload.aliases
                                + [payload.name, payload.object_id]
                            )
                        ),
                        "source_refs": list(
                            dict.fromkeys(canonical.source_refs + payload.source_refs)
                        ),
                        "duplicate_of": None,
                    }
                )
                payload = RiskObjectInput(**merged)
                change_type = "candidate_duplicate_merged"
            existing = self.repository.get_event_risk_object(
                event_id, payload.object_id
            )
            now = self._now()
            registry_expired = payload.registry_status != "active" or (
                payload.registry_valid_until is not None
                and payload.registry_valid_until <= now
            )
            item = EventRiskObject(
                **payload.model_copy(
                    update={
                        "canonical_object_id": payload.canonical_object_id
                        or payload.object_id
                    }
                ).model_dump(),
                event_id=event_id,
                verification_status=(
                    ObjectVerificationStatus.PENDING
                    if existing and existing.stale
                    else existing.verification_status
                    if existing
                    else ObjectVerificationStatus.PENDING
                ),
                verified_by=existing.verified_by if existing else None,
                verified_at=existing.verified_at if existing else None,
                verification_note=existing.verification_note if existing else "",
                created_at=existing.created_at if existing else now,
                version=(existing.version + 1) if existing else 1,
                updated_at=now,
                stale=registry_expired,
            )
            self.repository.save_event_risk_object(item)
            self._save_risk_object_version(
                item,
                change_type=change_type
                if duplicate_source_id
                else "candidate_updated"
                if existing
                else "candidate_added",
                operator_id=request.operator_id,
                terminal_id=request.terminal_id,
            )
            created.append(item)
            action = (
                change_type
                if duplicate_source_id
                else "candidate_updated"
                if existing
                else "candidate_added"
            )
            self._record(
                event_id=event_id,
                object_id=item.object_id,
                entry_type="risk_object",
                action=action,
                actor_id=request.operator_id,
                actor_role=request.operator_role.value,
                terminal_id=request.terminal_id,
                before_state={"risk_score": existing.risk_score} if existing else {},
                after_state={
                    "risk_score": item.risk_score,
                    "verification_status": item.verification_status.value,
                },
                detail={
                    "risk_score": item.risk_score,
                    "sources": item.source_refs,
                    "canonical_object_id": item.canonical_object_id,
                    "aliases": item.aliases,
                    "duplicate_source_id": duplicate_source_id,
                    "registry_status": item.registry_status,
                    "registry_valid_until": item.registry_valid_until.isoformat()
                    if item.registry_valid_until
                    else None,
                },
            )
        return created

    def list_risk_object_registry(
        self,
        *,
        area_id: str,
        operator_role: OperatorRole,
        include_inactive: bool = False,
    ) -> list[RiskObjectRegistryRecord]:
        records = self.repository.list_risk_object_registry(
            area_id=area_id,
            include_inactive=include_inactive,
        )
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
        result: list[RiskObjectRegistryRecord] = []
        for record in records:
            updates: dict[str, object] = {}
            if operator_role not in precise_location_roles:
                updates.update(
                    {
                        "location": "[精确位置已按岗位权限脱敏]",
                        "longitude": None,
                        "latitude": None,
                    }
                )
            if operator_role not in full_sensitive_roles:
                updates["sensitive_contacts"] = []
                if record.special_population_notes:
                    updates["special_population_notes"] = (
                        "[特殊人群信息已按岗位权限脱敏]"
                    )
            result.append(record.model_copy(update=updates))
        return result

    def list_risk_object_registry_versions(
        self,
        *,
        area_id: str,
        operator_role: OperatorRole,
        object_id: str | None = None,
    ) -> list[RiskObjectRegistryVersionSnapshot]:
        self._require_role(
            operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.ADMIN},
            "view risk-object master-data history",
        )
        return self.repository.list_risk_object_registry_versions(
            area_id=area_id,
            object_id=object_id,
        )

    def list_risk_object_imports(
        self, *, area_id: str, operator_role: OperatorRole
    ) -> list[RiskObjectRegistryImportResult]:
        self._require_role(
            operator_role,
            {
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.AUDITOR,
                OperatorRole.ADMIN,
            },
            "view risk-object import history",
        )
        return self.repository.list_risk_object_imports(area_id=area_id)

    def import_risk_object_registry(
        self, request: RiskObjectRegistryBatchImportRequest
    ) -> RiskObjectRegistryImportResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "import risk-object master data",
        )
        object_ids = [item.object_id for item in request.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(
                "one registry import cannot contain duplicate object_id values"
            )
        canonical = json.dumps(
            {
                "area_id": request.area_id,
                "source_version": request.source_version,
                "objects": [
                    item.model_dump(mode="json")
                    for item in sorted(request.objects, key=lambda item: item.object_id)
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self._commit_risk_object_registry_import(
            area_id=request.area_id,
            source_filename=request.source_filename,
            source_format="api_json",
            source_version=request.source_version,
            source_hash=hashlib.sha256(canonical).hexdigest(),
            source_bytes=len(canonical),
            objects=request.objects,
            quarantine=[],
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            terminal_id=request.terminal_id,
        )

    def import_risk_object_registry_file(
        self, request: RiskObjectRegistryFileImportRequest
    ) -> RiskObjectRegistryImportResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "import a risk-object master-data file",
        )
        parsed = parse_risk_object_file(
            request.file,
            source_version=request.source_version,
        )
        return self._commit_risk_object_registry_import(
            area_id=request.area_id,
            source_filename=request.file.filename,
            source_format=parsed.source_format,
            source_version=request.source_version,
            source_hash=parsed.source_hash,
            source_bytes=parsed.source_bytes,
            objects=parsed.objects,
            quarantine=parsed.quarantine,
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            terminal_id=request.terminal_id,
        )

    def _commit_risk_object_registry_import(
        self,
        *,
        area_id: str,
        source_filename: str,
        source_format: str,
        source_version: str,
        source_hash: str,
        source_bytes: int,
        objects: list[RiskObjectInput],
        quarantine: list[RiskObjectImportQuarantineItem],
        operator_id: str,
        operator_role: OperatorRole,
        terminal_id: str,
    ) -> RiskObjectRegistryImportResult:
        now = self._now()
        records: list[RiskObjectRegistryRecord] = []
        registry_snapshots: list[RiskObjectRegistryVersionSnapshot] = []
        changed_object_ids: list[str] = []
        created_count = 0
        updated_count = 0
        unchanged_count = 0
        existing_records = {
            item.object_id: item
            for item in self.repository.list_risk_object_registry(
                area_id=area_id,
                include_inactive=True,
            )
        }
        incoming_records = {item.object_id: item for item in objects}
        accepted_count = 0
        accepted_simulation_flags: list[bool] = []
        quarantine_items = list(quarantine)
        for item in objects:
            if (
                not item.duplicate_of
                and item.canonical_object_id
                and item.canonical_object_id != item.object_id
            ):
                quarantine_items.append(
                    RiskObjectImportQuarantineItem(
                        source_id=item.object_id,
                        reason_code="INVALID_CANONICAL_OBJECT",
                        reason=(
                            "a canonical record must use its own object_id as canonical_object_id"
                        ),
                    )
                )
                continue
            if item.duplicate_of:
                canonical = incoming_records.get(
                    item.duplicate_of
                ) or existing_records.get(item.duplicate_of)
                duplicate_error = ""
                if item.duplicate_of == item.object_id:
                    duplicate_error = "duplicate_of cannot reference the same object"
                elif canonical is None:
                    duplicate_error = "duplicate_of references an object outside the current area registry"
                elif canonical.duplicate_of:
                    duplicate_error = "duplicate_of must reference a canonical object, not another duplicate"
                elif (
                    item.canonical_object_id
                    and item.canonical_object_id != item.duplicate_of
                ):
                    duplicate_error = "canonical_object_id must equal duplicate_of for duplicate records"
                if duplicate_error:
                    quarantine_items.append(
                        RiskObjectImportQuarantineItem(
                            source_id=item.object_id,
                            reason_code="INVALID_CANONICAL_OBJECT",
                            reason=duplicate_error,
                        )
                    )
                    continue
            accepted_count += 1
            accepted_simulation_flags.append(item.is_simulated)
            existing = existing_records.get(item.object_id)
            item_payload = item.model_dump()
            item_payload.update(
                {
                    "canonical_object_id": item.canonical_object_id
                    or item.duplicate_of
                    or item.object_id,
                    "source_version": source_version,
                    "data_version": "pending-content-hash",
                }
            )
            preliminary = RiskObjectInput.model_validate(item_payload)
            content_payload = preliminary.model_dump(
                mode="json",
                exclude={"data_version"},
            )
            content_hash = hashlib.sha256(
                json.dumps(
                    content_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            normalized = preliminary.model_copy(
                update={
                    "data_version": f"{source_version}:{content_hash[:12]}",
                }
            )
            if existing is not None:
                if existing.content_hash == content_hash:
                    unchanged_count += 1
                    continue
            registry_version = existing.registry_version + 1 if existing else 1
            record = RiskObjectRegistryRecord(
                **normalized.model_dump(),
                area_id=area_id,
                registry_version=registry_version,
                source_hash=source_hash,
                content_hash=content_hash,
                source_filename=source_filename,
                created_by=existing.created_by if existing else operator_id,
                terminal_id=terminal_id,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            records.append(record)
            registry_snapshots.append(
                RiskObjectRegistryVersionSnapshot(
                    snapshot_id=self._id("REGVER"),
                    area_id=area_id,
                    object_id=record.object_id,
                    registry_version=record.registry_version,
                    record=record,
                    change_type="registry_updated" if existing else "registry_created",
                    changed_by=operator_id,
                    terminal_id=terminal_id,
                    created_at=now,
                )
            )
            changed_object_ids.append(record.object_id)
            if existing:
                updated_count += 1
            else:
                created_count += 1

        active_events = [
            event
            for event in self.repository.list_response_events()
            if event.area_id == area_id and event.status != EventStatus.CLOSED
        ]
        stale_runs: list[CandidateRunRecord] = []
        stale_event_objects: list[EventRiskObject] = []
        event_object_snapshots: list[RiskObjectVersionSnapshot] = []
        affected_event_ids: set[str] = set()
        changed_set = set(changed_object_ids)
        if changed_set:
            reason = (
                f"risk-object registry import {source_hash[:12]} changed master data"
            )
            for event in active_events:
                for run in self.repository.list_candidate_runs(event.event_id):
                    if run.status == "stale":
                        continue
                    stale_runs.append(
                        run.model_copy(
                            update={
                                "status": "stale",
                                "stale_at": now,
                                "stale_reason": reason,
                            }
                        )
                    )
                    affected_event_ids.add(event.event_id)
                for event_object in self.repository.list_event_risk_objects(
                    event.event_id
                ):
                    if event_object.object_id not in changed_set or event_object.stale:
                        continue
                    stale = event_object.model_copy(
                        update={
                            "stale": True,
                            "version": event_object.version + 1,
                            "updated_at": now,
                        }
                    )
                    stale_event_objects.append(stale)
                    event_object_snapshots.append(
                        RiskObjectVersionSnapshot(
                            snapshot_id=self._id("OBJVER"),
                            event_id=stale.event_id,
                            object_id=stale.object_id,
                            version=stale.version,
                            object=stale,
                            change_type="registry_change_marked_stale",
                            changed_by=operator_id,
                            terminal_id=terminal_id,
                            created_at=now,
                        )
                    )
                    affected_event_ids.add(event.event_id)

        result = RiskObjectRegistryImportResult(
            import_id=self._id("REGIMPORT"),
            area_id=area_id,
            source_filename=source_filename,
            source_format=source_format,
            source_version=source_version,
            source_hash=source_hash,
            source_bytes=source_bytes,
            imported_count=accepted_count,
            created_count=created_count,
            updated_count=updated_count,
            unchanged_count=unchanged_count,
            quarantined_count=len(quarantine_items),
            changed_object_ids=sorted(changed_object_ids),
            affected_event_ids=sorted(affected_event_ids),
            stale_candidate_run_count=len(stale_runs),
            quarantine=quarantine_items,
            is_simulated=bool(accepted_simulation_flags)
            and all(accepted_simulation_flags),
            imported_by=operator_id,
            terminal_id=terminal_id,
            created_at=now,
        )
        self.repository.commit_risk_object_registry_import(
            result=result,
            records=records,
            registry_snapshots=registry_snapshots,
            stale_candidate_runs=stale_runs,
            stale_event_objects=stale_event_objects,
            event_object_snapshots=event_object_snapshots,
        )
        for event_id in sorted(affected_event_ids):
            self._record(
                event_id=event_id,
                entry_type="risk_object_registry",
                action="risk_object_registry_imported",
                actor_id=operator_id,
                actor_role=operator_role.value,
                terminal_id=terminal_id,
                before_state={},
                after_state={"candidate_runs": "stale"},
                detail={
                    "import_id": result.import_id,
                    "source_hash": source_hash,
                    "changed_object_ids": result.changed_object_ids,
                    "quarantined_count": result.quarantined_count,
                },
            )
        return result

    def discover_risk_objects(
        self, event_id: str, request: CandidateDiscoveryRequest
    ) -> CandidateDiscoveryResult:
        self._require_role(
            request.operator_role,
            EDIT_ROLES,
            "discover candidate risk objects",
        )
        event = self._event(event_id, active=True)
        alerts = self.repository.list_alert_snapshots(event_id)
        if not alerts:
            raise ValueError("event has no alert snapshot")
        alert = alerts[-1]
        if alert.lifecycle_status in {
            AlertLifecycleStatus.REVOKED,
            AlertLifecycleStatus.EXPIRED,
        }:
            raise ValueError(
                "candidate discovery is disabled for revoked or expired warning revisions"
            )
        run_id = self._id("CANDRUN")
        now = self._now()
        registry_records = [
            item
            for item in self.repository.list_risk_object_registry(
                area_id=event.area_id,
                include_inactive=False,
            )
            if item.duplicate_of is None
            and (item.registry_valid_from is None or item.registry_valid_from <= now)
            and (item.registry_valid_until is None or item.registry_valid_until > now)
        ]
        requested_types = {
            item.strip().casefold() for item in request.entity_types if item.strip()
        }
        uses_core_registry = bool(registry_records)
        if uses_core_registry:
            profiles: list[object] = list(registry_records)
            if requested_types:
                profiles = [
                    item
                    for item in profiles
                    if isinstance(item, RiskObjectRegistryRecord)
                    and item.object_type.strip().casefold() in requested_types
                ]
            registry_fingerprint = [
                {
                    "object_id": item.object_id,
                    "registry_version": item.registry_version,
                    "content_hash": item.content_hash,
                }
                for item in registry_records
            ]
            risk_object_data_version = (
                "registry:"
                + hashlib.sha256(
                    json.dumps(
                        registry_fingerprint,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            algorithm_version = "candidate-registry-rules-v1"
            feature_version = "candidate-registry-features-v1"
        else:
            legacy_profiles = self.repository.list_v2_entity_profiles(
                area_id=event.area_id
            )
            if requested_types:
                legacy_profiles = [
                    item
                    for item in legacy_profiles
                    if item.entity_type.value.casefold() in requested_types
                ]
            profiles = list(legacy_profiles)
            risk_object_data_version = "legacy:v2-entity-profiles-v1"
            algorithm_version = "candidate-rules-v1"
            feature_version = "candidate-features-v1"
        scanned_profiles = len(profiles)
        spatially_evaluated = 0
        spatially_matched = 0
        excluded_unlocated = 0
        association_mode = "area_registry"
        limitations: list[str] = []
        if alert.affected_geometry is not None:
            association_mode = "gis_point_in_polygon"
            located_profiles = [
                item
                for item in profiles
                if getattr(item, "longitude", None) is not None
                and getattr(item, "latitude", None) is not None
            ]
            excluded_unlocated = len(profiles) - len(located_profiles)
            spatially_evaluated = len(located_profiles)
            profiles = [
                item
                for item in located_profiles
                if point_in_polygon(
                    float(getattr(item, "longitude")),
                    float(getattr(item, "latitude")),
                    alert.affected_geometry.coordinates,
                )
            ]
            spatially_matched = len(profiles)
            if excluded_unlocated:
                limitations.append(
                    f"{excluded_unlocated} 个对象缺少 EPSG:4326 坐标，未纳入本次精确空间筛查。"
                )
        else:
            limitations.append(
                "预警未提供 EPSG:4326 多边形，本次采用 area_id 区域关联，不代表精确 GIS 几何相交。"
            )
        scored: list[tuple[float, object]] = []
        for profile in profiles:
            if isinstance(profile, RiskObjectRegistryRecord):
                score = calculate_registry_risk_score(alert.level, profile)
            else:
                score = calculate_profile_risk_score(alert.level, profile)
            if score >= request.min_risk_score:
                scored.append((score, profile))
        scored.sort(
            key=lambda item: (
                -item[0],
                str(
                    getattr(
                        item[1],
                        "object_id",
                        getattr(item[1], "entity_id", ""),
                    )
                ),
            )
        )
        selected = scored[: request.max_candidates]
        objects: list[RiskObjectInput] = []
        for score, profile in selected:
            if isinstance(profile, RiskObjectRegistryRecord):
                objects.append(
                    build_registry_candidate(
                        profile,
                        score,
                        alert,
                        association_mode,
                    )
                )
            else:
                objects.append(
                    build_risk_object_candidate(
                        profile,
                        score,
                        alert,
                        association_mode,
                    )
                )
        candidates = (
            self.add_risk_objects(
                event_id,
                RiskObjectBatchRequest(
                    objects=objects,
                    operator_id=request.operator_id,
                    operator_role=request.operator_role,
                    terminal_id=request.terminal_id,
                ),
            )
            if objects
            else []
        )
        if candidates:
            candidates = [
                item.model_copy(update={"candidate_run_id": run_id})
                for item in candidates
            ]
            for item in candidates:
                self.repository.save_event_risk_object(item)
        self._record(
            event_id=event_id,
            entry_type="risk_object_discovery",
            action="candidate_discovery_completed",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "association_mode": association_mode,
                "candidate_run_id": run_id,
                "area_id": event.area_id,
                "registry_source": (
                    "response_risk_object_registry"
                    if uses_core_registry
                    else "legacy_v2_entity_profiles"
                ),
                "risk_object_data_version": risk_object_data_version,
                "alert_snapshot_id": alert.snapshot_id,
                "scanned_profiles": scanned_profiles,
                "spatially_evaluated_profiles": spatially_evaluated,
                "spatially_matched_profiles": spatially_matched,
                "excluded_unlocated_profiles": excluded_unlocated,
                "matched_profiles": len(candidates),
                "min_risk_score": request.min_risk_score,
            },
        )
        run = CandidateRunRecord(
            run_id=run_id,
            event_id=event_id,
            alert_snapshot_id=alert.snapshot_id,
            risk_object_data_version=risk_object_data_version,
            algorithm_version=algorithm_version,
            feature_version=feature_version,
            association_mode=association_mode,
            parameters={
                "entity_types": sorted(requested_types),
                "min_risk_score": request.min_risk_score,
                "max_candidates": request.max_candidates,
            },
            candidate_object_ids=[item.object_id for item in candidates],
            missing_features=["affected_geometry"]
            if alert.affected_geometry is None
            else [],
            limitations=limitations,
            created_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_candidate_run(run)
        return CandidateDiscoveryResult(
            run_id=run_id,
            event_id=event_id,
            alert_snapshot_id=alert.snapshot_id,
            association_mode=association_mode,
            scanned_profiles=scanned_profiles,
            matched_profiles=len(candidates),
            spatially_evaluated_profiles=spatially_evaluated,
            spatially_matched_profiles=spatially_matched,
            excluded_unlocated_profiles=excluded_unlocated,
            candidates=candidates,
            limitations=limitations,
            algorithm_version=algorithm_version,
            feature_version=feature_version,
            data_version=risk_object_data_version,
        )

    def verify_risk_object(
        self, event_id: str, object_id: str, request: RiskObjectVerificationRequest
    ) -> EventRiskObject:
        self._require_role(request.operator_role, EDIT_ROLES, "verify a risk object")
        self._event(event_id, active=True)
        if request.decision == ObjectVerificationStatus.PENDING:
            raise ValueError("verification decision must be confirmed or excluded")
        item = self.repository.get_event_risk_object(event_id, object_id)
        if item is None:
            raise LookupError(f"risk object not found: {object_id}")
        if item.registry_status != "active" or (
            item.registry_valid_until is not None
            and item.registry_valid_until <= self._now()
        ):
            raise ValueError(
                "expired or inactive risk-object registry records cannot be confirmed"
            )
        previous_status = item.verification_status
        item = item.model_copy(
            update={
                "verification_status": request.decision,
                "verified_by": request.operator_id,
                "verified_at": self._now(),
                "verification_note": request.note,
                "stale": False,
                "version": item.version + 1,
                "updated_at": self._now(),
            }
        )
        self.repository.save_event_risk_object(item)
        self._save_risk_object_version(
            item,
            change_type=f"candidate_{request.decision.value}",
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )
        self._record(
            event_id=event_id,
            object_id=object_id,
            entry_type="risk_object",
            action=f"candidate_{request.decision.value}",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state={"verification_status": previous_status.value},
            after_state={"verification_status": item.verification_status.value},
            detail={"note": request.note},
        )
        return item

    def create_task(self, event_id: str, request: TaskCreateRequest) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "create a task draft")
        self._event(event_id, active=True)
        if request.idempotency_key:
            existing_task = next(
                (
                    item
                    for item in self.repository.list_response_tasks(event_id)
                    if item.creation_idempotency_key == request.idempotency_key
                ),
                None,
            )
            if existing_task is not None:
                return existing_task
        risk_object = self.repository.get_event_risk_object(event_id, request.object_id)
        if (
            risk_object is None
            or risk_object.verification_status != ObjectVerificationStatus.CONFIRMED
            or risk_object.stale
        ):
            raise ValueError(
                "tasks may only be created for confirmed event risk objects"
            )
        now = self._now()
        task = ResponseTask(
            **request.model_dump(
                exclude={
                    "operator_id",
                    "operator_role",
                    "terminal_id",
                    "idempotency_key",
                }
            ),
            task_id=self._id("TASK"),
            event_id=event_id,
            drafted_by=request.operator_id,
            created_at=now,
            updated_at=now,
            creation_idempotency_key=request.idempotency_key,
        )
        self.repository.save_response_task(task)
        self._save_task_version_snapshot(
            task,
            change_type="created",
            changed_fields=list(TaskCreateRequest.model_fields),
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
            note="任务草案创建",
        )
        self._record(
            event_id=event_id,
            task_id=task.task_id,
            object_id=task.object_id,
            entry_type="task",
            action="task_draft_created",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            after_state={"status": task.status.value, "version": task.version},
            detail={"generated_by_ai": task.generated_by_ai, "version": task.version},
        )
        return task

    def generate_task_draft(
        self,
        event_id: str,
        object_id: str,
        request: TaskDraftGenerationRequest,
    ) -> TaskDraftGenerationResult:
        self._require_role(
            request.operator_role, EDIT_ROLES, "generate a grounded task draft"
        )
        self._event(event_id, active=True)
        risk_object = self.repository.get_event_risk_object(event_id, object_id)
        if risk_object is None:
            raise LookupError(f"risk object not found: {object_id}")
        if risk_object.verification_status != ObjectVerificationStatus.CONFIRMED:
            raise ValueError(
                "task drafts may only be generated for confirmed event risk objects"
            )
        if risk_object.stale:
            raise ValueError(
                "task drafts cannot use a stale risk object; rerun discovery and verification"
            )

        alerts = self.repository.list_alert_snapshots(event_id)
        if not alerts:
            raise ValueError("event has no alert snapshot")
        alert = alerts[-1]
        query = request.query.strip() or self._build_task_query(alert, risk_object)
        if request.retrieval_mode in {
            RetrievalMode.CANARY,
            RetrievalMode.DEFAULT,
        } and not self.is_feature_enabled(
            "feature.frc_rag_formal_enabled",
            event_id=event_id,
            role=request.operator_role,
        ):
            raise ValueError(
                "FRC-RAG formal retrieval is gated off; use SHADOW/BASELINE_ONLY/REVIEW until Gate 2 is approved"
            )
        policy_documents, retrieval_trace = self._run_retrieval_strategy(
            query, request.retrieval_mode
        )
        evidence = self._build_task_evidence(alert, risk_object, policy_documents)
        conflicts = self._detect_evidence_conflicts(evidence)
        covered_roles = {role.value for item in evidence for role in item.roles}
        role_coverage = {
            role.value: role.value in covered_roles for role in EvidenceRole
        }
        missing_roles = [role for role, covered in role_coverage.items() if not covered]
        plan_basis = self._build_plan_basis(policy_documents)
        action = self._extract_supported_action(policy_documents, risk_object)
        validation_errors = [f"缺少 {role} 证据" for role in missing_roles]
        if not plan_basis:
            validation_errors.append("缺少带版本和条款号的预案引用")
        if not action:
            validation_errors.append("检索证据不足以形成可执行处置动作")
        if conflicts:
            validation_errors.extend(
                f"存在未决{item.conflict_type}冲突：{item.field_name}"
                for item in conflicts
            )

        grounding_summary = self._build_grounding_summary(
            evidence, role_coverage, validation_errors
        )
        evidence_package = self._build_evidence_package(
            event_id=event_id,
            object_id=object_id,
            evidence=evidence,
            role_coverage=role_coverage,
            validation_errors=validation_errors,
            conflicts=conflicts,
            retrieval_mode=request.retrieval_mode,
            retrieval_trace=retrieval_trace,
            action=action,
            plan_basis=plan_basis,
            operator_id=request.operator_id,
        )
        self.repository.save_evidence_package(evidence_package)
        if validation_errors:
            self._record(
                event_id=event_id,
                object_id=object_id,
                entry_type="task_draft",
                action="task_draft_blocked_by_evidence_gate",
                actor_id=request.operator_id,
                actor_role=request.operator_role.value,
                terminal_id=request.terminal_id,
                detail={
                    "missing_roles": missing_roles,
                    "validation_errors": validation_errors,
                    "evidence_source_ids": [item.source_id for item in evidence],
                    "generation_version": self.GENERATION_VERSION,
                },
            )
            return TaskDraftGenerationResult(
                event_id=event_id,
                object_id=object_id,
                status="insufficient_evidence",
                evidence=evidence,
                role_coverage=role_coverage,
                missing_roles=missing_roles,
                validation_errors=validation_errors,
                grounding_summary=grounding_summary,
                generation_version=self.GENERATION_VERSION,
                evidence_package=evidence_package,
            )

        now = self._now()
        task = self.create_task(
            event_id,
            TaskCreateRequest(
                object_id=object_id,
                title=f"{risk_object.name}现场核查与响应处置",
                action=action,
                responsible_organization=risk_object.responsible_organization,
                responsible_role=risk_object.responsible_role,
                cooperate_roles=self._cooperate_roles(risk_object.object_type),
                deadline_at=now + timedelta(minutes=request.deadline_minutes),
                acknowledge_deadline_at=now
                + timedelta(minutes=request.acknowledge_minutes),
                required_evidence=self._required_evidence(risk_object.object_type),
                plan_basis=plan_basis,
                approval_policy=self._approval_policy(alert.level, action),
                escalation_rule="未确认时提醒成员单位联络员；超时或受阻时升级至防办审核员并记录异常单。",
                operator_id=request.operator_id,
                operator_role=request.operator_role,
                terminal_id=request.terminal_id,
                generated_by_ai=True,
                generation_version=self.GENERATION_VERSION,
                grounding_summary=grounding_summary,
                evidence_role_coverage=role_coverage,
                source_evidence=evidence,
                evidence_package_id=evidence_package.package_id,
                evidence_package_version=evidence_package.version,
                evidence_package_hash=evidence_package.content_hash,
                idempotency_key=f"draft:{event_id}:{object_id}:{evidence_package.content_hash}",
            ),
        )
        self._record(
            event_id=event_id,
            task_id=task.task_id,
            object_id=object_id,
            entry_type="task_draft",
            action="grounded_task_draft_generated",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            after_state={"status": task.status.value, "version": task.version},
            detail={
                "evidence_source_ids": [item.source_id for item in evidence],
                "role_coverage": role_coverage,
                "generation_version": self.GENERATION_VERSION,
            },
        )
        return TaskDraftGenerationResult(
            event_id=event_id,
            object_id=object_id,
            status="ready",
            task=task,
            evidence=evidence,
            role_coverage=role_coverage,
            grounding_summary=grounding_summary,
            generation_version=self.GENERATION_VERSION,
            evidence_package=evidence_package,
        )

    def list_candidate_runs(self, event_id: str) -> list[CandidateRunRecord]:
        self._event(event_id)
        return self.repository.list_candidate_runs(event_id)

    def list_evidence_packages(
        self, event_id: str, *, object_id: str | None = None
    ) -> list[EvidencePackageVersion]:
        self._event(event_id)
        return self.repository.list_evidence_packages(event_id, object_id=object_id)

    def get_evidence_package(self, package_id: str) -> EvidencePackageVersion:
        package = self.repository.get_latest_evidence_package(package_id)
        if package is None:
            raise LookupError(f"evidence package not found: {package_id}")
        return package

    def list_task_transitions(self, task_id: str) -> list[TimelineEntry]:
        task = self._task(task_id)
        return [
            item
            for item in self.repository.list_timeline_entries(task.event_id)
            if item.task_id == task_id
        ]

    def replay_event(self, event_id: str, request: TaskActionRequest) -> dict:
        self._require_role(
            request.operator_role,
            {
                OperatorRole.AUDITOR,
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.ADMIN,
            },
            "replay an event",
        )
        integrity = self.verify_timeline_integrity(event_id, request.operator_role)
        if integrity.status != "verified":
            raise ValueError("event timeline integrity must verify before replay")
        return {
            "mode": "read_only_replay",
            "integrity": integrity,
            "dashboard": self.get_dashboard(event_id),
        }

    def _build_evidence_package(
        self,
        *,
        event_id: str,
        object_id: str,
        evidence: list[TaskEvidenceRef],
        role_coverage: dict[str, bool],
        validation_errors: list[str],
        conflicts: list[EvidenceConflict],
        retrieval_mode: RetrievalMode = RetrievalMode.SHADOW,
        retrieval_trace: dict | None = None,
        action: str,
        plan_basis: list,
        operator_id: str,
    ) -> EvidencePackageVersion:
        supported = EvidenceFieldState.SUPPORTED
        missing = EvidenceFieldState.MISSING
        field_states = {
            "action": supported if action else missing,
            "responsible_party": supported
            if role_coverage.get(EvidenceRole.RESPONSIBILITY.value)
            else missing,
            "trigger_condition": supported
            if role_coverage.get(EvidenceRole.CONDITION.value)
            else missing,
            "procedure": supported
            if role_coverage.get(EvidenceRole.PROCEDURE.value)
            else missing,
            "exception": supported
            if role_coverage.get(EvidenceRole.EXCEPTION.value)
            else missing,
            "plan_basis": supported if plan_basis else missing,
        }
        for conflict in conflicts:
            field_states[conflict.field_name] = EvidenceFieldState.CONFLICTED
        canonical = json.dumps(
            {
                "event_id": event_id,
                "object_id": object_id,
                "field_states": {
                    key: value.value for key, value in field_states.items()
                },
                "role_coverage": role_coverage,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "conflicts": [item.model_dump(mode="json") for item in conflicts],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = self._now()
        complete = not validation_errors
        return EvidencePackageVersion(
            package_id=self._id("EVID"),
            event_id=event_id,
            object_id=object_id,
            status="frozen" if complete else "needs_review",
            retrieval_run_id=self._id("RETRIEVAL"),
            field_states=field_states,
            role_coverage=role_coverage,
            evidence=evidence,
            conflicts=conflicts,
            missing_fields=[
                key for key, value in field_states.items() if value == missing
            ],
            content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            created_by=operator_id,
            reviewed_by=operator_id if complete else None,
            frozen_at=now if complete else None,
            created_at=now,
            retrieval_mode=retrieval_mode,
            baseline_source_ids=list(
                (retrieval_trace or {}).get("baseline_source_ids", [])
            ),
            frc_source_ids=list((retrieval_trace or {}).get("frc_source_ids", [])),
            shadow_comparison=dict((retrieval_trace or {}).get("comparison", {})),
        )

    def resolve_evidence_conflict(
        self,
        package_id: str,
        conflict_id: str,
        request: EvidenceConflictResolutionRequest,
    ) -> EvidencePackageVersion:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "resolve an evidence conflict",
        )
        current = self.repository.get_latest_evidence_package(package_id)
        if current is None:
            raise LookupError(f"evidence package not found: {package_id}")
        target = next(
            (item for item in current.conflicts if item.conflict_id == conflict_id),
            None,
        )
        if target is None:
            raise LookupError(f"evidence conflict not found: {conflict_id}")
        if not set(request.selected_source_ids).issubset(
            set(target.evidence_source_ids)
        ):
            raise ValueError(
                "selected evidence sources must belong to the conflict group"
            )
        conflicts = [
            item.model_copy(
                update={
                    "resolution_status": "resolved",
                    "resolution_reason": request.reason,
                    "selected_source_ids": request.selected_source_ids,
                }
            )
            if item.conflict_id == conflict_id
            else item
            for item in current.conflicts
        ]
        unresolved = [
            item for item in conflicts if item.resolution_status != "resolved"
        ]
        field_states = dict(current.field_states)
        if not any(item.field_name == target.field_name for item in unresolved):
            field_states[target.field_name] = EvidenceFieldState.SUPPORTED
        canonical = json.dumps(
            {
                "previous_hash": current.content_hash,
                "field_states": {
                    key: value.value for key, value in field_states.items()
                },
                "conflicts": [item.model_dump(mode="json") for item in conflicts],
                "evidence": [item.model_dump(mode="json") for item in current.evidence],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = self._now()
        frozen = not unresolved and not current.missing_fields
        updated = current.model_copy(
            update={
                "version": current.version + 1,
                "status": "frozen" if frozen else "needs_review",
                "field_states": field_states,
                "conflicts": conflicts,
                "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "reviewed_by": request.operator_id,
                "frozen_at": now if frozen else None,
                "created_at": now,
            }
        )
        self.repository.save_evidence_package(updated)
        self._record(
            event_id=current.event_id,
            object_id=current.object_id,
            entry_type="evidence_package",
            action="evidence_conflict_resolved",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "package_id": package_id,
                "version": updated.version,
                "conflict_id": conflict_id,
                "selected_source_ids": request.selected_source_ids,
                "reason": request.reason,
            },
        )
        return updated

    def supplement_evidence_package(
        self, package_id: str, request: EvidenceManualSupplementRequest
    ) -> EvidencePackageVersion:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.ADMIN},
            "supplement an evidence package",
        )
        current = self.get_evidence_package(package_id)
        if current.status == "frozen":
            raise ValueError(
                "frozen evidence packages are immutable; generate a new retrieval run"
            )
        by_source = {item.source_id: item for item in current.evidence}
        for item in request.evidence:
            by_source[item.source_id] = item
        return self._revise_evidence_package(
            current,
            evidence=list(by_source.values()),
            actor_id=request.operator_id,
            actor_role=request.operator_role,
            terminal_id=request.terminal_id,
            reason=request.reason,
            freeze_when_complete=request.freeze_when_complete,
            action="manual_evidence_supplemented",
        )

    def freeze_evidence_package(
        self, package_id: str, request: EvidenceFreezeRequest
    ) -> EvidencePackageVersion:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.ADMIN},
            "freeze an evidence package",
        )
        current = self.get_evidence_package(package_id)
        if current.status == "frozen":
            return current
        return self._revise_evidence_package(
            current,
            evidence=current.evidence,
            actor_id=request.operator_id,
            actor_role=request.operator_role,
            terminal_id=request.terminal_id,
            reason=request.reason,
            freeze_when_complete=True,
            action="evidence_package_frozen",
        )

    def _revise_evidence_package(
        self,
        current: EvidencePackageVersion,
        *,
        evidence: list[TaskEvidenceRef],
        actor_id: str,
        actor_role: OperatorRole,
        terminal_id: str,
        reason: str,
        freeze_when_complete: bool,
        action: str,
    ) -> EvidencePackageVersion:
        covered_roles = {role.value for item in evidence for role in item.roles}
        role_coverage = {
            role.value: role.value in covered_roles for role in EvidenceRole
        }
        field_states = dict(current.field_states)
        role_fields = {
            EvidenceRole.CONDITION: ("trigger_condition",),
            EvidenceRole.RESPONSIBILITY: ("responsible_party",),
            EvidenceRole.PROCEDURE: ("procedure", "action"),
            EvidenceRole.EXCEPTION: ("exception",),
            EvidenceRole.ATTRIBUTION: ("plan_basis",),
        }
        for role, fields in role_fields.items():
            if role.value in covered_roles:
                for field in fields:
                    if field_states.get(field) != EvidenceFieldState.CONFLICTED:
                        field_states[field] = EvidenceFieldState.SUPPORTED
        unresolved = [
            item for item in current.conflicts if item.resolution_status != "resolved"
        ]
        missing_fields = [
            field
            for field, state in field_states.items()
            if state == EvidenceFieldState.MISSING
        ]
        complete = (
            not missing_fields
            and not unresolved
            and all(
                state == EvidenceFieldState.SUPPORTED for state in field_states.values()
            )
        )
        if freeze_when_complete and not complete:
            raise ValueError(
                "evidence package cannot be frozen while fields are missing or conflicts are unresolved"
            )
        canonical = json.dumps(
            {
                "previous_hash": current.content_hash,
                "field_states": {
                    key: value.value for key, value in field_states.items()
                },
                "role_coverage": role_coverage,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "conflicts": [
                    item.model_dump(mode="json") for item in current.conflicts
                ],
                "reason": reason,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = self._now()
        frozen = freeze_when_complete and complete
        updated = current.model_copy(
            update={
                "version": current.version + 1,
                "status": "frozen" if frozen else "needs_review",
                "field_states": field_states,
                "role_coverage": role_coverage,
                "evidence": evidence,
                "missing_fields": missing_fields,
                "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "reviewed_by": actor_id,
                "frozen_at": now if frozen else None,
                "created_at": now,
            }
        )
        self.repository.save_evidence_package(updated)
        self._record(
            event_id=current.event_id,
            object_id=current.object_id,
            entry_type="evidence_package",
            action=action,
            actor_id=actor_id,
            actor_role=actor_role.value,
            terminal_id=terminal_id,
            detail={
                "package_id": current.package_id,
                "version": updated.version,
                "reason": reason,
            },
        )
        return updated

    def update_task(self, task_id: str, request: TaskUpdateRequest) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "edit a task draft")
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status != TaskStatus.DRAFT:
            raise ValueError(
                "only draft tasks can be edited; return the task to draft first"
            )
        self._assert_expected_version(task, request.expected_version)
        changes = request.model_dump(
            exclude={
                "operator_id",
                "operator_role",
                "note",
                "terminal_id",
                "expected_version",
            },
            exclude_none=True,
        )
        if not changes:
            return task
        before_state = {field: getattr(task, field) for field in changes}
        changes.update({"version": task.version + 1, "updated_at": self._now()})
        task = task.model_copy(update=changes)
        self.repository.save_response_task(task)
        self._save_task_version_snapshot(
            task,
            change_type="revised",
            changed_fields=sorted(
                field for field in changes if field not in {"updated_at", "version"}
            ),
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
            note=request.note,
        )
        self._record(
            event_id=task.event_id,
            task_id=task.task_id,
            object_id=task.object_id,
            entry_type="task",
            action="task_draft_revised",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state=before_state,
            after_state={
                field: getattr(task, field)
                for field in changes
                if field not in {"updated_at", "version"}
            },
            detail={
                "version": task.version,
                "changed_fields": sorted(changes),
                "note": request.note,
            },
        )
        return task

    def submit_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "submit a task")
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status != TaskStatus.DRAFT:
            raise ValueError("only draft tasks can be submitted")
        self._assert_expected_version(task, request.expected_version)
        known_ids = {
            item.task_id for item in self.repository.list_response_tasks(task.event_id)
        }
        missing = [item for item in task.dependencies if item not in known_ids]
        if missing:
            raise ValueError(f"unknown task dependencies: {', '.join(missing)}")
        evaluation = self._evaluate_task_rules(task, request)
        if evaluation.overall_outcome == RuleOutcome.HARD_BLOCK:
            raise ValueError(
                "task is blocked by rules: "
                + "; ".join(
                    item.message
                    for item in evaluation.checks
                    if item.outcome == RuleOutcome.HARD_BLOCK
                )
            )
        warnings = [
            item.message
            for item in evaluation.checks
            if item.outcome == RuleOutcome.SOFT_WARNING
        ]
        ensure_transition(task.status, TaskStatus.PENDING_APPROVAL)
        task = task.model_copy(
            update={
                "status": TaskStatus.PENDING_APPROVAL,
                "last_rule_evaluation_id": evaluation.evaluation_id,
                "validation_warnings": warnings,
                "updated_at": self._now(),
            }
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_submitted",
            request,
            {"version": task.version, "previous_status": TaskStatus.DRAFT.value},
        )
        return task

    def decide_task(self, task_id: str, request: ApprovalRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, APPROVAL_ROLES, "approve or reject a task"
        )
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status != TaskStatus.PENDING_APPROVAL:
            raise ValueError("task is not pending approval")
        self._assert_expected_version(task, request.expected_version)
        if (
            task.approval_policy == ApprovalPolicy.COMMANDER_REQUIRED
            and request.operator_role != OperatorRole.COMMANDER
        ):
            raise PermissionError("this high-risk task requires commander approval")
        if (
            task.approval_policy == ApprovalPolicy.COMMANDER_REQUIRED
            and task.drafted_by == request.operator_id
        ):
            raise PermissionError("the drafter cannot approve their own high-risk task")
        evaluation = self._evaluate_task_rules(task, request)
        if (
            request.decision == ApprovalDecision.APPROVED
            and evaluation.overall_outcome == RuleOutcome.HARD_BLOCK
        ):
            raise ValueError("task approval is blocked by the current rule evaluation")
        task_payload_hash = self._task_payload_hash(task)
        evidence_package_hash = self._evidence_package_hash(task)
        record = ApprovalRecord(
            approval_id=self._id("APR"),
            event_id=task.event_id,
            task_id=task.task_id,
            task_version=task.version,
            decision=request.decision,
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            note=request.note,
            created_at=self._now(),
            task_payload_hash=task_payload_hash,
            evidence_package_hash=evidence_package_hash,
            rule_set_version=self.RULE_SET_VERSION,
            rule_evaluation_id=evaluation.evaluation_id,
        )
        previous_status = task.status
        outbox = None
        if request.decision == ApprovalDecision.APPROVED:
            ensure_transition(task.status, TaskStatus.ISSUED)
            outbox = self._build_dispatch_outbox(task, record)
            task = task.model_copy(
                update={
                    "status": TaskStatus.ISSUED,
                    "approved_version": task.version,
                    "approval_payload_hash": task_payload_hash,
                    "evidence_package_hash": evidence_package_hash,
                    "rule_set_version": self.RULE_SET_VERSION,
                    "dispatch_message_id": outbox.message_id,
                    "updated_at": self._now(),
                }
            )
            action = "task_approved_and_issued"
        else:
            ensure_transition(task.status, TaskStatus.DRAFT)
            task = task.model_copy(
                update={
                    "status": TaskStatus.DRAFT,
                    "version": task.version + 1,
                    "updated_at": self._now(),
                }
            )
            action = "task_rejected_to_draft"
        self.repository.commit_task_decision(record, task, outbox)
        if outbox is not None and self.is_feature_enabled(
            "feature.simulated_dispatch", event_id=task.event_id
        ):
            outbox = self._dispatch_outbox_message(outbox)
        if request.decision == ApprovalDecision.REJECTED:
            self._save_task_version_snapshot(
                task,
                change_type="approval_rejected",
                changed_fields=["status", "version"],
                operator_id=request.operator_id,
                terminal_id=request.terminal_id,
                note=request.note,
            )
        self._task_event(
            task,
            action,
            request,
            {
                "approval_id": record.approval_id,
                "task_payload_hash": record.task_payload_hash,
                "evidence_package_hash": record.evidence_package_hash,
                "dispatch_message_id": task.dispatch_message_id,
                "note": request.note,
                "previous_status": previous_status.value,
            },
        )
        return task

    def acknowledge_task(
        self, task_id: str, request: TaskActionRequest
    ) -> ResponseTask:
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if (
            task.approval_payload_hash
            and self._task_payload_hash(task) != task.approval_payload_hash
        ):
            raise ValueError(
                "approved task payload hash does not match the dispatch payload"
            )
        return self._transition(
            task_id,
            request,
            {TaskStatus.ISSUED, TaskStatus.ESCALATED, TaskStatus.BLOCKED},
            TaskStatus.ACKNOWLEDGED,
            "task_acknowledged",
        )

    @staticmethod
    def _assert_expected_version(
        task: ResponseTask, expected_version: int | None
    ) -> None:
        if expected_version is not None and expected_version != task.version:
            raise ValueError(
                f"task version conflict: expected {expected_version}, current {task.version}"
            )

    def _evaluate_task_rules(
        self, task: ResponseTask, request: TaskActionRequest
    ) -> RuleEvaluationRecord:
        risk_object = self.repository.get_event_risk_object(
            task.event_id, task.object_id
        )
        packages = self.repository.list_evidence_packages(
            task.event_id, object_id=task.object_id
        )
        package = next(
            (
                item
                for item in reversed(packages)
                if item.package_id == task.evidence_package_id
            ),
            None,
        )
        checks: list[RuleCheckResult] = []

        def add(
            rule_id: str,
            outcome: RuleOutcome,
            message: str,
            field_name: str | None = None,
        ):
            checks.append(
                RuleCheckResult(
                    rule_id=rule_id,
                    outcome=outcome,
                    message=message,
                    field_name=field_name,
                )
            )

        add(
            "RULE-OBJECT-CONFIRMED",
            RuleOutcome.PASS
            if risk_object
            and risk_object.verification_status == ObjectVerificationStatus.CONFIRMED
            else RuleOutcome.HARD_BLOCK,
            "风险对象已人工确认"
            if risk_object
            and risk_object.verification_status == ObjectVerificationStatus.CONFIRMED
            else "风险对象未人工确认",
            "object_id",
        )
        deadline_valid = (
            task.acknowledge_deadline_at
            <= (task.start_deadline_at or task.deadline_at)
            <= task.deadline_at
            <= (task.verification_deadline_at or task.deadline_at)
        )
        add(
            "RULE-DEADLINES-ORDERED",
            RuleOutcome.PASS if deadline_valid else RuleOutcome.HARD_BLOCK,
            "四类时限顺序有效"
            if deadline_valid
            else "接收、开始、完成和核验时限顺序无效",
            "deadline_at",
        )
        required_complete = bool(
            task.title.strip()
            and task.action.strip()
            and task.required_evidence
            and task.plan_basis
        )
        add(
            "RULE-REQUIRED-FIELDS",
            RuleOutcome.PASS if required_complete else RuleOutcome.HARD_BLOCK,
            "任务必填字段完整"
            if required_complete
            else "任务标题、动作、证据要求或预案依据缺失",
        )
        if task.generated_by_ai:
            package_ok = bool(
                package
                and package.status == "frozen"
                and not package.conflicts
                and not package.missing_fields
                and package.content_hash == task.evidence_package_hash
            )
            add(
                "RULE-EVIDENCE-PACKAGE-FROZEN",
                RuleOutcome.PASS if package_ok else RuleOutcome.HARD_BLOCK,
                "冻结证据包与草案哈希一致"
                if package_ok
                else "AI 草案缺少冻结证据包、存在未决冲突/缺失或哈希不一致",
                "evidence_package_id",
            )
            bound = bool(task.source_evidence) and all(
                item.source_id and item.roles for item in task.source_evidence
            )
            add(
                "RULE-EVIDENCE-BOUND",
                RuleOutcome.PASS if bound else RuleOutcome.HARD_BLOCK,
                "关键字段已绑定来源证据" if bound else "关键字段未绑定可定位来源证据",
                "source_evidence",
            )
        if task.task_id in task.dependencies:
            add(
                "RULE-NO-SELF-DEPENDENCY",
                RuleOutcome.HARD_BLOCK,
                "任务不能依赖自身",
                "dependencies",
            )
        else:
            add(
                "RULE-NO-SELF-DEPENDENCY",
                RuleOutcome.PASS,
                "任务依赖不包含自身",
                "dependencies",
            )
        add(
            "RULE-COOPERATION-DEFINED",
            RuleOutcome.PASS if task.cooperate_roles else RuleOutcome.SOFT_WARNING,
            "已定义协同岗位"
            if task.cooperate_roles
            else "未定义协同岗位，请审批人员确认是否为单岗位任务",
            "cooperate_roles",
        )
        overall = (
            RuleOutcome.HARD_BLOCK
            if any(item.outcome == RuleOutcome.HARD_BLOCK for item in checks)
            else RuleOutcome.SOFT_WARNING
            if any(item.outcome == RuleOutcome.SOFT_WARNING for item in checks)
            else RuleOutcome.PASS
        )
        record = RuleEvaluationRecord(
            evaluation_id=self._id("RULEEVAL"),
            event_id=task.event_id,
            task_id=task.task_id,
            task_version=task.version,
            rule_set_version=self.RULE_SET_VERSION,
            overall_outcome=overall,
            checks=checks,
            evaluated_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_rule_evaluation(record)
        self._task_event(
            task,
            "task_rules_evaluated",
            request,
            {
                "evaluation_id": record.evaluation_id,
                "outcome": overall.value,
                "hard_blocks": [
                    item.rule_id
                    for item in checks
                    if item.outcome == RuleOutcome.HARD_BLOCK
                ],
                "soft_warnings": [
                    item.rule_id
                    for item in checks
                    if item.outcome == RuleOutcome.SOFT_WARNING
                ],
            },
        )
        return record

    def list_feature_flags(self) -> list[FeatureFlagSetting]:
        return self.repository.list_feature_flags()

    def update_feature_flag(
        self, request: FeatureFlagUpdateRequest
    ) -> FeatureFlagSetting:
        self._require_role(
            request.operator_role, {OperatorRole.ADMIN}, "change a feature flag"
        )
        if not request.flag_key.startswith("feature."):
            raise ValueError("feature flag keys must start with 'feature.'")
        matching = [
            item
            for item in self.repository.list_feature_flags()
            if item.flag_key == request.flag_key
            and item.environment == request.environment
            and item.event_id == request.event_id
            and item.role == request.role
            and item.scenario == request.scenario
        ]
        setting = FeatureFlagSetting(
            flag_key=request.flag_key,
            enabled=request.enabled,
            environment=request.environment,
            event_id=request.event_id,
            role=request.role,
            scenario=request.scenario,
            version=(matching[0].version + 1) if matching else 1,
            reason=request.reason,
            updated_by=request.operator_id,
            terminal_id=request.terminal_id,
            updated_at=self._now(),
        )
        self.repository.save_feature_flag(setting)
        if request.event_id:
            self._record(
                event_id=request.event_id,
                entry_type="configuration",
                action="feature_flag_changed",
                actor_id=request.operator_id,
                actor_role=request.operator_role.value,
                terminal_id=request.terminal_id,
                after_state={
                    "flag_key": request.flag_key,
                    "enabled": request.enabled,
                    "version": setting.version,
                },
                detail={"reason": request.reason, "environment": request.environment},
            )
        return setting

    def is_feature_enabled(
        self,
        flag_key: str,
        *,
        environment: str | None = None,
        event_id: str | None = None,
        role: OperatorRole | None = None,
        scenario: str | None = None,
    ) -> bool:
        environment = environment or os.getenv("FLOOD_ENVIRONMENT", "development")
        candidates = [
            item
            for item in self.repository.list_feature_flags()
            if item.flag_key == flag_key
            and item.environment == environment
            and item.event_id in {None, event_id}
            and item.role in {None, role}
            and item.scenario in {None, scenario}
        ]
        if candidates:
            candidates.sort(
                key=lambda item: (
                    int(item.event_id is not None)
                    + int(item.role is not None)
                    + int(item.scenario is not None),
                    item.version,
                ),
                reverse=True,
            )
            return candidates[0].enabled
        return environment != "production" and flag_key in {
            "feature.new_warning_model",
            "feature.object_candidate_service",
            "feature.structured_task_draft",
            "feature.rule_engine",
            "feature.new_approval_flow",
            "feature.task_state_machine",
            "feature.simulated_dispatch",
            "feature.audit_hash_chain",
        }

    def list_outbox_messages(
        self, *, event_id: str | None = None
    ) -> list[OutboxMessage]:
        return self.repository.list_outbox_messages(event_id=event_id)

    def migration_status(self) -> dict:
        return {
            "schema_migrations": self.repository.list_schema_migrations(),
            "migration_batches": self.repository.list_migration_batches(),
            "legacy_adapter_calls": len(self.repository.list_legacy_adapter_calls()),
            "core_write_authority": "/response",
            "legacy_write_policy": "denied",
        }

    def list_document_versions(
        self, document_id: str | None = None
    ) -> list[DocumentVersionRecord]:
        return self.repository.list_document_versions(document_id)

    def task_schema_contract(self) -> dict:
        schema = ResponseTask.model_json_schema()
        canonical = json.dumps(
            schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return {
            "schema_version": "response-task-schema-v1",
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "schema": schema,
        }

    def rule_set_contract(self) -> dict:
        return {
            "rule_set_version": self.RULE_SET_VERSION,
            "outcomes": [item.value for item in RuleOutcome],
            "rules": [
                "RULE-OBJECT-CONFIRMED",
                "RULE-DEADLINES-ORDERED",
                "RULE-REQUIRED-FIELDS",
                "RULE-EVIDENCE-PACKAGE-FROZEN",
                "RULE-EVIDENCE-BOUND",
                "RULE-NO-SELF-DEPENDENCY",
                "RULE-COOPERATION-DEFINED",
            ],
        }

    def register_document(
        self, request: DocumentImportRequest
    ) -> DocumentVersionRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.DUTY_OFFICER, OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "register a document version",
        )
        if self.rag_service is None:
            raise ValueError("document indexing service is unavailable")
        source_hash = hashlib.sha256(request.content.encode("utf-8")).hexdigest()
        existing_versions = self.repository.list_document_versions(request.document_id)
        duplicate = next(
            (item for item in existing_versions if item.source_hash == source_hash),
            None,
        )
        if duplicate is not None:
            return duplicate
        replaced = None
        if request.replaces_version_id:
            replaced = self.repository.get_document_version(request.replaces_version_id)
            if replaced is None:
                raise LookupError(
                    f"replaced document version not found: {request.replaces_version_id}"
                )
            if replaced.document_id != request.document_id:
                raise ValueError(
                    "a document version may only replace a version of the same document"
                )
        clauses = self._parse_document_clauses(request.content)
        if not clauses:
            raise ValueError("document parsing produced no indexable clauses")
        version_number = (
            (existing_versions[-1].version_number + 1) if existing_versions else 1
        )
        version_id = f"DOCVER-{request.document_id}-{version_number}-{source_hash[:8]}"
        if replaced:
            superseded = []
            for document in self.rag_service.list_documents():
                if document.metadata.get("document_version_id") != replaced.version_id:
                    continue
                metadata = dict(document.metadata)
                metadata["status"] = "superseded"
                metadata["superseded_by"] = version_id
                superseded.append(document.model_copy(update={"metadata": metadata}))
            if superseded:
                self.rag_service.import_documents(superseded)
        index_version = f"frc-index-{source_hash[:12]}"
        documents = [
            RAGDocument(
                doc_id=f"{version_id}:{clause.clause_id}",
                title=request.title,
                corpus=CorpusType.POLICY,
                content=clause.text,
                metadata={
                    "document_id": request.document_id,
                    "document_version_id": version_id,
                    "doc_version": request.version_label,
                    "issuer": request.issuer,
                    "jurisdiction": request.jurisdiction,
                    "effective_at": request.effective_at.isoformat(),
                    "expires_at": request.expires_at.isoformat()
                    if request.expires_at
                    else None,
                    "section_number": clause.clause_id,
                    "section_title": clause.heading,
                    "status": "active",
                    "index_version": index_version,
                    "is_simulated": True,
                    "evidence_roles": [
                        role.value
                        for role in self._document_roles_from_text(clause.text)
                    ],
                },
            )
            for clause in clauses
        ]
        self.rag_service.import_documents(documents)
        record = DocumentVersionRecord(
            version_id=version_id,
            document_id=request.document_id,
            version_number=version_number,
            version_label=request.version_label,
            title=request.title,
            issuer=request.issuer,
            jurisdiction=request.jurisdiction,
            effective_at=request.effective_at,
            expires_at=request.expires_at,
            replaces_version_id=request.replaces_version_id,
            source_hash=source_hash,
            clauses=clauses,
            index_status="indexed",
            index_version=index_version,
            created_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_document_version(record)
        return record

    @staticmethod
    def _parse_document_clauses(content: str) -> list[DocumentClause]:
        paragraphs = [
            item.strip()
            for item in re.split(r"\n\s*\n|(?<=。)\s*\n", content)
            if item.strip()
        ]
        clauses = []
        for index, paragraph in enumerate(paragraphs, start=1):
            first_line = paragraph.splitlines()[0].strip()
            heading = first_line[:60] if len(first_line) <= 60 else f"条款 {index}"
            clauses.append(
                DocumentClause(
                    clause_id=f"C{index:03d}", heading=heading, text=paragraph
                )
            )
        return clauses

    @staticmethod
    def _document_roles_from_text(text: str) -> list[EvidenceRole]:
        mapping = {
            EvidenceRole.CONDITION: ("当", "达到", "预警", "出现"),
            EvidenceRole.OBJECT: ("下穿", "对象", "道路", "学校", "社区"),
            EvidenceRole.RESPONSIBILITY: ("责任", "部门", "防办", "住建", "交警"),
            EvidenceRole.PROCEDURE: ("应", "需", "核查", "组织", "启动", "封控"),
            EvidenceRole.EXCEPTION: ("若", "如", "受阻", "超时", "不足"),
            EvidenceRole.ATTRIBUTION: ("条", "规定", "依据"),
        }
        return [
            role
            for role, keywords in mapping.items()
            if any(keyword in text for keyword in keywords)
        ]

    def run_legacy_migration_inventory(
        self, request: LegacyMigrationRequest
    ) -> MigrationBatchRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "run a legacy migration inventory",
        )
        legacy_events = self.repository.list_v2_events(limit=10000)
        canonical_rows = [
            item.model_dump(mode="json")
            for item in sorted(legacy_events, key=lambda item: item.event_id)
        ]
        source_raw = json.dumps(
            canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        quarantine = []
        for item in legacy_events:
            raw = json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            quarantine.append(
                MigrationQuarantineItem(
                    source_id=item.event_id,
                    reason_code="MISSING_AUTHORITATIVE_WARNING_REVISION",
                    reason="旧事件缺少可验证的专业预警原始载荷、版本哈希和发布来源；不得伪造为新版正式事件。",
                    source_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                )
            )
        record = MigrationBatchRecord(
            batch_id=self._id("MIGRATION"),
            source_table="v2_events",
            target_domain="response_events",
            mapping_version=request.mapping_version,
            dry_run=request.dry_run,
            source_count=len(legacy_events),
            migrated_count=0,
            quarantined_count=len(quarantine),
            ignored_count=0,
            source_hash=hashlib.sha256(source_raw.encode("utf-8")).hexdigest(),
            status="completed_with_quarantine" if quarantine else "completed_empty",
            quarantine=quarantine,
            executed_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_migration_batch(record)
        return record

    def read_legacy_event(
        self,
        legacy_event_id: str,
        *,
        operator_id: str,
        terminal_id: str,
        trace_id: str,
    ) -> dict:
        legacy = self.repository.get_v2_event(legacy_event_id)
        status = "found" if legacy else "not_found"
        call = LegacyAdapterCallRecord(
            call_id=self._id("LEGACYCALL"),
            legacy_endpoint="GET /legacy/events/{event_id}",
            legacy_resource_id=legacy_event_id,
            mapping_version="legacy-v2-read-v1",
            trace_id=trace_id,
            result_status=status,
            operator_id=operator_id,
            terminal_id=terminal_id,
            created_at=self._now(),
        )
        self.repository.save_legacy_adapter_call(call)
        if legacy is None:
            raise LookupError(f"legacy event not found: {legacy_event_id}")
        return {
            "read_only": True,
            "mapping_version": call.mapping_version,
            "trace_id": trace_id,
            "legacy_event_id": legacy.event_id,
            "title": legacy.title,
            "area_id": legacy.area_id,
            "legacy_stage": legacy.current_stage.value,
            "legacy_risk_level": legacy.current_risk_level.value,
            "source_type": "legacy_simulation",
            "migration_eligibility": "quarantine_pending_authoritative_warning",
        }

    def process_outbox(self, request: OutboxProcessRequest) -> list[OutboxMessage]:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN, OperatorRole.LIAISON, OperatorRole.COMMANDER},
            "process the dispatch outbox",
        )
        if request.message_id:
            message = self.repository.get_outbox_message(request.message_id)
            if message is None:
                raise LookupError(f"outbox message not found: {request.message_id}")
            messages = [message]
        else:
            messages = self.repository.list_outbox_messages(
                status=OutboxStatus.PENDING, limit=request.max_messages
            )
        return [
            self._dispatch_outbox_message(
                item, simulation_scenario=request.simulation_scenario
            )
            for item in messages
        ]

    def list_dispatch_callbacks(self, message_id: str) -> list[DispatchCallbackRecord]:
        if self.repository.get_outbox_message(message_id) is None:
            raise LookupError(f"outbox message not found: {message_id}")
        return self.repository.list_dispatch_callbacks(message_id)

    def ingest_simulated_dispatch_callback(
        self, request: DispatchCallbackRequest
    ) -> DispatchCallbackRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.EXTERNAL_SERVICE},
            "submit a simulated dispatch callback",
        )
        if (
            os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
            == "production"
        ):
            raise PermissionError("simulated callbacks are disabled in production")
        expected_token = os.getenv(
            "FLOOD_SIMULATION_MOCK_TOKEN", "simulation-only-change-me"
        )
        if not hmac.compare_digest(request.mock_token, expected_token):
            raise PermissionError("invalid simulation callback token")
        if not request.is_simulated:
            raise ValueError("callback must be explicitly marked as simulated")
        message = self.repository.get_outbox_message(request.message_id)
        if message is None:
            raise LookupError(f"outbox message not found: {request.message_id}")
        callback = self._record_dispatch_callback(
            message,
            request.model_dump(
                mode="json",
                exclude={
                    "operator_id",
                    "operator_role",
                    "terminal_id",
                    "note",
                    "expected_version",
                    "mock_token",
                },
            ),
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
        )
        current = self.repository.get_outbox_message(message.message_id) or message
        updated = current.model_copy(
            update={
                "callback_count": len(
                    self.repository.list_dispatch_callbacks(message.message_id)
                ),
                "updated_at": self._now(),
            }
        )
        self.repository.save_outbox_message(updated)
        return callback

    @staticmethod
    def _task_payload_hash(task: ResponseTask) -> str:
        payload = {
            "task_id": task.task_id,
            "event_id": task.event_id,
            "object_id": task.object_id,
            "version": task.version,
            "title": task.title,
            "action": task.action,
            "responsible_organization": task.responsible_organization,
            "responsible_role": task.responsible_role,
            "cooperate_roles": task.cooperate_roles,
            "deadlines": {
                "acknowledge": task.acknowledge_deadline_at.isoformat(),
                "start": task.start_deadline_at.isoformat()
                if task.start_deadline_at
                else None,
                "complete": task.deadline_at.isoformat(),
                "verify": task.verification_deadline_at.isoformat()
                if task.verification_deadline_at
                else None,
            },
            "required_evidence": task.required_evidence,
            "plan_basis": [item.model_dump(mode="json") for item in task.plan_basis],
            "dependencies": task.dependencies,
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _evidence_package_hash(task: ResponseTask) -> str:
        if task.evidence_package_hash:
            return task.evidence_package_hash
        payload = [item.model_dump(mode="json") for item in task.source_evidence]
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_dispatch_outbox(
        self, task: ResponseTask, approval: ApprovalRecord
    ) -> OutboxMessage:
        idempotency_key = (
            f"dispatch:{task.task_id}:v{task.version}:{approval.task_payload_hash}"
        )
        existing = self.repository.get_outbox_message_by_idempotency_key(
            idempotency_key
        )
        if existing is not None:
            return existing
        now = self._now()
        message = OutboxMessage(
            message_id=self._id("OUTBOX"),
            event_id=task.event_id,
            task_id=task.task_id,
            idempotency_key=idempotency_key,
            payload_hash=approval.task_payload_hash,
            approval_id=approval.approval_id,
            task_version=task.version,
            created_at=now,
            updated_at=now,
        )
        return message

    def _dispatch_outbox_message(
        self,
        message: OutboxMessage,
        *,
        simulation_scenario: str = "normal",
    ) -> OutboxMessage:
        if message.status in {
            OutboxStatus.SENT,
            OutboxStatus.PARTIALLY_SENT,
            OutboxStatus.FAILED,
            OutboxStatus.MANUAL_TAKEOVER,
        }:
            return message
        task = self._task(message.task_id)
        if self._task_payload_hash(task) != message.payload_hash:
            failed = message.model_copy(
                update={
                    "status": OutboxStatus.FAILED,
                    "attempts": message.attempts + 1,
                    "last_error": "approved payload hash mismatch",
                    "updated_at": self._now(),
                }
            )
            self.repository.save_outbox_message(failed)
            return failed
        if (
            os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
            == "production"
        ):
            manual = message.model_copy(
                update={
                    "status": OutboxStatus.MANUAL_TAKEOVER,
                    "attempts": message.attempts + 1,
                    "last_error": "simulation gateway disabled in production",
                    "updated_at": self._now(),
                }
            )
            self.repository.save_outbox_message(manual)
            return manual
        now = self._now()
        transport_payload = {
            "event_id": task.event_id,
            "task_id": task.task_id,
            "task_version": task.version,
            "approval_id": message.approval_id,
            "delivery_targets": [task.responsible_organization, *task.cooperate_roles],
            "approved_payload_hash": message.payload_hash,
        }
        try:
            result = self.simulation_gateway.dispatch(
                message.destination,
                transport_payload,
                scenario=simulation_scenario,
                event_time=now,
                request_id=f"SIMREQ-{message.message_id}",
                trace_id=f"SIMTRACE-{message.event_id}",
                idempotency_key=message.idempotency_key,
            )
        except SimulatedGatewayTimeout as exc:
            pending = message.model_copy(
                update={
                    "status": OutboxStatus.PENDING,
                    "attempts": message.attempts + 1,
                    "simulation_scenario": simulation_scenario,
                    "gateway_status": "timeout",
                    "last_error": str(exc),
                    "updated_at": now,
                }
            )
            self.repository.save_outbox_message(pending)
            self._record(
                event_id=message.event_id,
                task_id=message.task_id,
                entry_type="dispatch",
                action="simulated_dispatch_timeout",
                actor_id="simulated-gateway",
                actor_role=OperatorRole.EXTERNAL_SERVICE.value,
                detail={
                    "message_id": message.message_id,
                    "scenario": simulation_scenario,
                },
            )
            return pending

        gateway_status = str(result["status"])
        if gateway_status == "rejected":
            status = OutboxStatus.FAILED
        elif gateway_status == "partial_success":
            status = OutboxStatus.PARTIALLY_SENT
        else:
            status = OutboxStatus.SENT
        for callback_payload in result.get("callbacks", []):
            self._record_dispatch_callback(message, callback_payload)
        callbacks = self.repository.list_dispatch_callbacks(message.message_id)
        dispatched = message.model_copy(
            update={
                "status": status,
                "attempts": message.attempts + 1,
                "simulation_scenario": simulation_scenario,
                "gateway_status": gateway_status,
                "external_request_id": result["request_id"],
                "trace_id": result["trace_id"],
                "callback_count": len(callbacks),
                "last_error": result.get("error_code"),
                "updated_at": now,
                "sent_at": now
                if status in {OutboxStatus.SENT, OutboxStatus.PARTIALLY_SENT}
                else None,
            }
        )
        self.repository.save_outbox_message(dispatched)
        self._record(
            event_id=message.event_id,
            task_id=message.task_id,
            entry_type="dispatch",
            action=f"simulated_dispatch_{gateway_status}",
            actor_id="simulated-gateway",
            actor_role=OperatorRole.EXTERNAL_SERVICE.value,
            before_state={"status": message.status.value},
            after_state={"status": status.value},
            detail={
                "message_id": message.message_id,
                "scenario": simulation_scenario,
                "accepted_count": result["accepted_count"],
                "rejected_count": result["rejected_count"],
                "callback_count": len(callbacks),
            },
        )
        return dispatched

    def _record_dispatch_callback(
        self,
        message: OutboxMessage,
        payload: dict,
        *,
        actor_id: str = "simulated-gateway",
        actor_role: str = OperatorRole.EXTERNAL_SERVICE.value,
        terminal_id: str = "simulation-gateway",
    ) -> DispatchCallbackRecord:
        idempotency_key = str(payload["idempotency_key"])
        existing = self.repository.get_dispatch_callback_by_idempotency_key(
            idempotency_key
        )
        if existing is not None:
            self._record(
                event_id=message.event_id,
                task_id=message.task_id,
                entry_type="dispatch_callback",
                action="duplicate_dispatch_callback_ignored",
                actor_id=actor_id,
                actor_role=actor_role,
                terminal_id=terminal_id,
                detail={
                    "message_id": message.message_id,
                    "callback_id": existing.callback_id,
                    "idempotency_key": idempotency_key,
                },
            )
            return existing
        previous = [
            item
            for item in self.repository.list_dispatch_callbacks(message.message_id)
            if item.external_id == str(payload["external_id"])
        ]
        version = int(payload["version"])
        sequence_state = (
            CallbackSequenceState.OUT_OF_ORDER
            if previous and version <= max(item.version for item in previous)
            else CallbackSequenceState.IN_ORDER
        )
        callback = DispatchCallbackRecord(
            callback_id=self._id("CALLBACK"),
            message_id=message.message_id,
            event_id=message.event_id,
            task_id=message.task_id,
            external_id=str(payload["external_id"]),
            source=str(payload["source"]),
            version=version,
            event_time=payload["event_time"],
            received_time=payload.get("received_time") or self._now(),
            request_id=str(payload["request_id"]),
            trace_id=str(payload["trace_id"]),
            idempotency_key=idempotency_key,
            status=DispatchCallbackStatus(str(payload["status"])),
            error_code=payload.get("error_code"),
            sequence_state=sequence_state,
            created_at=self._now(),
        )
        saved = self.repository.save_dispatch_callback(callback)
        if saved.callback_id != callback.callback_id:
            self._record(
                event_id=message.event_id,
                task_id=message.task_id,
                entry_type="dispatch_callback",
                action="duplicate_dispatch_callback_ignored",
                actor_id=actor_id,
                actor_role=actor_role,
                terminal_id=terminal_id,
                detail={
                    "message_id": message.message_id,
                    "callback_id": saved.callback_id,
                    "idempotency_key": idempotency_key,
                    "detected_during_insert": True,
                },
            )
            return saved
        action = (
            "out_of_order_dispatch_callback_recorded"
            if saved.sequence_state == CallbackSequenceState.OUT_OF_ORDER
            else "dispatch_callback_recorded"
        )
        self._record(
            event_id=message.event_id,
            task_id=message.task_id,
            entry_type="dispatch_callback",
            action=action,
            actor_id=actor_id,
            actor_role=actor_role,
            terminal_id=terminal_id,
            detail={
                "message_id": message.message_id,
                "callback_id": saved.callback_id,
                "external_id": saved.external_id,
                "version": saved.version,
                "status": saved.status.value,
                "sequence_state": saved.sequence_state.value,
                "formal_task_state_unchanged": True,
            },
        )
        return saved

    def start_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        task = self._task(task_id)
        if task.assignee_id is None:
            raise ValueError(
                "task must be assigned to a field operator before execution starts"
            )
        if (
            request.operator_role != OperatorRole.FIELD_OPERATOR
            or request.operator_id != task.assignee_id
        ):
            raise PermissionError(
                "only the assigned field operator may start this task"
            )
        return self._transition(
            task_id,
            request,
            {TaskStatus.ACKNOWLEDGED, TaskStatus.ESCALATED, TaskStatus.BLOCKED},
            TaskStatus.IN_PROGRESS,
            "task_started",
        )

    def assign_task(self, task_id: str, request: TaskAssignmentRequest) -> ResponseTask:
        self._require_role(
            request.operator_role,
            {OperatorRole.LIAISON, OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "assign a field operator",
        )
        if request.assignee_role != OperatorRole.FIELD_OPERATOR:
            raise ValueError("response tasks may only be assigned to a field operator")
        task = self._task(task_id)
        self._event(task.event_id, active=True)
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.ESCALATED,
            TaskStatus.BLOCKED,
            TaskStatus.PARTIALLY_COMPLETED,
            TaskStatus.TAKEN_OVER,
        }:
            raise ValueError("task cannot be assigned in the current status")
        previous_assignee = task.assignee_id
        if previous_assignee == request.assignee_id:
            return task
        assignment_version = task.assignment_version + 1
        task = task.model_copy(
            update={
                "assignee_id": request.assignee_id,
                "assignee_name": request.assignee_name,
                "assignee_role": request.assignee_role,
                "assignment_version": assignment_version,
                "updated_at": self._now(),
            }
        )
        assignment = TaskAssignmentRecord(
            assignment_id=self._id("ASSIGN"),
            event_id=task.event_id,
            task_id=task.task_id,
            assignment_version=assignment_version,
            previous_assignee_id=previous_assignee,
            assignee_id=request.assignee_id,
            assignee_name=request.assignee_name,
            assignee_role=request.assignee_role,
            reason=request.reason,
            assigned_by=request.operator_id,
            assigned_by_role=request.operator_role,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_response_task(task)
        self.repository.save_task_assignment(assignment)
        action = "task_reassigned" if previous_assignee else "task_assigned"
        self._task_event(
            task,
            action,
            request,
            {
                "assignment_id": assignment.assignment_id,
                "assignment_version": assignment_version,
                "previous_assignee_id": previous_assignee,
                "assignee_id": request.assignee_id,
                "reason": request.reason,
            },
        )
        return task

    def submit_feedback(self, task_id: str, request: FeedbackRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, EXECUTION_ROLES, "submit task feedback"
        )
        task = self._task(task_id)
        if (
            task.assignee_id
            and request.operator_role == OperatorRole.FIELD_OPERATOR
            and request.operator_id != task.assignee_id
        ):
            raise PermissionError(
                "only the assigned field operator may submit execution feedback"
            )
        self._event(task.event_id, active=True)
        previous_status = task.status
        category, classification_reason = classify_feedback(request)
        dedupe_key = feedback_dedupe_key(request)
        duplicate = next(
            (
                item
                for item in self.repository.list_task_feedback(task_id)
                if item.dedupe_key == dedupe_key
            ),
            None,
        )
        if duplicate is not None:
            self._task_event(
                task,
                "duplicate_feedback_merged",
                request,
                {
                    "duplicate_of": duplicate.feedback_id,
                    "category": duplicate.category.value,
                    "dedupe_key": dedupe_key,
                },
            )
            return task
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.ESCALATED,
            TaskStatus.BLOCKED,
            TaskStatus.PARTIALLY_COMPLETED,
            TaskStatus.TAKEN_OVER,
        }:
            raise ValueError("feedback cannot be submitted in the current task status")
        escalation_reason = request.blocked_reason or request.resource_gap
        if category == FeedbackCategory.COORDINATION_REQUEST:
            escalation_reason = request.summary
        if category == FeedbackCategory.COMPLETION:
            missing_types = missing_evidence_types(task, request.evidence)
            if missing_types:
                raise ValueError(
                    f"required evidence missing: {', '.join(missing_types)}"
                )
        feedback = TaskFeedback(
            feedback_id=self._id("FDB"),
            event_id=task.event_id,
            task_id=task.task_id,
            object_id=task.object_id,
            summary=request.summary,
            evidence=request.evidence,
            blocked_reason=request.blocked_reason,
            resource_gap=request.resource_gap,
            category=category,
            classification_reason=classification_reason,
            dedupe_key=dedupe_key,
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            created_at=self._now(),
        )
        self.repository.save_task_feedback(feedback)
        if escalation_reason:
            previous = task.status
            ensure_transition(task.status, TaskStatus.BLOCKED)
            task = task.model_copy(
                update={"status": TaskStatus.BLOCKED, "updated_at": self._now()}
            )
            escalation = self._save_escalation(
                task,
                previous,
                escalation_reason,
                "duty_officer",
                recommended_actions=recommend_feedback_alternatives(category),
            )
            action = "task_blocked_and_escalated"
        elif category == FeedbackCategory.COMPLETION:
            ensure_transition(task.status, TaskStatus.PENDING_VERIFICATION)
            task = task.model_copy(
                update={
                    "status": TaskStatus.PENDING_VERIFICATION,
                    "updated_at": self._now(),
                }
            )
            action = "completion_submitted"
        elif category == FeedbackCategory.PARTIAL_COMPLETION:
            ensure_transition(task.status, TaskStatus.PARTIALLY_COMPLETED)
            task = task.model_copy(
                update={
                    "status": TaskStatus.PARTIALLY_COMPLETED,
                    "updated_at": self._now(),
                }
            )
            action = "task_partially_completed"
        else:
            action = "situation_feedback_recorded"
        self.repository.save_response_task(task)
        self._task_event(
            task,
            action,
            request,
            {
                "feedback_id": feedback.feedback_id,
                "summary": request.summary,
                "category": category.value,
                "escalation_id": escalation.escalation_id
                if escalation_reason
                else None,
                "previous_status": previous_status.value,
            },
        )
        return task

    def verify_completion(
        self, task_id: str, approved: bool, request: TaskActionRequest
    ) -> ResponseTask:
        self._require_role(request.operator_role, EDIT_ROLES, "verify task completion")
        task = self._task(task_id)
        if task.status != TaskStatus.PENDING_VERIFICATION:
            raise ValueError("task is not pending verification")
        completion_feedback = [
            item
            for item in self.repository.list_task_feedback(task_id)
            if item.category == FeedbackCategory.COMPLETION
        ]
        if (
            completion_feedback
            and completion_feedback[-1].operator_id == request.operator_id
        ):
            raise PermissionError(
                "the completion submitter cannot verify their own result"
            )
        if task.assignee_id and task.assignee_id == request.operator_id:
            raise PermissionError("the assigned executor cannot verify their own task")
        target = TaskStatus.COMPLETED if approved else TaskStatus.IN_PROGRESS
        ensure_transition(task.status, target)
        previous_status = task.status
        task = task.model_copy(update={"status": target, "updated_at": self._now()})
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "completion_verified" if approved else "completion_returned",
            request,
            {"note": request.note, "previous_status": previous_status.value},
        )
        return task

    def request_deadline_extension(
        self, task_id: str, request: DeadlineExtensionRequest
    ) -> DeadlineExtensionRecord:
        self._require_role(
            request.operator_role, EXECUTION_ROLES, "request a deadline extension"
        )
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
            TaskStatus.ESCALATED,
            TaskStatus.PARTIALLY_COMPLETED,
        }:
            raise ValueError(
                "deadline extension cannot be requested in the current task status"
            )
        current_completion = task.effective_completion_deadline_at or task.deadline_at
        if request.proposed_completion_deadline_at <= current_completion:
            raise ValueError(
                "proposed completion deadline must be later than the current effective deadline"
            )
        record = DeadlineExtensionRecord(
            extension_id=self._id("EXT"),
            event_id=task.event_id,
            task_id=task.task_id,
            task_version=task.version,
            status=DeadlineExtensionStatus.REQUESTED,
            original_start_deadline_at=task.effective_start_deadline_at
            or task.start_deadline_at,
            original_completion_deadline_at=current_completion,
            original_verification_deadline_at=task.effective_verification_deadline_at
            or task.verification_deadline_at,
            proposed_start_deadline_at=request.proposed_start_deadline_at,
            proposed_completion_deadline_at=request.proposed_completion_deadline_at,
            proposed_verification_deadline_at=request.proposed_verification_deadline_at,
            request_reason=request.reason,
            requested_by=request.operator_id,
            requested_by_role=request.operator_role,
            created_at=self._now(),
        )
        self.repository.save_deadline_extension(record)
        self._task_event(
            task,
            "deadline_extension_requested",
            request,
            {"extension_id": record.extension_id, "reason": request.reason},
        )
        return record

    def decide_deadline_extension(
        self, extension_id: str, request: DeadlineExtensionDecisionRequest
    ) -> DeadlineExtensionRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "decide a deadline extension",
        )
        record = self.repository.get_deadline_extension(extension_id)
        if record is None:
            raise LookupError(f"deadline extension not found: {extension_id}")
        if record.status != DeadlineExtensionStatus.REQUESTED:
            raise ValueError("deadline extension has already been decided")
        if record.requested_by == request.operator_id:
            raise PermissionError(
                "the requester cannot decide their own deadline extension"
            )
        task = self._task(record.task_id)
        now = self._now()
        status = (
            DeadlineExtensionStatus.APPROVED
            if request.approved
            else DeadlineExtensionStatus.REJECTED
        )
        record = record.model_copy(
            update={
                "status": status,
                "decided_by": request.operator_id,
                "decision_reason": request.reason,
                "decided_at": now,
            }
        )
        self.repository.save_deadline_extension(record)
        if request.approved:
            task = task.model_copy(
                update={
                    "effective_start_deadline_at": record.proposed_start_deadline_at
                    or task.effective_start_deadline_at
                    or task.start_deadline_at,
                    "effective_completion_deadline_at": record.proposed_completion_deadline_at,
                    "effective_verification_deadline_at": record.proposed_verification_deadline_at
                    or task.effective_verification_deadline_at
                    or task.verification_deadline_at,
                    "active_extension_id": record.extension_id,
                    "updated_at": now,
                }
            )
            self.repository.save_response_task(task)
        self._task_event(
            task,
            "deadline_extension_approved"
            if request.approved
            else "deadline_extension_rejected",
            request,
            {"extension_id": record.extension_id, "reason": request.reason},
        )
        return record

    def cancel_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, {OperatorRole.COMMANDER}, "cancel a task"
        )
        if len(request.note.strip()) < 4:
            raise ValueError("task cancellation requires a reason")
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.WAIVED,
            TaskStatus.CANCELLED,
        }:
            raise ValueError("terminal tasks cannot be cancelled")
        previous = task.status
        ensure_transition(task.status, TaskStatus.CANCELLED)
        task = task.model_copy(
            update={"status": TaskStatus.CANCELLED, "updated_at": self._now()}
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_cancelled",
            request,
            {"previous_status": previous.value, "reason": request.note},
        )
        return task

    def take_over_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "take over a task",
        )
        if len(request.note.strip()) < 4:
            raise ValueError("manual takeover requires a reason")
        task = self._task(task_id)
        self._assert_expected_version(task, request.expected_version)
        if task.status not in {
            TaskStatus.ISSUED,
            TaskStatus.ACKNOWLEDGED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
            TaskStatus.ESCALATED,
            TaskStatus.PARTIALLY_COMPLETED,
        }:
            raise ValueError("task cannot be taken over in the current status")
        previous = task.status
        ensure_transition(task.status, TaskStatus.TAKEN_OVER)
        task = task.model_copy(
            update={
                "status": TaskStatus.TAKEN_OVER,
                "assignee_id": request.operator_id,
                "assignee_name": "人工接管岗位",
                "assignee_role": request.operator_role,
                "updated_at": self._now(),
            }
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_taken_over",
            request,
            {"previous_status": previous.value, "reason": request.note},
        )
        return task

    def waive_task(self, task_id: str, request: TaskActionRequest) -> ResponseTask:
        self._require_role(
            request.operator_role, {OperatorRole.COMMANDER}, "waive a task"
        )
        task = self._task(task_id)
        if task.status in {TaskStatus.COMPLETED, TaskStatus.WAIVED}:
            return task
        previous_status = task.status
        ensure_transition(task.status, TaskStatus.WAIVED)
        task = task.model_copy(
            update={"status": TaskStatus.WAIVED, "updated_at": self._now()}
        )
        self.repository.save_response_task(task)
        self._task_event(
            task,
            "task_waived",
            request,
            {"reason": request.note, "previous_status": previous_status.value},
        )
        return task

    def run_deadline_sweep(
        self, event_id: str, request: TaskActionRequest, *, now: datetime | None = None
    ) -> list[ResponseTask]:
        self._require_role(request.operator_role, EDIT_ROLES, "run deadline escalation")
        self._event(event_id, active=True)
        moment = now or self._now()
        escalated: list[ResponseTask] = []
        for task in self.repository.list_response_tasks(event_id):
            reason = None
            deadline_type = None
            start_deadline = task.effective_start_deadline_at or task.start_deadline_at
            completion_deadline = (
                task.effective_completion_deadline_at or task.deadline_at
            )
            verification_deadline = (
                task.effective_verification_deadline_at or task.verification_deadline_at
            )
            if (
                task.status == TaskStatus.ISSUED
                and task.acknowledge_deadline_at < moment
            ):
                reason = "任务超过确认时限仍未确认"
                deadline_type = "acknowledge"
            elif (
                task.status == TaskStatus.ACKNOWLEDGED
                and start_deadline
                and start_deadline < moment
            ):
                reason = "任务超过开始时限仍未开始"
                deadline_type = "start"
            elif (
                task.status
                in {
                    TaskStatus.IN_PROGRESS,
                    TaskStatus.PARTIALLY_COMPLETED,
                    TaskStatus.BLOCKED,
                }
                and completion_deadline < moment
            ):
                reason = "任务超过完成时限"
                deadline_type = "completion"
            elif (
                task.status == TaskStatus.PENDING_VERIFICATION
                and verification_deadline
                and verification_deadline < moment
            ):
                reason = "任务超过核验时限"
                deadline_type = "verification"
            if reason:
                previous = task.status
                ensure_transition(task.status, TaskStatus.ESCALATED)
                task = task.model_copy(
                    update={"status": TaskStatus.ESCALATED, "updated_at": self._now()}
                )
                self.repository.save_response_task(task)
                escalation = self._save_escalation(task, previous, reason, "reviewer")
                self._task_event(
                    task,
                    "deadline_escalated",
                    request,
                    {
                        "reason": reason,
                        "deadline_type": deadline_type,
                        "escalation_id": escalation.escalation_id,
                        "previous_status": previous.value,
                    },
                )
                escalated.append(task)
        return escalated

    def close_event(self, event_id: str, request: EventCloseRequest) -> EventDashboard:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "close an event",
        )
        event = self._event(event_id, active=True)
        tasks = self.repository.list_response_tasks(event_id)
        if not tasks:
            raise ValueError("an event with no tasks cannot be closed")
        incomplete = [
            task.task_id
            for task in tasks
            if task.status
            not in {TaskStatus.COMPLETED, TaskStatus.WAIVED, TaskStatus.CANCELLED}
        ]
        if incomplete:
            raise ValueError(f"event has incomplete tasks: {', '.join(incomplete)}")
        event = event.model_copy(
            update={
                "status": EventStatus.CLOSED,
                "updated_at": self._now(),
                "closed_at": self._now(),
                "closed_by": request.operator_id,
            }
        )
        self.repository.save_response_event(event)
        self._record(
            event_id=event_id,
            entry_type="event",
            action="event_closed",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state={"event_status": EventStatus.ACTIVE.value},
            after_state={"event_status": event.status.value},
            detail={"note": request.note, "task_count": len(tasks)},
        )
        self.generate_review_draft(
            event_id,
            ReviewDraftRequest(
                operator_id=request.operator_id,
                operator_role=request.operator_role,
                terminal_id=request.terminal_id,
                note="事件关闭时自动生成复盘草稿",
            ),
        )
        return self.get_dashboard(event_id)

    def generate_review_draft(
        self, event_id: str, request: ReviewDraftRequest
    ) -> EventReviewDraft:
        self._require_role(
            request.operator_role,
            {
                OperatorRole.DUTY_OFFICER,
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.AUDITOR,
            },
            "generate an event review draft",
        )
        event = self._event(event_id)
        tasks = self.repository.list_response_tasks(event_id)
        objects = self.repository.list_event_risk_objects(event_id)
        feedback = self.repository.list_event_feedback(event_id)
        escalations = self.repository.list_escalations(event_id)
        timeline = self.repository.list_timeline_entries(event_id)
        completed = [
            task
            for task in tasks
            if task.status
            in {TaskStatus.COMPLETED, TaskStatus.WAIVED, TaskStatus.CANCELLED}
        ]
        duplicates = [
            entry for entry in timeline if entry.action == "duplicate_feedback_merged"
        ]
        evidence_complete = 0
        task_by_id = {task.task_id: task for task in tasks}
        for item in feedback:
            task = task_by_id.get(item.task_id)
            if (
                task
                and item.category == FeedbackCategory.COMPLETION
                and not missing_evidence_types(task, item.evidence)
            ):
                evidence_complete += 1

        process_metrics: dict[str, int | float] = {
            "alert_versions": len(self.repository.list_alert_snapshots(event_id)),
            "candidate_objects": len(objects),
            "confirmed_objects": sum(
                item.verification_status == ObjectVerificationStatus.CONFIRMED
                for item in objects
            ),
            "task_count": len(tasks),
            "completed_tasks": len(completed),
            "feedback_count": len(feedback),
            "duplicate_feedback_merged": len(duplicates),
            "escalation_count": len(escalations),
            "evidence_complete_feedback": evidence_complete,
            "completion_rate": round(len(completed) / len(tasks), 4) if tasks else 0.0,
        }
        key_decisions = [
            review_timeline_line(entry)
            for entry in timeline
            if entry.action
            in {
                "task_approved_and_issued",
                "task_rejected_to_draft",
                "task_waived",
                "event_closed",
            }
        ]
        delays = [
            f"任务 {item.task_id}：{item.reason}；建议：{'、'.join(item.recommended_actions) or '人工协调'}"
            for item in escalations
        ]
        evidence_findings = [
            f"{evidence_complete} 条完成反馈通过必需证据类型校验。",
            f"{sum(bool(task.source_evidence) for task in tasks)} 项任务保留了成案来源证据。",
            f"系统合并了 {len(duplicates)} 条重复反馈，避免重复计入过程指标。",
        ]
        unresolved = [
            f"任务 {task.task_id} 当前状态为 {task.status.value}。"
            for task in tasks
            if task.status not in {TaskStatus.COMPLETED, TaskStatus.WAIVED}
        ]
        recommendations = build_review_recommendations(
            feedback,
            has_escalations=bool(escalations),
            has_duplicates=bool(duplicates),
            has_unresolved_tasks=bool(unresolved),
        )
        review = EventReviewDraft(
            review_id=self._id("REVIEW"),
            event_id=event_id,
            headline=f"{event.title}响应过程复盘草稿",
            executive_summary=(
                f"事件共关联 {len(objects)} 个候选对象、形成 {len(tasks)} 项任务，"
                f"完成或批准豁免 {len(completed)} 项，发生 {len(escalations)} 次升级。"
            ),
            process_metrics=process_metrics,
            key_decisions=key_decisions,
            delays_and_escalations=delays,
            evidence_findings=evidence_findings,
            unresolved_issues=unresolved,
            improvement_recommendations=recommendations,
            source_timeline_entry_ids=[entry.entry_id for entry in timeline],
            generated_by=request.operator_id,
            created_at=self._now(),
        )
        self.repository.save_event_review_draft(review)
        self._record(
            event_id=event_id,
            entry_type="review",
            action="event_review_draft_generated",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "review_id": review.review_id,
                "source_entry_count": len(timeline),
                "note": request.note,
            },
        )
        return review

    def run_scenario_evaluation(
        self, event_id: str, request: ScenarioEvaluationRequest
    ) -> DistrictScenarioReport:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.AUDITOR},
            "run a district scenario evaluation",
        )
        dashboard = self.get_dashboard(event_id)
        event = dashboard.event
        timeline = dashboard.timeline
        actions = {entry.action for entry in timeline}
        action_entries: dict[str, list[TimelineEntry]] = {}
        for entry in timeline:
            action_entries.setdefault(entry.action, []).append(entry)

        def refs(*names: str) -> list[str]:
            return [
                entry.entry_id
                for name in names
                for entry in action_entries.get(name, [])
            ]

        def duration(
            start_actions: tuple[str, ...], end_actions: tuple[str, ...]
        ) -> float | None:
            starts = [
                entry
                for name in start_actions
                for entry in action_entries.get(name, [])
            ]
            ends = [
                entry for name in end_actions for entry in action_entries.get(name, [])
            ]
            if not starts or not ends:
                return None
            start = min(starts, key=lambda item: item.created_at)
            later = [item for item in ends if item.created_at >= start.created_at]
            if not later:
                return None
            elapsed = (
                min(later, key=lambda item: item.created_at).created_at
                - start.created_at
            ).total_seconds() / 60
            return round(max(elapsed, 0.0), 3)

        def rate(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator * 100, 2) if denominator else None

        manual = request.manual_baseline
        object_ids = {item.object_id for item in dashboard.risk_objects}
        expected_ids = set(request.expected_object_ids)
        missing_expected = expected_ids - object_ids
        object_omission_rate = (
            rate(len(missing_expected), len(expected_ids)) if expected_ids else None
        )
        manual_corrections = sum(
            item.verification_status == ObjectVerificationStatus.EXCLUDED
            for item in dashboard.risk_objects
        ) + len(action_entries.get("candidate_updated", []))
        manual_correction_rate = rate(manual_corrections, len(dashboard.risk_objects))

        required_task_fields = (
            "object_id",
            "action",
            "responsible_organization",
            "responsible_role",
            "deadline_at",
            "acknowledge_deadline_at",
            "required_evidence",
            "plan_basis",
            "escalation_rule",
        )
        complete_tasks = sum(
            all(getattr(task, field) for field in required_task_fields)
            for task in dashboard.tasks
        )
        task_completeness = rate(complete_tasks, len(dashboard.tasks))
        approved_tasks = sum(
            task.approved_version is not None for task in dashboard.tasks
        )
        completed_tasks = sum(
            task.status in {TaskStatus.COMPLETED, TaskStatus.WAIVED}
            for task in dashboard.tasks
        )
        verified_tasks = {
            entry.task_id for entry in action_entries.get("completion_verified", [])
        }
        feedback_complete = 0
        task_by_id = {task.task_id: task for task in dashboard.tasks}
        for item in dashboard.feedback:
            task = task_by_id.get(item.task_id)
            if (
                task
                and item.category == FeedbackCategory.COMPLETION
                and not missing_evidence_types(task, item.evidence)
            ):
                feedback_complete += 1
        completion_feedback = sum(
            item.category == FeedbackCategory.COMPLETION for item in dashboard.feedback
        )
        verification_attempts = len(
            action_entries.get("completion_verified", [])
        ) + len(action_entries.get("completion_returned", []))

        escalation_task_ids = {
            entry.task_id
            for entry in timeline
            if entry.action in {"deadline_escalated", "task_blocked_and_escalated"}
        }
        recovered_escalations = {
            task_id
            for task_id in escalation_task_ids
            if task_id
            and task_by_id.get(task_id)
            and task_by_id[task_id].status in {TaskStatus.COMPLETED, TaskStatus.WAIVED}
        }
        escalation_durations: list[float] = []
        for escalation in [
            entry
            for entry in timeline
            if entry.action in {"deadline_escalated", "task_blocked_and_escalated"}
        ]:
            recovery = next(
                (
                    entry
                    for entry in timeline
                    if entry.task_id == escalation.task_id
                    and entry.created_at >= escalation.created_at
                    and entry.action
                    in {
                        "task_acknowledged",
                        "task_started",
                        "completion_submitted",
                        "task_waived",
                    }
                ),
                None,
            )
            if recovery:
                escalation_durations.append(
                    (recovery.created_at - escalation.created_at).total_seconds() / 60
                )

        expected_actions = {
            "event_created",
            "candidate_added",
            "candidate_confirmed",
            "task_submitted",
            "task_approved_and_issued",
        }
        if dashboard.tasks:
            expected_actions.add("task_draft_created")
        if completed_tasks:
            expected_actions.update(
                {
                    "task_acknowledged",
                    "task_started",
                    "completion_submitted",
                    "completion_verified",
                }
            )
        if event.status == EventStatus.CLOSED:
            expected_actions.update({"event_closed", "event_review_draft_generated"})
        timeline_coverage = rate(len(expected_actions & actions), len(expected_actions))
        linked_timeline = sum(
            entry.event_id == event_id
            and (entry.task_id is None or entry.task_id in task_by_id)
            for entry in timeline
        )
        timeline_integrity = rate(linked_timeline, len(timeline))

        metric_specs = [
            (
                "object_list_generation_time",
                "对象研判",
                "清单生成时间",
                "分钟",
                duration(("event_created",), ("candidate_added",)),
                refs("event_created", "candidate_added"),
            ),
            (
                "object_omission_rate",
                "对象研判",
                "对象遗漏率",
                "%",
                object_omission_rate,
                refs("candidate_added", "candidate_confirmed"),
            ),
            (
                "manual_correction_rate",
                "对象研判",
                "人工修正率",
                "%",
                manual_correction_rate,
                refs("candidate_updated", "candidate_excluded"),
            ),
            (
                "task_generation_time",
                "方案生成",
                "任务生成时间",
                "分钟",
                duration(
                    ("candidate_confirmed",),
                    ("task_draft_created", "grounded_task_draft_generated"),
                ),
                refs(
                    "candidate_confirmed",
                    "task_draft_created",
                    "grounded_task_draft_generated",
                ),
            ),
            (
                "task_element_completeness",
                "方案生成",
                "任务要素完整率",
                "%",
                task_completeness,
                [task.task_id for task in dashboard.tasks],
            ),
            (
                "approval_duration",
                "审批执行",
                "审批耗时",
                "分钟",
                duration(("task_submitted",), ("task_approved_and_issued",)),
                refs("task_submitted", "task_approved_and_issued"),
            ),
            (
                "acknowledgement_duration",
                "审批执行",
                "确认耗时",
                "分钟",
                duration(("task_approved_and_issued",), ("task_acknowledged",)),
                refs("task_approved_and_issued", "task_acknowledged"),
            ),
            (
                "on_time_completion_rate",
                "审批执行",
                "按时完成率",
                "%",
                rate(completed_tasks, len(dashboard.tasks)),
                refs("completion_verified", "task_waived"),
            ),
            (
                "timeout_detection_rate",
                "异常处理",
                "超时识别率",
                "%",
                100.0 if action_entries.get("deadline_escalated") else None,
                refs("deadline_escalated"),
            ),
            (
                "escalation_handling_time",
                "异常处理",
                "升级处理时间",
                "分钟",
                round(sum(escalation_durations) / len(escalation_durations), 3)
                if escalation_durations
                else None,
                refs(
                    "deadline_escalated",
                    "task_blocked_and_escalated",
                    "task_acknowledged",
                    "task_started",
                ),
            ),
            (
                "reassignment_success_rate",
                "异常处理",
                "改派/恢复成功率",
                "%",
                rate(len(recovered_escalations), len(escalation_task_ids)),
                refs(
                    "deadline_escalated",
                    "task_blocked_and_escalated",
                    "completion_verified",
                ),
            ),
            (
                "execution_evidence_completeness",
                "反馈核实",
                "执行证据完整率",
                "%",
                rate(feedback_complete, completion_feedback),
                [item.feedback_id for item in dashboard.feedback],
            ),
            (
                "verification_return_rate",
                "反馈核实",
                "待核实退回率",
                "%",
                rate(
                    len(action_entries.get("completion_returned", [])),
                    verification_attempts,
                ),
                refs("completion_returned", "completion_verified"),
            ),
            (
                "log_coverage",
                "复盘审计",
                "日志覆盖率",
                "%",
                timeline_coverage,
                refs(*sorted(expected_actions & actions)),
            ),
            (
                "timeline_completeness",
                "复盘审计",
                "时间线完整率",
                "%",
                timeline_integrity,
                [entry.entry_id for entry in timeline],
            ),
            (
                "report_generation_time",
                "复盘审计",
                "报告生成时间",
                "分钟",
                duration(("event_closed",), ("event_review_draft_generated",)),
                refs("event_closed", "event_review_draft_generated"),
            ),
        ]
        metrics: list[ScenarioMetricResult] = []
        observations: list[str] = []
        for metric_id, category, label, unit, system_value, evidence in metric_specs:
            manual_value = manual.get(metric_id)
            if system_value is None:
                status = "not_evaluable"
                interpretation = "当前场景没有足够的真值或异常样本，未计算该指标。"
                observations.append(
                    f"{label}缺少可计算样本；后续场景需补充人工标注真值或异常事件。"
                )
            elif manual_value is not None and unit == "分钟":
                status = "measured"
                saved = manual_value - system_value
                interpretation = f"相对人工参考基线节省 {round(saved, 2)} 分钟。"
            else:
                status = "measured"
                interpretation = "由事件台账、任务状态和证据记录直接计算。"
            metrics.append(
                ScenarioMetricResult(
                    metric_id=metric_id,
                    category=category,
                    label=label,
                    unit=unit,
                    manual_value=manual_value,
                    system_value=system_value,
                    status=status,
                    interpretation=interpretation,
                    evidence_refs=evidence,
                )
            )

        versions = [item.version for item in dashboard.alert_snapshots]
        task_fields_ok = complete_tasks == len(dashboard.tasks) and bool(
            dashboard.tasks
        )
        approval_ok = approved_tasks == len(dashboard.tasks) and bool(dashboard.tasks)
        trace_ok = timeline_coverage == 100.0 and timeline_integrity == 100.0
        timeout_ok = not any(
            task.status == TaskStatus.ESCALATED for task in dashboard.tasks
        ) or bool(escalation_task_ids)
        feedback_linked = all(
            item.event_id == event_id
            and item.task_id in task_by_id
            and item.object_id == task_by_id[item.task_id].object_id
            for item in dashboard.feedback
        ) and bool(dashboard.feedback)
        verified_ok = all(
            task.status == TaskStatus.WAIVED or task.task_id in verified_tasks
            for task in dashboard.tasks
        )
        review_ok = (
            event.status == EventStatus.CLOSED and dashboard.review_draft is not None
        )
        core_actions = {
            "event_created",
            "task_submitted",
            "task_approved_and_issued",
            "completion_verified",
            "event_closed",
        }
        manual_fallback_ok = core_actions.issubset(actions)
        actor_roles = {
            entry.actor_role
            for entry in timeline
            if entry.actor_role != "system" and entry.entry_type != "evaluation"
        }
        minimum_scope_ok = (
            event.status == EventStatus.CLOSED
            and len(dashboard.alert_snapshots) >= 1
            and len(dashboard.risk_objects) >= 1
            and len(dashboard.tasks) >= 1
            and len(actor_roles) >= 4
        )
        checks_data = [
            (
                "AC-01",
                "每个事件具有唯一编号和完整预警版本",
                bool(event.event_id) and versions == list(range(1, len(versions) + 1)),
                f"事件 {event.event_id}，预警版本 {versions}",
                [
                    event.event_id,
                    *[item.snapshot_id for item in dashboard.alert_snapshots],
                ],
            ),
            (
                "AC-02",
                "任务绑定对象、责任、时限、依据和反馈要求",
                task_fields_ok,
                f"{complete_tasks}/{len(dashboard.tasks)} 项任务要素完整",
                [task.task_id for task in dashboard.tasks],
            ),
            (
                "AC-03",
                "高风险任务未审批时不能下发",
                approval_ok,
                f"{approved_tasks}/{len(dashboard.tasks)} 项任务保留批准版本",
                refs("task_approved_and_issued"),
            ),
            (
                "AC-04",
                "所有审批和状态变化可追溯",
                trace_ok,
                f"日志覆盖率 {timeline_coverage or 0}% ，关联完整率 {timeline_integrity or 0}%",
                refs(*sorted(expected_actions & actions)),
            ),
            (
                "AC-05",
                "未确认、超时和受阻任务可提醒和升级",
                timeout_ok,
                f"记录 {len(escalation_task_ids)} 项升级；当前无未记录的升级状态",
                refs("deadline_escalated", "task_blocked_and_escalated"),
            ),
            (
                "AC-06",
                "现场反馈关联到事件、对象和任务",
                feedback_linked,
                f"{len(dashboard.feedback)} 条反馈均通过三类编号关联",
                [item.feedback_id for item in dashboard.feedback],
            ),
            (
                "AC-07",
                "任务完成后必须经过核实",
                verified_ok,
                f"{len(verified_tasks)} 项任务记录完成核实",
                refs("completion_verified"),
            ),
            (
                "AC-08",
                "事件关闭时生成时间线和过程指标",
                review_ok,
                f"时间线 {len(timeline)} 条，复盘草稿 {'已生成' if dashboard.review_draft else '未生成'}",
                [dashboard.review_draft.review_id] if dashboard.review_draft else [],
            ),
            (
                "AC-09",
                "AI 不可用时人工流程仍可继续运行",
                manual_fallback_ok,
                "确定性状态机已完成创建、审批、执行、核实和关闭，评测过程未调用模型",
                refs(*sorted(core_actions & actions)),
            ),
            (
                "AC-10",
                "至少一个最小区县场景完成端到端测试",
                minimum_scope_ok,
                f"1 个区县、{len(dashboard.risk_objects)} 类对象实例、{len(dashboard.alert_snapshots)} 版预警、{len(actor_roles)} 个岗位完成闭环",
                [event.event_id],
            ),
        ]
        checks = [
            ScenarioAcceptanceCheck(
                check_id=check_id,
                requirement=requirement,
                passed=passed,
                detail=detail,
                evidence_refs=evidence,
            )
            for check_id, requirement, passed, detail, evidence in checks_data
        ]
        failures = [
            ScenarioFailureCase(
                failure_id=self._id("FAIL"),
                stage="验收检查",
                severity="blocking",
                trigger=check.check_id,
                observed=check.detail,
                expected=check.requirement,
                recommendation="补齐对应业务记录后重新运行场景评测。",
            )
            for check in checks
            if not check.passed
        ]
        for escalation in dashboard.escalations:
            failures.append(
                ScenarioFailureCase(
                    failure_id=self._id("FAIL"),
                    stage="异常处理",
                    severity="handled",
                    trigger=escalation.reason,
                    observed=f"任务 {escalation.task_id} 触发升级并形成备选动作。",
                    expected="异常应被发现、升级并恢复到可执行状态。",
                    recommendation="复盘升级耗时，并验证推荐动作是否需要固化为标准预案。",
                )
            )
        all_passed = all(check.passed for check in checks)
        overall_status = (
            "failed"
            if not all_passed
            else ("passed_with_observations" if observations or failures else "passed")
        )
        report = DistrictScenarioReport(
            report_id=self._id("SCENARIO"),
            event_id=event_id,
            scenario_name=request.scenario_name,
            scenario_type=request.scenario_type,
            scope={
                "area_id": event.area_id,
                "alert_versions": len(dashboard.alert_snapshots),
                "risk_object_instances": len(dashboard.risk_objects),
                "risk_object_types": sorted(
                    {item.object_type for item in dashboard.risk_objects}
                ),
                "task_count": len(dashboard.tasks),
                "operator_roles": sorted(actor_roles),
            },
            baseline_note="人工基线为可配置参考假设，不代表真实对照组实测；系统值来自当前事件不可覆盖台账。",
            metrics=metrics,
            acceptance_checks=checks,
            failure_cases=failures,
            observations=observations,
            overall_status=overall_status,
            generated_by=request.operator_id,
            created_at=self._now(),
        )
        self.repository.save_district_scenario_report(report)
        self._record(
            event_id=event_id,
            entry_type="evaluation",
            action="scenario_evaluation_completed",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "report_id": report.report_id,
                "overall_status": overall_status,
                "note": request.note,
            },
        )
        return report

    def list_scenario_reports(self, event_id: str) -> list[DistrictScenarioReport]:
        self._event(event_id)
        return self.repository.list_district_scenario_reports(event_id)

    def verify_timeline_integrity(
        self, event_id: str, operator_role: OperatorRole
    ) -> TimelineIntegrityReport:
        self._require_role(
            operator_role,
            {
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.AUDITOR,
                OperatorRole.ADMIN,
            },
            "verify timeline integrity",
        )
        self._event(event_id)
        entries = self.repository.list_timeline_entries(event_id)
        issues: list[TimelineIntegrityIssue] = []
        expected_previous = "GENESIS"
        verified_entries = 0
        for entry in entries:
            entry_issues = 0
            if entry.previous_hash != expected_previous:
                issues.append(
                    TimelineIntegrityIssue(
                        entry_id=entry.entry_id,
                        issue=f"previous_hash mismatch: expected {expected_previous}, got {entry.previous_hash}",
                    )
                )
                entry_issues += 1
            calculated = self._timeline_entry_hash(entry)
            if not entry.record_hash:
                issues.append(
                    TimelineIntegrityIssue(
                        entry_id=entry.entry_id, issue="record_hash is missing"
                    )
                )
                entry_issues += 1
            elif calculated != entry.record_hash:
                issues.append(
                    TimelineIntegrityIssue(
                        entry_id=entry.entry_id, issue="record_hash verification failed"
                    )
                )
                entry_issues += 1
            if entry_issues == 0:
                verified_entries += 1
            expected_previous = entry.record_hash or calculated
        return TimelineIntegrityReport(
            event_id=event_id,
            status="verified" if not issues else "failed",
            total_entries=len(entries),
            verified_entries=verified_entries,
            head_hash=entries[-1].record_hash if entries else None,
            issues=issues,
            verified_at=self._now(),
        )

    def create_database_backup(
        self, request: BackupCreateRequest
    ) -> DatabaseBackupRecord:
        self._require_role(
            request.operator_role, {OperatorRole.ADMIN}, "create a database backup"
        )
        return self.repository.create_database_backup(
            label=request.label,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def list_database_backups(
        self, operator_role: OperatorRole
    ) -> list[DatabaseBackupRecord]:
        self._require_role(
            operator_role,
            {OperatorRole.ADMIN, OperatorRole.AUDITOR},
            "list database backups",
        )
        return self.repository.list_database_backups()

    def apply_backup_retention(
        self, request: BackupRetentionRequest
    ) -> BackupRetentionResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "apply the backup retention policy",
        )
        return self.repository.apply_backup_retention(
            keep_latest=request.keep_latest,
            max_age_days=request.max_age_days,
            dry_run=request.dry_run,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def create_audit_archive(
        self, event_id: str, request: AuditArchiveRequest
    ) -> AuditArchiveRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "create an encrypted audit archive",
        )
        dashboard = self.get_dashboard(event_id)
        integrity = self.verify_timeline_integrity(event_id, OperatorRole.ADMIN)
        if integrity.status != "verified":
            raise ValueError(
                "timeline integrity must be verified before audit archive creation"
            )
        bundle = {
            "format": "district-response-audit-archive-v1",
            "event_id": event_id,
            "exported_at": self._now().isoformat(),
            "dashboard": dashboard.model_dump(mode="json"),
            "task_versions": {
                task.task_id: [
                    item.model_dump(mode="json")
                    for item in self.list_task_versions(task.task_id)
                ]
                for task in dashboard.tasks
            },
            "timeline_integrity": integrity.model_dump(mode="json"),
        }
        return self.repository.create_audit_archive(
            event_id=event_id,
            label=request.label,
            bundle=bundle,
            timeline_entries=integrity.total_entries,
            timeline_head_hash=integrity.head_hash,
            retention_days=request.retention_days,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def list_audit_archives(
        self, event_id: str, operator_role: OperatorRole
    ) -> list[AuditArchiveRecord]:
        self._require_role(
            operator_role,
            {OperatorRole.ADMIN, OperatorRole.AUDITOR},
            "list audit archives",
        )
        self._event(event_id)
        return self.repository.list_audit_archives(event_id)

    def verify_audit_archive(
        self, archive_id: str, request: TaskActionRequest
    ) -> AuditArchiveVerificationResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN, OperatorRole.AUDITOR},
            "verify an audit archive",
        )
        return self.repository.verify_audit_archive(
            archive_id=archive_id,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def restore_database_backup(
        self, request: BackupRestoreRequest
    ) -> BackupRestoreResult:
        self._require_role(
            request.operator_role, {OperatorRole.ADMIN}, "run a backup restore drill"
        )
        return self.repository.restore_database_backup(
            backup_id=request.backup_id,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def import_database_backup(
        self, request: BackupImportRequest
    ) -> BackupImportResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "import a cross-host database backup",
        )
        return self.repository.import_database_backup(
            manifest_filename=request.manifest_filename,
            backup_filename=request.backup_filename,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

    def rotate_data_encryption_key(
        self, request: KeyRotationRequest
    ) -> KeyRotationResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "rotate the data encryption key",
        )
        configured = os.getenv("FLOOD_DATA_ENCRYPTION_KEY_NEXT", "").strip()
        if not configured:
            raise ValueError(
                "FLOOD_DATA_ENCRYPTION_KEY_NEXT is required for key rotation"
            )
        backup = self.repository.create_database_backup(
            label=f"pre-key-rotation-{request.label}",
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )
        return self.repository.rotate_data_encryption_key(
            new_key=configured.encode("ascii"),
            backup_id=backup.backup_id,
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )

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
            slots=[
                "触发条件和适用范围",
                "风险对象特征",
                "责任部门和责任岗位",
                "处置动作和执行顺序",
                "例外情况和升级条件",
                "文件版本和条款来源",
            ],
            required_roles=[role.value for role in EvidenceRole],
        )
        return [
            document
            for document in documents
            if str(document.metadata.get("status", "active")).strip().lower()
            not in {"superseded", "expired", "invalid", "quarantined"}
            if str(document.metadata.get("stage", "")).strip().lower()
            not in {"compensation", "recovery", "postmortem"}
        ]

    def _run_retrieval_strategy(
        self, query: str, mode: RetrievalMode
    ) -> tuple[list[RAGDocument], dict]:
        if self.rag_service is None:
            return [], {
                "baseline_source_ids": [],
                "frc_source_ids": [],
                "comparison": {},
            }
        baseline_query = getattr(self.rag_service, "query", None)
        baseline = self._filter_current_policy_documents(
            baseline_query(CorpusType.POLICY, query, top_k=6) if baseline_query else []
        )
        frc = self._query_task_evidence(query)
        baseline_ids = [item.doc_id for item in baseline]
        frc_ids = [item.doc_id for item in frc]
        comparison = {
            "overlap": sorted(set(baseline_ids) & set(frc_ids)),
            "baseline_only": sorted(set(baseline_ids) - set(frc_ids)),
            "frc_only": sorted(set(frc_ids) - set(baseline_ids)),
        }
        if mode in {RetrievalMode.BASELINE_ONLY, RetrievalMode.SHADOW}:
            selected = baseline
        elif mode == RetrievalMode.REVIEW:
            selected = list({item.doc_id: item for item in [*baseline, *frc]}.values())[
                :8
            ]
        else:
            selected = frc
        return selected, {
            "baseline_source_ids": baseline_ids,
            "frc_source_ids": frc_ids,
            "comparison": comparison,
        }

    @staticmethod
    def _filter_current_policy_documents(
        documents: list[RAGDocument],
    ) -> list[RAGDocument]:
        return [
            document
            for document in documents
            if str(document.metadata.get("status", "active")).strip().lower()
            not in {"superseded", "expired", "invalid", "quarantined"}
            and str(document.metadata.get("stage", "")).strip().lower()
            not in {"compensation", "recovery", "postmortem"}
        ]

    @staticmethod
    def _build_task_query(alert, risk_object: EventRiskObject) -> str:
        return " ".join(
            [
                alert.level,
                alert.disaster_type,
                risk_object.object_type,
                risk_object.name,
                risk_object.vulnerability,
                *risk_object.trigger_reasons,
                "触发条件 责任岗位 处置流程 反馈要求 例外升级 条款依据",
            ]
        )

    def _build_task_evidence(
        self, alert, risk_object: EventRiskObject, documents: list[RAGDocument]
    ) -> list[TaskEvidenceRef]:
        evidence = [
            TaskEvidenceRef(
                source_type="alert_snapshot",
                source_id=alert.snapshot_id,
                title=f"{alert.source_department}{alert.disaster_type}预警 V{alert.version}",
                excerpt=f"{alert.level}；影响范围：{alert.affected_area}；{alert.raw_content[:140]}",
                roles=[EvidenceRole.CONDITION],
                document_version=str(alert.version),
                trust_score=1.0,
            ),
            TaskEvidenceRef(
                source_type="risk_object_registry",
                source_id=risk_object.object_id,
                title=f"{risk_object.name}事件对象快照",
                excerpt=(
                    f"{risk_object.object_type}；{risk_object.location}；脆弱性：{risk_object.vulnerability}；"
                    f"责任：{risk_object.responsible_organization}/{risk_object.responsible_role}"
                ),
                roles=[EvidenceRole.OBJECT, EvidenceRole.RESPONSIBILITY],
                document_version="event-snapshot-v1",
                trust_score=1.0,
            ),
        ]
        for document in documents:
            roles = self._document_roles(document)
            if not roles:
                continue
            metadata = document.metadata
            selection = metadata.get("_evidence_selection", {})
            trust_score = (
                selection.get("trust_score") if isinstance(selection, dict) else None
            )
            evidence.append(
                TaskEvidenceRef(
                    source_type="plan_document",
                    source_id=document.doc_id,
                    title=document.title,
                    excerpt=document.content[:180],
                    roles=roles,
                    document_version=self._document_version(document),
                    clause=self._metadata_value(metadata, "section_number", "clause"),
                    trust_score=trust_score
                    if isinstance(trust_score, (int, float))
                    else None,
                    conflict_key=self._metadata_value(
                        metadata, "conflict_key", "decision_key"
                    ),
                    conflict_value=self._metadata_value(
                        metadata, "conflict_value", "decision_value"
                    ),
                    jurisdiction=self._metadata_value(
                        metadata, "jurisdiction", "area_id"
                    ),
                    superseded=str(metadata.get("status", "")).lower()
                    in {"superseded", "expired", "invalid"},
                )
            )
        return evidence

    def _detect_evidence_conflicts(
        self, evidence: list[TaskEvidenceRef]
    ) -> list[EvidenceConflict]:
        groups: dict[str, list[TaskEvidenceRef]] = {}
        for item in evidence:
            if item.superseded or not item.conflict_key or not item.conflict_value:
                continue
            groups.setdefault(item.conflict_key, []).append(item)
        conflicts: list[EvidenceConflict] = []
        for key, items in groups.items():
            values = {item.conflict_value for item in items}
            if len(values) < 2:
                continue
            source_ids = list(dict.fromkeys(item.source_id for item in items))
            if len(source_ids) < 2:
                continue
            conflicts.append(
                EvidenceConflict(
                    conflict_id=self._id("CONFLICT"),
                    field_name=key,
                    conflict_type="DECISION_VALUE",
                    severity="critical",
                    evidence_source_ids=source_ids,
                )
            )
        return conflicts

    @classmethod
    def _document_roles(cls, document: RAGDocument) -> list[EvidenceRole]:
        metadata_roles = document.metadata.get("evidence_roles", [])
        explicit = (
            {str(item) for item in metadata_roles}
            if isinstance(metadata_roles, list)
            else set()
        )
        text = f"{document.title} {document.content}"
        roles: list[EvidenceRole] = []
        keyword_map = {
            EvidenceRole.CONDITION: ("当", "达到", "预警", "阶段", "出现", "持续"),
            EvidenceRole.OBJECT: (
                "下穿",
                "地铁",
                "学校",
                "社区",
                "地下",
                "对象",
                "片区",
            ),
            EvidenceRole.RESPONSIBILITY: (
                "责任",
                "防办",
                "部门",
                "交警",
                "住建",
                "联络员",
                "值班",
            ),
            EvidenceRole.PROCEDURE: (
                "应",
                "需",
                "组织",
                "核查",
                "启动",
                "封控",
                "转移",
                "引导",
                "联动",
            ),
            EvidenceRole.EXCEPTION: (
                "若",
                "如",
                "否则",
                "受阻",
                "超时",
                "不足",
                "争议",
                "必要时",
            ),
        }
        for role, keywords in keyword_map.items():
            if role.value in explicit or any(keyword in text for keyword in keywords):
                roles.append(role)
        version = cls._document_version(document)
        clause = cls._metadata_value(document.metadata, "section_number", "clause")
        if EvidenceRole.ATTRIBUTION.value in explicit or (version and clause):
            roles.append(EvidenceRole.ATTRIBUTION)
        return list(dict.fromkeys(roles))

    @classmethod
    def _build_plan_basis(cls, documents: list[RAGDocument]):
        from .models import PlanBasis

        basis: list[PlanBasis] = []
        seen: set[tuple[str, str, str]] = set()
        for document in documents:
            version = cls._document_version(document)
            clause = cls._metadata_value(document.metadata, "section_number", "clause")
            if not version or not clause:
                continue
            key = (document.title, version, clause)
            if key in seen:
                continue
            seen.add(key)
            basis.append(
                PlanBasis(document=document.title, version=version, clause=clause)
            )
        return basis

    @staticmethod
    def _extract_supported_action(
        documents: list[RAGDocument], risk_object: EventRiskObject
    ) -> str:
        action_markers = (
            "应",
            "需",
            "组织",
            "核查",
            "启动",
            "封控",
            "转移",
            "引导",
            "联动",
        )
        object_markers = [risk_object.object_type]
        if any(keyword in risk_object.object_type for keyword in ("下穿", "隧道")):
            object_markers.extend(["下穿", "积水", "道路"])
        ranked: list[tuple[int, int, int, str]] = []
        for document_index, document in enumerate(documents):
            stage = str(document.metadata.get("stage", "")).lower()
            audience = str(document.metadata.get("audience", "")).lower()
            for sentence_index, sentence in enumerate(
                re.split(r"[。；;\n]+", document.content)
            ):
                cleaned = sentence.strip()
                if len(cleaned) < 8 or not any(
                    marker in cleaned for marker in action_markers
                ):
                    continue
                score = 4 * int(
                    any(marker and marker in cleaned for marker in object_markers)
                )
                score += 2 * int(stage == "response") + int(stage == "warning")
                score += int(audience == "government")
                ranked.append((score, document_index, sentence_index, cleaned))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        sentences: list[str] = []
        for _, _, _, sentence in ranked:
            if sentence not in sentences:
                sentences.append(sentence)
            if len(sentences) >= 3:
                break
        return "；".join(sentences) + ("。" if sentences else "")

    @staticmethod
    def _required_evidence(object_type: str) -> list[str]:
        if any(keyword in object_type for keyword in ("隧道", "下穿", "易涝", "地下")):
            return ["现场照片", "积水深度", "到场时间", "管控状态"]
        if any(keyword in object_type for keyword in ("学校", "医院", "养老", "社区")):
            return ["现场照片", "人员清点", "处置结果", "完成时间"]
        return ["现场照片", "处置结果", "到场时间"]

    @staticmethod
    def _cooperate_roles(object_type: str) -> list[str]:
        if any(keyword in object_type for keyword in ("隧道", "下穿", "道路", "易涝")):
            return ["交警联络员", "排水单位负责人", "属地街道值班员"]
        return ["属地街道值班员", "成员单位联络员"]

    @staticmethod
    def _approval_policy(alert_level: str, action: str) -> ApprovalPolicy:
        high_risk_levels = ("橙", "红", "orange", "red")
        high_risk_actions = ("转移", "封控", "停课", "停工", "公开发布", "资源调度")
        normalized = alert_level.lower()
        if any(level in normalized for level in high_risk_levels) or any(
            item in action for item in high_risk_actions
        ):
            return ApprovalPolicy.COMMANDER_REQUIRED
        return ApprovalPolicy.REVIEWER_REQUIRED

    @staticmethod
    def _document_version(document: RAGDocument) -> str | None:
        return ResponseWorkflowService._metadata_value(
            document.metadata, "document_version", "version", "effective_version"
        )

    @staticmethod
    def _metadata_value(metadata: dict, *keys: str) -> str | None:
        for key in keys:
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _build_grounding_summary(
        evidence: list[TaskEvidenceRef],
        role_coverage: dict[str, bool],
        validation_errors: list[str],
    ) -> str:
        covered = [role for role, value in role_coverage.items() if value]
        plan_sources = [
            item for item in evidence if item.source_type == "plan_document"
        ]
        if validation_errors:
            return f"证据闸门未通过：覆盖 {len(covered)}/6 个角色；" + "；".join(
                validation_errors
            )
        return (
            f"证据闸门通过：6/6 角色完整，使用 {len(plan_sources)} 条预案证据、"
            "1 版专业预警快照和 1 份事件对象责任快照形成草案。"
        )

    def bootstrap_demo(self) -> EventDashboard:
        existing = self.list_events()
        if existing:
            return self.get_dashboard(existing[0].event_id)
        now = self._now()
        from .models import AlertInput, RiskObjectInput

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
