from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..identity import (
    IdentityAssertionError,
    OperatorIdentity,
    development_identity_from_headers,
)
from .idempotency import create_idempotent_route_class

from ..response_workflow.models import (
    AlertAppendRequest,
    AuditArchiveRequest,
    ApprovalRequest,
    BackupCreateRequest,
    BackupImportRequest,
    BackupRestoreRequest,
    BackupRetentionRequest,
    CandidateDiscoveryRequest,
    DeadlineExtensionDecisionRequest,
    DeadlineExtensionRequest,
    DispatchCallbackRequest,
    DocumentImportRequest,
    EventCloseRequest,
    EvidenceConflictResolutionRequest,
    EvidenceFreezeRequest,
    EvidenceManualSupplementRequest,
    EventCreateRequest,
    FeedbackRequest,
    FeatureFlagUpdateRequest,
    RiskObjectBatchRequest,
    RiskObjectVerificationRequest,
    ReviewDraftRequest,
    ScenarioEvaluationRequest,
    KeyRotationRequest,
    LegacyMigrationRequest,
    OperatorRole,
    OutboxProcessRequest,
    TaskActionRequest,
    TaskAssignmentRequest,
    TaskCreateRequest,
    TaskDraftGenerationRequest,
    TaskUpdateRequest,
)


def create_response_router(system_provider: Callable[[], Any]) -> APIRouter:
    def resolve_identity(http_request: Request) -> OperatorIdentity:
        identity = getattr(http_request.state, "response_identity", None)
        if identity is None:
            try:
                identity = development_identity_from_headers(http_request.headers)
                if identity is None:
                    identity = system_provider().response_identity.verify(
                        http_request.headers,
                        method=http_request.method,
                        path=http_request.url.path,
                    )
            except IdentityAssertionError as exc:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "code": "UNAUTHORIZED",
                        "message": str(exc),
                        "retryable": False,
                    },
                ) from exc
            http_request.state.response_identity = identity
        callback_path = "/response/simulation/dispatch-callbacks"
        if identity.operator_role == OperatorRole.EXTERNAL_SERVICE and not (
            http_request.method.upper() == "POST"
            and http_request.url.path == callback_path
        ):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "FORBIDDEN",
                    "message": "external service identities are restricted to the simulation callback endpoint",
                    "retryable": False,
                },
            )
        if http_request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            idempotency_key = http_request.headers.get("idempotency-key", "").strip()
            if not idempotency_key:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "VALIDATION_ERROR",
                        "message": "Idempotency-Key is required for Core API write requests",
                        "retryable": False,
                    },
                )
        return identity

    def authenticate(http_request: Request) -> OperatorIdentity:
        return resolve_identity(http_request)

    router = APIRouter(
        prefix="/response",
        tags=["district-response-workflow"],
        dependencies=[Depends(authenticate)],
        route_class=create_idempotent_route_class(system_provider, resolve_identity),
    )

    def service():
        return system_provider().response_workflow

    def invoke(call):
        try:
            return call()
        except PermissionError as exc:
            raise HTTPException(
                status_code=403,
                detail={"code": "FORBIDDEN", "message": str(exc), "retryable": False},
            ) from exc
        except LookupError as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": "NOT_FOUND", "message": str(exc), "retryable": False},
            ) from exc
        except ValueError as exc:
            message = str(exc)
            if "version conflict" in message:
                code, status = "VERSION_CONFLICT", 409
            elif (
                "blocked by rules" in message
                or "blocked by the current rule" in message
            ):
                code, status = "RULE_HARD_BLOCK", 409
            elif any(
                token in message
                for token in ("current status", "cannot transition", "pending approval")
            ):
                code, status = "STATE_CONFLICT", 409
            else:
                code, status = "VALIDATION_ERROR", 400
            raise HTTPException(
                status_code=status,
                detail={"code": code, "message": message, "retryable": False},
            ) from exc

    def identity_for(http_request: Request) -> OperatorIdentity:
        return http_request.state.response_identity

    def bind_payload_identity(
        http_request: Request,
        payload: Any,
        *,
        minimum_assurance: str = "aal1",
    ) -> OperatorIdentity:
        identity = identity_for(http_request)
        try:
            identity.require_assurance(minimum_assurance)
        except IdentityAssertionError as exc:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "ASSURANCE_REQUIRED",
                    "message": str(exc),
                    "retryable": False,
                },
            ) from exc
        if (
            payload.operator_id != identity.operator_id
            or payload.operator_role != identity.operator_role
        ):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "IDENTITY_MISMATCH",
                    "message": "request operator does not match the trusted identity assertion",
                    "retryable": False,
                },
            )
        if payload.terminal_id != identity.terminal_id:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "TERMINAL_MISMATCH",
                    "message": "request terminal does not match the trusted identity assertion",
                    "retryable": False,
                },
            )
        return identity

    @router.get("/events")
    def list_events(http_request: Request):
        return service().list_events()

    @router.post("/events")
    def create_event(request: EventCreateRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().create_event(request))

    @router.post("/events/bootstrap-demo")
    def bootstrap_demo(http_request: Request):
        return invoke(service().bootstrap_demo)

    @router.get("/events/{event_id}")
    def get_event_dashboard(event_id: str, http_request: Request):
        return invoke(
            lambda: service().get_dashboard_for_identity(
                event_id, identity_for(http_request)
            )
        )

    @router.get("/events/{event_id}/integrity")
    def verify_event_integrity(event_id: str, http_request: Request):
        return invoke(
            lambda: service().verify_timeline_integrity(
                event_id, identity_for(http_request).operator_role
            )
        )

    @router.post("/events/{event_id}/alerts")
    def append_alert(event_id: str, request: AlertAppendRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().append_alert(event_id, request))

    @router.post("/events/{event_id}/risk-objects")
    def add_risk_objects(
        event_id: str, request: RiskObjectBatchRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().add_risk_objects(event_id, request))

    @router.post("/events/{event_id}/risk-objects/discover")
    def discover_risk_objects(
        event_id: str, request: CandidateDiscoveryRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().discover_risk_objects(event_id, request))

    @router.get("/events/{event_id}/candidate-runs")
    def list_candidate_runs(event_id: str, http_request: Request):
        return invoke(lambda: service().list_candidate_runs(event_id))

    @router.get("/events/{event_id}/evidence-packages")
    def list_evidence_packages(
        event_id: str, http_request: Request, object_id: str | None = None
    ):
        return invoke(
            lambda: service().list_evidence_packages(event_id, object_id=object_id)
        )

    @router.get("/evidence-packages/{package_id}")
    def get_evidence_package(package_id: str, http_request: Request):
        return invoke(lambda: service().get_evidence_package(package_id))

    @router.get("/events/{event_id}/timeline")
    def list_event_timeline(event_id: str, http_request: Request):
        return invoke(lambda: service().get_dashboard(event_id).timeline)

    @router.get("/events/{event_id}/audit-trail")
    def list_event_audit_trail(event_id: str, http_request: Request):
        identity = identity_for(http_request)
        return invoke(
            lambda: {
                "integrity": service().verify_timeline_integrity(
                    event_id, identity.operator_role
                ),
                "entries": service().get_dashboard(event_id).timeline,
            }
        )

    @router.post("/evidence-packages/{package_id}/conflicts/{conflict_id}/resolve")
    def resolve_evidence_conflict(
        package_id: str,
        conflict_id: str,
        request: EvidenceConflictResolutionRequest,
        http_request: Request,
    ):
        bind_payload_identity(http_request, request)
        return invoke(
            lambda: service().resolve_evidence_conflict(
                package_id, conflict_id, request
            )
        )

    @router.post("/evidence-packages/{package_id}/manual-evidence")
    def supplement_evidence_package(
        package_id: str, request: EvidenceManualSupplementRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(
            lambda: service().supplement_evidence_package(package_id, request)
        )

    @router.post("/evidence-packages/{package_id}/freeze")
    def freeze_evidence_package(
        package_id: str, request: EvidenceFreezeRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().freeze_evidence_package(package_id, request))

    @router.post("/events/{event_id}/risk-objects/{object_id}/verify")
    def verify_risk_object(
        event_id: str,
        object_id: str,
        request: RiskObjectVerificationRequest,
        http_request: Request,
    ):
        bind_payload_identity(http_request, request)
        return invoke(
            lambda: service().verify_risk_object(event_id, object_id, request)
        )

    @router.post("/events/{event_id}/tasks")
    def create_task(event_id: str, request: TaskCreateRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().create_task(event_id, request))

    @router.post("/events/{event_id}/risk-objects/{object_id}/task-draft")
    def generate_task_draft(
        event_id: str,
        object_id: str,
        request: TaskDraftGenerationRequest,
        http_request: Request,
    ):
        bind_payload_identity(http_request, request)
        return invoke(
            lambda: service().generate_task_draft(event_id, object_id, request)
        )

    @router.patch("/tasks/{task_id}")
    def update_task(task_id: str, request: TaskUpdateRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().update_task(task_id, request))

    @router.post("/tasks/{task_id}/submit")
    def submit_task(task_id: str, request: TaskActionRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().submit_task(task_id, request))

    @router.post("/tasks/{task_id}/decision")
    def decide_task(task_id: str, request: ApprovalRequest, http_request: Request):
        task = service().repository.get_response_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=404, detail=f"response task not found: {task_id}"
            )
        required_assurance = (
            "aal2" if task.approval_policy.value == "commander_required" else "aal1"
        )
        bind_payload_identity(
            http_request, request, minimum_assurance=required_assurance
        )
        return invoke(lambda: service().decide_task(task_id, request))

    @router.post("/tasks/{task_id}/acknowledge")
    def acknowledge_task(
        task_id: str, request: TaskActionRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().acknowledge_task(task_id, request))

    @router.post("/tasks/{task_id}/assign")
    def assign_task(
        task_id: str, request: TaskAssignmentRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().assign_task(task_id, request))

    @router.get("/tasks/{task_id}/versions")
    def list_task_versions(task_id: str, http_request: Request):
        return invoke(lambda: service().list_task_versions(task_id))

    @router.get("/tasks/{task_id}/transitions")
    def list_task_transitions(task_id: str, http_request: Request):
        return invoke(lambda: service().list_task_transitions(task_id))

    @router.post("/tasks/{task_id}/start")
    def start_task(task_id: str, request: TaskActionRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().start_task(task_id, request))

    @router.post("/tasks/{task_id}/feedback")
    def submit_feedback(task_id: str, request: FeedbackRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().submit_feedback(task_id, request))

    @router.post("/tasks/{task_id}/deadline-extensions")
    def request_deadline_extension(
        task_id: str, request: DeadlineExtensionRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().request_deadline_extension(task_id, request))

    @router.post("/deadline-extensions/{extension_id}/decision")
    def decide_deadline_extension(
        extension_id: str,
        request: DeadlineExtensionDecisionRequest,
        http_request: Request,
    ):
        bind_payload_identity(http_request, request)
        return invoke(
            lambda: service().decide_deadline_extension(extension_id, request)
        )

    @router.post("/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, request: TaskActionRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().cancel_task(task_id, request))

    @router.post("/tasks/{task_id}/take-over")
    def take_over_task(task_id: str, request: TaskActionRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().take_over_task(task_id, request))

    @router.post("/tasks/{task_id}/verify-completion")
    def verify_completion(
        task_id: str, approved: bool, request: TaskActionRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().verify_completion(task_id, approved, request))

    @router.post("/tasks/{task_id}/waive")
    def waive_task(task_id: str, request: TaskActionRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().waive_task(task_id, request))

    @router.post("/events/{event_id}/deadline-sweep")
    def run_deadline_sweep(
        event_id: str, request: TaskActionRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().run_deadline_sweep(event_id, request))

    @router.post("/events/{event_id}/close")
    def close_event(event_id: str, request: EventCloseRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().close_event(event_id, request))

    @router.post("/events/{event_id}/review-draft")
    def generate_review_draft(
        event_id: str, request: ReviewDraftRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().generate_review_draft(event_id, request))

    @router.post("/events/{event_id}/replay")
    def replay_event(event_id: str, request: TaskActionRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().replay_event(event_id, request))

    @router.post("/events/{event_id}/reports")
    def generate_event_report(
        event_id: str, request: ScenarioEvaluationRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().run_scenario_evaluation(event_id, request))

    @router.post("/events/{event_id}/scenario-evaluation")
    def run_scenario_evaluation(
        event_id: str, request: ScenarioEvaluationRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().run_scenario_evaluation(event_id, request))

    @router.get("/events/{event_id}/scenario-reports")
    def list_scenario_reports(event_id: str, http_request: Request):
        return invoke(lambda: service().list_scenario_reports(event_id))

    @router.post("/security/backups")
    def create_database_backup(request: BackupCreateRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().create_database_backup(request))

    @router.get("/security/backups")
    def list_database_backups(http_request: Request):
        return invoke(
            lambda: service().list_database_backups(
                identity_for(http_request).operator_role
            )
        )

    @router.post("/security/backups/restore")
    def restore_database_backup(request: BackupRestoreRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().restore_database_backup(request))

    @router.post("/security/backups/import")
    def import_database_backup(request: BackupImportRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().import_database_backup(request))

    @router.post("/security/backups/retention")
    def apply_backup_retention(request: BackupRetentionRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().apply_backup_retention(request))

    @router.post("/events/{event_id}/audit-archives")
    def create_audit_archive(
        event_id: str, request: AuditArchiveRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().create_audit_archive(event_id, request))

    @router.get("/events/{event_id}/audit-archives")
    def list_audit_archives(event_id: str, http_request: Request):
        return invoke(
            lambda: service().list_audit_archives(
                event_id, identity_for(http_request).operator_role
            )
        )

    @router.post("/security/audit-archives/{archive_id}/verify")
    def verify_audit_archive(
        archive_id: str, request: TaskActionRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().verify_audit_archive(archive_id, request))

    @router.post("/security/keys/rotate")
    def rotate_data_encryption_key(request: KeyRotationRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().rotate_data_encryption_key(request))

    @router.get("/configuration/feature-flags")
    def list_feature_flags(http_request: Request):
        return invoke(service().list_feature_flags)

    @router.put("/configuration/feature-flags")
    def update_feature_flag(request: FeatureFlagUpdateRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().update_feature_flag(request))

    @router.get("/dispatch/outbox")
    def list_outbox(http_request: Request, event_id: str | None = None):
        return invoke(lambda: service().list_outbox_messages(event_id=event_id))

    @router.post("/dispatch/outbox/process")
    def process_outbox(request: OutboxProcessRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().process_outbox(request))

    @router.get("/dispatch/outbox/{message_id}/callbacks")
    def list_dispatch_callbacks(message_id: str, http_request: Request):
        return invoke(lambda: service().list_dispatch_callbacks(message_id))

    @router.post("/simulation/dispatch-callbacks")
    def ingest_simulated_dispatch_callback(
        request: DispatchCallbackRequest, http_request: Request
    ):
        bind_payload_identity(http_request, request)
        header_key = http_request.headers.get("idempotency-key", "").strip()
        if header_key != request.idempotency_key:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "callback idempotency key must match the Idempotency-Key header",
                    "retryable": False,
                },
            )
        return invoke(lambda: service().ingest_simulated_dispatch_callback(request))

    @router.get("/migration/status")
    def migration_status(http_request: Request):
        return invoke(service().migration_status)

    @router.get("/documents")
    def list_document_versions(http_request: Request, document_id: str | None = None):
        return invoke(lambda: service().list_document_versions(document_id))

    @router.get("/contracts/task-schema")
    def task_schema_contract(http_request: Request):
        return invoke(service().task_schema_contract)

    @router.get("/contracts/rule-set")
    def rule_set_contract(http_request: Request):
        return invoke(service().rule_set_contract)

    @router.post("/documents")
    def register_document(request: DocumentImportRequest, http_request: Request):
        bind_payload_identity(http_request, request)
        return invoke(lambda: service().register_document(request))

    @router.post("/migration/inventory")
    def run_migration_inventory(request: LegacyMigrationRequest, http_request: Request):
        bind_payload_identity(http_request, request, minimum_assurance="aal2")
        return invoke(lambda: service().run_legacy_migration_inventory(request))

    @router.get("/legacy/events/{legacy_event_id}")
    def read_legacy_event(legacy_event_id: str, http_request: Request):
        identity = identity_for(http_request)
        trace_id = http_request.headers.get("X-Trace-Id") or f"trace-{legacy_event_id}"
        return invoke(
            lambda: service().read_legacy_event(
                legacy_event_id,
                operator_id=identity.operator_id,
                terminal_id=identity.terminal_id,
                trace_id=trace_id,
            )
        )

    return router
