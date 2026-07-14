from __future__ import annotations

from contextlib import asynccontextmanager
from collections import deque
import time
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from .config import load_settings
from .http.v3_router import create_v3_router
from .http.body_limit import RequestBodyLimitMiddleware
from .http.response_router import create_response_router
from .infrastructure.sse import repeated_snapshot_stream
from .system import FloodWarningSystem
from .response_workflow.risk_object_ingestion import configured_max_file_bytes
from .transport_security import TransportSecurityMiddleware
from .v2.llm_gateway import LLMGenerationError
from .v2.models import (
    AdvisoryRequest,
    BatchProposalResolutionRequest,
    DatasetBuildRequest,
    DatasetFetchRequest,
    DatasetSyncRequest,
    EntityProfileUpsertRequest,
    EventCreateRequest,
    ObservationBatchRequest,
    ProposalDraftUpdateRequest,
    ProposalResolutionRequest,
    RAGDocumentImportRequest,
    ReplayRequest,
    ResourceStatusUpdateRequest,
    SimulationUpdateRequest,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
)
from .v2.security import (
    AuthorizationError,
    ensure_operator_role,
    list_operator_capabilities,
    normalize_operator_role,
)


settings = load_settings()
api_request_durations_ms: deque[float] = deque(maxlen=2000)
api_request_total = 0
api_error_total = 0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    system.start_background_services()
    try:
        yield
    finally:
        system.stop_background_services()


app = FastAPI(title=settings.title, version=settings.version, lifespan=lifespan)
app.add_middleware(
    TransportSecurityMiddleware,
    require_https=settings.require_https,
    trust_proxy_headers=settings.trust_proxy_headers,
)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_bytes=(configured_max_file_bytes() * 4 // 3) + (1024 * 1024),
    paths={
        "/response/risk-objects/file-imports",
        "/api/v1/risk-objects/file-imports",
    },
)
system = FloodWarningSystem(settings.db_path)
production = system.production_platform
app.include_router(create_v3_router(lambda: system))
app.include_router(create_response_router(lambda: system))


@app.get("/health", tags=["operations"])
def health():
    return {"status": "ok", "service": settings.title, "version": settings.version}


@app.get("/ready", tags=["operations"])
def readiness():
    try:
        with system.repository._connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return {"status": "ready", "database": "ok"}
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail={"code": "DATABASE_NOT_READY", "message": str(exc)}
        ) from exc


@app.get("/metrics", response_class=PlainTextResponse, tags=["operations"])
def metrics():
    events = system.repository.list_response_events()
    active = sum(item.status.value != "closed" for item in events)
    outbox = system.repository.list_outbox_messages(limit=10000)
    pending = sum(item.status.value == "pending" for item in outbox)
    failed = sum(item.status.value == "failed" for item in outbox)
    sent = sum(item.status.value == "sent" for item in outbox)
    partially_sent = sum(item.status.value == "partially_sent" for item in outbox)
    manual_takeover = sum(item.status.value == "manual_takeover" for item in outbox)
    dispatch_callbacks = sum(item.callback_count for item in outbox)
    idempotency_records = system.repository.count_idempotency_records()
    rule_evaluations = system.repository.list_rule_evaluations()
    rule_pass = sum(item.overall_outcome.value == "PASS" for item in rule_evaluations)
    rule_warning = sum(
        item.overall_outcome.value == "SOFT_WARNING" for item in rule_evaluations
    )
    rule_block = sum(
        item.overall_outcome.value == "HARD_BLOCK" for item in rule_evaluations
    )
    timeline = [
        entry
        for event in events
        for entry in system.repository.list_timeline_entries(event.event_id)
    ]
    takeovers = sum(item.action == "task_taken_over" for item in timeline)
    stale_objects = sum(
        item.stale
        for event in events
        for item in system.repository.list_event_risk_objects(event.event_id)
    )
    registry_metrics = system.repository.risk_object_registry_metrics()
    durations = sorted(api_request_durations_ms)

    def percentile(fraction: float) -> float:
        if not durations:
            return 0.0
        return durations[min(len(durations) - 1, int((len(durations) - 1) * fraction))]

    return "\n".join(
        (
            "# HELP flood_response_events_total Response events persisted by the core workflow.",
            "# TYPE flood_response_events_total gauge",
            f"flood_response_events_total {len(events)}",
            "# HELP flood_response_events_active Active response events.",
            "# TYPE flood_response_events_active gauge",
            f"flood_response_events_active {active}",
            "# HELP flood_response_outbox_pending Pending simulated dispatch messages.",
            "# TYPE flood_response_outbox_pending gauge",
            f"flood_response_outbox_pending {pending}",
            "# HELP flood_response_outbox_failed Failed-closed simulated dispatch messages.",
            "# TYPE flood_response_outbox_failed gauge",
            f"flood_response_outbox_failed {failed}",
            "# HELP flood_response_outbox_sent Successfully processed simulated dispatch messages.",
            "# TYPE flood_response_outbox_sent gauge",
            f"flood_response_outbox_sent {sent}",
            "# HELP flood_response_outbox_partially_sent Partially accepted simulated dispatch messages.",
            "# TYPE flood_response_outbox_partially_sent gauge",
            f"flood_response_outbox_partially_sent {partially_sent}",
            "# HELP flood_response_outbox_manual_takeover Dispatch messages requiring manual takeover.",
            "# TYPE flood_response_outbox_manual_takeover gauge",
            f"flood_response_outbox_manual_takeover {manual_takeover}",
            "# HELP flood_response_dispatch_callbacks_total Unique simulated callbacks recorded.",
            "# TYPE flood_response_dispatch_callbacks_total gauge",
            f"flood_response_dispatch_callbacks_total {dispatch_callbacks}",
            "# HELP flood_response_idempotency_records Persisted Core API idempotency records by status.",
            "# TYPE flood_response_idempotency_records gauge",
            *(
                f'flood_response_idempotency_records{{status="{status}"}} {count}'
                for status, count in sorted(idempotency_records.items())
            ),
            "# HELP flood_response_rule_evaluations Rule outcomes by result.",
            "# TYPE flood_response_rule_evaluations gauge",
            f'flood_response_rule_evaluations{{outcome="PASS"}} {rule_pass}',
            f'flood_response_rule_evaluations{{outcome="SOFT_WARNING"}} {rule_warning}',
            f'flood_response_rule_evaluations{{outcome="HARD_BLOCK"}} {rule_block}',
            "# HELP flood_response_manual_takeovers_total Audited manual takeover actions.",
            "# TYPE flood_response_manual_takeovers_total gauge",
            f"flood_response_manual_takeovers_total {takeovers}",
            "# HELP flood_response_stale_objects Current stale event risk objects.",
            "# TYPE flood_response_stale_objects gauge",
            f"flood_response_stale_objects {stale_objects}",
            "# HELP flood_risk_object_registry Registry master-data objects by status.",
            "# TYPE flood_risk_object_registry gauge",
            f'flood_risk_object_registry{{status="active"}} {registry_metrics["active"]}',
            f'flood_risk_object_registry{{status="inactive"}} {registry_metrics["inactive"]}',
            "# HELP flood_risk_object_registry_stale_candidate_runs Candidate runs invalidated by registry changes.",
            "# TYPE flood_risk_object_registry_stale_candidate_runs gauge",
            "flood_risk_object_registry_stale_candidate_runs "
            f"{registry_metrics['stale_candidate_runs']}",
            "# HELP flood_risk_object_registry_quarantined_rows Rows rejected by governed registry imports.",
            "# TYPE flood_risk_object_registry_quarantined_rows gauge",
            "flood_risk_object_registry_quarantined_rows "
            f"{registry_metrics['quarantined_rows']}",
            "# HELP flood_candidate_object_lists_frozen Immutable confirmed candidate lists.",
            "# TYPE flood_candidate_object_lists_frozen gauge",
            "flood_candidate_object_lists_frozen "
            f"{registry_metrics['frozen_candidate_lists']}",
            "# HELP flood_api_requests_total Requests observed by the application middleware.",
            "# TYPE flood_api_requests_total counter",
            f"flood_api_requests_total {api_request_total}",
            "# HELP flood_api_errors_total HTTP 5xx responses observed by the application middleware.",
            "# TYPE flood_api_errors_total counter",
            f"flood_api_errors_total {api_error_total}",
            "# HELP flood_api_latency_ms Recent in-process request latency percentiles.",
            "# TYPE flood_api_latency_ms gauge",
            f'flood_api_latency_ms{{quantile="0.50"}} {percentile(0.50):.3f}',
            f'flood_api_latency_ms{{quantile="0.95"}} {percentile(0.95):.3f}',
            "",
        )
    )


@app.middleware("http")
async def rewrite_unified_agent_twin_paths(request, call_next):
    """Expose unified public prefixes while keeping existing internal routes stable."""

    global api_request_total, api_error_total
    started = time.perf_counter()
    correlation_id = request.headers.get("x-correlation-id", "").strip() or uuid4().hex
    request.state.correlation_id = correlation_id
    path = request.scope.get("path", "")
    alias_pairs = (
        ("/agent-twin", "/v3"),
        ("/platform", "/v2"),
        ("/api/v1", "/response"),
    )
    for public_prefix, internal_prefix in alias_pairs:
        if path == public_prefix:
            request.scope["path"] = internal_prefix
            break
        if path.startswith(f"{public_prefix}/"):
            request.scope["path"] = f"{internal_prefix}{path[len(public_prefix) :]}"
            break
    response = await call_next(request)
    api_request_total += 1
    if response.status_code >= 500:
        api_error_total += 1
    api_request_durations_ms.append((time.perf_counter() - started) * 1000)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


def _resolve_operator_role(
    *,
    explicit_role: str | None = None,
    header_role: str | None = None,
    action: str | None = None,
) -> str:
    role = normalize_operator_role(explicit_role or header_role)
    if action is not None:
        ensure_operator_role(action, role)
    return role


@app.get("/v2/security/capabilities")
def get_v2_operator_capabilities(operator_role: str | None = None):
    try:
        return list_operator_capabilities(operator_role)
    except AuthorizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/supervisor/status")
def get_v2_supervisor_status():
    return system.background_services_status()


@app.get("/v2/alerts")
def list_v2_operational_alerts(
    event_id: str | None = None,
    severity: str | None = None,
    source_type: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
    limit: int = 50,
):
    return production.list_operational_alerts(
        event_id=event_id,
        severity=severity,
        source_type=source_type,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=limit,
    )


@app.get("/v2/audit/records")
def list_v2_audit_records(
    event_id: str | None = None,
    severity: str | None = None,
    source_type: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
    limit: int = 100,
):
    return production.list_audit_records(
        event_id=event_id,
        severity=severity,
        source_type=source_type,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=limit,
    )


@app.get("/v2/archive/status")
def get_v2_archive_status():
    return production.get_archive_status()


@app.post("/v2/archive/run")
def run_v2_archive_cycle(
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="archive_run")
        return production.run_archive_cycle()
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/v2/admin/dataset/status")
def get_v2_dataset_status():
    return system.dataset_service.get_status()


@app.get("/v2/admin/dataset/jobs")
def list_v2_dataset_jobs():
    return system.dataset_service.list_jobs()


@app.post("/v2/admin/dataset/fetch")
def fetch_v2_dataset_sources(
    request: DatasetFetchRequest,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="dataset_manage")
        return system.dataset_service.start_fetch_sources(request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v2/admin/dataset/build")
def build_v2_dataset(
    request: DatasetBuildRequest,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="dataset_manage")
        return system.dataset_service.start_build(request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v2/admin/dataset/validate")
def validate_v2_dataset(
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="dataset_manage")
        return system.dataset_service.start_validate()
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v2/admin/dataset/sync-demo-db")
def sync_v2_dataset(
    request: DatasetSyncRequest,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="dataset_manage")
        return system.dataset_service.start_sync(request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v2/admin/dataset/jobs/{job_id}/cancel")
def cancel_v2_dataset_job(
    job_id: str,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="dataset_manage")
        return system.dataset_service.cancel_job(job_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/admin/dataset/jobs/{job_id}/retry")
def retry_v2_dataset_job(
    job_id: str,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="dataset_manage")
        return system.dataset_service.retry_job(job_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/events")
def create_v2_event(
    request: EventCreateRequest,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="event_create")
        return production.create_event(request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/events/{event_id}/observations")
def ingest_v2_observations(
    event_id: str,
    request: ObservationBatchRequest,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="event_ingest")
        return production.ingest_observations(event_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/events/{event_id}/simulation-updates")
def ingest_v2_simulation_update(
    event_id: str,
    request: SimulationUpdateRequest,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="simulation_ingest")
        return production.ingest_simulation_update(event_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/hazard-state")
def get_v2_hazard_state(event_id: str):
    try:
        return production.get_hazard_state(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/proposals/pending")
def list_v2_pending_regional_proposals():
    return production.get_pending_regional_proposals_snapshot()


@app.get("/v2/events/{event_id}/regional-proposals")
def list_v2_regional_proposals(event_id: str, status: str | None = None):
    try:
        statuses = [status] if status else None
        return production.list_regional_proposals(event_id, statuses=statuses)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/regional-analysis-packages")
def list_v2_regional_analysis_packages(event_id: str, include_pending: bool = True):
    try:
        return production.list_regional_analysis_packages(
            event_id, include_pending=include_pending
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/regional-analysis-packages/pending")
def get_v2_pending_regional_analysis_package(event_id: str):
    try:
        return production.get_pending_regional_analysis_package(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/v2/proposals/{proposal_id}/draft")
def update_v2_regional_proposal_draft(
    proposal_id: str,
    request: ProposalDraftUpdateRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_draft_edit"
        )
        return production.update_regional_proposal_draft(proposal_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/proposals/{proposal_id}/approve")
def approve_v2_regional_proposal(
    proposal_id: str,
    request: ProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.approve_regional_proposal(proposal_id, request)
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=f"{exc.code}: {exc}") from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/proposals/{proposal_id}/reject")
def reject_v2_regional_proposal(
    proposal_id: str,
    request: ProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.reject_regional_proposal(proposal_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/regional-analysis-packages/{package_id}/approve")
def approve_v2_regional_analysis_package(
    package_id: str,
    request: ProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.approve_regional_analysis_package(package_id, request)
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=f"{exc.code}: {exc}") from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/regional-analysis-packages/{package_id}/reject")
def reject_v2_regional_analysis_package(
    package_id: str,
    request: ProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.reject_regional_analysis_package(package_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/proposals/stream")
async def stream_v2_pending_regional_proposals():
    return StreamingResponse(
        repeated_snapshot_stream(
            production.get_pending_regional_proposals_snapshot,
            version_getter=lambda snapshot: snapshot.queue_version,
        ),
        media_type="text/event-stream",
    )


@app.get("/v2/entities/{entity_id}/impact")
def get_v2_entity_impact(entity_id: str, event_id: str | None = None):
    try:
        return production.get_entity_impact(entity_id, event_id=event_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/advisories/generate")
def generate_v2_advisory(request: AdvisoryRequest):
    try:
        return production.generate_advisory(request)
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=f"{exc.code}: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/copilot/sessions/bootstrap")
def bootstrap_v2_copilot_session(request: V2CopilotSessionRequest):
    try:
        return production.bootstrap_copilot_session(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/copilot/sessions/{session_id}")
def get_v2_copilot_session(session_id: str):
    try:
        return production.get_copilot_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/daily-reports")
def list_v2_daily_reports(event_id: str):
    try:
        return production.list_daily_reports(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/episode-summaries")
def list_v2_event_episode_summaries(event_id: str):
    try:
        return production.list_episode_summaries(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/long-term-memory")
def list_v2_long_term_memory(event_id: str):
    try:
        return production.list_long_term_memories(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/copilot/sessions/{session_id}/memory")
def get_v2_copilot_session_memory(session_id: str):
    try:
        session = production.get_copilot_session(session_id)
        return production.get_memory_bundle(session_id, session.event.event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v2/copilot/sessions/{session_id}/messages")
def send_v2_copilot_message(session_id: str, request: V2CopilotMessageRequest):
    try:
        return production.send_copilot_message(session_id, request)
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=f"{exc.code}: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/copilot/sessions/{session_id}/proposals/{proposal_id}/approve")
def approve_v2_copilot_proposal(
    session_id: str,
    proposal_id: str,
    request: ProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.approve_copilot_proposal(session_id, proposal_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/copilot/sessions/{session_id}/proposals/{proposal_id}/reject")
def reject_v2_copilot_proposal(
    session_id: str,
    proposal_id: str,
    request: ProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.reject_copilot_proposal(session_id, proposal_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/copilot/sessions/{session_id}/proposals/batch-approve")
def batch_approve_v2_copilot_proposals(
    session_id: str,
    request: BatchProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.batch_approve_copilot_proposals(session_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/copilot/sessions/{session_id}/proposals/batch-reject")
def batch_reject_v2_copilot_proposals(
    session_id: str,
    request: BatchProposalResolutionRequest,
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="proposal_resolve"
        )
        return production.batch_reject_copilot_proposals(session_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/admin/entity-profiles")
def list_v2_entity_profiles(area_id: str | None = None, entity_type: str | None = None):
    try:
        return production.list_entity_profiles(area_id=area_id, entity_type=entity_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/admin/entity-profiles/{entity_id}")
def get_v2_admin_entity_profile(entity_id: str):
    try:
        return production.get_entity_profile(entity_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v2/admin/entity-profiles")
def create_v2_admin_entity_profile(request: EntityProfileUpsertRequest):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="runtime_admin_write"
        )
        return production.save_entity_profile(request.profile)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/v2/admin/entity-profiles/{entity_id}")
def update_v2_admin_entity_profile(entity_id: str, request: EntityProfileUpsertRequest):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="runtime_admin_write"
        )
        if request.profile.entity_id != entity_id:
            raise ValueError("entity_id in path and payload must match.")
        return production.save_entity_profile(request.profile)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v2/admin/entity-profiles/{entity_id}")
def delete_v2_admin_entity_profile(
    entity_id: str,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(
            header_role=x_operator_role, action="runtime_admin_write"
        )
        production.delete_entity_profile(entity_id)
        return {"status": "deleted", "entity_id": entity_id}
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/admin/areas/{area_id}/resource-status")
def get_v2_area_resource_status(area_id: str):
    try:
        return production.get_area_resource_status_view(area_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/v2/admin/areas/{area_id}/resource-status")
def update_v2_area_resource_status(area_id: str, request: ResourceStatusUpdateRequest):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="runtime_admin_write"
        )
        if request.resource_status.area_id != area_id:
            raise ValueError("area_id in path and payload must match.")
        production.save_area_resource_status(request.resource_status)
        return production.get_area_resource_status_view(area_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/admin/events/{event_id}/resource-status")
def get_v2_event_resource_status(event_id: str):
    try:
        return production.get_event_resource_status_view(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/v2/admin/events/{event_id}/resource-status")
def update_v2_event_resource_status(
    event_id: str, request: ResourceStatusUpdateRequest
):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="runtime_admin_write"
        )
        production.save_event_resource_status(event_id, request.resource_status)
        return production.get_event_resource_status_view(event_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v2/admin/events/{event_id}/resource-status")
def delete_v2_event_resource_status(
    event_id: str,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(
            header_role=x_operator_role, action="runtime_admin_write"
        )
        production.delete_event_resource_status(event_id)
        return {"status": "deleted", "event_id": event_id}
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/admin/rag-documents")
def list_v2_rag_documents():
    return production.list_rag_documents()


@app.post("/v2/admin/rag-documents/import")
def import_v2_rag_documents(request: RAGDocumentImportRequest):
    try:
        _resolve_operator_role(
            explicit_role=request.operator_role, action="runtime_admin_write"
        )
        documents = production.import_rag_documents(request)
        return {
            "status": "imported",
            "document_count": len(documents),
            "documents": documents,
        }
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/admin/rag-documents/reload")
def reload_v2_rag_documents(
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(
            header_role=x_operator_role, action="runtime_admin_write"
        )
        documents = production.reload_rag_documents()
        return {
            "status": "reloaded",
            "document_count": len(documents),
            "documents": documents,
        }
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/agent-status")
def get_v2_agent_status(event_id: str):
    try:
        return production.get_agent_status(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/experience-context")
def get_v2_experience_context(event_id: str):
    try:
        return production.get_experience_context(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/entities/{entity_id}/strategy-history")
def get_v2_strategy_history(entity_id: str):
    try:
        return production.get_strategy_history(entity_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/agent-metrics")
def get_v2_agent_metrics():
    return production.get_agent_metrics()


@app.get("/v2/events/{event_id}/decision-report")
def get_v2_decision_report(event_id: str):
    try:
        return production.get_decision_report(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/evaluation/benchmarks")
def list_v2_evaluation_benchmarks():
    return production.list_evaluation_benchmarks()


@app.post("/v2/evaluation/run")
def run_v2_evaluation(
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="evaluation_run")
        return production.run_evaluation()
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/v2/evaluation/reports/{report_id}")
def get_v2_evaluation_report(report_id: str):
    try:
        return production.get_evaluation_report(report_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v2/evaluation/reports/{report_id}/replay")
def replay_v2_evaluation_report(
    report_id: str,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="evaluation_run")
        return production.replay_evaluation_report(report_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/agent-tasks")
def list_v2_agent_tasks(event_id: str):
    try:
        return production.list_agent_tasks(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/shared-memory")
def get_v2_shared_memory(event_id: str):
    try:
        return production.get_shared_memory_snapshot(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/trigger-events")
def list_v2_trigger_events(event_id: str):
    try:
        return production.list_trigger_events(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/agent-timeline")
def list_v2_agent_timeline(event_id: str):
    try:
        return production.list_agent_timeline(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v2/events/{event_id}/supervisor-runs")
def list_v2_supervisor_runs(event_id: str):
    try:
        return production.list_supervisor_runs(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v2/agent-tasks/{task_id}/replay")
def replay_v2_agent_task(
    task_id: str,
    request: ReplayRequest,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="agent_replay")
        return production.replay_agent_task(task_id, request)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/supervisor/tick")
def run_v2_supervisor_tick(
    event_id: str | None = None,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="supervisor_control")
        return system.supervisor_loop.tick_once(event_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/events/{event_id}/supervisor/run")
def run_v2_supervisor_for_event(
    event_id: str,
    x_operator_role: str | None = Header(default=None, alias="X-Operator-Role"),
):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="supervisor_control")
        return system.supervisor_loop.run_event_once(event_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
