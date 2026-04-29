from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from .config import load_settings
from .http.v3_router import create_v3_router
from .infrastructure.sse import repeated_snapshot_stream
from .system import FloodWarningSystem
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
from .v2.security import AuthorizationError, ensure_operator_role, list_operator_capabilities, normalize_operator_role


settings = load_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    system.start_background_services()
    try:
        yield
    finally:
        system.stop_background_services()


app = FastAPI(title=settings.title, version=settings.version, lifespan=lifespan)
system = FloodWarningSystem(settings.db_path)
production = system.production_platform
app.include_router(create_v3_router(lambda: system))


@app.middleware("http")
async def rewrite_unified_agent_twin_paths(request, call_next):
    """Expose unified public prefixes while keeping existing internal routes stable."""

    path = request.scope.get("path", "")
    alias_pairs = (
        ("/agent-twin", "/v3"),
        ("/platform", "/v2"),
    )
    for public_prefix, internal_prefix in alias_pairs:
        if path == public_prefix:
            request.scope["path"] = internal_prefix
            break
        if path.startswith(f"{public_prefix}/"):
            request.scope["path"] = f"{internal_prefix}{path[len(public_prefix):]}"
            break
    return await call_next(request)


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
def run_v2_archive_cycle(x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
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
def validate_v2_dataset(x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
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
def cancel_v2_dataset_job(job_id: str, x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="dataset_manage")
        return system.dataset_service.cancel_job(job_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/admin/dataset/jobs/{job_id}/retry")
def retry_v2_dataset_job(job_id: str, x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
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
        return production.list_regional_analysis_packages(event_id, include_pending=include_pending)
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_draft_edit")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="proposal_resolve")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="runtime_admin_write")
        return production.save_entity_profile(request.profile)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/v2/admin/entity-profiles/{entity_id}")
def update_v2_admin_entity_profile(entity_id: str, request: EntityProfileUpsertRequest):
    try:
        _resolve_operator_role(explicit_role=request.operator_role, action="runtime_admin_write")
        if request.profile.entity_id != entity_id:
            raise ValueError("entity_id in path and payload must match.")
        return production.save_entity_profile(request.profile)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v2/admin/entity-profiles/{entity_id}")
def delete_v2_admin_entity_profile(entity_id: str, x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="runtime_admin_write")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="runtime_admin_write")
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
def update_v2_event_resource_status(event_id: str, request: ResourceStatusUpdateRequest):
    try:
        _resolve_operator_role(explicit_role=request.operator_role, action="runtime_admin_write")
        production.save_event_resource_status(event_id, request.resource_status)
        return production.get_event_resource_status_view(event_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v2/admin/events/{event_id}/resource-status")
def delete_v2_event_resource_status(event_id: str, x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="runtime_admin_write")
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
        _resolve_operator_role(explicit_role=request.operator_role, action="runtime_admin_write")
        documents = production.import_rag_documents(request)
        return {"status": "imported", "document_count": len(documents), "documents": documents}
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v2/admin/rag-documents/reload")
def reload_v2_rag_documents(x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
    try:
        _resolve_operator_role(header_role=x_operator_role, action="runtime_admin_write")
        documents = production.reload_rag_documents()
        return {"status": "reloaded", "document_count": len(documents), "documents": documents}
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
def run_v2_evaluation(x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
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
def replay_v2_evaluation_report(report_id: str, x_operator_role: str | None = Header(default=None, alias="X-Operator-Role")):
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
