from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..infrastructure.sse import typed_event_stream
from ..schemas.v3 import AgentDialogRequest, ProposalGenerationRequest
from ..v2.llm_gateway import LLMGenerationError


def create_v3_router(system_provider: Callable[[], Any]) -> APIRouter:
    router = APIRouter(prefix="/v3", tags=["agent-twin-v3"])

    @router.get("/events/{event_id}/twin-overview")
    def get_twin_overview(event_id: str):
        try:
            return system_provider().agent_twin.get_twin_overview(event_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/events/{event_id}/objects/{object_id}")
    def get_focus_object(event_id: str, object_id: str):
        try:
            return system_provider().agent_twin.get_focus_object(event_id, object_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/events/{event_id}/dialog")
    def run_agent_dialog(event_id: str, request: AgentDialogRequest):
        try:
            return system_provider().agent_twin.run_dialog(event_id, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/events/{event_id}/agent-council")
    def get_agent_council(event_id: str):
        try:
            return system_provider().agent_twin.get_agent_council(event_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/events/{event_id}/proposals/generate")
    def generate_proposals(event_id: str, request: ProposalGenerationRequest):
        try:
            return system_provider().agent_twin.generate_proposals(event_id, request)
        except LLMGenerationError as exc:
            raise HTTPException(status_code=503, detail=f"{exc.code}: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/proposals/{proposal_id}/warnings/generate")
    def generate_warnings(proposal_id: str):
        try:
            return system_provider().agent_twin.generate_warnings(proposal_id)
        except LLMGenerationError as exc:
            raise HTTPException(status_code=503, detail=f"{exc.code}: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/events/{event_id}/stream")
    async def stream_event_updates(event_id: str, object_id: str | None = None):
        stream = typed_event_stream(
            lambda: system_provider().agent_twin.build_stream_events(event_id, focus_object_id=object_id),
            event_type_getter=lambda event: event.event_type,
            version_getter=lambda event: event.version,
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    return router
