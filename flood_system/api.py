from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request

from .config import AppSettings, load_settings
from .http.body_limit import RequestBodyLimitMiddleware
from .http.legacy_platform_router import create_legacy_platform_router
from .http.operations_router import ApiMetricsState, create_operations_router
from .http.response_router import create_response_router
from .http.agent_twin_router import create_v3_router
from .response_workflow.risk_object_ingestion import configured_max_file_bytes
from .system import FloodWarningSystem
from .transport_security import TransportSecurityMiddleware


class ApplicationRuntime:
    """Lazily owns the process-level system container for one ASGI app."""

    def __init__(
        self,
        settings: AppSettings,
        system: FloodWarningSystem | None = None,
    ) -> None:
        self.settings = settings
        self._system = system

    @property
    def system(self) -> FloodWarningSystem:
        if self._system is None:
            self._system = FloodWarningSystem(self.settings.db_path)
        return self._system


class _RuntimeProxy:
    def __init__(self, provider: Callable[[], Any]) -> None:
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider(), name)


def create_app(
    *,
    settings_override: AppSettings | None = None,
    system_override: FloodWarningSystem | None = None,
) -> FastAPI:
    """Create an isolated application instance with injectable runtime state."""

    app_settings = settings_override or load_settings()
    runtime = ApplicationRuntime(app_settings, system_override)
    metrics_state = ApiMetricsState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.system.start_background_services()
        try:
            yield
        finally:
            runtime.system.stop_background_services()

    application = FastAPI(
        title=app_settings.title,
        version=app_settings.version,
        lifespan=lifespan,
    )
    application.state.runtime = runtime
    application.state.metrics = metrics_state
    application.add_middleware(
        TransportSecurityMiddleware,
        require_https=app_settings.require_https,
        trust_proxy_headers=app_settings.trust_proxy_headers,
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=(configured_max_file_bytes() * 4 // 3) + (1024 * 1024),
        paths={
            "/response/risk-objects/file-imports",
            "/api/v1/risk-objects/file-imports",
            "/response/documents",
            "/api/v1/documents",
        },
        path_prefixes={"/response/documents/", "/api/v1/documents/"},
    )

    @application.middleware("http")
    async def rewrite_compatibility_paths(request: Request, call_next):
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
                request.scope["path"] = f"{internal_prefix}{path[len(public_prefix):]}"
                break
        response = await call_next(request)
        metrics_state.record(
            duration_ms=(time.perf_counter() - started) * 1000,
            status_code=response.status_code,
        )
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    def provider() -> FloodWarningSystem:
        return runtime.system
    application.include_router(create_operations_router(provider, app_settings, metrics_state))
    application.include_router(create_v3_router(provider))
    application.include_router(create_response_router(provider))
    application.include_router(create_legacy_platform_router(provider))
    return application


settings = load_settings()
app = create_app(settings_override=settings)
system = _RuntimeProxy(lambda: app.state.runtime.system)
production = _RuntimeProxy(lambda: app.state.runtime.system.production_platform)
