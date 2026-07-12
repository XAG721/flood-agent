from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class TransportSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, require_https: bool, trust_proxy_headers: bool) -> None:
        super().__init__(app)
        self.require_https = require_https
        self.trust_proxy_headers = trust_proxy_headers

    def _is_https(self, request: Request) -> bool:
        if request.url.scheme == "https":
            return True
        if self.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
            return forwarded == "https"
        return False

    async def dispatch(self, request: Request, call_next):
        secure = self._is_https(request)
        if self.require_https and not secure:
            return JSONResponse(
                status_code=426,
                content={"detail": "HTTPS is required for this deployment"},
                headers={"Upgrade": "TLS/1.2, HTTP/1.1"},
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
        if secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
