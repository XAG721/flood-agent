from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any


class RequestBodyTooLarge(ValueError):
    """Raised before request parsing when a bounded upload exceeds its limit."""


class RequestBodyLimitMiddleware:
    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        max_bytes: int,
        paths: set[str],
    ) -> None:
        self.app = app
        self.max_bytes = max(1, int(max_bytes))
        self.paths = frozenset(paths)

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if scope.get("type") != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._send_rejection(scope, receive, send)
                    return
            except ValueError:
                await self._send_rejection(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._send_rejection(scope, receive, send)

    async def _send_rejection(self, scope, receive, send) -> None:
        body = json.dumps(
            {
                "detail": {
                    "code": "REQUEST_TOO_LARGE",
                    "message": (
                        "risk-object import request exceeds the configured body limit"
                    ),
                    "retryable": False,
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
