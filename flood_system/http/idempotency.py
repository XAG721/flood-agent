from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse, Response

from ..identity import OperatorIdentity
from ..response_workflow.models import IdempotencyRecord, IdempotencyStatus


WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
SAFE_REPLAY_HEADERS = frozenset({"etag", "location", "content-language"})
DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_REPLAY_BYTES = 8 * 1024 * 1024


def canonical_idempotency_scope(
    identity: OperatorIdentity,
    *,
    method: str,
    path: str,
) -> str:
    return ":".join(
        (
            identity.operator_id,
            identity.operator_role.value,
            identity.terminal_id,
            method.upper(),
            path,
        )
    )


def request_fingerprint(
    *,
    method: str,
    path: str,
    query_string: bytes,
    content_type: str,
    body: bytes,
) -> str:
    digest = hashlib.sha256()
    for value in (
        method.upper().encode("utf-8"),
        path.encode("utf-8"),
        query_string,
        content_type.strip().lower().encode("utf-8"),
        body,
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _positive_int_from_environment(
    name: str, default: int, *, minimum: int, maximum: int
) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, parsed))


def _error_response(code: str, message: str, *, retryable: bool) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "code": code,
                "message": message,
                "retryable": retryable,
            }
        },
    )


def _response_reference(body: bytes, fallback: str) -> str:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    for key in (
        "event_id",
        "task_id",
        "message_id",
        "package_id",
        "version_id",
        "record_id",
        "backup_id",
        "report_id",
    ):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return fallback


async def _materialize_response(response: Response) -> tuple[bytes, Response]:
    body = getattr(response, "body", None)
    if body is not None:
        if isinstance(body, str):
            return body.encode(response.charset or "utf-8"), response
        return bytes(body), response

    chunks: list[bytes] = []
    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
        chunks.append(
            chunk.encode(response.charset or "utf-8")
            if isinstance(chunk, str)
            else bytes(chunk)
        )
    materialized = b"".join(chunks)
    rebuilt = Response(
        content=materialized,
        status_code=response.status_code,
        headers=dict(response.headers),
        background=response.background,
    )
    return materialized, rebuilt


def _replay_response(record: IdempotencyRecord) -> Response:
    if not record.replayable or record.response_status is None:
        return _error_response(
            "IDEMPOTENCY_RESPONSE_NOT_REPLAYABLE",
            "the original write completed but its response exceeded the replay limit; inspect the recorded resource reference",
            retryable=False,
        )
    try:
        body = base64.b64decode(record.response_body_base64, validate=True)
    except (ValueError, binascii.Error):
        return _error_response(
            "IDEMPOTENCY_RECORD_INVALID",
            "the stored idempotency response is invalid and requires operator reconciliation",
            retryable=False,
        )
    headers = dict(record.response_headers)
    if record.response_content_type:
        headers["content-type"] = record.response_content_type
    headers["X-Idempotent-Replay"] = "true"
    headers["X-Idempotency-Record-Id"] = record.record_id
    return Response(content=body, status_code=record.response_status, headers=headers)


def _existing_record_response(record: IdempotencyRecord, request_hash: str) -> Response:
    if record.request_hash != request_hash:
        return _error_response(
            "IDEMPOTENCY_KEY_REUSED",
            "the Idempotency-Key was already used for a different request in the same operator and route scope",
            retryable=False,
        )
    if record.status == IdempotencyStatus.COMPLETED:
        return _replay_response(record)
    if record.status == IdempotencyStatus.INDETERMINATE:
        return _error_response(
            "IDEMPOTENCY_OUTCOME_INDETERMINATE",
            "the original write may have changed state but did not complete safely; reconcile the recorded resource before retrying",
            retryable=False,
        )
    return _error_response(
        "IDEMPOTENCY_REQUEST_IN_PROGRESS",
        "an equivalent write request is already processing",
        retryable=True,
    )


def create_idempotent_route_class(
    system_provider: Callable[[], Any],
    identity_resolver: Callable[[Request], OperatorIdentity],
) -> type[APIRoute]:
    class IdempotentCoreRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Any]:
            original_handler = super().get_route_handler()

            async def idempotent_handler(request: Request) -> Response:
                if request.method.upper() not in WRITE_METHODS:
                    return await original_handler(request)

                identity = identity_resolver(request)
                idempotency_key = request.headers.get("idempotency-key", "").strip()
                if not idempotency_key or len(idempotency_key) > 200 or any(
                    ord(character) < 33 or ord(character) > 126
                    for character in idempotency_key
                ):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "detail": {
                                "code": "VALIDATION_ERROR",
                                "message": "Idempotency-Key must contain 1-200 visible ASCII characters",
                                "retryable": False,
                            }
                        },
                    )
                path = request.url.path
                body = await request.body()
                request_hash = request_fingerprint(
                    method=request.method,
                    path=path,
                    query_string=request.scope.get("query_string", b""),
                    content_type=request.headers.get("content-type", ""),
                    body=body,
                )
                scope = canonical_idempotency_scope(
                    identity,
                    method=request.method,
                    path=path,
                )
                repository = system_provider().repository
                existing = repository.get_idempotency_record(scope, idempotency_key)
                if existing is not None:
                    return _existing_record_response(existing, request_hash)

                now = datetime.now(timezone.utc)
                ttl_seconds = _positive_int_from_environment(
                    "FLOOD_IDEMPOTENCY_TTL_SECONDS",
                    DEFAULT_TTL_SECONDS,
                    minimum=60,
                    maximum=7 * 24 * 60 * 60,
                )
                reservation = IdempotencyRecord(
                    record_id=f"IDEMP-{uuid4().hex}",
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    expires_at=now + timedelta(seconds=ttl_seconds),
                    created_at=now,
                )
                if not repository.reserve_idempotency_record(reservation):
                    existing = repository.get_idempotency_record(scope, idempotency_key)
                    if existing is None:
                        return _error_response(
                            "IDEMPOTENCY_RESERVATION_CONFLICT",
                            "the idempotency reservation changed concurrently; retry with the same request",
                            retryable=True,
                        )
                    return _existing_record_response(existing, request_hash)

                try:
                    response = await original_handler(request)
                except (HTTPException, RequestValidationError):
                    repository.release_idempotency_record(
                        scope=scope,
                        idempotency_key=idempotency_key,
                        request_hash=request_hash,
                    )
                    raise
                except Exception:
                    repository.complete_idempotency_record(
                        reservation.model_copy(
                            update={
                                "status": IdempotencyStatus.INDETERMINATE,
                                "response_ref": path,
                                "replayable": False,
                                "completed_at": datetime.now(timezone.utc),
                            }
                        )
                    )
                    raise

                response_body, response = await _materialize_response(response)
                if 200 <= response.status_code < 300:
                    max_replay_bytes = _positive_int_from_environment(
                        "FLOOD_IDEMPOTENCY_MAX_REPLAY_BYTES",
                        DEFAULT_MAX_REPLAY_BYTES,
                        minimum=1024,
                        maximum=64 * 1024 * 1024,
                    )
                    replayable = len(response_body) <= max_replay_bytes
                    safe_headers = {
                        key: value
                        for key, value in response.headers.items()
                        if key.lower() in SAFE_REPLAY_HEADERS
                    }
                    completed = reservation.model_copy(
                        update={
                            "status": IdempotencyStatus.COMPLETED,
                            "response_status": response.status_code,
                            "response_content_type": response.headers.get(
                                "content-type"
                            ),
                            "response_headers": safe_headers,
                            "response_body_base64": (
                                base64.b64encode(response_body).decode("ascii")
                                if replayable
                                else ""
                            ),
                            "response_ref": _response_reference(response_body, path),
                            "replayable": replayable,
                            "completed_at": datetime.now(timezone.utc),
                        }
                    )
                    repository.complete_idempotency_record(completed)
                    response.headers["X-Idempotency-Record-Id"] = completed.record_id
                    return response

                if response.status_code < 500:
                    repository.release_idempotency_record(
                        scope=scope,
                        idempotency_key=idempotency_key,
                        request_hash=request_hash,
                    )
                    return response

                replayable = len(response_body) <= DEFAULT_MAX_REPLAY_BYTES
                repository.complete_idempotency_record(
                    reservation.model_copy(
                        update={
                            "status": IdempotencyStatus.INDETERMINATE,
                            "response_status": response.status_code,
                            "response_content_type": response.headers.get(
                                "content-type"
                            ),
                            "response_body_base64": (
                                base64.b64encode(response_body).decode("ascii")
                                if replayable
                                else ""
                            ),
                            "response_ref": path,
                            "replayable": replayable,
                            "completed_at": datetime.now(timezone.utc),
                        }
                    )
                )
                return response

            return idempotent_handler

    return IdempotentCoreRoute
