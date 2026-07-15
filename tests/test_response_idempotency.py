from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flood_system.http.idempotency import (
    canonical_idempotency_scope,
    request_fingerprint,
)
from flood_system.http.response_router import create_response_router
from flood_system.identity import OperatorIdentity, TrustedIdentityVerifier
from flood_system.response_workflow.models import (
    IdempotencyRecord,
    IdempotencyStatus,
    KeyRotationRequest,
    OperatorRole,
)
from flood_system.system import FloodWarningSystem


IDENTITY_SECRET = "pytest-idempotency-identity-secret-with-at-least-32-bytes"
PATH = "/response/events"


def _headers(*, nonce: str, idempotency_key: str) -> dict[str, str]:
    headers = TrustedIdentityVerifier.build_headers(
        secret=IDENTITY_SECRET,
        operator_id="idempotency-commander",
        operator_role=OperatorRole.COMMANDER,
        assurance_level="aal2",
        terminal_id="idempotency-console",
        nonce=nonce,
        method="POST",
        path=PATH,
    )
    headers["Idempotency-Key"] = idempotency_key
    headers["Content-Type"] = "application/json"
    headers["X-Correlation-ID"] = nonce
    return headers


def _event_payload(*, title: str = "幂等预警响应事件") -> dict:
    return {
        "title": title,
        "area_id": "district-idempotency",
        "alert": {
            "alert_id": "warning-idempotency-001",
            "source_department": "测试气象部门",
            "disaster_type": "暴雨",
            "level": "橙色",
            "issued_at": "2026-07-14T00:00:00+00:00",
            "valid_until": "2026-07-14T12:00:00+00:00",
            "affected_area": "幂等测试区",
            "raw_content": "幂等写接口测试原始预警",
            "source_type": "simulation",
            "source_version": "warning-idempotency-v1",
            "is_simulated": True,
        },
        "operator_id": "idempotency-commander",
        "operator_role": "commander",
        "terminal_id": "idempotency-console",
    }


def _client(tmp_path, *, raise_server_exceptions: bool = True):
    system = FloodWarningSystem(tmp_path / "idempotency.db")
    system.response_identity = TrustedIdentityVerifier(
        system.repository, IDENTITY_SECRET
    )
    app = FastAPI()
    app.include_router(create_response_router(lambda: system))
    return system, TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _identity() -> OperatorIdentity:
    return OperatorIdentity(
        operator_id="idempotency-commander",
        operator_role=OperatorRole.COMMANDER,
        assurance_level="aal2",
        terminal_id="idempotency-console",
        issued_at=datetime.now(timezone.utc),
        nonce="idempotency-scope-nonce",
    )


def test_core_write_replays_completed_response_and_rejects_key_reuse(tmp_path):
    system, client = _client(tmp_path)
    key = "create-event-stable-key-001"
    payload = _event_payload()

    created = client.post(
        PATH,
        headers=_headers(nonce="idempotency-create-first-0001", idempotency_key=key),
        json=payload,
    )
    replayed = client.post(
        PATH,
        headers=_headers(nonce="idempotency-create-replay-0002", idempotency_key=key),
        json=payload,
    )

    assert created.status_code == 200
    assert replayed.status_code == 200
    assert replayed.headers["X-Idempotent-Replay"] == "true"
    assert (
        replayed.headers["X-Idempotency-Record-Id"]
        == created.headers["X-Idempotency-Record-Id"]
    )
    assert replayed.content == created.content
    assert len(system.repository.list_response_events()) == 1

    changed = client.post(
        PATH,
        headers=_headers(nonce="idempotency-create-changed-0003", idempotency_key=key),
        json=_event_payload(title="同键不同事件不得写入"),
    )
    assert changed.status_code == 409
    assert changed.json()["detail"]["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert len(system.repository.list_response_events()) == 1
    assert system.repository.count_idempotency_records() == {
        "processing": 0,
        "completed": 1,
        "indeterminate": 0,
    }

    with system.repository._connect() as connection:
        encrypted = connection.execute(
            "SELECT payload FROM response_idempotency_records"
        ).fetchone()["payload"]
    envelope = json.loads(encrypted)
    assert envelope["protected"] is True
    assert "幂等预警响应事件" not in encrypted


def test_validation_failure_releases_reservation_for_corrected_retry(tmp_path):
    system, client = _client(tmp_path)
    key = "validation-retry-stable-key-001"

    missing_key = client.post(
        PATH,
        headers=_headers(nonce="idempotency-empty-key-0000", idempotency_key=""),
        json=_event_payload(),
    )
    assert missing_key.status_code == 400
    assert missing_key.json()["detail"]["code"] == "VALIDATION_ERROR"

    invalid = client.post(
        PATH,
        headers=_headers(nonce="idempotency-invalid-first-0001", idempotency_key=key),
        json={"title": "缺少必填预警载荷"},
    )
    assert invalid.status_code == 422
    assert system.repository.count_idempotency_records() == {
        "processing": 0,
        "completed": 0,
        "indeterminate": 0,
    }

    corrected = client.post(
        PATH,
        headers=_headers(nonce="idempotency-invalid-fixed-0002", idempotency_key=key),
        json=_event_payload(),
    )
    assert corrected.status_code == 200
    assert len(system.repository.list_response_events()) == 1


def test_repository_reservation_is_atomic_and_completion_is_encrypted(tmp_path):
    system, _ = _client(tmp_path)
    now = datetime.now(timezone.utc)
    base = {
        "scope": "operator:commander:terminal:POST:/response/events",
        "idempotency_key": "atomic-reservation-key-001",
        "request_hash": "a" * 64,
        "expires_at": now + timedelta(hours=1),
        "created_at": now,
    }

    def reserve(index: int) -> bool:
        return system.repository.reserve_idempotency_record(
            IdempotencyRecord(record_id=f"IDEMP-atomic-{index}", **base)
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(reserve, range(4)))

    assert results.count(True) == 1
    assert results.count(False) == 3
    record = system.repository.get_idempotency_record(
        base["scope"], base["idempotency_key"]
    )
    assert record is not None
    assert record.status == IdempotencyStatus.PROCESSING

    completed = record.model_copy(
        update={
            "status": IdempotencyStatus.COMPLETED,
            "response_status": 200,
            "response_content_type": "application/json",
            "response_body_base64": base64.b64encode(b'{"ok":true}').decode("ascii"),
            "response_ref": "event_id:FLOOD-IDEMPOTENT",
            "completed_at": datetime.now(timezone.utc),
        }
    )
    system.repository.complete_idempotency_record(completed)
    stored = system.repository.get_idempotency_record(
        base["scope"], base["idempotency_key"]
    )
    assert stored is not None
    assert stored.status == IdempotencyStatus.COMPLETED
    assert stored.response_ref == "event_id:FLOOD-IDEMPOTENT"
    assert (
        system.repository.release_idempotency_record(
            scope=base["scope"],
            idempotency_key=base["idempotency_key"],
            request_hash=base["request_hash"],
        )
        is False
    )


def test_in_progress_request_fails_closed_until_reservation_is_released(tmp_path):
    system, client = _client(tmp_path)
    key = "processing-reservation-key-001"
    payload = _event_payload()
    raw_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    identity = _identity()
    scope = canonical_idempotency_scope(identity, method="POST", path=PATH)
    request_hash = request_fingerprint(
        method="POST",
        path=PATH,
        query_string=b"",
        content_type="application/json",
        body=raw_body,
    )
    now = datetime.now(timezone.utc)
    assert system.repository.reserve_idempotency_record(
        IdempotencyRecord(
            record_id="IDEMP-pre-reserved",
            scope=scope,
            idempotency_key=key,
            request_hash=request_hash,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )
    )

    processing = client.post(
        PATH,
        headers=_headers(
            nonce="idempotency-processing-first-0001", idempotency_key=key
        ),
        content=raw_body,
    )
    assert processing.status_code == 409
    assert processing.json()["detail"]["code"] == "IDEMPOTENCY_REQUEST_IN_PROGRESS"
    assert len(system.repository.list_response_events()) == 0

    assert system.repository.release_idempotency_record(
        scope=scope,
        idempotency_key=key,
        request_hash=request_hash,
    )
    accepted = client.post(
        PATH,
        headers=_headers(
            nonce="idempotency-processing-fixed-0002", idempotency_key=key
        ),
        content=raw_body,
    )
    assert accepted.status_code == 200
    assert len(system.repository.list_response_events()) == 1


def test_unexpected_failure_becomes_indeterminate_and_is_not_reexecuted(
    tmp_path, monkeypatch
):
    system, client = _client(tmp_path, raise_server_exceptions=False)
    key = "indeterminate-write-key-001"
    attempts = 0

    def fail_after_unknown_outcome(_request):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("simulated persistence boundary failure")

    monkeypatch.setattr(
        system.response_workflow, "create_event", fail_after_unknown_outcome
    )
    first = client.post(
        PATH,
        headers=_headers(nonce="idempotency-unknown-first-0001", idempotency_key=key),
        json=_event_payload(),
    )
    second = client.post(
        PATH,
        headers=_headers(nonce="idempotency-unknown-retry-0002", idempotency_key=key),
        json=_event_payload(),
    )

    assert first.status_code == 500
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "IDEMPOTENCY_OUTCOME_INDETERMINATE"
    assert attempts == 1
    assert system.repository.count_idempotency_records()["indeterminate"] == 1


def test_completed_idempotency_response_survives_encryption_key_rotation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "FLOOD_BACKUP_MANIFEST_SECRET",
        "pytest-idempotency-rotation-manifest-secret-with-32-bytes",
    )
    monkeypatch.setenv(
        "FLOOD_DATA_ENCRYPTION_KEY_NEXT", Fernet.generate_key().decode("ascii")
    )
    system, client = _client(tmp_path)
    key = "rotation-replay-stable-key-001"
    payload = _event_payload(title="轮换后仍可幂等重放")
    created = client.post(
        PATH,
        headers=_headers(nonce="idempotency-rotation-first-0001", idempotency_key=key),
        json=payload,
    )
    assert created.status_code == 200

    rotation = system.response_workflow.rotate_data_encryption_key(
        KeyRotationRequest(
            label="idempotency-ledger",
            operator_id="rotation-admin",
            operator_role=OperatorRole.ADMIN,
            terminal_id="rotation-console",
        )
    )
    assert rotation.records_reencrypted > 0

    replayed = client.post(
        PATH,
        headers=_headers(nonce="idempotency-rotation-replay-0002", idempotency_key=key),
        json=payload,
    )
    assert replayed.status_code == 200
    assert replayed.headers["X-Idempotent-Replay"] == "true"
    assert replayed.content == created.content

    with system.repository._connect() as connection:
        key_ids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT json_extract(payload, '$.key_id') "
                "FROM response_idempotency_records"
            ).fetchall()
        }
    assert key_ids == {rotation.new_key_id}
