from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flood_system.http.response_router import create_response_router
from flood_system.identity import (
    IdentityAssertionError,
    OperatorIdentity,
    TrustedIdentityVerifier,
    development_identity_from_headers,
)
from flood_system.response_workflow.models import (
    AlertAppendRequest,
    AlertInput,
    GeoPolygon,
    OperatorRole,
    RiskObjectBatchRequest,
    RiskObjectInput,
    SensitiveContact,
)
from flood_system.system import FloodWarningSystem


IDENTITY_SECRET = "pytest-trusted-identity-secret-with-at-least-32-bytes"


def signed_headers(
    *,
    method: str,
    path: str,
    nonce: str,
    assurance: str = "aal1",
    operator_id: str = "console_commander",
    role: OperatorRole = OperatorRole.COMMANDER,
    terminal_id: str = "district-response-console",
    timestamp: int | None = None,
) -> dict[str, str]:
    headers = TrustedIdentityVerifier.build_headers(
        secret=IDENTITY_SECRET,
        operator_id=operator_id,
        operator_role=role,
        assurance_level=assurance,
        terminal_id=terminal_id,
        nonce=nonce,
        method=method,
        path=path,
        timestamp=timestamp,
    )
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        headers["Idempotency-Key"] = nonce
    headers["X-Correlation-ID"] = nonce
    return headers


def test_unsigned_development_identity_is_explicit_and_impossible_in_production(monkeypatch):
    headers = {
        "x-operator-id": "demo-admin",
        "x-operator-role": "admin",
        "x-operator-terminal": "demo-console",
    }
    monkeypatch.setenv("FLOOD_ENVIRONMENT", "development")
    monkeypatch.setenv("FLOOD_ALLOW_DEV_IDENTITY_HEADERS", "1")
    identity = development_identity_from_headers(headers)
    assert identity is not None
    assert identity.operator_role == OperatorRole.ADMIN
    assert identity.assurance_level == "aal2"

    monkeypatch.setenv("FLOOD_ENVIRONMENT", "production")
    assert development_identity_from_headers(headers) is None


def test_trusted_identity_rejects_tampering_expiry_and_replay(tmp_path):
    system = FloodWarningSystem(tmp_path / "identity.db")
    verifier = TrustedIdentityVerifier(system.repository, IDENTITY_SECRET, max_age_seconds=60)
    headers = signed_headers(method="POST", path="/response/events", nonce="nonce-valid-identity-001")

    identity = verifier.verify(headers, method="POST", path="/response/events")

    assert identity.operator_id == "console_commander"
    assert identity.operator_role == OperatorRole.COMMANDER
    with pytest.raises(IdentityAssertionError, match="already used"):
        verifier.verify(headers, method="POST", path="/response/events")

    tampered = signed_headers(method="GET", path="/response/events", nonce="nonce-tamper-identity-01")
    tampered["X-Identity-Operator-Role"] = "admin"
    with pytest.raises(IdentityAssertionError, match="signature verification failed"):
        verifier.verify(tampered, method="GET", path="/response/events")

    expired = signed_headers(
        method="GET",
        path="/response/events",
        nonce="nonce-expired-identity-1",
        timestamp=int(time.time()) - 600,
    )
    with pytest.raises(IdentityAssertionError, match="expired"):
        verifier.verify(expired, method="GET", path="/response/events")


def test_response_api_binds_payload_identity_and_requires_aal2_for_high_risk_approval(tmp_path):
    system = FloodWarningSystem(tmp_path / "identity-api.db")
    system.response_identity = TrustedIdentityVerifier(system.repository, IDENTITY_SECRET)
    dashboard = system.response_workflow.bootstrap_demo()
    task = dashboard.tasks[0]
    app = FastAPI()
    app.include_router(create_response_router(lambda: system))
    client = TestClient(app)
    path = f"/response/tasks/{task.task_id}/decision"
    payload = {
        "operator_id": "console_commander",
        "operator_role": "commander",
        "terminal_id": "district-response-console",
        "decision": "approved",
        "note": "AAL2 approval assertion verified",
    }

    missing = client.get("/response/events")
    assert missing.status_code == 401

    no_idempotency_headers = signed_headers(
        method="POST", path=path, nonce="nonce-api-no-idempotency", assurance="aal2"
    )
    no_idempotency_headers.pop("Idempotency-Key")
    no_idempotency = client.post(path, headers=no_idempotency_headers, json=payload)
    assert no_idempotency.status_code == 400
    assert "Idempotency-Key" in no_idempotency.json()["detail"]["message"]

    aal1 = client.post(
        path,
        headers=signed_headers(method="POST", path=path, nonce="nonce-api-aal1-approval", assurance="aal1"),
        json=payload,
    )
    assert aal1.status_code == 403
    assert "aal2 authentication is required" in aal1.json()["detail"]["message"]

    mismatch_payload = {**payload, "operator_id": "another-operator"}
    mismatch = client.post(
        path,
        headers=signed_headers(method="POST", path=path, nonce="nonce-api-mismatch-001", assurance="aal2"),
        json=mismatch_payload,
    )
    assert mismatch.status_code == 403
    assert "does not match" in mismatch.json()["detail"]["message"]

    approved = client.post(
        path,
        headers=signed_headers(method="POST", path=path, nonce="nonce-api-aal2-approval", assurance="aal2"),
        json=payload,
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "issued"

    replay = client.post(
        path,
        headers=signed_headers(method="POST", path=path, nonce="nonce-api-aal2-approval", assurance="aal2"),
        json=payload,
    )
    assert replay.status_code == 401
    assert "already used" in replay.json()["detail"]["message"]


def test_candidate_discovery_api_uses_trusted_operator_identity(tmp_path):
    system = FloodWarningSystem(tmp_path / "identity-candidate-api.db")
    system.response_identity = TrustedIdentityVerifier(system.repository, IDENTITY_SECRET)
    dashboard = system.response_workflow.bootstrap_demo()
    app = FastAPI()
    app.include_router(create_response_router(lambda: system))
    client = TestClient(app)
    path = f"/response/events/{dashboard.event.event_id}/risk-objects/discover"
    payload = {
        "entity_types": ["school"],
        "min_risk_score": 45,
        "max_candidates": 5,
        "operator_id": "duty-operator",
        "operator_role": "duty_officer",
        "terminal_id": "duty-terminal",
    }

    response = client.post(
        path,
        headers=signed_headers(
            method="POST",
            path=path,
            nonce="nonce-candidate-discovery-001",
            operator_id="duty-operator",
            role=OperatorRole.DUTY_OFFICER,
            terminal_id="duty-terminal",
        ),
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["association_mode"] == "area_registry"
    assert response.json()["matched_profiles"] == 1


def test_backup_retention_api_requires_aal2_and_records_dry_run(tmp_path):
    system = FloodWarningSystem(tmp_path / "identity-retention-api.db")
    system.response_identity = TrustedIdentityVerifier(system.repository, IDENTITY_SECRET)
    system.response_workflow.bootstrap_demo()
    app = FastAPI()
    app.include_router(create_response_router(lambda: system))
    client = TestClient(app)
    path = "/response/security/backups/retention"
    payload = {
        "keep_latest": 14,
        "max_age_days": 90,
        "dry_run": True,
        "operator_id": "console_admin",
        "operator_role": "admin",
        "terminal_id": "district-response-console",
        "note": "季度保留策略预演",
    }

    aal1 = client.post(
        path,
        headers=signed_headers(
            method="POST", path=path, nonce="nonce-retention-aal1", assurance="aal1",
            operator_id="console_admin", role=OperatorRole.ADMIN,
        ),
        json=payload,
    )
    assert aal1.status_code == 403

    aal2 = client.post(
        path,
        headers=signed_headers(
            method="POST", path=path, nonce="nonce-retention-aal2", assurance="aal2",
            operator_id="console_admin", role=OperatorRole.ADMIN,
        ),
        json=payload,
    )
    assert aal2.status_code == 200
    assert aal2.json()["dry_run"] is True


def test_dashboard_applies_role_based_redaction_to_location_contacts_and_special_population(tmp_path):
    system = FloodWarningSystem(tmp_path / "classified-view.db")
    dashboard = system.response_workflow.bootstrap_demo()
    event_id = dashboard.event.event_id
    current_alert = dashboard.alert_snapshots[-1]
    system.response_workflow.append_alert(
        event_id,
        AlertAppendRequest(
            alert=AlertInput(
                alert_id="ALERT-SPATIAL-REDACTION",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=datetime.now(timezone.utc),
                affected_area=current_alert.affected_area,
                affected_geometry=GeoPolygon(
                    coordinates=[
                        (108.95, 34.24),
                        (108.97, 34.24),
                        (108.97, 34.26),
                        (108.95, 34.24),
                    ]
                ),
                raw_content="带空间范围的预警",
            ),
            operator_id="console_commander",
            operator_role=OperatorRole.COMMANDER,
            terminal_id="district-response-console",
        ),
    )
    current = dashboard.risk_objects[0]
    system.response_workflow.add_risk_objects(
        event_id,
        RiskObjectBatchRequest(
            objects=[
                RiskObjectInput(
                    **current.model_dump(include=set(RiskObjectInput.model_fields) - {
                        "sensitive_contacts", "special_population_notes"
                    }),
                    sensitive_contacts=[
                        SensitiveContact(name="张值守", role="现场联系人", phone="13800000001")
                    ],
                    special_population_notes="附近安置点有 3 名行动不便人员",
                )
            ],
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="district-console",
        ),
    )

    def identity(role: OperatorRole) -> OperatorIdentity:
        return OperatorIdentity(
            operator_id=f"operator-{role.value}",
            operator_role=role,
            assurance_level="aal2",
            terminal_id="district-console",
            issued_at=datetime.now(timezone.utc),
            nonce=f"nonce-{role.value}-classified",
        )

    commander = system.response_workflow.get_dashboard_for_identity(event_id, identity(OperatorRole.COMMANDER))
    field = system.response_workflow.get_dashboard_for_identity(event_id, identity(OperatorRole.FIELD_OPERATOR))
    auditor = system.response_workflow.get_dashboard_for_identity(event_id, identity(OperatorRole.AUDITOR))
    assert commander.alert_snapshots[-1].affected_geometry is not None
    assert field.alert_snapshots[-1].affected_geometry is not None
    assert auditor.alert_snapshots[-1].affected_geometry is None

    assert commander.risk_objects[0].location == current.location
    assert commander.risk_objects[0].sensitive_contacts[0].phone == "13800000001"
    assert "行动不便" in commander.risk_objects[0].special_population_notes
    assert field.risk_objects[0].location == current.location
    assert field.risk_objects[0].sensitive_contacts == []
    assert field.risk_objects[0].special_population_notes.startswith("[特殊人群信息")
    assert auditor.risk_objects[0].location.startswith("[精确位置")
    assert auditor.risk_objects[0].sensitive_contacts == []
