from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flood_system.identity import TrustedIdentityVerifier
from flood_system.security import BackupManifestError, BackupManifestSigner, DataProtectionError, DataProtector
from flood_system.transport_security import TransportSecurityMiddleware


def build_transport_app(*, require_https: bool, trust_proxy_headers: bool) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        TransportSecurityMiddleware,
        require_https=require_https,
        trust_proxy_headers=trust_proxy_headers,
    )

    @app.get("/probe")
    def probe():
        return {"status": "ok"}

    return app


def test_https_enforcement_requires_trusted_forwarded_proto_and_sets_security_headers():
    client = TestClient(build_transport_app(require_https=True, trust_proxy_headers=True))

    insecure = client.get("/probe")
    secure = client.get("/probe", headers={"X-Forwarded-Proto": "https"})

    assert insecure.status_code == 426
    assert insecure.headers["upgrade"].startswith("TLS/1.2")
    assert secure.status_code == 200
    assert secure.headers["strict-transport-security"].startswith("max-age=31536000")
    assert secure.headers["x-content-type-options"] == "nosniff"
    assert secure.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in secure.headers["content-security-policy"]


def test_forwarded_proto_is_ignored_when_proxy_headers_are_not_trusted():
    client = TestClient(build_transport_app(require_https=True, trust_proxy_headers=False))
    response = client.get("/probe", headers={"X-Forwarded-Proto": "https"})
    assert response.status_code == 426


def test_production_refuses_to_start_without_independent_encryption_and_identity_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOOD_ENVIRONMENT", "production")
    monkeypatch.delenv("FLOOD_DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("FLOOD_TRUSTED_IDENTITY_SECRET", raising=False)

    with pytest.raises(DataProtectionError, match="required in production"):
        DataProtector.for_database(tmp_path / "production.db")
    with pytest.raises(ValueError, match="required in production"):
        TrustedIdentityVerifier.from_environment(repository=object())
    with pytest.raises(BackupManifestError, match="required in production"):
        BackupManifestSigner.from_environment()
