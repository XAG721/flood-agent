from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from flood_system.http.body_limit import RequestBodyLimitMiddleware


def test_risk_object_upload_body_limit_rejects_oversize_before_parsing():
    app = FastAPI()
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=64,
        paths={"/response/risk-objects/file-imports"},
    )

    @app.post("/response/risk-objects/file-imports")
    async def import_file(request: Request):
        return {"bytes": len(await request.body())}

    client = TestClient(app)
    accepted = client.post(
        "/response/risk-objects/file-imports",
        content=b"x" * 64,
    )
    rejected = client.post(
        "/response/risk-objects/file-imports",
        content=b"x" * 65,
    )

    assert accepted.status_code == 200
    assert accepted.json() == {"bytes": 64}
    assert rejected.status_code == 413
    assert rejected.json()["detail"]["code"] == "REQUEST_TOO_LARGE"
