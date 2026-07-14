from __future__ import annotations

from fastapi.testclient import TestClient

from flood_system.api import app


def test_health_readiness_metrics_and_openapi_are_available():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["database"] == "ok"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "flood_response_events_total" in metrics.text
    assert "flood_risk_object_registry" in metrics.text
    assert "flood_risk_object_registry_stale_candidate_runs" in metrics.text
    assert "flood_risk_object_registry_quarantined_rows" in metrics.text
    assert "flood_candidate_object_lists_frozen" in metrics.text
    assert "flood_api_latency_ms" in metrics.text
    assert metrics.headers["X-Correlation-ID"]
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/response/contracts/task-schema" in openapi.json()["paths"]
    assert "/response/dispatch/outbox" in openapi.json()["paths"]
    assert "/response/evidence-packages/{package_id}" in openapi.json()["paths"]
    assert "/response/events/{event_id}/replay" in openapi.json()["paths"]
    assert "/response/tasks/{task_id}/transitions" in openapi.json()["paths"]
    assert "/response/risk-objects" in openapi.json()["paths"]
    assert "/response/risk-objects/imports" in openapi.json()["paths"]
    assert "/response/risk-objects/file-imports" in openapi.json()["paths"]
    assert "/response/events/{event_id}/candidate-object-lists" in openapi.json()["paths"]
    assert (
        "/response/events/{event_id}/candidate-object-lists/freeze"
        in openapi.json()["paths"]
    )
