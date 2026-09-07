from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from ..config import AppSettings


@dataclass
class ApiMetricsState:
    request_durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    request_total: int = 0
    error_total: int = 0

    def record(self, *, duration_ms: float, status_code: int) -> None:
        self.request_total += 1
        if status_code >= 500:
            self.error_total += 1
        self.request_durations_ms.append(duration_ms)


class _RuntimeProxy:
    def __init__(self, provider: Callable[[], Any]) -> None:
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider(), name)


def create_operations_router(
    system_provider: Callable[[], Any],
    settings: AppSettings,
    metrics_state: ApiMetricsState,
) -> APIRouter:
    router = APIRouter(tags=["operations"])
    system = _RuntimeProxy(system_provider)

    @router.get("/health", tags=["operations"])
    def health():
        return {"status": "ok", "service": settings.title, "version": settings.version}


    @router.get("/ready", tags=["operations"])
    def readiness():
        try:
            system.repository.health_check()
            return {"status": "ready", "database": "ok"}
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail={"code": "DATABASE_NOT_READY", "message": str(exc)}
            ) from exc


    @router.get("/metrics", response_class=PlainTextResponse, tags=["operations"])
    def metrics():
        snapshot = system.repository.response_metrics_snapshot()
        outbox = snapshot["outbox"]
        registry_metrics = snapshot["registry"]
        document_metrics = snapshot["documents"]
        evidence_metrics = snapshot["evidence"]
        durations = sorted(metrics_state.request_durations_ms)

        def percentile(fraction: float) -> float:
            if not durations:
                return 0.0
            return durations[min(len(durations) - 1, int((len(durations) - 1) * fraction))]

        return "\n".join(
            (
                "# HELP flood_response_events_total Response events persisted by the core workflow.",
                "# TYPE flood_response_events_total gauge",
                f"flood_response_events_total {snapshot['events_total']}",
                "# HELP flood_response_events_active Active response events.",
                "# TYPE flood_response_events_active gauge",
                f"flood_response_events_active {snapshot['events_active']}",
                "# HELP flood_response_outbox_pending Pending simulated dispatch messages.",
                "# TYPE flood_response_outbox_pending gauge",
                f"flood_response_outbox_pending {outbox.get('pending', 0)}",
                "# HELP flood_response_outbox_failed Failed-closed simulated dispatch messages.",
                "# TYPE flood_response_outbox_failed gauge",
                f"flood_response_outbox_failed {outbox.get('failed', 0)}",
                "# HELP flood_response_outbox_sent Successfully processed simulated dispatch messages.",
                "# TYPE flood_response_outbox_sent gauge",
                f"flood_response_outbox_sent {outbox.get('sent', 0)}",
                "# HELP flood_response_outbox_partially_sent Partially accepted simulated dispatch messages.",
                "# TYPE flood_response_outbox_partially_sent gauge",
                f"flood_response_outbox_partially_sent {outbox.get('partially_sent', 0)}",
                "# HELP flood_response_outbox_manual_takeover Dispatch messages requiring manual takeover.",
                "# TYPE flood_response_outbox_manual_takeover gauge",
                f"flood_response_outbox_manual_takeover {outbox.get('manual_takeover', 0)}",
                "# HELP flood_response_dispatch_callbacks_total Unique simulated callbacks recorded.",
                "# TYPE flood_response_dispatch_callbacks_total gauge",
                f"flood_response_dispatch_callbacks_total {snapshot['dispatch_callbacks']}",
                "# HELP flood_response_idempotency_records Persisted Core API idempotency records by status.",
                "# TYPE flood_response_idempotency_records gauge",
                *(
                    f'flood_response_idempotency_records{{status="{status}"}} {count}'
                    for status, count in sorted(snapshot["idempotency"].items())
                ),
                "# HELP flood_response_rule_evaluations Rule outcomes by result.",
                "# TYPE flood_response_rule_evaluations gauge",
                f'flood_response_rule_evaluations{{outcome="PASS"}} {snapshot["rules"].get("PASS", 0)}',
                f'flood_response_rule_evaluations{{outcome="SOFT_WARNING"}} {snapshot["rules"].get("SOFT_WARNING", 0)}',
                f'flood_response_rule_evaluations{{outcome="HARD_BLOCK"}} {snapshot["rules"].get("HARD_BLOCK", 0)}',
                "# HELP flood_response_manual_takeovers_total Audited manual takeover actions.",
                "# TYPE flood_response_manual_takeovers_total gauge",
                f"flood_response_manual_takeovers_total {snapshot['manual_takeovers']}",
                "# HELP flood_response_stale_objects Current stale event risk objects.",
                "# TYPE flood_response_stale_objects gauge",
                f"flood_response_stale_objects {snapshot['stale_objects']}",
                "# HELP flood_risk_object_registry Registry master-data objects by status.",
                "# TYPE flood_risk_object_registry gauge",
                f'flood_risk_object_registry{{status="active"}} {registry_metrics["active"]}',
                f'flood_risk_object_registry{{status="inactive"}} {registry_metrics["inactive"]}',
                "# HELP flood_risk_object_registry_stale_candidate_runs Candidate runs invalidated by registry changes.",
                "# TYPE flood_risk_object_registry_stale_candidate_runs gauge",
                "flood_risk_object_registry_stale_candidate_runs "
                f"{registry_metrics['stale_candidate_runs']}",
                "# HELP flood_risk_object_registry_quarantined_rows Rows rejected by governed registry imports.",
                "# TYPE flood_risk_object_registry_quarantined_rows gauge",
                "flood_risk_object_registry_quarantined_rows "
                f"{registry_metrics['quarantined_rows']}",
                "# HELP flood_candidate_object_lists_frozen Immutable confirmed candidate lists.",
                "# TYPE flood_candidate_object_lists_frozen gauge",
                "flood_candidate_object_lists_frozen "
                f"{registry_metrics['frozen_candidate_lists']}",
                "# HELP flood_document_versions Immutable governed document versions.",
                "# TYPE flood_document_versions gauge",
                f"flood_document_versions {document_metrics['document_versions']}",
                "# HELP flood_document_parse_records Immutable parse results.",
                "# TYPE flood_document_parse_records gauge",
                f"flood_document_parse_records {document_metrics['parse_records']}",
                "# HELP flood_document_index_builds Index builds by terminal status.",
                "# TYPE flood_document_index_builds gauge",
                f'flood_document_index_builds{{status="completed"}} {document_metrics["index_builds_completed"]}',
                f'flood_document_index_builds{{status="failed"}} {document_metrics["index_builds_failed"]}',
                "# HELP flood_contract_versions Immutable task-schema and rule-set versions.",
                "# TYPE flood_contract_versions gauge",
                f"flood_contract_versions {document_metrics['contract_versions']}",
                "# HELP flood_evidence_package_versions Immutable evidence package versions.",
                "# TYPE flood_evidence_package_versions gauge",
                f"flood_evidence_package_versions {evidence_metrics['package_versions']}",
                "# HELP flood_evidence_packages Current logical evidence packages.",
                "# TYPE flood_evidence_packages gauge",
                f"flood_evidence_packages {evidence_metrics['packages']}",
                "# HELP flood_evidence_unresolved_conflicts Unresolved conflicts in current evidence packages.",
                "# TYPE flood_evidence_unresolved_conflicts gauge",
                f"flood_evidence_unresolved_conflicts {evidence_metrics['unresolved_conflicts']}",
                "# HELP flood_evidence_missing_fields Missing task fields in current evidence packages.",
                "# TYPE flood_evidence_missing_fields gauge",
                f'flood_evidence_missing_fields{{blocking="false"}} {evidence_metrics["missing_fields"]}',
                f'flood_evidence_missing_fields{{blocking="true"}} {evidence_metrics["blocking_missing_fields"]}',
                "# HELP flood_evidence_nli_packages Current evidence packages by degraded NLI state.",
                "# TYPE flood_evidence_nli_packages gauge",
                f'flood_evidence_nli_packages{{status="unavailable"}} {evidence_metrics["nli_unavailable"]}',
                f'flood_evidence_nli_packages{{status="partial"}} {evidence_metrics["nli_partial"]}',
                f'flood_evidence_nli_packages{{status="error"}} {evidence_metrics["nli_error"]}',
                "# HELP flood_api_requests_total Requests observed by the application middleware.",
                "# TYPE flood_api_requests_total counter",
                f"flood_api_requests_total {metrics_state.request_total}",
                "# HELP flood_api_errors_total HTTP 5xx responses observed by the application middleware.",
                "# TYPE flood_api_errors_total counter",
                f"flood_api_errors_total {metrics_state.error_total}",
                "# HELP flood_api_latency_ms Recent in-process request latency percentiles.",
                "# TYPE flood_api_latency_ms gauge",
                f'flood_api_latency_ms{{quantile="0.50"}} {percentile(0.50):.3f}',
                f'flood_api_latency_ms{{quantile="0.95"}} {percentile(0.95):.3f}',
                "",
            )
        )

    return router
