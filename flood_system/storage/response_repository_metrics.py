from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..response_workflow.models import EventRiskObject, TimelineEntry


class ResponseMetricsRepositoryMixin:
    """Aggregated operational reads that avoid per-event repository fan-out."""

    def health_check(self) -> None:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()

    def response_metrics_snapshot(self) -> dict[str, Any]:
        with self._connect() as connection:
            events = connection.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN status != 'closed' THEN 1 ELSE 0 END) AS active "
                "FROM response_events"
            ).fetchone()
            outbox_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM response_outbox GROUP BY status"
            ).fetchall()
            callback_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM response_dispatch_callbacks"
                ).fetchone()["count"]
            )
            idempotency_rows = connection.execute(
                "SELECT status, COUNT(*) AS count "
                "FROM response_idempotency_records GROUP BY status"
            ).fetchall()
            rule_rows = connection.execute(
                "SELECT outcome, COUNT(*) AS count "
                "FROM response_rule_evaluations GROUP BY outcome"
            ).fetchall()
            timeline_rows = connection.execute(
                "SELECT payload FROM response_timeline WHERE entry_type = 'task_status'"
            ).fetchall()
            object_rows = connection.execute(
                "SELECT payload FROM response_event_objects"
            ).fetchall()

        outbox = defaultdict(int)
        outbox.update({str(row["status"]): int(row["count"]) for row in outbox_rows})
        idempotency = {
            str(row["status"]): int(row["count"]) for row in idempotency_rows
        }
        rules = {str(row["outcome"]): int(row["count"]) for row in rule_rows}
        takeovers = sum(
            self._secure_load(TimelineEntry, row["payload"]).action
            == "task_taken_over"
            for row in timeline_rows
        )
        stale_objects = sum(
            self._secure_load(EventRiskObject, row["payload"]).stale
            for row in object_rows
        )
        return {
            "events_total": int(events["total"] or 0),
            "events_active": int(events["active"] or 0),
            "outbox": dict(outbox),
            "dispatch_callbacks": callback_count,
            "idempotency": idempotency,
            "rules": rules,
            "manual_takeovers": takeovers,
            "stale_objects": stale_objects,
            "registry": self.risk_object_registry_metrics(),
            "documents": self.document_governance_metrics(),
            "evidence": self.evidence_governance_metrics(),
        }
