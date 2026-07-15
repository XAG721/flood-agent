from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .models import AlertSeverity, AuditRecord, OperationalAlert


class AuditOperationsMixin:
    repository: object

    def add_audit_record(
        self,
        *,
        source_type: str,
        action: str,
        summary: str,
        details: dict | None = None,
        severity: AlertSeverity = AlertSeverity.INFO,
        event_id: str | None = None,
        session_id: str | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            audit_id=f"audit_{uuid4().hex[:12]}",
            source_type=source_type,
            action=action,
            summary=summary,
            details=details or {},
            severity=severity,
            event_id=event_id,
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
        )
        self.repository.add_audit_record(record)
        return record

    def save_operational_alert(
        self,
        *,
        source_type: str,
        severity: AlertSeverity,
        summary: str,
        details: str = "",
        event_id: str | None = None,
    ) -> OperationalAlert:
        now = datetime.now(timezone.utc)
        alert = OperationalAlert(
            alert_id=f"alert_{uuid4().hex[:12]}",
            source_type=source_type,
            severity=severity,
            summary=summary,
            details=details,
            event_id=event_id,
            first_seen_at=now,
            last_seen_at=now,
        )
        self.repository.save_operational_alert(alert)
        return alert
