from __future__ import annotations

from .models import ArchiveStatusView, AuditRecord, OperationalAlert


class PlatformGovernanceOpsMixin:
    def list_operational_alerts(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 50,
    ) -> list[OperationalAlert]:
        return self.repository.list_operational_alerts(
            event_id=event_id,
            severity=severity,
            source_type=source_type,
            status="open",
            from_ts=from_ts,
            to_ts=to_ts,
            limit=limit,
        )

    def list_audit_records(
        self,
        *,
        event_id: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        return self.repository.list_audit_records(
            event_id=event_id,
            severity=severity,
            source_type=source_type,
            from_ts=from_ts,
            to_ts=to_ts,
            limit=limit,
        )

    def get_archive_status(self) -> ArchiveStatusView:
        return self.repository.get_archive_status()

    def run_archive_cycle(self) -> ArchiveStatusView:
        status = self.repository.archive_operational_records()
        self.add_audit_record(
            source_type="housekeeping",
            action="archive_run_completed",
            summary="Manual archive cycle completed.",
            details={
                "hot_records_archived": status.last_archive_run.hot_records_archived if status.last_archive_run else 0,
                "expired_archives_deleted": status.last_archive_run.expired_archives_deleted if status.last_archive_run else 0,
            },
        )
        return status
