from __future__ import annotations

from ..v2.models import ExecutionLogEntry, NotificationDraft
from ..v3.models import AudienceWarningDraft


class NotificationRepositoryMixin:
    def save_v2_notification_draft(self, draft: NotificationDraft) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_notification_drafts (draft_id, event_id, proposal_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(draft_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    proposal_id = excluded.proposal_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (draft.draft_id, draft.event_id, draft.proposal_id, draft.created_at.isoformat(), draft.model_dump_json()),
            )

    def list_v2_notification_drafts(self, event_id: str) -> list[NotificationDraft]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_notification_drafts
                WHERE event_id = ?
                ORDER BY created_at DESC, draft_id DESC
                """,
                (event_id,),
            ).fetchall()
        return [NotificationDraft.model_validate_json(row["payload"]) for row in rows]

    def save_v2_execution_log(self, entry: ExecutionLogEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_execution_logs (log_id, event_id, proposal_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(log_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    proposal_id = excluded.proposal_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (entry.log_id, entry.event_id, entry.proposal_id, entry.created_at.isoformat(), entry.model_dump_json()),
            )

    def list_v2_execution_logs(self, event_id: str) -> list[ExecutionLogEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_execution_logs
                WHERE event_id = ?
                ORDER BY created_at DESC, log_id DESC
                """,
                (event_id,),
            ).fetchall()
        return [ExecutionLogEntry.model_validate_json(row["payload"]) for row in rows]

    def save_v3_audience_warning(self, warning: AudienceWarningDraft) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v3_audience_warnings (warning_id, event_id, proposal_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(warning_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    proposal_id = excluded.proposal_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    warning.warning_id,
                    warning.event_id,
                    warning.proposal_id,
                    warning.created_at.isoformat(),
                    warning.model_dump_json(),
                ),
            )

    def list_v3_audience_warnings(
        self,
        event_id: str,
        *,
        proposal_id: str | None = None,
    ) -> list[AudienceWarningDraft]:
        query = """
            SELECT payload FROM v3_audience_warnings
            WHERE event_id = ?
        """
        params: list[object] = [event_id]
        if proposal_id is not None:
            query += " AND proposal_id = ?"
            params.append(proposal_id)
        query += " ORDER BY created_at DESC, warning_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AudienceWarningDraft.model_validate_json(row["payload"]) for row in rows]
