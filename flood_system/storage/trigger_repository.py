from __future__ import annotations

from datetime import datetime, timezone

from ..v2.models import TriggerEvent, TriggerEventStatus


class TriggerRepositoryMixin:
    def publish_v2_trigger_event(
        self,
        trigger: TriggerEvent,
        *,
        dedupe_window_seconds: int = 15,
    ) -> TriggerEvent:
        with self._connect() as conn:
            if trigger.dedupe_key:
                cutoff = datetime.fromtimestamp(
                    datetime.now(timezone.utc).timestamp() - dedupe_window_seconds,
                    tz=timezone.utc,
                ).isoformat()
                row = conn.execute(
                    """
                    SELECT payload FROM v2_trigger_events
                    WHERE dedupe_key = ?
                      AND status IN (?, ?)
                      AND created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (
                        trigger.dedupe_key,
                        TriggerEventStatus.PENDING.value,
                        TriggerEventStatus.LEASED.value,
                        cutoff,
                    ),
                ).fetchone()
                if row:
                    return TriggerEvent.model_validate_json(row["payload"])
            conn.execute(
                """
                INSERT INTO v2_trigger_events (
                    trigger_id, event_id, trigger_type, status, dedupe_key, created_at, leased_at, processed_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trigger_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    trigger_type = excluded.trigger_type,
                    status = excluded.status,
                    dedupe_key = excluded.dedupe_key,
                    created_at = excluded.created_at,
                    leased_at = excluded.leased_at,
                    processed_at = excluded.processed_at,
                    payload = excluded.payload
                """,
                (
                    trigger.trigger_id,
                    trigger.event_id,
                    trigger.trigger_type.value,
                    trigger.status.value,
                    trigger.dedupe_key,
                    trigger.created_at.isoformat(),
                    trigger.leased_at.isoformat() if trigger.leased_at else None,
                    trigger.processed_at.isoformat() if trigger.processed_at else None,
                    trigger.model_dump_json(),
                ),
            )
        return trigger

    def lease_next_v2_trigger_event(self) -> TriggerEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_trigger_events
                WHERE status = ?
                ORDER BY created_at ASC, trigger_id ASC
                LIMIT 1
                """,
                (TriggerEventStatus.PENDING.value,),
            ).fetchone()
            if row is None:
                return None
            trigger = TriggerEvent.model_validate_json(row["payload"])
            leased = trigger.model_copy(
                update={
                    "status": TriggerEventStatus.LEASED,
                    "leased_at": datetime.now(timezone.utc),
                }
            )
            conn.execute(
                """
                UPDATE v2_trigger_events
                SET status = ?, leased_at = ?, payload = ?
                WHERE trigger_id = ?
                """,
                (
                    leased.status.value,
                    leased.leased_at.isoformat() if leased.leased_at else None,
                    leased.model_dump_json(),
                    leased.trigger_id,
                ),
            )
            return leased

    def mark_v2_trigger_event_processed(self, trigger_id: str) -> TriggerEvent | None:
        trigger = self.get_v2_trigger_event(trigger_id)
        if trigger is None:
            return None
        processed = trigger.model_copy(
            update={
                "status": TriggerEventStatus.PROCESSED,
                "processed_at": datetime.now(timezone.utc),
                "error_message": None,
            }
        )
        self._save_v2_trigger_event(processed)
        return processed

    def mark_v2_trigger_event_failed(self, trigger_id: str, *, error_message: str) -> TriggerEvent | None:
        trigger = self.get_v2_trigger_event(trigger_id)
        if trigger is None:
            return None
        failed = trigger.model_copy(
            update={
                "status": TriggerEventStatus.FAILED,
                "processed_at": datetime.now(timezone.utc),
                "error_message": error_message,
            }
        )
        self._save_v2_trigger_event(failed)
        return failed

    def _save_v2_trigger_event(self, trigger: TriggerEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_trigger_events (
                    trigger_id, event_id, trigger_type, status, dedupe_key, created_at, leased_at, processed_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trigger_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    trigger_type = excluded.trigger_type,
                    status = excluded.status,
                    dedupe_key = excluded.dedupe_key,
                    created_at = excluded.created_at,
                    leased_at = excluded.leased_at,
                    processed_at = excluded.processed_at,
                    payload = excluded.payload
                """,
                (
                    trigger.trigger_id,
                    trigger.event_id,
                    trigger.trigger_type.value,
                    trigger.status.value,
                    trigger.dedupe_key,
                    trigger.created_at.isoformat(),
                    trigger.leased_at.isoformat() if trigger.leased_at else None,
                    trigger.processed_at.isoformat() if trigger.processed_at else None,
                    trigger.model_dump_json(),
                ),
            )

    def get_v2_trigger_event(self, trigger_id: str) -> TriggerEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_trigger_events WHERE trigger_id = ?",
                (trigger_id,),
            ).fetchone()
        return TriggerEvent.model_validate_json(row["payload"]) if row else None

    def list_v2_trigger_events(self, event_id: str, *, limit: int = 40) -> list[TriggerEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_trigger_events
                WHERE event_id = ?
                ORDER BY created_at DESC, trigger_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [TriggerEvent.model_validate_json(row["payload"]) for row in rows]

    def count_pending_v2_trigger_events(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM v2_trigger_events WHERE status = ?",
                (TriggerEventStatus.PENDING.value,),
            ).fetchone()
        return int(row["count"]) if row else 0
