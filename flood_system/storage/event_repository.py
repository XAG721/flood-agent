from __future__ import annotations

from datetime import datetime, timezone

from ..v2.models import (
    EventRecord,
    EventStreamRecord,
    HazardState,
    ObservationIngestItem,
    SimulationUpdateRecord,
)


class EventRepositoryMixin:
    def save_v2_event(self, event: EventRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_events (event_id, area_id, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    area_id = excluded.area_id,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (event.event_id, event.area_id, event.updated_at.isoformat(), event.model_dump_json()),
            )

    def get_v2_event(self, event_id: str) -> EventRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return EventRecord.model_validate_json(row["payload"]) if row else None

    def list_v2_events(self, *, limit: int = 200) -> list[EventRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_events
                ORDER BY updated_at DESC, event_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [EventRecord.model_validate_json(row["payload"]) for row in rows]

    def get_latest_v2_event_id(self, area_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id FROM v2_events
                WHERE area_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (area_id,),
            ).fetchone()
        return str(row["event_id"]) if row else None

    def add_v2_observations(self, event_id: str, observations: list[ObservationIngestItem]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO v2_observations (event_id, observed_at, payload)
                VALUES (?, ?, ?)
                """,
                [
                    (event_id, observation.observed_at.isoformat(), observation.model_dump_json())
                    for observation in observations
                ],
            )

    def list_v2_observations(self, event_id: str) -> list[ObservationIngestItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_observations
                WHERE event_id = ?
                ORDER BY observed_at ASC, id ASC
                """,
                (event_id,),
            ).fetchall()
        return [ObservationIngestItem.model_validate_json(row["payload"]) for row in rows]

    def save_v2_simulation_update(self, update: SimulationUpdateRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_simulation_updates (simulation_update_id, event_id, generated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(simulation_update_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    generated_at = excluded.generated_at,
                    payload = excluded.payload
                """,
                (
                    update.simulation_update_id,
                    update.event_id,
                    update.generated_at.isoformat(),
                    update.model_dump_json(),
                ),
            )

    def get_latest_v2_simulation_update(self, event_id: str) -> SimulationUpdateRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_simulation_updates
                WHERE event_id = ?
                ORDER BY generated_at DESC, simulation_update_id DESC
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
        return SimulationUpdateRecord.model_validate_json(row["payload"]) if row else None

    def list_v2_simulation_updates(self, event_id: str, *, limit: int = 20) -> list[SimulationUpdateRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_simulation_updates
                WHERE event_id = ?
                ORDER BY generated_at DESC, simulation_update_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [SimulationUpdateRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_hazard_state(self, hazard_state: HazardState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_hazard_states (event_id, generated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    generated_at = excluded.generated_at,
                    payload = excluded.payload
                """,
                (hazard_state.event_id, hazard_state.generated_at.isoformat(), hazard_state.model_dump_json()),
            )

    def get_v2_hazard_state(self, event_id: str) -> HazardState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_hazard_states WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return HazardState.model_validate_json(row["payload"]) if row else None

    def add_v2_stream_record(self, record: EventStreamRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_stream_records (event_id, created_at, payload)
                VALUES (?, ?, ?)
                """,
                (record.event_id, record.created_at.isoformat(), record.model_dump_json()),
            )

    def add_v2_stream_record_for_payload(self, event_id: str, event_type: str, payload: dict) -> None:
        self.add_v2_stream_record(
            EventStreamRecord(
                event_id=event_id,
                event_type=event_type,
                payload=payload,
                created_at=datetime.now(timezone.utc),
            )
        )

    def list_v2_stream_records(self, event_id: str, *, limit: int = 20) -> list[EventStreamRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_stream_records
                WHERE event_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [EventStreamRecord.model_validate_json(row["payload"]) for row in rows]
