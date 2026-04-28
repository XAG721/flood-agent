from __future__ import annotations

from datetime import datetime, timezone

from ..models import ResourceStatus
from ..v2.models import EntityProfile


class RuntimeRepositoryMixin:
    def save_v2_entity_profile(self, entity: EntityProfile) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_entity_profiles (entity_id, area_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    area_id = excluded.area_id,
                    payload = excluded.payload
                """,
                (entity.entity_id, entity.area_id, entity.model_dump_json()),
            )

    def get_v2_entity_profile(self, entity_id: str) -> EntityProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_entity_profiles WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
        return EntityProfile.model_validate_json(row["payload"]) if row else None

    def list_v2_entity_profiles(
        self,
        *,
        area_id: str | None = None,
        entity_type: str | None = None,
    ) -> list[EntityProfile]:
        query = "SELECT payload FROM v2_entity_profiles"
        clauses: list[str] = []
        parameters: list[object] = []
        if area_id is not None:
            clauses.append("area_id = ?")
            parameters.append(area_id)
        with self._connect() as conn:
            if clauses:
                query = f"{query} WHERE {' AND '.join(clauses)}"
            query = f"{query} ORDER BY entity_id ASC"
            rows = conn.execute(query, parameters).fetchall()
        profiles = [EntityProfile.model_validate_json(row["payload"]) for row in rows]
        if entity_type is not None:
            profiles = [item for item in profiles if item.entity_type.value == entity_type]
        return profiles

    def delete_v2_entity_profile(self, entity_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM v2_entity_profiles WHERE entity_id = ?",
                (entity_id,),
            )
        return cursor.rowcount > 0

    def has_v2_entity_profiles(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM v2_entity_profiles LIMIT 1").fetchone()
        return row is not None

    def save_area_resource_status(self, resource_status: ResourceStatus) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_area_resource_status (area_id, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(area_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (resource_status.area_id, updated_at, resource_status.model_dump_json()),
            )

    def get_area_resource_status(self, area_id: str) -> ResourceStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_area_resource_status WHERE area_id = ?",
                (area_id,),
            ).fetchone()
        return ResourceStatus.model_validate_json(row["payload"]) if row else None

    def has_area_resource_statuses(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM v2_area_resource_status LIMIT 1").fetchone()
        return row is not None

    def save_event_resource_status(self, event_id: str, resource_status: ResourceStatus) -> None:
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_event_resource_status (event_id, area_id, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    area_id = excluded.area_id,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (event_id, resource_status.area_id, updated_at, resource_status.model_dump_json()),
            )

    def get_event_resource_status(self, event_id: str) -> ResourceStatus | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_event_resource_status WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return ResourceStatus.model_validate_json(row["payload"]) if row else None

    def delete_event_resource_status(self, event_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM v2_event_resource_status WHERE event_id = ?",
                (event_id,),
            )
        return cursor.rowcount > 0
