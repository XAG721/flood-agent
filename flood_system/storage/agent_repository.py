from __future__ import annotations

from ..v2.models import AgentResult, AgentTask, AgentTaskEvent, SharedMemorySnapshot


class AgentRepositoryMixin:
    def save_v2_agent_task(self, task: AgentTask) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_agent_tasks (task_id, event_id, agent_name, status, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    agent_name = excluded.agent_name,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    task.task_id,
                    task.event_id,
                    task.agent_name.value,
                    task.status.value,
                    task.created_at.isoformat(),
                    task.model_dump_json(),
                ),
            )

    def get_v2_agent_task(self, task_id: str) -> AgentTask | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM v2_agent_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return AgentTask.model_validate_json(row["payload"]) if row else None

    def list_v2_agent_tasks(self, event_id: str, *, limit: int = 50) -> list[AgentTask]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_agent_tasks
                WHERE event_id = ?
                ORDER BY created_at DESC, task_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [AgentTask.model_validate_json(row["payload"]) for row in rows]

    def add_v2_agent_task_event(self, record: AgentTaskEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_agent_task_events (
                    task_event_id, event_id, task_id, agent_name, event_type, created_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_event_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    task_id = excluded.task_id,
                    agent_name = excluded.agent_name,
                    event_type = excluded.event_type,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.task_event_id,
                    record.event_id,
                    record.task_id,
                    record.agent_name.value,
                    record.event_type.value,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_agent_task_events(
        self,
        event_id: str,
        *,
        limit: int = 120,
    ) -> list[AgentTaskEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_agent_task_events
                WHERE event_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [AgentTaskEvent.model_validate_json(row["payload"]) for row in rows]

    def count_v2_agent_task_events(self, *, event_type: str | None = None) -> int:
        query = "SELECT COUNT(*) AS count FROM v2_agent_task_events WHERE 1 = 1"
        parameters: list[object] = []
        if event_type is not None:
            query += " AND event_type = ?"
            parameters.append(event_type)
        with self._connect() as conn:
            row = conn.execute(query, tuple(parameters)).fetchone()
        return int(row["count"]) if row else 0

    def save_v2_agent_result(self, result: AgentResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_agent_results (result_id, event_id, agent_name, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(result_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    agent_name = excluded.agent_name,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    result.result_id,
                    result.event_id,
                    result.agent_name.value,
                    result.created_at.isoformat(),
                    result.model_dump_json(),
                ),
            )

    def list_v2_agent_results(self, event_id: str, *, limit: int = 30) -> list[AgentResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_agent_results
                WHERE event_id = ?
                ORDER BY created_at DESC, result_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [AgentResult.model_validate_json(row["payload"]) for row in rows]

    def save_v2_event_shared_memory(self, snapshot: SharedMemorySnapshot) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_event_shared_memory (event_id, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    snapshot.event_id,
                    snapshot.updated_at.isoformat(),
                    snapshot.model_dump_json(),
                ),
            )

    def get_v2_event_shared_memory(self, event_id: str) -> SharedMemorySnapshot | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM v2_event_shared_memory WHERE event_id = ?", (event_id,)).fetchone()
        return SharedMemorySnapshot.model_validate_json(row["payload"]) if row else None
