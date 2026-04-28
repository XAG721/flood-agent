from __future__ import annotations

import json
from datetime import datetime, timezone

from ..v2.models import (
    MemoryEventRecord,
    MemorySnapshot,
    PlanRunRecord,
    ToolExecutionAuditRecord,
    V2CopilotMessage,
)


class CopilotRepositoryMixin:
    def save_v2_copilot_session(self, session_id: str, event_id: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_sessions (session_id, event_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    payload = excluded.payload
                """,
                (session_id, event_id, json.dumps(payload)),
            )

    def get_v2_copilot_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, event_id, payload FROM v2_copilot_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        payload["session_id"] = row["session_id"]
        payload["event_id"] = row["event_id"]
        return payload

    def list_recent_v2_copilot_sessions(self, *, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, event_id, payload FROM v2_copilot_sessions
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        sessions: list[dict] = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["session_id"] = row["session_id"]
            payload["event_id"] = row["event_id"]
            sessions.append(payload)
        return sessions

    def list_v2_copilot_sessions_by_event(self, event_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, event_id, payload FROM v2_copilot_sessions
                WHERE event_id = ?
                ORDER BY rowid ASC
                """,
                (event_id,),
            ).fetchall()
        sessions: list[dict] = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["session_id"] = row["session_id"]
            payload["event_id"] = row["event_id"]
            sessions.append(payload)
        return sessions

    def save_v2_copilot_message(self, session_id: str, message: V2CopilotMessage) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_messages (message_id, session_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (message.message_id, session_id, message.created_at.isoformat(), message.model_dump_json()),
            )

    def list_v2_copilot_messages(self, session_id: str) -> list[V2CopilotMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_copilot_messages
                WHERE session_id = ?
                ORDER BY created_at ASC, message_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [V2CopilotMessage.model_validate_json(row["payload"]) for row in rows]

    def save_v2_copilot_memory_state(self, snapshot: MemorySnapshot) -> None:
        updated_dt = snapshot.updated_at or datetime.now(timezone.utc)
        payload = snapshot.model_copy(update={"updated_at": updated_dt}).model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_memory_state (session_id, updated_at, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (snapshot.session_id, updated_dt.isoformat(), payload),
            )

    def get_v2_copilot_memory_state(self, session_id: str) -> MemorySnapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_copilot_memory_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return MemorySnapshot.model_validate_json(row["payload"]) if row else None

    def add_v2_copilot_memory_event(self, record: MemoryEventRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_memory_events (memory_event_id, session_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_event_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.memory_event_id,
                    record.session_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_copilot_memory_events(self, session_id: str, *, limit: int = 20) -> list[MemoryEventRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_copilot_memory_events
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [MemoryEventRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_copilot_plan_run(self, record: PlanRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_plan_runs (plan_run_id, session_id, message_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plan_run_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    message_id = excluded.message_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.plan_run_id,
                    record.session_id,
                    record.message_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_copilot_plan_runs(
        self,
        session_id: str,
        *,
        message_id: str | None = None,
        limit: int = 20,
    ) -> list[PlanRunRecord]:
        query = """
            SELECT payload FROM v2_copilot_plan_runs
            WHERE session_id = ?
        """
        parameters: list[object] = [session_id]
        if message_id is not None:
            query += " AND message_id = ?"
            parameters.append(message_id)
        query += " ORDER BY created_at DESC, plan_run_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [PlanRunRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_copilot_tool_execution(self, record: ToolExecutionAuditRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_copilot_tool_executions (execution_id, session_id, message_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    message_id = excluded.message_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.execution_id,
                    record.session_id,
                    record.message_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_copilot_tool_executions(
        self,
        session_id: str,
        *,
        message_id: str | None = None,
        limit: int = 40,
    ) -> list[ToolExecutionAuditRecord]:
        query = """
            SELECT payload FROM v2_copilot_tool_executions
            WHERE session_id = ?
        """
        parameters: list[object] = [session_id]
        if message_id is not None:
            query += " AND message_id = ?"
            parameters.append(message_id)
        query += " ORDER BY created_at DESC, execution_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [ToolExecutionAuditRecord.model_validate_json(row["payload"]) for row in rows]
