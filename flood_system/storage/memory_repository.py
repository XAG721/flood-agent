from __future__ import annotations

from ..v2.models import (
    DailyReportRecord,
    DailyReportRunRecord,
    EventEpisodeSummaryRecord,
    ExperienceRecord,
    HighRiskEpisodeRecord,
    LongTermMemoryRecord,
)


class MemoryRepositoryMixin:
    def save_v2_experience_record(self, record: ExperienceRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_experience_records (
                    experience_id, event_id, entity_id, entity_type, risk_level, action_type, created_at, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experience_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    entity_id = excluded.entity_id,
                    entity_type = excluded.entity_type,
                    risk_level = excluded.risk_level,
                    action_type = excluded.action_type,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.experience_id,
                    record.event_id,
                    record.entity_id,
                    record.entity_type,
                    record.risk_level.value if record.risk_level else None,
                    record.action_type,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def save_v2_daily_report(self, record: DailyReportRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_daily_reports (report_id, event_id, report_date, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    report_date = excluded.report_date,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.report_id,
                    record.event_id,
                    record.report_date.isoformat(),
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_daily_reports(self, event_id: str, *, limit: int = 30) -> list[DailyReportRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_daily_reports
                WHERE event_id = ?
                ORDER BY report_date DESC, created_at DESC, report_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [DailyReportRecord.model_validate_json(row["payload"]) for row in rows]

    def get_v2_daily_report_by_event_and_date(self, event_id: str, report_date: str) -> DailyReportRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_daily_reports
                WHERE event_id = ? AND report_date = ?
                ORDER BY created_at DESC, report_id DESC
                LIMIT 1
                """,
                (event_id, report_date),
            ).fetchone()
        return DailyReportRecord.model_validate_json(row["payload"]) if row else None

    def save_v2_daily_report_run(self, record: DailyReportRunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_daily_report_runs (run_id, event_id, report_date, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    report_date = excluded.report_date,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.run_id,
                    record.event_id,
                    record.report_date.isoformat(),
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def get_v2_daily_report_run(self, event_id: str, report_date: str) -> DailyReportRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_daily_report_runs
                WHERE event_id = ? AND report_date = ?
                ORDER BY created_at DESC, run_id DESC
                LIMIT 1
                """,
                (event_id, report_date),
            ).fetchone()
        return DailyReportRunRecord.model_validate_json(row["payload"]) if row else None

    def save_v2_high_risk_episode(self, record: HighRiskEpisodeRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_high_risk_episodes (episode_id, event_id, started_at, status, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    started_at = excluded.started_at,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    record.episode_id,
                    record.event_id,
                    record.started_at.isoformat(),
                    record.status.value,
                    record.model_dump_json(),
                ),
            )

    def list_v2_high_risk_episodes(self, event_id: str, *, limit: int = 20) -> list[HighRiskEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_high_risk_episodes
                WHERE event_id = ?
                ORDER BY started_at DESC, episode_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [HighRiskEpisodeRecord.model_validate_json(row["payload"]) for row in rows]

    def list_open_v2_high_risk_episodes(self, event_id: str) -> list[HighRiskEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_high_risk_episodes
                WHERE event_id = ? AND status = 'open'
                ORDER BY started_at DESC, episode_id DESC
                """,
                (event_id,),
            ).fetchall()
        return [HighRiskEpisodeRecord.model_validate_json(row["payload"]) for row in rows]

    def list_pending_v2_episode_summaries(self, *, limit: int = 50) -> list[HighRiskEpisodeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.payload
                FROM v2_high_risk_episodes e
                LEFT JOIN v2_event_episode_summaries s ON s.episode_id = e.episode_id
                WHERE e.status = 'closed' AND s.summary_id IS NULL
                ORDER BY e.started_at ASC, e.episode_id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [HighRiskEpisodeRecord.model_validate_json(row["payload"]) for row in rows]

    def save_v2_event_episode_summary(self, record: EventEpisodeSummaryRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_event_episode_summaries (summary_id, episode_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(summary_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    event_id = excluded.event_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.summary_id,
                    record.episode_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_event_episode_summaries(self, event_id: str, *, limit: int = 20) -> list[EventEpisodeSummaryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM v2_event_episode_summaries
                WHERE event_id = ?
                ORDER BY created_at DESC, summary_id DESC
                LIMIT ?
                """,
                (event_id, limit),
            ).fetchall()
        return [EventEpisodeSummaryRecord.model_validate_json(row["payload"]) for row in rows]

    def get_v2_event_episode_summary_by_episode(self, episode_id: str) -> EventEpisodeSummaryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM v2_event_episode_summaries
                WHERE episode_id = ?
                ORDER BY created_at DESC, summary_id DESC
                LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
        return EventEpisodeSummaryRecord.model_validate_json(row["payload"]) if row else None

    def save_v2_long_term_memory(self, record: LongTermMemoryRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_long_term_memories (memory_id, event_id, created_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    created_at = excluded.created_at,
                    payload = excluded.payload
                """,
                (
                    record.memory_id,
                    record.event_id,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )

    def list_v2_long_term_memories(
        self,
        *,
        event_id: str | None = None,
        limit: int = 100,
    ) -> list[LongTermMemoryRecord]:
        query = "SELECT payload FROM v2_long_term_memories WHERE 1 = 1"
        params: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            params.append(event_id)
        query += " ORDER BY created_at DESC, memory_id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [LongTermMemoryRecord.model_validate_json(row["payload"]) for row in rows]

    def list_v2_experience_records(
        self,
        *,
        event_id: str | None = None,
        entity_id: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        query = "SELECT payload FROM v2_experience_records WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        if entity_id is not None:
            query += " AND entity_id = ?"
            parameters.append(entity_id)
        if entity_type is not None:
            query += " AND entity_type = ?"
            parameters.append(entity_type)
        query += " ORDER BY created_at DESC, experience_id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(parameters)).fetchall()
        return [ExperienceRecord.model_validate_json(row["payload"]) for row in rows]
