from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from ..models import CorpusType, RAGDocument, RiskLevel
from .models import (
    AlertSeverity,
    DailyReportRecord,
    DailyReportRunRecord,
    EventEpisodeSummaryRecord,
    EventRecord,
    GenerationSource,
    HighRiskEpisodeRecord,
    HighRiskEpisodeStatus,
    LongTermMemoryRecord,
    V2CopilotMessage,
)


HIGH_RISK_LEVELS = {RiskLevel.ORANGE, RiskLevel.RED}


def _risk_rank(level: RiskLevel | None) -> int:
    if level is None:
        return 0
    return {
        RiskLevel.NONE: 0,
        RiskLevel.BLUE: 1,
        RiskLevel.YELLOW: 2,
        RiskLevel.ORANGE: 3,
        RiskLevel.RED: 4,
    }[level]


def _dt_in_range(value: datetime | None, start: datetime, end: datetime) -> bool:
    if value is None:
        return False
    normalized = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return start <= normalized < end


def _coerce_strings(items: list[str] | tuple[str, ...] | None, *, limit: int | None = None) -> list[str]:
    values = [item.strip() for item in (items or []) if isinstance(item, str) and item.strip()]
    if limit is not None:
        return values[:limit]
    return values


def _format_report_items(items: list[str], empty_text: str) -> str:
    return "\uff1b".join(items) if items else empty_text


def _deliver_event_message(
    repository: object,
    *,
    event_id: str,
    content: str,
    created_at: datetime,
    delivered_session_ids: list[str] | None = None,
) -> list[str]:
    delivered = list(delivered_session_ids or [])
    for session in repository.list_v2_copilot_sessions_by_event(event_id):
        session_id = session.get("session_id")
        if not session_id or session_id in delivered:
            continue
        repository.save_v2_copilot_message(
            session_id,
            V2CopilotMessage(
                message_id=f"v2_msg_{uuid4().hex[:10]}",
                role="assistant",
                content=content,
                created_at=created_at,
            ),
        )
        delivered.append(session_id)
    return delivered


def format_daily_report_message(report: DailyReportRecord) -> str:
    unresolved = _format_report_items(report.unresolved_risks, "\u6682\u65e0\u65b0\u589e\u672a\u51b3\u98ce\u9669\u3002")
    next_steps = _format_report_items(report.next_day_recommendations, "\u6b21\u65e5\u7ee7\u7eed\u4fdd\u6301\u5e38\u6001\u5316\u590d\u6838\u4e0e\u503c\u5b88\u3002")
    return (
        f"{report.headline}\n"
        f"\u4e00\u3001\u6001\u52bf\u6982\u8ff0\n{report.situation_summary}\n"
        f"\u4e8c\u3001\u51b3\u7b56\u6458\u8981\n{report.decisions_summary}\n"
        f"\u4e09\u3001\u5904\u7f6e\u8fdb\u5c55\n{report.action_summary}\n"
        f"\u56db\u3001\u672a\u51b3\u98ce\u9669\n{unresolved}\n"
        f"\u4e94\u3001\u6b21\u65e5\u5efa\u8bae\n{next_steps}"
    )


def format_episode_summary_message(summary: EventEpisodeSummaryRecord) -> str:
    escalation_path = _format_report_items(summary.escalation_path, "\u6682\u65e0\u5b8c\u6574\u5347\u7ea7\u8def\u5f84\u8bb0\u5f55\u3002")
    key_decisions = _format_report_items(summary.key_decisions, "\u6682\u65e0\u5173\u952e\u51b3\u7b56\u8bb0\u5f55\u3002")
    successful_actions = _format_report_items(summary.successful_actions, "\u6682\u65e0\u5df2\u9a8c\u8bc1\u6709\u6548\u7684\u5904\u7f6e\u52a8\u4f5c\u3002")
    coordination_gaps = _format_report_items(summary.coordination_gaps, "\u6682\u65e0\u660e\u786e\u7684\u534f\u540c\u7f3a\u53e3\u3002")
    return (
        f"{summary.headline}\n"
        f"\u4e00\u3001\u98ce\u9669\u5347\u7ea7\u8def\u5f84\n{escalation_path}\n"
        f"\u4e8c\u3001\u5173\u952e\u51b3\u7b56\n{key_decisions}\n"
        f"\u4e09\u3001\u6709\u6548\u52a8\u4f5c\n{successful_actions}\n"
        f"\u56db\u3001\u534f\u540c\u7f3a\u53e3\n{coordination_gaps}"
    )


@dataclass
class LongTermMemoryStore:
    repository: object
    rag_service: object
    top_k: int = field(default_factory=lambda: max(1, int(os.getenv("FLOOD_LONG_TERM_MEMORY_TOP_K", "3"))))

    def save_memory(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        self.repository.save_v2_long_term_memory(record)
        document = RAGDocument(
            doc_id=f"memory_{record.memory_id}",
            corpus=CorpusType.MEMORY,
            title=record.headline,
            content=record.retrieval_text,
            metadata={
                "event_id": record.event_id,
                "summary_id": record.source_summary_id,
                "area_id": record.area_id,
                "risk_level": record.risk_level.value if record.risk_level else None,
                "entity_types": record.entity_types,
                "action_types": record.action_types,
                "tags": record.tags,
                "created_at": record.created_at.isoformat(),
                "memory_id": record.memory_id,
            },
        )
        self.rag_service.import_documents([document])
        return record

    def query_memories(
        self,
        *,
        area_id: str | None = None,
        entity_type: str | None = None,
        risk_level: RiskLevel | None = None,
        action_type: str | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[LongTermMemoryRecord]:
        records = self.repository.list_v2_long_term_memories(limit=300)
        wanted_tags = {item.strip().lower() for item in (tags or []) if isinstance(item, str) and item.strip()}
        ranked: list[tuple[float, LongTermMemoryRecord]] = []
        has_query = any([area_id, entity_type, risk_level, action_type, wanted_tags])
        for record in records:
            record_tags = {item.lower() for item in record.tags}
            record_entity_types = set(record.entity_types)
            record_action_types = set(record.action_types)

            area_match = bool(area_id and record.area_id == area_id)
            entity_match = bool(entity_type and entity_type in record_entity_types)
            action_match = bool(action_type and action_type in record_action_types)
            matched_tags = wanted_tags.intersection(record_tags)

            risk_score = 0.0
            if risk_level and record.risk_level:
                distance = abs(_risk_rank(risk_level) - _risk_rank(record.risk_level))
                risk_score = max(0.0, 5.5 - (distance * 1.8))

            score = 0.0
            if area_match:
                score += 8.0
            elif area_id:
                score -= 2.0

            score += risk_score

            if entity_match:
                score += 6.0
            elif entity_type:
                score -= 1.5

            if action_match:
                score += 3.5
            elif action_type:
                score -= 0.5

            if matched_tags:
                score += min(3.0, 1.2 * len(matched_tags))
            elif wanted_tags:
                score -= 0.5

            if record.recommendations:
                score += min(1.5, 0.4 * len(record.recommendations))
            if record.lessons:
                score += min(1.0, 0.3 * len(record.lessons))

            age_days = max(0.0, (datetime.now(UTC) - record.created_at.astimezone(UTC)).total_seconds() / 86400.0)
            score += max(0.0, 1.2 - min(1.2, age_days / 45.0))

            if has_query and score <= 0:
                continue
            ranked.append((score, record))

        ranked.sort(key=lambda item: (-item[0], -item[1].created_at.timestamp()))
        return [item for _, item in ranked[: (top_k or self.top_k)]]


@dataclass
class DailySummaryService:
    repository: object
    platform: object
    poll_seconds: float = 60.0
    timezone_name: str = field(default_factory=lambda: os.getenv("FLOOD_DAILY_SUMMARY_TIMEZONE", "Asia/Shanghai"))
    run_time_local: str = field(default_factory=lambda: os.getenv("FLOOD_DAILY_SUMMARY_TIME_LOCAL", "08:00"))
    enabled: bool = field(default_factory=lambda: os.getenv("FLOOD_DAILY_SUMMARY_ENABLED", "1").strip().lower() not in {"0", "false", "no"})
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="flood-daily-summary-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
        self._thread = None

    def run_once(self, *, now: datetime | None = None) -> list[DailyReportRecord]:
        if not self.enabled:
            return []
        tz = ZoneInfo(self.timezone_name)
        now_utc = (now or datetime.now(UTC)).astimezone(UTC)
        local_now = now_utc.astimezone(tz)
        hour, minute = [int(part) for part in self.run_time_local.split(":", maxsplit=1)]
        scheduled_local = datetime.combine(local_now.date(), time(hour=hour, minute=minute), tzinfo=tz)
        if local_now < scheduled_local:
            return []

        report_date = local_now.date() - timedelta(days=1)
        start_local = datetime.combine(report_date, time.min, tzinfo=tz)
        end_local = datetime.combine(report_date + timedelta(days=1), time.min, tzinfo=tz)
        start_utc = start_local.astimezone(UTC)
        end_utc = end_local.astimezone(UTC)

        generated: list[DailyReportRecord] = []
        for event in self.repository.list_v2_events(limit=500):
            if self.repository.get_v2_daily_report_run(event.event_id, report_date.isoformat()) is not None:
                continue
            activity_counts = self._activity_counts(event.event_id, start_utc, end_utc)
            if not any(activity_counts.values()):
                continue
            report = self._generate_report(
                event=event,
                report_date=report_date,
                start_utc=start_utc,
                end_utc=end_utc,
                activity_counts=activity_counts,
            )
            self.repository.save_v2_daily_report(report)
            delivered_ids = self._deliver_report(report)
            if delivered_ids:
                report = report.model_copy(update={"delivered_session_ids": delivered_ids})
                self.repository.save_v2_daily_report(report)
            self.repository.save_v2_daily_report_run(
                DailyReportRunRecord(
                    run_id=f"daily_run_{uuid4().hex[:12]}",
                    event_id=event.event_id,
                    report_date=report_date,
                    timezone=self.timezone_name,
                    status="completed",
                    report_id=report.report_id,
                    created_at=datetime.now(timezone.utc),
                )
            )
            self.platform.add_audit_record(
                source_type="daily_summary",
                action="daily_report_generated",
                summary=f"已生成事件 {event.event_id} 在 {report_date.isoformat()} 的前一日值班日报。",
                details={"report_id": report.report_id, "event_id": event.event_id, "report_date": report_date.isoformat()},
                event_id=event.event_id,
            )
            generated.append(report)
        return generated

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.poll_seconds):
            try:
                self.run_once()
            except Exception as exc:
                self.platform.add_audit_record(
                    source_type="daily_summary",
                    action="daily_report_failed",
                    summary="值班日报调度执行失败。",
                    details={"error": str(exc)},
                    severity=AlertSeverity.WARNING,
                )

    def _generate_report(
        self,
        *,
        event: EventRecord,
        report_date: date,
        start_utc: datetime,
        end_utc: datetime,
        activity_counts: dict[str, int],
    ) -> DailyReportRecord:
        shared = self.platform.get_shared_memory_snapshot(event.event_id)
        resolved_logs = self.platform.repository.list_v2_execution_logs(event.event_id)
        proposals = self.platform.repository.list_v2_action_proposals(event.event_id)
        proposals_in_range = [item for item in proposals if _dt_in_range(item.updated_at or item.created_at, start_utc, end_utc)]
        payload = {
            "event": event.model_dump(mode="json"),
            "report_date": report_date.isoformat(),
            "timezone": self.timezone_name,
            "activity_counts": activity_counts,
            "latest_risk_level": shared.latest_hazard_level.value if shared.latest_hazard_level else event.current_risk_level.value,
            "open_questions": shared.open_questions[:5],
            "blocked_by": shared.blocked_by[:5],
            "decision_path": shared.active_decision_path[-6:],
            "recent_logs": [item.model_dump(mode="json") for item in resolved_logs[:8]],
            "recent_proposals": [item.model_dump(mode="json") for item in proposals_in_range[:8]],
        }
        try:
            output = self.platform.llm_gateway.generate_daily_operations_summary(payload)
            generation_source = GenerationSource.LLM
            model_name = self.platform.llm_gateway.model_name
            grounding_summary = output.grounding_summary
            headline = output.headline
            situation_summary = output.situation_summary
            decisions_summary = output.decisions_summary
            action_summary = output.action_summary
            unresolved_risks = output.unresolved_risks
            next_day_recommendations = output.next_day_recommendations
        except Exception:
            generation_source = GenerationSource.SYSTEM
            model_name = ""
            grounding_summary = "基于前一日活动计数、共享记忆和执行日志生成。"
            latest_risk_level = shared.latest_hazard_level.value if shared.latest_hazard_level else event.current_risk_level.value
            headline = f"{report_date.isoformat()} 前一日值班日报：{event.title}"
            situation_summary = (
                f"{event.title} 在统计窗口内维持 {latest_risk_level} 风险态势，"
                f"新增 {activity_counts['observations']} 条观测、{activity_counts['simulation_updates']} 次模拟更新。"
            )
            decisions_summary = (
                f"系统完成 {activity_counts['supervisor_runs']} 次调度巡检，"
                f"处理 {activity_counts['proposals']} 条请示，并沉淀 {activity_counts['audit_records']} 条审计记录。"
            )
            action_summary = (
                f"期间形成 {activity_counts['notification_drafts']} 份通知草稿，"
                f"记录 {activity_counts['execution_logs']} 条执行日志。"
            )
            unresolved_risks = shared.open_questions[:4] or shared.blocked_by[:4]
            next_day_recommendations = [
                "优先跟进仍未关闭的高风险对象与阻塞项。",
                "复核共享记忆中的待决事项、执行缺口与资源约束。",
            ]
        return DailyReportRecord(
            report_id=f"daily_{uuid4().hex[:12]}",
            event_id=event.event_id,
            report_date=report_date,
            timezone=self.timezone_name,
            headline=headline,
            situation_summary=situation_summary,
            decisions_summary=decisions_summary,
            action_summary=action_summary,
            unresolved_risks=_coerce_strings(unresolved_risks, limit=5),
            next_day_recommendations=_coerce_strings(next_day_recommendations, limit=5),
            generation_source=generation_source,
            model_name=model_name,
            grounding_summary=grounding_summary,
            created_at=datetime.now(timezone.utc),
        )

    def _deliver_report(self, report: DailyReportRecord) -> list[str]:
        return _deliver_event_message(
            self.repository,
            event_id=report.event_id,
            content=format_daily_report_message(report),
            created_at=report.created_at,
            delivered_session_ids=report.delivered_session_ids,
        )

    def _activity_counts(self, event_id: str, start_utc: datetime, end_utc: datetime) -> dict[str, int]:
        observations = self.repository.list_v2_observations(event_id)
        simulation_updates = self.repository.list_v2_simulation_updates(event_id, limit=200)
        stream_records = self.repository.list_v2_stream_records(event_id, limit=300)
        supervisor_runs = self.repository.list_v2_supervisor_runs(event_id, limit=200)
        proposals = self.repository.list_v2_action_proposals(event_id)
        notification_drafts = self.repository.list_v2_notification_drafts(event_id)
        execution_logs = self.repository.list_v2_execution_logs(event_id)
        audit_records = self.repository.list_audit_records(
            event_id=event_id,
            from_ts=start_utc.isoformat(),
            to_ts=end_utc.isoformat(),
            limit=300,
        )
        return {
            "observations": sum(1 for item in observations if _dt_in_range(item.observed_at, start_utc, end_utc)),
            "simulation_updates": sum(1 for item in simulation_updates if _dt_in_range(item.generated_at, start_utc, end_utc)),
            "stream_records": sum(1 for item in stream_records if _dt_in_range(item.created_at, start_utc, end_utc)),
            "supervisor_runs": sum(1 for item in supervisor_runs if _dt_in_range(item.created_at, start_utc, end_utc)),
            "proposals": sum(1 for item in proposals if _dt_in_range(item.updated_at or item.created_at, start_utc, end_utc)),
            "notification_drafts": sum(1 for item in notification_drafts if _dt_in_range(item.created_at, start_utc, end_utc)),
            "execution_logs": sum(1 for item in execution_logs if _dt_in_range(item.created_at, start_utc, end_utc)),
            "audit_records": len(audit_records),
        }

@dataclass
class EventPostmortemService:
    repository: object
    platform: object
    long_term_memory_store: LongTermMemoryStore
    poll_seconds: float = 60.0
    enabled: bool = field(default_factory=lambda: os.getenv("FLOOD_EVENT_POSTMORTEM_ENABLED", "1").strip().lower() not in {"0", "false", "no"})
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="flood-postmortem-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
        self._thread = None

    def sync_risk_transition(
        self,
        *,
        event: EventRecord,
        previous_risk_level: RiskLevel | None,
        current_risk_level: RiskLevel,
        trigger_source: str,
        observed_at: datetime | None = None,
    ) -> None:
        if not self.enabled:
            return
        timestamp = observed_at or datetime.now(timezone.utc)
        open_episodes = self.repository.list_open_v2_high_risk_episodes(event.event_id)
        is_current_high = current_risk_level in HIGH_RISK_LEVELS
        was_previous_high = previous_risk_level in HIGH_RISK_LEVELS

        if is_current_high and not open_episodes:
            episode = HighRiskEpisodeRecord(
                episode_id=f"episode_{uuid4().hex[:12]}",
                event_id=event.event_id,
                started_at=timestamp,
                peak_risk_level=current_risk_level,
                trigger_source=trigger_source,
                status=HighRiskEpisodeStatus.OPEN,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self.repository.save_v2_high_risk_episode(episode)
            return

        if is_current_high and open_episodes:
            episode = open_episodes[0]
            peak_level = current_risk_level if _risk_rank(current_risk_level) > _risk_rank(episode.peak_risk_level) else episode.peak_risk_level
            if peak_level != episode.peak_risk_level:
                self.repository.save_v2_high_risk_episode(
                    episode.model_copy(update={"peak_risk_level": peak_level, "updated_at": timestamp})
                )
            return

        if not is_current_high and was_previous_high:
            for episode in open_episodes:
                self.repository.save_v2_high_risk_episode(
                    episode.model_copy(
                        update={
                            "ended_at": timestamp,
                            "status": HighRiskEpisodeStatus.CLOSED,
                            "updated_at": timestamp,
                        }
                    )
                )

    def run_once(self) -> list[EventEpisodeSummaryRecord]:
        if not self.enabled:
            return []
        created: list[EventEpisodeSummaryRecord] = []
        for episode in self.repository.list_pending_v2_episode_summaries(limit=50):
            summary = self._summarize_episode(episode)
            self.repository.save_v2_event_episode_summary(summary)
            self.repository.save_v2_high_risk_episode(
                episode.model_copy(update={"status": HighRiskEpisodeStatus.SUMMARIZED, "updated_at": datetime.now(timezone.utc)})
            )
            memory = self._build_long_term_memory(summary)
            self.long_term_memory_store.save_memory(memory)
            self._deliver_summary(summary)
            self.platform.add_audit_record(
                source_type="postmortem",
                action="episode_postmortem_generated",
                summary=f"已完成事件 {episode.event_id} 的高风险阶段复盘。",
                details={"summary_id": summary.summary_id, "episode_id": episode.episode_id},
                event_id=episode.event_id,
            )
            created.append(summary)
        return created

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.poll_seconds):
            try:
                self.run_once()
            except Exception as exc:
                self.platform.add_audit_record(
                    source_type="postmortem",
                    action="episode_postmortem_failed",
                    summary="高风险阶段复盘生成失败。",
                    details={"error": str(exc)},
                    severity=AlertSeverity.WARNING,
                )

    def _summarize_episode(self, episode: HighRiskEpisodeRecord) -> EventEpisodeSummaryRecord:
        event = self.platform.get_event(episode.event_id)
        end_time = episode.ended_at or datetime.now(timezone.utc)
        proposals = self.platform.repository.list_v2_action_proposals(event.event_id)
        relevant_proposals = [
            item
            for item in proposals
            if _dt_in_range(item.created_at, episode.started_at, end_time) or _dt_in_range(item.resolved_at, episode.started_at, end_time)
        ]
        execution_logs = [
            item
            for item in self.platform.repository.list_v2_execution_logs(event.event_id)
            if _dt_in_range(item.created_at, episode.started_at, end_time)
        ]
        audit_records = self.platform.repository.list_audit_records(
            event_id=event.event_id,
            from_ts=episode.started_at.isoformat(),
            to_ts=end_time.isoformat(),
            limit=300,
        )
        shared = self.platform.get_shared_memory_snapshot(event.event_id)
        approved_actions = [item.title for item in relevant_proposals if item.status.value == "approved"]
        rejected_actions = [item.title for item in relevant_proposals if item.status.value == "rejected"]
        payload = {
            "event": event.model_dump(mode="json"),
            "episode": episode.model_dump(mode="json"),
            "approved_actions": approved_actions,
            "rejected_actions": rejected_actions,
            "execution_logs": [item.model_dump(mode="json") for item in execution_logs[:10]],
            "open_questions": shared.open_questions[:6],
            "decision_path": shared.active_decision_path[-8:],
            "audit_records": [item.model_dump(mode="json") for item in audit_records[:12]],
        }
        try:
            output = self.platform.llm_gateway.generate_high_risk_postmortem_summary(payload)
            headline = output.headline
            escalation_path = output.escalation_path
            key_decisions = output.key_decisions
            successful_actions = output.successful_actions
            failed_actions = output.failed_actions
            coordination_gaps = output.coordination_gaps
            reusable_rules = output.reusable_rules
            memory_tags = output.memory_tags
        except Exception:
            headline = f"高风险事件复盘：{event.title}"
            escalation_path = [*shared.active_decision_path[-4:], f"峰值风险等级达到 {episode.peak_risk_level.value}。"]
            key_decisions = _coerce_strings([*approved_actions[:3], *rejected_actions[:2]])
            successful_actions = _coerce_strings([item.summary for item in execution_logs[:3]])
            failed_actions = _coerce_strings(rejected_actions[:3])
            coordination_gaps = _coerce_strings(shared.open_questions[:3] or shared.blocked_by[:3])
            reusable_rules = [
                "进入 Orange/Red 后应优先形成结构化请示，并保留人工审批门禁。",
                "执行前需复核共享记忆中的阻塞项、资源缺口和协同依赖。",
            ]
            memory_tags = [episode.peak_risk_level.value.lower(), event.area_id, "postmortem"]
        return EventEpisodeSummaryRecord(
            summary_id=f"episode_summary_{uuid4().hex[:12]}",
            episode_id=episode.episode_id,
            event_id=episode.event_id,
            headline=headline,
            escalation_path=_coerce_strings(escalation_path, limit=8),
            key_decisions=_coerce_strings(key_decisions, limit=8),
            successful_actions=_coerce_strings(successful_actions, limit=8),
            failed_actions=_coerce_strings(failed_actions, limit=8),
            coordination_gaps=_coerce_strings(coordination_gaps, limit=6),
            reusable_rules=_coerce_strings(reusable_rules, limit=6),
            memory_tags=_coerce_strings(memory_tags, limit=8),
            created_at=datetime.now(timezone.utc),
        )

    def _build_long_term_memory(self, summary: EventEpisodeSummaryRecord) -> LongTermMemoryRecord:
        event = self.platform.get_event(summary.event_id)
        episode = next((item for item in self.repository.list_v2_high_risk_episodes(summary.event_id) if item.episode_id == summary.episode_id), None)
        proposals = self.platform.repository.list_v2_action_proposals(summary.event_id)
        related_proposals = [item for item in proposals if item.risk_stage_key == summary.episode_id or _dt_in_range(item.created_at, episode.started_at, episode.ended_at or datetime.now(timezone.utc))] if episode else proposals
        entity_types: list[str] = []
        for proposal in related_proposals:
            for entity_id in proposal.high_risk_object_ids:
                try:
                    entity_type = self.platform.get_entity_profile(entity_id).entity_type.value
                except Exception:
                    continue
                if entity_type not in entity_types:
                    entity_types.append(entity_type)
        action_types = list(dict.fromkeys([item.action_type for item in related_proposals if item.action_type]))
        lessons = summary.reusable_rules[:4]
        pitfalls = summary.coordination_gaps[:4] + summary.failed_actions[:2]
        recommendations = summary.successful_actions[:3] or summary.key_decisions[:3]
        retrieval_lines = [
            summary.headline,
            *summary.escalation_path,
            *summary.key_decisions,
            *lessons,
            *pitfalls,
            *recommendations,
        ]
        return LongTermMemoryRecord(
            memory_id=f"ltm_{uuid4().hex[:12]}",
            event_id=summary.event_id,
            source_summary_id=summary.summary_id,
            area_id=event.area_id,
            risk_level=episode.peak_risk_level if episode else event.current_risk_level,
            entity_types=entity_types,
            action_types=action_types,
            tags=list(dict.fromkeys([*summary.memory_tags, event.area_id, "high_risk_episode"])),
            headline=summary.headline,
            summary="；".join(summary.key_decisions[:3] or summary.successful_actions[:3] or [summary.headline]),
            retrieval_text="\n".join(line for line in retrieval_lines if line),
            lessons=lessons,
            pitfalls=pitfalls,
            recommendations=recommendations,
            created_at=summary.created_at,
        )

    def _deliver_summary(self, summary: EventEpisodeSummaryRecord) -> None:
        _deliver_event_message(
            self.repository,
            event_id=summary.event_id,
            content=format_episode_summary_message(summary),
            created_at=summary.created_at,
        )
