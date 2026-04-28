from __future__ import annotations

from datetime import datetime, timezone

from .models import (
    AgentMetricsView,
    AgentResult,
    AgentTask,
    AgentTimelineEntry,
    BatchProposalResolutionRequest,
    DailyReportRecord,
    DecisionReportView,
    EventEpisodeSummaryRecord,
    ExperienceContextView,
    LongTermMemoryRecord,
    MemoryBundleView,
    ProposalResolutionRequest,
    ReplayRequest,
    SessionMemoryView,
    SharedMemorySnapshot,
    StrategyHistoryView,
    SupervisorRunRecord,
    TriggerEvent,
    TriggerEventType,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
    V2CopilotSessionView,
)


class PlatformAgentOpsMixin:
    def bootstrap_copilot_session(self, request: V2CopilotSessionRequest) -> V2CopilotSessionView:
        return self.copilot.bootstrap_session(request)

    def get_copilot_session(self, session_id: str) -> V2CopilotSessionView:
        return self.copilot.get_session_view(session_id)

    def send_copilot_message(self, session_id: str, request: V2CopilotMessageRequest) -> V2CopilotSessionView:
        return self.copilot.answer(session_id, request.content)

    def approve_copilot_proposal(
        self,
        session_id: str,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.approve_proposal(session_id, proposal_id, request)

    def reject_copilot_proposal(
        self,
        session_id: str,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.reject_proposal(session_id, proposal_id, request)

    def batch_approve_copilot_proposals(
        self,
        session_id: str,
        request: BatchProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.batch_approve_proposals(session_id, request)

    def batch_reject_copilot_proposals(
        self,
        session_id: str,
        request: BatchProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        return self.copilot.batch_reject_proposals(session_id, request)

    def run_supervisor(self, event_id: str, *, trigger_type: str = "manual_run", session_id: str | None = None) -> SupervisorRunRecord:
        return self.agent_supervisor.run_for_event(event_id, trigger_type=trigger_type, session_id=session_id)

    def tick_supervisor(self, event_id: str | None = None) -> list[SupervisorRunRecord]:
        return self.agent_supervisor.tick(event_id)

    def publish_trigger(
        self,
        event_id: str,
        *,
        trigger_type: TriggerEventType,
        payload: dict | None = None,
        dedupe: bool = True,
    ) -> TriggerEvent:
        trigger = self.agent_supervisor.publish_trigger(
            event_id,
            trigger_type=trigger_type,
            payload=payload,
            dedupe=dedupe,
        )
        self.add_audit_record(
            source_type="trigger_event",
            action="trigger_published",
            summary=f"已为事件 {event_id} 发布触发事件 {trigger_type.value}。",
            details={"trigger_id": trigger.trigger_id, "event_id": event_id},
            event_id=event_id,
        )
        return trigger

    def get_session_memory(self, session_id: str) -> SessionMemoryView:
        session = self.repository.get_v2_copilot_session(session_id)
        if session is None:
            raise ValueError("v2 copilot session not found.")
        event = self.get_event(session["event_id"])
        snapshot = self.session_memory_store.load_snapshot(session_id, area_id=event.area_id)
        events = self.repository.list_v2_copilot_memory_events(session_id, limit=20)
        return SessionMemoryView(session_id=session_id, memory_snapshot=snapshot, recent_events=events)

    def get_event_shared_memory(self, event_id: str) -> SharedMemorySnapshot:
        return self.agent_supervisor.shared_memory.load(event_id)

    def get_memory_bundle(self, session_id: str, event_id: str) -> MemoryBundleView:
        return MemoryBundleView(
            session_memory=self.get_session_memory(session_id),
            event_shared_memory=self.get_event_shared_memory(event_id),
        )

    def list_daily_reports(self, event_id: str) -> list[DailyReportRecord]:
        return self.repository.list_v2_daily_reports(event_id, limit=20)

    def list_episode_summaries(self, event_id: str) -> list[EventEpisodeSummaryRecord]:
        return self.repository.list_v2_event_episode_summaries(event_id, limit=20)

    def list_long_term_memories(self, event_id: str) -> list[LongTermMemoryRecord]:
        return self.repository.list_v2_long_term_memories(event_id=event_id, limit=20)

    def get_experience_context(self, event_id: str) -> ExperienceContextView:
        event = self.get_event(event_id)
        shared = self.get_event_shared_memory(event_id)
        records = self.operational_experience_store.query_similar_cases(
            entity_type=None,
            risk_level=shared.latest_hazard_level,
            limit=8,
        )
        entity_type: str | None = None
        if records and records[0].entity_type:
            entity_type = records[0].entity_type
        elif shared.focus_entity_ids:
            try:
                entity_type = self.get_entity_profile(shared.focus_entity_ids[0]).entity_type.value
            except ValueError:
                entity_type = None
        patterns = self.operational_experience_store.rank_strategy_patterns(
            entity_type=entity_type,
            risk_level=shared.latest_hazard_level,
            limit=5,
        )
        notes: list[str] = []
        if patterns:
            top = patterns[0]
            action_label = {
                "regional_notification": "区域通知",
                "regional_evacuation": "区域转移建议",
                "regional_resource_dispatch": "区域资源调度",
            }.get(top.action_type, top.action_type)
            notes.append(
                f"区域 {event.area_id} 的历史主导模式为 {action_label}，在 {top.sample_size} 个样本中的批准率为 {top.approval_rate:.0%}。"
            )
        if not records:
            notes.append("当前可用的历史经验仍较少，建议主要依赖实时态势与策略规则。")
        long_term_memories = self.long_term_memory_store.query_memories(
            area_id=event.area_id,
            entity_type=entity_type,
            risk_level=shared.latest_hazard_level,
            top_k=5,
        )
        return ExperienceContextView(
            event_id=event_id,
            relevant_records=records,
            strategy_patterns=patterns,
            outcome_risk_notes=notes,
            long_term_memories=long_term_memories,
        )

    def get_strategy_history(self, entity_id: str) -> StrategyHistoryView:
        entity = self.get_entity_profile(entity_id)
        records = self.operational_experience_store.query_similar_cases(
            entity_id=entity_id,
            entity_type=entity.entity_type.value,
            limit=20,
        )
        patterns = self.operational_experience_store.rank_strategy_patterns(
            entity_type=entity.entity_type.value,
            limit=5,
        )
        return StrategyHistoryView(entity_id=entity_id, records=records, patterns=patterns)

    def get_shared_memory_snapshot(self, event_id: str) -> SharedMemorySnapshot:
        return self.agent_supervisor.shared_memory.load(event_id)

    def list_agent_tasks(self, event_id: str) -> list[AgentTask]:
        return self.repository.list_v2_agent_tasks(event_id, limit=80)

    def list_agent_results(self, event_id: str) -> list[AgentResult]:
        return self.repository.list_v2_agent_results(event_id, limit=40)

    def list_supervisor_runs(self, event_id: str) -> list[SupervisorRunRecord]:
        return self.repository.list_v2_supervisor_runs(event_id, limit=30)

    def list_trigger_events(self, event_id: str) -> list[TriggerEvent]:
        return self.repository.list_v2_trigger_events(event_id, limit=40)

    def list_agent_timeline(self, event_id: str) -> list[AgentTimelineEntry]:
        return self.repository.list_v2_agent_timeline(event_id, limit=120)

    def get_decision_report(self, event_id: str) -> DecisionReportView:
        shared = self.get_shared_memory_snapshot(event_id)
        return DecisionReportView(
            event_id=event_id,
            latest_summary=shared.latest_summary,
            active_decision_path=shared.active_decision_path,
            blocked_by=shared.blocked_by,
            open_questions=shared.open_questions,
            recent_result_ids=shared.recent_result_ids,
        )

    def replay_agent_task(self, task_id: str, request: ReplayRequest) -> AgentTask:
        task = self.agent_supervisor.replay_task(task_id, replay_reason=request.replay_reason)
        self.add_audit_record(
            source_type="agent_task",
            action="agent_task_replayed",
            summary=f"Replayed task {task_id}.",
            details={"new_task_id": task.task_id, "replay_reason": request.replay_reason},
            event_id=task.event_id,
            session_id=task.session_id,
        )
        return task

    def get_agent_status(self, event_id: str) -> dict:
        return self.agent_supervisor.get_agent_status(event_id)

    def get_agent_metrics(self) -> AgentMetricsView:
        recent_reports = self.repository.list_v2_evaluation_reports(limit=1)
        with self.repository._connect() as conn:
            failure_rows = conn.execute(
                """
                SELECT agent_name, COUNT(*) AS count
                FROM v2_agent_task_events
                WHERE event_type = 'task_failed'
                GROUP BY agent_name
                """
            ).fetchall()
            task_rows = conn.execute("SELECT payload FROM v2_agent_tasks").fetchall()
            result_rows = conn.execute("SELECT payload FROM v2_agent_results").fetchall()
        total_tasks = len(task_rows)
        superseded = 0
        fanout_count = 0
        for row in task_rows:
            task = AgentTask.model_validate_json(row["payload"])
            if task.status.value == "superseded":
                superseded += 1
        stale_data_frequency = 0
        for row in result_rows:
            result = AgentResult.model_validate_json(row["payload"])
            if any("stale" in slot.lower() for slot in result.missing_slots):
                stale_data_frequency += 1
            fanout_count += len(result.recommended_next_tasks)
        latency_ms = 0.0
        recent_runs = []
        for area_id in self.area_profiles:
            event_id = self.repository.get_latest_v2_event_id(area_id)
            if event_id:
                recent_runs.extend(self.repository.list_v2_supervisor_runs(event_id, limit=5))
        durations = [
            max((run.completed_at - run.created_at).total_seconds() * 1000, 0.0)
            for run in recent_runs
            if run.completed_at is not None
        ]
        if durations:
            latency_ms = round(sum(durations) / len(durations), 1)
        return AgentMetricsView(
            generated_at=datetime.now(timezone.utc),
            task_graph_latency_ms=latency_ms,
            agent_failure_heatmap={str(row["agent_name"]): int(row["count"]) for row in failure_rows},
            stale_data_frequency=stale_data_frequency,
            auto_retry_success_rate=recent_reports[0].tool_selection_correctness if recent_reports else 0.0,
            superseded_task_ratio=round(superseded / total_tasks, 2) if total_tasks else 0.0,
            fanout_count=fanout_count,
            stale_data_replan_count=stale_data_frequency,
        )
