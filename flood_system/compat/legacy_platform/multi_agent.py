from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .llm_gateway import LLMGenerationError
from .models import (
    AgentName,
    AgentResult,
    AgentTask,
    AgentTaskEvent,
    AgentTaskEventType,
    AgentTaskStatus,
    AlertSeverity,
    ArchiveStatusView,
    AutonomyDecision,
    AutonomyLevel,
    CircuitState,
    ProposalStatus,
    RiskLevel,
    SharedMemorySnapshot,
    SupervisorHealthState,
    SupervisorRunRecord,
    SupervisorRunStatus,
    TriggerEvent,
    TriggerEventType,
)


class TaskQueue:
    def enqueue(
        self,
        *,
        event_id: str,
        agent_name: AgentName,
        task_type: str,
        input_payload: dict,
        priority: int = 0,
        session_id: str | None = None,
        parent_task_id: str | None = None,
        replayed_from_task_id: str | None = None,
        replay_reason: str | None = None,
        source_trigger_id: str | None = None,
    ) -> AgentTask:
        raise NotImplementedError

    def claim(self, task_id: str) -> AgentTask:
        raise NotImplementedError

    def complete(self, task: AgentTask, *, output_payload: dict) -> AgentTask:
        raise NotImplementedError

    def fail(self, task: AgentTask, *, failure_reason: str) -> AgentTask:
        raise NotImplementedError

    def cancel(self, task_id: str, *, reason: str) -> AgentTask:
        raise NotImplementedError

    def replay(self, task_id: str, *, replay_reason: str = "") -> AgentTask:
        raise NotImplementedError


class TriggerEventBus:
    def publish(
        self,
        *,
        event_id: str,
        trigger_type: TriggerEventType,
        payload: dict | None = None,
        dedupe: bool = True,
        dedupe_window_seconds: int = 15,
    ) -> TriggerEvent:
        raise NotImplementedError

    def lease(self) -> TriggerEvent | None:
        raise NotImplementedError

    def ack(self, trigger_id: str) -> TriggerEvent | None:
        raise NotImplementedError

    def fail(self, trigger_id: str, *, error_message: str) -> TriggerEvent | None:
        raise NotImplementedError


class SQLiteTaskQueue(TaskQueue):
    def __init__(self, repository) -> None:
        self.repository = repository

    def enqueue(
        self,
        *,
        event_id: str,
        agent_name: AgentName,
        task_type: str,
        input_payload: dict,
        priority: int = 0,
        session_id: str | None = None,
        parent_task_id: str | None = None,
        replayed_from_task_id: str | None = None,
        replay_reason: str | None = None,
        source_trigger_id: str | None = None,
    ) -> AgentTask:
        task = AgentTask(
            task_id=f"task_{uuid4().hex[:10]}",
            event_id=event_id,
            agent_name=agent_name,
            task_type=task_type,
            status=AgentTaskStatus.PENDING,
            input_payload=input_payload,
            priority=priority,
            session_id=session_id,
            parent_task_id=parent_task_id,
            replayed_from_task_id=replayed_from_task_id,
            replay_reason=replay_reason,
            source_trigger_id=source_trigger_id,
            created_at=datetime.now(timezone.utc),
        )
        self.repository.save_v2_agent_task(task)
        self._append_task_event(
            task,
            AgentTaskEventType.TASK_ENQUEUED,
            {"summary": f"{task.agent_name.value} 已进入任务队列，等待执行 {task.task_type}。", "source_trigger_id": source_trigger_id},
        )
        return task

    def claim(self, task_id: str) -> AgentTask:
        task = self.repository.get_v2_agent_task(task_id)
        if task is None:
            raise ValueError("agent task not found.")
        claimed = task.model_copy(update={"status": AgentTaskStatus.RUNNING, "started_at": datetime.now(timezone.utc)})
        self.repository.save_v2_agent_task(claimed)
        self._append_task_event(
            claimed,
            AgentTaskEventType.TASK_CLAIMED,
            {"summary": f"{claimed.agent_name.value} 已领取任务 {claimed.task_type}。"},
        )
        return claimed

    def complete(self, task: AgentTask, *, output_payload: dict) -> AgentTask:
        completed = task.model_copy(
            update={
                "status": AgentTaskStatus.COMPLETED,
                "output_payload": output_payload,
                "completed_at": datetime.now(timezone.utc),
            }
        )
        self.repository.save_v2_agent_task(completed)
        self._append_task_event(
            completed,
            AgentTaskEventType.TASK_COMPLETED,
            {"summary": f"{completed.agent_name.value} 已完成任务 {completed.task_type}。"},
        )
        return completed

    def fail(self, task: AgentTask, *, failure_reason: str) -> AgentTask:
        failed = task.model_copy(
            update={
                "status": AgentTaskStatus.FAILED,
                "failure_reason": failure_reason,
                "completed_at": datetime.now(timezone.utc),
            }
        )
        self.repository.save_v2_agent_task(failed)
        self._append_task_event(
            failed,
            AgentTaskEventType.TASK_FAILED,
            {"summary": f"{failed.agent_name.value} 在执行 {failed.task_type} 时失败。", "failure_reason": failure_reason},
        )
        return failed

    def cancel(self, task_id: str, *, reason: str) -> AgentTask:
        task = self.repository.get_v2_agent_task(task_id)
        if task is None:
            raise ValueError("agent task not found.")
        canceled = task.model_copy(
            update={"status": AgentTaskStatus.CANCELED, "failure_reason": reason, "completed_at": datetime.now(timezone.utc)}
        )
        self.repository.save_v2_agent_task(canceled)
        self._append_task_event(
            canceled,
            AgentTaskEventType.TASK_FAILED,
            {"summary": f"{canceled.agent_name.value} 已取消任务 {canceled.task_type}。", "failure_reason": reason},
        )
        return canceled

    def replay(self, task_id: str, *, replay_reason: str = "") -> AgentTask:
        original = self.repository.get_v2_agent_task(task_id)
        if original is None:
            raise ValueError("agent task not found.")
        if original.status not in {AgentTaskStatus.COMPLETED, AgentTaskStatus.FAILED}:
            raise ValueError("只有已完成或已失败的任务才能重放。")
        self._append_task_event(
            original,
            AgentTaskEventType.REPLAY_REQUESTED,
            {"summary": f"已请求重放任务 {original.task_type}。", "replay_reason": replay_reason},
        )
        replay_task = self.enqueue(
            event_id=original.event_id,
            agent_name=original.agent_name,
            task_type=original.task_type,
            input_payload=original.input_payload,
            priority=original.priority,
            session_id=original.session_id,
            parent_task_id=original.parent_task_id,
            replayed_from_task_id=original.task_id,
            replay_reason=replay_reason,
            source_trigger_id=original.source_trigger_id,
        )
        self._append_task_event(
            replay_task,
            AgentTaskEventType.REPLAY_COMPLETED,
            {"summary": f"任务 {original.task_type} 已重新入队等待重放。", "replayed_from_task_id": original.task_id, "replay_reason": replay_reason},
        )
        return replay_task

    def _append_task_event(self, task: AgentTask, event_type: AgentTaskEventType, payload: dict) -> None:
        self.repository.add_v2_agent_task_event(
            AgentTaskEvent(
                task_event_id=f"tke_{uuid4().hex[:10]}",
                event_id=task.event_id,
                task_id=task.task_id,
                agent_name=task.agent_name,
                event_type=event_type,
                trigger_id=task.source_trigger_id,
                payload=payload,
                created_at=datetime.now(timezone.utc),
            )
        )


class SQLiteTriggerEventBus(TriggerEventBus):
    def __init__(self, repository) -> None:
        self.repository = repository

    def publish(
        self,
        *,
        event_id: str,
        trigger_type: TriggerEventType,
        payload: dict | None = None,
        dedupe: bool = True,
        dedupe_window_seconds: int = 15,
    ) -> TriggerEvent:
        dedupe_key = f"{event_id}:{trigger_type.value}" if dedupe else None
        trigger = TriggerEvent(
            trigger_id=f"trg_{uuid4().hex[:10]}",
            event_id=event_id,
            trigger_type=trigger_type,
            payload=payload or {},
            dedupe_key=dedupe_key,
            created_at=datetime.now(timezone.utc),
        )
        return self.repository.publish_v2_trigger_event(trigger, dedupe_window_seconds=dedupe_window_seconds)

    def lease(self) -> TriggerEvent | None:
        return self.repository.lease_next_v2_trigger_event()

    def ack(self, trigger_id: str) -> TriggerEvent | None:
        return self.repository.mark_v2_trigger_event_processed(trigger_id)

    def fail(self, trigger_id: str, *, error_message: str) -> TriggerEvent | None:
        return self.repository.mark_v2_trigger_event_failed(trigger_id, error_message=error_message)


class EventSharedMemoryStore:
    def __init__(self, repository) -> None:
        self.repository = repository

    def load(self, event_id: str) -> SharedMemorySnapshot:
        snapshot = self.repository.get_v2_event_shared_memory(event_id)
        if snapshot is not None:
            return snapshot
        snapshot = SharedMemorySnapshot(event_id=event_id, updated_at=datetime.now(timezone.utc))
        self.save(snapshot)
        return snapshot

    def save(self, snapshot: SharedMemorySnapshot) -> SharedMemorySnapshot:
        normalized = snapshot.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        self.repository.save_v2_event_shared_memory(normalized)
        return normalized

    def apply(
        self,
        event_id: str,
        *,
        result: AgentResult,
        autonomy_level: AutonomyLevel | None = None,
    ) -> SharedMemorySnapshot:
        snapshot = self.load(event_id)
        output = result.structured_output
        active_agents = list(dict.fromkeys([*snapshot.active_agents, result.agent_name]))
        focus_entity_ids = list(dict.fromkeys([*snapshot.focus_entity_ids, *self._coerce_strings(output.get("focus_entity_ids"))]))
        focus_entity_names = list(dict.fromkeys([*snapshot.focus_entity_names, *self._coerce_strings(output.get("focus_entity_names"))]))
        top_risks = list(dict.fromkeys([*snapshot.top_risks, *self._coerce_strings(output.get("top_risks"))]))
        actions = list(dict.fromkeys([*snapshot.recommended_actions, *self._coerce_strings(output.get("recommended_actions"))]))
        pending_proposals = list(dict.fromkeys([*snapshot.pending_proposal_ids, *self._coerce_strings(output.get("pending_proposal_ids"))]))
        unresolved = list(dict.fromkeys([*snapshot.unresolved_items, *result.missing_slots]))
        decision_path = list(
            dict.fromkeys(
                [
                    *snapshot.active_decision_path,
                    f"{result.agent_name.value}: {result.summary}",
                ]
            )
        )
        open_questions = list(
            dict.fromkeys(
                [
                    *snapshot.open_questions,
                    *result.missing_slots,
                    *self._coerce_strings(output.get("open_questions")),
                ]
            )
        )
        blocked_by = list(
            dict.fromkeys(
                [
                    *snapshot.blocked_by,
                    *([result.stop_reason] if result.stop_reason else []),
                    *self._coerce_strings(output.get("blocked_by")),
                ]
            )
        )
        recent_result_ids = [result.result_id, *snapshot.recent_result_ids][:10]
        updated = snapshot.model_copy(
            update={
                "active_agents": active_agents,
                "focus_entity_ids": focus_entity_ids[:5],
                "focus_entity_names": focus_entity_names[:5],
                "top_risks": top_risks[:8],
                "recommended_actions": actions[:8],
                "pending_proposal_ids": pending_proposals[:12],
                "recent_result_ids": recent_result_ids,
                "unresolved_items": unresolved[:10],
                "active_decision_path": decision_path[:12],
                "open_questions": open_questions[:10],
                "blocked_by": blocked_by[:6],
                "latest_hazard_level": output.get("latest_hazard_level", snapshot.latest_hazard_level),
                "latest_summary": result.summary,
                "last_trigger": output.get("trigger_type", snapshot.last_trigger),
                "autonomy_level": autonomy_level or snapshot.autonomy_level,
            }
        )
        return self.save(updated)

    @staticmethod
    def _coerce_strings(value) -> list[str]:
        return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


@dataclass
class AgentDecisionGraph:
    TASK_TYPES: dict[AgentName, str] = field(
        default_factory=lambda: {
            AgentName.HAZARD: "assess_hazard",
            AgentName.EXPOSURE: "assess_exposure",
            AgentName.RESOURCE: "assess_resources",
            AgentName.PLANNING: "draft_plan",
            AgentName.POLICY: "assess_policy",
            AgentName.COMMS: "draft_comms",
        }
    )

    def initial_steps(self, trigger_type: str) -> list[tuple[AgentName, str]]:
        if trigger_type in {
            TriggerEventType.RESOURCE_OVERRIDE_UPDATED.value,
            TriggerEventType.RESOURCE_OVERRIDE_DELETED.value,
        }:
            return [(AgentName.RESOURCE, self.TASK_TYPES[AgentName.RESOURCE]), (AgentName.PLANNING, self.TASK_TYPES[AgentName.PLANNING])]
        if trigger_type == TriggerEventType.PROPOSAL_RESOLVED.value:
            return []
        if trigger_type == TriggerEventType.COPILOT_ESCALATION_REQUESTED.value:
            return [(AgentName.EXPOSURE, self.TASK_TYPES[AgentName.EXPOSURE]), (AgentName.PLANNING, self.TASK_TYPES[AgentName.PLANNING])]
        return [(AgentName.HAZARD, self.TASK_TYPES[AgentName.HAZARD])]

    def next_steps(
        self,
        *,
        trigger_type: str,
        latest_result: AgentResult,
        shared_memory: SharedMemorySnapshot,
    ) -> tuple[list[tuple[AgentName, str]], str | None]:
        if latest_result.stop_reason:
            return ([], latest_result.stop_reason)
        if latest_result.agent_name == AgentName.HAZARD:
            return (
                [
                    (AgentName.EXPOSURE, self.TASK_TYPES[AgentName.EXPOSURE]),
                    (AgentName.RESOURCE, self.TASK_TYPES[AgentName.RESOURCE]),
                ],
                None,
            )
        if latest_result.agent_name == AgentName.EXPOSURE:
            if trigger_type == TriggerEventType.SIMULATION_UPDATED.value:
                return ([(AgentName.PLANNING, self.TASK_TYPES[AgentName.PLANNING])], None)
            if not latest_result.structured_output.get("focus_entity_ids"):
                return ([], "当前没有识别到暴露对象，任务图已保守收敛。")
            return ([(AgentName.PLANNING, self.TASK_TYPES[AgentName.PLANNING])], None)
        if latest_result.agent_name == AgentName.RESOURCE:
            return ([(AgentName.PLANNING, self.TASK_TYPES[AgentName.PLANNING])], None)
        if latest_result.agent_name == AgentName.PLANNING:
            if latest_result.structured_output.get("pending_proposal_ids"):
                return ([(AgentName.POLICY, self.TASK_TYPES[AgentName.POLICY])], None)
            if latest_result.missing_slots:
                return ([], "Planning could only produce a conservative answer because key dependencies are still missing.")
            return ([(AgentName.POLICY, self.TASK_TYPES[AgentName.POLICY])], None)
        if latest_result.agent_name == AgentName.POLICY:
            autonomy = latest_result.structured_output.get("autonomy_level")
            if autonomy == AutonomyLevel.HUMAN_GATE_REQUIRED:
                return ([], "策略门禁要求先完成人工确认，任务图才能继续推进。")
            if trigger_type == TriggerEventType.SIMULATION_UPDATED.value:
                return ([], "当前模拟更新的区域策略审查已完成。")
            if shared_memory.pending_proposal_ids or latest_result.structured_output.get("recommended_actions"):
                return ([(AgentName.COMMS, self.TASK_TYPES[AgentName.COMMS])], None)
            return ([], "策略审查已完成，当前不需要追加通信跟进。")
        return ([], "当前任务图已经到达终止节点。")


@dataclass
class AgentRuntime:
    platform: object
    repository: object

    def run(self, task: AgentTask) -> AgentResult:
        event_id = task.event_id
        payload = task.input_payload
        if task.agent_name == AgentName.HAZARD:
            hazard = self.platform.get_hazard_state(event_id)
            summary = (
                f"风险态势代理确认当前总体风险为 {hazard.overall_risk_level.value}，"
                f"趋势为 {hazard.trend}，数据新鲜度约为 {hazard.freshness_seconds} 秒。"
            )
            output = {
                "latest_hazard_level": hazard.overall_risk_level,
                "top_risks": [f"{tile.area_name}: {tile.predicted_water_depth_cm:.0f} cm" for tile in hazard.hazard_tiles[:3]],
                "trigger_type": payload.get("trigger_type", ""),
            }
            return self._result(
                task,
                summary,
                output,
                confidence=0.92,
                decision_confidence=0.9,
                recommended_next_tasks=["assess_exposure", "assess_resources"],
            )

        if task.agent_name == AgentName.EXPOSURE:
            entity_type = payload.get("entity_type")
            exposure = self.platform.get_exposure_summary(event_id, entity_type=entity_type, top_k=5)
            focus_entities = exposure.affected_entities[:3]
            summary = f"暴露分析代理识别出 {len(exposure.affected_entities)} 个受影响对象。"
            output = {
                "focus_entity_ids": [item.entity.entity_id for item in focus_entities],
                "focus_entity_names": [item.entity.name for item in focus_entities],
                "top_risks": exposure.top_risks,
                "trigger_type": payload.get("trigger_type", ""),
            }
            missing = [] if exposure.affected_entities else ["当前没有识别到暴露对象。"]
            return self._result(
                task,
                summary,
                output,
                confidence=0.85,
                decision_confidence=0.78,
                missing_slots=missing,
                recommended_next_tasks=["draft_plan"] if focus_entities else [],
                stop_reason="当前没有识别到暴露对象，任务图已保守收敛。" if not focus_entities else None,
            )

        if task.agent_name == AgentName.RESOURCE:
            exposure = self.platform.get_exposure_summary(event_id, top_k=3)
            gaps: list[str] = []
            actions: list[str] = []
            for item in exposure.affected_entities[:2]:
                gaps.extend(item.resource_gap)
                if item.resource_gap:
                    actions.append(f"建议为 {item.entity.name} 预置资源：{', '.join(item.resource_gap[:2])}。")
            summary = "资源调度代理已结合高风险对象复核当前资源缺口。"
            output = {
                "top_risks": list(dict.fromkeys(gaps[:5])),
                "recommended_actions": actions,
                "trigger_type": payload.get("trigger_type", ""),
            }
            return self._result(
                task,
                summary,
                output,
                confidence=0.79,
                decision_confidence=0.74,
                missing_slots=[] if gaps else ["当前没有识别到明确的资源缺口信号。"],
                recommended_next_tasks=["draft_plan"],
            )

        if task.agent_name == AgentName.PLANNING:
            if payload.get("trigger_type") == TriggerEventType.SIMULATION_UPDATED.value:
                exposure = self.platform.get_exposure_summary(event_id, top_k=5)
                reconciliation = self.platform.reconcile_regional_proposals(
                    event_id,
                    previous_risk_level=payload.get("previous_risk_level"),
                )
                if reconciliation.get("llm_status") == "failed":
                    llm_error = str(reconciliation.get("llm_error") or "区域级主动决策模型当前不可用。")
                    return self._result(
                        task,
                        llm_error,
                        {
                            "focus_entity_ids": [],
                            "focus_entity_names": [],
                            "recommended_actions": [],
                            "pending_proposal_ids": [],
                            "risk_stage_key": None,
                            "trigger_type": payload.get("trigger_type", ""),
                            "llm_status": "failed",
                            "llm_error": llm_error,
                        },
                        confidence=0.22,
                        decision_confidence=0.18,
                        missing_slots=["区域级主动决策模型当前不可用。"],
                        stop_reason=llm_error,
                    )
                focus_entities = exposure.affected_entities[:3]
                pending_ids = reconciliation.get("pending_proposal_ids", [])
                summary = (
                    f"方案规划代理已为当前模拟更新生成 {len(pending_ids)} 条区域级请示。"
                    if pending_ids
                    else "方案规划代理判断当前模拟更新暂不需要新增区域级请示。"
                )
                return self._result(
                    task,
                    summary,
                    {
                        "focus_entity_ids": [item.entity.entity_id for item in focus_entities],
                        "focus_entity_names": [item.entity.name for item in focus_entities],
                        "recommended_actions": reconciliation.get("recommended_actions", []),
                        "pending_proposal_ids": pending_ids,
                        "risk_stage_key": reconciliation.get("risk_stage_key"),
                        "trigger_type": payload.get("trigger_type", ""),
                    },
                    confidence=0.84 if pending_ids else 0.68,
                    decision_confidence=0.8 if pending_ids else 0.62,
                    recommended_next_tasks=["assess_policy"],
                )
            exposure = self.platform.get_exposure_summary(event_id, top_k=1)
            if not exposure.affected_entities:
                return self._result(
                    task,
                    "方案规划代理未能生成响应方案，因为当前没有可规划的暴露目标。",
                    {"recommended_actions": [], "trigger_type": payload.get("trigger_type", "")},
                    confidence=0.42,
                    decision_confidence=0.35,
                    missing_slots=["当前没有可用于方案综合的暴露目标。"],
                    stop_reason="由于没有可规划的暴露目标，方案生成已停止。",
                )
            focus = exposure.affected_entities[0]
            try:
                advisory = self.platform.generate_advisory_for_impact(focus, request=None)
            except LLMGenerationError as exc:
                return self._result(
                    task,
                    "模型服务暂时波动，系统已自动降级重试",
                    {
                        "focus_entity_ids": [focus.entity.entity_id],
                        "focus_entity_names": [focus.entity.name],
                        "recommended_actions": [],
                        "pending_proposal_ids": [],
                        "trigger_type": payload.get("trigger_type", ""),
                        "llm_status": "failed",
                        "llm_error": str(exc),
                    },
                    confidence=0.22,
                    decision_confidence=0.18,
                    missing_slots=["对象级建议模型当前不可用。"],
                    stop_reason=str(exc),
                )
            proposal_ids = [advisory.proposal.proposal_id] if advisory.proposal else []
            output = {
                "focus_entity_ids": [focus.entity.entity_id],
                "focus_entity_names": [focus.entity.name],
                "recommended_actions": advisory.recommended_actions,
                "pending_proposal_ids": proposal_ids,
                "trigger_type": payload.get("trigger_type", ""),
            }
            return self._result(
                task,
                advisory.answer,
                output,
                confidence=advisory.confidence,
                decision_confidence=max(advisory.confidence - 0.05, 0.0),
                evidence_refs=[item.source_id for item in advisory.evidence],
                recommended_next_tasks=["assess_policy"],
            )

        if task.agent_name == AgentName.POLICY:
            if payload.get("trigger_type") == TriggerEventType.SIMULATION_UPDATED.value:
                pending = self.platform.list_regional_proposals(
                    event_id,
                    statuses=[ProposalStatus.PENDING.value],
                )
                if pending:
                    decision = AutonomyDecision(
                        autonomy_level=AutonomyLevel.HUMAN_GATE_REQUIRED,
                        reason="区域级动作请示正在等待指挥长确认。",
                    )
                else:
                    hazard = self.platform.get_hazard_state(event_id)
                    decision = AutonomyDecision(
                        autonomy_level=AutonomyLevel.AUTO_RECOMMEND
                        if hazard.overall_risk_level in {RiskLevel.ORANGE, RiskLevel.RED}
                        else AutonomyLevel.AUTO_OBSERVE,
                        reason=(
                            "区域风险尚未达到主动动作触发阈值。"
                            if hazard.overall_risk_level not in {RiskLevel.ORANGE, RiskLevel.RED}
                            else "区域规划已完成，但当前没有待处理的可执行请示。"
                        ),
                    )
                return self._result(
                    task,
                    decision.reason,
                    {
                        "autonomy_level": decision.autonomy_level,
                        "recommended_actions": [item.proposal.summary for item in pending[:3]],
                        "pending_proposal_ids": [item.proposal.proposal_id for item in pending],
                        "trigger_type": payload.get("trigger_type", ""),
                    },
                    confidence=0.86,
                    decision_confidence=0.83,
                    stop_reason=decision.reason if decision.autonomy_level == AutonomyLevel.HUMAN_GATE_REQUIRED else None,
                )
            exposure = self.platform.get_exposure_summary(event_id, top_k=1)
            if not exposure.affected_entities:
                decision = AutonomyDecision(
                    autonomy_level=AutonomyLevel.AUTO_OBSERVE,
                    reason="当前没有处于高风险状态的目标对象。",
                )
                return self._result(
                    task,
                    decision.reason,
                    {"autonomy_level": decision.autonomy_level, "trigger_type": payload.get("trigger_type", "")},
                    confidence=0.7,
                    decision_confidence=0.7,
                    stop_reason=decision.reason,
                )
            focus = exposure.affected_entities[0]
            constraint = self.platform.get_policy_constraints(focus.entity.entity_type.value, focus.risk_level.value)
            if constraint.requires_confirmation:
                decision = AutonomyDecision(
                    autonomy_level=AutonomyLevel.HUMAN_GATE_REQUIRED,
                    reason=f"{focus.entity.name} 当前属于 {focus.risk_level.value} 风险，需要人工确认。",
                )
            else:
                decision = AutonomyDecision(
                    autonomy_level=AutonomyLevel.AUTO_RECOMMEND,
                    reason=f"{focus.entity.name} 在当前策略约束下可继续保持自动建议模式。",
                )
            output = {
                "autonomy_level": decision.autonomy_level,
                "recommended_actions": constraint.actions_requiring_approval[:3],
                "focus_entity_ids": [focus.entity.entity_id],
                "focus_entity_names": [focus.entity.name],
                "trigger_type": payload.get("trigger_type", ""),
            }
            return self._result(
                task,
                decision.reason,
                output,
                confidence=0.88,
                decision_confidence=0.86,
                recommended_next_tasks=["draft_comms"] if decision.autonomy_level != AutonomyLevel.HUMAN_GATE_REQUIRED else [],
                stop_reason=decision.reason if decision.autonomy_level == AutonomyLevel.HUMAN_GATE_REQUIRED else None,
            )

        if task.agent_name == AgentName.COMMS:
            if payload.get("trigger_type") == TriggerEventType.SIMULATION_UPDATED.value:
                return self._result(
                    task,
                    "通信草案已纳入区域级请示审批流程，因此没有额外生成对象级通信草案。",
                    {"trigger_type": payload.get("trigger_type", "")},
                    confidence=0.72,
                    decision_confidence=0.7,
                    stop_reason="Regional proposal flow does not require a separate comms terminal step.",
                )
            exposure = self.platform.get_exposure_summary(event_id, top_k=1)
            if not exposure.affected_entities:
                return self._result(
                    task,
                    "通讯联动代理当前没有可生成通知草案的目标。",
                    {},
                    confidence=0.38,
                    decision_confidence=0.3,
                    missing_slots=["当前没有可用于通信草案生成的目标。"],
                    stop_reason="由于没有可通信目标，沟通草案生成已停止。",
                )
            focus = exposure.affected_entities[0]
            proposal = self.platform.draft_action_proposal(event_id, focus.entity.entity_id)
            if proposal is None:
                return self._result(
                    task,
                    f"通讯联动代理未为 {focus.entity.name} 生成正式草案，因为当前情势下不需要正式请示。",
                    {
                        "focus_entity_ids": [focus.entity.entity_id],
                        "focus_entity_names": [focus.entity.name],
                        "trigger_type": payload.get("trigger_type", ""),
                    },
                    confidence=0.64,
                    decision_confidence=0.58,
                    stop_reason=f"{focus.entity.name} 在当前情势下不需要正式通信草案。",
                )
            templates = proposal.payload.get("notification_templates", [])
            preview = []
            if isinstance(templates, list):
                for item in templates[:2]:
                    if isinstance(item, dict) and isinstance(item.get("content"), str):
                        preview.append(item["content"])
            output = {
                "focus_entity_ids": [focus.entity.entity_id],
                "focus_entity_names": [focus.entity.name],
                "pending_proposal_ids": [proposal.proposal_id],
                "recommended_actions": preview,
                "trigger_type": payload.get("trigger_type", ""),
            }
            return self._result(
                task,
                f"通讯联动代理已为 {focus.entity.name} 准备通知草稿。",
                output,
                confidence=0.77,
                decision_confidence=0.72,
                stop_reason="任务图已到达通信终点步骤。",
            )

        return self._result(
            task,
            f"当前尚未为 {task.agent_name.value} 实现任务 {task.task_type}。",
            {},
            confidence=0.2,
            decision_confidence=0.1,
            missing_slots=["当前任务类型暂不支持。"],
            stop_reason="当前请求的智能体任务类型暂不支持。",
        )

    @staticmethod
    def _result(
        task: AgentTask,
        summary: str,
        structured_output: dict,
        *,
        confidence: float,
        decision_confidence: float = 0.0,
        evidence_refs: list[str] | None = None,
        missing_slots: list[str] | None = None,
        handoff_recommendations: list[str] | None = None,
        recommended_next_tasks: list[str] | None = None,
        stop_reason: str | None = None,
        supersedes_task_ids: list[str] | None = None,
    ) -> AgentResult:
        return AgentResult(
            result_id=f"result_{uuid4().hex[:10]}",
            task_id=task.task_id,
            event_id=task.event_id,
            agent_name=task.agent_name,
            summary=summary,
            structured_output=structured_output,
            confidence=round(confidence, 2),
            decision_confidence=round(decision_confidence or confidence, 2),
            evidence_refs=evidence_refs or [],
            missing_slots=missing_slots or [],
            handoff_recommendations=handoff_recommendations or [],
            recommended_next_tasks=recommended_next_tasks or [],
            stop_reason=stop_reason,
            supersedes_task_ids=supersedes_task_ids or [],
            created_at=datetime.now(timezone.utc),
        )


class AgentSupervisor:
    AGENT_CHAIN: list[tuple[AgentName, str]] = [
        (AgentName.HAZARD, "assess_hazard"),
        (AgentName.EXPOSURE, "assess_exposure"),
        (AgentName.RESOURCE, "assess_resources"),
        (AgentName.PLANNING, "draft_plan"),
        (AgentName.POLICY, "assess_policy"),
        (AgentName.COMMS, "draft_comms"),
    ]

    def __init__(self, platform, repository) -> None:
        self.platform = platform
        self.repository = repository
        self.task_queue: TaskQueue = SQLiteTaskQueue(repository)
        self.trigger_bus: TriggerEventBus = SQLiteTriggerEventBus(repository)
        self.shared_memory = EventSharedMemoryStore(repository)
        self.runtime = AgentRuntime(platform, repository)
        self.decision_graph = AgentDecisionGraph()
        self._run_lock = threading.Lock()
        self._active_event_runs: set[str] = set()

    def publish_trigger(
        self,
        event_id: str,
        *,
        trigger_type: TriggerEventType,
        payload: dict | None = None,
        dedupe: bool = True,
    ) -> TriggerEvent:
        return self.trigger_bus.publish(
            event_id=event_id,
            trigger_type=trigger_type,
            payload=payload,
            dedupe=dedupe,
        )

    def replay_task(self, task_id: str, *, replay_reason: str = "") -> AgentTask:
        return self.task_queue.replay(task_id, replay_reason=replay_reason)

    def process_trigger(self, trigger: TriggerEvent) -> SupervisorRunRecord:
        run = self.run_for_event(
            trigger.event_id,
            trigger_type=trigger.trigger_type.value,
            session_id=trigger.payload.get("session_id") if isinstance(trigger.payload, dict) else None,
            entity_type=trigger.payload.get("entity_type") if isinstance(trigger.payload, dict) else None,
            trigger_payload=trigger.payload if isinstance(trigger.payload, dict) else None,
            source_trigger_id=trigger.trigger_id,
        )
        self.trigger_bus.ack(trigger.trigger_id)
        return run

    def run_for_event(
        self,
        event_id: str,
        *,
        trigger_type: str,
        session_id: str | None = None,
        entity_type: str | None = None,
        trigger_payload: dict | None = None,
        source_trigger_id: str | None = None,
    ) -> SupervisorRunRecord:
        with self._run_lock:
            if event_id in self._active_event_runs:
                recent = self.repository.list_v2_supervisor_runs(event_id, limit=1)
                if recent:
                    return recent[0]
            self._active_event_runs.add(event_id)

        run = SupervisorRunRecord(
            supervisor_run_id=f"sup_{uuid4().hex[:10]}",
            event_id=event_id,
            trigger_type=trigger_type,
            status=SupervisorRunStatus.RUNNING,
            session_id=session_id,
            summary="后台巡检已启动一次多智能体协作流程。",
            created_at=datetime.now(timezone.utc),
        )
        self.repository.save_v2_supervisor_run(run)
        snapshot = self.shared_memory.load(event_id)
        created_tasks: list[str] = []
        completed_tasks: list[str] = []
        failed_tasks: list[str] = []
        autonomy_level = snapshot.autonomy_level
        superseded_task_ids = self._supersede_open_tasks(event_id)
        pending_steps = list(self.decision_graph.initial_steps(trigger_type))
        seen_steps: set[tuple[AgentName, str]] = set()
        first_result = True

        try:
            while pending_steps:
                agent_name, task_type = pending_steps.pop(0)
                step_key = (agent_name, task_type)
                if step_key in seen_steps:
                    continue
                seen_steps.add(step_key)
                task = self.task_queue.enqueue(
                    event_id=event_id,
                    agent_name=agent_name,
                    task_type=task_type,
                    input_payload={
                        "trigger_type": trigger_type,
                        "entity_type": entity_type,
                        **(trigger_payload or {}),
                    },
                    session_id=session_id,
                    source_trigger_id=source_trigger_id,
                )
                created_tasks.append(task.task_id)
                try:
                    running = self.task_queue.claim(task.task_id)
                    result = self.runtime.run(running)
                    if first_result and superseded_task_ids and not result.supersedes_task_ids:
                        result = result.model_copy(update={"supersedes_task_ids": superseded_task_ids})
                    first_result = False
                    self.repository.save_v2_agent_result(result)
                    self.repository.add_v2_agent_task_event(
                        AgentTaskEvent(
                            task_event_id=f"tke_{uuid4().hex[:10]}",
                            event_id=event_id,
                            task_id=running.task_id,
                            agent_name=running.agent_name,
                            event_type=AgentTaskEventType.AGENT_RESULT_SAVED,
                            trigger_id=source_trigger_id,
                            payload={
                                "summary": result.summary,
                                "result_id": result.result_id,
                                "recommended_next_tasks": result.recommended_next_tasks,
                                "stop_reason": result.stop_reason,
                                "supersedes_task_ids": result.supersedes_task_ids,
                            },
                            created_at=datetime.now(timezone.utc),
                        )
                    )
                    autonomy_hint = result.structured_output.get("autonomy_level")
                    if autonomy_hint:
                        autonomy_level = autonomy_hint
                    self.task_queue.complete(running, output_payload=result.structured_output)
                    shared = self.shared_memory.apply(event_id, result=result, autonomy_level=autonomy_level)
                    completed_tasks.append(task.task_id)
                    next_steps, graph_stop = self.decision_graph.next_steps(
                        trigger_type=trigger_type,
                        latest_result=result,
                        shared_memory=shared,
                    )
                    if graph_stop:
                        shared = self.shared_memory.save(
                            shared.model_copy(
                                update={
                                    "blocked_by": list(dict.fromkeys([*shared.blocked_by, graph_stop]))[:6],
                                    "latest_summary": result.summary,
                                }
                            )
                        )
                    for next_agent_name, next_task_type in next_steps:
                        next_key = (next_agent_name, next_task_type)
                        if next_key not in seen_steps:
                            pending_steps.append((next_agent_name, next_task_type))
                except Exception as exc:
                    self.task_queue.fail(task, failure_reason=str(exc))
                    failed_tasks.append(task.task_id)
                    self.platform.add_audit_record(
                        source_type="agent_task",
                        action="agent_task_failed",
                        summary=f"{agent_name.value} 在执行 {task_type} 时失败。",
                        details={"task_id": task.task_id, "failure_reason": str(exc)},
                        severity=AlertSeverity.WARNING,
                        event_id=event_id,
                        session_id=session_id,
                    )

            latest_memory = self.shared_memory.load(event_id)
            completed = run.model_copy(
                update={
                    "status": SupervisorRunStatus.FAILED if failed_tasks else SupervisorRunStatus.COMPLETED,
                    "autonomy_level": latest_memory.autonomy_level,
                    "summary": (
                        f"{latest_memory.latest_summary} {len(failed_tasks)} task(s) failed."
                        if failed_tasks
                        else (latest_memory.latest_summary or "后台巡检已完成本轮多智能体扫描。")
                    ),
                    "created_tasks": created_tasks,
                    "completed_task_ids": completed_tasks,
                    "completed_at": datetime.now(timezone.utc),
                }
            )
            self.repository.save_v2_supervisor_run(completed)
            return completed
        finally:
            with self._run_lock:
                self._active_event_runs.discard(event_id)

    def tick(self, event_id: str | None = None) -> list[SupervisorRunRecord]:
        if event_id is not None:
            return [self.run_for_event(event_id, trigger_type=TriggerEventType.MANUAL_TICK.value)]
        runs: list[SupervisorRunRecord] = []
        event_ids: list[str] = []
        for area_id in self.platform.area_profiles:
            latest = self.repository.get_latest_v2_event_id(area_id)
            if latest:
                event_ids.append(latest)
        for item in dict.fromkeys(event_ids):
            runs.append(self.run_for_event(item, trigger_type="background_sweep"))
        return runs

    def get_agent_status(self, event_id: str) -> dict:
        tasks = self.repository.list_v2_agent_tasks(event_id, limit=50)
        shared = self.shared_memory.load(event_id)
        return {
            "event_id": event_id,
            "active_agents": [agent.value for agent in shared.active_agents],
            "autonomy_level": shared.autonomy_level.value,
            "latest_hazard_level": shared.latest_hazard_level.value if shared.latest_hazard_level else None,
            "pending_task_count": len([item for item in tasks if item.status == AgentTaskStatus.PENDING]),
            "running_task_count": len([item for item in tasks if item.status == AgentTaskStatus.RUNNING]),
            "completed_task_count": len([item for item in tasks if item.status == AgentTaskStatus.COMPLETED]),
            "superseded_task_count": len([item for item in tasks if item.status == AgentTaskStatus.SUPERSEDED]),
            "latest_summary": shared.latest_summary,
            "active_decision_path": shared.active_decision_path,
            "open_questions": shared.open_questions,
            "blocked_by": shared.blocked_by,
            "updated_at": shared.updated_at,
        }

    def _supersede_open_tasks(self, event_id: str) -> list[str]:
        superseded_ids: list[str] = []
        tasks = self.repository.list_v2_agent_tasks(event_id, limit=80)
        for task in tasks:
            if task.status not in {AgentTaskStatus.PENDING, AgentTaskStatus.RUNNING}:
                continue
            superseded = task.model_copy(
                update={
                    "status": AgentTaskStatus.SUPERSEDED,
                    "failure_reason": "已被更新的触发驱动任务图替代。",
                    "completed_at": datetime.now(timezone.utc),
                }
            )
            self.repository.save_v2_agent_task(superseded)
            self.repository.add_v2_agent_task_event(
                AgentTaskEvent(
                    task_event_id=f"tke_{uuid4().hex[:10]}",
                    event_id=event_id,
                    task_id=task.task_id,
                    agent_name=task.agent_name,
                    event_type=AgentTaskEventType.TASK_FAILED,
                    trigger_id=task.source_trigger_id,
                    payload={
                        "summary": f"{task.agent_name.value} 的任务 {task.task_type} 已被更新的任务图替代。",
                        "superseded": True,
                    },
                    created_at=datetime.now(timezone.utc),
                )
            )
            superseded_ids.append(task.task_id)
        return superseded_ids


@dataclass
class SupervisorLoopService:
    supervisor: AgentSupervisor
    interval_seconds: float = 60.0
    trigger_type: str = "background_sweep"
    max_retries: int = 2
    retry_backoffs_seconds: tuple[float, ...] = (2.0, 10.0)
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 300.0
    trigger_poll_seconds: float = 1.0
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _health_state: SupervisorHealthState | None = field(default=None, init=False)
    _next_sweep_due: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        existing = self.supervisor.repository.get_supervisor_health_state()
        if existing is not None:
            self._health_state = existing.model_copy(update={"interval_seconds": self.interval_seconds})
        else:
            self._health_state = SupervisorHealthState(
                interval_seconds=self.interval_seconds,
                updated_at=datetime.now(timezone.utc),
            )
            self.supervisor.repository.save_supervisor_health_state(self._health_state)
        self._next_sweep_due = datetime.now(timezone.utc) + timedelta(seconds=self.interval_seconds)

    def start(self) -> None:
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._next_sweep_due = datetime.now(timezone.utc) + timedelta(seconds=self.interval_seconds)
            self._update_health_state(running=True)
            self._thread = threading.Thread(target=self._run_loop, name="flood-supervisor-scheduler", daemon=True)
            self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        with self._state_lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None:
            thread.join(timeout_seconds)
        with self._state_lock:
            self._thread = None
            self._update_health_state(running=False)

    def is_running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def tick_once(self, event_id: str | None = None) -> list[SupervisorRunRecord]:
        try:
            runs = self.supervisor.tick(event_id)
            now = datetime.now(timezone.utc)
            self._close_circuit_if_needed()
            self._update_health_state(
                last_started_at=now,
                last_success_at=now,
                last_completed_at=now,
                last_error=None,
                consecutive_failures=0,
                retries_used_in_last_cycle=0,
                pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events(),
            )
            self.supervisor.platform.add_audit_record(
                source_type="supervisor_loop",
                action="manual_tick_completed",
                summary=f"人工单步巡检完成，本轮产生 {len(runs)} 次运行记录。",
                details={"run_count": len(runs)},
            )
            return runs
        except Exception as exc:
            now = datetime.now(timezone.utc)
            self._update_health_state(
                last_started_at=now,
                last_failure_at=now,
                last_completed_at=now,
                last_error=str(exc),
            )
            self.supervisor.platform.add_audit_record(
                source_type="supervisor_loop",
                action="manual_tick_failed",
                summary="Manual supervisor tick failed.",
                details={"error": str(exc)},
                severity=AlertSeverity.WARNING,
            )
            raise

    def run_event_once(self, event_id: str) -> SupervisorRunRecord:
        try:
            run = self.supervisor.run_for_event(event_id, trigger_type=TriggerEventType.MANUAL_RUN.value)
            now = datetime.now(timezone.utc)
            self._close_circuit_if_needed()
            self._update_health_state(
                last_started_at=now,
                last_success_at=now,
                last_completed_at=now,
                last_error=None,
                consecutive_failures=0,
                retries_used_in_last_cycle=0,
                pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events(),
            )
            self.supervisor.platform.add_audit_record(
                source_type="supervisor_loop",
                action="manual_run_completed",
                summary=f"已为事件 {event_id} 完成一次人工触发的后台巡检。",
                details={"supervisor_run_id": run.supervisor_run_id, "event_id": event_id},
            )
            return run
        except Exception as exc:
            now = datetime.now(timezone.utc)
            self._update_health_state(
                last_started_at=now,
                last_failure_at=now,
                last_completed_at=now,
                last_error=str(exc),
            )
            self.supervisor.platform.add_audit_record(
                source_type="supervisor_loop",
                action="manual_run_failed",
                summary=f"Manual supervisor run failed for {event_id}.",
                details={"error": str(exc), "event_id": event_id},
                severity=AlertSeverity.WARNING,
            )
            raise

    def status(self) -> dict:
        if self._health_state is None:
            self.__post_init__()
        state = self._health_state.model_copy(
            update={
                "running": self.is_running(),
                "interval_seconds": self.interval_seconds,
                "pending_trigger_count": self.supervisor.repository.count_pending_v2_trigger_events(),
                "recent_replay_count": self.supervisor.repository.count_v2_agent_task_events(
                    event_type=AgentTaskEventType.REPLAY_COMPLETED.value,
                ),
                "recent_timeline_failure_count": self.supervisor.repository.count_v2_agent_task_events(
                    event_type=AgentTaskEventType.TASK_FAILED.value,
                ),
            }
        )
        return state.model_dump(mode="json")

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.trigger_poll_seconds):
            if self._circuit_open():
                self._update_health_state(
                    skipped_sweeps=(self._health_state.skipped_sweeps + 1),
                    pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events(),
                )
                self.supervisor.platform.add_audit_record(
                    source_type="supervisor_loop",
                    action="sweep_skipped",
                    summary="Background supervisor sweep skipped because the circuit is open.",
                    details={"circuit_expires_at": self._health_state.circuit_expires_at.isoformat() if self._health_state.circuit_expires_at else None},
                    severity=AlertSeverity.WARNING,
                )
                continue
            self._drain_trigger_events()
            if self._next_sweep_due is not None and datetime.now(timezone.utc) >= self._next_sweep_due:
                self._run_background_cycle()
                self._next_sweep_due = datetime.now(timezone.utc) + timedelta(seconds=self.interval_seconds)

    def _drain_trigger_events(self) -> None:
        processed = False
        while not self._stop_event.is_set():
            trigger = self.supervisor.trigger_bus.lease()
            if trigger is None:
                break
            processed = True
            try:
                self.supervisor.process_trigger(trigger)
                self._update_health_state(
                    last_trigger_processed_at=datetime.now(timezone.utc),
                    pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events(),
                )
            except Exception as exc:
                self.supervisor.trigger_bus.fail(trigger.trigger_id, error_message=str(exc))
                self._update_health_state(
                    last_error=str(exc),
                    recent_timeline_failure_count=self._health_state.recent_timeline_failure_count + 1,
                    pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events(),
                )
                self.supervisor.platform.add_audit_record(
                    source_type="trigger_event",
                    action="trigger_processing_failed",
                    summary=f"处理触发事件 {trigger.trigger_type.value} 失败。",
                    details={"trigger_id": trigger.trigger_id, "error": str(exc)},
                    severity=AlertSeverity.WARNING,
                    event_id=trigger.event_id,
                )
                break
        if not processed:
            self._update_health_state(pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events())

    def _run_background_cycle(self) -> None:
        now = datetime.now(timezone.utc)
        self._update_health_state(last_started_at=now, retries_used_in_last_cycle=0)
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                runs = self.supervisor.tick()
                success_time = datetime.now(timezone.utc)
                severity = AlertSeverity.WARNING if attempt > 0 else AlertSeverity.INFO
                action = "background_sweep_recovered" if attempt > 0 else "background_sweep_completed"
                summary = (
                    f"Background supervisor sweep recovered after {attempt} retry attempt(s)."
                    if attempt > 0
                    else f"后台巡检已完成，本轮产生 {len(runs)} 次运行记录。"
                )
                self._close_circuit_if_needed(recovered=(attempt > 0))
                self._update_health_state(
                    last_success_at=success_time,
                    last_completed_at=success_time,
                    last_error=None,
                    consecutive_failures=0,
                    retries_used_in_last_cycle=attempt,
                    pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events(),
                )
                self.supervisor.platform.add_audit_record(
                    source_type="supervisor_loop",
                    action=action,
                    summary=summary,
                    details={"run_count": len(runs), "retry_count": attempt},
                    severity=severity,
                )
                if attempt > 0:
                    self.supervisor.platform.save_operational_alert(
                        source_type="supervisor_loop",
                        severity=AlertSeverity.WARNING,
                        summary="Background supervisor sweep recovered after retry.",
                        details=f"The latest sweep succeeded after {attempt} retry attempt(s).",
                    )
                return
            except Exception as exc:
                last_error = str(exc)
                failure_time = datetime.now(timezone.utc)
                self._update_health_state(
                    last_failure_at=failure_time,
                    last_completed_at=failure_time,
                    last_error=last_error,
                    retries_used_in_last_cycle=attempt,
                    pending_trigger_count=self.supervisor.repository.count_pending_v2_trigger_events(),
                )
                self.supervisor.platform.add_audit_record(
                    source_type="supervisor_loop",
                    action="background_sweep_failed_attempt",
                    summary=f"Background supervisor sweep failed on attempt {attempt + 1}.",
                    details={"error": last_error, "attempt": attempt + 1},
                    severity=AlertSeverity.WARNING,
                )
                if attempt < self.max_retries:
                    self._update_health_state(last_retry_at=datetime.now(timezone.utc))
                    backoff = self.retry_backoffs_seconds[min(attempt, len(self.retry_backoffs_seconds) - 1)]
                    if self._stop_event.wait(backoff):
                        return
                    continue

        next_failures = self._health_state.consecutive_failures + 1
        self._update_health_state(consecutive_failures=next_failures, retries_used_in_last_cycle=self.max_retries)
        self.supervisor.platform.save_operational_alert(
            source_type="supervisor_loop",
            severity=AlertSeverity.WARNING,
            summary="Background supervisor sweep exhausted retries.",
            details=last_error or "Unknown background sweep failure.",
        )
        if next_failures >= self.circuit_failure_threshold:
            opened_at = datetime.now(timezone.utc)
            expires_at = opened_at + timedelta(seconds=self.circuit_cooldown_seconds)
            self._update_health_state(
                circuit_state=CircuitState.OPEN,
                circuit_opened_at=opened_at,
                circuit_expires_at=expires_at,
            )
            self.supervisor.platform.save_operational_alert(
                source_type="supervisor_loop",
                severity=AlertSeverity.CRITICAL,
                summary="后台巡检因连续失败已进入熔断状态。",
                details=last_error or "Unknown repeated background sweep failure.",
            )
            self.supervisor.platform.add_audit_record(
                source_type="supervisor_loop",
                action="circuit_opened",
                summary="后台巡检因连续失败已进入熔断状态。",
                details={"error": last_error, "consecutive_failures": next_failures},
                severity=AlertSeverity.CRITICAL,
            )

    def _circuit_open(self) -> bool:
        state = self._health_state
        if state is None or state.circuit_state != CircuitState.OPEN:
            return False
        if state.circuit_expires_at and datetime.now(timezone.utc) >= state.circuit_expires_at:
            self._update_health_state(circuit_state=CircuitState.HALF_OPEN)
            return False
        return True

    def _close_circuit_if_needed(self, *, recovered: bool = False) -> None:
        state = self._health_state
        if state is None or state.circuit_state == CircuitState.CLOSED:
            return
        self._update_health_state(
            circuit_state=CircuitState.CLOSED,
            circuit_opened_at=None,
            circuit_expires_at=None,
            consecutive_failures=0,
        )
        self.supervisor.platform.save_operational_alert(
            source_type="supervisor_loop",
            severity=AlertSeverity.INFO,
            summary="后台巡检熔断已恢复。",
            details="A manual or background run succeeded and closed the circuit.",
        )
        self.supervisor.platform.add_audit_record(
            source_type="supervisor_loop",
            action="circuit_recovered",
            summary="后台巡检熔断已恢复。",
            details={"recovered_after_retry": recovered},
            severity=AlertSeverity.INFO,
        )

    def _update_health_state(self, **updates) -> None:
        current = self._health_state or SupervisorHealthState(
            interval_seconds=self.interval_seconds,
            updated_at=datetime.now(timezone.utc),
        )
        payload = {"interval_seconds": self.interval_seconds, "updated_at": datetime.now(timezone.utc), **updates}
        self._health_state = current.model_copy(update=payload)
        self.supervisor.repository.save_supervisor_health_state(self._health_state)


@dataclass
class HousekeepingService:
    repository: object
    platform: object
    interval_seconds: float = 21600.0
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="flood-housekeeping-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
        self._thread = None

    def run_once(self) -> ArchiveStatusView:
        started_at = datetime.now(timezone.utc)
        try:
            status = self.repository.archive_operational_records(now=started_at)
            self.platform.add_audit_record(
                source_type="housekeeping",
                action="archive_run_completed",
                summary="归档清理已成功完成。",
                details={
                    "hot_records_archived": status.last_archive_run.hot_records_archived if status.last_archive_run else 0,
                    "expired_archives_deleted": status.last_archive_run.expired_archives_deleted if status.last_archive_run else 0,
                },
            )
            return status
        except Exception as exc:
            self.platform.add_audit_record(
                source_type="housekeeping",
                action="archive_run_failed",
                summary="归档清理失败。",
                details={"error": str(exc)},
                severity=AlertSeverity.WARNING,
            )
            self.platform.save_operational_alert(
                source_type="housekeeping",
                severity=AlertSeverity.WARNING,
                summary="归档清理失败。",
                details=str(exc),
            )
            raise

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.run_once()
            except Exception:
                continue
