import { useState } from "react";
import styles from "../App.module.css";
import { autonomyText, riskText, taskStatusText } from "../config/consoleConfig";
import { agentText, normalizeAgentTerminology } from "../lib/agentUiText";
import {
  formatAgentHandoffTarget,
  formatAgentTaskEventType,
  formatAgentTaskType,
  formatRegionalActionType,
  formatStopReason,
  formatTriggerStatus,
  formatTriggerType,
} from "../lib/displayText";
import { formatPercent, formatTimestamp } from "../lib/consoleFormatting";
import { buildAgentDivergenceRows } from "../state/agentTwinSelectors";
import { AccessPolicyNotice, actionRequiredRoleText } from "./SecurityDesk";
import type {
  AgentMetricsView,
  AgentResult,
  AgentStatusView,
  AgentTask,
  AgentTaskStatus,
  AgentTimelineEntry,
  EventEpisodeSummaryView,
  DecisionReportView,
  EvaluationBenchmark,
  EvaluationReport,
  ExperienceContextView,
  SessionMemoryView,
  SharedMemorySnapshot,
  SupervisorLoopStatus,
  SupervisorRunRecord,
  TriggerEvent,
} from "../types/api";

function taskStatusClass(status: AgentTaskStatus) {
  return {
    pending: styles.statusPending,
    running: styles.executionTimeout,
    completed: styles.statusApproved,
    failed: styles.statusRejected,
    canceled: styles.executionSkipped,
    superseded: styles.executionSkipped,
  }[status];
}

function metric(label: string, value: string, hint?: string) {
  return (
    <div className={styles.metricBlock}>
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </div>
  );
}

export interface MultiAgentDeskProps {
  eventId?: string;
  agentStatus: AgentStatusView | null;
  agentTasks: AgentTask[];
  sessionMemoryView: SessionMemoryView | null;
  sharedMemorySnapshot: SharedMemorySnapshot | null;
  episodeSummaries: EventEpisodeSummaryView[];
  triggerEvents: TriggerEvent[];
  agentTimeline: AgentTimelineEntry[];
  supervisorRuns: SupervisorRunRecord[];
  supervisorLoopStatus: SupervisorLoopStatus | null;
  recentAgentResults: AgentResult[];
  experienceContext: ExperienceContextView | null;
  decisionReport: DecisionReportView | null;
  agentMetrics: AgentMetricsView | null;
  evaluationBenchmarks: EvaluationBenchmark[];
  latestEvaluationReport: EvaluationReport | null;
  busy: boolean;
  canControlSupervisor: boolean;
  canReplayTask: boolean;
  canRunEvaluation: boolean;
  onRunSupervisor: () => Promise<void>;
  onTickSupervisor: () => Promise<void>;
  onReplayTask: (taskId: string, replayReason: string) => Promise<void>;
  onRunEvaluation: () => Promise<void>;
  onReplayEvaluationReport: (reportId: string) => Promise<void>;
}

export function MultiAgentDesk({
  eventId,
  agentStatus,
  agentTasks,
  sessionMemoryView,
  sharedMemorySnapshot,
  episodeSummaries,
  triggerEvents,
  agentTimeline,
  supervisorRuns,
  supervisorLoopStatus,
  recentAgentResults,
  experienceContext,
  decisionReport,
  agentMetrics,
  evaluationBenchmarks,
  latestEvaluationReport,
  busy,
  canControlSupervisor,
  canReplayTask,
  canRunEvaluation,
  onRunSupervisor,
  onTickSupervisor,
  onReplayTask,
  onRunEvaluation,
  onReplayEvaluationReport,
}: MultiAgentDeskProps) {
  const activeAgents = sharedMemorySnapshot?.active_agents ?? agentStatus?.active_agents ?? [];
  const [replayReason, setReplayReason] = useState("");
  const latestEpisodeSummary = episodeSummaries[0] ?? null;
  const latestLongTermMemory = experienceContext?.long_term_memories[0] ?? null;
  const divergenceRows = buildAgentDivergenceRows({
    recentResults: recentAgentResults,
    sharedMemorySnapshot,
    decisionReport,
    maxRows: 5,
  });

  return (
    <div className={styles.agentDesk}>
      <div className={styles.panelHeader}>
        <div>
          <p className={styles.sectionLabel}>多代理协作</p>
          <h2>多代理调度与共享记忆</h2>
        </div>
        <div className={styles.bulkToolbar}>
          <button type="button" className={styles.secondaryButton} aria-label="run-agent-evaluation" disabled={busy || !canRunEvaluation} onClick={() => void onRunEvaluation()}>
            运行评测
          </button>
          <button type="button" className={styles.secondaryButton} aria-label="tick-supervisor" disabled={!eventId || busy || !canControlSupervisor} onClick={() => void onTickSupervisor()}>
            单步触发
          </button>
          <button type="button" className={styles.primaryButton} aria-label="run-supervisor-now" disabled={!eventId || busy || !canControlSupervisor} onClick={() => void onRunSupervisor()}>
            立即运行
          </button>
        </div>
      </div>
      <AccessPolicyNotice
        title="页面策略"
        summary="多代理运行状态对所有角色可见，但评测、手动控制和任务重放属于控制面动作，会单独受权限限制。"
        items={[
          { label: "运行或重放评测", allowed: canRunEvaluation, requiredRole: actionRequiredRoleText.evaluation_run, description: "评测会创建合成事件与会话流量，因此仅对区级处置员和指挥员开放。" },
          { label: "手动触发后台巡检", allowed: canControlSupervisor, requiredRole: actionRequiredRoleText.supervisor_control, description: "手动控制可以绕过后台节奏，因此保持在指挥权限下。" },
          { label: "重放智能体任务", allowed: canReplayTask, requiredRole: actionRequiredRoleText.agent_replay, description: "单任务重放会重新打开决策路径，因此视为人工覆盖动作。" },
        ]}
      />

      <div className={styles.agentStatusStrip}>
        {metric("自治状态", agentStatus ? autonomyText[agentStatus.autonomy_level] : "--", agentStatus?.latest_summary)}
        {metric("活跃代理", `${activeAgents.length}`, activeAgents.length ? activeAgents.map((agent) => agentText[agent]).join(", ") : "等待代理活动")}
        {metric("任务队列", `${agentTasks.length}`, agentStatus ? `${agentStatus.pending_task_count} 待处理 / ${agentStatus.running_task_count} 运行中` : "暂无任务")}
        {metric("调度运行", `${supervisorRuns.length}`, supervisorRuns[0]?.trigger_type ? formatTriggerType(supervisorRuns[0].trigger_type) : "暂无运行记录")}
        {metric("决策路径", `${decisionReport?.active_decision_path.length ?? 0}`, decisionReport?.blocked_by?.[0] ?? "当前没有阻塞原因")}
        {metric("后台巡检", supervisorLoopStatus ? (supervisorLoopStatus.running ? "运行中" : "已停止") : "--", supervisorLoopStatus ? `每 ${supervisorLoopStatus.interval_seconds} 秒一次` : "暂无巡检状态")}
      </div>

      <div className={styles.adminCard}>
        <div className={styles.adminCardHeader}>
          <div><p className={styles.sectionLabel}>会商差异</p><h3>Agent 分歧点与 supervisor 采纳理由</h3></div>
        </div>
        <div className={styles.executionList}>
          {divergenceRows.length ? (
            divergenceRows.map((row) => (
              <article key={`${row.result.result_id}-divergence`} className={styles.executionCard}>
                <div className={styles.executionTopline}>
                  <strong>{agentText[row.result.agent_name]} / {row.disposition}</strong>
                  <span>{Math.round(row.confidence * 100)}%</span>
                </div>
                <p>{normalizeAgentTerminology(row.result.summary)}</p>
                <div className={styles.executionMeta}>
                  <span>分歧点：{row.disagreement}</span>
                  <span>证据 {row.result.evidence_refs.length} 条</span>
                </div>
                <small>编排理由：{row.rationale}</small>
              </article>
            ))
          ) : <p className={styles.emptyState}>当前还没有足够的 agent result 来比较分歧与采纳理由。</p>}
        </div>
      </div>

      <div className={styles.agentGrid}>
        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>实时状态</p><h3>活跃代理</h3></div>
            {agentStatus ? <span className={styles.scopeBadge}>{autonomyText[agentStatus.autonomy_level]}</span> : null}
          </div>
          <div className={styles.agentChipRow}>
            {activeAgents.length ? activeAgents.map((agent) => <span key={agent} className={styles.agentChip}>{agentText[agent]}</span>) : <p className={styles.emptyState}>尚无代理发布共享状态。</p>}
          </div>
          <div className={styles.memoryList}>
            <div><span className={styles.operationLabel}>最新摘要</span><p>{agentStatus?.latest_summary || sharedMemorySnapshot?.latest_summary || "当前还没有生成共享摘要。"}</p></div>
            <div><span className={styles.operationLabel}>最新风险等级</span><p>{agentStatus?.latest_hazard_level ? riskText[agentStatus.latest_hazard_level] : "未知"}</p></div>
            <div><span className={styles.operationLabel}>更新时间</span><p>{formatTimestamp(agentStatus?.updated_at ?? sharedMemorySnapshot?.updated_at)}</p></div>
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>后台巡检</p><h3>后台巡检状态</h3></div>
            {supervisorLoopStatus ? <span className={`${styles.statusPill} ${supervisorLoopStatus.running ? styles.statusApproved : styles.executionSkipped}`}>{supervisorLoopStatus.running ? "运行中" : "已停止"}</span> : null}
          </div>
          <div className={styles.memoryList}>
            <div><span className={styles.operationLabel}>巡检周期</span><p>{supervisorLoopStatus ? `每 ${supervisorLoopStatus.interval_seconds} 秒一次` : "暂无巡检状态。"}</p></div>
            <div><span className={styles.operationLabel}>最近完成</span><p>{formatTimestamp(supervisorLoopStatus?.last_completed_at)}</p></div>
            <div><span className={styles.operationLabel}>最近启动</span><p>{formatTimestamp(supervisorLoopStatus?.last_started_at)}</p></div>
            <div><span className={styles.operationLabel}>最近错误</span><p>{supervisorLoopStatus?.last_error || "暂无后台巡检错误。"}</p></div>
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>共享记忆</p><h3>会话记忆与事件共享记忆</h3></div>
          </div>
          <div className={styles.memoryList}>
            <div><span className={styles.operationLabel}>会话关注对象</span><p>{sessionMemoryView?.memory_snapshot.focus_entity_name || "当前还没有会话关注对象。"}</p></div>
            <div><span className={styles.operationLabel}>会话目标</span><p>{sessionMemoryView?.memory_snapshot.current_goal || "当前还没有记录会话目标。"}</p></div>
            <div><span className={styles.operationLabel}>会话未决槽位</span><p>{sessionMemoryView?.memory_snapshot.unresolved_slots.length ? sessionMemoryView.memory_snapshot.unresolved_slots.join(" | ") : "当前没有待补足的会话槽位。"}</p></div>
            <div><span className={styles.operationLabel}>事件关注对象</span><p>{sharedMemorySnapshot?.focus_entity_names.length ? sharedMemorySnapshot.focus_entity_names.join(", ") : "当前还没有共享关注对象。"}</p></div>
            <div><span className={styles.operationLabel}>事件高风险摘要</span><p>{sharedMemorySnapshot?.top_risks.length ? sharedMemorySnapshot.top_risks.join(" | ") : "当前还没有高风险摘要。"}</p></div>
            <div><span className={styles.operationLabel}>事件建议动作</span><p>{sharedMemorySnapshot?.recommended_actions.length ? sharedMemorySnapshot.recommended_actions.join(" | ") : "当前还没有共享行动建议。"}</p></div>
            <div><span className={styles.operationLabel}>事件未决事项</span><p>{sharedMemorySnapshot?.unresolved_items.length ? sharedMemorySnapshot.unresolved_items.join(" | ") : "当前没有共享未决事项。"}</p></div>
            <div><span className={styles.operationLabel}>决策路径</span><p>{sharedMemorySnapshot?.active_decision_path.length ? sharedMemorySnapshot.active_decision_path.slice(-3).join(" | ") : "当前还没有活跃决策路径。"}</p></div>
            <div><span className={styles.operationLabel}>阻塞原因</span><p>{sharedMemorySnapshot?.blocked_by.length ? sharedMemorySnapshot.blocked_by.join(" | ") : "当前没有阻塞原因。"}</p></div>
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>经验层</p><h3>历史模式</h3></div>
          </div>
          <div className={styles.memoryList}>
            <div><span className={styles.operationLabel}>相关经验记录</span><p>{experienceContext?.relevant_records.length ? `上下文中已加载 ${experienceContext.relevant_records.length} 条相似历史。` : "当前还没有加载相似历史。"}</p></div>
            <div><span className={styles.operationLabel}>策略模式</span><p>{experienceContext?.strategy_patterns.length ? experienceContext.strategy_patterns.map((pattern) => `${formatRegionalActionType(pattern.action_type)}（样本 ${pattern.sample_size}）`).join(" | ") : "当前还没有提炼出稳定模式。"}</p></div>
            <div><span className={styles.operationLabel}>历史风险提示</span><p>{experienceContext?.outcome_risk_notes.length ? experienceContext.outcome_risk_notes.join(" | ") : "当前没有附带历史风险提示。"}</p></div>
            <div><span className={styles.operationLabel}>长期记忆</span><p>{experienceContext?.long_term_memories.length ? `已命中 ${experienceContext.long_term_memories.length} 条长期记忆。` : "当前还没有命中长期记忆。"}</p></div>
            <div><span className={styles.operationLabel}>最新长期记忆</span><p>{latestLongTermMemory ? `${latestLongTermMemory.headline}：${latestLongTermMemory.recommendations[0] ?? latestLongTermMemory.summary}` : "当前还没有沉淀的长期记忆摘要。"}</p></div>
            <div><span className={styles.operationLabel}>最新复盘摘要</span><p>{latestEpisodeSummary ? `${latestEpisodeSummary.headline}：${latestEpisodeSummary.reusable_rules[0] ?? latestEpisodeSummary.key_decisions[0] ?? "已完成高风险复盘。"}`
              : "当前还没有生成高风险复盘摘要。"}</p></div>
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>评测</p><h3>代理评测快照</h3></div>
          </div>
          <div className={styles.memoryList}>
            <div><span className={styles.operationLabel}>评测场景</span><p>{evaluationBenchmarks.length ? `已加载 ${evaluationBenchmarks.length} 个评测场景。` : "当前还没有评测场景定义。"}</p></div>
            <div><span className={styles.operationLabel}>工具选择正确率</span><p>{formatPercent(latestEvaluationReport?.tool_selection_correctness)}</p></div>
            <div><span className={styles.operationLabel}>动态分派正确率</span><p>{formatPercent(latestEvaluationReport?.dynamic_dispatch_correctness)}</p></div>
            <div><span className={styles.operationLabel}>共享记忆复用率</span><p>{formatPercent(latestEvaluationReport?.shared_memory_reuse_rate)}</p></div>
            <div><span className={styles.operationLabel}>任务图延迟</span><p>{typeof agentMetrics?.task_graph_latency_ms === "number" ? `${Math.round(agentMetrics.task_graph_latency_ms)} ms` : "--"}</p></div>
            <div><span className={styles.operationLabel}>人工升级正确率</span><p>{formatPercent(latestEvaluationReport?.human_escalation_correctness)}</p></div>
            <div><span className={styles.operationLabel}>幻觉风险</span><p>{formatPercent(latestEvaluationReport?.hallucination_rate)}</p></div>
            <div><span className={styles.operationLabel}>最近报告</span><p>{latestEvaluationReport ? formatTimestamp(latestEvaluationReport.created_at) : "当前还没有生成评测报告。"}</p></div>
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>任务队列</p><h3>任务队列</h3></div>
          </div>
          <div className={styles.executionList}>
            {agentTasks.length ? (
              agentTasks.slice(0, 8).map((task) => (
                <article key={task.task_id} className={styles.executionCard}>
                  <div className={styles.executionTopline}>
                    <strong>{agentText[task.agent_name]}</strong>
                    <span className={`${styles.statusPill} ${taskStatusClass(task.status)}`}>{taskStatusText[task.status]}</span>
                  </div>
                  <p>{formatAgentTaskType(task.task_type)}</p>
                  <div className={styles.executionMeta}>
                    <span>{formatTimestamp(task.created_at)}</span>
                    <span>P{task.priority}</span>
                    {task.replayed_from_task_id ? <span>回放任务</span> : null}
                  </div>
                  {task.failure_reason ? <small>{task.failure_reason}</small> : null}
                  {(task.status === "failed" || task.status === "completed") ? (
                    <button type="button" className={styles.secondaryButton} aria-label={`replay-task-${task.task_id}`} disabled={busy || !canReplayTask} onClick={() => void onReplayTask(task.task_id, replayReason || "从任务时间线发起人工回放。")}>
                      重放任务
                    </button>
                  ) : null}
                </article>
              ))
            ) : <p className={styles.emptyState}>当前事件还没有创建代理任务。</p>}
          </div>
          <label className={styles.fieldBlock}>
            <span>重放原因</span>
            <input className={styles.fieldInput} value={replayReason} onChange={(event) => setReplayReason(event.target.value)} placeholder="为单条任务重放补充原因说明。" />
          </label>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>时间线</p><h3>任务时间线</h3></div>
          </div>
          <div className={styles.executionList}>
            {agentTimeline.length ? (
              agentTimeline.slice(0, 8).map((entry) => (
                <article key={entry.entry_id} className={styles.executionCard}>
                  <div className={styles.executionTopline}>
                    <strong>{entry.entry_type === "trigger" ? formatTriggerType(entry.trigger_type) : formatAgentTaskEventType(entry.task_event_type)}</strong>
                    <span className={styles.statusPill}>{entry.agent_name ? agentText[entry.agent_name] : "触发事件"}</span>
                  </div>
                  <p>{normalizeAgentTerminology(entry.summary)}</p>
                  <div className={styles.executionMeta}>
                    {entry.task_id ? <span>{entry.task_id}</span> : null}
                    {entry.trigger_id ? <span>{entry.trigger_id}</span> : null}
                    <span>{formatTimestamp(entry.created_at)}</span>
                  </div>
                </article>
              ))
            ) : <p className={styles.emptyState}>尚未记录任务时间线。</p>}
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>触发总线</p><h3>触发总线</h3></div>
          </div>
          <div className={styles.executionList}>
            {triggerEvents.length ? (
              triggerEvents.slice(0, 8).map((trigger) => (
                <article key={trigger.trigger_id} className={styles.executionCard}>
                  <div className={styles.executionTopline}>
                    <strong>{formatTriggerType(trigger.trigger_type)}</strong>
                    <span className={`${styles.statusPill} ${trigger.status === "processed" ? styles.statusApproved : trigger.status === "failed" ? styles.statusRejected : styles.statusPending}`}>{formatTriggerStatus(trigger.status)}</span>
                  </div>
                  <p>{trigger.error_message || "触发事件已写入调度总线，等待代理处理。"}</p>
                  <div className={styles.executionMeta}>
                    <span>{formatTimestamp(trigger.created_at)}</span>
                    <span>{trigger.dedupe_key ?? "未设置去重键"}</span>
                  </div>
                </article>
              ))
            ) : <p className={styles.emptyState}>当前事件尚未发布任何触发事件。</p>}
          </div>
        </div>
      </div>

      <div className={styles.adminCard}>
        <div className={styles.adminCardHeader}>
          <div><p className={styles.sectionLabel}>结果输出</p><h3>最近代理结果</h3></div>
        </div>
        <div className={styles.executionList}>
          {recentAgentResults.length ? (
            recentAgentResults.slice(0, 6).map((result) => (
              <article key={result.result_id} className={styles.executionCard}>
                <div className={styles.executionTopline}>
                  <strong>{agentText[result.agent_name]}</strong>
                  <span>{Math.round(result.confidence * 100)}%</span>
                </div>
                <p>{normalizeAgentTerminology(result.summary)}</p>
                <div className={styles.executionMeta}>
                  <span>{result.evidence_refs.length} 条证据引用</span>
                  <span>{Math.round((result.decision_confidence ?? result.confidence) * 100)}% 决策置信度</span>
                  <span>{formatTimestamp(result.created_at)}</span>
                </div>
                {result.recommended_next_tasks?.length ? <small>建议下一步：{result.recommended_next_tasks.map((item) => formatAgentTaskType(item)).join("、")}</small> : null}
                {result.stop_reason ? <small>停止原因：{formatStopReason(result.stop_reason)}</small> : null}
                {result.handoff_recommendations.length ? <small>建议移交：{result.handoff_recommendations.map((item) => formatAgentHandoffTarget(item)).join("、")}</small> : null}
              </article>
            ))
          ) : <p className={styles.emptyState}>尚未有代理结果进入值班席视图。</p>}
        </div>
      </div>
    </div>
  );
}
