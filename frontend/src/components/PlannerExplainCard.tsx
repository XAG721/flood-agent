import styles from "../App.module.css";
import type {
  DataFreshnessSummary,
  MemorySnapshot,
  PlanRunRecord,
  StructuredAnswer,
  ToolExecutionResultView,
} from "../types/api";

function toolStatusLabel(status: ToolExecutionResultView["status"]) {
  return {
    success: "成功",
    failed: "失败",
    skipped: "跳过",
    timeout: "超时",
  }[status];
}

function toolStatusClass(status: ToolExecutionResultView["status"]) {
  return {
    success: styles.executionSuccess,
    failed: styles.executionFailed,
    skipped: styles.executionSkipped,
    timeout: styles.executionTimeout,
  }[status];
}

function formatFreshness(value?: number | null) {
  if (value === null || value === undefined) return "未知";
  if (value < 60) return `${value} 秒`;
  return `${Math.round(value / 60)} 分钟`;
}

function freshnessChips(dataFreshness?: DataFreshnessSummary) {
  if (!dataFreshness) return [];
  const chips: string[] = [];
  if (dataFreshness.hazard_state_freshness_seconds !== undefined) {
    chips.push(`风险态势 ${formatFreshness(dataFreshness.hazard_state_freshness_seconds)}`);
  }
  if (
    dataFreshness.traffic_freshness_seconds !== undefined &&
    dataFreshness.traffic_freshness_seconds !== null
  ) {
    chips.push(`交通路况 ${formatFreshness(dataFreshness.traffic_freshness_seconds)}`);
  }
  if (dataFreshness.profile_freshness_label) {
    chips.push(`对象画像 ${dataFreshness.profile_freshness_label}`);
  }
  if (dataFreshness.rag_document_recency_summary) {
    chips.push(dataFreshness.rag_document_recency_summary);
  }
  return chips;
}

function completionStatusLabel(status?: StructuredAnswer["completion_status"]) {
  return {
    direct_answer: "可直接回答",
    conservative_answer: "保守建议",
    human_escalation: "需要人工确认",
  }[status ?? "direct_answer"];
}

function memoryRows(memory?: MemorySnapshot | null) {
  if (!memory) return [];
  return [
    ["当前关注对象", memory.focus_entity_name || memory.focus_entity_id || "暂无"],
    ["当前目标", memory.current_goal || "暂无"],
    [
      "待处理请示",
      memory.pending_proposal_ids.length ? memory.pending_proposal_ids.join(", ") : "暂无",
    ],
    [
      "已执行请示",
      memory.executed_proposal_ids.length ? memory.executed_proposal_ids.join(", ") : "暂无",
    ],
    [
      "未解决槽位",
      memory.unresolved_slots.length ? memory.unresolved_slots.join(", ") : "暂无",
    ],
  ];
}

function planRunLabel(item: PlanRunRecord) {
  const layerLabel =
    item.planning_layer === "rule"
      ? "规则"
      : item.planning_layer === "llm"
        ? "模型"
        : item.planning_layer === "merged"
          ? "合并"
          : "重规划";
  return `${layerLabel} · 第 ${item.replan_round + 1} 轮`;
}

interface PlannerExplainCardProps {
  answer: StructuredAnswer | null;
}

export function PlannerExplainCard({ answer }: PlannerExplainCardProps) {
  if (!answer) {
    return (
      <p className={styles.emptyState}>
        发起一次问答后，这里会展示规划说明、工具执行矩阵、数据新鲜度和会话记忆摘要。
      </p>
    );
  }

  const toolSelectionReasoning = answer.tool_selection_reasoning ?? [];
  const skippedTools = answer.skipped_tools ?? [];
  const toolExecutions = answer.tool_executions ?? [];
  const evidenceGaps = answer.evidence_gaps ?? [];
  const freshness = freshnessChips(answer.data_freshness);
  const planningLayers = answer.planning_layers_summary ?? [];
  const planRuns = answer.plan_runs ?? [];
  const usedFallbacks = answer.used_fallbacks ?? [];
  const carriedContextNotes = answer.carried_context_notes ?? [];
  const memory = memoryRows(answer.memory_snapshot);

  return (
    <div className={styles.plannerCard}>
      <div className={styles.panelHeaderCompact}>
        <div>
          <p className={styles.sectionLabel}>规划解释</p>
          <h3>{answer.planner_summary || "本轮问答规划摘要"}</h3>
        </div>
      </div>

      {freshness.length ? (
        <div className={styles.explainChipRow}>
          {freshness.map((chip) => (
            <span key={chip}>{chip}</span>
          ))}
        </div>
      ) : null}

      <div className={styles.explainBlock}>
        <span className={styles.operationLabel}>完成状态</span>
        <div className={styles.executionMeta}>
          <span>{completionStatusLabel(answer.completion_status)}</span>
          {answer.termination_reason ? <span>{answer.termination_reason}</span> : null}
          {typeof answer.replan_count === "number" ? (
            <span>重规划 {answer.replan_count} 次</span>
          ) : null}
        </div>
      </div>

      {planningLayers.length ? (
        <div className={styles.explainBlock}>
          <span className={styles.operationLabel}>规划层</span>
          <ul className={styles.reasonList}>
            {planningLayers.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className={styles.plannerGrid}>
        <div className={styles.explainBlock}>
          <span className={styles.operationLabel}>为什么选择这些工具</span>
          {toolSelectionReasoning.length ? (
            <ul className={styles.reasonList}>
              {toolSelectionReasoning.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className={styles.emptyState}>当前没有额外的工具选择说明。</p>
          )}
        </div>

        <div className={styles.explainBlock}>
          <span className={styles.operationLabel}>未调用的工具</span>
          {skippedTools.length ? (
            <ul className={styles.reasonList}>
              {skippedTools.map((item) => (
                <li key={`${item.tool_name}_${item.reason}`}>
                  <strong>{item.tool_name}</strong>：{item.reason}
                </li>
              ))}
            </ul>
          ) : (
            <p className={styles.emptyState}>当前没有被显式跳过的工具。</p>
          )}
        </div>
      </div>

      <div className={styles.plannerGrid}>
        <div className={styles.explainBlock}>
          <span className={styles.operationLabel}>规划轮次</span>
          {planRuns.length ? (
            <ul className={styles.reasonList}>
              {planRuns.map((item) => (
                <li key={item.plan_run_id}>
                  <strong>{planRunLabel(item)}</strong>：{item.selected_tools.join(", ")}
                </li>
              ))}
            </ul>
          ) : (
            <p className={styles.emptyState}>当前没有记录到规划轮次。</p>
          )}
        </div>

        <div className={styles.explainBlock}>
          <span className={styles.operationLabel}>会话记忆</span>
          {memory.length ? (
            <ul className={styles.reasonList}>
              {memory.map(([label, value]) => (
                <li key={label}>
                  <strong>{label}</strong>：{value}
                </li>
              ))}
            </ul>
          ) : (
            <p className={styles.emptyState}>当前还没有可展示的会话记忆。</p>
          )}
        </div>
      </div>

      <div className={styles.explainBlock}>
        <span className={styles.operationLabel}>工具执行矩阵</span>
        {toolExecutions.length ? (
          <div className={styles.executionList}>
            {toolExecutions.map((item) => (
              <article
                key={`${item.execution_id ?? item.tool_name}_${item.duration_ms}_${item.status}`}
                className={styles.executionCard}
              >
                <div className={styles.executionTopline}>
                  <strong>{item.tool_name}</strong>
                  <span className={`${styles.statusPill} ${toolStatusClass(item.status)}`}>
                    {toolStatusLabel(item.status)}
                  </span>
                </div>
                <p>{item.output_summary}</p>
                <div className={styles.executionMeta}>
                  <span>{item.duration_ms} ms</span>
                  <span>数据新鲜度 {formatFreshness(item.data_freshness_seconds)}</span>
                  {item.parallel_group ? <span>并行组 {item.parallel_group}</span> : null}
                  {item.attempt && item.attempt > 1 ? <span>第 {item.attempt} 次尝试</span> : null}
                  {item.cache_hit ? <span>命中缓存</span> : null}
                  {item.fallback_from_tool ? <span>回退自 {item.fallback_from_tool}</span> : null}
                  {item.timed_out ? <span>已超时</span> : null}
                </div>
                {item.failure_reason ? <small>{item.failure_reason}</small> : null}
              </article>
            ))}
          </div>
        ) : (
          <p className={styles.emptyState}>当前没有记录到工具执行明细。</p>
        )}
      </div>

      {evidenceGaps.length ? (
        <div className={styles.explainBlock}>
          <span className={styles.operationLabel}>证据缺口</span>
          <ul className={styles.reasonList}>
            {evidenceGaps.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {usedFallbacks.length || carriedContextNotes.length ? (
        <div className={styles.plannerGrid}>
          <div className={styles.explainBlock}>
            <span className={styles.operationLabel}>回退链路</span>
            {usedFallbacks.length ? (
              <ul className={styles.reasonList}>
                {usedFallbacks.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className={styles.emptyState}>当前没有使用回退链路。</p>
            )}
          </div>
          <div className={styles.explainBlock}>
            <span className={styles.operationLabel}>上下文沿用说明</span>
            {carriedContextNotes.length ? (
              <ul className={styles.reasonList}>
                {carriedContextNotes.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className={styles.emptyState}>当前没有沿用上一轮上下文。</p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
