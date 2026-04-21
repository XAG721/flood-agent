import styles from "../App.module.css";
import {
  formatExecutionMode,
  formatGenerationSource,
  formatRegionalActionDisplayCategory,
  formatRegionalActionDisplayName,
  formatRegionalActionDisplayTagline,
  formatRegionalActionType,
  proposalStatusText,
  riskLevelText,
} from "../lib/displayText";
import type { RegionalProposalView } from "../types/api";

interface RegionalProposalHistoryPanelProps {
  items: RegionalProposalView[];
}

export function RegionalProposalHistoryPanel({ items }: RegionalProposalHistoryPanelProps) {
  if (!items.length) {
    return <p className={styles.emptyState}>当前事件还没有区域级请示历史。</p>;
  }

  return (
    <div style={{ display: "grid", gap: "14px" }}>
      {items.map((item) => (
        <article key={item.proposal.proposal_id} className={styles.panelFrame} style={{ padding: "18px" }}>
          <div className={styles.panelHeaderCompact}>
            <div>
              <p className={styles.sectionLabel}>{item.event_title}</p>
              <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap", marginBottom: "8px" }}>
                <span className={styles.statusPill}>{formatRegionalActionDisplayCategory(item.proposal)}</span>
              </div>
              <h3>{formatRegionalActionDisplayName(item.proposal)}</h3>
              <p className={styles.emptyState} style={{ marginTop: "6px" }}>
                {formatRegionalActionDisplayTagline(item.proposal)}
              </p>
            </div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span className={styles.statusPill}>{proposalStatusText[item.proposal.status]}</span>
              <span className={styles.statusPill}>{formatGenerationSource(item.proposal.generation_source)}</span>
            </div>
          </div>

          <div className={styles.routeSummary}>
            <div>
              <span>动作类型</span>
              <strong>{formatRegionalActionType(item.proposal.action_type)}</strong>
            </div>
            <div>
              <span>执行模式</span>
              <strong>{formatExecutionMode(item.proposal.execution_mode)}</strong>
            </div>
            <div>
              <span>风险等级</span>
              <strong>{riskLevelText[item.current_risk_level]}</strong>
            </div>
          </div>

          <ul className={styles.reasonList}>
            <li>{item.proposal.recommendation || item.proposal.summary}</li>
            <li>{item.proposal.evidence_summary || "暂无证据摘要"}</li>
            {item.proposal.grounding_summary ? <li>{item.proposal.grounding_summary}</li> : null}
          </ul>

          <div className={styles.routeSummary}>
            <div>
              <span>高风险对象</span>
              <strong>{item.high_risk_object_names.join("、") || "暂无"}</strong>
            </div>
            <div>
              <span>模型</span>
              <strong>{item.proposal.model_name || "未标记"}</strong>
            </div>
            <div>
              <span>最近更新</span>
              <strong>{item.proposal.updated_at ?? item.proposal.created_at}</strong>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
