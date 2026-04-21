import styles from "../App.module.css";
import type { RegionalAnalysisPackageView } from "../types/api";

interface RegionalAnalysisPackageHistoryPanelProps {
  items: RegionalAnalysisPackageView[];
}

const packageStatusText: Record<string, string> = {
  pending: "待审批",
  approved: "已批准",
  rejected: "已驳回",
  withdrawn: "已撤回",
  superseded: "已过期",
  partially_resolved: "部分处理",
};

export function RegionalAnalysisPackageHistoryPanel({
  items,
}: RegionalAnalysisPackageHistoryPanelProps) {
  if (!items.length) {
    return <p className={styles.emptyState}>当前还没有区域分析包历史。</p>;
  }

  return (
    <div style={{ display: "grid", gap: "14px" }}>
      {items.map((item) => (
        <article key={item.package_id} className={styles.panelFrame} style={{ padding: "18px" }}>
          <div className={styles.panelHeaderCompact}>
            <div>
              <p className={styles.sectionLabel}>分析包 {item.package_id}</p>
              <h3>{item.analysis_message}</h3>
            </div>
            <span className={styles.statusPill}>{packageStatusText[item.status] ?? item.status}</span>
          </div>

          <div className={styles.routeSummary}>
            <div>
              <span>风险等级</span>
              <strong>{item.current_risk_level}</strong>
            </div>
            <div>
              <span>重点对象</span>
              <strong>{item.focus_object_names.join(", ") || item.focus_object_ids.length}</strong>
            </div>
            <div>
              <span>动作数</span>
              <strong>{item.proposal_count}</strong>
            </div>
          </div>

          <ul className={styles.reasonList}>
            <li>{item.risk_assessment}</li>
            <li>{item.rescue_plan}</li>
            <li>{item.resource_dispatch_plan}</li>
          </ul>

          <div className={styles.routeSummary}>
            <div>
              <span>包含动作</span>
              <strong>{item.proposal_titles.join(", ")}</strong>
            </div>
            <div>
              <span>更新时间</span>
              <strong>{item.updated_at}</strong>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
