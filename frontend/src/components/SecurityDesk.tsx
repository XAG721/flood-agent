import styles from "../App.module.css";
import { operatorRoleTextMap } from "../lib/displayText";
import type { OperatorCapabilitiesView, OperatorRole } from "../types/api";

export const operatorRoleText: Record<OperatorRole, string> = operatorRoleTextMap;

export const actionRequiredRoleText = {
  proposal_resolve: "指挥长",
  runtime_admin_write: "指挥长",
  dataset_manage: "指挥长",
  supervisor_control: "指挥长",
  agent_replay: "指挥长",
  archive_run: "指挥长",
  evaluation_run: "区级值守员",
} as const;

export interface AccessPolicyItem {
  label: string;
  allowed: boolean;
  requiredRole: string;
  description: string;
}

export function AccessPolicyNotice({
  title,
  summary,
  items,
}: {
  title: string;
  summary: string;
  items: AccessPolicyItem[];
}) {
  return (
    <div className={styles.policyNotice}>
      <div className={styles.policyNoticeHeader}>
        <div>
          <span className={styles.operationLabel}>{title}</span>
          <p>{summary}</p>
        </div>
      </div>
      <div className={styles.policyRuleList}>
        {items.map((item) => (
          <article key={item.label} className={styles.policyRuleCard}>
            <div className={styles.executionTopline}>
              <strong>{item.label}</strong>
              <span
                className={`${styles.statusPill} ${
                  item.allowed ? styles.statusApproved : styles.executionSkipped
                }`}
              >
                {item.allowed ? "允许" : "受限"}
              </span>
            </div>
            <p>{item.description}</p>
            <small>所需角色：{item.requiredRole}</small>
          </article>
        ))}
      </div>
    </div>
  );
}

interface SecurityDeskProps {
  operatorRole: OperatorRole;
  operatorCapabilities: OperatorCapabilitiesView | null;
  onChangeRole: (role: OperatorRole) => void;
}

export function SecurityDesk({
  operatorRole,
  operatorCapabilities,
  onChangeRole,
}: SecurityDeskProps) {
  const capabilityEntries = operatorCapabilities
    ? Object.entries(operatorCapabilities.action_labels)
    : [];

  return (
    <div className={styles.panelFrame}>
      <div className={styles.panelHeaderCompact}>
        <div>
          <p className={styles.sectionLabel}>权限控制</p>
          <h3>角色能力与策略说明</h3>
        </div>
      </div>

      <label className={styles.fieldBlock}>
        <span>当前角色</span>
        <select
          className={styles.fieldInput}
          aria-label="operator-role-select"
          value={operatorRole}
          onChange={(event) => onChangeRole(event.target.value as OperatorRole)}
        >
          {Object.entries(operatorRoleText).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>

      <div className={styles.memoryList}>
        <div>
          <span className={styles.operationLabel}>当前角色</span>
          <p>{operatorRoleText[operatorRole]}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>角色等级</span>
          <p>{operatorCapabilities?.role_rank ?? "--"}</p>
        </div>
      </div>

      <div className={styles.documentList}>
        {capabilityEntries.length ? (
          capabilityEntries.map(([action, label]) => {
            const allowed = Boolean(operatorCapabilities?.capabilities[action]);
            return (
              <article key={action} className={styles.documentCard}>
                <div className={styles.executionTopline}>
                  <strong>{label}</strong>
                  <span
                    className={`${styles.statusPill} ${
                      allowed ? styles.statusApproved : styles.executionSkipped
                    }`}
                >
                  {allowed ? "允许" : "受限"}
                </span>
              </div>
                <p>按当前角色能力矩阵自动控制。</p>
              </article>
            );
          })
        ) : (
          <p className={styles.emptyState}>当前还没有加载到角色能力矩阵。</p>
        )}
      </div>

      <AccessPolicyNotice
        title="动作级策略说明"
        summary="当前页面中的高价值动作会根据角色能力矩阵自动放开或限制，并在操作点附近显示原因说明。"
        items={[
          {
            label: "处置方案审批",
            allowed: Boolean(operatorCapabilities?.capabilities.proposal_resolve),
            requiredRole: actionRequiredRoleText.proposal_resolve,
            description: "用于批准、驳回和批量处理区域请示队列，避免低权限角色直接推动高风险动作。",
          },
          {
            label: "运行期数据修改",
            allowed: Boolean(operatorCapabilities?.capabilities.runtime_admin_write),
            requiredRole: actionRequiredRoleText.runtime_admin_write,
            description: "用于编辑对象画像、资源状态与运行期知识文档，修改后会立刻影响评估和智能问答。",
          },
          {
            label: "多代理调度与回放",
            allowed:
              Boolean(operatorCapabilities?.capabilities.supervisor_control) ||
              Boolean(operatorCapabilities?.capabilities.agent_replay),
            requiredRole: actionRequiredRoleText.supervisor_control,
            description: "用于手动触发后台巡检、执行巡检、重放代理任务，属于高影响运维动作。",
          },
          {
            label: "归档与评测",
            allowed:
              Boolean(operatorCapabilities?.capabilities.archive_run) ||
              Boolean(operatorCapabilities?.capabilities.evaluation_run),
            requiredRole: actionRequiredRoleText.archive_run,
            description: "用于运行归档清理和代理评测，主要面向治理、审计与长期值班维护。",
          },
        ]}
      />
    </div>
  );
}
