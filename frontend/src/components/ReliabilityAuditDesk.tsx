import { FormEvent, useState } from "react";
import styles from "../App.module.css";
import { formatAuditAction, formatCircuitState, formatSourceType, formatSupervisorRunStatus } from "../lib/displayText";
import { formatTimestamp, severityText } from "../lib/consoleFormatting";
import { AccessPolicyNotice, actionRequiredRoleText } from "./SecurityDesk";
import type { ArchiveStatusView, AuditRecord, OperationalAlert, SupervisorLoopStatus } from "../types/api";

function severityClass(severity: "info" | "warning" | "critical") {
  return {
    info: styles.statusApproved,
    warning: styles.statusPending,
    critical: styles.statusRejected,
  }[severity];
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

export interface ReliabilityAuditDeskProps {
  eventId?: string;
  supervisorLoopStatus: SupervisorLoopStatus | null;
  alerts: OperationalAlert[];
  auditRecords: AuditRecord[];
  archiveStatus: ArchiveStatusView | null;
  busy: boolean;
  canRunArchive: boolean;
  onQueryAudit: (filters?: {
    severity?: string;
    sourceType?: string;
    fromTs?: string;
    toTs?: string;
    limit?: number;
  }) => Promise<void>;
  onRunArchive: () => Promise<void>;
}

export function ReliabilityAuditDesk({
  eventId,
  supervisorLoopStatus,
  alerts,
  auditRecords,
  archiveStatus,
  busy,
  canRunArchive,
  onQueryAudit,
  onRunArchive,
}: ReliabilityAuditDeskProps) {
  const [severityFilter, setSeverityFilter] = useState("");
  const [sourceTypeFilter, setSourceTypeFilter] = useState("");
  const [fromTs, setFromTs] = useState("");
  const [toTs, setToTs] = useState("");

  function handleAuditQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void onQueryAudit({
      severity: severityFilter || undefined,
      sourceType: sourceTypeFilter || undefined,
      fromTs: fromTs ? new Date(fromTs).toISOString() : undefined,
      toTs: toTs ? new Date(toTs).toISOString() : undefined,
      limit: 12,
    });
  }

  return (
    <div className={styles.agentDesk}>
      <div className={styles.panelHeader}>
        <div>
          <p className={styles.sectionLabel}>可靠性与审计</p>
          <h2>巡检健康状态、告警、归档留存与审计查询</h2>
        </div>
        <div className={styles.bulkToolbar}>
          <button type="button" className={styles.secondaryButton} disabled={!eventId || busy || !canRunArchive} aria-label="run-archive-cycle" onClick={() => void onRunArchive()}>
            执行归档周期
          </button>
        </div>
      </div>
      <AccessPolicyNotice
        title="页面策略"
        summary="运行健康遥测对所有角色可见，但会改变留存状态的动作仍保持严格权限控制。"
        items={[
          { label: "查看巡检健康、告警与审计记录", allowed: true, requiredRole: "观察员", description: "运行可见性默认向所有席位开放，便于理解系统为何降级或被阻断。" },
          { label: "执行归档与清理周期", allowed: canRunArchive, requiredRole: actionRequiredRoleText.archive_run, description: "归档任务会移动热数据和历史日志，因此保持为指挥长级别的控制动作。" },
        ]}
      />

      <div className={styles.agentStatusStrip}>
        {metric("巡检健康", supervisorLoopStatus?.running ? "运行中" : "已停止", supervisorLoopStatus ? `每 ${supervisorLoopStatus.interval_seconds} 秒一次` : "暂无巡检状态")}
        {metric("熔断状态", formatCircuitState(supervisorLoopStatus?.circuit_state), supervisorLoopStatus?.circuit_expires_at ? `持续到 ${formatTimestamp(supervisorLoopStatus.circuit_expires_at)}` : "当前没有熔断冷却")}
        {metric("开放告警", `${alerts.length}`, alerts[0]?.summary ?? "当前没有站内告警")}
        {metric("触发积压", `${supervisorLoopStatus?.pending_trigger_count ?? 0}`, supervisorLoopStatus?.last_trigger_processed_at ? `最近处理于 ${formatTimestamp(supervisorLoopStatus.last_trigger_processed_at)}` : "还没有处理过触发事件")}
        {metric("归档状态", archiveStatus ? `${archiveStatus.hot_record_count} 条热数据 / ${archiveStatus.archived_record_count} 条归档` : "--", formatSupervisorRunStatus(archiveStatus?.last_archive_run?.status) || "还没有归档记录")}
      </div>

      <div className={styles.agentGrid}>
        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>巡检健康</p><h3>巡检调度状态</h3></div>
            {supervisorLoopStatus ? <span className={`${styles.statusPill} ${supervisorLoopStatus.running ? styles.statusApproved : styles.executionSkipped}`}>{supervisorLoopStatus.running ? "运行中" : "已停止"}</span> : null}
          </div>
          <div className={styles.memoryList}>
            <div><span className={styles.operationLabel}>熔断状态</span><p>{formatCircuitState(supervisorLoopStatus?.circuit_state)}</p></div>
            <div><span className={styles.operationLabel}>连续失败</span><p>{supervisorLoopStatus?.consecutive_failures ?? 0}</p></div>
            <div><span className={styles.operationLabel}>最近成功</span><p>{formatTimestamp(supervisorLoopStatus?.last_success_at)}</p></div>
            <div><span className={styles.operationLabel}>最近失败</span><p>{formatTimestamp(supervisorLoopStatus?.last_failure_at)}</p></div>
            <div><span className={styles.operationLabel}>最近重试</span><p>{formatTimestamp(supervisorLoopStatus?.last_retry_at)}</p></div>
            <div><span className={styles.operationLabel}>跳过巡检次数</span><p>{supervisorLoopStatus?.skipped_sweeps ?? 0}</p></div>
            <div><span className={styles.operationLabel}>待处理触发</span><p>{supervisorLoopStatus?.pending_trigger_count ?? 0}</p></div>
            <div><span className={styles.operationLabel}>任务回放次数</span><p>{supervisorLoopStatus?.recent_replay_count ?? 0}</p></div>
            <div><span className={styles.operationLabel}>时间线失败次数</span><p>{supervisorLoopStatus?.recent_timeline_failure_count ?? 0}</p></div>
            <div><span className={styles.operationLabel}>最近错误</span><p>{supervisorLoopStatus?.last_error || "暂无后台巡检错误。"}</p></div>
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>运行告警</p><h3>站内告警</h3></div>
          </div>
          <div className={styles.executionList}>
            {alerts.length ? alerts.map((alert) => (
              <article key={alert.alert_id} className={styles.executionCard}>
                <div className={styles.executionTopline}>
                  <strong>{alert.summary}</strong>
                  <span className={`${styles.statusPill} ${severityClass(alert.severity)}`}>{severityText(alert.severity)}</span>
                </div>
                <p>{alert.details || "当前告警没有更多细节说明。"}</p>
                <div className={styles.executionMeta}>
                  <span>{formatSourceType(alert.source_type)}</span>
                  <span>{formatTimestamp(alert.last_seen_at)}</span>
                </div>
              </article>
            )) : <p className={styles.emptyState}>当前没有处于激活状态的站内告警。</p>}
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>审计查询</p><h3>查询最近记录</h3></div>
          </div>
          <form className={styles.formGrid} onSubmit={handleAuditQuery}>
            <label className={styles.fieldBlock}>
              <span>严重级别</span>
              <select className={styles.fieldInput} value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value)}>
                <option value="">全部</option>
                <option value="info">信息</option>
                <option value="warning">告警</option>
                <option value="critical">严重</option>
              </select>
            </label>
            <label className={styles.fieldBlock}>
              <span>来源</span>
              <input className={styles.fieldInput} value={sourceTypeFilter} onChange={(event) => setSourceTypeFilter(event.target.value)} placeholder="例如：后台巡检 / 运行期数据管理" />
            </label>
            <label className={styles.fieldBlock}>
              <span>开始时间</span>
              <input className={styles.fieldInput} type="datetime-local" value={fromTs} onChange={(event) => setFromTs(event.target.value)} />
            </label>
            <label className={styles.fieldBlock}>
              <span>结束时间</span>
              <input className={styles.fieldInput} type="datetime-local" value={toTs} onChange={(event) => setToTs(event.target.value)} />
            </label>
            <div className={styles.fieldBlockFull}>
              <button type="submit" className={styles.secondaryButton} disabled={busy} aria-label="query-audit-records">
                查询审计记录
              </button>
            </div>
          </form>
          <div className={styles.executionList}>
            {auditRecords.length ? auditRecords.map((record) => (
              <article key={record.audit_id} className={styles.executionCard}>
                <div className={styles.executionTopline}>
                  <strong>{record.summary}</strong>
                  <span className={`${styles.statusPill} ${severityClass(record.severity)}`}>{severityText(record.severity)}</span>
                </div>
                <p>{formatAuditAction(record.action)}</p>
                <div className={styles.executionMeta}>
                  <span>{formatSourceType(record.source_type)}</span>
                  <span>{formatTimestamp(record.created_at)}</span>
                </div>
              </article>
            )) : <p className={styles.emptyState}>当前筛选条件下没有审计记录。</p>}
          </div>
        </div>

        <div className={styles.adminCard}>
          <div className={styles.adminCardHeader}>
            <div><p className={styles.sectionLabel}>归档状态</p><h3>留存与清理</h3></div>
          </div>
          <div className={styles.memoryList}>
            <div><span className={styles.operationLabel}>热数据保留</span><p>{archiveStatus ? `${archiveStatus.hot_retention_days} 天` : "--"}</p></div>
            <div><span className={styles.operationLabel}>归档保留</span><p>{archiveStatus ? `${archiveStatus.archive_retention_days} 天` : "--"}</p></div>
            <div><span className={styles.operationLabel}>最近归档运行</span><p>{formatTimestamp(archiveStatus?.last_archive_run?.completed_at ?? archiveStatus?.last_archive_run?.started_at)}</p></div>
            <div><span className={styles.operationLabel}>本轮归档条数</span><p>{archiveStatus?.last_archive_run?.hot_records_archived ?? 0}</p></div>
            <div><span className={styles.operationLabel}>清理过期归档</span><p>{archiveStatus?.last_archive_run?.expired_archives_deleted ?? 0}</p></div>
            <div><span className={styles.operationLabel}>影响表</span><p>{archiveStatus?.last_archive_run?.tables_touched.length ? archiveStatus.last_archive_run.tables_touched.join(", ") : "暂无归档记录。"}</p></div>
          </div>
        </div>
      </div>
    </div>
  );
}
