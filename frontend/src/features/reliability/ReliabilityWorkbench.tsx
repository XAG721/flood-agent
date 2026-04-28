import type { ComponentProps } from "react";
import styles from "../../App.module.css";
import { ReliabilityAuditDesk } from "../../components/ReliabilityAuditDesk";
import { SecurityDesk } from "../../components/SecurityDesk";
import { ReliabilityPage } from "../../pages/ReliabilityPage";
import type {
  AgentCouncilView,
  AudienceWarningDraft,
  AuditRecord,
  OperationalAlert,
  OperatorCapabilitiesView,
  OperatorRole,
} from "../../types/api";

interface ReliabilityWorkbenchProps {
  agentCouncil?: AgentCouncilView | null;
  eventId?: string;
  supervisorLoopStatus?: ComponentProps<typeof ReliabilityAuditDesk>["supervisorLoopStatus"];
  alerts: OperationalAlert[];
  auditRecords: AuditRecord[];
  archiveStatus?: ComponentProps<typeof ReliabilityAuditDesk>["archiveStatus"];
  busy: boolean;
  canRunArchive: boolean;
  onQueryAudit: ComponentProps<typeof ReliabilityAuditDesk>["onQueryAudit"];
  onRunArchive: ComponentProps<typeof ReliabilityAuditDesk>["onRunArchive"];
  operatorRole: OperatorRole;
  operatorCapabilities: OperatorCapabilitiesView | null;
  onChangeRole: (role: OperatorRole) => void;
  pendingProposalCount: number;
  approvedProposalCount: number;
  warningDraftCount: number;
  latestWarningDrafts: AudienceWarningDraft[];
  twinStreamStatus: string;
}

export function ReliabilityWorkbench({
  agentCouncil,
  eventId,
  supervisorLoopStatus,
  alerts,
  auditRecords,
  archiveStatus,
  busy,
  canRunArchive,
  onQueryAudit,
  onRunArchive,
  operatorRole,
  operatorCapabilities,
  onChangeRole,
  pendingProposalCount,
  approvedProposalCount,
  warningDraftCount,
  latestWarningDrafts,
  twinStreamStatus,
}: ReliabilityWorkbenchProps) {
  const councilRoles = agentCouncil?.roles ?? [];

  return (
    <ReliabilityPage
      health={
        <div className={styles.primaryColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Agent Council Audit</p>
                <h3>会商与审计判定</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {councilRoles.map((role) => (
                <article key={role.role} className={styles.metricBlock}>
                  <span>{role.label}</span>
                  <strong>{role.summary}</strong>
                  <small>{role.recommended_action ?? "等待新的审计结论或治理动作。"}</small>
                </article>
              ))}
              {!councilRoles.length ? (
                <div className={styles.emptyState}>当前没有新的会商摘要，系统将继续等待触发和证据汇聚。</div>
              ) : null}
            </div>
            <div className={styles.answerTags}>
              <span>Audit: {agentCouncil?.audit_decision.status ?? "unknown"}</span>
              <span>Open questions: {agentCouncil?.open_questions.length ?? 0}</span>
              <span>Blocked by: {agentCouncil?.blocked_by.length ?? 0}</span>
            </div>
          </div>
          <div className={styles.panelFrame}>
            <ReliabilityAuditDesk
              eventId={eventId}
              supervisorLoopStatus={supervisorLoopStatus ?? null}
              alerts={alerts}
              auditRecords={auditRecords}
              archiveStatus={archiveStatus ?? null}
              busy={busy}
              canRunArchive={canRunArchive}
              onQueryAudit={onQueryAudit}
              onRunArchive={onRunArchive}
            />
          </div>
        </div>
      }
      governance={
        <div className={styles.sideColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Decision Boundary</p>
                <h3>权限与阻断边界</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              <article className={styles.metricBlock}>
                <span>Rationale</span>
                <strong>{agentCouncil?.audit_decision.rationale ?? "等待新的审计说明。"}</strong>
                <small>{(agentCouncil?.audit_decision.risk_flags ?? []).join(" / ") || "当前没有额外风险标记。"}</small>
              </article>
            </div>
            <div className={styles.answerTags}>
              <span>审批要求：{agentCouncil?.audit_decision.approval_required ? "需要人工放行" : "可自动推进"}</span>
              <span>SSE：{twinStreamStatus}</span>
            </div>
          </div>
          <SecurityDesk
            operatorRole={operatorRole}
            operatorCapabilities={operatorCapabilities}
            onChangeRole={onChangeRole}
          />
        </div>
      }
      closure={
        <div className={styles.primaryColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Closure Snapshot</p>
                <h3>审批与预警闭环</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              <article className={styles.metricBlock}>
                <span>Pending proposals</span>
                <strong>{pendingProposalCount}</strong>
                <small>等待审批的人机协同动作仍在主链路中。</small>
              </article>
              <article className={styles.metricBlock}>
                <span>Approved proposals</span>
                <strong>{approvedProposalCount}</strong>
                <small>已批准动作可继续生成 audience warnings 并进入执行留痕。</small>
              </article>
              <article className={styles.metricBlock}>
                <span>Warning drafts</span>
                <strong>{warningDraftCount || latestWarningDrafts.length}</strong>
                <small>分众预警草稿已与 proposal 闭环关联。</small>
              </article>
            </div>
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Latest Outputs</p>
                <h3>最新闭环产物</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {latestWarningDrafts.slice(0, 4).map((draft) => (
                <article key={draft.warning_id} className={styles.metricBlock}>
                  <span>{draft.audience}</span>
                  <strong>{draft.channel}</strong>
                  <small>{draft.grounding_summary || draft.content}</small>
                </article>
              ))}
              {!latestWarningDrafts.length ? (
                <div className={styles.emptyState}>当前还没有新的 warning draft，批准 proposal 后会在这里形成闭环结果。</div>
              ) : null}
            </div>
          </div>
        </div>
      }
    />
  );
}
