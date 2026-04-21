import { useEffect, useMemo, useState } from "react";
import styles from "../App.module.css";
import { entityText, proposalText, riskText } from "../config/consoleConfig";
import {
  formatGenerationSource,
  formatOperatorActor,
  formatOperatorRole,
  formatRegionalActionType,
} from "../lib/displayText";
import {
  coerceDrafts,
  coerceLogs,
  coerceStrings,
  coerceTemplates,
  formatTimestamp,
} from "../lib/consoleFormatting";
import { AccessPolicyNotice, actionRequiredRoleText } from "./SecurityDesk";
import type { ActionProposalV2, Advisory, EntityImpactView, RiskLevel } from "../types/api";

function proposalStatusClass(status: ActionProposalV2["status"]) {
  return {
    pending: styles.statusPending,
    approved: styles.statusApproved,
    rejected: styles.statusRejected,
    withdrawn: styles.executionSkipped,
    superseded: styles.executionSkipped,
  }[status];
}

export function AdvisoryCard({ advisory }: { advisory: Advisory | null }) {
  if (!advisory) {
    return <p className={styles.emptyState}>生成一条行动建议后，这里会展示对象级处置建议。</p>;
  }
  return (
    <div className={styles.advisoryCard}>
      <div className={styles.advisoryHeader}>
        <div>
          <p className={styles.sectionLabel}>行动建议</p>
          <h3>{advisory.answer}</h3>
          <div className={styles.answerTags}>
            <span>{formatGenerationSource(advisory.generation_source)}</span>
            {advisory.model_name ? <span>{advisory.model_name}</span> : null}
          </div>
        </div>
        <div className={styles.confidenceBlock}>
          <span>置信度</span>
          <strong>{Math.round(advisory.confidence * 100)}%</strong>
        </div>
      </div>
      <ul className={styles.actionList}>
        {advisory.recommended_actions.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      {advisory.grounding_summary ? <p className={styles.emptyState}>{advisory.grounding_summary}</p> : null}
    </div>
  );
}

export interface OperationPanelProps {
  proposals: ActionProposalV2[];
  advisory: Advisory | null;
  isBusy: boolean;
  canResolve: boolean;
  entityTypeLookup: Record<string, EntityImpactView["entity"]["entity_type"]>;
  onResolve: (proposalId: string, decision: "approve" | "reject", note: string) => Promise<void>;
  onBatchResolve: (proposalIds: string[], decision: "approve" | "reject", note: string) => Promise<void>;
}

export function OperationPanel({
  proposals,
  advisory,
  isBusy,
  canResolve,
  entityTypeLookup,
  onResolve,
  onBatchResolve,
}: OperationPanelProps) {
  const [selectedProposalId, setSelectedProposalId] = useState("");
  const [operatorNote, setOperatorNote] = useState("");
  const [entityFilter, setEntityFilter] = useState<"all" | EntityImpactView["entity"]["entity_type"]>("all");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const proposalType = (proposal: ActionProposalV2) =>
    typeof proposal.payload.entity_type === "string"
      ? (proposal.payload.entity_type as EntityImpactView["entity"]["entity_type"])
      : proposal.entity_id
        ? entityTypeLookup[proposal.entity_id]
        : undefined;

  const visibleProposals = useMemo(
    () => (entityFilter === "all" ? proposals : proposals.filter((proposal) => proposalType(proposal) === entityFilter)),
    [entityFilter, proposals],
  );

  useEffect(() => {
    if (!visibleProposals.length) {
      setSelectedProposalId("");
      return;
    }
    if (!visibleProposals.some((item) => item.proposal_id === selectedProposalId)) {
      setSelectedProposalId(visibleProposals[0].proposal_id);
    }
  }, [selectedProposalId, visibleProposals]);

  const selectedProposal =
    visibleProposals.find((item) => item.proposal_id === selectedProposalId) ?? visibleProposals[0] ?? null;

  useEffect(() => {
    setOperatorNote(selectedProposal?.resolution_note ?? "");
  }, [selectedProposal?.proposal_id]);

  const pendingVisible = visibleProposals.filter((item) => item.status === "pending");
  const selectedPendingIds = selectedIds.filter((proposalId) =>
    pendingVisible.some((item) => item.proposal_id === proposalId),
  );
  const drafts = coerceDrafts(selectedProposal?.payload.notification_drafts);
  const logs = coerceLogs(selectedProposal?.payload.execution_logs);
  const approvalActions = coerceStrings(selectedProposal?.payload.approval_actions);
  const recommendedActions = coerceStrings(selectedProposal?.payload.recommended_actions);
  const templatePreviews = coerceTemplates(selectedProposal?.payload.notification_templates);
  const typeOptions = Array.from(new Set(proposals.map((proposal) => proposalType(proposal)).filter(Boolean))) as EntityImpactView["entity"]["entity_type"][];

  return (
    <div className={styles.operationShell}>
      <div className={styles.operationHeader}>
        <div>
          <p className={styles.sectionLabel}>方案处置</p>
          <h3>审批与执行联动</h3>
        </div>
        <div className={styles.operationCounts}>
          <span>{proposals.filter((item) => item.status === "pending").length} 条待处理</span>
          <span>{proposals.filter((item) => item.status === "approved").length} 条已批准</span>
          <span>{proposals.filter((item) => item.status === "rejected").length} 条已驳回</span>
        </div>
      </div>
      <AccessPolicyNotice
        title="动作策略"
        summary="所有角色都可以查看区域请示，但真正关闭队列的动作仍保留人工门禁。"
        items={[
          {
            label: "批准或驳回区域请示",
            allowed: canResolve,
            requiredRole: actionRequiredRoleText.proposal_resolve,
            description: "关闭区域请示时会同步生成执行联动与操作备注，因此保持在指挥权限下。",
          },
        ]}
      />

      {proposals.length ? (
        <>
          <div className={styles.queueToolbar}>
            <div className={styles.filterGroup}>
              <label className={styles.operationLabel} htmlFor="proposal-entity-filter">对象类型</label>
              <select
                id="proposal-entity-filter"
                aria-label="proposal-entity-filter"
                className={styles.filterSelect}
                value={entityFilter}
                onChange={(event) => setEntityFilter(event.target.value === "all" ? "all" : (event.target.value as EntityImpactView["entity"]["entity_type"]))}
              >
                <option value="all">全部对象</option>
                {typeOptions.map((type) => <option key={type} value={type}>{entityText[type]}</option>)}
              </select>
            </div>
            <div className={styles.bulkToolbar}>
              <label className={styles.bulkToggle}>
                <input
                  type="checkbox"
                  aria-label="select-all-visible-proposals"
                  checked={pendingVisible.length > 0 && selectedPendingIds.length === pendingVisible.length}
                  onChange={(event) =>
                    setSelectedIds((current) =>
                      event.target.checked
                        ? Array.from(new Set([...current, ...pendingVisible.map((item) => item.proposal_id)]))
                        : current.filter((id) => !pendingVisible.some((item) => item.proposal_id === id)),
                    )
                  }
                  disabled={!pendingVisible.length || isBusy}
                />
                <span>已选择 {selectedPendingIds.length} 条</span>
              </label>
              <button type="button" className={styles.secondaryButton} aria-label="batch-reject-proposals" disabled={!selectedPendingIds.length || isBusy || !canResolve} onClick={() => void onBatchResolve(selectedPendingIds, "reject", operatorNote)}>批量驳回</button>
              <button type="button" className={styles.primaryButton} aria-label="batch-approve-proposals" disabled={!selectedPendingIds.length || isBusy || !canResolve} onClick={() => void onBatchResolve(selectedPendingIds, "approve", operatorNote)}>批量批准</button>
            </div>
          </div>

          <div className={styles.proposalQueue}>
            {visibleProposals.map((proposal) => (
              <div key={proposal.proposal_id} className={styles.proposalQueueRow}>
                <label className={styles.queueCheckbox}>
                  <input
                    type="checkbox"
                    aria-label={`select-proposal-${proposal.proposal_id}`}
                    checked={selectedIds.includes(proposal.proposal_id)}
                    onChange={(event) =>
                      setSelectedIds((current) =>
                        event.target.checked ? Array.from(new Set([...current, proposal.proposal_id])) : current.filter((item) => item !== proposal.proposal_id),
                      )
                    }
                    disabled={proposal.status !== "pending" || isBusy || !canResolve}
                  />
                </label>
                <button type="button" className={`${styles.proposalChip} ${proposal.proposal_id === selectedProposal?.proposal_id ? styles.proposalChipActive : ""}`} onClick={() => setSelectedProposalId(proposal.proposal_id)}>
                  <span>{proposal.title}</span>
                  <div className={styles.queueMeta}>
                    {proposalType(proposal) ? <em>{entityText[proposalType(proposal)!]}</em> : null}
                    <strong className={proposalStatusClass(proposal.status)}>{proposalText[proposal.status]}</strong>
                  </div>
                </button>
              </div>
            ))}
          </div>

          {selectedProposal ? (
            <div className={styles.proposalDetail}>
              <div className={styles.proposalHeadline}>
                <div><h4>{selectedProposal.title}</h4><p>{selectedProposal.summary}</p></div>
                <div className={styles.proposalMeta}>
                  <span className={`${styles.statusPill} ${proposalStatusClass(selectedProposal.status)}`}>{proposalText[selectedProposal.status]}</span>
                  <span>严重等级 {riskText[selectedProposal.severity as RiskLevel] ?? selectedProposal.severity}</span>
                </div>
              </div>
              <div className={styles.proposalGrid}>
                <div className={styles.operationBlock}>
                  <span className={styles.operationLabel}>审批边界</span>
                  <ul className={styles.actionList}>{(approvalActions.length ? approvalActions : ["当前没有附加审批边界。"]).map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
                <div className={styles.operationBlock}>
                  <span className={styles.operationLabel}>建议动作</span>
                  <ul className={styles.actionList}>{(recommendedActions.length ? recommendedActions : ["当前没有附加建议动作。"]).map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              </div>
              <div className={styles.roleRow}>
                <span className={styles.operationLabel}>允许角色</span>
                <div className={styles.roleTags}>{selectedProposal.required_operator_roles.map((role) => <span key={role}>{formatOperatorRole(role)}</span>)}</div>
              </div>
              {templatePreviews.length ? <div className={styles.templateList}>{templatePreviews.map((draft) => <article key={`${draft.audience}_${draft.channel}`} className={styles.templateCard}><strong>{draft.audience} / {draft.channel}</strong><p>{draft.content}</p></article>)}</div> : null}
              <div className={styles.operationBlock}>
                <label className={styles.operationLabel} htmlFor="operator-note">值班备注</label>
                <textarea id="operator-note" aria-label="operator-resolution-note" className={styles.operationNote} value={operatorNote} onChange={(event) => setOperatorNote(event.target.value)} disabled={isBusy || selectedProposal.status !== "pending"} rows={3} placeholder="记录值班安排、口头指令或审批原因。" />
              </div>
              {selectedProposal.status === "pending" ? (
                <div className={styles.operationActions}>
                  <button type="button" className={styles.secondaryButton} aria-label={`reject-proposal-${selectedProposal.proposal_id}`} disabled={isBusy || !canResolve} onClick={() => void onResolve(selectedProposal.proposal_id, "reject", operatorNote)}>驳回</button>
                  <button type="button" className={styles.primaryButton} aria-label={`approve-proposal-${selectedProposal.proposal_id}`} disabled={isBusy || !canResolve} onClick={() => void onResolve(selectedProposal.proposal_id, "approve", operatorNote)}>批准并联动执行</button>
                </div>
              ) : (
                <div className={styles.resolutionBanner}>
                  <strong>{proposalText[selectedProposal.status]}</strong>
                  <p>{selectedProposal.resolved_by ? `${formatOperatorActor(selectedProposal.resolved_by)} 于 ${formatTimestamp(selectedProposal.resolved_at)} 处理` : "已处理，但没有操作人元数据。"}</p>
                  <p>{selectedProposal.resolution_note || "没有记录值班备注。"}</p>
                </div>
              )}
              {drafts.length ? <div className={styles.templateList}>{drafts.map((draft) => <article key={draft.draft_id} className={styles.templateCard}><strong>{draft.audience} / {draft.channel}</strong><p>{draft.content}</p><small>{formatTimestamp(draft.created_at)}</small></article>)}</div> : null}
              {logs.length ? <div className={styles.executionList}>{logs.map((entry) => <article key={entry.log_id} className={styles.executionCard}><div className={styles.executionMeta}><strong>{formatRegionalActionType(entry.action_type)}</strong><span>{formatTimestamp(entry.created_at)}</span></div><p>{entry.summary}</p><small>{formatOperatorActor(entry.operator_id)}</small></article>)}</div> : null}
            </div>
          ) : null}
        </>
      ) : <p className={styles.emptyState}>当前没有待处理区域请示。你可以先发起问答，或生成一条行动建议来打开队列。</p>}

      {advisory && !proposals.length ? <AdvisoryCard advisory={advisory} /> : null}
    </div>
  );
}
