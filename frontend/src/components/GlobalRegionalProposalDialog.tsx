import { useEffect, useMemo, useState } from "react";
import styles from "../App.module.css";
import {
  formatActionScopeFieldLabel,
  formatRegionalActionDisplayCategory,
  formatRegionalActionDisplayName,
  formatRegionalActionDisplayTagline,
  formatExecutionMode,
  formatGenerationSource,
  riskLevelText,
} from "../lib/displayText";
import type { RegionalProposalQueueSnapshot } from "../types/api";

interface GlobalRegionalProposalDialogProps {
  open: boolean;
  busy: boolean;
  snapshot: RegionalProposalQueueSnapshot | null;
  onApprove: (proposalId: string, note: string) => Promise<void>;
  onReject: (proposalId: string, note: string) => Promise<void>;
  onSaveDraft: (proposalId: string, actionScope: Record<string, unknown>) => Promise<void>;
  onSnooze: () => void;
}

function normalizeFieldValue(value: unknown) {
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  if (typeof value === "number") {
    return String(value);
  }
  if (typeof value === "string") {
    return value;
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return "";
}

function parseFieldValue(source: string, original: unknown) {
  if (Array.isArray(original)) {
    return source
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (typeof original === "number") {
    const parsed = Number(source);
    return Number.isFinite(parsed) ? parsed : original;
  }
  return source;
}

export function GlobalRegionalProposalDialog({
  open,
  busy,
  snapshot,
  onApprove,
  onReject,
  onSaveDraft,
  onSnooze,
}: GlobalRegionalProposalDialogProps) {
  const [drafts, setDrafts] = useState<Record<string, Record<string, string>>>({});

  useEffect(() => {
    const nextDrafts: Record<string, Record<string, string>> = {};
    for (const item of snapshot?.items ?? []) {
      const actionScope = item.proposal.action_scope ?? {};
      nextDrafts[item.proposal.proposal_id] = Object.fromEntries(
        Object.entries(actionScope).map(([key, value]) => [key, normalizeFieldValue(value)]),
      );
    }
    setDrafts(nextDrafts);
  }, [snapshot?.queue_version]);

  const items = useMemo(() => snapshot?.items ?? [], [snapshot]);

  if (!open || !items.length) {
    return null;
  }

  return (
    <div
      aria-label="global-regional-proposal-dialog"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(8, 15, 28, 0.72)",
        backdropFilter: "blur(8px)",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "24px",
      }}
    >
      <div
        className={styles.panelFrame}
        style={{
          width: "min(1430px, calc(100vw - 48px))",
          maxHeight: "min(88vh, 960px)",
          overflow: "auto",
          padding: "20px",
          borderWidth: "1px",
          boxShadow: "0 24px 80px rgba(5, 12, 26, 0.45)",
        }}
      >
        <div className={styles.panelHeaderCompact}>
          <div>
            <p className={styles.sectionLabel}>智能体主动请示</p>
            <h3>以下区域动作由模型生成，需由指挥席逐条确认或驳回。</h3>
          </div>
          <div className={styles.operationActions}>
            <button
              type="button"
              className={styles.secondaryButton}
              aria-label="regional-proposal-snooze"
              disabled={busy}
              onClick={onSnooze}
            >
              稍后处理
            </button>
          </div>
        </div>

        <div style={{ display: "grid", gap: "16px", marginTop: "16px" }}>
          {items.map((item) => {
            const proposal = item.proposal;
            const proposalDraft = drafts[proposal.proposal_id] ?? {};
            const scopeEntries = Object.entries(proposal.action_scope ?? {});
            const displayName = formatRegionalActionDisplayName(proposal);
            const displayTagline = formatRegionalActionDisplayTagline(proposal);
            const displayCategory = formatRegionalActionDisplayCategory(proposal);
            return (
              <article key={proposal.proposal_id} className={styles.panelFrame} style={{ padding: "18px" }}>
                <div className={styles.panelHeaderCompact}>
                  <div>
                    <p className={styles.sectionLabel}>{item.event_title}</p>
                    <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap", marginBottom: "8px" }}>
                      <span className={styles.statusPill}>{displayCategory}</span>
                    </div>
                    <h3>{displayName}</h3>
                    <p className={styles.emptyState} style={{ marginTop: "6px" }}>{displayTagline}</p>
                  </div>
                  <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
                    <span
                      className={`${styles.riskBadge} ${styles[`risk${item.current_risk_level}` as keyof typeof styles] ?? styles.riskOrange}`}
                    >
                      {riskLevelText[item.current_risk_level]}
                    </span>
                    <span className={styles.statusPill}>{formatGenerationSource(proposal.generation_source)}</span>
                    <span className={styles.statusPill}>{formatExecutionMode(proposal.execution_mode)}</span>
                    {proposal.edited_by_commander ? <span className={styles.statusPill}>已编辑草稿</span> : null}
                    {proposal.has_new_system_suggestion ? <span className={styles.statusPill}>模型有新建议</span> : null}
                  </div>
                </div>

                <div className={styles.routeSummary}>
                  <div>
                    <span>触发原因</span>
                    <strong>{proposal.trigger_reason || "模型根据当前区域风险生成了新的请示。"}</strong>
                  </div>
                  <div>
                    <span>关联对象</span>
                    <strong>{item.high_risk_object_names.join("、") || "暂无"}</strong>
                  </div>
                  <div>
                    <span>模型</span>
                    <strong>{proposal.model_name || "未标记"}</strong>
                  </div>
                </div>

                {proposal.title && proposal.title !== displayName ? (
                  <p className={styles.emptyState} style={{ marginTop: "10px" }}>
                    系统标题：{proposal.title}
                  </p>
                ) : null}

                <ul className={styles.reasonList}>
                  <li>{proposal.recommendation || proposal.summary}</li>
                  <li>{proposal.evidence_summary || "暂无证据摘要"}</li>
                  {proposal.grounding_summary ? <li>{proposal.grounding_summary}</li> : null}
                </ul>

                {scopeEntries.length ? (
                  <div style={{ display: "grid", gap: "12px", marginTop: "12px" }}>
                    {scopeEntries.map(([key, value]) => (
                      <label key={key} style={{ display: "grid", gap: "6px" }}>
                        <span className={styles.sectionLabel}>{formatActionScopeFieldLabel(key)}</span>
                        <input
                          aria-label={key}
                          value={proposalDraft[key] ?? normalizeFieldValue(value)}
                          onChange={(event) =>
                            setDrafts((current) => ({
                              ...current,
                              [proposal.proposal_id]: {
                                ...current[proposal.proposal_id],
                                [key]: event.target.value,
                              },
                            }))
                          }
                          style={{
                            width: "100%",
                            borderRadius: "12px",
                            border: "1px solid rgba(148, 163, 184, 0.22)",
                            background: "rgba(255, 255, 255, 0.04)",
                            color: "inherit",
                            padding: "10px 12px",
                          }}
                        />
                      </label>
                    ))}
                  </div>
                ) : null}

                <div className={styles.operationActions} style={{ marginTop: "16px" }}>
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    aria-label={`save-regional-proposal-draft-${proposal.proposal_id}`}
                    disabled={busy}
                    onClick={() =>
                      onSaveDraft(
                        proposal.proposal_id,
                        Object.fromEntries(
                          scopeEntries.map(([key, value]) => [
                            key,
                            parseFieldValue(proposalDraft[key] ?? normalizeFieldValue(value), value),
                          ]),
                        ),
                      )
                    }
                  >
                    保存草稿
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    aria-label={`reject-regional-proposal-${proposal.proposal_id}`}
                    disabled={busy}
                    onClick={() => void onReject(proposal.proposal_id, "")}
                  >
                    驳回
                  </button>
                  <button
                    type="button"
                    className={styles.primaryButton}
                    aria-label={`approve-regional-proposal-${proposal.proposal_id}`}
                    disabled={busy}
                    onClick={() => void onApprove(proposal.proposal_id, "")}
                  >
                    确认执行
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}
