import { AnimatePresence, motion } from "framer-motion";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { DigitalTwinCesiumCanvas } from "./DigitalTwinCesiumCanvas";
import styles from "../styles/digital-twin-screen.module.css";
import type {
  ActionProposalV2,
  AgentDialogResponse,
  AgentDialogTranscriptEntry,
  FocusObjectView,
  RiskLevel,
  TwinOverviewView,
} from "../types/api";

type StreamStatus = "closed" | "connecting" | "open" | "error";

interface DigitalTwinImpactScreenProps {
  overview: TwinOverviewView | null;
  focusObject: FocusObjectView | null;
  pendingProposals: ActionProposalV2[];
  approvedProposals: ActionProposalV2[];
  dialogEntries: AgentDialogTranscriptEntry[];
  dialogOpen: boolean;
  dialogBusy: boolean;
  streamStatus: StreamStatus;
  onSelectObject: (objectId: string) => void | Promise<void>;
  onOpenDialog: () => void;
  onCloseDialog: () => void;
  onSendDialog: (message: string, objectId?: string) => void | Promise<void>;
  onGenerateProposals: () => void | Promise<void>;
  onGenerateWarnings: (proposalId: string) => void | Promise<void>;
  onResolveProposal: (proposalId: string, decision: "approve" | "reject", note: string) => void | Promise<void>;
  onOpenProposalQueue: () => void;
  onOpenOperations: () => void;
  actionBusy?: boolean;
  twinBusy?: boolean;
}

function riskClassName(riskLevel: RiskLevel) {
  return {
    None: styles.riskNone,
    Blue: styles.riskBlue,
    Yellow: styles.riskYellow,
    Orange: styles.riskOrange,
    Red: styles.riskRed,
  }[riskLevel];
}

function toneClassName(riskLevel: RiskLevel) {
  return {
    None: styles.toneNone,
    Blue: styles.toneBlue,
    Yellow: styles.toneYellow,
    Orange: styles.toneOrange,
    Red: styles.toneRed,
  }[riskLevel];
}

function streamStatusLabel(streamStatus: StreamStatus) {
  return {
    closed: "Offline",
    connecting: "Connecting",
    open: "Live",
    error: "Degraded",
  }[streamStatus];
}

function mapStateLabel(proposalState: string) {
  return {
    monitoring: "监测中",
    pending: "待审批方案",
    approved: "已批准动作",
    warning_generated: "预警已生成",
  }[proposalState] ?? proposalState;
}

function mapStateRank(proposalState: string) {
  return {
    warning_generated: 4,
    approved: 3,
    pending: 2,
    monitoring: 1,
  }[proposalState] ?? 0;
}

function formatTimestamp(timestamp: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function latestResponse(entries: AgentDialogTranscriptEntry[]): AgentDialogResponse | undefined {
  return [...entries].reverse().find((entry) => entry.response)?.response;
}

export function DigitalTwinImpactScreen({
  overview,
  focusObject,
  pendingProposals,
  approvedProposals,
  dialogEntries,
  dialogOpen,
  dialogBusy,
  streamStatus,
  onSelectObject,
  onOpenDialog,
  onCloseDialog,
  onSendDialog,
  onGenerateProposals,
  onGenerateWarnings,
  onResolveProposal,
  onOpenProposalQueue,
  onOpenOperations,
  actionBusy = false,
  twinBusy = false,
}: DigitalTwinImpactScreenProps) {
  const [draftMessage, setDraftMessage] = useState("");
  const [operatorNote, setOperatorNote] = useState("");
  const [selectedPendingProposalId, setSelectedPendingProposalId] = useState<string | null>(null);

  const latestApprovedProposal = approvedProposals[0] ?? null;
  const currentResponse = latestResponse(dialogEntries);
  const focusSignals = overview?.signals ?? [];
  const focusObjects = overview?.focus_objects ?? [];
  const mapLayers = overview?.map_layers ?? [];
  const recentWarningDrafts = (overview?.recent_warning_drafts ?? []).slice(0, 4);
  const selectedObjectId = focusObject?.object_id ?? overview?.lead_object_id ?? null;
  const selectedRiskLevel = focusObject?.risk_level ?? overview?.overall_risk_level ?? "None";
  const toneClass = toneClassName(selectedRiskLevel);
  const summaryText =
    overview?.summary ??
    "系统正在把事件态势、重点对象、智能体推理、审批 proposal 与分众 warning 收束到同一张数字孪生指挥主屏。";
  const recommendedActions = overview?.recommended_actions ?? [];
  const commandActions =
    (focusObject?.recommended_actions.length ? focusObject.recommended_actions : recommendedActions).slice(0, 4);
  const commandEvidence = (focusObject?.evidence ?? currentResponse?.evidence ?? []).slice(0, 3);
  const commandRiskReminders = (focusObject?.risk_reminders ?? currentResponse?.risk_reminders ?? []).slice(0, 3);
  const relatedProposalCount = focusObject?.related_proposals.length ?? 0;
  const dialogImpactSummary = (currentResponse?.impact_summary ?? focusObject?.risk_reasons ?? []).slice(0, 3);
  const focusObjectLookup = useMemo(
    () =>
      new Map(
        focusObjects.map((item) => [
          item.object_id,
          { village: item.village, timeToImpactMinutes: item.time_to_impact_minutes, summary: item.summary },
        ]),
      ),
    [focusObjects],
  );
  const spatialStatusItems = useMemo(
    () =>
      [...mapLayers]
        .sort((left, right) => mapStateRank(right.proposal_state) - mapStateRank(left.proposal_state))
        .slice(0, 5)
        .map((item) => ({
          ...item,
          label: mapStateLabel(item.proposal_state),
          village: focusObjectLookup.get(item.object_id)?.village ?? "未标注片区",
          timeToImpactMinutes: focusObjectLookup.get(item.object_id)?.timeToImpactMinutes,
          summary:
            focusObjectLookup.get(item.object_id)?.summary ??
            "当前对象正在与空间态势、proposal 状态和 warning 闭环联动。",
        })),
    [focusObjectLookup, mapLayers],
  );
  const relatedProposalIds = useMemo(
    () => new Set((focusObject?.related_proposals ?? []).map((item) => item.proposal.proposal_id)),
    [focusObject?.related_proposals],
  );
  const linkedPendingProposals = useMemo(() => {
    if (!focusObject?.object_id) {
      return pendingProposals.slice(0, 3);
    }

    const filtered = pendingProposals.filter(
      (proposal) =>
        proposal.entity_id === focusObject.object_id ||
        (proposal.high_risk_object_ids ?? []).includes(focusObject.object_id) ||
        relatedProposalIds.has(proposal.proposal_id),
    );
    return (filtered.length ? filtered : pendingProposals).slice(0, 3);
  }, [focusObject?.object_id, pendingProposals, relatedProposalIds]);
  const linkedApprovedProposal = useMemo(() => {
    if (!focusObject?.object_id) {
      return latestApprovedProposal;
    }
    return (
      approvedProposals.find(
        (proposal) =>
          proposal.entity_id === focusObject.object_id ||
          (proposal.high_risk_object_ids ?? []).includes(focusObject.object_id) ||
          relatedProposalIds.has(proposal.proposal_id),
      ) ?? latestApprovedProposal
    );
  }, [approvedProposals, focusObject?.object_id, latestApprovedProposal, relatedProposalIds]);

  const promptSuggestion = useMemo(() => {
    if (focusObject?.object_name) {
      return `请说明 ${focusObject.object_name} 的影响链、证据依据，以及指挥员下一步最应批准的动作。`;
    }
    return "请说明当前主事件的影响链、最危险对象，以及现在最值得批准的动作。";
  }, [focusObject?.object_name]);
  const followUpPrompts = (currentResponse?.follow_up_prompts ?? [promptSuggestion]).slice(0, 3);

  const selectedPendingProposal =
    linkedPendingProposals.find((proposal) => proposal.proposal_id === selectedPendingProposalId) ??
    linkedPendingProposals[0] ??
    null;

  const closureStatus = recentWarningDrafts.length
    ? "闭环已完成"
    : linkedApprovedProposal
      ? "待生成 warning"
      : linkedPendingProposals.length
        ? "待指挥员审批"
        : "待生成 proposal";
  const closureStatusDetail = recentWarningDrafts.length
    ? `${recentWarningDrafts.length} 条 audience warning 已完成闭环`
    : linkedApprovedProposal
      ? "已批准动作已经具备生成多受众 warning 的条件"
      : linkedPendingProposals.length
        ? `${linkedPendingProposals.length} 条 proposal 等待主屏内审批放行`
        : "先生成 proposal，再进入审批与 warning 微流程";

  const commandStatusItems = [
    {
      label: "影响到达",
      value: focusObject ? `${focusObject.time_to_impact_minutes} min` : "--",
      hint: focusObject?.village ?? "选择对象后显示到达窗口",
    },
    {
      label: "待审批队列",
      value: `${pendingProposals.length}`,
      hint: relatedProposalCount ? `${relatedProposalCount} 条与焦点对象相关` : "等待新的动作方案",
    },
    {
      label: "预警草稿",
      value: `${overview?.warning_draft_count ?? 0}`,
      hint: linkedApprovedProposal ? "已批准动作可直达 warning" : "尚未进入 warning 阶段",
    },
  ];

  const demoProofItems = [
    {
      label: "3D twin",
      value: mapLayers.length ? "Live" : "Fallback",
      detail: mapLayers.length ? `${mapLayers.length} 个空间对象已接 proposal 状态` : "3D 失效时仍保留列表和指挥面板",
    },
    {
      label: "Agent council",
      value: dialogEntries.some((entry) => entry.response) ? "Grounded" : "Ready",
      detail: commandEvidence.length ? `${commandEvidence.length} 条证据已入指挥语境` : "等待指挥员发起追问",
    },
    {
      label: "Human gate",
      value: pendingProposals.length ? `${pendingProposals.length} pending` : "Clear",
      detail: "高风险动作仍保留人工审批闸门",
    },
    {
      label: "Audience warnings",
      value: `${overview?.warning_draft_count ?? recentWarningDrafts.length}`,
      detail: "批准 proposal 后可生成多受众预警草稿",
    },
  ];

  const valueSummaryItems = [
    {
      label: "当前事件",
      value: overview?.event_title ?? "等待事件接入",
      detail: overview ? `${overview.active_alert_count} 条活动信号正在驱动主屏` : "接入事件后自动汇聚风险态势",
    },
    {
      label: "最危险对象",
      value: focusObject?.object_name ?? overview?.lead_object_name ?? "等待焦点对象",
      detail: focusObject
        ? `${focusObject.village} / ${focusObject.time_to_impact_minutes} 分钟到达窗口`
        : "对象选中后自动联动地图、指挥台和智能体对话",
    },
    {
      label: "建议动作",
      value: commandActions[0] ?? "等待推荐动作",
      detail: closureStatusDetail,
    },
  ];

  const degradationItems = [
    {
      label: "3D twin",
      status: mapLayers.length && !twinBusy ? "在线" : "降级",
      detail: mapLayers.length && !twinBusy ? "Cesium 主场景正常，可直接进行空间联动" : "自动切换为列表与指挥面板联动，不会白屏",
    },
    {
      label: "Realtime stream",
      status: streamStatusLabel(streamStatus),
      detail:
        streamStatus === "open"
          ? "SSE 正在持续刷新主屏、proposal 与 warning 状态"
          : streamStatus === "connecting"
            ? "正在恢复实时事件流"
            : streamStatus === "error"
              ? "进入降级态，保留最近一次稳定快照"
              : "当前使用静态快照，可继续演示主链路",
    },
    {
      label: "Agent response",
      status: dialogBusy ? "分析中" : commandEvidence.length ? "有依据" : "待追问",
      detail: dialogBusy ? "智能体正在生成结构化解释" : commandEvidence.length ? "当前回答已绑定证据与推荐动作" : "仍可通过建议追问快速拉起会商",
    },
    {
      label: "Command data",
      status: focusObjects.length ? "已加载" : "有限",
      detail: focusObjects.length ? "重点对象、proposal 和 warning 已可统一编排" : "缺少对象画像时仍展示事件级摘要与基础闭环",
    },
  ];
  const degradedCount = degradationItems.filter((item) => !["在线", "Live", "有依据", "已加载"].includes(item.status)).length;
  const degradationLead =
    degradedCount > 0 ? `当前有 ${degradedCount} 个子系统处于降级或恢复态` : "所有关键子系统均已准备好进行主链路演示";

  useEffect(() => {
    setSelectedPendingProposalId((current) => {
      if (current && linkedPendingProposals.some((proposal) => proposal.proposal_id === current)) {
        return current;
      }
      return linkedPendingProposals[0]?.proposal_id ?? null;
    });
  }, [linkedPendingProposals]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = draftMessage.trim();
    if (!value) {
      return;
    }
    void onSendDialog(value, selectedObjectId ?? undefined);
    setDraftMessage("");
  }

  return (
    <section className={styles.screen}>
      <motion.section
        className={`${styles.hero} ${toneClass}`}
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className={styles.heroInner}>
          <div>
            <p className={styles.eyebrow}>Digital Twin Agent Command Screen</p>
            <h2 className={styles.heroTitle}>数字孪生智能体洪水预警主屏</h2>
            <p className={styles.heroSummary}>{summaryText}</p>
            <div className={styles.heroActions}>
              <button type="button" className={styles.heroButtonPrimary} onClick={onOpenDialog}>
                打开智能体控制台
              </button>
              <button type="button" className={styles.heroButtonSecondary} onClick={() => void onGenerateProposals()}>
                生成 proposal
              </button>
              <button type="button" className={styles.heroButtonSecondary} onClick={onOpenOperations}>
                打开行动处置台
              </button>
            </div>
          </div>

          <div className={styles.heroStats}>
            <article className={styles.heroStat}>
              <span>Current event</span>
              <strong>{overview?.event_title ?? "Waiting for event"}</strong>
            </article>
            <article className={styles.heroStat}>
              <span>Overall risk</span>
              <strong className={overview ? riskClassName(overview.overall_risk_level) : undefined}>
                {overview?.overall_risk_level ?? "None"}
              </strong>
            </article>
            <article className={styles.heroSignal}>
              <span>Live chain</span>
              <p>
                实时链路 {streamStatusLabel(streamStatus)}。待审批 {overview?.pending_proposal_count ?? pendingProposals.length}，
                已批准 {overview?.approved_proposal_count ?? approvedProposals.length}，warning{" "}
                {overview?.warning_draft_count ?? 0}。
              </p>
            </article>
          </div>
        </div>
      </motion.section>

      <motion.section
        className={`${styles.valueSummaryStrip} ${toneClass}`}
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.38, delay: 0.04, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className={styles.valueSummaryIntro}>
          <span>Client value summary</span>
          <strong>给甲方的第一屏价值摘要</strong>
          <p>先讲清事件、最危险对象和建议动作，再进入空间联动、会商与闭环展示。</p>
        </div>
        <div className={styles.valueSummaryGrid}>
          {valueSummaryItems.map((item) => (
            <article key={item.label} className={styles.valueSummaryItem}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <small>{item.detail}</small>
            </article>
          ))}
        </div>
      </motion.section>

      <motion.section
        className={styles.demoProofStrip}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.38, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
        aria-label="production-demo-proof-strip"
      >
        <div className={styles.demoProofIntro}>
          <span>Production demo mode</span>
          <strong>面向甲方展示的端到端闭环</strong>
        </div>
        <div className={styles.demoProofGrid}>
          {demoProofItems.map((item) => (
            <article key={item.label} className={styles.demoProofItem}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
              <small>{item.detail}</small>
            </article>
          ))}
        </div>
      </motion.section>

      <section className={styles.degradationBanner}>
        <div className={styles.degradationLead}>
          <span>Graceful degradation</span>
          <strong>{degradationLead}</strong>
          <p>即使 3D、实时流或智能体处于恢复态，主屏也会保留稳定快照、对象列表、审批微流程和 warning 出口。</p>
        </div>
        <div className={styles.degradationList}>
          {degradationItems.map((item) => (
            <article key={item.label} className={styles.valueSummaryItem}>
              <span>{item.label}</span>
              <strong>{item.status}</strong>
              <small>{item.detail}</small>
            </article>
          ))}
        </div>
      </section>

      <div className={styles.mainGrid}>
        <aside className={styles.rail}>
          <section className={styles.railSection}>
            <div className={styles.railHeader}>
              <div>
                <span>Situation rail</span>
                <h3>重点对象</h3>
              </div>
              <span className={styles.streamPill}>{streamStatusLabel(streamStatus)}</span>
            </div>
            <div className={styles.summaryBlock}>
              <p>左侧对象带与实时信号带保持同一事件上下文，选中对象后会同步驱动地图聚焦、右侧指挥台和智能体对话控制台。</p>
            </div>
            <div className={styles.objectList}>
              {focusObjects.length ? (
                focusObjects.map((item) => (
                  <button
                    key={item.object_id}
                    type="button"
                    className={`${styles.objectChip} ${toneClassName(item.risk_level)} ${
                      selectedObjectId === item.object_id ? styles.objectChipActive : ""
                    }`}
                    onClick={() => void onSelectObject(item.object_id)}
                  >
                    <div className={styles.objectChipHead}>
                      <strong>{item.name}</strong>
                      <span className={`${styles.nodePill} ${riskClassName(item.risk_level)}`}>{item.risk_level}</span>
                    </div>
                    <div className={styles.objectChipMeta}>
                      {item.village} / {item.time_to_impact_minutes} min / {item.pending_proposal_ids.length} pending
                    </div>
                  </button>
                ))
              ) : (
                <p className={styles.objectSummary}>当前还没有重点对象。事件接入后，系统会自动把最需要关注的对象拉进这条态势带。</p>
              )}
            </div>
          </section>

          <section className={styles.railSection}>
            <div className={styles.sectionTitle}>
              <div>
                <span>Signal rail</span>
                <h4>态势信号</h4>
              </div>
            </div>
            <div className={styles.signalList}>
              {focusSignals.length ? (
                focusSignals.map((item) => (
                  <article key={item.signal_id} className={styles.signalItem}>
                    <div className={styles.signalItemMeta}>
                      <span>{item.severity}</span>
                      <span>{formatTimestamp(item.created_at)}</span>
                    </div>
                    <strong>{item.title}</strong>
                    <p>{item.detail}</p>
                  </article>
                ))
              ) : (
                <p className={styles.objectSummary}>当前没有新的实时信号。最新告警、道路变化和资源反馈会优先出现在这里。</p>
              )}
            </div>
          </section>
        </aside>

        <section className={styles.canvasPanel}>
          <div className={styles.canvasHeader}>
            <div>
              <span className={styles.eyebrow}>Twin Canvas</span>
              <h3 className={styles.canvasTitle}>City flood digital twin</h3>
              <p>中央画布使用真实 Cesium 场景承接对象选择、风险聚焦与 proposal 状态，不再停留在占位式主屏。</p>
            </div>
            <div className={styles.canvasStatus}>
              <span className={styles.statusPill}>Trend / {overview?.trend ?? "stable"}</span>
              <span className={styles.statusPill}>Focus / {overview?.focus_objects.length ?? 0}</span>
              <span className={styles.statusPill}>Signals / {overview?.active_alert_count ?? focusSignals.length}</span>
            </div>
          </div>

          <DigitalTwinCesiumCanvas
            eventTitle={overview?.event_title}
            layers={overview?.map_layers ?? []}
            selectedObjectId={selectedObjectId}
            selectedRiskLevel={selectedRiskLevel}
            onSelectObject={(objectId) => void onSelectObject(objectId)}
          />

          <div className={styles.canvasFooter}>
            <article className={`${styles.footerCard} ${toneClass}`}>
              <h4>Commander recommendations</h4>
              {recommendedActions.length ? (
                <ul>
                  {recommendedActions.map((item, index) => (
                    <li key={`${item}-${index}`}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p>系统仍在汇聚证据。推荐动作会在对象影响链和会商结果变得稳定后出现在这里。</p>
              )}
            </article>
            <article className={styles.footerCard}>
              <h4>How to operate this screen</h4>
              <p>先选对象，再打开智能体控制台追问；随后在右侧指挥台完成 proposal 审批与 warning 生成，尽量不跳出主屏。</p>
            </article>
            <article className={`${styles.footerCard} ${toneClass}`}>
              <h4>Spatial closure layer</h4>
              <div className={styles.spatialLedger}>
                {spatialStatusItems.length ? (
                  spatialStatusItems.map((item) => (
                    <button
                      key={item.object_id}
                      type="button"
                      className={`${styles.spatialRow} ${toneClassName(item.risk_level)} ${
                        selectedObjectId === item.object_id ? styles.spatialRowActive : ""
                      }`}
                      onClick={() => void onSelectObject(item.object_id)}
                    >
                      <div>
                        <strong>{item.name}</strong>
                        <p>
                          {item.village}
                          {item.timeToImpactMinutes !== undefined ? ` / ${item.timeToImpactMinutes} min` : ""}
                        </p>
                      </div>
                      <span className={styles.mapStateBadge}>{item.label}</span>
                    </button>
                  ))
                ) : (
                  <p>空间闭环层仍在初始化。稍后会把 proposal 与 warning 状态同步到地图对象与主屏底部台账。</p>
                )}
              </div>
            </article>
          </div>
        </section>

        <aside className={`${styles.detailRail} ${toneClass}`}>
          <section className={styles.railSection}>
            <div className={styles.sectionTitle}>
              <div>
                <span>Command deck</span>
                <h3 className={styles.objectTitle}>{focusObject?.object_name ?? "等待焦点对象"}</h3>
              </div>
              {focusObject ? (
                <span className={`${styles.statusPill} ${riskClassName(focusObject.risk_level)}`}>{focusObject.risk_level}</span>
              ) : null}
            </div>
            <p className={styles.objectSummary}>
              {focusObject?.summary ?? "选中一个重点对象后，这里会解锁影响链、证据阶梯、审批微流程和 warning 入口。"}
            </p>
            <div className={styles.commandGrid}>
              {commandStatusItems.map((item) => (
                <article key={item.label} className={`${styles.commandCard} ${toneClass}`}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                  <small>{item.hint}</small>
                </article>
              ))}
            </div>
            {focusObject ? (
              <div className={styles.objectMeta}>
                <div>
                  <span>Location</span>
                  <strong>{focusObject.village}</strong>
                </div>
                <div>
                  <span>Impact window</span>
                  <strong>{focusObject.time_to_impact_minutes} min</strong>
                </div>
                <div>
                  <span>Risk reminders</span>
                  <p>{commandRiskReminders.join(" / ") || "当前没有额外提醒。"}</p>
                </div>
              </div>
            ) : null}
            <div className={styles.detailRailActions}>
              <button type="button" className={styles.railButton} onClick={onOpenDialog}>
                智能体控制台
              </button>
              <button type="button" className={styles.railButton} onClick={() => void onGenerateProposals()}>
                生成 proposal
              </button>
              <button type="button" className={styles.railButton} onClick={onOpenOperations}>
                行动处置页
              </button>
            </div>
          </section>

          <section className={styles.railSection}>
            <div className={styles.sectionTitle}>
              <div>
                <span>Action matrix</span>
                <h4>推荐动作矩阵</h4>
              </div>
            </div>
            <div className={styles.actionMatrix}>
              {commandActions.length ? (
                commandActions.map((action, index) => (
                  <article key={`${action}-${index}`} className={`${styles.actionCard} ${toneClass}`}>
                    <div className={styles.warningMeta}>
                      <span>Action {index + 1}</span>
                      <span>{focusObject?.risk_level ?? overview?.overall_risk_level ?? "None"}</span>
                    </div>
                    <strong>{action}</strong>
                    <p>{focusObject?.risk_reasons[index] ?? summaryText}</p>
                  </article>
                ))
              ) : (
                <p className={styles.objectSummary}>当前还没有足够的会商结果来形成动作矩阵。继续追问或生成 proposal 后，这里会自动补齐。</p>
              )}
            </div>
          </section>

          <section className={styles.railSection}>
            <div className={styles.sectionTitle}>
              <div>
                <span>Evidence ladder</span>
                <h4>当前证据与触发条件</h4>
              </div>
            </div>
            <div className={styles.warningList}>
              {commandEvidence.length ? (
                commandEvidence.map((item) => (
                  <article key={item.source_id} className={styles.warningCard}>
                    <div className={styles.warningMeta}>
                      <span>{item.evidence_type}</span>
                      <span>{item.timestamp ? formatTimestamp(item.timestamp) : "evidence"}</span>
                    </div>
                    <strong>{item.title}</strong>
                    <p>{item.excerpt}</p>
                  </article>
                ))
              ) : (
                <p className={styles.objectSummary}>当前还没有更多证据。可以继续向智能体追问影响链、下游后果或推荐动作依据。</p>
              )}
            </div>
          </section>

          <section className={styles.railSection}>
            <div className={styles.sectionTitle}>
              <div>
                <span>Closure</span>
                <h4>Proposal 与 warning 闭环</h4>
              </div>
            </div>
            <article className={`${styles.closureSummaryCard} ${toneClass}`}>
              <span>Closure state</span>
              <strong>{closureStatus}</strong>
              <small>{closureStatusDetail}</small>
            </article>
            <div className={styles.microFlow}>
              <div className={styles.microFlowStage}>
                <div className={styles.microFlowHeader}>
                  <span>Stage 1</span>
                  <strong>待审批 proposal</strong>
                </div>
                {linkedPendingProposals.length ? (
                  <div className={styles.microProposalList}>
                    {linkedPendingProposals.map((proposal) => (
                      <button
                        key={proposal.proposal_id}
                        type="button"
                        className={`${styles.microProposalChip} ${
                          selectedPendingProposal?.proposal_id === proposal.proposal_id ? styles.microProposalChipActive : ""
                        }`}
                        onClick={() => setSelectedPendingProposalId(proposal.proposal_id)}
                      >
                        <strong>{proposal.title}</strong>
                        <small>{proposal.execution_mode ?? proposal.action_type ?? "proposal"}</small>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className={styles.objectSummary}>当前焦点对象没有待审批 proposal，你可以直接查看已批准动作和 warning 产物。</p>
                )}
                {selectedPendingProposal ? (
                  <div className={styles.microProposalDetail}>
                    <article className={styles.warningCard}>
                      <div className={styles.warningMeta}>
                        <span>pending</span>
                        <span>{selectedPendingProposal.execution_mode ?? "proposal"}</span>
                      </div>
                      <strong>{selectedPendingProposal.title}</strong>
                      <p>{selectedPendingProposal.summary}</p>
                    </article>
                    <textarea
                      className={styles.microNoteInput}
                      value={operatorNote}
                      onChange={(event) => setOperatorNote(event.target.value)}
                      placeholder="补充审批意见，例如：先行疏散学校北侧低洼区，保持消防通道畅通。"
                    />
                    <div className={styles.microFlowActions}>
                      <button
                        type="button"
                        className={styles.railButton}
                        disabled={actionBusy}
                        onClick={() => void onResolveProposal(selectedPendingProposal.proposal_id, "reject", operatorNote)}
                      >
                        驳回
                      </button>
                      <button
                        type="button"
                        className={styles.heroButtonPrimary}
                        disabled={actionBusy}
                        onClick={() => void onResolveProposal(selectedPendingProposal.proposal_id, "approve", operatorNote)}
                      >
                        批准动作
                      </button>
                    </div>
                  </div>
                ) : null}
              </div>

              <div className={styles.microFlowStage}>
                <div className={styles.microFlowHeader}>
                  <span>Stage 2</span>
                  <strong>warning 生成</strong>
                </div>
                {linkedApprovedProposal ? (
                  <div className={styles.microProposalDetail}>
                    <article className={styles.warningCard}>
                      <div className={styles.warningMeta}>
                        <span>approved</span>
                        <span>{linkedApprovedProposal.execution_mode}</span>
                      </div>
                      <strong>{linkedApprovedProposal.title}</strong>
                      <p>{linkedApprovedProposal.summary}</p>
                    </article>
                    <div className={styles.microFlowActions}>
                      <button
                        type="button"
                        className={styles.railButton}
                        disabled={actionBusy}
                        onClick={() => void onGenerateWarnings(linkedApprovedProposal.proposal_id)}
                      >
                        生成 audience warnings
                      </button>
                      <button type="button" className={styles.railButton} onClick={onOpenProposalQueue}>
                        打开完整队列
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className={styles.objectSummary}>还没有已批准动作。先在上一步完成审批，再在这里生成多受众 warning。</p>
                )}
                {recentWarningDrafts.length ? (
                  <div className={styles.warningList}>
                    {recentWarningDrafts.map((draft) => (
                      <article key={draft.warning_id} className={styles.warningCard}>
                        <div className={styles.warningMeta}>
                          <span>{draft.audience}</span>
                          <span>{draft.channel}</span>
                        </div>
                        <strong>{mapStateLabel("warning_generated")}</strong>
                        <p>{draft.grounding_summary || draft.content}</p>
                      </article>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
          </section>
        </aside>
      </div>

      <AnimatePresence>
        {dialogOpen ? (
          <motion.div
            className={styles.dialogOverlay}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.aside
              className={styles.dialogPanel}
              initial={{ x: 60, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 60, opacity: 0 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className={styles.dialogHeader}>
                <div>
                  <span>Agent Dialog Control</span>
                  <h3>Command conversation desk</h3>
                  <p className={styles.dialogSubhead}>
                    用这个侧边控制台追问影响链、检查证据，并直接进入 proposal 与 warning 流程，不必离开数字孪生主屏。
                  </p>
                </div>
                <button type="button" className={styles.dialogClose} onClick={onCloseDialog} aria-label="close-dialog">
                  x
                </button>
              </div>

              <div className={styles.dialogBrief}>
                <article className={`${styles.dialogBriefCard} ${toneClass}`}>
                  <span>Focus</span>
                  <strong>{focusObject?.object_name ?? overview?.lead_object_name ?? "No focus object selected"}</strong>
                  <small>{focusObject?.summary ?? "先选中一个对象，再把上下文注入到对话控制台。"}</small>
                </article>
                <article className={styles.dialogBriefCard}>
                  <span>Evidence</span>
                  <strong>{commandEvidence.length}</strong>
                  <small>{currentResponse?.grounding_summary ?? "等待新的 grounded response。"}</small>
                </article>
                <article className={styles.dialogBriefCard}>
                  <span>Queue</span>
                  <strong>{pendingProposals.length}</strong>
                  <small>{latestApprovedProposal ? "当前已有已批准动作。" : "当前还没有已批准动作。"}</small>
                </article>
              </div>

              <div className={styles.dialogWorkspace}>
                <div className={styles.dialogMessageList}>
                  {dialogEntries.map((entry) => (
                    <article
                      key={entry.id}
                      className={`${styles.dialogMessage} ${entry.role === "assistant" ? styles.dialogMessageAssistant : ""}`}
                    >
                      <div className={styles.dialogMeta}>
                        <span>{entry.role === "assistant" ? "assistant" : "operator"}</span>
                        <span>{formatTimestamp(entry.created_at)}</span>
                      </div>
                      <div className={styles.dialogBody}>
                        <strong>{entry.role === "assistant" ? "Agent response" : "Commander prompt"}</strong>
                        <p>{entry.content}</p>
                        {entry.response ? (
                          <>
                            {entry.response.impact_summary.length ? (
                              <div className={styles.dialogEvidence}>
                                <strong>Impact summary</strong>
                                <p>{entry.response.impact_summary.join(" / ")}</p>
                              </div>
                            ) : null}
                            {entry.response.evidence.slice(0, 2).map((item) => (
                              <div key={item.source_id} className={styles.dialogEvidence}>
                                <div className={styles.warningMeta}>
                                  <span>{item.evidence_type}</span>
                                  <span>{item.timestamp ? formatTimestamp(item.timestamp) : "evidence"}</span>
                                </div>
                                <strong>{item.title}</strong>
                                <p>{item.excerpt}</p>
                              </div>
                            ))}
                            {entry.response.recommended_actions.length ? (
                              <div className={styles.dialogEvidence}>
                                <strong>Recommended moves</strong>
                                <div className={styles.recommendationList}>
                                  {entry.response.recommended_actions.map((item, index) => (
                                    <p key={`${item}-${index}`}>{item}</p>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                            {entry.response.proposal_entry?.proposal ? (
                              <button type="button" className={styles.dialogButton} onClick={onOpenProposalQueue}>
                                打开 proposal 队列
                              </button>
                            ) : null}
                          </>
                        ) : null}
                      </div>
                    </article>
                  ))}
                </div>

                <aside className={styles.dialogIntel}>
                  <section className={`${styles.dialogIntelCard} ${toneClass}`}>
                    <span>Impact chain</span>
                    <strong>当前推理快照</strong>
                    {dialogImpactSummary.length ? (
                      <div className={styles.followUpList}>
                        {dialogImpactSummary.map((item, index) => (
                          <p key={`${item}-${index}`}>{item}</p>
                        ))}
                      </div>
                    ) : (
                      <p>继续追问影响顺序、下游后果或 proposal 依据，系统会把结构化结果回写到这里。</p>
                    )}
                  </section>

                  <section className={styles.dialogIntelCard}>
                    <span>Follow-up prompts</span>
                    <strong>建议追问</strong>
                    <div className={styles.promptStack}>
                      {followUpPrompts.length ? (
                        followUpPrompts.map((item, index) => (
                          <button key={`${item}-${index}`} type="button" className={styles.promptChip} onClick={() => setDraftMessage(item)}>
                            {item}
                          </button>
                        ))
                      ) : (
                        <button type="button" className={styles.promptChip} onClick={() => setDraftMessage(promptSuggestion)}>
                          {promptSuggestion}
                        </button>
                      )}
                    </div>
                  </section>

                  <section className={styles.dialogIntelCard}>
                    <span>Command moves</span>
                    <strong>Proposal 与闭环路径</strong>
                    <div className={styles.followUpList}>
                      {commandActions.length ? (
                        commandActions.map((item, index) => <p key={`${item}-${index}`}>{item}</p>)
                      ) : (
                        <p>先生成 proposal，再进入审批与 warning 流程。</p>
                      )}
                    </div>
                  </section>
                </aside>
              </div>

              <form className={styles.dialogComposer} onSubmit={handleSubmit}>
                <textarea
                  aria-label="agent-dialog-input"
                  value={draftMessage}
                  onChange={(event) => setDraftMessage(event.target.value)}
                  placeholder={promptSuggestion}
                />
                <div className={styles.dialogComposerActions}>
                  <div className={styles.dialogComposerRail}>
                    <button type="button" className={styles.dialogButton} onClick={() => setDraftMessage(promptSuggestion)}>
                      使用建议追问
                    </button>
                    <button type="button" className={styles.dialogButton} onClick={onOpenProposalQueue}>
                      Proposal 队列
                    </button>
                  </div>
                  <button type="submit" className={styles.heroButtonPrimary} disabled={dialogBusy}>
                    {dialogBusy ? "Analyzing..." : "Send to agent"}
                  </button>
                </div>
              </form>
            </motion.aside>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </section>
  );
}
