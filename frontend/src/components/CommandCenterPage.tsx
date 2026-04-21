import { motion } from "framer-motion";
import { FormEvent, KeyboardEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { entityText, riskText } from "../config/consoleConfig";
import { agentText, normalizeAgentTerminology } from "../lib/agentUiText";
import { formatTimestamp } from "../lib/consoleFormatting";
import styles from "../styles/command-center.module.css";
import type {
  ActionProposalV2,
  AgentStatusView,
  AgentTask,
  DailyReportView,
  EntityImpactView,
  EventEpisodeSummaryView,
  RegionalAnalysisPackageView,
  StructuredAnswer,
  V2CopilotMessage,
} from "../types/api";
import { CopilotMessageBubble } from "./CopilotMessageBubble";
import { PriorityObjectItem, PriorityObjectPanel } from "./PriorityObjectPanel";

interface CommandCenterPageProps {
  agentStatus: AgentStatusView | null;
  agentTasks: AgentTask[];
  dailyReports: DailyReportView[];
  episodeSummaries: EventEpisodeSummaryView[];
  input: string;
  isBusy: boolean;
  latestAnswer: StructuredAnswer | null;
  messages: V2CopilotMessage[];
  pendingRegionalAnalysisPackage: RegionalAnalysisPackageView | null;
  regionalAnalysisPackageHistory: RegionalAnalysisPackageView[];
  pendingProposals: ActionProposalV2[];
  priorityItems: PriorityObjectItem[];
  quickPrompts: string[];
  selectedImpact: EntityImpactView | null;
  selectedPriorityId: string | null;
  onChangeInput: (value: string) => void;
  onOpenOperations: () => void;
  onPrompt: (prompt: string) => void;
  onResolveRegionalAnalysisPackage: (packageId: string, decision: "approve" | "reject", note: string) => void;
  onResolveProposal: (proposalId: string, decision: "approve" | "reject", note: string) => void;
  onSelectPriority: (id: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onTextareaKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}

type FeedItem =
  | {
      kind: "message";
      key: string;
      createdAt: string;
      message: V2CopilotMessage;
    }
  | {
      kind: "task";
      key: string;
      createdAt: string;
      task: AgentTask;
    }
  | {
      kind: "guide";
      key: string;
      createdAt: string;
      prompt: string;
      title: string;
      summary: string;
    }
  | {
      kind: "regional_analysis_package";
      key: string;
      createdAt: string;
      item: RegionalAnalysisPackageView;
    }
  | {
      kind: "daily_report";
      key: string;
      createdAt: string;
      item: DailyReportView;
    }
  | {
      kind: "episode_summary";
      key: string;
      createdAt: string;
      item: EventEpisodeSummaryView;
    };

export function CommandCenterPage({
  agentStatus,
  agentTasks,
  dailyReports,
  episodeSummaries,
  input,
  isBusy,
  latestAnswer,
  messages,
  pendingRegionalAnalysisPackage,
  regionalAnalysisPackageHistory,
  pendingProposals,
  priorityItems,
  quickPrompts,
  selectedImpact,
  selectedPriorityId,
  onChangeInput,
  onOpenOperations,
  onPrompt,
  onResolveRegionalAnalysisPackage,
  onResolveProposal,
  onSelectPriority,
  onSubmit,
  onTextareaKeyDown,
}: CommandCenterPageProps) {
  const streamEndRef = useRef<HTMLDivElement | null>(null);
  const [selectedProposalId, setSelectedProposalId] = useState<string | null>(null);
  const liveTasks = useMemo(
    () =>
      [...agentTasks]
        .sort((left, right) => compareCreatedAt(left.created_at, right.created_at))
        .slice(-6),
    [agentTasks],
  );

  const selectedProposal = useMemo(
    () => pendingProposals.find((proposal) => proposal.proposal_id === selectedProposalId) ?? null,
    [pendingProposals, selectedProposalId],
  );
  const proposalGuidePrompt = useMemo(
    () => (selectedProposal ? getProposalFollowUpPrompt(selectedProposal) : null),
    [selectedProposal],
  );
  const activeRegionalAnalysisPackage = pendingRegionalAnalysisPackage ?? regionalAnalysisPackageHistory[0] ?? null;

  const feedItems = useMemo<FeedItem[]>(
    () =>
      [
        ...messages.map<FeedItem>((message) => ({
          kind: "message",
          key: message.message_id,
          createdAt: message.created_at,
          message,
        })),
        ...liveTasks.map<FeedItem>((task) => ({
          kind: "task",
          key: task.task_id,
          createdAt: task.completed_at ?? task.started_at ?? task.created_at,
          task,
        })),
        ...(selectedProposal && proposalGuidePrompt
          ? [
              {
                kind: "guide" as const,
                key: `guide_${selectedProposal.proposal_id}`,
                createdAt: selectedProposal.updated_at ?? selectedProposal.created_at,
                prompt: proposalGuidePrompt,
                title: "系统引导",
                summary: `已切换到事务“${selectedProposal.title}”，可以直接围绕它继续追问。`,
              },
            ]
          : []),
        ...(activeRegionalAnalysisPackage
          ? [
              {
                kind: "regional_analysis_package" as const,
                key: `regional_analysis_package_${activeRegionalAnalysisPackage.package_id}`,
                createdAt: activeRegionalAnalysisPackage.updated_at ?? activeRegionalAnalysisPackage.created_at,
                item: activeRegionalAnalysisPackage,
              },
            ]
          : []),
        ...dailyReports.map<FeedItem>((report) => ({
          kind: "daily_report",
          key: `daily_report_${report.report_id}`,
          createdAt: report.created_at,
          item: report,
        })),
        ...episodeSummaries.map<FeedItem>((summary) => ({
          kind: "episode_summary",
          key: `episode_summary_${summary.summary_id}`,
          createdAt: summary.created_at,
          item: summary,
        })),
      ]
        .sort((left, right) => compareCreatedAt(left.createdAt, right.createdAt))
        .slice(-20),
    [activeRegionalAnalysisPackage, dailyReports, episodeSummaries, liveTasks, messages, proposalGuidePrompt, selectedProposal],
  );

  const focusObject = selectedImpact?.entity;
  const openQuestionCount = agentStatus?.open_questions?.length ?? 0;
  const blockedCount = agentStatus?.blocked_by?.length ?? 0;
  const followUpPrompts = useMemo(
    () => getFollowUpPrompts(selectedImpact, agentStatus, pendingProposals, latestAnswer),
    [agentStatus, latestAnswer, pendingProposals, selectedImpact],
  );

  useEffect(() => {
    if (!pendingProposals.length) {
      setSelectedProposalId(null);
      return;
    }
    if (!selectedProposalId || !pendingProposals.some((proposal) => proposal.proposal_id === selectedProposalId)) {
      setSelectedProposalId(pendingProposals[0].proposal_id);
    }
  }, [pendingProposals, selectedProposalId]);

  useEffect(() => {
    streamEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [feedItems, isBusy]);

  function handleProposalClick(proposal: ActionProposalV2) {
    const prompt = getProposalFollowUpPrompt(proposal);
    setSelectedProposalId(proposal.proposal_id);
    onChangeInput(prompt);
  }

  function handleProposalKeyDown(event: KeyboardEvent<HTMLElement>, proposal: ActionProposalV2) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleProposalClick(proposal);
    }
  }

  function handleResolveProposal(
    event: MouseEvent<HTMLButtonElement>,
    proposal: ActionProposalV2,
    decision: "approve" | "reject",
  ) {
    event.stopPropagation();
    const note =
      decision === "approve"
        ? "已在主页事务栏直接批准。"
        : "已在主页事务栏直接驳回。";
    onResolveProposal(proposal.proposal_id, decision, note);
  }

  return (
    <div className={styles.commandCenter}>
      <section className={styles.centerColumn}>
        <div className={styles.liveBanner}>
          <div>
            <p className={styles.overviewCaption}>全局风险态势总览</p>
            <p className={styles.sectionLabel}>实时协同</p>
            <h3>智能体会把处理进展持续写进对话流</h3>
            <p className={styles.liveSummary}>
              {agentStatus?.latest_summary ?? "系统已接管当前事件，正在持续分析风险、资源和处置路径。"}
            </p>
          </div>
          <div className={styles.liveMetrics}>
            <article>
              <span>运行中</span>
              <strong>{agentStatus?.running_task_count ?? 0}</strong>
            </article>
            <article>
              <span>待继续</span>
              <strong>{agentStatus?.pending_task_count ?? 0}</strong>
            </article>
            <article>
              <span>待确认</span>
              <strong>{pendingProposals.length}</strong>
            </article>
          </div>
        </div>

        <div className={styles.chatShell}>
          <div className={styles.chatHeader}>
            <div>
              <p className={styles.sectionLabel}>对话主区</p>
            </div>
            <div className={styles.focusSummary}>
              <span>当前聚焦</span>
              <strong>
                {focusObject
                  ? `${focusObject.name} · ${entityText[focusObject.entity_type]}`
                  : "尚未选定重点对象"}
              </strong>
            </div>
          </div>

          <div className={styles.promptGroup}>
            <div className={styles.promptPanel}>
              <div className={styles.promptPanelHeader}>
                <div>
                  <p className={styles.sectionLabel}>快速开口</p>
                  <h3>先从这些问题开始</h3>
                </div>
              </div>
              <div className={styles.promptShelf}>
                {quickPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className={styles.promptChip}
                    disabled={isBusy}
                    onClick={() => onPrompt(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.promptPanel}>
              <div className={styles.promptPanelHeader}>
                <div>
                  <p className={styles.sectionLabel}>多轮追问</p>
                  <h3>下一轮可以这样继续</h3>
                </div>
                <span className={styles.promptHint}>系统会沿着当前上下文继续答，不需要每次重讲背景</span>
              </div>
              <div className={styles.followUpList}>
                {followUpPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className={styles.followUpChip}
                    disabled={isBusy}
                    onClick={() => onPrompt(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className={styles.messageStream} aria-live="polite">
            {feedItems.map((item) =>
              item.kind === "message" ? (
                <CopilotMessageBubble key={item.key} message={item.message} />
              ) : item.kind === "guide" ? (
                <motion.article
                  key={item.key}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={styles.systemGuideBubble}
                >
                  <div className={styles.agentActivityTopline}>
                    <span>{item.title}</span>
                    <span>{formatTimestamp(item.createdAt)}</span>
                  </div>
                  <div className={styles.systemGuideBody}>
                    <strong>{item.summary}</strong>
                    <p>{item.prompt}</p>
                  </div>
                  <div className={styles.systemGuideActions}>
                    <button
                      type="button"
                      className={styles.followUpChip}
                      disabled={isBusy}
                      onClick={() => onPrompt(item.prompt)}
                    >
                      立即追问
                    </button>
                    <button
                      type="button"
                      className={styles.promptChip}
                      disabled={isBusy}
                      onClick={() => onChangeInput(item.prompt)}
                    >
                      写入输入框
                    </button>
                  </div>
                </motion.article>
              ) : item.kind === "regional_analysis_package" ? (
                <RegionalAnalysisPackageCard
                  key={item.key}
                  item={item.item}
                  isBusy={isBusy}
                  onResolve={onResolveRegionalAnalysisPackage}
                />
              ) : item.kind === "daily_report" ? (
                <FormalDailyReportCard key={item.key} item={item.item} />
              ) : item.kind === "episode_summary" ? (
                <FormalEpisodeSummaryCard key={item.key} item={item.item} />
              ) : (
                <motion.article
                  key={item.key}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={styles.agentActivity}
                >
                  <div className={styles.agentActivityTopline}>
                    <span>{agentText[item.task.agent_name]}</span>
                    <span>{formatTimestamp(item.createdAt)}</span>
                  </div>
                  <div className={styles.agentActivityHeader}>
                    <strong>{formatTaskTitle(item.task)}</strong>
                    <span className={`${styles.statusBadge} ${styles[taskToneClass(item.task.status)]}`}>
                      {taskStatusLabel(item.task.status)}
                    </span>
                  </div>
                  <p>{formatTaskSummary(item.task)}</p>
                </motion.article>
              ),
            )}

            {isBusy ? <TypingBubble selectedImpact={selectedImpact} /> : null}
            <div ref={streamEndRef} />
          </div>

          <form className={styles.composer} onSubmit={onSubmit}>
            <div className={styles.composerGuide}>
              <strong>多轮对话建议</strong>
              <span>可以直接说“继续刚才那个对象”“把建议拆成步骤”“先复述你现在的理解”，系统会延续当前上下文。</span>
            </div>
            <textarea
              className={styles.composerInput}
              placeholder="例如：先复述一下你现在理解的需求，再告诉我你下一步准备处理什么。"
              value={input}
              onChange={(event) => onChangeInput(event.target.value)}
              onKeyDown={onTextareaKeyDown}
              disabled={isBusy}
            />
            <div className={styles.composerFooter}>
              <span>支持多轮追问。按 Ctrl / Cmd + Enter 发送，系统会延续当前对象、风险和事务上下文。</span>
              <button type="submit" className={styles.sendButton} disabled={isBusy || !input.trim()}>
                发送
              </button>
            </div>
          </form>
        </div>
      </section>

      <aside className={styles.sideRail}>
        <section className={styles.railSection}>
          <div className={styles.railHeader}>
            <div>
              <p className={styles.sectionLabel}>事务处理</p>
              <h3>右侧始终盯住待确认动作</h3>
            </div>
            <button type="button" className={styles.railAction} onClick={onOpenOperations}>
              打开处置页
            </button>
          </div>
          {activeRegionalAnalysisPackage ? (
            <RegionalAnalysisPackageSummary
              item={activeRegionalAnalysisPackage}
              isBusy={isBusy}
              onResolve={onResolveRegionalAnalysisPackage}
            />
          ) : null}
          <div className={styles.proposalList}>
            {pendingProposals.length ? (
              pendingProposals.slice(0, 5).map((proposal) => (
                <article
                  key={proposal.proposal_id}
                  role="button"
                  tabIndex={0}
                  className={`${styles.proposalCard} ${
                    selectedProposalId === proposal.proposal_id ? styles.proposalCardActive : ""
                  }`}
                  onClick={() => handleProposalClick(proposal)}
                  onKeyDown={(event) => handleProposalKeyDown(event, proposal)}
                >
                  <div className={styles.proposalTopline}>
                    <span>{proposal.action_display_category ?? "待处理事务"}</span>
                    <span className={`${styles.statusBadge} ${styles.statusPending}`}>待确认</span>
                  </div>
                  <strong>{proposal.title}</strong>
                  <p>{proposal.summary}</p>
                  <div className={styles.proposalActions}>
                    <button
                      type="button"
                      className={styles.proposalGhostAction}
                      disabled={isBusy}
                      onClick={(event) => handleResolveProposal(event, proposal, "reject")}
                    >
                      驳回
                    </button>
                    <button
                      type="button"
                      className={styles.proposalPrimaryAction}
                      disabled={isBusy}
                      onClick={(event) => handleResolveProposal(event, proposal, "approve")}
                    >
                      批准
                    </button>
                  </div>
                </article>
              ))
            ) : (
              <p className={styles.emptyState}>当前没有新的待确认事务，智能体会在生成动作后把它推到这里。</p>
            )}
          </div>
        </section>

        <section className={styles.railSection}>
          <div className={styles.railHeader}>
            <div>
              <p className={styles.sectionLabel}>当前对象</p>
              <h3>{focusObject?.name ?? "等待选中重点对象"}</h3>
            </div>
            {selectedImpact ? (
              <span className={`${styles.statusBadge} ${styles.riskBadge}`}>
                {riskText[selectedImpact.risk_level]}
              </span>
            ) : null}
          </div>
          {selectedImpact ? (
            <div className={styles.focusCard}>
              <div className={styles.focusMeta}>
                <div>
                  <span>对象类型</span>
                  <strong>{entityText[selectedImpact.entity.entity_type]}</strong>
                </div>
                <div>
                  <span>预计受影响时间</span>
                  <strong>{selectedImpact.time_to_impact_minutes} 分钟</strong>
                </div>
              </div>
              <p>{selectedImpact.risk_reason[0] ?? "围绕这个对象继续追问，系统会补充影响原因、证据和建议动作。"}</p>
            </div>
          ) : (
            <p className={styles.emptyState}>从下方列表选中一个重点对象后，右侧会固定展示它的风险与处理上下文。</p>
          )}
        </section>

        <section className={styles.railSection}>
          <div className={styles.railHeader}>
            <div>
              <p className={styles.sectionLabel}>协同状态</p>
              <h3>现在卡在哪，下一步去哪</h3>
            </div>
          </div>
          <div className={styles.statusGrid}>
            <article className={styles.statusCard}>
              <span>开放问题</span>
              <strong>{openQuestionCount}</strong>
            </article>
            <article className={styles.statusCard}>
              <span>阻塞项</span>
              <strong>{blockedCount}</strong>
            </article>
          </div>
          {agentStatus?.open_questions?.length ? (
            <ul className={styles.bulletList}>
              {agentStatus.open_questions.slice(0, 3).map((question) => (
                <li key={question}>{question}</li>
              ))}
            </ul>
          ) : (
            <p className={styles.emptyState}>当前没有新的开放问题，可以继续围绕对象、处置和风险追问。</p>
          )}
        </section>

        <PriorityObjectPanel
          subtitle="高风险对象"
          items={priorityItems}
          selectedId={selectedPriorityId}
          onSelect={onSelectPriority}
        />
      </aside>
    </div>
  );
}

function FormalDailyReportCard({ item }: { item: DailyReportView }) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${styles.systemGuideBubble} ${styles.regionalAnalysisPackageBubble}`}
    >
      <div className={styles.agentActivityTopline}>
        <span>前一日值班日报</span>
        <span>{formatTimestamp(item.created_at)}</span>
      </div>
      <div className={styles.systemGuideBody}>
        <strong>{item.headline}</strong>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>态势摘要</span>
          <p>{item.situation_summary}</p>
        </div>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>关键决策</span>
          <p>{item.decisions_summary}</p>
        </div>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>处置动作</span>
          <p>{item.action_summary}</p>
        </div>
      </div>
      <div className={styles.packageMeta}>
        <span>{item.report_date}</span>
        <span>{item.timezone}</span>
        <span>{item.delivered_session_ids.length} 个会话</span>
      </div>
      {item.unresolved_risks.length ? (
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>未解除风险</span>
          <ul className={styles.packageTitleList}>
            {item.unresolved_risks.slice(0, 4).map((risk) => (
              <li key={risk}>{risk}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {item.next_day_recommendations.length ? (
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>次日建议</span>
          <ul className={styles.packageTitleList}>
            {item.next_day_recommendations.slice(0, 4).map((recommendation) => (
              <li key={recommendation}>{recommendation}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </motion.article>
  );
}

function FormalEpisodeSummaryCard({ item }: { item: EventEpisodeSummaryView }) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${styles.systemGuideBubble} ${styles.regionalAnalysisPackageBubble}`}
    >
      <div className={styles.agentActivityTopline}>
        <span>高风险事件复盘</span>
        <span>{formatTimestamp(item.created_at)}</span>
      </div>
      <div className={styles.systemGuideBody}>
        <strong>{item.headline}</strong>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>升级路径</span>
          <p>{item.escalation_path.length ? item.escalation_path.join(" -> ") : "本次未记录升级路径。"}</p>
        </div>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>关键决策</span>
          <p>{item.key_decisions.length ? item.key_decisions.join("；") : "本次未记录关键决策。"}</p>
        </div>
      </div>
      <div className={styles.packageMeta}>
        <span>{item.successful_actions.length} 项成功动作</span>
        <span>{item.failed_actions.length} 项失败动作</span>
        <span>{item.memory_tags.length} 个记忆标签</span>
      </div>
      {item.reusable_rules.length ? (
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>可复用规则</span>
          <ul className={styles.packageTitleList}>
            {item.reusable_rules.slice(0, 4).map((rule) => (
              <li key={rule}>{rule}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {item.coordination_gaps.length ? (
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>协同缺口</span>
          <ul className={styles.packageTitleList}>
            {item.coordination_gaps.slice(0, 4).map((gap) => (
              <li key={gap}>{gap}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </motion.article>
  );
}

function RegionalAnalysisPackageCard({
  item,
  isBusy,
  onResolve,
}: {
  item: RegionalAnalysisPackageView;
  isBusy: boolean;
  onResolve: (packageId: string, decision: "approve" | "reject", note: string) => void;
}) {
  const isPending = item.status === "pending";
  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${styles.systemGuideBubble} ${styles.regionalAnalysisPackageBubble}`}
    >
      <div className={styles.agentActivityTopline}>
        <span>当前区域风险结论</span>
        <span>{formatTimestamp(item.updated_at ?? item.created_at)}</span>
      </div>
      <div className={styles.systemGuideBody}>
        <strong>{item.analysis_message}</strong>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>区域风险研判</span>
          <p>{item.risk_assessment}</p>
        </div>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>救援方案</span>
          <p>{item.rescue_plan}</p>
        </div>
        <div className={styles.packageSection}>
          <span className={styles.sectionLabel}>物资调度规划</span>
          <p>{item.resource_dispatch_plan}</p>
        </div>
      </div>
      <div className={styles.packageMeta}>
        <span>{item.proposal_count} 项动作</span>
        <span>{item.focus_object_names.length || item.focus_object_ids.length} 个重点对象</span>
        <span>{packageStatusText[item.status] ?? item.status}</span>
      </div>
      {isPending ? (
        <div className={styles.systemGuideActions}>
          <button
            type="button"
            className={styles.proposalGhostAction}
            disabled={isBusy}
            onClick={() => onResolve(item.package_id, "reject", "Reject the current regional analysis package.")}
          >
            整包驳回
          </button>
          <button
            type="button"
            className={styles.proposalPrimaryAction}
            disabled={isBusy}
            onClick={() => onResolve(item.package_id, "approve", "Approve the current regional analysis package.")}
          >
            整包批准
          </button>
        </div>
      ) : null}
    </motion.article>
  );
}

function RegionalAnalysisPackageSummary({
  item,
  isBusy,
  onResolve,
}: {
  item: RegionalAnalysisPackageView;
  isBusy: boolean;
  onResolve: (packageId: string, decision: "approve" | "reject", note: string) => void;
}) {
  const isPending = item.status === "pending";
  return (
    <article className={`${styles.proposalCard} ${styles.packageSummaryCard}`}>
      <div className={styles.proposalTopline}>
        <span>当前区域分析包</span>
        <span className={`${styles.statusBadge} ${styles[packageToneClass(item.status)]}`}>
          {packageStatusText[item.status] ?? item.status}
        </span>
      </div>
      <strong>{item.analysis_message}</strong>
      <p>{item.risk_assessment}</p>
      <ul className={styles.packageTitleList}>
        {item.proposal_titles.map((title) => (
          <li key={title}>{title}</li>
        ))}
      </ul>
      {isPending ? (
        <div className={styles.proposalActions}>
          <button
            type="button"
            className={styles.proposalGhostAction}
            disabled={isBusy}
            onClick={() => onResolve(item.package_id, "reject", "Reject the current regional analysis package.")}
          >
            整包驳回
          </button>
          <button
            type="button"
            className={styles.proposalPrimaryAction}
            disabled={isBusy}
            onClick={() => onResolve(item.package_id, "approve", "Approve the current regional analysis package.")}
          >
            整包批准
          </button>
        </div>
      ) : null}
    </article>
  );
}

function TypingBubble({ selectedImpact }: { selectedImpact: EntityImpactView | null }) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={styles.typingBubble}
    >
      <div className={styles.agentActivityTopline}>
        <span>系统处理中</span>
        <span>实时同步</span>
      </div>
      <div className={styles.typingHeader}>
        <strong>
          {selectedImpact ? `正在继续分析 ${selectedImpact.entity.name}` : "正在整理新一轮回答"}
        </strong>
        <div className={styles.typingDots} aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </div>
      <p>
        正在汇总证据、任务进展和待处理事务，稍后会把新的回应直接追加到对话流底部。
      </p>
    </motion.article>
  );
}

const packageStatusText: Record<string, string> = {
  pending: "待审批",
  approved: "已批准",
  rejected: "已驳回",
  withdrawn: "已撤回",
  superseded: "已过期",
  partially_resolved: "部分处理",
};

function packageToneClass(status: string) {
  return {
    pending: "statusPending",
    approved: "statusCompleted",
    rejected: "statusFailed",
    withdrawn: "statusMuted",
    superseded: "statusMuted",
    partially_resolved: "statusRunning",
  }[status] ?? "statusMuted";
}

function getFollowUpPrompts(
  selectedImpact: EntityImpactView | null,
  agentStatus: AgentStatusView | null,
  pendingProposals: ActionProposalV2[],
  latestAnswer: StructuredAnswer | null,
) {
  const llmPrompts = latestAnswer?.follow_up_prompts?.filter((prompt) => typeof prompt === "string" && prompt.trim());
  if (llmPrompts?.length) {
    return llmPrompts.slice(0, 4);
  }

  const prompts: string[] = [];

  if (latestAnswer?.recommended_actions?.length) {
    prompts.push(`请把“${latestAnswer.recommended_actions[0]}”展开成更具体的执行步骤。`);
  } else if (selectedImpact) {
    prompts.push(`请先复述一下你目前对${selectedImpact.entity.name}的判断依据。`);
    prompts.push(`如果现在就处理${selectedImpact.entity.name}，优先动作应该是什么？`);
  } else {
    prompts.push("请先根据当前总览告诉我你最建议优先关注哪个对象。");
  }

  if (agentStatus?.open_questions?.[0]) {
    prompts.push(`围绕这个问题继续展开：${agentStatus.open_questions[0]}`);
  } else {
    prompts.push("你现在还缺哪些信息，才会影响下一步决策？");
  }

  if (pendingProposals[0]) {
    prompts.push(`把“${pendingProposals[0].title}”拆成执行步骤，并说明每一步的目的。`);
  } else {
    prompts.push("如果我要继续多轮确认需求，你建议我下一句怎么问最有效？");
  }

  return prompts.slice(0, 4);
}

function getProposalFollowUpPrompt(proposal: ActionProposalV2) {
  if (proposal.chat_follow_up_prompt?.trim()) {
    return proposal.chat_follow_up_prompt.trim();
  }
  const actionName = proposal.action_display_name ?? proposal.title;
  return `围绕事务“${actionName}”继续追问：请先说明这项事务为什么现在要做、它依赖哪些前提、执行时最容易出错的环节是什么。`;
}

function compareCreatedAt(left: string, right: string) {
  return new Date(left).getTime() - new Date(right).getTime();
}

function formatTaskTitle(task: AgentTask) {
  const targetName =
    readString(task.input_payload.target_name) ??
    readString(task.input_payload.entity_name) ??
    readString(task.input_payload.focus_entity_name);

  if (targetName) {
    return `${normalizeLabel(task.task_type)} · ${targetName}`;
  }
  return normalizeLabel(task.task_type);
}

function formatTaskSummary(task: AgentTask) {
  const summary =
    readString(task.output_payload.summary) ??
    readString(task.output_payload.message) ??
    readString(task.output_payload.result_summary) ??
    task.failure_reason;

  if (summary) {
    return normalizeAgentTerminology(summary);
  }

  if (task.status === "running") {
    return "任务已经启动，系统正在拉取数据、组织证据并准备下一轮回应。";
  }

  if (task.status === "pending") {
    return "任务已进入待执行队列，完成前会继续在对话区同步进展。";
  }

  if (task.status === "completed") {
    return "该步骤已经完成，相关结果会继续汇入当前对话和事务清单。";
  }

  return "该任务状态已变化，系统会根据结果继续调整后续处理路径。";
}

function taskStatusLabel(status: AgentTask["status"]) {
  return {
    pending: "排队中",
    running: "处理中",
    completed: "已完成",
    failed: "失败",
    canceled: "已取消",
    superseded: "已替换",
  }[status];
}

function taskToneClass(status: AgentTask["status"]) {
  return {
    pending: "statusPending",
    running: "statusRunning",
    completed: "statusCompleted",
    failed: "statusFailed",
    canceled: "statusMuted",
    superseded: "statusMuted",
  }[status];
}

function normalizeLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function readString(value: unknown) {
  return typeof value === "string" && value.trim() ? value : null;
}
