import type { ComponentProps } from "react";
import styles from "../../App.module.css";
import { MultiAgentDesk } from "../../components/MultiAgentDesk";
import { SignalTimeline } from "../../components/SignalTimeline";
import { AgentsPage } from "../../pages/AgentsPage";
import { buildAgentDivergenceRows } from "../../state/agentTwinSelectors";
import { normalizeAgentTerminology } from "../../lib/agentUiText";
import type { AgentCouncilView, AudienceWarningDraft } from "../../types/api";

type MultiAgentDeskProps = ComponentProps<typeof MultiAgentDesk>;
type SignalTimelineItems = ComponentProps<typeof SignalTimeline>["items"];

interface AgentsWorkbenchProps {
  agentCouncil?: AgentCouncilView | null;
  eventId?: string;
  agentStatus: MultiAgentDeskProps["agentStatus"];
  agentTasks: MultiAgentDeskProps["agentTasks"];
  sessionMemoryView: MultiAgentDeskProps["sessionMemoryView"];
  sharedMemorySnapshot: MultiAgentDeskProps["sharedMemorySnapshot"];
  episodeSummaries: MultiAgentDeskProps["episodeSummaries"];
  triggerEvents: MultiAgentDeskProps["triggerEvents"];
  agentTimeline: MultiAgentDeskProps["agentTimeline"];
  agentTimelineItems: SignalTimelineItems;
  supervisorRuns: MultiAgentDeskProps["supervisorRuns"];
  supervisorLoopStatus: MultiAgentDeskProps["supervisorLoopStatus"];
  recentAgentResults: MultiAgentDeskProps["recentAgentResults"];
  experienceContext: MultiAgentDeskProps["experienceContext"];
  decisionReport: MultiAgentDeskProps["decisionReport"];
  agentMetrics: MultiAgentDeskProps["agentMetrics"];
  evaluationBenchmarks: MultiAgentDeskProps["evaluationBenchmarks"];
  latestEvaluationReport: MultiAgentDeskProps["latestEvaluationReport"];
  busy: boolean;
  canControlSupervisor: boolean;
  canReplayTask: boolean;
  canRunEvaluation: boolean;
  onRunSupervisor: MultiAgentDeskProps["onRunSupervisor"];
  onTickSupervisor: MultiAgentDeskProps["onTickSupervisor"];
  onReplayTask: MultiAgentDeskProps["onReplayTask"];
  onRunEvaluation: MultiAgentDeskProps["onRunEvaluation"];
  onReplayEvaluationReport: MultiAgentDeskProps["onReplayEvaluationReport"];
  pendingProposalCount: number;
  approvedProposalCount: number;
  warningDraftCount: number;
  latestWarningDrafts: AudienceWarningDraft[];
}

export function AgentsWorkbench({
  agentCouncil,
  eventId,
  agentStatus,
  agentTasks,
  sessionMemoryView,
  sharedMemorySnapshot,
  episodeSummaries,
  triggerEvents,
  agentTimeline,
  agentTimelineItems,
  supervisorRuns,
  supervisorLoopStatus,
  recentAgentResults,
  experienceContext,
  decisionReport,
  agentMetrics,
  evaluationBenchmarks,
  latestEvaluationReport,
  busy,
  canControlSupervisor,
  canReplayTask,
  canRunEvaluation,
  onRunSupervisor,
  onTickSupervisor,
  onReplayTask,
  onRunEvaluation,
  onReplayEvaluationReport,
  pendingProposalCount,
  approvedProposalCount,
  warningDraftCount,
  latestWarningDrafts,
}: AgentsWorkbenchProps) {
  const councilRoles = agentCouncil?.roles ?? [];
  const agentDecisionPath = agentCouncil?.decision_path ?? [];
  const agentOpenQuestions = agentCouncil?.open_questions ?? [];
  const agentBlockedBy = agentCouncil?.blocked_by ?? [];
  const recentCouncilResults = recentAgentResults.slice(0, 4);
  const evidenceCompareResults = recentCouncilResults.filter(
    (result) =>
      result.evidence_refs.length > 0 ||
      result.missing_slots.length > 0 ||
      result.handoff_recommendations.length > 0,
  );
  const agentDivergenceRows = buildAgentDivergenceRows({
    recentResults: recentCouncilResults,
    sharedMemorySnapshot,
    decisionReport,
    agentCouncil,
    maxRows: 4,
  });

  return (
    <AgentsPage
      briefing={
        <div className={styles.primaryColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Agent Council</p>
                <h3>多智能体会商摘要</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {councilRoles.map((role) => (
                <article key={role.role} className={styles.metricBlock}>
                  <span>{role.label}</span>
                  <strong>{role.summary}</strong>
                  <small>
                    {role.recommended_action ?? "等待新的任务或证据输入。"} / 证据 {role.evidence_count} 条
                  </small>
                </article>
              ))}
              {!councilRoles.length ? (
                <div className={styles.emptyState}>当前还没有可展示的智能体会商摘要。</div>
              ) : null}
            </div>
            <div className={styles.answerTags}>
              <span>审计状态：{agentCouncil?.audit_decision.status ?? "unknown"}</span>
              <span>开放问题：{agentCouncil?.open_questions.length ?? 0}</span>
              <span>阻断项：{agentCouncil?.blocked_by.length ?? 0}</span>
            </div>
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Difference Matrix</p>
                <h3>角色差异与分歧</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {agentDivergenceRows.length ? (
                agentDivergenceRows.map((row) => (
                  <article key={row.result.result_id} className={styles.metricBlock}>
                    <span>{row.result.agent_name} / {row.disposition}</span>
                    <strong>{normalizeAgentTerminology(row.result.summary)}</strong>
                    <small>
                      分歧点：{row.disagreement} / 置信度 {Math.round(row.confidence * 100)}% / 证据{" "}
                      {row.result.evidence_refs.length} 条
                    </small>
                    <small>编排理由：{row.rationale}</small>
                  </article>
                ))
              ) : (
                <div className={styles.emptyState}>当前没有足够的 recent results 来展示角色差异。</div>
              )}
            </div>
            <div className={styles.answerTags}>
              {agentOpenQuestions.slice(0, 4).map((item) => (
                <span key={item}>{item}</span>
              ))}
              {!agentOpenQuestions.length ? <span>当前没有新的 open questions</span> : null}
            </div>
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Evidence Compare</p>
                <h3>证据对照板</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {evidenceCompareResults.length ? (
                evidenceCompareResults.map((result) => (
                  <article key={`${result.result_id}-evidence`} className={styles.metricBlock}>
                    <span>{result.agent_name}</span>
                    <strong>
                      引用 {result.evidence_refs.length} 条 / 缺口 {result.missing_slots.length} 项
                    </strong>
                    <small>
                      {result.evidence_refs.slice(0, 3).join(" | ") || "当前没有明确证据引用"}
                      {result.missing_slots.length
                        ? ` / 缺口：${result.missing_slots.slice(0, 2).join(" | ")}`
                        : ""}
                      {result.handoff_recommendations.length
                        ? ` / 建议移交：${result.handoff_recommendations.slice(0, 2).join(" | ")}`
                        : ""}
                    </small>
                  </article>
                ))
              ) : (
                <div className={styles.emptyState}>当前还没有足够的证据引用差异可以展示。</div>
              )}
            </div>
          </div>
        </div>
      }
      chamber={
        <div className={styles.primaryColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Conference View</p>
                <h3>会商室主桌面</h3>
              </div>
            </div>
            <MultiAgentDesk
              eventId={eventId}
              agentStatus={agentStatus}
              agentTasks={agentTasks}
              sessionMemoryView={sessionMemoryView}
              sharedMemorySnapshot={sharedMemorySnapshot}
              episodeSummaries={episodeSummaries}
              triggerEvents={triggerEvents}
              agentTimeline={agentTimeline}
              supervisorRuns={supervisorRuns}
              supervisorLoopStatus={supervisorLoopStatus}
              recentAgentResults={recentAgentResults}
              experienceContext={experienceContext}
              decisionReport={decisionReport}
              agentMetrics={agentMetrics}
              evaluationBenchmarks={evaluationBenchmarks}
              latestEvaluationReport={latestEvaluationReport}
              busy={busy}
              canControlSupervisor={canControlSupervisor}
              canReplayTask={canReplayTask}
              canRunEvaluation={canRunEvaluation}
              onRunSupervisor={onRunSupervisor}
              onTickSupervisor={onTickSupervisor}
              onReplayTask={onReplayTask}
              onRunEvaluation={onRunEvaluation}
              onReplayEvaluationReport={onReplayEvaluationReport}
            />
          </div>
        </div>
      }
      orchestration={
        <div className={styles.primaryColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Supervisor Orchestration</p>
                <h3>编排结果与审计边界</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              <article className={styles.metricBlock}>
                <span>Audit rationale</span>
                <strong>{agentCouncil?.audit_decision.rationale ?? "等待新的 supervisor 编排结果。"}</strong>
                <small>
                  {agentCouncil?.audit_decision.approval_required
                    ? "当前仍要求人工审批后放行。"
                    : "当前编排允许自动推进。"}
                </small>
              </article>
              <article className={styles.metricBlock}>
                <span>Decision path</span>
                <strong>{agentDecisionPath[0] ?? "waiting"}</strong>
                <small>{agentDecisionPath.slice(1, 4).join(" -> ") || "当前还没有完整的 decision path。"}</small>
              </article>
            </div>
            <div className={styles.answerTags}>
              {agentDecisionPath.slice(0, 4).map((item) => (
                <span key={item}>{item}</span>
              ))}
              {!agentDecisionPath.length ? <span>当前还没有 supervisor decision path</span> : null}
              {agentBlockedBy.slice(0, 3).map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>
          <div className={styles.panelFrame}>
            <SignalTimeline
              title="智能体时间线"
              subtitle="最近触发与任务变化"
              items={agentTimelineItems}
              emptyText="当前没有新的智能体时间线记录。"
            />
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Closure Readiness</p>
                <h3>会商到闭环的推进状态</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              <article className={styles.metricBlock}>
                <span>Pending proposals</span>
                <strong>{pendingProposalCount}</strong>
                <small>待审批 proposal 仍需人工放行后才能进入 warning 生成。</small>
              </article>
              <article className={styles.metricBlock}>
                <span>Approved proposals</span>
                <strong>{approvedProposalCount}</strong>
                <small>已批准 proposal 可以直接进入 audience warnings 和执行留痕。</small>
              </article>
              <article className={styles.metricBlock}>
                <span>Warning drafts</span>
                <strong>{warningDraftCount || latestWarningDrafts.length}</strong>
                <small>warning drafts 数量可以帮助判断闭环是否已经落地。</small>
              </article>
            </div>
          </div>
        </div>
      }
      governance={
        <div className={styles.sideColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Open Questions</p>
                <h3>未解决问题</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {agentOpenQuestions.length ? (
                agentOpenQuestions.map((item) => (
                  <article key={item} className={styles.metricBlock}>
                    <span>question</span>
                    <strong>{item}</strong>
                    <small>该问题尚未被会商链完整关闭，需要继续追问或补证。</small>
                  </article>
                ))
              ) : (
                <div className={styles.emptyState}>当前没有新的 open questions。</div>
              )}
            </div>
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Blocked By</p>
                <h3>放行阻断项</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {agentBlockedBy.length ? (
                agentBlockedBy.map((item) => (
                  <article key={item} className={styles.metricBlock}>
                    <span>blocked</span>
                    <strong>{item}</strong>
                    <small>当前会商结论被此边界限制，不能直接自动推进。</small>
                  </article>
                ))
              ) : (
                <div className={styles.emptyState}>当前没有新的阻断项，会商结果可继续推进。</div>
              )}
            </div>
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Closure Link</p>
                <h3>闭环出口</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {latestWarningDrafts.slice(0, 3).map((draft) => (
                <article key={draft.warning_id} className={styles.metricBlock}>
                  <span>{draft.audience}</span>
                  <strong>{draft.channel}</strong>
                  <small>{draft.grounding_summary || draft.content}</small>
                </article>
              ))}
              {!latestWarningDrafts.length ? (
                <div className={styles.emptyState}>
                  当前还没有新的 warning draft，会商结果将在批准 proposal 后进入这里。
                </div>
              ) : null}
            </div>
          </div>
        </div>
      }
    />
  );
}
