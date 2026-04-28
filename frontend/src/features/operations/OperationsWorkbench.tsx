import type { ComponentProps } from "react";
import styles from "../../App.module.css";
import { ExecutionFlowBoard } from "../../components/ExecutionFlowBoard";
import { AdvisoryCard } from "../../components/OperationPanel";
import { RegionalAnalysisPackageHistoryPanel } from "../../components/RegionalAnalysisPackageHistoryPanel";
import { RegionalProposalHistoryPanel } from "../../components/RegionalProposalHistoryPanel";
import { SignalTimeline } from "../../components/SignalTimeline";
import { ToolExecutionSummary } from "../../components/ToolExecutionSummary";
import { buildExecutionFlowStats, operationsPageText } from "../../lib/appText";
import { OperationsPage } from "../../pages/OperationsPage";
import type { AudienceWarningDraft } from "../../types/api";

interface OperationsWorkbenchProps {
  executionFlowSteps: ComponentProps<typeof ExecutionFlowBoard>["steps"];
  latestAnswer: ComponentProps<typeof ToolExecutionSummary>["answer"];
  pendingProposalCount: number;
  proposalHistoryItems: ComponentProps<typeof RegionalProposalHistoryPanel>["items"];
  pendingProposalItems: ComponentProps<typeof RegionalProposalHistoryPanel>["items"];
  analysisPackageItems: ComponentProps<typeof RegionalAnalysisPackageHistoryPanel>["items"];
  agentTimelineItems: ComponentProps<typeof SignalTimeline>["items"];
  latestWarningDrafts: AudienceWarningDraft[];
  activeAdvisory: ComponentProps<typeof AdvisoryCard>["advisory"];
  toolExecutionCount: number;
}

export function OperationsWorkbench({
  executionFlowSteps,
  latestAnswer,
  pendingProposalCount,
  proposalHistoryItems,
  pendingProposalItems,
  analysisPackageItems,
  agentTimelineItems,
  latestWarningDrafts,
  activeAdvisory,
  toolExecutionCount,
}: OperationsWorkbenchProps) {
  return (
    <OperationsPage
      list={
        <div className={styles.primaryColumn}>
          <ExecutionFlowBoard
            title={operationsPageText.executionBoardTitle}
            description={operationsPageText.executionBoardDescription}
            stats={buildExecutionFlowStats({
              pendingProposalCount,
              proposalHistoryCount: proposalHistoryItems.length,
              toolExecutionCount,
            })}
            steps={executionFlowSteps}
          />
          <ToolExecutionSummary answer={latestAnswer} />
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>{operationsPageText.analysisPackageSectionLabel}</p>
                <h3>{operationsPageText.analysisPackageSectionTitle}</h3>
              </div>
            </div>
            <RegionalAnalysisPackageHistoryPanel items={analysisPackageItems} />
          </div>
        </div>
      }
      detail={
        <div className={styles.sideColumn}>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>{operationsPageText.proposalHistorySectionLabel}</p>
                <h3>{operationsPageText.proposalHistorySectionTitle}</h3>
              </div>
            </div>
            <RegionalProposalHistoryPanel items={proposalHistoryItems} />
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>{operationsPageText.pendingConfirmSectionLabel}</p>
                <h3>{operationsPageText.pendingConfirmSectionTitle}</h3>
              </div>
            </div>
            <RegionalProposalHistoryPanel items={pendingProposalItems} />
          </div>
          <div className={styles.panelFrame}>
            <SignalTimeline
              title={operationsPageText.timelineTitle}
              subtitle={operationsPageText.timelineSubtitle}
              items={agentTimelineItems}
              emptyText={operationsPageText.timelineEmpty}
            />
          </div>
          <div className={styles.panelFrame}>
            <div className={styles.panelHeaderCompact}>
              <div>
                <p className={styles.sectionLabel}>Warning Drafts</p>
                <h3>分众预警草稿</h3>
              </div>
            </div>
            <div className={styles.answerList}>
              {latestWarningDrafts.slice(0, 4).map((draft) => (
                <article key={draft.warning_id} className={styles.metricBlock}>
                  <span>{draft.audience}</span>
                  <strong>{`${draft.audience} warning`}</strong>
                  <small>{draft.grounding_summary || draft.content}</small>
                </article>
              ))}
              {!latestWarningDrafts.length ? (
                <div className={styles.emptyState}>当前还没有新的分众预警草稿，proposal 审批后会在这里出现。</div>
              ) : null}
            </div>
          </div>
          <div className={styles.panelFrame}>
            <AdvisoryCard advisory={activeAdvisory} />
          </div>
        </div>
      }
    />
  );
}
