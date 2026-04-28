import { AnimatePresence, motion } from "framer-motion";
import { useMemo } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import styles from "./App.module.css";
import { AppShell } from "./components/AppShell";
import { DigitalTwinImpactScreen } from "./components/DigitalTwinImpactScreen";
import { GlobalRegionalProposalDialog } from "./components/GlobalRegionalProposalDialog";
import { MetricStrip } from "./components/MetricStrip";
import { operatorRoleText } from "./components/SecurityDesk";
import {
  bootStateText,
  entityText,
  executionStatusText,
  healthStateText,
  pageMeta,
  riskText,
} from "./config/consoleConfig";
import { useAgentTwinConsole } from "./hooks/useAgentTwinConsole";
import { DataPage } from "./pages/DataPage";
import panelStyles from "./styles/shared-panels.module.css";
import {
  appShellText,
  buildAgentTimelineFallback,
  executionFlowText,
  formatTrendLabel,
  formatPendingMetricHint,
  overviewMetricText,
} from "./lib/appText";
import { AdminDesk } from "./features/dataManagement/AdminDesk";
import { AgentsWorkbench } from "./features/agents/AgentsWorkbench";
import { CopilotWorkbench } from "./features/copilot/CopilotWorkbench";
import { OperationsWorkbench } from "./features/operations/OperationsWorkbench";
import { ReliabilityWorkbench } from "./features/reliability/ReliabilityWorkbench";
import { normalizeAgentTerminology } from "./lib/agentUiText";
import {
  formatAgentTaskEventType,
  formatExecutionMode,
  formatProposalStreamStatus,
  formatRegionalActionType,
  formatTriggerType,
} from "./lib/displayText";
import {
  coerceDrafts,
  coerceLogs,
  coerceStrings,
  coerceTemplates,
  formatPercent,
  formatTimestamp,
  severityText,
} from "./lib/consoleFormatting";
import type {
  EntityImpactView,
  OperatorRole,
} from "./types/api";

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const consoleState = useAgentTwinConsole();
  const capabilityMap = consoleState.operatorCapabilities?.capabilities ?? {};
  const canEditRuntimeAdmin = Boolean(capabilityMap.runtime_admin_write);
  const canManageDataset = Boolean(capabilityMap.dataset_manage);
  const canControlSupervisor = Boolean(capabilityMap.supervisor_control);
  const canReplayTask = Boolean(capabilityMap.agent_replay);
  const canRunEvaluation = Boolean(capabilityMap.evaluation_run);
  const canRunArchive = Boolean(capabilityMap.archive_run);

  const topRisk = consoleState.twinOverview?.overall_risk_level ?? consoleState.hazardState?.overall_risk_level ?? "None";
  const selectedImpact = consoleState.selectedImpact as EntityImpactView;
  const leadImpact = useMemo(() => consoleState.topImpacts[0] ?? null, [consoleState.topImpacts]);
  const currentPage = pageMeta[location.pathname as keyof typeof pageMeta];

  if (!currentPage) {
    return <Navigate to="/" replace />;
  }

  const isOverviewPage = location.pathname === "/";
  const isCopilotPage = location.pathname === "/copilot";
  const isOperationsPage = location.pathname === "/operations";
  const isDataPage = location.pathname === "/data";
  const isAgentsPage = location.pathname === "/agents";
  const isReliabilityPage = location.pathname === "/reliability";

  const trendLabel = formatTrendLabel(consoleState.twinOverview?.trend ?? consoleState.hazardState?.trend);
  const highPriorityCount =
    consoleState.twinOverview?.focus_objects.length ??
    consoleState.topImpacts.filter((impact) => impact.risk_level !== "None").length;
  const approvedProposalCount =
    consoleState.twinOverview?.approved_proposal_count ??
    consoleState.proposals.filter((item) => item.status === "approved").length;
  const pendingProposalCount = consoleState.twinOverview?.pending_proposal_count ?? consoleState.pendingProposals.length;
  const warningDraftCount = consoleState.twinOverview?.warning_draft_count ?? 0;
  const latestToolExecutions = consoleState.latestAnswer?.tool_executions ?? [];
  const latestWarningDrafts =
    consoleState.warningDrafts.length > 0
      ? consoleState.warningDrafts
      : consoleState.twinOverview?.recent_warning_drafts ?? [];

  const primaryPaths = new Set(["/", "/operations", "/agents"]);
  const navigation = Object.entries(pageMeta)
    .filter(([path]) => primaryPaths.has(path))
    .map(([path, meta]) => ({ path, label: meta.label }));
  const utilityNavigation = Object.entries(pageMeta)
    .filter(([path]) => !primaryPaths.has(path))
    .map(([path, meta]) => ({ path, label: meta.label }));
  const shellCurrentPageLabel = isOverviewPage ? "数字孪生主屏" : currentPage.label;
  const shellCurrentPageTitle = isOverviewPage ? "数字孪生智能体洪水预警系统" : currentPage.title;
  const shellCurrentPageDescription = isOverviewPage ? undefined : currentPage.description;

  const pageMetricItems = isOverviewPage
    ? [
        {
          label: overviewMetricText.riskLabel,
          value: riskText[topRisk],
          hint: consoleState.twinOverview?.event_title ?? consoleState.event?.title ?? overviewMetricText.riskHintFallback,
          tone: topRisk === "Red" || topRisk === "Orange" ? ("risk" as const) : topRisk === "Yellow" ? ("warning" as const) : ("success" as const),
        },
        {
          label: overviewMetricText.trendLabel,
          value: trendLabel,
          hint: overviewMetricText.trendHint,
          tone: topRisk === "Red" || topRisk === "Orange" ? ("warning" as const) : ("neutral" as const),
        },
        {
          label: overviewMetricText.priorityLabel,
          value: `${highPriorityCount || consoleState.topImpacts.length}`,
          hint: overviewMetricText.priorityHint,
          tone: highPriorityCount ? ("warning" as const) : ("neutral" as const),
        },
        {
          label: overviewMetricText.pendingLabel,
          value: `${pendingProposalCount}`,
          hint: warningDraftCount
            ? `已生成 ${warningDraftCount} 条分众预警草稿`
            : formatPendingMetricHint(approvedProposalCount),
          tone: pendingProposalCount ? ("warning" as const) : ("success" as const),
        },
      ]
    : isOperationsPage
        ? [
            {
              label: "待确认动作",
              value: `${consoleState.pendingProposals.length}`,
              hint: "系统正在等待人工确认处置建议。",
              tone: consoleState.pendingProposals.length ? ("warning" as const) : ("success" as const),
            },
            {
              label: "历史建议数",
              value: `${consoleState.regionalProposalHistory.length}`,
              hint: approvedProposalCount ? `${approvedProposalCount} 条建议已批准执行` : "当前还没有形成历史建议。",
              tone: consoleState.regionalProposalHistory.length ? ("success" as const) : ("neutral" as const),
            },
            {
              label: "请示流状态",
              value: formatProposalStreamStatus(consoleState.proposalStreamStatus),
               hint: consoleState.agentStatus?.latest_summary ?? "等待处置链路继续推进。",
              tone: consoleState.proposalStreamStatus === "error" ? ("risk" as const) : consoleState.proposalStreamStatus === "open" ? ("success" as const) : ("warning" as const),
            },
            {
              label: "工具执行次数",
              value: `${latestToolExecutions.length}`,
              hint: latestToolExecutions.length ? "规划链路已触发关键能力调用。" : "当前尚未触发关键能力调用。",
              tone: latestToolExecutions.length ? ("success" as const) : ("neutral" as const),
            },
          ]
        : [
            {
               label: "总体风险",
              value: riskText[topRisk],
               hint: consoleState.event?.title ?? "当前事件",
              tone: topRisk === "Red" || topRisk === "Orange" ? ("risk" as const) : ("success" as const),
            },
            {
              label: "执行状态",
              value: executionStatusText[consoleState.executionStatus],
               hint: consoleState.errorMessage ?? consoleState.adminStatus ?? "运行稳定",
              tone: consoleState.executionStatus === "error" ? ("risk" as const) : consoleState.executionStatus === "running" ? ("warning" as const) : ("success" as const),
            },
          ];

  const priorityItems = consoleState.curatedEntities.map((entity) => {
    const impact = consoleState.entityImpacts[entity.id];
    return {
      id: entity.id,
      name: entity.name,
      typeLabel: entityText[entity.type],
      village: entity.village,
      emphasis: entity.emphasis,
      riskLabel: impact ? riskText[impact.risk_level] : undefined,
      riskTone: impact
        ? ({
            None: "none",
            Blue: "blue",
            Yellow: "yellow",
            Orange: "orange",
            Red: "red",
          }[impact.risk_level] as "none" | "blue" | "yellow" | "orange" | "red")
        : undefined,
    };
  });

  const overviewSignalItems = consoleState.openAlerts.length
    ? consoleState.openAlerts.slice(0, 4).map((alert) => ({
        id: alert.alert_id,
        title: alert.summary,
        detail: alert.details || executionFlowText.alertDetailFallback,
        meta: formatTimestamp(alert.last_seen_at ?? alert.first_seen_at),
        tone:
          alert.severity === "critical"
            ? ("critical" as const)
            : alert.severity === "warning"
              ? ("warning" as const)
              : ("info" as const),
      }))
    : consoleState.supervisorRuns.slice(0, 4).map((run) => ({
        id: run.supervisor_run_id,
        title: formatTriggerType(run.trigger_type),
        detail: run.summary || executionFlowText.supervisorRunFallback,
        meta: formatTimestamp(run.created_at),
        tone: run.status === "failed" ? ("critical" as const) : ("info" as const),
      }));

  const agentTimelineItems = consoleState.agentTimeline.slice(0, 5).map((entry) => ({
    id: entry.entry_id,
    title:
      entry.entry_type === "trigger"
        ? `触发：${formatTriggerType(entry.trigger_type)}`
        : `任务：${formatAgentTaskEventType(entry.task_event_type)}`,
    detail: normalizeAgentTerminology(entry.summary) || buildAgentTimelineFallback(entry.entry_type === "trigger"),
    meta: formatTimestamp(entry.created_at),
    tone:
      entry.entry_type === "trigger"
        ? ("warning" as const)
        : String(entry.payload.status ?? "").toLowerCase() === "failed"
          ? ("critical" as const)
          : ("info" as const),
  }));

  const executionFlowSteps = [
    {
      id: "sense",
      title: executionFlowText.senseTitle,
      summary: consoleState.hazardState
        ? `已汇聚当前事件水情、路网与监测状态，形成 ${riskText[topRisk]} 风险判断。`
        : executionFlowText.noHazardStateSummary,
      detail: consoleState.hazardState
        ? `趋势判断为 ${trendLabel}，可达路段 ${consoleState.hazardState.road_reachability?.length ?? 0} 条。`
        : executionFlowText.noHazardStateDetail,
      status: consoleState.hazardState ? ("complete" as const) : ("pending" as const),
    },
    {
      id: "impact",
      title: executionFlowText.impactTitle,
      summary: leadImpact
        ? `系统已识别 ${leadImpact.entity.name} 为首要影响对象。`
        : executionFlowText.noLeadImpactSummary,
      detail: leadImpact
        ? leadImpact.risk_reason[0] ?? "已结合对象属性、位置和脆弱性形成对象级研判。"
        : executionFlowText.noLeadImpactDetail,
      status: leadImpact ? ("complete" as const) : ("pending" as const),
    },
    {
      id: "plan",
      title: executionFlowText.planTitle,
      summary: consoleState.latestAnswer?.planner_summary ?? executionFlowText.noPlannerSummary,
      detail: consoleState.latestAnswer?.planning_layers_summary?.[0] ?? executionFlowText.noPlannerDetail,
      status: consoleState.latestAnswer ? ("complete" as const) : ("pending" as const),
    },
    {
      id: "tooling",
      title: executionFlowText.toolingTitle,
      summary: latestToolExecutions.length
        ? `本轮已触发 ${latestToolExecutions.length} 次关键能力调用。`
        : executionFlowText.noToolingSummary,
      detail: latestToolExecutions[0]?.output_summary ?? executionFlowText.noToolingDetail,
      status: latestToolExecutions.length ? ("active" as const) : ("pending" as const),
    },
    {
      id: "confirm",
      title: executionFlowText.confirmTitle,
      summary: consoleState.pendingProposals.length
        ? `${consoleState.pendingProposals.length} 条动作等待人工确认。`
        : executionFlowText.noPendingConfirmationSummary,
      detail: approvedProposalCount
        ? `已有 ${approvedProposalCount} 条动作完成批准闭环。`
        : executionFlowText.noPendingConfirmationDetail,
      status: consoleState.pendingProposals.length
        ? ("active" as const)
        : approvedProposalCount
          ? ("complete" as const)
          : ("pending" as const),
    },
  ];

  const analysisPackageItems = [
    ...(consoleState.pendingRegionalAnalysisPackage ? [consoleState.pendingRegionalAnalysisPackage] : []),
    ...consoleState.regionalAnalysisPackageHistory,
  ];
  const proposalHistoryItems = consoleState.regionalProposalHistory;
  const pendingProposalItems = consoleState.regionalProposalQueueSnapshot?.items ?? [];

  return (
    <>
      <GlobalRegionalProposalDialog
        open={!isCopilotPage && consoleState.regionalProposalModalOpen}
        busy={consoleState.isBusy}
        snapshot={consoleState.regionalProposalQueueSnapshot}
        onApprove={(proposalId, note) => consoleState.resolveProposal(proposalId, "approve", note)}
        onReject={(proposalId, note) => consoleState.resolveProposal(proposalId, "reject", note)}
        onSaveDraft={consoleState.updateRegionalProposalDraft}
        onSnooze={consoleState.snoozeRegionalProposalModal}
      />
      <AppShell
        brandTitle="数字孪生智能体洪水预警系统"
        currentPageLabel={shellCurrentPageLabel}
        currentPageTitle={shellCurrentPageTitle}
        currentPageDescription={shellCurrentPageDescription}
        navigation={navigation}
        utilityNavigation={utilityNavigation}
        operatorControl={
          <div className={styles.topbarActions}>
            <label className={styles.fieldBlock}>
              <span>{appShellText.currentRole}</span>
              <select
                className={styles.fieldInput}
                aria-label="topbar-operator-role"
                value={consoleState.operatorRole}
                onChange={(event) => consoleState.setOperatorRole(event.target.value as OperatorRole)}
              >
                {Object.entries(operatorRoleText).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" className={styles.secondaryButton} onClick={() => void consoleState.refresh()}>
              {appShellText.refresh}
            </button>
          </div>
        }
        statusSignals={
          <>
            <div className={panelStyles.statusSignal}>
              <span className={panelStyles.statusSignalLabel}>{appShellText.apiStatus}</span>
              <strong className={panelStyles.statusSignalValue}>{healthStateText[consoleState.healthState]}</strong>
            </div>
            <div className={panelStyles.statusSignal}>
              <span className={panelStyles.statusSignalLabel}>{appShellText.platformStatus}</span>
              <strong className={panelStyles.statusSignalValue}>{bootStateText[consoleState.bootState]}</strong>
            </div>
            <div className={panelStyles.statusSignal}>
              <span className={panelStyles.statusSignalLabel}>{appShellText.supervisorStatus}</span>
              <strong className={panelStyles.statusSignalValue}>
                {consoleState.supervisorLoopStatus?.running ? appShellText.supervisorRunning : appShellText.supervisorStopped}
              </strong>
            </div>
          </>
        }
        metrics={<MetricStrip items={pageMetricItems} />}
      >
        {isCopilotPage ? (
          <CopilotWorkbench
            agentStatus={consoleState.agentStatus}
            agentTasks={consoleState.agentTasks}
            dailyReports={consoleState.dailyReports}
            episodeSummaries={consoleState.episodeSummaries}
            isBusy={consoleState.isBusy}
            latestAnswer={consoleState.latestAnswer}
            messages={consoleState.messages}
            pendingRegionalAnalysisPackage={consoleState.pendingRegionalAnalysisPackage}
            regionalAnalysisPackageHistory={consoleState.regionalAnalysisPackageHistory}
            pendingProposals={consoleState.pendingRegionalAnalysisPackage ? [] : consoleState.pendingProposals}
            priorityItems={priorityItems}
            selectedImpact={selectedImpact}
            selectedPriorityId={consoleState.selectedEntityId}
            onAsk={(prompt) => void consoleState.ask(prompt)}
            onOpenOperations={() => navigate("/operations")}
            onResolveRegionalAnalysisPackage={(packageId, decision, note) =>
              void consoleState.resolveRegionalAnalysisPackage(packageId, decision, note)
            }
            onResolveProposal={(proposalId, decision, note) => void consoleState.resolveProposal(proposalId, decision, note)}
            onSelectPriority={(id) => void consoleState.selectEntity(id)}
          />
        ) : null}

        {isOverviewPage ? (
          <DigitalTwinImpactScreen
            overview={consoleState.twinOverview}
            focusObject={consoleState.focusObject}
            pendingProposals={consoleState.pendingProposals}
            approvedProposals={consoleState.approvedProposals}
            hazardState={consoleState.hazardState}
            areaResourceStatusView={consoleState.areaResourceStatusView}
            eventResourceStatusView={consoleState.eventResourceStatusView}
            dialogEntries={consoleState.dialogEntries}
            dialogOpen={consoleState.dialogOpen}
            dialogBusy={consoleState.dialogBusy}
            streamStatus={consoleState.twinStreamStatus}
            onSelectObject={(objectId) => void consoleState.selectTwinObject(objectId)}
            onOpenDialog={() => consoleState.setDialogOpen(true)}
            onCloseDialog={() => consoleState.setDialogOpen(false)}
            onSendDialog={(message, objectId) => void consoleState.sendAgentDialog(message, objectId)}
            onGenerateProposals={() => void consoleState.generateTwinProposals()}
            onGenerateWarnings={(proposalId) => void consoleState.generateAudienceWarnings(proposalId)}
            onResolveProposal={(proposalId, decision, note) => void consoleState.resolveProposal(proposalId, decision, note)}
            onOpenProposalQueue={consoleState.openProposalQueue}
            onOpenOperations={() => navigate("/operations")}
            actionBusy={consoleState.isBusy}
            twinBusy={consoleState.twinBusy}
          />
        ) : null}

        {isOperationsPage ? (
          <OperationsWorkbench
            executionFlowSteps={executionFlowSteps}
            latestAnswer={consoleState.latestAnswer}
            pendingProposalCount={consoleState.pendingProposals.length}
            proposalHistoryItems={proposalHistoryItems}
            pendingProposalItems={pendingProposalItems}
            analysisPackageItems={analysisPackageItems}
            agentTimelineItems={agentTimelineItems}
            latestWarningDrafts={latestWarningDrafts}
            activeAdvisory={consoleState.activeAdvisory}
            toolExecutionCount={latestToolExecutions.length}
          />
        ) : null}

        {isDataPage ? (
          <DataPage>
            <div className={styles.panelFrame}>
              <AdminDesk
                areaId={consoleState.event?.area_id ?? "beilin_10km2"}
                eventId={consoleState.event?.event_id}
                profiles={consoleState.managedProfiles}
                areaResourceStatusView={consoleState.areaResourceStatusView}
                eventResourceStatusView={consoleState.eventResourceStatusView}
                ragDocuments={consoleState.ragDocuments}
                datasetStatus={consoleState.datasetStatus}
                busy={consoleState.adminBusy || consoleState.isBusy}
                status={consoleState.adminStatus}
                canEditRuntimeAdmin={canEditRuntimeAdmin}
                canManageDataset={canManageDataset}
                onSaveProfile={consoleState.saveManagedProfile}
                onDeleteProfile={consoleState.deleteManagedProfile}
                onInspectProfile={consoleState.selectEntity}
                onSaveAreaResources={consoleState.saveAreaResources}
                onSaveEventResources={consoleState.saveEventResources}
                onClearEventResources={consoleState.clearEventResources}
                onImportRagDocuments={consoleState.importRagDocuments}
                onReloadRagDocuments={consoleState.reloadRagDocuments}
                onFetchDatasetSources={consoleState.fetchDatasetSources}
                onRetryDatasetSource={consoleState.retryDatasetSource}
                onBuildDatasetPackage={consoleState.buildDatasetPackage}
                onValidateDatasetPackage={consoleState.validateDatasetPackage}
                onSyncDatasetPackage={consoleState.syncDatasetPackage}
                onCancelDatasetJob={consoleState.cancelDatasetJob}
                onRetryDatasetJob={consoleState.retryDatasetJob}
              />
            </div>
          </DataPage>
        ) : null}

        {isAgentsPage ? (
          <AgentsWorkbench
            agentCouncil={consoleState.agentCouncil}
            eventId={consoleState.event?.event_id}
            agentStatus={consoleState.agentStatus}
            agentTasks={consoleState.agentTasks}
            sessionMemoryView={consoleState.sessionMemoryView}
            sharedMemorySnapshot={consoleState.sharedMemorySnapshot}
            episodeSummaries={consoleState.episodeSummaries}
            triggerEvents={consoleState.triggerEvents}
            agentTimeline={consoleState.agentTimeline}
            agentTimelineItems={agentTimelineItems}
            supervisorRuns={consoleState.supervisorRuns}
            supervisorLoopStatus={consoleState.supervisorLoopStatus}
            recentAgentResults={consoleState.recentAgentResults}
            experienceContext={consoleState.experienceContext}
            decisionReport={consoleState.decisionReport}
            agentMetrics={consoleState.agentMetrics}
            evaluationBenchmarks={consoleState.evaluationBenchmarks}
            latestEvaluationReport={consoleState.latestEvaluationReport}
            busy={consoleState.isBusy}
            canControlSupervisor={canControlSupervisor}
            canReplayTask={canReplayTask}
            canRunEvaluation={canRunEvaluation}
            onRunSupervisor={consoleState.runSupervisorNow}
            onTickSupervisor={consoleState.tickSupervisor}
            onReplayTask={consoleState.replayAgentTask}
            onRunEvaluation={consoleState.runEvaluation}
            onReplayEvaluationReport={consoleState.replayEvaluationReport}
            pendingProposalCount={pendingProposalCount}
            approvedProposalCount={approvedProposalCount}
            warningDraftCount={warningDraftCount}
            latestWarningDrafts={latestWarningDrafts}
          />
        ) : null}

        {isReliabilityPage ? (
          <ReliabilityWorkbench
            agentCouncil={consoleState.agentCouncil}
            eventId={consoleState.event?.event_id}
            supervisorLoopStatus={consoleState.supervisorLoopStatus}
            alerts={consoleState.openAlerts}
            auditRecords={consoleState.auditRecords}
            archiveStatus={consoleState.archiveStatus}
            busy={consoleState.reliabilityBusy || consoleState.isBusy}
            canRunArchive={canRunArchive}
            onQueryAudit={consoleState.queryAuditRecords}
            onRunArchive={consoleState.runArchiveCycle}
            operatorRole={consoleState.operatorRole}
            operatorCapabilities={consoleState.operatorCapabilities}
            onChangeRole={consoleState.setOperatorRole}
            pendingProposalCount={pendingProposalCount}
            approvedProposalCount={approvedProposalCount}
            warningDraftCount={warningDraftCount}
            latestWarningDrafts={latestWarningDrafts}
            twinStreamStatus={consoleState.twinStreamStatus}
          />
        ) : null}
      </AppShell>
    </>
  );
}



