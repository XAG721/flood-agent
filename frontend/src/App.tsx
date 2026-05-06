import { Navigate, useLocation, useNavigate } from "react-router-dom";
import styles from "./App.module.css";
import { AppShell } from "./components/AppShell";
import { DigitalTwinImpactScreen } from "./components/DigitalTwinImpactScreen";
import { GlobalRegionalProposalDialog } from "./components/GlobalRegionalProposalDialog";
import { MetricStrip } from "./components/MetricStrip";
import { operatorRoleText } from "./components/SecurityDesk";
import {
  bootStateText,
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
  formatTrendLabel,
  formatPendingMetricHint,
  overviewMetricText,
} from "./lib/appText";
import { AdminDesk } from "./features/dataManagement/AdminDesk";
import { CopilotTwinScreen } from "./features/copilot/CopilotTwinScreen";
import { formatProposalStreamStatus } from "./lib/displayText";
import type { OperatorRole } from "./types/api";

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const consoleState = useAgentTwinConsole();
  const capabilityMap = consoleState.operatorCapabilities?.capabilities ?? {};
  const canEditRuntimeAdmin = Boolean(capabilityMap.runtime_admin_write);
  const canManageDataset = Boolean(capabilityMap.dataset_manage);

  const topRisk = consoleState.twinOverview?.overall_risk_level ?? consoleState.hazardState?.overall_risk_level ?? "None";
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
  const isTwinDashboardPage = isOverviewPage || isOperationsPage || isAgentsPage || isReliabilityPage;
  const isImmersivePage = isTwinDashboardPage || isCopilotPage;

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

  const primaryPaths = new Set(["/", "/copilot", "/operations", "/agents"]);
  const navigation = Object.entries(pageMeta)
    .filter(([path]) => primaryPaths.has(path))
    .map(([path, meta]) => ({ path, label: meta.label }));
  const utilityNavigation = Object.entries(pageMeta)
    .filter(([path]) => !primaryPaths.has(path))
    .map(([path, meta]) => ({ path, label: meta.label }));
  const shellCurrentPageLabel = isOverviewPage || isCopilotPage ? "数字孪生主屏" : currentPage.label;
  const shellCurrentPageTitle = isOverviewPage || isCopilotPage ? "数字孪生智能体洪水预警系统" : currentPage.title;
  const shellCurrentPageDescription = isOverviewPage || isCopilotPage ? undefined : currentPage.description;

  const pageMetricItems = isOverviewPage || isCopilotPage
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
        immersive={isImmersivePage}
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
            {consoleState.demoMode ? (
              <div className={panelStyles.statusSignal}>
                <span className={panelStyles.statusSignalLabel}>演示模式</span>
                <strong className={panelStyles.statusSignalValue}>已锁定</strong>
              </div>
            ) : null}
          </>
        }
        metrics={<MetricStrip items={pageMetricItems} />}
      >
        {isCopilotPage ? (
          <CopilotTwinScreen
            overview={consoleState.twinOverview}
            focusObject={consoleState.focusObject}
            dialogEntries={consoleState.dialogEntries}
            busy={consoleState.dialogBusy}
            onSelectObject={(objectId) => void consoleState.selectTwinObject(objectId)}
            onAsk={(prompt) => void consoleState.sendAgentDialog(prompt)}
          />
        ) : null}

        {isTwinDashboardPage ? (
          <DigitalTwinImpactScreen
            variant={
              isOperationsPage
                ? "risk-warning"
                : isAgentsPage
                  ? "impact-analysis"
                  : isReliabilityPage
                    ? "event-replay"
                    : "overview"
            }
            overview={consoleState.twinOverview}
            focusObject={consoleState.focusObject}
            pendingProposals={consoleState.pendingProposals}
            approvedProposals={consoleState.approvedProposals}
            hazardState={consoleState.hazardState}
            areaResourceStatusView={consoleState.areaResourceStatusView}
            eventResourceStatusView={consoleState.eventResourceStatusView}
            dialogEntries={consoleState.dialogEntries}
            streamStatus={consoleState.twinStreamStatus}
            onSelectObject={(objectId) => void consoleState.selectTwinObject(objectId)}
            onGenerateProposals={() => void consoleState.generateTwinProposals()}
            onGenerateWarnings={(proposalId) => void consoleState.generateAudienceWarnings(proposalId)}
            onResolveProposal={(proposalId, decision, note) => void consoleState.resolveProposal(proposalId, decision, note)}
            onOpenOperations={() => navigate("/operations")}
            actionBusy={consoleState.isBusy}
            twinBusy={consoleState.twinBusy}
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

      </AppShell>
    </>
  );
}



