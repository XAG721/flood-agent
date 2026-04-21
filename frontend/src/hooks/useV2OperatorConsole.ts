import { useEffect, useMemo, useRef, useState } from "react";
import { curatedEntities } from "../data/v2ConsoleSeed";
import { api, setApiOperatorContext } from "../lib/api";
import type {
  Advisory,
  AgentResult,
  AgentMetricsView,
  AgentStatusView,
  AgentTask,
  AgentTimelineEntry,
  ArchiveStatusView,
  AuditRecord,
  DecisionReportView,
  DatasetPipelineStatusView,
  EntityImpactView,
  EntityProfile,
  EvaluationBenchmark,
  EvaluationReport,
  HazardStateV2,
  OperatorCapabilitiesView,
  OperatorRole,
  OperationalAlert,
  RAGDocument,
  RegionalAnalysisPackageView,
  RegionalProposalQueueSnapshot,
  RegionalProposalView,
  ResourceStatus,
  ResourceStatusView,
  RiskLevel,
  SessionMemoryView,
  SharedMemorySnapshot,
  SupervisorLoopStatus,
  SupervisorRunRecord,
  TriggerEvent,
  ExperienceContextView,
  V2CopilotSessionView,
  V2EventRecord,
} from "../types/api";
import type { ConsoleBootState, ExecutionStatus } from "../types/ui";
import {
  getErrorMessage,
  persistV2,
  riskPriority,
  useConsoleAdminActions,
  useConsoleAgentActions,
  useConsoleBootstrapActions,
  useConsoleRefreshActions,
  useConsoleReliabilityActions,
} from "./useV2OperatorConsoleSections";

const V2_SESSION_STORAGE_KEY = "activeV2CopilotSessionId";
const V2_EVENT_STORAGE_KEY = "activeV2EventId";
const FRONTEND_OPERATOR_ID = "frontend_console";
const FRONTEND_OPERATOR_ROLE: OperatorRole = "commander";
const DEFAULT_AREA_ID = "beilin_10km2";

type HealthState = "checking" | "online" | "offline";
type ProposalDecision = "approve" | "reject";
type ProposalStreamStatus = "closed" | "connecting" | "open" | "error";

export function useV2OperatorConsole() {
  const [bootState, setBootState] = useState<ConsoleBootState>("booting");
  const [healthState, setHealthState] = useState<HealthState>("checking");
  const [executionStatus, setExecutionStatus] = useState<ExecutionStatus>("planning");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sessionView, setSessionView] = useState<V2CopilotSessionView | null>(null);
  const [hazardState, setHazardState] = useState<HazardStateV2 | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState<string>(curatedEntities[0]?.id ?? "");
  const [operatorRole, setOperatorRole] = useState<OperatorRole>(FRONTEND_OPERATOR_ROLE);
  const [operatorCapabilities, setOperatorCapabilities] = useState<OperatorCapabilitiesView | null>(null);
  const [entityImpacts, setEntityImpacts] = useState<Record<string, EntityImpactView>>({});
  const [activeAdvisory, setActiveAdvisory] = useState<Advisory | null>(null);
  const [managedProfiles, setManagedProfiles] = useState<EntityProfile[]>([]);
  const [areaResourceStatusView, setAreaResourceStatusView] = useState<ResourceStatusView | null>(null);
  const [eventResourceStatusView, setEventResourceStatusView] = useState<ResourceStatusView | null>(null);
  const [ragDocuments, setRagDocuments] = useState<RAGDocument[]>([]);
  const [datasetStatus, setDatasetStatus] = useState<DatasetPipelineStatusView | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatusView | null>(null);
  const [agentTasks, setAgentTasks] = useState<AgentTask[]>([]);
  const [sharedMemorySnapshot, setSharedMemorySnapshot] = useState<SharedMemorySnapshot | null>(null);
  const [sessionMemoryView, setSessionMemoryView] = useState<SessionMemoryView | null>(null);
  const [supervisorRuns, setSupervisorRuns] = useState<SupervisorRunRecord[]>([]);
  const [supervisorLoopStatus, setSupervisorLoopStatus] = useState<SupervisorLoopStatus | null>(null);
  const [triggerEvents, setTriggerEvents] = useState<TriggerEvent[]>([]);
  const [agentTimeline, setAgentTimeline] = useState<AgentTimelineEntry[]>([]);
  const [recentAgentResults, setRecentAgentResults] = useState<AgentResult[]>([]);
  const [experienceContext, setExperienceContext] = useState<ExperienceContextView | null>(null);
  const [decisionReport, setDecisionReport] = useState<DecisionReportView | null>(null);
  const [agentMetrics, setAgentMetrics] = useState<AgentMetricsView | null>(null);
  const [evaluationBenchmarks, setEvaluationBenchmarks] = useState<EvaluationBenchmark[]>([]);
  const [latestEvaluationReport, setLatestEvaluationReport] = useState<EvaluationReport | null>(null);
  const [openAlerts, setOpenAlerts] = useState<OperationalAlert[]>([]);
  const [auditRecords, setAuditRecords] = useState<AuditRecord[]>([]);
  const [archiveStatus, setArchiveStatus] = useState<ArchiveStatusView | null>(null);
  const [regionalProposalHistory, setRegionalProposalHistory] = useState<RegionalProposalView[]>([]);
  const [regionalProposalQueueSnapshot, setRegionalProposalQueueSnapshot] = useState<RegionalProposalQueueSnapshot | null>(null);
  const [pendingRegionalAnalysisPackage, setPendingRegionalAnalysisPackage] = useState<RegionalAnalysisPackageView | null>(null);
  const [regionalAnalysisPackageHistory, setRegionalAnalysisPackageHistory] = useState<RegionalAnalysisPackageView[]>([]);
  const [regionalProposalModalOpen, setRegionalProposalModalOpen] = useState(false);
  const [regionalProposalSnoozedUntil, setRegionalProposalSnoozedUntil] = useState<number | null>(null);
  const [proposalStreamStatus, setProposalStreamStatus] = useState<ProposalStreamStatus>("closed");
  const [adminStatus, setAdminStatus] = useState<string | null>(null);
  const [adminBusy, setAdminBusy] = useState(false);
  const [reliabilityBusy, setReliabilityBusy] = useState(false);
  const proposalStreamRef = useRef<EventSource | null>(null);
  const regionalProposalSnoozedUntilRef = useRef<number | null>(null);

  const event = sessionView?.event ?? null;
  const messages = sessionView?.messages ?? [];
  const latestAnswer = sessionView?.latest_answer ?? null;
  const dailyReports = sessionView?.daily_reports ?? [];
  const episodeSummaries = sessionView?.episode_summaries ?? [];


  const {
    hydrateEvent,
    refreshAdminData,
    refreshRegionalProposalData,
    applyRegionalProposalSnapshot,
    refreshDatasetStatusOnly,
    refreshAgentMeshData,
    refreshReliabilityData,
    refreshOperatorCapabilities,
  } = useConsoleRefreshActions({
    sessionView,
    recentAgentResults,
    regionalProposalQueueVersion: regionalProposalQueueSnapshot?.queue_version,
    regionalProposalSnoozedUntilRef,
    setErrorMessage,
    setHazardState,
    setEntityImpacts,
    setSelectedEntityId,
    setManagedProfiles,
    setAreaResourceStatusView,
    setEventResourceStatusView,
    setRagDocuments,
    setDatasetStatus,
    setAgentStatus,
    setAgentTasks,
    setSharedMemorySnapshot,
    setSessionMemoryView,
    setSupervisorRuns,
    setSupervisorLoopStatus,
    setTriggerEvents,
    setAgentTimeline,
    setRecentAgentResults,
    setExperienceContext,
    setDecisionReport,
    setAgentMetrics,
    setEvaluationBenchmarks,
    setOpenAlerts,
    setAuditRecords,
    setArchiveStatus,
    setRegionalProposalHistory,
    setRegionalProposalQueueSnapshot,
    setPendingRegionalAnalysisPackage,
    setRegionalAnalysisPackageHistory,
    setRegionalProposalModalOpen,
    setRegionalProposalSnoozedUntil,
    setReliabilityBusy,
    operatorRole,
  });

  const { bootstrap } = useConsoleBootstrapActions({
    sessionStorageKey: V2_SESSION_STORAGE_KEY,
    eventStorageKey: V2_EVENT_STORAGE_KEY,
    defaultAreaId: DEFAULT_AREA_ID,
    frontendOperatorId: FRONTEND_OPERATOR_ID,
    operatorRole,
    setBootState,
    setHealthState,
    setExecutionStatus,
    setErrorMessage,
    setAdminStatus,
    setSessionView,
    hydrateEvent,
    refreshAdminData,
    refreshRegionalProposalData,
    refreshAgentMeshData,
    refreshReliabilityData,
  });

  async function refresh() {
    const currentSessionId = sessionView?.session_id;
    const currentEvent = sessionView?.event;
    if (!currentSessionId || !currentEvent) {
      await bootstrap();
      return;
    }

    setExecutionStatus("planning");
    setErrorMessage(null);
    try {
      const nextSession = await api.getV2CopilotSession(currentSessionId);
      setSessionView(nextSession);
      persistV2(V2_SESSION_STORAGE_KEY, V2_EVENT_STORAGE_KEY, nextSession.session_id, nextSession.event.event_id);
      await Promise.all([
        hydrateEvent(nextSession.event ?? currentEvent),
        refreshAdminData(nextSession.event ?? currentEvent),
        refreshRegionalProposalData(nextSession.event ?? currentEvent),
        refreshAgentMeshData(nextSession.event ?? currentEvent, nextSession),
        refreshReliabilityData(nextSession.event ?? currentEvent),
      ]);
      setExecutionStatus("idle");
      setBootState("ready");
    } catch (error) {
      setBootState("degraded");
      setExecutionStatus("error");
      setErrorMessage(getErrorMessage(error, "刷新 V2 控制台失败。"));
    }
  }
  async function ask(content: string) {
    const currentSessionId = sessionView?.session_id;
    if (!currentSessionId) {
      await bootstrap();
      return;
    }

    setExecutionStatus("running");
    setErrorMessage(null);
    try {
      const nextSession = await api.sendV2CopilotMessage(currentSessionId, content);
      setSessionView(nextSession);
      await Promise.all([refreshAgentMeshData(nextSession.event, nextSession), refreshReliabilityData(nextSession.event)]);
      setExecutionStatus("idle");
    } catch (error) {
      setBootState("degraded");
      setExecutionStatus("error");
      setErrorMessage(getErrorMessage(error, "智能问答请求失败。"));
    }
  }

  async function selectEntity(entityId: string) {
    setSelectedEntityId(entityId);
    setActiveAdvisory((current) => (current?.entity_id === entityId ? current : null));

    if (entityImpacts[entityId]) {
      return;
    }

    const eventId = sessionView?.event.event_id;
    if (!eventId) {
      return;
    }

    try {
      const impact = await api.getV2EntityImpact(entityId, eventId);
      setEntityImpacts((current) => ({ ...current, [entityId]: impact }));
    } catch (error) {
      setErrorMessage(getErrorMessage(error, "加载对象影响评估失败。"));
    }
  }

  async function generateAdvisory(entityId: string) {
    const eventId = sessionView?.event.event_id;
    const areaId = sessionView?.event.area_id ?? DEFAULT_AREA_ID;
    if (!eventId) {
      return;
    }

    setExecutionStatus("running");
    setErrorMessage(null);
    try {
      const advisory = await api.generateV2Advisory({
        event_id: eventId,
        area_id: areaId,
        entity_id: entityId,
        operator_role: operatorRole,
      });
      setActiveAdvisory(advisory);
      await refreshAgentMeshData(sessionView?.event ?? undefined);
      setExecutionStatus("idle");
    } catch (error) {
      setExecutionStatus("error");
      setErrorMessage(getErrorMessage(error, "生成处置建议失败。"));
    }
  }

  async function resolveProposal(
    proposalId: string,
    decision: ProposalDecision,
    note: string,
  ) {
    const currentEvent = sessionView?.event;
    if (!currentEvent) {
      return;
    }

    setExecutionStatus("running");
    setErrorMessage(null);
    try {
      const payload = {
        operator_id: FRONTEND_OPERATOR_ID,
        operator_role: operatorRole,
        note,
      };
      if (decision === "approve") {
        await api.approveRegionalProposal(proposalId, payload);
      } else {
        await api.rejectRegionalProposal(proposalId, payload);
      }
      await Promise.all([
        refreshRegionalProposalData(currentEvent),
        refreshAgentMeshData(currentEvent, sessionView),
        refreshReliabilityData(currentEvent),
      ]);
      setExecutionStatus("idle");
    } catch (error) {
      setExecutionStatus("error");
      setErrorMessage(getErrorMessage(error, "处理区域请示失败。"));
    }
  }

  async function resolveRegionalAnalysisPackage(
    packageId: string,
    decision: ProposalDecision,
    note: string,
  ) {
    const currentEvent = sessionView?.event;
    if (!currentEvent) {
      return;
    }

    setExecutionStatus("running");
    setErrorMessage(null);
    try {
      const payload = {
        operator_id: FRONTEND_OPERATOR_ID,
        operator_role: operatorRole,
        note,
      };
      if (decision === "approve") {
        await api.approveRegionalAnalysisPackage(packageId, payload);
      } else {
        await api.rejectRegionalAnalysisPackage(packageId, payload);
      }
      await Promise.all([
        refreshRegionalProposalData(currentEvent),
        refreshAgentMeshData(currentEvent, sessionView),
        refreshReliabilityData(currentEvent),
      ]);
      setExecutionStatus("idle");
    } catch (error) {
      setExecutionStatus("error");
      setErrorMessage(getErrorMessage(error, "处理区域分析包失败。"));
    }
  }

  async function resolveProposalBatch(
    proposalIds: string[],
    decision: ProposalDecision,
    note: string,
  ) {
    const currentEvent = sessionView?.event;
    if (!currentEvent || !proposalIds.length) {
      return;
    }

    setExecutionStatus("running");
    setErrorMessage(null);
    try {
      for (const proposalId of proposalIds) {
        // Keep batch support as a thin wrapper even though the new workflow emphasizes per-item confirmation.
        // eslint-disable-next-line no-await-in-loop
        await resolveProposal(proposalId, decision, note);
      }
      await Promise.all([
        refreshRegionalProposalData(currentEvent),
        refreshAgentMeshData(currentEvent, sessionView),
        refreshReliabilityData(currentEvent),
      ]);
      setExecutionStatus("idle");
    } catch (error) {
      setExecutionStatus("error");
      setErrorMessage(getErrorMessage(error, "批量处理区域请示失败。"));
    }
  }

  async function updateRegionalProposalDraft(
    proposalId: string,
    actionScope: Record<string, unknown>,
  ) {
    const currentEvent = sessionView?.event;
    if (!currentEvent) {
      return;
    }

    setExecutionStatus("running");
    setErrorMessage(null);
    try {
      await api.updateRegionalProposalDraft(proposalId, {
        operator_id: FRONTEND_OPERATOR_ID,
        operator_role: operatorRole,
        action_scope: actionScope,
      });
      await Promise.all([
        refreshRegionalProposalData(currentEvent),
        refreshReliabilityData(currentEvent),
      ]);
      setExecutionStatus("idle");
    } catch (error) {
      setExecutionStatus("error");
      setErrorMessage(getErrorMessage(error, "更新区域请示草稿失败。"));
    }
  }

  function snoozeRegionalProposalModal() {
    setRegionalProposalModalOpen(false);
    setRegionalProposalSnoozedUntil(Date.now() + 5 * 60 * 1000);
  }


  const {
    saveManagedProfile,
    deleteManagedProfile,
    saveAreaResources,
    saveEventResources,
    clearEventResources,
    importRagDocuments,
    reloadRagDocuments,
    fetchDatasetSources,
    retryDatasetSource,
    buildDatasetPackage,
    validateDatasetPackage,
    syncDatasetPackage,
    cancelDatasetJob,
    retryDatasetJob,
  } = useConsoleAdminActions({
    sessionView,
    operatorRole,
    frontendOperatorId: FRONTEND_OPERATOR_ID,
    defaultAreaId: DEFAULT_AREA_ID,
    managedProfiles,
    selectedEntityId,
    setAdminBusy,
    setAdminStatus,
    setErrorMessage,
    setManagedProfiles,
    setAreaResourceStatusView,
    setEventResourceStatusView,
    setRagDocuments,
    setDatasetStatus,
    setEntityImpacts,
    setSelectedEntityId,
    hydrateEvent,
    refreshAdminData,
    refreshAgentMeshData,
    refreshReliabilityData,
    refreshDatasetStatusOnly,
  });

  const {
    runSupervisorNow,
    tickSupervisor,
    replayAgentTask,
    runEvaluation,
    replayEvaluationReport,
  } = useConsoleAgentActions({
    sessionView,
    setExecutionStatus,
    setErrorMessage,
    setSupervisorRuns,
    setLatestEvaluationReport,
    hydrateEvent,
    refreshAgentMeshData,
    refreshReliabilityData,
  });

  const { queryAuditRecords, runArchiveCycle } = useConsoleReliabilityActions({
    event,
    setReliabilityBusy,
    setArchiveStatus,
    setAdminStatus,
    setErrorMessage,
    refreshReliabilityData,
  });

  useEffect(() => {
    setApiOperatorContext({ id: FRONTEND_OPERATOR_ID, role: operatorRole });
  }, [operatorRole]);

  useEffect(() => {
    regionalProposalSnoozedUntilRef.current = regionalProposalSnoozedUntil;
  }, [regionalProposalSnoozedUntil]);

  useEffect(() => {
    void refreshOperatorCapabilities(operatorRole).then((capabilities) => {
      setOperatorCapabilities(capabilities);
    });
  }, [operatorRole]);

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    if (proposalStreamRef.current) {
      proposalStreamRef.current.close();
      proposalStreamRef.current = null;
    }
    setProposalStreamStatus("connecting");
    const stream = api.openRegionalProposalStream({
      onSnapshot: (snapshot) => {
        setProposalStreamStatus("open");
        applyRegionalProposalSnapshot(snapshot);
      },
      onError: () => {
        setProposalStreamStatus("error");
      },
    });
    proposalStreamRef.current = stream;
    return () => {
      stream.close();
      proposalStreamRef.current = null;
      setProposalStreamStatus("closed");
    };
  }, [operatorRole]);

  useEffect(() => {
    if (!regionalProposalSnoozedUntil || regionalProposalModalOpen) {
      return;
    }
    const delay = regionalProposalSnoozedUntil - Date.now();
    if (delay <= 0) {
      if (regionalProposalQueueSnapshot?.items.length) {
        setRegionalProposalModalOpen(true);
      }
      setRegionalProposalSnoozedUntil(null);
      return;
    }
    const timer = window.setTimeout(() => {
      if (regionalProposalQueueSnapshot?.items.length) {
        setRegionalProposalModalOpen(true);
      }
      setRegionalProposalSnoozedUntil(null);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [regionalProposalModalOpen, regionalProposalQueueSnapshot?.queue_version, regionalProposalSnoozedUntil]);

  useEffect(() => {
    if (!datasetStatus?.active_job || !["pending", "running", "cancel_requested"].includes(datasetStatus.active_job.status)) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshDatasetStatusOnly().then((nextStatus) => {
        const activeJob = nextStatus?.active_job;
        if (!activeJob) {
          return;
        }
        setAdminStatus(activeJob.message || activeJob.result_summary || null);
        if (activeJob.status === "completed" && sessionView?.event) {
          void refreshAdminData(sessionView.event);
        }
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [datasetStatus?.active_job?.job_id, datasetStatus?.active_job?.status, sessionView?.event?.event_id]);

  const selectedImpact = selectedEntityId ? entityImpacts[selectedEntityId] ?? null : null;
  const topImpacts = useMemo(
    () =>
      Object.values(entityImpacts).sort(
        (left, right) =>
          riskPriority(right.risk_level) - riskPriority(left.risk_level) ||
          left.time_to_impact_minutes - right.time_to_impact_minutes ||
          right.entity.current_occupancy - left.entity.current_occupancy,
      ),
    [entityImpacts],
  );

  const askForEntity = async (entityId: string) => {
    const curatedDescriptor = curatedEntities.find((item) => item.id === entityId);
    const managedDescriptor = managedProfiles.find((item) => item.entity_id === entityId);
    const targetName = curatedDescriptor?.name ?? managedDescriptor?.name;
    if (!targetName) {
      return;
    }
    await ask(`当前洪水态势对 ${targetName} 意味着什么？`);
  };

  const proposals = useMemo(
    () => regionalProposalHistory.map((item) => item.proposal),
    [regionalProposalHistory],
  );

  const pendingProposals = useMemo(
    () => regionalProposalQueueSnapshot?.items.map((item) => item.proposal) ?? [],
    [regionalProposalQueueSnapshot],
  );

  return {
    bootState,
    healthState,
    executionStatus,
    errorMessage,
    operatorRole,
    operatorCapabilities,
    event,
    hazardState,
    latestAnswer,
    messages,
    dailyReports,
    episodeSummaries,
    selectedEntityId,
    selectedImpact,
    entityImpacts,
    topImpacts,
    activeAdvisory,
    proposals,
    pendingProposals,
    curatedEntities,
    managedProfiles,
    areaResourceStatusView,
    eventResourceStatusView,
    ragDocuments,
    datasetStatus,
    agentStatus,
    agentTasks,
    sharedMemorySnapshot,
    sessionMemoryView,
    triggerEvents,
    agentTimeline,
    supervisorRuns,
    supervisorLoopStatus,
    recentAgentResults,
    experienceContext,
    decisionReport,
    agentMetrics,
    evaluationBenchmarks,
    latestEvaluationReport,
    openAlerts,
    auditRecords,
    archiveStatus,
    regionalProposalHistory,
    regionalProposalQueueSnapshot,
    pendingRegionalAnalysisPackage,
    regionalAnalysisPackageHistory,
    regionalProposalModalOpen,
    regionalProposalSnoozedUntil,
    proposalStreamStatus,
    reliabilityBusy,
    adminStatus,
    adminBusy,
    isBusy: ["planning", "running"].includes(executionStatus) || bootState === "booting",
    ask,
    askForEntity,
    bootstrap,
    refresh,
    refreshAdminData,
    refreshRegionalProposalData,
    refreshReliabilityData,
    selectEntity,
    generateAdvisory,
    resolveProposal,
    resolveRegionalAnalysisPackage,
    resolveProposalBatch,
    updateRegionalProposalDraft,
    setRegionalProposalModalOpen,
    snoozeRegionalProposalModal,
    saveManagedProfile,
    deleteManagedProfile,
    saveAreaResources,
    saveEventResources,
    clearEventResources,
    importRagDocuments,
    reloadRagDocuments,
    fetchDatasetSources,
    retryDatasetSource,
    buildDatasetPackage,
    validateDatasetPackage,
    syncDatasetPackage,
    cancelDatasetJob,
    retryDatasetJob,
    runSupervisorNow,
    tickSupervisor,
    replayAgentTask,
    runEvaluation,
    replayEvaluationReport,
    queryAuditRecords,
    runArchiveCycle,
    setOperatorRole,
  };
}
