import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import { curatedEntities, platformSeedObservations } from "../data/platformConsoleSeed";
import { api } from "../lib/api";
import { formatDatasetAction } from "../lib/datasetUiText";
import type {
  AgentMetricsView,
  AgentResult,
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
  MemoryBundleView,
  OperationalAlert,
  OperatorCapabilitiesView,
  OperatorRole,
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

type Setter<T> = Dispatch<SetStateAction<T>>;

export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}

export function persistPlatformSession(
  sessionStorageKey: string,
  eventStorageKey: string,
  sessionId: string | null,
  eventId: string | null,
) {
  if (sessionId) {
    window.sessionStorage.setItem(sessionStorageKey, sessionId);
  } else {
    window.sessionStorage.removeItem(sessionStorageKey);
  }
  if (eventId) {
    window.sessionStorage.setItem(eventStorageKey, eventId);
  } else {
    window.sessionStorage.removeItem(eventStorageKey);
  }
}

export function riskPriority(level: RiskLevel) {
  return { None: 0, Blue: 1, Yellow: 2, Orange: 3, Red: 4 }[level];
}

export async function loadAdminDataset(eventId: string, areaId: string) {
  const [managedProfiles, areaResourceStatusView, eventResourceStatusView, ragDocuments, datasetStatus] = await Promise.all([
    api.listV2EntityProfiles({ areaId }),
    api.getAreaResourceStatus(areaId),
    api.getEventResourceStatus(eventId),
    api.listV2RagDocuments(),
    api.getV2DatasetStatus(),
  ]);
  return { managedProfiles, areaResourceStatusView, eventResourceStatusView, ragDocuments, datasetStatus };
}

export async function loadAgentDataset(eventId: string, sessionId?: string) {
  const [agentStatus, agentTasks, sharedMemorySnapshot, supervisorRuns, supervisorLoopStatus, triggerEvents, agentTimeline, sessionMemoryBundle, experienceContext, decisionReport, agentMetrics, evaluationBenchmarks] = await Promise.all([
    api.getV2AgentStatus(eventId),
    api.listV2AgentTasks(eventId),
    api.getV2SharedMemory(eventId),
    api.listV2SupervisorRuns(eventId),
    api.getV2SupervisorStatus(),
    api.listV2TriggerEvents(eventId),
    api.listV2AgentTimeline(eventId),
    sessionId ? api.getV2SessionMemory(sessionId) : Promise.resolve<MemoryBundleView>({}),
    api.getV2ExperienceContext(eventId),
    api.getV2DecisionReport(eventId),
    api.getV2AgentMetrics(),
    api.listV2EvaluationBenchmarks(),
  ]);
  return {
    agentStatus,
    agentTasks,
    sharedMemorySnapshot,
    supervisorRuns,
    supervisorLoopStatus,
    triggerEvents,
    agentTimeline,
    sessionMemoryBundle,
    experienceContext,
    decisionReport,
    agentMetrics,
    evaluationBenchmarks,
  };
}

export async function loadReliabilityDataset(eventId?: string) {
  const [alerts, auditRecords, archiveStatus] = await Promise.all([
    api.listV2Alerts({ eventId, limit: 8 }),
    api.listV2AuditRecords({ eventId, limit: 12 }),
    api.getV2ArchiveStatus(),
  ]);
  return { alerts, auditRecords, archiveStatus };
}

export function shouldOpenRegionalProposalModal(
  snapshot: RegionalProposalQueueSnapshot,
  snoozedUntil: number | null,
  previousQueueVersion?: string | null,
) {
  if (!snapshot.items.length) {
    return false;
  }
  if (!snoozedUntil || Date.now() >= snoozedUntil) {
    return true;
  }
  return !!previousQueueVersion && previousQueueVersion !== snapshot.queue_version;
}

type RefreshParams = {
  sessionView: V2CopilotSessionView | null;
  recentAgentResults: AgentResult[];
  regionalProposalQueueVersion?: string | null;
  regionalProposalSnoozedUntilRef: MutableRefObject<number | null>;
  setErrorMessage: Setter<string | null>;
  setHazardState: Setter<HazardStateV2 | null>;
  setEntityImpacts: Setter<Record<string, EntityImpactView>>;
  setSelectedEntityId: Setter<string>;
  setManagedProfiles: Setter<EntityProfile[]>;
  setAreaResourceStatusView: Setter<ResourceStatusView | null>;
  setEventResourceStatusView: Setter<ResourceStatusView | null>;
  setRagDocuments: Setter<RAGDocument[]>;
  setDatasetStatus: Setter<DatasetPipelineStatusView | null>;
  setAgentStatus: Setter<AgentStatusView | null>;
  setAgentTasks: Setter<AgentTask[]>;
  setSharedMemorySnapshot: Setter<SharedMemorySnapshot | null>;
  setSessionMemoryView: Setter<SessionMemoryView | null>;
  setSupervisorRuns: Setter<SupervisorRunRecord[]>;
  setSupervisorLoopStatus: Setter<SupervisorLoopStatus | null>;
  setTriggerEvents: Setter<TriggerEvent[]>;
  setAgentTimeline: Setter<AgentTimelineEntry[]>;
  setRecentAgentResults: Setter<AgentResult[]>;
  setExperienceContext: Setter<ExperienceContextView | null>;
  setDecisionReport: Setter<DecisionReportView | null>;
  setAgentMetrics: Setter<AgentMetricsView | null>;
  setEvaluationBenchmarks: Setter<EvaluationBenchmark[]>;
  setOpenAlerts: Setter<OperationalAlert[]>;
  setAuditRecords: Setter<AuditRecord[]>;
  setArchiveStatus: Setter<ArchiveStatusView | null>;
  setRegionalProposalHistory: Setter<RegionalProposalView[]>;
  setRegionalProposalQueueSnapshot: Setter<RegionalProposalQueueSnapshot | null>;
  setPendingRegionalAnalysisPackage: Setter<RegionalAnalysisPackageView | null>;
  setRegionalAnalysisPackageHistory: Setter<RegionalAnalysisPackageView[]>;
  setRegionalProposalModalOpen: Setter<boolean>;
  setRegionalProposalSnoozedUntil: Setter<number | null>;
  setReliabilityBusy: Setter<boolean>;
  operatorRole: OperatorRole;
};

export function useConsoleRefreshActions(params: RefreshParams) {
  const hydrateEvent = async (nextEvent: V2EventRecord) => {
    const [nextHazard, impacts] = await Promise.all([
      api.getV2HazardState(nextEvent.event_id),
      Promise.all(curatedEntities.map(async (entity) => [entity.id, await api.getV2EntityImpact(entity.id, nextEvent.event_id)] as const)),
    ]);
    params.setHazardState(nextHazard);
    params.setEntityImpacts(Object.fromEntries(impacts));
    const preferredEntityId = curatedEntities.find((entity) => impacts.some(([id]) => id === entity.id))?.id;
    if (preferredEntityId) {
      params.setSelectedEntityId(preferredEntityId);
    }
  };

  const refreshAdminData = async (nextEvent?: V2EventRecord) => {
    const currentEvent = nextEvent ?? params.sessionView?.event;
    if (!currentEvent) return;
    try {
      const adminData = await loadAdminDataset(currentEvent.event_id, currentEvent.area_id);
      params.setManagedProfiles(adminData.managedProfiles);
      params.setAreaResourceStatusView(adminData.areaResourceStatusView);
      params.setEventResourceStatusView(adminData.eventResourceStatusView);
      params.setRagDocuments(adminData.ragDocuments);
      params.setDatasetStatus(adminData.datasetStatus);
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "刷新管理视图失败。"));
    }
  };

  const refreshRegionalProposalData = async (nextEvent?: V2EventRecord) => {
    const currentEvent = nextEvent ?? params.sessionView?.event;
    if (!currentEvent) return;
    try {
      const [history, snapshot, pendingPackage, packageHistory] = await Promise.all([
        api.listRegionalProposalsByEvent(currentEvent.event_id),
        api.listPendingRegionalProposals(),
        api.getPendingRegionalAnalysisPackage(currentEvent.event_id),
        api.listRegionalAnalysisPackages(currentEvent.event_id, { includePending: false }),
      ]);
      const shouldOpenModal = shouldOpenRegionalProposalModal(
        snapshot,
        params.regionalProposalSnoozedUntilRef.current,
        params.regionalProposalQueueVersion,
      );
      params.setRegionalProposalHistory(history);
      params.setRegionalProposalQueueSnapshot(snapshot);
      params.setPendingRegionalAnalysisPackage(pendingPackage);
      params.setRegionalAnalysisPackageHistory(packageHistory);
      params.setRegionalProposalModalOpen(shouldOpenModal);
      if (!snapshot.items.length) {
        params.setRegionalProposalSnoozedUntil(null);
        params.regionalProposalSnoozedUntilRef.current = null;
      }
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "刷新区域请示队列失败。"));
    }
  };

  const applyRegionalProposalSnapshot = (snapshot: RegionalProposalQueueSnapshot) => {
    const shouldOpenModal = shouldOpenRegionalProposalModal(
      snapshot,
      params.regionalProposalSnoozedUntilRef.current,
      params.regionalProposalQueueVersion,
    );
    params.setRegionalProposalQueueSnapshot(snapshot);
    params.setRegionalProposalModalOpen(shouldOpenModal);
    if (!snapshot.items.length) {
      params.setRegionalProposalSnoozedUntil(null);
      params.regionalProposalSnoozedUntilRef.current = null;
    }
    if (params.sessionView?.event) {
      void refreshRegionalProposalData(params.sessionView.event);
    }
  };

  const refreshDatasetStatusOnly = async () => {
    try {
      const nextStatus = await api.getV2DatasetStatus();
      params.setDatasetStatus(nextStatus);
      return nextStatus;
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "刷新数据集状态失败。"));
      return null;
    }
  };

  const refreshAgentMeshData = async (nextEvent?: V2EventRecord, nextSession?: V2CopilotSessionView | null) => {
    const currentEvent = nextEvent ?? nextSession?.event ?? params.sessionView?.event;
    if (!currentEvent) return;
    try {
      const agentData = await loadAgentDataset(currentEvent.event_id, nextSession?.session_id ?? params.sessionView?.session_id);
      params.setAgentStatus(agentData.agentStatus);
      params.setAgentTasks(agentData.agentTasks);
      params.setSharedMemorySnapshot(
        nextSession?.shared_memory_snapshot ??
          agentData.sessionMemoryBundle.event_shared_memory ??
          agentData.sharedMemorySnapshot,
      );
      params.setSessionMemoryView(agentData.sessionMemoryBundle.session_memory ?? null);
      params.setSupervisorRuns(agentData.supervisorRuns);
      params.setSupervisorLoopStatus(agentData.supervisorLoopStatus);
      params.setTriggerEvents(agentData.triggerEvents);
      params.setAgentTimeline(agentData.agentTimeline);
      params.setRecentAgentResults(nextSession?.recent_agent_results ?? params.sessionView?.recent_agent_results ?? params.recentAgentResults);
      params.setExperienceContext(agentData.experienceContext);
      params.setDecisionReport(agentData.decisionReport);
      params.setAgentMetrics(agentData.agentMetrics);
      params.setEvaluationBenchmarks(agentData.evaluationBenchmarks);
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "刷新多智能体视图失败。"));
    }
  };

  const refreshReliabilityData = async (
    nextEvent?: V2EventRecord,
    filters?: { eventId?: string; severity?: string; sourceType?: string; fromTs?: string; toTs?: string; limit?: number },
  ) => {
    const eventId = filters?.eventId ?? nextEvent?.event_id ?? params.sessionView?.event.event_id;
    try {
      params.setReliabilityBusy(true);
      const baseline = await loadReliabilityDataset(eventId);
      const audits =
        filters?.severity || filters?.sourceType || filters?.fromTs || filters?.toTs || filters?.limit
          ? await api.listV2AuditRecords({
              eventId,
              severity: filters?.severity,
              sourceType: filters?.sourceType,
              fromTs: filters?.fromTs,
              toTs: filters?.toTs,
              limit: filters?.limit ?? 12,
            })
          : baseline.auditRecords;
      params.setOpenAlerts(baseline.alerts);
      params.setAuditRecords(audits);
      params.setArchiveStatus(baseline.archiveStatus);
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "刷新可靠性与审计视图失败。"));
    } finally {
      params.setReliabilityBusy(false);
    }
  };

  const refreshOperatorCapabilities = async (nextRole = params.operatorRole) => {
    try {
      return await api.getV2OperatorCapabilities(nextRole);
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "加载操作员能力失败。"));
      return null;
    }
  };

  return {
    hydrateEvent,
    refreshAdminData,
    refreshRegionalProposalData,
    applyRegionalProposalSnapshot,
    refreshDatasetStatusOnly,
    refreshAgentMeshData,
    refreshReliabilityData,
    refreshOperatorCapabilities,
  };
}

type BootstrapParams = {
  sessionStorageKey: string;
  eventStorageKey: string;
  defaultAreaId: string;
  frontendOperatorId: string;
  operatorRole: OperatorRole;
  setBootState: Setter<ConsoleBootState>;
  setHealthState: Setter<"checking" | "online" | "offline">;
  setExecutionStatus: Setter<ExecutionStatus>;
  setErrorMessage: Setter<string | null>;
  setAdminStatus: Setter<string | null>;
  setSessionView: Setter<V2CopilotSessionView | null>;
  hydrateEvent: (event: V2EventRecord) => Promise<void>;
  refreshAdminData: (event?: V2EventRecord) => Promise<void>;
  refreshRegionalProposalData: (event?: V2EventRecord) => Promise<void>;
  refreshAgentMeshData: (event?: V2EventRecord, session?: V2CopilotSessionView | null) => Promise<void>;
  refreshReliabilityData: (event?: V2EventRecord) => Promise<void>;
};

export function useConsoleBootstrapActions(params: BootstrapParams) {
  const createAndSeedEvent = async () => {
    const eventRecord = await api.createV2Event({
      area_id: params.defaultAreaId,
      title: "\u7891\u6797\u533a\u6d2a\u6c34\u9ad8\u98ce\u9669\u76d1\u6d4b\u4e8b\u4ef6",
      trigger_reason: "\u524d\u7aef\u542f\u52a8\u65f6\u81ea\u52a8\u521b\u5efa\u7684\u6f14\u793a\u4e8b\u4ef6",
      operator: params.frontendOperatorId,
      metadata: { source: "frontend_console_bootstrap" },
    });
    await api.ingestV2Observations(eventRecord.event_id, {
      operator: params.frontendOperatorId,
      observations: platformSeedObservations,
    });
    return eventRecord;
  };

  const bootstrap = async () => {
    params.setBootState("booting");
    params.setExecutionStatus("planning");
    params.setErrorMessage(null);
    params.setAdminStatus(null);
    try {
      await api.health();
      params.setHealthState("online");
    } catch (error) {
      params.setHealthState("offline");
      params.setBootState("error");
      params.setExecutionStatus("error");
      params.setErrorMessage(getErrorMessage(error, "后端服务不可达。"));
      return;
    }

    const cachedSessionId = window.sessionStorage.getItem(params.sessionStorageKey);
    const cachedEventId = window.sessionStorage.getItem(params.eventStorageKey);
    try {
      let activeEvent: V2EventRecord;
      let nextSession: V2CopilotSessionView | null = null;
      if (cachedSessionId) {
        nextSession = await api.getV2CopilotSession(cachedSessionId);
        params.setSessionView(nextSession);
        persistPlatformSession(params.sessionStorageKey, params.eventStorageKey, nextSession.session_id, nextSession.event.event_id);
        activeEvent = nextSession.event;
      } else {
        const eventRecord = cachedEventId
          ? ({ event_id: cachedEventId, area_id: params.defaultAreaId } as V2EventRecord)
          : await createAndSeedEvent();
        nextSession = await api.bootstrapV2CopilotSession({
          event_id: eventRecord.event_id,
          operator_role: params.operatorRole,
        });
        params.setSessionView(nextSession);
        persistPlatformSession(params.sessionStorageKey, params.eventStorageKey, nextSession.session_id, nextSession.event.event_id);
        activeEvent = nextSession.event;
      }
      await Promise.all([
        params.hydrateEvent(activeEvent),
        params.refreshAdminData(activeEvent),
        params.refreshRegionalProposalData(activeEvent),
        params.refreshAgentMeshData(activeEvent, nextSession),
        params.refreshReliabilityData(activeEvent),
      ]);
      params.setBootState("ready");
      params.setExecutionStatus("idle");
    } catch (error) {
      params.setBootState("error");
      params.setExecutionStatus("error");
      params.setErrorMessage(getErrorMessage(error, "初始化指挥控制台失败。"));
    }
  };

  return { bootstrap, createAndSeedEvent };
}

type AdminParams = {
  sessionView: V2CopilotSessionView | null;
  operatorRole: OperatorRole;
  frontendOperatorId: string;
  defaultAreaId: string;
  managedProfiles: EntityProfile[];
  selectedEntityId: string;
  setAdminBusy: Setter<boolean>;
  setAdminStatus: Setter<string | null>;
  setErrorMessage: Setter<string | null>;
  setManagedProfiles: Setter<EntityProfile[]>;
  setAreaResourceStatusView: Setter<ResourceStatusView | null>;
  setEventResourceStatusView: Setter<ResourceStatusView | null>;
  setRagDocuments: Setter<RAGDocument[]>;
  setDatasetStatus: Setter<DatasetPipelineStatusView | null>;
  setEntityImpacts: Setter<Record<string, EntityImpactView>>;
  setSelectedEntityId: Setter<string>;
  hydrateEvent: (event: V2EventRecord) => Promise<void>;
  refreshAdminData: (event?: V2EventRecord) => Promise<void>;
  refreshAgentMeshData: (event?: V2EventRecord, session?: V2CopilotSessionView | null) => Promise<void>;
  refreshReliabilityData: (event?: V2EventRecord) => Promise<void>;
  refreshDatasetStatusOnly: () => Promise<DatasetPipelineStatusView | null>;
};

export function useConsoleAdminActions(params: AdminParams) {
  const saveManagedProfile = async (profile: EntityProfile) => {
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      const exists = params.managedProfiles.some((item) => item.entity_id === profile.entity_id);
      const saved = exists
        ? await api.updateV2EntityProfile(profile, { operator_id: params.frontendOperatorId, operator_role: params.operatorRole })
        : await api.createV2EntityProfile(profile, { operator_id: params.frontendOperatorId, operator_role: params.operatorRole });
      const refreshedProfiles = await api.listV2EntityProfiles({ areaId: profile.area_id });
      params.setManagedProfiles(refreshedProfiles);
      params.setAdminStatus(exists ? "实体档案已更新，风险画像与检索上下文已同步刷新。" : "实体档案已创建，后续研判会纳入新的重点对象。");
      const eventId = params.sessionView?.event.event_id;
      if (eventId) {
        const impact = await api.getV2EntityImpact(saved.entity_id, eventId);
        params.setEntityImpacts((current) => ({ ...current, [saved.entity_id]: impact }));
        await Promise.all([params.refreshAgentMeshData(params.sessionView?.event), params.refreshReliabilityData(params.sessionView?.event)]);
      }
      params.setSelectedEntityId(saved.entity_id);
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "保存实体档案失败。"));
    } finally {
      params.setAdminBusy(false);
    }
  };

  const deleteManagedProfile = async (entityId: string) => {
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      await api.deleteV2EntityProfile(entityId);
      const areaId = params.sessionView?.event.area_id ?? params.defaultAreaId;
      params.setManagedProfiles(await api.listV2EntityProfiles({ areaId }));
      params.setEntityImpacts((current) => {
        const next = { ...current };
        delete next[entityId];
        return next;
      });
      if (params.selectedEntityId === entityId) {
        params.setSelectedEntityId(curatedEntities[0]?.id ?? "");
      }
      params.setAdminStatus("实体档案已删除，控制台已回退到默认对象列表。");
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "删除实体档案失败。"));
    } finally {
      params.setAdminBusy(false);
    }
  };

  const saveAreaResources = async (resourceStatus: ResourceStatus) => {
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      const nextView = await api.updateAreaResourceStatus(resourceStatus.area_id, resourceStatus, {
        operator_id: params.frontendOperatorId,
        operator_role: params.operatorRole,
      });
      params.setAreaResourceStatusView(nextView);
      if (params.sessionView?.event) {
        await Promise.all([params.hydrateEvent(params.sessionView.event), params.refreshAgentMeshData(params.sessionView.event), params.refreshReliabilityData(params.sessionView.event)]);
      }
      params.setAdminStatus("区域资源底座已更新，事件影响面与协同建议已重新计算。");
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "更新区域资源底座失败。"));
    } finally {
      params.setAdminBusy(false);
    }
  };

  const saveEventResources = async (resourceStatus: ResourceStatus) => {
    const eventId = params.sessionView?.event.event_id;
    if (!eventId || !params.sessionView?.event) return;
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      const nextView = await api.updateEventResourceStatus(eventId, resourceStatus, {
        operator_id: params.frontendOperatorId,
        operator_role: params.operatorRole,
      });
      params.setEventResourceStatusView(nextView);
      await Promise.all([params.hydrateEvent(params.sessionView.event), params.refreshAgentMeshData(params.sessionView.event), params.refreshReliabilityData(params.sessionView.event)]);
      params.setAdminStatus("事件级资源覆盖已更新，当前态势已按最新缺口重新评估。");
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "更新事件资源覆盖失败。"));
    } finally {
      params.setAdminBusy(false);
    }
  };

  const clearEventResources = async () => {
    const eventId = params.sessionView?.event.event_id;
    if (!eventId || !params.sessionView?.event) return;
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      await api.deleteEventResourceStatus(eventId);
      params.setEventResourceStatusView(await api.getEventResourceStatus(eventId));
      await Promise.all([params.hydrateEvent(params.sessionView.event), params.refreshAgentMeshData(params.sessionView.event), params.refreshReliabilityData(params.sessionView.event)]);
      params.setAdminStatus("事件级资源覆盖已清除，系统已恢复使用区域默认资源池。");
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "清除事件资源覆盖失败。"));
    } finally {
      params.setAdminBusy(false);
    }
  };

  const importRagDocuments = async (documents: RAGDocument[]) => {
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      const response = await api.importV2RagDocuments(documents, {
        operator_id: params.frontendOperatorId,
        operator_role: params.operatorRole,
      });
      params.setRagDocuments(response.documents);
      params.setAdminStatus(`已导入 ${response.document_count} 份运行期知识文档，新证据会立即进入问答与建议链路。`);
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "导入知识文档失败。"));
      throw error;
    } finally {
      params.setAdminBusy(false);
    }
  };

  const reloadRagDocuments = async () => {
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      const response = await api.reloadV2RagDocuments();
      params.setRagDocuments(response.documents);
      params.setAdminStatus(`运行期知识库已重载，共加载 ${response.document_count} 份文档。`);
    } catch (error) {
      params.setErrorMessage(getErrorMessage(error, "重载知识库失败。"));
    } finally {
      params.setAdminBusy(false);
    }
  };

  const runDatasetJob = async (statusMessage: string, jobPromise: Promise<{ job_id: string }>) => {
    params.setAdminBusy(true);
    params.setAdminStatus(null);
    params.setErrorMessage(null);
    try {
      const job = await jobPromise;
      params.setAdminStatus(statusMessage.replace("{jobId}", job.job_id));
      await params.refreshDatasetStatusOnly();
    } catch (error) {
      throw error;
    } finally {
      params.setAdminBusy(false);
    }
  };

  return {
    saveManagedProfile,
    deleteManagedProfile,
    saveAreaResources,
    saveEventResources,
    clearEventResources,
    importRagDocuments,
    reloadRagDocuments,
    fetchDatasetSources: async (download = true) => {
      try {
        await runDatasetJob(`已启动数据源任务：${formatDatasetAction(download ? "download" : "fetch")}。`, api.fetchV2DatasetSources({ download }));
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, "抓取公开数据源失败。"));
      }
    },
    retryDatasetSource: async (sourceId: string) => {
      try {
        await runDatasetJob(`已重试数据源 ${sourceId}，任务 {jobId} 正在执行。`, api.fetchV2DatasetSources({ download: true, sourceIds: [sourceId], forceRefresh: true }));
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, `重试数据源 ${sourceId} 失败。`));
      }
    },
    buildDatasetPackage: async (download = false, syncDemoDb = true) => {
      try {
        await runDatasetJob("已启动数据集构建任务 {jobId}。", api.buildV2Dataset({ download, syncDemoDb }));
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, "构建数据包失败。"));
      }
    },
    validateDatasetPackage: async () => {
      try {
        await runDatasetJob("已启动数据包校验任务 {jobId}。", api.validateV2Dataset());
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, "校验数据包失败。"));
      }
    },
    syncDatasetPackage: async () => {
      try {
        await runDatasetJob("已启动数据包同步任务 {jobId}。", api.syncV2Dataset());
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, "同步数据包失败。"));
      }
    },
    cancelDatasetJob: async (jobId: string) => {
      try {
        await runDatasetJob(`已请求取消数据集任务 ${jobId}。`, api.cancelV2DatasetJob(jobId));
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, `取消数据集任务 ${jobId} 失败。`));
      }
    },
    retryDatasetJob: async (jobId: string) => {
      try {
        await runDatasetJob(`已重试数据集任务 ${jobId}，新任务 {jobId} 正在执行。`, api.retryV2DatasetJob(jobId));
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, `重试数据集任务 ${jobId} 失败。`));
      }
    },
  };
}

type ReliabilityParams = {
  event: V2EventRecord | null;
  setReliabilityBusy: Setter<boolean>;
  setArchiveStatus: Setter<ArchiveStatusView | null>;
  setAdminStatus: Setter<string | null>;
  setErrorMessage: Setter<string | null>;
  refreshReliabilityData: (
    nextEvent?: V2EventRecord,
    filters?: { eventId?: string; severity?: string; sourceType?: string; fromTs?: string; toTs?: string; limit?: number },
  ) => Promise<void>;
};

export function useConsoleReliabilityActions(params: ReliabilityParams) {
  return {
    queryAuditRecords: async (filters?: { severity?: string; sourceType?: string; fromTs?: string; toTs?: string; limit?: number }) => {
      await params.refreshReliabilityData(params.event ?? undefined, filters);
    },
    runArchiveCycle: async () => {
      try {
        params.setReliabilityBusy(true);
        params.setArchiveStatus(await api.runV2ArchiveCycle());
        await params.refreshReliabilityData(params.event ?? undefined);
        params.setAdminStatus("归档清理已完成，可靠性与审计视图已刷新。");
      } catch (error) {
        params.setErrorMessage(getErrorMessage(error, "执行归档清理失败。"));
      } finally {
        params.setReliabilityBusy(false);
      }
    },
  };
}

type AgentParams = {
  sessionView: V2CopilotSessionView | null;
  setExecutionStatus: Setter<ExecutionStatus>;
  setErrorMessage: Setter<string | null>;
  setSupervisorRuns: Setter<SupervisorRunRecord[]>;
  setLatestEvaluationReport: Setter<EvaluationReport | null>;
  hydrateEvent: (event: V2EventRecord) => Promise<void>;
  refreshAgentMeshData: (event?: V2EventRecord, session?: V2CopilotSessionView | null) => Promise<void>;
  refreshReliabilityData: (event?: V2EventRecord) => Promise<void>;
};

export function useConsoleAgentActions(params: AgentParams) {
  return {
    runSupervisorNow: async () => {
      const currentEvent = params.sessionView?.event;
      if (!currentEvent) return;
      params.setExecutionStatus("running");
      params.setErrorMessage(null);
      try {
        const run = await api.runV2Supervisor(currentEvent.event_id);
        params.setSupervisorRuns((current) => [run, ...current.filter((item) => item.supervisor_run_id !== run.supervisor_run_id)].slice(0, 20));
        await Promise.all([params.hydrateEvent(currentEvent), params.refreshAgentMeshData(currentEvent), params.refreshReliabilityData(currentEvent)]);
        params.setExecutionStatus("idle");
      } catch (error) {
        params.setExecutionStatus("error");
        params.setErrorMessage(getErrorMessage(error, "执行当前事件的调度器失败。"));
      }
    },
    tickSupervisor: async () => {
      const currentEvent = params.sessionView?.event;
      params.setExecutionStatus("running");
      params.setErrorMessage(null);
      try {
        params.setSupervisorRuns(await api.tickV2Supervisor(currentEvent?.event_id));
        if (currentEvent) {
          await Promise.all([params.hydrateEvent(currentEvent), params.refreshAgentMeshData(currentEvent), params.refreshReliabilityData(currentEvent)]);
        }
        params.setExecutionStatus("idle");
      } catch (error) {
        params.setExecutionStatus("error");
        params.setErrorMessage(getErrorMessage(error, "执行调度器巡检失败。"));
      }
    },
    replayAgentTask: async (taskId: string, replayReason: string) => {
      const currentEvent = params.sessionView?.event;
      if (!currentEvent) return;
      params.setExecutionStatus("running");
      params.setErrorMessage(null);
      try {
        await api.replayV2AgentTask(taskId, replayReason);
        await Promise.all([params.refreshAgentMeshData(currentEvent, params.sessionView), params.refreshReliabilityData(currentEvent)]);
        params.setExecutionStatus("idle");
      } catch (error) {
        params.setExecutionStatus("error");
        params.setErrorMessage(getErrorMessage(error, "重放选中的 agent 任务失败。"));
      }
    },
    runEvaluation: async () => {
      params.setExecutionStatus("running");
      params.setErrorMessage(null);
      try {
        params.setLatestEvaluationReport(await api.runV2Evaluation());
        if (params.sessionView?.event) {
          await params.refreshAgentMeshData(params.sessionView.event, params.sessionView);
        }
        params.setExecutionStatus("idle");
      } catch (error) {
        params.setExecutionStatus("error");
        params.setErrorMessage(getErrorMessage(error, "运行 agent 评测失败。"));
      }
    },
    replayEvaluationReport: async (reportId: string) => {
      params.setExecutionStatus("running");
      params.setErrorMessage(null);
      try {
        params.setLatestEvaluationReport(await api.replayV2EvaluationReport(reportId));
        if (params.sessionView?.event) {
          await params.refreshAgentMeshData(params.sessionView.event, params.sessionView);
        }
        params.setExecutionStatus("idle");
      } catch (error) {
        params.setExecutionStatus("error");
        params.setErrorMessage(getErrorMessage(error, "重放评测报告失败。"));
      }
    },
  };
}
