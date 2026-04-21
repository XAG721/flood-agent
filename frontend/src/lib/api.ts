import type {
  AgentStatusView,
  AgentTask,
  AgentTimelineEntry,
  AgentMetricsView,
  ArchiveStatusView,
  AuditRecord,
  Advisory,
  DecisionReportView,
  DatasetJobView,
  DatasetPipelineStatusView,
  DailyReportView,
  EntityImpactView,
  EntityProfile,
  EventEpisodeSummaryView,
  EvaluationBenchmark,
  EvaluationReport,
  ExperienceContextView,
  HazardStateV2,
  OperatorCapabilitiesView,
  OperatorRole,
  ObservationIngestItem,
  OperationalAlert,
  ProposalDraftUpdateRequest,
  RAGDocument,
  RegionalAnalysisPackageView,
  RegionalProposalQueueSnapshot,
  RegionalProposalView,
  ResourceStatus,
  ResourceStatusView,
  SimulationUpdateRequest,
  LongTermMemoryView,
  MemoryBundleView,
  SharedMemorySnapshot,
  SupervisorLoopStatus,
  SupervisorRunRecord,
  TriggerEvent,
  V2CopilotSessionView,
  V2EventRecord,
  V2EventSnapshot,
} from "../types/api";

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

const operatorContext: { id: string; role: OperatorRole } = {
  id: "operator_console",
  role: "commander",
};

function buildUrl(path: string): string {
  return `${apiBaseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

export function setApiOperatorContext(context: { id?: string; role: OperatorRole }) {
  operatorContext.id = context.id?.trim() || "operator_console";
  operatorContext.role = context.role;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildUrl(path), {
    headers: {
      "Content-Type": "application/json",
      "X-Operator-Id": operatorContext.id,
      "X-Operator-Role": operatorContext.role,
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      const fallback = await response.text();
      if (fallback) {
        message = fallback;
      }
    }

    throw new Error(message);
  }

  return (await response.json()) as T;
}

export const api = {
  getV2OperatorCapabilities(operatorRole?: OperatorRole): Promise<OperatorCapabilitiesView> {
    const suffix = operatorRole ? `?operator_role=${encodeURIComponent(operatorRole)}` : "";
    return request(`/v2/security/capabilities${suffix}`, { method: "GET" });
  },

  health(): Promise<{ status: string }> {
    return request("/health", { method: "GET" });
  },

  createV2Event(payload: {
    area_id: string;
    title: string;
    trigger_reason?: string;
    operator?: string;
    metadata?: Record<string, unknown>;
  }): Promise<V2EventRecord> {
    return request("/v2/events", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  ingestV2Observations(
    eventId: string,
    payload: {
      operator?: string;
      observations: ObservationIngestItem[];
    },
  ): Promise<V2EventSnapshot> {
    return request(`/v2/events/${eventId}/observations`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  sendSimulationUpdate(
    eventId: string,
    payload: SimulationUpdateRequest,
  ): Promise<{
    event_id: string;
    overall_risk_level: string;
    risk_stage_key?: string | null;
    trigger_id: string;
    supervisor_run_id: string;
    queue_version: string;
    llm_status?: "ok" | "failed";
    llm_error?: string | null;
  }> {
    return request(`/v2/events/${eventId}/simulation-updates`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  getV2HazardState(eventId: string): Promise<HazardStateV2> {
    return request(`/v2/events/${eventId}/hazard-state`, {
      method: "GET",
    });
  },

  listPendingRegionalProposals(): Promise<RegionalProposalQueueSnapshot> {
    return request("/v2/proposals/pending", { method: "GET" });
  },

  listRegionalProposalsByEvent(
    eventId: string,
    status?: string,
  ): Promise<RegionalProposalView[]> {
    const suffix = status ? `?status=${encodeURIComponent(status)}` : "";
    return request(`/v2/events/${eventId}/regional-proposals${suffix}`, { method: "GET" });
  },

  listV2DailyReports(eventId: string): Promise<DailyReportView[]> {
    return request(`/v2/events/${eventId}/daily-reports`, { method: "GET" });
  },

  listV2EpisodeSummaries(eventId: string): Promise<EventEpisodeSummaryView[]> {
    return request(`/v2/events/${eventId}/episode-summaries`, { method: "GET" });
  },

  listV2LongTermMemory(eventId: string): Promise<LongTermMemoryView[]> {
    return request(`/v2/events/${eventId}/long-term-memory`, { method: "GET" });
  },

  getPendingRegionalAnalysisPackage(
    eventId: string,
  ): Promise<RegionalAnalysisPackageView | null> {
    return request(`/v2/events/${eventId}/regional-analysis-packages/pending`, { method: "GET" });
  },

  listRegionalAnalysisPackages(
    eventId: string,
    options?: { includePending?: boolean },
  ): Promise<RegionalAnalysisPackageView[]> {
    const includePending = options?.includePending ?? true;
    return request(
      `/v2/events/${eventId}/regional-analysis-packages?include_pending=${includePending ? "true" : "false"}`,
      { method: "GET" },
    );
  },

  updateRegionalProposalDraft(
    proposalId: string,
    payload: ProposalDraftUpdateRequest,
  ): Promise<RegionalProposalView> {
    return request(`/v2/proposals/${proposalId}/draft`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  },

  approveRegionalProposal(
    proposalId: string,
    payload: {
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<RegionalProposalView> {
    return request(`/v2/proposals/${proposalId}/approve`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  rejectRegionalProposal(
    proposalId: string,
    payload: {
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<RegionalProposalView> {
    return request(`/v2/proposals/${proposalId}/reject`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  approveRegionalAnalysisPackage(
    packageId: string,
    payload: {
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<RegionalAnalysisPackageView> {
    return request(`/v2/regional-analysis-packages/${packageId}/approve`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  rejectRegionalAnalysisPackage(
    packageId: string,
    payload: {
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<RegionalAnalysisPackageView> {
    return request(`/v2/regional-analysis-packages/${packageId}/reject`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  openRegionalProposalStream(handlers: {
    onSnapshot: (snapshot: RegionalProposalQueueSnapshot) => void;
    onError?: () => void;
  }): EventSource {
    const streamUrl = new URL(buildUrl("/v2/proposals/stream"), window.location.origin);
    streamUrl.searchParams.set("operator_role", operatorContext.role);
    const source = new EventSource(streamUrl.toString());
    source.onmessage = (event) => {
      const snapshot = JSON.parse(event.data) as RegionalProposalQueueSnapshot;
      handlers.onSnapshot(snapshot);
    };
    source.onerror = () => {
      handlers.onError?.();
    };
    return source;
  },

  getV2EntityImpact(
    entityId: string,
    eventId: string,
  ): Promise<EntityImpactView> {
    return request(
      `/v2/entities/${entityId}/impact?event_id=${encodeURIComponent(eventId)}`,
      { method: "GET" },
    );
  },

  generateV2Advisory(payload: {
    event_id: string;
    area_id: string;
    entity_id?: string;
    location_hint?: string;
    village?: string;
    operator_role?: string;
    profile_overrides?: Record<string, unknown>;
  }): Promise<Advisory> {
    return request("/v2/advisories/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  bootstrapV2CopilotSession(payload: {
    event_id: string;
    operator_role?: string;
  }): Promise<V2CopilotSessionView> {
    return request("/v2/copilot/sessions/bootstrap", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  getV2CopilotSession(sessionId: string): Promise<V2CopilotSessionView> {
    return request(`/v2/copilot/sessions/${sessionId}`, {
      method: "GET",
    });
  },

  sendV2CopilotMessage(
    sessionId: string,
    content: string,
  ): Promise<V2CopilotSessionView> {
    return request(`/v2/copilot/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
  },

  approveV2Proposal(
    sessionId: string,
    proposalId: string,
    payload: {
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<V2CopilotSessionView> {
    return request(`/v2/copilot/sessions/${sessionId}/proposals/${proposalId}/approve`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  rejectV2Proposal(
    sessionId: string,
    proposalId: string,
    payload: {
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<V2CopilotSessionView> {
    return request(`/v2/copilot/sessions/${sessionId}/proposals/${proposalId}/reject`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  batchApproveV2Proposals(
    sessionId: string,
    payload: {
      proposal_ids: string[];
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<V2CopilotSessionView> {
    return request(`/v2/copilot/sessions/${sessionId}/proposals/batch-approve`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  batchRejectV2Proposals(
    sessionId: string,
    payload: {
      proposal_ids: string[];
      operator_id: string;
      operator_role: string;
      note?: string;
    },
  ): Promise<V2CopilotSessionView> {
    return request(`/v2/copilot/sessions/${sessionId}/proposals/batch-reject`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  listV2EntityProfiles(params?: {
    areaId?: string;
    entityType?: string;
  }): Promise<EntityProfile[]> {
    const query = new URLSearchParams();
    if (params?.areaId) {
      query.set("area_id", params.areaId);
    }
    if (params?.entityType) {
      query.set("entity_type", params.entityType);
    }
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return request(`/v2/admin/entity-profiles${suffix}`, { method: "GET" });
  },

  createV2EntityProfile(
    profile: EntityProfile,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<EntityProfile> {
    return request("/v2/admin/entity-profiles", {
      method: "POST",
      body: JSON.stringify({
        profile,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  updateV2EntityProfile(
    profile: EntityProfile,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<EntityProfile> {
    return request(`/v2/admin/entity-profiles/${profile.entity_id}`, {
      method: "PUT",
      body: JSON.stringify({
        profile,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  deleteV2EntityProfile(entityId: string): Promise<{ status: string; entity_id: string }> {
    return request(`/v2/admin/entity-profiles/${entityId}`, {
      method: "DELETE",
    });
  },

  getAreaResourceStatus(areaId: string): Promise<ResourceStatusView> {
    return request(`/v2/admin/areas/${areaId}/resource-status`, {
      method: "GET",
    });
  },

  updateAreaResourceStatus(
    areaId: string,
    resourceStatus: ResourceStatus,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<ResourceStatusView> {
    return request(`/v2/admin/areas/${areaId}/resource-status`, {
      method: "PUT",
      body: JSON.stringify({
        resource_status: resourceStatus,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  getEventResourceStatus(eventId: string): Promise<ResourceStatusView> {
    return request(`/v2/admin/events/${eventId}/resource-status`, {
      method: "GET",
    });
  },

  updateEventResourceStatus(
    eventId: string,
    resourceStatus: ResourceStatus,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<ResourceStatusView> {
    return request(`/v2/admin/events/${eventId}/resource-status`, {
      method: "PUT",
      body: JSON.stringify({
        resource_status: resourceStatus,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  deleteEventResourceStatus(eventId: string): Promise<{ status: string; event_id: string }> {
    return request(`/v2/admin/events/${eventId}/resource-status`, {
      method: "DELETE",
    });
  },

  listV2RagDocuments(): Promise<RAGDocument[]> {
    return request("/v2/admin/rag-documents", { method: "GET" });
  },

  importV2RagDocuments(
    documents: RAGDocument[],
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<{ status: string; document_count: number; documents: RAGDocument[] }> {
    return request("/v2/admin/rag-documents/import", {
      method: "POST",
      body: JSON.stringify({
        documents,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  reloadV2RagDocuments(): Promise<{ status: string; document_count: number; documents: RAGDocument[] }> {
    return request("/v2/admin/rag-documents/reload", {
      method: "POST",
    });
  },

  getV2AgentStatus(eventId: string): Promise<AgentStatusView> {
    return request(`/v2/events/${eventId}/agent-status`, { method: "GET" });
  },

  listV2AgentTasks(eventId: string): Promise<AgentTask[]> {
    return request(`/v2/events/${eventId}/agent-tasks`, { method: "GET" });
  },

  getV2SharedMemory(eventId: string): Promise<SharedMemorySnapshot> {
    return request(`/v2/events/${eventId}/shared-memory`, { method: "GET" });
  },

  getV2SessionMemory(sessionId: string): Promise<MemoryBundleView> {
    return request(`/v2/copilot/sessions/${sessionId}/memory`, { method: "GET" });
  },

  listV2TriggerEvents(eventId: string): Promise<TriggerEvent[]> {
    return request(`/v2/events/${eventId}/trigger-events`, { method: "GET" });
  },

  listV2AgentTimeline(eventId: string): Promise<AgentTimelineEntry[]> {
    return request(`/v2/events/${eventId}/agent-timeline`, { method: "GET" });
  },

  getV2ExperienceContext(eventId: string): Promise<ExperienceContextView> {
    return request(`/v2/events/${eventId}/experience-context`, { method: "GET" });
  },

  getV2DecisionReport(eventId: string): Promise<DecisionReportView> {
    return request(`/v2/events/${eventId}/decision-report`, { method: "GET" });
  },

  getV2AgentMetrics(): Promise<AgentMetricsView> {
    return request("/v2/agent-metrics", { method: "GET" });
  },

  listV2EvaluationBenchmarks(): Promise<EvaluationBenchmark[]> {
    return request("/v2/evaluation/benchmarks", { method: "GET" });
  },

  runV2Evaluation(): Promise<EvaluationReport> {
    return request("/v2/evaluation/run", { method: "POST" });
  },

  getV2EvaluationReport(reportId: string): Promise<EvaluationReport> {
    return request(`/v2/evaluation/reports/${reportId}`, { method: "GET" });
  },

  replayV2EvaluationReport(reportId: string): Promise<EvaluationReport> {
    return request(`/v2/evaluation/reports/${reportId}/replay`, { method: "POST" });
  },

  replayV2AgentTask(taskId: string, replayReason = ""): Promise<AgentTask> {
    return request(`/v2/agent-tasks/${taskId}/replay`, {
      method: "POST",
      body: JSON.stringify({ replay_reason: replayReason }),
    });
  },

  listV2SupervisorRuns(eventId: string): Promise<SupervisorRunRecord[]> {
    return request(`/v2/events/${eventId}/supervisor-runs`, { method: "GET" });
  },

  getV2SupervisorStatus(): Promise<SupervisorLoopStatus> {
    return request("/v2/supervisor/status", { method: "GET" });
  },

  listV2Alerts(payload?: {
    eventId?: string;
    severity?: string;
    sourceType?: string;
    fromTs?: string;
    toTs?: string;
    limit?: number;
  }): Promise<OperationalAlert[]> {
    const params = new URLSearchParams();
    if (payload?.eventId) params.set("event_id", payload.eventId);
    if (payload?.severity) params.set("severity", payload.severity);
    if (payload?.sourceType) params.set("source_type", payload.sourceType);
    if (payload?.fromTs) params.set("from_ts", payload.fromTs);
    if (payload?.toTs) params.set("to_ts", payload.toTs);
    if (payload?.limit) params.set("limit", String(payload.limit));
    const suffix = params.size ? `?${params.toString()}` : "";
    return request(`/v2/alerts${suffix}`, { method: "GET" });
  },

  listV2AuditRecords(payload?: {
    eventId?: string;
    severity?: string;
    sourceType?: string;
    fromTs?: string;
    toTs?: string;
    limit?: number;
  }): Promise<AuditRecord[]> {
    const params = new URLSearchParams();
    if (payload?.eventId) params.set("event_id", payload.eventId);
    if (payload?.severity) params.set("severity", payload.severity);
    if (payload?.sourceType) params.set("source_type", payload.sourceType);
    if (payload?.fromTs) params.set("from_ts", payload.fromTs);
    if (payload?.toTs) params.set("to_ts", payload.toTs);
    if (payload?.limit) params.set("limit", String(payload.limit));
    const suffix = params.size ? `?${params.toString()}` : "";
    return request(`/v2/audit/records${suffix}`, { method: "GET" });
  },

  getV2ArchiveStatus(): Promise<ArchiveStatusView> {
    return request("/v2/archive/status", { method: "GET" });
  },

  runV2ArchiveCycle(): Promise<ArchiveStatusView> {
    return request("/v2/archive/run", { method: "POST" });
  },

  getV2DatasetStatus(): Promise<DatasetPipelineStatusView> {
    return request("/v2/admin/dataset/status", { method: "GET" });
  },

  getV2DatasetJobs(): Promise<DatasetJobView[]> {
    return request("/v2/admin/dataset/jobs", { method: "GET" });
  },

  fetchV2DatasetSources(payload?: { download?: boolean; sourceIds?: string[]; forceRefresh?: boolean }): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/fetch", {
      method: "POST",
      body: JSON.stringify({
        download: payload?.download ?? true,
        source_ids: payload?.sourceIds ?? [],
        force_refresh: payload?.forceRefresh ?? false,
      }),
    });
  },

  buildV2Dataset(payload?: { download?: boolean; syncDemoDb?: boolean }): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/build", {
      method: "POST",
      body: JSON.stringify({
        download: payload?.download ?? false,
        sync_demo_db: payload?.syncDemoDb ?? true,
      }),
    });
  },

  validateV2Dataset(): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/validate", { method: "POST", body: JSON.stringify({}) });
  },

  syncV2Dataset(payload?: { dbPath?: string }): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/sync-demo-db", {
      method: "POST",
      body: JSON.stringify({ db_path: payload?.dbPath ?? null }),
    });
  },

  cancelV2DatasetJob(jobId: string): Promise<DatasetJobView> {
    return request(`/v2/admin/dataset/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
  },

  retryV2DatasetJob(jobId: string): Promise<DatasetJobView> {
    return request(`/v2/admin/dataset/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" });
  },

  runV2Supervisor(eventId: string): Promise<SupervisorRunRecord> {
    return request(`/v2/events/${eventId}/supervisor/run`, { method: "POST" });
  },

  tickV2Supervisor(eventId?: string): Promise<SupervisorRunRecord[]> {
    const suffix = eventId ? `?event_id=${encodeURIComponent(eventId)}` : "";
    return request(`/v2/supervisor/tick${suffix}`, { method: "POST" });
  },
};
