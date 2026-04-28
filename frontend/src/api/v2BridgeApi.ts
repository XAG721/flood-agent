import { buildUrl, getApiOperatorContext, request } from "../lib/httpClient";
import type {
  Advisory,
  DailyReportView,
  EntityImpactView,
  EventEpisodeSummaryView,
  HazardStateV2,
  LongTermMemoryView,
  ObservationIngestItem,
  OperatorCapabilitiesView,
  OperatorRole,
  ProposalDraftUpdateRequest,
  RegionalAnalysisPackageView,
  RegionalProposalQueueSnapshot,
  RegionalProposalView,
  SimulationUpdateRequest,
  V2CopilotSessionView,
  V2EventRecord,
  V2EventSnapshot,
} from "../types/api";

export const v2BridgeApi = {
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
    streamUrl.searchParams.set("operator_role", getApiOperatorContext().role);
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
};
