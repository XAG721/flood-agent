import { buildUrl, request } from "../lib/httpClient";
import type {
  AgentDialogRequest,
  AgentDialogResponse,
  AgentCouncilView,
  FocusObjectView,
  ProposalGenerationRequest,
  ProposalGenerationResponse,
  TwinOverviewView,
  TwinStreamEvent,
  WarningGenerationResponse,
} from "../types/api";


export const agentTwinApi = {
  getOverview(eventId: string): Promise<TwinOverviewView> {
    return request(`/agent-twin/events/${eventId}/twin-overview`, { method: "GET" });
  },

  getFocusObject(eventId: string, objectId: string): Promise<FocusObjectView> {
    return request(`/agent-twin/events/${eventId}/objects/${objectId}`, { method: "GET" });
  },

  sendDialog(eventId: string, payload: AgentDialogRequest): Promise<AgentDialogResponse> {
    return request(`/agent-twin/events/${eventId}/dialog`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  getCouncil(eventId: string): Promise<AgentCouncilView> {
    return request(`/agent-twin/events/${eventId}/agent-council`, { method: "GET" });
  },

  generateProposals(eventId: string, payload: ProposalGenerationRequest): Promise<ProposalGenerationResponse> {
    return request(`/agent-twin/events/${eventId}/proposals/generate`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  generateWarnings(proposalId: string): Promise<WarningGenerationResponse> {
    return request(`/agent-twin/proposals/${proposalId}/warnings/generate`, {
      method: "POST",
    });
  },

  openTwinStream(
    eventId: string,
    handlers: {
      onEvent: (event: TwinStreamEvent) => void;
      onError?: () => void;
    },
    objectId?: string,
  ): EventSource {
    const streamUrl = new URL(buildUrl(`/agent-twin/events/${eventId}/stream`), window.location.origin);
    if (objectId) {
      streamUrl.searchParams.set("object_id", objectId);
    }
    const source = new EventSource(streamUrl.toString());
    source.onmessage = (event) => {
      const payload = JSON.parse(event.data) as TwinStreamEvent;
      handlers.onEvent(payload);
    };
    source.onerror = () => {
      handlers.onError?.();
    };
    return source;
  },
};
