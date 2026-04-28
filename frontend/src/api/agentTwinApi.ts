import { api } from "../lib/api";
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
    return api.getV3TwinOverview(eventId);
  },

  getFocusObject(eventId: string, objectId: string): Promise<FocusObjectView> {
    return api.getV3FocusObject(eventId, objectId);
  },

  sendDialog(eventId: string, payload: AgentDialogRequest): Promise<AgentDialogResponse> {
    return api.sendV3Dialog(eventId, payload);
  },

  getCouncil(eventId: string): Promise<AgentCouncilView> {
    return api.getV3AgentCouncil(eventId);
  },

  generateProposals(eventId: string, payload: ProposalGenerationRequest): Promise<ProposalGenerationResponse> {
    return api.generateV3Proposals(eventId, payload);
  },

  generateWarnings(proposalId: string): Promise<WarningGenerationResponse> {
    return api.generateV3Warnings(proposalId);
  },

  openTwinStream(
    eventId: string,
    handlers: {
      onEvent: (event: TwinStreamEvent) => void;
      onError?: () => void;
    },
    objectId?: string,
  ): EventSource {
    return api.openV3TwinStream(eventId, handlers, objectId);
  },
};
