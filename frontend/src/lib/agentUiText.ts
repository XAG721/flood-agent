import type { AgentName } from "../types/api";

export const agentText: Record<AgentName, string> = {
  hazard_agent: "Hazard Agent",
  exposure_agent: "Exposure Agent",
  resource_agent: "Resource Agent",
  planning_agent: "Planning Agent",
  policy_agent: "Policy Agent",
  comms_agent: "Comms Agent",
};

export function normalizeAgentTerminology(value?: string | null) {
  if (!value) {
    return value ?? "";
  }

  return value
    .split("多代理").join("多 Agent")
    .split("智能代理").join("Agent")
    .split("代理").join("Agent");
}
