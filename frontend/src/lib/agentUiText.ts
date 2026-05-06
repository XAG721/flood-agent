import type { AgentName } from "../types/api";

export const agentText: Record<AgentName, string> = {
  hazard_agent: "风险研判智能体",
  exposure_agent: "暴露分析智能体",
  resource_agent: "资源调度智能体",
  planning_agent: "行动规划智能体",
  policy_agent: "策略审计智能体",
  comms_agent: "预警沟通智能体",
};

export function normalizeAgentTerminology(value?: string | null) {
  if (!value) {
    return value ?? "";
  }

  return value
    .split("Multi-Agent").join("多智能体")
    .split("multi-agent").join("多智能体")
    .split("AgentTwin").join("智能体孪生")
    .split("supervisor").join("监督编排器")
    .split("Supervisor").join("监督编排器")
    .split("ImpactAgent").join("影响研判智能体")
    .split("ActionAgent").join("行动建议智能体")
    .split("AuditAgent").join("审计智能体")
    .split("WarningAgent").join("预警智能体")
    .split("Impact Agent").join("影响研判智能体")
    .split("Action Agent").join("行动建议智能体")
    .split("Audit Agent").join("审计智能体")
    .split("Warning Agent").join("预警智能体")
    .split("proposal").join("处置方案")
    .split("Proposal").join("处置方案")
    .split("warning draft").join("预警草稿")
    .split("Warning draft").join("预警草稿")
    .split("warning").join("预警")
    .split("Warning").join("预警")
    .split("audience").join("受众")
    .split("多代理").join("多智能体")
    .split("智能代理").join("智能体")
    .split("代理").join("智能体")
    .split("Agent").join("智能体");
}
