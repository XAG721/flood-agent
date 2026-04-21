import type {
  ActionProposalV2,
  AgentName,
  CorpusType,
  OperatorRole,
  RiskLevel,
} from "../types/api";

const actionTypeTokenText: Record<string, string> = {
  emergency: "应急",
  medical: "医疗",
  evacuation: "转移",
  underground: "地下",
  space: "空间",
  clearance: "清退",
  shelter: "安置",
  network: "网络",
  activation: "启动",
  hazmat: "危化",
  protective: "防护",
  action: "行动",
  critical: "关键",
  infrastructure: "基础设施",
  defense: "防御",
  traffic: "交通",
  control: "管制",
  regional: "区域",
  notification: "通知",
  resource: "资源",
  dispatch: "调度",
};

export const riskLevelText: Record<RiskLevel, string> = {
  None: "无",
  Blue: "蓝色",
  Yellow: "黄色",
  Orange: "橙色",
  Red: "红色",
};

export const proposalStatusText: Record<ActionProposalV2["status"], string> = {
  pending: "待处理",
  approved: "已批准",
  rejected: "已驳回",
  withdrawn: "已撤回",
  superseded: "已替换",
};

export const operatorRoleTextMap: Record<OperatorRole, string> = {
  observer: "观察员",
  street_operator: "街道值守员",
  district_operator: "区级值守员",
  commander: "指挥长",
};

export const agentNameTextMap: Record<AgentName, string> = {
  hazard_agent: "Hazard Agent",
  exposure_agent: "Exposure Agent",
  resource_agent: "Resource Agent",
  planning_agent: "Planning Agent",
  policy_agent: "Policy Agent",
  comms_agent: "Comms Agent",
};

const regionalActionTypeText: Record<string, string> = {
  regional_notification: "区域通知",
  regional_evacuation: "区域转移建议",
  regional_resource_dispatch: "区域资源调度",
};

const triggerTypeText: Record<string, string> = {
  observation_ingested: "监测数据入库",
  simulation_updated: "模拟结果更新",
  resource_override_updated: "事件资源覆盖更新",
  resource_override_deleted: "事件资源覆盖删除",
  proposal_resolved: "请示已处理",
  copilot_escalation_requested: "智能问答升级请求",
  freshness_expired: "数据时效过期",
  manual_tick: "人工巡检触发",
  manual_run: "人工执行调度",
};

const triggerStatusText: Record<string, string> = {
  pending: "待处理",
  leased: "处理中",
  processed: "已处理",
  failed: "失败",
};

const proposalStreamStatusText: Record<string, string> = {
  closed: "未连接",
  connecting: "连接中",
  open: "已连接",
  error: "异常",
};

const circuitStateText: Record<string, string> = {
  closed: "闭合",
  open: "熔断中",
  half_open: "半开",
};

const supervisorRunStatusText: Record<string, string> = {
  running: "运行中",
  completed: "已完成",
  failed: "失败",
};

const agentTaskEventTypeText: Record<string, string> = {
  task_enqueued: "任务入队",
  task_claimed: "任务领取",
  agent_result_saved: "结果写入",
  task_completed: "任务完成",
  task_failed: "任务失败",
  replay_requested: "请求重放",
  replay_completed: "完成重放",
  trigger_processed: "触发已处理",
};

const agentTaskTypeText: Record<string, string> = {
  assess_hazard: "评估风险态势",
  assess_exposure: "评估暴露对象",
  assess_resources: "评估资源缺口",
  draft_plan: "生成行动方案",
  assess_policy: "评估策略约束",
  draft_comms: "生成沟通草案",
};

const actorText: Record<string, string> = {
  frontend_console: "前端指挥台",
  console_operator: "控制台操作员",
};

const sourceTypeText: Record<string, string> = {
  supervisor_loop: "后台巡检",
  runtime_admin: "运行期数据管理",
  trigger_event: "触发总线",
  agent_task: "Agent 任务",
  dataset_pipeline: "数据管线",
  notification_gateway: "通知网关",
  proposal_resolution: "请示处置",
  regional_proposal: "区域请示",
  housekeeping: "归档清理",
  monitoring_point: "监测点",
  daily_summary: "值班日报",
  postmortem: "事件复盘",
};

const auditActionText: Record<string, string> = {
  fetch_sources: "抓取数据源",
  build_dataset: "构建数据包",
  validate_dataset: "校验数据包",
  sync_demo_db: "同步演示数据库",
  dataset_job_cancel_requested: "请求取消数据任务",
  dataset_job_retried: "重试数据任务",
  dataset_job_canceled: "数据任务已取消",
  dataset_job_retry_scheduled: "数据任务已安排重试",
  dataset_job_failed: "数据任务失败",
  dataset_sources_fetched: "已抓取数据源",
  dataset_built: "数据包构建完成",
  dataset_validated: "数据包校验完成",
  dataset_synced: "数据包同步完成",
  execution_bundle_generated: "已生成执行联动包",
  proposal_approved: "请示已批准",
  proposal_rejected: "请示已驳回",
  proposal_draft_updated: "请示草稿已更新",
  entity_profile_saved: "对象画像已保存",
  entity_profile_deleted: "对象画像已删除",
  area_resource_updated: "区域资源已更新",
  event_resource_override_updated: "事件资源覆盖已更新",
  event_resource_override_deleted: "事件资源覆盖已删除",
  rag_documents_imported: "运行期文档已导入",
  rag_documents_reloaded: "运行期文档已重载",
  trigger_published: "触发事件已发布",
  agent_task_replayed: "Agent 任务已重放",
  archive_run_completed: "归档周期已完成",
  archive_run_failed: "归档周期失败",
  agent_task_failed: "Agent 任务失败",
  manual_tick_completed: "人工巡检完成",
  manual_tick_failed: "人工巡检失败",
  manual_run_completed: "人工执行调度完成",
  manual_run_failed: "人工执行调度失败",
  sweep_skipped: "后台巡检已跳过",
  trigger_processing_failed: "触发处理失败",
  background_sweep_failed_attempt: "后台巡检尝试失败",
  circuit_opened: "熔断已打开",
  circuit_recovered: "熔断已恢复",
  daily_report_generated: "值班日报已生成",
  daily_report_failed: "值班日报生成失败",
  episode_postmortem_generated: "高风险复盘已生成",
  episode_postmortem_failed: "高风险复盘生成失败",
};

const actionScopeFieldText: Record<string, string> = {
  target_scope: "通知范围",
  channel: "通知渠道",
  priority_zone: "优先区域",
  resource_count: "资源数量",
  evacuation_scope: "转移范围",
  priority_groups: "优先人群",
  shelter_direction: "建议安置方向",
  execution_notes: "执行备注",
  resource_type: "资源类型",
  resource_notes: "调度备注",
  transport_count: "转运车辆数",
};

const vulnerabilityTagText: Record<string, string> = {
  elderly: "高龄",
  limited_mobility: "行动受限",
  chronic_disease: "慢性病",
  children: "儿童",
  dismissal_peak: "放学高峰",
  inventory: "库存敏感",
  hazmat_sensitive: "危化敏感",
  critical_service: "关键医疗服务",
  patients: "在院患者",
  bedridden: "卧床",
  medical_support: "医疗支持",
  underground: "地下空间",
  commuter_peak: "通勤高峰",
  complex_egress: "疏散路径复杂",
  low_lying: "低洼",
  basement: "地下室",
  mixed_population: "混合居住人群",
};

const mobilityConstraintText: Record<string, string> = {
  stairs: "楼梯通行受限",
  needs_assistance: "需要协助",
  stretcher: "担架转运",
  oxygen_support: "氧气支持",
};

const notificationPreferenceText: Record<string, string> = {
  app_push: "应用推送",
  sms: "短信",
  community_call: "社区电话通知",
  dashboard: "指挥看板",
  phone: "电话",
  broadcast: "广播",
};

const corpusTypeText: Record<CorpusType, string> = {
  policy: "政策",
  case: "案例",
  profile: "画像",
};

const stopReasonText: Record<string, string> = {
  "No exposed entity was identified; the task graph converged conservatively.": "当前没有识别到暴露对象，任务图已保守收敛。",
  "Planning stopped because no exposed target was available.": "由于没有可规划的暴露目标，方案生成已停止。",
  "Regional proposal flow does not require a separate comms terminal step.": "区域请示流程不需要额外的通信终端步骤。",
  "Comms stopped because no communication target was available.": "由于没有可通信目标，沟通草案生成已停止。",
  "The task graph reached the comms terminal step.": "任务图已到达通信终点步骤。",
  "The requested agent task type is unsupported.": "当前请求的 Agent 任务类型暂不支持。",
  "No high-risk target is active.": "当前没有处于高风险的目标对象。",
  "Regional action proposals are awaiting commander confirmation.": "区域动作请示正在等待指挥长确认。",
  "Regional risk remains below the active action threshold.": "区域风险尚未达到动作触发阈值。",
  "Regional planning completed without an actionable pending proposal.": "区域规划已完成，但当前没有待处理的可执行请示。",
};

function humanizeRegionalActionType(actionType: string) {
  const tokens = actionType
    .split("_")
    .map((token) => token.trim().toLowerCase())
    .filter(Boolean);
  if (!tokens.length) {
    return actionType;
  }
  if (tokens.includes("evacuation")) {
    return `启动${joinActionTokens(tokens, ["evacuation"])}转移`;
  }
  if (tokens.includes("clearance")) {
    return `执行${joinActionTokens(tokens, ["clearance"])}清退`;
  }
  if (tokens.includes("activation")) {
    return `启动${joinActionTokens(tokens, ["activation"])}`;
  }
  if (tokens.includes("control")) {
    return `执行${joinActionTokens(tokens, ["control"])}管制`;
  }
  if (tokens.includes("dispatch")) {
    return `下达${joinActionTokens(tokens, ["dispatch"])}调度`;
  }
  if (tokens.includes("defense")) {
    return `启动${joinActionTokens(tokens, ["defense"])}防护`;
  }
  return tokens.map((token) => actionTypeTokenText[token] ?? token).join("");
}

function joinActionTokens(tokens: string[], remove: string[]) {
  const filtered = tokens.filter((token) => !remove.includes(token));
  if (!filtered.length) {
    return "";
  }
  return filtered.map((token) => actionTypeTokenText[token] ?? token).join("");
}

function inferRegionalActionCategory(actionType?: string | null, executionMode?: string | null) {
  if (executionMode === "notification") {
    return "预警通知";
  }
  if (executionMode === "evacuation_task") {
    return "人员转运";
  }
  if (executionMode === "resource_dispatch") {
    return "资源调度";
  }
  const normalized = (actionType ?? "").toLowerCase();
  if (normalized.includes("clearance") || normalized.includes("control")) {
    return "空间管控";
  }
  if (normalized.includes("defense") || normalized.includes("protective")) {
    return "设施防护";
  }
  if (normalized.includes("shelter")) {
    return "安置联动";
  }
  return "指挥动作";
}

function joinMappedValues(values: string[], mapping: Record<string, string>) {
  return values.map((value) => mapping[value] ?? value).join(", ");
}

function parseMappedValues(source: string, mapping: Record<string, string>) {
  const reverse = Object.fromEntries(Object.entries(mapping).map(([key, value]) => [value, key]));
  return source
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => reverse[item] ?? item);
}

export function formatRegionalActionType(actionType?: string | null) {
  if (!actionType) {
    return "区域动作";
  }
  return regionalActionTypeText[actionType] ?? humanizeRegionalActionType(actionType);
}

export function formatRegionalActionDisplayName(
  proposal: Pick<ActionProposalV2, "action_display_name" | "title" | "action_type">,
) {
  return proposal.action_display_name || proposal.title || formatRegionalActionType(proposal.action_type);
}

export function formatRegionalActionDisplayTagline(
  proposal: Pick<ActionProposalV2, "action_display_tagline" | "summary" | "recommendation">,
) {
  return proposal.action_display_tagline || proposal.summary || proposal.recommendation || "围绕当前风险生成的指挥动作。";
}

export function formatRegionalActionDisplayCategory(
  proposal: Pick<ActionProposalV2, "action_display_category" | "action_type" | "execution_mode">,
) {
  return proposal.action_display_category || inferRegionalActionCategory(proposal.action_type, proposal.execution_mode);
}

export function formatTriggerType(triggerType?: string | null) {
  if (!triggerType) {
    return "未命名触发";
  }
  return triggerTypeText[triggerType] ?? triggerType;
}

export function formatTriggerStatus(status?: string | null) {
  if (!status) {
    return "未知";
  }
  return triggerStatusText[status] ?? status;
}

export function formatProposalStreamStatus(status?: string | null) {
  if (!status) {
    return "未知";
  }
  return proposalStreamStatusText[status] ?? status;
}

export function formatCircuitState(state?: string | null) {
  if (!state) {
    return "未知";
  }
  return circuitStateText[state] ?? state;
}

export function formatSupervisorRunStatus(status?: string | null) {
  if (!status) {
    return "暂无运行记录";
  }
  return supervisorRunStatusText[status] ?? status;
}

export function formatAgentTaskEventType(eventType?: string | null) {
  if (!eventType) {
    return "未命名任务事件";
  }
  return agentTaskEventTypeText[eventType] ?? eventType;
}

export function formatAgentTaskType(taskType?: string | null) {
  if (!taskType) {
    return "未命名任务";
  }
  return agentTaskTypeText[taskType] ?? taskType;
}

export function formatOperatorActor(actor?: string | null) {
  if (!actor) {
    return "未记录";
  }
  return actorText[actor] ?? actor;
}

export function formatOperatorRole(role?: string | null) {
  if (!role) {
    return "未知角色";
  }
  return operatorRoleTextMap[role as OperatorRole] ?? role;
}

export function formatSourceType(sourceType?: string | null) {
  if (!sourceType) {
    return "未知来源";
  }
  return sourceTypeText[sourceType] ?? sourceType;
}

export function formatAuditAction(action?: string | null) {
  if (!action) {
    return "未命名动作";
  }
  return auditActionText[action] ?? action;
}

export function formatStopReason(reason?: string | null) {
  if (!reason) {
    return "暂无停止原因";
  }
  return stopReasonText[reason] ?? reason;
}

export function formatAgentHandoffTarget(target?: string | null) {
  if (!target) {
    return "未指定";
  }
  return agentNameTextMap[target as AgentName] ?? formatAgentTaskType(target);
}

export function formatActionScopeFieldLabel(fieldName: string) {
  return actionScopeFieldText[fieldName] ?? fieldName;
}

const executionModeTextMap: Record<string, string> = {
  notification: "通知下发",
  evacuation_task: "转移任务",
  resource_dispatch: "资源调度",
  generic_task: "通用行动",
};

const generationSourceTextMap: Record<string, string> = {
  system: "系统生成",
  llm: "模型生成",
};

export function formatExecutionMode(mode?: string | null) {
  if (!mode) {
    return "未定义执行模式";
  }
  return executionModeTextMap[mode] ?? mode;
}

export function formatGenerationSource(source?: string | null) {
  if (!source) {
    return "未标记来源";
  }
  return generationSourceTextMap[source] ?? source;
}

export function formatCorpusType(corpus?: CorpusType | null) {
  if (!corpus) {
    return "未分类";
  }
  return corpusTypeText[corpus] ?? corpus;
}

export function formatVulnerabilityTags(values: string[]) {
  return joinMappedValues(values, vulnerabilityTagText);
}

export function parseVulnerabilityTags(source: string) {
  return parseMappedValues(source, vulnerabilityTagText);
}

export function formatMobilityConstraints(values: string[]) {
  return joinMappedValues(values, mobilityConstraintText);
}

export function parseMobilityConstraints(source: string) {
  return parseMappedValues(source, mobilityConstraintText);
}

export function formatNotificationPreferences(values: string[]) {
  return joinMappedValues(values, notificationPreferenceText);
}

export function parseNotificationPreferences(source: string) {
  return parseMappedValues(source, notificationPreferenceText);
}
