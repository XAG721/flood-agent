export const appShellText = {
  overviewPageLabel: "总览页",
  overviewPageTitle: "全局风险态势总览",
  currentRole: "当前角色",
  refresh: "刷新",
  apiStatus: "接口状态",
  platformStatus: "平台状态",
  supervisorStatus: "后台巡检",
  supervisorRunning: "运行中",
  supervisorStopped: "已停止",
  unnamedEvent: "未命名事件",
  defaultAgentSummary: "多 Agent 正在持续汇总监测、研判和建议。",
} as const;

export const overviewPageText = {
  unidentifiedLeadImpact: "系统尚未锁定首要影响对象",
  priorityPanelTitle: "高风险对象",
  priorityPanelSubtitle: "按风险与影响时效排序，支持快速切换查看",
  objectPreviewSectionLabel: "对象预览",
  waitingSelectObject: "请选择一个重点对象",
  objectTypeLabel: "对象类型",
  impactTimeLabel: "预计受影响时间",
  askAboutObjectAction: "前往对话页继续追问",
  objectPreviewFallback: "系统已结合对象属性、位置和脆弱性给出风险提示。",
  objectPreviewEmpty: "当前暂无对象研判结果，可先前往对话页请求新的处置建议。",
  signalTimelineTitle: "近期信号",
  signalTimelineSubtitle: "汇总最新告警与监督运行信息",
  signalTimelineEmpty: "当前没有新的告警或监督运行记录。",
} as const;

export const operationsPageText = {
  executionBoardTitle: "执行链路",
  executionBoardDescription: "展示从态势感知到对象识别、方案生成、工具执行和人工确认的闭环进展。",
  analysisPackageSectionLabel: "分析包",
  analysisPackageSectionTitle: "区域分析包与联动建议",
  proposalHistorySectionLabel: "历史建议",
  proposalHistorySectionTitle: "区域请示与动作建议回顾",
  pendingConfirmSectionLabel: "待确认",
  pendingConfirmSectionTitle: "当前待人工确认动作",
  timelineTitle: "多 Agent 时间线",
  timelineSubtitle: "查看最新触发、任务推进和监督运行记录",
  timelineEmpty: "当前没有新的多 Agent 时间线记录。",
} as const;

export const executionFlowText = {
  senseTitle: "态势感知",
  impactTitle: "对象识别",
  planTitle: "方案生成",
  toolingTitle: "工具执行",
  confirmTitle: "人工确认",
  noHazardStateSummary: "系统尚未形成新的综合风险判断。",
  noHazardStateDetail: "等待新的监测、路网或模拟输入。",
  noLeadImpactSummary: "系统尚未锁定首要影响对象。",
  noLeadImpactDetail: "待更多对象级影响研判结果生成后更新。",
  noPlannerSummary: "尚未形成新的处置规划摘要。",
  noPlannerDetail: "可前往对话页发起新一轮处置建议生成。",
  noToolingSummary: "本轮尚未触发关键工具调用。",
  noToolingDetail: "等待规划层或人工操作发起能力调用。",
  noPendingConfirmationSummary: "当前没有待人工确认动作。",
  noPendingConfirmationDetail: "如有新建议进入审批节点，这里会及时更新。",
  alertDetailFallback: "告警已触发，请在详情中查看最新处置上下文。",
  supervisorRunFallback: "后台巡检已记录本轮运行，但尚未生成摘要。",
  triggerTimelineFallback: "系统已记录本次触发事件。",
  taskTimelineFallback: "多 Agent 任务已推进，请查看详情了解最新状态。",
} as const;

export const overviewMetricText = {
  riskLabel: "当前风险等级",
  riskHintFallback: "当前事件",
  trendLabel: "涨势趋势",
  trendHint: "反映未来短时变化方向",
  priorityLabel: "高风险对象",
  priorityHint: "已按影响强度与时效排序重点对象。",
  pendingLabel: "待确认动作",
  pendingHintFallback: "当前没有待确认动作。",
} as const;

export function formatPendingMetricHint(approvedProposalCount: number) {
  return approvedProposalCount > 0
    ? `${approvedProposalCount} 条动作已完成闭环`
    : overviewMetricText.pendingHintFallback;
}

export function formatTrendLabel(value?: string | null) {
  if (!value) return "趋势未知";

  const map: Record<string, string> = {
    rising: "持续上涨",
    rapidly_rising: "快速上升",
    stable: "基本稳定",
    falling: "逐步回落",
    unknown: "未知",
  };

  return map[value] ?? value;
}

export function buildOverviewSummary(params: {
  highRisk: boolean;
  riskLabel: string;
  trendLabel: string;
}) {
  const { highRisk, riskLabel, trendLabel } = params;
  if (highRisk) {
    return `当前事件已进入${riskLabel}风险阶段，系统判断洪水影响正在${trendLabel}发展，建议持续跟踪重点对象并准备联动处置。`;
  }

  return `当前事件整体风险为${riskLabel}，趋势判断为${trendLabel}，系统建议继续保持巡检并关注后续变化。`;
}

export function buildExecutionFlowStats(params: {
  pendingProposalCount: number;
  proposalHistoryCount: number;
  toolExecutionCount: number;
}) {
  return [
    `${params.pendingProposalCount} 条待确认动作`,
    `${params.proposalHistoryCount} 条历史建议`,
    `${params.toolExecutionCount} 次工具执行`,
  ];
}

export function buildAgentTimelineFallback(isTriggerEntry: boolean) {
  return isTriggerEntry ? executionFlowText.triggerTimelineFallback : executionFlowText.taskTimelineFallback;
}
