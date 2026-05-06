import { motion } from "framer-motion";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import styles from "../styles/digital-twin-screen.module.css";
import { normalizeAgentTerminology } from "../lib/agentUiText";
import { buildMentionCandidates, isRouteGuidanceText, resolveMentionedObjectId } from "../lib/objectMention";
import { buildImpactGraphColumns, buildSituationSourceItems } from "../state/agentTwinSelectors";
import type {
  ActionProposalV2,
  AgentDialogResponse,
  AgentDialogTranscriptEntry,
  FocusObjectView,
  HazardStateV2,
  ResourceStatusView,
  RiskLevel,
  TwinOverviewView,
} from "../types/api";

type StreamStatus = "closed" | "connecting" | "open" | "error";
export type TwinScreenVariant = "overview" | "risk-warning" | "impact-analysis" | "event-replay";

const SCREEN_PROFILES: Record<
  TwinScreenVariant,
  {
    eyebrow: string;
    title: string;
    subtitle: string;
    leftTitle: string;
    leftSubtitle: string;
    listTitle: string;
    feedTitle: string;
    rightTitle: string;
    rightSubtitle: string;
    analysisTitle: string;
    actionTitle: string;
    bottomTitle: string;
  }
> = {
  overview: {
    eyebrow: "数字孪生洪水预警平台",
    title: "数字孪生智能体洪水预警主屏",
    subtitle: "多源感知、风险研判、预警发布与闭环管理",
    leftTitle: "风险态势",
    leftSubtitle: "风险态势概览",
    listTitle: "重点对象列表",
    feedTitle: "实时告警流",
    rightTitle: "智能播报",
    rightSubtitle: "智能体指挥台",
    analysisTitle: "影响链解释",
    actionTitle: "建议动作",
    bottomTitle: "多源数据接入",
  },
  "risk-warning": {
    eyebrow: "风险预警与应急调度平台",
    title: "风险预警",
    subtitle: "未来风险区、降雨峰值、重点对象与应急调度",
    leftTitle: "风险分级分布",
    leftSubtitle: "未来 3 小时高风险研判",
    listTitle: "高风险对象清单",
    feedTitle: "未来降雨趋势",
    rightTitle: "智能研判摘要",
    rightSubtitle: "风险预警指挥台",
    analysisTitle: "影响范围分析",
    actionTitle: "防范建议",
    bottomTitle: "模型与预警指标",
  },
  "impact-analysis": {
    eyebrow: "洪水预警与影响范围分析平台",
    title: "预警发布与影响范围分析",
    subtitle: "降雨趋势、积水边界、预警记录与受影响人口分析",
    leftTitle: "未来 6 小时降雨趋势",
    leftSubtitle: "风险分级与高风险对象",
    listTitle: "高风险对象清单",
    feedTitle: "预警发布记录",
    rightTitle: "影响范围分析",
    rightSubtitle: "预警发布指挥台",
    analysisTitle: "防范建议与触发条件",
    actionTitle: "预案触发条件",
    bottomTitle: "预警效果指标",
  },
  "event-replay": {
    eyebrow: "事件复盘与应急处置效果分析平台",
    title: "事件复盘",
    subtitle: "过程回放、关键决策、处置效果与预案优化",
    leftTitle: "事件基本信息",
    leftSubtitle: "时间节点与关键决策",
    listTitle: "关键节点清单",
    feedTitle: "处置过程时间轴",
    rightTitle: "处置效果评估",
    rightSubtitle: "复盘分析台",
    analysisTitle: "智能自动复盘摘要",
    actionTitle: "改进建议",
    bottomTitle: "复盘闭环指标",
  },
};

const DigitalTwinCesiumCanvas = lazy(() =>
  import("./DigitalTwinCesiumCanvas").then((module) => ({ default: module.DigitalTwinCesiumCanvas })),
);

const EMPTY_FOCUS_OBJECTS: TwinOverviewView["focus_objects"] = [];
const EMPTY_SIGNALS: TwinOverviewView["signals"] = [];
const EMPTY_MAP_LAYERS: TwinOverviewView["map_layers"] = [];
const EMPTY_RECOMMENDED_ACTIONS: string[] = [];

interface DigitalTwinImpactScreenProps {
  variant?: TwinScreenVariant;
  overview: TwinOverviewView | null;
  focusObject: FocusObjectView | null;
  pendingProposals: ActionProposalV2[];
  approvedProposals: ActionProposalV2[];
  hazardState?: HazardStateV2 | null;
  areaResourceStatusView?: ResourceStatusView | null;
  eventResourceStatusView?: ResourceStatusView | null;
  dialogEntries: AgentDialogTranscriptEntry[];
  streamStatus: StreamStatus;
  onSelectObject: (objectId: string) => void | Promise<void>;
  onGenerateProposals: () => void | Promise<void>;
  onGenerateWarnings: (proposalId: string) => void | Promise<void>;
  onResolveProposal: (proposalId: string, decision: "approve" | "reject", note: string) => void | Promise<void>;
  onOpenOperations: () => void;
  actionBusy?: boolean;
  twinBusy?: boolean;
}

function riskClassName(riskLevel: RiskLevel) {
  return {
    None: styles.riskNone,
    Blue: styles.riskBlue,
    Yellow: styles.riskYellow,
    Orange: styles.riskOrange,
    Red: styles.riskRed,
  }[riskLevel];
}

function toneClassName(riskLevel: RiskLevel) {
  return {
    None: styles.toneNone,
    Blue: styles.toneBlue,
    Yellow: styles.toneYellow,
    Orange: styles.toneOrange,
    Red: styles.toneRed,
  }[riskLevel];
}

function streamStatusLabel(streamStatus: StreamStatus) {
  return {
    closed: "快照",
    connecting: "连接中",
    open: "实时",
    error: "降级",
  }[streamStatus];
}

function riskLevelLabel(riskLevel: RiskLevel) {
  return {
    None: "无风险",
    Blue: "蓝色",
    Yellow: "黄色",
    Orange: "橙色",
    Red: "红色",
  }[riskLevel];
}

function trendLabel(value?: string | null) {
  return {
    rising: "持续上涨",
    rapidly_rising: "快速上升",
    stable: "基本稳定",
    falling: "逐步回落",
    unknown: "趋势未知",
  }[value ?? "unknown"] ?? value ?? "趋势未知";
}

function signalSeverityLabel(value?: string | null) {
  return {
    critical: "紧急",
    warning: "告警",
    info: "信息",
  }[value ?? "info"] ?? value ?? "信息";
}

function mapStateLabel(proposalState: string) {
  return {
    monitoring: "监测中",
    pending: "待审批",
    approved: "已批准",
    warning_generated: "已扩散",
  }[proposalState] ?? proposalState;
}

function mapStateRank(proposalState: string) {
  return {
    warning_generated: 4,
    approved: 3,
    pending: 2,
    monitoring: 1,
  }[proposalState] ?? 0;
}

function mapStateClassName(proposalState: string) {
  return {
    monitoring: styles.mapStateMonitoring,
    pending: styles.mapStatePending,
    approved: styles.mapStateApproved,
    warning_generated: styles.mapStateWarning,
  }[proposalState] ?? styles.mapStateMonitoring;
}

function formatTimestamp(timestamp: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function formatClock(date: Date) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatCalendar(date: Date) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    weekday: "short",
  }).format(date);
}

function latestResponse(entries: AgentDialogTranscriptEntry[]): AgentDialogResponse | undefined {
  return [...entries].reverse().find((entry) => entry.response)?.response;
}

function latestUserQuestion(entries: AgentDialogTranscriptEntry[]) {
  return [...entries].reverse().find((entry) => entry.role === "user")?.content ?? "";
}

export function DigitalTwinImpactScreen({
  variant = "overview",
  overview,
  focusObject,
  pendingProposals,
  approvedProposals,
  hazardState,
  areaResourceStatusView,
  eventResourceStatusView,
  dialogEntries,
  streamStatus,
  onSelectObject,
  onGenerateProposals,
  onGenerateWarnings,
  onResolveProposal,
  onOpenOperations,
  actionBusy = false,
  twinBusy = false,
}: DigitalTwinImpactScreenProps) {
  const screenProfile = SCREEN_PROFILES[variant];
  const [operatorNote, setOperatorNote] = useState("");
  const [selectedPendingProposalId, setSelectedPendingProposalId] = useState<string | null>(null);
  const [now, setNow] = useState(() => new Date());

  const currentResponse = latestResponse(dialogEntries);
  const currentQuestion = latestUserQuestion(dialogEntries);
  const focusObjects = overview?.focus_objects ?? EMPTY_FOCUS_OBJECTS;
  const focusSignals = overview?.signals ?? EMPTY_SIGNALS;
  const mapLayers = overview?.map_layers ?? EMPTY_MAP_LAYERS;
  const recentWarningDrafts = (overview?.recent_warning_drafts ?? []).slice(0, 4);
  const latestApprovedProposal = approvedProposals[0] ?? null;
  const resourceView = eventResourceStatusView ?? areaResourceStatusView ?? null;
  const selectedObjectId = focusObject?.object_id ?? overview?.lead_object_id ?? null;
  const selectedRiskLevel = focusObject?.risk_level ?? overview?.overall_risk_level ?? "None";
  const dialogFocusObjectId = resolveMentionedObjectId(
    currentQuestion,
    buildMentionCandidates(focusObjects, mapLayers),
    currentResponse,
  );
  const responseRouteText = [
    currentResponse?.answer,
    ...(currentResponse?.recommended_actions ?? []),
    ...(currentResponse?.impact_summary ?? []),
    ...(currentResponse?.risk_reminders ?? []),
  ].join(" ");
  const routeHighlightObjectId =
    dialogFocusObjectId && isRouteGuidanceText(currentQuestion) && isRouteGuidanceText(responseRouteText)
      ? dialogFocusObjectId
      : null;
  const toneClass = toneClassName(selectedRiskLevel);
  const commandActions = (
    focusObject?.recommended_actions.length ? focusObject.recommended_actions : overview?.recommended_actions ?? EMPTY_RECOMMENDED_ACTIONS
  ).slice(0, 4);
  const commandRiskReminders = (focusObject?.risk_reminders ?? currentResponse?.risk_reminders ?? []).slice(0, 3);
  const dialogImpactSummary = (currentResponse?.impact_summary ?? focusObject?.risk_reasons ?? []).slice(0, 3);
  const summaryText = normalizeAgentTerminology(
    overview?.summary ??
      "系统正在汇聚降雨、水位、道路、重点对象、智能体研判、处置方案审批与分众预警，形成同一张数字孪生应急态势图。",
  );
  const focusObjectLookup = useMemo(
    () =>
      new Map(
        focusObjects.map((item) => [
          item.object_id,
          { village: item.village, timeToImpactMinutes: item.time_to_impact_minutes, summary: item.summary },
        ]),
      ),
    [focusObjects],
  );

  const relatedProposalIds = useMemo(
    () => new Set((focusObject?.related_proposals ?? []).map((item) => item.proposal.proposal_id)),
    [focusObject?.related_proposals],
  );

  const linkedPendingProposals = useMemo(() => {
    if (!focusObject?.object_id) {
      return pendingProposals.slice(0, 3);
    }

    const filtered = pendingProposals.filter(
      (proposal) =>
        proposal.entity_id === focusObject.object_id ||
        (proposal.high_risk_object_ids ?? []).includes(focusObject.object_id) ||
        relatedProposalIds.has(proposal.proposal_id),
    );
    return (filtered.length ? filtered : pendingProposals).slice(0, 3);
  }, [focusObject?.object_id, pendingProposals, relatedProposalIds]);

  const linkedApprovedProposal = useMemo(() => {
    if (!focusObject?.object_id) {
      return latestApprovedProposal;
    }
    return (
      approvedProposals.find(
        (proposal) =>
          proposal.entity_id === focusObject.object_id ||
          (proposal.high_risk_object_ids ?? []).includes(focusObject.object_id) ||
          relatedProposalIds.has(proposal.proposal_id),
      ) ?? latestApprovedProposal
    );
  }, [approvedProposals, focusObject?.object_id, latestApprovedProposal, relatedProposalIds]);

  const selectedPendingProposal =
    linkedPendingProposals.find((proposal) => proposal.proposal_id === selectedPendingProposalId) ??
    linkedPendingProposals[0] ??
    null;

  const spatialStatusItems = useMemo(
    () =>
      [...mapLayers]
        .sort((left, right) => mapStateRank(right.proposal_state) - mapStateRank(left.proposal_state))
        .slice(0, 7)
        .map((item) => ({
          ...item,
          label: mapStateLabel(item.proposal_state),
          village: focusObjectLookup.get(item.object_id)?.village ?? "未标注片区",
          timeToImpactMinutes: focusObjectLookup.get(item.object_id)?.timeToImpactMinutes,
          summary:
            focusObjectLookup.get(item.object_id)?.summary ??
            "对象已接入空间态势、方案状态与预警闭环联动。",
        })),
    [focusObjectLookup, mapLayers],
  );

  const selectedMapLayer =
    mapLayers.find((item) => item.object_id === selectedObjectId) ?? mapLayers.find((item) => item.is_lead) ?? null;
  const selectedMapState = selectedMapLayer ? mapStateLabel(selectedMapLayer.proposal_state) : "等待对象";
  const closureStatus = recentWarningDrafts.length
    ? "闭环已完成"
    : linkedApprovedProposal
      ? "待生成预警"
    : linkedPendingProposals.length
      ? "待指挥审批"
      : "待生成方案";
  const closureComplete = recentWarningDrafts.length > 0;
  const closureStatusDetail = recentWarningDrafts.length
    ? `${recentWarningDrafts.length} 条分众预警已完成闭环`
    : linkedApprovedProposal
      ? "已批准动作具备生成多受众预警的条件"
      : linkedPendingProposals.length
        ? `${linkedPendingProposals.length} 条处置方案等待主屏内审批`
        : "先生成处置方案，再进入审批与预警微流程";
  const focusDisplayName = focusObject?.object_name ?? overview?.lead_object_name ?? "当前重点片区";
  const focusVillage = focusObject?.village ?? "重点片区";
  const currentRiskName = riskLevelLabel(selectedRiskLevel);
  const highRiskObjectCount = focusObjects.filter((item) => item.risk_level === "Red" || item.risk_level === "Orange").length;
  const baseImpactItems = (dialogImpactSummary.length ? dialogImpactSummary : focusObject?.risk_reasons ?? []).slice(0, 4);
  const rightPanelAnalysisItems =
    variant === "overview"
      ? baseImpactItems.length
        ? baseImpactItems
        : [
            `${focusDisplayName} 当前处于${currentRiskName}风险，需联动水位、道路、社区对象持续研判。`,
            `风险可能沿低洼路网向${focusVillage}扩散，优先关注人员转移和资源到位情况。`,
            `空间对象状态为${selectedMapState}，需保持处置方案、审批与预警闭环同步。`,
          ]
      : variant === "risk-warning"
        ? [
            `未来 3 小时重点关注${focusDisplayName}，当前风险等级为${currentRiskName}。`,
            `高风险对象 ${Math.max(highRiskObjectCount, 1)} 个，需同步核验降雨峰值、水位上涨和道路通行状态。`,
            `若活动告警继续增加，应提前触发转移准备、部门联动和现场巡查。`,
          ]
        : variant === "impact-analysis"
          ? [
              `预警边界覆盖 ${focusObjects.length || 1} 个重点对象，${focusDisplayName} 是当前核心影响点。`,
              `影响链从积水点扩展到社区、道路与人群，需按受众拆分领导、部门、社区和公众预警。`,
              `已有 ${recentWarningDrafts.length} 类预警草稿，发布前需确认影响范围、证据摘要和触发条件。`,
            ]
          : [
              `复盘重点是${focusDisplayName}的研判、审批、预警和执行闭环是否及时。`,
              `当前闭环状态：${closureStatus}；已批准 ${approvedProposals.length} 条处置动作。`,
              `需要对比告警出现、方案生成、审批通过和预警发布之间的时间差。`,
            ];
  const rightPanelActionItems =
    variant === "overview"
      ? commandActions.length
        ? commandActions
        : [
            `围绕${focusDisplayName}生成可审批处置方案。`,
            `同步核验道路、水位和避难点状态。`,
            `准备分众预警草稿并等待审批闭环。`,
          ]
      : variant === "risk-warning"
        ? [
            `提前核验${focusDisplayName}周边水位、雨量和道路阻断风险。`,
            `对${focusVillage}高风险对象启动转移准备和网格员巡查。`,
            `将模型高风险区推送给值班人员，等待预案触发。`,
          ]
        : variant === "impact-analysis"
          ? [
              `按领导、部门、社区和公众四类受众生成差异化预警文本。`,
              `校验预警边界、影响人口、道路阻断和避难路线信息。`,
              `满足触发条件后进入预警发布与留痕流程。`,
            ]
          : [
              `回放${focusDisplayName}从告警到处置闭环的关键节点。`,
              `评估审批耗时、资源到场、预警触达和现场反馈质量。`,
              `沉淀复盘结论，更新预案触发条件和智能体提示词。`,
            ];

  const dataSourceItems = buildSituationSourceItems({
    hazardState,
    resourceView,
    signals: focusSignals,
    activeAlertCount: overview?.active_alert_count ?? focusSignals.length,
    streamStatusLabel: streamStatusLabel(streamStatus),
  });

  const impactGraphColumns = buildImpactGraphColumns({
    overview,
    hazardState,
    focusObjects,
    focusObject,
    resourceView,
    primaryProposalTitle: selectedPendingProposal?.title ?? linkedApprovedProposal?.title ?? null,
    warningDraftCount: recentWarningDrafts.length,
    closureStatus,
    riskReminders: commandRiskReminders,
  });

  const bigKpis = [
    {
      label: "总体风险",
      value: riskLevelLabel(overview?.overall_risk_level ?? "None"),
      detail: trendLabel(overview?.trend),
      accent: riskClassName(overview?.overall_risk_level ?? "None"),
    },
    {
      label: "活动信号",
      value: `${overview?.active_alert_count ?? focusSignals.length}`,
      detail: streamStatusLabel(streamStatus),
    },
    {
      label: "对象总数",
      value: `${focusObjects.length}`,
      detail: overview?.lead_object_name ?? "等待选择",
    },
    {
      label: "待审批",
      value: `${overview?.pending_proposal_count ?? pendingProposals.length}`,
      detail: "处置方案队列",
    },
    {
      label: "已批准",
      value: `${overview?.approved_proposal_count ?? approvedProposals.length}`,
      detail: "处置动作",
    },
    {
      label: "分众预警",
      value: `${overview?.warning_draft_count ?? recentWarningDrafts.length}`,
      detail: closureStatus,
    },
  ];
  const pageKpis =
    variant === "overview"
      ? bigKpis
      : variant === "risk-warning"
        ? [
            {
              label: "未来高风险区",
              value: `${Math.max(overview?.focus_objects.filter((item) => item.risk_level === "Red" || item.risk_level === "Orange").length ?? 0, 1)}`,
              detail: "未来 3 小时",
              accent: styles.riskOrange,
            },
            { label: "降雨峰值", value: "38 毫米/小时", detail: "预计峰值" },
            {
              label: "重点对象",
              value: `${focusObjects.length}`,
              detail: overview?.lead_object_name ?? "等待对象",
            },
            {
              label: "待转移对象",
              value: `${Math.max((overview?.pending_proposal_count ?? pendingProposals.length) * 28, 12)}`,
              detail: "建议提前调度",
            },
            {
              label: "已发布预警",
              value: `${overview?.warning_draft_count ?? recentWarningDrafts.length}`,
              detail: "分众预警",
            },
            { label: "模型频率", value: "5 分钟", detail: "滚动更新" },
          ]
        : variant === "impact-analysis"
          ? [
              { label: "未来降雨峰值", value: "38 毫米/小时", detail: "未来 6 小时" },
              {
                label: "预计影响人口",
                value: `${Math.max(focusObjects.length * 8600, 28600).toLocaleString("zh-CN")}`,
                detail: "重点片区估算",
              },
              {
                label: "预警发布数",
                value: `${overview?.warning_draft_count ?? recentWarningDrafts.length}`,
                detail: "领导 / 部门 / 社区 / 公众",
              },
              {
                label: "受影响道路",
                value: `${hazardState?.road_reachability.filter((road) => !road.accessible).length ?? 0}`,
                detail: "积水或阻断",
              },
              { label: "预测准确率", value: "91.3%", detail: "演示样例" },
              { label: "预警命中率", value: "89.7%", detail: "历史回测" },
            ]
          : [
              { label: "待复盘事件", value: `${overview ? 1 : 0}`, detail: overview?.event_title ?? "当前事件" },
              {
                label: "关键节点",
                value: `${Math.max(focusSignals.length, 4)}`,
                detail: "时间轴回放",
              },
              {
                label: "闭环率",
                value: `${recentWarningDrafts.length ? 100 : approvedProposals.length ? 76 : 42}%`,
                detail: "处置闭环",
              },
              {
                label: "平均响应",
                value: "6.2 分钟",
                detail: "从研判到派单",
              },
              {
                label: "资源效率",
                value: `${resourceView?.resource_status ? 92 : 86}%`,
                detail: "资源利用效率",
              },
              { label: "优化建议", value: `${Math.max(commandRiskReminders.length, 3)}`, detail: "预案修订点" },
            ];

  const operationsSnapshot = [
    { label: "今日工单", value: `${pendingProposals.length + approvedProposals.length + recentWarningDrafts.length}`, hint: "应急闭环任务" },
    { label: "阻断道路", value: `${hazardState?.road_reachability.filter((road) => !road.accessible).length ?? 0}`, hint: "影响路线" },
    {
      label: "资源车辆",
      value: `${resourceView?.resource_status?.vehicle_count ?? 0}`,
      hint: resourceView?.resource_status ? `${resourceView.resource_status.staff_count} 人待命` : "等待资源表",
    },
  ];

  const riskDistributionItems = [
    { label: "红色预警", value: focusObjects.filter((item) => item.risk_level === "Red").length, className: styles.riskRed },
    { label: "橙色预警", value: focusObjects.filter((item) => item.risk_level === "Orange").length, className: styles.riskOrange },
    { label: "黄色预警", value: focusObjects.filter((item) => item.risk_level === "Yellow").length, className: styles.riskYellow },
    { label: "蓝色预警", value: focusObjects.filter((item) => item.risk_level === "Blue").length, className: styles.riskBlue },
  ];

  const footerMetrics = [
    { label: "数据源在线率", value: resourceView ? "98.6%" : "等待", detail: "多源接入" },
    { label: "摄像头在线", value: resourceView?.resource_status ? `${Math.max(resourceView.resource_status.vehicle_count * 8, 24)} / ${Math.max(resourceView.resource_status.vehicle_count * 8 + 12, 36)}` : "等待", detail: "视频感知" },
    { label: "雨量站在线", value: "48 / 50", detail: "雨量监测" },
    { label: "水位站在线", value: "35 / 36", detail: "河道水位" },
    { label: "处置闭环率", value: `${recentWarningDrafts.length ? 100 : approvedProposals.length ? 76 : 42}%`, detail: "审批到预警" },
    { label: "降雨强度", value: "12.6 毫米", detail: "近 1 小时" },
    { label: "平均水位", value: "2.38 米", detail: "重点断面" },
    { label: "预警命中率", value: "89.7%", detail: "历史回测" },
  ];

  const patrolBars = dataSourceItems.slice(0, 6).map((item, index) => ({
    ...item,
    height: 28 + ((index + 2) * 17) % 58,
  }));

  const variantFeedItems =
    variant === "risk-warning"
      ? [
          { meta: "现在 / 实测", title: "雨强维持中高位", detail: "1 小时累计降雨 12.6 毫米，低洼路段需持续巡查。" },
          { meta: "1 小时后 / 预测", title: "短时强降雨靠近", detail: "模型提示建设路片区仍有局地增强，可能触发排水压力。" },
          { meta: "2 小时后 / 预测", title: "风险向学校医院扩散", detail: "文艺路小学、碑林中心医院周边需提前预置人员。" },
          { meta: "3 小时后 / 研判", title: "橙色风险保持", detail: "若水位继续上升，建议触发社区转移准备预案。" },
        ]
      : variant === "impact-analysis"
        ? recentWarningDrafts.length
          ? recentWarningDrafts.map((draft) => ({
              meta: `${formatTimestamp(draft.created_at)} / ${draft.channel}`,
              title: `${draft.audience} 预警版本已生成`,
              detail: draft.grounding_summary,
            }))
          : [
              { meta: "10:15 / 领导版", title: "跨部门协同预警待生成", detail: "需要确认影响边界、重点对象和行动口径。" },
              { meta: "10:12 / 部门版", title: "排水与交通联动待校验", detail: "发布前需核对道路阻断与避难路线。" },
              { meta: "10:08 / 公众版", title: "社区避险提示待发布", detail: "需转换为短句、明确行动和避险位置。" },
            ]
        : variant === "event-replay"
          ? [
              { meta: "10:08 / 触发", title: "雨量站触发橙色风险", detail: "系统识别低洼网格积水深度持续上升。" },
              { meta: "10:12 / 研判", title: "智能体形成影响链", detail: "影响对象从道路扩展到学校、医院和社区人群。" },
              { meta: "10:18 / 决策", title: "生成处置方案并进入审批", detail: "资源调度、封控排涝和分众预警被纳入闭环。" },
              { meta: "10:23 / 复盘", title: "预警触达与处置效果回收", detail: "形成改进建议并沉淀为后续样例。" },
            ]
          : focusSignals.slice(0, 5).map((signal) => ({
              meta: `${formatTimestamp(signal.created_at)} / ${signalSeverityLabel(signal.severity)}`,
              title: normalizeAgentTerminology(signal.title),
              detail: normalizeAgentTerminology(signal.detail),
            }));

  const telemetryMetrics =
    variant === "risk-warning"
      ? [
          { label: "模型更新频率", value: "每 5 分钟", detail: "降雨与水位滚动刷新" },
          { label: "预测准确率", value: "91.3%", detail: "近 30 日回测" },
          { label: "未来高风险区", value: `${Math.max(highRiskObjectCount, 1)}`, detail: "3 小时预测" },
          { label: "降雨峰值", value: "38 毫米/小时", detail: "未来峰值" },
          { label: "水位站在线", value: "35 / 36", detail: "河道断面" },
          { label: "预案触发条件", value: "2 项临界", detail: "水位 / 道路" },
        ]
      : variant === "impact-analysis"
        ? [
            { label: "预警命中率", value: "89.7%", detail: "历史样本回测" },
            { label: "预计影响人口", value: "2.86 万", detail: "按重点对象估算" },
            { label: "分众版本", value: `${Math.max(recentWarningDrafts.length, 3)}`, detail: "多受众文本" },
            { label: "受众覆盖率", value: "94.2%", detail: "社区与部门触达" },
            { label: "发布留痕", value: `${recentWarningDrafts.length || 0} 条`, detail: "关联 proposal" },
            { label: "边界校验", value: "3 / 4", detail: "道路、人口、避难点" },
          ]
        : variant === "event-replay"
          ? [
              { label: "处置闭环率", value: `${recentWarningDrafts.length ? 100 : approvedProposals.length ? 76 : 42}%`, detail: "审批到发布" },
              { label: "平均响应时间", value: "6.2 分钟", detail: "研判到派单" },
              { label: "资源利用率", value: `${resourceView?.resource_status ? 92 : 86}%`, detail: "车辆与人员" },
              { label: "到场时长", value: "18.6 分钟", detail: "重点对象平均" },
              { label: "复盘节点", value: "4 个", detail: "触发、研判、审批、发布" },
              { label: "经验沉淀", value: `${Math.max(commandRiskReminders.length, 3)} 条`, detail: "预案优化" },
            ]
          : footerMetrics;

  const rightVariantSections =
    variant === "risk-warning"
      ? [
          {
            eyebrow: "智能研判摘要",
            title: "未来风险预测",
            mode: "timeline",
            items: [
              { label: "研判 1", text: `${focusDisplayName} 仍处于${currentRiskName}风险，未来 3 小时需关注短时强降雨与排水压力。` },
              { label: "研判 2", text: `高风险对象 ${Math.max(highRiskObjectCount, 1)} 个，学校、医院与社区入口是优先巡查对象。` },
              { label: "研判 3", text: "若河道水位继续上升 0.3 米，建议启动转移准备和道路临时管控。" },
            ],
          },
          {
            eyebrow: "预警触发条件",
            title: "防范建议与预案条件",
            mode: "timeline",
            items: [
              { label: "条件 1", text: "未来 1 小时降雨强度超过 35 毫米/小时。" },
              { label: "条件 2", text: "重点水位站连续两次上报超警戒水位。" },
              { label: "条件 3", text: "低洼道路出现回流或车辆通行能力下降。" },
            ],
          },
          {
            eyebrow: "资源预置建议",
            title: "提前调度准备",
            mode: "actions",
            items: [
              { label: "动作 1", text: "向学校、医院周边预置排涝车和网格员。" },
              { label: "动作 2", text: "通知交通与排水部门准备临时封控方案。" },
              { label: "动作 3", text: "提前生成社区避险提示，等待触发条件确认。" },
            ],
          },
        ]
      : variant === "impact-analysis"
        ? [
            {
              eyebrow: "预警发布记录",
              title: "发布进度",
              mode: "timeline",
              items: recentWarningDrafts.length
                ? recentWarningDrafts.map((draft) => ({
                    label: draft.audience,
                    text: `${draft.channel} 版本已生成：${draft.grounding_summary}`,
                  }))
                : [
                    { label: "领导版", text: "突出风险趋势、跨部门动作和人工审批边界。" },
                    { label: "部门版", text: "拆解排水、交通、社区和医疗侧执行要求。" },
                    { label: "公众版", text: "转成短句提醒，强调绕行、避险和关注官方消息。" },
                  ],
            },
            {
              eyebrow: "影响范围统计",
              title: "受众与对象覆盖",
              mode: "timeline",
              items: [
                { label: "人口", text: "预计影响 2.86 万人，重点覆盖低洼社区、学校和医院周边。" },
                { label: "道路", text: `${hazardState?.road_reachability.filter((road) => !road.accessible).length ?? 0} 条道路存在通行风险，需在发布前核对绕行建议。` },
                { label: "避难点", text: "避难点覆盖满足演示场景要求，但仍需现场确认容量。" },
              ],
            },
            {
              eyebrow: "发布前校验项",
              title: "发布校验",
              mode: "actions",
              items: [
                { label: "校验 1", text: "预警边界与地图范围一致。" },
                { label: "校验 2", text: "分众文本包含对象、风险、行动建议和证据摘要。" },
                { label: "校验 3", text: "发布记录、审批记录和 proposal_id 可追溯。" },
              ],
            },
          ]
        : variant === "event-replay"
          ? [
              {
                eyebrow: "处置效果评估",
                title: "响应质量",
                mode: "timeline",
                items: [
                  { label: "响应", text: "从风险识别到处置派单约 6.2 分钟，满足演示闭环要求。" },
                  { label: "资源", text: "车辆、人员与避难点联动完成，资源利用率保持在 90% 左右。" },
                  { label: "触达", text: "分众预警覆盖领导、部门、社区和公众四类对象。" },
                ],
              },
              {
                eyebrow: "AI 复盘摘要",
                title: "关键结论",
                mode: "timeline",
                items: [
                  { label: "结论 1", text: "低洼网格积水上涨是本次链路的主要触发因素。" },
                  { label: "结论 2", text: "学校和医院周边道路需要更早进入绕行提示。" },
                  { label: "结论 3", text: "审批后预警生成链路完整，但资源到场反馈可继续细化。" },
                ],
              },
              {
                eyebrow: "改进建议",
                title: "预案优化点",
                mode: "actions",
                items: [
                  { label: "优化 1", text: "将水位连续上涨阈值前置到预案触发条件。" },
                  { label: "优化 2", text: "补充学校、医院、老人等重点对象的专属避险话术。" },
                  { label: "优化 3", text: "把本次处置样例沉淀为后续智能体评测 fixture。" },
                ],
              },
            ]
          : [];

  const variantBottomPanels =
    variant === "risk-warning"
      ? {
          firstTitle: "模型与预警指标",
          firstSubtitle: "模型运行",
          firstBars: [
            { label: "更新频率", value: "5 分钟", height: 78 },
            { label: "准确率", value: "91.3%", height: 91 },
            { label: "命中率", value: "89.7%", height: 89 },
            { label: "雨量峰值", value: "38", height: 76 },
            { label: "水位趋势", value: "上升", height: 64 },
            { label: "触发项", value: "2", height: 52 },
          ],
          secondTitle: "未来风险对象",
          secondSubtitle: "预测对象",
          graph: focusObjects.slice(0, 4).map((item) => ({
            subtitle: riskLevelLabel(item.risk_level),
            title: item.name,
            text: `${item.village} / 预计 ${item.time_to_impact_minutes} 分钟内受影响`,
          })),
          thirdTitle: "资源预置",
          thirdSubtitle: "调度准备",
          snapshot: [
            { label: "排涝车辆", value: `${resourceView?.resource_status?.vehicle_count ?? 0}`, hint: "可预置到低洼点" },
            { label: "现场人员", value: `${resourceView?.resource_status?.staff_count ?? 0}`, hint: "网格与抢险联动" },
            { label: "触发预案", value: "2 项", hint: "转移与道路管控" },
          ],
        }
      : variant === "impact-analysis"
        ? {
            firstTitle: "预警效果指标",
            firstSubtitle: "发布质量",
            firstBars: [
              { label: "命中率", value: "89.7%", height: 90 },
              { label: "覆盖率", value: "94.2%", height: 94 },
              { label: "触达率", value: "92.6%", height: 92 },
              { label: "边界校验", value: "3/4", height: 75 },
              { label: "版本数", value: `${Math.max(recentWarningDrafts.length, 3)}`, height: 62 },
              { label: "留痕", value: `${recentWarningDrafts.length}`, height: 44 },
            ],
            secondTitle: "分众预警版本",
            secondSubtitle: "受众覆盖",
            graph: [
              { subtitle: "领导版", title: "指挥决策", text: "突出趋势、资源和跨部门协同。" },
              { subtitle: "部门版", title: "执行编排", text: "明确排水、交通、社区与医疗职责。" },
              { subtitle: "公众版", title: "避险提示", text: "短句提示绕行、避险与官方渠道。" },
            ],
            thirdTitle: "发布留痕",
            thirdSubtitle: "可追溯",
            snapshot: [
              { label: "预警草稿", value: `${recentWarningDrafts.length}`, hint: "关联当前方案" },
              { label: "影响对象", value: `${focusObjects.length}`, hint: "边界内对象" },
              { label: "校验项", value: "3 / 4", hint: "发布前检查" },
            ],
          }
        : variant === "event-replay"
          ? {
              firstTitle: "复盘闭环指标",
              firstSubtitle: "效果评估",
              firstBars: [
                { label: "闭环率", value: `${recentWarningDrafts.length ? 100 : approvedProposals.length ? 76 : 42}%`, height: recentWarningDrafts.length ? 100 : approvedProposals.length ? 76 : 42 },
                { label: "响应", value: "6.2", height: 82 },
                { label: "到场", value: "18.6", height: 66 },
                { label: "资源", value: "92%", height: 92 },
                { label: "触达", value: "94%", height: 94 },
                { label: "沉淀", value: "3", height: 58 },
              ],
              secondTitle: "过程时间轴",
              secondSubtitle: "回放节点",
              graph: [
                { subtitle: "10:08", title: "风险触发", text: "雨量、水位与道路状态形成初始告警。" },
                { subtitle: "10:12", title: "影响研判", text: "识别学校、医院、社区与道路影响链。" },
                { subtitle: "10:18", title: "处置审批", text: "方案进入审批并准备分众预警。" },
                { subtitle: "10:23", title: "复盘沉淀", text: "形成经验规则和后续改进项。" },
              ],
              thirdTitle: "经验沉淀",
              thirdSubtitle: "预案优化",
              snapshot: [
                { label: "经验规则", value: "3 条", hint: "水位、道路、重点对象" },
                { label: "样例链路", value: "1 条", hint: "可回放演示样例" },
                { label: "Prompt 修订", value: "2 项", hint: "证据和避险话术" },
              ],
            }
          : null;

  const bottomSourceBars = variantBottomPanels?.firstBars ?? patrolBars;
  const bottomGraphItems =
    variantBottomPanels?.graph ??
    impactGraphColumns.map((column) => ({
      subtitle: column.subtitle,
      title: column.title,
      text: normalizeAgentTerminology(column.items[0] ?? column.fallback ?? "等待数据"),
    }));
  const bottomSnapshotItems = variantBottomPanels?.snapshot ?? operationsSnapshot;

  useEffect(() => {
    setSelectedPendingProposalId((current) => {
      if (current && linkedPendingProposals.some((proposal) => proposal.proposal_id === current)) {
        return current;
      }
      return linkedPendingProposals[0]?.proposal_id ?? null;
    });
  }, [linkedPendingProposals]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className={`${styles.screen} ${styles.cityBigScreen} ${toneClass}`}>
      <div className={styles.cityAura} />

      <motion.header
        className={styles.cityHeader}
        initial={{ opacity: 0, y: -18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className={`${styles.cityHeaderWing} ${styles.cityBrandWing}`}>
          <div className={styles.cityLogoMark}>涝</div>
          <div>
            <span>智汇水务</span>
            <strong>碑林区重点片区</strong>
          </div>
          <em>区域选择</em>
        </div>
        <div className={styles.cityTitleBlock}>
          <p>{screenProfile.eyebrow}</p>
          <h2>{screenProfile.title}</h2>
          <small>{overview?.event_title ?? screenProfile.subtitle}</small>
        </div>
        <div className={`${styles.cityHeaderWing} ${styles.cityStatusWing}`}>
          <div>
            <span>当前时间</span>
            <strong>{formatClock(now)}</strong>
            <small>{formatCalendar(now)}</small>
          </div>
          <div>
            <span>天气</span>
            <strong>中雨 22-25℃</strong>
            <small>降雨持续</small>
          </div>
          <div>
            <span>值班状态</span>
            <strong>{streamStatusLabel(streamStatus)}</strong>
            <small>张建明 值守中</small>
          </div>
        </div>
      </motion.header>

      <nav className={styles.cityModuleTabs} aria-label="大屏功能分页">
        <NavLink to="/" end>态势总览</NavLink>
        <NavLink to="/copilot">智能问答</NavLink>
        <NavLink to="/operations">风险预警</NavLink>
        <NavLink to="/agents">预警分析</NavLink>
        <NavLink to="/reliability">事件复盘</NavLink>
      </nav>

      <section className={styles.cityModeBanner} aria-label="当前大屏场景">
        <span>{screenProfile.eyebrow}</span>
        <strong>{screenProfile.title}</strong>
        <small>{screenProfile.subtitle}</small>
      </section>

      <motion.section
        className={styles.cityKpiRibbon}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.36, delay: 0.04 }}
      >
        {pageKpis.map((item) => (
          <article key={item.label} className={styles.cityKpiCard}>
            <i aria-hidden="true" />
            <div>
              <span>{item.label}</span>
              <strong className={item.accent}>{item.value}</strong>
              <small>{item.detail}</small>
            </div>
          </article>
        ))}
      </motion.section>

      <div className={styles.cityDashboardGrid}>
        <motion.aside
          className={`${styles.citySidePanel} ${styles.cityLeftPanel}`}
          initial={{ opacity: 0, x: -24 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.42, delay: 0.08 }}
        >
          <section className={styles.cityPanelSection}>
            <div className={styles.cityPanelTitle}>
              <span>{screenProfile.leftSubtitle}</span>
              <h3>{screenProfile.leftTitle}</h3>
            </div>
            <div className={styles.situationGauge}>
              <div className={styles.situationGaugeRing}>
                <strong className={riskClassName(selectedRiskLevel)}>{riskLevelLabel(selectedRiskLevel)}</strong>
                <span>{hazardState ? `${Math.round(hazardState.overall_score)} 分` : "评分"}</span>
              </div>
              <div className={styles.riskDistributionMini}>
                {riskDistributionItems.map((item) => (
                  <article key={item.label}>
                    <i className={item.className} />
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                  </article>
                ))}
              </div>
            </div>
            <p className={styles.citySituationSummary}>{summaryText}</p>
          </section>

          <section className={styles.cityPanelSection}>
            <div className={styles.cityPanelTitle}>
              <span>{screenProfile.listTitle}</span>
              <h3>{screenProfile.listTitle}</h3>
            </div>
            <div className={styles.cityObjectList}>
              {focusObjects.slice(0, 7).map((item) => (
                <button
                  key={item.object_id}
                  type="button"
                  className={`${styles.cityObjectRow} ${item.object_id === selectedObjectId ? styles.cityObjectRowActive : ""}`}
                  onClick={() => void onSelectObject(item.object_id)}
                >
                  <i className={riskClassName(item.risk_level)} />
                  <span>
                    <strong>{item.name}</strong>
                    <small>{item.village} / {item.time_to_impact_minutes} 分钟</small>
                  </span>
                  <em>{riskLevelLabel(item.risk_level)}</em>
                </button>
              ))}
            </div>
          </section>

          <section className={styles.cityPanelSection}>
            <div className={styles.cityPanelTitle}>
              <span>{screenProfile.feedTitle}</span>
              <h3>{screenProfile.feedTitle}</h3>
            </div>
            <div className={styles.signalFeed}>
              {variantFeedItems.slice(0, 5).map((item, index) => (
                <article key={`${variant}-feed-${index}`}>
                  <span>{item.meta}</span>
                    <strong>{normalizeAgentTerminology(item.title)}</strong>
                    <p>{normalizeAgentTerminology(item.detail)}</p>
                </article>
              ))}
              {!variantFeedItems.length ? <p className={styles.cityEmpty}>等待雨量、水位、网格上报或舆情信号接入。</p> : null}
            </div>
          </section>
        </motion.aside>

        <motion.main
          className={styles.cityMapDeck}
          initial={{ opacity: 0, scale: 0.985 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.48, delay: 0.1 }}
        >
          <div className={styles.cityMapCanvasFrame}>
            <Suspense
              fallback={
                <div className={styles.canvasSkeleton}>
                  <strong>正在加载数字孪生</strong>
                  <p>正在加载三维城市模型、风险热区和指挥叙事镜头。</p>
                </div>
              }
            >
              <DigitalTwinCesiumCanvas
                layers={mapLayers}
                dialogFocusObjectId={dialogFocusObjectId}
                dialogFocusSerial={dialogEntries.length}
                routeHighlightObjectId={routeHighlightObjectId}
                selectedRiskLevel={selectedRiskLevel}
                onSelectObject={onSelectObject}
              />
            </Suspense>
          </div>

          <div className={styles.cityMapBottomOverlay}>
            {spatialStatusItems.slice(0, 4).map((item) => (
              <button
                key={item.object_id}
                type="button"
                className={`${styles.cityMapStatusChip} ${mapStateClassName(item.proposal_state)} ${
                  item.object_id === selectedObjectId ? styles.cityMapStatusChipActive : ""
                }`}
                onClick={() => void onSelectObject(item.object_id)}
              >
                <span>{item.label}</span>
                <strong>{item.name}</strong>
              </button>
            ))}
          </div>
        </motion.main>

        <motion.aside
          className={`${styles.citySidePanel} ${styles.cityRightPanel}`}
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.42, delay: 0.08 }}
        >
          {variant === "overview" ? (
            <>
          <section className={styles.cityPanelSection}>
            <div className={styles.cityPanelTitle}>
              <span>{screenProfile.analysisTitle}</span>
              <h3>{screenProfile.analysisTitle}</h3>
            </div>
            <div className={styles.commandTimeline}>
              {rightPanelAnalysisItems.slice(0, 4).map((item, index) => (
                <p key={`${variant}-analysis-${index}`}>{normalizeAgentTerminology(item)}</p>
              ))}
              {!rightPanelAnalysisItems.length ? (
                <p>选择对象后展示“风险源 → 道路/社区 → 人群资源 → 处置方案”的解释链。</p>
              ) : null}
            </div>
          </section>

          <section className={styles.cityPanelSection}>
            <div className={styles.cityPanelTitle}>
              <span>{screenProfile.actionTitle}</span>
              <h3>{screenProfile.actionTitle}</h3>
            </div>
            <div className={styles.commandActionStack}>
              {rightPanelActionItems.length ? (
                rightPanelActionItems.slice(0, 4).map((action, index) => (
                  <article key={`${action}-${index}`}>
                    <span>动作 {index + 1}</span>
                    <strong>{normalizeAgentTerminology(action)}</strong>
                  </article>
                ))
              ) : (
                <p className={styles.cityEmpty}>等待智能体会商或处置方案生成。</p>
              )}
            </div>
          </section>

          <section className={styles.cityPanelSection}>
            <div className={styles.cityPanelTitle}>
              <span>闭环流程</span>
              <h3>审批与预警闭环</h3>
            </div>
            <article className={`${styles.cityClosureCard} ${closureComplete ? styles.cityClosureComplete : ""}`}>
              <span>闭环状态</span>
              <strong>{closureStatus}</strong>
              <small>{closureStatusDetail}</small>
            </article>

            {selectedPendingProposal ? (
              <div className={styles.cityApprovalBox}>
                <span>待审批方案</span>
                <strong>{normalizeAgentTerminology(selectedPendingProposal.title)}</strong>
                <p>{normalizeAgentTerminology(selectedPendingProposal.summary)}</p>
                <textarea
                  value={operatorNote}
                  onChange={(event) => setOperatorNote(event.target.value)}
                  placeholder="输入审批意见，例如：先转移低洼院落居民，保留消防通道。"
                />
                <div className={styles.cityActionButtons}>
                  <button
                    type="button"
                    disabled={actionBusy}
                    onClick={() => void onResolveProposal(selectedPendingProposal.proposal_id, "reject", operatorNote)}
                  >
                    驳回
                  </button>
                  <button
                    type="button"
                    disabled={actionBusy}
                    onClick={() => void onResolveProposal(selectedPendingProposal.proposal_id, "approve", operatorNote)}
                  >
                    批准动作
                  </button>
                </div>
              </div>
            ) : null}

            <div className={styles.cityActionButtons}>
              <button type="button" onClick={() => void onGenerateProposals()}>
                生成方案
              </button>
              <button
                type="button"
                disabled={!linkedApprovedProposal || actionBusy}
                onClick={() => linkedApprovedProposal && void onGenerateWarnings(linkedApprovedProposal.proposal_id)}
              >
                生成预警
              </button>
            </div>
          </section>
            </>
          ) : (
            <>
              {rightVariantSections.map((section) => (
                <section className={styles.cityPanelSection} key={section.title}>
                  <div className={styles.cityPanelTitle}>
                    <span>{section.eyebrow}</span>
                    <h3>{section.title}</h3>
                  </div>
                  {section.mode === "actions" ? (
                    <div className={styles.commandActionStack}>
                      {section.items.map((item, index) => (
                        <article key={`${section.title}-${item.label}-${index}`}>
                          <span>{item.label}</span>
                          <strong>{normalizeAgentTerminology(item.text)}</strong>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.commandTimeline}>
                      {section.items.map((item, index) => (
                        <p key={`${section.title}-${item.label}-${index}`}>{normalizeAgentTerminology(item.text)}</p>
                      ))}
                    </div>
                  )}
                </section>
              ))}
            </>
          )}
        </motion.aside>

      </div>

      <motion.section
        className={styles.cityTelemetryDock}
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.16 }}
        aria-label="底部运行指标"
      >
        {telemetryMetrics.map((item) => (
          <article key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <small>{item.detail}</small>
          </article>
        ))}
      </motion.section>

      <motion.section
        className={styles.cityBottomGrid}
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.46, delay: 0.14 }}
      >
        <section className={styles.cityBottomPanel}>
          <div className={styles.cityPanelTitle}>
            <span>{variantBottomPanels?.firstSubtitle ?? screenProfile.bottomTitle}</span>
            <h3>{variantBottomPanels?.firstTitle ?? screenProfile.bottomTitle}</h3>
          </div>
          <div className={styles.sourceMeterGrid}>
            {bottomSourceBars.map((item) => (
              <article key={item.label}>
                <div className={styles.sourceBarTrack}>
                  <i style={{ height: `${item.height}%` }} />
                </div>
                <strong>{item.value}</strong>
                <span>{item.label}</span>
              </article>
            ))}
          </div>
        </section>

        <section className={styles.cityBottomPanel}>
          <div className={styles.cityPanelTitle}>
            <span>{variantBottomPanels?.secondSubtitle ?? "链路图谱"}</span>
            <h3>{variantBottomPanels?.secondTitle ?? "影响链图谱"}</h3>
          </div>
          <div className={styles.cityImpactGraph}>
            {bottomGraphItems.map((column, index) => (
              <article key={column.title}>
                <span>{column.subtitle}</span>
                <strong>{column.title}</strong>
                <p>{normalizeAgentTerminology(column.text)}</p>
                {index < bottomGraphItems.length - 1 ? <i /> : null}
              </article>
            ))}
          </div>
        </section>

        <section className={styles.cityBottomPanel}>
          <div className={styles.cityPanelTitle}>
            <span>{variantBottomPanels?.thirdSubtitle ?? "处置运行"}</span>
            <h3>{variantBottomPanels?.thirdTitle ?? "工单与资源"}</h3>
          </div>
          <div className={styles.opsSnapshot}>
            {bottomSnapshotItems.map((item) => (
              <article key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <small>{item.hint}</small>
              </article>
            ))}
          </div>
          <button type="button" className={styles.opsOpenButton} onClick={onOpenOperations}>
            进入协同处置页
          </button>
        </section>
      </motion.section>

    </section>
  );
}
