import { formatAgentHandoffTarget, formatAgentTaskType } from "../lib/displayText";
import type {
  AgentCouncilView,
  AgentResult,
  DecisionReportView,
  FocusObjectView,
  HazardStateV2,
  ResourceStatusView,
  SharedMemorySnapshot,
  TwinFocusObjectSummary,
  TwinOverviewView,
  TwinSignalView,
} from "../types/api";

export interface AgentDivergenceRow {
  result: AgentResult;
  disposition: string;
  rationale: string;
  disagreement: string;
  confidence: number;
}

export interface SituationSourceItem {
  label: string;
  value: string;
  status: string;
  detail: string;
}

export interface ImpactGraphColumn {
  title: string;
  subtitle: string;
  items: string[];
  fallback?: string;
}

export function buildAgentDivergenceRows(params: {
  recentResults: AgentResult[];
  sharedMemorySnapshot?: SharedMemorySnapshot | null;
  decisionReport?: DecisionReportView | null;
  agentCouncil?: AgentCouncilView | null;
  maxRows?: number;
}): AgentDivergenceRow[] {
  const acceptedResultIds = new Set([
    ...(params.sharedMemorySnapshot?.recent_result_ids ?? []),
    ...(params.decisionReport?.recent_result_ids ?? []),
    ...(params.agentCouncil?.recent_result_ids ?? []),
  ]);

  return params.recentResults.slice(0, params.maxRows ?? 5).map((result) => {
    const isAccepted = acceptedResultIds.has(result.result_id);
    const confidence = result.decision_confidence ?? result.confidence;
    const hasEvidenceGap = result.missing_slots.length > 0 || result.evidence_refs.length === 0;
    const disposition = isAccepted ? "已采纳" : hasEvidenceGap ? "保留补证" : confidence < 0.58 ? "暂不采纳" : "作为备选";
    const rationale = isAccepted
      ? params.agentCouncil?.overall_summary ||
        params.decisionReport?.latest_summary ||
        params.sharedMemorySnapshot?.latest_summary ||
        "Supervisor 已把该结果纳入当前编排路径。"
      : hasEvidenceGap
        ? `证据缺口：${result.missing_slots.slice(0, 2).join(" / ") || "缺少可追溯证据引用"}`
        : confidence < 0.58
          ? "决策置信度未达到自动编排阈值，保留给人工复核。"
          : result.handoff_recommendations.length
            ? `转交 ${result.handoff_recommendations.map((item) => formatAgentHandoffTarget(item)).slice(0, 2).join(" / ")} 后再推进。`
            : "建议方向合理，但当前主链路优先级更低。";
    const disagreement =
      result.missing_slots[0] ??
      result.handoff_recommendations.map((item) => formatAgentHandoffTarget(item))[0] ??
      result.recommended_next_tasks?.map((item) => formatAgentTaskType(item))[0] ??
      "与其他角色的主要差异集中在行动优先级、资源约束或证据覆盖。";

    return {
      result,
      disposition,
      rationale,
      disagreement,
      confidence,
    };
  });
}

export function buildSituationSourceItems(params: {
  hazardState?: HazardStateV2 | null;
  resourceView?: ResourceStatusView | null;
  signals: TwinSignalView[];
  activeAlertCount: number;
  streamStatusLabel: string;
}): SituationSourceItem[] {
  const monitoringPoints = params.hazardState?.monitoring_points ?? [];
  const roadReachability = params.hazardState?.road_reachability ?? [];
  const blockedRoads = roadReachability.filter((road) => !road.accessible);
  const topHazardTile =
    [...(params.hazardState?.hazard_tiles ?? [])].sort((left, right) => right.risk_score - left.risk_score)[0] ?? null;
  const maxRainfallMm = monitoringPoints.reduce((max, point) => Math.max(max, point.latest_rainfall_mm), 0);
  const maxWaterLevelM = monitoringPoints.reduce((max, point) => Math.max(max, point.latest_water_level_m), 0);
  const resourceStatus = params.resourceView?.resource_status ?? null;

  return [
    {
      label: "雨量",
      value: monitoringPoints.length ? `${Math.round(maxRainfallMm)} mm` : "--",
      status: monitoringPoints.length ? "Live telemetry" : "Waiting",
      detail: monitoringPoints[0]?.point_name ?? "接入气象站/泵站后显示峰值雨量",
    },
    {
      label: "水位",
      value: monitoringPoints.length ? `${maxWaterLevelM.toFixed(1)} m` : "--",
      status: params.hazardState ? params.hazardState.trend : "Snapshot",
      detail: topHazardTile ? `${topHazardTile.area_name} 预测积水 ${topHazardTile.predicted_water_depth_cm}cm` : "等待水位或积水预测模型输出",
    },
    {
      label: "道路",
      value: `${blockedRoads.length}/${roadReachability.length || "--"}`,
      status: blockedRoads.length ? "Blocked routes" : "Clear",
      detail: blockedRoads[0]?.failure_reason || blockedRoads[0]?.name || "道路可达性用于驱动阻断线与处置路线",
    },
    {
      label: "网格上报",
      value: `${params.signals.length}`,
      status: params.signals.length ? "Active reports" : "No new report",
      detail: params.signals[0]?.title ?? "网格员巡查、现场照片和补充描述会进入信号流",
    },
    {
      label: "群众报险 / 舆情",
      value: `${params.activeAlertCount}`,
      status: params.streamStatusLabel,
      detail: "群众报险、热线和舆情信号用于补足传感器盲区",
    },
    {
      label: "资源状态",
      value: resourceStatus ? `${resourceStatus.vehicle_count} 车 / ${resourceStatus.staff_count} 人` : "--",
      status: params.resourceView?.scope ?? "resource baseline",
      detail: resourceStatus
        ? `泵 ${resourceStatus.portable_pumps} / 舟艇 ${resourceStatus.rescue_boats} / 无人机 ${resourceStatus.drone_count}`
        : "接入区域或事件级资源表后显示调度余量",
    },
  ];
}

export function buildImpactGraphColumns(params: {
  overview?: TwinOverviewView | null;
  hazardState?: HazardStateV2 | null;
  focusObjects: TwinFocusObjectSummary[];
  focusObject?: FocusObjectView | null;
  resourceView?: ResourceStatusView | null;
  primaryProposalTitle?: string | null;
  warningDraftCount: number;
  closureStatus: string;
  riskReminders: string[];
}): ImpactGraphColumn[] {
  const blockedRoads = (params.hazardState?.road_reachability ?? []).filter((road) => !road.accessible);
  const topHazardTile =
    [...(params.hazardState?.hazard_tiles ?? [])].sort((left, right) => right.risk_score - left.risk_score)[0] ?? null;
  const affectedRoadNames = Array.from(
    new Set([
      ...(topHazardTile?.affected_roads ?? []),
      ...blockedRoads.map((road) => road.name),
      ...(params.hazardState?.road_reachability ?? []).slice(0, 2).map((road) => road.name),
    ]),
  ).slice(0, 3);
  const resourceStatus = params.resourceView?.resource_status ?? null;

  return [
    {
      title: "风险源",
      subtitle: "Rain / Water",
      items: [
        params.overview?.event_title ?? "当前洪水事件",
        topHazardTile ? `${topHazardTile.area_name} ${topHazardTile.predicted_water_depth_cm}cm` : "积水预测等待接入",
      ],
    },
    {
      title: "传导对象",
      subtitle: "Road / Facility",
      items: affectedRoadNames.length ? affectedRoadNames : ["道路可达性", "下穿/低洼点"],
    },
    {
      title: "对象链路",
      subtitle: "Community / Hospital / School",
      items: params.focusObjects.slice(0, 3).map((item) => `${item.name} ${item.time_to_impact_minutes}min`),
      fallback: params.focusObject?.object_name ?? params.overview?.lead_object_name ?? "等待对象画像",
    },
    {
      title: "人群与资源",
      subtitle: "People / Resources",
      items: [
        resourceStatus ? `可调车辆 ${resourceStatus.vehicle_count} / 人员 ${resourceStatus.staff_count}` : "资源基线待接入",
        params.riskReminders[0] ?? "脆弱人群与资源缺口待智能体补证",
      ],
    },
    {
      title: "处置闭环",
      subtitle: "Proposal / Warning",
      items: [
        params.primaryProposalTitle ?? "等待 proposal 生成",
        params.warningDraftCount ? `${params.warningDraftCount} 类 warning 已生成` : params.closureStatus,
      ],
    },
  ];
}
