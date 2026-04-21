import { proposalStatusText, riskLevelText } from "../lib/displayText";
import type {
  ActionProposalV2,
  AgentTaskStatus,
  AutonomyLevel,
  EntityImpactView,
  EntityProfile,
  ResourceStatus,
  RiskLevel,
  TravelMode,
} from "../types/api";
import type { ConsoleBootState, ExecutionStatus } from "../types/ui";

export const quickPrompts = [
  "当前水位变化对低洼区老年人意味着什么？",
  "这对附近小学现在意味着什么？",
  "这对周边工厂库存意味着什么？",
];

export const pageMeta = {
  "/": {
    label: "风险总览",
    title: "全局风险态势总览",
    description: "围绕当前洪涝事件，集中展示风险等级、变化趋势、重点对象和待处置事项。",
  },
  "/copilot": {
    label: "影响问答",
    title: "影响研判问答",
    description: "围绕重点对象和当前事件自由提问，查看影响过程、证据依据和下一步建议。",
  },
  "/operations": {
    label: "协同处置",
    title: "多 Agent 协同处置",
    description: "展示系统如何围绕当前风险任务完成协同、工具调用和方案生成。",
  },
  "/data": {
    label: "数据管理",
    title: "运行期数据源与数据管线",
    description: "管理画像、资源、知识文档和碑林区数据抓取刷新。",
  },
  "/agents": {
    label: "多 Agent 协同",
    title: "多 Agent 协同、共享记忆与评测",
    description: "查看多 Agent 调度、任务时间线、触发总线、经验上下文和评测结果。",
  },
  "/reliability": {
    label: "可靠性与审计",
    title: "运行健康、审计与权限策略",
    description: "查看后台巡检健康、告警、审计、归档状态和当前角色权限。",
  },
} as const;

export const entityTypeOptions: Array<{
  value: EntityImpactView["entity"]["entity_type"];
  label: string;
}> = [
  { value: "resident", label: "居民" },
  { value: "school", label: "学校" },
  { value: "factory", label: "工厂" },
  { value: "hospital", label: "医院" },
  { value: "nursing_home", label: "养老机构" },
  { value: "metro_station", label: "地铁站" },
  { value: "underground_space", label: "地下空间" },
  { value: "community", label: "社区" },
];

export const travelModeOptions: Array<{ value: TravelMode; label: string }> = [
  { value: "walk", label: "步行" },
  { value: "vehicle", label: "车辆转移" },
  { value: "assisted", label: "协助转移" },
];

export const resourceFields: Array<{
  key: keyof ResourceStatus;
  label: string;
  type: "number" | "textarea";
}> = [
  { key: "vehicle_count", label: "车辆", type: "number" },
  { key: "staff_count", label: "工作人员", type: "number" },
  { key: "supply_kits", label: "物资包", type: "number" },
  { key: "rescue_boats", label: "救援船艇", type: "number" },
  { key: "ambulance_count", label: "救护车", type: "number" },
  { key: "drone_count", label: "无人机", type: "number" },
  { key: "portable_pumps", label: "移动排水泵", type: "number" },
  { key: "power_generators", label: "发电机", type: "number" },
  { key: "medical_staff_count", label: "医护人员", type: "number" },
  { key: "volunteer_count", label: "志愿者", type: "number" },
  { key: "satellite_phones", label: "卫星电话", type: "number" },
  { key: "notes", label: "备注", type: "textarea" },
];

export const riskText: Record<RiskLevel, string> = riskLevelText;

export const entityText: Record<EntityImpactView["entity"]["entity_type"], string> = {
  resident: "居民",
  school: "学校",
  factory: "工厂",
  hospital: "医院",
  nursing_home: "养老机构",
  metro_station: "地铁站",
  underground_space: "地下空间",
  community: "社区",
};

export const proposalText: Record<ActionProposalV2["status"], string> = proposalStatusText;

export const autonomyText: Record<AutonomyLevel, string> = {
  auto_observe: "自动观测",
  auto_recommend: "自动建议",
  human_gate_required: "需要人工确认",
};

export const taskStatusText: Record<AgentTaskStatus, string> = {
  pending: "待处理",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  canceled: "已取消",
  superseded: "已替代",
};

export const bootStateText: Record<ConsoleBootState, string> = {
  booting: "启动中",
  ready: "就绪",
  degraded: "降级",
  error: "异常",
};

export const executionStatusText: Record<ExecutionStatus, string> = {
  idle: "待命",
  planning: "规划中",
  awaiting_confirmation: "等待确认",
  running: "执行中",
  success: "已完成",
  error: "执行异常",
};

export const healthStateText = {
  checking: "检查中",
  online: "在线",
  offline: "离线",
} as const;

export const blankProfile = (areaId: string): EntityProfile => ({
  entity_id: "",
  area_id: areaId,
  entity_type: "resident",
  name: "",
  village: "",
  location_hint: "",
  resident_count: 0,
  current_occupancy: 0,
  vulnerability_tags: [],
  mobility_constraints: [],
  key_assets: [],
  inventory_summary: "",
  continuity_requirement: "",
  preferred_transport_mode: "walk",
  notification_preferences: [],
  emergency_contacts: [],
  custom_attributes: {},
});

export const blankResourceStatus = (areaId: string): ResourceStatus => ({
  area_id: areaId,
  vehicle_count: 0,
  staff_count: 0,
  supply_kits: 0,
  rescue_boats: 0,
  ambulance_count: 0,
  drone_count: 0,
  portable_pumps: 0,
  power_generators: 0,
  medical_staff_count: 0,
  volunteer_count: 0,
  satellite_phones: 0,
  notes: "",
});

export const primaryPaths = new Set(["/", "/copilot", "/operations"]);
