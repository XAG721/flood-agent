import type { EntityType, RiskLevel, TwinObjectMapLayer } from "../../types/api";
import type {
  EvidenceRole,
  ResponseTaskStatus,
  RiskObjectRegistryRecord,
  SimulationDispatchScenario,
  WorkflowRole,
} from "../../types/response";

export const roleText: Record<WorkflowRole, string> = {
  duty_officer: "防办值班员",
  reviewer: "防办审核员",
  commander: "指挥审批员",
  liaison: "成员单位联络员",
  field_operator: "现场执行员",
  auditor: "审计查看员",
  admin: "系统管理员",
};

export const statusText: Record<ResponseTaskStatus, string> = {
  draft: "草稿",
  pending_approval: "待审批",
  issued: "已下发",
  acknowledged: "已确认",
  in_progress: "执行中",
  pending_verification: "待核实",
  completed: "已完成",
  escalated: "已升级",
  waived: "已豁免",
  blocked: "执行受阻",
  partially_completed: "部分完成",
  cancelled: "已撤回",
  taken_over: "人工接管",
};

export const actionText: Record<string, string> = {
  event_created: "响应事件已创建",
  alert_version_appended: "预警版本已追加",
  candidate_added: "风险对象进入待核验清单",
  candidate_updated: "风险对象信息已更新",
  candidate_discovery_completed: "区域对象台账筛查已完成",
  candidate_confirmed: "风险对象已确认",
  candidate_excluded: "风险对象已排除",
  candidate_object_list_frozen: "确认对象清单已冻结",
  task_draft_created: "任务草案已生成",
  task_draft_revised: "任务草案已修订",
  task_submitted: "任务已提交审批",
  task_approved_and_issued: "任务已审批并下发",
  task_rejected_to_draft: "任务已退回草稿",
  task_acknowledged: "责任岗位已确认接收",
  task_assigned: "任务已分派现场执行员",
  task_reassigned: "任务已重新分派",
  task_started: "任务开始执行",
  completion_submitted: "完成结果已提交核实",
  completion_verified: "完成结果核实通过",
  completion_returned: "完成结果退回补充",
  task_blocked_and_escalated: "任务受阻并升级",
  deadline_escalated: "任务超时并升级",
  task_waived: "任务已批准豁免",
  event_closed: "响应事件已关闭",
  grounded_task_draft_generated: "证据角色校验通过并生成任务草案",
  task_draft_blocked_by_evidence_gate: "任务草案因证据不足被阻断",
  duplicate_feedback_merged: "重复反馈已合并",
  event_review_draft_generated: "事件复盘草稿已生成",
  scenario_evaluation_completed: "区县场景评测已完成",
  situation_feedback_recorded: "现场态势更新已记录",
  risk_object_registry_imported: "风险对象主数据已更新，相关候选运行已失效",
};

export const dispatchScenarioText: Record<SimulationDispatchScenario, string> = {
  normal: "正常接收与送达",
  timeout: "网关超时（可重试）",
  reject: "外部通道拒收",
  partial_success: "批量部分成功",
  duplicate_callback: "重复回调",
  out_of_order_callback: "乱序回调",
};

export const evidenceRoleText: Record<EvidenceRole, string> = {
  condition: "触发条件",
  object: "风险对象",
  responsibility: "责任岗位",
  procedure: "处置流程",
  exception: "例外升级",
  attribution: "版本归因",
};

export const taskFieldText: Record<string, string> = {
  trigger_condition: "触发条件",
  risk_object: "风险对象",
  responsible_party: "责任主体",
  action: "处置动作",
  deadline: "完成时限",
  resource_dependency: "资源依赖",
  feedback_requirement: "反馈要求",
  escalation_condition: "升级条件",
  exception_condition: "例外条件",
};

export const missingReasonText: Record<string, string> = {
  SOURCE_ABSENT_CONFIRMED: "已确认来源无此规定",
  NOT_RETRIEVED: "本次检索未命中",
  INDEX_INCOMPLETE: "索引尚不完整",
  SOURCE_UNAVAILABLE: "来源当前不可用",
};

export function formatDate(value?: string | null) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(value));
}

const registryEntityTypes: Array<[string, EntityType]> = [
  ["学校", "school"],
  ["医院", "hospital"],
  ["养老", "nursing_home"],
  ["地铁", "metro_station"],
  ["地下", "underground_space"],
  ["下穿", "underground_space"],
  ["工厂", "factory"],
  ["企业", "factory"],
  ["居民", "resident"],
];

function registryRiskLevel(score: number): RiskLevel {
  if (score >= 85) return "Red";
  if (score >= 70) return "Orange";
  if (score >= 50) return "Yellow";
  return "Blue";
}

function registryEntityType(value: string): EntityType {
  const normalized = value.toLowerCase();
  return registryEntityTypes.find(([keyword]) => normalized.includes(keyword))?.[1] ?? "community";
}

export function registryMapLayers(records: RiskObjectRegistryRecord[]): TwinObjectMapLayer[] {
  const anchorLon = 108.94921153512861;
  const anchorLat = 34.24624474240188;
  const metersPerLongitudeDegree = 111_320 * Math.cos((anchorLat * Math.PI) / 180);
  return records
    .filter((item) => item.longitude != null && item.latitude != null && item.duplicate_of == null)
    .map((item, index) => ({
      object_id: item.object_id,
      name: item.name,
      risk_level: registryRiskLevel(item.risk_score),
      entity_type: registryEntityType(item.object_type),
      east_offset_m: (Number(item.longitude) - anchorLon) * metersPerLongitudeDegree,
      north_offset_m: (Number(item.latitude) - anchorLat) * 110_540,
      height_offset_m: 0,
      proposal_state: item.registry_status === "active" ? "monitoring" : "warning_generated",
      is_lead: index === 0,
    }));
}
