import type {
  ActionProposalV2,
  AgentCouncilView,
  AgentDialogResponse,
  AudienceWarningDraft,
  FocusObjectView,
  HazardStateV2,
  OperationalAlert,
  RegionalProposalQueueSnapshot,
  RegionalProposalView,
  ResourceStatusView,
  TwinOverviewView,
  V2EventRecord,
  WarningGenerationResponse,
} from "../types/api";

export const agentTwinDemoModeEnabled = import.meta.env.VITE_DEMO_MODE === "true";

const DEMO_NOW = "2026-04-28T10:30:00+08:00";
const DEMO_EVENT_ID = "event_demo_beilin_primary";
const DEMO_AREA_ID = "beilin_10km2";
const DEMO_EVENT_TITLE = "碑林区强降雨内涝风险演示事件";

export const demoEvent: V2EventRecord = {
  event_id: DEMO_EVENT_ID,
  area_id: DEMO_AREA_ID,
  title: DEMO_EVENT_TITLE,
  trigger_reason: "production_demo_mode",
  current_stage: "AgentTwin demo",
  current_risk_level: "Orange",
  status: "active",
  metadata: { demo_mode: true },
  created_at: DEMO_NOW,
  updated_at: DEMO_NOW,
};

const demoEvidence = [
  {
    evidence_type: "sensor",
    title: "南门雨量站 15 分钟累计雨量",
    source_id: "sensor_rain_nanmen_15m",
    excerpt: "15 分钟累计雨量 36mm，短临预报显示强回波仍在东移。",
    timestamp: DEMO_NOW,
    priority: 0,
  },
  {
    evidence_type: "road",
    title: "建设路低洼段道路通行状态",
    source_id: "road_jsl_underpass",
    excerpt: "低洼段积水深度已接近 42cm，社区网格员上报非机动车绕行需求。",
    timestamp: DEMO_NOW,
    priority: 1,
  },
  {
    evidence_type: "policy",
    title: "城市内涝黄色以上响应 SOP",
    source_id: "sop_urban_flood_yellow_plus",
    excerpt: "涉及学校、医院、地铁枢纽时，应先形成会商意见并进入人工审批闸门。",
    timestamp: DEMO_NOW,
    priority: 2,
  },
];

export const demoFocusObjects: FocusObjectView[] = [
  {
    event_id: DEMO_EVENT_ID,
    object_id: "community_jsl_grid",
    object_name: "建设路低洼网格三组",
    entity_type: "community",
    village: "建设路片区",
    risk_level: "Orange",
    time_to_impact_minutes: 12,
    summary: "低洼网格积水面正在外扩，地下室回流和社区出入口阻断风险同时上升。",
    risk_reasons: ["建设路低洼段水深接近 42cm。", "网格员连续两次上报地下室回流。", "最近避难路径需绕行北侧支路。"],
    recommended_actions: ["批准社区网格排涝封控 proposal。", "派出资源车辆沿北侧路线前置抽排设备。", "生成社区版和公众版 warning。"],
    risk_reminders: ["审批前不得自动封控主干道。", "需要保留医院急诊保障通道。"],
    evidence: demoEvidence,
    related_proposals: [],
  },
  {
    event_id: DEMO_EVENT_ID,
    object_id: "school_wyl_primary",
    object_name: "文艺路小学",
    entity_type: "school",
    village: "文艺路片区",
    risk_level: "Yellow",
    time_to_impact_minutes: 18,
    summary: "学校周边道路可能在放学前出现短时积水，建议提前通知家长错峰接送。",
    risk_reasons: ["校门口下穿通道水位抬升。", "学生离校窗口与强降雨重叠。"],
    recommended_actions: ["预置校门口引导人员。", "生成家长与学校管理方 warning。"],
    risk_reminders: ["不要引导家长进入低洼道路。"],
    evidence: demoEvidence.slice(0, 2),
    related_proposals: [],
  },
  {
    event_id: DEMO_EVENT_ID,
    object_id: "hospital_bl_center",
    object_name: "碑林中心医院",
    entity_type: "hospital",
    village: "南门片区",
    risk_level: "Orange",
    time_to_impact_minutes: 15,
    summary: "医院急诊与后勤入口需要保持连续通行，已批准入口保障动作并生成分众预警。",
    risk_reasons: ["急诊入口坡道存在倒灌风险。", "后勤车辆需要避开南侧低洼点。"],
    recommended_actions: ["维持急诊入口沙袋与排水泵。", "继续发布部门版 warning。"],
    risk_reminders: ["医院保障动作优先级高于普通路段封控。"],
    evidence: demoEvidence,
    related_proposals: [],
  },
  {
    event_id: DEMO_EVENT_ID,
    object_id: "metro_nsm_hub",
    object_name: "南门地铁换乘枢纽",
    entity_type: "metro_station",
    village: "南门片区",
    risk_level: "Yellow",
    time_to_impact_minutes: 24,
    summary: "地铁口客流与积水风险同步上升，需要强化出入口提示和绕行指引。",
    risk_reasons: ["枢纽南侧道路通行效率下降。", "群众报险量增加。"],
    recommended_actions: ["发布公众绕行 warning。", "安排地铁口临时引导。"],
    risk_reminders: ["避免把人流引向医院急诊保障通道。"],
    evidence: demoEvidence.slice(1),
    related_proposals: [],
  },
];

function createProposal(
  overrides: Partial<ActionProposalV2> & Pick<ActionProposalV2, "proposal_id" | "title" | "summary" | "status">,
): ActionProposalV2 {
  return {
    event_id: DEMO_EVENT_ID,
    entity_id: "community_jsl_grid",
    area_id: DEMO_AREA_ID,
    proposal_scope: "regional",
    action_type: "regional_flood_control",
    execution_mode: "resource_dispatch",
    action_display_name: overrides.title,
    action_display_tagline: overrides.summary,
    action_display_category: "空间联动处置",
    trigger_reason: "AgentTwin demo council",
    recommendation: overrides.summary,
    evidence_summary: "基于雨量、水位、道路状态、网格员上报和 SOP 约束生成。",
    severity: "Orange",
    requires_confirmation: true,
    required_operator_roles: ["commander"],
    payload: { demo_mode: true },
    high_risk_object_ids: ["community_jsl_grid", "hospital_bl_center"],
    action_scope: {
      target_scope: "建设路低洼网格、碑林中心医院急诊入口、北侧绕行路线",
      resource_plan: "1 台移动泵车、2 名网格员、1 组交警联动",
    },
    risk_stage_key: "orange_response",
    system_version_hash: "agent-twin-demo",
    generation_source: "system",
    model_name: "demo-fixture",
    prompt_profile: "production_demo",
    grounding_summary: "演示模式固定样例，字段结构与真实接口保持一致。",
    chat_follow_up_prompt: "请解释该 proposal 为什么需要人工审批。",
    source_session_id: "demo_session",
    updated_at: DEMO_NOW,
    edited_by_commander: false,
    last_editor: "demo_operator",
    has_new_system_suggestion: false,
    superseded_by_proposal_id: null,
    withdrawn_reason: "",
    resolved_at: overrides.status === "pending" ? null : DEMO_NOW,
    resolved_by: overrides.status === "pending" ? null : "frontend_console",
    resolution_note: overrides.status === "pending" ? "" : "演示模式预置批准动作。",
    created_at: DEMO_NOW,
    ...overrides,
  };
}

export const demoPendingProposalSeed: ActionProposalV2[] = [
  createProposal({
    proposal_id: "demo_proposal_community_jsl_grid",
    title: "建设路低洼网格地下室回流协助与排涝封控",
    summary: "对社区低洼网格执行临时排涝、地下室回流排查和北侧路线引导。",
    status: "pending",
  }),
];

export const demoApprovedProposalSeed: ActionProposalV2[] = [
  createProposal({
    proposal_id: "demo_proposal_hospital_bl_center",
    entity_id: "hospital_bl_center",
    title: "碑林中心医院急诊与后勤入口保障",
    summary: "保持急诊入口通行，前置排水泵与沙袋，后勤车辆绕开南侧低洼道路。",
    status: "approved",
    execution_mode: "evacuation_task",
    action_display_category: "医疗保障",
    high_risk_object_ids: ["hospital_bl_center", "community_jsl_grid"],
  }),
];

export const demoWarningDraftSeed: AudienceWarningDraft[] = [
  {
    warning_id: "demo_warning_leader_1",
    event_id: DEMO_EVENT_ID,
    proposal_id: "demo_proposal_hospital_bl_center",
    audience: "leader",
    channel: "dashboard",
    content: "建议继续保持医院急诊入口保障动作，并同步关注建设路低洼网格外扩风险。",
    grounding_summary: "领导版：突出风险趋势、跨部门动作和人工审批边界。",
    created_at: DEMO_NOW,
  },
  {
    warning_id: "demo_warning_department_1",
    event_id: DEMO_EVENT_ID,
    proposal_id: "demo_proposal_hospital_bl_center",
    audience: "department",
    channel: "work_order",
    content: "城管、交警、卫健部门按已批准 proposal 执行入口保障、绕行引导和排水设备前置。",
    grounding_summary: "部门版：拆解执行主体、路径和资源要求。",
    created_at: DEMO_NOW,
  },
  {
    warning_id: "demo_warning_public_1",
    event_id: DEMO_EVENT_ID,
    proposal_id: "demo_proposal_hospital_bl_center",
    audience: "public",
    channel: "sms",
    content: "南门至建设路片区积水风险上升，请避开低洼道路，勿进入地下空间。",
    grounding_summary: "公众版：短句、行动明确、避免专业术语。",
    created_at: DEMO_NOW,
  },
];

export function buildDemoOverview(params: {
  pendingProposals: ActionProposalV2[];
  approvedProposals: ActionProposalV2[];
  warningDrafts: AudienceWarningDraft[];
}): TwinOverviewView {
  const pendingIds = new Set(params.pendingProposals.flatMap((item) => item.high_risk_object_ids ?? []));
  const approvedIds = new Set(params.approvedProposals.flatMap((item) => item.high_risk_object_ids ?? []));
  const warningIds = new Set(
    params.warningDrafts.flatMap((draft) =>
      params.approvedProposals
        .filter((proposal) => proposal.proposal_id === draft.proposal_id)
        .flatMap((proposal) => proposal.high_risk_object_ids ?? []),
    ),
  );

  return {
    event_id: DEMO_EVENT_ID,
    area_id: DEMO_AREA_ID,
    event_title: DEMO_EVENT_TITLE,
    generated_at: DEMO_NOW,
    overall_risk_level: "Orange",
    trend: "rapidly_rising",
    summary: "演示模式已锁定碑林区主事件、重点对象、审批动作与分众 warning，适合甲方现场稳定展示。",
    lead_object_id: "community_jsl_grid",
    lead_object_name: "建设路低洼网格三组",
    focus_objects: demoFocusObjects.map((item, index) => ({
      object_id: item.object_id,
      name: item.object_name,
      entity_type: item.entity_type,
      village: item.village,
      risk_level: item.risk_level,
      time_to_impact_minutes: item.time_to_impact_minutes,
      summary: item.summary,
      recommended_action: item.recommended_actions[0] ?? "继续跟踪。",
      pending_proposal_ids: params.pendingProposals
        .filter((proposal) => (proposal.high_risk_object_ids ?? []).includes(item.object_id))
        .map((proposal) => proposal.proposal_id),
      canvas_position: { left: 22 + (index % 3) * 24, top: 28 + Math.floor(index / 3) * 24 },
    })),
    map_layers: demoFocusObjects.map((item, index) => ({
      object_id: item.object_id,
      name: item.object_name,
      risk_level: item.risk_level,
      entity_type: item.entity_type,
      is_lead: item.object_id === "community_jsl_grid",
      east_offset_m: [-40, -230, 130, 310][index] ?? 0,
      north_offset_m: [10, 150, -85, 120][index] ?? 0,
      height_offset_m: 20 + index * 5,
      proposal_state: warningIds.has(item.object_id)
        ? "warning_generated"
        : pendingIds.has(item.object_id)
          ? "pending"
          : approvedIds.has(item.object_id)
            ? "approved"
            : "monitoring",
    })),
    pending_proposal_count: params.pendingProposals.length,
    approved_proposal_count: params.approvedProposals.length,
    warning_draft_count: params.warningDrafts.length,
    active_alert_count: 6,
    recommended_actions: ["先处置建设路低洼网格地下室回流。", "保持碑林中心医院急诊入口保障。", "审批后同步生成分众 warning。"],
    signals: [
      {
        signal_id: "demo_signal_rain",
        title: "雨量站短时强降雨持续",
        detail: "南门雨量站 15 分钟累计雨量 36mm，预计 20 分钟内仍有强回波经过。",
        severity: "warning",
        created_at: DEMO_NOW,
      },
      {
        signal_id: "demo_signal_grid",
        title: "网格员上报地下室回流",
        detail: "建设路低洼网格三组出现地下室回流，需前置排涝和巡查。",
        severity: "critical",
        created_at: DEMO_NOW,
      },
      {
        signal_id: "demo_signal_route",
        title: "北侧绕行路线仍可用",
        detail: "交警反馈北侧支路可作为资源车辆和社区绕行路径。",
        severity: "info",
        created_at: DEMO_NOW,
      },
    ],
    recent_warning_drafts: params.warningDrafts,
  };
}

export const demoAgentCouncil: AgentCouncilView = {
  event_id: DEMO_EVENT_ID,
  generated_at: DEMO_NOW,
  overall_summary: "会商一致认为建设路低洼网格是当前最适合展示闭环处置的焦点对象。",
  decision_path: ["ImpactAgent 定位社区与医院的联动风险。", "ActionAgent 建议优先排涝封控和路线引导。", "AuditAgent 要求人工审批后再执行高风险动作。"],
  open_questions: ["是否需要同步学校家长通知？", "医院后勤车辆是否需要二次绕行？"],
  blocked_by: ["高风险封控动作必须由 commander 审批。"],
  roles: [
    {
      role: "impact_agent",
      label: "Impact Agent",
      status: "ready",
      summary: "识别社区低洼网格、医院急诊入口和学校放学窗口的耦合风险。",
      confidence: 0.86,
      evidence_count: 3,
      recommended_action: "先聚焦建设路低洼网格。",
    },
    {
      role: "action_agent",
      label: "Action Agent",
      status: "ready",
      summary: "建议用北侧路线前置资源车辆，并对地下空间进行巡查。",
      confidence: 0.81,
      evidence_count: 3,
      recommended_action: "生成区域排涝封控 proposal。",
    },
    {
      role: "audit_agent",
      label: "Audit Agent",
      status: "gate_required",
      summary: "封控和资源调度影响交通秩序，必须保留人工审批闸门。",
      confidence: 0.9,
      evidence_count: 2,
      recommended_action: "批准前不得自动执行。",
    },
  ],
  audit_decision: {
    status: "approved_for_review",
    summary: "可进入人工审批，但不能自动执行。",
    rationale: "证据足够支撑 proposal 生成，动作涉及道路封控和资源调度，需要指挥员确认。",
    risk_flags: ["traffic_control", "medical_access", "human_gate_required"],
    approval_required: true,
  },
  recent_result_ids: ["demo_result_impact", "demo_result_action", "demo_result_audit"],
};

export const demoHazardState: HazardStateV2 = {
  event_id: DEMO_EVENT_ID,
  area_id: DEMO_AREA_ID,
  generated_at: DEMO_NOW,
  overall_risk_level: "Orange",
  overall_score: 78,
  trend: "rapidly_rising",
  uncertainty: 0.22,
  freshness_seconds: 45,
  hazard_tiles: [
    {
      tile_id: "demo_tile_jsl",
      area_name: "建设路低洼网格",
      horizon_minutes: 30,
      risk_level: "Orange",
      risk_score: 82,
      predicted_water_depth_cm: 42,
      trend: "rising",
      uncertainty: 0.2,
      affected_roads: ["建设路低洼段", "北侧支路"],
    },
  ],
  road_reachability: [
    {
      road_id: "demo_route_north",
      name: "北侧资源绕行路线",
      from_village: "建设路片区",
      to_location: "体育学院避难点",
      accessible: true,
      travel_time_minutes: 13,
      depth_limit_cm: 28,
      failure_reason: "",
    },
  ],
  monitoring_points: [
    {
      point_name: "南门雨量站",
      latest_water_level_m: 1.36,
      latest_rainfall_mm: 36,
      status: "warning",
      updated_at: DEMO_NOW,
    },
  ],
};

export const demoResourceStatusView: ResourceStatusView = {
  scope: "event",
  area_id: DEMO_AREA_ID,
  event_id: DEMO_EVENT_ID,
  resource_status: {
    area_id: DEMO_AREA_ID,
    vehicle_count: 5,
    staff_count: 28,
    supply_kits: 36,
    rescue_boats: 1,
    ambulance_count: 2,
    drone_count: 2,
    portable_pumps: 4,
    power_generators: 2,
    medical_staff_count: 8,
    volunteer_count: 42,
    satellite_phones: 2,
    notes: "演示模式固定资源态势：1 台泵车可前置到建设路北侧。",
  },
};

export const demoAlerts: OperationalAlert[] = [
  {
    alert_id: "demo_alert_stream",
    source_type: "sse",
    severity: "info",
    status: "open",
    summary: "演示模式实时链路已固定",
    details: "VITE_DEMO_MODE=true 时前端使用固定快照，避免现场数据波动。",
    event_id: DEMO_EVENT_ID,
    first_seen_at: DEMO_NOW,
    last_seen_at: DEMO_NOW,
  },
];

export function findDemoFocusObject(objectId?: string | null): FocusObjectView {
  return demoFocusObjects.find((item) => item.object_id === objectId) ?? demoFocusObjects[0];
}

export function buildDemoRegionalView(proposal: ActionProposalV2): RegionalProposalView {
  return {
    proposal,
    event_title: DEMO_EVENT_TITLE,
    current_risk_level: "Orange",
    high_risk_object_names: (proposal.high_risk_object_ids ?? [])
      .map((id) => demoFocusObjects.find((item) => item.object_id === id)?.object_name)
      .filter(Boolean) as string[],
  };
}

export function buildDemoQueueSnapshot(pendingProposals: ActionProposalV2[]): RegionalProposalQueueSnapshot {
  return {
    queue_version: `demo_${pendingProposals.length}_${DEMO_NOW}`,
    generated_at: DEMO_NOW,
    items: pendingProposals.map(buildDemoRegionalView),
  };
}

export function buildDemoDialogResponse(message: string, objectId?: string | null): AgentDialogResponse {
  const focus = findDemoFocusObject(objectId);
  return {
    event_id: DEMO_EVENT_ID,
    object_id: focus.object_id,
    object_name: focus.object_name,
    message,
    answer: `${focus.object_name} 当前的核心风险是：${focus.summary} 建议先看证据，再进入 proposal 审批。`,
    impact_summary: focus.risk_reasons,
    evidence: focus.evidence,
    recommended_actions: focus.recommended_actions,
    risk_reminders: focus.risk_reminders,
    follow_up_prompts: ["为什么这个动作不能自动执行？", "请把影响链拆成风险源、对象、人群和资源。", "审批后应该生成哪些受众版本的 warning？"],
    grounding_summary: "演示模式结构化回答，固定返回影响链、证据、建议动作和审批入口。",
    proposal_entry: {
      blocked: false,
      proposal: buildDemoRegionalView(demoPendingProposalSeed[0]),
    },
    response_source: "demo_fixture",
    generated_at: DEMO_NOW,
  };
}

export function buildDemoWarnings(proposalId: string): WarningGenerationResponse {
  return {
    event_id: DEMO_EVENT_ID,
    proposal_id: proposalId,
    generated_at: DEMO_NOW,
    warnings: demoWarningDraftSeed.map((draft) => ({ ...draft, proposal_id: proposalId })),
  };
}
