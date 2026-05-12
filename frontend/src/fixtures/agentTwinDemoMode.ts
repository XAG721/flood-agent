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
  current_stage: "智能体演示",
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
    excerpt: "15 分钟累计雨量 36 毫米，短临预报显示强回波仍在东移。",
    timestamp: DEMO_NOW,
    priority: 0,
  },
  {
    evidence_type: "road",
    title: "建设路低洼段道路通行状态",
    source_id: "road_jsl_underpass",
    excerpt: "低洼段积水深度已接近 36 厘米，社区网格员上报非机动车绕行需求。",
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
    risk_reasons: ["建设路低洼段水深接近 36 厘米。", "网格员连续两次上报地下室回流。", "最近避难路径需绕行北侧支路。"],
    recommended_actions: ["批准社区网格排涝封控处置方案。", "派出资源车辆沿北侧路线前置抽排设备。", "生成社区版和公众版预警。"],
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
    recommended_actions: ["预置校门口引导人员。", "家长车辆沿文艺路北侧临停点接送，避开建设路低洼段。", "生成家长与学校管理方预警。"],
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
    recommended_actions: ["维持急诊入口沙袋与排水泵。", "救护车和后勤车辆沿北侧保障路线进出，避开南侧低洼点。", "继续发布部门版预警。"],
    risk_reminders: ["医院保障动作优先级高于普通路段封控。"],
    evidence: demoEvidence,
    related_proposals: [],
  },
  {
    event_id: DEMO_EVENT_ID,
    object_id: "resident_linnai_home",
    object_name: "李奶奶（独居老人）",
    entity_type: "resident",
    village: "建设路片区",
    risk_level: "Red",
    time_to_impact_minutes: 8,
    summary: "李奶奶住所位于一层低洼院落，出入口积水抬升，需要社区志愿者陪同转移。",
    risk_reasons: ["院落门前积水预计 8 分钟内达到 28-32 厘米。", "李奶奶行动不便，独自转移风险较高。", "安全转移路线需绕开建设路低洼段。"],
    recommended_actions: ["派社区志愿者上门确认李奶奶状态。", "沿文艺路北侧支路转移至南门街道临时安置点。", "同步通知家属和社区网格员。"],
    risk_reminders: ["不得让老人自行涉水外出。", "转移路线需避开地下空间和低洼下穿通道。"],
    evidence: [
      ...demoEvidence.slice(0, 2),
      {
        evidence_type: "profile",
        title: "重点人群关怀名单",
        source_id: "profile_elder_linnai",
        excerpt: "李奶奶，独居老人，行动不便，需志愿者或社区工作人员协助转移。",
        timestamp: DEMO_NOW,
        priority: 3,
      },
    ],
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
    recommended_actions: ["发布公众绕行预警。", "安排地铁口临时引导。"],
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
    trigger_reason: "智能体会商演示",
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
    model_name: "演示模型",
    prompt_profile: "production_demo",
    grounding_summary: "演示模式固定样例，字段结构与真实接口保持一致。",
    chat_follow_up_prompt: "请解释该处置方案为什么需要人工审批。",
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
    proposal_id: "demo_proposal_school_wyl_primary",
    entity_id: "school_wyl_primary",
    title: "文艺路小学放学避险与家长接送引导",
    summary: "在放学窗口前设置校门口引导，家长车辆沿北侧临停点接送，避开建设路低洼段。",
    status: "pending",
    execution_mode: "evacuation_task",
    action_display_category: "学校避险",
    high_risk_object_ids: ["school_wyl_primary", "community_jsl_grid"],
    action_scope: {
      target_scope: "文艺路小学、北侧临停点、建设路低洼段绕行路线",
      resource_plan: "2 名校门口引导员、1 组交警联动、1 名社区网格员协同",
    },
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
    content: "城管、交警、卫健部门按已批准处置方案执行入口保障、绕行引导和排水设备前置。",
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
    summary: "演示模式已锁定碑林区主事件、重点对象、审批动作与分众预警，适合甲方现场稳定展示。",
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
      east_offset_m: [-40, -230, 130, -115, 310][index] ?? 0,
      north_offset_m: [10, 150, -85, -165, 120][index] ?? 0,
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
    recommended_actions: ["先处置建设路低洼网格地下室回流。", "保障医院急诊入口和学校放学路线。", "优先协助李奶奶等重点人群转移。", "审批后同步生成分众预警。"],
    signals: [
      {
        signal_id: "demo_signal_rain",
        title: "雨量站短时强降雨持续",
        detail: "南门雨量站 15 分钟累计雨量 36 毫米，预计 20 分钟内仍有强回波经过。",
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
  overall_summary: "会商一致认为建设路低洼网格、学校、医院与李奶奶重点人群可以覆盖客户演示中的不同对象类型。",
  decision_path: ["影响研判智能体定位社区、学校、医院和独居老人的联动风险。", "行动建议智能体建议优先排涝封控、路线引导和重点人群转移。", "审计智能体要求人工审批后再执行高风险动作。"],
  open_questions: ["是否需要同步学校家长通知？", "医院后勤车辆是否需要二次绕行？", "李奶奶是否已经完成上门确认？"],
  blocked_by: ["高风险封控动作必须由指挥长审批。"],
  roles: [
    {
      role: "impact_agent",
      label: "影响研判智能体",
      status: "ready",
      summary: "识别社区低洼网格、医院急诊入口、学校放学窗口和李奶奶住所的耦合风险。",
      confidence: 0.86,
      evidence_count: 3,
      recommended_action: "先聚焦建设路低洼网格。",
    },
    {
      role: "action_agent",
      label: "行动建议智能体",
      status: "ready",
      summary: "建议用北侧路线前置资源车辆，并对地下空间进行巡查。",
      confidence: 0.81,
      evidence_count: 3,
      recommended_action: "生成区域排涝封控处置方案。",
    },
    {
      role: "audit_agent",
      label: "审计智能体",
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
    rationale: "证据足够支撑处置方案生成，动作涉及道路封控和资源调度，需要指挥员确认。",
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
      predicted_water_depth_cm: 36,
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
      latest_water_level_m: 0.36,
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

function buildDialogProposalForFocus(focus: FocusObjectView) {
  if (focus.object_id === "school_wyl_primary") {
    return demoPendingProposalSeed[0];
  }

  return createProposal({
    proposal_id: `demo_dialog_proposal_${focus.object_id}`,
    entity_id: focus.object_id,
    title: `${focus.object_name}避险与保障处置`,
    summary: focus.recommended_actions.slice(0, 2).join("；"),
    status: "pending",
    high_risk_object_ids: [focus.object_id],
    action_scope: {
      target_scope: `${focus.object_name}及周边安全路线`,
      resource_plan: focus.entity_type === "resident" ? "1 名网格员、2 名志愿者、1 辆转运车" : "现场引导人员与应急资源前置",
    },
    action_display_category:
      focus.entity_type === "school"
        ? "学校避险"
        : focus.entity_type === "hospital"
          ? "医疗保障"
          : focus.entity_type === "resident"
            ? "重点人群转移"
            : "空间联动处置",
  });
}

function joinDemoItems(items: string[]) {
  return items
    .map((item) => item.trim().replace(/[；;,.，、。！？]+$/u, ""))
    .filter(Boolean)
    .join("；");
}

function buildDemoDialogAnswer(message: string, focus: FocusObjectView) {
  const normalizedMessage = message.trim();
  const riskReasons = joinDemoItems(focus.risk_reasons.slice(0, 3));
  const actions = joinDemoItems(focus.recommended_actions.slice(0, 3));
  const reminders = joinDemoItems(focus.risk_reminders.slice(0, 2));
  const routeReminder = reminders || "路线执行前仍需现场人员确认道路未被临时占用";
  const defaultReminder = reminders || "执行前需要现场复核道路、水位和对象状态";
  const objectIntro = `${focus.object_name}当前被智能体识别为${focus.risk_level}级关注对象，预计影响窗口约${focus.time_to_impact_minutes}分钟。影响研判智能体综合雨量、水位、道路通行、对象画像和现场上报后，认为它的核心风险不是单一积水点，而是会影响周边道路、重点人群和应急资源调度的联动风险。主要证据包括：${riskReasons}。`;

  if (/路线|路径|绕行|转移|撤离|避险|怎么走|通行/.test(normalizedMessage)) {
    return `${objectIntro}行动建议智能体生成路线时，不采用对象之间的直线连接，而是先把起点吸附到三维地图中的主干道路，再避开建设路低洼段、地下空间入口和建筑密集区，沿北侧或中部主路接近目标区域。这样做的原因是：一方面可以减少涉水距离，另一方面可以保证排水泵车、救援车辆、社区转运车辆具备通行条件。针对${focus.object_name}，智能体建议的处置动作是：${actions}。如果涉及老人、学生或医院入口，路线还会优先考虑步行安全、车辆掉头空间和急诊通道连续性。需要提醒的是：${routeReminder}。因此，这条路径可以作为指挥员审批和现场调度的推荐路线，而不是自动强制执行的最终命令。`;
  }

  if (/审批|人工|自动执行|不能自动|闸门|确认/.test(normalizedMessage)) {
    return `${objectIntro}审计智能体判断，本次建议虽然证据较充分，但仍然不能直接自动执行。原因是处置动作涉及道路封控、资源车辆调度、重点人员转移和公众预警发布，这些都会影响交通秩序、医院急诊通道和居民出行权利，必须由具备权限的指挥员确认。智能体可以自动生成风险解释、推荐路线、处置方案和预警草稿，但最终是否封控、何时派车、由哪个部门执行，需要人工审批后才能进入执行链路。针对${focus.object_name}，系统建议先核对证据链：${riskReasons}，再确认动作边界：${actions}。这种设计体现的是“智能体生成方案、指挥员负责授权”的协同模式，既能提高研判效率，也避免在现场情况变化时误触发高影响动作。`;
  }

  if (/预警|发布|受众|领导|部门|公众|短信|通知/.test(normalizedMessage)) {
    return `${objectIntro}预警沟通智能体会在处置方案通过审批后，基于同一个风险事实生成不同受众版本，而不是简单复制一段通用通知。领导版会突出风险趋势、影响范围、跨部门协同和人工审批边界，帮助快速掌握事件级态势；部门版会拆解执行主体、资源需求、道路路径和到场要求，便于排水、交通、社区和医疗保障协同；公众版会尽量避免专业术语，用短句说明避开低洼道路、不要进入地下空间、按照现场引导绕行。围绕${focus.object_name}，智能体会引用这些证据：${riskReasons}，并把建议动作转成可发布口径：${actions}。因此，预警内容既有统一事实基础，又能匹配不同对象的理解能力和行动需求。`;
  }

  if (/影响链|证据|风险链|原因|为什么|资源|对象|拆/.test(normalizedMessage)) {
    return `${objectIntro}影响研判智能体会把当前事件拆成四层链路：第一层是风险源，包括短时强降雨、低洼路段积水和地下空间回流；第二层是传导道路，包括建设路低洼段、北侧绕行路线和医院周边保障通道；第三层是影响对象，包括社区、学校、医院、地铁口和独居老人等重点对象；第四层是处置资源，包括移动排水泵、网格员、志愿者、交通协同力量和后勤车辆。对于${focus.object_name}，当前证据链为：${riskReasons}。智能体据此生成的建议动作是：${actions}。这套回答的价值在于让指挥员不仅看到“风险等级”，还能理解风险为什么会扩散、优先保护谁、资源应该沿哪条路径进入，以及哪些动作必须进入审批闭环。`;
  }

  return `${objectIntro}行动建议智能体基于这些证据生成了面向现场的处置建议：${actions}。这些建议和地图上的路径都由智能体根据当前对象位置、道路可达性、积水范围、重点人群画像和资源状态综合生成，不是人工随意标注。系统同时给出风险提醒：${defaultReminder}。从指挥流程看，智能体先完成风险识别和影响链拆解，再生成可审批的处置方案、推荐路线和分众预警草稿；指挥员负责确认方案边界、调整执行口径并决定是否发布。这样既能让演示展示出自动化研判能力，也能体现应急场景中对人工授权和安全边界的尊重。`;
}

export function buildDemoDialogResponse(message: string, objectId?: string | null): AgentDialogResponse {
  const focus = findDemoFocusObject(objectId);
  const dialogProposal = buildDialogProposalForFocus(focus);
  return {
    event_id: DEMO_EVENT_ID,
    object_id: focus.object_id,
    object_name: focus.object_name,
    message,
    answer: buildDemoDialogAnswer(message, focus),
    impact_summary: focus.risk_reasons,
    evidence: focus.evidence,
    recommended_actions: focus.recommended_actions,
    risk_reminders: focus.risk_reminders,
    follow_up_prompts: [
      "请解释这个处置动作为什么必须由指挥员审批，而不能由智能体直接执行。",
      "请把当前影响链拆成风险源、传导道路、影响对象、重点人群和处置资源。",
      "审批通过后，智能体应该分别生成哪些面向领导、部门、社区和公众的预警版本？",
    ],
    grounding_summary: "演示模式结构化回答，固定返回影响链、证据、建议动作和审批入口。",
    proposal_entry: {
      blocked: false,
      proposal: buildDemoRegionalView(dialogProposal),
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
