import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type {
  ActionProposalV2,
  RegionalAnalysisPackageView,
  RegionalProposalQueueSnapshot,
  RegionalProposalView,
  RiskLevel,
} from "./types/api";

vi.mock("framer-motion", () => {
  const createMotion = (tag: string) =>
    React.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(({ children, ...props }, ref) =>
      React.createElement(tag, { ...props, ref }, children),
    );

  return {
    AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    motion: new Proxy(
      {},
      {
        get: (_target, property: string) => createMotion(property),
      },
    ),
  };
});

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  url: string;

  readyState = 1;

  onmessage: ((event: MessageEvent) => void) | null = null;

  onerror: (() => void) | null = null;

  constructor(url: string | URL) {
    this.url = String(url);
    FakeEventSource.instances.push(this);
  }

  close() {
    this.readyState = 2;
  }

  static reset() {
    FakeEventSource.instances = [];
  }

  static emit(snapshot: unknown) {
    const payload = snapshot as { queue_version?: string; event_type?: string };
    const instance = [...FakeEventSource.instances]
      .reverse()
      .find((candidate) => {
        if (payload.queue_version) {
          return candidate.url.includes("/api/v2/proposals/stream");
        }
        if (payload.event_type) {
          return candidate.url.includes("/api/v3/events/");
        }
        return true;
      });
    if (!instance?.onmessage) {
      throw new Error("No active EventSource listener.");
    }
    instance.onmessage({ data: JSON.stringify(snapshot) } as MessageEvent);
  }
}

const eventPayload = {
  event_id: "event_demo",
  area_id: "beilin_10km2",
  title: "碑林区积涝演练事件",
  trigger_reason: "frontend_seed",
  current_stage: "Monitoring",
  current_risk_level: "Orange",
  status: "active",
  metadata: {},
  created_at: "2026-04-01T12:00:00Z",
  updated_at: "2026-04-01T12:00:00Z",
};

const sharedMemoryPayload = {
  event_id: "event_demo",
  autonomy_level: "human_gate_required",
  active_agents: ["hazard_agent", "exposure_agent", "planning_agent", "policy_agent"],
  focus_entity_ids: ["school_wyl_primary"],
  focus_entity_names: ["五岳里小学"],
  top_risks: ["预计 10 分钟内学校门前道路将出现明显积水。"],
  recommended_actions: ["提前生成区域通知", "准备转移建议"],
  pending_proposal_ids: [],
  recent_result_ids: ["result_policy_1"],
  unresolved_items: ["需要补充校门口离校人数。"],
  active_decision_path: [
    "hazard_agent：确认当前为橙色风险。",
    "exposure_agent：识别出学校和社区为高风险对象。",
    "policy_agent：要求先进入人工确认。",
  ],
  open_questions: ["需要补充校门口离校人数。"],
  blocked_by: ["区域级请示尚未确认。"],
  latest_hazard_level: "Orange",
  latest_summary: "区域级动作需要指挥长做最终放行。",
  last_trigger: "simulation_updated",
  updated_at: "2026-04-01T12:06:30Z",
};

const agentStatusPayload = {
  event_id: "event_demo",
  active_agents: sharedMemoryPayload.active_agents,
  autonomy_level: sharedMemoryPayload.autonomy_level,
  latest_hazard_level: "Orange",
  pending_task_count: 0,
  running_task_count: 0,
  completed_task_count: 5,
  superseded_task_count: 1,
  active_decision_path: sharedMemoryPayload.active_decision_path,
  open_questions: sharedMemoryPayload.open_questions,
  blocked_by: sharedMemoryPayload.blocked_by,
  latest_summary: sharedMemoryPayload.latest_summary,
  updated_at: sharedMemoryPayload.updated_at,
};

const decisionReportPayload = {
  event_id: "event_demo",
  latest_summary: sharedMemoryPayload.latest_summary,
  active_decision_path: sharedMemoryPayload.active_decision_path,
  blocked_by: sharedMemoryPayload.blocked_by,
  open_questions: sharedMemoryPayload.open_questions,
  recent_result_ids: sharedMemoryPayload.recent_result_ids,
};

const sessionPayload = {
  session_id: "v2_session_demo",
  event: eventPayload,
  messages: [
    {
      message_id: "msg_welcome",
      role: "assistant",
      content: "已连接当前事件，可直接发起基于证据的影响评估问答。",
      created_at: "2026-04-01T12:05:00Z",
      structured_answer: null,
    },
  ],
  latest_answer: null,
  proposals: [],
  notification_drafts: [],
  execution_logs: [],
  shared_memory_snapshot: sharedMemoryPayload,
  active_agents: sharedMemoryPayload.active_agents,
  recent_agent_results: [],
  autonomy_level: sharedMemoryPayload.autonomy_level,
};

const replyStructuredAnswer = {
  answer: "建议在后巷积水成段前，优先协助李阿姨转移。",
  evidence: [
    {
      evidence_type: "realtime",
      title: "李阿姨所在网格实时积涝栅格",
      source_id: "resident_elderly_ls1_tile",
      excerpt: "积水趋势仍在上升。",
      timestamp: "2026-04-01T12:06:00Z",
      priority: 0,
    },
  ],
  impact_summary: ["预计 18 分钟内将影响李阿姨所在片区。"],
  recommended_actions: ["提前准备协助转移。", "优先使用北侧通道。"],
  follow_up_prompts: [
    "请继续拆解这次处置的执行步骤和确认节点。",
    "请说明如果延后处理，未来 30 分钟最可能先恶化的环节是什么。",
  ],
  confidence: 0.82,
  requires_human_confirmation: false,
  missing_data: [],
  proposal: null,
  planner_summary: "规划解释：对象影响评估。",
  tool_selection_reasoning: ["优先获取实时风险与对象画像。"],
  skipped_tools: [],
  tool_executions: [
    {
      tool_name: "get_hazard_tiles",
      status: "success",
      input: { event_id: "event_demo" },
      output_summary: "已加载 1 个风险栅格。",
      duration_ms: 18,
      timed_out: false,
      data_freshness_seconds: 90,
    },
  ],
  data_freshness: {
    hazard_state_freshness_seconds: 90,
    traffic_freshness_seconds: null,
    profile_freshness_label: "运行期画像库",
    rag_document_recency_summary: "当前回答未引用时效性文档。",
  },
  evidence_gaps: [],
  tool_trace: [{ tool_name: "get_hazard_tiles", summary: "已加载 1 个风险栅格。" }],
};

const replyPayload = {
  ...sessionPayload,
  messages: [
    ...sessionPayload.messages,
    {
      message_id: "msg_user",
      role: "user",
      content: "这对低洼区老人意味着什么？",
      created_at: "2026-04-01T12:06:00Z",
      structured_answer: null,
    },
    {
      message_id: "msg_answer",
      role: "assistant",
      content: replyStructuredAnswer.answer,
      created_at: "2026-04-01T12:06:03Z",
      structured_answer: replyStructuredAnswer,
    },
  ],
  latest_answer: replyStructuredAnswer,
};

const hazardPayload = {
  event_id: "event_demo",
  area_id: "beilin_10km2",
  generated_at: "2026-04-01T12:05:00Z",
  overall_risk_level: "Orange",
  overall_score: 73,
  trend: "rapidly_rising",
  uncertainty: 0.28,
  freshness_seconds: 90,
  hazard_tiles: [],
  road_reachability: [
    {
      road_id: "road_1",
      name: "东线转移通道",
      from_village: "联市街片区",
      to_location: "体育学院避难点",
      accessible: true,
      travel_time_minutes: 14,
      depth_limit_cm: 28,
      failure_reason: "",
    },
  ],
  monitoring_points: [],
};

function buildImpact(entityId: string, name: string, entityType: string, village: string) {
  return {
    event_id: "event_demo",
    entity: {
      entity_id: entityId,
      area_id: "beilin_10km2",
      entity_type: entityType,
      name,
      village,
      location_hint: `${village} 主路口`,
      resident_count: 12,
      current_occupancy: 12,
      vulnerability_tags: ["elderly"],
      mobility_constraints: ["needs_assistance"],
      key_assets: [],
      inventory_summary: "",
      continuity_requirement: "",
      preferred_transport_mode: "assisted",
      notification_preferences: ["sms"],
      emergency_contacts: [{ name: "值守负责人", phone: "13800000000", role: "lead" }],
      custom_attributes: {},
    },
    risk_level: entityId === "resident_elderly_ls1" ? "Orange" : "Yellow",
    time_to_impact_minutes: entityId === "resident_elderly_ls1" ? 18 : 28,
    risk_reason: [`预计 ${name} 将在 18 分钟内受到影响。`],
    safe_routes: [],
    blocked_routes: [],
    nearest_shelters: ["体育学院避难点"],
    resource_gap: [],
    evidence: [
      {
        evidence_type: "realtime",
        title: `${name} 风险栅格`,
        source_id: `${entityId}_tile`,
        excerpt: "实时积涝栅格",
        timestamp: "2026-04-01T12:05:00Z",
        priority: 0,
      },
    ],
    recent_warning_drafts: [
      {
        warning_id: "warning_public_1",
        proposal_id: "regional_dispatch_approved",
        event_id: "event_demo",
        audience: "public",
        title: "北部片区积涝提醒",
        summary: "请沿北侧通道有序避让，避免进入低洼路段。",
        content: "北部片区未来 30 分钟积涝风险继续上升，请公众远离下穿通道和低洼道路。",
        tone: "warning",
        channels: ["sms", "broadcast"],
        evidence_summary: "基于当前 hazard tiles 与对象影响评估生成。",
        created_at: "2026-04-01T12:06:20Z",
      },
    ],
  };
}

const knownImpacts: Record<string, ReturnType<typeof buildImpact>> = {
  resident_elderly_ls1: buildImpact("resident_elderly_ls1", "李阿姨", "resident", "联市街片区"),
  school_wyl_primary: buildImpact("school_wyl_primary", "五岳里小学", "school", "五岳里片区"),
  factory_wyr_bio: buildImpact("factory_wyr_bio", "五岳里生物制剂厂", "factory", "五岳里片区"),
  nursing_home_hpm: buildImpact("nursing_home_hpm", "和平门颐养中心", "nursing_home", "和平门片区"),
  metro_nsm_hub: buildImpact("metro_nsm_hub", "南门地铁换乘枢纽", "metro_station", "南门片区"),
  community_jsl_grid: buildImpact("community_jsl_grid", "建设里社区网格三组", "community", "建设里片区"),
};

function createTwinOverviewPayload() {
  const focusIds = ["resident_elderly_ls1", "school_wyl_primary", "community_jsl_grid"];
  const focusObjects = focusIds.map((objectId, index) => {
    const impact = knownImpacts[objectId];
    return {
      object_id: objectId,
      name: impact.entity.name,
      entity_type: impact.entity.entity_type,
      village: impact.entity.village,
      risk_level: impact.risk_level,
      time_to_impact_minutes: impact.time_to_impact_minutes,
      summary: impact.risk_reason[0],
      recommended_action: index === 0 ? "优先协助转移并保持北侧通道畅通。" : "继续跟踪并准备联动处置。",
      pending_proposal_ids: [],
      canvas_position: {
        left: 24 + index * 22,
        top: 30 + index * 18,
      },
    };
  });
  const mapLayers = focusIds.map((objectId, index) => {
    const impact = knownImpacts[objectId];
    return {
      object_id: objectId,
      name: impact.entity.name,
      risk_level: impact.risk_level,
      is_lead: index === 0,
      east_offset_m: -240 + index * 180,
      north_offset_m: 150 - index * 120,
      height_offset_m: 18 + index * 4,
      status_label: index === 0 ? "priority_focus" : "tracking",
    };
  });

  return {
    event_id: "event_demo",
    area_id: "beilin_10km2",
    event_title: eventPayload.title,
    generated_at: "2026-04-01T12:06:20Z",
    overall_risk_level: "Orange",
    trend: "rapidly_rising",
    summary: "主屏已汇聚重点对象、影响链与待审批动作，可直接进入智能体追问与审批闭环。",
    lead_object_id: focusObjects[0].object_id,
    lead_object_name: focusObjects[0].name,
    focus_objects: focusObjects,
    pending_proposal_count: 0,
    approved_proposal_count: 1,
    warning_draft_count: 0,
    active_alert_count: 2,
    map_layers: mapLayers,
    recommended_actions: ["优先追问首要影响对象。", "准备区域联动通知。"],
    signals: [
      {
        signal_id: "signal_water_1",
        title: "上游水位继续上涨",
        detail: "未来 30 分钟内北部片区积涝风险仍在抬升。",
        severity: "warning",
        created_at: "2026-04-01T12:05:30Z",
      },
      {
        signal_id: "signal_route_1",
        title: "北侧通道可达",
        detail: "当前仍可作为优先转移通道，但需保持持续监测。",
        severity: "info",
        created_at: "2026-04-01T12:05:50Z",
      },
    ],
  };
}

function createAgentCouncilPayload() {
  return {
    event_id: "event_demo",
    generated_at: "2026-04-01T12:06:20Z",
    roles: [
      {
        role_id: "impact_agent",
        role_name: "Impact Agent",
        status: "ready",
        summary: "识别北部片区学校、社区与独居老人是当前最优先关注对象。",
        evidence: ["实时风险栅格抬升", "对象脆弱性标签命中"],
        recommended_actions: ["优先追问学校与社区联动动作"],
        open_questions: ["学校离校人数是否已确认"],
      },
      {
        role_id: "action_agent",
        role_name: "Action Agent",
        status: "ready",
        summary: "建议先锁定北侧转移通道并准备区域提醒。",
        evidence: ["北侧通道当前可达", "现有抽排资源仍可调度"],
        recommended_actions: ["生成区域通知 proposal"],
        open_questions: [],
      },
      {
        role_id: "warning_agent",
        role_name: "Warning Agent",
        status: "ready",
        summary: "已准备公众版和部门版预警草稿模版。",
        evidence: ["通知模板可复用", "SOP 约束已满足"],
        recommended_actions: ["审批后直接生成 warning drafts"],
        open_questions: [],
      },
      {
        role_id: "audit_agent",
        role_name: "Audit Agent",
        status: "ready",
        summary: "当前建议进入人工审批，不建议自动执行。",
        evidence: ["区域请示链仍需指挥长确认"],
        recommended_actions: ["保留 human gate"],
        open_questions: ["请确认最终批准角色"],
      },
    ],
    audit_decision: {
      status: "approved_for_review",
      rationale: "证据充分，但涉及区域通知，仍需人工放行。",
      blocking_reasons: [],
      required_actions: ["等待指挥长审批 proposal"],
    },
    proposal_ids: ["regional_notification_1"],
    warning_ids: ["warning_public_1"],
  };
}

function createFocusObjectPayload(objectId: string) {
  const impact = knownImpacts[objectId] ?? knownImpacts.resident_elderly_ls1;
  return {
    event_id: "event_demo",
    object_id: impact.entity.entity_id,
    object_name: impact.entity.name,
    entity_type: impact.entity.entity_type,
    village: impact.entity.village,
    risk_level: impact.risk_level,
    time_to_impact_minutes: impact.time_to_impact_minutes,
    summary: impact.risk_reason[0],
    risk_reasons: impact.risk_reason,
    recommended_actions: ["优先联动现场值守人员。", "检查转移通道与到点时间。"],
    risk_reminders: ["当前趋势仍在上升。", "处置延后会增加对象暴露时间。"],
    evidence: impact.evidence,
    related_proposals: [approvedHistoryItem],
  };
}

function createRegionalProposal(overrides?: Partial<ActionProposalV2>): ActionProposalV2 {
  return {
    proposal_id: "regional_notification_1",
    event_id: "event_demo",
    area_id: "beilin_10km2",
    entity_id: null,
    title: "发布区域积涝提醒",
    summary: "建议对北部社区先行发布积涝通知。",
    severity: "Orange",
    requires_confirmation: true,
    required_operator_roles: ["commander"],
    payload: {},
    source_session_id: null,
    status: "pending",
    resolved_at: null,
    resolved_by: null,
    resolution_note: "",
    created_at: "2026-04-01T12:06:40Z",
    proposal_scope: "regional",
    action_type: "regional_notification",
    action_display_name: "发布区域积涝提醒",
    action_display_tagline: "面向重点区域快速组织预警触达与行动提醒。",
    action_display_category: "态势通知",
    trigger_reason: "区域综合风险进入 Orange，建议立即通知北部社区。",
    recommendation: "先向北部社区和学校家长发布积涝提醒。",
    evidence_summary: "水深与流速耦合结果显示北部片区 30 分钟内风险快速抬升。",
    high_risk_object_ids: ["school_wyl_primary", "community_jsl_grid"],
    action_scope: {
      target_scope: "北部社区与周边学校",
      channel: "短信 + 广播",
    },
    risk_stage_key: "risk_stage_orange_1",
    system_version_hash: "hash_notification_v1",
    updated_at: "2026-04-01T12:06:40Z",
    edited_by_commander: false,
    last_editor: "system",
    has_new_system_suggestion: false,
    superseded_by_proposal_id: null,
    withdrawn_reason: undefined,
    ...overrides,
  };
}

function createRegionalView(overrides?: {
  event_title?: string;
  current_risk_level?: RiskLevel;
  proposal?: Partial<ActionProposalV2>;
  high_risk_object_names?: string[];
}): RegionalProposalView {
  return {
    proposal: createRegionalProposal(overrides?.proposal),
    event_title: overrides?.event_title ?? "碑林区积涝演练事件",
    current_risk_level: overrides?.current_risk_level ?? "Orange",
    high_risk_object_names: overrides?.high_risk_object_names ?? ["五岳里小学", "建设里社区网格三组"],
  };
}

const approvedHistoryItem = createRegionalView({
  current_risk_level: "Red",
  proposal: {
    proposal_id: "regional_dispatch_approved",
    title: "完成区域资源调度",
    action_display_name: "完成区域资源调度",
    action_display_tagline: "继续维持北部片区抽排与通行保障。",
    action_display_category: "资源调度",
    summary: "已向北部片区派出抽排与交通引导资源。",
    status: "approved",
    action_type: "regional_resource_dispatch",
    recommendation: "继续维持北部片区抽排与交通引导。",
    evidence_summary: "上一轮调度已完成闭环。",
    action_scope: {
      priority_zone: "北部片区",
      resource_count: 4,
    },
    updated_at: "2026-04-01T12:05:00Z",
  },
});

function createQueueSnapshot(items: RegionalProposalView[], version = "queue_v1"): RegionalProposalQueueSnapshot {
  return {
    queue_version: version,
    generated_at: "2026-04-01T12:06:40Z",
    items,
  };
}

function createRegionalAnalysisPackage(
  overrides?: Partial<RegionalAnalysisPackageView>,
): RegionalAnalysisPackageView {
  return {
    package_id: "risk_stage_orange_1",
    event_id: "event_demo",
    current_risk_level: "Orange",
    trigger_type: "simulation_updated",
    focus_object_ids: ["school_wyl_primary", "community_jsl_grid"],
    focus_object_names: ["School", "Community"],
    proposal_ids: ["regional_notification_1"],
    proposal_titles: ["Regional package action"],
    proposal_count: 1,
    analysis_message: "A new regional analysis package is ready.",
    risk_assessment: "The district remains at elevated flood risk.",
    rescue_plan: "Coordinate rescue preparation around the current focus objects.",
    resource_dispatch_plan: "Pre-position pumps and warning capacity near the focus objects.",
    status: "pending",
    created_at: "2026-04-01T12:06:40Z",
    updated_at: "2026-04-01T12:06:40Z",
    ...overrides,
  };
}

function installFetchMock(options?: {
  initialQueueItems?: ReturnType<typeof createRegionalView>[];
  initialHistoryItems?: ReturnType<typeof createRegionalView>[];
}) {
  let queueVersion = 1;
  let regionalQueueItems = [...(options?.initialQueueItems ?? [])];
  let regionalHistoryItems = [...(options?.initialHistoryItems ?? [approvedHistoryItem])];
  let pendingRegionalAnalysisPackage =
    regionalQueueItems.length > 0
      ? createRegionalAnalysisPackage({
          proposal_ids: regionalQueueItems.map((item) => item.proposal.proposal_id),
          proposal_titles: regionalQueueItems.map((item) => item.proposal.title),
          proposal_count: regionalQueueItems.length,
        })
      : null;
  let regionalAnalysisPackageHistory: RegionalAnalysisPackageView[] = [];

  const buildQueueSnapshot = () => createQueueSnapshot(regionalQueueItems, `queue_v${queueVersion}`);

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const requestUrl = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    const url = new URL(requestUrl, "http://localhost");
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;

    if (url.pathname === "/api/health" && method === "GET") {
      return jsonResponse({ status: "ok" });
    }

    if (url.pathname === "/api/v2/security/capabilities" && method === "GET") {
      return jsonResponse({
        operator_role: "commander",
        role_rank: 3,
        capabilities: {
          proposal_resolve: true,
          proposal_draft_edit: true,
          runtime_admin_write: true,
          dataset_manage: true,
          supervisor_control: true,
          agent_replay: true,
          evaluation_run: true,
          archive_run: true,
        },
        action_labels: {
          event_create: "创建事件",
          event_ingest: "导入监测数据",
          simulation_ingest: "导入模拟结果",
          proposal_resolve: "处理区域请示",
          proposal_draft_edit: "编辑请示草稿",
          runtime_admin_write: "修改运行期数据",
          dataset_manage: "管理数据集任务",
          supervisor_control: "控制后台巡检",
          agent_replay: "重放智能体任务",
          archive_run: "执行归档清理",
          evaluation_run: "运行评测任务",
        },
      });
    }

    if (url.pathname === "/api/v2/events" && method === "POST") {
      return jsonResponse(eventPayload);
    }

    if (url.pathname === "/api/v2/events/event_demo/observations" && method === "POST") {
      return jsonResponse({ event: eventPayload, hazard_state: hazardPayload });
    }

    if (url.pathname === "/api/v2/copilot/sessions/bootstrap" && method === "POST") {
      return jsonResponse(sessionPayload);
    }

    if (url.pathname === "/api/v2/copilot/sessions/v2_session_demo" && method === "GET") {
      return jsonResponse(sessionPayload);
    }

    if (url.pathname === "/api/v2/copilot/sessions/v2_session_demo/messages" && method === "POST") {
      return jsonResponse(replyPayload);
    }

    if (url.pathname === "/api/v2/events/event_demo/hazard-state" && method === "GET") {
      return jsonResponse(hazardPayload);
    }

    if (url.pathname === "/api/v3/events/event_demo/twin-overview" && method === "GET") {
      return jsonResponse(createTwinOverviewPayload());
    }

    if (url.pathname === "/api/v3/events/event_demo/agent-council" && method === "GET") {
      return jsonResponse(createAgentCouncilPayload());
    }

    if (url.pathname.match(/^\/api\/v3\/events\/event_demo\/objects\/[^/]+$/) && method === "GET") {
      const objectId = url.pathname.split("/")[6];
      return jsonResponse(createFocusObjectPayload(objectId));
    }

    if (url.pathname === "/api/v3/events/event_demo/dialog" && method === "POST") {
      const objectId = body?.object_id ?? "resident_elderly_ls1";
      const focusObject = createFocusObjectPayload(objectId);
      return jsonResponse({
        event_id: "event_demo",
        object_id: focusObject.object_id,
        object_name: focusObject.object_name,
        message: body?.message ?? "",
        answer: "建议先锁定高风险对象并确认最短转移路径。",
        impact_summary: focusObject.risk_reasons,
        evidence: focusObject.evidence,
        recommended_actions: focusObject.recommended_actions,
        risk_reminders: focusObject.risk_reminders,
        follow_up_prompts: ["请继续说明执行顺序。", "请说明需要人工确认的边界。"],
        grounding_summary: "基于对象画像、实时栅格和区域历史建议生成。",
        proposal_entry: null,
        response_source: "fixture",
        generated_at: "2026-04-01T12:06:40Z",
      });
    }

    if (url.pathname === "/api/v3/events/event_demo/proposals/generate" && method === "POST") {
      return jsonResponse({
        event_id: "event_demo",
        queue_version: "queue_v3_fixture",
        generated_at: "2026-04-01T12:06:40Z",
        blocked: false,
        block_reason: null,
        proposals: [],
      });
    }

    if (url.pathname.match(/^\/api\/v3\/proposals\/[^/]+\/warnings\/generate$/) && method === "POST") {
      const proposalId = url.pathname.split("/")[4];
      return jsonResponse({
        event_id: "event_demo",
        proposal_id: proposalId,
        generated_at: "2026-04-01T12:06:40Z",
        warnings: [],
      });
    }

    if (url.pathname.startsWith("/api/v2/entities/") && url.pathname.endsWith("/impact") && method === "GET") {
      const entityId = url.pathname.split("/")[4];
      return jsonResponse(
        knownImpacts[entityId] ?? buildImpact(entityId, entityId, "community", "默认片区"),
      );
    }

    if (url.pathname === "/api/v2/entity-profiles" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/areas/beilin_10km2/resources" && method === "GET") {
      return jsonResponse({
        scope: "area",
        area_id: "beilin_10km2",
        status: {
          shelters_available: 6,
          rescue_teams_available: 4,
          vehicles_available: 8,
          pumps_available: 5,
          notes: [],
        },
        updated_at: "2026-04-01T12:05:00Z",
      });
    }

    if (url.pathname === "/api/v2/events/event_demo/resources" && method === "GET") {
      return jsonResponse({
        scope: "event",
        event_id: "event_demo",
        area_id: "beilin_10km2",
        status: {
          shelters_available: 4,
          rescue_teams_available: 3,
          vehicles_available: 6,
          pumps_available: 4,
          notes: [],
        },
        updated_at: "2026-04-01T12:05:30Z",
      });
    }

    if (url.pathname === "/api/v2/rag/documents" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/admin/dataset/status" && method === "GET") {
      return jsonResponse({
        area_id: "beilin_10km2",
        raw_dir: "data_sources/beilin/raw",
        normalized_dir: "data_sources/beilin/normalized",
        bootstrap_dir: "flood_system/bootstrap_data",
        runtime_rag_path: "data/rag_documents.runtime.json",
        source_count: 5,
        cached_source_count: 5,
        failed_source_count: 0,
        cached_file_count: 7,
        raw_ready: true,
        raw_completeness_percent: 100,
        missing_required_sources: [],
        stale_sources: [],
        sources: [],
        raw_cache_health: [],
        latest_download_log: [],
        latest_fetch_summary: {
          artifact_count: 4,
          downloaded_artifact_count: 4,
          failed_artifact_count: 0,
          progress_percent: 100,
          latest_run_at: "2026-04-01T12:09:00Z",
        },
        latest_build_summary: {
          shelters_csv: "flood_system/bootstrap_data/shelters.csv",
          roads_csv: "flood_system/bootstrap_data/roads.csv",
        },
        latest_validation: {
          shelter_count: 6,
          road_count: 8,
          entity_profile_count: 0,
          rag_query_hit_count: 0,
        },
        normalized_files: [],
        bootstrap_files: [],
        active_job: null,
        recent_jobs: [],
      });
    }

    if (url.pathname === "/api/v2/events/event_demo/regional-proposals" && method === "GET") {
      return jsonResponse(regionalHistoryItems);
    }

    if (url.pathname === "/api/v2/events/event_demo/regional-analysis-packages/pending" && method === "GET") {
      return jsonResponse(pendingRegionalAnalysisPackage);
    }

    if (url.pathname === "/api/v2/events/event_demo/regional-analysis-packages" && method === "GET") {
      const includePending = url.searchParams.get("include_pending");
      if (includePending === "false") {
        return jsonResponse(regionalAnalysisPackageHistory);
      }
      return jsonResponse(
        pendingRegionalAnalysisPackage
          ? [pendingRegionalAnalysisPackage, ...regionalAnalysisPackageHistory]
          : regionalAnalysisPackageHistory,
      );
    }

    if (url.pathname === "/api/v2/proposals/pending" && method === "GET") {
      return jsonResponse(buildQueueSnapshot());
    }

    if (url.pathname === "/api/v2/events/event_demo/agent-status" && method === "GET") {
      return jsonResponse(agentStatusPayload);
    }

    if (url.pathname === "/api/v2/events/event_demo/agent-tasks" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/events/event_demo/shared-memory" && method === "GET") {
      return jsonResponse(sharedMemoryPayload);
    }

    if (url.pathname === "/api/v2/events/event_demo/supervisor-runs" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/supervisor/status" && method === "GET") {
      return jsonResponse({
        running: true,
        interval_seconds: 60,
        consecutive_failures: 0,
        retries_used_in_last_cycle: 0,
        skipped_sweeps: 0,
        circuit_state: "closed",
        last_started_at: "2026-04-01T12:05:58Z",
        last_success_at: "2026-04-01T12:06:00Z",
        last_failure_at: null,
        last_retry_at: null,
        last_completed_at: "2026-04-01T12:06:00Z",
        last_error: null,
        circuit_opened_at: null,
        circuit_expires_at: null,
        pending_trigger_count: 0,
        last_trigger_processed_at: "2026-04-01T12:05:20Z",
        recent_replay_count: 0,
        recent_timeline_failure_count: 0,
      });
    }

    if (url.pathname === "/api/v2/events/event_demo/trigger-events" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/events/event_demo/agent-timeline" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/copilot/sessions/v2_session_demo/memory" && method === "GET") {
      return jsonResponse({
        session_memory: {
          session_id: "v2_session_demo",
          memory_snapshot: {
            session_id: "v2_session_demo",
            focus_entity_id: "school_wyl_primary",
            focus_entity_name: "五岳里小学",
            focus_area_id: "beilin_10km2",
            current_goal: "entity_impact",
            pending_proposal_ids: [],
            executed_proposal_ids: [],
            unresolved_slots: [],
            last_completion_status: "completed",
            updated_at: "2026-04-01T12:06:10Z",
          },
          recent_events: [],
        },
        event_shared_memory: sharedMemoryPayload,
      });
    }

    if (url.pathname === "/api/v2/events/event_demo/experience-context" && method === "GET") {
      return jsonResponse({
        event_id: "event_demo",
        relevant_records: [],
        strategy_patterns: [],
        outcome_risk_notes: [],
      });
    }

    if (url.pathname === "/api/v2/events/event_demo/decision-report" && method === "GET") {
      return jsonResponse(decisionReportPayload);
    }

    if (url.pathname === "/api/v2/agent-metrics" && method === "GET") {
      return jsonResponse({
        generated_at: "2026-04-01T12:07:00Z",
        task_graph_latency_ms: 850,
        agent_failure_heatmap: {},
        stale_data_frequency: 0,
        auto_retry_success_rate: 1,
        superseded_task_ratio: 0.1,
        fanout_count: 4,
        stale_data_replan_count: 0,
      });
    }

    if (url.pathname === "/api/v2/evaluation/benchmarks" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/alerts" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/audit/records" && method === "GET") {
      return jsonResponse([]);
    }

    if (url.pathname === "/api/v2/archive/status" && method === "GET") {
      return jsonResponse({
        hot_retention_days: 14,
        archive_retention_days: 180,
        hot_record_count: 28,
        archived_record_count: 134,
        last_archive_run: {
          archive_run_id: "archive_1",
          status: "completed",
          hot_records_archived: 4,
          expired_archives_deleted: 0,
          tables_touched: ["v2_agent_tasks"],
          error_message: null,
          started_at: "2026-04-01T12:07:00Z",
          completed_at: "2026-04-01T12:07:02Z",
        },
        latest_archive_runs: [],
      });
    }

    if (url.pathname === "/api/v2/advisories/generate" && method === "POST") {
      return jsonResponse({
        advisory_id: "advisory_1",
        event_id: "event_demo",
        entity_id: body?.entity_id ?? "resident_elderly_ls1",
        answer: "建议在后巷积水成段前，优先协助李阿姨转移。",
        impact_summary: ["建议提前协助转移。"],
        recommended_actions: ["提前准备协助转移。"],
        route_options: [],
        evidence: knownImpacts.resident_elderly_ls1.evidence,
        confidence: 0.82,
        requires_human_confirmation: false,
        missing_data: [],
        proposal: null,
        generated_at: "2026-04-01T12:06:30Z",
      });
    }

    if (url.pathname.match(/^\/api\/v2\/proposals\/[^/]+\/draft$/) && method === "PATCH") {
      const proposalId = url.pathname.split("/")[4];
      regionalQueueItems = regionalQueueItems.map((item) =>
        item.proposal.proposal_id === proposalId
          ? {
              ...item,
              proposal: {
                ...item.proposal,
                action_scope: body?.action_scope ?? item.proposal.action_scope,
                edited_by_commander: true,
                last_editor: "commander",
                updated_at: "2026-04-01T12:07:10Z",
              },
            }
          : item,
      );
      if (pendingRegionalAnalysisPackage) {
        pendingRegionalAnalysisPackage = {
          ...pendingRegionalAnalysisPackage,
          proposal_ids: regionalQueueItems.map((item) => item.proposal.proposal_id),
          proposal_titles: regionalQueueItems.map((item) => item.proposal.title),
          proposal_count: regionalQueueItems.length,
          updated_at: "2026-04-01T12:07:10Z",
        };
      }
      queueVersion += 1;
      return jsonResponse(regionalQueueItems.find((item) => item.proposal.proposal_id === proposalId));
    }

    if (url.pathname.match(/^\/api\/v2\/proposals\/[^/]+\/approve$/) && method === "POST") {
      const proposalId = url.pathname.split("/")[4];
      const approvedItem = regionalQueueItems.find((item) => item.proposal.proposal_id === proposalId);
      if (!approvedItem) {
        return jsonResponse({ detail: "not found" }, 404);
      }
      const resolved: RegionalProposalView = {
        ...approvedItem,
        proposal: {
          ...approvedItem.proposal,
          status: "approved",
          resolved_at: "2026-04-01T12:07:30Z",
          resolved_by: "frontend_console",
          resolution_note: body?.note ?? "",
          updated_at: "2026-04-01T12:07:30Z",
        },
      };
      regionalQueueItems = regionalQueueItems.filter((item) => item.proposal.proposal_id !== proposalId);
      regionalHistoryItems = [resolved, ...regionalHistoryItems.filter((item) => item.proposal.proposal_id !== proposalId)];
      queueVersion += 1;
      return jsonResponse(resolved);
    }

    if (url.pathname.match(/^\/api\/v2\/proposals\/[^/]+\/reject$/) && method === "POST") {
      const proposalId = url.pathname.split("/")[4];
      const rejectedItem = regionalQueueItems.find((item) => item.proposal.proposal_id === proposalId);
      if (!rejectedItem) {
        return jsonResponse({ detail: "not found" }, 404);
      }
      const resolved: RegionalProposalView = {
        ...rejectedItem,
        proposal: {
          ...rejectedItem.proposal,
          status: "rejected",
          resolved_at: "2026-04-01T12:07:30Z",
          resolved_by: "frontend_console",
          resolution_note: body?.note ?? "",
          updated_at: "2026-04-01T12:07:30Z",
        },
      };
      regionalQueueItems = regionalQueueItems.filter((item) => item.proposal.proposal_id !== proposalId);
      regionalHistoryItems = [resolved, ...regionalHistoryItems.filter((item) => item.proposal.proposal_id !== proposalId)];
      queueVersion += 1;
      return jsonResponse(resolved);
    }

    if (url.pathname.match(/^\/api\/v2\/regional-analysis-packages\/[^/]+\/approve$/) && method === "POST") {
      const packageId = url.pathname.split("/")[4];
      if (!pendingRegionalAnalysisPackage || pendingRegionalAnalysisPackage.package_id !== packageId) {
        return jsonResponse({ detail: "not found" }, 404);
      }
      const resolvedPackage: RegionalAnalysisPackageView = {
        ...pendingRegionalAnalysisPackage,
        status: "approved",
        updated_at: "2026-04-01T12:07:30Z",
      };
      regionalHistoryItems = [
        ...regionalQueueItems.map<RegionalProposalView>((item) => ({
          ...item,
          proposal: {
            ...item.proposal,
            status: "approved",
            resolved_at: "2026-04-01T12:07:30Z",
            resolved_by: "frontend_console",
            resolution_note: body?.note ?? "",
            updated_at: "2026-04-01T12:07:30Z",
          },
        })),
        ...regionalHistoryItems,
      ];
      regionalQueueItems = [];
      pendingRegionalAnalysisPackage = null;
      regionalAnalysisPackageHistory = [resolvedPackage, ...regionalAnalysisPackageHistory];
      queueVersion += 1;
      return jsonResponse(resolvedPackage);
    }

    if (url.pathname.match(/^\/api\/v2\/regional-analysis-packages\/[^/]+\/reject$/) && method === "POST") {
      const packageId = url.pathname.split("/")[4];
      if (!pendingRegionalAnalysisPackage || pendingRegionalAnalysisPackage.package_id !== packageId) {
        return jsonResponse({ detail: "not found" }, 404);
      }
      const resolvedPackage: RegionalAnalysisPackageView = {
        ...pendingRegionalAnalysisPackage,
        status: "rejected",
        updated_at: "2026-04-01T12:07:30Z",
      };
      regionalHistoryItems = [
        ...regionalQueueItems.map<RegionalProposalView>((item) => ({
          ...item,
          proposal: {
            ...item.proposal,
            status: "rejected",
            resolved_at: "2026-04-01T12:07:30Z",
            resolved_by: "frontend_console",
            resolution_note: body?.note ?? "",
            updated_at: "2026-04-01T12:07:30Z",
          },
        })),
        ...regionalHistoryItems,
      ];
      regionalQueueItems = [];
      pendingRegionalAnalysisPackage = null;
      regionalAnalysisPackageHistory = [resolvedPackage, ...regionalAnalysisPackageHistory];
      queueVersion += 1;
      return jsonResponse(resolvedPackage);
    }

    throw new Error(`Unexpected request: ${method} ${url.pathname}${url.search}`);
  });

  vi.stubGlobal("fetch", fetchMock);

  return {
    fetchMock,
    getQueueSnapshot: () => buildQueueSnapshot(),
  };
}

describe("App", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    FakeEventSource.reset();
    vi.useRealTimers();
    vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function renderApp(route = "/") {
    return render(
      <MemoryRouter initialEntries={[route]}>
        <App />
      </MemoryRouter>,
    );
  }

  it("总览页会渲染中文导航和智能体状态流摘要", async () => {
    installFetchMock();
    renderApp("/");

    expect((await screen.findAllByRole("heading", { name: /数字孪生智能体洪水预警系统/ })).length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: "风险总览" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "影响问答" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "协同处置" })).toBeInTheDocument();
    expect(screen.getAllByText("碑林区积涝演练事件").length).toBeGreaterThan(0);
    expect(screen.getAllByText("李阿姨").length).toBeGreaterThan(0);
    expect(screen.getByText("数字孪生智能体洪水预警主屏")).toBeInTheDocument();
    expect(screen.getByText("重点对象")).toBeInTheDocument();
    expect(screen.getByText(/1 条.*已完成闭环/)).toBeInTheDocument();
  });

  it("智能问答页展示上下文输入区，且不会注入新的审批弹框", async () => {
    installFetchMock();
    renderApp("/copilot");

    expect(await screen.findByRole("heading", { name: /对话查看研判、请示与总结/ })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/先复述一下你现在理解的需求/)).toBeInTheDocument();
    expect(screen.getAllByText("李阿姨").length).toBeGreaterThan(0);
    expect(screen.getByText("发送")).toBeInTheDocument();
    expect(screen.queryByText("智能体主动请示")).not.toBeInTheDocument();
  });

  it("方案处置页展示当前事件的区域请示历史，而不是旧的对象审批队列", async () => {
    installFetchMock({
      initialHistoryItems: [
        approvedHistoryItem,
        createRegionalView({
          proposal: {
            proposal_id: "regional_evacuation_2",
            title: "生成区域转移建议",
            action_display_name: "规划区域转移建议",
            action_display_tagline: "最新一版建议优先覆盖老旧社区与学校。",
            action_display_category: "人员转运",
            status: "superseded",
            action_type: "regional_evacuation",
            recommendation: "最新一版建议优先覆盖老旧社区与学校。",
          },
        }),
      ],
    });
    renderApp("/operations");

    expect((await screen.findAllByRole("heading", { name: /协同处置/ })).length).toBeGreaterThan(0);
    expect(screen.getByText("完成区域资源调度")).toBeInTheDocument();
    expect(screen.getByText("规划区域转移建议")).toBeInTheDocument();
    expect(screen.getAllByText(/时间线/).length).toBeGreaterThan(0);
    expect(screen.queryByLabelText("approve-proposal-proposal_school_1")).not.toBeInTheDocument();
  });

  it("收到新的区域级 pending 队列后会弹出全局对话框，并支持保存草稿", async () => {
    const pendingItem = createRegionalView();
    const { fetchMock } = installFetchMock({ initialQueueItems: [pendingItem] });
    renderApp("/operations");

    expect((await screen.findAllByRole("heading", { name: /协同处置/ })).length).toBeGreaterThan(0);
    expect(await screen.findByText("智能体主动请示")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("target_scope"), {
      target: { value: "北部社区、五岳里小学与周边家属区" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "save-regional-proposal-draft-regional_notification_1" }),
    );

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/v2/proposals/regional_notification_1/draft"),
        expect.objectContaining({ method: "PATCH" }),
      );
    });
  });

  it("收到新的区域级 pending 队列后会弹出全局对话框，并支持确认执行", async () => {
    const { fetchMock, getQueueSnapshot } = installFetchMock({ initialQueueItems: [createRegionalView()] });
    renderApp("/operations");

    expect((await screen.findAllByRole("heading", { name: /协同处置/ })).length).toBeGreaterThan(0);

    fireEvent.click(
      await screen.findByRole("button", {
        name: "approve-regional-proposal-regional_notification_1",
      }),
    );

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/v2/proposals/regional_notification_1/approve"),
        expect.objectContaining({ method: "POST" }),
      );
    });

    await waitFor(() => {
      expect(getQueueSnapshot().items).toHaveLength(0);
    });
    await waitFor(() => {
      expect(screen.queryByLabelText("global-regional-proposal-dialog")).not.toBeInTheDocument();
    });
  });

  it("稍后处理会先关闭弹框，并在 5 分钟后重新弹出同一批待确认动作", async () => {
    installFetchMock({ initialQueueItems: [createRegionalView()] });
    renderApp("/operations");

    expect((await screen.findAllByRole("heading", { name: /协同处置/ })).length).toBeGreaterThan(0);

    expect(await screen.findByText("智能体主动请示")).toBeInTheDocument();
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "regional-proposal-snooze" }));
    expect(screen.queryByText("智能体主动请示")).not.toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5 * 60 * 1000);
    });

    expect(screen.getByText("智能体主动请示")).toBeInTheDocument();
    expect(screen.getAllByText("发布区域积涝提醒").length).toBeGreaterThan(0);
  });

  it("开放式区域动作优先展示模型生成的指挥动作名与副标题", async () => {
    installFetchMock({
      initialQueueItems: [
        createRegionalView({
          proposal: {
            proposal_id: "generic_action_1",
            action_type: "underground_space_clearance",
            action_display_name: "执行地下空间清退",
            action_display_tagline: "优先清空重点地下空间并锁定出入口。",
            action_display_category: "空间管控",
            title: "地下空间处置动作",
          },
        }),
      ],
    });
    renderApp("/operations");

    expect((await screen.findAllByRole("heading", { name: /协同处置/ })).length).toBeGreaterThan(0);

    const dialog = await screen.findByLabelText("global-regional-proposal-dialog");
    expect(dialog).toHaveTextContent("执行地下空间清退");
    expect(dialog).toHaveTextContent("优先清空重点地下空间并锁定出入口。");
    expect(dialog).toHaveTextContent("空间管控");
  });
});
