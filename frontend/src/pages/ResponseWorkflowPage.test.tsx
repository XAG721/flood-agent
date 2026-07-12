import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { responseWorkflowApi } from "../api/responseWorkflowApi";
import type { EventDashboard } from "../types/response";
import { ResponseWorkflowPage } from "./ResponseWorkflowPage";

vi.mock("../api/responseWorkflowApi", () => ({
  responseWorkflowApi: {
    listEvents: vi.fn(),
    listDocuments: vi.fn(),
    registerDocument: vi.fn(),
    getDashboard: vi.fn(),
    bootstrapDemo: vi.fn(),
    discoverRiskObjects: vi.fn(),
    generateTaskDraft: vi.fn(),
    resolveEvidenceConflict: vi.fn(),
    supplementEvidence: vi.fn(),
    freezeEvidencePackage: vi.fn(),
    decideTask: vi.fn(),
    acknowledgeTask: vi.fn(),
    startTask: vi.fn(),
    completeTask: vi.fn(),
    reportResourceShortage: vi.fn(),
    reportPartialCompletion: vi.fn(),
    requestDeadlineExtension: vi.fn(),
    cancelTask: vi.fn(),
    takeOverTask: vi.fn(),
    verifyTask: vi.fn(),
    runDeadlineSweep: vi.fn(),
    closeEvent: vi.fn(),
    runScenarioEvaluation: vi.fn(),
  },
}));

const dashboard: EventDashboard = {
  event: {
    event_id: "FLOOD-TEST-001",
    title: "碑林区下穿通道响应事件",
    area_id: "beilin_10km2",
    status: "active",
    current_alert_version: 1,
    created_at: "2026-07-11T08:00:00Z",
    updated_at: "2026-07-11T08:00:00Z",
    workflow_engine_version: "response-workflow-v2",
    data_version: "response-schema-v1",
    source_type: "simulation",
    is_simulated: true,
  },
  alert_snapshots: [
    {
      snapshot_id: "ALT-1",
      event_id: "FLOOD-TEST-001",
      alert_id: "ALERT-1",
      source_department: "西安市气象部门",
      disaster_type: "暴雨",
      level: "橙色",
      issued_at: "2026-07-11T08:00:00Z",
      affected_area: "碑林区",
      raw_content: "强降雨预警",
      version: 1,
      created_at: "2026-07-11T08:00:00Z",
      raw_payload_hash: "alert-hash",
      data_version: "warning-v1",
      source_type: "simulation",
      source_version: "v1",
      is_simulated: true,
      lifecycle_status: "active",
    },
  ],
  risk_objects: [
    {
      event_id: "FLOOD-TEST-001",
      object_id: "TUNNEL-017",
      name: "长安北路下穿通道",
      object_type: "下穿通道",
      location: "长安北路",
      responsible_organization: "区住建局",
      responsible_role: "排水值班负责人",
      trigger_reasons: ["橙色预警覆盖"],
      source_refs: ["ALERT-1"],
      vulnerability: "低洼",
      historical_risk: "历史积水",
      risk_score: 87,
      system_explanation: "待核验候选",
      verification_status: "confirmed",
      verification_note: "已核验",
      source_type: "simulation",
      source_version: "registry-v1",
      data_version: "risk-object-v1",
      is_simulated: true,
      missing_fields: [],
      stale: false,
      version: 1,
      calibration_version: "candidate-logistic-v1",
      association_mode: "area_registry",
    },
  ],
  tasks: [
    {
      task_id: "TASK-1",
      event_id: "FLOOD-TEST-001",
      object_id: "TUNNEL-017",
      version: 1,
      title: "现场核查与响应处置",
      action: "核查积水并按预案组织交通管控。",
      responsible_organization: "区住建局",
      responsible_role: "排水值班负责人",
      cooperate_roles: ["交警联络员"],
      status: "draft",
      deadline_at: "2026-07-11T10:00:00Z",
      acknowledge_deadline_at: "2026-07-11T08:15:00Z",
      required_evidence: ["现场照片"],
      plan_basis: [{ document: "碑林区城市内涝预警与响应指引", version: "2026 演示有效版", clause: "3.4" }],
      approval_policy: "commander_required",
      escalation_rule: "超时升级",
      dependencies: [],
      generated_by_ai: true,
      generation_version: "response-rag-task-v1",
      drafted_by: "duty-1",
      assignment_version: 0,
      grounding_summary: "证据闸门通过：6/6 角色完整。",
      evidence_role_coverage: {
        condition: true,
        object: true,
        responsibility: true,
        procedure: true,
        exception: true,
        attribution: true,
      },
      source_evidence: [
        {
          source_type: "plan_document",
          source_id: "policy-1",
          title: "碑林区城市内涝预警与响应指引",
          excerpt: "橙色预警阶段应优先核查下穿通道。",
          roles: ["condition", "procedure", "attribution"],
          document_version: "2026 演示有效版",
          clause: "3.4",
        },
      ],
      validation_warnings: [],
      rule_set_version: "response-rules-v2",
      data_version: "response-schema-v1",
      is_simulated: true,
    },
  ],
  feedback: [],
  assignments: [],
  escalations: [],
  timeline: [],
  candidate_runs: [],
  evidence_packages: [],
  outbox: [],
  rule_evaluations: [],
  deadline_extensions: [],
  risk_object_versions: [],
  metrics: {
    alert_versions: 1,
    candidate_objects: 1,
    confirmed_objects: 1,
    task_count: 1,
    completed_tasks: 0,
    escalated_tasks: 0,
    timeline_entries: 0,
    completion_rate: 0,
  },
};

describe("ResponseWorkflowPage", () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(responseWorkflowApi.listEvents).mockResolvedValue([dashboard.event]);
    vi.mocked(responseWorkflowApi.listDocuments).mockResolvedValue([]);
    vi.mocked(responseWorkflowApi.getDashboard).mockResolvedValue(dashboard);
  });

  it("展示六类证据覆盖和带版本条款的来源", async () => {
    render(<ResponseWorkflowPage />);

    expect(await screen.findByRole("heading", { name: "碑林区下穿通道响应事件" })).toBeInTheDocument();
    expect(screen.getByText("6/6")).toBeInTheDocument();
    expect(screen.getByText("已覆盖 · 触发条件")).toBeInTheDocument();
    expect(screen.getByText("已覆盖 · 版本归因")).toBeInTheDocument();
    expect(screen.getByText("查看 1 条来源证据")).toBeInTheDocument();
  });

  it("明确区域台账筛查边界并刷新候选对象", async () => {
    vi.mocked(responseWorkflowApi.discoverRiskObjects).mockResolvedValue({});
    render(<ResponseWorkflowPage />);

    expect(await screen.findByText(/预警未提供空间多边形/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "筛查预警范围" }));

    await waitFor(() => {
      expect(responseWorkflowApi.discoverRiskObjects).toHaveBeenCalledWith("FLOOD-TEST-001", "commander");
      expect(responseWorkflowApi.getDashboard).toHaveBeenCalledTimes(2);
    });
  });

  it("有预警多边形时标明采用 EPSG:4326 点落区筛查", async () => {
    const spatialDashboard = structuredClone(dashboard);
    spatialDashboard.alert_snapshots[0].affected_geometry = {
      crs: "EPSG:4326",
      coordinates: [
        [108.9565, 34.2425],
        [108.9595, 34.2425],
        [108.9595, 34.2455],
        [108.9565, 34.2425],
      ],
    };
    vi.mocked(responseWorkflowApi.getDashboard).mockResolvedValue(spatialDashboard);

    render(<ResponseWorkflowPage />);

    expect(await screen.findByText(/按 EPSG:4326 预警多边形与对象坐标做点落区筛查/)).toBeInTheDocument();
    expect(screen.getByText(/结果仍须防办人工核验/)).toBeInTheDocument();
  });

  it("在任务内展示异常备选动作，并在关闭后展示复盘草稿", async () => {
    const upgraded = structuredClone(dashboard);
    upgraded.event.status = "closed";
    upgraded.tasks[0].status = "escalated";
    upgraded.escalations = [
      {
        escalation_id: "ESC-1",
        event_id: upgraded.event.event_id,
        task_id: upgraded.tasks[0].task_id,
        reason: "移动排水设备不足",
        previous_status: "in_progress",
        target_role: "duty_officer",
        recommended_actions: ["向相邻成员单位申请补充资源。"],
        resolved: false,
        created_at: "2026-07-11T09:00:00Z",
      },
    ];
    upgraded.review_draft = {
      review_id: "REVIEW-1",
      event_id: upgraded.event.event_id,
      status: "draft",
      headline: "碑林区下穿通道响应事件响应过程复盘草稿",
      executive_summary: "事件形成 1 项任务，发生 1 次升级。",
      process_metrics: { task_count: 1, escalation_count: 1 },
      key_decisions: [],
      delays_and_escalations: ["任务 TASK-1：移动排水设备不足"],
      evidence_findings: ["1 条反馈通过证据校验。"],
      unresolved_issues: [],
      improvement_recommendations: ["更新相邻单位支援清单。"],
      source_timeline_entry_ids: ["LOG-1"],
      generated_by: "commander-1",
      generation_source: "system",
      created_at: "2026-07-11T10:00:00Z",
    };
    upgraded.scenario_report = {
      report_id: "SCENARIO-1",
      event_id: upgraded.event.event_id,
      scenario_name: "碑林区下穿通道洪水响应仿真场景",
      scenario_type: "simulation",
      scope: {
        area_id: "beilin_10km2",
        alert_versions: 1,
        risk_object_instances: 1,
        risk_object_types: ["下穿通道"],
        task_count: 1,
        operator_roles: ["duty_officer", "commander", "liaison", "field_operator"],
      },
      baseline_note: "人工基线为可配置参考假设，不代表真实对照组实测。",
      metrics: [
        {
          metric_id: "approval_duration",
          category: "审批执行",
          label: "审批耗时",
          unit: "分钟",
          manual_value: 20,
          system_value: 1.25,
          status: "measured",
          interpretation: "相对人工参考基线节省 18.75 分钟。",
          evidence_refs: ["LOG-1", "LOG-2"],
        },
        {
          metric_id: "object_omission_rate",
          category: "对象研判",
          label: "对象遗漏率",
          unit: "%",
          system_value: null,
          status: "not_evaluable",
          interpretation: "当前场景没有足够的真值或异常样本，未计算该指标。",
          evidence_refs: [],
        },
      ],
      acceptance_checks: Array.from({ length: 10 }, (_, index) => ({
        check_id: `AC-${String(index + 1).padStart(2, "0")}`,
        requirement: index === 9 ? "至少一个最小区县场景完成端到端测试" : `验收要求 ${index + 1}`,
        passed: true,
        detail: "台账证据完整",
        evidence_refs: ["LOG-1"],
      })),
      failure_cases: [
        {
          failure_id: "FAIL-1",
          stage: "异常处理",
          severity: "handled",
          trigger: "移动排水设备不足",
          observed: "任务触发升级并形成备选动作。",
          expected: "异常应被发现并恢复。",
          recommendation: "复盘升级耗时。",
          reproducible: true,
        },
      ],
      observations: ["对象遗漏率缺少人工标注真值。"],
      overall_status: "passed_with_observations",
      generated_by: "auditor-1",
      created_at: "2026-07-11T10:05:00Z",
    };
    vi.mocked(responseWorkflowApi.getDashboard).mockResolvedValue(upgraded);

    render(<ResponseWorkflowPage />);

    expect(await screen.findByText("异常已升级")).toBeInTheDocument();
    expect(screen.getByText("向相邻成员单位申请补充资源。")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: upgraded.review_draft.headline })).toBeInTheDocument();
    expect(screen.getByText("更新相邻单位支援清单。")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "碑林区下穿通道洪水响应仿真场景" })).toBeInTheDocument();
    expect(screen.getByText(/10\/10 项验收/)).toBeInTheDocument();
    expect(screen.getByText("相对人工参考基线节省 18.75 分钟。")).toBeInTheDocument();
    expect(screen.getByText("数据不足")).toBeInTheDocument();
    expect(screen.getByText("对象遗漏率缺少人工标注真值。")).toBeInTheDocument();
  });
});
