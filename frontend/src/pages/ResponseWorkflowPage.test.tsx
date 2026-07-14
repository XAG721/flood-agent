import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { responseWorkflowApi } from "../api/responseWorkflowApi";
import type { EvidencePackageVersion, EvidenceRole, EventDashboard } from "../types/response";
import { ResponseWorkflowPage } from "./ResponseWorkflowPage";

vi.mock("../api/responseWorkflowApi", () => ({
  responseWorkflowApi: {
    listEvents: vi.fn(),
    listDocuments: vi.fn(),
    listRiskObjectRegistry: vi.fn(),
    importRiskObjectFile: vi.fn(),
    registerDocument: vi.fn(),
    createDocumentVersion: vi.fn(),
    parseDocumentVersion: vi.fn(),
    publishDocumentVersion: vi.fn(),
    retireDocumentVersion: vi.fn(),
    rebuildDocumentIndex: vi.fn(),
    getDashboard: vi.fn(),
    bootstrapDemo: vi.fn(),
    discoverRiskObjects: vi.fn(),
    verifyRiskObject: vi.fn(),
    freezeCandidateObjectList: vi.fn(),
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
    processOutbox: vi.fn(),
    listDispatchCallbacks: vi.fn(),
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
          source_locator: "rag://policy/policy-1#clause=3.4",
          section_path: ["第三章", "3.4"],
          field_support: { trigger_condition: 0.9, action: 0.85 },
          conflicts_with: [],
        },
      ],
      validation_warnings: [],
      task_schema_version: "response-task-schema-v2",
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
  candidate_object_lists: [],
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
    vi.mocked(responseWorkflowApi.listRiskObjectRegistry).mockResolvedValue([]);
    vi.mocked(responseWorkflowApi.getDashboard).mockResolvedValue(dashboard);
    vi.mocked(responseWorkflowApi.listDispatchCallbacks).mockResolvedValue([]);
  });

  it("展示六类证据覆盖和带版本条款的来源", async () => {
    render(<ResponseWorkflowPage />);

    expect(await screen.findByRole("heading", { name: "碑林区下穿通道响应事件" })).toBeInTheDocument();
    expect(screen.getByText("6/6")).toBeInTheDocument();
    expect(screen.getByText("已覆盖 · 触发条件")).toBeInTheDocument();
    expect(screen.getByText("已覆盖 · 版本归因")).toBeInTheDocument();
    expect(screen.getByText("查看 1 条来源证据")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "风险对象主数据" })).toBeInTheDocument();
    expect(screen.getByText(/当前区域尚无带 EPSG:4326 坐标/)).toBeInTheDocument();
  });

  it("展示九字段证据、原文定位、NLI 降级状态和版本差异", async () => {
    const evidenceDashboard = structuredClone(dashboard);
    const evidence = {
      source_type: "plan_document",
      source_id: "DOC-2026-4.1",
      title: "碑林区下穿通道响应规程",
      excerpt: "达到橙色预警阈值后，应由区住建局立即封控下穿通道。",
      roles: ["condition", "procedure", "attribution"] as EvidenceRole[],
      document_version: "2026-A",
      document_version_id: "DOCVER-2026-A",
      clause: "4.1",
      source_locator: "document://DOCVER-2026-A?page=12#clause=4.1",
      page_number: 12,
      section_path: ["第四章", "4.1"],
      field_support: { trigger_condition: 0.92, action: 0.88 },
      conflicts_with: [],
    };
    const common = {
      package_id: "EVID-1",
      event_id: evidenceDashboard.event.event_id,
      object_id: "TUNNEL-017",
      status: "needs_review",
      task_schema_version: "response-task-schema-v2",
      retrieval_strategy: "FRC-RAG",
      retrieval_run_id: "RETRIEVAL-1",
      retrieval_mode: "SHADOW" as const,
      baseline_source_ids: ["DOC-2026-4.1"],
      frc_source_ids: ["DOC-2026-4.1"],
      shadow_comparison: { overlap: ["DOC-2026-4.1"], baseline_only: [], frc_only: [] },
      role_coverage: { condition: true, object: true, responsibility: true, procedure: true, exception: true, attribution: true },
      evidence: [evidence],
      conflicts: [],
      required_fields: ["trigger_condition", "risk_object", "responsible_party", "action", "deadline", "feedback_requirement", "escalation_condition"],
      nli_status: "unavailable",
      nli_model_version: "nli-unavailable",
      nli_assessments: [{
        assessment_id: "NLI-1",
        left_source_id: "DOC-2026-4.1",
        right_source_id: "REGISTRY-1",
        shared_fields: ["action"],
        relation: "unavailable" as const,
        confidence: 0,
        model_version: "nli-unavailable",
        status: "unavailable",
        error: "No versioned NLI model adapter is configured",
      }],
      created_by: "duty-1",
      created_at: "2026-07-15T08:00:00Z",
    };
    const version1 = {
      ...common,
      version: 1,
      field_states: { trigger_condition: "SUPPORTED", action: "MISSING" },
      missing_fields: ["action"],
      blocking_missing_fields: ["action"],
      missing_reasons: { action: "NOT_RETRIEVED" },
      field_evidence_map: { trigger_condition: ["DOC-2026-4.1"], action: [] },
      content_hash: "a".repeat(64),
    } satisfies EvidencePackageVersion;
    const version2 = {
      ...common,
      version: 2,
      field_states: { trigger_condition: "SUPPORTED", action: "SUPPORTED" },
      missing_fields: [],
      blocking_missing_fields: [],
      missing_reasons: {},
      field_evidence_map: { trigger_condition: ["DOC-2026-4.1"], action: ["DOC-2026-4.1"] },
      content_hash: "b".repeat(64),
      created_at: "2026-07-15T08:05:00Z",
    } satisfies EvidencePackageVersion;
    evidenceDashboard.evidence_packages = [version1, version2];
    vi.mocked(responseWorkflowApi.getDashboard).mockResolvedValue(evidenceDashboard);

    render(<ResponseWorkflowPage />);

    expect(await screen.findByText("NLI unavailable · nli-unavailable")).toBeInTheDocument();
    expect(screen.getByText("V1 → V2")).toBeInTheDocument();
    expect(screen.getByText("字段状态变化 1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看原文定位" }));
    expect(screen.getByText("document://DOCVER-2026-A?page=12#clause=4.1")).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "证据原文定位" })).toBeInTheDocument();
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

  it("展示可追溯候选特征并允许人工确认", async () => {
    const candidateDashboard = structuredClone(dashboard);
    candidateDashboard.tasks = [];
    candidateDashboard.risk_objects[0].verification_status = "pending";
    candidateDashboard.candidate_runs = [{
      run_id: "CANDRUN-1",
      event_id: candidateDashboard.event.event_id,
      alert_snapshot_id: "ALT-1",
      risk_object_data_version: "risk-object-v1",
      algorithm_version: "candidate-registry-rules-v1",
      feature_version: "candidate-registry-features-v2",
      association_mode: "area_registry",
      parameters: {},
      candidate_object_ids: ["TUNNEL-017"],
      candidate_features: [{
        object_id: "TUNNEL-017",
        rank: 1,
        spatial_score: 0.6,
        temporal_score: 0.75,
        attribute_score: 0.87,
        semantic_score: 0.5,
        data_quality_score: 1,
        raw_score: 87,
        calibrated_confidence: 0.935,
        missing_features: ["affected_geometry"],
        explanations: { spatial: "按区域关联" },
        feature_version: "candidate-registry-features-v2",
      }],
      missing_features: ["affected_geometry"],
      limitations: ["区域关联不代表精确空间相交"],
      status: "completed",
      created_by: "console_commander",
      terminal_id: "district-response-console",
      created_at: "2026-07-11T08:01:00Z",
    }];
    vi.mocked(responseWorkflowApi.getDashboard).mockResolvedValue(candidateDashboard);
    vi.mocked(responseWorkflowApi.verifyRiskObject).mockResolvedValue({});

    render(<ResponseWorkflowPage />);

    expect(await screen.findByText("候选排名 #1")).toBeInTheDocument();
    expect(screen.getByText("空间 60")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "人工确认" }));

    await waitFor(() => expect(responseWorkflowApi.verifyRiskObject).toHaveBeenCalledWith(
      "FLOOD-TEST-001",
      "TUNNEL-017",
      "commander",
      "confirmed",
    ));
  });

  it("以最新版本号冻结人工确认对象清单", async () => {
    vi.mocked(responseWorkflowApi.freezeCandidateObjectList).mockResolvedValue({
      list_id: "CANDLIST-1",
      event_id: dashboard.event.event_id,
      version: 1,
      status: "frozen",
      alert_snapshot_id: "ALT-1",
      source_run_ids: [],
      objects: [{
        object_id: "TUNNEL-017",
        object_version: 1,
        data_version: "risk-object-v1",
        verification_hash: "a".repeat(64),
      }],
      content_hash: "b".repeat(64),
      note: "冻结确认清单",
      is_simulated: true,
      frozen_by: "console_commander",
      frozen_role: "commander",
      terminal_id: "district-response-console",
      frozen_at: "2026-07-11T08:05:00Z",
    });

    render(<ResponseWorkflowPage />);
    fireEvent.click(await screen.findByRole("button", { name: "冻结确认清单" }));

    await waitFor(() => expect(responseWorkflowApi.freezeCandidateObjectList).toHaveBeenCalledWith(
      "FLOOD-TEST-001",
      "commander",
      ["TUNNEL-017"],
      undefined,
    ));
  });

  it("展示文档不可变版本状态并按角色执行解析", async () => {
    const draftDocument = {
      version_id: "DOCVER-1",
      document_id: "SIM-DOC-下穿通道响应规程",
      version_number: 1,
      version_label: "2026-A",
      title: "下穿通道响应规程",
      issuer: "模拟区防办",
      jurisdiction: "district-simulation",
      effective_at: "2026-07-14T00:00:00Z",
      expires_at: null,
      replaces_version_id: null,
      lifecycle_status: "draft" as const,
      source_hash: "a".repeat(64),
      source_filename: "2026-A.txt",
      media_type: "text/plain",
      source_size: 64,
      clauses: [],
      parser_version: null,
      parse_method: null,
      parse_hash: null,
      parsed_at: null,
      index_status: "not_indexed",
      index_version: "",
      index_build_id: null,
      published_at: null,
      retired_at: null,
      superseded_by_version_id: null,
      is_simulated: true,
      created_by: "admin-1",
      terminal_id: "document-terminal",
      created_at: "2026-07-14T00:00:00Z",
    };
    vi.mocked(responseWorkflowApi.listDocuments).mockResolvedValue([draftDocument]);
    vi.mocked(responseWorkflowApi.parseDocumentVersion).mockResolvedValue({
      ...draftDocument,
      lifecycle_status: "parsed",
    });

    render(<ResponseWorkflowPage />);

    expect(await screen.findByText("2026-A · 待解析")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("当前业务岗位"), { target: { value: "admin" } });
    fireEvent.click(screen.getByRole("button", { name: "解析条款" }));

    await waitFor(() => expect(responseWorkflowApi.parseDocumentVersion).toHaveBeenCalledWith(
      "DOCVER-1",
      "admin",
      1,
    ));
  });

  it("以草稿方式登记人工条款并自动绑定上一版本", async () => {
    const currentDocument = {
      version_id: "DOCVER-CURRENT",
      document_id: "SIM-DOC-下穿通道响应规程",
      version_number: 1,
      version_label: "2026-A",
      title: "下穿通道响应规程",
      issuer: "模拟区防办",
      jurisdiction: "district-simulation",
      effective_at: "2026-07-14T00:00:00Z",
      lifecycle_status: "published" as const,
      source_hash: "b".repeat(64),
      source_filename: "2026-A.txt",
      media_type: "text/plain",
      source_size: 64,
      clauses: [],
      index_status: "indexed",
      index_version: "frc-index-v1",
      is_simulated: true,
      created_by: "admin-1",
      terminal_id: "document-terminal",
      created_at: "2026-07-14T00:00:00Z",
    };
    vi.mocked(responseWorkflowApi.listDocuments).mockResolvedValue([currentDocument]);
    vi.mocked(responseWorkflowApi.createDocumentVersion).mockResolvedValue({
      ...currentDocument,
      version_id: "DOCVER-DRAFT",
      version_number: 2,
      version_label: "2026-B",
      lifecycle_status: "draft",
      index_status: "not_indexed",
      index_version: "",
    });

    render(<ResponseWorkflowPage />);
    await screen.findByText("2026-A · 已发布");
    fireEvent.change(screen.getByLabelText("当前业务岗位"), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText("文档名称"), { target: { value: "下穿通道响应规程" } });
    fireEvent.change(screen.getByLabelText("版本"), { target: { value: "2026-B" } });
    fireEvent.change(screen.getByLabelText("条款原文"), { target: { value: "第一条 红色预警时应立即封控下穿通道。" } });
    fireEvent.click(screen.getByRole("button", { name: "登记草稿" }));

    await waitFor(() => expect(responseWorkflowApi.createDocumentVersion).toHaveBeenCalledWith(
      expect.objectContaining({
        documentId: "SIM-DOC-下穿通道响应规程",
        versionLabel: "2026-B",
        content: "第一条 红色预警时应立即封控下穿通道。",
        replacesVersionId: "DOCVER-CURRENT",
        operatorRole: "admin",
      }),
    ));
  });

  it("将带坐标的版本化风险对象主数据联动到 Cesium 图层", async () => {
    vi.mocked(responseWorkflowApi.listRiskObjectRegistry).mockResolvedValue([
      {
        area_id: "beilin_10km2",
        object_id: "SCHOOL-REGISTRY-001",
        canonical_object_id: "SCHOOL-REGISTRY-001",
        aliases: [],
        duplicate_of: null,
        name: "文艺路重点学校",
        object_type: "学校",
        location: "碑林区文艺路",
        longitude: 108.958,
        latitude: 34.244,
        responsible_organization: "区教育局",
        responsible_role: "学校防汛负责人",
        trigger_reasons: ["纳入区级风险对象台账"],
        source_refs: ["district-registry:school-2026"],
        vulnerability: "低龄学生集中",
        historical_risk: "",
        risk_score: 88,
        system_explanation: "主数据登记风险分",
        sensitive_contacts: [],
        special_population_notes: "",
        source_type: "governed_registry",
        source_version: "district-registry-2026.1",
        data_version: "district-registry-2026.1:abc123",
        is_simulated: true,
        missing_fields: [],
        registry_status: "active",
        registry_valid_from: null,
        registry_valid_until: null,
        registry_version: 1,
        source_hash: "a".repeat(64),
        content_hash: "b".repeat(64),
        source_filename: "risk-objects.xlsx",
        created_by: "registry-reviewer",
        terminal_id: "registry-console",
        created_at: "2026-07-14T00:00:00Z",
        updated_at: "2026-07-14T00:00:00Z",
      },
    ]);

    render(<ResponseWorkflowPage />);

    expect(await screen.findByLabelText("digital-twin-canvas")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /文艺路重点学校 \/ 监测中/ })).toBeInTheDocument();
    expect(screen.getByText("1 个可定位对象")).toBeInTheDocument();
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

  it("可选择故障场景处理单条 Outbox 并展示乱序回调", async () => {
    const dispatchDashboard = structuredClone(dashboard);
    dispatchDashboard.tasks[0].status = "issued";
    dispatchDashboard.tasks[0].dispatch_message_id = "OUTBOX-1";
    dispatchDashboard.outbox = [{
      message_id: "OUTBOX-1",
      event_id: dispatchDashboard.event.event_id,
      task_id: dispatchDashboard.tasks[0].task_id,
      destination: "simulated://member-unit",
      idempotency_key: "dispatch:TASK-1:v1",
      payload_hash: "payload-hash",
      approval_id: "APPROVAL-1",
      task_version: 1,
      status: "pending",
      attempts: 0,
      simulation_scenario: "normal",
      callback_count: 1,
      created_at: "2026-07-11T08:00:00Z",
      updated_at: "2026-07-11T08:00:00Z",
    }];
    vi.mocked(responseWorkflowApi.getDashboard).mockResolvedValue(dispatchDashboard);
    vi.mocked(responseWorkflowApi.processOutbox).mockResolvedValue([]);
    vi.mocked(responseWorkflowApi.listDispatchCallbacks).mockResolvedValue([{
      callback_id: "CALLBACK-1",
      message_id: "OUTBOX-1",
      event_id: dispatchDashboard.event.event_id,
      task_id: dispatchDashboard.tasks[0].task_id,
      external_id: "SIMDISPATCH-1",
      source: "deterministic_simulation_gateway",
      version: 2,
      event_time: "2026-07-11T08:01:00Z",
      received_time: "2026-07-11T08:01:01Z",
      request_id: "SIMREQ-1",
      trace_id: "SIMTRACE-1",
      idempotency_key: "callback-1",
      status: "delivered",
      sequence_state: "out_of_order",
      is_simulated: true,
      created_at: "2026-07-11T08:01:01Z",
    }]);

    render(<ResponseWorkflowPage />);

    expect(await screen.findByRole("region", { name: "模拟下发场景控制台" })).toBeInTheDocument();
    expect(await screen.findByText(/乱序 · SIMTRACE-1/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("故障注入场景"), { target: { value: "out_of_order_callback" } });
    fireEvent.click(screen.getByRole("button", { name: "执行受控场景" }));

    await waitFor(() => expect(responseWorkflowApi.processOutbox).toHaveBeenCalledWith("OUTBOX-1", "out_of_order_callback", "commander"));
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
