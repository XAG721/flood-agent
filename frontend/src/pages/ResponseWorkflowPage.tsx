import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { responseWorkflowApi } from "../api/responseWorkflowApi";
import { setApiOperatorContext } from "../lib/httpClient";
import type { EntityType, RiskLevel, TwinObjectMapLayer } from "../types/api";
import type { DeadlineExtensionRecord, DispatchCallbackRecord, DistrictScenarioReport, DocumentVersionRecord, EscalationRecord, EvidencePackageVersion, EvidenceRole, EventDashboard, EventReviewDraft, OutboxMessage, ResponseTask, ResponseTaskStatus, RetrievalMode, RiskObjectRegistryImportResult, RiskObjectRegistryRecord, RuleEvaluationRecord, SimulationDispatchScenario, WorkflowRole } from "../types/response";
import styles from "./response-workflow-page.module.css";

const DigitalTwinCesiumCanvas = lazy(() =>
  import("../components/DigitalTwinCesiumCanvas").then((module) => ({
    default: module.DigitalTwinCesiumCanvas,
  })),
);

const roleText: Record<WorkflowRole, string> = {
  duty_officer: "防办值班员",
  reviewer: "防办审核员",
  commander: "指挥审批员",
  liaison: "成员单位联络员",
  field_operator: "现场执行员",
  auditor: "审计查看员",
  admin: "系统管理员",
};

const statusText: Record<ResponseTaskStatus, string> = {
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

const actionText: Record<string, string> = {
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

const dispatchScenarioText: Record<SimulationDispatchScenario, string> = {
  normal: "正常接收与送达",
  timeout: "网关超时（可重试）",
  reject: "外部通道拒收",
  partial_success: "批量部分成功",
  duplicate_callback: "重复回调",
  out_of_order_callback: "乱序回调",
};

const evidenceRoleText: Record<EvidenceRole, string> = {
  condition: "触发条件",
  object: "风险对象",
  responsibility: "责任岗位",
  procedure: "处置流程",
  exception: "例外升级",
  attribution: "版本归因",
};

function formatDate(value?: string | null) {
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

function registryMapLayers(records: RiskObjectRegistryRecord[]): TwinObjectMapLayer[] {
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

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

async function prepareRegistryFile(file: File) {
  if (file.size > 10 * 1024 * 1024) {
    throw new Error("风险对象文件不得超过 10 MiB");
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error("当前浏览器不支持文件 SHA-256 校验");
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes));
  const sha256 = Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
  const suffix = file.name.split(".").pop()?.toLowerCase();
  const mediaType = file.type || {
    csv: "text/csv",
    xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    json: "application/json",
    geojson: "application/geo+json",
  }[suffix ?? ""];
  if (!mediaType) {
    throw new Error("只支持 CSV、XLSX、JSON 或 GeoJSON 风险对象文件");
  }
  return {
    filename: file.name,
    mediaType,
    contentBase64: bytesToBase64(bytes),
    sha256,
  };
}

async function prepareDocumentFile(file: File) {
  if (file.size > 10 * 1024 * 1024) {
    throw new Error("文档文件不得超过 10 MiB");
  }
  if (!globalThis.crypto?.subtle) {
    throw new Error("当前浏览器不支持文件 SHA-256 校验");
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes));
  const sha256 = Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
  const suffix = file.name.split(".").pop()?.toLowerCase();
  const mediaType = file.type || {
    pdf: "application/pdf",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    csv: "text/csv",
    txt: "text/plain",
    md: "text/markdown",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    tif: "image/tiff",
    tiff: "image/tiff",
  }[suffix ?? ""];
  if (!mediaType) {
    throw new Error("只支持 PDF、DOCX、XLSX、CSV、TXT、Markdown 或扫描图片");
  }
  return {
    filename: file.name,
    mediaType,
    contentBase64: bytesToBase64(bytes),
    sha256,
    scanned: mediaType.startsWith("image/"),
  };
}

export function ResponseWorkflowPage() {
  const [dashboard, setDashboard] = useState<EventDashboard | null>(null);
  const [documents, setDocuments] = useState<DocumentVersionRecord[]>([]);
  const [registry, setRegistry] = useState<RiskObjectRegistryRecord[]>([]);
  const [lastRegistryImport, setLastRegistryImport] = useState<RiskObjectRegistryImportResult | null>(null);
  const [selectedRegistryObjectId, setSelectedRegistryObjectId] = useState<string | null>(null);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [role, setRole] = useState<WorkflowRole>("commander");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("SHADOW");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("正在连接响应工作流……");

  useEffect(() => {
    setApiOperatorContext({
      id: `console_${role}`,
      role,
      terminalId: "district-response-console",
    });
  }, [role]);

  const load = async () => {
    setBusy(true);
    try {
      const events = await responseWorkflowApi.listEvents();
      const next = events.length ? await responseWorkflowApi.getDashboard(events[0].event_id) : await responseWorkflowApi.bootstrapDemo();
      const [nextDocuments, nextRegistry] = await Promise.all([
        responseWorkflowApi.listDocuments(),
        responseWorkflowApi.listRiskObjectRegistry(next.event.area_id),
      ]);
      setDashboard(next);
      setDocuments(nextDocuments);
      setRegistry(nextRegistry);
      setSelectedTaskId((current) => current ?? next.tasks[0]?.task_id ?? null);
      setMessage("业务台账已同步");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "响应工作流加载失败");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const selectedTask = useMemo(
    () => dashboard?.tasks.find((task) => task.task_id === selectedTaskId) ?? dashboard?.tasks[0] ?? null,
    [dashboard, selectedTaskId],
  );

  const run = async (label: string, operation: () => Promise<unknown>) => {
    setBusy(true);
    setMessage(`${label}处理中……`);
    try {
      await operation();
      if (dashboard) {
        const [next, nextDocuments, nextRegistry] = await Promise.all([
          responseWorkflowApi.getDashboard(dashboard.event.event_id),
          responseWorkflowApi.listDocuments(),
          responseWorkflowApi.listRiskObjectRegistry(dashboard.event.area_id),
        ]);
        setDashboard(next);
        setDocuments(nextDocuments);
        setRegistry(nextRegistry);
      }
      setMessage(`${label}完成，已写入事件台账`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : `${label}失败`);
    } finally {
      setBusy(false);
    }
  };

  const generateDraft = async (objectId: string) => {
    if (!dashboard) return;
    setBusy(true);
    setMessage("正在检索预案并检查六类证据角色……");
    try {
      const latestCandidateList = dashboard.candidate_object_lists[dashboard.candidate_object_lists.length - 1];
      const result = await responseWorkflowApi.generateTaskDraft(
        dashboard.event.event_id,
        objectId,
        role,
        retrievalMode,
        latestCandidateList?.version,
      );
      const next = await responseWorkflowApi.getDashboard(dashboard.event.event_id);
      setDashboard(next);
      if (result.task) setSelectedTaskId(result.task.task_id);
      setMessage(
        result.status === "ready"
          ? "证据闸门通过，结构化任务草案已写入草稿区"
          : `证据不足，未创建任务：${result.validation_errors.join("；")}`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "任务草案生成失败");
    } finally {
      setBusy(false);
    }
  };

  const importRegistryFile = async (file: File) => {
    if (!dashboard) return;
    if (!["reviewer", "admin"].includes(role)) {
      setMessage("只有业务复核岗或管理员可以导入风险对象主数据");
      return;
    }
    setBusy(true);
    setMessage("正在校验文件哈希并导入风险对象主数据……");
    try {
      const prepared = await prepareRegistryFile(file);
      const result = await responseWorkflowApi.importRiskObjectFile({
        areaId: dashboard.event.area_id,
        sourceVersion: `web-${file.lastModified || Date.now()}`,
        ...prepared,
        operatorRole: role,
      });
      const [nextDashboard, nextRegistry] = await Promise.all([
        responseWorkflowApi.getDashboard(dashboard.event.event_id),
        responseWorkflowApi.listRiskObjectRegistry(dashboard.event.area_id),
      ]);
      setDashboard(nextDashboard);
      setRegistry(nextRegistry);
      setLastRegistryImport(result);
      setMessage(
        `主数据导入完成：新增 ${result.created_count}、更新 ${result.updated_count}、隔离 ${result.quarantined_count}；${result.stale_candidate_run_count} 个候选运行已标记 STALE`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "风险对象主数据导入失败");
    } finally {
      setBusy(false);
    }
  };

  const mapLayers = useMemo(() => registryMapLayers(registry), [registry]);

  if (!dashboard) {
    return (
      <section className={styles.loadingState} aria-live="polite">
        <strong>响应闭环工作台</strong>
        <span>{message}</span>
        <button type="button" onClick={() => void load()} disabled={busy}>重新连接</button>
      </section>
    );
  }

  const latestAlert = dashboard.alert_snapshots[dashboard.alert_snapshots.length - 1];
  const latestCandidateRun = dashboard.candidate_runs[dashboard.candidate_runs.length - 1];
  const latestCandidateList = dashboard.candidate_object_lists[dashboard.candidate_object_lists.length - 1];
  const confirmedObjectIds = dashboard.risk_objects
    .filter((item) => item.verification_status === "confirmed" && !item.stale)
    .map((item) => item.object_id);
  const allTasksClosed = dashboard.tasks.length > 0 && dashboard.tasks.every((task) => ["completed", "waived"].includes(task.status));
  const draftableObject = dashboard.risk_objects.find(
    (item) => item.verification_status === "confirmed"
      && !dashboard.tasks.some((task) => task.object_id === item.object_id)
      && (!latestCandidateList || latestCandidateList.objects.some((reference) => reference.object_id === item.object_id)),
  );

  return (
    <div className={styles.page}>
      {dashboard.event.is_simulated ? (
        <div className={styles.simulationBanner} role="status">
          <strong>受控模拟环境</strong>
          <span>当前预警、对象和下发均为模拟数据，不连接真实跨部门端点。</span>
        </div>
      ) : null}
      <section className={styles.eventHeader} aria-labelledby="response-event-title">
        <div>
          <div className={styles.eventIdentity}>
            <span className={styles.levelBadge}>{latestAlert?.level ?? "待接入"}</span>
            <span>{dashboard.event.event_id}</span>
            <span>预警版本 V{dashboard.event.current_alert_version}</span>
            <span>{dashboard.event.workflow_engine_version}</span>
            <span>{dashboard.event.data_version}</span>
          </div>
          <h2 id="response-event-title">{dashboard.event.title}</h2>
          <p>{latestAlert?.source_department} · {latestAlert?.affected_area} · 发布于 {formatDate(latestAlert?.issued_at)}</p>
        </div>
        <div className={styles.eventControls}>
          <label>
            <span>当前业务岗位</span>
            <select value={role} onChange={(event) => setRole(event.target.value as WorkflowRole)} disabled={busy}>
              {Object.entries(roleText).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <button
            type="button"
            className={styles.secondaryButton}
            disabled={busy || dashboard.event.status === "closed"}
            onClick={() => void run("时限巡检", () => responseWorkflowApi.runDeadlineSweep(dashboard.event.event_id, role))}
          >
            执行时限巡检
          </button>
        </div>
      </section>

      <ol className={styles.processRail} aria-label="预警响应业务流程">
        {["专业预警", "对象核验", "任务成案", "人工审批", "部门执行", "反馈核实", "事件复盘"].map((item, index) => (
          <li key={item} className={index <= 4 ? styles.processActive : undefined}>
            <span>{index + 1}</span>{item}
          </li>
        ))}
      </ol>

      <section className={styles.metrics} aria-label="事件过程指标">
        <div><span>待核验/确认对象</span><strong>{dashboard.metrics.candidate_objects}/{dashboard.metrics.confirmed_objects}</strong></div>
        <div><span>任务总数</span><strong>{dashboard.metrics.task_count}</strong></div>
        <div><span>已完成</span><strong>{dashboard.metrics.completed_tasks}</strong></div>
        <div><span>异常升级</span><strong>{dashboard.metrics.escalated_tasks}</strong></div>
        <div><span>现场反馈</span><strong>{dashboard.metrics.feedback_count ?? dashboard.feedback.length}</strong></div>
        <div><span>台账记录</span><strong>{dashboard.metrics.timeline_entries}</strong></div>
      </section>

      <div className={styles.workspace}>
        <section className={styles.registrySection} aria-labelledby="risk-registry-heading">
          <header>
            <div>
              <h3 id="risk-registry-heading">风险对象主数据</h3>
              <p>受控导入、逐对象内容哈希、不可变版本与 Cesium 空间定位共用同一份主数据。</p>
            </div>
            <div className={styles.sectionActions}>
              <span>{registry.length} 个有效对象</span>
              <span>{mapLayers.length} 个可定位对象</span>
              <label className={styles.registryFileAction} aria-disabled={busy || !["reviewer", "admin"].includes(role)}>
                导入主数据文件
                <input
                  type="file"
                  accept=".csv,.xlsx,.json,.geojson"
                  disabled={busy || !["reviewer", "admin"].includes(role)}
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    event.currentTarget.value = "";
                    if (file) void importRegistryFile(file);
                  }}
                />
              </label>
            </div>
          </header>
          {lastRegistryImport ? (
            <div className={styles.registryImportResult} role="status">
              <strong>{lastRegistryImport.source_filename}</strong>
              <span>新增 {lastRegistryImport.created_count}</span>
              <span>更新 {lastRegistryImport.updated_count}</span>
              <span>未变 {lastRegistryImport.unchanged_count}</span>
              <span>隔离 {lastRegistryImport.quarantined_count}</span>
              <span>SHA-256 {lastRegistryImport.source_hash.slice(0, 12)}…</span>
            </div>
          ) : null}
          {mapLayers.length ? (
            <div className={styles.registryMap}>
              <Suspense fallback={<p className={styles.emptyText}>正在加载 Cesium 风险对象图层……</p>}>
                <DigitalTwinCesiumCanvas
                  layers={mapLayers}
                  dialogFocusObjectId={selectedRegistryObjectId}
                  onSelectObject={setSelectedRegistryObjectId}
                />
              </Suspense>
            </div>
          ) : (
            <p className={styles.emptyText}>当前区域尚无带 EPSG:4326 坐标的有效主数据；可导入文件后在此定位。</p>
          )}
        </section>

        <section className={styles.objectSection} aria-labelledby="risk-object-heading">
          <header>
            <div>
              <h3 id="risk-object-heading">待核验风险对象</h3>
              <p>
                {latestAlert?.affected_geometry
                  ? "按 EPSG:4326 预警多边形与对象坐标做点落区筛查；结果仍须防办人工核验。"
                  : "预警未提供空间多边形，当前按事件区域关联登记台账，不代表精确 GIS 相交。"}
              </p>
            </div>
            <div className={styles.sectionActions}>
              <span>{dashboard.risk_objects.length} 个候选</span>
              <span>{latestCandidateRun ? `运行 ${latestCandidateRun.run_id}` : "尚未运行"}</span>
              <span>{latestCandidateList ? `冻结清单 V${latestCandidateList.version} · ${latestCandidateList.objects.length} 项` : "尚未冻结确认清单"}</span>
              <button
                type="button"
                disabled={busy || dashboard.event.status === "closed" || !["duty_officer", "reviewer", "commander", "admin"].includes(role)}
                onClick={() => void run("风险对象筛查", () => responseWorkflowApi.discoverRiskObjects(dashboard.event.event_id, role))}
              >
                筛查预警范围
              </button>
              <button
                type="button"
                disabled={busy || !confirmedObjectIds.length || !["reviewer", "commander"].includes(role)}
                onClick={() => void run("冻结确认对象清单", () => responseWorkflowApi.freezeCandidateObjectList(
                  dashboard.event.event_id,
                  role,
                  confirmedObjectIds,
                  latestCandidateList?.version,
                ))}
              >
                冻结确认清单
              </button>
            </div>
          </header>
          <div className={styles.objectList}>
            {dashboard.risk_objects.length === 0 ? (
              <p className={styles.emptyText}>尚无候选对象。先筛查当前预警范围，再由防办逐项人工核验。</p>
            ) : null}
            {dashboard.risk_objects.map((item) => (
              <article key={item.object_id} className={styles.objectRow}>
                <div className={styles.riskScore} aria-label={`风险分 ${item.risk_score}`}>{item.risk_score}</div>
                <div className={styles.objectMain}>
                  <div className={styles.rowTitle}><strong>{item.name}</strong><span>{item.object_type}</span></div>
                  <p>{item.location}</p>
                  <div className={styles.reasonList}>{item.trigger_reasons.map((reason) => <span key={reason}>{reason}</span>)}</div>
                  <small>{item.responsible_organization} · {item.responsible_role}</small>
                  <small>{item.source_type} · {item.source_version} · {item.data_version}{item.stale ? " · STALE" : ""}</small>
                  <small>{item.association_mode} · 校准置信度 {item.calibrated_confidence == null ? "待计算" : `${Math.round(item.calibrated_confidence * 100)}%`} · {item.calibration_version}</small>
                  {item.missing_fields.length ? <small className={styles.missingData}>缺失字段：{item.missing_fields.join("、")}</small> : null}
                  {latestCandidateRun?.candidate_features
                    .filter((feature) => feature.object_id === item.object_id)
                    .map((feature) => (
                      <div className={styles.featureSnapshot} key={`${latestCandidateRun.run_id}-${feature.object_id}`}>
                        <strong>候选排名 #{feature.rank}</strong>
                        <span>空间 {Math.round(feature.spatial_score * 100)}</span>
                        <span>时效 {Math.round(feature.temporal_score * 100)}</span>
                        <span>属性 {Math.round(feature.attribute_score * 100)}</span>
                        <span>语义 {Math.round(feature.semantic_score * 100)}</span>
                        <span>质量 {Math.round(feature.data_quality_score * 100)}</span>
                        <small>{feature.feature_version}</small>
                      </div>
                    ))}
                  {item.verification_status === "pending" ? (
                    <div className={styles.objectActions}>
                      <button
                        type="button"
                        disabled={busy || !["duty_officer", "reviewer", "commander", "admin"].includes(role)}
                        onClick={() => void run("确认候选对象", () => responseWorkflowApi.verifyRiskObject(dashboard.event.event_id, item.object_id, role, "confirmed"))}
                      >
                        人工确认
                      </button>
                      <button
                        type="button"
                        disabled={busy || !["duty_officer", "reviewer", "commander", "admin"].includes(role)}
                        onClick={() => void run("排除候选对象", () => responseWorkflowApi.verifyRiskObject(dashboard.event.event_id, item.object_id, role, "excluded"))}
                      >
                        排除
                      </button>
                    </div>
                  ) : null}
                </div>
                <span className={`${styles.stateBadge} ${styles[item.verification_status]}`}>{item.verification_status === "confirmed" ? "已人工确认" : item.verification_status === "excluded" ? "已排除" : "待核验"}</span>
              </article>
            ))}
          </div>
        </section>

        <EvidenceWorkbench packages={dashboard.evidence_packages} role={role} busy={busy} run={run} />

        <DocumentRegistry documents={documents} role={role} busy={busy} run={run} />

        <section className={styles.taskSection} aria-labelledby="task-heading">
          <header>
            <div><h3 id="task-heading">对象级处置任务</h3><p>任务状态、责任、依据和证据要求均写入不可覆盖时间线。</p></div>
            <div className={styles.sectionActions}>
              <span>{dashboard.tasks.length} 项</span>
              <label>
                <span>检索模式</span>
                <select value={retrievalMode} onChange={(event) => setRetrievalMode(event.target.value as RetrievalMode)} disabled={busy}>
                  <option value="SHADOW">影子对照</option>
                  <option value="REVIEW">人工评审合并</option>
                  <option value="BASELINE_ONLY">基线检索</option>
                  <option value="CANARY">灰度 FRC（需 Gate 2）</option>
                  <option value="DEFAULT">FRC 正式（需 Gate 2）</option>
                </select>
              </label>
              {draftableObject ? (
                <button type="button" disabled={busy} onClick={() => void generateDraft(draftableObject.object_id)}>
                  生成证据草案
                </button>
              ) : null}
            </div>
          </header>
          <div className={styles.taskLayout}>
            <nav className={styles.taskList} aria-label="任务列表">
              {dashboard.tasks.map((task) => (
                <button
                  type="button"
                  key={task.task_id}
                  className={task.task_id === selectedTask?.task_id ? styles.taskSelected : undefined}
                  onClick={() => setSelectedTaskId(task.task_id)}
                >
                  <span>{task.responsible_organization}</span>
                  <strong>{task.title}</strong>
                  <small>{statusText[task.status]} · 截止 {formatDate(task.deadline_at)}</small>
                </button>
              ))}
            </nav>
            {selectedTask ? (
              <TaskDetail
                task={selectedTask}
                escalation={latestEscalation(dashboard.escalations, selectedTask.task_id)}
                evidencePackage={dashboard.evidence_packages.find((item) => item.package_id === selectedTask.evidence_package_id)}
                outboxMessage={dashboard.outbox.find((item) => item.message_id === selectedTask.dispatch_message_id)}
                ruleEvaluation={dashboard.rule_evaluations.filter((item) => item.task_id === selectedTask.task_id).slice(-1)[0]}
                deadlineExtension={dashboard.deadline_extensions.filter((item) => item.task_id === selectedTask.task_id).slice(-1)[0]}
                role={role}
                busy={busy}
                run={run}
              />
            ) : <p className={styles.emptyText}>当前事件尚无任务。</p>}
          </div>
        </section>

        <aside className={styles.timeline} aria-labelledby="timeline-heading">
          <header><h3 id="timeline-heading">事件台账</h3><span>追加记录</span></header>
          <ol>
            {[...dashboard.timeline].reverse().slice(0, 12).map((entry) => (
              <li key={entry.entry_id}>
                <time>{formatDate(entry.created_at)}</time>
                <strong>{actionText[entry.action] ?? entry.action}</strong>
                <span>{entry.actor_id} · {roleText[entry.actor_role as WorkflowRole] ?? entry.actor_role}</span>
              </li>
            ))}
          </ol>
        </aside>
      </div>

      {dashboard.review_draft ? <ReviewDraftSection review={dashboard.review_draft} /> : null}
      {dashboard.scenario_report ? <ScenarioReportSection report={dashboard.scenario_report} /> : null}

      <footer className={styles.eventFooter}>
        <p aria-live="polite">{message}</p>
        <div className={styles.eventFooterActions}>
          {dashboard.event.status === "closed" ? (
            <button
              type="button"
              disabled={busy || !["reviewer", "commander", "auditor"].includes(role)}
              onClick={() => void run(dashboard.scenario_report ? "重跑场景评测" : "运行场景评测", () => responseWorkflowApi.runScenarioEvaluation(dashboard.event.event_id, role))}
            >
              {dashboard.scenario_report ? "重跑区县场景评测" : "运行区县场景评测"}
            </button>
          ) : (
            <button
              type="button"
              disabled={busy || !allTasksClosed}
              onClick={() => void run("关闭事件", () => responseWorkflowApi.closeEvent(dashboard.event.event_id, role))}
            >
              核验闭环并关闭事件
            </button>
          )}
        </div>
      </footer>
    </div>
  );
}

function TaskDetail({ task, escalation, evidencePackage, outboxMessage, ruleEvaluation, deadlineExtension, role, busy, run }: { task: ResponseTask; escalation?: EscalationRecord; evidencePackage?: EvidencePackageVersion; outboxMessage?: OutboxMessage; ruleEvaluation?: RuleEvaluationRecord; deadlineExtension?: DeadlineExtensionRecord; role: WorkflowRole; busy: boolean; run: (label: string, operation: () => Promise<unknown>) => Promise<void> }) {
  const [dispatchScenario, setDispatchScenario] = useState<SimulationDispatchScenario>("normal");
  const [dispatchCallbacks, setDispatchCallbacks] = useState<DispatchCallbackRecord[]>([]);

  useEffect(() => {
    if (!outboxMessage) {
      setDispatchCallbacks([]);
      return;
    }
    void responseWorkflowApi.listDispatchCallbacks(outboxMessage.message_id)
      .then(setDispatchCallbacks)
      .catch(() => setDispatchCallbacks([]));
  }, [outboxMessage?.message_id, outboxMessage?.callback_count]);

  const primaryAction = (() => {
    if (task.status === "issued" || task.status === "escalated") return { label: "确认接收", action: () => responseWorkflowApi.acknowledgeTask(task.task_id, role) };
    if (task.status === "acknowledged" && !task.assignee_id) return { label: "分派现场执行员", action: () => responseWorkflowApi.assignTask(task.task_id, role) };
    if (task.status === "acknowledged") return { label: "开始执行", action: () => responseWorkflowApi.startTask(task.task_id, role) };
    if (["in_progress", "partially_completed", "taken_over"].includes(task.status)) return { label: "提交完成证据", action: () => responseWorkflowApi.completeTask(task.task_id, role, task.required_evidence) };
    if (task.status === "pending_verification") return { label: "核实通过", action: () => responseWorkflowApi.verifyTask(task.task_id, role, true) };
    return null;
  })();

  return (
    <article className={styles.taskDetail}>
      <div className={styles.taskHeadline}>
        <div><span>{task.task_id} · V{task.version}</span><h4>{task.title}</h4></div>
        <span className={`${styles.stateBadge} ${styles[task.status]}`}>{statusText[task.status]}</span>
      </div>
      <p className={styles.taskAction}>{task.action}</p>
      <dl className={styles.taskFacts}>
        <div><dt>责任岗位</dt><dd>{task.responsible_organization} · {task.responsible_role}</dd></div>
        <div><dt>确认时限</dt><dd>{formatDate(task.acknowledge_deadline_at)}</dd></div>
        <div><dt>开始时限</dt><dd>{formatDate(task.effective_start_deadline_at ?? task.start_deadline_at)}</dd></div>
        <div><dt>完成时限</dt><dd>{formatDate(task.effective_completion_deadline_at ?? task.deadline_at)}</dd></div>
        <div><dt>核验时限</dt><dd>{formatDate(task.effective_verification_deadline_at ?? task.verification_deadline_at)}</dd></div>
        <div><dt>审批要求</dt><dd>{task.approval_policy === "commander_required" ? "指挥审批员批准" : "防办审核员批准"}</dd></div>
        <div><dt>执行合同</dt><dd>{task.task_schema_version} · {task.rule_set_version}</dd></div>
        <div><dt>现场执行人</dt><dd>{task.assignee_name ? `${task.assignee_name} · 分派 V${task.assignment_version}` : "待成员单位联络员分派"}</dd></div>
      </dl>
      <div className={styles.taskEvidence}>
        <div><span>反馈证据</span>{task.required_evidence.map((item) => <strong key={item}>{item}</strong>)}</div>
        <div><span>预案依据</span>{task.plan_basis.map((item) => <strong key={`${item.document}-${item.clause}`}>{item.document} V{item.version} · {item.clause}</strong>)}</div>
      </div>
      {task.generated_by_ai ? <p className={styles.aiNotice}>智能辅助草案 · {task.generation_version}。未经人工审批不得作为正式指令。</p> : null}
      {task.generated_by_ai ? <EvidenceGate task={task} /> : null}
      {ruleEvaluation ? <RuleGate evaluation={ruleEvaluation} /> : null}
      {evidencePackage ? (
        <p className={styles.contractLine}>证据包 {evidencePackage.package_id} · V{evidencePackage.version} · {evidencePackage.status} · 哈希 {evidencePackage.content_hash.slice(0, 12)}</p>
      ) : null}
      {outboxMessage ? (
        <section className={styles.dispatchConsole} aria-label="模拟下发场景控制台">
          <header>
            <div>
              <strong>模拟下发场景控制台</strong>
              <p className={outboxMessage.status === "failed" ? styles.dispatchFailed : styles.contractLine}>
                {outboxMessage.status === "sent" ? "已送达" : outboxMessage.status === "partially_sent" ? "部分成功，等待人工协调" : outboxMessage.status === "failed" ? "失败并关闭" : outboxMessage.status === "manual_takeover" ? "已转人工接管" : "待处理"}
                {" · "}场景 {dispatchScenarioText[outboxMessage.simulation_scenario as SimulationDispatchScenario] ?? outboxMessage.simulation_scenario} · 回调 {outboxMessage.callback_count} 条 · 尝试 {outboxMessage.attempts} 次
              </p>
            </div>
            <span>{outboxMessage.message_id}</span>
          </header>
          {outboxMessage.status === "pending" ? (
            <div className={styles.dispatchControls}>
              <label>
                <span>故障注入场景</span>
                <select value={dispatchScenario} onChange={(event) => setDispatchScenario(event.target.value as SimulationDispatchScenario)} disabled={busy}>
                  {Object.entries(dispatchScenarioText).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                </select>
              </label>
              <button
                type="button"
                disabled={busy || !["admin", "liaison", "commander"].includes(role)}
                onClick={() => void run("运行模拟下发场景", () => responseWorkflowApi.processOutbox(outboxMessage.message_id, dispatchScenario, role))}
              >执行受控场景</button>
            </div>
          ) : null}
          {dispatchCallbacks.length ? (
            <ol className={styles.callbackList} aria-label="模拟下发回调记录">
              {dispatchCallbacks.map((callback) => (
                <li key={callback.callback_id}>
                  <strong>V{callback.version} · {callback.status}</strong>
                  <span>{callback.sequence_state === "out_of_order" ? "乱序" : "顺序正常"} · {callback.trace_id}</span>
                </li>
              ))}
            </ol>
          ) : <p className={styles.dispatchEmpty}>尚无回调；超时场景保持待处理，可切换正常场景重试。</p>}
        </section>
      ) : null}
      {deadlineExtension ? (
        <p className={styles.contractLine}>
          延期申请 {deadlineExtension.status === "approved" ? "已批准" : deadlineExtension.status === "rejected" ? "已拒绝" : "待审批"} · 新完成时限 {formatDate(deadlineExtension.proposed_completion_deadline_at)} · {deadlineExtension.request_reason}
        </p>
      ) : null}
      {task.status === "escalated" && escalation ? <EscalationAdvice escalation={escalation} /> : null}
      <div className={styles.taskButtons}>
        {task.status === "pending_approval" ? (
          <>
            <button type="button" disabled={busy} onClick={() => {
              const reason = window.prompt("请输入批准理由。批准后载荷将冻结并写入模拟下发 Outbox：");
              if (reason?.trim()) void run("批准任务", () => responseWorkflowApi.decideTask(task.task_id, role, "approved", reason.trim()));
            }}>批准并模拟下发</button>
            <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => {
              const reason = window.prompt("请输入退回修改理由：");
              if (reason?.trim()) void run("退回任务", () => responseWorkflowApi.decideTask(task.task_id, role, "rejected", reason.trim()));
            }}>退回修改</button>
          </>
        ) : null}
        {primaryAction ? <button type="button" disabled={busy} onClick={() => void run(primaryAction.label, primaryAction.action)}>{primaryAction.label}</button> : null}
        {task.status === "in_progress" ? (
          <>
            <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => void run("报告部分完成", () => responseWorkflowApi.reportPartialCompletion(task.task_id, role))}>报告部分完成</button>
            <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => void run("报告资源不足", () => responseWorkflowApi.reportResourceShortage(task.task_id, role))}>报告资源不足</button>
          </>
        ) : null}
        {["issued", "acknowledged", "in_progress", "blocked", "escalated", "partially_completed"].includes(task.status) ? (
          <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => {
            const reason = window.prompt("请输入延期原因；系统将提交给独立审批岗位：");
            if (!reason?.trim()) return;
            const completion = new Date(task.effective_completion_deadline_at ?? task.deadline_at);
            completion.setHours(completion.getHours() + 1);
            const verification = new Date(task.effective_verification_deadline_at ?? task.verification_deadline_at ?? task.deadline_at);
            verification.setHours(verification.getHours() + 1);
            void run("申请延期", () => responseWorkflowApi.requestDeadlineExtension(task.task_id, role, completion.toISOString(), verification.toISOString(), reason.trim()));
          }}>申请延期</button>
        ) : null}
        {["issued", "acknowledged", "in_progress", "blocked", "escalated", "partially_completed"].includes(task.status) && ["reviewer", "commander"].includes(role) ? (
          <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => {
            const reason = window.prompt("请输入人工接管原因：");
            if (reason?.trim()) void run("人工接管", () => responseWorkflowApi.takeOverTask(task.task_id, role, reason.trim()));
          }}>人工接管</button>
        ) : null}
        {!["completed", "waived", "cancelled"].includes(task.status) && role === "commander" ? (
          <button type="button" className={styles.dangerButton} disabled={busy} onClick={() => {
            const reason = window.prompt("危险操作：请输入撤回任务理由。此操作会写入不可变审计台账：");
            if (reason?.trim()) void run("撤回任务", () => responseWorkflowApi.cancelTask(task.task_id, role, reason.trim()));
          }}>撤回任务</button>
        ) : null}
        {task.status === "pending_verification" ? <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => void run("退回补证", () => responseWorkflowApi.verifyTask(task.task_id, role, false))}>退回补证</button> : null}
      </div>
    </article>
  );
}

function RuleGate({ evaluation }: { evaluation: RuleEvaluationRecord }) {
  return (
    <section className={styles.ruleGate} aria-label="规则校验结果">
      <header>
        <strong>规则校验 · {evaluation.rule_set_version}</strong>
        <span data-outcome={evaluation.overall_outcome}>{evaluation.overall_outcome}</span>
      </header>
      <ul>
        {evaluation.checks.map((check) => (
          <li key={check.rule_id}><b>{check.outcome}</b><span>{check.message}</span></li>
        ))}
      </ul>
    </section>
  );
}

function EvidenceWorkbench({ packages, role, busy, run }: { packages: EvidencePackageVersion[]; role: WorkflowRole; busy: boolean; run: (label: string, operation: () => Promise<unknown>) => Promise<void> }) {
  const [manualSourceId, setManualSourceId] = useState("");
  const [manualExcerpt, setManualExcerpt] = useState("");
  const [manualRole, setManualRole] = useState<EvidenceRole>("procedure");
  const latest = packages[packages.length - 1];
  if (!latest) {
    return (
      <section className={styles.evidenceWorkbench} aria-labelledby="evidence-workbench-heading">
        <header><div><h3 id="evidence-workbench-heading">证据工作台</h3><p>确认风险对象后生成 FRC-RAG 字段证据包。</p></div><span>尚无证据包</span></header>
      </section>
    );
  }
  return (
    <section className={styles.evidenceWorkbench} aria-labelledby="evidence-workbench-heading">
      <header>
        <div><h3 id="evidence-workbench-heading">证据工作台</h3><p>字段状态、冲突和缺失项与任务草案分离保存。</p></div>
        <span>{latest.retrieval_strategy} · {latest.retrieval_mode} · {packages.length} 个版本</span>
      </header>
      <div className={styles.fieldStateList}>
        {Object.entries(latest.field_states).map(([field, state]) => (
          <div key={field}>
            <span>{field}</span>
            <strong data-state={state}>{state === "SUPPORTED" ? "有支持" : state === "CONFLICTED" ? "有冲突" : "缺失"}</strong>
          </div>
        ))}
      </div>
      <div className={styles.retrievalTrace}>
        <span>Baseline {latest.baseline_source_ids.length}</span>
        <span>FRC {latest.frc_source_ids.length}</span>
        <span>重合 {latest.shadow_comparison.overlap?.length ?? 0}</span>
        <span>仅基线 {latest.shadow_comparison.baseline_only?.length ?? 0}</span>
        <span>仅 FRC {latest.shadow_comparison.frc_only?.length ?? 0}</span>
      </div>
      <footer>
        <span>{latest.package_id} · V{latest.version} · {latest.status}</span>
        <span>{latest.conflicts.length} 个冲突 · {latest.missing_fields.length} 个缺失字段 · 哈希 {latest.content_hash.slice(0, 12)}</span>
      </footer>
      {latest.conflicts.length ? (
        <div className={styles.conflictList}>
          {latest.conflicts.map((conflict) => (
            <article key={conflict.conflict_id}>
              <div><strong>{conflict.field_name} · {conflict.conflict_type}</strong><span>{conflict.severity} · {conflict.resolution_status}</span></div>
              <p>冲突来源：{conflict.evidence_source_ids.join("、")}</p>
              {conflict.resolution_status !== "resolved" && ["reviewer", "commander"].includes(role) ? (
                <button type="button" disabled={busy} onClick={() => {
                  const selected = window.prompt(`请输入采用的来源 ID：${conflict.evidence_source_ids.join(" / ")}`, conflict.evidence_source_ids[0]);
                  if (!selected || !conflict.evidence_source_ids.includes(selected)) return;
                  const reason = window.prompt("请输入冲突裁决理由：");
                  if (reason?.trim()) void run("裁决证据冲突", () => responseWorkflowApi.resolveEvidenceConflict(latest.package_id, conflict.conflict_id, selected, role, reason.trim()));
                }}>人工裁决</button>
              ) : null}
            </article>
          ))}
        </div>
      ) : null}
      {latest.status !== "frozen" && ["reviewer", "commander", "admin"].includes(role) ? (
        <form className={styles.manualEvidenceForm} onSubmit={(event) => {
          event.preventDefault();
          if (!manualSourceId.trim() || manualExcerpt.trim().length < 8) return;
          const reason = `人工补证：${manualSourceId.trim()}`;
          void run("补充人工核验证据", () => responseWorkflowApi.supplementEvidence(
            latest.package_id,
            manualSourceId.trim(),
            manualExcerpt.trim(),
            manualRole,
            role,
            reason,
          ));
          setManualSourceId("");
          setManualExcerpt("");
        }}>
          <label><span>来源 ID</span><input value={manualSourceId} onChange={(event) => setManualSourceId(event.target.value)} placeholder="文号或受控来源 ID" disabled={busy} /></label>
          <label><span>证据角色</span><select value={manualRole} onChange={(event) => setManualRole(event.target.value as EvidenceRole)} disabled={busy}>{Object.entries(evidenceRoleText).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className={styles.manualEvidenceExcerpt}><span>核验原文</span><textarea value={manualExcerpt} onChange={(event) => setManualExcerpt(event.target.value)} placeholder="粘贴已核验的最小直接证据片段。" disabled={busy} /></label>
          <button type="submit" disabled={busy || manualExcerpt.trim().length < 8}>保存补证版本</button>
        </form>
      ) : null}
      {latest.status !== "frozen" && latest.missing_fields.length === 0 && latest.conflicts.every((item) => item.resolution_status === "resolved") && ["reviewer", "commander", "admin"].includes(role) ? (
        <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => {
          const reason = window.prompt("冻结后证据包不可修改。请输入独立复核理由：");
          if (reason?.trim()) void run("冻结证据包", () => responseWorkflowApi.freezeEvidencePackage(latest.package_id, role, reason.trim()));
        }}>冻结证据包</button>
      ) : null}
    </section>
  );
}

function DocumentRegistry({ documents, role, busy, run }: { documents: DocumentVersionRecord[]; role: WorkflowRole; busy: boolean; run: (label: string, operation: () => Promise<unknown>) => Promise<void> }) {
  const [title, setTitle] = useState("");
  const [version, setVersion] = useState("");
  const [content, setContent] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [ocrText, setOcrText] = useState("");
  const latest = [...documents].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 6);
  const lifecycleText: Record<DocumentVersionRecord["lifecycle_status"], string> = {
    active: "兼容有效",
    draft: "待解析",
    parsed: "待发布",
    published: "已发布",
    superseded: "已替代",
    retired: "已退役",
    parse_failed: "解析失败",
    index_failed: "索引失败",
  };
  const canGovern = role === "admin" || role === "reviewer";
  return (
    <section className={styles.documentRegistry} aria-labelledby="document-registry-heading">
      <header>
        <div><h3 id="document-registry-heading">文档与规则中心</h3><p>预案按来源哈希、版本、条款与索引版本不可变登记；替代版本不会继续参与当前检索。</p></div>
        <span>{documents.length} 个版本</span>
      </header>
      <div className={styles.documentList}>
        {latest.map((item) => (
          <article key={item.version_id}>
            <div><strong>{item.title}</strong><span>{item.version_label} · {lifecycleText[item.lifecycle_status]}</span></div>
            <p>{item.issuer} · {item.jurisdiction} · {item.source_filename} · {item.clauses.length} 条</p>
            <small>{item.parser_version ?? "未解析"} · {item.index_version || "未建索引"} · {item.source_hash.slice(0, 12)} · {item.is_simulated ? "模拟" : "正式"}</small>
            {item.clauses[0]?.source_locator ? <small>原文定位：{item.clauses[0].source_locator}</small> : null}
            {canGovern ? (
              <div className={styles.documentActions}>
                {item.lifecycle_status === "draft" || item.lifecycle_status === "parse_failed" ? (
                  <button type="button" disabled={busy} onClick={() => void run("解析文档版本", () => responseWorkflowApi.parseDocumentVersion(item.version_id, role, item.version_number))}>解析条款</button>
                ) : null}
                {item.lifecycle_status === "parsed" || item.lifecycle_status === "index_failed" ? (
                  <button type="button" disabled={busy} onClick={() => void run("发布文档版本", () => responseWorkflowApi.publishDocumentVersion(item.version_id, role, item.version_number))}>复核并发布</button>
                ) : null}
                {item.lifecycle_status === "published" ? (
                  <>
                    <button type="button" disabled={busy} onClick={() => void run("重建文档索引", () => responseWorkflowApi.rebuildDocumentIndex(item.version_id, role))}>重建索引</button>
                    <button type="button" disabled={busy} onClick={() => {
                      const reason = window.prompt("退役后该版本不得进入当前证据集合。请输入退役理由：");
                      if (reason?.trim()) void run("退役文档版本", () => responseWorkflowApi.retireDocumentVersion(item.version_id, role, item.version_number, reason.trim()));
                    }}>退役</button>
                  </>
                ) : null}
              </div>
            ) : null}
          </article>
        ))}
        {!latest.length ? <p className={styles.emptyText}>尚未登记文档版本。</p> : null}
      </div>
      {role === "admin" ? (
        <form className={styles.documentForm} onSubmit={(event) => {
          event.preventDefault();
          if (!title.trim() || !version.trim() || (!sourceFile && content.trim().length < 8)) return;
          const documentId = `SIM-DOC-${title.trim().replace(/\s+/g, "-").toUpperCase()}`;
          const replacement = documents.filter((item) => item.document_id === documentId).sort((a, b) => b.version_number - a.version_number)[0];
          void run("登记文档源文件", async () => {
            const prepared = sourceFile ? await prepareDocumentFile(sourceFile) : undefined;
            if (prepared?.scanned && ocrText.trim().length < 8) {
              throw new Error("扫描图片需要填写经人工核验的 OCR 文本");
            }
            return responseWorkflowApi.createDocumentVersion({
              documentId,
              title: title.trim(),
              versionLabel: version.trim(),
              issuer: "模拟区防办",
              jurisdiction: "district-simulation",
              effectiveAt: new Date().toISOString(),
              content: prepared ? undefined : content.trim(),
              file: prepared,
              ocrText: ocrText.trim() || undefined,
              ocrEngineVersion: ocrText.trim() ? "human-verified-ocr-v1" : undefined,
              replacesVersionId: replacement?.version_id,
              operatorRole: role,
            });
          });
          setTitle("");
          setVersion("");
          setContent("");
          setSourceFile(null);
          setOcrText("");
        }}>
          <label><span>文档名称</span><input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="例如：下穿通道响应规程" disabled={busy} /></label>
          <label><span>版本</span><input value={version} onChange={(event) => setVersion(event.target.value)} placeholder="2026-A" disabled={busy} /></label>
          <label><span>源文件</span><input type="file" accept=".pdf,.docx,.xlsx,.csv,.txt,.md,.png,.jpg,.jpeg,.tif,.tiff" onChange={(event) => setSourceFile(event.target.files?.[0] ?? null)} disabled={busy} /></label>
          <label className={styles.documentContent}><span>{sourceFile ? "OCR 文本（扫描件必填）" : "条款原文"}</span><textarea value={sourceFile ? ocrText : content} onChange={(event) => sourceFile ? setOcrText(event.target.value) : setContent(event.target.value)} placeholder={sourceFile ? "扫描件需粘贴经人工核验的 OCR 文本；数字文档可留空。" : "每个自然段形成可定位条款。"} disabled={busy} /></label>
          <button type="submit" disabled={busy || (!sourceFile && content.trim().length < 8)}>登记草稿</button>
        </form>
      ) : null}
    </section>
  );
}

function latestEscalation(escalations: EscalationRecord[], taskId: string) {
  const matches = escalations.filter((item) => item.task_id === taskId);
  return matches[matches.length - 1];
}

function EscalationAdvice({ escalation }: { escalation: EscalationRecord }) {
  return (
    <section className={styles.escalationAdvice} aria-label="异常升级与备选方案">
      <header><strong>异常已升级</strong><span>目标岗位：{roleText[escalation.target_role as WorkflowRole] ?? escalation.target_role}</span></header>
      <p>{escalation.reason}</p>
      <ol>{escalation.recommended_actions.map((item) => <li key={item}>{item}</li>)}</ol>
    </section>
  );
}

function ReviewDraftSection({ review }: { review: EventReviewDraft }) {
  return (
    <section className={styles.reviewSection} aria-labelledby="review-draft-heading">
      <header>
        <div><h3 id="review-draft-heading">{review.headline}</h3><p>{review.executive_summary}</p></div>
        <span>系统草稿 · {formatDate(review.created_at)}</span>
      </header>
      <div className={styles.reviewColumns}>
        <div><h4>过程与证据</h4><ul>{review.evidence_findings.map((item) => <li key={item}>{item}</li>)}</ul></div>
        <div><h4>延误与升级</h4><ul>{review.delays_and_escalations.length ? review.delays_and_escalations.map((item) => <li key={item}>{item}</li>) : <li>本次事件未记录异常升级。</li>}</ul></div>
        <div><h4>改进建议</h4><ul>{review.improvement_recommendations.map((item) => <li key={item}>{item}</li>)}</ul></div>
      </div>
      <footer>引用 {review.source_timeline_entry_ids.length} 条事件台账记录，复盘草稿需由授权人员复核后归档。</footer>
    </section>
  );
}

function ScenarioReportSection({ report }: { report: DistrictScenarioReport }) {
  const passedCount = report.acceptance_checks.filter((item) => item.passed).length;
  const statusLabel = report.overall_status === "failed" ? "未通过" : report.overall_status === "passed" ? "通过" : "通过，有观察项";
  return (
    <section className={styles.scenarioReport} aria-labelledby="scenario-report-heading">
      <header>
        <div>
          <h3 id="scenario-report-heading">{report.scenario_name}</h3>
          <p>{report.baseline_note}</p>
        </div>
        <div className={styles.reportStatus}>
          <strong className={report.overall_status === "failed" ? styles.reportFailed : styles.reportPassed}>{statusLabel}</strong>
          <span>{passedCount}/10 项验收 · {formatDate(report.created_at)}</span>
        </div>
      </header>

      <div className={styles.metricTableWrap}>
        <table className={styles.metricTable}>
          <caption>人工流程参考基线与系统台账实测对照</caption>
          <thead><tr><th>类别</th><th>指标</th><th>人工参考</th><th>系统实测</th><th>结论</th></tr></thead>
          <tbody>
            {report.metrics.map((metric) => (
              <tr key={metric.metric_id}>
                <td>{metric.category}</td><th scope="row">{metric.label}</th>
                <td>{formatMetric(metric.manual_value, metric.unit)}</td>
                <td>{metric.status === "not_evaluable" ? "数据不足" : formatMetric(metric.system_value, metric.unit)}</td>
                <td>{metric.interpretation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={styles.reportDetails}>
        <section aria-labelledby="acceptance-heading">
          <h4 id="acceptance-heading">V3 验收证据</h4>
          <ol className={styles.acceptanceList}>
            {report.acceptance_checks.map((check) => (
              <li key={check.check_id}>
                <span className={check.passed ? styles.checkPassed : styles.checkFailed}>{check.passed ? "通过" : "未通过"}</span>
                <div><strong>{check.check_id} · {check.requirement}</strong><p>{check.detail} · {check.evidence_refs.length} 条证据引用</p></div>
              </li>
            ))}
          </ol>
        </section>
        <section aria-labelledby="failure-heading">
          <h4 id="failure-heading">失败案例与观察项</h4>
          {report.failure_cases.length ? (
            <ul className={styles.failureList}>{report.failure_cases.map((item) => (
              <li key={item.failure_id}><strong>{item.severity === "blocking" ? "阻断" : "已处置"} · {item.trigger}</strong><p>{item.observed}</p><span>{item.recommendation}</span></li>
            ))}</ul>
          ) : <p className={styles.reportEmpty}>本次场景未发现业务闭环失败案例。</p>}
          {report.observations.length ? <ul className={styles.observationList}>{report.observations.map((item) => <li key={item}>{item}</li>)}</ul> : null}
        </section>
      </div>
      <footer>报告 {report.report_id} · 区县 {report.scope.area_id} · {report.scope.operator_roles.length} 个岗位 · {report.scope.task_count} 项任务</footer>
    </section>
  );
}

function formatMetric(value: number | null | undefined, unit: string) {
  if (value === null || value === undefined) return "—";
  return `${Number.isInteger(value) ? value : value.toFixed(2)}${unit === "%" ? "%" : ` ${unit}`}`;
}

function EvidenceGate({ task }: { task: ResponseTask }) {
  const roles = Object.keys(evidenceRoleText) as EvidenceRole[];
  const coveredCount = roles.filter((role) => task.evidence_role_coverage?.[role]).length;
  const complete = coveredCount === roles.length;

  return (
    <section className={styles.evidenceGate} aria-label="任务证据角色校验">
      <header>
        <div><strong>任务证据校验</strong><span>{task.grounding_summary || "该草案尚未记录证据覆盖摘要。"}</span></div>
        <b className={complete ? styles.gatePassed : styles.gateIncomplete}>{coveredCount}/6</b>
      </header>
      <div className={styles.roleCoverage}>
        {roles.map((role) => (
          <span key={role} className={task.evidence_role_coverage?.[role] ? styles.roleCovered : styles.roleMissing}>
            {task.evidence_role_coverage?.[role] ? "已覆盖" : "缺失"} · {evidenceRoleText[role]}
          </span>
        ))}
      </div>
      {task.source_evidence?.length ? (
        <details>
          <summary>查看 {task.source_evidence.length} 条来源证据</summary>
          <ul>
            {task.source_evidence.map((item) => (
              <li key={`${item.source_type}-${item.source_id}`}>
                <strong>{item.title}</strong>
                <span>{item.roles.map((role) => evidenceRoleText[role]).join("、")}{item.clause ? ` · ${item.clause}` : ""}{item.document_version ? ` · ${item.document_version}` : ""}</span>
                <p>{item.excerpt}</p>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
