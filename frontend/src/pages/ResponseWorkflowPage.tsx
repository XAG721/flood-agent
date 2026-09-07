import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { responseWorkflowApi } from "../api/responseWorkflowApi";
import { setApiOperatorContext } from "../lib/httpClient";
import type { DocumentVersionRecord, EventDashboard, RetrievalMode, RiskObjectRegistryImportResult, RiskObjectRegistryRecord, WorkflowRole } from "../types/response";
import styles from "./response-workflow-page.module.css";
import {
  DocumentRegistry,
  EvidenceWorkbench,
  ReviewDraftSection,
  ScenarioReportSection,
  TaskDetail,
  latestEscalation,
} from "../features/response/ResponseWorkflowPanels";
import { prepareRegistryFile } from "../features/response/filePreparation";
import {
  actionText,
  formatDate,
  registryMapLayers,
  roleText,
  statusText,
} from "../features/response/presentation";

const DigitalTwinCesiumCanvas = lazy(() =>
  import("../components/DigitalTwinCesiumCanvas").then((module) => ({
    default: module.DigitalTwinCesiumCanvas,
  })),
);

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
