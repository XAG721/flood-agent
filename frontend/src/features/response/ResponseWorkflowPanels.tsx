import { useEffect, useState } from "react";
import { responseWorkflowApi } from "../../api/responseWorkflowApi";
import type {
  DeadlineExtensionRecord,
  DispatchCallbackRecord,
  DistrictScenarioReport,
  DocumentVersionRecord,
  EscalationRecord,
  EvidencePackageVersion,
  EvidenceRole,
  EventReviewDraft,
  OutboxMessage,
  ResponseTask,
  RuleEvaluationRecord,
  SimulationDispatchScenario,
  WorkflowRole,
} from "../../types/response";
import { prepareDocumentFile } from "./filePreparation";
import {
  dispatchScenarioText,
  evidenceRoleText,
  formatDate,
  missingReasonText,
  roleText,
  statusText,
  taskFieldText,
} from "./presentation";
import styles from "../../pages/response-workflow-page.module.css";

export function TaskDetail({ task, escalation, evidencePackage, outboxMessage, ruleEvaluation, deadlineExtension, role, busy, run }: { task: ResponseTask; escalation?: EscalationRecord; evidencePackage?: EvidencePackageVersion; outboxMessage?: OutboxMessage; ruleEvaluation?: RuleEvaluationRecord; deadlineExtension?: DeadlineExtensionRecord; role: WorkflowRole; busy: boolean; run: (label: string, operation: () => Promise<unknown>) => Promise<void> }) {
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

export function EvidenceWorkbench({ packages, role, busy, run }: { packages: EvidencePackageVersion[]; role: WorkflowRole; busy: boolean; run: (label: string, operation: () => Promise<unknown>) => Promise<void> }) {
  const [manualSourceId, setManualSourceId] = useState("");
  const [manualExcerpt, setManualExcerpt] = useState("");
  const [manualRole, setManualRole] = useState<EvidenceRole>("procedure");
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const latest = packages[packages.length - 1];
  if (!latest) {
    return (
      <section className={styles.evidenceWorkbench} aria-labelledby="evidence-workbench-heading">
        <header><div><h3 id="evidence-workbench-heading">证据工作台</h3><p>确认风险对象后生成 FRC-RAG 字段证据包。</p></div><span>尚无证据包</span></header>
      </section>
    );
  }
  const packageVersions = packages.filter((item) => item.package_id === latest.package_id);
  const previous = packageVersions.length > 1 ? packageVersions[packageVersions.length - 2] : null;
  const selectedEvidence = latest.evidence.find((item) => item.source_id === selectedSourceId) ?? null;
  const changedFields = previous
    ? Object.keys(latest.field_states).filter((field) => previous.field_states[field] !== latest.field_states[field])
    : [];
  const previousSources = new Set(previous?.evidence.map((item) => item.source_id) ?? []);
  const latestSources = new Set(latest.evidence.map((item) => item.source_id));
  const addedSources = previous ? [...latestSources].filter((sourceId) => !previousSources.has(sourceId)) : [];
  const removedSources = previous ? [...previousSources].filter((sourceId) => !latestSources.has(sourceId)) : [];
  const unresolvedConflicts = latest.conflicts.filter((item) => item.resolution_status !== "resolved");
  return (
    <section className={styles.evidenceWorkbench} aria-labelledby="evidence-workbench-heading">
      <header>
        <div><h3 id="evidence-workbench-heading">证据工作台</h3><p>字段状态、冲突和缺失项与任务草案分离保存。</p></div>
        <span>{latest.retrieval_strategy} · {latest.retrieval_mode} · {packageVersions.length} 个版本</span>
      </header>
      <div className={styles.fieldStateList}>
        {Object.entries(latest.field_states).map(([field, state]) => (
          <div key={field}>
            <span>{taskFieldText[field] ?? field}</span>
            <span>
              <strong data-state={state}>{state === "SUPPORTED" ? "有支持" : state === "CONFLICTED" ? "有冲突" : "缺失"}</strong>
              {state === "MISSING" ? <small>{missingReasonText[latest.missing_reasons[field] ?? ""] ?? "原因待核验"}</small> : null}
            </span>
          </div>
        ))}
      </div>
      <div className={styles.retrievalTrace}>
        <span>Baseline {latest.baseline_source_ids.length}</span>
        <span>FRC {latest.frc_source_ids.length}</span>
        <span>重合 {latest.shadow_comparison.overlap?.length ?? 0}</span>
        <span>仅基线 {latest.shadow_comparison.baseline_only?.length ?? 0}</span>
        <span>仅 FRC {latest.shadow_comparison.frc_only?.length ?? 0}</span>
        <span>NLI {latest.nli_status} · {latest.nli_model_version}</span>
        <span>语义评估 {latest.nli_assessments.length}</span>
      </div>
      {previous ? (
        <div className={styles.evidenceVersionDiff} aria-label="证据包版本差异">
          <strong>V{previous.version} → V{latest.version}</strong>
          <span>字段状态变化 {changedFields.length}</span>
          <span>新增来源 {addedSources.length}</span>
          <span>移除来源 {removedSources.length}</span>
          <span>冲突 {previous.conflicts.length} → {latest.conflicts.length}</span>
        </div>
      ) : null}
      <div className={styles.evidenceSourceGrid} aria-label="证据来源列表">
        {latest.evidence.map((item) => (
          <article key={`${item.source_type}:${item.source_id}`}>
            <header>
              <strong>{item.title}</strong>
              <span>{item.document_version ?? "无版本"}{item.clause ? ` · ${item.clause}` : ""}</span>
            </header>
            <p>{item.excerpt}</p>
            <div>
              {item.roles.map((itemRole) => <span key={itemRole}>{evidenceRoleText[itemRole]}</span>)}
              {Object.keys(item.field_support).map((field) => <span key={field}>{taskFieldText[field] ?? field}</span>)}
            </div>
            <button type="button" onClick={() => setSelectedSourceId(item.source_id)}>查看原文定位</button>
          </article>
        ))}
      </div>
      {selectedEvidence ? (
        <aside className={styles.evidenceSourceDetail} aria-label="证据原文定位">
          <header><strong>{selectedEvidence.title}</strong><button type="button" onClick={() => setSelectedSourceId(null)}>关闭</button></header>
          <dl>
            <div><dt>定位</dt><dd>{selectedEvidence.source_locator}</dd></div>
            <div><dt>版本/条款</dt><dd>{selectedEvidence.document_version ?? "—"} / {selectedEvidence.clause ?? "—"}</dd></div>
            <div><dt>页码/表格</dt><dd>{selectedEvidence.page_number ?? "—"} / {selectedEvidence.table_name ?? "—"}{selectedEvidence.row_start ? ` 第 ${selectedEvidence.row_start}-${selectedEvidence.row_end ?? selectedEvidence.row_start} 行` : ""}</dd></div>
          </dl>
          <blockquote>{selectedEvidence.excerpt}</blockquote>
        </aside>
      ) : null}
      <footer>
        <span>{latest.package_id} · V{latest.version} · {latest.status}</span>
        <span>{unresolvedConflicts.length} 个未决冲突 · {latest.blocking_missing_fields.length} 个关键缺失 / {latest.missing_fields.length} 个全部缺失 · 哈希 {latest.content_hash.slice(0, 12)}</span>
      </footer>
      {latest.conflicts.length ? (
        <div className={styles.conflictList}>
          {latest.conflicts.map((conflict) => (
            <article key={conflict.conflict_id}>
              <div><strong>{taskFieldText[conflict.field_name] ?? conflict.field_name} · {conflict.conflict_type}</strong><span>{conflict.severity} · {conflict.resolution_status}</span></div>
              <p>冲突来源：{conflict.evidence_source_ids.join("、")}</p>
              <p>检测：{conflict.detection_methods.join("、")} · 维度：{conflict.conflict_dimensions.join("、")}{conflict.nli_relation ? ` · NLI ${conflict.nli_relation} ${Math.round((conflict.nli_confidence ?? 0) * 100)}%` : ""}</p>
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
      {latest.status !== "frozen" && latest.blocking_missing_fields.length === 0 && latest.conflicts.every((item) => item.resolution_status === "resolved") && ["reviewer", "commander", "admin"].includes(role) ? (
        <button type="button" className={styles.secondaryButton} disabled={busy} onClick={() => {
          const reason = window.prompt("冻结后证据包不可修改。请输入独立复核理由：");
          if (reason?.trim()) void run("冻结证据包", () => responseWorkflowApi.freezeEvidencePackage(latest.package_id, role, reason.trim()));
        }}>冻结证据包</button>
      ) : null}
    </section>
  );
}

export function DocumentRegistry({ documents, role, busy, run }: { documents: DocumentVersionRecord[]; role: WorkflowRole; busy: boolean; run: (label: string, operation: () => Promise<unknown>) => Promise<void> }) {
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

export function latestEscalation(escalations: EscalationRecord[], taskId: string) {
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

export function ReviewDraftSection({ review }: { review: EventReviewDraft }) {
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

export function ScenarioReportSection({ report }: { report: DistrictScenarioReport }) {
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
