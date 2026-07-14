import { request } from "../lib/httpClient";
import type { CandidateObjectListVersion, DispatchCallbackRecord, DistrictScenarioReport, DocumentVersionRecord, EventDashboard, IndexBuildRecord, OutboxMessage, ResponseEvent, RetrievalMode, RiskObjectRegistryImportResult, RiskObjectRegistryRecord, SimulationDispatchScenario, TaskDraftGenerationResult, WorkflowRole } from "../types/response";

const actionPayload = (operatorRole: WorkflowRole, note = "") => ({
  operator_id: `console_${operatorRole}`,
  operator_role: operatorRole,
  terminal_id: "district-response-console",
  note,
});

export const responseWorkflowApi = {
  listEvents(): Promise<ResponseEvent[]> {
    return request("/response/events", { method: "GET" });
  },
  listDocuments(): Promise<DocumentVersionRecord[]> {
    return request("/response/documents", { method: "GET" });
  },
  listRiskObjectRegistry(areaId: string): Promise<RiskObjectRegistryRecord[]> {
    const query = new URLSearchParams({ area_id: areaId });
    return request(`/response/risk-objects?${query.toString()}`, { method: "GET" });
  },
  importRiskObjectFile(input: {
    areaId: string;
    sourceVersion: string;
    filename: string;
    mediaType: string;
    contentBase64: string;
    sha256: string;
    operatorRole: WorkflowRole;
  }): Promise<RiskObjectRegistryImportResult> {
    return request("/response/risk-objects/file-imports", {
      method: "POST",
      body: JSON.stringify({
        area_id: input.areaId,
        source_version: input.sourceVersion,
        file: {
          filename: input.filename,
          media_type: input.mediaType,
          content_base64: input.contentBase64,
          sha256: input.sha256,
        },
        ...actionPayload(input.operatorRole, "导入受控风险对象主数据文件"),
      }),
    });
  },
  registerDocument(input: {
    document_id: string;
    title: string;
    version_label: string;
    issuer: string;
    jurisdiction: string;
    effective_at: string;
    content: string;
    replaces_version_id?: string;
  }): Promise<DocumentVersionRecord> {
    return request("/response/documents", {
      method: "POST",
      body: JSON.stringify({
        ...input,
        ...actionPayload("admin", "登记并索引受控模拟预案版本"),
      }),
    });
  },
  createDocumentVersion(input: {
    documentId: string;
    title: string;
    versionLabel: string;
    issuer: string;
    jurisdiction: string;
    effectiveAt: string;
    expiresAt?: string;
    content?: string;
    file?: { filename: string; mediaType: string; contentBase64: string; sha256: string };
    ocrText?: string;
    ocrEngineVersion?: string;
    replacesVersionId?: string;
    operatorRole: WorkflowRole;
  }): Promise<DocumentVersionRecord> {
    return request(`/response/documents/${encodeURIComponent(input.documentId)}/versions`, {
      method: "POST",
      body: JSON.stringify({
        title: input.title,
        version_label: input.versionLabel,
        issuer: input.issuer,
        jurisdiction: input.jurisdiction,
        effective_at: input.effectiveAt,
        expires_at: input.expiresAt,
        content: input.content,
        file: input.file ? {
          filename: input.file.filename,
          media_type: input.file.mediaType,
          content_base64: input.file.contentBase64,
          sha256: input.file.sha256,
        } : undefined,
        ocr_text: input.ocrText,
        ocr_engine_version: input.ocrEngineVersion,
        replaces_version_id: input.replacesVersionId,
        auto_publish: false,
        is_simulated: true,
        ...actionPayload(input.operatorRole, "登记不可变文档源文件版本"),
      }),
    });
  },
  parseDocumentVersion(versionId: string, operatorRole: WorkflowRole, expectedVersion: number): Promise<DocumentVersionRecord> {
    return request(`/response/document-versions/${encodeURIComponent(versionId)}/parse`, {
      method: "POST",
      body: JSON.stringify({
        parser_version: "document-parser-v2",
        expected_version: expectedVersion,
        ...actionPayload(operatorRole, "解析页码、章节、条款和表格定位"),
      }),
    });
  },
  publishDocumentVersion(versionId: string, operatorRole: WorkflowRole, expectedVersion: number): Promise<DocumentVersionRecord> {
    return request(`/response/document-versions/${encodeURIComponent(versionId)}/publish`, {
      method: "POST",
      body: JSON.stringify({
        expected_version: expectedVersion,
        corpus_version: "district-policy-corpus-v1",
        embedding_version: "deterministic-hybrid-v1",
        ...actionPayload(operatorRole, "复核通过并发布当前文档版本"),
      }),
    });
  },
  retireDocumentVersion(versionId: string, operatorRole: WorkflowRole, expectedVersion: number, reason: string): Promise<DocumentVersionRecord> {
    return request(`/response/document-versions/${encodeURIComponent(versionId)}/retire`, {
      method: "POST",
      body: JSON.stringify({
        reason,
        expected_version: expectedVersion,
        ...actionPayload(operatorRole, reason),
      }),
    });
  },
  rebuildDocumentIndex(versionId: string, operatorRole: WorkflowRole): Promise<IndexBuildRecord> {
    return request("/response/index-builds", {
      method: "POST",
      body: JSON.stringify({
        document_version_id: versionId,
        ...actionPayload(operatorRole, "重建发布文档的可复现索引"),
      }),
    });
  },
  getDashboard(eventId: string): Promise<EventDashboard> {
    return request(`/response/events/${eventId}`, { method: "GET" });
  },
  bootstrapDemo(): Promise<EventDashboard> {
    return request("/response/events/bootstrap-demo", { method: "POST" });
  },
  discoverRiskObjects(eventId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/events/${eventId}/risk-objects/discover`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, "按事件区域关联对象登记台账，形成待人工核验候选清单"),
        min_risk_score: 45,
        max_candidates: 20,
      }),
    });
  },
  verifyRiskObject(eventId: string, objectId: string, operatorRole: WorkflowRole, decision: "confirmed" | "excluded"): Promise<unknown> {
    return request(`/response/events/${eventId}/risk-objects/${objectId}/verify`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, decision === "confirmed" ? "人工复核确认候选对象" : "人工复核排除候选对象"),
        decision,
      }),
    });
  },
  freezeCandidateObjectList(eventId: string, operatorRole: WorkflowRole, objectIds: string[], expectedVersion?: number): Promise<CandidateObjectListVersion> {
    return request(`/response/events/${eventId}/candidate-object-lists/freeze`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, "冻结本次人工确认对象清单，供后续任务成案绑定"),
        object_ids: objectIds,
        expected_version: expectedVersion,
      }),
    });
  },
  generateTaskDraft(eventId: string, objectId: string, operatorRole: WorkflowRole, retrievalMode: RetrievalMode, candidateObjectListVersion?: number): Promise<TaskDraftGenerationResult> {
    return request(`/response/events/${eventId}/risk-objects/${objectId}/task-draft`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, "基于当前预警、对象台账和有效预案生成结构化任务草案"),
        retrieval_mode: retrievalMode,
        candidate_object_list_version: candidateObjectListVersion,
      }),
    });
  },
  resolveEvidenceConflict(packageId: string, conflictId: string, selectedSourceId: string, operatorRole: WorkflowRole, reason: string): Promise<unknown> {
    return request(`/response/evidence-packages/${packageId}/conflicts/${conflictId}/resolve`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, reason),
        selected_source_ids: [selectedSourceId],
        reason,
      }),
    });
  },
  supplementEvidence(packageId: string, sourceId: string, excerpt: string, evidenceRole: string, operatorRole: WorkflowRole, reason: string): Promise<unknown> {
    return request(`/response/evidence-packages/${packageId}/manual-evidence`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, reason),
        reason,
        freeze_when_complete: false,
        evidence: [{
          source_type: "manual_verified_document",
          source_id: sourceId,
          title: `人工核验来源 ${sourceId}`,
          excerpt,
          roles: [evidenceRole],
          document_version: "manual-reviewed",
          clause: "manual-reference",
        }],
      }),
    });
  },
  freezeEvidencePackage(packageId: string, operatorRole: WorkflowRole, reason: string): Promise<unknown> {
    return request(`/response/evidence-packages/${packageId}/freeze`, {
      method: "POST",
      body: JSON.stringify({ ...actionPayload(operatorRole, reason), reason }),
    });
  },
  decideTask(taskId: string, operatorRole: WorkflowRole, decision: "approved" | "rejected", note = ""): Promise<unknown> {
    return request(`/response/tasks/${taskId}/decision`, {
      method: "POST",
      body: JSON.stringify({ ...actionPayload(operatorRole, note), decision }),
    });
  },
  acknowledgeTask(taskId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/tasks/${taskId}/acknowledge`, {
      method: "POST",
      body: JSON.stringify(actionPayload(operatorRole, "已确认接收任务")),
    });
  },
  assignTask(taskId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/tasks/${taskId}/assign`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, "由成员单位联络员完成现场执行分派"),
        assignee_id: "console_field_operator",
        assignee_name: "现场执行员",
        assignee_role: "field_operator",
        reason: "按对象责任和当前班次分派现场执行",
      }),
    });
  },
  startTask(taskId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/tasks/${taskId}/start`, {
      method: "POST",
      body: JSON.stringify(actionPayload(operatorRole, "开始现场执行")),
    });
  },
  completeTask(taskId: string, operatorRole: WorkflowRole, requiredEvidence: string[]): Promise<unknown> {
    return request(`/response/tasks/${taskId}/feedback`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole),
        summary: "现场处置已完成，提交值守端核实。",
        evidence: requiredEvidence.map((type) => ({ type, value: "演示核验记录", captured_at: new Date().toISOString() })),
      }),
    });
  },
  reportResourceShortage(taskId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/tasks/${taskId}/feedback`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole),
        summary: "现场资源不足，当前处置无法按原计划继续，请求协调支援。",
        evidence: [],
        resource_gap: "移动排水设备或现场支援人员不足",
      }),
    });
  },
  reportPartialCompletion(taskId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/tasks/${taskId}/feedback`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole),
        summary: "现场处置部分完成，剩余工作继续执行。",
        evidence: [],
        completion_percent: 50,
      }),
    });
  },
  requestDeadlineExtension(taskId: string, operatorRole: WorkflowRole, completionDeadline: string, verificationDeadline: string, reason: string): Promise<unknown> {
    return request(`/response/tasks/${taskId}/deadline-extensions`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, reason),
        proposed_completion_deadline_at: completionDeadline,
        proposed_verification_deadline_at: verificationDeadline,
        reason,
      }),
    });
  },
  cancelTask(taskId: string, operatorRole: WorkflowRole, reason: string): Promise<unknown> {
    return request(`/response/tasks/${taskId}/cancel`, {
      method: "POST",
      body: JSON.stringify(actionPayload(operatorRole, reason)),
    });
  },
  takeOverTask(taskId: string, operatorRole: WorkflowRole, reason: string): Promise<unknown> {
    return request(`/response/tasks/${taskId}/take-over`, {
      method: "POST",
      body: JSON.stringify(actionPayload(operatorRole, reason)),
    });
  },
  verifyTask(taskId: string, operatorRole: WorkflowRole, approved: boolean): Promise<unknown> {
    return request(`/response/tasks/${taskId}/verify-completion?approved=${String(approved)}`, {
      method: "POST",
      body: JSON.stringify(actionPayload(operatorRole, approved ? "反馈证据核验通过" : "证据不足，退回补充")),
    });
  },
  runDeadlineSweep(eventId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/events/${eventId}/deadline-sweep`, {
      method: "POST",
      body: JSON.stringify(actionPayload(operatorRole, "人工触发时限巡检")),
    });
  },
  processOutbox(messageId: string, scenario: SimulationDispatchScenario, operatorRole: WorkflowRole): Promise<OutboxMessage[]> {
    return request("/response/dispatch/outbox/process", {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, `运行受控模拟下发场景：${scenario}`),
        message_id: messageId,
        max_messages: 1,
        simulation_scenario: scenario,
      }),
    });
  },
  listDispatchCallbacks(messageId: string): Promise<DispatchCallbackRecord[]> {
    return request(`/response/dispatch/outbox/${messageId}/callbacks`, { method: "GET" });
  },
  closeEvent(eventId: string, operatorRole: WorkflowRole): Promise<unknown> {
    return request(`/response/events/${eventId}/close`, {
      method: "POST",
      body: JSON.stringify(actionPayload(operatorRole, "任务闭环和证据核验完成")),
    });
  },
  runScenarioEvaluation(eventId: string, operatorRole: WorkflowRole): Promise<DistrictScenarioReport> {
    return request(`/response/events/${eventId}/scenario-evaluation`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, "运行 V3 区县最小场景验收"),
        scenario_name: "碑林区下穿通道洪水响应仿真场景",
        scenario_type: "simulation",
      }),
    });
  },
};
