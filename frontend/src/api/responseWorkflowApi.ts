import { request } from "../lib/httpClient";
import type { DispatchCallbackRecord, DistrictScenarioReport, DocumentVersionRecord, EventDashboard, OutboxMessage, ResponseEvent, RetrievalMode, SimulationDispatchScenario, TaskDraftGenerationResult, WorkflowRole } from "../types/response";

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
  generateTaskDraft(eventId: string, objectId: string, operatorRole: WorkflowRole, retrievalMode: RetrievalMode): Promise<TaskDraftGenerationResult> {
    return request(`/response/events/${eventId}/risk-objects/${objectId}/task-draft`, {
      method: "POST",
      body: JSON.stringify({
        ...actionPayload(operatorRole, "基于当前预警、对象台账和有效预案生成结构化任务草案"),
        retrieval_mode: retrievalMode,
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
