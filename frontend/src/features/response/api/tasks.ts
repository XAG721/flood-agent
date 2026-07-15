import { request } from "../../../lib/httpClient";
import type { RetrievalMode, TaskDraftGenerationResult, WorkflowRole } from "../../../types/response";
import { actionPayload } from "./shared";

export const responseTaskApi = {
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
};
