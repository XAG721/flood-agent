import type { WorkflowRole } from "../../../types/response";

export const actionPayload = (operatorRole: WorkflowRole, note = "") => ({
  operator_id: `console_${operatorRole}`,
  operator_role: operatorRole,
  terminal_id: "district-response-console",
  note,
});
