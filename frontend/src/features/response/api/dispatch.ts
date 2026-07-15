import { request } from "../../../lib/httpClient";
import type { DispatchCallbackRecord, OutboxMessage, SimulationDispatchScenario, WorkflowRole } from "../../../types/response";
import { actionPayload } from "./shared";

export const responseDispatchApi = {
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
};
