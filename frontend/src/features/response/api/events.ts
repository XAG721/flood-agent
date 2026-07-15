import { request } from "../../../lib/httpClient";
import type { DistrictScenarioReport, EventDashboard, ResponseEvent, WorkflowRole } from "../../../types/response";
import { actionPayload } from "./shared";

export const responseEventApi = {
  listEvents(): Promise<ResponseEvent[]> {
    return request("/response/events", { method: "GET" });
  },
  getDashboard(eventId: string): Promise<EventDashboard> {
    return request(`/response/events/${eventId}`, { method: "GET" });
  },
  bootstrapDemo(): Promise<EventDashboard> {
    return request("/response/events/bootstrap-demo", { method: "POST" });
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
