import { request } from "../../../lib/httpClient";
import type { CandidateObjectListVersion, RiskObjectRegistryImportResult, RiskObjectRegistryRecord, WorkflowRole } from "../../../types/response";
import { actionPayload } from "./shared";

export const responseRegistryApi = {
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
};
