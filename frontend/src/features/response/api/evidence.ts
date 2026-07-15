import { request } from "../../../lib/httpClient";
import type { WorkflowRole } from "../../../types/response";
import { actionPayload } from "./shared";

export const responseEvidenceApi = {
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
          source_locator: `manual://verified/${sourceId}`,
          section_path: [],
          field_support: {},
          conflicts_with: [],
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
};
