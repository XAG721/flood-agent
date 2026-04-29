import { request } from "../lib/httpClient";
import type { ArchiveStatusView, AuditRecord, OperationalAlert } from "../types/api";

interface ReliabilityQuery {
  eventId?: string;
  severity?: string;
  sourceType?: string;
  fromTs?: string;
  toTs?: string;
  limit?: number;
}

function buildReliabilityQuery(payload?: ReliabilityQuery): string {
  const params = new URLSearchParams();
  if (payload?.eventId) params.set("event_id", payload.eventId);
  if (payload?.severity) params.set("severity", payload.severity);
  if (payload?.sourceType) params.set("source_type", payload.sourceType);
  if (payload?.fromTs) params.set("from_ts", payload.fromTs);
  if (payload?.toTs) params.set("to_ts", payload.toTs);
  if (payload?.limit) params.set("limit", String(payload.limit));
  return params.size ? `?${params.toString()}` : "";
}

export const reliabilityApi = {
  listV2Alerts(payload?: ReliabilityQuery): Promise<OperationalAlert[]> {
    return request(`/platform/alerts${buildReliabilityQuery(payload)}`, { method: "GET" });
  },

  listV2AuditRecords(payload?: ReliabilityQuery): Promise<AuditRecord[]> {
    return request(`/platform/audit/records${buildReliabilityQuery(payload)}`, { method: "GET" });
  },

  getV2ArchiveStatus(): Promise<ArchiveStatusView> {
    return request("/platform/archive/status", { method: "GET" });
  },

  runV2ArchiveCycle(): Promise<ArchiveStatusView> {
    return request("/platform/archive/run", { method: "POST" });
  },
};
