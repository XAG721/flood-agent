import { request } from "../lib/httpClient";
import type { DatasetJobView, DatasetPipelineStatusView } from "../types/api";

export const datasetApi = {
  getV2DatasetStatus(): Promise<DatasetPipelineStatusView> {
    return request("/v2/admin/dataset/status", { method: "GET" });
  },

  getV2DatasetJobs(): Promise<DatasetJobView[]> {
    return request("/v2/admin/dataset/jobs", { method: "GET" });
  },

  fetchV2DatasetSources(payload?: { download?: boolean; sourceIds?: string[]; forceRefresh?: boolean }): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/fetch", {
      method: "POST",
      body: JSON.stringify({
        download: payload?.download ?? true,
        source_ids: payload?.sourceIds ?? [],
        force_refresh: payload?.forceRefresh ?? false,
      }),
    });
  },

  buildV2Dataset(payload?: { download?: boolean; syncDemoDb?: boolean }): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/build", {
      method: "POST",
      body: JSON.stringify({
        download: payload?.download ?? false,
        sync_demo_db: payload?.syncDemoDb ?? true,
      }),
    });
  },

  validateV2Dataset(): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/validate", { method: "POST", body: JSON.stringify({}) });
  },

  syncV2Dataset(payload?: { dbPath?: string }): Promise<DatasetJobView> {
    return request("/v2/admin/dataset/sync-demo-db", {
      method: "POST",
      body: JSON.stringify({ db_path: payload?.dbPath ?? null }),
    });
  },

  cancelV2DatasetJob(jobId: string): Promise<DatasetJobView> {
    return request(`/v2/admin/dataset/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
  },

  retryV2DatasetJob(jobId: string): Promise<DatasetJobView> {
    return request(`/v2/admin/dataset/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" });
  },
};
