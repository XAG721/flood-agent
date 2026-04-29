import { request } from "../lib/httpClient";
import type {
  AgentMetricsView,
  AgentStatusView,
  AgentTask,
  AgentTimelineEntry,
  DecisionReportView,
  EvaluationBenchmark,
  EvaluationReport,
  ExperienceContextView,
  MemoryBundleView,
  SharedMemorySnapshot,
  SupervisorLoopStatus,
  SupervisorRunRecord,
  TriggerEvent,
} from "../types/api";

export const agentRuntimeApi = {
  getV2AgentStatus(eventId: string): Promise<AgentStatusView> {
    return request(`/platform/events/${eventId}/agent-status`, { method: "GET" });
  },

  listV2AgentTasks(eventId: string): Promise<AgentTask[]> {
    return request(`/platform/events/${eventId}/agent-tasks`, { method: "GET" });
  },

  getV2SharedMemory(eventId: string): Promise<SharedMemorySnapshot> {
    return request(`/platform/events/${eventId}/shared-memory`, { method: "GET" });
  },

  getV2SessionMemory(sessionId: string): Promise<MemoryBundleView> {
    return request(`/platform/copilot/sessions/${sessionId}/memory`, { method: "GET" });
  },

  listV2TriggerEvents(eventId: string): Promise<TriggerEvent[]> {
    return request(`/platform/events/${eventId}/trigger-events`, { method: "GET" });
  },

  listV2AgentTimeline(eventId: string): Promise<AgentTimelineEntry[]> {
    return request(`/platform/events/${eventId}/agent-timeline`, { method: "GET" });
  },

  getV2ExperienceContext(eventId: string): Promise<ExperienceContextView> {
    return request(`/platform/events/${eventId}/experience-context`, { method: "GET" });
  },

  getV2DecisionReport(eventId: string): Promise<DecisionReportView> {
    return request(`/platform/events/${eventId}/decision-report`, { method: "GET" });
  },

  getV2AgentMetrics(): Promise<AgentMetricsView> {
    return request("/platform/agent-metrics", { method: "GET" });
  },

  listV2EvaluationBenchmarks(): Promise<EvaluationBenchmark[]> {
    return request("/platform/evaluation/benchmarks", { method: "GET" });
  },

  runV2Evaluation(): Promise<EvaluationReport> {
    return request("/platform/evaluation/run", { method: "POST" });
  },

  getV2EvaluationReport(reportId: string): Promise<EvaluationReport> {
    return request(`/platform/evaluation/reports/${reportId}`, { method: "GET" });
  },

  replayV2EvaluationReport(reportId: string): Promise<EvaluationReport> {
    return request(`/platform/evaluation/reports/${reportId}/replay`, { method: "POST" });
  },

  replayV2AgentTask(taskId: string, replayReason = ""): Promise<AgentTask> {
    return request(`/platform/agent-tasks/${taskId}/replay`, {
      method: "POST",
      body: JSON.stringify({ replay_reason: replayReason }),
    });
  },

  listV2SupervisorRuns(eventId: string): Promise<SupervisorRunRecord[]> {
    return request(`/platform/events/${eventId}/supervisor-runs`, { method: "GET" });
  },

  getV2SupervisorStatus(): Promise<SupervisorLoopStatus> {
    return request("/platform/supervisor/status", { method: "GET" });
  },

  runV2Supervisor(eventId: string): Promise<SupervisorRunRecord> {
    return request(`/platform/events/${eventId}/supervisor/run`, { method: "POST" });
  },

  tickV2Supervisor(eventId?: string): Promise<SupervisorRunRecord[]> {
    const suffix = eventId ? `?event_id=${encodeURIComponent(eventId)}` : "";
    return request(`/platform/supervisor/tick${suffix}`, { method: "POST" });
  },
};
