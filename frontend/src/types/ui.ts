export type ConsoleBootState = "booting" | "ready" | "degraded" | "error";

export type ExecutionStatus =
  | "idle"
  | "planning"
  | "awaiting_confirmation"
  | "running"
  | "success"
  | "error";
