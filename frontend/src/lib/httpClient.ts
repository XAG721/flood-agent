import type { OperatorRole } from "../types/api";

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

const operatorContext: { id: string; role: OperatorRole } = {
  id: "operator_console",
  role: "commander",
};

export function buildUrl(path: string): string {
  return `${apiBaseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

export function setApiOperatorContext(context: { id?: string; role: OperatorRole }) {
  operatorContext.id = context.id?.trim() || "operator_console";
  operatorContext.role = context.role;
}

export function getApiOperatorContext(): Readonly<{ id: string; role: OperatorRole }> {
  return operatorContext;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildUrl(path), {
    headers: {
      "Content-Type": "application/json",
      "X-Operator-Id": operatorContext.id,
      "X-Operator-Role": operatorContext.role,
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      const fallback = await response.text();
      if (fallback) {
        message = fallback;
      }
    }

    throw new Error(message);
  }

  return (await response.json()) as T;
}
