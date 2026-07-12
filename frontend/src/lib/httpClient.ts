const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

const operatorContext: { id: string; role: string; terminalId: string } = {
  id: "operator_console",
  role: "commander",
  terminalId: "district-web-console",
};

export function buildUrl(path: string): string {
  return `${apiBaseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

export function setApiOperatorContext(context: { id?: string; role: string; terminalId?: string }) {
  operatorContext.id = context.id?.trim() || "operator_console";
  operatorContext.role = context.role;
  operatorContext.terminalId = context.terminalId?.trim() || operatorContext.terminalId;
}

export function getApiOperatorContext(): Readonly<{ id: string; role: string; terminalId: string }> {
  return operatorContext;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const requestId = globalThis.crypto?.randomUUID?.()
    ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const method = (init?.method ?? "GET").toUpperCase();
  const response = await fetch(buildUrl(path), {
    headers: {
      "Content-Type": "application/json",
      "X-Operator-Id": operatorContext.id,
      "X-Operator-Role": operatorContext.role,
      "X-Operator-Terminal": operatorContext.terminalId,
      "X-Correlation-ID": requestId,
      ...(method === "GET" || method === "HEAD" ? {} : { "Idempotency-Key": requestId }),
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const payload = (await response.json()) as {
        detail?: string | { code?: string; message?: string; retryable?: boolean };
      };
      if (payload.detail) {
        message = typeof payload.detail === "string"
          ? payload.detail
          : `${payload.detail.code ? `${payload.detail.code}：` : ""}${payload.detail.message ?? message}`;
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
