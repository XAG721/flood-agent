import type { EvidenceItem } from "../types/api";

export function severityText(severity?: string | null) {
  if (!severity) {
    return "--";
  }
  return {
    info: "提示",
    warning: "告警",
    critical: "严重",
  }[severity] ?? severity;
}

export function formatTimestamp(value?: string | null) {
  return value
    ? new Date(value).toLocaleString("zh-CN", {
        hour12: false,
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "--";
}

export function formatPercent(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${Math.round(value * 100)}%`;
}

export function coerceStrings(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

export function coerceDrafts(value: unknown) {
  return Array.isArray(value)
    ? value.filter(
        (
          item,
        ): item is { draft_id: string; audience: string; channel: string; content: string; created_at?: string } =>
          !!item &&
          typeof item === "object" &&
          typeof item.draft_id === "string" &&
          typeof item.audience === "string" &&
          typeof item.channel === "string" &&
          typeof item.content === "string",
      )
    : [];
}

export function coerceTemplates(value: unknown) {
  return Array.isArray(value)
    ? value.filter(
        (item): item is { audience: string; channel: string; content: string } =>
          !!item &&
          typeof item === "object" &&
          typeof item.audience === "string" &&
          typeof item.channel === "string" &&
          typeof item.content === "string",
      )
    : [];
}

export function coerceLogs(value: unknown) {
  return Array.isArray(value)
    ? value.filter(
        (
          item,
        ): item is { log_id: string; action_type: string; summary: string; operator_id: string; created_at?: string } =>
          !!item &&
          typeof item === "object" &&
          typeof item.log_id === "string" &&
          typeof item.action_type === "string" &&
          typeof item.summary === "string" &&
          typeof item.operator_id === "string",
      )
    : [];
}

export function parseCsv(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function joinCsv(value: string[]) {
  return value.join(", ");
}

export function formatTrend(value?: string | null) {
  if (!value) {
    return "暂无趋势判断";
  }
  const map: Record<string, string> = {
    rising: "持续上涨",
    rapidly_rising: "快速上升",
    stable: "基本稳定",
    falling: "逐步回落",
    unknown: "未知",
  };
  return map[value] ?? value;
}

export function evidenceExplainChips(item: EvidenceItem) {
  const explain = item.retrieval_explain ?? {};
  const chips: string[] = [];
  const matchedTerms = Array.isArray(explain.matched_terms)
    ? explain.matched_terms.filter((term): term is string => typeof term === "string")
    : [];

  if (matchedTerms.length) {
    chips.push(`命中：${matchedTerms.slice(0, 3).join("、")}`);
  }
  if (typeof explain.final_score === "number") {
    chips.push(`得分 ${explain.final_score.toFixed(2)}`);
  }
  if (typeof explain.recency_multiplier === "number") {
    chips.push(`新鲜度 x${explain.recency_multiplier.toFixed(2)}`);
  }

  return chips;
}
