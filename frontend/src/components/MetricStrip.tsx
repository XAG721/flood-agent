import metricStyles from "../styles/shared-panels.module.css";

export interface MetricStripItem {
  label: string;
  value: string;
  hint?: string;
  tone?: "risk" | "warning" | "success" | "neutral";
}

export function MetricStrip({ items }: { items: MetricStripItem[] }) {
  return (
    <>
      {items.map((item) => (
        <article
          key={`${item.label}-${item.value}`}
          className={`${metricStyles.metricCard} ${item.tone ? metricStyles[`metric${capitalize(item.tone)}`] : ""}`}
        >
          <span>{item.label}</span>
          <strong>{item.value}</strong>
          {item.hint ? <small>{item.hint}</small> : null}
        </article>
      ))}
    </>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
