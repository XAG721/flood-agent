import panelStyles from "../styles/shared-panels.module.css";

export interface SignalTimelineItem {
  id: string;
  title: string;
  detail: string;
  meta?: string;
  tone?: "info" | "warning" | "critical" | "neutral";
}

interface SignalTimelineProps {
  title: string;
  subtitle: string;
  items: SignalTimelineItem[];
  emptyText: string;
}

export function SignalTimeline({ title, subtitle, items, emptyText }: SignalTimelineProps) {
  return (
    <section className={panelStyles.signalPanel}>
      <div className={panelStyles.panelTitleRow}>
        <div>
          <p className={panelStyles.panelKicker}>{title}</p>
          <h3>{subtitle}</h3>
        </div>
      </div>
      <div className={panelStyles.signalTimeline}>
        {items.length ? (
          items.map((item) => (
            <article key={item.id} className={panelStyles.signalRow}>
              <span className={`${panelStyles.signalDot} ${signalToneClass(item.tone)}`} />
              <div className={panelStyles.signalContent}>
                <div className={panelStyles.signalTitleRow}>
                  <strong>{item.title}</strong>
                  {item.meta ? <span>{item.meta}</span> : null}
                </div>
                <p>{item.detail}</p>
              </div>
            </article>
          ))
        ) : (
          <p className={panelStyles.emptyText}>{emptyText}</p>
        )}
      </div>
    </section>
  );
}

function signalToneClass(tone?: SignalTimelineItem["tone"]) {
  return {
    info: panelStyles.signalInfo,
    warning: panelStyles.signalWarning,
    critical: panelStyles.signalCritical,
    neutral: panelStyles.signalNeutral,
    undefined: panelStyles.signalNeutral,
  }[String(tone) as keyof Record<string, string>];
}
