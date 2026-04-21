import panelStyles from "../styles/shared-panels.module.css";

export interface PriorityObjectItem {
  id: string;
  name: string;
  typeLabel: string;
  village: string;
  emphasis: string;
  riskLabel?: string;
  riskTone?: "none" | "blue" | "yellow" | "orange" | "red";
}

interface PriorityObjectPanelProps {
  title?: string;
  subtitle: string;
  items: PriorityObjectItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function PriorityObjectPanel({
  title,
  subtitle,
  items,
  selectedId,
  onSelect,
}: PriorityObjectPanelProps) {
  return (
    <section className={panelStyles.priorityPanel}>
      <div className={panelStyles.panelTitleRow}>
        <div>
          {title ? <p className={panelStyles.panelKicker}>{title}</p> : null}
          <h3>{subtitle}</h3>
        </div>
        <span className={panelStyles.panelCount}>{items.length} 个对象</span>
      </div>

      <div className={panelStyles.priorityList}>
        {items.length ? (
          items.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              className={`${panelStyles.priorityCard} ${
                selectedId === item.id ? panelStyles.priorityCardActive : ""
              }`}
            >
              <div className={panelStyles.priorityTopline}>
                <div>
                  <strong>{item.name}</strong>
                  <span>
                    {item.typeLabel} / {item.village}
                  </span>
                </div>
                {item.riskLabel ? (
                  <span className={`${panelStyles.riskBadge} ${riskToneClass(item.riskTone)}`}>
                    {item.riskLabel}
                  </span>
                ) : null}
              </div>
              <p>{item.emphasis}</p>
            </button>
          ))
        ) : (
          <p className={panelStyles.emptyText}>当前还没有进入优先对象池的重点目标。</p>
        )}
      </div>
    </section>
  );
}

function riskToneClass(tone?: PriorityObjectItem["riskTone"]) {
  if (!tone) {
    return panelStyles.riskNeutral;
  }
  return {
    none: panelStyles.riskNeutral,
    blue: panelStyles.riskBlue,
    yellow: panelStyles.riskYellow,
    orange: panelStyles.riskOrange,
    red: panelStyles.riskRed,
  }[tone];
}
