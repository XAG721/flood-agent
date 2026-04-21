import type { ReactNode } from "react";
import storyStyles from "../styles/story-panels.module.css";
import type { EntityImpactView, EvidenceItem, StructuredAnswer } from "../types/api";

interface CopilotContextPanelProps {
  impact: EntityImpactView | null;
  entityTypeLabel: string;
  riskLabel: string;
  evidence: EvidenceItem[];
  answer: StructuredAnswer | null;
  switcher: ReactNode;
  emptyClassName: string;
}

export function CopilotContextPanel({
  impact,
  entityTypeLabel,
  riskLabel,
  evidence,
  answer,
  switcher,
  emptyClassName,
}: CopilotContextPanelProps) {
  const reasoning = answer?.planning_layers_summary ?? [];

  return (
    <div className={storyStyles.contextPanel}>
      <div className={storyStyles.contextHeader}>
        <div>
          <p className={storyStyles.eyebrow}>研判上下文</p>
          <h3>{impact?.entity.name ?? "等待选择对象"}</h3>
        </div>
        {impact ? <span className={storyStyles.flowStatus}>{riskLabel}</span> : null}
      </div>

      {impact ? (
        <div className={storyStyles.contextBar}>
          <div className={storyStyles.contextLead}>
            <span>当前对象</span>
            <strong>{impact.entity.name}</strong>
            <p>{impact.risk_reason[0] ?? "已就当前对象建立影响研判上下文。"}</p>
          </div>
          <div className={storyStyles.contextPill}>
            <span>对象类型</span>
            <strong>{entityTypeLabel}</strong>
          </div>
          <div className={storyStyles.contextPill}>
            <span>预计影响时间</span>
            <strong>{impact.time_to_impact_minutes} 分钟</strong>
          </div>
          <div className={storyStyles.contextPill}>
            <span>证据数量</span>
            <strong>{evidence.length}</strong>
          </div>
        </div>
      ) : null}

      <div className={storyStyles.contextSection}>
        <div className={storyStyles.contextSectionHeader}>
          <h4>关键证据</h4>
          <p>优先显示最能支撑本轮分析的依据</p>
        </div>
        <div className={storyStyles.evidenceStack}>
          {evidence.length ? (
            evidence.slice(0, 3).map((item) => (
              <article key={`${item.evidence_type}_${item.source_id}`} className={storyStyles.evidenceCard}>
                <div className={storyStyles.evidenceMeta}>
                  <span>{item.evidence_type}</span>
                  <span>
                    {item.timestamp ? new Date(item.timestamp).toLocaleString("zh-CN", { hour12: false }) : "无时间戳"}
                  </span>
                </div>
                <h4>{item.title}</h4>
                <p>{item.excerpt}</p>
              </article>
            ))
          ) : (
            <p className={emptyClassName}>当前还没有可展示的证据，建议先围绕对象发起一轮具体提问。</p>
          )}
        </div>
      </div>

      <div className={storyStyles.contextSection}>
        <div className={storyStyles.contextSectionHeader}>
          <h4>本轮推理说明</h4>
          <p>只保留影响研判最关键的推理层</p>
        </div>
        {reasoning.length ? (
          <ul className={storyStyles.reasonListCompact}>
            {reasoning.slice(0, 4).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : (
          <p className={emptyClassName}>发起问答后，这里会同步显示本轮推理使用的关键思路。</p>
        )}
      </div>

      <div className={storyStyles.contextSection}>
        <div className={storyStyles.contextSectionHeader}>
          <h4>切换重点对象</h4>
          <p>保持自由提问，但始终围绕清晰对象上下文</p>
        </div>
        {switcher}
      </div>
    </div>
  );
}
