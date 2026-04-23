import { ReactNode } from "react";
import shellStyles from "../styles/mission-shell.module.css";

interface AgentsPageProps {
  briefing: ReactNode;
  chamber: ReactNode;
  orchestration: ReactNode;
  governance: ReactNode;
}

export function AgentsPage({ briefing, chamber, orchestration, governance }: AgentsPageProps) {
  return (
    <div className={shellStyles.shell}>
      <section className={shellStyles.hero}>
        <div className={shellStyles.heroBody}>
          <p className={shellStyles.eyebrow}>Multi-Agent Conference Room</p>
          <h2 className={shellStyles.title}>智能体会商室</h2>
          <p className={shellStyles.summary}>
            这一页不只展示最终建议，而是把多智能体围绕同一事件形成差异、被 supervisor 编排、被审计边界阻断，
            以及仍需人工追问的部分完整展开，方便指挥员快速判断是否继续自动推进。
          </p>
        </div>
        <div className={shellStyles.heroStats}>
          <article className={shellStyles.statCard}>
            <span>协同模式</span>
            <strong>Impact / Action / Warning / Audit</strong>
          </article>
          <article className={shellStyles.statCard}>
            <span>页面目标</span>
            <strong>突出差异、编排与放行边界</strong>
          </article>
        </div>
      </section>

      <div className={`${shellStyles.contentGrid} ${shellStyles.withRail}`}>
        <div className={shellStyles.contentFrame}>
          <section className={shellStyles.card}>
            <div>
              <p className={shellStyles.cardLabel}>Conference Briefing</p>
              <h3 className={shellStyles.cardTitle}>分歧与结论总览</h3>
              <p className={shellStyles.cardSummary}>
                先看不同角色对同一风险的判断差异、证据覆盖和建议动作，再决定是进入编排放行，还是继续追问补证。
              </p>
            </div>
            <div className={shellStyles.contentFrame}>{briefing}</div>
          </section>

          <section className={shellStyles.card}>
            <div>
              <p className={shellStyles.cardLabel}>Conference Chamber</p>
              <h3 className={shellStyles.cardTitle}>会商主桌面</h3>
              <p className={shellStyles.cardSummary}>
                会商主桌面负责呈现任务流、tool trace、记忆复用和 supervisor 运行状态，是分析过程的透明视图。
              </p>
            </div>
            <div className={shellStyles.contentFrame}>{chamber}</div>
          </section>
        </div>

        <div className={shellStyles.contentFrame}>
          <section className={shellStyles.card}>
            <div>
              <p className={shellStyles.cardLabel}>Orchestration Board</p>
              <h3 className={shellStyles.cardTitle}>编排与推理时间线</h3>
              <p className={shellStyles.cardSummary}>
                把 decision path、关键时间线和最新编排结果放到一处，便于判断当前处于补证、放行还是转人工的哪一步。
              </p>
            </div>
            <div className={shellStyles.contentFrame}>{orchestration}</div>
          </section>

          <section className={shellStyles.card}>
            <div>
              <p className={shellStyles.cardLabel}>Governance Rail</p>
              <h3 className={shellStyles.cardTitle}>开放问题与治理边界</h3>
              <p className={shellStyles.cardSummary}>
                收拢 open questions、阻断原因和闭环出口，让指挥员快速知道哪些地方还必须人工接管。
              </p>
            </div>
            <div className={shellStyles.contentFrame}>{governance}</div>
          </section>
        </div>
      </div>
    </div>
  );
}
