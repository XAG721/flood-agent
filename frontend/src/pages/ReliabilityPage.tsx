import { ReactNode } from "react";
import shellStyles from "../styles/mission-shell.module.css";

interface ReliabilityPageProps {
  health: ReactNode;
  governance: ReactNode;
  closure?: ReactNode;
}

export function ReliabilityPage({ health, governance, closure }: ReliabilityPageProps) {
  return (
    <div className={shellStyles.shell}>
      <section className={shellStyles.hero}>
        <div className={shellStyles.heroBody}>
          <p className={shellStyles.eyebrow}>Reliability And Governance</p>
          <h2 className={shellStyles.title}>审计与可靠性</h2>
          <p className={shellStyles.summary}>
            这一页承接模型运行健康、审计阻断、失败原因与可回放证据，确保数字孪生智能体系统不仅能给答案，也能解释为什么给出这个答案。
          </p>
        </div>
        <div className={shellStyles.heroStats}>
          <article className={shellStyles.statCard}>
            <span>监控主题</span>
            <strong>模型稳定性与数据新鲜度</strong>
          </article>
          <article className={shellStyles.statCard}>
            <span>治理主题</span>
            <strong>审计留痕与权限边界</strong>
          </article>
        </div>
      </section>

      <div className={`${shellStyles.contentGrid} ${shellStyles.withRail}`}>
        <section className={shellStyles.card}>
          <div>
            <p className={shellStyles.cardLabel}>System Health</p>
            <h3 className={shellStyles.cardTitle}>运行健康</h3>
            <p className={shellStyles.cardSummary}>
              关注模型调用、任务执行、数据时效和链路恢复状态，用来判断当前系统是否还适合继续自动化辅助决策。
            </p>
          </div>
          <div className={shellStyles.contentFrame}>{health}</div>
        </section>

        <aside className={shellStyles.card}>
          <div>
            <p className={shellStyles.cardLabel}>Governance Rail</p>
            <h3 className={shellStyles.cardTitle}>审计与权限</h3>
            <p className={shellStyles.cardSummary}>
              汇总关键审计记录、权限约束和高风险阻断结果，方便在答辩和复盘场景中快速解释系统边界。
            </p>
          </div>
          <div className={shellStyles.contentFrame}>{governance}</div>
        </aside>
      </div>

      {closure ? (
        <section className={shellStyles.card}>
          <div>
            <p className={shellStyles.cardLabel}>Closure And Replay</p>
            <h3 className={shellStyles.cardTitle}>闭环结果与复盘入口</h3>
            <p className={shellStyles.cardSummary}>
              在同一页收束 proposal 审批、warning 生成、审计留痕与复盘线索，避免可靠性视图只停留在健康监控。
            </p>
          </div>
          <div className={shellStyles.contentFrame}>{closure}</div>
        </section>
      ) : null}
    </div>
  );
}
