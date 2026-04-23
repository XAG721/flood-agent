import { ReactNode } from "react";
import shellStyles from "../styles/mission-shell.module.css";

interface OperationsPageProps {
  list: ReactNode;
  detail: ReactNode;
}

export function OperationsPage({ list, detail }: OperationsPageProps) {
  return (
    <div className={shellStyles.shell}>
      <section className={shellStyles.hero}>
        <div className={shellStyles.heroBody}>
          <p className={shellStyles.eyebrow}>Response Orchestration Desk</p>
          <h2 className={shellStyles.title}>协同处置</h2>
          <p className={shellStyles.summary}>
            这里不再只是旧版队列页，而是承接数字孪生主屏生成的 proposal、证据链、执行编排和分众预警草稿的行动中枢。
            页面左侧聚焦待处置动作与历史闭环，右侧持续显示当前决策上下文、风险提醒与后续分发入口。
          </p>
        </div>
        <div className={shellStyles.heroStats}>
          <article className={shellStyles.statCard}>
            <span>处置定位</span>
            <strong>Proposal 审批与执行编排</strong>
          </article>
          <article className={shellStyles.statCard}>
            <span>闭环目标</span>
            <strong>从请示到分众预警</strong>
          </article>
        </div>
      </section>

      <div className={`${shellStyles.contentGrid} ${shellStyles.withRail}`}>
        <section className={shellStyles.card}>
          <div>
            <p className={shellStyles.cardLabel}>Action Queue</p>
            <h3 className={shellStyles.cardTitle}>当前处置链</h3>
            <p className={shellStyles.cardSummary}>
              优先展示当前事件的区域级 proposal、历史处置结果以及需要人工确认的动作，不再回到旧的对象审批视图。
            </p>
          </div>
          <div className={shellStyles.contentFrame}>{list}</div>
        </section>

        <aside className={shellStyles.card}>
          <div>
            <p className={shellStyles.cardLabel}>Decision Context</p>
            <h3 className={shellStyles.cardTitle}>行动细节与联动出口</h3>
            <p className={shellStyles.cardSummary}>
              将证据、风险说明、审批动作和预警生成入口收束到同一侧栏，保证指挥视角不被分散。
            </p>
          </div>
          <div className={shellStyles.contentFrame}>{detail}</div>
        </aside>
      </div>
    </div>
  );
}
