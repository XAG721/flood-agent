import { motion } from "framer-motion";
import storyStyles from "../styles/story-panels.module.css";

interface OverviewHeroProps {
  eventTitle: string;
  riskLabel: string;
  trendLabel: string;
  summary: string;
  agentSummary: string;
  leadImpactLabel: string;
  highPriorityCount: number;
  pendingProposalCount: number;
  onPrimaryAction: () => void;
  onSecondaryAction: () => void;
  isBusy: boolean;
  primaryButtonClassName: string;
  secondaryButtonClassName: string;
}

export function OverviewHero({
  eventTitle,
  riskLabel,
  trendLabel,
  summary,
  agentSummary,
  leadImpactLabel,
  highPriorityCount,
  pendingProposalCount,
  onPrimaryAction,
  onSecondaryAction,
  isBusy,
  primaryButtonClassName,
  secondaryButtonClassName,
}: OverviewHeroProps) {
  return (
    <motion.section
      className={storyStyles.heroPanel}
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className={storyStyles.heroGlow} />
      <div className={storyStyles.heroGrid}>
        <div className={storyStyles.heroCopy}>
          <div>
            <p className={storyStyles.eyebrow}>风险总览</p>
            <h3 className={storyStyles.heroTitle}>{riskLabel}风险正在抬升</h3>
          </div>
          <p className={storyStyles.heroSummary}>{summary}</p>
          <div className={storyStyles.heroActions}>
            <button type="button" className={primaryButtonClassName} onClick={onPrimaryAction} disabled={isBusy}>
              进入影响问答
            </button>
            <button type="button" className={secondaryButtonClassName} onClick={onSecondaryAction} disabled={isBusy}>
              查看协同处置
            </button>
          </div>
          <div className={storyStyles.heroKeyline}>
            <article>
              <span>当前事件</span>
              <strong>{eventTitle}</strong>
            </article>
            <article>
              <span>首要影响对象</span>
              <strong>{leadImpactLabel}</strong>
            </article>
            <article>
              <span>趋势判断</span>
              <strong>{trendLabel}</strong>
            </article>
          </div>
        </div>

        <div className={storyStyles.heroPrimaryMeta}>
          <div className={storyStyles.heroMetaCard}>
            <span className={storyStyles.heroMetaLabel}>共享研判摘要</span>
            <strong className={storyStyles.heroMetaValue}>{agentSummary}</strong>
            <p className={storyStyles.heroMetaHint}>
              当前平台已经把全局风险、重点对象和待确认动作压缩到同一条决策线里。
            </p>
          </div>
          <div className={storyStyles.heroSideNote}>
            <span className={storyStyles.sideNoteLabel}>下一步建议</span>
            <p className={storyStyles.sideNoteBody}>
              当前有 {highPriorityCount} 个重点对象值得继续追问，另有 {pendingProposalCount} 条动作等待确认。
            </p>
          </div>
        </div>
      </div>
    </motion.section>
  );
}
