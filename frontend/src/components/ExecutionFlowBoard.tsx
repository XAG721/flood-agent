import { motion } from "framer-motion";
import storyStyles from "../styles/story-panels.module.css";

export interface ExecutionFlowStep {
  id: string;
  title: string;
  summary: string;
  detail: string;
  status: "complete" | "active" | "pending";
}

interface ExecutionFlowBoardProps {
  title: string;
  description: string;
  stats: string[];
  steps: ExecutionFlowStep[];
}

export function ExecutionFlowBoard({
  title,
  description,
  stats,
  steps,
}: ExecutionFlowBoardProps) {
  return (
    <motion.section
      className={storyStyles.flowBoard}
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className={storyStyles.flowHeader}>
        <div>
          <p className={storyStyles.eyebrow}>协同处置</p>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
        <div className={storyStyles.flowStats}>
          {stats.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </div>

      <div className={storyStyles.flowChain}>
        {steps.map((step, index) => (
          <motion.article
            key={step.id}
            className={storyStyles.flowStep}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.06, duration: 0.32 }}
          >
            <div className={storyStyles.flowMarker}>
              <span className={storyStyles.flowIndex}>{String(index + 1).padStart(2, "0")}</span>
              <span
                className={`${storyStyles.flowDot} ${
                  step.status === "complete"
                    ? storyStyles.flowDotComplete
                    : step.status === "active"
                      ? storyStyles.flowDotActive
                      : ""
                }`}
              />
              {index < steps.length - 1 ? <span className={storyStyles.flowLine} /> : null}
            </div>
            <div className={storyStyles.flowContent}>
              <div className={storyStyles.flowTopline}>
                <strong>{step.title}</strong>
                <span className={storyStyles.flowStatus}>
                  {step.status === "complete" ? "已完成" : step.status === "active" ? "进行中" : "待推进"}
                </span>
              </div>
              <p className={storyStyles.flowSummary}>{step.summary}</p>
              <p className={storyStyles.flowDetail}>{step.detail}</p>
            </div>
          </motion.article>
        ))}
      </div>
    </motion.section>
  );
}
