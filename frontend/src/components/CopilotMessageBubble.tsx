import { motion } from "framer-motion";
import styles from "../App.module.css";
import { formatGenerationSource } from "../lib/displayText";
import { formatTimestamp } from "../lib/consoleFormatting";
import type { V2CopilotMessage } from "../types/api";

export function CopilotMessageBubble({ message }: { message: V2CopilotMessage }) {
  const structured = message.structured_answer;
  const evidencePreview = structured?.evidence.slice(0, 2) ?? [];

  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${styles.messageBubble} ${
        message.role === "user" ? styles.messageUser : styles.messageAssistant
      }`}
    >
      <div className={styles.messageTopline}>
        <span>{message.role === "user" ? "值班席" : "智能协同"}</span>
        <span>{formatTimestamp(message.created_at)}</span>
      </div>
      <p>{message.content}</p>
      {structured ? (
        <div className={styles.answerMeta}>
          <div className={styles.answerTags}>
            <span>证据 {structured.evidence.length}</span>
            <span>动作 {structured.recommended_actions.length}</span>
            <span>置信度{Math.round(structured.confidence * 100)}%</span>
            <span>{formatGenerationSource(structured.generation_source)}</span>
            {structured.model_name ? <span>{structured.model_name}</span> : null}
          </div>

          {structured.impact_summary.length ? (
            <section className={styles.answerSection}>
              <strong>影响过程</strong>
              <ul className={styles.answerList}>
                {structured.impact_summary.slice(0, 3).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>
          ) : null}

          {evidencePreview.length ? (
            <section className={styles.answerSection}>
              <strong>依据摘录</strong>
              <ul className={styles.answerList}>
                {evidencePreview.map((item) => (
                  <li key={`${item.evidence_type}_${item.source_id}`}>{item.title}</li>
                ))}
              </ul>
            </section>
          ) : null}

          <section className={styles.answerSection}>
            <strong>研判结论</strong>
            <p>{structured.answer}</p>
          </section>

          {structured.recommended_actions.length ? (
            <section className={styles.answerSection}>
              <strong>建议动作</strong>
              <ul className={styles.answerList}>
                {structured.recommended_actions.slice(0, 3).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>
          ) : null}
        </div>
      ) : null}
    </motion.article>
  );
}
