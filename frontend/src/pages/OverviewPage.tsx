import { ReactNode } from "react";
import layoutStyles from "../styles/page-layout.module.css";

interface OverviewPageProps {
  situation: ReactNode;
  priority: ReactNode;
  summary: ReactNode;
  signals: ReactNode;
}

export function OverviewPage({ situation, priority, summary, signals }: OverviewPageProps) {
  return (
    <div className={layoutStyles.pageStack}>
      <section className={layoutStyles.overviewGrid}>
        <div>{summary}</div>
        <div>{signals}</div>
      </section>
    </div>
  );
}
