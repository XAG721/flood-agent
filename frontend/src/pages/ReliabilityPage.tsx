import { ReactNode } from "react";
import layoutStyles from "../styles/page-layout.module.css";

interface ReliabilityPageProps {
  health: ReactNode;
  governance: ReactNode;
}

export function ReliabilityPage({ health, governance }: ReliabilityPageProps) {
  return (
    <div className={layoutStyles.primaryWithRail}>
      <section className={layoutStyles.primaryContent}>{health}</section>
      <aside className={layoutStyles.sideRail}>{governance}</aside>
    </div>
  );
}
