import { ReactNode } from "react";
import layoutStyles from "../styles/page-layout.module.css";

interface AgentsPageProps {
  overview: ReactNode;
  timeline: ReactNode;
}

export function AgentsPage({ overview, timeline }: AgentsPageProps) {
  return (
    <div className={layoutStyles.pageStack}>
      <section>{overview}</section>
      <section>{timeline}</section>
    </div>
  );
}
