import { ReactNode } from "react";
import layoutStyles from "../styles/page-layout.module.css";

interface OperationsPageProps {
  list: ReactNode;
  detail: ReactNode;
}

export function OperationsPage({ list, detail }: OperationsPageProps) {
  return (
    <div className={layoutStyles.primaryWithRail}>
      <section className={layoutStyles.primaryContent}>{list}</section>
      <aside className={layoutStyles.sideRail}>{detail}</aside>
    </div>
  );
}
