import { ReactNode } from "react";
import layoutStyles from "../styles/page-layout.module.css";

export function DataPage({ children }: { children: ReactNode }) {
  return <div className={layoutStyles.pageStack}>{children}</div>;
}
