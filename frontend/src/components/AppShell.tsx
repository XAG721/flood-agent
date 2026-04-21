import { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import shellStyles from "../styles/app-shell.module.css";

export interface AppNavItem {
  path: string;
  label: string;
}

export interface AppMetricItem {
  label: string;
  value: string;
  hint?: string;
  tone?: "risk" | "warning" | "success" | "neutral";
}

interface AppShellProps {
  brandTitle: string;
  brandCopy?: string;
  currentPageLabel: string;
  currentPageTitle: string;
  currentPageDescription?: string;
  operatorControl: ReactNode;
  statusSignals: ReactNode;
  navigation: AppNavItem[];
  utilityNavigation?: AppNavItem[];
  metrics: ReactNode;
  children: ReactNode;
}

export function AppShell({
  brandTitle,
  brandCopy,
  currentPageLabel,
  currentPageTitle,
  currentPageDescription,
  operatorControl,
  statusSignals,
  navigation,
  utilityNavigation,
  metrics,
  children,
}: AppShellProps) {
  return (
    <div className={shellStyles.appShell}>
      <div className={shellStyles.masthead}>
        <header className={shellStyles.topbar}>
          <div className={shellStyles.brandBlock}>
            <p className={shellStyles.kicker}>应急指挥中台</p>
            <h1>{brandTitle}</h1>
            {brandCopy ? <p className={shellStyles.brandCopy}>{brandCopy}</p> : null}
          </div>
          <div className={shellStyles.topbarAside}>
            <div className={shellStyles.operatorControl}>{operatorControl}</div>
            <div className={shellStyles.statusSignalRow}>{statusSignals}</div>
          </div>
        </header>

        <div className={shellStyles.navigationBand}>
          <nav className={shellStyles.topNavigation} aria-label="primary-navigation">
            {navigation.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === "/"}
                className={({ isActive }) =>
                  `${shellStyles.navLink} ${isActive ? shellStyles.navLinkActive : ""}`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          {utilityNavigation?.length ? (
            <nav className={shellStyles.utilityNavigation} aria-label="secondary-navigation">
              {utilityNavigation.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === "/"}
                  className={({ isActive }) =>
                    `${shellStyles.utilityLink} ${isActive ? shellStyles.utilityLinkActive : ""}`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          ) : null}
        </div>

        <section className={shellStyles.pageHeader}>
          <div>
            <p className={shellStyles.sectionLabel}>{currentPageLabel}</p>
            <h2>{currentPageTitle}</h2>
            {currentPageDescription ? <p className={shellStyles.pageDescription}>{currentPageDescription}</p> : null}
          </div>
        </section>

        <section className={shellStyles.metricStrip}>{metrics}</section>
      </div>

      <main className={shellStyles.pageBody}>{children}</main>
    </div>
  );
}
