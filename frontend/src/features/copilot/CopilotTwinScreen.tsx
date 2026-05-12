import { lazy, Suspense, useEffect, useMemo, useState, type FormEvent } from "react";
import { NavLink } from "react-router-dom";
import styles from "./copilot-twin-screen.module.css";
import dashboardStyles from "../../styles/digital-twin-screen.module.css";
import { buildMentionCandidates, isRouteGuidanceText, resolveMentionedObjectId } from "../../lib/objectMention";
import type {
  AgentDialogResponse,
  AgentDialogTranscriptEntry,
  FocusObjectView,
  RiskLevel,
  TwinOverviewView,
} from "../../types/api";

const DigitalTwinCesiumCanvas = lazy(() =>
  import("../../components/DigitalTwinCesiumCanvas").then((module) => ({ default: module.DigitalTwinCesiumCanvas })),
);

type CopilotTwinScreenProps = {
  overview: TwinOverviewView | null;
  focusObject: FocusObjectView | null;
  dialogEntries: AgentDialogTranscriptEntry[];
  busy?: boolean;
  onSelectObject: (objectId: string) => void | Promise<void>;
  onAsk: (prompt: string) => void | Promise<void>;
};

const PROMPT_SUGGESTIONS = [
  "文艺路小学即将进入放学时段，请说明智能体如何判断学生、家长车辆和周边道路的叠加风险，并给出避险安排。",
  "碑林中心医院急诊入口需要连续通行，请说明智能体如何生成入口保障、后勤车辆绕行和排水资源前置方案。",
  "李奶奶属于独居老人且行动不便，请说明智能体建议谁去接、走哪条避险路线、为什么不能让老人自行离开。",
  "请把当前对象的证据、风险链、处置资源、推荐路径和人工审批边界完整说明清楚。",
];

function latestResponse(entries: AgentDialogTranscriptEntry[]): AgentDialogResponse | undefined {
  return [...entries].reverse().find((entry) => entry.response)?.response;
}

function latestUserQuestion(entries: AgentDialogTranscriptEntry[]) {
  return [...entries].reverse().find((entry) => entry.role === "user")?.content ?? "";
}

function riskLabel(riskLevel?: RiskLevel | null) {
  return {
    None: "常态",
    Blue: "蓝色",
    Yellow: "黄色",
    Orange: "橙色",
    Red: "红色",
  }[riskLevel ?? "None"];
}

function formatClock(date: Date) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatCalendar(date: Date) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    weekday: "short",
  }).format(date);
}

export function CopilotTwinScreen({
  overview,
  focusObject,
  dialogEntries,
  busy = false,
  onSelectObject,
  onAsk,
}: CopilotTwinScreenProps) {
  const [input, setInput] = useState("");
  const [now, setNow] = useState(() => new Date());
  const focusObjects = overview?.focus_objects ?? [];
  const mapLayers = overview?.map_layers ?? [];
  const currentResponse = latestResponse(dialogEntries);
  const currentQuestion = latestUserQuestion(dialogEntries);
  const mentionCandidates = useMemo(() => buildMentionCandidates(focusObjects, mapLayers), [focusObjects, mapLayers]);
  const dialogFocusObjectId = resolveMentionedObjectId(currentQuestion, mentionCandidates, currentResponse);
  const responseRouteText = [
    currentResponse?.answer,
    ...(currentResponse?.recommended_actions ?? []),
    ...(currentResponse?.impact_summary ?? []),
    ...(currentResponse?.risk_reminders ?? []),
  ].join(" ");
  const routeHighlightObjectId =
    dialogFocusObjectId && isRouteGuidanceText(currentQuestion) && isRouteGuidanceText(responseRouteText)
      ? dialogFocusObjectId
      : null;
  const focusSummary =
    (dialogFocusObjectId ? focusObjects.find((item) => item.object_id === dialogFocusObjectId) : undefined) ??
    (focusObject
      ? {
          object_id: focusObject.object_id,
          name: focusObject.object_name,
          entity_type: focusObject.entity_type,
          village: focusObject.village,
          risk_level: focusObject.risk_level,
          time_to_impact_minutes: focusObject.time_to_impact_minutes,
          summary: focusObject.summary,
          recommended_action: focusObject.recommended_actions[0] ?? "继续跟踪对象风险。",
          pending_proposal_ids: [],
          canvas_position: {},
        }
      : focusObjects[0]);
  const selectedRiskLevel = focusSummary?.risk_level ?? overview?.overall_risk_level ?? "None";
  const pageKpis = [
    {
      label: "总体风险",
      value: riskLabel(overview?.overall_risk_level ?? selectedRiskLevel),
      detail: "快速上升",
      accent: dashboardStyles.riskOrange,
    },
    {
      label: "活动信号",
      value: `${overview?.active_alert_count ?? 0}`,
      detail: "实时",
      accent: dashboardStyles.riskBlue,
    },
    {
      label: "对象总数",
      value: `${focusObjects.length}`,
      detail: focusSummary?.name ?? "等待对象",
      accent: dashboardStyles.riskBlue,
    },
    {
      label: "待审批",
      value: `${overview?.pending_proposal_count ?? 0}`,
      detail: "处置方案队列",
      accent: dashboardStyles.riskOrange,
    },
    {
      label: "已批准",
      value: `${overview?.approved_proposal_count ?? 0}`,
      detail: "处置动作",
      accent: dashboardStyles.riskGreen,
    },
    {
      label: "分众预警",
      value: `${overview?.warning_draft_count ?? 0}`,
      detail: "闭环已完成",
      accent: dashboardStyles.riskBlue,
    },
  ];

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = input.trim();
    if (!question) {
      return;
    }
    void onAsk(question);
    setInput("");
  }

  return (
    <section className={`${dashboardStyles.screen} ${dashboardStyles.cityBigScreen} ${styles.screenFrame}`}>
      <div className={dashboardStyles.cityAura} />

      <header className={dashboardStyles.cityHeader}>
        <div className={`${dashboardStyles.cityHeaderWing} ${dashboardStyles.cityBrandWing}`}>
          <div className={dashboardStyles.cityLogoMark}>涝</div>
          <div>
            <span>智汇水务</span>
            <strong>碑林区重点片区</strong>
          </div>
          <em>区域选择</em>
        </div>
        <div className={dashboardStyles.cityTitleBlock}>
          <p>数字孪生洪水预警平台</p>
          <h2>数字孪生智能体洪水预警主屏</h2>
          <small>{overview?.event_title ?? "多源感知、风险研判、预警发布与闭环管理"}</small>
        </div>
        <div className={`${dashboardStyles.cityHeaderWing} ${dashboardStyles.cityStatusWing}`}>
          <div>
            <span>当前时间</span>
            <strong>{formatClock(now)}</strong>
            <small>{formatCalendar(now)}</small>
          </div>
          <div>
            <span>天气</span>
            <strong>中雨 22-25℃</strong>
            <small>降雨持续</small>
          </div>
          <div>
            <span>值班状态</span>
            <strong>实时</strong>
            <small>张建明 值守中</small>
          </div>
        </div>
      </header>

      <nav className={dashboardStyles.cityModuleTabs} aria-label="大屏功能分页">
        <NavLink to="/" end>态势总览</NavLink>
        <NavLink to="/copilot">智能问答</NavLink>
        <NavLink to="/operations">风险预警</NavLink>
        <NavLink to="/agents">预警分析</NavLink>
        <NavLink to="/reliability">事件复盘</NavLink>
      </nav>

      <section className={dashboardStyles.cityModeBanner} aria-label="当前大屏场景">
        <span>数字孪生洪水预警平台</span>
        <strong>数字孪生智能体洪水预警主屏</strong>
        <small>多源感知、风险研判、预警发布与闭环管理</small>
      </section>

      <section className={dashboardStyles.cityKpiRibbon}>
        {pageKpis.map((item) => (
          <article key={item.label} className={dashboardStyles.cityKpiCard}>
            <i aria-hidden="true" />
            <div>
              <span>{item.label}</span>
              <strong className={item.accent}>{item.value}</strong>
              <small>{item.detail}</small>
            </div>
          </article>
        ))}
      </section>

      <div className={styles.workspace}>
        <aside className={styles.leftRail}>
        <div className={styles.panelTitle}>
          <span>当前聚焦对象</span>
          <h2>{focusSummary?.name ?? "等待对象"}</h2>
        </div>

        <article className={styles.focusCard}>
          <div>
            <span>风险等级</span>
            <strong>{riskLabel(selectedRiskLevel)}</strong>
          </div>
          <div>
            <span>所属片区</span>
            <strong>{focusSummary?.village ?? "未选择"}</strong>
          </div>
          <p>{focusSummary?.summary ?? "请在右侧输入对象问题，系统会联动中间三维地图定位对象。"}</p>
        </article>

        <div className={styles.promptPanel}>
          <div className={styles.panelTitle}>
            <span>提问建议</span>
            <h3>可直接点击</h3>
          </div>
          {PROMPT_SUGGESTIONS.map((prompt) => (
            <button key={prompt} type="button" onClick={() => void onAsk(prompt)}>
              {prompt}
            </button>
          ))}
        </div>

        <div className={styles.objectList}>
          <div className={styles.panelTitle}>
            <span>演示对象</span>
            <h3>学校 / 医院 / 重点人群</h3>
          </div>
          {focusObjects.slice(0, 6).map((item) => (
            <button key={item.object_id} type="button" onClick={() => void onSelectObject(item.object_id)}>
              <strong>{item.name}</strong>
              <span>{riskLabel(item.risk_level)} / {item.time_to_impact_minutes} 分钟</span>
            </button>
          ))}
        </div>
        </aside>

        <main className={styles.mapStage}>
        <Suspense
          fallback={
            <div className={styles.canvasFallback}>
              <strong>正在加载三维地图</strong>
              <p>正在准备城市建筑、对象点位和问答联动路线。</p>
            </div>
          }
        >
          <DigitalTwinCesiumCanvas
            layers={mapLayers}
            dialogFocusObjectId={dialogFocusObjectId}
            dialogFocusSerial={dialogEntries.length}
            routeHighlightObjectId={routeHighlightObjectId}
            selectedRiskLevel={selectedRiskLevel}
            onSelectObject={onSelectObject}
          />
        </Suspense>
        </main>

        <aside className={styles.chatRail}>
        <div className={styles.panelTitle}>
          <span>智能体问答</span>
          <h2>现场追问助手</h2>
        </div>

        <div className={styles.messageList}>
          {dialogEntries.length ? (
            dialogEntries.map((entry) => (
              <article key={entry.id} className={entry.role === "assistant" ? styles.assistantMessage : styles.userMessage}>
                <span>{entry.role === "assistant" ? "智能体" : "指挥员"}</span>
                <p>{entry.content}</p>
                {entry.response?.recommended_actions.length ? (
                  <div className={styles.actionList}>
                    {entry.response.recommended_actions.slice(0, 3).map((action) => (
                      <small key={action}>{action}</small>
                    ))}
                  </div>
                ) : null}
              </article>
            ))
          ) : (
            <div className={styles.emptyChat}>
              <strong>输入一个对象问题开始演示</strong>
              <p>例如“李奶奶怎么转移”或“医院急诊入口怎么保障”。</p>
            </div>
          )}
        </div>

        <form className={styles.composer} onSubmit={submitQuestion}>
          <textarea
            aria-label="智能体问答输入框"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="请输入要追问的对象、风险或避险路线..."
          />
          <button type="submit" disabled={busy || !input.trim()}>
            {busy ? "研判中" : "发送问题"}
          </button>
        </form>
        </aside>
      </div>
    </section>
  );
}
