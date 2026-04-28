import { AnimatePresence, motion } from "framer-motion";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import styles from "./App.module.css";
import { AppShell } from "./components/AppShell";
import { CommandCenterPage } from "./components/CommandCenterPage";
import { DigitalTwinImpactScreen } from "./components/DigitalTwinImpactScreen";
import { ExecutionFlowBoard } from "./components/ExecutionFlowBoard";
import { GlobalRegionalProposalDialog } from "./components/GlobalRegionalProposalDialog";
import { MetricStrip } from "./components/MetricStrip";
import { MultiAgentDesk } from "./components/MultiAgentDesk";
import { AdvisoryCard } from "./components/OperationPanel";
import { RegionalAnalysisPackageHistoryPanel } from "./components/RegionalAnalysisPackageHistoryPanel";
import { RegionalProposalHistoryPanel } from "./components/RegionalProposalHistoryPanel";
import { ReliabilityAuditDesk } from "./components/ReliabilityAuditDesk";
import {
  AccessPolicyNotice,
  actionRequiredRoleText,
  operatorRoleText,
  SecurityDesk,
} from "./components/SecurityDesk";
import { SignalTimeline } from "./components/SignalTimeline";
import { ToolExecutionSummary } from "./components/ToolExecutionSummary";
import {
  bootStateText,
  entityText,
  entityTypeOptions,
  executionStatusText,
  healthStateText,
  pageMeta,
  quickPrompts,
  resourceFields,
  riskText,
  travelModeOptions,
} from "./config/consoleConfig";
import { useAgentTwinConsole } from "./hooks/useAgentTwinConsole";
import { AgentsPage } from "./pages/AgentsPage";
import { DataPage } from "./pages/DataPage";
import { OperationsPage } from "./pages/OperationsPage";
import { ReliabilityPage } from "./pages/ReliabilityPage";
import { buildAgentDivergenceRows } from "./state/agentTwinSelectors";
import panelStyles from "./styles/shared-panels.module.css";
import {
  appShellText,
  buildAgentTimelineFallback,
  buildExecutionFlowStats,
  executionFlowText,
  formatTrendLabel,
  formatPendingMetricHint,
  overviewMetricText,
  operationsPageText,
} from "./lib/appText";
import {
  cacheStatusText,
  completenessText,
  datasetJobStatusText,
  datasetPanelText,
  formatDatasetAction,
  formatDatasetSourceCategory,
  formatDatasetStep,
  formatParserKind,
} from "./lib/datasetUiText";
import { createBlankProfile, createBlankResourceStatus } from "./features/dataManagement/dataModels";
import { normalizeAgentTerminology } from "./lib/agentUiText";
import {
  formatAgentTaskEventType,
  formatExecutionMode,
  formatCorpusType,
  formatMobilityConstraints,
  formatNotificationPreferences,
  formatVulnerabilityTags,
  parseMobilityConstraints,
  parseNotificationPreferences,
  parseVulnerabilityTags,
  formatProposalStreamStatus,
  formatRegionalActionType,
  formatTriggerType,
} from "./lib/displayText";
import {
  coerceDrafts,
  coerceLogs,
  coerceStrings,
  coerceTemplates,
  formatPercent,
  formatTimestamp,
  joinCsv,
  parseCsv,
  severityText,
} from "./lib/consoleFormatting";
import type {
  DatasetPipelineStatusView,
  EntityImpactView,
  EntityProfile,
  OperatorRole,
  RAGDocument,
  ResourceStatus,
  ResourceStatusView,
  TravelMode,
} from "./types/api";

function completenessClass(status: string) {
  return {
    parsed: styles.statusApproved,
    cached: styles.statusPending,
    partial_cached: styles.executionTimeout,
    manifest_only: styles.executionSkipped,
    missing: styles.statusRejected,
  }[status] ?? styles.executionSkipped;
}

function metric(label: string, value: string, hint?: string) {
  return (
    <div className={styles.metricBlock}>
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </div>
  );
}

interface ProfileEditorProps {
  areaId: string;
  profiles: EntityProfile[];
  busy: boolean;
  onSave: (profile: EntityProfile) => Promise<void>;
  onDelete: (entityId: string) => Promise<void>;
  onInspect: (entityId: string) => Promise<void>;
}

function ProfileEditor({ areaId, profiles, busy, onSave, onDelete, onInspect }: ProfileEditorProps) {
  const [selectedProfileId, setSelectedProfileId] = useState<string>("__new__");
  const [draft, setDraft] = useState<EntityProfile>(() => createBlankProfile(areaId));

  useEffect(() => {
    if (selectedProfileId === "__new__") {
      setDraft(createBlankProfile(areaId));
      return;
    }
    const selectedProfile = profiles.find((item) => item.entity_id === selectedProfileId);
    if (selectedProfile) {
      setDraft(selectedProfile);
      return;
    }
    setSelectedProfileId("__new__");
    setDraft(createBlankProfile(areaId));
  }, [areaId, profiles, selectedProfileId]);

  const isNewProfile = selectedProfileId === "__new__";

  return (
    <div className={styles.adminCard}>
      <div className={styles.adminCardHeader}>
        <div>
          <p className={styles.sectionLabel}>鐢诲儚绠＄悊</p>
          <h3>运行时对象档案管理</h3>
        </div>
        <button type="button" className={styles.secondaryButton} onClick={() => setSelectedProfileId("__new__")} disabled={busy}>
          鏂板缓鐢诲儚
        </button>
      </div>

      <div className={styles.profileChipList}>
        {profiles.map((profile) => (
          <button
            key={profile.entity_id}
            type="button"
            className={`${styles.profileChip} ${selectedProfileId === profile.entity_id ? styles.profileChipActive : ""}`}
            onClick={() => setSelectedProfileId(profile.entity_id)}
          >
            <strong>{profile.name}</strong>
            <span>{entityText[profile.entity_type]} / {profile.village}</span>
          </button>
        ))}
      </div>

      <div className={styles.formGrid}>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>瀵硅薄缂栧彿</span>
          <input
            aria-label="admin-profile-entity-id"
            className={styles.fieldInput}
            value={draft.entity_id}
            onChange={(event) => setDraft((current) => ({ ...current, entity_id: event.target.value.trim() }))}
            disabled={busy || !isNewProfile}
            placeholder="灞呮皯_婕旂ず_001"
          />
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>鍚嶇О</span>
          <input
            aria-label="admin-profile-name"
            className={styles.fieldInput}
            value={draft.name}
            onChange={(event) => setDraft((current) => ({ ...current, name: event.target.value }))}
            disabled={busy}
            placeholder="瀵硅薄鍚嶇О"
          />
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>瀵硅薄绫诲瀷</span>
          <select
            aria-label="admin-profile-entity-type"
            className={styles.fieldInput}
            value={draft.entity_type}
            onChange={(event) => setDraft((current) => ({ ...current, entity_type: event.target.value as EntityProfile["entity_type"] }))}
            disabled={busy}
          >
            {entityTypeOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>所属村镇</span>
          <input
            aria-label="admin-profile-village"
            className={styles.fieldInput}
            value={draft.village}
            onChange={(event) => setDraft((current) => ({ ...current, village: event.target.value }))}
            disabled={busy}
            placeholder="片区名称"
          />
        </label>
        <label className={styles.fieldBlockFull}>
          <span className={styles.operationLabel}>位置说明</span>
          <input
            aria-label="admin-profile-location-hint"
            className={styles.fieldInput}
            value={draft.location_hint}
            onChange={(event) => setDraft((current) => ({ ...current, location_hint: event.target.value }))}
            disabled={busy}
            placeholder="如学校沿河一侧、转移通道靠近桥梁"
          />
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>当前人数</span>
          <input
            aria-label="admin-profile-current-occupancy"
            className={styles.fieldInput}
            type="number"
            min={0}
            value={draft.current_occupancy}
            onChange={(event) => setDraft((current) => ({ ...current, current_occupancy: Number(event.target.value) }))}
            disabled={busy}
          />
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>常住人数</span>
          <input
            aria-label="admin-profile-resident-count"
            className={styles.fieldInput}
            type="number"
            min={0}
            value={draft.resident_count}
            onChange={(event) => setDraft((current) => ({ ...current, resident_count: Number(event.target.value) }))}
            disabled={busy}
          />
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>转移方式</span>
          <select
            aria-label="admin-profile-transport-mode"
            className={styles.fieldInput}
            value={draft.preferred_transport_mode}
            onChange={(event) => setDraft((current) => ({ ...current, preferred_transport_mode: event.target.value as TravelMode }))}
            disabled={busy}
          >
            {travelModeOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label className={styles.fieldBlockFull}>
          <span className={styles.operationLabel}>脆弱性标签</span>
          <input
            aria-label="admin-profile-vulnerability-tags"
            className={styles.fieldInput}
            value={formatVulnerabilityTags(draft.vulnerability_tags)}
            onChange={(event) => setDraft((current) => ({ ...current, vulnerability_tags: parseVulnerabilityTags(event.target.value) }))}
            disabled={busy}
            placeholder="高龄, 轮椅, 冷链物资"
          />
        </label>
        <label className={styles.fieldBlockFull}>
          <span className={styles.operationLabel}>行动限制</span>
          <input
            aria-label="admin-profile-mobility-constraints"
            className={styles.fieldInput}
            value={formatMobilityConstraints(draft.mobility_constraints)}
            onChange={(event) => setDraft((current) => ({ ...current, mobility_constraints: parseMobilityConstraints(event.target.value) }))}
            disabled={busy}
            placeholder="行动不便, 夜间转移困难"
          />
        </label>
        <label className={styles.fieldBlockFull}>
          <span className={styles.operationLabel}>通知偏好</span>
          <input
            aria-label="admin-profile-notification-preferences"
            className={styles.fieldInput}
            value={formatNotificationPreferences(draft.notification_preferences)}
            onChange={(event) => setDraft((current) => ({ ...current, notification_preferences: parseNotificationPreferences(event.target.value) }))}
            disabled={busy}
            placeholder="短信, 电话, 值班播报"
          />
        </label>
        <label className={styles.fieldBlockFull}>
          <span className={styles.operationLabel}>关键资产</span>
          <input
            aria-label="admin-profile-key-assets"
            className={styles.fieldInput}
            value={joinCsv(draft.key_assets)}
            onChange={(event) => setDraft((current) => ({ ...current, key_assets: parseCsv(event.target.value) }))}
            disabled={busy}
            placeholder="冷库, 发电机房"
          />
        </label>
        <label className={styles.fieldBlockFull}>
          <span className={styles.operationLabel}>库存摘要</span>
          <textarea
            aria-label="admin-profile-inventory-summary"
            className={styles.fieldTextarea}
            value={draft.inventory_summary}
            onChange={(event) => setDraft((current) => ({ ...current, inventory_summary: event.target.value }))}
            disabled={busy}
            rows={3}
          />
        </label>
        <label className={styles.fieldBlockFull}>
          <span className={styles.operationLabel}>连续运行要求</span>
          <textarea
            aria-label="admin-profile-continuity-requirement"
            className={styles.fieldTextarea}
            value={draft.continuity_requirement}
            onChange={(event) => setDraft((current) => ({ ...current, continuity_requirement: event.target.value }))}
            disabled={busy}
            rows={3}
          />
        </label>
      </div>

      <div className={styles.operationActions}>
        {!isNewProfile && draft.entity_id ? (
          <button type="button" className={styles.secondaryButton} aria-label="admin-delete-profile" disabled={busy} onClick={() => void onDelete(draft.entity_id)}>
            删除
          </button>
        ) : null}
        {draft.entity_id ? (
          <button type="button" className={styles.secondaryButton} aria-label="admin-inspect-profile-impact" disabled={busy} onClick={() => void onInspect(draft.entity_id)}>
            查看影响
          </button>
        ) : null}
        <button type="button" className={styles.primaryButton} aria-label="admin-save-profile" disabled={busy || !draft.entity_id || !draft.name || !draft.village} onClick={() => void onSave(draft)}>
          {isNewProfile ? "创建画像" : "保存画像"}
        </button>
      </div>
    </div>
  );
}

interface ResourceStatusEditorProps {
  title: string;
  label: string;
  view: ResourceStatusView | null;
  areaId: string;
  busy: boolean;
  saveLabel: string;
  onSave: (resourceStatus: ResourceStatus) => Promise<void>;
  onClear?: () => Promise<void>;
}

function ResourceStatusEditor({ title, label, view, areaId, busy, saveLabel, onSave, onClear }: ResourceStatusEditorProps) {
  const [draft, setDraft] = useState<ResourceStatus>(() => createBlankResourceStatus(areaId));

  useEffect(() => {
    setDraft(view?.resource_status ?? createBlankResourceStatus(areaId));
  }, [areaId, view]);

  return (
    <div className={styles.adminCard}>
      <div className={styles.adminCardHeader}>
        <div>
          <p className={styles.sectionLabel}>{label}</p>
          <h3>{title}</h3>
        </div>
        {view ? <span className={styles.scopeBadge}>{view.scope.split("_").join(" ")}</span> : null}
      </div>

      <div className={styles.resourceGrid}>
        {resourceFields.map((field) => (
          <label key={field.key} className={field.type === "textarea" ? styles.fieldBlockFull : styles.fieldBlock}>
            <span className={styles.operationLabel}>{field.label}</span>
            {field.type === "textarea" ? (
              <textarea
                aria-label={`resource-${label}-${field.key}`}
                className={styles.fieldTextarea}
                value={String(draft[field.key] ?? "")}
                onChange={(event) => setDraft((current) => ({ ...current, [field.key]: event.target.value }))}
                rows={3}
                disabled={busy}
              />
            ) : (
              <input
                aria-label={`resource-${label}-${field.key}`}
                className={styles.fieldInput}
                type="number"
                min={0}
                value={Number(draft[field.key] ?? 0)}
                onChange={(event) => setDraft((current) => ({ ...current, [field.key]: Number(event.target.value) }))}
                disabled={busy}
              />
            )}
          </label>
        ))}
      </div>

      <div className={styles.operationActions}>
        {onClear ? (
          <button type="button" className={styles.secondaryButton} aria-label={`clear-${label}-resource-status`} disabled={busy} onClick={() => void onClear()}>
            娓呴櫎瑕嗙洊
          </button>
        ) : null}
        <button type="button" className={styles.primaryButton} aria-label={`save-${label}-resource-status`} disabled={busy} onClick={() => void onSave(draft)}>
          {saveLabel}
        </button>
      </div>
    </div>
  );
}

interface RagImportPanelProps {
  documents: RAGDocument[];
  busy: boolean;
  status: string | null;
  onImport: (documents: RAGDocument[]) => Promise<void>;
  onReload: () => Promise<void>;
}

function RagImportPanel({ documents, busy, status, onImport, onReload }: RagImportPanelProps) {
  const [payload, setPayload] = useState(
    JSON.stringify(
      [
        {
          doc_id: "policy_demo_route_clearance",
          corpus: "policy",
          title: "涓存椂閫氳娓呴殰瑙勫垯",
          content: "当学校周边积水超过 15 厘米且接送车辆通行受限时，应优先评估步行疏散与车辆绕行方案。",
          metadata: {
            updated_at: "2026-04-02T08:00:00Z",
            tags: ["school", "transport"],
          },
        },
      ],
      null,
      2,
    ),
  );
  const [inputError, setInputError] = useState<string | null>(null);

  async function handleImport() {
    setInputError(null);

    try {
      const parsed = JSON.parse(payload) as unknown;
      const importedDocuments = Array.isArray(parsed)
        ? (parsed as RAGDocument[])
        : typeof parsed === "object" && parsed !== null && Array.isArray((parsed as { documents?: unknown }).documents)
          ? ((parsed as { documents: RAGDocument[] }).documents)
          : null;

      if (!importedDocuments) {
        throw new Error("导入 JSON 必须是数组，或包含 documents 数组字段的对象。");
      }

      await onImport(importedDocuments);
    } catch (error) {
      setInputError(error instanceof Error ? error.message : "JSON 解析失败。");
    }
  }

  return (
    <div className={styles.adminCard}>
      <div className={styles.adminCardHeader}>
        <div>
          <p className={styles.sectionLabel}>知识库导入</p>
          <h3>RAG 文档管理</h3>
        </div>
        <div className={styles.operationCounts}>
          <span>{documents.length} 浠借繍琛屾湡鏂囨。</span>
        </div>
      </div>

      <label className={styles.fieldBlockFull}>
        <span className={styles.operationLabel}>瀵煎叆鍐呭</span>
        <textarea
          aria-label="rag-import-json"
          className={styles.codeTextarea}
          value={payload}
          onChange={(event) => setPayload(event.target.value)}
          rows={10}
          disabled={busy}
        />
      </label>

      {inputError ? <p className={styles.inlineError}>{inputError}</p> : null}
      {status ? <p className={styles.inlineStatus}>{status}</p> : null}

      <div className={styles.operationActions}>
        <button type="button" className={styles.secondaryButton} aria-label="rag-reload" disabled={busy} onClick={() => void onReload()}>
          重载运行期文档
        </button>
        <button type="button" className={styles.primaryButton} aria-label="rag-import-submit" disabled={busy} onClick={() => void handleImport()}>
          瀵煎叆 JSON
        </button>
      </div>

      <div className={styles.documentList}>
        {documents.length ? (
          documents.slice(0, 6).map((document) => (
            <article key={document.doc_id} className={styles.documentCard}>
              <div className={styles.evidenceMeta}>
                <span>{formatCorpusType(document.corpus)}</span>
                <span>{document.doc_id}</span>
              </div>
              <h4>{document.title}</h4>
              <p>{document.content}</p>
            </article>
          ))
        ) : (
          <p className={styles.emptyState}>当前还没有可展示的知识库文档。</p>
        )}
      </div>
    </div>
  );
}

interface DatasetPipelinePanelProps {
  datasetStatus: DatasetPipelineStatusView | null;
  busy: boolean;
  status: string | null;
  onFetch: (download?: boolean) => Promise<void>;
  onRetrySource: (sourceId: string) => Promise<void>;
  onBuild: (download?: boolean, syncDemoDb?: boolean) => Promise<void>;
  onValidate: () => Promise<void>;
  onSync: () => Promise<void>;
}

function DatasetPipelinePanel({
  datasetStatus,
  busy,
  status,
  onFetch,
  onRetrySource,
  onBuild,
  onValidate,
  onSync,
}: DatasetPipelinePanelProps) {
  const validation = datasetStatus?.latest_validation ?? {};
  const buildSummary = datasetStatus?.latest_build_summary ?? {};
  const fetchSummary = datasetStatus?.latest_fetch_summary ?? {};
  const recentFetchDetails = datasetStatus?.latest_download_log ?? [];
  const activeJob = datasetStatus?.active_job ?? null;

  return (
    <div className={styles.adminCard}>
      <div className={styles.adminCardHeader}>
        <div>
          <p className={styles.sectionLabel}>{datasetPanelText.pipelineSectionLabel}</p>
          <h3>{datasetPanelText.pipelineTitle}</h3>
        </div>
        <div className={styles.operationCounts}>
          <span>{datasetStatus?.cached_source_count ?? 0}/{datasetStatus?.source_count ?? 0} 个数据源已缓存</span>
          <span>{`${datasetStatus?.cached_file_count ?? 0} ${datasetPanelText.cachedFileUnit}`}</span>
        </div>
      </div>

      {status ? <p className={styles.inlineStatus}>{status}</p> : null}

      {activeJob ? (
        <div className={styles.auditPanel}>
          <div className={styles.panelHeader}>
            <div>
              <p className={styles.sectionLabel}>{datasetPanelText.currentTaskLabel}</p>
              <h3>{formatDatasetAction(activeJob.action)}</h3>
            </div>
            <div className={styles.operationCounts}>
              <span>{datasetJobStatusText(activeJob.status)}</span>
              <span>{activeJob.progress_percent}%</span>
            </div>
          </div>
          <div
            aria-label="dataset-active-progress"
            style={{
              width: "100%",
              height: 8,
              background: "rgba(15, 23, 42, 0.08)",
              borderRadius: 999,
              overflow: "hidden",
              marginBottom: 12,
            }}
          >
            <div
              style={{
                width: `${Math.max(0, Math.min(100, activeJob.progress_percent))}%`,
                height: "100%",
                background: "linear-gradient(90deg, #0f766e, #f59e0b)",
                transition: "width 180ms ease",
              }}
            />
          </div>
          <p className={styles.routeMeta}>
            {formatDatasetStep(activeJob.current_step)} | {activeJob.message || activeJob.result_summary || datasetPanelText.waitingProgress}
          </p>
        </div>
      ) : null}

      <div className={styles.agentStatusStrip}>
        {metric(datasetPanelText.rawCacheLabel, `${datasetStatus?.cached_file_count ?? 0}`, datasetStatus?.raw_dir)}
        {metric(datasetPanelText.normalizedOutputLabel, `${datasetStatus?.normalized_files.length ?? 0}`, datasetStatus?.normalized_dir)}
        {metric(datasetPanelText.bootstrapFileLabel, `${datasetStatus?.bootstrap_files.length ?? 0}`, datasetStatus?.bootstrap_dir)}
        {metric(datasetPanelText.fetchProgressLabel, `${Number(fetchSummary.progress_percent ?? 0)}%`, `${Number(fetchSummary.downloaded_artifact_count ?? 0)}/${Number(fetchSummary.artifact_count ?? 0)} ${datasetPanelText.cachedFileUnit}`)}
        {metric(datasetPanelText.validationHitLabel, `${Number(validation.rag_query_hit_count ?? 0)}`, datasetPanelText.validationHitHint)}
      </div>

      <div className={styles.operationActions}>
        <button type="button" className={styles.secondaryButton} aria-label="dataset-fetch-sources" disabled={busy} onClick={() => void onFetch(true)}>
          {datasetPanelText.fetchSourcesAction}
        </button>
        <button type="button" className={styles.secondaryButton} aria-label="dataset-validate" disabled={busy} onClick={() => void onValidate()}>
          {datasetPanelText.validateAction}
        </button>
        <button type="button" className={styles.secondaryButton} aria-label="dataset-sync-db" disabled={busy} onClick={() => void onSync()}>
          {datasetPanelText.syncAction}
        </button>
        <button type="button" className={styles.primaryButton} aria-label="dataset-build-sync" disabled={busy} onClick={() => void onBuild(false, true)}>
          {datasetPanelText.buildAndSyncAction}
        </button>
      </div>

      <div className={styles.memoryList}>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.latestValidationLabel}</span>
          <p>{datasetStatus ? `${Number(validation.shelter_count ?? 0)} 个避难点 | ${Number(validation.road_count ?? 0)} 条道路 | ${Number(validation.entity_profile_count ?? 0)} 份对象档案` : datasetPanelText.noDatasetStatusAlt}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.latestBuildLabel}</span>
          <p>{Object.keys(buildSummary).length ? Object.keys(buildSummary).join(" | ") : datasetPanelText.noBuildSummaryAlt}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.latestFetchLabel}</span>
          <p>{datasetStatus ? `成功 ${Number(fetchSummary.downloaded_artifact_count ?? 0)} / 失败 ${Number(fetchSummary.failed_artifact_count ?? 0)}` : datasetPanelText.noFetchSummaryAlt}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.runtimeKnowledgeBaseLabel}</span>
          <p>{datasetStatus?.runtime_rag_path ?? "--"}</p>
        </div>
      </div>

      <div className={styles.documentList}>
        {datasetStatus?.sources.length ? (
          datasetStatus.sources.map((source) => (
            <article key={source.source_id} className={styles.documentCard}>
              <div className={styles.evidenceMeta}>
                <span>{formatDatasetSourceCategory(source.category)}</span>
                <span className={completenessClass(source.completeness_status)}>{completenessText(source.completeness_status)}</span>
              </div>
              <h4>{source.title}</h4>
              <p>{source.notes}</p>
              <p className={styles.routeMeta}>
                {source.downloaded_artifact_count}/{source.artifact_count} ${datasetPanelText.cachedFileUnit} | {source.progress_percent}% | {formatParserKind(source.parser_kind)}
              </p>
              {source.last_error ? <p className={styles.emptyState}>{source.last_error}</p> : null}
              <small>{source.cached_files.length ? source.cached_files.slice(0, 2).join(" | ") : datasetPanelText.noCachedFiles}</small>
              {source.retryable ? (
                <div className={styles.bulkToolbar}>
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    aria-label={`dataset-retry-source-${source.source_id}`}
                    disabled={busy}
                    onClick={() => void onRetrySource(source.source_id)}
                  >
                    {datasetPanelText.fetchSourceAction}
                  </button>
                </div>
              ) : null}
            </article>
          ))
        ) : (
          <p className={styles.emptyState}>{datasetPanelText.noSourceStatusAlt}</p>
        )}
      </div>

      <div className={styles.auditPanel}>
        <div className={styles.panelHeader}>
          <div>
            <span>{`${recentFetchDetails.length} 条记录`}</span>
            <span>{datasetStatus?.failed_source_count ?? 0} 个失败源</span>
          </div>
          <div className={styles.operationCounts}>
            <span>{`${recentFetchDetails.length} 条记录`}</span>
            <span>{datasetStatus?.failed_source_count ?? 0} 个失败源</span>
          </div>
        </div>
        {recentFetchDetails.length ? (
          <div className={styles.auditList}>
            {recentFetchDetails.slice(0, 8).map((entry, index) => (
              <article key={`${String(entry.source_id ?? "entry")}-${index}`} className={styles.auditCard}>
                <div className={styles.evidenceMeta}>
                  <span>{String(entry.source_id ?? "--")}</span>
                  <span>{cacheStatusText(String(entry.status ?? "--"))}</span>
                </div>
                <h4>{String(entry.artifact ?? datasetPanelText.defaultArtifactName)}</h4>
                <p>{String(entry.target_path ?? entry.url ?? datasetPanelText.noTargetPath)}</p>
                {entry.error ? <small>{String(entry.error)}</small> : <small>{String(entry.fetched_at ?? "--")}</small>}
              </article>
            ))}
          </div>
        ) : (
          <p className={styles.emptyState}>{datasetPanelText.noDownloadLogAlt}</p>
        )}
      </div>
    </div>
  );
}

interface DatasetOperationsPanelProps extends DatasetPipelinePanelProps {
  onCancelJob: (jobId: string) => Promise<void>;
  onRetryJob: (jobId: string) => Promise<void>;
}

function DatasetOperationsPanel({
  datasetStatus,
  busy,
  status,
  onFetch,
  onRetrySource,
  onBuild,
  onValidate,
  onSync,
  onCancelJob,
  onRetryJob,
}: DatasetOperationsPanelProps) {
  const validation = datasetStatus?.latest_validation ?? {};
  const buildSummary = datasetStatus?.latest_build_summary ?? {};
  const fetchSummary = datasetStatus?.latest_fetch_summary ?? {};
  const recentFetchDetails = datasetStatus?.latest_download_log ?? [];
  const activeJob = datasetStatus?.active_job ?? null;
  const recentJobs = datasetStatus?.recent_jobs ?? [];
  const rawHealth = datasetStatus?.raw_cache_health ?? [];
  const canCancelActiveJob = Boolean(activeJob && ["pending", "running", "cancel_requested"].includes(activeJob.status));

  return (
    <div className={styles.adminCard}>
      <div className={styles.adminCardHeader}>
        <div>
          <p className={styles.sectionLabel}>{datasetPanelText.pipelineSectionLabel}</p>
          <h3>{datasetPanelText.operationsTitle}</h3>
        </div>
        <div className={styles.operationCounts}>
          <span>{datasetStatus?.cached_source_count ?? 0}/{datasetStatus?.source_count ?? 0} 个数据源已缓存</span>
          <span>{`${datasetStatus?.cached_file_count ?? 0} ${datasetPanelText.cachedFileUnit}`}</span>
        </div>
      </div>

      {status ? <p className={styles.inlineStatus}>{status}</p> : null}

      {activeJob ? (
        <div className={styles.auditPanel}>
          <div className={styles.panelHeader}>
            <div>
              <p className={styles.sectionLabel}>{datasetPanelText.currentTaskLabel}</p>
              <h3>{formatDatasetAction(activeJob.action)}</h3>
            </div>
            <div className={styles.operationCounts}>
              <span>{datasetJobStatusText(activeJob.status)}</span>
              <span>{activeJob.progress_percent}%</span>
            </div>
          </div>
          <div
            aria-label="dataset-active-progress"
            style={{
              width: "100%",
              height: 8,
              background: "rgba(15, 23, 42, 0.08)",
              borderRadius: 999,
              overflow: "hidden",
              marginBottom: 12,
            }}
          >
            <div
              style={{
                width: `${Math.max(0, Math.min(100, activeJob.progress_percent))}%`,
                height: "100%",
                background: "linear-gradient(90deg, #0f766e, #f59e0b)",
                transition: "width 180ms ease",
              }}
            />
          </div>
          <p className={styles.routeMeta}>
            {formatDatasetStep(activeJob.current_step)} | {activeJob.message || activeJob.result_summary || datasetPanelText.waitingProgressAlt}
          </p>
          <div className={styles.memoryList}>
            <div>
              <span className={styles.operationLabel}>{datasetPanelText.attemptCountLabel}</span>
              <p>{activeJob.attempt_count ?? 0}/{activeJob.max_attempts ?? 1}</p>
            </div>
            <div>
              <span className={styles.operationLabel}>{datasetPanelText.retryCountLabel}</span>
              <p>{activeJob.retry_count ?? 0}</p>
            </div>
            <div>
              <span className={styles.operationLabel}>{datasetPanelText.autoRetryLabel}</span>
              <p>{activeJob.auto_retry_enabled ? datasetPanelText.enabled : datasetPanelText.disabled}</p>
            </div>
            <div>
              <span className={styles.operationLabel}>{datasetPanelText.retryOriginLabel}</span>
              <p>{activeJob.retry_of_job_id ?? "--"}</p>
            </div>
          </div>
          <div className={styles.bulkToolbar}>
            <button
              type="button"
              className={styles.secondaryButton}
              aria-label={`dataset-cancel-job-${activeJob.job_id}`}
              disabled={busy || !canCancelActiveJob || activeJob.status === "cancel_requested"}
              onClick={() => void onCancelJob(activeJob.job_id)}
            >
              {activeJob.status === "cancel_requested" ? datasetPanelText.cancelRequestedAction : datasetPanelText.cancelJobAction}
            </button>
          </div>
        </div>
      ) : null}

      <div className={styles.agentStatusStrip}>
        {metric(datasetPanelText.rawCacheLabel, `${datasetStatus?.cached_file_count ?? 0}`, datasetStatus?.raw_dir)}
        {metric(datasetPanelText.rawReadinessLabel, datasetStatus?.raw_ready ? datasetPanelText.ready : datasetPanelText.notReady, `${datasetStatus?.raw_completeness_percent ?? 0}%`)}
        {metric(datasetPanelText.normalizedOutputLabel, `${datasetStatus?.normalized_files.length ?? 0}`, datasetStatus?.normalized_dir)}
        {metric(datasetPanelText.bootstrapFileLabel, `${datasetStatus?.bootstrap_files.length ?? 0}`, datasetStatus?.bootstrap_dir)}
        {metric(datasetPanelText.fetchProgressLabel, `${Number(fetchSummary.progress_percent ?? 0)}%`, `${Number(fetchSummary.downloaded_artifact_count ?? 0)}/${Number(fetchSummary.artifact_count ?? 0)} ${datasetPanelText.cachedFileUnit}`)}
        {metric(datasetPanelText.validationHitLabel, `${Number(validation.rag_query_hit_count ?? 0)}`, datasetPanelText.validationHitHint)}
      </div>

      <div className={styles.operationActions}>
        <button type="button" className={styles.secondaryButton} aria-label="dataset-fetch-sources" disabled={busy} onClick={() => void onFetch(true)}>
          {datasetPanelText.fetchSourcesAction}
        </button>
        <button type="button" className={styles.secondaryButton} aria-label="dataset-validate" disabled={busy} onClick={() => void onValidate()}>
          {datasetPanelText.validateAction}
        </button>
        <button type="button" className={styles.secondaryButton} aria-label="dataset-sync-db" disabled={busy} onClick={() => void onSync()}>
          {datasetPanelText.syncAction}
        </button>
        <button type="button" className={styles.primaryButton} aria-label="dataset-build-sync" disabled={busy} onClick={() => void onBuild(false, true)}>
          {datasetPanelText.buildAndSyncAction}
        </button>
      </div>

      <div className={styles.memoryList}>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.latestValidationLabel}</span>
          <p>{datasetStatus ? `${Number(validation.shelter_count ?? 0)} 个避难点 | ${Number(validation.road_count ?? 0)} 条道路 | ${Number(validation.entity_profile_count ?? 0)} 份对象档案` : datasetPanelText.noDatasetStatus}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.latestBuildLabel}</span>
          <p>{Object.keys(buildSummary).length ? Object.keys(buildSummary).join(" | ") : datasetPanelText.noBuildSummary}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.latestFetchLabel}</span>
          <p>{datasetStatus ? `成功 ${Number(fetchSummary.downloaded_artifact_count ?? 0)} / 失败 ${Number(fetchSummary.failed_artifact_count ?? 0)}` : datasetPanelText.noFetchSummary}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.missingRequiredSourcesLabel}</span>
          <p>{datasetStatus?.missing_required_sources?.length ? datasetStatus.missing_required_sources.join(" | ") : datasetPanelText.none}</p>
        </div>
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.runtimeKnowledgeBaseLabel}</span>
          <p>{datasetStatus?.runtime_rag_path ?? "--"}</p>
        </div>
      </div>

      <div className={styles.auditPanel}>
        <div className={styles.panelHeader}>
          <div>
            <p className={styles.sectionLabel}>{datasetPanelText.rawHealthSectionLabel}</p>
            <h3>{datasetPanelText.rawHealthTitle}</h3>
          </div>
          <div className={styles.operationCounts}>
            <span>{rawHealth.filter((source) => source.parsed).length} 个已解析</span>
            <span>{`${datasetStatus?.missing_required_sources?.length ?? 0} 个缺失`}</span>
          </div>
        </div>
        {rawHealth.length ? (
          <div className={styles.auditList}>
            {rawHealth.map((source) => (
              <article key={source.source_id} className={styles.auditCard}>
                <div className={styles.evidenceMeta}>
                  <span>{source.title}</span>
                  <span className={completenessClass(source.completeness_status)}>{completenessText(source.completeness_status)}</span>
                </div>
                <h4>{cacheStatusText(source.cache_status)}</h4>
                <p>
                  {source.raw_file_count} 个原始文件 | {source.downloaded_artifact_count} 已下载 | {source.failed_artifact_count} 失败
                </p>
                <small>
                  {source.parsed ? datasetPanelText.parsedReadyText : datasetPanelText.parsedPendingText}
                  {source.missing_artifact_types.length ? ` 缺失：${source.missing_artifact_types.join(", ")}` : ""}
                </small>
                {source.last_error ? <small>{source.last_error}</small> : null}
              </article>
            ))}
          </div>
        ) : (
          <p className={styles.emptyState}>{datasetPanelText.noRawHealth}</p>
        )}
      </div>

      <div className={styles.documentList}>
        {datasetStatus?.sources.length ? (
          datasetStatus.sources.map((source) => (
            <article key={source.source_id} className={styles.documentCard}>
              <div className={styles.evidenceMeta}>
                <span>{formatDatasetSourceCategory(source.category)}</span>
                <span className={completenessClass(source.completeness_status)}>{completenessText(source.completeness_status)}</span>
              </div>
              <h4>{source.title}</h4>
              <p>{source.notes}</p>
              <p className={styles.routeMeta}>
                {source.downloaded_artifact_count}/{source.artifact_count} ${datasetPanelText.cachedFileUnit} | {source.progress_percent}% | {formatParserKind(source.parser_kind)}
              </p>
              {source.last_error ? <p className={styles.emptyState}>{source.last_error}</p> : null}
              <small>
                {source.cached_files.length ? source.cached_files.slice(0, 2).join(" | ") : datasetPanelText.noCachedFiles}
                {source.missing_artifact_types.length ? ` | 缺失：${source.missing_artifact_types.join(", ")}` : ""}
              </small>
              {source.retryable ? (
                <div className={styles.bulkToolbar}>
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    aria-label={`dataset-retry-source-${source.source_id}`}
                    disabled={busy}
                    onClick={() => void onRetrySource(source.source_id)}
                  >
                    {datasetPanelText.retrySourceAction}
                  </button>
                </div>
              ) : null}
            </article>
          ))
        ) : (
          <p className={styles.emptyState}>{datasetPanelText.noSourceStatus}</p>
        )}
      </div>

      <div className={styles.auditPanel}>
        <div className={styles.panelHeader}>
          <div>
            <p className={styles.sectionLabel}>{datasetPanelText.fetchDetailSectionLabel}</p>
            <h3>{datasetPanelText.fetchDetailTitle}</h3>
          </div>
          <div className={styles.operationCounts}>
            <span>{`${recentFetchDetails.length} 条记录`}</span>
            <span>{datasetStatus?.failed_source_count ?? 0} 个失败源</span>
          </div>
        </div>
        {recentFetchDetails.length ? (
          <div className={styles.auditList}>
            {recentFetchDetails.slice(0, 8).map((entry, index) => (
              <article key={`${String(entry.source_id ?? "entry")}-${index}`} className={styles.auditCard}>
                <div className={styles.evidenceMeta}>
                  <span>{String(entry.source_id ?? "--")}</span>
                  <span>{cacheStatusText(String(entry.status ?? "--"))}</span>
                </div>
                <h4>{String(entry.artifact ?? datasetPanelText.defaultArtifactName)}</h4>
                <p>{String(entry.target_path ?? entry.url ?? datasetPanelText.noTargetPath)}</p>
                {entry.error ? <small>{String(entry.error)}</small> : <small>{String(entry.fetched_at ?? "--")}</small>}
              </article>
            ))}
          </div>
        ) : (
          <p className={styles.emptyState}>{datasetPanelText.noDownloadLog}</p>
        )}
      </div>

      <div className={styles.auditPanel}>
        <div className={styles.panelHeader}>
          <div>
            <span>{recentFetchDetails.length} ???</span>
            <span>{datasetStatus?.failed_source_count ?? 0} ????</span>
          </div>
          <div className={styles.operationCounts}>
            <span>{`${recentJobs.length} 条任务`}</span>
            <span>{`${recentJobs.filter((job) => job.status === "failed").length} 条失败`}</span>
          </div>
        </div>
        {recentJobs.length ? (
          <div className={styles.auditList}>
            {recentJobs.map((job) => (
              <article key={job.job_id} className={styles.auditCard}>
                <div className={styles.evidenceMeta}>
                  <span>{job.action}</span>
                  <span>{datasetJobStatusText(job.status)}</span>
                </div>
                <h4>{job.job_id}</h4>
                <p>{job.message || job.result_summary || datasetPanelText.noJobMessage}</p>
                <small>
                  灏濊瘯 {job.attempt_count ?? 0}/{job.max_attempts ?? 1} | 閲嶈瘯 {job.retry_count ?? 0} | 鍚姩浜?{formatTimestamp(job.started_at)}
                </small>
                {job.error ? <small>{job.error}</small> : null}
                <div className={styles.bulkToolbar}>
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    aria-label={`dataset-retry-job-${job.job_id}`}
                    disabled={busy || !["failed", "canceled"].includes(job.status)}
                    onClick={() => void onRetryJob(job.job_id)}
                  >
                    {datasetPanelText.retryJobAction}
                  </button>
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    aria-label={`dataset-cancel-job-history-${job.job_id}`}
                    disabled={busy || !["pending", "running", "cancel_requested"].includes(job.status)}
                    onClick={() => void onCancelJob(job.job_id)}
                  >
                    {datasetPanelText.cancelJobAction}
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className={styles.emptyState}>{datasetPanelText.noJobRecords}</p>
        )}
      </div>
    </div>
  );
}

interface AdminDeskProps {
  areaId: string;
  eventId?: string;
  profiles: EntityProfile[];
  areaResourceStatusView: ResourceStatusView | null;
  eventResourceStatusView: ResourceStatusView | null;
  ragDocuments: RAGDocument[];
  datasetStatus: DatasetPipelineStatusView | null;
  busy: boolean;
  status: string | null;
  canEditRuntimeAdmin: boolean;
  canManageDataset: boolean;
  onSaveProfile: (profile: EntityProfile) => Promise<void>;
  onDeleteProfile: (entityId: string) => Promise<void>;
  onInspectProfile: (entityId: string) => Promise<void>;
  onSaveAreaResources: (resourceStatus: ResourceStatus) => Promise<void>;
  onSaveEventResources: (resourceStatus: ResourceStatus) => Promise<void>;
  onClearEventResources: () => Promise<void>;
  onImportRagDocuments: (documents: RAGDocument[]) => Promise<void>;
  onReloadRagDocuments: () => Promise<void>;
  onFetchDatasetSources: (download?: boolean) => Promise<void>;
  onRetryDatasetSource: (sourceId: string) => Promise<void>;
  onBuildDatasetPackage: (download?: boolean, syncDemoDb?: boolean) => Promise<void>;
  onValidateDatasetPackage: () => Promise<void>;
  onSyncDatasetPackage: () => Promise<void>;
  onCancelDatasetJob: (jobId: string) => Promise<void>;
  onRetryDatasetJob: (jobId: string) => Promise<void>;
}

function AdminDesk(props: AdminDeskProps) {
  return (
    <div className={styles.adminDesk}>
      <div className={styles.panelHeader}>
        <div>
          <p className={styles.sectionLabel}>数据管线运维</p>
          <h2>运行时知识与资源管理</h2>
        </div>
        <div className={styles.operationCounts}>
          <span>{`${props.profiles.length} 份档案`}</span>
          <span>{`${props.ragDocuments.length} 份文档`}</span>
          <span>{`${props.datasetStatus?.cached_file_count ?? 0} ${datasetPanelText.cachedFileUnit}`}</span>
          {props.eventId ? <span>事件级资源视图</span> : null}
        </div>
      </div>
      <AccessPolicyNotice
        title="权限边界说明"
        summary="以下操作会直接修改运行时档案、资源状态和数据管线任务，请在具备相应角色权限时执行。"
        items={[
          {
            label: "运行时档案与知识库维护",
            allowed: props.canEditRuntimeAdmin,
            requiredRole: actionRequiredRoleText.runtime_admin_write,
            description: "允许维护对象档案、导入知识文档并更新运行时参考信息。",
          },
          {
            label: "数据管线抓取与构建",
            allowed: props.canManageDataset,
            requiredRole: actionRequiredRoleText.dataset_manage,
            description: "允许抓取源数据、构建知识包并同步到运行时数据库。",
          },
        ]}
      />

      <div className={styles.adminGrid}>
        <ProfileEditor
          areaId={props.areaId}
          profiles={props.profiles}
          busy={props.busy || !props.canEditRuntimeAdmin}
          onSave={props.onSaveProfile}
          onDelete={props.onDeleteProfile}
          onInspect={props.onInspectProfile}
        />
        <div className={styles.adminStack}>
          <ResourceStatusEditor
            title="区域资源状态"
            label="鍖哄煙"
            view={props.areaResourceStatusView}
            areaId={props.areaId}
            busy={props.busy || !props.canEditRuntimeAdmin}
            saveLabel="保存区域资源状态"
            onSave={props.onSaveAreaResources}
          />
          <ResourceStatusEditor
            title="事件资源状态"
            label="浜嬩欢"
            view={props.eventResourceStatusView}
            areaId={props.areaId}
            busy={props.busy || !props.canEditRuntimeAdmin}
            saveLabel="淇濆瓨浜嬩欢瑕嗙洊"
            onSave={props.onSaveEventResources}
            onClear={props.onClearEventResources}
          />
          <DatasetOperationsPanel
            datasetStatus={props.datasetStatus}
            busy={props.busy || !props.canManageDataset}
            status={props.status}
            onFetch={props.onFetchDatasetSources}
            onRetrySource={props.onRetryDatasetSource}
            onBuild={props.onBuildDatasetPackage}
            onValidate={props.onValidateDatasetPackage}
            onSync={props.onSyncDatasetPackage}
            onCancelJob={props.onCancelDatasetJob}
            onRetryJob={props.onRetryDatasetJob}
          />
        </div>
        <RagImportPanel
          documents={props.ragDocuments}
          busy={props.busy || !props.canEditRuntimeAdmin}
          status={props.status}
          onImport={props.onImportRagDocuments}
          onReload={props.onReloadRagDocuments}
        />
      </div>
    </div>
  );
}

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const consoleState = useAgentTwinConsole();
  const [input, setInput] = useState("");
  const capabilityMap = consoleState.operatorCapabilities?.capabilities ?? {};
  const canEditRuntimeAdmin = Boolean(capabilityMap.runtime_admin_write);
  const canManageDataset = Boolean(capabilityMap.dataset_manage);
  const canControlSupervisor = Boolean(capabilityMap.supervisor_control);
  const canReplayTask = Boolean(capabilityMap.agent_replay);
  const canRunEvaluation = Boolean(capabilityMap.evaluation_run);
  const canRunArchive = Boolean(capabilityMap.archive_run);

  const topRisk = consoleState.twinOverview?.overall_risk_level ?? consoleState.hazardState?.overall_risk_level ?? "None";
  const selectedImpact = consoleState.selectedImpact as EntityImpactView;
  const leadImpact = useMemo(() => consoleState.topImpacts[0] ?? null, [consoleState.topImpacts]);
  const currentPage = pageMeta[location.pathname as keyof typeof pageMeta];

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!input.trim()) return;
    void consoleState.ask(input.trim());
    setInput("");
  }

  function handleTextareaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  if (!currentPage) {
    return <Navigate to="/" replace />;
  }

  const isOverviewPage = location.pathname === "/";
  const isCopilotPage = location.pathname === "/copilot";
  const isOperationsPage = location.pathname === "/operations";
  const isDataPage = location.pathname === "/data";
  const isAgentsPage = location.pathname === "/agents";
  const isReliabilityPage = location.pathname === "/reliability";

  const trendLabel = formatTrendLabel(consoleState.twinOverview?.trend ?? consoleState.hazardState?.trend);
  const highPriorityCount =
    consoleState.twinOverview?.focus_objects.length ??
    consoleState.topImpacts.filter((impact) => impact.risk_level !== "None").length;
  const approvedProposalCount =
    consoleState.twinOverview?.approved_proposal_count ??
    consoleState.proposals.filter((item) => item.status === "approved").length;
  const pendingProposalCount = consoleState.twinOverview?.pending_proposal_count ?? consoleState.pendingProposals.length;
  const warningDraftCount = consoleState.twinOverview?.warning_draft_count ?? 0;
  const latestToolExecutions = consoleState.latestAnswer?.tool_executions ?? [];
  const councilRoles = consoleState.agentCouncil?.roles ?? [];
  const latestWarningDrafts =
    consoleState.warningDrafts.length > 0
      ? consoleState.warningDrafts
      : consoleState.twinOverview?.recent_warning_drafts ?? [];
  const agentDecisionPath = consoleState.agentCouncil?.decision_path ?? [];
  const agentOpenQuestions = consoleState.agentCouncil?.open_questions ?? [];
  const agentBlockedBy = consoleState.agentCouncil?.blocked_by ?? [];
  const recentCouncilResults = consoleState.recentAgentResults.slice(0, 4);
  const evidenceCompareResults = recentCouncilResults.filter(
    (result) => result.evidence_refs.length > 0 || result.missing_slots.length > 0 || result.handoff_recommendations.length > 0,
  );
  const agentDivergenceRows = buildAgentDivergenceRows({
    recentResults: recentCouncilResults,
    sharedMemorySnapshot: consoleState.sharedMemorySnapshot,
    decisionReport: consoleState.decisionReport,
    agentCouncil: consoleState.agentCouncil,
    maxRows: 4,
  });

  const primaryPaths = new Set(["/", "/operations", "/agents"]);
  const navigation = Object.entries(pageMeta)
    .filter(([path]) => primaryPaths.has(path))
    .map(([path, meta]) => ({ path, label: meta.label }));
  const utilityNavigation = Object.entries(pageMeta)
    .filter(([path]) => !primaryPaths.has(path))
    .map(([path, meta]) => ({ path, label: meta.label }));
  const shellCurrentPageLabel = isOverviewPage ? "数字孪生主屏" : currentPage.label;
  const shellCurrentPageTitle = isOverviewPage ? "数字孪生智能体洪水预警系统" : currentPage.title;
  const shellCurrentPageDescription = isOverviewPage ? undefined : currentPage.description;

  const pageMetricItems = isOverviewPage
    ? [
        {
          label: overviewMetricText.riskLabel,
          value: riskText[topRisk],
          hint: consoleState.twinOverview?.event_title ?? consoleState.event?.title ?? overviewMetricText.riskHintFallback,
          tone: topRisk === "Red" || topRisk === "Orange" ? ("risk" as const) : topRisk === "Yellow" ? ("warning" as const) : ("success" as const),
        },
        {
          label: overviewMetricText.trendLabel,
          value: trendLabel,
          hint: overviewMetricText.trendHint,
          tone: topRisk === "Red" || topRisk === "Orange" ? ("warning" as const) : ("neutral" as const),
        },
        {
          label: overviewMetricText.priorityLabel,
          value: `${highPriorityCount || consoleState.topImpacts.length}`,
          hint: overviewMetricText.priorityHint,
          tone: highPriorityCount ? ("warning" as const) : ("neutral" as const),
        },
        {
          label: overviewMetricText.pendingLabel,
          value: `${pendingProposalCount}`,
          hint: warningDraftCount
            ? `已生成 ${warningDraftCount} 条分众预警草稿`
            : formatPendingMetricHint(approvedProposalCount),
          tone: pendingProposalCount ? ("warning" as const) : ("success" as const),
        },
      ]
    : isOperationsPage
        ? [
            {
              label: "待确认动作",
              value: `${consoleState.pendingProposals.length}`,
              hint: "系统正在等待人工确认处置建议。",
              tone: consoleState.pendingProposals.length ? ("warning" as const) : ("success" as const),
            },
            {
              label: "历史建议数",
              value: `${consoleState.regionalProposalHistory.length}`,
              hint: approvedProposalCount ? `${approvedProposalCount} 条建议已批准执行` : "当前还没有形成历史建议。",
              tone: consoleState.regionalProposalHistory.length ? ("success" as const) : ("neutral" as const),
            },
            {
              label: "请示流状态",
              value: formatProposalStreamStatus(consoleState.proposalStreamStatus),
               hint: consoleState.agentStatus?.latest_summary ?? "等待处置链路继续推进。",
              tone: consoleState.proposalStreamStatus === "error" ? ("risk" as const) : consoleState.proposalStreamStatus === "open" ? ("success" as const) : ("warning" as const),
            },
            {
              label: "工具执行次数",
              value: `${latestToolExecutions.length}`,
              hint: latestToolExecutions.length ? "规划链路已触发关键能力调用。" : "当前尚未触发关键能力调用。",
              tone: latestToolExecutions.length ? ("success" as const) : ("neutral" as const),
            },
          ]
        : [
            {
               label: "总体风险",
              value: riskText[topRisk],
               hint: consoleState.event?.title ?? "当前事件",
              tone: topRisk === "Red" || topRisk === "Orange" ? ("risk" as const) : ("success" as const),
            },
            {
              label: "执行状态",
              value: executionStatusText[consoleState.executionStatus],
               hint: consoleState.errorMessage ?? consoleState.adminStatus ?? "运行稳定",
              tone: consoleState.executionStatus === "error" ? ("risk" as const) : consoleState.executionStatus === "running" ? ("warning" as const) : ("success" as const),
            },
          ];

  const priorityItems = consoleState.curatedEntities.map((entity) => {
    const impact = consoleState.entityImpacts[entity.id];
    return {
      id: entity.id,
      name: entity.name,
      typeLabel: entityText[entity.type],
      village: entity.village,
      emphasis: entity.emphasis,
      riskLabel: impact ? riskText[impact.risk_level] : undefined,
      riskTone: impact
        ? ({
            None: "none",
            Blue: "blue",
            Yellow: "yellow",
            Orange: "orange",
            Red: "red",
          }[impact.risk_level] as "none" | "blue" | "yellow" | "orange" | "red")
        : undefined,
    };
  });

  const overviewSignalItems = consoleState.openAlerts.length
    ? consoleState.openAlerts.slice(0, 4).map((alert) => ({
        id: alert.alert_id,
        title: alert.summary,
        detail: alert.details || executionFlowText.alertDetailFallback,
        meta: formatTimestamp(alert.last_seen_at ?? alert.first_seen_at),
        tone:
          alert.severity === "critical"
            ? ("critical" as const)
            : alert.severity === "warning"
              ? ("warning" as const)
              : ("info" as const),
      }))
    : consoleState.supervisorRuns.slice(0, 4).map((run) => ({
        id: run.supervisor_run_id,
        title: formatTriggerType(run.trigger_type),
        detail: run.summary || executionFlowText.supervisorRunFallback,
        meta: formatTimestamp(run.created_at),
        tone: run.status === "failed" ? ("critical" as const) : ("info" as const),
      }));

  const agentTimelineItems = consoleState.agentTimeline.slice(0, 5).map((entry) => ({
    id: entry.entry_id,
    title:
      entry.entry_type === "trigger"
        ? `触发：${formatTriggerType(entry.trigger_type)}`
        : `任务：${formatAgentTaskEventType(entry.task_event_type)}`,
    detail: normalizeAgentTerminology(entry.summary) || buildAgentTimelineFallback(entry.entry_type === "trigger"),
    meta: formatTimestamp(entry.created_at),
    tone:
      entry.entry_type === "trigger"
        ? ("warning" as const)
        : String(entry.payload.status ?? "").toLowerCase() === "failed"
          ? ("critical" as const)
          : ("info" as const),
  }));

  const executionFlowSteps = [
    {
      id: "sense",
      title: executionFlowText.senseTitle,
      summary: consoleState.hazardState
        ? `已汇聚当前事件水情、路网与监测状态，形成 ${riskText[topRisk]} 风险判断。`
        : executionFlowText.noHazardStateSummary,
      detail: consoleState.hazardState
        ? `趋势判断为 ${trendLabel}，可达路段 ${consoleState.hazardState.road_reachability?.length ?? 0} 条。`
        : executionFlowText.noHazardStateDetail,
      status: consoleState.hazardState ? ("complete" as const) : ("pending" as const),
    },
    {
      id: "impact",
      title: executionFlowText.impactTitle,
      summary: leadImpact
        ? `系统已识别 ${leadImpact.entity.name} 为首要影响对象。`
        : executionFlowText.noLeadImpactSummary,
      detail: leadImpact
        ? leadImpact.risk_reason[0] ?? "已结合对象属性、位置和脆弱性形成对象级研判。"
        : executionFlowText.noLeadImpactDetail,
      status: leadImpact ? ("complete" as const) : ("pending" as const),
    },
    {
      id: "plan",
      title: executionFlowText.planTitle,
      summary: consoleState.latestAnswer?.planner_summary ?? executionFlowText.noPlannerSummary,
      detail: consoleState.latestAnswer?.planning_layers_summary?.[0] ?? executionFlowText.noPlannerDetail,
      status: consoleState.latestAnswer ? ("complete" as const) : ("pending" as const),
    },
    {
      id: "tooling",
      title: executionFlowText.toolingTitle,
      summary: latestToolExecutions.length
        ? `本轮已触发 ${latestToolExecutions.length} 次关键能力调用。`
        : executionFlowText.noToolingSummary,
      detail: latestToolExecutions[0]?.output_summary ?? executionFlowText.noToolingDetail,
      status: latestToolExecutions.length ? ("active" as const) : ("pending" as const),
    },
    {
      id: "confirm",
      title: executionFlowText.confirmTitle,
      summary: consoleState.pendingProposals.length
        ? `${consoleState.pendingProposals.length} 条动作等待人工确认。`
        : executionFlowText.noPendingConfirmationSummary,
      detail: approvedProposalCount
        ? `已有 ${approvedProposalCount} 条动作完成批准闭环。`
        : executionFlowText.noPendingConfirmationDetail,
      status: consoleState.pendingProposals.length
        ? ("active" as const)
        : approvedProposalCount
          ? ("complete" as const)
          : ("pending" as const),
    },
  ];

  const analysisPackageItems = [
    ...(consoleState.pendingRegionalAnalysisPackage ? [consoleState.pendingRegionalAnalysisPackage] : []),
    ...consoleState.regionalAnalysisPackageHistory,
  ];
  const proposalHistoryItems = consoleState.regionalProposalHistory;
  const pendingProposalItems = consoleState.regionalProposalQueueSnapshot?.items ?? [];

  return (
    <>
      <GlobalRegionalProposalDialog
        open={!isCopilotPage && consoleState.regionalProposalModalOpen}
        busy={consoleState.isBusy}
        snapshot={consoleState.regionalProposalQueueSnapshot}
        onApprove={(proposalId, note) => consoleState.resolveProposal(proposalId, "approve", note)}
        onReject={(proposalId, note) => consoleState.resolveProposal(proposalId, "reject", note)}
        onSaveDraft={consoleState.updateRegionalProposalDraft}
        onSnooze={consoleState.snoozeRegionalProposalModal}
      />
      <AppShell
        brandTitle="数字孪生智能体洪水预警系统"
        currentPageLabel={shellCurrentPageLabel}
        currentPageTitle={shellCurrentPageTitle}
        currentPageDescription={shellCurrentPageDescription}
        navigation={navigation}
        utilityNavigation={utilityNavigation}
        operatorControl={
          <div className={styles.topbarActions}>
            <label className={styles.fieldBlock}>
              <span>{appShellText.currentRole}</span>
              <select
                className={styles.fieldInput}
                aria-label="topbar-operator-role"
                value={consoleState.operatorRole}
                onChange={(event) => consoleState.setOperatorRole(event.target.value as OperatorRole)}
              >
                {Object.entries(operatorRoleText).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" className={styles.secondaryButton} onClick={() => void consoleState.refresh()}>
              {appShellText.refresh}
            </button>
          </div>
        }
        statusSignals={
          <>
            <div className={panelStyles.statusSignal}>
              <span className={panelStyles.statusSignalLabel}>{appShellText.apiStatus}</span>
              <strong className={panelStyles.statusSignalValue}>{healthStateText[consoleState.healthState]}</strong>
            </div>
            <div className={panelStyles.statusSignal}>
              <span className={panelStyles.statusSignalLabel}>{appShellText.platformStatus}</span>
              <strong className={panelStyles.statusSignalValue}>{bootStateText[consoleState.bootState]}</strong>
            </div>
            <div className={panelStyles.statusSignal}>
              <span className={panelStyles.statusSignalLabel}>{appShellText.supervisorStatus}</span>
              <strong className={panelStyles.statusSignalValue}>
                {consoleState.supervisorLoopStatus?.running ? appShellText.supervisorRunning : appShellText.supervisorStopped}
              </strong>
            </div>
          </>
        }
        metrics={<MetricStrip items={pageMetricItems} />}
      >
        {isCopilotPage ? (
          <>
            <div className={styles.panelFrame}>
              <div className={styles.panelHeaderCompact}>
                <div>
                  <p className={styles.sectionLabel}>对话指挥</p>
                  <h2>通过对话查看研判、请示与总结</h2>
                </div>
              </div>
              <p className={styles.emptyState}>
                在这里可以查看智能体的多轮分析、审批请求、日报和高风险复盘，并继续追问对象风险、联动建议和执行策略。
              </p>
              <div className={styles.routeSummary}>
                <div>
                  <span>使用方式</span>
                  <strong>输入问题或使用快捷提示，系统会结合事件上下文持续生成建议。</strong>
                </div>
              </div>
            </div>
            <CommandCenterPage
              agentStatus={consoleState.agentStatus}
              agentTasks={consoleState.agentTasks}
              dailyReports={consoleState.dailyReports}
              episodeSummaries={consoleState.episodeSummaries}
              input={input}
              isBusy={consoleState.isBusy}
              latestAnswer={consoleState.latestAnswer}
              messages={consoleState.messages}
              pendingRegionalAnalysisPackage={consoleState.pendingRegionalAnalysisPackage}
              regionalAnalysisPackageHistory={consoleState.regionalAnalysisPackageHistory}
              pendingProposals={
                consoleState.pendingRegionalAnalysisPackage ? [] : consoleState.pendingProposals
              }
              priorityItems={priorityItems}
              quickPrompts={quickPrompts}
              selectedImpact={selectedImpact}
              selectedPriorityId={consoleState.selectedEntityId}
              onChangeInput={setInput}
              onOpenOperations={() => navigate("/operations")}
              onPrompt={(prompt) => void consoleState.ask(prompt)}
              onResolveRegionalAnalysisPackage={(packageId, decision, note) =>
                void consoleState.resolveRegionalAnalysisPackage(packageId, decision, note)
              }
              onResolveProposal={(proposalId, decision, note) => void consoleState.resolveProposal(proposalId, decision, note)}
              onSelectPriority={(id) => void consoleState.selectEntity(id)}
              onSubmit={handleSubmit}
              onTextareaKeyDown={handleTextareaKeyDown}
            />
          </>
        ) : null}

        {isOverviewPage ? (
          <DigitalTwinImpactScreen
            overview={consoleState.twinOverview}
            focusObject={consoleState.focusObject}
            pendingProposals={consoleState.pendingProposals}
            approvedProposals={consoleState.approvedProposals}
            hazardState={consoleState.hazardState}
            areaResourceStatusView={consoleState.areaResourceStatusView}
            eventResourceStatusView={consoleState.eventResourceStatusView}
            dialogEntries={consoleState.dialogEntries}
            dialogOpen={consoleState.dialogOpen}
            dialogBusy={consoleState.dialogBusy}
            streamStatus={consoleState.twinStreamStatus}
            onSelectObject={(objectId) => void consoleState.selectTwinObject(objectId)}
            onOpenDialog={() => consoleState.setDialogOpen(true)}
            onCloseDialog={() => consoleState.setDialogOpen(false)}
            onSendDialog={(message, objectId) => void consoleState.sendAgentDialog(message, objectId)}
            onGenerateProposals={() => void consoleState.generateTwinProposals()}
            onGenerateWarnings={(proposalId) => void consoleState.generateAudienceWarnings(proposalId)}
            onResolveProposal={(proposalId, decision, note) => void consoleState.resolveProposal(proposalId, decision, note)}
            onOpenProposalQueue={consoleState.openProposalQueue}
            onOpenOperations={() => navigate("/operations")}
            actionBusy={consoleState.isBusy}
            twinBusy={consoleState.twinBusy}
          />
        ) : null}

        {isOperationsPage ? (
          <OperationsPage
            list={
              <div className={styles.primaryColumn}>
                <ExecutionFlowBoard
                  title={operationsPageText.executionBoardTitle}
                  description={operationsPageText.executionBoardDescription}
                  stats={buildExecutionFlowStats({
                    pendingProposalCount: consoleState.pendingProposals.length,
                    proposalHistoryCount: proposalHistoryItems.length,
                    toolExecutionCount: latestToolExecutions.length,
                  })}
                  steps={executionFlowSteps}
                />
                <ToolExecutionSummary answer={consoleState.latestAnswer} />
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>{operationsPageText.analysisPackageSectionLabel}</p>
                      <h3>{operationsPageText.analysisPackageSectionTitle}</h3>
                    </div>
                  </div>
                  <RegionalAnalysisPackageHistoryPanel items={analysisPackageItems} />
                </div>
              </div>
            }
            detail={
              <div className={styles.sideColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>{operationsPageText.proposalHistorySectionLabel}</p>
                      <h3>{operationsPageText.proposalHistorySectionTitle}</h3>
                    </div>
                  </div>
                  <RegionalProposalHistoryPanel items={proposalHistoryItems} />
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>{operationsPageText.pendingConfirmSectionLabel}</p>
                      <h3>{operationsPageText.pendingConfirmSectionTitle}</h3>
                    </div>
                  </div>
                  <RegionalProposalHistoryPanel items={pendingProposalItems} />
                </div>
                <div className={styles.panelFrame}>
                  <SignalTimeline
                    title={operationsPageText.timelineTitle}
                    subtitle={operationsPageText.timelineSubtitle}
                    items={agentTimelineItems}
                    emptyText={operationsPageText.timelineEmpty}
                  />
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Warning Drafts</p>
                      <h3>分众预警草稿</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {latestWarningDrafts.slice(0, 4).map((draft) => (
                      <article key={draft.warning_id} className={styles.metricBlock}>
                        <span>{draft.audience}</span>
                        <strong>{`${draft.audience} warning`}</strong>
                        <small>{draft.grounding_summary || draft.content}</small>
                      </article>
                    ))}
                    {!latestWarningDrafts.length ? (
                      <div className={styles.emptyState}>当前还没有新的分众预警草稿，proposal 审批后会在这里出现。</div>
                    ) : null}
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <AdvisoryCard advisory={consoleState.activeAdvisory} />
                </div>
              </div>
            }
          />
        ) : null}

        {isDataPage ? (
          <DataPage>
            <div className={styles.panelFrame}>
              <AdminDesk
                areaId={consoleState.event?.area_id ?? "beilin_10km2"}
                eventId={consoleState.event?.event_id}
                profiles={consoleState.managedProfiles}
                areaResourceStatusView={consoleState.areaResourceStatusView}
                eventResourceStatusView={consoleState.eventResourceStatusView}
                ragDocuments={consoleState.ragDocuments}
                datasetStatus={consoleState.datasetStatus}
                busy={consoleState.adminBusy || consoleState.isBusy}
                status={consoleState.adminStatus}
                canEditRuntimeAdmin={canEditRuntimeAdmin}
                canManageDataset={canManageDataset}
                onSaveProfile={consoleState.saveManagedProfile}
                onDeleteProfile={consoleState.deleteManagedProfile}
                onInspectProfile={consoleState.selectEntity}
                onSaveAreaResources={consoleState.saveAreaResources}
                onSaveEventResources={consoleState.saveEventResources}
                onClearEventResources={consoleState.clearEventResources}
                onImportRagDocuments={consoleState.importRagDocuments}
                onReloadRagDocuments={consoleState.reloadRagDocuments}
                onFetchDatasetSources={consoleState.fetchDatasetSources}
                onRetryDatasetSource={consoleState.retryDatasetSource}
                onBuildDatasetPackage={consoleState.buildDatasetPackage}
                onValidateDatasetPackage={consoleState.validateDatasetPackage}
                onSyncDatasetPackage={consoleState.syncDatasetPackage}
                onCancelDatasetJob={consoleState.cancelDatasetJob}
                onRetryDatasetJob={consoleState.retryDatasetJob}
              />
            </div>
          </DataPage>
        ) : null}

        {isAgentsPage ? (
          <AgentsPage
            briefing={
              <div className={styles.primaryColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Agent Council</p>
                      <h3>多智能体会商摘要</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {(councilRoles.length ? councilRoles : []).map((role) => (
                      <article key={role.role} className={styles.metricBlock}>
                        <span>{role.label}</span>
                        <strong>{role.summary}</strong>
                        <small>
                          {role.recommended_action ?? "等待新的任务或证据输入。"} / 证据 {role.evidence_count} 条
                        </small>
                      </article>
                    ))}
                    {!councilRoles.length ? (
                      <div className={styles.emptyState}>当前还没有可展示的智能体会商摘要。</div>
                    ) : null}
                  </div>
                  <div className={styles.answerTags}>
                    <span>审计状态：{consoleState.agentCouncil?.audit_decision.status ?? "unknown"}</span>
                    <span>开放问题：{consoleState.agentCouncil?.open_questions.length ?? 0}</span>
                    <span>阻断项：{consoleState.agentCouncil?.blocked_by.length ?? 0}</span>
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Difference Matrix</p>
                      <h3>角色差异与分歧</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {agentDivergenceRows.length ? (
                      agentDivergenceRows.map((row) => (
                        <article key={row.result.result_id} className={styles.metricBlock}>
                          <span>{row.result.agent_name} / {row.disposition}</span>
                          <strong>{normalizeAgentTerminology(row.result.summary)}</strong>
                          <small>
                            分歧点：{row.disagreement} / 置信度 {Math.round(row.confidence * 100)}% / 证据{" "}
                            {row.result.evidence_refs.length} 条
                          </small>
                          <small>编排理由：{row.rationale}</small>
                        </article>
                      ))
                    ) : (
                      <div className={styles.emptyState}>当前没有足够的 recent results 来展示角色差异。</div>
                    )}
                  </div>
                  <div className={styles.answerTags}>
                    {agentOpenQuestions.slice(0, 4).map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                    {!agentOpenQuestions.length ? <span>当前没有新的 open questions</span> : null}
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Evidence Compare</p>
                      <h3>证据对照板</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {evidenceCompareResults.length ? (
                      evidenceCompareResults.map((result) => (
                        <article key={`${result.result_id}-evidence`} className={styles.metricBlock}>
                          <span>{result.agent_name}</span>
                          <strong>
                            引用 {result.evidence_refs.length} 条 / 缺口 {result.missing_slots.length} 项
                          </strong>
                          <small>
                            {result.evidence_refs.slice(0, 3).join(" | ") || "当前没有明确证据引用"}
                            {result.missing_slots.length
                              ? ` / 缺口：${result.missing_slots.slice(0, 2).join(" | ")}`
                              : ""}
                            {result.handoff_recommendations.length
                              ? ` / 建议移交：${result.handoff_recommendations.slice(0, 2).join(" | ")}`
                              : ""}
                          </small>
                        </article>
                      ))
                    ) : (
                      <div className={styles.emptyState}>当前还没有足够的证据引用差异可以展示。</div>
                    )}
                  </div>
                </div>
              </div>
            }
            chamber={
              <div className={styles.primaryColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Conference View</p>
                      <h3>会商室主桌面</h3>
                    </div>
                  </div>
                  <MultiAgentDesk
                    eventId={consoleState.event?.event_id}
                    agentStatus={consoleState.agentStatus}
                    agentTasks={consoleState.agentTasks}
                    sessionMemoryView={consoleState.sessionMemoryView}
                    sharedMemorySnapshot={consoleState.sharedMemorySnapshot}
                    episodeSummaries={consoleState.episodeSummaries}
                    triggerEvents={consoleState.triggerEvents}
                    agentTimeline={consoleState.agentTimeline}
                    supervisorRuns={consoleState.supervisorRuns}
                    supervisorLoopStatus={consoleState.supervisorLoopStatus}
                    recentAgentResults={consoleState.recentAgentResults}
                    experienceContext={consoleState.experienceContext}
                    decisionReport={consoleState.decisionReport}
                    agentMetrics={consoleState.agentMetrics}
                    evaluationBenchmarks={consoleState.evaluationBenchmarks}
                    latestEvaluationReport={consoleState.latestEvaluationReport}
                    busy={consoleState.isBusy}
                    canControlSupervisor={canControlSupervisor}
                    canReplayTask={canReplayTask}
                    canRunEvaluation={canRunEvaluation}
                    onRunSupervisor={consoleState.runSupervisorNow}
                    onTickSupervisor={consoleState.tickSupervisor}
                    onReplayTask={consoleState.replayAgentTask}
                    onRunEvaluation={consoleState.runEvaluation}
                    onReplayEvaluationReport={consoleState.replayEvaluationReport}
                  />
                </div>
              </div>
            }
            orchestration={
              <div className={styles.primaryColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Supervisor Orchestration</p>
                      <h3>编排结果与审计边界</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    <article className={styles.metricBlock}>
                      <span>Audit rationale</span>
                      <strong>{consoleState.agentCouncil?.audit_decision.rationale ?? "等待新的 supervisor 编排结果。"}</strong>
                      <small>
                        {consoleState.agentCouncil?.audit_decision.approval_required
                          ? "当前仍要求人工审批后放行。"
                          : "当前编排允许自动推进。"}
                      </small>
                    </article>
                    <article className={styles.metricBlock}>
                      <span>Decision path</span>
                      <strong>{agentDecisionPath[0] ?? "waiting"}</strong>
                      <small>{agentDecisionPath.slice(1, 4).join(" -> ") || "当前还没有完整的 decision path。"}</small>
                    </article>
                  </div>
                  <div className={styles.answerTags}>
                    {agentDecisionPath.slice(0, 4).map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                    {!agentDecisionPath.length ? <span>当前还没有 supervisor decision path</span> : null}
                    {agentBlockedBy.slice(0, 3).map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <SignalTimeline
                    title="智能体时间线"
                    subtitle="最近触发与任务变化"
                    items={agentTimelineItems}
                    emptyText="当前没有新的智能体时间线记录。"
                  />
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Closure Readiness</p>
                      <h3>会商到闭环的推进状态</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    <article className={styles.metricBlock}>
                      <span>Pending proposals</span>
                      <strong>{pendingProposalCount}</strong>
                      <small>待审批 proposal 仍需人工放行后才能进入 warning 生成。</small>
                    </article>
                    <article className={styles.metricBlock}>
                      <span>Approved proposals</span>
                      <strong>{approvedProposalCount}</strong>
                      <small>已批准 proposal 可以直接进入 audience warnings 和执行留痕。</small>
                    </article>
                    <article className={styles.metricBlock}>
                      <span>Warning drafts</span>
                      <strong>{warningDraftCount || latestWarningDrafts.length}</strong>
                      <small>warning drafts 数量可以帮助判断闭环是否已经落地。</small>
                    </article>
                  </div>
                </div>
              </div>
            }
            governance={
              <div className={styles.sideColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Open Questions</p>
                      <h3>未解决问题</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {agentOpenQuestions.length ? (
                      agentOpenQuestions.map((item) => (
                        <article key={item} className={styles.metricBlock}>
                          <span>question</span>
                          <strong>{item}</strong>
                          <small>该问题尚未被会商链完整关闭，需要继续追问或补证。</small>
                        </article>
                      ))
                    ) : (
                      <div className={styles.emptyState}>当前没有新的 open questions。</div>
                    )}
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Blocked By</p>
                      <h3>放行阻断项</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {agentBlockedBy.length ? (
                      agentBlockedBy.map((item) => (
                        <article key={item} className={styles.metricBlock}>
                          <span>blocked</span>
                          <strong>{item}</strong>
                          <small>当前会商结论被此边界限制，不能直接自动推进。</small>
                        </article>
                      ))
                    ) : (
                      <div className={styles.emptyState}>当前没有新的阻断项，会商结果可继续推进。</div>
                    )}
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Closure Link</p>
                      <h3>闭环出口</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {latestWarningDrafts.slice(0, 3).map((draft) => (
                      <article key={draft.warning_id} className={styles.metricBlock}>
                        <span>{draft.audience}</span>
                        <strong>{draft.channel}</strong>
                        <small>{draft.grounding_summary || draft.content}</small>
                      </article>
                    ))}
                    {!latestWarningDrafts.length ? (
                      <div className={styles.emptyState}>当前还没有新的 warning draft，会商结果将在批准 proposal 后进入这里。</div>
                    ) : null}
                  </div>
                </div>
              </div>
            }
          />
        ) : null}

        {isReliabilityPage ? (
          <ReliabilityPage
            health={
              <div className={styles.primaryColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Agent Council Audit</p>
                      <h3>会商与审计判定</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {(councilRoles.length ? councilRoles : []).map((role) => (
                      <article key={role.role} className={styles.metricBlock}>
                        <span>{role.label}</span>
                        <strong>{role.summary}</strong>
                        <small>{role.recommended_action ?? "等待新的审计结论或治理动作。"}</small>
                      </article>
                    ))}
                    {!councilRoles.length ? (
                      <div className={styles.emptyState}>当前没有新的会商摘要，系统将继续等待触发和证据汇聚。</div>
                    ) : null}
                  </div>
                  <div className={styles.answerTags}>
                    <span>Audit: {consoleState.agentCouncil?.audit_decision.status ?? "unknown"}</span>
                    <span>Open questions: {consoleState.agentCouncil?.open_questions.length ?? 0}</span>
                    <span>Blocked by: {consoleState.agentCouncil?.blocked_by.length ?? 0}</span>
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <ReliabilityAuditDesk
                    eventId={consoleState.event?.event_id}
                    supervisorLoopStatus={consoleState.supervisorLoopStatus}
                    alerts={consoleState.openAlerts}
                    auditRecords={consoleState.auditRecords}
                    archiveStatus={consoleState.archiveStatus}
                    busy={consoleState.reliabilityBusy || consoleState.isBusy}
                    canRunArchive={canRunArchive}
                    onQueryAudit={consoleState.queryAuditRecords}
                    onRunArchive={consoleState.runArchiveCycle}
                  />
                </div>
              </div>
            }
            governance={
              <div className={styles.sideColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Decision Boundary</p>
                      <h3>权限与阻断边界</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    <article className={styles.metricBlock}>
                      <span>Rationale</span>
                      <strong>{consoleState.agentCouncil?.audit_decision.rationale ?? "等待新的审计说明。"}</strong>
                      <small>
                        {(consoleState.agentCouncil?.audit_decision.risk_flags ?? []).join(" / ") || "当前没有额外风险标记。"}
                      </small>
                    </article>
                  </div>
                  <div className={styles.answerTags}>
                    <span>审批要求：{consoleState.agentCouncil?.audit_decision.approval_required ? "需要人工放行" : "可自动推进"}</span>
                    <span>SSE：{consoleState.twinStreamStatus}</span>
                  </div>
                </div>
                <SecurityDesk
                  operatorRole={consoleState.operatorRole}
                  operatorCapabilities={consoleState.operatorCapabilities}
                  onChangeRole={consoleState.setOperatorRole}
                />
              </div>
            }
            closure={
              <div className={styles.primaryColumn}>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Closure Snapshot</p>
                      <h3>审批与预警闭环</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    <article className={styles.metricBlock}>
                      <span>Pending proposals</span>
                      <strong>{pendingProposalCount}</strong>
                      <small>等待审批的人机协同动作仍在主链路中。</small>
                    </article>
                    <article className={styles.metricBlock}>
                      <span>Approved proposals</span>
                      <strong>{approvedProposalCount}</strong>
                      <small>已批准动作可继续生成 audience warnings 并进入执行留痕。</small>
                    </article>
                    <article className={styles.metricBlock}>
                      <span>Warning drafts</span>
                      <strong>{warningDraftCount || latestWarningDrafts.length}</strong>
                      <small>分众预警草稿已与 proposal 闭环关联。</small>
                    </article>
                  </div>
                </div>
                <div className={styles.panelFrame}>
                  <div className={styles.panelHeaderCompact}>
                    <div>
                      <p className={styles.sectionLabel}>Latest Outputs</p>
                      <h3>最新闭环产物</h3>
                    </div>
                  </div>
                  <div className={styles.answerList}>
                    {latestWarningDrafts.slice(0, 4).map((draft) => (
                      <article key={draft.warning_id} className={styles.metricBlock}>
                        <span>{draft.audience}</span>
                        <strong>{draft.channel}</strong>
                        <small>{draft.grounding_summary || draft.content}</small>
                      </article>
                    ))}
                    {!latestWarningDrafts.length ? (
                      <div className={styles.emptyState}>当前还没有新的 warning draft，批准 proposal 后会在这里形成闭环结果。</div>
                    ) : null}
                  </div>
                </div>
              </div>
            }
          />
        ) : null}
      </AppShell>
    </>
  );
}























