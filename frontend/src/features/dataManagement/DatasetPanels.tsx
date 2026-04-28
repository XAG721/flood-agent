import styles from "../../App.module.css";
import {
  cacheStatusText,
  completenessText,
  datasetJobStatusText,
  datasetPanelText,
  formatDatasetAction,
  formatDatasetSourceCategory,
  formatDatasetStep,
  formatParserKind,
} from "../../lib/datasetUiText";
import { formatTimestamp } from "../../lib/consoleFormatting";
import type { DatasetPipelineStatusView } from "../../types/api";

export interface DatasetPipelinePanelProps {
  datasetStatus: DatasetPipelineStatusView | null;
  busy: boolean;
  status: string | null;
  onFetch: (download?: boolean) => Promise<void>;
  onRetrySource: (sourceId: string) => Promise<void>;
  onBuild: (download?: boolean, syncDemoDb?: boolean) => Promise<void>;
  onValidate: () => Promise<void>;
  onSync: () => Promise<void>;
}

interface DatasetOperationsPanelProps extends DatasetPipelinePanelProps {
  onCancelJob: (jobId: string) => Promise<void>;
  onRetryJob: (jobId: string) => Promise<void>;
}

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

function ActiveJobPanel({
  activeJob,
  busy,
  onCancelJob,
  showRetryMeta = false,
}: {
  activeJob: NonNullable<DatasetPipelineStatusView["active_job"]>;
  busy: boolean;
  onCancelJob?: (jobId: string) => Promise<void>;
  showRetryMeta?: boolean;
}) {
  const canCancelActiveJob = Boolean(onCancelJob && ["pending", "running", "cancel_requested"].includes(activeJob.status));

  return (
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
      {showRetryMeta ? (
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
      ) : null}
      {onCancelJob ? (
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
      ) : null}
    </div>
  );
}

function MetricStrip({ datasetStatus }: { datasetStatus: DatasetPipelineStatusView | null }) {
  const validation = datasetStatus?.latest_validation ?? {};
  const fetchSummary = datasetStatus?.latest_fetch_summary ?? {};

  return (
    <div className={styles.agentStatusStrip}>
      {metric(datasetPanelText.rawCacheLabel, `${datasetStatus?.cached_file_count ?? 0}`, datasetStatus?.raw_dir)}
      {metric(datasetPanelText.rawReadinessLabel, datasetStatus?.raw_ready ? datasetPanelText.ready : datasetPanelText.notReady, `${datasetStatus?.raw_completeness_percent ?? 0}%`)}
      {metric(datasetPanelText.normalizedOutputLabel, `${datasetStatus?.normalized_files.length ?? 0}`, datasetStatus?.normalized_dir)}
      {metric(datasetPanelText.bootstrapFileLabel, `${datasetStatus?.bootstrap_files.length ?? 0}`, datasetStatus?.bootstrap_dir)}
      {metric(datasetPanelText.fetchProgressLabel, `${Number(fetchSummary.progress_percent ?? 0)}%`, `${Number(fetchSummary.downloaded_artifact_count ?? 0)}/${Number(fetchSummary.artifact_count ?? 0)} ${datasetPanelText.cachedFileUnit}`)}
      {metric(datasetPanelText.validationHitLabel, `${Number(validation.rag_query_hit_count ?? 0)}`, datasetPanelText.validationHitHint)}
    </div>
  );
}

function DatasetActions({
  busy,
  onFetch,
  onValidate,
  onSync,
  onBuild,
}: Pick<DatasetPipelinePanelProps, "busy" | "onFetch" | "onValidate" | "onSync" | "onBuild">) {
  return (
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
  );
}

function DatasetSummary({
  datasetStatus,
  compact = false,
}: {
  datasetStatus: DatasetPipelineStatusView | null;
  compact?: boolean;
}) {
  const validation = datasetStatus?.latest_validation ?? {};
  const buildSummary = datasetStatus?.latest_build_summary ?? {};
  const fetchSummary = datasetStatus?.latest_fetch_summary ?? {};

  return (
    <div className={styles.memoryList}>
      <div>
        <span className={styles.operationLabel}>{datasetPanelText.latestValidationLabel}</span>
        <p>{datasetStatus ? `${Number(validation.shelter_count ?? 0)} 个避难点 | ${Number(validation.road_count ?? 0)} 条道路 | ${Number(validation.entity_profile_count ?? 0)} 份对象档案` : compact ? datasetPanelText.noDatasetStatusAlt : datasetPanelText.noDatasetStatus}</p>
      </div>
      <div>
        <span className={styles.operationLabel}>{datasetPanelText.latestBuildLabel}</span>
        <p>{Object.keys(buildSummary).length ? Object.keys(buildSummary).join(" | ") : compact ? datasetPanelText.noBuildSummaryAlt : datasetPanelText.noBuildSummary}</p>
      </div>
      <div>
        <span className={styles.operationLabel}>{datasetPanelText.latestFetchLabel}</span>
        <p>{datasetStatus ? `成功 ${Number(fetchSummary.downloaded_artifact_count ?? 0)} / 失败 ${Number(fetchSummary.failed_artifact_count ?? 0)}` : compact ? datasetPanelText.noFetchSummaryAlt : datasetPanelText.noFetchSummary}</p>
      </div>
      {!compact ? (
        <div>
          <span className={styles.operationLabel}>{datasetPanelText.missingRequiredSourcesLabel}</span>
          <p>{datasetStatus?.missing_required_sources?.length ? datasetStatus.missing_required_sources.join(" | ") : datasetPanelText.none}</p>
        </div>
      ) : null}
      <div>
        <span className={styles.operationLabel}>{datasetPanelText.runtimeKnowledgeBaseLabel}</span>
        <p>{datasetStatus?.runtime_rag_path ?? "--"}</p>
      </div>
    </div>
  );
}

function SourceList({
  datasetStatus,
  busy,
  onRetrySource,
  compact = false,
}: {
  datasetStatus: DatasetPipelineStatusView | null;
  busy: boolean;
  onRetrySource: (sourceId: string) => Promise<void>;
  compact?: boolean;
}) {
  return (
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
              {!compact && source.missing_artifact_types.length ? ` | 缺失：${source.missing_artifact_types.join(", ")}` : ""}
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
                  {compact ? datasetPanelText.fetchSourceAction : datasetPanelText.retrySourceAction}
                </button>
              </div>
            ) : null}
          </article>
        ))
      ) : (
        <p className={styles.emptyState}>{compact ? datasetPanelText.noSourceStatusAlt : datasetPanelText.noSourceStatus}</p>
      )}
    </div>
  );
}

function FetchLogPanel({
  datasetStatus,
  compact = false,
}: {
  datasetStatus: DatasetPipelineStatusView | null;
  compact?: boolean;
}) {
  const recentFetchDetails = datasetStatus?.latest_download_log ?? [];

  return (
    <div className={styles.auditPanel}>
      <div className={styles.panelHeader}>
        <div>
          {compact ? (
            <>
              <span>{`${recentFetchDetails.length} 条记录`}</span>
              <span>{datasetStatus?.failed_source_count ?? 0} 个失败源</span>
            </>
          ) : (
            <>
              <p className={styles.sectionLabel}>{datasetPanelText.fetchDetailSectionLabel}</p>
              <h3>{datasetPanelText.fetchDetailTitle}</h3>
            </>
          )}
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
        <p className={styles.emptyState}>{compact ? datasetPanelText.noDownloadLogAlt : datasetPanelText.noDownloadLog}</p>
      )}
    </div>
  );
}

export function DatasetPipelinePanel({
  datasetStatus,
  busy,
  status,
  onFetch,
  onRetrySource,
  onBuild,
  onValidate,
  onSync,
}: DatasetPipelinePanelProps) {
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
      {activeJob ? <ActiveJobPanel activeJob={activeJob} busy={busy} /> : null}
      <MetricStrip datasetStatus={datasetStatus} />
      <DatasetActions busy={busy} onFetch={onFetch} onValidate={onValidate} onSync={onSync} onBuild={onBuild} />
      <DatasetSummary datasetStatus={datasetStatus} compact />
      <SourceList datasetStatus={datasetStatus} busy={busy} onRetrySource={onRetrySource} compact />
      <FetchLogPanel datasetStatus={datasetStatus} compact />
    </div>
  );
}

function RawHealthPanel({ datasetStatus }: { datasetStatus: DatasetPipelineStatusView | null }) {
  const rawHealth = datasetStatus?.raw_cache_health ?? [];

  return (
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
  );
}

function JobHistoryPanel({
  datasetStatus,
  busy,
  onCancelJob,
  onRetryJob,
}: Pick<DatasetOperationsPanelProps, "datasetStatus" | "busy" | "onCancelJob" | "onRetryJob">) {
  const recentJobs = datasetStatus?.recent_jobs ?? [];

  return (
    <div className={styles.auditPanel}>
      <div className={styles.panelHeader}>
        <div>
          <p className={styles.sectionLabel}>任务历史</p>
          <h3>数据任务回放</h3>
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
                尝试 {job.attempt_count ?? 0}/{job.max_attempts ?? 1} | 重试 {job.retry_count ?? 0} | 启动于 {formatTimestamp(job.started_at)}
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
  );
}

export function DatasetOperationsPanel({
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
  const activeJob = datasetStatus?.active_job ?? null;

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
      {activeJob ? <ActiveJobPanel activeJob={activeJob} busy={busy} onCancelJob={onCancelJob} showRetryMeta /> : null}
      <MetricStrip datasetStatus={datasetStatus} />
      <DatasetActions busy={busy} onFetch={onFetch} onValidate={onValidate} onSync={onSync} onBuild={onBuild} />
      <DatasetSummary datasetStatus={datasetStatus} />
      <RawHealthPanel datasetStatus={datasetStatus} />
      <SourceList datasetStatus={datasetStatus} busy={busy} onRetrySource={onRetrySource} />
      <FetchLogPanel datasetStatus={datasetStatus} />
      <JobHistoryPanel datasetStatus={datasetStatus} busy={busy} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
    </div>
  );
}
