export const datasetPanelText = {
  pipelineSectionLabel: "数据管线",
  pipelineTitle: "数据抓取、构建与同步状态",
  operationsTitle: "数据管线运维",
  currentTaskLabel: "当前任务",
  attemptCountLabel: "尝试次数",
  retryCountLabel: "重试次数",
  autoRetryLabel: "自动重试",
  retryOriginLabel: "重试来源",
  waitingProgress: "等待任务开始或返回最新进度。",
  waitingProgressAlt: "等待数据管线任务进入执行状态。",
  cachedFileUnit: "个文件",
  rawCacheLabel: "原始缓存",
  rawReadinessLabel: "原始数据就绪度",
  normalizedOutputLabel: "标准化输出",
  bootstrapFileLabel: "引导文件",
  fetchProgressLabel: "抓取进度",
  validationHitLabel: "校验命中",
  validationHitHint: "校验阶段命中的知识检索次数。",
  fetchSourcesAction: "抓取数据源",
  validateAction: "校验数据包",
  syncAction: "同步数据库",
  buildAndSyncAction: "构建并同步",
  latestValidationLabel: "最新校验",
  latestBuildLabel: "最新构建",
  latestFetchLabel: "最新抓取",
  runtimeKnowledgeBaseLabel: "运行时知识库",
  missingRequiredSourcesLabel: "缺失必需数据源",
  rawHealthSectionLabel: "原始缓存健康",
  rawHealthTitle: "源状态、解析结果与缓存健康",
  fetchDetailSectionLabel: "抓取明细",
  fetchDetailTitle: "最新下载记录与缓存状态",
  taskHistorySectionLabel: "任务历史",
  taskHistoryTitle: "最近任务与处理结果",
  noDatasetStatus: "当前没有可用的数据管线状态。",
  noDatasetStatusAlt: "当前没有最新校验结果。",
  noBuildSummary: "当前没有构建摘要。",
  noBuildSummaryAlt: "当前没有最新构建结果。",
  noFetchSummary: "当前没有抓取摘要。",
  noFetchSummaryAlt: "当前没有最新抓取结果。",
  noCachedFiles: "当前没有缓存文件",
  noSourceStatus: "当前没有数据源状态。",
  noSourceStatusAlt: "当前没有可展示的数据源状态。",
  noDownloadLog: "当前没有下载记录。",
  noDownloadLogAlt: "当前没有最新下载记录。",
  noRawHealth: "当前没有原始缓存健康信息。",
  noJobMessage: "当前没有任务说明。",
  noJobRecords: "当前没有数据管线任务记录。",
  noTargetPath: "当前没有目标路径",
  defaultArtifactName: "未命名制品",
  fetchSourceAction: "重新抓取",
  retrySourceAction: "重试数据源",
  retryJobAction: "重试任务",
  cancelJobAction: "取消任务",
  cancelRequestedAction: "已请求取消",
  enabled: "已开启",
  disabled: "已关闭",
  parsedReadyText: "已解析完成",
  parsedPendingText: "等待解析",
  ready: "就绪",
  notReady: "未就绪",
  none: "无",
} as const;

export function completenessText(status?: string | null) {
  if (!status) {
    return "--";
  }
  return {
    parsed: "已解析",
    cached: "已缓存",
    partial_cached: "部分缓存",
    manifest_only: "仅有清单",
    missing: "缺失",
  }[status] ?? status;
}

export function cacheStatusText(status?: string | null) {
  if (!status) {
    return "--";
  }
  return {
    parsed: "已解析",
    cached: "已缓存",
    partial_cached: "部分缓存",
    manifest_only: "仅有清单",
    missing: "缺失",
    download_failed: "下载失败",
  }[status] ?? status;
}

export function datasetJobStatusText(status?: string | null) {
  if (!status) {
    return "--";
  }
  return {
    pending: "等待中",
    running: "执行中",
    cancel_requested: "已请求取消",
    completed: "已完成",
    failed: "失败",
    canceled: "已取消",
  }[status] ?? status;
}

export function formatDatasetAction(action?: string | null) {
  if (!action) {
    return "--";
  }
  return {
    fetch_sources: "抓取数据源",
    build_dataset: "构建数据包",
    validate_dataset: "校验数据包",
    sync_demo_db: "同步演示数据库",
    download: "下载数据源",
    fetch: "刷新数据清单",
  }[action] ?? action;
}

export function formatDatasetStep(step?: string | null) {
  if (!step) {
    return "--";
  }
  return {
    fetch: "抓取",
    normalize: "标准化",
    profiles: "档案生成",
    observations: "观测整理",
    rag: "知识库构建",
    sync: "同步",
    validation: "校验",
  }[step] ?? step;
}

export function formatDatasetSourceCategory(category?: string | null) {
  if (!category) {
    return "--";
  }
  return {
    shelters: "避难场所",
    traffic: "交通路网",
    weather: "气象天气",
    statistics: "统计资料",
  }[category] ?? category;
}

export function formatParserKind(parserKind?: string | null) {
  if (!parserKind) {
    return "--";
  }
  return {
    html_digest: "HTML 摘要解析",
    api_reference: "接口数据解析",
    csv_table: "CSV 表格解析",
    osm_digest: "OSM 路网解析",
  }[parserKind] ?? parserKind;
}
