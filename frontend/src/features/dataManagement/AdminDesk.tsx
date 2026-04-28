import styles from "../../App.module.css";
import {
  AccessPolicyNotice,
  actionRequiredRoleText,
} from "../../components/SecurityDesk";
import { datasetPanelText } from "../../lib/datasetUiText";
import type {
  DatasetPipelineStatusView,
  EntityProfile,
  RAGDocument,
  ResourceStatus,
  ResourceStatusView,
} from "../../types/api";
import { DatasetOperationsPanel } from "./DatasetPanels";
import { ProfileEditor } from "./ProfileEditor";
import { RagImportPanel } from "./RagImportPanel";
import { ResourceStatusEditor } from "./ResourceStatusEditor";

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

export function AdminDesk(props: AdminDeskProps) {
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
            label="区域"
            view={props.areaResourceStatusView}
            areaId={props.areaId}
            busy={props.busy || !props.canEditRuntimeAdmin}
            saveLabel="保存区域资源状态"
            onSave={props.onSaveAreaResources}
          />
          <ResourceStatusEditor
            title="事件资源状态"
            label="事件"
            view={props.eventResourceStatusView}
            areaId={props.areaId}
            busy={props.busy || !props.canEditRuntimeAdmin}
            saveLabel="保存事件覆盖"
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
