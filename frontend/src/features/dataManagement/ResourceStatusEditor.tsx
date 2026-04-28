import { useEffect, useState } from "react";
import styles from "../../App.module.css";
import { resourceFields } from "../../config/consoleConfig";
import type { ResourceStatus, ResourceStatusView } from "../../types/api";
import { createBlankResourceStatus } from "./dataModels";

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

export function ResourceStatusEditor({ title, label, view, areaId, busy, saveLabel, onSave, onClear }: ResourceStatusEditorProps) {
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
            清除覆盖
          </button>
        ) : null}
        <button type="button" className={styles.primaryButton} aria-label={`save-${label}-resource-status`} disabled={busy} onClick={() => void onSave(draft)}>
          {saveLabel}
        </button>
      </div>
    </div>
  );
}
