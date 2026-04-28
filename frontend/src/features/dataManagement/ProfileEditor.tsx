import { useEffect, useState } from "react";
import styles from "../../App.module.css";
import { entityText, entityTypeOptions, travelModeOptions } from "../../config/consoleConfig";
import {
  formatMobilityConstraints,
  formatNotificationPreferences,
  formatVulnerabilityTags,
  parseMobilityConstraints,
  parseNotificationPreferences,
  parseVulnerabilityTags,
} from "../../lib/displayText";
import { joinCsv, parseCsv } from "../../lib/consoleFormatting";
import type { EntityProfile, TravelMode } from "../../types/api";
import { createBlankProfile } from "./dataModels";

interface ProfileEditorProps {
  areaId: string;
  profiles: EntityProfile[];
  busy: boolean;
  onSave: (profile: EntityProfile) => Promise<void>;
  onDelete: (entityId: string) => Promise<void>;
  onInspect: (entityId: string) => Promise<void>;
}

export function ProfileEditor({ areaId, profiles, busy, onSave, onDelete, onInspect }: ProfileEditorProps) {
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
          <p className={styles.sectionLabel}>画像管理</p>
          <h3>运行时对象档案管理</h3>
        </div>
        <button type="button" className={styles.secondaryButton} onClick={() => setSelectedProfileId("__new__")} disabled={busy}>
          新建画像
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
          <span className={styles.operationLabel}>对象编号</span>
          <input
            aria-label="admin-profile-entity-id"
            className={styles.fieldInput}
            value={draft.entity_id}
            onChange={(event) => setDraft((current) => ({ ...current, entity_id: event.target.value.trim() }))}
            disabled={busy || !isNewProfile}
            placeholder="居民_演示_001"
          />
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>名称</span>
          <input
            aria-label="admin-profile-name"
            className={styles.fieldInput}
            value={draft.name}
            onChange={(event) => setDraft((current) => ({ ...current, name: event.target.value }))}
            disabled={busy}
            placeholder="对象名称"
          />
        </label>
        <label className={styles.fieldBlock}>
          <span className={styles.operationLabel}>对象类型</span>
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
