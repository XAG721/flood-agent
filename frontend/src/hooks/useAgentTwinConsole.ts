import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { agentTwinApi } from "../api/agentTwinApi";
import type {
  AgentCouncilView,
  AgentDialogTranscriptEntry,
  AudienceWarningDraft,
  FocusObjectView,
  ProposalGenerationResponse,
  TwinOverviewView,
  TwinStreamEvent,
  WarningGenerationResponse,
} from "../types/api";
import { useV2OperatorConsole } from "./useV2OperatorConsole";

type TwinStreamStatus = "closed" | "connecting" | "open" | "error";

function nowIso() {
  return new Date().toISOString();
}

export function useAgentTwinConsole() {
  const base = useV2OperatorConsole();
  const currentEventId = base.event?.event_id ?? null;
  const [twinOverview, setTwinOverview] = useState<TwinOverviewView | null>(null);
  const [focusObject, setFocusObject] = useState<FocusObjectView | null>(null);
  const [agentCouncil, setAgentCouncil] = useState<AgentCouncilView | null>(null);
  const [warningDrafts, setWarningDrafts] = useState<AudienceWarningDraft[]>([]);
  const [dialogEntries, setDialogEntries] = useState<AgentDialogTranscriptEntry[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogBusy, setDialogBusy] = useState(false);
  const [twinBusy, setTwinBusy] = useState(false);
  const [twinStreamStatus, setTwinStreamStatus] = useState<TwinStreamStatus>("closed");
  const twinStreamRef = useRef<EventSource | null>(null);
  const baseEventRef = useRef(base.event);
  const refreshRegionalProposalDataRef = useRef(base.refreshRegionalProposalData);

  useEffect(() => {
    baseEventRef.current = base.event;
    refreshRegionalProposalDataRef.current = base.refreshRegionalProposalData;
  }, [base.event, base.refreshRegionalProposalData]);

  const refreshTwinOverview = useCallback(async () => {
    if (!currentEventId) {
      setTwinOverview(null);
      return;
    }

    setTwinBusy(true);
    try {
      const overview = await agentTwinApi.getOverview(currentEventId);
      setTwinOverview(overview);
      setWarningDrafts(overview.recent_warning_drafts ?? []);
    } catch (error) {
      setTwinOverview((current) => current);
      setTwinStreamStatus("error");
      console.error("Failed to load twin overview:", error);
    } finally {
      setTwinBusy(false);
    }
  }, [currentEventId]);

  const refreshAgentCouncil = useCallback(async () => {
    if (!currentEventId) {
      setAgentCouncil(null);
      return;
    }

    try {
      const council = await agentTwinApi.getCouncil(currentEventId);
      setAgentCouncil(council);
    } catch (error) {
      setAgentCouncil((current) => current);
      console.error("Failed to load agent council:", error);
    }
  }, [currentEventId]);

  const loadFocusObject = useCallback(
    async (objectId: string) => {
      if (!currentEventId) {
        setFocusObject(null);
        return null;
      }
      const payload = await agentTwinApi.getFocusObject(currentEventId, objectId);
      setFocusObject(payload);
      return payload;
    },
    [currentEventId],
  );

  useEffect(() => {
    if (!currentEventId) {
      return;
    }
    void refreshTwinOverview();
    void refreshAgentCouncil();
  }, [currentEventId, refreshAgentCouncil, refreshTwinOverview]);

  useEffect(() => {
    if (!currentEventId) {
      setFocusObject(null);
      return;
    }
    if (focusObject && focusObject.event_id === currentEventId) {
      return;
    }
    if (!twinOverview?.lead_object_id) {
      return;
    }
    void loadFocusObject(twinOverview.lead_object_id);
  }, [currentEventId, focusObject, loadFocusObject, twinOverview?.lead_object_id]);

  useEffect(() => {
    if (twinStreamRef.current) {
      twinStreamRef.current.close();
      twinStreamRef.current = null;
    }
    if (!currentEventId) {
      setTwinStreamStatus("closed");
      return;
    }

    setTwinStreamStatus("connecting");
    const stream = agentTwinApi.openTwinStream(
      currentEventId,
      {
        onEvent: (event: TwinStreamEvent) => {
          setTwinStreamStatus("open");
          if (event.event_type === "twin_overview_updated") {
            const payload = event.payload as { overview?: TwinOverviewView };
            if (payload.overview) {
              setTwinOverview(payload.overview);
            }
            return;
          }
          if (event.event_type === "focus_object_updated") {
            const payload = event.payload as { focus_object?: FocusObjectView | null };
            if (payload.focus_object) {
              setFocusObject(payload.focus_object);
            }
            return;
          }
          if (event.event_type === "agent_council_updated") {
            const payload = event.payload as { council?: AgentCouncilView };
            if (payload.council) {
              setAgentCouncil(payload.council);
            }
            return;
          }
          if (event.event_type === "proposal_generated" || event.event_type === "proposal_status_changed") {
            if (baseEventRef.current) {
              void refreshRegionalProposalDataRef.current(baseEventRef.current);
            }
            void refreshTwinOverview();
            void refreshAgentCouncil();
            return;
          }
          if (event.event_type === "warnings_generated") {
            const payload = event.payload as { warnings?: AudienceWarningDraft[] };
            if (payload.warnings) {
              setWarningDrafts(payload.warnings);
            }
            void refreshTwinOverview();
          }
        },
        onError: () => {
          setTwinStreamStatus("error");
        },
      },
      focusObject?.object_id ?? twinOverview?.lead_object_id ?? undefined,
    );

    twinStreamRef.current = stream;
    return () => {
      stream.close();
      twinStreamRef.current = null;
    };
  }, [currentEventId, focusObject?.object_id, refreshAgentCouncil, refreshTwinOverview, twinOverview?.lead_object_id]);

  const selectTwinObject = useCallback(
    async (objectId: string) => {
      await base.selectEntity(objectId);
      return loadFocusObject(objectId);
    },
    [base, loadFocusObject],
  );

  const sendAgentDialog = useCallback(
    async (message: string, objectId?: string) => {
      if (!currentEventId || !message.trim()) {
        return null;
      }

      const targetObjectId = objectId ?? focusObject?.object_id ?? twinOverview?.lead_object_id ?? undefined;
      if (targetObjectId) {
        await selectTwinObject(targetObjectId);
      }

      setDialogOpen(true);
      setDialogBusy(true);
      setDialogEntries((current) => [
        ...current,
        {
          id: `dialog_user_${Date.now()}`,
          role: "user",
          content: message.trim(),
          created_at: nowIso(),
        },
      ]);

      try {
        const response = await agentTwinApi.sendDialog(currentEventId, {
          object_id: targetObjectId,
          message: message.trim(),
        });
        setDialogEntries((current) => [
          ...current,
          {
            id: `dialog_assistant_${Date.now()}`,
            role: "assistant",
            content: response.answer,
            created_at: response.generated_at,
            response,
          },
        ]);
        if (response.object_id) {
          await loadFocusObject(response.object_id);
        }
        if (response.proposal_entry?.proposal && base.event) {
          await base.refreshRegionalProposalData(base.event);
          await refreshTwinOverview();
          await refreshAgentCouncil();
        }
        return response;
      } finally {
        setDialogBusy(false);
      }
    },
    [base, currentEventId, focusObject?.object_id, loadFocusObject, refreshAgentCouncil, refreshTwinOverview, selectTwinObject, twinOverview?.lead_object_id],
  );

  const generateTwinProposals = useCallback(
    async (objectIds?: string[]) => {
      if (!currentEventId) {
        return null;
      }
      setTwinBusy(true);
      try {
        const response: ProposalGenerationResponse = await agentTwinApi.generateProposals(currentEventId, {
          object_ids:
            objectIds && objectIds.length
              ? objectIds
              : twinOverview?.focus_objects.map((item) => item.object_id) ?? [],
        });
        if (base.event) {
          await base.refreshRegionalProposalData(base.event);
        }
        await refreshTwinOverview();
        await refreshAgentCouncil();
        if (response.proposals.length) {
          setDialogOpen(true);
        }
        return response;
      } finally {
        setTwinBusy(false);
      }
    },
    [base, currentEventId, refreshAgentCouncil, refreshTwinOverview, twinOverview?.focus_objects],
  );

  const generateAudienceWarnings = useCallback(
    async (proposalId: string) => {
      const response: WarningGenerationResponse = await agentTwinApi.generateWarnings(proposalId);
      setWarningDrafts(response.warnings);
      await refreshTwinOverview();
      return response;
    },
    [refreshTwinOverview],
  );

  const resolveTwinProposal = useCallback(
    async (proposalId: string, decision: "approve" | "reject", note: string) => {
      await base.resolveProposal(proposalId, decision, note);
      await refreshTwinOverview();
      await refreshAgentCouncil();
    },
    [base, refreshAgentCouncil, refreshTwinOverview],
  );

  const openProposalQueue = useCallback(() => {
    base.setRegionalProposalModalOpen(true);
  }, [base]);

  const approvedProposals = useMemo(
    () => base.proposals.filter((item) => item.status === "approved"),
    [base.proposals],
  );

  const fallbackTwinOverview = useMemo<TwinOverviewView | null>(() => {
    if (twinOverview || !base.event) {
      return twinOverview;
    }

    const fallbackFocusObjects = base.topImpacts.slice(0, 6).map((impact, index) => ({
      object_id: impact.entity.entity_id,
      name: impact.entity.name,
      entity_type: impact.entity.entity_type,
      village: impact.entity.village,
      risk_level: impact.risk_level,
      time_to_impact_minutes: impact.time_to_impact_minutes,
      summary: impact.risk_reason[0] ?? `${impact.entity.name} 需要继续跟踪。`,
      recommended_action:
        impact.risk_reason[1] ?? base.latestAnswer?.recommended_actions?.[index] ?? "继续跟踪对象风险并准备联动处置。",
      pending_proposal_ids: base.pendingProposals
        .filter((proposal) => proposal.entity_id === impact.entity.entity_id)
        .map((proposal) => proposal.proposal_id),
      canvas_position: {
        left: 22 + (index % 3) * 24,
        top: 24 + Math.floor(index / 3) * 24,
      },
    }));

    return {
      event_id: base.event.event_id,
      area_id: base.event.area_id,
      event_title: base.event.title,
      generated_at: nowIso(),
      overall_risk_level: base.hazardState?.overall_risk_level ?? "None",
      trend: base.hazardState?.trend ?? "unknown",
      summary:
        base.agentStatus?.latest_summary ??
        "系统正在使用现有事件与对象态势构建主屏，等待 V3 聚合视图返回更完整的影响链解释。",
      lead_object_id: fallbackFocusObjects[0]?.object_id ?? null,
      lead_object_name: fallbackFocusObjects[0]?.name ?? null,
      focus_objects: fallbackFocusObjects,
      pending_proposal_count: base.pendingProposals.length,
      approved_proposal_count: approvedProposals.length,
      warning_draft_count: 0,
      active_alert_count: base.openAlerts.length,
      map_layers: fallbackFocusObjects.map((item, index) => ({
        object_id: item.object_id,
        name: item.name,
        risk_level: item.risk_level,
        entity_type: item.entity_type,
        is_lead: index === 0,
        east_offset_m: -240 + (index % 3) * 180,
        north_offset_m: 150 - Math.floor(index / 3) * 120,
        height_offset_m: 18 + index * 4,
        proposal_state: item.pending_proposal_ids.length ? "proposal_pending" : "tracking",
      })),
      recommended_actions: base.latestAnswer?.recommended_actions ?? [],
      signals: base.openAlerts.slice(0, 4).map((alert) => ({
        signal_id: alert.alert_id,
        title: alert.summary,
        detail: alert.details || "系统已记录新的态势信号。",
        severity: alert.severity,
        created_at: alert.last_seen_at ?? alert.first_seen_at,
      })),
      recent_warning_drafts: [],
    };
  }, [approvedProposals.length, base.agentStatus?.latest_summary, base.event, base.hazardState?.overall_risk_level, base.hazardState?.trend, base.latestAnswer?.recommended_actions, base.openAlerts, base.pendingProposals, base.topImpacts, twinOverview]);

  const fallbackFocusObject = useMemo<FocusObjectView | null>(() => {
    if (focusObject || !base.selectedImpact) {
      return focusObject;
    }

    const entityId = base.selectedImpact.entity.entity_id;
    const relatedProposals = [
      ...(base.regionalProposalQueueSnapshot?.items ?? []),
      ...base.regionalProposalHistory,
    ].filter(
      (item) =>
        item.proposal.entity_id === entityId ||
        (item.proposal.high_risk_object_ids ?? []).includes(entityId),
    );

    return {
      event_id: base.selectedImpact.event_id,
      object_id: entityId,
      object_name: base.selectedImpact.entity.name,
      entity_type: base.selectedImpact.entity.entity_type,
      village: base.selectedImpact.entity.village,
      risk_level: base.selectedImpact.risk_level,
      time_to_impact_minutes: base.selectedImpact.time_to_impact_minutes,
      summary: base.selectedImpact.risk_reason[0] ?? `${base.selectedImpact.entity.name} 需要优先研判。`,
      risk_reasons: base.selectedImpact.risk_reason,
      recommended_actions: base.latestAnswer?.recommended_actions ?? [],
      risk_reminders: base.selectedImpact.risk_reason.slice(0, 2),
      evidence: base.selectedImpact.evidence,
      related_proposals: relatedProposals,
    };
  }, [base.latestAnswer?.recommended_actions, base.regionalProposalHistory, base.regionalProposalQueueSnapshot?.items, base.selectedImpact, focusObject]);

  return {
    ...base,
    twinOverview: fallbackTwinOverview,
    focusObject: fallbackFocusObject,
    agentCouncil,
    warningDrafts,
    dialogEntries,
    dialogOpen,
    dialogBusy,
    twinBusy,
    twinStreamStatus,
    approvedProposals,
    isBusy: base.isBusy || dialogBusy || twinBusy,
    setDialogOpen,
    refreshTwinOverview,
    selectTwinObject,
    sendAgentDialog,
    generateTwinProposals,
    generateAudienceWarnings,
    resolveProposal: resolveTwinProposal,
    openProposalQueue,
  };
}
