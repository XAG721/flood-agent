import { request } from "../lib/httpClient";
import type {
  EntityProfile,
  RAGDocument,
  ResourceStatus,
  ResourceStatusView,
} from "../types/api";

export const adminApi = {
  listV2EntityProfiles(params?: {
    areaId?: string;
    entityType?: string;
  }): Promise<EntityProfile[]> {
    const query = new URLSearchParams();
    if (params?.areaId) {
      query.set("area_id", params.areaId);
    }
    if (params?.entityType) {
      query.set("entity_type", params.entityType);
    }
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return request(`/platform/admin/entity-profiles${suffix}`, { method: "GET" });
  },

  createV2EntityProfile(
    profile: EntityProfile,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<EntityProfile> {
    return request("/platform/admin/entity-profiles", {
      method: "POST",
      body: JSON.stringify({
        profile,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  updateV2EntityProfile(
    profile: EntityProfile,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<EntityProfile> {
    return request(`/platform/admin/entity-profiles/${profile.entity_id}`, {
      method: "PUT",
      body: JSON.stringify({
        profile,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  deleteV2EntityProfile(entityId: string): Promise<{ status: string; entity_id: string }> {
    return request(`/platform/admin/entity-profiles/${entityId}`, {
      method: "DELETE",
    });
  },

  getAreaResourceStatus(areaId: string): Promise<ResourceStatusView> {
    return request(`/platform/admin/areas/${areaId}/resource-status`, {
      method: "GET",
    });
  },

  updateAreaResourceStatus(
    areaId: string,
    resourceStatus: ResourceStatus,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<ResourceStatusView> {
    return request(`/platform/admin/areas/${areaId}/resource-status`, {
      method: "PUT",
      body: JSON.stringify({
        resource_status: resourceStatus,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  getEventResourceStatus(eventId: string): Promise<ResourceStatusView> {
    return request(`/platform/admin/events/${eventId}/resource-status`, {
      method: "GET",
    });
  },

  updateEventResourceStatus(
    eventId: string,
    resourceStatus: ResourceStatus,
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<ResourceStatusView> {
    return request(`/platform/admin/events/${eventId}/resource-status`, {
      method: "PUT",
      body: JSON.stringify({
        resource_status: resourceStatus,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  deleteEventResourceStatus(eventId: string): Promise<{ status: string; event_id: string }> {
    return request(`/platform/admin/events/${eventId}/resource-status`, {
      method: "DELETE",
    });
  },

  listV2RagDocuments(): Promise<RAGDocument[]> {
    return request("/platform/admin/rag-documents", { method: "GET" });
  },

  importV2RagDocuments(
    documents: RAGDocument[],
    payload?: { operator_id?: string; operator_role?: string },
  ): Promise<{ status: string; document_count: number; documents: RAGDocument[] }> {
    return request("/platform/admin/rag-documents/import", {
      method: "POST",
      body: JSON.stringify({
        documents,
        operator_id: payload?.operator_id,
        operator_role: payload?.operator_role,
      }),
    });
  },

  reloadV2RagDocuments(): Promise<{ status: string; document_count: number; documents: RAGDocument[] }> {
    return request("/platform/admin/rag-documents/reload", {
      method: "POST",
    });
  },
};
