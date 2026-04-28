from __future__ import annotations

from ..models import RAGDocument, ResourceStatus
from ..rag_runtime import validate_rag_document_payload
from .models import RAGDocumentImportRequest, ResourceStatusView, TriggerEventType


class RuntimeAdminMixin:
    def get_area_resource_status(self, area_id: str) -> ResourceStatus:
        resource_status = self.repository.get_area_resource_status(area_id)
        if resource_status is None:
            fallback = self.bootstrap_resource_status.get(area_id)
            if fallback is None:
                raise ValueError(f"Unknown area_id: {area_id}")
            self.repository.save_area_resource_status(fallback)
            resource_status = fallback
        return resource_status

    def save_area_resource_status(self, resource_status: ResourceStatus) -> ResourceStatus:
        if resource_status.area_id not in self.area_profiles:
            raise ValueError(f"Unknown area_id: {resource_status.area_id}")
        self.repository.save_area_resource_status(resource_status)
        self.add_audit_record(
            source_type="runtime_admin",
            action="area_resource_updated",
            summary=f"已更新区域 {resource_status.area_id} 的默认资源。",
            details={"area_id": resource_status.area_id},
        )
        return resource_status

    def get_event_resource_status(self, event_id: str) -> ResourceStatus | None:
        return self.repository.get_event_resource_status(event_id)

    def save_event_resource_status(self, event_id: str, resource_status: ResourceStatus) -> ResourceStatus:
        event = self.get_event(event_id)
        if resource_status.area_id != event.area_id:
            raise ValueError("event resource override area_id must match the event area_id.")
        self.repository.save_event_resource_status(event_id, resource_status)
        self.repository.add_v2_stream_record_for_payload(
            event_id,
            "impact_recomputed",
            {"resource_override_updated": True, "area_id": resource_status.area_id},
        )
        self.add_audit_record(
            source_type="runtime_admin",
            action="event_resource_override_updated",
            summary=f"已更新事件 {event_id} 的资源覆盖。",
            details={"event_id": event_id, "area_id": resource_status.area_id},
            event_id=event_id,
        )
        self.publish_trigger(
            event_id,
            trigger_type=TriggerEventType.RESOURCE_OVERRIDE_UPDATED,
            payload={"area_id": resource_status.area_id},
        )
        return resource_status

    def delete_event_resource_status(self, event_id: str) -> None:
        if not self.repository.delete_event_resource_status(event_id):
            raise ValueError(f"No event resource override found for event {event_id}.")
        self.add_audit_record(
            source_type="runtime_admin",
            action="event_resource_override_deleted",
            summary=f"已清除事件 {event_id} 的资源覆盖。",
            details={"event_id": event_id},
            event_id=event_id,
        )
        self.publish_trigger(
            event_id,
            trigger_type=TriggerEventType.RESOURCE_OVERRIDE_DELETED,
            payload={"event_id": event_id},
        )

    def get_resource_status(self, area_id: str, *, event_id: str | None = None) -> ResourceStatus:
        if event_id is not None:
            override = self.repository.get_event_resource_status(event_id)
            if override is not None:
                return override
        return self.get_area_resource_status(area_id)

    def get_area_resource_status_view(self, area_id: str) -> ResourceStatusView:
        resource_status = self.get_area_resource_status(area_id)
        return ResourceStatusView(scope="area_default", area_id=area_id, resource_status=resource_status)

    def get_event_resource_status_view(self, event_id: str) -> ResourceStatusView:
        event = self.get_event(event_id)
        override = self.repository.get_event_resource_status(event_id)
        if override is not None:
            return ResourceStatusView(
                scope="event_override",
                area_id=event.area_id,
                event_id=event_id,
                resource_status=override,
            )
        return ResourceStatusView(
            scope="area_default",
            area_id=event.area_id,
            event_id=event_id,
            resource_status=self.get_area_resource_status(event.area_id),
        )

    def list_rag_documents(self) -> list[RAGDocument]:
        return self.rag_service.list_documents()

    def import_rag_documents(self, request: RAGDocumentImportRequest) -> list[RAGDocument]:
        documents = [item.to_document() for item in request.documents]
        validate_rag_document_payload(documents)
        imported = self.rag_service.import_documents(documents)
        self.add_audit_record(
            source_type="runtime_admin",
            action="rag_documents_imported",
            summary=f"已导入 {len(imported)} 份运行期知识文档。",
            details={"document_ids": [item.doc_id for item in imported]},
        )
        return imported

    def reload_rag_documents(self) -> list[RAGDocument]:
        documents = self.rag_service.reload_rag_store()
        self.add_audit_record(
            source_type="runtime_admin",
            action="rag_documents_reloaded",
            summary=f"已重载 {len(documents)} 份运行期知识文档。",
            details={"document_ids": [item.doc_id for item in documents]},
        )
        return documents
