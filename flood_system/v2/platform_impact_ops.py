from __future__ import annotations

from ..models import CorpusType, RiskLevel
from .models import (
    Advisory,
    AdvisoryRequest,
    EntityImpactView,
    EntityProfile,
    EntityType,
    EvidenceItem,
    PolicyConstraint,
)


class PlatformImpactOpsMixin:
    def list_entity_profiles(
        self,
        *,
        area_id: str | None = None,
        entity_type: str | None = None,
    ) -> list[EntityProfile]:
        return self.repository.list_v2_entity_profiles(area_id=area_id, entity_type=entity_type)

    def get_entity_profile(self, entity_id: str) -> EntityProfile:
        entity = self.repository.get_v2_entity_profile(entity_id)
        if entity is None:
            raise ValueError(f"Unknown entity_id: {entity_id}")
        return entity

    def save_entity_profile(self, entity: EntityProfile) -> EntityProfile:
        if entity.area_id not in self.area_profiles:
            raise ValueError(f"Unknown area_id: {entity.area_id}")
        self.repository.save_v2_entity_profile(entity)
        self.add_audit_record(
            source_type="runtime_admin",
            action="entity_profile_saved",
            summary=f"已保存运行期对象画像 {entity.name}。",
            details={"entity_id": entity.entity_id, "area_id": entity.area_id},
        )
        return entity

    def delete_entity_profile(self, entity_id: str) -> None:
        if not self.repository.delete_v2_entity_profile(entity_id):
            raise ValueError(f"Unknown entity_id: {entity_id}")
        self.add_audit_record(
            source_type="runtime_admin",
            action="entity_profile_deleted",
            summary=f"已删除运行期对象画像 {entity_id}。",
            details={"entity_id": entity_id},
        )

    def get_entity_impact(self, entity_id: str, *, event_id: str | None = None) -> EntityImpactView:
        entity = self.get_entity_profile(entity_id)
        resolved_event_id = event_id or self.repository.get_latest_v2_event_id(entity.area_id)
        if resolved_event_id is None:
            raise ValueError(f"No active v2 event found for area {entity.area_id}.")
        hazard_state = self.get_hazard_state(resolved_event_id)
        area_profile = self.area_profiles[entity.area_id]
        resource_status = self.get_resource_status(entity.area_id, event_id=resolved_event_id)
        evidence = self.get_knowledge_evidence(
            event_id=resolved_event_id,
            area_id=entity.area_id,
            entity_id=entity.entity_id,
        )
        return self.exposure_engine.assess_entity(
            resolved_event_id,
            entity,
            area_profile,
            hazard_state,
            resource_status,
            evidence=evidence,
        )

    def get_exposure_summary(self, event_id: str, *, entity_type: str | None = None, top_k: int = 5):
        event = self.get_event(event_id)
        hazard_state = self.get_hazard_state(event_id)
        area_profile = self.area_profiles[event.area_id]
        resource_status = self.get_resource_status(event.area_id, event_id=event_id)
        profiles = self.list_entity_profiles(area_id=event.area_id, entity_type=entity_type)
        filtered = {item.entity_id: item for item in profiles}
        summary = self.exposure_engine.summarize(
            event_id,
            area_profile,
            hazard_state,
            filtered,
            resource_status,
            evidence=self.get_knowledge_evidence(event_id=event_id, area_id=event.area_id, entity_id=None),
        )
        summary.affected_entities = summary.affected_entities[:top_k]
        return summary

    def generate_advisory(self, request: AdvisoryRequest) -> Advisory:
        if request.entity_id:
            impact = self.get_entity_impact(request.entity_id, event_id=request.event_id)
        else:
            impact = self._impact_from_location_request(request)
        advisory = self.generate_advisory_for_impact(impact, request=request)
        self.repository.save_v2_advisory(advisory)
        self.repository.add_v2_stream_record_for_payload(
            advisory.event_id,
            "advisory_generated",
            {
                "entity_id": advisory.entity_id,
                "requires_human_confirmation": advisory.requires_human_confirmation,
                "confidence": advisory.confidence,
            },
        )
        return advisory

    def generate_advisory_for_impact(
        self,
        impact: EntityImpactView,
        *,
        request: AdvisoryRequest | None = None,
    ) -> Advisory:
        request = request or AdvisoryRequest(
            event_id=impact.event_id,
            entity_id=impact.entity.entity_id,
            area_id=impact.entity.area_id,
        )
        advisory = self.decision_engine.generate_advisory(
            request=request,
            impact=impact,
            additional_evidence=self.get_knowledge_evidence(
                event_id=impact.event_id,
                area_id=impact.entity.area_id,
                entity_id=impact.entity.entity_id,
            ),
            allow_proposal=False,
        )
        advisory.proposal = None
        return advisory

    def draft_action_proposal(self, event_id: str, entity_id: str):
        return None

    def get_shelter_capacity(self, area_id: str) -> list[dict]:
        return [
            {
                "shelter_id": item.shelter_id,
                "name": item.name,
                "village": item.village,
                "capacity": item.capacity,
                "available_capacity": item.available_capacity,
                "accessible": item.accessible,
            }
            for item in self.area_profiles[area_id].shelters
        ]

    def get_live_traffic(self, event_id: str) -> list[dict]:
        event = self.get_event(event_id)
        traffic = self.route_planner.build_live_traffic(
            self.area_profiles[event.area_id],
            self.get_hazard_state(event_id),
        )
        return [
            {"road_id": item.road_id, "congestion_index": item.congestion_index, "note": item.note}
            for item in traffic
        ]

    def get_policy_constraints(self, entity_type: str, risk_level: str) -> PolicyConstraint:
        return self.decision_engine.get_policy_constraints(EntityType(entity_type), RiskLevel(risk_level))

    def resolve_target_entity(
        self,
        event_id: str,
        question: str,
        *,
        preferred_entity_id: str | None = None,
        entity_type: str | None = None,
    ) -> dict:
        if preferred_entity_id:
            entity = self.get_entity_profile(preferred_entity_id)
            return {
                "entity_id": entity.entity_id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type.value,
                "reason": f"Resolved {entity.name} from the current plan context.",
            }
        exposure = self.get_exposure_summary(event_id, entity_type=entity_type, top_k=1)
        if exposure.affected_entities:
            entity = exposure.affected_entities[0].entity
            return {
                "entity_id": entity.entity_id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type.value,
                "reason": f"Resolved {entity.name} from exposure ranking for: {question}",
            }
        raise ValueError("当前问题没有可用的暴露对象。")

    def get_knowledge_evidence(
        self,
        *,
        event_id: str,
        area_id: str,
        entity_id: str | None = None,
    ) -> list[EvidenceItem]:
        entity = None
        if entity_id:
            try:
                entity = self.get_entity_profile(entity_id)
            except ValueError:
                entity = None
        risk_level = self.get_hazard_state(event_id).overall_risk_level
        return self._knowledge_evidence(entity, risk_level, area_id=area_id)

    def _impact_from_location_request(self, request: AdvisoryRequest) -> EntityImpactView:
        area_id = request.area_id
        village = request.village or request.profile_overrides.get("village") or self.area_profiles[area_id].villages[0]
        synthetic = EntityProfile(
            entity_id="ad_hoc_location",
            area_id=area_id,
            entity_type=EntityType(request.profile_overrides.get("entity_type", "resident")),
            name=request.profile_overrides.get("name", request.location_hint or "临时建议对象"),
            village=village,
            location_hint=request.location_hint or village,
            resident_count=int(request.profile_overrides.get("resident_count", 1)),
            current_occupancy=int(
                request.profile_overrides.get(
                    "current_occupancy",
                    request.profile_overrides.get("resident_count", 1),
                )
            ),
            vulnerability_tags=list(request.profile_overrides.get("vulnerability_tags", [])),
            mobility_constraints=list(request.profile_overrides.get("mobility_constraints", [])),
            key_assets=list(request.profile_overrides.get("key_assets", [])),
            inventory_summary=str(request.profile_overrides.get("inventory_summary", "")),
            continuity_requirement=str(request.profile_overrides.get("continuity_requirement", "")),
            preferred_transport_mode=request.profile_overrides.get("preferred_transport_mode", "walk"),
            notification_preferences=list(request.profile_overrides.get("notification_preferences", [])),
            emergency_contacts=[],
            custom_attributes=dict(request.profile_overrides),
        )
        resolved_event_id = request.event_id or self.repository.get_latest_v2_event_id(area_id)
        if resolved_event_id is None:
            raise ValueError(f"No active v2 event found for area {area_id}.")
        hazard_state = self.get_hazard_state(resolved_event_id)
        return self.exposure_engine.assess_entity(
            hazard_state.event_id,
            synthetic,
            self.area_profiles[area_id],
            hazard_state,
            self.get_resource_status(area_id, event_id=resolved_event_id),
            evidence=self.get_knowledge_evidence(
                event_id=hazard_state.event_id,
                area_id=area_id,
                entity_id=None,
            ),
        )

    def _knowledge_evidence(
        self,
        entity: EntityProfile | None,
        risk_level: RiskLevel,
        *,
        area_id: str | None = None,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        resolved_area_id = entity.area_id if entity is not None else (area_id or "beilin_10km2")
        filters = {"region": self.area_profiles[resolved_area_id].region}
        policy_docs = self.rag_service.query(
            CorpusType.POLICY,
            f"{risk_level.value} response evacuation",
            filters=filters,
            top_k=2,
        )
        case_docs = self.rag_service.query(
            CorpusType.CASE,
            "vulnerable school factory flood",
            filters=filters,
            top_k=2,
        )
        profile_docs = self.rag_service.query(
            CorpusType.PROFILE,
            "shelter vulnerable medical",
            filters=filters,
            top_k=2,
        )
        memory_records = self.long_term_memory_store.query_memories(
            area_id=resolved_area_id,
            entity_type=entity.entity_type.value if entity is not None else None,
            risk_level=risk_level,
            top_k=2,
        )
        for priority, docs, evidence_type in (
            (2, policy_docs, "policy"),
            (3, case_docs, "case"),
            (4, profile_docs, "profile"),
        ):
            for doc in docs:
                evidence.append(
                    EvidenceItem(
                        evidence_type=evidence_type,
                        title=doc.title,
                        source_id=doc.doc_id,
                        excerpt=doc.content[:120],
                        priority=priority,
                        retrieval_explain=self.rag_service.explain(doc),
                    )
                )
        for record in memory_records:
            evidence.append(
                EvidenceItem(
                    evidence_type="memory",
                    title=record.headline,
                    source_id=record.memory_id,
                    excerpt=record.summary[:120],
                    priority=5,
                    retrieval_explain={
                        "tags": record.tags,
                        "entity_types": record.entity_types,
                        "action_types": record.action_types,
                        "source_summary_id": record.source_summary_id,
                    },
                )
            )
        if entity is not None:
            evidence.insert(
                0,
                EvidenceItem(
                    evidence_type="profile",
                    title=f"{entity.name}画像快照",
                    source_id=entity.entity_id,
                    excerpt=(
                        f"{entity.location_hint}; type={entity.entity_type.value}; "
                        f"key vulnerability={','.join(entity.vulnerability_tags[:3]) or 'none'}"
                    ),
                    priority=1,
                )
            )
        return evidence
