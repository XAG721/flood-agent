from __future__ import annotations

import hashlib
import json

from .candidate_discovery import (
    build_candidate_feature_snapshot,
    build_registry_candidate,
    build_risk_object_candidate,
    calculate_profile_risk_score,
    calculate_registry_risk_score,
    point_in_polygon,
)
from .risk_object_ingestion import parse_risk_object_file

from .models import (
    AlertLifecycleStatus,
    CandidateDiscoveryRequest,
    CandidateDiscoveryResult,
    CandidateObjectListFreezeRequest,
    CandidateObjectListVersion,
    CandidateObjectReference,
    CandidateRunRecord,
    EventRiskObject,
    EventStatus,
    ObjectVerificationStatus,
    OperatorRole,
    RiskObjectBatchRequest,
    RiskObjectImportQuarantineItem,
    RiskObjectInput,
    RiskObjectRegistryBatchImportRequest,
    RiskObjectRegistryFileImportRequest,
    RiskObjectRegistryImportResult,
    RiskObjectRegistryRecord,
    RiskObjectRegistryVersionSnapshot,
    RiskObjectVerificationRequest,
    RiskObjectVersionSnapshot,
)

from .service_shared import EDIT_ROLES


class RiskObjectWorkflowMixin:
    def add_risk_objects(
        self, event_id: str, request: RiskObjectBatchRequest
    ) -> list[EventRiskObject]:
        self._require_role(
            request.operator_role, EDIT_ROLES, "add candidate risk objects"
        )
        self._event(event_id, active=True)
        created: list[EventRiskObject] = []
        for payload in request.objects:
            change_type = "candidate_added"
            duplicate_source_id: str | None = None
            if payload.duplicate_of:
                canonical = self.repository.get_event_risk_object(
                    event_id, payload.duplicate_of
                )
                if canonical is None:
                    raise ValueError(
                        f"duplicate risk object references unknown canonical object: {payload.duplicate_of}"
                    )
                duplicate_source_id = payload.object_id
                merged = canonical.model_dump(include=set(RiskObjectInput.model_fields))
                merged.update(
                    {
                        "canonical_object_id": canonical.canonical_object_id
                        or canonical.object_id,
                        "aliases": sorted(
                            set(
                                canonical.aliases
                                + payload.aliases
                                + [payload.name, payload.object_id]
                            )
                        ),
                        "source_refs": list(
                            dict.fromkeys(canonical.source_refs + payload.source_refs)
                        ),
                        "duplicate_of": None,
                    }
                )
                payload = RiskObjectInput(**merged)
                change_type = "candidate_duplicate_merged"
            existing = self.repository.get_event_risk_object(
                event_id, payload.object_id
            )
            now = self._now()
            registry_expired = payload.registry_status != "active" or (
                payload.registry_valid_until is not None
                and payload.registry_valid_until <= now
            )
            item = EventRiskObject(
                **payload.model_copy(
                    update={
                        "canonical_object_id": payload.canonical_object_id
                        or payload.object_id
                    }
                ).model_dump(),
                event_id=event_id,
                verification_status=(
                    ObjectVerificationStatus.PENDING
                    if existing and existing.stale
                    else existing.verification_status
                    if existing
                    else ObjectVerificationStatus.PENDING
                ),
                verified_by=existing.verified_by if existing else None,
                verified_at=existing.verified_at if existing else None,
                verification_note=existing.verification_note if existing else "",
                created_at=existing.created_at if existing else now,
                version=(existing.version + 1) if existing else 1,
                updated_at=now,
                stale=registry_expired,
            )
            self.repository.save_event_risk_object(item)
            self._save_risk_object_version(
                item,
                change_type=change_type
                if duplicate_source_id
                else "candidate_updated"
                if existing
                else "candidate_added",
                operator_id=request.operator_id,
                terminal_id=request.terminal_id,
            )
            created.append(item)
            action = (
                change_type
                if duplicate_source_id
                else "candidate_updated"
                if existing
                else "candidate_added"
            )
            self._record(
                event_id=event_id,
                object_id=item.object_id,
                entry_type="risk_object",
                action=action,
                actor_id=request.operator_id,
                actor_role=request.operator_role.value,
                terminal_id=request.terminal_id,
                before_state={"risk_score": existing.risk_score} if existing else {},
                after_state={
                    "risk_score": item.risk_score,
                    "verification_status": item.verification_status.value,
                },
                detail={
                    "risk_score": item.risk_score,
                    "sources": item.source_refs,
                    "canonical_object_id": item.canonical_object_id,
                    "aliases": item.aliases,
                    "duplicate_source_id": duplicate_source_id,
                    "registry_status": item.registry_status,
                    "registry_valid_until": item.registry_valid_until.isoformat()
                    if item.registry_valid_until
                    else None,
                },
            )
        return created

    def list_risk_object_registry(
        self,
        *,
        area_id: str,
        operator_role: OperatorRole,
        include_inactive: bool = False,
    ) -> list[RiskObjectRegistryRecord]:
        records = self.repository.list_risk_object_registry(
            area_id=area_id,
            include_inactive=include_inactive,
        )
        full_sensitive_roles = {
            OperatorRole.DUTY_OFFICER,
            OperatorRole.REVIEWER,
            OperatorRole.COMMANDER,
            OperatorRole.ADMIN,
        }
        precise_location_roles = full_sensitive_roles | {
            OperatorRole.LIAISON,
            OperatorRole.FIELD_OPERATOR,
        }
        result: list[RiskObjectRegistryRecord] = []
        for record in records:
            updates: dict[str, object] = {}
            if operator_role not in precise_location_roles:
                updates.update(
                    {
                        "location": "[精确位置已按岗位权限脱敏]",
                        "longitude": None,
                        "latitude": None,
                    }
                )
            if operator_role not in full_sensitive_roles:
                updates["sensitive_contacts"] = []
                if record.special_population_notes:
                    updates["special_population_notes"] = (
                        "[特殊人群信息已按岗位权限脱敏]"
                    )
            result.append(record.model_copy(update=updates))
        return result

    def list_risk_object_registry_versions(
        self,
        *,
        area_id: str,
        operator_role: OperatorRole,
        object_id: str | None = None,
    ) -> list[RiskObjectRegistryVersionSnapshot]:
        self._require_role(
            operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.ADMIN},
            "view risk-object master-data history",
        )
        return self.repository.list_risk_object_registry_versions(
            area_id=area_id,
            object_id=object_id,
        )

    def list_risk_object_imports(
        self, *, area_id: str, operator_role: OperatorRole
    ) -> list[RiskObjectRegistryImportResult]:
        self._require_role(
            operator_role,
            {
                OperatorRole.REVIEWER,
                OperatorRole.COMMANDER,
                OperatorRole.AUDITOR,
                OperatorRole.ADMIN,
            },
            "view risk-object import history",
        )
        return self.repository.list_risk_object_imports(area_id=area_id)

    def import_risk_object_registry(
        self, request: RiskObjectRegistryBatchImportRequest
    ) -> RiskObjectRegistryImportResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "import risk-object master data",
        )
        object_ids = [item.object_id for item in request.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(
                "one registry import cannot contain duplicate object_id values"
            )
        canonical = json.dumps(
            {
                "area_id": request.area_id,
                "source_version": request.source_version,
                "objects": [
                    item.model_dump(mode="json")
                    for item in sorted(request.objects, key=lambda item: item.object_id)
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self._commit_risk_object_registry_import(
            area_id=request.area_id,
            source_filename=request.source_filename,
            source_format="api_json",
            source_version=request.source_version,
            source_hash=hashlib.sha256(canonical).hexdigest(),
            source_bytes=len(canonical),
            objects=request.objects,
            quarantine=[],
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            terminal_id=request.terminal_id,
        )

    def import_risk_object_registry_file(
        self, request: RiskObjectRegistryFileImportRequest
    ) -> RiskObjectRegistryImportResult:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "import a risk-object master-data file",
        )
        parsed = parse_risk_object_file(
            request.file,
            source_version=request.source_version,
        )
        return self._commit_risk_object_registry_import(
            area_id=request.area_id,
            source_filename=request.file.filename,
            source_format=parsed.source_format,
            source_version=request.source_version,
            source_hash=parsed.source_hash,
            source_bytes=parsed.source_bytes,
            objects=parsed.objects,
            quarantine=parsed.quarantine,
            operator_id=request.operator_id,
            operator_role=request.operator_role,
            terminal_id=request.terminal_id,
        )

    def _commit_risk_object_registry_import(
        self,
        *,
        area_id: str,
        source_filename: str,
        source_format: str,
        source_version: str,
        source_hash: str,
        source_bytes: int,
        objects: list[RiskObjectInput],
        quarantine: list[RiskObjectImportQuarantineItem],
        operator_id: str,
        operator_role: OperatorRole,
        terminal_id: str,
    ) -> RiskObjectRegistryImportResult:
        now = self._now()
        records: list[RiskObjectRegistryRecord] = []
        registry_snapshots: list[RiskObjectRegistryVersionSnapshot] = []
        changed_object_ids: list[str] = []
        created_count = 0
        updated_count = 0
        unchanged_count = 0
        existing_records = {
            item.object_id: item
            for item in self.repository.list_risk_object_registry(
                area_id=area_id,
                include_inactive=True,
            )
        }
        incoming_records = {item.object_id: item for item in objects}
        accepted_count = 0
        accepted_simulation_flags: list[bool] = []
        quarantine_items = list(quarantine)
        for item in objects:
            if (
                not item.duplicate_of
                and item.canonical_object_id
                and item.canonical_object_id != item.object_id
            ):
                quarantine_items.append(
                    RiskObjectImportQuarantineItem(
                        source_id=item.object_id,
                        reason_code="INVALID_CANONICAL_OBJECT",
                        reason=(
                            "a canonical record must use its own object_id as canonical_object_id"
                        ),
                    )
                )
                continue
            if item.duplicate_of:
                canonical = incoming_records.get(
                    item.duplicate_of
                ) or existing_records.get(item.duplicate_of)
                duplicate_error = ""
                if item.duplicate_of == item.object_id:
                    duplicate_error = "duplicate_of cannot reference the same object"
                elif canonical is None:
                    duplicate_error = "duplicate_of references an object outside the current area registry"
                elif canonical.duplicate_of:
                    duplicate_error = "duplicate_of must reference a canonical object, not another duplicate"
                elif (
                    item.canonical_object_id
                    and item.canonical_object_id != item.duplicate_of
                ):
                    duplicate_error = "canonical_object_id must equal duplicate_of for duplicate records"
                if duplicate_error:
                    quarantine_items.append(
                        RiskObjectImportQuarantineItem(
                            source_id=item.object_id,
                            reason_code="INVALID_CANONICAL_OBJECT",
                            reason=duplicate_error,
                        )
                    )
                    continue
            accepted_count += 1
            accepted_simulation_flags.append(item.is_simulated)
            existing = existing_records.get(item.object_id)
            item_payload = item.model_dump()
            item_payload.update(
                {
                    "canonical_object_id": item.canonical_object_id
                    or item.duplicate_of
                    or item.object_id,
                    "source_version": source_version,
                    "data_version": "pending-content-hash",
                }
            )
            preliminary = RiskObjectInput.model_validate(item_payload)
            content_payload = preliminary.model_dump(
                mode="json",
                exclude={"data_version"},
            )
            content_hash = hashlib.sha256(
                json.dumps(
                    content_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            normalized = preliminary.model_copy(
                update={
                    "data_version": f"{source_version}:{content_hash[:12]}",
                }
            )
            if existing is not None:
                if existing.content_hash == content_hash:
                    unchanged_count += 1
                    continue
            registry_version = existing.registry_version + 1 if existing else 1
            record = RiskObjectRegistryRecord(
                **normalized.model_dump(),
                area_id=area_id,
                registry_version=registry_version,
                source_hash=source_hash,
                content_hash=content_hash,
                source_filename=source_filename,
                created_by=existing.created_by if existing else operator_id,
                terminal_id=terminal_id,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            records.append(record)
            registry_snapshots.append(
                RiskObjectRegistryVersionSnapshot(
                    snapshot_id=self._id("REGVER"),
                    area_id=area_id,
                    object_id=record.object_id,
                    registry_version=record.registry_version,
                    record=record,
                    change_type="registry_updated" if existing else "registry_created",
                    changed_by=operator_id,
                    terminal_id=terminal_id,
                    created_at=now,
                )
            )
            changed_object_ids.append(record.object_id)
            if existing:
                updated_count += 1
            else:
                created_count += 1

        active_events = [
            event
            for event in self.repository.list_response_events()
            if event.area_id == area_id and event.status != EventStatus.CLOSED
        ]
        stale_runs: list[CandidateRunRecord] = []
        stale_event_objects: list[EventRiskObject] = []
        event_object_snapshots: list[RiskObjectVersionSnapshot] = []
        affected_event_ids: set[str] = set()
        changed_set = set(changed_object_ids)
        if changed_set:
            reason = (
                f"risk-object registry import {source_hash[:12]} changed master data"
            )
            for event in active_events:
                for run in self.repository.list_candidate_runs(event.event_id):
                    if run.status == "stale":
                        continue
                    stale_runs.append(
                        run.model_copy(
                            update={
                                "status": "stale",
                                "stale_at": now,
                                "stale_reason": reason,
                            }
                        )
                    )
                    affected_event_ids.add(event.event_id)
                for event_object in self.repository.list_event_risk_objects(
                    event.event_id
                ):
                    if event_object.object_id not in changed_set or event_object.stale:
                        continue
                    stale = event_object.model_copy(
                        update={
                            "stale": True,
                            "version": event_object.version + 1,
                            "updated_at": now,
                        }
                    )
                    stale_event_objects.append(stale)
                    event_object_snapshots.append(
                        RiskObjectVersionSnapshot(
                            snapshot_id=self._id("OBJVER"),
                            event_id=stale.event_id,
                            object_id=stale.object_id,
                            version=stale.version,
                            object=stale,
                            change_type="registry_change_marked_stale",
                            changed_by=operator_id,
                            terminal_id=terminal_id,
                            created_at=now,
                        )
                    )
                    affected_event_ids.add(event.event_id)

        result = RiskObjectRegistryImportResult(
            import_id=self._id("REGIMPORT"),
            area_id=area_id,
            source_filename=source_filename,
            source_format=source_format,
            source_version=source_version,
            source_hash=source_hash,
            source_bytes=source_bytes,
            imported_count=accepted_count,
            created_count=created_count,
            updated_count=updated_count,
            unchanged_count=unchanged_count,
            quarantined_count=len(quarantine_items),
            changed_object_ids=sorted(changed_object_ids),
            affected_event_ids=sorted(affected_event_ids),
            stale_candidate_run_count=len(stale_runs),
            quarantine=quarantine_items,
            is_simulated=bool(accepted_simulation_flags)
            and all(accepted_simulation_flags),
            imported_by=operator_id,
            terminal_id=terminal_id,
            created_at=now,
        )
        self.repository.commit_risk_object_registry_import(
            result=result,
            records=records,
            registry_snapshots=registry_snapshots,
            stale_candidate_runs=stale_runs,
            stale_event_objects=stale_event_objects,
            event_object_snapshots=event_object_snapshots,
        )
        for event_id in sorted(affected_event_ids):
            self._record(
                event_id=event_id,
                entry_type="risk_object_registry",
                action="risk_object_registry_imported",
                actor_id=operator_id,
                actor_role=operator_role.value,
                terminal_id=terminal_id,
                before_state={},
                after_state={"candidate_runs": "stale"},
                detail={
                    "import_id": result.import_id,
                    "source_hash": source_hash,
                    "changed_object_ids": result.changed_object_ids,
                    "quarantined_count": result.quarantined_count,
                },
            )
        return result

    def discover_risk_objects(
        self, event_id: str, request: CandidateDiscoveryRequest
    ) -> CandidateDiscoveryResult:
        self._require_role(
            request.operator_role,
            EDIT_ROLES,
            "discover candidate risk objects",
        )
        event = self._event(event_id, active=True)
        alerts = self.repository.list_alert_snapshots(event_id)
        if not alerts:
            raise ValueError("event has no alert snapshot")
        alert = alerts[-1]
        if alert.lifecycle_status in {
            AlertLifecycleStatus.REVOKED,
            AlertLifecycleStatus.EXPIRED,
        }:
            raise ValueError(
                "candidate discovery is disabled for revoked or expired warning revisions"
            )
        run_id = self._id("CANDRUN")
        now = self._now()
        registry_records = [
            item
            for item in self.repository.list_risk_object_registry(
                area_id=event.area_id,
                include_inactive=False,
            )
            if item.duplicate_of is None
            and (item.registry_valid_from is None or item.registry_valid_from <= now)
            and (item.registry_valid_until is None or item.registry_valid_until > now)
        ]
        requested_types = {
            item.strip().casefold() for item in request.entity_types if item.strip()
        }
        uses_core_registry = bool(registry_records)
        if uses_core_registry:
            profiles: list[object] = list(registry_records)
            if requested_types:
                profiles = [
                    item
                    for item in profiles
                    if isinstance(item, RiskObjectRegistryRecord)
                    and item.object_type.strip().casefold() in requested_types
                ]
            registry_fingerprint = [
                {
                    "object_id": item.object_id,
                    "registry_version": item.registry_version,
                    "content_hash": item.content_hash,
                }
                for item in registry_records
            ]
            risk_object_data_version = (
                "registry:"
                + hashlib.sha256(
                    json.dumps(
                        registry_fingerprint,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            algorithm_version = "candidate-registry-rules-v1"
            feature_version = "candidate-registry-features-v2"
        else:
            legacy_profiles = self.repository.list_v2_entity_profiles(
                area_id=event.area_id
            )
            if requested_types:
                legacy_profiles = [
                    item
                    for item in legacy_profiles
                    if item.entity_type.value.casefold() in requested_types
                ]
            profiles = list(legacy_profiles)
            risk_object_data_version = "legacy:v2-entity-profiles-v1"
            algorithm_version = "candidate-rules-v1"
            feature_version = "candidate-features-v2"
        scanned_profiles = len(profiles)
        spatially_evaluated = 0
        spatially_matched = 0
        excluded_unlocated = 0
        association_mode = "area_registry"
        limitations: list[str] = []
        if alert.affected_geometry is not None:
            association_mode = "gis_point_in_polygon"
            located_profiles = [
                item
                for item in profiles
                if getattr(item, "longitude", None) is not None
                and getattr(item, "latitude", None) is not None
            ]
            excluded_unlocated = len(profiles) - len(located_profiles)
            spatially_evaluated = len(located_profiles)
            profiles = [
                item
                for item in located_profiles
                if point_in_polygon(
                    float(getattr(item, "longitude")),
                    float(getattr(item, "latitude")),
                    alert.affected_geometry.coordinates,
                )
            ]
            spatially_matched = len(profiles)
            if excluded_unlocated:
                limitations.append(
                    f"{excluded_unlocated} 个对象缺少 EPSG:4326 坐标，未纳入本次精确空间筛查。"
                )
        else:
            limitations.append(
                "预警未提供 EPSG:4326 多边形，本次采用 area_id 区域关联，不代表精确 GIS 几何相交。"
            )
        scored: list[tuple[float, object]] = []
        for profile in profiles:
            if isinstance(profile, RiskObjectRegistryRecord):
                score = calculate_registry_risk_score(alert.level, profile)
            else:
                score = calculate_profile_risk_score(alert.level, profile)
            if score >= request.min_risk_score:
                scored.append((score, profile))
        scored.sort(
            key=lambda item: (
                -item[0],
                str(
                    getattr(
                        item[1],
                        "object_id",
                        getattr(item[1], "entity_id", ""),
                    )
                ),
            )
        )
        selected = scored[: request.max_candidates]
        objects: list[RiskObjectInput] = []
        candidate_features = []
        for rank, (score, profile) in enumerate(selected, start=1):
            candidate_features.append(
                build_candidate_feature_snapshot(
                    profile,
                    score=score,
                    rank=rank,
                    alert=alert,
                    association_mode=association_mode,
                    now=now,
                )
            )
            if isinstance(profile, RiskObjectRegistryRecord):
                objects.append(
                    build_registry_candidate(
                        profile,
                        score,
                        alert,
                        association_mode,
                    )
                )
            else:
                objects.append(
                    build_risk_object_candidate(
                        profile,
                        score,
                        alert,
                        association_mode,
                    )
                )
        candidates = (
            self.add_risk_objects(
                event_id,
                RiskObjectBatchRequest(
                    objects=objects,
                    operator_id=request.operator_id,
                    operator_role=request.operator_role,
                    terminal_id=request.terminal_id,
                ),
            )
            if objects
            else []
        )
        if candidates:
            candidates = [
                item.model_copy(update={"candidate_run_id": run_id})
                for item in candidates
            ]
            for item in candidates:
                self.repository.save_event_risk_object(item)
        self._record(
            event_id=event_id,
            entry_type="risk_object_discovery",
            action="candidate_discovery_completed",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "association_mode": association_mode,
                "candidate_run_id": run_id,
                "area_id": event.area_id,
                "registry_source": (
                    "response_risk_object_registry"
                    if uses_core_registry
                    else "legacy_v2_entity_profiles"
                ),
                "risk_object_data_version": risk_object_data_version,
                "alert_snapshot_id": alert.snapshot_id,
                "scanned_profiles": scanned_profiles,
                "spatially_evaluated_profiles": spatially_evaluated,
                "spatially_matched_profiles": spatially_matched,
                "excluded_unlocated_profiles": excluded_unlocated,
                "matched_profiles": len(candidates),
                "min_risk_score": request.min_risk_score,
                "feature_families": [
                    "spatial",
                    "temporal",
                    "attribute",
                    "semantic",
                    "data_quality",
                ],
            },
        )
        missing_feature_set = {
            feature
            for snapshot in candidate_features
            for feature in snapshot.missing_features
        }
        if alert.affected_geometry is None:
            missing_feature_set.add("affected_geometry")
        missing_features = sorted(missing_feature_set)
        run = CandidateRunRecord(
            run_id=run_id,
            event_id=event_id,
            alert_snapshot_id=alert.snapshot_id,
            risk_object_data_version=risk_object_data_version,
            algorithm_version=algorithm_version,
            feature_version=feature_version,
            association_mode=association_mode,
            parameters={
                "entity_types": sorted(requested_types),
                "min_risk_score": request.min_risk_score,
                "max_candidates": request.max_candidates,
            },
            candidate_object_ids=[item.object_id for item in candidates],
            candidate_features=candidate_features,
            missing_features=missing_features,
            limitations=limitations,
            created_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_candidate_run(run)
        return CandidateDiscoveryResult(
            run_id=run_id,
            event_id=event_id,
            alert_snapshot_id=alert.snapshot_id,
            association_mode=association_mode,
            scanned_profiles=scanned_profiles,
            matched_profiles=len(candidates),
            spatially_evaluated_profiles=spatially_evaluated,
            spatially_matched_profiles=spatially_matched,
            excluded_unlocated_profiles=excluded_unlocated,
            candidates=candidates,
            limitations=limitations,
            algorithm_version=algorithm_version,
            feature_version=feature_version,
            data_version=risk_object_data_version,
        )

    def verify_risk_object(
        self, event_id: str, object_id: str, request: RiskObjectVerificationRequest
    ) -> EventRiskObject:
        self._require_role(request.operator_role, EDIT_ROLES, "verify a risk object")
        self._event(event_id, active=True)
        if request.decision == ObjectVerificationStatus.PENDING:
            raise ValueError("verification decision must be confirmed or excluded")
        item = self.repository.get_event_risk_object(event_id, object_id)
        if item is None:
            raise LookupError(f"risk object not found: {object_id}")
        if item.registry_status != "active" or (
            item.registry_valid_until is not None
            and item.registry_valid_until <= self._now()
        ):
            raise ValueError(
                "expired or inactive risk-object registry records cannot be confirmed"
            )
        previous_status = item.verification_status
        item = item.model_copy(
            update={
                "verification_status": request.decision,
                "verified_by": request.operator_id,
                "verified_at": self._now(),
                "verification_note": request.note,
                "stale": False,
                "version": item.version + 1,
                "updated_at": self._now(),
            }
        )
        self.repository.save_event_risk_object(item)
        self._save_risk_object_version(
            item,
            change_type=f"candidate_{request.decision.value}",
            operator_id=request.operator_id,
            terminal_id=request.terminal_id,
        )
        self._record(
            event_id=event_id,
            object_id=object_id,
            entry_type="risk_object",
            action=f"candidate_{request.decision.value}",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            before_state={"verification_status": previous_status.value},
            after_state={"verification_status": item.verification_status.value},
            detail={"note": request.note},
        )
        return item

    @staticmethod
    def _candidate_object_verification_hash(item: EventRiskObject) -> str:
        canonical = json.dumps(
            {
                "object_id": item.object_id,
                "version": item.version,
                "verification_status": item.verification_status.value,
                "verified_by": item.verified_by,
                "verified_at": item.verified_at.isoformat()
                if item.verified_at
                else None,
                "candidate_run_id": item.candidate_run_id,
                "data_version": item.data_version,
                "source_refs": item.source_refs,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def list_candidate_object_lists(
        self, event_id: str
    ) -> list[CandidateObjectListVersion]:
        self._event(event_id)
        return self.repository.list_candidate_object_lists(event_id)

    def freeze_candidate_object_list(
        self,
        event_id: str,
        request: CandidateObjectListFreezeRequest,
    ) -> CandidateObjectListVersion:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "freeze a confirmed candidate object list",
        )
        event = self._event(event_id, active=True)
        alerts = self.repository.list_alert_snapshots(event_id)
        if not alerts:
            raise ValueError("event has no alert snapshot")
        alert = alerts[-1]
        existing_lists = self.repository.list_candidate_object_lists(event_id)
        latest = existing_lists[-1] if existing_lists else None
        if (
            request.expected_version is not None
            and (latest is None or request.expected_version != latest.version)
        ):
            actual = latest.version if latest else 0
            raise ValueError(
                f"version conflict: candidate object list expected {request.expected_version}, actual {actual}"
            )
        if len(set(request.object_ids)) != len(request.object_ids):
            raise ValueError("candidate object list contains duplicate object_ids")

        all_objects = {
            item.object_id: item
            for item in self.repository.list_event_risk_objects(event_id)
        }
        selected_ids = request.object_ids or sorted(
            item.object_id
            for item in all_objects.values()
            if item.verification_status == ObjectVerificationStatus.CONFIRMED
            and not item.stale
        )
        if not selected_ids:
            raise ValueError(
                "candidate object list requires at least one confirmed non-stale object"
            )

        runs = {
            item.run_id: item for item in self.repository.list_candidate_runs(event_id)
        }
        references: list[CandidateObjectReference] = []
        source_run_ids: set[str] = set()
        for object_id in sorted(selected_ids):
            item = all_objects.get(object_id)
            if item is None:
                raise LookupError(f"risk object not found: {object_id}")
            if (
                item.verification_status != ObjectVerificationStatus.CONFIRMED
                or item.stale
            ):
                raise ValueError(
                    f"candidate object list may only contain confirmed non-stale objects: {object_id}"
                )
            if item.candidate_run_id:
                run = runs.get(item.candidate_run_id)
                if run is None:
                    raise ValueError(
                        f"candidate run is missing for confirmed object: {object_id}"
                    )
                if run.status != "completed" or run.stale_at is not None:
                    raise ValueError(
                        f"stale candidate runs cannot be frozen: {run.run_id}"
                    )
                if run.alert_snapshot_id != alert.snapshot_id:
                    raise ValueError(
                        f"candidate run does not use the latest warning revision: {run.run_id}"
                    )
                source_run_ids.add(run.run_id)
            references.append(
                CandidateObjectReference(
                    object_id=item.object_id,
                    object_version=item.version,
                    candidate_run_id=item.candidate_run_id,
                    data_version=item.data_version,
                    verification_hash=self._candidate_object_verification_hash(item),
                )
            )

        canonical = json.dumps(
            {
                "event_id": event_id,
                "alert_snapshot_id": alert.snapshot_id,
                "source_run_ids": sorted(source_run_ids),
                "objects": [item.model_dump(mode="json") for item in references],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if latest is not None and latest.content_hash == content_hash:
            return latest

        now = self._now()
        frozen = CandidateObjectListVersion(
            list_id=f"CANDLIST-{hashlib.sha256(event_id.encode('utf-8')).hexdigest()[:16]}",
            event_id=event_id,
            version=(latest.version + 1) if latest else 1,
            alert_snapshot_id=alert.snapshot_id,
            source_run_ids=sorted(source_run_ids),
            objects=references,
            content_hash=content_hash,
            note=request.note,
            is_simulated=event.is_simulated
            and all(all_objects[item.object_id].is_simulated for item in references),
            frozen_by=request.operator_id,
            frozen_role=request.operator_role,
            terminal_id=request.terminal_id,
            frozen_at=now,
        )
        self.repository.save_candidate_object_list(frozen)
        self._record(
            event_id=event_id,
            entry_type="candidate_object_list",
            action="candidate_object_list_frozen",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            after_state={
                "list_id": frozen.list_id,
                "version": frozen.version,
                "status": frozen.status,
            },
            detail={
                "content_hash": frozen.content_hash,
                "object_ids": [item.object_id for item in frozen.objects],
                "source_run_ids": frozen.source_run_ids,
                "alert_snapshot_id": frozen.alert_snapshot_id,
                "note": request.note,
            },
        )
        return frozen

    def _resolve_candidate_object_list(
        self,
        event_id: str,
        object_id: str,
        requested_version: int | None,
    ) -> CandidateObjectListVersion | None:
        versions = self.repository.list_candidate_object_lists(event_id)
        if not versions:
            return None
        latest = versions[-1]
        if requested_version is None:
            selected = latest
        else:
            selected = next(
                (item for item in versions if item.version == requested_version),
                None,
            )
            if selected is None:
                raise LookupError(
                    f"candidate object list version not found: {requested_version}"
                )
        if selected.version != latest.version:
            raise ValueError(
                "task drafts must use the latest frozen candidate object list"
            )
        alerts = self.repository.list_alert_snapshots(event_id)
        if not alerts or selected.alert_snapshot_id != alerts[-1].snapshot_id:
            raise ValueError(
                "frozen candidate object list does not match the latest warning revision"
            )
        reference = next(
            (item for item in selected.objects if item.object_id == object_id),
            None,
        )
        if reference is None:
            raise ValueError(
                "risk object is not present in the latest frozen candidate object list"
            )
        current = self.repository.get_event_risk_object(event_id, object_id)
        if current is None:
            raise LookupError(f"risk object not found: {object_id}")
        if (
            current.version != reference.object_version
            or self._candidate_object_verification_hash(current)
            != reference.verification_hash
        ):
            raise ValueError(
                "frozen candidate object list no longer matches the current verified object"
            )
        return selected
