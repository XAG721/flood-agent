from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from ..models import CorpusType, RAGDocument
from .evidence_governance import (
    REQUIRED_TASK_FIELDS,
    TASK_FIELD_QUERIES,
    build_field_evidence_map,
    candidate_nli_pairs,
    infer_task_field_support,
)

from .models import (
    ApprovalPolicy,
    EventRiskObject,
    EvidenceRole,
    EvidenceFieldState,
    EvidenceConflict,
    EvidenceConflictResolutionRequest,
    EvidenceFreezeRequest,
    EvidenceMissingReason,
    EvidenceManualSupplementRequest,
    EvidenceNliAssessment,
    EvidencePackageVersion,
    OperatorRole,
    NliRelation,
    RetrievalMode,
    TaskDraftGenerationRequest,
    TaskEvidenceRef,
)



class EvidenceWorkflowMixin:
    def list_evidence_packages(
        self, event_id: str, *, object_id: str | None = None
    ) -> list[EvidencePackageVersion]:
        self._event(event_id)
        return self.repository.list_evidence_packages(event_id, object_id=object_id)

    def get_evidence_package(self, package_id: str) -> EvidencePackageVersion:
        package = self.repository.get_latest_evidence_package(package_id)
        if package is None:
            raise LookupError(f"evidence package not found: {package_id}")
        return package

    def list_evidence_package_versions(
        self, package_id: str
    ) -> list[EvidencePackageVersion]:
        versions = self.repository.list_evidence_package_versions(package_id)
        if not versions:
            raise LookupError(f"evidence package not found: {package_id}")
        return versions

    def compare_evidence_package_versions(
        self,
        package_id: str,
        *,
        from_version: int | None = None,
        to_version: int | None = None,
    ) -> dict:
        versions = self.list_evidence_package_versions(package_id)
        by_version = {item.version: item for item in versions}
        target_version = to_version or versions[-1].version
        base_version = from_version or max(1, target_version - 1)
        before = by_version.get(base_version)
        after = by_version.get(target_version)
        if before is None or after is None:
            raise LookupError(
                f"evidence package comparison versions not found: {base_version}->{target_version}"
            )
        if before.version >= after.version:
            raise ValueError("from_version must be earlier than to_version")
        fields = sorted(set(before.field_states) | set(after.field_states))
        field_state_changes = {
            field: {
                "before": before.field_states.get(field),
                "after": after.field_states.get(field),
            }
            for field in fields
            if before.field_states.get(field) != after.field_states.get(field)
        }
        missing_reason_changes = {
            field: {
                "before": before.missing_reasons.get(field),
                "after": after.missing_reasons.get(field),
            }
            for field in sorted(set(before.missing_reasons) | set(after.missing_reasons))
            if before.missing_reasons.get(field) != after.missing_reasons.get(field)
        }
        before_sources = {item.source_id for item in before.evidence}
        after_sources = {item.source_id for item in after.evidence}
        before_conflicts = {
            item.conflict_id: item.resolution_status for item in before.conflicts
        }
        after_conflicts = {
            item.conflict_id: item.resolution_status for item in after.conflicts
        }
        conflict_changes = {
            conflict_id: {
                "before": before_conflicts.get(conflict_id),
                "after": after_conflicts.get(conflict_id),
            }
            for conflict_id in sorted(set(before_conflicts) | set(after_conflicts))
            if before_conflicts.get(conflict_id) != after_conflicts.get(conflict_id)
        }
        return {
            "package_id": package_id,
            "from_version": before.version,
            "to_version": after.version,
            "from_hash": before.content_hash,
            "to_hash": after.content_hash,
            "field_state_changes": field_state_changes,
            "missing_reason_changes": missing_reason_changes,
            "added_source_ids": sorted(after_sources - before_sources),
            "removed_source_ids": sorted(before_sources - after_sources),
            "conflict_changes": conflict_changes,
            "nli_status_change": {
                "before": before.nli_status,
                "after": after.nli_status,
            },
        }

    def _build_evidence_package(
        self,
        *,
        event_id: str,
        object_id: str,
        evidence: list[TaskEvidenceRef],
        role_coverage: dict[str, bool],
        validation_errors: list[str],
        conflicts: list[EvidenceConflict],
        nli_assessments: list[EvidenceNliAssessment] | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.SHADOW,
        retrieval_trace: dict | None = None,
        action: str,
        plan_basis: list,
        operator_id: str,
    ) -> EvidencePackageVersion:
        evidence = self._normalize_evidence_refs(evidence)
        nli_assessments = nli_assessments or []
        supported = EvidenceFieldState.SUPPORTED
        missing = EvidenceFieldState.MISSING
        field_evidence_map = build_field_evidence_map(evidence)
        field_states = {
            field: supported if field_evidence_map[field] else missing
            for field in TASK_FIELD_QUERIES
        }
        for conflict in conflicts:
            field_states[conflict.field_name] = EvidenceFieldState.CONFLICTED
        missing_fields = [
            key for key, value in field_states.items() if value == missing
        ]
        blocking_missing_fields = [
            field for field in REQUIRED_TASK_FIELDS if field in missing_fields
        ]
        missing_reason = (
            EvidenceMissingReason.SOURCE_UNAVAILABLE
            if self.rag_service is None
            else EvidenceMissingReason.NOT_RETRIEVED
        )
        missing_reasons = {field: missing_reason for field in missing_fields}
        nli_status = self._nli_status(nli_assessments)
        nli_model_version = str(
            getattr(self.nli_adapter, "model_version", "nli-unavailable")
        )
        canonical = json.dumps(
            {
                "event_id": event_id,
                "object_id": object_id,
                "task_schema_version": self.TASK_SCHEMA_VERSION,
                "field_states": {
                    key: value.value for key, value in field_states.items()
                },
                "role_coverage": role_coverage,
                "required_fields": list(REQUIRED_TASK_FIELDS),
                "field_evidence_map": field_evidence_map,
                "missing_reasons": {
                    key: value.value for key, value in missing_reasons.items()
                },
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "conflicts": [item.model_dump(mode="json") for item in conflicts],
                "nli_status": nli_status,
                "nli_model_version": nli_model_version,
                "nli_assessments": [
                    item.model_dump(mode="json") for item in nli_assessments
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = self._now()
        complete = not validation_errors and not blocking_missing_fields and not conflicts
        return EvidencePackageVersion(
            package_id=self._id("EVID"),
            event_id=event_id,
            object_id=object_id,
            status="frozen" if complete else "needs_review",
            task_schema_version=self.TASK_SCHEMA_VERSION,
            retrieval_run_id=self._id("RETRIEVAL"),
            field_states=field_states,
            role_coverage=role_coverage,
            evidence=evidence,
            conflicts=conflicts,
            missing_fields=missing_fields,
            required_fields=list(REQUIRED_TASK_FIELDS),
            blocking_missing_fields=blocking_missing_fields,
            missing_reasons=missing_reasons,
            field_evidence_map=field_evidence_map,
            nli_status=nli_status,
            nli_model_version=nli_model_version,
            nli_assessments=nli_assessments,
            content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            created_by=operator_id,
            reviewed_by=operator_id if complete else None,
            frozen_at=now if complete else None,
            created_at=now,
            retrieval_mode=retrieval_mode,
            baseline_source_ids=list(
                (retrieval_trace or {}).get("baseline_source_ids", [])
            ),
            frc_source_ids=list((retrieval_trace or {}).get("frc_source_ids", [])),
            shadow_comparison=dict((retrieval_trace or {}).get("comparison", {})),
        )

    def resolve_evidence_conflict(
        self,
        package_id: str,
        conflict_id: str,
        request: EvidenceConflictResolutionRequest,
    ) -> EvidencePackageVersion:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER},
            "resolve an evidence conflict",
        )
        current = self.repository.get_latest_evidence_package(package_id)
        if current is None:
            raise LookupError(f"evidence package not found: {package_id}")
        target = next(
            (item for item in current.conflicts if item.conflict_id == conflict_id),
            None,
        )
        if target is None:
            raise LookupError(f"evidence conflict not found: {conflict_id}")
        if not set(request.selected_source_ids).issubset(
            set(target.evidence_source_ids)
        ):
            raise ValueError(
                "selected evidence sources must belong to the conflict group"
            )
        conflicts = [
            item.model_copy(
                update={
                    "resolution_status": "resolved",
                    "resolution_reason": request.reason,
                    "selected_source_ids": request.selected_source_ids,
                }
            )
            if item.conflict_id == conflict_id
            else item
            for item in current.conflicts
        ]
        unresolved = [
            item for item in conflicts if item.resolution_status != "resolved"
        ]
        field_states = dict(current.field_states)
        if not any(item.field_name == target.field_name for item in unresolved):
            field_states[target.field_name] = EvidenceFieldState.SUPPORTED
        canonical = json.dumps(
            {
                "previous_hash": current.content_hash,
                "field_states": {
                    key: value.value for key, value in field_states.items()
                },
                "conflicts": [item.model_dump(mode="json") for item in conflicts],
                "evidence": [item.model_dump(mode="json") for item in current.evidence],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = self._now()
        frozen = not unresolved and not current.blocking_missing_fields
        updated = current.model_copy(
            update={
                "version": current.version + 1,
                "status": "frozen" if frozen else "needs_review",
                "field_states": field_states,
                "conflicts": conflicts,
                "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "reviewed_by": request.operator_id,
                "frozen_at": now if frozen else None,
                "created_at": now,
            }
        )
        self.repository.save_evidence_package(updated)
        self._record(
            event_id=current.event_id,
            object_id=current.object_id,
            entry_type="evidence_package",
            action="evidence_conflict_resolved",
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
            detail={
                "package_id": package_id,
                "version": updated.version,
                "conflict_id": conflict_id,
                "selected_source_ids": request.selected_source_ids,
                "reason": request.reason,
            },
        )
        return updated

    def supplement_evidence_package(
        self, package_id: str, request: EvidenceManualSupplementRequest
    ) -> EvidencePackageVersion:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.ADMIN},
            "supplement an evidence package",
        )
        current = self.get_evidence_package(package_id)
        if current.status == "frozen":
            raise ValueError(
                "frozen evidence packages are immutable; generate a new retrieval run"
            )
        by_source = {item.source_id: item for item in current.evidence}
        for item in request.evidence:
            by_source[item.source_id] = item
        return self._revise_evidence_package(
            current,
            evidence=list(by_source.values()),
            actor_id=request.operator_id,
            actor_role=request.operator_role,
            terminal_id=request.terminal_id,
            reason=request.reason,
            freeze_when_complete=request.freeze_when_complete,
            action="manual_evidence_supplemented",
        )

    def freeze_evidence_package(
        self, package_id: str, request: EvidenceFreezeRequest
    ) -> EvidencePackageVersion:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.ADMIN},
            "freeze an evidence package",
        )
        current = self.get_evidence_package(package_id)
        if current.status == "frozen":
            return current
        return self._revise_evidence_package(
            current,
            evidence=current.evidence,
            actor_id=request.operator_id,
            actor_role=request.operator_role,
            terminal_id=request.terminal_id,
            reason=request.reason,
            freeze_when_complete=True,
            action="evidence_package_frozen",
        )

    def _revise_evidence_package(
        self,
        current: EvidencePackageVersion,
        *,
        evidence: list[TaskEvidenceRef],
        actor_id: str,
        actor_role: OperatorRole,
        terminal_id: str,
        reason: str,
        freeze_when_complete: bool,
        action: str,
    ) -> EvidencePackageVersion:
        evidence = self._normalize_evidence_refs(evidence)
        covered_roles = {
            role.value
            for item in evidence
            if item.source_type != "workflow_contract"
            for role in item.roles
        }
        role_coverage = {
            role.value: role.value in covered_roles for role in EvidenceRole
        }
        field_evidence_map = build_field_evidence_map(evidence)
        field_states = {
            field: EvidenceFieldState.SUPPORTED
            if field_evidence_map[field]
            else EvidenceFieldState.MISSING
            for field in TASK_FIELD_QUERIES
        }
        detected_conflicts, nli_assessments = self._analyze_evidence_conflicts(evidence)
        prior_conflicts = {item.conflict_id: item for item in current.conflicts}
        conflicts = []
        for conflict in detected_conflicts:
            previous = prior_conflicts.get(conflict.conflict_id)
            if previous is not None and previous.resolution_status == "resolved":
                conflict = conflict.model_copy(
                    update={
                        "resolution_status": previous.resolution_status,
                        "resolution_reason": previous.resolution_reason,
                        "selected_source_ids": previous.selected_source_ids,
                    }
                )
            conflicts.append(conflict)
        unresolved = [
            item for item in conflicts if item.resolution_status != "resolved"
        ]
        for conflict in unresolved:
            field_states[conflict.field_name] = EvidenceFieldState.CONFLICTED
        missing_fields = [
            field
            for field, state in field_states.items()
            if state == EvidenceFieldState.MISSING
        ]
        blocking_missing_fields = [
            field for field in REQUIRED_TASK_FIELDS if field in missing_fields
        ]
        missing_reasons = {
            field: current.missing_reasons.get(
                field,
                EvidenceMissingReason.SOURCE_UNAVAILABLE
                if self.rag_service is None
                else EvidenceMissingReason.NOT_RETRIEVED,
            )
            for field in missing_fields
        }
        complete = (
            not blocking_missing_fields
            and not unresolved
            and all(role_coverage.values())
        )
        if freeze_when_complete and not complete:
            raise ValueError(
                "evidence package cannot be frozen while fields are missing or conflicts are unresolved"
            )
        canonical = json.dumps(
            {
                "previous_hash": current.content_hash,
                "field_states": {
                    key: value.value for key, value in field_states.items()
                },
                "role_coverage": role_coverage,
                "required_fields": list(REQUIRED_TASK_FIELDS),
                "field_evidence_map": field_evidence_map,
                "missing_reasons": {
                    key: value.value for key, value in missing_reasons.items()
                },
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "conflicts": [item.model_dump(mode="json") for item in conflicts],
                "nli_assessments": [
                    item.model_dump(mode="json") for item in nli_assessments
                ],
                "reason": reason,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        now = self._now()
        frozen = freeze_when_complete and complete
        updated = current.model_copy(
            update={
                "version": current.version + 1,
                "status": "frozen" if frozen else "needs_review",
                "field_states": field_states,
                "role_coverage": role_coverage,
                "evidence": evidence,
                "conflicts": conflicts,
                "missing_fields": missing_fields,
                "required_fields": list(REQUIRED_TASK_FIELDS),
                "blocking_missing_fields": blocking_missing_fields,
                "missing_reasons": missing_reasons,
                "field_evidence_map": field_evidence_map,
                "nli_status": self._nli_status(nli_assessments),
                "nli_model_version": str(
                    getattr(self.nli_adapter, "model_version", "nli-unavailable")
                ),
                "nli_assessments": nli_assessments,
                "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "reviewed_by": actor_id,
                "frozen_at": now if frozen else None,
                "created_at": now,
            }
        )
        self.repository.save_evidence_package(updated)
        self._record(
            event_id=current.event_id,
            object_id=current.object_id,
            entry_type="evidence_package",
            action=action,
            actor_id=actor_id,
            actor_role=actor_role.value,
            terminal_id=terminal_id,
            detail={
                "package_id": current.package_id,
                "version": updated.version,
                "reason": reason,
            },
        )
        return updated

    def _run_retrieval_strategy(
        self, query: str, mode: RetrievalMode
    ) -> tuple[list[RAGDocument], dict]:
        if self.rag_service is None:
            return [], {
                "baseline_source_ids": [],
                "frc_source_ids": [],
                "comparison": {},
            }
        baseline_query = getattr(self.rag_service, "query", None)
        baseline = self._filter_current_policy_documents(
            baseline_query(CorpusType.POLICY, query, top_k=6) if baseline_query else []
        )
        frc = self._query_task_evidence(query)
        baseline_ids = [item.doc_id for item in baseline]
        frc_ids = [item.doc_id for item in frc]
        comparison = {
            "overlap": sorted(set(baseline_ids) & set(frc_ids)),
            "baseline_only": sorted(set(baseline_ids) - set(frc_ids)),
            "frc_only": sorted(set(frc_ids) - set(baseline_ids)),
        }
        if mode in {RetrievalMode.BASELINE_ONLY, RetrievalMode.SHADOW}:
            selected = baseline
        elif mode == RetrievalMode.REVIEW:
            selected = list({item.doc_id: item for item in [*baseline, *frc]}.values())[
                :8
            ]
        else:
            selected = frc
        return selected, {
            "baseline_source_ids": baseline_ids,
            "frc_source_ids": frc_ids,
            "comparison": comparison,
        }

    @staticmethod
    def _filter_current_policy_documents(
        documents: list[RAGDocument],
    ) -> list[RAGDocument]:
        return [
            document
            for document in documents
            if EvidenceWorkflowMixin._document_is_current(document)
            and str(document.metadata.get("stage", "")).strip().lower()
            not in {"compensation", "recovery", "postmortem"}
        ]

    @staticmethod
    def _document_is_current(document: RAGDocument) -> bool:
        metadata = document.metadata
        status = str(metadata.get("status", "active")).strip().lower()
        if status not in {"", "active", "published"}:
            return False
        now = datetime.now(timezone.utc)

        def parse(value) -> datetime | None:
            if value is None or value == "":
                return None
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        effective_at = parse(metadata.get("effective_at"))
        expires_at = parse(metadata.get("expires_at"))
        if metadata.get("effective_at") and effective_at is None:
            return False
        if metadata.get("expires_at") and expires_at is None:
            return False
        return not (
            (effective_at is not None and effective_at > now)
            or (expires_at is not None and expires_at <= now)
        )

    @staticmethod
    def _build_task_query(alert, risk_object: EventRiskObject) -> str:
        return " ".join(
            [
                alert.level,
                alert.disaster_type,
                risk_object.object_type,
                risk_object.name,
                risk_object.vulnerability,
                *risk_object.trigger_reasons,
                "触发条件 风险对象 责任主体 处置动作 完成时限 资源依赖 "
                "反馈要求 升级条件 例外条件 文件版本 条款依据",
            ]
        )

    def _build_task_evidence(
        self,
        alert,
        risk_object: EventRiskObject,
        documents: list[RAGDocument],
        *,
        request: TaskDraftGenerationRequest,
    ) -> list[TaskEvidenceRef]:
        evidence = [
            TaskEvidenceRef(
                source_type="alert_snapshot",
                source_id=alert.snapshot_id,
                title=f"{alert.source_department}{alert.disaster_type}预警 V{alert.version}",
                excerpt=f"{alert.level}；影响范围：{alert.affected_area}；{alert.raw_content[:140]}",
                roles=[EvidenceRole.CONDITION],
                document_version=str(alert.version),
                trust_score=1.0,
                source_locator=f"response://alerts/{alert.snapshot_id}",
                field_support={"trigger_condition": 1.0},
            ),
            TaskEvidenceRef(
                source_type="risk_object_registry",
                source_id=risk_object.object_id,
                title=f"{risk_object.name}事件对象快照",
                excerpt=(
                    f"{risk_object.object_type}；{risk_object.location}；脆弱性：{risk_object.vulnerability}；"
                    f"责任：{risk_object.responsible_organization}/{risk_object.responsible_role}"
                ),
                roles=[EvidenceRole.OBJECT, EvidenceRole.RESPONSIBILITY],
                document_version="event-snapshot-v1",
                trust_score=1.0,
                source_locator=(
                    f"response://events/{risk_object.event_id}/risk-objects/"
                    f"{risk_object.object_id}"
                ),
                field_support={
                    "risk_object": 1.0,
                    "responsible_party": 1.0,
                },
            ),
            TaskEvidenceRef(
                source_type="workflow_contract",
                source_id="response-task-schema-v2:deadline",
                title="响应任务时限字段契约",
                excerpt=(
                    f"任务必须在 {request.acknowledge_minutes} 分钟内确认，并在 "
                    f"{request.deadline_minutes} 分钟内完成；审批时校验接收、开始、完成和核验时限顺序。"
                ),
                roles=[EvidenceRole.ATTRIBUTION],
                document_version=self.TASK_SCHEMA_VERSION,
                source_locator="contract://response-task-schema-v2#deadline",
                field_support={"deadline": 1.0},
                trust_score=1.0,
            ),
            TaskEvidenceRef(
                source_type="workflow_contract",
                source_id="response-task-schema-v2:feedback",
                title="响应任务反馈证据契约",
                excerpt=(
                    "任务反馈必须绑定时间、位置和附件证据；系统按风险对象类型校验照片、视频、回执或检查记录。"
                ),
                roles=[EvidenceRole.ATTRIBUTION],
                document_version=self.TASK_SCHEMA_VERSION,
                source_locator="contract://response-task-schema-v2#required-evidence",
                field_support={"feedback_requirement": 1.0},
                trust_score=1.0,
            ),
            TaskEvidenceRef(
                source_type="workflow_contract",
                source_id=f"{self.RULE_SET_VERSION}:escalation",
                title="响应任务升级规则契约",
                excerpt="任务未确认、执行超时或反馈受阻时，系统升级至防办审核员并保留人工接管和改派记录。",
                roles=[EvidenceRole.ATTRIBUTION],
                document_version=self.RULE_SET_VERSION,
                source_locator=f"contract://{self.RULE_SET_VERSION}#escalation",
                field_support={"escalation_condition": 1.0},
                trust_score=1.0,
            ),
        ]
        for document in documents:
            roles = self._document_roles(document)
            if not roles:
                continue
            metadata = document.metadata
            selection = metadata.get("_evidence_selection", {})
            trust_score = (
                selection.get("trust_score") if isinstance(selection, dict) else None
            )
            evidence.append(
                TaskEvidenceRef(
                    source_type="plan_document",
                    source_id=document.doc_id,
                    title=document.title,
                    excerpt=document.content[:180],
                    roles=roles,
                    document_version=self._document_version(document),
                    document_version_id=self._metadata_value(
                        metadata, "document_version_id"
                    ),
                    clause=self._metadata_value(metadata, "section_number", "clause"),
                    source_locator=self._metadata_value(metadata, "source_locator")
                    or (
                        f"rag://policy/{document.doc_id}#clause="
                        f"{self._metadata_value(metadata, 'section_number', 'clause') or 'unknown'}"
                    ),
                    page_number=self._metadata_positive_int(metadata, "page_number"),
                    section_path=self._metadata_string_list(metadata, "section_path"),
                    table_name=self._metadata_value(metadata, "table_name"),
                    row_start=self._metadata_positive_int(metadata, "row_start"),
                    row_end=self._metadata_positive_int(metadata, "row_end"),
                    field_support=infer_task_field_support(
                        document.content,
                        [role.value for role in roles],
                        metadata,
                    ),
                    conflicts_with=self._metadata_string_list(
                        metadata, "conflicts_with"
                    ),
                    trust_score=trust_score
                    if isinstance(trust_score, (int, float))
                    else None,
                    conflict_key=self._metadata_value(
                        metadata, "conflict_key", "decision_key"
                    ),
                    conflict_value=self._metadata_value(
                        metadata, "conflict_value", "decision_value"
                    ),
                    jurisdiction=self._metadata_value(
                        metadata, "jurisdiction", "area_id"
                    ),
                    superseded=str(metadata.get("status", "")).lower()
                    in {"superseded", "expired", "invalid"},
                )
            )
        return evidence

    @staticmethod
    def _metadata_positive_int(metadata: dict, key: str) -> int | None:
        try:
            value = int(metadata.get(key))
        except (TypeError, ValueError):
            return None
        return value if value >= 1 else None

    @staticmethod
    def _metadata_string_list(metadata: dict, key: str) -> list[str]:
        value = metadata.get(key, [])
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _stable_evidence_id(prefix: str, *parts: object) -> str:
        canonical = "|".join(str(part) for part in parts)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}-{digest}"

    @staticmethod
    def _conflict_dimensions(field_name: str) -> list[str]:
        lowered = field_name.casefold()
        mapping = {
            "version": ("version", "版本", "effective", "时效", "status"),
            "jurisdiction": ("jurisdiction", "辖区", "区域", "area"),
            "subject": ("subject", "主体", "责任", "department", "role"),
            "threshold": ("threshold", "阈值", "level", "等级"),
            "time": ("time", "deadline", "时限", "时间", "minute", "hour"),
            "action": ("action", "动作", "处置", "procedure", "流程"),
        }
        dimensions = [
            dimension
            for dimension, keywords in mapping.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        return dimensions or ["semantic"]

    @staticmethod
    def _nli_status(assessments: list[EvidenceNliAssessment]) -> str:
        statuses = {item.status for item in assessments}
        if not assessments:
            return "not_required"
        if "error" in statuses:
            return "error"
        if statuses == {"unavailable"}:
            return "unavailable"
        if "unavailable" in statuses:
            return "partial"
        return "completed"

    def _analyze_evidence_conflicts(
        self, evidence: list[TaskEvidenceRef]
    ) -> tuple[list[EvidenceConflict], list[EvidenceNliAssessment]]:
        groups: dict[str, list[TaskEvidenceRef]] = {}
        for item in evidence:
            if item.superseded or not item.conflict_key or not item.conflict_value:
                continue
            groups.setdefault(item.conflict_key, []).append(item)

        conflicts: dict[str, EvidenceConflict] = {}
        for key, items in groups.items():
            values = {item.conflict_value for item in items}
            source_ids = sorted({item.source_id for item in items})
            if len(values) < 2 or len(source_ids) < 2:
                continue
            conflict_id = self._stable_evidence_id(
                "CONFLICT", "decision_value", key, *source_ids
            )
            conflicts[conflict_id] = EvidenceConflict(
                conflict_id=conflict_id,
                field_name=key,
                conflict_type="DECISION_VALUE",
                severity="critical",
                evidence_source_ids=source_ids,
                conflict_dimensions=self._conflict_dimensions(key),
                detection_methods=["deterministic_rule"],
            )

        by_id = {item.source_id: item for item in evidence}
        for item in evidence:
            for other_id in item.conflicts_with:
                if other_id not in by_id or other_id == item.source_id:
                    continue
                source_ids = sorted({item.source_id, other_id})
                shared_fields = sorted(
                    set(item.field_support) & set(by_id[other_id].field_support)
                )
                field_name = shared_fields[0] if shared_fields else "source_assertion"
                conflict_id = self._stable_evidence_id(
                    "CONFLICT", "explicit_source", field_name, *source_ids
                )
                conflicts.setdefault(
                    conflict_id,
                    EvidenceConflict(
                        conflict_id=conflict_id,
                        field_name=field_name,
                        conflict_type="EXPLICIT_SOURCE_CONFLICT",
                        severity="high",
                        evidence_source_ids=source_ids,
                        conflict_dimensions=self._conflict_dimensions(field_name),
                        detection_methods=["source_metadata"],
                    ),
                )

        assessments: list[EvidenceNliAssessment] = []
        for left, right, shared_fields in candidate_nli_pairs(evidence):
            source_ids = sorted([left.source_id, right.source_id])
            assessment_id = self._stable_evidence_id(
                "NLI", *source_ids, *shared_fields
            )
            try:
                result = self.nli_adapter.classify(left.excerpt, right.excerpt)
                relation = NliRelation(str(result.relation).lower())
                confidence = max(0.0, min(1.0, float(result.confidence)))
                status = str(result.status)
                model_version = str(result.model_version)
                error = str(result.error)
            except Exception as exc:  # fail explicit; rule-based detection remains active
                relation = NliRelation.ERROR
                confidence = 0.0
                status = "error"
                model_version = str(
                    getattr(self.nli_adapter, "model_version", "nli-unknown")
                )
                error = f"{type(exc).__name__}: {exc}"
            assessment = EvidenceNliAssessment(
                assessment_id=assessment_id,
                left_source_id=left.source_id,
                right_source_id=right.source_id,
                shared_fields=shared_fields,
                relation=relation,
                confidence=confidence,
                model_version=model_version,
                status=status,
                error=error,
            )
            assessments.append(assessment)
            if relation != NliRelation.CONTRADICTION:
                continue
            for field_name in shared_fields:
                conflict_id = self._stable_evidence_id(
                    "CONFLICT", "nli", field_name, *source_ids
                )
                conflicts.setdefault(
                    conflict_id,
                    EvidenceConflict(
                        conflict_id=conflict_id,
                        field_name=field_name,
                        conflict_type="SEMANTIC_CONTRADICTION",
                        severity="high",
                        evidence_source_ids=source_ids,
                        conflict_dimensions=self._conflict_dimensions(field_name),
                        detection_methods=["versioned_nli"],
                        nli_assessment_id=assessment_id,
                        nli_relation=relation,
                        nli_confidence=confidence,
                    ),
                )
        return list(conflicts.values()), assessments

    def _detect_evidence_conflicts(
        self, evidence: list[TaskEvidenceRef]
    ) -> list[EvidenceConflict]:
        conflicts, _ = self._analyze_evidence_conflicts(evidence)
        return conflicts

    @classmethod
    def _document_roles(cls, document: RAGDocument) -> list[EvidenceRole]:
        metadata_roles = document.metadata.get("evidence_roles", [])
        explicit = (
            {str(item) for item in metadata_roles}
            if isinstance(metadata_roles, list)
            else set()
        )
        text = f"{document.title} {document.content}"
        roles: list[EvidenceRole] = []
        keyword_map = {
            EvidenceRole.CONDITION: ("当", "达到", "预警", "阶段", "出现", "持续"),
            EvidenceRole.OBJECT: (
                "下穿",
                "地铁",
                "学校",
                "社区",
                "地下",
                "对象",
                "片区",
            ),
            EvidenceRole.RESPONSIBILITY: (
                "责任",
                "防办",
                "部门",
                "交警",
                "住建",
                "联络员",
                "值班",
            ),
            EvidenceRole.PROCEDURE: (
                "应",
                "需",
                "组织",
                "核查",
                "启动",
                "封控",
                "转移",
                "引导",
                "联动",
            ),
            EvidenceRole.EXCEPTION: (
                "若",
                "如",
                "否则",
                "受阻",
                "超时",
                "不足",
                "争议",
                "必要时",
            ),
        }
        for role, keywords in keyword_map.items():
            if role.value in explicit or any(keyword in text for keyword in keywords):
                roles.append(role)
        version = cls._document_version(document)
        clause = cls._metadata_value(document.metadata, "section_number", "clause")
        if EvidenceRole.ATTRIBUTION.value in explicit or (version and clause):
            roles.append(EvidenceRole.ATTRIBUTION)
        return list(dict.fromkeys(roles))

    @classmethod
    def _build_plan_basis(cls, documents: list[RAGDocument]):
        from .models import PlanBasis

        basis: list[PlanBasis] = []
        seen: set[tuple[str, str, str]] = set()
        for document in documents:
            version = cls._document_version(document)
            clause = cls._metadata_value(document.metadata, "section_number", "clause")
            if not version or not clause:
                continue
            key = (document.title, version, clause)
            if key in seen:
                continue
            seen.add(key)
            basis.append(
                PlanBasis(document=document.title, version=version, clause=clause)
            )
        return basis

    @staticmethod
    def _extract_supported_action(
        documents: list[RAGDocument], risk_object: EventRiskObject
    ) -> str:
        action_markers = (
            "应",
            "需",
            "组织",
            "核查",
            "启动",
            "封控",
            "转移",
            "引导",
            "联动",
        )
        object_markers = [risk_object.object_type]
        if any(keyword in risk_object.object_type for keyword in ("下穿", "隧道")):
            object_markers.extend(["下穿", "积水", "道路"])
        ranked: list[tuple[int, int, int, str]] = []
        for document_index, document in enumerate(documents):
            stage = str(document.metadata.get("stage", "")).lower()
            audience = str(document.metadata.get("audience", "")).lower()
            for sentence_index, sentence in enumerate(
                re.split(r"[。；;\n]+", document.content)
            ):
                cleaned = sentence.strip()
                if len(cleaned) < 8 or not any(
                    marker in cleaned for marker in action_markers
                ):
                    continue
                score = 4 * int(
                    any(marker and marker in cleaned for marker in object_markers)
                )
                score += 2 * int(stage == "response") + int(stage == "warning")
                score += int(audience == "government")
                ranked.append((score, document_index, sentence_index, cleaned))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        sentences: list[str] = []
        for _, _, _, sentence in ranked:
            if sentence not in sentences:
                sentences.append(sentence)
            if len(sentences) >= 3:
                break
        return "；".join(sentences) + ("。" if sentences else "")

    @staticmethod
    def _required_evidence(object_type: str) -> list[str]:
        if any(keyword in object_type for keyword in ("隧道", "下穿", "易涝", "地下")):
            return ["现场照片", "积水深度", "到场时间", "管控状态"]
        if any(keyword in object_type for keyword in ("学校", "医院", "养老", "社区")):
            return ["现场照片", "人员清点", "处置结果", "完成时间"]
        return ["现场照片", "处置结果", "到场时间"]

    @staticmethod
    def _cooperate_roles(object_type: str) -> list[str]:
        if any(keyword in object_type for keyword in ("隧道", "下穿", "道路", "易涝")):
            return ["交警联络员", "排水单位负责人", "属地街道值班员"]
        return ["属地街道值班员", "成员单位联络员"]

    @staticmethod
    def _approval_policy(alert_level: str, action: str) -> ApprovalPolicy:
        high_risk_levels = ("橙", "红", "orange", "red")
        high_risk_actions = ("转移", "封控", "停课", "停工", "公开发布", "资源调度")
        normalized = alert_level.lower()
        if any(level in normalized for level in high_risk_levels) or any(
            item in action for item in high_risk_actions
        ):
            return ApprovalPolicy.COMMANDER_REQUIRED
        return ApprovalPolicy.REVIEWER_REQUIRED

    @staticmethod
    def _document_version(document: RAGDocument) -> str | None:
        return EvidenceWorkflowMixin._metadata_value(
            document.metadata, "document_version", "version", "effective_version"
        )

    @staticmethod
    def _metadata_value(metadata: dict, *keys: str) -> str | None:
        for key in keys:
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _build_grounding_summary(
        evidence: list[TaskEvidenceRef],
        role_coverage: dict[str, bool],
        validation_errors: list[str],
    ) -> str:
        covered = [role for role, value in role_coverage.items() if value]
        plan_sources = [
            item for item in evidence if item.source_type == "plan_document"
        ]
        if validation_errors:
            return f"证据闸门未通过：覆盖 {len(covered)}/6 个角色；" + "；".join(
                validation_errors
            )
        return (
            f"证据闸门通过：6/6 角色完整，使用 {len(plan_sources)} 条预案证据、"
            "1 版专业预警快照和 1 份事件对象责任快照形成草案。"
        )
