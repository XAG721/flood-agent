from __future__ import annotations

import hashlib
import json

from ..models import CorpusType, RAGDocument
from .document_governance import (
    ParsedDocument,
    build_document_source,
    parse_document_source,
)
from .evidence_governance import (
    infer_task_field_support,
)

from .models import (
    ContractVersionRecord,
    DocumentLifecycleEvent,
    DocumentLifecycleStatus,
    DocumentParseRecord,
    DocumentParseRequest,
    DocumentPublishRequest,
    DocumentRetireRequest,
    DocumentVersionCreateRequest,
    IndexBuildRecord,
    IndexBuildRequest,
    IndexBuildStatus,
    DocumentImportRequest,
    DocumentVersionRecord,
    EvidenceRole,
    OperatorRole,
    ResponseTask,
    RuleOutcome,
    TaskActionRequest,
)



class DocumentWorkflowMixin:
    def list_document_versions(
        self, document_id: str | None = None
    ) -> list[DocumentVersionRecord]:
        return [
            self._materialize_document_version(item)
            for item in self.repository.list_document_versions(document_id)
        ]

    def get_document_version(self, version_id: str) -> DocumentVersionRecord:
        record = self.repository.get_document_version(version_id)
        if record is None:
            raise LookupError(f"document version not found: {version_id}")
        return self._materialize_document_version(record)

    def document_version_history(self, version_id: str) -> dict:
        record = self.repository.get_document_version(version_id)
        if record is None:
            raise LookupError(f"document version not found: {version_id}")
        source = self.repository.get_document_source(version_id)
        return {
            "document": self._materialize_document_version(record),
            "source": (
                {
                    "filename": source.filename,
                    "media_type": source.media_type,
                    "source_hash": source.source_hash,
                    "source_size": source.source_size,
                    "ocr_engine_version": source.ocr_engine_version,
                    "created_at": source.created_at,
                }
                if source
                else None
            ),
            "lifecycle": self.repository.list_document_lifecycle_events(version_id),
            "parses": self.repository.list_document_parses(version_id),
            "index_builds": self.repository.list_index_builds(version_id),
        }

    def task_schema_contract(self) -> dict:
        schema = ResponseTask.model_json_schema()
        record = self._ensure_contract_version(
            "task_schema", self.TASK_SCHEMA_VERSION, {"schema": schema}
        )
        return {
            "schema_version": record.version_id,
            "sha256": record.content_hash,
            "schema": schema,
            "supersedes_version_id": record.supersedes_version_id,
        }

    def rule_set_contract(self) -> dict:
        payload = {
            "outcomes": [item.value for item in RuleOutcome],
            "rules": [
                "RULE-OBJECT-CONFIRMED",
                "RULE-DEADLINES-ORDERED",
                "RULE-REQUIRED-FIELDS",
                "RULE-EVIDENCE-PACKAGE-FROZEN",
                "RULE-EVIDENCE-BOUND",
                "RULE-NO-SELF-DEPENDENCY",
                "RULE-COOPERATION-DEFINED",
            ],
        }
        record = self._ensure_contract_version(
            "rule_set", self.RULE_SET_VERSION, payload
        )
        return {
            "rule_set_version": record.version_id,
            "sha256": record.content_hash,
            **payload,
            "supersedes_version_id": record.supersedes_version_id,
        }

    def list_contract_versions(self, contract_type: str) -> list[ContractVersionRecord]:
        if contract_type not in {"task_schema", "rule_set"}:
            raise ValueError("contract_type must be task_schema or rule_set")
        if contract_type == "task_schema":
            self.task_schema_contract()
        else:
            self.rule_set_contract()
        return self.repository.list_contract_versions(contract_type)

    def _ensure_contract_version(
        self, contract_type: str, version_id: str, payload: dict
    ) -> ContractVersionRecord:
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        existing = self.repository.list_contract_versions(contract_type)
        same_version = next(
            (item for item in existing if item.version_id == version_id), None
        )
        if same_version is not None:
            if same_version.content_hash != content_hash:
                raise RuntimeError(
                    f"immutable {contract_type} version {version_id} changed content"
                )
            return same_version
        record = ContractVersionRecord(
            contract_type=contract_type,
            version_id=version_id,
            content_hash=content_hash,
            payload=payload,
            supersedes_version_id=existing[-1].version_id if existing else None,
            created_at=self._now(),
        )
        self.repository.save_contract_version(record)
        return record

    def register_document(
        self, request: DocumentImportRequest
    ) -> DocumentVersionRecord:
        return self.create_document_version(
            request.document_id,
            DocumentVersionCreateRequest.model_validate(
                request.model_dump(exclude={"document_id"})
            ),
        )

    def create_document_version(
        self,
        document_id: str,
        request: DocumentVersionCreateRequest,
    ) -> DocumentVersionRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.DUTY_OFFICER, OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "create a document version",
        )
        if request.auto_publish and self.rag_service is None:
            raise ValueError("document indexing service is unavailable")
        now = self._now()
        provisional = build_document_source("pending", request, now)
        existing_versions = self.repository.list_document_versions(document_id)
        duplicate = next(
            (
                item
                for item in existing_versions
                if item.source_hash == provisional.source_hash
            ),
            None,
        )
        if duplicate is not None:
            return self._materialize_document_version(duplicate)
        replaced = None
        if request.replaces_version_id:
            replaced = self.repository.get_document_version(request.replaces_version_id)
            if replaced is None:
                raise LookupError(
                    f"replaced document version not found: {request.replaces_version_id}"
                )
            if replaced.document_id != document_id:
                raise ValueError(
                    "a document version may only replace a version of the same document"
                )
        version_number = (
            (existing_versions[-1].version_number + 1) if existing_versions else 1
        )
        version_id = (
            f"DOCVER-{hashlib.sha256(document_id.encode('utf-8')).hexdigest()[:12]}-"
            f"{version_number}-{provisional.source_hash[:8]}"
        )
        source = provisional.model_copy(update={"version_id": version_id})
        parsed: ParsedDocument | None = None
        if request.auto_publish:
            parsed = parse_document_source(
                source, parser_version=self.DOCUMENT_PARSER_VERSION
            )
        record = DocumentVersionRecord(
            version_id=version_id,
            document_id=document_id,
            version_number=version_number,
            version_label=request.version_label,
            title=request.title,
            issuer=request.issuer,
            jurisdiction=request.jurisdiction,
            effective_at=request.effective_at,
            expires_at=request.expires_at,
            replaces_version_id=request.replaces_version_id,
            source_hash=source.source_hash,
            source_filename=source.filename,
            media_type=source.media_type,
            source_size=source.source_size,
            lifecycle_status=DocumentLifecycleStatus.DRAFT,
            is_simulated=request.is_simulated,
            created_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=now,
        )
        created_event = self._document_lifecycle_event(
            record,
            DocumentLifecycleStatus.DRAFT,
            request,
            reason="document source snapshot registered",
        )
        self.repository.save_document_version_bundle(record, source, created_event)
        if parsed is not None:
            self._save_document_parse(record, parsed, request)
            return self.publish_document_version(
                version_id,
                DocumentPublishRequest(
                    operator_id=request.operator_id,
                    operator_role=request.operator_role,
                    terminal_id=request.terminal_id,
                    note=request.note,
                    expected_version=request.expected_version,
                ),
            )
        return self._materialize_document_version(record)

    def parse_document_version(
        self, version_id: str, request: DocumentParseRequest
    ) -> DocumentVersionRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.DUTY_OFFICER, OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "parse a document version",
        )
        record = self.repository.get_document_version(version_id)
        if record is None:
            raise LookupError(f"document version not found: {version_id}")
        self._require_document_version_match(record, request.expected_version)
        current = self._materialize_document_version(record)
        if current.lifecycle_status in {
            DocumentLifecycleStatus.RETIRED,
            DocumentLifecycleStatus.SUPERSEDED,
        }:
            raise ValueError(
                f"document cannot be parsed in current status {current.lifecycle_status.value}"
            )
        source = self.repository.get_document_source(version_id)
        if source is None:
            raise ValueError("document source snapshot is unavailable")
        try:
            parsed = parse_document_source(source, parser_version=request.parser_version)
        except ValueError as exc:
            self.repository.save_document_lifecycle_event(
                self._document_lifecycle_event(
                    record,
                    DocumentLifecycleStatus.PARSE_FAILED,
                    request,
                    reason=str(exc),
                )
            )
            raise
        self._save_document_parse(record, parsed, request)
        return self.get_document_version(version_id)

    def publish_document_version(
        self, version_id: str, request: DocumentPublishRequest
    ) -> DocumentVersionRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "publish a document version",
        )
        if self.rag_service is None:
            raise ValueError("document indexing service is unavailable")
        record = self.repository.get_document_version(version_id)
        if record is None:
            raise LookupError(f"document version not found: {version_id}")
        self._require_document_version_match(record, request.expected_version)
        current = self._materialize_document_version(record)
        if current.lifecycle_status == DocumentLifecycleStatus.PUBLISHED:
            return current
        if current.lifecycle_status in {
            DocumentLifecycleStatus.RETIRED,
            DocumentLifecycleStatus.SUPERSEDED,
        }:
            raise ValueError(
                f"document cannot be published in current status {current.lifecycle_status.value}"
            )
        parses = self.repository.list_document_parses(version_id)
        if not parses:
            raise ValueError("document must be parsed before publishing")
        if record.expires_at is not None and record.expires_at <= self._now():
            raise ValueError("expired document versions cannot be published")
        other_current = [
            self._materialize_document_version(item)
            for item in self.repository.list_document_versions(record.document_id)
            if item.version_id != record.version_id
        ]
        other_published = [
            item
            for item in other_current
            if item.lifecycle_status == DocumentLifecycleStatus.PUBLISHED
        ]
        if other_published and record.replaces_version_id != other_published[-1].version_id:
            raise ValueError(
                "a new published document version must explicitly replace the current published version"
            )
        if record.replaces_version_id:
            replaced_current = next(
                (
                    item
                    for item in other_current
                    if item.version_id == record.replaces_version_id
                ),
                None,
            )
            if replaced_current is None:
                raise LookupError(
                    f"replaced document version not found: {record.replaces_version_id}"
                )
            if (
                replaced_current.lifecycle_status
                == DocumentLifecycleStatus.SUPERSEDED
                and replaced_current.superseded_by_version_id != record.version_id
            ):
                raise ValueError(
                    "replaced document version is already superseded by another version"
                )
        parsed = parses[-1]
        build = self._build_document_index(record, parsed, request)
        self.repository.save_document_lifecycle_event(
            self._document_lifecycle_event(
                record,
                DocumentLifecycleStatus.PUBLISHED,
                request,
                reason=f"index build {build.build_id} completed",
            )
        )
        if record.replaces_version_id:
            replaced = self.repository.get_document_version(record.replaces_version_id)
            if replaced is None:
                raise LookupError(
                    f"replaced document version not found: {record.replaces_version_id}"
                )
            replaced_current = self._materialize_document_version(replaced)
            if replaced_current.lifecycle_status not in {
                DocumentLifecycleStatus.RETIRED,
                DocumentLifecycleStatus.SUPERSEDED,
            }:
                self._set_rag_document_status(
                    replaced.version_id,
                    "superseded",
                    related_version_id=record.version_id,
                )
                self.repository.save_document_lifecycle_event(
                    self._document_lifecycle_event(
                        replaced,
                        DocumentLifecycleStatus.SUPERSEDED,
                        request,
                        reason=f"replaced by {record.version_id}",
                        related_version_id=record.version_id,
                    )
                )
        return self.get_document_version(version_id)

    def retire_document_version(
        self, version_id: str, request: DocumentRetireRequest
    ) -> DocumentVersionRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "retire a document version",
        )
        record = self.repository.get_document_version(version_id)
        if record is None:
            raise LookupError(f"document version not found: {version_id}")
        self._require_document_version_match(record, request.expected_version)
        current = self._materialize_document_version(record)
        if current.lifecycle_status == DocumentLifecycleStatus.RETIRED:
            return current
        self._set_rag_document_status(version_id, "retired")
        self.repository.save_document_lifecycle_event(
            self._document_lifecycle_event(
                record,
                DocumentLifecycleStatus.RETIRED,
                request,
                reason=request.reason,
            )
        )
        return self.get_document_version(version_id)

    def create_index_build(self, request: IndexBuildRequest) -> IndexBuildRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.REVIEWER, OperatorRole.ADMIN},
            "build a document index",
        )
        if self.rag_service is None:
            raise ValueError("document indexing service is unavailable")
        record = self.repository.get_document_version(request.document_version_id)
        if record is None:
            raise LookupError(
                f"document version not found: {request.document_version_id}"
            )
        current = self._materialize_document_version(record)
        if current.lifecycle_status != DocumentLifecycleStatus.PUBLISHED:
            raise ValueError("only published document versions can be reindexed")
        parses = self.repository.list_document_parses(record.version_id)
        if not parses:
            raise ValueError("document has no reproducible parse record")
        return self._build_document_index(record, parses[-1], request)

    def get_index_build(self, build_id: str) -> IndexBuildRecord:
        record = self.repository.get_index_build(build_id)
        if record is None:
            raise LookupError(f"index build not found: {build_id}")
        return record

    def _save_document_parse(
        self,
        record: DocumentVersionRecord,
        parsed: ParsedDocument,
        request: TaskActionRequest,
    ) -> DocumentParseRecord:
        existing = self.repository.list_document_parses(record.version_id)
        duplicate = next(
            (
                item
                for item in existing
                if item.parser_version == parsed.parser_version
                and item.source_hash == record.source_hash
                and item.result_hash == parsed.result_hash
            ),
            None,
        )
        if duplicate is not None:
            return duplicate
        parse_record = DocumentParseRecord(
            parse_id=self._id("DOCPARSE"),
            version_id=record.version_id,
            parser_version=parsed.parser_version,
            parse_method=parsed.parse_method,
            source_hash=record.source_hash,
            result_hash=parsed.result_hash,
            clauses=parsed.clauses,
            created_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_document_parse(parse_record)
        self.repository.save_document_lifecycle_event(
            self._document_lifecycle_event(
                record,
                DocumentLifecycleStatus.PARSED,
                request,
                reason=f"{len(parsed.clauses)} clauses parsed by {parsed.parser_version}",
            )
        )
        return parse_record

    def _build_document_index(
        self,
        record: DocumentVersionRecord,
        parsed: DocumentParseRecord,
        request: DocumentPublishRequest | IndexBuildRequest,
    ) -> IndexBuildRecord:
        now = self._now()
        config_hash = hashlib.sha256(
            f"{request.corpus_version}|{parsed.parser_version}|{request.embedding_version}".encode(
                "utf-8"
            )
        ).hexdigest()
        index_version = f"frc-index-{record.source_hash[:12]}-{config_hash[:8]}"
        build_id = self._id("INDEX")
        try:
            documents = [
                RAGDocument(
                    doc_id=f"{record.version_id}:{clause.clause_id}",
                    title=record.title,
                    corpus=CorpusType.POLICY,
                    content=clause.text,
                    metadata={
                        "document_id": record.document_id,
                        "document_version_id": record.version_id,
                        "doc_version": record.version_label,
                        "issuer": record.issuer,
                        "jurisdiction": record.jurisdiction,
                        "effective_at": record.effective_at.isoformat(),
                        "expires_at": record.expires_at.isoformat()
                        if record.expires_at
                        else None,
                        "section_number": clause.clause_number or clause.clause_id,
                        "section_title": clause.heading,
                        "page_number": clause.page_number,
                        "source_locator": clause.source_locator,
                        "table_name": clause.table_name,
                        "row_start": clause.row_start,
                        "row_end": clause.row_end,
                        "text_hash": clause.text_hash,
                        "status": "active",
                        "source_hash": record.source_hash,
                        "parser_version": parsed.parser_version,
                        "parse_hash": parsed.result_hash,
                        "index_version": index_version,
                        "corpus_version": request.corpus_version,
                        "embedding_version": request.embedding_version,
                        "is_simulated": record.is_simulated,
                        "evidence_roles": [
                            role.value
                            for role in self._document_roles_from_text(clause.text)
                        ],
                        "task_field_support": infer_task_field_support(
                            clause.text,
                            [
                                role.value
                                for role in self._document_roles_from_text(clause.text)
                            ],
                        ),
                    },
                )
                for clause in parsed.clauses
            ]
            self.rag_service.import_documents(documents)
            build = IndexBuildRecord(
                build_id=build_id,
                document_version_id=record.version_id,
                corpus_version=request.corpus_version,
                parser_version=parsed.parser_version,
                embedding_version=request.embedding_version,
                index_version=index_version,
                status=IndexBuildStatus.COMPLETED,
                clause_count=len(documents),
                source_hash=record.source_hash,
                created_by=request.operator_id,
                terminal_id=request.terminal_id,
                created_at=now,
                finished_at=self._now(),
            )
        except Exception as exc:
            build = IndexBuildRecord(
                build_id=build_id,
                document_version_id=record.version_id,
                corpus_version=request.corpus_version,
                parser_version=parsed.parser_version,
                embedding_version=request.embedding_version,
                index_version=index_version,
                status=IndexBuildStatus.FAILED,
                error=str(exc),
                source_hash=record.source_hash,
                created_by=request.operator_id,
                terminal_id=request.terminal_id,
                created_at=now,
                finished_at=self._now(),
            )
            self.repository.save_index_build(build)
            if isinstance(request, DocumentPublishRequest):
                self.repository.save_document_lifecycle_event(
                    self._document_lifecycle_event(
                        record,
                        DocumentLifecycleStatus.INDEX_FAILED,
                        request,
                        reason=f"index build {build_id} failed: {exc}",
                    )
                )
            raise ValueError(f"document index build failed: {exc}") from exc
        self.repository.save_index_build(build)
        return build

    def _materialize_document_version(
        self, record: DocumentVersionRecord
    ) -> DocumentVersionRecord:
        events = self.repository.list_document_lifecycle_events(record.version_id)
        parses = self.repository.list_document_parses(record.version_id)
        builds = self.repository.list_index_builds(record.version_id)
        successful_builds = [
            item for item in builds if item.status == IndexBuildStatus.COMPLETED
        ]
        status = events[-1].status if events else record.lifecycle_status
        if status == DocumentLifecycleStatus.ACTIVE:
            status = DocumentLifecycleStatus.PUBLISHED
        parse = parses[-1] if parses else None
        build = successful_builds[-1] if successful_builds else None
        published_event = next(
            (
                item
                for item in reversed(events)
                if item.status == DocumentLifecycleStatus.PUBLISHED
            ),
            None,
        )
        retired_event = next(
            (
                item
                for item in reversed(events)
                if item.status == DocumentLifecycleStatus.RETIRED
            ),
            None,
        )
        superseded_event = next(
            (
                item
                for item in reversed(events)
                if item.status == DocumentLifecycleStatus.SUPERSEDED
            ),
            None,
        )
        return record.model_copy(
            update={
                "lifecycle_status": status,
                "clauses": parse.clauses if parse else record.clauses,
                "parser_version": parse.parser_version if parse else record.parser_version,
                "parse_method": parse.parse_method if parse else record.parse_method,
                "parse_hash": parse.result_hash if parse else record.parse_hash,
                "parsed_at": parse.created_at if parse else record.parsed_at,
                "index_status": "indexed" if build else record.index_status,
                "index_version": build.index_version if build else record.index_version,
                "index_build_id": build.build_id if build else record.index_build_id,
                "published_at": published_event.created_at
                if published_event
                else record.published_at,
                "retired_at": retired_event.created_at
                if retired_event
                else record.retired_at,
                "superseded_by_version_id": superseded_event.related_version_id
                if superseded_event
                else record.superseded_by_version_id,
            }
        )

    def _document_lifecycle_event(
        self,
        record: DocumentVersionRecord,
        status: DocumentLifecycleStatus,
        request: TaskActionRequest,
        *,
        reason: str,
        related_version_id: str | None = None,
    ) -> DocumentLifecycleEvent:
        return DocumentLifecycleEvent(
            lifecycle_event_id=self._id("DOCLIFE"),
            version_id=record.version_id,
            status=status,
            reason=reason,
            related_version_id=related_version_id,
            actor_id=request.operator_id,
            actor_role=request.operator_role,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )

    @staticmethod
    def _require_document_version_match(
        record: DocumentVersionRecord, expected_version: int | None
    ) -> None:
        if expected_version is not None and expected_version != record.version_number:
            raise ValueError(
                f"version conflict: document expected {expected_version}, actual {record.version_number}"
            )

    def _set_rag_document_status(
        self,
        version_id: str,
        status: str,
        *,
        related_version_id: str | None = None,
    ) -> None:
        if self.rag_service is None:
            raise ValueError("document indexing service is unavailable")
        updated = []
        for document in self.rag_service.list_documents():
            if document.metadata.get("document_version_id") != version_id:
                continue
            metadata = dict(document.metadata)
            metadata["status"] = status
            if related_version_id:
                metadata["superseded_by"] = related_version_id
            updated.append(document.model_copy(update={"metadata": metadata}))
        if updated:
            self.rag_service.import_documents(updated)

    @staticmethod
    def _document_roles_from_text(text: str) -> list[EvidenceRole]:
        mapping = {
            EvidenceRole.CONDITION: ("当", "达到", "预警", "出现"),
            EvidenceRole.OBJECT: ("下穿", "对象", "道路", "学校", "社区"),
            EvidenceRole.RESPONSIBILITY: ("责任", "部门", "防办", "住建", "交警"),
            EvidenceRole.PROCEDURE: ("应", "需", "核查", "组织", "启动", "封控"),
            EvidenceRole.EXCEPTION: ("若", "如", "受阻", "超时", "不足"),
            EvidenceRole.ATTRIBUTION: ("条", "规定", "依据"),
        }
        return [
            role
            for role, keywords in mapping.items()
            if any(keyword in text for keyword in keywords)
        ]
