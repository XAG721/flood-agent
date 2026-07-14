from __future__ import annotations

import base64
import hashlib
import io
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from flood_system.models import CorpusType, RAGDocument
from flood_system.http.response_router import create_response_router
from flood_system.identity import TrustedIdentityVerifier
from flood_system.response_workflow.document_governance import (
    build_document_source,
    parse_document_source,
)
from flood_system.response_workflow.models import (
    DocumentLifecycleStatus,
    DocumentParseRequest,
    DocumentPublishRequest,
    DocumentRetireRequest,
    DocumentVersionCreateRequest,
    IndexBuildStatus,
    IngestionFileEnvelope,
    OperatorRole,
)
from flood_system.system import FloodWarningSystem


IDENTITY_SECRET = "document-governance-test-secret-with-at-least-32-bytes"


def _headers(method: str, path: str, nonce: str) -> dict[str, str]:
    headers = TrustedIdentityVerifier.build_headers(
        secret=IDENTITY_SECRET,
        operator_id="admin-1",
        operator_role=OperatorRole.ADMIN,
        assurance_level="aal2",
        terminal_id="document-terminal",
        nonce=nonce,
        method=method,
        path=path,
    )
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        headers["Idempotency-Key"] = nonce
    headers["X-Correlation-ID"] = nonce
    return headers


def _envelope(filename: str, media_type: str, payload: bytes) -> IngestionFileEnvelope:
    return IngestionFileEnvelope(
        filename=filename,
        media_type=media_type,
        content_base64=base64.b64encode(payload).decode("ascii"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _request(
    *,
    file: IngestionFileEnvelope | None = None,
    content: str | None = None,
    ocr_text: str | None = None,
    ocr_engine_version: str | None = None,
    version_label: str = "2026-A",
    replaces_version_id: str | None = None,
    auto_publish: bool = False,
) -> DocumentVersionCreateRequest:
    return DocumentVersionCreateRequest(
        title="区级防汛响应规程",
        version_label=version_label,
        issuer="模拟区防办",
        jurisdiction="district-simulation",
        effective_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        file=file,
        content=content,
        ocr_text=ocr_text,
        ocr_engine_version=ocr_engine_version,
        replaces_version_id=replaces_version_id,
        auto_publish=auto_publish,
        operator_id="admin-1",
        operator_role=OperatorRole.ADMIN,
        terminal_id="document-terminal",
    )


def _docx_payload() -> bytes:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>第一条 橙色预警时核查下穿通道。</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>责任部门</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>住建局</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
  </w:body>
</w:document>"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def _xlsx_payload() -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "处置表"
    worksheet.append(["对象", "动作", "责任部门"])
    worksheet.append(["下穿通道", "封控", "住建局"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def _pdf_payload() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = StreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Flood response clause one.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.parametrize(
    ("create_request", "expected_method", "expected_content_type"),
    [
        (
            _request(content="第一条 橙色预警时核查下穿通道。\n\n第二条 住建局负责封控。"),
            "plain_text",
            "paragraph",
        ),
        (
            _request(
                file=_envelope(
                    "plan.csv",
                    "text/csv",
                    "对象,动作\n下穿通道,封控".encode("utf-8"),
                )
            ),
            "csv_table",
            "table_row",
        ),
        (
            _request(
                file=_envelope(
                    "plan.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    _xlsx_payload(),
                )
            ),
            "xlsx_table",
            "table_row",
        ),
        (
            _request(
                file=_envelope(
                    "plan.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    _docx_payload(),
                )
            ),
            "docx_ooxml",
            "paragraph",
        ),
        (
            _request(
                file=_envelope("plan.pdf", "application/pdf", _pdf_payload()),
            ),
            "pdf_text",
            "paragraph",
        ),
        (
            _request(
                file=_envelope("scan.png", "image/png", PNG_1X1),
                ocr_text="第一条 扫描规程要求封控下穿通道。",
                ocr_engine_version="verified-ocr-v1",
            ),
            "verified_ocr",
            "ocr_paragraph",
        ),
        (
            _request(
                file=_envelope("scan.pdf", "application/pdf", b"%PDF-invalid-scan"),
                ocr_text="第一条 扫描 PDF 要求住建局核查下穿通道。",
                ocr_engine_version="verified-ocr-v1",
            ),
            "pdf_text_or_ocr",
            "ocr_paragraph",
        ),
    ],
)
def test_supported_document_formats_produce_traceable_clauses(
    create_request: DocumentVersionCreateRequest,
    expected_method: str,
    expected_content_type: str,
):
    source = build_document_source(
        "DOCVER-FORMAT-1", create_request, datetime.now(timezone.utc)
    )
    parsed = parse_document_source(source)

    assert parsed.parse_method == expected_method
    assert parsed.clauses
    assert parsed.clauses[0].content_type == expected_content_type
    assert parsed.clauses[0].source_locator
    assert len(parsed.clauses[0].text_hash) == 64
    assert len(parsed.result_hash) == 64


def test_document_upload_rejects_path_hash_mismatch_and_unverified_scans():
    payload = b"safe text payload"
    with pytest.raises(ValueError, match="must not contain a path"):
        build_document_source(
            "DOCVER-UNSAFE",
            _request(file=_envelope("../plan.txt", "text/plain", payload)),
            datetime.now(timezone.utc),
        )

    bad_hash = _envelope("plan.txt", "text/plain", payload).model_copy(
        update={"sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="SHA-256"):
        build_document_source(
            "DOCVER-BAD-HASH",
            _request(file=bad_hash),
            datetime.now(timezone.utc),
        )

    scan_source = build_document_source(
        "DOCVER-SCAN",
        _request(file=_envelope("scan.png", "image/png", PNG_1X1)),
        datetime.now(timezone.utc),
    )
    with pytest.raises(ValueError, match="verified ocr_text"):
        parse_document_source(scan_source)


def test_document_lifecycle_is_append_only_and_retrieval_uses_only_current_versions(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "document-lifecycle.db")
    workflow = system.response_workflow
    draft = workflow.create_document_version(
        "DISTRICT-PLAN-001",
        _request(content="第一条 橙色预警时住建局应核查下穿通道。"),
    )
    assert draft.lifecycle_status == DocumentLifecycleStatus.DRAFT
    assert draft.clauses == []

    parsed = workflow.parse_document_version(
        draft.version_id,
        DocumentParseRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="document-terminal",
            expected_version=1,
        ),
    )
    assert parsed.lifecycle_status == DocumentLifecycleStatus.PARSED
    assert parsed.parser_version == "document-parser-v2"
    assert parsed.clauses[0].source_locator

    published = workflow.publish_document_version(
        draft.version_id,
        DocumentPublishRequest(
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="document-terminal",
            expected_version=1,
        ),
    )
    assert published.lifecycle_status == DocumentLifecycleStatus.PUBLISHED
    assert published.index_status == "indexed"
    assert published.index_build_id
    history = workflow.document_version_history(draft.version_id)
    assert history["source"]["source_hash"] == published.source_hash
    assert "content_base64" not in history["source"]
    assert [item.status for item in history["lifecycle"]] == [
        DocumentLifecycleStatus.DRAFT,
        DocumentLifecycleStatus.PARSED,
        DocumentLifecycleStatus.PUBLISHED,
    ]
    assert history["index_builds"][-1].status == IndexBuildStatus.COMPLETED

    replacement = workflow.create_document_version(
        "DISTRICT-PLAN-001",
        _request(
            content="第一条 红色预警时住建局应立即封控下穿通道。",
            version_label="2026-B",
            replaces_version_id=draft.version_id,
            auto_publish=True,
        ),
    )
    assert replacement.lifecycle_status == DocumentLifecycleStatus.PUBLISHED
    assert (
        workflow.get_document_version(draft.version_id).lifecycle_status
        == DocumentLifecycleStatus.SUPERSEDED
    )
    current = workflow._query_task_evidence("红色预警 下穿通道 封控")
    assert current
    assert all(
        item.metadata.get("document_version_id") != draft.version_id
        for item in current
    )

    retired = workflow.retire_document_version(
        replacement.version_id,
        DocumentRetireRequest(
            reason="模拟版本到期退役",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="document-terminal",
            expected_version=2,
        ),
    )
    assert retired.lifecycle_status == DocumentLifecycleStatus.RETIRED
    assert all(
        item.metadata.get("document_version_id") != replacement.version_id
        for item in workflow._query_task_evidence("红色预警 下穿通道 封控")
    )

    with system.repository._connect() as conn:
        for table in (
            "response_document_versions",
            "response_document_sources",
            "response_document_parses",
            "response_document_lifecycle_events",
            "response_index_builds",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute(f"UPDATE {table} SET payload = payload WHERE rowid = 1")


def test_task_schema_and_rule_set_contracts_have_immutable_version_history(tmp_path):
    system = FloodWarningSystem(tmp_path / "contracts.db")
    workflow = system.response_workflow
    task_schema = workflow.task_schema_contract()
    rule_set = workflow.rule_set_contract()

    assert task_schema["schema_version"] == workflow.TASK_SCHEMA_VERSION
    assert (
        task_schema["schema"]["properties"]["task_schema_version"]["default"]
        == workflow.TASK_SCHEMA_VERSION
    )
    assert rule_set["rule_set_version"] == workflow.RULE_SET_VERSION
    assert len(task_schema["sha256"]) == len(rule_set["sha256"]) == 64
    assert len(workflow.list_contract_versions("task_schema")) == 1
    assert len(workflow.list_contract_versions("rule_set")) == 1

    with system.repository._connect() as conn:
        payload = conn.execute(
            "SELECT payload FROM response_contract_versions WHERE contract_type = 'task_schema'"
        ).fetchone()[0]
        assert "task_schema_version" not in payload
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE response_contract_versions SET payload = payload "
                "WHERE contract_type = 'task_schema'"
            )


def test_initial_index_failure_is_audited_and_can_be_retried_without_source_loss(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "index-retry.db")
    workflow = system.response_workflow
    draft = workflow.create_document_version(
        "DISTRICT-PLAN-INDEX-RETRY",
        _request(content="第一条 橙色预警时住建局应核查下穿通道。"),
    )
    workflow.parse_document_version(
        draft.version_id,
        DocumentParseRequest(
            operator_id="admin-1",
            operator_role=OperatorRole.ADMIN,
            terminal_id="document-terminal",
        ),
    )
    healthy_rag = workflow.rag_service

    class FailingRag:
        @staticmethod
        def list_documents():
            return []

        @staticmethod
        def import_documents(_documents):
            raise RuntimeError("simulated index dependency outage")

    workflow.rag_service = FailingRag()
    publish_request = DocumentPublishRequest(
        operator_id="admin-1",
        operator_role=OperatorRole.ADMIN,
        terminal_id="document-terminal",
    )
    with pytest.raises(ValueError, match="index build failed"):
        workflow.publish_document_version(draft.version_id, publish_request)
    failed = workflow.get_document_version(draft.version_id)
    assert failed.lifecycle_status == DocumentLifecycleStatus.INDEX_FAILED
    assert workflow.document_version_history(draft.version_id)["index_builds"][-1].status == (
        IndexBuildStatus.FAILED
    )

    workflow.rag_service = healthy_rag
    recovered = workflow.publish_document_version(draft.version_id, publish_request)
    assert recovered.lifecycle_status == DocumentLifecycleStatus.PUBLISHED
    assert recovered.index_status == "indexed"


def test_future_expired_and_retired_documents_are_excluded_from_current_retrieval():
    now = datetime.now(timezone.utc)

    def document(doc_id: str, status: str, effective_at, expires_at):
        return RAGDocument(
            doc_id=doc_id,
            title=doc_id,
            corpus=CorpusType.POLICY,
            content="下穿通道封控",
            metadata={
                "status": status,
                "effective_at": effective_at,
                "expires_at": expires_at,
            },
        )

    documents = [
        document(
            "current",
            "published",
            (now - timedelta(days=1)).isoformat(),
            (now + timedelta(days=1)).isoformat(),
        ),
        document("future", "published", (now + timedelta(days=1)).isoformat(), None),
        document("expired", "published", None, (now - timedelta(seconds=1)).isoformat()),
        document("retired", "retired", None, None),
        document("bad-date", "published", "not-a-date", None),
    ]

    current = system_filter(documents)
    assert [item.doc_id for item in current] == ["current"]


def test_document_governance_core_api_exposes_create_parse_publish_history_and_retire(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "document-api.db")
    system.response_identity = TrustedIdentityVerifier(system.repository, IDENTITY_SECRET)
    app = FastAPI()
    app.include_router(create_response_router(lambda: system))
    client = TestClient(app)
    create_path = "/response/documents/DISTRICT-PLAN-API/versions"
    payload = {
        "title": "区级下穿通道规程",
        "version_label": "2026-API",
        "issuer": "模拟区防办",
        "jurisdiction": "district-simulation",
        "effective_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        "content": "第一条 橙色预警时住建局应核查下穿通道。",
        "auto_publish": False,
        "operator_id": "admin-1",
        "operator_role": "admin",
        "terminal_id": "document-terminal",
    }
    created = client.post(
        create_path,
        headers=_headers("POST", create_path, "document-create-nonce-001"),
        json=payload,
    )
    assert created.status_code == 200, created.text
    version_id = created.json()["version_id"]
    assert created.json()["lifecycle_status"] == "draft"

    parse_path = f"/response/document-versions/{version_id}/parse"
    parsed = client.post(
        parse_path,
        headers=_headers("POST", parse_path, "document-parse-nonce-001"),
        json={
            "parser_version": "document-parser-v2",
            "expected_version": 1,
            "operator_id": "admin-1",
            "operator_role": "admin",
            "terminal_id": "document-terminal",
        },
    )
    assert parsed.status_code == 200
    assert parsed.json()["lifecycle_status"] == "parsed"

    publish_path = f"/response/document-versions/{version_id}/publish"
    published = client.post(
        publish_path,
        headers=_headers("POST", publish_path, "document-publish-nonce-001"),
        json={
            "expected_version": 1,
            "operator_id": "admin-1",
            "operator_role": "admin",
            "terminal_id": "document-terminal",
        },
    )
    assert published.status_code == 200
    assert published.json()["lifecycle_status"] == "published"
    assert published.json()["clauses"][0]["source_locator"]

    history_path = f"/response/document-versions/{version_id}/history"
    history = client.get(
        history_path,
        headers=_headers("GET", history_path, "document-history-nonce-001"),
    )
    assert history.status_code == 200
    assert len(history.json()["lifecycle"]) == 3
    assert "content_base64" not in history.json()["source"]

    retire_path = f"/response/document-versions/{version_id}/retire"
    retired = client.post(
        retire_path,
        headers=_headers("POST", retire_path, "document-retire-nonce-001"),
        json={
            "reason": "模拟规程到期退役",
            "expected_version": 1,
            "operator_id": "admin-1",
            "operator_role": "admin",
            "terminal_id": "document-terminal",
        },
    )
    assert retired.status_code == 200
    assert retired.json()["lifecycle_status"] == "retired"


def system_filter(documents: list[RAGDocument]) -> list[RAGDocument]:
    from flood_system.response_workflow.service import ResponseWorkflowService

    return ResponseWorkflowService._filter_current_policy_documents(documents)
