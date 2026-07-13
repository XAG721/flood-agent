from __future__ import annotations

import base64
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook

from flood_system.http.response_router import create_response_router
from flood_system.identity import TrustedIdentityVerifier
from flood_system.response_workflow.models import (
    AlertInput,
    CandidateDiscoveryRequest,
    EventCreateRequest,
    GeoPolygon,
    IngestionFileEnvelope,
    KeyRotationRequest,
    ObjectVerificationStatus,
    OperatorRole,
    RiskObjectInput,
    RiskObjectRegistryBatchImportRequest,
    RiskObjectRegistryFileImportRequest,
    RiskObjectVerificationRequest,
    SensitiveContact,
    TaskDraftGenerationRequest,
)
from flood_system.response_workflow.risk_object_ingestion import (
    RiskObjectIngestionError,
)
from flood_system.system import FloodWarningSystem


IDENTITY_SECRET = "pytest-risk-registry-identity-secret-with-at-least-32-bytes"


def _risk_object(
    *,
    object_id: str = "SCHOOL-REG-001",
    risk_score: float = 82,
    longitude: float | None = 108.958,
    latitude: float | None = 34.244,
) -> RiskObjectInput:
    return RiskObjectInput(
        object_id=object_id,
        name="文艺路重点学校",
        object_type="学校",
        location="碑林区文艺路",
        longitude=longitude,
        latitude=latitude,
        responsible_organization="区教育局",
        responsible_role="学校防汛负责人",
        trigger_reasons=["纳入区级风险对象台账"],
        source_refs=["district-registry:school-2026"],
        vulnerability="低龄学生集中，转移组织时间较长",
        risk_score=risk_score,
        system_explanation="主数据登记风险分，候选发现后仍需人工核验",
        sensitive_contacts=[
            SensitiveContact(
                name="张老师",
                role="防汛联系人",
                phone="13800000000",
            )
        ],
        special_population_notes="低龄学生",
        source_type="governed_registry",
        is_simulated=False,
    )


def _batch_request(
    *objects: RiskObjectInput,
    source_version: str = "district-registry-2026.1",
) -> RiskObjectRegistryBatchImportRequest:
    return RiskObjectRegistryBatchImportRequest(
        area_id="district-registry",
        source_version=source_version,
        source_filename="district-risk-objects.json",
        objects=list(objects),
        operator_id="registry-reviewer",
        operator_role=OperatorRole.REVIEWER,
        terminal_id="registry-console",
    )


def _event(system: FloodWarningSystem):
    now = datetime.now(timezone.utc)
    return system.response_workflow.create_event(
        EventCreateRequest(
            title="风险对象主数据联动验证",
            area_id="district-registry",
            alert=AlertInput(
                alert_id="ALERT-REGISTRY-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                valid_until=now + timedelta(hours=3),
                affected_area="文艺路片区",
                affected_geometry=GeoPolygon(
                    coordinates=[
                        (108.956, 34.242),
                        (108.960, 34.242),
                        (108.960, 34.246),
                        (108.956, 34.246),
                        (108.956, 34.242),
                    ]
                ),
                raw_content="橙色暴雨预警覆盖文艺路片区",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="duty-console",
        )
    )


def _file_envelope(
    filename: str,
    media_type: str,
    payload: bytes,
    *,
    sha256: str | None = None,
) -> IngestionFileEnvelope:
    return IngestionFileEnvelope(
        filename=filename,
        media_type=media_type,
        content_base64=base64.b64encode(payload).decode("ascii"),
        sha256=sha256 or hashlib.sha256(payload).hexdigest(),
    )


def _file_request(
    envelope: IngestionFileEnvelope,
    *,
    source_version: str,
) -> RiskObjectRegistryFileImportRequest:
    return RiskObjectRegistryFileImportRequest(
        area_id="district-registry",
        source_version=source_version,
        file=envelope,
        operator_id="registry-reviewer",
        operator_role=OperatorRole.REVIEWER,
        terminal_id="registry-console",
    )


def _signed_headers(
    *,
    method: str,
    path: str,
    nonce: str,
    role: OperatorRole,
    assurance: str = "aal2",
    operator_id: str = "registry-reviewer",
    terminal_id: str = "registry-console",
    idempotency_key: str | None = None,
) -> dict[str, str]:
    headers = TrustedIdentityVerifier.build_headers(
        secret=IDENTITY_SECRET,
        operator_id=operator_id,
        operator_role=role,
        assurance_level=assurance,
        terminal_id=terminal_id,
        nonce=nonce,
        method=method,
        path=path,
    )
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        headers["Idempotency-Key"] = idempotency_key or nonce
    headers["X-Correlation-ID"] = nonce
    return headers


def test_registry_drives_candidate_discovery_and_changes_mark_dependents_stale(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "risk-registry.db")
    workflow = system.response_workflow

    imported = workflow.import_risk_object_registry(_batch_request(_risk_object()))
    assert imported.created_count == 1
    assert imported.updated_count == 0
    assert imported.quarantined_count == 0
    assert imported.changed_object_ids == ["SCHOOL-REG-001"]

    unchanged = workflow.import_risk_object_registry(_batch_request(_risk_object()))
    assert unchanged.unchanged_count == 1
    assert unchanged.changed_object_ids == []
    assert (
        len(
            workflow.list_risk_object_registry_versions(
                area_id="district-registry",
                object_id="SCHOOL-REG-001",
                operator_role=OperatorRole.REVIEWER,
            )
        )
        == 1
    )

    dashboard = _event(system)
    discovery = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            entity_types=["学校"],
            min_risk_score=40,
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="duty-console",
        ),
    )
    assert discovery.algorithm_version == "candidate-registry-rules-v1"
    assert discovery.data_version.startswith("registry:")
    assert [item.object_id for item in discovery.candidates] == ["SCHOOL-REG-001"]
    candidate = discovery.candidates[0]
    assert candidate.longitude == 108.958
    assert candidate.latitude == 34.244
    assert candidate.association_mode == "gis_point_in_polygon"
    assert any(ref.startswith("risk_registry:") for ref in candidate.source_refs)

    workflow.verify_risk_object(
        dashboard.event.event_id,
        candidate.object_id,
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="属地复核通过",
            operator_id="registry-reviewer",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="registry-console",
        ),
    )
    changed = workflow.import_risk_object_registry(
        _batch_request(
            _risk_object(risk_score=94),
            source_version="district-registry-2026.2",
        )
    )
    assert changed.updated_count == 1
    assert changed.stale_candidate_run_count == 1
    assert changed.affected_event_ids == [dashboard.event.event_id]

    run = workflow.list_candidate_runs(dashboard.event.event_id)[0]
    stale_object = workflow.repository.get_event_risk_object(
        dashboard.event.event_id,
        candidate.object_id,
    )
    assert run.status == "stale"
    assert run.stale_at is not None
    assert "registry import" in run.stale_reason
    assert stale_object is not None and stale_object.stale is True
    with pytest.raises(ValueError, match="stale risk object"):
        workflow.generate_task_draft(
            dashboard.event.event_id,
            candidate.object_id,
            TaskDraftGenerationRequest(
                operator_id="duty-1",
                operator_role=OperatorRole.DUTY_OFFICER,
                terminal_id="duty-console",
            ),
        )

    with system.repository._connect() as connection:
        payloads = {
            table: connection.execute(
                f"SELECT payload FROM {table} LIMIT 1"
            ).fetchone()[0]
            for table in (
                "response_risk_object_registry",
                "response_risk_object_registry_versions",
                "response_risk_object_imports",
            )
        }
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE response_risk_object_registry_versions SET payload = payload"
            )
    assert all(
        json.loads(payload)["protected"] is True for payload in payloads.values()
    )
    assert all("张老师" not in payload for payload in payloads.values())


def test_csv_import_quarantines_bad_rows_and_rejects_unsafe_envelopes(tmp_path):
    system = FloodWarningSystem(tmp_path / "risk-registry-csv.db")
    workflow = system.response_workflow
    csv_payload = (
        "object_id,name,object_type,responsible_organization,responsible_role,"
        "risk_score,longitude,latitude\n"
        "TUNNEL-CSV-001,长安路下穿通道,下穿通道,区住建局,排水值班负责人,88,"
        "108.95,34.24\n"
        "TUNNEL-CSV-002,缺少风险分对象,下穿通道,区住建局,排水值班负责人,,"
        "108.95,34.24\n"
    ).encode("utf-8")
    result = workflow.import_risk_object_registry_file(
        _file_request(
            _file_envelope("risk-objects.csv", "text/csv", csv_payload),
            source_version="csv-2026.1",
        )
    )
    assert result.imported_count == 1
    assert result.created_count == 1
    assert result.quarantined_count == 1
    assert result.quarantine[0].source_row == 3
    assert result.quarantine[0].reason_code == "INVALID_RISK_OBJECT"

    with pytest.raises(RiskObjectIngestionError, match="plain filename"):
        workflow.import_risk_object_registry_file(
            _file_request(
                _file_envelope("../risk-objects.csv", "text/csv", csv_payload),
                source_version="csv-unsafe-name",
            )
        )
    with pytest.raises(RiskObjectIngestionError, match="does not match"):
        workflow.import_risk_object_registry_file(
            _file_request(
                _file_envelope(
                    "risk-objects.csv",
                    "application/json",
                    csv_payload,
                ),
                source_version="csv-wrong-mime",
            )
        )
    with pytest.raises(RiskObjectIngestionError, match="SHA-256"):
        workflow.import_risk_object_registry_file(
            _file_request(
                _file_envelope(
                    "risk-objects.csv",
                    "text/csv",
                    csv_payload,
                    sha256="0" * 64,
                ),
                source_version="csv-wrong-hash",
            )
        )

    malformed_csv = b'object_id,name\n"unterminated,value\n'
    with pytest.raises(RiskObjectIngestionError, match="CSV input is malformed"):
        workflow.import_risk_object_registry_file(
            _file_request(
                _file_envelope("malformed.csv", "text/csv", malformed_csv),
                source_version="csv-malformed",
            )
        )

    deeply_nested_json = ("[" * 70 + "{}" + "]" * 70).encode("utf-8")
    with pytest.raises(RiskObjectIngestionError, match="nesting depth"):
        workflow.import_risk_object_registry_file(
            _file_request(
                _file_envelope(
                    "deep.json",
                    "application/json",
                    deeply_nested_json,
                ),
                source_version="json-too-deep",
            )
        )


def test_registry_versions_only_changed_objects_and_rejects_duplicate_chains(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "risk-registry-content-version.db")
    workflow = system.response_workflow
    first = _risk_object(object_id="SCHOOL-CONTENT-001")
    second = _risk_object(object_id="SCHOOL-CONTENT-002", risk_score=76)
    workflow.import_risk_object_registry(_batch_request(first, second))

    changed = workflow.import_risk_object_registry(
        _batch_request(
            first.model_copy(update={"risk_score": 93}),
            second,
        )
    )
    assert changed.updated_count == 1
    assert changed.unchanged_count == 1

    stable_source = workflow.import_risk_object_registry(
        _batch_request(
            first.model_copy(update={"risk_score": 93}),
            second,
        )
    )
    assert stable_source.updated_count == 0
    assert stable_source.unchanged_count == 2

    duplicate = _risk_object(object_id="SCHOOL-DUPLICATE-001").model_copy(
        update={
            "duplicate_of": "SCHOOL-CONTENT-001",
            "canonical_object_id": "SCHOOL-CONTENT-001",
        }
    )
    duplicate_chain = _risk_object(object_id="SCHOOL-DUPLICATE-002").model_copy(
        update={
            "duplicate_of": "SCHOOL-DUPLICATE-001",
            "canonical_object_id": "SCHOOL-DUPLICATE-001",
        }
    )
    duplicate_result = workflow.import_risk_object_registry(
        _batch_request(
            duplicate,
            duplicate_chain,
            source_version="district-registry-duplicates",
        )
    )
    assert duplicate_result.created_count == 1
    assert duplicate_result.quarantined_count == 1
    assert duplicate_result.quarantine[0].reason_code == "INVALID_CANONICAL_OBJECT"


def test_xlsx_and_geojson_imports_are_bounded_and_geometry_aware(tmp_path):
    system = FloodWarningSystem(tmp_path / "risk-registry-files.db")
    workflow = system.response_workflow

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(
        [
            "object_id",
            "name",
            "object_type",
            "responsible_organization",
            "responsible_role",
            "risk_score",
            "longitude",
            "latitude",
        ]
    )
    worksheet.append(
        [
            "HOSPITAL-XLSX-001",
            "区人民医院",
            "医院",
            "区卫生健康局",
            "医院应急负责人",
            91,
            108.96,
            34.25,
        ]
    )
    xlsx_buffer = io.BytesIO()
    workbook.save(xlsx_buffer)
    workbook.close()
    result = workflow.import_risk_object_registry_file(
        _file_request(
            _file_envelope(
                "risk-objects.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                xlsx_buffer.getvalue(),
            ),
            source_version="xlsx-2026.1",
        )
    )
    assert result.source_format == "xlsx"
    assert result.created_count == 1

    formula_workbook = Workbook()
    formula_sheet = formula_workbook.active
    formula_sheet.append(
        [
            "object_id",
            "name",
            "object_type",
            "responsible_organization",
            "responsible_role",
            "risk_score",
        ]
    )
    formula_sheet.append(
        [
            "FORMULA-XLSX-001",
            '=CONCAT("危险","对象")',
            "学校",
            "区教育局",
            "学校防汛负责人",
            80,
        ]
    )
    formula_buffer = io.BytesIO()
    formula_workbook.save(formula_buffer)
    formula_workbook.close()
    with pytest.raises(RiskObjectIngestionError, match="formulas are forbidden"):
        workflow.import_risk_object_registry_file(
            _file_request(
                _file_envelope(
                    "formula.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    formula_buffer.getvalue(),
                ),
                source_version="xlsx-formula",
            )
        )

    geojson_payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [108.97, 34.26]},
                    "properties": {
                        "object_id": "SHELTER-GEO-001",
                        "name": "区体育馆安置点",
                        "object_type": "安置点",
                        "responsible_organization": "区应急管理局",
                        "responsible_role": "安置点负责人",
                        "risk_score": 64,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[108.0, 34.0], [109.0, 34.0], [108.0, 34.0]]],
                    },
                    "properties": {
                        "object_id": "INVALID-GEO-001",
                        "name": "错误几何对象",
                        "object_type": "安置点",
                        "responsible_organization": "区应急管理局",
                        "responsible_role": "安置点负责人",
                        "risk_score": 60,
                    },
                },
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    geojson_result = workflow.import_risk_object_registry_file(
        _file_request(
            _file_envelope(
                "risk-objects.geojson",
                "application/geo+json",
                geojson_payload,
            ),
            source_version="geojson-2026.1",
        )
    )
    assert geojson_result.source_format == "geojson"
    assert geojson_result.created_count == 1
    assert geojson_result.quarantined_count == 1
    stored = workflow.repository.get_risk_object_registry_record(
        "district-registry",
        "SHELTER-GEO-001",
    )
    assert stored is not None
    assert (stored.longitude, stored.latitude) == (108.97, 34.26)


def test_registry_api_requires_aal2_is_idempotent_and_redacts_by_role(tmp_path):
    system = FloodWarningSystem(tmp_path / "risk-registry-api.db")
    system.response_identity = TrustedIdentityVerifier(
        system.repository,
        IDENTITY_SECRET,
    )
    app = FastAPI()
    app.include_router(create_response_router(lambda: system))
    client = TestClient(app)
    path = "/response/risk-objects/imports"
    payload = _batch_request(_risk_object()).model_dump(mode="json")

    denied = client.post(
        path,
        headers=_signed_headers(
            method="POST",
            path=path,
            nonce="registry-import-aal1",
            role=OperatorRole.REVIEWER,
            assurance="aal1",
        ),
        json=payload,
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "ASSURANCE_REQUIRED"

    first = client.post(
        path,
        headers=_signed_headers(
            method="POST",
            path=path,
            nonce="registry-import-first",
            role=OperatorRole.REVIEWER,
            idempotency_key="registry-import-stable-key",
        ),
        json=payload,
    )
    replay = client.post(
        path,
        headers=_signed_headers(
            method="POST",
            path=path,
            nonce="registry-import-replay",
            role=OperatorRole.REVIEWER,
            idempotency_key="registry-import-stable-key",
        ),
        json=payload,
    )
    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.headers["X-Idempotent-Replay"] == "true"
    assert first.content == replay.content
    assert (
        len(system.repository.list_risk_object_imports(area_id="district-registry"))
        == 1
    )

    list_path = "/response/risk-objects"
    auditor_view = client.get(
        list_path,
        params={"area_id": "district-registry"},
        headers=_signed_headers(
            method="GET",
            path=list_path,
            nonce="registry-list-auditor",
            role=OperatorRole.AUDITOR,
            operator_id="registry-auditor",
            terminal_id="audit-console",
        ),
    )
    assert auditor_view.status_code == 200
    assert auditor_view.json()[0]["longitude"] is None
    assert auditor_view.json()[0]["location"].startswith("[精确位置")
    assert auditor_view.json()[0]["sensitive_contacts"] == []

    versions_path = "/response/risk-objects/versions"
    forbidden_history = client.get(
        versions_path,
        params={"area_id": "district-registry"},
        headers=_signed_headers(
            method="GET",
            path=versions_path,
            nonce="registry-versions-auditor",
            role=OperatorRole.AUDITOR,
            operator_id="registry-auditor",
            terminal_id="audit-console",
        ),
    )
    assert forbidden_history.status_code == 403


def test_registry_history_survives_encryption_key_rotation(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FLOOD_BACKUP_MANIFEST_SECRET",
        "pytest-risk-registry-rotation-manifest-secret-32-bytes",
    )
    monkeypatch.setenv(
        "FLOOD_DATA_ENCRYPTION_KEY_NEXT",
        Fernet.generate_key().decode("ascii"),
    )
    system = FloodWarningSystem(tmp_path / "risk-registry-rotation.db")
    system.response_workflow.import_risk_object_registry(_batch_request(_risk_object()))
    rotation = system.response_workflow.rotate_data_encryption_key(
        # Key rotation is intentionally exercised after immutable history exists.
        KeyRotationRequest(
            label="risk-registry-history",
            operator_id="rotation-admin",
            operator_role=OperatorRole.ADMIN,
            terminal_id="rotation-console",
        )
    )
    assert rotation.records_reencrypted > 0
    assert (
        system.repository.get_risk_object_registry_record(
            "district-registry", "SCHOOL-REG-001"
        )
        is not None
    )
    with system.repository._connect() as connection:
        for table in (
            "response_risk_object_registry",
            "response_risk_object_registry_versions",
            "response_risk_object_imports",
        ):
            key_ids = {
                row[0]
                for row in connection.execute(
                    f"SELECT DISTINCT json_extract(payload, '$.key_id') FROM {table}"
                ).fetchall()
            }
            assert key_ids == {rotation.new_key_id}
