from __future__ import annotations

import shutil
from datetime import datetime, timezone

import pytest

from flood_system.postgis_migration import build_projection_records
from flood_system.response_workflow.models import (
    AlertInput,
    CandidateDiscoveryRequest,
    CandidateObjectListFreezeRequest,
    DocumentImportRequest,
    EventCreateRequest,
    ObjectVerificationStatus,
    OperatorRole,
    RiskObjectInput,
    RiskObjectRegistryBatchImportRequest,
    RiskObjectVerificationRequest,
)
from flood_system.system import FloodWarningSystem
from flood_system.security import DataProtectionError
from scripts.seed_response_migration_fixture import seed_fixture


def test_simulated_response_projection_preserves_hashes_and_postgis_polygon(
    tmp_path,
) -> None:
    db_path = tmp_path / "postgis-source.db"
    seed_fixture(db_path)

    records, quarantined = build_projection_records(db_path)

    assert quarantined == []
    assert {record["record_type"] for record in records} == {
        "response_event",
        "alert_snapshot",
        "risk_object",
        "risk_object_version",
        "task",
        "task_version",
        "approval",
        "outbox",
        "dispatch_callback",
        "feedback",
        "evidence_package",
        "timeline",
    }
    alert = next(
        record for record in records if record["record_type"] == "alert_snapshot"
    )
    assert alert["affected_geometry_wkt"].startswith("POLYGON((108.95 34.24")
    assert all(record["is_simulated"] is True for record in records)
    assert all(len(record["payload_sha256"]) == 64 for record in records)
    assert all(len(record["canonical_payload_sha256"]) == 64 for record in records)
    assert all('"event_id"' not in record["payload_ciphertext"] for record in records)
    assert sum(record["record_type"] == "approval" for record in records) == 1
    assert sum(record["record_type"] == "outbox" for record in records) == 1
    assert sum(record["record_type"] == "dispatch_callback" for record in records) == 2
    assert sum(record["record_type"] == "feedback" for record in records) == 1
    assert sum(record["record_type"] == "evidence_package" for record in records) == 1


def test_non_simulated_source_is_quarantined_from_simulation_schema(tmp_path) -> None:
    db_path = tmp_path / "real-source.db"
    workflow = FloodWarningSystem(db_path).response_workflow
    workflow.create_event(
        EventCreateRequest(
            title="非模拟记录",
            area_id="district-real",
            alert=AlertInput(
                alert_id="REAL-ALERT-001",
                source_department="专业部门",
                disaster_type="暴雨",
                level="黄色",
                issued_at=datetime.now(timezone.utc),
                affected_area="真实片区",
                raw_content="真实来源占位记录",
                source_type="authority",
                source_version="authority-v1",
                is_simulated=False,
            ),
            operator_id="duty-real",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    records, quarantined = build_projection_records(db_path)

    assert records == []
    assert quarantined
    assert {item["error_code"] for item in quarantined} == {"ValueError"}
    assert all("non-simulated record" in item["error_detail"] for item in quarantined)


def test_document_governance_history_projects_to_simulation_shadow_without_plaintext(
    tmp_path,
) -> None:
    db_path = tmp_path / "document-shadow.db"
    workflow = FloodWarningSystem(db_path).response_workflow
    document = workflow.register_document(
        DocumentImportRequest(
            document_id="DISTRICT-PLAN-SHADOW",
            title="模拟区级下穿通道规程",
            version_label="2026-shadow",
            issuer="模拟区防办",
            jurisdiction="district-simulation",
            effective_at=datetime.now(timezone.utc),
            content="第一条 橙色预警时住建局应核查下穿通道。",
            operator_id="simulation-admin",
            operator_role=OperatorRole.ADMIN,
            terminal_id="simulation-console",
        )
    )

    records, quarantined = build_projection_records(db_path)

    assert quarantined == []
    document_records = {
        item["record_type"]: item
        for item in records
        if item["record_type"].startswith("document_")
        or item["record_type"] == "index_build"
    }
    assert set(document_records) == {
        "document_version",
        "document_source",
        "document_parse",
        "document_lifecycle",
        "index_build",
    }
    assert document_records["document_version"]["source_id"] == document.version_id
    assert all(item["is_simulated"] is True for item in document_records.values())
    assert all(
        "橙色预警" not in item["payload_ciphertext"]
        for item in document_records.values()
    )


def test_simulated_risk_object_registry_projects_versions_imports_and_point(
    tmp_path,
) -> None:
    db_path = tmp_path / "postgis-registry-source.db"
    workflow = FloodWarningSystem(db_path).response_workflow
    workflow.import_risk_object_registry(
        RiskObjectRegistryBatchImportRequest(
            area_id="district-simulated-registry",
            source_version="simulated-registry-v1",
            objects=[
                RiskObjectInput(
                    object_id="SIM-REGISTRY-001",
                    name="模拟重点学校",
                    object_type="学校",
                    location="模拟片区",
                    longitude=108.958,
                    latitude=34.244,
                    responsible_organization="模拟教育局",
                    responsible_role="模拟学校防汛负责人",
                    trigger_reasons=["模拟台账导入"],
                    source_refs=["simulation-registry-source"],
                    vulnerability="模拟脆弱性",
                    risk_score=80,
                    system_explanation="仅用于 PostGIS 影子迁移测试",
                    is_simulated=True,
                )
            ],
            operator_id="simulation-reviewer",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="simulation-console",
        )
    )
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="模拟候选对象清单迁移",
            area_id="district-simulated-registry",
            alert=AlertInput(
                alert_id="SIM-CANDIDATE-ALERT-001",
                source_department="模拟气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                affected_area="模拟片区",
                raw_content="仅用于 PostGIS 影子迁移测试",
                source_type="simulation",
                source_version="simulation-v1",
                is_simulated=True,
            ),
            operator_id="simulation-duty",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="simulation-console",
        )
    )
    discovery = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            entity_types=["学校"],
            min_risk_score=40,
            operator_id="simulation-duty",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="simulation-console",
        ),
    )
    object_id = discovery.candidates[0].object_id
    workflow.verify_risk_object(
        dashboard.event.event_id,
        object_id,
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="模拟确认",
            operator_id="simulation-reviewer",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="simulation-console",
        ),
    )
    frozen = workflow.freeze_candidate_object_list(
        dashboard.event.event_id,
        CandidateObjectListFreezeRequest(
            object_ids=[object_id],
            operator_id="simulation-reviewer",
            operator_role=OperatorRole.REVIEWER,
            terminal_id="simulation-console",
        ),
    )

    records, quarantined = build_projection_records(db_path)

    assert quarantined == []
    registry = next(
        item for item in records if item["record_type"] == "risk_object_registry"
    )
    assert registry["area_id"] == "district-simulated-registry"
    assert registry["version"] == 1
    assert registry["object_location_wkt"] == "POINT(108.958 34.244)"
    candidate_list = next(
        item for item in records if item["record_type"] == "candidate_object_list"
    )
    assert candidate_list["version"] == frozen.version
    assert candidate_list["event_id"] == dashboard.event.event_id
    assert candidate_list["is_simulated"] is True
    assert len(candidate_list["canonical_payload_sha256"]) == 64
    assert {
        item["record_type"]
        for item in records
        if item["record_type"].startswith("risk_object_registry")
    } == {
        "risk_object_registry",
        "risk_object_registry_version",
        "risk_object_registry_import",
    }


def test_projection_refuses_to_create_a_replacement_source_key(tmp_path) -> None:
    original = tmp_path / "original.db"
    seed_fixture(original)
    copied = tmp_path / "copied-without-key.db"
    shutil.copy2(original, copied)

    with pytest.raises(DataProtectionError, match="refusing to create"):
        build_projection_records(copied)

    assert not copied.with_suffix(".db.key").exists()
