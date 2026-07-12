from __future__ import annotations

import shutil
from datetime import datetime, timezone

import pytest

from flood_system.postgis_migration import build_projection_records
from flood_system.response_workflow.models import AlertInput, EventCreateRequest, OperatorRole
from flood_system.system import FloodWarningSystem
from flood_system.security import DataProtectionError
from scripts.seed_response_migration_fixture import seed_fixture


def test_simulated_response_projection_preserves_hashes_and_postgis_polygon(tmp_path) -> None:
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
        "feedback",
        "evidence_package",
        "timeline",
    }
    alert = next(record for record in records if record["record_type"] == "alert_snapshot")
    assert alert["affected_geometry_wkt"].startswith("POLYGON((108.95 34.24")
    assert all(record["is_simulated"] is True for record in records)
    assert all(len(record["payload_sha256"]) == 64 for record in records)
    assert all(len(record["canonical_payload_sha256"]) == 64 for record in records)
    assert all('"event_id"' not in record["payload_ciphertext"] for record in records)
    assert sum(record["record_type"] == "approval" for record in records) == 1
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


def test_projection_refuses_to_create_a_replacement_source_key(tmp_path) -> None:
    original = tmp_path / "original.db"
    seed_fixture(original)
    copied = tmp_path / "copied-without-key.db"
    shutil.copy2(original, copied)

    with pytest.raises(DataProtectionError, match="refusing to create"):
        build_projection_records(copied)

    assert not copied.with_suffix(".db.key").exists()
