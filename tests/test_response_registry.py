from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flood_system.response_workflow.models import (
    AlertInput,
    CandidateDiscoveryRequest,
    EventCreateRequest,
    GeoPolygon,
    ObjectVerificationStatus,
    OperatorRole,
    RiskObjectBatchRequest,
    RiskObjectInput,
    RiskObjectVerificationRequest,
)
from flood_system.system import FloodWarningSystem

from tests.support.response_workflow import (
    seed_workflow,
)


def test_area_registry_candidate_discovery_is_explainable_and_requires_manual_verification(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "candidate-discovery.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="碑林区候选风险对象筛查",
            area_id="beilin_10km2",
            alert=AlertInput(
                alert_id="ALERT-DISCOVERY-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                valid_until=now + timedelta(hours=3),
                affected_area="碑林区重点片区",
                raw_content="橙色暴雨预警",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    result = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            entity_types=["school", "hospital", "underground_space"],
            min_risk_score=45,
            max_candidates=10,
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
            terminal_id="duty-terminal",
        ),
    )

    assert result.association_mode == "area_registry"
    assert result.scanned_profiles == 3
    assert result.matched_profiles == 3
    assert result.limitations and "GIS" in result.limitations[0]
    assert all(
        item.verification_status == ObjectVerificationStatus.PENDING
        for item in result.candidates
    )
    assert all(
        result.alert_snapshot_id in item.source_refs for item in result.candidates
    )
    assert all(
        f"entity_profile:{item.object_id}" in item.source_refs
        for item in result.candidates
    )
    assert all(
        "不替代水动力风险分析" in item.system_explanation for item in result.candidates
    )
    assert all(
        item.raw_candidate_score == item.risk_score for item in result.candidates
    )
    assert all(item.calibrated_confidence is not None for item in result.candidates)
    assert all(
        item.calibration_version == "candidate-logistic-v1"
        for item in result.candidates
    )

    verified = workflow.verify_risk_object(
        dashboard.event.event_id,
        result.candidates[0].object_id,
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="台账与属地电话复核通过",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
        ),
    )
    assert verified.verification_status == ObjectVerificationStatus.CONFIRMED


def test_risk_object_alias_duplicate_is_merged_and_expired_registry_is_blocked(
    tmp_path,
):
    _, workflow, event_id, _ = seed_workflow(tmp_path)
    canonical = workflow.repository.get_event_risk_object(event_id, "TUNNEL-001")
    assert canonical is not None
    base = canonical.model_dump(include=set(RiskObjectInput.model_fields))
    duplicate = RiskObjectInput(
        **{
            **base,
            "object_id": "TUNNEL-ALIAS-001",
            "canonical_object_id": None,
            "duplicate_of": "TUNNEL-001",
            "name": "测试隧道（旧称）",
            "aliases": ["测试通道"],
            "source_refs": ["LEGACY-REGISTRY-1"],
        }
    )

    merged = workflow.add_risk_objects(
        event_id,
        RiskObjectBatchRequest(
            objects=[duplicate],
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )[0]

    assert merged.object_id == "TUNNEL-001"
    assert merged.canonical_object_id == "TUNNEL-001"
    assert {"测试隧道（旧称）", "测试通道", "TUNNEL-ALIAS-001"} <= set(merged.aliases)
    assert "LEGACY-REGISTRY-1" in merged.source_refs
    assert len(workflow.repository.list_event_risk_objects(event_id)) == 1
    versions = [
        item
        for item in workflow.repository.list_risk_object_versions(event_id)
        if item.object_id == "TUNNEL-001"
    ]
    assert versions[-1].change_type == "candidate_duplicate_merged"
    assert any(
        item.action == "candidate_duplicate_merged"
        for item in workflow.repository.list_timeline_entries(event_id)
    )

    expired = RiskObjectInput(
        **{
            **base,
            "object_id": "TUNNEL-EXPIRED-001",
            "canonical_object_id": None,
            "duplicate_of": None,
            "name": "已过期下穿通道台账",
            "registry_valid_until": datetime.now(timezone.utc) - timedelta(minutes=1),
        }
    )
    expired_record = workflow.add_risk_objects(
        event_id,
        RiskObjectBatchRequest(
            objects=[expired],
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )[0]
    assert expired_record.stale is True
    with pytest.raises(ValueError, match="expired or inactive"):
        workflow.verify_risk_object(
            event_id,
            expired_record.object_id,
            RiskObjectVerificationRequest(
                decision=ObjectVerificationStatus.CONFIRMED,
                note="不应允许确认过期台账",
                operator_id="reviewer-1",
                operator_role=OperatorRole.REVIEWER,
            ),
        )


def test_candidate_discovery_returns_empty_result_when_area_has_no_registered_profiles(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "candidate-discovery-empty.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="无台账片区筛查",
            area_id="area-without-profile",
            alert=AlertInput(
                alert_id="ALERT-DISCOVERY-EMPTY",
                source_department="气象部门",
                disaster_type="暴雨",
                level="黄色",
                issued_at=now,
                affected_area="无台账片区",
                raw_content="黄色暴雨预警",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    result = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )

    assert result.scanned_profiles == 0
    assert result.matched_profiles == 0
    assert result.candidates == []


def test_missing_registry_object_can_be_manually_added_and_confirmed(tmp_path):
    system = FloodWarningSystem(tmp_path / "candidate-manual-supplement.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="台账缺失对象人工补录",
            area_id="area-without-profile",
            alert=AlertInput(
                alert_id="ALERT-MANUAL-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                affected_area="无台账片区",
                raw_content="现场发现台账外地下空间",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )
    event_id = dashboard.event.event_id
    discovered = workflow.discover_risk_objects(
        event_id,
        CandidateDiscoveryRequest(
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )
    assert discovered.candidates == []

    supplemented = workflow.add_risk_objects(
        event_id,
        RiskObjectBatchRequest(
            objects=[
                RiskObjectInput(
                    object_id="MANUAL-SPACE-001",
                    name="现场发现地下空间",
                    object_type="underground_space",
                    location="无台账片区临街入口",
                    responsible_organization="属地街道",
                    responsible_role="防汛联络员",
                    trigger_reasons=["现场巡查发现且预警期间存在倒灌风险"],
                    source_refs=["field_report:REPORT-001"],
                    vulnerability="地下入口低于路面",
                    risk_score=72,
                    system_explanation="台账无记录，按现场报告人工补录，必须由复核岗确认。",
                    association_mode="manual_supplement",
                )
            ],
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )[0]
    confirmed = workflow.verify_risk_object(
        event_id,
        supplemented.object_id,
        RiskObjectVerificationRequest(
            decision=ObjectVerificationStatus.CONFIRMED,
            note="现场照片与属地电话复核通过",
            operator_id="reviewer-1",
            operator_role=OperatorRole.REVIEWER,
        ),
    )

    assert confirmed.association_mode == "manual_supplement"
    assert confirmed.verification_status == ObjectVerificationStatus.CONFIRMED
    actions = [
        item.action for item in workflow.repository.list_timeline_entries(event_id)
    ]
    assert "candidate_added" in actions
    assert "candidate_confirmed" in actions


def test_candidate_discovery_uses_epsg4326_point_in_polygon_when_alert_geometry_is_available(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "candidate-discovery-gis.db")
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="学校周边精确空间筛查",
            area_id="beilin_10km2",
            alert=AlertInput(
                alert_id="ALERT-GIS-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                affected_area="文艺路小学周边",
                affected_geometry=GeoPolygon(
                    coordinates=[
                        (108.9565, 34.2425),
                        (108.9595, 34.2425),
                        (108.9595, 34.2455),
                        (108.9565, 34.2455),
                        (108.9565, 34.2425),
                    ]
                ),
                raw_content="学校周边短时强降雨预警范围",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    result = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            entity_types=["school", "hospital"],
            min_risk_score=45,
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )

    assert result.association_mode == "gis_point_in_polygon"
    assert result.scanned_profiles == 2
    assert result.spatially_evaluated_profiles == 2
    assert result.spatially_matched_profiles == 1
    assert result.excluded_unlocated_profiles == 0
    assert [item.object_id for item in result.candidates] == ["school_wyl_primary"]
    assert any(
        ref.startswith("alert_geometry:") for ref in result.candidates[0].source_refs
    )
    assert result.candidates[0].trigger_reasons[0] == "对象登记点落入橙色暴雨预警多边形"
    assert "108." not in result.candidates[0].trigger_reasons[0]
    assert result.limitations == []


def test_candidate_discovery_excludes_missing_coordinates_with_explicit_limitation(
    tmp_path,
):
    system = FloodWarningSystem(tmp_path / "candidate-discovery-missing-coordinate.db")
    profile = system.repository.get_v2_entity_profile("school_wyl_primary")
    assert profile is not None
    system.repository.save_v2_entity_profile(
        profile.model_copy(update={"longitude": None, "latitude": None})
    )
    workflow = system.response_workflow
    now = datetime.now(timezone.utc)
    dashboard = workflow.create_event(
        EventCreateRequest(
            title="缺少坐标对象精确筛查",
            area_id="beilin_10km2",
            alert=AlertInput(
                alert_id="ALERT-GIS-MISSING-001",
                source_department="气象部门",
                disaster_type="暴雨",
                level="橙色",
                issued_at=now,
                affected_area="文艺路小学周边",
                affected_geometry=GeoPolygon(
                    coordinates=[
                        (108.9565, 34.2425),
                        (108.9595, 34.2425),
                        (108.9595, 34.2455),
                        (108.9565, 34.2455),
                        (108.9565, 34.2425),
                    ]
                ),
                raw_content="学校周边短时强降雨预警范围",
            ),
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        )
    )

    result = workflow.discover_risk_objects(
        dashboard.event.event_id,
        CandidateDiscoveryRequest(
            entity_types=["school"],
            min_risk_score=45,
            operator_id="duty-1",
            operator_role=OperatorRole.DUTY_OFFICER,
        ),
    )

    assert result.scanned_profiles == 1
    assert result.spatially_evaluated_profiles == 0
    assert result.excluded_unlocated_profiles == 1
    assert result.candidates == []
    assert result.limitations == [
        "1 个对象缺少 EPSG:4326 坐标，未纳入本次精确空间筛查。"
    ]
