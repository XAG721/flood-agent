from __future__ import annotations

import math
from datetime import datetime

from ..v2.models import EntityProfile
from .models import (
    AlertSnapshot,
    CandidateFeatureSnapshot,
    RiskObjectInput,
    RiskObjectRegistryRecord,
    SensitiveContact,
)


ALERT_LEVEL_WEIGHTS = {
    "blue": 8,
    "蓝色": 8,
    "yellow": 16,
    "黄色": 16,
    "orange": 24,
    "橙色": 24,
    "red": 32,
    "红色": 32,
}

ENTITY_TYPE_WEIGHTS = {
    "resident": 12,
    "community": 10,
    "school": 18,
    "factory": 14,
    "hospital": 24,
    "nursing_home": 26,
    "metro_station": 22,
    "underground_space": 25,
}

ENTITY_TYPE_LABELS = {
    "resident": "重点居民",
    "community": "社区",
    "school": "学校",
    "factory": "工厂",
    "hospital": "医院",
    "nursing_home": "养老机构",
    "metro_station": "地铁站",
    "underground_space": "地下空间",
}

ENTITY_RESPONSIBILITIES = {
    "resident": ("属地街道", "社区网格员"),
    "community": ("属地街道", "社区负责人"),
    "school": ("区教育局", "学校防汛负责人"),
    "factory": ("区应急管理局", "企业安全负责人"),
    "hospital": ("区卫生健康局", "医院应急负责人"),
    "nursing_home": ("区民政局", "机构应急负责人"),
    "metro_station": ("轨道交通运营单位", "车站值班负责人"),
    "underground_space": ("区住建局", "地下空间管理负责人"),
}


def calculate_profile_risk_score(alert_level: str, profile: EntityProfile) -> float:
    """Return an explainable screening score, not a hydraulic/GIS risk result."""
    normalized_level = alert_level.strip().lower()
    alert_points = max(
        (
            weight
            for label, weight in ALERT_LEVEL_WEIGHTS.items()
            if label in normalized_level
        ),
        default=12,
    )
    population = max(profile.current_occupancy, profile.resident_count)
    population_points = min(16.0, population / 75.0)
    score = (
        20.0
        + alert_points
        + ENTITY_TYPE_WEIGHTS.get(profile.entity_type.value, 10)
        + min(12, len(profile.vulnerability_tags) * 3)
        + min(10, len(profile.mobility_constraints) * 4)
        + population_points
    )
    return round(min(100.0, score), 1)


def calculate_registry_risk_score(
    alert_level: str, record: RiskObjectRegistryRecord
) -> float:
    """Combine the governed registry score with the current warning level."""
    normalized_level = alert_level.strip().lower()
    alert_points = max(
        (
            weight
            for label, weight in ALERT_LEVEL_WEIGHTS.items()
            if label in normalized_level
        ),
        default=12,
    )
    completeness_penalty = min(20.0, len(record.missing_fields) * 3.0)
    score = record.risk_score * 0.7 + alert_points - completeness_penalty
    return round(min(100.0, max(0.0, score)), 1)


def build_candidate_feature_snapshot(
    profile: EntityProfile | RiskObjectRegistryRecord,
    *,
    score: float,
    rank: int,
    alert: AlertSnapshot,
    association_mode: str,
    now: datetime,
) -> CandidateFeatureSnapshot:
    """Freeze the five FR-03 feature families used by candidate review."""
    is_registry = isinstance(profile, RiskObjectRegistryRecord)
    object_id = profile.object_id if is_registry else profile.entity_id
    missing: list[str] = []

    if association_mode == "gis_point_in_polygon":
        spatial_score = 1.0
        spatial_explanation = "EPSG:4326 登记点位于当前预警多边形内"
    else:
        spatial_score = 0.6
        spatial_explanation = "预警缺少多边形，空间特征降级为同 area_id 区域匹配"
        missing.append("affected_geometry")

    if is_registry:
        has_validity_window = (
            profile.registry_valid_from is not None
            or profile.registry_valid_until is not None
        )
        within_validity_window = (
            (profile.registry_valid_from is None or profile.registry_valid_from <= now)
            and (
                profile.registry_valid_until is None
                or profile.registry_valid_until > now
            )
        )
        temporal_score = (
            1.0 if has_validity_window and within_validity_window else 0.75
        )
        if not has_validity_window:
            missing.append("registry_validity_window")
        temporal_explanation = (
            "台账状态有效，且当前时刻位于登记有效期内"
            if has_validity_window
            else "台账状态有效，但未登记明确有效期边界"
        )
        attribute_score = round(profile.risk_score / 100.0, 4)
        semantic_text = " ".join(
            profile.trigger_reasons
            + [
                profile.object_type,
                profile.vulnerability,
                profile.historical_risk,
                profile.system_explanation,
            ]
        ).casefold()
        data_quality_score = round(
            max(0.0, 1.0 - min(1.0, len(profile.missing_fields) / 5.0)),
            4,
        )
        missing.extend(profile.missing_fields)
        feature_version = "candidate-registry-features-v2"
    else:
        temporal_score = 0.5
        temporal_explanation = "Legacy 对象未提供版本化有效期，只能使用当前台账快照"
        missing.append("registry_validity_window")
        attribute_score = round(min(1.0, score / 100.0), 4)
        semantic_text = " ".join(
            list(profile.vulnerability_tags)
            + list(profile.mobility_constraints)
            + [
                profile.entity_type.value,
                profile.location_hint,
                str(profile.custom_attributes.get("historical_risk", "")),
            ]
        ).casefold()
        quality_checks = (
            profile.longitude is not None and profile.latitude is not None,
            bool(profile.location_hint.strip()),
            bool(profile.emergency_contacts),
            bool(profile.vulnerability_tags or profile.mobility_constraints),
        )
        data_quality_score = round(sum(quality_checks) / len(quality_checks), 4)
        if not quality_checks[0]:
            missing.append("object_coordinates")
        if not quality_checks[2]:
            missing.append("emergency_contacts")
        feature_version = "candidate-features-v2"

    alert_text = " ".join(
        [
            alert.disaster_type,
            alert.level,
            alert.affected_area,
            alert.raw_content,
        ]
    ).casefold()
    semantic_terms = [
        term
        for term in ("洪水", "暴雨", "内涝", "积水", "排水", "防汛", "地下")
        if term in alert_text
    ]
    semantic_matches = [term for term in semantic_terms if term in semantic_text]
    semantic_score = (
        round(len(semantic_matches) / len(semantic_terms), 4)
        if semantic_terms
        else 0.5
    )
    if not semantic_matches:
        missing.append("semantic_risk_tag_match")

    calibrated_confidence = round(
        1.0 / (1.0 + math.exp(-((score - 55.0) / 12.0))),
        4,
    )
    return CandidateFeatureSnapshot(
        object_id=object_id,
        rank=rank,
        spatial_score=spatial_score,
        temporal_score=temporal_score,
        attribute_score=attribute_score,
        semantic_score=semantic_score,
        data_quality_score=data_quality_score,
        raw_score=score,
        calibrated_confidence=calibrated_confidence,
        missing_features=sorted(set(missing)),
        explanations={
            "spatial": spatial_explanation,
            "temporal": temporal_explanation,
            "attribute": "对象登记属性与风险分形成排序属性特征",
            "semantic": (
                "预警语义标签与对象风险描述匹配：" + "、".join(semantic_matches)
                if semantic_matches
                else "预警语义标签未在对象风险描述中直接命中"
            ),
            "data_quality": "按坐标、责任、联系人、风险标签及显式缺失字段计算",
        },
        feature_version=feature_version,
    )


def point_in_polygon(
    longitude: float,
    latitude: float,
    ring: list[tuple[float, float]],
) -> bool:
    """Boundary-inclusive ray casting for an EPSG:4326 exterior ring."""
    inside = False
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        cross = (longitude - x1) * (y2 - y1) - (latitude - y1) * (x2 - x1)
        on_segment = (
            abs(cross) <= 1e-12
            and min(x1, x2) - 1e-12 <= longitude <= max(x1, x2) + 1e-12
            and min(y1, y2) - 1e-12 <= latitude <= max(y1, y2) + 1e-12
        )
        if on_segment:
            return True
        if (y1 > latitude) != (y2 > latitude):
            crossing_x = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if longitude < crossing_x:
                inside = not inside
    return inside


def build_risk_object_candidate(
    profile: EntityProfile,
    score: float,
    alert: AlertSnapshot,
    association_mode: str = "area_registry",
) -> RiskObjectInput:
    entity_type = profile.entity_type.value
    organization, role = ENTITY_RESPONSIBILITIES.get(
        entity_type, ("属地街道", "防汛责任人")
    )
    vulnerability_parts = list(profile.vulnerability_tags) + list(
        profile.mobility_constraints
    )
    population = max(profile.current_occupancy, profile.resident_count)
    if population:
        vulnerability_parts.append(f"在场或登记人数约 {population} 人")

    contacts = [
        SensitiveContact(
            name=item.name, role=item.role or "应急联系人", phone=item.phone
        )
        for item in profile.emergency_contacts
        if item.phone
    ]
    source_ref = str(profile.custom_attributes.get("source_ref", "")).strip()
    sources = [alert.snapshot_id, f"entity_profile:{profile.entity_id}"]
    if alert.affected_geometry is not None:
        sources.append(f"alert_geometry:{alert.snapshot_id}:EPSG4326")
    if source_ref:
        sources.append(source_ref)

    if (
        alert.affected_geometry is not None
        and profile.longitude is not None
        and profile.latitude is not None
    ):
        spatial_reason = f"对象登记点落入{alert.level}{alert.disaster_type}预警多边形"
    else:
        spatial_reason = (
            f"{alert.level}{alert.disaster_type}预警与对象同属区域 {profile.area_id}"
        )
    trigger_reasons = [
        spatial_reason,
        f"对象类型 {ENTITY_TYPE_LABELS.get(entity_type, entity_type)} 纳入候选筛查",
    ]
    if profile.vulnerability_tags:
        trigger_reasons.append(
            "存在脆弱性标签：" + "、".join(profile.vulnerability_tags)
        )

    return RiskObjectInput(
        object_id=profile.entity_id,
        name=profile.name,
        object_type=ENTITY_TYPE_LABELS.get(entity_type, entity_type),
        location=f"{profile.village}；{profile.location_hint}",
        longitude=profile.longitude,
        latitude=profile.latitude,
        responsible_organization=organization,
        responsible_role=role,
        trigger_reasons=trigger_reasons,
        source_refs=list(dict.fromkeys(sources)),
        vulnerability="；".join(vulnerability_parts) or "区域台账未登记额外脆弱性",
        historical_risk=str(profile.custom_attributes.get("historical_risk", "")),
        risk_score=score,
        raw_candidate_score=score,
        calibrated_confidence=round(
            1.0 / (1.0 + math.exp(-((score - 55.0) / 12.0))), 4
        ),
        calibration_version="candidate-logistic-v1",
        association_mode=association_mode,
        system_explanation=(
            "基于预警等级、对象类型、登记人数、脆弱性和行动约束形成候选排序；"
            + (
                "空间关联采用 EPSG:4326 点落预警多边形判断；"
                if alert.affected_geometry is not None
                else "空间关联降级为 area_id 区域台账匹配；"
            )
            + "结果仅用于人工核验优先级，不替代水动力风险分析。"
        ),
        sensitive_contacts=contacts,
        special_population_notes="、".join(vulnerability_parts),
    )


def build_registry_candidate(
    record: RiskObjectRegistryRecord,
    score: float,
    alert: AlertSnapshot,
    association_mode: str,
) -> RiskObjectInput:
    payload = record.model_dump(include=set(RiskObjectInput.model_fields))
    source_refs = list(
        dict.fromkeys(
            record.source_refs
            + [
                alert.snapshot_id,
                f"risk_registry:{record.area_id}:{record.object_id}:v{record.registry_version}",
            ]
        )
    )
    trigger_reasons = list(record.trigger_reasons)
    if alert.affected_geometry is not None:
        trigger_reasons.insert(
            0,
            f"对象登记点落入{alert.level}{alert.disaster_type}预警多边形",
        )
    else:
        trigger_reasons.insert(
            0,
            f"{alert.level}{alert.disaster_type}预警与对象同属区域 {record.area_id}",
        )
    payload.update(
        {
            "trigger_reasons": list(dict.fromkeys(trigger_reasons)),
            "source_refs": source_refs,
            "raw_candidate_score": score,
            "calibrated_confidence": round(
                1.0 / (1.0 + math.exp(-((score - 55.0) / 12.0))), 4
            ),
            "association_mode": association_mode,
            "system_explanation": (
                "基于版本化风险对象主数据、预警等级、登记风险分和数据完整度形成候选排序；"
                + (
                    "空间关联采用 EPSG:4326 点落预警多边形判断；"
                    if alert.affected_geometry is not None
                    else "空间关联降级为 area_id 区域台账匹配；"
                )
                + "结果仅用于人工核验优先级。"
            ),
        }
    )
    return RiskObjectInput.model_validate(payload)
