from __future__ import annotations

from datetime import datetime, timezone

from ..models import AreaProfile, ResourceStatus, RiskLevel
from .models import EntityImpactView, EntityProfile, EvidenceItem, EvidenceType, ExposureSummary, HazardState, HazardTile
from .routing import RoutePlanningService


class ExposureEngine:
    _RISK_LEVEL_TEXT = {
        RiskLevel.NONE: "无明显风险",
        RiskLevel.BLUE: "蓝色风险",
        RiskLevel.YELLOW: "黄色风险",
        RiskLevel.ORANGE: "橙色风险",
        RiskLevel.RED: "红色风险",
    }

    _VULNERABILITY_TEXT = {
        "elderly": "老年人居多",
        "limited_mobility": "行动不便",
        "chronic_disease": "慢性病用药需求",
        "children": "儿童集中",
        "dismissal_peak": "放学高峰",
        "inventory": "库存物资集中",
        "hazmat_sensitive": "危化品敏感",
        "critical_service": "关键公共服务",
        "patients": "患者集中",
        "bedridden": "卧床人员较多",
        "medical_support": "医疗支持依赖高",
        "underground": "地下空间风险",
        "commuter_peak": "通勤高峰",
        "complex_egress": "疏散路径复杂",
        "low_lying": "地势低洼",
        "basement": "地下空间较多",
        "mixed_population": "人群构成复杂",
    }

    def __init__(self, route_planner: RoutePlanningService) -> None:
        self.route_planner = route_planner

    def summarize(
        self,
        event_id: str,
        area_profile: AreaProfile,
        hazard_state: HazardState,
        entities: dict[str, EntityProfile],
        resource_status: ResourceStatus,
        *,
        evidence: list[EvidenceItem] | None = None,
    ) -> ExposureSummary:
        impacts = [
            self.assess_entity(
                event_id,
                entity,
                area_profile,
                hazard_state,
                resource_status,
                evidence=evidence or [],
            )
            for entity in entities.values()
        ]
        impacts.sort(key=lambda item: (-self._risk_rank(item.risk_level), item.time_to_impact_minutes))
        top_risks = [f"{item.entity.name}: {item.risk_reason[0]}" for item in impacts[:5] if item.risk_reason]
        return ExposureSummary(
            event_id=event_id,
            area_id=area_profile.area_id,
            generated_at=datetime.now(timezone.utc),
            affected_entities=impacts,
            top_risks=top_risks,
        )

    def assess_entity(
        self,
        event_id: str,
        entity: EntityProfile,
        area_profile: AreaProfile,
        hazard_state: HazardState,
        resource_status: ResourceStatus,
        *,
        evidence: list[EvidenceItem],
    ) -> EntityImpactView:
        tile = self._select_hazard_tile(entity, hazard_state)
        tile_label = tile.area_name or entity.village
        horizon_minutes = tile.horizon_minutes or 10

        entity_score = tile.risk_score + len(entity.vulnerability_tags) * 3.5
        if entity.entity_type.value in {"nursing_home", "metro_station", "underground_space"}:
            entity_score += 12.0
        if entity.entity_type.value in {"school", "hospital"}:
            entity_score += 8.0
        if entity.custom_attributes.get("hazardous_material"):
            entity_score += 8.0

        risk_level = self._risk_from_score(entity_score)
        time_to_impact = max(8, 80 - int(entity_score))
        safe_routes, blocked_routes = self.route_planner.route_entity(area_profile, hazard_state, entity)
        nearest_shelters = [route.destination_name for route in safe_routes[:2]]

        risk_reason = [
            (
                f"{tile_label} {horizon_minutes}分钟风险栅格评分为 {tile.risk_score:.0f}，"
                f"预测积水深度约 {tile.predicted_water_depth_cm:.0f} 厘米。"
            )
        ]
        if entity.vulnerability_tags:
            risk_reason.append(
                f"对象脆弱性因素会放大影响程度：{self._join_vulnerability_tags(entity.vulnerability_tags[:3])}。"
            )
        if blocked_routes:
            risk_reason.append(
                f"该对象已有 {len(blocked_routes)} 条通行或转移路线受阻，现场处置窗口正在缩小。"
            )

        resource_gap: list[str] = []
        if entity.entity_type.value in {"resident", "nursing_home"} and "needs_assistance" in entity.mobility_constraints:
            if resource_status.vehicle_count < 4:
                resource_gap.append("协助转运车辆储备不足")
            if resource_status.medical_staff_count < 12:
                resource_gap.append("医疗支援覆盖不足")
        if entity.entity_type.value == "factory" and resource_status.power_generators < 4:
            resource_gap.append("备用电源保障不足")

        entity_evidence = list(evidence)
        entity_evidence.insert(
            0,
            EvidenceItem(
                evidence_type=EvidenceType.REALTIME,
                title=f"{tile_label} 实时风险栅格",
                source_id=tile.tile_id,
                excerpt=(
                    f"{horizon_minutes}分钟风险栅格当前为{self._RISK_LEVEL_TEXT.get(tile.risk_level, tile.risk_level.value)}，"
                    f"预测积水深度约 {tile.predicted_water_depth_cm:.0f} 厘米。"
                ),
                timestamp=hazard_state.generated_at,
                priority=0,
            ),
        )

        return EntityImpactView(
            event_id=event_id,
            entity=entity,
            risk_level=risk_level,
            time_to_impact_minutes=time_to_impact,
            risk_reason=risk_reason,
            safe_routes=safe_routes,
            blocked_routes=blocked_routes,
            nearest_shelters=nearest_shelters,
            resource_gap=resource_gap,
            evidence=entity_evidence[:6],
        )

    @staticmethod
    def _risk_from_score(score: float) -> RiskLevel:
        if score >= 86:
            return RiskLevel.RED
        if score >= 68:
            return RiskLevel.ORANGE
        if score >= 48:
            return RiskLevel.YELLOW
        if score >= 28:
            return RiskLevel.BLUE
        return RiskLevel.NONE

    @staticmethod
    def _risk_rank(level: RiskLevel) -> int:
        return {RiskLevel.RED: 4, RiskLevel.ORANGE: 3, RiskLevel.YELLOW: 2, RiskLevel.BLUE: 1, RiskLevel.NONE: 0}[level]

    @staticmethod
    def _select_hazard_tile(entity: EntityProfile, hazard_state: HazardState) -> HazardTile:
        exact_match = [tile for tile in hazard_state.hazard_tiles if tile.area_name == entity.village and tile.horizon_minutes == 10]
        if exact_match:
            return exact_match[0]

        village_match = [tile for tile in hazard_state.hazard_tiles if tile.area_name == entity.village]
        if village_match:
            village_match.sort(key=lambda item: abs(item.horizon_minutes - 10))
            return village_match[0]

        short_term_tiles = [tile for tile in hazard_state.hazard_tiles if tile.horizon_minutes == 10]
        if short_term_tiles:
            return short_term_tiles[0]

        return hazard_state.hazard_tiles[0]

    @classmethod
    def _join_vulnerability_tags(cls, tags: list[str]) -> str:
        translated = [cls._VULNERABILITY_TEXT.get(tag, tag) for tag in tags]
        return "、".join(translated)
