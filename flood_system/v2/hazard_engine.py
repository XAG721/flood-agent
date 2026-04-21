from __future__ import annotations

from datetime import datetime, timezone

from ..models import AreaProfile, RiskLevel
from .models import HazardState, HazardTile, MonitoringPointState, ObservationIngestItem, RoadReachability


class HazardEngine:
    def compute(self, event_id: str, area_profile: AreaProfile, observations: list[ObservationIngestItem]) -> HazardState:
        if not observations:
            raise ValueError("At least one v2 observation is required.")

        ordered = sorted(observations, key=lambda item: item.observed_at)
        recent = ordered[-3:]
        latest = ordered[-1]
        rainfall_total = sum(item.rainfall_mm for item in ordered)
        rainfall_recent = sum(item.rainfall_mm for item in recent)
        max_level = max(item.water_level_m for item in ordered)
        blocked_count = sum(1 for item in ordered if item.road_blocked)
        citizen_reports = sum(item.citizen_reports for item in recent)

        base_score = min(
            100.0,
            rainfall_total * 0.18 + rainfall_recent * 0.45 + max_level * 9.5 + blocked_count * 7.0 + citizen_reports * 1.8,
        )
        trend = "stable"
        if len(ordered) >= 2 and ordered[-1].water_level_m > ordered[-2].water_level_m:
            trend = "rising"
        if rainfall_recent >= 60 or max_level >= 4.2:
            trend = "rapidly_rising"

        now = datetime.now(timezone.utc)
        uncertainty = round(min(0.48, 0.16 + blocked_count * 0.03 + citizen_reports * 0.01), 2)
        village_modifiers = self._village_modifiers(area_profile)
        hazard_tiles: list[HazardTile] = []
        peak_by_village: dict[str, float] = {}

        for village in area_profile.villages:
            modifier = village_modifiers.get(village, 0.0)
            for horizon_minutes, score_factor, depth_bonus in ((10, 0.92, 4.0), (30, 1.0, 10.0), (60, 1.08, 18.0)):
                score = max(0.0, min(100.0, base_score * score_factor + modifier))
                depth_cm = max(0.0, (max_level * 16.0) + depth_bonus + modifier * 0.5)
                hazard_tiles.append(
                    HazardTile(
                        tile_id=f"{village}_{horizon_minutes}",
                        area_name=village,
                        horizon_minutes=horizon_minutes,
                        risk_level=self._risk_from_score(score),
                        risk_score=round(score, 2),
                        predicted_water_depth_cm=round(depth_cm, 1),
                        trend=trend,
                        uncertainty=uncertainty,
                        affected_roads=[road.name for road in area_profile.roads if road.from_village == village][:2],
                    )
                )
                peak_by_village[village] = max(peak_by_village.get(village, 0.0), score)

        road_reachability: list[RoadReachability] = []
        for index, road in enumerate(area_profile.roads, start=1):
            local_score = peak_by_village.get(road.from_village, base_score)
            risk_bonus = 0.0
            if any(marker in road.risk_note for marker in ("涵洞", "低洼", "桥下", "地下")):
                risk_bonus += 12.0
            if blocked_count:
                risk_bonus += 8.0
            travel_time = max(6, 7 + index + int(local_score // 18))
            depth_limit = 15.0 if any(marker in road.risk_note for marker in ("涵洞", "桥下")) else 28.0
            road_score = local_score + risk_bonus
            accessible = road.accessible and road_score < 78.0
            road_reachability.append(
                RoadReachability(
                    road_id=road.road_id,
                    name=road.name,
                    from_village=road.from_village,
                    to_location=road.to_location,
                    accessible=accessible,
                    travel_time_minutes=travel_time,
                    depth_limit_cm=depth_limit,
                    failure_reason="" if accessible else f"{road.name} 可能因积水和拥堵失去通行能力。",
                )
            )

        monitoring_points = [
            MonitoringPointState(
                point_name=point_name,
                latest_water_level_m=round(latest.water_level_m + idx * 0.06, 2),
                latest_rainfall_mm=round(max(0.0, latest.rainfall_mm - idx * 1.3), 1),
                status="严重" if base_score >= 65 else "关注" if base_score >= 40 else "正常",
                updated_at=latest.observed_at,
            )
            for idx, point_name in enumerate(area_profile.monitoring_points[:6], start=1)
        ]

        return HazardState(
            event_id=event_id,
            area_id=area_profile.area_id,
            generated_at=now,
            overall_risk_level=self._risk_from_score(base_score),
            overall_score=round(base_score, 2),
            trend=trend,
            uncertainty=uncertainty,
            freshness_seconds=max(0, int((now - latest.observed_at.astimezone(timezone.utc)).total_seconds())),
            hazard_tiles=hazard_tiles,
            road_reachability=road_reachability,
            monitoring_points=monitoring_points,
        )

    @staticmethod
    def _risk_from_score(score: float) -> RiskLevel:
        if score >= 85:
            return RiskLevel.RED
        if score >= 68:
            return RiskLevel.ORANGE
        if score >= 48:
            return RiskLevel.YELLOW
        if score >= 28:
            return RiskLevel.BLUE
        return RiskLevel.NONE

    @staticmethod
    def _village_modifiers(area_profile: AreaProfile) -> dict[str, float]:
        modifiers: dict[str, float] = {village: 0.0 for village in area_profile.villages}
        for village in area_profile.villages:
            for spot in area_profile.flood_prone_spots:
                if village in spot:
                    modifiers[village] += 8.0
            for facility in [*area_profile.schools, *area_profile.medical_facilities]:
                if village in facility:
                    modifiers[village] += 3.5
        return modifiers
