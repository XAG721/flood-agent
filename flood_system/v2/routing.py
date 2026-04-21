from __future__ import annotations

from dataclasses import dataclass

from ..models import AreaProfile
from .models import EntityProfile, HazardState, RouteOption, TravelMode


@dataclass(slots=True)
class LiveTrafficSnapshot:
    road_id: str
    congestion_index: float
    note: str


class RoutePlanningService:
    def build_live_traffic(self, area_profile: AreaProfile, hazard_state: HazardState) -> list[LiveTrafficSnapshot]:
        road_risk = {item.road_id: item for item in hazard_state.road_reachability}
        traffic: list[LiveTrafficSnapshot] = []
        for index, road in enumerate(area_profile.roads, start=1):
            reachability = road_risk.get(road.road_id)
            if reachability is None:
                continue
            congestion = 0.35 + (index % 4) * 0.12
            if not reachability.accessible:
                congestion += 0.35
            traffic.append(
                LiveTrafficSnapshot(
                    road_id=road.road_id,
                    congestion_index=round(min(1.0, congestion), 2),
                    note="受积水与应急绕行影响" if congestion >= 0.6 else "通行压力可控",
                )
            )
        return traffic

    def route_entity(self, area_profile: AreaProfile, hazard_state: HazardState, entity: EntityProfile) -> tuple[list[RouteOption], list[RouteOption]]:
        road_lookup = {item.from_village: [] for item in hazard_state.road_reachability}
        for road in hazard_state.road_reachability:
            road_lookup.setdefault(road.from_village, []).append(road)

        reachable_roads = road_lookup.get(entity.village, [])
        travel_mode = entity.preferred_transport_mode
        available_shelters = [item for item in area_profile.shelters if item.accessible and item.available_capacity > 0]
        same_village_shelters = [item for item in available_shelters if item.village == entity.village]
        candidate_shelters = same_village_shelters or available_shelters[:3]

        safe_routes: list[RouteOption] = []
        blocked_routes: list[RouteOption] = []

        for idx, shelter in enumerate(candidate_shelters, start=1):
            road = reachable_roads[(idx - 1) % len(reachable_roads)] if reachable_roads else None
            if road is None:
                blocked_routes.append(
                    RouteOption(
                        route_id=f"{entity.entity_id}_blocked_{idx}",
                        summary=f"{entity.name} 暂无可用疏散通道前往 {shelter.name}",
                        destination_name=shelter.name,
                        destination_type="shelter",
                        travel_mode=travel_mode,
                        eta_minutes=999,
                        risk_score=100.0,
                        segments=[entity.location_hint, shelter.name],
                        blocked_reason="缺少可达路网数据",
                        available=False,
                    )
                )
                continue

            route = RouteOption(
                route_id=f"{entity.entity_id}_route_{idx}",
                summary=f"{entity.village} 经 {road.name} 前往 {shelter.name}",
                destination_name=shelter.name,
                destination_type="shelter",
                travel_mode=travel_mode,
                eta_minutes=max(8, road.travel_time_minutes + idx * 3 + (6 if travel_mode == TravelMode.ASSISTED else 0)),
                risk_score=55.0 if road.accessible else 92.0,
                segments=[entity.location_hint, road.name, shelter.name],
                risk_segments=[road.name] if not road.accessible or road.travel_time_minutes >= 20 else [],
                blocked_reason=road.failure_reason,
                available=road.accessible,
            )
            if route.available:
                safe_routes.append(route)
            else:
                blocked_routes.append(route)

        if not blocked_routes:
            for road in reachable_roads:
                if road.accessible:
                    continue
                blocked_routes.append(
                    RouteOption(
                        route_id=f"{entity.entity_id}_{road.road_id}_blocked",
                        summary=f"{road.name} 当前不建议通行",
                        destination_name=road.to_location,
                        destination_type="road_segment",
                        travel_mode=travel_mode,
                        eta_minutes=max(12, road.travel_time_minutes),
                        risk_score=95.0,
                        segments=[entity.location_hint, road.name],
                        risk_segments=[road.name],
                        blocked_reason=road.failure_reason or "道路积水超过通行阈值",
                        available=False,
                    )
                )

        safe_routes.sort(key=lambda item: (item.risk_score, item.eta_minutes))
        blocked_routes.sort(key=lambda item: (-item.risk_score, item.eta_minutes))
        return safe_routes[:3], blocked_routes[:3]
