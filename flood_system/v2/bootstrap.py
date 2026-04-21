from __future__ import annotations

import json
import os
from pathlib import Path

from ..models import AreaProfile
from .models import ContactProfile, EntityProfile, EntityType, TravelMode


def build_entity_profiles(area_profile: AreaProfile) -> dict[str, EntityProfile]:
    external_profiles = _load_external_entity_profiles(area_profile.area_id)
    if external_profiles is not None:
        return external_profiles
    area_id = area_profile.area_id
    return {
        "resident_elderly_ls1": EntityProfile(
            entity_id="resident_elderly_ls1",
            area_id=area_id,
            entity_type=EntityType.RESIDENT,
            name="李叔家庭",
            village="连石街坊",
            location_hint="连石街坊北巷 17 号院",
            resident_count=2,
            current_occupancy=2,
            vulnerability_tags=["elderly", "limited_mobility", "chronic_disease"],
            mobility_constraints=["stairs", "needs_assistance"],
            preferred_transport_mode=TravelMode.ASSISTED,
            notification_preferences=["sms", "community_call"],
            emergency_contacts=[ContactProfile(name="社区网格员张敏", phone="13800000001", role="社区联络员")],
            custom_attributes={"wheelchair": True, "medication_window_minutes": 40},
        ),
        "school_wyl_primary": EntityProfile(
            entity_id="school_wyl_primary",
            area_id=area_id,
            entity_type=EntityType.SCHOOL,
            name="五一路小学",
            village="五一路片区",
            location_hint="五一路小学东门及接送区",
            resident_count=860,
            current_occupancy=820,
            vulnerability_tags=["children", "dismissal_peak"],
            preferred_transport_mode=TravelMode.WALK,
            notification_preferences=["dashboard", "sms"],
            emergency_contacts=[ContactProfile(name="校长陈芳", phone="13800000002", role="校方负责人")],
            custom_attributes={"school_bus_count": 6, "after_school_peak": True},
        ),
        "factory_wyr_bio": EntityProfile(
            entity_id="factory_wyr_bio",
            area_id=area_id,
            entity_type=EntityType.FACTORY,
            name="渭源生物工厂",
            village="五一路片区",
            location_hint="A 区装卸平台与冷链仓库",
            resident_count=120,
            current_occupancy=74,
            vulnerability_tags=["inventory", "hazmat_sensitive"],
            preferred_transport_mode=TravelMode.VEHICLE,
            notification_preferences=["dashboard", "phone"],
            emergency_contacts=[ContactProfile(name="厂区值班长赵峰", phone="13800000003", role="工厂值守")],
            inventory_summary="冷链成品与原材料库存约 3200 箱。",
            continuity_requirement="关键生产线停机超过 2 小时会造成明显损失。",
            custom_attributes={"inventory_value_cny": 6800000, "hazardous_material": True},
        ),
        "hospital_bl_central": EntityProfile(
            entity_id="hospital_bl_central",
            area_id=area_id,
            entity_type=EntityType.HOSPITAL,
            name="碑林中心医院",
            village="中心城区",
            location_hint="急诊楼和后勤出入口",
            resident_count=650,
            current_occupancy=480,
            vulnerability_tags=["critical_service", "patients"],
            preferred_transport_mode=TravelMode.VEHICLE,
            notification_preferences=["dashboard", "phone"],
            emergency_contacts=[ContactProfile(name="医务处王蕾", phone="13800000004", role="医院协调员")],
            continuity_requirement="急诊、ICU 和备用供电不能中断。",
            custom_attributes={"icu_beds": 18, "backup_power_hours": 3},
        ),
        "nursing_home_hpm": EntityProfile(
            entity_id="nursing_home_hpm",
            area_id=area_id,
            entity_type=EntityType.NURSING_HOME,
            name="和平养老院",
            village="和平街坊",
            location_hint="1 至 3 层护理区",
            resident_count=140,
            current_occupancy=128,
            vulnerability_tags=["elderly", "bedridden", "medical_support"],
            mobility_constraints=["stretcher", "oxygen_support"],
            preferred_transport_mode=TravelMode.ASSISTED,
            notification_preferences=["dashboard", "phone"],
            emergency_contacts=[ContactProfile(name="院长刘静", phone="13800000005", role="养老院负责人")],
            custom_attributes={"bedridden_count": 22, "oxygen_patients": 9},
        ),
        "metro_nsm_hub": EntityProfile(
            entity_id="metro_nsm_hub",
            area_id=area_id,
            entity_type=EntityType.METRO_STATION,
            name="南门地铁换乘站",
            village="南门片区",
            location_hint="B/C 出入口与下沉广场",
            resident_count=1800,
            current_occupancy=1250,
            vulnerability_tags=["underground", "commuter_peak"],
            preferred_transport_mode=TravelMode.WALK,
            notification_preferences=["dashboard", "broadcast"],
            emergency_contacts=[ContactProfile(name="站务经理周凯", phone="13800000006", role="地铁值班")],
            custom_attributes={"platform_depth_m": 14, "entrance_count": 6},
        ),
        "underground_wyl_mall": EntityProfile(
            entity_id="underground_wyl_mall",
            area_id=area_id,
            entity_type=EntityType.UNDERGROUND_SPACE,
            name="五一路地下商业街",
            village="五一路片区",
            location_hint="地下负一层主通道",
            resident_count=540,
            current_occupancy=410,
            vulnerability_tags=["underground", "complex_egress"],
            preferred_transport_mode=TravelMode.WALK,
            notification_preferences=["dashboard", "broadcast"],
            emergency_contacts=[ContactProfile(name="商场招商主管孙磊", phone="13800000007", role="商业运营")],
            custom_attributes={"exit_count": 5, "basement_levels": 2},
        ),
        "community_jsl_grid": EntityProfile(
            entity_id="community_jsl_grid",
            area_id=area_id,
            entity_type=EntityType.COMMUNITY,
            name="建设里网格 3",
            village="建设里片区",
            location_hint="老旧小区地下室与南侧支路",
            resident_count=2300,
            current_occupancy=2300,
            vulnerability_tags=["low_lying", "basement", "mixed_population"],
            preferred_transport_mode=TravelMode.WALK,
            notification_preferences=["dashboard", "sms"],
            emergency_contacts=[ContactProfile(name="网格长许晴", phone="13800000008", role="社区网格长")],
            custom_attributes={"basement_buildings": 11},
        ),
    }


def operations_manual_snippets() -> list[dict[str, str]]:
    return [
        {
            "title": "低洼老人家庭转移流程",
            "excerpt": "当预估 30 分钟内可能形成巷道积水时，应优先确认老人家庭联系人、转移工具和药品随身包，必要时安排社区和医疗协同转运。",
        },
        {
            "title": "学校停课与接送区疏导规则",
            "excerpt": "学校周边道路积水达到预警阈值后，应同步启动接送区交通疏导、学生点名和监护人通知，避免校门口形成二次拥堵。",
        },
        {
            "title": "工厂库存保护与停工边界",
            "excerpt": "涉及冷链、高价值库存或危化材料的工厂，应先执行库存上移、断电检查和装卸区封控，再评估是否需要临时停工停产。",
        },
        {
            "title": "地下空间先期封控指引",
            "excerpt": "地铁口、地下商场和下沉广场一旦出现倒灌风险，应先关闭低位入口，设置反向引导，并同步广播疏散提示。",
        },
    ]


def _load_external_entity_profiles(area_id: str) -> dict[str, EntityProfile] | None:
    bootstrap_root = Path(os.getenv("FLOOD_BOOTSTRAP_DATA_DIR", Path(__file__).resolve().parent.parent / "bootstrap_data"))
    path = bootstrap_root / f"entity_profiles.{area_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = [EntityProfile.model_validate(item) for item in payload]
    return {item.entity_id: item for item in profiles}
