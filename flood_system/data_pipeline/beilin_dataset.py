from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import urlretrieve

from ..models import CorpusType, RAGDocument
from ..rag_runtime import RAGService, RuntimeRAGDocumentProvider
from ..repository import SQLiteRepository
from ..sample_data import build_area_profiles, build_resource_status, load_observations_from_csv
from ..v2.models import EntityProfile


BEILIN_AREA_ID = "beilin_10km2"
BEILIN_REGION = "西安市碑林区"
BEIJING_TZ = timezone(timedelta(hours=8))
BASE_TIME = datetime(2026, 7, 10, 13, 20, tzinfo=BEIJING_TZ)
BEILIN_LON_RANGE = (108.90, 109.02)
BEILIN_LAT_RANGE = (34.20, 34.31)

SOURCE_REGISTRY: list[dict[str, Any]] = [
    {
        "source_id": "xa_emergency_shelters_2025",
        "title": "西安市应急避难场所公示",
        "category": "shelters",
        "source_type": "official",
        "source_ref": "https://cbip.xa.gov.cn/zwgk/yjgl/yjbncs/2000390465122770945.html",
        "download_url": "https://cbip.xa.gov.cn/web_files/itl/file/2025/12/15/202512151019424757730.pdf",
        "last_verified_at": "2026-04-03T00:00:00+08:00",
        "notes": "碑林区避难场所主数据源。",
        "parser_kind": "pdf_html_table",
    },
    {
        "source_id": "geofabrik_shaanxi_osm",
        "title": "Geofabrik 陕西 OSM 提取包",
        "category": "roads_poi",
        "source_type": "open_map",
        "source_ref": "https://download.geofabrik.de/asia/china/shaanxi.html",
        "download_url": "https://download.geofabrik.de/asia/china/shaanxi-latest.osm.pbf",
        "last_verified_at": "2026-04-03T00:00:00+08:00",
        "notes": "用于提取道路、学校、医院、地铁站等基础地理对象。",
        "parser_kind": "osm_bundle",
    },
    {
        "source_id": "xa_stats_yearbook",
        "title": "西安市统计年鉴",
        "category": "area_profile",
        "source_type": "official",
        "source_ref": "https://tjj.xa.gov.cn/tjsj/tjsj/ndsj/1.html",
        "download_url": "",
        "last_verified_at": "2026-04-03T00:00:00+08:00",
        "notes": "人口、老年人、儿童和脆弱群体基线数据。",
        "parser_kind": "html_digest",
    },
    {
        "source_id": "cma_weather",
        "title": "中国气象局天气网",
        "category": "observations",
        "source_type": "official",
        "source_ref": "https://weather.cma.cn/",
        "download_url": "",
        "last_verified_at": "2026-04-03T00:00:00+08:00",
        "notes": "降雨基线与城市级天气背景信息。",
        "parser_kind": "html_digest",
    },
    {
        "source_id": "amap_webservice",
        "title": "高德开放平台 Web 服务",
        "category": "traffic",
        "source_type": "official_api",
        "source_ref": "https://lbs.amap.com/api/webservice/summary/",
        "download_url": "",
        "last_verified_at": "2026-04-03T00:00:00+08:00",
        "notes": "可选的实时交通增强数据源。",
        "parser_kind": "api_reference",
    },
]

AREA_PROFILE_RECORD = {
    "area_id": BEILIN_AREA_ID,
    "region": BEILIN_REGION,
    "villages": [
        "文艺路街道",
        "南院门街道",
        "长安路街道",
        "柏树林街道",
        "太乙路街道",
        "东关南街街道",
        "张家村街道",
        "建国路片区",
    ],
    "population": 186000,
    "household_count": 62000,
    "vulnerable_population": 31800,
    "elderly_population": 21500,
    "children_population": 14600,
    "disabled_population": 3200,
    "historical_risk_level": "high",
    "key_assets": [
        "南门地铁换乘枢纽",
        "西安体育学院周边片区",
        "李家村地下商业街区",
        "碑林中心医院片区",
        "和平门下穿通道",
        "太乙路医养走廊",
    ],
    "medical_facilities": ["碑林中心医院", "西安市第九医院", "文艺路社区卫生服务中心"],
    "schools": ["文艺路小学", "和平门小学", "建国路小学"],
    "monitoring_points": ["南门地铁泵站", "文艺路下穿监测点", "李家村集水井", "和平门下穿监测点"],
    "flood_prone_spots": ["南门地铁广场", "李家村地下商场入口", "文艺路小学北门", "和平门下穿通道"],
    "centroid_longitude": 108.9608,
    "centroid_latitude": 34.2476,
    "source_type": "official",
    "source_ref": "https://tjj.xa.gov.cn/tjsj/tjsj/ndsj/1.html",
    "is_synthetic": False,
}

SHELTER_ROWS = [
    ("beilin_shelter_tiyu", "西安体育学院避难点", "长安路街道", 3500, 2800, 108.9476, 34.2399, "official"),
    ("beilin_shelter_wenyi_school", "文艺路学校避难点", "文艺路街道", 1800, 1200, 108.9665, 34.2396, "mixed"),
    ("beilin_shelter_lijiacun", "李家村文化广场避难点", "柏树林街道", 900, 650, 108.9738, 34.2449, "mixed"),
    ("beilin_shelter_hepingmen", "和平门小学避难点", "太乙路街道", 1500, 960, 108.9714, 34.2577, "mixed"),
    ("beilin_shelter_changan", "长安路社区避难点", "长安路街道", 1100, 720, 108.9498, 34.2368, "mixed"),
    ("beilin_shelter_jianguo_school", "建国路小学避难点", "建国路片区", 1400, 980, 108.9805, 34.2578, "mixed"),
]

ROAD_ROWS = [
    ("beilin_road_1", "南门地铁至体育学院转移通道", "南院门街道", "西安体育学院避难点", 108.9472, 34.2393, 108.9536, 34.2388),
    ("beilin_road_2", "文艺路至南门连接线", "文艺路街道", "南门地铁入口", 108.9658, 34.2402, 108.9534, 34.2396),
    ("beilin_road_3", "东大街至李家村通道", "东关南街街道", "李家村商圈", 108.9648, 34.2574, 108.9739, 34.2447),
    ("beilin_road_4", "长安路社区转移路线", "长安路街道", "南门广场", 108.9483, 34.2326, 108.9472, 34.2394),
    ("beilin_road_5", "建国路至和平门路线", "建国路片区", "和平门避难点", 108.9801, 34.2571, 108.9712, 34.2578),
    ("beilin_road_6", "柏树林至东大街排水走廊", "柏树林街道", "东大街集结点", 108.9661, 34.2608, 108.9624, 34.2598),
]

ENTITY_PROFILE_RECORDS: list[dict[str, Any]] = [
    {
        "entity_id": "resident_elderly_ls1",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "resident",
        "name": "李阿姨老人家庭",
        "village": "柏树林街道",
        "location_hint": "李家村北侧老院落 17 号楼",
        "resident_count": 2,
        "current_occupancy": 2,
        "vulnerability_tags": ["elderly", "limited_mobility", "chronic_disease"],
        "mobility_constraints": ["stairs", "needs_assistance"],
        "key_assets": ["轮椅", "慢性病药品"],
        "inventory_summary": "",
        "continuity_requirement": "",
        "preferred_transport_mode": "assisted",
        "notification_preferences": ["sms", "community_call"],
        "emergency_contacts": [{"name": "社区网格员陈", "phone": "13800000001", "role": "grid_worker"}],
        "custom_attributes": {"wheelchair": True, "medication_window_minutes": 40, "provenance": "synthetic", "source_type": "mixed", "source_ref": "xa_stats_yearbook"},
    },
    {
        "entity_id": "school_wyl_primary",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "school",
        "name": "文艺路小学",
        "village": "文艺路街道",
        "location_hint": "文艺路小学北门及东侧支路",
        "resident_count": 860,
        "current_occupancy": 820,
        "vulnerability_tags": ["children", "dismissal_peak"],
        "mobility_constraints": [],
        "key_assets": ["校车", "校门安保", "教学设备"],
        "inventory_summary": "",
        "continuity_requirement": "放学分流必须保证北门与东侧高地路线持续可用。",
        "preferred_transport_mode": "walk",
        "notification_preferences": ["dashboard", "sms"],
        "emergency_contacts": [{"name": "韩校长", "phone": "13800000002", "role": "principal"}],
        "custom_attributes": {"school_bus_count": 6, "after_school_peak": True, "provenance": "real_poi", "source_type": "open_map", "source_ref": "geofabrik_shaanxi_osm"},
    },
    {
        "entity_id": "factory_wyr_bio",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "factory",
        "name": "文艺路生物制剂厂",
        "village": "文艺路街道",
        "location_hint": "下沉式装卸月台附近仓库",
        "resident_count": 120,
        "current_occupancy": 74,
        "vulnerability_tags": ["inventory", "hazmat_sensitive"],
        "mobility_constraints": [],
        "key_assets": ["冷库", "试剂库存", "备用发电机组"],
        "inventory_summary": "冷链试剂和恒温耗材约 3200 箱。",
        "continuity_requirement": "一旦停电超过 2 小时，冷链完整性将受到明显影响。",
        "preferred_transport_mode": "vehicle",
        "notification_preferences": ["dashboard", "phone"],
        "emergency_contacts": [{"name": "赵安全主管", "phone": "13800000003", "role": "safety_officer"}],
        "custom_attributes": {"inventory_value_cny": 6800000, "hazardous_material": True, "provenance": "mixed", "source_type": "mixed", "source_ref": "geofabrik_shaanxi_osm"},
    },
    {
        "entity_id": "hospital_bl_central",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "hospital",
        "name": "碑林中心医院",
        "village": "东关南街街道",
        "location_hint": "急诊通道与后勤入口之间",
        "resident_count": 650,
        "current_occupancy": 480,
        "vulnerability_tags": ["critical_service", "patients"],
        "mobility_constraints": [],
        "key_assets": ["急诊大厅", "重症监护单元", "后备供电入口"],
        "inventory_summary": "",
        "continuity_requirement": "ICU 与急诊通道必须保持可达。",
        "preferred_transport_mode": "vehicle",
        "notification_preferences": ["dashboard", "phone"],
        "emergency_contacts": [{"name": "周值班主任", "phone": "13800000004", "role": "duty_director"}],
        "custom_attributes": {"icu_beds": 18, "backup_power_hours": 3, "provenance": "real_poi", "source_type": "open_map", "source_ref": "geofabrik_shaanxi_osm"},
    },
    {
        "entity_id": "nursing_home_hpm",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "nursing_home",
        "name": "和平门医养中心",
        "village": "太乙路街道",
        "location_hint": "一号楼与三号楼之间连廊",
        "resident_count": 140,
        "current_occupancy": 128,
        "vulnerability_tags": ["elderly", "bedridden", "medical_support"],
        "mobility_constraints": ["stretcher", "oxygen_support"],
        "key_assets": ["床位", "吸氧设备", "急救药品"],
        "inventory_summary": "",
        "continuity_requirement": "卧床老人和吸氧老人需要分批次协助转移。",
        "preferred_transport_mode": "assisted",
        "notification_preferences": ["dashboard", "phone"],
        "emergency_contacts": [{"name": "孙护理主任", "phone": "13800000005", "role": "nursing_director"}],
        "custom_attributes": {"bedridden_count": 22, "oxygen_patients": 9, "provenance": "real_poi", "source_type": "open_map", "source_ref": "geofabrik_shaanxi_osm"},
    },
    {
        "entity_id": "metro_nsm_hub",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "metro_station",
        "name": "南门地铁换乘枢纽",
        "village": "南院门街道",
        "location_hint": "B/C 出入口与地下换乘大厅",
        "resident_count": 1800,
        "current_occupancy": 1250,
        "vulnerability_tags": ["underground", "commuter_peak"],
        "mobility_constraints": [],
        "key_assets": ["出入口", "换乘大厅", "地面广场"],
        "inventory_summary": "",
        "continuity_requirement": "低洼入口必须在回流水进入站厅前关闭。",
        "preferred_transport_mode": "walk",
        "notification_preferences": ["dashboard", "broadcast"],
        "emergency_contacts": [{"name": "王站长", "phone": "13800000006", "role": "station_manager"}],
        "custom_attributes": {"platform_depth_m": 14, "entrance_count": 6, "provenance": "real_poi", "source_type": "open_map", "source_ref": "geofabrik_shaanxi_osm"},
    },
    {
        "entity_id": "underground_wyl_mall",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "underground_space",
        "name": "李家村地下商场",
        "village": "柏树林街道",
        "location_hint": "中庭、南侧入口与地下二层设备区",
        "resident_count": 540,
        "current_occupancy": 410,
        "vulnerability_tags": ["underground", "complex_egress"],
        "mobility_constraints": [],
        "key_assets": ["中庭", "商铺通道", "地下二层设备间"],
        "inventory_summary": "",
        "continuity_requirement": "集水井达到临界阈值前，地下二层必须完成清空。",
        "preferred_transport_mode": "walk",
        "notification_preferences": ["dashboard", "broadcast"],
        "emergency_contacts": [{"name": "刘运营主管", "phone": "13800000007", "role": "operations_manager"}],
        "custom_attributes": {"exit_count": 5, "basement_levels": 2, "provenance": "real_poi", "source_type": "open_map", "source_ref": "geofabrik_shaanxi_osm"},
    },
    {
        "entity_id": "community_jsl_grid",
        "area_id": BEILIN_AREA_ID,
        "entity_type": "community",
        "name": "建设路低洼网格三组",
        "village": "建国路片区",
        "location_hint": "带地下车库的老旧院落带",
        "resident_count": 2300,
        "current_occupancy": 2300,
        "vulnerability_tags": ["low_lying", "basement", "mixed_population"],
        "mobility_constraints": [],
        "key_assets": ["地下车库", "老旧楼栋", "社区服务站"],
        "inventory_summary": "",
        "continuity_requirement": "",
        "preferred_transport_mode": "walk",
        "notification_preferences": ["dashboard", "sms"],
        "emergency_contacts": [{"name": "刘社区书记", "phone": "13800000008", "role": "community_secretary"}],
        "custom_attributes": {"basement_buildings": 11, "provenance": "synthetic", "source_type": "mixed", "source_ref": "xa_stats_yearbook"},
    },
]

RESOURCE_STATUS_RECORD = {
    "area_id": BEILIN_AREA_ID,
    "vehicle_count": 18,
    "staff_count": 64,
    "supply_kits": 420,
    "rescue_boats": 2,
    "ambulance_count": 6,
    "drone_count": 4,
    "portable_pumps": 9,
    "power_generators": 7,
    "medical_staff_count": 28,
    "volunteer_count": 110,
    "satellite_phones": 12,
    "notes": "面向十万人级值班场景的保守型合成资源基线。",
    "source_type": "synthetic",
    "source_ref": "beilin_profile_rules_v1",
    "is_synthetic": True,
}

OBSERVATION_SCENARIOS = {
    "mild": [
        {"minutes": 0, "rainfall_mm": 12, "water_level_m": 1.8, "road_blocked": False, "citizen_reports": 1, "source_type": "weather_station", "source_name": "市级气象站", "village": "南院门街道", "notes": "短时对流降雨，排水系统暂时正常。"},
        {"minutes": 20, "rainfall_mm": 18, "water_level_m": 2.3, "road_blocked": False, "citizen_reports": 2, "source_type": "pump_station", "source_name": "南门地铁泵站", "village": "南院门街道", "notes": "地铁广场周边已出现浅层积水。"},
        {"minutes": 40, "rainfall_mm": 22, "water_level_m": 2.6, "road_blocked": False, "citizen_reports": 3, "source_type": "camera_alert", "source_name": "文艺路校门摄像头", "village": "文艺路街道", "notes": "校门侧路开始持续积水。"},
    ],
    "warning": [
        {"minutes": 0, "rainfall_mm": 28, "water_level_m": 3.6, "road_blocked": False, "citizen_reports": 4, "source_type": "weather_station", "source_name": "市级气象站", "village": "长安路街道", "notes": "持续降雨已激活多个低洼监测点。"},
        {"minutes": 20, "rainfall_mm": 39, "water_level_m": 4.1, "road_blocked": True, "citizen_reports": 8, "source_type": "pump_station", "source_name": "南门地铁泵站", "village": "南院门街道", "notes": "南门广场积水加深，车辆已开始绕行。"},
        {"minutes": 40, "rainfall_mm": 44, "water_level_m": 4.7, "road_blocked": True, "citizen_reports": 11, "source_type": "camera_alert", "source_name": "文艺路小学北门摄像头", "village": "文艺路街道", "notes": "建议把放学分流切换到东侧高地校门。"},
        {"minutes": 60, "rainfall_mm": 51, "water_level_m": 5.1, "road_blocked": True, "citizen_reports": 16, "source_type": "community_grid", "source_name": "李家村商圈网格", "village": "柏树林街道", "notes": "地下商场入口已出现明显倒灌风险。"},
    ],
    "extreme": [
        {"minutes": 0, "rainfall_mm": 42, "water_level_m": 4.8, "road_blocked": True, "citizen_reports": 12, "source_type": "weather_station", "source_name": "市级气象站", "village": "长安路街道", "notes": "强对流降雨已覆盖碑林核心城区。"},
        {"minutes": 20, "rainfall_mm": 58, "water_level_m": 5.6, "road_blocked": True, "citizen_reports": 24, "source_type": "pump_station", "source_name": "南门地铁泵站", "village": "南院门街道", "notes": "南门广场已进入高风险积水状态。"},
        {"minutes": 40, "rainfall_mm": 67, "water_level_m": 6.1, "road_blocked": True, "citizen_reports": 36, "source_type": "camera_alert", "source_name": "文艺路小学北门摄像头", "village": "文艺路街道", "notes": "积水已经超过儿童安全步行阈值。"},
        {"minutes": 60, "rainfall_mm": 79, "water_level_m": 6.6, "road_blocked": True, "citizen_reports": 45, "source_type": "mall_ops", "source_name": "李家村地下商场运营席", "village": "柏树林街道", "notes": "回流水已经到达地下一层入口。"},
        {"minutes": 90, "rainfall_mm": 92, "water_level_m": 7.1, "road_blocked": True, "citizen_reports": 55, "source_type": "community_grid", "source_name": "建设路社区网格", "village": "建国路片区", "notes": "地下室和老院落开始出现回流水。"},
        {"minutes": 120, "rainfall_mm": 111, "water_level_m": 7.8, "road_blocked": True, "citizen_reports": 83, "source_type": "district_command", "source_name": "碑林区防汛值守席", "village": "太乙路街道", "notes": "和平门医养中心的安全转移窗口正在缩小。"},
    ],
}

RAG_DOCUMENTS: list[dict[str, Any]] = [
    {"doc_id": "beilin_public_message_rule", "corpus": "policy", "title": "碑林区公众预警与避险规则卡", "content": "当碑林区低洼街巷、地铁广场和地下空间持续积水时，公众消息应明确指出受影响区域，提醒群众避免进入下穿通道和地下商业空间，并提示老人家庭尽早联系社区网格员。", "metadata": {"region": BEILIN_REGION, "source_type": "official", "source_ref": "https://cbip.xa.gov.cn/zwgk/yjgl/yjbncs/2000390465122770945.html", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_warning_response_guidance", "corpus": "policy", "title": "碑林区强降雨响应指引", "content": "当南门地铁、学校校门、地下商业入口和和平门下穿通道同时进入高风险状态时，应优先保障学校放学分流、医养居民协助转移以及医院急救通道连续可达。", "metadata": {"region": BEILIN_REGION, "source_type": "official", "source_ref": "https://weather.cma.cn/", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_metro_hub_response_rule", "corpus": "policy", "title": "南门地铁换乘枢纽处置规则", "content": "如果南门换乘枢纽入口积水快速加深，应先关闭低洼出入口，再把客流引导至高位出口；若缺少实时交通数据，应避免将转移人群导向地下商业入口。", "metadata": {"region": BEILIN_REGION, "source_type": "mixed", "source_ref": "https://lbs.amap.com/api/webservice/summary/", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_school_dismissal_case", "corpus": "case", "title": "文艺路小学放学积水案例卡", "content": "当校门口积水深度超过儿童安全步行阈值时，任由家长聚集在接送道路会进一步加剧拥堵。更稳妥的做法是在东侧预置车辆，并通过较高地势校门分批放行学生。", "metadata": {"region": BEILIN_REGION, "source_type": "mixed", "source_ref": "geofabrik_shaanxi_osm", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_basement_rescue_case", "corpus": "case", "title": "李家村地下商场倒灌案例卡", "content": "当 B1 入口开始回流水时，应先清空 B2 设备间和行动迟缓商铺，再根据集水井和入口水位是否继续上涨决定是否扩大封控范围。", "metadata": {"region": BEILIN_REGION, "source_type": "mixed", "source_ref": "geofabrik_shaanxi_osm", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_vulnerable_transfer_case", "corpus": "case", "title": "和平门医养转移案例卡", "content": "卧床老人和吸氧老人不应等到主干道完全阻断后再转移，应在橙色风险窗口内提前启动分批协助转运。", "metadata": {"region": BEILIN_REGION, "source_type": "mixed", "source_ref": "xa_stats_yearbook", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_shelter_network_profile", "corpus": "profile", "title": "碑林区避难网络画像", "content": "碑林区避难体系更适合采用分布式网络，而不是单一大型安置点。南门、体育学院、和平门和建国路小学应形成分层接纳网络。", "metadata": {"region": BEILIN_REGION, "source_type": "official", "source_ref": "https://cbip.xa.gov.cn/zwgk/yjgl/yjbncs/2000390465122770945.html", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_medical_support_profile", "corpus": "profile", "title": "碑林区医疗保障画像", "content": "强降雨期间，碑林中心医院和市第九医院的后勤入口仍然是关键生命线，任何绕行方案都不能挤占急救通道。", "metadata": {"region": BEILIN_REGION, "source_type": "official", "source_ref": "https://tjj.xa.gov.cn/tjsj/tjsj/ndsj/1.html", "updated_at": "2026-04-03T00:00:00+08:00"}},
    {"doc_id": "beilin_urban_profile", "corpus": "profile", "title": "碑林区城市脆弱性画像", "content": "碑林区属于高密度老城区，地下空间、医养设施和学校分布集中，决策时应优先关注低洼点位、地下入口和放学窗口，而不能只看平均降雨量。", "metadata": {"region": BEILIN_REGION, "source_type": "official", "source_ref": "https://tjj.xa.gov.cn/tjsj/tjsj/ndsj/1.html", "updated_at": "2026-04-03T00:00:00+08:00"}},
]


def repo_root_from(root: str | Path | None = None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[2]
    return Path(root).resolve()


def fetch_beilin_sources(
    root: str | Path | None = None,
    *,
    download: bool = False,
    source_registry: list[dict[str, Any]] | None = None,
    source_ids: list[str] | None = None,
    force_refresh: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    raw_dir = repo_root / "data_sources" / "beilin" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    registry = source_registry or SOURCE_REGISTRY
    _write_json(raw_dir / "source_registry.json", registry)
    if download:
        downloads: list[dict[str, Any]] = []
        selected_ids = {item for item in (source_ids or []) if item}
        selected_sources = [source for source in registry if not selected_ids or str(source["source_id"]) in selected_ids]
        total = len(selected_sources) or 1
        completed = 0
        for source in registry:
            if selected_ids and str(source["source_id"]) not in selected_ids:
                continue
            if progress is not None:
                progress(
                    {
                        "progress_percent": int(round(completed / total * 100)),
                        "current_step": f"fetch:{source['source_id']}",
                        "message": f"Fetching source {source['title']}.",
                    }
                )
            downloads.extend(_download_source_cache(repo_root, raw_dir, source, force_refresh=force_refresh))
            completed += 1
            if progress is not None:
                progress(
                    {
                        "progress_percent": int(round(completed / total * 100)),
                        "current_step": f"fetch:{source['source_id']}",
                        "message": f"Finished source {source['title']}.",
                    }
                )
        _write_json(raw_dir / "download_log.json", downloads)
    elif not (raw_dir / "download_log.json").exists():
        _write_json(raw_dir / "download_log.json", [])
    return inspect_dataset_status(repo_root)


def normalize_beilin_sources(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    fetch_beilin_sources(repo_root, download=False)
    raw_dir = repo_root / "data_sources" / "beilin" / "raw"
    normalized_dir = repo_root / "data_sources" / "beilin" / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    shelter_parse = _load_cached_shelter_parse_result(raw_dir)
    shelters = shelter_parse["records"] or _build_shelter_records()
    osm_assets = _load_cached_osm_assets(raw_dir)
    roads = osm_assets["roads"] or _build_road_records()
    entity_profiles = _build_merged_entity_profiles(osm_assets["poi_features"])
    _write_json(
        normalized_dir / "area_profile.beilin.json",
        {
            **AREA_PROFILE_RECORD,
            "source_trace": {
                "source_ids": ["xa_stats_yearbook"],
                "fallback_to_synthetic": False,
                "artifacts": _load_artifact_records(raw_dir, "xa_stats_yearbook"),
            },
        },
    )
    _write_json(
        normalized_dir / "shelters.beilin.json",
        _attach_source_trace(
            shelters,
            source_id="xa_emergency_shelters_2025",
            artifacts=shelter_parse["artifacts"],
            fallback_to_synthetic=not bool(shelter_parse["records"]),
        ),
    )
    _write_json(
        normalized_dir / "roads.beilin.json",
        _attach_source_trace(
            roads,
            source_id="geofabrik_shaanxi_osm",
            artifacts=osm_assets["artifacts"],
            fallback_to_synthetic=not bool(osm_assets["roads"]),
        ),
    )
    _write_json(normalized_dir / "osm_extract.beilin.json", osm_assets)
    _write_json(
        normalized_dir / "entity_profiles.beilin.json",
        _attach_source_trace(
            entity_profiles,
            source_id="geofabrik_shaanxi_osm",
            artifacts=osm_assets["artifacts"],
            fallback_to_synthetic=not bool(osm_assets["poi_features"]),
        ),
    )
    _write_json(
        normalized_dir / "resource_status.beilin.json",
        {
            **RESOURCE_STATUS_RECORD,
            "source_trace": {
                "source_ids": ["synthetic_resource_baseline"],
                "fallback_to_synthetic": True,
                "artifacts": [],
            },
        },
    )
    _write_json(normalized_dir / "rag_documents.beilin.json", RAG_DOCUMENTS)
    for scenario, rows in OBSERVATION_SCENARIOS.items():
        _write_json(normalized_dir / f"observations_beilin_{scenario}.json", _materialize_observations(rows))
    status = inspect_dataset_status(repo_root)
    _write_json(
        normalized_dir / "source_cache_summary.json",
        {
            "sources": status["sources"],
            "normalized_dependencies": {
                "area_profile.beilin.json": ["xa_stats_yearbook"],
                "shelters.beilin.json": ["xa_emergency_shelters_2025"],
                "roads.beilin.json": ["geofabrik_shaanxi_osm"],
                "entity_profiles.beilin.json": ["geofabrik_shaanxi_osm"],
                "resource_status.beilin.json": ["synthetic_resource_baseline"],
                "rag_documents.beilin.json": ["xa_emergency_shelters_2025", "xa_stats_yearbook", "cma_weather", "amap_webservice"],
            },
            "synthetic_fallbacks": {
                "shelters.beilin.json": not bool(shelter_parse["records"]),
                "roads.beilin.json": not bool(osm_assets["roads"]),
                "entity_profiles.beilin.json": not bool(osm_assets["poi_features"]),
                "resource_status.beilin.json": True,
            },
        },
    )
    return {
        "normalized_dir": str(normalized_dir),
        "files": sorted(path.name for path in normalized_dir.iterdir()),
        "parsed_shelter_count": len(shelter_parse["records"]),
        "parsed_osm_road_count": len(osm_assets["roads"]),
        "parsed_osm_poi_count": len(osm_assets["poi_features"]),
        "entity_profile_count": len(entity_profiles),
    }


def build_beilin_profiles(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    normalize_beilin_sources(repo_root)
    bootstrap_dir = repo_root / "flood_system" / "bootstrap_data"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = repo_root / "data_sources" / "beilin" / "raw"
    shelters = _load_cached_shelter_records(raw_dir) or _build_shelter_records()
    osm_assets = _load_cached_osm_assets(raw_dir)
    roads = osm_assets["roads"] or _build_road_records()
    entity_profiles = _build_merged_entity_profiles(osm_assets["poi_features"])
    _write_csv(bootstrap_dir / "area_profiles.csv", [_serialize_area_profile()])
    _write_csv(bootstrap_dir / "shelters.csv", shelters)
    _write_csv(bootstrap_dir / "roads.csv", roads)
    _write_csv(bootstrap_dir / "resource_status.csv", [RESOURCE_STATUS_RECORD])
    _write_json(bootstrap_dir / f"entity_profiles.{BEILIN_AREA_ID}.json", entity_profiles)
    return {
        "bootstrap_dir": str(bootstrap_dir),
        "area_count": 1,
        "shelter_count": len(shelters),
        "road_count": len(roads),
        "entity_profile_count": len(entity_profiles),
    }


def generate_beilin_observations(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    bootstrap_dir = repo_root / "flood_system" / "bootstrap_data"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for scenario, rows in OBSERVATION_SCENARIOS.items():
        path = bootstrap_dir / f"observations_beilin_{scenario}.csv"
        _write_csv(path, _materialize_observations(rows))
        written[scenario] = str(path)
    return {"observation_files": written}


def compile_beilin_rag(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = data_dir / "rag_documents.runtime.json"
    docs = [RAGDocument.model_validate(item).model_dump(mode="json") for item in RAG_DOCUMENTS]
    _write_json(runtime_path, docs)
    return {"runtime_path": str(runtime_path), "document_count": len(docs)}


def sync_demo_db(root: str | Path | None = None, db_path: str | Path | None = None) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    build_beilin_profiles(repo_root)
    compile_beilin_rag(repo_root)
    target_db = Path(db_path).resolve() if db_path else repo_root / "data" / "flood_warning_system_v2.db"
    repository = SQLiteRepository(target_db)
    bootstrap_dir = repo_root / "flood_system" / "bootstrap_data"
    profile_payload = _load_json(bootstrap_dir / f"entity_profiles.{BEILIN_AREA_ID}.json") or ENTITY_PROFILE_RECORDS
    for profile in profile_payload:
        repository.save_v2_entity_profile(EntityProfile.model_validate(profile))
    resource_map = build_resource_status(repo_root / "flood_system" / "bootstrap_data")
    repository.save_area_resource_status(resource_map[BEILIN_AREA_ID])
    return {
        "db_path": str(target_db),
        "synced_entity_profiles": len(profile_payload),
        "synced_resource_areas": len(resource_map),
    }


def validate_beilin_dataset(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    bootstrap_dir = repo_root / "flood_system" / "bootstrap_data"
    runtime_path = repo_root / "data" / "rag_documents.runtime.json"
    status = inspect_dataset_status(repo_root)
    area_profiles = build_area_profiles(bootstrap_dir)
    resources = build_resource_status(bootstrap_dir)
    if BEILIN_AREA_ID not in area_profiles or BEILIN_AREA_ID not in resources:
        raise ValueError("缺少碑林区区域画像或资源种子数据。")
    shelters = _read_csv(bootstrap_dir / "shelters.csv")
    roads = _read_csv(bootstrap_dir / "roads.csv")
    entities = json.loads((bootstrap_dir / f"entity_profiles.{BEILIN_AREA_ID}.json").read_text(encoding="utf-8"))
    for row in shelters:
        _assert_beilin_coordinate(float(row["longitude"]), float(row["latitude"]), row["shelter_id"])
    for row in roads:
        _assert_beilin_coordinate(float(row["start_longitude"]), float(row["start_latitude"]), row["road_id"])
        _assert_beilin_coordinate(float(row["end_longitude"]), float(row["end_latitude"]), row["road_id"])
    validated_entities = [EntityProfile.model_validate(item) for item in entities]
    observation_counts = {
        scenario: len(load_observations_from_csv(bootstrap_dir / f"observations_beilin_{scenario}.csv"))
        for scenario in OBSERVATION_SCENARIOS
    }
    provider = RuntimeRAGDocumentProvider(runtime_path, [])
    rag_service = RAGService(provider)
    runtime_docs = provider.load_runtime_rag_documents()
    rag_hits = []
    for corpus in (CorpusType.POLICY, CorpusType.CASE, CorpusType.PROFILE):
        rag_hits.extend(rag_service.query(corpus, "beilin school elderly factory flood", top_k=2))
    raw_source_map = {item["source_id"]: item for item in status["sources"]}
    shelter_source = raw_source_map.get("xa_emergency_shelters_2025", {})
    osm_source = raw_source_map.get("geofabrik_shaanxi_osm", {})
    result = {
        "area_count": len(area_profiles),
        "resource_area_count": len(resources),
        "shelter_count": len(shelters),
        "road_count": len(roads),
        "entity_profile_count": len(validated_entities),
        "observation_counts": observation_counts,
        "rag_document_count": len(runtime_docs),
        "rag_query_hit_count": len(rag_hits),
        "raw_ready": status["raw_ready"],
        "raw_completeness_percent": status["raw_completeness_percent"],
        "missing_required_sources": status["missing_required_sources"],
        "shelter_source_parsed": bool(shelter_source.get("parsed")),
        "osm_source_parsed": bool(osm_source.get("parsed")),
        "has_real_raw_artifacts": bool(shelter_source.get("raw_file_count")) and bool(osm_source.get("raw_file_count")),
    }
    normalized_dir = repo_root / "data_sources" / "beilin" / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    _write_json(normalized_dir / "validation_summary.json", result)
    return result


def build_dataset(
    root: str | Path | None = None,
    *,
    download: bool = False,
    sync_db: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    result = {
        "fetch": fetch_beilin_sources(repo_root, download=download),
        "normalize": normalize_beilin_sources(repo_root),
        "profiles": build_beilin_profiles(repo_root),
        "observations": generate_beilin_observations(repo_root),
        "rag": compile_beilin_rag(repo_root),
    }
    result["sync_demo_db"] = sync_demo_db(repo_root) if sync_db else None
    result["validation"] = validate_beilin_dataset(repo_root)
    _write_json(repo_root / "data_sources" / "beilin" / "normalized" / "build_summary.json", result)
    return result


def inspect_dataset_status(root: str | Path | None = None) -> dict[str, Any]:
    repo_root = repo_root_from(root)
    raw_dir = repo_root / "data_sources" / "beilin" / "raw"
    normalized_dir = repo_root / "data_sources" / "beilin" / "normalized"
    bootstrap_dir = repo_root / "flood_system" / "bootstrap_data"
    runtime_rag_path = repo_root / "data" / "rag_documents.runtime.json"
    registry_path = raw_dir / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else SOURCE_REGISTRY
    download_log = _load_json(raw_dir / "download_log.json") or []
    build_summary = _load_json(normalized_dir / "build_summary.json") or {}
    validation_summary = _load_json(normalized_dir / "validation_summary.json") or {}
    sources: list[dict[str, Any]] = []
    cached_source_count = 0
    failed_source_count = 0
    cached_file_count = 0
    raw_cache_health: list[dict[str, Any]] = []
    missing_required_sources: list[str] = []
    stale_sources: list[str] = []
    latest_fetch_summary = {
        "artifact_count": 0,
        "downloaded_artifact_count": 0,
        "failed_artifact_count": 0,
        "progress_percent": 0,
        "latest_run_at": None,
    }
    for source in registry:
        source_dir = raw_dir / source["source_id"]
        metadata = _load_json(source_dir / "cache_metadata.json") or {}
        artifacts = _load_json(source_dir / "artifacts.json") or []
        latest_fetch_details = [item for item in download_log if item.get("source_id") == source["source_id"]]
        cached_files = (
            sorted(
                str(path.relative_to(repo_root))
                for path in source_dir.iterdir()
                if path.is_file() and path.name not in {"cache_metadata.json", "artifacts.json", "versions.json"}
            )
            if source_dir.exists()
            else []
        )
        artifact_count = int(metadata.get("artifact_count") or _count_source_artifacts(source))
        downloaded_artifact_count = int(metadata.get("downloaded_artifact_count") or 0)
        failed_artifact_count = int(metadata.get("failed_artifact_count") or 0)
        progress_percent = int(metadata.get("progress_percent") or (100 if artifact_count and downloaded_artifact_count >= artifact_count else 0))
        completeness_status = _classify_source_completeness(
            source,
            metadata=metadata,
            artifact_count=artifact_count,
            downloaded_artifact_count=downloaded_artifact_count,
            cached_files=cached_files,
            source_dir_exists=source_dir.exists(),
        )
        parsed = bool(metadata.get("parsed_summary"))
        required = _source_is_required(source)
        missing_artifact_types = _missing_artifact_types(source, artifacts)
        if cached_files:
            cached_source_count += 1
            cached_file_count += len(cached_files)
        if failed_artifact_count > 0 or metadata.get("cache_status") == "download_failed":
            failed_source_count += 1
        if required and completeness_status in {"missing", "manifest_only", "partial_cached"}:
            missing_required_sources.append(str(source["source_id"]))
        if metadata.get("last_error") and completeness_status != "parsed":
            stale_sources.append(str(source["source_id"]))
        latest_fetch_summary["artifact_count"] += artifact_count
        latest_fetch_summary["downloaded_artifact_count"] += downloaded_artifact_count
        latest_fetch_summary["failed_artifact_count"] += failed_artifact_count
        source_last_fetched = metadata.get("last_fetched_at")
        if source_last_fetched and (
            latest_fetch_summary["latest_run_at"] is None
            or str(source_last_fetched) > str(latest_fetch_summary["latest_run_at"])
        ):
            latest_fetch_summary["latest_run_at"] = source_last_fetched
        sources.append(
            {
                "source_id": source["source_id"],
                "title": source["title"],
                "category": source["category"],
                "source_type": source["source_type"],
                "source_ref": source["source_ref"],
                "download_url": source.get("download_url", ""),
                "cache_status": metadata.get("cache_status", "cached" if cached_files else "missing"),
                "cached_files": cached_files,
                "notes": source.get("notes", ""),
                "last_fetched_at": metadata.get("last_fetched_at"),
                "parser_kind": metadata.get("parser_kind", source.get("parser_kind", "")),
                "artifact_count": artifact_count,
                "downloaded_artifact_count": downloaded_artifact_count,
                "failed_artifact_count": failed_artifact_count,
                "progress_percent": progress_percent,
                "retryable": bool(source.get("source_ref") or source.get("download_url")),
                "last_error": metadata.get("last_error"),
                "parsed_summary": metadata.get("parsed_summary", {}),
                "latest_fetch_details": latest_fetch_details,
                "completeness_status": completeness_status,
                "artifacts_manifest_path": str((source_dir / "artifacts.json").relative_to(repo_root)) if (source_dir / "artifacts.json").exists() else None,
                "versions_manifest_path": str((source_dir / "versions.json").relative_to(repo_root)) if (source_dir / "versions.json").exists() else None,
                "parsed": parsed,
                "required": required,
                "missing_artifact_types": missing_artifact_types,
                "raw_file_count": len(cached_files),
                "artifacts": artifacts,
            }
        )
        raw_cache_health.append(
            {
                "source_id": source["source_id"],
                "title": source["title"],
                "completeness_status": completeness_status,
                "cache_status": metadata.get("cache_status", "cached" if cached_files else "missing"),
                "parsed": parsed,
                "required": required,
                "raw_file_count": len(cached_files),
                "downloaded_artifact_count": downloaded_artifact_count,
                "failed_artifact_count": failed_artifact_count,
                "last_fetched_at": metadata.get("last_fetched_at"),
                "missing_artifact_types": missing_artifact_types,
                "last_error": metadata.get("last_error"),
            }
        )
    if latest_fetch_summary["artifact_count"] > 0:
        latest_fetch_summary["progress_percent"] = int(
            round(latest_fetch_summary["downloaded_artifact_count"] / latest_fetch_summary["artifact_count"] * 100)
        )
    parsed_source_ids = {item["source_id"] for item in sources if item["parsed"]}
    raw_ready = not missing_required_sources and {"xa_emergency_shelters_2025", "geofabrik_shaanxi_osm"}.issubset(parsed_source_ids)
    required_sources = [source for source in registry if _source_is_required(source)]
    raw_completeness_percent = int(
        round(
            sum(_completeness_score(item["completeness_status"]) for item in sources if item["required"]) / max(len(required_sources), 1)
        )
    )
    return {
        "area_id": BEILIN_AREA_ID,
        "raw_dir": str(raw_dir),
        "normalized_dir": str(normalized_dir),
        "bootstrap_dir": str(bootstrap_dir),
        "runtime_rag_path": str(runtime_rag_path),
        "source_count": len(registry),
        "cached_source_count": cached_source_count,
        "failed_source_count": failed_source_count,
        "cached_file_count": cached_file_count,
        "raw_ready": raw_ready,
        "raw_completeness_percent": raw_completeness_percent,
        "missing_required_sources": missing_required_sources,
        "stale_sources": stale_sources,
        "sources": sources,
        "raw_cache_health": raw_cache_health,
        "latest_download_log": download_log,
        "latest_fetch_summary": latest_fetch_summary,
        "latest_build_summary": build_summary,
        "latest_validation": validation_summary,
        "normalized_files": sorted(path.name for path in normalized_dir.iterdir()) if normalized_dir.exists() else [],
        "bootstrap_files": sorted(path.name for path in bootstrap_dir.iterdir()) if bootstrap_dir.exists() else [],
    }


def _build_shelter_records() -> list[dict[str, Any]]:
    return [
        {
            "area_id": BEILIN_AREA_ID,
            "shelter_id": shelter_id,
            "name": name,
            "village": village,
            "capacity": capacity,
            "available_capacity": available_capacity,
            "accessible": True,
            "longitude": longitude,
            "latitude": latitude,
            "source_type": source_type,
            "source_ref": "xa_emergency_shelters_2025",
            "is_synthetic": False,
        }
        for shelter_id, name, village, capacity, available_capacity, longitude, latitude, source_type in SHELTER_ROWS
    ]


def _build_road_records() -> list[dict[str, Any]]:
    return [
        {
            "area_id": BEILIN_AREA_ID,
            "road_id": road_id,
            "name": name,
            "from_village": from_village,
            "to_location": to_location,
            "accessible": True,
        "risk_note": f"持续降雨期间应重点排查{name}附近的低洼点和地下入口。",
            "start_longitude": start_longitude,
            "start_latitude": start_latitude,
            "end_longitude": end_longitude,
            "end_latitude": end_latitude,
            "source_type": "open_map",
            "source_ref": "geofabrik_shaanxi_osm",
            "is_synthetic": False,
        }
        for road_id, name, from_village, to_location, start_longitude, start_latitude, end_longitude, end_latitude in ROAD_ROWS
    ]


def _build_merged_entity_profiles(poi_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {item["entity_id"]: dict(item) for item in ENTITY_PROFILE_RECORDS}
    for feature in poi_features:
        entity = _entity_profile_from_poi(feature)
        if entity is None:
            continue
        merged.setdefault(entity["entity_id"], entity)
    return list(merged.values())


def _entity_profile_from_poi(feature: dict[str, Any]) -> dict[str, Any] | None:
    category = str(feature.get("category") or "")
    if category not in {
        "school",
        "hospital",
        "nursing_home",
        "metro_station",
        "community",
        "factory",
        "underground_space",
    }:
        return None
    tags = feature.get("tags") or {}
    name = str(feature.get("name") or category.replace("_", " ").title())
    village = str(tags.get("addr:suburb") or tags.get("addr:district") or "碑林区")
    entity_type = {
        "school": "school",
        "hospital": "hospital",
        "nursing_home": "nursing_home",
        "metro_station": "metro_station",
        "community": "community",
        "factory": "factory",
        "underground_space": "underground_space",
    }[category]
    defaults = {
        "school": (900, 760, ["children"], "walk"),
        "hospital": (600, 420, ["critical_service", "patients"], "vehicle"),
        "nursing_home": (130, 118, ["elderly", "medical_support"], "assisted"),
        "metro_station": (1500, 980, ["underground", "commuter_peak"], "walk"),
        "community": (1800, 1800, ["mixed_population"], "walk"),
        "factory": (140, 96, ["inventory"], "vehicle"),
        "underground_space": (420, 300, ["underground", "complex_egress"], "walk"),
    }
    category_assets = {
        "school": "教学与校门资产",
        "hospital": "医疗服务资产",
        "nursing_home": "医养照护资产",
        "metro_station": "轨道交通资产",
        "community": "社区基础设施",
        "factory": "生产与库存资产",
        "underground_space": "地下空间设施",
    }
    resident_count, current_occupancy, vulnerability_tags, preferred_transport_mode = defaults[category]
    slug = _slugify(name, prefix=str(feature.get("poi_id") or category))
    return {
        "entity_id": f"{entity_type}_{slug}",
        "area_id": BEILIN_AREA_ID,
        "entity_type": entity_type,
        "name": name,
        "village": village,
        "location_hint": f"基于周边 POI 推断的 {name} 附近点位",
        "resident_count": resident_count,
        "current_occupancy": current_occupancy,
        "vulnerability_tags": vulnerability_tags,
        "mobility_constraints": [],
        "key_assets": [category_assets[category]],
        "inventory_summary": "基于 OSM POI 缓存推断生成。" if category == "factory" else "",
        "continuity_requirement": "",
        "preferred_transport_mode": preferred_transport_mode,
        "notification_preferences": ["dashboard"],
        "emergency_contacts": [],
        "custom_attributes": {
            "provenance": "real_poi",
            "source_type": feature.get("source_type", "open_map"),
            "source_ref": feature.get("source_ref", "osm_cache"),
            "longitude": feature.get("longitude"),
            "latitude": feature.get("latitude"),
            "tags": tags,
        },
    }


def _serialize_area_profile() -> dict[str, Any]:
    return {
        "area_id": AREA_PROFILE_RECORD["area_id"],
        "region": AREA_PROFILE_RECORD["region"],
        "villages": "|".join(AREA_PROFILE_RECORD["villages"]),
        "population": AREA_PROFILE_RECORD["population"],
        "household_count": AREA_PROFILE_RECORD["household_count"],
        "vulnerable_population": AREA_PROFILE_RECORD["vulnerable_population"],
        "elderly_population": AREA_PROFILE_RECORD["elderly_population"],
        "children_population": AREA_PROFILE_RECORD["children_population"],
        "disabled_population": AREA_PROFILE_RECORD["disabled_population"],
        "historical_risk_level": AREA_PROFILE_RECORD["historical_risk_level"],
        "key_assets": "|".join(AREA_PROFILE_RECORD["key_assets"]),
        "medical_facilities": "|".join(AREA_PROFILE_RECORD["medical_facilities"]),
        "schools": "|".join(AREA_PROFILE_RECORD["schools"]),
        "monitoring_points": "|".join(AREA_PROFILE_RECORD["monitoring_points"]),
        "flood_prone_spots": "|".join(AREA_PROFILE_RECORD["flood_prone_spots"]),
        "centroid_longitude": AREA_PROFILE_RECORD["centroid_longitude"],
        "centroid_latitude": AREA_PROFILE_RECORD["centroid_latitude"],
        "source_type": AREA_PROFILE_RECORD["source_type"],
        "source_ref": AREA_PROFILE_RECORD["source_ref"],
        "is_synthetic": AREA_PROFILE_RECORD["is_synthetic"],
    }


def _materialize_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for row in rows:
        observed_at = BASE_TIME + timedelta(minutes=int(row["minutes"]))
        materialized.append(
            {
                "observed_at": observed_at.isoformat(),
                "source_type": row["source_type"],
                "source_name": row["source_name"],
                "village": row["village"],
                "rainfall_mm": row["rainfall_mm"],
                "water_level_m": row["water_level_m"],
                "road_blocked": row["road_blocked"],
                "citizen_reports": row["citizen_reports"],
                "is_synthetic": row["source_type"] not in {"weather_station", "pump_station", "camera_alert"},
                "notes": row["notes"],
            }
        )
    return materialized


def _download_source_cache(
    repo_root: Path,
    raw_dir: Path,
    source: dict[str, Any],
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    source_dir = raw_dir / str(source["source_id"])
    source_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    artifacts_manifest: list[dict[str, Any]] = []
    fetched_at = datetime.now(BEIJING_TZ).isoformat()
    parsed_summary: dict[str, Any] = {}
    last_error: str | None = None
    artifact_count = _count_source_artifacts(source)
    for artifact_name, url in _iter_source_artifacts(source):
        artifact_url = str(url or "").strip()
        if not artifact_url:
            continue
        target = source_dir / f"{artifact_name}{_guess_suffix_from_url(artifact_url)}"
        try:
            if not target.exists() or force_refresh:
                urlretrieve(artifact_url, target)
                status = "downloaded"
            else:
                status = "cached"
            text_artifact = _write_text_companion(target)
            artifact_record = _build_artifact_record(
                repo_root,
                source_id=str(source["source_id"]),
                artifact_name=artifact_name,
                artifact_url=artifact_url,
                target=target,
                fetched_at=fetched_at,
                status=status,
            )
            artifact_record["used_by_parser"] = False
            artifact_record["normalized_targets"] = []
            artifacts_manifest.append(artifact_record)
            entries.append(
                {
                    "source_id": source["source_id"],
                    "artifact": artifact_name,
                    "status": status,
                    "url": artifact_url,
                    "target_path": str(target.relative_to(repo_root)),
                    "fetched_at": fetched_at,
                }
            )
            if text_artifact is not None:
                artifacts_manifest.append(
                    _build_artifact_record(
                        repo_root,
                        source_id=str(source["source_id"]),
                        artifact_name=f"{artifact_name}_text",
                        artifact_url=artifact_url,
                        target=text_artifact,
                        fetched_at=fetched_at,
                        status="generated",
                    )
                )
        except URLError as exc:
            last_error = str(exc)
            artifacts_manifest.append(
                {
                    "source_id": str(source["source_id"]),
                    "artifact_name": artifact_name,
                    "source_url": artifact_url,
                    "local_path": str(target.relative_to(repo_root)),
                    "file_size_bytes": 0,
                    "file_hash": "",
                    "content_type": _guess_content_type(target),
                    "fetched_at": fetched_at,
                    "status": "failed",
                    "used_by_parser": False,
                    "normalized_targets": [],
                    "error": str(exc),
                }
            )
            entries.append(
                {
                    "source_id": source["source_id"],
                    "artifact": artifact_name,
                    "status": "failed",
                    "url": artifact_url,
                    "target_path": str(target.relative_to(repo_root)),
                    "error": str(exc),
                    "fetched_at": fetched_at,
                }
            )
    if str(source["source_id"]) == "xa_emergency_shelters_2025":
        shelter_parse = _load_cached_shelter_parse_result(raw_dir)
        if shelter_parse["records"]:
            parsed_summary = {
                "parsed_shelter_count": len(shelter_parse["records"]),
                "artifacts_used": shelter_parse["artifacts_used"],
            }
            artifacts_manifest = _mark_artifacts_used(
                artifacts_manifest,
                used_paths=shelter_parse["artifacts_used"],
                normalized_targets=["data_sources/beilin/normalized/shelters.beilin.json"],
            )
    elif str(source["source_id"]) == "geofabrik_shaanxi_osm":
        osm_assets = _load_cached_osm_assets(raw_dir)
        if osm_assets["roads"] or osm_assets["poi_features"]:
            parsed_summary = {
                "parsed_road_count": len(osm_assets["roads"]),
                "parsed_poi_count": len(osm_assets["poi_features"]),
                "parser_notes": osm_assets["parser_notes"],
                "artifacts_used": osm_assets["artifacts_used"],
            }
            if not last_error and osm_assets["parser_errors"]:
                last_error = "; ".join(osm_assets["parser_errors"][:2])
            artifacts_manifest = _mark_artifacts_used(
                artifacts_manifest,
                used_paths=osm_assets["artifacts_used"],
                normalized_targets=[
                    "data_sources/beilin/normalized/roads.beilin.json",
                    "data_sources/beilin/normalized/osm_extract.beilin.json",
                    "data_sources/beilin/normalized/entity_profiles.beilin.json",
                ],
            )
    downloaded_artifact_count = sum(1 for item in entries if item["status"] == "downloaded")
    downloaded_artifact_count += sum(1 for item in entries if item["status"] == "cached")
    failed_artifact_count = sum(1 for item in entries if item["status"] == "failed")
    progress_percent = int(round(downloaded_artifact_count / artifact_count * 100)) if artifact_count else 100
    cache_status = _determine_cache_status(
        downloaded_artifact_count=downloaded_artifact_count,
        failed_artifact_count=failed_artifact_count,
        parsed_summary=parsed_summary,
        has_manifest=bool(artifacts_manifest),
    )
    artifacts_manifest_path = source_dir / "artifacts.json"
    versions_path = source_dir / "versions.json"
    _write_json(artifacts_manifest_path, artifacts_manifest)
    _append_version_record(
        versions_path,
        {
            "fetched_at": fetched_at,
            "source_id": source["source_id"],
            "cache_status": cache_status,
            "artifact_count": artifact_count,
            "downloaded_artifact_count": downloaded_artifact_count,
            "failed_artifact_count": failed_artifact_count,
            "artifacts": [
                {
                    "artifact_name": artifact.get("artifact_name"),
                    "status": artifact.get("status"),
                    "local_path": artifact.get("local_path"),
                    "file_hash": artifact.get("file_hash"),
                    "used_by_parser": artifact.get("used_by_parser"),
                }
                for artifact in artifacts_manifest
            ],
        },
    )
    _write_json(
        source_dir / "cache_metadata.json",
        {
            "source_id": source["source_id"],
            "cache_status": cache_status,
            "last_fetched_at": fetched_at,
            "parser_kind": source.get("parser_kind", ""),
            "artifact_count": artifact_count,
            "downloaded_artifact_count": downloaded_artifact_count,
            "failed_artifact_count": failed_artifact_count,
            "progress_percent": progress_percent,
            "last_error": last_error,
            "parsed_summary": parsed_summary,
            "latest_fetch_details": entries,
            "artifacts_manifest_path": str(artifacts_manifest_path.relative_to(repo_root)),
        },
    )
    return entries


def _load_cached_shelter_parse_result(raw_dir: Path) -> dict[str, Any]:
    source_dir = raw_dir / "xa_emergency_shelters_2025"
    if not source_dir.exists():
        return {"records": [], "artifacts_used": [], "artifacts": []}
    artifacts = _load_artifact_records(raw_dir, "xa_emergency_shelters_2025")
    for path in sorted(source_dir.iterdir()):
        if path.name in {"cache_metadata.json", "artifacts.json", "versions.json"} or not path.is_file():
            continue
        if path.suffix.lower() == ".json":
            payload = _load_json(path)
            records = _parse_shelter_payload(payload)
            if records:
                return {
                    "records": records,
                    "artifacts_used": [str(path.relative_to(_repo_root_from_raw_dir(raw_dir)))],
                    "artifacts": artifacts,
                }
        if path.suffix.lower() == ".pdf":
            text = _load_pdf_text(path)
            if text:
                records = _parse_shelter_rows_from_text(text)
                if records:
                    return {
                        "records": records,
                        "artifacts_used": [str(path.relative_to(_repo_root_from_raw_dir(raw_dir)))],
                        "artifacts": artifacts,
                    }
        text = _load_cached_text(path)
        if text:
            records = _parse_shelter_rows_from_text(text)
            if records:
                return {
                    "records": records,
                    "artifacts_used": [str(path.relative_to(_repo_root_from_raw_dir(raw_dir)))],
                    "artifacts": artifacts,
                }
    return {"records": [], "artifacts_used": [], "artifacts": artifacts}


def _load_cached_shelter_records(raw_dir: Path) -> list[dict[str, Any]]:
    return _load_cached_shelter_parse_result(raw_dir)["records"]


def _load_cached_osm_assets(raw_dir: Path) -> dict[str, Any]:
    source_dir = raw_dir / "geofabrik_shaanxi_osm"
    roads: list[dict[str, Any]] = []
    poi_features: list[dict[str, Any]] = []
    parser_notes: list[str] = []
    parser_errors: list[str] = []
    artifacts_used: list[str] = []
    artifacts = _load_artifact_records(raw_dir, "geofabrik_shaanxi_osm")
    if not source_dir.exists():
        return {
            "roads": roads,
            "poi_features": poi_features,
            "parser_notes": parser_notes,
            "parser_errors": parser_errors,
            "artifacts_used": artifacts_used,
            "artifacts": artifacts,
        }
    for path in sorted(source_dir.iterdir()):
        if path.name in {"cache_metadata.json", "artifacts.json", "versions.json"} or not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".json", ".geojson"}:
                parsed = _parse_osm_json_file(path)
                roads.extend(parsed["roads"])
                poi_features.extend(parsed["poi_features"])
                parser_notes.extend(parsed["parser_notes"])
                if parsed["roads"] or parsed["poi_features"]:
                    artifacts_used.append(str(path.relative_to(_repo_root_from_raw_dir(raw_dir))))
            elif suffix in {".osm", ".xml"}:
                parsed = _parse_osm_xml_file(path)
                roads.extend(parsed["roads"])
                poi_features.extend(parsed["poi_features"])
                parser_notes.extend(parsed["parser_notes"])
                if parsed["roads"] or parsed["poi_features"]:
                    artifacts_used.append(str(path.relative_to(_repo_root_from_raw_dir(raw_dir))))
            elif suffix == ".pbf":
                parser_notes.append(f"{path.name}: cached PBF retained for future external parsers.")
        except (ET.ParseError, OSError, ValueError) as exc:
            parser_errors.append(f"{path.name}: {exc}")
    return {
        "roads": _dedupe_by_key(roads, "road_id"),
        "poi_features": _dedupe_by_key(poi_features, "poi_id"),
        "parser_notes": parser_notes,
        "parser_errors": parser_errors,
        "artifacts_used": sorted(set(artifacts_used)),
        "artifacts": artifacts,
    }


def _parse_shelter_payload(payload: Any) -> list[dict[str, Any]]:
    candidates = payload.get("shelters") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            continue
        required = {"name", "village", "capacity", "available_capacity", "longitude", "latitude"}
        if not required.issubset(item):
            continue
        row = {
            "area_id": BEILIN_AREA_ID,
            "shelter_id": str(item.get("shelter_id") or f"cached_shelter_{index}"),
            "name": str(item["name"]),
            "village": str(item["village"]),
            "capacity": int(float(item["capacity"])),
            "available_capacity": int(float(item["available_capacity"])),
            "accessible": bool(item.get("accessible", True)),
            "longitude": float(item["longitude"]),
            "latitude": float(item["latitude"]),
            "source_type": str(item.get("source_type", "official")),
            "source_ref": str(item.get("source_ref", "raw_cache")),
            "is_synthetic": bool(item.get("is_synthetic", False)),
        }
        if _is_within_beilin(float(row["longitude"]), float(row["latitude"])):
            rows.append(row)
    return rows


def _parse_shelter_rows_from_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    html_pattern = re.compile(
        r"<tr(?P<attrs>[^>]*)>\s*"
        r"<td[^>]*>(?P<name>.*?)</td>\s*"
        r"<td[^>]*>(?P<village>.*?)</td>\s*"
        r"<td[^>]*>(?P<capacity>\d+)</td>\s*"
        r"<td[^>]*>(?P<available>\d+)</td>\s*"
        r"<td[^>]*>(?P<longitude>\d+(?:\.\d+)?)</td>\s*"
        r"<td[^>]*>(?P<latitude>\d+(?:\.\d+)?)</td>\s*"
        r"</tr>",
        re.IGNORECASE | re.DOTALL,
    )
    for index, match in enumerate(html_pattern.finditer(text), start=1):
        attrs = match.group("attrs") or ""
        id_match = re.search(r'data-shelter-id="([^"]+)"', attrs)
        name = _strip_html(match.group("name"))
        village = _strip_html(match.group("village"))
        longitude = float(match.group("longitude"))
        latitude = float(match.group("latitude"))
        if not name or not village or not _is_within_beilin(longitude, latitude):
            continue
        rows.append(
            {
                "area_id": BEILIN_AREA_ID,
                "shelter_id": id_match.group(1) if id_match else _slugify(name, prefix=f"cached_shelter_{index}"),
                "name": name,
                "village": village,
                "capacity": int(match.group("capacity")),
                "available_capacity": int(match.group("available")),
                "accessible": True,
                "longitude": longitude,
                "latitude": latitude,
                "source_type": "official",
                "source_ref": "raw_cache",
                "is_synthetic": False,
            }
        )
    if rows:
        return rows

    line_rows: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 6:
            continue
        name, village, capacity, available_capacity, longitude, latitude = parts
        if not capacity.isdigit() or not available_capacity.isdigit():
            continue
        try:
            lon_value = float(longitude)
            lat_value = float(latitude)
        except ValueError:
            continue
        if not _is_within_beilin(lon_value, lat_value):
            continue
        line_rows.append(
            {
                "area_id": BEILIN_AREA_ID,
                "shelter_id": _slugify(name, prefix=f"cached_shelter_{index}"),
                "name": name,
                "village": village,
                "capacity": int(capacity),
                "available_capacity": int(available_capacity),
                "accessible": True,
                "longitude": lon_value,
                "latitude": lat_value,
                "source_type": "official",
                "source_ref": "raw_cache",
                "is_synthetic": False,
            }
        )
    return line_rows


def _parse_osm_json_file(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    features = payload.get("features") if isinstance(payload, dict) else None
    if isinstance(features, list):
        return _parse_geojson_features(features, path.name)
    elements = payload.get("elements") if isinstance(payload, dict) else None
    if isinstance(elements, list):
        return _parse_overpass_elements(elements, path.name)
    return {"roads": [], "poi_features": [], "parser_notes": [f"{path.name}: unsupported JSON layout."]}


def _parse_osm_xml_file(path: Path) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall(".//node"):
        node_id = node.attrib.get("id")
        lat = node.attrib.get("lat")
        lon = node.attrib.get("lon")
        if not node_id or lat is None or lon is None:
            continue
        nodes[node_id] = (float(lon), float(lat))
    roads: list[dict[str, Any]] = []
    poi_features: list[dict[str, Any]] = []
    for node in root.findall(".//node"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in node.findall("tag")}
        category = _map_osm_poi_category(tags)
        if not category:
            continue
        lon = float(node.attrib["lon"])
        lat = float(node.attrib["lat"])
        if not _is_within_beilin(lon, lat):
            continue
        poi_features.append(_build_poi_feature(f"osm_node_{node.attrib['id']}", tags.get("name") or category, category, lon, lat, tags, path.name))
    for way in root.findall(".//way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        refs = [nd.attrib.get("ref") for nd in way.findall("nd") if nd.attrib.get("ref") in nodes]
        coordinates = [nodes[ref] for ref in refs if ref]
        roads.extend(_build_roads_from_coordinates(f"osm_way_{way.attrib.get('id', 'road')}", tags, coordinates, path.name))
        category = _map_osm_poi_category(tags)
        if category and coordinates:
            centroid_lon = sum(item[0] for item in coordinates) / len(coordinates)
            centroid_lat = sum(item[1] for item in coordinates) / len(coordinates)
            if _is_within_beilin(centroid_lon, centroid_lat):
                poi_features.append(_build_poi_feature(f"osm_way_{way.attrib.get('id', 'poi')}", tags.get("name") or category, category, centroid_lon, centroid_lat, tags, path.name))
    return {
        "roads": roads,
        "poi_features": poi_features,
        "parser_notes": [f"{path.name}: parsed OSM XML roads={len(roads)} poi={len(poi_features)}"],
    }


def _parse_geojson_features(features: list[Any], source_name: str) -> dict[str, Any]:
    roads: list[dict[str, Any]] = []
    poi_features: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        geom_type = str(geometry.get("type") or "")
        coordinates = geometry.get("coordinates") or []
        if geom_type == "LineString":
            roads.extend(_build_roads_from_coordinates(f"{_slugify(str(properties.get('name') or source_name), prefix='geojson')}_{index}", properties, coordinates, source_name))
        elif geom_type == "Point" and len(coordinates) >= 2:
            lon, lat = float(coordinates[0]), float(coordinates[1])
            category = _map_osm_poi_category(properties)
            if category and _is_within_beilin(lon, lat):
                poi_features.append(_build_poi_feature(f"{source_name}_{index}", str(properties.get("name") or category), category, lon, lat, properties, source_name))
    return {
        "roads": roads,
        "poi_features": poi_features,
        "parser_notes": [f"{source_name}: parsed GeoJSON roads={len(roads)} poi={len(poi_features)}"],
    }


def _parse_overpass_elements(elements: list[Any], source_name: str) -> dict[str, Any]:
    nodes: dict[int, tuple[float, float]] = {}
    roads: list[dict[str, Any]] = []
    poi_features: list[dict[str, Any]] = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "node" and "lon" in item and "lat" in item:
            nodes[int(item["id"])] = (float(item["lon"]), float(item["lat"]))
            category = _map_osm_poi_category(item.get("tags") or {})
            if category and _is_within_beilin(float(item["lon"]), float(item["lat"])):
                poi_features.append(
                    _build_poi_feature(
                        f"osm_node_{item['id']}",
                        str((item.get("tags") or {}).get("name") or category),
                        category,
                        float(item["lon"]),
                        float(item["lat"]),
                        item.get("tags") or {},
                        source_name,
                    )
                )
    for item in elements:
        if not isinstance(item, dict) or item.get("type") != "way":
            continue
        tags = item.get("tags") or {}
        coordinates = []
        if isinstance(item.get("geometry"), list):
            coordinates = [(float(point["lon"]), float(point["lat"])) for point in item["geometry"] if "lon" in point and "lat" in point]
        elif isinstance(item.get("nodes"), list):
            coordinates = [nodes[int(node_id)] for node_id in item["nodes"] if int(node_id) in nodes]
        roads.extend(_build_roads_from_coordinates(f"osm_way_{item.get('id', 'road')}", tags, coordinates, source_name))
        category = _map_osm_poi_category(tags)
        if category and coordinates:
            centroid_lon = sum(item[0] for item in coordinates) / len(coordinates)
            centroid_lat = sum(item[1] for item in coordinates) / len(coordinates)
            if _is_within_beilin(centroid_lon, centroid_lat):
                poi_features.append(_build_poi_feature(f"osm_way_{item.get('id', 'poi')}", str(tags.get("name") or category), category, centroid_lon, centroid_lat, tags, source_name))
    return {
        "roads": roads,
        "poi_features": poi_features,
        "parser_notes": [f"{source_name}: parsed Overpass JSON roads={len(roads)} poi={len(poi_features)}"],
    }


def _build_roads_from_coordinates(
    road_id: str,
    properties: dict[str, Any],
    coordinates: list[Any],
    source_name: str,
) -> list[dict[str, Any]]:
    if len(coordinates) < 2:
        return []
    normalized = []
    for coordinate in coordinates:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
            continue
        lon, lat = float(coordinate[0]), float(coordinate[1])
        if _is_within_beilin(lon, lat):
            normalized.append((lon, lat))
    if len(normalized) < 2:
        return []
    name = str(properties.get("name") or properties.get("ref") or "OSM road segment")
    village = str(properties.get("addr:suburb") or properties.get("addr:district") or "碑林区")
    destination = str(properties.get("to") or properties.get("destination") or name)
    start_lon, start_lat = normalized[0]
    end_lon, end_lat = normalized[-1]
    return [
        {
            "area_id": BEILIN_AREA_ID,
            "road_id": road_id,
            "name": name,
            "village": village,
            "destination": destination,
            "start_longitude": start_lon,
            "start_latitude": start_lat,
            "end_longitude": end_lon,
            "end_latitude": end_lat,
            "road_type": str(properties.get("highway") or properties.get("railway") or "road"),
            "source_type": "open_map",
            "source_ref": source_name,
            "is_synthetic": False,
        }
    ]


def _build_poi_feature(
    poi_id: str,
    name: str,
    category: str,
    longitude: float,
    latitude: float,
    tags: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    return {
        "poi_id": poi_id,
        "name": name,
        "category": category,
        "longitude": longitude,
        "latitude": latitude,
        "tags": tags,
        "source_type": "open_map",
        "source_ref": source_name,
    }


def _map_osm_poi_category(tags: dict[str, Any]) -> str | None:
    amenity = str(tags.get("amenity") or "")
    healthcare = str(tags.get("healthcare") or "")
    station = str(tags.get("station") or "")
    railway = str(tags.get("railway") or "")
    public_transport = str(tags.get("public_transport") or "")
    landuse = str(tags.get("landuse") or "")
    building = str(tags.get("building") or "")
    shop = str(tags.get("shop") or "")
    if amenity in {"school", "college", "kindergarten"}:
        return "school"
    if amenity in {"hospital", "clinic"}:
        return "hospital"
    if healthcare in {"hospital", "clinic"}:
        return "hospital"
    if amenity in {"nursing_home", "social_facility"}:
        return "nursing_home"
    if railway == "station" or public_transport == "station" or station == "subway":
        return "metro_station"
    if amenity in {"community_centre", "townhall"}:
        return "community"
    if landuse == "industrial" or building in {"industrial", "warehouse"}:
        return "factory"
    if shop in {"mall"} or building in {"retail", "commercial"}:
        return "underground_space"
    return None


def _repo_root_from_raw_dir(raw_dir: Path) -> Path:
    return raw_dir.resolve().parents[2]


def _source_is_required(source: dict[str, Any]) -> bool:
    return str(source.get("source_id") or "") != "amap_webservice"


def _iter_source_artifacts(source: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("source_ref", str(source.get("source_ref", ""))),
        ("download", str(source.get("download_url", ""))),
    ]


def _load_artifact_records(raw_dir: Path, source_id: str) -> list[dict[str, Any]]:
    source_dir = raw_dir / source_id
    payload = _load_json(source_dir / "artifacts.json")
    return payload if isinstance(payload, list) else []


def _guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_artifact_record(
    repo_root: Path,
    *,
    source_id: str,
    artifact_name: str,
    artifact_url: str,
    target: Path,
    fetched_at: str,
    status: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "artifact_name": artifact_name,
        "source_url": artifact_url,
        "local_path": str(target.relative_to(repo_root)),
        "file_size_bytes": target.stat().st_size if target.exists() else 0,
        "file_hash": _file_sha256(target) if target.exists() else "",
        "content_type": _guess_content_type(target),
        "fetched_at": fetched_at,
        "status": status,
        "used_by_parser": False,
        "normalized_targets": [],
    }


def _write_text_companion(path: Path) -> Path | None:
    if not path.exists():
        return None
    text = ""
    if path.suffix.lower() == ".pdf":
        text = _load_pdf_text(path)
    elif path.suffix.lower() in {".html", ".htm", ".xml"}:
        text = _strip_html(_load_cached_text(path))
    elif path.suffix.lower() in {".json", ".geojson", ".osm"}:
        text = _load_cached_text(path)
    if not text.strip():
        return None
    target = path.with_suffix(path.suffix + ".txt")
    target.write_text(text, encoding="utf-8")
    return target


def _append_version_record(path: Path, record: dict[str, Any]) -> None:
    payload = _load_json(path)
    versions = payload if isinstance(payload, list) else []
    versions.append(record)
    _write_json(path, versions[-20:])


def _mark_artifacts_used(
    artifacts: list[dict[str, Any]],
    *,
    used_paths: list[str],
    normalized_targets: list[str],
) -> list[dict[str, Any]]:
    used = set(used_paths)
    updated: list[dict[str, Any]] = []
    for artifact in artifacts:
        next_artifact = dict(artifact)
        if str(next_artifact.get("local_path") or "") in used:
            next_artifact["used_by_parser"] = True
            next_artifact["normalized_targets"] = normalized_targets
        updated.append(next_artifact)
    return updated


def _determine_cache_status(
    *,
    downloaded_artifact_count: int,
    failed_artifact_count: int,
    parsed_summary: dict[str, Any],
    has_manifest: bool,
) -> str:
    if parsed_summary:
        return "parsed"
    if failed_artifact_count > 0 and downloaded_artifact_count == 0:
        return "download_failed"
    if downloaded_artifact_count > 0 and failed_artifact_count > 0:
        return "partial_cached"
    if downloaded_artifact_count > 0:
        return "cached"
    if has_manifest:
        return "download_failed"
    return "missing"


def _classify_source_completeness(
    source: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact_count: int,
    downloaded_artifact_count: int,
    cached_files: list[str],
    source_dir_exists: bool,
) -> str:
    cache_status = str(metadata.get("cache_status") or "")
    if metadata.get("parsed_summary"):
        return "parsed"
    if cache_status in {"download_failed", "partial_cached"}:
        return "partial_cached"
    if not cached_files and (metadata or source_dir_exists or artifact_count > 0):
        return "manifest_only"
    if artifact_count and downloaded_artifact_count < artifact_count:
        return "partial_cached"
    if cached_files:
        return "cached"
    return "missing"


def _missing_artifact_types(source: dict[str, Any], artifacts: list[dict[str, Any]]) -> list[str]:
    existing = {str(item.get("artifact_name") or "") for item in artifacts}
    missing = []
    for artifact_name, url in _iter_source_artifacts(source):
        if str(url or "").strip() and artifact_name not in existing:
            missing.append(artifact_name)
    return missing


def _completeness_score(status: str) -> int:
    return {
        "missing": 0,
        "manifest_only": 25,
        "partial_cached": 60,
        "cached": 85,
        "parsed": 100,
    }.get(status, 0)


def _attach_source_trace(
    rows: list[dict[str, Any]],
    *,
    source_id: str,
    artifacts: list[dict[str, Any]],
    fallback_to_synthetic: bool,
) -> list[dict[str, Any]]:
    traced: list[dict[str, Any]] = []
    for row in rows:
        traced.append(
            {
                **row,
                "source_trace": {
                    "source_id": source_id,
                    "artifacts": [artifact.get("local_path") for artifact in artifacts if artifact.get("used_by_parser")],
                    "fallback_to_synthetic": fallback_to_synthetic,
                },
            }
        )
    return traced


def _guess_suffix_from_url(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix:
        return suffix
    if "html" in url:
        return ".html"
    return ".bin"


def _count_source_artifacts(source: dict[str, Any]) -> int:
    return sum(1 for value in (source.get("source_ref", ""), source.get("download_url", "")) if str(value or "").strip())


def _dedupe_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        seen[str(row[key])] = row
    return list(seen.values())


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_pdf_text(path: Path) -> str:
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages).strip()
    except Exception:
        pass
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return ""


def _load_cached_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            break
    return ""


def _strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"&nbsp;?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _slugify(value: str, *, prefix: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or prefix


def _is_within_beilin(longitude: float, latitude: float) -> bool:
    return BEILIN_LON_RANGE[0] <= longitude <= BEILIN_LON_RANGE[1] and BEILIN_LAT_RANGE[0] <= latitude <= BEILIN_LAT_RANGE[1]


def _assert_beilin_coordinate(longitude: float, latitude: float, label: str) -> None:
    if not _is_within_beilin(longitude, latitude):
        raise ValueError(f"{label} coordinate ({longitude}, {latitude}) is outside Beilin bounds.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Beilin data package for the flood warning system.")
    parser.add_argument("--root", default=None, help="Repository root.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch-beilin-sources")
    fetch_parser.add_argument("--download", action="store_true")
    subparsers.add_parser("normalize-beilin-sources")
    subparsers.add_parser("build-beilin-profiles")
    subparsers.add_parser("generate-beilin-observations")
    subparsers.add_parser("compile-beilin-rag")
    subparsers.add_parser("validate-beilin-dataset")
    sync_parser = subparsers.add_parser("sync-demo-db")
    sync_parser.add_argument("--db-path", default=None)
    build_cmd = subparsers.add_parser("build")
    build_cmd.add_argument("--download", action="store_true")
    build_cmd.add_argument("--sync-demo-db", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root
    if args.command == "fetch-beilin-sources":
        result = fetch_beilin_sources(root, download=args.download)
    elif args.command == "normalize-beilin-sources":
        result = normalize_beilin_sources(root)
    elif args.command == "build-beilin-profiles":
        result = build_beilin_profiles(root)
    elif args.command == "generate-beilin-observations":
        result = generate_beilin_observations(root)
    elif args.command == "compile-beilin-rag":
        result = compile_beilin_rag(root)
    elif args.command == "validate-beilin-dataset":
        result = validate_beilin_dataset(root)
    elif args.command == "sync-demo-db":
        result = sync_demo_db(root, db_path=args.db_path)
    elif args.command == "build":
        result = build_dataset(root, download=args.download, sync_db=args.sync_demo_db)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
