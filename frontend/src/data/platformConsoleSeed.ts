import type { EntityType, ObservationIngestItem } from "../types/api";

export interface CuratedEntity {
  id: string;
  name: string;
  type: EntityType;
  village: string;
  emphasis: string;
}

export const platformSeedObservations: ObservationIngestItem[] = [
  {
    observed_at: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
    source_type: "monitoring_point",
    source_name: "西侧积涝监测点",
    village: "联市街片区",
    rainfall_mm: 28,
    water_level_m: 3.6,
    citizen_reports: 2,
  },
  {
    observed_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    source_type: "water_level_sensor",
    source_name: "南门涵洞水位传感器",
    village: "南门片区",
    rainfall_mm: 36,
    water_level_m: 4.1,
    road_blocked: true,
    citizen_reports: 4,
    notes: "支路通行已经出现局部受阻。",
  },
  {
    observed_at: new Date(Date.now() - 8 * 60 * 1000).toISOString(),
    source_type: "camera_alert",
    source_name: "和平门下穿通道摄像头",
    village: "和平门片区",
    rainfall_mm: 32,
    water_level_m: 4.4,
    road_blocked: true,
    citizen_reports: 5,
    notes: "附近已出现地下空间进水上报。",
  },
];

export const curatedEntities: CuratedEntity[] = [
  {
    id: "resident_elderly_ls1",
    name: "李阿姨",
    type: "resident",
    village: "联市街片区",
    emphasis: "居住在低洼院落，转移时需要协助与照护衔接。",
  },
  {
    id: "school_wyl_primary",
    name: "五岳里小学",
    type: "school",
    village: "五岳里片区",
    emphasis: "放学高峰与积水叠加时，校门口极易形成转运瓶颈。",
  },
  {
    id: "factory_wyr_bio",
    name: "五岳里生物制剂厂",
    type: "factory",
    village: "五岳里片区",
    emphasis: "冷链库存和装卸月台对时间非常敏感，需要提前转移。",
  },
  {
    id: "nursing_home_hpm",
    name: "和平门颐养中心",
    type: "nursing_home",
    village: "和平门片区",
    emphasis: "失能老人较多，转运需要分批次并保障供氧连续。",
  },
  {
    id: "metro_nsm_hub",
    name: "南门地铁换乘枢纽",
    type: "metro_station",
    village: "南门片区",
    emphasis: "一旦入口积水，地下客流秩序会迅速恶化。",
  },
  {
    id: "community_jsl_grid",
    name: "建设里社区网格三组",
    type: "community",
    village: "建设里片区",
    emphasis: "混合住户与地下空间较多，通知时机直接影响转移效率。",
  },
];
