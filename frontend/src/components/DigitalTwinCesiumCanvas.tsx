import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import "cesium/Build/Cesium/Widgets/widgets.css";
import {
  buildCalibratedModelMatrix,
  buildSimpleModelMatrix,
  computeModelFocusSphere,
  loadSourceMetadata,
  normalizeSceneConfig,
  resolveModelAssetUrl,
  type SceneConfig,
  type SourceMetadata,
} from "../lib/cityengineCalibration";
import type { RiskLevel, TwinObjectMapLayer } from "../types/api";
import styles from "../styles/digital-twin-map.module.css";

type CesiumModule = typeof import("cesium");
type TwinEntityBundle = {
  marker: any;
  heat?: any;
  pulse?: any;
  route?: any;
  water?: any;
  waterColumn?: any;
  buildingHighlight?: any;
  roadClosure?: any;
  evacuationRoute?: any;
  resourceVehicle?: any;
  warningSpread?: any;
  warningRings?: any[];
};

type NarrativeStepKey = "overview" | "risk" | "water" | "route" | "closure";

type NarrativeStep = {
  key: NarrativeStepKey;
  shortLabel: string;
  title: string;
  narrative: string;
};

const NARRATIVE_STEPS: NarrativeStep[] = [
  {
    key: "overview",
    shortLabel: "01",
    title: "全域态势",
    narrative: "展示城市模型、风险热区、信号流和当前指挥焦点。",
  },
  {
    key: "risk",
    shortLabel: "02",
    title: "风险源",
    narrative: "定位最高风险对象及其上游触发因素。",
  },
  {
    key: "water",
    shortLabel: "03",
    title: "水位态势",
    narrative: "查看焦点对象周边的动态积水面和水位柱。",
  },
  {
    key: "route",
    shortLabel: "04",
    title: "处置路线",
    narrative: "高亮疏散路径、道路阻断线和资源车辆位置。",
  },
  {
    key: "closure",
    shortLabel: "05",
    title: "审批闭环",
    narrative: "聚焦处置方案审批状态及闭环落点对象。",
  },
];

interface DigitalTwinCesiumCanvasProps {
  layers: TwinObjectMapLayer[];
  dialogFocusObjectId?: string | null;
  dialogFocusSerial?: number;
  routeHighlightObjectId?: string | null;
  selectedRiskLevel?: RiskLevel | null;
  onSelectObject: (objectId: string) => void;
}

function toneClassName(riskLevel?: RiskLevel | null) {
  return {
    None: styles.toneNone,
    Blue: styles.toneBlue,
    Yellow: styles.toneYellow,
    Orange: styles.toneOrange,
    Red: styles.toneRed,
  }[riskLevel ?? "None"];
}

function stateLabel(proposalState?: string | null) {
  const normalizedState = proposalState ?? "monitoring";
  return {
    monitoring: "监测中",
    pending: "待审批方案",
    approved: "已批准动作",
    warning_generated: "预警已生成",
  }[normalizedState] ?? normalizedState;
}

function stateColor(proposalState?: string | null) {
  const normalizedState = proposalState ?? "monitoring";
  return {
    monitoring: "#28d8ff",
    pending: "#ff9d38",
    approved: "#35e6bf",
    warning_generated: "#ff6b2d",
  }[normalizedState] ?? "#28d8ff";
}

function riskRadius(riskLevel?: RiskLevel | null, proposalState?: string) {
  const base = {
    None: 52,
    Blue: 72,
    Yellow: 96,
    Orange: 124,
    Red: 156,
  }[riskLevel ?? "None"];
  return base + (proposalState === "pending" ? 26 : proposalState === "warning_generated" ? 38 : 0);
}

function waterDepthCm(riskLevel?: RiskLevel | null, proposalState?: string) {
  const base = {
    None: 12,
    Blue: 20,
    Yellow: 26,
    Orange: 32,
    Red: 38,
  }[riskLevel ?? "None"];
  return Math.min(40, base + (proposalState === "pending" ? 4 : proposalState === "warning_generated" ? 2 : 0));
}

function riskMotionIntensity(riskLevel?: RiskLevel | null, proposalState?: string) {
  const base = {
    None: 0.35,
    Blue: 0.55,
    Yellow: 0.72,
    Orange: 0.92,
    Red: 1.14,
  }[riskLevel ?? "None"];
  return base + (proposalState === "pending" ? 0.16 : proposalState === "warning_generated" ? 0.24 : 0);
}

function buildingHighlightSize(entityType?: string | null) {
  const dimensions = {
    community: { width: 92, depth: 72, height: 34 },
    school: { width: 74, depth: 58, height: 42 },
    hospital: { width: 82, depth: 62, height: 58 },
    resident: { width: 38, depth: 34, height: 24 },
    nursing_home: { width: 66, depth: 54, height: 38 },
    metro_station: { width: 62, depth: 46, height: 28 },
    underground_space: { width: 72, depth: 52, height: 22 },
    factory: { width: 90, depth: 70, height: 40 },
  }[entityType ?? ""];
  return dimensions ?? { width: 58, depth: 48, height: 32 };
}

type RouteAnchor = Pick<TwinObjectMapLayer, "east_offset_m" | "north_offset_m" | "height_offset_m" | "entity_type">;
type RouteOffsetPoint = { east: number; north: number; height: number };

const ROAD_CORRIDOR_X = [-220, 0, 310];
const ROAD_CORRIDOR_Y = [-165, -70, 150];
const PRIMARY_ROAD_EAST = 0;
const PRIMARY_ROAD_NORTH = -70;
const ROAD_ROUTE_HEIGHT_M = 7;

function nearestCorridor(value: number, corridors: number[]) {
  return corridors.reduce((best, current) => (Math.abs(current - value) < Math.abs(best - value) ? current : best), corridors[0]);
}

function appendRoutePoint(points: RouteOffsetPoint[], point: RouteOffsetPoint) {
  const previous = points[points.length - 1];
  if (
    previous &&
    Math.abs(previous.east - point.east) < 1 &&
    Math.abs(previous.north - point.north) < 1 &&
    Math.abs(previous.height - point.height) < 1
  ) {
    return;
  }
  points.push(point);
}

function appendRoadSegment(points: RouteOffsetPoint[], from: RouteOffsetPoint, to: RouteOffsetPoint) {
  const sameEast = Math.abs(from.east - to.east) < 1;
  const sameNorth = Math.abs(from.north - to.north) < 1;

  if (!sameEast && !sameNorth) {
    appendRoadSegment(points, from, { east: to.east, north: from.north, height: from.height });
    appendRoadSegment(points, { east: to.east, north: from.north, height: from.height }, to);
    return;
  }

  appendRoutePoint(points, from);

  if (!sameEast) {
    const [minEast, maxEast] = from.east < to.east ? [from.east, to.east] : [to.east, from.east];
    const corridorPoints = ROAD_CORRIDOR_X.filter((east) => east > minEast && east < maxEast);
    const orderedEast = from.east < to.east ? corridorPoints : [...corridorPoints].reverse();
    for (const east of orderedEast) {
      appendRoutePoint(points, { east, north: from.north, height: from.height });
    }
  } else if (!sameNorth) {
    const [minNorth, maxNorth] = from.north < to.north ? [from.north, to.north] : [to.north, from.north];
    const corridorPoints = ROAD_CORRIDOR_Y.filter((north) => north > minNorth && north < maxNorth);
    const orderedNorth = from.north < to.north ? corridorPoints : [...corridorPoints].reverse();
    for (const north of orderedNorth) {
      appendRoutePoint(points, { east: from.east, north, height: from.height });
    }
  }

  appendRoutePoint(points, to);
}

function targetMainRoadEast(anchor: RouteAnchor) {
  if (anchor.east_offset_m > 80) {
    return 310;
  }
  if (anchor.east_offset_m < -120) {
    return -220;
  }
  return PRIMARY_ROAD_EAST;
}

function buildRoadFollowingRoute(source: RouteAnchor, target: RouteAnchor) {
  const height = ROAD_ROUTE_HEIGHT_M;
  const sourceRoadNorth = nearestCorridor(source.north_offset_m, ROAD_CORRIDOR_Y);
  const targetRoadNorth = nearestCorridor(target.north_offset_m, ROAD_CORRIDOR_Y);
  const start = { east: PRIMARY_ROAD_EAST, north: sourceRoadNorth, height };
  const turn = { east: PRIMARY_ROAD_EAST, north: targetRoadNorth, height };
  const end = { east: targetMainRoadEast(target), north: targetRoadNorth, height };
  const route: RouteOffsetPoint[] = [];

  appendRoadSegment(route, start, turn);
  appendRoadSegment(route, turn, end);

  if (route.length < 2) {
    appendRoutePoint(route, { east: 310, north: sourceRoadNorth, height });
  }

  return route;
}

function buildRoadClosureRoute(anchor: RouteAnchor, radius: number) {
  const height = ROAD_ROUTE_HEIGHT_M;
  const closeOnVerticalRoad = Math.abs(anchor.east_offset_m - PRIMARY_ROAD_EAST) <= Math.abs(anchor.north_offset_m - PRIMARY_ROAD_NORTH);
  const halfLength = Math.max(36, Math.min(radius * 0.44, 72));

  if (closeOnVerticalRoad) {
    return [
      { east: PRIMARY_ROAD_EAST, north: nearestCorridor(anchor.north_offset_m, ROAD_CORRIDOR_Y) - halfLength, height },
      { east: PRIMARY_ROAD_EAST, north: nearestCorridor(anchor.north_offset_m, ROAD_CORRIDOR_Y) + halfLength, height },
    ];
  }

  return [
    { east: PRIMARY_ROAD_EAST - halfLength, north: PRIMARY_ROAD_NORTH, height },
    { east: PRIMARY_ROAD_EAST + halfLength, north: PRIMARY_ROAD_NORTH, height },
  ];
}

function objectPhaseSeed(objectId: string) {
  return (
    objectId.split("").reduce((total, char, index) => total + char.charCodeAt(0) * (index + 3), 0) % 628
  ) / 100;
}

function nowSeconds() {
  if (typeof performance !== "undefined") {
    return performance.now() / 1000;
  }
  return Date.now() / 1000;
}

function extractObjectId(entityId?: string) {
  return entityId?.split(":")[0] ?? "";
}

function isJsdomRuntime() {
  return typeof navigator !== "undefined" && /jsdom/i.test(navigator.userAgent);
}

function formatCanvasError(caught: unknown) {
  if (caught instanceof Error) {
    return caught.message || "未知三维地图错误";
  }
  if (typeof caught === "string") {
    return caught || "未知三维地图错误";
  }
  try {
    return JSON.stringify(caught) || "未知三维地图错误";
  } catch {
    return "未知三维地图错误";
  }
}

export function DigitalTwinCesiumCanvas({
  layers,
  dialogFocusObjectId,
  dialogFocusSerial = 0,
  routeHighlightObjectId,
  selectedRiskLevel,
  onSelectObject,
}: DigitalTwinCesiumCanvasProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<any>(null);
  const handlerRef = useRef<any>(null);
  const entityMapRef = useRef<Map<string, TwinEntityBundle>>(new Map());
  const tourTimersRef = useRef<number[]>([]);
  const cesiumRef = useRef<CesiumModule | null>(null);
  const sceneFocusRef = useRef<any>(null);
  const lastDialogFocusRef = useRef<string | null>(null);
  const onSelectObjectRef = useRef(onSelectObject);
  const [status, setStatus] = useState("正在加载数字孪生画布");
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [tourRunning, setTourRunning] = useState(false);
  const [tourNarrative, setTourNarrative] = useState("指挥镜头已就绪");
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [activeNarrativeStep, setActiveNarrativeStep] = useState<NarrativeStepKey>("overview");

  const leadLayer = useMemo(
    () => layers.find((item) => item.is_lead) ?? layers[0] ?? null,
    [layers],
  );
  const layerStats = useMemo(
    () => ({
      monitoring: layers.filter((item) => item.proposal_state === "monitoring").length,
      pending: layers.filter((item) => item.proposal_state === "pending").length,
      approved: layers.filter((item) => item.proposal_state === "approved").length,
      warningGenerated: layers.filter((item) => item.proposal_state === "warning_generated").length,
    }),
    [layers],
  );
  const layerById = useMemo(() => new Map(layers.map((item) => [item.object_id, item])), [layers]);
  useEffect(() => {
    onSelectObjectRef.current = onSelectObject;
  }, [onSelectObject]);
  const spotlightLayer =
    (hoveredObjectId ? layerById.get(hoveredObjectId) : undefined) ??
    (dialogFocusObjectId ? layerById.get(dialogFocusObjectId) : undefined) ??
    (routeHighlightObjectId ? layerById.get(routeHighlightObjectId) : undefined) ??
    leadLayer ??
    null;
  const toneClass = toneClassName(selectedRiskLevel ?? spotlightLayer?.risk_level ?? leadLayer?.risk_level ?? "None");
  const narrativeStepIndex = NARRATIVE_STEPS.findIndex((item) => item.key === activeNarrativeStep);
  const narrativeLayers = useMemo(() => {
    const riskSource =
      leadLayer ??
      layers.find((item) => item.risk_level === "Red" || item.risk_level === "Orange") ??
      layers[0] ??
      null;
    const focusLayer = dialogFocusObjectId ? layerById.get(dialogFocusObjectId) ?? riskSource : riskSource;
    const routeLayer =
      (routeHighlightObjectId ? layerById.get(routeHighlightObjectId) : undefined) ??
      layers.find((item) => item.object_id !== riskSource?.object_id && item.proposal_state === "approved") ??
      layers.find((item) => item.object_id !== riskSource?.object_id && item.proposal_state === "pending") ??
      layers.find((item) => item.object_id !== riskSource?.object_id) ??
      riskSource;
    const closureLayer =
      layers.find((item) => item.proposal_state === "pending") ??
      layers.find((item) => item.proposal_state === "approved") ??
      focusLayer;
    return {
      overview: null,
      risk: riskSource,
      water: focusLayer,
      route: routeLayer,
      closure: closureLayer,
    } satisfies Record<NarrativeStepKey, TwinObjectMapLayer | null>;
  }, [dialogFocusObjectId, layerById, layers, leadLayer, routeHighlightObjectId]);

  const resolveNarrativeEntity = (stepKey: NarrativeStepKey) => {
    const layer = narrativeLayers[stepKey];
    if (stepKey === "overview") {
      return { layer, entity: sceneFocusRef.current, kind: "overview" as const };
    }
    if (!layer) {
      return { layer: null, entity: sceneFocusRef.current, kind: "overview" as const };
    }

    const bundle = entityMapRef.current.get(layer.object_id);
    if (stepKey === "risk") {
      return { layer, entity: bundle?.waterColumn ?? bundle?.water ?? bundle?.marker, kind: "entity" as const };
    }
    if (stepKey === "water") {
      return { layer, entity: bundle?.waterColumn ?? bundle?.water ?? bundle?.marker, kind: "water" as const };
    }
    if (stepKey === "route") {
      return {
        layer,
        entity: bundle?.evacuationRoute ?? bundle?.resourceVehicle ?? bundle?.route ?? bundle?.marker,
        kind: "route" as const,
      };
    }
    if (stepKey === "closure") {
      return { layer, entity: bundle?.marker ?? bundle?.roadClosure, kind: "entity" as const };
    }
    return { layer, entity: bundle?.marker, kind: "entity" as const };
  };

  const flyToNarrativeStep = (stepKey: NarrativeStepKey, options?: { duration?: number; autoSelect?: boolean }) => {
    const step = NARRATIVE_STEPS.find((item) => item.key === stepKey);
    if (!step) {
      return;
    }
    setActiveNarrativeStep(stepKey);
    setTourNarrative(step.narrative);

    const viewer = viewerRef.current;
    const Cesium = cesiumRef.current;
    const target = resolveNarrativeEntity(stepKey);
    if (target.layer && options?.autoSelect !== false) {
      onSelectObjectRef.current(target.layer.object_id);
    }
    if (!viewer || !Cesium || !target.entity) {
      return;
    }
    if (target.kind === "overview") {
      viewer.camera.flyToBoundingSphere(target.entity, {
        duration: options?.duration ?? 1.1,
        offset: new Cesium.HeadingPitchRange(Cesium.Math.toRadians(22), Cesium.Math.toRadians(-42), 2600),
      });
      return;
    }
    viewer.flyTo(target.entity, {
      duration: options?.duration ?? 1.1,
      offset: new Cesium.HeadingPitchRange(
        Cesium.Math.toRadians(target.kind === "route" ? 72 : 38),
        Cesium.Math.toRadians(-28),
        target.kind === "route" ? 740 : 430,
      ),
    });
  };

  const runCommandFlythrough = () => {
    const viewer = viewerRef.current;
    const Cesium = cesiumRef.current;
    if (!viewer || !Cesium || layers.length === 0) {
      return;
    }

    for (const timer of tourTimersRef.current) {
      window.clearTimeout(timer);
    }
    tourTimersRef.current = [];
    setTourRunning(true);
    flyToNarrativeStep("overview", { duration: 1.1, autoSelect: false });
    NARRATIVE_STEPS.slice(1).forEach((step, index) => {
      const timer = window.setTimeout(() => {
        flyToNarrativeStep(step.key, { duration: 1.15 });
      }, (index + 1) * 1600);
      tourTimersRef.current.push(timer);
    });
    const stopTimer = window.setTimeout(() => {
      setTourRunning(false);
      setTourNarrative("指挥叙事完成：影响、路线与审批闭环已对齐。");
      tourTimersRef.current = [];
    }, NARRATIVE_STEPS.length * 1600 + 700);
    tourTimersRef.current.push(stopTimer);
  };

  useEffect(() => {
    if (!hostRef.current || isJsdomRuntime()) {
      return;
    }

    let disposed = false;

    async function initViewer() {
      try {
        const Cesium = await import("cesium");
        cesiumRef.current = Cesium;
        setStatus("正在加载三维场景配置");

        const response = await fetch("/agent-twin-assets/scene-config.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`scene-config.json returned ${response.status}`);
        }

        const sceneConfig = normalizeSceneConfig((await response.json()) as Partial<SceneConfig>);
        const {
          BoundingSphere,
          CallbackProperty,
          Cartesian2,
          Cartesian3,
          Color,
          ColorMaterialProperty,
          EllipsoidTerrainProvider,
          HeadingPitchRange,
          HorizontalOrigin,
          Ion,
          IonWorldImageryStyle,
          Math: CesiumMath,
          Matrix4,
          Model,
          PolylineArrowMaterialProperty,
          PolylineGlowMaterialProperty,
          ScreenSpaceEventHandler,
          ScreenSpaceEventType,
          Transforms,
          VerticalOrigin,
          Viewer,
          createWorldImageryAsync,
          createWorldTerrainAsync,
          defined,
          sampleTerrainMostDetailed,
        } = Cesium;

        let terrainProvider: any = new EllipsoidTerrainProvider();
        let imageryProvider: any;
        let anchorHeight = sceneConfig.anchorHeight;

        if (__CESIUM_ION_TOKEN__) {
          Ion.defaultAccessToken = __CESIUM_ION_TOKEN__;
          terrainProvider = await createWorldTerrainAsync();
          imageryProvider = await createWorldImageryAsync({ style: IonWorldImageryStyle.AERIAL });
          const sampled = await sampleTerrainMostDetailed(terrainProvider, [
            Cesium.Cartographic.fromDegrees(sceneConfig.anchorLon, sceneConfig.anchorLat),
          ]);
          if (Number.isFinite(sampled[0]?.height)) {
            anchorHeight = sampled[0].height as number;
          }
        }

        if (disposed || !hostRef.current) {
          return;
        }

        setStatus("正在启动三维地图引擎");
        const viewer = new Viewer(hostRef.current, {
          terrainProvider,
          baseLayer: false,
          animation: false,
          timeline: false,
          homeButton: false,
          baseLayerPicker: false,
          fullscreenButton: false,
          geocoder: false,
          infoBox: false,
          navigationHelpButton: false,
          sceneModePicker: false,
          selectionIndicator: false,
        });

        viewer.scene.globe.depthTestAgainstTerrain = false;
        if (viewer.scene.skyAtmosphere) {
          viewer.scene.skyAtmosphere.show = true;
        }
        viewer.scene.fog.enabled = false;
        viewer.scene.requestRenderMode = false;
        if (imageryProvider) {
          viewer.imageryLayers.addImageryProvider(imageryProvider);
        }
        viewerRef.current = viewer;

        setStatus("正在分析城市建筑模型坐标");
        let sourceMetadata: SourceMetadata | null = null;
        if (sceneConfig.placementMode !== "simple") {
          try {
            sourceMetadata = await loadSourceMetadata(Cesium, sceneConfig);
          } catch (metadataError) {
            console.warn("城市建筑模型坐标分析失败，切换为锚点定位", metadataError);
          }
        }

        setStatus("正在恢复城市建筑模型位置");
        const anchor = Cartesian3.fromDegrees(sceneConfig.anchorLon, sceneConfig.anchorLat, anchorHeight);
        const anchorFrame = Transforms.eastNorthUpToFixedFrame(anchor);
        const modelMatrix = sourceMetadata
          ? buildCalibratedModelMatrix(Cesium, sceneConfig, sourceMetadata, anchorHeight)
          : buildSimpleModelMatrix(Cesium, sceneConfig, anchorHeight);

        const model = await Model.fromGltfAsync({
          url: resolveModelAssetUrl(sceneConfig.modelUrl),
          modelMatrix,
          color: Color.fromCssColorString("#d9f8ff").withAlpha(1),
          colorBlendMode: Cesium.ColorBlendMode.MIX,
          colorBlendAmount: 0.08,
          silhouetteColor: Color.fromCssColorString("#34dcff"),
          silhouetteSize: 0,
          minimumPixelSize: 0,
          upAxis: Cesium.Axis.Z,
          forwardAxis: Cesium.Axis.X,
        });
        if (!disposed) {
          viewer.scene.primitives.add(model);
        }

        entityMapRef.current.clear();
        const layerPositions = new Map<string, any>();
        for (const layer of layers) {
          const position = Matrix4.multiplyByPoint(
            anchorFrame,
            new Cartesian3(layer.east_offset_m, layer.north_offset_m, layer.height_offset_m),
            new Cartesian3(),
          );
          layerPositions.set(layer.object_id, position);
        }

        for (const layer of layers) {
          try {
            const position = layerPositions.get(layer.object_id);
            const color = Color.fromCssColorString(stateColor(layer.proposal_state));
            const radius = riskRadius(layer.risk_level, layer.proposal_state);
            const depthCm = waterDepthCm(layer.risk_level, layer.proposal_state);
            const columnHeight = Math.max(14, depthCm * 0.55);
            const highlightSize = buildingHighlightSize(layer.entity_type);
            const intensity = riskMotionIntensity(layer.risk_level, layer.proposal_state);
            const phaseSeed = objectPhaseSeed(layer.object_id);
            const closurePositions = buildRoadClosureRoute(layer, radius).map((point) =>
              Matrix4.multiplyByPoint(
                anchorFrame,
                new Cartesian3(point.east, point.north, point.height),
                new Cartesian3(),
              ),
            );
            const heat = viewer.entities.add({
            id: `${layer.object_id}:risk-heat`,
            name: `${layer.name} risk heat`,
            position,
            ellipse: {
              semiMajorAxis: radius,
              semiMinorAxis: radius * 0.72,
              material: color.withAlpha(layer.proposal_state === "monitoring" ? 0.12 : 0.2),
              outline: true,
              outlineColor: color.withAlpha(0.38),
            },
          });
          const water = viewer.entities.add({
            id: `${layer.object_id}:flood-water`,
            name: `${layer.name} flood water surface`,
            position,
            ellipse: {
              semiMajorAxis: new CallbackProperty(
                () => radius * (1.06 + Math.sin(nowSeconds() * 1.45 + phaseSeed) * 0.035 * intensity),
                false,
              ),
              semiMinorAxis: new CallbackProperty(
                () => radius * (0.58 + Math.cos(nowSeconds() * 1.38 + phaseSeed) * 0.028 * intensity),
                false,
              ),
              material: new ColorMaterialProperty(
                new CallbackProperty(
                  () =>
                    Color.fromCssColorString("#20cfff").withAlpha(
                      (layer.risk_level === "Red" || layer.risk_level === "Orange" ? 0.26 : 0.13) +
                        Math.max(0, Math.sin(nowSeconds() * 1.35 + phaseSeed)) * 0.06 * intensity,
                    ),
                  false,
                ),
              ),
              outline: true,
              outlineColor: new CallbackProperty(
                () => Color.fromCssColorString("#9af4ff").withAlpha(0.22 + Math.max(0, Math.sin(nowSeconds() * 1.2 + phaseSeed)) * 0.2),
                false,
              ),
            },
          });
          const waterColumn = viewer.entities.add({
            id: `${layer.object_id}:water-column`,
            name: `${layer.name} water level column`,
            position: new CallbackProperty(
              () =>
                Matrix4.multiplyByPoint(
                  anchorFrame,
                  new Cartesian3(
                    layer.east_offset_m,
                    layer.north_offset_m,
                    layer.height_offset_m + columnHeight / 2 + 4 + Math.sin(nowSeconds() * 1.65 + phaseSeed) * (1.8 + intensity),
                  ),
                  new Cartesian3(),
                ),
              false,
            ) as any,
            cylinder: {
              length: columnHeight,
              topRadius: 7,
              bottomRadius: 11,
              material: new ColorMaterialProperty(
                new CallbackProperty(
                  () => Color.fromCssColorString("#5fe7ff").withAlpha(0.36 + Math.max(0, Math.sin(nowSeconds() * 2.05 + phaseSeed)) * 0.18),
                  false,
                ),
              ),
              outline: true,
              outlineColor: new CallbackProperty(
                () => Color.fromCssColorString("#e4fbff").withAlpha(0.34 + Math.max(0, Math.cos(nowSeconds() * 1.8 + phaseSeed)) * 0.24),
                false,
              ),
            },
            label: {
              text: `${depthCm} 厘米`,
              showBackground: true,
              backgroundColor: Color.fromCssColorString("rgba(8,20,37,0.76)"),
              fillColor: Color.WHITE,
              font: "700 11px 'Segoe UI'",
              pixelOffset: new Cartesian2(0, -18),
              verticalOrigin: VerticalOrigin.BOTTOM,
              horizontalOrigin: HorizontalOrigin.CENTER,
            },
          });
          const buildingHighlight = viewer.entities.add({
            id: `${layer.object_id}:building-highlight`,
            name: `${layer.name} building highlight`,
            show: false,
            position: Matrix4.multiplyByPoint(
              anchorFrame,
              new Cartesian3(
                layer.east_offset_m,
                layer.north_offset_m,
                layer.height_offset_m + highlightSize.height / 2 + 6,
              ),
              new Cartesian3(),
            ),
            box: {
              dimensions: new Cartesian3(highlightSize.width, highlightSize.depth, highlightSize.height),
              material: new ColorMaterialProperty(
                new CallbackProperty(
                  () =>
                    Color.fromCssColorString("#24d8ff").withAlpha(
                      0.18 + Math.max(0, Math.sin(nowSeconds() * 2.1 + phaseSeed)) * 0.12,
                    ),
                  false,
                ),
              ),
              outline: true,
              outlineColor: new CallbackProperty(
                () => Color.fromCssColorString("#8af4ff").withAlpha(0.68 + Math.max(0, Math.cos(nowSeconds() * 1.9 + phaseSeed)) * 0.24),
                false,
              ),
            },
            label: {
              text: `已聚焦：${layer.name}`,
              showBackground: true,
              backgroundColor: Color.fromCssColorString("rgba(4,18,34,0.82)"),
              fillColor: Color.fromCssColorString("#dffbff"),
              font: "700 12px 'Segoe UI'",
              pixelOffset: new Cartesian2(0, -42),
              verticalOrigin: VerticalOrigin.BOTTOM,
              horizontalOrigin: HorizontalOrigin.CENTER,
            },
          });
          const pulse = viewer.entities.add({
            id: `${layer.object_id}:risk-pulse`,
            name: `${layer.name} 指挥脉冲`,
            position,
            ellipse: {
              semiMajorAxis: radius * 1.22,
              semiMinorAxis: radius * 0.9,
              material: color.withAlpha(layer.proposal_state === "monitoring" ? 0.04 : 0.1),
              outline: layer.proposal_state !== "monitoring",
              outlineColor: color.withAlpha(0.62),
            },
          });
          const roadClosure =
            layer.proposal_state !== "monitoring" || layer.risk_level === "Red" || layer.risk_level === "Orange"
              ? viewer.entities.add({
                  id: `${layer.object_id}:road-closure`,
                  name: `${layer.name} 道路阻断线`,
                  polyline: {
                    positions: closurePositions,
                    width: layer.risk_level === "Red" ? 7 : 5,
                    material: new PolylineGlowMaterialProperty({
                      glowPower: 0.22,
                      color: Color.fromCssColorString("#ff6b2d").withAlpha(0.88),
                    }),
                  },
                })
              : undefined;
          const warningRings =
            layer.proposal_state === "warning_generated"
              ? [0, 1, 2].map((ringIndex) =>
                  viewer.entities.add({
                    id: `${layer.object_id}:warning-spread-${ringIndex + 1}`,
                    name: `${layer.name} 预警扩散圈 ${ringIndex + 1}`,
                    position,
                    ellipse: {
                      semiMajorAxis: new CallbackProperty(() => {
                        const cycle = (nowSeconds() * 0.42 + ringIndex / 3 + phaseSeed * 0.03) % 1;
                        return radius * (1.24 + cycle * (1.28 + intensity * 0.16));
                      }, false),
                      semiMinorAxis: new CallbackProperty(() => {
                        const cycle = (nowSeconds() * 0.42 + ringIndex / 3 + phaseSeed * 0.03) % 1;
                        return radius * (0.86 + cycle * (0.92 + intensity * 0.12));
                      }, false),
                      material: new ColorMaterialProperty(
                        new CallbackProperty(() => {
                          const cycle = (nowSeconds() * 0.42 + ringIndex / 3 + phaseSeed * 0.03) % 1;
                          return Color.fromCssColorString("#ff8a2a").withAlpha(Math.max(0.02, 0.18 * (1 - cycle)));
                        }, false),
                      ),
                      outline: true,
                      outlineColor: new CallbackProperty(() => {
                        const cycle = (nowSeconds() * 0.42 + ringIndex / 3 + phaseSeed * 0.03) % 1;
                        return Color.fromCssColorString("#ffc06a").withAlpha(Math.max(0.12, 0.78 * (1 - cycle)));
                      }, false),
                    },
                  }),
                )
              : [];
          const warningSpread = warningRings[0];
          const marker = viewer.entities.add({
            id: layer.object_id,
            name: layer.name,
            position,
            point: {
              pixelSize: layer.proposal_state === "warning_generated" ? 20 : layer.is_lead ? 18 : 14,
              color,
              outlineColor: Color.fromCssColorString("#081425"),
              outlineWidth: layer.proposal_state === "approved" || layer.proposal_state === "warning_generated" ? 4 : 3,
            },
            label: {
              text: `${layer.name}\n${stateLabel(layer.proposal_state)}`,
              showBackground: true,
              backgroundColor: Color.fromCssColorString("rgba(8,20,37,0.78)"),
              fillColor: Color.WHITE,
              font: "600 12px 'Segoe UI'",
              pixelOffset: new Cartesian2(0, -28),
              verticalOrigin: VerticalOrigin.BOTTOM,
              horizontalOrigin: HorizontalOrigin.CENTER,
            },
          });
            entityMapRef.current.set(layer.object_id, {
              marker,
              heat,
              pulse,
              water,
              waterColumn,
              buildingHighlight,
              roadClosure,
              warningSpread,
              warningRings,
            });
          } catch (overlayError) {
            console.warn("数字孪生对象叠加层创建失败，已跳过该对象", layer.object_id, overlayError);
          }
        }

        const leadPosition = leadLayer ? layerPositions.get(leadLayer.object_id) : undefined;
        if (leadPosition) {
          for (const layer of layers) {
            const targetPosition = layerPositions.get(layer.object_id);
            if (!targetPosition) {
              continue;
            }
            try {
            const color = Color.fromCssColorString(stateColor(layer.proposal_state));
            const routeTarget =
              layer.object_id === leadLayer?.object_id
                ? {
                    ...layer,
                    east_offset_m: layer.east_offset_m + 260,
                    north_offset_m: layer.north_offset_m + 150,
                    height_offset_m: layer.height_offset_m,
                  }
                : layer;
            const routeSource = leadLayer ?? layer;
            const routeOffsets = buildRoadFollowingRoute(routeSource, routeTarget);
            if (routeOffsets.length < 2) {
              continue;
            }
            const routePositions = routeOffsets.map((point) =>
              Matrix4.multiplyByPoint(
                anchorFrame,
                new Cartesian3(point.east, point.north, point.height),
                new Cartesian3(),
              ),
            );
            const vehicleOffset = routeOffsets[Math.max(1, Math.floor(routeOffsets.length / 2))] ?? routeOffsets[0];
            const route = viewer.entities.add({
              id: `${layer.object_id}:command-link`,
              name: `${layer.name} command link`,
              polyline: {
                positions: routePositions,
                width: layer.proposal_state === "monitoring" ? 4 : 10,
                material: new PolylineGlowMaterialProperty({
                  glowPower: layer.proposal_state === "monitoring" ? 0.28 : 0.56,
                  color: Color.fromCssColorString(layer.proposal_state === "approved" ? "#35e6bf" : "#ffb13b").withAlpha(
                    layer.proposal_state === "monitoring" ? 0.42 : 0.92,
                  ),
                }),
              },
            });
            const evacuationRoute = viewer.entities.add({
              id: `${layer.object_id}:evacuation-route`,
              name: `${layer.name} evacuation route arrow`,
              show: layer.proposal_state !== "monitoring",
              polyline: {
                positions: routePositions,
                width: 11,
                material: new PolylineArrowMaterialProperty(
                  Color.fromCssColorString(layer.proposal_state === "approved" ? "#43ffd0" : "#ffc247").withAlpha(0.98),
                ),
              },
            });
            const resourceVehicle =
              layer.proposal_state === "pending" || layer.proposal_state === "approved" || layer.proposal_state === "warning_generated"
                ? viewer.entities.add({
                    id: `${layer.object_id}:resource-vehicle`,
                    name: `${layer.name} resource vehicle`,
                    position: Matrix4.multiplyByPoint(anchorFrame, new Cartesian3(vehicleOffset.east, vehicleOffset.north, vehicleOffset.height + 6), new Cartesian3()),
                    point: {
                      pixelSize: 11,
                      color: Color.fromCssColorString("#f4fbff"),
                      outlineColor: color,
                      outlineWidth: 4,
                    },
                    label: {
                      text: layer.proposal_state === "approved" ? "救援车队" : "资源单元",
                      showBackground: true,
                      backgroundColor: Color.fromCssColorString("rgba(8,20,37,0.78)"),
                      fillColor: Color.WHITE,
                      font: "700 11px 'Segoe UI'",
                      pixelOffset: new Cartesian2(0, -20),
                      verticalOrigin: VerticalOrigin.BOTTOM,
                      horizontalOrigin: HorizontalOrigin.CENTER,
                    },
                  })
                : undefined;
            const bundle = entityMapRef.current.get(layer.object_id);
            if (bundle) {
              bundle.route = route;
              bundle.evacuationRoute = evacuationRoute;
              bundle.resourceVehicle = resourceVehicle;
            }
            } catch (routeError) {
              console.warn("数字孪生处置路线创建失败，已跳过该路线", layer.object_id, routeError);
            }
          }
        }

        const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
        handlerRef.current = handler;
        handler.setInputAction((movement: { position: any }) => {
          const picked = viewer.scene.pick(movement.position) as { id?: { id?: string } } | undefined;
          const objectId = extractObjectId(String(picked?.id?.id ?? ""));
          if (defined(picked) && objectId) {
            onSelectObjectRef.current(objectId);
          }
        }, ScreenSpaceEventType.LEFT_CLICK);
        handler.setInputAction((movement: { endPosition: any }) => {
          const picked = viewer.scene.pick(movement.endPosition) as { id?: { id?: string } } | undefined;
          const objectId = extractObjectId(String(picked?.id?.id ?? ""));
          if (defined(picked) && objectId) {
            setHoveredObjectId(objectId);
            return;
          }
          setHoveredObjectId(null);
        }, ScreenSpaceEventType.MOUSE_MOVE);

        const preset = sceneConfig.cameraPresets?.overview ?? { heading: 20, pitch: -42, range: 2400, fitWholeModel: true };
        const cameraTarget =
          sourceMetadata && preset.fitWholeModel !== false
            ? computeModelFocusSphere(Cesium, sceneConfig, sourceMetadata, anchorHeight)
            : new BoundingSphere(anchor, Math.max(preset.focusRadius ?? 1600, 1600));
        sceneFocusRef.current = cameraTarget;
        viewer.camera.flyToBoundingSphere(cameraTarget, {
          duration: 0,
          offset: new HeadingPitchRange(
            CesiumMath.toRadians(preset.heading),
            CesiumMath.toRadians(preset.pitch),
            preset.range,
          ),
        });

        viewer.scene.requestRender();
        setStatus("数字孪生画布运行中");
        setReady(true);

        if (disposed) {
          handler.destroy();
          viewer.destroy();
        }
      } catch (caught) {
        console.error("数字孪生三维画布初始化失败", caught);
        setError(formatCanvasError(caught));
        setStatus("数字孪生画布已切换为降级模式");
      }
    }

    void initViewer();
    return () => {
      disposed = true;
      handlerRef.current?.destroy?.();
      handlerRef.current = null;
      viewerRef.current?.destroy();
      viewerRef.current = null;
      entityMapRef.current.clear();
      for (const timer of tourTimersRef.current) {
        window.clearTimeout(timer);
      }
      tourTimersRef.current = [];
      sceneFocusRef.current = null;
    };
  }, [layers]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) {
      return;
    }
    const Cesium = cesiumRef.current;

    entityMapRef.current.forEach((bundle, objectId) => {
      const layer = layerById.get(objectId);
      const basePixelSize = layer?.proposal_state === "warning_generated" ? 20 : layer?.is_lead ? 18 : 14;
      const baseOutlineWidth =
        layer?.proposal_state === "approved" || layer?.proposal_state === "warning_generated" ? 4 : 3;
      const dialogFocused = objectId === dialogFocusObjectId;
      const focused = dialogFocused || objectId === hoveredObjectId;
      const routeGuided = objectId === routeHighlightObjectId;
      const routeStory = activeNarrativeStep === "route";
      const closureStory = activeNarrativeStep === "closure";
      if (!bundle.marker?.point) {
        return;
      }
      bundle.marker.point.outlineWidth = focused || (closureStory && layer?.proposal_state !== "monitoring") ? baseOutlineWidth + 2 : baseOutlineWidth;
      bundle.marker.point.pixelSize = focused || (closureStory && layer?.proposal_state !== "monitoring") ? basePixelSize + 5 : basePixelSize;
      if (bundle.pulse?.ellipse) {
        bundle.pulse.show = focused || routeGuided || layer?.proposal_state !== "monitoring";
      }
      if (bundle.buildingHighlight?.box) {
        bundle.buildingHighlight.show = dialogFocused;
      }
      if (Cesium && layer && bundle.route?.polyline) {
        const routeFocused = focused || routeGuided || (routeStory && layer.proposal_state !== "monitoring");
        const color = Cesium.Color.fromCssColorString(
          routeGuided || layer.proposal_state !== "approved" ? "#ffb13b" : "#35e6bf",
        );
        bundle.route.polyline.width = routeGuided ? 18 : routeFocused ? 15 : layer.proposal_state === "monitoring" ? 5 : 11;
        bundle.route.polyline.material = new Cesium.PolylineGlowMaterialProperty({
          glowPower: routeGuided ? 0.68 : routeFocused ? 0.58 : 0.42,
          color: color.withAlpha(routeStory && !routeFocused ? 0.62 : routeFocused ? 0.98 : 0.82),
        });
      }
      if (bundle.evacuationRoute?.polyline) {
        const routeFocused = focused || routeGuided || routeStory;
        bundle.evacuationRoute.polyline.width = routeGuided ? 17 : routeFocused ? 14 : 11;
        if (Cesium) {
          bundle.evacuationRoute.polyline.material = new Cesium.PolylineArrowMaterialProperty(
            Cesium.Color.fromCssColorString(routeGuided ? "#ffd166" : layer?.proposal_state === "approved" ? "#43ffd0" : "#ffc247").withAlpha(
              routeGuided ? 1 : 0.96,
            ),
          );
        }
        bundle.evacuationRoute.show = routeGuided || routeStory || layer?.proposal_state !== "monitoring";
      }
      if (bundle.resourceVehicle?.point) {
        const routeFocused = focused || routeGuided || routeStory;
        bundle.resourceVehicle.point.pixelSize = routeFocused ? 15 : 11;
      }
      if (bundle.roadClosure?.polyline && Cesium) {
        bundle.roadClosure.polyline.width = focused || closureStory ? 8 : layer?.risk_level === "Red" ? 7 : 5;
      }
      if (bundle.warningRings?.length) {
        bundle.warningRings.forEach((ring) => {
          ring.show = focused || layer?.proposal_state === "warning_generated";
        });
      }
    });

    const dialogFocusEntity = dialogFocusObjectId
      ? entityMapRef.current.get(dialogFocusObjectId)?.buildingHighlight ??
        entityMapRef.current.get(dialogFocusObjectId)?.waterColumn ??
        entityMapRef.current.get(dialogFocusObjectId)?.marker
      : undefined;
    const dialogFocusKey = dialogFocusObjectId ? `${dialogFocusObjectId}:${dialogFocusSerial}` : null;
    if (dialogFocusEntity && !tourRunning && lastDialogFocusRef.current !== dialogFocusKey) {
      lastDialogFocusRef.current = dialogFocusKey;
      const layer = dialogFocusObjectId ? layerById.get(dialogFocusObjectId) : null;
      setTourNarrative(layer ? `已根据问答定位到 ${layer.name}，空间对象进入高亮研判。` : "已根据问答定位到目标对象。");
      viewer.flyTo(dialogFocusEntity, { duration: 1.05 });
    }
    viewer.scene.requestRender();
  }, [activeNarrativeStep, dialogFocusObjectId, dialogFocusSerial, hoveredObjectId, layerById, routeHighlightObjectId, tourRunning]);

  if (isJsdomRuntime()) {
    return (
      <div className={styles.fallbackShell} aria-label="digital-twin-canvas">
        <div className={styles.fallbackGrid} />
        <div className={styles.fallbackNarrative} aria-label="narrative-camera-controls">
          <button type="button" className={styles.narrativePlayButton} onClick={runCommandFlythrough}>
            播放指挥镜头
          </button>
          <div className={styles.narrativeStepGrid}>
            {NARRATIVE_STEPS.map((step) => (
              <button
                key={step.key}
                type="button"
                className={`${styles.narrativeStepButton} ${
                  activeNarrativeStep === step.key ? styles.narrativeStepActive : ""
                }`}
                onClick={() => flyToNarrativeStep(step.key)}
              >
                <span>{step.shortLabel}</span>
                {step.title}
              </button>
            ))}
          </div>
        </div>
        {layers.map((layer) => (
          <div
            key={layer.object_id}
            className={styles.fallbackObject}
            style={
              {
                "--node-color": stateColor(layer.proposal_state),
                left: `${18 + (layer.east_offset_m + 340) / 9}%`,
                top: `${22 + (260 - layer.north_offset_m) / 8}%`,
              } as CSSProperties
            }
          >
            <span className={styles.fallbackHalo} />
            <button
              type="button"
              className={`${styles.fallbackNode} ${
                dialogFocusObjectId === layer.object_id || routeHighlightObjectId === layer.object_id ? styles.fallbackNodeActive : ""
              }`}
              onClick={() => onSelectObjectRef.current(layer.object_id)}
            >
              {`${layer.name} / ${stateLabel(layer.proposal_state)}`}
            </button>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={`${styles.canvasShell} ${toneClass}`} aria-label="digital-twin-canvas">
      <div ref={hostRef} className={styles.viewerHost} />
      <div className={styles.overlayActions}>
        <div className={styles.narrativeTopline}>
          <button
            type="button"
            className={styles.narrativePlayButton}
            onClick={runCommandFlythrough}
            disabled={!ready || tourRunning}
          >
            {tourRunning ? "镜头播放中" : "播放指挥镜头"}
          </button>
          <span>
            镜头 {Math.max(1, narrativeStepIndex + 1)} / {NARRATIVE_STEPS.length}
          </span>
        </div>
        <div className={styles.narrativeStepGrid} aria-label="narrative-camera-controls">
          {NARRATIVE_STEPS.map((step) => (
            <button
              key={step.key}
              type="button"
              className={`${styles.narrativeStepButton} ${
                activeNarrativeStep === step.key ? styles.narrativeStepActive : ""
              }`}
              onClick={() => flyToNarrativeStep(step.key)}
              disabled={!ready || tourRunning}
            >
              <span>{step.shortLabel}</span>
              {step.title}
            </button>
          ))}
        </div>
        <p>{tourNarrative}</p>
        <div className={styles.tacticalLegend} aria-label="tactical-map-legend">
          <span className={styles.tacticalLegendItem}>
            <i className={styles.tacticalSwatchWater} />
            积水
          </span>
          <span className={styles.tacticalLegendItem}>
            <i className={styles.tacticalSwatchRoute} />
            路线
          </span>
          <span className={styles.tacticalLegendItem}>
            <i className={styles.tacticalSwatchWarning} />
            预警
          </span>
          <span className={styles.tacticalLegendItem}>
            <i className={styles.tacticalSwatchClosure} />
            闭环
          </span>
        </div>
      </div>
      <div className={styles.overlayLegend}>
        <div className={styles.legendRow}>
          <span className={styles.legendDotMonitoring} />
          <strong>{layerStats.monitoring}</strong>
          <small>监测中</small>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendDotPending} />
          <strong>{layerStats.pending}</strong>
          <small>待审批</small>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendDotApproved} />
          <strong>{layerStats.approved}</strong>
          <small>已批准</small>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendDotWarning} />
          <strong>{layerStats.warningGenerated}</strong>
          <small>已预警</small>
        </div>
      </div>
      {error ? (
        <div className={styles.errorPanel}>
          <strong>三维画布已降级</strong>
          <p>{error}</p>
        </div>
      ) : null}
      {!ready && !error ? (
        <div className={styles.loadingPanel}>
          <strong>数字孪生画布</strong>
          <p>{status}</p>
        </div>
      ) : null}
    </div>
  );
}
