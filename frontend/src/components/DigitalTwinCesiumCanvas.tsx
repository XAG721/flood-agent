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
  roadClosure?: any;
  evacuationRoute?: any;
  resourceVehicle?: any;
  warningSpread?: any;
  warningRings?: any[];
};

type NarrativeStepKey = "overview" | "risk" | "water" | "route" | "closure" | "warning";

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
    title: "City overview",
    narrative: "Full-city flood posture: model, heat zones, signals and command focus.",
  },
  {
    key: "risk",
    shortLabel: "02",
    title: "Risk source",
    narrative: "Locate the highest-risk source and its upstream impact trigger.",
  },
  {
    key: "water",
    shortLabel: "03",
    title: "Water posture",
    narrative: "Read the animated inundation surface and water-level column around the focus object.",
  },
  {
    key: "route",
    shortLabel: "04",
    title: "Response route",
    narrative: "Highlight evacuation paths, blocked roads and resource vehicle movement.",
  },
  {
    key: "closure",
    shortLabel: "05",
    title: "Closure loop",
    narrative: "Focus on proposal approval status and the spatial object where the loop closes.",
  },
  {
    key: "warning",
    shortLabel: "06",
    title: "Warning spread",
    narrative: "Show warning diffusion rings after approved actions produce audience drafts.",
  },
];

interface DigitalTwinCesiumCanvasProps {
  eventTitle?: string;
  layers: TwinObjectMapLayer[];
  selectedObjectId?: string | null;
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
    monitoring: "Monitoring",
    pending: "Pending proposal",
    approved: "Approved action",
    warning_generated: "Warnings ready",
  }[normalizedState] ?? normalizedState;
}

function stateColor(proposalState?: string | null) {
  const normalizedState = proposalState ?? "monitoring";
  return {
    monitoring: "#5cc8ff",
    pending: "#ffad42",
    approved: "#9be36f",
    warning_generated: "#f66f9d",
  }[normalizedState] ?? "#5cc8ff";
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
    None: 8,
    Blue: 18,
    Yellow: 34,
    Orange: 58,
    Red: 86,
  }[riskLevel ?? "None"];
  return base + (proposalState === "pending" ? 12 : proposalState === "warning_generated" ? 18 : 0);
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

export function DigitalTwinCesiumCanvas({
  eventTitle,
  layers,
  selectedObjectId,
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
  const [status, setStatus] = useState("Loading digital twin canvas");
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [tourRunning, setTourRunning] = useState(false);
  const [tourNarrative, setTourNarrative] = useState("Ready for command flythrough");
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [activeNarrativeStep, setActiveNarrativeStep] = useState<NarrativeStepKey>("overview");

  const leadLayer = useMemo(
    () => layers.find((item) => item.object_id === selectedObjectId) ?? layers.find((item) => item.is_lead) ?? layers[0] ?? null,
    [layers, selectedObjectId],
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
  const spotlightLayer =
    (hoveredObjectId ? layerById.get(hoveredObjectId) : undefined) ??
    (selectedObjectId ? layerById.get(selectedObjectId) : undefined) ??
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
    const focusLayer = selectedObjectId ? layerById.get(selectedObjectId) ?? riskSource : riskSource;
    const routeLayer =
      layers.find((item) => item.object_id !== riskSource?.object_id && item.proposal_state === "approved") ??
      layers.find((item) => item.object_id !== riskSource?.object_id && item.proposal_state === "pending") ??
      layers.find((item) => item.object_id !== riskSource?.object_id) ??
      riskSource;
    const closureLayer =
      layers.find((item) => item.proposal_state === "pending") ??
      layers.find((item) => item.proposal_state === "approved") ??
      focusLayer;
    const warningLayer =
      layers.find((item) => item.proposal_state === "warning_generated") ??
      layers.find((item) => item.proposal_state === "approved") ??
      closureLayer;

    return {
      overview: null,
      risk: riskSource,
      water: focusLayer,
      route: routeLayer,
      closure: closureLayer,
      warning: warningLayer,
    } satisfies Record<NarrativeStepKey, TwinObjectMapLayer | null>;
  }, [layerById, layers, leadLayer, selectedObjectId]);

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
    return {
      layer,
      entity: bundle?.warningSpread ?? bundle?.warningRings?.[0] ?? bundle?.marker,
      kind: "warning" as const,
    };
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
      onSelectObject(target.layer.object_id);
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
        Cesium.Math.toRadians(target.kind === "warning" ? -36 : -28),
        target.kind === "warning" ? 680 : target.kind === "route" ? 740 : 430,
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
        flyToNarrativeStep(step.key, { duration: step.key === "warning" ? 1.3 : 1.15 });
      }, (index + 1) * 1600);
      tourTimersRef.current.push(timer);
    });
    const stopTimer = window.setTimeout(() => {
      setTourRunning(false);
      setTourNarrative("Command story complete: impact, route, approval and warning loop are aligned.");
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
        setStatus("Loading 3D scene configuration");

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
          if (sampled[0]?.height !== undefined) {
            anchorHeight = sampled[0].height + sceneConfig.anchorHeight;
          }
        }

        if (disposed || !hostRef.current) {
          return;
        }

        setStatus("Starting Cesium viewer");
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

        setStatus("Analyzing CityEngine source coordinates");
        let sourceMetadata: SourceMetadata | null = null;
        try {
          sourceMetadata = await loadSourceMetadata(Cesium, sceneConfig);
        } catch (metadataError) {
          console.warn("Falling back to simple CityEngine placement", metadataError);
        }

        const anchor = Cartesian3.fromDegrees(sceneConfig.anchorLon, sceneConfig.anchorLat, anchorHeight);
        const anchorFrame = Transforms.eastNorthUpToFixedFrame(anchor);
        const modelMatrix = sourceMetadata
          ? buildCalibratedModelMatrix(Cesium, sceneConfig, sourceMetadata, anchorHeight)
          : buildSimpleModelMatrix(Cesium, sceneConfig, anchorHeight);

        const model = await Model.fromGltfAsync({
          url: resolveModelAssetUrl(sceneConfig.modelUrl),
          modelMatrix,
          color: Color.WHITE.withAlpha(0.96),
          silhouetteColor: Color.fromCssColorString("#ff9a4a"),
          silhouetteSize: 0,
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
          const position = layerPositions.get(layer.object_id);
          const color = Color.fromCssColorString(stateColor(layer.proposal_state));
          const radius = riskRadius(layer.risk_level, layer.proposal_state);
          const depthCm = waterDepthCm(layer.risk_level, layer.proposal_state);
          const columnHeight = Math.max(24, depthCm * 0.95);
          const intensity = riskMotionIntensity(layer.risk_level, layer.proposal_state);
          const phaseSeed = objectPhaseSeed(layer.object_id);
          const closureStart = Matrix4.multiplyByPoint(
            anchorFrame,
            new Cartesian3(layer.east_offset_m - radius * 0.62, layer.north_offset_m - radius * 0.18, layer.height_offset_m + 6),
            new Cartesian3(),
          );
          const closureEnd = Matrix4.multiplyByPoint(
            anchorFrame,
            new Cartesian3(layer.east_offset_m + radius * 0.62, layer.north_offset_m + radius * 0.18, layer.height_offset_m + 6),
            new Cartesian3(),
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
                    Color.fromCssColorString("#48c8ff").withAlpha(
                      (layer.risk_level === "Red" || layer.risk_level === "Orange" ? 0.26 : 0.13) +
                        Math.max(0, Math.sin(nowSeconds() * 1.35 + phaseSeed)) * 0.06 * intensity,
                    ),
                  false,
                ),
              ),
              outline: true,
              outlineColor: new CallbackProperty(
                () => Color.fromCssColorString("#a9edff").withAlpha(0.22 + Math.max(0, Math.sin(nowSeconds() * 1.2 + phaseSeed)) * 0.2),
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
                  () => Color.fromCssColorString("#6fe5ff").withAlpha(0.36 + Math.max(0, Math.sin(nowSeconds() * 2.05 + phaseSeed)) * 0.18),
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
              text: `${depthCm}cm`,
              showBackground: true,
              backgroundColor: Color.fromCssColorString("rgba(8,20,37,0.76)"),
              fillColor: Color.WHITE,
              font: "700 11px 'Segoe UI'",
              pixelOffset: new Cartesian2(0, -18),
              verticalOrigin: VerticalOrigin.BOTTOM,
              horizontalOrigin: HorizontalOrigin.CENTER,
            },
          });
          const pulse = viewer.entities.add({
            id: `${layer.object_id}:risk-pulse`,
            name: `${layer.name} command pulse`,
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
                  name: `${layer.name} road closure line`,
                  polyline: {
                    positions: [closureStart, closureEnd],
                    width: layer.risk_level === "Red" ? 7 : 5,
                    material: new PolylineGlowMaterialProperty({
                      glowPower: 0.22,
                      color: Color.fromCssColorString("#ff5b52").withAlpha(0.86),
                    }),
                  },
                })
              : undefined;
          const warningRings =
            layer.proposal_state === "warning_generated"
              ? [0, 1, 2].map((ringIndex) =>
                  viewer.entities.add({
                    id: `${layer.object_id}:warning-spread-${ringIndex + 1}`,
                    name: `${layer.name} warning spread ring ${ringIndex + 1}`,
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
                          return Color.fromCssColorString("#ff5f9f").withAlpha(Math.max(0.02, 0.14 * (1 - cycle)));
                        }, false),
                      ),
                      outline: true,
                      outlineColor: new CallbackProperty(() => {
                        const cycle = (nowSeconds() * 0.42 + ringIndex / 3 + phaseSeed * 0.03) % 1;
                        return Color.fromCssColorString("#ff9fc4").withAlpha(Math.max(0.12, 0.72 * (1 - cycle)));
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
            roadClosure,
            warningSpread,
            warningRings,
          });
        }

        const leadPosition = leadLayer ? layerPositions.get(leadLayer.object_id) : undefined;
        if (leadPosition) {
          for (const layer of layers) {
            if (layer.object_id === leadLayer?.object_id) {
              continue;
            }
            const targetPosition = layerPositions.get(layer.object_id);
            if (!targetPosition) {
              continue;
            }
            const color = Color.fromCssColorString(stateColor(layer.proposal_state));
            const routePositions = [leadPosition, targetPosition];
            const route = viewer.entities.add({
              id: `${layer.object_id}:command-link`,
              name: `${layer.name} command link`,
              polyline: {
                positions: routePositions,
                width: layer.proposal_state === "monitoring" ? 2 : 4,
                material: new PolylineGlowMaterialProperty({
                  glowPower: layer.proposal_state === "monitoring" ? 0.12 : 0.28,
                  color: color.withAlpha(layer.proposal_state === "monitoring" ? 0.28 : 0.74),
                }),
              },
            });
            const evacuationRoute =
              layer.proposal_state !== "monitoring"
                ? viewer.entities.add({
                    id: `${layer.object_id}:evacuation-route`,
                    name: `${layer.name} evacuation route arrow`,
                    polyline: {
                      positions: routePositions,
                      width: 7,
                      material: new PolylineArrowMaterialProperty(
                        Color.fromCssColorString(layer.proposal_state === "approved" ? "#9be36f" : "#ffd166").withAlpha(0.82),
                      ),
                    },
                  })
                : undefined;
            const resourceVehicle =
              layer.proposal_state === "pending" || layer.proposal_state === "approved" || layer.proposal_state === "warning_generated"
                ? viewer.entities.add({
                    id: `${layer.object_id}:resource-vehicle`,
                    name: `${layer.name} resource vehicle`,
                    position: Matrix4.multiplyByPoint(
                      anchorFrame,
                      new Cartesian3(
                        (layer.east_offset_m + (leadLayer?.east_offset_m ?? 0)) / 2 + 24,
                        (layer.north_offset_m + (leadLayer?.north_offset_m ?? 0)) / 2 - 18,
                        Math.max(layer.height_offset_m, leadLayer?.height_offset_m ?? 0) + 10,
                      ),
                      new Cartesian3(),
                    ),
                    point: {
                      pixelSize: 11,
                      color: Color.fromCssColorString("#f4fbff"),
                      outlineColor: color,
                      outlineWidth: 4,
                    },
                    label: {
                      text: layer.proposal_state === "approved" ? "Rescue convoy" : "Resource unit",
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
          }
        }

        const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
        handlerRef.current = handler;
        handler.setInputAction((movement: { position: any }) => {
          const picked = viewer.scene.pick(movement.position) as { id?: { id?: string } } | undefined;
          const objectId = extractObjectId(String(picked?.id?.id ?? ""));
          if (defined(picked) && objectId) {
            onSelectObject(objectId);
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
            : new BoundingSphere(anchor, 1600);
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
        setStatus("Digital twin canvas is live");
        setReady(true);

        if (disposed) {
          handler.destroy();
          viewer.destroy();
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Unknown Cesium error");
        setStatus("Digital twin canvas switched to fallback mode");
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
  }, [layers, onSelectObject]);

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
      const focused = objectId === selectedObjectId || objectId === hoveredObjectId;
      const routeStory = activeNarrativeStep === "route";
      const closureStory = activeNarrativeStep === "closure";
      const warningStory = activeNarrativeStep === "warning";
      if (!bundle.marker?.point) {
        return;
      }
      bundle.marker.point.outlineWidth = focused || (closureStory && layer?.proposal_state !== "monitoring") ? baseOutlineWidth + 2 : baseOutlineWidth;
      bundle.marker.point.pixelSize = focused || (closureStory && layer?.proposal_state !== "monitoring") ? basePixelSize + 5 : basePixelSize;
      if (bundle.pulse?.ellipse) {
        bundle.pulse.show = focused || layer?.proposal_state !== "monitoring" || warningStory;
      }
      if (Cesium && layer && bundle.route?.polyline) {
        const routeFocused = focused || (routeStory && layer.proposal_state !== "monitoring");
        const color = Cesium.Color.fromCssColorString(stateColor(layer.proposal_state));
        bundle.route.polyline.width = routeFocused ? 7 : layer.proposal_state === "monitoring" ? 2 : 4;
        bundle.route.polyline.material = new Cesium.PolylineGlowMaterialProperty({
          glowPower: routeFocused ? 0.34 : 0.14,
          color: color.withAlpha(routeStory && !routeFocused ? 0.18 : routeFocused ? 0.86 : 0.46),
        });
      }
      if (bundle.evacuationRoute?.polyline) {
        const routeFocused = focused || routeStory;
        bundle.evacuationRoute.polyline.width = routeFocused ? 10 : 7;
        bundle.evacuationRoute.show = !routeStory || routeFocused || layer?.proposal_state !== "monitoring";
      }
      if (bundle.resourceVehicle?.point) {
        const routeFocused = focused || routeStory;
        bundle.resourceVehicle.point.pixelSize = routeFocused ? 15 : 11;
      }
      if (bundle.roadClosure?.polyline && Cesium) {
        bundle.roadClosure.polyline.width = focused || closureStory ? 8 : layer?.risk_level === "Red" ? 7 : 5;
      }
      if (bundle.warningRings?.length) {
        bundle.warningRings.forEach((ring) => {
          ring.show = warningStory || focused || layer?.proposal_state === "warning_generated";
        });
      }
    });

    const selectedEntity = selectedObjectId ? entityMapRef.current.get(selectedObjectId)?.marker : undefined;
    if (selectedEntity && !tourRunning) {
      viewer.flyTo(selectedEntity, { duration: 0.9 });
    }
    viewer.scene.requestRender();
  }, [activeNarrativeStep, hoveredObjectId, layerById, selectedObjectId, tourRunning]);

  if (isJsdomRuntime()) {
    return (
      <div className={styles.fallbackShell} aria-label="digital-twin-canvas">
        <div className={styles.fallbackGrid} />
        <div className={styles.fallbackNarrative} aria-label="narrative-camera-controls">
          <button type="button" className={styles.narrativePlayButton} onClick={runCommandFlythrough}>
            Play command story
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
              className={`${styles.fallbackNode} ${selectedObjectId === layer.object_id ? styles.fallbackNodeActive : ""}`}
              onClick={() => onSelectObject(layer.object_id)}
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
      <div className={styles.overlayHud}>
        <div>
          <span className={styles.hudLabel}>Twin Status</span>
          <strong>{status}</strong>
        </div>
        <div>
          <span className={styles.hudLabel}>Current Focus</span>
          <strong>{leadLayer?.name ?? eventTitle ?? "Waiting for focus"}</strong>
        </div>
        <div>
          <span className={styles.hudLabel}>Objects</span>
          <strong>{layers.length}</strong>
        </div>
      </div>
      <div className={styles.overlayActions}>
        <div className={styles.narrativeTopline}>
          <button
            type="button"
            className={styles.narrativePlayButton}
            onClick={runCommandFlythrough}
            disabled={!ready || tourRunning}
          >
            {tourRunning ? "Story running" : "Play command story"}
          </button>
          <span>
            Step {Math.max(1, narrativeStepIndex + 1)} / {NARRATIVE_STEPS.length}
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
            Water
          </span>
          <span className={styles.tacticalLegendItem}>
            <i className={styles.tacticalSwatchRoute} />
            Route
          </span>
          <span className={styles.tacticalLegendItem}>
            <i className={styles.tacticalSwatchWarning} />
            Warning
          </span>
          <span className={styles.tacticalLegendItem}>
            <i className={styles.tacticalSwatchClosure} />
            Closure
          </span>
        </div>
      </div>
      <div className={styles.overlayLegend}>
        <div className={styles.legendRow}>
          <span className={styles.legendDotMonitoring} />
          <strong>{layerStats.monitoring}</strong>
          <small>Monitoring</small>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendDotPending} />
          <strong>{layerStats.pending}</strong>
          <small>Pending</small>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendDotApproved} />
          <strong>{layerStats.approved}</strong>
          <small>Approved</small>
        </div>
        <div className={styles.legendRow}>
          <span className={styles.legendDotWarning} />
          <strong>{layerStats.warningGenerated}</strong>
          <small>Warnings</small>
        </div>
      </div>
      {spotlightLayer ? (
        <div className={`${styles.spotlightPanel} ${toneClass}`}>
          <div className={styles.spotlightHeader}>
            <span className={styles.hudLabel}>Spatial spotlight</span>
            <strong>{spotlightLayer.name}</strong>
          </div>
          <p className={styles.spotlightSummary}>
            {stateLabel(spotlightLayer.proposal_state)} / {spotlightLayer.entity_type} /{" "}
            {spotlightLayer.is_lead ? "Lead focus" : "Linked object"}
          </p>
          <div className={styles.spotlightMeta}>
            <span>{Math.round(spotlightLayer.east_offset_m)}m east</span>
            <span>{Math.round(spotlightLayer.north_offset_m)}m north</span>
            <span>{Math.round(spotlightLayer.height_offset_m)}m vertical</span>
          </div>
        </div>
      ) : null}
      {error ? (
        <div className={styles.errorPanel}>
          <strong>3D canvas degraded</strong>
          <p>{error}</p>
        </div>
      ) : null}
      {!ready && !error ? (
        <div className={styles.loadingPanel}>
          <strong>Digital twin canvas</strong>
          <p>{status}</p>
        </div>
      ) : null}
    </div>
  );
}
