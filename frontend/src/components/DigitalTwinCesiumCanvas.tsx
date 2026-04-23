import { useEffect, useMemo, useRef, useState } from "react";
import "cesium/Build/Cesium/Widgets/widgets.css";
import type { RiskLevel, TwinObjectMapLayer } from "../types/api";
import styles from "../styles/digital-twin-map.module.css";

type CesiumModule = typeof import("cesium");

interface SceneConfig {
  modelUrl: string;
  anchorLon: number;
  anchorLat: number;
  anchorHeight: number;
  scale: number;
  heading: number;
  pitch: number;
  roll: number;
  offsetEast: number;
  offsetNorth: number;
  offsetUp: number;
  cameraPresets?: {
    overview?: { heading: number; pitch: number; range: number };
  };
}

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

function stateLabel(proposalState: string) {
  return {
    monitoring: "Monitoring",
    pending: "Pending proposal",
    approved: "Approved action",
    warning_generated: "Warnings ready",
  }[proposalState] ?? proposalState;
}

function stateColor(proposalState: string) {
  return {
    monitoring: "#5cc8ff",
    pending: "#ffad42",
    approved: "#9be36f",
    warning_generated: "#f66f9d",
  }[proposalState] ?? "#5cc8ff";
}

function isJsdomRuntime() {
  return typeof navigator !== "undefined" && /jsdom/i.test(navigator.userAgent);
}

function normalizeConfig(raw: Partial<SceneConfig>): SceneConfig {
  return {
    modelUrl: raw.modelUrl ?? "/agent-twin-assets/models/cityengine_scene.glb",
    anchorLon: raw.anchorLon ?? 108.94921153512861,
    anchorLat: raw.anchorLat ?? 34.24624474240188,
    anchorHeight: raw.anchorHeight ?? 404.36,
    scale: raw.scale ?? 1,
    heading: raw.heading ?? 0,
    pitch: raw.pitch ?? 0,
    roll: raw.roll ?? 0,
    offsetEast: raw.offsetEast ?? 0,
    offsetNorth: raw.offsetNorth ?? 0,
    offsetUp: raw.offsetUp ?? 0,
    cameraPresets: raw.cameraPresets ?? {
      overview: { heading: 20, pitch: -42, range: 2400 },
    },
  };
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
  const entityMapRef = useRef<Map<string, { point?: { outlineWidth?: number; pixelSize?: number } }>>(new Map());
  const cesiumRef = useRef<CesiumModule | null>(null);
  const [status, setStatus] = useState("Loading digital twin canvas");
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);

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

        const sceneConfig = normalizeConfig((await response.json()) as Partial<SceneConfig>);
        const {
          BoundingSphere,
          Cartesian2,
          Cartesian3,
          Color,
          EllipsoidTerrainProvider,
          HeadingPitchRange,
          HeadingPitchRoll,
          HorizontalOrigin,
          Ion,
          IonWorldImageryStyle,
          Math: CesiumMath,
          Matrix4,
          Model,
          Quaternion,
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
        viewer.scene.requestRenderMode = true;
        if (imageryProvider) {
          viewer.imageryLayers.addImageryProvider(imageryProvider);
        }
        viewerRef.current = viewer;

        const anchor = Cartesian3.fromDegrees(sceneConfig.anchorLon, sceneConfig.anchorLat, anchorHeight);
        const anchorFrame = Transforms.eastNorthUpToFixedFrame(anchor);
        const modelTransform = Matrix4.fromTranslationQuaternionRotationScale(
          new Cartesian3(sceneConfig.offsetEast, sceneConfig.offsetNorth, sceneConfig.offsetUp),
          Quaternion.fromHeadingPitchRoll(
            new HeadingPitchRoll(
              CesiumMath.toRadians(sceneConfig.heading),
              CesiumMath.toRadians(sceneConfig.pitch),
              CesiumMath.toRadians(sceneConfig.roll),
            ),
          ),
          new Cartesian3(sceneConfig.scale, sceneConfig.scale, sceneConfig.scale),
        );
        const modelMatrix = Matrix4.multiply(anchorFrame, modelTransform, new Matrix4());

        const model = await Model.fromGltfAsync({
          url: sceneConfig.modelUrl.replace("/models/", "/agent-twin-assets/models/"),
          modelMatrix,
          color: Color.WHITE.withAlpha(0.96),
          silhouetteColor: Color.fromCssColorString("#ff9a4a"),
          silhouetteSize: 0,
        });
        if (!disposed) {
          viewer.scene.primitives.add(model);
        }

        entityMapRef.current.clear();
        for (const layer of layers) {
          const position = Matrix4.multiplyByPoint(
            anchorFrame,
            new Cartesian3(layer.east_offset_m, layer.north_offset_m, layer.height_offset_m),
            new Cartesian3(),
          );
          const entity = viewer.entities.add({
            id: layer.object_id,
            name: layer.name,
            position,
            point: {
              pixelSize: layer.proposal_state === "warning_generated" ? 20 : layer.is_lead ? 18 : 14,
              color: Color.fromCssColorString(stateColor(layer.proposal_state)),
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
          }) as { point?: { outlineWidth?: number; pixelSize?: number } };
          entityMapRef.current.set(layer.object_id, entity);
        }

        const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
        handlerRef.current = handler;
        handler.setInputAction((movement: { position: any }) => {
          const picked = viewer.scene.pick(movement.position) as { id?: { id?: string } } | undefined;
          if (defined(picked) && picked?.id?.id) {
            onSelectObject(String(picked.id.id));
          }
        }, ScreenSpaceEventType.LEFT_CLICK);
        handler.setInputAction((movement: { endPosition: any }) => {
          const picked = viewer.scene.pick(movement.endPosition) as { id?: { id?: string } } | undefined;
          if (defined(picked) && picked?.id?.id) {
            setHoveredObjectId(String(picked.id.id));
            return;
          }
          setHoveredObjectId(null);
        }, ScreenSpaceEventType.MOUSE_MOVE);

        const preset = sceneConfig.cameraPresets?.overview ?? { heading: 20, pitch: -42, range: 2400 };
        viewer.camera.flyToBoundingSphere(new BoundingSphere(anchor, 1600), {
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
    };
  }, [layers, onSelectObject]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) {
      return;
    }

    entityMapRef.current.forEach((entity, objectId) => {
      const layer = layerById.get(objectId);
      const basePixelSize = layer?.proposal_state === "warning_generated" ? 20 : layer?.is_lead ? 18 : 14;
      const baseOutlineWidth =
        layer?.proposal_state === "approved" || layer?.proposal_state === "warning_generated" ? 4 : 3;
      if (!entity.point) {
        return;
      }
      entity.point.outlineWidth = objectId === selectedObjectId ? baseOutlineWidth + 1 : baseOutlineWidth;
      entity.point.pixelSize = objectId === selectedObjectId ? basePixelSize + 4 : basePixelSize;
    });

    const selectedEntity = selectedObjectId ? entityMapRef.current.get(selectedObjectId) : undefined;
    if (selectedEntity) {
      viewer.flyTo(selectedEntity, { duration: 0.9 });
    }
    viewer.scene.requestRender();
  }, [layerById, selectedObjectId]);

  if (isJsdomRuntime()) {
    return (
      <div className={styles.fallbackShell} aria-label="digital-twin-canvas">
        <div className={styles.fallbackGrid} />
        {layers.map((layer) => (
          <button
            key={layer.object_id}
            type="button"
            className={`${styles.fallbackNode} ${selectedObjectId === layer.object_id ? styles.fallbackNodeActive : ""}`}
            style={{
              left: `${18 + (layer.east_offset_m + 340) / 9}%`,
              top: `${22 + (260 - layer.north_offset_m) / 8}%`,
            }}
            onClick={() => onSelectObject(layer.object_id)}
          >
            {`${layer.name} / ${stateLabel(layer.proposal_state)}`}
          </button>
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
