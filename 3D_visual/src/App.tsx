import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Axis,
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorBlendMode,
  EllipsoidTerrainProvider,
  ImageryLayer,
  Ion,
  IonWorldImageryStyle,
  Model,
  Viewer,
  createWorldImageryAsync,
  createWorldTerrainAsync,
  sampleTerrainMostDetailed
} from "cesium";

import { LEGACY_STORAGE_KEYS, STORAGE_KEY } from "./sceneConfig";
import {
  buildCopyPayload,
  computeModelMatrix,
  flyToPresetWithFallback,
  loadSourceMetadata,
  normalizeSceneConfig,
  placementFromConfig,
  readSavedPlacement,
} from "./scenePlacement";
import type {
  CameraKey,
  DebugFlags,
  DiagnosticCullResult,
  DiagnosticMode,
  PlacementState,
  SceneConfig,
  SourceMetadata,
} from "./sceneTypes";

const UI = {
  loadingConfig: "正在读取场景配置...",
  loadingSourceMetadata: "正在分析 CityEngine GLB 的源坐标信息...",
  sourceMetadataReady: "源坐标信息已准备完成，开始初始化 Cesium 场景。",
  startingCesium: "正在启动 Cesium 场景...",
  samplingTerrain: "正在采样锚点位置的真实地形高程...",
  terrainReady: "地形高程采样完成。",
  loadingGlb: "正在加载 CityEngine 导出的 GLB 模型...",
  modelReady: "模型已完成加载，并按照真实中心完成归一和定位。",
  heroEyebrow: "CITYENGINE 三维模型定位",
  heroTitle: "GLB 地球查看器",
  heroBody:
    "这是一个面向 CityEngine 模型的 Cesium 校准查看器。它会先读取 GLB 根节点坐标，再按固定轴向映射到地球锚点，并用真实地形高程作为落地基准。",
  panelTitle: "场景校准",
  visibility: "显示控制",
  placement: "位置调整",
  camera: "相机视角",
  debug: "调试辅助",
  actions: "操作",
  showModel: "显示模型",
  showTerrain: "显示地形",
  showImagery: "显示影像",
  showAnchor: "显示锚点",
  hideBaseMap: "隐藏底图，只看模型",
  restoreBaseMap: "恢复底图",
  scale: "缩放",
  heading: "航向角",
  pitch: "俯仰角",
  roll: "翻滚角",
  offsetEast: "东向偏移",
  offsetNorth: "北向偏移",
  offsetUp: "高度偏移",
  overview: "全局视角",
  oblique: "斜视视角",
  close: "近景视角",
  locateModel: "定位模型",
  showBounds: "显示包围球",
  brightMaterial: "高亮材质",
  resetPlacement: "重置位置",
  copyCurrent: "复制当前配置",
  clearSaved: "清除缓存",
  modelPath: "模型路径：",
  anchorLabel: "模型锚点",
  tokenRequired: "缺少 Cesium ion token",
  tokenHelp:
    "请把你的 Cesium ion token 保存到 3D_visual/cesium_token.txt，然后重新启动开发服务器。",
  tokenMore:
    "当前查看器依赖 Cesium World Terrain 和影像底图，未提供 token 时会停在说明页，不会静默失败。",
  viewerError: "场景加载失败",
  loadingSceneConfig: "正在加载场景配置",
  loadingSceneConfigBody:
    "查看器正在读取 scene-config.json，并准备根节点坐标归一、地形采样和模型落地逻辑。",
  preparingScene: "正在准备三维场景",
  preparingSceneBody:
    "场景会先读取 GLB 根节点、采样锚点地形、构建归一矩阵，然后再加载 Cesium 模型。",
  copySuccess: "当前配置已复制到剪贴板。"
} as const;

export default function App() {
  const viewerHostRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const imageryLayerRef = useRef<ImageryLayer | null>(null);
  const terrainProviderRef = useRef<Awaited<ReturnType<typeof createWorldTerrainAsync>> | null>(
    null
  );
  const modelRef = useRef<Model | null>(null);
  const anchorEntityRef = useRef<{ show?: boolean } | null>(null);

  const [sceneConfig, setSceneConfig] = useState<SceneConfig | null>(null);
  const [sourceMetadata, setSourceMetadata] = useState<SourceMetadata | null>(null);
  const [defaultPlacement, setDefaultPlacement] = useState<PlacementState | null>(null);
  const [placement, setPlacement] = useState<PlacementState | null>(null);
  const [debugFlags, setDebugFlags] = useState<DebugFlags>({
    showAnchor: true,
    showBoundingSphere: false,
    brightMaterial: true
  });
  const [statusText, setStatusText] = useState<string>(UI.loadingConfig);
  const [terrainSampleHeight, setTerrainSampleHeight] = useState<number | null>(null);
  const [errorText, setErrorText] = useState("");
  const [copyFeedback, setCopyFeedback] = useState("");
  const [viewerReady, setViewerReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [showModel, setShowModel] = useState(true);
  const [showTerrain, setShowTerrain] = useState(true);
  const [showImagery, setShowImagery] = useState(true);
  const [modelOnlyMode, setModelOnlyMode] = useState(false);
  const [diagnosticMode, setDiagnosticMode] = useState<DiagnosticMode>("default");

  const hasToken = Boolean(__CESIUM_ION_TOKEN__);

  useEffect(() => {
    let active = true;

    async function loadConfig() {
      try {
        const response = await fetch("/scene-config.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`scene-config.json 返回 ${response.status}`);
        }
        const config = normalizeSceneConfig(
          (await response.json()) as Partial<SceneConfig>
        );
        if (!active) {
          return;
        }

        const defaults = placementFromConfig(config);
        setSceneConfig(config);
        setDefaultPlacement(defaults);
        setPlacement({ ...defaults, ...readSavedPlacement(defaults) });
        setDebugFlags(config.debug);
        setStatusText(UI.loadingSourceMetadata);
      } catch (error) {
        if (active) {
          setErrorText(
            error instanceof Error ? error.message : "场景配置读取失败。"
          );
        }
      }
    }

    void loadConfig();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!sceneConfig) {
      return;
    }

    const currentConfig = sceneConfig;
    let active = true;
    async function prepareMetadata() {
      try {
        const metadata = await loadSourceMetadata(currentConfig);
        if (!active) {
          return;
        }
        setSourceMetadata(metadata);
        setStatusText(UI.sourceMetadataReady);
      } catch (error) {
        if (active) {
          setErrorText(
            error instanceof Error ? error.message : "GLB 源坐标解析失败。"
          );
        }
      }
    }

    void prepareMetadata();
    return () => {
      active = false;
    };
  }, [sceneConfig]);

  useEffect(() => {
    if (placement) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(placement));
    }
  }, [placement]);

  useEffect(() => {
    if (!sceneConfig || !hasToken || !viewerHostRef.current || viewerRef.current) {
      return;
    }

    const currentConfig = sceneConfig;
    let cancelled = false;
    async function initViewer() {
      try {
        setStatusText(UI.startingCesium);
        Ion.defaultAccessToken = __CESIUM_ION_TOKEN__;
        const terrainProvider = await createWorldTerrainAsync();
        const imageryProvider = await createWorldImageryAsync({
          style: IonWorldImageryStyle.AERIAL
        });
        if (cancelled || !viewerHostRef.current) {
          return;
        }

        const viewer = new Viewer(viewerHostRef.current, {
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
          selectionIndicator: false
        });

        viewer.scene.globe.depthTestAgainstTerrain = false;
        viewer.scene.fog.enabled = true;
        if (viewer.scene.skyAtmosphere) {
          viewer.scene.skyAtmosphere.show = true;
        }

        imageryLayerRef.current = viewer.imageryLayers.addImageryProvider(imageryProvider);
        terrainProviderRef.current = terrainProvider;
        viewerRef.current = viewer;
        anchorEntityRef.current = viewer.entities.add({
          position: Cartesian3.fromDegrees(
            currentConfig.anchorLon,
            currentConfig.anchorLat,
            currentConfig.anchorHeight
          ),
          point: {
            pixelSize: 12,
            color: Color.fromCssColorString("#66e5ff"),
            outlineColor: Color.WHITE,
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY
          },
          label: {
            text: UI.anchorLabel,
            font: "14px sans-serif",
            fillColor: Color.WHITE,
            showBackground: true,
            backgroundColor: Color.fromCssColorString("#0a1426").withAlpha(0.84),
            pixelOffset: new Cartesian2(0, -28)
          },
          show: debugFlags.showAnchor
        });

        setViewerReady(true);

        let resolvedHeight = currentConfig.anchorHeight;
        if (currentConfig.terrainAnchorMode === "sample-most-detailed") {
          setStatusText(UI.samplingTerrain);
          try {
            const sampled = await sampleTerrainMostDetailed(terrainProvider, [
              Cartographic.fromDegrees(currentConfig.anchorLon, currentConfig.anchorLat)
            ]);
            if (Number.isFinite(sampled[0]?.height)) {
              resolvedHeight = sampled[0].height as number;
              setStatusText(UI.terrainReady);
            }
          } catch {
            resolvedHeight = currentConfig.anchorHeight;
          }
        }

        if (!cancelled) {
          setTerrainSampleHeight(resolvedHeight);
        }
      } catch (error) {
        setErrorText(
          error instanceof Error ? error.message : "Cesium 初始化失败。"
        );
      }
    }

    void initViewer();
    return () => {
      cancelled = true;
      modelRef.current = null;
      imageryLayerRef.current = null;
      terrainProviderRef.current = null;
      anchorEntityRef.current = null;
      setViewerReady(false);
      setModelReady(false);
      if (viewerRef.current) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, [debugFlags.showAnchor, hasToken, sceneConfig]);

  useEffect(() => {
    if (
      !viewerReady ||
      !viewerRef.current ||
      !sceneConfig ||
      !sourceMetadata ||
      !placement ||
      terrainSampleHeight === null ||
      modelRef.current
    ) {
      return;
    }

    const currentConfig = sceneConfig;
    const currentPlacement = placement;
    const currentSourceMetadata = sourceMetadata;
    const currentTerrainHeight = terrainSampleHeight;
    let cancelled = false;
    async function loadModel() {
      try {
        setStatusText(UI.loadingGlb);
        const model = await Model.fromGltfAsync({
          url: currentConfig.modelUrl,
          modelMatrix: computeModelMatrix(
            currentConfig,
            currentPlacement,
            currentSourceMetadata,
            currentTerrainHeight
          ),
          show: showModel,
          minimumPixelSize: 0,
          cull: true,
          backFaceCulling: true,
          upAxis: Axis.Z,
          forwardAxis: Axis.X
        });

        if (cancelled || !viewerRef.current) {
          return;
        }

        viewerRef.current.scene.primitives.add(model);
        modelRef.current = model;
        model.readyEvent.addEventListener(() => {
          if (
            cancelled ||
            !viewerRef.current ||
            currentTerrainHeight === null
          ) {
            return;
          }
          setModelReady(true);
          setStatusText(UI.modelReady);
          flyToPresetWithFallback(
            viewerRef.current,
            currentConfig,
            currentTerrainHeight,
            currentConfig.cameraPresets.oblique,
            currentPlacement,
            currentSourceMetadata
          );
        });
      } catch (error) {
        setErrorText(
          error instanceof Error ? error.message : "GLB 模型加载失败。"
        );
      }
    }

    void loadModel();
    return () => {
      cancelled = true;
    };
  }, [
    placement,
    sceneConfig,
    showModel,
    sourceMetadata,
    terrainSampleHeight,
    viewerReady
  ]);

  useEffect(() => {
    if (!sceneConfig || !placement || !sourceMetadata || terrainSampleHeight === null) {
      return;
    }

    if (modelRef.current) {
      modelRef.current.modelMatrix = computeModelMatrix(
        sceneConfig,
        placement,
        sourceMetadata,
        terrainSampleHeight
      );
    }
  }, [placement, sceneConfig, sourceMetadata, terrainSampleHeight]);

  useEffect(() => {
    const model = modelRef.current;
    if (!model) {
      return;
    }

    model.show = showModel;
    model.debugShowBoundingVolume = debugFlags.showBoundingSphere;
    model.color = debugFlags.brightMaterial
      ? Color.fromCssColorString("#7cf5ff").withAlpha(0.96)
      : Color.WHITE;
    model.colorBlendMode = debugFlags.brightMaterial
      ? ColorBlendMode.REPLACE
      : ColorBlendMode.HIGHLIGHT;
    model.colorBlendAmount = debugFlags.brightMaterial ? 1 : 0;
    model.silhouetteColor = Color.BLACK;
    model.silhouetteSize = debugFlags.brightMaterial ? 3.2 : 0;
  }, [debugFlags, modelReady, showModel]);

  useEffect(() => {
    const anchorEntity = anchorEntityRef.current;
    if (anchorEntity) {
      anchorEntity.show = debugFlags.showAnchor;
    }
  }, [debugFlags.showAnchor]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) {
      return;
    }

    viewer.scene.globe.show = !modelOnlyMode;
    viewer.scene.fog.enabled = !modelOnlyMode;
    if (viewer.scene.skyAtmosphere) {
      viewer.scene.skyAtmosphere.show = !modelOnlyMode;
    }

    viewer.terrainProvider = showTerrain
      ? terrainProviderRef.current ?? new EllipsoidTerrainProvider()
      : new EllipsoidTerrainProvider();

    if (imageryLayerRef.current) {
      imageryLayerRef.current.show = showImagery && !modelOnlyMode;
    }
  }, [modelOnlyMode, showImagery, showTerrain]);

  const updatePlacement =
    (key: keyof PlacementState) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const nextValue = Number(event.target.value);
      if (!Number.isNaN(nextValue)) {
        setPlacement((current) =>
          current ? { ...current, [key]: nextValue } : current
        );
      }
    };

  const toggleDebug =
    (key: keyof DebugFlags) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      setDebugFlags((current) => ({
        ...current,
        [key]: event.target.checked
      }));
    };

  async function copyCurrentPlacement() {
    if (!sceneConfig || !placement) {
      return;
    }
    await navigator.clipboard.writeText(
      JSON.stringify(buildCopyPayload(sceneConfig, placement), null, 2)
    );
    setCopyFeedback(UI.copySuccess);
    window.setTimeout(() => setCopyFeedback(""), 1800);
  }

  function resetPlacement() {
    if (defaultPlacement) {
      setPlacement({ ...defaultPlacement });
    }
  }

  function clearSavedPlacement() {
    localStorage.removeItem(STORAGE_KEY);
    for (const key of LEGACY_STORAGE_KEYS) {
      localStorage.removeItem(key);
    }
    resetPlacement();
  }

  function flyToKey(key: CameraKey) {
    if (
      !viewerRef.current ||
      !sceneConfig ||
      !placement ||
      !sourceMetadata ||
      terrainSampleHeight === null
    ) {
      return;
    }

    flyToPresetWithFallback(
      viewerRef.current,
      sceneConfig,
      terrainSampleHeight,
      sceneConfig.cameraPresets[key],
      placement,
      sourceMetadata
    );
  }

  const currentCopyPreview =
    sceneConfig && placement
      ? JSON.stringify(buildCopyPayload(sceneConfig, placement), null, 2)
      : "{}";

  useEffect(() => {
    if (!viewerRef.current || !sceneConfig) {
      return;
    }

    const diagnosticCullResult: DiagnosticCullResult = {
      activeMode: diagnosticMode,
      defaultCull: true,
      defaultBackFaceCulling: true,
      runtimeMutable: false
    };

    window.__GLB_DEBUG__ = {
      viewer: viewerRef.current,
      model: modelRef.current,
      sceneConfig,
      placement,
      modelReady,
      sourceRootCenter: sourceMetadata
        ? {
            x: sourceMetadata.sourceRootCenter.x,
            y: sourceMetadata.sourceRootCenter.y,
            z: sourceMetadata.sourceRootCenter.z
          }
        : null,
      terrainSampleHeight,
      computedBaseHeight: sourceMetadata?.computedBaseHeight ?? null,
      finalAnchor: {
        lon: sceneConfig.anchorLon,
        lat: sceneConfig.anchorLat,
        height: terrainSampleHeight ?? sceneConfig.anchorHeight
      },
      diagnosticCullResult,
      setDiagnosticMode
    };
  }, [diagnosticMode, modelReady, placement, sceneConfig, sourceMetadata, terrainSampleHeight]);

  return (
    <div className="app-shell">
      <div className="viewer-root">
        <div ref={viewerHostRef} className="viewer-surface" />
      </div>

      <div className="hud">
        <div className="hero">
          <p className="eyebrow">{UI.heroEyebrow}</p>
          <h1>{UI.heroTitle}</h1>
          <p>{UI.heroBody}</p>
        </div>

        <aside className="panel">
          <div>
            <h2>{UI.panelTitle}</h2>
            <p className="panel-copy">
              {UI.modelPath}
              <code>{sceneConfig?.modelUrl ?? "/models/cityengine_scene.glb"}</code>
            </p>
            <p className="status-text">{statusText}</p>
            {copyFeedback ? <p className="status-text">{copyFeedback}</p> : null}
          </div>

          <section className="section">
            <h3>{UI.visibility}</h3>
            <div className="toggle-grid">
              <label className="toggle">
                <span>{UI.showModel}</span>
                <input
                  type="checkbox"
                  checked={showModel}
                  onChange={(event) => setShowModel(event.target.checked)}
                />
              </label>
              <label className="toggle">
                <span>{UI.showTerrain}</span>
                <input
                  type="checkbox"
                  checked={showTerrain}
                  onChange={(event) => setShowTerrain(event.target.checked)}
                />
              </label>
              <label className="toggle">
                <span>{UI.showImagery}</span>
                <input
                  type="checkbox"
                  checked={showImagery}
                  onChange={(event) => setShowImagery(event.target.checked)}
                />
              </label>
              <label className="toggle">
                <span>{UI.showAnchor}</span>
                <input
                  type="checkbox"
                  checked={debugFlags.showAnchor}
                  onChange={toggleDebug("showAnchor")}
                />
              </label>
            </div>
            <div className="button-grid">
              <button
                className="action-btn"
                onClick={() => setModelOnlyMode((current) => !current)}
              >
                {modelOnlyMode ? UI.restoreBaseMap : UI.hideBaseMap}
              </button>
            </div>
          </section>

          <section className="section">
            <h3>{UI.placement}</h3>
            <div className="metric-grid">
              <label className="control-card">
                <span className="field-label">{UI.scale}</span>
                <input
                  className="field-input"
                  type="number"
                  step="0.01"
                  value={placement?.scale ?? 0}
                  onChange={updatePlacement("scale")}
                />
              </label>
              <label className="control-card">
                <span className="field-label">{UI.heading}</span>
                <input
                  className="field-input"
                  type="number"
                  step="0.1"
                  value={placement?.heading ?? 0}
                  onChange={updatePlacement("heading")}
                />
              </label>
              <label className="control-card">
                <span className="field-label">{UI.pitch}</span>
                <input
                  className="field-input"
                  type="number"
                  step="0.1"
                  value={placement?.pitch ?? 0}
                  onChange={updatePlacement("pitch")}
                />
              </label>
              <label className="control-card">
                <span className="field-label">{UI.roll}</span>
                <input
                  className="field-input"
                  type="number"
                  step="0.1"
                  value={placement?.roll ?? 0}
                  onChange={updatePlacement("roll")}
                />
              </label>
              <label className="control-card">
                <span className="field-label">{UI.offsetEast}</span>
                <input
                  className="field-input"
                  type="number"
                  step="1"
                  value={placement?.offsetEast ?? 0}
                  onChange={updatePlacement("offsetEast")}
                />
              </label>
              <label className="control-card">
                <span className="field-label">{UI.offsetNorth}</span>
                <input
                  className="field-input"
                  type="number"
                  step="1"
                  value={placement?.offsetNorth ?? 0}
                  onChange={updatePlacement("offsetNorth")}
                />
              </label>
              <label className="control-card">
                <span className="field-label">{UI.offsetUp}</span>
                <input
                  className="field-input"
                  type="number"
                  step="0.5"
                  value={placement?.offsetUp ?? 0}
                  onChange={updatePlacement("offsetUp")}
                />
              </label>
            </div>
            <p className="panel-note">
              当前查看器采用真实比例优先，场景垂直缩放为 {sceneConfig?.verticalScale ?? 1}
              x。只有用户主动调整时，模型才会离开默认地形落位结果。
            </p>
          </section>

          <section className="section">
            <h3>{UI.camera}</h3>
            <div className="button-grid">
              <button className="action-btn" onClick={() => flyToKey("overview")}>
                {UI.overview}
              </button>
              <button className="action-btn" onClick={() => flyToKey("oblique")}>
                {UI.oblique}
              </button>
              <button className="action-btn" onClick={() => flyToKey("close")}>
                {UI.close}
              </button>
              <button className="action-btn" onClick={() => flyToKey("oblique")}>
                {UI.locateModel}
              </button>
            </div>
          </section>

          <section className="section">
            <h3>{UI.debug}</h3>
            <div className="toggle-grid">
              <label className="toggle">
                <span>{UI.showBounds}</span>
                <input
                  type="checkbox"
                  checked={debugFlags.showBoundingSphere}
                  onChange={toggleDebug("showBoundingSphere")}
                />
              </label>
              <label className="toggle">
                <span>{UI.brightMaterial}</span>
                <input
                  type="checkbox"
                  checked={debugFlags.brightMaterial}
                  onChange={toggleDebug("brightMaterial")}
                />
              </label>
            </div>
          </section>

          <section className="section">
            <h3>{UI.actions}</h3>
            <div className="button-grid">
              <button className="action-btn" onClick={resetPlacement}>
                {UI.resetPlacement}
              </button>
              <button className="action-btn" onClick={() => void copyCurrentPlacement()}>
                {UI.copyCurrent}
              </button>
              <button className="action-btn secondary" onClick={clearSavedPlacement}>
                {UI.clearSaved}
              </button>
            </div>
            <pre className="code-block">{currentCopyPreview}</pre>
          </section>
        </aside>
      </div>

      {!hasToken ? (
        <div className="overlay">
          <div className="overlay-card">
            <h2>{UI.tokenRequired}</h2>
            <p>{UI.tokenHelp}</p>
            <div className="token-path">3D_visual/cesium_token.txt</div>
            <p>{UI.tokenMore}</p>
          </div>
        </div>
      ) : null}

      {errorText ? (
        <div className="overlay">
          <div className="overlay-card">
            <h2>{UI.viewerError}</h2>
            <p>{errorText}</p>
          </div>
        </div>
      ) : null}

      {!sceneConfig && !errorText ? (
        <div className="overlay">
          <div className="overlay-card">
            <h2>{UI.loadingSceneConfig}</h2>
            <p>{UI.loadingSceneConfigBody}</p>
          </div>
        </div>
      ) : null}

      {hasToken &&
      sceneConfig &&
      sourceMetadata &&
      terrainSampleHeight !== null &&
      !modelReady &&
      !errorText ? (
        <div className="overlay">
          <div className="overlay-card">
            <h2>{UI.preparingScene}</h2>
            <p>{UI.preparingSceneBody}</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
