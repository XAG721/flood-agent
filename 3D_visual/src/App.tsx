import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Axis,
  BoundingSphere,
  Cartesian2,
  Cartesian3,
  Cartographic,
  Color,
  ColorBlendMode,
  EllipsoidTerrainProvider,
  HeadingPitchRange,
  HeadingPitchRoll,
  ImageryLayer,
  Ion,
  IonWorldImageryStyle,
  Math as CesiumMath,
  Matrix4,
  Model,
  Quaternion,
  Transforms,
  Viewer,
  createWorldImageryAsync,
  createWorldTerrainAsync,
  sampleTerrainMostDetailed
} from "cesium";

type CameraKey = "overview" | "oblique" | "close";
type SourceAxisToken = "x" | "y" | "z" | "-x" | "-y" | "-z";
type SourceCenterMode = "root-node-centroid" | "bounds-center";
type TerrainAnchorMode = "sample-most-detailed";
type DiagnosticMode = "default" | "no-cull" | "no-backface-cull";

type CameraPreset = {
  heading: number;
  pitch: number;
  range: number;
  fitWholeModel?: boolean;
  focusRadius?: number;
};
type DebugFlags = {
  showAnchor: boolean;
  showBoundingSphere: boolean;
  brightMaterial: boolean;
};
type SourceFrame = {
  east: SourceAxisToken;
  north: SourceAxisToken;
  up: SourceAxisToken;
};
type SceneConfig = {
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
  verticalScale: number;
  sourceFrame: SourceFrame;
  sourceCenterMode: SourceCenterMode;
  terrainAnchorMode: TerrainAnchorMode;
  cameraPresets: Record<CameraKey, CameraPreset>;
  debug: DebugFlags;
};
type PlacementState = {
  scale: number;
  heading: number;
  pitch: number;
  roll: number;
  offsetEast: number;
  offsetNorth: number;
  offsetUp: number;
};
type SourceBounds = { min: Cartesian3; max: Cartesian3 };
type SourceMetadata = {
  sourceRootCenter: Cartesian3;
  sourceBounds: SourceBounds;
  computedBaseHeight: number;
  sourceNormalizationMatrix: Matrix4;
  localFocusCenter: Cartesian3;
  normalizedSizes: Cartesian3;
};
type DiagnosticCullResult = {
  activeMode: DiagnosticMode;
  defaultCull: boolean;
  defaultBackFaceCulling: boolean;
  runtimeMutable: boolean;
};
type GlTfNode = {
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
};
type GlTfPrimitive = { attributes?: { POSITION?: number } };
type GlTfMesh = { primitives?: GlTfPrimitive[] };
type GlTfAccessor = { min?: number[]; max?: number[] };
type GlTfScene = { nodes?: number[] };
type GlTfDocument = {
  scene?: number;
  scenes?: GlTfScene[];
  nodes?: GlTfNode[];
  meshes?: GlTfMesh[];
  accessors?: GlTfAccessor[];
};
const STORAGE_KEY = "cityengine-glb-viewer-placement-v8";
const LEGACY_STORAGE_KEYS = [
  "cityengine-glb-viewer-placement-v1",
  "cityengine-glb-viewer-placement-v2",
  "cityengine-glb-viewer-placement-v3",
  "cityengine-glb-viewer-placement-v4",
  "cityengine-glb-viewer-placement-v5",
  "cityengine-glb-viewer-placement-v6",
  "cityengine-glb-viewer-placement-v7"
] as const;
const DEFAULT_SOURCE_FRAME: SourceFrame = { east: "x", north: "-z", up: "y" };

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

function placementFromConfig(config: SceneConfig): PlacementState {
  return {
    scale: config.scale,
    heading: config.heading,
    pitch: config.pitch,
    roll: config.roll,
    offsetEast: config.offsetEast,
    offsetNorth: config.offsetNorth,
    offsetUp: config.offsetUp
  };
}

function sanitizeSavedPlacement(
  saved: Partial<PlacementState>,
  defaults: PlacementState
): Partial<PlacementState> {
  const readNumber = (
    key: keyof PlacementState,
    fallback: number,
    options?: { min?: number; max?: number }
  ) => {
    const value = saved[key];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return fallback;
    }
    if (options?.min !== undefined && value < options.min) {
      return fallback;
    }
    if (options?.max !== undefined && value > options.max) {
      return fallback;
    }
    return value;
  };

  return {
    scale: readNumber("scale", defaults.scale, { min: 0.01, max: 100 }),
    heading: readNumber("heading", defaults.heading, { min: -360, max: 360 }),
    pitch: readNumber("pitch", defaults.pitch, { min: -180, max: 180 }),
    roll: readNumber("roll", defaults.roll, { min: -180, max: 180 }),
    offsetEast: readNumber("offsetEast", defaults.offsetEast, { min: -5000, max: 5000 }),
    offsetNorth: readNumber("offsetNorth", defaults.offsetNorth, { min: -5000, max: 5000 }),
    offsetUp: readNumber("offsetUp", defaults.offsetUp, { min: -80, max: 80 })
  };
}

function readSavedPlacement(defaults: PlacementState): Partial<PlacementState> {
  try {
    for (const key of LEGACY_STORAGE_KEYS) {
      localStorage.removeItem(key);
    }
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    return sanitizeSavedPlacement(
      JSON.parse(raw) as Partial<PlacementState>,
      defaults
    );
  } catch {
    return {};
  }
}

function normalizeSceneConfig(raw: Partial<SceneConfig>): SceneConfig {
  return {
    modelUrl: raw.modelUrl ?? "/models/cityengine_scene.glb",
    anchorLon: raw.anchorLon ?? 108.94921153512861,
    anchorLat: raw.anchorLat ?? 34.24624474240188,
    anchorHeight: raw.anchorHeight ?? 404.3599853515625,
    scale: raw.scale ?? 1,
    heading: raw.heading ?? 0,
    pitch: raw.pitch ?? 0,
    roll: raw.roll ?? 0,
    offsetEast: raw.offsetEast ?? 0,
    offsetNorth: raw.offsetNorth ?? 0,
    offsetUp: raw.offsetUp ?? 0,
    verticalScale: raw.verticalScale ?? 1,
    sourceFrame: raw.sourceFrame ?? DEFAULT_SOURCE_FRAME,
    sourceCenterMode: raw.sourceCenterMode ?? "bounds-center",
    terrainAnchorMode: raw.terrainAnchorMode ?? "sample-most-detailed",
    cameraPresets:
      raw.cameraPresets ?? {
        overview: { heading: 20, pitch: -42, range: 2400, fitWholeModel: true },
        oblique: { heading: 38, pitch: -24, range: 950, focusRadius: 180 },
        close: { heading: 12, pitch: -14, range: 320, focusRadius: 80 }
      },
    debug:
      raw.debug ?? {
        showAnchor: true,
        showBoundingSphere: false,
        brightMaterial: true
      }
  };
}

function parseAxisToken(token: SourceAxisToken) {
  return token.startsWith("-")
    ? { axis: token.slice(1) as "x" | "y" | "z", sign: -1 as const }
    : { axis: token as "x" | "y" | "z", sign: 1 as const };
}

function getAxisValue(vector: Cartesian3, axis: "x" | "y" | "z") {
  return axis === "x" ? vector.x : axis === "y" ? vector.y : vector.z;
}

function getAxisSpan(bounds: SourceBounds, token: SourceAxisToken) {
  const parsed = parseAxisToken(token);
  return Math.abs(
    getAxisValue(bounds.max, parsed.axis) - getAxisValue(bounds.min, parsed.axis)
  );
}

function getAxisBaseValue(bounds: SourceBounds, token: SourceAxisToken) {
  const parsed = parseAxisToken(token);
  const min = getAxisValue(bounds.min, parsed.axis);
  const max = getAxisValue(bounds.max, parsed.axis);
  return parsed.sign > 0 ? min : max;
}

function getNodeMatrix(node: GlTfNode) {
  if (Array.isArray(node.matrix) && node.matrix.length === 16) {
    return Matrix4.fromColumnMajorArray(node.matrix, new Matrix4());
  }

  const translation = node.translation
    ? Cartesian3.fromArray(node.translation)
    : new Cartesian3(0, 0, 0);
  const rotation = node.rotation
    ? new Quaternion(
        node.rotation[0],
        node.rotation[1],
        node.rotation[2],
        node.rotation[3]
      )
    : new Quaternion(0, 0, 0, 1);
  const scale = node.scale
    ? Cartesian3.fromArray(node.scale)
    : new Cartesian3(1, 1, 1);

  return Matrix4.fromTranslationQuaternionRotationScale(
    translation,
    rotation,
    scale
  );
}

function getBoundingCorners(min: number[], max: number[]) {
  return [
    new Cartesian3(min[0], min[1], min[2]),
    new Cartesian3(min[0], min[1], max[2]),
    new Cartesian3(min[0], max[1], min[2]),
    new Cartesian3(min[0], max[1], max[2]),
    new Cartesian3(max[0], min[1], min[2]),
    new Cartesian3(max[0], min[1], max[2]),
    new Cartesian3(max[0], max[1], min[2]),
    new Cartesian3(max[0], max[1], max[2])
  ];
}

function parseGlbJson(buffer: ArrayBuffer): GlTfDocument {
  const header = new DataView(buffer, 0, 12);
  if (header.getUint32(0, true) !== 0x46546c67) {
    throw new Error("GLB 文件头无效，无法读取 JSON 描述。");
  }

  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const chunkHeader = new DataView(buffer, offset, 8);
    const chunkLength = chunkHeader.getUint32(0, true);
    const chunkType = chunkHeader.getUint32(4, true);
    offset += 8;
    if (chunkType === 0x4e4f534a) {
      const jsonText = new TextDecoder("utf-8").decode(
        new Uint8Array(buffer, offset, chunkLength)
      );
      return JSON.parse(jsonText) as GlTfDocument;
    }
    offset += chunkLength;
  }

  throw new Error("GLB 文件中没有 JSON chunk。");
}

function computeSphereRadius(size: Cartesian3) {
  return Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z) / 2;
}

function collectSourceMetadata(
  gltf: GlTfDocument,
  config: SceneConfig
): SourceMetadata {
  const nodes = gltf.nodes ?? [];
  const meshes = gltf.meshes ?? [];
  const accessors = gltf.accessors ?? [];
  const scene = (gltf.scenes ?? [])[gltf.scene ?? 0];
  const roots = scene?.nodes ?? [];

  if (roots.length === 0) {
    throw new Error("当前 GLB 场景没有可用的根节点。");
  }

  const bounds: SourceBounds = {
    min: new Cartesian3(Infinity, Infinity, Infinity),
    max: new Cartesian3(-Infinity, -Infinity, -Infinity)
  };
  const rootTranslations: Cartesian3[] = [];

  const updateBounds = (point: Cartesian3) => {
    bounds.min.x = Math.min(bounds.min.x, point.x);
    bounds.min.y = Math.min(bounds.min.y, point.y);
    bounds.min.z = Math.min(bounds.min.z, point.z);
    bounds.max.x = Math.max(bounds.max.x, point.x);
    bounds.max.y = Math.max(bounds.max.y, point.y);
    bounds.max.z = Math.max(bounds.max.z, point.z);
  };

  const walkNode = (nodeIndex: number, parentMatrix: Matrix4) => {
    const node = nodes[nodeIndex];
    if (!node) {
      return;
    }

    const localMatrix = getNodeMatrix(node);
    const worldMatrix = Matrix4.multiply(parentMatrix, localMatrix, new Matrix4());

    if (node.mesh !== undefined) {
      const mesh = meshes[node.mesh];
      for (const primitive of mesh?.primitives ?? []) {
        const accessorIndex = primitive.attributes?.POSITION;
        const accessor = accessorIndex !== undefined ? accessors[accessorIndex] : undefined;
        if (!accessor?.min || !accessor?.max) {
          continue;
        }
        for (const corner of getBoundingCorners(accessor.min, accessor.max)) {
          updateBounds(
            Matrix4.multiplyByPoint(worldMatrix, corner, new Cartesian3())
          );
        }
      }
    }

    for (const childIndex of node.children ?? []) {
      walkNode(childIndex, worldMatrix);
    }
  };

  for (const rootIndex of roots) {
    const rootNode = nodes[rootIndex];
    if (!rootNode) {
      continue;
    }
    const rootMatrix = getNodeMatrix(rootNode);
    rootTranslations.push(Matrix4.getTranslation(rootMatrix, new Cartesian3()));
    walkNode(rootIndex, Matrix4.IDENTITY);
  }

  if (!Number.isFinite(bounds.min.x) || rootTranslations.length === 0) {
    throw new Error("无法从 GLB 根节点和 POSITION accessor 推导有效范围。");
  }

  const summedRootCenter = rootTranslations.reduce(
    (sum, current) =>
      new Cartesian3(sum.x + current.x, sum.y + current.y, sum.z + current.z),
    new Cartesian3(0, 0, 0)
  );
  const sourceRootCenter = new Cartesian3(
    summedRootCenter.x / rootTranslations.length,
    summedRootCenter.y / rootTranslations.length,
    summedRootCenter.z / rootTranslations.length
  );
  const worldBoundsCenter = new Cartesian3(
    (bounds.min.x + bounds.max.x) / 2,
    (bounds.min.y + bounds.max.y) / 2,
    (bounds.min.z + bounds.max.z) / 2
  );
  const horizontalCenter =
    config.sourceCenterMode === "root-node-centroid"
      ? sourceRootCenter
      : worldBoundsCenter;

  const eastAxis = parseAxisToken(config.sourceFrame.east);
  const northAxis = parseAxisToken(config.sourceFrame.north);
  const upAxis = parseAxisToken(config.sourceFrame.up);
  const computedBaseHeight = getAxisBaseValue(bounds, config.sourceFrame.up);

  const sourceNormalizationMatrix = Matrix4.fromRowMajorArray([
    eastAxis.axis === "x" ? eastAxis.sign : 0,
    eastAxis.axis === "y" ? eastAxis.sign : 0,
    eastAxis.axis === "z" ? eastAxis.sign : 0,
    -eastAxis.sign * getAxisValue(horizontalCenter, eastAxis.axis),
    northAxis.axis === "x" ? northAxis.sign : 0,
    northAxis.axis === "y" ? northAxis.sign : 0,
    northAxis.axis === "z" ? northAxis.sign : 0,
    -northAxis.sign * getAxisValue(horizontalCenter, northAxis.axis),
    upAxis.axis === "x" ? upAxis.sign : 0,
    upAxis.axis === "y" ? upAxis.sign : 0,
    upAxis.axis === "z" ? upAxis.sign : 0,
    -upAxis.sign * computedBaseHeight,
    0,
    0,
    0,
    1
  ]);

  const localFocusCenter = Matrix4.multiplyByPoint(
    sourceNormalizationMatrix,
    worldBoundsCenter,
    new Cartesian3()
  );

  return {
    sourceRootCenter,
    sourceBounds: bounds,
    computedBaseHeight,
    sourceNormalizationMatrix,
    localFocusCenter,
    normalizedSizes: new Cartesian3(
      getAxisSpan(bounds, config.sourceFrame.east),
      getAxisSpan(bounds, config.sourceFrame.north),
      getAxisSpan(bounds, config.sourceFrame.up)
    )
  };
}

async function loadSourceMetadata(config: SceneConfig) {
  const response = await fetch(config.modelUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`GLB 读取失败：${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  return collectSourceMetadata(parseGlbJson(buffer), config);
}

function computeModelMatrix(
  config: SceneConfig,
  placement: PlacementState,
  sourceMetadata: SourceMetadata,
  anchorHeight: number
) {
  const anchor = Cartesian3.fromDegrees(
    config.anchorLon,
    config.anchorLat,
    anchorHeight
  );
  const headingPitchRoll = new HeadingPitchRoll(
    CesiumMath.toRadians(placement.heading),
    CesiumMath.toRadians(placement.pitch),
    CesiumMath.toRadians(placement.roll)
  );
  const calibration = Matrix4.fromTranslationQuaternionRotationScale(
    new Cartesian3(
      placement.offsetEast,
      placement.offsetNorth,
      placement.offsetUp
    ),
    Quaternion.fromHeadingPitchRoll(headingPitchRoll),
    new Cartesian3(
      placement.scale,
      placement.scale,
      placement.scale * config.verticalScale
    )
  );

  return Matrix4.multiply(
    Transforms.eastNorthUpToFixedFrame(anchor),
    Matrix4.multiply(
      calibration,
      sourceMetadata.sourceNormalizationMatrix,
      new Matrix4()
    ),
    new Matrix4()
  );
}

function computePlacementFrame(
  config: SceneConfig,
  placement: PlacementState,
  anchorHeight: number
) {
  const anchor = Cartesian3.fromDegrees(
    config.anchorLon,
    config.anchorLat,
    anchorHeight
  );
  const headingPitchRoll = new HeadingPitchRoll(
    CesiumMath.toRadians(placement.heading),
    CesiumMath.toRadians(placement.pitch),
    CesiumMath.toRadians(placement.roll)
  );
  const calibration = Matrix4.fromTranslationQuaternionRotationScale(
    new Cartesian3(
      placement.offsetEast,
      placement.offsetNorth,
      placement.offsetUp
    ),
    Quaternion.fromHeadingPitchRoll(headingPitchRoll),
    new Cartesian3(
      placement.scale,
      placement.scale,
      placement.scale * config.verticalScale
    )
  );

  return Matrix4.multiply(
    Transforms.eastNorthUpToFixedFrame(anchor),
    calibration,
    new Matrix4()
  );
}

function computeModelFocusSphere(
  config: SceneConfig,
  placement: PlacementState,
  sourceMetadata: SourceMetadata,
  anchorHeight: number
) {
  const center = computeTargetModelCenter(
    config,
    placement,
    sourceMetadata,
    anchorHeight
  );
  const scaledSize = new Cartesian3(
    sourceMetadata.normalizedSizes.x * placement.scale,
    sourceMetadata.normalizedSizes.y * placement.scale,
    sourceMetadata.normalizedSizes.z * placement.scale * config.verticalScale
  );
  return new BoundingSphere(center, Math.max(computeSphereRadius(scaledSize), 1));
}

function computeTargetModelCenter(
  config: SceneConfig,
  placement: PlacementState,
  sourceMetadata: SourceMetadata,
  anchorHeight: number
) {
  const placementFrame = computePlacementFrame(
    config,
    placement,
    anchorHeight
  );
  return Matrix4.multiplyByPoint(
    placementFrame,
    sourceMetadata.localFocusCenter,
    new Cartesian3()
  );
}

function buildCopyPayload(config: SceneConfig, placement: PlacementState) {
  return {
    modelUrl: config.modelUrl,
    anchorLon: config.anchorLon,
    anchorLat: config.anchorLat,
    anchorHeight: config.anchorHeight,
    scale: placement.scale,
    heading: placement.heading,
    pitch: placement.pitch,
    roll: placement.roll,
    offsetEast: placement.offsetEast,
    offsetNorth: placement.offsetNorth,
    offsetUp: placement.offsetUp,
    verticalScale: config.verticalScale,
    sourceFrame: config.sourceFrame,
    sourceCenterMode: config.sourceCenterMode,
    terrainAnchorMode: config.terrainAnchorMode
  };
}

function isFiniteCartesian3(point: Cartesian3) {
  return (
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    Number.isFinite(point.z)
  );
}

function computeAnchorSphere(
  config: SceneConfig,
  anchorHeight: number,
  placement?: PlacementState
) {
  const center = Cartesian3.fromDegrees(
    config.anchorLon,
    config.anchorLat,
    anchorHeight + (placement?.offsetUp ?? 0)
  );
  return new BoundingSphere(center, 1);
}

function flyToPreset(viewer: Viewer, sphere: BoundingSphere, preset: CameraPreset) {
  viewer.camera.flyToBoundingSphere(sphere, {
    duration: 1.3,
    offset: new HeadingPitchRange(
      CesiumMath.toRadians(preset.heading),
      CesiumMath.toRadians(preset.pitch),
      preset.range
    )
  });
}

function buildCameraTargetSphere(targetSphere: BoundingSphere, preset: CameraPreset) {
  if (preset.fitWholeModel) {
    return targetSphere;
  }

  const radius = Math.max(
    1,
    Math.min(targetSphere.radius, preset.focusRadius ?? Math.max(20, preset.range * 0.18))
  );
  return new BoundingSphere(targetSphere.center, radius);
}

function flyToPresetWithFallback(
  viewer: Viewer,
  config: SceneConfig,
  anchorHeight: number,
  preset: CameraPreset,
  placement: PlacementState,
  sourceMetadata?: SourceMetadata | null
) {
  const targetSphere =
    sourceMetadata &&
    isFiniteCartesian3(sourceMetadata.localFocusCenter) &&
    isFiniteCartesian3(sourceMetadata.normalizedSizes)
      ? computeModelFocusSphere(config, placement, sourceMetadata, anchorHeight)
      : computeAnchorSphere(config, anchorHeight, placement);

  if (!isFiniteCartesian3(targetSphere.center) || !Number.isFinite(targetSphere.radius)) {
    flyToPreset(
      viewer,
      computeAnchorSphere(config, anchorHeight, placement),
      preset
    );
    return;
  }

  flyToPreset(viewer, buildCameraTargetSphere(targetSphere, preset), preset);
}

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
