import {
  BoundingSphere,
  Cartesian3,
  HeadingPitchRange,
  HeadingPitchRoll,
  Math as CesiumMath,
  Matrix4,
  Quaternion,
  Transforms,
  Viewer,
} from "cesium";

import { DEFAULT_SOURCE_FRAME, LEGACY_STORAGE_KEYS, STORAGE_KEY } from "./sceneConfig";
import type {
  CameraPreset,
  GlTfDocument,
  GlTfNode,
  PlacementState,
  SceneConfig,
  SourceAxisToken,
  SourceBounds,
  SourceMetadata,
} from "./sceneTypes";

export function placementFromConfig(config: SceneConfig): PlacementState {
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

export function readSavedPlacement(defaults: PlacementState): Partial<PlacementState> {
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

export function normalizeSceneConfig(raw: Partial<SceneConfig>): SceneConfig {
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

export async function loadSourceMetadata(config: SceneConfig) {
  const response = await fetch(config.modelUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`GLB 读取失败：${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  return collectSourceMetadata(parseGlbJson(buffer), config);
}

export function computeModelMatrix(
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

export function buildCopyPayload(config: SceneConfig, placement: PlacementState) {
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

export function flyToPresetWithFallback(
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
