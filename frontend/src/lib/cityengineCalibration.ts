type CesiumModule = typeof import("cesium");
type SourceAxisToken = "x" | "y" | "z" | "-x" | "-y" | "-z";
type SourceCenterMode = "root-node-centroid" | "bounds-center";

export type SourceFrame = {
  east: SourceAxisToken;
  north: SourceAxisToken;
  up: SourceAxisToken;
};

export interface SceneConfig {
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
  cameraPresets?: {
    overview?: { heading: number; pitch: number; range: number; fitWholeModel?: boolean; focusRadius?: number };
  };
}

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

export type SourceMetadata = {
  sourceNormalizationMatrix: any;
  localFocusCenter: any;
  normalizedSizes: any;
};

const DEFAULT_SOURCE_FRAME: SourceFrame = { east: "x", north: "-z", up: "y" };

export function normalizeSceneConfig(raw: Partial<SceneConfig>): SceneConfig {
  return {
    modelUrl: raw.modelUrl ?? "/models/cityengine_scene.glb",
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
    verticalScale: raw.verticalScale ?? 1,
    sourceFrame: raw.sourceFrame ?? DEFAULT_SOURCE_FRAME,
    sourceCenterMode: raw.sourceCenterMode ?? "bounds-center",
    cameraPresets: raw.cameraPresets ?? {
      overview: { heading: 20, pitch: -42, range: 2400, fitWholeModel: true },
    },
  };
}

export function resolveModelAssetUrl(modelUrl: string) {
  if (modelUrl.startsWith("/agent-twin-assets/")) {
    return modelUrl;
  }
  if (modelUrl.startsWith("/models/")) {
    return modelUrl.replace("/models/", "/agent-twin-assets/models/");
  }
  return modelUrl;
}

function parseAxisToken(token: SourceAxisToken) {
  return token.startsWith("-")
    ? { axis: token.slice(1) as "x" | "y" | "z", sign: -1 as const }
    : { axis: token as "x" | "y" | "z", sign: 1 as const };
}

function getAxisValue(vector: { x: number; y: number; z: number }, axis: "x" | "y" | "z") {
  return axis === "x" ? vector.x : axis === "y" ? vector.y : vector.z;
}

function getAxisSpan(bounds: { min: any; max: any }, token: SourceAxisToken) {
  const parsed = parseAxisToken(token);
  return Math.abs(getAxisValue(bounds.max, parsed.axis) - getAxisValue(bounds.min, parsed.axis));
}

function getAxisBaseValue(bounds: { min: any; max: any }, token: SourceAxisToken) {
  const parsed = parseAxisToken(token);
  const min = getAxisValue(bounds.min, parsed.axis);
  const max = getAxisValue(bounds.max, parsed.axis);
  return parsed.sign > 0 ? min : max;
}

function parseGlbJson(buffer: ArrayBuffer): GlTfDocument {
  const header = new DataView(buffer, 0, 12);
  if (header.getUint32(0, true) !== 0x46546c67) {
    throw new Error("Invalid GLB header");
  }

  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const chunkHeader = new DataView(buffer, offset, 8);
    const chunkLength = chunkHeader.getUint32(0, true);
    const chunkType = chunkHeader.getUint32(4, true);
    offset += 8;
    if (chunkType === 0x4e4f534a) {
      const jsonText = new TextDecoder("utf-8").decode(new Uint8Array(buffer, offset, chunkLength));
      return JSON.parse(jsonText) as GlTfDocument;
    }
    offset += chunkLength;
  }

  throw new Error("GLB JSON chunk not found");
}

function getNodeMatrix(Cesium: CesiumModule, node: GlTfNode) {
  const { Cartesian3, Matrix4, Quaternion } = Cesium;
  if (Array.isArray(node.matrix) && node.matrix.length === 16) {
    return Matrix4.fromColumnMajorArray(node.matrix, new Matrix4());
  }

  const translation = node.translation ? Cartesian3.fromArray(node.translation) : new Cartesian3(0, 0, 0);
  const rotation = node.rotation
    ? new Quaternion(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3])
    : new Quaternion(0, 0, 0, 1);
  const scale = node.scale ? Cartesian3.fromArray(node.scale) : new Cartesian3(1, 1, 1);
  return Matrix4.fromTranslationQuaternionRotationScale(translation, rotation, scale);
}

function getBoundingCorners(Cesium: CesiumModule, min: number[], max: number[]) {
  const { Cartesian3 } = Cesium;
  return [
    new Cartesian3(min[0], min[1], min[2]),
    new Cartesian3(min[0], min[1], max[2]),
    new Cartesian3(min[0], max[1], min[2]),
    new Cartesian3(min[0], max[1], max[2]),
    new Cartesian3(max[0], min[1], min[2]),
    new Cartesian3(max[0], min[1], max[2]),
    new Cartesian3(max[0], max[1], min[2]),
    new Cartesian3(max[0], max[1], max[2]),
  ];
}

function collectSourceMetadata(Cesium: CesiumModule, gltf: GlTfDocument, config: SceneConfig): SourceMetadata {
  const { Cartesian3, Matrix4 } = Cesium;
  const nodes = gltf.nodes ?? [];
  const meshes = gltf.meshes ?? [];
  const accessors = gltf.accessors ?? [];
  const scene = (gltf.scenes ?? [])[gltf.scene ?? 0];
  const roots = scene?.nodes ?? [];

  if (roots.length === 0) {
    throw new Error("GLB scene has no root nodes");
  }

  const bounds = {
    min: new Cartesian3(Infinity, Infinity, Infinity),
    max: new Cartesian3(-Infinity, -Infinity, -Infinity),
  };
  const rootTranslations: any[] = [];

  const updateBounds = (point: any) => {
    bounds.min.x = Math.min(bounds.min.x, point.x);
    bounds.min.y = Math.min(bounds.min.y, point.y);
    bounds.min.z = Math.min(bounds.min.z, point.z);
    bounds.max.x = Math.max(bounds.max.x, point.x);
    bounds.max.y = Math.max(bounds.max.y, point.y);
    bounds.max.z = Math.max(bounds.max.z, point.z);
  };

  const walkNode = (nodeIndex: number, parentMatrix: any) => {
    const node = nodes[nodeIndex];
    if (!node) {
      return;
    }

    const localMatrix = getNodeMatrix(Cesium, node);
    const worldMatrix = Matrix4.multiply(parentMatrix, localMatrix, new Matrix4());

    if (node.mesh !== undefined) {
      const mesh = meshes[node.mesh];
      for (const primitive of mesh?.primitives ?? []) {
        const accessorIndex = primitive.attributes?.POSITION;
        const accessor = accessorIndex !== undefined ? accessors[accessorIndex] : undefined;
        if (!accessor?.min || !accessor?.max) {
          continue;
        }
        for (const corner of getBoundingCorners(Cesium, accessor.min, accessor.max)) {
          updateBounds(Matrix4.multiplyByPoint(worldMatrix, corner, new Cartesian3()));
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
    const rootMatrix = getNodeMatrix(Cesium, rootNode);
    rootTranslations.push(Matrix4.getTranslation(rootMatrix, new Cartesian3()));
    walkNode(rootIndex, Matrix4.IDENTITY);
  }

  if (!Number.isFinite(bounds.min.x) || rootTranslations.length === 0) {
    throw new Error("Unable to derive GLB source bounds");
  }

  const summedRootCenter = rootTranslations.reduce(
    (sum, current) => new Cartesian3(sum.x + current.x, sum.y + current.y, sum.z + current.z),
    new Cartesian3(0, 0, 0),
  );
  const sourceRootCenter = new Cartesian3(
    summedRootCenter.x / rootTranslations.length,
    summedRootCenter.y / rootTranslations.length,
    summedRootCenter.z / rootTranslations.length,
  );
  const worldBoundsCenter = new Cartesian3(
    (bounds.min.x + bounds.max.x) / 2,
    (bounds.min.y + bounds.max.y) / 2,
    (bounds.min.z + bounds.max.z) / 2,
  );
  const horizontalCenter = config.sourceCenterMode === "root-node-centroid" ? sourceRootCenter : worldBoundsCenter;
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
    1,
  ]);

  return {
    sourceNormalizationMatrix,
    localFocusCenter: Matrix4.multiplyByPoint(sourceNormalizationMatrix, worldBoundsCenter, new Cartesian3()),
    normalizedSizes: new Cartesian3(
      getAxisSpan(bounds, config.sourceFrame.east),
      getAxisSpan(bounds, config.sourceFrame.north),
      getAxisSpan(bounds, config.sourceFrame.up),
    ),
  };
}

export async function loadSourceMetadata(Cesium: CesiumModule, config: SceneConfig) {
  const response = await fetch(resolveModelAssetUrl(config.modelUrl), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`GLB source metadata returned ${response.status}`);
  }
  return collectSourceMetadata(Cesium, parseGlbJson(await response.arrayBuffer()), config);
}

export function buildPlacementFrame(Cesium: CesiumModule, config: SceneConfig, anchorHeight: number) {
  const { Cartesian3, HeadingPitchRoll, Math: CesiumMath, Matrix4, Quaternion, Transforms } = Cesium;
  const anchor = Cartesian3.fromDegrees(config.anchorLon, config.anchorLat, anchorHeight);
  const calibration = Matrix4.fromTranslationQuaternionRotationScale(
    new Cartesian3(config.offsetEast, config.offsetNorth, config.offsetUp),
    Quaternion.fromHeadingPitchRoll(
      new HeadingPitchRoll(
        CesiumMath.toRadians(config.heading),
        CesiumMath.toRadians(config.pitch),
        CesiumMath.toRadians(config.roll),
      ),
    ),
    new Cartesian3(config.scale, config.scale, config.scale * config.verticalScale),
  );
  return Matrix4.multiply(Transforms.eastNorthUpToFixedFrame(anchor), calibration, new Matrix4());
}

export function buildSimpleModelMatrix(Cesium: CesiumModule, config: SceneConfig, anchorHeight: number) {
  return buildPlacementFrame(Cesium, config, anchorHeight);
}

export function buildCalibratedModelMatrix(
  Cesium: CesiumModule,
  config: SceneConfig,
  sourceMetadata: SourceMetadata,
  anchorHeight: number,
) {
  const { Matrix4 } = Cesium;
  return Matrix4.multiply(
    buildPlacementFrame(Cesium, config, anchorHeight),
    sourceMetadata.sourceNormalizationMatrix,
    new Matrix4(),
  );
}

export function computeModelFocusSphere(
  Cesium: CesiumModule,
  config: SceneConfig,
  sourceMetadata: SourceMetadata,
  anchorHeight: number,
) {
  const { BoundingSphere, Cartesian3, Matrix4 } = Cesium;
  const placementFrame = buildPlacementFrame(Cesium, config, anchorHeight);
  const center = Matrix4.multiplyByPoint(placementFrame, sourceMetadata.localFocusCenter, new Cartesian3());
  const scaledSize = new Cartesian3(
    sourceMetadata.normalizedSizes.x * config.scale,
    sourceMetadata.normalizedSizes.y * config.scale,
    sourceMetadata.normalizedSizes.z * config.scale * config.verticalScale,
  );
  const radius = Math.sqrt(scaledSize.x * scaledSize.x + scaledSize.y * scaledSize.y + scaledSize.z * scaledSize.z) / 2;
  return new BoundingSphere(center, Math.max(radius, 1));
}
