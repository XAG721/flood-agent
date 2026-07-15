import type { Cartesian3, Matrix4 } from "cesium";

export type CameraKey = "overview" | "oblique" | "close";
export type SourceAxisToken = "x" | "y" | "z" | "-x" | "-y" | "-z";
export type SourceCenterMode = "root-node-centroid" | "bounds-center";
export type TerrainAnchorMode = "sample-most-detailed";
export type DiagnosticMode = "default" | "no-cull" | "no-backface-cull";

export type CameraPreset = {
  heading: number;
  pitch: number;
  range: number;
  fitWholeModel?: boolean;
  focusRadius?: number;
};
export type DebugFlags = {
  showAnchor: boolean;
  showBoundingSphere: boolean;
  brightMaterial: boolean;
};
export type SourceFrame = {
  east: SourceAxisToken;
  north: SourceAxisToken;
  up: SourceAxisToken;
};
export type SceneConfig = {
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
export type PlacementState = {
  scale: number;
  heading: number;
  pitch: number;
  roll: number;
  offsetEast: number;
  offsetNorth: number;
  offsetUp: number;
};
export type SourceBounds = { min: Cartesian3; max: Cartesian3 };
export type SourceMetadata = {
  sourceRootCenter: Cartesian3;
  sourceBounds: SourceBounds;
  computedBaseHeight: number;
  sourceNormalizationMatrix: Matrix4;
  localFocusCenter: Cartesian3;
  normalizedSizes: Cartesian3;
};
export type DiagnosticCullResult = {
  activeMode: DiagnosticMode;
  defaultCull: boolean;
  defaultBackFaceCulling: boolean;
  runtimeMutable: boolean;
};
export type GlTfNode = {
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
};
export type GlTfPrimitive = { attributes?: { POSITION?: number } };
export type GlTfMesh = { primitives?: GlTfPrimitive[] };
export type GlTfAccessor = { min?: number[]; max?: number[] };
export type GlTfScene = { nodes?: number[] };
export type GlTfDocument = {
  scene?: number;
  scenes?: GlTfScene[];
  nodes?: GlTfNode[];
  meshes?: GlTfMesh[];
  accessors?: GlTfAccessor[];
};
