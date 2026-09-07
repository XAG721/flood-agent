import type { SourceFrame } from "./sceneTypes";

export const STORAGE_KEY = "cityengine-glb-viewer-placement-v8";
export const LEGACY_STORAGE_KEYS = [
  "cityengine-glb-viewer-placement-v1",
  "cityengine-glb-viewer-placement-v2",
  "cityengine-glb-viewer-placement-v3",
  "cityengine-glb-viewer-placement-v4",
  "cityengine-glb-viewer-placement-v5",
  "cityengine-glb-viewer-placement-v6",
  "cityengine-glb-viewer-placement-v7"
] as const;
export const DEFAULT_SOURCE_FRAME: SourceFrame = { east: "x", north: "-z", up: "y" };
