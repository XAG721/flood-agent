/// <reference types="vite/client" />

declare global {
  const __CESIUM_ION_TOKEN__: string;

  interface Window {
    __GLB_DEBUG__?: Record<string, unknown>;
  }
}

export {};
