import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "../3D_visual/node_modules/vite-plugin-static-copy/dist/index.js";

const rootDir = __dirname;
const cesiumRoot = resolve(rootDir, "../3D_visual/node_modules/cesium");
const cesiumEngineRoot = resolve(rootDir, "../3D_visual/node_modules/@cesium/engine");
const cesiumWidgetsRoot = resolve(rootDir, "../3D_visual/node_modules/@cesium/widgets");
const tokenPath = resolve(rootDir, "../3D_visual/cesium_token.txt");
const cesiumToken = existsSync(tokenPath) ? readFileSync(tokenPath, "utf8").trim() : "";
const reactPlugin: any = (react as any)();
const staticCopyPlugin: any = (viteStaticCopy as any)({
  targets: [
    {
      src: "../3D_visual/public/scene-config.json",
      dest: "agent-twin-assets",
    },
    {
      src: "../3D_visual/public/models/*",
      dest: "agent-twin-assets/models",
    },
    {
      src: "../3D_visual/node_modules/cesium/Build/Cesium/Workers",
      dest: "cesium",
    },
    {
      src: "../3D_visual/node_modules/cesium/Build/Cesium/ThirdParty",
      dest: "cesium",
    },
    {
      src: "../3D_visual/node_modules/cesium/Build/Cesium/Assets",
      dest: "cesium",
    },
    {
      src: "../3D_visual/node_modules/cesium/Build/Cesium/Widgets",
      dest: "cesium",
    },
  ],
});

const config: any = {
  plugins: [reactPlugin, staticCopyPlugin],
  define: {
    CESIUM_BASE_URL: JSON.stringify("/cesium"),
    __CESIUM_ION_TOKEN__: JSON.stringify(cesiumToken),
  },
  resolve: {
    alias: {
      cesium: cesiumRoot,
      "@cesium/engine": cesiumEngineRoot,
      "@cesium/widgets": cesiumWidgetsRoot,
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
  },
};

export default config;
