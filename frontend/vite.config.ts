import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { extname, relative, resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "../3D_visual/node_modules/vite-plugin-static-copy/dist/index.js";

const rootDir = __dirname;
const cesiumRoot = resolve(rootDir, "../3D_visual/node_modules/cesium");
const cesiumEngineRoot = resolve(rootDir, "../3D_visual/node_modules/@cesium/engine");
const cesiumWidgetsRoot = resolve(rootDir, "../3D_visual/node_modules/@cesium/widgets");
const tokenPath = resolve(rootDir, "../3D_visual/cesium_token.txt");
const agentTwinAssetRoot = resolve(rootDir, "../3D_visual/public");
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

const contentTypes: Record<string, string> = {
  ".json": "application/json; charset=utf-8",
  ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};

const devAgentTwinAssetPlugin: any = {
  name: "agent-twin-dev-assets",
  configureServer(server: any) {
    server.middlewares.use("/agent-twin-assets", (request: any, response: any, next: any) => {
      const requestPath = decodeURIComponent((request.url ?? "/").split("?")[0])
        .replace(/^\/agent-twin-assets\/?/, "")
        .replace(/^\/+/, "");
      const assetPath = resolve(agentTwinAssetRoot, requestPath);
      const assetRelativePath = relative(agentTwinAssetRoot, assetPath);

      if (assetRelativePath.startsWith("..") || assetRelativePath === "" || assetRelativePath.includes(":")) {
        next();
        return;
      }

      try {
        const assetStat = statSync(assetPath);
        if (!assetStat.isFile()) {
          next();
          return;
        }

        response.setHeader("Content-Type", contentTypes[extname(assetPath).toLowerCase()] ?? "application/octet-stream");
        response.setHeader("Content-Length", String(assetStat.size));
        createReadStream(assetPath).pipe(response);
      } catch {
        next();
      }
    });
  },
};

const config: any = {
  plugins: [reactPlugin, devAgentTwinAssetPlugin, staticCopyPlugin],
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
  build: {
    chunkSizeWarningLimit: 6200,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes("node_modules/cesium") || id.includes("node_modules/@cesium")) {
            return "cesium-runtime";
          }
          if (id.includes("framer-motion")) {
            return "motion-runtime";
          }
          if (id.includes("node_modules/react") || id.includes("node_modules/react-router")) {
            return "react-runtime";
          }
          return undefined;
        },
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
