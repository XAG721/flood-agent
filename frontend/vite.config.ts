import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { createHmac, randomUUID } from "node:crypto";
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
const trustedIdentitySecret = process.env.FLOOD_TRUSTED_IDENTITY_SECRET
  ?? "local-development-identity-secret-change-before-production-v1";
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
        configure(proxy: any) {
          proxy.on("proxyReq", (proxyRequest: any, incomingRequest: any) => {
            const operatorId = String(incomingRequest.headers["x-operator-id"] ?? "operator_console");
            const operatorRole = String(incomingRequest.headers["x-operator-role"] ?? "commander");
            const terminalId = String(incomingRequest.headers["x-operator-terminal"] ?? "district-web-console");
            const assuranceLevel = ["commander", "reviewer", "admin"].includes(operatorRole) ? "aal2" : "aal1";
            const timestamp = String(Math.floor(Date.now() / 1000));
            const nonce = randomUUID().replace(/-/g, "");
            const path = new URL(proxyRequest.path, "http://127.0.0.1").pathname;
            const canonical = [
              operatorId,
              operatorRole,
              assuranceLevel,
              terminalId,
              timestamp,
              nonce,
              String(incomingRequest.method ?? "GET").toUpperCase(),
              path,
            ].join("\n");
            const signature = createHmac("sha256", trustedIdentitySecret).update(canonical).digest("hex");
            proxyRequest.setHeader("X-Identity-Operator-Id", operatorId);
            proxyRequest.setHeader("X-Identity-Operator-Role", operatorRole);
            proxyRequest.setHeader("X-Identity-Assurance-Level", assuranceLevel);
            proxyRequest.setHeader("X-Identity-Terminal-Id", terminalId);
            proxyRequest.setHeader("X-Identity-Timestamp", timestamp);
            proxyRequest.setHeader("X-Identity-Nonce", nonce);
            proxyRequest.setHeader("X-Identity-Signature", signature);
          });
        },
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
