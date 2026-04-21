import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const rootDir = __dirname;
const tokenPath = resolve(rootDir, "cesium_token.txt");
const cesiumToken = existsSync(tokenPath)
  ? readFileSync(tokenPath, "utf8").trim()
  : "";

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/cesium/Build/Cesium/Workers",
          dest: "cesium"
        },
        {
          src: "node_modules/cesium/Build/Cesium/ThirdParty",
          dest: "cesium"
        },
        {
          src: "node_modules/cesium/Build/Cesium/Assets",
          dest: "cesium"
        },
        {
          src: "node_modules/cesium/Build/Cesium/Widgets",
          dest: "cesium"
        }
      ]
    })
  ],
  define: {
    CESIUM_BASE_URL: JSON.stringify("/cesium"),
    __CESIUM_ION_TOKEN__: JSON.stringify(cesiumToken)
  },
  server: {
    host: "127.0.0.1",
    port: 4173
  }
});
