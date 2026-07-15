import { defineConfig, devices } from "@playwright/test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const externalBaseUrl = process.env.PLAYWRIGHT_BASE_URL?.replace(/\/$/, "");
const configDir = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: "./e2e",
  outputDir: "../tmp/playwright-results",
  reporter: process.env.CI ? [["line"], ["html", { outputFolder: "../tmp/playwright-report", open: "never" }]] : "line",
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: externalBaseUrl ?? "http://127.0.0.1:4173",
    locale: "zh-CN",
    timezoneId: "Asia/Shanghai",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: externalBaseUrl
    ? undefined
    : [
        {
          command: "python -m uvicorn flood_system.api:app --host 127.0.0.1 --port 8000",
          cwd: resolve(configDir, ".."),
          env: {
            ...process.env,
            FLOOD_DB_PATH: resolve(configDir, "../tmp/playwright-response.db"),
            FLOOD_ALLOW_DEV_IDENTITY_HEADERS: "1",
            FLOOD_ENVIRONMENT: "development",
            FLOOD_SUPERVISOR_LOOP_ENABLED: "0",
          },
          url: "http://127.0.0.1:8000/ready",
          reuseExistingServer: !process.env.CI,
          timeout: 60_000,
        },
        {
          command: "npm run dev -- --host 127.0.0.1 --port 4173",
          cwd: configDir,
          url: "http://127.0.0.1:4173/response",
          reuseExistingServer: !process.env.CI,
          timeout: 60_000,
        },
      ],
});
