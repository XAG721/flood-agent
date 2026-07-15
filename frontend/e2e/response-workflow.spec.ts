import { expect, test } from "@playwright/test";

test("响应工作台在桌面与移动视口中保持可读、可追溯且无控制台错误", async ({ page }, testInfo) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));

  await page.goto("/response");
  await expect(page.getByRole("status").filter({ hasText: "受控模拟环境" })).toBeVisible();
  await expect(page.getByRole("heading", { name: /碑林区.*响应事件/ })).toBeVisible();
  await expect(page.getByText(/预警版本 V\d+/)).toBeVisible();
  await expect(page.getByRole("list", { name: "预警响应业务流程" })).toContainText("人工审批");
  await expect(page.getByText(/deterministic|模拟|STALE/).first()).toBeVisible();

  const eventHeading = await page.getByRole("heading", { name: /碑林区.*响应事件/ }).innerText();
  await page.reload();
  await expect(page.getByRole("status").filter({ hasText: "受控模拟环境" })).toBeVisible();
  await expect(page.getByRole("heading", { name: eventHeading })).toBeVisible();
  await expect(page.getByText(/预警版本 V\d+/)).toBeVisible();

  const desktopOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(desktopOverflow).toBeLessThanOrEqual(1);
  await page.screenshot({ path: testInfo.outputPath("response-workbench-desktop.png"), fullPage: true });

  const guardedAction = page.getByRole("button", { name: /批准并模拟下发|申请延期|人工接管/ }).first();
  if (await guardedAction.isVisible()) {
    page.once("dialog", async (dialog) => {
      expect(dialog.message()).toMatch(/原因|理由/);
      await dialog.dismiss();
    });
    await guardedAction.click();
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("status").filter({ hasText: "受控模拟环境" })).toBeVisible();
  const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(mobileOverflow).toBeLessThanOrEqual(1);
  expect(consoleErrors).toEqual([]);
});

test("保留的旧风险调度页面可运行并形成回退截图基线", async ({ page }, testInfo) => {
  await page.goto("/operations");
  await expect(page.getByRole("link", { name: "风险预警" }).first()).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("link", { name: "响应闭环" }).first()).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("legacy-operations-baseline.png"), fullPage: true });
});
