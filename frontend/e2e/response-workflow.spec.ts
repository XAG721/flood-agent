import { expect, test } from "@playwright/test";

test("响应工作台在桌面与移动视口中保持可读、可追溯且无控制台错误", async ({ page }) => {
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

  const desktopOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(desktopOverflow).toBeLessThanOrEqual(1);

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
