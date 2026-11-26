import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.mjs",
  outputDir: "test-results",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  forbidOnly: true,
  reporter: [["list"]],
  use: {
    browserName: "chromium",
    headless: true,
    permissions: ["clipboard-read", "clipboard-write"],
    reducedMotion: "no-preference",
    serviceWorkers: "block",
    trace: "retain-on-failure",
    launchOptions: {
      // Software WebGL so MapLibre renders on GPU-less hosts and CI.
      args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
    },
  },
});
