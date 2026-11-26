#!/usr/bin/env node
// Entry point for the browser smoke lane. Skips cleanly (exit 0) when the
// pinned Playwright package or its Chromium build is not installed; otherwise
// runs the suite and propagates Playwright's exit code.
import { spawnSync } from "node:child_process";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

function skip(reason) {
  console.log(`browser smoke lane: SKIPPED — ${reason}`);
  console.log("Setup: (cd tests/browser && npm ci && npx playwright install chromium)");
  process.exit(0);
}

let cli;
let chromium;
try {
  cli = require.resolve("@playwright/test/cli");
  require.resolve("maplibre-gl/dist/maplibre-gl.js");
  ({ chromium } = await import("@playwright/test"));
} catch (error) {
  skip(`Playwright or MapLibre package unavailable (${error.code || error.message})`);
}
// Probe the exact (headless) Chromium build the suite launches. Only a missing
// browser install is a skip; any other launch failure is a real error.
try {
  const browser = await chromium.launch({ headless: true });
  await browser.close();
} catch (error) {
  if (String(error?.message).includes("Executable doesn't exist")) {
    skip(`Chromium for Playwright is not installed (${String(error.message).split("\n")[0]})`);
  }
  throw error;
}

const result = spawnSync(process.execPath, [cli, "test", "--config", path.join(HERE, "playwright.config.mjs"), ...process.argv.slice(2)], {
  cwd: HERE,
  stdio: "inherit",
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
