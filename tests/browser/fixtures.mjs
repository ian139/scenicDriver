// Playwright fixtures: ephemeral fake server, offline network guard, and a
// controllable gate for basemap tile responses (the real MapLibre style
// loading dependency used by the deferred-render scenario).
import { test as base, expect } from "@playwright/test";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { BASEMAP_TILE_PNG, startFakeServer } from "./fake-api.mjs";

const require = createRequire(import.meta.url);
const MAPLIBRE_DIST = path.dirname(require.resolve("maplibre-gl/dist/maplibre-gl.js"));
// index.html loads these from unpkg; the lane serves the identical pinned
// release (tests/browser/package.json) from node_modules instead.
const LOCAL_ASSETS = new Map([
  ["https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js", ["maplibre-gl.js", "text/javascript"]],
  ["https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css", ["maplibre-gl.css", "text/css"]],
]);
const BASEMAP_TILE = /^https:\/\/[abc]\.basemaps\.cartocdn\.com\/rastertiles\/voyager\/\d+\/\d+\/\d+\.png$/;

class TileGate {
  constructor() {
    this.served = 0;
    this.held = [];
    this.holding = false;
  }
  hold() {
    this.holding = true;
  }
  async handle(route) {
    if (!this.holding) return this.#fulfill(route);
    this.held.push(route);
  }
  async release() {
    this.holding = false;
    const routes = this.held.splice(0);
    await Promise.all(routes.map((route) => this.#fulfill(route)));
    return routes.length;
  }
  async #fulfill(route) {
    this.served += 1;
    // MapLibre aborts tiles it no longer needs; a late fulfil is harmless.
    await route.fulfill({ status: 200, contentType: "image/png", body: BASEMAP_TILE_PNG }).catch(() => {});
  }
}

export const test = base.extend({
  fakeServer: [
    // eslint-disable-next-line no-empty-pattern
    async ({}, use) => {
      const server = await startFakeServer();
      await use(server);
      await server.close();
    },
    { scope: "worker" },
  ],
  baseURL: async ({ fakeServer }, use) => use(fakeServer.origin),
  tiles: async ({}, use) => use(new TileGate()),
  blockedRequests: [
    async ({ context, fakeServer, tiles }, use) => {
      const blocked = [];
      await context.route("**/*", async (route) => {
        const url = route.request().url();
        if (url.startsWith(`${fakeServer.origin}/`)) return route.continue();
        const asset = LOCAL_ASSETS.get(url);
        if (asset) {
          const body = await fs.readFile(path.join(MAPLIBRE_DIST, asset[0]));
          return route.fulfill({ status: 200, contentType: asset[1], body });
        }
        if (BASEMAP_TILE.test(url)) return tiles.handle(route);
        blocked.push(url);
        return route.abort("blockedbyclient");
      });
      await use(blocked);
      // Every scenario must stay fully offline.
      expect(blocked, "unexpected external requests").toEqual([]);
    },
    { auto: true },
  ],
  pageErrors: [
    async ({ page }, use) => {
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await use(errors);
      expect(errors, "uncaught page errors").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
