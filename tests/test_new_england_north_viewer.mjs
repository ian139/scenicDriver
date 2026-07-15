import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const viewerSource = fs.readFileSync(
  new URL("../apps/new_england_north/viewer.js", import.meta.url),
  "utf8"
);
const installStart = viewerSource.indexOf("function installAddressSearch");
const installEnd = viewerSource.indexOf("\n\nasync function planRoute", installStart);
assert.ok(installStart >= 0 && installEnd > installStart);
const installSource = viewerSource.slice(installStart, installEnd);
const failureStart = viewerSource.indexOf("function routeFailurePresentation");
const failureEnd = viewerSource.indexOf("\n\nconst REASON_PROSE", failureStart);
assert.ok(failureStart >= 0 && failureEnd > failureStart);
const failureSource = viewerSource.slice(failureStart, failureEnd);
const regionStart = viewerSource.indexOf("function resolveRegionSelection");
const regionEnd = viewerSource.indexOf("\n\nfunction setText", regionStart);
assert.ok(regionStart >= 0 && regionEnd > regionStart);
const regionSource = viewerSource.slice(regionStart, regionEnd);

class FakeElement {
  constructor(tag = "div") {
    this.tagName = tag;
    this.value = "";
    this.children = [];
    this.dataset = {};
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = {
      add: () => {},
      remove: () => {},
    };
  }

  addEventListener(type, callback) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(callback);
    this.listeners.set(type, callbacks);
  }

  dispatch(type, event = {}) {
    for (const callback of this.listeners.get(type) || []) {
      callback({ target: this, relatedTarget: null, preventDefault() {}, ...event });
    }
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  replaceChildren(...children) {
    this.children = children;
  }

  append(...children) {
    this.children.push(...children);
  }

  appendChild(child) {
    this.children.push(child);
  }
  contains() {
    return false;
  }

  focus() {}
}

function makeRegionHarness(search = "", payload = { regions: [] }) {
  const regionSelect = new FakeElement("select");
  const runSelect = new FakeElement("select");
  const regionStatus = new FakeElement("span");
  const replacements = [];
  const location = new URL(`http://test.local/viewer${search}`);
  const context = {
    URL,
    Number,
    String,
    Array,
    Error,
    DEFAULTS: {
      displayRange: "new_england_north",
      sourceRegion: "new_england_north",
      workingRun: "new_england_north_z14_v6_learned",
      zoom: 6.2,
    },
    REGION_BOUNDS: [
      [-73.5205078125, 42.488301979602255],
      [-66.796875, 47.50235895196859],
    ],
    activeRegionBounds: null,
    selectedRegionMetadata: null,
    CONFIG: {
      displayRange: "new_england_north",
      sourceRegion: new URLSearchParams(search).get("region") ||
        new URLSearchParams(search).get("source") ||
        "new_england_north",
      workingRun: new URLSearchParams(search).get("run") ||
        "new_england_north_z14_v6_learned",
    },
    params: new URLSearchParams(search),
    el: { regionSelect, runSelect, regionStatus },
    document: { createElement: (tag) => new FakeElement(tag) },
    window: {
      location,
      history: {
        replaceState(_state, _title, url) {
          replacements.push(String(url));
          context.window.location = new URL(String(url));
        },
      },
    },
    api: (path) => path,
    fetch: async () => ({ ok: true, json: async () => payload }),
    setText: (node, text) => {
      if (node) node.textContent = text;
    },
  };
  vm.runInNewContext(
    `${regionSource}
globalThis.resolveRegionSelection = resolveRegionSelection;
globalThis.selectionUrl = selectionUrl;
globalThis.validHeatmapMetadata = validHeatmapMetadata;
globalThis.loadSupportedRegions = loadSupportedRegions;`,
    context
  );
  return { context, regionSelect, runSelect, regionStatus, replacements };
}

const supportedRegions = {
  regions: [
    {
      region: "new_england_north",
      display_name: "New England North",
      latest_run_name: "new_england_north_z14_v6_learned",
      graph_exists: true,
      is_default: true,
      bbox: {
        min_lat: 42.48,
        min_lon: -73.52,
        max_lat: 47.5,
        max_lon: -66.79,
      },
      map: { center: { lat: 44.99, lon: -70.15 }, zoom: 6.2 },

    },
    {
      region: "masswhites",
      display_name: "Masswhites",
      latest_run_name: "masswhites_z14_learned_h4_v2",
      graph_exists: true,
      is_default: false,
      bbox: {
        min_lat: 41.19,
        min_lon: -73.52,
        max_lat: 44.51,
        max_lon: -72.97,
      },
      map: { center: { lat: 42.85, lon: -73.22 }, zoom: 8.2 },
    },
  ],
};

test("default region metadata keeps canonical URL and selection unchanged", async () => {
  const harness = makeRegionHarness("", supportedRegions);
  await harness.context.loadSupportedRegions();
  assert.equal(harness.context.CONFIG.sourceRegion, "new_england_north");
  assert.equal(
    harness.context.CONFIG.workingRun,
    "new_england_north_z14_v6_learned"
  );
  assert.equal(harness.context.window.location.search, "");
  assert.deepEqual(harness.replacements, []);
  assert.equal(harness.regionSelect.children.length, 2);
  assert.equal(harness.regionSelect.children[0].selected, true);
});

test("URL-selected region resolves its run and persists source/run alignment", async () => {
  const harness = makeRegionHarness(
    "?source=masswhites&run=stale-run",
    supportedRegions
  );
  await harness.context.loadSupportedRegions();
  assert.equal(harness.context.CONFIG.sourceRegion, "masswhites");
  assert.equal(
    harness.context.CONFIG.workingRun,
    "masswhites_z14_learned_h4_v2"
  );
  assert.equal(harness.replacements.length, 1);
  const persisted = new URL(harness.replacements[0]);
  assert.equal(persisted.searchParams.get("source"), "masswhites");
  assert.equal(persisted.searchParams.get("run"), "masswhites_z14_learned_h4_v2");
  assert.equal(persisted.searchParams.has("region"), false);
  assert.equal(
    harness.runSelect.children[0].value,
    "masswhites_z14_learned_h4_v2"
  );
});

test("supported-region heatmap metadata accepts noncanonical tile zooms", () => {
  const harness = makeRegionHarness("", supportedRegions);
  assert.equal(
    harness.context.validHeatmapMetadata({
      tile_zoom: 16,
      bounds: { min_lon: -75.23, min_lat: 40.01, max_lon: -75.18, max_lat: 40.08 },
    }),
    true
  );
  assert.equal(
    harness.context.validHeatmapMetadata({
      tile_zoom: 0,
      bounds: { min_lon: -75.23, min_lat: 40.01, max_lon: -75.18, max_lat: 40.08 },
    }),
    false
  );
});

test("invalid and unavailable metadata fall back or report no run without crashing", async () => {
  const fallback = makeRegionHarness("?source=unknown&run=missing", supportedRegions);
  await fallback.context.loadSupportedRegions();
  assert.equal(fallback.context.CONFIG.sourceRegion, "new_england_north");
  assert.match(fallback.replacements[0], /source=new_england_north/);

  const noRunPayload = {
    regions: [
      {
        region: "new_england_north",
        display_name: "New England North",
        latest_run_name: null,
        is_default: true,
      },
    ],
  };
  const unavailable = makeRegionHarness("", noRunPayload);
  await unavailable.context.loadSupportedRegions();
  assert.equal(unavailable.context.CONFIG.workingRun, null);
  assert.equal(unavailable.runSelect.disabled, true);
  assert.match(unavailable.regionStatus.textContent, /no configured scenic run/i);

  const empty = makeRegionHarness("", { regions: [] });
  await assert.rejects(
    () => empty.context.loadSupportedRegions(),
    /no supported regions/i
  );
});

function makeHarness() {
  const input = new FakeElement("input");
  const suggestions = new FakeElement("div");
  const clear = new FakeElement("button");
  const selectedRoutePoints = { start: null, end: null };
  const pending = [];
  let sequence = 0;
  const context = {
    CONFIG: { sourceRegion: "test-region" },
    selectedRoutePoints,
    crypto: { randomUUID: () => `session-${++sequence}` },
    AbortController,
    URL: class TestURL extends URL {
      constructor(path, base) {
        super(path, base || "http://test.local");
      }
    },
    setTimeout,
    clearTimeout,
    Number,
    JSON,
    String,
    Math,
    console,
    document: {
      createElement: (tag) => new FakeElement(tag),
    },
    api: (path) => path,
    parsePoint: (value) => {
      const match = String(value).match(/^(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)$/);
      return match ? { lat: Number(match[1]), lon: Number(match[2]) } : null;
    },
    fetch: (url, options) => {
      const record = { url: String(url), options };
      pending.push(record);
      return new Promise((resolve, reject) => {
        record.resolve = resolve;
        record.reject = reject;
        options.signal.addEventListener("abort", () => {
          const error = new Error("aborted");
          error.name = "AbortError";
          reject(error);
        });
      });
    },
  };
  vm.runInNewContext(
    `function newSearchSessionToken() { return "session-fixed"; }\n${installSource}\nglobalThis.installAddressSearch = installAddressSearch;`,
    context
  );
  context.el = {
    startInput: input,
    startSuggestions: suggestions,
    clearStartBtn: clear,
    endInput: new FakeElement("input"),
    endSuggestions: new FakeElement("div"),
    clearEndBtn: new FakeElement("button"),
  };
  context.installAddressSearch("start");
  return { context, input, suggestions, clear, pending, selectedRoutePoints };
}

const response = (payload) => ({ ok: true, json: async () => payload });
const tick = (ms = 230) => new Promise((resolve) => setTimeout(resolve, ms));

test("route failure presentation distinguishes no-route constraints", () => {
  const context = {};
  vm.runInNewContext(
    `${failureSource}\nglobalThis.routeFailurePresentation = routeFailurePresentation;`,
    context
  );
  const noRoute = context.routeFailurePresentation("no_route_found");
  assert.equal(noRoute.status, "API: no route found");
  assert.equal(noRoute.title, "No route found");
  const other = context.routeFailurePresentation("server_error");
  assert.equal(other.status, "API: route failed");
  assert.equal(other.title, "Route failed");
});

test("address suggest/retrieve guards reject stale responses and close invalidates", async () => {
  const harness = makeHarness();
  const { input, suggestions, pending, selectedRoutePoints } = harness;

  input.value = "old query";
  input.dispatch("input");
  await tick();
  assert.equal(pending.length, 1);
  input.value = "new query";
  input.dispatch("input");
  await tick();
  assert.equal(pending.length, 2);

  pending[0].resolve(response({ suggestions: [{ mapbox_id: "old", name: "Old" }] }));
  pending[1].resolve(
    response({
      suggestions: [
        { mapbox_id: "first", name: "First" },
        { mapbox_id: "second", name: "Second" },
      ],
    })
  );
  await tick(20);
  assert.equal(suggestions.children.length, 2);
  assert.equal(suggestions.children[0].children[0].textContent, "First");

  const firstSuggestion = suggestions.children[0];
  const secondSuggestion = suggestions.children[1];
  firstSuggestion.dispatch("mousedown");
  secondSuggestion.dispatch("mousedown");
  assert.equal(pending.length, 4);
  pending[2].resolve(
    response({ result: { lat: 1, lon: 1, full_address: "Old address" } })
  );
  pending[3].resolve(
    response({ result: { lat: 2, lon: 2, full_address: "New address" } })
  );
  await tick(0);
  assert.equal(selectedRoutePoints.start.lat, 2);
  assert.equal(selectedRoutePoints.start.lon, 2);
  assert.equal(input.value, "New address");

  // A delayed suggest response after Escape must not reopen the list.
  input.value = "delayed query";
  input.dispatch("input");
  await tick();
  const delayed = pending.at(-1);
  input.dispatch("keydown", { key: "Escape" });
  delayed.resolve(response({ suggestions: [{ mapbox_id: "late", name: "Late" }] }));
  await tick(0);
  assert.equal(suggestions.children.length, 0);
});
