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

const routeStart = viewerSource.indexOf("function routeLayer");
const routeEnd = viewerSource.indexOf("\n\nfunction clearRoute", routeStart);
assert.ok(routeStart >= 0 && routeEnd > routeStart);
const routeSource = viewerSource.slice(routeStart, routeEnd);
const failureStart = viewerSource.indexOf("function routeFailurePresentation");
const failureEnd = viewerSource.indexOf("\n\nconst REASON_PROSE", failureStart);
assert.ok(failureStart >= 0 && failureEnd > failureStart);
const failureSource = viewerSource.slice(failureStart, failureEnd);
const regionStart = viewerSource.indexOf("function resolveRegionSelection");
const regionEnd = viewerSource.indexOf("\n\nfunction setText", regionStart);
assert.ok(regionStart >= 0 && regionEnd > regionStart);
const regionSource = viewerSource.slice(regionStart, regionEnd);
const formatterStart = viewerSource.indexOf("function escapeHtml");
const formatterEnd = viewerSource.indexOf("\n\nfunction renderTrainingResults", formatterStart);
assert.ok(formatterStart >= 0 && formatterEnd > formatterStart);
const formatterSource = viewerSource.slice(formatterStart, formatterEnd);
const displayStart = viewerSource.indexOf("function displayRatio");
const displayEnd = viewerSource.indexOf("\n\nfunction formatWithUnit", displayStart);
assert.ok(displayStart >= 0 && displayEnd > displayStart);
const displaySource = viewerSource.slice(displayStart, displayEnd);
const diagnosticsStart = viewerSource.indexOf("function routeDiagnosticsMarkup");
const diagnosticsEnd = viewerSource.indexOf("\n\nfunction routeOutputMarkup", diagnosticsStart);
assert.ok(diagnosticsStart >= 0 && diagnosticsEnd > diagnosticsStart);
const diagnosticsSource = viewerSource.slice(diagnosticsStart, diagnosticsEnd);
const outputStart = viewerSource.indexOf("function routeOutputMarkup");
const outputEnd = viewerSource.indexOf("\n\nfunction setRouteResultsVerbose", outputStart);
assert.ok(outputStart >= 0 && outputEnd > outputStart);
const outputSource = viewerSource.slice(outputStart, outputEnd);

function makeRouteOutputHarness() {
  const context = {
    Number,
    String,
    Array,
    Object,
    Math,
    Set,
    ROUTE_SOURCE: "route-source",
    ROUTE_BASELINE: "route-baseline",
    ROUTE_SCENIC: "route-scenic",
    ROUTE_ENDPOINT_SOURCE: "route-endpoint-source",
    ROUTE_ENDPOINT_CONNECTORS: "route-endpoint-connectors",
  };
  vm.runInNewContext(
    `${formatterSource}
${displaySource}
${diagnosticsSource}
${routeSource}
let verboseRouteResults = false;
${outputSource}
globalThis.routeOutputMarkup = routeOutputMarkup;
globalThis.setVerboseRouteResultsForTest = (value) => {
  verboseRouteResults = Boolean(value);
};`,
    context
  );
  return context;
}




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
      workingRun: "prompt_two_candidate_exp02_fresh_test20_20260810",
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
        "prompt_two_candidate_exp02_fresh_test20_20260810",
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
globalThis.routePlanningIntentionallyDisabled = routePlanningIntentionallyDisabled;
globalThis.routePlanningAvailable = routePlanningAvailable;
globalThis.loadSupportedRegions = loadSupportedRegions;`,
    context
  );
  return { context, regionSelect, runSelect, regionStatus, replacements };
}


test("route output stays compact until diagnostics are explicitly enabled", () => {
  const harness = makeRouteOutputHarness();
  const payload = {
    routes: { scenic: { highway_count: 1, score_run: ["edge-1"] } },
    diagnostics: {
      planning_elapsed_ms: 24,
      requested_scenic_weight: 0.8,
    },
  };
  const compact = harness.routeOutputMarkup(payload);
  assert.equal(compact, "Scenic and baseline routes are shown on the map.");
  assert.doesNotMatch(compact, /route-diagnostics|Planning/);

  harness.setVerboseRouteResultsForTest(true);
  const verbose = harness.routeOutputMarkup(payload);
  assert.match(verbose, /route-diagnostics/);
  assert.match(verbose, /Planning 24 ms/);
  assert.doesNotMatch(verbose, /--/);
  assert.equal(
    harness.routeOutputMarkup({
      routes: { scenic: {} },
      diagnostics: { planning_elapsed_ms: null, score_mapping_coverage: null },
    }),
    "Scenic and baseline routes are shown on the map."
  );
  assert.equal(
    harness.routeOutputMarkup({}),
    "Scenic and baseline routes are shown on the map."
  );
});

test("route output identifies road-snapped selected endpoints", () => {
  const harness = makeRouteOutputHarness();
  const output = harness.routeOutputMarkup({
    geojson: {
      features: [
        {
          properties: {
            route_kind: "scenic",
            snapped_start: [44.47588, -73.214],
            snapped_end: [44.80162, -68.7713],
          },
        },
      ],
    },
  });
  assert.match(output, /Road-snapped endpoints/);
  assert.match(output, /44\.47588/);
  assert.match(output, /-68\.77130/);
});



const supportedRegions = {
  regions: [
    {
      region: "new_england_north",
      display_name: "New England North",
      latest_run_name: "prompt_two_candidate_exp02_fresh_test20_20260810",
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
    "prompt_two_candidate_exp02_fresh_test20_20260810"
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

test("route capability distinguishes disabled routing from a missing graph", () => {
  const harness = makeRegionHarness("", supportedRegions);
  const intentionallyDisabled = {
    route_planning_enabled: false,
    graph_exists: false,
  };
  const missingConfiguredGraph = {
    route_planning_enabled: true,
    graph_exists: false,
  };
  assert.equal(
    harness.context.routePlanningIntentionallyDisabled(intentionallyDisabled),
    true
  );
  assert.equal(harness.context.routePlanningAvailable(intentionallyDisabled), false);
  assert.equal(
    harness.context.routePlanningIntentionallyDisabled(missingConfiguredGraph),
    false
  );
  assert.equal(harness.context.routePlanningAvailable(missingConfiguredGraph), false);
  assert.equal(
    harness.context.routePlanningAvailable({
      route_planning_enabled: true,
      graph_exists: true,
    }),
    true
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

function makeRouteGeojson() {
  const coordinates = Array.from({ length: 3001 }, (_, index) => [
    -75.2 + index * 0.000001,
    40.04 + index * 0.000001,
  ]);
  return {
    type: "FeatureCollection",
    features: ["scenic", "baseline"].map((routeKind) => ({
      type: "Feature",
      properties: {
        route_kind: routeKind,
        snapped_start: [40.04, -75.2],
        snapped_end: [40.043, -75.197],
      },
      geometry: { type: "LineString", coordinates: coordinates.map((point) => [...point]) },
    })),
  };
}

function makeRouteHarness() {
  const layers = new Map();
  const previousSource = { type: "geojson", data: { old: true } };
  const sources = new Map([["route-source", previousSource]]);
  const operations = [];
  const fitted = [];
  const map = {
    getLayer(id) {
      return layers.get(id) || null;
    },
    removeLayer(id) {
      operations.push(["remove-layer", id]);
      layers.delete(id);
    },
    getSource(id) {
      return sources.get(id) || null;
    },
    removeSource(id) {
      operations.push(["remove-source", id]);
      sources.delete(id);
    },
    addSource(id, source) {
      operations.push(["add-source", id]);
      sources.set(id, source);
    },
    addLayer(layer) {
      operations.push(["add-layer", layer.id]);
      layers.set(layer.id, layer);
    },
  };
  const context = {
    Array,
    Error,
    Math,
    Number,
    Set,
    ROUTE_SOURCE: "route-source",
    ROUTE_BASELINE: "route-baseline",
    ROUTE_SCENIC: "route-scenic",
    ROUTE_ENDPOINT_SOURCE: "route-endpoint-source",
    ROUTE_ENDPOINT_CONNECTORS: "route-endpoint-connectors",
    map,
    removeLayer: (id) => {
      if (map.getLayer(id)) map.removeLayer(id);
    },
    removeSource: (id) => {
      if (map.getSource(id)) map.removeSource(id);
    },
    fitToGeojson: (geojson) => fitted.push(geojson),
  };
  vm.runInNewContext(
    `${routeSource}
globalThis.renderRoute = renderRoute;
globalThis.validateRouteGeojson = validateRouteGeojson;
globalThis.buildRouteEndpointConnectors = buildRouteEndpointConnectors;`,
    context
  );
  return { context, layers, sources, operations, fitted, previousSource };
}

test("route rendering preserves long road geometry and installs separate endpoint connectors", () => {
  const harness = makeRouteHarness();
  const request = {
    start: { lat: 40.03, lon: -75.22 },
    end: { lat: 40.065, lon: -75.19 },
  };
  const geojson = makeRouteGeojson();
  harness.context.renderRoute(geojson, request);

  const installed = harness.sources.get("route-source");
  assert.deepEqual(installed.data, geojson);
  assert.equal(installed.data.features[0].geometry.coordinates.length, 3001);
  assert.deepEqual(
    installed.data.features[0].geometry.coordinates.at(-1),
    [-75.197, 40.043]
  );
  const connectorSource = harness.sources.get("route-endpoint-source");
  assert.deepEqual(JSON.parse(JSON.stringify(connectorSource.data.features)), [
    {
      type: "Feature",
      properties: { connector_kind: "start" },
      geometry: {
        type: "LineString",
        coordinates: [[-75.22, 40.03], [-75.2, 40.04]],
      },
    },
    {
      type: "Feature",
      properties: { connector_kind: "end" },
      geometry: {
        type: "LineString",
        coordinates: [[-75.197, 40.043], [-75.19, 40.065]],
      },
    },
  ]);
  assert.equal(harness.layers.size, 3);
  assert.equal(harness.layers.get("route-endpoint-connectors").paint["line-color"], "#f5c66b");
  assert.deepEqual(
    JSON.parse(JSON.stringify(harness.fitted[0].features.slice(-2))),
    JSON.parse(JSON.stringify(connectorSource.data.features))
  );
});

test("connector builder accepts artifact arrays and omits zero-length connectors", () => {
  const harness = makeRouteHarness();
  const geojson = makeRouteGeojson();
  const connectors = harness.context.buildRouteEndpointConnectors(geojson, {
    start: [40.04, -75.2],
    end: [40.065, -75.19],
  });
  assert.equal(connectors.features.length, 1);
  assert.equal(connectors.features[0].properties.connector_kind, "end");
  assert.deepEqual(
    Array.from(connectors.features[0].geometry.coordinates.at(-1)),
    [-75.19, 40.065]
  );
});

test("malformed request coordinates cannot replace an installed route", () => {
  const harness = makeRouteHarness();
  const geojson = makeRouteGeojson();
  harness.context.renderRoute(geojson, {
    start: { lat: 40.03, lon: -75.22 },
    end: { lat: 40.065, lon: -75.19 },
  });
  const sourceBeforeInvalid = harness.sources.get("route-source");
  const fitCount = harness.fitted.length;
  assert.throws(
    () => harness.context.renderRoute(makeRouteGeojson(), {
      start: { lat: Number.NaN, lon: -75.22 },
      end: { lat: 40.065, lon: -75.19 },
    }),
    /requested start must be a finite coordinate pair/
  );
  assert.strictEqual(harness.sources.get("route-source"), sourceBeforeInvalid);
  assert.equal(harness.fitted.length, fitCount);
});

test("array requests with extra coordinates cannot replace an installed route", () => {
  const harness = makeRouteHarness();
  const geojson = makeRouteGeojson();
  harness.context.renderRoute(geojson, {
    start: { lat: 40.03, lon: -75.22 },
    end: { lat: 40.065, lon: -75.19 },
  });
  const sourcesBeforeInvalid = new Map(harness.sources);
  const layersBeforeInvalid = new Map(harness.layers);
  const operationCount = harness.operations.length;
  const fitCount = harness.fitted.length;

  assert.throws(
    () => harness.context.renderRoute(makeRouteGeojson(), {
      start: [40.03, -75.22, 123],
      end: [40.065, -75.19],
    }),
    /requested start must be a finite coordinate pair/
  );
  assert.deepEqual(new Map(harness.sources), sourcesBeforeInvalid);
  assert.deepEqual(new Map(harness.layers), layersBeforeInvalid);
  assert.equal(harness.operations.length, operationCount);
  assert.equal(harness.fitted.length, fitCount);
});

test("rendering snapped endpoints removes stale connector source and layer", () => {
  const harness = makeRouteHarness();
  const geojson = makeRouteGeojson();
  harness.context.renderRoute(geojson, {
    start: { lat: 40.03, lon: -75.22 },
    end: { lat: 40.065, lon: -75.19 },
  });
  harness.context.renderRoute(geojson, {
    start: [40.04, -75.2],
    end: [40.043, -75.197],
  });
  assert.equal(harness.sources.has("route-endpoint-source"), false);
  assert.equal(harness.layers.has("route-endpoint-connectors"), false);
});
test("pending renders retain their request and clear removes both route sources", () => {
  assert.match(
    viewerSource,
    /pendingRouteRender = \{ requestId, geojson: routeGeojson, request: payload\.request \}/
  );
  assert.match(viewerSource, /renderRoute\(pending\.geojson, pending\.request\)/);
  assert.match(viewerSource, /removeLayer\(ROUTE_ENDPOINT_CONNECTORS\)/);
  assert.match(viewerSource, /removeSource\(ROUTE_ENDPOINT_SOURCE\)/);
});


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
  const outside = context.routeFailurePresentation(
    "route_endpoint_outside_coverage"
  );
  assert.equal(outside.status, "API: outside route coverage");
  assert.equal(outside.title, "Outside route coverage");
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
