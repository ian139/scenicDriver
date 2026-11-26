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
const deadlineSource = viewerSource.slice(
  viewerSource.indexOf("const FETCH_DEADLINE_MS"),
  viewerSource.indexOf("\nfunction resolveRegionSelection")
);
const publicStatusSource = viewerSource.slice(
  viewerSource.indexOf("function setPublicRouteStatus"),
  viewerSource.indexOf("\nfunction escapeHtml")
);

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
    this.disabled = false;
    this.children = [];
    this.dataset = {};
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = {
      add: () => {},
      remove: () => {},
      toggle: () => {},
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
    AbortController,
    setTimeout,
    clearTimeout,
    DEFAULTS: {
      displayRange: "new_england_north",
      sourceRegion: "new_england_north",
      workingRun: "prompt_two_candidate_exp02_fresh_test20_20260810",
      zoom: 1,
    },
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
    `${deadlineSource}
${regionSource}
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

test("selected metadata alone supplies region bounds and fallback camera", async () => {
  const region = {
    region: "other_region",
    display_name: "Other region",
    latest_run_name: "other-run",
    bbox: { min_lon: 145, min_lat: -44, max_lon: 150, max_lat: -40 },
  };
  const harness = makeRegionHarness("?source=other_region", { regions: [region] });
  assert.equal(harness.context.regionCamera(null), null);
  await harness.context.loadSupportedRegions();
  assert.deepEqual(JSON.parse(JSON.stringify(harness.context.activeRegionBounds)), [
    [145, -44], [150, -40],
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(harness.context.regionCamera(region))), {
    center: [147.5, -42], zoom: 1,
  });
});

test("invalid selected bbox rejects metadata before publishing region state", async () => {
  const valid = supportedRegions.regions[0].bbox;
  const invalidBounds = [
    undefined, null, {},
    { ...valid, min_lon: undefined },
    { ...valid, min_lon: null },
    { ...valid, min_lon: "" },
    { ...valid, min_lon: String(valid.min_lon) },
    { ...valid, min_lon: false },
    { ...valid, min_lon: NaN },
    { ...valid, max_lat: Infinity },
    { ...valid, min_lon: -181 },
    { ...valid, max_lon: 181 },
    { ...valid, min_lat: -91 },
    { ...valid, max_lat: 91 },
    { ...valid, min_lon: valid.max_lon },
    { ...valid, min_lon: valid.max_lon + 1 },
    { ...valid, min_lat: valid.max_lat },
    { ...valid, min_lat: valid.max_lat + 1 },
  ];
  for (const bbox of invalidBounds) {
    const selected = { ...supportedRegions.regions[1], bbox };
    const harness = makeRegionHarness("?source=masswhites&run=stale", {
      regions: [supportedRegions.regions[0], selected],
    });
    await assert.rejects(() => harness.context.loadSupportedRegions(), /invalid region bounds/i);
    assert.equal(harness.context.activeRegionBounds, null);
    assert.equal(harness.context.selectedRegionMetadata, null);
    assert.equal(harness.context.CONFIG.workingRun, "stale");
    assert.equal(harness.regionSelect.children.length, 0);
    assert.deepEqual(harness.replacements, []);
  }
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
        bbox: supportedRegions.regions[0].bbox,
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
    `${deadlineSource}\nfunction newSearchSessionToken() { return "session-fixed"; }\n${installSource}\nglobalThis.installAddressSearch = installAddressSearch;`,
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
function makeLifecycleHarness() {
  const harness = makeRouteHarness();
  const { context } = harness;
  const events = new Map();
  let styleLoaded = false;
  context.map.on = (event, callback) => events.set(event, callback);
  context.map.isStyleLoaded = () => styleLoaded;
  context.routeRequestSequence = 0;
  context.pendingRouteRender = null;
  context.activeRouteRequest = null;
  context.latestRoutePayload = null;
  context.displayedRouteRequest = null;
  context.mapReady = false;
  context.CONFIG = { sourceRegion: "test-region", workingRun: "test-run" };
  context.AbortController = AbortController;
  context.URL = URL;
  context.setTimeout = setTimeout;
  context.clearTimeout = clearTimeout;
  context.el = {
    startInput: new FakeElement("input"),
    endInput: new FakeElement("input"),
    detourFactor: new FakeElement("input"),
    avoidHighways: new FakeElement("input"),
    submitRoute: new FakeElement("button"),
    shareRouteBtn: new FakeElement("button"),
    routeStatus: new FakeElement("p"),
  };
  context.el.startInput.value = "40.03,-75.22";
  context.el.endInput.value = "40.065,-75.19";
  context.el.detourFactor.value = "1.8";
  context.el.avoidHighways.checked = true;
  context.selectedRoutePoints = { start: null, end: null };
  context.parsePoint = (text) => {
    const [lat, lon] = text.split(",").map(Number);
    return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
  };
  context.api = (path) => path;
  context.setRouteOutput = (title, message) => { harness.output = { title, message }; };
  context.setApiStatus = () => {};
  context.syncZoomElasticity = () => {};
  context.routePlanningAvailable = () => true;
  context.routeOutputMarkup = () => "Routes shown";
  context.renderRouteComparison = (payload) => { harness.comparison = payload; };
  context.escapeHtml = (text) => String(text).replaceAll("<", "&lt;");
  context.setText = (node, text) => { if (node) node.textContent = text; };
  context.updateRouteControls = () => {
    context.el.submitRoute.disabled = !context.mapReady || !context.routePlanningAvailable();
  };
  context.window = {
    location: new URL("https://test.local/nested/viewer"),
    setTimeout: () => {},
  };
  context.navigator = { clipboard: { writeText: async (text) => { harness.shared = text; } } };
  const pending = [];
  context.fetch = (_url, options) => new Promise((resolve) => pending.push({ resolve, options }));
  const lifecycleStart = viewerSource.indexOf("function drainPendingRouteRender");
  const lifecycleEnd = viewerSource.indexOf("\nfunction signedNumber", lifecycleStart);
  const planStart = viewerSource.indexOf("async function planRoute");
  const planEnd = viewerSource.indexOf("\nfunction initBindings", planStart);
  vm.runInNewContext(
    `${deadlineSource}
${publicStatusSource}
${failureSource}
${viewerSource.slice(lifecycleStart, lifecycleEnd)}
${viewerSource.slice(planStart, planEnd)}
globalThis.installRouteRenderLifecycle = installRouteRenderLifecycle;
globalThis.clearRoute = clearRoute;
globalThis.planRoute = planRoute;
globalThis.copyShareUrl = copyShareUrl;`,
    context
  );
  context.installRouteRenderLifecycle();
  return Object.assign(harness, {
    pending,
    emit: (event) => events.get(event)?.(),
    setStyleLoaded: (value) => { styleLoaded = value; },
  });
}

test("deferred route drains after load, shares its request, and clear cancels pending routes", async () => {
  const harness = makeLifecycleHarness();
  const { context } = harness;
  harness.setStyleLoaded(true);
  harness.emit("load");
  harness.setStyleLoaded(false); // Tiles started loading after the one-time load.
  const planning = context.planRoute({ preventDefault() {} });
  harness.pending[0].resolve(routeResponse({ geojson: makeRouteGeojson() }));
  await planning;
  assert.equal(harness.fitted.length, 0);
  harness.emit("idle");
  assert.equal(harness.fitted.length, 0);
  harness.setStyleLoaded(true);
  harness.emit("idle");
  assert.equal(harness.fitted.length, 1);
  assert.ok(harness.sources.has("route-endpoint-source"));
  harness.emit("idle");
  assert.equal(harness.fitted.length, 1);

  context.el.startInput.value = "1,2";
  context.el.detourFactor.value = "3";
  context.el.avoidHighways.checked = false;
  await context.copyShareUrl();
  const shared = new URL(harness.shared);
  assert.equal(shared.searchParams.get("start"), "40.03,-75.22");
  assert.equal(shared.searchParams.get("max_detour"), "1.8");
  assert.equal(shared.searchParams.get("avoid_highways"), "true");

  harness.setStyleLoaded(false);
  const next = context.planRoute({ preventDefault() {} });
  await context.copyShareUrl();
  assert.equal(harness.shared, shared.toString());
  harness.pending[1].resolve(routeResponse({ geojson: makeRouteGeojson() }));
  await next;
  context.clearRoute();
  harness.setStyleLoaded(true);
  harness.emit("idle");
  assert.equal(harness.fitted.length, 1);
  assert.equal(harness.sources.has("route-source"), false);
  assert.equal(harness.sources.has("route-endpoint-source"), false);
  assert.equal(harness.layers.has("route-endpoint-connectors"), false);
  harness.shared = null;
  await context.copyShareUrl();
  assert.equal(harness.shared, null);
});

test("pre-load responses wait for readiness and superseded requests cannot publish", async () => {
  const harness = makeLifecycleHarness();
  const { context } = harness;
  const old = context.planRoute({ preventDefault() {} });
  context.el.startInput.value = "1,2";
  const current = context.planRoute({ preventDefault() {} });
  harness.pending[1].resolve(routeResponse({ geojson: makeRouteGeojson() }));
  await current;
  harness.pending[0].resolve(routeResponse({ geojson: makeRouteGeojson() }));
  await old;
  harness.setStyleLoaded(true);
  harness.emit("idle");
  assert.equal(harness.fitted.length, 0);
  harness.emit("load");
  assert.equal(harness.fitted.length, 1);
  await context.copyShareUrl();
  assert.equal(new URL(harness.shared).searchParams.get("start"), "1,2");
});

test("structured route failures render the server's reason without emitting markup", async () => {
  // Failure-detail dicts exactly as the FastAPI /v1/route/compare error path
  // emits them (src/scenicdriver/app_api/main.py, mirrored by tests/test_app_api.py) — not
  // invented payloads.  `reason` carries the structured cause's internal
  // reason string; the viewer renders it and falls back to `message` when the
  // server omits it.
  const noRouteDetail = {
    error: "no_route_found",
    message: "No route satisfies the requested controls.",
    hint: "Try different points in region 'new_england_north' or increase max detour.",
    diagnostics: {
      graph_nodes: 4,
      graph_edges: 3,
      start_snap_km: null,
      end_snap_km: null,
      start_all_road_snap_km: null,
      end_all_road_snap_km: null,
      start_node_id: null,
      end_node_id: null,
    },
    reason: "No route found between the given coordinates.",
  };
  const coverageDetail = {
    error: "route_endpoint_outside_coverage",
    message: "The start point is too far from the supported road network.",
    hint: "Choose a point within the selected region's route coverage.",
    reason: "The start point is too far from the supported road network.",
    endpoint: "start",
    snap_distance_km: 1.234,
    max_snap_distance_km: 1.0,
  };
  const supervisorDetail = {
    error: "routing_supervisor_unavailable",
    message: "The routing service is temporarily unavailable.",
  };

  const harness = makeLifecycleHarness();
  const failing = (detail) => ({
    ok: false,
    status: 422,
    statusText: "Unprocessable Entity",
    json: async () => ({ detail }),
  });

  const noRoute = harness.context.planRoute({ preventDefault() {} });
  harness.pending[0].resolve(failing(noRouteDetail));
  await noRoute;
  assert.equal(harness.output.title, "No route found");
  assert.equal(
    harness.output.message,
    "No route satisfies the requested controls. — No route found between the given coordinates. — Try different points in region 'new_england_north' or increase max detour."
  );

  const coverage = harness.context.planRoute({ preventDefault() {} });
  harness.pending[1].resolve(failing(coverageDetail));
  await coverage;
  assert.equal(harness.output.title, "Outside route coverage");
  // The coverage cause only restates the message, so it is not repeated.
  assert.equal(
    harness.output.message,
    "The start point is too far from the supported road network. — Choose a point within the selected region's route coverage."
  );

  const supervisor = harness.context.planRoute({ preventDefault() {} });
  harness.pending[2].resolve(failing(supervisorDetail));
  await supervisor;
  assert.equal(harness.output.title, "Route failed");
  // No reason: the message carries the text instead.
  assert.equal(
    harness.output.message,
    "The routing service is temporarily unavailable."
  );

  // Same real shape with hostile field content, to prove every rendered field
  // stays escaped rather than becoming markup.
  const hostile = harness.context.planRoute({ preventDefault() {} });
  harness.pending[3].resolve(
    failing({
      ...noRouteDetail,
      hint: "Try different points in region '<img>' or increase max detour.",
    })
  );
  await hostile;
  assert.equal(harness.output.title, "No route found");
  assert.equal(
    harness.output.message,
    "No route satisfies the requested controls. — No route found between the given coordinates. — Try different points in region '&lt;img>' or increase max detour."
  );
});

test("local assets resolve at the root even for a nested app-shell URL", () => {
  const html = fs.readFileSync(new URL("../apps/new_england_north/index.html", import.meta.url), "utf8");
  const localAssets = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css)\?[^"]+)"/g)]
    .map((match) => match[1]);
  for (const base of ["https://test.local/route/deep", "https://test.local/route/deep/"]) {
    assert.deepEqual(localAssets.map((asset) => new URL(asset, base).pathname), ["/styles.css", "/viewer.js"]);
  }
});


const response = (payload) => ({ ok: true, json: async () => payload });
const routeResponse = (payload) => response({ routes: { scenic: {} }, diagnostics: {}, ...payload });
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

test("refocusing address search cancels delayed blur closure", async () => {
  const { input, suggestions, pending } = makeHarness();
  input.value = "Boston";
  input.dispatch("input");
  await tick();
  pending[0].resolve(response({ suggestions: [{ mapbox_id: "bos", name: "Boston" }] }));
  await tick(0);
  input.dispatch("blur");
  input.dispatch("focus");
  await tick(140);
  assert.equal(input.attributes.get("aria-expanded"), "true");
  assert.equal(suggestions.children[0].children[0].textContent, "Boston");
  input.dispatch("blur");
  await tick(140);
  assert.equal(input.attributes.get("aria-expanded"), "false");
});

test("heatmap metadata and keyboard navigation work without a summary or pointer input", async () => {
  const canvas = new FakeElement("canvas");
  const inspector = new FakeElement();
  const inspectorScore = new FakeElement();
  const inspectorCoords = new FakeElement();
  const events = new Map();
  let center = { lng: 1, lat: 2 };
  const context = {
    el: { inspector, inspectorScore, inspectorCoords },
    document: { activeElement: null },
    map: {
      getCanvas: () => canvas,
      getCenter: () => center,
      on: (event, callback) => events.set(event, callback),
      unproject: ([x, y]) => ({ lng: x, lat: y }),
    },
    lngLatToTile: (lng) => ({ x: lng, y: 2 }),
    setText: (node, text) => { node.textContent = text; },
  };
  context.AbortController = AbortController;
  context.setTimeout = setTimeout;
  context.clearTimeout = clearTimeout;
  context.CONFIG = { sourceRegion: "test", workingRun: "test-run" };
  context.api = (path) => `https://test.local${path}`;
  context.URL = class extends URL {
    static createObjectURL() { return "blob:test"; }
  };
  context.validHeatmapMetadata = () => true;
  context.setApiStatus = () => {};
  context.latestHeatmap = null;
  const start = viewerSource.indexOf("function installHeatmapInspector");
  const end = viewerSource.indexOf("\nasync function main", start);
  const heatmapStart = viewerSource.indexOf("async function loadHeatmap");
  const heatmapEnd = viewerSource.indexOf("\nasync function checkApiHealth", heatmapStart);
  vm.runInNewContext(
    `${deadlineSource}
${viewerSource.slice(heatmapStart, heatmapEnd)}
${viewerSource.slice(start, end)}
globalThis.installHeatmapInspector = installHeatmapInspector;
globalThis.loadHeatmap = loadHeatmap;`, context);
  const values = new DataView(new ArrayBuffer(8));
  values.setFloat32(0, 0.75, true);
  values.setFloat32(4, 0.25, true);
  context.fetch = async (url) => {
    const path = new URL(url).pathname;
    if (path === "/v1/heatmap") {
      return response({ total_tiles: 2, summary: null, tile_zoom: 14, bounds: {} });
    }
    if (path === "/v1/heatmap-image") {
      return { ok: true, blob: async () => ({}) };
    }
    return {
      ok: true,
      headers: new Map([
        ["X-Scenic-Tile-Zoom", "14"], ["X-Scenic-Min-X", "1"], ["X-Scenic-Min-Y", "2"],
        ["X-Scenic-Grid-Width", "2"], ["X-Scenic-Grid-Height", "1"],
      ]),
      arrayBuffer: async () => values.buffer,
    };
  };
  const heatmap = await context.loadHeatmap();
  await heatmap.gridReady;
  context.installHeatmapInspector(heatmap);
  assert.equal(inspectorScore.textContent, "2 tiles");
  context.document.activeElement = canvas;
  canvas.dispatch("focus");
  assert.equal(inspectorScore.textContent, "0.75");
  assert.equal(inspector.attributes.get("aria-live"), "polite");
  center = { lng: 2, lat: 2 };
  events.get("moveend")();
  assert.equal(inspectorScore.textContent, "0.25");
  canvas.dispatch("pointerleave");
  assert.equal(inspectorScore.textContent, "0.25");
  canvas.dispatch("blur");
  context.document.activeElement = null;
  center = { lng: 3, lat: 2 };
  events.get("moveend")();
  assert.equal(inspectorScore.textContent, "0.25");
  canvas.dispatch("pointermove", { offsetX: 3, offsetY: 2 });
  assert.equal(inspectorScore.textContent, "--");
  assert.equal(inspectorCoords.textContent, "No learned score for this tile");
});

test("replacement result publishes geometry, metrics and new share identity together", async () => {
  const harness = makeLifecycleHarness();
  const { context } = harness;
  harness.setStyleLoaded(true);
  harness.emit("load");
  const firstPayload = { geojson: makeRouteGeojson(), routes: { scenic: { total_distance_km: 10 } }, diagnostics: {} };
  const first = context.planRoute({ preventDefault() {} });
  harness.pending[0].resolve(routeResponse(firstPayload));
  await first;
  assert.equal(context.el.shareRouteBtn.disabled, false);
  assert.equal(harness.comparison.routes.scenic.total_distance_km, 10);
  await context.copyShareUrl();
  const firstShare = harness.shared;

  harness.setStyleLoaded(false);
  context.el.startInput.value = "41,-74";
  context.el.endInput.value = "42,-73";
  const second = context.planRoute({ preventDefault() {} });
  assert.equal(context.el.shareRouteBtn.disabled, true);
  assert.match(context.el.shareRouteBtn.textContent, /pending/i);
  const secondPayload = { geojson: makeRouteGeojson(), routes: { scenic: { total_distance_km: 20 } }, diagnostics: {} };
  harness.pending[1].resolve(routeResponse(secondPayload));
  await second;
  assert.equal(context.el.shareRouteBtn.disabled, true);
  assert.equal(harness.comparison.routes.scenic.total_distance_km, 10);
  assert.equal(harness.fitted.length, 1);
  assert.match(context.el.routeStatus.textContent, /Waiting for the map/);
  await context.copyShareUrl();
  assert.equal(harness.shared, firstShare);
  assert.match(context.el.routeStatus.textContent, /Share unavailable/);

  harness.setStyleLoaded(true);
  harness.emit("idle");
  assert.equal(harness.fitted.length, 2);
  assert.equal(harness.comparison.routes.scenic.total_distance_km, 20);
  assert.equal(context.el.shareRouteBtn.disabled, false);
  assert.match(context.el.routeStatus.textContent, /Route ready/);
  await context.copyShareUrl();
  const shared = new URL(harness.shared);
  assert.equal(shared.searchParams.get("start"), "41,-74");
  assert.equal(shared.searchParams.get("end"), "42,-73");
});

test("public form status exposes validation and request errors with invalid endpoints", async () => {
  const harness = makeLifecycleHarness();
  const { context } = harness;
  context.el.startInput.value = "not coordinates";
  await context.planRoute({ preventDefault() {} });
  assert.equal(harness.pending.length, 0);
  assert.match(context.el.routeStatus.textContent, /valid coordinates/);
  assert.equal(context.el.startInput.attributes.get("aria-invalid"), "true");
  assert.equal(context.el.endInput.attributes.get("aria-invalid"), "false");
  context.el.startInput.value = "40,-75";
  const planned = context.planRoute({ preventDefault() {} });
  assert.match(context.el.routeStatus.textContent, /Planning/);
  harness.pending[0].resolve({
    ok: false,
    json: async () => ({ detail: {
      error: "route_endpoint_outside_coverage", endpoint: "end",
      message: "Choose an endpoint inside coverage.", hint: "Try a nearer road.",
    } }),
  });
  await planned;
  assert.match(context.el.routeStatus.textContent, /Outside route coverage: Choose an endpoint inside coverage.*Try a nearer road/);
  assert.equal(context.el.endInput.attributes.get("aria-invalid"), "true");
  assert.equal(context.el.startInput.attributes.get("aria-invalid"), "false");
  const rejected = context.planRoute({ preventDefault() {} });
  harness.pending[1].resolve({ ok: false, json: async () => ({ detail: "Server rejected the request." }) });
  await rejected;
  assert.match(context.el.routeStatus.textContent, /Server rejected the request/);
  const html = fs.readFileSync(new URL("../apps/new_england_north/index.html", import.meta.url), "utf8");
  assert.ok(html.indexOf('id="routeStatus"') < html.indexOf('<details class="dev-panel">'));
  for (const id of ["startInput", "endInput", "submitRoute"]) {
    assert.match(html, new RegExp(`id="${id}"[^>]*aria-describedby="routeStatus"`));
  }
});

function makeStartupHarness(search = "") {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new FakeElement());
    return elements.get(id);
  };
  element("startInput").value = "44.475884,-73.214003";
  element("endInput").value = "44.801616,-68.771305";
  Object.assign(element("detourFactor"), { value: "1.8", min: "1", max: "3" });
  element("avoidHighways").checked = true;
  const dialog = element("routeResultsDialog");
  dialog.showModal = () => { dialog.open = true; };
  dialog.close = () => { dialog.open = false; };
  const timers = new Map();
  let timerId = 0;
  const requests = [];
  let map;
  class FakeMap {
    constructor(options) {
      this.options = options;
      this.events = new Map();
      this.sources = new Map();
      this.layers = new Map();
      this.canvas = new FakeElement("canvas");
      this.styleLoaded = false;
      this.boundsCalls = [];
      this.minimumZooms = [];
      map = this;
    }
    on(event, callback) {
      const callbacks = this.events.get(event) || [];
      callbacks.push(callback);
      this.events.set(event, callbacks);
    }
    emit(event) {
      if (event === "load") this.styleLoaded = true;
      for (const callback of this.events.get(event) || []) callback({});
    }
    isStyleLoaded() { return this.styleLoaded; }
    addControl() {}
    cameraForBounds(bounds) {
      this.boundsCalls.push(bounds);
      return { zoom: 6 };
    }
    setMinZoom(zoom) { this.minimumZooms.push(zoom); }
    getCanvas() { return this.canvas; }
    getSource(id) { return this.sources.get(id); }
    getLayer(id) { return this.layers.get(id); }
    addSource(id, source) {
      this.sources.set(id, { ...source, setData(data) { this.data = data; } });
    }
    addLayer(layer) { this.layers.set(layer.id, layer); }
    removeSource(id) { this.sources.delete(id); }
    removeLayer(id) { this.layers.delete(id); }
    fitBounds() {}
    jumpTo(camera) { this.camera = camera; }
  }
  class FakeBounds {
    extend() { return this; }
  }
  const location = new URL(`https://test.local/viewer${search}`);
  const context = {
    URL: class extends URL { static createObjectURL() { return "blob:heatmap"; } },
    URLSearchParams,
    AbortController,
    crypto: { randomUUID: () => "search-session" },
    maplibregl: { Map: FakeMap, NavigationControl: class {}, LngLatBounds: FakeBounds },
    document: {
      getElementById: element,
      createElement: (tag) => new FakeElement(tag),
      activeElement: null,
    },
    window: {
      location, innerWidth: 1200, innerHeight: 900,
      history: { replaceState: (_state, _title, url) => { context.window.location = new URL(url); } },
      setTimeout: () => {},
    },
    setTimeout: (callback, delay) => {
      const id = ++timerId;
      timers.set(id, { callback, delay });
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
    fetch: (url, options) => new Promise((resolve, reject) => {
      const request = { path: new URL(url).pathname, url: String(url), options, resolve, reject };
      requests.push(request);
      options.signal.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      }, { once: true });
    }),
  };
  vm.runInNewContext(
    `${viewerSource.slice(0, viewerSource.indexOf("\nmain().catch"))}
globalThis.main = main;`,
    context
  );
  const startup = context.main();
  return {
    context, element, requests, startup,
    get map() { return map; },
    expireDeadlines() {
      for (const [id, timer] of [...timers]) {
        if (timer.delay === 15000) {
          timers.delete(id);
          timer.callback();
        }
      }
    },
    request(path) { return requests.find((request) => request.path === path); },
  };
}

function resolveStartupResources(harness, { gridFails = false, regionsFail = false, regions = supportedRegions } = {}) {
  harness.request("/api/v1/regions").resolve(regionsFail
    ? { ok: false, status: 503, json: async () => ({}) }
    : response(regions));
  harness.request("/api/v1/healthz").resolve(response({ status: "ok" }));
  harness.request("/api/v1/training-results").resolve(response({}));
  harness.request("/api/v1/heatmap").resolve(response({
    tile_zoom: 14, total_tiles: 1,
    bounds: { min_lon: -73.52, min_lat: 42.48, max_lon: -66.79, max_lat: 47.5 },
  }));
  harness.request("/api/v1/heatmap-image").resolve({ ok: true, blob: async () => ({}) });
  harness.request("/api/v1/heatmap-scores.bin").resolve(gridFails
    ? { ok: false, status: 503, statusText: "Unavailable" }
    : {
      ok: true,
      headers: new Map([
        ["X-Scenic-Tile-Zoom", "14"], ["X-Scenic-Min-X", "1"], ["X-Scenic-Min-Y", "2"],
        ["X-Scenic-Grid-Width", "1"], ["X-Scenic-Grid-Height", "1"],
      ]),
      arrayBuffer: async () => new ArrayBuffer(4),
    });
}

function validatedArtifact() {
  return {
    geojson: makeRouteGeojson(),
    request: { start: [40.04, -75.2], end: [40.043, -75.197], max_detour_factor: 1.8, avoid_highways: true },
    routes: { scenic: {}, baseline: {} }, score_mapping: {},
  };
}

test("shared URL plans restored controls after map readiness without installing the default artifact", async () => {
  const harness = makeStartupHarness("?start=44.5,-72.5&end=45,-70&max_detour=2.5&avoid_highways=false");
  resolveStartupResources(harness);
  await tick(0);
  assert.equal(harness.request("/api/v1/route/compare"), undefined);
  assert.equal(harness.map.sources.has("route-source"), false);
  assert.doesNotMatch(harness.element("routeTitle").textContent || "", /validated|Burlington/);
  harness.map.emit("load");
  await tick(0);
  const route = harness.request("/api/v1/route/compare");
  const body = JSON.parse(route.options.body);
  assert.deepEqual(body.start, { lat: 44.5, lon: -72.5 });
  assert.deepEqual(body.end, { lat: 45, lon: -70 });
  assert.equal(body.max_detour_factor, 2.5);
  assert.equal(body.avoid_highways, false);
  assert.equal(harness.request("/api/v1/validated-route"), undefined);
  route.resolve(routeResponse({ geojson: makeRouteGeojson() }));
  await harness.startup;
  assert.equal(harness.requests.filter((request) => request.path === "/api/v1/route/compare").length, 1);
  assert.equal(harness.element("routeTitle").textContent, "Route computed");
  assert.equal(harness.element("shareRouteBtn").disabled, false);
});

test("grid failure preserves image source and explicitly disables only the score inspector", async () => {
  const harness = makeStartupHarness();
  resolveStartupResources(harness, { gridFails: true });
  harness.map.emit("load");
  await tick(0);
  harness.request("/api/v1/validated-route").resolve(response(validatedArtifact()));
  await harness.startup;
  assert.deepEqual(JSON.parse(JSON.stringify(harness.map.boundsCalls)), [
    [[-73.52, 42.48], [-66.79, 47.5]],
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(harness.map.camera)), {
    center: [-70.15, 44.99], zoom: 6.2,
  });
  assert.equal(harness.map.sources.get("scenic-heatmap-source").url, "blob:heatmap");
  assert.equal(harness.map.layers.has("scenic-heatmap-fill"), true);
  assert.equal(harness.element("inspector").attributes.get("aria-disabled"), "true");
  assert.match(harness.element("inspectorCoords").textContent, /inspector unavailable.*503/i);
});

test("startup constructs basemap and initiates independent resources concurrently with deadlines", async () => {
  const harness = makeStartupHarness();
  assert.ok(harness.map);
  assert.equal(harness.map.options.style.sources.voyager.type, "raster");
  assert.deepEqual(Array.from(harness.map.options.center), [0, 0]);
  assert.equal(harness.map.options.zoom, 1);
  harness.map.emit("resize");
  harness.map.emit("zoomend");
  assert.deepEqual(harness.map.boundsCalls, []);
  assert.equal(harness.map.camera, undefined);
  assert.deepEqual(harness.requests.map((request) => request.path).sort(), [
    "/api/v1/healthz", "/api/v1/heatmap", "/api/v1/heatmap-image",
    "/api/v1/heatmap-scores.bin", "/api/v1/regions", "/api/v1/training-results",
  ]);
  assert.ok(harness.requests.every((request) => !request.options.signal.aborted));
  harness.expireDeadlines();
  await harness.startup;
  assert.ok(harness.requests.every((request) => request.options.signal.aborted));
  assert.equal(harness.element("submitRoute").disabled, true);
  assert.match(harness.element("routeStatus").textContent, /metadata unavailable.*timed out/i);
});

test("unknown region metadata keeps every planning control disabled and forbids planning", async () => {
  const harness = makeStartupHarness();
  assert.equal(harness.element("startInput").disabled, true);
  resolveStartupResources(harness, { regionsFail: true });
  harness.map.emit("load");
  await harness.startup;
  harness.map.emit("resize");
  harness.map.emit("zoomend");
  assert.deepEqual(harness.map.boundsCalls, []);
  assert.equal(harness.map.camera, undefined);
  assert.equal(harness.map.minimumZooms.at(-1), 0);
  for (const id of ["submitRoute", "startInput", "endInput", "detourFactor", "avoidHighways"]) {
    assert.equal(harness.element(id).disabled, true, id);
  }
  assert.equal(harness.request("/api/v1/validated-route"), undefined);
  harness.element("routeForm").dispatch("submit");
  await tick(0);
  assert.equal(harness.request("/api/v1/route/compare"), undefined);
  assert.match(harness.element("routeStatus").textContent, /authoritative region capabilities/);
});

test("missing metadata bounds fail closed without aligning camera or installing region layers", async () => {
  const harness = makeStartupHarness();
  const region = { ...supportedRegions.regions[0] };
  delete region.bbox;
  resolveStartupResources(harness, { regions: { regions: [region] } });
  harness.map.emit("load");
  await harness.startup;
  harness.map.emit("resize");
  harness.map.emit("zoomend");
  assert.deepEqual(harness.map.boundsCalls, []);
  assert.equal(harness.map.camera, undefined);
  assert.equal(harness.map.minimumZooms.at(-1), 0);
  assert.equal(harness.map.sources.size, 0);
  assert.equal(harness.request("/api/v1/validated-route"), undefined);
  assert.equal(harness.request("/api/v1/route/compare"), undefined);
  assert.match(harness.element("regionStatus").textContent, /Region metadata unavailable:.*invalid region bounds/);
  for (const id of ["submitRoute", "startInput", "endInput", "detourFactor", "avoidHighways"]) {
    assert.equal(harness.element(id).disabled, true, id);
  }
});

test("route planning deadline aborts the request and reports a public actionable timeout", async () => {
  const harness = makeStartupHarness("?start=44.5,-72.5&end=45,-70");
  resolveStartupResources(harness);
  harness.map.emit("load");
  await tick(0);
  const request = harness.request("/api/v1/route/compare");
  assert.equal(request.options.signal.aborted, false);
  harness.expireDeadlines();
  await harness.startup;
  assert.equal(request.options.signal.aborted, true);
  assert.match(harness.element("routeStatus").textContent, /timed out.*try again/i);
  assert.equal(harness.element("shareRouteBtn").disabled, true);
  assert.equal(harness.element("submitRoute").disabled, false);
});

test("route publication reuses installed geometry and connector sources", () => {
  const harness = makeRouteHarness();
  const request = { start: { lat: 40.03, lon: -75.22 }, end: { lat: 40.065, lon: -75.19 } };
  harness.context.renderRoute(makeRouteGeojson(), request);
  const routeSource = harness.sources.get("route-source");
  const connectorSource = harness.sources.get("route-endpoint-source");
  let routeUpdates = 0;
  let connectorUpdates = 0;
  routeSource.setData = (data) => { routeSource.data = data; routeUpdates += 1; };
  connectorSource.setData = (data) => { connectorSource.data = data; connectorUpdates += 1; };
  const operationCount = harness.operations.length;
  harness.context.renderRoute(makeRouteGeojson(), request);
  assert.equal(harness.sources.get("route-source"), routeSource);
  assert.equal(harness.sources.get("route-endpoint-source"), connectorSource);
  assert.equal(routeUpdates, 1);
  assert.equal(connectorUpdates, 1);
  assert.equal(harness.operations.length, operationCount);
});

test("image publishes before a slow grid and remains visible when the grid deadline expires", async () => {
  const harness = makeStartupHarness();
  const gridRequest = harness.request("/api/v1/heatmap-scores.bin");
  const gridResolve = gridRequest.resolve;
  gridRequest.resolve = () => {};
  resolveStartupResources(harness);
  gridRequest.resolve = gridResolve;
  harness.map.emit("load");
  await tick(0);
  harness.request("/api/v1/validated-route").resolve(response(validatedArtifact()));
  await tick(0);
  assert.equal(harness.map.sources.get("scenic-heatmap-source").url, "blob:heatmap");
  assert.match(harness.element("inspectorCoords").textContent, /loading/i);
  harness.expireDeadlines();
  await harness.startup;
  assert.equal(gridRequest.options.signal.aborted, true);
  assert.equal(harness.map.sources.get("scenic-heatmap-source").url, "blob:heatmap");
  assert.match(harness.element("inspectorCoords").textContent, /unavailable.*timed out/i);
});

test("late clipboard completion cannot replace the pending share label", async () => {
  const harness = makeLifecycleHarness();
  harness.setStyleLoaded(true);
  harness.emit("load");
  const first = harness.context.planRoute({ preventDefault() {} });
  harness.pending[0].resolve(routeResponse({ geojson: makeRouteGeojson() }));
  await first;
  let finishCopy;
  harness.context.navigator.clipboard.writeText = () => new Promise((resolve) => { finishCopy = resolve; });
  const copy = harness.context.copyShareUrl();
  harness.setStyleLoaded(false);
  const second = harness.context.planRoute({ preventDefault() {} });
  finishCopy();
  await copy;
  assert.equal(harness.context.el.shareRouteBtn.disabled, true);
  assert.match(harness.context.el.shareRouteBtn.textContent, /pending/i);
  harness.pending[1].resolve(routeResponse({ geojson: makeRouteGeojson() }));
  await second;
  harness.context.clearRoute();
});
