const DEFAULTS = Object.freeze({
  displayRange: "new_england_north",
  sourceRegion: "new_england_north",
  workingRun: "prompt_two_candidate_exp02_fresh_test20_20260810",
  apiBase: `${window.location.origin}/api`,
  zoom: 6.2,
  scenicWeight: 0.8,
});

const REGION_BOUNDS = Object.freeze([
  Object.freeze([-73.5205078125, 42.488301979602255]),
  Object.freeze([-66.796875, 47.50235895196859]),
]);
// Navigation stays unconstrained to the learned rectangle. This soft fit is
// only a visual reference: a small lower buffer lets users pull back farther
// before the map gently settles to the region overview.
const MAP_NAVIGATION = Object.freeze({
  softFitPaddingPx: 48, // Keep the soft overview just inside the viewport.
  zoomOverscroll: 0.85, // Less than one zoom level of deliberate elastic pull.
  bounceDurationMs: 360, // Short settle animation, close to a page-edge bounce.
});

const ZOOM_BOUNCE_EVENT = Object.freeze({ scenicZoomBounce: true });
const PROGRAMMATIC_CAMERA_EVENT = Object.freeze({
  scenicProgrammaticCamera: true,
});


let softMinimumZoom = null;
let zoomBounceActive = false;


const params = new URLSearchParams(window.location.search);
let CONFIG = {
  displayRange: params.get("source") || params.get("region") || params.get("display") || DEFAULTS.displayRange,
  sourceRegion: params.get("source") || params.get("region") || DEFAULTS.sourceRegion,
  workingRun: params.get("run") || DEFAULTS.workingRun,
};

let apiBase =
  params.get("api") ||
  (["localhost", "127.0.0.1"].includes(window.location.hostname) &&
  window.location.port === "3000"
    ? "http://localhost:8080"
    : DEFAULTS.apiBase);

const HEATMAP_SOURCE = "scenic-heatmap-source";
const HEATMAP_FILL = "scenic-heatmap-fill";
const ROUTE_SOURCE = "route-source";
const ROUTE_BASELINE = "route-baseline";
const ROUTE_SCENIC = "route-scenic";
const ROUTE_ENDPOINT_SOURCE = "route-endpoint-source";
const ROUTE_ENDPOINT_CONNECTORS = "route-endpoint-connectors";

const el = {
  apiStatus: document.getElementById("apiStatus"),
  regionSelect: document.getElementById("regionSelect"),
  runSelect: document.getElementById("runSelect"),
  regionStatus: document.getElementById("regionStatus"),
  routeForm: document.getElementById("routeForm"),
  startInput: document.getElementById("startInput"),
  endInput: document.getElementById("endInput"),
  startSuggestions: document.getElementById("startSuggestions"),
  endSuggestions: document.getElementById("endSuggestions"),
  detourFactor: document.getElementById("detourFactor"),
  detourOut: document.getElementById("detourOut"),
  avoidHighways: document.getElementById("avoidHighways"),
  submitRoute: document.getElementById("submitRoute"),
  routeTitle: document.getElementById("routeTitle"),
  routeOutput: document.getElementById("routeOutput"),
  trainingResults: document.getElementById("trainingResults"),
  clearRoute: document.getElementById("clearRoute"),
  clearStartBtn: document.getElementById("clearStartBtn"),
  clearEndBtn: document.getElementById("clearEndBtn"),
  inspectorScore: document.getElementById("inspectorScore"),
  inspectorCoords: document.getElementById("inspectorCoords"),
  routeResultsDialog: document.getElementById("routeResultsDialog"),
  verboseRouteResults: document.getElementById("verboseRouteResults"),
  routeDiagnostics: document.getElementById("routeDiagnostics"),
  scenicDistance: document.getElementById("scenicDistance"),
  scenicDuration: document.getElementById("scenicDuration"),
  scenicScore: document.getElementById("scenicScore"),
  scenicRawScore: document.getElementById("scenicRawScore"),
  scenicNormalizedScore: document.getElementById("scenicNormalizedScore"),
  scenicDistanceDelta: document.getElementById("scenicDistanceDelta"),
  scenicDurationDelta: document.getElementById("scenicDurationDelta"),
  scenicScoreDelta: document.getElementById("scenicScoreDelta"),
  baselineDistance: document.getElementById("baselineDistance"),
  baselineDuration: document.getElementById("baselineDuration"),
  baselineScore: document.getElementById("baselineScore"),
  baselineRawScore: document.getElementById("baselineRawScore"),
  baselineNormalizedScore: document.getElementById("baselineNormalizedScore"),
  scenicScoreUpliftAbsolute: document.getElementById("scenicScoreUpliftAbsolute"),
  scenicScoreUpliftRelative: document.getElementById("scenicScoreUpliftRelative"),
  requestedScenicWeight: document.getElementById("requestedScenicWeight"),
  appliedScenicWeight: document.getElementById("appliedScenicWeight"),
  requestedDetourCap: document.getElementById("requestedDetourCap"),
  appliedDetourCap: document.getElementById("appliedDetourCap"),
  actualDurationRatio: document.getElementById("actualDurationRatio"),
  optimizationMode: document.getElementById("optimizationMode"),
  searchMode: document.getElementById("searchMode"),
  searchTimeLimit: document.getElementById("searchTimeLimit"),
  searchLabelsGenerated: document.getElementById("searchLabelsGenerated"),
  searchLabelsExpanded: document.getElementById("searchLabelsExpanded"),
  searchLabelsPruned: document.getElementById("searchLabelsPruned"),
  searchMaxFrontierSize: document.getElementById("searchMaxFrontierSize"),
  searchRemainingFrontierSize: document.getElementById("searchRemainingFrontierSize"),
  searchElapsed: document.getElementById("searchElapsed"),
  searchDeadlineReached: document.getElementById("searchDeadlineReached"),

  optimizationStatus: document.getElementById("optimizationStatus"),
  optimalityGap: document.getElementById("optimalityGap"),
  certifiedUpperBound: document.getElementById("certifiedUpperBound"),
  sameRoute: document.getElementById("sameRoute"),
  noBetterRouteReason: document.getElementById("noBetterRouteReason"),
  shareRouteBtn: document.getElementById("shareRouteBtn"),
  closeRouteDialogBtn: document.getElementById("closeRouteDialogBtn"),
};
let routeRequestSequence = 0;
let activeRouteRequest = null;
let pendingRouteRender = null;
let verboseRouteResults = false;
let latestRoutePayload = null;
let mapReady = false;

let map;
let latestHeatmap = null;
const selectedRoutePoints = { start: null, end: null };
let selectedRegionMetadata = null;
let activeRegionBounds = REGION_BOUNDS;

function api(path) {
  return `${apiBase.replace(/\/+$/, "")}${path}`;
}

function resolveRegionSelection(payload, requestedRegion, requestedRun) {
  const regions = Array.isArray(payload?.regions)
    ? payload.regions.filter(
        (region) => region && typeof region.region === "string" && region.region.trim()
      )
    : [];
  if (!regions.length) throw new Error("API returned no supported regions");
  const requested = String(requestedRegion || "").toLowerCase();
  const region =
    regions.find((item) => item.region.toLowerCase() === requested) ||
    regions.find((item) => item.is_default) ||
    regions.find((item) => item.region.toLowerCase() === DEFAULTS.sourceRegion) ||
    regions[0];
  const latestRun =
    typeof region.latest_run_name === "string" && region.latest_run_name.trim()
      ? region.latest_run_name
      : null;
  return {
    regions,
    region,
    run: requestedRun === latestRun ? requestedRun : latestRun,
  };
}

function boundsFromRegion(region) {
  const bbox = region?.bbox;
  if (
    !bbox ||
    !["min_lon", "min_lat", "max_lon", "max_lat"].every((key) =>
      Number.isFinite(Number(bbox[key]))
    )
  ) {
    return REGION_BOUNDS;
  }
  return [
    [Number(bbox.min_lon), Number(bbox.min_lat)],
    [Number(bbox.max_lon), Number(bbox.max_lat)],
  ];
}

function regionCamera(region) {
  const center = region?.map?.center;
  const lon = Number(center?.lon);
  const lat = Number(center?.lat);
  const zoom = Number(region?.map?.zoom);
  return {
    center:
      Number.isFinite(lon) && Number.isFinite(lat)
        ? [lon, lat]
        : [
            (activeRegionBounds[0][0] + activeRegionBounds[1][0]) / 2,
            (activeRegionBounds[0][1] + activeRegionBounds[1][1]) / 2,
          ],
    zoom: Number.isFinite(zoom) ? zoom : DEFAULTS.zoom,
  };
}

function selectionUrl(region, run) {
  const url = new URL(window.location.href);
  url.searchParams.set("source", region);
  if (run) url.searchParams.set("run", run);
  else url.searchParams.delete("run");
  url.searchParams.delete("region");
  url.searchParams.delete("display");
  return url;
}

function populateRegionControls(selection) {
  if (!el.regionSelect || !el.runSelect) return;
  el.regionSelect.replaceChildren(
    ...selection.regions.map((region) => {
      const option = document.createElement("option");
      option.value = region.region;
      option.textContent = region.display_name || region.region;
      option.dataset.run = region.latest_run_name || "";
      option.selected = region === selection.region;
      return option;
    })
  );
  const runOption = document.createElement("option");
  runOption.value = selection.run || "";
  runOption.textContent = selection.run || "No run available";
  el.runSelect.replaceChildren(runOption);
  el.regionSelect.disabled = false;
  el.runSelect.disabled = !selection.run;
}

function validHeatmapMetadata(payload) {
  const bounds = payload?.bounds;
  return (
    Number.isInteger(payload?.tile_zoom) &&
    payload.tile_zoom > 0 &&
    bounds &&
    ["min_lon", "min_lat", "max_lon", "max_lat"].every(
      (key) => Number.isFinite(Number(bounds[key]))
    )
  );
}

async function loadSupportedRegions() {
  const response = await fetch(api("/v1/regions"));
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const selection = resolveRegionSelection(
    await response.json(),
    CONFIG.sourceRegion,
    params.get("run")
  );
  selectedRegionMetadata = selection.region;
  activeRegionBounds = boundsFromRegion(selection.region);
  CONFIG = {
    ...CONFIG,
    displayRange: selection.region.display_name || selection.region.region,
    sourceRegion: selection.region.region,
    workingRun: selection.run,
  };
  populateRegionControls(selection);
  setText(
    el.regionStatus,
    selection.run
      ? `${CONFIG.displayRange} · ${selection.run}`
      : `${CONFIG.displayRange} has no configured scenic run`
  );
  if (params.has("region") || params.has("source") || params.has("run")) {
    const alignedUrl = selectionUrl(CONFIG.sourceRegion, CONFIG.workingRun);
    if (alignedUrl.href !== window.location.href) {
      window.history.replaceState(null, "", alignedUrl);
    }
  }
  return selection;
}


function routePlanningIntentionallyDisabled(region = selectedRegionMetadata) {
  return region?.route_planning_enabled === false;
}


function routePlanningAvailable(region = selectedRegionMetadata) {
  return !routePlanningIntentionallyDisabled(region) && region?.graph_exists !== false;
}

function setText(node, text) {
  if (node) node.textContent = text;
}

function setApiStatus(text, kind = "") {
  if (!el.apiStatus) return;
  el.apiStatus.textContent = text;
  el.apiStatus.classList.toggle("ok", kind === "ok");
  el.apiStatus.classList.toggle("warn", kind === "warn");
}

function setRouteOutput(title, html) {
  setText(el.routeTitle, title);
  if (el.routeOutput) el.routeOutput.innerHTML = html;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || value === "") return "--";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "--";
}

function renderTrainingResults(payload) {
  const metrics = payload?.metrics;
  const valid =
    typeof payload?.run_name === "string" &&
    payload.run_name.length > 0 &&
    typeof payload?.checkpoint === "string" &&
    payload.checkpoint.length > 0 &&
    typeof payload?.updated_at === "string" &&
    payload.updated_at.length > 0 &&
    metrics &&
    ["corr", "mae", "rmse"].every(
      (key) => typeof metrics[key] === "number" && Number.isFinite(metrics[key])
    ) &&
    Number.isInteger(metrics.samples);
  if (!valid) throw new Error("invalid training result");

  const title = document.createElement("strong");
  title.textContent = payload.run_name;
  const timestamp = document.createElement("span");
  timestamp.textContent = payload.updated_at;
  const metricGrid = document.createElement("div");
  metricGrid.className = "training-metrics";
  const metricValues = [
    ["Correlation", metrics.corr.toFixed(3)],
    ["MAE", metrics.mae.toFixed(3)],
    ["RMSE", metrics.rmse.toFixed(3)],
    ["Samples", String(metrics.samples)],
  ];
  for (const [labelText, valueText] of metricValues) {
    const metric = document.createElement("div");
    const label = document.createElement("span");
    label.textContent = labelText;
    const value = document.createElement("strong");
    value.textContent = valueText;
    metric.append(label, value);
    metricGrid.appendChild(metric);
  }
  const checkpoint = document.createElement("span");
  checkpoint.textContent = `Checkpoint: ${payload.checkpoint.split(/[\\/]/).pop()}`;
  el.trainingResults.className = "route-output training-results";
  el.trainingResults.replaceChildren(title, timestamp, metricGrid, checkpoint);
}

async function loadTrainingResults() {
  try {
    const response = await fetch(api("/v1/training-results"));
    if (!response.ok) throw new Error(String(response.status));
    renderTrainingResults(await response.json());
  } catch {
    el.trainingResults.className = "route-output route-error";
    el.trainingResults.textContent = "Remote training results are unavailable.";
  }
}

function parsePoint(value) {
  const match = String(value || "").trim().match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (!match) return null;
  const lat = Number(match[1]);
  const lon = Number(match[2]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

function lngLatToTile(lng, lat, zoom) {
  const scale = 2 ** zoom;
  const latitude = Math.max(-85.05112878, Math.min(85.05112878, lat));
  return {
    x: Math.floor(((lng + 180) / 360) * scale),
    y: Math.floor(
      ((1 - Math.asinh(Math.tan((latitude * Math.PI) / 180)) / Math.PI) / 2) *
        scale
    ),
  };
}


function cartoVoyagerStyle() {
  return {
    version: 8,
    sources: {
      voyager: {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
          "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
          "https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors © CARTO",
      },
    },
    layers: [{ id: "voyager", type: "raster", source: "voyager" }],
  };
}


function removeLayer(id) {
  if (!map) return;
  if (map.getLayer(id)) map.removeLayer(id);
}

function removeSource(id) {
  if (!map) return;
  if (map.getSource(id)) map.removeSource(id);
}


function collectCoordinates(geometry, out) {
  if (!geometry) return;
  const visit = (coords) => {
    if (!Array.isArray(coords)) return;
    if (typeof coords[0] === "number" && typeof coords[1] === "number") {
      out.push(coords);
      return;
    }
    coords.forEach(visit);
  };
  visit(geometry.coordinates);
}

function boundsForGeojson(geojson) {
  const coords = [];
  geojson.features.forEach((feature) => collectCoordinates(feature.geometry, coords));
  if (!coords.length) return null;
  const bounds = new maplibregl.LngLatBounds(coords[0], coords[0]);
  coords.forEach((coord) => bounds.extend(coord));
  return bounds;
}

function fitToGeojson(geojson, options = {}) {
  const bounds = boundsForGeojson(geojson);
  if (!bounds) return;
  const bottomPadding = window.innerWidth > 760 ? 320 : Math.round(Math.min(window.innerHeight * 0.44, 380)) + 36;
  cancelZoomBounce();
  map.fitBounds(
    bounds,
    {
      padding: { top: 58, right: 54, bottom: bottomPadding, left: 54 },
      duration: 600,
      maxZoom: options.maxZoom ?? 12,
    },
    PROGRAMMATIC_CAMERA_EVENT
  );
}


function syncZoomElasticity() {
  const camera = map.cameraForBounds(activeRegionBounds, {
    padding: MAP_NAVIGATION.softFitPaddingPx,
  });
  if (Number.isFinite(camera?.zoom)) {
    softMinimumZoom = camera.zoom;
    map.setMinZoom(Math.max(0, softMinimumZoom - MAP_NAVIGATION.zoomOverscroll));
  }
}

function cancelZoomBounce() {
  if (!zoomBounceActive) return;
  zoomBounceActive = false;
  map.stop();
}

function handleZoomStart(event) {
  // Programmatic camera transitions explicitly cancel a pending bounce before
  // starting, then carry source metadata so this handler leaves them alone.
  if (
    zoomBounceActive &&
    !event?.scenicZoomBounce &&
    !event?.scenicProgrammaticCamera
  ) {
    cancelZoomBounce();
  }
}


function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches === true;
}

function handleZoomEnd(event) {
  if (event?.scenicZoomBounce || event?.scenicProgrammaticCamera) {
    zoomBounceActive = false;
    return;
  }
  // A gesture that interrupted a bounce owns this zoomend; re-check it
  // rather than leaving the map below the soft region overview.
  zoomBounceActive = false;
  if (!Number.isFinite(softMinimumZoom) || map.getZoom() >= softMinimumZoom) return;
  zoomBounceActive = true;
  if (prefersReducedMotion()) {
    // jumpTo is immediate and keeps the correction from overriding the user's
    // reduced-motion preference.
    map.jumpTo({ zoom: softMinimumZoom }, ZOOM_BOUNCE_EVENT);
    return;
  }
  map.easeTo(
    {
      zoom: softMinimumZoom,
      duration: MAP_NAVIGATION.bounceDurationMs,
      easing: (progress) => 1 - (1 - progress) ** 3,
    },
    ZOOM_BOUNCE_EVENT
  );
}


async function loadHeatmap() {
  if (!CONFIG.workingRun) {
    throw new Error(`No scenic run is configured for ${CONFIG.displayRange}`);
  }
  const url = new URL(api("/v1/heatmap"));
  url.searchParams.set("region", CONFIG.sourceRegion);
  url.searchParams.set("run_name", CONFIG.workingRun);
  url.searchParams.set("max_points", "1");
  url.searchParams.set("max_tiles", "1");

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const payload = await response.json();
    const bounds = payload.bounds;
    if (!validHeatmapMetadata(payload)) {
      throw new Error("API returned invalid learned heatmap metadata");
    }
    const imageUrl = new URL(api("/v1/heatmap-image"));
    imageUrl.searchParams.set("region", CONFIG.sourceRegion);
    imageUrl.searchParams.set("run_name", CONFIG.workingRun);
    const imageResponse = await fetch(imageUrl);
    if (!imageResponse.ok) {
      throw new Error(`${imageResponse.status} ${imageResponse.statusText}`);
    }
    const imageBlob = await imageResponse.blob();
    const imageObjectUrl = URL.createObjectURL(imageBlob);
    const scoreGridUrl = new URL(api("/v1/heatmap-scores.bin"));
    scoreGridUrl.searchParams.set("region", CONFIG.sourceRegion);
    scoreGridUrl.searchParams.set("run_name", CONFIG.workingRun);
    const scoreGridResponse = await fetch(scoreGridUrl);
    if (!scoreGridResponse.ok) {
      throw new Error(`${scoreGridResponse.status} ${scoreGridResponse.statusText}`);
    }
    const scoreGrid = {
      zoom: Number(scoreGridResponse.headers.get("X-Scenic-Tile-Zoom")),
      minX: Number(scoreGridResponse.headers.get("X-Scenic-Min-X")),
      minY: Number(scoreGridResponse.headers.get("X-Scenic-Min-Y")),
      width: Number(scoreGridResponse.headers.get("X-Scenic-Grid-Width")),
      height: Number(scoreGridResponse.headers.get("X-Scenic-Grid-Height")),
      values: new DataView(await scoreGridResponse.arrayBuffer()),
    };
    if (
      scoreGrid.zoom !== payload.tile_zoom ||
      !Number.isInteger(scoreGrid.minX) ||
      !Number.isInteger(scoreGrid.minY) ||
      !Number.isInteger(scoreGrid.width) ||
      !Number.isInteger(scoreGrid.height) ||
      scoreGrid.values.byteLength !== scoreGrid.width * scoreGrid.height * 4
    ) {
      throw new Error("API returned invalid learned score grid");
    }
    latestHeatmap = {
      source: "api",
      imageUrl: imageObjectUrl,
      bounds,
      runName: payload.run_name,
      tileZoom: payload.tile_zoom,
      normalization: payload.normalization,
      summary: payload.summary,
      scoreGrid,
    };
    setApiStatus("API: online", "ok");
    return latestHeatmap;
  } catch (error) {
    latestHeatmap = null;
    setApiStatus("API: heatmap unavailable", "warn");
    throw new Error(`Learned heatmap unavailable: ${error.message || error}`);
  }
}

async function checkApiHealth() {
  try {
    const response = await fetch(api("/v1/healthz"));
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    setApiStatus("API: online", "ok");
  } catch {
    setApiStatus("API: offline", "warn");
  }
}



function routeLayer(filterKind, color, width, opacity) {
  const paint = {
    "line-color": color,
    "line-width": width,
    "line-opacity": opacity,
  };
  if (filterKind === "baseline") {
    paint["line-dasharray"] = [2, 2];
  }
  return {
    id: filterKind === "scenic" ? ROUTE_SCENIC : ROUTE_BASELINE,
    type: "line",
    source: ROUTE_SOURCE,
    filter: ["==", ["get", "route_kind"], filterKind],
    layout: { "line-join": "round", "line-cap": "round" },
    paint,
  };
}
function routeEndpointConnectorLayer() {
  return {
    id: ROUTE_ENDPOINT_CONNECTORS,
    type: "line",
    source: ROUTE_ENDPOINT_SOURCE,
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": "#f5c66b",
      "line-width": 3,
      "line-opacity": 0.9,
      "line-dasharray": [1.5, 1.5],
    },
  };
}

function routeCoordinatesMatch(first, second) {
  return first.every((value, index) => {
    const difference = Math.abs(value - second[index]);
    return difference <= Math.max(1e-9, 1e-12 * Math.max(Math.abs(value), Math.abs(second[index])));
  });
}

function validateRouteCoordinate(value, label) {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    value.some((item) => typeof item !== "number" || !Number.isFinite(item))
  ) {
    throw new Error(`${label} must be a finite coordinate pair`);
  }
  const [lon, lat] = value;
  if (lon < -180 || lon > 180 || lat < -90 || lat > 90) {
    throw new Error(`${label} is outside coordinate bounds`);
  }
  return [lon, lat];
}
function routeRequestCoordinate(request, endpoint) {
  const value = request?.[endpoint];
  if (Array.isArray(value)) {
    if (value.length !== 2) {
      throw new Error(`requested ${endpoint} must be a finite coordinate pair`);
    }
    return validateRouteCoordinate([value[1], value[0]], `requested ${endpoint}`);
  }
  return validateRouteCoordinate(
    [value?.lon, value?.lat],
    `requested ${endpoint}`
  );
}

function buildRouteEndpointConnectors(geojson, request) {
  const connectors = { type: "FeatureCollection", features: [] };
  if (request === null || request === undefined) return connectors;
  const requestedStart = routeRequestCoordinate(request, "start");
  const requestedEnd = routeRequestCoordinate(request, "end");
  const scenicFeature = geojson.features.find(
    (feature) => feature.properties?.route_kind === "scenic"
  );
  const snappedStart = scenicFeature.geometry.coordinates[0];
  const snappedEnd = scenicFeature.geometry.coordinates.at(-1);
  const addConnector = (connectorKind, coordinates) => {
    if (routeCoordinatesMatch(coordinates[0], coordinates[1])) return;
    connectors.features.push({
      type: "Feature",
      properties: { connector_kind: connectorKind },
      geometry: { type: "LineString", coordinates },
    });
  };
  addConnector("start", [requestedStart, snappedStart]);
  addConnector("end", [snappedEnd, requestedEnd]);
  return connectors;
}


function validateRouteGeojson(geojson) {
  if (
    !geojson ||
    geojson.type !== "FeatureCollection" ||
    !Array.isArray(geojson.features)
  ) {
    throw new Error("Route response is not a FeatureCollection");
  }
  const expectedKinds = new Set(["scenic", "baseline"]);
  const seenKinds = new Set();
  for (const [index, feature] of geojson.features.entries()) {
    const properties = feature?.properties;
    const routeKind = properties?.route_kind;
    if (
      !feature ||
      typeof feature !== "object" ||
      !properties ||
      typeof properties !== "object" ||
      !expectedKinds.has(routeKind)
    ) {
      throw new Error(`Route feature ${index} has an unexpected route kind`);
    }
    if (seenKinds.has(routeKind)) {
      throw new Error(`Route response contains duplicate ${routeKind} features`);
    }
    seenKinds.add(routeKind);
    const geometry = feature.geometry;
    if (
      !geometry ||
      geometry.type !== "LineString" ||
      !Array.isArray(geometry.coordinates) ||
      geometry.coordinates.length < 2
    ) {
      throw new Error(`${routeKind} route geometry is missing or invalid`);
    }
    const coordinates = geometry.coordinates.map((coordinate, coordinateIndex) =>
      validateRouteCoordinate(
        coordinate,
        `${routeKind} geometry coordinate ${coordinateIndex}`
      )
    );
    // Road geometry starts and ends at the feature's snapped road positions.
  }
  for (const routeKind of expectedKinds) {
    if (!seenKinds.has(routeKind)) {
      throw new Error(`Route response is missing ${routeKind} geometry`);
    }
  }
  return geojson;
}

function renderRoute(geojson, request = null) {
  validateRouteGeojson(geojson);
  const connectors = buildRouteEndpointConnectors(geojson, request);
  if (
    !map ||
    (typeof mapReady !== "undefined" && !mapReady) ||
    typeof map.isStyleLoaded === "function" && !map.isStyleLoaded()
  ) {
    return;
  }
  const source = map.getSource(ROUTE_SOURCE);
  if (source && typeof source.setData === "function") {
    source.setData(geojson);
  } else {
    if (source) removeSource(ROUTE_SOURCE);
    map.addSource(ROUTE_SOURCE, { type: "geojson", data: geojson });
    if (!map.getLayer(ROUTE_BASELINE)) {
      map.addLayer(routeLayer("baseline", "#ffffff", 4, 0.62));
    }
    if (!map.getLayer(ROUTE_SCENIC)) {
      map.addLayer(routeLayer("scenic", "#62c58a", 5.5, 0.96));
    }
  }
  const connectorSource = map.getSource(ROUTE_ENDPOINT_SOURCE);
  if (connectors.features.length) {
    if (connectorSource && typeof connectorSource.setData === "function") {
      connectorSource.setData(connectors);
    } else {
      if (connectorSource) removeSource(ROUTE_ENDPOINT_SOURCE);
      map.addSource(ROUTE_ENDPOINT_SOURCE, { type: "geojson", data: connectors });
    }
    if (!map.getLayer(ROUTE_ENDPOINT_CONNECTORS)) {
      map.addLayer(routeEndpointConnectorLayer());
    }
  } else {
    removeLayer(ROUTE_ENDPOINT_CONNECTORS);
    removeSource(ROUTE_ENDPOINT_SOURCE);
  }
  fitToGeojson(
    { type: "FeatureCollection", features: [...geojson.features, ...connectors.features] },
    { maxZoom: 12 }
  );
}

function clearRoute() {
  routeRequestSequence += 1;
  activeRouteRequest?.controller.abort();
  pendingRouteRender = null;
  activeRouteRequest = null;
  latestRoutePayload = null;
  removeLayer(ROUTE_ENDPOINT_CONNECTORS);
  removeLayer(ROUTE_SCENIC);
  removeLayer(ROUTE_BASELINE);
  removeSource(ROUTE_ENDPOINT_SOURCE);
  removeSource(ROUTE_SOURCE);
  if (el.submitRoute) {
    el.submitRoute.disabled = !mapReady || !CONFIG.workingRun || !routePlanningAvailable();
    el.submitRoute.textContent = "Plan route";
  }
  setRouteOutput("Waiting for submit", "Enter start/end coordinates as <code>lat, lon</code>, then submit.");
}

function signedNumber(value, digits, suffix = "") {
  if (value === null || value === undefined || value === "") return "--";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(digits)}${suffix}`;
}

function displayRatio(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "--";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(digits)}×` : "--";
}

function displayText(value) {
  return value === null || value === undefined || value === "" ? "--" : String(value);
}

function formatWithUnit(value, digits, unit) {
  const formatted = formatNumber(value, digits);
  return formatted === "--" ? "--" : `${formatted} ${unit}`;
}

function displayBoolean(value) {
  return typeof value === "boolean" ? (value ? "Yes" : "No") : "--";
}

function getSearchDiagnostics(diagnostics) {
  const searchDiagnostics = diagnostics?.search_diagnostics;
  return searchDiagnostics &&
    typeof searchDiagnostics === "object" &&
    !Array.isArray(searchDiagnostics)
    ? searchDiagnostics
    : {};
}
function routeFailurePresentation(code) {
  if (code === "no_route_found") {
    return { status: "API: no route found", title: "No route found" };
  }
  if (code === "route_endpoint_outside_coverage") {
    return {
      status: "API: outside route coverage",
      title: "Outside route coverage",
    };
  }
  return { status: "API: route failed", title: "Route failed" };
}

const REASON_PROSE = Object.freeze({
  approximation_did_not_find_scenic_improvement:
    "Search found no more-scenic route",
  same_route: "Scenic and fastest routes use the same edge sequence",
  no_route: "No route satisfies the requested controls",
  no_feasible_route: "No route satisfies the requested controls",
  avoid_highways_no_route: "No route satisfies the avoid-highways constraint",
});

function humanizeReason(reason, optimalityGap, deadlineReached = false) {
  if (reason === null || reason === undefined || reason === "") return "--";
  const code = String(reason).trim();
  const deadlineWasReached = deadlineReached === true;
  const phrase =
    deadlineWasReached &&
    code === "approximation_did_not_find_scenic_improvement"
      ? "Search did not find a better weighted route within the search budget"
      : REASON_PROSE[code] ||
        code
          .replace(/[_-]+/g, " ")
          .replace(/\b\w/g, (letter) => letter.toUpperCase());
  if (
    optimalityGap === null ||
    optimalityGap === undefined ||
    optimalityGap === ""
  ) {
    return phrase;
  }
  const gap = Number(optimalityGap);
  if (
    code === "approximation_did_not_find_scenic_improvement" &&
    Number.isFinite(gap)
  ) {
    return `${phrase}; optimality gap remains ${gap.toFixed(4)}.`;
  }
  return phrase;
}

function humanizeRouteState(value) {
  if (typeof value === "boolean") {
    return value ? "Yes (same route)" : "No (different route)";
  }
  return humanizeReason(value);
}

function routeDiagnosticsMarkup(payload) {
  const diagnostics = payload?.diagnostics;
  if (
    !diagnostics ||
    typeof diagnostics !== "object" ||
    Array.isArray(diagnostics) ||
    Object.keys(diagnostics).length === 0
  ) {
    return "";
  }
  const scenic = payload?.routes?.scenic || {};
  const scoreMapping = payload?.score_mapping || {};
  const parts = [];
  const append = (label, value) => {
    if (value !== null && value !== undefined && value !== "" && value !== "--") {
      parts.push(`${label} ${value}`);
    }
  };
  const elapsedRaw = diagnostics.planning_elapsed_ms;
  const elapsedMs = Number(elapsedRaw);
  if (
    elapsedRaw !== null &&
    elapsedRaw !== undefined &&
    elapsedRaw !== "" &&
    Number.isFinite(elapsedMs)
  ) {
    append("Planning", `${formatNumber(elapsedMs, 0)} ms`);
  }
  append(
    "Requested weight",
    displayText(formatNumber(diagnostics.requested_scenic_weight, 2))
  );
  append(
    "Applied weight",
    displayText(formatNumber(diagnostics.applied_scenic_weight, 2))
  );
  append("Requested cap", displayRatio(diagnostics.requested_max_detour_factor));
  append("Applied cap", displayRatio(diagnostics.applied_max_detour_factor));
  append("Actual ratio", displayRatio(diagnostics.scenic_fastest_duration_ratio));
  const optimization = [
    displayText(diagnostics.optimization_mode),
    displayText(diagnostics.optimization_status),
  ]
    .filter((value) => value !== "--")
    .join(" / ");
  append("Optimization", optimization);
  append("Gap", displayText(formatNumber(diagnostics.optimality_gap, 4)));
  append(
    "Certified UB",
    displayText(formatNumber(diagnostics.certified_upper_bound, 4))
  );
  const mappingCoverageRaw = diagnostics.score_mapping_coverage;
  const mappingCoverage = Number(mappingCoverageRaw);
  if (
    mappingCoverageRaw !== null &&
    mappingCoverageRaw !== undefined &&
    mappingCoverageRaw !== "" &&
    Number.isFinite(mappingCoverage)
  ) {
    append("Tile/report score coverage", `${formatNumber(mappingCoverage * 100, 0)}%`);
  }
  append("Highways", displayText(scenic.highway_count));
  if (Array.isArray(scenic.score_run)) {
    append("Score run", `${scenic.score_run.length} edges`);
  }
  if (diagnostics.highway_avoidance_fallback === true) {
    append("Scenic highway avoidance", "best effort; major road required");
  } else if (diagnostics.highway_avoidance_mode === "strict") {
    append("Scenic highway avoidance", "strict");
  } else if (typeof diagnostics.avoid_highways_applied === "boolean") {
    append("Scenic highway avoidance", diagnostics.avoid_highways_applied ? "on" : "off");
  }
  if (typeof diagnostics.baseline_avoid_highways_applied === "boolean") {
    append(
      "Baseline avoids highways",
      diagnostics.baseline_avoid_highways_applied ? "on" : "off",
    );
  }
  if (scoreMapping.normalization) {
    append("Normalization", String(scoreMapping.normalization));
  }
  return parts.length
    ? `<small class="route-diagnostics">${escapeHtml(parts.join(" · "))}</small>`
    : "";
}

function routeOutputMarkup(payload) {
  const scenicFeature = payload.geojson?.features?.find(
    (feature) => feature?.properties?.route_kind === "scenic"
  );
  const snappedStart = scenicFeature?.properties?.snapped_start;
  const snappedEnd = scenicFeature?.properties?.snapped_end;
  const snapped =
    Array.isArray(snappedStart) &&
    snappedStart.length === 2 &&
    Array.isArray(snappedEnd) &&
    snappedEnd.length === 2
      ? ` Road-snapped endpoints: ${formatNumber(snappedStart[0], 5)}, ${formatNumber(snappedStart[1], 5)} → ${formatNumber(snappedEnd[0], 5)}, ${formatNumber(snappedEnd[1], 5)}.`
      : "";
  const connectors = buildRouteEndpointConnectors(payload.geojson, payload.request);
  const connectorCopy = connectors.features.length
    ? " Dashed lines connect selected points to the road network."
    : "";
  const status = `Scenic and baseline routes are shown on the map.${connectorCopy}${snapped}`;
  return verboseRouteResults ? `${status}${routeDiagnosticsMarkup(payload)}` : status;
}

function setRouteResultsVerbose(verbose) {
  verboseRouteResults = Boolean(verbose);
  if (el.verboseRouteResults) {
    el.verboseRouteResults.checked = verboseRouteResults;
  }
  el.routeResultsDialog?.classList?.toggle?.("verbose", verboseRouteResults);
  if (el.routeDiagnostics) {
    el.routeDiagnostics.hidden = !verboseRouteResults;
    el.routeDiagnostics.setAttribute("aria-hidden", String(!verboseRouteResults));
  }
  if (latestRoutePayload) {
    setRouteOutput("Route computed", routeOutputMarkup(latestRoutePayload));
  }
}

function renderRouteComparison(payload) {
  const scenic = payload.routes?.scenic;
  const baseline = payload.routes?.baseline;
  const diagnostics = payload.diagnostics;
  const deltas = payload.deltas;
  if (!scenic || !diagnostics) {
    throw new Error("API returned an incomplete route comparison");
  }
  const searchDiagnostics = getSearchDiagnostics(diagnostics);
  setText(el.scenicDistance, `${formatNumber(scenic.total_distance_km, 1)} km`);
  setText(el.scenicDuration, `${formatNumber(scenic.estimated_duration_minutes, 0)} min`);
  setText(el.scenicScore, `${formatNumber(scenic.raw_scenic_score, 2)} / 10`);
  setText(el.scenicRawScore, formatNumber(scenic.raw_scenic_score, 2));
  setText(el.scenicNormalizedScore, formatNumber(scenic.normalized_scenic_score, 4));
  setText(
    el.baselineDistance,
    baseline ? `${formatNumber(baseline.total_distance_km, 1)} km` : "--"
  );
  setText(
    el.baselineDuration,
    baseline ? `${formatNumber(baseline.estimated_duration_minutes, 0)} min` : "--"
  );
  setText(
    el.baselineScore,
    baseline ? `${formatNumber(baseline.raw_scenic_score, 2)} / 10` : "--"
  );
  setText(
    el.baselineRawScore,
    baseline ? formatNumber(baseline.raw_scenic_score, 2) : "--"
  );
  setText(
    el.baselineNormalizedScore,
    baseline ? formatNumber(baseline.normalized_scenic_score, 4) : "--"
  );
  setText(
    el.scenicDistanceDelta,
    deltas ? signedNumber(deltas.distance_km, 1, " km") : "--"
  );
  setText(
    el.scenicDurationDelta,
    deltas ? signedNumber(deltas.duration_min, 0, " min") : "--"
  );
  setText(
    el.scenicScoreDelta,
    deltas ? signedNumber(deltas.scenic_score_absolute, 2) : "--"
  );
  setText(
    el.scenicScoreUpliftAbsolute,
    signedNumber(diagnostics.scenic_score_delta_absolute, 2)
  );
  setText(
    el.scenicScoreUpliftRelative,
    diagnostics.scenic_score_delta_relative !== null &&
      diagnostics.scenic_score_delta_relative !== undefined &&
      Number.isFinite(Number(diagnostics.scenic_score_delta_relative))
      ? signedNumber(
          Number(diagnostics.scenic_score_delta_relative) * 100,
          1,
          "%"
        )
      : "--"
  );
  setText(
    el.requestedScenicWeight,
    formatNumber(diagnostics.requested_scenic_weight, 2)
  );
  setText(
    el.appliedScenicWeight,
    formatNumber(diagnostics.applied_scenic_weight, 2)
  );
  setText(
    el.requestedDetourCap,
    displayRatio(diagnostics.requested_max_detour_factor)
  );
  setText(
    el.appliedDetourCap,
    displayRatio(diagnostics.applied_max_detour_factor)
  );
  setText(
    el.actualDurationRatio,
    displayRatio(diagnostics.scenic_fastest_duration_ratio)
  );
  setText(el.optimizationMode, diagnostics.optimization_mode);
  setText(el.optimizationStatus, diagnostics.optimization_status);
  setText(el.searchMode, displayText(searchDiagnostics.mode));
  setText(
    el.searchTimeLimit,
    formatWithUnit(searchDiagnostics.time_limit_seconds, 2, "s")
  );
  setText(
    el.searchLabelsGenerated,
    formatNumber(searchDiagnostics.labels_generated, 0)
  );
  setText(
    el.searchLabelsExpanded,
    formatNumber(searchDiagnostics.labels_expanded, 0)
  );
  setText(
    el.searchLabelsPruned,
    formatNumber(searchDiagnostics.labels_pruned, 0)
  );
  setText(
    el.searchMaxFrontierSize,
    formatNumber(searchDiagnostics.max_frontier_size, 0)
  );
  setText(
    el.searchRemainingFrontierSize,
    formatNumber(searchDiagnostics.remaining_frontier_size, 0)
  );
  setText(
    el.searchElapsed,
    formatWithUnit(searchDiagnostics.elapsed_ms, 0, "ms")
  );
  setText(
    el.searchDeadlineReached,
    displayBoolean(searchDiagnostics.deadline_reached)
  );
  setText(el.optimalityGap, formatNumber(diagnostics.optimality_gap, 4));
  setText(
    el.certifiedUpperBound,
    formatNumber(diagnostics.certified_upper_bound, 4)
  );
  setText(el.sameRoute, humanizeRouteState(diagnostics.same_route));
  setText(
    el.noBetterRouteReason,
    humanizeReason(
      diagnostics.no_better_route_reason,
      diagnostics.optimality_gap,
      searchDiagnostics.deadline_reached === true
    )
  );
  if (!el.routeResultsDialog.open) {
    el.routeResultsDialog.showModal();
    el.closeRouteDialogBtn.focus();
  }
}


async function fetchValidatedRoute() {
  if (!CONFIG.workingRun) {
    throw new Error(`No validated route run is configured for ${CONFIG.displayRange}`);
  }
  const url = new URL(api("/v1/validated-route"));
  url.searchParams.set("region", CONFIG.sourceRegion);
  url.searchParams.set("run_name", CONFIG.workingRun);
  if (routePlanningIntentionallyDisabled()) {
    throw new Error(`Route planning is disabled for ${CONFIG.displayRange}`);
  }
  if (selectedRegionMetadata?.graph_exists === false) {
    throw new Error(`Route graph is unavailable for ${CONFIG.displayRange}`);
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const result = await response.json();
  validateRouteGeojson(result.geojson);
  const kinds = new Set(
    result.geojson?.features?.map((feature) => feature.properties?.route_kind)
  );
  if (
    !kinds.has("scenic") ||
    !kinds.has("baseline") ||
    !result.routes?.scenic ||
    !result.routes?.baseline ||
    !result.score_mapping
  ) {
    throw new Error("API returned invalid validated route artifacts");
  }
  return result;
}

function newSearchSessionToken() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function installAddressSearch(which) {
  const input = which === "start" ? el.startInput : el.endInput;
  const suggestions = which === "start" ? el.startSuggestions : el.endSuggestions;
  const clearButton = which === "start" ? el.clearStartBtn : el.clearEndBtn;
  let sessionToken = newSearchSessionToken();
  let currentSuggestions = [];
  let activeSuggestion = -1;
  let timer = null;
  let addressRequestSequence = 0;
  let activeAddressRequest = null;

  const beginAddressRequest = (fingerprint) => {
    activeAddressRequest?.controller.abort();
    const token = {
      id: ++addressRequestSequence,
      fingerprint,
      controller: new AbortController(),
    };
    activeAddressRequest = token;
    return token;
  };
  const isCurrentAddressRequest = (token, fingerprint) =>
    activeAddressRequest === token &&
    token.id === addressRequestSequence &&
    token.fingerprint === fingerprint;

  const closeSuggestions = ({ invalidate = true } = {}) => {
    if (invalidate) {
      beginAddressRequest(JSON.stringify({ kind: "close" }));
    }
    currentSuggestions = [];
    activeSuggestion = -1;
    input.removeAttribute("aria-activedescendant");
    input.setAttribute("aria-expanded", "false");
    suggestions.replaceChildren();
    suggestions.classList.remove("open");
  };
  const renderStatus = (text, kind) => {
    currentSuggestions = [];
    activeSuggestion = -1;
    input.removeAttribute("aria-activedescendant");
    input.setAttribute("aria-expanded", "true");
    const row = document.createElement("div");
    row.className = `address-suggestion-status ${kind}`;
    row.setAttribute("role", "option");
    row.setAttribute("aria-disabled", "true");
    row.textContent = text;
    suggestions.replaceChildren(row);
    suggestions.classList.add("open");
  };
  const selectSuggestion = async (suggestion) => {
    const fingerprint = JSON.stringify({
      kind: "retrieve",
      mapbox_id: suggestion.mapbox_id,
      session_token: sessionToken,
      region: CONFIG.sourceRegion,
    });
    const token = beginAddressRequest(fingerprint);
    const url = new URL(api("/v1/search/retrieve"));
    url.searchParams.set("mapbox_id", suggestion.mapbox_id);
    url.searchParams.set("session_token", sessionToken);
    renderStatus("Searching…", "loading");
    try {
      const response = await fetch(url, { signal: token.controller.signal });
      if (!isCurrentAddressRequest(token, fingerprint)) return;
      if (!response.ok) throw new Error(String(response.status));
      const result = (await response.json()).result;
      if (!isCurrentAddressRequest(token, fingerprint)) return;
      if (!Number.isFinite(result?.lat) || !Number.isFinite(result?.lon)) {
        throw new Error("invalid address result");
      }
      selectedRoutePoints[which] = { lat: result.lat, lon: result.lon };
      input.value = result.full_address || result.name || suggestion.full_address;
      sessionToken = newSearchSessionToken();
      closeSuggestions({ invalidate: false });
    } catch (error) {
      if (
        error?.name !== "AbortError" &&
        isCurrentAddressRequest(token, fingerprint)
      ) {
        renderStatus("Address search unavailable", "error");
      }
    }
  };
  const renderSuggestions = (rows) => {
    if (!rows.length) {
      renderStatus("No matching locations", "empty");
      return;
    }
    currentSuggestions = rows;
    activeSuggestion = -1;
    input.removeAttribute("aria-activedescendant");
    input.setAttribute("aria-expanded", "true");
    suggestions.replaceChildren();
    rows.forEach((row, index) => {
      const option = document.createElement("div");
      option.id = `${which}-suggestion-${index}`;
      option.className = "address-suggestion";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      const name = document.createElement("strong");
      name.textContent = row.name;
      const address = document.createElement("small");
      address.textContent = row.full_address || row.name;
      option.append(name, address);
      option.addEventListener("mousedown", (event) => {
        event.preventDefault();
        void selectSuggestion(row);
      });
      suggestions.appendChild(option);
    });
    suggestions.classList.add("open");
  };

  input.addEventListener("input", () => {
    selectedRoutePoints[which] = null;
    clearTimeout(timer);
    const query = input.value.trim();
    const fingerprint = JSON.stringify({
      kind: "suggest",
      query,
      session_token: sessionToken,
      region: CONFIG.sourceRegion,
    });
    const token = beginAddressRequest(fingerprint);
    if (parsePoint(query) || query.length < 3) {
      closeSuggestions();
      return;
    }
    timer = setTimeout(async () => {
      if (!isCurrentAddressRequest(token, fingerprint)) return;
      const url = new URL(api("/v1/search/suggest"));
      url.searchParams.set("q", query);
      url.searchParams.set("session_token", sessionToken);
      url.searchParams.set("region", CONFIG.sourceRegion);
      renderStatus("Searching…", "loading");
      try {
        const response = await fetch(url, { signal: token.controller.signal });
        if (!isCurrentAddressRequest(token, fingerprint)) return;
        if (!response.ok) throw new Error(String(response.status));
        const result = await response.json();
        if (!isCurrentAddressRequest(token, fingerprint)) return;
        renderSuggestions(result.suggestions || []);
      } catch (error) {
        if (
          error?.name !== "AbortError" &&
          isCurrentAddressRequest(token, fingerprint)
        ) {
          renderStatus("Address search unavailable", "error");
        }
      }
    }, 200);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeSuggestions();
      return;
    }
    if (!currentSuggestions.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const delta = event.key === "ArrowDown" ? 1 : -1;
      activeSuggestion =
        (activeSuggestion + delta + currentSuggestions.length) %
        currentSuggestions.length;
      suggestions.querySelectorAll(".address-suggestion").forEach((option, index) => {
        option.setAttribute("aria-selected", String(index === activeSuggestion));
      });
      input.setAttribute("aria-activedescendant", `${which}-suggestion-${activeSuggestion}`);
    } else if (event.key === "Enter" && activeSuggestion >= 0) {
      event.preventDefault();
      void selectSuggestion(currentSuggestions[activeSuggestion]);
    }
  });
  input.addEventListener("blur", (event) => {
    if (!suggestions.contains(event.relatedTarget)) {
      setTimeout(closeSuggestions, 120);
    }
  });
  clearButton?.addEventListener("click", () => {
    clearTimeout(timer);
    beginAddressRequest(JSON.stringify({ kind: "clear" }));
    input.value = "";
    selectedRoutePoints[which] = null;
    sessionToken = newSearchSessionToken();
    closeSuggestions();
    input.focus();
  });
}


async function planRoute(event) {
  event.preventDefault();
  const requestId = ++routeRequestSequence;
  pendingRouteRender = null;
  latestRoutePayload = null;
  activeRouteRequest?.controller.abort();
  activeRouteRequest = null;
  const start = parsePoint(el.startInput.value) || selectedRoutePoints.start;
  const end = parsePoint(el.endInput.value) || selectedRoutePoints.end;
  if (!start || !end) {
    setRouteOutput("Invalid input", "Enter coordinates or choose an address suggestion.");
    (!start ? el.startInput : el.endInput).focus();
    return;
  }

  const requestBody = Object.freeze({
    region: CONFIG.sourceRegion,
    run_name: CONFIG.workingRun,
    start: Object.freeze({ lat: start.lat, lon: start.lon }),
    end: Object.freeze({ lat: end.lat, lon: end.lon }),
    scenic_weight: DEFAULTS.scenicWeight,
    max_detour_factor: Number(el.detourFactor.value),
    avoid_highways: Boolean(el.avoidHighways?.checked),
    include_baseline: true,
  });
  const fingerprint = JSON.stringify(requestBody);
  const controller = new AbortController();
  const requestToken = Object.freeze({
    requestId,
    fingerprint,
    controller,
    endpoints: Object.freeze({
      start: requestBody.start,
      end: requestBody.end,
    }),
  });
  activeRouteRequest = requestToken;

  if (el.submitRoute) {
    el.submitRoute.disabled = true;
    el.submitRoute.textContent = "Planning...";
  }
  setRouteOutput(
    "Computing route",
    "Calling <code>/v1/route/compare</code> with the same run used by the heatmap."
  );

  const isCurrent = () =>
    activeRouteRequest === requestToken &&
    requestToken.requestId === routeRequestSequence &&
    requestToken.fingerprint === fingerprint;
  try {
    const response = await fetch(api("/v1/route/compare"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Route-Request-ID": String(requestId),
        "X-Route-Request-Fingerprint": fingerprint,
      },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
    if (!isCurrent()) return;
    const payload = await response.json();
    if (!isCurrent()) return;
    if (!response.ok) {
      const detail = payload?.detail;
      const failureCode =
        detail && typeof detail === "object" ? detail.error : null;
      const message =
        detail && typeof detail === "object" ? detail.message : detail;
      const hint = detail && typeof detail === "object" ? detail.hint : null;
      const error = new Error(
        [message || `${response.status} ${response.statusText}`, hint]
          .filter(Boolean)
          .join(" — ")
      );
      error.routeFailureCode = failureCode;
      throw error;
    }
    const routeGeojson = validateRouteGeojson(payload.geojson);
    if (mapReady && map?.isStyleLoaded?.()) {
      renderRoute(routeGeojson, payload.request);
    } else {
      pendingRouteRender = { requestId, geojson: routeGeojson, request: payload.request };
    }
    renderRouteComparison(payload);
    latestRoutePayload = payload;
    setRouteOutput("Route computed", routeOutputMarkup(payload));
    setApiStatus("API: online", "ok");
  } catch (error) {
    if (error?.name === "AbortError" || !isCurrent()) return;
    const presentation = routeFailurePresentation(error?.routeFailureCode);
    setApiStatus(presentation.status, "warn");
    setRouteOutput(presentation.title, escapeHtml(error.message || error));
  } finally {
    if (isCurrent()) {
      activeRouteRequest = null;
      if (el.submitRoute) {
        el.submitRoute.disabled = false;
        el.submitRoute.textContent = "Plan route";
      }
    }
  }
}

function buildShareUrl({ start, end, maxDetour, avoidHighways }) {
  const url = new URL(window.location.pathname, window.location.origin);
  url.searchParams.set("start", `${start.lat},${start.lon}`);
  url.searchParams.set("end", `${end.lat},${end.lon}`);
  url.searchParams.set("max_detour", String(maxDetour));
  url.searchParams.set("avoid_highways", avoidHighways ? "true" : "false");
  url.searchParams.set("source", CONFIG.sourceRegion);
  if (CONFIG.workingRun) url.searchParams.set("run", CONFIG.workingRun);
  return url.toString();
}

function applyUrlParams() {
  const start = parsePoint(params.get("start") || "");
  const end = parsePoint(params.get("end") || "");
  if (start) el.startInput.value = `${start.lat},${start.lon}`;
  if (end) el.endInput.value = `${end.lat},${end.lon}`;
  const applyRange = (key, input) => {
    const raw = params.get(key);
    if (raw === null) return;
    const value = Number(raw);
    const min = Number(input.min);
    const max = Number(input.max);
    if (Number.isFinite(value) && value >= min && value <= max) {
      input.value = String(value);
    }
  };
  applyRange("max_detour", el.detourFactor);
  const avoidHighways = params.get("avoid_highways");
  if (avoidHighways !== null && el.avoidHighways) {
    el.avoidHighways.checked = ["1", "true", "yes", "on"].includes(
      avoidHighways.toLowerCase()
    );
  }
}

async function copyShareUrl() {
  const start = parsePoint(el.startInput.value) || selectedRoutePoints.start;
  const end = parsePoint(el.endInput.value) || selectedRoutePoints.end;
  if (!start || !end) return;
  const shareUrl = buildShareUrl({
    start,
    end,
    maxDetour: Number(el.detourFactor.value),
    avoidHighways: Boolean(el.avoidHighways?.checked),
  });
  const originalLabel = el.shareRouteBtn.textContent;
  let copied = false;
  try {
    const writeText = globalThis.navigator?.clipboard?.writeText;
    if (typeof writeText === "function") {
      await writeText.call(globalThis.navigator.clipboard, shareUrl);
      copied = true;
    }
  } catch {
    copied = false;
  }
  if (!copied) {
    const textarea = document.createElement("textarea");
    try {
      textarea.value = shareUrl;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      copied = document.execCommand("copy") === true;
    } catch {
      copied = false;
    } finally {
      textarea.remove();
    }
  }
  el.shareRouteBtn.textContent = copied ? "Copied" : "Copy failed";
  window.setTimeout(() => {
    el.shareRouteBtn.textContent = originalLabel;
  }, 1600);
}

function initBindings() {
  applyUrlParams();
  setRouteResultsVerbose(false);
  el.submitRoute.disabled = true;
  const syncRangeOutputs = () => {
    setText(el.detourOut, `${Number(el.detourFactor.value).toFixed(2)}×`);
  };
  el.detourFactor?.addEventListener("input", syncRangeOutputs);
  syncRangeOutputs();
  el.routeForm?.addEventListener("submit", planRoute);
  el.verboseRouteResults?.addEventListener("change", () => {
    setRouteResultsVerbose(el.verboseRouteResults.checked);
  });
  el.regionSelect?.addEventListener("change", () => {
    const region = el.regionSelect.value;
    const run =
      el.regionSelect.selectedOptions[0]?.dataset.run ||
      el.runSelect?.value ||
      "";
    window.location.assign(selectionUrl(region, run));
  });
  el.runSelect?.addEventListener("change", () => {
    window.location.assign(selectionUrl(el.regionSelect.value, el.runSelect.value));
  });
  el.clearRoute?.addEventListener("click", clearRoute);
  el.closeRouteDialogBtn?.addEventListener("click", () => el.routeResultsDialog.close());
  el.shareRouteBtn?.addEventListener("click", copyShareUrl);
  el.routeResultsDialog?.addEventListener("click", (event) => {
    if (event.target === el.routeResultsDialog) el.routeResultsDialog.close();
  });
  el.routeResultsDialog?.addEventListener("close", () => el.submitRoute.focus());
}

function initAddressSearch() {
  installAddressSearch("start");
  installAddressSearch("end");
}

async function main() {
  initBindings();
  const artifactErrors = [];
  try {
    await loadSupportedRegions();
  } catch (error) {
    const message = `Region metadata unavailable: ${error.message || error}`;
    artifactErrors.push(message);
    setText(el.regionStatus, message);
    setApiStatus("API: regions unavailable", "warn");
  }
  initAddressSearch();
  void loadTrainingResults();
  await checkApiHealth();

  let heatmap = null;
  let validatedRoute = null;
  try {
    heatmap = await loadHeatmap();
  } catch (error) {
    artifactErrors.push(error.message || String(error));
  }
  if (!routePlanningIntentionallyDisabled()) {
    try {
      validatedRoute = await fetchValidatedRoute();
    } catch (error) {
      artifactErrors.push(`Validated route unavailable: ${error.message || error}`);
    }
  }

  const style = cartoVoyagerStyle();
  if (heatmap) {
    style.sources[HEATMAP_SOURCE] = {
      type: "image",
      url: heatmap.imageUrl,
      coordinates: [
        [heatmap.bounds.min_lon, heatmap.bounds.max_lat],
        [heatmap.bounds.max_lon, heatmap.bounds.max_lat],
        [heatmap.bounds.max_lon, heatmap.bounds.min_lat],
        [heatmap.bounds.min_lon, heatmap.bounds.min_lat],
      ],
    };
    style.layers.push({
      id: HEATMAP_FILL,
      type: "raster",
      source: HEATMAP_SOURCE,
      paint: {
        "raster-opacity": 0.78,
        "raster-resampling": "nearest",
      },
    });
  }

  if (validatedRoute) {
    const validatedConnectors = buildRouteEndpointConnectors(
      validatedRoute.geojson,
      validatedRoute.request
    );
    style.sources[ROUTE_SOURCE] = { type: "geojson", data: validatedRoute.geojson };
    style.layers.push(
      routeLayer("baseline", "#ffffff", 4, 0.62),
      routeLayer("scenic", "#62c58a", 5.5, 0.96)
    );
    if (validatedConnectors.features.length) {
      style.sources[ROUTE_ENDPOINT_SOURCE] = {
        type: "geojson",
        data: validatedConnectors,
      };
      style.layers.push(routeEndpointConnectorLayer());
    }
  }

  const camera = regionCamera(selectedRegionMetadata);
  map = new maplibregl.Map({
    container: "map",
    style,
    center: camera.center,
    zoom: camera.zoom,
    attributionControl: true,
    dragRotate: false,
    pitchWithRotate: false,
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
  map.on("zoomstart", handleZoomStart);
  map.on("zoomend", handleZoomEnd);
  map.on("load", () => {
    mapReady = true;
    const pending = pendingRouteRender;
    pendingRouteRender = null;
    if (pending?.requestId === routeRequestSequence) {
      renderRoute(pending.geojson, pending.request);
    }
    syncZoomElasticity();
    el.submitRoute.disabled = !CONFIG.workingRun || !routePlanningAvailable();
  });
  map.on("resize", syncZoomElasticity);
  if (heatmap) {

    const mapCanvas = map.getCanvas();
    mapCanvas.addEventListener("pointermove", (event) => {
      const lngLat = map.unproject([event.offsetX, event.offsetY]);
      const tile = lngLatToTile(lngLat.lng, lngLat.lat, heatmap.tileZoom);
      const gridX = tile.x - heatmap.scoreGrid.minX;
      const gridY = tile.y - heatmap.scoreGrid.minY;
      const inGrid =
        gridX >= 0 &&
        gridY >= 0 &&
        gridX < heatmap.scoreGrid.width &&
        gridY < heatmap.scoreGrid.height;
      const score = inGrid
        ? heatmap.scoreGrid.values.getFloat32(
            (gridY * heatmap.scoreGrid.width + gridX) * 4,
            true
          )
        : Number.NaN;
      if (Number.isFinite(score)) {
        setText(el.inspectorScore, score.toFixed(2));
        setText(
          el.inspectorCoords,
          `${lngLat.lat.toFixed(4)}, ${lngLat.lng.toFixed(4)} · z${heatmap.tileZoom}/${tile.x}/${tile.y}`
        );
      } else {
        setText(el.inspectorScore, "--");
        setText(el.inspectorCoords, "No learned score for this tile");
      }
    });
    mapCanvas.addEventListener("pointerleave", () => {
      setText(el.inspectorScore, "--");
      setText(el.inspectorCoords, "Move over a heatmap cell");
    });
  }
  if (heatmap) {
    setText(el.inspectorScore, `${heatmap.summary.total_tiles.toLocaleString()} tiles`);
    setText(el.inspectorCoords, `Learned scores · z${heatmap.tileZoom}`);
  }
  if (heatmap && routePlanningIntentionallyDisabled()) {
    const message =
      selectedRegionMetadata.description ||
      "Route planning is unavailable because this heatmap has no configured road graph.";
    setText(el.regionStatus, `${CONFIG.displayRange} · heatmap only`);
    setRouteOutput(`${CONFIG.displayRange} heatmap`, escapeHtml(message));
  }
  if (validatedRoute) {
    const title =
      CONFIG.sourceRegion === DEFAULTS.sourceRegion
        ? "Burlington → Bangor"
        : `${CONFIG.displayRange} validated route`;
    setRouteOutput(title, "Validated scenic and baseline routes are shown on the map.");
  }
  if (artifactErrors.length) {
    setApiStatus("API: artifacts unavailable", "warn");
    setText(el.regionStatus, artifactErrors.join(" "));
    if (!validatedRoute) {
      setRouteOutput(
        `${CONFIG.displayRange} artifacts unavailable`,
        escapeHtml(artifactErrors.join(" "))
      );
    }
  }
}

main().catch((error) => {
  setApiStatus("Viewer failed", "warn");
  setRouteOutput("Viewer failed", escapeHtml(error.message || error));
});
