const DEFAULTS = Object.freeze({
  displayRange: "new_england_north",
  sourceRegion: "new_england_north",
  workingRun: "new_england_north_z14_v6_learned",
  sourceModel: "models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt",
  activeRegistryModel: "models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt",
  apiBase: `${window.location.origin}/api`,
  center: [-70.15869140625, 44.99533046578542],
  zoom: 6.2,
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
const CONFIG = Object.freeze({
  displayRange: params.get("display") || DEFAULTS.displayRange,
  sourceRegion: params.get("source") || DEFAULTS.sourceRegion,
  workingRun: params.get("run") || DEFAULTS.workingRun,
  sourceModel: DEFAULTS.sourceModel,
  activeRegistryModel: DEFAULTS.activeRegistryModel,
});

// API base is a runtime variable (no longer bound to a DOM input). It can be
// overridden via ?api=... and, if an `#apiBase` input is ever re-introduced, by
// the user editing that field. Removed DOM must not break fetch paths.
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

// Element lookups are null-safe: removed sections simply yield null and are
// guarded by the helpers below (setText, etc.). Core route IDs must exist.
const el = {
  apiStatus: document.getElementById("apiStatus"),
  displayRange: document.getElementById("displayRange"),
  sourceRegion: document.getElementById("sourceRegion"),
  runName: document.getElementById("runName"),
  modelNote: document.getElementById("modelNote"),
  cellCount: document.getElementById("cellCount"),
  avgScore: document.getElementById("avgScore"),
  peakScore: document.getElementById("peakScore"),
  routeForm: document.getElementById("routeForm"),
  startInput: document.getElementById("startInput"),
  endInput: document.getElementById("endInput"),
  startSuggestions: document.getElementById("startSuggestions"),
  endSuggestions: document.getElementById("endSuggestions"),
  scenicWeight: document.getElementById("scenicWeight"),
  weightOut: document.getElementById("weightOut"),
  detourFactor: document.getElementById("detourFactor"),
  detourOut: document.getElementById("detourOut"),
  submitRoute: document.getElementById("submitRoute"),
  routeTitle: document.getElementById("routeTitle"),
  routeOutput: document.getElementById("routeOutput"),
  trainingResults: document.getElementById("trainingResults"),
  clearRoute: document.getElementById("clearRoute"),
  clearStartBtn: document.getElementById("clearStartBtn"),
  clearEndBtn: document.getElementById("clearEndBtn"),
  apiBase: document.getElementById("apiBase"),
  inspectorScore: document.getElementById("inspectorScore"),
  inspectorCoords: document.getElementById("inspectorCoords"),
  routeResultsDialog: document.getElementById("routeResultsDialog"),
  scenicDistance: document.getElementById("scenicDistance"),
  scenicDuration: document.getElementById("scenicDuration"),
  scenicScore: document.getElementById("scenicScore"),
  scenicDistanceDelta: document.getElementById("scenicDistanceDelta"),
  scenicDurationDelta: document.getElementById("scenicDurationDelta"),
  scenicScoreDelta: document.getElementById("scenicScoreDelta"),
  baselineDistance: document.getElementById("baselineDistance"),
  baselineDuration: document.getElementById("baselineDuration"),
  baselineScore: document.getElementById("baselineScore"),
  shareRouteBtn: document.getElementById("shareRouteBtn"),
  closeRouteDialogBtn: document.getElementById("closeRouteDialogBtn"),
};

let map;
let latestHeatmap = null;
const selectedRoutePoints = { start: null, end: null };

function api(path) {
  return `${apiBase.replace(/\/+$/, "")}${path}`;
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
  if (map.getLayer(id)) map.removeLayer(id);
}

function removeSource(id) {
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
  const camera = map.cameraForBounds(REGION_BOUNDS, {
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
    if (
      payload.tile_zoom !== 14 ||
      !bounds ||
      !["min_lon", "min_lat", "max_lon", "max_lat"].every(
        (key) => Number.isFinite(Number(bounds[key]))
      )
    ) {
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

function renderRoute(geojson) {
  removeLayer(ROUTE_SCENIC);
  removeLayer(ROUTE_BASELINE);
  removeSource(ROUTE_SOURCE);

  map.addSource(ROUTE_SOURCE, { type: "geojson", data: geojson });
  map.addLayer(routeLayer("baseline", "#ffffff", 4, 0.62));
  map.addLayer(routeLayer("scenic", "#62c58a", 5.5, 0.96));
  fitToGeojson(geojson, { maxZoom: 12 });
}

function clearRoute() {
  removeLayer(ROUTE_SCENIC);
  removeLayer(ROUTE_BASELINE);
  removeSource(ROUTE_SOURCE);
  setRouteOutput("Waiting for submit", "Enter start/end coordinates as <code>lat, lon</code>, then submit.");
}

function signedNumber(value, digits, suffix = "") {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(digits)}${suffix}`;
}

function renderRouteComparison(payload) {
  const scenic = payload.routes?.scenic;
  const baseline = payload.routes?.baseline;
  const deltas = payload.deltas;
  if (!scenic || !baseline || deltas == null) {
    throw new Error("API returned incomplete route comparison");
  }
  setText(el.scenicDistance, `${formatNumber(scenic.total_distance_km, 1)} km`);
  setText(el.scenicDuration, `${formatNumber(scenic.estimated_duration_minutes, 0)} min`);
  setText(el.scenicScore, `${formatNumber(scenic.average_scenic_score, 2)} / 10`);
  setText(el.baselineDistance, `${formatNumber(baseline.total_distance_km, 1)} km`);
  setText(el.baselineDuration, `${formatNumber(baseline.estimated_duration_minutes, 0)} min`);
  setText(el.baselineScore, `${formatNumber(baseline.average_scenic_score, 2)} / 10`);
  setText(el.scenicDistanceDelta, signedNumber(deltas.distance_km, 1, " km"));
  setText(el.scenicDurationDelta, signedNumber(deltas.duration_min, 0, " min"));
  setText(el.scenicScoreDelta, signedNumber(deltas.scenic_score, 2));
  el.routeResultsDialog.showModal();
  el.closeRouteDialogBtn.focus();
}


async function fetchValidatedRoute() {
  const url = new URL(api("/v1/validated-route"));
  url.searchParams.set("region", CONFIG.sourceRegion);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const result = await response.json();
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
  let request = null;

  const closeSuggestions = () => {
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
    request?.abort();
    request = new AbortController();
    const url = new URL(api("/v1/search/retrieve"));
    url.searchParams.set("mapbox_id", suggestion.mapbox_id);
    url.searchParams.set("session_token", sessionToken);
    renderStatus("Searching…", "loading");
    try {
      const response = await fetch(url, { signal: request.signal });
      if (!response.ok) throw new Error(String(response.status));
      const result = (await response.json()).result;
      if (!Number.isFinite(result?.lat) || !Number.isFinite(result?.lon)) {
        throw new Error("invalid address result");
      }
      selectedRoutePoints[which] = { lat: result.lat, lon: result.lon };
      input.value = result.full_address || result.name || suggestion.full_address;
      sessionToken = newSearchSessionToken();
      closeSuggestions();
    } catch (error) {
      if (error.name !== "AbortError") {
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
    request?.abort();
    const query = input.value.trim();
    if (parsePoint(query) || query.length < 3) {
      closeSuggestions();
      return;
    }
    timer = setTimeout(async () => {
      request = new AbortController();
      const url = new URL(api("/v1/search/suggest"));
      url.searchParams.set("q", query);
      url.searchParams.set("session_token", sessionToken);
      url.searchParams.set("region", CONFIG.sourceRegion);
      renderStatus("Searching…", "loading");
      try {
        const response = await fetch(url, { signal: request.signal });
        if (!response.ok) throw new Error(String(response.status));
        renderSuggestions((await response.json()).suggestions || []);
      } catch (error) {
        if (error.name !== "AbortError") {
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
    request?.abort();
    input.value = "";
    selectedRoutePoints[which] = null;
    sessionToken = newSearchSessionToken();
    closeSuggestions();
    input.focus();
  });
}


async function planRoute(event) {
  event.preventDefault();
  const start = parsePoint(el.startInput.value) || selectedRoutePoints.start;
  const end = parsePoint(el.endInput.value) || selectedRoutePoints.end;
  if (!start || !end) {
    setRouteOutput("Invalid input", "Enter coordinates or choose an address suggestion.");
    (!start ? el.startInput : el.endInput).focus();
    return;
  }

  if (el.submitRoute) {
    el.submitRoute.disabled = true;
    el.submitRoute.textContent = "Planning...";
  }
  setRouteOutput("Computing route", "Calling <code>/v1/route/compare</code> with the same run used by the heatmap.");

  try {
    const response = await fetch(api("/v1/route/compare"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        region: CONFIG.sourceRegion,
        run_name: CONFIG.workingRun,
        start,
        end,
        scenic_weight: Number(el.scenicWeight.value),
        max_detour_factor: Number(el.detourFactor.value),
        avoid_highways: false,
        include_baseline: true,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.detail?.message || payload?.detail || `${response.status} ${response.statusText}`);
    }
    renderRoute(payload.geojson);
    renderRouteComparison(payload);
    setRouteOutput("Route computed", "Scenic and baseline routes are shown on the map.");
    setApiStatus("API: online", "ok");
  } catch (error) {
    setApiStatus("API: route failed", "warn");
    setRouteOutput("Route failed", escapeHtml(error.message || error));
  } finally {
    if (el.submitRoute) {
      el.submitRoute.disabled = false;
      el.submitRoute.textContent = "Plan route";
    }
  }
}

function buildShareUrl({ start, end, scenicWeight, maxDetour }) {
  const url = new URL(window.location.pathname, window.location.origin);
  url.searchParams.set("start", `${start.lat},${start.lon}`);
  url.searchParams.set("end", `${end.lat},${end.lon}`);
  url.searchParams.set("scenic_weight", String(scenicWeight));
  url.searchParams.set("max_detour", String(maxDetour));
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
  applyRange("scenic_weight", el.scenicWeight);
  applyRange("max_detour", el.detourFactor);
}

async function copyShareUrl() {
  const start = parsePoint(el.startInput.value) || selectedRoutePoints.start;
  const end = parsePoint(el.endInput.value) || selectedRoutePoints.end;
  if (!start || !end) return;
  const shareUrl = buildShareUrl({
    start,
    end,
    scenicWeight: Number(el.scenicWeight.value),
    maxDetour: Number(el.detourFactor.value),
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
  el.submitRoute.disabled = true;
  setText(el.displayRange, CONFIG.displayRange);
  setText(el.sourceRegion, CONFIG.sourceRegion);
  setText(el.runName, CONFIG.workingRun);
  if (el.apiBase) el.apiBase.value = apiBase;
  if (el.modelNote) {
    el.modelNote.textContent = `Scoring provenance: ${CONFIG.workingRun} at z14 using ${CONFIG.sourceModel}.`;
  }
  const syncRangeOutputs = () => {
    setText(el.weightOut, Number(el.scenicWeight.value).toFixed(2));
    setText(el.detourOut, `${Number(el.detourFactor.value).toFixed(2)}×`);
  };
  el.scenicWeight?.addEventListener("input", syncRangeOutputs);
  el.detourFactor?.addEventListener("input", syncRangeOutputs);
  syncRangeOutputs();
  el.routeForm?.addEventListener("submit", planRoute);
  installAddressSearch("start");
  installAddressSearch("end");
  el.clearRoute?.addEventListener("click", clearRoute);
  el.closeRouteDialogBtn?.addEventListener("click", () => el.routeResultsDialog.close());
  el.shareRouteBtn?.addEventListener("click", copyShareUrl);
  el.routeResultsDialog?.addEventListener("click", (event) => {
    if (event.target === el.routeResultsDialog) el.routeResultsDialog.close();
  });
  el.routeResultsDialog?.addEventListener("close", () => el.submitRoute.focus());
}

async function main() {
  initBindings();
  void loadTrainingResults();
  await checkApiHealth();

  let heatmap = null;
  let validatedRoute = null;
  const artifactErrors = [];
  try {
    heatmap = await loadHeatmap();
  } catch (error) {
    artifactErrors.push(error.message || String(error));
  }
  try {
    validatedRoute = await fetchValidatedRoute();
  } catch (error) {
    artifactErrors.push(`Validated route unavailable: ${error.message || error}`);
  }

  const style = cartoVoyagerStyle();

  if (validatedRoute) {
    style.sources[ROUTE_SOURCE] = { type: "geojson", data: validatedRoute.geojson };
    style.layers.push(
      routeLayer("baseline", "#ffffff", 4, 0.62),
      routeLayer("scenic", "#62c58a", 5.5, 0.96)
    );
  }

  map = new maplibregl.Map({
    container: "map",
    style,
    center: DEFAULTS.center,
    zoom: DEFAULTS.zoom,
    attributionControl: true,
    dragRotate: false,
    pitchWithRotate: false,
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
  map.on("zoomstart", handleZoomStart);
  map.on("zoomend", handleZoomEnd);
  map.on("load", () => {
    syncZoomElasticity();
    el.submitRoute.disabled = false;
  });
  map.on("resize", syncZoomElasticity);
  if (heatmap) {
    map.once("load", () => {
      map.addSource(HEATMAP_SOURCE, {
        type: "image",
        url: heatmap.imageUrl,
        coordinates: [
          [heatmap.bounds.min_lon, heatmap.bounds.max_lat],
          [heatmap.bounds.max_lon, heatmap.bounds.max_lat],
          [heatmap.bounds.max_lon, heatmap.bounds.min_lat],
          [heatmap.bounds.min_lon, heatmap.bounds.min_lat],
        ],
      });
      map.addLayer(
        {
          id: HEATMAP_FILL,
          type: "raster",
          source: HEATMAP_SOURCE,
          paint: {
            "raster-opacity": 0.78,
            "raster-resampling": "nearest",
          },
        },
        validatedRoute ? ROUTE_BASELINE : undefined
      );
    });

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
  if (validatedRoute) {
    setRouteOutput("Burlington → Bangor", "Validated scenic and baseline routes are shown on the map.");
  }
  if (artifactErrors.length) {
    setApiStatus("API: artifacts unavailable", "warn");
    if (!validatedRoute) {
      setRouteOutput("Canonical artifacts unavailable", escapeHtml(artifactErrors.join(" ")));
    }
  }
}

main().catch((error) => {
  setApiStatus("Viewer failed", "warn");
  setRouteOutput("Viewer failed", escapeHtml(error.message || error));
});
