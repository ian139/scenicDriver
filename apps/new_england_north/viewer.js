const DEFAULTS = Object.freeze({
  displayRange: "new_england_north",
  sourceRegion: "masswhites",
  workingRun: "masswhites_z14_learned_h4_v2",
  sourceModel: "models/scenic_regression_baseline_masswhites_z14_mixed5000_v2_weighted_h4.pt",
  activeRegistryModel: "models/scenic_regression_baseline_masswhites_z14_mixed5000_v6_vast_weighted_h4.pt",
  apiBase: "http://localhost:8080",
  center: [-73.22, 42.85],
  zoom: 8.2,
});

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
let apiBase = params.get("api") || DEFAULTS.apiBase;

const STATIC_HEATMAP_MODULE = "./data/new_england_north_heatmap_cells.js";
const HEATMAP_SOURCE = "scenic-heatmap-source";
const HEATMAP_FILL = "scenic-heatmap-fill";
const HEATMAP_LINE = "scenic-heatmap-line";
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
  scenicWeight: document.getElementById("scenicWeight"),
  weightOut: document.getElementById("weightOut"),
  detourFactor: document.getElementById("detourFactor"),
  detourOut: document.getElementById("detourOut"),
  submitRoute: document.getElementById("submitRoute"),
  routeTitle: document.getElementById("routeTitle"),
  routeOutput: document.getElementById("routeOutput"),
  clearRoute: document.getElementById("clearRoute"),
  scaleTab: document.getElementById("scaleTab"),
  scaleToggle: document.getElementById("scaleToggle"),
  scalePanel: document.getElementById("scalePanel"),
  apiBase: document.getElementById("apiBase"),
  inspectorScore: document.getElementById("inspectorScore"),
  inspectorCoords: document.getElementById("inspectorCoords"),
};

let map;
let latestHeatmap = null;

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

function parsePoint(value) {
  const match = String(value || "").trim().match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (!match) return null;
  const lat = Number(match[1]);
  const lon = Number(match[2]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

function scenicColorExpression() {
  return [
    "interpolate",
    ["linear"],
    ["get", "score_norm"],
    0,
    "#2b685f",
    0.35,
    "#87a65d",
    0.6,
    "#e2b75b",
    0.78,
    "#d77642",
    1,
    "#913633",
  ];
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

function normalizeScore(rawScore, rawNorm) {
  const norm = Number(rawNorm);
  if (Number.isFinite(norm)) return Math.max(0, Math.min(1, norm));
  const score = Number(rawScore);
  if (!Number.isFinite(score)) return 0;
  return Math.max(0, Math.min(1, score <= 1 ? score : score / 10));
}

function normalizeHeatmapFeatureCollection(collection) {
  const features = Array.isArray(collection?.features) ? collection.features : [];
  return {
    type: "FeatureCollection",
    features: features
      .filter((feature) => feature?.geometry)
      .map((feature) => {
        const score = Number(feature.properties?.score);
        const scoreNorm = normalizeScore(score, feature.properties?.score_norm);
        return {
          ...feature,
          properties: {
            ...(feature.properties || {}),
            score: Number.isFinite(score) ? score : scoreNorm * 10,
            score_norm: scoreNorm,
          },
        };
      }),
  };
}

function cellsToFeatureCollection(cells) {
  return {
    type: "FeatureCollection",
    features: cells.map(([west, south, east, north, score], index) => {
      const scoreNorm = normalizeScore(score, score);
      return {
        type: "Feature",
        properties: { index, score: scoreNorm * 10, score_norm: scoreNorm },
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [west, south],
              [east, south],
              [east, north],
              [west, north],
              [west, south],
            ],
          ],
        },
      };
    }),
  };
}

function removeLayer(id) {
  if (map.getLayer(id)) map.removeLayer(id);
}

function removeSource(id) {
  if (map.getSource(id)) map.removeSource(id);
}

function renderHeatmap(geojson) {
  removeLayer(HEATMAP_LINE);
  removeLayer(HEATMAP_FILL);
  removeSource(HEATMAP_SOURCE);

  map.addSource(HEATMAP_SOURCE, { type: "geojson", data: geojson });
  map.addLayer({
    id: HEATMAP_FILL,
    type: "fill",
    source: HEATMAP_SOURCE,
    paint: {
      "fill-color": scenicColorExpression(),
      "fill-opacity": ["interpolate", ["linear"], ["get", "score_norm"], 0, 0.2, 0.55, 0.46, 1, 0.74],
    },
  });
  map.addLayer({
    id: HEATMAP_LINE,
    type: "line",
    source: HEATMAP_SOURCE,
    paint: {
      "line-color": "rgba(255,255,255,0.34)",
      "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0, 10, 0.35, 13, 0.8],
      "line-opacity": 0.42,
    },
  });

  updateHeatmapStats(geojson);
  bindHeatmapInspector();
  fitToGeojson(geojson, { maxZoom: 9.5 });
}

function updateHeatmapStats(geojson) {
  // Stats cards are removed from the minimal viewer; keep the computation
  // null-safe so re-adding the IDs later needs no JS change.
  if (!el.cellCount && !el.avgScore && !el.peakScore) return;
  const scores = geojson.features.map((feature) => Number(feature.properties?.score)).filter(Number.isFinite);
  const count = scores.length;
  const avg = count ? scores.reduce((sum, score) => sum + score, 0) / count : NaN;
  const peak = count ? Math.max(...scores) : NaN;
  setText(el.cellCount, count ? count.toLocaleString() : "--");
  setText(el.avgScore, Number.isFinite(avg) ? avg.toFixed(1) : "--");
  setText(el.peakScore, Number.isFinite(peak) ? peak.toFixed(1) : "--");
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
  map.fitBounds(bounds, {
    padding: { top: 58, right: 54, bottom: bottomPadding, left: 54 },
    duration: 600,
    maxZoom: options.maxZoom ?? 12,
  });
}

function bindHeatmapInspector() {
  if (map.__scenicInspectorBound) return;
  map.__scenicInspectorBound = true;
  map.on("mousemove", HEATMAP_FILL, (event) => {
    const feature = event.features?.[0];
    if (!feature) return;
    const score = Number(feature.properties?.score);
    const lngLat = event.lngLat;
    setText(el.inspectorScore, Number.isFinite(score) ? score.toFixed(2) : "--");
    setText(el.inspectorCoords, `${lngLat.lat.toFixed(4)}, ${lngLat.lng.toFixed(4)}`);
    map.getCanvas().style.cursor = "crosshair";
  });
  map.on("mouseleave", HEATMAP_FILL, () => {
    setText(el.inspectorScore, "--");
    setText(el.inspectorCoords, "Move over a heatmap cell");
    map.getCanvas().style.cursor = "";
  });
}

async function loadHeatmap() {
  const url = new URL(api("/v1/heatmap"));
  url.searchParams.set("region", CONFIG.sourceRegion);
  url.searchParams.set("run_name", CONFIG.workingRun);
  url.searchParams.set("max_points", "1");
  url.searchParams.set("max_tiles", "6000");

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const payload = await response.json();
    const geojson = normalizeHeatmapFeatureCollection(payload.geojson_tiles || payload.geojson);
    if (!geojson.features.length) throw new Error("API returned no heatmap features");
    latestHeatmap = { source: "api", geojson, runName: payload.run_name };
    setApiStatus("API: online", "ok");
    return latestHeatmap;
  } catch (error) {
    const module = await import(STATIC_HEATMAP_MODULE);
    const geojson = normalizeHeatmapFeatureCollection(cellsToFeatureCollection(module.default));
    latestHeatmap = { source: "static", geojson, error };
    setApiStatus("API: static fallback", "warn");
    return latestHeatmap;
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
  return {
    id: filterKind === "scenic" ? ROUTE_SCENIC : ROUTE_BASELINE,
    type: "line",
    source: ROUTE_SOURCE,
    filter: ["==", ["get", "route_kind"], filterKind],
    layout: { "line-join": "round", "line-cap": "round" },
    paint: {
      "line-color": color,
      "line-width": width,
      "line-opacity": opacity,
    },
  };
}

function renderRoute(geojson) {
  removeLayer(ROUTE_SCENIC);
  removeLayer(ROUTE_BASELINE);
  removeSource(ROUTE_SOURCE);

  map.addSource(ROUTE_SOURCE, { type: "geojson", data: geojson });
  map.addLayer(routeLayer("baseline", "#386f9f", 4, 0.62));
  map.addLayer(routeLayer("scenic", "#b9653d", 5.5, 0.92));
  fitToGeojson(geojson, { maxZoom: 12 });
}

function clearRoute() {
  removeLayer(ROUTE_SCENIC);
  removeLayer(ROUTE_BASELINE);
  removeSource(ROUTE_SOURCE);
  setRouteOutput("Waiting for submit", "Enter start/end coordinates as <code>lat, lon</code>, then submit.");
}

function metricsMarkup(result) {
  const scenic = result.routes?.scenic || {};
  const deltas = result.deltas || {};
  return `
    <div class="metric-list">
      <div><span>Scenic distance</span><b>${formatNumber(scenic.total_distance_km, 1)} km</b></div>
      <div><span>Estimated duration</span><b>${formatNumber(scenic.estimated_duration_minutes, 0)} min</b></div>
      <div><span>Average scenic score</span><b>${formatNumber(scenic.average_scenic_score, 2)} / 10</b></div>
      <div><span>Scenic score delta</span><b>${formatNumber(deltas.scenic_score, 2)}</b></div>
    </div>
  `;
}

async function planRoute(event) {
  event.preventDefault();
  const start = parsePoint(el.startInput.value);
  const end = parsePoint(el.endInput.value);
  if (!start || !end) {
    setRouteOutput("Invalid input", "Use decimal coordinates formatted as <code>lat, lon</code>.");
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
    setRouteOutput("Route computed", metricsMarkup(payload));
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

function initBindings() {
  // Optional provenance bindings (removed from the minimal viewer).
  setText(el.displayRange, CONFIG.displayRange);
  setText(el.sourceRegion, CONFIG.sourceRegion);
  setText(el.runName, CONFIG.workingRun);
  if (el.apiBase) el.apiBase.value = apiBase;
  if (el.modelNote) {
    el.modelNote.textContent = `Scoring provenance: ${CONFIG.workingRun} report (${CONFIG.sourceModel}) feeds both heatmap and route edge costs. Active registry candidate ${CONFIG.activeRegistryModel} was discovered, but no local v6-scored report is present in this checkout.`;
  }

  if (el.scenicWeight) {
    el.scenicWeight.addEventListener("input", () => {
      setText(el.weightOut, Number(el.scenicWeight.value).toFixed(2));
    });
  }
  if (el.detourFactor) {
    el.detourFactor.addEventListener("input", () => {
      setText(el.detourOut, `${Number(el.detourFactor.value).toFixed(2)}×`);
    });
  }
  if (el.routeForm) el.routeForm.addEventListener("submit", planRoute);
  if (el.clearRoute) el.clearRoute.addEventListener("click", clearRoute);
  if (el.scaleToggle && el.scaleTab) {
    el.scaleToggle.addEventListener("click", () => {
      const isClosed = el.scaleTab.classList.toggle("is-closed");
      el.scaleToggle.setAttribute("aria-expanded", String(!isClosed));
    });
  }
  // If an #apiBase input is ever re-added, keep it in sync with the runtime
  // variable and reload heatmap on change. Removed by default → no-op.
  if (el.apiBase) {
    el.apiBase.addEventListener("change", async () => {
      apiBase = el.apiBase.value || DEFAULTS.apiBase;
      await checkApiHealth();
      const heatmap = await loadHeatmap();
      renderHeatmap(heatmap.geojson);
    });
  }
}

async function main() {
  initBindings();
  void checkApiHealth();
  map = new maplibregl.Map({
    container: "map",
    style: cartoVoyagerStyle(),
    center: DEFAULTS.center,
    zoom: DEFAULTS.zoom,
    attributionControl: true,
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");

  map.on("load", async () => {
    await checkApiHealth();
    const heatmap = await loadHeatmap();
    renderHeatmap(heatmap.geojson);
    if (heatmap.source === "static") {
      setRouteOutput(
        "Heatmap fallback loaded",
        "Static Masswhites heatmap cells are visible. Start the FastAPI service on port 8080 before submitting routes."
      );
    }
  });
}

main().catch((error) => {
  setApiStatus("Viewer failed", "warn");
  setRouteOutput("Viewer failed", escapeHtml(error.message || error));
});
