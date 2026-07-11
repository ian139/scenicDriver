const el = {
  controls: document.querySelector(".controls"),
  apiBase: document.getElementById("apiBase"),
  basemapProvider: document.getElementById("basemapProvider"),
  basemapStatus: document.getElementById("basemapStatus"),
  region: document.getElementById("region"),
  modeAddressBtn: document.getElementById("modeAddressBtn"),
  modeCoordsBtn: document.getElementById("modeCoordsBtn"),
  prefFastBtn: document.getElementById("prefFastBtn"),
  prefBalancedBtn: document.getElementById("prefBalancedBtn"),
  prefScenicBtn: document.getElementById("prefScenicBtn"),
  mapStyleMapBtn: document.getElementById("mapStyleMapBtn"),
  mapStyleSatelliteBtn: document.getElementById("mapStyleSatelliteBtn"),
  refreshRegionsBtn: document.getElementById("refreshRegionsBtn"),
  apiHealth: document.getElementById("apiHealth"),
  startLat: document.getElementById("startLat"),
  startLon: document.getElementById("startLon"),
  endLat: document.getElementById("endLat"),
  endLon: document.getElementById("endLon"),
  startAddress: document.getElementById("startAddress"),
  endAddress: document.getElementById("endAddress"),
  startSuggestBox: document.getElementById("startSuggestBox"),
  endSuggestBox: document.getElementById("endSuggestBox"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  scenicCanvas: document.getElementById("scenicCanvas"),
  heatmapLegend: document.getElementById("heatmapLegend"),
  surfaceInspector: document.getElementById("surfaceInspector"),
  surfaceInspectorLabel: document.getElementById("surfaceInspectorLabel"),
  surfaceInspectorScore: document.getElementById("surfaceInspectorScore"),
  surfaceInspectorCoords: document.getElementById("surfaceInspectorCoords"),
  surfaceCells: document.getElementById("surfaceCells"),
  surfaceAverage: document.getElementById("surfaceAverage"),
  surfaceTop: document.getElementById("surfaceTop"),
  surfaceHighShare: document.getElementById("surfaceHighShare"),
  menuBtn: document.getElementById("menuBtn"),
  scenicWeight: document.getElementById("scenicWeight"),
  weightValue: document.getElementById("weightValue"),
  maxDetour: document.getElementById("maxDetour"),
  detourValue: document.getElementById("detourValue"),
  showHeatmap: document.getElementById("showHeatmap"),
  swapBtn: document.getElementById("swapBtn"),
  planBtn: document.getElementById("planBtn"),
  saveRouteBtn: document.getElementById("saveRouteBtn"),
  shareRouteBtn: document.getElementById("shareRouteBtn"),
  clearBtn: document.getElementById("clearBtn"),
  status: document.getElementById("status"),
  savedTrips: document.getElementById("savedTrips"),
  routeList: document.getElementById("routeList"),
  trainingResults: document.getElementById("trainingResults"),
  routePanel: document.getElementById("routePanel"),
  routeCollapseBtn: document.getElementById("routeCollapseBtn"),
  sheetToggleBtn: document.getElementById("sheetToggleBtn"),
};

let map;
let lastRouteGeojson = null;
let activeRouteKind = null;
let lastPlan = null;
const routeSourceId = "route-data";
const heatmapSourceId = "scenic-heatmap";
const heatmapLayerId = "scenic-heatmap-layer";
const heatmapSoftSourceId = "scenic-heatmap-soft";
const heatmapSoftLayerId = "scenic-heatmap-soft-layer";
const heatmapMaskSourceId = "scenic-surface-mask";
const heatmapMaskLayerId = "scenic-surface-mask-layer";
const heatmapBoundsSourceId = "scenic-surface-bounds";
const heatmapBoundsLayerId = "scenic-surface-bounds-layer";
const heatmapTopSourceId = "scenic-surface-top";
const heatmapTopFillLayerId = "scenic-surface-top-fill";
const heatmapTopLineLayerId = "scenic-surface-top-line";
const heatmapTilesSourceId = "scenic-heatmap-tiles";
const heatmapTilesLayerId = "scenic-heatmap-tiles-layer";
let heatmapZoomMode = "z16";
let heatmapGeojson = null;
let heatmapInspectorBound = false;
let heatmapLoadPromise = null;
let surfaceScoreBreaks = { moderate: 0.38, high: 0.68, peak: 0.85 };
let regionsMeta = [];
let busy = false;
let inputMode = "address";
let routePreference = "balanced";
let basemapApplySeq = 0;
let planSeq = 0;
let pendingRouteGeojson = null;
let pendingRouteRenderOptions = {};
let lastRoutePayload = null;
const suggestTimer = { start: null, end: null };
const suggestState = {
  start: { items: [], active: -1, selected: null },
  end: { items: [], active: -1, selected: null },
};
const suggestSeq = { start: 0, end: 0 };

const STATIC_HEATMAP_MODULE = "./data/masswhites_heatmap_cells.js?v=masswhites-v4-cells";
const DEMO_HEATMAP_BOUNDS = {
  minLat: 41.88,
  minLon: -73.72,
  maxLat: 44.2,
  maxLon: -72.46,
};
const STORAGE_KEYS = {
  basemapProvider: "scenicdrive.basemapProvider",
  savedTrips: "scenicdrive.savedTrips",
  routePanelCollapsed: "scenicdrive.routePanelCollapsed",
  activeTab: "scenicdrive.activeTab",
  settingsSheetOpen: "scenicdrive.settingsSheetOpen",
  routePreference: "scenicdrive.routePreference",
};

const ROUTE_PRESETS = {
  fast: { scenic_weight: 0.15, max_detour_factor: 1.25, label: "Fastest" },
  balanced: { scenic_weight: 0.65, max_detour_factor: 1.65, label: "Balanced" },
  scenic: { scenic_weight: 0.9, max_detour_factor: 2.4, label: "Scenic" },
};

const LOCAL_PLACES = [
  { label: "Pittsfield, Massachusetts", lat: 42.4501, lon: -73.2454 },
  { label: "Great Barrington, Massachusetts", lat: 42.1959, lon: -73.3629 },
  { label: "North Adams, Massachusetts", lat: 42.7009, lon: -73.1087 },
  { label: "Williamstown, Massachusetts", lat: 42.712, lon: -73.2037 },
  { label: "Lenox, Massachusetts", lat: 42.3565, lon: -73.2848 },
  { label: "Adams, Massachusetts", lat: 42.6243, lon: -73.1176 },
  { label: "Bennington, Vermont", lat: 42.8781, lon: -73.1968 },
  { label: "Greenfield, Massachusetts", lat: 42.5876, lon: -72.5995 },
];

function requireElement(name, node) {
  if (!node) throw new Error(`Missing required UI element: ${name}`);
}

function setStatus(msg) {
  if (!el.status) return;
  el.status.textContent = String(msg || "").replace(/\n+/g, " • ");
  el.status.classList.add("hidden");
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showRouteMessage(msg, className = "route-empty route-error") {
  el.routeList.innerHTML = `<div class="${className}">${escapeHtml(msg)}</div>`;
}

function setBusy(next) {
  busy = Boolean(next);
  el.planBtn.disabled = busy;
  el.planBtn.textContent = busy ? "Comparing..." : "Compare Route";
  el.loadingOverlay.classList.toggle("hidden", !busy);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function toggleMenu() {
  if (isMobileLayout()) {
    setSheetOpen(true);
    return;
  }
  el.controls?.classList.toggle("menu-collapsed");
}

function api(path) {
  return `${el.apiBase.value.replace(/\/+$/, "")}${path}`;
}

function normalizePlaceText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\b(ma|mass|massachusetts|vt|vermont|usa|united states)\b/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function localGeocode(query) {
  const q = normalizePlaceText(query);
  if (!q) return [];
  return LOCAL_PLACES.filter((place) => normalizePlaceText(place.label).includes(q) || q.includes(normalizePlaceText(place.label)))
    .slice(0, 6)
    .map((place) => ({
      label: place.label,
      match_type: "local_masswhites",
      lat: place.lat,
      lon: place.lon,
      latlon: `${place.lat.toFixed(6)},${place.lon.toFixed(6)}`,
      wkt: `POINT(${place.lon.toFixed(6)} ${place.lat.toFixed(6)})`,
    }));
}

function parseCoordinateInput(value) {
  const text = String(value || "").trim();
  const match = text.match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (!match) return null;
  const first = Number(match[1]);
  const second = Number(match[2]);
  if (!Number.isFinite(first) || !Number.isFinite(second)) return null;
  const candidates = [
    { lat: first, lon: second },
    { lat: second, lon: first },
  ].filter((p) => Math.abs(p.lat) <= 90 && Math.abs(p.lon) <= 180);
  const key = String(el.region?.value || "").toLowerCase();
  const region = regionsMeta.find((r) => String(r.region || "").toLowerCase() === key);
  const bbox = region?.bbox;
  if (bbox) {
    const minLat = Number(bbox.min_lat);
    const minLon = Number(bbox.min_lon);
    const maxLat = Number(bbox.max_lat);
    const maxLon = Number(bbox.max_lon);
    const inRegion = candidates.find((p) => p.lat >= minLat && p.lat <= maxLat && p.lon >= minLon && p.lon <= maxLon);
    if (inRegion) return inRegion;
  }
  return candidates[0] || null;
}

function coordinateRow(value) {
  const point = parseCoordinateInput(value);
  if (!point) return null;
  return {
    label: `${point.lat.toFixed(6)}, ${point.lon.toFixed(6)}`,
    match_type: "coordinate",
    lat: point.lat,
    lon: point.lon,
    latlon: `${point.lat.toFixed(6)},${point.lon.toFixed(6)}`,
    wkt: `POINT(${point.lon.toFixed(6)} ${point.lat.toFixed(6)})`,
  };
}

function syncInputMode() {
  const addr = inputMode === "address";
  document.querySelectorAll(".coords-only").forEach((n) => n.classList.toggle("hidden", addr));
  document.querySelectorAll(".addr-only").forEach((n) => n.classList.toggle("hidden", !addr));
  el.modeAddressBtn.classList.toggle("active", addr);
  el.modeAddressBtn.classList.toggle("secondary", !addr);
  el.modeCoordsBtn.classList.toggle("active", !addr);
  el.modeCoordsBtn.classList.toggle("secondary", addr);
}

function normalizeBasemapProvider(provider) {
  const p = String(provider || "").toLowerCase();
  if (p === "satellite") return "satellite";
  return "carto_voyager";
}

function basemapTileSource(provider) {
  if (provider === "satellite") {
    return {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: "Tiles © Esri",
    };
  }
  if (provider === "carto_voyager") {
    return {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: "© CARTO © OpenStreetMap contributors",
    };
  }
  return {
    type: "raster",
    tiles: [
      "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
      "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
      "https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    ],
    tileSize: 256,
    attribution: "© CARTO © OpenStreetMap contributors",
  };
}

function buildBasemapStyle(provider) {
  const source = basemapTileSource(provider);
  return {
    version: 8,
    sources: {
      basemap: source,
    },
    layers: [{ id: "basemap", type: "raster", source: "basemap" }],
  };
}

function applyBasemapStatus(provider, hasKey) {
  const names = {
    carto_voyager: "Map",
    satellite: "Satellite",
  };
  const name = names[provider] || "Map";
  if (el.basemapStatus) el.basemapStatus.textContent = `Basemap: ${name}`;
  [el.mapStyleMapBtn, el.mapStyleSatelliteBtn].forEach((btn) => {
    if (!btn) return;
    const active = btn.dataset.basemap === provider;
    btn.classList.toggle("active", active);
    btn.classList.toggle("secondary", !active);
  });
}

function applyBasemap() {
  if (!map) return;
  const seq = ++basemapApplySeq;
  const provider = normalizeBasemapProvider(el.basemapProvider.value);
  applyBasemapStatus(provider);
  if (provider !== el.basemapProvider.value) {
    el.basemapProvider.value = provider;
  }
  localStorage.setItem(STORAGE_KEYS.basemapProvider, provider);
  const center = map.getCenter();
  const zoom = map.getZoom();
  const bearing = map.getBearing();
  const pitch = map.getPitch();
  map.setStyle(buildBasemapStyle(provider), { diff: false });
  map.once("style.load", () => {
    if (seq !== basemapApplySeq) return;
    if (center) map.jumpTo({ center, zoom, bearing, pitch });
    if (el.showHeatmap.checked) {
      const heatmapReady = heatmapGeojson ? Promise.resolve(renderMapHeatmapLayer(heatmapGeojson, { fit: false })) : loadHeatmap();
      heatmapReady
        .catch(() => showRouteMessage("Heatmap unavailable for this region."))
        .finally(() => {
          if (lastRouteGeojson) scheduleRouteRender(lastRouteGeojson, { fit: false });
        });
    } else if (lastRouteGeojson) {
      scheduleRouteRender(lastRouteGeojson, { fit: false });
    }
  });
}

async function checkHealth() {
  el.apiHealth.textContent = "API: checking...";
  try {
    const resp = await fetch(api("/v1/healthz"));
    if (!resp.ok) throw new Error(String(resp.status));
    const payload = await resp.json();
    el.apiHealth.textContent = `API: healthy • regions=${Number(payload.regions_available || 0)}`;
  } catch {
    el.apiHealth.textContent = "API: unavailable";
  }
}

async function loadRegions() {
  const resp = await fetch(api("/v1/regions"));
  if (!resp.ok) throw new Error(`regions failed: ${resp.status}`);
  const data = await resp.json();
  const regions = data.regions || [];
  regionsMeta = regions;
  el.region.innerHTML = "";
  for (const r of regions) {
    const opt = document.createElement("option");
    opt.value = r.region;
    opt.textContent = r.display_name || r.region;
    el.region.appendChild(opt);
  }
  const defaultRegion = regions.find((r) => r.is_default) || regions.find((r) => r.region === "masswhites");
  if (defaultRegion) el.region.value = defaultRegion.region;
}

function renderTrainingResults(payload) {
  const metrics = payload?.metrics;
  if (
    typeof payload?.run_name !== "string" ||
    typeof payload?.checkpoint !== "string" ||
    typeof payload?.updated_at !== "string" ||
    !metrics ||
    !["corr", "mae", "rmse", "samples"].every((key) => Number.isFinite(Number(metrics[key])))
  ) {
    throw new Error("invalid training result");
  }

  const card = document.createElement("article");
  card.className = "route-card";
  const title = document.createElement("h3");
  title.textContent = payload.run_name;
  const timestamp = document.createElement("div");
  timestamp.className = "route-sub";
  timestamp.textContent = payload.updated_at;
  const meta = document.createElement("div");
  meta.className = "meta";
  const values = [
    ["Correlation", Number(metrics.corr).toFixed(3)],
    ["MAE", Number(metrics.mae).toFixed(3)],
    ["RMSE", Number(metrics.rmse).toFixed(3)],
    ["Samples", String(Math.trunc(Number(metrics.samples)))],
  ];
  for (const [labelText, valueText] of values) {
    const item = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = labelText;
    const value = document.createElement("span");
    value.textContent = valueText;
    item.append(label, value);
    meta.appendChild(item);
  }
  const checkpoint = document.createElement("div");
  checkpoint.className = "route-sub";
  checkpoint.textContent = `Checkpoint: ${payload.checkpoint.split(/[\\/]/).pop()}`;
  card.append(title, timestamp, meta, checkpoint);
  el.trainingResults.replaceChildren(card);
}

async function loadTrainingResults() {
  try {
    const resp = await fetch(api("/v1/training-results"));
    if (!resp.ok) throw new Error(String(resp.status));
    renderTrainingResults(await resp.json());
  } catch {
    const unavailable = document.createElement("div");
    unavailable.className = "route-empty route-error";
    unavailable.textContent = "Remote training results are unavailable.";
    el.trainingResults.replaceChildren(unavailable);
  }
}

function loadFallbackRegions() {
  const fallback = {
    region: "masswhites",
    display_name: "Masswhites",
    is_default: true,
    bbox: {
      min_lat: 41.19518982948958,
      min_lon: -73.5205078125,
      max_lat: 44.512176171071054,
      max_lon: -72.97119140625,
    },
    map: { center: { lat: 42.85, lon: -73.22 }, zoom: 8.2 },
  };
  regionsMeta = [fallback];
  el.region.innerHTML = "";
  const opt = document.createElement("option");
  opt.value = fallback.region;
  opt.textContent = fallback.display_name;
  el.region.appendChild(opt);
  el.region.value = fallback.region;
}

function applyRegionDefaults() {
  const key = String(el.region.value || "").toLowerCase();
  const region = regionsMeta.find((r) => String(r.region || "").toLowerCase() === key);
  const bbox = region?.bbox;
  let startLat = 40.0;
  let startLon = -75.0;
  let endLat = 40.01;
  let endLon = -74.99;
  if (bbox) {
    const minLat = Number(bbox.min_lat);
    const minLon = Number(bbox.min_lon);
    const maxLat = Number(bbox.max_lat);
    const maxLon = Number(bbox.max_lon);
    startLat = minLat + (maxLat - minLat) * 0.25;
    startLon = minLon + (maxLon - minLon) * 0.25;
    endLat = minLat + (maxLat - minLat) * 0.75;
    endLon = minLon + (maxLon - minLon) * 0.75;
  }
  if (region?.map?.center) {
    const center = region.map.center;
    if (map) {
      map.easeTo({
        center: [Number(center.lon), Number(center.lat)],
        zoom: Number(region.map.zoom || map.getZoom()),
        duration: 0,
      });
    }
  }
  el.startLat.value = startLat.toFixed(6);
  el.startLon.value = startLon.toFixed(6);
  el.endLat.value = endLat.toFixed(6);
  el.endLon.value = endLon.toFixed(6);
  const looksLikePoint = (v) => /^-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?$/.test(String(v || "").trim());
  if (!el.startAddress.value || looksLikePoint(el.startAddress.value)) {
    el.startAddress.value = `${el.startLat.value},${el.startLon.value}`;
  }
  if (!el.endAddress.value || looksLikePoint(el.endAddress.value)) {
    el.endAddress.value = `${el.endLat.value},${el.endLon.value}`;
  }
  suggestState.start.selected = null;
  suggestState.end.selected = null;
}

function applyRoutePreference(pref) {
  routePreference = ROUTE_PRESETS[pref] ? pref : "balanced";
  const preset = ROUTE_PRESETS[routePreference];
  el.scenicWeight.value = String(preset.scenic_weight);
  el.maxDetour.value = String(preset.max_detour_factor);
  el.weightValue.textContent = String(preset.scenic_weight);
  el.detourValue.textContent = String(preset.max_detour_factor);
  document.querySelectorAll("[data-pref]").forEach((btn) => {
    const active = btn.dataset.pref === routePreference;
    btn.classList.toggle("active", active);
    btn.classList.toggle("secondary", !active);
  });
  localStorage.setItem(STORAGE_KEYS.routePreference, routePreference);
}

function initMap() {
  let storedProvider = localStorage.getItem(STORAGE_KEYS.basemapProvider) || "";
  if (!storedProvider) {
    storedProvider = "carto_voyager";
    localStorage.setItem(STORAGE_KEYS.basemapProvider, storedProvider);
  }
  const provider = normalizeBasemapProvider(storedProvider);
  el.basemapProvider.value = provider;
  applyBasemapStatus(provider);
  map = new maplibregl.Map({
    container: "map",
    style: buildBasemapStyle(provider),
    center: [-75.2, 40.045],
    zoom: 12,
  });
}

function isMapStyleEditable() {
  try {
    return Boolean(map?.getStyle()?.layers?.length);
  } catch {
    return false;
  }
}

function whenMapStyleReady(callback) {
  if (!map) return;
  if (isMapStyleEditable()) {
    callback();
    return;
  }
  let done = false;
  const run = () => {
    if (done || !isMapStyleEditable()) return;
    done = true;
    callback();
  };
  map.once("load", run);
  map.once("style.load", run);
  map.once("idle", run);
}

async function geocodeMany(query) {
  const coords = coordinateRow(query);
  if (coords) return [coords];
  const localRows = localGeocode(query);
  if (localRows.length && String(el.region.value || "").toLowerCase() === "masswhites") {
    return localRows;
  }
  const resp = await fetch(
    `${api("/v1/geocode")}?q=${encodeURIComponent(query)}&region=${encodeURIComponent(el.region.value)}`
  ).catch(() => null);
  if (!resp) return localRows;
  if (!resp.ok) {
    return localRows;
  }
  const payload = await resp.json();
  const remoteRows = payload.results || [];
  return remoteRows.length ? remoteRows : localRows;
}

async function geocodeOne(query) {
  const rows = await geocodeMany(query);
  if (!rows.length) throw new Error(`No match found for "${query}"`);
  return rows[0];
}

function hideSuggestions(which) {
  const box = which === "start" ? el.startSuggestBox : el.endSuggestBox;
  box.innerHTML = "";
  box.classList.add("hidden");
}

function chooseSuggestion(which, row) {
  const isStart = which === "start";
  const input = isStart ? el.startAddress : el.endAddress;
  input.value = row.label || row.latlon || `${row.lat},${row.lon}`;
  suggestState[which].selected = row;
  suggestState[which].active = -1;
  hideSuggestions(which);
}

function renderSuggestionBox(which) {
  const box = which === "start" ? el.startSuggestBox : el.endSuggestBox;
  const state = suggestState[which];
  if (!state.items.length) {
    hideSuggestions(which);
    return;
  }
  box.innerHTML = state.items
    .map((item, idx) => {
      const cls = idx === state.active ? "suggest-item active" : "suggest-item";
      const text = item.label || item.latlon || `${item.lat},${item.lon}`;
      return `<div class="${cls}" data-idx="${idx}">${text}</div>`;
    })
    .join("");
  box.classList.remove("hidden");
  box.querySelectorAll(".suggest-item").forEach((node) => {
    node.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      const idx = Number(node.dataset.idx || 0);
      chooseSuggestion(which, state.items[idx]);
    });
  });
}

function setSuggestLoading(which) {
  const box = which === "start" ? el.startSuggestBox : el.endSuggestBox;
  box.innerHTML = `<div class="suggest-item">Searching...</div>`;
  box.classList.remove("hidden");
}

function setSuggestEmpty(which) {
  const box = which === "start" ? el.startSuggestBox : el.endSuggestBox;
  box.innerHTML = `<div class="suggest-item">No matches</div>`;
  box.classList.remove("hidden");
}

function onAddressInput(which) {
  const isStart = which === "start";
  const input = isStart ? el.startAddress : el.endAddress;
  suggestState[which].selected = null;
  const query = String(input.value || "").trim();
  if (suggestTimer[which]) clearTimeout(suggestTimer[which]);
  const seq = ++suggestSeq[which];
  suggestTimer[which] = setTimeout(async () => {
    if (query.length < 1) {
      suggestState[which].items = [];
      suggestState[which].active = -1;
      hideSuggestions(which);
      return;
    }
    const coords = coordinateRow(query);
    if (coords) {
      suggestState[which].items = [coords];
      suggestState[which].active = 0;
      renderSuggestionBox(which);
      return;
    }
    setSuggestLoading(which);
    let rows = [];
    try {
      rows = await geocodeMany(query);
    } catch {
      rows = [];
    }
    if (seq !== suggestSeq[which]) return;
    suggestState[which].items = rows.slice(0, 6);
    suggestState[which].active = rows.length ? 0 : -1;
    if (!rows.length) {
      setSuggestEmpty(which);
    } else {
      renderSuggestionBox(which);
    }
  }, 80);
}

function onAddressKey(which, ev) {
  const state = suggestState[which];
  if (ev.key === "ArrowDown" && state.items.length) {
    ev.preventDefault();
    state.active = Math.min(state.active + 1, state.items.length - 1);
    renderSuggestionBox(which);
    return;
  }
  if (ev.key === "ArrowUp" && state.items.length) {
    ev.preventDefault();
    state.active = Math.max(state.active - 1, 0);
    renderSuggestionBox(which);
    return;
  }
  if (ev.key === "Escape") {
    hideSuggestions(which);
    return;
  }
  if (ev.key === "Enter") {
    ev.preventDefault();
    if (state.items.length && state.active >= 0) {
      chooseSuggestion(which, state.items[state.active]);
    }
    planRoute().catch(showRouteError);
  }
}

async function resolvePoint(which) {
  const isStart = which === "start";
  const input = isStart ? el.startAddress : el.endAddress;
  const cached = suggestState[which].selected;
  if (cached && input.value === (cached.label || cached.latlon || `${cached.lat},${cached.lon}`)) {
    return { lat: Number(cached.lat), lon: Number(cached.lon), label: cached.label || cached.latlon };
  }
  const coords = coordinateRow(input.value);
  if (coords) {
    suggestState[which].selected = coords;
    input.value = coords.label;
    return { lat: Number(coords.lat), lon: Number(coords.lon), label: coords.label };
  }
  const best = await geocodeOne(input.value);
  const resolved = { lat: Number(best.lat), lon: Number(best.lon), label: best.label || best.latlon };
  suggestState[which].selected = best;
  input.value = resolved.label;
  return resolved;
}

function swapPoints() {
  [el.startLat.value, el.endLat.value] = [el.endLat.value, el.startLat.value];
  [el.startLon.value, el.endLon.value] = [el.endLon.value, el.startLon.value];
  [el.startAddress.value, el.endAddress.value] = [el.endAddress.value, el.startAddress.value];
  [suggestState.start.selected, suggestState.end.selected] = [suggestState.end.selected, suggestState.start.selected];
}

function clearRoute() {
  lastRouteGeojson = null;
  activeRouteKind = null;
  if (map?.getSource(routeSourceId)) {
    map.getSource(routeSourceId).setData({ type: "FeatureCollection", features: [] });
  }
  el.routeList.innerHTML = '<div class="route-empty">API route output appears here.</div>';
  setRoutePanelCollapsed(true, { persist: true });
}

async function loadHeatmap() {
  if (heatmapLoadPromise) return heatmapLoadPromise;
  heatmapLoadPromise = loadHeatmapNow().finally(() => {
    heatmapLoadPromise = null;
  });
  return heatmapLoadPromise;
}

async function loadHeatmapNow() {
  let payload = null;
  try {
    const resp = await fetch(
      `${api("/v1/heatmap")}?region=${encodeURIComponent(el.region.value)}&max_points=3000&max_tiles=0`
    );
    if (resp.ok) payload = await resp.json();
  } catch {
    payload = null;
  }
  if (!payload) payload = await loadStaticHeatmapPayload();
  if (!payload) payload = buildDemoHeatmapPayload();
  let geojson = payload.geojson || { type: "FeatureCollection", features: [] };
  if (!geojson.features?.length) {
    payload = buildDemoHeatmapPayload();
    geojson = payload.geojson;
  }
  heatmapGeojson = normalizeHeatmapGeojson(geojson);
  if (el.heatmapLegend) {
    el.heatmapLegend.dataset.source = heatmapGeojson.features?.[0]?.properties?.source || "unknown";
  }
  updateSurfaceStats(heatmapGeojson);
  syncHeatmapUi(true);
  const tileZoom = Number(payload.tile_zoom || 16);
  heatmapZoomMode = tileZoom <= 14 ? "z14" : "z16";
  clearScenicCanvas();
  renderMapHeatmapLayer(heatmapGeojson, { fit: true });
}

async function loadStaticHeatmapPayload() {
  if (String(el.region.value || "").toLowerCase() !== "masswhites") return null;
  try {
    const mod = await import(STATIC_HEATMAP_MODULE);
    const cells = Array.isArray(mod.default) ? mod.default : [];
    return {
      tile_zoom: 14,
      geojson: {
        type: "FeatureCollection",
        features: cells.map(([west, south, east, north, score]) => ({
          type: "Feature",
          properties: {
            score_norm: Number(score),
            score: Number(score) * 10,
            source: "masswhites_static_cells_v4",
          },
          geometry: {
            type: "Polygon",
            coordinates: [[
              [Number(west), Number(south)],
              [Number(east), Number(south)],
              [Number(east), Number(north)],
              [Number(west), Number(north)],
              [Number(west), Number(south)],
            ]],
          },
        })),
      },
    };
  } catch {
    return null;
  }
}

function normalizeHeatmapGeojson(geojson) {
  const features = (geojson?.features || []).map((feature) => {
    const props = feature?.properties || {};
    const raw = Number(props.score_norm ?? props.scenic_norm ?? props.scenic_score ?? props.score ?? 0);
    const scoreNorm = Number.isFinite(raw) ? (raw > 1 ? clamp(raw / 10, 0, 1) : clamp(raw, 0, 1)) : 0;
    return {
      ...feature,
      properties: {
        ...props,
        score_norm: scoreNorm,
      },
    };
  });
  return { type: "FeatureCollection", features };
}

function renderMapHeatmapLayer(geojson, { fit = false } = {}) {
  if (!map || !geojson) return;
  if (!isMapStyleEditable()) {
    whenMapStyleReady(() => renderMapHeatmapLayer(geojson, { fit }));
    return;
  }
  removeMaplibreHeatmapLayer();
  const bounds = heatmapBounds(geojson);
  if (bounds) {
    const padded = padBounds(bounds, 0.08);
    renderHeatmapSurfaceMask(padded);
    constrainMapToHeatmap(bounds);
    if (fit) focusHeatmapBounds(padded);
  }
  map.addSource(heatmapSourceId, { type: "geojson", data: geojson });
  const geometryType = geojson.features?.[0]?.geometry?.type;
  if (geometryType === "Polygon" || geometryType === "MultiPolygon") {
    renderFilledHeatmapLayer();
    renderTopHeatmapLayer(geojson);
    bindHeatmapInspector();
    return;
  }
  renderPointHeatmapLayer();
  renderTopHeatmapLayer(geojson);
  bindHeatmapInspector();
}

function scoreLabel(scoreNorm) {
  if (scoreNorm >= surfaceScoreBreaks.peak) return "Peak";
  if (scoreNorm >= surfaceScoreBreaks.high) return "High";
  if (scoreNorm >= surfaceScoreBreaks.moderate) return "Moderate";
  return "Low";
}

function score10(scoreNorm) {
  return clamp(Number(scoreNorm) || 0, 0, 1) * 10;
}

function updateSurfaceStats(geojson) {
  const scores = (geojson?.features || [])
    .map((feature) => Number(feature?.properties?.score_norm))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  const count = scores.length;
  if (!count) {
    el.surfaceCells.textContent = "--";
    el.surfaceAverage.textContent = "--";
    el.surfaceTop.textContent = "--";
    el.surfaceHighShare.textContent = "--";
    return;
  }
  const avg = scores.reduce((sum, score) => sum + score, 0) / count;
  const top = scores[count - 1];
  surfaceScoreBreaks = {
    moderate: scores[Math.floor(count * 0.5)],
    high: scores[Math.floor(count * 0.8)],
    peak: scores[Math.floor(count * 0.95)],
  };
  const highShare = scores.filter((score) => score >= surfaceScoreBreaks.high).length / count;
  el.surfaceCells.textContent = count.toLocaleString();
  el.surfaceAverage.textContent = score10(avg).toFixed(1);
  el.surfaceTop.textContent = score10(top).toFixed(1);
  el.surfaceHighShare.textContent = `${Math.round(highShare * 100)}%`;
}

function topScenicFeatures(geojson, percentile = 0.9) {
  const features = geojson?.features || [];
  const scores = features
    .map((feature) => Number(feature?.properties?.score_norm))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!scores.length) return { type: "FeatureCollection", features: [] };
  const idx = Math.max(0, Math.min(scores.length - 1, Math.floor(scores.length * percentile)));
  const threshold = scores[idx];
  return {
    type: "FeatureCollection",
    features: features.filter((feature) => Number(feature?.properties?.score_norm) >= threshold),
  };
}

function walkCoordinates(coords, visit) {
  if (!Array.isArray(coords)) return;
  if (typeof coords[0] === "number" && typeof coords[1] === "number") {
    visit(Number(coords[0]), Number(coords[1]));
    return;
  }
  coords.forEach((child) => walkCoordinates(child, visit));
}

function heatmapBounds(geojson) {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  (geojson?.features || []).forEach((feature) => {
    walkCoordinates(feature?.geometry?.coordinates, (lon, lat) => {
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) return;
      west = Math.min(west, lon);
      south = Math.min(south, lat);
      east = Math.max(east, lon);
      north = Math.max(north, lat);
    });
  });
  if (![west, south, east, north].every(Number.isFinite)) return null;
  return { west, south, east, north };
}

function padBounds(bounds, ratio) {
  const lonPad = Math.max((bounds.east - bounds.west) * ratio, 0.02);
  const latPad = Math.max((bounds.north - bounds.south) * ratio, 0.02);
  return {
    west: clamp(bounds.west - lonPad, -179.9, 179.9),
    south: clamp(bounds.south - latPad, -84.9, 84.9),
    east: clamp(bounds.east + lonPad, -179.9, 179.9),
    north: clamp(bounds.north + latPad, -84.9, 84.9),
  };
}

function rectangleFeature(west, south, east, north) {
  return {
    type: "Feature",
    properties: {},
    geometry: {
      type: "Polygon",
      coordinates: [[
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ]],
    },
  };
}

function renderHeatmapSurfaceMask(bounds) {
  const world = { west: -180, south: -85, east: 180, north: 85 };
  const features = [
    rectangleFeature(world.west, world.south, bounds.west, world.north),
    rectangleFeature(bounds.east, world.south, world.east, world.north),
    rectangleFeature(bounds.west, world.south, bounds.east, bounds.south),
    rectangleFeature(bounds.west, bounds.north, bounds.east, world.north),
  ];
  map.addSource(heatmapMaskSourceId, {
    type: "geojson",
    data: { type: "FeatureCollection", features },
  });
  map.addLayer({
    id: heatmapMaskLayerId,
    type: "fill",
    source: heatmapMaskSourceId,
    paint: {
      "fill-color": "rgba(246, 248, 246, 0.88)",
      "fill-opacity": 1,
    },
  });
  map.addSource(heatmapBoundsSourceId, {
    type: "geojson",
    data: rectangleFeature(bounds.west, bounds.south, bounds.east, bounds.north),
  });
  map.addLayer({
    id: heatmapBoundsLayerId,
    type: "line",
    source: heatmapBoundsSourceId,
    paint: {
      "line-color": "rgba(36, 79, 69, 0.34)",
      "line-width": [
        "interpolate",
        ["linear"],
        ["zoom"],
        7,
        0.6,
        12,
        0.9,
        16,
        1.2,
      ],
      "line-opacity": 0.42,
    },
  });
}

function focusHeatmapBounds(bounds) {
  const compact = window.matchMedia("(max-width: 760px)").matches;
  map.fitBounds(
    [
      [bounds.west, bounds.south],
      [bounds.east, bounds.north],
    ],
    {
      padding: compact
        ? { top: 58, right: 28, bottom: Math.round(window.innerHeight * 0.58), left: 28 }
        : { top: 54, right: 330, bottom: 54, left: 340 },
      duration: 0,
    }
  );
}

function constrainMapToHeatmap(bounds) {
  const loose = padBounds(bounds, 1.6);
  try {
    map.setMaxBounds([
      [loose.west, loose.south],
      [loose.east, loose.north],
    ]);
  } catch {
    // Older MapLibre builds can be picky about maxBounds during style swaps.
  }
}

function scenicColorExpression() {
  return [
    "interpolate",
    ["linear"],
    ["get", "score_norm"],
    0,
    "rgba(49, 86, 97, 0.9)",
    0.25,
    "rgba(62, 124, 101, 0.92)",
    0.52,
    "rgba(146, 162, 92, 0.94)",
    0.72,
    "rgba(223, 164, 74, 0.96)",
    0.88,
    "rgba(203, 102, 58, 0.98)",
    1,
    "rgba(139, 58, 48, 1)",
  ];
}

function renderFilledHeatmapLayer() {
  map.addLayer({
    id: heatmapLayerId,
    type: "fill",
    source: heatmapSourceId,
    paint: {
      "fill-color": scenicColorExpression(),
      "fill-opacity": 0.64,
      "fill-outline-color": "rgba(0, 0, 0, 0)",
      "fill-antialias": true,
    },
  });
  map.addSource(heatmapSoftSourceId, { type: "geojson", data: cellCenterGeojson(heatmapGeojson) });
  map.addLayer({
    id: heatmapSoftLayerId,
    type: "circle",
    source: heatmapSoftSourceId,
    layout: {
      "circle-sort-key": ["get", "score_norm"],
    },
    paint: {
      "circle-color": scenicColorExpression(),
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["zoom"],
        5,
        10,
        8,
        20,
        10,
        31,
        12,
        44,
        14,
        76,
        16,
        128,
        18,
        220,
      ],
      "circle-blur": 1.35,
      "circle-opacity": [
        "interpolate",
        ["linear"],
        ["zoom"],
        5,
        0.16,
        12,
        0.2,
        16,
        0.28,
        18,
        0.32,
      ],
    },
  });
  if (map.getLayer("baseline-line")) map.moveLayer("baseline-line");
  if (map.getLayer("scenic-line")) map.moveLayer("scenic-line");
}

function renderTopHeatmapLayer(geojson) {
  const topGeojson = topScenicFeatures(geojson, 0.9);
  if (!topGeojson.features.length) return;
  map.addSource(heatmapTopSourceId, { type: "geojson", data: topGeojson });
  const geometryType = topGeojson.features?.[0]?.geometry?.type;
  if (geometryType === "Polygon" || geometryType === "MultiPolygon") {
    map.addLayer({
      id: heatmapTopFillLayerId,
      type: "fill",
      source: heatmapTopSourceId,
      paint: {
        "fill-color": "rgba(255, 236, 190, 0.16)",
        "fill-opacity": 1,
      },
    });
  } else {
    map.addLayer({
      id: heatmapTopFillLayerId,
      type: "circle",
      source: heatmapTopSourceId,
      paint: {
        "circle-color": "rgba(139, 58, 48, 0.34)",
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          7,
          10,
          12,
          28,
          16,
          74,
        ],
        "circle-stroke-color": "rgba(139, 58, 48, 0.72)",
        "circle-stroke-width": 1.2,
        "circle-blur": 0.55,
      },
    });
  }
  if (map.getLayer(heatmapBoundsLayerId)) map.moveLayer(heatmapBoundsLayerId);
  if (map.getLayer("baseline-line")) map.moveLayer("baseline-line");
  if (map.getLayer("scenic-line")) map.moveLayer("scenic-line");
}

function cellCenterGeojson(geojson) {
  const features = (geojson?.features || [])
    .map((feature) => {
      const ring = feature?.geometry?.coordinates?.[0] || [];
      if (!ring.length) return null;
      let minLon = Infinity;
      let minLat = Infinity;
      let maxLon = -Infinity;
      let maxLat = -Infinity;
      for (const [lon, lat] of ring) {
        minLon = Math.min(minLon, Number(lon));
        minLat = Math.min(minLat, Number(lat));
        maxLon = Math.max(maxLon, Number(lon));
        maxLat = Math.max(maxLat, Number(lat));
      }
      if (![minLon, minLat, maxLon, maxLat].every(Number.isFinite)) return null;
      return {
        type: "Feature",
        properties: feature.properties || {},
        geometry: {
          type: "Point",
          coordinates: [(minLon + maxLon) / 2, (minLat + maxLat) / 2],
        },
      };
    })
    .filter(Boolean);
  return { type: "FeatureCollection", features };
}

function renderPointHeatmapLayer() {
  map.addLayer({
    id: heatmapLayerId,
    type: "circle",
    source: heatmapSourceId,
    layout: {
      "circle-sort-key": ["get", "score_norm"],
    },
    paint: {
      "circle-color": scenicColorExpression(),
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["zoom"],
        5,
        heatmapZoomMode === "z14" ? 8 : 6,
        8,
        heatmapZoomMode === "z14" ? 17 : 13,
        10,
        heatmapZoomMode === "z14" ? 28 : 22,
        12,
        heatmapZoomMode === "z14" ? 46 : 36,
        14,
        heatmapZoomMode === "z14" ? 88 : 68,
        16,
        heatmapZoomMode === "z14" ? 138 : 104,
        18,
        heatmapZoomMode === "z14" ? 220 : 166,
      ],
      "circle-blur": [
        "interpolate",
        ["linear"],
        ["zoom"],
        5,
        1.65,
        11,
        1.35,
        16,
        1.1,
        18,
        1.15,
      ],
      "circle-opacity": [
        "interpolate",
        ["linear"],
        ["zoom"],
        5,
        0.34,
        12,
        0.42,
        16,
        0.5,
        18,
        0.54,
      ],
    },
  });
  if (map.getLayer("baseline-line")) map.moveLayer("baseline-line");
  if (map.getLayer("scenic-line")) map.moveLayer("scenic-line");
}

function bindHeatmapInspector() {
  if (heatmapInspectorBound || !map) return;
  heatmapInspectorBound = true;
  const inspect = (ev) => {
    if (!el.showHeatmap?.checked || !map.getLayer(heatmapLayerId)) {
      hideSurfaceInspector();
      return;
    }
    const features = map.queryRenderedFeatures(ev.point, { layers: [heatmapLayerId] });
    const feature = features.find((item) => Number.isFinite(Number(item?.properties?.score_norm)));
    if (!feature) {
      hideSurfaceInspector();
      return;
    }
    showSurfaceInspector(feature, ev.lngLat);
  };
  map.on("mousemove", inspect);
  map.on("click", inspect);
  map.on("mouseout", hideSurfaceInspector);
}

function showSurfaceInspector(feature, lngLat) {
  const scoreNorm = clamp(Number(feature?.properties?.score_norm) || 0, 0, 1);
  el.surfaceInspectorLabel.textContent = scoreLabel(scoreNorm);
  el.surfaceInspectorScore.textContent = score10(scoreNorm).toFixed(1);
  el.surfaceInspectorCoords.textContent = `${Number(lngLat.lat).toFixed(4)}, ${Number(lngLat.lng).toFixed(4)}`;
  el.surfaceInspector.classList.remove("hidden");
}

function hideSurfaceInspector() {
  el.surfaceInspector?.classList.add("hidden");
}

function buildDemoHeatmapPayload() {
  const features = [];
  const bounds = DEMO_HEATMAP_BOUNDS;
  const cols = 36;
  const rows = 66;
  for (let y = 0; y < rows; y += 1) {
    const lat = bounds.minLat + ((bounds.maxLat - bounds.minLat) * y) / (rows - 1);
    for (let x = 0; x < cols; x += 1) {
      const lon = bounds.minLon + ((bounds.maxLon - bounds.minLon) * x) / (cols - 1);
      const t = (lat - bounds.minLat) / (bounds.maxLat - bounds.minLat);
      const ridgeLon = -73.28 + Math.sin(t * Math.PI * 2.4) * 0.1;
      const ridge = Math.exp(-Math.pow((lon - ridgeLon) / 0.18, 2));
      const easternRidge = Math.exp(-Math.pow((lon + 72.78) / 0.16, 2)) * (0.55 + 0.2 * Math.sin(lat * 5));
      const valleyDip = Math.exp(-Math.pow((lon + 73.08) / 0.08, 2)) * 0.22;
      const texture = (Math.sin(lat * 12.7 + lon * 8.1) + Math.sin(lat * 7.4 - lon * 10.3)) * 0.045;
      const score = clamp(0.24 + ridge * 0.45 + easternRidge * 0.28 - valleyDip + texture, 0.08, 0.98);
      features.push({
        type: "Feature",
        properties: {
          score_norm: score,
          score,
          source: "demo_surface",
        },
        geometry: {
          type: "Point",
          coordinates: [lon, lat],
        },
      });
    }
  }
  return {
    tile_zoom: 14,
    geojson: { type: "FeatureCollection", features },
  };
}

function hideHeatmap() {
  clearScenicCanvas();
  removeMaplibreHeatmapLayer();
  hideSurfaceInspector();
  try {
    map?.setMaxBounds(null);
  } catch {
    // Let the existing map extent stand if this MapLibre build rejects null bounds.
  }
  syncHeatmapUi(false);
}

function syncHeatmapUi(visible = Boolean(el.showHeatmap?.checked)) {
  el.heatmapLegend?.classList.toggle("hidden", !visible);
  if (!visible) hideSurfaceInspector();
  document.body.classList.toggle("scenic-layer-active", visible);
}

function enableDefaultScenicLayer() {
  el.showHeatmap.checked = true;
  syncHeatmapUi(true);
  loadHeatmap().catch(() => showRouteMessage("Heatmap unavailable for this region."));
}

function removeMaplibreHeatmapLayer() {
  if (!map) return;
  if (map.getLayer(heatmapTopLineLayerId)) map.removeLayer(heatmapTopLineLayerId);
  if (map.getLayer(heatmapTopFillLayerId)) map.removeLayer(heatmapTopFillLayerId);
  if (map.getLayer(heatmapSoftLayerId)) map.removeLayer(heatmapSoftLayerId);
  if (map.getLayer(heatmapLayerId)) map.removeLayer(heatmapLayerId);
  if (map.getLayer(heatmapBoundsLayerId)) map.removeLayer(heatmapBoundsLayerId);
  if (map.getLayer(heatmapMaskLayerId)) map.removeLayer(heatmapMaskLayerId);
  if (map.getLayer(heatmapTilesLayerId)) map.removeLayer(heatmapTilesLayerId);
  if (map.getSource(heatmapTopSourceId)) map.removeSource(heatmapTopSourceId);
  if (map.getSource(heatmapSoftSourceId)) map.removeSource(heatmapSoftSourceId);
  if (map.getSource(heatmapSourceId)) map.removeSource(heatmapSourceId);
  if (map.getSource(heatmapBoundsSourceId)) map.removeSource(heatmapBoundsSourceId);
  if (map.getSource(heatmapMaskSourceId)) map.removeSource(heatmapMaskSourceId);
  if (map.getSource(heatmapTilesSourceId)) map.removeSource(heatmapTilesSourceId);
}

function clearScenicCanvas() {
  const canvas = el.scenicCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width || 0, canvas.height || 0);
}

function formatDistance(km) {
  const miles = Number(km) * 0.621371;
  if (!Number.isFinite(miles)) return "n/a";
  return `${miles.toFixed(1)} mi`;
}

function formatMinutes(minutes) {
  if (!Number.isFinite(minutes)) return "n/a";
  return `${Number(minutes).toFixed(1)} min`;
}

function formatScore(score) {
  if (!Number.isFinite(score)) return "n/a";
  return Number(score).toFixed(2);
}

function formatDelta(value, suffix) {
  if (!Number.isFinite(value)) return "n/a";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "±";
  const abs = Math.abs(Number(value)).toFixed(1);
  return `${sign}${abs}${suffix}`;
}

function setRouteHighlight(kind) {
  if (!map) return;
  const scenicLayer = "scenic-line";
  const baselineLayer = "baseline-line";
  if (!map.getLayer(scenicLayer) || !map.getLayer(baselineLayer)) return;
  const scenicActive = kind === "scenic";
  const baselineActive = kind === "baseline";
  map.setPaintProperty(scenicLayer, "line-opacity", scenicActive ? 0.95 : 0.45);
  map.setPaintProperty(baselineLayer, "line-opacity", baselineActive ? 0.9 : 0.35);
  map.setPaintProperty(scenicLayer, "line-width", scenicActive ? 7 : 4);
  map.setPaintProperty(baselineLayer, "line-width", baselineActive ? 6 : 3);
}

function setActiveRoute(kind) {
  activeRouteKind = kind;
  document.querySelectorAll(".route-card").forEach((node) => {
    node.classList.toggle("active", node.dataset.kind === kind);
  });
  setRouteHighlight(kind);
}

function renderRouteList(payload) {
  const routes = payload.routes || {};
  const scenic = routes.scenic;
  const baseline = routes.baseline;
  const deltas = payload.deltas || {};
  const diagnostics = payload.diagnostics || {};
  const cards = [];
  if (scenic) {
    const deltaTime = Number(deltas.duration_min || 0);
    const badge = Number.isFinite(deltaTime) && deltaTime !== 0 ? formatDelta(deltaTime, " min") : "Scenic";
    cards.push({
      kind: "scenic",
      title: "Scenic",
      badge,
      deltaTime,
      deltaScore: Number(deltas.scenic_score || 0),
      time: scenic.estimated_duration_minutes,
      distance: scenic.total_distance_km,
      score: scenic.average_scenic_score,
    });
  }
  if (baseline) {
    cards.push({
      kind: "baseline",
      title: "Fastest",
      badge: "Baseline",
      deltaTime: 0,
      deltaScore: 0,
      time: baseline.estimated_duration_minutes,
      distance: baseline.total_distance_km,
      score: baseline.average_scenic_score,
    });
  }
  if (!cards.length) {
    el.routeList.innerHTML = '<div class="route-empty">Plan a route to compare options.</div>';
    return;
  }
  el.routeList.innerHTML = cards
    .map(
      (card) => `
        <div class="route-card" data-kind="${card.kind}">
          <div class="row">
            <div class="title">${card.title}</div>
            <span class="badge ${card.kind === "scenic" ? "badge-warm" : "badge-cool"}">${card.badge}</span>
          </div>
          <div class="meta">
            <div class="metric">
              <div class="k">Time</div>
              <div class="v">${formatMinutes(card.time)}</div>
            </div>
            <div class="metric">
              <div class="k">Distance</div>
              <div class="v">${formatDistance(card.distance)}</div>
            </div>
            <div class="metric">
              <div class="k">Scenic</div>
              <div class="v">${formatScore(card.score)}</div>
            </div>
          </div>
          ${
            card.kind === "scenic"
              ? `
            <div class="delta-row">
              <span class="delta-chip">${formatDelta(card.deltaTime, " min")}</span>
              <span class="delta-chip">${formatDelta(card.deltaScore, " scenic")}</span>
            </div>
          `
              : ""
          }
        </div>
      `
    )
    .join("") +
    (diagnostics.start_snap_km || diagnostics.end_snap_km
      ? `<div class="route-diagnostics">Snapped to road graph: start ${Number(
          diagnostics.start_snap_km || 0
        ).toFixed(2)} km, end ${Number(diagnostics.end_snap_km || 0).toFixed(2)} km.</div>`
      : "");
  document.querySelectorAll(".route-card").forEach((node) => {
    node.addEventListener("click", () => setActiveRoute(node.dataset.kind));
  });
  setRoutePanelCollapsed(false, { persist: true });
  if (!activeRouteKind && scenic) {
    setActiveRoute("scenic");
  } else if (!activeRouteKind && baseline) {
    setActiveRoute("baseline");
  } else if (activeRouteKind) {
    setActiveRoute(activeRouteKind);
  }
}

function syncRoutePanelState() {
  const stored = localStorage.getItem(STORAGE_KEYS.routePanelCollapsed);
  const collapsed = stored == null ? true : stored === "1";
  setRoutePanelCollapsed(collapsed, { persist: false });
}

function setRoutePanelCollapsed(collapsed, { persist = true } = {}) {
  el.routePanel.classList.toggle("collapsed", collapsed);
  el.routeCollapseBtn.textContent = collapsed ? "Show" : "Hide";
  if (persist) localStorage.setItem(STORAGE_KEYS.routePanelCollapsed, collapsed ? "1" : "0");
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 760px)").matches;
}

function setSheetOpen(open) {
  el.controls.classList.toggle("sheet-open", open);
  localStorage.setItem(STORAGE_KEYS.settingsSheetOpen, open ? "1" : "0");
  if (el.sheetToggleBtn) {
    el.sheetToggleBtn.textContent = open ? "Collapse" : "Expand";
  }
}

function setActiveTab(tab, { persist = true } = {}) {
  const next = ["search", "routes", "settings", "none"].includes(tab) ? tab : "search";
  document.body.dataset.activeTab = next;
  if (persist) localStorage.setItem(STORAGE_KEYS.activeTab, next);
  if (next === "settings") {
    const stored = localStorage.getItem(STORAGE_KEYS.settingsSheetOpen);
    const open = stored ? stored === "1" : true;
    setSheetOpen(open);
  } else {
    setSheetOpen(false);
  }
}

function syncTabState() {
  document.body.dataset.activeTab = "search";
  setSheetOpen(false);
}

function readSavedTrips() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.savedTrips);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeSavedTrips(trips) {
  localStorage.setItem(STORAGE_KEYS.savedTrips, JSON.stringify(trips));
}

function renderSavedTrips() {
  const trips = readSavedTrips();
  if (!trips.length) {
    el.savedTrips.innerHTML = '<div class="route-empty">No saved trips yet.</div>';
    return;
  }
  el.savedTrips.innerHTML = trips
    .map(
      (trip) => `
        <div class="saved-item" data-id="${trip.id}">
          <div class="title">${trip.title}</div>
          <div class="meta">${trip.region} • ${new Date(trip.timestamp).toLocaleDateString()}</div>
          ${
            trip.summary
              ? `<div class="summary">${trip.summary}</div>`
              : ""
          }
          <div class="saved-actions">
            <button class="ghost-btn" data-action="rename">Rename</button>
            <button class="ghost-btn" data-action="delete">Delete</button>
          </div>
        </div>
      `
    )
    .join("");
  el.savedTrips.querySelectorAll(".saved-item").forEach((node) => {
    node.addEventListener("click", () => {
      const trip = trips.find((t) => t.id === node.dataset.id);
      if (!trip) return;
      el.region.value = trip.region;
      el.scenicWeight.value = String(trip.scenic_weight);
      el.weightValue.textContent = String(trip.scenic_weight);
      el.maxDetour.value = String(trip.max_detour_factor);
      el.detourValue.textContent = String(trip.max_detour_factor);
      el.startLat.value = trip.start.lat;
      el.startLon.value = trip.start.lon;
      el.endLat.value = trip.end.lat;
      el.endLon.value = trip.end.lon;
      el.startAddress.value = trip.start_label || `${trip.start.lat},${trip.start.lon}`;
      el.endAddress.value = trip.end_label || `${trip.end.lat},${trip.end.lon}`;
      inputMode = "address";
      syncInputMode();
      planRoute().catch(showRouteError);
    });
    node.querySelectorAll(".ghost-btn").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const action = btn.dataset.action;
        const trip = trips.find((t) => t.id === node.dataset.id);
        if (!trip) return;
        if (action === "delete") {
          const next = trips.filter((t) => t.id !== trip.id);
          writeSavedTrips(next);
          renderSavedTrips();
          return;
        }
        if (action === "rename") {
          const name = prompt("Rename saved trip:", trip.title);
          if (!name) return;
          trip.title = name.trim();
          writeSavedTrips(trips);
          renderSavedTrips();
        }
      });
    });
  });
}

function buildShareUrl(plan) {
  const url = new URL(window.location.href);
  url.searchParams.set("region", plan.region);
  url.searchParams.set("scenic_weight", String(plan.scenic_weight));
  url.searchParams.set("max_detour", String(plan.max_detour_factor));
  url.searchParams.set("start", `${plan.start.lat},${plan.start.lon}`);
  url.searchParams.set("end", `${plan.end.lat},${plan.end.lon}`);
  if (plan.start_label) url.searchParams.set("start_label", plan.start_label);
  if (plan.end_label) url.searchParams.set("end_label", plan.end_label);
  return url.toString();
}

function updateShareUrl(plan) {
  const url = buildShareUrl(plan);
  window.history.replaceState({}, "", url);
}

function applyUrlParams() {
  const params = new URLSearchParams(window.location.search);
  if (!params.size) return;
  let shouldPlan = false;
  const region = params.get("region");
  if (region && Array.from(el.region.options).some((opt) => opt.value === region)) {
    el.region.value = region;
  }
  const scenicWeight = params.get("scenic_weight");
  if (scenicWeight) {
    el.scenicWeight.value = scenicWeight;
    el.weightValue.textContent = scenicWeight;
  }
  const maxDetour = params.get("max_detour");
  if (maxDetour) {
    el.maxDetour.value = maxDetour;
    el.detourValue.textContent = maxDetour;
  }
  const start = params.get("start");
  const end = params.get("end");
  const startLabel = params.get("start_label");
  const endLabel = params.get("end_label");
  const parsePair = (value) => {
    if (!value) return null;
    const parts = value.split(",").map((v) => Number(v.trim()));
    if (parts.length !== 2 || parts.some((v) => !Number.isFinite(v))) return null;
    return { lat: parts[0], lon: parts[1] };
  };
  const startPair = parsePair(start);
  const endPair = parsePair(end);
  if (startPair) {
    el.startLat.value = startPair.lat;
    el.startLon.value = startPair.lon;
    shouldPlan = true;
  }
  if (endPair) {
    el.endLat.value = endPair.lat;
    el.endLon.value = endPair.lon;
    shouldPlan = true;
  }
  if (startLabel) el.startAddress.value = startLabel;
  if (endLabel) el.endAddress.value = endLabel;
  if (startLabel || endLabel) {
    inputMode = "address";
    shouldPlan = true;
  } else if (startPair || endPair) {
    inputMode = "coords";
  }
  syncInputMode();
  return shouldPlan;
}

function renderRouteGeojson(geojson, options = {}) {
  const fit = options.fit !== false;
  lastRouteGeojson = geojson;
  const featureCount = geojson?.features?.length || 0;
  if (!featureCount) {
    showRouteMessage("Route returned no geometry.");
    return;
  }
  if (map.getLayer("scenic-line")) map.removeLayer("scenic-line");
  if (map.getLayer("baseline-line")) map.removeLayer("baseline-line");
  if (map.getSource(routeSourceId)) map.removeSource(routeSourceId);

  map.addSource(routeSourceId, { type: "geojson", data: geojson });
  map.addLayer({
    id: "baseline-line",
    type: "line",
    source: routeSourceId,
    filter: ["==", ["get", "route_kind"], "baseline"],
    paint: { "line-color": "#4ca8ff", "line-width": 5, "line-opacity": 0.85 },
  });
  map.addLayer({
    id: "scenic-line",
    type: "line",
    source: routeSourceId,
    filter: ["!=", ["get", "route_kind"], "baseline"],
    paint: { "line-color": "#ff7d3b", "line-width": 6, "line-opacity": 0.95 },
  });
  if (map.getLayer("baseline-line")) map.moveLayer("baseline-line");
  if (map.getLayer("scenic-line")) map.moveLayer("scenic-line");
  if (activeRouteKind) setRouteHighlight(activeRouteKind);

  const bounds = new maplibregl.LngLatBounds();
  for (const feature of geojson.features || []) {
    const coords = feature?.geometry?.coordinates || [];
    for (const [lon, lat] of coords) bounds.extend([lon, lat]);
  }
  if (fit && !bounds.isEmpty()) map.fitBounds(bounds, { padding: 36, duration: 0 });
}

function scheduleRouteRender(geojson, options = {}) {
  pendingRouteGeojson = geojson;
  pendingRouteRenderOptions = options;
  if (!map) return;
  if (map.isStyleLoaded()) {
    renderRouteGeojson(pendingRouteGeojson, pendingRouteRenderOptions);
    pendingRouteGeojson = null;
    pendingRouteRenderOptions = {};
    return;
  }
  map.once("style.load", () => {
    if (!pendingRouteGeojson) return;
    renderRouteGeojson(pendingRouteGeojson, pendingRouteRenderOptions);
    pendingRouteGeojson = null;
    pendingRouteRenderOptions = {};
  });
}

async function planRoute() {
  const seq = ++planSeq;
  setBusy(true);
  if (map?.getSource(routeSourceId)) {
    map.getSource(routeSourceId).setData({ type: "FeatureCollection", features: [] });
  }
  el.routeList.innerHTML = '<div class="route-empty">Planning new route...</div>';
  hideSuggestions("start");
  hideSuggestions("end");
  try {
    let start;
    let end;
    let startLabel;
    let endLabel;
    if (inputMode === "address") {
      const [s, e] = await Promise.all([resolvePoint("start"), resolvePoint("end")]);
      start = { lat: s.lat, lon: s.lon };
      end = { lat: e.lat, lon: e.lon };
      startLabel = s.label || `${s.lat},${s.lon}`;
      endLabel = e.label || `${e.lat},${e.lon}`;
      el.startLat.value = String(s.lat);
      el.startLon.value = String(s.lon);
      el.endLat.value = String(e.lat);
      el.endLon.value = String(e.lon);
    } else {
      start = { lat: Number(el.startLat.value), lon: Number(el.startLon.value) };
      end = { lat: Number(el.endLat.value), lon: Number(el.endLon.value) };
      startLabel = `${start.lat},${start.lon}`;
      endLabel = `${end.lat},${end.lon}`;
    }
    lastPlan = {
      start,
      end,
      region: el.region.value,
      scenic_weight: Number(el.scenicWeight.value),
      max_detour_factor: Number(el.maxDetour.value),
      start_label: startLabel,
      end_label: endLabel,
    };
    let resp;
    try {
      resp = await fetch(api("/v1/route/compare"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          start,
          end,
          scenic_weight: Number(el.scenicWeight.value),
          region: el.region.value,
          max_detour_factor: Number(el.maxDetour.value),
          avoid_highways: false,
          include_baseline: true,
        }),
      });
    } catch (err) {
      throw new Error(
        "API request failed. Check API Base URL, server status, and HTTPS/mixed-content blocking."
      );
    }
    let payload = null;
    try {
      payload = await resp.json();
    } catch {
      payload = null;
    }
    if (!resp.ok) {
      if (resp.status === 422 && payload?.detail?.error === "no_route_found") {
        const hint = payload?.detail?.hint ? ` ${payload.detail.hint}` : "";
        const diag = payload?.detail?.diagnostics || {};
        const snap = diag.start_snap_km || diag.end_snap_km
          ? ` Nearest roads: start ${Number(diag.start_snap_km || 0).toFixed(1)} km, end ${Number(
              diag.end_snap_km || 0
            ).toFixed(1)} km.`
          : "";
        throw new Error(`No route found in ${el.region.value}.${snap}${hint}`);
      }
      if (payload?.detail) {
        const detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
        throw new Error(`Route request failed (${resp.status}): ${detail}`);
      }
      throw new Error(`Route request failed (${resp.status}).`);
    }

    if (seq !== planSeq) return;
    renderRouteList(payload);
    lastRoutePayload = payload;
    if (seq === planSeq) scheduleRouteRender(payload.geojson);
    updateShareUrl(lastPlan);
  } finally {
    if (seq === planSeq) setBusy(false);
  }
}

function showRouteError(err) {
  const msg = String(err?.message || err || "Route request failed.");
  showRouteMessage(msg);
}

function installAddressInput(which) {
  const input = which === "start" ? el.startAddress : el.endAddress;
  input.addEventListener("input", () => onAddressInput(which));
  input.addEventListener("focus", () => onAddressInput(which));
  input.addEventListener("keydown", (ev) => onAddressKey(which, ev));
  input.addEventListener("blur", () => {
    setTimeout(() => hideSuggestions(which), 120);
  });
}

async function main() {
  Object.entries(el).forEach(([name, node]) => requireElement(name, node));

  initMap();
  await checkHealth();
  try {
    await loadRegions();
  } catch {
    loadFallbackRegions();
  }
  applyRegionDefaults();
  await loadTrainingResults();
  syncInputMode();
  syncTabState();
  applyRoutePreference(localStorage.getItem(STORAGE_KEYS.routePreference) || "balanced");
  const autoPlan = applyUrlParams();
  renderSavedTrips();
  el.routeList.innerHTML = '<div class="route-empty">API route output appears here.</div>';
  enableDefaultScenicLayer();
  if (autoPlan) {
    planRoute().catch(showRouteError);
  }

  el.refreshRegionsBtn.addEventListener("click", async () => {
    await checkHealth();
    await loadRegions();
    await loadTrainingResults();
    applyRegionDefaults();
  });
  el.region.addEventListener("change", () => {
    applyRegionDefaults();
    if (el.showHeatmap.checked && map.loaded()) {
      loadHeatmap();
    }
  });
  el.modeAddressBtn.addEventListener("click", () => {
    inputMode = "address";
    syncInputMode();
  });
  el.modeCoordsBtn.addEventListener("click", () => {
    inputMode = "coords";
    syncInputMode();
  });
  el.scenicWeight.addEventListener("input", () => {
    el.weightValue.textContent = el.scenicWeight.value;
  });
  el.maxDetour.addEventListener("input", () => {
    el.detourValue.textContent = el.maxDetour.value;
  });
  [el.prefFastBtn, el.prefBalancedBtn, el.prefScenicBtn].forEach((btn) => {
    btn.addEventListener("click", () => applyRoutePreference(btn.dataset.pref));
  });
  el.swapBtn.addEventListener("click", swapPoints);
  el.clearBtn.addEventListener("click", clearRoute);
  el.menuBtn.addEventListener("click", toggleMenu);
  el.showHeatmap.addEventListener("change", () => {
    if (!map.loaded()) return;
    if (el.showHeatmap.checked) loadHeatmap();
    else hideHeatmap();
  });
  el.planBtn.addEventListener("click", () => {
    planRoute().catch(showRouteError);
  });
  el.saveRouteBtn.addEventListener("click", () => {
    if (!lastPlan) {
      showRouteMessage("Plan a route before saving.", "route-empty");
      return;
    }
    const trips = readSavedTrips();
    const title = `${lastPlan.start_label} → ${lastPlan.end_label}`;
    const scenic = lastRoutePayload?.routes?.scenic;
    const summary = scenic
      ? `${formatScore(scenic.average_scenic_score)} scenic • ${formatDistance(
          scenic.total_distance_km
        )} • ${formatMinutes(scenic.estimated_duration_minutes)}`
      : "";
    trips.unshift({
      id: String(Date.now()),
      title,
      start: lastPlan.start,
      end: lastPlan.end,
      region: lastPlan.region,
      scenic_weight: lastPlan.scenic_weight,
      max_detour_factor: lastPlan.max_detour_factor,
      start_label: lastPlan.start_label,
      end_label: lastPlan.end_label,
      summary,
      timestamp: Date.now(),
    });
    writeSavedTrips(trips.slice(0, 10));
    renderSavedTrips();
  });
  el.shareRouteBtn.addEventListener("click", async () => {
    if (!lastPlan) {
      showRouteMessage("Plan a route before sharing.", "route-empty");
      return;
    }
    const url = buildShareUrl(lastPlan);
    const original = el.shareRouteBtn.textContent;
    try {
      await navigator.clipboard.writeText(url);
      el.shareRouteBtn.textContent = "Copied";
      setTimeout(() => {
        el.shareRouteBtn.textContent = original;
      }, 1200);
    } catch {
      el.shareRouteBtn.textContent = "Copied";
      setTimeout(() => {
        el.shareRouteBtn.textContent = original;
      }, 1800);
    }
  });
  if (el.sheetToggleBtn) {
    el.sheetToggleBtn.addEventListener("click", () => {
      const open = !el.controls.classList.contains("sheet-open");
      setSheetOpen(open);
    });
  }
  el.routeCollapseBtn.addEventListener("click", () => {
    setRoutePanelCollapsed(!el.routePanel.classList.contains("collapsed"), { persist: true });
  });
  el.basemapProvider.addEventListener("change", () => {
    applyBasemap();
  });
  [el.mapStyleMapBtn, el.mapStyleSatelliteBtn].forEach((btn) => {
    btn.addEventListener("click", () => {
      el.basemapProvider.value = btn.dataset.basemap || "carto_voyager";
      applyBasemap();
    });
  });

  installAddressInput("start");
  installAddressInput("end");
  syncRoutePanelState();
}

main().catch((err) => {
  setStatus(String(err));
  if (el.routeList) showRouteMessage(String(err));
});
