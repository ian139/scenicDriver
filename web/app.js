const el = {
  controls: document.querySelector(".controls"),
  apiBase: document.getElementById("apiBase"),
  basemapProvider: document.getElementById("basemapProvider"),
  maptilerKey: document.getElementById("maptilerKey"),
  basemapStatus: document.getElementById("basemapStatus"),
  region: document.getElementById("region"),
  regionBadge: document.getElementById("regionBadge"),
  modeAddressBtn: document.getElementById("modeAddressBtn"),
  modeCoordsBtn: document.getElementById("modeCoordsBtn"),
  prefFastBtn: document.getElementById("prefFastBtn"),
  prefBalancedBtn: document.getElementById("prefBalancedBtn"),
  prefScenicBtn: document.getElementById("prefScenicBtn"),
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
const heatmapTilesSourceId = "scenic-heatmap-tiles";
const heatmapTilesLayerId = "scenic-heatmap-tiles-layer";
let heatmapZoomMode = "z16";
let regionsMeta = [];
let busy = false;
let inputMode = "address";
let routePreference = "balanced";
let basemapApplying = false;
let planSeq = 0;
let pendingRouteGeojson = null;
let lastRoutePayload = null;
const suggestTimer = { start: null, end: null };
const suggestState = {
  start: { items: [], active: -1, selected: null },
  end: { items: [], active: -1, selected: null },
};
const suggestSeq = { start: 0, end: 0 };

const MAPTILER_STYLE = "streets-v2";
const STORAGE_KEYS = {
  maptilerKey: "scenicdrive.maptilerKey",
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

function requireElement(name, node) {
  if (!node) throw new Error(`Missing required UI element: ${name}`);
}

function setStatus(msg) {
  el.status.textContent = String(msg || "").replace(/\n+/g, " • ");
}

function setBusy(next) {
  busy = Boolean(next);
  el.planBtn.disabled = busy;
  el.planBtn.textContent = busy ? "Planning..." : "Plan Route";
  el.loadingOverlay.classList.toggle("hidden", !busy);
}

function toggleMenu() {
  if (isMobileLayout()) {
    setActiveTab("settings");
    return;
  }
  el.controls?.classList.toggle("menu-collapsed");
}

function api(path) {
  return `${el.apiBase.value.replace(/\/+$/, "")}${path}`;
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

function normalizeBasemapProvider(provider, key) {
  const p = String(provider || "").toLowerCase();
  if (p === "maptiler" && !key) return "carto_voyager";
  if (["carto_voyager", "carto_positron", "maptiler", "osm"].includes(p)) return p;
  return key ? "maptiler" : "carto_voyager";
}

function basemapTileSource(provider, key) {
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
  if (provider === "carto_positron") {
    return {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: "© CARTO © OpenStreetMap contributors",
    };
  }
  if (provider === "maptiler") {
    const encoded = encodeURIComponent(key);
    return {
      type: "raster",
      tiles: [`https://api.maptiler.com/maps/${MAPTILER_STYLE}/{z}/{x}/{y}.png?key=${encoded}`],
      tileSize: 256,
      attribution: "© MapTiler © OpenStreetMap contributors",
    };
  }
  return {
    type: "raster",
    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    tileSize: 256,
    attribution: "© OpenStreetMap contributors",
  };
}

function buildBasemapStyle(provider, key) {
  const source = basemapTileSource(provider, key);
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
    carto_voyager: "CARTO Voyager",
    carto_positron: "CARTO Light",
    maptiler: "MapTiler",
    osm: "OpenStreetMap",
  };
  const name = names[provider] || "CARTO Voyager";
  const note = provider === "maptiler" ? "" : " (no key)";
  el.basemapStatus.textContent = `Basemap: ${name}${note}`;
}

function applyBasemap() {
  if (!map || basemapApplying) return;
  basemapApplying = true;
  const key = String(el.maptilerKey.value || "").trim();
  const provider = normalizeBasemapProvider(el.basemapProvider.value, key);
  const hasKey = Boolean(key);
  applyBasemapStatus(provider, hasKey);
  if (provider !== el.basemapProvider.value) {
    el.basemapProvider.value = provider;
  }
  const center = map.getCenter();
  const zoom = map.getZoom();
  map.setStyle(buildBasemapStyle(provider, key));
  map.once("style.load", () => {
    if (center) map.setCenter(center);
    if (Number.isFinite(zoom)) map.setZoom(zoom);
    if (el.showHeatmap.checked) {
      loadHeatmap()
        .catch(() => setStatus("Heatmap unavailable for this region."))
        .finally(() => {
          if (lastRouteGeojson) scheduleRouteRender(lastRouteGeojson);
        });
    } else if (lastRouteGeojson) {
      scheduleRouteRender(lastRouteGeojson);
    }
    basemapApplying = false;
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
  if (el.regionBadge) {
    el.regionBadge.textContent = region?.is_default ? "Primary" : "Region";
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
  const storedKey = localStorage.getItem(STORAGE_KEYS.maptilerKey) || "";
  let storedProvider = localStorage.getItem(STORAGE_KEYS.basemapProvider) || "";
  if (!storedKey && (!storedProvider || storedProvider === "osm")) {
    storedProvider = "carto_voyager";
    localStorage.setItem(STORAGE_KEYS.basemapProvider, storedProvider);
  }
  if (storedKey) el.maptilerKey.value = storedKey;
  const provider = normalizeBasemapProvider(storedProvider, storedKey);
  el.basemapProvider.value = provider;
  applyBasemapStatus(provider, Boolean(storedKey));
  map = new maplibregl.Map({
    container: "map",
    style: buildBasemapStyle(provider, storedKey),
    center: [-75.2, 40.045],
    zoom: 12,
  });
}

async function geocodeMany(query) {
  const resp = await fetch(
    `${api("/v1/geocode")}?q=${encodeURIComponent(query)}&region=${encodeURIComponent(el.region.value)}`
  );
  if (!resp.ok) {
    let detail = "";
    try {
      const payload = await resp.json();
      detail = payload?.detail ? ` (${payload.detail})` : "";
    } catch {
      detail = "";
    }
    setStatus(`Geocoding failed${detail}. Check API base and MAPBOX_ACCESS_TOKEN.`);
    return [];
  }
  const payload = await resp.json();
  return payload.results || [];
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
    planRoute().catch((err) => setStatus(String(err)));
  }
}

async function resolvePoint(which) {
  const isStart = which === "start";
  const input = isStart ? el.startAddress : el.endAddress;
  const cached = suggestState[which].selected;
  if (cached && input.value === (cached.label || cached.latlon || `${cached.lat},${cached.lon}`)) {
    return { lat: Number(cached.lat), lon: Number(cached.lon), label: cached.label || cached.latlon };
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
  setStatus("Swapped start and destination.");
}

function clearRoute() {
  lastRouteGeojson = null;
  activeRouteKind = null;
  if (map?.getSource(routeSourceId)) {
    map.getSource(routeSourceId).setData({ type: "FeatureCollection", features: [] });
  }
  el.routeList.innerHTML = '<div class="route-empty">Plan a route to compare options.</div>';
  setStatus("Route cleared.");
}

async function loadHeatmap() {
  const resp = await fetch(
    `${api("/v1/heatmap")}?region=${encodeURIComponent(el.region.value)}&max_points=3000&max_tiles=0`
  );
  if (!resp.ok) return;
  const payload = await resp.json();
  const geojson = payload.geojson || { type: "FeatureCollection", features: [] };
  const tileZoom = Number(payload.tile_zoom || 16);
  heatmapZoomMode = tileZoom <= 14 ? "z14" : "z16";
  const blurStart = heatmapZoomMode === "z14" ? 6.4 : 8.0;
  const blurMid = heatmapZoomMode === "z14" ? 9.8 : 11.2;
  const blurEnd = heatmapZoomMode === "z14" ? 13.6 : 14.8;
  if (!map.getSource(heatmapSourceId)) {
    map.addSource(heatmapSourceId, { type: "geojson", data: geojson });
    map.addLayer({
      id: heatmapLayerId,
      type: "heatmap",
      source: heatmapSourceId,
      maxzoom: blurEnd,
      paint: {
        "heatmap-weight": [
          "interpolate",
          ["linear"],
          ["get", "score_norm"],
          0,
          0.04,
          0.25,
          0.28,
          0.55,
          0.72,
          1,
          1,
        ],
        "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 6, 0.55, 10, 0.85, 13, 1.1],
        "heatmap-color": [
          "interpolate",
          ["linear"],
          ["heatmap-density"],
          0,
          "rgba(65, 93, 112, 0)",
          0.08,
          "rgba(63, 128, 106, 0.22)",
          0.22,
          "rgba(92, 149, 107, 0.38)",
          0.46,
          "rgba(207, 158, 76, 0.52)",
          1,
          "rgba(190, 96, 62, 0.62)",
        ],
        "heatmap-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          5,
          26,
          8,
          42,
          10,
          64,
          12,
          86,
          14,
          110,
        ],
        "heatmap-opacity": [
          "interpolate",
          ["linear"],
          ["zoom"],
          blurStart,
          0.18,
          blurMid,
          0.3,
          blurEnd,
          0.36,
        ],
      },
    });
    if (map.getLayer("baseline-line")) map.moveLayer("baseline-line");
    if (map.getLayer("scenic-line")) map.moveLayer("scenic-line");
  } else {
    map.getSource(heatmapSourceId).setData(geojson);
    map.setLayoutProperty(heatmapLayerId, "visibility", "visible");
  }
}

function hideHeatmap() {
  if (map.getLayer(heatmapLayerId)) {
    map.setLayoutProperty(heatmapLayerId, "visibility", "none");
  }
  if (map.getLayer(heatmapTilesLayerId)) {
    map.setLayoutProperty(heatmapTilesLayerId, "visibility", "none");
  }
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
  if (!activeRouteKind && scenic) {
    setActiveRoute("scenic");
  } else if (!activeRouteKind && baseline) {
    setActiveRoute("baseline");
  } else if (activeRouteKind) {
    setActiveRoute(activeRouteKind);
  }
}

function syncRoutePanelState() {
  const collapsed = localStorage.getItem(STORAGE_KEYS.routePanelCollapsed) === "1";
  el.routePanel.classList.toggle("collapsed", collapsed);
  el.routeCollapseBtn.textContent = collapsed ? "Expand" : "Minimize";
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 980px)").matches;
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
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === next);
  });
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
  setActiveTab("search", { persist: false });
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
      planRoute().catch((err) => setStatus(String(err)));
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
          setStatus("Saved trip deleted.");
          return;
        }
        if (action === "rename") {
          const name = prompt("Rename saved trip:", trip.title);
          if (!name) return;
          trip.title = name.trim();
          writeSavedTrips(trips);
          renderSavedTrips();
          setStatus("Saved trip renamed.");
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

function renderRouteGeojson(geojson) {
  lastRouteGeojson = geojson;
  const featureCount = geojson?.features?.length || 0;
  if (!featureCount) {
    setStatus("Route returned no geometry.");
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
  if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 36, duration: 0 });
}

function scheduleRouteRender(geojson) {
  pendingRouteGeojson = geojson;
  if (!map) return;
  if (map.isStyleLoaded()) {
    renderRouteGeojson(pendingRouteGeojson);
    pendingRouteGeojson = null;
    return;
  }
  map.once("style.load", () => {
    if (!pendingRouteGeojson) return;
    renderRouteGeojson(pendingRouteGeojson);
    pendingRouteGeojson = null;
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
      setStatus("Finding locations...");
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
    setStatus("Building routes...");
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
    if (isMobileLayout()) setActiveTab("routes");
    lastRoutePayload = payload;
    if (seq === planSeq) scheduleRouteRender(payload.geojson);
    updateShareUrl(lastPlan);
    if (payload.retry_used && payload.retry_max_detour_factor) {
      setStatus(`Route ready (auto-bumped detour to ${payload.retry_max_detour_factor.toFixed(1)}).`);
    } else {
      setStatus("Route ready.");
    }
  } finally {
    if (seq === planSeq) setBusy(false);
  }
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
  let usingFallbackRegion = false;
  try {
    await loadRegions();
  } catch {
    loadFallbackRegions();
    usingFallbackRegion = true;
  }
  applyRegionDefaults();
  syncInputMode();
  syncTabState();
  applyRoutePreference(localStorage.getItem(STORAGE_KEYS.routePreference) || "balanced");
  const autoPlan = applyUrlParams();
  renderSavedTrips();
  el.routeList.innerHTML = '<div class="route-empty">Plan a route to compare options.</div>';
  map.once("load", () => {
    if (el.showHeatmap.checked) {
      loadHeatmap().catch(() => setStatus("Scenic layer unavailable for this region."));
    }
  });
  if (autoPlan) {
    planRoute().catch((err) => setStatus(String(err)));
  }

  el.refreshRegionsBtn.addEventListener("click", async () => {
    await checkHealth();
    await loadRegions();
    applyRegionDefaults();
    setStatus("Regions refreshed.");
  });
  el.region.addEventListener("change", () => {
    applyRegionDefaults();
    if (el.showHeatmap.checked && map.loaded()) {
      loadHeatmap().catch(() => setStatus("Heatmap unavailable for this region."));
    }
    setStatus(`Region set to ${el.region.value}.`);
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
    if (el.showHeatmap.checked) loadHeatmap().catch(() => setStatus("Heatmap unavailable for this region."));
    else hideHeatmap();
  });
  el.planBtn.addEventListener("click", () => {
    planRoute().catch((err) => setStatus(String(err)));
  });
  el.saveRouteBtn.addEventListener("click", () => {
    if (!lastPlan) {
      setStatus("Plan a route before saving.");
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
    setStatus("Route saved.");
  });
  el.shareRouteBtn.addEventListener("click", async () => {
    if (!lastPlan) {
      setStatus("Plan a route before sharing.");
      return;
    }
    const url = buildShareUrl(lastPlan);
    const original = el.shareRouteBtn.textContent;
    try {
      await navigator.clipboard.writeText(url);
      el.shareRouteBtn.textContent = "Copied";
      setStatus("Share link copied to clipboard.");
      setTimeout(() => {
        el.shareRouteBtn.textContent = original;
      }, 1200);
    } catch {
      el.shareRouteBtn.textContent = "Copied";
      setStatus(`Share link ready: ${url}`);
      setTimeout(() => {
        el.shareRouteBtn.textContent = original;
      }, 1800);
    }
  });
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const current = document.body.dataset.activeTab || "search";
      if (btn.dataset.tab === current) {
        setActiveTab("none");
      } else {
        setActiveTab(btn.dataset.tab);
      }
    });
  });
  if (el.sheetToggleBtn) {
    el.sheetToggleBtn.addEventListener("click", () => {
      const open = !el.controls.classList.contains("sheet-open");
      setSheetOpen(open);
    });
  }
  el.routeCollapseBtn.addEventListener("click", () => {
    const collapsed = el.routePanel.classList.toggle("collapsed");
    localStorage.setItem(STORAGE_KEYS.routePanelCollapsed, collapsed ? "1" : "0");
    el.routeCollapseBtn.textContent = collapsed ? "Expand" : "Minimize";
  });
  el.basemapProvider.addEventListener("change", () => {
    localStorage.setItem(STORAGE_KEYS.basemapProvider, el.basemapProvider.value);
    applyBasemap();
  });
  el.maptilerKey.addEventListener("input", () => {
    const value = String(el.maptilerKey.value || "").trim();
    if (value) localStorage.setItem(STORAGE_KEYS.maptilerKey, value);
    else localStorage.removeItem(STORAGE_KEYS.maptilerKey);
  });
  el.maptilerKey.addEventListener("blur", () => {
    applyBasemap();
  });

  installAddressInput("start");
  installAddressInput("end");
  syncRoutePanelState();
  setStatus(usingFallbackRegion ? "API unavailable. Showing Masswhites shell." : "Ready");
}

main().catch((err) => setStatus(String(err)));
