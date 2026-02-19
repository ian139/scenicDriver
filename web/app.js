const el = {
  controls: document.querySelector(".controls"),
  apiBase: document.getElementById("apiBase"),
  region: document.getElementById("region"),
  modeAddressBtn: document.getElementById("modeAddressBtn"),
  modeCoordsBtn: document.getElementById("modeCoordsBtn"),
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
  clearBtn: document.getElementById("clearBtn"),
  status: document.getElementById("status"),
  metrics: document.getElementById("metrics"),
};

let map;
const routeSourceId = "route-data";
const heatmapSourceId = "scenic-heatmap";
const heatmapLayerId = "scenic-heatmap-layer";
const heatmapCircleLayerId = "scenic-heatmap-circle";
const heatmapTilesSourceId = "scenic-heatmap-tiles";
const heatmapTilesLayerId = "scenic-heatmap-tiles-layer";
let heatmapZoomMode = "z16";
let regionsMeta = [];
let busy = false;
let inputMode = "address";
const suggestTimer = { start: null, end: null };
const suggestState = {
  start: { items: [], active: -1, selected: null },
  end: { items: [], active: -1, selected: null },
};
const suggestSeq = { start: 0, end: 0 };

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
    opt.textContent = r.region;
    el.region.appendChild(opt);
  }
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

function initMap() {
  map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      sources: {
        osm: {
          type: "raster",
          tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
          tileSize: 256,
        },
      },
      layers: [{ id: "osm", type: "raster", source: "osm" }],
    },
    center: [-75.2, 40.045],
    zoom: 12,
  });
}

async function geocodeMany(query) {
  const resp = await fetch(
    `${api("/v1/geocode")}?q=${encodeURIComponent(query)}&region=${encodeURIComponent(el.region.value)}`
  );
  if (!resp.ok) return [];
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
    const rows = await geocodeMany(query);
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
  if (map?.getSource(routeSourceId)) {
    map.getSource(routeSourceId).setData({ type: "FeatureCollection", features: [] });
  }
  el.metrics.innerHTML = "";
  setStatus("Route cleared.");
}

async function loadHeatmap() {
  const resp = await fetch(`${api("/v1/heatmap")}?region=${encodeURIComponent(el.region.value)}&max_points=3000`);
  if (!resp.ok) return;
  const payload = await resp.json();
  const geojson = payload.geojson || { type: "FeatureCollection", features: [] };
  const geojsonTiles = payload.geojson_tiles || { type: "FeatureCollection", features: [] };
  const tileZoom = Number(payload.tile_zoom || 16);
  heatmapZoomMode = tileZoom <= 14 ? "z14" : "z16";
  const tileMinZoom = heatmapZoomMode === "z14" ? 9.8 : 11.2;
  const tileFadeMid = heatmapZoomMode === "z14" ? 10.8 : 12.2;
  const tileFadeEnd = heatmapZoomMode === "z14" ? 11.8 : 13.2;
  const circleFadeStart = heatmapZoomMode === "z14" ? 9.0 : 10.8;
  const circleFadeMid = heatmapZoomMode === "z14" ? 10.4 : 12.4;
  const circleFadeEnd = heatmapZoomMode === "z14" ? 12.2 : 13.8;
  if (!map.getSource(heatmapSourceId)) {
    map.addSource(heatmapSourceId, { type: "geojson", data: geojson });
    map.addSource(heatmapTilesSourceId, { type: "geojson", data: geojsonTiles });
    map.addLayer({
      id: heatmapLayerId,
      type: "heatmap",
      source: heatmapSourceId,
      maxzoom: 0.1,
      paint: {
        "heatmap-weight": ["interpolate", ["linear"], ["get", "score_norm"], 0, 0, 1, 0.75],
        "heatmap-intensity": 0.35,
        "heatmap-color": [
          "interpolate",
          ["linear"],
          ["heatmap-density"],
          0,
          "rgba(33,102,172,0)",
          0.30,
          "#4aa3ff",
          0.60,
          "#7fd34e",
          1.0,
          "#ffd24a",
        ],
        "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 5, 14, 8, 18, 10, 24],
        "heatmap-opacity": 0.45,
      },
    });
    map.addLayer({
      id: heatmapCircleLayerId,
      type: "circle",
      source: heatmapSourceId,
      minzoom: 0,
      maxzoom: 13.6,
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 14, 8, 16, 12, 18, 13.6, 20],
        "circle-color": [
          "interpolate",
          ["linear"],
          ["get", "score_norm"],
          0,
          "#4aa3ff",
          0.5,
          "#7fd34e",
          0.8,
          "#ffd24a",
          1,
          "#ff4f3d",
        ],
        "circle-opacity": ["interpolate", ["linear"], ["zoom"], circleFadeStart, 0.42, circleFadeMid, 0.28, circleFadeEnd, 0.0],
        "circle-blur": 0.82,
      },
    });
    map.addLayer({
      id: heatmapTilesLayerId,
      type: "fill",
      source: heatmapTilesSourceId,
      minzoom: tileMinZoom,
      paint: {
        "fill-color": [
          "interpolate",
          ["linear"],
          ["get", "score_norm"],
          0,
          "#4aa3ff",
          0.5,
          "#7fd34e",
          0.8,
          "#ffd24a",
          1,
          "#ff4f3d",
        ],
        "fill-opacity": ["interpolate", ["linear"], ["zoom"], tileMinZoom, 0.0, tileFadeMid, 0.32, tileFadeEnd, 0.56],
        "fill-outline-color": "rgba(0,0,0,0)",
        "fill-antialias": false,
      },
    });
  } else {
    map.getSource(heatmapSourceId).setData(geojson);
    if (map.getSource(heatmapTilesSourceId)) {
      map.getSource(heatmapTilesSourceId).setData(geojsonTiles);
    }
    if (map.getLayer(heatmapCircleLayerId)) {
      map.setPaintProperty(
        heatmapCircleLayerId,
        "circle-opacity",
        ["interpolate", ["linear"], ["zoom"], circleFadeStart, 0.42, circleFadeMid, 0.28, circleFadeEnd, 0.0]
      );
    }
    if (map.getLayer(heatmapTilesLayerId)) {
      map.setPaintProperty(
        heatmapTilesLayerId,
        "fill-opacity",
        ["interpolate", ["linear"], ["zoom"], tileMinZoom, 0.0, tileFadeMid, 0.32, tileFadeEnd, 0.56]
      );
    }
    map.setLayoutProperty(heatmapLayerId, "visibility", "visible");
    if (map.getLayer(heatmapCircleLayerId)) {
      map.setLayoutProperty(heatmapCircleLayerId, "visibility", "visible");
    }
    if (map.getLayer(heatmapTilesLayerId)) {
      map.setLayoutProperty(heatmapTilesLayerId, "visibility", "visible");
    }
  }
}

function hideHeatmap() {
  if (map.getLayer(heatmapLayerId)) {
    map.setLayoutProperty(heatmapLayerId, "visibility", "none");
  }
  if (map.getLayer(heatmapCircleLayerId)) {
    map.setLayoutProperty(heatmapCircleLayerId, "visibility", "none");
  }
  if (map.getLayer(heatmapTilesLayerId)) {
    map.setLayoutProperty(heatmapTilesLayerId, "visibility", "none");
  }
}

function renderMetrics(payload) {
  const scenic = payload.routes?.scenic;
  const baseline = payload.routes?.baseline;
  const d = payload.deltas || {};
  const items = [
    ["Scenic Score", scenic ? Number(scenic.average_scenic_score).toFixed(2) : "n/a"],
    ["Baseline Score", baseline ? Number(baseline.average_scenic_score).toFixed(2) : "n/a"],
    ["Delta Scenic", Number(d.scenic_score || 0).toFixed(2)],
    ["Scenic Time (min)", scenic ? Number(scenic.estimated_duration_minutes).toFixed(1) : "n/a"],
    ["Baseline Time (min)", baseline ? Number(baseline.estimated_duration_minutes).toFixed(1) : "n/a"],
    ["Delta Time (min)", Number(d.duration_min || 0).toFixed(1)],
  ];
  el.metrics.innerHTML = items
    .map(([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");
}

function renderRouteGeojson(geojson) {
  if (!map.getSource(routeSourceId)) {
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
  } else {
    map.getSource(routeSourceId).setData(geojson);
  }

  const bounds = new maplibregl.LngLatBounds();
  for (const feature of geojson.features || []) {
    const coords = feature?.geometry?.coordinates || [];
    for (const [lon, lat] of coords) bounds.extend([lon, lat]);
  }
  if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 36, duration: 0 });
}

async function planRoute() {
  if (busy) return;
  setBusy(true);
  hideSuggestions("start");
  hideSuggestions("end");
  try {
    let start;
    let end;
    if (inputMode === "address") {
      setStatus("Finding locations...");
      const [s, e] = await Promise.all([resolvePoint("start"), resolvePoint("end")]);
      start = { lat: s.lat, lon: s.lon };
      end = { lat: e.lat, lon: e.lon };
      el.startLat.value = String(s.lat);
      el.startLon.value = String(s.lon);
      el.endLat.value = String(e.lat);
      el.endLon.value = String(e.lon);
    } else {
      start = { lat: Number(el.startLat.value), lon: Number(el.startLon.value) };
      end = { lat: Number(el.endLat.value), lon: Number(el.endLon.value) };
    }
    setStatus("Building routes...");
    const resp = await fetch(api("/v1/route/compare"), {
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
    let payload = null;
    try {
      payload = await resp.json();
    } catch {
      payload = null;
    }
    if (!resp.ok) {
      if (resp.status === 422 && payload?.detail?.error === "no_route_found") {
        throw new Error("No route found between these points in this region.");
      }
      throw new Error(`Route request failed (${resp.status}).`);
    }

    renderMetrics(payload);
    if (map.loaded()) renderRouteGeojson(payload.geojson);
    else map.once("load", () => renderRouteGeojson(payload.geojson));
    setStatus("Route ready.");
  } finally {
    setBusy(false);
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
  await loadRegions();
  applyRegionDefaults();
  syncInputMode();
  el.weightValue.textContent = el.scenicWeight.value;
  el.detourValue.textContent = el.maxDetour.value;

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

  installAddressInput("start");
  installAddressInput("end");
  setStatus("Ready");
}

main().catch((err) => setStatus(String(err)));
