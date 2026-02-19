const el = {
  apiBase: document.getElementById("apiBase"),
  region: document.getElementById("region"),
  inputMode: document.getElementById("inputMode"),
  startLat: document.getElementById("startLat"),
  startLon: document.getElementById("startLon"),
  endLat: document.getElementById("endLat"),
  endLon: document.getElementById("endLon"),
  startAddress: document.getElementById("startAddress"),
  endAddress: document.getElementById("endAddress"),
  startSuggestions: document.getElementById("startSuggestions"),
  endSuggestions: document.getElementById("endSuggestions"),
  scenicWeight: document.getElementById("scenicWeight"),
  weightValue: document.getElementById("weightValue"),
  planBtn: document.getElementById("planBtn"),
  status: document.getElementById("status"),
  metrics: document.getElementById("metrics"),
};

function requireElement(name, node) {
  if (!node) {
    throw new Error(`Missing required UI element: ${name}`);
  }
}

let map;
let routeSourceId = "route-data";
let startSuggestTimer = null;
let endSuggestTimer = null;
let regionsMeta = [];

function setStatus(msg) {
  el.status.textContent = msg;
}

function api(path) {
  return `${el.apiBase.value.replace(/\/+$/, "")}${path}`;
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
  if (!regions.length) {
    const opt = document.createElement("option");
    opt.value = "philadelphia";
    opt.textContent = "philadelphia";
    el.region.appendChild(opt);
  }
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

function applyRegionDefaults() {
  const key = String(el.region.value || "").toLowerCase();
  const region = regionsMeta.find((r) => String(r.region || "").toLowerCase() === key);
  const bbox = region?.bbox;
  if (!bbox) {
    // Fallback to generic values if bbox metadata is unavailable.
    el.startLat.value = "40.000000";
    el.startLon.value = "-75.000000";
    el.endLat.value = "40.010000";
    el.endLon.value = "-74.990000";
  } else {
    const minLat = Number(bbox.min_lat);
    const minLon = Number(bbox.min_lon);
    const maxLat = Number(bbox.max_lat);
    const maxLon = Number(bbox.max_lon);
    const latSpan = Math.max(0.0005, maxLat - minLat);
    const lonSpan = Math.max(0.0005, maxLon - minLon);
    const startLat = minLat + latSpan * 0.25;
    const startLon = minLon + lonSpan * 0.25;
    const endLat = minLat + latSpan * 0.75;
    const endLon = minLon + lonSpan * 0.75;
    el.startLat.value = startLat.toFixed(6);
    el.startLon.value = startLon.toFixed(6);
    el.endLat.value = endLat.toFixed(6);
    el.endLon.value = endLon.toFixed(6);
  }
  // In address mode, default to canonical point syntax (auto-parsed by /v1/geocode).
  if (el.startAddress) el.startAddress.value = `${el.startLat.value},${el.startLon.value}`;
  if (el.endAddress) el.endAddress.value = `${el.endLat.value},${el.endLon.value}`;
}

function syncInputMode() {
  const addr = el.inputMode.value === "address";
  document.querySelectorAll(".coords-only").forEach((n) => n.classList.toggle("hidden", addr));
  document.querySelectorAll(".addr-only").forEach((n) => n.classList.toggle("hidden", !addr));
}

async function geocodeOne(query) {
  const resp = await fetch(
    `${api("/v1/geocode")}?q=${encodeURIComponent(query)}&region=${encodeURIComponent(el.region.value)}`
  );
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`geocode failed: ${resp.status} ${text}`);
  }
  const payload = await resp.json();
  const first = (payload.results || [])[0];
  if (!first) throw new Error(`No geocoding result for: ${query}`);
  return { provider: payload.provider, ...first };
}

async function geocodeMany(query) {
  const resp = await fetch(
    `${api("/v1/geocode")}?q=${encodeURIComponent(query)}&region=${encodeURIComponent(el.region.value)}`
  );
  if (!resp.ok) return [];
  const payload = await resp.json();
  return payload.results || [];
}

function renderSuggestions(listEl, results) {
  if (!listEl) return;
  listEl.innerHTML = "";
  for (const row of results) {
    const opt = document.createElement("option");
    opt.value = row.label || row.latlon || `${row.lat},${row.lon}`;
    listEl.appendChild(opt);
  }
}

function debounceSuggest(which) {
  const isStart = which === "start";
  const inputEl = isStart ? el.startAddress : el.endAddress;
  const listEl = isStart ? el.startSuggestions : el.endSuggestions;
  const query = String(inputEl.value || "").trim();
  if (isStart && startSuggestTimer) clearTimeout(startSuggestTimer);
  if (!isStart && endSuggestTimer) clearTimeout(endSuggestTimer);

  const timer = setTimeout(async () => {
    if (query.length < 3) {
      renderSuggestions(listEl, []);
      return;
    }
    try {
      const rows = await geocodeMany(query);
      renderSuggestions(listEl, rows);
    } catch {
      renderSuggestions(listEl, []);
    }
  }, 220);
  if (isStart) startSuggestTimer = timer;
  else endSuggestTimer = timer;
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
    .map(
      ([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`
    )
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
      paint: { "line-color": "#66c2ff", "line-width": 4, "line-opacity": 0.9 },
    });
    map.addLayer({
      id: "scenic-line",
      type: "line",
      source: routeSourceId,
      filter: ["!=", ["get", "route_kind"], "baseline"],
      paint: { "line-color": "#ffb347", "line-width": 6, "line-opacity": 0.95 },
    });
  } else {
    map.getSource(routeSourceId).setData(geojson);
  }

  const bounds = new maplibregl.LngLatBounds();
  for (const feature of geojson.features || []) {
    const coords = feature?.geometry?.coordinates || [];
    for (const [lon, lat] of coords) bounds.extend([lon, lat]);
  }
  if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 30, duration: 0 });
}

async function planRoute() {
  setStatus("Planning route...");
  let start;
  let end;
  if (el.inputMode.value === "address") {
    setStatus("Geocoding addresses...");
    const [s, e] = await Promise.all([geocodeOne(el.startAddress.value), geocodeOne(el.endAddress.value)]);
    start = { lat: Number(s.lat), lon: Number(s.lon) };
    end = { lat: Number(e.lat), lon: Number(e.lon) };
    // Normalize to the matched official label/syntax.
    el.startAddress.value = s.label || s.latlon || el.startAddress.value;
    el.endAddress.value = e.label || e.latlon || el.endAddress.value;
    el.startLat.value = String(start.lat);
    el.startLon.value = String(start.lon);
    el.endLat.value = String(end.lat);
    el.endLon.value = String(end.lon);
    setStatus(
      `Matched start=${s.latlon || `${s.lat},${s.lon}`} (${s.provider || "geo"})\n` +
        `Matched end=${e.latlon || `${e.lat},${e.lon}`} (${e.provider || "geo"})`
    );
  } else {
    start = { lat: Number(el.startLat.value), lon: Number(el.startLon.value) };
    end = { lat: Number(el.endLat.value), lon: Number(el.endLon.value) };
  }
  const body = {
    start,
    end,
    scenic_weight: Number(el.scenicWeight.value),
    region: el.region.value,
    max_detour_factor: 1.8,
    avoid_highways: false,
    include_baseline: true,
  };
  const resp = await fetch(api("/v1/route/compare"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = "";
    try {
      const json = await resp.json();
      detail = json.detail ? ` ${JSON.stringify(json.detail)}` : ` ${JSON.stringify(json)}`;
    } catch {
      detail = ` ${await resp.text()}`;
    }
    throw new Error(`route compare failed: ${resp.status}${detail}`);
  }
  const payload = await resp.json();
  renderMetrics(payload);
  const scenicSeg = Number(payload.routes?.scenic?.segments || 0);
  const baseSeg = Number(payload.routes?.baseline?.segments || 0);
  if (scenicSeg === 0 && baseSeg === 0) {
    setStatus(
      `No route found for current region/points.\n` +
      `Try changing region or choosing points closer together.`
    );
    return;
  }
  if (map.loaded()) renderRouteGeojson(payload.geojson);
  else map.once("load", () => renderRouteGeojson(payload.geojson));
  const ratio = payload.score_mapping?.matched_ratio ?? 0;
  setStatus(
    `OK • region=${payload.request?.graph_geojson ? el.region.value : "?"} • matched=${(ratio * 100).toFixed(1)}%\n` +
      `Start=${start.lat.toFixed(6)},${start.lon.toFixed(6)} End=${end.lat.toFixed(6)},${end.lon.toFixed(6)}`
  );
}

async function main() {
  requireElement("apiBase", el.apiBase);
  requireElement("region", el.region);
  requireElement("inputMode", el.inputMode);
  requireElement("startLat", el.startLat);
  requireElement("startLon", el.startLon);
  requireElement("endLat", el.endLat);
  requireElement("endLon", el.endLon);
  requireElement("startAddress", el.startAddress);
  requireElement("endAddress", el.endAddress);
  requireElement("startSuggestions", el.startSuggestions);
  requireElement("endSuggestions", el.endSuggestions);
  requireElement("planBtn", el.planBtn);
  requireElement("status", el.status);

  initMap();
  await loadRegions();
  applyRegionDefaults();
  syncInputMode();
  el.region.addEventListener("change", () => {
    applyRegionDefaults();
    setStatus(`Region changed to ${el.region.value}. Defaults applied.`);
  });
  el.weightValue.textContent = el.scenicWeight.value;
  el.inputMode.addEventListener("change", syncInputMode);
  el.scenicWeight.addEventListener("input", () => {
    el.weightValue.textContent = el.scenicWeight.value;
  });
  el.startAddress.addEventListener("input", () => debounceSuggest("start"));
  el.endAddress.addEventListener("input", () => debounceSuggest("end"));
  el.planBtn.addEventListener("click", () => {
    planRoute().catch((err) => setStatus(String(err)));
  });
  setStatus("Ready");
}

main().catch((err) => setStatus(String(err)));
