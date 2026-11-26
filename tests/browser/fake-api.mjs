// Ephemeral, deterministic fixture server for the browser smoke lane.
//
// Serves apps/new_england_north (index.html, viewer.js, styles.css) and a fake
// `/api/v1/*` that mirrors the response shapes viewer.js consumes. It binds to
// 127.0.0.1 on an OS-assigned port and lives only for one Playwright worker.
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const APP_DIR = path.resolve(HERE, "../../apps/new_england_north");

export const RUN_NAME = "prompt_two_candidate_exp02_fresh_test20_20260810";
export const REGION = "new_england_north";
export const BBOX = Object.freeze({ min_lat: 42.48, min_lon: -73.52, max_lat: 47.5, max_lon: -66.79 });
const TILE_ZOOM = 7;
// Snapped road positions sit this far (degrees) from the requested points so
// the viewer draws its dashed endpoint connectors.
const SNAP_OFFSET = 0.01;

export const REGIONS_PAYLOAD = Object.freeze({
  regions: [
    {
      region: REGION,
      display_name: "New England North",
      latest_run_name: RUN_NAME,
      graph_exists: true,
      route_planning_enabled: true,
      is_default: true,
      bbox: BBOX,
      map: { center: { lat: 44.99, lon: -70.15 }, zoom: 6.2 },
    },
  ],
});

export const ADDRESS_FIXTURES = Object.freeze([
  { mapbox_id: "fixture.portland-me", name: "Portland", full_address: "Portland, Maine, United States", lat: 43.6591, lon: -70.2568 },
  { mapbox_id: "fixture.portsmouth-nh", name: "Portsmouth", full_address: "Portsmouth, New Hampshire, United States", lat: 43.0718, lon: -70.7626 },
  { mapbox_id: "fixture.port-clyde-me", name: "Port Clyde", full_address: "Port Clyde, Maine, United States", lat: 43.927, lon: -69.2559 },
]);

const VALIDATED_REQUEST = Object.freeze({
  start: { lat: 44.475884, lon: -73.214003 },
  end: { lat: 44.801616, lon: -68.771305 },
  max_detour_factor: 1.8,
  avoid_highways: true,
});

// ---------------------------------------------------------------- PNG fixtures
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buffer) {
  let c = 0xffffffff;
  for (const byte of buffer) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngChunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
}
/** Solid RGBA PNG; deterministic bytes for a given size/colour. */
export function solidPng(width, height, [r, g, b, a]) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8; // bit depth
  header[9] = 6; // RGBA
  const row = Buffer.alloc(1 + width * 4);
  for (let x = 0; x < width; x += 1) row.set([r, g, b, a], 1 + x * 4);
  const raw = Buffer.concat(Array.from({ length: height }, () => row));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", header),
    pngChunk("IDAT", zlib.deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}
// Neutral grey basemap and a translucent blue heatmap: neither contains the
// scenic route green, so screenshot pixels prove the route layer rendered.
export const BASEMAP_TILE_PNG = solidPng(256, 256, [232, 232, 232, 255]);
const HEATMAP_PNG = solidPng(16, 16, [59, 76, 192, 150]);

// ------------------------------------------------------------- route fixtures
function lngLatToTile(lng, lat, zoom) {
  const scale = 2 ** zoom;
  return {
    x: Math.floor(((lng + 180) / 360) * scale),
    y: Math.floor(((1 - Math.asinh(Math.tan((lat * Math.PI) / 180)) / Math.PI) / 2) * scale),
  };
}
const GRID_MIN = lngLatToTile(BBOX.min_lon, BBOX.max_lat, TILE_ZOOM);
const GRID_MAX = lngLatToTile(BBOX.max_lon, BBOX.min_lat, TILE_ZOOM);
const GRID_WIDTH = GRID_MAX.x - GRID_MIN.x + 1;
const GRID_HEIGHT = GRID_MAX.y - GRID_MIN.y + 1;
function scoreGrid() {
  const values = Buffer.alloc(GRID_WIDTH * GRID_HEIGHT * 4);
  for (let i = 0; i < GRID_WIDTH * GRID_HEIGHT; i += 1) values.writeFloatLE(5 + (i % 7) * 0.5, i * 4);
  return values;
}

function haversineKm(a, b) {
  const rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad;
  const dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}
const round = (value, digits) => Number(value.toFixed(digits));

function inCoverage(point) {
  return point.lat >= BBOX.min_lat && point.lat <= BBOX.max_lat && point.lon >= BBOX.min_lon && point.lon <= BBOX.max_lon;
}

/** Deterministic comparison payload derived only from the request body. */
export function routeComparison(request) {
  const snappedStart = [round(request.start.lon + SNAP_OFFSET, 6), round(request.start.lat, 6)];
  const snappedEnd = [round(request.end.lon - SNAP_OFFSET, 6), round(request.end.lat, 6)];
  const bend = [
    round((snappedStart[0] + snappedEnd[0]) / 2, 6),
    round((snappedStart[1] + snappedEnd[1]) / 2 + 0.25, 6),
  ];
  const baselineKm = round(haversineKm(request.start, request.end) * 1.2, 3);
  const scenicKm = round(baselineKm * 1.15, 3);
  const baselineMin = round((baselineKm / 80) * 60, 2);
  const scenicMin = round((scenicKm / 72) * 60, 2);
  const scenicScore = 7.42;
  const baselineScore = 6.18;
  const properties = { snapped_start: [snappedStart[1], snappedStart[0]], snapped_end: [snappedEnd[1], snappedEnd[0]] };
  return {
    request,
    geojson: {
      type: "FeatureCollection",
      features: [
        { type: "Feature", properties: { route_kind: "baseline", ...properties }, geometry: { type: "LineString", coordinates: [snappedStart, snappedEnd] } },
        { type: "Feature", properties: { route_kind: "scenic", ...properties }, geometry: { type: "LineString", coordinates: [snappedStart, bend, snappedEnd] } },
      ],
    },
    routes: {
      scenic: { total_distance_km: scenicKm, estimated_duration_minutes: scenicMin, raw_scenic_score: scenicScore, normalized_scenic_score: 0.7421, highway_count: 0, score_run: ["e1", "e2", "e3"] },
      baseline: { total_distance_km: baselineKm, estimated_duration_minutes: baselineMin, raw_scenic_score: baselineScore, normalized_scenic_score: 0.6183, highway_count: 2, score_run: ["e1", "e4"] },
    },
    deltas: {
      distance_km: round(scenicKm - baselineKm, 3),
      duration_min: round(scenicMin - baselineMin, 2),
      scenic_score_absolute: round(scenicScore - baselineScore, 2),
    },
    score_mapping: { normalization: "global_minmax" },
    diagnostics: {
      planning_elapsed_ms: 42,
      requested_max_detour_factor: request.max_detour_factor,
      applied_max_detour_factor: request.max_detour_factor,
      scenic_fastest_duration_ratio: round(scenicMin / baselineMin, 4),
      optimization_mode: "approximate",
      optimization_status: "feasible",
      optimality_gap: 0.0123,
      certified_upper_bound: 7.51,
      same_route: false,
      no_better_route_reason: null,
      scenic_score_delta_absolute: round(scenicScore - baselineScore, 2),
      scenic_score_delta_relative: round((scenicScore - baselineScore) / baselineScore, 4),
      avoid_highways_applied: request.avoid_highways,
      search_diagnostics: { mode: "label_setting", time_limit_seconds: 2, labels_generated: 1200, labels_expanded: 800, labels_pruned: 300, max_frontier_size: 90, remaining_frontier_size: 0, elapsed_ms: 40, deadline_reached: false },
    },
  };
}

// --------------------------------------------------------------------- server
function sendJson(res, status, payload) {
  const body = Buffer.from(JSON.stringify(payload));
  res.writeHead(status, { "Content-Type": "application/json", "Content-Length": body.length, "Cache-Control": "no-store" });
  res.end(body);
}
async function readJson(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}
function validPoint(point) {
  return point && Number.isFinite(point.lat) && Number.isFinite(point.lon);
}

async function handleApi(req, res, url) {
  const route = `${req.method} ${url.pathname.replace(/^\/api/, "")}`;
  switch (route) {
    case "GET /v1/healthz":
      return sendJson(res, 200, { status: "ok" });
    case "GET /v1/regions":
      return sendJson(res, 200, REGIONS_PAYLOAD);
    case "GET /v1/training-results":
      return sendJson(res, 200, {
        run_name: RUN_NAME,
        checkpoint: `runs/${RUN_NAME}/best.pt`,
        updated_at: "2026-08-10T12:00:00Z",
        metrics: { corr: 0.812, mae: 0.534, rmse: 0.701, samples: 4096 },
      });
    case "GET /v1/heatmap":
      return sendJson(res, 200, {
        run_name: RUN_NAME,
        tile_zoom: TILE_ZOOM,
        total_tiles: GRID_WIDTH * GRID_HEIGHT,
        bounds: BBOX,
        normalization: { total_tiles: GRID_WIDTH * GRID_HEIGHT, mean: 6.1, median: 6, std: 1.1, min: 1, max: 9.6 },
        points: [],
      });
    case "GET /v1/heatmap-image":
      res.writeHead(200, { "Content-Type": "image/png", "Content-Length": HEATMAP_PNG.length });
      return res.end(HEATMAP_PNG);
    case "GET /v1/heatmap-scores.bin": {
      const values = scoreGrid();
      res.writeHead(200, {
        "Content-Type": "application/octet-stream",
        "Content-Length": values.length,
        "X-Scenic-Tile-Zoom": String(TILE_ZOOM),
        "X-Scenic-Min-X": String(GRID_MIN.x),
        "X-Scenic-Min-Y": String(GRID_MIN.y),
        "X-Scenic-Grid-Width": String(GRID_WIDTH),
        "X-Scenic-Grid-Height": String(GRID_HEIGHT),
      });
      return res.end(values);
    }
    case "GET /v1/validated-route":
      return sendJson(res, 200, routeComparison(VALIDATED_REQUEST));
    case "GET /v1/search/suggest": {
      const query = (url.searchParams.get("q") || "").trim().toLowerCase();
      const suggestions = ADDRESS_FIXTURES.filter((row) => row.name.toLowerCase().startsWith(query)).map(
        ({ mapbox_id, name, full_address }) => ({ mapbox_id, name, full_address, feature_type: "place" })
      );
      return sendJson(res, 200, { suggestions });
    }
    case "GET /v1/search/retrieve": {
      const row = ADDRESS_FIXTURES.find((item) => item.mapbox_id === url.searchParams.get("mapbox_id"));
      if (!row) return sendJson(res, 404, { detail: "Address not found" });
      return sendJson(res, 200, { result: { name: row.name, full_address: row.full_address, lat: row.lat, lon: row.lon } });
    }
    case "POST /v1/route/compare": {
      const body = await readJson(req);
      if (body.region !== REGION || body.run_name !== RUN_NAME || !validPoint(body.start) || !validPoint(body.end)) {
        return sendJson(res, 422, { detail: { error: "invalid_request", message: "Route request is invalid." } });
      }
      for (const endpoint of ["start", "end"]) {
        if (!inCoverage(body[endpoint])) {
          // Same shape as app_api.main's RouteCoverageError handler.
          return sendJson(res, 422, {
            detail: {
              error: "route_endpoint_outside_coverage",
              message: `The ${endpoint} point is too far from the supported road network.`,
              hint: "Choose a point within the selected region's route coverage.",
              reason: `${endpoint} snap distance 212.400 km exceeds 5.000 km`,
              endpoint,
              snap_distance_km: 212.4,
              max_snap_distance_km: 5,
            },
          });
        }
      }
      return sendJson(res, 200, routeComparison(body));
    }
    default:
      return sendJson(res, 404, { detail: "Not Found" });
  }
}

const STATIC_TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8" };
const STATIC_FILES = new Map([
  ["/", "index.html"],
  ["/index.html", "index.html"],
  ["/viewer.js", "viewer.js"],
  ["/styles.css", "styles.css"],
]);

async function handleStatic(res, url) {
  const name = STATIC_FILES.get(url.pathname);
  if (!name) {
    res.writeHead(404, { "Content-Type": "text/plain" });
    return res.end("Not Found");
  }
  const body = await fs.readFile(path.join(APP_DIR, name));
  res.writeHead(200, { "Content-Type": STATIC_TYPES[path.extname(name)], "Content-Length": body.length, "Cache-Control": "no-store" });
  return res.end(body);
}

/** Start the fixture server on 127.0.0.1:<ephemeral>; resolves to { origin, close }. */
export async function startFakeServer() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://127.0.0.1");
    const task = url.pathname.startsWith("/api/") ? handleApi(req, res, url) : handleStatic(res, url);
    task.catch((error) => {
      if (!res.headersSent) sendJson(res, 500, { detail: String(error?.message || error) });
      else res.destroy(error);
    });
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const { port } = server.address();
  return {
    origin: `http://127.0.0.1:${port}`,
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}
