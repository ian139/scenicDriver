from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
import json
import os
from io import BytesIO

import numpy as np
from PIL import Image


def build_report(
    *,
    tiles: list[dict[str, Any]],
    report_dir: str | Path,
    raw_dir: str | Path,
    run_info: dict[str, Any],
    histogram_bins: int = 20,
    thumb_size: int = 128,
    include_thumbs: bool = True,
) -> dict[str, Any]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    thumbs_dir = report_path / "thumbs"
    if include_thumbs:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        _attach_thumbnails(tiles, raw_dir, thumbs_dir, thumb_size)
    else:
        for idx, tile in enumerate(tiles):
            tile["index"] = idx

    scores = np.array([float(t["scenic_score"]) for t in tiles], dtype=np.float32)
    hist_counts, hist_edges = np.histogram(scores, bins=histogram_bins, range=(0.0, 10.0))

    summary = {
        "total_tiles": int(len(tiles)),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }

    grid = _build_grid_mapping(tiles)
    feature_summary = _build_feature_summary(tiles)

    report_json = {
        "summary": summary,
        "feature_summary": feature_summary,
        "histogram": {
            "counts": [int(c) for c in hist_counts.tolist()],
            "edges": [float(e) for e in hist_edges.tolist()],
        },
        "tiles": tiles,
        "grid": grid,
        "run_info": run_info,
    }

    with open(report_path / "report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    with open(report_path / "index.html", "w", encoding="utf-8") as f:
        f.write(_render_index_html())

    _copy_report_config(report_path)

    return report_json


def _attach_thumbnails(
    tiles: list[dict[str, Any]],
    raw_dir: str | Path,
    thumbs_dir: Path,
    thumb_size: int,
) -> None:
    raw_dir_str = str(raw_dir)
    raw_root = Path(raw_dir_str)

    s3_bucket = os.getenv("SCENIC_S3_BUCKET")
    s3_prefix = None
    s3_only = False
    if raw_dir_str.startswith("s3://"):
        s3_parts = raw_dir_str.replace("s3://", "", 1).split("/", 1)
        parsed_bucket = s3_parts[0] if s3_parts else None
        parsed_prefix = s3_parts[1] if len(s3_parts) > 1 else ""
        # Treat s3:// raw_dir as authoritative; env flag can explicitly disable.
        s3_only = os.getenv("SCENIC_S3_ONLY", "1").lower() not in ("0", "false", "no")
        s3_bucket = parsed_bucket or s3_bucket
        s3_prefix = parsed_prefix

    s3 = None
    if s3_only:
        if not s3_bucket:
            raise ValueError("SCENIC_S3_BUCKET and s3:// raw_dir required for S3-only mode.")
        import boto3

        # Reuse a single client for all thumbnail fetches.
        s3 = boto3.client("s3")

    for idx, tile in enumerate(tiles):
        image_path = raw_root / tile["image_path"]
        thumb_name = f"{idx:05d}.jpg"
        thumb_path = thumbs_dir / thumb_name

        if s3_only:
            key = f"{s3_prefix}/{tile['image_path']}" if s3_prefix else tile["image_path"]
            resp = s3.get_object(Bucket=s3_bucket, Key=key)
            image = Image.open(BytesIO(resp["Body"].read())).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        image.save(thumb_path, format="JPEG", quality=85)

        tile["index"] = idx
        tile["thumb"] = f"thumbs/{thumb_name}"


def _build_grid_mapping(tiles: list[dict[str, Any]]) -> dict[str, Any]:
    coords = [(t.get("x"), t.get("y"), t.get("z")) for t in tiles]
    has_coords = all(x is not None and y is not None and z is not None for x, y, z in coords)
    if not has_coords:
        return {"has_coords": False, "index_by_coord": {}}

    xs = [int(t["x"]) for t in tiles]
    ys = [int(t["y"]) for t in tiles]
    zs = [int(t["z"]) for t in tiles]
    zoom = zs[0] if len(set(zs)) == 1 else None

    index_by_coord = {f"{t['x']},{t['y']}": int(t["index"]) for t in tiles}

    return {
        "has_coords": True,
        "zoom": zoom,
        "min_x": int(min(xs)),
        "max_x": int(max(xs)),
        "min_y": int(min(ys)),
        "max_y": int(max(ys)),
        "index_by_coord": index_by_coord,
    }


def _build_feature_summary(tiles: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        ("relief", "Relief (m)"),
        ("roughness", "Roughness"),
        ("slope_mean", "Slope Mean"),
        ("water_proxy", "Water Proxy"),
        ("veg_proxy", "Veg Proxy"),
        ("water_fraction", "Water Fraction"),
        ("texture", "Texture"),
    ]
    summary: dict[str, Any] = {"fields": []}
    for key, label in fields:
        values = [float(t[key]) for t in tiles if key in t]
        if not values:
            continue
        arr = np.array(values, dtype=np.float32)
        summary["fields"].append(
            {
                "key": key,
                "label": label,
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    return summary


def _render_index_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Heuristic Scoring Report</title>
    <link
      href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css"
      rel="stylesheet"
    />
    <style>
      :root {
        --bg: #0f1a1c;
        --panel: #162326;
        --accent: #ffb347;
        --accent-2: #7bdff2;
        --text: #e4f2f2;
        --muted: #9bb4b4;
      }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Fira Sans", "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #1c2f33, #0f1a1c 55%);
        color: var(--text);
      }
      .container {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 16px;
        padding: 24px;
      }
      .card {
        background: var(--panel);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
      }
      h1, h2, h3 {
        margin: 0 0 12px 0;
        font-weight: 600;
      }
      .title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 8px;
        flex-wrap: wrap;
      }
      h1 {
        font-size: 24px;
      }
      .mode-badge {
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.16);
      }
      .mode-badge.learned {
        background: rgba(123, 223, 242, 0.16);
        color: #bdefff;
      }
      .mode-badge.heuristic {
        background: rgba(255, 179, 71, 0.16);
        color: #ffd8a1;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
      }
      .summary-item {
        background: rgba(255, 255, 255, 0.04);
        padding: 10px;
        border-radius: 8px;
      }
      .summary-item span {
        display: block;
        color: var(--muted);
        font-size: 12px;
      }
      .summary-item strong {
        font-size: 18px;
      }
      .cluster-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
      }
      .cluster-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px;
      }
      .cluster-card h4 {
        margin: 0 0 8px 0;
        font-size: 14px;
      }
      .cluster-card .muted {
        font-size: 11px;
      }
      #histogram {
        width: 100%;
        height: 200px;
      }
      #mapbox-map {
        width: 100%;
        height: 520px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
      }
      #map-controls {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 8px;
      }
      #map-controls label {
        font-size: 12px;
        color: var(--muted);
      }
      #heatmap-opacity {
        width: 160px;
      }
      .panel {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .nav {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }
      .nav button {
        background: rgba(255, 255, 255, 0.08);
        color: var(--text);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 6px 12px;
        cursor: pointer;
      }
      .nav button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }
      .tile-preview {
        width: 100%;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
      }
      .muted {
        color: var(--muted);
      }
      .breakdown {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 8px;
      }
      .pill {
        background: rgba(123, 223, 242, 0.12);
        color: var(--accent-2);
        border-radius: 999px;
        padding: 4px 10px;
        display: inline-block;
        font-size: 12px;
      }
      @media (max-width: 900px) {
        .container {
          grid-template-columns: 1fr;
        }
        #heatmap {
          height: 320px;
        }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <div class="title-row">
          <h1 id="report-title">Scenic Scoring Report</h1>
          <div class="mode-badge heuristic" id="scoring-mode-badge">Mode: heuristic</div>
        </div>
        <div class="summary-grid" id="summary"></div>
        <h3>Terrain Feature Summary</h3>
        <div class="summary-grid" id="feature-summary"></div>
        <h3>Cluster View</h3>
        <div class="cluster-grid" id="cluster-summary"></div>
        <h3>Score Histogram</h3>
        <canvas id="histogram"></canvas>
        <h3>Interactive Map</h3>
        <div id="map-controls">
          <label>Base:
            <select id="base-layer">
              <option value="osm">Map</option>
              <option value="sat">Satellite</option>
              <option value="none">Heatmap only</option>
            </select>
          </label>
          <label>Heatmap Opacity:
            <input id="heatmap-opacity" type="range" min="0" max="1" step="0.05" value="0.6" />
          </label>
        </div>
        <div id="mapbox-map"></div>
      <div class="muted" id="mapbox-note">
          Map overlay requires a Mapbox token (set in report_config.json or window.MAPBOX_TOKEN).
      </div>
      <div class="muted" id="route-legend" style="display:none;">
          Route overlay: <span style="color:#ffb347;">scenic</span> vs <span style="color:#66c2ff;">baseline</span>
      </div>
      <div id="route-stats-wrap" style="display:none; margin-top:8px;">
        <div class="summary-grid" id="route-stats"></div>
      </div>
      <div class="muted" id="route-stats-note">Route metrics appear when route overlay data is available.</div>
      </div>
      <div class="card panel">
        <h2>Tile Details</h2>
        <div class="nav">
          <button id="prev-tile">Prev</button>
          <button id="next-tile">Next</button>
          <div class="pill" id="tile-id">Select a tile</div>
        </div>
        <img id="tile-thumb" class="tile-preview" src="" alt="tile thumbnail" />
        <div id="tile-score"></div>
        <div class="breakdown" id="tile-breakdown"></div>
        <div class="muted" id="tile-path"></div>
        <div class="muted">Use heatmap clicks or arrow keys to navigate.</div>
      </div>
    </div>
    <script>
      const summaryEl = document.getElementById("summary");
      const featureSummaryEl = document.getElementById("feature-summary");
      const reportTitleEl = document.getElementById("report-title");
      const scoringModeBadgeEl = document.getElementById("scoring-mode-badge");
      const histogramCanvas = document.getElementById("histogram");
      const clusterSummaryEl = document.getElementById("cluster-summary");
      const heatmapCanvas = document.createElement("canvas");
      const baseLayerSelect = document.getElementById("base-layer");
      const heatmapOpacityInput = document.getElementById("heatmap-opacity");
      const mapboxNote = document.getElementById("mapbox-note");
      const routeLegend = document.getElementById("route-legend");
      const routeStatsWrap = document.getElementById("route-stats-wrap");
      const routeStatsEl = document.getElementById("route-stats");
      const routeStatsNote = document.getElementById("route-stats-note");
      const tileIdEl = document.getElementById("tile-id");
      const tileThumbEl = document.getElementById("tile-thumb");
      const tileScoreEl = document.getElementById("tile-score");
      const tileBreakdownEl = document.getElementById("tile-breakdown");
      const tilePathEl = document.getElementById("tile-path");
      const prevBtn = document.getElementById("prev-tile");
      const nextBtn = document.getElementById("next-tile");

      const colorStops = [
        { t: 0.0, c: [20, 50, 90] },
        { t: 0.35, c: [50, 160, 140] },
        { t: 0.6, c: [240, 200, 90] },
        { t: 0.8, c: [240, 140, 70] },
        { t: 1.0, c: [220, 60, 60] },
      ];

      const colorForScore = (score) => {
        const t = Math.min(1, Math.max(0, score / 10));
        for (let i = 0; i < colorStops.length - 1; i++) {
          const a = colorStops[i];
          const b = colorStops[i + 1];
          if (t >= a.t && t <= b.t) {
            const span = (t - a.t) / (b.t - a.t);
            const r = Math.round(a.c[0] + span * (b.c[0] - a.c[0]));
            const g = Math.round(a.c[1] + span * (b.c[1] - a.c[1]));
            const bcol = Math.round(a.c[2] + span * (b.c[2] - a.c[2]));
            return `rgb(${r}, ${g}, ${bcol})`;
          }
        }
        return "rgb(200, 200, 200)";
      };

      let tiles = [];
      let currentIndex = 0;
      let mapboxToken = null;
      let mapboxReady = false;
      let mapInstance = null;
      let heatmapLayerId = "scenic-heatmap-layer";

      fetch(`report.json?ts=${Date.now()}`)
        .then((r) => r.json())
        .then((data) => {
          renderHeader(data.run_info);
          renderSummary(data.summary, data.run_info);
          renderFeatureSummary(data.feature_summary);
          renderClusterSummary(data.tiles || []);
          renderHistogram(data.histogram);
          tiles = data.tiles || [];
          const cacheBust =
            data.run_info && data.run_info.timestamp ? data.run_info.timestamp : Date.now();
          tiles.forEach((tile) => {
            if (tile.thumb) {
              tile.thumb_url = `${tile.thumb}?v=${cacheBust}`;
            }
          });
          renderHeatmap(data);
          if (tiles.length > 0) {
            setTile(tiles[0]);
          }

          Promise.all([loadToken(), loadMapboxGL()]).then(([token]) => {
            mapboxToken = token;
            initMapbox(data);
          });
        });

      function inferScoringMode(runInfo) {
        if (!runInfo) return "heuristic";
        const mode = String(runInfo.scoring_mode || "").trim().toLowerCase();
        if (mode === "learned") return "learned";
        return "heuristic";
      }

      function renderHeader(runInfo) {
        const mode = inferScoringMode(runInfo);
        reportTitleEl.textContent = mode === "learned" ? "Learned Scenic Scoring Report" : "Heuristic Scenic Scoring Report";
        scoringModeBadgeEl.textContent = `Mode: ${mode}`;
        scoringModeBadgeEl.classList.remove("learned", "heuristic");
        scoringModeBadgeEl.classList.add("mode-badge", mode);
      }

      function loadMapboxGL() {
        if (window.maplibregl) {
          mapboxReady = true;
          return Promise.resolve();
        }
        const candidates = [
          "https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js",
        ];
        return new Promise((resolve) => {
          const script = document.createElement("script");
          let idx = 0;
          const tryNext = () => {
            if (idx >= candidates.length) return resolve();
            script.src = candidates[idx++];
            script.onload = () => {
              mapboxReady = true;
              resolve();
            };
            script.onerror = () => tryNext();
          };
          tryNext();
          document.head.appendChild(script);
        });
      }

      function loadToken() {
        if (window.MAPBOX_TOKEN) {
          return Promise.resolve(window.MAPBOX_TOKEN);
        }
        return fetch("report_config.json")
          .then((resp) => {
            if (!resp.ok) return null;
            return resp.json();
          })
          .then((cfg) => (cfg && cfg.mapbox_token ? cfg.mapbox_token : null))
          .catch(() => null);
      }

      function normalizeFeatureCollection(routeGeo, fallbackKind) {
        if (!routeGeo) return null;
        if (routeGeo.type === "FeatureCollection" && Array.isArray(routeGeo.features)) {
          const features = routeGeo.features
            .filter((feature) => feature && feature.type === "Feature")
            .map((feature) => {
              const properties = Object.assign({}, feature.properties || {});
              if (!properties.route_kind) properties.route_kind = fallbackKind;
              return Object.assign({}, feature, { properties });
            });
          return { type: "FeatureCollection", features };
        }
        if (routeGeo.type === "Feature") {
          const properties = Object.assign({}, routeGeo.properties || {});
          if (!properties.route_kind) properties.route_kind = fallbackKind;
          return {
            type: "FeatureCollection",
            features: [Object.assign({}, routeGeo, { properties })],
          };
        }
        return null;
      }

      function pickRouteFeature(routeGeo, preferredKind, fallbackKind) {
        const normalized = normalizeFeatureCollection(routeGeo, fallbackKind);
        if (!normalized || normalized.features.length === 0) return null;
        const preferred = normalized.features.find((feature) => {
          const kind = String((feature.properties || {}).route_kind || "").toLowerCase();
          return kind === preferredKind;
        });
        return preferred || normalized.features[0];
      }

      function fetchRouteGeojson(path) {
        return fetch(`${path}?ts=${Date.now()}`)
          .then((resp) => (resp.ok ? resp.json() : null))
          .catch(() => null);
      }

      async function loadRouteOverlay() {
        const primary = await fetchRouteGeojson("route.geojson");
        if (primary) {
          return normalizeFeatureCollection(primary, "scenic");
        }

        const [scenicGeo, fastGeo] = await Promise.all([
          fetchRouteGeojson("route_scenic.geojson"),
          fetchRouteGeojson("route_fast.geojson"),
        ]);
        const features = [];
        const scenicFeature = pickRouteFeature(scenicGeo, "scenic", "scenic");
        const baselineFeature = pickRouteFeature(fastGeo, "baseline", "baseline");

        if (scenicFeature) {
          const properties = Object.assign({}, scenicFeature.properties || {}, {
            route_kind: "scenic",
          });
          features.push(Object.assign({}, scenicFeature, { properties }));
        }
        if (baselineFeature) {
          const properties = Object.assign({}, baselineFeature.properties || {}, {
            route_kind: "baseline",
          });
          features.push(Object.assign({}, baselineFeature, { properties }));
        }

        if (features.length === 0) return null;
        return { type: "FeatureCollection", features };
      }

      function _routeFeatureByKind(routeGeo, kind) {
        if (!routeGeo || !Array.isArray(routeGeo.features)) return null;
        const target = String(kind || "").toLowerCase();
        for (const feature of routeGeo.features) {
          const current = String((feature.properties || {}).route_kind || "scenic").toLowerCase();
          if (current === target) return feature;
        }
        return null;
      }

      function _toFiniteNumber(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
      }

      function _formatNumber(value, digits) {
        const n = _toFiniteNumber(value);
        if (n === null) return "n/a";
        return n.toFixed(digits);
      }

      function _formatSigned(value, digits) {
        const n = _toFiniteNumber(value);
        if (n === null) return "n/a";
        const sign = n > 0 ? "+" : "";
        return `${sign}${n.toFixed(digits)}`;
      }

      function _routeMetricsFromFeature(feature) {
        const props = (feature && feature.properties) || {};
        return {
          distanceKm: _toFiniteNumber(props.total_distance_km),
          durationMin: _toFiniteNumber(props.estimated_duration_minutes),
          scenicScore: _toFiniteNumber(props.average_scenic_score),
          segments: _toFiniteNumber(props.segments),
        };
      }

      function renderRouteStats(routeGeo) {
        if (!routeStatsWrap || !routeStatsEl || !routeStatsNote) return;
        if (!routeGeo || !Array.isArray(routeGeo.features) || routeGeo.features.length === 0) {
          routeStatsWrap.style.display = "none";
          routeStatsEl.innerHTML = "";
          routeStatsNote.style.display = "block";
          return;
        }

        const baselineFeature = _routeFeatureByKind(routeGeo, "baseline");
        const scenicFeature = _routeFeatureByKind(routeGeo, "scenic")
          || routeGeo.features.find((f) => f !== baselineFeature)
          || null;

        const scenic = _routeMetricsFromFeature(scenicFeature);
        const baseline = _routeMetricsFromFeature(baselineFeature);

        const items = [];
        if (scenicFeature) {
          items.push(["Scenic Dist (km)", _formatNumber(scenic.distanceKm, 2)]);
          items.push(["Scenic Time (min)", _formatNumber(scenic.durationMin, 1)]);
          items.push(["Scenic Score", _formatNumber(scenic.scenicScore, 2)]);
        }
        if (baselineFeature) {
          items.push(["Base Dist (km)", _formatNumber(baseline.distanceKm, 2)]);
          items.push(["Base Time (min)", _formatNumber(baseline.durationMin, 1)]);
          items.push(["Base Score", _formatNumber(baseline.scenicScore, 2)]);
        }
        if (scenicFeature && baselineFeature) {
          items.push(["Delta Dist (km)", _formatSigned(scenic.distanceKm - baseline.distanceKm, 2)]);
          items.push(["Delta Time (min)", _formatSigned(scenic.durationMin - baseline.durationMin, 1)]);
          items.push(["Delta Scenic", _formatSigned(scenic.scenicScore - baseline.scenicScore, 2)]);
        }

        routeStatsEl.innerHTML = items
          .map(([label, value]) => {
            return `<div class="summary-item"><span>${label}</span><strong>${value}</strong></div>`;
          })
          .join("");
        routeStatsWrap.style.display = "block";
        routeStatsNote.style.display = "none";
      }

      function renderSummary(summary, runInfo) {
        const scoringMode = inferScoringMode(runInfo);
        const items = [
          ["Total Tiles", summary.total_tiles],
          ["Mean Score", summary.mean.toFixed(2)],
          ["Median Score", summary.median.toFixed(2)],
          ["Std Dev", summary.std.toFixed(2)],
          ["Min Score", summary.min.toFixed(2)],
          ["Max Score", summary.max.toFixed(2)],
          ["Scoring Mode", scoringMode],
          ["Classifier", runInfo.used_classifier ? "on" : "off"],
          ["Device", runInfo.device],
          ["Missing Pairs", runInfo.counts.missing_pairs],
        ];
        summaryEl.innerHTML = items
          .map(([label, value]) => {
            return `<div class="summary-item"><span>${label}</span><strong>${value}</strong></div>`;
          })
          .join("");
      }

      function renderFeatureSummary(featureSummary) {
        if (!featureSummary || !featureSummary.fields || featureSummary.fields.length === 0) {
          featureSummaryEl.innerHTML = `<div class="summary-item"><span>Terrain</span><strong>n/a</strong></div>`;
          return;
        }
        const items = featureSummary.fields.flatMap((field) => {
          return [
            [`${field.label} Mean`, Number(field.mean).toFixed(2)],
            [`${field.label} Median`, Number(field.median).toFixed(2)],
            [`${field.label} Min`, Number(field.min).toFixed(2)],
            [`${field.label} Max`, Number(field.max).toFixed(2)],
          ];
        });
        featureSummaryEl.innerHTML = items
          .map(([label, value]) => {
            return `<div class="summary-item"><span>${label}</span><strong>${value}</strong></div>`;
          })
          .join("");
      }

      function _safeNum(value, fallback = 0) {
        const n = Number(value);
        return Number.isFinite(n) ? n : fallback;
      }

      function _minMax(values) {
        if (!values.length) return [0, 1];
        let lo = values[0];
        let hi = values[0];
        for (const v of values) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        if (lo === hi) return [lo, lo + 1];
        return [lo, hi];
      }

      function _normalize(v, lo, hi) {
        return (v - lo) / Math.max(hi - lo, 1e-6);
      }

      function _kmeans(vectors, k, iters = 10) {
        if (!vectors.length) return [];
        const dim = vectors[0].length;
        const centers = [];
        const n = vectors.length;
        const step = Math.max(1, Math.floor(n / k));
        for (let i = 0; i < k; i++) {
          centers.push(vectors[Math.min(i * step, n - 1)].slice());
        }
        const assign = new Array(n).fill(0);

        for (let iter = 0; iter < iters; iter++) {
          for (let i = 0; i < n; i++) {
            let best = 0;
            let bestDist = Infinity;
            for (let c = 0; c < k; c++) {
              let dist = 0;
              for (let d = 0; d < dim; d++) {
                const diff = vectors[i][d] - centers[c][d];
                dist += diff * diff;
              }
              if (dist < bestDist) {
                bestDist = dist;
                best = c;
              }
            }
            assign[i] = best;
          }

          const sums = Array.from({ length: k }, () => new Array(dim).fill(0));
          const counts = new Array(k).fill(0);
          for (let i = 0; i < n; i++) {
            const c = assign[i];
            counts[c] += 1;
            for (let d = 0; d < dim; d++) sums[c][d] += vectors[i][d];
          }
          for (let c = 0; c < k; c++) {
            if (counts[c] === 0) continue;
            for (let d = 0; d < dim; d++) centers[c][d] = sums[c][d] / counts[c];
          }
        }
        return assign;
      }

      function renderClusterSummary(tileRows) {
        if (!clusterSummaryEl) return;
        if (!Array.isArray(tileRows) || tileRows.length < 8) {
          clusterSummaryEl.innerHTML = `<div class="summary-item"><span>Clusters</span><strong>n/a</strong></div>`;
          return;
        }

        const reliefs = tileRows.map((t) => _safeNum(t.relief));
        const slopes = tileRows.map((t) => _safeNum(t.slope_mean));
        const vegs = tileRows.map((t) => _safeNum(t.veg_proxy));
        const [relLo, relHi] = _minMax(reliefs);
        const [sloLo, sloHi] = _minMax(slopes);
        const [vegLo, vegHi] = _minMax(vegs);

        const vectors = tileRows.map((t) => [
          _safeNum(t.scenic_score) / 10.0,
          _normalize(_safeNum(t.relief), relLo, relHi),
          _normalize(_safeNum(t.slope_mean), sloLo, sloHi),
          _normalize(_safeNum(t.veg_proxy), vegLo, vegHi),
        ]);
        const k = Math.max(3, Math.min(6, Math.floor(Math.sqrt(tileRows.length / 80))));
        const assign = _kmeans(vectors, k, 12);
        const groups = Array.from({ length: k }, () => []);
        tileRows.forEach((tile, i) => groups[assign[i]].push(tile));

        const cards = groups
          .map((rows, i) => {
            if (!rows.length) return null;
            const meanScore = rows.reduce((a, t) => a + _safeNum(t.scenic_score), 0) / rows.length;
            const meanRelief = rows.reduce((a, t) => a + _safeNum(t.relief), 0) / rows.length;
            const classCounts = {};
            for (const row of rows) {
              const name = String(row.class_name || "unknown");
              classCounts[name] = (classCounts[name] || 0) + 1;
            }
            const topClass = Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0];
            return {
              idx: i + 1,
              rows: rows.length,
              meanScore,
              meanRelief,
              topClass: topClass ? `${topClass[0]} (${topClass[1]})` : "n/a",
            };
          })
          .filter(Boolean)
          .sort((a, b) => b.meanScore - a.meanScore);

        clusterSummaryEl.innerHTML = cards
          .map((c) => {
            return `<div class="cluster-card">
              <h4>Cluster ${c.idx}</h4>
              <div class="summary-item"><span>Tiles</span><strong>${c.rows}</strong></div>
              <div class="summary-item"><span>Mean Score</span><strong>${c.meanScore.toFixed(2)}</strong></div>
              <div class="summary-item"><span>Mean Relief</span><strong>${c.meanRelief.toFixed(1)}</strong></div>
              <div class="muted">Top Class: ${c.topClass}</div>
            </div>`;
          })
          .join("");
      }

      function renderHistogram(histogram) {
        const ctx = histogramCanvas.getContext("2d");
        const width = histogramCanvas.clientWidth;
        const height = histogramCanvas.clientHeight;
        histogramCanvas.width = width;
        histogramCanvas.height = height;
        ctx.clearRect(0, 0, width, height);

        const counts = histogram.counts;
        const maxCount = Math.max(...counts, 1);
        const barWidth = width / counts.length;

        counts.forEach((count, idx) => {
          const barHeight = (count / maxCount) * (height - 20);
          ctx.fillStyle = colorForScore((idx / counts.length) * 10);
          ctx.fillRect(idx * barWidth, height - barHeight, barWidth - 2, barHeight);
        });
      }

      function tileToLatLon(x, y, z) {
        const n = Math.pow(2, z);
        const lon = (x / n) * 360 - 180;
        const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n)));
        const lat = (latRad * 180) / Math.PI;
        return [lat, lon];
      }

      function initMapbox(data) {
        if (!mapboxReady || !window.maplibregl) {
          mapboxNote.textContent =
            "MapLibre failed to load. Check network access or allow https://unpkg.com.";
          return;
        }
        if (!data.grid || !data.grid.has_coords) {
          mapboxNote.textContent = "No tile coordinates available for map overlay.";
          return;
        }
        const sources = {
          "osm": {
            "type": "raster",
            "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            "tileSize": 256,
          },
        };
        const layers = [
          {
            "id": "osm",
            "type": "raster",
            "source": "osm",
          },
        ];
        if (mapboxToken) {
          sources["satellite"] = {
            "type": "raster",
            "tiles": [
              `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=${mapboxToken}`,
            ],
            "tileSize": 256,
          };
          layers.push({
            "id": "satellite",
            "type": "raster",
            "source": "satellite",
            "layout": { "visibility": "none" },
          });
          mapboxNote.style.display = "none";
        } else {
          mapboxNote.textContent =
            "Satellite layer requires a Mapbox token (set in report_config.json or window.MAPBOX_TOKEN).";
          mapboxNote.style.display = "block";
          const satOption = baseLayerSelect.querySelector('option[value="sat"]');
          if (satOption) satOption.disabled = true;
        }

        const map = new maplibregl.Map({
          container: "mapbox-map",
          style: {
            "version": 8,
            "sources": sources,
            "layers": layers
          },
          center: [-72.5, 42.4],
          zoom: data.grid.zoom || 12,
        });
        mapInstance = map;

        map.on("load", () => {
          const grid = data.grid;
          const [nwLat, nwLon] = tileToLatLon(grid.min_x, grid.min_y, grid.zoom);
          const [seLat, seLon] = tileToLatLon(grid.max_x + 1, grid.max_y + 1, grid.zoom);
          const coordinates = [
            [nwLon, nwLat],
            [seLon, nwLat],
            [seLon, seLat],
            [nwLon, seLat],
          ];

          const overlayUrl = heatmapCanvas.toDataURL("image/png");
          map.addSource("scenic-heatmap", {
            type: "image",
            url: overlayUrl,
            coordinates,
          });
          map.addLayer({
            id: "scenic-heatmap-layer",
            type: "raster",
            source: "scenic-heatmap",
            paint: { "raster-opacity": 0.6 },
          });
          map.fitBounds([ [nwLon, seLat], [seLon, nwLat] ], { padding: 20, duration: 0 });

          loadRouteOverlay()
            .then((routeGeo) => {
              if (!routeGeo || !routeGeo.features || routeGeo.features.length === 0) return;
              map.addSource("scenic-route", { type: "geojson", data: routeGeo });
              routeLegend.style.display = "block";
              renderRouteStats(routeGeo);

              // Baseline (non-scenic) route in blue.
              map.addLayer({
                id: "scenic-route-line-baseline",
                type: "line",
                source: "scenic-route",
                filter: ["==", ["coalesce", ["get", "route_kind"], "scenic"], "baseline"],
                paint: {
                  "line-color": "#66c2ff",
                  "line-width": 4,
                  "line-opacity": 0.92,
                },
              });

              // Scenic-optimized route in orange.
              map.addLayer({
                id: "scenic-route-line",
                type: "line",
                source: "scenic-route",
                filter: ["!=", ["coalesce", ["get", "route_kind"], "scenic"], "baseline"],
                paint: { "line-color": "#ffb347", "line-width": 6, "line-opacity": 0.95 },
              });
            })
            .catch(() => null);
          applyMapControls();
        });

        map.on("click", (event) => {
          if (!data.grid || !data.grid.has_coords) return;
          const z = data.grid.zoom;
          const lat = event.lngLat.lat;
          const lon = event.lngLat.lng;
          const n = Math.pow(2, z);
          const x = Math.floor(((lon + 180) / 360) * n);
          const y = Math.floor(
            (1 - Math.log(Math.tan((lat * Math.PI) / 180) + 1 / Math.cos((lat * Math.PI) / 180)) / Math.PI) / 2 * n
          );
          const key = `${x},${y}`;
          const idx = data.grid.index_by_coord[key];
          if (idx !== undefined) {
            setTile(tiles[idx]);
          }
        });
        baseLayerSelect.addEventListener("change", applyMapControls);
        heatmapOpacityInput.addEventListener("input", applyMapControls);
      }

      function applyMapControls() {
        if (!mapInstance) return;
        const opacity = parseFloat(heatmapOpacityInput.value || "0.6");
        if (mapInstance.getLayer(heatmapLayerId)) {
          mapInstance.setPaintProperty(heatmapLayerId, "raster-opacity", opacity);
        }
        const base = baseLayerSelect.value;
        const showOsm = base === "osm";
        const showSat = base === "sat";
        const showNone = base === "none";
        if (mapInstance.getLayer("osm")) {
          mapInstance.setLayoutProperty("osm", "visibility", showOsm && !showNone ? "visible" : "none");
        }
        if (mapInstance.getLayer("satellite")) {
          mapInstance.setLayoutProperty("satellite", "visibility", showSat && !showNone ? "visible" : "none");
        }
      }

      function renderHeatmap(data) {
        const grid = data.grid;
        if (!grid.has_coords) {
          return;
        }

        tiles = data.tiles || [];
        const gridW = grid.max_x - grid.min_x + 1;
        const gridH = grid.max_y - grid.min_y + 1;
        const maxDim = 512;
        const baseScale = Math.max(1, Math.ceil(Math.max(gridW, gridH) / maxDim));

        const ctx = heatmapCanvas.getContext("2d");
        drawHeatmap(ctx, baseScale, {
          minX: grid.min_x,
          maxX: grid.max_x,
          minY: grid.min_y,
          maxY: grid.max_y,
        });
      }

      function drawHeatmap(ctx, baseScale, view) {
        const viewW = view.maxX - view.minX + 1;
        const viewH = view.maxY - view.minY + 1;
        const scale = Math.max(1, Math.ceil(Math.max(viewW, viewH) / 512));
        heatmapCanvas.width = Math.ceil(viewW / scale);
        heatmapCanvas.height = Math.ceil(viewH / scale);
        ctx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);

        tiles.forEach((tile) => {
          if (
            tile.x < view.minX ||
            tile.x > view.maxX ||
            tile.y < view.minY ||
            tile.y > view.maxY
          ) {
            return;
          }

          const px = Math.floor((tile.x - view.minX) / scale);
          const py = Math.floor((tile.y - view.minY) / scale);
          ctx.fillStyle = colorForScore(tile.scenic_score);
          ctx.fillRect(px, py, 1, 1);
        });
      }

      function setTile(tile) {
        currentIndex = tile.index;
        tileIdEl.textContent = `Tile ${tile.index + 1} / ${tiles.length}`;
        tileThumbEl.src = tile.thumb_url || tile.thumb || "";
        tileScoreEl.innerHTML = `<strong>Score:</strong> ${tile.scenic_score.toFixed(
          2
        )} <span class="muted">(class: ${
          tile.class_name || "unknown"
        }, class score: ${Number(tile.class_score).toFixed(2)})</span>`;

        const breakdown = [
          ["Relief (m)", tile.relief],
          ["Roughness", tile.roughness],
          ["Slope Mean", tile.slope_mean],
          ["Water Proxy", tile.water_proxy],
          ["Veg Proxy", tile.veg_proxy],
          ["Water Fraction", tile.water_fraction],
          ["Texture", tile.texture],
          ["Class Score", tile.class_score],
        ];
        tileBreakdownEl.innerHTML = breakdown
          .map(([label, value]) => {
            if (value === undefined || value === null || Number.isNaN(Number(value))) {
              return `<div class="summary-item"><span>${label}</span><strong>n/a</strong></div>`;
            }
            return `<div class="summary-item"><span>${label}</span><strong>${Number(
              value
            ).toFixed(3)}</strong></div>`;
          })
          .join("");
        tilePathEl.textContent = tile.image_path;
        prevBtn.disabled = currentIndex <= 0;
        nextBtn.disabled = currentIndex >= tiles.length - 1;
      }

      prevBtn.addEventListener("click", () => {
        if (currentIndex > 0) {
          setTile(tiles[currentIndex - 1]);
        }
      });
      nextBtn.addEventListener("click", () => {
        if (currentIndex < tiles.length - 1) {
          setTile(tiles[currentIndex + 1]);
        }
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "ArrowLeft") {
          prevBtn.click();
        }
        if (event.key === "ArrowRight") {
          nextBtn.click();
        }
      });
    </script>
  </body>
</html>
"""


def _copy_report_config(report_path: Path) -> None:
    global_config = Path("data/processed/report_config.json")
    if global_config.exists():
        try:
            shutil.copy(global_config, report_path / "report_config.json")
        except OSError:
            pass
