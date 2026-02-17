"""
Plan a scenic route from a GeoJSON LineString graph and emit route GeoJSON.

Example:
  uv run python scripts/route_demo_geojson.py \
    --geojson data/processed/sample_road_graph.geojson \
    --start 42.40 -72.70 \
    --end 42.48 -72.62 \
    --scenic-weight 0.6 \
    --output-geojson data/processed/sample_route.geojson \
    --report-dir data/processed/heuristic_runs/<run_name>/report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.service import RouteRequest, plan_routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan scenic route from GeoJSON graph")
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--start", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--end", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--scenic-weight", type=float, default=0.6)
    parser.add_argument("--avoid-highways", action="store_true")
    parser.add_argument("--max-detour-factor", type=float, default=1.8)
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute a non-scenic baseline route (scenic_weight=0.0) for comparison",
    )
    parser.add_argument("--output-geojson", type=Path, default=Path("data/processed/sample_route.geojson"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="If set, also copy route.geojson into this report directory for overlay",
    )
    parser.add_argument(
        "--tile-scores-json",
        type=Path,
        default=None,
        help="Optional report.json with per-tile scenic scores used to overwrite edge scenic scores.",
    )
    parser.add_argument(
        "--tile-score-zoom",
        type=int,
        default=None,
        help="Tile zoom to use for score lookup (defaults to inferred zoom from tile-scores JSON).",
    )
    parser.add_argument(
        "--tile-score-fallback",
        type=float,
        default=None,
        help="Fallback scenic score for edges with no tile match (default: keep existing edge scenic score).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_scores_path = args.tile_scores_json
    if tile_scores_path is None and args.report_dir is not None:
        candidate = args.report_dir / "report.json"
        if candidate.exists():
            tile_scores_path = candidate

    request = RouteRequest(
        graph_geojson=str(args.geojson),
        start=(args.start[0], args.start[1]),
        end=(args.end[0], args.end[1]),
        scenic_weight=args.scenic_weight,
        avoid_highways=args.avoid_highways,
        max_detour_factor=args.max_detour_factor,
        include_baseline=args.include_baseline,
        tile_scores_json=str(tile_scores_path) if tile_scores_path else None,
        tile_score_zoom=args.tile_score_zoom,
        tile_score_fallback=args.tile_score_fallback,
    )
    result = plan_routes(request)

    mapping = result.get("score_mapping", {})
    if mapping.get("enabled"):
        matched = int(mapping.get("matched_edges", 0))
        total = int(mapping.get("total_edges", 0))
        pct = float(mapping.get("matched_ratio", 0.0)) * 100.0
        source = mapping.get("source")
        zoom = mapping.get("zoom")
        print(f"Applied tile scenic scores from {source} at z{zoom}: matched {matched}/{total} edges ({pct:.1f}%)")
    else:
        print("No tile-score map provided; using scenic_score values embedded in graph.")

    routes = {r["route_kind"]: r["metrics"] for r in result.get("routes", [])}
    scenic = routes.get("scenic")
    if scenic:
        print("[Scenic Route]")
        print(f"Segments: {int(scenic.get('segments', 0))}")
        print(f"Total distance (km): {float(scenic.get('total_distance_km', 0.0)):.2f}")
        print(f"Avg scenic score: {float(scenic.get('average_scenic_score', 0.0)):.2f}")
        print(f"Estimated duration (min): {float(scenic.get('estimated_duration_minutes', 0.0)):.1f}")

    baseline = routes.get("baseline")
    if baseline:
        print("[Baseline Route]")
        print(f"Segments: {int(baseline.get('segments', 0))}")
        print(f"Total distance (km): {float(baseline.get('total_distance_km', 0.0)):.2f}")
        print(f"Avg scenic score: {float(baseline.get('average_scenic_score', 0.0)):.2f}")
        print(f"Estimated duration (min): {float(baseline.get('estimated_duration_minutes', 0.0)):.1f}")

    geojson = result["geojson"]
    args.output_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.output_geojson.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_geojson}")

    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_route = args.report_dir / "route.geojson"
        shutil.copy(args.output_geojson, report_route)
        print(f"Copied overlay to {report_route}")


if __name__ == "__main__":
    main()
