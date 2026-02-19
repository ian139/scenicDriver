"""
Single-call routing wrapper:
- loads graph + optional tile score report from run_name
- computes scenic and baseline routes
- writes route.geojson and metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.service import RouteRequest, plan_routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route compare service wrapper")
    parser.add_argument("--start", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--end", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--scenic-weight", type=float, default=0.8)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--graph-geojson",
        type=Path,
        default=Path("data/processed/road_graphs/pittsfield_core/road_graph.geojson"),
    )
    parser.add_argument("--max-detour-factor", type=float, default=1.8)
    parser.add_argument("--avoid-highways", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path("data/processed/heuristic_runs") / args.run_name / "report"
    report_json = report_dir / "report.json"
    route_geojson = report_dir / "route.geojson"
    metrics_json = report_dir / "route_metrics.json"

    request = RouteRequest(
        graph_geojson=str(args.graph_geojson),
        start=(args.start[0], args.start[1]),
        end=(args.end[0], args.end[1]),
        scenic_weight=args.scenic_weight,
        avoid_highways=args.avoid_highways,
        max_detour_factor=args.max_detour_factor,
        include_baseline=args.include_baseline,
        tile_scores_json=str(report_json) if report_json.exists() else None,
    )
    result = plan_routes(request)

    report_dir.mkdir(parents=True, exist_ok=True)
    route_geojson.write_text(json.dumps(result["geojson"], indent=2), encoding="utf-8")

    routes = {r["route_kind"]: r["metrics"] for r in result.get("routes", [])}
    payload = {
        "request": request.to_dict(),
        "score_mapping": result.get("score_mapping", {}),
        "scenic": routes.get("scenic"),
        "baseline": routes.get("baseline"),
        "route_geojson": str(route_geojson),
    }
    metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {route_geojson}")
    print(f"Wrote {metrics_json}")


if __name__ == "__main__":
    main()
