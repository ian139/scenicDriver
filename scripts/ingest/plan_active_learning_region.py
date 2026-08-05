"""Deterministic active-learning regional planning, inventory, and acquisition CLI."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.data_pipeline.mapbox import MapboxTileSource, tile_to_lat_lon_center
from src.data_pipeline.region_planning import get_builtin_region_spec, parse_and_validate_region_spec
from src.data_pipeline.tile_inventory import build_inventory_report, scan_tile_inventory

COLUMNS = ["region", "z", "x", "y", "lat", "lon", "satellite_path", "terrain_path", "satellite_present", "terrain_present"]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import io
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({column: row[column] for column in COLUMNS} for row in rows)
    _atomic_write_text(path, output.getvalue())


@contextmanager
def _quiet_mapbox_logging():
    """Prevent source logs from exposing credential fragments or request URLs."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _load_spec(path: str | None) -> dict[str, Any]:
    if path is None:
        return get_builtin_region_spec()
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _manifest_rows(planned: dict[str, Any], zoom: int, image_root: Path) -> list[dict[str, Any]]:
    rows = []
    for x, y in planned["ordered_coords"]:
        region = planned["coord_to_region"][(x, y)]
        lat, lon = tile_to_lat_lon_center(x, y, zoom)
        rows.append({
            "region": region, "z": zoom, "x": x, "y": y, "lat": f"{lat:.9f}", "lon": f"{lon:.9f}",
            "satellite_path": str(image_root / "satellite" / f"z{zoom}" / region / f"{x}_{y}.png"),
            "terrain_path": str(image_root / "terrain" / f"z{zoom}" / region / f"{x}_{y}.png"),
            "satellite_present": False, "terrain_present": False,
        })
    return rows


def acquire_missing(rows: list[dict[str, Any]], *, image_root: Path, zoom: int, failures: list[dict[str, Any]]) -> None:
    """Acquire only missing/invalid pair members, recording stable failure entries."""
    for style, style_id, path_key in (("satellite", "mapbox.satellite", "satellite_path"), ("terrain", "mapbox.terrain-rgb", "terrain_path")):
        with _quiet_mapbox_logging():
            source = MapboxTileSource(cache_dir=image_root / ".mapbox_cache" / style, style_id=style_id)
        for row in rows:
            if row[f"{style}_present"]:
                continue
            path = Path(row[path_key])
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with _quiet_mapbox_logging():
                    tile = source.get_tile(int(row["x"]), int(row["y"]), zoom)
                Image.fromarray(tile.image).save(path, format="PNG", optimize=True)
                row[f"{style}_present"] = True
            except Exception as exc:
                failures.append({"region": row["region"], "z": int(row["z"]), "x": int(row["x"]), "y": int(row["y"]), "style": style, "reason": type(exc).__name__})


def plan_run(*, run_name: str, spec: dict[str, Any] | None = None, spec_path: str | None = None, output_root: Path | str = "data/processed/active_learning", image_root: Path | str = "data/raw/images", config_path: Path | str = "config/app_regions.json", zoom: int = 14, acquire: bool = False, budget: int = 370000) -> dict[str, Any]:
    spec = spec if spec is not None else _load_spec(spec_path)
    planned = parse_and_validate_region_spec(spec, app_regions_path=config_path, max_budget_coords=budget, zoom=zoom)
    run_dir = Path(output_root) / run_name
    rows = _manifest_rows(planned, zoom, Path(image_root))
    rows, counts = scan_tile_inventory(rows, image_root=image_root)
    failures: list[dict[str, Any]] = []
    if acquire:
        acquire_missing(rows, image_root=Path(image_root), zoom=zoom, failures=failures)
        rows, counts = scan_tile_inventory(rows, image_root=image_root)
    report = build_inventory_report(rows, counts, failures)
    manifest = {
        "schema_version": 1, "run_name": run_name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "zoom": zoom, "styles": ["satellite", "terrain"], "tile_budget_coordinates": budget,
        "tile_budget_rasters": budget * 2, "geometry_digest": planned["geometry_digest"],
        "region_spec": spec, "region_tile_counts": planned["region_tile_counts"],
        "unique_coordinates": planned["unique_coordinates_count"], "total_rasters": planned["total_rasters_count"],
        "inventory_counts": counts, "acquisition_requested": acquire, "failures": failures,
        "sources": {"satellite": "Mapbox satellite", "terrain": "Mapbox Terrain-RGB"},
        "stage_state": "acquired" if acquire else "planned", "ready_for_selection": not failures and counts["complete_pairs"] == counts["coordinates"],
    }
    _atomic_json(run_dir / "region_manifest.json", manifest)
    _atomic_csv(run_dir / "tile_manifest.csv", rows)
    _atomic_json(run_dir / "inventory_report.json", report)
    return {"run_dir": str(run_dir), "region_manifest": manifest, "inventory_report": report, "tile_manifest_rows": len(rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--region-spec", help="Versioned JSON region specification; defaults to built-in inland expansion")
    parser.add_argument("--output-root", default="data/processed/active_learning")
    parser.add_argument("--image-root", default="data/raw/images")
    parser.add_argument("--config", default="config/app_regions.json")
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument("--budget", type=int, default=370000)
    parser.add_argument("--acquire", action="store_true", help="Acquire missing or invalid pair members")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = plan_run(run_name=args.run_name, spec_path=args.region_spec, output_root=args.output_root, image_root=args.image_root, config_path=args.config, zoom=args.zoom, acquire=args.acquire, budget=args.budget)
    print(json.dumps({"run_dir": result["run_dir"], "tile_manifest_rows": result["tile_manifest_rows"], "inventory": result["inventory_report"]["counts"]}, sort_keys=True))

if __name__ == "__main__":
    main()
