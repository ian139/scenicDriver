"""
Extract terrain features from terrain-rgb tiles (optionally paired with satellite tiles).

Example:
  uv run python scripts/extract_terrain_features.py \
    --terrain-dir data/raw/images/terrain/z16/rocky_mountains \
    --satellite-dir data/raw/images/satellite/z16/rocky_mountains \
    --output data/processed/rocky_mountains_terrain_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.terrain import extract_terrain_features_from_tile

_COORD_RE = re.compile(r"z(\d+)[/_-]*x(\d+)[/_-]*y(\d+)", re.IGNORECASE)


def _parse_tile_coords(path: Path) -> tuple[int, int, int] | None:
    match = _COORD_RE.search(path.as_posix())
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    try:
        zoom = None
        if path.parent.name.lower().startswith("z"):
            zoom = int(path.parent.name[1:])
        elif path.parent.parent.name.lower().startswith("z"):
            zoom = int(path.parent.parent.name[1:])
        stem_parts = path.stem.split("_")
        if zoom is not None and len(stem_parts) == 2:
            x, y = int(stem_parts[0]), int(stem_parts[1])
            return zoom, x, y
    except ValueError:
        return None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract terrain features from tiles")
    parser.add_argument("--terrain-dir", type=Path, default=None)
    parser.add_argument("--satellite-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--all-regions", action="store_true")
    parser.add_argument("--terrain-root", type=Path, default=None)
    parser.add_argument("--satellite-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_regions:
        if args.terrain_root is None or args.output_dir is None:
            raise ValueError("--terrain-root and --output-dir are required with --all-regions")
        terrain_root = args.terrain_root
        satellite_root = args.satellite_root
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        regions = [p for p in terrain_root.iterdir() if p.is_dir()]
        if not regions:
            raise ValueError(f"No region folders found in: {terrain_root}")

        for region_dir in regions:
            region = region_dir.name
            sat_dir = satellite_root / region if satellite_root else None
            out_path = output_dir / f"{region}_terrain_features.csv"
            _process_region(region_dir, sat_dir, out_path, args.max_tiles)
    else:
        if args.terrain_dir is None or args.output is None:
            raise ValueError("--terrain-dir and --output are required unless --all-regions is set")
        terrain_dir = args.terrain_dir
        satellite_dir = args.satellite_dir
        output_path = args.output
        _process_region(terrain_dir, satellite_dir, output_path, args.max_tiles)


def _process_region(
    terrain_dir: Path,
    satellite_dir: Path | None,
    output_path: Path,
    max_tiles: int | None,
) -> None:
    if not terrain_dir.exists():
        raise FileNotFoundError(f"Terrain dir not found: {terrain_dir}")

    terrain_files = sorted([p for p in terrain_dir.glob("**/*") if p.suffix.lower() == ".png"])
    if max_tiles is not None:
        terrain_files = terrain_files[: max_tiles]

    rows = []
    for terrain_path in terrain_files:
        rel_path = terrain_path.relative_to(terrain_dir)
        sat_path = None
        if satellite_dir is not None:
            candidate = satellite_dir / rel_path
            if candidate.exists():
                sat_path = candidate

        features = extract_terrain_features_from_tile(
            terrain_path=terrain_path,
            satellite_path=sat_path,
        )

        coords = _parse_tile_coords(terrain_path)
        row = {
            "terrain_path": rel_path.as_posix(),
            "satellite_path": sat_path.relative_to(satellite_dir).as_posix()
            if sat_path and satellite_dir
            else "",
            "slope_variation": features.slope_variation,
            "elevation_change": features.elevation_change,
            "water_proximity": features.water_proximity,
            "vegetation_density": features.vegetation_density,
            "coastal": features.coastal,
            "has_lake": features.has_lake,
            "has_river": features.has_river,
        }
        if coords:
            row.update({"z": coords[0], "x": coords[1], "y": coords[2]})
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
