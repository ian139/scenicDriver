from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.ingest.plan_active_learning_region import plan_run
from src.data_pipeline.region_planning import (
    enumerate_bbox_tiles,
    get_builtin_region_spec,
    parse_and_validate_region_spec,
)
from src.data_pipeline.tile_inventory import validate_png_image


def test_builtin_spec_covers_baseline_and_stays_under_budget():
    planned = parse_and_validate_region_spec(get_builtin_region_spec())
    assert planned["nen_tile_count"] > 0
    assert planned["unique_coordinates_count"] <= 370_000
    assert planned["total_rasters_count"] == planned["unique_coordinates_count"] * 2


def test_overlapping_regions_dedupe_coordinates():
    baseline = get_builtin_region_spec()["regions"][0]
    spec = {"version": 1, "regions": [baseline, baseline]}
    planned = parse_and_validate_region_spec(spec)
    assert planned["unique_coordinates_count"] == planned["nen_tile_count"]


def test_north_expansion_rejected():
    spec = get_builtin_region_spec()
    spec["regions"].append({"name": "north", "type": "bbox", "bbox": {"min_lat": 47.5, "min_lon": -73.5, "max_lat": 48.0, "max_lon": -73.0}})
    with pytest.raises(ValueError, match="North"):
        parse_and_validate_region_spec(spec)


def test_budget_rejected_before_output(tmp_path: Path):
    spec = {"version": 1, "regions": [get_builtin_region_spec()["regions"][0]]}
    with pytest.raises(ValueError, match="budget cap"):
        plan_run(run_name="over", spec=spec, output_root=tmp_path / "runs", image_root=tmp_path / "images", budget=1)
    assert not (tmp_path / "runs").exists()


def test_inventory_reuses_valid_pair_and_manifest_contract(tmp_path: Path):
    image_root = tmp_path / "images"
    row = enumerate_bbox_tiles(42.49, -73.52, 42.50, -73.51, zoom=14)[0]
    x, y = row
    region_dir_sat = image_root / "satellite" / "z14" / "new_england_north"
    region_dir_ter = image_root / "terrain" / "z14" / "new_england_north"
    region_dir_sat.mkdir(parents=True)
    region_dir_ter.mkdir(parents=True)
    Image.new("RGB", (256, 256), (1, 2, 3)).save(region_dir_sat / f"{x}_{y}.png")
    Image.new("RGB", (256, 256), (1, 2, 3)).save(region_dir_ter / f"{x}_{y}.png")
    assert validate_png_image(region_dir_sat / f"{x}_{y}.png")["valid"]

    spec = {"version": 1, "regions": [{"name": "new_england_north", "type": "bbox", "bbox": {"min_lat": 42.488301979602255, "min_lon": -73.5205078125, "max_lat": 47.50235895196859, "max_lon": -66.796875}}]}
    result = plan_run(run_name="inventory", spec=spec, output_root=tmp_path / "runs", image_root=image_root)
    assert result["inventory_report"]["counts"]["complete_pairs"] >= 1
    with open(tmp_path / "runs" / "inventory" / "tile_manifest.csv", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["region", "z", "x", "y", "lat", "lon", "satellite_path", "terrain_path", "satellite_present", "terrain_present"]
