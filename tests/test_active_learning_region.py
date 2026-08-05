from __future__ import annotations

import csv
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.ingest import plan_active_learning_region as planner
from scripts.ingest.plan_active_learning_region import acquire_missing, plan_run
from src.data_pipeline.region_planning import (
    enumerate_bbox_tiles,
    get_builtin_region_spec,
    parse_and_validate_region_spec,
)
from src.data_pipeline.tile_inventory import scan_s3_inventory, validate_png_image


def test_builtin_spec_covers_baseline_and_stays_under_budget():
    planned = parse_and_validate_region_spec(get_builtin_region_spec())
    assert planned["nen_tile_count"] > 0
    assert planned["unique_coordinates_count"] <= 370_000
    assert planned["unique_coordinates_count"] >= 360_000
    assert planned["total_rasters_count"] == planned["unique_coordinates_count"] * 2
    spec = get_builtin_region_spec()
    assert spec["geographic_source"]
    assert spec["included_jurisdictions"]
    assert spec["known_non_target_coverage"]
    assert spec["limitations"]


def test_overlapping_regions_dedupe_coordinates():
    baseline = get_builtin_region_spec()["regions"][0]
    spec = {"version": 1, "regions": [baseline, baseline]}
    planned = parse_and_validate_region_spec(spec)
    assert planned["unique_coordinates_count"] == planned["nen_tile_count"]


def test_north_expansion_rejected():
    spec = get_builtin_region_spec()
    spec["regions"].append(
        {
            "name": "north",
            "type": "bbox",
            "bbox": {
                "min_lat": 47.5,
                "min_lon": -73.5,
                "max_lat": 48.0,
                "max_lon": -73.0,
            },
        }
    )
    with pytest.raises(ValueError, match="North"):
        parse_and_validate_region_spec(spec)


def test_budget_rejected_before_output(tmp_path: Path):
    spec = {"version": 1, "regions": [get_builtin_region_spec()["regions"][0]]}
    with pytest.raises(ValueError, match="budget cap"):
        plan_run(
            run_name="over",
            spec=spec,
            output_root=tmp_path / "runs",
            image_root=tmp_path / "images",
            budget=1,
        )
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

    spec = {
        "version": 1,
        "regions": [
            {
                "name": "new_england_north",
                "type": "bbox",
                "bbox": {
                    "min_lat": 42.488301979602255,
                    "min_lon": -73.5205078125,
                    "max_lat": 47.50235895196859,
                    "max_lon": -66.796875,
                },
            }
        ],
    }
    result = plan_run(
        run_name="inventory",
        spec=spec,
        output_root=tmp_path / "runs",
        image_root=image_root,
    )
    assert result["inventory_report"]["counts"]["complete_pairs"] >= 1
    with open(
        tmp_path / "runs" / "inventory" / "tile_manifest.csv",
        newline="",
        encoding="utf-8",
    ) as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "region",
            "z",
            "x",
            "y",
            "lat",
            "lon",
            "image_path",
            "satellite_path",
            "terrain_path",
            "satellite_present",
            "terrain_present",
            "satellite_s3_present",
            "terrain_s3_present",
            "satellite_s3_uri",
            "terrain_s3_uri",
        ]


def test_s3_inventory_reuses_nonempty_canonical_pair() -> None:
    class Paginator:
        def paginate(self, *, Bucket: str, Prefix: str):
            style = "satellite" if "/satellite/" in Prefix else "terrain"
            assert Bucket == "scenic"
            yield {
                "Contents": [
                    {"Key": f"raw/images/{style}/z14/west/10_20.png", "Size": 123}
                ]
            }

    class Client:
        def get_paginator(self, name: str):
            assert name == "list_objects_v2"
            return Paginator()

    rows, counts = scan_s3_inventory(
        [
            {
                "region": "west",
                "z": 14,
                "x": 10,
                "y": 20,
                "satellite_present": False,
                "terrain_present": False,
            }
        ],
        bucket="scenic",
        s3_client=Client(),
    )
    assert counts["complete_pairs"] == 1
    assert (
        rows[0]["satellite_s3_uri"]
        == "s3://scenic/raw/images/satellite/z14/west/10_20.png"
    )


def test_acquisition_uses_bounded_workers_and_writes_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_lock = threading.Lock()
    state = {"active": 0, "maximum": 0}

    class Source:
        DEFAULT_RATE_LIMIT = 10

        def __init__(self, **kwargs):
            assert kwargs["rate_limit"] == 5

        def get_tile(self, x: int, y: int, zoom: int):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.02)
            with state_lock:
                state["active"] -= 1
            return SimpleNamespace(image=np.zeros((8, 8, 3), dtype=np.uint8))

    monkeypatch.setattr(planner, "MapboxTileSource", Source)
    rows = [
        {
            "region": "fixture",
            "z": 14,
            "x": index,
            "y": 20,
            "satellite_path": str(tmp_path / "satellite" / f"{index}.png"),
            "terrain_path": str(tmp_path / "terrain" / f"{index}.png"),
            "satellite_present": False,
            "terrain_present": False,
        }
        for index in range(4)
    ]
    failures: list[dict] = []

    acquire_missing(
        rows,
        image_root=tmp_path,
        zoom=14,
        failures=failures,
        max_workers=2,
    )

    assert not failures
    assert state["maximum"] == 2
    assert all(row["satellite_present"] and row["terrain_present"] for row in rows)
    assert all(Path(row["satellite_path"]).is_file() for row in rows)
    assert all(Path(row["terrain_path"]).is_file() for row in rows)
