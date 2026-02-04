from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.heuristics.labeler import parse_tile_coords, run_heuristic_labeling


def _write_image(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def test_parse_tile_coords():
    assert parse_tile_coords(Path("/tmp/z16/x123/y456.png")) == (16, 123, 456)
    assert parse_tile_coords(Path("z16_x123_y456.png")) == (16, 123, 456)
    assert parse_tile_coords(Path("tile.png")) is None


def test_no_pairs_raises(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw"
    sat_dir = raw_dir / "images" / "satellite" / "z16"
    terrain_dir = raw_dir / "images" / "terrain" / "z16"

    _write_image(sat_dir / "z16_x1_y2.png")
    terrain_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        run_heuristic_labeling(
            satellite_dir=sat_dir,
            terrain_dir=terrain_dir,
            raw_dir=raw_dir,
            max_tiles=10,
            use_classifier=False,
            classifier_best_ckpt="missing.pt",
            classifier_use_resisc45_stats=True,
            default_lat=0.0,
            default_lon=0.0,
            seed=123,
            device="cpu",
        )


def test_paired_tiles_expected_columns(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw"
    sat_dir = raw_dir / "images" / "satellite" / "z16" / "x1"
    terrain_dir = raw_dir / "images" / "terrain" / "z16" / "x1"

    _write_image(sat_dir / "y2.png", seed=1)
    _write_image(terrain_dir / "y2.png", seed=2)

    labels_df, tiles, run_info = run_heuristic_labeling(
        satellite_dir=sat_dir.parent,
        terrain_dir=terrain_dir.parent,
        raw_dir=raw_dir,
        max_tiles=10,
        use_classifier=False,
        classifier_best_ckpt="missing.pt",
        classifier_use_resisc45_stats=True,
        default_lat=0.0,
        default_lon=0.0,
        seed=123,
        device="cpu",
    )

    assert list(labels_df.columns) == ["image_path", "scenic_score", "lat", "lon", "class_id"]
    assert len(labels_df) == 1
    assert len(tiles) == 1
    assert run_info["counts"]["paired"] == 1


def test_deterministic_output(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw"
    sat_dir = raw_dir / "images" / "satellite" / "z16" / "x1"
    terrain_dir = raw_dir / "images" / "terrain" / "z16" / "x1"

    _write_image(sat_dir / "y2.png", seed=7)
    _write_image(terrain_dir / "y2.png", seed=9)

    labels_a, _, _ = run_heuristic_labeling(
        satellite_dir=sat_dir.parent,
        terrain_dir=terrain_dir.parent,
        raw_dir=raw_dir,
        max_tiles=10,
        use_classifier=False,
        classifier_best_ckpt="missing.pt",
        classifier_use_resisc45_stats=True,
        default_lat=0.0,
        default_lon=0.0,
        seed=42,
        device="cpu",
    )

    labels_b, _, _ = run_heuristic_labeling(
        satellite_dir=sat_dir.parent,
        terrain_dir=terrain_dir.parent,
        raw_dir=raw_dir,
        max_tiles=10,
        use_classifier=False,
        classifier_best_ckpt="missing.pt",
        classifier_use_resisc45_stats=True,
        default_lat=0.0,
        default_lon=0.0,
        seed=42,
        device="cpu",
    )

    pd.testing.assert_frame_equal(labels_a, labels_b)
