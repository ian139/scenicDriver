from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.annotation.annotate_scenic_web import _serve_image_bytes
from scripts.annotation.build_source_shift_batch import build_source_shift_batch

COORDS = [("sne", 14, 4800 + i, 6000 + i) for i in range(100)]


def _manifest(source_root: Path, identity: str) -> pd.DataFrame:
    rows = []
    for region, z, x, y in COORDS:
        satellite = source_root / "satellite" / f"z{z}" / region / f"{x}_{y}.png"
        terrain = source_root / "terrain" / f"z{z}" / region / f"{x}_{y}.png"
        for path, kind in ((satellite, b"sat"), (terrain, b"ter")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{identity}:{x}:{y}:".encode() + kind)
        rows.append(
            {
                "region": region,
                "z": z,
                "x": x,
                "y": y,
                "satellite_path": str(satellite),
                "terrain_path": str(terrain),
                "source_identity": identity,
            }
        )
    return pd.DataFrame(rows)


def test_image_paths_are_relative_and_served_from_one_raw_dir(tmp_path: Path) -> None:
    sources = tmp_path / "data" / "raw" / "sources"
    old = _manifest(sources / "mapbox_v1", "mapbox_v1")
    new = _manifest(sources / "naip_3dep_v1", "naip_3dep_v1")

    summary = build_source_shift_batch(
        old_manifest=old,
        new_manifest=new,
        output_batch_csv=tmp_path / "batch.csv",
        sample_size=100,
    )

    assert Path(summary["image_root"]) == sources.resolve()
    batch = pd.read_csv(tmp_path / "batch.csv")
    assert batch["image_path"].is_unique
    for row in batch.itertuples():
        assert not Path(row.image_path).is_absolute()
        assert row.image_path.split("/")[0] == row.source_identity
        body, _ = _serve_image_bytes(
            summary["image_root"], row.image_path, s3_client=None
        )
        assert body == Path(row.satellite_path).read_bytes()
    unblinded = summary["pair_unblinding"]["pair_0001"]
    pair_rows = batch.loc[batch["pair_id"] == "pair_0001"].set_index("source_variant")
    assert unblinded["old_image_path"] == pair_rows.loc["old", "image_path"]
    assert unblinded["new_image_path"] == pair_rows.loc["new", "image_path"]


def test_explicit_image_root_must_contain_both_sources(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    old = _manifest(sources / "mapbox_v1", "mapbox_v1")
    new = _manifest(tmp_path / "elsewhere" / "naip_3dep_v1", "naip_3dep_v1")

    with pytest.raises(ValueError, match="outside image root"):
        build_source_shift_batch(
            old_manifest=old,
            new_manifest=new,
            output_batch_csv=tmp_path / "batch.csv",
            sample_size=100,
            image_root=sources,
        )
    assert not (tmp_path / "batch.csv").exists()


def test_explicit_image_root_prefixes_identities(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    old = _manifest(raw / "sources" / "mapbox_v1", "mapbox_v1")
    new = _manifest(raw / "sources" / "naip_3dep_v1", "naip_3dep_v1")

    summary = build_source_shift_batch(
        old_manifest=old,
        new_manifest=new,
        output_batch_csv=tmp_path / "batch.csv",
        sample_size=100,
        image_root=raw,
    )

    batch = pd.read_csv(tmp_path / "batch.csv")
    assert summary["image_root"] == str(raw.resolve())
    assert batch["image_path"].str.startswith("sources/").all()
