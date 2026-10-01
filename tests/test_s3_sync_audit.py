from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest


def test_lifecycle_rule_targets_canonical_raw_images_prefix() -> None:
    lifecycle = json.loads(Path("config/s3_lifecycle.json").read_text())
    enabled_rules = [
        rule for rule in lifecycle["Rules"] if rule.get("Status") == "Enabled"
    ]

    raw_rules = [
        rule for rule in enabled_rules if rule.get("ID") == "raw-images-transition"
    ]

    assert len(raw_rules) == 1
    raw_rule = raw_rules[0]
    assert raw_rule["Filter"]["Prefix"] == "raw/images/"
    assert raw_rule["Transitions"] == [
        {"Days": 30, "StorageClass": "STANDARD_IA"},
        {"Days": 180, "StorageClass": "GLACIER"},
    ]


def test_labeler_s3_prefix_from_local_dir_matches_data_contract() -> None:
    from src.heuristics.labeler import _s3_prefix_from_local_dir

    assert (
        _s3_prefix_from_local_dir(Path("data/raw/images/satellite/z14/masswhites"))
        == "raw/images/satellite/z14/masswhites"
    )
    assert (
        _s3_prefix_from_local_dir(Path("data/raw/images/terrain/z14/masswhites"))
        == "raw/images/terrain/z14/masswhites"
    )


def test_relative_paths_reject_non_raw_s3_keys() -> None:
    from src.heuristics.labeler import S3ImageRef, _relative_paths_or_raise

    with pytest.raises(ValueError, match="raw/"):
        _relative_paths_or_raise(
            S3ImageRef("images/satellite/z14/masswhites/1_2.png"),
            S3ImageRef("images/terrain/z14/masswhites/1_2.png"),
            Path("data/raw"),
            s3_only=True,
            s3_bucket="bucket",
        )


def _write_aws_shim(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "aws-calls.txt"
    shim = bin_dir / "aws"
    shim.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {calls_path}\nexit 0\n"
    )
    shim.chmod(0o755)
    return calls_path


def test_s3_sync_dry_run_uses_expected_prefixes_without_aws(tmp_path: Path) -> None:
    calls_path = _write_aws_shim(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    env["SCENIC_S3_BUCKET"] = "unit-test-bucket"
    env["DRY_RUN"] = "1"

    result = subprocess.run(
        ["bash", "scripts/ingest/s3_sync.sh"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "Syncing raw tiles...",
        "Syncing processed artifacts...",
        "Syncing models (optional)...",
        "Done.",
    ]
    sync_calls = calls_path.read_text().splitlines()
    assert sync_calls == [
        "s3 sync data/raw/ s3://unit-test-bucket/raw/ --dryrun --exclude *.tif --exclude *.npy",
        "s3 sync data/processed/ s3://unit-test-bucket/processed/ --dryrun",
        "s3 sync models/ s3://unit-test-bucket/models/ --dryrun",
    ]


def test_s3_sync_requires_bucket(tmp_path: Path) -> None:
    _write_aws_shim(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    env.pop("SCENIC_S3_BUCKET", None)

    result = subprocess.run(
        ["bash", "scripts/ingest/s3_sync.sh"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 1
    assert "SCENIC_S3_BUCKET is required" in result.stdout


def test_iter_objects_uses_canonical_paginator_across_pages() -> None:
    from src.data_pipeline.s3 import _iter_objects

    calls: list[tuple[str, dict[str, str]]] = []

    class FakePaginator:
        def paginate(self, **kwargs: str) -> list[dict[str, list[dict[str, str]]]]:
            calls.append(("paginate", kwargs))
            return [
                {"Contents": [{"Key": "prefix/first.bin"}]},
                {"Contents": [{"Key": "prefix/second.bin"}]},
            ]

    class FakeS3Client:
        def get_paginator(self, operation: str) -> FakePaginator:
            calls.append(("get_paginator", {"operation": operation}))
            return FakePaginator()

    assert list(_iter_objects(FakeS3Client(), "bucket", "prefix/")) == [
        {"Key": "prefix/first.bin"},
        {"Key": "prefix/second.bin"},
    ]
    assert calls == [
        ("get_paginator", {"operation": "list_objects_v2"}),
        ("paginate", {"Bucket": "bucket", "Prefix": "prefix/"}),
    ]


def test_report_thumbnails_use_explicit_s3_raw_dir_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.heuristics.report import _attach_thumbnails

    png_buffer = BytesIO()
    Image.new("RGB", (4, 4), color="green").save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()
    calls: list[tuple[str, str]] = []

    class FakeS3Client:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
            calls.append((Bucket, Key))
            return {"Body": BytesIO(png_bytes)}

    fake_boto3 = SimpleNamespace(client=lambda service: FakeS3Client())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setenv("SCENIC_S3_ONLY", "1")
    monkeypatch.setenv("SCENIC_S3_BUCKET", "env-bucket")

    thumbs_dir = tmp_path / "thumbs"
    thumbs_dir.mkdir()
    tiles = [
        {
            "image_path": "images/satellite/z14/masswhites/1_2.png",
            "scenic_score": 5.0,
        }
    ]

    _attach_thumbnails(
        tiles=tiles,
        raw_dir="s3://unit-test-bucket/raw",
        thumbs_dir=thumbs_dir,
        thumb_size=64,
    )

    assert calls == [
        (
            "unit-test-bucket",
            "raw/images/satellite/z14/masswhites/1_2.png",
        )
    ]
    assert tiles[0]["index"] == 0
    assert tiles[0]["thumb"] == "thumbs/00000.jpg"
    assert (thumbs_dir / "00000.jpg").is_file()


class _DummyLabelsFrame:
    def to_csv(self, path: Path, index: bool = False) -> None:
        path.write_text(
            "image_path,scenic_score,class_id\nraw/images/satellite/z14/new_england_north/1_2.png,5.0,-1\n"
        )


def test_heuristic_report_region_all_tiles_delegates_uncapped_s3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.reports import heuristic_report, heuristic_report_region

    captured: dict[str, object] = {}

    def fake_labeling(
        **kwargs: object,
    ) -> tuple[_DummyLabelsFrame, list[dict[str, object]], dict[str, object]]:
        captured.update(kwargs)
        return (
            _DummyLabelsFrame(),
            [],
            {"counts": {}, "config": {"max_tiles": kwargs["max_tiles"]}},
        )

    def fake_report(**kwargs: object) -> dict[str, object]:
        return {"summary": {}, "histogram": {}}

    cfg = heuristic_report.HeuristicLabelerConfig()
    cfg.processed_dir = str(tmp_path)
    monkeypatch.setattr(heuristic_report, "HeuristicLabelerConfig", lambda: cfg)
    monkeypatch.setattr(heuristic_report, "run_heuristic_labeling", fake_labeling)
    monkeypatch.setattr(heuristic_report, "build_report", fake_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "heuristic_report_region.py",
            "--region",
            "new_england_north",
            "--zoom",
            "14",
            "--raw-dir",
            "s3://scenicdriver-data/raw",
            "--s3-only",
            "--write-labels",
            "--all-tiles",
            "--no-classifier",
        ],
    )

    heuristic_report_region.main()

    assert (
        captured["satellite_dir"] == "data/raw/images/satellite/z14/new_england_north"
    )
    assert captured["terrain_dir"] == "data/raw/images/terrain/z14/new_england_north"
    assert captured["raw_dir"] == "s3://scenicdriver-data/raw"
    assert captured["max_tiles"] is None


def test_heuristic_report_preview_preserves_256_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.reports import heuristic_report, heuristic_report_region

    captured: dict[str, object] = {}

    def fake_labeling(
        **kwargs: object,
    ) -> tuple[_DummyLabelsFrame, list[dict[str, object]], dict[str, object]]:
        captured.update(kwargs)
        return (
            _DummyLabelsFrame(),
            [],
            {"counts": {}, "config": {"max_tiles": kwargs["max_tiles"]}},
        )

    def fake_report(**kwargs: object) -> dict[str, object]:
        return {"summary": {}, "histogram": {}}

    cfg = heuristic_report.HeuristicLabelerConfig()
    cfg.processed_dir = str(tmp_path)
    monkeypatch.setattr(heuristic_report, "HeuristicLabelerConfig", lambda: cfg)
    monkeypatch.setattr(heuristic_report, "run_heuristic_labeling", fake_labeling)
    monkeypatch.setattr(heuristic_report, "build_report", fake_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "heuristic_report_region.py",
            "--region",
            "new_england_north",
            "--zoom",
            "14",
            "--preview",
            "--no-classifier",
        ],
    )

    heuristic_report_region.main()

    assert captured["max_tiles"] == 256


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("human_override", 3.0),
        ("heuristic", 1.5),
        ("manual", 0.5),
        ("weak", 0.5),
        (None, 0.5),
        (float("nan"), 0.5),
    ],
)
def test_regression_dataset_sample_weight_uses_canonical_label_sources(
    source: str | None, expected: float
) -> None:
    from scripts.modeling.export_regression_dataset import _sample_weight_for_row

    assert (
        _sample_weight_for_row(
            pd.Series({"label_source": source}),
            sample_weight_column=None,
            label_source_column="label_source",
            has_label_source=True,
            has_scenic_human=False,
            human_weight=3.0,
            heuristic_weight=1.5,
            default_weight=0.5,
        )
        == expected
    )


def test_regression_dataset_sample_weight_preserves_explicit_and_missing_behavior() -> (
    None
):
    from scripts.modeling.export_regression_dataset import _sample_weight_for_row

    kwargs = {
        "sample_weight_column": "sample_weight",
        "label_source_column": "label_source",
        "has_label_source": True,
        "has_scenic_human": False,
        "human_weight": 3.0,
        "heuristic_weight": 1.5,
        "default_weight": 0.5,
    }
    assert _sample_weight_for_row(pd.Series({"sample_weight": 2.25}), **kwargs) == 2.25
    with pytest.raises(ValueError, match="sample_weight"):
        _sample_weight_for_row(pd.Series({"sample_weight": float("nan")}), **kwargs)


def test_classifier_resume_requires_canonical_best_acc() -> None:
    from src.classifier.train import _best_acc_from_checkpoint

    assert _best_acc_from_checkpoint({"best_acc": 0.8, "val_acc": 0.1}) == 0.8
    with pytest.raises(ValueError, match="required 'best_acc'"):
        _best_acc_from_checkpoint({"val_acc": 0.8})


def test_upload_prefix_verified_success(tmp_path: Path) -> None:
    import hashlib
    from src.data_pipeline.s3 import upload_prefix

    local_file = tmp_path / "test.bin"
    data = b"hello s3 verification"
    local_file.write_bytes(data)
    md5_hex = hashlib.md5(data).hexdigest()

    uploaded: list[tuple[str, str, str]] = []
    head_calls: list[tuple[str, str]] = []

    class FakeS3Client:
        def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
            uploaded.append((Filename, Bucket, Key))

        def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
            head_calls.append((Bucket, Key))
            return {
                "ContentLength": len(data),
                "ETag": f'"{md5_hex}"',
            }

    count = upload_prefix(local_file, "mybucket", "artifacts/v1", client=FakeS3Client())
    assert count == 1
    assert uploaded == [(str(local_file), "mybucket", "artifacts/v1/test.bin")]
    assert head_calls == [("mybucket", "artifacts/v1/test.bin")]


def test_upload_prefix_missing_remote_object(tmp_path: Path) -> None:
    from src.data_pipeline.s3 import upload_prefix

    local_file = tmp_path / "missing.bin"
    local_file.write_bytes(b"payload")

    class FakeS3Client:
        def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
            pass

        def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
            raise RuntimeError("404 NoSuchKey")

    with pytest.raises(RuntimeError, match="Remote verification failed"):
        upload_prefix(local_file, "mybucket", "prefix", client=FakeS3Client())


def test_upload_prefix_size_mismatch(tmp_path: Path) -> None:
    from src.data_pipeline.s3 import upload_prefix

    local_file = tmp_path / "data.bin"
    local_file.write_bytes(b"payload")

    class FakeS3Client:
        def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
            pass

        def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
            return {"ContentLength": 999}

    with pytest.raises(ValueError, match="size mismatch"):
        upload_prefix(local_file, "mybucket", "prefix", client=FakeS3Client())


def test_upload_prefix_checksum_mismatch(tmp_path: Path) -> None:
    from src.data_pipeline.s3 import upload_prefix

    local_file = tmp_path / "data.bin"
    local_file.write_bytes(b"payload")

    class FakeS3Client:
        def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
            pass

        def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
            return {
                "ContentLength": len(b"payload"),
                "ETag": '"00000000000000000000000000000000"',
            }

    with pytest.raises(ValueError, match="checksum mismatch"):
        upload_prefix(local_file, "mybucket", "prefix", client=FakeS3Client())


def test_upload_prefix_directory_deterministic_and_relative_keys(
    tmp_path: Path,
) -> None:
    from src.data_pipeline.s3 import upload_prefix

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "b.txt").write_bytes(b"file b")
    sub = src_dir / "sub"
    sub.mkdir()
    (sub / "a.txt").write_bytes(b"file a")

    uploaded_keys: list[str] = []
    verified_keys: list[str] = []
    sizes = {
        "prefix/b.txt": len(b"file b"),
        "prefix/sub/a.txt": len(b"file a"),
    }

    class FakeS3Client:
        def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
            uploaded_keys.append(Key)

        def head_object(self, Bucket: str, Key: str) -> dict[str, object]:
            verified_keys.append(Key)
            return {"ContentLength": sizes[Key]}

    count = upload_prefix(src_dir, "bucket", "prefix", client=FakeS3Client())
    assert count == 2
    assert uploaded_keys == ["prefix/b.txt", "prefix/sub/a.txt"]
    assert verified_keys == ["prefix/b.txt", "prefix/sub/a.txt"]


@pytest.mark.parametrize(
    "key",
    ["a/../../x", "/etc/x", r"a\b", "a/../x", "C:/outside.tif", "../folder/"],
)
def test_download_prefix_rejects_unsafe_keys_without_writes(
    tmp_path: Path, key: str
) -> None:
    from src.data_pipeline.s3 import download_prefix

    downloads: list[str] = []

    class FakePaginator:
        def paginate(self, **kwargs: str) -> list[dict]:
            return [{"Contents": [{"Key": "ok/a.tif"}, {"Key": key}]}]

    class FakeS3Client:
        def get_paginator(self, operation: str) -> FakePaginator:
            return FakePaginator()

        def download_file(self, bucket: str, key: str, filename: str) -> None:
            downloads.append(key)

    dest = tmp_path / "download"
    with pytest.raises(ValueError, match="Unsafe S3"):
        download_prefix("bucket", "", dest, client=FakeS3Client())

    assert downloads == []
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("symlink_kind", ["directory", "file"])
def test_download_prefix_rejects_resolved_target_outside_destination(
    tmp_path: Path, symlink_kind: str
) -> None:
    from src.data_pipeline.s3 import download_prefix

    dest = tmp_path / "download"
    dest.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "a.tif"
    sentinel.write_bytes(b"unchanged")
    if symlink_kind == "directory":
        (dest / "ok").symlink_to(outside, target_is_directory=True)
    else:
        (dest / "ok").mkdir()
        (dest / "ok" / "a.tif").symlink_to(sentinel)
    downloads: list[str] = []

    class FakePaginator:
        def paginate(self, **kwargs: str) -> list[dict]:
            return [{"Contents": [{"Key": "first.tif"}, {"Key": "ok/a.tif"}]}]

    class FakeS3Client:
        def get_paginator(self, operation: str) -> FakePaginator:
            return FakePaginator()

        def download_file(self, bucket: str, key: str, filename: str) -> None:
            downloads.append(key)

    with pytest.raises(ValueError, match="Unsafe S3"):
        download_prefix("bucket", "", dest, client=FakeS3Client())

    assert downloads == []
    assert not (dest / "first.tif").exists()
    assert sentinel.read_bytes() == b"unchanged"
    assert list(outside.iterdir()) == [sentinel]


@pytest.mark.parametrize("prefix", ["", "raw/", "raw"])
def test_download_prefix_writes_normal_relative_key(
    tmp_path: Path, prefix: str
) -> None:
    from src.data_pipeline.s3 import download_prefix

    key = f"{prefix.rstrip('/')}/ok/a.tif" if prefix else "ok/a.tif"

    class FakePaginator:
        def paginate(self, **kwargs: str) -> list[dict]:
            return [{"Contents": [{"Key": key}]}]

    class FakeS3Client:
        def get_paginator(self, operation: str) -> FakePaginator:
            return FakePaginator()

        def download_file(self, bucket: str, key: str, filename: str) -> None:
            Path(filename).write_bytes(b"tile payload")

    dest = tmp_path / "download"
    assert download_prefix("bucket", prefix, dest, client=FakeS3Client()) == 1
    assert (dest / "ok" / "a.tif").read_bytes() == b"tile payload"


def test_inventory_does_not_attribute_or_reuse_unproven_valid_tiles(
    tmp_path: Path,
) -> None:
    from src.data_pipeline.tile_inventory import (
        build_inventory_report,
        scan_tile_inventory,
    )

    for style in ("satellite", "terrain"):
        path = tmp_path / style / "z14" / "fixture" / "1_2.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4), color="green").save(path)
    source = {
        "region": "fixture",
        "z": 14,
        "x": 1,
        "y": 2,
        "satellite_provider": "usda_naip",
        "satellite_collection": "naip-visualization",
        "satellite_asset_ids": '["planned-naip-asset"]',
        "terrain_provider": "usgs",
        "terrain_collection": "usgs-3dep-13as",
        "terrain_asset_ids": '["planned-3dep-asset"]',
    }

    rows, counts = scan_tile_inventory([source], image_root=tmp_path)

    row = rows[0]
    assert counts["complete_pairs"] == 1
    assert counts["reusable_pairs"] == 0
    assert not row["satellite_present"]
    assert not row["terrain_present"]
    for style in ("satellite", "terrain"):
        assert row[f"{style}_reason"] == "missing_acquisition_provenance"
        for field in ("provider", "collection", "asset_ids"):
            assert not row[f"{style}_{field}"]
    assert not row.get("acquisition_tile_identity_sha256")
    assert build_inventory_report(rows, counts)["reusable_pairs"] == 0


@pytest.mark.parametrize("changed", ["", "satellite", "terrain", "identity"])
def test_inventory_reuse_requires_matching_acquisition_and_output_hashes(
    tmp_path: Path, changed: str
) -> None:
    import hashlib

    from src.data_pipeline.source_contracts import compute_acquisition_tile_identity
    from src.data_pipeline.tile_inventory import scan_tile_inventory

    row = {
        "region": "fixture",
        "z": 14,
        "x": 1,
        "y": 2,
        "grid_sha256": "1" * 64,
        "source_contract_sha256": "2" * 64,
        "preprocessing_contract_sha256": "3" * 64,
        "satellite_provider": "usda_naip",
        "terrain_provider": "usgs",
    }
    paths = {}
    for style in ("satellite", "terrain"):
        path = tmp_path / style / "z14" / "fixture" / "1_2.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4), color="green").save(path)
        paths[style] = path
        row[f"{style}_output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row["acquisition_tile_identity_sha256"] = compute_acquisition_tile_identity(
        z=14,
        x=1,
        y=2,
        region="fixture",
        target_grid_sha256=row["grid_sha256"],
        satellite_output_sha256=row["satellite_output_sha256"],
        terrain_output_sha256=row["terrain_output_sha256"],
        source_contract_sha256=row["source_contract_sha256"],
        preprocessing_contract_sha256=row["preprocessing_contract_sha256"],
    ).sha256()
    if changed in paths:
        Image.new("RGB", (4, 4), color="blue").save(paths[changed])
    elif changed == "identity":
        row["acquisition_tile_identity_sha256"] = "f" * 64

    rows, counts = scan_tile_inventory([row], image_root=tmp_path)

    if not changed:
        assert counts["reusable_pairs"] == 1
        assert rows[0]["satellite_present"]
        assert rows[0]["terrain_present"]
        assert rows[0]["satellite_provider"] == "usda_naip"
        assert rows[0]["terrain_provider"] == "usgs"
    else:
        assert counts["reusable_pairs"] == 0
        assert not rows[0].get("acquisition_tile_identity_sha256")
        if changed == "identity":
            assert not rows[0]["satellite_present"]
            assert not rows[0]["terrain_present"]
        else:
            assert not rows[0][f"{changed}_present"]


def test_inventory_report_does_not_reuse_unproven_s3_pairs() -> None:
    from src.data_pipeline.tile_inventory import build_inventory_report

    report = build_inventory_report(
        [{"satellite_s3_present": True, "terrain_s3_present": True}], {}
    )

    assert report["reusable_pairs"] == 0
