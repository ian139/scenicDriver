from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def test_lifecycle_rule_targets_canonical_raw_images_prefix() -> None:
    lifecycle = json.loads(Path("config/s3_lifecycle.json").read_text())
    enabled_rules = [rule for rule in lifecycle["Rules"] if rule.get("Status") == "Enabled"]

    raw_rules = [
        rule
        for rule in enabled_rules
        if rule.get("ID") == "raw-images-transition"
    ]

    assert len(raw_rules) == 1
    raw_rule = raw_rules[0]
    assert raw_rule["Filter"]["Prefix"] == "raw/images/"
    assert raw_rule["Transitions"] == [
        {"Days": 30, "StorageClass": "STANDARD_IA"},
        {"Days": 180, "StorageClass": "GLACIER"},
    ]


def test_download_bbox_resolves_canonical_prefixes() -> None:
    from scripts.ingest.download_bbox_tiles import _resolve_s3_prefix

    output_dir = Path("data/raw/images/satellite/z14/masswhites")

    assert (
        _resolve_s3_prefix(
            prefix=None,
            style="mapbox.satellite",
            output_dir=output_dir,
            zoom=14,
        )
        == "raw/images/satellite/z14/masswhites"
    )
    assert (
        _resolve_s3_prefix(
            prefix="terrain",
            style="mapbox.terrain-rgb",
            output_dir=Path("data/raw/images/terrain/z14/masswhites"),
            zoom=14,
        )
        == "raw/images/terrain/z14/masswhites"
    )
    assert (
        _resolve_s3_prefix(
            prefix="images/satellite",
            style="mapbox.satellite",
            output_dir=output_dir,
            zoom=14,
        )
        == "raw/images/satellite/z14/masswhites"
    )


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
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {calls_path}\n"
        "exit 0\n"
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
        path.write_text("image_path,scenic_score,class_id\nraw/images/satellite/z14/new_england_north/1_2.png,5.0,-1\n")


def test_heuristic_report_region_all_tiles_delegates_uncapped_s3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.reports import heuristic_report, heuristic_report_region

    captured: dict[str, object] = {}

    def fake_labeling(**kwargs: object) -> tuple[_DummyLabelsFrame, list[dict[str, object]], dict[str, object]]:
        captured.update(kwargs)
        return _DummyLabelsFrame(), [], {"counts": {}, "config": {"max_tiles": kwargs["max_tiles"]}}

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

    assert captured["satellite_dir"] == "data/raw/images/satellite/z14/new_england_north"
    assert captured["terrain_dir"] == "data/raw/images/terrain/z14/new_england_north"
    assert captured["raw_dir"] == "s3://scenicdriver-data/raw"
    assert captured["max_tiles"] is None


def test_heuristic_report_preview_preserves_256_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.reports import heuristic_report, heuristic_report_region

    captured: dict[str, object] = {}

    def fake_labeling(**kwargs: object) -> tuple[_DummyLabelsFrame, list[dict[str, object]], dict[str, object]]:
        captured.update(kwargs)
        return _DummyLabelsFrame(), [], {"counts": {}, "config": {"max_tiles": kwargs["max_tiles"]}}

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