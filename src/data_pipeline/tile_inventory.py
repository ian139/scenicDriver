"""Deterministic local inventory and PNG validation for paired tiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict

from PIL import Image


def validate_png_image(
    path: Path | str, expected_dimensions: tuple[int, int] | None = None
) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "valid": False,
        "width": None,
        "height": None,
        "reason": None,
    }
    if not path.is_file():
        result["reason"] = "missing"
        return result
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            result["width"], result["height"] = width, height
            if expected_dimensions and (width, height) != expected_dimensions:
                result["reason"] = f"unexpected_dimensions:{width}x{height}"
                return result
        result["valid"] = True
    except Exception as exc:
        result["reason"] = f"invalid_png:{type(exc).__name__}"
    return result


def scan_tile_inventory(
    rows: Iterable[dict[str, Any]],
    *,
    image_root: Path | str = "data/raw/images",
    expected_dimensions: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate canonical satellite/terrain paths for manifest rows."""
    root = Path(image_root)
    output: list[dict[str, Any]] = []
    counts = {
        "coordinates": 0,
        "satellite_valid": 0,
        "terrain_valid": 0,
        "complete_pairs": 0,
        "incomplete_pairs": 0,
        "invalid_files": 0,
    }
    for source_row in rows:
        row = dict(source_row)
        sat = validate_png_image(
            root
            / "satellite"
            / f"z{row['z']}"
            / row["region"]
            / f"{row['x']}_{row['y']}.png",
            expected_dimensions,
        )
        ter = validate_png_image(
            root
            / "terrain"
            / f"z{row['z']}"
            / row["region"]
            / f"{row['x']}_{row['y']}.png",
            expected_dimensions,
        )
        row["satellite_present"] = bool(sat["valid"])
        row["terrain_present"] = bool(ter["valid"])
        row["satellite_reason"] = sat["reason"]
        row["terrain_reason"] = ter["reason"]
        counts["coordinates"] += 1
        counts["satellite_valid"] += int(sat["valid"])
        counts["terrain_valid"] += int(ter["valid"])
        counts["complete_pairs"] += int(sat["valid"] and ter["valid"])
        counts["incomplete_pairs"] += int(not (sat["valid"] and ter["valid"]))
        counts["invalid_files"] += int(sat["present"] and not sat["valid"]) + int(
            ter["present"] and not ter["valid"]
        )
        output.append(row)
    output.sort(key=lambda r: (r["region"], int(r["z"]), int(r["x"]), int(r["y"])))
    return output, counts


def scan_s3_inventory(
    rows: Iterable[dict[str, Any]],
    *,
    bucket: str,
    prefix_root: str = "raw/images",
    s3_client: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge canonical non-empty S3 objects into an existing local inventory."""
    if not bucket or "/" in bucket or bucket.startswith("s3://"):
        raise ValueError("bucket must be a bare S3 bucket name")
    prefix_root = prefix_root.strip().strip("/")
    if not prefix_root:
        raise ValueError("prefix_root is required")
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")

    output = [dict(row) for row in rows]
    regions_by_style: dict[str, set[str]] = defaultdict(set)
    for row in output:
        for style in ("satellite", "terrain"):
            regions_by_style[style].add(str(row["region"]))

    objects: dict[tuple[str, str], dict[str, int]] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for style in ("satellite", "terrain"):
        for region in sorted(regions_by_style[style]):
            prefix = f"{prefix_root}/{style}/z{int(output[0]['z'])}/{region}/"
            found: dict[str, int] = {}
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = str(item.get("Key", ""))
                    size = int(item.get("Size", 0))
                    if size > 0 and key.endswith(".png"):
                        found[Path(key).name] = size
            objects[(style, region)] = found

    for row in output:
        for style in ("satellite", "terrain"):
            filename = f"{int(row['x'])}_{int(row['y'])}.png"
            size = objects.get((style, str(row["region"])), {}).get(filename)
            s3_present = size is not None
            row[f"{style}_s3_present"] = s3_present
            row[f"{style}_s3_bytes"] = size
            row[f"{style}_s3_uri"] = (
                f"s3://{bucket}/{prefix_root}/{style}/z{int(row['z'])}/{row['region']}/{filename}"
                if s3_present
                else ""
            )
            row[f"{style}_present"] = bool(row.get(f"{style}_present")) or s3_present

    counts = {
        "coordinates": len(output),
        "satellite_valid": sum(bool(row.get("satellite_present")) for row in output),
        "terrain_valid": sum(bool(row.get("terrain_present")) for row in output),
        "complete_pairs": sum(
            bool(row.get("satellite_present")) and bool(row.get("terrain_present"))
            for row in output
        ),
        "incomplete_pairs": sum(
            not (
                bool(row.get("satellite_present")) and bool(row.get("terrain_present"))
            )
            for row in output
        ),
        "invalid_files": sum(
            str(row.get(f"{style}_reason", "")).startswith("invalid_png")
            for row in output
            for style in ("satellite", "terrain")
        ),
        "satellite_s3_objects": sum(
            bool(row.get("satellite_s3_present")) for row in output
        ),
        "terrain_s3_objects": sum(
            bool(row.get("terrain_s3_present")) for row in output
        ),
    }
    return output, counts


def build_inventory_report(
    rows: list[dict[str, Any]],
    counts: dict[str, Any],
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "counts": dict(counts),
        "reusable_pairs": sum(
            int(r["satellite_present"] and r["terrain_present"]) for r in rows
        ),
        "failures": sorted(
            failures or [],
            key=lambda x: (
                x.get("region", ""),
                x.get("x", 0),
                x.get("y", 0),
                x.get("style", ""),
            ),
        ),
    }
