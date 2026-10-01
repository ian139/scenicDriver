"""Deterministic local inventory and PNG validation for paired tiles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from src.data_pipeline.source_contracts import compute_acquisition_tile_identity


def acquisition_provenance_reason(row: Mapping[str, Any]) -> str | None:
    """Return why acquisition fields are unproven, or None for a matching identity."""
    fields = (
        "grid_sha256",
        "source_contract_sha256",
        "preprocessing_contract_sha256",
        "satellite_output_sha256",
        "terrain_output_sha256",
        "acquisition_tile_identity_sha256",
    )
    if not all(row.get(field) for field in fields):
        return "missing_acquisition_provenance"
    try:
        identity = compute_acquisition_tile_identity(
            z=int(row["z"]),
            x=int(row["x"]),
            y=int(row["y"]),
            region=row["region"],
            target_grid_sha256=row["grid_sha256"],
            satellite_output_sha256=row["satellite_output_sha256"],
            terrain_output_sha256=row["terrain_output_sha256"],
            source_contract_sha256=row["source_contract_sha256"],
            preprocessing_contract_sha256=row["preprocessing_contract_sha256"],
        ).sha256()
    except (KeyError, TypeError, ValueError):
        return "invalid_acquisition_provenance"
    if identity != row["acquisition_tile_identity_sha256"]:
        return "acquisition_identity_mismatch"
    return None


def _proven_output_reason(
    row: dict[str, Any],
    style: str,
    validation: dict[str, Any],
    provenance_reason: str | None,
) -> str | None:
    if not validation["valid"]:
        return validation["reason"]
    if provenance_reason:
        return provenance_reason
    with Path(validation["path"]).open("rb") as stream:
        output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if output_hash != row[f"{style}_output_sha256"]:
        return "output_sha256_mismatch"
    return None


def _clear_unproven_attribution(row: dict[str, Any], style: str) -> None:
    for suffix in (
        "provider",
        "collection",
        "asset_ids",
        "acquisition_year",
        "capture_date",
        "license",
        "attribution",
        "source_checksums",
        "output_sha256",
        "valid_fraction",
        "vertical_datum",
        "native_resolution",
    ):
        field = f"{style}_{suffix}"
        if field in row:
            row[field] = ""


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
    """Validate file bytes and acquisition provenance before admitting local tiles."""
    root = Path(image_root)
    output: list[dict[str, Any]] = []
    counts = {
        "coordinates": 0,
        "satellite_valid": 0,
        "terrain_valid": 0,
        "complete_pairs": 0,
        "reusable_pairs": 0,
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
        provenance_reason = acquisition_provenance_reason(row)
        satellite_reason = _proven_output_reason(row, "satellite", sat, provenance_reason)
        terrain_reason = _proven_output_reason(row, "terrain", ter, provenance_reason)
        row["satellite_present"] = satellite_reason is None
        row["terrain_present"] = terrain_reason is None
        row["satellite_reason"] = satellite_reason
        row["terrain_reason"] = terrain_reason
        if provenance_reason or any(
            reason == "output_sha256_mismatch"
            for reason in (satellite_reason, terrain_reason)
        ):
            row["acquisition_tile_identity_sha256"] = ""
        for style, validation, reason in (
            ("satellite", sat, satellite_reason),
            ("terrain", ter, terrain_reason),
        ):
            if validation["present"] and reason is not None:
                _clear_unproven_attribution(row, style)
                if "mosaic_contributions" in row:
                    row["mosaic_contributions"] = ""
        counts["coordinates"] += 1
        counts["satellite_valid"] += int(sat["valid"])
        counts["terrain_valid"] += int(ter["valid"])
        counts["complete_pairs"] += int(sat["valid"] and ter["valid"])
        counts["reusable_pairs"] += int(
            row["satellite_present"] and row["terrain_present"]
        )
        counts["incomplete_pairs"] += int(not (sat["valid"] and ter["valid"]))
        counts["invalid_files"] += int(sat["present"] and not sat["valid"]) + int(
            ter["present"] and not ter["valid"]
        )
        output.append(row)
    output.sort(key=lambda r: (r["region"], int(r["z"]), int(r["x"]), int(r["y"])))
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
            int(
                (
                    bool(r.get("satellite_present"))
                    or bool(r.get("satellite_s3_present"))
                )
                and (
                    bool(r.get("terrain_present")) or bool(r.get("terrain_s3_present"))
                )
                and acquisition_provenance_reason(r) is None
            )
            for r in rows
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
