"""Deterministic local inventory and PNG validation for paired tiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def validate_png_image(path: Path | str, expected_dimensions: tuple[int, int] | None = None) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {"path": str(path), "present": path.is_file(), "valid": False, "width": None, "height": None, "reason": None}
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
    counts = {"coordinates": 0, "satellite_valid": 0, "terrain_valid": 0, "complete_pairs": 0, "incomplete_pairs": 0, "invalid_files": 0}
    for source_row in rows:
        row = dict(source_row)
        sat = validate_png_image(root / "satellite" / f"z{row['z']}" / row["region"] / f"{row['x']}_{row['y']}.png", expected_dimensions)
        ter = validate_png_image(root / "terrain" / f"z{row['z']}" / row["region"] / f"{row['x']}_{row['y']}.png", expected_dimensions)
        row["satellite_present"] = bool(sat["valid"])
        row["terrain_present"] = bool(ter["valid"])
        row["satellite_reason"] = sat["reason"]
        row["terrain_reason"] = ter["reason"]
        counts["coordinates"] += 1
        counts["satellite_valid"] += int(sat["valid"])
        counts["terrain_valid"] += int(ter["valid"])
        counts["complete_pairs"] += int(sat["valid"] and ter["valid"])
        counts["incomplete_pairs"] += int(not (sat["valid"] and ter["valid"]))
        counts["invalid_files"] += int(sat["present"] and not sat["valid"]) + int(ter["present"] and not ter["valid"])
        output.append(row)
    output.sort(key=lambda r: (r["region"], int(r["z"]), int(r["x"]), int(r["y"])))
    return output, counts


def build_inventory_report(rows: list[dict[str, Any]], counts: dict[str, Any], failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"schema_version": 1, "counts": dict(counts), "reusable_pairs": sum(int(r["satellite_present"] and r["terrain_present"]) for r in rows), "failures": sorted(failures or [], key=lambda x: (x.get("region", ""), x.get("x", 0), x.get("y", 0), x.get("style", "")))}
