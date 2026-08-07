"""Deterministic active-learning regional planning, inventory, and acquisition CLI."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import threading
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.mapbox import MapboxTileSource, tile_to_lat_lon_center  # noqa: E402
from src.data_pipeline.region_planning import (  # noqa: E402
    get_builtin_region_spec,
    parse_and_validate_region_spec,
)
from src.data_pipeline.tile_inventory import (  # noqa: E402
    build_inventory_report,
    scan_s3_inventory,
    scan_tile_inventory,
)
from src.active_learning.common import validate_run_name  # noqa: E402
from src.active_learning.water import (  # noqa: E402
    MAX_SELECTABLE_WATER_FRACTION,
    compute_satellite_water_fraction,
    evaluate_water_status,
    WATER_FILTER_STATUS_EXCESSIVE,
)
from src.terrain.features import compute_terrain_sea_level_fraction  # noqa: E402


PLANNING_SCHEMA_VERSION = 3


COLUMNS = [
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
    "satellite_water_fraction",
    "terrain_sea_level_fraction",
    "effective_water_fraction",
    "water_filter_status",
    "unusable_reason",
]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {column: row.get(column, "") for column in COLUMNS} for row in rows
    )
    _atomic_write_text(path, output.getvalue())


_QUIET_LOG_LOCK = threading.Lock()
_quiet_log_users = 0
_quiet_log_previous = logging.NOTSET


@contextmanager
def _quiet_mapbox_logging():
    """Prevent source logs from exposing request URLs or credential fragments."""
    global _quiet_log_previous, _quiet_log_users
    with _QUIET_LOG_LOCK:
        if _quiet_log_users == 0:
            _quiet_log_previous = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
        _quiet_log_users += 1
    try:
        yield
    finally:
        with _QUIET_LOG_LOCK:
            _quiet_log_users -= 1
            if _quiet_log_users == 0:
                logging.disable(_quiet_log_previous)


def _load_spec(path: str | None) -> dict[str, Any]:
    if path is None:
        return get_builtin_region_spec()
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resume_identity(
    run_dir: Path,
    *,
    planned: dict[str, Any],
    region_spec: dict[str, Any],
    spec_path: str | None,
    config_path: Path | str,
    zoom: int,
    budget: int,
) -> tuple[str, str | None]:
    manifest_path = run_dir / "region_manifest.json"
    preflight_path = run_dir / "acquisition_preflight.json"
    if not manifest_path.exists() and not preflight_path.exists():
        return _identity_digest(region_spec), None
    existing: dict[str, Any] = {}
    for path in (manifest_path, preflight_path):
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                existing.update(json.load(stream))
    spec_digest = (
        _file_digest(Path(spec_path)) if spec_path else _identity_digest(region_spec)
    )
    config_digest = _file_digest(Path(config_path))
    expected = {
        "geometry_digest": planned["geometry_digest"],
        "zoom": zoom,
        "tile_budget_coordinates": budget,
        "region_spec_sha256": spec_digest,
        "app_regions_sha256": config_digest,
    }
    old_spec = existing.get("region_spec")
    old_inputs = existing.get("inputs", {})
    old_spec_digest = old_inputs.get("region_spec_sha256")
    if old_spec_digest is None and old_spec is not None:
        old_spec_digest = _identity_digest(old_spec)
    old_config_digest = old_inputs.get("app_regions_sha256")
    checks = {
        "geometry_digest": existing.get("geometry_digest"),
        "zoom": existing.get("zoom"),
        "tile_budget_coordinates": existing.get("tile_budget_coordinates"),
        "region_spec_sha256": old_spec_digest,
        "app_regions_sha256": old_config_digest,
    }
    for key, old in checks.items():
        if old is not None and old != expected[key]:
            raise ValueError(f"existing run identity drift: {key}")
    created = existing.get("created_at_utc")
    return expected["region_spec_sha256"], created


def _repository_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _manifest_rows(
    planned: dict[str, Any], zoom: int, image_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for x, y in planned["ordered_coords"]:
        region = planned["coord_to_region"][(x, y)]
        lat, lon = tile_to_lat_lon_center(x, y, zoom)
        filename = f"{x}_{y}.png"
        rows.append(
            {
                "region": region,
                "z": zoom,
                "x": x,
                "y": y,
                "lat": f"{lat:.9f}",
                "lon": f"{lon:.9f}",
                "image_path": str(
                    Path(image_root.name) / "satellite" / f"z{zoom}" / region / filename
                ),
                "satellite_path": str(
                    image_root / "satellite" / f"z{zoom}" / region / filename
                ),
                "terrain_path": str(
                    image_root / "terrain" / f"z{zoom}" / region / filename
                ),
                "satellite_present": False,
                "terrain_present": False,
                "satellite_s3_present": False,
                "terrain_s3_present": False,
                "satellite_s3_uri": "",
                "terrain_s3_uri": "",
                "terrain_sea_level_fraction": "",
                "effective_water_fraction": "",
                "water_filter_status": "unknown",
                "unusable_reason": "",
            }
        )
    return rows


def acquire_missing(
    rows: list[dict[str, Any]],
    *,
    image_root: Path,
    zoom: int,
    failures: list[dict[str, Any]],
    max_workers: int = 8,
) -> None:
    """Acquire missing pair members with bounded concurrency and source-level retries."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    for style, style_id, path_key in (
        ("satellite", "mapbox.satellite", "satellite_path"),
        ("terrain", "mapbox.terrain-rgb", "terrain_path"),
    ):
        pending_rows = [
            row
            for row in rows
            if not row[f"{style}_present"] and not row.get(f"{style}_s3_present")
        ]
        worker_state = threading.local()

        def fetch(row: dict[str, Any]) -> None:
            source = getattr(worker_state, "source", None)
            if source is None:
                with _quiet_mapbox_logging():
                    source = MapboxTileSource(
                        cache_dir=image_root / ".mapbox_cache" / style,
                        style_id=style_id,
                        rate_limit=MapboxTileSource.DEFAULT_RATE_LIMIT / max_workers,
                    )
                worker_state.source = source
            path = Path(row[path_key])
            path.parent.mkdir(parents=True, exist_ok=True)
            with _quiet_mapbox_logging():
                tile = source.get_tile(int(row["x"]), int(row["y"]), zoom)
            Image.fromarray(tile.image).save(path, format="PNG", optimize=True)
            row[f"{style}_present"] = True

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"active-learning-{style}"
        ) as executor:
            iterator = iter(pending_rows)
            pending: dict[Future[None], dict[str, Any]] = {}
            for row in iterator:
                pending[executor.submit(fetch, row)] = row
                if len(pending) >= max_workers * 2:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        failed_row = pending.pop(future)
                        try:
                            future.result()
                        except Exception as exc:
                            failures.append(
                                {
                                    "region": failed_row["region"],
                                    "z": int(failed_row["z"]),
                                    "x": int(failed_row["x"]),
                                    "y": int(failed_row["y"]),
                                    "style": style,
                                    "reason": type(exc).__name__,
                                }
                            )
            for future, failed_row in pending.items():
                try:
                    future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "region": failed_row["region"],
                            "z": int(failed_row["z"]),
                            "x": int(failed_row["x"]),
                            "y": int(failed_row["y"]),
                            "style": style,
                            "reason": type(exc).__name__,
                        }
                    )


def _inventory(
    rows: list[dict[str, Any]],
    *,
    image_root: Path | str,
    s3_bucket: str | None,
    s3_prefix_root: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventoried, counts = scan_tile_inventory(rows, image_root=image_root)
    if s3_bucket:
        inventoried, counts = scan_s3_inventory(
            inventoried,
            bucket=s3_bucket,
            prefix_root=s3_prefix_root,
        )
    return inventoried, counts


def _assess_water_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assessed = 0
    excessive = 0
    unknown = 0
    for row in rows:
        satellite_path_text = row.get("satellite_path")
        satellite_path = Path(satellite_path_text) if satellite_path_text else None
        satellite_fraction = None
        if row.get("satellite_present") and satellite_path and satellite_path.is_file():
            with Image.open(satellite_path) as satellite_image:
                satellite_fraction = compute_satellite_water_fraction(satellite_image)

        terrain_path_text = row.get("terrain_path")
        terrain_path = Path(terrain_path_text) if terrain_path_text else None
        terrain_fraction = None
        if row.get("terrain_present") and terrain_path and terrain_path.is_file():
            terrain_fraction = compute_terrain_sea_level_fraction(terrain_path)

        effective, status, reason = evaluate_water_status(
            satellite_fraction,
            terrain_fraction,
        )
        row["satellite_water_fraction"] = (
            round(float(satellite_fraction), 6)
            if satellite_fraction is not None
            else ""
        )
        row["terrain_sea_level_fraction"] = (
            round(float(terrain_fraction), 6) if terrain_fraction is not None else ""
        )
        row["effective_water_fraction"] = (
            round(float(effective), 6) if effective is not None else ""
        )
        row["water_filter_status"] = status
        row["unusable_reason"] = reason or ""

        if effective is None:
            unknown += 1
        else:
            assessed += 1
            if status == WATER_FILTER_STATUS_EXCESSIVE:
                excessive += 1

    return {
        "assessed": assessed,
        "excessive": excessive,
        "eligible": assessed - excessive,
        "unknown": unknown,
        "threshold": MAX_SELECTABLE_WATER_FRACTION,
    }


def plan_run(
    *,
    run_name: str,
    spec: dict[str, Any] | None = None,
    spec_path: str | None = None,
    output_root: Path | str = "data/processed/active_learning",
    image_root: Path | str = "data/raw/images",
    config_path: Path | str = "config/app_regions.json",
    zoom: int = 14,
    acquire: bool = False,
    budget: int = 370_000,
    s3_bucket: str | None = None,
    s3_prefix_root: str = "raw/images",
    workers: int = 8,
) -> dict[str, Any]:
    run_name = validate_run_name(run_name)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if zoom != 14:
        raise ValueError("active-learning acquisition requires zoom 14")
    if budget < 1 or budget > 370_000:
        raise ValueError("budget must be between 1 and 370000 tile coordinates")
    region_spec = spec if spec is not None else _load_spec(spec_path)
    planned = parse_and_validate_region_spec(
        region_spec,
        app_regions_path=config_path,
        max_budget_coords=budget,
        zoom=zoom,
    )
    run_dir = Path(output_root) / run_name
    spec_digest, created_at = _resume_identity(
        run_dir,
        planned=planned,
        region_spec=region_spec,
        spec_path=spec_path,
        config_path=config_path,
        zoom=zoom,
        budget=budget,
    )
    config_digest = _file_digest(Path(config_path))
    rows = _manifest_rows(planned, zoom, Path(image_root))
    rows, counts = _inventory(
        rows,
        image_root=image_root,
        s3_bucket=s3_bucket,
        s3_prefix_root=s3_prefix_root,
    )
    water_counts = _assess_water_inventory(rows)
    preflight = {
        "schema_version": PLANNING_SCHEMA_VERSION,
        "run_name": run_name,
        "geometry_digest": planned["geometry_digest"],
        "budget_valid": planned["unique_coordinates_count"] <= budget,
        "unique_coordinates": planned["unique_coordinates_count"],
        "total_rasters": planned["total_rasters_count"],
        "tile_budget_coordinates": budget,
        "existing_satellite": counts["satellite_valid"],
        "existing_terrain": counts["terrain_valid"],
        "existing_complete_pairs": counts["complete_pairs"],
        "missing_satellite": sum(
            not row.get("satellite_present") and not row.get("satellite_s3_present")
            for row in rows
        ),
        "missing_terrain": sum(
            not row.get("terrain_present") and not row.get("terrain_s3_present")
            for row in rows
        ),
        "estimated_missing_storage_bytes": (
            sum(
                not row.get("satellite_present") and not row.get("satellite_s3_present")
                for row in rows
            )
            + sum(
                not row.get("terrain_present") and not row.get("terrain_s3_present")
                for row in rows
            )
        )
        * 262_144,
        "region_spec_sha256": spec_digest,
        "app_regions_sha256": config_digest,
        "state": "planned",
        "water_assessed": water_counts["assessed"],
        "water_excessive": water_counts["excessive"],
        "water_unknown": water_counts["unknown"],
        "water_threshold": MAX_SELECTABLE_WATER_FRACTION,
        "water_filter": water_counts,
        "storage_estimate_assumption": "256 KiB average compressed PNG per raster",
    }
    _atomic_csv(run_dir / "tile_manifest.csv", rows)
    _atomic_json(run_dir / "acquisition_preflight.json", preflight)
    failures: list[dict[str, Any]] = []
    missing_any = any(
        not row.get(f"{style}_present") and not row.get(f"{style}_s3_present")
        for style in ("satellite", "terrain")
        for row in rows
    )
    if acquire:
        if missing_any:
            if not os.environ.get("MAPBOX_ACCESS_TOKEN"):
                raise RuntimeError(
                    "MAPBOX_ACCESS_TOKEN is required for acquisition; dry-run artifacts were written"
                )
            acquire_missing(
                rows,
                image_root=Path(image_root),
                zoom=zoom,
                failures=failures,
                max_workers=workers,
            )
            rows, counts = _inventory(
                rows,
                image_root=image_root,
                s3_bucket=s3_bucket,
                s3_prefix_root=s3_prefix_root,
            )
            water_counts = _assess_water_inventory(rows)
        preflight.update(
            {
                "existing_satellite": counts["satellite_valid"],
                "existing_terrain": counts["terrain_valid"],
                "existing_complete_pairs": counts["complete_pairs"],
                "missing_satellite": sum(
                    not row.get("satellite_present")
                    and not row.get("satellite_s3_present")
                    for row in rows
                ),
                "missing_terrain": sum(
                    not row.get("terrain_present") and not row.get("terrain_s3_present")
                    for row in rows
                ),
                "estimated_missing_storage_bytes": (
                    sum(
                        not row.get("satellite_present")
                        and not row.get("satellite_s3_present")
                        for row in rows
                    )
                    + sum(
                        not row.get("terrain_present")
                        and not row.get("terrain_s3_present")
                        for row in rows
                    )
                )
                * 262_144,
                "state": "acquired",
                "water_assessed": water_counts["assessed"],
                "water_excessive": water_counts["excessive"],
                "water_unknown": water_counts["unknown"],
                "water_threshold": MAX_SELECTABLE_WATER_FRACTION,
                "water_filter": water_counts,
            }
        )
        _atomic_json(run_dir / "acquisition_preflight.json", preflight)
    failures.sort(
        key=lambda item: (
            item.get("region", ""),
            item.get("z", 0),
            item.get("x", 0),
            item.get("y", 0),
            item.get("style", ""),
        )
    )
    report = build_inventory_report(rows, counts, failures)
    manifest = {
        "schema_version": PLANNING_SCHEMA_VERSION,
        "run_name": run_name,
        "created_at_utc": created_at or datetime.now(timezone.utc).isoformat(),
        "repository_revision": _repository_revision(),
        "seeds": {},
        "zoom": zoom,
        "styles": ["satellite", "terrain"],
        "tile_budget_coordinates": budget,
        "tile_budget_rasters": budget * 2,
        "geometry_digest": planned["geometry_digest"],
        "geography": {
            "source": region_spec.get(
                "geographic_source", "versioned region specification"
            ),
            "included_jurisdictions": region_spec.get("included_jurisdictions", []),
            "excluded_jurisdictions": region_spec.get("excluded_jurisdictions", []),
            "known_non_target_coverage": region_spec.get(
                "known_non_target_coverage", []
            ),
            "limitations": region_spec.get("limitations", []),
        },
        "region_spec": region_spec,
        "region_tile_counts": planned["region_tile_counts"],
        "unique_coordinates": planned["unique_coordinates_count"],
        "total_rasters": planned["total_rasters_count"],
        "inventory_counts": counts,
        "water_assessed": water_counts["assessed"],
        "water_excessive": water_counts["excessive"],
        "water_unknown": water_counts["unknown"],
        "water_threshold": MAX_SELECTABLE_WATER_FRACTION,
        "water_filter": water_counts,
        "acquisition_requested": acquire,
        "failures": failures,
        "inputs": {
            "region_spec_sha256": spec_digest,
            "app_regions_path": str(config_path),
            "app_regions_sha256": config_digest,
            "acquisition_preflight_path": str(run_dir / "acquisition_preflight.json"),
            "acquisition_preflight_sha256": _file_digest(
                run_dir / "acquisition_preflight.json"
            ),
            "model_inputs": {
                "active_model_scored": False,
                "identity_source": "scoring_manifest.json after manifest scoring",
            },
        },
        "sources": {
            "satellite": "Mapbox satellite",
            "terrain": "Mapbox Terrain-RGB",
            "s3_bucket": s3_bucket,
            "s3_prefix_root": s3_prefix_root if s3_bucket else None,
            "provider_urls": {
                "satellite": "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png",
                "terrain": "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}@2x.png",
            },
        },
        "reuse_validation": "local PNG decode/hash validation; canonical S3 listing is reusable for acquisition only",
        "acquisition": {
            "workers": workers,
            "transient_retry_policy": "MapboxTileSource: 3 retries, exponential backoff for 429/5xx",
            "failure_ordering": "region/z/x/y/style after inventory report canonicalization",
        },
        "stage_state": "acquired" if acquire else "planned",
        "ready_for_selection": not failures
        and counts["reusable_pairs"] == counts["coordinates"],
    }
    _atomic_json(run_dir / "region_manifest.json", manifest)
    _atomic_csv(run_dir / "tile_manifest.csv", rows)
    _atomic_json(run_dir / "inventory_report.json", report)
    return {
        "run_dir": str(run_dir),
        "region_manifest": manifest,
        "inventory_report": report,
        "tile_manifest_rows": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--region-spec",
        help="Versioned JSON region specification; defaults to built-in inland expansion",
    )
    parser.add_argument("--output-root", default="data/processed/active_learning")
    parser.add_argument("--image-root", default="data/raw/images")
    parser.add_argument("--config", default="config/app_regions.json")
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument("--budget", type=int, default=370_000)
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="Acquire only pair members absent locally and in S3 inventory",
    )
    parser.add_argument(
        "--s3-bucket",
        help="Inventory and reuse canonical non-empty objects in this S3 bucket",
    )
    parser.add_argument("--s3-prefix-root", default="raw/images")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Bounded concurrent acquisition workers",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    result = plan_run(
        run_name=args.run_name,
        spec_path=args.region_spec,
        output_root=args.output_root,
        image_root=args.image_root,
        config_path=args.config,
        zoom=args.zoom,
        acquire=args.acquire,
        budget=args.budget,
        s3_bucket=args.s3_bucket,
        s3_prefix_root=args.s3_prefix_root,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "run_dir": result["run_dir"],
                "tile_manifest_rows": result["tile_manifest_rows"],
                "inventory": result["inventory_report"]["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
