"""Deterministic active-learning regional planning, inventory, and acquisition CLI.

Open-data source selection (NAIP ``naip-visualization`` imagery + USGS 3DEP
1/3 arc-second terrain) with a source-contract configuration file.  The default
mode is plan-only: geometry, catalogs, and source contracts are validated and
an exact asset/tile/range plan is computed with zero network access.

Authorization boundaries (all fail closed, all atomic):

- missing local catalog snapshot -> ``discovery_authorization_request.json``;
- complete catalogs but no execution authorization -> immutable
  ``execution_plan.json`` + ``execution_plan.sha256`` plus
  ``acquisition_authorization_request.json`` (null caps, requester-pays false);
- ``--execute-plan`` requires the plan path, its expected SHA-256, positive
  request/transfer/local/requester-pays-spend caps and ``--allow-requester-pays``.

No Mapbox acquisition path exists in this CLI.  The only admissible imagery
source is ``naip-visualization`` and the only admissible terrain source is
``usgs-3dep-13as``; unknown or paid sources are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from PIL import Image
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.active_learning.common import validate_run_name  # noqa: E402
from src.active_learning.water import (  # noqa: E402
    MAX_SELECTABLE_WATER_FRACTION,
    compute_satellite_water_fraction,
    evaluate_water_status,
    WATER_FILTER_STATUS_EXCESSIVE,
)
from src.data_pipeline.region_planning import (  # noqa: E402
    get_builtin_region_spec,
    parse_and_validate_region_spec,
)
from src.data_pipeline.tile_inventory import (  # noqa: E402
    build_inventory_report,
    scan_tile_inventory,
)
from src.data_pipeline.web_mercator import (  # noqa: E402
    tile_bounds_wgs84,
    tile_to_lat_lon_center,
)
from src.terrain.features import compute_terrain_sea_level_fraction  # noqa: E402

from scripts.ingest.naip_3dep_workflow import (  # noqa: E402
    MANIFEST_COLUMNS,
    MANIFEST_SCHEMA_VERSION,
    MAX_COORDINATES,
    MAX_RASTERS,
    MAX_WORKERS,
    PILOT_MAX_COORDINATES,
    PILOT_SELECTION_SEED,
    SUPPORTED_IMAGERY_SOURCE,
    SUPPORTED_STYLES,
    SUPPORTED_TERRAIN_SOURCE,
    NetworkCaps,
    atomic_write_json,
    atomic_write_text,
    authorization_payload_digest,
    build_acquisition_authorization_request,
    build_discovery_authorization_request,
    build_execution_plan,
    build_tile_plans,
    catalog_sha,
    canonical_json_bytes,
    discover_catalog_metadata,
    discover_raw_catalogs,
    execute_plan,
    file_sha256,
    identity_digest,
    load_3dep_catalog,
    load_naip_catalog,
    load_source_contract,
    plan_digest,
    repository_source_tree_digest,
    select_pilot_frame,
    validate_imagery_source,
    validate_terrain_source,
    write_manifest_csv,
)

# Kept for downstream tooling that reads the manifest column contract.
COLUMNS = list(MANIFEST_COLUMNS)
PLANNING_SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_spec(path: str | None) -> dict[str, Any]:
    if path is None:
        return get_builtin_region_spec()
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _repository_revision() -> str | None:
    import subprocess

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


def _check_plan_identity(
    run_dir: Path,
    *,
    spec_sha: str,
    contract_sha: str,
    zoom: int,
    budget: int,
    pilot_budget: int | None,
    geometry_digest: str,
    repository_source_tree_digest: str | None = None,
) -> None:
    """Refuse to rewrite an existing run whose planning inputs drifted."""
    manifest_path = run_dir / "region_manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open(encoding="utf-8") as stream:
        existing = json.load(stream)
    inputs = existing.get("inputs", {})
    checks = {
        "region_spec_sha256": (inputs.get("region_spec_sha256"), spec_sha),
        "source_contract_sha256": (inputs.get("source_contract_sha256"), contract_sha),
        "zoom": (existing.get("zoom"), zoom),
        "tile_budget_coordinates": (existing.get("tile_budget_coordinates"), budget),
        "pilot_budget": (existing.get("pilot_budget"), pilot_budget),
        "geometry_digest": (existing.get("geometry_digest"), geometry_digest),
        "repository_source_tree_digest": (
            inputs.get("repository_source_tree_digest")
            or existing.get("repository_source_tree_digest"),
            repository_source_tree_digest,
        ),
    }
    for name, (old, new) in checks.items():
        if old is not None and old != new:
            raise ValueError(f"existing run identity drift: {name}")


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Manifest rows
# ---------------------------------------------------------------------------


def _manifest_rows(
    *,
    planned: dict[str, Any],
    ordered_coords: list[tuple[int, int]],
    zoom: int,
    image_root: Path,
    contract: dict[str, Any],
    contract_sha: str,
    spec_sha: str,
    tile_plans: dict[str, dict[str, Any]] | None,
    tree_digest: str | None = None,
) -> list[dict[str, Any]]:
    preprocessing = contract["preprocessing"]
    imagery = contract["imagery"]
    terrain = contract["terrain"]
    processing_version = str(preprocessing["contract_id"])
    preprocessing_sha = identity_digest(preprocessing)
    geometry_digest = str(planned["geometry_digest"])
    admission_reasons = planned.get("coord_to_admission_reason", {})

    rows: list[dict[str, Any]] = []
    for x, y in ordered_coords:
        region = planned["coord_to_region"][(x, y)]
        lat, lon = tile_to_lat_lon_center(x, y, zoom)
        filename = f"{x}_{y}.png"
        tile_plan = (tile_plans or {}).get(f"{x}_{y}", {})

        satellite_asset_ids = tile_plan.get("satellite_asset_ids", [])
        terrain_asset_ids = tile_plan.get("terrain_asset_ids", [])

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
                "satellite_water_fraction": "",
                "terrain_sea_level_fraction": "",
                "effective_water_fraction": "",
                "water_filter_status": "unknown",
                "unusable_reason": "",
                "admission_reason": admission_reasons.get((x, y), ""),
                "land_fraction": f"{planned.get('coord_to_land_fraction', {}).get((x, y), 0.0):.12f}",
                "state": planned.get("coord_to_state", {}).get((x, y))
                or tile_plan.get("state")
                or "",
                "source_contract_sha256": contract_sha,
                "preprocessing_contract_sha256": preprocessing_sha,
                "repository_source_tree_digest": tree_digest
                or repository_source_tree_digest(),
                "boundary_geometry_sha256": geometry_digest,
                "grid_sha256": "",
                "acquisition_tile_identity_sha256": "",
                "satellite_provider": imagery["provider"],
                "satellite_collection": imagery["collection"],
                "satellite_asset_ids": _compact_json(satellite_asset_ids),
                "satellite_acquisition_year": (
                    tile_plan.get("satellite_acquisition_year") or ""
                ),
                "satellite_capture_date": tile_plan.get("satellite_capture_date") or "",
                "satellite_license": imagery["license"],
                "satellite_attribution": imagery["attribution"],
                "satellite_source_checksums": _compact_json(
                    tile_plan.get("satellite_source_checksums", {})
                ),
                "satellite_output_sha256": "",
                "satellite_valid_fraction": "",
                "terrain_provider": terrain["provider"],
                "terrain_collection": terrain["collection"],
                "terrain_asset_ids": _compact_json(terrain_asset_ids),
                "terrain_vertical_datum": (
                    tile_plan.get("terrain_vertical_datum") or ""
                ),
                "terrain_native_resolution": (
                    tile_plan.get("terrain_native_resolution") or ""
                ),
                "terrain_license": terrain["license"],
                "terrain_attribution": terrain["attribution"],
                "terrain_source_checksums": _compact_json(
                    tile_plan.get("terrain_source_checksums", {})
                ),
                "terrain_output_sha256": "",
                "terrain_valid_fraction": "",
                "mosaic_contributions": "",
                "processing_version": processing_version,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_run(
    *,
    run_name: str,
    spec: dict[str, Any] | None = None,
    spec_path: str | None = None,
    output_root: Path | str = "data/processed/active_learning",
    image_root: Path | str = "data/raw/sources/naip_3dep_v1/images",
    config_path: Path | str = "config/app_regions.json",
    source_contract_path: Path | str = "config/data_sources/naip_3dep_v1.json",
    imagery_source: str = SUPPORTED_IMAGERY_SOURCE,
    terrain_source: str = SUPPORTED_TERRAIN_SOURCE,
    zoom: int = 14,
    budget: int = MAX_COORDINATES,
    pilot_budget: int | None = None,
    max_runtime_minutes: int | None = None,
) -> dict[str, Any]:
    """Validate geometry, catalogs, and contracts; compute the exact plan.

    Never touches the network.  Missing catalogs produce a discovery
    authorization request; a complete plan without execution caps produces the
    immutable execution plan plus an acquisition authorization request.
    """
    run_name = validate_run_name(run_name)
    validate_imagery_source(imagery_source)
    validate_terrain_source(terrain_source)
    if zoom != 14:
        raise ValueError("active-learning acquisition requires zoom 14")
    if (
        not isinstance(budget, int)
        or isinstance(budget, bool)
        or not (1 <= budget <= MAX_COORDINATES)
    ):
        raise ValueError(
            f"budget must be between 1 and {MAX_COORDINATES} tile coordinates"
        )
    if budget * 2 > MAX_RASTERS:
        raise ValueError(f"budget {budget} exceeds the {MAX_RASTERS} raster cap")
    if pilot_budget is not None and (
        not isinstance(pilot_budget, int)
        or isinstance(pilot_budget, bool)
        or not (1 <= pilot_budget <= PILOT_MAX_COORDINATES)
    ):
        raise ValueError(
            f"pilot_budget must be between 1 and {PILOT_MAX_COORDINATES}, got {pilot_budget!r}"
        )
    if pilot_budget is not None and pilot_budget > budget:
        raise ValueError("pilot_budget cannot exceed budget")
    if max_runtime_minutes is not None and (
        not isinstance(max_runtime_minutes, int)
        or isinstance(max_runtime_minutes, bool)
        or max_runtime_minutes < 1
    ):
        raise ValueError("max_runtime_minutes must be a positive int or None")

    contract, contract_sha = load_source_contract(source_contract_path)
    region_spec = spec if spec is not None else _load_spec(spec_path)
    planned = parse_and_validate_region_spec(
        region_spec,
        app_regions_path=config_path,
        max_budget_coords=budget,
        zoom=zoom,
    )
    spec_sha = (
        file_sha256(Path(spec_path)) if spec_path else identity_digest(region_spec)
    )
    land_entries = [
        entry
        for entry in region_spec.get("regions", [])
        if isinstance(entry, dict)
        and entry.get("type") == "jurisdiction_land_intersection"
    ]
    if len(land_entries) > 1:
        raise ValueError("region spec must bind exactly one canonical land geometry")
    land_geometry_path: Path | None = None
    land_geometry_sha256: str | None = None
    if land_entries:
        land_geometry_path = Path(land_entries[0]["land_geometry_file"])
        land_geometry_sha256 = file_sha256(land_geometry_path)

    run_dir = Path(output_root) / run_name
    tree_digest = repository_source_tree_digest()
    _check_plan_identity(
        run_dir,
        spec_sha=spec_sha,
        contract_sha=contract_sha,
        zoom=zoom,
        budget=budget,
        pilot_budget=pilot_budget,
        geometry_digest=str(planned["geometry_digest"]),
        repository_source_tree_digest=tree_digest,
    )

    if pilot_budget is not None:
        ordered_coords, pilot_frame_sha = select_pilot_frame(
            planned["ordered_coords"],
            planned["coord_to_region"],
            pilot_budget,
        )
    else:
        ordered_coords = list(planned["ordered_coords"])
        pilot_frame_sha = None
    tile_planning = dict(planned)
    tile_planning["ordered_coords"] = ordered_coords
    tile_planning["coord_to_region"] = {
        coord: planned["coord_to_region"][coord] for coord in ordered_coords
    }
    coord_to_state = planned.get("coord_to_state", {})
    tile_planning["coord_to_state"] = {
        coord: coord_to_state.get(coord) for coord in ordered_coords
    }

    imagery_catalog_path = Path(contract["imagery"]["catalog_snapshot"])
    terrain_catalog_path = Path(contract["terrain"]["catalog_snapshot"])
    imagery_catalog = load_naip_catalog(
        imagery_catalog_path, contract["imagery"].get("catalog_sha256")
    )
    terrain_assets = load_3dep_catalog(
        terrain_catalog_path, contract["terrain"].get("catalog_sha256")
    )
    imagery_catalog_sha = catalog_sha(imagery_catalog_path)
    terrain_catalog_sha = catalog_sha(terrain_catalog_path)

    rows = _manifest_rows(
        planned=planned,
        ordered_coords=ordered_coords,
        zoom=zoom,
        image_root=Path(image_root),
        contract=contract,
        contract_sha=contract_sha,
        spec_sha=spec_sha,
        tile_plans=None,
        tree_digest=tree_digest,
    )
    rows, counts = scan_tile_inventory(rows, image_root=image_root)
    water_counts = _assess_water_inventory(rows)

    missing_catalogs: list[str] = []
    if imagery_catalog is None:
        missing_catalogs.append("imagery")
    if terrain_assets is None:
        missing_catalogs.append("terrain")

    tile_plans_result: dict[str, Any] | None = None
    execution_plan_path: Path | None = None
    plan_sha256: str | None = None
    authorization_path: Path | None = None

    if not missing_catalogs:
        tile_plans_result = build_tile_plans(
            planned=tile_planning,
            contract=contract,
            imagery_catalog=imagery_catalog,
            terrain_assets=terrain_assets,
            zoom=zoom,
        )
        rows = _manifest_rows(
            planned=planned,
            ordered_coords=ordered_coords,
            zoom=zoom,
            image_root=Path(image_root),
            contract=contract,
            contract_sha=contract_sha,
            spec_sha=spec_sha,
            tile_plans=tile_plans_result["tile_plans"],
            tree_digest=tree_digest,
        )
        rows, counts = scan_tile_inventory(rows, image_root=image_root)
        water_counts = _assess_water_inventory(rows)

        write_manifest_csv(run_dir / "tile_manifest.csv", rows)
        manifest_sha = file_sha256(run_dir / "tile_manifest.csv")
        execution_plan = build_execution_plan(
            run_name=run_name,
            zoom=zoom,
            contract=contract,
            contract_sha=contract_sha,
            contract_path=source_contract_path,
            spec_path=spec_path,
            spec_sha=spec_sha,
            geometry_digest=str(planned["geometry_digest"]),
            imagery_catalog_path=imagery_catalog_path,
            imagery_catalog_sha=imagery_catalog_sha,
            terrain_catalog_path=terrain_catalog_path,
            terrain_catalog_sha=terrain_catalog_sha,
            tile_plans_result=tile_plans_result,
            planned_rows=rows,
            tile_manifest_sha=manifest_sha,
            pilot_frame_sha=pilot_frame_sha,
            budget=budget,
            pilot_budget=pilot_budget,
            image_root=image_root,
            output_root=output_root,
            config_path=config_path,
            imagery_source=imagery_source,
            terrain_source=terrain_source,
            max_runtime_minutes=max_runtime_minutes,
            land_geometry_path=land_geometry_path,
            land_geometry_sha256=land_geometry_sha256,
            missing_satellite=sum(not row.get("satellite_present") for row in rows),
            missing_terrain=sum(not row.get("terrain_present") for row in rows),
            local_reuse_count=sum(
                bool(row.get("satellite_present")) and bool(row.get("terrain_present"))
                for row in rows
            ),
            repository_source_tree_digest=tree_digest,
        )
        plan_sha256 = plan_digest(execution_plan)
        execution_plan_path = run_dir / "execution_plan.json"
        # Write the plan byte-exactly as its canonical compact JSON: the digest
        # is the SHA-256 of the file itself, and load_and_verify_plan hashes the
        # file bytes, so the two must agree without reformatting.
        atomic_write_text(
            execution_plan_path,
            canonical_json_bytes(execution_plan).decode("utf-8"),
        )
        atomic_write_text(run_dir / "execution_plan.sha256", plan_sha256 + "\n")
        authorization = build_acquisition_authorization_request(
            run_name=run_name,
            plan=execution_plan,
            plan_sha=plan_sha256,
            contract=contract,
            repository_source_tree_digest=tree_digest,
        )
        authorization_path = run_dir / "acquisition_authorization_request.json"
        atomic_write_json(authorization_path, authorization)
    else:
        write_manifest_csv(run_dir / "tile_manifest.csv", rows)
        jurisdictions = region_spec.get("included_jurisdictions", [])
        min_x = min(x for x, _ in ordered_coords)
        max_x = max(x for x, _ in ordered_coords)
        min_y = min(y for _, y in ordered_coords)
        max_y = max(y for _, y in ordered_coords)
        min_lon, min_lat, _, _ = tile_bounds_wgs84(min_x, max_y, zoom)
        _, _, max_lon, max_lat = tile_bounds_wgs84(max_x, min_y, zoom)
        target_bbox = (min_lon, min_lat, max_lon, max_lat)
        authorization = build_discovery_authorization_request(
            run_name=run_name,
            contract=contract,
            contract_sha=contract_sha,
            region_spec_sha=spec_sha,
            geometry_digest=str(planned["geometry_digest"]),
            coordinate_count=len(ordered_coords),
            jurisdictions=jurisdictions,
            missing=missing_catalogs,
            target_bbox=target_bbox,
            resume_command="",
            repository_source_tree_digest=tree_digest,
        )
        req_auth = authorization["required_authorization"]
        authorization_path = run_dir / "discovery_authorization_request.json"
        # Stable payload digest (excludes authorization_digest/resume_command so
        # the artifact can carry its own digest without a circular self-hash).
        authorization_sha = authorization_payload_digest(authorization)

        resume_cmd = (
            "uv run python scripts/ingest/plan_active_learning_region.py "
            f"--run-name {run_name} "
            f"--discover-raw-catalogs "
            f"--discovery-request {authorization_path} "
            f"--expected-discovery-request-sha256 {authorization_sha} "
            f"--source-contract {source_contract_path} "
            f"--imagery-source {imagery_source} --terrain-source {terrain_source} "
            f"--image-root {image_root} --output-root {output_root} "
            f"--budget {budget}"
            + (f" --pilot-budget {pilot_budget}" if pilot_budget is not None else "")
            + (f" --region-spec {spec_path}" if spec_path else "")
            + (f" --region-source {config_path}" if config_path else "")
            + (
                f" --max-runtime-minutes {max_runtime_minutes}"
                if max_runtime_minutes is not None
                else ""
            )
            + " --allow-requester-pays "
            f"--max-source-requests {req_auth['max_requests']} "
            f"--max-transfer-bytes {req_auth['max_transfer_bytes']} "
            f"--max-local-bytes {req_auth['max_local_bytes']} "
            f"--max-requester-pays-usd {req_auth['max_spend_usd']}"
        )
        authorization["resume_command"] = resume_cmd
        atomic_write_json(authorization_path, authorization)

    # --- preflight and manifest -------------------------------------------
    missing_satellite = sum(not row.get("satellite_present") for row in rows)
    missing_terrain = sum(not row.get("terrain_present") for row in rows)
    preflight = {
        "schema_version": PLANNING_SCHEMA_VERSION,
        "run_name": run_name,
        "geometry_digest": planned["geometry_digest"],
        "budget_valid": planned["unique_coordinates_count"] <= budget,
        "unique_coordinates": planned["unique_coordinates_count"],
        "planned_coordinates": len(ordered_coords),
        "total_rasters": planned["total_rasters_count"],
        "tile_budget_coordinates": budget,
        "tile_budget_rasters": budget * 2,
        "pilot_budget": pilot_budget,
        "existing_satellite": counts["satellite_valid"],
        "existing_terrain": counts["terrain_valid"],
        "existing_complete_pairs": counts["complete_pairs"],
        "missing_satellite": missing_satellite,
        "missing_terrain": missing_terrain,
        "estimated_missing_storage_bytes": (missing_satellite + missing_terrain)
        * 262_144,
        "region_spec_sha256": spec_sha,
        "source_contract_sha256": contract_sha,
        "state": "needs_discovery" if missing_catalogs else "planned",
        "water_assessed": water_counts["assessed"],
        "water_excessive": water_counts["excessive"],
        "water_unknown": water_counts["unknown"],
        "water_threshold": MAX_SELECTABLE_WATER_FRACTION,
        "state_tile_counts": planned.get("state_tile_counts", {}),
        "excluded_water_coordinates": planned.get("excluded_water_count", 0),
        "admission_reason_counts": planned.get("admission_reason_counts", {}),
        "land_fraction_summary": planned.get("land_fraction_summary", {}),
        "geometry_source_hashes": planned.get("geometry_source_hashes", {}),
        "water_filter": water_counts,
        "storage_estimate_assumption": "256 KiB average compressed PNG per raster",
        "execution_plan_path": str(execution_plan_path)
        if execution_plan_path
        else None,
        "execution_plan_sha256": plan_sha256,
        "authorization_path": str(authorization_path) if authorization_path else None,
        "authorization_type": (
            "acquisition_authorization_request"
            if not missing_catalogs
            else "discovery_authorization_request"
        ),
    }
    atomic_write_json(run_dir / "acquisition_preflight.json", preflight)

    failures: list[dict[str, Any]] = []
    report = build_inventory_report(rows, counts, failures)
    manifest = {
        "schema_version": PLANNING_SCHEMA_VERSION,
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": _repository_revision(),
        "repository_source_tree_digest": tree_digest,
        "styles": list(SUPPORTED_STYLES),
        "imagery_source": imagery_source,
        "terrain_source": terrain_source,
        "tile_budget_coordinates": budget,
        "tile_budget_rasters": budget * 2,
        "pilot_budget": pilot_budget,
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
        "state_tile_counts": planned.get("state_tile_counts", {}),
        "excluded_water_coordinates": planned.get("excluded_water_count", 0),
        "admission_reason_counts": planned.get("admission_reason_counts", {}),
        "land_fraction_summary": planned.get("land_fraction_summary", {}),
        "geometry_source_hashes": planned.get("geometry_source_hashes", {}),
        "unique_coordinates": planned["unique_coordinates_count"],
        "planned_coordinates": len(ordered_coords),
        "total_rasters": planned["total_rasters_count"],
        "inventory_counts": counts,
        "water_assessed": water_counts["assessed"],
        "water_excessive": water_counts["excessive"],
        "water_unknown": water_counts["unknown"],
        "water_threshold": MAX_SELECTABLE_WATER_FRACTION,
        "water_filter": water_counts,
        "acquisition_requested": False,
        "failures": failures,
        "inputs": {
            "region_spec_sha256": spec_sha,
            "region_spec_path": spec_path,
            "source_contract_sha256": contract_sha,
            "source_contract_path": str(source_contract_path),
            "repository_source_tree_digest": tree_digest,
            "terrain_catalog_sha256": terrain_catalog_sha,
            "app_regions_path": str(config_path),
            "app_regions_sha256": file_sha256(Path(config_path)),
            "acquisition_preflight_path": str(run_dir / "acquisition_preflight.json"),
            "acquisition_preflight_sha256": file_sha256(
                run_dir / "acquisition_preflight.json"
            ),
            "model_inputs": {
                "active_model_scored": False,
                "identity_source": "scoring_manifest.json after manifest scoring",
            },
        },
        "sources": {
            "satellite": {
                "provider": contract["imagery"]["provider"],
                "collection": contract["imagery"]["collection"],
                "bucket": contract["imagery"]["bucket"],
                "requester_pays": bool(contract["imagery"].get("requester_pays")),
                "license": contract["imagery"]["license"],
                "attribution": contract["imagery"]["attribution"],
                "provider_url": contract["imagery"].get(
                    "provider_url", "https://registry.opendata.aws/naip/"
                ),
                "access_date": contract["imagery"].get("access_date", "2026-08-10"),
            },
            "terrain": {
                "provider": contract["terrain"]["provider"],
                "collection": contract["terrain"]["collection"],
                "bucket": contract["terrain"]["bucket"],
                "requester_pays": bool(contract["terrain"].get("requester_pays")),
                "license": contract["terrain"]["license"],
                "attribution": contract["terrain"]["attribution"],
                "provider_url": contract["terrain"].get(
                    "provider_url", "https://www.usgs.gov/3d-elevation-program"
                ),
                "access_date": contract["terrain"].get("access_date", "2026-08-10"),
            },
            "boundaries": {
                "provider": contract.get("boundaries", {}).get(
                    "provider", "us_census_bureau"
                ),
                "collection": contract.get("boundaries", {}).get(
                    "collection", "tigerweb_2025_state_county"
                ),
                "provider_url": contract.get("boundaries", {}).get(
                    "provider_url",
                    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0/query",
                ),
                "access_date": contract.get("boundaries", {}).get(
                    "access_date", "2026-08-10"
                ),
            },
            "rate_card": {
                "source": contract["rate_card"]["source"],
                "date": contract["rate_card"]["date"],
                "provider_url": contract["rate_card"].get(
                    "provider_url", "https://aws.amazon.com/s3/pricing/"
                ),
                "access_date": contract["rate_card"].get("access_date", "2026-08-10"),
            },
            "provider_urls": {
                "naip_registry": contract["imagery"].get(
                    "provider_url", "https://registry.opendata.aws/naip/"
                ),
                "usgs_3dep": contract["terrain"].get(
                    "provider_url", "https://www.usgs.gov/3d-elevation-program"
                ),
                "census_boundaries": contract.get("boundaries", {}).get(
                    "provider_url",
                    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0/query",
                ),
                "aws_s3_pricing": contract["rate_card"].get(
                    "provider_url", "https://aws.amazon.com/s3/pricing/"
                ),
            },
            "access_dates": {
                "naip_registry": contract["imagery"].get("access_date", "2026-08-10"),
                "usgs_3dep": contract["terrain"].get("access_date", "2026-08-10"),
                "census_boundaries": contract.get("boundaries", {}).get(
                    "access_date", "2026-08-10"
                ),
                "aws_s3_pricing": contract["rate_card"].get(
                    "access_date", "2026-08-10"
                ),
            },
            "mapbox": {
                "used": False,
                "note": "Mapbox acquisition is not part of this pipeline",
            },
        },
        "reuse_validation": (
            "local PNG decode/hash validation plus per-tile run records; "
            "a file is only reused when its run record carries this exact run id "
            "and execution-plan SHA-256 and every recorded output SHA-256 matches"
        ),
        "acquisition": {
            "mode": "execute_plan_only",
            "note": "acquisition requires --execute-plan, --expected-plan-sha256, positive caps and --allow-requester-pays",
        },
        "execution_plan_sha256": plan_sha256,
        "stage_state": "needs_discovery" if missing_catalogs else "planned",
        "ready_for_selection": False,
    }
    atomic_write_json(run_dir / "region_manifest.json", manifest)
    atomic_write_json(run_dir / "inventory_report.json", report)
    atomic_write_json(
        run_dir / "source_contract.json",
        json.loads(canonical_json_bytes(contract).decode("utf-8")),
    )
    atomic_write_json(
        run_dir / "preprocessing_contract.json",
        json.loads(canonical_json_bytes(contract["preprocessing"]).decode("utf-8")),
    )
    if pilot_frame_sha is not None:
        atomic_write_json(
            run_dir / "pilot_frame.json",
            {
                "seed": PILOT_SELECTION_SEED,
                "budget": pilot_budget,
                "coordinates": [list(coord) for coord in ordered_coords],
                "pilot_frame_sha256": pilot_frame_sha,
            },
        )
    write_manifest_csv(run_dir / "tile_manifest.csv", rows)

    return {
        "run_dir": str(run_dir),
        "plan_state": "needs_discovery" if missing_catalogs else "planned",
        "region_manifest": manifest,
        "inventory_report": report,
        "tile_manifest_rows": len(rows),
        "execution_plan_path": str(execution_plan_path)
        if execution_plan_path
        else None,
        "execution_plan_sha256": plan_sha256,
        "authorization_path": str(authorization_path) if authorization_path else None,
        "missing_catalogs": missing_catalogs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None, help="Plan-mode run slug")
    parser.add_argument(
        "--region-spec",
        help="Versioned JSON region specification; defaults to the built-in spec",
    )
    parser.add_argument("--imagery-source", default=SUPPORTED_IMAGERY_SOURCE)
    parser.add_argument("--terrain-source", default=SUPPORTED_TERRAIN_SOURCE)
    parser.add_argument(
        "--source-contract",
        default="config/data_sources/naip_3dep_v1.json",
        help="Source-contract configuration JSON",
    )
    parser.add_argument(
        "--image-root",
        default="data/raw/sources/naip_3dep_v1/images",
        help="Source-versioned output images root (satellite/ and terrain/ live beneath it)",
    )
    parser.add_argument("--output-root", default="data/processed/active_learning")
    parser.add_argument("--region-source", default="config/app_regions.json")
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument("--budget", type=int, default=MAX_COORDINATES)
    parser.add_argument(
        "--pilot-budget",
        type=int,
        default=None,
        help=f"Deterministic capped pilot frame of at most {PILOT_MAX_COORDINATES} coordinates",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=int,
        default=None,
        help="Optional execution deadline; checkpoint and pause when exceeded",
    )
    parser.add_argument(
        "--execute-plan",
        default=None,
        help="Execute an immutable execution plan produced by plan mode",
    )
    parser.add_argument("--expected-plan-sha256", default=None)
    parser.add_argument("--allow-requester-pays", action="store_true")
    parser.add_argument("--max-source-requests", type=int, default=None)
    parser.add_argument("--max-transfer-bytes", type=int, default=None)
    parser.add_argument("--max-local-bytes", type=int, default=None)
    parser.add_argument("--max-requester-pays-usd", default=None)
    parser.add_argument(
        "--discover-raw-catalogs",
        action="store_true",
        help="Boundary 1: Execute raw catalog discovery over metered transport and emit metadata authorization request",
    )
    parser.add_argument("--discovery-request", default=None)
    parser.add_argument("--expected-discovery-request-sha256", default=None)
    parser.add_argument(
        "--discover-catalog-metadata",
        action="store_true",
        help="Boundary 2: Execute catalog metadata observations over metered transport and build strict normalized catalogs",
    )
    parser.add_argument("--metadata-request", default=None)
    parser.add_argument("--expected-metadata-request-sha256", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=f"Bounded concurrent processing workers (1..{MAX_WORKERS})",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    if args.discover_raw_catalogs:
        if not args.run_name:
            raise SystemExit("--run-name is required for raw discovery")
        if not args.discovery_request or not args.expected_discovery_request_sha256:
            raise SystemExit(
                "--discover-raw-catalogs requires --discovery-request and --expected-discovery-request-sha256"
            )
        for name in ("max_source_requests", "max_transfer_bytes", "max_local_bytes"):
            value = getattr(args, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SystemExit(
                    f"--{name.replace('_', '-')} must be a positive integer for discovery"
                )
        if not args.max_requester_pays_usd:
            raise SystemExit(
                "--max-requester-pays-usd must be a positive decimal for discovery"
            )
        try:
            spend = Decimal(str(args.max_requester_pays_usd))
        except Exception as exc:
            raise SystemExit(f"invalid --max-requester-pays-usd: {exc}")
        if not spend.is_finite() or spend <= 0:
            raise SystemExit("--max-requester-pays-usd must be positive for discovery")

        caps = NetworkCaps(
            max_requests=args.max_source_requests,
            max_transfer_bytes=args.max_transfer_bytes,
            max_local_bytes=args.max_local_bytes,
            max_requester_pays_usd=spend,
            allow_requester_pays=args.allow_requester_pays,
        )
        contract, _ = load_source_contract(args.source_contract)
        missing = []
        if load_naip_catalog(contract["imagery"]["catalog_snapshot"]) is None:
            missing.append("imagery")
        if load_3dep_catalog(contract["terrain"]["catalog_snapshot"]) is None:
            missing.append("terrain")

        run_dir = Path(args.output_root) / args.run_name
        res = discover_raw_catalogs(
            discovery_request_path=args.discovery_request,
            expected_discovery_request_sha256=args.expected_discovery_request_sha256,
            contract_path=args.source_contract,
            missing=missing,
            caps=caps,
            allow_requester_pays=args.allow_requester_pays,
            run_dir=run_dir,
        )
        print(json.dumps(res, sort_keys=True))
        return

    if args.discover_catalog_metadata:
        if not args.run_name:
            raise SystemExit("--run-name is required for metadata discovery")
        if not args.metadata_request or not args.expected_metadata_request_sha256:
            raise SystemExit(
                "--discover-catalog-metadata requires --metadata-request and --expected-metadata-request-sha256"
            )
        for name in ("max_source_requests", "max_transfer_bytes", "max_local_bytes"):
            value = getattr(args, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SystemExit(
                    f"--{name.replace('_', '-')} must be a positive integer for metadata discovery"
                )
        if not args.max_requester_pays_usd:
            raise SystemExit(
                "--max-requester-pays-usd must be a positive decimal for metadata discovery"
            )
        try:
            spend = Decimal(str(args.max_requester_pays_usd))
        except Exception as exc:
            raise SystemExit(f"invalid --max-requester-pays-usd: {exc}")
        if not spend.is_finite() or spend <= 0:
            raise SystemExit(
                "--max-requester-pays-usd must be positive for metadata discovery"
            )

        caps = NetworkCaps(
            max_requests=args.max_source_requests,
            max_transfer_bytes=args.max_transfer_bytes,
            max_local_bytes=args.max_local_bytes,
            max_requester_pays_usd=spend,
            allow_requester_pays=args.allow_requester_pays,
        )
        run_dir = Path(args.output_root) / args.run_name
        meta_result = discover_catalog_metadata(
            metadata_request_path=args.metadata_request,
            expected_metadata_request_sha256=args.expected_metadata_request_sha256,
            contract_path=args.source_contract,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=args.allow_requester_pays,
        )
        # Resumes plan_run after metadata discovery using the resolved source
        # contract copy (tracked config contract is never mutated).
        result = plan_run(
            run_name=args.run_name,
            spec_path=args.region_spec,
            output_root=args.output_root,
            image_root=args.image_root,
            config_path=args.region_source,
            source_contract_path=meta_result["resolved_contract_path"],
            imagery_source=args.imagery_source,
            terrain_source=args.terrain_source,
            zoom=args.zoom,
            budget=args.budget,
            pilot_budget=args.pilot_budget,
            max_runtime_minutes=args.max_runtime_minutes,
        )
        print(
            json.dumps(
                {
                    "run_dir": result["run_dir"],
                    "plan_state": result["plan_state"],
                    "tile_manifest_rows": result["tile_manifest_rows"],
                    "inventory": result["inventory_report"]["counts"],
                    "execution_plan_sha256": result["execution_plan_sha256"],
                    "authorization_path": result["authorization_path"],
                    "missing_catalogs": result["missing_catalogs"],
                    "imagery_catalog_sha256": meta_result["imagery_catalog_sha256"],
                    "terrain_catalog_sha256": meta_result["terrain_catalog_sha256"],
                    "resolved_contract_path": meta_result["resolved_contract_path"],
                    "resolved_contract_sha256": meta_result["resolved_contract_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if args.execute_plan is not None:
        if args.run_name is not None:
            raise SystemExit("--run-name must not be combined with --execute-plan")
        if not args.expected_plan_sha256:
            raise SystemExit(
                "--execute-plan requires --expected-plan-sha256 (the plan digest)"
            )
        for name in (
            "max_source_requests",
            "max_transfer_bytes",
            "max_local_bytes",
        ):
            value = getattr(args, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SystemExit(
                    f"--{name.replace('_', '-')} must be a positive integer for execution"
                )
        if not args.max_requester_pays_usd:
            raise SystemExit(
                "--max-requester-pays-usd must be a positive decimal for execution"
            )
        try:
            spend = Decimal(str(args.max_requester_pays_usd))
        except Exception as exc:
            raise SystemExit(f"invalid --max-requester-pays-usd: {exc}")
        if not spend.is_finite() or spend <= 0:
            raise SystemExit("--max-requester-pays-usd must be positive for execution")
        if not args.allow_requester_pays:
            raise SystemExit(
                "--execute-plan requires --allow-requester-pays because the NAIP "
                "bucket is requester-pays; zero spend is not an execution cap"
            )
        if not (1 <= args.workers <= MAX_WORKERS):
            raise SystemExit(f"--workers must be between 1 and {MAX_WORKERS}")

        result = execute_plan(
            args.execute_plan,
            expected_plan_sha256=args.expected_plan_sha256,
            allow_requester_pays=args.allow_requester_pays,
            max_source_requests=args.max_source_requests,
            max_transfer_bytes=args.max_transfer_bytes,
            max_local_bytes=args.max_local_bytes,
            max_requester_pays_usd=spend,
            workers=args.workers,
            max_runtime_minutes=args.max_runtime_minutes,
        )
        print(
            json.dumps(
                {
                    "run_dir": result["run_dir"],
                    "plan_sha256": result["plan_sha256"],
                    "state": result["state"],
                    "completed_coordinates": result["completed_coordinates"],
                    "failed_coordinates": result["failed_coordinates"],
                    "pending_coordinates": result["pending_coordinates"],
                    "counters": result["counters"],
                },
                sort_keys=True,
            )
        )
        return

    if not args.run_name:
        raise SystemExit("--run-name is required in plan mode")

    result = plan_run(
        run_name=args.run_name,
        spec_path=args.region_spec,
        output_root=args.output_root,
        image_root=args.image_root,
        config_path=args.region_source,
        source_contract_path=args.source_contract,
        imagery_source=args.imagery_source,
        terrain_source=args.terrain_source,
        zoom=args.zoom,
        budget=args.budget,
        pilot_budget=args.pilot_budget,
        max_runtime_minutes=args.max_runtime_minutes,
    )
    print(
        json.dumps(
            {
                "run_dir": result["run_dir"],
                "plan_state": result["plan_state"],
                "tile_manifest_rows": result["tile_manifest_rows"],
                "inventory": result["inventory_report"]["counts"],
                "execution_plan_sha256": result["execution_plan_sha256"],
                "authorization_path": result["authorization_path"],
                "missing_catalogs": result["missing_catalogs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
