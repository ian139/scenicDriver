"""USGS 3DEP elevation source adapter for Scenic Drive.

Fail-closed adapter over hash-pinned local exports of the official 3DEP ImageServer catalog.
Consumes local catalog JSON exports, validates 1/3-arc-second records, and emits SourceAsset
instances or pre-discovery authorization requests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any
from src.data_pipeline.source_contracts import SourceAsset


# Exceptions
class USGS3DEPError(Exception):
    """Base exception for USGS 3DEP adapter errors."""


class CatalogNotFoundError(USGS3DEPError, FileNotFoundError):
    """Raised when the local catalog export file is missing."""


class CatalogValidationError(USGS3DEPError, ValueError):
    """Raised when catalog hash-pinning or format verification fails."""


class RecordValidationError(USGS3DEPError, ValueError):
    """Raised when a 3DEP catalog record fails strict validation."""


class WeakIdentityError(RecordValidationError):
    """Raised when asset identity is dynamic, mutable, or weak."""


class ResolutionValidationError(RecordValidationError):
    """Raised when native resolution does not match 1/3 arc-second magnitude."""


class MetadataValidationError(RecordValidationError):
    """Raised when metadata fields (CRS, vertical datum, nodata, units) are invalid."""


# Constants
TARGET_RESOLUTION_DEG = 1.0 / (3.0 * 3600.0)  # ~9.259259e-05 degrees
RESOLUTION_TOLERANCE_DEG = 1e-7

VALID_HORIZONTAL_CRS = {"EPSG:4269", "EPSG:4326", "NAD83", "WGS84"}
VALID_VERTICAL_DATUMS = {"NAVD88", "EPSG:5703", "NAVD 88", "NAVD88_HEIGHT"}
VALID_ELEVATION_UNITS = {"m", "meter", "meters"}

# Canonical S3 key pattern: prd-tnm/StagedProducts/Elevation/13/TIFF/current/.../USGS_13_*.tif
CANONICAL_URI_REGEX = re.compile(
    r"^(?:s3://|https://[^/]+/)?(?:prd-tnm/)?StagedProducts/Elevation/13/TIFF/current/(?:[^/]+/)*(USGS_13_[a-zA-Z0-9_\-]+\.tif)$"
)


def load_hash_pinned_catalog(
    catalog_path: Path | str, expected_sha256: str | None = None
) -> tuple[list[dict[str, Any]], str]:
    """Load local catalog JSON export with SHA256 hash-pinning.

    Returns (records_list, computed_sha256).
    """
    path = Path(catalog_path)
    if not path.is_file():
        raise CatalogNotFoundError(f"3DEP catalog export not found at {path}")

    raw_bytes = path.read_bytes()
    computed_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    if expected_sha256 is not None:
        if computed_sha256.lower() != expected_sha256.lower():
            raise CatalogValidationError(
                f"Catalog SHA256 mismatch for {path}: computed {computed_sha256}, expected {expected_sha256}"
            )

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise CatalogValidationError(
            f"Invalid JSON in catalog file {path}: {exc}"
        ) from exc

    if isinstance(data, dict) and "records" in data:
        records = data["records"]
    elif isinstance(data, list):
        records = data
    else:
        raise CatalogValidationError(
            f"Catalog file {path} must contain a list of records or dict with 'records'"
        )

    return records, computed_sha256


def create_discovery_authorization_request(
    target_bbox: tuple[float, float, float, float] | None = None,
    details: dict[str, Any] | None = None,
    per_asset_bytes: int = 268435456,
) -> dict[str, Any]:
    """Emit deterministic discovery authorization request payload without making network calls."""
    target_geo = None
    if target_bbox:
        min_lon, min_lat, max_lon, max_lat = target_bbox
        target_geo = {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        }

    planned_prefixes = ["StagedProducts/Elevation/13/TIFF/current/"]
    proposed_operations = []
    for prefix in planned_prefixes:
        proposed_operations.append(
            {
                "provider": "usgs",
                "collection": "USGS_3DEP_13",
                "bucket": "prd-tnm",
                "operation": "s3:ListBucket",
                "prefix": prefix,
                "target": "catalog_index",
                "requests": 1,
                "reserved_bytes": per_asset_bytes,
                "requester_pays": False,
            }
        )
        proposed_operations.append(
            {
                "provider": "usgs",
                "collection": "USGS_3DEP_13",
                "bucket": "prd-tnm",
                "operation": "s3:GetObject",
                "key": f"{prefix}catalog_index.json",
                "prefix": prefix,
                "target": "catalog_snapshot",
                "requests": 1,
                "reserved_bytes": per_asset_bytes,
                "requester_pays": False,
            }
        )

    total_requests = len(proposed_operations)
    total_bytes = sum(op["reserved_bytes"] for op in proposed_operations)

    req_cost = Decimal("0.0000004") * Decimal(total_requests)
    gb_transferred = Decimal(total_bytes) / Decimal(1073741824)
    trans_cost = Decimal("0.09") * gb_transferred
    total_cost = req_cost + trans_cost
    max_spend_usd = f"{total_cost:.6f}"

    payload: dict[str, Any] = {
        "authorization_type": "discovery_authorization_request",
        "version": "1.0",
        "provider": "USGS",
        "collection": "USGS_3DEP_13",
        "target_geometry": target_geo,
        "target_bbox": list(target_bbox) if target_bbox else None,
        "planned_prefixes": planned_prefixes,
        "planned_operations": proposed_operations,
        "per_operation_byte_reservation": per_asset_bytes,
        "caps": {
            "max_requests": total_requests,
            "max_bytes": total_bytes,
            "max_cost_usd": max_spend_usd,
            "allow_requester_pays": False,
        },
        "required_authorization": {
            "currency": "USD",
            "max_requests": total_requests,
            "max_transfer_bytes": total_bytes,
            "max_spend_usd": max_spend_usd,
            "allow_requester_pays": False,
        },
        "rate_card_source": "USGS 3DEP Public Domain / S3 Index 2026",
        "resume_command": "python -m src.data_pipeline.usgs_3dep --catalog-path <path> --expected-sha256 <sha256>",
        "network_call_made": False,
        "details": details or {},
    }

    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
    return payload


def extract_resolution(record: dict[str, Any]) -> tuple[float, float]:
    """Extract (res_x, res_y) from record in any supported representation."""
    if "resolution_x" in record and "resolution_y" in record:
        return float(record["resolution_x"]), float(record["resolution_y"])
    if "pixel_size_x" in record and "pixel_size_y" in record:
        return float(record["pixel_size_x"]), float(record["pixel_size_y"])
    if "dx" in record and "dy" in record:
        return float(record["dx"]), float(record["dy"])
    if "native_resolution" in record:
        res = record["native_resolution"]
        if isinstance(res, (int, float)):
            return float(res), float(res)
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            return float(res[0]), float(res[1])
        if isinstance(res, dict):
            rx = res.get("x", res.get("res_x", res.get("dx")))
            ry = res.get("y", res.get("res_y", res.get("dy")))
            if rx is not None and ry is not None:
                return float(rx), float(ry)
    if "cell_size_x" in record and "cell_size_y" in record:
        return float(record["cell_size_x"]), float(record["cell_size_y"])
    raise ResolutionValidationError("Record lacks resolution fields")


def parse_footprint_bounds(
    footprint: dict[str, Any] | str,
) -> tuple[float, float, float, float]:
    """Parse GeoJSON geometry and return bounding box (min_lon, min_lat, max_lon, max_lat)."""
    if isinstance(footprint, str):
        try:
            footprint_dict = json.loads(footprint)
        except Exception as exc:
            raise RecordValidationError(
                f"Invalid GeoJSON footprint JSON string: {exc}"
            ) from exc
    elif isinstance(footprint, dict):
        footprint_dict = footprint
    else:
        raise RecordValidationError("Footprint must be a GeoJSON dict or JSON string")

    geom_type = footprint_dict.get("type")
    coords = footprint_dict.get("coordinates")
    if not geom_type or not coords:
        raise RecordValidationError("Footprint GeoJSON missing type or coordinates")

    lons: list[float] = []
    lats: list[float] = []

    def _collect(c: Any) -> None:
        if isinstance(c, (list, tuple)):
            if (
                len(c) >= 2
                and isinstance(c[0], (int, float))
                and isinstance(c[1], (int, float))
            ):
                lon, lat = float(c[0]), float(c[1])
                if not (math.isfinite(lat) and math.isfinite(lon)):
                    raise RecordValidationError(
                        "Footprint contains non-finite coordinates"
                    )
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise RecordValidationError(
                        f"Footprint coordinates out of WGS84 range: ({lat}, {lon})"
                    )
                lons.append(lon)
                lats.append(lat)
            else:
                for elem in c:
                    _collect(elem)

    _collect(coords)

    if not lats or not lons:
        raise RecordValidationError(
            "Footprint GeoJSON contains no valid coordinate points"
        )

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    return min_lon, min_lat, max_lon, max_lat


def validate_3dep_record(record: dict[str, Any]) -> SourceAsset:
    """Strictly validate a 3DEP catalog record and convert to SourceAsset.

    Fail closed if any invariant is violated.
    """
    # 1. Reject dynamic service output & mutable/weak identity
    if record.get("is_dynamic") or record.get("dynamic_service"):
        raise WeakIdentityError(
            "Dynamic service output cannot be treated as immutable asset"
        )

    if record.get("weak_identity") or record.get("mutable_identity"):
        raise WeakIdentityError(
            "Record explicitly flagged with weak or mutable identity"
        )

    uri = record.get("canonical_uri") or record.get("uri") or record.get("url") or ""
    if not uri or not isinstance(uri, str):
        raise WeakIdentityError("Missing or non-string canonical URI")

    # Reject URIs that rely solely on dynamic endpoints, OBJECTID, Best, or lack specific USGS_13_ filename
    match = CANONICAL_URI_REGEX.match(uri)
    if not match:
        raise WeakIdentityError(
            f"Canonical URI '{uri}' does not match official 3DEP 1/3-arc-second TIFF pattern "
            f"'prd-tnm/StagedProducts/Elevation/13/TIFF/current/.../USGS_13_*.tif'"
        )

    filename = match.group(1)
    filename_asset_id = filename.rsplit(".", 1)[0]
    asset_id = record.get("asset_id") or filename_asset_id
    if asset_id != filename_asset_id:
        raise WeakIdentityError(
            f"Asset ID '{asset_id}' does not match canonical URI object "
            f"'{filename_asset_id}'"
        )

    # Weak identity rejections: ETag alone without header/metadata observation, or OBJECTID alone
    if record.get("identity_source") in ("etag_alone", "objectid_alone", "best_alone"):
        raise WeakIdentityError(
            "Asset identity based on ETag, OBJECTID, or Best alone is weak"
        )

    # 2. Native resolution verification (magnitude ~ 9.259259e-05 deg)
    res_x, res_y = extract_resolution(record)
    mag_x = abs(res_x)
    mag_y = abs(res_y)

    if abs(mag_x - TARGET_RESOLUTION_DEG) > RESOLUTION_TOLERANCE_DEG:
        raise ResolutionValidationError(
            f"Native X resolution magnitude {mag_x:.8e} deviates from 1/3 arc-second target {TARGET_RESOLUTION_DEG:.8e}"
        )
    if abs(mag_y - TARGET_RESOLUTION_DEG) > RESOLUTION_TOLERANCE_DEG:
        raise ResolutionValidationError(
            f"Native Y resolution magnitude {mag_y:.8e} deviates from 1/3 arc-second target {TARGET_RESOLUTION_DEG:.8e}"
        )

    # 3. Horizontal CRS verification
    crs = record.get("horizontal_crs") or record.get("crs")
    if not crs or not isinstance(crs, str) or crs.upper() not in VALID_HORIZONTAL_CRS:
        raise MetadataValidationError(
            f"Invalid or missing horizontal CRS: {crs}; expected one of {sorted(VALID_HORIZONTAL_CRS)}"
        )

    # 4. Vertical datum realization verification (NAVD88)
    vdatum = record.get("vertical_datum") or record.get("vdatum")
    if (
        not vdatum
        or not isinstance(vdatum, str)
        or vdatum.upper() not in VALID_VERTICAL_DATUMS
    ):
        raise MetadataValidationError(
            f"Invalid or missing vertical datum: {vdatum}; NAVD88 realization required"
        )

    # 5. Elevation units verification (meters)
    units = record.get("elevation_unit") or record.get("units") or record.get("unit")
    if (
        not units
        or not isinstance(units, str)
        or units.lower() not in VALID_ELEVATION_UNITS
    ):
        raise MetadataValidationError(
            f"Invalid or missing elevation units: {units}; meters required"
        )

    # 6. No-data value verification
    nodata = record.get("nodata")
    if (
        nodata is None
        or not isinstance(nodata, (int, float))
        or not math.isfinite(float(nodata))
    ):
        raise MetadataValidationError(f"Invalid or missing no-data value: {nodata}")

    # 7. Metadata SHA256 & COG/header observation verification
    metadata_sha256 = record.get("metadata_sha256")
    if (
        not metadata_sha256
        or not isinstance(metadata_sha256, str)
        or len(metadata_sha256) != 64
    ):
        raise MetadataValidationError(
            "Missing or invalid 64-character metadata SHA256 string"
        )
    try:
        int(metadata_sha256, 16)
    except ValueError:
        raise MetadataValidationError(
            "metadata_sha256 must be a valid hexadecimal string"
        )

    cog_observed = record.get("cog_header_observed") or record.get("header_observed")
    if not cog_observed:
        raise MetadataValidationError(
            "Missing COG/header record observation confirmation"
        )
    object_size_bytes = record.get("object_size_bytes")
    if (
        isinstance(object_size_bytes, bool)
        or not isinstance(object_size_bytes, int)
        or object_size_bytes <= 0
    ):
        raise MetadataValidationError("Missing or invalid positive object_size_bytes")
    last_modified = record.get("last_modified")
    if not isinstance(last_modified, str) or not last_modified:
        raise MetadataValidationError("Missing last_modified object observation")
    if record.get("accept_ranges") is not True:
        raise MetadataValidationError("Missing confirmed byte-range support")
    if not any(record.get(name) for name in ("etag", "version_id", "checksum_sha256")):
        raise WeakIdentityError("Missing immutable object identity observation")

    # 8. Footprint GeoJSON verification
    footprint_raw = record.get("footprint_geojson") or record.get("footprint")
    if not footprint_raw:
        raise RecordValidationError("Missing footprint GeoJSON")

    if isinstance(footprint_raw, dict):
        footprint_str = json.dumps(footprint_raw, sort_keys=True, separators=(",", ":"))
    elif isinstance(footprint_raw, str):
        footprint_str = footprint_raw
    else:
        raise RecordValidationError("footprint must be dict or JSON string")

    # Validate coordinates inside footprint
    parse_footprint_bounds(footprint_raw)

    # Build canonical URI
    canonical_uri = uri
    if not canonical_uri.startswith("s3://"):
        if "prd-tnm/" in canonical_uri:
            idx = canonical_uri.index("prd-tnm/")
            canonical_uri = "s3://" + canonical_uri[idx:]
        else:
            canonical_uri = (
                f"s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/{filename}"
            )

    return SourceAsset(
        provider="USGS",
        collection="3DEP 1/3 arc-second",
        asset_id=asset_id,
        canonical_uri=canonical_uri,
        state_or_region=record.get("state_or_region"),
        acquisition_year=record.get("acquisition_year"),
        capture_date=record.get("capture_date"),
        published_at=record.get("published_at"),
        license="USGS Public Domain",
        attribution="USGS 3D Elevation Program",
        horizontal_crs=crs.upper(),
        vertical_datum="NAVD88",
        native_resolution=(mag_x, mag_y),
        band_contract="float32_elevation_meters",
        nodata=float(nodata),
        etag=record.get("etag"),
        version_id=record.get("version_id"),
        checksum_sha256=record.get("checksum_sha256"),
        metadata_sha256=metadata_sha256.lower(),
        object_size_bytes=object_size_bytes,
        last_modified=last_modified,
        accept_ranges=True,
        footprint_geojson=footprint_str,
    )


class USGS3DEPAdapter:
    """Fail-closed adapter for USGS 3DEP ImageServer catalog exports."""

    def __init__(
        self,
        catalog_path: Path | str | None = None,
        catalog_records: list[dict[str, Any]] | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.expected_sha256 = expected_sha256
        self.catalog_records: list[dict[str, Any]] | None = catalog_records
        self.catalog_sha256: str | None = None

        if self.catalog_path and self.catalog_records is None:
            self._load_catalog()

    def _load_catalog(self) -> None:
        if not self.catalog_path:
            return
        records, sha256_val = load_hash_pinned_catalog(
            self.catalog_path, self.expected_sha256
        )
        self.catalog_records = records
        self.catalog_sha256 = sha256_val

    def is_catalog_available(self) -> bool:
        """Return True if local catalog is loaded and validated."""
        return self.catalog_records is not None

    def get_discovery_authorization_request(
        self, target_bbox: tuple[float, float, float, float] | None = None
    ) -> dict[str, Any]:
        """Emit deterministic pre-discovery authorization request."""
        details = {}
        if self.catalog_path:
            details["catalog_path"] = str(self.catalog_path)
        if self.expected_sha256:
            details["expected_sha256"] = self.expected_sha256
        return create_discovery_authorization_request(target_bbox, details)

    def select_assets_for_region(
        self, target_bbox: tuple[float, float, float, float]
    ) -> list[SourceAsset]:
        """Select intersecting 3DEP assets for target bounding box (min_lat, min_lon, max_lat, max_lon).

        Fail-closed if local catalog is absent or invalid.
        """
        if not self.is_catalog_available() or self.catalog_records is None:
            raise CatalogNotFoundError(
                "No valid local hash-pinned 3DEP catalog available"
            )

        min_lon, min_lat, max_lon, max_lat = target_bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(f"Invalid bounding box coordinates: {target_bbox}")

        matched_assets: list[SourceAsset] = []
        for record in self.catalog_records:
            asset = validate_3dep_record(record)
            r_min_lon, r_min_lat, r_max_lon, r_max_lat = parse_footprint_bounds(
                asset.footprint_geojson
                or record.get("footprint_geojson")
                or record.get("footprint")
            )

            # Spatial intersection check (lon/lat standard)
            if (
                r_min_lon < max_lon
                and r_max_lon > min_lon
                and r_min_lat < max_lat
                and r_max_lat > min_lat
            ):
                matched_assets.append(asset)
        # Sort deterministically by asset_id
        matched_assets.sort(key=lambda a: (a.asset_id, a.canonical_uri))
        return matched_assets
