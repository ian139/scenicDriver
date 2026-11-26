"""Fail-closed NAIP (naip-visualization) raster source adapter over hash-pinned local catalog JSON."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from typing import Any, Sequence

from .source_contracts import SourceAsset


class CatalogValidationError(ValueError):
    """Raised when NAIP normalized catalog JSON fails integrity or identity checks."""

    pass


class CrossStateSubstitutionError(ValueError):
    """Raised when cross-state asset substitution is attempted."""

    pass


ALLOWED_STATES = {
    "ct",
    "dc",
    "de",
    "ma",
    "md",
    "me",
    "mi",
    "nh",
    "nj",
    "ny",
    "oh",
    "pa",
    "ri",
    "va",
    "vt",
    "wv",
}
EXPECTED_COLLECTION = "naip-visualization"
EXPECTED_BUCKET = "naip-visualization"
EXPECTED_REGION = "us-west-2"
EXPECTED_REQUESTER_PAYS = True


def create_discovery_authorization_request(
    states: Sequence[str] | None = None,
    per_asset_bytes: int = 268435456,
) -> dict[str, Any]:
    """Return a deterministic discovery-authorization request artifact for NAIP S3 access."""
    target_states = sorted(list(set(s.lower() for s in (states or ALLOWED_STATES))))

    proposed_operations = []
    for state in target_states:
        proposed_operations.append(
            {
                "provider": "usda_naip",
                "collection": EXPECTED_COLLECTION,
                "bucket": EXPECTED_BUCKET,
                "operation": "s3:ListBucket",
                "prefix": f"{state}/",
                "target": "catalog_index",
                "requests": 1,
                "reserved_bytes": per_asset_bytes,
                "requester_pays": EXPECTED_REQUESTER_PAYS,
            }
        )
        proposed_operations.append(
            {
                "provider": "usda_naip",
                "collection": EXPECTED_COLLECTION,
                "bucket": EXPECTED_BUCKET,
                "operation": "s3:GetObject",
                "key": f"{state}/manifest.json",
                "prefix": f"{state}/",
                "target": "catalog_manifest",
                "requests": 1,
                "reserved_bytes": per_asset_bytes,
                "requester_pays": EXPECTED_REQUESTER_PAYS,
            }
        )

    total_requests = len(proposed_operations)
    total_bytes = sum(op["reserved_bytes"] for op in proposed_operations)

    req_cost = Decimal("0.0000004") * Decimal(total_requests)
    gb_transferred = Decimal(total_bytes) / Decimal(1073741824)
    trans_cost = Decimal("0.09") * gb_transferred
    total_cost = req_cost + trans_cost
    max_spend_usd = f"{total_cost:.6f}"

    return {
        "provider": "usda_naip",
        "collection": EXPECTED_COLLECTION,
        "bucket": EXPECTED_BUCKET,
        "region": EXPECTED_REGION,
        "requester_pays": EXPECTED_REQUESTER_PAYS,
        "target_states": target_states,
        "proposed_operations": proposed_operations,
        "per_operation_byte_reservation": per_asset_bytes,
        "required_authorization": {
            "currency": "USD",
            "max_requests": total_requests,
            "max_transfer_bytes": total_bytes,
            "max_local_bytes": total_bytes,
            "max_spend_usd": max_spend_usd,
            "allow_requester_pays": True,
        },
        "caps": {
            "max_requests": total_requests,
            "max_transfer_bytes": total_bytes,
            "max_spend_usd": max_spend_usd,
            "allow_requester_pays": True,
        },
        "has_secrets": False,
        "secrets_present": False,
    }


def calculate_intersection_area(geom1: Any, geom2: Any) -> float:
    """Return the exact planar intersection area for supported geometries."""
    from shapely.geometry import box, shape
    from shapely.geometry.base import BaseGeometry

    def _to_shapely(value: Any) -> BaseGeometry:
        if isinstance(value, BaseGeometry):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 4:
            return box(*(float(part) for part in value))
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            if value.get("type") == "Feature":
                value = value.get("geometry")
            if not isinstance(value, dict):
                raise ValueError("Feature lacks a geometry")
            return shape(value)
        raise TypeError(f"Unsupported geometry type: {type(value).__name__}")

    return float(_to_shapely(geom1).intersection(_to_shapely(geom2)).area)


def validate_catalog_data(catalog_data: dict[str, Any]) -> dict[str, Any]:
    """Validate normalized NAIP catalog JSON structure, hashes, and identity."""
    if not isinstance(catalog_data, dict):
        raise CatalogValidationError("Catalog data must be a dictionary")

    coll = catalog_data.get("collection")
    if not coll or str(coll).lower() != EXPECTED_COLLECTION:
        raise CatalogValidationError(
            f"Invalid collection identity '{coll}', expected '{EXPECTED_COLLECTION}'"
        )

    bucket = catalog_data.get("bucket")
    if not bucket or str(bucket).lower() != EXPECTED_BUCKET:
        raise CatalogValidationError(
            f"Invalid bucket identity '{bucket}', expected '{EXPECTED_BUCKET}'"
        )

    region = catalog_data.get("region")
    if not region or str(region).lower() != EXPECTED_REGION:
        raise CatalogValidationError(
            f"Invalid region identity '{region}', expected '{EXPECTED_REGION}'"
        )

    req_pays = catalog_data.get("requester_pays")
    if req_pays is not True:
        raise CatalogValidationError(
            f"Invalid requester_pays setting '{req_pays}', expected True"
        )

    # Hash verifications (pinned local catalog hashes)
    raw_hash = catalog_data.get("raw_hash") or catalog_data.get("raw_sha256")
    catalog_hash = catalog_data.get("catalog_hash") or catalog_data.get(
        "catalog_sha256"
    )
    parser_hash = catalog_data.get("parser_hash") or catalog_data.get("parser_sha256")
    region_hash = catalog_data.get("region_hash") or catalog_data.get("region_sha256")

    for h_name, h_val in [
        ("raw_hash", raw_hash),
        ("catalog_hash", catalog_hash),
        ("parser_hash", parser_hash),
        ("region_hash", region_hash),
    ]:
        if h_val is None:
            raise CatalogValidationError(f"Missing required catalog hash '{h_name}'")
        if (
            not isinstance(h_val, str)
            or len(h_val) != 64
            or not re.match(r"^[0-9a-fA-F]{64}$", h_val)
        ):
            raise CatalogValidationError(f"Invalid {h_name} hex format: '{h_val}'")

    assets = catalog_data.get("assets")
    if not isinstance(assets, list):
        raise CatalogValidationError("Catalog 'assets' field must be a list")

    if catalog_hash:
        computed = hashlib.sha256(
            json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if catalog_hash.lower() != computed.lower():
            raise CatalogValidationError(
                f"Catalog hash mismatch: expected {catalog_hash}, computed {computed}"
            )

    for idx, raw_asset in enumerate(assets):
        if not isinstance(raw_asset, dict):
            raise CatalogValidationError(f"Asset at index {idx} is not a dictionary")

        key = raw_asset.get("key") or raw_asset.get("canonical_uri")
        if not key or not str(key).startswith(f"s3://{EXPECTED_BUCKET}/"):
            raise CatalogValidationError(
                f"Asset at index {idx} key '{key}' does not start with verbatim 's3://{EXPECTED_BUCKET}/'"
            )

        # Path parsing: s3://naip-visualization/{state}/{year}/...
        subpath = str(key)[len(f"s3://{EXPECTED_BUCKET}/") :]
        parts = [p for p in subpath.split("/") if p]
        if len(parts) < 2:
            raise CatalogValidationError(
                f"Asset key '{key}' does not follow state/year pattern"
            )

        key_state = parts[0].lower()
        key_year_str = parts[1]

        if key_state not in ALLOWED_STATES:
            raise CatalogValidationError(
                f"Asset state '{key_state}' in key '{key}' is not in allowed states {ALLOWED_STATES}"
            )

        asset_state = str(
            raw_asset.get("state") or raw_asset.get("state_or_region") or ""
        ).lower()
        if asset_state != key_state:
            raise CatalogValidationError(
                f"Asset state mismatch: key state '{key_state}' vs record state '{asset_state}'"
            )

        try:
            key_year = int(key_year_str)
        except ValueError:
            raise CatalogValidationError(f"Invalid year in asset key: '{key_year_str}'")

        rec_year = raw_asset.get("acquisition_year")
        if rec_year is not None and int(rec_year) != key_year:
            raise CatalogValidationError(
                f"Year mismatch: key year '{key_year}' vs record year '{rec_year}'"
            )

        capture_date = raw_asset.get("capture_date")
        if capture_date:
            date_year = str(capture_date).split("-")[0]
            if date_year.isdigit() and int(date_year) != key_year:
                raise CatalogValidationError(
                    f"Capture date year mismatch: key year '{key_year}' vs date '{capture_date}'"
                )

        footprint = raw_asset.get("footprint") or raw_asset.get("footprint_geojson")
        if not footprint:
            raise CatalogValidationError(
                f"Asset at index {idx} missing footprint GeoJSON"
            )
        required_fields = (
            "capture_date",
            "license",
            "attribution",
            "horizontal_crs",
            "native_resolution",
            "band_contract",
            "metadata_sha256",
            "etag",
            "object_size_bytes",
            "last_modified",
        )
        missing = [
            name for name in required_fields if raw_asset.get(name) in (None, "")
        ]
        if missing:
            raise CatalogValidationError(
                f"Asset at index {idx} is missing required metadata: {missing}"
            )
        metadata_sha256 = raw_asset["metadata_sha256"]
        if (
            not isinstance(metadata_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", metadata_sha256) is None
        ):
            raise CatalogValidationError(
                f"Asset at index {idx} has invalid metadata_sha256"
            )
        if raw_asset.get("accept_ranges") is not True:
            raise CatalogValidationError(
                f"Asset at index {idx} lacks an observed byte-range contract"
            )
        bands = raw_asset["band_contract"]
        if isinstance(bands, dict):
            bands_list = (
                bands.get("bands") or bands.get("channels") or bands.get("color_space")
            )
            normalized_bands = (
                [str(v).lower() for v in bands_list]
                if isinstance(bands_list, list)
                else [str(bands_list).lower()]
            )
        elif isinstance(bands, list):
            normalized_bands = [str(value).lower() for value in bands]
        else:
            normalized_bands = [str(bands).lower()]
        if normalized_bands not in (["red", "green", "blue"], ["rgb"], ["r", "g", "b"]):
            raise CatalogValidationError(
                f"Asset at index {idx} is not a 3-band RGB visualization COG"
            )

    return catalog_data


class NaipAdapter:
    """Fail-closed adapter for NAIP visualization assets over local catalog JSON."""

    def __init__(self, catalog_data: dict[str, Any] | str | None = None) -> None:
        self._catalog_data: dict[str, Any] | None = None
        if catalog_data is not None:
            if isinstance(catalog_data, str):
                catalog_data = json.loads(catalog_data)
            self._catalog_data = validate_catalog_data(catalog_data)

    @property
    def is_valid(self) -> bool:
        return self._catalog_data is not None

    def select_assets(
        self,
        desired_state: str,
        desired_vintage: int | Sequence[int] | None = None,
        target_geometry: Any = None,
        native_resolution: float | None = None,
    ) -> tuple[list[SourceAsset], list[str]]:
        """Deterministically select positive-area intersecting NAIP assets.

        Returns (selected_assets, contributor_order).
        Refuses cross-state substitution strictly.
        """
        if not self.is_valid or self._catalog_data is None:
            raise CatalogValidationError(
                "Cannot select assets from absent or invalid catalog snapshot"
            )

        norm_state = desired_state.lower().strip()
        if norm_state not in ALLOWED_STATES:
            raise CrossStateSubstitutionError(
                f"Refusing state '{desired_state}': only states {sorted(list(ALLOWED_STATES))} supported"
            )

        vintages: set[int] | None = None
        if desired_vintage is not None:
            if isinstance(desired_vintage, int):
                vintages = {desired_vintage}
            else:
                vintages = {int(v) for v in desired_vintage}

        raw_assets = self._catalog_data.get("assets", [])
        candidates: list[SourceAsset] = []

        for raw in raw_assets:
            key = str(raw.get("key") or raw.get("canonical_uri"))
            parts = [p for p in key[len(f"s3://{EXPECTED_BUCKET}/") :].split("/") if p]
            asset_state = parts[0].lower()

            # STRICT CROSS-STATE SUBSTITUTION REFUSAL
            if asset_state != norm_state:
                continue

            rec_state = str(
                raw.get("state") or raw.get("state_or_region") or ""
            ).lower()
            if rec_state != norm_state:
                raise CrossStateSubstitutionError(
                    f"Asset {key} has mismatched state '{rec_state}' for requested state '{norm_state}'"
                )

            year = int(parts[1])
            if vintages is not None and year not in vintages:
                continue

            footprint_raw = raw.get("footprint") or raw.get("footprint_geojson")
            if isinstance(footprint_raw, dict):
                footprint_str = json.dumps(
                    footprint_raw, sort_keys=True, separators=(",", ":")
                )
            else:
                footprint_str = str(footprint_raw)

            # Spatial filter: positive-area intersection check
            if target_geometry is not None:
                area = calculate_intersection_area(footprint_raw, target_geometry)
                if area <= 0.0:
                    continue

            res = raw.get("native_resolution")
            if res is not None:
                res = float(res)
                if res <= 0.0:
                    continue
                if (
                    native_resolution is not None
                    and abs(res - float(native_resolution)) > 1e-5
                ):
                    continue

            source_asset = SourceAsset(
                provider="aws-naip",
                collection=EXPECTED_COLLECTION,
                asset_id=key,
                canonical_uri=key,
                state_or_region=norm_state,
                acquisition_year=year,
                capture_date=str(raw["capture_date"]),
                published_at=raw.get("published_at"),
                license=str(raw["license"]),
                attribution=str(raw["attribution"]),
                horizontal_crs=str(raw["horizontal_crs"]),
                vertical_datum=raw.get("vertical_datum"),
                native_resolution=float(raw["native_resolution"]),
                band_contract=raw["band_contract"],
                nodata=raw.get("nodata"),
                etag=raw.get("etag"),
                version_id=raw.get("version_id"),
                checksum_sha256=raw.get("checksum_sha256"),
                metadata_sha256=str(raw["metadata_sha256"]),
                object_size_bytes=int(raw["object_size_bytes"]),
                last_modified=str(raw["last_modified"]),
                accept_ranges=True,
                footprint_geojson=footprint_str,
            )
            candidates.append(source_asset)

        def _date_key(value: str | None) -> int:
            digits = "".join(
                character for character in (value or "") if character.isdigit()
            )
            return int(digits[:8]) if len(digits) >= 8 else 0

        ordered_assets = sorted(
            candidates,
            key=lambda asset: (
                -(asset.acquisition_year or 0),
                -_date_key(asset.capture_date),
                asset.native_resolution or float("inf"),
                asset.canonical_uri,
            ),
        )
        contributor_order = [asset.asset_id for asset in ordered_assets]

        return ordered_assets, contributor_order


def select_naip_assets(
    catalog_data: dict[str, Any] | str | None,
    state: str,
    vintage: int | Sequence[int] | None = None,
    target_geometry: Any = None,
) -> tuple[list[SourceAsset], list[str]] | dict[str, Any]:
    """Helper function to select NAIP assets or return discovery authorization if catalog is absent."""
    if catalog_data is None:
        return create_discovery_authorization_request()

    adapter = NaipAdapter(catalog_data)
    return adapter.select_assets(state, vintage, target_geometry)
