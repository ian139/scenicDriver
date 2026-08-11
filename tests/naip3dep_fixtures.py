"""Shared fixture builders for NAIP + 3DEP planner/workflow focused tests.

Provides deterministic tiny valid NAIP RGB GeoTIFF bytes and 3DEP float32 GeoTIFF
bytes, companion FGDC XML metadata, normalized catalog records, and a metered
fake transport for unit/workflow testing without network calls.
"""

from __future__ import annotations

import hashlib
import math
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from src.data_pipeline.metered_transport import (
    CapExceededError,
    Counters,
    HeadObjectResult,
    IdentityMismatchError,
    ListObjectsResult,
    MalformedResponseError,
    MeteredLedger,
    NetworkCaps,
    NotFoundError,
    ObjectMeta,
    RangeResult,
    RateCard,
    SharedMeteredBudget,
    _validate_range_request,
)
from src.data_pipeline.region_planning import enumerate_bbox_tiles
from src.data_pipeline.web_mercator import tile_bounds_wgs84

# Two real adjacent z14 Web Mercator tiles in Connecticut (Hartford area).
TILE_COORDS = enumerate_bbox_tiles(41.60, -72.75, 41.80, -72.60, zoom=14)[:2]
_TILE_BOUNDS = [tile_bounds_wgs84(x, y, 14) for x, y in TILE_COORDS]
CT_BBOX = (
    min(bounds[0] for bounds in _TILE_BOUNDS),
    min(bounds[1] for bounds in _TILE_BOUNDS),
    max(bounds[2] for bounds in _TILE_BOUNDS),
    max(bounds[3] for bounds in _TILE_BOUNDS),
)

NAIP_SAT_KEY = (
    "s3://naip-visualization/ct/2021/100m/rgbir/37072/m_4107201_ne_18_060-20210816.tif"
)
NAIP_OLD_KEY = (
    "s3://naip-visualization/ct/2019/100m/rgbir/37072/m_4107201_ne_18_060-20190601.tif"
)
TNM_DEM_KEY = (
    "s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n42w072/USGS_13_n42w072.tif"
)
TNM_XML_KEY = (
    "s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n42w072/USGS_13_n42w072.xml"
)

FGDC_3DEP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<metadata>
  <idinfo>
    <spdoinfo><direct>Raster</direct></spdoinfo>
  </idinfo>
  <vsys>
    <altunits>meters</altunits>
    <altdatum>NAVD88</altdatum>
  </vsys>
</metadata>
"""


def _footprint(bbox: tuple[float, float, float, float]) -> dict:
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
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


def write_geotiff(
    path: str | Path | None,
    arr: np.ndarray,
    crs: str,
    bounds: tuple[float, float, float, float],
    nodata: float | None = None,
    colorinterp: list | None = None,
    tags: dict | None = None,
) -> bytes:
    """Write or return a tiny GeoTIFF fixture (uint8 RGB or float32 single band)."""
    with MemoryFile() as memfile:
        height, width = arr.shape[-2], arr.shape[-1]
        count = 1 if arr.ndim == 2 else arr.shape[0]
        minx, miny, maxx, maxy = bounds
        with memfile.open(
            driver="GTiff",
            width=width,
            height=height,
            count=count,
            dtype=arr.dtype,
            crs=crs,
            transform=from_bounds(minx, miny, maxx, maxy, width, height),
            nodata=nodata,
            tiled=True,
            blockxsize=16,
            blockysize=16,
        ) as dst:
            if arr.ndim == 2:
                dst.write(arr, 1)
            else:
                dst.write(arr)
            if colorinterp:
                dst.colorinterp = colorinterp
            if tags:
                dst.update_tags(**tags)
        content = memfile.read()
    if path is not None:
        Path(path).write_bytes(content)
    return content


def _covering_grid(
    bounds: tuple[float, float, float, float],
    resolution: float = 1.0 / 10800.0,
) -> tuple[int, int, tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bounds
    width = max(32, math.ceil((maxx - minx) / resolution))
    height = max(32, math.ceil((maxy - miny) / resolution))
    return (
        width,
        height,
        (
            minx,
            miny,
            minx + width * resolution,
            miny + height * resolution,
        ),
    )


def build_naip_geotiff_bytes(
    bounds: tuple[float, float, float, float] = CT_BBOX,
    crs: str = "EPSG:4326",
) -> bytes:
    """Build deterministic tiled 3-band uint8 RGB GeoTIFF bytes."""
    width, height, raster_bounds = _covering_grid(bounds)
    rgb = np.zeros((3, height, width), dtype=np.uint8)
    rgb[0] = 120
    rgb[1] = 80
    rgb[2] = 30
    return write_geotiff(
        None,
        rgb,
        crs,
        raster_bounds,
        colorinterp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
    )


def build_3dep_geotiff_bytes(
    bounds: tuple[float, float, float, float] = CT_BBOX,
    crs: str = "EPSG:4269",
    nodata: float = -9999.0,
    tags: dict | None = None,
) -> bytes:
    """Build a tiled float32 GeoTIFF covering the fixture tiles at 1/3 arc-second."""
    width, height, raster_bounds = _covering_grid(bounds)
    dem = np.full((height, width), 250.5, dtype=np.float32)
    return write_geotiff(None, dem, crs, raster_bounds, nodata=nodata, tags=tags)


def build_fixture_objects(tile_coords=TILE_COORDS) -> dict[str, bytes]:
    """Tiny valid GeoTIFF bytes and companion metadata covering test tiles."""
    naip_bytes = build_naip_geotiff_bytes()
    dep_bytes = build_3dep_geotiff_bytes()
    xml_bytes = FGDC_3DEP_XML.encode("utf-8")
    manifest_bytes = f"{NAIP_SAT_KEY}\n".encode("utf-8")

    return {
        NAIP_SAT_KEY: naip_bytes,
        TNM_DEM_KEY: dep_bytes,
        TNM_XML_KEY: xml_bytes,
        "manifest.txt": manifest_bytes,
        "s3://naip-visualization/manifest.txt": manifest_bytes,
    }


def naip_asset(
    state: str = "ct",
    year: int = 2021,
    name: str | None = None,
    bbox: tuple[float, float, float, float] = CT_BBOX,
    key: str | None = None,
    content_bytes: bytes | None = None,
    capture_date: str | None = None,
) -> dict:
    """A catalog record that passes validate_catalog_data."""
    if capture_date is None:
        capture_date = f"{year}-08-16"
    date_digits = capture_date.replace("-", "")
    if name is None:
        name = f"m_4107201_ne_18_060-{date_digits}"
    if key is None:
        key = f"s3://naip-visualization/{state}/{year}/100m/rgbir/37072/{name}.tif"
    fp = _footprint(bbox)
    content_sha = (
        hashlib.sha256(content_bytes).hexdigest() if content_bytes else "a" * 64
    )
    size = len(content_bytes) if content_bytes else 4096
    etag = (
        f'"{hashlib.sha256(content_bytes).hexdigest()[:16]}"'
        if content_bytes
        else '"etag"'
    )
    metadata_sha = hashlib.sha256(
        json.dumps(
            {
                "canonical_uri": key,
                "object_size_bytes": size,
                "etag": etag,
                "last_modified": f"{year}-08-16T00:00:00Z",
                "capture_date": capture_date,
                "horizontal_crs": "EPSG:3857",
                "native_resolution": 0.6,
                "band_contract": ["red", "green", "blue"],
                "accept_ranges": True,
                "footprint": fp,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "provider": "usda_naip",
        "collection": "naip-visualization",
        "asset_id": key,
        "canonical_uri": key,
        "key": key,
        "state": state,
        "state_or_region": state,
        "acquisition_year": year,
        "capture_date": capture_date,
        "license": "Public Domain",
        "attribution": "USDA NAIP",
        "horizontal_crs": "EPSG:3857",
        "native_resolution": 0.6,
        "band_contract": ["red", "green", "blue"],
        "metadata_sha256": metadata_sha,
        "etag": etag,
        "checksum_sha256": content_sha,
        "object_size_bytes": size,
        "last_modified": f"{year}-08-16T00:00:00Z",
        "accept_ranges": True,
        "footprint": fp,
        "footprint_geojson": json.dumps(fp, sort_keys=True, separators=(",", ":")),
    }


def make_naip_catalog(assets: list[dict]) -> dict:
    """A normalized catalog dict that passes validate_catalog_data."""
    catalog_hash = hashlib.sha256(
        json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "collection": "naip-visualization",
        "bucket": "naip-visualization",
        "region": "us-west-2",
        "requester_pays": True,
        "raw_hash": "a" * 64,
        "catalog_hash": catalog_hash,
        "parser_hash": "b" * 64,
        "region_hash": "c" * 64,
        "assets": assets,
    }


def tnm_record(
    bbox: tuple[float, float, float, float] = CT_BBOX,
    quad: str = "n42w072",
    name: str = "USGS_13_n42w072.tif",
    content_bytes: bytes | None = None,
) -> dict:
    """A 3DEP 1/3 arc-second record that passes validate_3dep_record."""
    key = f"s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/{quad}/{name}"
    fp = _footprint(bbox)
    content_sha = (
        hashlib.sha256(content_bytes).hexdigest() if content_bytes else "c" * 64
    )
    size = len(content_bytes) if content_bytes else 4096
    etag = (
        f'"{hashlib.sha256(content_bytes).hexdigest()[:16]}"'
        if content_bytes
        else '"etag"'
    )
    res = 1.0 / 10800.0
    metadata_sha = hashlib.sha256(
        json.dumps(
            {
                "canonical_uri": key,
                "object_size_bytes": size,
                "etag": etag,
                "last_modified": "2021-08-16T00:00:00Z",
                "horizontal_crs": "EPSG:4269",
                "resolution_x": res,
                "resolution_y": res,
                "nodata": -9999.0,
                "vertical_datum": "NAVD88",
                "elevation_unit": "meters",
                "accept_ranges": True,
                "footprint": fp,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "provider": "USGS",
        "collection": "USGS_3DEP_13",
        "asset_id": f"USGS_13_{quad}",
        "canonical_uri": key,
        "uri": key,
        "license": "Public Domain",
        "attribution": "USGS 3DEP",
        "horizontal_crs": "EPSG:4269",
        "crs": "EPSG:4269",
        "vertical_datum": "NAVD88",
        "vdatum": "NAVD88",
        "elevation_unit": "meters",
        "units": "meters",
        "nodata": -9999.0,
        "native_resolution": [res, res],
        "resolution_x": res,
        "resolution_y": res,
        "metadata_sha256": metadata_sha,
        "cog_header_observed": True,
        "header_observed": True,
        "etag": etag,
        "checksum_sha256": content_sha,
        "object_size_bytes": size,
        "last_modified": "2021-08-16T00:00:00Z",
        "accept_ranges": True,
        "footprint": fp,
        "footprint_geojson": json.dumps(fp, sort_keys=True, separators=(",", ":")),
    }


def fake_planned_dict(tile_coords=TILE_COORDS) -> dict:
    """A tiny ``parse_and_validate_region_spec``-shaped result."""
    return {
        "spec_data": {
            "version": 1,
            "geographic_source": "fixture",
            "included_jurisdictions": ["Connecticut"],
            "excluded_jurisdictions": [],
            "known_non_target_coverage": [],
            "limitations": ["fixture geometry"],
        },
        "zoom": 14,
        "unique_coordinates_count": len(tile_coords),
        "total_rasters_count": len(tile_coords) * 2,
        "nen_tile_count": 0,
        "coord_to_region": {coord: "sne_pilot" for coord in tile_coords},
        "ordered_coords": list(tile_coords),
        "region_tile_counts": {"sne_pilot": len(tile_coords)},
        "geometry_digest": hashlib.sha256(b"fixture-geometry").hexdigest(),
        "coord_to_admission_reason": {coord: "center_on_land" for coord in tile_coords},
    }


def write_contract_with_catalogs(
    tmp_path: Path,
    *,
    imagery_catalog: dict | None,
    terrain_records: list[dict] | None,
) -> Path:
    """Copy the repo source contract and point its catalogs at fixtures."""
    import shutil

    contract_path = tmp_path / "naip_3dep_v1.json"
    shutil.copy("config/data_sources/naip_3dep_v1.json", contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if imagery_catalog is not None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = catalog_dir / "naip_visualization_catalog.json"
        catalog_path.write_text(json.dumps(imagery_catalog))
        contract["imagery"]["catalog_snapshot"] = str(catalog_path)
    else:
        contract["imagery"]["catalog_snapshot"] = str(tmp_path / "missing_naip.json")
    if terrain_records is not None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = catalog_dir / "usgs_3dep_13as_catalog.json"
        catalog_path.write_text(json.dumps({"records": terrain_records}))
        contract["terrain"]["catalog_snapshot"] = str(catalog_path)
    else:
        contract["terrain"]["catalog_snapshot"] = str(tmp_path / "missing_tnm.json")
    contract_path.write_text(json.dumps(contract, indent=2))
    return contract_path


class FakeTransport:
    """A MeteredTransport over in-memory bytes with pre-dispatch cap checks.

    Mirrors the real transport's fail-closed semantics: reservations are
    checked before dispatch, a request attempt counts even when it fails,
    pagination is honored, expected ETags are checked, and HEAD Accept-Ranges
    status is returned.
    """

    def __init__(
        self,
        bucket: str,
        objects: dict[str, bytes],
        caps: NetworkCaps,
        ledger: MeteredLedger,
        rate_card: RateCard,
        requester_pays: bool,
        shared_budget: SharedMeteredBudget,
        etags: dict[str, str] | None = None,
        accept_ranges: dict[str, bool | None] | None = None,
        last_modifieds: dict[str, str] | None = None,
        missing_keys: set[str] | None = None,
        head_sizes: dict[str, int] | None = None,
    ) -> None:
        self.bucket = bucket
        self.objects = dict(objects)
        self.caps = caps
        self.ledger = ledger
        self.rate_card = rate_card
        self.requester_pays = requester_pays
        self.shared_budget = shared_budget
        self.etags = etags or {}
        self.accept_ranges = accept_ranges or {}
        self.last_modifieds = last_modifieds or {}
        self.missing_keys = missing_keys or set()
        self.head_sizes = head_sizes or {}

    def _reserve(self, transfer: int) -> None:
        estimated = self.rate_card.cost_for(transfer)
        if self.requester_pays and not (
            self.caps.allow_requester_pays
            and self.caps.max_requester_pays_usd > Decimal("0")
        ):
            raise CapExceededError(
                "requester_pays_usd",
                self.shared_budget.cost_usd,
                estimated,
                Decimal("0"),
            )
        self.shared_budget.reserve(
            requests=1,
            transfer_bytes=transfer,
            local_bytes=transfer,
            cost_usd=estimated,
            requester_pays=self.requester_pays,
        )
        self.ledger.append(
            {
                "event": "reserve",
                "bucket": self.bucket,
                "reserved_transfer_bytes": transfer,
                "reserved_cost_usd": str(estimated),
            }
        )

    def _resolve_key(self, key: str) -> str | None:
        if key in self.missing_keys:
            return None
        if key in self.objects:
            return key
        clean = key.replace("s3://naip-visualization/", "").replace("s3://prd-tnm/", "")
        if clean in self.missing_keys:
            return None
        if clean in self.objects:
            return clean
        for k in self.objects:
            if k == clean or k.endswith(key) or key.endswith(k):
                return k
        return None

    def _get_etag(self, resolved_key: str, content: bytes) -> str:
        if resolved_key in self.etags:
            return self.etags[resolved_key]
        full_uri = (
            f"s3://{self.bucket}/{resolved_key}"
            if not resolved_key.startswith("s3://")
            else resolved_key
        )
        if full_uri in self.etags:
            return self.etags[full_uri]
        return f'"{hashlib.sha256(content).hexdigest()[:16]}"'

    def head_object(
        self,
        key: str,
        *,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> HeadObjectResult:
        self._reserve(0)
        rkey = self._resolve_key(key)
        if rkey is None:
            raise NotFoundError(f"Object '{key}' not found in fake transport")
        content = self.objects[rkey]
        etag = self._get_etag(rkey, content)
        if expected_etag is not None and etag != expected_etag:
            raise IdentityMismatchError(
                f"head_object ETag mismatch for '{key}': expected {expected_etag}, got {etag}"
            )
        last_modified = self.last_modifieds.get(
            rkey, self.last_modifieds.get(key, "2021-08-16T00:00:00Z")
        )
        accept_ranges = self.accept_ranges.get(rkey, self.accept_ranges.get(key, True))
        return HeadObjectResult(
            key=key,
            size=self.head_sizes.get(rkey, self.head_sizes.get(key, len(content))),
            etag=etag,
            version_id=expected_version_id,
            last_modified=last_modified,
            accept_ranges=accept_ranges,
        )

    def list_objects(
        self,
        prefix: str = "",
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
        max_response_bytes: int | None = None,
    ) -> ListObjectsResult:
        if not (1 <= max_keys <= 1000):
            raise ValueError(f"max_keys must be in [1, 1000], got {max_keys}")
        if max_response_bytes is None or max_response_bytes <= 0:
            raise ValueError("list_objects requires positive max_response_bytes")

        self._reserve(max_response_bytes)

        matching_keys = []
        seen_clean = set()
        for k in sorted(self.objects.keys()):
            clean = k.replace("s3://naip-visualization/", "").replace(
                "s3://prd-tnm/", ""
            )
            if clean in seen_clean:
                continue
            if clean.startswith(prefix) or k.startswith(prefix):
                matching_keys.append(k)
                seen_clean.add(clean)
        if continuation_token is not None:
            try:
                start_idx = int(continuation_token)
            except ValueError:
                start_idx = 0
                for i, k in enumerate(matching_keys):
                    if k > continuation_token:
                        start_idx = i
                        break
        else:
            start_idx = 0

        page_keys = matching_keys[start_idx : start_idx + max_keys]
        has_more = (start_idx + max_keys) < len(matching_keys)
        next_token = str(start_idx + max_keys) if has_more else None

        items = []
        for k in page_keys:
            content = self.objects[k]
            clean = k.replace("s3://naip-visualization/", "").replace(
                "s3://prd-tnm/", ""
            )
            etag = self._get_etag(k, content)
            last_mod = self.last_modifieds.get(
                k, self.last_modifieds.get(clean, "2021-08-16T00:00:00Z")
            )
            items.append(
                ObjectMeta(
                    key=clean,
                    size=len(content),
                    etag=etag,
                    last_modified=last_mod,
                )
            )

        raw_keys = [item.key for item in items]
        raw_bytes_len = len(json.dumps(raw_keys).encode("utf-8"))
        if raw_bytes_len > max_response_bytes:
            raise MalformedResponseError(
                f"list_objects response size {raw_bytes_len} exceeds max_response_bytes {max_response_bytes}"
            )

        return ListObjectsResult(
            objects=tuple(items),
            is_truncated=has_more,
            next_continuation_token=next_token,
        )

    def get_range(
        self,
        key: str,
        *,
        start: int | None = None,
        end: int | None = None,
        max_response_bytes: int | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> RangeResult:
        start_valid, end_valid, reservation = _validate_range_request(
            start, end, max_response_bytes
        )
        rkey = self._resolve_key(key)
        if rkey is None:
            raise NotFoundError(f"Object '{key}' not found in fake transport")
        content = self.objects[rkey]
        etag = self._get_etag(rkey, content)
        if expected_etag is not None and etag != expected_etag:
            raise IdentityMismatchError(
                f"get_range ETag mismatch for '{key}': expected {expected_etag}, got {etag}"
            )
        s_idx = start_valid if start_valid is not None else 0
        e_idx = end_valid + 1 if end_valid is not None else len(content)
        actual = content[s_idx:e_idx]
        if len(actual) > reservation:
            raise MalformedResponseError(
                f"get_range response length {len(actual)} exceeds max_response_bytes {reservation}"
            )
        self._reserve(len(actual))
        return RangeResult(
            key=key,
            content=actual,
            start=start_valid,
            end=end_valid,
            etag=etag,
            version_id=expected_version_id,
        )

    def get_object(
        self,
        key: str,
        *,
        max_response_bytes: int | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> bytes:
        return self.get_range(
            key=key,
            start=None,
            end=None,
            max_response_bytes=max_response_bytes,
            expected_etag=expected_etag,
            expected_version_id=expected_version_id,
        ).content

    def counters(self) -> Counters:
        return self.shared_budget.counters()


def make_transport_factory(
    objects: dict[str, bytes],
    etags: dict[str, str] | None = None,
    accept_ranges: dict[str, bool | None] | None = None,
    last_modifieds: dict[str, str] | None = None,
    missing_keys: set[str] | None = None,
    head_sizes: dict[str, int] | None = None,
):
    """Return a transport_factory callable for execute_plan injection."""

    def factory(
        bucket: str,
        caps,
        ledger,
        rate_card,
        requester_pays: bool,
        shared_budget: SharedMeteredBudget,
        *args,
        **kwargs,
    ):
        if "naip" in bucket:
            sub = {
                k: v
                for k, v in objects.items()
                if "naip-visualization" in k or k == "manifest.txt"
            }
        else:
            sub = {
                k: v
                for k, v in objects.items()
                if "prd-tnm" in k or "StagedProducts" in k
            }
        return FakeTransport(
            bucket=bucket,
            objects=sub,
            caps=caps,
            ledger=ledger,
            rate_card=rate_card,
            requester_pays=requester_pays,
            shared_budget=shared_budget,
            etags=etags,
            accept_ranges=accept_ranges,
            last_modifieds=last_modifieds,
            missing_keys=missing_keys,
            head_sizes=head_sizes,
        )

    return factory
