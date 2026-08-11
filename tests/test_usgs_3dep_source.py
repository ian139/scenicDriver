"""Tests for USGS 3DEP elevation source adapter."""

from __future__ import annotations

import hashlib
import json
import pytest
from pathlib import Path

from src.data_pipeline.usgs_3dep import (
    USGS3DEPAdapter,
    CatalogNotFoundError,
    CatalogValidationError,
    MetadataValidationError,
    ResolutionValidationError,
    WeakIdentityError,
    load_hash_pinned_catalog,
    validate_3dep_record,
)


def _make_valid_record(
    asset_id: str = "USGS_13_n42w072",
    res_x: float = 9.259259e-05,
    res_y: float = -9.259259e-05,
    min_lon: float = -72.0,
    min_lat: float = 41.0,
    max_lon: float = -71.0,
    max_lat: float = 42.0,
) -> dict:
    return {
        "provider": "USGS",
        "collection": "3DEP 1/3 arc-second",
        "asset_id": asset_id,
        "canonical_uri": f"s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n42w072/{asset_id}.tif",
        "state_or_region": "MA",
        "acquisition_year": 2021,
        "horizontal_crs": "EPSG:4269",
        "vertical_datum": "NAVD88",
        "elevation_unit": "meters",
        "nodata": -9999.0,
        "resolution_x": res_x,
        "resolution_y": res_y,
        "metadata_sha256": "a" * 64,
        "cog_header_observed": True,
        "etag": '"a1b2c3d4e5f6"',
        "object_size_bytes": 4096,
        "last_modified": "2021-10-01T00:00:00Z",
        "accept_ranges": True,
        "footprint_geojson": {
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
        },
    }


def test_positive_and_negative_y_resolution():
    """Test that positive and negative Y resolutions near 9.259259e-05 pass, while invalid resolutions fail."""
    # Negative Y resolution (typical north-up raster)
    rec_neg = _make_valid_record(res_x=9.259259e-05, res_y=-9.259259e-05)
    asset_neg = validate_3dep_record(rec_neg)
    assert asset_neg.asset_id == "USGS_13_n42w072"
    assert asset_neg.native_resolution == (9.259259e-05, 9.259259e-05)

    # Positive Y resolution
    rec_pos = _make_valid_record(res_x=9.259259e-05, res_y=9.259259e-05)
    asset_pos = validate_3dep_record(rec_pos)
    assert asset_pos.native_resolution == (9.259259e-05, 9.259259e-05)

    # 1-arc-second (~2.777e-4) resolution should fail
    rec_1arc = _make_valid_record(
        res_x=2.777777777777778e-04, res_y=-2.777777777777778e-04
    )
    with pytest.raises(ResolutionValidationError, match="deviates from 1/3 arc-second"):
        validate_3dep_record(rec_1arc)

    # 1-meter (~8.99e-6) resolution should fail
    rec_1m = _make_valid_record(res_x=8.99e-06, res_y=-8.99e-06)
    with pytest.raises(ResolutionValidationError, match="deviates from 1/3 arc-second"):
        validate_3dep_record(rec_1m)


def test_mutable_weak_identity_rejection():
    """Test that records with dynamic service endpoints or weak/mutable identity are rejected."""
    # Dynamic service flag
    rec_dyn = _make_valid_record()
    rec_dyn["is_dynamic"] = True
    with pytest.raises(WeakIdentityError, match="Dynamic service output"):
        validate_3dep_record(rec_dyn)

    # Missing USGS_13_ filename pattern in URI
    rec_bad_uri = _make_valid_record()
    rec_bad_uri["canonical_uri"] = (
        "s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n42w072/latest.tif"
    )
    with pytest.raises(
        WeakIdentityError,
        match="does not match official 3DEP 1/3-arc-second TIFF pattern",
    ):
        validate_3dep_record(rec_bad_uri)

    # ETag alone without verified header observation
    rec_etag = _make_valid_record()
    rec_etag["identity_source"] = "etag_alone"
    with pytest.raises(
        WeakIdentityError, match="ETag, OBJECTID, or Best alone is weak"
    ):
        validate_3dep_record(rec_etag)

    # Mutable identity flag
    rec_mut = _make_valid_record()
    rec_mut["mutable_identity"] = True
    with pytest.raises(WeakIdentityError, match="weak or mutable identity"):
        validate_3dep_record(rec_mut)


def test_missing_crs_datum_nodata_units_rejection():
    """Test rejection when horizontal CRS, vertical datum, units, nodata, or metadata SHA are missing/invalid."""
    # Invalid CRS
    rec = _make_valid_record()
    rec["horizontal_crs"] = "EPSG:3857"
    with pytest.raises(
        MetadataValidationError, match="Invalid or missing horizontal CRS"
    ):
        validate_3dep_record(rec)

    # Invalid vertical datum (e.g. NGVD29)
    rec = _make_valid_record()
    rec["vertical_datum"] = "NGVD29"
    with pytest.raises(MetadataValidationError, match="vertical datum"):
        validate_3dep_record(rec)

    # Invalid units (feet)
    rec = _make_valid_record()
    rec["elevation_unit"] = "feet"
    with pytest.raises(MetadataValidationError, match="elevation units"):
        validate_3dep_record(rec)

    # Missing nodata
    rec = _make_valid_record()
    rec["nodata"] = None
    with pytest.raises(MetadataValidationError, match="no-data value"):
        validate_3dep_record(rec)

    # Invalid metadata SHA256
    rec = _make_valid_record()
    rec["metadata_sha256"] = "short"
    with pytest.raises(MetadataValidationError, match="64-character metadata SHA256"):
        validate_3dep_record(rec)

    # Missing COG header observation
    rec = _make_valid_record()
    rec["cog_header_observed"] = False
    with pytest.raises(MetadataValidationError, match="COG/header record observation"):
        validate_3dep_record(rec)


def test_deterministic_region_selection():
    """Test spatial bounding box filtering and deterministic asset ordering."""
    rec1 = _make_valid_record(
        "USGS_13_n42w072", min_lon=-72.0, min_lat=41.0, max_lon=-71.0, max_lat=42.0
    )
    rec2 = _make_valid_record(
        "USGS_13_n43w072", min_lon=-72.0, min_lat=42.0, max_lon=-71.0, max_lat=43.0
    )
    rec3 = _make_valid_record(
        "USGS_13_n44w072", min_lon=-72.0, min_lat=43.0, max_lon=-71.0, max_lat=44.0
    )

    # Pass in unsorted order
    adapter = USGS3DEPAdapter(catalog_records=[rec3, rec1, rec2])

    # Region matching rec1 and rec2 only (min_lon, min_lat, max_lon, max_lat)
    selected = adapter.select_assets_for_region((-71.5, 41.5, -71.2, 42.5))
    assert len(selected) == 2
    assert [a.asset_id for a in selected] == ["USGS_13_n42w072", "USGS_13_n43w072"]

    # Region matching rec1 only
    selected_1 = adapter.select_assets_for_region((-71.9, 41.1, -71.1, 41.9))
    assert len(selected_1) == 1
    assert selected_1[0].asset_id == "USGS_13_n42w072"


def test_authorization_boundary_and_hash_pinning(tmp_path: Path):
    """Test pre-discovery authorization request generation and hash-pinned catalog loading."""
    # Absent catalog adapter emits discovery authorization payload without network calls
    adapter_empty = USGS3DEPAdapter()
    assert not adapter_empty.is_catalog_available()
    auth_req = adapter_empty.get_discovery_authorization_request(
        target_bbox=(-72.0, 41.0, -71.0, 42.0)
    )

    assert auth_req["authorization_type"] == "discovery_authorization_request"
    assert auth_req["network_call_made"] is False
    assert auth_req["provider"] == "USGS"
    assert auth_req["collection"] == "USGS_3DEP_13"
    assert "sha256" in auth_req
    assert len(auth_req["sha256"]) == 64
    assert isinstance(auth_req["required_authorization"]["max_requests"], int)
    assert auth_req["required_authorization"]["max_requests"] > 0
    assert auth_req["required_authorization"]["allow_requester_pays"] is False

    # Save catalog to file and test hash-pinning
    catalog_content = json.dumps({"records": [_make_valid_record()]}).encode("utf-8")
    cat_file = tmp_path / "catalog.json"
    cat_file.write_bytes(catalog_content)
    expected_sha = hashlib.sha256(catalog_content).hexdigest()

    records, sha = load_hash_pinned_catalog(cat_file, expected_sha)
    assert len(records) == 1
    assert sha == expected_sha

    # Mismatched SHA256 fails hash-pinning
    with pytest.raises(CatalogValidationError, match="Catalog SHA256 mismatch"):
        load_hash_pinned_catalog(cat_file, "0" * 64)

    # Missing file raises CatalogNotFoundError
    with pytest.raises(CatalogNotFoundError):
        load_hash_pinned_catalog(tmp_path / "nonexistent.json")
