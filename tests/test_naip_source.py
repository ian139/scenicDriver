"""Unit tests for fail-closed NAIP source adapter and catalog integrity checks."""

from __future__ import annotations

import hashlib
import json
import unittest
from typing import Any

from src.data_pipeline.naip import (
    ALLOWED_STATES,
    CatalogValidationError,
    CrossStateSubstitutionError,
    NaipAdapter,
    SourceAsset,
    create_discovery_authorization_request,
    select_naip_assets,
    validate_catalog_data,
)


def _make_valid_catalog(assets: list[dict[str, Any]]) -> dict[str, Any]:
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


def _make_ct_asset(
    key_suffix: str = "m_4107201_ne_18_060-20210816.tif",
    year: int = 2021,
    res: float = 0.6,
    capture_date: str = "2021-08-16",
) -> dict[str, Any]:
    key = f"s3://naip-visualization/ct/{year}/100m/rgbir/37072/{key_suffix}"
    return {
        "key": key,
        "state": "ct",
        "acquisition_year": year,
        "capture_date": capture_date,
        "published_at": f"{year}-10-01T00:00:00Z",
        "license": "Public Domain",
        "attribution": "USDA Farm Service Agency",
        "horizontal_crs": "EPSG:26918",
        "vertical_datum": None,
        "native_resolution": res,
        "band_contract": ["red", "green", "blue"],
        "nodata": 0,
        "etag": '"a1b2c3d4e5f6"',
        "version_id": None,
        "checksum_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
        "object_size_bytes": 1024,
        "last_modified": f"{year}-10-01T00:00:00Z",
        "accept_ranges": True,
        "footprint": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-73.5, 41.0],
                    [-73.0, 41.0],
                    [-73.0, 41.5],
                    [-73.5, 41.5],
                    [-73.5, 41.0],
                ]
            ],
        },
    }


class TestNaipSourceAdapter(unittest.TestCase):
    def test_invalid_snapshots(self) -> None:
        ct_asset = _make_ct_asset()
        base_cat = _make_valid_catalog([ct_asset])

        # 1. Invalid collection
        bad_coll = dict(base_cat, collection="invalid-collection")
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(bad_coll)

        # 2. Invalid bucket
        bad_bucket = dict(base_cat, bucket="wrong-bucket")
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(bad_bucket)

        # 3. Invalid region
        bad_region = dict(base_cat, region="us-east-1")
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(bad_region)

        # 4. Requester pays False (must be True for NAIP)
        bad_req_pays = dict(base_cat, requester_pays=False)
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(bad_req_pays)

        # 5. Catalog hash mismatch
        bad_hash = dict(base_cat, catalog_hash="f" * 64)
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(bad_hash)

        # 6. Key not starting with s3://naip-visualization/
        bad_key_asset = dict(ct_asset, key="http://naip-visualization/ct/2021/tile.tif")
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(_make_valid_catalog([bad_key_asset]))

        # 7. Key state vs record state mismatch
        bad_state_asset = dict(ct_asset, state="ma")
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(_make_valid_catalog([bad_state_asset]))

        # 8. Key year vs record year mismatch
        bad_year_asset = dict(ct_asset, acquisition_year=2023)
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(_make_valid_catalog([bad_year_asset]))

        # 9. Key year vs capture date year mismatch
        bad_date_asset = dict(ct_asset, capture_date="2019-05-01")
        with self.assertRaises(CatalogValidationError):
            validate_catalog_data(_make_valid_catalog([bad_date_asset]))

    def test_cross_state_substitution_refusal(self) -> None:
        ct_asset = _make_ct_asset("ct_tile.tif", 2021)
        ri_asset = dict(
            _make_ct_asset("ri_tile.tif", 2021),
            key="s3://naip-visualization/ri/2021/100m/rgbir/37072/ri_tile.tif",
            state="ri",
        )
        ma_asset = dict(
            _make_ct_asset("ma_tile.tif", 2021),
            key="s3://naip-visualization/ma/2021/100m/rgbir/37072/ma_tile.tif",
            state="ma",
        )

        catalog = _make_valid_catalog([ct_asset, ri_asset, ma_asset])
        adapter = NaipAdapter(catalog)

        # Query CT: MUST ONLY return CT asset, refusing RI and MA tiles even though footprints overlap
        bbox = (-73.5, 41.0, -73.0, 41.5)
        selected, contributors = adapter.select_assets(
            desired_state="ct", desired_vintage=2021, target_geometry=bbox
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].state_or_region, "ct")
        self.assertIn("ct/2021", selected[0].canonical_uri)
        self.assertEqual(len(contributors), 1)

        # Query unsupported state -> raises CrossStateSubstitutionError
        with self.assertRaises(CrossStateSubstitutionError):
            adapter.select_assets(
                desired_state="ca", desired_vintage=2021, target_geometry=bbox
            )

    def test_deterministic_ordering_mosaic_contributors(self) -> None:
        # Create 4 tiles with different vintage, capture date, resolution, and key
        t1 = _make_ct_asset(
            "b_tile_2021_06.tif", year=2021, res=0.6, capture_date="2021-08-16"
        )
        t2 = _make_ct_asset(
            "a_tile_2021_06.tif", year=2021, res=0.6, capture_date="2021-08-16"
        )
        t3 = _make_ct_asset(
            "c_tile_2021_10.tif", year=2021, res=1.0, capture_date="2021-08-16"
        )
        t4 = _make_ct_asset(
            "d_tile_2023_06.tif", year=2023, res=0.6, capture_date="2023-07-01"
        )

        catalog = _make_valid_catalog([t1, t2, t3, t4])
        adapter = NaipAdapter(catalog)

        bbox = (-73.5, 41.0, -73.0, 41.5)
        selected, contributors = adapter.select_assets(
            desired_state="ct", target_geometry=bbox
        )

        self.assertEqual(len(selected), 4)
        # Priority order:
        # 1. Vintage 2023 first (t4)
        # 2. Vintage 2021, finest resolution (0.6 over 1.0) -> t2 and t1 before t3
        # 3. Canonical URI tie-break -> "a_tile" (t2) before "b_tile" (t1)
        expected_keys = [
            t4["key"],
            t2["key"],
            t1["key"],
            t3["key"],
        ]
        actual_keys = [s.canonical_uri for s in selected]
        self.assertEqual(actual_keys, expected_keys)
        self.assertEqual(contributors, [s.asset_id for s in selected])

    def test_no_snapshot_authorization_artifact(self) -> None:
        auth_req = select_naip_assets(catalog_data=None, state="ct")
        self.assertIsInstance(auth_req, dict)
        assert isinstance(auth_req, dict)

        self.assertEqual(auth_req["provider"], "usda_naip")
        self.assertEqual(auth_req["collection"], "naip-visualization")
        self.assertEqual(auth_req["bucket"], "naip-visualization")
        self.assertEqual(auth_req["region"], "us-west-2")
        self.assertTrue(auth_req["requester_pays"])

        states = auth_req["target_states"]
        self.assertEqual(states, sorted(ALLOWED_STATES))

        proposed = auth_req["proposed_operations"]
        prefixes = [op["prefix"] for op in proposed]
        self.assertIn("ct/", prefixes)
        self.assertIn("ri/", prefixes)
        self.assertIn("ma/", prefixes)

        req_auth = auth_req["required_authorization"]
        self.assertIsInstance(req_auth["max_requests"], int)
        self.assertGreater(req_auth["max_requests"], 0)
        self.assertIsInstance(req_auth["max_transfer_bytes"], int)
        self.assertGreater(req_auth["max_transfer_bytes"], 0)
        self.assertIsInstance(req_auth["max_spend_usd"], str)
        self.assertTrue(req_auth["allow_requester_pays"])

        self.assertFalse(auth_req["secrets_present"])

    def test_secret_free_identities(self) -> None:
        ct_asset = _make_ct_asset()
        cat = _make_valid_catalog([ct_asset])
        adapter = NaipAdapter(cat)
        selected, _ = adapter.select_assets(desired_state="ct")

        def _check_no_secrets(obj: Any) -> None:
            s = json.dumps(obj)
            self.assertNotIn("AKIA", s)
            self.assertNotIn("bearer", s.lower())
            self.assertNotIn("password", s.lower())
            # Ensure no value contains actual secret key tokens
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        self.assertNotIn("secret_key", v.lower())
                        self.assertNotIn("aws_secret", v.lower())

        for asset in selected:
            _check_no_secrets(asset.to_dict())

        auth_req = create_discovery_authorization_request()
        _check_no_secrets(auth_req)

    def test_source_asset_serialization(self) -> None:
        asset = SourceAsset(
            provider="usda_naip",
            collection="naip-visualization",
            asset_id="asset123",
            canonical_uri="s3://naip-visualization/ct/2021/tile.tif",
            state_or_region="ct",
            acquisition_year=2021,
            native_resolution=0.6,
        )
        d = asset.to_dict()
        reconstructed = SourceAsset.from_dict(d)
        self.assertEqual(asset, reconstructed)
        self.assertEqual(asset.sha256(), reconstructed.sha256())


if __name__ == "__main__":
    unittest.main()
