"""Focused tests for the canonical open-data planner CLI
(scripts/ingest/plan_active_learning_region.py).

Covers: exact flag contract with no Mapbox options, plan-only behavior,
missing-catalog discovery authorization, cap-required execution mode, budget
and pilot bounds, identity drift, and the extended manifest contract.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.ingest import plan_active_learning_region as planner
from scripts.ingest.naip_3dep_workflow import PLAN_SCHEMA_VERSION
from tests.naip3dep_fixtures import (
    CT_BBOX,
    NAIP_SAT_KEY,
    TILE_COORDS,
    fake_planned_dict,
    make_naip_catalog,
    naip_asset,
    tnm_record,
    write_contract_with_catalogs,
)


@pytest.fixture(autouse=True)
def _fast_region_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan tests use a tiny planned dict instead of the full built-in region
    (which would enumerate tens of thousands of coordinates)."""

    def fake_parse(
        spec_data,
        app_regions_path="config/app_regions.json",
        max_budget_coords=370000,
        zoom=14,
    ):
        planned = fake_planned_dict()
        planned["spec_data"] = spec_data
        return planned

    monkeypatch.setattr(planner, "parse_and_validate_region_spec", fake_parse)


def _spec() -> dict:
    return {
        "version": 1,
        "geographic_source": "fixture",
        "included_jurisdictions": ["Connecticut"],
        "excluded_jurisdictions": [],
        "known_non_target_coverage": [],
        "limitations": ["fixture geometry"],
        "regions": [
            {
                "name": "sne_pilot",
                "type": "bbox",
                "bbox": {
                    "min_lat": 41.60,
                    "min_lon": -72.75,
                    "max_lat": 41.80,
                    "max_lon": -72.60,
                },
            }
        ],
    }


def _plan_run(tmp_path: Path, **overrides) -> dict:
    catalog = make_naip_catalog(
        [naip_asset("ct", 2021, "m_4107201_ne_18_060-20210816", CT_BBOX)]
    )
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=catalog, terrain_records=[tnm_record(CT_BBOX)]
    )
    kwargs = dict(
        run_name="pilot",
        spec=_spec(),
        output_root=tmp_path / "runs",
        image_root=tmp_path / "images",
        config_path="config/app_regions.json",
        source_contract_path=contract_path,
        budget=370000,
        pilot_budget=500,
    )
    kwargs.update(overrides)
    return planner.plan_run(**kwargs)


def test_parse_args_exposes_canonical_flags_and_no_mapbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["plan_active_learning_region.py"])
    args = planner.parse_args()
    namespace = vars(args)
    for flag in (
        "run_name",
        "region_spec",
        "region_source",
        "imagery_source",
        "terrain_source",
        "source_contract",
        "image_root",
        "output_root",
        "budget",
        "pilot_budget",
        "execute_plan",
        "expected_plan_sha256",
        "allow_requester_pays",
        "max_source_requests",
        "max_transfer_bytes",
        "max_local_bytes",
        "max_requester_pays_usd",
        "workers",
        "max_runtime_minutes",
    ):
        assert flag in namespace, f"missing canonical flag --{flag.replace('_', '-')}"
    # No active Mapbox acquisition option may exist.
    for forbidden in ("acquire", "mapbox", "s3_bucket", "s3_prefix_root"):
        assert forbidden not in namespace, (
            f"obsolete Mapbox-era option {forbidden} present"
        )
    assert args.imagery_source == "naip-visualization"
    assert args.terrain_source == "usgs-3dep-13as"
    assert args.budget == 370_000


def test_plan_only_writes_immutable_plan_and_authorization(tmp_path: Path) -> None:
    result = _plan_run(tmp_path)
    assert result["plan_state"] == "planned"
    assert result["missing_catalogs"] == []
    run_dir = tmp_path / "runs" / "pilot"

    plan_path = Path(result["execution_plan_path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["coordinate_count"] == len(TILE_COORDS)
    assert plan["missing_satellite"] == len(TILE_COORDS)
    assert plan["missing_terrain"] == len(TILE_COORDS)
    assert plan["caps"] == {
        "max_requests": None,
        "max_transfer_bytes": None,
        "max_local_bytes": None,
        "max_requester_pays_usd": None,
        "allow_requester_pays": False,
    }
    assert plan["requester_pays"]["imagery"] is True
    assert plan["requester_pays"]["terrain"] is False
    sha_file = (run_dir / "execution_plan.sha256").read_text().strip()
    assert sha_file == result["execution_plan_sha256"]
    assert sha_file == planner.plan_digest(plan)
    # The digest is the SHA-256 of the plan file bytes themselves.
    import hashlib

    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == sha_file

    auth = json.loads(
        (run_dir / "acquisition_authorization_request.json").read_text(encoding="utf-8")
    )
    assert auth["authorization_type"] == "acquisition_authorization_request"
    assert auth["execution_plan_sha256"] == sha_file
    assert auth["required_authorization"]["currency"] == "USD"
    assert auth["policy"]["requester_pays"] is False
    assert auth["policy"]["network_call_made"] is False

    preflight = json.loads(
        (run_dir / "acquisition_preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["state"] == "planned"
    assert preflight["pilot_budget"] == 500

    pilot_frame = json.loads((run_dir / "pilot_frame.json").read_text(encoding="utf-8"))
    assert len(pilot_frame["coordinates"]) == len(TILE_COORDS)
    assert pilot_frame["seed"] == 42


def test_plan_only_writes_extended_manifest(tmp_path: Path) -> None:
    _plan_run(tmp_path)
    with open(
        tmp_path / "runs" / "pilot" / "tile_manifest.csv",
        newline="",
        encoding="utf-8",
    ) as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    assert fieldnames == planner.COLUMNS
    assert len(rows) == len(TILE_COORDS)
    row = rows[0]
    assert row["satellite_asset_ids"] == json.dumps([NAIP_SAT_KEY])
    assert row["state"] == "ct"
    assert row["admission_reason"] == "center_on_land"
    assert row["source_contract_sha256"]
    assert row["preprocessing_contract_sha256"]
    assert row["boundary_geometry_sha256"]
    assert row["satellite_provider"] == "usda_naip"
    assert row["terrain_provider"] == "usgs"
    assert row["processing_version"]
    # Provenance scalars are populated only after execution.
    assert row["satellite_output_sha256"] == ""
    assert row["grid_sha256"] == ""


def test_missing_catalog_emits_discovery_authorization(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    result = planner.plan_run(
        run_name="disc",
        spec=_spec(),
        output_root=tmp_path / "runs",
        image_root=tmp_path / "images",
        config_path="config/app_regions.json",
        source_contract_path=contract_path,
        budget=370000,
    )
    assert result["plan_state"] == "needs_discovery"
    assert result["execution_plan_path"] is None
    assert result["execution_plan_sha256"] is None
    disc = json.loads(
        (tmp_path / "runs" / "disc" / "discovery_authorization_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert disc["authorization_type"] == "discovery_authorization_request"
    assert set(disc["missing_catalogs"]) == {"imagery", "terrain"}
    assert disc["policy"]["network_call_made"] is False
    assert isinstance(disc["required_authorization"]["max_requests"], int)
    assert disc["required_authorization"]["max_requests"] > 0
    assert isinstance(disc["required_authorization"]["max_spend_usd"], str)
    assert disc["required_authorization"]["currency"] == "USD"
    assert len(disc["proposed_operations"]) == 2
    preflight = json.loads(
        (tmp_path / "runs" / "disc" / "acquisition_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight["state"] == "needs_discovery"


def test_unknown_or_paid_sources_refused(tmp_path: Path) -> None:
    from scripts.ingest.naip_3dep_workflow import SourceContractError

    with pytest.raises(SourceContractError, match="not supported"):
        planner.plan_run(
            run_name="bad",
            spec=_spec(),
            output_root=tmp_path / "runs",
            image_root=tmp_path / "images",
            source_contract_path="config/data_sources/naip_3dep_v1.json",
            imagery_source="mapbox.satellite",
        )
    with pytest.raises(SourceContractError, match="not supported"):
        planner.plan_run(
            run_name="bad",
            spec=_spec(),
            output_root=tmp_path / "runs",
            image_root=tmp_path / "images",
            source_contract_path="config/data_sources/naip_3dep_v1.json",
            terrain_source="unknown-provider",
        )


def test_budget_and_pilot_bounds_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="budget"):
        _plan_run(tmp_path, budget=370_001)
    with pytest.raises(ValueError, match="pilot_budget"):
        _plan_run(tmp_path, pilot_budget=501)
    with pytest.raises(ValueError, match="pilot_budget"):
        _plan_run(tmp_path, pilot_budget=0)
    with pytest.raises(ValueError, match="exceed"):
        _plan_run(tmp_path, budget=200, pilot_budget=500)


def test_plan_run_rejects_invalid_run_names(tmp_path: Path) -> None:
    for bad in ("", "../traversal", "a/b", "invalid name"):
        with pytest.raises(ValueError):
            _plan_run(tmp_path, run_name=bad)


def test_existing_run_rejects_identity_drift(tmp_path: Path) -> None:
    first = _plan_run(tmp_path)
    manifest_path = Path(first["run_dir"]) / "region_manifest.json"
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="identity drift"):
        _plan_run(tmp_path, budget=300_000)
    assert manifest_path.read_bytes() == before

    manifest_data = json.loads(before.decode("utf-8"))
    manifest_data["inputs"]["repository_source_tree_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    with pytest.raises(
        ValueError, match="existing run identity drift: repository_source_tree_digest"
    ):
        _plan_run(tmp_path)

    manifest_path.write_bytes(before)
    # Identical inputs re-plan deterministically (same plan digest).
    again = _plan_run(tmp_path)
    assert again["execution_plan_sha256"] == first["execution_plan_sha256"]


def test_execute_mode_requires_plan_hash_and_positive_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed = {}

    def fake_execute(plan_path, **kwargs):
        executed["plan_path"] = plan_path
        executed["kwargs"] = kwargs
        return {
            "run_dir": "run",
            "plan_sha256": "0" * 64,
            "state": "acquired",
            "completed_coordinates": 1,
            "failed_coordinates": 0,
            "pending_coordinates": 0,
            "counters": {},
        }

    monkeypatch.setattr(planner, "execute_plan", fake_execute)

    # Execute mode without --expected-plan-sha256 -> SystemExit before any work.
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_active_learning_region.py",
            "--execute-plan",
            "execution_plan.json",
            "--allow-requester-pays",
            "--max-source-requests",
            "10",
            "--max-transfer-bytes",
            "1000",
            "--max-local-bytes",
            "1000",
            "--max-requester-pays-usd",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        planner.main()

    # Execute mode without --allow-requester-pays -> SystemExit.
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_active_learning_region.py",
            "--execute-plan",
            "execution_plan.json",
            "--expected-plan-sha256",
            "0" * 64,
            "--max-source-requests",
            "10",
            "--max-transfer-bytes",
            "1000",
            "--max-local-bytes",
            "1000",
            "--max-requester-pays-usd",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        planner.main()

    # Execute mode with zero request cap -> SystemExit.
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_active_learning_region.py",
            "--execute-plan",
            "execution_plan.json",
            "--expected-plan-sha256",
            "0" * 64,
            "--allow-requester-pays",
            "--max-source-requests",
            "0",
            "--max-transfer-bytes",
            "1000",
            "--max-local-bytes",
            "1000",
            "--max-requester-pays-usd",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        planner.main()

    # A fully-specified execute invocation reaches execute_plan.
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_active_learning_region.py",
            "--execute-plan",
            str(tmp_path / "execution_plan.json"),
            "--expected-plan-sha256",
            "0" * 64,
            "--allow-requester-pays",
            "--max-source-requests",
            "10",
            "--max-transfer-bytes",
            "1000",
            "--max-local-bytes",
            "1000",
            "--max-requester-pays-usd",
            "1",
            "--workers",
            "2",
        ],
    )
    planner.main()
    assert executed["plan_path"] == str(tmp_path / "execution_plan.json")
    assert executed["kwargs"]["max_source_requests"] == 10
    assert executed["kwargs"]["allow_requester_pays"] is True


def test_source_contract_and_preprocessing_artifacts_written(tmp_path: Path) -> None:
    _plan_run(tmp_path)
    run_dir = tmp_path / "runs" / "pilot"
    source_contract = json.loads(
        (run_dir / "source_contract.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads(
        (run_dir / "preprocessing_contract.json").read_text(encoding="utf-8")
    )
    assert source_contract["contract_id"] == "naip_3dep_v1"
    assert preprocessing["zoom"] == 14
    assert (preprocessing["tile_width"], preprocessing["tile_height"]) == (512, 512)
    manifest = json.loads(
        (run_dir / "region_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sources"]["satellite"]["collection"] == "naip-visualization"
    assert manifest["sources"]["mapbox"]["used"] is False
    assert manifest["stage_state"] == "planned"
    assert manifest["ready_for_selection"] is False

    assert len(manifest["repository_source_tree_digest"]) == 64
    assert (
        manifest["inputs"]["repository_source_tree_digest"]
        == manifest["repository_source_tree_digest"]
    )

    provider_urls = manifest["sources"]["provider_urls"]
    assert provider_urls is not None
    assert provider_urls["naip_registry"] == "https://registry.opendata.aws/naip/"
    assert provider_urls["usgs_3dep"] == "https://www.usgs.gov/3d-elevation-program"
    assert (
        provider_urls["census_boundaries"]
        == "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0/query"
    )
    assert provider_urls["aws_s3_pricing"] == "https://aws.amazon.com/s3/pricing/"

    access_dates = manifest["sources"]["access_dates"]
    assert access_dates["naip_registry"] == "2026-08-10"
    assert access_dates["usgs_3dep"] == "2026-08-10"
    assert access_dates["census_boundaries"] == "2026-08-10"
    assert access_dates["aws_s3_pricing"] == "2026-08-10"

    plan = json.loads((run_dir / "execution_plan.json").read_text(encoding="utf-8"))
    assert (
        plan["hashes"]["repository_source_tree_digest"]
        == manifest["repository_source_tree_digest"]
    )

    auth = json.loads(
        (run_dir / "acquisition_authorization_request.json").read_text(encoding="utf-8")
    )
    assert (
        auth["identities"]["repository_source_tree_digest"]
        == manifest["repository_source_tree_digest"]
    )
