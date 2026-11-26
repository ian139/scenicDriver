"""Focused tests for the canonical open-data planner CLI
(scripts/ingest/plan_active_learning_region.py).

Covers: exact flag contract with no Mapbox options, plan-only behavior,
missing-catalog discovery authorization, cap-required execution mode, budget
and pilot bounds, authorized catalog resolution, acquisition continuation,
terminal exit statuses, identity drift, and the extended manifest contract.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

from scripts.ingest import plan_active_learning_region as planner
from scripts.ingest import naip_3dep_workflow as workflow
from scripts.ingest.naip_3dep_workflow import PLAN_SCHEMA_VERSION
from tests.naip3dep_fixtures import (
    CT_BBOX,
    NAIP_SAT_KEY,
    TILE_COORDS,
    build_fixture_objects,
    fake_planned_dict,
    make_naip_catalog,
    make_transport_factory,
    naip_asset,
    tnm_record,
    write_contract_with_catalogs,
)


@pytest.fixture(autouse=True)
def _fast_region_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan tests use a tiny planned dict instead of the full configured region
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
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_active_learning_region.py",
            "--region-spec",
            "config/data_sources/regions_v1.json",
        ],
    )
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


def test_planning_requires_explicit_region_spec_with_canonical_config_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["plan_active_learning_region.py", "--run-name", "pilot"]
    )
    with pytest.raises(SystemExit) as exc:
        planner.parse_args()
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "--region-spec is required for planning" in message
    assert "--region-spec config/data_sources/regions_v1.json" in message
    assert "config/app_regions.json" in message


def test_programmatic_planning_has_no_implicit_region_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config/app_regions.json"):
        _plan_run(tmp_path, spec=None)


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


@pytest.fixture
def discovered_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real metered raw discovery, ready for the authorized metadata boundary."""
    source_tree_sha = workflow.repository_source_tree_digest()
    monkeypatch.setattr(planner, "repository_source_tree_digest", lambda: source_tree_sha)
    monkeypatch.setattr(workflow, "repository_source_tree_digest", lambda: source_tree_sha)
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    land_path = tmp_path / "land geometry.geojson"
    min_lon, min_lat, max_lon, max_lat = CT_BBOX
    land_path.write_text(json.dumps({
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat],
            [min_lon, max_lat], [min_lon, min_lat],
        ]],
    }), encoding="utf-8")
    spec_path = tmp_path / "region spec.json"
    spec = _spec()
    spec["regions"][0].update(
        type="jurisdiction_land_intersection", land_geometry_file=str(land_path)
    )
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    options = {
        "run_name": "discovered",
        "spec_path": str(spec_path),
        "output_root": tmp_path / "output runs",
        "image_root": tmp_path / "output images",
        "source_contract_path": contract_path,
        "budget": 50,
        "pilot_budget": 2,
        "max_runtime_minutes": 3,
    }
    initial = planner.plan_run(**options)
    caps = workflow.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50"),
        allow_requester_pays=True,
    )
    transport_factory = make_transport_factory(build_fixture_objects())
    auth_path = Path(initial["authorization_path"])
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    raw = workflow.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=workflow.authorization_payload_digest(auth),
        contract_path=contract_path,
        missing=initial["missing_catalogs"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=initial["run_dir"],
        transport_factory=transport_factory,
    )
    metadata_options = {
        "metadata_request_path": raw["metadata_authorization_path"],
        "expected_metadata_request_sha256": raw["metadata_authorization_sha256"],
        "contract_path": contract_path,
        "run_dir": initial["run_dir"],
        "caps": caps,
        "allow_requester_pays": True,
        "transport_factory": transport_factory,
    }
    return options, initial, metadata_options, transport_factory


def test_metadata_discovery_cli_continues_frozen_acquisition(
    discovered_run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    options, initial, metadata_options, transport_factory = discovered_run
    original_contract = Path(options["source_contract_path"]).read_bytes()
    discover = planner.discover_catalog_metadata

    def offline_discover(**kwargs):
        return discover(**kwargs, transport_factory=transport_factory)

    monkeypatch.setattr(planner, "discover_catalog_metadata", offline_discover)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_active_learning_region.py",
            "--discover-catalog-metadata",
            "--run-name", options["run_name"],
            "--region-spec", options["spec_path"],
            "--output-root", str(options["output_root"]),
            "--image-root", str(options["image_root"]),
            "--source-contract", str(options["source_contract_path"]),
            "--budget", "50", "--pilot-budget", "2",
            "--max-runtime-minutes", "3",
            "--metadata-request", str(metadata_options["metadata_request_path"]),
            "--expected-metadata-request-sha256",
            metadata_options["expected_metadata_request_sha256"],
            "--allow-requester-pays",
            "--max-source-requests", "1000",
            "--max-transfer-bytes", "1000000000",
            "--max-local-bytes", "1000000000",
            "--max-requester-pays-usd", "50",
        ],
    )
    planner.main()
    result = json.loads(capsys.readouterr().out)
    assert result["plan_state"] == "planned"
    assert Path(options["source_contract_path"]).read_bytes() == original_contract
    run_dir = Path(initial["run_dir"])
    manifest = json.loads((run_dir / "region_manifest.json").read_text())
    assert (
        manifest["inputs"]["source_contract_sha256"]
        == result["resolved_contract_sha256"]
    )
    assert (
        manifest["source_contract_transition"]["from_sha256"]
        == initial["region_manifest"]["inputs"]["source_contract_sha256"]
    )
    plan_path = run_dir / "execution_plan.json"
    frozen_plan = plan_path.read_bytes()
    acquired = workflow.execute_plan(
        plan_path,
        expected_plan_sha256=result["execution_plan_sha256"],
        allow_requester_pays=True,
        max_source_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50"),
        workers=1,
        transport_factory=transport_factory,
    )
    assert acquired["state"] == "acquired"
    assert acquired["completed_coordinates"] == len(TILE_COORDS)
    assert acquired["failed_coordinates"] == 0
    assert plan_path.read_bytes() == frozen_plan
    for x, y in TILE_COORDS:
        for style in ("satellite", "terrain"):
            image_path = (
                options["image_root"] / style / "z14" / "sne_pilot" / f"{x}_{y}.png"
            )
            with Image.open(image_path) as image:
                assert image.size == (512, 512)
                assert image.mode == "RGB"


@pytest.mark.parametrize("state", ["failed", "aborted_cap", "acquired", "paused_deadline"])
def test_execute_cli_terminal_status(
    state: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        planner, "execute_plan",
        lambda *args, **kwargs: {
            "run_dir": "run", "plan_sha256": "a" * 64, "state": state,
            "completed_coordinates": 0, "failed_coordinates": int(state == "failed"),
            "pending_coordinates": 1, "counters": {},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_active_learning_region.py", "--execute-plan", "execution_plan.json",
            "--expected-plan-sha256", "a" * 64, "--allow-requester-pays",
            "--max-source-requests", "10", "--max-transfer-bytes", "1000",
            "--max-local-bytes", "1000", "--max-requester-pays-usd", "1",
        ],
    )
    if state in {"failed", "aborted_cap"}:
        with pytest.raises(SystemExit) as exc:
            planner.main()
        assert exc.value.code == 1
    else:
        assert planner.main() is None
    assert json.loads(capsys.readouterr().out)["state"] == state


@pytest.mark.parametrize(
    "mutation",
    [
        "unacknowledged", "origin", "resolved_hash", "preprocessing", "rate_card",
        "catalog_path", "catalog_hash", "budget", "pilot_budget", "geometry",
        "source_tree", "image_root", "frozen_plan", "wrong_stage",
    ],
)
def test_source_resolution_rejects_unauthorized_mutations(
    discovered_run, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, initial, metadata_options, _ = discovered_run
    resolution = workflow.discover_catalog_metadata(**metadata_options)
    resolved_path = Path(resolution["resolved_contract_path"])
    contract = json.loads(resolved_path.read_text())
    options = dict(options, source_contract_path=resolved_path)
    run_dir = Path(initial["run_dir"])
    manifest_path = run_dir / "region_manifest.json"
    original_manifest = manifest_path.read_bytes()
    if mutation == "unacknowledged":
        resolution = None
    elif mutation == "origin":
        contract["resolved_from_source_contract_sha256"] = "0" * 64
    elif mutation == "resolved_hash":
        resolution["resolved_contract_sha256"] = "0" * 64
    elif mutation == "preprocessing":
        contract["preprocessing"]["contract_id"] = "unauthorized"
    elif mutation == "rate_card":
        contract["rate_card"]["request_cost_usd"] = "1"
    elif mutation == "catalog_path":
        copy_path = resolved_path.parent / "unauthorized_catalog.json"
        copy_path.write_bytes(Path(contract["imagery"]["catalog_snapshot"]).read_bytes())
        contract["imagery"]["catalog_snapshot"] = str(copy_path)
    elif mutation == "catalog_hash":
        contract["imagery"]["catalog_sha256"] = "0" * 64
    elif mutation == "budget":
        options["budget"] = 49
    elif mutation == "pilot_budget":
        options["pilot_budget"] = None
    elif mutation == "geometry":
        def changed_geometry(*args, **kwargs):
            planned = fake_planned_dict()
            planned["geometry_digest"] = "0" * 64
            return planned
        monkeypatch.setattr(planner, "parse_and_validate_region_spec", changed_geometry)
    elif mutation == "source_tree":
        monkeypatch.setattr(planner, "repository_source_tree_digest", lambda: "0" * 64)
    elif mutation == "image_root":
        options["image_root"] = run_dir / "different_images"
    elif mutation == "frozen_plan":
        (run_dir / "execution_plan.json").write_text("{}")
    elif mutation == "wrong_stage":
        manifest = json.loads(original_manifest)
        manifest["stage_state"] = "planned"
        manifest_path.write_text(json.dumps(manifest))
    if mutation in {"origin", "preprocessing", "rate_card", "catalog_path", "catalog_hash"}:
        resolved_path.write_text(json.dumps(contract))
        resolution["resolved_contract_sha256"] = workflow.file_sha256(resolved_path)
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError):
        planner.plan_run(**options, metadata_resolution=resolution)
    assert manifest_path.read_bytes() == before


def test_source_resolution_cannot_rebind_frozen_plan(discovered_run) -> None:
    options, initial, metadata_options, _ = discovered_run
    resolution = workflow.discover_catalog_metadata(**metadata_options)
    options = dict(options, source_contract_path=resolution["resolved_contract_path"])
    planned = planner.plan_run(**options, metadata_resolution=resolution)
    run_dir = Path(initial["run_dir"])
    frozen_plan = (run_dir / "execution_plan.json").read_bytes()
    frozen_manifest = (run_dir / "region_manifest.json").read_bytes()
    again = planner.plan_run(**options)
    replanned_manifest = (run_dir / "region_manifest.json").read_bytes()
    assert again["execution_plan_sha256"] == planned["execution_plan_sha256"]
    assert (
        again["region_manifest"]["source_contract_transition"]
        == planned["region_manifest"]["source_contract_transition"]
    )
    with pytest.raises(ValueError, match="unfrozen discovery run"):
        planner.plan_run(**options, metadata_resolution=resolution)
    assert (run_dir / "execution_plan.json").read_bytes() == frozen_plan
    # A valid ordinary re-plan may refresh timestamps, but refused resolution
    # must leave that already-published manifest untouched.
    current_manifest = (run_dir / "region_manifest.json").read_bytes()
    assert json.loads(current_manifest)["inputs"] == json.loads(frozen_manifest)["inputs"]
    assert current_manifest == replanned_manifest


def test_metadata_discovery_rejects_insufficient_caps_without_resolution(
    discovered_run,
) -> None:
    options, initial, metadata_options, _ = discovered_run
    run_dir = Path(initial["run_dir"])
    original_manifest = (run_dir / "region_manifest.json").read_bytes()
    original_contract = Path(options["source_contract_path"]).read_bytes()
    metadata_options["caps"] = workflow.NetworkCaps(
        max_requests=1,
        max_transfer_bytes=1,
        max_local_bytes=1,
        max_requester_pays_usd=Decimal("0.0000000001"),
        allow_requester_pays=True,
    )

    def forbidden_transport(*args, **kwargs):
        pytest.fail("insufficient authorization must fail before transport construction")

    metadata_options["transport_factory"] = forbidden_transport
    with pytest.raises(ValueError):
        workflow.discover_catalog_metadata(**metadata_options)
    assert (run_dir / "region_manifest.json").read_bytes() == original_manifest
    assert Path(options["source_contract_path"]).read_bytes() == original_contract
    assert not (run_dir / "execution_plan.json").exists()
