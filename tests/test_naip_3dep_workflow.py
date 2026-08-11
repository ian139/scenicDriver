"""Focused tests for the NAIP + 3DEP ingest workflow (planning, authorization,
execution, resume, caps, provenance). All fixtures are local; no network."""

from __future__ import annotations

import math
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

from scripts.ingest import naip_3dep_workflow as wf
from scripts.ingest.naip_3dep_workflow import (
    PlanDriftError,
    atomic_write_json,
    build_execution_plan,
    build_tile_plans,
    execute_plan,
    file_sha256,
    load_and_verify_plan,
    load_source_contract,
    plan_digest,
    select_pilot_frame,
)
from src.data_pipeline.metered_transport import (
    MeteredLedger,
    NetworkCaps,
    NotFoundError,
    RateCard,
    SharedMeteredBudget,
)
from tests.naip3dep_fixtures import (
    CT_BBOX,
    NAIP_SAT_KEY,
    TNM_DEM_KEY,
    TNM_XML_KEY,
    TILE_COORDS,
    build_3dep_geotiff_bytes,
    build_fixture_objects,
    fake_planned_dict,
    make_naip_catalog,
    make_transport_factory,
    naip_asset,
    tnm_record,
    write_contract_with_catalogs,
)


def _plan_and_execute(
    tmp_path: Path,
    *,
    run_name: str = "wf_run",
    max_source_requests: int = 200,
    allow_requester_pays: bool = True,
    workers: int = 2,
    terrain_records: list[dict] | None = None,
) -> tuple[dict, dict, dict]:
    fixture_objects = build_fixture_objects()
    naip_bytes = fixture_objects[NAIP_SAT_KEY]
    terrain_bytes = fixture_objects[TNM_DEM_KEY]
    assets = [
        naip_asset(
            "ct",
            2021,
            "m_4107201_ne_18_060-20210816",
            CT_BBOX,
            capture_date="2021-08-16",
            content_bytes=naip_bytes,
        ),
        naip_asset(
            "ct",
            2019,
            "m_4107201_ne_18_060-20190601",
            CT_BBOX,
            capture_date="2019-06-01",
            content_bytes=naip_bytes,
        ),
    ]
    catalog = make_naip_catalog(assets)
    records = (
        terrain_records
        if terrain_records is not None
        else [tnm_record(CT_BBOX, content_bytes=terrain_bytes)]
    )
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=catalog, terrain_records=records
    )

    planned = fake_planned_dict()
    contract, contract_sha = load_source_contract(contract_path)
    imagery_catalog = wf.load_naip_catalog(
        contract["imagery"]["catalog_snapshot"], None
    )
    terrain_assets = wf.load_3dep_catalog(contract["terrain"]["catalog_snapshot"], None)
    tile_plans_result = build_tile_plans(
        planned=planned,
        contract=contract,
        imagery_catalog=imagery_catalog,
        terrain_assets=terrain_assets,
        zoom=14,
    )
    rows = [
        {
            "region": planned["coord_to_region"][(x, y)],
            "z": 14,
            "x": x,
            "y": y,
            "satellite_present": False,
            "terrain_present": False,
            "source_contract_sha256": contract_sha,
            "preprocessing_contract_sha256": wf.identity_digest(
                contract["preprocessing"]
            ),
            "boundary_geometry_sha256": planned["geometry_digest"],
        }
        for x, y in planned["ordered_coords"]
    ]
    min_lon, min_lat, max_lon, max_lat = CT_BBOX
    land_geometry_path = tmp_path / "land.geojson"
    land_geometry_path.write_text(
        json.dumps(
            {
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
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plan = build_execution_plan(
        run_name=run_name,
        zoom=14,
        contract=contract,
        contract_sha=contract_sha,
        contract_path=contract_path,
        spec_path=None,
        spec_sha="s" * 64,
        geometry_digest=planned["geometry_digest"],
        imagery_catalog_path=contract["imagery"]["catalog_snapshot"],
        imagery_catalog_sha=wf.catalog_sha(contract["imagery"]["catalog_snapshot"]),
        terrain_catalog_path=contract["terrain"]["catalog_snapshot"],
        terrain_catalog_sha=wf.catalog_sha(contract["terrain"]["catalog_snapshot"]),
        tile_plans_result=tile_plans_result,
        planned_rows=rows,
        tile_manifest_sha="m" * 64,
        pilot_frame_sha=None,
        budget=370000,
        pilot_budget=None,
        image_root=str(tmp_path / "images"),
        output_root=str(tmp_path / "runs"),
        config_path="config/app_regions.json",
        imagery_source="naip-visualization",
        terrain_source="usgs-3dep-13as",
        max_runtime_minutes=None,
        land_geometry_path=land_geometry_path,
        land_geometry_sha256=wf.file_sha256(land_geometry_path),
        missing_satellite=len(rows),
        missing_terrain=len(rows),
        local_reuse_count=0,
        repository_source_tree_digest=wf.repository_source_tree_digest(),
    )
    plan_sha = plan_digest(plan)
    run_dir = tmp_path / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "execution_plan.json"
    wf.atomic_write_text(plan_path, wf.canonical_json_bytes(plan).decode("utf-8"))
    (run_dir / "execution_plan.sha256").write_text(plan_sha + "\n")
    wf.write_manifest_csv(run_dir / "tile_manifest.csv", rows)
    return plan, plan_sha, plan_path


def test_source_contract_rejects_unknown_collections(tmp_path: Path) -> None:
    import shutil

    path = tmp_path / "contract.json"
    shutil.copy("config/data_sources/naip_3dep_v1.json", path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["imagery"]["collection"] = "mapbox.satellite"
    path.write_text(json.dumps(contract, indent=2))
    with pytest.raises(wf.SourceContractError, match="not admissible|unsupported"):
        load_source_contract(path)


def test_execute_plan_rejects_repository_source_tree_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, plan_sha, plan_path = _plan_and_execute(
        tmp_path, run_name="source_tree_drift"
    )
    monkeypatch.setattr(wf, "repository_source_tree_digest", lambda root=None: "0" * 64)

    with pytest.raises(PlanDriftError, match="repository source-tree digest drift"):
        execute_plan(
            plan_path,
            expected_plan_sha256=plan_sha,
            allow_requester_pays=True,
            max_source_requests=200,
            max_transfer_bytes=100_000_000,
            max_local_bytes=100_000_000,
            max_requester_pays_usd=Decimal("500"),
            workers=1,
            transport_factory=make_transport_factory(build_fixture_objects()),
        )


def test_plan_digest_binds_plan_bytes(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path)
    assert load_and_verify_plan(plan_path, plan_sha) is not None
    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["coordinate_count"] += 1
    plan_path.write_text(json.dumps(tampered, sort_keys=True))
    with pytest.raises(PlanDriftError, match="SHA-256 mismatch"):
        load_and_verify_plan(plan_path, plan_sha)


def test_plan_recompute_drift_rejected(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path)
    plan["hashes"]["imagery_catalog_sha256"] = "0" * 64
    plan_path.write_text(wf.canonical_json_bytes(plan).decode("utf-8"))
    with pytest.raises(PlanDriftError):
        execute_plan(
            plan_path,
            expected_plan_sha256=plan_sha,
            allow_requester_pays=True,
            max_source_requests=200,
            max_transfer_bytes=100_000_000,
            max_local_bytes=100_000_000,
            max_requester_pays_usd=Decimal("500"),
            workers=1,
            transport_factory=make_transport_factory(build_fixture_objects()),
        )


def test_execute_requires_positive_caps(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        execute_plan(
            plan_path,
            expected_plan_sha256=plan_sha,
            allow_requester_pays=True,
            max_source_requests=0,
            max_transfer_bytes=100,
            max_local_bytes=100,
            max_requester_pays_usd=Decimal("1"),
            workers=1,
            transport_factory=make_transport_factory(build_fixture_objects()),
        )


def test_synthetic_two_tile_execution_end_to_end(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path, run_name="e2e")
    res = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=2,
        transport_factory=make_transport_factory(build_fixture_objects()),
    )
    assert res["state"] == "acquired"
    assert res["completed_coordinates"] == 2
    assert res["failed_coordinates"] == 0

    image_root = tmp_path / "images"
    for x, y in TILE_COORDS:
        sat = image_root / "satellite" / "z14" / "sne_pilot" / f"{x}_{y}.png"
        dem = image_root / "terrain" / "z14" / "sne_pilot" / f"{x}_{y}.png"
        assert sat.is_file() and dem.is_file()
        with Image.open(sat) as img:
            assert img.size == (512, 512)
            assert img.mode == "RGB"


def test_single_vintage_asset_selection(tmp_path: Path) -> None:
    planned = fake_planned_dict()
    assets = [
        naip_asset(
            "ct",
            2021,
            "m_4107201_ne_18_060-20210816",
            CT_BBOX,
            capture_date="2021-08-16",
        ),
        naip_asset(
            "ct",
            2019,
            "m_4107201_ne_18_060-20190601",
            CT_BBOX,
            capture_date="2019-06-01",
        ),
    ]
    catalog = make_naip_catalog(assets)
    records = [tnm_record(CT_BBOX)]
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=catalog, terrain_records=records
    )
    contract, _ = load_source_contract(contract_path)
    imagery_catalog = wf.load_naip_catalog(
        contract["imagery"]["catalog_snapshot"], None
    )
    terrain_assets = wf.load_3dep_catalog(contract["terrain"]["catalog_snapshot"], None)
    result = build_tile_plans(
        planned=planned,
        contract=contract,
        imagery_catalog=imagery_catalog,
        terrain_assets=terrain_assets,
        zoom=14,
    )
    key = list(result["tile_plans"].keys())[0]
    assert result["tile_plans"][key]["satellite_acquisition_year"] == 2021
    assert result["tile_plans"][key]["state"] == "ct"


def test_pilot_frame_deterministic_and_bounded() -> None:
    coords = [(x, y) for x in range(10, 40) for y in range(60, 90)]
    coord_to_region = {c: "sne_pilot" if c[0] < 25 else "nen" for c in coords}
    pilot_coords1, digest1 = select_pilot_frame(coords, coord_to_region, 50)
    pilot_coords2, digest2 = select_pilot_frame(coords, coord_to_region, 50)
    assert (pilot_coords1, digest1) == (pilot_coords2, digest2)
    assert len(pilot_coords1) == 50
    with pytest.raises(ValueError, match="between 1 and 500"):
        select_pilot_frame(coords, coord_to_region, 501)


def test_resume_skips_valid_and_repairs_corrupt(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path, run_name="resume")
    objects = build_fixture_objects()
    execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )

    image_root = tmp_path / "images"
    x0, y0 = TILE_COORDS[0]
    corrupt_png = image_root / "satellite" / "z14" / "sne_pilot" / f"{x0}_{y0}.png"
    corrupt_png.write_bytes(b"not-a-png")

    res = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )
    assert res["state"] == "acquired"
    with Image.open(corrupt_png) as img:
        assert img.size == (512, 512)
    quarantined = list(corrupt_png.parent.glob(f"{corrupt_png.name}.quarantine.*"))
    assert len(quarantined) == 1


def test_resume_repairs_interrupted_pair_with_installing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, plan_sha, plan_path = _plan_and_execute(
        tmp_path, run_name="resume_interrupted"
    )
    objects = build_fixture_objects()
    image_root = tmp_path / "images"
    run_dir = tmp_path / "runs" / "resume_interrupted"
    x0, y0 = TILE_COORDS[0]

    sat_dest = image_root / "satellite" / "z14" / "sne_pilot" / f"{x0}_{y0}.png"
    dem_dest = image_root / "terrain" / "z14" / "sne_pilot" / f"{x0}_{y0}.png"
    sat_dest.parent.mkdir(parents=True, exist_ok=True)
    dem_dest.parent.mkdir(parents=True, exist_ok=True)
    sat_dest.write_bytes(b"corrupt-sat-bytes")
    dem_dest.write_bytes(b"corrupt-dem-bytes")

    record_path = run_dir / "tile_records" / f"sne_pilot_z14_{x0}_{y0}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "state": "installing",
        "run_id": "resume_interrupted",
        "execution_plan_sha256": plan_sha,
        "satellite": {"path": str(sat_dest)},
        "terrain": {"path": str(dem_dest)},
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")

    res = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )
    assert res["state"] == "acquired"
    with Image.open(sat_dest) as img:
        assert img.size == (512, 512)
    assert list(sat_dest.parent.glob(f"{sat_dest.name}.quarantine.*"))
    assert list(dem_dest.parent.glob(f"{dem_dest.name}.quarantine.*"))


def test_resume_corrupt_tile_with_failed_reacquisition_remains_pending(
    tmp_path: Path,
) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(
        tmp_path, run_name="corrupt_resume_fail"
    )
    objects = build_fixture_objects()
    res1 = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )
    assert res1["state"] == "acquired"
    assert res1["completed_coordinates"] == len(TILE_COORDS)
    assert res1["pending_coordinates"] == 0

    run_dir = tmp_path / "runs" / "corrupt_resume_fail"
    checkpoint_before = json.loads(
        (run_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    bad_key = f"{TILE_COORDS[0][0]}_{TILE_COORDS[0][1]}"
    assert bad_key in checkpoint_before["completed_coords"]

    image_root = tmp_path / "images"
    x0, y0 = TILE_COORDS[0]
    corrupt_png = image_root / "satellite" / "z14" / "sne_pilot" / f"{x0}_{y0}.png"
    corrupt_png.write_bytes(b"corrupt-satellite-bytes")

    for cached_file in (tmp_path / "cache" / "objects").iterdir():
        cached_file.unlink()

    res2 = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory({}),
    )

    assert res2["state"] == "failed"
    assert res2["completed_coordinates"] == len(TILE_COORDS) - 1
    assert res2["pending_coordinates"] == 1
    assert res2["failed_coordinates"] == 1

    checkpoint_after = json.loads(
        (run_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert bad_key not in checkpoint_after["completed_coords"]
    assert checkpoint_after["state"] == "failed"
    assert len(checkpoint_after["completed_coords"]) == len(TILE_COORDS) - 1


def test_generated_outputs_are_reserved_before_write(tmp_path: Path) -> None:
    _plan, plan_sha, plan_path = _plan_and_execute(tmp_path, run_name="generated_cap")
    objects = build_fixture_objects()

    res = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )
    assert res["state"] == "aborted_cap"
    assert res["abort_reason"] == "cap"

    image_root = tmp_path / "images"
    assert not list(image_root.glob("**/*.png"))


def test_foreign_owner_output_never_overwritten(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path, run_name="foreign")
    objects = build_fixture_objects()

    execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )

    run_dir = tmp_path / "runs" / "foreign"
    x0, y0 = TILE_COORDS[0]
    record_path = run_dir / "tile_records" / f"sne_pilot_z14_{x0}_{y0}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["run_id"] = "other_run"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    res = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )
    assert res["state"] == "failed"
    failures = [
        json.loads(line)
        for line in (run_dir / "failures.jsonl").read_text().splitlines()
    ]
    assert failures[-1]["reason"] == "foreign_owner_refused"


def test_cap_abort_leaves_truthful_state(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path, run_name="cap")
    objects = build_fixture_objects()
    res = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=1,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=1,
        transport_factory=make_transport_factory(objects),
    )
    assert res["state"] == "aborted_cap"
    assert res["abort_reason"] == "cap"

    failures = [
        json.loads(line)
        for line in (tmp_path / "runs" / "cap" / "failures.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(item["category"] == "cap" for item in failures)


def test_discovery_authorization_is_network_free(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="disc",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="s" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="uv run python scripts/ingest/plan_active_learning_region.py --run-name disc",
        repository_source_tree_digest="9" * 64,
    )
    assert auth["identities"]["repository_source_tree_digest"] == "9" * 64
    assert auth["policy"]["network_call_made"] is False
    assert auth["policy"]["requester_pays"] is True
    assert auth["required_authorization"]["currency"] == "USD"
    assert isinstance(auth["required_authorization"]["max_requests"], int)
    assert auth["required_authorization"]["max_requests"] > 0
    assert isinstance(auth["required_authorization"]["max_transfer_bytes"], int)
    assert auth["required_authorization"]["max_transfer_bytes"] > 0
    assert isinstance(auth["required_authorization"]["max_local_bytes"], int)
    assert auth["required_authorization"]["max_local_bytes"] > 0
    assert isinstance(auth["required_authorization"]["max_spend_usd"], str)
    assert auth["required_authorization"]["allow_requester_pays"] is True
    assert auth["has_secrets"] is False
    assert len(auth["proposed_operations"]) == 2
    expected_terrain_requests = math.ceil(
        contract["terrain"]["catalog_max_objects"]
        / contract["terrain"]["catalog_page_size"]
    )
    for op in auth["proposed_operations"]:
        assert op["operation"] in ("s3:ListBucket", "s3:GetObject")
        if op["operation"] == "s3:GetObject":
            assert op["requests"] == 1
        elif op["operation"] == "s3:ListBucket":
            assert op["requests"] == expected_terrain_requests
        assert op["reserved_bytes"] > 0


def test_acquisition_authorization_null_caps_and_operations(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path, run_name="authz")
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    contract, _ = load_source_contract(
        write_contract_with_catalogs(
            tmp_path, imagery_catalog=None, terrain_records=None
        )
    )
    auth = wf.build_acquisition_authorization_request(
        run_name="authz", plan=plan_data, plan_sha=plan_sha, contract=contract
    )
    assert auth["authorization_type"] == "acquisition_authorization_request"
    assert auth["policy"]["requester_pays"] is False
    assert auth["required_authorization"]["currency"] == "USD"
    assert auth["required_authorization"]["max_requests"] is None
    assert auth["execution_plan_sha256"] == plan_sha
    ops = auth["proposed_operations"]
    assert any(op.get("operation") == "s3:GetObject" for op in ops)
    assert any(op.get("operation") == "local_raster_processing" for op in ops)


def test_discovery_authorization_exact_operations_and_caps(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="disc_exact",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="uv run python scripts/ingest/plan_active_learning_region.py --run-name disc_exact",
    )
    assert auth["authorization_type"] == "discovery_authorization_request"
    assert auth["missing_catalogs"] == ["imagery", "terrain"]
    ops = auth["proposed_operations"]
    expected_terrain_requests = math.ceil(
        contract["terrain"]["catalog_max_objects"]
        / contract["terrain"]["catalog_page_size"]
    )
    for op in ops:
        assert "provider" in op
        assert "collection" in op
        assert "bucket" in op
        assert op["operation"] in ("s3:ListBucket", "s3:GetObject")
        assert "target" in op
        if op["operation"] == "s3:GetObject":
            assert op["requests"] == 1
        elif op["operation"] == "s3:ListBucket":
            assert op["requests"] == expected_terrain_requests
        assert isinstance(op["reserved_bytes"], int) and op["reserved_bytes"] > 0
        assert isinstance(op["requester_pays"], bool)


def test_discovery_authorization_malformed_contract_fails_closed(
    tmp_path: Path,
) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()

    bad_contract = json.loads(json.dumps(contract))
    bad_contract["imagery"]["estimates"]["per_asset_requests"] = -1
    with pytest.raises(wf.SourceContractError):
        wf.build_discovery_authorization_request(
            run_name="disc_bad",
            contract=bad_contract,
            contract_sha=contract_sha,
            region_spec_sha="a" * 64,
            geometry_digest=planned["geometry_digest"],
            coordinate_count=len(TILE_COORDS),
            jurisdictions=["Connecticut"],
            missing=["imagery"],
            target_bbox=None,
            resume_command="resume",
        )

    bad_contract2 = json.loads(json.dumps(contract))
    bad_contract2["rate_card"]["request_cost_usd"] = "invalid_cost"
    with pytest.raises(wf.SourceContractError):
        wf.build_discovery_authorization_request(
            run_name="disc_bad2",
            contract=bad_contract2,
            contract_sha=contract_sha,
            region_spec_sha="a" * 64,
            geometry_digest=planned["geometry_digest"],
            coordinate_count=len(TILE_COORDS),
            jurisdictions=["Connecticut"],
            missing=["imagery"],
            target_bbox=None,
            resume_command="resume",
        )


def test_discovery_request_hash_mismatch_fails_before_network(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="disc_mismatch",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    auth_path = tmp_path / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    class TrapFactory:
        def __call__(self, *args, **kwargs):
            raise AssertionError(
                "network transport factory must not be called when hash mismatches"
            )

    with pytest.raises(wf.SourceContractError) as excinfo:
        wf.discover_raw_catalogs(
            discovery_request_path=auth_path,
            expected_discovery_request_sha256="0" * 64,
            contract_path=contract_path,
            missing=["imagery", "terrain"],
            caps=caps,
            allow_requester_pays=True,
            run_dir=tmp_path / "run",
            transport_factory=TrapFactory(),
        )
    assert "digest mismatch before network call" in str(excinfo.value).lower()


def test_repository_source_digest_fails_closed_without_git_tree(
    tmp_path: Path,
) -> None:
    with pytest.raises(wf.SourceContractError, match="could not be enumerated by Git"):
        wf.repository_source_tree_digest(tmp_path)


def test_discovery_request_source_tree_drift_fails_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    auth = wf.build_discovery_authorization_request(
        run_name="source_drift",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=fake_planned_dict()["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "source_drift"
    run_dir.mkdir(parents=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)
    monkeypatch.setattr(wf, "repository_source_tree_digest", lambda root=None: "0" * 64)

    class TrapFactory:
        def __call__(self, *args, **kwargs):
            raise AssertionError("network transport must not be constructed")

    with pytest.raises(wf.SourceContractError, match="source-tree identity"):
        wf.discover_raw_catalogs(
            discovery_request_path=auth_path,
            expected_discovery_request_sha256=auth["authorization_digest"],
            contract_path=contract_path,
            missing=["imagery", "terrain"],
            caps=wf.NetworkCaps(
                max_requests=1000,
                max_transfer_bytes=1_000_000_000,
                max_local_bytes=1_000_000_000,
                max_requester_pays_usd=Decimal("50"),
                allow_requester_pays=True,
            ),
            allow_requester_pays=True,
            run_dir=run_dir,
            transport_factory=TrapFactory(),
        )


def test_discover_raw_catalogs_emits_metadata_authorization_and_no_strict_catalog(
    tmp_path: Path,
) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="raw_disc",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "raw_disc"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)
    auth_sha = auth["authorization_digest"]

    objects = build_fixture_objects()
    transport_factory = make_transport_factory(objects)
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth_sha,
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=transport_factory,
    )

    assert res["state"] == "needs_metadata_discovery"
    meta_path = Path(res["metadata_authorization_path"])
    assert meta_path.is_file()

    assert (run_dir / "raw_naip_manifest.txt").is_file()
    assert (run_dir / "raw_3dep_listing.json").is_file()

    assert not Path(contract["imagery"]["catalog_snapshot"]).is_file()
    assert not Path(contract["terrain"]["catalog_snapshot"]).is_file()

    contract_after, _ = load_source_contract(contract_path)
    assert contract_after["imagery"].get("catalog_sha256") is None
    assert contract_after["terrain"].get("catalog_sha256") is None


def test_discover_catalog_metadata_resumes_planner(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="meta_disc",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "meta_disc"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)
    auth_sha = auth["authorization_digest"]

    objects = build_fixture_objects()
    transport_factory = make_transport_factory(objects)
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth_sha,
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=transport_factory,
    )

    meta_req_path = raw_res["metadata_authorization_path"]
    meta_req_sha = raw_res["metadata_authorization_sha256"]

    meta_hashes = wf.discover_catalog_metadata(
        metadata_request_path=meta_req_path,
        expected_metadata_request_sha256=meta_req_sha,
        contract_path=contract_path,
        run_dir=run_dir,
        caps=caps,
        allow_requester_pays=True,
        transport_factory=transport_factory,
    )

    assert "imagery_catalog_sha256" in meta_hashes
    assert "terrain_catalog_sha256" in meta_hashes

    snap_naip = Path(contract["imagery"]["catalog_snapshot"])
    snap_tnm = Path(contract["terrain"]["catalog_snapshot"])
    assert snap_naip.is_file() and snap_tnm.is_file()

    assert file_sha256(snap_naip) == meta_hashes["imagery_catalog_sha256"]
    assert file_sha256(snap_tnm) == meta_hashes["terrain_catalog_sha256"]


def test_discover_raw_catalogs_multi_page_success(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    contract["terrain"]["catalog_page_size"] = 1
    contract["terrain"]["catalog_max_objects"] = 10
    contract_path.write_text(json.dumps(contract, indent=2))
    contract, contract_sha = load_source_contract(contract_path)

    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="multi_page",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "multi_page"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)
    auth_sha = auth["authorization_digest"]

    objects = build_fixture_objects()
    dep2 = build_3dep_geotiff_bytes(
        bounds=(-73.0, 41.0, -73.0 + 8 * (1.0 / 10800.0), 41.0 + 8 * (1.0 / 10800.0))
    )
    key2 = "s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n42w073/USGS_13_n42w073.tif"
    clean2 = "StagedProducts/Elevation/13/TIFF/current/n42w073/USGS_13_n42w073.tif"
    objects[key2] = dep2
    objects[clean2] = dep2

    tf = make_transport_factory(objects)
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth_sha,
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )
    assert res["state"] == "needs_metadata_discovery"
    listing = json.loads((run_dir / "raw_3dep_listing.json").read_text())
    assert len(listing) >= 2


def test_discover_raw_catalogs_page_and_object_bound_failure(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    contract["terrain"]["catalog_max_objects"] = 1
    contract["terrain"]["catalog_page_size"] = 1
    contract_path.write_text(json.dumps(contract, indent=2))
    contract, contract_sha = load_source_contract(contract_path)

    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="bound_fail",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "bound_fail"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)
    auth_sha = auth["authorization_digest"]

    objects = build_fixture_objects()
    for i in range(5):
        k = f"s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n42w07{i}/USGS_13_n42w07{i}.tif"
        objects[k] = build_3dep_geotiff_bytes()

    tf = make_transport_factory(objects)
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    with pytest.raises(
        wf.SourceContractError, match="catalog max_objects|catalog page bound"
    ):
        wf.discover_raw_catalogs(
            discovery_request_path=auth_path,
            expected_discovery_request_sha256=auth_sha,
            contract_path=contract_path,
            missing=["imagery", "terrain"],
            caps=caps,
            allow_requester_pays=True,
            run_dir=run_dir,
            transport_factory=tf,
        )


def test_discovery_and_metadata_tampered_digest_rejected_before_transport(
    tmp_path: Path,
) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="tampered",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "tampered"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    class TrapFactory:
        def __call__(self, *args, **kwargs):
            raise AssertionError(
                "Transport must not be constructed when digest mismatches"
            )

    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    with pytest.raises(
        wf.SourceContractError, match="digest mismatch before network call"
    ):
        wf.discover_raw_catalogs(
            discovery_request_path=auth_path,
            expected_discovery_request_sha256="0" * 64,
            contract_path=contract_path,
            missing=["imagery", "terrain"],
            caps=caps,
            allow_requester_pays=True,
            run_dir=run_dir,
            transport_factory=TrapFactory(),
        )

    objects = build_fixture_objects()
    tf = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )
    meta_req_path = raw_res["metadata_authorization_path"]

    with pytest.raises(
        wf.SourceContractError, match="digest mismatch before network call"
    ):
        wf.discover_catalog_metadata(
            metadata_request_path=meta_req_path,
            expected_metadata_request_sha256="0" * 64,
            contract_path=contract_path,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=True,
            transport_factory=TrapFactory(),
        )


def test_undersized_caps_fails_closed(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="low_caps",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "low_caps"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    low_caps = wf.NetworkCaps(
        max_requests=1,
        max_transfer_bytes=100,
        max_local_bytes=100,
        max_requester_pays_usd=Decimal("0.01"),
        allow_requester_pays=True,
    )

    objects = build_fixture_objects()
    tf = make_transport_factory(objects)

    with pytest.raises(wf.SourceContractError, match="network caps are below"):
        wf.discover_raw_catalogs(
            discovery_request_path=auth_path,
            expected_discovery_request_sha256=auth["authorization_digest"],
            contract_path=contract_path,
            missing=["imagery", "terrain"],
            caps=low_caps,
            allow_requester_pays=True,
            run_dir=run_dir,
            transport_factory=tf,
        )


def test_metadata_discovery_missing_etag_accept_ranges_date(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="missing_meta",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "missing_meta"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    objects = build_fixture_objects()
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    tf = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )
    meta_req_path = raw_res["metadata_authorization_path"]
    meta_req_sha = raw_res["metadata_authorization_sha256"]

    tf_no_etag = make_transport_factory(objects, etags={TNM_DEM_KEY: ""})
    with pytest.raises(wf.SourceContractError, match="returned no ETag"):
        wf.discover_catalog_metadata(
            metadata_request_path=meta_req_path,
            expected_metadata_request_sha256=meta_req_sha,
            contract_path=contract_path,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=True,
            transport_factory=tf_no_etag,
        )

    tf_no_ranges = make_transport_factory(objects, accept_ranges={NAIP_SAT_KEY: False})
    with pytest.raises(wf.SourceContractError, match="Accept-Ranges"):
        wf.discover_catalog_metadata(
            metadata_request_path=meta_req_path,
            expected_metadata_request_sha256=meta_req_sha,
            contract_path=contract_path,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=True,
            transport_factory=tf_no_ranges,
        )


def test_metadata_discovery_malformed_tiff(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="malformed_tiff",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "malformed_tiff"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    objects = build_fixture_objects()
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    tf = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )

    bad_objects = dict(objects)
    bad_objects[NAIP_SAT_KEY] = b"not-a-geotiff-file"
    clean_sat = NAIP_SAT_KEY.replace("s3://naip-visualization/", "")
    bad_objects[clean_sat] = b"not-a-geotiff-file"
    bad_tf = make_transport_factory(bad_objects)

    with pytest.raises(wf.SourceContractError, match="not a parseable GeoTIFF"):
        wf.discover_catalog_metadata(
            metadata_request_path=raw_res["metadata_authorization_path"],
            expected_metadata_request_sha256=raw_res["metadata_authorization_sha256"],
            contract_path=contract_path,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=True,
            transport_factory=bad_tf,
        )


def test_metadata_discovery_missing_vertical_metadata(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="missing_vert",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "missing_vert"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    objects = build_fixture_objects()
    objects.pop(TNM_XML_KEY, None)
    clean_xml = TNM_XML_KEY.replace("s3://prd-tnm/", "")
    objects.pop(clean_xml, None)

    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    tf = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )

    with pytest.raises(
        wf.SourceContractError, match="vertical datum/units were not observed"
    ):
        wf.discover_catalog_metadata(
            metadata_request_path=raw_res["metadata_authorization_path"],
            expected_metadata_request_sha256=raw_res["metadata_authorization_sha256"],
            contract_path=contract_path,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=True,
            transport_factory=tf,
        )


def test_metadata_discovery_exact_full_byte_and_etag_binding(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="byte_binding",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "byte_binding"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    objects = build_fixture_objects()
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    tf = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )

    bad_objects = dict(objects)
    bad_objects[NAIP_SAT_KEY] = objects[NAIP_SAT_KEY] + b"\x00"
    clean_sat = NAIP_SAT_KEY.replace("s3://naip-visualization/", "")
    bad_objects[clean_sat] = objects[NAIP_SAT_KEY] + b"\x00"

    orig_etag = f'"{hashlib.sha256(objects[NAIP_SAT_KEY]).hexdigest()[:16]}"'
    tf_mismatch = make_transport_factory(
        bad_objects,
        etags={NAIP_SAT_KEY: orig_etag, clean_sat: orig_etag},
        head_sizes={
            NAIP_SAT_KEY: len(objects[NAIP_SAT_KEY]),
            clean_sat: len(objects[NAIP_SAT_KEY]),
        },
    )

    with pytest.raises(wf.SourceContractError, match="GET body length"):
        wf.discover_catalog_metadata(
            metadata_request_path=raw_res["metadata_authorization_path"],
            expected_metadata_request_sha256=raw_res["metadata_authorization_sha256"],
            contract_path=contract_path,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=True,
            transport_factory=tf_mismatch,
        )


def test_discover_catalog_metadata_emits_resolved_contract_and_leaves_source_contract_unmutated(
    tmp_path: Path,
) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="resolved_contract",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "resolved_contract"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    objects = build_fixture_objects()
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    tf = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )

    meta_hashes = wf.discover_catalog_metadata(
        metadata_request_path=raw_res["metadata_authorization_path"],
        expected_metadata_request_sha256=raw_res["metadata_authorization_sha256"],
        contract_path=contract_path,
        run_dir=run_dir,
        caps=caps,
        allow_requester_pays=True,
        transport_factory=tf,
    )

    resolved_path = Path(meta_hashes["resolved_contract_path"])
    assert resolved_path.is_file()
    resolved_data = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert (
        resolved_data["imagery"]["catalog_sha256"]
        == meta_hashes["imagery_catalog_sha256"]
    )
    assert (
        resolved_data["terrain"]["catalog_sha256"]
        == meta_hashes["terrain_catalog_sha256"]
    )
    assert resolved_data["resolved_from_source_contract_sha256"] == contract_sha

    contract_after, contract_after_sha = load_source_contract(contract_path)
    assert contract_after_sha == contract_sha
    assert contract_after["imagery"].get("catalog_sha256") is None
    assert contract_after["terrain"].get("catalog_sha256") is None


def test_discover_catalog_metadata_strict_loaders(tmp_path: Path) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="strict_loaders",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "strict_loaders"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    objects = build_fixture_objects()
    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    tf = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf,
    )

    meta_hashes = wf.discover_catalog_metadata(
        metadata_request_path=raw_res["metadata_authorization_path"],
        expected_metadata_request_sha256=raw_res["metadata_authorization_sha256"],
        contract_path=contract_path,
        run_dir=run_dir,
        caps=caps,
        allow_requester_pays=True,
        transport_factory=tf,
    )

    snap_naip = Path(contract["imagery"]["catalog_snapshot"])
    snap_tnm = Path(contract["terrain"]["catalog_snapshot"])

    naip_cat = wf.load_naip_catalog(snap_naip, meta_hashes["imagery_catalog_sha256"])
    tnm_assets = wf.load_3dep_catalog(snap_tnm, meta_hashes["terrain_catalog_sha256"])

    assert naip_cat is not None
    assert tnm_assets is not None
    assert len(naip_cat["assets"]) >= 1
    assert len(tnm_assets) >= 1


def test_metadata_discovery_partial_failure_leaves_catalog_and_contract_unchanged(
    tmp_path: Path,
) -> None:
    contract_path = write_contract_with_catalogs(
        tmp_path, imagery_catalog=None, terrain_records=None
    )
    contract, contract_sha = load_source_contract(contract_path)
    planned = fake_planned_dict()
    auth = wf.build_discovery_authorization_request(
        run_name="partial_fail",
        contract=contract,
        contract_sha=contract_sha,
        region_spec_sha="a" * 64,
        geometry_digest=planned["geometry_digest"],
        coordinate_count=len(TILE_COORDS),
        jurisdictions=["Connecticut"],
        missing=["imagery", "terrain"],
        target_bbox=None,
        resume_command="resume",
    )
    run_dir = tmp_path / "runs" / "partial_fail"
    run_dir.mkdir(parents=True, exist_ok=True)
    auth_path = run_dir / "discovery_authorization_request.json"
    atomic_write_json(auth_path, auth)

    objects = build_fixture_objects()
    dep2 = build_3dep_geotiff_bytes(
        bounds=(-73.0, 41.0, -73.0 + 8 * (1.0 / 10800.0), 41.0 + 8 * (1.0 / 10800.0))
    )
    key2 = "s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/n42w073/USGS_13_n42w073.tif"
    clean2 = "StagedProducts/Elevation/13/TIFF/current/n42w073/USGS_13_n42w073.tif"
    objects[key2] = dep2
    objects[clean2] = dep2

    caps = wf.NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )

    tf_good = make_transport_factory(objects)
    raw_res = wf.discover_raw_catalogs(
        discovery_request_path=auth_path,
        expected_discovery_request_sha256=auth["authorization_digest"],
        contract_path=contract_path,
        missing=["imagery", "terrain"],
        caps=caps,
        allow_requester_pays=True,
        run_dir=run_dir,
        transport_factory=tf_good,
    )

    bad_objects = dict(objects)
    bad_objects[key2] = b"malformed-tiff"
    bad_objects[clean2] = b"malformed-tiff"
    tf_bad = make_transport_factory(bad_objects)

    snap_naip = Path(contract["imagery"]["catalog_snapshot"])
    snap_tnm = Path(contract["terrain"]["catalog_snapshot"])
    resolved_final = snap_naip.parent / f"{contract['contract_id']}.resolved.json"

    with pytest.raises(wf.SourceContractError):
        wf.discover_catalog_metadata(
            metadata_request_path=raw_res["metadata_authorization_path"],
            expected_metadata_request_sha256=raw_res["metadata_authorization_sha256"],
            contract_path=contract_path,
            run_dir=run_dir,
            caps=caps,
            allow_requester_pays=True,
            transport_factory=tf_bad,
        )

    assert not snap_naip.is_file()
    assert not snap_tnm.is_file()
    assert not resolved_final.is_file()

    contract_after, contract_after_sha = load_source_contract(contract_path)
    assert contract_after_sha == contract_sha
    assert contract_after["imagery"].get("catalog_sha256") is None
    assert contract_after["terrain"].get("catalog_sha256") is None


# ---------------------------------------------------------------------------
# Concurrent object acquisition (keyed single-flight)
# ---------------------------------------------------------------------------


class _CountingTransport:
    """MeteredTransport decorator that counts head/get attempts."""

    def __init__(self, inner, counts: dict) -> None:
        self._inner = inner
        self._counts = counts

    def head_object(self, *args, **kwargs):
        with self._counts["lock"]:
            self._counts["head_object"] += 1
        return self._inner.head_object(*args, **kwargs)

    def get_range(self, *args, **kwargs):
        with self._counts["lock"]:
            self._counts["get_range"] += 1
        return self._inner.get_range(*args, **kwargs)

    def get_object(self, *args, **kwargs):
        with self._counts["lock"]:
            self._counts["get_object"] += 1
        return self._inner.get_object(*args, **kwargs)

    def list_objects(self, *args, **kwargs):
        with self._counts["lock"]:
            self._counts["list_objects"] += 1
        return self._inner.list_objects(*args, **kwargs)


def _counting_transport_factory(objects: dict, counts: dict):
    base = make_transport_factory(objects)

    def factory(
        bucket, caps, ledger, rate_card, requester_pays, shared_budget, *args, **kwargs
    ):
        return _CountingTransport(
            base(
                bucket,
                caps,
                ledger,
                rate_card,
                requester_pays,
                shared_budget,
                *args,
                **kwargs,
            ),
            counts,
        )

    return factory


class _GatedTransport:
    """MeteredTransport whose get_range blocks until released (deterministic)."""

    def __init__(
        self,
        inner,
        entered: threading.Event,
        release: threading.Event,
        state: dict,
        guard: threading.Lock,
    ) -> None:
        self._inner = inner
        self._entered = entered
        self._release = release
        self._state = state
        self._guard = guard

    def head_object(self, *args, **kwargs):
        return self._inner.head_object(*args, **kwargs)

    def get_range(self, *args, **kwargs):
        with self._guard:
            self._state["depth"] += 1
            self._state["max_depth"] = max(
                self._state["max_depth"], self._state["depth"]
            )
        self._entered.set()
        try:
            self._release.wait()
            return self._inner.get_range(*args, **kwargs)
        finally:
            with self._guard:
                self._state["depth"] -= 1


def _object_ensure_env(tmp_path: Path, objects: dict, missing_keys=None):
    """Shared caps/ledger/budget/transport for direct _ensure_object tests."""
    caps = NetworkCaps(
        max_requests=1000,
        max_transfer_bytes=1_000_000_000,
        max_local_bytes=1_000_000_000,
        max_requester_pays_usd=Decimal("50.0"),
        allow_requester_pays=True,
    )
    ledger = MeteredLedger(tmp_path / "ledger.jsonl")
    budget = SharedMeteredBudget(caps)
    rate_card = RateCard(
        source="fixture",
        date="2026-01-01",
        request_cost_usd=Decimal("0.0004"),
        transfer_cost_per_gb_usd=Decimal("0.09"),
    )
    transport = make_transport_factory(objects, missing_keys=missing_keys)(
        "naip-visualization",
        caps,
        ledger,
        rate_card,
        True,
        budget,
    )
    return transport, ledger, budget


def _ensure_call(transport, cache_root: Path, ledger, flights, key, etag):
    return wf._ensure_object(
        transport=transport,
        cache_root=cache_root,
        bucket="naip-visualization",
        key=key,
        etag=etag,
        max_response_bytes=100_000_000,
        ledger=ledger,
        flights=flights,
    )


def test_concurrent_same_key_acquisition_is_single_flight(tmp_path: Path) -> None:
    objects = build_fixture_objects()
    content = objects[NAIP_SAT_KEY]
    etag = f'"{hashlib.sha256(content).hexdigest()[:16]}"'
    counts = {
        "lock": threading.Lock(),
        "get_range": 0,
        "head_object": 0,
        "get_object": 0,
        "list_objects": 0,
    }
    transport, ledger, budget = _object_ensure_env(tmp_path, objects)
    transport = _CountingTransport(transport, counts)
    cache_root = tmp_path / "cache"
    flights = wf._SingleFlight()

    def call():
        return _ensure_call(transport, cache_root, ledger, flights, NAIP_SAT_KEY, etag)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: call(), range(8)))

    paths = {path for path, _ in results}
    shas = {sha for _, sha in results}
    assert len(paths) == 1
    assert len(shas) == 1
    path, sha = results[0]
    assert path.is_file()
    assert file_sha256(path) == sha
    assert list((cache_root / "objects").glob("*.bin")) == [path]

    # One remote acquisition regardless of caller count.
    assert counts["get_range"] == 1
    assert counts["head_object"] == 0
    assert counts["get_object"] == 0
    counters = budget.counters()
    assert counters.requests == 1
    assert counters.transfer_bytes == len(content)

    # Every caller that did not perform the download recorded a cache hit.
    events = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert sum(1 for e in events if e["event"] == "cache_hit") == 7
    assert sum(1 for e in events if e["event"] == "object_meta") == 0


def test_concurrent_unknown_etag_dedups_head_and_get(tmp_path: Path) -> None:
    objects = build_fixture_objects()
    counts = {
        "lock": threading.Lock(),
        "get_range": 0,
        "head_object": 0,
        "get_object": 0,
        "list_objects": 0,
    }
    transport, ledger, _budget = _object_ensure_env(tmp_path, objects)
    transport = _CountingTransport(transport, counts)
    cache_root = tmp_path / "cache"
    flights = wf._SingleFlight()

    def call():
        return _ensure_call(transport, cache_root, ledger, flights, NAIP_SAT_KEY, None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: call(), range(4)))

    # Unknown ETag: the metadata HEAD and the GET are each deduped too.
    assert counts["head_object"] == 1
    assert counts["get_range"] == 1
    assert len({sha for _, sha in results}) == 1
    path, sha = results[0]
    assert path.is_file()
    assert file_sha256(path) == sha


def test_single_flight_same_key_never_parallel_acquisition(tmp_path: Path) -> None:
    objects = build_fixture_objects()
    content = objects[NAIP_SAT_KEY]
    etag = f'"{hashlib.sha256(content).hexdigest()[:16]}"'
    transport, ledger, _budget = _object_ensure_env(tmp_path, objects)
    entered = threading.Event()
    release = threading.Event()
    state = {"depth": 0, "max_depth": 0}
    guard = threading.Lock()
    transport = _GatedTransport(transport, entered, release, state, guard)
    cache_root = tmp_path / "cache"
    flights = wf._SingleFlight()

    def call():
        return _ensure_call(transport, cache_root, ledger, flights, NAIP_SAT_KEY, etag)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(call)
        assert entered.wait(5)
        f2 = pool.submit(call)
        # The follower cannot enter a second acquisition while the leader is
        # still blocked inside its download.
        time.sleep(0.2)
        assert state["depth"] == 1
        assert state["max_depth"] == 1
        release.set()
        r1 = f1.result()
        r2 = f2.result()

    assert r1 == r2
    assert r1[0].is_file()
    assert file_sha256(r1[0]) == r1[1]


def test_single_flight_different_keys_proceed_concurrently(tmp_path: Path) -> None:
    objects = build_fixture_objects()
    key1 = NAIP_SAT_KEY
    key2 = "s3://naip-visualization/ct/2019/100m/rgbir/37072/other.tif"
    content1 = objects[key1]
    content2 = b"distinct-bytes-for-key2"
    objects[key2] = content2
    etag1 = f'"{hashlib.sha256(content1).hexdigest()[:16]}"'
    etag2 = f'"{hashlib.sha256(content2).hexdigest()[:16]}"'

    transport, ledger, _budget = _object_ensure_env(tmp_path, objects)
    entered = threading.Event()
    release = threading.Event()
    state = {"depth": 0, "max_depth": 0}
    guard = threading.Lock()
    transport = _GatedTransport(transport, entered, release, state, guard)
    cache_root = tmp_path / "cache"
    flights = wf._SingleFlight()

    def call(key, etag):
        return _ensure_call(transport, cache_root, ledger, flights, key, etag)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(call, key1, etag1)
        assert entered.wait(5)
        f2 = pool.submit(call, key2, etag2)
        # Different keys are never serialized: the second acquisition enters
        # while the first is still blocked inside its download.
        deadline = time.monotonic() + 5
        while state["depth"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state["depth"] == 2
        assert state["max_depth"] == 2
        release.set()
        r1 = f1.result()
        r2 = f2.result()

    assert r1[1] != r2[1]
    assert r1[0].is_file() and r2[0].is_file()
    assert file_sha256(r1[0]) == r1[1]
    assert file_sha256(r2[0]) == r2[1]


def test_single_flight_failure_cleans_up_and_does_not_poison_retries(
    tmp_path: Path,
) -> None:
    objects = build_fixture_objects()
    content = objects[NAIP_SAT_KEY]
    etag = f'"{hashlib.sha256(content).hexdigest()[:16]}"'
    counts = {
        "lock": threading.Lock(),
        "get_range": 0,
        "head_object": 0,
        "get_object": 0,
        "list_objects": 0,
    }
    transport, ledger, _budget = _object_ensure_env(
        tmp_path, objects, missing_keys={NAIP_SAT_KEY}
    )
    transport = _CountingTransport(transport, counts)
    cache_root = tmp_path / "cache"
    flights = wf._SingleFlight()

    def call():
        return _ensure_call(transport, cache_root, ledger, flights, NAIP_SAT_KEY, etag)

    # A failed acquisition raises, installs nothing, and removes its
    # in-flight entry so later calls start fresh.
    with pytest.raises(NotFoundError):
        call()
    assert flights._inflight == {}
    assert not list((cache_root / "objects").glob("*"))
    assert counts["get_range"] == 1

    # A concurrent batch during the outage also surfaces the failure (the
    # follower re-raises the leader's error rather than retrying remotely)
    # and leaves no residue.
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(call)
        f2 = pool.submit(call)
        with pytest.raises(NotFoundError):
            f1.result()
        with pytest.raises(NotFoundError):
            f2.result()
    assert flights._inflight == {}
    assert not list((cache_root / "objects").glob("*"))
    attempts_during_outage = counts["get_range"]

    # Retries are not poisoned: once the object is available, a concurrent
    # retry succeeds with exactly one fresh acquisition for all callers.
    transport._inner.missing_keys.discard(NAIP_SAT_KEY)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: call(), range(2)))
    assert len({sha for _, sha in results}) == 1
    path, sha = results[0]
    assert path.is_file()
    assert file_sha256(path) == sha
    assert counts["get_range"] == attempts_during_outage + 1
    assert counts["head_object"] == 0


def test_execute_plan_acquires_each_distinct_asset_once(tmp_path: Path) -> None:
    plan, plan_sha, plan_path = _plan_and_execute(tmp_path, run_name="dedup")
    objects = build_fixture_objects()
    counts = {
        "lock": threading.Lock(),
        "get_range": 0,
        "head_object": 0,
        "get_object": 0,
        "list_objects": 0,
    }
    res = execute_plan(
        plan_path,
        expected_plan_sha256=plan_sha,
        allow_requester_pays=True,
        max_source_requests=200,
        max_transfer_bytes=100_000_000,
        max_local_bytes=100_000_000,
        max_requester_pays_usd=Decimal("500"),
        workers=2,
        transport_factory=_counting_transport_factory(objects, counts),
    )
    assert res["state"] == "acquired"
    assert res["completed_coordinates"] == 2

    # Both tiles share the same 2021 NAIP asset and the same terrain record;
    # concurrent workers must fetch each distinct object exactly once.
    assert counts["head_object"] == 0
    assert counts["get_range"] == 2
    assert res["counters"]["requests"] == 2

    cache_objects = list((tmp_path / "cache" / "objects").glob("*.bin"))
    assert len(cache_objects) == 2

    ledger_events = [
        json.loads(line)
        for line in (tmp_path / "runs" / "dedup" / "request_transfer_ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    assert sum(1 for e in ledger_events if e["event"] == "cache_hit") == 2
