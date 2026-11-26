from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.routing import check_beta_artifacts as release_checker

from scenicdriver.data_pipeline.web_mercator import lat_lon_to_tile
from scenicdriver.route_planner import service as route_service
from scenicdriver.route_planner.graph import (
    Edge,
    Node,
    RoadGraph,
    _compact_score_sidecar_identity,
    write_compact_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_COMMAND = "uv run python scripts/deploy/bootstrap_beta_artifacts.py"


@pytest.fixture
def app_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from scenicdriver.app_api.contrib_repo import ContribConfig, ContribRepo

    # Importing the exported app must not create a repository outside the fixture.
    repo = ContribRepo(ContribConfig(root_dir=tmp_path / "contrib"))
    with patch("scenicdriver.app_api.contrib_repo.ContribRepo", return_value=repo):
        app_api = importlib.import_module("scenicdriver.app_api.main")
    app_api.ContribRepo = ContribRepo
    monkeypatch.setattr(app_api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_api, "APP_REGIONS_PATH", tmp_path / "app_regions.json")
    yield app_api
    route_service.clear_route_caches()


def _region_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = json.loads((REPO_ROOT / "config/app_regions.json").read_text())
    region = next(row for row in config["regions"] if row["region"] == config["default_region"])
    manifest = tmp_path / region["graph"]
    manifest.parent.mkdir(parents=True)
    sqlite = manifest.with_name("road_graph.sqlite3")
    graph = RoadGraph()
    for node in (
        Node("start", 42.0, -72.0),
        Node("fast", 42.05, -72.0),
        Node("scenic", 42.0, -71.95),
        Node("end", 42.05, -71.95),
    ):
        graph.add_node(node)
    for edge in (
        Edge("fast-0", "start", "fast", distance_km=5.56, scenic_score=5.0,
             road_type="trunk", speed_limit_kmh=90),
        Edge("fast-1", "fast", "end", distance_km=4.13, scenic_score=5.0,
             road_type="trunk", speed_limit_kmh=90),
        Edge("scenic-0", "start", "scenic", distance_km=4.13, scenic_score=5.0,
             road_type="secondary", speed_limit_kmh=60),
        Edge("scenic-1", "scenic", "end", distance_km=5.56, scenic_score=5.0,
             road_type="secondary", speed_limit_kmh=60),
    ):
        graph.add_edge(edge)
    graph.save(sqlite)
    record = write_compact_graph(sqlite, manifest)
    (manifest.parent / "run.json").write_text(json.dumps({
        "output_graph": str(sqlite.relative_to(tmp_path)),
        "graph_sha256": record["source_sha256"],
        "graph_size_bytes": record["source_size_bytes"],
        "runtime_graph": str(manifest.relative_to(tmp_path)),
        "compact_manifest_sha256": record["manifest_sha256"],
    }))
    report = tmp_path / "report.json"
    tiles = []
    for edge in graph.edges.values():
        start = graph.get_node(edge.start_node_id)
        end = graph.get_node(edge.end_node_id)
        x, y = lat_lon_to_tile((start.lat + end.lat) / 2, (start.lon + end.lon) / 2, 14)
        tiles.append({"z": 14, "x": x, "y": y,
                      "scenic_score": 9.0 if edge.id.startswith("scenic") else 2.0})
    assert len({(tile["x"], tile["y"]) for tile in tiles}) == 4
    report.write_text(json.dumps({"tiles": tiles}))
    config["regions"] = [region]
    (tmp_path / "app_regions.json").write_text(json.dumps(config))
    return sqlite, manifest, report


def test_default_region_uses_deployed_compact_pair_and_content_sidecars(
    app_config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_config, "PROJECT_ROOT", REPO_ROOT)
    monkeypatch.setattr(app_config, "APP_REGIONS_PATH", REPO_ROOT / "config/app_regions.json")
    rows = {row["region"]: row for row in app_config._configured_regions()}
    default = rows[app_config._default_region_key()]
    artifacts = {row["name"]: row for row in json.loads(
        (REPO_ROOT / "deploy/beta_artifacts.json").read_text()
    )["artifacts"]}
    assert default["region"] == "new_england_north"
    assert default["graph"] == artifacts["graph.compact.manifest"]["destination"]
    assert default["graph_path"] == REPO_ROOT / default["graph"]
    assert Path(default["graph"]).with_suffix(".bin").as_posix() == artifacts[
        "graph.compact.payload"
    ]["destination"]
    identity = _compact_score_sidecar_identity(
        report_sha256=artifacts["report"]["sha256"],
        graph_sha256=artifacts["graph"]["sha256"],
        zoom=14, fallback=None, normalization=route_service._NORMALIZATION_VERSION,
    )
    for suffix, name in (("json", "manifest"), ("bin", "payload")):
        expected = Path(default["graph"]).with_name(
            f"road_graph.score.{identity}.z14.fbnull.nlinear-v1.{suffix}"
        )
        assert expected.as_posix() == artifacts[f"graph.compact.score.{name}"]["destination"]
    expanded = rows["northeast_expanded"]
    assert expanded["graph"] == artifacts["expanded_graph.compact.manifest"]["destination"]
    assert expanded["run_name"] == "prompt_two_candidate_exp02_expanded_20260810"
    assert expanded["route_deadline_seconds"] == 120


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("avoid_highways", [False, True])
def test_region_compact_admission_preserves_sqlite_planning(
    tmp_path: Path, app_config, reverse: bool, avoid_highways: bool,
) -> None:
    sqlite, manifest, report = _region_fixture(tmp_path)
    start, end = (42.0, -72.0), (42.05, -71.95)
    if reverse:
        start, end = end, start
    results = []
    for graph_path in (sqlite, manifest):
        config_path = tmp_path / "app_regions.json"
        config = json.loads(config_path.read_text())
        config["regions"][0]["graph"] = str(graph_path.relative_to(tmp_path))
        config_path.write_text(json.dumps(config))
        admitted = app_config._region_to_graph(app_config._default_region_key())
        assert admitted == graph_path
        preloaded = route_service.preload_route_assets(admitted, tile_scores_path=report)
        assert preloaded["score_mapping"]["matched_edges"] == 4
        result = route_service.plan_routes(route_service.RouteRequest(
            graph_geojson=str(admitted), start=start, end=end,
            avoid_highways=avoid_highways, max_detour_factor=2.0,
            tile_scores_json=str(report), include_baseline=True,
        ))
        results.append(result)
        route_service.clear_route_caches()
    eager, compact = results
    assert [row["route_kind"] for row in compact["routes"]] == ["scenic", "baseline"]
    for eager_route, compact_route in zip(eager["routes"], compact["routes"], strict=True):
        eager_metrics, compact_metrics = eager_route["metrics"], compact_route["metrics"]
        for key in ("edge_ids", "traversal_ids", "segment_identity", "segments",
                    "requested_start", "requested_end", "snapped_start", "snapped_end",
                    "exactness_status", "same_route", "highway_count"):
            assert compact_metrics[key] == eager_metrics[key]
        for key in ("total_distance_km", "average_scenic_score", "estimated_duration_minutes",
                    "actual_duration_ratio", "objective_value", "normalized_scenic_score"):
            assert compact_metrics[key] == pytest.approx(eager_metrics[key])
    for eager_feature, compact_feature in zip(
        eager["geojson"]["features"], compact["geojson"]["features"], strict=True,
    ):
        assert compact_feature["geometry"] == eager_feature["geometry"]
    expected_scenic = ["scenic-1", "scenic-0"] if reverse else ["scenic-0", "scenic-1"]
    expected_fastest = ["fast-1", "fast-0"] if reverse else ["fast-0", "fast-1"]
    assert compact["routes"][0]["metrics"]["edge_ids"] == expected_scenic
    assert compact["routes"][1]["metrics"]["edge_ids"] == expected_fastest
    assert compact["routes"][0]["metrics"]["same_route"] is False


@pytest.mark.parametrize("missing", ["manifest", "payload"])
@pytest.mark.parametrize("entrypoint", ["preload", "plan"])
def test_missing_configured_compact_artifact_has_bootstrap_action(
    tmp_path: Path, app_config, missing: str, entrypoint: str,
) -> None:
    sqlite, manifest, report = _region_fixture(tmp_path)
    configured = app_config._app_region(app_config._default_region_key())["graph_path"]
    assert configured == manifest
    payload = manifest.parent / json.loads(manifest.read_text())["bin_path"]
    missing_path = manifest if missing == "manifest" else payload
    missing_path.unlink()
    with pytest.raises((FileNotFoundError, ValueError)) as error:
        if entrypoint == "preload":
            route_service.preload_route_assets(configured, tile_scores_path=report)
        else:
            route_service.plan_routes(route_service.RouteRequest(
                graph_geojson=str(configured), start=(42.0, -72.0), end=(42.05, -71.95),
                tile_scores_json=str(report),
            ))
    assert str(missing_path) in str(error.value)
    assert BOOTSTRAP_COMMAND in str(error.value)
    assert sqlite.is_file()  # The audit source is not a runtime fallback.


@pytest.mark.parametrize(
    "config_text, expected",
    [
        (None, "canonical region config is required"),
        ("{", "invalid:"),
        ("[]", "config payload is not a dict"),
        ('{"regions": []}', "default_region is missing"),
        ('{"default_region": " ", "regions": []}', "default_region is missing"),
        ('{"default_region": "candidate"}', "regions array is missing"),
        ('{"default_region": "candidate", "regions": []}', "candidate is not configured"),
        (json.dumps({"default_region": "candidate", "regions": [
            {"region": "candidate", "run_name": "changed-run"},
        ]}), "graph is missing"),
        (json.dumps({"default_region": "candidate", "regions": [
            {"region": "candidate", "graph": "changed.sqlite3"},
        ]}), "run_name is missing"),
        (json.dumps({"default_region": "candidate", "regions": [
            {"region": "candidate", "graph": "changed.sqlite3", "run_name": " "},
        ]}), "run_name is missing"),
    ],
)
@pytest.mark.parametrize("cli_graph", [None, Path("override.sqlite3")])
def test_release_checker_never_synthesizes_region_defaults(
    tmp_path: Path, capsys, config_text: str | None, expected: str,
    cli_graph: Path | None,
) -> None:
    if config_text is not None:
        config_path = tmp_path / release_checker.CONFIG_PATH
        config_path.parent.mkdir()
        config_path.write_text(config_text)
    regions, issues = release_checker._configured_route_regions(tmp_path, cli_graph)
    assert regions == []
    assert any(expected in issue for issue in issues)
    assert release_checker.check_artifacts(tmp_path, graph=cli_graph) == 1
    output = capsys.readouterr().out
    assert expected in output
    assert "new_england_north" not in output
    assert "prompt_two_candidate" not in output


@pytest.mark.parametrize(
    "lat, lon, run_name",
    [(35.0, -110.0, "desert-release"), (46.0, -120.0, "mountain-release")],
)
def test_release_checker_uses_configured_geometry_and_run(
    tmp_path: Path, lat: float, lon: float, run_name: str,
) -> None:
    bbox = {"min_lat": lat - .01, "max_lat": lat + .01,
            "min_lon": lon - .01, "max_lon": lon + .01}
    probes = {"local_anchor": {"lat": lat, "lon": lon}}
    region = {
        "region": "candidate", "graph": f"graphs/{run_name}/road_graph.sqlite3",
        "run_name": run_name, "max_route_snap_km": 1.0,
        "bbox": bbox, "coverage_probes": probes,
    }
    config_path = tmp_path / release_checker.CONFIG_PATH
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"default_region": "candidate", "regions": [region]}))
    graph_path = tmp_path / region["graph"]
    graph = RoadGraph()
    graph.add_node(Node("start", lat, lon))
    graph.add_node(Node("end", lat + .001, lon))
    graph.add_edge(Edge("road", "start", "end", distance_km=.111,
                        scenic_score=5, road_type="primary", speed_limit_kmh=50))
    graph.save(graph_path, metadata={
        "bbox": bbox,
        "coverage_probes": {"local_anchor": {**probes["local_anchor"], "distance_km": 0}},
    })
    graph.close()

    regions, issues = release_checker._configured_route_regions(tmp_path)
    assert issues == []
    assert len(regions) == 1
    admitted = regions[0]
    assert admitted["graph_path"] == graph_path
    assert admitted["run_name"] == run_name
    assert admitted["region_dict"] == region
    assert release_checker._score_report_path(tmp_path, admitted) == (
        tmp_path / "data/processed/heuristic_runs" / run_name / "report/report.json"
    )
    assert release_checker._validate_sqlite_graph(graph_path, admitted["region_dict"]) == []
    assert release_checker._validate_admitted_graph(graph_path, admitted["region_dict"]) == []
    without_probes = {key: value for key, value in region.items() if key != "coverage_probes"}
    assert release_checker._validate_sqlite_graph(graph_path, without_probes)
    assert release_checker._validate_admitted_graph(graph_path, without_probes)

    regions, issues = release_checker._configured_route_regions(tmp_path, Path("override.sqlite3"))
    assert issues == []
    assert regions[0]["graph_path"] == tmp_path / "override.sqlite3"
    assert regions[0]["region_dict"] == region
    assert regions[0]["run_name"] == run_name


@pytest.mark.parametrize(
    "registry_text",
    [None, "{", "[]", "{}", '{"active": {}}', '{"active": {"checkpoint": " "}}'],
)
def test_release_checker_unavailable_checkpoint_has_no_pinned_fallback(
    tmp_path: Path, capsys, registry_text: str | None,
) -> None:
    config_path = tmp_path / release_checker.CONFIG_PATH
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({
        "default_region": "candidate",
        "regions": [{"region": "candidate", "graph": "candidate.sqlite3",
                     "run_name": "changed-run"}],
    }))
    if registry_text is not None:
        registry_path = tmp_path / release_checker.REGISTRY_PATH
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text(registry_text)
    checkpoint, issues = release_checker._active_checkpoint(tmp_path)
    assert checkpoint is None
    assert issues
    assert release_checker.check_artifacts(tmp_path) == 1
    output = capsys.readouterr().out
    assert str(release_checker.REGISTRY_PATH) in output
    assert "checkpoints/" not in output


@pytest.mark.parametrize("registry_text", [None, "{", "{}"])
@pytest.mark.parametrize("matching_model", [True, False])
def test_release_checker_requires_registry_but_preserves_region_model_override(
    tmp_path: Path, app_config, capsys, registry_text: str | None, matching_model: bool,
) -> None:
    model = tmp_path / "models/region.pt"
    model.parent.mkdir()
    model.write_bytes(b"region-specific model")
    config_path = tmp_path / release_checker.CONFIG_PATH
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({
        "default_region": "candidate",
        "regions": [{"region": "candidate", "graph": "candidate.sqlite3",
                     "run_name": "changed-run", "model_checkpoint": "models/region.pt"}],
    }))
    if registry_text is not None:
        registry_path = tmp_path / release_checker.REGISTRY_PATH
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text(registry_text)
    identity = {name: "ab" * 32 for name in (
        "source_contract_sha256", "preprocessing_contract_sha256", "grid_contract_sha256",
        "classifier_checkpoint_sha256", "calibration_artifact_sha256",
    )}
    identity.update(
        regression_checkpoint_sha256=(
            release_checker.hashlib.sha256(model.read_bytes()).hexdigest()
            if matching_model else "ff" * 32
        ),
        score_schema_version="1", label_schema_version="1", scoring_mode="learned",
    )
    report = tmp_path / "data/processed/heuristic_runs/changed-run/report/report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "run_info": identity, "summary": {"min": 0, "max": 10},
        "tiles": [{"x": 0, "y": 0, "z": 0, "scenic_score": 5}],
    }))

    assert release_checker.check_artifacts(tmp_path) == 1
    output = capsys.readouterr().out
    assert str(release_checker.REGISTRY_PATH) in output
    mismatch = "report regression checkpoint identity does not match configured model"
    assert (mismatch in output) is not matching_model
