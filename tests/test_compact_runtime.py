from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from threading import Event

import pytest

from src.data_pipeline.web_mercator import lat_lon_to_tile
from src.route_planner import service as route_service
from src.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)
from src.route_planner.graph import (
    CompactRoadGraph,
    Edge,
    Node,
    RoadGraph,
    write_compact_graph,
)
from src.route_planner.planner import ScenicRoutePlanner

# The eager score pass lives in the service module; import it there to avoid
# depending on private graph-module aliases.
from src.route_planner.service import (
    RouteRequest,
    _apply_tile_scores_to_graph_native,
    _file_signature,
    _resolved_path_key,
    _signature_digest,
    load_tile_scores,
)


@pytest.fixture(autouse=True)
def _clear_route_caches() -> None:
    yield
    route_service.clear_route_caches()


def _spy_compact_resource_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[int, int]:
    """Count per-graph resource finalizer runs, keyed by binary mmap identity.

    Compact graphs are closed by a close-on-final-release finalizer rather
    than by a direct ``close`` call, so the finalizer itself is spied and the
    graph's captured resources are asserted directly.
    """
    close_counts: dict[int, int] = {}
    original = route_service._close_compact_graph_resources

    def close_spy(
        sections: object,
        bin_mmap: object,
        bin_file: object,
        score_sidecar: object,
        projection_mmap: object,
        projection_file: object,
    ) -> None:
        close_counts[id(bin_mmap)] = close_counts.get(id(bin_mmap), 0) + 1
        original(
            sections,
            bin_mmap,
            bin_file,
            score_sidecar,
            projection_mmap,
            projection_file,
        )

    monkeypatch.setattr(
        route_service, "_close_compact_graph_resources", close_spy
    )
    return close_counts


def _capture_compact_resources(
    graph: CompactRoadGraph,
) -> dict[str, object]:
    """Snapshot the mmap/file handles owned by one compact graph."""
    resources: dict[str, object] = {}
    if graph._bin_mmap is not None:
        resources["bin_mmap"] = graph._bin_mmap
    if graph._bin_file is not None:
        resources["bin_file"] = graph._bin_file
    sidecar = graph._active_score_sidecar
    if sidecar is not None:
        if getattr(sidecar, "_mmap", None) is not None:
            resources["score_mmap"] = sidecar._mmap
        if getattr(sidecar, "_file", None) is not None:
            resources["score_file"] = sidecar._file
    epi = graph._nearest_edge_projection_index
    if epi is not None:
        if getattr(epi, "_mmap", None) is not None:
            resources["projection_mmap"] = epi._mmap
        if getattr(epi, "_file", None) is not None:
            resources["projection_file"] = epi._file
    return resources


def _assert_resources_closed(resources: dict[str, object]) -> None:
    assert resources
    for kind, resource in resources.items():
        assert resource.closed, f"{kind} still open after final release"


def _assert_resources_open(resources: dict[str, object]) -> None:
    for kind, resource in resources.items():
        assert not resource.closed, f"{kind} closed prematurely"


def _fixture_graph() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node("n0", 42.0, -72.0))
    graph.add_node(Node("n1", 42.05, -72.0))
    graph.add_node(Node("n2", 42.1, -72.0))
    graph.add_node(Node("n3", 42.1, -72.1))
    graph.add_edge(
        Edge(
            "e0",
            "n0",
            "n1",
            distance_km=5.56,
            scenic_score=5.0,
            road_name="Main St",
            road_type="secondary",
            speed_limit_kmh=40,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e1",
            "n1",
            "n2",
            distance_km=5.56,
            scenic_score=8.0,
            road_name=None,
            road_type="secondary",
            speed_limit_kmh=80,
            one_way=False,
        )
    )
    graph.add_edge(
        Edge(
            "e2",
            "n2",
            "n3",
            distance_km=7.5,
            scenic_score=3.0,
            road_name="River Rd",
            road_type="residential",
            speed_limit_kmh=30,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e3",
            "n1",
            "n3",
            distance_km=9.8,
            scenic_score=7.0,
            road_name="Hwy 9",
            road_type="trunk",
            speed_limit_kmh=90,
            one_way=True,
        )
    )
    return graph


def _publish_compact(
    tmp_path: Path, graph: RoadGraph
) -> tuple[Path, Path, dict[str, object]]:
    sqlite_path = tmp_path / "road_graph.sqlite3"
    graph.save(sqlite_path)
    manifest_path = sqlite_path.with_name("road_graph.compact.json")
    record = write_compact_graph(sqlite_path, manifest_path)
    return sqlite_path, manifest_path, record


def _tile_report(
    tmp_path: Path, graph: RoadGraph, scores: dict[str, float], zoom: int = 14
) -> tuple[Path, dict[tuple[int, int, int], float]]:
    tiles: dict[tuple[int, int, int], float] = {}
    rows: list[dict[str, object]] = []
    for edge in graph.edges.values():
        score = scores.get(edge.id)
        if score is None:
            continue
        start = graph.get_node(edge.start_node_id)
        end = graph.get_node(edge.end_node_id)
        midpoint = (
            0.5 * (start.lat + end.lat),
            0.5 * (start.lon + end.lon),
        )
        tile_x, tile_y = lat_lon_to_tile(*midpoint, zoom)
        score = scores[edge.id]
        tiles[(zoom, tile_x, tile_y)] = score
        rows.append(
            {"z": zoom, "x": tile_x, "y": tile_y, "scenic_score": score}
        )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"tiles": rows}), encoding="utf-8"
    )
    return report_path, tiles


def test_compact_round_trip_preserves_graph_and_metadata(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, record = _publish_compact(tmp_path, graph)
    loaded = CompactRoadGraph.load(manifest_path)
    reference = RoadGraph.load(sqlite_path)

    assert len(loaded.nodes) == len(reference.nodes) == 4
    assert len(loaded.edges) == len(reference.edges) == 4
    assert list(loaded.nodes) == list(reference.nodes)
    assert list(loaded.edges) == list(reference.edges)
    for node_id in reference.nodes:
        expected = reference.nodes[node_id]
        actual = loaded.nodes[node_id]
        assert (actual.id, actual.lat, actual.lon) == (
            expected.id,
            expected.lat,
            expected.lon,
        )
    for edge_id in reference.edges:
        expected = reference.edges[edge_id]
        actual = loaded.edges[edge_id]
        assert actual.id == expected.id
        assert actual.start_node_id == expected.start_node_id
        assert actual.end_node_id == expected.end_node_id
        assert actual.distance_km == expected.distance_km
        assert actual.scenic_score == expected.scenic_score
        assert actual.road_name == expected.road_name
        assert actual.road_type == expected.road_type
        assert actual.speed_limit_kmh == expected.speed_limit_kmh
        assert actual.one_way == expected.one_way

    # Directed traversal parity (forward then reverse, in canonical order).
    for node_id in reference.nodes:
        eager_refs = [
            (edge.id, edge.start_node_id, edge.end_node_id)
            for edge in reference.iter_edges(node_id)
        ]
        compact_refs = [
            (edge.id, edge.start_node_id, edge.end_node_id)
            for edge in loaded.iter_edges(node_id)
        ]
        assert compact_refs == eager_refs

    assert loaded.artifact_metadata["format"] == "scenic-roadgraph-compact"
    assert loaded.artifact_metadata["source"]["sha256"] == record["source_sha256"]
    assert loaded.artifact_metadata["graph"]["edge_count"] == 4
    assert loaded.artifact_metadata["graph"]["traversal_count"] == 5
    assert loaded.source_sha256 == record["source_sha256"]

    # The persisted projection sidecar loads with a read-only mmap.
    status = dict(loaded.edge_projection_index_status)
    assert status["state"] == "loaded"
    assert status["mmap_read_only"] is True

    # Numeric sections are read-only mmap views.
    assert loaded._sections["node_lat"].flags.writeable is False
    with pytest.raises(ValueError):
        loaded._sections["node_lat"][0] = 1.0
    with pytest.raises(RuntimeError):
        loaded.add_node(Node("x", 0.0, 0.0))
    with pytest.raises(RuntimeError):
        loaded.add_edge(
            Edge("x", "n0", "n1", 1.0, 5.0)
        )


def test_compact_publication_is_deterministic(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _first = _publish_compact(tmp_path, graph)
    first_manifest = manifest_path.read_bytes()
    first_bin = (tmp_path / "road_graph.compact.bin").read_bytes()

    other = tmp_path / "other"
    other.mkdir()
    second_manifest_path = other / "road_graph.compact.json"
    second = write_compact_graph(sqlite_path, second_manifest_path)
    assert (other / "road_graph.compact.bin").read_bytes() == first_bin
    assert second_manifest_path.read_bytes() == first_manifest
    assert second["bin_sha256"] == _first["bin_sha256"]
    assert second["manifest_sha256"] == _first["manifest_sha256"]


def test_compact_rejects_tampered_source_hash(tmp_path: Path) -> None:
    graph = _fixture_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        CompactRoadGraph.load(manifest_path)


def test_compact_rejects_tampered_binary_payload(tmp_path: Path) -> None:
    graph = _fixture_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    bin_path = tmp_path / "road_graph.compact.bin"
    bin_path.write_bytes(bin_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="SHA-256|size|truncated"):
        CompactRoadGraph.load(manifest_path)


def test_compact_rejects_unknown_format_and_schema(tmp_path: Path) -> None:
    graph = _fixture_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "not-the-compact-format"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        CompactRoadGraph.load(manifest_path)

    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        CompactRoadGraph.load(manifest_path)


def test_compact_rejects_stale_mask_and_missing_sections(tmp_path: Path) -> None:
    graph = _fixture_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["highway_road_types"] = "motorway"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="highway road-type mask"):
        CompactRoadGraph.load(manifest_path)

    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["sections"]["node_lat"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing sections"):
        CompactRoadGraph.load(manifest_path)


def test_compact_route_parity_with_sqlite(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    reference = RoadGraph.load(sqlite_path)
    compact = CompactRoadGraph.load(manifest_path)

    def route_key(route: object) -> tuple[object, ...]:
        return (
            tuple(route.edge_ids),
            tuple(route.traversal_ids),
            route.total_distance_km,
            route.estimated_duration_minutes,
            route.raw_scenic_score,
            route.normalized_scenic_score,
            route.exactness_status,
            route.highway_count,
            tuple(tuple(round(value, 9) for value in point) for point in route.waypoints),
        )

    start = (42.0, -72.0)
    end = (42.1, -72.1)
    planner_reference = ScenicRoutePlanner(graph=reference)
    planner_compact = ScenicRoutePlanner(graph=compact)
    reference_fastest = planner_reference.find_fastest_route(
        start, end, avoid_highways=False
    )
    compact_fastest = planner_compact.find_fastest_route(
        start, end, avoid_highways=False
    )
    assert route_key(compact_fastest) == route_key(reference_fastest)
    reference_scenic = planner_reference.find_scenic_route(
        start, end, scenic_weight=0.8, avoid_highways=False
    )
    compact_scenic = planner_compact.find_scenic_route(
        start, end, scenic_weight=0.8, avoid_highways=False
    )
    assert route_key(compact_scenic) == route_key(reference_scenic)


def test_score_sidecar_deterministic_and_matches_eager(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, record = _publish_compact(tmp_path, graph)
    report_path, _tiles = _tile_report(
        tmp_path,
        graph,
        {"e0": 6.5, "e1": 9.0, "e2": 2.0, "e3": 7.5},
    )
    score_map, _zoom = load_tile_scores(report_path)

    eager = RoadGraph.load(sqlite_path)
    _apply_tile_scores_to_graph_native(eager, score_map, zoom=14, fallback=None)

    compact = CompactRoadGraph.load(manifest_path)
    report_signature = _signature_digest(
        _resolved_path_key(report_path), _file_signature(report_path)
    )
    matched, total = compact.activate_report_scores(
        score_map,
        zoom=14,
        fallback=None,
        report_signature=report_signature,
        normalization="linear-v1",
        tile_scores_path=report_path,
    )
    assert (matched, total) == (4, 4)
    assert compact._route_service_score_mapping == (4, 4, 0)
    for edge_id, expected in eager.edges.items():
        assert compact.edges[edge_id].scenic_score == pytest.approx(
            expected.scenic_score
        )

    # Deterministic regeneration: identical sidecar bytes for identical inputs.
    json_path, bin_path = compact._score_sidecar_paths(
        report_signature, 14, None, "linear-v1"
    )
    first_json = json_path.read_bytes()
    first_bin = bin_path.read_bytes()
    json_path.unlink()
    bin_path.unlink()
    compact_again = CompactRoadGraph.load(manifest_path)
    compact_again.activate_report_scores(
        score_map,
        zoom=14,
        fallback=None,
        report_signature=report_signature,
        normalization="linear-v1",
        tile_scores_path=report_path,
    )
    json_path_again, bin_path_again = compact_again._score_sidecar_paths(
        report_signature, 14, None, "linear-v1"
    )
    assert json_path_again.read_bytes() == first_json
    assert bin_path_again.read_bytes() == first_bin

    sidecar_manifest = json.loads(first_json.decode("utf-8"))
    assert sidecar_manifest["report_signature"] == report_signature
    assert sidecar_manifest["source"]["sha256"] == record["source_sha256"]
    assert sidecar_manifest["zoom"] == 14
    assert sidecar_manifest["fallback"] is None
    assert sidecar_manifest["normalization"] == "linear-v1"
    assert sidecar_manifest["counts"] == {
        "matched_edges": 4,
        "fallback_edges": 0,
        "total_edges": 4,
    }


def test_score_sidecar_fallback_provenance(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    # A report covering only two edges: unmatched edges fall back to 4.0.
    report_path, _tiles = _tile_report(
        tmp_path, graph, {"e0": 6.5, "e1": 9.0}, zoom=14
    )
    score_map, _zoom = load_tile_scores(report_path)
    compact = CompactRoadGraph.load(manifest_path)
    report_signature = _signature_digest(
        _resolved_path_key(report_path), _file_signature(report_path)
    )
    matched, total = compact.activate_report_scores(
        score_map,
        zoom=14,
        fallback=4.0,
        report_signature=report_signature,
        normalization="linear-v1",
        tile_scores_path=report_path,
    )
    assert (matched, total) == (2, 4)
    assert compact._route_service_score_mapping == (2, 4, 2)
    assert compact.edges["e0"].scenic_score == pytest.approx(6.5)
    assert compact.edges["e1"].scenic_score == pytest.approx(9.0)
    assert compact.edges["e2"].scenic_score == pytest.approx(4.0)
    assert compact.edges["e3"].scenic_score == pytest.approx(4.0)


def test_compact_load_avoids_eager_object_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.route_planner.graph as graph_module

    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)

    counts = {"node": 0, "edge": 0}
    node_init = graph_module.Node.__init__
    edge_init = graph_module.Edge.__init__

    def counting_node(self: object, *args: object, **kwargs: object) -> None:
        counts["node"] += 1
        node_init(self, *args, **kwargs)

    def counting_edge(self: object, *args: object, **kwargs: object) -> None:
        counts["edge"] += 1
        edge_init(self, *args, **kwargs)

    monkeypatch.setattr(graph_module.Node, "__init__", counting_node)
    monkeypatch.setattr(graph_module.Edge, "__init__", counting_edge)

    compact = graph_module.CompactRoadGraph.load(manifest_path)
    assert counts == {"node": 0, "edge": 0}
    assert len(compact.edges) == 4
    assert list(compact.edges) == ["e0", "e1", "e2", "e3"]
    assert counts == {"node": 0, "edge": 0}
    assert len(compact.nodes) == 4
    assert "n1" in compact.nodes
    assert counts == {"node": 0, "edge": 0}

    # Lazy per-record materialization only.
    edge = compact.edges["e1"]
    assert counts["edge"] == 1
    assert edge.scenic_score == 8.0
    node = compact.get_node("n1")
    assert counts["node"] == 1
    assert (node.lat, node.lon) == (42.05, -72.0)

    # Projection attachment binds rank-based keys, never an O(E) tuple.
    index = compact._nearest_edge_projection_index
    assert index is not None
    assert isinstance(
        index._canonical_keys, graph_module._CompactEdgeKeySequence
    )
    assert counts == {"node": 1, "edge": 1}


def test_compact_scored_plan_routes_avoids_native_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    report_path, _tiles = _tile_report(
        tmp_path, graph, {"e0": 6.5, "e1": 9.0, "e2": 2.0, "e3": 7.5}
    )

    def fail_clone(*args: object, **kwargs: object) -> object:
        pytest.fail("native clone must not run for compact graphs")

    monkeypatch.setattr(route_service, "_clone_graph_for_scoring", fail_clone)
    monkeypatch.setattr(
        route_service, "_apply_tile_scores_to_graph_native", fail_clone
    )

    request = RouteRequest(
        graph_geojson=str(manifest_path),
        start=(42.0, -72.0),
        end=(42.1, -72.1),
        scenic_weight=0.8,
        tile_scores_json=str(report_path),
    )
    result = route_service.plan_routes(request)
    assert result["score_mapping"]["enabled"] is True
    assert result["score_mapping"]["matched_edges"] == 4
    assert result["score_mapping"]["total_edges"] == 4
    assert result["score_mapping"]["fallback_edges"] == 0
    assert result["score_mapping"]["report_signature"] is not None
    assert result["score_mapping"]["graph_signature"] is not None
    assert result["diagnostics"]["scored_graph_cache_hit"] is False
    assert result["diagnostics"]["route_response_cache_hit"] is False

    second = route_service.plan_routes(request)
    assert second["diagnostics"]["scored_graph_cache_hit"] is True
    assert second["diagnostics"]["route_response_cache_hit"] is True
    assert second["score_mapping"] == result["score_mapping"]
    assert (
        second["routes"][0]["metrics"]["edge_ids"]
        == result["routes"][0]["metrics"]["edge_ids"]
    )
    second["routes"][0]["metrics"]["edge_ids"].append("mutated")
    third = route_service.plan_routes(request)
    assert third["diagnostics"]["route_response_cache_hit"] is True
    assert (
        third["routes"][0]["metrics"]["edge_ids"]
        == result["routes"][0]["metrics"]["edge_ids"]
    )

    class Cancelled:
        def is_set(self) -> bool:
            return True

    with pytest.raises(RoutingCancelled):
        route_service.plan_routes(
            request,
            deadline=RoutingDeadline(cancel_event=Cancelled()),
        )

    route_service.clear_route_caches()

    cancel_event = Event()
    real_dumps = route_service.pickle.dumps

    def cancel_after_serialize(value: object, *args: object, **kwargs: object) -> bytes:
        serialized = real_dumps(value, *args, **kwargs)
        cancel_event.set()
        return serialized

    monkeypatch.setattr(route_service.pickle, "dumps", cancel_after_serialize)
    with pytest.raises(RoutingCancelled):
        route_service.plan_routes(
            request,
            deadline=RoutingDeadline(cancel_event=cancel_event),
        )


def test_compact_load_honours_cancellation(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)

    class Cancelled:
        def is_set(self) -> bool:
            return True

    deadline = RoutingDeadline(cancel_event=Cancelled())
    with pytest.raises(RoutingCancelled):
        CompactRoadGraph.load(manifest_path, check_cancelled=deadline.check)


def test_compact_plan_routes_honours_deadline(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    request = RouteRequest(
        graph_geojson=str(manifest_path),
        start=(42.0, -72.0),
        end=(42.1, -72.1),
    )
    with pytest.raises(RoutingTimeout):
        route_service.plan_routes(
            request, deadline=RoutingDeadline.after(0.0)
        )


def test_compact_load_post_construction_cancellation_closes_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import mmap

    graph = _fixture_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    opened_files = []
    opened_mmaps = []
    original_open = builtins.open
    original_mmap = mmap.mmap

    def tracking_open(*args, **kwargs):
        file_obj = original_open(*args, **kwargs)
        if args and str(args[0]).endswith(".bin"):
            opened_files.append(file_obj)
        return file_obj

    def tracking_mmap(*args, **kwargs):
        mmap_obj = original_mmap(*args, **kwargs)
        opened_mmaps.append(mmap_obj)
        return mmap_obj

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr("mmap.mmap", tracking_mmap)
    call_count = 0

    def check_cancelled() -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise RoutingCancelled("cancelled post-construction")

    with pytest.raises(RoutingCancelled):
        CompactRoadGraph.load(
            manifest_path, check_cancelled=check_cancelled, verify=False
        )

    assert len(opened_files) == 1 and opened_files[0].closed
    assert len(opened_mmaps) == 1 and opened_mmaps[0].closed


def test_compact_preload_fails_closed_without_projection_sidecar(
    tmp_path: Path,
) -> None:
    graph = _fixture_graph()
    sqlite_path = tmp_path / "road_graph.sqlite3"
    graph.save(sqlite_path)
    sidecar = sqlite_path.with_name("road_graph.sqlite3.edge_projection_index")
    sidecar.unlink()
    manifest_path = sqlite_path.with_name("road_graph.compact.json")
    write_compact_graph(sqlite_path, manifest_path)
    with pytest.raises(RuntimeError, match="projection"):
        route_service.preload_route_assets(manifest_path)


def test_road_graph_load_dispatches_compact_manifest(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    loaded = RoadGraph.load(manifest_path)
    assert isinstance(loaded, CompactRoadGraph)
    assert len(loaded.edges) == 4
    # The SQLite path still returns the eager object graph.
    eager = RoadGraph.load(sqlite_path)
    assert not isinstance(eager, CompactRoadGraph)
    assert len(eager.edges) == 4


def test_compact_large_path_parity_with_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route parity through the production large-graph search paths.

    Thresholds are lowered so the compact graph is routed through the
    target-bounded bidirectional core and Lagrangian scenic search instead of
    the small-graph oracle, while the eager SQLite graph uses its existing
    multi-access compiled paths.
    """
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    reference = RoadGraph.load(sqlite_path)
    compact = CompactRoadGraph.load(manifest_path)

    monkeypatch.setattr(
        ScenicRoutePlanner, "_ENDPOINT_OVERLAY_MAX_NODES", 2
    )
    monkeypatch.setattr(
        ScenicRoutePlanner, "_LARGE_GRAPH_EDGE_THRESHOLD", 2
    )
    monkeypatch.setattr(
        ScenicRoutePlanner, "_COMPILED_SCENIC_MIN_NODES", 2
    )

    start = (42.0, -72.0)
    end = (42.1, -72.1)
    reference_fastest = ScenicRoutePlanner(graph=reference).find_fastest_route(
        start, end, avoid_highways=False
    )
    compact_fastest = ScenicRoutePlanner(graph=compact).find_fastest_route(
        start, end, avoid_highways=False
    )
    assert tuple(compact_fastest.edge_ids) == tuple(reference_fastest.edge_ids)
    assert tuple(compact_fastest.traversal_ids) == tuple(
        reference_fastest.traversal_ids
    )
    assert compact_fastest.estimated_duration_minutes == pytest.approx(
        reference_fastest.estimated_duration_minutes
    )
    assert compact_fastest.total_distance_km == pytest.approx(
        reference_fastest.total_distance_km
    )
    assert compact_fastest.exactness_status == "exact"

    reference_scenic = ScenicRoutePlanner(graph=reference).find_scenic_route(
        start, end, scenic_weight=0.8, avoid_highways=False
    )
    compact_scenic = ScenicRoutePlanner(graph=compact).find_scenic_route(
        start, end, scenic_weight=0.8, avoid_highways=False
    )
    assert compact_scenic.exactness_status == reference_scenic.exactness_status
    assert compact_scenic.estimated_duration_minutes == pytest.approx(
        reference_scenic.estimated_duration_minutes
    )
    assert compact_scenic.total_distance_km == pytest.approx(
        reference_scenic.total_distance_km
    )
    assert compact_scenic.objective_value == pytest.approx(
        reference_scenic.objective_value
    )
    assert compact_scenic.normalized_scenic_score == pytest.approx(
        reference_scenic.normalized_scenic_score
    )
    # Both searches must return a valid simple route between the endpoints.
    assert compact_scenic.edge_ids
    assert compact_scenic.edge_ids[0] in {"e0"}
    assert compact_scenic.waypoints[0] == start
    assert compact_scenic.waypoints[-1] == end
    assert compact_scenic.edge_ids == reference_scenic.edge_ids

    # Highway avoidance blocks the trunk edge; the compact Lagrangian
    # override path must skip blocked (non-finite) base weights.
    reference_avoid = ScenicRoutePlanner(graph=reference).find_scenic_route(
        start, end, scenic_weight=0.8, avoid_highways=True
    )
    compact_avoid = ScenicRoutePlanner(graph=compact).find_scenic_route(
        start, end, scenic_weight=0.8, avoid_highways=True
    )
    assert tuple(compact_avoid.edge_ids) == tuple(reference_avoid.edge_ids)
    assert "e3" not in compact_avoid.edge_ids
    assert compact_avoid.estimated_duration_minutes == pytest.approx(
        reference_avoid.estimated_duration_minutes
    )
    assert compact_avoid.exactness_status == reference_avoid.exactness_status


def test_compact_include_baseline_runs_one_native_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """include_baseline plans baseline+scenic with one native ranked search.

    The unrestricted baseline fastest path and the scenic fastest-path
    reference are the identical compact multiplier=0/cost_limit=None search.
    The scenic request reuses the baseline request's frozen overlay and
    ranked fastest result instead of running the search a second time.
    Thresholds are lowered so the small fixture exercises the production
    large-graph compact paths (native ranked search plus Lagrangian scenic).
    """
    graph = _fixture_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)

    monkeypatch.setattr(
        ScenicRoutePlanner, "_ENDPOINT_OVERLAY_MAX_NODES", 2
    )
    monkeypatch.setattr(
        ScenicRoutePlanner, "_LARGE_GRAPH_EDGE_THRESHOLD", 2
    )
    monkeypatch.setattr(
        ScenicRoutePlanner, "_COMPILED_SCENIC_MIN_NODES", 2
    )
    monkeypatch.setattr(ScenicRoutePlanner, "_EXACT_ORACLE_MAX_NODES", 0)
    monkeypatch.setattr(ScenicRoutePlanner, "_EXACT_ORACLE_MAX_EDGES", 0)

    calls: list[object] = []
    original = ScenicRoutePlanner._multi_access_builtin_path

    def counting(
        self: ScenicRoutePlanner,
        *args: object,
        **kwargs: object,
    ) -> object:
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        ScenicRoutePlanner, "_multi_access_builtin_path", counting
    )

    start = (42.0, -72.0)
    end = (42.1, -72.1)
    result = route_service.plan_routes(
        RouteRequest(
            graph_geojson=str(manifest_path),
            start=start,
            end=end,
            include_baseline=True,
        )
    )
    # Baseline fastest path and scenic fastest-path reference are the same
    # exact search; the scenic request reuses the baseline result.
    assert len(calls) == 1
    assert [r["route_kind"] for r in result["routes"]] == [
        "scenic",
        "baseline",
    ]
    scenic_with_baseline = result["routes"][0]["metrics"]
    baseline = result["routes"][1]["metrics"]
    assert baseline["edge_ids"]
    assert result["geojson"]["type"] == "FeatureCollection"
    assert len(result["geojson"]["features"]) == 2

    # A scenic-only request (no baseline first) runs its own search and must
    # produce an identical scenic route.
    scenic_only = route_service.plan_routes(
        RouteRequest(
            graph_geojson=str(manifest_path),
            start=start,
            end=end,
            include_baseline=False,
        )
    )
    assert len(calls) == 2
    assert [r["route_kind"] for r in scenic_only["routes"]] == ["scenic"]
    scenic_without_baseline = scenic_only["routes"][0]["metrics"]
    for key in (
        "edge_ids",
        "traversal_ids",
        "total_distance_km",
        "estimated_duration_minutes",
        "average_scenic_score",
        "normalized_scenic_score",
        "objective_value",
        "exactness_status",
        "algorithm",
    ):
        assert scenic_without_baseline[key] == scenic_with_baseline[key], key

    # Reuse is request-scoped and keyed by graph identity/stamp, endpoints,
    # and constraints: the same endpoints on the same planner reuse the
    # fastest result, while a different endpoint must run a new search.
    planner = ScenicRoutePlanner(graph=compact)
    planner.find_fastest_route(start, end, avoid_highways=False)
    assert len(calls) == 3
    planner.find_scenic_route(start, end, scenic_weight=0.8)
    assert len(calls) == 3
    planner.find_scenic_route(start, (42.1, -72.0), scenic_weight=0.8)
    assert len(calls) == 4


def _grid_graph() -> RoadGraph:
    import math

    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return r * 2 * math.asin(math.sqrt(a))

    graph = RoadGraph()
    lat0, lon0 = 42.0, -72.0
    for i in range(4):
        for j in range(4):
            graph.add_node(Node(f"n{i}_{j}", lat0 + 0.01 * i, lon0 + 0.01 * j))
    edge_id = 0
    for i in range(4):
        for j in range(4):
            if j + 1 < 4:
                distance = haversine(
                    lat0 + 0.01 * i,
                    lon0 + 0.01 * j,
                    lat0 + 0.01 * i,
                    lon0 + 0.01 * (j + 1),
                )
                graph.add_edge(
                    Edge(
                        f"e{edge_id}",
                        f"n{i}_{j}",
                        f"n{i}_{j + 1}",
                        distance,
                        5.0 + (i + j) % 5,
                        road_type="secondary",
                        speed_limit_kmh=40,
                        one_way=False,
                    )
                )
                edge_id += 1
            if i + 1 < 4:
                distance = haversine(
                    lat0 + 0.01 * i,
                    lon0 + 0.01 * j,
                    lat0 + 0.01 * (i + 1),
                    lon0 + 0.01 * j,
                )
                graph.add_edge(
                    Edge(
                        f"e{edge_id}",
                        f"n{i}_{j}",
                        f"n{i + 1}_{j}",
                        distance,
                        3.0 + (i * j) % 4,
                        road_type="primary",
                        speed_limit_kmh=60,
                        one_way=False,
                    )
                )
                edge_id += 1
    return graph


def test_compact_frontier_search_path(tmp_path: Path) -> None:
    """The deadline-bounded frontier search works on a compact base graph."""
    graph = _grid_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    assert compact.node_count == 16
    assert compact.traversal_count == 48

    ScenicRoutePlanner._LARGE_GRAPH_EDGE_THRESHOLD = 2
    ScenicRoutePlanner._ENDPOINT_OVERLAY_MAX_NODES = 2
    ScenicRoutePlanner._COMPILED_SCENIC_MIN_NODES = 10**9
    planner = ScenicRoutePlanner(graph=compact)
    route = planner.find_scenic_route(
        (42.0, -72.0), (42.03, -71.97), scenic_weight=0.8
    )
    assert route.algorithm == "production-multilabel-frontier"
    assert route.exactness_status == "exact"
    assert route.edge_ids
    assert route.waypoints[0] == (42.0, -72.0)
    assert route.waypoints[-1] == (42.03, -71.97)
    assert route.search_diagnostics["mode"] == "frontier"
    assert route.search_diagnostics["deadline_reached"] is False
    ScenicRoutePlanner.clear_shared_caches()


def test_compact_preload_and_plan_without_scores(tmp_path: Path) -> None:
    graph = _fixture_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    preload = route_service.preload_route_assets(manifest_path)
    assert preload["graph_nodes"] == 4
    assert preload["score_mapping"]["enabled"] is False
    assert preload["edge_projection_index"]["state"] == "loaded"
    assert preload["planner_preload"]["available"] is True
    request = RouteRequest(
        graph_geojson=str(manifest_path),
        start=(42.0, -72.0),
        end=(42.1, -72.1),
    )
    result = route_service.plan_routes(request)
    assert result["routes"][0]["route_kind"] == "scenic"
    assert result["diagnostics"]["graph_cache_hit"] is True


def test_compact_preload_with_scores(tmp_path: Path) -> None:
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    report_path, _tiles = _tile_report(
        tmp_path, graph, {"e0": 6.5, "e1": 9.0, "e2": 2.0, "e3": 7.5}
    )
    preload = route_service.preload_route_assets(
        manifest_path,
        tile_scores_path=report_path,
        tile_score_zoom=14,
    )
    assert preload["score_mapping"]["enabled"] is True
    assert preload["score_mapping"]["matched_edges"] == 4
    assert preload["score_mapping"]["total_edges"] == 4
    assert preload["score_mapping"]["report_signature"] is not None
    request = RouteRequest(
        graph_geojson=str(manifest_path),
        start=(42.0, -72.0),
        end=(42.1, -72.1),
        tile_scores_json=str(report_path),
        tile_score_zoom=14,
    )
    result = route_service.plan_routes(request)
    assert result["diagnostics"]["scored_graph_cache_hit"] is True
    assert result["score_mapping"]["matched_edges"] == 4


def test_compact_no_scan_no_copy_and_compiled_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify offline metadata speed bound, zero-copy prewarm, and C search parity."""
    import json
    import numpy as np
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["geodesic_bound_speed"]["all"] is None
    assert manifest_data["geodesic_bound_speed"]["avoid_highways"] is None

    compact = CompactRoadGraph.load(manifest_path)
    assert compact.endpoint_geodesic_bound_speed(False) is None
    assert compact.endpoint_geodesic_bound_speed(True) is None

    def fail_scan(*args: object, **kwargs: object) -> None:
        raise AssertionError("Graph-wide trigonometric scan was invoked")

    import src.route_planner.graph as graph_module
    monkeypatch.setattr(graph_module, "_haversine_km", fail_scan)
    assert compact.endpoint_geodesic_bound_speed(False) is None
    preload = route_service.preload_route_assets(manifest_path)
    assert preload["planner_preload"]["available"] is True
    assert preload["planner_preload"]["data_variants"] == 0

    reference = RoadGraph.load(sqlite_path)
    start = (42.0, -72.0)
    end = (42.1, -72.1)
    ref_planner = ScenicRoutePlanner(graph=reference)
    compact_planner = ScenicRoutePlanner(graph=compact)

    ref_route = ref_planner.find_fastest_route(start, end)
    compact_route = compact_planner.find_fastest_route(start, end)
    assert compact_route.edge_ids == ref_route.edge_ids
    assert compact_route.estimated_duration_minutes == pytest.approx(
        ref_route.estimated_duration_minutes
    )


def _long_chain_graph(edge_count: int = 2600) -> RoadGraph:
    """One-way chain long enough that a bidirectional meeting happens after
    more than 1024 traversals on each side."""
    graph = RoadGraph()
    for index in range(edge_count + 1):
        graph.add_node(Node(f"c{index}", 42.0 + 0.001 * index, -72.0))
    for index in range(edge_count):
        graph.add_edge(
            Edge(
                f"e{index}",
                f"c{index}",
                f"c{index + 1}",
                distance_km=0.2,
                scenic_score=5.0,
                road_type="secondary",
                speed_limit_kmh=50,
                one_way=True,
            )
        )
    return graph


def test_compact_search_returns_original_seed_indices_after_filtering(
    tmp_path: Path,
) -> None:
    from src.route_planner._compact_search import run_compact_bidirectional_search

    graph = _long_chain_graph(4)
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None

    result = run_compact_bidirectional_search(
        topology,
        planner._make_fastest_cost_function(),
        [("missing-forward", 0.0, (), None), ("c0", 0.0, (), None)],
        [("missing-reverse", 0.0, (), None), ("c4", 0.0, (), None)],
    )

    assert result is not None
    positions, _total_cost, fwd_index, rev_index = result
    assert positions == [0, 1, 2, 3]
    assert (fwd_index, rev_index) == (1, 1)


def _diamond_graph() -> RoadGraph:
    """Two equal-duration paths n0 -> n3: canonical-smaller e0/e1 and
    canonical-larger e2/e3.

    The e2/e3 path is inserted *first*, so its traversals occupy earlier CSR
    positions: the forward row of n0 relaxes e2 before e0 and the naive CSR
    insertion-order search meets on the e2/e3 side.  The Python ranked
    search must still choose the lexicographically smaller canonical edge-id
    sequence e0/e1, and the native search must match it.
    """
    graph = RoadGraph()
    graph.add_node(Node("n0", 42.0, -72.0))
    graph.add_node(Node("n1", 42.05, -72.0))
    graph.add_node(Node("n2", 42.05, -72.1))
    graph.add_node(Node("n3", 42.1, -72.05))
    # Canonical edge ranks: e2=0, e3=1, e0=2, e1=3.
    graph.add_edge(
        Edge(
            "e2",
            "n0",
            "n2",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e3",
            "n2",
            "n3",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e0",
            "n0",
            "n1",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e1",
            "n1",
            "n3",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    return graph


def _shared_prefix_merge_graph() -> RoadGraph:
    """Two equal-cost n0 -> n5 paths share the first edge (e0), split on the
    middle suffix, and rejoin at n3 before sharing e5:

        n0 -e0-> n1 -e1-> n2 -e2-> n3 -e5-> n5
                     \--e3-> n4 -e4-> n3 -e5-> n5

    The lexicographically larger suffix (e3/e4) is inserted *before* the
    smaller one (e1/e2), so its traversals occupy earlier CSR positions and
    the naive insertion-order search meets the larger path first.  A
    comparator that stops after the shared first edge cannot tell the
    candidates apart and keeps the larger path; only a full oldest-to-newest
    comparison of every canonical edge id picks e0/e1/e2 like Python tuple
    ordering.
    """
    graph = RoadGraph()
    graph.add_node(Node("n0", 42.0, -72.0))
    graph.add_node(Node("n1", 42.01, -72.0))
    graph.add_node(Node("n2", 42.02, -72.01))
    graph.add_node(Node("n4", 42.02, -71.99))
    graph.add_node(Node("n3", 42.03, -72.0))
    graph.add_node(Node("n5", 42.04, -72.0))
    # Reverse canonical insertion order: e5 < e4 < e3 < e2 < e1 < e0.
    graph.add_edge(
        Edge(
            "e5",
            "n3",
            "n5",
            distance_km=100.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e4",
            "n4",
            "n3",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e3",
            "n1",
            "n4",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e2",
            "n2",
            "n3",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e1",
            "n1",
            "n2",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e0",
            "n0",
            "n1",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    return graph


def _deep_prefix_parallel_end_graph(edge_count: int = 131072) -> RoadGraph:
    """One-way chain n0..nN that forks only on the final parallel edge pair.

    The chain e0..e(N-1) is a shared equal-cost prefix; from nN two parallel
    one-way edges (canonical-larger ``ez`` inserted first so its traversal
    occupies the earlier CSR position, then canonical-smaller ``ea``) reach
    nN+1 with equal duration.  Forward from n0 and reverse from nN+1 settle
    the whole chain, so the second label at nN+1 forces the native
    equal-cost path-key comparison of two length-(edge_count+1) chains that
    share the full edge_count-prefix - deep enough that a recursive
    comparator would exhaust the C stack.  The canonical-smaller ``ea``
    route must win exactly like Python tuple ordering.
    """
    graph = RoadGraph()
    for index in range(edge_count + 2):
        graph.add_node(Node(f"c{index}", 42.0 + 0.001 * index, -72.0))
    for index in range(edge_count):
        graph.add_edge(
            Edge(
                f"e{index}",
                f"c{index}",
                f"c{index + 1}",
                distance_km=0.2,
                scenic_score=5.0,
                road_type="secondary",
                speed_limit_kmh=50,
                one_way=True,
            )
        )
    # Parallel final edges in canonical order ea < ez; the larger id is
    # inserted first so CSR relaxes it before the smaller one.
    graph.add_edge(
        Edge(
            "ez",
            f"c{edge_count}",
            f"c{edge_count + 1}",
            distance_km=0.2,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "ea",
            f"c{edge_count}",
            f"c{edge_count + 1}",
            distance_km=0.2,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    return graph


def _split_meeting_nodes_graph() -> RoadGraph:
    """Two equal-duration routes whose forward paths share only the first
    edge and end at two distinct reverse seeds:

        n0 -e0-> nA -ez-> X -e1-> n5a
                     '--ea-> Y -e2-> n5b

    The ez/e1 route is canonical-larger but is inserted first, so CSR order
    favors it: nA's forward row relaxes ez before ea.  Its prefix is also
    *shorter* (ez costs 1.0 vs ea 2.0), so the forward search labels its
    meeting node first and its reverse-seed candidate is considered last.
    Both routes are equal-cost and each meets a *different* reverse seed, so
    every meeting candidate's middle key is a full route and the two
    reverse-seed candidates have empty reverse chains.  A middle-key
    comparator that stops after the shared first forward edge ties the
    seed candidates (e0 == e0, both reverse chains empty) and keeps the
    first-arriving larger ez route; only a full oldest-to-newest forward
    comparison (ez > ea) picks the ea route exactly like Python tuple
    ordering.
    """
    graph = RoadGraph()
    graph.add_node(Node("n0", 42.0, -72.0))
    graph.add_node(Node("nA", 42.01, -72.0))
    graph.add_node(Node("X", 42.02, -72.01))
    graph.add_node(Node("Y", 42.02, -71.99))
    graph.add_node(Node("n5a", 42.03, -72.0))
    graph.add_node(Node("n5b", 42.03, -72.02))
    graph.add_edge(
        Edge(
            "e0",
            "n0",
            "nA",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "ez",
            "nA",
            "X",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e1",
            "X",
            "n5a",
            distance_km=2.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "ea",
            "nA",
            "Y",
            distance_km=2.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e2",
            "Y",
            "n5b",
            distance_km=1.0,
            scenic_score=5.0,
            road_type="secondary",
            speed_limit_kmh=60,
            one_way=True,
        )
    )
    return graph


def test_compact_search_duplicate_node_seed_returns_winning_index(
    tmp_path: Path,
) -> None:
    """Duplicate seeds on the same node must resolve by (cost, rank): the
    actual winning seed index is returned, not the first match of a
    reconstruction rescan."""
    from src.route_planner._compact_search import run_compact_bidirectional_search

    graph = _long_chain_graph(4)
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()
    reverse = [(f"c{4}", 0.0, (0, 0), None)]

    # Later cheaper duplicate seed wins and its original index is returned.
    result = run_compact_bidirectional_search(
        topology,
        cost_function,
        [("c0", 5.0, (0, 0), None), ("c0", 1.0, (1, 0), None)],
        reverse,
    )
    assert result is not None
    positions, total_cost, fwd_index, rev_index = result
    assert positions == [0, 1, 2, 3]
    assert (fwd_index, rev_index) == (1, 0)
    # Total = winning seed cost (1.0) plus the selected path traversals.
    assert total_cost == pytest.approx(
        1.0 + float(sum(topology.travel_time_minutes[p] for p in positions))
    )

    # Cheaper first seed: the more expensive duplicate must not win.
    result = run_compact_bidirectional_search(
        topology,
        cost_function,
        [("c0", 0.0, (0, 0), None), ("c0", 5.0, (1, 0), None)],
        reverse,
    )
    assert result is not None
    assert result[2] == 0

    # Equal-cost duplicates with the same rank pair: first seed wins.
    result = run_compact_bidirectional_search(
        topology,
        cost_function,
        [("c0", 0.0, (0, 0), None), ("c0", 0.0, (0, 0), None)],
        reverse,
    )
    assert result is not None
    assert result[2] == 0

    # Equal cost, strictly smaller rank pair: the later seed wins.
    result = run_compact_bidirectional_search(
        topology,
        cost_function,
        [("c0", 0.0, (2, 0), None), ("c0", 0.0, (1, 0), None)],
        reverse,
    )
    assert result is not None
    assert result[2] == 1


def test_compact_equal_cost_paths_match_eager_ranked_ordering(
    tmp_path: Path,
) -> None:
    """Equal-total alternatives inserted in reverse CSR order must resolve by
    the Python ranked (cost, rank_key) ordering: the lexicographically
    smaller canonical edge-id sequence wins, deterministically, matching the
    eager reference route."""
    from src.route_planner._compact_search import run_compact_bidirectional_search

    graph = _diamond_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    reference = RoadGraph.load(sqlite_path)
    compact = CompactRoadGraph.load(manifest_path)
    start = (42.0, -72.0)
    end = (42.1, -72.05)

    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()

    results = []
    for _ in range(3):
        result = run_compact_bidirectional_search(
            topology,
            cost_function,
            [("n0", 0.0, (0, 0), None)],
            [("n3", 0.0, (0, 0), None)],
        )
        assert result is not None
        results.append(result)

    first = results[0]
    assert first[1] == pytest.approx(2.0)
    assert [topology.edge_refs[p][0] for p in first[0]] == ["e0", "e1"]
    assert (first[2], first[3]) == (0, 0)
    for result in results[1:]:
        assert result[0] == first[0]
        assert result[1] == first[1]

    compact_route = ScenicRoutePlanner(graph=compact).find_fastest_route(
        start, end, avoid_highways=False
    )
    reference_route = ScenicRoutePlanner(graph=reference).find_fastest_route(
        start, end, avoid_highways=False
    )
    assert tuple(compact_route.edge_ids) == ("e0", "e1")
    assert tuple(compact_route.edge_ids) == tuple(reference_route.edge_ids)
    assert compact_route.estimated_duration_minutes == pytest.approx(
        reference_route.estimated_duration_minutes
    )


def test_compact_shared_first_edge_suffix_tie_matches_eager(
    tmp_path: Path,
) -> None:
    """Equal-cost paths sharing the first edge but differing in a later
    suffix resolve by the full chronological edge sequence.

    The lexicographically larger suffix is inserted first (reverse canonical
    CSR order), so the native search labels the merge node with the larger
    path before the smaller one arrives.  The equal-cost replacement must
    compare every canonical edge id oldest-to-newest - not just the shared
    first edge - to pick e0/e1/e2 exactly like Python tuple ordering and the
    eager reference route."""
    from src.route_planner._compact_search import run_compact_bidirectional_search

    graph = _shared_prefix_merge_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    reference = RoadGraph.load(sqlite_path)
    compact = CompactRoadGraph.load(manifest_path)
    start = (42.0, -72.0)
    end = (42.04, -72.0)

    # The graph is inserted in reverse canonical order and contains both
    # equal-duration alternatives sharing the first edge e0.
    assert list(compact.edges) == ["e5", "e4", "e3", "e2", "e1", "e0"]
    for edge_ids in (("e0", "e1", "e2", "e5"), ("e0", "e3", "e4", "e5")):
        duration = sum(
            compact.edges[edge_id].distance_km
            / compact.edges[edge_id].speed_limit_kmh
            * 60.0
            for edge_id in edge_ids
        )
        assert duration == pytest.approx(103.0)
        for first_id, second_id in zip(edge_ids[:-1], edge_ids[1:]):
            assert compact.edges[first_id].end_node_id == compact.edges[
                second_id
            ].start_node_id

    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()

    results = []
    for _ in range(3):
        result = run_compact_bidirectional_search(
            topology,
            cost_function,
            [("n0", 0.0, (0, 0), None)],
            [("n5", 0.0, (0, 0), None)],
        )
        assert result is not None
        results.append(result)

    first = results[0]
    assert first[1] == pytest.approx(103.0)
    assert [topology.edge_refs[p][0] for p in first[0]] == [
        "e0",
        "e1",
        "e2",
        "e5",
    ]
    assert (first[2], first[3]) == (0, 0)
    for result in results[1:]:
        assert result[0] == first[0]
        assert result[1] == first[1]

    compact_route = ScenicRoutePlanner(graph=compact).find_fastest_route(
        start, end, avoid_highways=False
    )
    reference_route = ScenicRoutePlanner(graph=reference).find_fastest_route(
        start, end, avoid_highways=False
    )
    assert tuple(compact_route.edge_ids) == ("e0", "e1", "e2", "e5")
    assert tuple(compact_route.edge_ids) == tuple(reference_route.edge_ids)
    assert compact_route.estimated_duration_minutes == pytest.approx(
        reference_route.estimated_duration_minutes
    )


def test_compact_meeting_candidates_compare_full_forward_then_reverse(
    tmp_path: Path,
) -> None:
    """Equal-cost meeting candidates at *different* reverse seeds whose
    forward paths share only the first edge resolve by comparing every
    forward edge oldest-to-newest before the reverse side.

    The canonical-larger ez/e1 route is inserted first, so CSR order labels
    its meeting node before the ea/e2 route arrives.  Both reverse-seed
    candidates have empty reverse chains, so a comparator that stops after
    the shared first forward edge ties them and keeps the first-arriving
    larger ez route; the full forward comparison (ez > ea) must pick the ea
    route exactly like Python tuple ordering and the eager ranked search."""
    from src.route_planner._compact_search import run_compact_bidirectional_search

    graph = _split_meeting_nodes_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)

    # Canonical edge order must be e0 < e1 < e2 < ea < ez: the ez/e1 route
    # is the lexicographically larger alternative and it is inserted first.
    assert list(compact.edges) == ["e0", "ez", "e1", "ea", "e2"]
    for edge_ids in (("e0", "ez", "e1"), ("e0", "ea", "e2")):
        duration = sum(
            compact.edges[edge_id].distance_km
            / compact.edges[edge_id].speed_limit_kmh
            * 60.0
            for edge_id in edge_ids
        )
        assert duration == pytest.approx(4.0)
        for first_id, second_id in zip(edge_ids[:-1], edge_ids[1:]):
            assert compact.edges[first_id].end_node_id == compact.edges[
                second_id
            ].start_node_id

    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()

    results = []
    for _ in range(3):
        result = run_compact_bidirectional_search(
            topology,
            cost_function,
            [("n0", 0.0, (0, 0), None)],
            [
                ("n5a", 0.0, (0, 0), None),
                ("n5b", 0.0, (0, 0), None),
            ],
        )
        assert result is not None
        results.append(result)

    first = results[0]
    assert first[1] == pytest.approx(4.0)
    assert [topology.edge_refs[p][0] for p in first[0]] == ["e0", "ea", "e2"]
    assert (first[2], first[3]) == (0, 1)
    for result in results[1:]:
        assert result[0] == first[0]
        assert result[1] == first[1]
        assert (result[2], result[3]) == (first[2], first[3])

    # The eager ranked search over the compact graph is the reference: it
    # compares the full middle-key tuple and must pick the ea route too.
    reference_result = planner._bidirectional_search_core(
        cost_function,
        [("n0", 0.0, (), None)],
        [("n5a", 0.0, (), None), ("n5b", 0.0, (), None)],
    )
    assert reference_result is not None
    reference_path, reference_rank_key = reference_result
    assert tuple(edge.id for edge in reference_path) == ("e0", "ea", "e2")
    assert reference_rank_key[-1] == ("e0", "ea", "e2")


def test_compact_deep_equal_cost_shared_prefix_no_stack_overflow(
    tmp_path: Path,
) -> None:
    """Two equal-cost labels sharing an extremely deep prefix must compare
    with bounded C stack use.

    The forward search labels nN+1 once per parallel final edge; the second
    label forces the equal-cost path-key comparison of two
    length-(edge_count+1) chains that share the full edge_count-prefix.  A
    recursive comparator would nest once per record: at 262144 records that
    is over 12 MiB of call-stack frames (48 bytes per arm64 frame), far
    beyond the 8 MiB default thread stack.  The iterative comparator
    materializes the chains in bounded transient scratch instead, completes,
    and picks the canonical-smaller ``ea`` final edge exactly like Python
    tuple ordering."""
    from src.route_planner._compact_search import run_compact_bidirectional_search

    edge_count = 262144
    graph = _deep_prefix_parallel_end_graph(edge_count)
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None

    positions, total_cost, fwd_index, rev_index = (
        run_compact_bidirectional_search(
            topology,
            planner._make_fastest_cost_function(),
            [("c0", 0.0, (), None)],
            [(f"c{edge_count + 1}", 0.0, (), None)],
        )
    )
    assert (fwd_index, rev_index) == (0, 0)
    assert total_cost == pytest.approx(
        float(sum(topology.travel_time_minutes[p] for p in positions))
    )
    refs = [topology.edge_refs[p][0] for p in positions]
    assert refs == [f"e{index}" for index in range(edge_count)] + ["ea"]
    assert compact.edges[refs[0]].start_node_id == "c0"
    assert compact.edges[refs[-1]].end_node_id == f"c{edge_count + 1}"
    for first_id, second_id in zip(refs[:-1], refs[1:]):
        assert compact.edges[first_id].end_node_id == compact.edges[
            second_id
        ].start_node_id


def test_compact_long_meeting_reconstruction_is_connected_with_eager_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compact bidirectional search must reconstruct one connected
    directed traversal when both meeting sides exceed the former fixed
    1024-position buffers, matching the eager reference route exactly."""
    import src.route_planner.planner as planner_module
    from src.route_planner._compact_search import (
        run_compact_bidirectional_search,
    )

    edge_count = 2600
    graph = _long_chain_graph(edge_count)
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    reference = RoadGraph.load(sqlite_path)
    compact = CompactRoadGraph.load(manifest_path)

    start = (42.0, -72.0)
    end = (42.0 + 0.001 * edge_count, -72.0)

    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    positions, total_cost, fwd_index, rev_index = (
        run_compact_bidirectional_search(
            topology,
            planner._make_fastest_cost_function(),
            [("c0", 0.0, (), None)],
            [(f"c{edge_count}", 0.0, (), None)],
        )
    )
    # The meeting sits mid-chain (1300/1300), so every traversal must be
    # reconstructed from both sides; the emitted sequence is the full chain.
    assert positions == list(range(edge_count))
    assert (fwd_index, rev_index) == (0, 0)
    assert total_cost == pytest.approx(
        float(sum(topology.travel_time_minutes))
    )
    # Every consecutive position is the exact next directed traversal.
    for position in positions:
        edge_id, reverse = topology.edge_refs[position]
        assert edge_id == f"e{position}"
        assert reverse is False

    calls: list[object] = []

    def capturing_search(*args: object, **kwargs: object) -> object:
        result = run_compact_bidirectional_search(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(
        planner_module,
        "run_compact_bidirectional_search",
        capturing_search,
    )
    compact_route = ScenicRoutePlanner(graph=compact).find_fastest_route(
        start, end, avoid_highways=False
    )
    assert calls and calls[0] is not None
    assert tuple(calls[0][0]) == tuple(range(edge_count))
    reference_route = ScenicRoutePlanner(graph=reference).find_fastest_route(
        start, end, avoid_highways=False
    )
    assert tuple(compact_route.edge_ids) == tuple(reference_route.edge_ids)
    assert tuple(compact_route.edge_ids) == tuple(
        f"e{index}" for index in range(edge_count)
    )
    assert compact_route.estimated_duration_minutes == pytest.approx(
        reference_route.estimated_duration_minutes
    )
    assert compact_route.exactness_status == "exact"
    assert compact_route.waypoints[0] == start
    assert compact_route.waypoints[-1] == end
    # Every consecutive directed edge connects end -> start of the next.
    for first_id, second_id in zip(
        compact_route.edge_ids[:-1], compact_route.edge_ids[1:]
    ):
        first = compact.edges[first_id]
        second = compact.edges[second_id]
        assert first.end_node_id == second.start_node_id
    assert compact.edges[compact_route.edge_ids[0]].start_node_id == "c0"
    assert (
        compact.edges[compact_route.edge_ids[-1]].end_node_id
        == f"c{edge_count}"
    )


def test_compact_search_native_deadline_interrupts_long_search(
    tmp_path: Path,
) -> None:
    """A compiled compact search must fail closed inside the native loop when
    its CPU budget expires, instead of running to completion first.

    The search chain is long enough that expanding both frontiers takes
    orders of magnitude longer than the requested budget, so a deadline that
    is only checked before or after the call would let the search complete
    and return a route; the wrapper must raise RoutingTimeout mid-search.
    The unbounded call keeps returning the exact full chain, so the timeout
    path stays distinct from generic search failure.
    """
    from src.route_planner._compact_search import (
        run_compact_bidirectional_search,
    )

    edge_count = 100_000
    graph = _long_chain_graph(edge_count)
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)

    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()
    forward_seeds = [("c0", 0.0, (), None)]
    reverse_seeds = [(f"c{edge_count}", 0.0, (), None)]

    # Unbounded parity baseline: the full directed chain comes back exactly.
    positions, total_cost, fwd_index, rev_index = (
        run_compact_bidirectional_search(
            topology, cost_function, forward_seeds, reverse_seeds
        )
    )
    assert positions == list(range(edge_count))
    assert (fwd_index, rev_index) == (0, 0)
    assert total_cost == pytest.approx(
        float(sum(topology.travel_time_minutes))
    )

    # Zero budget: the native call must fail closed immediately, never
    # returning a route and never falling through to a Python fallback.
    with pytest.raises(RoutingTimeout, match="deadline"):
        run_compact_bidirectional_search(
            topology,
            cost_function,
            forward_seeds,
            reverse_seeds,
            deadline_seconds=0.0,
        )

    # Tiny positive budget: the search loop runs for far longer than this, so
    # a deadline enforced only before/after the call would still complete and
    # return positions here; only an in-loop check raises.
    with pytest.raises(RoutingTimeout, match="deadline"):
        run_compact_bidirectional_search(
            topology,
            cost_function,
            forward_seeds,
            reverse_seeds,
            deadline_seconds=0.0005,
        )


def test_compact_route_longer_than_legacy_output_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route longer than the former fixed 4096-position output cap must be
    returned in full by the compact search, with no Python fallback."""
    import src.route_planner.planner as planner_module
    from src.route_planner._compact_search import (
        run_compact_bidirectional_search,
    )

    edge_count = 8192
    assert edge_count > 4096
    graph = _long_chain_graph(edge_count)
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    reference = RoadGraph.load(sqlite_path)
    compact = CompactRoadGraph.load(manifest_path)

    start = (42.0, -72.0)
    end = (42.0 + 0.001 * edge_count, -72.0)

    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    positions, total_cost, fwd_index, rev_index = (
        run_compact_bidirectional_search(
            topology,
            planner._make_fastest_cost_function(),
            [("c0", 0.0, (), None)],
            [(f"c{edge_count}", 0.0, (), None)],
        )
    )
    # The full chain must come back: the old fixed 4096 cap made the wrapper
    # return None here and forced the eager Python fallback.
    assert len(positions) > 4096
    assert positions == list(range(edge_count))
    assert (fwd_index, rev_index) == (0, 0)
    assert total_cost == pytest.approx(
        float(sum(topology.travel_time_minutes))
    )

    calls: list[object] = []

    def capturing_search(*args: object, **kwargs: object) -> object:
        result = run_compact_bidirectional_search(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(
        planner_module,
        "run_compact_bidirectional_search",
        capturing_search,
    )
    compact_route = ScenicRoutePlanner(graph=compact).find_fastest_route(
        start, end, avoid_highways=False
    )
    assert calls and calls[0] is not None
    assert tuple(calls[0][0]) == tuple(range(edge_count))
    reference_route = ScenicRoutePlanner(graph=reference).find_fastest_route(
        start, end, avoid_highways=False
    )
    assert tuple(compact_route.edge_ids) == tuple(reference_route.edge_ids)
    assert tuple(compact_route.edge_ids) == tuple(
        f"e{index}" for index in range(edge_count)
    )
    assert compact_route.estimated_duration_minutes == pytest.approx(
        reference_route.estimated_duration_minutes
    )
    assert compact_route.exactness_status == "exact"
    assert compact_route.waypoints[0] == start
    assert compact_route.waypoints[-1] == end

def _fixture_graph_b() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node("bn0", 43.0, -71.0))
    graph.add_node(Node("bn1", 43.05, -71.0))
    graph.add_edge(
        Edge(
            "be0",
            "bn0",
            "bn1",
            distance_km=10.0,
            scenic_score=4.0,
            road_name="North Rd",
            road_type="primary",
            speed_limit_kmh=60,
            one_way=False,
        )
    )
    return graph


def _fixture_graph_c() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node("cn0", 44.0, -70.0))
    graph.add_node(Node("cn1", 44.05, -70.0))
    graph.add_edge(
        Edge(
            "ce0",
            "cn0",
            "cn1",
            distance_km=12.0,
            scenic_score=6.0,
            road_name="East Rd",
            road_type="trunk",
            speed_limit_kmh=90,
            one_way=False,
        )
    )
    return graph


def test_dual_compact_graph_preloading_and_scoring_independence(
    tmp_path: Path,
) -> None:
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_a.mkdir()
    dir_b.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a_path, _ = _tile_report(dir_a, graph_a, {"e0": 9.0, "e1": 8.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b_path, _ = _tile_report(dir_b, graph_b, {"be0": 2.0})

    route_service.clear_route_caches()

    res_a = route_service.preload_route_assets(manifest_a, report_a_path)
    assert res_a["graph_cache_hit"] is False
    assert len(route_service._GRAPH_CACHE) == 1
    assert len(route_service._ACTIVE_GRAPH_VARIANT_KEYS) == 1

    res_b = route_service.preload_route_assets(manifest_b, report_b_path)
    assert res_b["graph_cache_hit"] is False
    assert len(route_service._GRAPH_CACHE) == 2
    assert len(route_service._SCORED_GRAPH_CACHE) == 2
    assert len(route_service._ACTIVE_GRAPH_VARIANT_KEYS) == 2

    # Second preload call for A hits both graph and scored graph caches without evicting B
    res_a2 = route_service.preload_route_assets(manifest_a, report_a_path)
    assert res_a2["graph_cache_hit"] is True
    assert res_a2["scored_graph_cache_hit"] is True
    assert len(route_service._GRAPH_CACHE) == 2
    assert len(route_service._SCORED_GRAPH_CACHE) == 2
    assert len(route_service._ACTIVE_GRAPH_VARIANT_KEYS) == 2


def test_dual_compact_graph_report_switching_isolation(tmp_path: Path) -> None:
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_a.mkdir()
    dir_b.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a1_tmp, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})
    report_a1 = dir_a / "report_a1.json"
    report_a1_tmp.rename(report_a1)
    report_a2, _ = _tile_report(dir_a, graph_a, {"e0": 3.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    route_service.clear_route_caches()

    route_service.preload_route_assets(manifest_a, report_a1)
    route_service.preload_route_assets(manifest_b, report_b)
    assert len(route_service._GRAPH_CACHE) == 2

    # Switch report for Graph A
    route_service.preload_route_assets(manifest_a, report_a2)
    assert len(route_service._GRAPH_CACHE) == 2
    assert len(route_service._SCORED_GRAPH_CACHE) == 2

    # Verify Graph B is still cached and valid
    res_b = route_service.preload_route_assets(manifest_b, report_b)
    assert res_b["graph_cache_hit"] is True
    assert res_b["scored_graph_cache_hit"] is True


def test_dual_compact_graph_stale_invalidation_isolation(
    tmp_path: Path,
) -> None:
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_a.mkdir()
    dir_b.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    route_service.clear_route_caches()

    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)

    # Touch manifest_a to change mtime / file signature
    new_mtime = manifest_a.stat().st_mtime_ns + 10_000_000
    import os

    os.utime(manifest_a, ns=(new_mtime, new_mtime))

    # Graph A reloaded because signature changed
    res_a = route_service.preload_route_assets(manifest_a, report_a)
    assert res_a["graph_cache_hit"] is False

    # Graph B remained cached
    res_b = route_service.preload_route_assets(manifest_b, report_b)
    assert res_b["graph_cache_hit"] is True
    assert res_b["scored_graph_cache_hit"] is True


def test_dual_compact_graph_lru_eviction(tmp_path: Path) -> None:
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_c = tmp_path / "graph_c"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_c.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    graph_c = _fixture_graph_c()
    _, manifest_c, _ = _publish_compact(dir_c, graph_c)
    report_c, _ = _tile_report(dir_c, graph_c, {"ce0": 7.0})

    route_service.clear_route_caches()

    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)
    assert len(route_service._GRAPH_CACHE) == 2

    # Preloading graph C evicts oldest graph A coherently
    route_service.preload_route_assets(manifest_c, report_c)
    assert len(route_service._GRAPH_CACHE) == 2
    assert len(route_service._SCORED_GRAPH_CACHE) == 2

    # Graph A is now a cache miss
    res_a = route_service.preload_route_assets(manifest_a, report_a)
    assert res_a["graph_cache_hit"] is False

    # Graph C and Graph A are now in cache, Graph B was evicted as LRU
    res_c = route_service.preload_route_assets(manifest_c, report_c)
    assert res_c["graph_cache_hit"] is True


def test_dual_compact_graph_clear_route_caches(tmp_path: Path) -> None:
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_a.mkdir()
    dir_b.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)

    assert len(route_service._GRAPH_CACHE) == 2
    assert len(route_service._TILE_SCORE_CACHE) == 2
    assert len(route_service._SCORED_GRAPH_CACHE) == 2
    assert len(route_service._ACTIVE_GRAPH_VARIANT_KEYS) == 2

    route_service.clear_route_caches()

    assert len(route_service._GRAPH_CACHE) == 0
    assert len(route_service._TILE_SCORE_CACHE) == 0
    assert len(route_service._SCORED_GRAPH_CACHE) == 0
    assert len(route_service._ACTIVE_GRAPH_VARIANT_KEYS) == 0


def test_compact_cache_clear_closes_final_graphs_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_counts = _spy_compact_resource_closes(monkeypatch)
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_a.mkdir()
    dir_b.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    route_service.clear_route_caches()
    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)

    resources_by_graph = [
        _capture_compact_resources(graph)
        for graph in route_service._GRAPH_CACHE.values()
    ]
    assert len(resources_by_graph) == 2
    for resources in resources_by_graph:
        _assert_resources_open(resources)

    # Clear drops the final cache reference to every retained graph; each
    # graph's resources must be closed exactly once, never twice.
    route_service.clear_route_caches()

    assert len(route_service._GRAPH_CACHE) == 0
    assert len(route_service._SCORED_GRAPH_CACHE) == 0
    for resources in resources_by_graph:
        _assert_resources_closed(resources)
        assert close_counts.get(id(resources["bin_mmap"])) == 1
    assert all(count == 1 for count in close_counts.values())


def test_compact_scored_invalidation_closes_evicted_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    close_counts = _spy_compact_resource_closes(monkeypatch)
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_a.mkdir()
    dir_b.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    route_service.clear_route_caches()
    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)

    graph_key_a = (_resolved_path_key(manifest_a), _file_signature(manifest_a))
    variant_a_resources = _capture_compact_resources(
        route_service._GRAPH_CACHE[graph_key_a]
    )
    variant_b_resources = [
        _capture_compact_resources(graph)
        for gkey, graph in route_service._GRAPH_CACHE.items()
        if gkey != graph_key_a
    ]
    assert len(variant_b_resources) == 1
    _assert_resources_open(variant_a_resources)
    _assert_resources_open(variant_b_resources[0])

    # A changed report signature invalidates every scored variant keyed on
    # the old report while leaving graph B's variant retained and open.
    new_mtime = report_a.stat().st_mtime_ns + 10_000_000
    os.utime(report_a, ns=(new_mtime, new_mtime))
    route_service.preload_route_assets(manifest_a, report_a)

    _assert_resources_closed(variant_a_resources)
    assert close_counts.get(id(variant_a_resources["bin_mmap"])) == 1
    _assert_resources_open(variant_b_resources[0])
    assert id(variant_b_resources[0]["bin_mmap"]) not in close_counts
    assert all(count == 1 for count in close_counts.values())


def test_compact_scored_replacement_closes_superseded_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_counts = _spy_compact_resource_closes(monkeypatch)
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_a.mkdir()
    dir_b.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    route_service.clear_route_caches()
    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)

    graph_key_a = (_resolved_path_key(manifest_a), _file_signature(manifest_a))
    variant_a_resources = _capture_compact_resources(
        route_service._GRAPH_CACHE[graph_key_a]
    )
    variant_b_resources = [
        _capture_compact_resources(graph)
        for gkey, graph in route_service._GRAPH_CACHE.items()
        if gkey != graph_key_a
    ]
    assert len(variant_b_resources) == 1
    _assert_resources_open(variant_a_resources)
    _assert_resources_open(variant_b_resources[0])

    # The same graph and report with a different fallback is a fresh scored
    # key, so the active variant is superseded and released; graph B's
    # variant stays cached and open.
    route_service.preload_route_assets(
        manifest_a, report_a, tile_score_fallback=4.0
    )

    _assert_resources_closed(variant_a_resources)
    assert close_counts.get(id(variant_a_resources["bin_mmap"])) == 1
    _assert_resources_open(variant_b_resources[0])
    assert id(variant_b_resources[0]["bin_mmap"]) not in close_counts
    assert all(count == 1 for count in close_counts.values())


def test_compact_third_graph_lru_eviction_closes_evicted_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_counts = _spy_compact_resource_closes(monkeypatch)
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_c = tmp_path / "graph_c"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_c.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    graph_c = _fixture_graph_c()
    _, manifest_c, _ = _publish_compact(dir_c, graph_c)
    report_c, _ = _tile_report(dir_c, graph_c, {"ce0": 7.0})

    route_service.clear_route_caches()
    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)

    graph_key_a = (_resolved_path_key(manifest_a), _file_signature(manifest_a))
    graph_key_b = (_resolved_path_key(manifest_b), _file_signature(manifest_b))
    variant_a_resources = _capture_compact_resources(
        route_service._GRAPH_CACHE[graph_key_a]
    )
    variant_b_resources = _capture_compact_resources(
        route_service._GRAPH_CACHE[graph_key_b]
    )
    _assert_resources_open(variant_a_resources)
    _assert_resources_open(variant_b_resources)

    # Preloading the third graph evicts the least-recently-used graph A and
    # closes it exactly once; retained graphs B and C stay open.
    route_service.preload_route_assets(manifest_c, report_c)

    _assert_resources_closed(variant_a_resources)
    assert close_counts.get(id(variant_a_resources["bin_mmap"])) == 1
    _assert_resources_open(variant_b_resources)
    assert id(variant_b_resources["bin_mmap"]) not in close_counts
    assert all(count == 1 for count in close_counts.values())


def test_compact_scored_cache_lru_eviction_closes_evicted_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_counts = _spy_compact_resource_closes(monkeypatch)
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_c = tmp_path / "graph_c"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_c.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    graph_c = _fixture_graph_c()
    _, manifest_c, _ = _publish_compact(dir_c, graph_c)

    # Graphs A and B share one report so their scored variants outnumber the
    # scored-cache capacity once graph C is preloaded.
    report_r, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})
    report_c, _ = _tile_report(dir_c, graph_c, {"ce0": 7.0})

    route_service.clear_route_caches()
    route_service.preload_route_assets(manifest_a, report_r)
    route_service.preload_route_assets(manifest_b, report_r)

    graph_key_a = (_resolved_path_key(manifest_a), _file_signature(manifest_a))
    graph_key_b = (_resolved_path_key(manifest_b), _file_signature(manifest_b))
    variant_a_resources = _capture_compact_resources(
        route_service._GRAPH_CACHE[graph_key_a]
    )
    variant_b_resources = _capture_compact_resources(
        route_service._GRAPH_CACHE[graph_key_b]
    )
    _assert_resources_open(variant_a_resources)
    _assert_resources_open(variant_b_resources)

    # The third scored variant exceeds scored-cache capacity, evicting the
    # least-recently-used scored variant for graph A.
    route_service.preload_route_assets(manifest_c, report_c)

    _assert_resources_closed(variant_a_resources)
    assert close_counts.get(id(variant_a_resources["bin_mmap"])) == 1
    _assert_resources_open(variant_b_resources)
    assert id(variant_b_resources["bin_mmap"]) not in close_counts
    graph_key_c = (_resolved_path_key(manifest_c), _file_signature(manifest_c))
    _assert_resources_open(
        _capture_compact_resources(route_service._GRAPH_CACHE[graph_key_c])
    )
    assert all(count == 1 for count in close_counts.values())


def test_compact_eviction_defers_close_until_final_reference_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    close_counts = _spy_compact_resource_closes(monkeypatch)
    dir_a = tmp_path / "graph_a"
    dir_b = tmp_path / "graph_b"
    dir_c = tmp_path / "graph_c"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_c.mkdir()

    graph_a = _fixture_graph()
    _, manifest_a, _ = _publish_compact(dir_a, graph_a)
    report_a, _ = _tile_report(dir_a, graph_a, {"e0": 9.0})

    graph_b = _fixture_graph_b()
    _, manifest_b, _ = _publish_compact(dir_b, graph_b)
    report_b, _ = _tile_report(dir_b, graph_b, {"be0": 5.0})

    graph_c = _fixture_graph_c()
    _, manifest_c, _ = _publish_compact(dir_c, graph_c)
    report_c, _ = _tile_report(dir_c, graph_c, {"ce0": 7.0})

    route_service.clear_route_caches()
    route_service.preload_route_assets(manifest_a, report_a)
    route_service.preload_route_assets(manifest_b, report_b)

    graph_key_a = (_resolved_path_key(manifest_a), _file_signature(manifest_a))
    in_flight = route_service._GRAPH_CACHE[graph_key_a]
    variant_a_resources = _capture_compact_resources(in_flight)

    # Eviction drops the cache references, but the in-flight request reference
    # keeps the graph alive: no premature close, resources stay usable.
    route_service.preload_route_assets(manifest_c, report_c)

    assert id(variant_a_resources["bin_mmap"]) not in close_counts
    _assert_resources_open(variant_a_resources)
    assert len(in_flight._sections) > 0
    assert in_flight.edge_projection_index_status["state"] == "loaded"

    # Releasing the final reference makes the graph's reference web
    # unreachable; collecting it fires the finalizer exactly once.
    del in_flight
    gc.collect()

    _assert_resources_closed(variant_a_resources)
    assert close_counts.get(id(variant_a_resources["bin_mmap"])) == 1
    assert all(count == 1 for count in close_counts.values())



def _native_library():
    from src.route_planner._compact_search import _ensure_compiled_library

    lib = _ensure_compiled_library()
    if lib is None:
        pytest.skip("compiled compact search unavailable")
    return lib


def _native_cost_spec(planner) -> object:
    """Replicate the wrapper's CostSpec construction for the fastest
    cost function, so native-level probes exercise the same parameters."""
    import ctypes as _ctypes

    from src.route_planner._compact_search import _CostSpec

    cost_function = planner._make_fastest_cost_function()
    weights = getattr(cost_function, "weights", None)
    return _CostSpec(
        scenic_weight=float(getattr(cost_function, "scenic_weight", 0.0)),
        strict_highways=(
            1 if getattr(cost_function, "strict_highways", False) else 0
        ),
        highway_preference=float(
            getattr(cost_function, "highway_preference", 0.0)
        ),
        travel_weight=(
            float(getattr(weights, "travel_time", 1.0)) if weights else 1.0
        ),
        scenic_reward=(
            float(getattr(weights, "scenic_reward", 0.0)) if weights else 0.0
        ),
        highway_penalty=(
            float(getattr(weights, "highway_penalty", 0.0)) if weights else 0.0
        ),
        scenic_byway_bonus=(
            float(getattr(weights, "scenic_byway_bonus", 0.0))
            if weights
            else 0.0
        ),
        lagrangian_multiplier=0.0,
        cost_limit=0.0,
    )


def _scenic_cost_spec() -> object:
    """CostSpec matching ScenicRoutePlanner._make_cost_function(0.8) on a
    default-weights planner (travel_time=1, scenic_reward=1)."""
    import ctypes as _ctypes

    from src.route_planner._compact_search import _CostSpec

    return _CostSpec(
        scenic_weight=0.8,
        strict_highways=0,
        highway_preference=0.0,
        travel_weight=1.0,
        scenic_reward=1.0,
        highway_penalty=2.0,
        scenic_byway_bonus=1.0,
        lagrangian_multiplier=0.0,
        cost_limit=0.0,
    )


def _rank_permutation_graph() -> RoadGraph:
    """Four one-way edges whose CSR traversal order permutes the canonical
    edge ranks: positions [e1, e3, e2, e0] vs ranks [0, 1, 2, 3].  Route A
    is s->a->t (e1, e2) and route B is s->b->t (e3, e0); every traversal
    lasts exactly one minute."""
    graph = RoadGraph()
    graph.add_node(Node("n0", 42.0, -72.0))  # s (rank 0)
    graph.add_node(Node("n1", 42.05, -72.0))  # t (rank 1)
    graph.add_node(Node("n2", 42.1, -72.0))  # a (rank 2)
    graph.add_node(Node("n3", 42.1, -72.1))  # b (rank 3)
    # Edge insertion order defines canonical edge ranks.
    graph.add_edge(
        Edge(
            "e0", "n3", "n1",
            distance_km=1.0, scenic_score=10.0,
            road_type="secondary", speed_limit_kmh=60, one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e1", "n0", "n2",
            distance_km=1.0, scenic_score=2.0,
            road_type="secondary", speed_limit_kmh=60, one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e2", "n2", "n1",
            distance_km=1.0, scenic_score=8.0,
            road_type="secondary", speed_limit_kmh=60, one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            "e3", "n0", "n3",
            distance_km=1.0, scenic_score=2.0,
            road_type="secondary", speed_limit_kmh=60, one_way=True,
        )
    )
    return graph


def _native_graph_spec(compact: CompactRoadGraph, arrays) -> object:
    """Build a _CompactGraphSpec from raw numpy sections so probes can pass
    tampered or synthetic rank data straight into the native search."""
    import ctypes as _ctypes

    import numpy as _np

    from src.route_planner._compact_search import _CompactGraphSpec

    def get_ptr(name: str, ctype) -> object:
        arr = arrays.get(name)
        if arr is None or not hasattr(arr, "ctypes"):
            return None
        return arr.ctypes.data_as(_ctypes.POINTER(ctype))

    return _CompactGraphSpec(
        node_count=int(compact.node_count),
        edge_count=int(compact.edge_count),
        traversal_count=int(compact.traversal_count),
        forward_indptr=get_ptr("forward_indptr", _ctypes.c_int64),
        forward_indices=get_ptr("forward_indices", _ctypes.c_int32),
        reverse_indptr=get_ptr("reverse_indptr", _ctypes.c_int64),
        reverse_indices=get_ptr("reverse_indices", _ctypes.c_int32),
        reverse_positions=get_ptr("reverse_positions", _ctypes.c_int64),
        trav_travel_time=get_ptr(
            "trav_travel_time_minutes", _ctypes.c_double
        ),
        trav_highway_mask=get_ptr("trav_highway_mask", _ctypes.c_uint8),
        trav_scenic_score=get_ptr("trav_scenic_score", _ctypes.c_double),
        trav_scenic_byway_mask=get_ptr(
            "trav_scenic_byway_mask", _ctypes.c_uint8
        ),
        trav_edge_rank=get_ptr("trav_edge_rank", _ctypes.c_int32),
        trav_reverse=get_ptr("trav_reverse", _ctypes.c_uint8),
        edge_id_strings=(
            _np.frombuffer(
                arrays["edge_id_strings"], dtype=_np.uint8
            ).ctypes.data_as(_ctypes.POINTER(_ctypes.c_uint8))
            if arrays.get("edge_id_strings") is not None
            else None
        ),
        edge_id_offsets=get_ptr("edge_id_offsets", _ctypes.c_int64),
    )


def _call_native_search(
    lib,
    spec,
    cost_spec,
    fwd_seeds,
    rev_seeds,
    deadline_seconds: float = -1.0,
):
    """Direct native call with explicit (rank, cost) seed pairs; returns
    (rc, positions, out_cost, out_fwd_idx, out_rev_idx)."""
    import ctypes as _ctypes

    fwd_nodes = (_ctypes.c_int32 * len(fwd_seeds))()
    fwd_costs = (_ctypes.c_double * len(fwd_seeds))()
    for i, (rank, cost) in enumerate(fwd_seeds):
        fwd_nodes[i] = int(rank)
        fwd_costs[i] = float(cost)
    rev_nodes = (_ctypes.c_int32 * len(rev_seeds))()
    rev_costs = (_ctypes.c_double * len(rev_seeds))()
    for i, (rank, cost) in enumerate(rev_seeds):
        rev_nodes[i] = int(rank)
        rev_costs[i] = float(cost)

    out_pos_ptr = _ctypes.POINTER(_ctypes.c_int64)()
    out_cost = _ctypes.c_double(0.0)
    out_fwd = _ctypes.c_int32(-1)
    out_rev = _ctypes.c_int32(-1)
    rc = lib.compact_bidirectional_search_alloc(
        _ctypes.byref(spec),
        _ctypes.byref(cost_spec),
        fwd_nodes,
        fwd_costs,
        len(fwd_seeds),
        rev_nodes,
        rev_costs,
        len(rev_seeds),
        _ctypes.byref(out_pos_ptr),
        _ctypes.byref(out_cost),
        _ctypes.byref(out_fwd),
        _ctypes.byref(out_rev),
        float(deadline_seconds),
    )
    positions = None
    if rc >= 0:
        positions = [int(out_pos_ptr[i]) for i in range(rc)]
        lib.compact_free_positions(out_pos_ptr)
    return rc, positions, float(out_cost.value), int(out_fwd.value), int(
        out_rev.value
    )


def _call_native_search_ranked(
    lib,
    spec,
    cost_spec,
    fwd_seeds,
    rev_seeds,
    deadline_seconds: float = -1.0,
):
    """Direct ranked-entry call with (rank, cost, rank_primary,
    rank_secondary) seed tuples; returns (rc, positions, out_cost,
    out_fwd_idx, out_rev_idx)."""
    import ctypes as _ctypes

    fwd_nodes = (_ctypes.c_int32 * len(fwd_seeds))()
    fwd_costs = (_ctypes.c_double * len(fwd_seeds))()
    fwd_primary = (_ctypes.c_int32 * len(fwd_seeds))()
    fwd_secondary = (_ctypes.c_int32 * len(fwd_seeds))()
    for i, (rank, cost, primary, secondary) in enumerate(fwd_seeds):
        fwd_nodes[i] = int(rank)
        fwd_costs[i] = float(cost)
        fwd_primary[i] = int(primary)
        fwd_secondary[i] = int(secondary)
    rev_nodes = (_ctypes.c_int32 * len(rev_seeds))()
    rev_costs = (_ctypes.c_double * len(rev_seeds))()
    rev_primary = (_ctypes.c_int32 * len(rev_seeds))()
    rev_secondary = (_ctypes.c_int32 * len(rev_seeds))()
    for i, (rank, cost, primary, secondary) in enumerate(rev_seeds):
        rev_nodes[i] = int(rank)
        rev_costs[i] = float(cost)
        rev_primary[i] = int(primary)
        rev_secondary[i] = int(secondary)

    out_pos_ptr = _ctypes.POINTER(_ctypes.c_int64)()
    out_cost = _ctypes.c_double(0.0)
    out_fwd = _ctypes.c_int32(-1)
    out_rev = _ctypes.c_int32(-1)
    rc = lib.compact_bidirectional_search_alloc_ranked(
        _ctypes.byref(spec),
        _ctypes.byref(cost_spec),
        fwd_nodes,
        fwd_costs,
        fwd_primary,
        fwd_secondary,
        len(fwd_seeds),
        rev_nodes,
        rev_costs,
        rev_primary,
        rev_secondary,
        len(rev_seeds),
        _ctypes.byref(out_pos_ptr),
        _ctypes.byref(out_cost),
        _ctypes.byref(out_fwd),
        _ctypes.byref(out_rev),
        float(deadline_seconds),
    )
    positions = None
    if rc >= 0:
        positions = [int(out_pos_ptr[i]) for i in range(rc)]
        lib.compact_free_positions(out_pos_ptr)
    return rc, positions, float(out_cost.value), int(out_fwd.value), int(
        out_rev.value
    )


def _call_native_search_edge_scores(
    lib,
    spec,
    cost_spec,
    edge_scenic_score_by_rank,
    fwd_seeds,
    rev_seeds,
    deadline_seconds: float = -1.0,
):
    """Direct edge-score-sidecar entry call with (rank, cost, rank_primary,
    rank_secondary) seed tuples plus the canonical-edge-rank-indexed scenic
    score array; returns (rc, positions, out_cost, out_fwd_idx, out_rev_idx)."""
    import ctypes as _ctypes

    fwd_nodes = (_ctypes.c_int32 * len(fwd_seeds))()
    fwd_costs = (_ctypes.c_double * len(fwd_seeds))()
    fwd_primary = (_ctypes.c_int32 * len(fwd_seeds))()
    fwd_secondary = (_ctypes.c_int32 * len(fwd_seeds))()
    for i, (rank, cost, primary, secondary) in enumerate(fwd_seeds):
        fwd_nodes[i] = int(rank)
        fwd_costs[i] = float(cost)
        fwd_primary[i] = int(primary)
        fwd_secondary[i] = int(secondary)
    rev_nodes = (_ctypes.c_int32 * len(rev_seeds))()
    rev_costs = (_ctypes.c_double * len(rev_seeds))()
    rev_primary = (_ctypes.c_int32 * len(rev_seeds))()
    rev_secondary = (_ctypes.c_int32 * len(rev_seeds))()
    for i, (rank, cost, primary, secondary) in enumerate(rev_seeds):
        rev_nodes[i] = int(rank)
        rev_costs[i] = float(cost)
        rev_primary[i] = int(primary)
        rev_secondary[i] = int(secondary)

    out_pos_ptr = _ctypes.POINTER(_ctypes.c_int64)()
    out_cost = _ctypes.c_double(0.0)
    out_fwd = _ctypes.c_int32(-1)
    out_rev = _ctypes.c_int32(-1)
    rc = lib.compact_bidirectional_search_alloc_ranked_edge_scores(
        _ctypes.byref(spec),
        _ctypes.byref(cost_spec),
        edge_scenic_score_by_rank,
        fwd_nodes,
        fwd_costs,
        fwd_primary,
        fwd_secondary,
        len(fwd_seeds),
        rev_nodes,
        rev_costs,
        rev_primary,
        rev_secondary,
        len(rev_seeds),
        _ctypes.byref(out_pos_ptr),
        _ctypes.byref(out_cost),
        _ctypes.byref(out_fwd),
        _ctypes.byref(out_rev),
        float(deadline_seconds),
    )
    positions = None
    if rc >= 0:
        positions = [int(out_pos_ptr[i]) for i in range(rc)]
        lib.compact_free_positions(out_pos_ptr)
    return rc, positions, float(out_cost.value), int(out_fwd.value), int(
        out_rev.value
    )


def test_compact_native_ranked_duplicate_seed_returns_winning_index(
    tmp_path: Path,
) -> None:
    """The ranked native entry must select duplicate-node seeds by
    (cost, rank) and return the actual winning compact seed index, not the
    first input match."""
    lib = _native_library()
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    cost_spec = _native_cost_spec(planner)
    arrays = dict(compact._sections)
    spec = _native_graph_spec(compact, arrays)

    # Later cheaper duplicate seed (rank 0) wins over the expensive first.
    rc, positions, cost, fwd_idx, rev_idx = _call_native_search_ranked(
        lib,
        spec,
        cost_spec,
        [(0, 5.0, 0, 0), (0, 1.0, 1, 0)],
        [(3, 0.0, 0, 0)],
    )
    assert rc >= 0
    assert positions is not None
    topology = planner._csr_topology()
    assert topology is not None
    refs = [topology.edge_refs[p][0] for p in positions]
    assert refs[0] == "e0"  # starts at n0 through the winning seed
    assert (fwd_idx, rev_idx) == (1, 0)
    # Total = winning seed cost (1.0) plus the selected path traversals.
    assert cost == pytest.approx(
        1.0 + float(sum(topology.travel_time_minutes[p] for p in positions))
    )

    # Equal cost, strictly smaller rank pair: the later duplicate wins.
    rc, positions, _cost, fwd_idx, _rev_idx = _call_native_search_ranked(
        lib,
        spec,
        cost_spec,
        [(0, 0.0, 2, 0), (0, 0.0, 1, 0)],
        [(3, 0.0, 0, 0)],
    )
    assert rc >= 0
    assert positions is not None
    assert fwd_idx == 1

    # Equal cost, equal rank pair: the first duplicate wins.
    rc, positions, _cost, fwd_idx, _rev_idx = _call_native_search_ranked(
        lib,
        spec,
        cost_spec,
        [(0, 0.0, 0, 0), (0, 0.0, 0, 0)],
        [(3, 0.0, 0, 0)],
    )
    assert rc >= 0
    assert positions is not None
    assert fwd_idx == 0


def test_compact_native_equal_cost_paths_beat_csr_insertion_order(
    tmp_path: Path,
) -> None:
    """The ranked native entry must choose the lexicographically smaller
    canonical edge-id sequence for equal-total paths even when the CSR
    insertion order visits the larger sequence first; the legacy entry keeps
    its deterministic behavior for callers without rank metadata."""
    lib = _native_library()
    graph = _diamond_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    cost_spec = _native_cost_spec(planner)
    arrays = dict(compact._sections)
    spec = _native_graph_spec(compact, arrays)
    topology = planner._csr_topology()
    assert topology is not None

    rc, positions, cost, fwd_idx, rev_idx = _call_native_search_ranked(
        lib,
        spec,
        cost_spec,
        [(0, 0.0, 0, 0)],
        [(3, 0.0, 0, 0)],
    )
    assert rc >= 0
    assert positions is not None
    assert cost == pytest.approx(2.0)
    assert [topology.edge_refs[p][0] for p in positions] == ["e0", "e1"]
    assert (fwd_idx, rev_idx) == (0, 0)

    # The e2/e3 path occupies earlier CSR positions (n0's forward row relaxes
    # e2 before e0); the ranked search must ignore CSR order.  The legacy
    # entry delegates to the same implementation with (0, 0) rank pairs, so
    # it resolves the tie identically.
    rc, positions, _cost, _fwd_idx, _rev_idx = _call_native_search(
        lib,
        spec,
        cost_spec,
        [(0, 0.0)],
        [(3, 0.0)],
    )
    assert rc >= 0
    assert positions is not None
    assert [topology.edge_refs[p][0] for p in positions] == ["e0", "e1"]

    # Repeated ranked calls stay byte-identical.
    rc, positions2, _cost, _fwd_idx, _rev_idx = _call_native_search_ranked(
        lib,
        spec,
        cost_spec,
        [(0, 0.0, 0, 0)],
        [(3, 0.0, 0, 0)],
    )
    assert rc >= 0
    assert positions2 == positions


def test_compact_native_search_is_deterministic_across_calls(
    tmp_path: Path,
) -> None:
    """Repeated native calls on a long chain must return byte-identical
    routes.  The dense per-call state reuses heap pages between calls, so
    this probes the explicit seen-state sentinel against stale bytes."""
    from src.route_planner._compact_search import (
        run_compact_bidirectional_search,
    )

    edge_count = 8192
    graph = _long_chain_graph(edge_count)
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    topology = planner._csr_topology()
    assert topology is not None
    cost_function = planner._make_fastest_cost_function()

    results = []
    for _ in range(4):
        result = run_compact_bidirectional_search(
            topology,
            cost_function,
            [("c0", 0.0, (), None)],
            [(f"c{edge_count}", 0.0, (), None)],
        )
        assert result is not None
        results.append(result)

    first = results[0]
    assert first[0] == list(range(edge_count))
    for result in results[1:]:
        assert result[0] == first[0]
        assert result[1] == first[1]
        assert (result[2], result[3]) == (first[2], first[3])


def test_compact_native_rejects_invalid_seed_and_neighbor_ranks(
    tmp_path: Path,
) -> None:
    """Out-of-range seed ranks must be skipped and out-of-range neighbor
    ranks must never be indexed; both must fail cleanly, never crash."""
    lib = _native_library()
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    cost_spec = _native_cost_spec(planner)
    arrays = dict(compact._sections)
    spec = _native_graph_spec(compact, arrays)

    # Negative seed rank: rejected as invalid, so the forward frontier stays
    # empty and the search reports generic failure instead of indexing OOB.
    rc, positions, _cost, _fwd, _rev = _call_native_search(
        lib, spec, cost_spec, [(-5, 0.0)], [(3, 0.0)]
    )
    assert rc == -1
    assert positions is None

    # Seed rank >= node_count: same safe rejection.
    rc, positions, _cost, _fwd, _rev = _call_native_search(
        lib, spec, cost_spec, [(999, 0.0)], [(3, 0.0)]
    )
    assert rc == -1
    assert positions is None

    # A neighbor rank equal to node_count must be skipped, not indexed.
    import numpy as np

    tampered = dict(arrays)
    indices = np.array(arrays["forward_indices"], copy=True)
    tampered_id = len(indices) - 1
    assert int(indices[tampered_id]) == compact.node_count - 1
    indices[tampered_id] = int(compact.node_count)  # out of range
    tampered["forward_indices"] = indices
    spec_tampered = _native_graph_spec(compact, tampered)

    rc, positions, _cost, _fwd, _rev = _call_native_search(
        lib, spec_tampered, cost_spec, [(0, 0.0)], [(3, 0.0)]
    )
    assert rc >= 0
    assert positions is not None
    # The out-of-range traversal must never be indexed or emitted.
    assert tampered_id not in positions
    # The surviving route still connects n0 -> n1 -> n3 through the valid
    # traversals (e0 and e3); the tampered edge (n2 -> n3) is bypassed.
    topology = planner._csr_topology()
    assert topology is not None
    refs = [topology.edge_refs[p] for p in positions]
    assert [edge_id for edge_id, _reverse in refs] == ["e0", "e3"]
    assert refs[0][1] is False and refs[1][1] is False


def test_compact_native_rejects_absurd_graph_metadata(
    tmp_path: Path,
) -> None:
    """Zero or unrepresentable node counts must fail with generic -1 before
    any per-node allocation is attempted."""
    import ctypes as _ctypes

    lib = _native_library()
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    cost_spec = _native_cost_spec(planner)
    arrays = compact._sections

    for bad_count in (0, -1, 2**31):
        spec = _native_graph_spec(compact, arrays)
        spec.node_count = bad_count
        rc, positions, _cost, _fwd, _rev = _call_native_search(
            lib, spec, cost_spec, [(0, 0.0)], [(3, 0.0)]
        )
        assert rc == -1
        assert positions is None


def test_compact_native_negative_seed_count_and_deadline_return_code(
    tmp_path: Path,
) -> None:
    """Negative seed counts fail generically; an exhausted deadline must
    surface as the native -2 code before any route is produced."""
    import ctypes as _ctypes

    lib = _native_library()
    graph = _fixture_graph()
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    planner = ScenicRoutePlanner(graph=compact)
    cost_spec = _native_cost_spec(planner)
    arrays = compact._sections
    spec = _native_graph_spec(compact, arrays)

    fwd_nodes = (_ctypes.c_int32 * 1)(0)
    fwd_costs = (_ctypes.c_double * 1)(0.0)
    rev_nodes = (_ctypes.c_int32 * 1)(3)
    rev_costs = (_ctypes.c_double * 1)(0.0)
    out_pos_ptr = _ctypes.POINTER(_ctypes.c_int64)()
    out_cost = _ctypes.c_double(0.0)
    out_fwd = _ctypes.c_int32(-1)
    out_rev = _ctypes.c_int32(-1)
    rc = lib.compact_bidirectional_search_alloc(
        _ctypes.byref(spec),
        _ctypes.byref(cost_spec),
        fwd_nodes,
        fwd_costs,
        -1,
        rev_nodes,
        rev_costs,
        1,
        _ctypes.byref(out_pos_ptr),
        _ctypes.byref(out_cost),
        _ctypes.byref(out_fwd),
        _ctypes.byref(out_rev),
        -1.0,
    )
    assert rc == -1

    # Long chain with a zero budget: the in-loop deadline check must fail
    # closed with -2, never return a route.
    edge_count = 8192
    chain = _long_chain_graph(edge_count)
    chain_sqlite, chain_manifest, _rec = _publish_compact(
        tmp_path / "chain", chain
    )
    chain_compact = CompactRoadGraph.load(chain_manifest)
    chain_planner = ScenicRoutePlanner(graph=chain_compact)
    chain_cost = _native_cost_spec(chain_planner)
    chain_arrays = chain_compact._sections
    chain_spec = _native_graph_spec(chain_compact, chain_arrays)
    rc, positions, _cost, _fwd, _rev = _call_native_search(
        lib,
        chain_spec,
        chain_cost,
        [(0, 0.0)],
        [(edge_count, 0.0)],
        deadline_seconds=0.0,
    )
    assert rc == -2
    assert positions is None


def test_compact_sidecar_native_uses_edge_rank_indexed_scores(
    tmp_path: Path,
) -> None:
    """A score sidecar whose CSR traversal order permutes the canonical edge
    ranks must be read through trav_edge_rank: the native route and cost
    match the eager and pure-Python CSR expectations, and traversal-position
    indexing of the sidecar would pick the *other* route."""
    import ctypes as _ctypes

    import numpy as np

    from src.route_planner._compact_search import (
        run_compact_bidirectional_search,
    )

    _native_library()  # skips when the compiled search is unavailable
    graph = _rank_permutation_graph()
    assert all(
        edge.travel_time_minutes == pytest.approx(1.0)
        for edge in graph.edges.values()
    )
    sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    scores = {"e0": 10.0, "e1": 2.0, "e2": 8.0, "e3": 2.0}
    report_path, _tiles = _tile_report(tmp_path, graph, scores)
    score_map, _zoom = load_tile_scores(report_path)

    eager = RoadGraph.load(sqlite_path)
    _apply_tile_scores_to_graph_native(eager, score_map, zoom=14, fallback=None)
    compact = CompactRoadGraph.load(manifest_path)
    report_signature = _signature_digest(
        _resolved_path_key(report_path), _file_signature(report_path)
    )
    matched, total = compact.activate_report_scores(
        score_map,
        zoom=14,
        fallback=None,
        report_signature=report_signature,
        normalization="linear-v1",
        tile_scores_path=report_path,
    )
    assert (matched, total) == (4, 4)
    sidecar = compact._active_score_sidecar
    assert sidecar is not None
    sidecar_values = np.asarray(sidecar.values)
    assert list(sidecar_values) == pytest.approx([10.0, 2.0, 8.0, 2.0])

    # Discriminating invariant: traversal positions permute the ranks, so
    # sidecar[position] and sidecar[trav_edge_rank[position]] disagree on
    # every traversal of this graph.
    assert list(compact._sections["trav_edge_rank"]) == [1, 3, 2, 0]

    eager_planner = ScenicRoutePlanner(graph=eager)
    compact_planner = ScenicRoutePlanner(graph=compact)
    cost_function = compact_planner._make_cost_function(0.8)
    fwd = [("n0", 0.0, (), None)]
    rev = [("n1", 0.0, (), None)]

    # Eager Python expectation (edge/rank-indexed scores).
    eager_result = eager_planner._bidirectional_search_core(cost_function, fwd, rev)
    assert eager_result is not None
    eager_path, _eager_key = eager_result
    expected_edges = [edge.id for edge in eager_path]
    expected_cost = sum(cost_function.calculate(edge) for edge in eager_path)
    assert expected_edges == ["e3", "e0"]
    assert expected_cost == pytest.approx(1.04)

    # Native production wrapper (sidecar dispatch) picks the same route and
    # cost.
    topology = compact_planner._csr_topology()
    native = run_compact_bidirectional_search(
        topology,
        cost_function,
        [("n0", 0.0, (0, 0), None)],
        [("n1", 0.0, (0, 0), None)],
    )
    assert native is not None
    positions, native_cost, _fwd_idx, _rev_idx = native
    assert [topology.edge_refs[pos][0] for pos in positions] == ["e3", "e0"]
    assert native_cost == pytest.approx(expected_cost)

    # The native path never materializes the E-sized rank->score remap: the
    # compact CSR scenic score stays a lazy callable after the search.
    assert callable(compact.compact_csr_arrays()._scenic_score)

    # Pure-Python compact CSR expectation with the same sidecar (np.take by
    # trav_edge_rank), forced past the native dispatch.
    signature = compact_planner._built_in_cost_signature(cost_function)
    weights = compact_planner._vectorized_builtin_weights(topology, signature)
    assert weights is not None
    py_csr = compact_planner._bidirectional_search_core(
        cost_function,
        fwd,
        rev,
        base_weight_override=lambda pos: float(weights[pos]),
    )
    assert py_csr is not None
    assert [edge.id for edge in py_csr[0]] == ["e3", "e0"]

    # Direct native probe through the sidecar ABI entry: same route/cost.
    lib = _native_library()
    arrays = dict(compact._sections)
    spec = _native_graph_spec(compact, arrays)
    scenic_spec = _scenic_cost_spec()
    rc, positions, direct_cost, _fi, _ri = _call_native_search_edge_scores(
        lib,
        spec,
        scenic_spec,
        sidecar.values.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
        [(0, 0.0, 0, 0)],
        [(1, 0.0, 0, 0)],
    )
    assert rc == 2
    assert [topology.edge_refs[pos][0] for pos in positions] == ["e3", "e0"]
    assert direct_cost == pytest.approx(expected_cost)

    # Flip proof: with these per-rank scores, rank-indexed costs prefer route
    # B while traversal-position-indexed costs prefer route A, so any
    # implementation reading sidecar[position] fails this test.
    def cost_of(score: float, duration: float = 1.0) -> float:
        return (
            (1.0 - 0.8) * duration
            + 0.8 * duration * (10.0 - score) / 10.0 * 1.0
        )

    s = list(sidecar_values)
    rank_a = cost_of(s[1]) + cost_of(s[2])  # route A by edge rank: e1, e2
    rank_b = cost_of(s[3]) + cost_of(s[0])  # route B by edge rank: e3, e0
    assert rank_b < rank_a
    pos_a = cost_of(s[0]) + cost_of(s[2])  # route A positions 0, 2
    pos_b = cost_of(s[1]) + cost_of(s[3])  # route B positions 1, 3
    assert pos_a < pos_b
    assert expected_cost == pytest.approx(rank_b)
    assert native_cost == pytest.approx(rank_b)


def test_compact_native_edge_score_entry_rejects_malformed_ranks(
    tmp_path: Path,
) -> None:
    """Out-of-range traversal ranks in sidecar mode must fail closed with -1
    before any search allocation; the legacy traversal-indexed entry keeps
    working with the same graph and rank table."""
    import ctypes as _ctypes

    import numpy as np

    lib = _native_library()
    graph = _rank_permutation_graph()
    _sqlite_path, manifest_path, _record = _publish_compact(tmp_path, graph)
    compact = CompactRoadGraph.load(manifest_path)
    arrays = dict(compact._sections)
    edge_count = int(compact.edge_count)
    assert edge_count == 4

    edge_scores = np.array([10.0, 2.0, 8.0, 2.0], dtype=np.float64)
    edge_scores_ptr = edge_scores.ctypes.data_as(
        _ctypes.POINTER(_ctypes.c_double)
    )
    scenic_spec = _scenic_cost_spec()
    fwd = [(0, 0.0, 0, 0)]
    rev = [(1, 0.0, 0, 0)]

    # Control: valid ranks through the sidecar entry succeed.
    spec_ok = _native_graph_spec(compact, arrays)
    rc, positions, _cost, _fi, _ri = _call_native_search_edge_scores(
        lib, spec_ok, scenic_spec, edge_scores_ptr, fwd, rev
    )
    assert rc == 2

    # Rank at or above edge_count is malformed.
    bad_upper = np.array(arrays["trav_edge_rank"], dtype=np.int32, copy=True)
    bad_upper[0] = edge_count
    arrays_upper = dict(arrays)
    arrays_upper["trav_edge_rank"] = bad_upper
    rc, positions, _cost, _fi, _ri = _call_native_search_edge_scores(
        lib,
        _native_graph_spec(compact, arrays_upper),
        scenic_spec,
        edge_scores_ptr,
        fwd,
        rev,
    )
    assert rc == -1
    assert positions is None

    # Negative rank is malformed.
    bad_neg = np.array(arrays["trav_edge_rank"], dtype=np.int32, copy=True)
    bad_neg[1] = -1
    arrays_neg = dict(arrays)
    arrays_neg["trav_edge_rank"] = bad_neg
    rc, positions, _cost, _fi, _ri = _call_native_search_edge_scores(
        lib,
        _native_graph_spec(compact, arrays_neg),
        scenic_spec,
        edge_scores_ptr,
        fwd,
        rev,
    )
    assert rc == -1
    assert positions is None

    # Missing rank table cannot index by rank in sidecar mode.
    arrays_no_rank = {
        name: value for name, value in arrays.items()
        if name != "trav_edge_rank"
    }
    rc, positions, _cost, _fi, _ri = _call_native_search_edge_scores(
        lib,
        _native_graph_spec(compact, arrays_no_rank),
        scenic_spec,
        edge_scores_ptr,
        fwd,
        rev,
    )
    assert rc == -1
    assert positions is None

    # The legacy traversal-indexed entry is untouched by the sidecar contract:
    # the same (tampered) rank table still routes through the ranked entry.
    rc, positions, _cost, _fi, _ri = _call_native_search_ranked(
        lib,
        _native_graph_spec(compact, arrays_upper),
        scenic_spec,
        fwd,
        rev,
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# Comparator transient-scratch allocation failure
# ---------------------------------------------------------------------------

_COMPARE_FAIL_INTERPOSER_C = r"""
/* Test-only malloc interposer: fails malloc calls made from the compact
 * search's transient path-comparison scratch (path_record_materialize_forward
 * and its inlined callers) when OMP_COMPACT_FAIL_COMPARE_ALLOC=even is set,
 * failing every second such call so the "left chain allocated, right chain
 * failed" cleanup path is exercised on every failing search.  All other
 * allocations pass through to the default malloc zone untouched.  The
 * dladdr check runs before getenv because getenv is unsafe during the
 * earliest dyld/libSystem initialization. */
#include <dlfcn.h>
#include <malloc/malloc.h>
#include <stdlib.h>
#include <string.h>

static int fail_enabled(void) {
    const char *value = getenv("OMP_COMPACT_FAIL_COMPARE_ALLOC");
    return value != NULL && strcmp(value, "even") == 0;
}

static int is_compare_scratch_alloc(void) {
    Dl_info info;
    if (dladdr(__builtin_return_address(0), &info) == 0 ||
        info.dli_sname == NULL) {
        return 0;
    }
    return strstr(info.dli_sname, "path_record_materialize_forward") != NULL ||
           strstr(info.dli_sname, "path_record_compare") != NULL ||
           strstr(info.dli_sname, "middle_key_compare") != NULL;
}

static void *my_malloc(size_t size) {
    if (is_compare_scratch_alloc() && fail_enabled()) {
        static unsigned long compare_alloc_count = 0;
        if ((++compare_alloc_count % 2) == 0) {
            return NULL;
        }
    }
    return malloc_zone_malloc(malloc_default_zone(), size);
}

__attribute__((used)) static struct {
    const void *replacement;
    const void *replacee;
} _interpose_malloc __attribute__((section("__DATA,__interpose"))) = {
    (const void *)my_malloc,
    (const void *)malloc,
};
"""

_COMPARE_FAIL_PROBE_PY = r"""
import resource
import sys
from pathlib import Path

sys.path.insert(0, ".")

import tests.test_compact_runtime as tcr

ITERATIONS = int(sys.argv[1])
EXPECT = sys.argv[2]  # "fail" or "ok"
EDGE_COUNT = int(sys.argv[3])

import tempfile

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    graph = tcr._deep_prefix_parallel_end_graph(EDGE_COUNT)
    _sqlite, manifest, _record = tcr._publish_compact(td, graph)
    compact = tcr.CompactRoadGraph.load(manifest)
    planner = tcr.ScenicRoutePlanner(graph=compact)
    cost_spec = tcr._native_cost_spec(planner)
    arrays = dict(compact._sections)
    spec = tcr._native_graph_spec(compact, arrays)
    lib = tcr._native_library()
    for _ in range(ITERATIONS):
        rc, positions, _cost, _fwd, _rev = tcr._call_native_search_ranked(
            lib,
            spec,
            cost_spec,
            [(0, 0.0, 0, 0)],
            [(EDGE_COUNT + 1, 0.0, 0, 0)],
        )
        if EXPECT == "fail":
            assert rc == -1 and positions is None, (rc, positions)
        else:
            assert rc >= 0 and positions is not None, (rc, positions)
    if EXPECT == "fail":
        # An exhausted deadline must still surface as -2 even when the
        # comparator scratch allocation would fail later in the search.
        rc, positions, _cost, _fwd, _rev = tcr._call_native_search_ranked(
            lib,
            spec,
            cost_spec,
            [(0, 0.0, 0, 0)],
            [(EDGE_COUNT + 1, 0.0, 0, 0)],
            deadline_seconds=0.0,
        )
        assert rc == -2 and positions is None, (rc, positions)
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="malloc interposition test requires DYLD_INSERT_LIBRARIES",
)
def test_compact_comparator_alloc_failure_preserves_minus_one_and_frees_scratch(
    tmp_path: Path,
) -> None:
    """A transient comparator-scratch allocation failure must abort the
    search with the generic -1 return (never -2, never a wrong route), and
    the already-allocated scratch must be freed on the failure path so
    repeated failing searches do not grow memory.

    A small malloc interposer is injected into a subprocess and fails every
    second comparator-scratch allocation (the "left chain allocated, right
    chain failed" case).  The control subprocess runs identical successful
    searches; a leaked half-megabyte of scratch per failing search over 120
    iterations would exceed the allowed peak-memory gap."""
    import os
    import subprocess
    import sys as _sys

    interposer_src = tmp_path / "fail_compare_alloc.c"
    interposer_src.write_text(_COMPARE_FAIL_INTERPOSER_C)
    dylib = tmp_path / "fail_compare_alloc.dylib"
    subprocess.run(
        [
            "clang",
            "-O2",
            "-dynamiclib",
            "-Wl,-undefined,dynamic_lookup",
            "-o",
            str(dylib),
            str(interposer_src),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = tmp_path / "compare_alloc_probe.py"
    probe.write_text(_COMPARE_FAIL_PROBE_PY)

    edge_count = 131072
    iterations = 120
    env = dict(os.environ)
    env["DYLD_INSERT_LIBRARIES"] = str(dylib)
    env["OMP_COMPACT_FAIL_COMPARE_ALLOC"] = "even"
    failing = subprocess.run(
        [_sys.executable, str(probe), str(iterations), "fail", str(edge_count)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert failing.returncode == 0, failing.stderr
    control = subprocess.run(
        [_sys.executable, str(probe), str(iterations), "ok", str(edge_count)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert control.returncode == 0, control.stderr
    failing_rss = int(failing.stdout.strip().splitlines()[-1])
    control_rss = int(control.stdout.strip().splitlines()[-1])
    # ~512 KiB of scratch per failing search would leak ~60 MiB over the
    # 120 iterations; the successful control peak is the same baseline, so a
    # leak shows up as a far larger gap than the generous 48 MiB allowance.
    assert failing_rss <= control_rss + 48 * 1024 * 1024, (
        failing_rss,
        control_rss,
    )

