from __future__ import annotations

import json
from pathlib import Path

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

    second = route_service.plan_routes(request)
    assert second["diagnostics"]["scored_graph_cache_hit"] is True
    assert second["score_mapping"] == result["score_mapping"]
    assert (
        second["routes"][0]["metrics"]["edge_ids"]
        == result["routes"][0]["metrics"]["edge_ids"]
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
