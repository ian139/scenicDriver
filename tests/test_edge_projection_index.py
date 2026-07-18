from __future__ import annotations

import math
import random
import struct
from pathlib import Path

import pytest
import numpy as np

from src.route_planner import _edge_projection as edge_projection_module
from src.route_planner._edge_projection import (
    EdgeProjectionIndex,
    _EPI_HEADER_SIZE,
    _EPI_SECTION_DESCRIPTOR_SIZE,
    _EPI_VERSION,
    _NUM_SECTIONS,
)
from src.route_planner.graph import Edge, Node, RoadGraph


def _sidecar_path(path: Path) -> Path:
    return EdgeProjectionIndex.sidecar_path(path)


def _basic_graph() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="B", lat=42.1, lon=-72.0))
    graph.add_edge(
        Edge(
            id="AB",
            start_node_id="A",
            end_node_id="B",
            distance_km=12.0,
            scenic_score=7.5,
            road_name="Main Road",
            road_type="secondary",
            speed_limit_kmh=72,
        )
    )
    return graph


def test_sidecar_round_trip_loads_and_matches_rebuild(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    loaded = RoadGraph.load(path)
    status = loaded.edge_projection_index_status
    assert status["state"] == "loaded"
    assert status["mmap_read_only"] is True
    assert status["edge_count"] == len(loaded.edges)
    assert status["format_version"] == _EPI_VERSION
    assert status["algorithm"] == "bvh-spherical-lb"
    assert status["invalid_reason"] is None
    assert status["payload_size_bytes"] == _sidecar_path(path).stat().st_size

    sidecar = _sidecar_path(path)
    sidecar.unlink()
    rebuilt = RoadGraph.load(path)
    assert rebuilt.edge_projection_index_status["state"] == "missing"
    assert loaded.find_nearest_edge_positions_with_distance(
        42.05, -72.0
    ) == rebuilt.find_nearest_edge_positions_with_distance(42.05, -72.0)


def test_sidecar_missing_rebuilds(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    _sidecar_path(path).unlink()
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "missing"
    loaded.find_nearest_edge_positions_with_distance(42.05, -72.0)
    assert loaded.edge_projection_index_status["state"] == "rebuilt"
    assert loaded.edge_projection_index_status["mmap_read_only"] is False


@pytest.mark.parametrize(
    "truncate_to",
    [
        0,
        40,
        _EPI_HEADER_SIZE,
        _EPI_HEADER_SIZE + 20,
        _EPI_HEADER_SIZE + _NUM_SECTIONS * _EPI_SECTION_DESCRIPTOR_SIZE - 10,
        -100,
    ],
)
def test_sidecar_truncation_recovers(truncate_to: int, tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    sidecar = _sidecar_path(path)
    data = sidecar.read_bytes()
    trunc = truncate_to if truncate_to >= 0 else len(data) + truncate_to
    sidecar.write_bytes(data[:trunc])
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "invalid"
    assert loaded.edge_projection_index_status["invalid_reason"] is not None
    loaded.find_nearest_edge_positions_with_distance(42.05, -72.0)
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def test_sidecar_payload_corruption_recovers(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    sidecar = _sidecar_path(path)
    data = bytearray(sidecar.read_bytes())
    data[-1] ^= 0xFF
    sidecar.write_bytes(bytes(data))
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "invalid"
    loaded.find_nearest_edge_positions_with_distance(42.05, -72.0)
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def test_sidecar_version_mismatch_recovers(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    sidecar = _sidecar_path(path)
    data = bytearray(sidecar.read_bytes())
    data[8:12] = struct.pack("<I", 0x7FFFFFFF)
    sidecar.write_bytes(bytes(data))
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "invalid"
    loaded.find_nearest_edge_positions_with_distance(42.05, -72.0)
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def test_sidecar_stale_graph_recovers(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    with open(path, "ab") as f:
        f.write(b"x")
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "invalid"
    loaded.find_nearest_edge_positions_with_distance(42.05, -72.0)
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def test_sidecar_root_bit_flip_recovers(tmp_path: Path) -> None:
    graph = RoadGraph()
    for index in range(256):
        lat = 40.0 + index * 0.01
        graph.add_node(Node(id=f"a{index}", lat=lat, lon=-72.0))
        graph.add_node(Node(id=f"b{index}", lat=lat + 0.005, lon=-71.99))
        graph.add_edge(
            Edge(
                id=f"e{index}",
                start_node_id=f"a{index}",
                end_node_id=f"b{index}",
                distance_km=1.0,
                scenic_score=0.0,
            )
        )
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    sidecar = _sidecar_path(path)
    data = bytearray(sidecar.read_bytes())
    # root_node is the signed int at header offset 52 (<8sIIQQQQIi32s).
    root_offset = 8 + 4 + 4 + 8 + 8 + 8 + 8 + 4
    original_root = struct.unpack_from("<i", data, root_offset)[0]
    assert original_root > 0
    struct.pack_into("<i", data, root_offset, original_root - 1)
    sidecar.write_bytes(bytes(data))
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "invalid"
    loaded.find_nearest_edge_positions_with_distance(40.3, -72.0)
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def test_sidecar_atomic_write_failure_preserves_prior(
    monkeypatch, tmp_path: Path
) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    sidecar = _sidecar_path(path)
    prior = sidecar.read_bytes()
    replace = edge_projection_module.os.replace

    def fail_publication(source: object, destination: object) -> None:
        if Path(destination) == sidecar:
            raise OSError("publication failure")
        replace(source, destination)

    monkeypatch.setattr(edge_projection_module.os, "replace", fail_publication)
    with pytest.raises(OSError, match="publication failure"):
        graph.persist_edge_projection_index(path)
    assert sidecar.read_bytes() == prior
    assert not list(sidecar.parent.glob(f".{sidecar.name}.*.tmp"))


def test_mutation_invalidates_loaded_sidecar(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "loaded"
    list(loaded.edges.values())[0].road_type = "motorway"
    assert loaded.edge_projection_index_status["state"] == "invalid"
    assert (
        loaded.edge_projection_index_status["invalid_reason"]
        == "graph_mutated"
    )
    loaded.find_nearest_edge_positions_with_distance(42.05, -72.0)
    assert loaded.edge_projection_index_status["state"] == "rebuilt"


def test_cancellation_during_persist(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    calls = 0

    def _cancel() -> None:
        nonlocal calls
        calls += 1
        if calls > 2:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        graph.persist_edge_projection_index(path, check_cancelled=_cancel)
    assert not list(_sidecar_path(path).parent.glob("*.tmp"))


def test_loaded_keys_are_existing_graph_key_objects(tmp_path: Path) -> None:
    graph = _basic_graph()
    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    loaded = RoadGraph.load(path)
    projections, _ = loaded.find_nearest_edge_positions_with_distance(42.05, -72.0)
    assert len(projections) == 1
    edge_key = projections[0].edge.id
    assert edge_key in loaded.edges
    assert edge_key is list(loaded.edges.keys())[0]


def _distractor_graph(count: int = 2048) -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node(id="near_a", lat=0.0, lon=-0.005))
    graph.add_node(Node(id="near_b", lat=0.0, lon=0.005))
    graph.add_edge(
        Edge(
            id="near",
            start_node_id="near_a",
            end_node_id="near_b",
            distance_km=1.0,
            scenic_score=5.0,
        )
    )
    for i in range(count):
        lat = 10.0 + 70.0 * i / max(count - 1, 1)
        a = f"d{i}a"
        b = f"d{i}b"
        graph.add_node(Node(id=a, lat=lat, lon=0.0))
        graph.add_node(Node(id=b, lat=lat + 0.001, lon=0.001))
        graph.add_edge(
            Edge(
                id=f"d{i}",
                start_node_id=a,
                end_node_id=b,
                distance_km=1.0,
                scenic_score=0.0,
            )
        )
    return graph


def test_bvh_prunes_distractor_heavy_graph() -> None:
    graph = _distractor_graph(2048)
    index = graph._build_nearest_edge_projection_index()
    (projs, _dist), stats = index.query(graph, 0.0, 0.0, with_stats=True)
    assert len(projs) == 1
    assert projs[0].edge.id == "near"
    assert stats["scanned_edges"] < 200
    assert stats["max_candidates"] < 100


def test_bvh_never_prunes_adversarial_long_segment() -> None:
    graph = RoadGraph()
    # Long meridian segment with endpoints far in longitude.
    graph.add_node(Node(id="long_a", lat=0.0, lon=-100.0))
    graph.add_node(Node(id="long_b", lat=0.0, lon=100.0))
    graph.add_edge(
        Edge(
            id="long",
            start_node_id="long_a",
            end_node_id="long_b",
            distance_km=100.0,
            scenic_score=0.0,
        )
    )
    # Distractors clustered around latitude 10.
    for i in range(128):
        lat = 10.0 + 0.1 * i
        a = f"d{i}a"
        b = f"d{i}b"
        graph.add_node(Node(id=a, lat=lat, lon=-0.1))
        graph.add_node(Node(id=b, lat=lat, lon=0.1))
        graph.add_edge(
            Edge(
                id=f"d{i}",
                start_node_id=a,
                end_node_id=b,
                distance_km=1.0,
                scenic_score=0.0,
            )
        )
    index = graph._build_nearest_edge_projection_index()
    projs, dist = index.query(graph, 0.0, 0.0)
    assert projs[0].edge.id == "long"
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_exclusions_filter_and_tie_order() -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=0.0, lon=0.0))
    graph.add_node(Node(id="b", lat=0.0, lon=1.0))
    graph.add_node(Node(id="c", lat=0.0, lon=-1.0))
    graph.add_edge(
        Edge(
            id="included",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=0.0,
            road_type="secondary",
        )
    )
    graph.add_edge(
        Edge(
            id="excluded",
            start_node_id="a",
            end_node_id="c",
            distance_km=1.0,
            scenic_score=0.0,
            road_type="motorway",
        )
    )
    projections, _ = graph.find_nearest_edge_positions_with_distance(
        0.0, 0.0, excluded_road_types=frozenset({"motorway"})
    )
    assert [p.edge.id for p in projections] == ["included"]

    # Two collinear segments at the same distance should be returned in edge-id order.
    graph2 = RoadGraph()
    graph2.add_node(Node(id="a", lat=0.0, lon=0.0))
    graph2.add_node(Node(id="b", lat=0.0, lon=1.0))
    graph2.add_node(Node(id="c", lat=0.0, lon=-1.0))
    graph2.add_edge(
        Edge(
            id="z",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=0.0,
        )
    )
    graph2.add_edge(
        Edge(
            id="a",
            start_node_id="a",
            end_node_id="c",
            distance_km=1.0,
            scenic_score=0.0,
        )
    )
    projections2, _ = graph2.find_nearest_edge_positions_with_distance(0.0, 0.0)
    assert [p.edge.id for p in projections2] == ["a", "z"]


class _CountedDict(dict):
    iter_count = 0

    def __iter__(self):
        _CountedDict.iter_count += 1
        return super().__iter__()


def test_query_reuses_canonical_keys_without_mapping_iteration() -> None:
    graph = _basic_graph()
    index = graph._build_nearest_edge_projection_index()
    index.attach(graph)
    counted = _CountedDict(graph.edges)
    graph.edges = counted
    index.query(graph, 42.05, -72.0)
    index.query(graph, 42.05, -72.0)
    assert counted.iter_count == 0


def test_sidecar_resolves_valid_rank_after_skipped_nonfinite_edge(
    tmp_path: Path,
) -> None:
    graph = RoadGraph()
    # First edge has a non-finite endpoint, so it is skipped from the index.
    graph.add_node(Node(id="bad_a", lat=float("inf"), lon=0.0))
    graph.add_node(Node(id="bad_b", lat=0.0, lon=1.0))
    graph.add_node(Node(id="good_a", lat=0.0, lon=0.0))
    graph.add_node(Node(id="good_b", lat=0.0, lon=1.0))
    graph.add_edge(
        Edge(
            id="bad",
            start_node_id="bad_a",
            end_node_id="bad_b",
            distance_km=1.0,
            scenic_score=0.0,
        )
    )
    graph.add_edge(
        Edge(
            id="good",
            start_node_id="good_a",
            end_node_id="good_b",
            distance_km=1.0,
            scenic_score=0.0,
        )
    )
    index = graph._build_nearest_edge_projection_index()
    assert index.edge_count == 1
    assert index.total_edge_count == 2
    projections, _ = index.query(graph, 0.0, 0.5)
    assert [p.edge.id for p in projections] == ["good"]

    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    loaded = RoadGraph.load(path)
    projections2, _ = loaded.find_nearest_edge_positions_with_distance(0.0, 0.5)
    assert [p.edge.id for p in projections2] == ["good"]
    assert loaded.edge_projection_index_status["state"] == "loaded"


def test_sidecar_roundtrip_with_only_nonfinite_segments(tmp_path: Path) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="a", lat=float("inf"), lon=0.0))
    graph.add_node(Node(id="b", lat=0.0, lon=1.0))
    graph.add_edge(
        Edge(
            id="only",
            start_node_id="a",
            end_node_id="b",
            distance_km=1.0,
            scenic_score=0.0,
        )
    )
    index = graph._build_nearest_edge_projection_index()
    assert index.edge_count == 0
    assert index.total_edge_count == 1

    path = tmp_path / "g.sqlite3"
    graph.save(path)
    graph.persist_edge_projection_index(path)
    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "loaded"
    assert loaded.edge_projection_index_status["edge_count"] == 0
    with pytest.raises(ValueError, match="no eligible finite segment"):
        loaded.find_nearest_edge_positions_with_distance(0.0, 0.5)


def test_compatible_sidecar_query_never_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "g.sqlite3"
    _basic_graph().save(path)
    loaded = RoadGraph.load(path)
    persisted_index = loaded._nearest_edge_projection_index
    assert persisted_index is not None

    def fail_build(**_kwargs: object) -> EdgeProjectionIndex:
        raise AssertionError("compatible persisted index was rebuilt")

    monkeypatch.setattr(
        loaded,
        "_build_nearest_edge_projection_index",
        fail_build,
    )
    projections, _distance = loaded.find_nearest_edge_positions_with_distance(
        42.05,
        -72.0,
    )
    assert [projection.edge.id for projection in projections] == ["AB"]
    assert loaded._nearest_edge_projection_index is persisted_index
    assert loaded.edge_projection_index_status["state"] == "loaded"


def test_scenic_score_mutation_preserves_loaded_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "g.sqlite3"
    _basic_graph().save(path)
    loaded = RoadGraph.load(path)
    persisted_index = loaded._nearest_edge_projection_index
    assert persisted_index is not None
    loaded.edges["AB"].scenic_score = 1.25

    def fail_build(**_kwargs: object) -> EdgeProjectionIndex:
        raise AssertionError("scenic scoring invalidated projection geometry")

    monkeypatch.setattr(
        loaded,
        "_build_nearest_edge_projection_index",
        fail_build,
    )
    projections, _distance = loaded.find_nearest_edge_positions_with_distance(
        42.05,
        -72.0,
    )
    assert [projection.edge.scenic_score for projection in projections] == [1.25]
    assert loaded._nearest_edge_projection_index is persisted_index
    assert loaded.edge_projection_index_status["state"] == "loaded"


def test_sidecar_load_preserves_cancellation_exception_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "g.sqlite3"
    _basic_graph().save(path)
    loaded = RoadGraph.load(path)
    error = InterruptedError("cancel sidecar load")

    def cancel() -> None:
        raise error

    with pytest.raises(InterruptedError) as raised:
        loaded._try_load_edge_projection_index(
            path,
            check_cancelled=cancel,
        )
    assert raised.value is error


def test_read_only_graph_and_sidecar_load_without_writes(tmp_path: Path) -> None:
    path = tmp_path / "g.sqlite3"
    _basic_graph().save(path)
    sidecar = _sidecar_path(path)
    path.chmod(0o444)
    sidecar.chmod(0o444)

    loaded = RoadGraph.load(path)
    projections, _distance = loaded.find_nearest_edge_positions_with_distance(
        42.05,
        -72.0,
    )
    assert [projection.edge.id for projection in projections] == ["AB"]
    assert loaded.edge_projection_index_status["state"] == "loaded"


def test_empty_sqlite_graph_sidecar_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite3"
    RoadGraph().save(path)

    loaded = RoadGraph.load(path)
    assert loaded.edge_projection_index_status["state"] == "loaded"
    assert loaded.edge_projection_index_status["edge_count"] == 0
    with pytest.raises(ValueError, match="no eligible finite segment"):
        loaded.find_nearest_edge_positions_with_distance(0.0, 0.0)


def test_bvh_query_starts_at_postorder_root_not_leaf_zero() -> None:
    graph = RoadGraph()
    for index in range(129):
        latitude = -80.0 + index * 0.5
        graph.add_node(Node(id=f"a{index}", lat=latitude, lon=0.0))
        graph.add_node(Node(id=f"b{index}", lat=latitude, lon=0.01))
        graph.add_edge(
            Edge(
                id=f"far{index}",
                start_node_id=f"a{index}",
                end_node_id=f"b{index}",
                distance_km=1.0,
                scenic_score=0.0,
            )
        )
    graph.add_node(Node(id="near-a", lat=80.0, lon=-0.01))
    graph.add_node(Node(id="near-b", lat=80.0, lon=0.01))
    graph.add_edge(
        Edge(
            id="near",
            start_node_id="near-a",
            end_node_id="near-b",
            distance_km=1.0,
            scenic_score=0.0,
        )
    )

    index = graph._build_nearest_edge_projection_index()
    near_position = int(np.flatnonzero(index.edge_ranks == 129)[0])
    assert index.root_node == index.node_count - 1
    assert near_position >= int(index.bvh_stop[0])
    projections, distance = index.query(graph, 80.0, 0.0)
    assert [projection.edge.id for projection in projections] == ["near"]
    assert distance == pytest.approx(0.0, abs=1e-9)


def test_bvh_queries_match_full_scan_on_deterministic_graph() -> None:
    randomizer = random.Random(8317)
    graph = RoadGraph()
    for index in range(257):
        start_lat = randomizer.uniform(41.0, 47.0)
        start_lon = randomizer.uniform(-73.5, -67.0)
        end_lat = start_lat + randomizer.uniform(-0.08, 0.08)
        end_lon = start_lon + randomizer.uniform(-0.08, 0.08)
        graph.add_node(Node(id=f"a{index}", lat=start_lat, lon=start_lon))
        graph.add_node(Node(id=f"b{index}", lat=end_lat, lon=end_lon))
        graph.add_edge(
            Edge(
                id=f"edge-{index:03d}",
                start_node_id=f"a{index}",
                end_node_id=f"b{index}",
                distance_km=1.0,
                scenic_score=0.0,
                road_type="motorway" if index % 7 == 0 else "secondary",
            )
        )
    index = graph._build_nearest_edge_projection_index()
    canonical_keys = tuple(graph.edges)

    for query_index in range(20):
        query_lat = randomizer.uniform(41.0, 47.0)
        query_lon = randomizer.uniform(-73.5, -67.0)
        excluded = frozenset({"motorway"}) if query_index % 2 else frozenset()
        actual, actual_distance = index.query(
            graph,
            query_lat,
            query_lon,
            excluded_road_types=excluded,
        )

        fractions, projected_latitudes, projected_longitudes, distances = (
            graph._project_edge_chunk(
                query_lat,
                query_lon,
                math.cos(math.radians(query_lat)),
                index.start_latitudes,
                index.start_longitudes,
                index.end_latitudes,
                index.end_longitudes,
                0,
                index.edge_count,
            )
        )
        allowed_types = np.asarray(
            [road_type not in excluded for road_type in index.road_type_names],
            dtype=np.bool_,
        )
        eligible = allowed_types[index.road_type_codes] & np.isfinite(distances)
        reference_distance = float(np.min(distances[eligible]))
        tied = eligible & (distances <= reference_distance + 1e-9)
        reference = []
        for local_index in np.flatnonzero(tied):
            edge_key = canonical_keys[int(index.edge_ranks[local_index])]
            edge = graph.edges[edge_key]
            reference.append(
                (
                    (str(edge.id), str(edge_key)),
                    float(fractions[local_index]),
                    float(projected_latitudes[local_index]),
                    float(projected_longitudes[local_index]),
                    float(distances[local_index]),
                )
            )
        reference.sort(key=lambda value: value[0])

        assert actual_distance == pytest.approx(reference_distance)
        assert [projection.edge.id for projection in actual] == [
            value[0][0] for value in reference
        ]
        assert [projection.fraction for projection in actual] == pytest.approx(
            [value[1] for value in reference]
        )
        assert [projection.snap_distance_km for projection in actual] == pytest.approx(
            [value[4] for value in reference]
        )


def test_longitude_bound_prunes_same_latitude_distractors() -> None:
    graph = RoadGraph()
    edge_count = 2_048
    for index in range(edge_count):
        longitude = -170.0 + 340.0 * index / (edge_count - 1)
        graph.add_node(Node(id=f"a{index}", lat=0.0, lon=longitude))
        graph.add_node(
            Node(id=f"b{index}", lat=0.0, lon=longitude + 0.001)
        )
        graph.add_edge(
            Edge(
                id=f"edge-{index}",
                start_node_id=f"a{index}",
                end_node_id=f"b{index}",
                distance_km=0.1,
                scenic_score=0.0,
            )
        )
    expected_index = 1_137
    query_lon = -170.0 + 340.0 * expected_index / (edge_count - 1)
    index = graph._build_nearest_edge_projection_index()

    (projections, _distance), stats = index.query(
        graph,
        0.0,
        query_lon,
        with_stats=True,
    )

    assert [projection.edge.id for projection in projections] == [
        f"edge-{expected_index}"
    ]
    assert stats["scanned_edges"] < 200
