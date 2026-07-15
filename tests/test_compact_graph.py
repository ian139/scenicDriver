from __future__ import annotations

from pathlib import Path

import pytest

from scripts.routing.compact_graph import (
    CompactArtifactError,
    convert_json_to_compact,
    validate_compact_artifact,
)
from src.route_planner.graph import Edge, Node, RoadGraph


def _sample_graph() -> RoadGraph:
    graph = RoadGraph()
    for node_id, lat, lon in (
        ("A", 42.0, -72.0),
        ("B", 42.01, -72.01),
        ("C", 42.02, -72.02),
        ("D", 42.0, -72.0),
    ):
        graph.add_node(Node(id=node_id, lat=lat, lon=lon))
    graph.add_edge(
        Edge(
            id="edge-z",
            start_node_id="A",
            end_node_id="B",
            distance_km=1.25,
            scenic_score=8.75,
            road_name="Pine Road",
            road_type="primary",
            speed_limit_kmh=45,
            one_way=False,
        )
    )
    graph.add_edge(
        Edge(
            id="edge-a",
            start_node_id="B",
            end_node_id="C",
            distance_km=2.5,
            scenic_score=3.125,
            road_name=None,
            road_type="secondary",
            speed_limit_kmh=50,
            one_way=True,
        )
    )
    graph.add_edge(
        Edge(
            id="edge-loop",
            start_node_id="C",
            end_node_id="A",
            distance_km=4.0,
            scenic_score=6.0,
            road_name="Loop",
            road_type="tertiary",
            speed_limit_kmh=30,
            one_way=False,
        )
    )
    return graph


def _public_edges(graph: RoadGraph, node_id: str) -> list[tuple[object, ...]]:
    return [
        (
            edge.id,
            edge.start_node_id,
            edge.end_node_id,
            edge.distance_km,
            edge.scenic_score,
            edge.road_name,
            edge.road_type,
            edge.speed_limit_kmh,
            edge.one_way,
        )
        for edge in graph.get_edges(node_id)
    ]


def test_converter_and_loader_preserve_graph_behavior(tmp_path: Path) -> None:
    source = tmp_path / "road_graph.json"
    compact = tmp_path / "road_graph.compact"
    expected = _sample_graph()
    expected.save(source)

    result = convert_json_to_compact(source, compact)
    assert result.destination == compact
    assert result.node_count == 4
    assert result.edge_count == 3
    assert result.arc_count == 5
    assert validate_compact_artifact(compact).node_count == 4

    loaded = RoadGraph.load(compact)
    assert list(loaded.nodes) == list(expected.nodes)
    assert list(loaded.edges) == list(expected.edges)
    assert loaded.adjacency == expected.adjacency
    for node_id in expected.nodes:
        assert _public_edges(loaded, node_id) == _public_edges(expected, node_id)
        assert [
            getattr(edge, "traversal_id", f"forward:{edge.id}") for edge in loaded.iter_edges(node_id)
        ] == [
            getattr(edge, "traversal_id", f"forward:{edge.id}") for edge in expected.iter_edges(node_id)
        ]

    # Equal raw-coordinate distances retain insertion-order tie semantics.
    assert loaded.find_nearest_node(42.0, -72.0).id == "A"
    assert loaded.find_nearest_node_with_distance(42.0, -72.0)[0].id == "A"


def test_compact_artifact_rejects_corruption_and_version_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "road_graph.json"
    compact = tmp_path / "road_graph.compact"
    _sample_graph().save(source)
    convert_json_to_compact(source, compact)

    corrupt = tmp_path / "corrupt.compact"
    data = bytearray(compact.read_bytes())
    data[-1] ^= 0x01
    corrupt.write_bytes(data)
    with pytest.raises(CompactArtifactError, match="checksum mismatch"):
        RoadGraph.load(corrupt)

    mismatched = tmp_path / "mismatched.compact"
    data = bytearray(compact.read_bytes())
    data[ len(b"SCENIC-COMPACT-GRAPH\\n") : len(b"SCENIC-COMPACT-GRAPH\\n") + 4 ] = (999).to_bytes(4, "big")
    mismatched.write_bytes(data)
    with pytest.raises(CompactArtifactError, match="unsupported artifact version"):
        validate_compact_artifact(mismatched)


def test_json_loading_and_one_way_default_remain_compatible(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"nodes": [{"id": "1", "lat": 1, "lon": 2}, {"id": "2", "lat": 2, "lon": 3}], '
        '"edges": [{"id": "e", "start": "1", "end": "2", "distance_km": 1.5}]}'
    )

    graph = RoadGraph.load(path)
    assert graph.edges["e"].one_way is True
    assert graph.adjacency == {"1": [("e", False)], "2": []}
