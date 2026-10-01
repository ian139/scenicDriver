from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.route_planner.graph as graph_module
from src.route_planner.graph import CompactRoadGraph, Edge, Node, RoadGraph, write_compact_graph


def _save_graph(path: Path, scenic_score: float) -> None:
    graph = RoadGraph()
    graph.add_node(Node(id="A", lat=42.0, lon=-72.0))
    graph.add_node(Node(id="B", lat=42.1, lon=-72.0))
    graph.add_edge(Edge(id="AB", start_node_id="A", end_node_id="B", distance_km=12.0,
                        scenic_score=scenic_score, road_type="secondary", one_way=False))
    graph.save(path)


@pytest.mark.parametrize("failure", ["manifest", "cancel", "hash", "cancel_after_generation"])
def test_failed_compact_publication_preserves_previous_generation(tmp_path: Path, monkeypatch, failure: str) -> None:
    source = tmp_path / "road_graph.sqlite3"
    manifest = tmp_path / "road_graph.compact.json"
    _save_graph(source, 2.0)
    write_compact_graph(source, manifest)
    previous_manifest = manifest.read_bytes()
    _save_graph(source, 9.0)
    original_hash = graph_module._sha256_file
    original_publish = graph_module._publish_compact_generation
    cancelled = False

    def fail_manifest(path, payload):
        raise OSError("injected publication failure")

    def hash_file(path, *, check_cancelled=None):
        nonlocal cancelled
        if Path(path) == graph_module.EdgeProjectionIndex.sidecar_path(source):
            if failure == "hash":
                raise OSError("injected hash failure")
            cancelled = True
        return original_hash(path, check_cancelled=check_cancelled)

    def publish(*args, **kwargs):
        nonlocal cancelled
        result = original_publish(*args, **kwargs)
        cancelled = True
        return result

    def check_cancelled():
        if cancelled:
            raise RuntimeError("injected cancellation")

    if failure == "manifest":
        monkeypatch.setattr(graph_module, "_write_atomic_bytes", fail_manifest)
    elif failure == "cancel_after_generation":
        monkeypatch.setattr(graph_module, "_publish_compact_generation", publish)
    else:
        monkeypatch.setattr(graph_module, "_sha256_file", hash_file)
    with pytest.raises((OSError, RuntimeError), match="injected"):
        write_compact_graph(source, manifest, check_cancelled=check_cancelled)
    assert manifest.read_bytes() == previous_manifest
    with CompactRoadGraph.load(manifest) as loaded:
        assert loaded.edges["AB"].scenic_score == 2.0
        assert loaded.edge_projection_index_status["state"] == "loaded"


def test_cross_directory_compact_publication_resolves_all_dependencies(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    source = inputs / "road_graph.sqlite3"
    _save_graph(source, 2.0)
    _save_graph(outputs / source.name, 9.0)
    manifest = outputs / "road_graph.compact.json"
    write_compact_graph(source, manifest)
    data = json.loads(manifest.read_text())
    for name in [data["source"]["path"], data["bin_path"], data["projection_index"]["path"]]:
        assert (manifest.parent / name).is_file()
    with CompactRoadGraph.load(manifest) as loaded:
        assert loaded.edges["AB"].scenic_score == 2.0
        assert loaded.edge_projection_index_status["state"] == "loaded"


def test_successful_republication_keeps_previous_dependencies_immutable(tmp_path: Path) -> None:
    source = tmp_path / "road_graph.sqlite3"
    manifest = tmp_path / "road_graph.compact.json"
    _save_graph(source, 2.0)
    write_compact_graph(source, manifest)
    previous = tmp_path / "previous.compact.json"
    previous.write_bytes(manifest.read_bytes())
    _save_graph(source, 9.0)
    write_compact_graph(source, manifest)
    with CompactRoadGraph.load(previous) as loaded:
        assert loaded.edges["AB"].scenic_score == 2.0
    with CompactRoadGraph.load(manifest) as loaded:
        assert loaded.edges["AB"].scenic_score == 9.0


def test_identical_compact_publication_reuses_immutable_dependencies(tmp_path: Path) -> None:
    source = tmp_path / "road_graph.sqlite3"
    manifest = tmp_path / "road_graph.compact.json"
    _save_graph(source, 2.0)
    write_compact_graph(source, manifest)
    payload = manifest.read_bytes()
    data = json.loads(payload)
    dependencies = [
        manifest.parent / data["source"]["path"],
        manifest.parent / data["bin_path"],
        manifest.parent / data["projection_index"]["path"],
    ]
    originals = [(path.read_bytes(), path.stat().st_ino) for path in dependencies]
    write_compact_graph(source, manifest)
    assert manifest.read_bytes() == payload
    assert [(path.read_bytes(), path.stat().st_ino) for path in dependencies] == originals
    with CompactRoadGraph.load(manifest) as loaded:
        assert loaded.edges["AB"].scenic_score == 2.0
