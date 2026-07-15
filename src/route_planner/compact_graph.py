from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping, Sequence

import msgspec

COMPACT_MAGIC = b"SCENIC-COMPACT-GRAPH\n"
COMPACT_SCHEMA = "scenic-drive.compact-graph"
COMPACT_VERSION = 1
_HEADER = struct.Struct(">IQ32s")
COMPACT_HEADER_SIZE = len(COMPACT_MAGIC) + _HEADER.size


class CompactArtifactError(ValueError):
    """Raised when a compact graph artifact is malformed or incompatible."""


@dataclass(frozen=True)
class CompactArtifactInfo:
    path: Path
    node_count: int
    edge_count: int
    arc_count: int
    checksum_sha256: str
    source_sha256: str | None


def is_compact_artifact(path: Path) -> bool:
    """Return whether *path* begins with the compact-artifact magic bytes."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(COMPACT_MAGIC)) == COMPACT_MAGIC
    except OSError as exc:
        raise CompactArtifactError(f"cannot read graph artifact {path}: {exc}") from exc


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactArtifactError(f"{label} must be an object")
    return value


def _as_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompactArtifactError(f"metadata {label} must be a non-negative integer")
    return value


def _as_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CompactArtifactError(f"{label} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactArtifactError(f"{label} must be a finite number") from exc
    if not math.isfinite(converted):
        raise CompactArtifactError(f"{label} must be a finite number")
    return converted


def _decode_payload(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CompactArtifactError(f"cannot read compact artifact {path}: {exc}") from exc
    if len(raw) < COMPACT_HEADER_SIZE:
        raise CompactArtifactError(
            f"compact artifact {path} has a truncated header "
            f"(expected at least {COMPACT_HEADER_SIZE} bytes)"
        )
    if raw[: len(COMPACT_MAGIC)] != COMPACT_MAGIC:
        raise CompactArtifactError(
            f"{path} is not a compact graph artifact (magic header is missing)"
        )
    version, payload_length, expected_digest = _HEADER.unpack_from(raw, len(COMPACT_MAGIC))
    if version != COMPACT_VERSION:
        raise CompactArtifactError(
            f"unsupported artifact version {version}; expected {COMPACT_VERSION}"
        )
    payload_start = COMPACT_HEADER_SIZE
    payload_end = payload_start + payload_length
    if payload_end > len(raw):
        raise CompactArtifactError(
            f"compact artifact payload is truncated: header declares {payload_length} bytes, "
            f"but only {len(raw) - payload_start} are present"
        )
    if payload_end != len(raw):
        raise CompactArtifactError(
            f"compact artifact has {len(raw) - payload_end} unexpected trailing bytes"
        )
    payload = raw[payload_start:payload_end]
    actual_digest = hashlib.sha256(payload).digest()
    if actual_digest != expected_digest:
        raise CompactArtifactError(
            "compact artifact checksum mismatch: "
            f"expected {expected_digest.hex()}, got {actual_digest.hex()}"
        )
    try:
        decoded = msgspec.msgpack.decode(payload)
    except Exception as exc:  # msgspec uses a version-specific decode exception
        raise CompactArtifactError(f"compact artifact payload is not valid MessagePack: {exc}") from exc
    envelope = dict(_as_mapping(decoded, "artifact payload"))
    _validate_payload(envelope)
    return envelope, actual_digest.hex()


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != COMPACT_SCHEMA:
        raise CompactArtifactError(
            f"unsupported compact graph schema {payload.get('schema')!r}; expected {COMPACT_SCHEMA!r}"
        )
    version = payload.get("version")
    if version != COMPACT_VERSION:
        raise CompactArtifactError(
            f"unsupported artifact payload version {version!r}; expected {COMPACT_VERSION}"
        )
    metadata = _as_mapping(payload.get("metadata"), "artifact metadata")
    node_rows = payload.get("nodes")
    edge_rows = payload.get("edges")
    adjacency_rows = payload.get("adjacency")
    if not isinstance(node_rows, list):
        raise CompactArtifactError("artifact nodes must be an array")
    if not isinstance(edge_rows, list):
        raise CompactArtifactError("artifact edges must be an array")
    if not isinstance(adjacency_rows, list):
        raise CompactArtifactError("artifact adjacency must be an array")

    node_count = _as_count(metadata.get("node_count"), "node_count")
    edge_count = _as_count(metadata.get("edge_count"), "edge_count")
    arc_count = _as_count(metadata.get("arc_count"), "arc_count")
    if node_count != len(node_rows):
        raise CompactArtifactError(
            f"node count mismatch: metadata says {node_count}, artifact contains {len(node_rows)}"
        )
    if edge_count != len(edge_rows):
        raise CompactArtifactError(
            f"edge count mismatch: metadata says {edge_count}, artifact contains {len(edge_rows)}"
        )
    if metadata.get("distance_unit") != "km":
        raise CompactArtifactError("metadata distance_unit must be 'km'")
    if metadata.get("coordinate_order") != "lat_lon":
        raise CompactArtifactError("metadata coordinate_order must be 'lat_lon'")
    if metadata.get("checksum_algorithm") != "sha256":
        raise CompactArtifactError("metadata checksum_algorithm must be 'sha256'")
    if metadata.get("adjacency_order") != "insertion":
        raise CompactArtifactError("metadata adjacency_order must be 'insertion'")
    if metadata.get("one_way_default") is not True:
        raise CompactArtifactError("metadata one_way_default must be true")
    source_sha256 = metadata.get("source_sha256")
    if source_sha256 is not None and (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256)
    ):
        raise CompactArtifactError("metadata source_sha256 must be a lowercase SHA-256 hex digest")

    node_ids: list[str] = []
    node_id_set: set[str] = set()
    for index, row in enumerate(node_rows):
        if not isinstance(row, list) or len(row) != 3:
            raise CompactArtifactError(f"node row {index} must be [id, lat, lon]")
        node_id, lat, lon = row
        if not isinstance(node_id, str):
            raise CompactArtifactError(f"node row {index} has an invalid string ID")
        if node_id in node_id_set:
            raise CompactArtifactError(f"duplicate node ID {node_id!r} at row {index}")
        _as_finite_float(lat, f"node {node_id!r} latitude")
        _as_finite_float(lon, f"node {node_id!r} longitude")
        node_ids.append(node_id)
        node_id_set.add(node_id)

    edge_ids: list[str] = []
    edge_by_id: dict[str, list[Any]] = {}
    for index, row in enumerate(edge_rows):
        if not isinstance(row, list) or len(row) != 9:
            raise CompactArtifactError(
                f"edge row {index} must be [id, start, end, distance_km, scenic_score, road_name, road_type, speed_limit_kmh, one_way]"
            )
        edge_id, start, end, distance, score, road_name, road_type, speed, one_way = row
        if not isinstance(edge_id, str) or not edge_id:
            raise CompactArtifactError(f"edge row {index} has an invalid string ID")
        if edge_id in edge_by_id:
            raise CompactArtifactError(f"duplicate edge ID {edge_id!r} at row {index}")
        if not isinstance(start, str) or start not in node_id_set:
            raise CompactArtifactError(f"edge {edge_id!r} references unknown start node {start!r}")
        _as_finite_float(distance, f"edge {edge_id!r} distance_km")
        _as_finite_float(score, f"edge {edge_id!r} scenic_score")
        if isinstance(speed, bool) or not isinstance(speed, int):
            raise CompactArtifactError(f"edge {edge_id!r} speed_limit_kmh must be an integer")
        if not isinstance(road_type, str):
            raise CompactArtifactError(f"edge {edge_id!r} road_type must be a string")
        if speed is not None and (isinstance(speed, bool) or not isinstance(speed, int)):
            raise CompactArtifactError(f"edge {edge_id!r} speed_limit_kmh must be an integer or null")
        if not isinstance(one_way, bool):
            raise CompactArtifactError(f"edge {edge_id!r} one_way must be a boolean")
        edge_ids.append(edge_id)
        edge_by_id[edge_id] = row

    if len(adjacency_rows) != node_count:
        raise CompactArtifactError(
            f"adjacency node count mismatch: expected {node_count}, got {len(adjacency_rows)}"
        )
    adjacency: dict[str, list[tuple[str, bool]]] = {}
    for index, row in enumerate(adjacency_rows):
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str):
            raise CompactArtifactError(f"adjacency row {index} must be [node_id, arcs]")
        node_id, arcs = row
        if node_id not in node_id_set:
            raise CompactArtifactError(f"adjacency row {index} references unknown node {node_id!r}")
        if node_id in adjacency:
            raise CompactArtifactError(f"duplicate adjacency node ID {node_id!r}")
        if not isinstance(arcs, list):
            raise CompactArtifactError(f"adjacency for node {node_id!r} must be an array")
        parsed_arcs: list[tuple[str, bool]] = []
        for arc_index, arc in enumerate(arcs):
            if not isinstance(arc, list) or len(arc) != 2:
                raise CompactArtifactError(f"adjacency arc {node_id!r}[{arc_index}] must be [edge_id, reverse]")
            edge_id, reverse = arc
            if not isinstance(edge_id, str) or edge_id not in edge_by_id:
                raise CompactArtifactError(
                    f"adjacency for node {node_id!r} references unknown edge {edge_id!r}"
                )
            if not isinstance(reverse, bool):
                raise CompactArtifactError(
                    f"adjacency arc {node_id!r}[{arc_index}] reverse must be a boolean"
                )
            start, end, one_way = edge_by_id[edge_id][1], edge_by_id[edge_id][2], edge_by_id[edge_id][8]
            expected_endpoint = end if reverse else start
            if node_id != expected_endpoint:
                direction = "reverse" if reverse else "forward"
                raise CompactArtifactError(
                    f"adjacency arc {node_id!r}[{arc_index}] has invalid {direction} endpoint for edge {edge_id!r}"
                )
            if reverse and one_way:
                raise CompactArtifactError(f"one-way edge {edge_id!r} cannot have a reverse adjacency arc")
            parsed_arcs.append((edge_id, reverse))
        adjacency[node_id] = parsed_arcs

    if list(adjacency) != node_ids:
        raise CompactArtifactError("adjacency node order does not match node insertion order")
    expected_adjacency: dict[str, list[tuple[str, bool]]] = {node_id: [] for node_id in node_ids}
    for row in edge_rows:
        edge_id, start, end, _distance, _score, _name, _type, _speed, one_way = row
        expected_adjacency[start].append((edge_id, False))
        if not one_way:
            expected_adjacency[end].append((edge_id, True))
    if adjacency != expected_adjacency:
        raise CompactArtifactError("adjacency order/content does not match directed edge insertion order")
    actual_arc_count = sum(len(arcs) for arcs in adjacency.values())
    if arc_count != actual_arc_count:
        raise CompactArtifactError(
            f"arc count mismatch: metadata says {arc_count}, artifact contains {actual_arc_count}"
        )


def _read_validated(path: Path) -> tuple[dict[str, Any], CompactArtifactInfo]:
    payload, checksum = _decode_payload(path)
    metadata = _as_mapping(payload["metadata"], "artifact metadata")
    return payload, CompactArtifactInfo(
        path=path,
        node_count=int(metadata["node_count"]),
        edge_count=int(metadata["edge_count"]),
        arc_count=int(metadata["arc_count"]),
        checksum_sha256=checksum,
        source_sha256=metadata.get("source_sha256"),
    )


def validate_compact_artifact(path: Path) -> CompactArtifactInfo:
    """Validate a compact artifact and return its counts/checksum metadata."""
    _payload, info = _read_validated(Path(path))
    return info


def load_compact_graph(path: Path):
    """Validate and materialize a compact artifact as a normal ``RoadGraph``."""
    payload, _info = _read_validated(Path(path))
    # Imported lazily so graph.py can autodetect this format without a cycle at import time.
    from .graph import Edge, Node, RoadGraph

    graph = RoadGraph()
    for node_id, lat, lon in payload["nodes"]:
        graph.add_node(Node(id=node_id, lat=float(lat), lon=float(lon)))
    for row in payload["edges"]:
        (
            edge_id,
            start_node_id,
            end_node_id,
            distance_km,
            scenic_score,
            road_name,
            road_type,
            speed_limit_kmh,
            one_way,
        ) = row
        graph.add_edge(
            Edge(
                id=edge_id,
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                distance_km=float(distance_km),
                scenic_score=float(scenic_score),
                road_name=road_name,
                road_type=road_type,
                speed_limit_kmh=int(speed_limit_kmh) if speed_limit_kmh is not None else 50,
                one_way=one_way,
            )
        )
    return graph


def _graph_payload(graph: Any, source_sha256: str | None) -> dict[str, Any]:
    nodes = [[str(node.id), float(node.lat), float(node.lon)] for node in graph.nodes.values()]
    edges = [
        [
            str(edge.id),
            str(edge.start_node_id),
            str(edge.end_node_id),
            float(edge.distance_km),
            float(edge.scenic_score),
            edge.road_name,
            str(edge.road_type),
            int(edge.speed_limit_kmh) if edge.speed_limit_kmh is not None else None,
            bool(edge.one_way),
        ]
        for edge in graph.edges.values()
    ]
    adjacency = [
        [str(node_id), [[str(edge_id), bool(reverse)] for edge_id, reverse in graph.adjacency.get(node_id, [])]]
        for node_id in graph.nodes
    ]
    return {
        "schema": COMPACT_SCHEMA,
        "version": COMPACT_VERSION,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "arc_count": sum(len(row[1]) for row in adjacency),
            "source_sha256": source_sha256,
            "distance_unit": "km",
            "coordinate_order": "lat_lon",
            "checksum_algorithm": "sha256",
            "adjacency_order": "insertion",
            "one_way_default": True,
        },
        "nodes": nodes,
        "edges": edges,
        "adjacency": adjacency,
    }


def encode_compact_graph(graph: Any, source_sha256: str | None = None) -> bytes:
    """Encode *graph* as a validated compact artifact byte stream."""
    payload = _graph_payload(graph, source_sha256)
    _validate_payload(payload)
    payload_bytes = msgspec.msgpack.encode(payload)
    digest = hashlib.sha256(payload_bytes).digest()
    return COMPACT_MAGIC + _HEADER.pack(COMPACT_VERSION, len(payload_bytes), digest) + payload_bytes


def write_compact_graph(graph: Any, destination: Path, source_sha256: str | None = None) -> CompactArtifactInfo:
    """Atomically write a compact artifact and validate the resulting file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_compact_graph(graph, source_sha256=source_sha256)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
    return validate_compact_artifact(destination)
