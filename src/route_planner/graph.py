from __future__ import annotations

from dataclasses import dataclass
from array import array
import json
import math
from pathlib import Path
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import msgspec
import numpy as np


class _NodeRow(msgspec.Struct):
    # Legacy artifacts contain numeric IDs and numeric values serialized as
    # strings, so normalization remains explicit in the bulk loader.
    id: Any
    lat: Any
    lon: Any


class _EdgeRow(msgspec.Struct):
    id: Any
    start_node_id: Any = msgspec.field(name="start")
    end_node_id: Any = msgspec.field(name="end")
    distance_km: Any
    scenic_score: Any = 5.0
    road_name: Any = None
    road_type: Any = "secondary"
    speed_limit_kmh: Any = 50
    # Historical graph JSONs often omitted one_way.  Keep directed-edge
    # semantics for those artifacts while leaving Edge's public default alone.
    one_way: Any = True


class _GraphRows(msgspec.Struct):
    nodes: List[_NodeRow] = msgspec.field(default_factory=list)
    edges: List[_EdgeRow] = msgspec.field(default_factory=list)


@dataclass
class Node:
    id: str
    lat: float
    lon: float
    _coordinate_mutation_epoch: ClassVar[int] = 0

    def __setattr__(self, name: str, value: Any) -> None:
        coordinate_was_present = name in ("lat", "lon") and name in self.__dict__
        object.__setattr__(self, name, value)
        if coordinate_was_present:
            Node._coordinate_mutation_epoch += 1


    @property
    def coords(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class Edge:
    id: str
    start_node_id: str
    end_node_id: str
    distance_km: float
    scenic_score: float
    road_name: Optional[str] = None
    road_type: str = "secondary"
    speed_limit_kmh: int = 50
    one_way: bool = False
    _mutation_epoch: ClassVar[int] = 0

    def __setattr__(self, name: str, value: Any) -> None:
        public_field_was_present = not name.startswith("_") and name in self.__dict__
        object.__setattr__(self, name, value)
        if public_field_was_present:
            Edge._mutation_epoch += 1

    @property
    def travel_time_minutes(self) -> float:
        speed = max(float(self.speed_limit_kmh), 1.0)
        return (self.distance_km / speed) * 60.0


class _NodeMapping(dict[str, Node]):
    """Dictionary facade that keeps the nearest-node index synchronized."""

    def __init__(self, owner: Any) -> None:
        super().__init__()
        self._owner = owner

    def _changed(self) -> None:
        self._owner._invalidate_nearest_spatial_index()
        self._owner._advance_heuristic_epoch()

    def __setitem__(self, key: str, node: Node) -> None:
        super().__setitem__(key, node)
        self._changed()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._changed()

    def update(self, *args: Any, **kwargs: Node) -> None:
        values = dict(*args, **kwargs)
        for key, node in values.items():
            self[key] = node

    def setdefault(self, key: str, default: Optional[Node] = None) -> Node:
        if key in self:
            return self[key]
        self[key] = default  # type: ignore[assignment]
        return default  # type: ignore[return-value]

    def pop(self, key: str, *args: Any) -> Node:
        value = super().pop(key, *args)
        self._changed()
        return value

    def popitem(self) -> Tuple[str, Node]:
        value = super().popitem()
        self._changed()
        return value

    def clear(self) -> None:
        if self:
            super().clear()
            self._changed()

    def __ior__(self, other: Any) -> "_NodeMapping":
        self.update(other)
        return self


class _EdgeMapping(dict[str, Edge]):
    """Dictionary facade that advances the graph heuristic epoch on mutation."""

    def __init__(self, owner: Any) -> None:
        super().__init__()
        self._owner = owner

    def _changed(self) -> None:
        self._owner._advance_heuristic_epoch()

    def __setitem__(self, key: str, edge: Edge) -> None:
        super().__setitem__(key, edge)
        self._changed()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._changed()

    def update(self, *args: Any, **kwargs: Edge) -> None:
        values = dict(*args, **kwargs)
        for key, edge in values.items():
            self[key] = edge

    def setdefault(self, key: str, default: Optional[Edge] = None) -> Edge:
        if key in self:
            return self[key]
        self[key] = default  # type: ignore[assignment]
        return default  # type: ignore[return-value]

    def pop(self, key: str, *args: Any) -> Edge:
        value = super().pop(key, *args)
        self._changed()
        return value

    def popitem(self) -> Tuple[str, Edge]:
        value = super().popitem()
        self._changed()
        return value

    def clear(self) -> None:
        if self:
            super().clear()
            self._changed()

    def __ior__(self, other: Any) -> "_EdgeMapping":
        self.update(other)
        return self


class RoadGraph:
    def __init__(self) -> None:
        self._heuristic_structure_epoch = 0
        self.nodes: Dict[str, Node] = _NodeMapping(self)
        self.edges: Dict[str, Edge] = _EdgeMapping(self)
        # node_id -> list[(edge_id, is_reverse_view)]
        self.adjacency: Dict[str, List[Tuple[str, bool]]] = {}
        # Built on the first nearest-node query and discarded whenever the
        # node mapping changes or any Node coordinate is mutated.  The arrays
        # hold only primitive index data; node objects and their IDs remain
        # owned by ``self.nodes``.  The leading epoch is a class-wide
        # coordinate-mutation snapshot used to detect stale coordinates.
        self._nearest_spatial_index: Optional[
            Tuple[int, Tuple[str, ...], array, array, array, array, array, array]
        ] = None

    def _advance_heuristic_epoch(self) -> None:
        self._heuristic_structure_epoch += 1

    def _heuristic_cache_stamp(self) -> Tuple[int, int, int]:
        return (
            self._heuristic_structure_epoch,
            Node._coordinate_mutation_epoch,
            Edge._mutation_epoch,
        )

    def _invalidate_nearest_spatial_index(self) -> None:
        self._nearest_spatial_index = None

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.adjacency.setdefault(node.id, [])

    def _build_nearest_spatial_index(
        self,
    ) -> Tuple[int, Tuple[str, ...], array, array, array, array, array, array]:
        """Build a compact balanced 2-D kd-tree over the current nodes.

        ``order`` is temporary compact construction storage.  The retained
        index has two coordinate arrays and four integer arrays, plus a tuple
        of IDs; it does not create per-node wrapper objects.
        """
        coordinate_epoch = Node._coordinate_mutation_epoch
        node_ids = tuple(self.nodes)
        latitudes = array("d", (self.nodes[node_id].lat for node_id in node_ids))
        longitudes = array("d", (self.nodes[node_id].lon for node_id in node_ids))
        order = array("i", range(len(node_ids)))
        # NumPy views let partitioning reorder the existing rank buffer in C
        order_view = np.frombuffer(order, dtype=np.intc)
        latitude_view = np.frombuffer(latitudes, dtype=np.dtype("d"))
        longitude_view = np.frombuffer(longitudes, dtype=np.dtype("d"))
        tree_ranks = array("i")
        left_children = array("i")
        right_children = array("i")
        subtree_min_ranks = array("i")


        def select(lo: int, hi: int, target: int, axis: int) -> None:
            """Place the target order statistic in ``order[target]``."""
            if hi - lo <= 1:
                return
            values = latitude_view if axis == 0 else longitude_view
            segment = order_view[lo:hi]
            # Indexing by ``segment`` creates only a temporary coordinate
            # copy; argpartition itself performs selection in NumPy's C loop.
            permutation = np.argpartition(values[segment], target - lo)
            segment[:] = segment[permutation]

        def build(lo: int, hi: int, depth: int) -> int:
            if lo >= hi:
                return -1
            axis = depth & 1
            mid = (lo + hi) // 2
            select(lo, hi, mid, axis)
            rank = order[mid]
            position = len(tree_ranks)
            tree_ranks.append(rank)
            left_children.append(-1)
            right_children.append(-1)
            subtree_min_ranks.append(rank)
            left = build(lo, mid, depth + 1)
            right = build(mid + 1, hi, depth + 1)
            left_children[position] = left
            right_children[position] = right
            minimum = rank
            if left >= 0:
                minimum = min(minimum, subtree_min_ranks[left])
            if right >= 0:
                minimum = min(minimum, subtree_min_ranks[right])
            subtree_min_ranks[position] = minimum
            return position

        build(0, len(order), 0)
        return (
            coordinate_epoch,
            node_ids,
            latitudes,
            longitudes,
            tree_ranks,
            left_children,
            right_children,
            subtree_min_ranks,
        )

    def add_edge(self, edge: Edge) -> None:
        if edge.start_node_id not in self.nodes:
            raise ValueError(f"Unknown start node: {edge.start_node_id}")
        if edge.end_node_id not in self.nodes:
            raise ValueError(f"Unknown end node: {edge.end_node_id}")
        if edge.id in self.edges:
            raise ValueError(f"Duplicate edge id: {edge.id}")

        self.edges[edge.id] = edge
        self.adjacency.setdefault(edge.start_node_id, []).append((edge.id, False))
        if not edge.one_way:
            self.adjacency.setdefault(edge.end_node_id, []).append((edge.id, True))
    def _bulk_load(self, nodes: List[_NodeRow], edges: List[_EdgeRow]) -> None:
        """Populate rows without firing mapping mutation hooks per item.

        The graph is private to ``load`` while rows are normalized and
        inserted sequentially.  Node assignment retains ``add_node``'s
        overwrite behavior: duplicate IDs replace the node while preserving
        insertion position and existing adjacency.
        """
        # The graph is private to ``load`` until this method returns, so
        # normalize and insert directly.  Base-dict calls bypass per-row
        # nearest-index/heuristic hooks while preserving normal dict order and
        # add_node's duplicate overwrite behavior.
        for row in nodes:
            node_id = str(row.id)
            dict.__setitem__(
                self.nodes,
                node_id,
                Node(
                    id=node_id,
                    lat=float(row.lat),
                    lon=float(row.lon),
                ),
            )
            self.adjacency.setdefault(node_id, [])

        for row in edges:
            edge_id = str(row.id)
            start_node_id = str(row.start_node_id)
            end_node_id = str(row.end_node_id)
            if start_node_id not in self.nodes:
                raise ValueError(f"Unknown start node: {start_node_id}")
            if end_node_id not in self.nodes:
                raise ValueError(f"Unknown end node: {end_node_id}")
            if edge_id in self.edges:
                raise ValueError(f"Duplicate edge id: {edge_id}")
            road_type = str(row.road_type)
            edge = Edge(
                id=edge_id,
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                distance_km=float(row.distance_km),
                scenic_score=float(row.scenic_score),
                road_name=row.road_name,
                road_type=road_type,
                speed_limit_kmh=_parse_speed_limit_kmh(
                    row.speed_limit_kmh,
                    road_type,
                ),
                one_way=_parse_one_way(row.one_way, default=True),
            )
            dict.__setitem__(self.edges, edge_id, edge)
            self.adjacency.setdefault(start_node_id, []).append((edge_id, False))
            if not edge.one_way:
                self.adjacency.setdefault(end_node_id, []).append((edge_id, True))

        if nodes:
            self._invalidate_nearest_spatial_index()
        if nodes or edges:
            self._advance_heuristic_epoch()

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def get_edges(self, node_id: str) -> List[Edge]:
        out: List[Edge] = []
        for edge_id, reverse in self.adjacency.get(node_id, []):
            edge = self.edges[edge_id]
            if not reverse:
                out.append(edge)
                continue
            out.append(
                Edge(
                    id=f"{edge.id}::rev",
                    start_node_id=edge.end_node_id,
                    end_node_id=edge.start_node_id,
                    distance_km=edge.distance_km,
                    scenic_score=edge.scenic_score,
                    road_name=edge.road_name,
                    road_type=edge.road_type,
                    speed_limit_kmh=edge.speed_limit_kmh,
                    one_way=False,
                )
            )
        return out

    def find_nearest_node(self, lat: float, lon: float) -> Node:
        return self.find_nearest_node_with_distance(lat, lon)[0]

    def find_nearest_node_with_distance(self, lat: float, lon: float) -> tuple[Node, float]:
        if not self.nodes:
            raise ValueError("Road graph has no nodes")

        index = self._nearest_spatial_index
        if index is None or index[0] != Node._coordinate_mutation_epoch:
            index = self._build_nearest_spatial_index()
            self._nearest_spatial_index = index
        (
            _coordinate_epoch,
            node_ids,
            latitudes,
            longitudes,
            tree_ranks,
            left_children,
            right_children,
            subtree_min_ranks,
        ) = index

        # The tree is searched by the same squared lat/lon distance used by
        # the historical scan.  A plane bound is enough to prune the far
        # branch, while ``<=`` retains equal-distance candidates needed for
        # insertion-order tie semantics.
        query_lat = float(lat)
        query_lon = float(lon)
        best_rank = len(node_ids)
        best_dist = float("inf")
        best_found = False
        stack: List[Tuple[int, int]] = [(0, 0)]
        while stack:
            position, depth = stack.pop()
            if position < 0:
                continue
            rank = tree_ranks[position]
            dlat = latitudes[rank] - query_lat
            dlon = longitudes[rank] - query_lon
            distance = dlat**2 + dlon**2
            if distance < best_dist or (
                best_found and distance == best_dist and rank < best_rank
            ):
                best_dist = distance
                best_rank = rank
                best_found = True

            axis = depth & 1
            query_coordinate = query_lat if axis == 0 else query_lon
            split_coordinate = latitudes[rank] if axis == 0 else longitudes[rank]
            delta = query_coordinate - split_coordinate
            if delta < 0.0:
                near, far = left_children[position], right_children[position]
            else:
                near, far = right_children[position], left_children[position]

            plane_distance = delta**2
            if far >= 0 and (
                plane_distance < best_dist
                or (
                    plane_distance == best_dist
                    and subtree_min_ranks[far] < best_rank
                )
            ):
                stack.append((far, depth + 1))
            if near >= 0:
                stack.append((near, depth + 1))

        assert best_found
        best = self.nodes[node_ids[best_rank]]
        return best, _haversine_km(query_lat, query_lon, best.lat, best.lon)

    def save(self, path: Path) -> None:
        data = {
            "nodes": [
                {"id": n.id, "lat": n.lat, "lon": n.lon}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "start": e.start_node_id,
                    "end": e.end_node_id,
                    "distance_km": e.distance_km,
                    "scenic_score": e.scenic_score,
                    "road_name": e.road_name,
                    "road_type": e.road_type,
                    "speed_limit_kmh": e.speed_limit_kmh,
                    "one_way": e.one_way,
                }
                for e in self.edges.values()
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        # Decode bytes directly into compact typed rows.  This avoids a
        # temporary dict for every node and edge while retaining the legacy
        # JSON object/array schema.
        rows = msgspec.json.decode(path.read_bytes(), type=_GraphRows, strict=True)
        graph = cls()
        graph._bulk_load(rows.nodes, rows.edges)
        return graph

    @classmethod
    def from_geojson(cls, path: Path) -> "RoadGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != "FeatureCollection":
            raise ValueError("GeoJSON must be a FeatureCollection")

        graph = cls()
        node_index: Dict[Tuple[float, float], str] = {}
        edge_counter = 0

        def node_id(lat: float, lon: float) -> str:
            key = (round(float(lat), 7), round(float(lon), 7))
            if key in node_index:
                return node_index[key]
            nid = f"n{len(node_index)}"
            node_index[key] = nid
            graph.add_node(Node(id=nid, lat=float(lat), lon=float(lon)))
            return nid

        for feat in data.get("features", []):
            geom = feat.get("geometry", {}) or {}
            if geom.get("type") != "LineString":
                continue
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue

            props = feat.get("properties", {}) or {}
            base_edge_id = props.get("id")
            road_name = props.get("road_name")
            road_type = _normalize_road_type(props.get("road_type", "secondary"))
            scenic_score = float(props.get("scenic_score", 5.0))
            speed_limit = _parse_speed_limit_kmh(props.get("speed_limit_kmh"), road_type)
            one_way = _parse_one_way(props.get("one_way", props.get("oneway")), default=True)
            if "bidirectional" in props:
                one_way = not _parse_bool(props.get("bidirectional"), default=False)

            for idx in range(len(coords) - 1):
                lon1, lat1 = coords[idx]
                lon2, lat2 = coords[idx + 1]
                start = node_id(lat1, lon1)
                end = node_id(lat2, lon2)
                dist_km = _haversine_km(lat1, lon1, lat2, lon2)
                edge_id = f"{base_edge_id}_{idx}" if base_edge_id else f"e{edge_counter}"
                edge_counter += 1
                graph.add_edge(
                    Edge(
                        id=edge_id,
                        start_node_id=start,
                        end_node_id=end,
                        distance_km=float(dist_km),
                        scenic_score=float(scenic_score),
                        road_name=road_name,
                        road_type=road_type,
                        speed_limit_kmh=speed_limit,
                        one_way=one_way,
                    )
                )

        if not graph.nodes or not graph.edges:
            raise ValueError(f"No usable LineString edges found in {path}")
        return graph

    @classmethod
    def from_osm(cls, osm_file: Path, scenic_scores: Optional[Dict[str, float]] = None) -> "RoadGraph":
        try:
            import osmnx as ox
        except ImportError as exc:
            raise ImportError("osmnx is required for OSM import. Run: uv sync --extra geo") from exc

        scenic_scores = scenic_scores or {}
        osm_path = Path(osm_file)
        if osm_path.suffix == ".graphml":
            G = ox.load_graphml(osm_path)
        else:
            G = ox.graph_from_xml(osm_path)
        return _graph_from_osmnx(G, scenic_scores)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def _graph_from_osmnx(G: Any, scenic_scores: Dict[str, float]) -> RoadGraph:
    graph = RoadGraph()

    for node_id, data in G.nodes(data=True):
        graph.add_node(
            Node(
                id=str(node_id),
                lat=float(data.get("y")),
                lon=float(data.get("x")),
            )
        )

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_id = f"{u}-{v}-{key}"
        scenic_score = float(scenic_scores.get(str(data.get("osmid", edge_id)), 5.0))
        road_name = data.get("name")
        road_type = _normalize_road_type(data.get("highway", "secondary"))
        speed_kmh = _parse_speed_limit_kmh(data.get("maxspeed"), road_type)
        # osmnx graphs are directed. Keep this edge as directed by default
        # and only synthesize reverse traversal when we are sure it's needed.
        one_way = _parse_one_way(data.get("oneway"), default=True)
        if not one_way and G.has_edge(v, u):
            one_way = True

        start_id = str(u)
        end_id = str(v)
        start_node = graph.get_node(start_id)
        end_node = graph.get_node(end_id)

        # OSMnx stores LineString coordinates as (lon, lat).  Keep the original
        # graph endpoints and add deterministic nodes for each interior point.
        geometry = data.get("geometry")
        geometry_coords: List[Tuple[float, float]] = []
        if geometry is not None:
            try:
                raw_coords = list(geometry.coords)
            except (AttributeError, NotImplementedError, TypeError):
                raw_coords = geometry if isinstance(geometry, (list, tuple)) else []
            for point in raw_coords:
                if len(point) >= 2:
                    geometry_coords.append((float(point[0]), float(point[1])))

        if len(geometry_coords) >= 2:
            start_xy = (start_node.lon, start_node.lat)
            end_xy = (end_node.lon, end_node.lat)
            forward_error = (
                (geometry_coords[0][0] - start_xy[0]) ** 2
                + (geometry_coords[0][1] - start_xy[1]) ** 2
                + (geometry_coords[-1][0] - end_xy[0]) ** 2
                + (geometry_coords[-1][1] - end_xy[1]) ** 2
            )
            reverse_error = (
                (geometry_coords[-1][0] - start_xy[0]) ** 2
                + (geometry_coords[-1][1] - start_xy[1]) ** 2
                + (geometry_coords[0][0] - end_xy[0]) ** 2
                + (geometry_coords[0][1] - end_xy[1]) ** 2
            )
            if reverse_error < forward_error:
                geometry_coords.reverse()
            interior_coords = geometry_coords[1:-1]
        else:
            interior_coords = []

        node_ids = [start_id]
        for coordinate_index, (lon, lat) in enumerate(interior_coords, start=1):
            intermediate_id = f"{u}-{v}-{key}-coord-{coordinate_index}"
            graph.add_node(Node(id=intermediate_id, lat=lat, lon=lon))
            node_ids.append(intermediate_id)
        node_ids.append(end_id)

        for coordinate_index, (segment_start, segment_end) in enumerate(
            zip(node_ids, node_ids[1:])
        ):
            segment_start_node = graph.get_node(segment_start)
            segment_end_node = graph.get_node(segment_end)
            segment_distance_km = _haversine_km(
                segment_start_node.lat,
                segment_start_node.lon,
                segment_end_node.lat,
                segment_end_node.lon,
            )
            graph.add_edge(
                Edge(
                    id=f"{u}-{v}-{key}-segment-{coordinate_index}",
                    start_node_id=segment_start,
                    end_node_id=segment_end,
                    distance_km=float(segment_distance_km),
                    scenic_score=float(max(0.0, min(10.0, scenic_score))),
                    road_name=road_name,
                    road_type=road_type,
                    speed_limit_kmh=speed_kmh,
                    one_way=one_way,
                )
            )

    return graph


_ROAD_TYPE_SPEED_KMH = {
    "motorway": 100,
    "trunk": 90,
    "primary": 80,
    "secondary": 60,
    "tertiary": 50,
    "residential": 35,
    "service": 25,
    "unclassified": 40,
    "living_street": 20,
}


def _normalize_road_type(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        raw = raw[0]
    value = str(raw).strip().lower() if raw is not None else ""
    return value if value else "secondary"


def _parse_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if text in {"0", "false", "no", "n", "f", "off"}:
        return False
    return default


def _parse_one_way(raw: Any, default: bool) -> bool:
    """
    Parse one-way flags from OSM/GeoJSON-style values.

    OSM may encode oneway as yes/no/1/0/-1.
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return default

    text = str(raw).strip().lower()
    if text in {"-1", "reverse"}:
        return True
    return _parse_bool(raw, default=default)


def _parse_speed_limit_kmh(raw_speed: Any, road_type: str) -> int:
    """
    Parse OSM maxspeed into km/h.

    Handles values like:
      - 50
      - "50"
      - "35 mph"
      - "50 km/h"
      - "50;70"
      - ["50 mph", ...]
    Falls back to road-type defaults when missing/unparseable.
    """
    fallback = int(_ROAD_TYPE_SPEED_KMH.get(road_type, 50))
    if raw_speed is None:
        return fallback

    if isinstance(raw_speed, list):
        raw_speed = raw_speed[0] if raw_speed else None
        if raw_speed is None:
            return fallback

    if isinstance(raw_speed, (int, float)):
        speed = float(raw_speed)
        if speed <= 0:
            return fallback
        return int(round(min(max(speed, 10.0), 140.0)))

    text = str(raw_speed).strip().lower()
    if not text:
        return fallback
    if ";" in text:
        text = text.split(";", 1)[0].strip()

    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return fallback

    value = float(match.group(0))
    if value <= 0:
        return fallback

    if "mph" in text:
        value *= 1.60934

    value = min(max(value, 10.0), 140.0)
    return int(round(value))
