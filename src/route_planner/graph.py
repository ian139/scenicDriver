from __future__ import annotations

from dataclasses import dataclass
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Callable, ClassVar, Dict, Generic, List, Optional, Tuple, TypeVar, overload

import msgspec
import numpy as np

_KD_SMALL_SUBTREE_CUTOFF = 32
_EDGE_PROJECTION_CHUNK_SIZE = 65_536
_EDGE_PROJECTION_TIE_TOLERANCE_KM = 1e-9
_CANCELLATION_CHECK_INTERVAL = 1024


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
    speed_limit_kmh: Any = None
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

@dataclass(frozen=True)
class EdgeProjection:
    edge: Edge
    fraction: float
    lat: float
    lon: float
    snap_distance_km: float


class _ReverseEdgeView:
    """Allocation-free reverse traversal view over one canonical edge."""

    __slots__ = ("_edge",)

    def __init__(self, edge: Edge) -> None:
        self._edge = edge

    @property
    def id(self) -> str:
        return f"{self._edge.id}::rev"
    @property
    def canonical_edge_id(self) -> str:
        return str(self._edge.id)

    @property
    def direction(self) -> str:
        return "reverse"

    @property
    def traversal_id(self) -> str:
        return f"reverse:{self._edge.id}"

    @property
    def start_node_id(self) -> str:
        return self._edge.end_node_id

    @property
    def end_node_id(self) -> str:
        return self._edge.start_node_id

    @property
    def distance_km(self) -> float:
        return self._edge.distance_km

    @property
    def scenic_score(self) -> float:
        return self._edge.scenic_score

    @property
    def road_name(self) -> Optional[str]:
        return self._edge.road_name

    @property
    def road_type(self) -> str:
        return self._edge.road_type

    @property
    def speed_limit_kmh(self) -> int:
        return self._edge.speed_limit_kmh

    @property
    def travel_time_minutes(self) -> float:
        return self._edge.travel_time_minutes

    @property
    def one_way(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._edge, name)


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
        if key not in self:
            return super().pop(key, *args)
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
        if key not in self:
            return super().pop(key, *args)
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
        self._reverse_edge_views: Dict[str, _ReverseEdgeView] = {}
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
        self._nearest_edge_projection_index: Optional[Tuple[Any, ...]] = None
        self.artifact_metadata: dict[str, Any] = {}

    def _advance_heuristic_epoch(self) -> None:
        self._heuristic_structure_epoch += 1
        self._reverse_edge_views.clear()
        self._invalidate_nearest_edge_projection_index()


    def _heuristic_cache_stamp(self) -> Tuple[int, int, int]:
        return (
            self._heuristic_structure_epoch,
            Node._coordinate_mutation_epoch,
            Edge._mutation_epoch,
        )

    def _invalidate_nearest_spatial_index(self) -> None:
        self._nearest_spatial_index = None
        self._nearest_edge_projection_index = None

    def _invalidate_nearest_edge_projection_index(self) -> None:
        self._nearest_edge_projection_index = None


    def _build_nearest_edge_projection_index(
        self,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> Tuple[Any, ...]:
        """Build compact primitive arrays for canonical finite edge segments."""
        if check_cancelled is not None:
            check_cancelled()
        stamp = self._heuristic_cache_stamp()
        edge_ids: List[str] = []
        edge_keys: List[str] = []
        road_type_codes: List[int] = []
        start_latitudes: List[float] = []
        start_longitudes: List[float] = []
        end_latitudes: List[float] = []
        end_longitudes: List[float] = []
        road_type_index: Dict[str, int] = {}

        for edge_index, (edge_key, edge) in enumerate(self.edges.items()):
            if (
                check_cancelled is not None
                and edge_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            start = self.nodes.get(edge.start_node_id)
            end = self.nodes.get(edge.end_node_id)
            if start is None or end is None:
                continue
            try:
                start_lat = float(start.lat)
                start_lon = float(start.lon)
                end_lat = float(end.lat)
                end_lon = float(end.lon)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(
                math.isfinite(value)
                for value in (start_lat, start_lon, end_lat, end_lon)
            ):
                continue
            road_type = _normalize_road_type(edge.road_type)
            code = road_type_index.get(road_type)
            if code is None:
                code = len(road_type_index)
                road_type_index[road_type] = code
            edge_ids.append(str(edge.id))
            edge_keys.append(edge_key)
            road_type_codes.append(code)
            start_latitudes.append(start_lat)
            start_longitudes.append(start_lon)
            end_latitudes.append(end_lat)
            end_longitudes.append(end_lon)

        if check_cancelled is not None:
            check_cancelled()
        index = (
            stamp,
            tuple(edge_ids),
            tuple(edge_keys),
            tuple(road_type_index),
            np.asarray(road_type_codes, dtype=np.int32),
            np.asarray(start_latitudes, dtype=np.float64),
            np.asarray(start_longitudes, dtype=np.float64),
            np.asarray(end_latitudes, dtype=np.float64),
            np.asarray(end_longitudes, dtype=np.float64),
        )
        if check_cancelled is not None:
            check_cancelled()
        return index

    @staticmethod
    def _project_edge_chunk(
        query_lat: float,
        query_lon: float,
        longitude_scale: float,
        start_latitudes: np.ndarray,
        start_longitudes: np.ndarray,
        end_latitudes: np.ndarray,
        end_longitudes: np.ndarray,
        start: int,
        stop: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start_lat = start_latitudes[start:stop]
        start_lon = start_longitudes[start:stop]
        delta_lat = end_latitudes[start:stop] - start_lat
        delta_lon = (end_longitudes[start:stop] - start_lon) * longitude_scale
        query_delta_lat = query_lat - start_lat
        query_delta_lon = (query_lon - start_lon) * longitude_scale
        denominator = delta_lat * delta_lat + delta_lon * delta_lon
        numerator = query_delta_lat * delta_lat + query_delta_lon * delta_lon
        fractions = np.zeros_like(denominator)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            np.divide(numerator, denominator, out=fractions, where=denominator > 0.0)
        np.clip(fractions, 0.0, 1.0, out=fractions)
        projected_latitudes = start_lat + fractions * (end_latitudes[start:stop] - start_lat)
        projected_longitudes = start_lon + fractions * (end_longitudes[start:stop] - start_lon)

        with np.errstate(over="ignore", invalid="ignore"):
            dlat = np.radians(projected_latitudes - query_lat)
            dlon = np.radians(projected_longitudes - query_lon)
            haversine = np.sin(dlat / 2.0) ** 2
            haversine += (
                np.cos(np.radians(query_lat))
                * np.cos(np.radians(projected_latitudes))
                * np.sin(dlon / 2.0) ** 2
            )
            np.clip(haversine, 0.0, 1.0, out=haversine)
            np.sqrt(haversine, out=haversine)
            np.arcsin(haversine, out=haversine)
            haversine *= 2.0 * 6371.0
        return fractions, projected_latitudes, projected_longitudes, haversine
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.adjacency.setdefault(node.id, [])

    def _build_nearest_spatial_index(
        self,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> Tuple[int, Tuple[str, ...], array, array, array, array, array, array]:
        """Build a compact balanced 2-D kd-tree over the current nodes.

        ``order`` is temporary compact construction storage.  The retained
        index has two coordinate arrays and four integer arrays, plus a tuple
        of IDs; it does not create per-node wrapper objects.
        """
        if check_cancelled is not None:
            check_cancelled()
        coordinate_epoch = Node._coordinate_mutation_epoch
        node_ids = tuple(self.nodes)
        if check_cancelled is not None:
            check_cancelled()
        latitudes = array("d")
        longitudes = array("d")
        for node_index, node_id in enumerate(node_ids):
            if (
                check_cancelled is not None
                and node_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            latitudes.append(self.nodes[node_id].lat)
            longitudes.append(self.nodes[node_id].lon)
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
            if hi - lo <= _KD_SMALL_SUBTREE_CUTOFF:
                coordinates = latitudes if axis == 0 else longitudes
                sorted_ranks = sorted(
                    order[lo:hi],
                    key=lambda rank: (coordinates[rank], rank),
                )
                order[lo:hi] = array("i", sorted_ranks)
                return
            values = latitude_view if axis == 0 else longitude_view
            segment = order_view[lo:hi]
            # Indexing by ``segment`` creates only a temporary coordinate
            # copy; argpartition itself performs selection in NumPy's C loop.
            if check_cancelled is not None:
                check_cancelled()
            permutation = np.argpartition(values[segment], target - lo)
            if check_cancelled is not None:
                check_cancelled()
            segment[:] = segment[permutation]

        build_calls = 0

        def build(lo: int, hi: int, depth: int) -> int:
            nonlocal build_calls
            if check_cancelled is not None and build_calls & (
                _CANCELLATION_CHECK_INTERVAL - 1
            ) == 0:
                check_cancelled()
            build_calls += 1
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
        if check_cancelled is not None:
            check_cancelled()
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
    def _bulk_load(
        self,
        nodes: Iterable[_NodeRow],
        edges: Iterable[_EdgeRow],
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> tuple[bool, bool]:
        """Populate rows without firing mapping mutation hooks per item."""
        if check_cancelled is not None:
            check_cancelled()
        saw_nodes = False
        saw_edges = False
        # The graph is private to ``load`` until this method returns, so
        # normalize and insert directly.  Base-dict calls bypass per-row
        # nearest-index/heuristic hooks while preserving normal dict order and
        # add_node's duplicate overwrite behavior.
        for node_index, row in enumerate(nodes):
            if (
                check_cancelled is not None
                and node_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            saw_nodes = True
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

        if check_cancelled is not None:
            check_cancelled()
        for edge_index, row in enumerate(edges):
            if (
                check_cancelled is not None
                and edge_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
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

        if check_cancelled is not None:
            check_cancelled()
        if saw_nodes:
            self._invalidate_nearest_spatial_index()
        if saw_nodes or saw_edges:
            self._advance_heuristic_epoch()
        if check_cancelled is not None:
            check_cancelled()
        return saw_nodes, saw_edges

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
    def iter_edges(self, node_id: str):
        """Yield outgoing traversals without allocating a result list."""
        for edge_id, reverse in self.adjacency.get(node_id, ()):
            edge = self.edges[edge_id]
            if not reverse:
                yield edge
                continue
            reverse_view = self._reverse_edge_views.get(edge_id)
            if reverse_view is None:
                reverse_view = _ReverseEdgeView(edge)
                self._reverse_edge_views[edge_id] = reverse_view
            yield reverse_view


    def find_nearest_node(
        self,
        lat: float,
        lon: float,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> Node:
        if check_cancelled is not None:
            check_cancelled()
        return self.find_nearest_node_with_distance(
            lat,
            lon,
            check_cancelled=check_cancelled,
        )[0]

    def find_nearest_node_with_distance(
        self,
        lat: float,
        lon: float,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> tuple[Node, float]:
        if check_cancelled is not None:
            check_cancelled()
        if not self.nodes:
            raise ValueError("Road graph has no nodes")

        index = self._nearest_spatial_index
        if index is None or index[0] != Node._coordinate_mutation_epoch:
            index = self._build_nearest_spatial_index(
                check_cancelled=check_cancelled,
            )
            if check_cancelled is not None:
                check_cancelled()
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
        search_steps = 0
        while stack:
            if (
                check_cancelled is not None
                and search_steps & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            search_steps += 1
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
        if check_cancelled is not None:
            check_cancelled()
        best = self.nodes[node_ids[best_rank]]
        if check_cancelled is not None:
            check_cancelled()
        snap_distance = _haversine_km(query_lat, query_lon, best.lat, best.lon)
        if check_cancelled is not None:
            check_cancelled()
        return best, snap_distance

    def find_nearest_edge_positions_with_distance(
        self,
        lat: float,
        lon: float,
        *,
        excluded_road_types: frozenset = frozenset(),
        check_cancelled: Callable[[], None] | None = None,
    ) -> tuple[list[EdgeProjection], float]:
        """Return canonical edge projections tied at the nearest segment."""
        if check_cancelled is not None:
            check_cancelled()
        query_lat = float(lat)
        query_lon = float(lon)
        if not math.isfinite(query_lat) or not math.isfinite(query_lon):
            raise ValueError("Query coordinates must be finite")

        stamp = self._heuristic_cache_stamp()
        index = self._nearest_edge_projection_index
        if index is None or index[0] != stamp:
            index = self._build_nearest_edge_projection_index(
                check_cancelled=check_cancelled,
            )
            if check_cancelled is not None:
                check_cancelled()
            self._nearest_edge_projection_index = index
        (
            _stamp,
            edge_ids,
            edge_keys,
            road_type_names,
            road_type_codes,
            start_latitudes,
            start_longitudes,
            end_latitudes,
            end_longitudes,
        ) = index
        if not edge_ids:
            raise ValueError("Road graph has no eligible finite segment")

        excluded = {
            str(road_type).strip().lower()
            for road_type in (excluded_road_types or ())
        }
        if check_cancelled is not None:
            check_cancelled()
        allowed_types = np.asarray(
            [road_type not in excluded for road_type in road_type_names],
            dtype=np.bool_,
        )
        if check_cancelled is not None:
            check_cancelled()
        if not np.any(allowed_types):
            raise ValueError("Road graph has no eligible finite segment")

        longitude_scale = math.cos(math.radians(query_lat))
        best_distance = float("inf")
        edge_count = len(edge_ids)
        for start in range(0, edge_count, _EDGE_PROJECTION_CHUNK_SIZE):
            stop = min(start + _EDGE_PROJECTION_CHUNK_SIZE, edge_count)
            if check_cancelled is not None:
                check_cancelled()
            _fractions, _projected_latitudes, _projected_longitudes, distances = (
                self._project_edge_chunk(
                    query_lat,
                    query_lon,
                    longitude_scale,
                    start_latitudes,
                    start_longitudes,
                    end_latitudes,
                    end_longitudes,
                    start,
                    stop,
                )
            )
            if check_cancelled is not None:
                check_cancelled()
            eligible = allowed_types[road_type_codes[start:stop]]
            finite = eligible & np.isfinite(distances)
            if np.any(finite):
                best_distance = min(best_distance, float(np.min(distances[finite])))

        if not math.isfinite(best_distance):
            raise ValueError("Road graph has no eligible finite segment")

        cutoff = best_distance + _EDGE_PROJECTION_TIE_TOLERANCE_KM
        projections: List[Tuple[str, str, EdgeProjection]] = []
        for start in range(0, edge_count, _EDGE_PROJECTION_CHUNK_SIZE):
            stop = min(start + _EDGE_PROJECTION_CHUNK_SIZE, edge_count)
            if check_cancelled is not None:
                check_cancelled()
            fractions, projected_latitudes, projected_longitudes, distances = (
                self._project_edge_chunk(
                    query_lat,
                    query_lon,
                    longitude_scale,
                    start_latitudes,
                    start_longitudes,
                    end_latitudes,
                    end_longitudes,
                    start,
                    stop,
                )
            )
            if check_cancelled is not None:
                check_cancelled()
            eligible = allowed_types[road_type_codes[start:stop]]
            tied = eligible & np.isfinite(distances) & (distances <= cutoff)
            if check_cancelled is not None:
                check_cancelled()
            for projection_index, local_index in enumerate(np.flatnonzero(tied)):
                if (
                    check_cancelled is not None
                    and projection_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
                ):
                    check_cancelled()
                index_in_cache = start + int(local_index)
                projections.append(
                    (
                        edge_ids[index_in_cache],
                        str(edge_keys[index_in_cache]),
                        EdgeProjection(
                            edge=self.edges[edge_keys[index_in_cache]],
                            fraction=float(fractions[local_index]),
                            lat=float(projected_latitudes[local_index]),
                            lon=float(projected_longitudes[local_index]),
                            snap_distance_km=float(distances[local_index]),
                        ),
                    )
                )
        if check_cancelled is not None:
            check_cancelled()
        projections.sort(key=lambda item: (item[0], item[1]))
        if check_cancelled is not None:
            check_cancelled()
        return [projection for _edge_id, _edge_key, projection in projections], best_distance
    def save(
        self,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        path = Path(path)
        if path.suffix.lower() == ".sqlite3":
            metadata_values = dict(self.artifact_metadata)
            if metadata is not None:
                metadata_values.update(metadata)
            _write_sqlite_graph(
                path,
                _iter_graph_rows(self.nodes.values(), self.edges.values()),
                metadata=metadata_values,
            )
            return

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
    def load(
        cls,
        path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> "RoadGraph":
        if check_cancelled is not None:
            check_cancelled()
        path = Path(path)
        if check_cancelled is not None:
            check_cancelled()
        if path.suffix.lower() == ".sqlite3":
            return _load_sqlite_graph(path, check_cancelled=check_cancelled)
        # Decode bytes directly into compact typed rows.  This avoids a
        # temporary dict for every node and edge while retaining the legacy
        # JSON object/array schema.
        if check_cancelled is not None:
            check_cancelled()
        payload = path.read_bytes()
        if check_cancelled is not None:
            check_cancelled()
        rows = msgspec.json.decode(payload, type=_GraphRows, strict=True)
        if check_cancelled is not None:
            check_cancelled()
        graph = cls()
        graph._bulk_load(rows.nodes, rows.edges, check_cancelled=check_cancelled)
        if check_cancelled is not None:
            check_cancelled()
        return graph
    @classmethod
    def from_geojson(
        cls,
        path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> "RoadGraph":
        if check_cancelled is not None:
            check_cancelled()
        raw = path.read_text(encoding="utf-8")
        if check_cancelled is not None:
            check_cancelled()
        data = json.loads(raw)
        if check_cancelled is not None:
            check_cancelled()
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
        for feature_index, feat in enumerate(data.get("features", [])):
            if (
                check_cancelled is not None
                and feature_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
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
            for segment_index, idx in enumerate(range(len(coords) - 1)):
                if (
                    check_cancelled is not None
                    and segment_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
                ):
                    check_cancelled()
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
        if check_cancelled is not None:
            check_cancelled()
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


_T = TypeVar("_T")
_TraversalRef = Tuple[str, bool]


class _StructuralOverlayMapping(Mapping[str, _T], Generic[_T]):
    """Read-only merged view with base entries ordered before local entries."""

    __slots__ = ("_base", "_local")

    def __init__(self, base: Mapping[str, _T], local: Dict[str, _T]) -> None:
        self._base = base
        self._local = local

    def __getitem__(self, key: str) -> _T:
        try:
            return self._local[key]
        except KeyError:
            return self._base[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._base
        for key in self._local:
            if key not in self._base:
                yield key

    def __len__(self) -> int:
        return len(self._base) + sum(
            1 for key in self._local if key not in self._base
        )


class _TraversalSequence(Sequence[_TraversalRef]):
    """Read-only concatenation of base and request-local traversal refs."""

    __slots__ = ("_base", "_local")

    def __init__(
        self,
        base: Sequence[_TraversalRef],
        local: Sequence[_TraversalRef],
    ) -> None:
        self._base = base
        self._local = local

    def __iter__(self) -> Iterator[_TraversalRef]:
        yield from self._base
        yield from self._local

    def __len__(self) -> int:
        return len(self._base) + len(self._local)

    @overload
    def __getitem__(self, index: int) -> _TraversalRef: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_TraversalRef]: ...

    def __getitem__(
        self, index: int | slice
    ) -> _TraversalRef | Sequence[_TraversalRef]:
        if isinstance(index, slice):
            return tuple(self)[index]
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        base_length = len(self._base)
        if index < base_length:
            return self._base[index]
        return self._local[index - base_length]


class _StructuralAdjacencyMapping(Mapping[str, _TraversalSequence]):
    """Read-only adjacency view that never exposes mutable base lists."""

    __slots__ = ("_base", "_local")

    def __init__(
        self,
        base: Mapping[str, List[_TraversalRef]],
        local: Dict[str, List[_TraversalRef]],
    ) -> None:
        self._base = base
        self._local = local

    def __getitem__(self, key: str) -> _TraversalSequence:
        if key not in self._base and key not in self._local:
            raise KeyError(key)
        return _TraversalSequence(
            self._base.get(key, ()),
            self._local.get(key, ()),
        )

    def __iter__(self) -> Iterator[str]:
        yield from self._base
        for key in self._local:
            if key not in self._base:
                yield key

    def __len__(self) -> int:
        return len(self._base) + sum(
            1 for key in self._local if key not in self._base
        )


class EndpointRoadGraph(RoadGraph):
    """Request-local endpoint additions over an immutable shared base graph."""

    def __init__(self, base: RoadGraph) -> None:
        self.base_graph = base
        self._local_nodes: Dict[str, Node] = {}
        self._local_edges: Dict[str, Edge] = {}
        self._local_adjacency: Dict[str, List[_TraversalRef]] = {}
        self._local_predecessors: Dict[str, List[_TraversalRef]] = {}
        self._local_structure_epoch = 0
        self._frozen = False
        self._route_endpoint_node_ids: Tuple[str, str] | None = None
        self.nodes = _StructuralOverlayMapping(  # type: ignore[assignment]
            base.nodes, self._local_nodes
        )
        self.edges = _StructuralOverlayMapping(  # type: ignore[assignment]
            base.edges, self._local_edges
        )
        self.adjacency = _StructuralAdjacencyMapping(  # type: ignore[assignment]
            base.adjacency, self._local_adjacency
        )
        self._reverse_edge_views: Dict[str, _ReverseEdgeView] = {}
        self._nearest_spatial_index = None
        self._nearest_edge_projection_index = None
        self.artifact_metadata = base.artifact_metadata

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("endpoint graph is frozen")

    def _advance_heuristic_epoch(self) -> None:
        self._local_structure_epoch += 1
        self._reverse_edge_views.clear()

    def _heuristic_cache_stamp(self) -> Tuple[int, int, int]:
        base_structure, node_epoch, edge_epoch = (
            self.base_graph._heuristic_cache_stamp()
        )
        return (
            base_structure + self._local_structure_epoch,
            node_epoch,
            edge_epoch,
        )

    def add_node(self, node: Node) -> None:
        self._ensure_mutable()
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self._local_nodes[node.id] = node
        self._local_adjacency.setdefault(node.id, [])
        self._advance_heuristic_epoch()

    def add_edge(self, edge: Edge) -> None:
        self._ensure_mutable()
        if edge.start_node_id not in self.nodes:
            raise ValueError(f"Unknown start node: {edge.start_node_id}")
        if edge.end_node_id not in self.nodes:
            raise ValueError(f"Unknown end node: {edge.end_node_id}")
        if edge.id in self.edges:
            raise ValueError(f"Duplicate edge id: {edge.id}")
        self._local_edges[edge.id] = edge
        self._local_adjacency.setdefault(edge.start_node_id, []).append(
            (edge.id, False)
        )
        if not edge.one_way:
            self._local_adjacency.setdefault(edge.end_node_id, []).append(
                (edge.id, True)
            )
        target_id = edge.end_node_id
        self._local_predecessors.setdefault(target_id, []).append(
            (edge.id, False)
        )
        if not edge.one_way:
            self._local_predecessors.setdefault(edge.start_node_id, []).append(
                (edge.id, True)
            )
        self._advance_heuristic_epoch()
    def iter_local_edges(self, node_id: str) -> Iterator[_TraversalRef]:
        """Yield request-local outgoing refs without exposing mutable lists."""
        return iter(self._local_adjacency.get(node_id, ()))

    def iter_local_predecessors(
        self, node_id: str
    ) -> Iterator[_TraversalRef]:
        """Yield request-local incoming refs without exposing mutable lists."""
        return iter(self._local_predecessors.get(node_id, ()))

        self._advance_heuristic_epoch()

    def freeze(self) -> None:
        """Reject subsequent endpoint collection mutation."""
        self._frozen = True


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

def _parse_osmnx_length_km(raw_length: Any) -> Optional[float]:
    """Return a positive finite OSMnx edge length converted from metres."""
    if isinstance(raw_length, bool):
        return None
    try:
        length_m = float(raw_length)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(length_m) or length_m <= 0.0:
        return None
    return length_m / 1000.0
def _stable_osm_sort_key(value: Any) -> tuple[int, object, str]:
    """Sort OSM identifiers deterministically without assuming one ID type."""
    if isinstance(value, bool):
        return (1, str(value), "")
    try:
        return (0, int(value), "")
    except (TypeError, ValueError, OverflowError):
        return (1, type(value).__name__, str(value))


def _osmnx_node_data(G: Any) -> Mapping[Any, Any]:
    nodes = getattr(G, "nodes")
    if hasattr(nodes, "__getitem__"):
        return nodes
    return {node_id: data for node_id, data in nodes(data=True)}


def _iter_osmnx_base_nodes(G: Any) -> Iterator[Node]:
    nodes = getattr(G, "nodes")
    if callable(nodes) and not hasattr(nodes, "__getitem__"):
        rows = sorted(nodes(data=True), key=lambda row: _stable_osm_sort_key(row[0]))
        for node_id, data in rows:
            yield Node(
                id=str(node_id),
                lat=float(data.get("y")),
                lon=float(data.get("x")),
            )
        return

    for node_id in sorted(nodes, key=_stable_osm_sort_key):
        data = nodes[node_id]
        yield Node(
            id=str(node_id),
            lat=float(data.get("y")),
            lon=float(data.get("x")),
        )


def _iter_osmnx_edge_triples(
    G: Any,
) -> Iterator[tuple[Any, Any, Any, Mapping[str, Any]]]:
    successors = getattr(G, "succ", None)
    if successors is not None and hasattr(successors, "__getitem__"):
        for start_id in sorted(successors, key=_stable_osm_sort_key):
            neighbours = successors[start_id]
            for end_id in sorted(neighbours, key=_stable_osm_sort_key):
                keyed_edges = neighbours[end_id]
                for key in sorted(keyed_edges, key=_stable_osm_sort_key):
                    yield start_id, end_id, key, keyed_edges[key]
        return

    rows = list(G.edges(keys=True, data=True))
    rows.sort(
        key=lambda row: (
            _stable_osm_sort_key(row[0]),
            _stable_osm_sort_key(row[1]),
            _stable_osm_sort_key(row[2]),
        )
    )
    yield from rows


def _convert_osmnx_edge(
    G: Any,
    node_data: Mapping[Any, Any],
    u: Any,
    v: Any,
    key: Any,
    data: Mapping[str, Any],
    scenic_scores: Mapping[str, float],
) -> tuple[tuple[Node, ...], tuple[Edge, ...]]:
    edge_id = f"{u}-{v}-{key}"
    total_length_km = _parse_osmnx_length_km(data.get("length"))
    scenic_score = float(scenic_scores.get(str(data.get("osmid", edge_id)), 5.0))
    road_name = data.get("name")
    if road_name is not None and not isinstance(road_name, str):
        road_name = str(road_name)
    road_type = _normalize_road_type(data.get("highway", "secondary"))
    speed_kmh = _parse_speed_limit_kmh(data.get("maxspeed"), road_type)
    one_way = _parse_one_way(data.get("oneway"), default=True)
    if not one_way and G.has_edge(v, u):
        one_way = True

    start_id = str(u)
    end_id = str(v)
    start_data = node_data[u]
    end_data = node_data[v]
    start_node = Node(
        id=start_id,
        lat=float(start_data.get("y")),
        lon=float(start_data.get("x")),
    )
    end_node = Node(
        id=end_id,
        lat=float(end_data.get("y")),
        lon=float(end_data.get("x")),
    )

    geometry = data.get("geometry")
    geometry_coords: list[tuple[float, float]] = []
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

    interior_nodes = tuple(
        Node(
            id=f"{u}-{v}-{key}-coord-{coordinate_index}",
            lat=lat,
            lon=lon,
        )
        for coordinate_index, (lon, lat) in enumerate(interior_coords, start=1)
    )
    coordinate_nodes = (start_node, *interior_nodes, end_node)
    node_ids = [node.id for node in coordinate_nodes]
    chord_distances_km = [
        _haversine_km(
            segment_start.lat,
            segment_start.lon,
            segment_end.lat,
            segment_end.lon,
        )
        for segment_start, segment_end in zip(
            coordinate_nodes, coordinate_nodes[1:]
        )
    ]
    chord_total_km = math.fsum(chord_distances_km)
    if total_length_km is not None:
        if math.isfinite(chord_total_km) and chord_total_km > 0.0:
            segment_distances_km = [
                total_length_km * chord / chord_total_km
                for chord in chord_distances_km
            ]
        else:
            segment_distances_km = [
                total_length_km / len(chord_distances_km)
                for _ in chord_distances_km
            ]
    else:
        segment_distances_km = chord_distances_km

    edges = tuple(
        Edge(
            id=f"{u}-{v}-{key}-segment-{coordinate_index}",
            start_node_id=segment_start,
            end_node_id=segment_end,
            distance_km=float(segment_distances_km[coordinate_index]),
            scenic_score=float(max(0.0, min(10.0, scenic_score))),
            road_name=road_name,
            road_type=road_type,
            speed_limit_kmh=speed_kmh,
            one_way=one_way,
        )
        for coordinate_index, (segment_start, segment_end) in enumerate(
            zip(node_ids, node_ids[1:])
        )
    )
    return interior_nodes, edges


def _iter_osmnx_edge_conversions(
    G: Any,
    scenic_scores: Mapping[str, float],
) -> Iterator[tuple[tuple[Node, ...], tuple[Edge, ...]]]:
    node_data = _osmnx_node_data(G)
    for u, v, key, data in _iter_osmnx_edge_triples(G):
        yield _convert_osmnx_edge(G, node_data, u, v, key, data, scenic_scores)


def _iter_osmnx_graph_rows(
    G: Any,
    scenic_scores: Mapping[str, float],
) -> Iterator[tuple[str, Node | Edge]]:
    for node in _iter_osmnx_base_nodes(G):
        yield "node", node
    for interior_nodes, edges in _iter_osmnx_edge_conversions(G, scenic_scores):
        for node in interior_nodes:
            yield "node", node
        for edge in edges:
            yield "edge", edge


def _iter_graph_rows(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
) -> Iterator[tuple[str, Node | Edge]]:
    for node in nodes:
        yield "node", node
    for edge in edges:
        yield "edge", edge


_SQLITE_GRAPH_FORMAT = "scenic-roadgraph-sqlite"
_SQLITE_SCHEMA_VERSION = 1
_SQLITE_BATCH_SIZE = 100_000


def _json_metadata_value(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("SQLite graph metadata must be JSON-serializable") from exc


def _write_sqlite_graph(
    path: Path,
    rows: Iterable[tuple[str, Node | Edge]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> tuple[int, int]:
    """Write graph rows to a sibling temporary SQLite database atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_values = dict(metadata or {})
    existing_format = metadata_values.get("graph_format", _SQLITE_GRAPH_FORMAT)
    existing_version = metadata_values.get("schema_version", _SQLITE_SCHEMA_VERSION)
    if existing_format != _SQLITE_GRAPH_FORMAT:
        raise ValueError(f"Unsupported SQLite graph format: {existing_format!r}")
    if isinstance(existing_version, bool) or existing_version != _SQLITE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SQLite graph schema version: {existing_version!r}"
        )
    metadata_values["graph_format"] = _SQLITE_GRAPH_FORMAT
    metadata_values["schema_version"] = _SQLITE_SCHEMA_VERSION

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    node_count = 0
    edge_count = 0
    node_batch: list[tuple[str, float, float]] = []
    edge_batch: list[tuple[object, ...]] = []

    def flush_nodes(connection: sqlite3.Connection) -> None:
        if node_batch:
            connection.executemany(
                "INSERT INTO nodes(id, lat, lon) VALUES (?, ?, ?)",
                node_batch,
            )
            node_batch.clear()

    def flush_edges(connection: sqlite3.Connection) -> None:
        if edge_batch:
            connection.executemany(
                """
                INSERT INTO edges(
                    id, start_node_id, end_node_id, distance_km,
                    scenic_score, road_name, road_type, speed_limit_kmh, one_way
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                edge_batch,
            )
            edge_batch.clear()

    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.executescript(
                """
                CREATE TABLE metadata(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE nodes(
                    id TEXT PRIMARY KEY,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL
                );
                CREATE TABLE edges(
                    id TEXT PRIMARY KEY,
                    start_node_id TEXT NOT NULL,
                    end_node_id TEXT NOT NULL,
                    distance_km REAL NOT NULL,
                    scenic_score REAL NOT NULL,
                    road_name TEXT,
                    road_type TEXT NOT NULL,
                    speed_limit_kmh REAL,
                    one_way INTEGER NOT NULL CHECK(one_way IN (0, 1))
                );
                """
            )
            for kind, row in rows:
                if kind == "node" and isinstance(row, Node):
                    node_batch.append((str(row.id), float(row.lat), float(row.lon)))
                    node_count += 1
                    if len(node_batch) >= _SQLITE_BATCH_SIZE:
                        flush_nodes(connection)
                elif kind == "edge" and isinstance(row, Edge):
                    edge_batch.append(
                        (
                            str(row.id),
                            str(row.start_node_id),
                            str(row.end_node_id),
                            float(row.distance_km),
                            float(row.scenic_score),
                            None if row.road_name is None else str(row.road_name),
                            str(row.road_type),
                            None
                            if row.speed_limit_kmh is None
                            else float(row.speed_limit_kmh),
                            int(bool(row.one_way)),
                        )
                    )
                    edge_count += 1
                    if len(edge_batch) >= _SQLITE_BATCH_SIZE:
                        flush_edges(connection)
                else:
                    raise ValueError(f"Invalid graph row kind or value: {kind!r}")
            flush_nodes(connection)
            flush_edges(connection)
            metadata_values["node_count"] = node_count
            metadata_values["edge_count"] = edge_count
            metadata_values["counts"] = {"nodes": node_count, "edges": edge_count}
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [
                    (str(key), _json_metadata_value(value))
                    for key, value in sorted(metadata_values.items())
                ],
            )
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise ValueError(f"SQLite integrity check failed: {integrity!r}")
            connection.commit()
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return node_count, edge_count


def _sqlite_metadata(
    connection: sqlite3.Connection,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if check_cancelled is not None:
        check_cancelled()
    try:
        if check_cancelled is not None:
            check_cancelled()
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        if check_cancelled is not None:
            check_cancelled()
    except sqlite3.DatabaseError as exc:
        raise ValueError("SQLite graph metadata table is unavailable") from exc
    metadata: dict[str, Any] = {}
    for row_index, (key, value) in enumerate(rows):
        if (
            check_cancelled is not None
            and row_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
        ):
            check_cancelled()
        if not isinstance(key, str) or key in metadata:
            raise ValueError("SQLite graph metadata contains duplicate keys")
        try:
            metadata[key] = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid SQLite graph metadata value for {key!r}") from exc
    if metadata.get("graph_format") != _SQLITE_GRAPH_FORMAT:
        raise ValueError(
            f"Unsupported SQLite graph format: {metadata.get('graph_format')!r}"
        )
    schema_version = metadata.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _SQLITE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SQLite graph schema version: {schema_version!r}"
        )
    return metadata

def _iter_sqlite_nodes(
    connection: sqlite3.Connection,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> Iterator[_NodeRow]:
    if check_cancelled is not None:
        check_cancelled()
    cursor = connection.execute("SELECT id, lat, lon FROM nodes ORDER BY rowid")
    if check_cancelled is not None:
        check_cancelled()
    row_index = 0
    while True:
        if check_cancelled is not None:
            check_cancelled()
        batch = cursor.fetchmany(_SQLITE_BATCH_SIZE)
        if check_cancelled is not None:
            check_cancelled()
        if not batch:
            return
        for node_id, lat, lon in batch:
            if (
                check_cancelled is not None
                and row_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            row_index += 1
            yield _NodeRow(id=node_id, lat=lat, lon=lon)


def _iter_sqlite_edges(
    connection: sqlite3.Connection,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> Iterator[_EdgeRow]:
    if check_cancelled is not None:
        check_cancelled()
    cursor = connection.execute(
        """
        SELECT id, start_node_id, end_node_id, distance_km,
               scenic_score, road_name, road_type, speed_limit_kmh, one_way
        FROM edges
        ORDER BY rowid
        """
    )
    if check_cancelled is not None:
        check_cancelled()
    row_index = 0
    while True:
        if check_cancelled is not None:
            check_cancelled()
        batch = cursor.fetchmany(_SQLITE_BATCH_SIZE)
        if check_cancelled is not None:
            check_cancelled()
        if not batch:
            return
        for row in batch:
            if (
                check_cancelled is not None
                and row_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            row_index += 1
            yield _EdgeRow(
                id=row[0],
                start_node_id=row[1],
                end_node_id=row[2],
                distance_km=row[3],
                scenic_score=row[4],
                road_name=row[5],
                road_type=row[6],
                speed_limit_kmh=row[7],
                one_way=row[8],
            )


def _load_sqlite_graph(
    path: Path,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> RoadGraph:
    from urllib.parse import quote

    if check_cancelled is not None:
        check_cancelled()
    resolved = Path(path).expanduser().resolve()
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    try:
        if check_cancelled is not None:
            check_cancelled()
        connection = sqlite3.connect(uri, uri=True)
        if check_cancelled is not None:
            check_cancelled()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Invalid SQLite road graph: {path}") from exc
    try:
        if check_cancelled is not None:
            check_cancelled()
        connection.execute("PRAGMA query_only=ON")
        if check_cancelled is not None:
            check_cancelled()
        metadata = _sqlite_metadata(connection, check_cancelled=check_cancelled)
        graph = RoadGraph()
        graph._bulk_load(
            _iter_sqlite_nodes(connection, check_cancelled=check_cancelled),
            _iter_sqlite_edges(connection, check_cancelled=check_cancelled),
            check_cancelled=check_cancelled,
        )
        graph.artifact_metadata = metadata
        if check_cancelled is not None:
            check_cancelled()
        return graph
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Invalid SQLite road graph: {path}") from exc
    finally:
        connection.close()


def _graph_from_osmnx(G: Any, scenic_scores: Dict[str, float]) -> RoadGraph:
    graph = RoadGraph()
    for kind, row in _iter_osmnx_graph_rows(G, scenic_scores):
        if kind == "node":
            graph.add_node(row)  # type: ignore[arg-type]
        else:
            graph.add_edge(row)  # type: ignore[arg-type]
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
