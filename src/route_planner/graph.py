from __future__ import annotations

from dataclasses import dataclass
from array import array
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Callable, ClassVar, Dict, Generic, List, Optional, Tuple, TypeVar, Union, overload
import zlib

from ._edge_projection import (
    EdgeProjection,
    EdgeProjectionIndex,
    _EPI_VERSION,
    _SidecarCorruptError,
    _SidecarCancellation,
    _SidecarMissingError,
    _SidecarStaleError,
    _SidecarTruncatedError,
    _SidecarVersionError,
)

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
    _projection_epoch: ClassVar[int] = 0

    def __setattr__(self, name: str, value: Any) -> None:
        public_field_was_present = not name.startswith("_") and name in self.__dict__
        projection_field_was_present = (
            public_field_was_present and name in ("start_node_id", "end_node_id", "road_type")
        )
        object.__setattr__(self, name, value)
        if public_field_was_present:
            Edge._mutation_epoch += 1
        if projection_field_was_present:
            Edge._projection_epoch += 1

    @property
    def travel_time_minutes(self) -> float:
        speed = max(float(self.speed_limit_kmh), 1.0)
        return (self.distance_km / speed) * 60.0



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
        self._nearest_edge_projection_index: EdgeProjectionIndex | None = None
        self._edge_projection_index_status: str = "missing"
        self._edge_projection_index_invalid_reason: str | None = None
        self._edge_projection_index_stamp: Tuple[int, int, int] | None = None
        self._edge_projection_index_path: str | None = None
        self._edge_projection_index_payload_size: int | None = None
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

    def _edge_projection_stamp(self) -> Tuple[int, int, int]:
        return (
            self._heuristic_structure_epoch,
            Node._coordinate_mutation_epoch,
            Edge._projection_epoch,
        )

    def _invalidate_nearest_spatial_index(self) -> None:
        self._nearest_spatial_index = None
        self._invalidate_nearest_edge_projection_index()

    def _invalidate_nearest_edge_projection_index(self) -> None:
        if (
            self._nearest_edge_projection_index is not None
            or self._edge_projection_index_status in ("saved", "loaded", "memory")
        ):
            self._edge_projection_index_status = "stale"
            self._edge_projection_index_invalid_reason = "graph_mutated"
        epi = getattr(self, "_nearest_edge_projection_index", None)
        if epi is not None:
            if hasattr(epi, "close"):
                try:
                    epi.close()
                except Exception:
                    pass
        self._nearest_edge_projection_index = None
        self._edge_projection_index_stamp = None
    def close(self) -> None:
        """Close backing resources, if any."""
        epi = getattr(self, "_nearest_edge_projection_index", None)
        if epi is not None:
            if hasattr(epi, "close"):
                try:
                    epi.close()
                except Exception:
                    pass
            self._nearest_edge_projection_index = None

    def __enter__(self) -> "RoadGraph":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()



    def _build_nearest_edge_projection_index(
        self,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> EdgeProjectionIndex:
        """Build a typed, immutable nearest-edge projection index."""
        return EdgeProjectionIndex.build(self, check_cancelled=check_cancelled)

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
        """Project a query point onto a finite segment chunk."""
        return EdgeProjectionIndex._project_edge_chunk(
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

    def persist_edge_projection_index(
        self,
        path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> str:
        """Generate or upgrade the edge-projection sidecar for a SQLite graph."""
        sidecar_path = EdgeProjectionIndex.sidecar_path(path)
        self._edge_projection_index_path = str(sidecar_path)
        stamp = self._edge_projection_stamp()
        if (
            self._nearest_edge_projection_index is None
            or self._edge_projection_index_stamp != stamp
        ):
            self._nearest_edge_projection_index = self._build_nearest_edge_projection_index(
                check_cancelled=check_cancelled,
            )
            self._edge_projection_index_stamp = stamp
        EdgeProjectionIndex.write(
            self._nearest_edge_projection_index,
            graph_path=path,
            sidecar_path=sidecar_path,
            check_cancelled=check_cancelled,
        )
        self._edge_projection_index_status = "saved"
        self._edge_projection_index_invalid_reason = None
        self._edge_projection_index_payload_size = sidecar_path.stat().st_size
        return self._edge_projection_index_status

    def _try_load_edge_projection_index(
        self,
        path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
        verify: bool = True,
    ) -> str:
        """Attempt a compatible sidecar load; fall back to lazy rebuild."""
        sidecar_path = EdgeProjectionIndex.sidecar_path(path)
        self._edge_projection_index_path = str(sidecar_path)
        try:
            self._edge_projection_index_payload_size = sidecar_path.stat().st_size
        except FileNotFoundError:
            self._edge_projection_index_status = "missing"
            self._edge_projection_index_invalid_reason = None
            self._nearest_edge_projection_index = None
            self._edge_projection_index_stamp = None
            return self._edge_projection_index_status
        except OSError as exc:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = f"os_error: {exc}"
            self._nearest_edge_projection_index = None
            self._edge_projection_index_stamp = None
            return self._edge_projection_index_status
        try:
            index = EdgeProjectionIndex.load(
                sidecar_path, path, check_cancelled=check_cancelled, verify=verify
            )
            index.attach(self)
            self._nearest_edge_projection_index = index
            self._edge_projection_index_stamp = self._edge_projection_stamp()
            self._edge_projection_index_status = "loaded"
            self._edge_projection_index_invalid_reason = None
            self._edge_projection_index_payload_size = sidecar_path.stat().st_size
        except _SidecarCancellation as exc:
            raise exc.error
        except _SidecarMissingError:
            self._edge_projection_index_status = "missing"
            self._edge_projection_index_invalid_reason = "missing"
        except _SidecarVersionError:
            self._edge_projection_index_status = "version_mismatch"
            self._edge_projection_index_invalid_reason = "version_mismatch"
        except _SidecarStaleError:
            self._edge_projection_index_status = "stale"
            self._edge_projection_index_invalid_reason = "stale"
        except _SidecarTruncatedError:
            self._edge_projection_index_status = "truncated"
            self._edge_projection_index_invalid_reason = "truncated"
        except _SidecarCorruptError:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = "corrupt"
        except ValueError as exc:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = (
                f"attachment_error: {exc}"
            )
        except OSError as exc:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = f"os_error: {exc}"
        if self._edge_projection_index_status != "loaded":
            self._nearest_edge_projection_index = None
            self._edge_projection_index_stamp = None
        return self._edge_projection_index_status

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.adjacency.setdefault(node.id, [])

    @property
    def edge_projection_index_status(self) -> Mapping[str, Any]:
        """Machine-readable sidecar status: state, path, version, algorithm, and health."""
        if (
            self._nearest_edge_projection_index is not None
            and self._edge_projection_index_stamp
            != self._edge_projection_stamp()
        ):
            self._invalidate_nearest_edge_projection_index()
        internal_state = self._edge_projection_index_status
        if internal_state in ("saved", "loaded"):
            state = "loaded"
        elif internal_state == "memory":
            state = "rebuilt"
        elif internal_state == "missing":
            state = "missing"
        else:
            state = "invalid"

        index = self._nearest_edge_projection_index
        path = self._edge_projection_index_path
        payload_size = self._edge_projection_index_payload_size
        if payload_size is None and path is not None:
            try:
                payload_size = Path(path).stat().st_size
            except OSError:
                payload_size = None

        return MappingProxyType(
            {
                "state": state,
                "path": path,
                "format_version": _EPI_VERSION if index is not None else None,
                "algorithm": "bvh-spherical-lb" if index is not None else None,
                "mmap_read_only": (
                    index is not None and index._mmap is not None
                ),
                "edge_count": index.edge_count if index is not None else 0,
                "payload_size_bytes": payload_size,
                "invalid_reason": self._edge_projection_index_invalid_reason,
            }
        )

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

        stamp = self._edge_projection_stamp()
        index = self._nearest_edge_projection_index
        if index is None or self._edge_projection_index_stamp != stamp:
            index = self._build_nearest_edge_projection_index(
                check_cancelled=check_cancelled,
            )
            if check_cancelled is not None:
                check_cancelled()
            self._nearest_edge_projection_index = index
            self._edge_projection_index_stamp = stamp
            self._edge_projection_index_status = "memory"
            self._edge_projection_index_invalid_reason = None
            self._edge_projection_index_payload_size = None

        if index.edge_count == 0:
            raise ValueError("Road graph has no eligible finite segment")

        index.attach(self)

        return index.query(
            self,
            query_lat,
            query_lon,
            excluded_road_types=excluded_road_types,
            check_cancelled=check_cancelled,
        )

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
            self.persist_edge_projection_index(path)
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
        if path.name.endswith(".compact.json"):
            # Deployment admission validates payload/source digests once before
            # startup. Runtime loads validate the signed manifest structure,
            # exact sizes, and mmap bounds without rereading both multi-GB files.
            return CompactRoadGraph.load(
                path,
                check_cancelled=check_cancelled,
                verify=False,
            )
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
        self._edge_projection_index_status = "missing"
        self._edge_projection_index_invalid_reason = None
        self._edge_projection_index_stamp = None
        self._edge_projection_index_path = None
        self._edge_projection_index_payload_size = None
        self.artifact_metadata = base.artifact_metadata

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("endpoint graph is frozen")

    def _advance_heuristic_epoch(self) -> None:
        self._local_structure_epoch += 1
        self._reverse_edge_views.clear()
        self._invalidate_nearest_edge_projection_index()

    def _heuristic_cache_stamp(self) -> Tuple[int, int, int]:
        base_structure, node_epoch, edge_epoch = (
            self.base_graph._heuristic_cache_stamp()
        )
        return (
            base_structure + self._local_structure_epoch,
            node_epoch,
            edge_epoch,
        )

    def _edge_projection_stamp(self) -> Tuple[int, int, int]:
        base_stamp = self.base_graph._edge_projection_stamp()
        return (
            base_stamp[0] + self._local_structure_epoch,
            base_stamp[1],
            base_stamp[2],
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
    return G.nodes


def _iter_osmnx_base_nodes(G: Any) -> Iterator[Node]:
    rows = sorted(G.nodes(data=True), key=lambda row: _stable_osm_sort_key(row[0]))
    for node_id, data in rows:
        yield Node(
            id=str(node_id),
            lat=float(data.get("y")),
            lon=float(data.get("x")),
        )


def _iter_osmnx_edge_triples(
    G: Any,
) -> Iterator[tuple[Any, Any, Any, Mapping[str, Any]]]:
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


def _path_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
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
    initial_path_identity = _path_identity(resolved)
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
        try:
            current_path_identity = _path_identity(resolved)
        except OSError:
            current_path_identity = None
        if current_path_identity == initial_path_identity:
            graph._try_load_edge_projection_index(
                resolved,
                check_cancelled=check_cancelled,
            )
        else:
            graph._edge_projection_index_path = str(
                EdgeProjectionIndex.sidecar_path(resolved)
            )
            graph._edge_projection_index_status = "stale"
            graph._edge_projection_index_invalid_reason = (
                "graph_replaced_during_load"
            )
        try:
            final_path_identity = _path_identity(resolved)
        except OSError:
            final_path_identity = None
        if final_path_identity != initial_path_identity:
            graph._nearest_edge_projection_index = None
            graph._edge_projection_index_stamp = None
            graph._edge_projection_index_status = "stale"
            graph._edge_projection_index_invalid_reason = (
                "graph_replaced_during_load"
            )
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


# ---------------------------------------------------------------------------
# Canonical compact runtime
# ---------------------------------------------------------------------------
#
# ``road_graph.compact.json`` is the explicit runtime entry point for large
# graphs.  It references a little-endian binary section payload
# (``road_graph.compact.bin``) plus the source SQLite audit artifact, and it
# never materializes Node/Edge objects: nodes, canonical edges, and directed
# traversals are mmap-backed numeric/string-table arrays.  A deterministic
# per-report score sidecar (``road_graph.compact.score.<signature>*.bin``)
# stores scenic costs keyed by canonical edge rank.

_COMPACT_FORMAT = "scenic-roadgraph-compact"
_COMPACT_SCHEMA_VERSION = 1
_COMPACT_SCORE_FORMAT = "scenic-roadgraph-compact-score"
_COMPACT_SCORE_SCHEMA_VERSION = 1
_COMPACT_BIN_SUFFIX = ".compact.bin"
_COMPACT_MANIFEST_SUFFIX = ".compact.json"
_COMPACT_ID_CACHE_CAPACITY = 65_536
_COMPACT_SCENIC_BYWAY_ROAD_TYPE = "scenic_byway"
# Mirrors ``src.route_planner.cost.HIGHWAY_ROAD_TYPES``.  Both the converter
# and the runtime derive the mask from the same module (lazy import avoids the
# cost<->graph import cycle); the manifest records the joined set so a stale
# artifact is rejected instead of silently re-masked.
_COMPACT_HIGHWAY_ROAD_TYPES = frozenset(
    {
        "highway",
        "motorway",
        "motorway_link",
        "primary",
        "primary_link",
        "trunk",
        "trunk_link",
    }
)


def _compact_highway_road_types() -> frozenset:
    # ``cost`` imports this module, so the import must stay deferred.
    from .cost import HIGHWAY_ROAD_TYPES

    return frozenset(HIGHWAY_ROAD_TYPES)


_COMPACT_SECTION_SPECS: Tuple[Tuple[str, str], ...] = (
    ("node_lat", "f8"),
    ("node_lon", "f8"),
    ("node_id_strings", "raw"),
    ("node_id_offsets", "i8"),
    ("node_hash_keys", "u8"),
    ("node_hash_values", "i8"),
    ("edge_id_strings", "raw"),
    ("edge_id_offsets", "i8"),
    ("edge_hash_keys", "u8"),
    ("edge_hash_values", "i8"),
    ("edge_start_rank", "i4"),
    ("edge_end_rank", "i4"),
    ("edge_distance_km", "f8"),
    ("edge_scenic_score", "f8"),
    ("edge_speed_limit_kmh", "f8"),
    ("edge_one_way", "u1"),
    ("edge_road_type_codes", "i4"),
    ("road_type_strings", "raw"),
    ("road_type_offsets", "i8"),
    ("edge_road_name_strings", "raw"),
    ("edge_road_name_offsets", "i8"),
    ("forward_indptr", "i8"),
    ("forward_indices", "i4"),
    ("reverse_indptr", "i8"),
    ("reverse_indices", "i4"),
    ("reverse_positions", "i8"),
    ("trav_edge_rank", "i4"),
    ("trav_reverse", "u1"),
    ("trav_distance_km", "f8"),
    ("trav_travel_time_minutes", "f8"),
    ("trav_scenic_score", "f8"),
    ("trav_highway_mask", "u1"),
    ("trav_scenic_byway_mask", "u1"),
)

_COMPACT_SECTION_NAMES = frozenset(name for name, _dtype in _COMPACT_SECTION_SPECS)
_COMPACT_DTYPE_ITEMSIZE = {
    "f8": 8,
    "i8": 8,
    "i4": 4,
    "u8": 8,
    "u1": 1,
    "raw": None,
}


def _compact_hash_slots(count: int) -> int:
    """Return the deterministic open-addressing table size for ``count`` keys."""
    if count <= 0:
        return 8
    size = 1 << (2 * count - 1).bit_length()
    return max(8, size)


def _compact_id_hash(key_bytes: bytes) -> int:
    # crc32 is deterministic across processes and platforms; ``+1`` reserves
    # zero as the empty-slot sentinel.
    return (zlib.crc32(key_bytes) & 0xFFFFFFFF) + 1


def _sha256_file(path: Path, check_cancelled: Callable[[], None] | None = None) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            if check_cancelled is not None:
                check_cancelled()
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if check_cancelled is not None:
        check_cancelled()
    return digest.digest()


def _fsync_parent(path: Path) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_parent(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _iter_sqlite_edge_rank_rows(
    connection: sqlite3.Connection,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> Iterator[tuple[int, ...]]:
    """Stream canonical edge rows with resolved node ranks in rowid order.

    Rank resolution uses the rowid ordering contract of ``_bulk_load``: node
    rank equals ``rowid - 1`` (the writer never deletes rows).  The JOIN
    rejects edges whose endpoints are absent from the node table.
    """
    cursor = connection.execute(
        """
        SELECT
            e.id, e.start_node_id, e.end_node_id,
            e.distance_km, e.scenic_score, e.road_name, e.road_type,
            e.speed_limit_kmh, e.one_way,
            ns.rowid - 1 AS start_rank,
            ne.rowid - 1 AS end_rank,
            e.rowid - 1 AS edge_rank
        FROM edges AS e
        JOIN nodes AS ns ON ns.id = e.start_node_id
        JOIN nodes AS ne ON ne.id = e.end_node_id
        ORDER BY e.rowid
        """
    )
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
            yield row


def _compact_string_tables(
    values: Sequence[str],
) -> tuple[bytearray, array]:
    """Build a deterministic concatenated UTF-8 string table with offsets."""
    data = bytearray()
    offsets = array("q", [0])
    for value in values:
        encoded = str(value).encode("utf-8")
        data.extend(encoded)
        offsets.append(len(data))
    return data, offsets


def write_compact_graph(
    sqlite_path: Path,
    manifest_path: Path,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build the canonical compact runtime artifacts from a SQLite graph.

    The SQLite artifact is the authoritative source; this converter streams
    its rows in deterministic rowid order and never rebuilds Node/Edge
    objects.  All sections are little-endian, digest-signed, and published
    atomically next to the manifest.
    """
    if array("q").itemsize != 8 or array("i").itemsize != 4:
        raise RuntimeError("compact artifact requires 64-bit array support")
    from urllib.parse import quote

    sqlite_path = Path(sqlite_path)
    manifest_path = Path(manifest_path)
    if check_cancelled is not None:
        check_cancelled()
    resolved_source = sqlite_path.expanduser().resolve()
    uri = f"file:{quote(str(resolved_source), safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Invalid SQLite road graph: {sqlite_path}") from exc
    try:
        connection.execute("PRAGMA query_only=ON")
        metadata = _sqlite_metadata(connection, check_cancelled=check_cancelled)
        if check_cancelled is not None:
            check_cancelled()
        node_total = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        node_max_rowid = connection.execute("SELECT MAX(rowid) FROM nodes").fetchone()[0]
        edge_total = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        edge_max_rowid = connection.execute("SELECT MAX(rowid) FROM edges").fetchone()[0]
        if int(node_max_rowid) != int(node_total):
            raise ValueError(
                "Compact conversion requires gapless node rowids "
                f"(count {node_total}, max rowid {node_max_rowid})"
            )
        if int(edge_max_rowid) != int(edge_total):
            raise ValueError(
                "Compact conversion requires gapless edge rowids "
                f"(count {edge_total}, max rowid {edge_max_rowid})"
            )
        if check_cancelled is not None:
            check_cancelled()

        # --- Node sections -------------------------------------------------
        node_lat = array("d")
        node_lon = array("d")
        node_id_data = bytearray()
        node_id_offsets = array("q", [0])
        node_hash_size = _compact_hash_slots(node_total)
        node_hash_keys = array("Q", [0]) * node_hash_size
        node_hash_values = array("q", [-1]) * node_hash_size
        node_cursor = connection.execute(
            "SELECT id, lat, lon FROM nodes ORDER BY rowid"
        )
        node_index = 0
        while True:
            if check_cancelled is not None:
                check_cancelled()
            batch = node_cursor.fetchmany(_SQLITE_BATCH_SIZE)
            if check_cancelled is not None:
                check_cancelled()
            if not batch:
                break
            for node_id, lat, lon in batch:
                if (
                    check_cancelled is not None
                    and node_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
                ):
                    check_cancelled()
                lat_value = float(lat)
                lon_value = float(lon)
                if not (math.isfinite(lat_value) and math.isfinite(lon_value)):
                    raise ValueError(
                        f"Node {node_id!r} has non-finite coordinates"
                    )
                node_lat.append(lat_value)
                node_lon.append(lon_value)
                encoded = str(node_id).encode("utf-8")
                node_id_data.extend(encoded)
                node_id_offsets.append(len(node_id_data))
                hash_value = _compact_id_hash(encoded)
                slot = (hash_value - 1) & (node_hash_size - 1)
                while node_hash_keys[slot] != 0:
                    slot = (slot + 1) & (node_hash_size - 1)
                node_hash_keys[slot] = hash_value
                node_hash_values[slot] = node_index
                node_index += 1
        if node_index != node_total:
            raise ValueError(
                f"Node count mismatch during conversion: {node_index} != {node_total}"
            )
        if check_cancelled is not None:
            check_cancelled()

        # --- Edge traversal counting ---------------------------------------
        forward_counts = array("q", [0]) * (node_total + 1)
        reverse_counts = array("q", [0]) * (node_total + 1)
        road_type_index: Dict[str, int] = {}
        road_type_names: List[str] = []
        edge_id_data = bytearray()
        edge_id_offsets = array("q", [0])
        edge_hash_size = _compact_hash_slots(edge_total)
        edge_hash_keys = array("Q", [0]) * edge_hash_size
        edge_hash_values = array("q", [-1]) * edge_hash_size
        road_name_data = bytearray()
        road_name_offsets = array("q", [0])
        edge_start_rank = array("i")
        edge_end_rank = array("i")
        edge_distance_km = array("d")
        edge_scenic_score = array("d")
        edge_speed_limit_kmh = array("d")
        edge_one_way = array("B")
        edge_road_type_codes = array("i")
        geodesic_valid_all = True
        geodesic_max_speed_all = 1.0
        geodesic_valid_avoid_highways = True
        geodesic_max_speed_avoid_highways = 1.0
        highway_types = _compact_highway_road_types()
        edge_index = 0
        for row in _iter_sqlite_edge_rank_rows(
            connection, check_cancelled=check_cancelled
        ):
            (
                _edge_id,
                _start_id,
                _end_id,
                distance,
                scenic,
                _road_name,
                road_type,
                speed,
                one_way,
                start_rank,
                end_rank,
                edge_rank,
            ) = row
            start_rank = int(start_rank)
            end_rank = int(end_rank)
            edge_rank = int(edge_rank)
            if edge_rank != edge_index:
                raise ValueError(
                    f"Edge rowid order mismatch at rank {edge_rank}"
                )
            distance_value = float(distance)
            scenic_value = float(scenic)
            if not math.isfinite(distance_value) or distance_value < 0.0:
                raise ValueError(f"Edge {_edge_id!r} has invalid distance")
            if not math.isfinite(scenic_value):
                raise ValueError(f"Edge {_edge_id!r} has non-finite scenic score")
            one_way_value = _parse_one_way(one_way, default=True)
            road_type_value = str(road_type)
            speed_value = int(
                _parse_speed_limit_kmh(speed, road_type_value)
            )
            if speed_value <= 0:
                raise ValueError(f"Edge {_edge_id!r} has invalid speed limit")
            edge_start_rank.append(start_rank)
            edge_end_rank.append(end_rank)
            edge_distance_km.append(distance_value)
            edge_scenic_score.append(scenic_value)
            edge_speed_limit_kmh.append(float(speed_value))
            edge_one_way.append(1 if one_way_value else 0)
            type_code = road_type_index.get(road_type_value)
            if type_code is None:
                type_code = len(road_type_names)
                road_type_index[road_type_value] = type_code
                road_type_names.append(road_type_value)
            edge_road_type_codes.append(type_code)
            encoded_id = str(_edge_id).encode("utf-8")
            edge_id_data.extend(encoded_id)
            edge_id_offsets.append(len(edge_id_data))
            hash_value = _compact_id_hash(encoded_id)
            slot = (hash_value - 1) & (edge_hash_size - 1)
            while edge_hash_keys[slot] != 0:
                slot = (slot + 1) & (edge_hash_size - 1)
            edge_hash_keys[slot] = hash_value
            edge_hash_values[slot] = edge_index
            encoded_name = (
                b"" if _road_name is None else str(_road_name).encode("utf-8")
            )
            road_name_data.extend(encoded_name)
            road_name_offsets.append(len(road_name_data))
            forward_counts[start_rank + 1] += 1
            reverse_counts[end_rank + 1] += 1
            if not one_way_value:
                forward_counts[end_rank + 1] += 1
                reverse_counts[start_rank + 1] += 1
            start_lat = node_lat[start_rank]
            start_lon = node_lon[start_rank]
            end_lat = node_lat[end_rank]
            end_lon = node_lon[end_rank]
            endpoint_km = _haversine_km(start_lat, start_lon, end_lat, end_lon)
            highway = road_type_value.lower() in highway_types
            is_valid_geodesic = (
                math.isfinite(distance_value)
                and distance_value >= 0.0
                and math.isfinite(speed_value)
                and math.isfinite(endpoint_km)
                and endpoint_km >= 0.0
                and distance_value >= (endpoint_km - 1e-9)
            )
            if is_valid_geodesic:
                geodesic_max_speed_all = max(geodesic_max_speed_all, float(speed_value))
            else:
                geodesic_valid_all = False

            if not highway:
                if is_valid_geodesic:
                    geodesic_max_speed_avoid_highways = max(
                        geodesic_max_speed_avoid_highways, float(speed_value)
                    )
                else:
                    geodesic_valid_avoid_highways = False
            edge_index += 1
        if edge_index != edge_total:
            raise ValueError(
                f"Edge count mismatch during conversion: {edge_index} != {edge_total}"
            )
        # Note: reverse rows are indexed by the forward traversal target, so
        # ``reverse_counts[target + 1]`` counts one reverse row per forward
        # traversal exactly as ``_build_csr_topology`` does.
        forward_indptr = array("q", [0]) * (node_total + 1)
        for index in range(node_total):
            forward_indptr[index + 1] = (
                forward_indptr[index] + forward_counts[index + 1]
            )
        traversal_total = int(forward_indptr[-1])
        reverse_indptr = array("q", [0]) * (node_total + 1)
        for index in range(node_total):
            reverse_indptr[index + 1] = (
                reverse_indptr[index] + reverse_counts[index + 1]
            )
        reverse_total = int(reverse_indptr[-1])
        if reverse_total != traversal_total:
            raise ValueError(
                "Reverse traversal count mismatch "
                f"({reverse_total} != {traversal_total})"
            )
        forward_cursor = array("q", forward_indptr[:-1])
        forward_indices = array("i", [0]) * traversal_total
        trav_edge_rank = array("i", [0]) * traversal_total
        trav_reverse = array("B", [0]) * traversal_total
        trav_distance_km = array("d", [0.0]) * traversal_total
        trav_travel_time_minutes = array("d", [0.0]) * traversal_total
        trav_scenic_score = array("d", [0.0]) * traversal_total
        trav_highway_mask = array("B", [0]) * traversal_total
        trav_scenic_byway_mask = array("B", [0]) * traversal_total
        highway_types = _compact_highway_road_types()
        edge_index = 0
        for row in _iter_sqlite_edge_rank_rows(
            connection, check_cancelled=check_cancelled
        ):
            (
                _edge_id,
                _start_id,
                _end_id,
                distance,
                scenic,
                _road_name,
                road_type,
                speed,
                one_way,
                start_rank,
                end_rank,
                edge_rank,
            ) = row
            start_rank = int(start_rank)
            end_rank = int(end_rank)
            edge_rank = int(edge_rank)
            if edge_rank != edge_index:
                raise ValueError(
                    f"Edge rowid order mismatch at rank {edge_rank}"
                )
            one_way_value = _parse_one_way(one_way, default=True)
            road_type_value = str(road_type)
            distance_value = float(distance)
            speed_value = int(_parse_speed_limit_kmh(speed, road_type_value))
            duration_value = (distance_value / max(speed_value, 1.0)) * 60.0
            if not math.isfinite(duration_value) or duration_value < 0.0:
                raise ValueError(f"Edge {_edge_id!r} has invalid duration")
            scenic_value = float(scenic)
            road_type_lower = road_type_value.lower()
            highway = road_type_lower in highway_types
            byway = road_type_lower == _COMPACT_SCENIC_BYWAY_ROAD_TYPE

            def emit_traversal(
                source_rank: int,
                target_rank: int,
                reverse: bool,
            ) -> None:
                position = int(forward_cursor[source_rank])
                forward_cursor[source_rank] += 1
                forward_indices[position] = target_rank
                trav_edge_rank[position] = edge_rank
                trav_reverse[position] = 1 if reverse else 0
                trav_distance_km[position] = distance_value
                trav_travel_time_minutes[position] = duration_value
                trav_scenic_score[position] = scenic_value
                trav_highway_mask[position] = 1 if highway else 0
                trav_scenic_byway_mask[position] = 1 if byway else 0

            emit_traversal(start_rank, end_rank, False)
            if not one_way_value:
                emit_traversal(end_rank, start_rank, True)
            edge_index += 1
        if check_cancelled is not None:
            check_cancelled()
        for rank in range(node_total):
            if forward_cursor[rank] != forward_indptr[rank + 1]:
                raise ValueError(
                    f"Forward row {rank} was not fully populated during conversion"
                )
        # Reverse rows are filled in forward-position order exactly like
        # ``_build_csr_topology``: one reverse row per forward traversal,
        # indexed by the forward target, storing the predecessor node and the
        # forward position.
        reverse_indices = array("i", [0]) * reverse_total
        reverse_positions = array("q", [0]) * reverse_total
        reverse_cursor = array("q", reverse_indptr[:-1])
        for source_rank in range(node_total):
            row_start = int(forward_indptr[source_rank])
            row_end = int(forward_indptr[source_rank + 1])
            for position in range(row_start, row_end):
                if (
                    check_cancelled is not None
                    and position & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
                ):
                    check_cancelled()
                target_rank = int(forward_indices[position])
                reverse_position = int(reverse_cursor[target_rank])
                reverse_cursor[target_rank] += 1
                reverse_indices[reverse_position] = source_rank
                reverse_positions[reverse_position] = position
        for rank in range(node_total):
            if reverse_cursor[rank] != reverse_indptr[rank + 1]:
                raise ValueError(
                    f"Reverse row {rank} was not fully populated during conversion"
                )
        if check_cancelled is not None:
            check_cancelled()

        # --- Section payloads ----------------------------------------------
        road_type_data, road_type_offsets = _compact_string_tables(road_type_names)

        def array_bytes(values: array, dtype: str) -> bytes:
            if len(values) == 0:
                return b""
            return np.frombuffer(values, dtype="<" + dtype, count=len(values)).tobytes()

        section_payloads: Dict[str, bytes] = {
            "node_lat": array_bytes(node_lat, "f8"),
            "node_lon": array_bytes(node_lon, "f8"),
            "node_id_strings": bytes(node_id_data),
            "node_id_offsets": array_bytes(node_id_offsets, "i8"),
            "node_hash_keys": array_bytes(node_hash_keys, "u8"),
            "node_hash_values": array_bytes(node_hash_values, "i8"),
            "edge_id_strings": bytes(edge_id_data),
            "edge_id_offsets": array_bytes(edge_id_offsets, "i8"),
            "edge_hash_keys": array_bytes(edge_hash_keys, "u8"),
            "edge_hash_values": array_bytes(edge_hash_values, "i8"),
            "edge_start_rank": array_bytes(edge_start_rank, "i4"),
            "edge_end_rank": array_bytes(edge_end_rank, "i4"),
            "edge_distance_km": array_bytes(edge_distance_km, "f8"),
            "edge_scenic_score": array_bytes(edge_scenic_score, "f8"),
            "edge_speed_limit_kmh": array_bytes(edge_speed_limit_kmh, "f8"),
            "edge_one_way": array_bytes(edge_one_way, "u1"),
            "edge_road_type_codes": array_bytes(edge_road_type_codes, "i4"),
            "road_type_strings": bytes(road_type_data),
            "road_type_offsets": array_bytes(road_type_offsets, "i8"),
            "edge_road_name_strings": bytes(road_name_data),
            "edge_road_name_offsets": array_bytes(road_name_offsets, "i8"),
            "forward_indptr": array_bytes(forward_indptr, "i8"),
            "forward_indices": array_bytes(forward_indices, "i4"),
            "reverse_indptr": array_bytes(reverse_indptr, "i8"),
            "reverse_indices": array_bytes(reverse_indices, "i4"),
            "reverse_positions": array_bytes(reverse_positions, "i8"),
            "trav_edge_rank": array_bytes(trav_edge_rank, "i4"),
            "trav_reverse": array_bytes(trav_reverse, "u1"),
            "trav_distance_km": array_bytes(trav_distance_km, "f8"),
            "trav_travel_time_minutes": array_bytes(
                trav_travel_time_minutes, "f8"
            ),
            "trav_scenic_score": array_bytes(trav_scenic_score, "f8"),
            "trav_highway_mask": array_bytes(trav_highway_mask, "u1"),
            "trav_scenic_byway_mask": array_bytes(
                trav_scenic_byway_mask, "u1"
            ),
        }
        for name, _dtype in _COMPACT_SECTION_SPECS:
            if name not in section_payloads:
                raise AssertionError(f"missing compact section {name}")

        if check_cancelled is not None:
            check_cancelled()
        source_sha256 = _sha256_file(
            resolved_source, check_cancelled=check_cancelled
        )
        if check_cancelled is not None:
            check_cancelled()

        # --- Binary payload with per-section digests ------------------------
        bin_path = manifest_path.with_name(
            manifest_path.name[: -len(_COMPACT_MANIFEST_SUFFIX)] + _COMPACT_BIN_SUFFIX
        )
        section_descriptors: Dict[str, Dict[str, Any]] = {}
        bin_offset = 0
        bin_digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            prefix=f".{bin_path.name}.",
            suffix=".tmp",
            dir=bin_path.parent,
            delete=False,
        ) as stream:
            temporary_bin = Path(stream.name)
            try:
                for name, dtype in _COMPACT_SECTION_SPECS:
                    if check_cancelled is not None:
                        check_cancelled()
                    payload = section_payloads[name]
                    if dtype == "raw":
                        count = len(payload)
                    else:
                        itemsize = _COMPACT_DTYPE_ITEMSIZE[dtype]
                        if len(payload) % itemsize != 0:
                            raise AssertionError(
                                f"compact section {name} has misaligned payload"
                            )
                        count = len(payload) // itemsize
                    section_digest = hashlib.sha256(payload).hexdigest()
                    section_descriptors[name] = {
                        "offset": bin_offset,
                        "length": len(payload),
                        "count": count,
                        "dtype": dtype,
                        "sha256": section_digest,
                    }
                    stream.write(payload)
                    bin_digest.update(payload)
                    bin_offset += len(payload)
                stream.flush()
                os.fsync(stream.fileno())
            except Exception:
                temporary_bin.unlink(missing_ok=True)
                raise
        os.replace(temporary_bin, bin_path)
        _fsync_parent(bin_path)
        if check_cancelled is not None:
            check_cancelled()
        projection_path = EdgeProjectionIndex.sidecar_path(resolved_source)
        projection_info: dict[str, Any] = {
            "path": projection_path.name,
        }
        if projection_path.is_file():
            projection_info["sha256"] = _sha256_file(
                projection_path, check_cancelled=check_cancelled
            ).hex()
            projection_info["size_bytes"] = int(projection_path.stat().st_size)
        manifest = {
            "format": _COMPACT_FORMAT,
            "schema_version": _COMPACT_SCHEMA_VERSION,
            "source": {
                "path": resolved_source.name,
                "sha256": source_sha256.hex(),
                "size_bytes": int(resolved_source.stat().st_size),
                "schema_version": int(metadata["schema_version"]),
                "node_count": int(node_total),
                "edge_count": int(edge_total),
            },
            "graph": {
                "node_count": int(node_total),
                "edge_count": int(edge_total),
                "traversal_count": int(traversal_total),
                "geodesic_bound_speed": {
                    "all": float(geodesic_max_speed_all) if geodesic_valid_all else None,
                    "avoid_highways": (
                        float(geodesic_max_speed_avoid_highways)
                        if geodesic_valid_avoid_highways
                        else None
                    ),
                },
            },
            "geodesic_bound_speed": {
                "all": float(geodesic_max_speed_all) if geodesic_valid_all else None,
                "avoid_highways": (
                    float(geodesic_max_speed_avoid_highways)
                    if geodesic_valid_avoid_highways
                    else None
                ),
            },
            "bin_path": bin_path.name,
            "bin_sha256": bin_digest.hexdigest(),
            "bin_size_bytes": int(bin_offset),
            "sections": section_descriptors,
            "projection_index": projection_info,
            "scenic_byway_road_type": _COMPACT_SCENIC_BYWAY_ROAD_TYPE,
            "highway_road_types": ",".join(sorted(highway_types)),
            "builder": {
                "name": "write_compact_graph",
                "row_order": "sqlite-rowid",
            },
            "score": {"present": False, "path": None},
        }
        manifest_payload = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        _write_atomic_bytes(manifest_path, manifest_payload)
        return {
            "manifest_path": str(manifest_path),
            "bin_path": str(bin_path),
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "bin_sha256": manifest["bin_sha256"],
            "bin_size_bytes": int(bin_offset),
            "node_count": int(node_total),
            "edge_count": int(edge_total),
            "traversal_count": int(traversal_total),
            "source_sha256": manifest["source"]["sha256"],
            "source_size_bytes": int(manifest["source"]["size_bytes"]),
        }
    finally:
        connection.close()


@dataclass(frozen=True)
class _CompactCSRArrays:
    """Persisted CSR topology plus lazy rank resolvers for one graph epoch."""

    node_count: int
    node_ids: Sequence[str]
    node_index: Mapping[str, int]
    indptr: np.ndarray
    indices: np.ndarray
    reverse_indptr: np.ndarray
    reverse_indices: np.ndarray
    reverse_positions: np.ndarray
    edge_refs: Sequence[Tuple[str, bool]]
    distance_km: np.ndarray
    travel_time_minutes: np.ndarray
    _scenic_score: Union[np.ndarray, Callable[[], np.ndarray]]
    highway_mask: np.ndarray
    scenic_byway_mask: np.ndarray

    @property
    def scenic_score(self) -> np.ndarray:
        if callable(self._scenic_score):
            value = self._scenic_score()
            object.__setattr__(self, "_scenic_score", value)
            return value
        return self._scenic_score


@dataclass(frozen=True)
class _ScoreSidecar:
    path: Path
    manifest: Mapping[str, Any]
    values: np.ndarray
    _mmap: Optional[mmap.mmap] = None
    _file: Optional[Any] = None

    def close(self) -> None:
        """Close backing memory-map and file handle if held."""
        object.__setattr__(self, "values", np.empty(0, dtype=np.float64))
        mm = getattr(self, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass
            object.__setattr__(self, "_mmap", None)
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            object.__setattr__(self, "_file", None)

    def __enter__(self) -> "_ScoreSidecar":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

class _CompactRankStringSequence(Sequence[str]):
    """Lazy rank->id decoding from a compact string table."""

    __slots__ = ("_graph", "_kind", "_cache", "_capacity")

    def __init__(self, graph: "CompactRoadGraph", kind: str) -> None:
        self._graph = graph
        self._kind = kind
        self._cache: "OrderedDict[int, str]" = OrderedDict()
        self._capacity = _COMPACT_ID_CACHE_CAPACITY

    def __len__(self) -> int:
        graph = self._graph
        return graph.node_count if self._kind == "node" else graph.edge_count

    def __getitem__(self, index: int) -> str:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        graph = self._graph
        value = (
            graph._node_id_at_rank(index)
            if self._kind == "node"
            else graph._edge_id_at_rank(index)
        )
        if len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)
        self._cache[index] = value
        return value

    def __iter__(self) -> Iterator[str]:
        for index in range(len(self)):
            yield self[index]


class _CompactTraversalSequence(Sequence[Tuple[str, bool]]):
    """Lazy traversal-position -> (edge id, reverse) decoding."""

    __slots__ = ("_graph", "_cache", "_capacity")

    def __init__(self, graph: "CompactRoadGraph") -> None:
        self._graph = graph
        self._cache: "OrderedDict[int, Tuple[str, bool]]" = OrderedDict()
        self._capacity = _COMPACT_ID_CACHE_CAPACITY

    def __len__(self) -> int:
        return self._graph.traversal_count

    def __getitem__(self, position: int) -> Tuple[str, bool]:
        cached = self._cache.get(position)
        if cached is not None:
            return cached
        value = self._graph._traversal_ref(position)
        if len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)
        self._cache[position] = value
        return value

    def __iter__(self) -> Iterator[Tuple[str, bool]]:
        for position in range(len(self)):
            yield self[position]


class _CompactHashLookup(Mapping[str, int]):
    """Read-only id->rank lookup over a persisted open-addressing table."""

    __slots__ = ("_graph", "_kind")

    def __init__(self, graph: "CompactRoadGraph", kind: str) -> None:
        self._graph = graph
        self._kind = kind

    def __len__(self) -> int:
        graph = self._graph
        return graph.node_count if self._kind == "node" else graph.edge_count

    def __getitem__(self, key: str) -> int:
        rank = self._graph._rank_for_id(str(key), self._kind)
        if rank is None:
            raise KeyError(key)
        return rank

    def __contains__(self, key: object) -> bool:
        return self._graph._rank_for_id(str(key), self._kind) is not None

    def __iter__(self) -> Iterator[str]:
        graph = self._graph
        resolver = (
            graph._node_id_at_rank if self._kind == "node" else graph._edge_id_at_rank
        )
        count = len(self)
        for rank in range(count):
            yield resolver(rank)

    def get(self, key: str, default: Any = None) -> Any:
        rank = self._graph._rank_for_id(str(key), self._kind)
        return rank if rank is not None else default


class _CompactEdgeKeySequence(Sequence[str]):
    """Lazy canonical edge keys used by the projection sidecar attachment."""

    __slots__ = ("_graph",)

    def __init__(self, graph: "CompactRoadGraph") -> None:
        self._graph = graph

    def __len__(self) -> int:
        return self._graph.edge_count

    def __getitem__(self, rank: int) -> str:
        return self._graph._edge_id_at_rank(rank)


class _RankProjectionIndex(EdgeProjectionIndex):
    """Projection sidecar that binds to compact graphs without key tuples."""

    __slots__ = ()

    def attach(self, graph: Any) -> None:
        owner = id(graph)
        if self._canonical_keys is not None:
            if self._canonical_keys_owner != owner:
                raise ValueError("Edge projection index is attached to another graph")
            return
        self._canonical_keys = _CompactEdgeKeySequence(graph)
        self._canonical_keys_owner = owner


class _CompactNodeMapping(Mapping[str, Node]):
    __slots__ = ("_graph",)

    def __init__(self, graph: "CompactRoadGraph") -> None:
        self._graph = graph

    def __len__(self) -> int:
        return self._graph.node_count

    def __contains__(self, key: object) -> bool:
        return self._graph._rank_for_id(str(key), "node") is not None

    def __getitem__(self, key: str) -> Node:
        rank = self._graph._rank_for_id(str(key), "node")
        if rank is None:
            raise KeyError(key)
        return self._graph._node_at_rank(rank)

    def __iter__(self) -> Iterator[str]:
        graph = self._graph
        for rank in range(graph.node_count):
            yield graph._node_id_at_rank(rank)

    def get(self, key: str, default: Any = None) -> Any:
        rank = self._graph._rank_for_id(str(key), "node")
        return self._graph._node_at_rank(rank) if rank is not None else default


class _CompactEdgeMapping(Mapping[str, Edge]):
    __slots__ = ("_graph",)

    def __init__(self, graph: "CompactRoadGraph") -> None:
        self._graph = graph

    def __len__(self) -> int:
        return self._graph.edge_count

    def __contains__(self, key: object) -> bool:
        return self._graph._rank_for_id(str(key), "edge") is not None

    def __getitem__(self, key: str) -> Edge:
        rank = self._graph._rank_for_id(str(key), "edge")
        if rank is None:
            raise KeyError(key)
        return self._graph._edge_at_rank(rank)

    def __iter__(self) -> Iterator[str]:
        graph = self._graph
        for rank in range(graph.edge_count):
            yield graph._edge_id_at_rank(rank)

    def get(self, key: str, default: Any = None) -> Any:
        rank = self._graph._rank_for_id(str(key), "edge")
        return self._graph._edge_at_rank(rank) if rank is not None else default


class _CompactAdjacencyMapping(Mapping[str, List[Tuple[str, bool]]]):
    """Lazy per-node traversal references in canonical edge order."""

    __slots__ = ("_graph",)

    def __init__(self, graph: "CompactRoadGraph") -> None:
        self._graph = graph

    def __len__(self) -> int:
        return self._graph.node_count

    def __contains__(self, key: object) -> bool:
        return self._graph._rank_for_id(str(key), "node") is not None

    def _row(self, key: str) -> List[Tuple[str, bool]]:
        graph = self._graph
        rank = graph._rank_for_id(str(key), "node")
        if rank is None:
            raise KeyError(key)
        row_start = int(graph._sections["forward_indptr"][rank])
        row_end = int(graph._sections["forward_indptr"][rank + 1])
        refs: List[Tuple[str, bool]] = []
        for position in range(row_start, row_end):
            refs.append(graph._traversal_ref(position))
        return refs

    def __getitem__(self, key: str) -> List[Tuple[str, bool]]:
        return self._row(key)

    def __iter__(self) -> Iterator[str]:
        graph = self._graph
        for rank in range(graph.node_count):
            yield graph._node_id_at_rank(rank)

    def get(self, key: str, default: Any = None) -> Any:
        rank = self._graph._rank_for_id(str(key), "node")
        if rank is None:
            return default
        return self._row(key)


class CompactRoadGraph(RoadGraph):
    """Immutable mmap-backed road graph loaded from a compact manifest.

    Nodes, canonical edges, and directed traversals are stored as
    little-endian numeric arrays and string tables.  Node/Edge objects are
    materialized lazily only for path/endpoint records; no eager object graph
    is ever constructed.  The runtime selects this backend only when an
    explicit ``*.compact.json`` manifest path is supplied.
    """

    def __init__(
        self,
        manifest: Mapping[str, Any],
        sections: Mapping[str, np.ndarray | memoryview],
        bin_mmap: mmap.mmap,
        bin_file: Any,
        manifest_path: Path,
        source_path: Path,
    ) -> None:
        self._heuristic_structure_epoch = 0
        self._reverse_edge_views: Dict[str, _ReverseEdgeView] = {}
        self._nearest_spatial_index: Optional[Tuple[object, ...]] = None
        self._nearest_edge_projection_index: Optional[EdgeProjectionIndex] = None
        self._edge_projection_index_status: str = "missing"
        self._edge_projection_index_invalid_reason: str | None = None
        self._edge_projection_index_stamp: Tuple[int, int, int] | None = None
        self._edge_projection_index_path: str | None = None
        self._edge_projection_index_payload_size: int | None = None
        self.artifact_metadata: dict[str, Any] = dict(manifest)
        self._manifest = manifest
        self._manifest_path = Path(manifest_path)
        self._source_path = Path(source_path)
        self._bin_mmap = bin_mmap
        self._bin_file = bin_file
        self._sections = sections
        self._geodesic_bound_speed = manifest.get("geodesic_bound_speed") or manifest.get("graph", {}).get("geodesic_bound_speed")
        self._active_score_sidecar: Optional[_ScoreSidecar] = None
        self._csr_arrays_cache: Optional[_CompactCSRArrays] = None
        self._csr_arrays_cache_stamp: object = None
        self._edge_highway_mask_cache: Optional[np.ndarray] = None
        self.nodes = _CompactNodeMapping(self)  # type: ignore[assignment]
        self.edges = _CompactEdgeMapping(self)  # type: ignore[assignment]
        self.adjacency = _CompactAdjacencyMapping(self)  # type: ignore[assignment]

    # -- counts ------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return int(self._manifest["graph"]["node_count"])

    @property
    def edge_count(self) -> int:
        return int(self._manifest["graph"]["edge_count"])

    @property
    def traversal_count(self) -> int:
        return int(self._manifest["graph"]["traversal_count"])

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def source_path(self) -> Path:
        return self._source_path

    @property
    def source_sha256(self) -> str:
        return str(self._manifest["source"]["sha256"])

    # -- sections ----------------------------------------------------------

    def _section(self, name: str) -> np.ndarray | memoryview:
        return self._sections[name]

    def _string_at(
        self,
        strings: np.ndarray | memoryview,
        offsets: np.ndarray,
        index: int,
    ) -> str:
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        return bytes(strings[start:stop]).decode("utf-8")

    def _node_id_at_rank(self, rank: int) -> str:
        return self._string_at(
            self._sections["node_id_strings"],
            self._sections["node_id_offsets"],
            rank,
        )

    def _edge_id_at_rank(self, rank: int) -> str:
        return self._string_at(
            self._sections["edge_id_strings"],
            self._sections["edge_id_offsets"],
            rank,
        )

    def _road_name_at_rank(self, rank: int) -> Optional[str]:
        start = int(self._sections["edge_road_name_offsets"][rank])
        stop = int(self._sections["edge_road_name_offsets"][rank + 1])
        if start == stop:
            return None
        return bytes(self._sections["edge_road_name_strings"][start:stop]).decode(
            "utf-8"
        )

    def _road_type_at_rank(self, rank: int) -> str:
        code = int(self._sections["edge_road_type_codes"][rank])
        return self._string_at(
            self._sections["road_type_strings"],
            self._sections["road_type_offsets"],
            code,
        )

    def _rank_for_id(self, key: str, kind: str) -> Optional[int]:
        table_keys = self._sections[f"{kind}_hash_keys"]
        table_values = self._sections[f"{kind}_hash_values"]
        key_bytes = key.encode("utf-8")
        hash_value = _compact_id_hash(key_bytes)
        mask = len(table_keys) - 1
        slot = (hash_value - 1) & mask
        resolver = (
            self._node_id_at_rank if kind == "node" else self._edge_id_at_rank
        )
        while True:
            stored = int(table_keys[slot])
            if stored == 0:
                return None
            if stored == hash_value:
                rank = int(table_values[slot])
                if rank >= 0 and resolver(rank) == key:
                    return rank
            slot = (slot + 1) & mask

    # -- lazy object materialization ----------------------------------------

    def _node_at_rank(self, rank: int) -> Node:
        return Node(
            id=self._node_id_at_rank(rank),
            lat=float(self._sections["node_lat"][rank]),
            lon=float(self._sections["node_lon"][rank]),
        )

    def _scenic_at_rank(self, rank: int) -> float:
        sidecar = self._active_score_sidecar
        if sidecar is not None:
            return float(sidecar.values[rank])
        return float(self._sections["edge_scenic_score"][rank])

    def _edge_at_rank(self, rank: int) -> Edge:
        return Edge(
            id=self._edge_id_at_rank(rank),
            start_node_id=self._node_id_at_rank(
                int(self._sections["edge_start_rank"][rank])
            ),
            end_node_id=self._node_id_at_rank(
                int(self._sections["edge_end_rank"][rank])
            ),
            distance_km=float(self._sections["edge_distance_km"][rank]),
            scenic_score=self._scenic_at_rank(rank),
            road_name=self._road_name_at_rank(rank),
            road_type=self._road_type_at_rank(rank),
            speed_limit_kmh=int(self._sections["edge_speed_limit_kmh"][rank]),
            one_way=bool(self._sections["edge_one_way"][rank]),
        )

    def _traversal_ref(self, position: int) -> Tuple[str, bool]:
        rank = int(self._sections["trav_edge_rank"][position])
        return (
            self._edge_id_at_rank(rank),
            bool(self._sections["trav_reverse"][position]),
        )

    def _edge_projection_stamp(self) -> Tuple[int, int, int]:
        # Compact graphs are structurally immutable and score sidecars never
        # alter geometry, so the projection index stays valid for the graph
        # lifetime.  The CSR heuristic stamp still changes with score
        # sidecars, which is what drives weight/topology cache rebuilds.
        return (
            0,
            Node._coordinate_mutation_epoch,
            Edge._projection_epoch,
        )

    # -- immutability --------------------------------------------------------

    def add_node(self, node: Node) -> None:
        raise RuntimeError("compact road graphs are immutable")

    def add_edge(self, edge: Edge) -> None:
        raise RuntimeError("compact road graphs are immutable")

    def _bulk_load(self, nodes: Any, edges: Any, **kwargs: Any) -> Any:
        raise RuntimeError("compact road graphs are immutable")
    def close(self) -> None:
        """Close backing mmap, file handles, and sidecar projection index."""
        super().close()
        if getattr(self, "_sections", None) is not None and hasattr(self._sections, "clear"):
            self._sections.clear()
        self._sections = {}
        self._csr_arrays_cache = None
        self._csr_arrays_cache_stamp = None
        self._edge_highway_mask_cache = None
        sidecar = getattr(self, "_active_score_sidecar", None)
        if sidecar is not None:
            if hasattr(sidecar, "close"):
                try:
                    sidecar.close()
                except Exception:
                    pass
            self._active_score_sidecar = None
        mm = getattr(self, "_bin_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass
            self._bin_mmap = None
        f = getattr(self, "_bin_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._bin_file = None


    # -- compact planner arrays ----------------------------------------------

    def _edge_highway_mask(self) -> np.ndarray:
        cached = self._edge_highway_mask_cache
        if cached is not None:
            return cached
        codes = self._sections["edge_road_type_codes"]
        names = [
            self._string_at(
                self._sections["road_type_strings"],
                self._sections["road_type_offsets"],
                code,
            )
            for code in range(len(self._sections["road_type_offsets"]) - 1)
        ]
        highway_types = _compact_highway_road_types()
        mask = np.zeros(len(codes), dtype=np.bool_)
        for code, name in enumerate(names):
            if name.lower() in highway_types:
                mask[codes == code] = True
        self._edge_highway_mask_cache = mask
        return mask

    def endpoint_geodesic_bound_speed(self, avoid_highways: bool) -> Optional[float]:
        """Vectorized equivalent of the planner's graph-wide speed bound."""
        if hasattr(self, "_geodesic_bound_speed") and isinstance(self._geodesic_bound_speed, dict):
            key = "avoid_highways" if avoid_highways else "all"
            if key in self._geodesic_bound_speed:
                val = self._geodesic_bound_speed[key]
                return float(val) if val is not None else None
        distances = self._sections["edge_distance_km"]
        speeds = self._sections["edge_speed_limit_kmh"]
        start_ranks = self._sections["edge_start_rank"]
        end_ranks = self._sections["edge_end_rank"]
        node_lat = self._sections["node_lat"]
        node_lon = self._sections["node_lon"]
        start_lat = node_lat[start_ranks]
        start_lon = node_lon[start_ranks]
        end_lat = node_lat[end_ranks]
        end_lon = node_lon[end_ranks]
        dlat = np.radians(end_lat - start_lat)
        dlon = np.radians(end_lon - start_lon)
        a = np.sin(dlat / 2.0) ** 2
        a += np.cos(np.radians(start_lat)) * np.cos(np.radians(end_lat)) * np.sin(
            dlon / 2.0
        ) ** 2
        np.clip(a, 0.0, 1.0, out=a)
        endpoint_km = 2.0 * 6371.0 * np.arcsin(np.sqrt(a))
        eligible = (
            ~self._edge_highway_mask() if avoid_highways else np.ones(len(distances), dtype=np.bool_)
        )
        valid_edges = eligible & (
            np.isfinite(distances)
            & (distances >= 0.0)
            & np.isfinite(speeds)
            & np.isfinite(endpoint_km)
            & (endpoint_km >= 0.0)
            & (distances >= endpoint_km)
        )
        if np.any(eligible & ~valid_edges):
            return None
        if not np.any(valid_edges):
            return 1.0
        return float(np.maximum(np.max(speeds[valid_edges]), 1.0))

    def compact_csr_arrays(self) -> _CompactCSRArrays:
        """Return the persisted CSR topology for the active graph epoch."""
        stamp = self._heuristic_cache_stamp()
        cached = self._csr_arrays_cache
        if cached is not None and self._csr_arrays_cache_stamp == stamp:
            return cached
        trav_edge_rank = self._sections["trav_edge_rank"]
        sidecar = self._active_score_sidecar
        if sidecar is not None:
            scenic_score = lambda: np.take(sidecar.values, trav_edge_rank)
        else:
            scenic_score = self._sections["trav_scenic_score"]
        arrays = _CompactCSRArrays(
            node_count=self.node_count,
            node_ids=_CompactRankStringSequence(self, "node"),
            node_index=_CompactHashLookup(self, "node"),
            indptr=self._sections["forward_indptr"],
            indices=self._sections["forward_indices"],
            reverse_indptr=self._sections["reverse_indptr"],
            reverse_indices=self._sections["reverse_indices"],
            reverse_positions=self._sections["reverse_positions"],
            edge_refs=_CompactTraversalSequence(self),
            distance_km=self._sections["trav_distance_km"],
            travel_time_minutes=self._sections["trav_travel_time_minutes"],
            _scenic_score=scenic_score,
            highway_mask=self._sections["trav_highway_mask"].view(np.bool_),
            scenic_byway_mask=self._sections["trav_scenic_byway_mask"].view(np.bool_),
        )
        self._csr_arrays_cache = arrays
        self._csr_arrays_cache_stamp = stamp
        return arrays

    # -- edge projection sidecar ---------------------------------------------

    def _build_nearest_edge_projection_index(
        self,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> EdgeProjectionIndex:
        raise RuntimeError(
            "compact road graphs require a persisted edge projection sidecar; "
            "publish it from SQLite instead of materializing edge objects"
        )

    def _try_load_edge_projection_index(
        self,
        path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
        verify: bool = True,
    ) -> str:
        sidecar_path = EdgeProjectionIndex.sidecar_path(path)
        self._edge_projection_index_path = str(sidecar_path)
        try:
            self._edge_projection_index_payload_size = sidecar_path.stat().st_size
        except FileNotFoundError:
            self._edge_projection_index_status = "missing"
            self._edge_projection_index_invalid_reason = None
            self._nearest_edge_projection_index = None
            self._edge_projection_index_stamp = None
            return self._edge_projection_index_status
        except OSError as exc:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = f"os_error: {exc}"
            self._nearest_edge_projection_index = None
            self._edge_projection_index_stamp = None
            return self._edge_projection_index_status
        try:
            index = _RankProjectionIndex.load(
                sidecar_path, path, check_cancelled=check_cancelled, verify=verify
            )
            index.attach(self)
            self._nearest_edge_projection_index = index
            self._edge_projection_index_stamp = self._edge_projection_stamp()
            self._edge_projection_index_status = "loaded"
            self._edge_projection_index_invalid_reason = None
            self._edge_projection_index_payload_size = sidecar_path.stat().st_size
        except _SidecarCancellation as exc:
            raise exc.error
        except _SidecarMissingError:
            self._edge_projection_index_status = "missing"
            self._edge_projection_index_invalid_reason = "missing"
        except _SidecarVersionError:
            self._edge_projection_index_status = "version_mismatch"
            self._edge_projection_index_invalid_reason = "version_mismatch"
        except _SidecarStaleError:
            self._edge_projection_index_status = "stale"
            self._edge_projection_index_invalid_reason = "stale"
        except _SidecarTruncatedError:
            self._edge_projection_index_status = "truncated"
            self._edge_projection_index_invalid_reason = "truncated"
        except _SidecarCorruptError:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = "corrupt"
        except ValueError as exc:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = (
                f"attachment_error: {exc}"
            )
        except OSError as exc:
            self._edge_projection_index_status = "corrupt"
            self._edge_projection_index_invalid_reason = f"os_error: {exc}"
        if self._edge_projection_index_status != "loaded":
            self._nearest_edge_projection_index = None
            self._edge_projection_index_stamp = None
        return self._edge_projection_index_status

    # -- deterministic scenic-cost sidecar ------------------------------------

    def _score_sidecar_paths(
        self,
        report_signature: str,
        zoom: int,
        fallback: float | None,
        normalization: str,
    ) -> tuple[Path, Path]:
        base = self._manifest_path.name[: -len(_COMPACT_MANIFEST_SUFFIX)]
        tag = (
            f"score.{report_signature}.z{int(zoom)}."
            f"fb{json.dumps(fallback, separators=(',', ':'))}."
            f"n{normalization}"
        )
        json_path = self._manifest_path.with_name(f"{base}.{tag}.json")
        bin_path = self._manifest_path.with_name(f"{base}.{tag}.bin")
        return json_path, bin_path

    def activate_report_scores(
        self,
        score_map: Mapping[tuple[int, int, int], float],
        *,
        zoom: int,
        fallback: float | None,
        report_signature: str,
        normalization: str,
        tile_scores_path: Path,
        check_cancelled: Callable[[], None] | None = None,
        verify: bool = True,
    ) -> tuple[int, int]:
        """Bind one deterministic per-report score sidecar to this graph.

        The sidecar stores clamped scenic costs keyed by canonical edge rank
        and is generated deterministically on first use.  Activating it bumps
        the structure epoch so every planner cache rebuilds for the report.
        """
        json_path, bin_path = self._score_sidecar_paths(
            report_signature, zoom, fallback, normalization
        )
        manifest = _ensure_compact_score_sidecar(
            self,
            score_map,
            zoom=zoom,
            fallback=fallback,
            report_signature=report_signature,
            normalization=normalization,
            tile_scores_path=tile_scores_path,
            json_path=json_path,
            bin_path=bin_path,
            check_cancelled=check_cancelled,
            verify=verify,
        )
        file_size = bin_path.stat().st_size
        if file_size != self.edge_count * 8:
            raise ValueError(
                f"Score sidecar size {file_size} does not match "
                f"{self.edge_count} canonical edges"
            )
        file_obj = open(bin_path, "rb")
        mm = None
        try:
            mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            values = np.frombuffer(mm, dtype="<f8", count=self.edge_count)
            values.flags.writeable = False
            sidecar = _ScoreSidecar(
                path=bin_path,
                manifest=manifest,
                values=values,
                _mmap=mm,
                _file=file_obj,
            )
        except Exception:
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
            try:
                file_obj.close()
            except Exception:
                pass
            raise

        prev_sidecar = getattr(self, "_active_score_sidecar", None)
        if prev_sidecar is not None:
            if hasattr(prev_sidecar, "close"):
                try:
                    prev_sidecar.close()
                except Exception:
                    pass
        self._active_score_sidecar = sidecar
        self._heuristic_structure_epoch += 1
        matched = int(manifest["counts"]["matched_edges"])
        total = int(manifest["counts"]["total_edges"])
        fallback_edges = int(manifest["counts"]["fallback_edges"])
        object.__setattr__(
            self,
            "_route_service_score_mapping",
            (matched, total, fallback_edges),
        )
        return matched, total

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
        verify: bool = True,
    ) -> "CompactRoadGraph":
        """Load and structurally validate a compact manifest + binary payload."""
        if check_cancelled is not None:
            check_cancelled()
        manifest_path = Path(path).expanduser().resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(
                f"Invalid compact graph manifest: {manifest_path}"
            ) from exc
        if manifest.get("format") != _COMPACT_FORMAT:
            raise ValueError(
                f"Unsupported compact graph format: {manifest.get('format')!r}"
            )
        schema_version = manifest.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != _COMPACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported compact graph schema version: {schema_version!r}"
            )
        graph_counts = manifest.get("graph")
        if not isinstance(graph_counts, Mapping):
            raise ValueError("Compact manifest is missing graph counts")
        node_count = int(graph_counts.get("node_count", -1))
        edge_count = int(graph_counts.get("edge_count", -1))
        traversal_count = int(graph_counts.get("traversal_count", -1))
        if node_count < 0 or edge_count < 0 or traversal_count < 0:
            raise ValueError("Compact manifest has invalid graph counts")
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Compact manifest is missing source metadata")
        source_name = str(source.get("path", ""))
        if not source_name:
            raise ValueError("Compact manifest is missing the source graph path")
        source_path = (manifest_path.parent / source_name).resolve()
        if not source_path.is_file():
            raise ValueError(f"Compact source graph is missing: {source_path}")
        source_schema = source.get("schema_version")
        if (
            isinstance(source_schema, bool)
            or source_schema != _SQLITE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported compact source schema version: {source_schema!r}"
            )
        if int(source.get("node_count", -1)) != node_count:
            raise ValueError("Compact source node count does not match graph")
        if int(source.get("edge_count", -1)) != edge_count:
            raise ValueError("Compact source edge count does not match graph")

        bin_name = str(manifest.get("bin_path", ""))
        if not bin_name:
            raise ValueError("Compact manifest is missing the binary payload path")
        bin_path = (manifest_path.parent / bin_name).resolve()
        if not bin_path.is_file():
            raise ValueError(f"Compact binary payload is missing: {bin_path}")

        if verify:
            if check_cancelled is not None:
                check_cancelled()
            actual_bin_sha = _sha256_file(
                bin_path, check_cancelled=check_cancelled
            ).hex()
            if actual_bin_sha != str(manifest.get("bin_sha256", "")):
                raise ValueError(
                    "Compact binary payload SHA-256 does not match its manifest"
                )
            if check_cancelled is not None:
                check_cancelled()
            actual_source_sha = _sha256_file(
                source_path, check_cancelled=check_cancelled
            ).hex()
            if actual_source_sha != str(source.get("sha256", "")):
                raise ValueError(
                    "Compact source graph SHA-256 does not match its manifest"
                )

        expected_sections = dict(manifest.get("sections") or {})
        missing = _COMPACT_SECTION_NAMES.difference(expected_sections)
        if missing:
            raise ValueError(
                f"Compact manifest is missing sections: {sorted(missing)}"
            )
        for name, descriptor in expected_sections.items():
            if name not in _COMPACT_SECTION_NAMES:
                raise ValueError(f"Compact manifest has unknown section: {name}")
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"Compact section {name} has no descriptor")
            dtype = descriptor.get("dtype")
            if dtype not in _COMPACT_DTYPE_ITEMSIZE:
                raise ValueError(f"Compact section {name} has unknown dtype")
            offset = int(descriptor.get("offset", -1))
            length = int(descriptor.get("length", -1))
            count = int(descriptor.get("count", -1))
            if offset < 0 or length < 0 or count < 0:
                raise ValueError(f"Compact section {name} has invalid bounds")
            if dtype == "raw":
                if length != count:
                    raise ValueError(
                        f"Compact section {name} raw length/count mismatch"
                    )
            elif length != count * _COMPACT_DTYPE_ITEMSIZE[dtype]:
                raise ValueError(f"Compact section {name} size/count mismatch")
            if offset + length > int(manifest.get("bin_size_bytes", -1)):
                raise ValueError(f"Compact section {name} extends past payload")
        section_names = list(_COMPACT_SECTION_SPECS)
        previous_end = 0
        for name, _dtype in _COMPACT_SECTION_SPECS:
            descriptor = expected_sections[name]
            offset = int(descriptor["offset"])
            if offset != previous_end:
                raise ValueError(
                    f"Compact section {name} offset is not contiguous"
                )
            previous_end = offset + int(descriptor["length"])
        if previous_end != int(manifest.get("bin_size_bytes", -1)):
            raise ValueError("Compact section payload has trailing bytes")

        if (
            str(manifest.get("scenic_byway_road_type"))
            != _COMPACT_SCENIC_BYWAY_ROAD_TYPE
        ):
            raise ValueError("Compact manifest scenic byway marker mismatch")
        if str(manifest.get("highway_road_types", "")) != ",".join(
            sorted(_compact_highway_road_types())
        ):
            raise ValueError(
                "Compact manifest highway road-type mask is stale; rebuild it"
            )

        if check_cancelled is not None:
            check_cancelled()
        file_size = bin_path.stat().st_size
        if file_size != int(manifest.get("bin_size_bytes", -1)):
            raise ValueError("Compact binary payload size mismatch")
        bin_file = open(bin_path, "rb")
        try:
            mm = mmap.mmap(bin_file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            bin_file.close()
            raise

        graph: CompactRoadGraph | None = None
        try:
            sections: Dict[str, np.ndarray | memoryview] = {}
            for name, dtype in _COMPACT_SECTION_SPECS:
                descriptor = expected_sections[name]
                offset = int(descriptor["offset"])
                length = int(descriptor["length"])
                count = int(descriptor["count"])
                if dtype == "raw":
                    sections[name] = memoryview(mm)[offset : offset + length]
                else:
                    arr = np.frombuffer(
                        mm,
                        dtype=np.dtype("<" + dtype),
                        count=count,
                        offset=offset,
                    )
                    arr.flags.writeable = False
                    sections[name] = arr

            graph = cls(
                manifest,
                sections,
                mm,
                bin_file,
                manifest_path,
                source_path,
            )
            if check_cancelled is not None:
                check_cancelled()
            try:
                current_path_identity = _path_identity(manifest_path)
            except OSError:
                current_path_identity = None
            if current_path_identity is not None:
                graph._try_load_edge_projection_index(
                    source_path,
                    check_cancelled=check_cancelled,
                    verify=verify,
                )
            else:
                graph._edge_projection_index_path = str(
                    EdgeProjectionIndex.sidecar_path(source_path)
                )
                graph._edge_projection_index_status = "stale"
                graph._edge_projection_index_invalid_reason = (
                    "graph_replaced_during_load"
                )
            if check_cancelled is not None:
                check_cancelled()
            return graph
        except BaseException:
            arr = None
            sections.clear()
            if graph is not None:
                graph.close()
            else:
                try:
                    mm.close()
                except Exception:
                    pass
                try:
                    bin_file.close()
                except Exception:
                    pass
            raise


def _validate_compact_score_manifest(
    manifest: Mapping[str, Any],
    *,
    report_signature: str,
    zoom: int,
    fallback: float | None,
    normalization: str,
    source_sha256: str,
    edge_count: int,
) -> None:
    if manifest.get("format") != _COMPACT_SCORE_FORMAT:
        raise ValueError(
            f"Unsupported score sidecar format: {manifest.get('format')!r}"
        )
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != _COMPACT_SCORE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported score sidecar schema version: {schema_version!r}"
        )
    if str(manifest.get("report_signature")) != report_signature:
        raise ValueError("Score sidecar report signature mismatch")
    if int(manifest.get("zoom", -1)) != int(zoom):
        raise ValueError("Score sidecar zoom mismatch")
    expected_fallback = manifest.get("fallback")
    if (
        expected_fallback is None and fallback is not None
    ) or (
        expected_fallback is not None
        and (fallback is None or float(expected_fallback) != float(fallback))
    ):
        raise ValueError("Score sidecar fallback mismatch")
    if str(manifest.get("normalization")) != normalization:
        raise ValueError("Score sidecar normalization version mismatch")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Score sidecar is missing source metadata")
    if str(source.get("sha256", "")) != source_sha256:
        raise ValueError("Score sidecar source graph mismatch")
    if int(source.get("edge_count", -1)) != edge_count:
        raise ValueError("Score sidecar edge count mismatch")


def _ensure_compact_score_sidecar(
    graph: CompactRoadGraph,
    score_map: Mapping[tuple[int, int, int], float],
    *,
    zoom: int,
    fallback: float | None,
    report_signature: str,
    normalization: str,
    tile_scores_path: Path,
    json_path: Path,
    bin_path: Path,
    check_cancelled: Callable[[], None] | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Generate or validate one deterministic per-report score sidecar."""
    if check_cancelled is not None:
        check_cancelled()
    if json_path.is_file() and bin_path.is_file():
        try:
            manifest = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"Invalid score sidecar manifest: {json_path}") from exc
        _validate_compact_score_manifest(
            manifest,
            report_signature=report_signature,
            zoom=zoom,
            fallback=fallback,
            normalization=normalization,
            source_sha256=graph.source_sha256,
            edge_count=graph.edge_count,
        )
        if bin_path.stat().st_size != graph.edge_count * 8:
            raise ValueError("Score sidecar payload size mismatch")
        if verify and _sha256_file(
            bin_path, check_cancelled=check_cancelled
        ).hex() != str(manifest.get("bin_sha256", "")):
            raise ValueError("Score sidecar payload SHA-256 mismatch")
        return manifest

    # Deterministic vectorized tile-score computation over canonical edges.
    from src.data_pipeline.web_mercator import WEB_MERCATOR_MAX_LAT

    if check_cancelled is not None:
        check_cancelled()
    start_ranks = graph._sections["edge_start_rank"]
    end_ranks = graph._sections["edge_end_rank"]
    node_lat = graph._sections["node_lat"]
    node_lon = graph._sections["node_lon"]
    mid_lat = 0.5 * (node_lat[start_ranks] + node_lat[end_ranks])
    mid_lon = 0.5 * (node_lon[start_ranks] + node_lon[end_ranks])
    clamped_lat = np.clip(
        mid_lat, -float(WEB_MERCATOR_MAX_LAT), float(WEB_MERCATOR_MAX_LAT)
    )
    n = 1 << int(zoom)
    tile_x = ((mid_lon + 180.0) / 360.0 * n).astype(np.int64)
    tile_y = (
        (1.0 - np.arcsinh(np.tan(np.radians(clamped_lat))) / np.pi)
        / 2.0
        * n
    ).astype(np.int64)
    tile_x = np.clip(tile_x, 0, n - 1)
    tile_y = np.clip(tile_y, 0, n - 1)
    tiles = np.stack((tile_x, tile_y), axis=1)
    unique_tiles, inverse = np.unique(tiles, axis=0, return_inverse=True)
    per_tile_score = np.full(len(unique_tiles), np.nan, dtype=np.float64)
    for tile_index, (tx, ty) in enumerate(unique_tiles):
        if check_cancelled is not None and tile_index & 1023 == 0:
            check_cancelled()
        value = score_map.get((int(zoom), int(tx), int(ty)))
        if value is not None:
            per_tile_score[tile_index] = float(value)
    per_edge_tile = per_tile_score[inverse]
    present = np.isfinite(per_edge_tile)
    values = np.array(graph._sections["edge_scenic_score"], dtype=np.float64, copy=True)
    values[present] = np.clip(per_edge_tile[present], 0.0, 10.0)
    if fallback is not None:
        values[~present] = float(min(max(float(fallback), 0.0), 10.0))
    matched = int(np.count_nonzero(present))
    total = graph.edge_count
    fallback_edges = int(total - matched) if fallback is not None else 0
    del per_edge_tile, per_tile_score
    if check_cancelled is not None:
        check_cancelled()
    payload = values.astype("<f8", copy=False).tobytes()
    bin_sha256 = hashlib.sha256(payload).hexdigest()
    _write_atomic_bytes(bin_path, payload)
    tile_sha256 = _sha256_file(
        Path(tile_scores_path), check_cancelled=check_cancelled
    ).hex()
    manifest = {
        "format": _COMPACT_SCORE_FORMAT,
        "schema_version": _COMPACT_SCORE_SCHEMA_VERSION,
        "report_signature": report_signature,
        "tile_scores_source": str(tile_scores_path),
        "tile_scores_sha256": tile_sha256,
        "source": {
            "path": graph.source_path.name,
            "sha256": graph.source_sha256,
            "size_bytes": int(graph.source_path.stat().st_size),
            "edge_count": total,
        },
        "zoom": int(zoom),
        "fallback": fallback,
        "normalization": normalization,
        "counts": {
            "matched_edges": matched,
            "fallback_edges": fallback_edges,
            "total_edges": total,
        },
        "bin_path": bin_path.name,
        "bin_sha256": bin_sha256,
        "bin_size_bytes": len(payload),
    }
    _write_atomic_bytes(
        json_path,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return manifest




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
