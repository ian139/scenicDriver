from __future__ import annotations

import hashlib
import heapq
import math
import mmap
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np

_EARTH_RADIUS_KM = 6371.0
_EDGE_PROJECTION_BVH_LEAF_SIZE = 64
_EDGE_PROJECTION_TIE_TOLERANCE_KM = 1e-9
_CANCELLATION_CHECK_INTERVAL = 1024

_EPI_MAGIC = b"SCENEDGE"
_EPI_VERSION = 2
_EPI_HEADER_SIZE = 88
_EPI_SECTION_DESCRIPTOR_SIZE = 64

# Fixed section order for version 2 (rank-based keys).
_SECTION_EDGE_RANKS = 0
_SECTION_TYPE_NAMES_DATA = 1
_SECTION_TYPE_NAMES_OFFSETS = 2
_SECTION_TYPE_CODES = 3
_SECTION_START_LAT = 4
_SECTION_START_LON = 5
_SECTION_END_LAT = 6
_SECTION_END_LON = 7
_SECTION_BVH_MIN_LAT = 8
_SECTION_BVH_MIN_LON = 9
_SECTION_BVH_MAX_LAT = 10
_SECTION_BVH_MAX_LON = 11
_SECTION_BVH_LEFT = 12
_SECTION_BVH_RIGHT = 13
_SECTION_BVH_START = 14
_SECTION_BVH_STOP = 15
_NUM_SECTIONS = 16

_DTYPE_CODE_TO_NAME: dict[int, str | None] = {
    0: None,
    1: "<f8",
    2: "<i8",
    3: "<i4",
}
_DTYPE_NAME_TO_CODE: dict[str, int] = {
    "<f8": 1,
    "<i8": 2,
    "<i4": 3,
}


class _SidecarError(Exception):
    """Base for sidecar validation failures."""


class _SidecarMissingError(_SidecarError):
    pass


class _SidecarVersionError(_SidecarError):
    pass


class _SidecarStaleError(_SidecarError):
    pass


class _SidecarTruncatedError(_SidecarError):
    pass


class _SidecarCorruptError(_SidecarError):
    pass


class _SidecarCancellation(BaseException):
    """Carry a caller cancellation through corruption-recovery boundaries."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _normalize_road_type(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        value = str(raw[0]).strip().lower()
    else:
        value = str(raw).strip().lower() if raw is not None else ""
    return value if value else "secondary"


def _sha256_buffer(
    payload: memoryview,
    check_cancelled: Callable[[], None] | None = None,
) -> bytes:
    digest = hashlib.sha256()
    for start in range(0, len(payload), 8 * 1024 * 1024):
        if check_cancelled is not None:
            check_cancelled()
        digest.update(payload[start : start + 8 * 1024 * 1024])
    if check_cancelled is not None:
        check_cancelled()
    return digest.digest()


@dataclass(frozen=True)
class EdgeProjection:
    edge: Any
    fraction: float
    lat: float
    lon: float
    snap_distance_km: float


class EdgeProjectionIndex:
    """Typed, immutable spatial index for nearest finite edge-segment queries."""

    __slots__ = (
        "total_edge_count",
        "edge_ranks",
        "road_type_names",
        "road_type_codes",
        "start_latitudes",
        "start_longitudes",
        "end_latitudes",
        "end_longitudes",
        "bvh_min_lat",
        "bvh_min_lon",
        "bvh_max_lat",
        "bvh_max_lon",
        "bvh_left",
        "bvh_right",
        "bvh_start",
        "bvh_stop",
        "leaf_size",
        "root_node",
        "_mmap",
        "_file",
        "_canonical_keys",
        "_canonical_keys_owner",
    )

    def __init__(
        self,
        edge_ranks: np.ndarray,
        total_edge_count: int,
        road_type_names: Tuple[str, ...],
        road_type_codes: np.ndarray,
        start_latitudes: np.ndarray,
        start_longitudes: np.ndarray,
        end_latitudes: np.ndarray,
        end_longitudes: np.ndarray,
        bvh_min_lat: np.ndarray,
        bvh_min_lon: np.ndarray,
        bvh_max_lat: np.ndarray,
        bvh_max_lon: np.ndarray,
        bvh_left: np.ndarray,
        bvh_right: np.ndarray,
        bvh_start: np.ndarray,
        bvh_stop: np.ndarray,
        leaf_size: int,
        root_node: int = 0,
        mmap_ref: mmap.mmap | None = None,
        file_ref: Any = None,
    ) -> None:
        self.edge_ranks = edge_ranks
        self.road_type_names = road_type_names
        self.road_type_codes = road_type_codes
        self.start_latitudes = start_latitudes
        self.start_longitudes = start_longitudes
        self.end_latitudes = end_latitudes
        self.end_longitudes = end_longitudes
        self.bvh_min_lat = bvh_min_lat
        self.bvh_min_lon = bvh_min_lon
        self.bvh_max_lat = bvh_max_lat
        self.bvh_max_lon = bvh_max_lon
        self.bvh_left = bvh_left
        self.bvh_right = bvh_right
        self.bvh_start = bvh_start
        self.bvh_stop = bvh_stop
        self.leaf_size = leaf_size
        self.root_node = root_node
        self._mmap = mmap_ref
        self._file = file_ref
        self._canonical_keys: tuple[str, ...] | None = None
        self._canonical_keys_owner: int | None = None
        self.total_edge_count = total_edge_count

    def attach(self, graph: Any) -> None:
        """Bind once to the graph's canonical edge-key order."""
        owner = id(graph)
        if self._canonical_keys is not None:
            if self._canonical_keys_owner != owner:
                raise ValueError("Edge projection index is attached to another graph")
            return
        canonical_keys = tuple(graph.edges)
        if len(canonical_keys) != self.total_edge_count:
            raise ValueError("Graph edge count does not match sidecar total edge count")
        self._canonical_keys = canonical_keys
        self._canonical_keys_owner = owner
    def close(self) -> None:
        """Close backing memory-map and file handle if held."""
        self.edge_ranks = np.empty(0, dtype=np.int64)
        self.road_type_codes = np.empty(0, dtype=np.int32)
        self.start_latitudes = np.empty(0, dtype=np.float64)
        self.start_longitudes = np.empty(0, dtype=np.float64)
        self.end_latitudes = np.empty(0, dtype=np.float64)
        self.end_longitudes = np.empty(0, dtype=np.float64)
        self.bvh_min_lat = np.empty(0, dtype=np.float64)
        self.bvh_min_lon = np.empty(0, dtype=np.float64)
        self.bvh_max_lat = np.empty(0, dtype=np.float64)
        self.bvh_max_lon = np.empty(0, dtype=np.float64)
        self.bvh_left = np.empty(0, dtype=np.int64)
        self.bvh_right = np.empty(0, dtype=np.int64)
        self.bvh_start = np.empty(0, dtype=np.int64)
        self.bvh_stop = np.empty(0, dtype=np.int64)
        mm = getattr(self, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass
            self._mmap = None
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._file = None
    def __enter__(self) -> "EdgeProjectionIndex":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


    @property
    def edge_count(self) -> int:
        return len(self.edge_ranks)

    @property
    def node_count(self) -> int:
        return len(self.bvh_left)

    @classmethod
    def build(
        cls,
        graph: Any,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> "EdgeProjectionIndex":
        """Build exact finite-segment arrays and a balanced BVH over edge order."""
        if check_cancelled is not None:
            check_cancelled()

        total_edge_count = len(graph.edges)
        edge_rank_values: list[int] | np.ndarray = []
        road_type_codes: list[int] = []
        road_type_index: dict[str, int] = {}
        start_latitudes: list[float] = []
        start_longitudes: list[float] = []
        end_latitudes: list[float] = []
        end_longitudes: list[float] = []

        for edge_index, (edge_key, edge) in enumerate(graph.edges.items()):
            if (
                check_cancelled is not None
                and edge_index & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            start = graph.nodes.get(edge.start_node_id)
            end = graph.nodes.get(edge.end_node_id)
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
            edge_rank_values.append(edge_index)
            road_type_codes.append(code)
            start_latitudes.append(start_lat)
            start_longitudes.append(start_lon)
            end_latitudes.append(end_lat)
            end_longitudes.append(end_lon)

        if check_cancelled is not None:
            check_cancelled()

        if not edge_rank_values:
            index = cls._empty(total_edge_count=total_edge_count)
            index.attach(graph)
            return index

        road_type_names = tuple(road_type_index.keys())
        codes = np.asarray(road_type_codes, dtype=np.int32)
        slat = np.asarray(start_latitudes, dtype=np.float64)
        slon = np.asarray(start_longitudes, dtype=np.float64)
        elat = np.asarray(end_latitudes, dtype=np.float64)
        elon = np.asarray(end_longitudes, dtype=np.float64)

        order = np.arange(len(edge_rank_values), dtype=np.int64)

        bvh_min_lat: list[float] = []
        bvh_min_lon: list[float] = []
        bvh_max_lat: list[float] = []
        bvh_max_lon: list[float] = []
        bvh_left: list[int] = []
        bvh_right: list[int] = []
        bvh_start: list[int] = []
        bvh_stop: list[int] = []
        build_calls = 0

        def build_bvh(lo: int, hi: int, depth: int) -> int:
            nonlocal build_calls
            if check_cancelled is not None and (
                build_calls & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            build_calls += 1

            if hi - lo <= _EDGE_PROJECTION_BVH_LEAF_SIZE:
                segment = order[lo:hi]
                min_lat = float(np.min(slat[segment]))
                max_lat = float(np.max(slat[segment]))
                min_lon = float(np.min(slon[segment]))
                max_lon = float(np.max(slon[segment]))
                min_lon = min(
                    min_lon,
                    float(np.min(elon[segment])),
                )
                max_lon = max(
                    max_lon,
                    float(np.max(elon[segment])),
                )
                min_lat = min(min_lat, float(np.min(elat[segment])))
                max_lat = max(max_lat, float(np.max(elat[segment])))
                pos = len(bvh_left)
                bvh_min_lat.append(min_lat)
                bvh_min_lon.append(min_lon)
                bvh_max_lat.append(max_lat)
                bvh_max_lon.append(max_lon)
                bvh_left.append(-1)
                bvh_right.append(-1)
                bvh_start.append(lo)
                bvh_stop.append(hi)
                return pos

            segment = order[lo:hi]
            min_lat = min(
                float(np.min(slat[segment])),
                float(np.min(elat[segment])),
            )
            max_lat = max(
                float(np.max(slat[segment])),
                float(np.max(elat[segment])),
            )
            min_lon = min(
                float(np.min(slon[segment])),
                float(np.min(elon[segment])),
            )
            max_lon = max(
                float(np.max(slon[segment])),
                float(np.max(elon[segment])),
            )
            longitude_scale = max(
                0.0,
                math.cos(math.radians((min_lat + max_lat) * 0.5)),
            )
            axis = (
                1
                if (max_lon - min_lon) * longitude_scale
                > max_lat - min_lat
                else 0
            )
            mid = (lo + hi) // 2
            if axis == 0:
                centers = (slat[segment] + elat[segment]) * 0.5
            else:
                centers = (slon[segment] + elon[segment]) * 0.5
            kth = mid - lo
            permutation = np.argpartition(centers, kth)
            order[lo:hi] = order[lo:hi][permutation]

            left = build_bvh(lo, mid, depth + 1)
            right = build_bvh(mid, hi, depth + 1)

            pos = len(bvh_left)
            bvh_min_lat.append(min(bvh_min_lat[left], bvh_min_lat[right]))
            bvh_min_lon.append(min(bvh_min_lon[left], bvh_min_lon[right]))
            bvh_max_lat.append(max(bvh_max_lat[left], bvh_max_lat[right]))
            bvh_max_lon.append(max(bvh_max_lon[left], bvh_max_lon[right]))
            bvh_left.append(left)
            bvh_right.append(right)
            bvh_start.append(-1)
            bvh_stop.append(-1)
            return pos

        root = build_bvh(0, len(edge_rank_values), 0)

        if check_cancelled is not None:
            check_cancelled()

        # Reorder arrays to match the BVH edge order so leaf ranges are contiguous.
        edge_rank_values = np.asarray([edge_rank_values[i] for i in order], dtype=np.int64)  # type: ignore[assignment]
        codes = codes[order]
        slat = slat[order]
        slon = slon[order]
        elat = elat[order]
        elon = elon[order]

        if check_cancelled is not None:
            check_cancelled()

        index = cls(
            edge_ranks=edge_rank_values,
            total_edge_count=total_edge_count,
            road_type_names=road_type_names,
            road_type_codes=codes,
            start_latitudes=slat,
            start_longitudes=slon,
            end_latitudes=elat,
            end_longitudes=elon,
            bvh_min_lat=np.asarray(bvh_min_lat, dtype=np.float64),
            bvh_min_lon=np.asarray(bvh_min_lon, dtype=np.float64),
            bvh_max_lat=np.asarray(bvh_max_lat, dtype=np.float64),
            bvh_max_lon=np.asarray(bvh_max_lon, dtype=np.float64),
            bvh_left=np.asarray(bvh_left, dtype=np.int64),
            bvh_right=np.asarray(bvh_right, dtype=np.int64),
            bvh_start=np.asarray(bvh_start, dtype=np.int64),
            bvh_stop=np.asarray(bvh_stop, dtype=np.int64),
            leaf_size=_EDGE_PROJECTION_BVH_LEAF_SIZE,
            root_node=root,
        )
        index.attach(graph)
        return index

    @classmethod
    def _empty(cls, total_edge_count: int = 0) -> "EdgeProjectionIndex":
        return cls(
            edge_ranks=np.empty(0, dtype=np.int64),
            total_edge_count=total_edge_count,  # type: ignore[call-arg]
            road_type_names=(),
            road_type_codes=np.empty(0, dtype=np.int32),
            start_latitudes=np.empty(0, dtype=np.float64),
            start_longitudes=np.empty(0, dtype=np.float64),
            end_latitudes=np.empty(0, dtype=np.float64),
            end_longitudes=np.empty(0, dtype=np.float64),
            bvh_min_lat=np.empty(0, dtype=np.float64),
            bvh_min_lon=np.empty(0, dtype=np.float64),
            bvh_max_lat=np.empty(0, dtype=np.float64),
            bvh_max_lon=np.empty(0, dtype=np.float64),
            bvh_left=np.empty(0, dtype=np.int64),
            bvh_right=np.empty(0, dtype=np.int64),
            bvh_start=np.empty(0, dtype=np.int64),
            bvh_stop=np.empty(0, dtype=np.int64),
            leaf_size=_EDGE_PROJECTION_BVH_LEAF_SIZE,
            root_node=-1,
        )

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
        """Project a query point onto a chunk of finite edge segments."""
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
        projected_latitudes = start_lat + fractions * (
            end_latitudes[start:stop] - start_lat
        )
        projected_longitudes = start_lon + fractions * (
            end_longitudes[start:stop] - start_lon
        )

        with np.errstate(over="ignore", invalid="ignore"):
            dlat = np.radians(projected_latitudes - query_lat)
            dlon = np.radians(projected_longitudes - query_lon)
            haversine = np.sin(dlat / 2.0) ** 2
            haversine += (
                math.cos(math.radians(query_lat))
                * np.cos(np.radians(projected_latitudes))
                * np.sin(dlon / 2.0) ** 2
            )
            np.clip(haversine, 0.0, 1.0, out=haversine)
            np.sqrt(haversine, out=haversine)
            np.arcsin(haversine, out=haversine)
            haversine *= 2.0 * _EARTH_RADIUS_KM
        return fractions, projected_latitudes, projected_longitudes, haversine

    def _bvh_lower_bound(
        self,
        query_lat: float,
        query_lon: float,
        node: int,
    ) -> float:
        """Certified great-circle lower bound to a latitude/longitude box."""
        min_lat = float(self.bvh_min_lat[node])
        max_lat = float(self.bvh_max_lat[node])
        min_lon = float(self.bvh_min_lon[node])
        max_lon = float(self.bvh_max_lon[node])
        if not (
            -90.0 <= query_lat <= 90.0
            and -90.0 <= min_lat <= max_lat <= 90.0
        ):
            return 0.0

        latitude_delta = max(
            0.0,
            min_lat - query_lat,
            query_lat - max_lat,
        )
        longitude_span = max_lon - min_lon
        if longitude_span >= 360.0:
            longitude_delta = 0.0
        else:
            center = min_lon + longitude_span * 0.5
            normalized_delta = math.remainder(query_lon, 360.0)
            normalized_delta -= math.remainder(center, 360.0)
            shifted_query_lon = center + math.remainder(
                normalized_delta,
                360.0,
            )
            longitude_delta = max(
                0.0,
                min_lon - shifted_query_lon,
                shifted_query_lon - max_lon,
            )
            longitude_delta = min(longitude_delta, 180.0)

        latitude_term = math.sin(math.radians(latitude_delta) * 0.5) ** 2
        minimum_box_cosine = min(
            math.cos(math.radians(min_lat)),
            math.cos(math.radians(max_lat)),
        )
        longitude_term = (
            max(0.0, math.cos(math.radians(query_lat)) * minimum_box_cosine)
            * math.sin(math.radians(longitude_delta) * 0.5) ** 2
        )
        haversine = min(1.0, max(0.0, latitude_term + longitude_term))
        return (
            2.0
            * _EARTH_RADIUS_KM
            * math.asin(math.sqrt(haversine))
        )

    def query(
        self,
        graph: Any,
        query_lat: float,
        query_lon: float,
        excluded_road_types: frozenset | set | None = None,
        *,
        check_cancelled: Callable[[], None] | None = None,
        with_stats: bool = False,
    ) -> (
        tuple[list[EdgeProjection], float]
        | tuple[tuple[list[EdgeProjection], float], dict[str, Any]]
    ):
        """Best-first BVH query; exact-refine leaves, apply exclusions, return ties."""
        if check_cancelled is not None:
            check_cancelled()

        query_lat = float(query_lat)
        query_lon = float(query_lon)
        if not math.isfinite(query_lat) or not math.isfinite(query_lon):
            raise ValueError("Query coordinates must be finite")

        if self.edge_count == 0 or self.root_node < 0:
            raise ValueError("Road graph has no eligible finite segment")

        excluded = {
            str(road_type).strip().lower() for road_type in (excluded_road_types or ())
        }
        allowed = np.asarray(
            [road_type not in excluded for road_type in self.road_type_names],
            dtype=np.bool_,
        )
        if not np.any(allowed):
            raise ValueError("Road graph has no eligible finite segment")

        longitude_scale = math.cos(math.radians(query_lat))
        tol = _EDGE_PROJECTION_TIE_TOLERANCE_KM
        heap: list[tuple[float, int]] = [(0.0, self.root_node)]
        best_distance = float("inf")
        candidates: list[tuple[int, float, float, float, float]] = []
        search_steps = 0
        visited_leaves = 0
        scanned_edges = 0
        max_candidates = 0

        while heap:
            if (
                check_cancelled is not None
                and search_steps & (_CANCELLATION_CHECK_INTERVAL - 1) == 0
            ):
                check_cancelled()
            search_steps += 1

            lower_bound, node = heapq.heappop(heap)
            if lower_bound > best_distance + tol:
                break

            if self.bvh_left[node] < 0:
                start = int(self.bvh_start[node])
                stop = int(self.bvh_stop[node])
                if start >= stop:
                    continue
                visited_leaves += 1
                scanned_edges += stop - start
                fractions, projected_latitudes, projected_longitudes, distances = (
                    graph._project_edge_chunk(
                        query_lat,
                        query_lon,
                        longitude_scale,
                        self.start_latitudes,
                        self.start_longitudes,
                        self.end_latitudes,
                        self.end_longitudes,
                        start,
                        stop,
                    )
                )
                if check_cancelled is not None:
                    check_cancelled()

                codes = self.road_type_codes[start:stop]
                eligible = allowed[codes] & np.isfinite(distances)
                if not np.any(eligible):
                    continue

                local_min = float(np.min(distances[eligible]))
                if local_min < best_distance:
                    best_distance = local_min
                    cutoff = best_distance + tol
                    candidates = [c for c in candidates if c[4] <= cutoff]

                cutoff = best_distance + tol
                for local_index in np.flatnonzero(eligible):
                    dist = float(distances[local_index])
                    if dist <= cutoff:
                        candidates.append(
                            (
                                start + int(local_index),
                                float(fractions[local_index]),
                                float(projected_latitudes[local_index]),
                                float(projected_longitudes[local_index]),
                                dist,
                            )
                        )
                max_candidates = max(max_candidates, len(candidates))
                continue

            left = int(self.bvh_left[node])
            right = int(self.bvh_right[node])
            lb_left = self._bvh_lower_bound(query_lat, query_lon, left)
            lb_right = self._bvh_lower_bound(query_lat, query_lon, right)
            if lb_left <= best_distance + tol:
                heapq.heappush(heap, (lb_left, left))
            if lb_right <= best_distance + tol:
                heapq.heappush(heap, (lb_right, right))

        if not math.isfinite(best_distance):
            raise ValueError("Road graph has no eligible finite segment")

        canonical_keys = self._canonical_keys
        if canonical_keys is None or self._canonical_keys_owner != id(graph):
            raise RuntimeError("Edge projection index is not attached to this graph")

        cutoff = best_distance + tol
        keyed_projections: list[tuple[tuple[str, str], EdgeProjection]] = []
        for idx, fraction, plat, plon, dist in candidates:
            if dist > cutoff:
                continue
            rank = int(self.edge_ranks[idx])
            if rank < 0 or rank >= len(canonical_keys):
                # Sidecar/graph mismatch; skip defensively.
                continue
            edge_key = canonical_keys[rank]
            edge = graph.edges.get(edge_key)
            if edge is None:
                # Graph mutated under the index; skip defensively.
                continue
            keyed_projections.append(
                (
                    (str(edge.id), edge_key),
                    EdgeProjection(
                        edge=edge,
                        fraction=fraction,
                        lat=plat,
                        lon=plon,
                        snap_distance_km=dist,
                    ),
                )
            )

        keyed_projections.sort(key=lambda item: item[0])
        projections = [projection for _key, projection in keyed_projections]
        if with_stats:
            stats = {
                "visited_leaves": visited_leaves,
                "scanned_edges": scanned_edges,
                "max_candidates": max_candidates,
                "final_candidates": len(keyed_projections),
                "search_steps": search_steps,
            }
            return (projections, best_distance), stats
        return projections, best_distance

    def _write_sections(
        self,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> list[tuple[memoryview, int, int, int, bytes]]:
        """Expose deterministic little-endian section buffers without copies."""
        type_data = b"".join(name.encode("utf-8") for name in self.road_type_names)
        type_offsets = np.empty(len(self.road_type_names) + 1, dtype=np.int64)
        offset = 0
        for i, name in enumerate(self.road_type_names):
            type_offsets[i] = offset
            offset += len(name.encode("utf-8"))
        type_offsets[-1] = offset

        def as_section(
            arr: np.ndarray,
            dtype_name: str,
        ) -> tuple[memoryview, int, int, int, bytes]:
            target = np.dtype(dtype_name)
            if arr.dtype != target:
                arr = arr.astype(target)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr, dtype=target)
            payload = memoryview(bytes(arr)).cast("B")
            return (
                payload,
                len(payload),
                int(arr.size),
                _DTYPE_NAME_TO_CODE[dtype_name],
                _sha256_buffer(payload, check_cancelled),
            )

        raw_types = memoryview(type_data)
        return [
            as_section(self.edge_ranks, "<i8"),
            (
                raw_types,
                len(raw_types),
                len(raw_types),
                0,
                _sha256_buffer(raw_types, check_cancelled),
            ),
            as_section(type_offsets, "<i8"),
            as_section(self.road_type_codes, "<i4"),
            as_section(self.start_latitudes, "<f8"),
            as_section(self.start_longitudes, "<f8"),
            as_section(self.end_latitudes, "<f8"),
            as_section(self.end_longitudes, "<f8"),
            as_section(self.bvh_min_lat, "<f8"),
            as_section(self.bvh_min_lon, "<f8"),
            as_section(self.bvh_max_lat, "<f8"),
            as_section(self.bvh_max_lon, "<f8"),
            as_section(self.bvh_left, "<i8"),
            as_section(self.bvh_right, "<i8"),
            as_section(self.bvh_start, "<i8"),
            as_section(self.bvh_stop, "<i8"),
        ]

    @staticmethod
    def _aligned_size(size: int, alignment: int = 8) -> int:
        return ((size + alignment - 1) // alignment) * alignment

    @classmethod
    def write(
        cls,
        index: "EdgeProjectionIndex",
        graph_path: Path,
        sidecar_path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        """Write a versioned, graph-bound sidecar atomically."""
        if check_cancelled is not None:
            check_cancelled()

        graph_path = Path(graph_path)
        sidecar_path = Path(sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)

        file_size = graph_path.stat().st_size
        digest = hashlib.sha256()
        with open(graph_path, "rb") as f:
            while True:
                if check_cancelled is not None:
                    check_cancelled()
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        graph_sha256 = digest.digest()

        if check_cancelled is not None:
            check_cancelled()

        descriptors = index._write_sections(check_cancelled=check_cancelled)
        if check_cancelled is not None:
            check_cancelled()

        section_count = _NUM_SECTIONS
        header_size = cls._aligned_size(
            _EPI_HEADER_SIZE + section_count * _EPI_SECTION_DESCRIPTOR_SIZE
        )
        current_offset = header_size
        final_descriptors: list[tuple[int, int, int, int, bytes]] = []
        for payload, size, count, dtype_code, sha in descriptors:
            desc_size = cls._aligned_size(size)
            final_descriptors.append((current_offset, size, count, dtype_code, sha))
            current_offset += desc_size

        buffer = bytearray()
        buffer.extend(
            struct.pack(
                "<8sIIQQQQIi32s",
                _EPI_MAGIC,
                _EPI_VERSION,
                header_size,
                file_size,
                index.edge_count,
                index.total_edge_count,
                index.node_count,
                index.leaf_size,
                index.root_node,
                graph_sha256,
            )
        )
        for offset, size, count, dtype_code, sha in final_descriptors:
            buffer.extend(
                struct.pack(
                    "<QQQB7x32s",
                    offset,
                    size,
                    count,
                    dtype_code,
                    sha,
                )
            )
        buffer.extend(b"\x00" * (header_size - len(buffer)))

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{sidecar_path.name}.",
            suffix=".tmp",
            dir=sidecar_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                stream.write(buffer)
                for payload, _size, _count, _dtype, _sha in descriptors:
                    for start in range(0, len(payload), 8 * 1024 * 1024):
                        if check_cancelled is not None:
                            check_cancelled()
                        stream.write(payload[start : start + 8 * 1024 * 1024])
                    padding = cls._aligned_size(len(payload)) - len(payload)
                    if padding:
                        stream.write(b"\x00" * padding)
                stream.flush()
                os.fsync(stream.fileno())
            if check_cancelled is not None:
                check_cancelled()
            os.replace(temporary_path, sidecar_path)
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(sidecar_path.parent, directory_flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        sidecar_path: Path,
        graph_path: Path,
        *,
        check_cancelled: Callable[[], None] | None = None,
        verify: bool = True,
    ) -> "EdgeProjectionIndex":
        """Load and validate a sidecar, raising on any inconsistency."""
        callback = check_cancelled
        if callback is not None:

            def checked_cancelled() -> None:
                try:
                    callback()
                except BaseException as exc:
                    raise _SidecarCancellation(exc) from exc

            check_cancelled = checked_cancelled
        sidecar_path = Path(sidecar_path)
        graph_path = Path(graph_path)
        if not sidecar_path.exists():
            raise _SidecarMissingError(f"Sidecar missing: {sidecar_path}")

        if check_cancelled is not None:
            check_cancelled()

        file_size = sidecar_path.stat().st_size
        if file_size < _EPI_HEADER_SIZE:
            raise _SidecarTruncatedError("Sidecar smaller than header")

        file_obj = open(sidecar_path, "rb")
        mm: mmap.mmap | None = None
        try:
            mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            header = mm[:_EPI_HEADER_SIZE]
            (
                magic,
                version,
                header_size,
                bound_file_size,
                edge_count,
                total_edge_count,
                bvh_node_count,
                leaf_size,
                root_node,
                bound_sha256,
            ) = struct.unpack("<8sIIQQQQIi32s", header)

            if magic != _EPI_MAGIC:
                raise _SidecarCorruptError("Sidecar magic mismatch")
            if version != _EPI_VERSION:
                raise _SidecarVersionError(
                    f"Unsupported sidecar version: {version}"
                )

            expected_header_size = cls._aligned_size(
                _EPI_HEADER_SIZE + _NUM_SECTIONS * _EPI_SECTION_DESCRIPTOR_SIZE
            )
            if header_size != expected_header_size:
                raise _SidecarCorruptError("Sidecar header size mismatch")

            if file_size < header_size:
                raise _SidecarTruncatedError(
                    f"Sidecar file size {file_size} is smaller than header directory {header_size}"
                )

            if check_cancelled is not None:
                check_cancelled()

            # Validate graph binding.
            graph_stat = graph_path.stat()
            if graph_stat.st_size != bound_file_size:
                raise _SidecarStaleError("Graph file size mismatch")
            if verify:
                digest = hashlib.sha256()
                with open(graph_path, "rb") as gf:
                    while True:
                        if check_cancelled is not None:
                            check_cancelled()
                        chunk = gf.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                if digest.digest() != bound_sha256:
                    raise _SidecarStaleError("Graph file SHA-256 mismatch")

            if check_cancelled is not None:
                check_cancelled()

            # Read section directory.
            directory = []
            for i in range(_NUM_SECTIONS):
                offset = _EPI_HEADER_SIZE + i * _EPI_SECTION_DESCRIPTOR_SIZE
                desc = mm[offset : offset + _EPI_SECTION_DESCRIPTOR_SIZE]
                section_offset, section_size, section_count, dtype_code, sha = (
                    struct.unpack("<QQQB7x32s", desc)
                )
                directory.append(
                    (section_offset, section_size, section_count, dtype_code, sha)
                )

            expected_section_offset = header_size
            for section_index, (
                section_offset,
                section_size,
                _section_count,
                dtype_code,
                _sha,
            ) in enumerate(directory):
                if section_offset != expected_section_offset:
                    raise _SidecarCorruptError(
                        f"Section {section_index} offset mismatch"
                    )
                if dtype_code not in _DTYPE_CODE_TO_NAME:
                    raise _SidecarCorruptError(
                        f"Section {section_index} dtype code is unknown"
                    )
                expected_section_offset += cls._aligned_size(section_size)
            if expected_section_offset > file_size:
                raise _SidecarTruncatedError("Sidecar payload is truncated")
            if expected_section_offset != file_size:
                raise _SidecarCorruptError("Sidecar has trailing payload bytes")

            if check_cancelled is not None:
                check_cancelled()

            # Load numeric/string sections.
            def load_numeric(
                idx: int, expected_dtype: str, expected_count: int | None = None
            ) -> np.ndarray:
                offset, size, count, dtype_code, sha = directory[idx]
                if _DTYPE_CODE_TO_NAME[dtype_code] != expected_dtype:
                    raise _SidecarCorruptError(f"Section {idx} dtype mismatch")
                if expected_count is not None and count != expected_count:
                    raise _SidecarCorruptError(f"Section {idx} count mismatch")
                if size != count * np.dtype(expected_dtype).itemsize:
                    raise _SidecarCorruptError(f"Section {idx} size mismatch")
                if offset + size > len(mm):
                    raise _SidecarTruncatedError(f"Section {idx} extends past file")
                payload = memoryview(mm)[offset : offset + size]
                if verify and _sha256_buffer(payload, check_cancelled) != sha:
                    raise _SidecarCorruptError(f"Section {idx} SHA-256 mismatch")
                arr = np.frombuffer(
                    mm, dtype=np.dtype(expected_dtype), count=count, offset=offset
                )
                arr.flags.writeable = False
                return arr

            def load_bytes(idx: int) -> memoryview:
                offset, size, count, dtype_code, sha = directory[idx]
                if dtype_code != 0:
                    raise _SidecarCorruptError(f"Section {idx} is not raw bytes")
                if count != size:
                    raise _SidecarCorruptError(f"Section {idx} raw count mismatch")
                if offset + size > len(mm):
                    raise _SidecarTruncatedError(f"Section {idx} extends past file")
                payload = memoryview(mm)[offset : offset + size]
                if verify and _sha256_buffer(payload, check_cancelled) != sha:
                    raise _SidecarCorruptError(f"Section {idx} SHA-256 mismatch")
                return payload

            def validate_offsets(offsets: np.ndarray, data_size: int) -> None:
                if offsets.dtype.kind not in ("i", "u"):
                    raise _SidecarCorruptError("Offsets dtype must be integer")
                if len(offsets) == 0:
                    raise _SidecarCorruptError("Offsets array is empty")
                if offsets[0] != 0:
                    raise _SidecarCorruptError("Offsets must start at zero")
                if len(offsets) == 1:
                    if data_size != 0:
                        raise _SidecarCorruptError(
                            "Single offset must span empty data"
                        )
                    return
                if not all(
                    offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1)
                ):
                    raise _SidecarCorruptError("Offsets must be nondecreasing")
                if offsets[-1] > data_size:
                    raise _SidecarCorruptError("Final offset exceeds data size")
                if offsets[-1] != data_size:
                    raise _SidecarCorruptError("Final offset must match data size")

            def decode_strings(
                data: memoryview, offsets: np.ndarray
            ) -> tuple[str, ...]:
                validate_offsets(offsets, len(data))
                if len(offsets) == 1:
                    return ()
                strings = []
                for i in range(len(offsets) - 1):
                    start = int(offsets[i])
                    stop = int(offsets[i + 1])
                    try:
                        strings.append(data[start:stop].tobytes().decode("utf-8"))
                    except UnicodeDecodeError as exc:
                        raise _SidecarCorruptError("UTF-8 decode error") from exc
                return tuple(strings)

            edge_ranks = load_numeric(_SECTION_EDGE_RANKS, "<i8", edge_count)
            type_data = load_bytes(_SECTION_TYPE_NAMES_DATA)
            type_offsets = load_numeric(_SECTION_TYPE_NAMES_OFFSETS, "<i8")

            road_type_codes = load_numeric(_SECTION_TYPE_CODES, "<i4", edge_count)
            start_latitudes = load_numeric(_SECTION_START_LAT, "<f8", edge_count)
            start_longitudes = load_numeric(_SECTION_START_LON, "<f8", edge_count)
            end_latitudes = load_numeric(_SECTION_END_LAT, "<f8", edge_count)
            end_longitudes = load_numeric(_SECTION_END_LON, "<f8", edge_count)

            if bvh_node_count == 0 and edge_count > 0:
                raise _SidecarCorruptError("Missing BVH nodes")

            bvh_min_lat = load_numeric(_SECTION_BVH_MIN_LAT, "<f8", bvh_node_count)
            bvh_min_lon = load_numeric(_SECTION_BVH_MIN_LON, "<f8", bvh_node_count)
            bvh_max_lat = load_numeric(_SECTION_BVH_MAX_LAT, "<f8", bvh_node_count)
            bvh_max_lon = load_numeric(_SECTION_BVH_MAX_LON, "<f8", bvh_node_count)
            bvh_left = load_numeric(_SECTION_BVH_LEFT, "<i8", bvh_node_count)
            bvh_right = load_numeric(_SECTION_BVH_RIGHT, "<i8", bvh_node_count)
            bvh_start = load_numeric(_SECTION_BVH_START, "<i8", bvh_node_count)
            bvh_stop = load_numeric(_SECTION_BVH_STOP, "<i8", bvh_node_count)

            if check_cancelled is not None:
                check_cancelled()

            road_type_names = decode_strings(type_data, type_offsets)
            type_count = len(road_type_names)
            if type_count == 0 and edge_count > 0:
                raise _SidecarCorruptError("Missing road type names")
            if edge_count > 0 and (
                road_type_codes.min() < 0 or road_type_codes.max() >= type_count
            ):
                raise _SidecarCorruptError("Road type code out of range")

            if not (
                np.all(np.isfinite(start_latitudes))
                and np.all(np.isfinite(start_longitudes))
                and np.all(np.isfinite(end_latitudes))
                and np.all(np.isfinite(end_longitudes))
            ):
                raise _SidecarCorruptError(
                    "Indexed segment coordinates must be finite"
                )
            if edge_count > total_edge_count:
                raise _SidecarCorruptError(
                    "Indexed edge count exceeds total graph edge count"
                )
            if edge_count > 0:
                if edge_ranks.min() < 0 or edge_ranks.max() >= total_edge_count:
                    raise _SidecarCorruptError("Edge rank out of bounds")
                if len(np.unique(edge_ranks)) != edge_count:
                    raise _SidecarCorruptError("Edge ranks must be unique")

            if edge_count == 0:
                if bvh_node_count != 0 or root_node != -1:
                    raise _SidecarCorruptError(
                        "Empty index has inconsistent BVH metadata"
                    )
            else:
                if leaf_size != _EDGE_PROJECTION_BVH_LEAF_SIZE:
                    raise _SidecarVersionError(
                        f"Unsupported BVH leaf size: {leaf_size}"
                    )
                if root_node < 0 or root_node >= bvh_node_count:
                    raise _SidecarCorruptError("Invalid BVH root node index")
                if not (
                    np.all(np.isfinite(bvh_min_lat))
                    and np.all(np.isfinite(bvh_min_lon))
                    and np.all(np.isfinite(bvh_max_lat))
                    and np.all(np.isfinite(bvh_max_lon))
                    and np.all(bvh_min_lat <= bvh_max_lat)
                    and np.all(bvh_min_lon <= bvh_max_lon)
                ):
                    raise _SidecarCorruptError("Invalid BVH bounds")
                leaves = (bvh_left == -1) & (bvh_right == -1)
                internals = (
                    (bvh_left >= 0)
                    & (bvh_left < bvh_node_count)
                    & (bvh_right >= 0)
                    & (bvh_right < bvh_node_count)
                )
                if not np.all(leaves | internals):
                    raise _SidecarCorruptError("Invalid BVH child indices")
                if not np.all(
                    (bvh_start[leaves] >= 0)
                    & (bvh_stop[leaves] > bvh_start[leaves])
                    & (bvh_stop[leaves] <= edge_count)
                ):
                    raise _SidecarCorruptError("Invalid BVH leaf range")
                if not np.all(
                    (bvh_start[internals] == -1) & (bvh_stop[internals] == -1)
                ):
                    raise _SidecarCorruptError(
                        "Internal BVH nodes contain leaf ranges"
                    )

                visit_state = np.zeros(bvh_node_count, dtype=np.uint8)
                stack = [(int(root_node), False)]
                visits = 0
                while stack:
                    node, exiting = stack.pop()
                    if exiting:
                        visit_state[node] = 2
                        continue
                    if visit_state[node] != 0:
                        raise _SidecarCorruptError(
                            "BVH contains a cycle or shared child"
                        )
                    visit_state[node] = 1
                    visits += 1
                    if (
                        check_cancelled is not None
                        and visits
                        & (_CANCELLATION_CHECK_INTERVAL - 1)
                        == 0
                    ):
                        check_cancelled()
                    stack.append((node, True))
                    if internals[node]:
                        stack.append((int(bvh_right[node]), False))
                        stack.append((int(bvh_left[node]), False))
                if not np.all(visit_state == 2):
                    raise _SidecarCorruptError(
                        "BVH contains unreachable nodes"
                    )

                leaf_starts = bvh_start[leaves]
                leaf_stops = bvh_stop[leaves]
                leaf_order = np.argsort(leaf_starts)
                sorted_starts = leaf_starts[leaf_order]
                sorted_stops = leaf_stops[leaf_order]
                if (
                    sorted_starts[0] != 0
                    or sorted_stops[-1] != edge_count
                    or np.any(sorted_stops[:-1] != sorted_starts[1:])
                ):
                    raise _SidecarCorruptError(
                        "BVH leaves do not partition indexed edges"
                    )

                internal_nodes = np.flatnonzero(internals)
                left_nodes = bvh_left[internal_nodes]
                right_nodes = bvh_right[internal_nodes]
                if not (
                    np.all(
                        bvh_min_lat[internal_nodes]
                        <= bvh_min_lat[left_nodes]
                    )
                    and np.all(
                        bvh_min_lat[internal_nodes]
                        <= bvh_min_lat[right_nodes]
                    )
                    and np.all(
                        bvh_min_lon[internal_nodes]
                        <= bvh_min_lon[left_nodes]
                    )
                    and np.all(
                        bvh_min_lon[internal_nodes]
                        <= bvh_min_lon[right_nodes]
                    )
                    and np.all(
                        bvh_max_lat[internal_nodes]
                        >= bvh_max_lat[left_nodes]
                    )
                    and np.all(
                        bvh_max_lat[internal_nodes]
                        >= bvh_max_lat[right_nodes]
                    )
                    and np.all(
                        bvh_max_lon[internal_nodes]
                        >= bvh_max_lon[left_nodes]
                    )
                    and np.all(
                        bvh_max_lon[internal_nodes]
                        >= bvh_max_lon[right_nodes]
                    )
                ):
                    raise _SidecarCorruptError(
                        "BVH parent bounds do not contain children"
                    )
            if check_cancelled is not None:
                check_cancelled()

            return cls(
                edge_ranks=edge_ranks,
                total_edge_count=total_edge_count,
                road_type_names=road_type_names,
                road_type_codes=road_type_codes,
                start_latitudes=start_latitudes,
                start_longitudes=start_longitudes,
                end_latitudes=end_latitudes,
                end_longitudes=end_longitudes,
                bvh_min_lat=bvh_min_lat,
                bvh_min_lon=bvh_min_lon,
                bvh_max_lat=bvh_max_lat,
                bvh_max_lon=bvh_max_lon,
                bvh_left=bvh_left,
                bvh_right=bvh_right,
                bvh_start=bvh_start,
                bvh_stop=bvh_stop,
                leaf_size=leaf_size,
                root_node=root_node,
                mmap_ref=mm,
                file_ref=file_obj,
            )
        except Exception as exc:
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
            try:
                file_obj.close()
            except Exception:
                pass
            if isinstance(exc, _SidecarError):
                raise
            if isinstance(exc, (struct.error, ValueError, TypeError, KeyError, OSError)):
                raise _SidecarCorruptError(
                    "Sidecar payload could not be parsed"
                ) from exc
            raise
    @staticmethod
    def sidecar_path(graph_path: Path) -> Path:
        return Path(f"{graph_path}.edge_projection_index")
