from __future__ import annotations
import time

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import heapq
import math
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

try:
    from scipy.sparse import csr_matrix as _scipy_csr_matrix
    from scipy.sparse.csgraph import shortest_path as _scipy_shortest_path
except ImportError:  # pragma: no cover - exercised only without optional runtime
    _scipy_csr_matrix = None
    _scipy_shortest_path = None

from .cancellation import RoutingDeadline

from .cost import (
    CostWeights,
    HIGHWAY_ROAD_TYPES,
    RoutingPolicy,
    ScenicCostFunction,
    SCENIC_NORMALIZATION_VERSION,
    clamp_scenic_score,
    distance_weighted_scenic_score,
    evaluate_path,
    duration_component,
    is_better_path,
    is_highway_road_type,
    resolve_routing_policy,
)
from .graph import Edge, EdgeProjection, EndpointRoadGraph, Node, RoadGraph

_ORIGINAL_SCENIC_CALCULATE = ScenicCostFunction.calculate
_ORIGINAL_SCENIC_ROAD_TYPE_ADJUSTMENT = (
    ScenicCostFunction._road_type_adjustment
)
_SEARCH_DIAGNOSTIC_KEYS = (
    "time_limit_seconds",
    "labels_generated",
    "labels_expanded",
    "labels_pruned",
    "max_frontier_size",
    "remaining_frontier_size",
    "deadline_reached",
    "elapsed_ms",
    "mode",
)


_ACTIVE_ROUTING_DEADLINE: ContextVar[RoutingDeadline | None] = ContextVar(
    "active_routing_deadline", default=None
)


def _check_active_deadline() -> None:
    deadline = _ACTIVE_ROUTING_DEADLINE.get()
    if deadline is not None:
        deadline.check()


def _check_active_deadline_at(counter: int) -> None:
    if counter & 1023 == 0:
        _check_active_deadline()


@contextmanager
def _routing_deadline_scope(
    deadline: RoutingDeadline | None,
) -> Iterator[None]:
    active = _ACTIVE_ROUTING_DEADLINE.get()
    token = (
        _ACTIVE_ROUTING_DEADLINE.set(deadline)
        if deadline is not None and deadline is not active
        else None
    )
    try:
        _check_active_deadline()
        yield
        _check_active_deadline()
    finally:
        if token is not None:
            _ACTIVE_ROUTING_DEADLINE.reset(token)


def _exact_search_diagnostics() -> Dict[str, object]:
    return {
        "time_limit_seconds": 0.0,
        "labels_generated": 0,
        "labels_expanded": 0,
        "labels_pruned": 0,
        "max_frontier_size": 0,
        "remaining_frontier_size": 0,
        "deadline_reached": False,
        "elapsed_ms": 0.0,
        "mode": "exact",
    }


def _normalize_search_diagnostics(
    diagnostics: Optional[Dict[str, object]],
) -> Dict[str, object]:
    normalized = _exact_search_diagnostics()
    if diagnostics is not None:
        for key in _SEARCH_DIAGNOSTIC_KEYS:
            if key in diagnostics:
                normalized[key] = diagnostics[key]
    return normalized



@dataclass
class RouteSegment:
    start: Tuple[float, float]  # (lat, lon)
    end: Tuple[float, float]
    distance_km: float
    scenic_score: float
    road_name: Optional[str]
    road_type: str
    # The traversal identity is retained so callers can recompute metrics and
    # distinguish a reverse view from its canonical source edge.
    edge_id: str = ""
    direction: str = "forward"
    traversal_id: str = ""
    # Exact per-edge travel duration retained for independent recomputation.
    duration_minutes: float = 0.0


@dataclass
class Route:
    segments: List[RouteSegment]
    total_distance_km: float
    average_scenic_score: float
    estimated_duration_minutes: float
    waypoints: List[Tuple[float, float]]
    # Optimization diagnostics.  The first five fields above intentionally
    # retain their historical order so direct callers remain source-compatible.
    edge_ids: Tuple[str, ...] = ()
    traversal_ids: Tuple[str, ...] = ()
    raw_scenic_score: float = 0.0
    normalized_scenic_score: float = 0.0
    duration_utility: float = 0.0
    objective_value: float = 0.0
    fastest_duration_minutes: float = 0.0
    requested_max_detour_factor: float = 1.0
    applied_max_detour_factor: float = 1.0
    duration_cap_minutes: float = 0.0
    actual_duration_ratio: float = 1.0
    exact: bool = False
    exactness_status: str = "uncertified"
    optimality_gap: Optional[float] = None
    certified_upper_bound: Optional[float] = None
    highway_count: int = 0
    score_coverage: float = 1.0
    score_run: Tuple[Tuple[str, float], ...] = ()
    algorithm: str = "uncertified-production-search"
    zero_improvement_reason: Optional[str] = None
    no_route_reason: Optional[str] = None
    normalization_version: str = "linear-v1"
    search_diagnostics: Dict[str, object] = field(
        default_factory=_exact_search_diagnostics
    )

    @property
    def is_exact(self) -> bool:
        return self.exact

    @property
    def status(self) -> str:
        return self.exactness_status

    @property
    def objective(self) -> float:
        return self.objective_value

    @property
    def raw_scenic(self) -> float:
        return self.raw_scenic_score

    @property
    def normalized_scenic(self) -> float:
        return self.normalized_scenic_score
    @property
    def scenic_score_normalized(self) -> float:
        """Compatibility spelling used by service serializers."""
        return self.normalized_scenic_score

    @property
    def requested_cap(self) -> float:
        return self.requested_max_detour_factor

    @property
    def applied_cap(self) -> float:
        return self.applied_max_detour_factor

    @property
    def actual_ratio(self) -> float:
        return self.actual_duration_ratio

@dataclass
class _PathLabel:
    label_id: int
    node_id: str
    cumulative_distance_km: float
    cumulative_duration_minutes: float
    cumulative_cost: float
    predecessor_label_id: Optional[int]
    incoming_edge: Optional[Edge]


@dataclass
class _FrontierLabel:
    label_id: int
    node_id: str
    cumulative_duration_minutes: float
    cumulative_distance_km: float
    normalized_scenic_exposure: float
    predecessor_label_id: Optional[int]
    incoming_edge: Optional[Edge]
    visited_nodes: frozenset[str]
    edge_sequence: Tuple[str, ...]
    cumulative_highway_duration: float = 0.0
    root_traversal_id: str = ""

@dataclass
class _ReversePredecessorSnapshot:
    graph: RoadGraph
    stamp: object
    avoid_highways: bool
    predecessors: Dict[
        str, List[Tuple[str, float, float, Optional[float]]]
    ]

class _ZeroBounds(dict[str, float]):
    """Allocation-free view that disables lower-bound pruning."""

    def get(self, key: str, default: object = None) -> float:
        del key, default
        return 0.0


@dataclass(frozen=True)
class _CSRTopology:
    """Immutable compact directed traversal topology for one graph epoch."""

    graph: RoadGraph
    stamp: object
    # Kept as a compatibility marker; one all-traversal topology is shared by
    # both highway-filter modes, with filtered weights masked to infinity.
    avoid_highways: bool
    node_ids: Tuple[str, ...]
    node_index: Dict[str, int]
    indptr: np.ndarray
    indices: np.ndarray
    # Reverse rows are compact numeric views into the forward CSR.  A reverse
    # entry stores its predecessor node and the exact forward traversal
    # position, so reverse search never needs a Python incoming-edge map.
    reverse_indptr: np.ndarray
    reverse_indices: np.ndarray
    reverse_positions: np.ndarray
    edge_refs: Tuple[Tuple[str, bool], ...]
    distance_km: np.ndarray
    travel_time_minutes: np.ndarray
    scenic_score: np.ndarray
    highway_mask: np.ndarray
    scenic_byway_mask: np.ndarray

@dataclass(frozen=True)
class _CSRData:
    topology: _CSRTopology
    signature: Tuple[object, ...]
    matrix: object
    weights: np.ndarray

@dataclass
class _EndpointAccessRequest:
    """One projected endpoint request and its frozen route-local overlay."""

    start: Tuple[float, float]
    end: Tuple[float, float]
    start_projections: List[EdgeProjection]
    end_projections: List[EdgeProjection]
    overlay: EndpointRoadGraph
    start_node_id: str
    end_node_id: str
    start_accesses: Tuple[Tuple[int, int, Optional[Edge]], ...]
    end_accesses: Tuple[Tuple[int, int, Optional[Edge]], ...]
    direct_candidates: Tuple[Tuple[int, int, int, Edge], ...]
    graph_stamp: object

class ScenicRoutePlanner:
    _MINIMUM_COST_CACHE_CAPACITY = 8
    _FASTEST_PATH_CACHE_CAPACITY = 8
    _CSR_DATA_CACHE_CAPACITY = 8
    _REVERSE_INDEX_CACHE_CAPACITY = 2
    _ENDPOINT_OVERLAY_MAX_NODES = 2_000
    _REVERSE_PREPROCESS_EDGE_THRESHOLD = 256
    _LARGE_GRAPH_EDGE_THRESHOLD = 100_000
    _SHORT_ROUTE_CAP_KM = 5.0
    # Exhaustive simple-path enumeration is intentionally reserved for
    # bounded oracle-sized graphs. Production graphs use the complete
    # deadline-bounded multi-label frontier and report a certified
    # upper bound/gap when interrupted.
    _EXACT_ORACLE_MAX_NODES = 12
    _EXACT_ORACLE_MAX_EDGES = 48
    _PRODUCTION_FRONTIER_TIME_LIMIT_SECONDS = 4.0
    _MAX_FRONTIER_TIME_LIMIT_SECONDS = 60.0
    _CORRIDOR_WARM_START_ITERATIONS = 24
    _CORRIDOR_OVERLAP_PENALTY = 0.08
    _CORRIDOR_DURATION_COEFFICIENT_DIVISOR = 8.0
    _ELIGIBLE_REACHABILITY_CACHE_CAPACITY = 32
    _ELIGIBLE_REACHABILITY_SHARED_CACHE: OrderedDict[
        Tuple[RoadGraph, object, str, str, bool, float], bool
    ] = OrderedDict()
    _ELIGIBLE_REACHABILITY_SHARED_GRAPH: Optional[RoadGraph] = None
    _ELIGIBLE_REACHABILITY_SHARED_STAMP: object = None
    _REVERSE_INDEX_CACHE: OrderedDict[
        Tuple[int, object, bool, float],
        Tuple[RoadGraph, Dict[str, List[Tuple[str, str, bool]]]],
    ] = OrderedDict()
    # CSR state is shared by planner instances because the service constructs
    # a short-lived planner for each request.  The active topology reference
    # bounds this cache to one production graph and one mutation epoch.
    _CSR_TOPOLOGY_CACHE: Optional[_CSRTopology] = None
    _CSR_DATA_CACHE: OrderedDict[
        Tuple[int, object, Tuple[object, ...]], _CSRData
    ] = OrderedDict()
    _FASTEST_PATH_SHARED_CACHE: OrderedDict[
        Tuple[RoadGraph, object, str, str, bool, float], Tuple[Edge, ...]
    ] = OrderedDict()
    _FASTEST_PATH_SHARED_GRAPH: Optional[RoadGraph] = None
    _FASTEST_PATH_SHARED_STAMP: object = None

    @classmethod
    def clear_shared_caches(cls) -> None:
        """Release graph-backed process caches used across planner instances."""

        cls._ELIGIBLE_REACHABILITY_SHARED_CACHE.clear()
        cls._ELIGIBLE_REACHABILITY_SHARED_GRAPH = None
        cls._ELIGIBLE_REACHABILITY_SHARED_STAMP = None
        cls._REVERSE_INDEX_CACHE.clear()
        cls._CSR_TOPOLOGY_CACHE = None
        cls._CSR_DATA_CACHE.clear()
        cls._FASTEST_PATH_SHARED_CACHE.clear()
        cls._FASTEST_PATH_SHARED_GRAPH = None
        cls._FASTEST_PATH_SHARED_STAMP = None

    @classmethod
    def validate_frontier_time_limit_seconds(cls, value: object) -> float:
        try:
            limit = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "frontier_time_limit_seconds must be finite and between 0 and 60 seconds"
            ) from exc
        if (
            not math.isfinite(limit)
            or limit < 0.0
            or limit > cls._MAX_FRONTIER_TIME_LIMIT_SECONDS
        ):
            raise ValueError(
                "frontier_time_limit_seconds must be finite and between 0 and 60 seconds"
            )
        return limit


    def __init__(
        self,
        graph: Optional[RoadGraph] = None,
        cost_function: Optional[ScenicCostFunction] = None,
        frontier_time_limit_seconds: Optional[float] = None,
    ) -> None:
        self._frontier_time_limit_seconds = (
            self._PRODUCTION_FRONTIER_TIME_LIMIT_SECONDS
            if frontier_time_limit_seconds is None
            else self.validate_frontier_time_limit_seconds(
                frontier_time_limit_seconds
            )
        )
        self._monotonic = time.monotonic
        self._request_lock = RLock()
        self.graph = graph
        self.cost_function = cost_function or ScenicCostFunction()
        # Heuristic scans are graph-state dependent.  Keep the graph reference
        # alongside each stamp so identity checks remain safe if ``self.graph``
        # is replaced.  The ratio cache is an LRU so a caller varying scenic
        # weights cannot retain an unbounded number of signatures.
        self._geodesic_lower_bounds_cache: Optional[
            Tuple[RoadGraph, object, bool]
        ] = None
        self._minimum_cost_per_km_cache: OrderedDict[
            Tuple[int, object, Tuple[object, ...]], Tuple[RoadGraph, float]
        ] = OrderedDict()
        self._minimum_cost_per_km_cache_context: Optional[
            Tuple[RoadGraph, object]
        ] = None
        self._active_reverse_snapshot: Optional[
            _ReversePredecessorSnapshot
        ] = None

        self._fastest_path_cache = type(self)._FASTEST_PATH_SHARED_CACHE
        self._csr_topology_cache = type(self)._CSR_TOPOLOGY_CACHE
        self._csr_data_cache = type(self)._CSR_DATA_CACHE
        self._last_endpoint_access_request: Optional[
            _EndpointAccessRequest
        ] = None
    def _make_cost_function(self, scenic_weight: float) -> ScenicCostFunction:
        strict = bool(
            getattr(
                self.cost_function,
                "avoid_highways",
                getattr(self.cost_function, "strict_highways", False),
            )
        )
        preference = float(getattr(self.cost_function, "highway_preference", 0.0))
        return ScenicCostFunction(
            scenic_weight=scenic_weight,
            avoid_highways=strict,
            strict_highways=strict,
            highway_preference=preference,
            weights=self.cost_function.weights,
        )

    def _make_fastest_cost_function(self) -> ScenicCostFunction:
        # Fastest routing is always the true travel-duration objective.  Do
        # not inherit user scenic/custom weights, which could otherwise alter
        # the detour baseline.
        strict = bool(
            getattr(
                self.cost_function,
                "avoid_highways",
                getattr(self.cost_function, "strict_highways", False),
            )
        )
        return ScenicCostFunction(
            scenic_weight=0.0,
            avoid_highways=strict,
            strict_highways=strict,
            highway_preference=0.0,
            weights=CostWeights(
                travel_time=1.0,
                scenic_reward=0.0,
                highway_penalty=0.0,
                scenic_byway_bonus=0.0,
            ),
        )

    def prewarm_routing_cache(self) -> Dict[str, object]:
        """Build graph-level CSR state before the first endpoint request.

        No endpoints are needed: topology and edge attributes are enough to
        materialize exact fastest matrices for both highway-filter modes.
        Scenic objective data remains lazy and is vectorized on first use.
        """
        if self.graph is None:
            raise RuntimeError("Road graph not loaded")
        if _scipy_csr_matrix is None or _scipy_shortest_path is None:
            return {
                "available": False,
                "topology_built": False,
                "fastest_variants": 0,
            }
        topology = self._csr_topology()
        if topology is None:
            return {
                "available": True,
                "topology_built": False,
                "fastest_variants": 0,
            }
        variants = 0
        for avoid_highways in (False, True):
            cost = ScenicCostFunction(
                scenic_weight=0.0,
                avoid_highways=avoid_highways,
                weights=CostWeights(
                    travel_time=1.0,
                    scenic_reward=0.0,
                    highway_penalty=0.0,
                    scenic_byway_bonus=0.0,
                ),
            )
            signature = self._built_in_cost_signature(cost)
            if signature is None or self._csr_data(cost, signature) is None:
                continue
            variants += 1
        return {
            "available": True,
            "topology_built": True,
            "topology_traversals": len(topology.edge_refs),
            "fastest_variants": variants,
            "data_variants": len(type(self)._CSR_DATA_CACHE),
        }
    def _build_csr_topology(
        self,
        avoid_highways: bool = False,
        *,
        owner: Optional[RoadGraph] = None,
    ) -> Optional[_CSRTopology]:
        """Build one all-traversal CSR topology for the current graph epoch."""
        del avoid_highways  # Highway filtering is a data mask, not topology.
        graph = owner if owner is not None else self.graph
        if graph is None or _scipy_csr_matrix is None:
            return None
        stamp = graph._heuristic_cache_stamp()
        try:
            node_ids = tuple(graph.nodes)
            node_index = {node_id: index for index, node_id in enumerate(node_ids)}
            indices: List[int] = []
            edge_refs: List[Tuple[str, bool]] = []
            indptr = [0]
            distances: List[float] = []
            durations: List[float] = []
            scenic_scores: List[float] = []
            highway_mask: List[bool] = []
            scenic_byway_mask: List[bool] = []
            adjacency = graph.adjacency
            edges = graph.edges
            for source_index, source_id in enumerate(node_ids):
                _check_active_deadline_at(source_index)
                for edge_id, reverse in adjacency.get(source_id, ()):
                    edge = edges[edge_id]
                    target_id = (
                        edge.start_node_id if reverse else edge.end_node_id
                    )
                    target_index = node_index.get(target_id)
                    if target_index is None:
                        return None
                    try:
                        distance = float(edge.distance_km)
                    except (TypeError, ValueError, OverflowError):
                        distance = float("nan")
                    try:
                        speed = max(float(edge.speed_limit_kmh), 1.0)
                        duration = (distance / speed) * 60.0
                    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                        duration = float("nan")
                    try:
                        scenic_score = float(edge.scenic_score)
                    except (TypeError, ValueError, OverflowError):
                        scenic_score = float("nan")
                    road_type = str(edge.road_type).lower()
                    indices.append(target_index)
                    edge_refs.append((edge_id, bool(reverse)))
                    distances.append(distance)
                    durations.append(duration)
                    scenic_scores.append(scenic_score)
                    highway_mask.append(is_highway_road_type(road_type))
                    scenic_byway_mask.append(road_type == "scenic_byway")
                indptr.append(len(indices))
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        _check_active_deadline()
        active_graph = self.graph
        owner_is_active = active_graph is graph or (
            isinstance(active_graph, EndpointRoadGraph)
            and active_graph.base_graph is graph
        )
        if (
            not owner_is_active
            or graph._heuristic_cache_stamp() != stamp
        ):
            return None

        # Build incoming rows from the forward CSR in two numeric passes.  The
        # temporary cursor/count arrays are bounded by V; no Python
        # predecessor lists or edge/weight copies are retained.
        node_count = len(node_ids)
        forward_indptr = np.asarray(indptr, dtype=np.int64)
        forward_indices = np.asarray(indices, dtype=np.int32)
        reverse_indptr = np.zeros(node_count + 1, dtype=np.int64)
        for target_index in forward_indices:
            reverse_indptr[int(target_index) + 1] += 1
        np.cumsum(reverse_indptr, out=reverse_indptr)
        reverse_indices = np.empty(len(forward_indices), dtype=np.int32)
        reverse_positions = np.empty(len(forward_indices), dtype=np.int64)
        reverse_cursor = reverse_indptr[:-1].copy()
        for source_index in range(node_count):
            _check_active_deadline_at(source_index)
            row_start = int(forward_indptr[source_index])
            row_end = int(forward_indptr[source_index + 1])
            for position in range(row_start, row_end):
                target_index = int(forward_indices[position])
                reverse_position = int(reverse_cursor[target_index])
                reverse_indices[reverse_position] = source_index
                reverse_positions[reverse_position] = position
                reverse_cursor[target_index] += 1
        _check_active_deadline()
        active_graph = self.graph
        owner_is_active = active_graph is graph or (
            isinstance(active_graph, EndpointRoadGraph)
            and active_graph.base_graph is graph
        )
        if (
            not owner_is_active
            or graph._heuristic_cache_stamp() != stamp
        ):
            return None
        return _CSRTopology(
            graph=graph,
            stamp=stamp,
            avoid_highways=False,
            node_ids=node_ids,
            node_index=node_index,
            indptr=forward_indptr,
            indices=forward_indices,
            reverse_indptr=reverse_indptr,
            reverse_indices=reverse_indices,
            reverse_positions=reverse_positions,
            edge_refs=tuple(edge_refs),
            distance_km=np.asarray(distances, dtype=np.float64),
            travel_time_minutes=np.asarray(durations, dtype=np.float64),
            scenic_score=np.asarray(scenic_scores, dtype=np.float64),
            highway_mask=np.asarray(highway_mask, dtype=np.bool_),
            scenic_byway_mask=np.asarray(scenic_byway_mask, dtype=np.bool_),
        )

    @staticmethod
    def _compiled_distance_equal(first: float, second: float) -> bool:
        scale = max(1.0, abs(first), abs(second))
        return abs(first - second) <= max(
            1e-12, 8.0 * np.finfo(np.float64).eps * scale
        )

    def _compiled_shortest_duration_scenic_path(
        self,
        start: Node,
        goal: Node,
        scenic_cost: ScenicCostFunction,
    ) -> Optional[List[Edge]]:
        """Optimize scenic cost over the globally shortest-duration subgraph."""
        if _scipy_shortest_path is None or start.id == goal.id:
            return [] if start.id == goal.id else None
        fastest_cost = ScenicCostFunction(
            scenic_weight=0.0,
            avoid_highways=self._avoids_highways(scenic_cost),
            weights=CostWeights(
                travel_time=1.0,
                scenic_reward=0.0,
                highway_penalty=0.0,
                scenic_byway_bonus=0.0,
            ),
        )
        fastest_signature = self._built_in_cost_signature(fastest_cost)
        scenic_signature = self._built_in_cost_signature(scenic_cost)
        if fastest_signature is None or scenic_signature is None:
            return None
        duration_data = self._csr_data(fastest_cost, fastest_signature)
        scenic_data = self._csr_data(scenic_cost, scenic_signature)
        if duration_data is None or scenic_data is None:
            return None
        topology = duration_data.topology
        start_index = topology.node_index.get(start.id)
        goal_index = topology.node_index.get(goal.id)
        if start_index is None or goal_index is None:
            return None
        forward_distances = _scipy_shortest_path(
            duration_data.matrix,
            directed=True,
            indices=start_index,
            return_predecessors=False,
            unweighted=False,
            method="D",
        )
        total_duration = float(forward_distances[goal_index])
        if not math.isfinite(total_duration):
            return None
        row_lengths = np.diff(topology.indptr)
        source_indices = np.repeat(
            np.arange(len(topology.node_ids), dtype=np.int64),
            row_lengths,
        )
        target_indices = topology.indices
        with np.errstate(over="ignore", invalid="ignore"):
            candidate_prefixes = (
                forward_distances[source_indices] + duration_data.weights
            )
            prefix_differences = np.abs(
                candidate_prefixes - forward_distances[target_indices]
            )
        prefix_scales = np.maximum(
            1.0,
            np.maximum(
                np.abs(candidate_prefixes),
                np.abs(forward_distances[target_indices]),
            ),
        )
        shortest_edge_mask = (
            np.isfinite(candidate_prefixes)
            & np.isfinite(forward_distances[target_indices])
            & (forward_distances[target_indices] <= total_duration)
            & (
                prefix_differences
                <= np.maximum(
                    1e-12,
                    8.0 * np.finfo(np.float64).eps * prefix_scales,
                )
            )
        )
        scenic_weights = scenic_data.weights.copy()
        scenic_weights[~shortest_edge_mask] = np.inf
        scenic_matrix = _scipy_csr_matrix(
            (
                scenic_weights,
                topology.indices,
                topology.indptr,
            ),
            shape=(len(topology.node_ids), len(topology.node_ids)),
            copy=False,
        )
        scenic_distances, predecessors = _scipy_shortest_path(
            scenic_matrix,
            directed=True,
            indices=start_index,
            return_predecessors=True,
            unweighted=False,
            method="D",
        )
        if not math.isfinite(float(scenic_distances[goal_index])):
            return None
        path_reversed: List[Edge] = []
        current_index = goal_index
        visited = {current_index}
        while current_index != start_index:
            predecessor = int(predecessors[current_index])
            if (
                predecessor < 0
                or predecessor >= len(topology.node_ids)
                or predecessor in visited
            ):
                return None
            row_start = int(topology.indptr[predecessor])
            row_end = int(topology.indptr[predecessor + 1])
            best_position: Optional[int] = None
            best_residual = float("inf")
            predecessor_distance = float(scenic_distances[predecessor])
            current_distance = float(scenic_distances[current_index])
            for position in range(row_start, row_end):
                if (
                    int(topology.indices[position]) != current_index
                    or not shortest_edge_mask[position]
                ):
                    continue
                residual = abs(
                    predecessor_distance
                    + float(scenic_weights[position])
                    - current_distance
                )
                if residual < best_residual:
                    best_residual = residual
                    best_position = position
            if best_position is None or best_residual > max(
                1e-9, abs(current_distance) * 1e-10
            ):
                return None
            edge_id, reverse = topology.edge_refs[best_position]
            path_reversed.append(
                self._edge_from_reverse_index(edge_id, reverse)
            )
            current_index = predecessor
            visited.add(current_index)
        if (
            self.graph is not topology.graph
            or topology.graph._heuristic_cache_stamp() != topology.stamp
        ):
            return None
        path_reversed.reverse()
        return path_reversed
    def _compiled_weighted_path_with_positions(
        self,
        topology: _CSRTopology,
        start_index: int,
        goal_index: int,
        weights: np.ndarray,
    ) -> Optional[Tuple[List[Edge], np.ndarray]]:
        """Return a deterministic CSR path and its traversal positions."""
        if _scipy_shortest_path is None:
            return None
        matrix = _scipy_csr_matrix(
            (
                weights,
                topology.indices,
                topology.indptr,
            ),
            shape=(len(topology.node_ids), len(topology.node_ids)),
            copy=False,
        )
        distances, predecessors = _scipy_shortest_path(
            matrix,
            directed=True,
            indices=start_index,
            return_predecessors=True,
            unweighted=False,
            method="D",
        )
        if not math.isfinite(float(distances[goal_index])):
            return None
        path_reversed: List[Edge] = []
        positions_reversed: List[int] = []
        current_index = goal_index
        visited = {current_index}
        while current_index != start_index:
            predecessor = int(predecessors[current_index])
            if (
                predecessor < 0
                or predecessor >= len(topology.node_ids)
                or predecessor in visited
            ):
                return None
            row_start = int(topology.indptr[predecessor])
            row_end = int(topology.indptr[predecessor + 1])
            predecessor_distance = float(distances[predecessor])
            current_distance = float(distances[current_index])
            best_position: Optional[int] = None
            best_key: Tuple[float, str, bool] | None = None
            for position in range(row_start, row_end):
                if int(topology.indices[position]) != current_index:
                    continue
                edge_weight = float(weights[position])
                residual = abs(
                    predecessor_distance + edge_weight - current_distance
                )
                if residual > max(1e-9, abs(current_distance) * 1e-10):
                    continue
                edge_id, reverse = topology.edge_refs[position]
                key = (residual, str(edge_id), bool(reverse))
                if best_key is None or key < best_key:
                    best_key = key
                    best_position = position
            if best_position is None:
                return None
            edge_id, reverse = topology.edge_refs[best_position]
            path_reversed.append(
                self._edge_from_reverse_index(edge_id, reverse)
            )
            positions_reversed.append(best_position)
            current_index = predecessor
            visited.add(current_index)
        if (
            self.graph is not topology.graph
            or topology.graph._heuristic_cache_stamp() != topology.stamp
        ):
            return None
        path_reversed.reverse()
        positions_reversed.reverse()
        return path_reversed, np.asarray(positions_reversed, dtype=np.int64)

    def _compiled_weighted_path(
        self,
        topology: _CSRTopology,
        start_index: int,
        goal_index: int,
        weights: np.ndarray,
    ) -> Optional[List[Edge]]:
        """Return one deterministic nonnegative-weight CSR shortest path."""
        result = self._compiled_weighted_path_with_positions(
            topology,
            start_index,
            goal_index,
            weights,
        )
        return result[0] if result is not None else None

    def _compiled_shortest_duration_average_path(
        self,
        start: Node,
        goal: Node,
        avoid_highways: bool,
    ) -> Optional[List[Edge]]:
        """Maximize distance-weighted scenic score among fastest paths.

        Dinkelbach iterations solve the fractional distance-score objective on
        the shortest-duration DAG.  A duration shift makes every transformed
        edge weight nonnegative without changing the ranking because every
        feasible path has identical total duration.
        """
        if _scipy_shortest_path is None or start.id == goal.id:
            return [] if start.id == goal.id else None
        fastest_cost = ScenicCostFunction(
            scenic_weight=0.0,
            avoid_highways=avoid_highways,
            weights=CostWeights(
                travel_time=1.0,
                scenic_reward=0.0,
                highway_penalty=0.0,
                scenic_byway_bonus=0.0,
            ),
        )
        signature = self._built_in_cost_signature(fastest_cost)
        if signature is None:
            return None
        duration_data = self._csr_data(fastest_cost, signature)
        if duration_data is None:
            return None
        topology = duration_data.topology
        start_index = topology.node_index.get(start.id)
        goal_index = topology.node_index.get(goal.id)
        if start_index is None or goal_index is None:
            return None
        forward = _scipy_shortest_path(
            duration_data.matrix,
            directed=True,
            indices=start_index,
            return_predecessors=False,
            unweighted=False,
            method="D",
        )
        reverse = _scipy_shortest_path(
            duration_data.matrix.transpose(),
            directed=True,
            indices=goal_index,
            return_predecessors=False,
            unweighted=False,
            method="D",
        )
        total_duration = float(forward[goal_index])
        if not math.isfinite(total_duration):
            return None
        row_lengths = np.diff(topology.indptr)
        source_indices = np.repeat(
            np.arange(len(topology.node_ids), dtype=np.int64), row_lengths
        )
        target_indices = topology.indices
        prefix = forward[source_indices] + duration_data.weights
        suffix = reverse[target_indices]
        residual = np.abs(prefix + suffix - total_duration)
        scale = np.maximum(
            1.0,
            np.maximum(np.abs(prefix + suffix), abs(total_duration)),
        )
        shortest_mask = (
            np.isfinite(prefix)
            & np.isfinite(suffix)
            & (
                residual <= np.maximum(1e-12, 1e-12 * scale)
            )
        )
        if avoid_highways:
            shortest_mask &= ~topology.highway_mask
        if not shortest_mask.any():
            return None
        distance = topology.distance_km
        reward = distance * np.minimum(np.maximum(topology.scenic_score, 0.0), 10.0)
        duration = topology.travel_time_minutes
        ratio = 0.0
        best_path: Optional[List[Edge]] = None
        for _ in range(12):
            transformed = np.full_like(duration, np.inf, dtype=np.float64)
            active = shortest_mask & np.isfinite(distance) & (distance >= 0.0)
            valid_duration = active & np.isfinite(duration) & (duration > 0.0)
            shift = 0.0
            if valid_duration.any():
                shift = float(
                    np.max(
                        np.maximum(
                            0.0,
                            (reward[valid_duration] - ratio * distance[valid_duration])
                            / duration[valid_duration],
                        )
                    )
                )
            transformed[active] = (
                shift * duration[active]
                + ratio * distance[active]
                - reward[active]
            )
            transformed[active] = np.maximum(transformed[active], 0.0)
            candidate = self._compiled_weighted_path(
                topology, start_index, goal_index, transformed
            )
            if candidate is None:
                return best_path
            candidate_score = distance_weighted_scenic_score(candidate)
            if best_path is not None and abs(candidate_score - ratio) <= 1e-10:
                best_path = candidate
                break
            best_path = candidate
            ratio = candidate_score
        return best_path
    def _csr_topology(
        self,
        avoid_highways: bool = False,
        *,
        owner: Optional[RoadGraph] = None,
    ) -> Optional[_CSRTopology]:
        del avoid_highways
        active_graph = self.graph
        graph = owner
        if graph is None and active_graph is not None:
            if isinstance(active_graph, EndpointRoadGraph):
                base_graph = active_graph.base_graph
                try:
                    graph = (
                        base_graph
                        if len(base_graph.edges)
                        > self._LARGE_GRAPH_EDGE_THRESHOLD
                        else active_graph
                    )
                except (AttributeError, TypeError):
                    graph = active_graph
            else:
                graph = active_graph
        if graph is None or _scipy_csr_matrix is None:
            return None
        stamp = graph._heuristic_cache_stamp()
        cls = type(self)
        cached = cls._CSR_TOPOLOGY_CACHE
        if cached is not None:
            if cached.graph is graph and cached.stamp == stamp:
                if graph._heuristic_cache_stamp() == stamp:
                    self._csr_topology_cache = cached
                    self._csr_data_cache = cls._CSR_DATA_CACHE
                    return cached
            cls._CSR_TOPOLOGY_CACHE = None
            cls._CSR_DATA_CACHE.clear()
            self._csr_topology_cache = None
            self._csr_data_cache = cls._CSR_DATA_CACHE
        topology = self._build_csr_topology(owner=graph)
        if topology is not None:
            cls._CSR_TOPOLOGY_CACHE = topology
            self._csr_topology_cache = topology
        self._csr_data_cache = cls._CSR_DATA_CACHE
        return topology

    @staticmethod
    def _vectorized_builtin_weights(
        topology: _CSRTopology,
        signature: Tuple[object, ...],
    ) -> Optional[np.ndarray]:
        """Compute built-in scalar edge weights from cached primitive arrays."""
        strict_highways = bool(signature[1])
        highway_preference = float(signature[2])
        blocked = strict_highways & topology.highway_mask
        active = ~blocked
        valid = (
            np.isfinite(topology.distance_km)
            & (topology.distance_km >= 0.0)
            & np.isfinite(topology.travel_time_minutes)
            & (topology.travel_time_minutes >= 0.0)
        )
        if np.any(~valid & active):
            return None
        if (
            float(signature[0]) == 0.0
            and float(signature[2]) == 0.0
            and float(signature[3]) == 1.0
            and float(signature[4]) == 0.0
            and float(signature[5]) == 0.0
            and float(signature[6]) == 0.0
        ):
            weights = topology.travel_time_minutes.copy()
        else:
            def finite_nonnegative(value: object) -> float:
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError):
                    return 0.0
                return number if math.isfinite(number) and number >= 0.0 else 0.0

            scenic_weight = finite_nonnegative(signature[0])
            travel_weight = finite_nonnegative(signature[3])
            scenic_reward = finite_nonnegative(signature[4])
            highway_penalty = finite_nonnegative(signature[5])
            byway_bonus = min(finite_nonnegative(signature[6]), 0.5)
            duration = topology.travel_time_minutes
            scenic_score = np.minimum(
                np.maximum(np.nan_to_num(topology.scenic_score, nan=0.0), 0.0),
                10.0,
            )
            weighted_base = (
                (1.0 - scenic_weight) * duration * travel_weight
                + scenic_weight
                * duration
                * (10.0 - scenic_score)
                / 10.0
                * scenic_reward
            )
            base = np.maximum(weighted_base, 1e-6 * duration)
            base = np.where(
                topology.scenic_byway_mask,
                base * (1.0 - byway_bonus),
                base,
            )
            adjustment = np.where(
                (highway_preference > 0.0)
                & topology.highway_mask,
                duration * highway_preference,
                np.where(
                    strict_highways & topology.highway_mask,
                    duration * highway_penalty,
                    0.0,
                ),
            )
            with np.errstate(over="ignore", invalid="ignore"):
                raw = base + adjustment
            weights = np.where(
                np.isfinite(raw),
                np.maximum(raw, 1e-6),
                np.where(raw > 0.0, np.finfo(np.float64).max, 1e-6),
            )
        weights = np.asarray(weights, dtype=np.float64)
        weights[blocked] = np.inf
        return weights

    def _csr_data(
        self,
        cost_function: ScenicCostFunction,
        signature: Tuple[object, ...],
    ) -> Optional[_CSRData]:
        graph = self.graph
        if graph is None or _scipy_csr_matrix is None:
            return None
        topology = self._csr_topology(self._avoids_highways(cost_function))
        if topology is None:
            return None
        stamp = topology.stamp
        owner = topology.graph
        key = (id(owner), stamp, signature)
        data_cache = type(self)._CSR_DATA_CACHE
        self._csr_data_cache = data_cache
        cached = data_cache.get(key)
        if cached is not None:
            if (
                cached.topology is topology
                and owner._heuristic_cache_stamp() == stamp
                and self._built_in_cost_signature(cost_function) == signature
            ):
                data_cache.move_to_end(key)
                return cached
            data_cache.pop(key, None)

        weight_array = self._vectorized_builtin_weights(topology, signature)
        if weight_array is None:
            # Keep historical lazy validation: an invalid disconnected edge
            # must not make a valid route fail merely because CSR is present.
            return None
        matrix = _scipy_csr_matrix(
            (
                weight_array,
                topology.indices,
                topology.indptr,
            ),
            shape=(len(topology.node_ids), len(topology.node_ids)),
            copy=False,
        )
        result = _CSRData(
            topology=topology,
            signature=signature,
            matrix=matrix,
            weights=weight_array,
        )
        if (
            owner._heuristic_cache_stamp() != stamp
            or self._built_in_cost_signature(cost_function) != signature
        ):
            return None
        data_cache[key] = result
        data_cache.move_to_end(key)
        while len(data_cache) > self._CSR_DATA_CACHE_CAPACITY:
            data_cache.popitem(last=False)
        self._csr_data_cache = data_cache
        return result

    def _compiled_duration_tie_path(
        self,
        compiled: _CSRData,
        start_index: int,
        goal_index: int,
        distances: np.ndarray,
    ) -> Optional[List[Edge]]:
        """Choose the shortest-duration path among equal scalar optima.

        The historical Python search naturally prefers the fastest feasible
        incumbent when scalar costs tie.  Restricting a second Dijkstra to
        scalar-optimal transitions preserves that deterministic route
        semantics without perturbing the compiled objective weights.
        """
        topology = compiled.topology
        graph = self.graph
        if graph is None:
            return None
        node_count = len(topology.node_ids)
        durations = [float("inf")] * node_count
        parent_positions: List[Optional[int]] = [None] * node_count
        durations[start_index] = 0.0
        frontier: List[Tuple[float, int]] = [(0.0, start_index)]
        while frontier:
            current_duration, current_index = heapq.heappop(frontier)
            if current_duration != durations[current_index]:
                continue
            row_start = int(topology.indptr[current_index])
            row_end = int(topology.indptr[current_index + 1])
            current_distance = float(distances[current_index])
            for position in range(row_start, row_end):
                target_index = int(topology.indices[position])
                target_distance = float(distances[target_index])
                if not math.isfinite(target_distance):
                    continue
                residual = abs(
                    current_distance
                    + float(compiled.weights[position])
                    - target_distance
                )
                if residual > max(1e-12, abs(target_distance) * 1e-12):
                    continue
                edge_id, _reverse = topology.edge_refs[position]
                edge_duration = float(topology.travel_time_minutes[position])
                next_duration = current_duration + edge_duration
                if not math.isfinite(next_duration):
                    continue
                if next_duration >= durations[target_index]:
                    continue
                durations[target_index] = next_duration
                parent_positions[target_index] = position
                heapq.heappush(frontier, (next_duration, target_index))
        if not math.isfinite(durations[goal_index]):
            return None
        path_reversed: List[Edge] = []
        current_index = goal_index
        visited = {current_index}
        while current_index != start_index:
            position = parent_positions[current_index]
            if position is None:
                return None
            # CSR row boundaries recover the predecessor without storing a
            # per-traversal source array.
            predecessor_index = int(
                np.searchsorted(topology.indptr, position, side="right")
                - 1
            )
            edge_id, reverse = topology.edge_refs[position]
            edge = self._edge_from_reverse_index(edge_id, reverse)
            if predecessor_index in visited:
                return None
            path_reversed.append(edge)
            current_index = predecessor_index
            visited.add(current_index)
        if graph is not self.graph or graph._heuristic_cache_stamp() != topology.stamp:
            return None
        path_reversed.reverse()
        return path_reversed

    def _compiled_reachability_cache_key(
        self,
        start: Node,
        goal: Node,
        cost_function: ScenicCostFunction,
        stamp: object,
    ) -> Tuple[RoadGraph, object, str, str, bool, float]:
        graph = self.graph
        assert graph is not None
        return (
            graph,
            stamp,
            str(start.id),
            str(goal.id),
            bool(self._avoids_highways(cost_function)),
            float(getattr(cost_function, "highway_preference", 0.0)),
        )

    def _compiled_reachability_result(
        self,
        start: Node,
        goal: Node,
        cost_function: ScenicCostFunction,
    ) -> Optional[bool]:
        graph = self.graph
        if graph is None:
            return None
        stamp = graph._heuristic_cache_stamp()
        cls = type(self)
        if (
            cls._ELIGIBLE_REACHABILITY_SHARED_GRAPH is not graph
            or cls._ELIGIBLE_REACHABILITY_SHARED_STAMP != stamp
        ):
            cls._ELIGIBLE_REACHABILITY_SHARED_CACHE.clear()
            cls._ELIGIBLE_REACHABILITY_SHARED_GRAPH = graph
            cls._ELIGIBLE_REACHABILITY_SHARED_STAMP = stamp
        cache = cls._ELIGIBLE_REACHABILITY_SHARED_CACHE
        result = cache.get(
            self._compiled_reachability_cache_key(
                start, goal, cost_function, stamp
            )
        )
        if result is not None:
            cache.move_to_end(
                self._compiled_reachability_cache_key(
                    start, goal, cost_function, stamp
                )
            )
            return bool(result)
        return None

    def _record_compiled_reachability(
        self,
        start: Node,
        goal: Node,
        cost_function: ScenicCostFunction,
        stamp: object,
        reachable: bool,
    ) -> None:
        graph = self.graph
        if graph is None or graph._heuristic_cache_stamp() != stamp:
            return
        cls = type(self)
        if (
            cls._ELIGIBLE_REACHABILITY_SHARED_GRAPH is not graph
            or cls._ELIGIBLE_REACHABILITY_SHARED_STAMP != stamp
        ):
            cls._ELIGIBLE_REACHABILITY_SHARED_CACHE.clear()
            cls._ELIGIBLE_REACHABILITY_SHARED_GRAPH = graph
            cls._ELIGIBLE_REACHABILITY_SHARED_STAMP = stamp
        cache = cls._ELIGIBLE_REACHABILITY_SHARED_CACHE
        key = self._compiled_reachability_cache_key(
            start, goal, cost_function, stamp
        )
        cache[key] = bool(reachable)
        cache.move_to_end(key)
        while len(cache) > self._ELIGIBLE_REACHABILITY_CACHE_CAPACITY:
            cache.popitem(last=False)

    def _compiled_builtin_path(
        self,
        start: Node,
        goal: Node,
        cost_function: ScenicCostFunction,
    ) -> Optional[List[Edge]]:
        """Return an exact built-in scalar shortest path via SciPy Dijkstra."""
        if _scipy_shortest_path is None:
            return None
        signature = self._built_in_cost_signature(cost_function)
        if signature is None:
            return None
        graph = self.graph
        if graph is None or start.id == goal.id:
            return [] if start.id == goal.id else None
        compiled = self._csr_data(cost_function, signature)
        if compiled is None:
            return None
        topology = compiled.topology
        start_index = topology.node_index.get(start.id)
        goal_index = topology.node_index.get(goal.id)
        if start_index is None or goal_index is None:
            return None
        _check_active_deadline()
        distances, predecessors = _scipy_shortest_path(
            compiled.matrix,
            directed=True,
            indices=start_index,
            return_predecessors=True,
            unweighted=False,
            method="D",
        )
        _check_active_deadline()
        goal_distance = float(distances[goal_index])
        self._record_compiled_reachability(
            start,
            goal,
            cost_function,
            topology.stamp,
            math.isfinite(goal_distance),
        )
        if not math.isfinite(goal_distance):
            return None
        duration_tie_path = self._compiled_duration_tie_path(
            compiled,
            start_index,
            goal_index,
            distances,
        )
        if duration_tie_path is not None:
            return duration_tie_path

        path_reversed: List[Edge] = []
        current_index = goal_index
        visited = {current_index}
        while current_index != start_index:
            predecessor = int(predecessors[current_index])
            if (
                predecessor < 0
                or predecessor >= len(topology.node_ids)
                or predecessor in visited
            ):
                return None
            row_start = int(topology.indptr[predecessor])
            row_end = int(topology.indptr[predecessor + 1])
            best_position: Optional[int] = None
            best_residual = float("inf")
            predecessor_distance = float(distances[predecessor])
            current_distance = float(distances[current_index])
            for position in range(row_start, row_end):
                if int(topology.indices[position]) != current_index:
                    continue
                residual = abs(
                    predecessor_distance
                    + float(compiled.weights[position])
                    - current_distance
                )
                if residual < best_residual:
                    best_residual = residual
                    best_position = position
            tolerance = max(1e-9, abs(current_distance) * 1e-10)
            if best_position is None or best_residual > tolerance:
                return None
            edge_id, reverse = topology.edge_refs[best_position]
            path_reversed.append(
                self._edge_from_reverse_index(edge_id, reverse)
            )
            current_index = predecessor
            visited.add(current_index)
        if (
            graph is not self.graph
            or graph._heuristic_cache_stamp() != topology.stamp
            or self._built_in_cost_signature(cost_function) != signature
        ):
            return None
        path_reversed.reverse()
        return path_reversed

    def _avoids_highways(
        self, cost_function: Optional[ScenicCostFunction] = None
    ) -> bool:
        selected = cost_function if cost_function is not None else self.cost_function
        value = getattr(selected, "avoid_highways", None)
        if value is None:
            value = getattr(selected, "strict_highways", False)
        return bool(value)

    def _edge_is_eligible(
        self, edge: Edge, cost_function: Optional[ScenicCostFunction] = None
    ) -> bool:
        return not (
            self._avoids_highways(cost_function)
            and is_highway_road_type(edge.road_type)
        )
    def _cached_fastest_edges(
        self,
        start: Node,
        goal: Node,
        avoid_highways: bool,
        highway_preference: float = 0.0,
    ) -> Optional[List[Edge]]:
        graph = self.graph
        assert graph is not None
        stamp = graph._heuristic_cache_stamp()
        cls = type(self)
        if (
            cls._FASTEST_PATH_SHARED_GRAPH is not graph
            or cls._FASTEST_PATH_SHARED_STAMP != stamp
        ):
            cls._FASTEST_PATH_SHARED_CACHE.clear()
            cls._FASTEST_PATH_SHARED_GRAPH = graph
            cls._FASTEST_PATH_SHARED_STAMP = stamp
        self._fastest_path_cache = cls._FASTEST_PATH_SHARED_CACHE
        key = (
            graph,
            stamp,
            start.id,
            goal.id,
            bool(avoid_highways),
            float(highway_preference),
        )
        cached = self._fastest_path_cache.get(key)
        if cached is not None:
            if (
                graph is self.graph
                and graph._heuristic_cache_stamp() == stamp
            ):
                self._fastest_path_cache.move_to_end(key)
                return list(cached)
            self._fastest_path_cache.pop(key, None)

        shortest_edges = self._a_star(
            start,
            goal,
            cost_function=self._make_fastest_cost_function(),
        )
        if shortest_edges is None:
            return None
        if (
            graph is not self.graph
            or graph._heuristic_cache_stamp() != stamp
        ):
            raise RuntimeError("road graph changed during fastest-path search")
        self._fastest_path_cache[key] = tuple(shortest_edges)
        self._fastest_path_cache.move_to_end(key)
        while len(self._fastest_path_cache) > self._FASTEST_PATH_CACHE_CAPACITY:
            self._fastest_path_cache.popitem(last=False)
        return list(shortest_edges)

    def _iter_edges(self, node_id: str):
        """Iterate outgoing traversals without requiring list materialization."""
        graph = self.graph
        assert graph is not None
        iterator = getattr(graph, "iter_edges", None)
        if callable(iterator):
            return iterator(node_id)
        return graph.get_edges(node_id)

    def _cached_reverse_index(
        self, cost_function: ScenicCostFunction
    ) -> Optional[Dict[str, List[Tuple[str, str, bool]]]]:
        """Return a reusable incoming-traversal index for the active graph."""
        graph = self.graph
        assert graph is not None
        stamp = graph._heuristic_cache_stamp()
        avoid_highways = self._avoids_highways(cost_function)
        highway_preference = float(
            getattr(cost_function, "highway_preference", 0.0)
        )
        key = (id(graph), stamp, avoid_highways, highway_preference)
        cached = self._REVERSE_INDEX_CACHE.get(key)
        if cached is not None:
            cached_graph, predecessors = cached
            if (
                cached_graph is graph
                and graph is self.graph
                and graph._heuristic_cache_stamp() == stamp
            ):
                self._REVERSE_INDEX_CACHE.move_to_end(key)
                return predecessors
            self._REVERSE_INDEX_CACHE.pop(key, None)

        try:
            adjacency = graph.adjacency
            edges = graph.edges
        except AttributeError:
            return None
        predecessors: Dict[str, List[Tuple[str, str, bool]]] = {}
        try:
            for source_index, (source_id, traversals) in enumerate(
                adjacency.items()
            ):
                _check_active_deadline_at(source_index)
                for edge_id, reverse in traversals:
                    edge = edges[edge_id]
                    if avoid_highways and is_highway_road_type(
                        edge.road_type
                    ):
                        continue
                    target_id = (
                        edge.start_node_id if reverse else edge.end_node_id
                    )
                    predecessors.setdefault(target_id, []).append(
                        (str(source_id), str(edge_id), bool(reverse))
                    )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        _check_active_deadline()
        if (
            graph is not self.graph
            or graph._heuristic_cache_stamp() != stamp
        ):
            return None
        self._REVERSE_INDEX_CACHE[key] = (graph, predecessors)
        self._REVERSE_INDEX_CACHE.move_to_end(key)
        while len(self._REVERSE_INDEX_CACHE) > self._REVERSE_INDEX_CACHE_CAPACITY:
            self._REVERSE_INDEX_CACHE.popitem(last=False)
        return predecessors

    def _edge_from_reverse_index(
        self, edge_id: str, reverse: bool
    ) -> Edge:
        graph = self.graph
        assert graph is not None
        edge = graph.edges[edge_id]
        if not reverse:
            return edge
        reverse_edge = Edge(
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
        # Preserve source identity separately from the historical display ID;
        # canonical IDs may themselves end in ``::rev``.
        reverse_edge.traversal_id = f"reverse:{edge.id}"
        reverse_edge.canonical_edge_id = str(edge.id)
        reverse_edge.direction = "reverse"
        reverse_edge._canonical_edge_id = str(edge.id)
        reverse_edge._is_reverse_traversal = True
        return reverse_edge

    def _bidirectional_search_core(
        self,
        cost_function: ScenicCostFunction,
        forward_seeds: List[Tuple[str, float, Tuple[object, ...], Optional[Edge]]],
        reverse_seeds: List[Tuple[str, float, Tuple[object, ...], Optional[Edge]]],
        *,
        strict_mutation: bool = False,
    ) -> Optional[Tuple[List[Edge], Tuple[object, ...]]]:
        """Shared ranked bidirectional search over compact CSR and local edges."""
        graph = self.graph
        assert graph is not None
        active_stamp = graph._heuristic_cache_stamp()
        topology = self._csr_topology()
        if topology is None:
            return None
        signature = self._built_in_cost_signature(cost_function)
        if signature is None:
            return None
        data = self._csr_data(cost_function, signature)
        weights = (
            data.weights
            if data is not None and data.topology is topology
            else self._vectorized_builtin_weights(topology, signature)
        )
        base_index = topology.node_index
        topology_graph = topology.graph

        def local_cost(edge_id: str, reverse: bool) -> float:
            edge = graph.edges[edge_id]
            traversed = (
                self._edge_from_reverse_index(edge_id, True) if reverse else edge
            )
            return self._validated_nonnegative(
                cost_function.calculate(traversed), "edge calculated cost"
            )

        def base_cost(position: int) -> float:
            if weights is not None:
                return float(weights[position])
            edge_id, reverse = topology.edge_refs[position]
            edge = topology_graph.edges[edge_id]
            if self._avoids_highways(cost_function) and is_highway_road_type(
                edge.road_type
            ):
                return float("inf")
            return self._validated_nonnegative(
                cost_function.calculate(
                    self._edge_from_reverse_index(edge_id, reverse)
                    if reverse
                    else edge
                ),
                "edge calculated cost",
            )

        def forward_steps(
            node_id: str,
        ) -> Iterator[Tuple[str, float, Tuple[object, ...]]]:
            base_node_index = base_index.get(node_id)
            if base_node_index is not None:
                row_start = int(topology.indptr[base_node_index])
                row_end = int(topology.indptr[base_node_index + 1])
                for position in range(row_start, row_end):
                    edge_cost = base_cost(position)
                    if math.isfinite(edge_cost):
                        yield (
                            str(topology.node_ids[int(topology.indices[position])]),
                            edge_cost,
                            ("base", position),
                        )
            if isinstance(graph, EndpointRoadGraph):
                for edge_id, reverse in graph.iter_local_edges(node_id):
                    edge = graph.edges[edge_id]
                    if self._avoids_highways(cost_function) and is_highway_road_type(
                        edge.road_type
                    ):
                        continue
                    edge_cost = local_cost(str(edge_id), bool(reverse))
                    if math.isfinite(edge_cost):
                        yield (
                            str(edge.start_node_id if reverse else edge.end_node_id),
                            edge_cost,
                            ("local", str(edge_id), bool(reverse)),
                        )

        def reverse_steps(
            node_id: str,
        ) -> Iterator[Tuple[str, float, Tuple[object, ...]]]:
            base_node_index = base_index.get(node_id)
            if base_node_index is not None:
                row_start = int(topology.reverse_indptr[base_node_index])
                row_end = int(topology.reverse_indptr[base_node_index + 1])
                for reverse_position in range(row_start, row_end):
                    predecessor_index = int(
                        topology.reverse_indices[reverse_position]
                    )
                    position = int(topology.reverse_positions[reverse_position])
                    edge_cost = base_cost(position)
                    if math.isfinite(edge_cost):
                        yield (
                            str(topology.node_ids[predecessor_index]),
                            edge_cost,
                            ("base", position),
                        )
            if isinstance(graph, EndpointRoadGraph):
                for edge_id, reverse in graph.iter_local_predecessors(node_id):
                    edge = graph.edges[edge_id]
                    if self._avoids_highways(cost_function) and is_highway_road_type(
                        edge.road_type
                    ):
                        continue
                    edge_cost = local_cost(str(edge_id), bool(reverse))
                    if math.isfinite(edge_cost):
                        yield (
                            str(edge.end_node_id if reverse else edge.start_node_id),
                            edge_cost,
                            ("local", str(edge_id), bool(reverse)),
                        )

        def token_edge_id(token: Tuple[object, ...]) -> str:
            if token[0] == "base":
                return str(topology.edge_refs[int(str(token[1]))][0])
            return str(token[1])

        def edge_for_token(token: Tuple[object, ...]) -> Edge:
            if token[0] == "base":
                edge_id, reverse = topology.edge_refs[int(str(token[1]))]
            else:
                edge_id, reverse = str(token[1]), bool(token[2])
            return self._edge_from_reverse_index(edge_id, reverse)

        labels: Dict[str, Dict[int, Dict[str, Any]]] = {"f": {}, "r": {}}
        best_at_node: Dict[str, Dict[str, int]] = {"f": {}, "r": {}}
        frontiers: Dict[str, List[Tuple[float, Any, str, int]]] = {
            "f": [],
            "r": [],
        }
        next_ids = {"f": 0, "r": 0}

        def add_label(
            side: str,
            node_id: str,
            cost: float,
            key: Tuple[object, ...],
            parent: Optional[int],
            token: Optional[Tuple[object, ...]],
            seed_edge: Optional[Edge],
            rank: Tuple[object, ...],
        ) -> Optional[int]:
            old_id = best_at_node[side].get(node_id)
            if old_id is not None:
                old = labels[side][old_id]
                if (cost, key) >= (float(old["cost"]), old["key"]):
                    return None
            label_id = next_ids[side]
            next_ids[side] += 1
            labels[side][label_id] = {
                "node": node_id,
                "cost": cost,
                "key": key,
                "parent": parent,
                "token": token,
                "seed_edge": seed_edge,
                "rank": rank,
            }
            best_at_node[side][node_id] = label_id
            heapq.heappush(
                frontiers[side], (cost, key, node_id, label_id)
            )
            return label_id

        for node_id, cost, rank, seed_edge in forward_seeds:
            add_label("f", node_id, cost, (rank, ()), None, None, seed_edge, rank)
        for node_id, cost, rank, seed_edge in reverse_seeds:
            add_label("r", node_id, cost, (rank, ()), None, None, seed_edge, rank)

        best: Optional[Tuple[float, int, int, Tuple[object, ...]]] = None

        def consider(forward_id: int, reverse_id: int) -> None:
            nonlocal best
            forward = labels["f"][forward_id]
            reverse = labels["r"][reverse_id]
            middle_key = tuple(forward["key"][1]) + tuple(reverse["key"][1])
            forward_rank = tuple(forward["rank"])
            reverse_rank = tuple(reverse["rank"])
            rank_key = (
                (forward_rank[0], reverse_rank[0], forward_rank[1], reverse_rank[1], middle_key)
                if forward_rank and reverse_rank
                else (middle_key,)
            )
            candidate = (
                float(forward["cost"]) + float(reverse["cost"]),
                forward_id,
                reverse_id,
                rank_key,
            )
            if best is None or (candidate[0], candidate[3]) < (
                best[0],
                best[3],
            ):
                best = candidate

        expanded = 0
        while frontiers["f"] and frontiers["r"]:
            _check_active_deadline_at(expanded)
            expanded += 1
            if (
                best is not None
                and frontiers["f"][0][0] + frontiers["r"][0][0] > best[0]
            ):
                break
            expand_side = (
                "f"
                if frontiers["f"][0][0] <= frontiers["r"][0][0]
                else "r"
            )
            current_cost, current_key, current_node, label_id = heapq.heappop(
                frontiers[expand_side]
            )
            label = labels[expand_side].get(label_id)
            if (
                label is None
                or best_at_node[expand_side].get(current_node) != label_id
                or label["cost"] != current_cost
                or label["key"] != current_key
            ):
                continue
            other_id = best_at_node["r" if expand_side == "f" else "f"].get(
                current_node
            )
            if other_id is not None:
                if expand_side == "f":
                    consider(label_id, other_id)
                else:
                    consider(other_id, label_id)
            steps = forward_steps(current_node) if expand_side == "f" else reverse_steps(current_node)
            for neighbor, edge_cost, token in steps:
                next_cost = current_cost + edge_cost
                if not math.isfinite(next_cost):
                    raise ValueError(
                        "cumulative calculated cost must be finite and non-negative"
                    )
                rank = tuple(label["rank"])
                middle_key = tuple(label["key"][1])
                next_key = (
                    (rank, middle_key + (token_edge_id(token),))
                    if expand_side == "f"
                    else (rank, (token_edge_id(token),) + middle_key)
                )
                new_id = add_label(
                    expand_side,
                    neighbor,
                    next_cost,
                    next_key,
                    label_id,
                    token,
                    label["seed_edge"],
                    rank,
                )
                if new_id is not None:
                    other_id = best_at_node[
                        "r" if expand_side == "f" else "f"
                    ].get(neighbor)
                    if other_id is not None:
                        if expand_side == "f":
                            consider(new_id, other_id)
                        else:
                            consider(other_id, new_id)

        invalidated = (
            graph is not self.graph
            or graph._heuristic_cache_stamp() != active_stamp
            or topology.graph._heuristic_cache_stamp() != topology.stamp
            or self._built_in_cost_signature(cost_function) != signature
        )
        if invalidated:
            if strict_mutation:
                raise RuntimeError("road graph changed during fastest-path search")
            return None
        if best is None:
            return None

        _, forward_id, reverse_id, rank_key = best
        forward = labels["f"][forward_id]
        reverse = labels["r"][reverse_id]
        forward_tokens: List[Tuple[object, ...]] = []
        cursor = forward
        while cursor["parent"] is not None:
            forward_tokens.append(cursor["token"])
            cursor = labels["f"][int(cursor["parent"])]
        forward_tokens.reverse()
        path: List[Edge] = []
        if cursor["seed_edge"] is not None:
            path.append(cursor["seed_edge"])
        path.extend(edge_for_token(token) for token in forward_tokens)
        cursor = reverse
        while cursor["parent"] is not None:
            path.append(edge_for_token(cursor["token"]))
            cursor = labels["r"][int(cursor["parent"])]
        if cursor["seed_edge"] is not None:
            path.append(cursor["seed_edge"])
        return path, rank_key

    def _bidirectional_builtin_path(
        self,
        start: Node,
        goal: Node,
        cost_function: ScenicCostFunction,
    ) -> Optional[List[Edge]]:
        """Run target-bounded Dijkstra over compact base plus local edges."""
        if start.id == goal.id:
            return []
        result = self._bidirectional_search_core(
            cost_function,
            [(str(start.id), 0.0, (), None)],
            [(str(goal.id), 0.0, (), None)],
        )
        return None if result is None else result[0]


    @staticmethod
    def _duration_within_cap(duration: float, cap: float) -> bool:
        tolerance = 1e-12 * max(1.0, abs(duration), abs(cap))
        return duration <= cap + tolerance

    def _is_bounded_oracle_graph(self) -> bool:
        graph = self.graph
        if graph is None:
            return False
        try:
            return (
                len(graph.nodes) <= self._EXACT_ORACLE_MAX_NODES
                and len(graph.edges) <= self._EXACT_ORACLE_MAX_EDGES
            )
        except (AttributeError, TypeError):
            return False

    def _canonical_fastest_edges(
        self, start: Node, goal: Node, avoid_highways: bool
    ) -> Optional[List[Edge]]:
        """Find a duration-optimal path with a deterministic edge tie-break.

        The state key is ``(duration, edge-id sequence)``.  Positive-duration
        cycles cannot improve a label; zero-duration cycles are explicitly
        rejected by the simple-node check, so a returned path is always
        simple.
        """
        if start.id == goal.id:
            return []
        frontier: List[Tuple[float, Tuple[str, ...], str, Tuple[Edge, ...], Tuple[str, ...]]] = [
            (0.0, (), start.id, (), (start.id,))
        ]
        best: Dict[str, Tuple[float, Tuple[str, ...]]] = {start.id: (0.0, ())}
        expanded = 0
        while frontier:
            _check_active_deadline_at(expanded)
            expanded += 1
            duration, sequence, node_id, path, nodes = heapq.heappop(frontier)
            if best.get(node_id) != (duration, sequence):
                continue
            if node_id == goal.id:
                return list(path)
            for edge in sorted(
                self._iter_edges(node_id),
                key=lambda item: (str(item.id), str(item.end_node_id)),
            ):
                if avoid_highways and is_highway_road_type(edge.road_type):
                    continue
                neighbor_id = str(edge.end_node_id)
                if neighbor_id in nodes:
                    continue
                self._validated_nonnegative(edge.distance_km, "edge distance_km")
                edge_duration = self._edge_duration_minutes(edge)
                next_duration = duration + edge_duration
                if not math.isfinite(next_duration):
                    raise ValueError(
                        "cumulative duration must be finite and non-negative"
                    )
                next_sequence = sequence + (str(edge.id),)
                next_key = (next_duration, next_sequence)
                previous = best.get(neighbor_id)
                if previous is not None and next_key >= previous:
                    continue
                best[neighbor_id] = next_key
                heapq.heappush(
                    frontier,
                    (
                        next_duration,
                        next_sequence,
                        neighbor_id,
                        path + (edge,),
                        nodes + (neighbor_id,),
                    ),
                )
        return None

    def _enumerate_simple_optimum(
        self,
        start: Node,
        goal: Node,
        *,
        q: float,
        kappa: float,
        fastest_duration_minutes: float,
        duration_cap_minutes: float,
        policy: RoutingPolicy,
    ) -> Tuple[Optional[List[Edge]], object]:
        """Enumerate every feasible simple path on a bounded oracle graph."""
        best_edges: Optional[List[Edge]] = None
        best_evaluation = None
        visited = {start.id}

        def visit(node_id: str, path: Tuple[Edge, ...], duration: float) -> None:
            nonlocal best_edges, best_evaluation
            _check_active_deadline()
            if node_id == goal.id:
                evaluation = evaluate_path(
                    path,
                    q=q,
                    kappa=kappa,
                    fastest_duration_minutes=fastest_duration_minutes,
                    policy=policy,
                    check_cancelled=_check_active_deadline,
                )
                if (
                    best_evaluation is None
                    or is_better_path(evaluation, best_evaluation)
                ):
                    best_evaluation = evaluation
                    best_edges = list(path)
                return
            outgoing = sorted(
                self._iter_edges(node_id),
                key=lambda item: (str(item.id), str(item.end_node_id)),
            )
            for edge in outgoing:
                if policy.strict_highways and is_highway_road_type(edge.road_type):
                    continue
                neighbor_id = str(edge.end_node_id)
                if neighbor_id in visited:
                    continue
                self._validated_nonnegative(edge.distance_km, "edge distance_km")
                edge_duration = self._edge_duration_minutes(edge)
                next_duration = duration + edge_duration
                if not math.isfinite(next_duration):
                    raise ValueError(
                        "cumulative duration must be finite and non-negative"
                    )
                if not self._duration_within_cap(
                    next_duration, duration_cap_minutes
                ):
                    continue
                visited.add(neighbor_id)
                visit(neighbor_id, path + (edge,), next_duration)
                visited.remove(neighbor_id)

        visit(start.id, (), 0.0)
        return best_edges, best_evaluation

    def _frontier_reverse_duration_lower_bounds(
        self,
        goal: Node,
        avoid_highways: bool,
        duration_cap: float,
        deadline: Optional[float] = None,
    ) -> Dict[str, float]:
        def expired() -> bool:
            _check_active_deadline()
            return deadline is not None and self._monotonic() >= deadline

        def zero_bounds() -> Dict[str, float]:
            return _ZeroBounds()

        fastest = ScenicRoutePlanner._make_fastest_cost_function(self)
        fastest.avoid_highways = bool(avoid_highways)
        signature = self._built_in_cost_signature(fastest)
        graph = self.graph
        if graph is None or signature is None or expired():
            return zero_bounds()
        active_stamp = graph._heuristic_cache_stamp()
        topology = self._csr_topology()
        if topology is None:
            return zero_bounds()
        data = self._csr_data(fastest, signature)
        weights = (
            data.weights
            if data is not None and data.topology is topology
            else self._vectorized_builtin_weights(topology, signature)
        )
        if weights is None:
            return zero_bounds()
        goal_id = str(goal.id)
        bounds: Dict[str, float] = {goal_id: 0.0}
        queue: List[Tuple[float, str]] = [(0.0, goal_id)]
        while queue:
            if expired():
                return zero_bounds()
            distance, node_id = heapq.heappop(queue)
            if distance != bounds.get(node_id):
                continue
            base_node_index = topology.node_index.get(node_id)
            if base_node_index is not None:
                row_start = int(topology.reverse_indptr[base_node_index])
                row_end = int(topology.reverse_indptr[base_node_index + 1])
                for reverse_position in range(row_start, row_end):
                    predecessor_index = int(
                        topology.reverse_indices[reverse_position]
                    )
                    position = int(topology.reverse_positions[reverse_position])
                    edge_duration = float(
                        topology.travel_time_minutes[position]
                    )
                    if not math.isfinite(float(weights[position])):
                        continue
                    predecessor = str(topology.node_ids[predecessor_index])
                    next_distance = distance + edge_duration
                    if (
                        not math.isfinite(next_distance)
                        or not self._duration_within_cap(
                            next_distance, duration_cap
                        )
                    ):
                        continue
                    if next_distance < bounds.get(
                        predecessor, float("inf")
                    ):
                        bounds[predecessor] = next_distance
                        heapq.heappush(queue, (next_distance, predecessor))
            if isinstance(graph, EndpointRoadGraph):
                for edge_id, reverse in graph.iter_local_predecessors(node_id):
                    edge = graph.edges[edge_id]
                    if avoid_highways and is_highway_road_type(
                        edge.road_type
                    ):
                        continue
                    edge_duration = self._edge_duration_minutes(edge)
                    predecessor = str(
                        edge.end_node_id
                        if reverse
                        else edge.start_node_id
                    )
                    next_distance = distance + edge_duration
                    if (
                        not math.isfinite(next_distance)
                        or not self._duration_within_cap(
                            next_distance, duration_cap
                        )
                    ):
                        continue
                    if next_distance < bounds.get(
                        predecessor, float("inf")
                    ):
                        bounds[predecessor] = next_distance
                        heapq.heappush(queue, (next_distance, predecessor))
        if (
            graph is not self.graph
            or graph._heuristic_cache_stamp() != active_stamp
            or topology.graph._heuristic_cache_stamp() != topology.stamp
        ):
            return zero_bounds()
        return bounds

    @staticmethod
    def _frontier_label_dominates(
        first: _FrontierLabel, second: _FrontierLabel
    ) -> bool:
        """Safe dominance across all continuations.

        ``first`` must leave at least as many unvisited nodes available.
        Its lower duration, lower distance, and higher accumulated exposure
        preserve both duration utility and every possible continuation ratio.
        """
        if first.node_id != second.node_id:
            return False
        if not first.visited_nodes.issubset(second.visited_nodes):
            return False
        if first.cumulative_duration_minutes > second.cumulative_duration_minutes:
            return False
        if first.cumulative_distance_km > second.cumulative_distance_km:
            return False
        if (
            first.cumulative_highway_duration
            > second.cumulative_highway_duration
        ):
            return False
        if (
            first.normalized_scenic_exposure
            < second.normalized_scenic_exposure
        ):
            return False
        metrics_strict = (
            first.cumulative_duration_minutes
            < second.cumulative_duration_minutes
            or first.cumulative_distance_km < second.cumulative_distance_km
            or first.normalized_scenic_exposure
            > second.normalized_scenic_exposure
            or first.cumulative_highway_duration
            < second.cumulative_highway_duration
            or first.visited_nodes != second.visited_nodes
        )
        return metrics_strict and first.edge_sequence <= second.edge_sequence

    @staticmethod
    def _frontier_heap_key(
        label: _FrontierLabel,
        upper_bound: float,
    ) -> tuple[
        float, float, float, float, float, Tuple[str, ...], int
    ]:
        prefix_density = (
            label.normalized_scenic_exposure / label.cumulative_distance_km
            if label.cumulative_distance_km > 0.0
            else 0.0
        )
        return (
            -float(upper_bound),
            -float(prefix_density),
            -float(label.normalized_scenic_exposure),
            float(label.cumulative_duration_minutes),
            float(label.cumulative_distance_km),
            label.edge_sequence,
            label.label_id,
        )

    def _frontier_beam_warm_start_paths(
        self,
        start: Node,
        goal: Node,
        avoid_highways: bool,
        deadline: Optional[float],
        duration_cap_minutes: Optional[float],
        reverse_bounds: Optional[Dict[str, float]],
        max_distance_per_minute: Optional[float],
        zero_duration_distance: float,
    ) -> List[List[Edge]]:
        """Find a few scenic feasible paths with a bounded label beam."""
        if duration_cap_minutes is None or reverse_bounds is None:
            return []

        def expired() -> bool:
            _check_active_deadline()
            return deadline is not None and self._monotonic() >= deadline
        beam_width = 256
        per_root_limit = 16
        max_expanded = 16384
        labels: Dict[int, _FrontierLabel] = {
            0: _FrontierLabel(
                0,
                start.id,
                0.0,
                0.0,
                0.0,
                None,
                None,
                frozenset((start.id,)),
                (),
                root_traversal_id="",
            )
        }
        beam_ids = [0]
        goal_ids: List[int] = []
        next_label_id = 1
        expanded = 0
        maximum_suffix_rate = (
            max_distance_per_minute
            if max_distance_per_minute is not None
            and math.isfinite(max_distance_per_minute)
            and max_distance_per_minute >= 0.0
            else 0.0
        )

        def rank(label: _FrontierLabel) -> tuple[float, float, float, float, Tuple[str, ...]]:
            remaining_budget = max(
                0.0,
                duration_cap_minutes - label.cumulative_duration_minutes,
            )
            maximum_suffix_distance = (
                zero_duration_distance
                + maximum_suffix_rate * remaining_budget
            )
            denominator = (
                label.cumulative_distance_km + maximum_suffix_distance
            )
            upper = (
                (
                    label.normalized_scenic_exposure
                    + maximum_suffix_distance
                )
                / denominator
                if denominator > 0.0
                else 0.0
            )
            partial = (
                label.normalized_scenic_exposure
                / label.cumulative_distance_km
                if label.cumulative_distance_km > 0.0
                else 0.0
            )
            return (
                -float(min(1.0, upper)),
                -float(partial),
                -float(label.normalized_scenic_exposure),
                float(label.cumulative_duration_minutes),
                label.edge_sequence,
            )

        while beam_ids and expanded < max_expanded:
            if expired():
                break
            next_ids: List[int] = []
            for label_id in beam_ids:
                label = labels[label_id]
                if label.node_id == goal.id:
                    goal_ids.append(label_id)
                    continue
                expanded += 1
                for edge in sorted(
                    self._iter_edges(label.node_id),
                    key=lambda item: (
                        -float(clamp_scenic_score(item.scenic_score)),
                        float(self._edge_duration_minutes(item)),
                        str(getattr(item, "traversal_id", "") or item.id),
                        str(item.end_node_id),
                    ),
                ):
                    if expired():
                        break
                    if (
                        avoid_highways
                        and is_highway_road_type(edge.road_type)
                    ):
                        continue
                    neighbor = str(edge.end_node_id)
                    if neighbor in label.visited_nodes:
                        continue
                    distance = self._validated_nonnegative(
                        edge.distance_km, "edge distance_km"
                    )
                    edge_duration = self._edge_duration_minutes(edge)
                    next_distance = label.cumulative_distance_km + distance
                    next_duration = (
                        label.cumulative_duration_minutes + edge_duration
                    )
                    if not math.isfinite(next_distance) or not math.isfinite(
                        next_duration
                    ):
                        continue
                    remaining = reverse_bounds.get(neighbor, float("inf"))
                    if not math.isfinite(remaining) or not self._duration_within_cap(
                        next_duration + remaining,
                        duration_cap_minutes,
                    ):
                        continue
                    edge_token = str(
                        getattr(edge, "traversal_id", "") or edge.id
                    )
                    candidate = _FrontierLabel(
                        next_label_id,
                        neighbor,
                        next_duration,
                        next_distance,
                        label.normalized_scenic_exposure
                        + distance * clamp_scenic_score(edge.scenic_score) / 10.0,
                        label_id,
                        edge,
                        label.visited_nodes | frozenset((neighbor,)),
                        label.edge_sequence + (edge_token,),
                        root_traversal_id=(
                            label.root_traversal_id or edge_token
                        ),
                    )
                    labels[next_label_id] = candidate
                    next_ids.append(next_label_id)
                    next_label_id += 1
                if expired():
                    break
            if not next_ids:
                break
            ranked_ids = sorted(
                next_ids, key=lambda label_id: rank(labels[label_id])
            )
            beam_ids: List[int] = []
            root_counts: Dict[str, int] = {}
            selected_ids: set[int] = set()
            for candidate_id in ranked_ids:
                root = labels[candidate_id].root_traversal_id
                if root in root_counts:
                    continue
                root_counts[root] = 1
                beam_ids.append(candidate_id)
                selected_ids.add(candidate_id)
                if len(beam_ids) >= beam_width:
                    break
            if len(beam_ids) < beam_width:
                for candidate_id in ranked_ids:
                    if candidate_id in selected_ids:
                        continue
                    root = labels[candidate_id].root_traversal_id
                    root_count = root_counts.get(root, 0)
                    if root_count >= per_root_limit:
                        continue
                    root_counts[root] = root_count + 1
                    beam_ids.append(candidate_id)
                    selected_ids.add(candidate_id)
                    if len(beam_ids) >= beam_width:
                        break
            if len(beam_ids) < beam_width:
                for candidate_id in ranked_ids:
                    if candidate_id in selected_ids:
                        continue
                    beam_ids.append(candidate_id)
                    selected_ids.add(candidate_id)
                    if len(beam_ids) >= beam_width:
                        break

        paths: List[List[Edge]] = []
        seen: set[Tuple[str, ...]] = set()
        for label_id in sorted(goal_ids, key=lambda item: rank(labels[item])):
            path = self._frontier_path(labels, label_id)
            if not self._simple_edge_path(path):
                continue
            identity = tuple(
                str(getattr(edge, "traversal_id", "") or edge.id)
                for edge in path
            )
            if identity not in seen:
                seen.add(identity)
                paths.append(path)
                if len(paths) >= 32:
                    break
        return paths

    def _frontier_corridor_warm_start_paths(
        self,
        topology: _CSRTopology,
        start_index: int,
        goal_index: int,
        *,
        valid: np.ndarray,
        scenic_disutility: np.ndarray,
        duration_scale: float,
        minimum_duration_minutes: float,
        duration_cap_minutes: float,
        deadline: Optional[float],
    ) -> List[List[Edge]]:
        """Generate cap-aware alternatives by progressively penalizing overlap.

        Each scalar Dijkstra is only a candidate generator. Returned paths are
        still ranked by the canonical whole-route scenic objective, and the
        exact frontier remains responsible for certification.
        """
        if (
            not math.isfinite(minimum_duration_minutes)
            or not math.isfinite(duration_cap_minutes)
            or not self._duration_within_cap(
                minimum_duration_minutes,
                duration_cap_minutes,
            )
            or duration_cap_minutes
            <= minimum_duration_minutes
            + 1e-12 * max(1.0, abs(minimum_duration_minutes))
        ):
            return []

        def expired() -> bool:
            _check_active_deadline()
            return deadline is not None and self._monotonic() >= deadline

        scale = (
            float(duration_scale)
            if math.isfinite(duration_scale) and duration_scale >= 0.0
            else 0.0
        )
        coefficient = (
            scale / self._CORRIDOR_DURATION_COEFFICIENT_DIVISOR
        )
        with np.errstate(over="ignore", invalid="ignore"):
            raw_weights = (
                scenic_disutility
                + coefficient * topology.travel_time_minutes
            )
        weights = np.where(
            valid & np.isfinite(raw_weights) & (raw_weights >= 0.0),
            raw_weights,
            np.inf,
        ).astype(np.float64, copy=True)
        paths: List[List[Edge]] = []
        seen: set[Tuple[str, ...]] = set()

        for _ in range(self._CORRIDOR_WARM_START_ITERATIONS):
            if expired():
                break
            result = self._compiled_weighted_path_with_positions(
                topology,
                start_index,
                goal_index,
                weights,
            )
            if result is None:
                break
            path, positions = result
            if expired():
                break
            if self._simple_edge_path(path) and self._duration_within_cap(
                self._path_duration_minutes(path),
                duration_cap_minutes,
            ):
                identity = tuple(
                    str(getattr(edge, "traversal_id", "") or edge.id)
                    for edge in path
                )
                if identity not in seen:
                    seen.add(identity)
                    paths.append(path)
            if positions.size == 0:
                break
            penalty_distances = np.maximum(
                topology.distance_km[positions],
                1e-6,
            )
            weights[positions] += (
                self._CORRIDOR_OVERLAP_PENALTY * penalty_distances
            )
        return paths


    def _frontier_warm_start_paths(
        self,
        start: Node,
        goal: Node,
        avoid_highways: bool,
        deadline: Optional[float] = None,
        duration_cap_minutes: Optional[float] = None,
        reverse_bounds: Optional[Dict[str, float]] = None,
        max_distance_per_minute: Optional[float] = None,
        zero_duration_distance: float = 0.0,
    ) -> List[List[Edge]]:
        """Return feasible-quality incumbents without limiting the frontier.

        These scalar paths only warm-start branch-and-bound. They never remove
        labels, define the searched space, or support an exactness claim.
        """

        def expired() -> bool:
            _check_active_deadline()
            return deadline is not None and self._monotonic() >= deadline

        if expired():
            return []
        topology = self._csr_topology(False)
        if topology is None:
            return []
        start_index = topology.node_index.get(start.id)
        goal_index = topology.node_index.get(goal.id)
        if start_index is None or goal_index is None:
            return []
        normalized = np.clip(topology.scenic_score, 0.0, 10.0) / 10.0
        scenic_disutility = topology.distance_km * (1.0 - normalized)
        valid = (
            np.isfinite(scenic_disutility)
            & np.isfinite(topology.travel_time_minutes)
            & (scenic_disutility >= 0.0)
            & (topology.travel_time_minutes >= 0.0)
        )
        if avoid_highways:
            valid &= ~topology.highway_mask
        positive_duration = valid & (topology.travel_time_minutes > 0.0)
        if np.any(positive_duration):
            ratios = (
                scenic_disutility[positive_duration]
                / topology.travel_time_minutes[positive_duration]
            )
            scale = float(np.median(ratios[np.isfinite(ratios)]))
        else:
            scale = 0.0
        paths: List[List[Edge]] = []
        seen: set[Tuple[str, ...]] = set()

        def add_path(path: Optional[List[Edge]]) -> None:
            if path is None or not self._simple_edge_path(path):
                return
            identity = tuple(
                str(getattr(edge, "traversal_id", "") or edge.id)
                for edge in path
            )
            if identity not in seen:
                seen.add(identity)
                paths.append(path)

        for path in self._frontier_beam_warm_start_paths(
            start,
            goal,
            avoid_highways,
            deadline,
            duration_cap_minutes,
            reverse_bounds,
            max_distance_per_minute,
            zero_duration_distance,
        ):
            add_path(path)
        for coefficient in (0.0, 0.25 * scale, scale, 4.0 * scale):
            if expired():
                break
            weights = scenic_disutility + coefficient * topology.travel_time_minutes
            weights = np.where(valid, weights, np.inf)
            path = self._compiled_weighted_path(
                topology, start_index, goal_index, weights
            )
            if expired():
                break
            if path is None or not self._simple_edge_path(path):
                continue
            add_path(path)
        if duration_cap_minutes is not None and reverse_bounds is not None:
            minimum_duration = reverse_bounds.get(start.id, float("inf"))
            for path in self._frontier_corridor_warm_start_paths(
                topology,
                start_index,
                goal_index,
                valid=valid,
                scenic_disutility=scenic_disutility,
                duration_scale=scale,
                minimum_duration_minutes=minimum_duration,
                duration_cap_minutes=duration_cap_minutes,
                deadline=deadline,
            ):
                add_path(path)
        if duration_cap_minutes is not None and paths:
            def path_stats(path: List[Edge]) -> tuple[float, float, float]:
                distance = 0.0
                duration = 0.0
                exposure = 0.0
                for edge in path:
                    edge_distance = self._validated_nonnegative(
                        edge.distance_km,
                        "edge distance_km",
                    )
                    edge_duration = self._edge_duration_minutes(edge)
                    distance += edge_distance
                    duration += edge_duration
                    exposure += (
                        edge_distance
                        * clamp_scenic_score(edge.scenic_score)
                        / 10.0
                    )
                return distance, duration, exposure

            pair_candidates: List[tuple[float, List[Edge]]] = []
            pair_budget = 20_000
            for base_path in sorted(
                paths,
                key=lambda candidate: (
                    -(
                        path_stats(candidate)[2]
                        / path_stats(candidate)[0]
                        if path_stats(candidate)[0] > 0.0
                        else 0.0
                    ),
                    tuple(
                        str(getattr(edge, "traversal_id", "") or edge.id)
                        for edge in candidate
                    ),
                ),
            )[:4]:
                base_distance, base_duration, base_exposure = path_stats(
                    base_path
                )
                if base_duration > duration_cap_minutes + 1e-12:
                    continue
                indexed_edges = list(enumerate(base_path[:64]))
                alternatives: List[List[Edge]] = []
                for _, edge in indexed_edges:
                    current_identity = str(
                        getattr(edge, "traversal_id", "") or edge.id
                    )
                    alternatives.append(
                        [
                            alternative
                            for alternative in self._iter_edges(
                                edge.start_node_id
                            )
                            if str(alternative.end_node_id)
                            == str(edge.end_node_id)
                            and str(
                                getattr(
                                    alternative,
                                    "traversal_id",
                                    "",
                                )
                                or alternative.id
                            )
                            != current_identity
                            and not (
                                avoid_highways
                                and is_highway_road_type(
                                    alternative.road_type
                                )
                            )
                        ]
                    )
                for left_index, (_, left_edge) in enumerate(indexed_edges):
                    if not alternatives[left_index]:
                        continue
                    for right_index in range(
                        left_index + 1,
                        len(indexed_edges),
                    ):
                        if not alternatives[right_index]:
                            continue
                        for left_alternative in alternatives[left_index]:
                            for right_alternative in alternatives[right_index]:
                                pair_budget -= 1
                                if pair_budget < 0:
                                    break
                                if expired():
                                    break
                                candidate = list(base_path)
                                candidate[left_index] = left_alternative
                                candidate[right_index] = right_alternative
                                distance, duration, exposure = path_stats(
                                    candidate
                                )
                                if (
                                    duration
                                    > duration_cap_minutes + 1e-12
                                    or distance <= 0.0
                                ):
                                    continue
                                score = exposure / distance
                                base_score = (
                                    base_exposure / base_distance
                                    if base_distance > 0.0
                                    else 0.0
                                )
                                if score <= base_score + 1e-12:
                                    continue
                                pair_candidates.append((score, candidate))
                                pair_candidates.sort(
                                    key=lambda item: (
                                        -item[0],
                                        tuple(
                                            str(
                                                getattr(
                                                    edge,
                                                    "traversal_id",
                                                    "",
                                                )
                                                or edge.id
                                            )
                                            for edge in item[1]
                                        ),
                                    )
                                )
                                del pair_candidates[32:]
                            if pair_budget < 0 or expired():
                                break
                        if pair_budget < 0 or expired():
                            break
                    if pair_budget < 0 or expired():
                        break
                if pair_budget < 0 or expired():
                    break
            for _, candidate in pair_candidates:
                add_path(candidate)
        if duration_cap_minutes is not None and paths:
            for candidate in self._frontier_local_detour_paths(
                paths,
                avoid_highways=avoid_highways,
                duration_cap_minutes=duration_cap_minutes,
            ):
                add_path(candidate)
        return paths

    def _frontier_local_detour_paths(
        self,
        base_paths: List[List[Edge]],
        *,
        avoid_highways: bool,
        duration_cap_minutes: float,
        budget: int = 20_000,
    ) -> List[List[Edge]]:
        if budget <= 0 or not math.isfinite(duration_cap_minutes):
            return []

        def traversal_token(edge: Edge) -> str:
            return str(getattr(edge, "traversal_id", "") or edge.id)

        def edge_stats(edge: Edge) -> tuple[float, float, float]:
            distance = self._validated_nonnegative(
                edge.distance_km, "edge distance_km"
            )
            duration = self._edge_duration_minutes(edge)
            exposure = (
                distance * clamp_scenic_score(edge.scenic_score) / 10.0
            )
            if not all(
                math.isfinite(value) for value in (distance, duration, exposure)
            ):
                raise ValueError("local detour edge metrics must be finite")
            return distance, duration, exposure

        def path_stats(path: List[Edge]) -> tuple[float, float, float]:
            distance = 0.0
            duration = 0.0
            exposure = 0.0
            for edge in path:
                edge_distance, edge_duration, edge_exposure = edge_stats(edge)
                distance += edge_distance
                duration += edge_duration
                exposure += edge_exposure
            if not all(
                math.isfinite(value) for value in (distance, duration, exposure)
            ):
                raise ValueError("local detour path metrics must be finite")
            return distance, duration, exposure

        ranked_bases: List[
            tuple[
                float,
                float,
                Tuple[str, ...],
                List[Edge],
                float,
                float,
            ]
        ] = []
        for path in base_paths:
            try:
                distance, duration, exposure = path_stats(path)
            except (TypeError, ValueError, OverflowError):
                continue
            score = exposure / distance if distance > 0.0 else 0.0
            ranked_bases.append(
                (
                    score,
                    duration,
                    tuple(traversal_token(edge) for edge in path),
                    path,
                    distance,
                    exposure,
                )
            )
        ranked_bases.sort(key=lambda item: (-item[0], item[1], item[2]))

        def sorted_outgoing(node_id: str) -> List[Edge]:
            valid_edges: List[tuple[Edge, float, float]] = []
            for edge in self._iter_edges(node_id):
                try:
                    _, edge_duration, edge_score = (
                        self._validated_nonnegative(
                            edge.distance_km, "edge distance_km"
                        ),
                        self._edge_duration_minutes(edge),
                        clamp_scenic_score(edge.scenic_score),
                    )
                except (TypeError, ValueError, OverflowError):
                    continue
                if avoid_highways and is_highway_road_type(edge.road_type):
                    continue
                valid_edges.append((edge, edge_duration, edge_score))
            valid_edges.sort(
                key=lambda item: (
                    -item[2],
                    item[1],
                    traversal_token(item[0]),
                    str(item[0].end_node_id),
                )
            )
            return [edge for edge, _, _ in valid_edges]

        examined = 0
        retained: List[
            tuple[float, float, Tuple[str, ...], List[Edge]]
        ] = []
        retained_identities: set[Tuple[str, ...]] = set()

        for (
            base_score,
            base_duration,
            _,
            base_path,
            base_distance,
            base_exposure,
        ) in ranked_bases[:4]:
            if examined >= budget:
                break
            base_node_ids = {
                str(edge.start_node_id) for edge in base_path
            }
            if base_path:
                base_node_ids.add(str(base_path[-1].end_node_id))
            for edge_index, edge in enumerate(base_path[:64]):
                if examined >= budget:
                    break
                start_node_id = str(edge.start_node_id)
                end_node_id = str(edge.end_node_id)
                forbidden_nodes = base_node_ids - {
                    start_node_id,
                    end_node_id,
                }
                replacements: List[List[Edge]] = []

                def collect(
                    node_id: str,
                    replacement: List[Edge],
                    visited_nodes: set[str],
                ) -> None:
                    nonlocal examined
                    depth = len(replacement)
                    if node_id == end_node_id:
                        if depth in (2, 3):
                            replacement_nodes = [
                                start_node_id
                            ] + [
                                str(item.end_node_id) for item in replacement
                            ]
                            if not any(
                                node in forbidden_nodes
                                for node in replacement_nodes[1:-1]
                            ):
                                replacements.append(list(replacement))
                        return
                    if examined >= budget:
                        return
                    if depth >= 3:
                        return
                    for next_edge in sorted_outgoing(node_id):
                        if examined >= budget:
                            return
                        examined += 1
                        next_node_id = str(next_edge.end_node_id)
                        if next_node_id in visited_nodes:
                            continue
                        collect(
                            next_node_id,
                            replacement + [next_edge],
                            visited_nodes | {next_node_id},
                        )
                        if examined >= budget:
                            return

                collect(start_node_id, [], {start_node_id})
                try:
                    (
                        replaced_distance,
                        replaced_duration,
                        replaced_exposure,
                    ) = edge_stats(edge)
                except (TypeError, ValueError, OverflowError):
                    continue
                for replacement in replacements:
                    try:
                        (
                            replacement_distance,
                            replacement_duration,
                            replacement_exposure,
                        ) = path_stats(replacement)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    distance = (
                        base_distance
                        - replaced_distance
                        + replacement_distance
                    )
                    duration = (
                        base_duration
                        - replaced_duration
                        + replacement_duration
                    )
                    exposure = (
                        base_exposure
                        - replaced_exposure
                        + replacement_exposure
                    )
                    if (
                        not all(
                            math.isfinite(value)
                            for value in (distance, duration, exposure)
                        )
                        or
                        distance <= 0.0
                        or duration > duration_cap_minutes + 1e-12
                    ):
                        continue
                    score = exposure / distance
                    if score <= base_score + 1e-12:
                        continue
                    candidate = (
                        base_path[:edge_index]
                        + replacement
                        + base_path[edge_index + 1 :]
                    )
                    if not self._simple_edge_path(candidate):
                        continue
                    identity = tuple(
                        traversal_token(item) for item in candidate
                    )
                    if identity in retained_identities:
                        continue
                    retained_identities.add(identity)
                    retained.append((score, duration, identity, candidate))
                    retained.sort(
                        key=lambda item: (-item[0], item[1], item[2])
                    )
                    del retained[32:]
                    retained_identities = {
                        item[2] for item in retained
                    }

        return [item[3] for item in retained]

    @staticmethod
    def _frontier_path(
        labels: Dict[int, _FrontierLabel], label_id: int
    ) -> List[Edge]:
        path: List[Edge] = []
        current = labels[label_id]
        while current.incoming_edge is not None:
            path.append(current.incoming_edge)
            predecessor = current.predecessor_label_id
            assert predecessor is not None
            current = labels[predecessor]
        path.reverse()
        return path

    def _production_frontier_search(
        self,
        start: Node,
        goal: Node,
        *,
        q: float,
        kappa: float,
        fastest_duration_minutes: float,
        duration_cap_minutes: float,
        policy: RoutingPolicy,
        fastest_edges: List[Edge],
        fastest_evaluation: object,
        started_at: Optional[float] = None,
    ) -> Tuple[List[Edge], object, bool, float, Dict[str, object]]:
        del started_at
        _check_active_deadline()
        try:
            time_limit_seconds = float(self._frontier_time_limit_seconds)
        except (TypeError, ValueError, OverflowError):
            time_limit_seconds = float(self._PRODUCTION_FRONTIER_TIME_LIMIT_SECONDS)
        time_limit_seconds = max(0.0, time_limit_seconds)
        search_stamp = self.graph._heuristic_cache_stamp()
        reverse_bounds = self._frontier_reverse_duration_lower_bounds(
            goal, policy.strict_highways, duration_cap_minutes, deadline=None
        )
        if self.graph._heuristic_cache_stamp() != search_stamp:
            raise RuntimeError(
                "road graph changed during frontier-bound construction"
            )
        labels: Dict[int, _FrontierLabel] = {
            0: _FrontierLabel(
                0,
                start.id,
                0.0,
                0.0,
                0.0,
                None,
                None,
                frozenset((start.id,)),
                (),
                root_traversal_id="",
            )
        }
        active: set[int] = {0}
        labels_at_node: Dict[str, set[int]] = {start.id: {0}}
        frontier: List[
            Tuple[float, float, float, float, float, Tuple[str, ...], int]
        ] = []
        labels_generated = 1
        labels_expanded = 0
        labels_pruned = 0
        max_frontier_size = 0

        max_distance_per_minute: Optional[float] = None
        zero_duration_distance = 0.0
        topology = self._csr_topology(False)
        if (
            topology is not None
            and topology.graph is self.graph
            and topology.stamp == search_stamp
        ):
            eligible = (
                ~topology.highway_mask
                if policy.strict_highways
                else np.ones(len(topology.edge_refs), dtype=np.bool_)
            )
            distances = topology.distance_km[eligible]
            durations = topology.travel_time_minutes[eligible]
            finite = (
                np.isfinite(distances)
                & np.isfinite(durations)
                & (distances >= 0.0)
                & (durations >= 0.0)
            )
            distances = distances[finite]
            durations = durations[finite]
            zero_duration_distance = float(
                np.sum(distances[durations == 0.0], dtype=np.float64)
            )
            positive = durations > 0.0
            if np.any(positive):
                max_distance_per_minute = float(
                    np.max(distances[positive] / durations[positive])
                )

        def upper_bound(label: _FrontierLabel) -> float:
            remaining = reverse_bounds.get(label.node_id, float("inf"))
            if not math.isfinite(remaining):
                return float("-inf")
            if not self._duration_within_cap(
                label.cumulative_duration_minutes + remaining,
                duration_cap_minutes,
            ):
                return float("-inf")
            rounding_margin = 1e-12 * max(1.0, abs(remaining))
            admissible_remaining = max(0.0, remaining - rounding_margin)
            minimum_final_duration = (
                label.cumulative_duration_minutes + admissible_remaining
            )
            duration_ub = duration_component(
                minimum_final_duration,
                fastest_duration_minutes,
                kappa,
            )
            scenic_ub = 1.0
            if max_distance_per_minute is not None:
                remaining_budget = max(
                    0.0,
                    duration_cap_minutes - label.cumulative_duration_minutes,
                )
                maximum_suffix_distance = (
                    zero_duration_distance
                    + max_distance_per_minute * remaining_budget
                )
                denominator = (
                    label.cumulative_distance_km + maximum_suffix_distance
                )
                if denominator > 0.0:
                    scenic_ub = min(
                        1.0,
                        (
                            label.normalized_scenic_exposure
                            + maximum_suffix_distance
                        )
                        / denominator,
                    )
            if policy.scenic_priority:
                return float(scenic_ub)
            return float((1.0 - q) * duration_ub + q * scenic_ub)

        heapq.heappush(
            frontier, self._frontier_heap_key(labels[0], upper_bound(labels[0]))
        )
        max_frontier_size = max(max_frontier_size, len(frontier))
        next_label_id = 1
        incumbent_edges = list(fastest_edges)
        incumbent_evaluation = fastest_evaluation
        for warm_path in self._frontier_warm_start_paths(
            start,
            goal,
            policy.strict_highways,
            None,
            duration_cap_minutes,
            reverse_bounds,
            max_distance_per_minute,
            zero_duration_distance,
        ):
            if not self._duration_within_cap(
                self._path_duration_minutes(warm_path),
                duration_cap_minutes,
            ):
                continue
            warm_evaluation = evaluate_path(
                warm_path,
                q=q,
                kappa=kappa,
                fastest_duration_minutes=fastest_duration_minutes,
                policy=policy,
                check_cancelled=_check_active_deadline,
            )
            if is_better_path(warm_evaluation, incumbent_evaluation):
                incumbent_edges = warm_path
                incumbent_evaluation = warm_evaluation
        frontier_started = self._monotonic()
        deadline = frontier_started + time_limit_seconds
        last_observed_at = frontier_started
        timed_out = False

        def deadline_reached() -> bool:
            nonlocal last_observed_at
            _check_active_deadline()
            try:
                now = self._monotonic()
            except (TypeError, ValueError, OverflowError):
                return False
            last_observed_at = now
            return now >= deadline

        def epoch_changed() -> bool:
            return self.graph._heuristic_cache_stamp() != search_stamp

        while frontier:
            if epoch_changed():
                raise RuntimeError(
                    "road graph changed during frontier search"
                )
            if deadline_reached():
                timed_out = True
                break
            label_id = heapq.heappop(frontier)[-1]
            if label_id not in active:
                continue
            label = labels[label_id]
            label_ub = upper_bound(label)
            if label_ub == float("-inf"):
                active.remove(label_id)
                labels_pruned += 1
                continue
            incumbent_objective = float(
                getattr(incumbent_evaluation, "objective")
            )
            if label_ub < incumbent_objective - 1e-12:
                active.remove(label_id)
                labels_pruned += 1
                continue
            labels_expanded += 1
            if label.node_id == goal.id:
                candidate = self._frontier_path(labels, label_id)
                evaluation = evaluate_path(
                    candidate,
                    q=q,
                    kappa=kappa,
                    fastest_duration_minutes=fastest_duration_minutes,
                    policy=policy,
                    check_cancelled=_check_active_deadline,
                )
                if is_better_path(evaluation, incumbent_evaluation):
                    incumbent_edges = candidate
                    incumbent_evaluation = evaluation
                active.remove(label_id)
                continue
            for edge in sorted(
                self._iter_edges(label.node_id),
                key=lambda item: (
                    str(getattr(item, "traversal_id", "") or item.id),
                    str(item.end_node_id),
                ),
            ):
                if epoch_changed():
                    raise RuntimeError(
                        "road graph changed during frontier search"
                    )
                if deadline_reached():
                    timed_out = True
                    break
                if policy.strict_highways and is_highway_road_type(edge.road_type):
                    continue
                neighbor = str(edge.end_node_id)
                if neighbor in label.visited_nodes:
                    continue
                distance = self._validated_nonnegative(
                    edge.distance_km, "edge distance_km"
                )
                edge_duration = self._edge_duration_minutes(edge)
                next_distance = label.cumulative_distance_km + distance
                next_duration = label.cumulative_duration_minutes + edge_duration
                if not math.isfinite(next_distance) or not math.isfinite(
                    next_duration
                ):
                    raise ValueError(
                        "cumulative traversed resource must be finite and non-negative"
                    )
                remaining = reverse_bounds.get(neighbor, float("inf"))
                if not math.isfinite(remaining) or not self._duration_within_cap(
                    next_duration + remaining, duration_cap_minutes
                ):
                    continue
                exposure = label.normalized_scenic_exposure + (
                    distance * clamp_scenic_score(edge.scenic_score) / 10.0
                )
                if not math.isfinite(exposure):
                    raise ValueError(
                        "cumulative scenic exposure must be finite and non-negative"
                    )
                highway_duration = label.cumulative_highway_duration + (
                    edge_duration
                    if is_highway_road_type(edge.road_type)
                    else 0.0
                )
                if not math.isfinite(highway_duration):
                    raise ValueError(
                        "cumulative highway duration must be finite and non-negative"
                    )
                edge_token = str(
                    getattr(edge, "traversal_id", "") or edge.id
                )
                candidate = _FrontierLabel(
                    next_label_id,
                    neighbor,
                    next_duration,
                    next_distance,
                    exposure,
                    label_id,
                    edge,
                    label.visited_nodes | frozenset((neighbor,)),
                    label.edge_sequence + (edge_token,),
                    cumulative_highway_duration=highway_duration,
                    root_traversal_id=(
                        label.root_traversal_id or edge_token
                    ),
                )
                existing_ids = labels_at_node.get(neighbor, set())
                if any(
                    existing_id in active
                    and self._frontier_label_dominates(
                        labels[existing_id], candidate
                    )
                    for existing_id in existing_ids
                ):
                    labels_generated += 1
                    labels_pruned += 1
                    continue
                dominated_ids = [
                    existing_id
                    for existing_id in existing_ids
                    if existing_id in active
                    and self._frontier_label_dominates(
                        candidate, labels[existing_id]
                    )
                ]
                labels_pruned += len(dominated_ids)
                for dominated_id in dominated_ids:
                    active.discard(dominated_id)
                    existing_ids.discard(dominated_id)
                labels[next_label_id] = candidate
                labels_generated += 1
                active.add(next_label_id)
                labels_at_node.setdefault(neighbor, set()).add(next_label_id)
                heapq.heappush(
                    frontier,
                    self._frontier_heap_key(candidate, upper_bound(candidate)),
                )
                max_frontier_size = max(max_frontier_size, len(active))
                next_label_id += 1
            if not timed_out:
                active.discard(label_id)
            if timed_out:
                break

        if timed_out:
            certified_upper_bound = float(
                getattr(incumbent_evaluation, "objective")
            )
            for live_id in active:
                certified_upper_bound = max(
                    certified_upper_bound, upper_bound(labels[live_id])
                )
            exact = False
        else:
            certified_upper_bound = float(
                getattr(incumbent_evaluation, "objective")
            )
            exact = True
        certified_upper_bound = max(
            float(getattr(incumbent_evaluation, "objective")),
            certified_upper_bound,
        )
        try:
            finished_at = self._monotonic()
        except StopIteration:
            finished_at = last_observed_at
        elapsed_ms = max(0.0, (finished_at - frontier_started) * 1000.0)
        search_diagnostics = {
            "time_limit_seconds": time_limit_seconds,
            "labels_generated": int(labels_generated),
            "labels_expanded": int(labels_expanded),
            "labels_pruned": int(labels_pruned),
            "max_frontier_size": int(max_frontier_size),
            "remaining_frontier_size": int(len(active)),
            "deadline_reached": bool(timed_out),
            "elapsed_ms": float(elapsed_ms),
            "mode": "frontier",
        }
        return (
            incumbent_edges,
            incumbent_evaluation,
            exact,
            certified_upper_bound,
            search_diagnostics,
        )

    @staticmethod
    def _copy_endpoint_overlay(base: RoadGraph) -> EndpointRoadGraph:
        """Create an O(endpoint additions) structural view over the base graph."""
        _check_active_deadline()
        overlay = EndpointRoadGraph(base)
        _check_active_deadline()
        return overlay

    def _endpoint_graph(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        avoid_highways: bool = False,
    ) -> RoadGraph:
        """Return a request-local graph with legal partial-edge endpoints."""
        base = self.graph
        if base is None:
            raise RuntimeError("Road graph not loaded")
        _check_active_deadline()
        excluded_road_types = (
            HIGHWAY_ROAD_TYPES if avoid_highways else frozenset()
        )
        try:
            start_projections, _ = base.find_nearest_edge_positions_with_distance(
                *start,
                excluded_road_types=excluded_road_types,
                check_cancelled=_check_active_deadline,
            )
            end_projections, _ = base.find_nearest_edge_positions_with_distance(
                *end,
                excluded_road_types=excluded_road_types,
                check_cancelled=_check_active_deadline,
            )
        except ValueError:
            start_node, _ = base.find_nearest_node_with_distance(
                *start, check_cancelled=_check_active_deadline
            )
            end_node, _ = base.find_nearest_node_with_distance(
                *end, check_cancelled=_check_active_deadline
            )
            if start != end or start_node.id != end_node.id:
                raise ValueError("No route found between the given coordinates.")
            overlay = self._copy_endpoint_overlay(base)
            overlay._route_endpoint_node_ids = (start_node.id, end_node.id)
            overlay.freeze()
            return overlay
        overlay = self._copy_endpoint_overlay(base)

        epsilon = 1e-12

        def endpoint_node_id(
            projections: List[EdgeProjection], virtual_id: str
        ) -> str:
            """Collapse an endpoint only when every tied projection agrees."""
            boundary_node_ids: set[str] = set()
            for projection in projections:
                fraction = float(projection.fraction)
                if fraction <= epsilon:
                    boundary_node_ids.add(str(projection.edge.start_node_id))
                elif fraction >= 1.0 - epsilon:
                    boundary_node_ids.add(str(projection.edge.end_node_id))
                else:
                    return virtual_id
            if len(boundary_node_ids) == 1:
                return next(iter(boundary_node_ids))
            return virtual_id

        start_projection = start_projections[0]
        end_projection = end_projections[0]
        start_id = endpoint_node_id(
            start_projections, "__route_request_start__"
        )
        end_id = endpoint_node_id(end_projections, "__route_request_end__")
        while start_id in overlay.nodes and start_id.startswith("__route_request_start__"):
            start_id += "_"
        while (
            end_id in overlay.nodes
            and end_id.startswith("__route_request_end__")
        ) or (
            end_id == start_id
            and end_id.startswith("__route_request_end__")
        ):
            end_id += "_"
        if start_id not in overlay.nodes:
            overlay.add_node(
                Node(
                    start_id,
                    float(start_projection.lat),
                    float(start_projection.lon),
                )
            )
        if end_id not in overlay.nodes:
            overlay.add_node(
                Node(
                    end_id,
                    float(end_projection.lat),
                    float(end_projection.lon),
                )
            )
        start_is_virtual = start_id.startswith("__route_request_start__")
        end_is_virtual = end_id.startswith("__route_request_end__")

        def partial_edge(
            edge: Edge,
            edge_id: str,
            from_node: str,
            to_node: str,
            fraction: float,
            *,
            start_coordinate: Tuple[float, float] | None = None,
            end_coordinate: Tuple[float, float] | None = None,
        ) -> Edge:
            fraction = min(1.0, max(0.0, float(fraction)))
            value = Edge(
                id=edge_id,
                start_node_id=from_node,
                end_node_id=to_node,
                distance_km=float(edge.distance_km) * fraction,
                scenic_score=float(edge.scenic_score),
                road_name=edge.road_name,
                road_type=edge.road_type,
                speed_limit_kmh=edge.speed_limit_kmh,
                one_way=True,
            )
            value.canonical_edge_id = str(edge.id)
            value.direction = (
                "reverse" if ":reverse" in edge_id else "forward"
            )
            value.source_fraction = fraction
            if start_coordinate is not None:
                value.route_start_coordinate = start_coordinate
            if end_coordinate is not None:
                value.route_end_coordinate = end_coordinate
            return value

        if start_is_virtual:
            for projection_index, projection in enumerate(start_projections):
                edge = projection.edge
                fraction = float(projection.fraction)
                overlay.add_edge(
                    partial_edge(
                        edge,
                        f"__route_start__:{projection_index}:forward",
                        start_id,
                        edge.end_node_id,
                        1.0 - fraction,
                        start_coordinate=(
                            float(projection.lat),
                            float(projection.lon),
                        ),
                    )
                )
                if not edge.one_way:
                    overlay.add_edge(
                        partial_edge(
                            edge,
                            f"__route_start__:{projection_index}:reverse",
                            start_id,
                            edge.start_node_id,
                            fraction,
                            start_coordinate=(
                                float(projection.lat),
                                float(projection.lon),
                            ),
                        )
                    )

        if end_is_virtual:
            for projection_index, projection in enumerate(end_projections):
                edge = projection.edge
                fraction = float(projection.fraction)
                overlay.add_edge(
                    partial_edge(
                        edge,
                        f"__route_end__:{projection_index}:forward",
                        edge.start_node_id,
                        end_id,
                        fraction,
                        end_coordinate=(
                            float(projection.lat),
                            float(projection.lon),
                        ),
                    )
                )
        if start_is_virtual and end_is_virtual:
            for start_index, projection in enumerate(start_projections):
                edge = projection.edge
                start_fraction = float(projection.fraction)
                for end_index, end_projection in enumerate(end_projections):
                    if str(edge.id) != str(end_projection.edge.id):
                        continue
                    end_fraction = float(end_projection.fraction)
                    if start_fraction <= end_fraction:
                        overlay.add_edge(
                            partial_edge(
                                edge,
                                f"__route_direct__:{start_index}:{end_index}:forward",
                                start_id,
                                end_id,
                                end_fraction - start_fraction,
                                start_coordinate=(
                                    float(projection.lat),
                                    float(projection.lon),
                                ),
                                end_coordinate=(
                                    float(end_projection.lat),
                                    float(end_projection.lon),
                                ),
                            )
                        )
                    if not edge.one_way and start_fraction >= end_fraction:
                        overlay.add_edge(
                            partial_edge(
                                edge,
                                f"__route_direct__:{start_index}:{end_index}:reverse",
                                start_id,
                                end_id,
                                start_fraction - end_fraction,
                                start_coordinate=(
                                    float(projection.lat),
                                    float(projection.lon),
                                ),
                                end_coordinate=(
                                    float(end_projection.lat),
                                    float(end_projection.lon),
                                ),
                            )
                        )
        setattr(overlay, "_route_start_projections", tuple(start_projections))
        setattr(overlay, "_route_end_projections", tuple(end_projections))
        overlay._route_endpoint_node_ids = (start_id, end_id)
        overlay.freeze()
        _check_active_deadline()
        return overlay

    def _routing_endpoint_nodes(
        self, start: Tuple[float, float], end: Tuple[float, float]
    ) -> Tuple[Node, Node]:
        endpoint_ids = getattr(self.graph, "_route_endpoint_node_ids", None)
        if endpoint_ids is not None:
            return (
                self.graph.get_node(endpoint_ids[0]),
                self.graph.get_node(endpoint_ids[1]),
            )
        return (
            self.graph.find_nearest_node(
                *start, check_cancelled=_check_active_deadline
            ),
            self.graph.find_nearest_node(
                *end, check_cancelled=_check_active_deadline
            ),
        )

    def _build_endpoint_access_request(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        avoid_highways: bool = False,
    ) -> _EndpointAccessRequest:
        """Project both endpoints and freeze one reusable local overlay."""
        base = self.graph
        if base is None:
            raise RuntimeError("Road graph not loaded")
        active_stamp = base._heuristic_cache_stamp()
        overlay = cast(
            EndpointRoadGraph,
            self._endpoint_graph(
                start,
                end,
                avoid_highways=avoid_highways,
            ),
        )
        start_projections: List[EdgeProjection] = list(
            getattr(overlay, "_route_start_projections", ())
        )
        end_projections: List[EdgeProjection] = list(
            getattr(overlay, "_route_end_projections", ())
        )
        endpoint_ids = getattr(overlay, "_route_endpoint_node_ids", None)
        if endpoint_ids is None:
            raise RuntimeError("endpoint overlay lost endpoint nodes")
        start_id, end_id = (str(endpoint_ids[0]), str(endpoint_ids[1]))
        start_accesses = tuple(
            (index, direction, overlay.edges[edge_id])
            for index in range(len(start_projections))
            for direction, edge_id in (
                (0, f"__route_start__:{index}:forward"),
                (1, f"__route_start__:{index}:reverse"),
            )
            if edge_id in overlay.edges
        )
        end_accesses = tuple(
            (index, direction, overlay.edges[edge_id])
            for index in range(len(end_projections))
            for direction, edge_id in (
                (0, f"__route_end__:{index}:forward"),
                (1, f"__route_end__:{index}:reverse"),
            )
            if edge_id in overlay.edges
        )
        if not start_accesses:
            start_accesses = ((0, 0, None),)
        if not end_accesses:
            end_accesses = ((0, 0, None),)
        direct_candidates = tuple(
            (start_index, end_index, direction, edge)
            for start_index in range(len(start_projections))
            for end_index in range(len(end_projections))
            for direction, edge_id in (
                (
                    0,
                    f"__route_direct__:{start_index}:{end_index}:forward",
                ),
                (
                    1,
                    f"__route_direct__:{start_index}:{end_index}:reverse",
                ),
            )
            if (edge := overlay.edges.get(edge_id)) is not None
        )
        if (
            base is not self.graph
            or base._heuristic_cache_stamp() != active_stamp
        ):
            raise RuntimeError("road graph changed during endpoint setup")
        request = _EndpointAccessRequest(
            start=start,
            end=end,
            start_projections=start_projections,
            end_projections=end_projections,
            overlay=overlay,
            start_node_id=start_id,
            end_node_id=end_id,
            start_accesses=start_accesses,
            end_accesses=end_accesses,
            direct_candidates=direct_candidates,
            graph_stamp=active_stamp,
        )
        self._last_endpoint_access_request = request
        return request

    def _multi_access_builtin_path(
        self,
        overlay: EndpointRoadGraph,
        starts: List[EdgeProjection],
        ends: List[EdgeProjection],
        cost_function: ScenicCostFunction,
        direct_candidates: Tuple[Tuple[int, int, int, Edge], ...] = (),
    ) -> Optional[Tuple[float, List[Edge], int, int, Tuple[object, ...]]]:
        """Run one ranked search, including all direct endpoint candidates."""
        def access_cost(edge: Edge) -> float:
            return self._validated_nonnegative(
                cost_function.calculate(edge), "edge calculated cost"
            )

        if (
            not starts
            or not ends
            or "__route_start__:0:forward" not in overlay.edges
            or "__route_end__:0:forward" not in overlay.edges
        ):
            endpoint_ids = getattr(overlay, "_route_endpoint_node_ids", None)
            if endpoint_ids is None:
                return None
            result = self._bidirectional_search_core(
                cost_function,
                [(str(endpoint_ids[0]), 0.0, (0, 0), None)],
                [(str(endpoint_ids[1]), 0.0, (0, 0), None)],
                strict_mutation=True,
            )
            if result is None:
                return None
            path, rank_key = result
            return (
                self._path_duration_minutes(path),
                path,
                0,
                0,
                rank_key,
            )

        forward_seeds: List[
            Tuple[str, float, Tuple[object, ...], Optional[Edge]]
        ] = []
        reverse_seeds: List[
            Tuple[str, float, Tuple[object, ...], Optional[Edge]]
        ] = []
        for index in range(len(starts)):
            _check_active_deadline_at(index)
            prefix = overlay.edges[f"__route_start__:{index}:forward"]
            forward_seeds.append(
                (
                    str(prefix.end_node_id),
                    access_cost(prefix),
                    (index, 0),
                    prefix,
                )
            )
            reverse_id = f"__route_start__:{index}:reverse"
            if reverse_id in overlay.edges:
                prefix = overlay.edges[reverse_id]
                forward_seeds.append(
                    (
                        str(prefix.end_node_id),
                        access_cost(prefix),
                        (index, 1),
                        prefix,
                    )
                )
        for index in range(len(ends)):
            _check_active_deadline_at(len(starts) + index)
            suffix = overlay.edges[f"__route_end__:{index}:forward"]
            reverse_seeds.append(
                (
                    str(suffix.start_node_id),
                    access_cost(suffix),
                    (index, 0),
                    suffix,
                )
            )
            reverse_id = f"__route_end__:{index}:reverse"
            if reverse_id in overlay.edges:
                suffix = overlay.edges[reverse_id]
                reverse_seeds.append(
                    (
                        str(suffix.start_node_id),
                        access_cost(suffix),
                        (index, 1),
                        suffix,
                    )
                )
        result = self._bidirectional_search_core(
            cost_function,
            forward_seeds,
            reverse_seeds,
            strict_mutation=True,
        )
        best = None if result is None else (
            self._path_duration_minutes(result[0]),
            result[0],
            int(str(result[1][0])),
            int(str(result[1][1])),
            result[1],
        )
        for start_index, end_index, direction, edge in direct_candidates:
            _check_active_deadline_at(start_index + end_index)
            duration = self._path_duration_minutes([edge])
            rank_key = (start_index, end_index, 2, direction, ())
            candidate = (duration, [edge], start_index, end_index, rank_key)
            if best is None or (duration, rank_key) < (best[0], best[4]):
                best = candidate
        return best

    def _large_graph_fastest_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        avoid_highways: bool,
    ) -> Route:
        """Solve all legal projected endpoint accesses in one traversal."""
        base = self.graph
        assert base is not None
        request = self._build_endpoint_access_request(
            start,
            end,
            avoid_highways=avoid_highways,
        )
        self.graph = request.overlay
        try:
            policy = resolve_routing_policy(
                scenic_weight=0.0,
                kappa=1.0,
                avoid_highways=avoid_highways,
            )
            self.cost_function.strict_highways = policy.strict_highways
            self.cost_function.avoid_highways = policy.strict_highways
            self.cost_function.highway_preference = 0.0
            middle = self._multi_access_builtin_path(
                request.overlay,
                request.start_projections,
                request.end_projections,
                self._make_fastest_cost_function(),
                request.direct_candidates,
            )
            if middle is None:
                raise ValueError("No route found between the given coordinates.")
            _, best_edges, start_index, end_index, _ = middle
            if base._heuristic_cache_stamp() != request.graph_stamp:
                raise RuntimeError("road graph changed during fastest-path search")
            render_graph = RoadGraph()
            for node_id in (request.start_node_id, request.end_node_id):
                render_graph.add_node(request.overlay.nodes[node_id])
            for edge in best_edges:
                for node_id in (str(edge.start_node_id), str(edge.end_node_id)):
                    if node_id in render_graph.nodes:
                        continue
                    render_graph.add_node(
                        request.overlay.nodes.get(node_id, base.get_node(node_id))
                    )
            duration = self._path_duration_minutes(best_edges)
            evaluation = evaluate_path(
                best_edges,
                q=0.0,
                kappa=1.0,
                fastest_duration_minutes=duration,
                policy=policy,
                check_cancelled=_check_active_deadline,
            )
            self.graph = render_graph
            try:
                return self._path_to_route(
                    best_edges,
                    start_node=render_graph.get_node(request.start_node_id),
                    goal_node=render_graph.get_node(request.end_node_id),
                    evaluation=evaluation,
                    fastest_duration_minutes=duration,
                    requested_max_detour_factor=1.0,
                    exact=True,
                    exactness_status="exact",
                    algorithm="endpoint-access-duration-dijkstra",
                )
            finally:
                self.graph = request.overlay
        finally:
            self.graph = base


    @staticmethod
    def _simple_edge_path(edges: List[Edge]) -> bool:
        """Check that a path is a simple, connected directed traversal."""
        if not edges:
            return True
        nodes: set[str] = set()
        previous_end: Optional[str] = None
        for edge in edges:
            start_id = str(edge.start_node_id)
            end_id = str(edge.end_node_id)
            if previous_end is not None and start_id != previous_end:
                return False
            if start_id in nodes:
                return False
            nodes.add(start_id)
            previous_end = end_id
        return previous_end not in nodes

    @staticmethod
    def _has_scenic_improvement(candidate: object, baseline: object) -> bool:
        candidate_score = float(getattr(candidate, "normalized_scenic_score"))
        baseline_score = float(getattr(baseline, "normalized_scenic_score"))
        tolerance = 1e-12 * max(1.0, abs(candidate_score), abs(baseline_score))
        return candidate_score > baseline_score + tolerance

    def find_scenic_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        max_detour_factor: float = 1.8,
        *,
        q: Optional[float] = None,
        kappa: Optional[float] = None,
        highway_preference: Optional[float] = None,
        strict_highways: Optional[bool] = None,
        scenic_priority: bool = False,
        _endpoint_graph: bool = False,
        deadline: RoutingDeadline | None = None,
    ) -> Route:
        """Optimize scenic score within one shared request deadline."""
        with self._request_lock, _routing_deadline_scope(deadline):
            return self._find_scenic_route(
                start,
                end,
                scenic_weight,
                avoid_highways,
                max_detour_factor,
                q=q,
                kappa=kappa,
                highway_preference=highway_preference,
                strict_highways=strict_highways,
                scenic_priority=scenic_priority,
                _endpoint_graph=_endpoint_graph,
            )

    def _find_scenic_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        max_detour_factor: float = 1.8,
        *,
        q: Optional[float] = None,
        kappa: Optional[float] = None,
        highway_preference: Optional[float] = None,
        strict_highways: Optional[bool] = None,
        scenic_priority: bool = False,
        _endpoint_graph: bool = False,
    ) -> Route:
        """Optimize scenic score within a hard duration cap when requested."""
        if self.graph is None:
            raise RuntimeError("Road graph not loaded")
        if not _endpoint_graph:
            base_graph = self.graph
            request = self._build_endpoint_access_request(
                start,
                end,
                avoid_highways=(
                    bool(strict_highways)
                    if strict_highways is not None
                    else bool(avoid_highways)
                ),
            )
            setattr(request.overlay, "_route_access_request", request)
            self.graph = request.overlay
            try:
                return self._find_scenic_route(
                    start,
                    end,
                    scenic_weight,
                    avoid_highways,
                    max_detour_factor,
                    q=q,
                    kappa=kappa,
                    highway_preference=highway_preference,
                    strict_highways=strict_highways,
                    scenic_priority=scenic_priority,
                    _endpoint_graph=True,
                )
            finally:
                self.graph = base_graph
        if q is not None:
            scenic_weight = q
        if kappa is not None:
            max_detour_factor = kappa
        policy = resolve_routing_policy(
            scenic_weight=scenic_weight,
            kappa=max_detour_factor,
            avoid_highways=avoid_highways,
            highway_preference=(
                0.0 if highway_preference is None else highway_preference
            ),
            strict_highways=strict_highways,
            scenic_priority=scenic_priority,
        )
        self.cost_function.strict_highways = policy.strict_highways
        self.cost_function.avoid_highways = policy.strict_highways
        self.cost_function.highway_preference = policy.highway_preference
        q_value = policy.scenic_weight
        kappa_value = policy.kappa
        start_node, end_node = self._routing_endpoint_nodes(start, end)
        bounded_graph = self._is_bounded_oracle_graph()
        if bounded_graph:
            shortest_edges = self._canonical_fastest_edges(
                start_node, end_node, policy.strict_highways
            )
        else:
            request = getattr(self.graph, "_route_access_request", None)
            if (
                request is not None
                and len(request.overlay.base_graph.nodes)
                > self._ENDPOINT_OVERLAY_MAX_NODES
            ):
                ranked = self._multi_access_builtin_path(
                    request.overlay,
                    request.start_projections,
                    request.end_projections,
                    self._make_fastest_cost_function(),
                    request.direct_candidates,
                )
                shortest_edges = None if ranked is None else ranked[1]
            else:
                shortest_edges = self._cached_fastest_edges(
                    start_node,
                    end_node,
                    policy.strict_highways,
                    policy.highway_preference,
                )
        if shortest_edges is None:
            raise ValueError("No route found between the given coordinates.")
        fastest_duration = self._path_duration_minutes(shortest_edges)
        duration_cap = fastest_duration * kappa_value
        if not math.isfinite(duration_cap):
            raise ValueError("kappa must produce a finite duration cap")
        fastest_evaluation = evaluate_path(
            shortest_edges,
            q=q_value,
            kappa=kappa_value,
            fastest_duration_minutes=fastest_duration,
            policy=policy,
            check_cancelled=_check_active_deadline,
        )

        if (
            q_value == 0.0
            and policy.highway_preference == 0.0
            and not policy.scenic_priority
        ):
            return self._path_to_route(
                shortest_edges,
                start_node=start_node,
                goal_node=end_node,
                evaluation=fastest_evaluation,
                fastest_duration_minutes=fastest_duration,
                requested_max_detour_factor=kappa_value,
                exact=True,
                exactness_status="exact",
                algorithm=(
                    "canonical-duration-dijkstra"
                    if bounded_graph
                    else "compiled-duration-dijkstra"
                ),
            )

        if bounded_graph:
            path_edges, evaluation = self._enumerate_simple_optimum(
                start_node,
                end_node,
                q=q_value,
                kappa=kappa_value,
                fastest_duration_minutes=fastest_duration,
                duration_cap_minutes=duration_cap,
                policy=policy,
            )
            no_scenic_improvement = not self._has_scenic_improvement(
                evaluation, fastest_evaluation
            )
            return self._path_to_route(
                path_edges,
                start_node=start_node,
                goal_node=end_node,
                evaluation=evaluation,
                fastest_duration_minutes=fastest_duration,
                requested_max_detour_factor=kappa_value,
                exact=True,
                exactness_status="exact",
                algorithm="exact-simple-path-oracle",
                zero_improvement_reason=(
                    "no_feasible_scenic_improvement"
                    if no_scenic_improvement
                    else None
                ),
            )

        (
            best_edges,
            best_evaluation,
            exact,
            certified_upper_bound,
            search_diagnostics,
        ) = self._production_frontier_search(
            start_node,
            end_node,
            q=q_value,
            kappa=kappa_value,
            fastest_duration_minutes=fastest_duration,
            duration_cap_minutes=duration_cap,
            policy=policy,
            fastest_edges=shortest_edges,
            fastest_evaluation=fastest_evaluation,
        )
        no_scenic_improvement = not self._has_scenic_improvement(
            best_evaluation, fastest_evaluation
        )
        return self._path_to_route(
            best_edges,
            start_node=start_node,
            goal_node=end_node,
            evaluation=best_evaluation,
            fastest_duration_minutes=fastest_duration,
            requested_max_detour_factor=kappa_value,
            exact=exact,
            exactness_status=(
                "exact" if exact else "approximate-certified"
            ),
            optimality_gap=max(
                0.0,
                certified_upper_bound
                - float(getattr(best_evaluation, "objective")),
            ),
            certified_upper_bound=certified_upper_bound,
            search_diagnostics=search_diagnostics,
            algorithm="production-multilabel-frontier",
            zero_improvement_reason=(
                "no_feasible_scenic_improvement"
                if no_scenic_improvement
                else None
            ),
        )

    def find_fastest_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        avoid_highways: bool = False,
        _endpoint_graph: bool = False,
        deadline: RoutingDeadline | None = None,
    ) -> Route:
        """Return the minimum-duration route within one request deadline."""
        with self._request_lock, _routing_deadline_scope(deadline):
            return self._find_fastest_route(
                start,
                end,
                avoid_highways=avoid_highways,
                _endpoint_graph=_endpoint_graph,
            )

    def _find_fastest_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        avoid_highways: bool = False,
        _endpoint_graph: bool = False,
    ) -> Route:
        if self.graph is None:
            raise RuntimeError("Road graph not loaded")
        if not _endpoint_graph:
            base_graph = self.graph
            cached_request = self._last_endpoint_access_request
            reusable = (
                cached_request is not None
                and cached_request.overlay.base_graph is base_graph
                and cached_request.start == start
                and cached_request.end == end
                and cached_request.graph_stamp
                == base_graph._heuristic_cache_stamp()
            )
            if reusable:
                self.graph = cached_request.overlay
                try:
                    return self.find_fastest_route(
                        start,
                        end,
                        avoid_highways=avoid_highways,
                        _endpoint_graph=True,
                    )
                finally:
                    self.graph = base_graph
            if len(base_graph.nodes) > self._ENDPOINT_OVERLAY_MAX_NODES:
                return self._large_graph_fastest_route(
                    start,
                    end,
                    avoid_highways=avoid_highways,
                )
            request = self._build_endpoint_access_request(
                start,
                end,
                avoid_highways=bool(avoid_highways),
            )
            setattr(request.overlay, "_route_access_request", request)
            self.graph = request.overlay
            try:
                return self.find_fastest_route(
                    start,
                    end,
                    avoid_highways=avoid_highways,
                    _endpoint_graph=True,
                )
            finally:
                self.graph = base_graph
        policy = resolve_routing_policy(
            scenic_weight=0.0,
            kappa=1.0,
            avoid_highways=avoid_highways,
        )
        self.cost_function.strict_highways = policy.strict_highways
        self.cost_function.avoid_highways = policy.strict_highways
        self.cost_function.highway_preference = 0.0
        start_node, end_node = self._routing_endpoint_nodes(start, end)
        bounded_graph = self._is_bounded_oracle_graph()
        if bounded_graph:
            shortest_edges = self._canonical_fastest_edges(
                start_node, end_node, policy.strict_highways
            )
        else:
            request = getattr(self.graph, "_route_access_request", None)
            if (
                request is not None
                and len(request.overlay.base_graph.nodes)
                > self._ENDPOINT_OVERLAY_MAX_NODES
            ):
                ranked = self._multi_access_builtin_path(
                    request.overlay,
                    request.start_projections,
                    request.end_projections,
                    self._make_fastest_cost_function(),
                    request.direct_candidates,
                )
                shortest_edges = None if ranked is None else ranked[1]
            else:
                shortest_edges = self._cached_fastest_edges(
                    start_node, end_node, policy.strict_highways, 0.0
                )
        if shortest_edges is None:
            raise ValueError("No route found between the given coordinates.")
        fastest_duration = self._path_duration_minutes(shortest_edges)
        evaluation = evaluate_path(
            shortest_edges,
            q=0.0,
            kappa=1.0,
            fastest_duration_minutes=fastest_duration,
            policy=policy,
            check_cancelled=_check_active_deadline,
        )
        return self._path_to_route(
            shortest_edges,
            start_node=start_node,
            goal_node=end_node,
            evaluation=evaluation,
            fastest_duration_minutes=fastest_duration,
            requested_max_detour_factor=1.0,
            exact=True,
            exactness_status="exact",
            algorithm=(
                "canonical-duration-dijkstra"
                if bounded_graph
                else "compiled-duration-dijkstra"
            ),
        )

    def _a_star(
        self,
        start: Node,
        goal: Node,
        *,
        cost_function: ScenicCostFunction,
        max_path_km: float | None = None,
        max_feasible_cost: float | None = None,
        shortest_distance_km: float | None = None,
        max_path_minutes: float | None = None,
        shortest_duration_minutes: float | None = None,
    ) -> Optional[List[Edge]]:
        # Keep the historical distance arguments for private callers.  Scenic
        # routing uses the explicit duration resource.
        if max_path_km is not None and max_path_minutes is not None:
            raise ValueError("only one constrained resource may be supplied")
        if max_path_minutes is not None:
            return self._resource_constrained_path(
                start,
                goal,
                cost_function=cost_function,
                max_path_minutes=max_path_minutes,
                max_feasible_cost=max_feasible_cost,
                shortest_duration_minutes=shortest_duration_minutes,
                resource_kind="duration",
            )
        if max_path_km is not None:
            return self._resource_constrained_path(
                start,
                goal,
                cost_function=cost_function,
                max_path_km=max_path_km,
                max_feasible_cost=max_feasible_cost,
                shortest_distance_km=shortest_distance_km,
                resource_kind="distance",
            )
        if shortest_duration_minutes is not None:
            raise ValueError(
                "shortest_duration_minutes requires max_path_minutes"
            )
        eligible_builtin = self._reverse_cost_eligible(cost_function)
        large_builtin = eligible_builtin and (
            len(self.graph.edges) > self._LARGE_GRAPH_EDGE_THRESHOLD
            or (
                isinstance(self.graph, EndpointRoadGraph)
                and len(self.graph.base_graph.edges)
                > self._LARGE_GRAPH_EDGE_THRESHOLD
            )
        )
        if large_builtin:
            # The compact target-bounded search must precede any full-source
            # SciPy invocation on production-sized graphs.
            bidirectional_path = self._bidirectional_builtin_path(
                start, goal, cost_function
            )
            if bidirectional_path is not None:
                return bidirectional_path
        elif eligible_builtin:
            reachability = self._compiled_reachability_result(
                start, goal, cost_function
            )
            if reachability is False:
                return None
            compiled_path = self._compiled_builtin_path(
                start, goal, cost_function
            )
            if compiled_path is not None:
                return compiled_path
            if (
                self._compiled_reachability_result(
                    start, goal, cost_function
                )
                is False
            ):
                return None

        minimum_cost_per_km = (
            self._minimum_cost_per_km(cost_function)
            if self._reverse_cost_eligible(cost_function)
            else 0.0
        )
        start_to_goal_km = self._haversine(
            start.lat, start.lon, goal.lat, goal.lon
        )
        frontier: List[Tuple[float, float, str]] = []
        heapq.heappush(
            frontier, (minimum_cost_per_km * start_to_goal_km, 0.0, start.id)
        )  # (priority, cost_so_far, node_id)

        came_from: Dict[str, Tuple[str, Edge]] = {}
        best_cost: Dict[str, float] = {start.id: 0.0}
        best_distance_km: Dict[str, float] = {start.id: 0.0}
        if "get_edges" in getattr(self.graph, "__dict__", {}):
            edge_iterator = self.graph.get_edges
        else:
            edge_iterator = getattr(self.graph, "iter_edges", None)
            if not callable(edge_iterator):
                edge_iterator = self.graph.get_edges

        expanded = 0
        while frontier:
            _check_active_deadline_at(expanded)
            expanded += 1
            _, current_cost, current_id = heapq.heappop(frontier)
            if current_id == goal.id:
                return self._reconstruct_path(came_from, goal.id)
            if current_cost > best_cost.get(current_id, float("inf")):
                continue

            for edge in edge_iterator(current_id):
                if not self._edge_is_eligible(edge, cost_function):
                    continue
                neighbor_id = edge.end_node_id
                edge_distance_km = self._validated_nonnegative(
                    edge.distance_km, "edge distance_km"
                )
                self._edge_duration_minutes(edge)
                next_distance = (
                    best_distance_km[current_id] + edge_distance_km
                )
                if not math.isfinite(next_distance):
                    raise ValueError(
                        "cumulative traversed distance must be finite and non-negative"
                    )
                edge_cost = self._validated_nonnegative(
                    cost_function.calculate(edge), "edge calculated cost"
                )
                next_cost = current_cost + edge_cost
                if not math.isfinite(next_cost):
                    raise ValueError(
                        "cumulative calculated cost must be finite and non-negative"
                    )
                if next_cost >= best_cost.get(neighbor_id, float("inf")):
                    continue

                best_cost[neighbor_id] = next_cost
                best_distance_km[neighbor_id] = next_distance
                came_from[neighbor_id] = (current_id, edge)

                neighbor = self.graph.get_node(neighbor_id)
                h = self._haversine(
                    neighbor.lat, neighbor.lon, goal.lat, goal.lon
                )
                heuristic = minimum_cost_per_km * h
                heapq.heappush(
                    frontier, (next_cost + heuristic, next_cost, neighbor_id)
                )

    @staticmethod
    def _label_dominates(
        first: _PathLabel,
        second: _PathLabel,
        resource_kind: str = "distance",
    ) -> bool:
        """Return whether ``first`` is strictly better in cost and resource."""
        first_resource = (
            first.cumulative_duration_minutes
            if resource_kind == "duration"
            else first.cumulative_distance_km
        )
        second_resource = (
            second.cumulative_duration_minutes
            if resource_kind == "duration"
            else second.cumulative_distance_km
        )
        return (
            first_resource <= second_resource
            and first.cumulative_cost <= second.cumulative_cost
            and (
                first_resource < second_resource
                or first.cumulative_cost < second.cumulative_cost
            )
        )

    @staticmethod
    def _built_in_cost_signature(
        cost_function: ScenicCostFunction,
    ) -> Optional[Tuple[object, ...]]:
        """Return immutable inputs only for the untouched built-in cost."""
        if type(cost_function) is not ScenicCostFunction:
            return None
        if type(cost_function.weights) is not CostWeights:
            return None
        if "calculate" in cost_function.__dict__:
            return None
        if "_road_type_adjustment" in cost_function.__dict__:
            return None
        try:
            if type(cost_function).calculate is not _ORIGINAL_SCENIC_CALCULATE:
                return None
            if (
                type(cost_function)._road_type_adjustment
                is not _ORIGINAL_SCENIC_ROAD_TYPE_ADJUSTMENT
            ):
                return None
        except (AttributeError, TypeError):
            return None
        weights = cost_function.weights
        return (
            float(cost_function.scenic_weight),
            bool(getattr(cost_function, "strict_highways", False)),
            float(getattr(cost_function, "highway_preference", 0.0)),
            float(weights.travel_time),
            float(weights.scenic_reward),
            float(weights.highway_penalty),
            float(weights.scenic_byway_bonus),
        )

    @staticmethod
    def _reverse_cost_eligible(
        cost_function: ScenicCostFunction,
    ) -> bool:
        """Return whether the exact original built-in implementation is active."""
        return ScenicRoutePlanner._built_in_cost_signature(cost_function) is not None

    @staticmethod
    def _validated_nonnegative(value: object, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite and non-negative") from exc
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def _edge_distances_are_geodesic_lower_bounds(self) -> bool:
        """Return whether stored traversable distances cover endpoint geodesics."""
        graph = self.graph
        stamp = graph._heuristic_cache_stamp()
        cached = self._geodesic_lower_bounds_cache
        if cached is not None:
            cached_graph, cached_stamp, cached_result = cached
            if cached_graph is graph and cached_stamp == stamp:
                if (
                    graph is self.graph
                    and graph._heuristic_cache_stamp() == stamp
                ):
                    return cached_result
                self._geodesic_lower_bounds_cache = None

        result = True
        for edge in graph.edges.values():
            try:
                distance_km = float(edge.distance_km)
            except (TypeError, ValueError, OverflowError):
                result = False
                break
            if not math.isfinite(distance_km) or distance_km < 0.0:
                result = False
                break
            start = graph.get_node(edge.start_node_id)
            end = graph.get_node(edge.end_node_id)
            endpoint_km = self._haversine(start.lat, start.lon, end.lat, end.lon)
            if not math.isfinite(endpoint_km) or endpoint_km < 0.0:
                result = False
                break
            if distance_km < endpoint_km:
                result = False
                break

        # A scan can observe a mixed graph state.  Never retain or return a
        # positive result under a stamp that no longer describes the graph.
        final_stamp = graph._heuristic_cache_stamp()
        if graph is not self.graph or final_stamp != stamp:
            return False
        self._geodesic_lower_bounds_cache = (graph, stamp, result)
        return result

    def _minimum_cost_per_km(self, cost_function: ScenicCostFunction) -> float:
        """Return a safe graph-wide nonnegative cost-per-kilometre lower bound."""
        graph = self.graph
        stamp = graph._heuristic_cache_stamp()
        cache_context = self._minimum_cost_per_km_cache_context
        if (
            cache_context is None
            or cache_context[0] is not graph
            or cache_context[1] != stamp
        ):
            self._minimum_cost_per_km_cache = OrderedDict()
            self._minimum_cost_per_km_cache_context = (graph, stamp)
        signature = self._built_in_cost_signature(cost_function)
        cache_key = None
        if signature is not None:
            cache_key = (id(graph), stamp, signature)
            cached = self._minimum_cost_per_km_cache.get(cache_key)
            if cached is not None and cached[0] is graph:
                if (
                    graph is self.graph
                    and graph._heuristic_cache_stamp() == stamp
                ):
                    self._minimum_cost_per_km_cache.move_to_end(cache_key)
                    return cached[1]
                self._minimum_cost_per_km_cache.pop(cache_key, None)

        if not self._edge_distances_are_geodesic_lower_bounds():
            minimum_cost_per_km = 0.0
        else:
            minimum_cost_per_km = float("inf")
            for edge in graph.edges.values():
                if not self._edge_is_eligible(edge, cost_function):
                    continue
                distance_km = self._validated_nonnegative(
                    edge.distance_km, "edge distance_km"
                )
                if distance_km <= 0.0:
                    continue
                edge_cost = self._validated_nonnegative(
                    cost_function.calculate(edge), "edge calculated cost"
                )
                ratio = edge_cost / distance_km
                if math.isfinite(ratio) and ratio < minimum_cost_per_km:
                    minimum_cost_per_km = ratio
            if minimum_cost_per_km == float("inf"):
                minimum_cost_per_km = 0.0

        # Recheck after the complete ratio/geometry work.  A changed graph
        # invalidates the observed positive bound; custom cost functions
        # intentionally bypass this cache.
        final_stamp = graph._heuristic_cache_stamp()
        if graph is not self.graph or final_stamp != stamp:
            self._minimum_cost_per_km_cache = OrderedDict()
            self._minimum_cost_per_km_cache_context = None
            return 0.0
        if signature is not None:
            assert cache_key is not None
            self._minimum_cost_per_km_cache[cache_key] = (
                graph,
                minimum_cost_per_km,
            )
            self._minimum_cost_per_km_cache.move_to_end(cache_key)
            while len(self._minimum_cost_per_km_cache) > (
                self._MINIMUM_COST_CACHE_CAPACITY
            ):
                self._minimum_cost_per_km_cache.popitem(last=False)
        return minimum_cost_per_km

    def _build_reverse_predecessor_snapshot(
        self,
        cost_function: Optional[ScenicCostFunction] = None,
        *,
        deadline: Optional[float] = None,
    ) -> Optional[_ReversePredecessorSnapshot]:
        """Build one directed predecessor snapshot for all reverse bounds."""
        graph = self.graph
        stamp = graph._heuristic_cache_stamp()
        include_costs = (
            cost_function is not None
            and self._reverse_cost_eligible(cost_function)
        )
        predecessors: Dict[
            str, List[Tuple[str, float, float, Optional[float]]]
        ] = {}
        avoid_highways = self._avoids_highways(cost_function)

        def expired() -> bool:
            _check_active_deadline()
            return deadline is not None and self._monotonic() >= deadline

        # Materialize node IDs before enumeration.  The final stamp check
        # rejects a concurrent mapping/edge mutation rather than mixing epochs.
        if expired():
            return None
        for node_id in tuple(graph.nodes):
            try:
                edges = graph.get_edges(node_id)
            except Exception:
                return None
            for edge in edges:
                if expired():
                    return None
                if avoid_highways and is_highway_road_type(edge.road_type):
                    continue
                distance_km = self._validated_nonnegative(
                    edge.distance_km, "edge distance_km"
                )
                duration_minutes = self._edge_duration_minutes(edge)
                edge_cost: Optional[float] = None
                if include_costs:
                    edge_cost = self._validated_nonnegative(
                        cost_function.calculate(edge), "edge calculated cost"
                    )
                predecessors.setdefault(edge.end_node_id, []).append(
                    (node_id, distance_km, duration_minutes, edge_cost)
                )
        if graph is not self.graph or graph._heuristic_cache_stamp() != stamp:
            return None
        return _ReversePredecessorSnapshot(
            graph, stamp, avoid_highways, predecessors
        )

    def _active_or_build_reverse_snapshot(
        self,
        cost_function: Optional[ScenicCostFunction] = None,
    ) -> Optional[_ReversePredecessorSnapshot]:
        snapshot = self._active_reverse_snapshot
        if snapshot is not None:
            if (
                snapshot.graph is self.graph
                and self.graph._heuristic_cache_stamp() == snapshot.stamp
                and snapshot.avoid_highways == self._avoids_highways(cost_function)
            ):
                if cost_function is None or not self._reverse_cost_eligible(
                    cost_function
                ) or all(
                    edge_cost is not None
                    for entries in snapshot.predecessors.values()
                    for _, _, _, edge_cost in entries
                ):
                    return snapshot
            return None
        return self._build_reverse_predecessor_snapshot(cost_function)

    def _reverse_distance_lower_bounds(
        self,
        goal: Node,
        max_path_km: float,
    ) -> Optional[Dict[str, float]]:
        """Return exact distance-to-goal bounds for the current traversal graph."""
        cap = self._validated_nonnegative(max_path_km, "max_path_km")
        snapshot = self._active_or_build_reverse_snapshot()
        if snapshot is None:
            return None
        lower_bounds: Dict[str, float] = {goal.id: 0.0}
        frontier: List[Tuple[float, str]] = [(0.0, goal.id)]
        while frontier:
            _check_active_deadline()
            distance_km, node_id = heapq.heappop(frontier)
            if lower_bounds.get(node_id) != distance_km:
                continue
            for (
                predecessor_id,
                edge_distance_km,
                _edge_duration_minutes,
                _edge_cost,
            ) in snapshot.predecessors.get(node_id, ()):
                next_distance_km = distance_km + edge_distance_km
                if not math.isfinite(next_distance_km):
                    return None
                if next_distance_km > cap:
                    continue
                previous_distance_km = lower_bounds.get(predecessor_id)
                if previous_distance_km is None or (
                    next_distance_km < previous_distance_km
                ):
                    lower_bounds[predecessor_id] = next_distance_km
                    heapq.heappush(
                        frontier, (next_distance_km, predecessor_id)
                    )
        if (
            snapshot.graph is not self.graph
            or snapshot.graph._heuristic_cache_stamp() != snapshot.stamp
        ):
            return None
        return lower_bounds

    def _reverse_duration_lower_bounds(
        self,
        goal: Node,
        max_path_minutes: float,
    ) -> Optional[Dict[str, float]]:
        """Return exact travel-duration-to-goal bounds in minutes."""
        cap = self._validated_nonnegative(
            max_path_minutes, "max_path_minutes"
        )
        snapshot = self._active_or_build_reverse_snapshot()
        if snapshot is None:
            return None
        lower_bounds: Dict[str, float] = {goal.id: 0.0}
        frontier: List[Tuple[float, str]] = [(0.0, goal.id)]
        while frontier:
            _check_active_deadline()
            duration_minutes, node_id = heapq.heappop(frontier)
            if lower_bounds.get(node_id) != duration_minutes:
                continue
            for (
                predecessor_id,
                _edge_distance_km,
                edge_duration_minutes,
                _edge_cost,
            ) in snapshot.predecessors.get(node_id, ()):
                next_duration_minutes = (
                    duration_minutes + edge_duration_minutes
                )
                if not math.isfinite(next_duration_minutes):
                    return None
                if next_duration_minutes > cap:
                    continue
                previous_duration_minutes = lower_bounds.get(predecessor_id)
                if previous_duration_minutes is None or (
                    next_duration_minutes < previous_duration_minutes
                ):
                    lower_bounds[predecessor_id] = next_duration_minutes
                    heapq.heappush(
                        frontier, (next_duration_minutes, predecessor_id)
                    )
        if (
            snapshot.graph is not self.graph
            or snapshot.graph._heuristic_cache_stamp() != snapshot.stamp
        ):
            return None
        return lower_bounds


    def _reverse_cost_lower_bounds(
        self,
        goal: Node,
        cost_function: ScenicCostFunction,
    ) -> Optional[Dict[str, float]]:
        """Return exact unconstrained cost-to-goal bounds for built-in costs."""
        if not self._reverse_cost_eligible(cost_function):
            return None
        snapshot = self._active_or_build_reverse_snapshot(cost_function)
        if snapshot is None:
            return None
        lower_bounds: Dict[str, float] = {goal.id: 0.0}
        frontier: List[Tuple[float, str]] = [(0.0, goal.id)]
        while frontier:
            _check_active_deadline()
            cost, node_id = heapq.heappop(frontier)
            if lower_bounds.get(node_id) != cost:
                continue
            for (
                predecessor_id,
                _distance,
                _duration_minutes,
                edge_cost,
            ) in snapshot.predecessors.get(node_id, ()):
                if edge_cost is None:
                    return None
                next_cost = cost + edge_cost
                if not math.isfinite(next_cost):
                    return None
                previous_cost = lower_bounds.get(predecessor_id)
                if previous_cost is None or next_cost < previous_cost:
                    lower_bounds[predecessor_id] = next_cost
                    heapq.heappush(
                        frontier, (next_cost, predecessor_id)
                    )
        if (
            snapshot.graph is not self.graph
            or snapshot.graph._heuristic_cache_stamp() != snapshot.stamp
        ):
            return None
        return lower_bounds

    def _reverse_augmented_cost_lower_bounds(
        self,
        goal: Node,
        cost_function: ScenicCostFunction,
        lambda_value: float,
        resource_kind: str = "distance",
    ) -> Optional[Dict[str, float]]:
        """Return a local reverse potential for ``cost + lambda * resource``."""
        if not self._reverse_cost_eligible(cost_function):
            return None
        lagrangian_lambda = self._validated_nonnegative(
            lambda_value, "lambda_value"
        )
        snapshot = self._active_or_build_reverse_snapshot(cost_function)
        if snapshot is None:
            return None
        lower_bounds: Dict[str, float] = {goal.id: 0.0}
        frontier: List[Tuple[float, str]] = [(0.0, goal.id)]
        while frontier:
            _check_active_deadline()
            augmented_cost, node_id = heapq.heappop(frontier)
            if lower_bounds.get(node_id) != augmented_cost:
                continue
            for (
                predecessor_id,
                edge_distance_km,
                edge_duration_minutes,
                edge_cost,
            ) in snapshot.predecessors.get(node_id, ()):
                if edge_cost is None:
                    return None
                edge_resource = (
                    edge_duration_minutes
                    if resource_kind == "duration"
                    else edge_distance_km
                )
                augmented_edge_cost = (
                    edge_cost + lagrangian_lambda * edge_resource
                )
                if (
                    not math.isfinite(augmented_edge_cost)
                    or augmented_edge_cost < 0.0
                ):
                    return None
                next_augmented_cost = augmented_cost + augmented_edge_cost
                if (
                    not math.isfinite(next_augmented_cost)
                    or next_augmented_cost < 0.0
                ):
                    return None
                previous_cost = lower_bounds.get(predecessor_id)
                if previous_cost is None or next_augmented_cost < previous_cost:
                    lower_bounds[predecessor_id] = next_augmented_cost
                    heapq.heappush(
                        frontier, (next_augmented_cost, predecessor_id)
                    )
        if (
            snapshot.graph is not self.graph
            or snapshot.graph._heuristic_cache_stamp() != snapshot.stamp
        ):
            return None
        return lower_bounds

    @staticmethod
    def _lagrangian_bound_exceeds_incumbent(
        cumulative_cost: float,
        augmented_lower_bound: float,
        cumulative_distance_km: float,
        lambda_value: float,
        max_path_km: float,
        incumbent_cost: float,
    ) -> Optional[bool]:
        """Check ``g + P_lambda + lambda*g_d - lambda*C`` conservatively."""
        try:
            cumulative_cost = float(cumulative_cost)
            augmented_lower_bound = float(augmented_lower_bound)
            cumulative_distance_km = float(cumulative_distance_km)
            lambda_value = float(lambda_value)
            max_path_km = float(max_path_km)
            incumbent_cost = float(incumbent_cost)
        except (TypeError, ValueError, OverflowError):
            return None
        lambda_distance = lambda_value * cumulative_distance_km
        lambda_cap = lambda_value * max_path_km
        if (
            not math.isfinite(cumulative_cost)
            or cumulative_cost < 0.0
            or not math.isfinite(augmented_lower_bound)
            or augmented_lower_bound < 0.0
            or not math.isfinite(cumulative_distance_km)
            or cumulative_distance_km < 0.0
            or not math.isfinite(lambda_value)
            or lambda_value < 0.0
            or not math.isfinite(max_path_km)
            or max_path_km < 0.0
            or not math.isfinite(incumbent_cost)
            or not math.isfinite(lambda_distance)
            or not math.isfinite(lambda_cap)
        ):
            return None
        total_bound = (
            cumulative_cost + augmented_lower_bound + lambda_distance - lambda_cap
        )
        if not math.isfinite(total_bound):
            return None
        tolerance = 1e-9 * max(
            1.0,
            abs(cumulative_cost),
            abs(augmented_lower_bound),
            abs(lambda_distance),
            abs(lambda_cap),
            abs(incumbent_cost),
        )
        if not math.isfinite(tolerance):
            return None
        threshold = incumbent_cost + tolerance
        if not math.isfinite(threshold):
            return None
        return total_bound > threshold

    @staticmethod
    def _cost_bound_exceeds_incumbent(
        cumulative_cost: float,
        reverse_lower_bound: float,
        incumbent_cost: float,
    ) -> bool:
        """Check a cost bound with a conservative round-off allowance."""
        total_cost = cumulative_cost + reverse_lower_bound
        tolerance = 1e-9 * max(
            1.0,
            abs(cumulative_cost),
            abs(reverse_lower_bound),
            abs(incumbent_cost),
        )
        return total_cost > incumbent_cost + tolerance

    @staticmethod
    def _resource_bound_exceeds_cap(
        cumulative_distance_km: float,
        reverse_lower_bound_km: float,
        max_path_km: float,
    ) -> bool:
        total_distance_km = cumulative_distance_km + reverse_lower_bound_km
        tolerance_km = 1e-9 * max(
            1.0,
            abs(cumulative_distance_km),
            abs(reverse_lower_bound_km),
            abs(max_path_km),
        )
        return total_distance_km > max_path_km + tolerance_km

    def _resource_constrained_path(
        self,
        start: Node,
        goal: Node,
        *,
        cost_function: ScenicCostFunction,
        max_path_km: float | None = None,
        max_feasible_cost: float | None = None,
        shortest_distance_km: float | None = None,
        max_path_minutes: float | None = None,
        shortest_duration_minutes: float | None = None,
        resource_kind: str = "distance",
    ) -> Optional[List[Edge]]:
        """Find the least-cost path under a nonnegative additive resource.

        Scenic calls use travel duration in minutes.  Distance remains a
        compatibility resource for direct callers of the historical private
        helper.
        """
        if resource_kind not in {"distance", "duration"}:
            raise ValueError("resource_kind must be distance or duration")
        if resource_kind == "duration":
            if max_path_minutes is None:
                raise ValueError("max_path_minutes is required")
            cap = self._validated_nonnegative(
                max_path_minutes, "max_path_minutes"
            )
            shortest_for_lagrangian = None
            if shortest_duration_minutes is not None:
                shortest_for_lagrangian = self._validated_nonnegative(
                    shortest_duration_minutes, "shortest_duration_minutes"
                )
        else:
            if max_path_km is None:
                raise ValueError("max_path_km is required")
            cap = self._validated_nonnegative(max_path_km, "max_path_km")
            shortest_for_lagrangian = None
            if shortest_distance_km is not None:
                shortest_for_lagrangian = self._validated_nonnegative(
                    shortest_distance_km, "shortest_distance_km"
                )
        incumbent_cost: float | None = None
        if max_feasible_cost is not None:
            incumbent_cost = self._validated_nonnegative(
                max_feasible_cost, "max_feasible_cost"
            )
        reverse_cost_lower_bounds: Optional[Dict[str, float]] = None
        reverse_lower_bounds: Optional[Dict[str, float]] = None
        lagrangian_potentials: List[
            Tuple[float, Dict[str, float], object]
        ] = []
        reverse_snapshot: Optional[_ReversePredecessorSnapshot] = None
        preprocess_reverse = (
            len(self.graph.edges) <= self._LARGE_GRAPH_EDGE_THRESHOLD
            and not (
                len(self.graph.edges) > self._REVERSE_PREPROCESS_EDGE_THRESHOLD
                and cap <= (
                    self._SHORT_ROUTE_CAP_KM
                    if resource_kind == "distance"
                    else 5.0
                )
            )
        )
        if preprocess_reverse:
            reverse_snapshot = self._build_reverse_predecessor_snapshot(
                cost_function
            )
            if reverse_snapshot is not None:
                self._active_reverse_snapshot = reverse_snapshot
                try:
                    if incumbent_cost is not None:
                        reverse_cost_lower_bounds = (
                            self._reverse_cost_lower_bounds(
                                goal, cost_function
                            )
                        )
                        if (
                            shortest_for_lagrangian is not None
                            and shortest_for_lagrangian > 0.0
                        ):
                            base_lambda = (
                                incumbent_cost / shortest_for_lagrangian
                            )
                            if math.isfinite(base_lambda) and base_lambda > 0.0:
                                for lambda_value in (
                                    base_lambda,
                                    base_lambda * 4.0,
                                ):
                                    if (
                                        not math.isfinite(lambda_value)
                                        or lambda_value <= 0.0
                                    ):
                                        continue
                                    potential = (
                                        self._reverse_augmented_cost_lower_bounds(
                                            goal,
                                            cost_function,
                                            lambda_value,
                                            resource_kind=resource_kind,
                                        )
                                    )
                                    if potential is not None:
                                        lagrangian_potentials.append(
                                            (
                                                lambda_value,
                                                potential,
                                                reverse_snapshot.stamp,
                                            )
                                        )
                    reverse_lower_bounds = (
                        self._reverse_duration_lower_bounds(goal, cap)
                        if resource_kind == "duration"
                        else self._reverse_distance_lower_bounds(goal, cap)
                    )
                finally:
                    self._active_reverse_snapshot = None

        if (
            reverse_snapshot is not None
            and self.graph._heuristic_cache_stamp() != reverse_snapshot.stamp
        ):
            reverse_snapshot = None
            reverse_cost_lower_bounds = None
            reverse_lower_bounds = None
            lagrangian_potentials = []
        use_reverse_cost_pruning = (
            reverse_snapshot is not None
            and reverse_cost_lower_bounds is not None
        )
        use_reverse_resource_pruning = (
            reverse_snapshot is not None and reverse_lower_bounds is not None
        )
        # Duration search is ordered directly by cost; a cost-per-distance
        # scan is neither admissible nor useful for that resource.
        minimum_cost_per_km = (
            self._minimum_cost_per_km(cost_function)
            if resource_kind == "distance"
            and self._reverse_cost_eligible(cost_function)
            else 0.0
        )
        # A distance/geodesic lower bound is not a safe duration bound:
        # deliberately disable it for the duration resource.
        use_geodesic_pruning = (
            resource_kind == "distance"
            and self._edge_distances_are_geodesic_lower_bounds()
        )
        start_to_goal_km = self._haversine(
            start.lat, start.lon, goal.lat, goal.lon
        )
        labels: Dict[int, _PathLabel] = {
            0: _PathLabel(0, start.id, 0.0, 0.0, 0.0, None, None)
        }
        labels_at_node: Dict[str, List[int]] = {start.id: [0]}
        active_labels = {0}
        frontier: List[Tuple[float, float, int]] = [
            (minimum_cost_per_km * start_to_goal_km, 0.0, 0)
        ]
        next_label_id = 1
        search_stamp = self.graph._heuristic_cache_stamp()
        if "get_edges" in getattr(self.graph, "__dict__", {}):
            edge_iterator = self.graph.get_edges
        else:
            edge_iterator = getattr(self.graph, "iter_edges", None)
            if not callable(edge_iterator):
                edge_iterator = self.graph.get_edges
        builtin_cost = self._reverse_cost_eligible(cost_function)
        avoid_highways = self._avoids_highways(cost_function)
        validate_nonnegative = self._validated_nonnegative
        validate_duration = self._edge_duration_minutes
        calculate_cost = cost_function.calculate
        expanded = 0
        while frontier:
            _check_active_deadline_at(expanded)
            expanded += 1
            current_stamp = self.graph._heuristic_cache_stamp()
            if current_stamp != search_stamp:
                # A snapshot from an older graph epoch is never mixed with
                # newer labels.  Disable every reverse bound atomically.
                reverse_cost_lower_bounds = None
                reverse_lower_bounds = None
                lagrangian_potentials = []
                use_reverse_cost_pruning = False
                use_reverse_resource_pruning = False
                minimum_cost_per_km = 0.0
                use_geodesic_pruning = False
                search_stamp = current_stamp
            _, _, label_id = heapq.heappop(frontier)
            if label_id not in active_labels:
                continue
            if lagrangian_potentials:
                current_stamp = self.graph._heuristic_cache_stamp()
                lagrangian_potentials = [
                    entry
                    for entry in lagrangian_potentials
                    if entry[2] == current_stamp
                ]
            label = labels[label_id]

            # A label can have become dominated after it was queued.
            node_labels = labels_at_node.get(label.node_id, [])
            if any(
                other_id != label_id
                and other_id in active_labels
                and self._label_dominates(labels[other_id], label, resource_kind)
                for other_id in node_labels
            ):
                active_labels.remove(label_id)
                continue
            if use_reverse_cost_pruning:
                reverse_cost_lower_bound = reverse_cost_lower_bounds.get(
                    label.node_id
                )
                if (
                    reverse_cost_lower_bound is None
                    or not math.isfinite(reverse_cost_lower_bound)
                    or self._cost_bound_exceeds_incumbent(
                        label.cumulative_cost,
                        reverse_cost_lower_bound,
                        incumbent_cost,
                    )
                ):
                    continue
            if lagrangian_potentials and incumbent_cost is not None:
                lagrangian_prune = False
                valid_potentials: List[
                    Tuple[float, Dict[str, float], object]
                ] = []
                for (
                    lambda_value,
                    potential,
                    potential_stamp,
                ) in lagrangian_potentials:
                    augmented_lower_bound = potential.get(label.node_id)
                    if augmented_lower_bound is None:
                        continue
                    if not math.isfinite(augmented_lower_bound):
                        continue
                    label_resource = (
                        label.cumulative_duration_minutes
                        if resource_kind == "duration"
                        else label.cumulative_distance_km
                    )
                    lambda_resource = lambda_value * label_resource
                    lambda_cap = lambda_value * cap
                    if (
                        not math.isfinite(lambda_resource)
                        or not math.isfinite(lambda_cap)
                    ):
                        continue
                    lagrangian_exceeds = (
                        self._lagrangian_bound_exceeds_incumbent(
                            label.cumulative_cost,
                            augmented_lower_bound,
                            label_resource,
                            lambda_value,
                            cap,
                            incumbent_cost,
                        )
                    )
                    if lagrangian_exceeds is None:
                        continue
                    valid_potentials.append(
                        (lambda_value, potential, potential_stamp)
                    )
                    if lagrangian_exceeds:
                        lagrangian_prune = True
                        break
                lagrangian_potentials = valid_potentials
                if lagrangian_prune:
                    continue


            if label.node_id == goal.id:
                path: List[Edge] = []
                current = label
                while current.incoming_edge is not None:
                    path.append(current.incoming_edge)
                    predecessor_id = current.predecessor_label_id
                    assert predecessor_id is not None
                    current = labels[predecessor_id]
                path.reverse()
                return path
            if use_reverse_resource_pruning:
                reverse_lower_bound = reverse_lower_bounds.get(label.node_id)
                label_resource = (
                    label.cumulative_duration_minutes
                    if resource_kind == "duration"
                    else label.cumulative_distance_km
                )
                if (
                    reverse_lower_bound is None
                    or self._resource_bound_exceeds_cap(
                        label_resource,
                        reverse_lower_bound,
                        cap,
                    )
                ):
                    continue
            elif use_geodesic_pruning:
                current_node = self.graph.get_node(label.node_id)
                straight_line_to_goal = self._haversine(
                    current_node.lat, current_node.lon, goal.lat, goal.lon
                )
                if (
                    label.cumulative_distance_km + straight_line_to_goal
                    > cap
                ):
                    continue

            for edge in edge_iterator(label.node_id):
                if builtin_cost:
                    if avoid_highways and is_highway_road_type(
                        edge.road_type
                    ):
                        continue
                elif not self._edge_is_eligible(edge, cost_function):
                    continue
                edge_distance_km = validate_nonnegative(
                    edge.distance_km, "edge distance_km"
                )
                edge_duration_minutes = validate_duration(edge)
                next_distance = label.cumulative_distance_km + edge_distance_km
                next_duration = (
                    label.cumulative_duration_minutes + edge_duration_minutes
                )
                if not math.isfinite(next_distance) or not math.isfinite(
                    next_duration
                ):
                    raise ValueError(
                        "cumulative traversed resource must be finite and non-negative"
                    )
                next_resource = (
                    next_duration
                    if resource_kind == "duration"
                    else next_distance
                )
                if next_resource > cap:
                    continue
                neighbor_to_goal = 0.0
                if use_geodesic_pruning:
                    neighbor = self.graph.get_node(edge.end_node_id)
                    neighbor_to_goal = self._haversine(
                        neighbor.lat, neighbor.lon, goal.lat, goal.lon
                    )
                if use_reverse_resource_pruning:
                    reverse_lower_bound = reverse_lower_bounds.get(
                        edge.end_node_id
                    )
                    if (
                        reverse_lower_bound is None
                        or self._resource_bound_exceeds_cap(
                            next_resource,
                            reverse_lower_bound,
                            cap,
                        )
                    ):
                        continue
                elif use_geodesic_pruning:
                    if next_distance + neighbor_to_goal > cap:
                        continue
                edge_cost = validate_nonnegative(
                    calculate_cost(edge), "edge calculated cost"
                )
                next_cost = label.cumulative_cost + edge_cost
                if not math.isfinite(next_cost):
                    raise ValueError(
                        "cumulative calculated cost must be finite and non-negative"
                    )
                if use_reverse_cost_pruning:
                    reverse_cost_lower_bound = reverse_cost_lower_bounds.get(
                        edge.end_node_id
                    )
                    if (
                        reverse_cost_lower_bound is None
                        or not math.isfinite(reverse_cost_lower_bound)
                        or self._cost_bound_exceeds_incumbent(
                            next_cost,
                            reverse_cost_lower_bound,
                            incumbent_cost,
                        )
                    ):
                        continue
                if lagrangian_potentials and incumbent_cost is not None:
                    lagrangian_prune = False
                    valid_potentials = []
                    current_stamp = self.graph._heuristic_cache_stamp()
                    for (
                        lambda_value,
                        potential,
                        potential_stamp,
                    ) in lagrangian_potentials:
                        if potential_stamp != current_stamp:
                            continue
                        augmented_lower_bound = potential.get(
                            edge.end_node_id
                        )
                        if augmented_lower_bound is None:
                            continue
                        if not math.isfinite(augmented_lower_bound):
                            continue
                        edge_resource = (
                            next_duration
                            if resource_kind == "duration"
                            else next_distance
                        )
                        lambda_resource = lambda_value * edge_resource
                        lambda_cap = lambda_value * cap
                        if (
                            not math.isfinite(lambda_resource)
                            or not math.isfinite(lambda_cap)
                        ):
                            continue
                        lagrangian_exceeds = (
                            self._lagrangian_bound_exceeds_incumbent(
                                next_cost,
                                augmented_lower_bound,
                                edge_resource,
                                lambda_value,
                                cap,
                                incumbent_cost,
                            )
                        )
                        if lagrangian_exceeds is None:
                            continue
                        valid_potentials.append(
                            (lambda_value, potential, potential_stamp)
                        )
                        if lagrangian_exceeds:
                            lagrangian_prune = True
                            break
                    lagrangian_potentials = valid_potentials
                    if lagrangian_prune:
                        continue

                candidate = _PathLabel(
                    next_label_id,
                    edge.end_node_id,
                    next_distance,
                    next_duration,
                    next_cost,
                    label_id,
                    edge,
                )
                existing_ids = labels_at_node.get(edge.end_node_id, [])
                candidate_resource = (
                    candidate.cumulative_duration_minutes
                    if resource_kind == "duration"
                    else candidate.cumulative_distance_km
                )
                candidate_dominated = False
                candidate_dominators: List[int] = []
                for existing_id in existing_ids:
                    if existing_id not in active_labels:
                        continue
                    existing = labels[existing_id]
                    existing_resource = (
                        existing.cumulative_duration_minutes
                        if resource_kind == "duration"
                        else existing.cumulative_distance_km
                    )
                    if (
                        existing_resource == candidate_resource
                        and existing.cumulative_cost
                        == candidate.cumulative_cost
                    ):
                        candidate_dominated = True
                        break
                    if self._label_dominates(
                        existing, candidate, resource_kind
                    ):
                        candidate_dominated = True
                        break
                    if self._label_dominates(
                        candidate, existing, resource_kind
                    ):
                        candidate_dominators.append(existing_id)
                if candidate_dominated:
                    continue
                for existing_id in candidate_dominators:
                    active_labels.remove(existing_id)
                labels_at_node[edge.end_node_id] = [
                    existing_id
                    for existing_id in existing_ids
                    if existing_id in active_labels
                ]
                labels[next_label_id] = candidate
                labels_at_node[edge.end_node_id].append(next_label_id)
                active_labels.add(next_label_id)
                estimated_total_cost = next_cost
                if use_geodesic_pruning:
                    estimated_total_cost += minimum_cost_per_km * neighbor_to_goal
                heapq.heappush(
                    frontier, (estimated_total_cost, next_cost, next_label_id)
                )
                next_label_id += 1

        return None


    @staticmethod
    def _edge_duration_minutes(edge: Edge) -> float:
        return ScenicRoutePlanner._validated_nonnegative(
            edge.travel_time_minutes, "edge travel_time_minutes"
        )

    @classmethod
    def _path_duration_minutes(cls, edges: List[Edge]) -> float:
        return float(sum(cls._edge_duration_minutes(edge) for edge in edges))

    @staticmethod
    def _path_distance_km(edges: List[Edge]) -> float:
        return float(sum(edge.distance_km for edge in edges))

    def _reconstruct_path(
        self, came_from: Dict[str, Tuple[str, Edge]], goal_id: str
    ) -> List[Edge]:
        edges: List[Edge] = []
        node_id = goal_id
        while node_id in came_from:
            prev_node, edge = came_from[node_id]
            edges.append(edge)
            node_id = prev_node
        edges.reverse()
        return edges

    def _path_to_route(
        self,
        edges: List[Edge],
        *,
        start_node: Optional[Node] = None,
        goal_node: Optional[Node] = None,
        evaluation: object = None,
        fastest_duration_minutes: Optional[float] = None,
        requested_max_detour_factor: float = 1.0,
        exact: bool = False,
        exactness_status: str = "uncertified",
        optimality_gap: Optional[float] = None,
        certified_upper_bound: Optional[float] = None,
        algorithm: str = "uncertified-production-search",
        zero_improvement_reason: Optional[str] = None,
        search_diagnostics: Optional[Dict[str, object]] = None,
    ) -> Route:
        _check_active_deadline()
        if evaluation is None:
            fallback_fastest = (
                self._path_duration_minutes(edges)
                if fastest_duration_minutes is None
                else float(fastest_duration_minutes)
            )
            evaluation = evaluate_path(
                edges,
                q=0.0,
                kappa=requested_max_detour_factor,
                fastest_duration_minutes=fallback_fastest,
                check_cancelled=_check_active_deadline,
            )
        total_distance_km = float(getattr(evaluation, "total_distance_km"))
        total_minutes = float(getattr(evaluation, "duration_minutes"))
        raw_scenic = float(getattr(evaluation, "raw_scenic_score"))
        normalized_scenic = float(
            getattr(evaluation, "normalized_scenic_score")
        )
        evaluation_edge_ids = tuple(getattr(evaluation, "edge_ids"))
        score_run = tuple(getattr(evaluation, "score_run"))
        duration_utility = float(getattr(evaluation, "duration_utility"))
        objective_value = float(getattr(evaluation, "objective"))
        score_values = [float(score) for _, score in score_run]
        if len(score_values) != len(edges):
            raise ValueError("path evaluation score run does not match edges")
        del evaluation_edge_ids
        fastest = (
            total_minutes
            if fastest_duration_minutes is None
            else float(fastest_duration_minutes)
        )
        cap = fastest * float(requested_max_detour_factor)
        if fastest > 0.0:
            actual_ratio = total_minutes / fastest
        else:
            actual_ratio = 1.0 if total_minutes == 0.0 else float("inf")

        segments: List[RouteSegment] = []
        canonical_edge_ids: List[str] = []
        traversal_ids: List[str] = []
        waypoints: List[Tuple[float, float]] = []
        normalized_score_run: List[Tuple[str, float]] = []
        if edges:
            start_node = start_node or self.graph.get_node(edges[0].start_node_id)
            goal_node = goal_node or self.graph.get_node(edges[-1].end_node_id)
            if not self._simple_edge_path(edges):
                raise ValueError("route edge traversal is disconnected or cyclic")
            if (
                edges[0].start_node_id != start_node.id
                or edges[-1].end_node_id != goal_node.id
            ):
                raise ValueError("route edge traversal has invalid endpoints")
        elif start_node is None or goal_node is None:
            raise ValueError("empty path requires start and goal nodes")
        if not edges:
            waypoints = [
                (start_node.lat, start_node.lon),
                (goal_node.lat, goal_node.lon),
            ]
        for index, edge in enumerate(edges):
            _check_active_deadline_at(index)
            start_node = self.graph.get_node(edge.start_node_id)
            end_node = self.graph.get_node(edge.end_node_id)
            start_coordinate = getattr(
                edge,
                "route_start_coordinate",
                (start_node.lat, start_node.lon),
            )
            end_coordinate = getattr(
                edge,
                "route_end_coordinate",
                (end_node.lat, end_node.lon),
            )
            direction = str(
                getattr(
                    edge,
                    "direction",
                    "reverse"
                    if bool(getattr(edge, "_is_reverse_traversal", False))
                    else "forward",
                )
            )
            canonical_id = str(
                getattr(
                    edge,
                    "canonical_edge_id",
                    getattr(edge, "_canonical_edge_id", edge.id),
                )
            )
            traversal_id = f"{index}:{direction}:{canonical_id}"
            canonical_edge_ids.append(canonical_id)
            traversal_ids.append(traversal_id)
            normalized_score_run.append((traversal_id, score_values[index]))
            segment = RouteSegment(
                start=(
                    float(start_coordinate[0]),
                    float(start_coordinate[1]),
                ),
                end=(
                    float(end_coordinate[0]),
                    float(end_coordinate[1]),
                ),
                distance_km=float(edge.distance_km),
                scenic_score=score_values[index],
                road_name=edge.road_name,
                road_type=edge.road_type,
                edge_id=canonical_id,
                direction=direction,
                traversal_id=traversal_id,
                duration_minutes=self._edge_duration_minutes(edge),
            )
            segment.source_edge_id = getattr(edge, "canonical_edge_id", None)
            segment.source_fraction = getattr(edge, "source_fraction", None)
            segments.append(segment)
            if not waypoints:
                waypoints.append(
                    (float(start_coordinate[0]), float(start_coordinate[1]))
                )
            waypoints.append(
                (float(end_coordinate[0]), float(end_coordinate[1]))
            )

        edge_ids = tuple(canonical_edge_ids)
        score_run = tuple(normalized_score_run)
        return Route(
            segments=segments,
            total_distance_km=total_distance_km,
            average_scenic_score=raw_scenic,
            estimated_duration_minutes=total_minutes,
            waypoints=waypoints,
            edge_ids=edge_ids,
            traversal_ids=tuple(traversal_ids),
            raw_scenic_score=raw_scenic,
            normalized_scenic_score=normalized_scenic,
            duration_utility=duration_utility,
            objective_value=objective_value,
            fastest_duration_minutes=fastest,
            requested_max_detour_factor=float(requested_max_detour_factor),
            applied_max_detour_factor=float(requested_max_detour_factor),
            duration_cap_minutes=cap,
            actual_duration_ratio=actual_ratio,
            exact=bool(exact),
            exactness_status=str(exactness_status),
            optimality_gap=(
                None if optimality_gap is None else float(optimality_gap)
            ),
            certified_upper_bound=(
                None
                if certified_upper_bound is None
                else float(certified_upper_bound)
            ),
            highway_count=int(getattr(evaluation, "highway_count")),
            score_coverage=float(getattr(evaluation, "score_coverage")),
            score_run=score_run,
            algorithm=str(algorithm),
            zero_improvement_reason=zero_improvement_reason,
            no_route_reason=None,
            normalization_version=str(
                getattr(
                    evaluation,
                    "normalization_version",
                    SCENIC_NORMALIZATION_VERSION,
                )
            ),
            search_diagnostics=_normalize_search_diagnostics(search_diagnostics),
        )

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import asin, cos, radians, sin, sqrt

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return 6371.0 * c
