from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Optional, Tuple

from .cost import CostWeights, ScenicCostFunction
from .graph import Edge, Node, RoadGraph

_ORIGINAL_SCENIC_CALCULATE = ScenicCostFunction.calculate
_ORIGINAL_SCENIC_ROAD_TYPE_ADJUSTMENT = (
    ScenicCostFunction._road_type_adjustment
)


@dataclass
class RouteSegment:
    start: Tuple[float, float]  # (lat, lon)
    end: Tuple[float, float]
    distance_km: float
    scenic_score: float
    road_name: Optional[str]
    road_type: str


@dataclass
class Route:
    segments: List[RouteSegment]
    total_distance_km: float
    average_scenic_score: float
    estimated_duration_minutes: float
    waypoints: List[Tuple[float, float]]


@dataclass
class _PathLabel:
    label_id: int
    node_id: str
    cumulative_distance_km: float
    cumulative_cost: float
    predecessor_label_id: Optional[int]
    incoming_edge: Optional[Edge]


class ScenicRoutePlanner:
    def __init__(
        self,
        graph: Optional[RoadGraph] = None,
        cost_function: Optional[ScenicCostFunction] = None,
    ) -> None:
        self.graph = graph
        self.cost_function = cost_function or ScenicCostFunction()
        # Heuristic scans are graph-state dependent.  Keep the graph reference
        # alongside each stamp so identity checks remain safe if ``self.graph``
        # is replaced, and retain one ratio entry per built-in cost signature.
        self._geodesic_lower_bounds_cache: Optional[
            Tuple[RoadGraph, object, bool]
        ] = None
        self._minimum_cost_per_km_cache: Dict[
            Tuple[int, object, Tuple[object, ...]], Tuple[RoadGraph, float]
        ] = {}
        self._minimum_cost_per_km_cache_context: Optional[
            Tuple[RoadGraph, object]
        ] = None

    def _make_cost_function(self, scenic_weight: float) -> ScenicCostFunction:
        return ScenicCostFunction(
            scenic_weight=scenic_weight,
            avoid_highways=self.cost_function.avoid_highways,
            weights=self.cost_function.weights,
        )

    def find_scenic_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        max_detour_factor: float = 1.8,
    ) -> Route:
        if self.graph is None:
            raise RuntimeError("Road graph not loaded")
        if max_detour_factor < 1.0:
            raise ValueError("max_detour_factor must be >= 1.0")

        scenic_weight = float(min(max(scenic_weight, 0.0), 1.0))
        self.cost_function.avoid_highways = bool(avoid_highways)

        if scenic_weight <= 0.0:
            return self.find_fastest_route(start, end, avoid_highways=avoid_highways)

        start_node = self.graph.find_nearest_node(*start)
        end_node = self.graph.find_nearest_node(*end)

        shortest_cost = self._make_cost_function(0.0)
        shortest_edges = self._a_star(
            start_node,
            end_node,
            cost_function=shortest_cost,
            max_path_km=None,
        )
        if shortest_edges is None:
            raise ValueError("No route found between the given coordinates.")

        shortest_km = self._path_distance_km(shortest_edges)
        detour_cap = float(max_detour_factor)
        effective_detour = 1.0 + scenic_weight * (detour_cap - 1.0)
        effective_detour = max(1.2, effective_detour)
        max_path_km = max(0.2, shortest_km * effective_detour)

        scenic_cost = self._make_cost_function(scenic_weight)
        scenic_upper_bound = sum(
            scenic_cost.calculate(edge) for edge in shortest_edges
        )
        if not math.isfinite(scenic_upper_bound):
            scenic_upper_bound = None
        path_edges = self._a_star(
            start_node,
            end_node,
            cost_function=scenic_cost,
            max_path_km=max_path_km,
            max_feasible_cost=scenic_upper_bound,
            shortest_distance_km=shortest_km,
        )
        if path_edges is None:
            raise ValueError("No route found between the given coordinates.")
        return self._path_to_route(path_edges)

    def find_fastest_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        avoid_highways: bool = False,
    ) -> Route:
        if self.graph is None:
            raise RuntimeError("Road graph not loaded")
        self.cost_function.avoid_highways = bool(avoid_highways)
        start_node = self.graph.find_nearest_node(*start)
        end_node = self.graph.find_nearest_node(*end)
        shortest_cost = self._make_cost_function(0.0)
        shortest_edges = self._a_star(
            start_node,
            end_node,
            cost_function=shortest_cost,
            max_path_km=None,
        )
        if shortest_edges is None:
            raise ValueError("No route found between the given coordinates.")
        return self._path_to_route(shortest_edges)

    def _a_star(
        self,
        start: Node,
        goal: Node,
        *,
        cost_function: ScenicCostFunction,
        max_path_km: float | None,
        max_feasible_cost: float | None = None,
        shortest_distance_km: float | None = None,
    ) -> Optional[List[Edge]]:
        if max_path_km is not None:
            return self._resource_constrained_path(
                start,
                goal,
                cost_function=cost_function,
                max_path_km=max_path_km,
                max_feasible_cost=max_feasible_cost,
                shortest_distance_km=shortest_distance_km,
            )

        minimum_cost_per_km = self._minimum_cost_per_km(cost_function)
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

        while frontier:
            _, current_cost, current_id = heapq.heappop(frontier)
            if current_id == goal.id:
                return self._reconstruct_path(came_from, goal.id)
            if current_cost > best_cost.get(current_id, float("inf")):
                continue

            for edge in self.graph.get_edges(current_id):
                neighbor_id = edge.end_node_id
                next_distance = best_distance_km[current_id] + edge.distance_km
                if max_path_km is not None and next_distance > max_path_km:
                    continue

                edge_cost = cost_function.calculate(edge)
                next_cost = current_cost + edge_cost
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

        return None

    @staticmethod
    def _label_dominates(first: _PathLabel, second: _PathLabel) -> bool:
        """Return whether ``first`` is strictly better in at least one resource."""
        return (
            first.cumulative_distance_km <= second.cumulative_distance_km
            and first.cumulative_cost <= second.cumulative_cost
            and (
                first.cumulative_distance_km < second.cumulative_distance_km
                or first.cumulative_cost < second.cumulative_cost
            )
        )

    @staticmethod
    def _built_in_cost_signature(
        cost_function: ScenicCostFunction,
    ) -> Optional[Tuple[object, ...]]:
        """Return immutable public inputs for the exact built-in cost function."""
        if type(cost_function) is not ScenicCostFunction:
            return None
        if type(cost_function.weights) is not CostWeights:
            return None
        if "calculate" in cost_function.__dict__:
            return None
        if "_road_type_adjustment" in cost_function.__dict__:
            return None
        weights = cost_function.weights
        return (
            float(cost_function.scenic_weight),
            bool(cost_function.avoid_highways),
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
        try:
            return (
                ScenicRoutePlanner._built_in_cost_signature(cost_function)
                is not None
                and type(cost_function).calculate is _ORIGINAL_SCENIC_CALCULATE
                and type(cost_function)._road_type_adjustment
                is _ORIGINAL_SCENIC_ROAD_TYPE_ADJUSTMENT
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def _edge_distances_are_geodesic_lower_bounds(self) -> bool:
        """Return whether stored traversable distances cover endpoint geodesics."""
        graph = self.graph
        stamp = graph._heuristic_cache_stamp()
        cached = self._geodesic_lower_bounds_cache
        if cached is not None:
            cached_graph, cached_stamp, cached_result = cached
            if cached_graph is graph and cached_stamp == stamp:
                return cached_result

        tolerance_km = 1e-9
        result = True
        for edge in graph.edges.values():
            distance_km = float(edge.distance_km)
            start = graph.get_node(edge.start_node_id)
            end = graph.get_node(edge.end_node_id)
            endpoint_km = self._haversine(start.lat, start.lon, end.lat, end.lon)
            if endpoint_km > tolerance_km and not (
                distance_km + tolerance_km >= endpoint_km
            ):
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
        """Return a safe graph-wide nonnegative cost-per-kilometre lower bound.

        The bound is only admissible when every positive-distance edge stores at
        least its endpoint geodesic distance.  If that invariant is not proven,
        return zero so A* remains correct for imported or manually-mutated
        graphs.
        """
        graph = self.graph
        stamp = graph._heuristic_cache_stamp()
        cache_context = self._minimum_cost_per_km_cache_context
        if (
            cache_context is None
            or cache_context[0] is not graph
            or cache_context[1] != stamp
        ):
            self._minimum_cost_per_km_cache = {}
            self._minimum_cost_per_km_cache_context = (graph, stamp)
        signature = self._built_in_cost_signature(cost_function)
        cache_key = None
        if signature is not None:
            cache_key = (id(graph), stamp, signature)
            cached = self._minimum_cost_per_km_cache.get(cache_key)
            if cached is not None and cached[0] is graph:
                return cached[1]

        if not self._edge_distances_are_geodesic_lower_bounds():
            minimum_cost_per_km = 0.0
        else:
            minimum_cost_per_km = float("inf")
            for edge in graph.edges.values():
                distance_km = float(edge.distance_km)
                if distance_km <= 0.0:
                    continue
                edge_cost = max(float(cost_function.calculate(edge)), 0.0)
                ratio = edge_cost / distance_km
                if ratio < minimum_cost_per_km:
                    minimum_cost_per_km = ratio
            if minimum_cost_per_km == float("inf"):
                minimum_cost_per_km = 0.0

        # Recheck after the complete ratio/geometry work.  A changed graph
        # invalidates the observed positive bound; custom cost functions
        # intentionally bypass this cache.
        final_stamp = graph._heuristic_cache_stamp()
        if graph is not self.graph or final_stamp != stamp:
            self._minimum_cost_per_km_cache = {}
            self._minimum_cost_per_km_cache_context = None
            return 0.0
        if signature is not None:
            assert cache_key is not None
            self._minimum_cost_per_km_cache[cache_key] = (
                graph,
                minimum_cost_per_km,
            )
        return minimum_cost_per_km

    def _reverse_distance_lower_bounds(
        self,
        goal: Node,
        max_path_km: float,
    ) -> Optional[Dict[str, float]]:
        """Return exact distance-to-goal bounds for the current traversal graph.

        The predecessor multigraph is built from ``get_edges`` rather than the
        stored edge table so generated reverse views have exactly the same
        directed semantics as the forward search.  A malformed resource or
        traversal distance disables this optimization and lets the existing
        conservative pruning remain in force.
        """
        graph = self.graph
        try:
            cap = float(max_path_km)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(cap) or cap < 0.0:
            return None

        stamp = graph._heuristic_cache_stamp()
        predecessors: Dict[str, List[Tuple[str, float]]] = {}
        for node_id in graph.nodes:
            for edge in graph.get_edges(node_id):
                try:
                    distance_km = float(edge.distance_km)
                except (TypeError, ValueError, OverflowError):
                    return None
                if not math.isfinite(distance_km) or distance_km < 0.0:
                    return None
                predecessors.setdefault(edge.end_node_id, []).append(
                    (node_id, distance_km)
                )

        # A graph mutation during enumeration makes the snapshot unsafe.
        if graph is not self.graph or graph._heuristic_cache_stamp() != stamp:
            return None

        lower_bounds: Dict[str, float] = {goal.id: 0.0}
        frontier: List[Tuple[float, str]] = [(0.0, goal.id)]
        while frontier:
            distance_km, node_id = heapq.heappop(frontier)
            if lower_bounds.get(node_id) != distance_km:
                continue
            for predecessor_id, edge_distance_km in predecessors.get(
                node_id, ()
            ):
                next_distance_km = distance_km + edge_distance_km
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
        if graph is not self.graph or graph._heuristic_cache_stamp() != stamp:
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

        graph = self.graph
        stamp = graph._heuristic_cache_stamp()
        predecessors: Dict[str, List[Tuple[str, float]]] = {}
        for node_id in graph.nodes:
            try:
                edges = graph.get_edges(node_id)
            except Exception:
                return None
            for edge in edges:
                try:
                    edge_cost = float(cost_function.calculate(edge))
                except Exception:
                    return None
                if not math.isfinite(edge_cost) or edge_cost < 0.0:
                    return None
                predecessors.setdefault(edge.end_node_id, []).append(
                    (node_id, edge_cost)
                )

        if graph is not self.graph or graph._heuristic_cache_stamp() != stamp:
            return None

        lower_bounds: Dict[str, float] = {goal.id: 0.0}
        frontier: List[Tuple[float, str]] = [(0.0, goal.id)]
        while frontier:
            cost, node_id = heapq.heappop(frontier)
            if lower_bounds.get(node_id) != cost:
                continue
            for predecessor_id, edge_cost in predecessors.get(node_id, ()):
                next_cost = cost + edge_cost
                # Overflow cannot improve any finite incumbent; leave this
                # predecessor absent so callers treat it as no completion.
                if not math.isfinite(next_cost):
                    continue
                previous_cost = lower_bounds.get(predecessor_id)
                if previous_cost is None or next_cost < previous_cost:
                    lower_bounds[predecessor_id] = next_cost
                    heapq.heappush(frontier, (next_cost, predecessor_id))

        if graph is not self.graph or graph._heuristic_cache_stamp() != stamp:
            return None
        return lower_bounds

    def _reverse_augmented_cost_lower_bounds(
        self,
        goal: Node,
        cost_function: ScenicCostFunction,
        lambda_value: float,
    ) -> Optional[Dict[str, float]]:
        """Return a local reverse potential for ``cost + lambda * distance``."""
        if not self._reverse_cost_eligible(cost_function):
            return None
        try:
            lagrangian_lambda = float(lambda_value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(lagrangian_lambda) or lagrangian_lambda < 0.0:
            return None

        graph = self.graph
        stamp = graph._heuristic_cache_stamp()
        predecessors: Dict[str, List[Tuple[str, float]]] = {}
        try:
            for node_id in graph.nodes:
                for edge in graph.get_edges(node_id):
                    distance_km = float(edge.distance_km)
                    edge_cost = float(cost_function.calculate(edge))
                    if (
                        not math.isfinite(distance_km)
                        or distance_km < 0.0
                        or not math.isfinite(edge_cost)
                        or edge_cost < 0.0
                    ):
                        return None
                    distance_term = lagrangian_lambda * distance_km
                    augmented_cost = edge_cost + distance_term
                    if (
                        not math.isfinite(distance_term)
                        or distance_term < 0.0
                        or not math.isfinite(augmented_cost)
                        or augmented_cost < 0.0
                    ):
                        return None
                    predecessors.setdefault(edge.end_node_id, []).append(
                        (node_id, augmented_cost)
                    )
        except Exception:
            return None

        if graph is not self.graph or graph._heuristic_cache_stamp() != stamp:
            return None

        lower_bounds: Dict[str, float] = {goal.id: 0.0}
        frontier: List[Tuple[float, str]] = [(0.0, goal.id)]
        while frontier:
            augmented_cost, node_id = heapq.heappop(frontier)
            if lower_bounds.get(node_id) != augmented_cost:
                continue
            for predecessor_id, edge_augmented_cost in predecessors.get(
                node_id, ()
            ):
                next_augmented_cost = augmented_cost + edge_augmented_cost
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

        if graph is not self.graph or graph._heuristic_cache_stamp() != stamp:
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
        """Check a resource bound with a conservative round-off allowance."""
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
        max_path_km: float,
        max_feasible_cost: float | None = None,
        shortest_distance_km: float | None = None,
    ) -> Optional[List[Edge]]:
        """Find the least-cost path subject to an additive distance resource."""
        incumbent_cost: float | None = None
        if max_feasible_cost is not None:
            try:
                candidate_incumbent = float(max_feasible_cost)
            except (TypeError, ValueError, OverflowError):
                candidate_incumbent = float("nan")
            if math.isfinite(candidate_incumbent):
                incumbent_cost = candidate_incumbent
        reverse_cost_lower_bounds: Optional[Dict[str, float]] = None
        if incumbent_cost is not None:
            reverse_cost_lower_bounds = self._reverse_cost_lower_bounds(
                goal, cost_function
            )
        use_reverse_cost_pruning = (
            incumbent_cost is not None and reverse_cost_lower_bounds is not None
        )
        lagrangian_potentials: List[
            Tuple[float, Dict[str, float], object]
        ] = []
        if incumbent_cost is not None:
            try:
                cap_for_lagrangian = float(max_path_km)
                shortest_for_lagrangian = float(shortest_distance_km)
                base_lambda = (
                    incumbent_cost / shortest_for_lagrangian
                )
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                base_lambda = float("nan")
                cap_for_lagrangian = float("nan")
            if (
                math.isfinite(cap_for_lagrangian)
                and cap_for_lagrangian >= 0.0
                and math.isfinite(base_lambda)
                and base_lambda > 0.0
            ):
                for lambda_value in (base_lambda, base_lambda * 4.0):
                    if (
                        not math.isfinite(lambda_value)
                        or lambda_value <= 0.0
                    ):
                        continue
                    potential_stamp = self.graph._heuristic_cache_stamp()
                    potential = self._reverse_augmented_cost_lower_bounds(
                        goal, cost_function, lambda_value
                    )
                    if (
                        potential is not None
                        and self.graph is not None
                        and self.graph._heuristic_cache_stamp()
                        == potential_stamp
                    ):
                        lagrangian_potentials.append(
                            (lambda_value, potential, potential_stamp)
                        )
        reverse_lower_bounds = self._reverse_distance_lower_bounds(
            goal, max_path_km
        )
        use_reverse_resource_pruning = reverse_lower_bounds is not None
        minimum_cost_per_km = self._minimum_cost_per_km(cost_function)
        use_geodesic_pruning = self._edge_distances_are_geodesic_lower_bounds()
        start_to_goal_km = self._haversine(
            start.lat, start.lon, goal.lat, goal.lon
        )
        labels: Dict[int, _PathLabel] = {
            0: _PathLabel(0, start.id, 0.0, 0.0, None, None)
        }
        labels_at_node: Dict[str, List[int]] = {start.id: [0]}
        active_labels = {0}
        frontier: List[Tuple[float, float, int]] = [
            (minimum_cost_per_km * start_to_goal_km, 0.0, 0)
        ]
        next_label_id = 1

        while frontier:
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
                and self._label_dominates(labels[other_id], label)
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
                    lambda_distance = (
                        lambda_value * label.cumulative_distance_km
                    )
                    lambda_cap = lambda_value * max_path_km
                    if (
                        not math.isfinite(lambda_distance)
                        or not math.isfinite(lambda_cap)
                    ):
                        continue
                    lagrangian_exceeds = (
                        self._lagrangian_bound_exceeds_incumbent(
                            label.cumulative_cost,
                            augmented_lower_bound,
                            label.cumulative_distance_km,
                            lambda_value,
                            max_path_km,
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
                if (
                    reverse_lower_bound is None
                    or self._resource_bound_exceeds_cap(
                        label.cumulative_distance_km,
                        reverse_lower_bound,
                        max_path_km,
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
                    > max_path_km
                ):
                    continue

            for edge in self.graph.get_edges(label.node_id):
                next_distance = label.cumulative_distance_km + edge.distance_km
                if next_distance > max_path_km:
                    continue
                neighbor = self.graph.get_node(edge.end_node_id)
                neighbor_to_goal = 0.0
                if use_geodesic_pruning:
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
                            next_distance,
                            reverse_lower_bound,
                            max_path_km,
                        )
                    ):
                        continue
                elif use_geodesic_pruning:
                    if next_distance + neighbor_to_goal > max_path_km:
                        continue
                next_cost = label.cumulative_cost + cost_function.calculate(edge)
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
                        lambda_distance = lambda_value * next_distance
                        lambda_cap = lambda_value * max_path_km
                        if (
                            not math.isfinite(lambda_distance)
                            or not math.isfinite(lambda_cap)
                        ):
                            continue
                        lagrangian_exceeds = (
                            self._lagrangian_bound_exceeds_incumbent(
                                next_cost,
                                augmented_lower_bound,
                                next_distance,
                                lambda_value,
                                max_path_km,
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
                    next_cost,
                    label_id,
                    edge,
                )
                existing_ids = labels_at_node.get(edge.end_node_id, [])
                if any(
                    existing_id in active_labels
                    and labels[existing_id].cumulative_distance_km
                    == candidate.cumulative_distance_km
                    and labels[existing_id].cumulative_cost
                    == candidate.cumulative_cost
                    for existing_id in existing_ids
                ):
                    continue
                if any(
                    existing_id in active_labels
                    and self._label_dominates(labels[existing_id], candidate)
                    for existing_id in existing_ids
                ):
                    continue

                for existing_id in existing_ids:
                    if existing_id in active_labels and self._label_dominates(
                        candidate, labels[existing_id]
                    ):
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

    def _path_to_route(self, edges: List[Edge]) -> Route:
        segments: List[RouteSegment] = []
        total_distance_km = 0.0
        scenic_distance_sum = 0.0
        total_minutes = 0.0
        waypoints: List[Tuple[float, float]] = []

        for edge in edges:
            start_node = self.graph.get_node(edge.start_node_id)
            end_node = self.graph.get_node(edge.end_node_id)

            segments.append(
                RouteSegment(
                    start=(start_node.lat, start_node.lon),
                    end=(end_node.lat, end_node.lon),
                    distance_km=edge.distance_km,
                    scenic_score=edge.scenic_score,
                    road_name=edge.road_name,
                    road_type=edge.road_type,
                )
            )

            total_distance_km += edge.distance_km
            scenic_distance_sum += edge.scenic_score * edge.distance_km
            total_minutes += edge.travel_time_minutes

            if not waypoints:
                waypoints.append((start_node.lat, start_node.lon))
            waypoints.append((end_node.lat, end_node.lon))

        avg_scenic = (
            scenic_distance_sum / total_distance_km if total_distance_km > 0 else 0.0
        )
        return Route(
            segments=segments,
            total_distance_km=total_distance_km,
            average_scenic_score=avg_scenic,
            estimated_duration_minutes=total_minutes,
            waypoints=waypoints,
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
