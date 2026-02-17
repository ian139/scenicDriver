from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Tuple

from .cost import ScenicCostFunction
from .graph import Edge, Node, RoadGraph


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


class ScenicRoutePlanner:
    def __init__(
        self,
        graph: Optional[RoadGraph] = None,
        cost_function: Optional[ScenicCostFunction] = None,
    ) -> None:
        self.graph = graph
        self.cost_function = cost_function or ScenicCostFunction()

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

        self.cost_function.scenic_weight = float(min(max(scenic_weight, 0.0), 1.0))
        self.cost_function.avoid_highways = bool(avoid_highways)

        start_node = self.graph.find_nearest_node(*start)
        end_node = self.graph.find_nearest_node(*end)

        path_edges = self._a_star(start_node, end_node, max_detour_factor=max_detour_factor)
        if path_edges is None:
            raise ValueError("No route found between the given coordinates.")
        return self._path_to_route(path_edges)

    def _a_star(
        self,
        start: Node,
        goal: Node,
        *,
        max_detour_factor: float,
    ) -> Optional[List[Edge]]:
        direct_km = self._haversine(start.lat, start.lon, goal.lat, goal.lon)
        max_path_km = max(0.2, direct_km * max_detour_factor)

        frontier: List[Tuple[float, float, str]] = []
        heapq.heappush(frontier, (0.0, 0.0, start.id))  # (priority, cost_so_far, node_id)

        came_from: Dict[str, Tuple[str, Edge]] = {}
        best_cost: Dict[str, float] = {start.id: 0.0}
        best_distance_km: Dict[str, float] = {start.id: 0.0}

        while frontier:
            _, current_cost, current_id = heapq.heappop(frontier)
            if current_id == goal.id:
                return self._reconstruct_path(came_from, goal.id)
            if current_cost > best_cost.get(current_id, float("inf")):
                continue

            current_node = self.graph.get_node(current_id)
            for edge in self.graph.get_edges(current_id):
                neighbor_id = edge.end_node_id
                next_distance = best_distance_km[current_id] + edge.distance_km
                if next_distance > max_path_km:
                    continue

                edge_cost = self.cost_function.calculate(edge)
                next_cost = current_cost + edge_cost
                if next_cost >= best_cost.get(neighbor_id, float("inf")):
                    continue

                best_cost[neighbor_id] = next_cost
                best_distance_km[neighbor_id] = next_distance
                came_from[neighbor_id] = (current_id, edge)

                neighbor = self.graph.get_node(neighbor_id)
                # Mild admissible-ish heuristic: lower-bounded by travel-time-style term.
                h = self._haversine(neighbor.lat, neighbor.lon, goal.lat, goal.lon)
                alpha = 1.0 - self.cost_function.scenic_weight
                heuristic = alpha * h
                heapq.heappush(frontier, (next_cost + heuristic, next_cost, neighbor_id))

        return None

    def _reconstruct_path(self, came_from: Dict[str, Tuple[str, Edge]], goal_id: str) -> List[Edge]:
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

        avg_scenic = scenic_distance_sum / total_distance_km if total_distance_km > 0 else 0.0
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
