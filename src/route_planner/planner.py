"""
Scenic Route Planner
Owner: progno-backend agent

A* pathfinding with scenic cost optimization.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq

from .graph import RoadGraph, Node, Edge
from .cost import ScenicCostFunction


@dataclass
class RouteSegment:
    """A segment of the planned route."""
    start: Tuple[float, float]  # (lat, lon)
    end: Tuple[float, float]
    distance_km: float
    scenic_score: float
    road_name: Optional[str]
    road_type: str


@dataclass
class Route:
    """A complete planned route."""
    segments: List[RouteSegment]
    total_distance_km: float
    average_scenic_score: float
    estimated_duration_minutes: float
    waypoints: List[Tuple[float, float]]

    @property
    def scenic_distance_ratio(self) -> float:
        """Ratio of scenic value to distance traveled."""
        if self.total_distance_km == 0:
            return 0
        return self.average_scenic_score / (self.total_distance_km / 100)


class ScenicRoutePlanner:
    """
    Plan routes that optimize for scenic beauty.

    Uses A* algorithm with a custom cost function that balances
    travel distance with scenic scores along the route.

    Target performance: < 3 seconds for 500km routes
    """

    def __init__(
        self,
        graph: Optional[RoadGraph] = None,
        cost_function: Optional[ScenicCostFunction] = None
    ):
        self.graph = graph
        self.cost_function = cost_function or ScenicCostFunction()

    def load_graph(self, region: str) -> None:
        """Load road graph for a region."""
        # TODO: Load from database or file
        raise NotImplementedError("Implement graph loading")

    def find_scenic_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        max_detour_factor: float = 1.5
    ) -> Route:
        """
        Find the most scenic route between two points.

        Args:
            start: Starting coordinates (lat, lon)
            end: Ending coordinates (lat, lon)
            scenic_weight: Balance between distance (0) and scenery (1)
            avoid_highways: If True, prefer smaller roads
            max_detour_factor: Maximum route length as multiple of direct distance

        Returns:
            Optimal Route object
        """
        if self.graph is None:
            raise RuntimeError("Road graph not loaded")

        # Find nearest nodes to start and end
        start_node = self.graph.find_nearest_node(*start)
        end_node = self.graph.find_nearest_node(*end)

        # Configure cost function
        self.cost_function.scenic_weight = scenic_weight
        self.cost_function.avoid_highways = avoid_highways

        # Run A* search
        path = self._astar(start_node, end_node, max_detour_factor)

        if path is None:
            raise ValueError("No route found between the given points")

        return self._path_to_route(path)

    def _astar(
        self,
        start: Node,
        goal: Node,
        max_detour_factor: float
    ) -> Optional[List[Edge]]:
        """
        A* pathfinding with scenic cost function.

        Returns:
            List of edges forming the optimal path, or None if no path found
        """
        # Calculate direct distance for max path length
        direct_dist = self._haversine(
            start.lat, start.lon, goal.lat, goal.lon
        )
        max_path_length = direct_dist * max_detour_factor

        # Priority queue: (f_score, node_id, path_edges, g_score)
        open_set = [(0, start.id, [], 0)]
        closed_set = set()
        g_scores = {start.id: 0}

        while open_set:
            f_score, current_id, path, g_score = heapq.heappop(open_set)

            if current_id == goal.id:
                return path

            if current_id in closed_set:
                continue

            closed_set.add(current_id)

            # Check path length constraint
            if g_score > max_path_length:
                continue

            current = self.graph.get_node(current_id)

            for edge in self.graph.get_edges(current_id):
                neighbor_id = edge.end_node_id

                if neighbor_id in closed_set:
                    continue

                # Calculate cost for this edge
                edge_cost = self.cost_function.calculate(edge)
                tentative_g = g_score + edge_cost

                if neighbor_id not in g_scores or tentative_g < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = tentative_g

                    # Heuristic: straight-line distance to goal
                    neighbor = self.graph.get_node(neighbor_id)
                    h_score = self._haversine(
                        neighbor.lat, neighbor.lon, goal.lat, goal.lon
                    )

                    f_score = tentative_g + h_score
                    heapq.heappush(
                        open_set,
                        (f_score, neighbor_id, path + [edge], tentative_g)
                    )

        return None  # No path found

    def _path_to_route(self, edges: List[Edge]) -> Route:
        """Convert a path of edges to a Route object."""
        segments = []
        total_distance = 0
        total_scenic = 0
        waypoints = []

        for edge in edges:
            start_node = self.graph.get_node(edge.start_node_id)
            end_node = self.graph.get_node(edge.end_node_id)

            segment = RouteSegment(
                start=(start_node.lat, start_node.lon),
                end=(end_node.lat, end_node.lon),
                distance_km=edge.distance_km,
                scenic_score=edge.scenic_score,
                road_name=edge.road_name,
                road_type=edge.road_type
            )
            segments.append(segment)

            total_distance += edge.distance_km
            total_scenic += edge.scenic_score * edge.distance_km

            if not waypoints:
                waypoints.append((start_node.lat, start_node.lon))
            waypoints.append((end_node.lat, end_node.lon))

        avg_scenic = total_scenic / total_distance if total_distance > 0 else 0

        # Estimate duration (assume 60 km/h average)
        duration = total_distance / 60 * 60  # minutes

        return Route(
            segments=segments,
            total_distance_km=total_distance,
            average_scenic_score=avg_scenic,
            estimated_duration_minutes=duration,
            waypoints=waypoints
        )

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points in km."""
        from math import radians, cos, sin, asin, sqrt

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        return 6371 * c  # Earth radius in km
