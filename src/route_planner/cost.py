"""
Scenic Cost Function
Owner: progno-backend agent

Custom cost function for A* that balances distance with scenic value.
"""

from dataclasses import dataclass
from .graph import Edge


@dataclass
class CostWeights:
    """Configurable weights for cost calculation."""
    distance: float = 1.0
    scenic: float = 1.0
    highway_penalty: float = 0.5
    scenic_byway_bonus: float = 0.3


class ScenicCostFunction:
    """
    Calculate edge costs for scenic route optimization.

    Cost formula:
        cost = distance_weight * distance
             + scenic_weight * (10 - scenic_score)
             + road_type_adjustment

    Lower scenic scores result in higher cost, making scenic routes preferred.
    """

    # Road type speed estimates (km/h)
    ROAD_SPEEDS = {
        "highway": 100,
        "primary": 80,
        "secondary": 60,
        "residential": 40,
        "scenic_byway": 50
    }

    def __init__(
        self,
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        weights: CostWeights = None
    ):
        """
        Initialize cost function.

        Args:
            scenic_weight: Balance between distance (0) and scenery (1)
            avoid_highways: Add penalty for highway segments
            weights: Custom weight configuration
        """
        self.scenic_weight = scenic_weight
        self.avoid_highways = avoid_highways
        self.weights = weights or CostWeights()

    def calculate(self, edge: Edge) -> float:
        """
        Calculate the cost to traverse an edge.

        Args:
            edge: Road segment to evaluate

        Returns:
            Cost value (lower = better)
        """
        # Base distance cost
        distance_cost = edge.distance_km * self.weights.distance

        # Scenic cost (invert so high scenic = low cost)
        scenic_cost = (10 - edge.scenic_score) * self.weights.scenic

        # Road type adjustments
        road_adjustment = self._road_type_cost(edge.road_type)

        # Weighted combination
        # scenic_weight of 0 = pure distance optimization
        # scenic_weight of 1 = pure scenic optimization
        cost = (
            (1 - self.scenic_weight) * distance_cost +
            self.scenic_weight * scenic_cost +
            road_adjustment
        )

        return cost

    def _road_type_cost(self, road_type: str) -> float:
        """Calculate cost adjustment based on road type."""
        adjustment = 0.0

        if road_type == "highway" and self.avoid_highways:
            adjustment += self.weights.highway_penalty

        if road_type == "scenic_byway":
            adjustment -= self.weights.scenic_byway_bonus

        return adjustment

    def estimate_travel_time(self, edge: Edge) -> float:
        """
        Estimate travel time for an edge in minutes.

        Uses road type to estimate speed.
        """
        speed = self.ROAD_SPEEDS.get(edge.road_type, 50)
        if edge.speed_limit_kmh:
            speed = edge.speed_limit_kmh

        return (edge.distance_km / speed) * 60


class MultiObjectiveCost:
    """
    Multi-objective cost function for Pareto-optimal routes.

    Returns multiple cost dimensions:
    - Distance/time
    - Scenic value
    - Road quality

    Useful for presenting multiple route options to users.
    """

    def calculate(self, edge: Edge) -> dict:
        """Calculate multi-dimensional cost."""
        return {
            "distance_km": edge.distance_km,
            "time_minutes": (edge.distance_km / 60) * 60,
            "scenic_value": edge.scenic_score,
            "road_quality": self._road_quality_score(edge.road_type)
        }

    def _road_quality_score(self, road_type: str) -> float:
        """Score road quality/comfort."""
        scores = {
            "highway": 0.9,
            "primary": 0.8,
            "secondary": 0.6,
            "residential": 0.5,
            "scenic_byway": 0.7
        }
        return scores.get(road_type, 0.5)
