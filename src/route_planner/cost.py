from __future__ import annotations

from dataclasses import dataclass

from .graph import Edge


@dataclass
class CostWeights:
    travel_time: float = 1.0
    scenic_reward: float = 1.0
    highway_penalty: float = 2.0
    scenic_byway_bonus: float = 1.0


class ScenicCostFunction:
    """
    Edge cost for scenic routing.

    All terms are additive minutes (or minute-scaled multipliers): travel time,
    travel-time-weighted scenic exposure, and road-type adjustments.  Scenic
    exposure is the time spent on an edge multiplied by its clamped scenic
    disutility, so subdividing a road geometry does not change its total cost.
    Costs are always clamped to ``>= 1e-6`` for shortest-path search.
    """

    def __init__(
        self,
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        weights: CostWeights | None = None,
    ) -> None:
        self.scenic_weight = float(min(max(scenic_weight, 0.0), 1.0))
        self.avoid_highways = bool(avoid_highways)
        self.weights = weights or CostWeights()

    def calculate(self, edge: Edge) -> float:
        alpha = 1.0 - self.scenic_weight
        beta = self.scenic_weight

        travel_time_minutes = max(float(edge.travel_time_minutes), 0.0)
        travel = travel_time_minutes * self.weights.travel_time
        scenic_score = float(min(max(edge.scenic_score, 0.0), 10.0))
        scenic_exposure = (
            travel_time_minutes
            * (10.0 - scenic_score)
            / 10.0
            * self.weights.scenic_reward
        )
        road_adjustment = (
            travel_time_minutes * self._road_type_adjustment(edge.road_type)
        )

        raw = alpha * travel + beta * scenic_exposure + road_adjustment
        # Keep non-negative costs for Dijkstra/A* style search.
        return max(raw, 1e-6)

    def _road_type_adjustment(self, road_type: str) -> float:
        rt = str(road_type).lower()
        adj = 0.0
        if self.avoid_highways and rt in {"highway", "motorway", "trunk"}:
            adj += self.weights.highway_penalty
        if rt in {"scenic_byway"}:
            adj -= self.weights.scenic_byway_bonus
        return adj
