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

    Uses:
        cost = alpha * travel_time_minutes + beta * (10 - scenic_score) + road_adjustment
    where:
        alpha = 1 - scenic_weight
        beta = scenic_weight
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

        travel = max(edge.travel_time_minutes, 0.0) * self.weights.travel_time
        scenic_score = float(min(max(edge.scenic_score, 0.0), 10.0))
        scenic_penalty = (10.0 - scenic_score) * self.weights.scenic_reward
        road_adj = self._road_type_adjustment(edge.road_type)

        raw = alpha * travel + beta * scenic_penalty + road_adj
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
