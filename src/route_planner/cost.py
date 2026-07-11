from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from .graph import Edge

SCENIC_BYWAY_DISCOUNT_CAP = 0.5
MIN_EDGE_COST = 1e-6


def _finite_nonnegative(value: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number) or number < 0.0:
        return default
    return number


@dataclass
class CostWeights:
    """Weights for the additive base cost and road-type adjustments.

    ``scenic_byway_bonus`` is a fractional discount of the nonnegative base
    travel/scenic cost.  It is clamped to
    ``[0, SCENIC_BYWAY_DISCOUNT_CAP]`` before use, so even very large bonuses
    leave at least half of an edge's base cost intact.
    """

    travel_time: float = 1.0
    scenic_reward: float = 1.0
    highway_penalty: float = 2.0
    scenic_byway_bonus: float = 1.0


class ScenicCostFunction:
    """
    Edge cost for scenic routing.

    Travel time and scenic exposure form a nonnegative base cost.  A scenic
    byway applies a bounded multiplicative discount to that base, while a
    highway avoidance penalty remains an additive, duration-scaled adjustment.
    Scenic exposure is the time spent on an edge multiplied by its clamped
    scenic disutility, so subdividing a road geometry does not change its
    total cost.  Costs are always finite and clamped to ``>= 1e-6`` for
    shortest-path search.
    """

    def __init__(
        self,
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        weights: CostWeights | None = None,
    ) -> None:
        try:
            scenic_weight_value = float(scenic_weight)
        except (TypeError, ValueError, OverflowError):
            scenic_weight_value = 0.5
        if not math.isfinite(scenic_weight_value):
            scenic_weight_value = 0.5
        self.scenic_weight = min(max(scenic_weight_value, 0.0), 1.0)
        self.avoid_highways = bool(avoid_highways)
        self.weights = weights or CostWeights()

    def calculate(self, edge: Edge) -> float:
        alpha = 1.0 - self.scenic_weight
        beta = self.scenic_weight

        travel_time_minutes = _finite_nonnegative(edge.travel_time_minutes)
        travel = travel_time_minutes * _finite_nonnegative(self.weights.travel_time)
        scenic_score = min(_finite_nonnegative(edge.scenic_score), 10.0)
        scenic_exposure = (
            travel_time_minutes
            * (10.0 - scenic_score)
            / 10.0
            * _finite_nonnegative(self.weights.scenic_reward)
        )
        weighted_base_cost = alpha * travel + beta * scenic_exposure
        # Keep even a perfectly scenic, fully weighted edge duration-sensitive.
        # This is a proportional floor rather than a per-edge constant, so
        # splitting an edge cannot manufacture extra floor cost.
        base_cost = max(
            weighted_base_cost,
            MIN_EDGE_COST * travel_time_minutes,
        )
        if str(edge.road_type).lower() == "scenic_byway":
            discount = min(
                _finite_nonnegative(self.weights.scenic_byway_bonus),
                SCENIC_BYWAY_DISCOUNT_CAP,
            )
            base_cost *= 1.0 - discount

        # Highway penalties intentionally remain duration-scaled and additive.
        road_adjustment = (
            travel_time_minutes * self._road_type_adjustment(edge.road_type)
        )
        raw = base_cost + road_adjustment
        if not math.isfinite(raw):
            return sys.float_info.max if raw > 0.0 else MIN_EDGE_COST
        return max(raw, MIN_EDGE_COST)

    def _road_type_adjustment(self, road_type: str) -> float:
        rt = str(road_type).lower()
        if self.avoid_highways and rt in {"highway", "motorway", "trunk"}:
            return _finite_nonnegative(self.weights.highway_penalty)
        return 0.0
