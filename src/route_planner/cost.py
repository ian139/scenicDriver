from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

from .graph import Edge

HIGHWAY_ROAD_TYPES = frozenset(
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
SCENIC_BYWAY_DISCOUNT_CAP = 0.5


def is_highway_road_type(road_type: object) -> bool:
    return str(road_type).lower() in HIGHWAY_ROAD_TYPES


MIN_EDGE_COST = 1e-6
SCENIC_NORMALIZATION_VERSION = "linear-v1"


def _required_finite(value: object, name: str, *, minimum: float | None = None) -> float:
    """Return a finite number, rejecting malformed optimization inputs."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return number

@dataclass(frozen=True)
class RoutingPolicy:
    """Resolved routing controls shared by every path evaluation and search."""

    scenic_weight: float
    kappa: float
    strict_highways: bool = False
    highway_preference: float = 0.0
    scenic_priority: bool = False



def resolve_routing_policy(
    *,
    scenic_weight: object,
    kappa: object,
    avoid_highways: object = False,
    highway_preference: object = 0.0,
    strict_highways: object | None = None,
    scenic_priority: object = False,
) -> RoutingPolicy:
    """Normalize request controls once, before search or cache lookup."""
    q_value = _required_finite(scenic_weight, "q")
    if not 0.0 <= q_value <= 1.0:
        raise ValueError("q must be finite and in [0, 1]")
    kappa_value = _required_finite(kappa, "kappa", minimum=1.0)
    preference_value = _required_finite(
        highway_preference, "highway preference", minimum=0.0
    )
    strict_value = (
        bool(avoid_highways)
        if strict_highways is None
        else bool(strict_highways)
    )
    return RoutingPolicy(
        scenic_weight=q_value,
        kappa=kappa_value,
        strict_highways=strict_value,
        highway_preference=preference_value,
        scenic_priority=bool(scenic_priority),
    )


def _checked_add(left: float, right: float, name: str) -> float:
    result = left + right
    if not math.isfinite(result):
        raise ValueError(f"{name} overflowed finite range")
    return result


def _checked_product(left: float, right: float, name: str) -> float:
    result = left * right
    if not math.isfinite(result):
        raise ValueError(f"{name} overflowed finite range")
    return result


def clamp_scenic_score(score: object) -> float:
    """Clamp an edge's scenic score to the immutable report range ``[0, 10]``."""
    return min(max(_required_finite(score, "scenic score"), 0.0), 10.0)


def normalize_scenic_score(score: object) -> float:
    """Apply the stable linear ``linear-v1`` normalization to a raw score."""
    return clamp_scenic_score(score) / 10.0


def _edge_value(edge: object, name: str) -> object:
    if isinstance(edge, dict):
        try:
            return edge[name]
        except KeyError as exc:
            raise ValueError(f"path edge is missing {name}") from exc
    try:
        return getattr(edge, name)
    except AttributeError as exc:
        raise ValueError(f"path edge is missing {name}") from exc
def _optional_edge_value(edge: object, name: str) -> object | None:
    if isinstance(edge, dict):
        return edge.get(name)
    return getattr(edge, name, None)




def _edge_duration(edge: object) -> float:
    return _required_finite(_edge_value(edge, "travel_time_minutes"), "edge duration", minimum=0.0)
def _edge_distance(edge: object) -> float:
    return _required_finite(_edge_value(edge, "distance_km"), "edge distance", minimum=0.0)


def _edge_id(edge: object, position: int) -> str:
    # Traversal identity distinguishes reverse/synthetic views even when their
    # canonical edge IDs overlap another edge's textual ``::rev`` suffix.
    value = _optional_edge_value(edge, "traversal_id")
    if value is None or str(value) == "":
        value = _edge_value(edge, "id")
    if value is None or str(value) == "":
        raise ValueError(f"path edge {position} is missing id")
    return str(value)



def distance_weighted_scenic_score(path: object) -> float:
    """Return ``sum(distance * clamp(score)) / sum(distance)`` for ``path``.

    Empty and zero-distance paths have score ``0``.  Malformed edge values are
    rejected rather than silently changing the optimization objective.
    """
    try:
        edges = tuple(path)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("path must be an iterable of edges") from exc
    total_distance = 0.0
    weighted_score = 0.0
    for edge in edges:
        distance = _edge_distance(edge)
        score = clamp_scenic_score(_edge_value(edge, "scenic_score"))
        weighted_edge_score = _checked_product(
            distance, score, "distance-weighted scenic score"
        )
        total_distance = _checked_add(total_distance, distance, "total distance")
        weighted_score = _checked_add(
            weighted_score, weighted_edge_score, "weighted scenic score"
        )
    if total_distance == 0.0:
        return 0.0
    return weighted_score / total_distance


def duration_component(
    duration_minutes: object,
    fastest_duration_minutes: object,
    kappa: object,
) -> float:
    """Return the normalized duration utility under ``T <= kappa*T_fast``."""
    duration = _required_finite(duration_minutes, "duration", minimum=0.0)
    fastest = _required_finite(
        fastest_duration_minutes, "fastest duration", minimum=0.0
    )
    detour = _required_finite(kappa, "kappa", minimum=1.0)
    if detour == 1.0:
        return 1.0 if duration == fastest else 0.0
    if fastest == 0.0:
        return 1.0 if duration == 0.0 else 0.0
    duration_cap = _checked_product(detour, fastest, "duration cap")
    denominator = _checked_product(
        detour - 1.0, fastest, "duration utility denominator"
    )
    numerator = duration_cap - duration
    if not math.isfinite(numerator):
        raise ValueError("duration utility numerator overflowed finite range")
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError("duration utility overflowed finite range")
    return min(1.0, max(0.0, value))


def combined_utility(
    q: object,
    kappa: object,
    duration_minutes: object,
    fastest_duration_minutes: object,
    normalized_scenic_score: object,
) -> float:
    """Compute ``(1-q)*duration_component + q*normalized_scenic_score``."""
    weight = _required_finite(q, "q")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("q must be in [0, 1]")
    scenic = _required_finite(
        normalized_scenic_score, "normalized scenic score"
    )
    if not 0.0 <= scenic <= 1.0:
        raise ValueError("normalized scenic score must be in [0, 1]")
    duration = duration_component(
        duration_minutes, fastest_duration_minutes, kappa
    )
    return (1.0 - weight) * duration + weight * scenic


@dataclass(frozen=True)
class PathEvaluation:
    """Independent, serializable diagnostics for one feasible candidate path."""

    edge_ids: tuple[str, ...]
    total_distance_km: float
    duration_minutes: float
    raw_scenic_score: float
    normalized_scenic_score: float
    duration_utility: float
    objective: float
    highway_count: int
    score_coverage: float
    score_run: tuple[tuple[str, float], ...]
    normalization_version: str = SCENIC_NORMALIZATION_VERSION
    highway_cost: float = 0.0
    policy: RoutingPolicy | None = None

    @property
    def canonical_edge_sequence(self) -> tuple[str, ...]:
        return self.edge_ids


def evaluate_path(
    path: object,
    *,
    q: object,
    kappa: object,
    fastest_duration_minutes: object,
    policy: RoutingPolicy | None = None,
    highway_preference: object = 0.0,
    scenic_priority: object = False,
    check_cancelled: Callable[[], None] | None = None,
) -> PathEvaluation:
    """Evaluate a path with the resolved policy objective."""
    resolved = policy or resolve_routing_policy(
        scenic_weight=q,
        kappa=kappa,
        highway_preference=highway_preference,
        scenic_priority=scenic_priority,
    )
    try:
        edges = iter(path)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("path must be an iterable of edges") from exc
    edge_ids: list[str] = []
    score_run: list[tuple[str, float]] = []
    total_distance = 0.0
    total_duration = 0.0
    weighted_score = 0.0
    scored_distance = 0.0
    highway_count = 0
    highway_duration = 0.0
    for position, edge in enumerate(edges):
        if check_cancelled is not None and position & 1023 == 0:
            check_cancelled()
        edge_id = _edge_id(edge, position)
        distance = _edge_distance(edge)
        duration = _edge_duration(edge)
        score = clamp_scenic_score(_edge_value(edge, "scenic_score"))
        weighted_edge_score = _checked_product(
            distance, score, "distance-weighted scenic score"
        )
        edge_ids.append(edge_id)
        score_run.append((edge_id, score))
        total_distance = _checked_add(total_distance, distance, "total distance")
        total_duration = _checked_add(total_duration, duration, "total duration")
        weighted_score = _checked_add(
            weighted_score, weighted_edge_score, "weighted scenic score"
        )
        scored_distance = _checked_add(
            scored_distance, distance, "scored distance"
        )
        if is_highway_road_type(_edge_value(edge, "road_type")):
            highway_count += 1
            highway_duration = _checked_add(
                highway_duration, duration, "highway duration"
            )
    if check_cancelled is not None:
        check_cancelled()
    raw_score = weighted_score / total_distance if total_distance else 0.0
    normalized_score = raw_score / 10.0
    coverage = scored_distance / total_distance if total_distance else 1.0
    duration_utility = duration_component(
        total_duration, fastest_duration_minutes, resolved.kappa
    )
    if resolved.scenic_priority:
        # Scenic score is the primary objective; duration is a hard
        # feasibility constraint and only breaks equal-score ties.
        base_objective = normalized_score
    else:
        base_objective = combined_utility(
            resolved.scenic_weight,
            resolved.kappa,
            total_duration,
            fastest_duration_minutes,
            normalized_score,
        )
    fastest = _required_finite(
        fastest_duration_minutes, "fastest duration", minimum=0.0
    )
    highway_cost = (
        resolved.highway_preference
        * highway_duration
        / max(fastest, MIN_EDGE_COST)
    )
    objective = base_objective - highway_cost
    return PathEvaluation(
        edge_ids=tuple(edge_ids),
        total_distance_km=total_distance,
        duration_minutes=total_duration,
        raw_scenic_score=raw_score,
        normalized_scenic_score=normalized_score,
        duration_utility=duration_utility,
        objective=objective,
        highway_count=highway_count,
        score_coverage=coverage,
        score_run=tuple(score_run),
        highway_cost=highway_cost,
        policy=resolved,
    )



def compare_path_evaluations(
    candidate: PathEvaluation, incumbent: PathEvaluation
) -> int:
    """Return ``1`` when candidate wins, ``-1`` when incumbent wins, else ``0``."""
    scenic_priority = bool(
        candidate.policy is not None and candidate.policy.scenic_priority
    )
    if scenic_priority:
        highway_preference = float(
            candidate.policy.highway_preference
            if candidate.policy is not None
            else 0.0
        )
        if highway_preference > 0.0:
            candidate_key = (
                candidate.objective,
                candidate.normalized_scenic_score,
                -candidate.duration_minutes,
                -candidate.total_distance_km,
            )
            incumbent_key = (
                incumbent.objective,
                incumbent.normalized_scenic_score,
                -incumbent.duration_minutes,
                -incumbent.total_distance_km,
            )
        else:
            candidate_key = (
                candidate.normalized_scenic_score,
                -candidate.duration_minutes,
                -candidate.total_distance_km,
            )
            incumbent_key = (
                incumbent.normalized_scenic_score,
                -incumbent.duration_minutes,
                -incumbent.total_distance_km,
            )
    else:
        candidate_key = (
            candidate.objective,
            candidate.raw_scenic_score,
            -candidate.duration_minutes,
        )
        incumbent_key = (
            incumbent.objective,
            incumbent.raw_scenic_score,
            -incumbent.duration_minutes,
        )
    if candidate_key > incumbent_key:
        return 1
    if candidate_key < incumbent_key:
        return -1
    if candidate.canonical_edge_sequence < incumbent.canonical_edge_sequence:
        return 1
    if candidate.canonical_edge_sequence > incumbent.canonical_edge_sequence:
        return -1
    return 0


def is_better_path(candidate: PathEvaluation, incumbent: PathEvaluation) -> bool:
    return compare_path_evaluations(candidate, incumbent) > 0


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
    shortest-path search.  At ``scenic_weight=1`` this remains a positive
    additive disutility objective, not a route-average scenic-score ratio.
    """

    def __init__(
        self,
        scenic_weight: float = 0.5,
        avoid_highways: bool = False,
        weights: CostWeights | None = None,
        *,
        highway_preference: float = 0.0,
        strict_highways: bool | None = None,
    ) -> None:
        try:
            scenic_weight_value = float(scenic_weight)
        except (TypeError, ValueError, OverflowError):
            scenic_weight_value = 0.5
        if not math.isfinite(scenic_weight_value):
            scenic_weight_value = 0.5
        self.scenic_weight = min(max(scenic_weight_value, 0.0), 1.0)
        self.strict_highways = (
            bool(avoid_highways)
            if strict_highways is None
            else bool(strict_highways)
        )
        self.highway_preference = _finite_nonnegative(highway_preference)
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
        if not is_highway_road_type(road_type):
            return 0.0
        if self.highway_preference > 0.0:
            return self.highway_preference
        if self.strict_highways:
            return _finite_nonnegative(self.weights.highway_penalty)
        return 0.0
