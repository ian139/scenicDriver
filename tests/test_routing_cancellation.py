from threading import Event

import pytest
import src.route_planner.planner as planner_module

from src.route_planner.cost import evaluate_path
from src.route_planner.graph import Edge, Node, RoadGraph
from src.route_planner.planner import ScenicRoutePlanner

from src.route_planner.cancellation import (
    CancelToken,
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)


class _CustomCancelToken:
    def __init__(self, value: bool = False) -> None:
        self._value = value

    def is_set(self) -> bool:
        return self._value

    def set(self) -> None:
        self._value = True


def test_deadline_rejects_already_expired_budget() -> None:
    deadline = RoutingDeadline.after(0.0, clock=lambda: 12.0)

    with pytest.raises(RoutingTimeout, match="deadline exceeded"):
        deadline.check()


def test_deadline_propagates_explicit_cancellation() -> None:
    cancelled = Event()
    deadline = RoutingDeadline(cancel_event=cancelled, clock=lambda: 1.0)
    cancelled.set()

    with pytest.raises(RoutingCancelled, match="was cancelled"):
        deadline.remaining_seconds()


def test_deadline_reports_one_shared_remaining_budget() -> None:
    ticks = iter((10.0, 11.5, 12.0))
    deadline = RoutingDeadline.after(3.0, clock=lambda: next(ticks))

    assert deadline.remaining_seconds() == pytest.approx(1.5)


def _line_graph() -> RoadGraph:
    graph = RoadGraph()
    graph.add_node(Node("a", 42.0, -72.0))
    graph.add_node(Node("b", 42.0, -71.99))
    graph.add_edge(
        Edge(
            "ab",
            "a",
            "b",
            distance_km=1.0,
            scenic_score=5.0,
            speed_limit_kmh=60,
        )
    )
    return graph


def test_planner_rejects_expired_deadline_before_mutating_graph() -> None:
    graph = _line_graph()
    planner = ScenicRoutePlanner(graph)

    with pytest.raises(RoutingTimeout, match="deadline exceeded"):
        planner.find_fastest_route(
            (42.0, -72.0),
            (42.0, -71.99),
            deadline=RoutingDeadline.after(0.0),
        )

    assert planner.graph is graph


def test_planner_cancellation_interrupts_endpoint_resolution_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _line_graph()
    planner = ScenicRoutePlanner(graph)
    cancelled = Event()
    deadline = RoutingDeadline(cancel_event=cancelled)
    original = graph.find_nearest_edge_positions_with_distance
    calls = 0

    def cancelling_projection(*args: object, **kwargs: object):
        nonlocal calls
        callback = kwargs.get("check_cancelled")
        assert callable(callback)
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            cancelled.set()
        return result

    monkeypatch.setattr(
        graph,
        "find_nearest_edge_positions_with_distance",
        cancelling_projection,
    )

    with pytest.raises(RoutingCancelled, match="was cancelled"):
        planner.find_fastest_route(
            (42.0, -72.0),
            (42.0, -71.99),
            deadline=deadline,
        )

    assert calls == 1
    assert planner.graph is graph


def test_native_search_boundary_propagates_cancellation_and_restores_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if planner_module._scipy_shortest_path is None:
        pytest.skip("SciPy routing backend is unavailable")
    graph = _line_graph()
    planner = ScenicRoutePlanner(graph)
    cancelled = Event()
    deadline = RoutingDeadline(cancel_event=cancelled)
    original = planner_module._scipy_shortest_path

    def cancelling_shortest_path(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(planner, "_EXACT_ORACLE_MAX_NODES", 0)
    monkeypatch.setattr(planner, "_EXACT_ORACLE_MAX_EDGES", 0)
    monkeypatch.setattr(
        planner_module,
        "_scipy_shortest_path",
        cancelling_shortest_path,
    )

    with pytest.raises(RoutingCancelled, match="was cancelled"):
        planner.find_fastest_route(
            (42.0, -72.0),
            (42.0, -71.99),
            deadline=deadline,
        )

    assert planner.graph is graph


def test_path_scoring_stops_consuming_edges_after_cancellation() -> None:
    cancelled = Event()
    deadline = RoutingDeadline(cancel_event=cancelled)
    edge = _line_graph().edges["ab"]
    consumed = 0

    def edges():
        nonlocal consumed
        for index in range(3_000):
            if index == 1_024:
                cancelled.set()
            consumed += 1
            yield edge

    with pytest.raises(RoutingCancelled, match="was cancelled"):
        evaluate_path(
            edges(),
            q=0.5,
            kappa=1.8,
            fastest_duration_minutes=1.0,
            check_cancelled=deadline.check,
        )

    assert consumed == 1_025
def test_deadline_accepts_cancel_token_protocol() -> None:
    token = _CustomCancelToken()
    deadline = RoutingDeadline.after(10.0, cancel_event=token)
    assert deadline.remaining_seconds() is not None
    token.set()
    with pytest.raises(RoutingCancelled, match="was cancelled"):
        deadline.check()


def test_cancel_token_protocol_is_runtime_checkable() -> None:
    assert isinstance(_CustomCancelToken(), CancelToken)
    assert not isinstance(object(), CancelToken)
