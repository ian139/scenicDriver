from threading import Event

import pytest

from src.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)


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
