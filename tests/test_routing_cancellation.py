from threading import Event
from typing import Any

import pytest

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
