"""Request-scoped routing deadlines and cooperative cancellation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Event
import time
from typing import Callable


class RoutingCancelled(RuntimeError):
    """Raised when a routing request is explicitly cancelled."""


class RoutingTimeout(TimeoutError):
    """Raised when a routing request exhausts its monotonic deadline."""


@dataclass(frozen=True, slots=True)
class RoutingDeadline:
    """One absolute monotonic deadline shared by every routing stage."""

    expires_at: float | None = None
    cancel_event: Event | None = None
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def after(
        cls,
        seconds: float | None,
        *,
        cancel_event: Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> "RoutingDeadline":
        if seconds is None:
            return cls(cancel_event=cancel_event, clock=clock)
        value = float(seconds)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("routing deadline seconds must be finite and non-negative")
        return cls(expires_at=clock() + value, cancel_event=cancel_event, clock=clock)

    def remaining_seconds(self) -> float | None:
        now = self._checked_now()
        if self.expires_at is None:
            return None
        return self.expires_at - now

    def check(self) -> None:
        self._checked_now()

    def _checked_now(self) -> float:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RoutingCancelled("routing request was cancelled")
        now = self.clock()
        if self.expires_at is not None and now >= self.expires_at:
            raise RoutingTimeout("routing request deadline exceeded")
        return now
