"""Request-scoped routing deadlines and cooperative cancellation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import mmap
import os
import time
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class CancelToken(Protocol):
    """Minimal protocol for anything that can report cancellation."""

    def is_set(self) -> bool:
        ...


class RoutingCancelled(RuntimeError):
    """Raised when a routing request is explicitly cancelled."""


class RoutingTimeout(TimeoutError):
    """Raised when a routing request exhausts its monotonic deadline."""


class SharedCancelFlag:
    """Kill-safe cancellation byte shared with forked children.

    The flag is one byte of anonymous ``MAP_SHARED`` memory created before
    ``fork``.  No lock guards it: single-byte loads and stores are atomic, so a
    reader that is SIGKILLed at any instruction cannot leave the flag in a
    state that blocks anyone else.

    Owner ordering: the creating process is the only writer.  It calls
    ``clear`` before handing a job to a child and ``set`` to cancel that job;
    forked children only call ``is_set``.  ``set``/``clear`` from any other
    process raise ``RuntimeError``.
    """

    __slots__ = ("_memory", "_owner_pid")

    def __init__(self) -> None:
        self._memory = mmap.mmap(-1, 1)
        self._owner_pid = os.getpid()

    def is_set(self) -> bool:
        return self._memory[0] != 0

    def set(self) -> None:
        self._write(1)

    def clear(self) -> None:
        self._write(0)

    def _write(self, value: int) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("only the creating process may change a SharedCancelFlag")
        self._memory[0] = value


@dataclass(frozen=True, slots=True)
class RoutingDeadline:
    """One absolute monotonic deadline shared by every routing stage."""

    expires_at: float | None = None
    cancel_event: CancelToken | None = None
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def after(
        cls,
        seconds: float | None,
        *,
        cancel_event: CancelToken | None = None,
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
