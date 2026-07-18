from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time
from typing import Any

import pytest

from src.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)
from src.route_planner import supervisor as supervisor_module
from src.route_planner.supervisor import (
    PreloadedRouteSupervisor,
    SupervisorError,
    get_preload_marker,
)


# Module-level state is observed by the persistent worker child.  It is *not*
# shared with the test process after fork; it proves the worker process is reused.
_MODULE_COUNTER = 0

# Payload size deliberately well above typical PIPE_BUF (512 bytes on macOS).
_LARGE_PAYLOAD_SIZE = 2 * 1024 * 1024


def _job_success(deadline, token: str) -> dict:
    return {"token": token, "expires_at": deadline.expires_at}


def _job_deadline_identity_and_budget(deadline, budget: float) -> dict:
    helper_id = _inner_deadline_id(deadline)
    return {
        "id": id(deadline),
        "helper_id": helper_id,
        "expires_at": deadline.expires_at,
        "remaining": deadline.remaining_seconds(),
        "budget": budget,
    }


def _inner_deadline_id(deadline) -> int:
    return id(deadline)


def _job_return_expires_at(deadline) -> tuple[float | None, float]:
    """Return the deadline's absolute expiry and the child's current monotonic clock."""
    return deadline.expires_at, time.monotonic()


def _job_raise_value_error(deadline, message: str) -> None:
    raise ValueError(message)


def _job_uncooperative(deadline) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.05)


def _job_get_marker(deadline) -> Any:
    return get_preload_marker()


def _job_return_pid(deadline) -> int:
    return os.getpid()


def _job_increment_counter(deadline) -> tuple[int, int]:
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    return _MODULE_COUNTER, os.getpid()


def _job_read_counter(deadline) -> tuple[int, int]:
    return _MODULE_COUNTER, os.getpid()


def _job_large_response(deadline) -> bytes:
    return b"x" * _LARGE_PAYLOAD_SIZE


def _job_large_response_with_deadline(deadline, prefix: str) -> dict:
    # Return a large payload nested in a dict to exercise pickle framing.
    return {"prefix": prefix, "payload": b"y" * _LARGE_PAYLOAD_SIZE}


def _job_uncooperative_until_cancelled(deadline) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while deadline.cancel_event is None or not deadline.cancel_event.is_set():
        time.sleep(0.02)


def _job_late_success(deadline) -> str:
    # Ignore SIGTERM so the supervisor has to hard-stop or reject the late result.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    remaining = deadline.remaining_seconds()
    if remaining is not None:
        time.sleep(remaining + 0.1)
    return "late"


def _job_late_error(deadline) -> str:
    # Return a non-timeout error after expires_at; the supervisor must still
    # convert the late response to RoutingTimeout and dispose the worker.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    remaining = deadline.remaining_seconds()
    if remaining is not None:
        time.sleep(remaining + 0.1)
    raise ValueError("this should not win over the deadline")


def _wait_for_process_exit(pid: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise AssertionError(f"process {pid} is still alive")


def test_successful_response() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        result = sup.run_job(_job_success, "hello", deadline_seconds=5.0)

    assert result["token"] == "hello"
    assert result["expires_at"] is not None


def test_one_deadline_identity_and_absolute_budget() -> None:
    budget = 10.0
    deadline = RoutingDeadline.after(budget)
    expected_expires = deadline.expires_at
    with PreloadedRouteSupervisor.start() as sup:
        result = sup.run_job(
            _job_deadline_identity_and_budget,
            budget,
            deadline=deadline,
            grace_seconds=0.5,
        )

    # Exactly one RoutingDeadline object is created in the child and passed
    # through to helpers, using the caller's absolute expires_at.
    assert result["id"] == result["helper_id"]
    assert result["expires_at"] == expected_expires
    assert 0.0 < result["remaining"] <= budget


def test_deadline_object_exact_expiry_observed_by_worker() -> None:
    budget = 10.0
    deadline = RoutingDeadline.after(budget)
    with PreloadedRouteSupervisor.start() as sup:
        child_expires, child_now = sup.run_job(
            _job_return_expires_at,
            deadline=deadline,
            grace_seconds=0.5,
        )
    assert child_expires is not None
    assert child_expires == deadline.expires_at
    # Child received the same absolute deadline; its own clock is consistent.
    assert deadline.expires_at is not None
    assert child_now <= deadline.expires_at + budget

def test_queued_delay_does_not_extend_deadline() -> None:
    """A RoutingDeadline passed before queue/IPC keeps its original budget."""
    budget = 0.5
    deadline = RoutingDeadline.after(budget)
    expected_expires = deadline.expires_at
    time.sleep(0.15)  # simulate caller-side queuing/serialization delay
    sent_at = time.monotonic()

    with PreloadedRouteSupervisor.start() as sup:
        child_expires, _ = sup.run_job(
            _job_return_expires_at,
            deadline=deadline,
            grace_seconds=0.2,
        )
    # Expiry did not shift because of the sleep or IPC.
    assert child_expires is not None
    assert child_expires == expected_expires
    assert child_expires - sent_at < budget

def test_already_expired_deadline_fails_before_send() -> None:
    """Caller-side check rejects an expired deadline before any IPC."""
    deadline = RoutingDeadline(expires_at=time.monotonic() - 0.01)
    with PreloadedRouteSupervisor.start() as sup:
        with pytest.raises(RoutingTimeout, match="deadline exceeded"):
            sup.run_job(_job_success, "too-late", deadline=deadline)


def test_deadline_and_deadline_seconds_are_mutually_exclusive() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        with pytest.raises(ValueError, match="mutually exclusive"):
            sup.run_job(
                _job_success,
                "x",
                deadline=RoutingDeadline.after(5.0),
                deadline_seconds=5.0,
            )

def test_cancellation_set_after_dispatch_stops_work_and_replaces_worker() -> None:
    """A cancel_event set after dispatch is mirrored to the worker and supervisor."""
    cancel_event = threading.Event()
    deadline = RoutingDeadline.after(10.0, cancel_event=cancel_event)

    with PreloadedRouteSupervisor.start() as sup:
        pid1 = sup.run_job(_job_return_pid, deadline_seconds=5.0)

        def _trigger() -> None:
            time.sleep(0.1)
            cancel_event.set()

        trigger = threading.Thread(target=_trigger)
        trigger.start()
        try:
            with pytest.raises(RoutingCancelled, match="was cancelled") as exc_info:
                sup.run_job(
                    _job_uncooperative_until_cancelled,
                    deadline=deadline,
                    grace_seconds=0.5,
                )
        finally:
            trigger.join(timeout=2.0)

        # After cancellation the worker must be replaced; the next request still works.
        result = sup.run_job(_job_success, "after-cancel", deadline_seconds=5.0)
        pid2 = sup.run_job(_job_return_pid, deadline_seconds=5.0)

    assert result["token"] == "after-cancel"
    assert pid1 != pid2

    worker_pid = getattr(exc_info.value, "worker_pid", None)
    assert worker_pid is not None
    _wait_for_process_exit(worker_pid)


def test_late_response_at_expiry_rejected() -> None:
    """A success response received after expires_at must be rejected as a timeout."""
    with PreloadedRouteSupervisor.start() as sup:
        start = time.monotonic()
        with pytest.raises(RoutingTimeout, match="deadline exceeded") as exc_info:
            sup.run_job(
                _job_late_success,
                deadline_seconds=0.1,
                grace_seconds=0.3,
            )
        elapsed = time.monotonic() - start

    # Result arrives after expiry (0.1 + 0.1 sleep) but before hard kill (0.4).
    assert 0.15 <= elapsed <= 0.35, f"unexpected elapsed {elapsed}"
    worker_pid = getattr(exc_info.value, "worker_pid", None)
    assert worker_pid is not None
    _wait_for_process_exit(worker_pid)


def test_late_non_timeout_error_still_becomes_timeout() -> None:
    """A ValueError raised after expires_at must be overridden by RoutingTimeout."""
    with PreloadedRouteSupervisor.start() as sup:
        start = time.monotonic()
        with pytest.raises(RoutingTimeout, match="deadline exceeded") as exc_info:
            sup.run_job(
                _job_late_error,
                deadline_seconds=0.1,
                grace_seconds=0.3,
            )
        elapsed = time.monotonic() - start

    # The late error is discarded as a timeout; the worker is disposed.
    assert 0.15 <= elapsed <= 0.35, f"unexpected elapsed {elapsed}"
    worker_pid = getattr(exc_info.value, "worker_pid", None)
    assert worker_pid is not None
    _wait_for_process_exit(worker_pid)

def test_uncooperative_child_hard_killed_within_bounded_wall_time() -> None:
    deadline_seconds = 0.1
    grace_seconds = 0.2
    with PreloadedRouteSupervisor.start() as sup:
        start = time.monotonic()
        with pytest.raises(RoutingTimeout, match="deadline exceeded") as exc_info:
            sup.run_job(
                _job_uncooperative,
                deadline_seconds=deadline_seconds,
                grace_seconds=grace_seconds,
            )
        elapsed = time.monotonic() - start

    # The hard wall time is bounded by deadline + grace + small IPC overhead.
    assert elapsed <= deadline_seconds + grace_seconds + 0.3
    worker_pid = getattr(exc_info.value, "worker_pid", None)
    assert worker_pid is not None
    _wait_for_process_exit(worker_pid)


def test_persistent_worker_reused_across_successes_and_cache_persists() -> None:
    """Successful jobs must not fork a new child; mutated caches survive."""
    with PreloadedRouteSupervisor.start() as sup:
        count1, pid1 = sup.run_job(_job_increment_counter, deadline_seconds=5.0)
        count2, pid2 = sup.run_job(_job_increment_counter, deadline_seconds=5.0)
        count3, pid3 = sup.run_job(_job_read_counter, deadline_seconds=5.0)

    assert pid1 == pid2 == pid3, "worker should be reused across successful jobs"
    assert count1 == 1
    assert count2 == 2
    assert count3 == 2


def test_new_worker_forked_after_hard_kill() -> None:
    """A hard kill replaces the worker, resetting the worker-local cache."""
    with PreloadedRouteSupervisor.start() as sup:
        count1, pid1 = sup.run_job(_job_increment_counter, deadline_seconds=5.0)
        with pytest.raises(RoutingTimeout, match="deadline exceeded"):
            sup.run_job(
                _job_uncooperative,
                deadline_seconds=0.05,
                grace_seconds=0.1,
            )
        count2, pid2 = sup.run_job(_job_increment_counter, deadline_seconds=5.0)

    assert pid1 != pid2, "a fresh worker should be forked after a hard kill"
    assert count1 == 1
    # The fresh worker starts from the inherited module counter state (0).
    assert count2 == 1


def test_supervisor_persists_after_timeout_and_handles_subsequent_request() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        with pytest.raises(RoutingTimeout, match="deadline exceeded"):
            sup.run_job(
                _job_uncooperative,
                deadline_seconds=0.05,
                grace_seconds=0.1,
            )
        # Supervisor is still alive and the inherited state is still usable.
        assert sup.health()["alive"]
        result = sup.run_job(_job_success, "after-timeout", deadline_seconds=5.0)
    assert result["token"] == "after-timeout"


def test_inherited_preload_marker_visible_to_child() -> None:
    marker = {"loaded": True, "graph": "tiny"}
    with PreloadedRouteSupervisor.start(preload_marker=marker) as sup:
        result = sup.run_job(_job_get_marker, deadline_seconds=5.0)
    assert result == marker


def test_child_exception_propagation() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        with pytest.raises(ValueError, match="boom") as exc_info:
            sup.run_job(_job_raise_value_error, "boom", deadline_seconds=5.0)
    assert str(exc_info.value) == "boom"


def test_large_response_payload_bytes() -> None:
    """IPC must carry payloads well above PIPE_BUF."""
    with PreloadedRouteSupervisor.start() as sup:
        result = sup.run_job(
            _job_large_response,
            deadline_seconds=30.0,
        )
    assert isinstance(result, bytes)
    assert len(result) == _LARGE_PAYLOAD_SIZE


def test_large_response_payload_dict() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        result = sup.run_job(
            _job_large_response_with_deadline,
            "start-",
            deadline_seconds=30.0,
        )
    assert result["prefix"] == "start-"
    assert len(result["payload"]) == _LARGE_PAYLOAD_SIZE


def test_clean_shutdown_no_live_descendants() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        supervisor_pid = sup._process.pid

    assert not sup._process.is_alive()
    _wait_for_process_exit(worker_pid)

    # No direct children of the test process should remain.
    assert not multiprocessing.active_children()

    # No children of the supervisor should remain either.
    if supervisor_pid is not None and os.name == "posix":
        _assert_no_descendants(supervisor_pid)


def _assert_no_descendants(pid: int) -> None:
    """Best-effort descendant check using pgrep; recursive for macOS/Linux."""
    import subprocess

    seen = {pid}
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        try:
            output = subprocess.check_output(
                ["pgrep", "-P", str(parent)], text=True, stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in output.strip().splitlines():
            if not line:
                continue
            child_pid = int(line)
            if child_pid not in seen:
                seen.add(child_pid)
                frontier.append(child_pid)
    live = [p for p in seen if p != pid and _is_process_alive(p)]
    assert not live, f"live descendants of supervisor {pid}: {live}"


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_explicit_platform_boundary_on_non_posix(monkeypatch) -> None:
    monkeypatch.setattr(supervisor_module.os, "name", "nt")
    with pytest.raises(SupervisorError, match="POSIX"):
        PreloadedRouteSupervisor.start()


def test_health_check() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        health = sup.health()
        assert health["alive"] is True
        assert health["pid"] is not None
        assert health["closed"] is False

    health = sup.health()
    assert health["alive"] is False
    assert health["closed"] is True
