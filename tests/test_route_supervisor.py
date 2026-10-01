from __future__ import annotations

import multiprocessing
import os
import pickle
import signal
import socket
import threading
import time
from typing import Any, Callable

import pytest

from src.route_planner.cancellation import (
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
    SharedCancelFlag,
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


def _job_die_inside_cancellation_checks(deadline) -> None:
    """SIGKILL this worker while other threads are inside cancellation checks."""

    def spin() -> None:
        while True:
            deadline.check()

    for _ in range(3):
        threading.Thread(target=spin, daemon=True).start()
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGKILL)


def _job_sleep(deadline, seconds: float) -> str:
    time.sleep(seconds)
    return "slow-done"


def _job_raise_route_coverage_error(deadline) -> None:
    from src.route_planner.service import RouteCoverageError

    raise RouteCoverageError("start", 1.234, 1.0)


def _job_raise_unicode_decode_error(deadline) -> None:
    b"ok\xff".decode("utf-8")


_ALLOW_UNCONSTRUCTIBLE = False


class _UnconstructibleError(Exception):
    """Only constructible in a process that opted in (the worker)."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "_UnconstructibleError":
        if not _ALLOW_UNCONSTRUCTIBLE:
            raise RuntimeError("cannot construct outside the worker")
        return super().__new__(cls, *args, **kwargs)


def _job_raise_unconstructible(deadline) -> None:
    global _ALLOW_UNCONSTRUCTIBLE
    _ALLOW_UNCONSTRUCTIBLE = True
    raise _UnconstructibleError("worker only")


def _job_raise_unpicklable_args(deadline) -> None:
    raise ValueError(lambda: None)


def _abort_error_restore(kind: str) -> None:
    if kind == "SystemExit":
        raise SystemExit("error state restore aborted")
    raise KeyboardInterrupt("error state restore interrupted")


class _AbortingErrorArgument:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def __reduce__(self):
        return _abort_error_restore, (self.kind,)


def _job_raise_aborting_state(deadline, kind: str) -> None:
    raise ValueError(_AbortingErrorArgument(kind))


def _run_bounded(func: Callable[[], Any], timeout: float) -> tuple[Any, BaseException | None]:
    """Run ``func`` on a daemon thread; fail if it does not finish in ``timeout``."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AssertionError(f"call did not finish within {timeout}s (wedged)")
    return outcome.get("value"), outcome.get("error")


def test_worker_killed_inside_cancellation_check_does_not_wedge_cancellation() -> None:
    """A worker SIGKILLed mid cancellation check must not wedge later cancels."""
    with PreloadedRouteSupervisor.start() as sup:
        for _ in range(5):
            _, error = _run_bounded(
                lambda: sup.run_job(
                    _job_die_inside_cancellation_checks, deadline_seconds=5.0
                ),
                timeout=5.0,
            )
            assert isinstance(error, SupervisorError)

            cancel_event = threading.Event()
            deadline = RoutingDeadline.after(10.0, cancel_event=cancel_event)
            threading.Timer(0.1, cancel_event.set).start()
            start = time.monotonic()
            _, error = _run_bounded(
                lambda: sup.run_job(
                    _job_uncooperative_until_cancelled,
                    deadline=deadline,
                    grace_seconds=0.5,
                ),
                timeout=5.0,
            )
            assert isinstance(error, RoutingCancelled)
            assert time.monotonic() - start < 2.0

        result = sup.run_job(_job_success, "after-kills", deadline_seconds=5.0)
    assert result["token"] == "after-kills"


def test_queued_caller_with_expiring_deadline_times_out_without_waiting() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        slow: dict[str, Any] = {}
        slow_thread = threading.Thread(
            target=lambda: slow.setdefault(
                "value", sup.run_job(_job_sleep, 1.0, deadline_seconds=5.0)
            )
        )
        slow_thread.start()
        time.sleep(0.2)  # the slow job now holds the only job slot

        start = time.monotonic()
        with pytest.raises(RoutingTimeout, match="deadline exceeded"):
            sup.run_job(_job_success, "queued", deadline_seconds=0.1)
        timed_out_after = time.monotonic() - start

        cancel_event = threading.Event()
        threading.Timer(0.1, cancel_event.set).start()
        start = time.monotonic()
        with pytest.raises(RoutingCancelled, match="was cancelled"):
            sup.run_job(
                _job_success,
                "queued-cancel",
                deadline=RoutingDeadline.after(5.0, cancel_event=cancel_event),
            )
        cancelled_after = time.monotonic() - start

        slow_thread.join(timeout=5.0)

    assert timed_out_after < 0.4
    assert cancelled_after < 0.4
    # The active job is unaffected by the queued callers giving up.
    assert slow["value"] == "slow-done"


def test_route_coverage_error_survives_transport_with_fields() -> None:
    from src.route_planner.service import RouteCoverageError

    with PreloadedRouteSupervisor.start() as sup:
        with pytest.raises(RouteCoverageError) as exc_info:
            sup.run_job(_job_raise_route_coverage_error, deadline_seconds=5.0)
        result = sup.run_job(_job_success, "after-coverage", deadline_seconds=5.0)

    error = exc_info.value
    assert error.endpoint == "start"
    assert error.snap_distance_km == pytest.approx(1.234)
    assert error.max_snap_distance_km == pytest.approx(1.0)
    assert str(error) == "The start point is too far from the supported road network."
    assert result["token"] == "after-coverage"


def test_multi_arg_builtin_error_round_trips_and_supervisor_survives() -> None:
    with PreloadedRouteSupervisor.start() as sup:
        with pytest.raises(UnicodeDecodeError) as exc_info:
            sup.run_job(_job_raise_unicode_decode_error, deadline_seconds=5.0)
        result = sup.run_job(_job_success, "after-unicode", deadline_seconds=5.0)

    error = exc_info.value
    assert error.encoding == "utf-8"
    assert error.object == b"ok\xff"
    assert (error.start, error.end) == (2, 3)
    assert result["token"] == "after-unicode"


@pytest.mark.parametrize(
    "job", [_job_raise_unconstructible, _job_raise_unpicklable_args]
)
def test_unreconstructible_error_is_job_scoped_supervisor_error(job) -> None:
    with PreloadedRouteSupervisor.start() as sup:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        with pytest.raises(SupervisorError):
            sup.run_job(job, deadline_seconds=5.0)
        assert sup.health()["alive"]
        result = sup.run_job(_job_success, "after-unknown", deadline_seconds=5.0)
        next_worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)

    assert result["token"] == "after-unknown"
    # A job-scoped error does not cost the warm worker.
    assert next_worker_pid == worker_pid


@pytest.mark.parametrize("kind", ["SystemExit", "KeyboardInterrupt"])
def test_error_state_base_exception_is_job_scoped(kind: str) -> None:
    with PreloadedRouteSupervisor.start() as sup:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        with pytest.raises(SupervisorError, match="could not be reconstructed"):
            sup.run_job(_job_raise_aborting_state, kind, deadline_seconds=5.0)
        assert sup.health()["alive"]
        result = sup.run_job(_job_success, "after-aborted-restore", deadline_seconds=5.0)
        assert sup.run_job(_job_return_pid, deadline_seconds=5.0) == worker_pid
    assert result["token"] == "after-aborted-restore"


def test_oversized_result_is_job_scoped_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Forked supervisor/worker inherit the reduced frame limit.
    monkeypatch.setattr(supervisor_module, "_MAX_FRAME_BYTES", 256 * 1024)
    with PreloadedRouteSupervisor.start() as sup:
        with pytest.raises(SupervisorError, match="too large"):
            sup.run_job(_job_large_response, deadline_seconds=5.0)
        result = sup.run_job(_job_success, "after-oversize", deadline_seconds=5.0)
    assert result["token"] == "after-oversize"



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


def test_parent_deadline_overrides_error_delayed_in_final_ipc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with PreloadedRouteSupervisor.start() as sup:
        original_worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        original_recv = supervisor_module._FramedChannel.recv

        def delayed_recv(channel: Any, *, timeout: float | None) -> Any:
            result = original_recv(channel, timeout=timeout)
            if isinstance(result, supervisor_module._JobResult):
                time.sleep(0.08)
            return result

        # Patched after start: only this (caller) process sees the delay.
        monkeypatch.setattr(supervisor_module._FramedChannel, "recv", delayed_recv)
        with pytest.raises(RoutingTimeout, match="deadline exceeded") as exc_info:
            sup.run_job(
                _job_raise_value_error,
                "worker error must not win",
                deadline_seconds=0.05,
            )
        replacement_worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)

    discarded_worker_pid = getattr(exc_info.value, "worker_pid", None)
    assert discarded_worker_pid == original_worker_pid
    assert replacement_worker_pid != original_worker_pid
    _wait_for_process_exit(original_worker_pid)


def test_cancellation_arriving_during_error_reconstruction_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_event = threading.Event()
    deadline = RoutingDeadline.after(5.0, cancel_event=cancel_event)
    with PreloadedRouteSupervisor.start() as sup:
        original_worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        original_reconstruct = supervisor_module._reconstruct_error

        def cancelling_reconstruct(serialized: Any) -> BaseException:
            error = original_reconstruct(serialized)
            cancel_event.set()
            return error

        monkeypatch.setattr(
            supervisor_module, "_reconstruct_error", cancelling_reconstruct
        )
        with pytest.raises(RoutingCancelled, match="was cancelled") as exc_info:
            sup.run_job(
                _job_raise_value_error,
                "worker error must not win",
                deadline=deadline,
            )
        replacement_worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)

    discarded_worker_pid = getattr(exc_info.value, "worker_pid", None)
    assert discarded_worker_pid == original_worker_pid
    assert replacement_worker_pid != original_worker_pid
    _wait_for_process_exit(original_worker_pid)


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


class _FakeProcess:
    """Peer process stand-in that ignores SIGTERM and records hard kills."""

    pid = 999_999

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return not self.killed

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def join(self, timeout: float | None = None) -> None:
        pass


def _stalled_peer(claimed_bytes: int, sent_bytes: int) -> tuple[socket.socket, socket.socket]:
    """Socket pair whose peer wrote a frame header and a partial body, then stalled."""
    ours, peer = socket.socketpair()
    peer.sendall(
        supervisor_module._FRAME_HEADER.pack(claimed_bytes) + b"x" * sent_bytes
    )
    return ours, peer


def _handle_reading_from(ours: socket.socket) -> tuple[Any, _FakeProcess]:
    process = _FakeProcess()
    handle = supervisor_module._WorkerHandle(
        multiprocessing.get_context("fork"), SharedCancelFlag()
    )
    handle._proc = process
    handle._channel = supervisor_module._FramedChannel(ours)
    return handle, process


def _worker_job(expires_at: float | None, grace_seconds: float) -> Any:
    return supervisor_module._Job(
        func=_job_success,
        args=("x",),
        kwargs={},
        expires_at=expires_at,
        grace_seconds=grace_seconds,
    )


def _job_error(result: Any) -> BaseException:
    assert result.error is not None
    return supervisor_module._reconstruct_error(result.error)


def test_partial_frame_then_stall_is_hard_stopped_at_deadline() -> None:
    ours, peer = _stalled_peer(claimed_bytes=1_000, sent_bytes=10)
    handle, process = _handle_reading_from(ours)
    start = time.monotonic()
    result = handle.run_job(_worker_job(start + 0.2, grace_seconds=0.1))
    elapsed = time.monotonic() - start
    peer.close()

    assert isinstance(_job_error(result), RoutingTimeout)
    assert process.terminated and process.killed
    assert 0.25 <= elapsed < 0.6, f"hard stop not enforced at deadline: {elapsed}"


def test_partial_frame_without_deadline_is_bounded_by_transfer_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "_FRAME_TRANSFER_BUDGET_SECONDS", 0.2)
    ours, peer = _stalled_peer(claimed_bytes=1_000, sent_bytes=10)
    handle, process = _handle_reading_from(ours)
    start = time.monotonic()
    result = handle.run_job(_worker_job(None, grace_seconds=0.1))
    elapsed = time.monotonic() - start
    peer.close()

    assert isinstance(_job_error(result), SupervisorError)
    assert process.killed
    assert elapsed < 0.6


def test_oversized_incoming_frame_kills_worker_immediately() -> None:
    ours, peer = _stalled_peer(
        claimed_bytes=supervisor_module._MAX_FRAME_BYTES + 1, sent_bytes=0
    )
    handle, process = _handle_reading_from(ours)
    start = time.monotonic()
    result = handle.run_job(_worker_job(start + 5.0, grace_seconds=0.1))
    elapsed = time.monotonic() - start
    peer.close()

    assert isinstance(_job_error(result), SupervisorError)
    assert process.killed
    assert elapsed < 0.5


def test_caller_abandons_supervisor_that_stalls_mid_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "_SUPERVISOR_RESPONSE_MARGIN_SECONDS", 0.2)
    ours, peer = _stalled_peer(claimed_bytes=1_000, sent_bytes=10)
    process = _FakeProcess()
    sup = PreloadedRouteSupervisor(
        process=process,  # type: ignore[arg-type]
        channel=supervisor_module._FramedChannel(ours),
        cancel_flag=SharedCancelFlag(),
        default_deadline_seconds=None,
        default_grace_seconds=0.1,
    )
    start = time.monotonic()
    with pytest.raises(SupervisorError):
        sup.run_job(_job_success, "x", deadline_seconds=0.1)
    elapsed = time.monotonic() - start
    peer.close()

    assert process.killed
    assert sup.health()["closed"] is True
    assert elapsed < 0.8


class _ProgressingSocket:
    """Real socket that advances a deterministic clock on every byte transferred."""

    def __init__(self, sock: socket.socket, *, writing: bool = False) -> None:
        self.sock = sock
        self.now = 0.0
        self.writing = writing

    def setblocking(self, value: bool) -> None:
        self.sock.setblocking(value)

    def close(self) -> None:
        self.sock.close()

    def send(self, data: memoryview) -> int:
        count = self.sock.send(data[:1] if self.writing else data)
        if self.writing:
            self.now += 0.001
        return count

    def recv_into(self, target: memoryview, size: int) -> int:
        count = self.sock.recv_into(target, min(size, 1))
        self.now += 0.001
        return count


@pytest.mark.parametrize("bound", ["deadline", "budget"])
def test_continuously_progressing_send_is_bounded(monkeypatch, bound: str) -> None:
    ours, peer = socket.socketpair()
    progressing = _ProgressingSocket(ours, writing=True)
    channel = supervisor_module._FramedChannel(progressing)
    monkeypatch.setattr(supervisor_module.time, "monotonic", lambda: progressing.now)
    monkeypatch.setattr(
        supervisor_module, "_FRAME_TRANSFER_BUDGET_SECONDS",
        0.02 if bound == "budget" else 10.0,
    )
    try:
        with pytest.raises(supervisor_module._ChannelBroken, match="deadline"):
            channel.send("x" * 100, deadline_at=0.02 if bound == "deadline" else None)
        assert progressing.now <= 0.021
    finally:
        channel.close()
        peer.close()


@pytest.mark.parametrize("bound", ["deadline", "budget"])
def test_continuously_progressing_receive_is_bounded(monkeypatch, bound: str) -> None:
    ours, peer = socket.socketpair()
    message = "x" * 100
    payload = pickle.dumps(message)
    peer.sendall(supervisor_module._FRAME_HEADER.pack(len(payload)) + payload)
    progressing = _ProgressingSocket(ours)
    channel = supervisor_module._FramedChannel(progressing)
    monkeypatch.setattr(supervisor_module.time, "monotonic", lambda: progressing.now)
    monkeypatch.setattr(
        supervisor_module, "_FRAME_TRANSFER_BUDGET_SECONDS",
        0.02 if bound == "budget" else 10.0,
    )
    expected_error = (
        supervisor_module._ChannelBroken if bound == "budget"
        else supervisor_module._FrameTimeout
    )
    try:
        with pytest.raises(expected_error):
            channel.recv(timeout=0.02 if bound == "deadline" else None)
        assert progressing.now <= 0.022
        if bound == "deadline":
            assert channel.recv(timeout=1.0) == message
    finally:
        channel.close()
        peer.close()


def test_continuously_progressing_worker_reply_is_hard_stopped(monkeypatch) -> None:
    ours, peer = socket.socketpair()
    payload = pickle.dumps(supervisor_module._WorkerResult(value="x" * 100))
    peer.sendall(supervisor_module._FRAME_HEADER.pack(len(payload)) + payload)
    progressing = _ProgressingSocket(ours)
    handle, process = _handle_reading_from(progressing)
    monkeypatch.setattr(supervisor_module.time, "monotonic", lambda: progressing.now)
    try:
        result = handle.run_job(_worker_job(0.05, grace_seconds=0.02))
        assert isinstance(_job_error(result), RoutingTimeout)
        assert process.terminated and process.killed
        assert progressing.now <= 0.072
    finally:
        handle.kill()
        peer.close()
