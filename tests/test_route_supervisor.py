from __future__ import annotations

import _thread
import ctypes
import multiprocessing
import os
import pickle
import select
import signal
import socket
import struct
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


_PID_RECORD = struct.Struct("!q")
_LOSS_EXIT_TIMEOUT_SECONDS = 5.0
# Larger than any pipe buffer, so a blocking write(2) of it never completes.
_NATIVE_BLOCK_BYTES = 4 * 1024 * 1024


def _report_pid(fd: int, pid: int) -> None:
    os.write(fd, _PID_RECORD.pack(pid))


def _read_pids(fd: int, count: int, timeout: float = 10.0) -> list[int]:
    """Read ``count`` PID records from a readiness pipe within ``timeout``."""
    stop_at = time.monotonic() + timeout
    data = b""
    while len(data) < count * _PID_RECORD.size:
        remaining = stop_at - time.monotonic()
        if remaining <= 0.0:
            raise AssertionError(f"readiness pipe produced {data!r} within {timeout}s")
        readable, _, _ = select.select([fd], [], [], remaining)
        if readable:
            chunk = os.read(fd, count * _PID_RECORD.size - len(data))
            if not chunk:
                raise AssertionError(f"readiness pipe closed after {data!r}")
            data += chunk
    return [
        _PID_RECORD.unpack_from(data, offset)[0]
        for offset in range(0, len(data), _PID_RECORD.size)
    ]


class _Readiness:
    """Pipes through which a blocking job proves it is running and where.

    ``pid`` carries PID records.  ``block`` is never drained: a native job
    writes ``_NATIVE_BLOCK_BYTES`` into it from one GIL-holding ``write(2)``
    call, so the first byte becoming readable proves that call has started,
    and it cannot return while this test holds the read end.
    """

    def __init__(self) -> None:
        self.pid_read, self.pid_write = os.pipe()
        self.block_read, self.block_write = os.pipe()

    def read(self, count: int, mode: str, timeout: float = 10.0) -> list[int]:
        pids = _read_pids(self.pid_read, count, timeout)
        if mode == "native":
            readable, _, _ = select.select([self.block_read], [], [], timeout)
            if not readable:
                raise AssertionError(f"native job did not block within {timeout}s")
        return pids

    def close(self) -> None:
        for fd in (self.pid_read, self.pid_write, self.block_read, self.block_write):
            os.close(fd)


def _kill_recorded(pids: list[int]) -> None:
    """Cleanup by recorded PID only; never by process group or pgrep."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _children(pid: int) -> list[int]:
    if os.path.isdir("/proc/self"):
        children = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as stat:
                    fields = stat.read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(fields[1]) == pid:
                children.append(int(entry))
        return children
    import subprocess

    try:
        output = subprocess.check_output(
            ["pgrep", "-P", str(pid)], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return []
    return [int(line) for line in output.split()]


def _snapshot_tree(*roots: int) -> list[int]:
    """Roots plus their descendants, captured while nothing is reparented yet.

    Taken at readiness and before any kill, so helpers the implementation forks
    (watchdogs, guardians) are tracked by PID once they are reparented.
    """
    seen = list(dict.fromkeys(roots))
    frontier = list(seen)
    while frontier:
        for child in _children(frontier.pop()):
            if child not in seen:
                seen.append(child)
                frontier.append(child)
    return seen


def _process_state(pid: int) -> str:
    """``gone`` once reaped, ``zombie`` if exited but unreaped, else ``running``."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "running"
    if os.path.isdir("/proc/self"):
        try:
            with open(f"/proc/{pid}/stat") as stat:
                state = stat.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            return "gone"
    else:
        import subprocess

        try:
            state = subprocess.check_output(
                ["ps", "-o", "stat=", "-p", str(pid)], text=True
            ).strip()
        except subprocess.CalledProcessError:
            return "gone"
    return "zombie" if state.startswith("Z") else "running"


def _wait_for_all_exit(
    pids: list[int], *, reaped: bool, timeout: float = _LOSS_EXIT_TIMEOUT_SECONDS
) -> None:
    """Wait until every PID has exited; with ``reaped`` it must also be reaped.

    ``reaped`` asserts a surviving owner waited for the process.  Without it,
    a zombie counts as exited: after its ancestors are lost, reaping belongs
    to whatever adopted it.
    """
    accepted = {"gone"} if reaped else {"gone", "zombie"}
    stop_at = time.monotonic() + timeout
    while True:
        states = {pid: _process_state(pid) for pid in pids}
        if all(state in accepted for state in states.values()):
            return
        if time.monotonic() >= stop_at:
            raise AssertionError(f"recorded processes not finished: {states}")
        time.sleep(0.02)


def _job_report_pid_and_block(deadline, pid_fd: int, block_fd: int, mode: str) -> None:
    """Ignore SIGTERM, announce this worker's PID, then never return.

    ``native`` blocks inside one ``write(2)`` called through ``PyDLL``, which
    keeps holding the GIL, so no Python thread in the worker can run.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _report_pid(pid_fd, os.getpid())
    if mode == "native":
        libc = ctypes.PyDLL(None)
        libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        buffer = ctypes.create_string_buffer(_NATIVE_BLOCK_BYTES)
        while True:
            libc.write(block_fd, buffer, _NATIVE_BLOCK_BYTES)
    while True:
        time.sleep(0.05)


def _job_exit_with_code(deadline, code: int) -> None:
    os._exit(code)


def _job_sigkill_self(deadline) -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def _sigterm_when_restored_in(target_pid: int) -> str:
    if os.getpid() == target_pid:
        os.kill(target_pid, signal.SIGTERM)
    return "restored"


class _SigtermOnRestore:
    """Unpickling this inside ``target_pid`` delivers SIGTERM to that process."""

    def __init__(self, target_pid: int) -> None:
        self.target_pid = target_pid

    def __reduce__(self):
        return _sigterm_when_restored_in, (self.target_pid,)


def _job_raise_sigterm_on_restore(deadline, target_pid: int) -> None:
    raise ValueError(_SigtermOnRestore(target_pid))


def _caller_entry(pid_fd: int, block_fd: int, phase: str, mode: str) -> None:
    """Intermediate caller: owns a supervisor, reports PIDs, then is SIGKILLed."""
    sup = PreloadedRouteSupervisor.start(default_deadline_seconds=None)
    _report_pid(pid_fd, sup._process.pid)
    if phase == "active":
        sup.run_job(_job_report_pid_and_block, pid_fd, block_fd, mode)
    else:
        _report_pid(pid_fd, sup.run_job(_job_return_pid))
    while True:
        time.sleep(1.0)


@pytest.mark.parametrize(
    ("phase", "mode"),
    [("active", "python"), ("active", "native"), ("idle", "python")],
)
def test_abrupt_caller_loss_leaves_no_supervisor_or_worker(phase: str, mode: str) -> None:
    ready = _Readiness()
    caller = multiprocessing.get_context("fork").Process(
        target=_caller_entry,
        args=(ready.pid_write, ready.block_write, phase, mode),
        daemon=False,
    )
    recorded: list[int] = []
    try:
        caller.start()
        supervisor_pid, worker_pid = ready.read(2, mode if phase == "active" else "python")
        recorded = _snapshot_tree(supervisor_pid, worker_pid)
        assert caller.pid not in recorded and supervisor_pid != worker_pid

        os.kill(caller.pid, signal.SIGKILL)
        caller.join(timeout=5.0)
        assert caller.exitcode == -signal.SIGKILL

        # The tree was reparented away from this process, so its exit is
        # observed by the PIDs recorded at readiness, not by the process tree.
        _wait_for_all_exit(recorded, reaped=False)
    finally:
        _kill_recorded(recorded)
        if caller.is_alive():
            caller.kill()
        caller.join(timeout=5.0)
        ready.close()


@pytest.mark.parametrize("mode", ["python", "native"])
def test_supervisor_killed_during_unbounded_job_leaves_no_worker(mode: str) -> None:
    ready = _Readiness()
    sup = PreloadedRouteSupervisor.start(default_deadline_seconds=None)
    supervisor_pid = sup._process.pid
    recorded = [supervisor_pid]
    outcome: dict[str, BaseException] = {}

    def call() -> None:
        try:
            sup.run_job(_job_report_pid_and_block, ready.pid_write, ready.block_write, mode)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            outcome["error"] = exc

    caller = threading.Thread(target=call, daemon=True)
    try:
        caller.start()
        (worker_pid,) = ready.read(1, mode)
        recorded = _snapshot_tree(supervisor_pid, worker_pid)

        os.kill(supervisor_pid, signal.SIGKILL)
        caller.join(timeout=5.0)
        assert not caller.is_alive(), "caller wedged after supervisor loss"
        assert isinstance(outcome.get("error"), SupervisorError)
        _wait_for_all_exit(recorded, reaped=False)
    finally:
        _kill_recorded(recorded)
        sup.close(timeout=1.0)
        ready.close()


def test_supervisor_killed_while_worker_idle_leaves_no_worker() -> None:
    sup = PreloadedRouteSupervisor.start(default_deadline_seconds=None)
    supervisor_pid = sup._process.pid
    recorded = [supervisor_pid]
    try:
        worker_pid = sup.run_job(_job_return_pid)
        recorded = _snapshot_tree(supervisor_pid, worker_pid)

        os.kill(supervisor_pid, signal.SIGKILL)
        _wait_for_all_exit(
            [pid for pid in recorded if pid != supervisor_pid], reaped=False
        )
        with pytest.raises(SupervisorError):
            sup.run_job(_job_return_pid)
    finally:
        _kill_recorded(recorded)
        sup.close(timeout=1.0)


@pytest.mark.parametrize("mode", ["python", "native"])
def test_forced_close_after_interrupted_job_reaps_supervisor_and_worker(mode: str) -> None:
    ready = _Readiness()
    sup = PreloadedRouteSupervisor.start(default_deadline_seconds=None)
    recorded = [sup._process.pid]
    worker_pids: list[int] = []

    def interrupt_when_worker_ready() -> None:
        try:
            worker_pids.extend(ready.read(1, mode))
        finally:
            # Always unblock the main thread, even if readiness never came.
            _thread.interrupt_main()

    interrupter = threading.Thread(target=interrupt_when_worker_ready, daemon=True)
    try:
        interrupter.start()
        with pytest.raises(KeyboardInterrupt):
            sup.run_job(_job_report_pid_and_block, ready.pid_write, ready.block_write, mode)
        interrupter.join(timeout=5.0)
        assert len(worker_pids) == 1, "worker never reported readiness"
        recorded = _snapshot_tree(recorded[0], worker_pids[0])

        # The supervisor is still busy with the abandoned job, so close() must
        # escalate; the escalation may not orphan the planning worker.
        sup.close(timeout=0.5)
        assert not sup._process.is_alive()
        _wait_for_all_exit(recorded, reaped=True)
    finally:
        _kill_recorded(recorded)
        sup.close(timeout=1.0)
        ready.close()


@pytest.mark.parametrize("boundary", ["job-decoding", "error-reconstruction"])
def test_sigterm_inside_supervisor_unpickling_stops_and_reaps(boundary: str) -> None:
    """SIGTERM landing inside a contained unpickle must still shut the supervisor down.

    The reducer signals only when restored in the supervisor process, either
    while it decodes the caller's job or while it rebuilds a worker error.
    """
    sup = PreloadedRouteSupervisor.start()
    supervisor_pid = sup._process.pid
    recorded = [supervisor_pid]
    try:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        recorded = _snapshot_tree(supervisor_pid, worker_pid)

        with pytest.raises(SupervisorError):
            if boundary == "job-decoding":
                sup.run_job(_job_success, _SigtermOnRestore(supervisor_pid), deadline_seconds=5.0)
            else:
                sup.run_job(_job_raise_sigterm_on_restore, supervisor_pid, deadline_seconds=5.0)

        # Not resumed with SIGTERM ignored: the supervisor is gone and it
        # killed and reaped everything it owned on the way out.
        sup._process.join(timeout=_LOSS_EXIT_TIMEOUT_SECONDS)
        assert sup._process.exitcode is not None, "supervisor kept serving after SIGTERM"
        _wait_for_all_exit(recorded, reaped=True)
    finally:
        _kill_recorded(recorded)
        sup.close(timeout=1.0)


_KILL_RECORD = struct.Struct("!qq")


class _GuardianFaults:
    """Fault injection inherited by every process the supervisor forks.

    Installed in this process before ``start``.  It logs every ``os.kill``
    made by the supervisor (the only child of this process), and can SIGSTOP
    a guardian around its single worker-exit status record: right after
    sending it, or instead of sending it once the worker is already reaped.
    ``exit-report-delayed`` instead holds the reaped worker's report for
    ``delay_seconds`` with SIGTERM ignored, then sends it and exits normally.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, stall: str, delay_seconds: float = 0.0
    ) -> None:
        self.kill_read, self.kill_write = os.pipe()
        self.fired_read, self.fired_write = os.pipe()
        os.set_blocking(self.kill_read, False)
        test_pid = os.getpid()
        real_kill = os.kill
        kill_write, fired_write = self.kill_write, self.fired_write

        def recording_kill(pid: int, sig: int) -> None:
            if os.getpid() != test_pid and os.getppid() == test_pid:
                os.write(kill_write, _KILL_RECORD.pack(pid, int(sig)))
            real_kill(pid, sig)

        monkeypatch.setattr(os, "kill", recording_kill)
        if stall not in ("after-exit-report", "exit-report-lost", "exit-report-delayed"):
            return
        real_sendall = socket.socket.sendall
        exit_tag = supervisor_module._STATUS_WORKER_EXIT
        record_size = supervisor_module._GUARDIAN_STATUS.size

        def stalling_sendall(sock: socket.socket, data: Any, *args: Any) -> None:
            if not (
                os.getpid() != test_pid
                and len(data) == record_size
                and bytes(data[:1]) == exit_tag
            ):
                return real_sendall(sock, data, *args)
            os.write(fired_write, _PID_RECORD.pack(os.getpid()))
            if stall == "exit-report-delayed":
                previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(delay_seconds)
                signal.signal(signal.SIGTERM, previous)
                return real_sendall(sock, data, *args)
            if stall == "after-exit-report":
                real_sendall(sock, data, *args)
            real_kill(os.getpid(), signal.SIGSTOP)
            return None

        monkeypatch.setattr(socket.socket, "sendall", stalling_sendall)

    def supervisor_kills(self) -> list[tuple[int, int]]:
        data = b""
        while True:
            try:
                chunk = os.read(self.kill_read, 65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            data += chunk
        return [
            _KILL_RECORD.unpack_from(data, offset)
            for offset in range(0, len(data), _KILL_RECORD.size)
        ]

    def close(self) -> None:
        for fd in (self.kill_read, self.kill_write, self.fired_read, self.fired_write):
            os.close(fd)


@pytest.mark.parametrize("stop", ["deadline-hard-kill", "close"])
@pytest.mark.parametrize("stall", ["stopped-before-kill", "after-exit-report", "exit-report-lost"])
def test_controlled_stop_reaps_stalled_guardian_and_worker(
    monkeypatch: pytest.MonkeyPatch, stall: str, stop: str
) -> None:
    """A stalled worker helper never costs reaping or a foreign-PID signal.

    The helper (guardian) is SIGSTOPped before the stop, right after it
    reports the worker's exit, or instead of reporting an exit it already
    reaped.  With the caller and supervisor alive, a controlled stop must
    leave every recorded PID reaped, and the supervisor must never signal
    the planning worker's PID itself: once reaped, that PID may be reused.
    """
    faults = _GuardianFaults(monkeypatch, stall)
    sup = PreloadedRouteSupervisor.start()
    supervisor_pid = sup._process.pid
    recorded = [supervisor_pid]
    try:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        recorded = _snapshot_tree(supervisor_pid, worker_pid)
        helpers = [pid for pid in recorded if pid not in (supervisor_pid, worker_pid)]
        assert helpers, "expected the worker's guardian among the recorded descendants"
        if stall == "stopped-before-kill":
            for pid in helpers:
                os.kill(pid, signal.SIGSTOP)

        if stop == "deadline-hard-kill":
            with pytest.raises(RoutingTimeout, match="deadline exceeded"):
                sup.run_job(_job_uncooperative, deadline_seconds=0.05, grace_seconds=0.1)
            _wait_for_all_exit(helpers + [worker_pid], reaped=True)
            assert sup.health()["alive"]
            replacement_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
            assert replacement_pid != worker_pid
            recorded += _snapshot_tree(supervisor_pid, replacement_pid)[1:]
        sup.close()

        assert sup._process.exitcode is not None
        _wait_for_all_exit(recorded, reaped=True)
        if stall != "stopped-before-kill":
            (stalled_pid,) = _read_pids(faults.fired_read, 1, timeout=1.0)
            assert stalled_pid in helpers, "fault injection did not stall the guardian"
        signalled = [target for target, _ in faults.supervisor_kills()]
        assert worker_pid not in signalled
    finally:
        _kill_recorded(recorded)
        sup.close(timeout=1.0)
        faults.close()


def test_guardian_reporting_after_cleanup_gave_up_is_reaped_while_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guardian that reports only after every cleanup wait is still reaped.

    The guardian reaps the killed worker, then holds its exit report longer
    than the supervisor's whole escalation (each guardian response wait plus
    the guardian join).  The supervisor is then left idle: no further job
    or process start may be needed for the late guardian to be reaped.
    """
    escalation = 3 * supervisor_module._GUARDIAN_RESPONSE_SECONDS + 2.0
    delay = escalation + 1.0
    faults = _GuardianFaults(monkeypatch, "exit-report-delayed", delay_seconds=delay)
    sup = PreloadedRouteSupervisor.start()
    supervisor_pid = sup._process.pid
    recorded = [supervisor_pid]
    try:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        recorded = _snapshot_tree(supervisor_pid, worker_pid)
        helpers = [pid for pid in recorded if pid not in (supervisor_pid, worker_pid)]
        assert helpers, "expected the worker's guardian among the recorded descendants"

        with pytest.raises(RoutingTimeout, match="deadline exceeded"):
            sup.run_job(_job_uncooperative, deadline_seconds=0.05, grace_seconds=0.1)
        (stalled_pid,) = _read_pids(faults.fired_read, 1, timeout=5.0)
        assert stalled_pid in helpers, "fault injection did not delay the guardian"

        # Idle from here on: the late guardian must be collected without help.
        _wait_for_all_exit(
            helpers + [worker_pid],
            reaped=True,
            timeout=delay + _LOSS_EXIT_TIMEOUT_SECONDS,
        )
        assert sup.health()["alive"]
        assert worker_pid not in [target for target, _ in faults.supervisor_kills()]
    finally:
        _kill_recorded(recorded)
        sup.close(timeout=1.0)
        faults.close()


@pytest.mark.parametrize("budget", ["deadline", "grace"])
def test_next_job_waits_out_a_late_guardian_reap(
    monkeypatch: pytest.MonkeyPatch, budget: str
) -> None:
    """A slow guardian reap delays the next job within budget instead of failing it.

    The guardian holds the killed worker's exit report longer than every
    guardian response wait.  A job with a deadline must wait that reap out
    and run on a fresh worker; one without a deadline gives up only after
    the bounded reap grace, never waiting indefinitely.
    """
    delay = 3.5
    faults = _GuardianFaults(monkeypatch, "exit-report-delayed", delay_seconds=delay)
    sup = PreloadedRouteSupervisor.start()
    supervisor_pid = sup._process.pid
    recorded = [supervisor_pid]
    try:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        recorded = _snapshot_tree(supervisor_pid, worker_pid)
        helpers = [pid for pid in recorded if pid not in (supervisor_pid, worker_pid)]
        assert helpers, "expected the worker's guardian among the recorded descendants"

        with pytest.raises(RoutingTimeout, match="deadline exceeded"):
            sup.run_job(_job_uncooperative, deadline_seconds=0.05, grace_seconds=0.1)
        (stalled_pid,) = _read_pids(faults.fired_read, 1, timeout=5.0)
        assert stalled_pid in helpers, "fault injection did not delay the guardian"

        if budget == "deadline":
            replacement_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
            assert replacement_pid != worker_pid
        else:
            start = time.monotonic()
            with pytest.raises(SupervisorError, match="still being reaped"):
                sup.run_job(_job_return_pid, deadline=RoutingDeadline())
            waited = time.monotonic() - start
            assert waited >= supervisor_module._UNRESOLVED_REAP_GRACE_SECONDS
            assert waited < delay
        assert sup.health()["alive"]
    finally:
        _kill_recorded(recorded)
        sup.close(timeout=1.0)
        faults.close()


@pytest.mark.parametrize(
    ("job", "args", "exitcode"),
    [(_job_exit_with_code, (7,), 7), (_job_sigkill_self, (), -signal.SIGKILL)],
)
def test_early_worker_exit_before_deadline_reports_pid_and_exitcode(
    job: Callable[..., Any], args: tuple[Any, ...], exitcode: int
) -> None:
    with PreloadedRouteSupervisor.start() as sup:
        worker_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)
        start = time.monotonic()
        with pytest.raises(SupervisorError) as exc_info:
            sup.run_job(job, *args, deadline_seconds=5.0)
        elapsed = time.monotonic() - start
        replacement_pid = sup.run_job(_job_return_pid, deadline_seconds=5.0)

    # Well before expiry, so this is a crash rather than a timeout.
    assert elapsed < 2.0
    error = exc_info.value
    assert getattr(error, "worker_pid", None) == worker_pid
    assert getattr(error, "worker_exitcode", None) == exitcode
    assert f"worker pid {worker_pid}, exitcode {exitcode}" in str(error)
    assert replacement_pid != worker_pid
    _wait_for_process_exit(worker_pid)


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
