"""Preloaded hard-stop supervisor for route planning jobs.

The supervisor is started after route caches are preloaded in the parent
process.  On POSIX it uses ``fork`` so the long-lived supervisor inherits the
read-only preload state.  The supervisor owns a single persistent disposable
planning child.  Successful jobs reuse that child so path/projection caches
mutated by planning survive across requests.  Each job is passed one absolute
``RoutingDeadline``; if the child ignores the deadline, it is SIGTERM'd at the
deadline and SIGKILL'd at ``deadline + grace``.  After a hard kill, a fresh
child is forked from the still-preloaded supervisor.  The supervisor remains
reusable.

Cancellation uses one lock-free ``SharedCancelFlag`` created by the parent in
``start`` and inherited by the supervisor and every worker.  The parent is its
only writer: it clears the flag after admitting a job and before sending it,
and sets it when the caller's ``cancel_event`` fires.  The supervisor and the
workers only read it, so killing either at any instruction can never wedge
the parent.

IPC uses length-prefixed pickle frames over non-blocking socket pairs.  Every
read and write is bounded by the job's absolute deadline and by a per-frame
transfer budget, and partial input is retained between short polls, so
deadline enforcement is never blocked behind a stalled or half-written frame;
a broken stream is classified as a worker failure and the worker is killed.

Process ownership survives abrupt loss of any ancestor.  The supervisor polls
its caller's PID (captured before ``fork``) both while idle and while a job
runs, and turns SIGTERM into a shutdown that kills and reaps the worker.  The
supervisor's direct child is a tiny guardian that forks the actual planning
worker, relays SIGTERM/SIGKILL requests to it, ``waitpid``\\ s it and reports
its real exit code.  The guardian holds a control socket whose only other
end is the supervisor's; when the supervisor dies the guardian SIGKILLs and
reaps the worker, independent of the worker's GIL or signal handling, then
exits (its own adoption by init is then unavoidable).  The supervisor joins
the guardian on every controlled stop, and SIGKILLs and joins an
unresponsive one.  Children close every inherited channel endpoint they do
not own.

Exceptions cross process boundaries as explicit envelopes carrying the exact
type plus its ``args`` and instance attributes, so structured errors such as
``RouteCoverageError`` keep their fields.  An envelope that cannot be rebuilt
becomes a job-scoped ``SupervisorError``; the dispatch loop keeps serving.

On non-POSIX systems the supervisor refuses to start, because neither
``fork`` nor true hard cancellation are available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import multiprocessing
from multiprocessing.context import ForkContext
from multiprocessing.process import BaseProcess
import os
import pickle
import select
import signal
import socket
import struct
import sys
import threading
import time
import traceback
from typing import Any, Callable

from .cancellation import (
    CancelToken,
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
    SharedCancelFlag,
)


_PRELOAD_MARKER: Any = None

_FRAME_HEADER = struct.Struct("!Q")
# Largest pickled message accepted in either direction.
_MAX_FRAME_BYTES = 512 * 1024 * 1024
# A frame, once started, must finish within this budget even without a deadline.
_FRAME_TRANSFER_BUDGET_SECONDS = 10.0
_RECV_CHUNK_BYTES = 1 << 20
# Granularity at which waits re-check deadlines and cancellation.
_POLL_SLICE_SECONDS = 0.05
# How long past ``deadline + grace`` the caller waits for the supervisor.
_SUPERVISOR_RESPONSE_MARGIN_SECONDS = 2.0
# How long a forced caller-side shutdown lets the supervisor reap its worker
# after SIGTERM before it is SIGKILL'd.
_SUPERVISOR_SHUTDOWN_GRACE_SECONDS = 2.0
# How often an idle worker guardian re-checks that its worker is still alive.
_GUARDIAN_POLL_SECONDS = 0.5
# How long the supervisor waits for a guardian to report a requested kill
# before it SIGKILLs the worker and the guardian itself.
_GUARDIAN_RESPONSE_SECONDS = 0.5
# How long a job without a deadline waits for the previous worker's guardian
# to finish reaping it before giving up.
_UNRESOLVED_REAP_GRACE_SECONDS = 1.0


def get_preload_marker() -> Any:
    """Return the marker set by the preloader in the supervisor process."""
    return _PRELOAD_MARKER


def _set_preload_marker(value: Any) -> None:
    global _PRELOAD_MARKER
    _PRELOAD_MARKER = value


@dataclass(frozen=True)
class _Ready:
    ok: bool = True


@dataclass(frozen=True)
class _Stop:
    pass


@dataclass(frozen=True)
class _DiscardWorker:
    pass


@dataclass(frozen=True)
class _Discarded:
    pass


@dataclass(frozen=True)
class _Job:
    """Job sent from the caller to the supervisor."""

    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    expires_at: float | None
    grace_seconds: float


@dataclass(frozen=True)
class _WorkerJob:
    """Job forwarded from the supervisor to the persistent worker."""

    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    expires_at: float | None


@dataclass(frozen=True)
class _SerializedError:
    """Transport envelope: exact exception type plus its pickled state.

    ``state`` is ``pickle((args, instance __dict__))`` or ``None`` when that
    state could not be pickled.  It is kept as bytes so a payload that cannot
    be unpickled on the receiving side only fails reconstruction of this one
    error, never the frame that carries it.
    """

    module_name: str
    qualname: str
    message: str
    state: bytes | None


@dataclass(frozen=True)
class _WorkerResult:
    value: Any = None
    error: _SerializedError | None = None
    worker_pid: int | None = None


@dataclass(frozen=True)
class _JobResult:
    value: Any = None
    error: _SerializedError | None = None
    worker_pid: int | None = None


class SupervisorError(RuntimeError):
    """The supervisor process could not start or communicate."""


class _SupervisorShutdown(BaseException):
    """The supervisor must stop now: SIGTERM, or its caller process is gone.

    A ``BaseException`` so the per-job error envelope never contains it.
    """


def _raise_shutdown(signum: int, frame: Any) -> None:
    """SIGTERM handler of the supervisor; one-shot so cleanup is not re-entered."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise _SupervisorShutdown("supervisor received SIGTERM")


def _check_owner(owner_pid: int | None) -> None:
    """Raise ``_SupervisorShutdown`` once the process that forked us is gone.

    ``owner_pid`` is captured before ``fork``, so an owner that dies before
    the child first runs is still detected after reparenting.
    """
    if owner_pid is not None and os.getppid() != owner_pid:
        raise _SupervisorShutdown(f"owner process {owner_pid} exited")


# Guardian -> supervisor status records: worker PID once forked, exit code once reaped.
_GUARDIAN_STATUS = struct.Struct("!cq")
_STATUS_WORKER_PID = b"P"
_STATUS_WORKER_EXIT = b"X"
# Supervisor -> guardian requests.
_CONTROL_TERMINATE = b"T"
_CONTROL_KILL = b"K"


def _ignore_signal(signum: int, frame: Any) -> None:
    """Guardian signal handler; the wakeup fd carries the signal number."""


def _guard_worker(control: socket.socket, worker_pid: int, wakeup_fd: int) -> None:
    """Relay control requests to the worker, reap it and report its exit code.

    A closed control socket means the supervisor is gone or done with this
    worker; SIGTERM to the guardian means the same.  Either SIGKILLs the
    worker.  Only this process ever waits for ``worker_pid``, so it is never
    reused while a request is relayed to it.
    """
    control.settimeout(1.0)
    try:
        control.sendall(_GUARDIAN_STATUS.pack(_STATUS_WORKER_PID, worker_pid))
    except OSError:
        pass
    while True:
        kill = False
        try:
            readable, _, _ = select.select(
                [control, wakeup_fd], [], [], _GUARDIAN_POLL_SECONDS
            )
        except (OSError, ValueError):
            readable, kill = [], True
        if wakeup_fd in readable:
            try:
                signals = os.read(wakeup_fd, 512)
            except BlockingIOError:
                signals = b""
            if int(signal.SIGTERM) in signals:
                kill = True
        if control in readable:
            try:
                requests: bytes | None = control.recv(512)
            except (BlockingIOError, TimeoutError):
                requests = None
            except OSError:
                requests = b""
            if requests is not None:
                if not requests or _CONTROL_KILL in requests:
                    kill = True
                elif _CONTROL_TERMINATE in requests:
                    # Resume a stopped worker first: its SIGTERM is only
                    # delivered once it runs again.  The guardian alone
                    # signals the worker, so a reaped PID is never targeted.
                    os.kill(worker_pid, signal.SIGCONT)
                    os.kill(worker_pid, signal.SIGTERM)
        if kill:
            os.kill(worker_pid, signal.SIGKILL)
        pid, status = os.waitpid(worker_pid, 0 if kill else os.WNOHANG)
        if pid:
            break
    try:
        control.sendall(
            _GUARDIAN_STATUS.pack(_STATUS_WORKER_EXIT, os.waitstatus_to_exitcode(status))
        )
    except OSError:
        pass


def _guardian_entry(
    worker_sock: socket.socket,
    worker_peer: socket.socket,
    control: socket.socket,
    control_peer: socket.socket,
    cancel_flag: SharedCancelFlag,
    ancestor_socks: tuple[socket.socket, ...],
) -> None:
    """Own one planning worker: fork it, relay control, reap it, report its exit.

    Closes every inherited endpoint it does not own (the supervisor's side of
    both channels and the supervisor's caller channel) before forking, so the
    worker inherits none of them and the supervisor's death is observable.
    """
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    worker_peer.close()
    control_peer.close()
    for ancestor in ancestor_socks:
        ancestor.close()
    worker_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    wakeup_read, wakeup_write = os.pipe()
    os.set_blocking(wakeup_read, False)
    os.set_blocking(wakeup_write, False)
    signal.signal(signal.SIGTERM, _ignore_signal)
    signal.signal(signal.SIGCHLD, _ignore_signal)
    signal.set_wakeup_fd(wakeup_write)
    worker_pid = os.fork()
    if worker_pid == 0:
        exitcode = 1
        try:
            signal.set_wakeup_fd(-1)
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            # The deadline SIGTERM is meant to terminate the job by default.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            if worker_sigint is not None:
                signal.signal(signal.SIGINT, worker_sigint)
            os.close(wakeup_read)
            os.close(wakeup_write)
            control.close()
            _persistent_worker_loop(worker_sock, cancel_flag)
            exitcode = 0
        except BaseException:
            traceback.print_exc()
        finally:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            os._exit(exitcode)
    worker_sock.close()
    try:
        _guard_worker(control, worker_pid, wakeup_read)
    finally:
        signal.set_wakeup_fd(-1)
        control.close()
        os.close(wakeup_read)
        os.close(wakeup_write)


class _GuardedWorker:
    """Process-like view of the planning worker owned by its guardian.

    ``pid`` and ``exitcode`` are the actual worker's.  The worker is only ever
    signalled by its guardian, which alone reaps it, so no request can reach
    a reused PID.  The supervisor signals only the guardian, its own unjoined
    child, and never kills it before the worker's reap has been reported.
    ``join`` reaps the guardian, so a controlled stop leaves no descendant.
    """

    def __init__(self, guardian: BaseProcess, control: socket.socket) -> None:
        control.setblocking(False)
        self._guardian = guardian
        self._control = control
        self._buffer = bytearray()
        self._detached = False
        self.pid: int | None = None
        self.exitcode: int | None = None

    def _detach(self) -> None:
        if not self._detached:
            self._detached = True
            self._control.close()

    def _read_status(self, timeout: float | None, *, need_pid: bool = False) -> None:
        """Consume guardian status until the worker exited or ``timeout`` passes."""
        stop_at = None if timeout is None else time.monotonic() + timeout
        while not (
            self._detached
            or self.exitcode is not None
            or (need_pid and self.pid is not None)
        ):
            wait = None if stop_at is None else max(0.0, stop_at - time.monotonic())
            try:
                readable, _, _ = select.select([self._control], [], [], wait)
            except (OSError, ValueError):
                self._detach()
                return
            if not readable:
                return
            try:
                data = self._control.recv(512)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if not data:
                self._detach()
                return
            self._buffer += data
            while len(self._buffer) >= _GUARDIAN_STATUS.size:
                tag, value = _GUARDIAN_STATUS.unpack_from(self._buffer)
                del self._buffer[: _GUARDIAN_STATUS.size]
                if tag == _STATUS_WORKER_PID:
                    self.pid = value
                elif tag == _STATUS_WORKER_EXIT:
                    self.exitcode = value

    def _send(self, request: bytes) -> None:
        if not self._detached:
            try:
                self._control.send(request)
            except OSError:
                pass

    def wait_started(self, timeout: float) -> None:
        self._read_status(timeout, need_pid=True)
        if self.pid is None:
            raise SupervisorError("worker guardian did not report a planning worker")

    def is_alive(self) -> bool:
        """Whether the worker may still run (its exit is unreported)."""
        self._read_status(0.0)
        return self.exitcode is None and not self._detached and self._guardian.is_alive()

    def _signal_guardian(self, signum: int) -> None:
        # Unjoined child: its PID cannot be reused while is_alive() is true.
        if self._guardian.is_alive():
            try:
                os.kill(self._guardian.pid, signum)  # type: ignore[arg-type]
            except ProcessLookupError:
                pass

    def terminate(self) -> None:
        """Ask the guardian to SIGTERM the worker at the job deadline.

        The guardian is resumed here and resumes the worker in turn before
        relaying the SIGTERM, so a stopped worker still receives it; a
        running one is unaffected.
        """
        if self.exitcode is None:
            self._signal_guardian(signal.SIGCONT)
            self._send(_CONTROL_TERMINATE)

    def kill(self) -> None:
        """Have the guardian SIGKILL and reap the worker, resuming it if stopped.

        Escalates from a kill request to SIGTERM on the guardian, which runs
        the same kill-and-reap path.  The guardian is never killed while it
        may be the worker's only reaper; one that has not reported yet stays
        unresolved for ``collected`` to finish later.
        """
        for escalation in (_CONTROL_KILL, signal.SIGTERM):
            if self.exitcode is not None or self._detached:
                return
            self._signal_guardian(signal.SIGCONT)
            if escalation is _CONTROL_KILL:
                self._send(_CONTROL_KILL)
            else:
                self._signal_guardian(signal.SIGTERM)
            self._read_status(_GUARDIAN_RESPONSE_SECONDS)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker's reported exit, then reap the guardian."""
        self._read_status(timeout)
        if self.exitcode is None and not self._detached:
            return
        if self._guardian.is_alive():
            self._signal_guardian(signal.SIGCONT)
            self._guardian.join(timeout=_GUARDIAN_RESPONSE_SECONDS)
        if self.exitcode is not None and self._guardian.is_alive():
            # The worker's reap is known, so the guardian owns nothing more.
            self._guardian.kill()
            self._guardian.join(timeout=2.0)
        if not self._guardian.is_alive():
            self._detach()

    def collected(self) -> bool:
        """Make non-blocking progress; true once the guardian has been reaped.

        A reaped guardian has itself reaped the worker, unless the guardian
        was killed from outside, which leaves the worker to init.
        """
        self.join(timeout=0)
        return not self._guardian.is_alive()


def _is_posix() -> bool:
    return os.name == "posix"


def _serialize_error(exc: BaseException) -> _SerializedError:
    cls = type(exc)
    try:
        state: bytes | None = pickle.dumps(
            (exc.args, dict(vars(exc))), protocol=pickle.HIGHEST_PROTOCOL
        )
    except Exception:
        state = None
    return _SerializedError(
        module_name=cls.__module__,
        qualname=cls.__qualname__,
        message=str(exc),
        state=state,
    )


def _rebuild_error(serialized: _SerializedError) -> BaseException:
    if serialized.state is None:
        raise ValueError("exception state could not be transferred")
    # Only classes already loaded here can be caught here; never import.
    target: Any = sys.modules[serialized.module_name]
    for part in serialized.qualname.split("."):
        target = getattr(target, part)
    if not (isinstance(target, type) and issubclass(target, BaseException)):
        raise TypeError(f"{serialized.qualname} is not an exception class")
    args, attributes = pickle.loads(serialized.state)
    try:
        error = target(*args)
    except Exception:
        # Constructors whose signature differs from ``args`` (for example
        # keyword-field errors) are rebuilt the way pickle restores state.
        error = target.__new__(target, *args)
    error.args = args
    error.__dict__.update(attributes)
    return error


def _reconstruct_error(serialized: _SerializedError) -> BaseException:
    """Rebuild an exception, containing even BaseException from pickle reducers.

    Only the supervisor's own shutdown request escapes the containment.
    """
    try:
        return _rebuild_error(serialized)
    except _SupervisorShutdown:
        raise
    except BaseException as exc:
        return SupervisorError(
            f"{serialized.module_name}.{serialized.qualname}: {serialized.message} "
            f"(could not be reconstructed: {exc!r})"
        )


class _ChannelBroken(Exception):
    """The stream is unusable: EOF, stalled transfer, or protocol violation."""


class _FrameTimeout(Exception):
    """No complete frame arrived in time; partial input is retained."""


class _FrameDecodeError(Exception):
    """A complete frame could not be unpickled; the stream is still in sync."""


class _FrameEncodeError(Exception):
    """A message could not be framed; nothing was written."""


class _FramedChannel:
    """Length-prefixed pickle frames over a non-blocking stream socket.

    ``recv`` never waits longer than its timeout, and a started frame must
    complete within ``_FRAME_TRANSFER_BUDGET_SECONDS``.  ``send`` gives up at
    the earlier of its deadline and that budget.  Partial input survives a
    ``recv`` timeout, so callers interleave deadline checks with transfers.
    """

    def __init__(self, sock: socket.socket) -> None:
        sock.setblocking(False)
        self._sock = sock
        self._header = bytearray(_FRAME_HEADER.size)
        self._header_received = 0
        self._payload: bytearray | None = None
        self._payload_received = 0
        self._frame_started_at: float | None = None

    def close(self) -> None:
        self._sock.close()

    def send(self, message: Any, *, deadline_at: float | None = None) -> None:
        try:
            payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise _FrameEncodeError(f"message could not be pickled: {exc!r}") from exc
        if len(payload) > _MAX_FRAME_BYTES:
            raise _FrameEncodeError(
                f"message of {len(payload)} bytes is too large "
                f"(limit {_MAX_FRAME_BYTES} bytes)"
            )
        stop_at = time.monotonic() + _FRAME_TRANSFER_BUDGET_SECONDS
        if deadline_at is not None:
            stop_at = min(stop_at, deadline_at)
        self._send_all(memoryview(_FRAME_HEADER.pack(len(payload))), stop_at)
        self._send_all(memoryview(payload), stop_at)

    def _send_all(self, view: memoryview, stop_at: float) -> None:
        while view:
            remaining = stop_at - time.monotonic()
            if remaining <= 0.0:
                raise _ChannelBroken("send exceeded its deadline")
            try:
                sent = self._sock.send(view)
            except BlockingIOError:
                sent = 0
            except OSError as exc:
                raise _ChannelBroken(f"send failed: {exc!r}") from exc
            remaining = stop_at - time.monotonic()
            if remaining <= 0.0:
                raise _ChannelBroken("send exceeded its deadline")
            view = view[sent:]
            if view and not sent:
                try:
                    select.select([], [self._sock], [], remaining)
                except (OSError, ValueError) as exc:
                    raise _ChannelBroken(f"send failed: {exc!r}") from exc

    def recv(self, *, timeout: float | None) -> Any:
        """Return the next message, waiting at most ``timeout`` seconds.

        ``timeout=None`` waits indefinitely for a frame to start; a started
        frame is still bounded by the transfer budget.
        """
        wait_until = None if timeout is None else time.monotonic() + timeout
        while True:
            # Progress does not extend either absolute bound.  Check before
            # accepting a completed frame as well as before reading more bytes.
            now = time.monotonic()
            limit = wait_until
            if self._frame_started_at is not None:
                budget_end = self._frame_started_at + _FRAME_TRANSFER_BUDGET_SECONDS
                if now >= budget_end:
                    raise _ChannelBroken("frame transfer exceeded its budget")
                limit = budget_end if limit is None else min(limit, budget_end)
            if wait_until is not None and now >= wait_until:
                raise _FrameTimeout()
            frame = self._take_frame()
            if frame is not None:
                try:
                    return pickle.loads(frame)
                except _SupervisorShutdown:
                    raise
                except BaseException as exc:
                    raise _FrameDecodeError(repr(exc)) from exc
            if self._receive_some():
                continue
            try:
                select.select([self._sock], [], [], None if limit is None else limit - now)
            except (OSError, ValueError) as exc:
                raise _ChannelBroken(f"receive failed: {exc!r}") from exc

    def _receive_some(self) -> bool:
        """Read whatever is available; return ``False`` if nothing was."""
        if self._payload is None:
            target = memoryview(self._header)[self._header_received :]
        else:
            target = memoryview(self._payload)[self._payload_received :]
        try:
            count = self._sock.recv_into(target, min(len(target), _RECV_CHUNK_BYTES))
        except BlockingIOError:
            return False
        except OSError as exc:
            raise _ChannelBroken(f"receive failed: {exc!r}") from exc
        if count == 0:
            raise _ChannelBroken("peer closed the channel")
        if self._frame_started_at is None:
            self._frame_started_at = time.monotonic()
        if self._payload is not None:
            self._payload_received += count
            return True
        self._header_received += count
        if self._header_received == _FRAME_HEADER.size:
            (length,) = _FRAME_HEADER.unpack(self._header)
            if length > _MAX_FRAME_BYTES:
                raise _ChannelBroken(
                    f"incoming frame of {length} bytes exceeds the "
                    f"{_MAX_FRAME_BYTES}-byte limit"
                )
            self._payload = bytearray(length)
            self._payload_received = 0
        return True

    def _take_frame(self) -> bytearray | None:
        payload = self._payload
        if payload is None or self._payload_received < len(payload):
            return None
        self._payload = None
        self._header_received = 0
        self._frame_started_at = None
        return payload


def _send_result(
    channel: _FramedChannel, result: _WorkerResult | _JobResult
) -> None:
    """Send ``result``; if it cannot be framed, send a job-scoped error instead."""
    try:
        channel.send(result)
    except _FrameEncodeError as exc:
        failure = SupervisorError(f"job result could not be transferred: {exc}")
        channel.send(replace(result, value=None, error=_serialize_error(failure)))


def _run_one_job(job: _WorkerJob, cancel_flag: SharedCancelFlag) -> _WorkerResult:
    """Execute a single job in the persistent worker."""
    try:
        deadline = RoutingDeadline(expires_at=job.expires_at, cancel_event=cancel_flag)
        value = job.func(deadline, *job.args, **job.kwargs)
        return _WorkerResult(value=value, worker_pid=os.getpid())
    except BaseException as exc:
        return _WorkerResult(error=_serialize_error(exc), worker_pid=os.getpid())


def _persistent_worker_loop(sock: socket.socket, cancel_flag: SharedCancelFlag) -> None:
    """Main loop of the persistent disposable planning worker."""
    channel = _FramedChannel(sock)
    worker_pid = os.getpid()
    try:
        while True:
            try:
                message = channel.recv(timeout=None)
            except _FrameDecodeError as exc:
                result = _WorkerResult(
                    error=_serialize_error(
                        SupervisorError(f"worker could not decode the job: {exc}")
                    ),
                    worker_pid=worker_pid,
                )
            else:
                if not isinstance(message, _WorkerJob):
                    return
                result = _run_one_job(message, cancel_flag)
            _send_result(channel, result)
    except _ChannelBroken:
        return
    finally:
        channel.close()


def _reap_process(process: BaseProcess | _GuardedWorker, timeout: float = 5.0) -> None:
    """Join a process; escalate to SIGTERM/SIGKILL if necessary."""
    if process.is_alive():
        process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)
    # Collect whatever already exited (for a guarded worker, its guardian).
    process.join(timeout=0)


def _force_stop_supervisor(process: BaseProcess) -> None:
    """SIGTERM so the supervisor kills and reaps its worker, then SIGKILL.

    If the supervisor cannot finish in time, its worker's guardian still
    SIGKILLs and reaps the worker once the supervisor's control socket closes.
    """
    if process.is_alive():
        process.terminate()
        process.join(timeout=_SUPERVISOR_SHUTDOWN_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)


class _WorkerHandle:
    """One persistent disposable child of the supervisor."""

    def __init__(
        self,
        ctx: ForkContext,
        cancel_flag: SharedCancelFlag,
        *,
        owner_pid: int | None = None,
        ancestor_socks: tuple[socket.socket, ...] = (),
    ) -> None:
        self._ctx = ctx
        self._cancel_flag = cancel_flag
        # PID of the supervisor's caller; a job is abandoned once it is gone.
        self._owner_pid = owner_pid
        # Endpoints inherited by every worker that only the supervisor owns.
        self._ancestor_socks = ancestor_socks
        self._proc: _GuardedWorker | None = None
        self._channel: _FramedChannel | None = None
        # A stopped worker whose guardian has not been reaped yet.  It stays
        # owned and is collected from the idle loop; no replacement is forked
        # until it is gone.
        self._unresolved: _GuardedWorker | None = None

    def _ensure_worker(
        self, reap_budget: float
    ) -> tuple[_GuardedWorker, _FramedChannel]:
        if self._proc is None or self._channel is None or not self._proc.is_alive():
            self._shutdown_worker()
            # A guardian that reaps its worker slowly is waited out within the
            # caller's own budget instead of failing the job outright; the
            # idle loop keeps collecting it either way.
            if not self.collect_until(reap_budget):
                raise SupervisorError(
                    "previous planning worker is still being reaped; retry later"
                )
            parent_sock, child_sock = socket.socketpair()
            control_sock, guardian_control = socket.socketpair()
            guardian = self._ctx.Process(
                target=_guardian_entry,
                args=(
                    child_sock,
                    parent_sock,
                    guardian_control,
                    control_sock,
                    self._cancel_flag,
                    self._ancestor_socks,
                ),
                daemon=False,
            )
            try:
                guardian.start()
            except BaseException:
                parent_sock.close()
                control_sock.close()
                raise
            finally:
                child_sock.close()
                guardian_control.close()
            self._proc = _GuardedWorker(guardian, control_sock)
            self._channel = _FramedChannel(parent_sock)
            self._proc.wait_started(_FRAME_TRANSFER_BUDGET_SECONDS)
        return self._proc, self._channel

    def _release_worker(self) -> None:
        """Drop the current worker; keep owning a guardian not reaped yet."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        proc, self._proc = self._proc, None
        if isinstance(proc, _GuardedWorker) and not proc.collected():
            self._unresolved = proc

    def _shutdown_worker(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        if self._proc is not None:
            _reap_process(self._proc, timeout=2.0)
        self._release_worker()

    def collect(self) -> bool:
        """Reap a guardian left unresolved by an earlier stop; true once none is left."""
        if self._unresolved is not None and self._unresolved.collected():
            self._unresolved = None
        return self._unresolved is None

    def collect_until(self, timeout: float) -> bool:
        stop_at = time.monotonic() + timeout
        while not self.collect():
            if time.monotonic() >= stop_at:
                return False
            time.sleep(_POLL_SLICE_SECONDS)
        return True

    def shutdown(self) -> None:
        """Politely stop the worker, then reap it."""
        if self._channel is not None and self._proc is not None and self._proc.is_alive():
            try:
                self._channel.send(_Stop(), deadline_at=time.monotonic() + 0.5)
            except _ChannelBroken:
                pass
        self._shutdown_worker()

    def kill(self) -> None:
        """Hard-stop the worker; the next job forks a fresh one."""
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(timeout=_GUARDIAN_RESPONSE_SECONDS)
        self._release_worker()

    def _dispose(self, error: BaseException, worker_pid: int | None) -> _JobResult:
        self.kill()
        return _JobResult(error=_serialize_error(error), worker_pid=worker_pid)

    def _failure_error(self, deadline_at: float | None, reason: str) -> BaseException:
        """Classify a worker that broke the protocol or died without a result."""
        if self._cancel_flag.is_set():
            return RoutingCancelled("routing request was cancelled")
        if deadline_at is not None and time.monotonic() >= deadline_at:
            return RoutingTimeout("routing request deadline exceeded")
        return SupervisorError(reason)

    def _dispose_failure(
        self, deadline_at: float | None, reason: str, worker_pid: int | None
    ) -> _JobResult:
        """Classify a worker failure at the moment it was seen, then dispose.

        A worker failure that is neither cancellation nor actual expiry is a
        ``SupervisorError`` carrying the worker's PID and exit code, which is
        read after the kill so a worker that already exited reports its own.
        """
        error = self._failure_error(deadline_at, reason)
        proc = self._proc
        self.kill()
        if isinstance(error, SupervisorError):
            exitcode = getattr(proc, "exitcode", None)
            error = SupervisorError(
                f"{reason} (worker pid {worker_pid}, exitcode {exitcode})"
            )
            error.worker_pid = worker_pid  # type: ignore[attr-defined]
            error.worker_exitcode = exitcode  # type: ignore[attr-defined]
        return _JobResult(error=_serialize_error(error), worker_pid=worker_pid)

    def run_job(self, job: _Job) -> _JobResult:
        """Forward a job to the persistent child and enforce its deadline."""
        deadline_at = job.expires_at
        # A slow reap of the previous worker is waited out within this job's
        # own budget (or a small bounded grace when it has no deadline).
        reap_budget = (
            _UNRESOLVED_REAP_GRACE_SECONDS
            if deadline_at is None
            else max(0.0, deadline_at - time.monotonic())
        )
        proc, channel = self._ensure_worker(reap_budget)
        worker_pid = proc.pid
        hard_at = None if deadline_at is None else deadline_at + job.grace_seconds

        worker_job = _WorkerJob(
            func=job.func,
            args=job.args,
            kwargs=job.kwargs,
            expires_at=deadline_at,
        )
        try:
            channel.send(worker_job, deadline_at=hard_at)
        except _FrameEncodeError as exc:
            return _JobResult(
                error=_serialize_error(
                    SupervisorError(f"job could not be sent to the worker: {exc}")
                ),
                worker_pid=worker_pid,
            )
        except _ChannelBroken as exc:
            return self._dispose_failure(
                deadline_at, f"worker channel broke: {exc}", worker_pid
            )

        terminated = False
        while True:
            _check_owner(self._owner_pid)
            now = time.monotonic()
            if hard_at is not None and now >= hard_at:
                return self._dispose(
                    RoutingTimeout("routing request deadline exceeded"), worker_pid
                )
            if deadline_at is not None and not terminated and now >= deadline_at:
                if proc.is_alive():
                    proc.terminate()
                terminated = True
            if self._cancel_flag.is_set():
                return self._dispose(
                    RoutingCancelled("routing request was cancelled"), worker_pid
                )
            wait = _POLL_SLICE_SECONDS
            if hard_at is not None:
                wait = min(wait, hard_at - now)
            try:
                worker_result = channel.recv(timeout=wait)
            except _FrameTimeout:
                continue
            except _FrameDecodeError as exc:
                return self._dispose_failure(
                    deadline_at, f"worker result could not be decoded: {exc}", worker_pid
                )
            except _ChannelBroken as exc:
                return self._dispose_failure(
                    deadline_at, f"worker exited without a result: {exc}", worker_pid
                )
            break

        if not isinstance(worker_result, _WorkerResult):
            return self._dispose(
                SupervisorError(f"unexpected message from worker: {worker_result!r}"),
                worker_pid,
            )
        # A late response that crossed expires_at or a cancellation is
        # rejected before any worker-produced error can win.
        if deadline_at is not None and time.monotonic() >= deadline_at:
            return self._dispose(
                RoutingTimeout("routing request deadline exceeded"), worker_pid
            )
        if self._cancel_flag.is_set():
            return self._dispose(
                RoutingCancelled("routing request was cancelled"), worker_pid
            )
        if worker_result.error is None:
            return _JobResult(value=worker_result.value, worker_pid=worker_result.worker_pid)

        # Cancellation and timeout from the worker dispose the child; other
        # exceptions keep it alive and travel on in their original envelope.
        error = _reconstruct_error(worker_result.error)
        if isinstance(error, (RoutingCancelled, RoutingTimeout)):
            return self._dispose(error, worker_result.worker_pid)
        return _JobResult(error=worker_result.error, worker_pid=worker_result.worker_pid)


def _supervisor_entry(
    sock: socket.socket,
    peer_sock: socket.socket,
    preload_marker: Any,
    cancel_flag: SharedCancelFlag,
    owner_pid: int,
) -> None:
    """Main loop of the long-lived supervisor process.

    ``owner_pid`` is the caller's PID captured before ``fork``.  Losing the
    caller, SIGTERM, or a broken caller channel hard-stops and reaps the
    worker; only an explicit ``_Stop`` asks the worker to exit politely.
    """
    signal.signal(signal.SIGTERM, _raise_shutdown)
    peer_sock.close()
    _set_preload_marker(preload_marker)
    channel = _FramedChannel(sock)
    worker = _WorkerHandle(
        multiprocessing.get_context("fork"),
        cancel_flag,
        owner_pid=owner_pid,
        ancestor_socks=(sock,),
    )
    polite = False
    try:
        _check_owner(owner_pid)
        channel.send(_Ready())
        while True:
            _check_owner(owner_pid)
            worker.collect()
            try:
                message = channel.recv(timeout=_POLL_SLICE_SECONDS)
            except _FrameTimeout:
                continue
            except _FrameDecodeError as exc:
                channel.send(
                    _JobResult(
                        error=_serialize_error(
                            SupervisorError(f"supervisor could not decode the job: {exc}")
                        )
                    )
                )
                continue
            if isinstance(message, _DiscardWorker):
                worker.kill()
                channel.send(_Discarded())
                continue
            if not isinstance(message, _Job):
                polite = isinstance(message, _Stop)
                return
            try:
                result = worker.run_job(message)
            except _SupervisorShutdown:
                raise
            except BaseException as exc:
                worker.kill()
                result = _JobResult(
                    error=_serialize_error(
                        SupervisorError(f"supervisor failed while running the job: {exc!r}")
                    )
                )
            _send_result(channel, result)
    except (_ChannelBroken, _SupervisorShutdown):
        return
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except _SupervisorShutdown:
            pass  # SIGTERM landed first; its handler already ignores repeats.
        if polite:
            worker.shutdown()
        else:
            worker.kill()
        # Bounded here; multiprocessing's exit handler then still joins a
        # guardian that is slower, unless the caller escalates to SIGKILL.
        worker.collect_until(_SUPERVISOR_SHUTDOWN_GRACE_SECONDS / 2)
        channel.close()


class PreloadedRouteSupervisor:
    """Persistent fork-based supervisor for routing jobs.

    The caller preloads route assets in the current process and then starts
    the supervisor with ``start()``.  Because the supervisor is forked from
    the caller, it inherits the preload caches and marker without a
    per-request load.

    Jobs run one at a time.  A queued caller waits for the job slot no longer
    than its own deadline and gives up on its own cancellation without
    touching the active job.  The supervisor maintains a single persistent
    disposable planning child.  Successful jobs reuse that child so mutated
    caches survive, while an unresponsive child is SIGTERM'd at the deadline
    and SIGKILL'd at ``deadline + grace``; a fresh child is then forked from
    the still-preloaded supervisor.  A kill-safe ``SharedCancelFlag``
    propagates ``RoutingDeadline.cancel_event`` across processes.

    Example::

        marker = preload_route_assets("graph.geojson")
        with PreloadedRouteSupervisor.start(preload_marker=marker) as sup:
            result = sup.run_job(plan_routes, request)

    On non-POSIX platforms ``start`` raises ``SupervisorError``.
    """

    def __init__(
        self,
        process: BaseProcess,
        channel: _FramedChannel,
        cancel_flag: SharedCancelFlag,
        default_deadline_seconds: float | None,
        default_grace_seconds: float,
    ) -> None:
        self._process = process
        self._channel = channel
        self._cancel_flag = cancel_flag
        self._default_deadline_seconds = default_deadline_seconds
        self._default_grace_seconds = default_grace_seconds
        self._slot = threading.Lock()
        self._closed = False

    @classmethod
    def start(
        cls,
        *,
        preload_marker: Any = None,
        default_deadline_seconds: float | None = 60.0,
        default_grace_seconds: float = 2.0,
        start_timeout: float = 30.0,
    ) -> "PreloadedRouteSupervisor":
        """Fork a long-lived supervisor process that inherits preload state."""
        if not _is_posix():
            raise SupervisorError(
                "PreloadedRouteSupervisor requires POSIX for fork-based "
                "hard-stop and inherited read-only caches"
            )

        ctx: ForkContext = multiprocessing.get_context("fork")
        cancel_flag = SharedCancelFlag()
        parent_sock, child_sock = socket.socketpair()
        # Captured before fork: a caller that dies before the supervisor first
        # runs must still be seen as lost after reparenting.
        owner_pid = os.getpid()
        proc = ctx.Process(
            target=_supervisor_entry,
            args=(child_sock, parent_sock, preload_marker, cancel_flag, owner_pid),
            daemon=False,
        )
        proc.start()
        child_sock.close()
        channel = _FramedChannel(parent_sock)

        try:
            try:
                ready = channel.recv(timeout=start_timeout)
            except _FrameTimeout as exc:
                raise SupervisorError("supervisor process did not become ready") from exc
            except (_ChannelBroken, _FrameDecodeError) as exc:
                raise SupervisorError("supervisor process failed during startup") from exc
            if not isinstance(ready, _Ready):
                raise SupervisorError(
                    f"supervisor sent unexpected ready message: {ready!r}"
                )
        except Exception:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5.0)
            channel.close()
            raise

        return cls(
            process=proc,
            channel=channel,
            cancel_flag=cancel_flag,
            default_deadline_seconds=default_deadline_seconds,
            default_grace_seconds=default_grace_seconds,
        )

    def _parent_deadline(
        self,
        deadline: RoutingDeadline | None,
        deadline_seconds: float | None,
    ) -> RoutingDeadline:
        """Build the caller-side RoutingDeadline used for checks and mirroring."""
        if deadline is not None:
            return deadline
        effective_seconds = (
            deadline_seconds
            if deadline_seconds is not None
            else self._default_deadline_seconds
        )
        if effective_seconds is not None:
            return RoutingDeadline.after(float(effective_seconds))
        return RoutingDeadline()

    def _acquire_slot(self, deadline: RoutingDeadline) -> None:
        """Wait for the job slot no longer than the caller's own deadline.

        A queued caller only consults its own deadline and cancel token, so
        giving up never touches the shared flag or channel of the active job.
        """
        while True:
            remaining = deadline.remaining_seconds()
            wait = _POLL_SLICE_SECONDS if remaining is None else min(_POLL_SLICE_SECONDS, remaining)
            if self._slot.acquire(timeout=wait):
                break
        try:
            deadline.check()
        except BaseException:
            self._slot.release()
            raise

    def _abandon(self) -> None:
        """Force-stop a supervisor whose channel can no longer be trusted."""
        self._closed = True
        self._channel.close()
        _force_stop_supervisor(self._process)

    def _recv_bounded(
        self, stop_at: float | None, caller_cancel: CancelToken | None = None
    ) -> Any:
        """Receive the supervisor's next message, abandoning it at ``stop_at``.

        While waiting, a set ``caller_cancel`` is mirrored into the shared flag.
        """
        while True:
            if caller_cancel is not None and caller_cancel.is_set():
                self._cancel_flag.set()
            if not self._process.is_alive():
                self._abandon()
                raise SupervisorError("supervisor process died")
            now = time.monotonic()
            if stop_at is not None and now >= stop_at:
                self._abandon()
                raise SupervisorError("supervisor did not respond before its deadline")
            wait = _POLL_SLICE_SECONDS if stop_at is None else min(_POLL_SLICE_SECONDS, stop_at - now)
            try:
                return self._channel.recv(timeout=wait)
            except _FrameTimeout:
                continue
            except _FrameDecodeError as exc:
                raise SupervisorError(f"supervisor reply could not be decoded: {exc}") from exc
            except _ChannelBroken as exc:
                self._abandon()
                raise SupervisorError("supervisor process closed unexpectedly") from exc

    def _discard_worker(self) -> None:
        """Dispose the idle planning child before another job can start."""
        try:
            self._channel.send(_DiscardWorker())
        except _ChannelBroken as exc:
            self._abandon()
            raise SupervisorError("supervisor process closed while discarding worker") from exc
        response = self._recv_bounded(time.monotonic() + _SUPERVISOR_RESPONSE_MARGIN_SECONDS)
        if not isinstance(response, _Discarded):
            raise SupervisorError(
                f"unexpected discard response from supervisor: {response!r}"
            )

    def run_job(
        self,
        func: Callable[..., Any],
        *args: Any,
        deadline: RoutingDeadline | None = None,
        deadline_seconds: float | None = None,
        grace_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send a picklable job to the supervisor and return its result.

        The job callable receives a single ``RoutingDeadline`` as its first
        positional argument, followed by ``*args`` and ``**kwargs``.  It may
        raise ``RoutingTimeout``/``RoutingCancelled`` or any other exception,
        which is rebuilt with its type, ``args`` and attributes where the
        exception class is loaded in this process, and otherwise surfaces as
        ``SupervisorError``.

        ``deadline`` and ``deadline_seconds`` are mutually exclusive.  If
        ``deadline`` is supplied, its absolute ``expires_at`` and any
        ``cancel_event`` are used; the local ``cancel_event`` is mirrored into
        the shared cancellation flag so the worker and supervisor observe it.
        If ``deadline_seconds`` is supplied, ``expires_at`` is computed from
        the current monotonic clock before queue/IPC so delays do not extend
        the budget.  The same deadline bounds the wait for the job slot.
        ``grace_seconds`` defaults to the value supplied to ``start``.
        """
        if self._closed:
            raise SupervisorError("supervisor is closed")
        if not self._process.is_alive():
            raise SupervisorError("supervisor process is not alive")

        if deadline is not None and deadline_seconds is not None:
            raise ValueError("deadline and deadline_seconds are mutually exclusive")

        parent_deadline = self._parent_deadline(deadline, deadline_seconds)
        grace = float(
            self._default_grace_seconds if grace_seconds is None else grace_seconds
        )
        if grace < 0.0 or not math.isfinite(grace):
            raise ValueError("grace_seconds must be non-negative and finite")

        job = _Job(
            func=func,
            args=args,
            kwargs=kwargs,
            expires_at=parent_deadline.expires_at,
            grace_seconds=grace,
        )

        parent_deadline.check()
        self._acquire_slot(parent_deadline)
        try:
            if self._closed:
                raise SupervisorError("supervisor is closed")
            return self._run_admitted(job, parent_deadline)
        finally:
            self._slot.release()

    def _run_admitted(self, job: _Job, parent_deadline: RoutingDeadline) -> Any:
        hard_at = None if job.expires_at is None else job.expires_at + job.grace_seconds
        # Owner ordering: clear the shared flag before the job is sent, so the
        # supervisor and worker only ever observe this job's cancellation.
        self._cancel_flag.clear()
        try:
            self._channel.send(job, deadline_at=hard_at)
        except _FrameEncodeError as exc:
            raise SupervisorError(f"routing job could not be sent: {exc}") from exc
        except _ChannelBroken as exc:
            self._abandon()
            raise SupervisorError("supervisor process closed unexpectedly") from exc

        response_at = (
            None if hard_at is None else hard_at + _SUPERVISOR_RESPONSE_MARGIN_SECONDS
        )
        result = self._recv_bounded(response_at, parent_deadline.cancel_event)

        if not isinstance(result, _JobResult):
            raise SupervisorError(f"unexpected result from supervisor: {result!r}")
        error = None if result.error is None else _reconstruct_error(result.error)
        if error is not None and result.worker_pid is not None:
            error.worker_pid = result.worker_pid  # type: ignore[attr-defined]
        if isinstance(error, (RoutingCancelled, RoutingTimeout)):
            raise error
        try:
            parent_deadline.check()
        except (RoutingCancelled, RoutingTimeout) as exc:
            if result.worker_pid is not None:
                exc.worker_pid = result.worker_pid  # type: ignore[attr-defined]
            self._discard_worker()
            raise
        if error is not None:
            raise error
        return result.value

    def health(self) -> dict[str, Any]:
        """Return a small health snapshot."""
        return {
            "alive": self._process.is_alive(),
            "pid": self._process.pid,
            "closed": self._closed,
        }

    def close(self, timeout: float = 10.0) -> None:
        """Stop the supervisor and reap the supervisor process.

        Should not be called concurrently with ``run_job``.
        """
        if self._closed:
            return
        self._closed = True

        try:
            self._channel.send(_Stop(), deadline_at=time.monotonic() + 1.0)
        except _ChannelBroken:
            pass
        self._channel.close()

        if self._process.is_alive():
            self._process.join(timeout=timeout)
        _force_stop_supervisor(self._process)

    def __enter__(self) -> "PreloadedRouteSupervisor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
