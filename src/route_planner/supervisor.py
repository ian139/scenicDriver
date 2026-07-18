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

The caller computes a single absolute ``expires_at`` value before queue/IPC and
passes it to ``run_job`` as either a ``RoutingDeadline`` or a
``deadline_seconds`` convenience value.  The supervisor and worker use that exact
``expires_at`` so queued or IPC delays cannot extend the request budget.

IPC uses ``Connection.send_bytes`` / ``recv_bytes`` with explicit ``pickle``
framing so payloads are not limited to ``PIPE_BUF``.  The implementation
retries on ``InterruptedError`` and never relies on a single ``os.write``.

On non-POSIX systems the supervisor refuses to start, because neither
``fork`` nor true hard cancellation are available.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import math
import multiprocessing
from multiprocessing.context import ForkContext
from multiprocessing.process import BaseProcess
import os
import pickle
import sys
import time
import threading
import traceback
from typing import Any, Callable

from .cancellation import RoutingCancelled, RoutingDeadline, RoutingTimeout


_PRELOAD_MARKER: Any = None


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
    type_name: str
    module_name: str | None
    message: str
    traceback_text: str


@dataclass(frozen=True)
class _WorkerResult:
    value: Any = None
    error: _SerializedError | None = None
    worker_pid: int | None = None


@dataclass(frozen=True)
class _JobResult:
    value: Any = None
    error: BaseException | None = None
    worker_pid: int | None = None


class SupervisorError(RuntimeError):
    """The supervisor process could not start or communicate."""


def _is_posix() -> bool:
    return os.name == "posix"


def _serialize_error(exc: BaseException) -> _SerializedError:
    return _SerializedError(
        type_name=type(exc).__name__,
        module_name=getattr(type(exc), "__module__", None),
        message=str(exc),
        traceback_text=traceback.format_exc(),
    )


def _deserialize_error(serialized: _SerializedError) -> BaseException:
    """Reconstruct an exception where the class is available locally."""
    # Exact routing cancellation exceptions first.
    if serialized.module_name == "src.route_planner.cancellation":
        if serialized.type_name == "RoutingTimeout":
            return RoutingTimeout(serialized.message)
        if serialized.type_name == "RoutingCancelled":
            return RoutingCancelled(serialized.message)

    # Built-in exceptions.
    builtin = getattr(builtins, serialized.type_name, None)
    if isinstance(builtin, type) and issubclass(builtin, BaseException):
        return builtin(serialized.message)

    # Try to locate the class dynamically.
    module = sys.modules.get(serialized.module_name or "")
    if module is not None:
        cls = getattr(module, serialized.type_name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            try:
                return cls(serialized.message)
            except Exception:
                pass

    return SupervisorError(f"{serialized.type_name}: {serialized.message}")


def _send_object(conn, obj: Any) -> None:
    """Send a pickled object over a Connection, retrying on EINTR.

    ``Connection.send_bytes`` handles the write-all loop internally; we add
    our own ``pickle`` framing so callers are not limited to ``PIPE_BUF``.
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    while True:
        try:
            conn.send_bytes(data)
            return
        except InterruptedError:
            continue
        except (BrokenPipeError, EOFError, OSError):
            raise


def _recv_object(conn, timeout: float | None = None) -> Any:
    """Receive a pickled object from a Connection, retrying on EINTR."""
    if timeout is not None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if conn.poll(timeout):
                    break
                raise TimeoutError("IPC read timed out")
            except InterruptedError:
                timeout = max(0.0, deadline - time.monotonic())
                continue
            except (BrokenPipeError, EOFError, OSError):
                raise

    while True:
        try:
            data = conn.recv_bytes()
            return pickle.loads(data)
        except InterruptedError:
            continue
        except (BrokenPipeError, EOFError, OSError):
            raise


def _run_one_job(job: _WorkerJob) -> _WorkerResult:
    """Execute a single job in the persistent worker."""
    try:
        deadline = (
            RoutingDeadline(expires_at=job.expires_at)
            if job.expires_at is not None
            else RoutingDeadline()
        )
        value = job.func(deadline, *job.args, **job.kwargs)
        return _WorkerResult(value=value, error=None, worker_pid=os.getpid())
    except BaseException as exc:
        return _WorkerResult(
            value=None, error=_serialize_error(exc), worker_pid=os.getpid()
        )


def _persistent_worker_entry(work_conn) -> None:
    """Main loop of the persistent disposable planning child."""
    try:
        while True:
            try:
                msg = _recv_object(work_conn)
            except EOFError:
                break
            if msg is None or isinstance(msg, _Stop):
                break
            if isinstance(msg, _WorkerJob):
                result = _run_one_job(msg)
                try:
                    _send_object(work_conn, result)
                except (BrokenPipeError, EOFError, OSError):
                    break
    finally:
        try:
            work_conn.close()
        except Exception:
            pass


def _reap_process(process: BaseProcess, timeout: float = 5.0) -> None:
    """Join a process; escalate to SIGTERM/SIGKILL if necessary."""
    if process.is_alive():
        process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)


class _WorkerHandle:
    """One persistent disposable child of the supervisor."""

    def __init__(self, ctx: ForkContext) -> None:
        self._ctx = ctx
        self._proc: BaseProcess | None = None
        self._work_conn: Any | None = None

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._shutdown_worker()
        work_parent, work_child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_persistent_worker_entry,
            args=(work_child,),
            daemon=False,
        )
        proc.start()
        work_child.close()
        self._proc = proc
        self._work_conn = work_parent

    def _shutdown_worker(self) -> None:
        if self._work_conn is not None:
            try:
                self._work_conn.close()
            except Exception:
                pass
            self._work_conn = None
        if self._proc is not None:
            _reap_process(self._proc, timeout=2.0)
            self._proc = None

    def shutdown(self) -> None:
        """Politely stop the worker, then reap it."""
        if (
            self._work_conn is not None
            and self._proc is not None
            and self._proc.is_alive()
        ):
            try:
                _send_object(self._work_conn, _Stop())
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._shutdown_worker()

    def run_job(self, job: _Job) -> _JobResult:
        """Forward a job to the persistent child and enforce its deadline."""
        self._ensure_worker()
        assert self._work_conn is not None and self._proc is not None

        worker_pid = self._proc.pid
        deadline_at = job.expires_at
        hard_at = (
            deadline_at + job.grace_seconds
            if deadline_at is not None and job.grace_seconds is not None
            else None
        )

        worker_job = _WorkerJob(
            func=job.func,
            args=job.args,
            kwargs=job.kwargs,
            expires_at=job.expires_at,
        )

        try:
            _send_object(self._work_conn, worker_job)
        except (BrokenPipeError, EOFError, OSError):
            pid = self._proc.pid if self._proc is not None else worker_pid
            self._shutdown_worker()
            error = SupervisorError("worker pipe closed before job could be sent")
            if pid is not None:
                error.worker_pid = pid  # type: ignore[attr-defined]
            return _JobResult(value=None, error=error, worker_pid=pid)

        terminated = False
        worker_result: _WorkerResult | None = None
        while True:
            now = time.monotonic()
            if hard_at is not None and now >= hard_at:
                # Hard stop: SIGKILL immediately and do not wait through a
                # graceful join sequence.
                pid = self._proc.pid if self._proc is not None else worker_pid
                if self._proc is not None and self._proc.is_alive():
                    self._proc.kill()
                    self._proc.join(timeout=1.0)
                self._shutdown_worker()
                error = RoutingTimeout("routing request deadline exceeded")
                if pid is not None:
                    error.worker_pid = pid  # type: ignore[attr-defined]
                return _JobResult(value=None, error=error, worker_pid=pid)
            if deadline_at is not None and not terminated and now >= deadline_at:
                if self._proc is not None and self._proc.is_alive():
                    self._proc.terminate()
                terminated = True
            try:
                worker_result = _recv_object(self._work_conn, timeout=0.05)
                break
            except TimeoutError:
                continue
            except (EOFError, BrokenPipeError, OSError):
                break

        if worker_result is None:
            pid = self._proc.pid if self._proc is not None else worker_pid
            self._shutdown_worker()
            if hard_at is not None:
                error: BaseException = RoutingTimeout(
                    "routing request deadline exceeded"
                )
            else:
                error = SupervisorError("worker process did not return a result")
            if pid is not None:
                error.worker_pid = pid  # type: ignore[attr-defined]
            return _JobResult(value=None, error=error, worker_pid=pid)

        # Worker responded.  If it returned an error, deserialize it; if it
        # returned a value, keep the child alive for the next job.
        if worker_result.error is not None:
            error = _deserialize_error(worker_result.error)
            if worker_result.worker_pid is not None:
                error.worker_pid = worker_result.worker_pid  # type: ignore[attr-defined]
            return _JobResult(value=None, error=error, worker_pid=worker_result.worker_pid)

        return _JobResult(
            value=worker_result.value,
            error=None,
            worker_pid=worker_result.worker_pid,
        )


def _supervisor_entry(
    cmd_conn,
    preload_marker: Any,
) -> None:
    """Main loop of the long-lived supervisor process."""
    _set_preload_marker(preload_marker)
    ctx: ForkContext = multiprocessing.get_context("fork")
    worker = _WorkerHandle(ctx)
    try:
        _send_object(cmd_conn, _Ready())
        while True:
            try:
                msg = _recv_object(cmd_conn)
            except EOFError:
                break
            if msg is None or isinstance(msg, _Stop):
                break
            if isinstance(msg, _Job):
                result = worker.run_job(msg)
                _send_object(cmd_conn, result)
    except (BrokenPipeError, OSError):
        pass
    finally:
        worker.shutdown()
        try:
            cmd_conn.close()
        except Exception:
            pass


class PreloadedRouteSupervisor:
    """Persistent fork-based supervisor for routing jobs.

    The caller preloads route assets in the current process and then starts
    the supervisor with ``start()``.  Because the supervisor is forked from
    the caller, it inherits the preload caches and marker without a
    per-request load.

    Jobs are run sequentially: ``run_job`` acquires a lock, so concurrent
    calls are serialized.  The supervisor maintains a single persistent
    disposable planning child.  Successful jobs reuse that child so mutated
    caches survive, while an unresponsive child is SIGTERM'd at the deadline
    and SIGKILL'd at ``deadline + grace``; a fresh child is then forked from
    the still-preloaded supervisor.

    IPC uses ``Connection.send_bytes`` / ``recv_bytes`` with explicit
    ``pickle`` framing so that payloads are not limited to ``PIPE_BUF``.

    Example::

        marker = preload_route_assets("graph.geojson")
        with PreloadedRouteSupervisor.start(preload_marker=marker) as sup:
            result = sup.run_job(plan_routes, request)

    On non-POSIX platforms ``start`` raises ``SupervisorError``.
    """

    def __init__(
        self,
        process: BaseProcess,
        cmd_conn: Any,
        default_deadline_seconds: float | None,
        default_grace_seconds: float,
    ) -> None:
        self._process = process
        self._cmd = cmd_conn
        self._default_deadline_seconds = default_deadline_seconds
        self._default_grace_seconds = default_grace_seconds
        self._lock = threading.Lock()
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
        cmd_parent, cmd_child = ctx.Pipe(duplex=True)
        proc = ctx.Process(
            target=_supervisor_entry,
            args=(cmd_child, preload_marker),
            daemon=False,
        )
        proc.start()
        cmd_child.close()

        try:
            if not cmd_parent.poll(timeout=start_timeout):
                proc.terminate()
                proc.join(timeout=5.0)
                raise SupervisorError(
                    "supervisor process did not become ready"
                )
            ready = _recv_object(cmd_parent)
            if not isinstance(ready, _Ready):
                raise SupervisorError(
                    f"supervisor sent unexpected ready message: {ready!r}"
                )
        except Exception:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5.0)
            cmd_parent.close()
            raise

        return cls(
            process=proc,
            cmd_conn=cmd_parent,
            default_deadline_seconds=default_deadline_seconds,
            default_grace_seconds=default_grace_seconds,
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
        which will be propagated to the caller where the exception class is
        available.

        ``deadline`` and ``deadline_seconds`` are mutually exclusive.  If
        ``deadline`` is supplied, its absolute ``expires_at`` is used and a
        caller-side cooperative check is performed before the job is sent.
        If ``deadline_seconds`` is supplied, ``expires_at`` is computed from
        the current monotonic clock before queue/IPC so delays do not extend
        the budget.  ``grace_seconds`` defaults to the value supplied to
        ``start``.
        """
        if self._closed:
            raise SupervisorError("supervisor is closed")
        if not self._process.is_alive():
            raise SupervisorError("supervisor process is not alive")

        if deadline is not None and deadline_seconds is not None:
            raise ValueError("deadline and deadline_seconds are mutually exclusive")

        if deadline is not None:
            # Caller-side cooperative cancellation/timeout check before IPC.
            deadline.check()
            expires_at = deadline.expires_at
        else:
            if deadline_seconds is None:
                deadline_seconds = self._default_deadline_seconds
            if deadline_seconds is not None:
                deadline_seconds = float(deadline_seconds)
                if deadline_seconds < 0.0 or not math.isfinite(deadline_seconds):
                    raise ValueError("deadline_seconds must be non-negative and finite")
                expires_at = time.monotonic() + deadline_seconds
            else:
                expires_at = None

        if grace_seconds is None:
            grace_seconds = self._default_grace_seconds
        if grace_seconds is not None:
            grace_seconds = float(grace_seconds)
            if grace_seconds < 0.0 or not math.isfinite(grace_seconds):
                raise ValueError("grace_seconds must be non-negative and finite")

        job = _Job(
            func=func,
            args=args,
            kwargs=kwargs,
            expires_at=expires_at,
            grace_seconds=grace_seconds,
        )

        with self._lock:
            try:
                _send_object(self._cmd, job)
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise SupervisorError(
                    "supervisor process closed unexpectedly"
                ) from exc
            try:
                result = _recv_object(self._cmd)
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise SupervisorError(
                    "supervisor process closed unexpectedly"
                ) from exc

        if not isinstance(result, _JobResult):
            raise SupervisorError(
                f"unexpected result from supervisor: {result!r}"
            )
        if result.error is not None:
            if result.worker_pid is not None:
                result.error.worker_pid = result.worker_pid  # type: ignore[attr-defined]
            raise result.error
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
            _send_object(self._cmd, _Stop())
        except (BrokenPipeError, EOFError, OSError):
            pass

        try:
            self._cmd.close()
        except Exception:
            pass

        if self._process.is_alive():
            self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2.0)

    def __enter__(self) -> "PreloadedRouteSupervisor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
