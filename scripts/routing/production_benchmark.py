"""Deterministic production-artifact routing benchmark.

The runner deliberately calls the strict ``plan_routes`` service response for
one preloaded graph/report and records every point in the q/kappa matrix.  It
never silently drops a case: a timeout or a no-route response is a row with an
exact reason.  Large output belongs under ignored ``data/processed``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import pickle
import struct
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import resource
import select
import signal
import sys
import atexit
import tempfile
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Callable, Iterator, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner import service as route_service  # noqa: E402
from src.route_planner.cancellation import (  # noqa: E402
    RoutingCancelled,
    RoutingDeadline,
    RoutingTimeout,
)
from src.route_planner.planner import ScenicRoutePlanner  # noqa: E402
from src.route_planner.service import (  # noqa: E402
    RouteRequest,
    clear_route_caches,
    plan_routes,
    preload_route_assets,
)


DEFAULT_CORPUS = Path("scripts/routing/production_benchmark_pairs.json")
DEFAULT_OUTPUT = Path(
    "data/processed/routing_benchmarks/production_artifact_benchmark.json"
)
DEFAULT_GRAPH = Path(
    "data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3"
)
DEFAULT_REPORT = Path(
    "data/processed/heuristic_runs/new_england_north_z14_v6_learned/report/report.json"
)

_BENCHMARK_IMPLEMENTATION_PATHS = {
    "production_benchmark": Path(__file__).resolve(),
    "route_cost": PROJECT_ROOT / "src/route_planner/cost.py",
    "route_edge_projection": PROJECT_ROOT / "src/route_planner/_edge_projection.py",
    "route_graph": PROJECT_ROOT / "src/route_planner/graph.py",
    "route_planner": PROJECT_ROOT / "src/route_planner/planner.py",
    "route_service": PROJECT_ROOT / "src/route_planner/service.py",
}
Q_VALUES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
KAPPA_VALUES = (1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 3.0)
HIGHWAY_TYPES = frozenset(
    {
        "highway",
        "motorway",
        "motorway_link",
        "primary",
        "primary_link",
        "trunk",
        "trunk_link",
    }
)
REQUIRED_CATEGORIES = (
    "short urban",
    "medium regional",
    "long regional",
    "obvious scenic candidate",
    "highway-sensitive",
    "no-highway-free",
    "detour1.0 binding",
    "detour1.8 unlock candidate",
    "checked-in default reproduction",
)


@dataclass(frozen=True)
class CaseSpec:
    pair_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    q: float
    kappa: float
    avoid_highways: bool


class CaseTimeout(TimeoutError):
    """A per-case alarm, retained as a distinct machine-readable reason."""


_SUPERVISOR_GRACE_SECONDS = 2.0
_HAS_POSIX_FORK = hasattr(os, "fork")

def _process_peak_rss_bytes() -> int:
    """Return this process's peak resident set in platform-independent bytes."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


@contextmanager
def _case_deadline(seconds: float) -> Iterator[RoutingDeadline]:
    """Yield one RoutingDeadline and arm SIGALRM as an outer fallback.

    A non-positive deadline disables both the cooperative deadline and the
    alarm; this is useful for focused tests.
    """
    if seconds <= 0.0 or not hasattr(signal, "SIGALRM"):
        yield RoutingDeadline()
        return

    deadline = RoutingDeadline.after(seconds)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise RoutingTimeout(f"case exceeded {seconds:g}s deadline")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    remaining = deadline.remaining_seconds()
    if remaining is None or remaining <= 0.0:
        remaining = 1e-9
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

def _reap_child(pid: int) -> None:
    try:
        while True:
            waited, _ = os.waitpid(pid, 0)
            if waited == pid:
                break
    except (ChildProcessError, OSError):
        pass


def _reconstruct_exception(name: str, message: str) -> BaseException:
    if name == "KeyboardInterrupt":
        raise KeyboardInterrupt(message)
    if name == "CaseTimeout" or name == "RoutingTimeout":
        return CaseTimeout(message)
    if name == "RoutingCancelled":
        return RoutingCancelled(message)
    return RuntimeError(f"{name}: {message}")


class _PersistentPlanningChild:
    """One forked planning child, reused across benchmark cases.

    The child inherits the parent's preloaded context via ``os.fork`` and
    keeps it warm across sequential requests.  The parent enforces a per-case
    deadline plus a small grace and only replaces the child when it becomes
    unresponsive.
    """

    _HEADER_FMT = "!I"
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(
        self,
        context: tuple[dict[str, Any], ScenicRoutePlanner, dict[str, Any], dict[str, Any]],
        route_error_cache: dict[tuple[str, bool], tuple[str, str]],
        grace_seconds: float = _SUPERVISOR_GRACE_SECONDS,
    ):
        self.context = context
        self.route_error_cache = route_error_cache
        self.grace_seconds = grace_seconds
        self._child_pid: int | None = None
        self._request_write: int | None = None
        self._response_read: int | None = None
        self._start()

    def _start(self) -> None:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(request_write)
            os.close(response_read)
            try:
                self._child_loop(request_read, response_write)
            finally:
                os._exit(0)
        os.close(request_read)
        os.close(response_write)
        self._child_pid = pid
        self._request_write = request_write
        self._response_read = response_read

    def _child_loop(self, request_read: int, response_write: int) -> None:
        while True:
            request = self._recv(request_read, timeout=None)
            if request is None:
                break
            try:
                row = _execute_case(
                    spec=request["spec"],
                    index=request["index"],
                    graph_path=Path(request["graph_path"]),
                    report_path=Path(request["report_path"]),
                    case_timeout_seconds=request["case_timeout_seconds"],
                    strict_service_full=request["strict_service_full"],
                    context=self.context,
                    route_error_cache=self.route_error_cache,
                )
                self._send(response_write, {"ok": row})
            except BaseException as error:
                self._send(
                    response_write,
                    {
                        "err": str(error),
                        "type": type(error).__name__,
                    },
                )

    def _write_all(
        self, fd: int, data: bytes, timeout: float | None
    ) -> None:
        deadline = monotonic() + timeout if timeout is not None else None
        while data:
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("planning child write timeout")
                _, ready, _ = select.select([], [fd], [], max(0.0, remaining))
                if not ready:
                    raise TimeoutError("planning child write timeout")
            try:
                written = os.write(fd, data)
            except InterruptedError:
                continue
            if written == 0:
                raise RuntimeError("zero write to planning child pipe")
            data = data[written:]

    def _send(self, fd: int, obj: Any, timeout: float | None = None) -> None:
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack(self._HEADER_FMT, len(payload))
        self._write_all(fd, header + payload, timeout)

    def _recv(self, fd: int, timeout: float | None) -> Any | None:
        header = self._read_exact(fd, self._HEADER_SIZE, timeout)
        if not header:
            return None
        (length,) = struct.unpack(self._HEADER_FMT, header)
        payload = self._read_exact(fd, length, timeout)
        if payload is None or len(payload) != length:
            raise RuntimeError("incomplete payload from planning child")
        return pickle.loads(payload)

    def _read_exact(
        self, fd: int, n: int, timeout: float | None
    ) -> bytes | None:
        deadline = monotonic() + timeout if timeout is not None else None
        seen = 0
        chunks: list[bytes] = []
        while seen < n:
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("planning child read timeout")
                ready, _, _ = select.select([fd], [], [], max(0.0, remaining))
                if not ready:
                    raise TimeoutError("planning child read timeout")
            else:
                select.select([fd], [], [], None)
            try:
                chunk = os.read(fd, n - seen)
            except InterruptedError:
                continue
            if not chunk:
                return None
            chunks.append(chunk)
            seen += len(chunk)
        return b"".join(chunks)

    def run(
        self,
        spec: CaseSpec,
        index: int,
        graph_path: Path,
        report_path: Path,
        case_timeout_seconds: float,
        strict_service_full: bool,
    ) -> dict[str, Any]:
        case_id = _case_id(spec)
        case_deadline_seconds = _case_deadline_seconds(case_id, case_timeout_seconds)
        request = {
            "spec": spec,
            "index": index,
            "graph_path": str(graph_path),
            "report_path": str(report_path),
            "case_timeout_seconds": case_timeout_seconds,
            "strict_service_full": strict_service_full,
        }
        self._send(self._request_write, request)
        try:
            response = self._recv(
                self._response_read,
                case_deadline_seconds + self.grace_seconds,
            )
        except TimeoutError:
            self._replace_child()
            raise CaseTimeout("case exceeded hard supervisor deadline")
        if response is None:
            self._replace_child()
            raise CaseTimeout("case exceeded hard supervisor deadline")
        if "ok" in response:
            return response["ok"]
        raise _reconstruct_exception(
            response.get("type", "RuntimeError"),
            response.get("err", ""),
        )

    def _replace_child(self) -> None:
        self._kill_and_reap()
        self._close_fds()
        self._start()

    def _kill_and_reap(self) -> None:
        if self._child_pid is not None:
            try:
                os.kill(self._child_pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            _reap_child(self._child_pid)
            self._child_pid = None

    def _close_fds(self) -> None:
        for fd in (self._request_write, self._response_read):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._request_write = None
        self._response_read = None

    def close(self) -> None:
        if self._request_write is not None:
            try:
                os.close(self._request_write)
            except OSError:
                pass
            self._request_write = None
        self._kill_and_reap()
        self._close_fds()

def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_identity(path: Path) -> dict[str, Any]:
    try:
        before = path.stat()
    except OSError:
        return {"exists": False}
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    after = path.stat()
    if (
        int(before.st_size) != int(after.st_size)
        or int(before.st_mtime_ns) != int(after.st_mtime_ns)
    ):
        raise RuntimeError(f"artifact changed while fingerprinting: {path}")
    return {
        "exists": True,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _checkpoint_fingerprint(
    *,
    corpus_path: Path,
    corpus: Mapping[str, Any],
    graph_path: Path,
    report_path: Path,
    case_timeout_seconds: float,
    strict_service_full: bool,
    workers: int,
    group_size: int,
) -> str:
    return _json_digest(
        {
            "fingerprint_schema_version": 2,
            "corpus": corpus,
            "graph": _path_identity(graph_path),
            "report": _path_identity(report_path),
            "implementation": {
                name: _path_identity(path)
                for name, path in _BENCHMARK_IMPLEMENTATION_PATHS.items()
            },
            "case_timeout_seconds": float(case_timeout_seconds),
            "strict_service_full": bool(strict_service_full),
            "workers": int(workers),
            "group_size": int(group_size),
            "q_values": list(Q_VALUES),
            "kappa_values": list(KAPPA_VALUES),
            "strict_service_case_ids": sorted(STRICT_SERVICE_CASE_IDS),
            "required_activation_case_ids": sorted(
                REQUIRED_ACTIVATION_CASE_IDS
            ),
            "activation_case_timeout_seconds": (
                ACTIVATION_CASE_TIMEOUT_SECONDS
            ),
        }
    )


def _haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*first, *second))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(h)))


def _normalise_score(raw: float) -> float:
    return min(max(float(raw), 0.0), 10.0) / 10.0


def _segment_rows(feature: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    properties = feature.get("properties", {})
    rows = properties.get("segment_identity", [])
    return [row for row in rows if isinstance(row, Mapping)]


def _edge_from_identity(
    row: Mapping[str, Any],
    edge_index: Mapping[str, Any] | None,
) -> Any | None:
    if edge_index is None:
        return None
    canonical_id = row.get("canonical_edge_id")
    if canonical_id is None:
        canonical_id = row.get("edge_id")
    if canonical_id is None:
        return None
    return edge_index.get(str(canonical_id))


def _close_enough(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return False
    return abs(first - second) <= 1e-8 * max(1.0, abs(first), abs(second))


def _dedupe_geometry_coordinates(
    coordinates: list[list[float]],
) -> list[list[float]]:
    result: list[list[float]] = []
    for coordinate in coordinates:
        if not result or coordinate != result[-1]:
            result.append(coordinate)
    return result


def _metadata_coordinate(value: Any) -> tuple[float, float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        return None
    latitude = float(value[0])
    longitude = float(value[1])
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        return None
    return latitude, longitude


def _geojson_coordinate(value: Any) -> tuple[float, float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        return None
    longitude = float(value[0])
    latitude = float(value[1])
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        return None
    return longitude, latitude


def _latlon_geojson_match(
    metadata: tuple[float, float] | None,
    geometry: tuple[float, float] | None,
) -> bool:
    return (
        metadata is not None
        and geometry is not None
        and _close_enough(metadata[0], geometry[1])
        and _close_enough(metadata[1], geometry[0])
    )


def _edge_row_is_partial(row: Mapping[str, Any], edge: Any) -> bool:
    """Return whether an emitted row describes less than a canonical edge."""
    source_edge_id = row.get("source_edge_id")
    source_fraction = row.get("source_fraction")
    row_distance = _as_float(row.get("distance_km"))
    edge_distance = _as_float(getattr(edge, "distance_km", None))
    return (
        source_edge_id is not None
        or source_fraction is not None
        or not _close_enough(row_distance, edge_distance)
    )


def _edge_partial_metrics_consistent(
    row: Mapping[str, Any],
    edge: Any,
) -> bool:
    """Validate emitted partial metrics against a canonical edge."""
    row_distance = _as_float(row.get("distance_km"))
    edge_distance = _as_float(getattr(edge, "distance_km", None))
    if (
        row_distance is None
        or row_distance < 0.0
        or edge_distance is None
        or edge_distance < 0.0
    ):
        return False
    canonical_id = str(getattr(edge, "id", ""))
    source_edge_id = row.get("source_edge_id")
    if source_edge_id is not None and str(source_edge_id) != canonical_id:
        return False
    raw_fraction = row.get("source_fraction")
    fraction = _as_float(raw_fraction)
    if raw_fraction is not None and fraction is None:
        return False
    if fraction is None:
        if edge_distance > 0.0:
            fraction = row_distance / edge_distance
        elif row_distance == 0.0:
            fraction = 0.0
        else:
            return False
    if fraction < 0.0 or fraction > 1.0:
        return False
    if not _close_enough(row_distance, edge_distance * fraction):
        return False
    row_duration = _as_float(row.get("duration_minutes"))
    edge_duration = _as_float(
        getattr(edge, "travel_time_minutes", None)
    )
    return (
        row_duration is not None
        and row_duration >= 0.0
        and edge_duration is not None
        and edge_duration >= 0.0
        and _close_enough(row_duration, edge_duration * fraction)
    )


def _segment_parameter(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float | None:
    """Locate a latitude/longitude point on one canonical edge segment."""
    lon_scale = math.cos(math.radians((start[0] + end[0]) / 2.0))
    delta_lat = end[0] - start[0]
    delta_lon = (end[1] - start[1]) * lon_scale
    denominator = delta_lat * delta_lat + delta_lon * delta_lon
    if denominator <= 1e-24:
        if _close_enough(point[0], start[0]) and _close_enough(
            point[1], start[1]
        ):
            return 0.0
        return None
    point_lat = point[0] - start[0]
    point_lon = (point[1] - start[1]) * lon_scale
    fraction = (
        point_lat * delta_lat + point_lon * delta_lon
    ) / denominator
    if fraction < -1e-8 or fraction > 1.0 + 1e-8:
        return None
    fraction = min(1.0, max(0.0, fraction))
    projected_lat = start[0] + fraction * (end[0] - start[0])
    projected_lon = start[1] + fraction * (end[1] - start[1])
    if not _close_enough(point[0], projected_lat) or not _close_enough(
        point[1], projected_lon
    ):
        return None
    return fraction


def _partial_edge_geometry_consistent(
    row: Mapping[str, Any],
    edge: Any,
    start_node: Any,
    end_node: Any,
    direction: str,
) -> bool:
    row_start = _metadata_coordinate(row.get("start"))
    row_end = _metadata_coordinate(row.get("end"))
    if row_start is None or row_end is None:
        return False
    canonical_start = (float(start_node.lat), float(start_node.lon))
    canonical_end = (float(end_node.lat), float(end_node.lon))
    edge_distance = _as_float(getattr(edge, "distance_km", None))
    row_distance = _as_float(row.get("distance_km"))
    if edge_distance == 0.0 and row_distance == 0.0:
        return (
            _close_enough(canonical_start[0], canonical_end[0])
            and _close_enough(canonical_start[1], canonical_end[1])
            and _close_enough(row_start[0], canonical_start[0])
            and _close_enough(row_start[1], canonical_start[1])
            and _close_enough(row_end[0], canonical_start[0])
            and _close_enough(row_end[1], canonical_start[1])
        )
    start_fraction = _segment_parameter(
        row_start, canonical_start, canonical_end
    )
    end_fraction = _segment_parameter(row_end, canonical_start, canonical_end)
    if start_fraction is None or end_fraction is None:
        return False
    source_fraction = _as_float(row.get("source_fraction"))
    if source_fraction is None:
        if edge_distance is None or edge_distance <= 0.0 or row_distance is None:
            source_fraction = 0.0 if row_distance == 0.0 else None
        else:
            source_fraction = row_distance / edge_distance
    if source_fraction is None or not _close_enough(
        abs(end_fraction - start_fraction), source_fraction
    ):
        return False
    if direction == "reverse":
        return end_fraction <= start_fraction + 1e-8
    return start_fraction <= end_fraction + 1e-8



def _topology_point_key(
    point: tuple[float, float],
    start_node: Any,
    end_node: Any,
) -> str:
    if _close_enough(point[0], float(start_node.lat)) and _close_enough(
        point[1], float(start_node.lon)
    ):
        return str(start_node.id)
    if _close_enough(point[0], float(end_node.lat)) and _close_enough(
        point[1], float(end_node.lon)
    ):
        return str(end_node.id)
    return f"@{point[0]:.12f},{point[1]:.12f}"


def _edge_topology_pair(
    row: Mapping[str, Any],
    edge: Any,
    node_index: Mapping[str, Any] | None,
) -> tuple[str, str] | None:
    reverse = str(row.get("direction", "forward")).lower() == "reverse"
    canonical_pair = (
        (str(edge.end_node_id), str(edge.start_node_id))
        if reverse
        else (str(edge.start_node_id), str(edge.end_node_id))
    )
    if not _edge_row_is_partial(row, edge) or node_index is None:
        return canonical_pair
    start_node = node_index.get(str(edge.start_node_id))
    end_node = node_index.get(str(edge.end_node_id))
    row_start = _metadata_coordinate(row.get("start"))
    row_end = _metadata_coordinate(row.get("end"))
    if (
        start_node is None
        or end_node is None
        or row_start is None
        or row_end is None
    ):
        return None
    return (
        _topology_point_key(row_start, start_node, end_node),
        _topology_point_key(row_end, start_node, end_node),
    )

def _edge_segment_consistent(
    edge_id: str,
    row: Mapping[str, Any],
    edge: Any,
    node_index: Mapping[str, Any] | None,
) -> bool:
    partial = _edge_row_is_partial(row, edge)
    if partial:
        if not _edge_partial_metrics_consistent(row, edge):
            return False
    elif not _close_enough(
        _as_float(row.get("distance_km")),
        _as_float(getattr(edge, "distance_km", None)),
    ):
        return False
    if not _close_enough(
        _as_float(row.get("scenic_score")),
        _as_float(getattr(edge, "scenic_score", None)),
    ):
        return False
    row_duration = _as_float(row.get("duration_minutes"))
    edge_duration_value = _as_float(
        getattr(edge, "travel_time_minutes", None)
    )
    if row_duration is None:
        if partial or node_index is not None:
            return False
    elif partial:
        # The proportional check above validates this duration.
        pass
    elif not _close_enough(row_duration, edge_duration_value):
        return False
    if str(row.get("road_type", "")).lower() != str(
        getattr(edge, "road_type", "")
    ).lower():
        return False
    direction = str(row.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse"}:
        return False
    if node_index is None:
        return True
    canonical_start_node = node_index.get(str(getattr(edge, "start_node_id", None)))
    canonical_end_node = node_index.get(str(getattr(edge, "end_node_id", None)))
    if canonical_start_node is None or canonical_end_node is None:
        return False
    if partial:
        return _partial_edge_geometry_consistent(
            row,
            edge,
            canonical_start_node,
            canonical_end_node,
            direction,
        )
    reverse = direction == "reverse"
    start_node = canonical_end_node if reverse else canonical_start_node
    end_node = canonical_start_node if reverse else canonical_end_node
    row_start = row.get("start")
    row_end = row.get("end")
    return (
        isinstance(row_start, (list, tuple))
        and isinstance(row_end, (list, tuple))
        and len(row_start) == 2
        and len(row_end) == 2
        and _close_enough(_as_float(row_start[0]), float(start_node.lat))
        and _close_enough(_as_float(row_start[1]), float(start_node.lon))
        and _close_enough(_as_float(row_end[0]), float(end_node.lat))
        and _close_enough(_as_float(row_end[1]), float(end_node.lon))
    )


def recompute_feature_metrics(
    feature: Mapping[str, Any],
    *,
    edge_index: Mapping[str, Any] | None = None,
    node_index: Mapping[str, Any] | None = None,
    requested_start: tuple[float, float] | None = None,
    requested_end: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Recompute metrics from emitted identities, segments, and graph edges.

    Aggregate response fields are never used as an unverified duration source.
    Segment durations must be present when no canonical edge index is supplied;
    production passes the preloaded scored-edge index and checks both sources.
    """
    properties = feature.get("properties", {})
    rows = _segment_rows(feature)
    traversal_ids = [
        str(row.get("traversal_id", row.get("edge_id", "")))
        for row in rows
    ]
    canonical_ids = [
        str(row.get("canonical_edge_id", row.get("edge_id", "")))
        for row in rows
    ]
    edge_ids = traversal_ids
    declared_edge_ids = properties.get("edge_ids")
    declared_traversal_ids = properties.get("traversal_ids")
    declared_edge_ids = (
        [str(edge_id) for edge_id in declared_edge_ids]
        if isinstance(declared_edge_ids, (list, tuple))
        else None
    )
    declared_traversal_ids = (
        [str(edge_id) for edge_id in declared_traversal_ids]
        if isinstance(declared_traversal_ids, (list, tuple))
        else None
    )
    top_level_edge_ids_match = (
        declared_edge_ids == canonical_ids
        and declared_traversal_ids == traversal_ids
    )
    segment_distance = sum(
        max(0.0, _as_float(row.get("distance_km"), 0.0) or 0.0)
        for row in rows
    )
    segment_scenic_distance = sum(
        max(0.0, _as_float(row.get("distance_km"), 0.0) or 0.0)
        * (_as_float(row.get("scenic_score"), 0.0) or 0.0)
        for row in rows
    )
    segment_raw_score = (
        segment_scenic_distance / segment_distance
        if segment_distance > 0.0
        else 0.0
    )
    segment_normalized_score = _normalise_score(segment_raw_score)
    geometry_value = feature.get("geometry", {})
    geometry_coordinates = (
        geometry_value.get("coordinates", [])
        if isinstance(geometry_value, Mapping)
        else []
    )
    geometry_points = (
        [_geojson_coordinate(coordinate) for coordinate in geometry_coordinates]
        if isinstance(geometry_coordinates, list)
        else []
    )
    actual_snapped_start = (
        _metadata_coordinate(properties.get("snapped_start"))
    )
    actual_snapped_end = _metadata_coordinate(properties.get("snapped_end"))
    snapped_endpoints_ok = (
        len(geometry_points) >= 2
        and all(point is not None for point in geometry_points)
        and _latlon_geojson_match(actual_snapped_start, geometry_points[0])
        and _latlon_geojson_match(actual_snapped_end, geometry_points[-1])
    )
    expected_requested_start = _metadata_coordinate(requested_start)
    expected_requested_end = _metadata_coordinate(requested_end)
    actual_requested_start = _metadata_coordinate(
        properties.get("requested_start")
    )
    actual_requested_end = _metadata_coordinate(properties.get("requested_end"))
    if requested_start is None and requested_end is None:
        requested_endpoints_ok = (
            actual_requested_start is not None
            and actual_requested_end is not None
        )
    else:
        requested_endpoints_ok = (
            expected_requested_start is not None
            and expected_requested_end is not None
            and actual_requested_start == expected_requested_start
            and actual_requested_end == expected_requested_end
        )
    zero_edge_geometry_ok = (
        not rows
        and len(geometry_points) == 2
        and all(point is not None for point in geometry_points)
        and _close_enough(
            geometry_points[0][0], geometry_points[1][0]
        )
        and _close_enough(
            geometry_points[0][1], geometry_points[1][1]
        )
    )
    zero_edge_route = zero_edge_geometry_ok
    segment_durations = [
        _as_float(row.get("duration_minutes")) for row in rows
    ]
    segment_duration_available = (
        zero_edge_route
        or (
            bool(rows)
            and all(
                duration is not None for duration in segment_durations
            )
        )
    )
    segment_duration = (
        0.0
        if zero_edge_route
        else (
            sum(float(duration) for duration in segment_durations)
            if segment_duration_available
            else None
        )
    )
    segment_highway_count = sum(
        1
        for row in rows
        if str(row.get("road_type", "")).lower() in HIGHWAY_TYPES
    )
    canonical_edges = [
        _edge_from_identity(row, edge_index)
        for row in rows
    ]
    edge_metrics_available = bool(edge_ids) and all(
        edge is not None for edge in canonical_edges
    )
    topology_declared = edge_metrics_available and all(
        hasattr(edge, "start_node_id") and hasattr(edge, "end_node_id")
        for edge in canonical_edges
    )
    topology_pairs = (
        [
            _edge_topology_pair(row, edge, node_index)
            for row, edge in zip(rows, canonical_edges)
        ]
        if topology_declared
        else []
    )
    topology_available = topology_declared and all(
        pair is not None for pair in topology_pairs
    )
    oriented_node_pairs = (
        [pair for pair in topology_pairs if pair is not None]
        if topology_available
        else []
    )
    route_nodes: list[str] = []
    if oriented_node_pairs:
        route_nodes.append(oriented_node_pairs[0][0])
        for _, end_node in oriented_node_pairs:
            if route_nodes[-1] != end_node:
                route_nodes.append(end_node)
    continuity_ok = not topology_available or all(
        previous[1] == current[0] and not previous[1].startswith("@")
        for previous, current in zip(oriented_node_pairs, oriented_node_pairs[1:])
    )
    simple_path_ok = not topology_available or len(route_nodes) == len(set(route_nodes))
    traversal_identity_ok = all(
        str(row.get("traversal_id", ""))
        == f"{index}:{str(row.get('direction', 'forward'))}:{canonical_id}"
        for index, (row, canonical_id) in enumerate(zip(rows, canonical_ids))
    )
    segment_geometry = (
        [
            [float(rows[0]["start"][1]), float(rows[0]["start"][0])]
        ]
        + [
            [float(row["end"][1]), float(row["end"][0])]
            for row in rows
        ]
        if rows
        and all(
            isinstance(row.get("start"), (list, tuple))
            and len(row["start"]) == 2
            and isinstance(row.get("end"), (list, tuple))
            and len(row["end"]) == 2
            for row in rows
        )
        else []
    )
    segment_start_continuity_ok = all(
        (
            (previous_end := _metadata_coordinate(previous.get("end")))
            is not None
        )
        and (
            (current_start := _metadata_coordinate(current.get("start")))
            is not None
        )
        and _close_enough(previous_end[0], current_start[0])
        and _close_enough(previous_end[1], current_start[1])
        for previous, current in zip(rows, rows[1:])
    )
    if rows:
        geometry_sequence_coordinates = (
            _dedupe_geometry_coordinates(geometry_coordinates)
            if isinstance(geometry_coordinates, list)
            and all(
                isinstance(coordinate, list)
                for coordinate in geometry_coordinates
            )
            else geometry_coordinates
        )
        expected_geometry = _dedupe_geometry_coordinates(segment_geometry)
        geometry_sequence_ok = (
            segment_start_continuity_ok
            and geometry_sequence_coordinates == expected_geometry
        )
    else:
        geometry_sequence_ok = zero_edge_geometry_ok
    edge_segment_consistency = (
        all(
            _edge_segment_consistent(
                edge_id, row, edge, node_index
            )
            for edge_id, row, edge in zip(
                edge_ids, rows, canonical_edges
            )
        )
        if edge_metrics_available
        else False
    )
    partial_rows = (
        edge_metrics_available
        and any(
            _edge_row_is_partial(row, edge)
            for row, edge in zip(rows, canonical_edges)
        )
    )
    emitted_partial_metrics_trusted = (
        bool(partial_rows) and edge_segment_consistency
    )
    edge_distance = (
        segment_distance
        if emitted_partial_metrics_trusted
        else (
            sum(max(0.0, float(edge.distance_km)) for edge in canonical_edges)
            if edge_metrics_available
            else None
        )
    )
    edge_scenic_distance = (
        segment_scenic_distance
        if emitted_partial_metrics_trusted
        else (
            sum(
                max(0.0, float(edge.distance_km)) * float(edge.scenic_score)
                for edge in canonical_edges
            )
            if edge_metrics_available
            else None
        )
    )
    edge_raw_score = (
        edge_scenic_distance / edge_distance
        if edge_scenic_distance is not None
        and edge_distance is not None
        and edge_distance > 0.0
        else (
            0.0
            if edge_scenic_distance == 0.0 and edge_distance == 0.0
            else None
        )
    )
    edge_duration = (
        segment_duration
        if emitted_partial_metrics_trusted and segment_duration_available
        else (
            sum(float(edge.travel_time_minutes) for edge in canonical_edges)
            if edge_metrics_available
            else None
        )
    )
    edge_highway_count = (
        segment_highway_count
        if emitted_partial_metrics_trusted
        else (
            sum(
                1
                for edge in canonical_edges
                if str(edge.road_type).lower() in HIGHWAY_TYPES
            )
            if edge_metrics_available
            else None
        )
    )
    declared_distance = _as_float(properties.get("total_distance_km"))
    declared_raw_score = _as_float(properties.get("raw_scenic_score"))
    declared_average_score = _as_float(properties.get("average_scenic_score"))
    declared_normalized_score = _as_float(
        properties.get("normalized_scenic_score")
    )
    declared_duration = _as_float(properties.get("estimated_duration_minutes"))
    declared_highway_count = int(properties.get("highway_count", 0) or 0)
    segment_consistency = {
        "top_level_edge_ids": top_level_edge_ids_match,
        "distance": _close_enough(segment_distance, declared_distance),
        "segments": True,
        "raw_scenic_score": _close_enough(
            segment_raw_score, declared_raw_score
        ),
        "average_scenic_score": _close_enough(
            segment_raw_score, declared_average_score
        ),
        "normalized_scenic_score": _close_enough(
            segment_normalized_score, declared_normalized_score
        ),
        "duration": segment_duration_available
        and _close_enough(segment_duration, declared_duration),
        "highway_count": segment_highway_count == declared_highway_count,
        "geometry_sequence": geometry_sequence_ok,
        "requested_endpoints": requested_endpoints_ok,
        "snapped_endpoints": snapped_endpoints_ok,
    }
    if edge_metrics_available:
        edge_consistency = {
            "top_level_edge_ids": top_level_edge_ids_match,
            "segments": edge_segment_consistency,
            "continuity": continuity_ok,
            "simple_path": simple_path_ok,
            "traversal_identity": traversal_identity_ok,
            "geometry_sequence": geometry_sequence_ok,
            "requested_endpoints": requested_endpoints_ok,
            "snapped_endpoints": snapped_endpoints_ok,
            "distance": segment_consistency["distance"]
            and _close_enough(edge_distance, segment_distance),
            "raw_scenic_score": segment_consistency["raw_scenic_score"]
            and _close_enough(edge_raw_score, segment_raw_score),
            "average_scenic_score": segment_consistency[
                "average_scenic_score"
            ]
            and _close_enough(edge_raw_score, segment_raw_score),
            "normalized_scenic_score": segment_consistency[
                "normalized_scenic_score"
            ]
            and _close_enough(
                _normalise_score(edge_raw_score or 0.0),
                segment_normalized_score,
            ),
            "duration": segment_consistency["duration"]
            and _close_enough(edge_duration, segment_duration),
            "highway_count": segment_consistency["highway_count"]
            and edge_highway_count == segment_highway_count,
        }
        if node_index is None and rows and not any(
            "duration_minutes" in row
            or "direction" in row
            or "canonical_edge_id" in row
            or "traversal_id" in row
            for row in rows
        ):
            edge_consistency = {
                "distance": bool(edge_consistency["distance"]),
                "scenic_score": bool(edge_consistency["raw_scenic_score"]),
                "duration": _close_enough(
                    edge_duration, declared_duration
                ),
                "highway_count": bool(edge_consistency["highway_count"]),
                "requested_endpoints": requested_endpoints_ok,
                "snapped_endpoints": snapped_endpoints_ok,
            }
    else:
        edge_consistency = segment_consistency
    distance = edge_distance if edge_distance is not None else segment_distance
    raw_score = (
        edge_raw_score if edge_raw_score is not None else segment_raw_score
    )
    highway_count = (
        edge_highway_count
        if edge_highway_count is not None
        else segment_highway_count
    )
    duration = (
        edge_duration
        if edge_duration is not None
        else segment_duration
    )
    geometry = feature.get("geometry", {})
    coordinates = (
        geometry.get("coordinates", [])
        if isinstance(geometry, Mapping)
        else []
    )
    return {
        "distance_km": float(distance),
        "distance_km_segments": float(segment_distance),
        "raw_scenic_score": float(raw_score),
        "raw_scenic_score_segments": float(segment_raw_score),
        "normalized_scenic_score": float(_normalise_score(raw_score)),
        "highway_count": int(highway_count),
        "segment_count": len(edge_ids),
        "segment_identity_sha256": _json_digest(edge_ids),
        "segment_identity_first": edge_ids[:3],
        "segment_identity_last": edge_ids[-3:] if edge_ids else [],
        "geometry_sha256": _json_digest(coordinates),
        "geometry_point_count": len(coordinates),
        "geometry_start": coordinates[0] if coordinates else None,
        "geometry_end": coordinates[-1] if coordinates else None,
        "distance_declared_km": declared_distance,
        "raw_scenic_declared": declared_raw_score,
        "average_scenic_score_declared": declared_average_score,
        "normalized_scenic_declared": declared_normalized_score,
        "duration_minutes_declared": declared_duration,
        "duration_minutes_segments": segment_duration,
        "duration_minutes_recomputed": duration,
        "duration_minutes": duration,
        "highway_count_declared": declared_highway_count,
        "edge_metric_consistency": edge_consistency,
        "requested_endpoints": requested_endpoints_ok,
        "snapped_endpoints": snapped_endpoints_ok,
        "exactness_status": properties.get("exactness_status"),
        "objective_value_declared": _as_float(properties.get("objective_value")),
        "optimality_gap": _as_float(properties.get("optimality_gap")),
        "certified_upper_bound": _as_float(
            properties.get("certified_upper_bound")
        ),
        "edge_ids": edge_ids,
        "canonical_edge_ids": canonical_ids,
        "traversal_ids": traversal_ids,
        "segment_identity": [dict(row) for row in rows],
        "requested_scenic_weight": _as_float(
            properties.get("requested_scenic_weight")
        ),
        "applied_scenic_weight": _as_float(
            properties.get("applied_scenic_weight")
        ),
        "requested_max_detour_factor": _as_float(
            properties.get("requested_max_detour_factor")
        ),
        "applied_max_detour_factor": _as_float(
            properties.get("applied_max_detour_factor")
        ),
        "scenic_score_delta_absolute": _as_float(
            properties.get("scenic_score_delta_absolute")
        ),
        "scenic_score_delta_relative": _as_float(
            properties.get("scenic_score_delta_relative")
        ),
        "same_route": properties.get("same_route"),
        "no_better_route_reason": properties.get("no_better_route_reason"),
    }


def _duration_utility(
    scenic_duration: float, fastest_duration: float, kappa: float
) -> float:
    values = (scenic_duration, fastest_duration, kappa)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("duration objective inputs must be finite")
    if scenic_duration < 0.0 or fastest_duration < 0.0 or kappa < 1.0:
        raise ValueError("duration objective inputs are outside their domains")
    if kappa == 1.0:
        return 1.0 if scenic_duration == fastest_duration else 0.0
    if fastest_duration == 0.0:
        return 1.0 if scenic_duration == 0.0 else 0.0
    value = (
        kappa * fastest_duration - scenic_duration
    ) / ((kappa - 1.0) * fastest_duration)
    return min(1.0, max(0.0, value))


def recompute_objective(
    *,
    q: float,
    kappa: float,
    scenic: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    scenic_priority: bool = False,
) -> dict[str, float | None]:
    """Recompute service objective components from independent path metrics."""
    scenic_duration = _as_float(scenic.get("duration_minutes_recomputed"))
    fastest_duration = None
    if baseline is not None:
        fastest_duration = _as_float(
            baseline.get("duration_minutes_recomputed")
        )
    else:
        fastest_duration = scenic_duration
    scenic_utility = _normalise_score(
        _as_float(scenic.get("raw_scenic_score"), 0.0) or 0.0
    )
    baseline_raw = (
        _as_float(baseline.get("raw_scenic_score"))
        if baseline is not None
        else None
    )
    absolute = (
        (_as_float(scenic.get("raw_scenic_score"), 0.0) or 0.0) - baseline_raw
        if baseline_raw is not None
        else None
    )
    relative = (
        absolute / abs(baseline_raw)
        if absolute is not None and baseline_raw
        else None
    )
    if scenic_duration is None or fastest_duration is None:
        return {
            "duration_utility": None,
            "scenic_utility": float(scenic_utility),
            "objective_value": None,
            "actual_duration_ratio": None,
            "scenic_score_delta_absolute": absolute,
            "scenic_score_delta_relative": relative,
        }
    ratio = (
        scenic_duration / fastest_duration
        if fastest_duration > 0.0
        else (1.0 if scenic_duration == 0.0 else math.inf)
    )
    duration_utility = _duration_utility(
        scenic_duration, fastest_duration, kappa
    )
    objective = scenic_utility if scenic_priority else (
        (1.0 - q) * duration_utility + q * scenic_utility
    )
    return {
        "duration_utility": float(duration_utility),
        "scenic_utility": float(scenic_utility),
        "objective_value": float(objective),
        "actual_duration_ratio": ratio,
        "scenic_score_delta_absolute": absolute,
        "scenic_score_delta_relative": relative,
    }


def _route_snapshot(
    feature: Mapping[str, Any],
    *,
    edge_index: Mapping[str, Any] | None = None,
    node_index: Mapping[str, Any] | None = None,
    requested_start: tuple[float, float] | None = None,
    requested_end: tuple[float, float] | None = None,
) -> dict[str, Any]:
    metrics = recompute_feature_metrics(
        feature,
        edge_index=edge_index,
        node_index=node_index,
        requested_start=requested_start,
        requested_end=requested_end,
    )
    properties = feature.get("properties", {})
    declared = {
        "objective_value": _as_float(properties.get("objective_value")),
        "duration_utility": _as_float(properties.get("duration_utility")),
        "actual_duration_ratio": _as_float(properties.get("actual_duration_ratio")),
    }
    metrics["objective_declared"] = declared["objective_value"]
    metrics["duration_utility_declared"] = declared["duration_utility"]
    metrics["actual_duration_ratio_declared"] = declared["actual_duration_ratio"]
    return metrics


def _route_metrics_row_consistent(
    route_row: Any,
    feature: Mapping[str, Any],
) -> bool:
    if not isinstance(route_row, Mapping):
        return False
    properties = feature.get("properties", {})
    scalar_fields = (
        ("total_distance_km", "total_distance_km"),
        ("average_scenic_score", "average_scenic_score"),
        ("raw_scenic_score", "raw_scenic_score"),
        ("normalized_scenic_score", "normalized_scenic_score"),
        ("estimated_duration_minutes", "estimated_duration_minutes"),
        ("highway_count", "highway_count"),
        ("objective_value", "objective_value"),
    )
    if any(
        not _close_enough(
            _as_float(route_row.get(row_key)),
            _as_float(properties.get(feature_key)),
        )
        for row_key, feature_key in scalar_fields
    ):
        return False
    if route_row.get("edge_ids") != properties.get("edge_ids"):
        return False
    if route_row.get("segment_identity") != properties.get("segment_identity"):
        return False
    return True


def _certification_consistent(
    route: Mapping[str, Any],
    objective_value: float | None,
) -> bool:
    status = str(route.get("exactness_status") or "")
    gap = _as_float(route.get("optimality_gap"))
    upper_bound = _as_float(route.get("certified_upper_bound"))
    if objective_value is None:
        return False
    if status == "exact":
        return (gap is None or _close_enough(gap, 0.0)) and (
            upper_bound is None
            or _close_enough(upper_bound, float(objective_value))
        )
    if status == "approximate-certified":
        return (
            gap is not None
            and upper_bound is not None
            and upper_bound + 1e-8 >= float(objective_value)
            and _close_enough(
                gap, upper_bound - float(objective_value)
            )
        )
    return False

def _no_better_reason_consistent(
    route: Mapping[str, Any],
    objective_value: float | None,
) -> bool:
    reason = route.get("no_better_route_reason")
    uplift = _as_float(route.get("scenic_score_delta_absolute"))
    same_route = route.get("same_route")
    status = str(route.get("exactness_status") or "")
    gap = _as_float(route.get("optimality_gap"))
    if same_route is True:
        if gap is not None and gap > 1e-8:
            return reason == "approximation_did_not_find_scenic_improvement"
        return reason == "same_route"
    if uplift is not None and uplift <= 0.0:
        if gap is not None and gap > 1e-8:
            return reason == "approximation_did_not_find_scenic_improvement"
        return reason == "no_better_route"
    return reason is None

def evaluate_service_response(
    result: Mapping[str, Any],
    *,
    q: float,
    kappa: float,
    avoid_highways: bool,
    edge_index: Mapping[str, Any] | None = None,
    node_index: Mapping[str, Any] | None = None,
    requested_start: tuple[float, float] | None = None,
    requested_end: tuple[float, float] | None = None,
    scenic_priority: bool = False,
) -> dict[str, Any]:
    """Evaluate a strict response against canonical graph-edge metrics."""
    routes = {
        str(row.get("route_kind")): row.get("metrics", {})
        for row in result.get("routes", [])
        if isinstance(row, Mapping)
    }
    features = {
        str(feature.get("properties", {}).get("route_kind")): feature
        for feature in result.get("geojson", {}).get("features", [])
        if isinstance(feature, Mapping)
    }
    scenic_feature = features.get("scenic")
    baseline_feature = features.get("baseline")
    if scenic_feature is None:
        raise ValueError("strict response did not contain a scenic feature")
    scenic = _route_snapshot(
        scenic_feature,
        edge_index=edge_index,
        node_index=node_index,
        requested_start=requested_start,
        requested_end=requested_end,
    )
    baseline = (
        _route_snapshot(
            baseline_feature,
            edge_index=edge_index,
            node_index=node_index,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        if baseline_feature is not None
        else None
    )
    recomputed = recompute_objective(
        q=q,
        kappa=kappa,
        scenic=scenic,
        baseline=baseline,
        scenic_priority=scenic_priority,
    )
    declared_properties = scenic_feature.get("properties", {})
    declared_objective = _as_float(
        declared_properties.get(
            "objective_value",
            routes.get("scenic", {}).get("objective_value"),
        )
    )
    recomputed_objective = _as_float(recomputed.get("objective_value"))
    objective_error = (
        abs(declared_objective - recomputed_objective)
        if declared_objective is not None and recomputed_objective is not None
        else None
    )
    scenic_duration = _as_float(scenic.get("duration_minutes"))
    fastest_duration = (
        _as_float(baseline.get("duration_minutes"))
        if baseline is not None
        else None
    )
    duration_available = (
        scenic_duration is not None
        and fastest_duration is not None
    )
    if duration_available:
        ratio = (
            1.0
            if scenic_duration == 0.0 and fastest_duration == 0.0
            else (
                scenic_duration / fastest_duration
                if fastest_duration > 0.0
                else None
            )
        )
    else:
        ratio = None
    baseline_present = baseline is not None
    cap_ok = (
        baseline_present
        and duration_available
        and (
            scenic_duration <= fastest_duration * kappa + 1e-9
            if fastest_duration > 0.0
            else scenic_duration == 0.0
        )
    )
    highway_ok = (
        baseline_present
        and not avoid_highways
        or (
            baseline_present
            and avoid_highways
            and scenic["highway_count"] == 0
            and baseline["highway_count"] == 0
        )
    )
    q0_fastest = (
        baseline_present
        and duration_available
        and (q != 0.0 or _close_enough(scenic_duration, fastest_duration))
    )
    edge_consistency = [
        route.get("edge_metric_consistency")
        for route in (scenic, baseline)
        if route is not None and route.get("edge_metric_consistency") is not None
    ]
    edge_metrics_ok = bool(edge_consistency) and all(
        all(consistency.values()) for consistency in edge_consistency
    )
    route_metrics_ok = all(
        _route_metrics_row_consistent(
            routes.get(route_kind), feature
        )
        for route_kind, feature in (
            ("scenic", scenic_feature),
            ("baseline", baseline_feature),
        )
    )
    diagnostics_payload = result.get("diagnostics", {})
    request_payload = result.get("request", {})
    settings_ok = (
        isinstance(request_payload, Mapping)
        and _close_enough(_as_float(request_payload.get("scenic_weight")), q)
        and _close_enough(
            _as_float(request_payload.get("max_detour_factor")), kappa
        )
        and request_payload.get("avoid_highways") is bool(avoid_highways)
        and all(
            _close_enough(route.get("requested_scenic_weight"), q)
            and _close_enough(route.get("applied_scenic_weight"), q)
            and _close_enough(route.get("requested_max_detour_factor"), kappa)
            and _close_enough(route.get("applied_max_detour_factor"), kappa)
            for route in (scenic, baseline)
            if route is not None
        )
        and isinstance(diagnostics_payload, Mapping)
        and _close_enough(
            _as_float(diagnostics_payload.get("requested_scenic_weight")), q
        )
        and _close_enough(
            _as_float(diagnostics_payload.get("applied_scenic_weight")), q
        )
        and _close_enough(
            _as_float(diagnostics_payload.get("requested_max_detour_factor")),
            kappa,
        )
        and _close_enough(
            _as_float(diagnostics_payload.get("applied_max_detour_factor")),
            kappa,
        )
        and diagnostics_payload.get("avoid_highways_applied")
        is bool(avoid_highways)
    )
    derived_same_route = (
        baseline is not None
        and scenic.get("traversal_ids") == baseline.get("traversal_ids")
    )
    recomputed_absolute = _as_float(
        recomputed.get("scenic_score_delta_absolute")
    )
    recomputed_relative = _as_float(
        recomputed.get("scenic_score_delta_relative")
    )
    declared_comparison_ok = (
        scenic.get("same_route") is derived_same_route
        and _close_enough(
            _as_float(scenic.get("scenic_score_delta_absolute")),
            recomputed_absolute,
        )
        and (
            (
                scenic.get("scenic_score_delta_relative") is None
                and recomputed_relative is None
            )
            or _close_enough(
                _as_float(scenic.get("scenic_score_delta_relative")),
                recomputed_relative,
            )
        )
    )
    certification_data_finite = all(
        value is None
        or (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
        for value in (
            declared_properties.get("optimality_gap"),
            declared_properties.get("certified_upper_bound"),
        )
    )
    certification_ok = certification_data_finite and _certification_consistent(
        scenic, recomputed_objective
    )
    reason_ok = _no_better_reason_consistent(
        scenic, recomputed_objective
    )
    diagnostics_consistency = (
        isinstance(diagnostics_payload, Mapping)
        and diagnostics_payload.get("exactness_status")
        == scenic.get("exactness_status")
        and (
            (
                diagnostics_payload.get("optimality_gap") is None
                and scenic.get("optimality_gap") is None
            )
            or _close_enough(
                _as_float(diagnostics_payload.get("optimality_gap")),
                _as_float(scenic.get("optimality_gap")),
            )
        )
        and (
            (
                diagnostics_payload.get("certified_upper_bound") is None
                and scenic.get("certified_upper_bound") is None
            )
            or _close_enough(
                _as_float(diagnostics_payload.get("certified_upper_bound")),
                _as_float(scenic.get("certified_upper_bound")),
            )
        )
        and _close_enough(
            _as_float(diagnostics_payload.get("normalized_scenic_score")),
            _as_float(scenic.get("normalized_scenic_score")),
        )
        and _close_enough(
            _as_float(diagnostics_payload.get("scenic_score_delta_absolute")),
            _as_float(scenic.get("scenic_score_delta_absolute")),
        )
        and (
            (
                diagnostics_payload.get("scenic_score_delta_relative") is None
                and scenic.get("scenic_score_delta_relative") is None
            )
            or _close_enough(
                _as_float(diagnostics_payload.get("scenic_score_delta_relative")),
                _as_float(scenic.get("scenic_score_delta_relative")),
            )
        )
        and diagnostics_payload.get("same_route")
        == scenic.get("same_route")
        and diagnostics_payload.get("no_better_route_reason")
        == scenic.get("no_better_route_reason")
    )
    failed_invariants = [
        name
        for name, passed in {
            "baseline_present": baseline_present,
            "duration_cap": cap_ok,
            "prohibited_highways": highway_ok,
            "q0_fastest": q0_fastest,
            "objective_recomputation": objective_error is not None
            and objective_error <= 1e-8,
            "edge_metric_recomputation": edge_metrics_ok,
            "route_metrics_consistency": route_metrics_ok,
            "diagnostics_consistency": diagnostics_consistency,
            "settings_consistency": settings_ok,
            "route_comparison_recomputation": declared_comparison_ok,
            "certification_consistency": certification_ok,
            "no_better_reason_consistency": reason_ok,
        }.items()
        if not passed
    ]
    return {
        "status": "ok" if not failed_invariants else "invalid",
        "diagnostics": dict(result.get("diagnostics", {})),
        "score_mapping": dict(result.get("score_mapping", {})),
        "routes": {"scenic": scenic, "baseline": baseline},
        "objective": {
            "declared": declared_objective,
            "recomputed": recomputed["objective_value"],
            "absolute_error": objective_error,
            "components_recomputed": recomputed,
        },
        "distance_km": scenic["distance_km"],
        "duration_minutes": scenic_duration,
        "duration_ratio": ratio,
        "raw_scenic_score": scenic["raw_scenic_score"],
        "normalized_scenic_score": scenic["normalized_scenic_score"],
        "uplift_absolute": recomputed["scenic_score_delta_absolute"],
        "uplift_relative": recomputed["scenic_score_delta_relative"],
        "highway_count": scenic["highway_count"],
        "exactness_status": scenic["exactness_status"],
        "optimality_gap": scenic["optimality_gap"],
        "certified_upper_bound": scenic["certified_upper_bound"],
        "no_route_reason": declared_properties.get("no_route_reason"),
        "failed_invariants": list(failed_invariants),
        "invariants": {
            "baseline_present": bool(baseline_present),
            "duration_cap": bool(cap_ok),
            "prohibited_highways": bool(highway_ok),
            "q0_fastest": bool(q0_fastest),
            "objective_recomputation": objective_error is not None
            and objective_error <= 1e-8,
            "edge_metric_recomputation": bool(edge_metrics_ok),
            "route_metrics_consistency": bool(route_metrics_ok),
            "diagnostics_consistency": bool(diagnostics_consistency),
            "settings_consistency": bool(settings_ok),
            "certification_data_finite": bool(certification_data_finite),
            "route_comparison_recomputation": bool(declared_comparison_ok),
            "certification_consistency": bool(certification_ok),
            "no_better_reason_consistency": bool(reason_ok),
        },
    }


def _error_reason(error: BaseException) -> str:
    if isinstance(error, (CaseTimeout, RoutingTimeout, TimeoutError)):
        return "timeout"
    if isinstance(error, RoutingCancelled):
        return "cancelled"
    message = str(error)
    if "no route" in message.lower():
        return "no_route"
    if isinstance(error, ValueError):
        return "invalid_request_or_response"
    return type(error).__name__.lower()


def _load_corpus(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 20:
        raise ValueError("benchmark corpus must contain at least 20 pairs")
    ids = [str(row.get("id")) for row in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark corpus pair IDs must be unique")
    return payload


def _pair_distance(pair: Mapping[str, Any]) -> float:
    start = tuple(float(value) for value in pair["start"])
    end = tuple(float(value) for value in pair["end"])
    return _haversine_km(start, end)


def _evaluations_match(
    strict_evaluation: Mapping[str, Any],
    direct_evaluation: Mapping[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for route_kind in ("scenic", "baseline"):
        strict_route = strict_evaluation.get("routes", {}).get(route_kind)
        direct_route = direct_evaluation.get("routes", {}).get(route_kind)
        checks[f"{route_kind}_present"] = isinstance(
            strict_route, Mapping
        ) and isinstance(direct_route, Mapping)
        if not checks[f"{route_kind}_present"]:
            continue
        checks[f"{route_kind}_traversal_ids"] = (
            strict_route.get("traversal_ids") == direct_route.get("traversal_ids")
        )
        checks[f"{route_kind}_canonical_edge_ids"] = (
            strict_route.get("canonical_edge_ids")
            == direct_route.get("canonical_edge_ids")
        )
        for metric in ("distance_km", "duration_minutes", "raw_scenic_score"):
            checks[f"{route_kind}_{metric}"] = _close_enough(
                _as_float(strict_route.get(metric)),
                _as_float(direct_route.get(metric)),
            )
        checks[f"{route_kind}_exactness"] = (
            strict_route.get("exactness_status")
            == direct_route.get("exactness_status")
        )
        checks[f"{route_kind}_reason"] = (
            strict_route.get("no_better_route_reason")
            == direct_route.get("no_better_route_reason")
        )
        checks[f"{route_kind}_same_route"] = (
            strict_route.get("same_route") == direct_route.get("same_route")
        )
        approximate_pair = (
            str(strict_route.get("exactness_status") or "").startswith(
                "approximate"
            )
            and str(direct_route.get("exactness_status") or "").startswith(
                "approximate"
            )
        )
        for metric in (
            "objective_value_declared",
            "optimality_gap",
            "certified_upper_bound",
            "scenic_score_delta_absolute",
            "scenic_score_delta_relative",
        ):
            strict_value = _as_float(strict_route.get(metric))
            direct_value = _as_float(direct_route.get(metric))
            checks[f"{route_kind}_{metric}"] = (
                approximate_pair
                and metric in {"optimality_gap", "certified_upper_bound"}
            ) or (
                (strict_value is None and direct_value is None)
                or _close_enough(strict_value, direct_value)
            )
    strict_scenic = strict_evaluation.get("routes", {}).get("scenic", {})
    direct_scenic = direct_evaluation.get("routes", {}).get("scenic", {})
    checks["same_route"] = (
        strict_scenic.get("same_route") == direct_scenic.get("same_route")
    )
    checks["highway_count"] = (
        strict_evaluation.get("highway_count")
        == direct_evaluation.get("highway_count")
    )
    checks["normalized_scenic_score"] = _close_enough(
        _as_float(strict_evaluation.get("normalized_scenic_score")),
        _as_float(direct_evaluation.get("normalized_scenic_score")),
    )
    for metric in ("scenic_score_delta_absolute", "scenic_score_delta_relative"):
        strict_value = _as_float(strict_scenic.get(metric))
        direct_value = _as_float(direct_scenic.get(metric))
        checks[metric] = (
            strict_value is None and direct_value is None
        ) or _close_enough(strict_value, direct_value)
    checks["objective"] = _close_enough(
        _as_float(strict_evaluation.get("objective", {}).get("recomputed")),
        _as_float(direct_evaluation.get("objective", {}).get("recomputed")),
    )
    checks["duration_ratio"] = _close_enough(
        _as_float(strict_evaluation.get("duration_ratio")),
        _as_float(direct_evaluation.get("duration_ratio")),
    )
    return checks


def _classify_pairs(
    corpus: Mapping[str, Any], rows: list[Mapping[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    discovered: dict[str, list[str]] = defaultdict(list)
    for pair in corpus["pairs"]:
        pair_id = str(pair["id"])
        pair_rows = by_pair[pair_id]
        successful = [row for row in pair_rows if row.get("evaluation", {}).get("status") == "ok"]
        if not successful:
            continue
        distance = _pair_distance(pair)
        if distance <= 10.0:
            discovered["short urban"].append(pair_id)
        elif distance <= 100.0:
            discovered["medium regional"].append(pair_id)
        else:
            discovered["long regional"].append(pair_id)
        if pair_id == "checked_in_default_reproduction" and any(
            row["q"] == 0.8
            and row["kappa"] == 1.8
            and row["avoid_highways"] is False
            for row in successful
        ):
            discovered["checked-in default reproduction"].append(pair_id)
        false_rows = [row for row in successful if not row["avoid_highways"]]
        true_rows = [row for row in successful if row["avoid_highways"]]
        if false_rows and true_rows:
            false_ids = {
                row["evaluation"]["routes"]["scenic"]["segment_identity_sha256"]
                for row in false_rows
            }
            true_ids = {
                row["evaluation"]["routes"]["scenic"]["segment_identity_sha256"]
                for row in true_rows
            }
            if false_ids != true_ids or any(row["evaluation"]["highway_count"] > 0 for row in false_rows):
                discovered["highway-sensitive"].append(pair_id)
        if any(
            row["avoid_highways"] is False and row.get("evaluation", {}).get("status") == "ok"
            for row in pair_rows
        ) and any(row["avoid_highways"] and row.get("reason") == "no_route" for row in pair_rows):
            discovered["no-highway-free"].append(pair_id)
        scenic_improvement = any(
            row["q"] > 0.0
            and row["evaluation"].get("status") == "ok"
            and (row["evaluation"].get("optimality_gap") or 0.0) <= 1e-12
            and (row["evaluation"].get("uplift_absolute") or 0.0) > 1e-6
            and row["evaluation"]["routes"]["scenic"]["segment_identity_sha256"]
            != (row["evaluation"]["routes"]["baseline"] or {}).get("segment_identity_sha256")
            for row in successful
        )
        if scenic_improvement:
            discovered["obvious scenic candidate"].append(pair_id)
        q1 = [
            row
            for row in successful
            if row["q"] > 0.0 and abs(row["kappa"] - 1.0) < 1e-9
        ]
        if any(abs((row["evaluation"].get("duration_ratio") or 1.0) - 1.0) <= 1e-8 for row in q1):
            discovered["detour1.0 binding"].append(pair_id)
        unlock = False
        for q in Q_VALUES[1:]:
            at_1 = next((row for row in successful if row["q"] == q and row["kappa"] == 1.0 and not row["avoid_highways"]), None)
            at_18 = next((row for row in successful if row["q"] == q and row["kappa"] == 1.8 and not row["avoid_highways"]), None)
            if at_1 and at_18 and at_1["evaluation"].get("uplift_absolute", 0.0) <= 1e-6 < at_18["evaluation"].get("uplift_absolute", 0.0):
                if (
                    (at_1["evaluation"].get("optimality_gap") or 0.0) > 1e-12
                    or (at_18["evaluation"].get("optimality_gap") or 0.0) > 1e-12
                ):
                    continue
                unlock = True
                break
        if unlock:
            discovered["detour1.8 unlock candidate"].append(pair_id)
    discovered.setdefault("historical screenshot coordinates", [])
    missing = {
        category: (
            "No completed matrix response satisfied the empirical predicate; positive-gap/timeout rows are not treated as proof."
            if category not in discovered or not discovered[category]
            else ""
        )
        for category in REQUIRED_CATEGORIES
        if not discovered.get(category)
    }
    return {key: sorted(set(value)) for key, value in discovered.items()}, missing


def _invariant_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    completed = [row for row in rows if row.get("evaluation", {}).get("status") == "ok"]
    for row in completed:
        for name, passed in row["evaluation"].get("invariants", {}).items():
            counts[name]["pass" if passed else "fail"] += 1
    # q monotonicity uses the chosen route's scenic utility; kappa monotonicity
    # uses declared objective. A positive gap downgrades a decrease to
    # not_proven_positive_gap rather than claiming a global optimum.
    groups_q: dict[tuple[str, float, bool], list[Mapping[str, Any]]] = defaultdict(list)
    groups_k: dict[tuple[str, float, bool], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        groups_q[(str(row["pair_id"]), float(row["kappa"]), bool(row["avoid_highways"]))].append(row)
        groups_k[(str(row["pair_id"]), float(row["q"]), bool(row["avoid_highways"]))].append(row)
    for group in groups_q.values():
        ordered = sorted(group, key=lambda row: row["q"])
        for previous, current in zip(ordered, ordered[1:]):
            prev_value = previous["evaluation"].get("normalized_scenic_score")
            current_value = current["evaluation"].get("normalized_scenic_score")
            if (
                (previous["evaluation"].get("optimality_gap") or 0.0) > 0.0
                or (current["evaluation"].get("optimality_gap") or 0.0) > 0.0
            ):
                counts["monotonic_best_utility_q"]["not_proven_positive_gap"] += 1
                continue
            if prev_value is None or current_value is None:
                counts["monotonic_best_utility_q"]["skip"] += 1
            elif current_value + 1e-8 >= prev_value:
                counts["monotonic_best_utility_q"]["pass"] += 1
            elif (current["evaluation"].get("optimality_gap") or 0.0) > 0.0:
                counts["monotonic_best_utility_q"]["not_proven_positive_gap"] += 1
            else:
                counts["monotonic_best_utility_q"]["fail"] += 1
    for group in groups_k.values():
        ordered = sorted(group, key=lambda row: row["kappa"])
        for previous, current in zip(ordered, ordered[1:]):
            prev_value = previous["evaluation"]["objective"].get("recomputed")
            current_value = current["evaluation"]["objective"].get("recomputed")
            if (
                (previous["evaluation"].get("optimality_gap") or 0.0) > 0.0
                or (current["evaluation"].get("optimality_gap") or 0.0) > 0.0
            ):
                counts["monotonic_best_utility_kappa"]["not_proven_positive_gap"] += 1
                continue
            if current_value + 1e-8 >= prev_value:
                counts["monotonic_best_utility_kappa"]["pass"] += 1
            elif (current["evaluation"].get("optimality_gap") or 0.0) > 0.0:
                counts["monotonic_best_utility_kappa"]["not_proven_positive_gap"] += 1
            else:
                counts["monotonic_best_utility_kappa"]["fail"] += 1
    # One report/graph identity and warm cache hits are the cache boundary
    # assertions for this single-report benchmark.
    signature_pairs = {
        (
            row.get("evaluation", {}).get("score_mapping", {}).get(
                "graph_signature"
            ),
            row.get("evaluation", {}).get("score_mapping", {}).get(
                "report_signature"
            ),
        )
        for row in completed
    }
    missing_signature = any(
        graph_signature is None or report_signature is None
        for graph_signature, report_signature in signature_pairs
    )
    counts["cache_boundaries"][
        "pass"
        if completed
        and not missing_signature
        and len(signature_pairs) <= 1
        else "fail"
    ] += 1
    counts["cache_isolation"]["not_proven_single_variant"] += 1
    strict_completed = [
        row for row in completed if row.get("execution_mode") == "strict_service"
    ]
    if len(strict_completed) > 1:
        hits = all(
            row.get("evaluation", {}).get("diagnostics", {}).get("graph_cache_hit")
            is True
            and row.get("evaluation", {}).get("diagnostics", {}).get(
                "tile_score_cache_hit"
            )
            is True
            and row.get("evaluation", {}).get("diagnostics", {}).get(
                "scored_graph_cache_hit"
            )
            is True
            for row in strict_completed[1:]
        )
        counts["warm_cache_hits"]["pass" if hits else "fail"] += 1
    else:
        counts["warm_cache_hits"]["skip"] += 1
    return {name: dict(values) for name, values in sorted(counts.items())}


def _row_within_case_deadline(
    row: Mapping[str, Any],
    configured_deadline_seconds: float,
) -> bool:
    deadline = _as_float(
        row.get("case_timeout_seconds"), configured_deadline_seconds
    )
    return (
        deadline is not None
        and deadline > 0.0
        and float(row["wall_ms"]) < deadline * 1000.0
        and row.get("reason") != "timeout"
    )


ACTIVATION_CASE_TIMEOUT_SECONDS = 1_800.0
REQUIRED_ACTIVATION_CASE_IDS = frozenset(
    {"full_bbox_rutland_lisbon|q=0.8|kappa=1.8|avoid=false"}
)
EXTRA_Q08_PAIR_IDS = frozenset(
    {"checked_in_default_reproduction", "full_bbox_rutland_lisbon"}
)
STRICT_SERVICE_CASE_IDS = frozenset(
    {
        "short_burlington_01|q=0|kappa=1|avoid=false",
        "short_burlington_01|q=0.9|kappa=1.8|avoid=false",
        "short_burlington_01|q=0|kappa=1|avoid=true",
        "medium_bangor_bar_harbor|q=0|kappa=1|avoid=false",
        "checked_in_default_reproduction|q=0|kappa=1|avoid=false",
        "checked_in_default_reproduction|q=0.9|kappa=1.8|avoid=false",
        "checked_in_default_reproduction|q=0.8|kappa=1.8|avoid=false",
    }
) | REQUIRED_ACTIVATION_CASE_IDS


def _case_specs(corpus: Mapping[str, Any]) -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    for pair in corpus["pairs"]:
        pair_id = str(pair["id"])
        start = tuple(float(value) for value in pair["start"])
        end = tuple(float(value) for value in pair["end"])
        for avoid in (False, True):
            for kappa in KAPPA_VALUES:
                for q in Q_VALUES:
                    specs.append(CaseSpec(pair_id, start, end, q, kappa, avoid))
        if pair_id in EXTRA_Q08_PAIR_IDS:
            specs.append(CaseSpec(pair_id, start, end, 0.8, 1.8, False))
    return specs


def _case_deadline_seconds(case_id: str, configured_seconds: float) -> float:
    if configured_seconds <= 0.0:
        return configured_seconds
    if case_id in REQUIRED_ACTIVATION_CASE_IDS:
        return max(configured_seconds, ACTIVATION_CASE_TIMEOUT_SECONDS)
    return configured_seconds


def _direct_planner_response(
    planner: ScenicRoutePlanner,
    request: RouteRequest,
    *,
    score_mapping: Mapping[str, Any],
    deadline: RoutingDeadline | None = None,
) -> dict[str, Any]:
    """Adapt one direct planner call to the strict service response shape.

    Matrix cases use the already loaded/scored graph and one long-lived
    planner.  A small deterministic subset still calls ``plan_routes`` below
    so strict API serialization and service cache boundaries remain exercised.
    The same ``RoutingDeadline`` is forwarded unchanged to every routing,
    scoring, and serialization stage.
    """
    if deadline is None:
        deadline = RoutingDeadline()
    deadline.check()
    scenic_route = planner.find_scenic_route(
        start=request.start,
        end=request.end,
        scenic_weight=request.scenic_weight,
        avoid_highways=request.avoid_highways,
        max_detour_factor=request.max_detour_factor,
        scenic_priority=True,
        deadline=deadline,
    )
    deadline.check()
    baseline_route = (
        planner.find_fastest_route(
            start=request.start,
            end=request.end,
            avoid_highways=request.avoid_highways,
            deadline=deadline,
        )
        if request.include_baseline
        else None
    )
    deadline.check()
    objective = route_service._objective_components(
        request, scenic_route, baseline_route, deadline=deadline
    )
    deadline.check()
    scenic_feature = route_service.route_to_feature(
        scenic_route,
        "scenic",
        objective=objective,
        score_provenance=score_mapping,
        requested_start=request.start,
        requested_end=request.end,
        deadline=deadline,
    )
    deadline.check()
    features = [scenic_feature]
    routes = [{"route_kind": "scenic", "metrics": scenic_feature["properties"]}]
    if baseline_route is not None:
        baseline_normalized = _normalise_score(
            baseline_route.average_scenic_score
        )
        baseline_objective = {
            "duration_utility": 1.0,
            "scenic_utility": baseline_normalized,
            "objective_value": baseline_normalized,
            "optimization_mode": "fastest_duration_baseline",
            "raw_scenic_score": float(baseline_route.average_scenic_score),
            "normalized_scenic_score": baseline_normalized,
            "requested_scenic_weight": float(request.scenic_weight),
            "applied_scenic_weight": float(request.scenic_weight),
            "requested_max_detour_factor": float(request.max_detour_factor),
            "applied_max_detour_factor": float(request.max_detour_factor),
            "actual_duration_ratio": 1.0,
            "scenic_score_delta_absolute": None,
            "scenic_score_delta_relative": None,
            "same_route": None,
            "no_better_route_reason": None,
        }
        deadline.check()
        baseline_feature = route_service.route_to_feature(
            baseline_route,
            "baseline",
            objective=baseline_objective,
            score_provenance=score_mapping,
            requested_start=request.start,
            requested_end=request.end,
            deadline=deadline,
        )
        deadline.check()
        features.append(baseline_feature)
        routes.append(
            {"route_kind": "baseline", "metrics": baseline_feature["properties"]}
        )
    return {
        "request": request.to_dict(),
        "diagnostics": {
            "graph_nodes": len(planner.graph.nodes) if planner.graph is not None else 0,
            "graph_edges": len(planner.graph.edges) if planner.graph is not None else 0,
            "graph_cache_hit": None,
            "tile_score_cache_hit": None,
            "scored_graph_cache_hit": None,
            "cache_measurement": "inapplicable_direct_planner",
            "requested_scenic_weight": float(request.scenic_weight),
            "applied_scenic_weight": float(request.scenic_weight),
            "requested_max_detour_factor": float(request.max_detour_factor),
            "applied_max_detour_factor": float(request.max_detour_factor),
            "avoid_highways_applied": bool(request.avoid_highways),
            "score_mapping_coverage": score_mapping.get("matched_ratio", 0.0),
            "score_report_identity": score_mapping.get("score_run"),
            "score_report_signature": score_mapping.get("report_signature"),
            "graph_load_elapsed_ms": 0.0,
            "tile_score_load_elapsed_ms": 0.0,
            "score_application_elapsed_ms": 0.0,
            "planning_elapsed_ms": 0.0,
            "exactness_status": scenic_feature["properties"].get(
                "exactness_status"
            ),
            "optimality_gap": scenic_feature["properties"].get(
                "optimality_gap"
            ),
            "certified_upper_bound": scenic_feature["properties"].get(
                "certified_upper_bound"
            ),
            "normalized_scenic_score": scenic_feature["properties"].get(
                "normalized_scenic_score"
            ),
            "scenic_score_delta_absolute": scenic_feature["properties"].get(
                "scenic_score_delta_absolute"
            ),
            "scenic_score_delta_relative": scenic_feature["properties"].get(
                "scenic_score_delta_relative"
            ),
            "same_route": scenic_feature["properties"].get("same_route"),
            "no_better_route_reason": scenic_feature["properties"].get(
                "no_better_route_reason"
            ),
        },
        "score_mapping": dict(score_mapping),
        "routes": routes,
        "geojson": {"type": "FeatureCollection", "features": features},
    }




def _case_id(spec: CaseSpec) -> str:
    return (
        f"{spec.pair_id}|q={spec.q:g}|kappa={spec.kappa:g}|"
        f"avoid={str(spec.avoid_highways).lower()}"
    )


def _validate_execution_options(workers: int, group_size: int) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 0
    ):
        raise ValueError("group_size must be a positive integer or zero")



def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _group_size_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed

def _repair_checkpoint_tail(jsonl_path: Path) -> None:
    if not jsonl_path.exists():
        return
    with jsonl_path.open("rb+") as stream:
        valid_offset = 0
        while True:
            line = stream.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                stream.seek(valid_offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                return
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stream.seek(valid_offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                return
            if not isinstance(row, Mapping) or not isinstance(row.get("case_id"), str):
                stream.seek(valid_offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                return
            valid_offset = stream.tell()


def _read_persisted_rows(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    persisted: dict[str, dict[str, Any]] = {}
    if not jsonl_path.exists():
        return persisted
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = row.get("case_id") if isinstance(row, Mapping) else None
            if isinstance(case_id, str) and case_id not in persisted:
                persisted[case_id] = dict(row)
    return persisted


def _atomic_write_text(path: Path, text: str) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _prepare_benchmark_context(
    graph_path: Path, report_path: Path
) -> tuple[
    dict[str, Any],
    float,
    ScenicRoutePlanner,
    dict[str, Any],
    dict[str, Any],
    Any,
]:
    clear_route_caches()
    preload_started = perf_counter()
    preload = preload_route_assets(
        graph_path,
        report_path,
        exclusive_scoring=True,
    )
    preload_wall_ms = (perf_counter() - preload_started) * 1000.0
    scored_graph = next(iter(route_service._SCORED_GRAPH_CACHE.values()), None)
    if scored_graph is None:
        raise RuntimeError("preload did not publish a scored graph variant")
    matrix_planner = ScenicRoutePlanner(graph=scored_graph)
    planner_preload = matrix_planner.prewarm_routing_cache()
    edge_index = {str(edge_id): edge for edge_id, edge in scored_graph.edges.items()}
    node_index = {str(node_id): node for node_id, node in scored_graph.nodes.items()}
    preload["process_peak_rss_bytes"] = _process_peak_rss_bytes()
    return (
        preload,
        preload_wall_ms,
        matrix_planner,
        edge_index,
        node_index,
        planner_preload,
    )


def _execute_case(
    *,
    spec: CaseSpec,
    index: int,
    graph_path: Path,
    report_path: Path,
    case_timeout_seconds: float,
    strict_service_full: bool,
    context: tuple[dict[str, Any], ScenicRoutePlanner, dict[str, Any], dict[str, Any]],
    route_error_cache: dict[tuple[str, bool], tuple[str, str]],
) -> dict[str, Any]:
    preload, matrix_planner, edge_index, node_index = context
    request = RouteRequest(
        graph_geojson=str(graph_path),
        start=spec.start,
        end=spec.end,
        scenic_weight=spec.q,
        max_detour_factor=spec.kappa,
        avoid_highways=spec.avoid_highways,
        include_baseline=True,
        tile_scores_json=str(report_path),
    )
    started = perf_counter()
    case_id = _case_id(spec)
    case_deadline_seconds = _case_deadline_seconds(
        case_id, case_timeout_seconds
    )
    strict_service = strict_service_full or case_id in STRICT_SERVICE_CASE_IDS
    row: dict[str, Any] = {
        "case_index": index,
        "case_id": case_id,
        "execution_mode": "strict_service" if strict_service else "direct_planner",
        "pair_id": spec.pair_id,
        "start": list(spec.start),
        "end": list(spec.end),
        "geodesic_distance_km": _haversine_km(spec.start, spec.end),
        "q": spec.q,
        "kappa": spec.kappa,
        "avoid_highways": spec.avoid_highways,
        "case_timeout_seconds": case_deadline_seconds,
    }
    route_error_key = (spec.pair_id, spec.avoid_highways)
    cached_error = route_error_cache.get(route_error_key)
    if cached_error is not None and not strict_service:
        row["evaluation"] = {"status": "error"}
        row["reason"], row["error"] = cached_error
        row["route_error_cache_hit"] = True
    else:
        try:
            with _case_deadline(case_deadline_seconds) as deadline:
                deadline.check()
                if strict_service:
                    response = plan_routes(request, deadline=deadline)
                else:
                    planning_started = perf_counter()
                    response = _direct_planner_response(
                        matrix_planner,
                        request,
                        score_mapping=preload["score_mapping"],
                        deadline=deadline,
                    )
                    response["diagnostics"]["planning_elapsed_ms"] = (
                        perf_counter() - planning_started
                    ) * 1000.0
                deadline.check()
                row["evaluation"] = evaluate_service_response(
                    response,
                    q=spec.q,
                    kappa=spec.kappa,
                    avoid_highways=spec.avoid_highways,
                    edge_index=edge_index,
                    node_index=node_index,
                    requested_start=spec.start,
                    requested_end=spec.end,
                    scenic_priority=True,
                )
                deadline.check()
                row["reason"] = (
                    None
                    if row["evaluation"].get("status") == "ok"
                    else "invalid:"
                    + ",".join(row["evaluation"].get("failed_invariants", []))
                )
                row["route_error_cache_hit"] = False
                if (
                    spec.pair_id == "checked_in_default_reproduction"
                    and spec.q == 0.8
                    and spec.kappa == 1.8
                    and not spec.avoid_highways
                    and row["evaluation"].get("status") == "ok"
                ):
                    deadline.check()
                    direct_response = _direct_planner_response(
                        matrix_planner,
                        request,
                        score_mapping=preload["score_mapping"],
                        deadline=deadline,
                    )
                    deadline.check()
                    direct_evaluation = evaluate_service_response(
                        direct_response,
                        q=spec.q,
                        kappa=spec.kappa,
                        avoid_highways=spec.avoid_highways,
                        edge_index=edge_index,
                        node_index=node_index,
                        requested_start=spec.start,
                        requested_end=spec.end,
                        scenic_priority=True,
                    )
                    deadline.check()
                    parity = _evaluations_match(row["evaluation"], direct_evaluation)
                    row["ui_reproduction_parity"] = {
                        "strict_service": row["evaluation"],
                        "direct_planner": direct_evaluation,
                        "checks": parity,
                        "pass": bool(parity) and all(parity.values()),
                    }
                    if not row["ui_reproduction_parity"]["pass"]:
                        row["evaluation"]["status"] = "invalid"
                        row["evaluation"].setdefault("failed_invariants", []).append(
                            "strict_direct_parity"
                        )
                        row["reason"] = "invalid:strict_direct_parity"
        except Exception as error:
            row["evaluation"] = {"status": "error"}
            row["reason"] = _error_reason(error)
            row["error"] = str(error)
            row["route_error_cache_hit"] = False
            if row["reason"] == "no_route":
                route_error_cache[route_error_key] = (
                    str(row["reason"]),
                    str(row["error"]),
                )
    row["wall_ms"] = (perf_counter() - started) * 1000.0
    evaluation = row["evaluation"]
    if evaluation.get("status") == "ok":
        diagnostics = evaluation.get("diagnostics", {})
        row["phase_ms"] = {
            "graph_load": diagnostics.get("graph_load_elapsed_ms"),
            "tile_score_load": diagnostics.get("tile_score_load_elapsed_ms"),
            "score_application": diagnostics.get("score_application_elapsed_ms"),
            "planning": diagnostics.get("planning_elapsed_ms"),
        }
        row["cache"] = {
            "graph_cache_hit": diagnostics.get("graph_cache_hit"),
            "tile_score_cache_hit": diagnostics.get("tile_score_cache_hit"),
            "scored_graph_cache_hit": diagnostics.get("scored_graph_cache_hit"),
            "report_signature": evaluation.get("score_mapping", {}).get("report_signature"),
            "graph_signature": evaluation.get("score_mapping", {}).get("graph_signature"),
        }
    else:
        row["phase_ms"] = {}
        row["cache"] = {"route_error_cache_hit": bool(row.get("route_error_cache_hit"))}
    return row



_WORKER_CONTEXT: tuple[
    dict[str, Any], ScenicRoutePlanner, dict[str, Any], dict[str, Any]
] | None = None
_WORKER_SUPERVISOR: _PersistentPlanningChild | None = None
_WORKER_ERRORS: dict[tuple[str, bool], tuple[str, str]] = {}
_WORKER_PRELOAD: dict[str, Any] = {}
_WORKER_PRELOAD_WALL_MS = 0.0
_WORKER_PLANNER_PRELOAD: dict[str, Any] = {}


def _init_case_worker(graph_path: Path, report_path: Path) -> None:
    global _WORKER_CONTEXT, _WORKER_SUPERVISOR, _WORKER_ERRORS, _WORKER_PRELOAD, _WORKER_PRELOAD_WALL_MS, _WORKER_PLANNER_PRELOAD
    (
        preload,
        preload_wall_ms,
        planner,
        edges,
        nodes,
        planner_preload,
    ) = _prepare_benchmark_context(graph_path, report_path)
    _WORKER_CONTEXT = (preload, planner, edges, nodes)
    _WORKER_ERRORS = {}
    _WORKER_PRELOAD = preload
    _WORKER_PRELOAD_WALL_MS = preload_wall_ms
    _WORKER_PLANNER_PRELOAD = planner_preload
    if _HAS_POSIX_FORK:
        _WORKER_SUPERVISOR = _PersistentPlanningChild(
            _WORKER_CONTEXT, _WORKER_ERRORS
        )
        atexit.register(_close_worker_supervisor)


def _close_worker_supervisor() -> None:
    global _WORKER_SUPERVISOR
    if _WORKER_SUPERVISOR is not None:
        _WORKER_SUPERVISOR.close()
        _WORKER_SUPERVISOR = None


def _worker_context_snapshot() -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("case worker was not initialized")
    return {
        "preload": _WORKER_PRELOAD,
        "preload_wall_ms": _WORKER_PRELOAD_WALL_MS,
        "planner_matrix_preload": _WORKER_PLANNER_PRELOAD,
    }


def _run_case_worker(
    args: tuple[int, CaseSpec, Path, Path, float, bool]
) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("case worker was not initialized")
    index, spec, graph_path, report_path, timeout, strict_full = args
    if _WORKER_SUPERVISOR is not None:
        started = perf_counter()
        try:
            return _WORKER_SUPERVISOR.run(
                spec,
                index,
                graph_path,
                report_path,
                timeout,
                strict_full,
            )
        except TimeoutError:
            wall_ms = (perf_counter() - started) * 1000.0
            case_id = _case_id(spec)
            case_deadline_seconds = _case_deadline_seconds(case_id, timeout)
            return _timeout_row(
                index, spec, case_deadline_seconds, strict_full, wall_ms
            )
    return _execute_case(
        spec=spec,
        index=index,
        graph_path=graph_path,
        report_path=report_path,
        case_timeout_seconds=timeout,
        strict_service_full=strict_full,
        context=_WORKER_CONTEXT,
        route_error_cache=_WORKER_ERRORS,
    )

def _timeout_row(
    index: int,
    spec: CaseSpec,
    case_deadline_seconds: float,
    strict_service_full: bool,
    wall_ms: float,
) -> dict[str, Any]:
    case_id = _case_id(spec)
    return {
        "case_index": index,
        "case_id": case_id,
        "execution_mode": (
            "strict_service"
            if strict_service_full or case_id in STRICT_SERVICE_CASE_IDS
            else "direct_planner"
        ),
        "pair_id": spec.pair_id,
        "start": list(spec.start),
        "end": list(spec.end),
        "geodesic_distance_km": _haversine_km(spec.start, spec.end),
        "q": spec.q,
        "kappa": spec.kappa,
        "avoid_highways": spec.avoid_highways,
        "case_timeout_seconds": case_deadline_seconds,
        "evaluation": {"status": "error"},
        "reason": "timeout",
        "error": "case exceeded hard supervisor deadline",
        "route_error_cache_hit": False,
        "wall_ms": wall_ms,
        "phase_ms": {},
        "cache": {"route_error_cache_hit": False},
    }


def _worker_failure_row(
    index: int, spec: CaseSpec, error: BaseException, strict_full: bool
) -> dict[str, Any]:
    return {
        "case_index": index,
        "case_id": _case_id(spec),
        "execution_mode": (
            "strict_service"
            if strict_full or _case_id(spec) in STRICT_SERVICE_CASE_IDS
            else "direct_planner"
        ),
        "pair_id": spec.pair_id,
        "start": list(spec.start),
        "end": list(spec.end),
        "geodesic_distance_km": _haversine_km(spec.start, spec.end),
        "q": spec.q,
        "kappa": spec.kappa,
        "avoid_highways": spec.avoid_highways,
        "evaluation": {"status": "error"},
        "reason": "worker_error",
        "error": str(error),
        "route_error_cache_hit": False,
        "wall_ms": 0.0,
        "phase_ms": {},
        "cache": {"route_error_cache_hit": False},
    }

def run_benchmark(
    *,
    corpus_path: Path = DEFAULT_CORPUS,
    graph_path: Path = DEFAULT_GRAPH,
    report_path: Path = DEFAULT_REPORT,
    output_path: Path = DEFAULT_OUTPUT,
    case_timeout_seconds: float = 10.0,
    strict_service_full: bool = False,
    resume: bool = False,
    workers: int = 1,
    group_size: int = 0,
) -> dict[str, Any]:
    _validate_execution_options(workers, group_size)
    corpus = _load_corpus(corpus_path)
    graph_path = Path(graph_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")
    checkpoint_fingerprint = _checkpoint_fingerprint(
        corpus_path=Path(corpus_path),
        corpus=corpus,
        graph_path=graph_path,
        report_path=report_path,
        case_timeout_seconds=case_timeout_seconds,
        strict_service_full=strict_service_full,
        workers=workers,
        group_size=group_size,
    )
    if resume:
        _repair_checkpoint_tail(jsonl_path)
        persisted = _read_persisted_rows(jsonl_path)
        incompatible = [
            case_id
            for case_id, row in persisted.items()
            if row.get("checkpoint_fingerprint") != checkpoint_fingerprint
        ]
        if incompatible:
            raise ValueError(
                "benchmark checkpoint fingerprint mismatch; use a new output path "
                "or rerun without --resume"
            )
    else:
        persisted = {}
    specs = _case_specs(corpus)
    planned_ids = [_case_id(spec) for spec in specs]
    rows: list[dict[str, Any]] = [
        persisted[case_id] for case_id in planned_ids if case_id in persisted
    ]
    if workers == 1:
        (
            preload,
            preload_wall_ms,
            matrix_planner,
            edge_index,
            node_index,
            planner_preload,
        ) = _prepare_benchmark_context(graph_path, report_path)
        context = (preload, matrix_planner, edge_index, node_index)
        route_error_cache: dict[tuple[str, bool], tuple[str, str]] = {}
    else:
        preload = {}
        preload_wall_ms = 0.0
        planner_preload = {"isolated_workers": workers}
        context = None
    if workers == 1:
        pending_cases = (
            (index, spec)
            for index, spec in enumerate(specs)
            if _case_id(spec) not in persisted
        )
        groups: list[list[tuple[int, CaseSpec]]] = []
    else:
        remaining = [
            (index, spec)
            for index, spec in enumerate(specs)
            if _case_id(spec) not in persisted
        ]
        effective_group_size = group_size or workers
        groups = [
            remaining[start : start + effective_group_size]
            for start in range(0, len(remaining), effective_group_size)
        ]
    stream_mode = "a" if resume and jsonl_path.exists() else "w"
    supervisor: _PersistentPlanningChild | None = None
    if workers == 1 and _HAS_POSIX_FORK:
        supervisor = _PersistentPlanningChild(context, route_error_cache)
    try:
        with jsonl_path.open(stream_mode, encoding="utf-8") as stream:
            def persist(row: dict[str, Any]) -> None:
                row = dict(row)
                row["checkpoint_fingerprint"] = checkpoint_fingerprint
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            if workers == 1:
                for index, spec in pending_cases:
                    started = perf_counter()
                    try:
                        if supervisor is not None:
                            row = supervisor.run(
                                spec,
                                index,
                                graph_path,
                                report_path,
                                case_timeout_seconds,
                                strict_service_full,
                            )
                        else:
                            row = _execute_case(
                                spec=spec,
                                index=index,
                                graph_path=graph_path,
                                report_path=report_path,
                                case_timeout_seconds=case_timeout_seconds,
                                strict_service_full=strict_service_full,
                                context=context,
                                route_error_cache=route_error_cache,
                            )
                    except TimeoutError:
                        wall_ms = (perf_counter() - started) * 1000.0
                        case_id = _case_id(spec)
                        case_deadline_seconds = _case_deadline_seconds(
                            case_id, case_timeout_seconds
                        )
                        row = _timeout_row(
                            index,
                            spec,
                            case_deadline_seconds,
                            strict_service_full,
                            wall_ms,
                        )
                    persist(row)
                    if (
                        row.get("reason") == "no_route"
                        and not row.get("route_error_cache_hit")
                    ):
                        route_error_cache[(spec.pair_id, spec.avoid_highways)] = (
                            str(row["reason"]),
                            str(row["error"]),
                        )
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_case_worker,
                    initargs=(graph_path, report_path),
                ) as pool:
                    worker_metadata = pool.submit(_worker_context_snapshot).result()
                    preload = dict(worker_metadata["preload"])
                    preload_wall_ms = float(worker_metadata["preload_wall_ms"])
                    planner_preload = {
                        **dict(worker_metadata["planner_matrix_preload"]),
                        "isolated_workers": workers,
                    }
                    for group in groups:
                        for window_start in range(0, len(group), workers):
                            window = group[window_start : window_start + workers]
                            futures = {
                                pool.submit(
                                    _run_case_worker,
                                    (
                                        index,
                                        spec,
                                        graph_path,
                                        report_path,
                                        case_timeout_seconds,
                                        strict_service_full,
                                    ),
                                ): (index, spec)
                                for index, spec in window
                            }
                            for future in as_completed(futures):
                                index, spec = futures[future]
                                try:
                                    row = future.result()
                                except Exception as error:
                                    row = _worker_failure_row(
                                        index, spec, error, strict_service_full
                                    )
                                persist(row)
    finally:
        if supervisor is not None:
            supervisor.close()
    rows.sort(key=lambda row: planned_ids.index(str(row["case_id"])))
    discovered, missing = _classify_pairs(corpus, rows)
    completed = [row for row in rows if row["evaluation"].get("status") == "ok"]
    wall_values = [float(row["wall_ms"]) for row in rows]
    completed_wall = [float(row["wall_ms"]) for row in completed]
    no_route_wall = [
        float(row["wall_ms"])
        for row in rows
        if row.get("reason") == "no_route" and not row.get("route_error_cache_hit")
    ]
    no_route_cache_wall = [
        float(row["wall_ms"])
        for row in rows
        if row.get("reason") == "no_route" and row.get("route_error_cache_hit")
    ]
    timeout_wall = [
        float(row["wall_ms"]) for row in rows if row.get("reason") == "timeout"
    ]

    def latency_stats(values: list[float]) -> dict[str, float | int | None]:
        ordered = sorted(values)
        if not ordered:
            return {
                "count": 0,
                "median": None,
                "p95": None,
                "max": None,
                "under_10s_count": 0,
                "under_10s_rate": None,
            }
        percentile = lambda fraction: ordered[
            min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)
        ]
        under = sum(1 for value in ordered if value < 10000.0)
        return {
            "count": len(ordered),
            "median": percentile(0.5),
            "p95": percentile(0.95),
            "max": max(ordered),
            "under_10s_count": under,
            "under_10s_rate": under / len(ordered),
        }
    improvements = [float(row["evaluation"].get("uplift_absolute") or 0.0) for row in completed]
    gaps = [row["evaluation"].get("optimality_gap") for row in completed]
    exact = sum(1 for row in completed if row["evaluation"].get("exactness_status") == "exact")
    approx = sum(1 for row in completed if str(row["evaluation"].get("exactness_status", "")).startswith("approximate"))
    no_route_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("reason"):
            no_route_reasons[str(row["reason"])] += 1
    ui_reproduction_parity = next(
        (
            row["ui_reproduction_parity"]
            for row in rows
            if "ui_reproduction_parity" in row
        ),
        None,
    )
    invariant_summary = _invariant_summary(rows)
    if case_timeout_seconds > 0.0:
        latency_pass = all(
            _row_within_case_deadline(row, case_timeout_seconds)
            for row in rows
        )
        invariant_summary["all_case_latency_sla"] = {
            "pass" if latency_pass else "fail": 1
        }
    else:
        invariant_summary["all_case_latency_sla"] = {"skip": 1}
    payload = {
        "schema_version": 1,
        "benchmark": "production_artifact_routing",
        "corpus_path": str(corpus_path),
        "graph_path": str(graph_path),
        "report_path": str(report_path),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "historical_screenshot_coordinates": corpus.get("historical_screenshot_coordinates"),
        "historical_screenshot_blocker": corpus.get("historical_screenshot_blocker"),
        "matrix": {
            "pair_count": len(corpus["pairs"]),
            "q_values": list(Q_VALUES),
            "kappa_values": list(KAPPA_VALUES),
            "avoid_highways_values": [False, True],
            "planned_cases": len(specs),
            "completed_cases": len(completed),
            "failed_cases": len(specs) - len(completed),
            "case_timeout_seconds": case_timeout_seconds,
            "strict_service_full": bool(strict_service_full),
            "all_cases_persisted": (
                len(rows) == len(specs)
                and {str(row.get("case_id")) for row in rows} == set(planned_ids)
            ),
            "execution_mode_counts": {
                "direct_planner": sum(
                    1 for row in rows if row.get("execution_mode") == "direct_planner"
                ),
                "strict_service": sum(
                    1 for row in rows if row.get("execution_mode") == "strict_service"
                ),
            },
            "strict_service_case_ids": sorted(STRICT_SERVICE_CASE_IDS),
            "required_activation_case_ids": sorted(
                REQUIRED_ACTIVATION_CASE_IDS
            ),
            "activation_case_timeout_seconds": (
                ACTIVATION_CASE_TIMEOUT_SECONDS
            ),
        },
        "preload": {
            **preload,
            "preload_wall_ms": preload_wall_ms,
            "planner_matrix_preload": planner_preload,
            "preload_scope": "parent" if workers == 1 else "isolated_worker",
            "preload_worker_count": workers,
        },
        "categories": {
            "required": list(REQUIRED_CATEGORIES),
            "discovered": discovered,
            "not_discoverable": missing,
        },
        "invariants": invariant_summary,
        "improvement": {
            "completed_routes": len(improvements),
            "positive_uplift_count": sum(1 for value in improvements if value > 1e-8),
            "positive_uplift_rate": (
                sum(1 for value in improvements if value > 1e-8) / len(improvements)
                if improvements
                else None
            ),
            "absolute_uplift_max": max(improvements) if improvements else None,
            "absolute_uplift_median": (
                sorted(improvements)[len(improvements) // 2] if improvements else None
            ),
            "relative_uplift_max": max(
                [float(row["evaluation"].get("uplift_relative")) for row in completed if row["evaluation"].get("uplift_relative") is not None],
                default=None,
            ),
            "positive_gap_rows_not_proof": sum(
                1 for gap in gaps if gap is not None and float(gap) > 0.0
            ),
        },
        "latency_ms": {
            "completed_routes": latency_stats(completed_wall),
            "no_route_first_computation": latency_stats(no_route_wall),
            "no_route_cache_hits": latency_stats(no_route_cache_wall),
            "timeouts": latency_stats(timeout_wall),
            "all_cases": latency_stats(wall_values),
        },
        "exactness": {"exact_count": exact, "approximate_count": approx, "completed_count": len(completed)},
        "no_route_reasons": dict(sorted(no_route_reasons.items())),
        "ui_reproduction_parity": ui_reproduction_parity,
        "results_jsonl": str(jsonl_path),
    }
    if payload["matrix"]["all_cases_persisted"]:
        jsonl_text = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        )
        _atomic_write_text(jsonl_path, jsonl_text)
        _atomic_write_text(
            output_path,
            json.dumps({**payload, "results": rows}, indent=2, sort_keys=True),
        )
    return payload






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=10.0,
        help="Per-case deadline; <=0 disables it. Timeouts are persisted, never skipped.",
    )
    parser.add_argument(
        "--strict-service-full",
        action="store_true",
        help="Execute every matrix case through plan_routes instead of the direct planner adapter.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the sibling JSONL checkpoint, skipping persisted case IDs.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Bounded isolated worker processes (default: 1).",
    )
    parser.add_argument(
        "--group-size",
        type=_group_size_arg,
        default=0,
        help="Cases per checkpoint window; zero uses one worker-width window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        corpus_path=args.corpus,
        graph_path=args.graph,
        report_path=args.report,
        output_path=args.output,
        case_timeout_seconds=args.case_timeout_seconds,
        strict_service_full=args.strict_service_full,
        resume=args.resume,
        workers=args.workers,
        group_size=args.group_size,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
