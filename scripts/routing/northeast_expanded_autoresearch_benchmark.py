#!/usr/bin/env python3
"""Benchmark exact uncached Northeast Expanded Burlington-to-Pittsburgh routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.route_planner.cancellation import RoutingDeadline  # noqa: E402
from src.route_planner.service import (  # noqa: E402
    RouteRequest,
    clear_route_caches,
    clear_route_response_cache,
    plan_routes,
    preload_route_assets,
)

GRAPH_PATH = Path(
    "data/processed/road_graphs/northeast_expanded_scored_tiles_v1/road_graph.compact.json"
)
REPORT_PATH = Path(
    "data/processed/heuristic_runs/"
    "prompt_two_candidate_exp02_expanded_20260810/report/report.json"
)
OUTPUT_DIR = Path("data/processed/routing_benchmarks/northeast_expanded_autoresearch")
ORACLE_PATH = OUTPUT_DIR / "baseline_oracle.json"
LATEST_PATH = OUTPUT_DIR / "latest_result.json"
DEADLINE_SECONDS = 120.0
WORKER_TIMEOUT_SECONDS = DEADLINE_SECONDS + 60.0
MAX_RSS_BYTES = 24 * 1024**3
EXPECTED_GRAPH_SHA256 = (
    "26c5a61392a83056729848f3f12cf898e2a1f5a2e5cb71ecd909f588ff8b4195"
)
EXPECTED_REPORT_SHA256 = (
    "eb656ca4abf5e1cc1b9b53849ddf3f94a3e3d323b81cf0609a89a999dd0b51ff"
)
EXPECTED_SEMANTIC_FINGERPRINT_SHA256 = (
    "226c52550ba6c0a91ad6ca54969422e2a946a1b3cb7189b7715542388896ef28"
)
_PRELOAD_RESULT_PREFIX = "PRELOAD_RESULT "
_VOLATILE_DIAGNOSTIC_KEYS = {
    "elapsed_ms",
    "graph_cache_hit",
    "graph_load_elapsed_ms",
    "planning_elapsed_ms",
    "route_response_cache_hit",
    "score_application_elapsed_ms",
    "scored_graph_cache_hit",
    "tile_score_cache_hit",
    "tile_score_load_elapsed_ms",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _semantic_response(response: dict[str, Any]) -> dict[str, Any]:
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in value.items()
                if key not in _VOLATILE_DIAGNOSTIC_KEYS
                and not key.endswith("_elapsed_ms")
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return normalize(response)


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _resource_counters() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "block_input_operations": int(usage.ru_inblock),
        "block_output_operations": int(usage.ru_oublock),
    }


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


def _command_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _host_metadata() -> dict[str, Any]:
    native_library = PROJECT_ROOT / "src/route_planner/_compact_search_c.so"
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "commit": _command_output("git", "rev-parse", "HEAD"),
        "branch": _command_output("git", "branch", "--show-current"),
        "worktree": str(PROJECT_ROOT),
        "cpu": _command_output("sysctl", "-n", "machdep.cpu.brand_string"),
        "hardware_model": _command_output("sysctl", "-n", "hw.model"),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": _command_output("sysctl", "-n", "hw.memsize"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "load_average": list(os.getloadavg()),
        "free_disk_bytes": disk.free,
        "compiler": _command_output("clang", "--version"),
        "native_compile_flags": ["-O3", "-shared", "-fPIC"],
        "native_library": {
            "path": str(native_library),
            "sha256": _sha256_file(native_library) if native_library.exists() else None,
            "size_bytes": native_library.stat().st_size
            if native_library.exists()
            else None,
        },
    }


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _request() -> RouteRequest:
    return RouteRequest(
        graph_geojson=str(GRAPH_PATH),
        tile_scores_json=str(REPORT_PATH),
        start=(44.475884, -73.214003),
        end=(40.44062, -79.99589),
        scenic_weight=0.8,
        max_detour_factor=1.8,
        avoid_highways=False,
        include_baseline=True,
        max_snap_distance_km=1.0,
    )


def _artifact_identity() -> dict[str, dict[str, Any]]:
    expected = {
        "graph": (GRAPH_PATH, EXPECTED_GRAPH_SHA256),
        "report": (REPORT_PATH, EXPECTED_REPORT_SHA256),
    }
    identity: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"artifact SHA-256 mismatch for {path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        identity[name] = {
            "path": str(path),
            "sha256": observed_sha256,
            "size_bytes": path.stat().st_size,
        }
    return identity


def _load_pinned_oracle() -> tuple[dict[str, Any], dict[str, Any]]:
    if not ORACLE_PATH.is_file():
        raise FileNotFoundError(
            f"pinned semantic oracle is required and must not be self-seeded: {ORACLE_PATH}"
        )
    oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    fingerprint = oracle.get("semantic_fingerprint_sha256")
    if fingerprint != EXPECTED_SEMANTIC_FINGERPRINT_SHA256:
        raise RuntimeError(
            "oracle fingerprint does not match the source-pinned fingerprint: "
            f"expected {EXPECTED_SEMANTIC_FINGERPRINT_SHA256}, observed {fingerprint}"
        )
    semantic = oracle.get("semantic_response")
    complete = oracle.get("complete_response")
    if not isinstance(semantic, dict) or not isinstance(complete, dict):
        raise RuntimeError(
            "pinned oracle must contain semantic_response and complete_response"
        )
    complete_semantic = _semantic_response(complete)
    if semantic != complete_semantic:
        raise RuntimeError("pinned oracle semantic and complete responses diverge")
    observed_fingerprint = _json_digest(semantic)
    if observed_fingerprint != EXPECTED_SEMANTIC_FINGERPRINT_SHA256:
        raise RuntimeError(
            "pinned oracle payload hash mismatch: "
            f"expected {EXPECTED_SEMANTIC_FINGERPRINT_SHA256}, "
            f"observed {observed_fingerprint}"
        )
    return oracle, semantic


def _timed_request(request: RouteRequest) -> tuple[dict[str, Any], float, float]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    response = plan_routes(
        request,
        deadline=RoutingDeadline.after(DEADLINE_SECONDS),
    )
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    return response, wall_seconds, cpu_seconds


def _measure_preload() -> dict[str, Any]:
    before = _resource_counters()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    diagnostics = preload_route_assets(
        GRAPH_PATH,
        REPORT_PATH,
        exclusive_scoring=True,
        deadline=RoutingDeadline.after(DEADLINE_SECONDS),
    )
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    after = _resource_counters()
    for key in ("graph_cache_hit", "tile_score_cache_hit", "scored_graph_cache_hit"):
        if diagnostics.get(key) is not False:
            raise RuntimeError(
                f"fresh preload unexpectedly reported {key}={diagnostics.get(key)!r}"
            )
    return {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "resource_delta": _counter_delta(before, after),
        "diagnostics": diagnostics,
    }


def _require_timing(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} is not numeric: {value!r}")
    timing = float(value)
    if not math.isfinite(timing) or timing <= 0.0:
        raise RuntimeError(f"{field} must be finite and positive: {timing!r}")
    return timing


def _validate_rss(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{field} is not a positive integer: {value!r}")
    if value > MAX_RSS_BYTES:
        raise MemoryError(f"{field} {value} exceeds {MAX_RSS_BYTES}-byte hard stop")
    return value


def _run_preload_worker() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--preload-worker"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh preload worker failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    payload_lines = [
        line.removeprefix(_PRELOAD_RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(_PRELOAD_RESULT_PREFIX)
    ]
    if len(payload_lines) != 1:
        raise RuntimeError(
            "fresh preload worker emitted an invalid result count: "
            f"{len(payload_lines)}; stdout={completed.stdout!r}"
        )
    record = json.loads(payload_lines[0])
    if not isinstance(record, dict):
        raise RuntimeError("fresh preload worker result is not an object")
    _require_timing(record.get("wall_seconds"), field="preload.wall_seconds")
    _require_timing(record.get("cpu_seconds"), field="preload.cpu_seconds")
    _validate_rss(record.get("peak_rss_bytes"), field="preload.peak_rss_bytes")
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("fresh preload worker diagnostics are missing")
    for key in ("graph_cache_hit", "tile_score_cache_hit", "scored_graph_cache_hit"):
        if diagnostics.get(key) is not False:
            raise RuntimeError(
                f"fresh preload worker unexpectedly reported {key}={diagnostics.get(key)!r}"
            )
    return record


def _assert_response_cache_state(
    response: dict[str, Any],
    *,
    expected_hit: bool,
    context: str,
) -> None:
    diagnostics = response.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError(f"{context} response diagnostics are missing")
    observed_hit = diagnostics.get("route_response_cache_hit")
    if observed_hit is not expected_hit:
        raise RuntimeError(
            f"{context} route_response_cache_hit must be {expected_hit}, "
            f"observed {observed_hit!r}"
        )
    if not expected_hit:
        for key in (
            "graph_cache_hit",
            "tile_score_cache_hit",
            "scored_graph_cache_hit",
        ):
            if diagnostics.get(key) is not True:
                raise RuntimeError(
                    f"{context} must reuse preloaded assets; "
                    f"observed {key}={diagnostics.get(key)!r}"
                )


def _validate_response_semantics(
    response: dict[str, Any],
    *,
    pinned_semantic: dict[str, Any],
    context: str,
) -> str:
    semantic = _semantic_response(response)
    fingerprint = _json_digest(semantic)
    if fingerprint != EXPECTED_SEMANTIC_FINGERPRINT_SHA256:
        raise RuntimeError(
            f"{context} route semantics differ from the pinned baseline: "
            f"expected {EXPECTED_SEMANTIC_FINGERPRINT_SHA256}, observed {fingerprint}"
        )
    if semantic != pinned_semantic:
        raise RuntimeError(f"{context} response differs from the exact pinned oracle")
    return fingerprint


def _execution_evidence(response: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    routes = response.get("routes")
    if not isinstance(routes, list):
        return evidence
    for route in routes:
        if not isinstance(route, dict):
            continue
        metrics = route.get("metrics")
        if not isinstance(metrics, dict):
            continue
        evidence.append(
            {
                "route_kind": route.get("route_kind"),
                "algorithm": metrics.get("algorithm"),
                "exactness_status": metrics.get("exactness_status"),
                "optimization_mode": metrics.get("optimization_mode"),
                "search_diagnostics": metrics.get("search_diagnostics"),
            }
        )
    return evidence


def run_benchmark(*, timed_runs: int, preload_runs: int) -> dict[str, Any]:
    if timed_runs < 1:
        raise ValueError("timed_runs must be positive")
    if preload_runs < 1:
        raise ValueError("preload_runs must be positive")

    artifact_identity = _artifact_identity()
    oracle, pinned_semantic = _load_pinned_oracle()

    fresh_preloads = [_run_preload_worker() for _ in range(preload_runs)]
    preload_wall_seconds = [float(record["wall_seconds"]) for record in fresh_preloads]
    preload_cpu_seconds = [float(record["cpu_seconds"]) for record in fresh_preloads]

    clear_route_caches()
    session_preload = _measure_preload()
    request = _request()

    clear_route_response_cache()
    first_response, first_wall_seconds, first_cpu_seconds = _timed_request(request)
    _assert_response_cache_state(
        first_response,
        expected_hit=False,
        context="first uncached warm-up",
    )
    first_fingerprint = _validate_response_semantics(
        first_response,
        pinned_semantic=pinned_semantic,
        context="first uncached warm-up",
    )

    responses: list[dict[str, Any]] = []
    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    fingerprints: list[str] = []
    for index in range(timed_runs):
        clear_route_response_cache()
        response, wall, cpu = _timed_request(request)
        context = f"timed uncached request {index + 1}"
        _assert_response_cache_state(response, expected_hit=False, context=context)
        fingerprint = _validate_response_semantics(
            response,
            pinned_semantic=pinned_semantic,
            context=context,
        )
        responses.append(response)
        wall_seconds.append(wall)
        cpu_seconds.append(cpu)
        fingerprints.append(fingerprint)

    warm_response, warm_wall_seconds, warm_cpu_seconds = _timed_request(request)
    _assert_response_cache_state(
        warm_response,
        expected_hit=True,
        context="warm response-cache control",
    )
    warm_fingerprint = _validate_response_semantics(
        warm_response,
        pinned_semantic=pinned_semantic,
        context="warm response-cache control",
    )

    parent_peak_rss_bytes = _peak_rss_bytes()
    peak_rss_bytes = max(
        parent_peak_rss_bytes,
        *(int(record["peak_rss_bytes"]) for record in fresh_preloads),
    )
    _validate_rss(peak_rss_bytes, field="peak_rss_bytes")

    result = {
        "schema_version": 2,
        "request": request.to_dict(),
        "deadline_seconds": DEADLINE_SECONDS,
        "warmup_runs": 1,
        "timed_runs": timed_runs,
        "preload_runs": preload_runs,
        "uncached_plan_routes_wall_seconds": wall_seconds,
        "uncached_plan_routes_cpu_seconds": cpu_seconds,
        "uncached_plan_routes_median_seconds": statistics.median(wall_seconds),
        "uncached_plan_routes_min_seconds": min(wall_seconds),
        "uncached_plan_routes_max_seconds": max(wall_seconds),
        "uncached_plan_routes_cpu_median_seconds": statistics.median(cpu_seconds),
        "first_uncached_plan_routes": {
            "wall_seconds": first_wall_seconds,
            "cpu_seconds": first_cpu_seconds,
            "semantic_fingerprint_sha256": first_fingerprint,
            "diagnostics": first_response.get("diagnostics", {}),
        },
        "warm_response_cache_hit": {
            "wall_seconds": warm_wall_seconds,
            "cpu_seconds": warm_cpu_seconds,
            "semantic_fingerprint_sha256": warm_fingerprint,
            "diagnostics": warm_response.get("diagnostics", {}),
        },
        "preload_wall_seconds": preload_wall_seconds,
        "preload_cpu_seconds": preload_cpu_seconds,
        "preload_median_seconds": statistics.median(preload_wall_seconds),
        "preload_min_seconds": min(preload_wall_seconds),
        "preload_max_seconds": max(preload_wall_seconds),
        "preload_cpu_median_seconds": statistics.median(preload_cpu_seconds),
        "fresh_preloads": fresh_preloads,
        "session_preload": session_preload,
        "parent_peak_rss_bytes": parent_peak_rss_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "semantic_fingerprint_sha256": EXPECTED_SEMANTIC_FINGERPRINT_SHA256,
        "semantic_fingerprints_sha256": [
            first_fingerprint,
            *fingerprints,
            warm_fingerprint,
        ],
        "artifact_identity": {
            **artifact_identity,
            "oracle": {
                "path": str(ORACLE_PATH),
                "sha256": _sha256_file(ORACLE_PATH),
                "semantic_fingerprint_sha256": oracle["semantic_fingerprint_sha256"],
            },
        },
        "host": _host_metadata(),
        "phase_diagnostics": [
            response.get("diagnostics", {}) for response in responses
        ],
        "execution_evidence": _execution_evidence(responses[0]),
        "semantic_response": _semantic_response(responses[0]),
        "complete_response": responses[0],
    }
    _atomic_json_write(LATEST_PATH, result)
    return result


def _preload_worker_main() -> None:
    _artifact_identity()
    clear_route_caches()
    record = _measure_preload()
    _validate_rss(record["peak_rss_bytes"], field="preload.peak_rss_bytes")
    print(
        _PRELOAD_RESULT_PREFIX
        + json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--preload-runs", type=int, default=3)
    parser.add_argument("--preload-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preload_worker:
        _preload_worker_main()
        return

    result = run_benchmark(
        timed_runs=args.timed_runs,
        preload_runs=args.preload_runs,
    )
    print(
        "METRIC uncached_plan_routes_median_seconds="
        f"{result['uncached_plan_routes_median_seconds']:.9f}"
    )
    print(
        "METRIC uncached_plan_routes_min_seconds="
        f"{result['uncached_plan_routes_min_seconds']:.9f}"
    )
    print(
        "METRIC uncached_plan_routes_max_seconds="
        f"{result['uncached_plan_routes_max_seconds']:.9f}"
    )
    print(
        "METRIC uncached_plan_routes_cpu_median_seconds="
        f"{result['uncached_plan_routes_cpu_median_seconds']:.9f}"
    )
    print(
        "METRIC first_uncached_plan_routes_seconds="
        f"{result['first_uncached_plan_routes']['wall_seconds']:.9f}"
    )
    print(
        "METRIC warm_response_cache_hit_seconds="
        f"{result['warm_response_cache_hit']['wall_seconds']:.9f}"
    )
    print(f"METRIC preload_median_seconds={result['preload_median_seconds']:.9f}")
    print(f"METRIC preload_min_seconds={result['preload_min_seconds']:.9f}")
    print(f"METRIC preload_max_seconds={result['preload_max_seconds']:.9f}")
    print(
        f"METRIC preload_cpu_median_seconds={result['preload_cpu_median_seconds']:.9f}"
    )
    print(f"METRIC peak_rss_gib={result['peak_rss_bytes'] / 1024**3:.9f}")


if __name__ == "__main__":
    main()
