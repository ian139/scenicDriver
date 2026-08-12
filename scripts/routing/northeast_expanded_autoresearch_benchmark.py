#!/usr/bin/env python3
"""Benchmark the production Northeast Expanded Burlington-to-Pittsburgh request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
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
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(response)


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


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
    return {
        "commit": _command_output("git", "rev-parse", "HEAD"),
        "cpu": _command_output("sysctl", "-n", "machdep.cpu.brand_string"),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": _command_output("sysctl", "-n", "hw.memsize"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "load_average": list(os.getloadavg()),
        "compiler": _command_output("clang", "--version"),
        "native_compile_flags": ["-O3", "-shared", "-fPIC"],
        "native_library": {
            "path": str(native_library),
            "sha256": _sha256_file(native_library) if native_library.exists() else None,
            "size_bytes": native_library.stat().st_size if native_library.exists() else None,
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


def run_benchmark(*, timed_runs: int) -> dict[str, Any]:
    if timed_runs < 1:
        raise ValueError("timed_runs must be positive")
    for path in (GRAPH_PATH, REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact_hashes = {
        GRAPH_PATH: _sha256_file(GRAPH_PATH),
        REPORT_PATH: _sha256_file(REPORT_PATH),
    }
    expected_hashes = {
        GRAPH_PATH: EXPECTED_GRAPH_SHA256,
        REPORT_PATH: EXPECTED_REPORT_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = artifact_hashes[path]
        if observed != expected:
            raise RuntimeError(
                f"artifact SHA-256 mismatch for {path}: "
                f"expected {expected}, observed {observed}"
            )


    clear_route_caches()
    preload_started = time.perf_counter()
    preload = preload_route_assets(
        GRAPH_PATH,
        REPORT_PATH,
        exclusive_scoring=True,
    )
    preload_seconds = time.perf_counter() - preload_started

    request = _request()
    plan_routes(request, deadline=RoutingDeadline.after(DEADLINE_SECONDS))

    responses: list[dict[str, Any]] = []
    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    fingerprints: list[str] = []
    for _ in range(timed_runs):
        response, wall, cpu = _timed_request(request)
        semantic = _semantic_response(response)
        fingerprint = _json_digest(semantic)
        responses.append(response)
        wall_seconds.append(wall)
        cpu_seconds.append(cpu)
        fingerprints.append(fingerprint)

    if len(set(fingerprints)) != 1:
        raise RuntimeError(f"timed route responses diverged: {fingerprints}")

    semantic_response = _semantic_response(responses[0])
    fingerprint = fingerprints[0]
    if fingerprint != EXPECTED_SEMANTIC_FINGERPRINT_SHA256:
        raise RuntimeError(
            "route semantics differ from the pinned baseline: "
            f"expected {EXPECTED_SEMANTIC_FINGERPRINT_SHA256}, "
            f"observed {fingerprint}"
        )
    if ORACLE_PATH.exists():
        oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
        expected = oracle.get("semantic_fingerprint_sha256")
        if expected != fingerprint:
            raise RuntimeError(
                "route semantics differ from baseline oracle: "
                f"expected {expected}, observed {fingerprint}"
            )
    else:
        _atomic_json_write(
            ORACLE_PATH,
            {
                "semantic_fingerprint_sha256": fingerprint,
                "semantic_response": semantic_response,
                "complete_response": responses[0],
            },
        )

    peak_rss_bytes = _peak_rss_bytes()
    if peak_rss_bytes > MAX_RSS_BYTES:
        raise MemoryError(
            f"peak RSS {peak_rss_bytes} exceeds {MAX_RSS_BYTES}-byte hard stop"
        )

    result = {
        "schema_version": 1,
        "request": request.to_dict(),
        "deadline_seconds": DEADLINE_SECONDS,
        "warmup_runs": 1,
        "timed_runs": timed_runs,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "wall_median_seconds": statistics.median(wall_seconds),
        "wall_min_seconds": min(wall_seconds),
        "wall_max_seconds": max(wall_seconds),
        "cpu_median_seconds": statistics.median(cpu_seconds),
        "preload_seconds": preload_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "semantic_fingerprint_sha256": fingerprint,
        "semantic_fingerprints_sha256": fingerprints,
        "artifact_identity": {
            "graph": {
                "path": str(GRAPH_PATH),
                "sha256": artifact_hashes[GRAPH_PATH],
                "size_bytes": GRAPH_PATH.stat().st_size,
            },
            "report": {
                "path": str(REPORT_PATH),
                "sha256": artifact_hashes[REPORT_PATH],
                "size_bytes": REPORT_PATH.stat().st_size,
            },
        },
        "host": _host_metadata(),
        "preload": preload,
        "phase_diagnostics": [response.get("diagnostics", {}) for response in responses],
        "complete_response": responses[0],
    }
    _atomic_json_write(LATEST_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timed-runs", type=int, default=3)
    args = parser.parse_args()
    result = run_benchmark(timed_runs=args.timed_runs)
    print(f"METRIC warm_plan_routes_median_seconds={result['wall_median_seconds']:.9f}")
    print(f"METRIC warm_plan_routes_min_seconds={result['wall_min_seconds']:.9f}")
    print(f"METRIC warm_plan_routes_max_seconds={result['wall_max_seconds']:.9f}")
    print(f"METRIC warm_plan_routes_cpu_median_seconds={result['cpu_median_seconds']:.9f}")
    print(f"METRIC preload_seconds={result['preload_seconds']:.9f}")
    print(f"METRIC peak_rss_gib={result['peak_rss_bytes'] / 1024**3:.9f}")


if __name__ == "__main__":
    main()
