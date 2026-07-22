"""Deterministic bounded routing profile for local autoresearch gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.routing import production_benchmark  # noqa: E402

CORPUS_PATH = Path("scripts/routing/production_benchmark_pairs.json")
GRAPH_PATH = Path(
    "data/processed/road_graphs/new_england_north_full_bbox_v1/road_graph.sqlite3"
)
REPORT_PATH = Path(
    "data/processed/heuristic_runs/new_england_north_z14_v6_learned/report/report.json"
)
SIDECAR_PATH = GRAPH_PATH.with_name(f"{GRAPH_PATH.name}.edge_projection_index")
EVIDENCE_ROOT = Path(
    "data/processed/routing_benchmarks/autoresearch_profile_runs"
)

EXPECTED_INPUTS = {
    "corpus": (
        CORPUS_PATH,
        3_007,
        "b92cfcbe67b9ce864752c62364b0c550ca5567eabe39af0018b98f2a60c59d6a",
    ),
    "graph": (
        GRAPH_PATH,
        3_049_209_856,
        "48c5b052deaf23a5e0fee29262fd384caec859a6317f6f7eb7a90eef07ae09b8",
    ),
    "report": (
        REPORT_PATH,
        66_101_946,
        "249fc55085b6fe17396fa022ff550100e7accbaca8902813cc6f53374dc88b39",
    ),
    "sidecar": (
        SIDECAR_PATH,
        508_427_024,
        "a2e271c78fcb4791067bff804161f72df575283a672d3e448eed98d602fdbd08",
    ),
}

CASE_MATRIX = (
    ("short_burlington_01", 0.0, 1.0, False),
    ("medium_burlington_montpelier", 0.9, 1.8, False),
    ("long_burlington_st_johnsbury", 0.9, 1.8, False),
    ("checked_in_default_reproduction", 0.8, 1.8, False),
)
CASE_TIMEOUT_SECONDS = 10.0


def _verify_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, (path, expected_size, expected_digest) in EXPECTED_INPUTS.items():
        stat = path.stat()
        if stat.st_size != expected_size:
            raise RuntimeError(
                f"immutable {name} size mismatch: expected {expected_size}, "
                f"observed {stat.st_size}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        observed_digest = digest.hexdigest()
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"immutable {name} digest mismatch: expected {expected_digest}, "
                f"observed {observed_digest}"
            )
        observed[name] = observed_digest
    return observed


def _case_specs(corpus: dict[str, Any]) -> list[production_benchmark.CaseSpec]:
    pairs = {str(pair["id"]): pair for pair in corpus["pairs"]}
    specs: list[production_benchmark.CaseSpec] = []
    for pair_id, q, kappa, avoid_highways in CASE_MATRIX:
        pair = pairs[pair_id]
        specs.append(
            production_benchmark.CaseSpec(
                pair_id=pair_id,
                start=(float(pair["start"][0]), float(pair["start"][1])),
                end=(float(pair["end"][0]), float(pair["end"][1])),
                q=q,
                kappa=kappa,
                avoid_highways=avoid_highways,
            )
        )
    return specs


def _fingerprint(
    specs: list[production_benchmark.CaseSpec], input_digests: dict[str, str]
) -> str:
    implementation = {
        name: production_benchmark._path_identity(path)
        for name, path in production_benchmark._BENCHMARK_IMPLEMENTATION_PATHS.items()
    }
    implementation["autoresearch_profile"] = production_benchmark._path_identity(
        Path(__file__).resolve()
    )
    return production_benchmark._json_digest(
        {
            "schema_version": 1,
            "inputs": input_digests,
            "implementation": implementation,
            "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
            "cases": [
                {
                    "case_id": production_benchmark._case_id(spec),
                    "start": spec.start,
                    "end": spec.end,
                }
                for spec in specs
            ],
            "seed": None,
            "cache_policy": "one_fresh_process_clear_then_prewarm_sequential_cases",
        }
    )


def _cpu_class() -> str:
    cpu = platform.processor().strip()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return "_".join(cpu.split()) or "unknown"


def main() -> None:
    total_started = perf_counter()
    input_digests = _verify_inputs()
    corpus = production_benchmark._load_corpus(CORPUS_PATH)
    specs = _case_specs(corpus)
    fingerprint = _fingerprint(specs, input_digests)

    (
        preload,
        preload_wall_ms,
        planner,
        edge_index,
        node_index,
        planner_preload,
    ) = production_benchmark._prepare_benchmark_context(GRAPH_PATH, REPORT_PATH)
    context = (preload, planner, edge_index, node_index)
    route_error_cache: dict[tuple[str, bool], tuple[str, str]] = {}
    rows = [
        production_benchmark._execute_case(
            spec=spec,
            index=index,
            graph_path=GRAPH_PATH,
            report_path=REPORT_PATH,
            case_timeout_seconds=CASE_TIMEOUT_SECONDS,
            strict_service_full=False,
            context=context,
            route_error_cache=route_error_cache,
        )
        for index, spec in enumerate(specs)
    ]

    completed = [row for row in rows if row["evaluation"].get("status") == "ok"]
    if not completed:
        raise RuntimeError("bounded profile produced no valid completed route")
    for row in completed:
        failed_invariants = row["evaluation"].get("failed_invariants", [])
        if failed_invariants:
            raise RuntimeError(
                f"protected invariants failed for {row['case_id']}: {failed_invariants}"
            )
    unexpected = [
        (row["case_id"], row.get("reason"))
        for row in rows
        if row["evaluation"].get("status") != "ok"
        and row.get("reason") not in {"timeout", "no_route"}
    ]
    if unexpected:
        raise RuntimeError(f"unexpected bounded-profile failures: {unexpected!r}")

    wall_values = sorted(float(row["wall_ms"]) for row in rows)
    p95_ms = wall_values[min(len(wall_values) - 1, math.ceil(0.95 * len(wall_values)) - 1)]
    under_10s_count = sum(value < 10_000.0 for value in wall_values)
    timeout_count = sum(row.get("reason") == "timeout" for row in rows)
    no_route_count = sum(row.get("reason") == "no_route" for row in rows)
    q0_rows = [row for row in completed if float(row["q"]) == 0.0]
    q0_fastest_pass = sum(
        row["evaluation"].get("invariants", {}).get("q0_fastest") is True
        for row in q0_rows
    )
    strict_service_count = sum(
        row.get("execution_mode") == "strict_service" for row in rows
    )
    parity = next(
        (row.get("ui_reproduction_parity") for row in rows if "ui_reproduction_parity" in row),
        None,
    )
    if isinstance(parity, dict) and parity.get("pass") is not True:
        raise RuntimeError("strict-service/direct-planner parity failed")

    summary = {
        "schema_version": 1,
        "benchmark": "autoresearch_bounded_routing_profile",
        "fingerprint": fingerprint,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
        "denominator": len(rows),
        "cache_policy": "one_fresh_process_clear_then_prewarm_sequential_cases",
        "input_digests": input_digests,
        "preload": {
            **preload,
            "preload_wall_ms": preload_wall_ms,
            "planner_matrix_preload": planner_preload,
        },
        "rows": rows,
        "metrics": {
            "profile_case_median_ms": median(wall_values),
            "profile_case_p95_ms": p95_ms,
            "under_10s_count": under_10s_count,
            "under_10s_rate": under_10s_count / len(rows),
            "completed_cases": len(completed),
            "timeout_cases": timeout_count,
            "no_route_cases": no_route_count,
            "q0_fastest_pass": q0_fastest_pass,
            "strict_service_cases": strict_service_count,
            "ui_reproduction_parity_available": int(isinstance(parity, dict)),
            "ui_reproduction_parity_pass": int(
                isinstance(parity, dict) and parity.get("pass") is True
            ),
            "correctness_failures": 0,
        },
        "total_wall_ms": (perf_counter() - total_started) * 1000.0,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    evidence_digest = hashlib.sha256(encoded.encode()).hexdigest()
    evidence_directory = EVIDENCE_ROOT / f"{fingerprint}-{evidence_digest[:16]}"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_directory / "profile.json"
    temporary_path = evidence_path.with_suffix(".json.tmp")
    temporary_path.write_text(encoded, encoding="utf-8")
    temporary_path.replace(evidence_path)

    metrics = summary["metrics"]
    print(f"ASI evidence_path={evidence_path}")
    print(f"ASI benchmark_fingerprint={fingerprint}")
    print(f"ASI cpu_class={_cpu_class()}")
    print(f"ASI architecture={platform.machine()}")
    print(f"ASI denominator={len(rows)}")
    print("ASI seed=none_no_rng")
    print(f"ASI configured_case_timeout_seconds={CASE_TIMEOUT_SECONDS}")
    print("ASI workers=1")
    print("ASI group_size=4")
    print("ASI cache_policy=one_fresh_process_clear_then_prewarm_sequential_cases")
    print(f"METRIC profile_case_median_ms={metrics['profile_case_median_ms']:.6f}")
    print(f"METRIC profile_case_p95_ms={metrics['profile_case_p95_ms']:.6f}")
    print(f"METRIC under_10s_rate={metrics['under_10s_rate']:.12f}")
    print(f"METRIC under_10s_count={metrics['under_10s_count']}")
    print(f"METRIC completed_cases={metrics['completed_cases']}")
    print(f"METRIC timeout_cases={metrics['timeout_cases']}")
    print(f"METRIC no_route_cases={metrics['no_route_cases']}")
    print(f"METRIC q0_fastest_pass={metrics['q0_fastest_pass']}")
    print(f"METRIC strict_service_cases={metrics['strict_service_cases']}")
    print(
        "METRIC ui_reproduction_parity_available="
        f"{metrics['ui_reproduction_parity_available']}"
    )
    print(
        "METRIC ui_reproduction_parity_pass="
        f"{metrics['ui_reproduction_parity_pass']}"
    )
    print("METRIC correctness_failures=0")
    print(f"METRIC preload_ms={preload_wall_ms:.6f}")
    print(f"METRIC total_wall_ms={summary['total_wall_ms']:.6f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
