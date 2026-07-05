from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_promote_regression_model_smoke(tmp_path: Path) -> None:
    candidate_metrics = tmp_path / "candidate.json"
    baseline_metrics = tmp_path / "baseline.json"
    registry_json = tmp_path / "model_registry.json"
    candidate_ckpt = tmp_path / "candidate.pt"
    candidate_ckpt.write_bytes(b"placeholder")

    candidate_metrics.write_text(json.dumps({"corr": 0.95, "mae": 0.2, "rmse": 0.3, "samples": 100}), encoding="utf-8")
    baseline_metrics.write_text(json.dumps({"corr": 0.9, "mae": 0.25, "rmse": 0.35, "samples": 100}), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/modeling/promote_regression_model.py"),
        "--candidate-metrics",
        str(candidate_metrics),
        "--baseline-metrics",
        str(baseline_metrics),
        "--candidate-checkpoint",
        str(candidate_ckpt),
        "--registry-json",
        str(registry_json),
        "--run-name",
        "smoke_vx",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    registry = json.loads(registry_json.read_text(encoding="utf-8"))
    assert registry["active"] is not None
    assert registry["active"]["run_name"] == "smoke_vx"
    assert registry["active"]["checkpoint"] == str(candidate_ckpt)


def test_promote_regression_model_from_benchmark_comparison(tmp_path: Path) -> None:
    active_comparison = tmp_path / "active_compare.json"
    control_comparison = tmp_path / "control_compare.json"
    registry_json = tmp_path / "model_registry.json"
    candidate_ckpt = tmp_path / "candidate.pt"
    candidate_ckpt.write_bytes(b"placeholder")

    comparison_payload = {
        "baseline": {"corr": 0.9, "mae": 0.25, "rmse": 0.35, "samples": 20},
        "candidate": {"corr": 0.95, "mae": 0.2, "rmse": 0.3, "samples": 20},
        "deltas": {"corr": 0.05, "mae": -0.05, "rmse": -0.05},
    }
    active_comparison.write_text(json.dumps(comparison_payload), encoding="utf-8")
    control_comparison.write_text(json.dumps(comparison_payload), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/modeling/promote_regression_model.py"),
        "--benchmark-comparison",
        str(active_comparison),
        "--required-control-comparison",
        str(control_comparison),
        "--candidate-checkpoint",
        str(candidate_ckpt),
        "--registry-json",
        str(registry_json),
        "--run-name",
        "benchmark_vx",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    registry = json.loads(registry_json.read_text(encoding="utf-8"))
    assert registry["active"] is not None
    assert registry["active"]["run_name"] == "benchmark_vx"
    assert registry["active"]["metrics"] == {"corr": 0.95, "mae": 0.2, "rmse": 0.3, "samples": 20}
    assert registry["active"]["source_metrics"] == str(active_comparison)


def test_route_compare_service_smoke(tmp_path: Path) -> None:
    graph = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "e0",
                    "road_type": "secondary",
                    "scenic_score": 5.0,
                    "one_way": True,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-72.0, 42.0], [-72.0, 42.1]],
                },
            }
        ],
    }

    root = tmp_path
    graph_path = root / "graph.geojson"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    run_name = "smoke_run"
    report_dir = root / "data/processed/heuristic_runs" / run_name / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tiles": [
            {"z": 14, "x": 4915, "y": 6078, "scenic_score": 6.5},
        ]
    }
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/routing/route_compare_service.py"),
        "--start",
        "42.0",
        "-72.0",
        "--end",
        "42.1",
        "-72.0",
        "--run-name",
        run_name,
        "--graph-geojson",
        str(graph_path),
    ]
    subprocess.run(cmd, check=True, cwd=root)

    route_geojson = report_dir / "route.geojson"
    route_metrics = report_dir / "route_metrics.json"
    assert route_geojson.exists()
    assert route_metrics.exists()
    metrics = json.loads(route_metrics.read_text(encoding="utf-8"))
    assert "scenic" in metrics
    assert "score_mapping" in metrics
