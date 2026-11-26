import os
from pathlib import Path
import subprocess
import sys

from scenicdriver.data_pipeline.web_mercator import lat_lon_to_tile


def test_lat_lon_to_tile_shape() -> None:
    x, y = lat_lon_to_tile(42.36, -72.52, 16)
    assert isinstance(x, int)
    assert isinstance(y, int)


def test_installed_namespace_imports_outside_checkout(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import scenicdriver.app_api.main; "
            "import scenicdriver.active_learning; "
            "import scenicdriver.classifier; "
            "import scenicdriver.data_pipeline; "
            "import scenicdriver.heuristics; "
            "import scenicdriver.route_planner; "
            "import scenicdriver.scenic_scorer; "
            "import scenicdriver.terrain",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_api_resource_root_matches_checkout() -> None:
    from scenicdriver.app_api import main

    root = Path(__file__).resolve().parents[1]
    assert main.PROJECT_ROOT == root
    assert main.APP_REGIONS_PATH == root / "config/app_regions.json"
    assert main.APP_REGIONS_PATH.is_file()
    assert main.ROAD_GRAPHS_DIR == root / "data/processed/road_graphs"
    assert main.RUNS_DIR == root / "data/processed/heuristic_runs"
    assert main.MODEL_REGISTRY_PATH == root / "data/processed/regression/model_registry.json"
