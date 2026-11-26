from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _ignored(path: str) -> bool:
    """Apply the repository's Docker globs to a path and its parent directories."""
    parts = path.strip("/").split("/")
    candidates = ["/".join(parts[:end]) for end in range(1, len(parts) + 1)]
    ignored = False
    for line in (ROOT / ".dockerignore").read_text().splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        include = pattern.startswith("!")
        pattern = pattern.removeprefix("!").strip("/")
        regex = re.escape(pattern)
        regex = regex.replace(r"\*\*/", "(?:.*/)?").replace(r"\*\*", ".*")
        regex = regex.replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
        if any(re.fullmatch(regex, candidate) for candidate in candidates):
            ignored = not include
    return ignored


@pytest.mark.parametrize("path", [
    ".worktrees/worker/src/module.py",
    ".todos/state.json",
    ".omp/session.json",
    "scenic_drive.egg-info/PKG-INFO",
    "src/scenic_drive.egg-info/PKG-INFO",
    "build/lib/module.py",
    "src/build/lib/module.py",
    "dist/package.whl",
    "src/dist/package.whl",
    "_compact_search_c.so",
    "src/scenicdriver/route_planner/_compact_search_c.so",
    "src/deep/nested/extension.so",
    "module.pyc",
    "src/deep/nested/module.pyc",
    "__pycache__/module.pyc",
    "src/deep/__pycache__/metadata.json",
])
def test_host_operator_and_build_state_is_excluded(path: str) -> None:
    assert _ignored(path), f"{path} is admitted to the Docker build context"


def test_runtime_sources_and_copy_inputs_remain_in_context() -> None:
    paths = ["pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "data/README.md",
             "src/scenicdriver/route_planner/_compact_search_c.c"]
    for directory in ("src", "scripts", "notebooks"):
        paths.extend(str(path.relative_to(ROOT)) for path in (ROOT / directory).rglob("*.py"))
    paths.extend(str(path.relative_to(ROOT)) for path in (ROOT / "config").rglob("*") if path.is_file())
    for path in paths:
        assert not _ignored(path), f"Required runtime input {path} is excluded"


def test_remote_image_builds_and_loads_target_native_library() -> None:
    text = (ROOT / "Dockerfile.remote-training").read_text().replace("\\\n", " ")
    stages = {}
    for stage in re.split(r"^FROM\s+", text, flags=re.MULTILINE)[1:]:
        header, _, body = stage.partition("\n")
        base, name = header.split(" AS ")
        stages[name] = (base, body)
    assert stages["cuda-base"][0].startswith(
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:"
    )
    assert stages["native-builder"][0] == stages["runtime"][0] == "cuda-base"
    build = stages["native-builder"][1]
    assert re.search(r"apt-get install\b[^\n]*\bgcc\b[^\n]*\blibc6-dev\b", build)
    assert "COPY src/scenicdriver/route_planner/_compact_search_c.c ./" in build
    assert "RUN gcc -O3 -shared -fPIC _compact_search_c.c -o _compact_search_c.so -lm" in build
    runtime = stages["runtime"][1]
    copy = "COPY --from=native-builder /build/_compact_search_c.so src/scenicdriver/route_planner/_compact_search_c.so"
    assert runtime.index("COPY src/ src/") < runtime.index(copy)
    assert "touch -r src/scenicdriver/route_planner/_compact_search_c.c src/scenicdriver/route_planner/_compact_search_c.so" in runtime
    assert "from scenicdriver.route_planner._compact_search import compact_search_available; assert compact_search_available()" in runtime
    assert "coreutils" in runtime
    assert "--constraint /tmp/cuda-constraints.txt" in runtime
    assert 'assert torch.version.cuda == "12.8"' in runtime
    assert "apt-get install" in runtime and "libc6-dev" not in runtime
