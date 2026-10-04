from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.remote import cmux_vast_host


def tar_names(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        return set(archive.getnames())


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def path_components(name: str) -> set[str]:
    return {part for part in name.split("/") if part}


def git(source: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=source, check=True, capture_output=True, text=True)


def init_repo(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    git(source, "init", "-b", "test-branch")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test User")


def test_make_overlay_tarball_excludes_git_and_generated_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    init_repo(source)
    write_file(source / "pkg/.git/objects/pack/nested.pack")
    write_file(source / "data/processed/regression/out.json")
    write_file(source / "models/model.bin")
    write_file(source / "scenic_artifacts/run.txt")
    write_file(source / "cache/item")
    write_file(source / ".venv/bin/python")
    write_file(source / ".cmux-vast/state/task.json")
    write_file(source / ".secrets/aws.env")
    write_file(source / "src/__pycache__/module.pyc")
    write_file(source / "ordinary/project_file.py")
    write_file(source / ".worktrees/full-checkout/src/module.py")

    tarball = cmux_vast_host.make_overlay_tarball(source)
    try:
        names = tar_names(tarball)
    finally:
        tarball.unlink(missing_ok=True)

    for name in names:
        components = path_components(name)
        assert ".git" not in components
        assert "__pycache__" not in components
        assert ".pytest_cache" not in components

    assert not any(name.endswith("data/processed/regression/out.json") for name in names)
    assert not any(name.endswith("models/model.bin") for name in names)
    assert not any(name.endswith("scenic_artifacts/run.txt") for name in names)
    assert not any(name.endswith("cache/item") for name in names)
    assert not any(".venv" in path_components(name) for name in names)
    assert not any(".cmux-vast" in path_components(name) for name in names)
    assert not any(".worktrees" in path_components(name) for name in names)
    assert not any(".secrets" in path_components(name) for name in names)
    assert any(name.endswith("ordinary/project_file.py") for name in names)


def test_make_overlay_tarball_excludes_gitignored_credentials_present_on_disk(tmp_path: Path) -> None:
    source = tmp_path / "source"
    init_repo(source)
    write_file(source / ".gitignore", "*.env\ncredentials/\n")
    write_file(source / "ordinary/tracked.py")
    git(source, "add", ".gitignore", "ordinary/tracked.py")
    git(source, "commit", "-m", "initial")
    write_file(source / "config/aws.env", "AWS_SECRET_ACCESS_KEY=leak")
    write_file(source / "credentials/token.json", '{"token": "leak"}')
    write_file(source / "ordinary/untracked.py")

    tarball = cmux_vast_host.make_overlay_tarball(source)
    try:
        names = tar_names(tarball)
        with tarfile.open(tarball, "r:gz") as archive:
            payload = b"".join(
                archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()
            )
    finally:
        tarball.unlink(missing_ok=True)

    assert (source / "config/aws.env").exists()
    assert not any(name.endswith("aws.env") for name in names)
    assert not any("credentials" in path_components(name) for name in names)
    assert b"leak" not in payload
    assert "ordinary/tracked.py" in names
    assert "ordinary/untracked.py" in names


def test_make_overlay_tarball_refuses_source_without_git_ignore_rules(tmp_path: Path) -> None:
    source = tmp_path / "plain"
    write_file(source / "config/aws.env", "AWS_SECRET_ACCESS_KEY=leak")

    with pytest.raises(RuntimeError, match="git ls-files"):
        cmux_vast_host.make_overlay_tarball(source)


def test_make_repo_tarball_excludes_clone_git_and_preserves_project_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "repo"
    init_repo(source)
    write_file(source / ".gitignore", "*.env\n")
    write_file(source / "ordinary/project_file.py")
    git(source, "add", ".gitignore", "ordinary/project_file.py")
    git(source, "commit", "-m", "initial")
    write_file(source / "ordinary/uncommitted.py")
    write_file(source / "ordinary/local.env", "AWS_SECRET_ACCESS_KEY=leak")
    write_file(source / ".git/objects/pack/big.pack")
    monkeypatch.setattr(cmux_vast_host, "PROJECT_ROOT", source)

    tarball = cmux_vast_host.make_repo_tarball("test-branch")
    try:
        names = tar_names(tarball)
    finally:
        tarball.unlink(missing_ok=True)

    for name in names:
        assert ".git" not in path_components(name)
    assert not any(name.endswith("big.pack") for name in names)
    assert "scenic-drive/ordinary/project_file.py" in names
    assert "scenic-drive/ordinary/uncommitted.py" in names
    assert "scenic-drive/ordinary/local.env" not in names


def test_record_startup_timing_writes_elapsed_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[dict] = []
    state = {"task_name": "timing-smoke"}

    monkeypatch.setattr(cmux_vast_host.time, "monotonic", lambda: 12.345)
    monkeypatch.setattr(cmux_vast_host, "write_state", lambda current: writes.append(dict(current)))

    cmux_vast_host.record_startup_timing(state, "repo_upload", 10.0)

    assert state["startup_timings_seconds"]["repo_upload"] == 2.345
    assert writes == [state]
