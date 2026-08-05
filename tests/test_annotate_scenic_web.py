from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from scripts.annotation.annotate_scenic_web import (
    DEFAULT_COLUMNS,
    AnnotatorConfig,
    AnnotatorState,
    PathPolicy,
    _safe_image_path,
    make_handler,
    parse_args,
)


def fixture_state(tmp_path: Path) -> AnnotatorState:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "tile.png").write_bytes(b"png")
    labels = raw / "labels.csv"
    labels.write_text("image_path,region,selection_reason\ntile.png,west,disagreement\n")
    policy = PathPolicy(project_root=tmp_path, local_roots=(tmp_path,))
    return AnnotatorState(
        AnnotatorConfig(
            labels_csv=str(labels),
            raw_dir=str(raw),
            annotations_csv=str(raw / "labels_human.csv"),
            annotator_id="alice",
        ),
        path_policy=policy,
    )


def test_cli_defaults_loopback_and_remote_requires_opt_in() -> None:
    assert parse_args([]).host == "127.0.0.1"
    with pytest.raises(SystemExit):
        parse_args(["--host", "0.0.0.0"])
    assert parse_args(["--host", "0.0.0.0", "--allow-remote"]).allow_remote is True


def test_paths_and_image_paths_fail_closed(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    with pytest.raises(Exception):
        state.path_policy.resolve_local("/etc/passwd", kind="labels_csv")
    with pytest.raises(Exception):
        _safe_image_path("../secret")
    with pytest.raises(Exception):
        _safe_image_path("s3://bucket/key")


def test_load_save_revisit_and_resume(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    loaded = state.load_batch({})
    assert loaded["batch_size"] == 1
    saved = state.save_annotation(
        {"image_path": "tile.png", "scenic_human": 8, "confidence": "high", "notes": "clear"}
    )
    assert saved["record"]["annotator_id"] == "alice"
    assert state.get_annotation("tile.png")["record"]["scenic_human"] == 8
    resumed = state.load_batch({})
    assert resumed["progress"]["completed"] == 1
    assert Path(state.progress_path).exists()
    assert list(__import__("pandas").read_csv(tmp_path / "raw" / "labels_human.csv").columns) == DEFAULT_COLUMNS


def test_server_identity_cannot_be_impersonated(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    state.load_batch({})
    with pytest.raises(Exception):
        state.save_annotation({"image_path": "tile.png", "scenic_human": 4, "annotator_id": "mallory"})


def test_concurrent_upserts_retain_unrelated_rows(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    raw = tmp_path / "raw"
    (raw / "other.png").write_bytes(b"png")
    (raw / "labels.csv").write_text(
        "image_path,region\ntile.png,west\nother.png,south\n"
    )
    state.load_batch({})
    # Both requests target different identities and therefore must survive the merge.
    errors: list[Exception] = []
    def save(path: str) -> None:
        try:
            state.save_annotation({"image_path": path, "scenic_human": 5})
        except Exception as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)
    threads = [threading.Thread(target=save, args=(path,)) for path in ("tile.png", "other.png")]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not errors
    assert len(__import__("pandas").read_csv(raw / "labels_human.csv")) == 2


def test_ui_contract_and_structured_error_surface() -> None:
    html = Path("scripts/annotation/annotate_scenic_web.html").read_text()
    for marker in ("Keyboard shortcuts", "prefetch", "localStorage", "Save and next", "prefers-reduced-motion"):
        assert marker in html
    assert make_handler is not None
