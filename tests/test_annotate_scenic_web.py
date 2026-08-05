from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
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
    labels.write_text(
        "image_path,region,selection_reason\ntile.png,west,disagreement\n"
    )
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
    with pytest.raises(SystemExit):
        parse_args(["--host", "0.0.0.0", "--allow-remote"])
    with pytest.raises(SystemExit):
        parse_args(
            ["--host", "0.0.0.0", "--allow-remote", "--session-token", "secret123"]
        )
    parsed = parse_args(
        [
            "--host",
            "0.0.0.0",
            "--allow-remote",
            "--session-token",
            "secret123",
            "--tls-cert",
            "server.crt",
            "--tls-key",
            "server.key",
        ]
    )
    assert parsed.allow_remote is True
    assert parsed.session_token == "secret123"


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
        {
            "image_path": "tile.png",
            "scenic_human": 8,
            "confidence": "high",
            "notes": "clear",
        }
    )
    assert saved["record"]["annotator_id"] == "alice"
    assert state.get_annotation("tile.png")["record"]["scenic_human"] == 8
    resumed = state.load_batch({})
    assert resumed["progress"]["completed"] == 1
    summary = state.summary()
    assert summary["confidence"]["high"] == 1
    assert summary["coverage"] == {"west": {"completed": 1, "total": 1}}
    assert Path(state.progress_path).exists()
    assert (
        list(
            __import__("pandas").read_csv(tmp_path / "raw" / "labels_human.csv").columns
        )
        == DEFAULT_COLUMNS
    )


def test_server_identity_cannot_be_impersonated(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    state.load_batch({})
    with pytest.raises(Exception):
        state.save_annotation(
            {"image_path": "tile.png", "scenic_human": 4, "annotator_id": "mallory"}
        )


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

    threads = [
        threading.Thread(target=save, args=(path,))
        for path in ("tile.png", "other.png")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(__import__("pandas").read_csv(raw / "labels_human.csv")) == 2


def test_summary_reports_defined_overlap_consistency(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    state.load_batch({})
    annotations = tmp_path / "raw" / "labels_human.csv"
    annotations.write_text(
        "image_path,scenic_human,confidence,skip,annotator_id,timestamp,notes\n"
        "tile.png,7,high,False,alice,2026-01-01T00:00:00Z,\n"
        "tile.png,8,medium,False,bob,2026-01-01T00:01:00Z,\n"
    )

    summary = state.summary()

    assert summary["overlap_consistency"] == {
        "tiles_with_overlap": 1,
        "within_one_point_fraction": 1.0,
        "definition": "fraction of multi-annotator tiles with score range <= 1",
    }


def test_ui_contract_and_structured_error_surface() -> None:
    html = Path("scripts/annotation/annotate_scenic_web.html").read_text()
    for marker in (
        "Keyboard shortcuts",
        "Scoring anchors",
        "prefetch",
        "localStorage",
        "confidenceSummary",
        "coverageSummary",
        "overlapSummary",
        "Save and next",
        "prefers-reduced-motion",
    ):
        assert marker in html
    assert make_handler is not None


def test_blind_qa_overlap_is_exposed_without_restoring_prior_answer(
    tmp_path: Path,
) -> None:
    state = fixture_state(tmp_path)
    state.load_batch({})
    state.save_annotation(
        {
            "image_path": "tile.png",
            "scenic_human": 5,
            "confidence": "high",
            "skip": False,
            "notes": "earlier answer",
        }
    )
    batch_csv = tmp_path / "qa_batch.csv"
    batch_csv.write_text(
        "image_path,is_qa_overlap,selection_reason\ntile.png,True,qa\n"
    )

    loaded = state.load_batch({"batch_csv": str(batch_csv)})
    assert loaded["batch"][0]["is_qa_overlap"] is True
    assert state.get_annotation("tile.png") == {
        "found": False,
        "image_path": "tile.png",
    }


def test_remote_session_bootstrap_and_authenticated_access(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    session_token = "secret_session_token_12345"
    handler_class = make_handler(state, remote=True, session_token=session_token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    host, port = server.server_address
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    try:
        url_base = f"http://{host}:{port}"

        # 1. Access protected route without cookie -> 401 Unauthorized
        req_unauth = urllib.request.Request(f"{url_base}/api/state")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_unauth)
        assert exc_info.value.code == 401

        # 2. Bootstrap session without proof -> 401 Unauthorized
        req_no_proof = urllib.request.Request(f"{url_base}/api/session")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_no_proof)
        assert exc_info.value.code == 401

        # 3. Bootstrap session with wrong bearer proof -> 401 Unauthorized
        req_wrong_proof = urllib.request.Request(
            f"{url_base}/api/session",
            headers={"Authorization": "Bearer wrong_token"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_wrong_proof)
        assert exc_info.value.code == 401

        # 4. Bootstrap session with correct bearer proof -> 200 OK & Set-Cookie
        req_bootstrap = urllib.request.Request(
            f"{url_base}/api/session",
            headers={"Authorization": f"Bearer {session_token}"},
        )
        with urllib.request.urlopen(req_bootstrap) as resp:
            assert resp.status == 200
            cookie_header = resp.headers.get("Set-Cookie")
            assert cookie_header is not None
            assert "scenic_session=secret_session_token_12345" in cookie_header
            assert "HttpOnly" in cookie_header
            assert "SameSite=Strict" in cookie_header
            assert "Secure" in cookie_header
            body = resp.read().decode("utf-8")
            assert "secret_session_token_12345" not in body

        # 5. Access protected route with cookie -> 200 OK
        req_auth = urllib.request.Request(
            f"{url_base}/api/state",
            headers={"Cookie": f"scenic_session={session_token}"},
        )
        with urllib.request.urlopen(req_auth) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["schema_version"] == 1
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
