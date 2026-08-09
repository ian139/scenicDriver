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
    ApiError,
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
        "coverageSummary",
        "overlapSummary",
        "Save and next",
        "prefers-reduced-motion",
        "id=\"score\"",
        "id=\"unusable\"",
        "confidence:'medium'",
        "notes:''",
        "0–9</kbd> save score and advance",
        "type <kbd>10</kbd>, then <kbd>Enter</kbd>",
        "e.preventDefault();$('score').value=k;save();return",
    ):
        assert marker in html
    for removed in (
        "id=\"confidence\"",
        "id=\"notes\"",
        "id=\"confidenceSummary\"",
        "H/M/L",
        "$('confidence')",
        "$('notes')",
        "d.confidence",
        "d.notes",
    ):
        assert removed not in html
    assert "JSON.stringify({score:$('score').value,unusable:$('unusable').value})" in html
    assert "if($('settings').contains(active))return" in html
    assert "const editing=['INPUT','TEXTAREA','SELECT'].includes(active.tagName)" in html
    assert "if(/^[0-9]$/.test(k)&&(!editing||active===$('score')))" in html
    assert "if(editing&&k!=='enter')return" in html
    assert "0–9 save score and advance · type 10, then Enter" in html
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


def test_strict_annotations_csv_schema_validation(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    annotations_file = tmp_path / "raw" / "labels_human.csv"
    # Write CSV missing required column 'confidence'
    annotations_file.write_text(
        "image_path,scenic_human,skip,annotator_id,timestamp,notes\ntile.png,5,False,alice,2026-01-01T00:00:00Z,note\n"
    )
    with pytest.raises(ApiError) as exc_info:
        state.load_batch({})
    assert exc_info.value.code == "invalid_annotations_schema"

    annotations_file.write_text(
        "image_path,scenic_human,confidence,skip,annotator_id,timestamp,notes\ntile.png,5,high,False,alice,2026-01-01T00:00:00Z,note\n"
    )
    state.load_batch({})

    # Write CSV with extra column
    annotations_file.write_text(
        "image_path,scenic_human,confidence,skip,annotator_id,timestamp,notes,extra\ntile.png,5,high,False,alice,2026-01-01T00:00:00Z,note,extra_val\n"
    )
    with pytest.raises(ApiError) as exc_info:
        state.get_annotation("tile.png")
    assert exc_info.value.code == "invalid_annotations_schema"

    # Write valid 7-column header CSV
    annotations_file.write_text(
        "image_path,scenic_human,confidence,skip,annotator_id,timestamp,notes\ntile.png,5,high,False,alice,2026-01-01T00:00:00Z,note\n"
    )
    res = state.get_annotation("tile.png")
    assert res["found"] is True
    assert res["record"]["notes"] == "note"


def test_explicit_unusable_reason_persistence_and_restoration(tmp_path: Path) -> None:
    state = fixture_state(tmp_path)
    state.load_batch({})

    # 1. Save annotation with explicit unusable reason and user notes
    saved = state.save_annotation(
        {
            "image_path": "tile.png",
            "skip": True,
            "unusable_reason": "missing_imagery",
            "notes": "blurry photo",
        }
    )
    assert saved["saved"] is True

    # Check 7-column CSV record on disk
    df = __import__("pandas").read_csv(tmp_path / "raw" / "labels_human.csv")
    assert list(df.columns) == DEFAULT_COLUMNS
    assert df.iloc[0]["notes"] == "[unusable: missing_imagery] blurry photo"

    # Check progress state JSON
    progress_file = tmp_path / "raw" / "labels_human.annotation_progress.json"
    progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
    assert (
        progress_data["batches"][state.batch_id]["alice"]["unusable"]["tile.png"]
        == "missing_imagery"
    )

    # 2. Get annotation with progress file intact -> restores reason and user notes
    res = state.get_annotation("tile.png")
    assert res["unusable_reason"] == "missing_imagery"
    assert res["record"]["notes"] == "blurry photo"

    # 3. Simulate progress state loss by removing progress JSON file
    progress_file.unlink()

    # Restoration from formatted notes in 7-column CSV record
    res_restored = state.get_annotation("tile.png")
    assert res_restored["unusable_reason"] == "missing_imagery"
    assert res_restored["record"]["notes"] == "blurry photo"

    # 4. Save with unusable_reason and empty notes
    state.save_annotation(
        {
            "image_path": "tile.png",
            "skip": True,
            "unusable_reason": "excessive_water",
            "notes": "",
        }
    )
    df2 = __import__("pandas").read_csv(tmp_path / "raw" / "labels_human.csv")
    assert list(df2.columns) == DEFAULT_COLUMNS
    assert df2.iloc[0]["notes"] == "[unusable: excessive_water]"

    progress_file.unlink()
    res_empty_notes = state.get_annotation("tile.png")
    assert res_empty_notes["unusable_reason"] == "excessive_water"
    assert res_empty_notes["record"]["notes"] == ""


def test_conflicting_csv_and_sidecar_unusable_reason_precedence(tmp_path: Path) -> None:
    import pandas as pd

    state = fixture_state(tmp_path)
    state.load_batch({})

    # 1. Save annotation initially with excessive_water in progress sidecar
    state.save_annotation(
        {
            "image_path": "tile.png",
            "skip": True,
            "unusable_reason": "excessive_water",
            "notes": "initial note",
        }
    )

    csv_path = tmp_path / "raw" / "labels_human.csv"

    # 2. Conflicting CSV notes reason vs sidecar reason: CSV reason must win
    df = pd.read_csv(csv_path)
    df.loc[df["image_path"] == "tile.png", "notes"] = (
        "[unusable: missing_imagery] csv specific note"
    )
    df.to_csv(csv_path, index=False)

    res_conflict = state.get_annotation("tile.png")
    assert res_conflict["found"] is True
    assert res_conflict["unusable_reason"] == "missing_imagery"
    assert res_conflict["record"]["notes"] == "csv specific note"

    # 3. Legacy record lacking encoded CSV reason: sidecar fallback must still work
    df.loc[df["image_path"] == "tile.png", "notes"] = (
        "legacy note without unusable prefix"
    )
    df.to_csv(csv_path, index=False)

    res_legacy = state.get_annotation("tile.png")
    assert res_legacy["found"] is True
    assert res_legacy["unusable_reason"] == "excessive_water"
    assert res_legacy["record"]["notes"] == "legacy note without unusable prefix"


def test_ui_error_and_retry_markers() -> None:
    html = Path("scripts/annotation/annotate_scenic_web.html").read_text()
    assert "retryImage" in html
    assert "Retry image" in html
    assert "showImageError" in html
    assert "clearImageError" in html
    assert "onerror" in html
    assert "image.onerror" in html or "onerror=()=>{}" in html
    assert "Failed to load image" in html
    assert "loadTileImage" in html
    assert "dataset.navGen" in html
    assert "dataset.imagePath" in html
    assert "gen!==navGen" in html or "gen !== navGen" in html
    assert "image_path!==path" in html or "image_path !== path" in html
    assert "const loader=new Image()" in html
    assert "requestId!==imageRequestId" in html
    assert "tileImg.onload" not in html
    assert "tileImg.onerror" not in html
    assert "tileImg.removeAttribute('src')" in html
    assert "tileImg.style.visibility='hidden'" in html
    assert "tileImg.style.visibility='visible'" in html
    assert html.count("<script>") == html.count("</script>") == 1
    assert html.index("<script>") < html.index("</script>") < html.index("</body>")
