from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.deploy.bootstrap_beta_artifacts import ArtifactBootstrapError, bootstrap, load_manifest


def _manifest(
    path: Path,
    *,
    destination: str = "runtime/artifact.bin",
    source_key: str = "objects/artifact.bin",
    payload: bytes = b"artifact",
    compression: str | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "name": "artifact",
                        "destination": destination,
                        "source_key": source_key,
                        "compression": compression,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class FakeS3:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.calls.append((bucket, key, filename))
        Path(filename).write_bytes(self.payload)


def test_check_only_accepts_matching_file_without_s3_configuration(tmp_path: Path) -> None:
    payload = b"already bootstrapped"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, payload=payload)
    destination = tmp_path / "runtime/artifact.bin"
    destination.parent.mkdir()
    destination.write_bytes(payload)

    statuses = bootstrap(manifest, project_root=tmp_path, check_only=True)

    assert statuses == ("ok: artifact",)

def test_matching_artifact_is_not_downloaded_again(tmp_path: Path) -> None:
    payload = b"already present"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, payload=payload)
    destination = tmp_path / "runtime/artifact.bin"
    destination.parent.mkdir()
    destination.write_bytes(payload)
    client = FakeS3(b"would replace it")

    statuses = bootstrap(
        manifest,
        project_root=tmp_path,
        bucket="example-bucket",
        s3_client=client,
    )

    assert statuses == ("ok: artifact",)
    assert client.calls == []
    assert destination.read_bytes() == payload


def test_missing_artifact_downloads_from_prefix_and_verifies(tmp_path: Path) -> None:
    payload = b"download me"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, payload=payload)
    client = FakeS3(payload)

    statuses = bootstrap(
        manifest,
        project_root=tmp_path,
        bucket="example-bucket",
        prefix="beta/v1/",
        s3_client=client,
    )

    assert statuses == ("downloaded: artifact",)
    assert (tmp_path / "runtime/artifact.bin").read_bytes() == payload
    assert client.calls[0][:2] == ("example-bucket", "beta/v1/objects/artifact.bin")

def test_gzip_download_is_decompressed_before_verification(tmp_path: Path) -> None:
    payload = b"compressed artifact" * 100
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, payload=payload, source_key="objects/artifact.bin.gz", compression="gzip")
    client = FakeS3(gzip.compress(payload))

    statuses = bootstrap(
        manifest,
        project_root=tmp_path,
        bucket="example-bucket",
        s3_client=client,
    )

    assert statuses == ("downloaded: artifact",)
    assert (tmp_path / "runtime/artifact.bin").read_bytes() == payload
    assert client.calls[0][:2] == ("example-bucket", "objects/artifact.bin.gz")


def test_mismatched_download_is_refused_and_existing_file_is_preserved(tmp_path: Path) -> None:
    expected = b"expected"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, payload=expected)
    destination = tmp_path / "runtime/artifact.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"local version")

    with pytest.raises(ArtifactBootstrapError, match="does not match manifest"):
        bootstrap(
            manifest,
            project_root=tmp_path,
            bucket="example-bucket",
            s3_client=FakeS3(b"wrong remote bytes"),
        )

    assert destination.read_bytes() == b"local version"


def test_manifest_rejects_project_root_escape(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, destination="../outside.bin")

    with pytest.raises(ArtifactBootstrapError, match="unsafe path components"):
        load_manifest(manifest)


def test_check_only_reports_missing_or_mismatched_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)

    with pytest.raises(ArtifactBootstrapError, match="missing or mismatched"):
        bootstrap(manifest, project_root=tmp_path, check_only=True)

def test_plain_checksum_manifest_matches_json_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    specs = load_manifest(root / "deploy/beta_artifacts.json")
    expected = {spec.destination: spec.sha256 for spec in specs}
    actual: dict[str, str] = {}
    for line in (root / "deploy/beta_artifacts.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, destination = line.split(maxsplit=1)
        actual[destination] = digest

    assert actual == expected
