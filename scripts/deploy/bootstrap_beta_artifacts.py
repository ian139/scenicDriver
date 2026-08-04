#!/usr/bin/env python3
"""Bootstrap the ignored beta deployment artifacts from S3.

The manifest deliberately keeps the destination path and S3-relative ``source_key``
separate.  The object key is ``<prefix>/<source_key>`` (with a single slash),
where ``prefix`` comes from ``--s3-prefix``/``SCENIC_S3_PREFIX``.  Gzip sources
are decompressed into a temporary file before the final digest and size check.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import boto3

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "deploy" / "beta_artifacts.json"


class ArtifactBootstrapError(RuntimeError):
    """Raised when an artifact cannot be safely bootstrapped or verified."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    destination: str
    source_key: str
    sha256: str
    size_bytes: int
    compression: str | None = None


def _relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactBootstrapError(f"manifest {field} must be a non-empty relative path")
    raw = value.strip()
    path = Path(raw)
    # PureWindowsPath catches drive-qualified and UNC paths even on POSIX runners.
    if path.is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ArtifactBootstrapError(f"manifest {field} must be relative: {raw!r}")
    parts = raw.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactBootstrapError(f"manifest {field} contains unsafe path components: {raw!r}")
    return "/".join(parts)


def load_manifest(path: Path) -> list[ArtifactSpec]:
    """Load and validate artifact specifications from *path*."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactBootstrapError(f"could not read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        raise ArtifactBootstrapError("manifest must contain an 'artifacts' list")

    specs: list[ArtifactSpec] = []
    destinations: set[str] = set()
    names: set[str] = set()
    for index, item in enumerate(payload["artifacts"]):
        if not isinstance(item, dict):
            raise ArtifactBootstrapError(f"manifest artifact {index} must be an object")
        try:
            name = item["name"]
            destination = item.get("destination", item.get("path"))
            source_key = item.get("source_key", item.get("key"))
            sha256 = item["sha256"]
            size_bytes = item.get("size_bytes", item.get("size"))
            compression = item.get("compression")
        except KeyError as exc:
            raise ArtifactBootstrapError(f"manifest artifact {index} missing {exc.args[0]!r}") from exc
        if not isinstance(name, str) or not name.strip():
            raise ArtifactBootstrapError(f"manifest artifact {index} has an invalid name")
        name = name.strip()
        if name in names:
            raise ArtifactBootstrapError(f"duplicate artifact name: {name}")
        destination = _relative_path(destination, field=f"artifact {name} destination")
        source_key = _relative_path(source_key, field=f"artifact {name} source_key")
        if destination in destinations:
            raise ArtifactBootstrapError(f"duplicate artifact destination: {destination}")
        if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha256):
            raise ArtifactBootstrapError(f"artifact {name} sha256 must be a 64-character hexadecimal digest")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ArtifactBootstrapError(f"artifact {name} size_bytes must be a non-negative integer")
        if compression not in (None, "gzip"):
            raise ArtifactBootstrapError(f"artifact {name} compression must be null or 'gzip'")
        specs.append(ArtifactSpec(name, destination, source_key, sha256.lower(), size_bytes, compression))
        names.add(name)
        destinations.add(destination)
    if not specs:
        raise ArtifactBootstrapError("manifest contains no artifacts")
    return specs


def _safe_destination(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ArtifactBootstrapError(f"artifact destination escapes project root: {relative!r}") from exc
    return destination


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _matches(path: Path, spec: ArtifactSpec) -> bool:
    if not path.is_file():
        return False
    try:
        size, digest = _file_digest(path)
    except OSError as exc:
        raise ArtifactBootstrapError(f"could not read artifact {path}: {exc}") from exc
    return size == spec.size_bytes and digest == spec.sha256


def _s3_key(prefix: str, source_key: str) -> str:
    return f"{prefix.strip('/')}/{source_key}" if prefix.strip("/") else source_key


def _download_with_boto3(client: Any, bucket: str, key: str, destination: Path) -> None:
    try:
        client.download_file(bucket, key, str(destination))
    except Exception as exc:
        raise ArtifactBootstrapError(f"boto3 failed downloading s3://{bucket}/{key}: {exc}") from exc


def _download(bucket: str, key: str, destination: Path, client: Any | None) -> None:
    if client is None:
        try:
            client = boto3.client("s3")
        except Exception as exc:
            raise ArtifactBootstrapError(f"boto3 failed downloading s3://{bucket}/{key}: {exc}") from exc
    _download_with_boto3(client, bucket, key, destination)


def bootstrap(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    project_root: Path | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    check_only: bool = False,
    s3_client: Any | None = None,
) -> tuple[str, ...]:
    """Verify and, unless checking only, download all manifest artifacts.

    Existing files with the expected digest and size are left untouched.  Downloads
    are written beside the destination and atomically renamed only after verification.
    """
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    specs = load_manifest(manifest_path)
    missing_or_mismatched: list[tuple[ArtifactSpec, Path]] = []
    statuses: list[str] = []
    for spec in specs:
        destination = _safe_destination(root, spec.destination)
        if _matches(destination, spec):
            statuses.append(f"ok: {spec.name}")
        else:
            missing_or_mismatched.append((spec, destination))

    if check_only:
        if missing_or_mismatched:
            details = ", ".join(spec.name for spec, _ in missing_or_mismatched)
            raise ArtifactBootstrapError(f"artifact check failed (missing or mismatched): {details}")
        return tuple(statuses)
    if missing_or_mismatched and not (bucket and bucket.strip()):
        raise ArtifactBootstrapError("--s3-bucket/SCENIC_S3_BUCKET is required when artifacts need downloading")
    if prefix is None:
        prefix = ""

    for spec, destination in missing_or_mismatched:
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_name: str | None = None
        materialized_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as stream:
                download_path = Path(stream.name)
            download_name = str(download_path)
            _download(bucket.strip(), _s3_key(prefix, spec.source_key), download_path, s3_client)

            candidate = download_path
            if spec.compression == "gzip":
                with tempfile.NamedTemporaryFile(
                    prefix=f".{destination.name}.", dir=destination.parent, delete=False
                ) as stream:
                    materialized_path = Path(stream.name)
                materialized_name = str(materialized_path)
                try:
                    with gzip.open(download_path, "rb") as source, materialized_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                except (OSError, gzip.BadGzipFile) as exc:
                    raise ArtifactBootstrapError(f"downloaded artifact {spec.name} is not valid gzip: {exc}") from exc
                candidate = materialized_path

            if not _matches(candidate, spec):
                size, digest = _file_digest(candidate)
                raise ArtifactBootstrapError(
                    f"downloaded artifact {spec.name} does not match manifest "
                    f"(expected {spec.size_bytes} bytes/{spec.sha256}, got {size} bytes/{digest})"
                )
            os.replace(candidate, destination)
            statuses.append(f"downloaded: {spec.name}")
        finally:
            if download_name:
                Path(download_name).unlink(missing_ok=True)
            if materialized_name:
                Path(materialized_name).unlink(missing_ok=True)
    return tuple(statuses)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--s3-bucket", dest="bucket", default=None)
    parser.add_argument("--s3-prefix", dest="prefix", default=None)
    parser.add_argument("--check-only", action="store_true", help="verify local artifacts without contacting S3")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bucket = args.bucket or os.environ.get("SCENIC_S3_BUCKET")
    if args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = os.environ.get("SCENIC_S3_PREFIX") or ""
    manifest = args.manifest
    if not manifest.is_absolute():
        root = (args.project_root or Path(__file__).resolve().parents[2]).resolve()
        manifest = root / manifest
    try:
        statuses = bootstrap(
            manifest,
            project_root=args.project_root,
            bucket=bucket,
            prefix=prefix,
            check_only=args.check_only,
        )
    except ArtifactBootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for status in statuses:
        print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
