"""S3 transfer helpers for Scenic Drive training artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str



def normalize_s3_only(value: str | bool | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid SCENIC_S3_ONLY value: {value!r}")




def _client(client: Any | None) -> Any:
    if client is not None:
        return client
    import boto3

    return boto3.client("s3")


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _missing(bucket: str, key: str, *, required: bool) -> bool:
    uri = _s3_uri(bucket, key)
    if required:
        raise FileNotFoundError(f"S3 artifact not found: {uri}")
    print(f"optional S3 artifact missing: {uri}")
    return False


def check_prefix(bucket: str, prefix: str, *, max_keys: int = 1, client: Any | None = None) -> bool:
    s3 = _client(client)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    return bool(response.get("Contents"))


def download_file(
    bucket: str,
    key: str,
    dest: Path,
    *,
    client: Any | None = None,
    required: bool = True,
) -> bool:
    s3 = _client(client)
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "get", lambda *_: {}) ("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return _missing(bucket, key, required=required)
        if required:
            raise
        return _missing(bucket, key, required=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(dest))
    return True


def _iter_objects(s3: Any, bucket: str, prefix: str) -> Any:
    paginator_factory = getattr(s3, "get_paginator", None)
    if paginator_factory is not None:
        paginator = paginator_factory("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            yield from page.get("Contents", [])
        return
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    yield from response.get("Contents", [])


def download_prefix(
    bucket: str,
    prefix: str,
    dest: Path,
    *,
    client: Any | None = None,
    required: bool = True,
) -> int:
    s3 = _client(client)
    objects = [obj for obj in _iter_objects(s3, bucket, prefix) if not obj.get("Key", "").endswith("/")]
    if not objects:
        _missing(bucket, prefix, required=required)
        return 0
    count = 0
    for obj in objects:
        key = obj["Key"]
        relative = key[len(prefix) :].lstrip("/") if key.startswith(prefix) else Path(key).name
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))
        count += 1
    return count


def upload_prefix(
    src: Path,
    bucket: str,
    prefix: str,
    *,
    client: Any | None = None,
    required: bool = True,
) -> int:
    if not src.exists():
        uri = _s3_uri(bucket, prefix)
        if required:
            raise FileNotFoundError(f"S3 artifact not found: {uri}")
        print(f"optional S3 artifact missing: {uri}")
        return 0
    s3 = _client(client)
    files = [src] if src.is_file() else [path for path in src.rglob("*") if path.is_file()]
    if not files:
        if required:
            raise FileNotFoundError(f"S3 artifact not found: {_s3_uri(bucket, prefix)}")
        print(f"optional S3 artifact missing: {_s3_uri(bucket, prefix)}")
        return 0
    normalized_prefix = prefix.strip("/")
    count = 0
    for path in files:
        relative = path.name if src.is_file() else path.relative_to(src).as_posix()
        key = f"{normalized_prefix}/{relative}" if normalized_prefix else relative
        s3.upload_file(str(path), bucket, key)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer Scenic Drive S3 artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-prefix", help="Check whether an S3 prefix has objects")
    check.add_argument("--bucket", required=True)
    check.add_argument("--prefix", required=True)
    check.add_argument("--required", action="store_true", default=False)
    check.add_argument("--optional", action="store_true", default=False)

    download_one = subparsers.add_parser("download-file", help="Download one S3 object")
    download_one.add_argument("--bucket", required=True)
    download_one.add_argument("--key", required=True)
    download_one.add_argument("--dest", type=Path, required=True)
    download_one.add_argument("--required", action="store_true", default=False)
    download_one.add_argument("--optional", action="store_true", default=False)

    download_many = subparsers.add_parser("download-prefix", help="Download all objects under a prefix")
    download_many.add_argument("--bucket", required=True)
    download_many.add_argument("--prefix", required=True)
    download_many.add_argument("--dest", type=Path, required=True)
    download_many.add_argument("--required", action="store_true", default=False)
    download_many.add_argument("--optional", action="store_true", default=False)

    upload = subparsers.add_parser("upload-prefix", help="Upload a local file or directory to a prefix")
    upload.add_argument("--src", type=Path, required=True)
    upload.add_argument("--bucket", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--required", action="store_true", default=False)
    upload.add_argument("--optional", action="store_true", default=False)
    return parser.parse_args()


def _required(args: argparse.Namespace) -> bool:
    if args.optional:
        return False
    if args.required:
        return True
    return True


def main() -> None:
    args = parse_args()
    required = _required(args)
    if args.command == "check-prefix":
        found = check_prefix(args.bucket, args.prefix)
        if not found:
            _missing(args.bucket, args.prefix, required=required)
        return
    if args.command == "download-file":
        download_file(args.bucket, args.key, args.dest, required=required)
        return
    if args.command == "download-prefix":
        download_prefix(args.bucket, args.prefix, args.dest, required=required)
        return
    if args.command == "upload-prefix":
        upload_prefix(args.src, args.bucket, args.prefix, required=required)
        return
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
