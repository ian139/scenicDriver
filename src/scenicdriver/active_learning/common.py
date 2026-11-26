"""Small, dependency-light helpers shared by active-learning CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
import re


def validate_run_name(name: Any) -> str:
    """Validate that *name* is a valid safe single-path-component run slug."""
    if not isinstance(name, str):
        raise ValueError("run_name must be a string")
    name = name.strip()
    if not name:
        raise ValueError("run_name cannot be empty")
    if os.path.isabs(name) or name.startswith("/") or name.startswith("\\"):
        raise ValueError(f"run_name cannot be an absolute path: {name!r}")
    if (
        "/" in name
        or "\\" in name
        or os.sep in name
        or (os.altsep and os.altsep in name)
    ):
        raise ValueError(f"run_name cannot contain path separators: {name!r}")
    if name in (".", "..") or ".." in name:
        raise ValueError(
            f"run_name cannot be '.' or '..' or contain traversal: {name!r}"
        )
    if not re.match(r"^[a-zA-Z0-9._-]+$", name):
        raise ValueError(f"run_name must be a valid slug: {name!r}")
    return name


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    """Serialize JSON with a canonical ordering suitable for hashing."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = stable_json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, payload: str) -> None:
    """Atomically replace a UTF-8 text file in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def jsonable(value: Any) -> Any:
    """Convert common NumPy/Pandas scalar values into JSON-safe values."""

    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)
