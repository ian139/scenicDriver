"""Safe, dependency-light localhost web UI for absolute scenic annotation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import parse_qs, urlparse

import pandas as pd

try:  # fcntl is available on the supported macOS/Linux annotation hosts.
    import fcntl
except ImportError:  # pragma: no cover - Windows gets process-local serialization.
    fcntl = None


DEFAULT_COLUMNS = [
    "image_path",
    "scenic_human",
    "confidence",
    "skip",
    "annotator_id",
    "timestamp",
    "notes",
]
SCHEMA_VERSION = 1
MAX_JSON_BYTES = 1_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = Path(__file__).with_name("annotate_scenic_web.html")
HTML_PAGE = TEMPLATE_PATH.read_text(encoding="utf-8")
SECRET_COLUMN = re.compile(
    r"(^|_)(secret|token|password|passwd|credential|authorization|cookie|api_?key|access_?key)($|_)",
    re.IGNORECASE,
)
PUBLIC_BATCH_COLUMNS = {
    "image_path",
    "satellite_path",
    "terrain_path",
    "region",
    "selection_reason",
    "selection_rank",
    "rank",
    "batch_id",
    "run_id",
    "z",
    "x",
    "y",
    "lat",
    "lon",
    "class_id",
    "prediction",
    "model_prediction",
    "heuristic_score",
    "uncertainty",
    "disagreement_score",
    "diversity_score",
    "geographic_score",
    "underrepresentation_score",
    "score_bin",
    "is_repeat",
    "qa_repeat",
}
UNUSABLE_REASONS = {
    "missing_imagery",
    "corrupted_image",
    "cloud_or_obstruction",
    "excessive_water",
    "duplicate",
    "other",
}
CONFIDENCE_VALUES = {"low", "medium", "high"}


class ApiError(Exception):
    """Expected request failure safe to return to a browser."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class AnnotatorConfig:
    labels_csv: str = "data/raw/labels.csv"
    batch_csv: str = ""
    raw_dir: str = "data/raw"
    annotations_csv: str = "data/raw/labels_human.csv"
    sample_size: int = 500
    seed: int = 42
    stratify_by_class: bool = True
    annotator_id: str = os.getenv("USER", "annotator")


class PathPolicy:
    """Canonicalize browser-configurable paths beneath explicitly approved roots."""

    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        *,
        local_roots: Sequence[Path] | None = None,
        s3_roots: Sequence[str] = (),
    ) -> None:
        self.project_root = project_root.resolve()
        roots = local_roots or (
            self.project_root / "data" / "raw",
            self.project_root / "data" / "processed",
        )
        self.local_roots = tuple(Path(root).resolve() for root in roots)
        self.s3_roots = tuple(
            self._normalize_s3_root(root) for root in s3_roots if root
        )

    @staticmethod
    def _normalize_s3_root(value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "approved S3 roots must be credential-free s3://bucket/prefix URIs"
            )
        if (
            parsed.query
            or parsed.fragment
            or any(part == ".." for part in parsed.path.split("/"))
        ):
            raise ValueError("approved S3 root is invalid")
        return f"s3://{parsed.netloc}/{parsed.path.strip('/')}".rstrip("/")

    def resolve_local(self, value: str, *, kind: str, must_exist: bool = False) -> Path:
        raw = os.path.expandvars(str(value).strip())
        if not raw:
            raise ApiError(400, "invalid_path", f"{kind} is required")
        if urlparse(raw).scheme:
            raise ApiError(400, "invalid_path", f"{kind} must be a project-local path")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve(strict=False)
        if not any(
            resolved == root or root in resolved.parents for root in self.local_roots
        ):
            raise ApiError(
                403,
                "path_not_approved",
                f"{kind} must stay under an approved data root",
            )
        if must_exist and not resolved.exists():
            raise ApiError(404, "path_not_found", f"{kind} was not found")
        return resolved

    def resolve_raw_root(self, value: str) -> str:
        raw = os.path.expandvars(str(value).strip())
        if raw.startswith("s3://"):
            normalized = self._normalize_s3_root(raw)
            if not any(
                normalized == root or normalized.startswith(f"{root}/")
                for root in self.s3_roots
            ):
                raise ApiError(
                    403,
                    "s3_root_not_approved",
                    "raw_dir is outside the approved S3 roots",
                )
            return normalized
        return str(self.resolve_local(raw, kind="raw_dir", must_exist=True))

    def display(self, value: str | Path) -> str:
        text = str(value)
        if text.startswith("s3://"):
            return text
        path = Path(text).resolve(strict=False)
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.name


@contextmanager
def _locked_path(path: Path, process_lock: threading.RLock) -> Iterator[None]:
    """Serialize threads and cooperating processes using a stable sidecar lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with process_lock:
        with lock_path.open("a+b") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _read_annotations(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    frame = pd.read_csv(path)
    for column in DEFAULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[DEFAULT_COLUMNS].copy()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    rows = frame.where(pd.notna(frame), "").to_dict(orient="records")
    with tempfile.SpooledTemporaryFile(
        mode="w+", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=DEFAULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            (
                {column: row.get(column, "") for column in DEFAULT_COLUMNS}
                for row in rows
            )
        )
        stream.seek(0)
        return stream.read().encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"schema_version": SCHEMA_VERSION, "batches": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(
            500, "progress_store_invalid", "The progress store is unreadable"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ApiError(
            500, "progress_store_invalid", "The progress store schema is unsupported"
        )
    value.setdefault("batches", {})
    return value


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "t"}


def _sample_stratified(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if "class_id" not in frame.columns or frame["class_id"].isna().all():
        return frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(
            drop=True
        )
    groups = sorted(frame["class_id"].dropna().unique().tolist(), key=str)
    chunks: list[pd.DataFrame] = []
    remaining = min(n, len(frame))
    for index, class_id in enumerate(groups):
        group = frame[frame["class_id"] == class_id]
        take = min(len(group), max(1, remaining // max(1, len(groups) - index)))
        chunks.append(group.sample(n=take, random_state=seed))
        remaining -= take
    sampled = pd.concat(chunks, ignore_index=True) if chunks else frame.iloc[0:0].copy()
    needed = min(n, len(frame)) - len(sampled)
    if needed > 0:
        remainder = frame.loc[~frame["image_path"].isin(sampled["image_path"])]
        sampled = pd.concat(
            [
                sampled,
                remainder.sample(n=min(needed, len(remainder)), random_state=seed),
            ],
            ignore_index=True,
        )
    return sampled.drop_duplicates(subset=["image_path"]).reset_index(drop=True)


def _safe_image_path(image_path: str) -> str:
    value = str(image_path).strip().replace("\\", "/")
    parsed = urlparse(value)
    parts = value.split("/")
    if (
        not value
        or parsed.scheme
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ApiError(
            400, "invalid_image_path", "image_path must be a relative canonical path"
        )
    return value


def _parse_s3(raw_dir: str) -> tuple[str, str] | None:
    if not raw_dir.startswith("s3://"):
        return None
    parsed = urlparse(raw_dir)
    return parsed.netloc, parsed.path.strip("/")


def _serve_image_bytes(
    raw_dir: str, image_path: str, *, s3_client: Any | None
) -> tuple[bytes, str]:
    safe_path = _safe_image_path(image_path)
    s3_info = _parse_s3(raw_dir)
    if s3_info is not None:
        if s3_client is None:
            raise ApiError(503, "s3_unavailable", "S3 image access is not configured")
        bucket, prefix = s3_info
        key = f"{prefix}/{safe_path}" if prefix else safe_path
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
        except Exception as exc:
            raise ApiError(
                404, "image_not_found", "The requested image could not be loaded"
            ) from exc
        content_type = (
            obj.get("ContentType")
            or mimetypes.guess_type(safe_path)[0]
            or "application/octet-stream"
        )
        return body, content_type

    root = Path(raw_dir).resolve()
    path = (root / safe_path).resolve()
    if root not in path.parents:
        raise ApiError(403, "path_traversal", "The requested image is outside raw_dir")
    try:
        body = path.read_bytes()
    except FileNotFoundError as exc:
        raise ApiError(
            404, "image_not_found", "The requested image was not found"
        ) from exc
    return body, mimetypes.guess_type(path.name)[0] or "application/octet-stream"


class AnnotatorState:
    def __init__(
        self, config: AnnotatorConfig, *, path_policy: PathPolicy | None = None
    ):
        self._lock = threading.RLock()
        self.path_policy = path_policy or PathPolicy()
        self.config = self._validated_config(config, browser_update=False)
        self.batch: list[dict[str, Any]] = []
        self.batch_id = ""
        self.batch_source = ""
        self.started_at = time.monotonic()

    @property
    def progress_path(self) -> Path:
        annotations = Path(self.config.annotations_csv)
        return annotations.with_name(f"{annotations.stem}.annotation_progress.json")

    def _validated_config(
        self, config: AnnotatorConfig, *, browser_update: bool
    ) -> AnnotatorConfig:
        if browser_update and config.annotator_id != self.config.annotator_id:
            raise ApiError(
                403, "identity_locked", "Annotator identity is fixed by the server"
            )
        if config.sample_size < 1 or config.sample_size > 100_000:
            raise ApiError(
                400, "invalid_sample_size", "sample_size must be between 1 and 100000"
            )
        labels = self.path_policy.resolve_local(
            config.labels_csv, kind="labels_csv", must_exist=False
        )
        batch = ""
        if config.batch_csv.strip():
            batch = str(
                self.path_policy.resolve_local(
                    config.batch_csv, kind="batch_csv", must_exist=True
                )
            )
        annotations = self.path_policy.resolve_local(
            config.annotations_csv, kind="annotations_csv", must_exist=False
        )
        if annotations.suffix.lower() != ".csv":
            raise ApiError(
                400, "invalid_output_path", "annotations_csv must end in .csv"
            )
        raw_dir = self.path_policy.resolve_raw_root(config.raw_dir)
        return replace(
            config,
            labels_csv=str(labels),
            batch_csv=batch,
            raw_dir=raw_dir,
            annotations_csv=str(annotations),
            annotator_id=str(config.annotator_id).strip() or "annotator",
        )

    def public_config(self) -> dict[str, Any]:
        with self._lock:
            config = self.config
        return {
            "labels_csv": self.path_policy.display(config.labels_csv),
            "batch_csv": self.path_policy.display(config.batch_csv)
            if config.batch_csv
            else "",
            "raw_dir": self.path_policy.display(config.raw_dir),
            "annotations_csv": self.path_policy.display(config.annotations_csv),
            "sample_size": config.sample_size,
            "seed": config.seed,
            "stratify_by_class": config.stratify_by_class,
            "annotator_id": config.annotator_id,
        }

    def _config_from_browser(self, values: dict[str, Any]) -> AnnotatorConfig:
        if not isinstance(values, dict):
            raise ApiError(400, "invalid_config", "config must be an object")
        with self._lock:
            current = self.config
        allowed = {
            "labels_csv",
            "batch_csv",
            "raw_dir",
            "annotations_csv",
            "sample_size",
            "seed",
            "stratify_by_class",
            "annotator_id",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ApiError(400, "invalid_config", "config contains unsupported fields")
        try:
            updated = replace(
                current,
                labels_csv=str(values.get("labels_csv", current.labels_csv)),
                batch_csv=str(values.get("batch_csv", current.batch_csv)),
                raw_dir=str(values.get("raw_dir", current.raw_dir)),
                annotations_csv=str(
                    values.get("annotations_csv", current.annotations_csv)
                ),
                sample_size=int(values.get("sample_size", current.sample_size)),
                seed=int(values.get("seed", current.seed)),
                stratify_by_class=_to_bool(
                    values.get("stratify_by_class", current.stratify_by_class)
                ),
                annotator_id=str(values.get("annotator_id", current.annotator_id)),
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(
                400, "invalid_config", "sample_size and seed must be integers"
            ) from exc
        return self._validated_config(updated, browser_update=True)

    @staticmethod
    def _reject_secret_columns(frame: pd.DataFrame) -> None:
        blocked = sorted(
            column for column in frame.columns if SECRET_COLUMN.search(str(column))
        )
        if blocked:
            raise ApiError(
                400,
                "sensitive_metadata",
                "The batch contains secret-like metadata columns",
            )

    @staticmethod
    def _public_batch(frame: pd.DataFrame) -> list[dict[str, Any]]:
        columns = [column for column in frame.columns if column in PUBLIC_BATCH_COLUMNS]
        return _json_ready(frame[columns].to_dict(orient="records"))

    @staticmethod
    def _source_id(frame: pd.DataFrame, config: AnnotatorConfig, source: str) -> str:
        canonical = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "paths": frame["image_path"].astype(str).tolist(),
            "sample_size": config.sample_size,
            "seed": config.seed,
            "stratify_by_class": config.stratify_by_class,
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"batch-{digest[:16]}"

    def _read_batch_progress(
        self,
        batch_id: str,
        *,
        progress_path: Path | None = None,
        annotator_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            path = progress_path if progress_path is not None else self.progress_path
            identity = annotator_id if annotator_id is not None else self.config.annotator_id
        with _locked_path(path, self._lock):
            store = _read_json(path)
            value = store["batches"].get(batch_id, {}).get(identity, {})
            return dict(value) if isinstance(value, dict) else {}

    def _update_progress(
        self,
        changes: dict[str, Any],
        *,
        batch_id: str | None = None,
        progress_path: Path | None = None,
        annotator_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            bid = batch_id if batch_id is not None else self.batch_id
            identity = annotator_id if annotator_id is not None else self.config.annotator_id
            path = progress_path if progress_path is not None else self.progress_path
        if not bid:
            raise ApiError(409, "batch_not_loaded", "Load a batch before updating progress")
        with _locked_path(path, self._lock):
            store = _read_json(path)
            batch_state = store["batches"].setdefault(bid, {})
            state = batch_state.setdefault(identity, {})
            state.update(changes)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write(path, _json_bytes(store))
            return dict(state)

    def load_batch(self, config_data: dict[str, Any]) -> dict[str, Any]:
        config = self._config_from_browser(config_data)
        labels = pd.DataFrame(columns=["image_path"])
        labels_path = Path(config.labels_csv)
        if labels_path.exists():
            labels = pd.read_csv(labels_path)
            self._reject_secret_columns(labels)
            if "image_path" not in labels.columns:
                raise ApiError(
                    400, "invalid_labels", "labels_csv must contain image_path"
                )
            labels = labels.dropna(subset=["image_path"]).copy()
            labels["image_path"] = labels["image_path"].map(_safe_image_path)

        source = "labels_csv"
        candidates = labels
        if config.batch_csv:
            candidates = pd.read_csv(config.batch_csv)
            self._reject_secret_columns(candidates)
            if candidates.empty:
                raise ApiError(400, "empty_batch", "batch_csv is empty")
            if "image_path" not in candidates.columns:
                raise ApiError(
                    400, "invalid_batch", "batch_csv must contain image_path"
                )
            candidates = candidates.dropna(subset=["image_path"]).copy()
            candidates["image_path"] = candidates["image_path"].map(_safe_image_path)
            candidates = candidates.drop_duplicates(subset=["image_path"], keep="first")
            source = "batch_csv"
        elif candidates.empty:
            if labels_path.exists():
                raise ApiError(400, "empty_labels", "labels_csv is empty")
            raise ApiError(404, "labels_not_found", "labels_csv was not found")

        batch_id = self._source_id(candidates, config, source)
        with self._lock:
            self.config = config
            self.batch_id = batch_id
            self.batch_source = source
        saved_progress = self._read_batch_progress(batch_id)
        saved_order = saved_progress.get("image_paths")
        if isinstance(saved_order, list):
            by_path = candidates.set_index("image_path", drop=False)
            ordered = [str(path) for path in saved_order if str(path) in by_path.index]
            if source == "batch_csv":
                seen = set(ordered)
                ordered.extend(
                    str(path) for path in candidates["image_path"] if str(path) not in seen
                )
            batch_frame = (
                by_path.loc[ordered].reset_index(drop=True)
                if ordered
                else candidates.iloc[0:0].copy()
            )
        elif source == "batch_csv":
            batch_frame = candidates.copy().reset_index(drop=True)
        elif config.stratify_by_class:
            batch_frame = _sample_stratified(
                candidates, config.sample_size, config.seed
            )
        else:
            batch_frame = candidates.sample(
                n=min(config.sample_size, len(candidates)), random_state=config.seed
            ).reset_index(drop=True)

        public_batch = self._public_batch(batch_frame)
        with self._lock:
            self.batch = public_batch
        if not saved_order:
            saved_progress = self._update_progress(
                {
                    "cursor": 0,
                    "image_paths": [row["image_path"] for row in public_batch],
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        counts = self._annotation_counts()
        cursor = min(
            max(int(saved_progress.get("cursor", 0)), 0), max(0, len(public_batch) - 1)
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.public_config(),
            "batch_id": batch_id,
            "batch_source": source,
            "batch_size": len(public_batch),
            "candidate_pool": len(candidates),
            "batch": public_batch,
            "progress": {"cursor": cursor, **counts},
            "unusable_reasons": sorted(UNUSABLE_REASONS),
        }

    def _current_paths(self) -> set[str]:
        with self._lock:
            return {str(row["image_path"]) for row in self.batch}

    def _assert_batch_image(self, image_path: Any) -> str:
        safe = _safe_image_path(str(image_path))
        if safe not in self._current_paths():
            raise ApiError(
                403, "image_not_in_batch", "image_path is not in the active batch"
            )
        return safe

    def get_annotation(self, image_path: Any) -> dict[str, Any]:
        safe = self._assert_batch_image(image_path)
        with self._lock:
            config = self.config
        path = Path(config.annotations_csv)
        with _locked_path(path, self._lock):
            frame = _read_annotations(path)
        if frame.empty:
            return {"found": False, "image_path": safe}
        matches = frame.loc[
            (frame["image_path"].astype(str) == safe)
            & (frame["annotator_id"].astype(str) == config.annotator_id)
        ]
        if matches.empty:
            return {"found": False, "image_path": safe}
        record = _json_ready(matches.iloc[-1][DEFAULT_COLUMNS].to_dict())
        auxiliary = (
            self._read_batch_progress(self.batch_id).get("unusable", {}).get(safe)
        )
        return {"found": True, "record": record, "unusable_reason": auxiliary}

    def save_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ApiError(400, "invalid_payload", "Request body must be an object")
        with self._lock:
            config = self.config
            batch_id = self.batch_id
            batch_paths = {str(row["image_path"]) for row in self.batch}
            progress_path = self.progress_path
        if not batch_id or not batch_paths:
            raise ApiError(409, "batch_not_loaded", "Load a batch before updating progress")
        claimed_identity = payload.get("annotator_id")
        if claimed_identity is not None and str(claimed_identity) != config.annotator_id:
            raise ApiError(403, "identity_mismatch", "annotator_id is fixed by the server")
        image_path = _safe_image_path(str(payload.get("image_path", "")))
        if image_path not in batch_paths:
            raise ApiError(403, "image_not_in_batch", "image_path is not in the active batch")
        skip = _to_bool(payload.get("skip", False))
        score_value = payload.get("scenic_human")
        if score_value in (None, ""):
            if not skip:
                raise ApiError(400, "score_required", "scenic_human is required unless the image is skipped")
            score: float | str = ""
        else:
            try:
                score = float(score_value)
            except (TypeError, ValueError) as exc:
                raise ApiError(400, "invalid_score", "scenic_human must be a number") from exc
            if not 0 <= score <= 10:
                raise ApiError(400, "invalid_score", "scenic_human must be between 0 and 10")
        confidence = str(payload.get("confidence", "medium")).lower()
        if confidence not in CONFIDENCE_VALUES:
            raise ApiError(400, "invalid_confidence", "confidence must be low, medium, or high")
        notes = str(payload.get("notes", ""))
        if len(notes) > 4000:
            raise ApiError(400, "notes_too_long", "notes must not exceed 4000 characters")
        reason = payload.get("unusable_reason")
        if reason not in (None, "") and str(reason) not in UNUSABLE_REASONS:
            raise ApiError(400, "invalid_unusable_reason", "unusable_reason is unsupported")
        if reason and not skip:
            raise ApiError(400, "invalid_unusable_reason", "unusable_reason requires skip=true")
        record = {
            "image_path": image_path,
            "scenic_human": score,
            "confidence": confidence,
            "skip": skip,
            "annotator_id": config.annotator_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        annotations_path = Path(config.annotations_csv)
        with _locked_path(annotations_path, self._lock):
            frame = _read_annotations(annotations_path)
            if not frame.empty:
                matching = (frame["image_path"].astype(str) == image_path) & (
                    frame["annotator_id"].astype(str) == config.annotator_id
                )
                frame = frame.loc[~matching].copy()
            frame = pd.concat([frame, pd.DataFrame([record], columns=DEFAULT_COLUMNS)], ignore_index=True)
            _atomic_write(annotations_path, _csv_bytes(frame))
            row_count = len(frame)
        progress = self._read_batch_progress(
            batch_id, progress_path=progress_path, annotator_id=config.annotator_id
        )
        unusable = dict(progress.get("unusable", {}))
        if reason:
            unusable[image_path] = str(reason)
        else:
            unusable.pop(image_path, None)
        self._update_progress(
            {"unusable": unusable, "last_saved_image": image_path},
            batch_id=batch_id,
            progress_path=progress_path,
            annotator_id=config.annotator_id,
        )
        return {"saved": True, "row_count": row_count, "record": record, "progress": self._annotation_counts()}

    def save_progress(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            size = len(self.batch)
        try:
            cursor = int(payload.get("cursor", 0))
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "invalid_cursor", "cursor must be an integer") from exc
        if size and not 0 <= cursor < size:
            raise ApiError(400, "invalid_cursor", "cursor is outside the active batch")
        state = self._update_progress({"cursor": cursor})
        return {"saved": True, "cursor": int(state["cursor"])}

    def _annotation_counts(self) -> dict[str, Any]:
        with self._lock:
            config = self.config
            batch = [dict(row) for row in self.batch]
            paths = [str(row.get("image_path", "")) for row in batch]
        annotations_path = Path(config.annotations_csv)
        with _locked_path(annotations_path, self._lock):
            frame = _read_annotations(annotations_path)
        if frame.empty or not paths:
            own = frame.iloc[0:0]
        else:
            own = frame.loc[
                frame["image_path"].astype(str).isin(paths)
                & (frame["annotator_id"].astype(str) == config.annotator_id)
            ]
        completed_paths = (
            set(own["image_path"].astype(str)) if not own.empty else set()
        )
        completed = len(completed_paths)
        confidence = {name: 0 for name in sorted(CONFIDENCE_VALUES)}
        if not own.empty:
            confidence.update(
                {
                    str(key): int(value)
                    for key, value in own["confidence"].value_counts().items()
                }
            )
        skipped = int(own["skip"].map(_to_bool).sum()) if not own.empty else 0
        coverage: dict[str, dict[str, int]] = {}
        for row in batch:
            region = str(row.get("region") or "unknown")
            region_counts = coverage.setdefault(region, {"completed": 0, "total": 0})
            region_counts["total"] += 1
            region_counts["completed"] += int(
                str(row.get("image_path", "")) in completed_paths
            )
        overlap = frame.loc[
            frame["image_path"].astype(str).isin(paths)
            & ~frame["skip"].map(_to_bool)
        ].copy()
        overlap["scenic_human"] = pd.to_numeric(
            overlap["scenic_human"], errors="coerce"
        )
        overlap = overlap.dropna(subset=["scenic_human"])
        ranges = (
            overlap.groupby("image_path")["scenic_human"].agg(["count", "min", "max"])
            if not overlap.empty
            else pd.DataFrame()
        )
        ranges = ranges.loc[ranges["count"] >= 2] if not ranges.empty else ranges
        consistency = (
            float(((ranges["max"] - ranges["min"]) <= 1.0).mean())
            if not ranges.empty
            else None
        )
        return {
            "completed": completed,
            "remaining": max(0, len(paths) - completed),
            "skipped": skipped,
            "confidence": confidence,
            "coverage": coverage,
            "overlap_consistency": {
                "tiles_with_overlap": int(len(ranges)),
                "within_one_point_fraction": consistency,
                "definition": "fraction of multi-annotator tiles with score range <= 1",
            },
        }

    def summary(self) -> dict[str, Any]:
        counts = self._annotation_counts()
        elapsed = max(time.monotonic() - self.started_at, 0.001)
        return {
            "schema_version": SCHEMA_VERSION,
            "batch_id": self.batch_id or None,
            "batch_size": len(self._current_paths()),
            **counts,
            "session": {
                "elapsed_seconds": round(elapsed, 1),
                "throughput_per_hour": round(counts["completed"] * 3600 / elapsed, 1),
            },
        }

    def image_bytes(
        self, image_path: Any, *, s3_client: Any | None
    ) -> tuple[bytes, str]:
        safe = self._assert_batch_image(image_path)
        with self._lock:
            raw_dir = self.config.raw_dir
        return _serve_image_bytes(raw_dir, safe, s3_client=s3_client)


def make_handler(
    state: AnnotatorState, s3_client: Any | None = None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ScenicAnnotator/1"

        def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(
                _json_ready(value), allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, error: ApiError) -> None:
            payload: dict[str, Any] = {
                "error": {"code": error.code, "message": error.message}
            }
            if error.details:
                payload["error"]["details"] = error.details
            self._send_json(payload, error.status)

        def _read_json_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "0")
            except ValueError as exc:
                raise ApiError(
                    400, "invalid_content_length", "Content-Length must be an integer"
                ) from exc
            if length <= 0 or length > MAX_JSON_BYTES:
                raise ApiError(
                    413 if length > MAX_JSON_BYTES else 400,
                    "invalid_body",
                    "A bounded JSON body is required",
                )
            try:
                value = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ApiError(
                    400, "invalid_json", "Request body must be valid UTF-8 JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ApiError(
                    400, "invalid_json", "Request body must be a JSON object"
                )
            return value

        def _route(self, method: str) -> None:
            parsed = urlparse(self.path)
            if method == "GET" and parsed.path == "/":
                try:
                    html = TEMPLATE_PATH.read_bytes()
                except OSError as exc:
                    raise ApiError(
                        500,
                        "template_unavailable",
                        "The annotation interface is unavailable",
                    ) from exc
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'self'",
                )
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if method == "GET" and parsed.path in {"/api/default-config", "/api/state"}:
                self._send_json(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "config": state.public_config(),
                        "summary": state.summary(),
                    }
                )
                return
            if method == "GET" and parsed.path == "/api/summary":
                self._send_json(state.summary())
                return
            if method == "GET" and parsed.path == "/api/annotation":
                image_path = (parse_qs(parsed.query).get("image_path") or [""])[0]
                self._send_json(state.get_annotation(image_path))
                return
            if method == "GET" and parsed.path == "/api/image":
                image_path = (parse_qs(parsed.query).get("image_path") or [""])[0]
                body, content_type = state.image_bytes(image_path, s3_client=s3_client)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "private, max-age=300")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if method == "POST":
                payload = self._read_json_body()
                if parsed.path == "/api/load-batch":
                    self._send_json(state.load_batch(payload.get("config", {})))
                    return
                if parsed.path == "/api/save-annotation":
                    self._send_json(state.save_annotation(payload))
                    return
                if parsed.path == "/api/progress":
                    self._send_json(state.save_progress(payload))
                    return
            raise ApiError(404, "not_found", "The requested endpoint does not exist")

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._route("GET")
            except ApiError as error:
                self._send_error(error)
            except Exception:
                self._send_error(
                    ApiError(
                        500, "internal_error", "The request could not be completed"
                    )
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._route("POST")
            except ApiError as error:
                self._send_error(error)
            except Exception:
                self._send_error(
                    ApiError(
                        500, "internal_error", "The request could not be completed"
                    )
                )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local web annotator for scenic tiles"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--labels-csv", type=str, default="data/raw/labels.csv")
    parser.add_argument("--batch-csv", type=str, default="")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument(
        "--annotations-csv", type=str, default="data/raw/labels_human.csv"
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-by-class", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--annotator-id", type=str, default=os.getenv("USER", "annotator")
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Explicitly allow binding to a non-loopback host",
    )
    parser.add_argument(
        "--allowed-s3-root",
        action="append",
        default=[],
        help="Credential-free s3://bucket/prefix root approved for image reads (repeatable)",
    )
    args = parser.parse_args(argv)
    if not _is_loopback_host(args.host) and not args.allow_remote:
        parser.error("non-loopback --host requires explicit --allow-remote")
    return args


def main() -> None:
    args = parse_args()
    s3_roots = list(args.allowed_s3_root)
    if str(args.raw_dir).startswith("s3://") and args.raw_dir not in s3_roots:
        s3_roots.append(args.raw_dir)
    policy = PathPolicy(s3_roots=s3_roots)
    config = AnnotatorConfig(
        labels_csv=args.labels_csv,
        batch_csv=args.batch_csv,
        raw_dir=args.raw_dir,
        annotations_csv=args.annotations_csv,
        sample_size=args.sample_size,
        seed=args.seed,
        stratify_by_class=args.stratify_by_class,
        annotator_id=args.annotator_id,
    )
    state = AnnotatorState(config, path_policy=policy)
    s3_client = None
    if state.config.raw_dir.startswith("s3://"):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required when raw_dir uses s3://") from exc
        s3_client = boto3.client("s3")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, s3_client))
    print(f"Scenic annotator running on http://{args.host}:{args.port}")
    print(f"Annotator identity: {state.config.annotator_id}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
