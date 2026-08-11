"""Metered, fail-closed object-storage transport for catalog and range reads.

Every LIST/HEAD/range attempt reserves request, transfer-byte, local-byte,
and cost capacity against ``NetworkCaps`` *before* dispatch; actual bytes and
cost reconcile the reservation afterwards. Retries are owned by the transport:
only declared transient failures (throttles, 5xx, connection/read errors) are
retried, with bounded exponential backoff that is injectable for tests, and
every retry is its own separately reserved attempt. Authentication,
authorization, checksum, malformed-response, range/identity-mismatch, and cap
failures are never retried.

Requester-Pays buckets fail closed: a transport constructed with
``requester_pays=True`` refuses every operation with
``RequesterPaysDeniedError`` unless the caps explicitly authorize spend
(``allow_requester_pays=True`` and ``max_requester_pays_usd > 0``), and no
request is dispatched without that authorization.

All accounting is written to a ``MeteredLedger`` as deterministic,
secret-free JSONL records (one ``reserve`` plus one ``settle`` record per
attempt), so the ledger alone proves how many requests, bytes, and dollars a
run consumed, including failed and retried attempts.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from http.client import IncompleteRead as _IncompleteRead
from pathlib import Path

# ``http.client`` named this class ``IncompleteRead`` before Python 3.12 and
# ``IncompleteReadError`` from 3.12 on; accept either name.
try:
    from http.client import IncompleteReadError as _IncompleteReadError
except ImportError:  # pragma: no cover - Python < 3.12
    _IncompleteReadError = _IncompleteRead
from typing import Any, Callable, Protocol

import botocore.exceptions as _boto_exc

__all__ = [
    "AuthenticationError",
    "Boto3MeteredTransport",
    "CapExceededError",
    "ChecksumMismatchError",
    "Counters",
    "HeadObjectResult",
    "IdentityMismatchError",
    "ListObjectsResult",
    "MalformedResponseError",
    "MeteredLedger",
    "MeteredTransport",
    "MeteredTransportError",
    "NetworkCaps",
    "NotFoundError",
    "ObjectMeta",
    "RangeMismatchError",
    "RangeResult",
    "RateCard",
    "RequesterPaysDeniedError",
    "RetriesExhaustedError",
]

# S3 list_objects_v2 caps MaxKeys at 1000.
_MAX_S3_LIST_KEYS = 1000

_TRANSIENT_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

_TRANSIENT_ERROR_CODES = frozenset(
    {
        "InternalError",
        "PriorRequestNotComplete",
        "ProvisionedThroughputExceededException",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)

_AUTH_ERROR_CODES = frozenset(
    {
        "AccessDenied",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidSecurity",
        "InvalidToken",
        "MissingAuthenticationToken",
        "RequestTimeTooSkewed",
        "SignatureDoesNotMatch",
        "TokenRefreshRequired",
    }
)

_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")

_AUTH_EXCEPTIONS = tuple(
    getattr(_boto_exc, name)
    for name in (
        "CredentialRetrievalError",
        "NoCredentialsError",
        "PartialCredentialsError",
        "SSOTokenLoadError",
        "TokenRetrievalError",
    )
    if hasattr(_boto_exc, name)
)

_CONNECTION_EXCEPTIONS = tuple(
    getattr(_boto_exc, name)
    for name in (
        "ConnectionClosedError",
        "ConnectionError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "ReadTimeoutError",
    )
    if hasattr(_boto_exc, name)
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MeteredTransportError(Exception):
    """Base class for metered transport failures."""


class CapExceededError(MeteredTransportError):
    """A reservation would exceed one of the ``NetworkCaps`` limits.

    Raised before dispatch: the attempt is never sent to the network.
    """

    def __init__(
        self,
        cap: str,
        used: int | Decimal,
        requested: int | Decimal,
        limit: int | Decimal,
    ) -> None:
        super().__init__(
            f"{cap} cap exceeded: used {used} + reserved {requested} > limit {limit}"
        )
        self.cap = cap
        self.used = used
        self.requested = requested
        self.limit = limit


class RequesterPaysDeniedError(MeteredTransportError):
    """A requester-pays bucket was contacted without explicit authorization."""


class AuthenticationError(MeteredTransportError):
    """Authentication or authorization failed (never retried)."""


class NotFoundError(MeteredTransportError):
    """The object or version does not exist (never retried)."""


class ChecksumMismatchError(MeteredTransportError):
    """Response checksum validation failed (never retried)."""


class MalformedResponseError(MeteredTransportError):
    """The response violated the transport contract (never retried)."""


class RangeMismatchError(MalformedResponseError):
    """The served status/range/body did not match the request (never retried)."""


class IdentityMismatchError(MeteredTransportError):
    """An expected ETag/version did not match the response (never retried)."""


class RetriesExhaustedError(MeteredTransportError):
    """A declared transient failure persisted across all bounded attempts."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateCard:
    """Declared rate card used for cost reservation and settlement."""

    source: str
    date: str
    request_cost_usd: Decimal
    transfer_cost_per_gb_usd: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("RateCard.source must be a non-empty string")
        if not isinstance(self.date, str) or not self.date:
            raise ValueError("RateCard.date must be a non-empty string")
        for name in ("request_cost_usd", "transfer_cost_per_gb_usd"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, Decimal))
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"RateCard.{name} must be a non-negative Decimal, got {value!r}"
                )

    def cost_for(self, transfer_bytes: int) -> Decimal:
        """Cost of one attempt transferring ``transfer_bytes`` bytes."""
        byte_cost = (
            self.transfer_cost_per_gb_usd * Decimal(transfer_bytes) / Decimal(10**9)
        )
        return self.request_cost_usd + byte_cost


@dataclass(frozen=True)
class NetworkCaps:
    """Explicit, fail-closed capacity limits.

    Every limit defaults to zero, which denies all network execution: a zero
    request cap means no dispatch at all, a zero byte cap means no transfer,
    and a zero requester-pays spend cap (or a missing ``allow_requester_pays``
    acknowledgement) denies requester-pays buckets. Zero never means
    unlimited.
    """

    max_requests: int = 0
    max_transfer_bytes: int = 0
    max_local_bytes: int = 0
    max_requester_pays_usd: Decimal = Decimal("0")
    allow_requester_pays: bool = False

    def __post_init__(self) -> None:
        for name in ("max_requests", "max_transfer_bytes", "max_local_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"NetworkCaps.{name} must be a non-negative int, got {value!r}"
                )
        raw_spend = self.max_requester_pays_usd
        if not isinstance(raw_spend, (int, Decimal)) or isinstance(raw_spend, bool):
            raise ValueError(
                "NetworkCaps.max_requester_pays_usd must be a non-negative finite "
                f"Decimal, got {raw_spend!r}"
            )
        spend = Decimal(raw_spend)
        if not spend.is_finite() or spend < 0:
            raise ValueError(
                "NetworkCaps.max_requester_pays_usd must be a non-negative finite "
                f"Decimal, got {raw_spend!r}"
            )
        if not isinstance(self.allow_requester_pays, bool):
            raise ValueError("NetworkCaps.allow_requester_pays must be a bool")


@dataclass(frozen=True)
class Counters:
    """Reserved-and-settled usage counters, safe to read at any time."""

    requests: int
    transfer_bytes: int
    local_bytes: int
    cost_usd: Decimal


class SharedMeteredBudget:
    """One reservation ledger for every transport and generated local artifact."""

    def __init__(self, caps: NetworkCaps) -> None:
        if not isinstance(caps, NetworkCaps):
            raise TypeError("caps must be a NetworkCaps")
        self.caps = caps
        self.lock = threading.Lock()
        self.requests = 0
        self.transfer_bytes = 0
        self.local_bytes = 0
        self.cost_usd = Decimal("0")
        self.requester_pays_cost_usd = Decimal("0")

    def counters(self) -> Counters:
        with self.lock:
            return Counters(
                requests=self.requests,
                transfer_bytes=self.transfer_bytes,
                local_bytes=self.local_bytes,
                cost_usd=self.cost_usd,
            )

    def reserve(
        self,
        *,
        requests: int,
        transfer_bytes: int,
        local_bytes: int,
        cost_usd: Decimal,
        requester_pays: bool = False,
    ) -> None:
        with self.lock:
            values = (
                ("requests", self.requests, requests, self.caps.max_requests),
                (
                    "transfer_bytes",
                    self.transfer_bytes,
                    transfer_bytes,
                    self.caps.max_transfer_bytes,
                ),
                (
                    "local_bytes",
                    self.local_bytes,
                    local_bytes,
                    self.caps.max_local_bytes,
                ),
                *(
                    (
                        (
                            "requester_pays_usd",
                            self.requester_pays_cost_usd,
                            cost_usd,
                            self.caps.max_requester_pays_usd,
                        ),
                    )
                    if requester_pays
                    else ()
                ),
            )
            for name, used, requested, limit in values:
                if requested < 0:
                    raise ValueError(f"{name} reservation must be non-negative")
                if used + requested > limit:
                    raise CapExceededError(name, used, requested, limit)
            self.requests += requests
            self.transfer_bytes += transfer_bytes
            self.local_bytes += local_bytes
            self.cost_usd += cost_usd
            if requester_pays:
                self.requester_pays_cost_usd += cost_usd

    def adjust(
        self,
        *,
        requests: int = 0,
        transfer_bytes: int = 0,
        local_bytes: int = 0,
        cost_usd: Decimal = Decimal("0"),
        requester_pays: bool = False,
    ) -> None:
        with self.lock:
            values = (
                ("requests", self.requests, requests, self.caps.max_requests),
                (
                    "transfer_bytes",
                    self.transfer_bytes,
                    transfer_bytes,
                    self.caps.max_transfer_bytes,
                ),
                (
                    "local_bytes",
                    self.local_bytes,
                    local_bytes,
                    self.caps.max_local_bytes,
                ),
                *(
                    (
                        (
                            "requester_pays_usd",
                            self.requester_pays_cost_usd,
                            cost_usd,
                            self.caps.max_requester_pays_usd,
                        ),
                    )
                    if requester_pays
                    else ()
                ),
            )
            for name, used, delta, limit in values:
                if delta > 0 and used + delta > limit:
                    raise CapExceededError(name, used, delta, limit)
                if used + delta < 0:
                    raise RuntimeError(f"metered budget accounting underflow: {name}")
            if self.cost_usd + cost_usd < 0:
                raise RuntimeError("metered budget accounting underflow: cost_usd")
            self.requests += requests
            self.transfer_bytes += transfer_bytes
            self.local_bytes += local_bytes
            self.cost_usd += cost_usd
            if requester_pays:
                self.requester_pays_cost_usd += cost_usd


@dataclass(frozen=True)
class ObjectMeta:
    key: str
    size: int
    etag: str | None = None
    version_id: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class ListObjectsResult:
    objects: tuple[ObjectMeta, ...]
    is_truncated: bool
    next_continuation_token: str | None


@dataclass(frozen=True)
class HeadObjectResult:
    key: str
    size: int
    etag: str | None
    version_id: str | None
    last_modified: str | None
    accept_ranges: bool | None = None


@dataclass(frozen=True)
class RangeResult:
    key: str
    content: bytes
    start: int | None
    end: int | None
    etag: str | None
    version_id: str | None


# ---------------------------------------------------------------------------
# Secret-free deterministic ledger
# ---------------------------------------------------------------------------

_SECRET_KEY_HINTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "signature",
    "x-amz-",
    "security-token",
    "access-key",
    "session-token",
    "aws-secret",
)

_SECRET_VALUE_HINTS = (
    "X-Amz-Signature=",
    "x-amz-security-token",
    "aws_secret_access_key=",
    "aws_access_key_id=",
    "Authorization:",
    "Bearer ",
)

_AWS_KEY_ID_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")


def _check_secret_free(record: dict[str, Any]) -> None:
    """Raise ValueError if a record could leak credentials or signed URLs."""

    def visit(key: str | None, value: Any) -> None:
        if key is not None:
            lowered = key.lower()
            for hint in _SECRET_KEY_HINTS:
                if hint in lowered:
                    raise ValueError(
                        f"ledger record rejected: key {key!r} matches secret hint {hint!r}"
                    )
        if isinstance(value, str):
            if _AWS_KEY_ID_RE.search(value):
                raise ValueError(
                    "ledger record rejected: value matches AWS access key pattern"
                )
            for hint in _SECRET_VALUE_HINTS:
                if hint in value:
                    raise ValueError(
                        f"ledger record rejected: value matches secret hint {hint!r}"
                    )
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                visit(str(sub_key), sub_value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(None, item)

    for record_key, record_value in record.items():
        visit(str(record_key), record_value)


def _iso_timestamp(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


class MeteredLedger:
    """Append-only, deterministic, secret-free JSONL audit trail.

    Every ``append`` writes one compact JSON object (sorted keys, no
    whitespace, ``allow_nan=False``) on its own line. ``seq`` and ``ts`` are
    injected by the ledger, so records are byte-deterministic for a given
    sequence of appends and a fixed clock. Records must not contain
    credentials, signed URLs, or other secrets; ``append`` rejects them.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock if clock is not None else _time.time
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._next_seq = self._initial_seq()

    def _initial_seq(self) -> int:
        if self._path is None or not self._path.exists():
            return 0
        last = 0
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = parsed.get("seq")
            if isinstance(seq, int) and seq >= last:
                last = seq + 1
        return last

    def append(self, record: dict[str, Any]) -> None:
        """Append one stamped record; raises ValueError for secrets or
        non-JSON-serializable values."""
        if not isinstance(record, dict):
            raise TypeError("ledger records must be dicts")
        _check_secret_free(record)
        with self._lock:
            stamped = dict(record)
            stamped["seq"] = self._next_seq
            self._next_seq += 1
            stamped["ts"] = _iso_timestamp(self._clock())
            try:
                line = json.dumps(
                    stamped, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ledger record must contain only JSON-serializable values"
                ) from exc
            self._records.append(json.loads(line))
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()

    def records(self) -> list[dict[str, Any]]:
        """This session's records, in append order."""
        with self._lock:
            return [dict(record) for record in self._records]


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------


class MeteredTransport(Protocol):
    """Interface adapters consume; implemented by ``Boto3MeteredTransport``."""

    def list_objects(
        self,
        prefix: str,
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
        max_response_bytes: int | None = None,
    ) -> ListObjectsResult: ...

    def head_object(
        self,
        key: str,
        *,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> HeadObjectResult: ...

    def get_range(
        self,
        key: str,
        *,
        start: int | None = None,
        end: int | None = None,
        max_response_bytes: int | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> RangeResult: ...
    def get_object(
        self,
        key: str,
        *,
        max_response_bytes: int | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> bytes: ...
    def counters(self) -> Counters: ...


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _classify_error(exc: Exception) -> str | None:
    """Classify an exception as ``"transient"`` or a terminal category.

    Only the declared transient set is ever retried; every other category is
    terminal, and ``None`` (unrecognized) is terminal by fail-closed default.
    """
    if isinstance(exc, CapExceededError):
        return "cap"
    if isinstance(exc, (RequesterPaysDeniedError, AuthenticationError)):
        return "auth"
    if isinstance(exc, NotFoundError):
        return "not_found"
    if isinstance(exc, ChecksumMismatchError):
        return "checksum"
    if isinstance(exc, RangeMismatchError):
        return "range"
    if isinstance(exc, MalformedResponseError):
        return "malformed"
    if isinstance(exc, IdentityMismatchError):
        return "identity"
    if isinstance(exc, RetriesExhaustedError):
        return "transient"
    if isinstance(exc, _boto_exc.ClientError):
        error = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
        code = str(error.get("Code", ""))
        status = error.get("HTTPStatusCode")
        if code in _TRANSIENT_ERROR_CODES:
            return "transient"
        if code in _AUTH_ERROR_CODES:
            return "auth"
        if status in _TRANSIENT_HTTP_STATUS:
            return "transient"
        if status == 403:
            return "auth"
        if status == 404:
            return "not_found"
        if status == 412:
            return "identity"
        if status == 416:
            return "range"
        return "malformed"
    if isinstance(exc, _AUTH_EXCEPTIONS):
        return "auth"
    if isinstance(exc, _boto_exc.ChecksumError):
        return "checksum"
    if isinstance(exc, _CONNECTION_EXCEPTIONS):
        return "transient"
    if isinstance(exc, _boto_exc.ResponseStreamingError):
        # The HTTP stream broke mid-read; the transfer did not complete.
        return "transient"
    if isinstance(exc, _IncompleteReadError):
        return "transient"
    return None


def _describe_error(exc: Exception) -> str:
    """A secret-free description of an error for messages and records."""
    if isinstance(exc, _boto_exc.ClientError):
        error = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
        return (
            f"client error HTTP {error.get('HTTPStatusCode')} code={error.get('Code')}"
        )
    return type(exc).__name__


def _raise_domain(category: str | None, exc: Exception) -> Exception:
    """Map a terminal category to its domain error; unknown stays as-is."""
    if category == "auth":
        return AuthenticationError(
            f"authentication/authorization failure: {_describe_error(exc)}"
        )
    if category == "not_found":
        return NotFoundError(f"object not found: {_describe_error(exc)}")
    if category == "checksum":
        return ChecksumMismatchError(
            f"checksum validation failed: {_describe_error(exc)}"
        )
    if category == "identity":
        return IdentityMismatchError(
            f"expected ETag/version mismatch: {_describe_error(exc)}"
        )
    if category == "range":
        return RangeMismatchError(f"byte range mismatch: {_describe_error(exc)}")
    if category == "malformed":
        return MalformedResponseError(f"malformed response: {_describe_error(exc)}")
    return exc


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _http_status(response: dict[str, Any]) -> int | None:
    meta = response.get("ResponseMetadata")
    if isinstance(meta, dict):
        return meta.get("HTTPStatusCode")
    return None


def _unquote(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip('"')


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _partial_bytes(exc: Exception) -> int:
    if isinstance(exc, _IncompleteReadError):
        return len(exc.partial)
    cause = exc.args[0] if exc.args else None
    if isinstance(cause, _IncompleteReadError):
        return len(cause.partial)
    return 0


def _list_response_bytes(response: dict[str, Any]) -> int:
    """Bytes actually read from a LIST response.

    Uses the response ``content-length`` header when present; otherwise falls
    back to the byte length of the canonical compact JSON serialization of the
    parsed response, which is the data this transport actually consumed.
    """
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders")
    if isinstance(headers, dict):
        for name, value in headers.items():
            if str(name).lower() == "content-length":
                try:
                    return int(value)
                except (TypeError, ValueError):
                    break
    return len(
        json.dumps(
            response, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _parse_object_meta(item: dict[str, Any]) -> ObjectMeta:
    key = item.get("Key")
    size = item.get("Size")
    if not isinstance(key, str) or not isinstance(size, int) or isinstance(size, bool):
        raise MalformedResponseError("list_objects_v2 object entry missing Key or Size")
    return ObjectMeta(
        key=key,
        size=size,
        etag=_unquote(item.get("ETag")),
        version_id=item.get("VersionId"),
        last_modified=_to_iso(item.get("LastModified")),
    )


def _check_identity(
    *,
    expected_etag: str | None,
    expected_version_id: str | None,
    etag: str | None,
    version_id: str | None,
) -> None:
    if expected_etag is not None and etag != expected_etag:
        raise IdentityMismatchError(f"expected ETag {expected_etag!r}, got {etag!r}")
    if expected_version_id is not None and version_id != expected_version_id:
        raise IdentityMismatchError(
            f"expected version {expected_version_id!r}, got {version_id!r}"
        )


def _validate_range_request(
    start: int | None,
    end: int | None,
    max_response_bytes: int | None,
) -> tuple[int | None, int | None, int]:
    """Validate a range request and derive its reservation in bytes."""
    if start is not None and (
        not isinstance(start, int) or isinstance(start, bool) or start < 0
    ):
        raise ValueError(f"start must be a non-negative int or None, got {start!r}")
    if end is not None and (
        not isinstance(end, int) or isinstance(end, bool) or end < 0
    ):
        raise ValueError(f"end must be a non-negative int or None, got {end!r}")
    if start is not None and end is not None and end < start:
        raise ValueError(f"empty range: end {end} < start {start}")
    if start is None and end is not None:
        raise ValueError(
            "end without start is ambiguous; request a closed range or declare "
            "max_response_bytes for an open-ended range"
        )
    if max_response_bytes is not None and (
        not isinstance(max_response_bytes, int)
        or isinstance(max_response_bytes, bool)
        or max_response_bytes <= 0
    ):
        raise ValueError("max_response_bytes must be a positive int or None")
    if start is not None and end is not None:
        length = end - start + 1
        if max_response_bytes is not None and max_response_bytes < length:
            raise ValueError(
                f"max_response_bytes {max_response_bytes} < closed range length {length}"
            )
        return start, end, length
    if max_response_bytes is None:
        raise ValueError(
            "open-ended or whole-object request requires max_response_bytes "
            "(the declared maximum response size to reserve)"
        )
    return start, end, max_response_bytes


def _validate_get_response(
    response: dict[str, Any],
    content: bytes,
    start: int | None,
    end: int | None,
    reservation: int,
    expected_etag: str | None,
    expected_version_id: str | None,
) -> None:
    """Validate status, Content-Range, body length, and expected identity."""
    status = _http_status(response)
    if status is None:
        raise MalformedResponseError("get_object response missing HTTPStatusCode")
    if start is not None and end is not None:
        if status != 206:
            raise RangeMismatchError(f"ranged request got HTTP {status}; expected 206")
        content_range = response.get("ContentRange")
        match = (
            _CONTENT_RANGE_RE.fullmatch(content_range)
            if isinstance(content_range, str)
            else None
        )
        if match is None:
            raise MalformedResponseError(
                "206 response missing a valid Content-Range header"
            )
        served_start, served_end, total = (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
        )
        if served_start != start or served_end != end:
            raise RangeMismatchError(
                f"served range bytes={served_start}-{served_end} != "
                f"requested bytes={start}-{end}"
            )
        if len(content) != end - start + 1:
            raise RangeMismatchError(
                f"body length {len(content)} != requested range length "
                f"{end - start + 1}"
            )
        if total != "*" and int(total) <= end:
            raise RangeMismatchError(
                f"Content-Range total {total} <= requested end {end}"
            )
    elif start is not None:
        if status != 206:
            raise RangeMismatchError(
                f"open-ended range request got HTTP {status}; expected 206"
            )
        content_range = response.get("ContentRange")
        match = (
            _CONTENT_RANGE_RE.fullmatch(content_range)
            if isinstance(content_range, str)
            else None
        )
        if match is None:
            raise MalformedResponseError(
                "206 response missing a valid Content-Range header"
            )
        served_start, served_end, total = (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
        )
        if served_start != start:
            raise RangeMismatchError(
                f"served range starts at {served_start}; requested start {start}"
            )
        if len(content) != served_end - served_start + 1:
            raise RangeMismatchError(
                f"body length {len(content)} != served range length "
                f"{served_end - served_start + 1}"
            )
        if len(content) > reservation:
            raise RangeMismatchError(
                f"body length {len(content)} exceeds declared max_response_bytes "
                f"{reservation}"
            )
        if total != "*" and int(total) <= served_end:
            raise RangeMismatchError(
                f"Content-Range total {total} <= served end {served_end}"
            )
    else:
        if status != 200:
            raise MalformedResponseError(
                f"whole-object request got HTTP {status}; expected 200"
            )
        if len(content) > reservation:
            raise MalformedResponseError(
                f"body length {len(content)} exceeds declared max_response_bytes "
                f"{reservation}"
            )
    content_length = response.get("ContentLength")
    if content_length is not None and int(content_length) != len(content):
        raise MalformedResponseError(
            f"Content-Length {content_length} != body length {len(content)}"
        )
    _check_identity(
        expected_etag=expected_etag,
        expected_version_id=expected_version_id,
        etag=_unquote(response.get("ETag")),
        version_id=response.get("VersionId"),
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class Boto3MeteredTransport:
    """S3 metered transport with pre-dispatch reservations and owned retries.

    Parameters:
        bucket: S3 bucket name.
        caps: explicit capacity limits; all operations fail closed against them.
        ledger: audit trail receiving one ``reserve`` + one ``settle`` record
            per attempt.
        rate_card: cost model used to reserve and settle spend.
        requester_pays: True when the bucket requires ``RequestPayer=requester``.
            Operations are refused with ``RequesterPaysDeniedError`` unless the
            caps explicitly authorize requester-pays spend.
        client: botocore S3 client to use; when omitted, a client with implicit
            SDK retries disabled is created (``retries={"total_max_attempts": 1, "mode": "disabled"}``).
        max_attempts: total attempts per call (1 = no retries); retries are
            bounded and each counts separately against the caps.
        base_backoff_seconds / max_backoff_seconds: exponential backoff bounds.
        jitter: multiply each backoff by ``uniform(0.5, 1.0)``.
        backoff_fn: injectable ``backoff(attempt) -> seconds``; overrides the
            default exponential formula (used by tests for determinism).
        sleep: injectable sleep callable (tests pass a recorder).
        connect_timeout / read_timeout: applied to the created client only.
    """

    def __init__(
        self,
        bucket: str,
        *,
        caps: NetworkCaps,
        ledger: MeteredLedger,
        rate_card: RateCard,
        requester_pays: bool = False,
        shared_budget: SharedMeteredBudget | None = None,
        client: Any | None = None,
        max_attempts: int = 4,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        jitter: bool = True,
        backoff_fn: Callable[[int], float] | None = None,
        sleep: Callable[[float], None] = _time.sleep,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
    ) -> None:
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("bucket must be a non-empty string")
        if not isinstance(caps, NetworkCaps):
            raise TypeError("caps must be a NetworkCaps")
        if not isinstance(ledger, MeteredLedger):
            raise TypeError("ledger must be a MeteredLedger")
        if not isinstance(rate_card, RateCard):
            raise TypeError("rate_card must be a RateCard")
        if shared_budget is not None and not isinstance(
            shared_budget, SharedMeteredBudget
        ):
            raise TypeError("shared_budget must be a SharedMeteredBudget or None")
        if shared_budget is not None and shared_budget.caps != caps:
            raise ValueError("shared_budget caps must equal transport caps")
        if not isinstance(requester_pays, bool):
            raise TypeError("requester_pays must be a bool")
        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be an int >= 1")
        if (
            not isinstance(base_backoff_seconds, (int, float))
            or base_backoff_seconds <= 0
        ):
            raise ValueError("base_backoff_seconds must be positive")
        if (
            not isinstance(max_backoff_seconds, (int, float))
            or max_backoff_seconds < base_backoff_seconds
        ):
            raise ValueError("max_backoff_seconds must be >= base_backoff_seconds")
        if not isinstance(jitter, bool):
            raise TypeError("jitter must be a bool")
        if backoff_fn is not None and not callable(backoff_fn):
            raise TypeError("backoff_fn must be callable or None")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("connect_timeout and read_timeout must be positive")

        self.bucket = bucket
        self._caps = caps
        self.ledger = ledger
        self._rate_card = rate_card
        self._requester_pays = requester_pays
        self._max_attempts = max_attempts
        self._base_backoff = base_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._jitter = jitter
        self._backoff_fn = backoff_fn
        self._sleep = sleep

        self._budget = shared_budget or SharedMeteredBudget(caps)

        self._client = (
            client
            if client is not None
            else self._build_client(connect_timeout, read_timeout)
        )

    # -- public read-only surface -------------------------------------------

    @property
    def caps(self) -> NetworkCaps:
        return self._caps

    @property
    def rate_card(self) -> RateCard:
        return self._rate_card

    @property
    def requester_pays(self) -> bool:
        return self._requester_pays

    def counters(self) -> Counters:
        return self._budget.counters()

    # -- client construction ------------------------------------------------

    @staticmethod
    def _build_client(connect_timeout: float, read_timeout: float) -> Any:
        import boto3
        from botocore.config import Config

        # One total attempt means the SDK never retries. Retries are owned by
        # this transport so every attempt is reserved, metered, and recorded.
        return boto3.client(
            "s3",
            config=Config(
                retries={"total_max_attempts": 1, "mode": "standard"},
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            ),
        )

    # -- backoff -------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        if self._backoff_fn is not None:
            return self._backoff_fn(attempt)
        delay = min(self._base_backoff * (2.0 ** (attempt - 1)), self._max_backoff)
        if self._jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay

    # -- accounting ----------------------------------------------------------

    def _cost_for(self, transfer_bytes: int) -> Decimal:
        return self._rate_card.cost_for(transfer_bytes)

    def _check_requester_pays_allowed(self) -> None:
        if not (
            self._caps.allow_requester_pays and self._caps.max_requester_pays_usd > 0
        ):
            raise RequesterPaysDeniedError(
                "bucket requires RequestPayer=requester; authorize with "
                "NetworkCaps(allow_requester_pays=True, max_requester_pays_usd>0)"
            )

    def _reserve(
        self,
        *,
        operation: str,
        key: str | None,
        prefix: str | None,
        range_start: int | None,
        range_end: int | None,
        attempt: int,
        reserve_transfer: int,
        reserve_local: int,
    ) -> Decimal:
        estimated = self._cost_for(reserve_transfer)
        self._budget.reserve(
            requests=1,
            transfer_bytes=reserve_transfer,
            local_bytes=reserve_local,
            cost_usd=estimated,
            requester_pays=self._requester_pays,
        )
        try:
            record = {
                "event": "reserve",
                "operation": operation,
                "key": key,
                "prefix": prefix,
                "range_start": range_start,
                "range_end": range_end,
                "attempt": attempt,
                "requester_pays": self._requester_pays,
                "reserved_requests": 1,
                "reserved_transfer_bytes": reserve_transfer,
                "reserved_local_bytes": reserve_local,
                "reserved_cost_usd": str(estimated),
            }
            self.ledger.append(record)
        except Exception:
            self._budget.adjust(
                requests=-1,
                transfer_bytes=-reserve_transfer,
                local_bytes=-reserve_local,
                cost_usd=-estimated,
                requester_pays=self._requester_pays,
            )
            raise
        return estimated

    def _settle(
        self,
        *,
        operation: str,
        key: str | None,
        prefix: str | None,
        range_start: int | None,
        range_end: int | None,
        attempt: int,
        reserved_transfer: int,
        reserved_local: int,
        estimated: Decimal,
        actual_transfer: int,
        actual_local: int,
        outcome: str,
        error_type: str | None = None,
        error_category: str | None = None,
    ) -> None:
        actual_cost = self._cost_for(actual_transfer)
        self._budget.adjust(
            transfer_bytes=actual_transfer - reserved_transfer,
            local_bytes=actual_local - reserved_local,
            cost_usd=actual_cost - estimated,
            requester_pays=self._requester_pays,
        )
        record = {
            "event": "settle",
            "operation": operation,
            "key": key,
            "prefix": prefix,
            "range_start": range_start,
            "range_end": range_end,
            "attempt": attempt,
            "requester_pays": self._requester_pays,
            "outcome": outcome,
            "error_type": error_type,
            "error_category": error_category,
            "actual_transfer_bytes": actual_transfer,
            "actual_local_bytes": actual_local,
            "actual_cost_usd": str(actual_cost),
        }
        try:
            self.ledger.append(record)
        except Exception:
            self._budget.adjust(
                transfer_bytes=reserved_transfer - actual_transfer,
                local_bytes=reserved_local - actual_local,
                cost_usd=estimated - actual_cost,
                requester_pays=self._requester_pays,
            )
            raise

    # -- retry loop ----------------------------------------------------------

    def _call(
        self,
        *,
        operation: str,
        key: str | None = None,
        prefix: str | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        reserve_transfer: int,
        reserve_local: int,
        run: Callable[[dict[str, int]], Any],
    ) -> Any:
        if self._requester_pays:
            self._check_requester_pays_allowed()
        attempt = 0
        while True:
            attempt += 1
            estimated = self._reserve(
                operation=operation,
                key=key,
                prefix=prefix,
                range_start=range_start,
                range_end=range_end,
                attempt=attempt,
                reserve_transfer=reserve_transfer,
                reserve_local=reserve_local,
            )
            state: dict[str, int] = {"transfer": 0, "local": 0}
            try:
                payload = run(state)
            except Exception as exc:
                actual_transfer = state["transfer"]
                actual_local = state["local"]
                category = _classify_error(exc)
                if category == "transient" and attempt < self._max_attempts:
                    self._settle(
                        operation=operation,
                        key=key,
                        prefix=prefix,
                        range_start=range_start,
                        range_end=range_end,
                        attempt=attempt,
                        reserved_transfer=reserve_transfer,
                        reserved_local=reserve_local,
                        estimated=estimated,
                        actual_transfer=actual_transfer,
                        actual_local=actual_local,
                        outcome="retry",
                        error_type=type(exc).__name__,
                        error_category="transient",
                    )
                    self._sleep(self._backoff(attempt))
                    continue
                if category == "transient":
                    error = RetriesExhaustedError(
                        f"transient failure after {attempt} attempt(s): "
                        f"{_describe_error(exc)}",
                        attempts=attempt,
                        last_error=exc,
                    )
                    self._settle(
                        operation=operation,
                        key=key,
                        prefix=prefix,
                        range_start=range_start,
                        range_end=range_end,
                        attempt=attempt,
                        reserved_transfer=reserve_transfer,
                        reserved_local=reserve_local,
                        estimated=estimated,
                        actual_transfer=actual_transfer,
                        actual_local=actual_local,
                        outcome="error",
                        error_type="RetriesExhaustedError",
                        error_category="transient",
                    )
                    raise error from exc
                domain = _raise_domain(category, exc)
                self._settle(
                    operation=operation,
                    key=key,
                    prefix=prefix,
                    range_start=range_start,
                    range_end=range_end,
                    attempt=attempt,
                    reserved_transfer=reserve_transfer,
                    reserved_local=reserve_local,
                    estimated=estimated,
                    actual_transfer=actual_transfer,
                    actual_local=actual_local,
                    outcome="error",
                    error_type=type(domain).__name__,
                    error_category=category if category is not None else "unknown",
                )
                if domain is exc:
                    raise
                raise domain from exc
            else:
                actual_transfer = state["transfer"]
                actual_local = state["local"]
                self._settle(
                    operation=operation,
                    key=key,
                    prefix=prefix,
                    range_start=range_start,
                    range_end=range_end,
                    attempt=attempt,
                    reserved_transfer=reserve_transfer,
                    reserved_local=reserve_local,
                    estimated=estimated,
                    actual_transfer=actual_transfer,
                    actual_local=actual_local,
                    outcome="ok",
                )
                return payload

    # -- operations ----------------------------------------------------------

    def list_objects(
        self,
        prefix: str,
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
        max_response_bytes: int | None = None,
    ) -> ListObjectsResult:
        if (
            not isinstance(max_keys, int)
            or isinstance(max_keys, bool)
            or not 1 <= max_keys <= _MAX_S3_LIST_KEYS
        ):
            raise ValueError(
                f"max_keys must be an int in [1, {_MAX_S3_LIST_KEYS}], got {max_keys!r}"
            )
        if continuation_token is not None and not isinstance(continuation_token, str):
            raise ValueError("continuation_token must be a string or None")
        if (
            max_response_bytes is None
            or not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or max_response_bytes <= 0
        ):
            raise ValueError(
                "list_objects requires caller-provided max_response_bytes "
                "(a positive integer declared maximum response size to reserve)"
            )
        reservation = max_response_bytes

        def run(state: dict[str, int]) -> ListObjectsResult:
            params: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": max_keys,
            }
            if continuation_token is not None:
                params["ContinuationToken"] = continuation_token
            if self._requester_pays:
                params["RequestPayer"] = "requester"
            response = self._client.list_objects_v2(**params)
            if not isinstance(response, dict):
                raise MalformedResponseError(
                    "list_objects_v2 returned a non-dict response"
                )
            if _http_status(response) != 200:
                raise MalformedResponseError(
                    f"list_objects_v2 unexpected HTTP status {_http_status(response)}"
                )
            actual = _list_response_bytes(response)
            state["transfer"] = actual
            contents = response.get("Contents")
            if contents is None:
                contents = []
            if not isinstance(contents, list) or any(
                not isinstance(item, dict) for item in contents
            ):
                raise MalformedResponseError(
                    "list_objects_v2 Contents must be a list of objects"
                )
            if len(contents) > max_keys:
                raise MalformedResponseError(
                    f"list_objects_v2 returned {len(contents)} objects for "
                    f"max_keys={max_keys}"
                )
            if actual > reservation:
                raise MalformedResponseError(
                    f"list_objects_v2 response of {actual} bytes exceeds declared "
                    f"max_response_bytes {reservation}"
                )
            return ListObjectsResult(
                objects=tuple(_parse_object_meta(item) for item in contents),
                is_truncated=bool(response.get("IsTruncated", False)),
                next_continuation_token=response.get("NextContinuationToken"),
            )

        return self._call(
            operation="list_objects",
            prefix=prefix,
            reserve_transfer=reservation,
            reserve_local=0,
            run=run,
        )

    def head_object(
        self,
        key: str,
        *,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> HeadObjectResult:
        def run(state: dict[str, int]) -> HeadObjectResult:
            params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
            if expected_version_id is not None:
                params["VersionId"] = expected_version_id
            if self._requester_pays:
                params["RequestPayer"] = "requester"
            response = self._client.head_object(**params)
            if not isinstance(response, dict):
                raise MalformedResponseError("head_object returned a non-dict response")
            if _http_status(response) != 200:
                raise MalformedResponseError(
                    f"head_object unexpected HTTP status {_http_status(response)}"
                )
            size = response.get("ContentLength")
            if not isinstance(size, int) or isinstance(size, bool):
                raise MalformedResponseError("head_object missing ContentLength")
            etag = _unquote(response.get("ETag"))
            version_id = response.get("VersionId")
            _check_identity(
                expected_etag=expected_etag,
                expected_version_id=expected_version_id,
                etag=etag,
                version_id=version_id,
            )
            accept_ranges_header = response.get("AcceptRanges")
            return HeadObjectResult(
                key=key,
                size=size,
                etag=etag,
                version_id=version_id,
                last_modified=_to_iso(response.get("LastModified")),
                accept_ranges=(
                    True
                    if accept_ranges_header == "bytes"
                    else False
                    if isinstance(accept_ranges_header, str)
                    else None
                ),
            )

        return self._call(
            operation="head_object",
            key=key,
            reserve_transfer=0,
            reserve_local=0,
            run=run,
        )

    def get_object(
        self,
        key: str,
        *,
        max_response_bytes: int | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> bytes:
        return self.get_range(
            key=key,
            start=None,
            end=None,
            max_response_bytes=max_response_bytes,
            expected_etag=expected_etag,
            expected_version_id=expected_version_id,
        ).content

    def get_range(
        self,
        key: str,
        *,
        start: int | None = None,
        end: int | None = None,
        max_response_bytes: int | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
    ) -> RangeResult:
        start, end, reservation = _validate_range_request(
            start, end, max_response_bytes
        )
        if start is not None and end is not None:
            range_header = f"bytes={start}-{end}"
        elif start is not None:
            range_header = f"bytes={start}-"
        else:
            range_header = None

        def run(state: dict[str, int]) -> RangeResult:
            params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
            if range_header is not None:
                params["Range"] = range_header
            if expected_etag is not None:
                params["IfMatch"] = expected_etag
            if expected_version_id is not None:
                params["VersionId"] = expected_version_id
            if self._requester_pays:
                params["RequestPayer"] = "requester"
            response = self._client.get_object(**params)
            if not isinstance(response, dict):
                raise MalformedResponseError("get_object returned a non-dict response")
            body = response.get("Body")
            if body is None or not hasattr(body, "read"):
                raise MalformedResponseError(
                    "get_object response missing a readable Body"
                )
            try:
                content = body.read()
            except Exception as exc:
                state["transfer"] = state["local"] = _partial_bytes(exc)
                raise
            state["transfer"] = state["local"] = len(content)
            _validate_get_response(
                response,
                content,
                start,
                end,
                reservation,
                expected_etag,
                expected_version_id,
            )
            return RangeResult(
                key=key,
                content=content,
                start=start,
                end=end,
                etag=_unquote(response.get("ETag")),
                version_id=response.get("VersionId"),
            )

        return self._call(
            operation="get_range",
            key=key,
            range_start=start,
            range_end=end,
            reserve_transfer=reservation,
            reserve_local=reservation,
            run=run,
        )
