"""Focused tests for the metered object-storage transport contract.

All tests use fake clients and fixed clocks; nothing here touches the network.
"""

from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest
from botocore.exceptions import ChecksumError, ClientError, NoCredentialsError

from src.data_pipeline.metered_transport import (
    AuthenticationError,
    Boto3MeteredTransport,
    CapExceededError,
    ChecksumMismatchError,
    IdentityMismatchError,
    ListObjectsResult,
    MalformedResponseError,
    MeteredLedger,
    NetworkCaps,
    NotFoundError,
    RangeMismatchError,
    RangeResult,
    RateCard,
    RequesterPaysDeniedError,
    RetriesExhaustedError,
)

RATE = RateCard(
    source="https://example.test/s3-pricing",
    date="2026-08-10",
    request_cost_usd=Decimal("0.0004"),
    transfer_cost_per_gb_usd=Decimal("0.09"),
)

FIXED_CLOCK = lambda: 1000.0  # noqa: E731


class FakeS3Client:
    """Botocore-shaped fake: canned responses per method, call recording."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._plan: dict[str, list] = {}

    def plan(self, method: str, *responses: object) -> None:
        self._plan.setdefault(method, []).extend(responses)

    def _next(self, method: str, kwargs: dict) -> object:
        self.calls.append((method, dict(kwargs)))
        queued = self._plan.get(method)
        if not queued:
            raise AssertionError(f"no planned response for {method}")
        response = queued.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response()
        return response

    def get_object(self, **kwargs) -> object:
        return self._next("get_object", kwargs)

    def head_object(self, **kwargs) -> object:
        return self._next("head_object", kwargs)

    def list_objects_v2(self, **kwargs) -> object:
        return self._next("list_objects_v2", kwargs)


def resp(status: int = 200, headers: dict | None = None, **extra: object) -> dict:
    return {
        "ResponseMetadata": {"HTTPStatusCode": status, "HTTPHeaders": headers or {}},
        **extra,
    }


def range_resp(
    content: bytes,
    *,
    status: int = 206,
    start: int = 0,
    end: int | None = None,
    total: int | None = None,
    etag: str = '"abc"',
    version_id: str | None = None,
    content_length: int | None = None,
) -> dict:
    end = end if end is not None else start + len(content) - 1
    total = total if total is not None else max(end + 1, len(content))
    return resp(
        status,
        Body=io.BytesIO(content),
        ContentLength=content_length if content_length is not None else len(content),
        ContentRange=f"bytes {start}-{end}/{total}",
        ETag=etag,
        VersionId=version_id,
    )


def list_resp(
    objects: list[tuple[str, int, str | None]] | None = None,
    *,
    is_truncated: bool = False,
    next_token: str | None = None,
    content_length: int | None = None,
) -> dict:
    contents = []
    for key, size, etag in objects or []:
        entry: dict = {"Key": key, "Size": size}
        if etag is not None:
            entry["ETag"] = etag
        contents.append(entry)
    body: dict = {"Contents": contents, "IsTruncated": is_truncated}
    if next_token is not None:
        body["NextContinuationToken"] = next_token
    headers = (
        {"content-length": str(content_length)} if content_length is not None else {}
    )
    return resp(200, headers, **body)


def client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "HTTPStatusCode": status}, "ResponseMetadata": {}},
        "GetObject",
    )


def make_ledger(tmp_path: Path) -> MeteredLedger:
    return MeteredLedger(tmp_path / "ledger.jsonl", clock=FIXED_CLOCK)


def make_transport(
    client: FakeS3Client,
    caps: NetworkCaps,
    ledger: MeteredLedger,
    **kwargs: object,
) -> Boto3MeteredTransport:
    kwargs.setdefault("sleep", lambda _seconds: None)
    kwargs.setdefault("backoff_fn", lambda _attempt: 0.0)
    return Boto3MeteredTransport(
        "bucket", client=client, caps=caps, ledger=ledger, rate_card=RATE, **kwargs
    )


def base_caps(**overrides: object) -> NetworkCaps:
    values = {
        "max_requests": 10,
        "max_transfer_bytes": 100_000,
        "max_local_bytes": 100_000,
    }
    values.update(overrides)
    return NetworkCaps(**values)


def assert_canonical_jsonl(path: Path) -> None:
    for line in path.read_text().splitlines():
        parsed = json.loads(line)
        assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == line


# ---------------------------------------------------------------------------
# Caps and requester-pays authorization
# ---------------------------------------------------------------------------


def test_caps_defaults_deny_all_operations(tmp_path: Path) -> None:
    client = FakeS3Client()
    ledger = make_ledger(tmp_path)
    transport = make_transport(client, NetworkCaps(), ledger)
    with pytest.raises(CapExceededError) as excinfo:
        transport.head_object("k")
    assert excinfo.value.cap == "requests"
    with pytest.raises(CapExceededError):
        transport.list_objects("prefix", max_response_bytes=1000)
    with pytest.raises(CapExceededError):
        transport.get_range("k", start=0, end=9)
    assert client.calls == []
    assert transport.counters().requests == 0
    assert transport.counters().transfer_bytes == 0
    assert ledger.records() == []


def test_cap_exceeded_stops_before_dispatch(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", range_resp(b"x" * 60, start=0, end=59, total=60))
    transport = make_transport(
        client,
        NetworkCaps(max_requests=5, max_transfer_bytes=100, max_local_bytes=100),
        make_ledger(tmp_path),
    )
    transport.get_range("a", start=0, end=59)
    with pytest.raises(CapExceededError) as excinfo:
        transport.get_range("b", start=0, end=49)
    assert excinfo.value.cap == "transfer_bytes"
    assert [call[0] for call in client.calls] == ["get_object"]
    counters = transport.counters()
    assert counters.requests == 1
    assert counters.transfer_bytes == 60
    assert counters.local_bytes == 60


def test_local_bytes_cap_enforced(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", range_resp(b"x" * 10, start=0, end=9, total=10))
    transport = make_transport(
        client,
        NetworkCaps(max_requests=5, max_transfer_bytes=1000, max_local_bytes=9),
        make_ledger(tmp_path),
    )
    with pytest.raises(CapExceededError) as excinfo:
        transport.get_range("a", start=0, end=9)
    assert excinfo.value.cap == "local_bytes"
    assert client.calls == []


def test_requester_pays_denied_without_acknowledgement(tmp_path: Path) -> None:
    client = FakeS3Client()
    ledger = make_ledger(tmp_path)
    transport = make_transport(client, base_caps(), ledger, requester_pays=True)
    with pytest.raises(RequesterPaysDeniedError):
        transport.get_range("k", start=0, end=9)
    with pytest.raises(RequesterPaysDeniedError):
        transport.list_objects("p", max_response_bytes=1000)
    with pytest.raises(RequesterPaysDeniedError):
        transport.head_object("k")
    assert client.calls == []
    assert transport.counters().requests == 0
    assert ledger.records() == []


def test_requester_pays_zero_spend_still_denied(tmp_path: Path) -> None:
    caps = base_caps(allow_requester_pays=True)  # but spend cap is zero
    transport = make_transport(
        FakeS3Client(), caps, make_ledger(tmp_path), requester_pays=True
    )
    with pytest.raises(RequesterPaysDeniedError):
        transport.get_range("k", start=0, end=9)


def test_requester_pays_authorized_sends_request_payer(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", range_resp(b"0123456789", start=0, end=9, total=100))
    client.plan("head_object", resp(200, ContentLength=100, ETag='"e1"'))
    client.plan("list_objects_v2", list_resp([("a", 5, '"e"')], content_length=120))
    caps = base_caps(max_requester_pays_usd=Decimal("1"), allow_requester_pays=True)
    transport = make_transport(client, caps, make_ledger(tmp_path), requester_pays=True)
    transport.get_range("k", start=0, end=9)
    transport.head_object("k")
    transport.list_objects("p", max_response_bytes=1000)
    assert len(client.calls) == 3
    for _method, kwargs in client.calls:
        assert kwargs["RequestPayer"] == "requester"


def test_requester_pays_cost_cap_enforced(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", range_resp(b"x" * 50, start=0, end=49, total=50))
    caps = base_caps(
        allow_requester_pays=True,
        max_requester_pays_usd=RATE.cost_for(50) + Decimal("0.0000000001"),
    )
    transport = make_transport(client, caps, make_ledger(tmp_path), requester_pays=True)
    transport.get_range("k", start=0, end=49)
    with pytest.raises(CapExceededError) as excinfo:
        transport.get_range("k2", start=0, end=49)
    assert excinfo.value.cap == "requester_pays_usd"
    assert len(client.calls) == 1
    assert transport.counters().requests == 1


# ---------------------------------------------------------------------------
# Reservations, reconciliation, and determinism
# ---------------------------------------------------------------------------


def test_range_reservation_and_reconciliation(tmp_path: Path) -> None:
    client = FakeS3Client()
    payload = b"0123456789"
    client.plan(
        "get_object",
        range_resp(payload, start=0, end=9, total=1000, etag='"e1"', version_id="v1"),
    )
    ledger = make_ledger(tmp_path)
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=10, max_local_bytes=10),
        ledger,
    )
    result = transport.get_range(
        "k.tif", start=0, end=9, expected_etag="e1", expected_version_id="v1"
    )
    assert isinstance(result, RangeResult)
    assert result.content == payload
    assert result.etag == "e1"
    assert result.version_id == "v1"
    counters = transport.counters()
    assert counters.requests == 1
    assert counters.transfer_bytes == 10
    assert counters.local_bytes == 10
    assert counters.cost_usd == RATE.cost_for(10)

    records = ledger.records()
    assert [r["event"] for r in records] == ["reserve", "settle"]
    reserve, settle = records
    assert reserve["reserved_requests"] == 1
    assert reserve["reserved_transfer_bytes"] == 10
    assert reserve["reserved_local_bytes"] == 10
    assert reserve["reserved_cost_usd"] == str(RATE.cost_for(10))
    assert settle["outcome"] == "ok"
    assert settle["actual_transfer_bytes"] == 10
    assert settle["actual_local_bytes"] == 10
    assert settle["actual_cost_usd"] == str(RATE.cost_for(10))
    assert settle["error_type"] is None
    assert settle["error_category"] is None
    assert_canonical_jsonl(tmp_path / "ledger.jsonl")


def test_failed_attempt_counts_request_and_reconciles_bytes(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object", range_resp(b"0123456789", status=200, start=0, end=9, total=10)
    )
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(RangeMismatchError):
        transport.get_range("k", start=0, end=9)
    counters = transport.counters()
    assert counters.requests == 1  # the failed attempt still counted
    assert counters.transfer_bytes == 10  # bytes were read before validation failed
    assert counters.cost_usd == RATE.cost_for(10)


def test_whole_object_get_reserves_declared_max(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object", resp(200, Body=io.BytesIO(b"0123456789"), ContentLength=10)
    )
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=100, max_local_bytes=100),
        make_ledger(tmp_path),
    )
    result = transport.get_range("catalog.json", max_response_bytes=100)
    assert result.content == b"0123456789"
    assert result.start is None and result.end is None
    assert "Range" not in client.calls[0][1]
    assert transport.counters().transfer_bytes == 10


def test_whole_object_exceeding_declared_max_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", resp(200, Body=io.BytesIO(b"x" * 50), ContentLength=50))
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(MalformedResponseError):
        transport.get_range("k", max_response_bytes=10)


def test_open_ended_range_requires_declared_max(tmp_path: Path) -> None:
    client = FakeS3Client()
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(ValueError):
        transport.get_range("k", start=100)
    with pytest.raises(ValueError):
        transport.get_range("k")
    with pytest.raises(ValueError):
        transport.get_range("k", end=5)
    with pytest.raises(ValueError):
        transport.get_range("k", start=10, end=5)
    assert client.calls == []


def test_open_ended_range_success(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", range_resp(b"0123456789", start=100, end=109, total=500))
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=1000, max_local_bytes=1000),
        make_ledger(tmp_path),
    )
    result = transport.get_range("k", start=100, max_response_bytes=100)
    assert result.content == b"0123456789"
    assert client.calls[0][1]["Range"] == "bytes=100-"
    assert transport.counters().transfer_bytes == 10


def test_open_ended_range_exceeding_declared_max_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object", range_resp(b"0123456789" * 3, start=100, end=129, total=500)
    )
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(RangeMismatchError):
        transport.get_range("k", start=100, max_response_bytes=10)


def test_retries_count_separately_and_reconcile_cost(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object",
        client_error("ServiceUnavailable", 503),
        client_error("SlowDown", 503),
        range_resp(b"0123456789", start=0, end=9, total=100),
    )
    sleeps: list[float] = []
    ledger = make_ledger(tmp_path)
    transport = make_transport(
        client,
        base_caps(),
        ledger,
        sleep=sleeps.append,
        backoff_fn=lambda attempt: float(attempt) * 0.1,
    )
    result = transport.get_range("k", start=0, end=9)
    assert result.content == b"0123456789"
    assert sleeps == [0.1, 0.2]  # backoff injected per retry
    assert len(client.calls) == 3  # each retry is a separate dispatch
    counters = transport.counters()
    assert counters.requests == 3
    assert counters.transfer_bytes == 10
    # failed attempts cost the per-request fee; only actual bytes are billed
    assert counters.cost_usd == RATE.request_cost_usd * 2 + RATE.cost_for(10)
    events = [(r["event"], r.get("outcome"), r["attempt"]) for r in ledger.records()]
    assert events == [
        ("reserve", None, 1),
        ("settle", "retry", 1),
        ("reserve", None, 2),
        ("settle", "retry", 2),
        ("reserve", None, 3),
        ("settle", "ok", 3),
    ]


def test_transient_retries_exhausted(tmp_path: Path) -> None:
    client = FakeS3Client()
    for _ in range(4):
        client.plan("get_object", client_error("ServiceUnavailable", 503))
    sleeps: list[float] = []
    ledger = make_ledger(tmp_path)
    transport = make_transport(
        client,
        base_caps(),
        ledger,
        sleep=sleeps.append,
        backoff_fn=lambda _attempt: 0.25,
    )
    with pytest.raises(RetriesExhaustedError) as excinfo:
        transport.get_range("k", start=0, end=9)
    assert excinfo.value.attempts == 4
    assert len(client.calls) == 4
    assert sleeps == [0.25, 0.25, 0.25]
    counters = transport.counters()
    assert counters.requests == 4
    assert counters.transfer_bytes == 0
    settles = [r for r in ledger.records() if r["event"] == "settle"]
    assert [r["outcome"] for r in settles] == ["retry", "retry", "retry", "error"]
    assert settles[-1]["error_category"] == "transient"
    assert settles[-1]["error_type"] == "RetriesExhaustedError"


def test_bounded_exponential_backoff_default(tmp_path: Path) -> None:
    client = FakeS3Client()
    for _ in range(5):
        client.plan("get_object", client_error("InternalError", 500))
    sleeps: list[float] = []
    transport = make_transport(
        client,
        base_caps(),
        make_ledger(tmp_path),
        sleep=sleeps.append,
        backoff_fn=None,
        jitter=False,
        base_backoff_seconds=0.5,
        max_backoff_seconds=2.0,
        max_attempts=5,
    )
    with pytest.raises(RetriesExhaustedError):
        transport.get_range("k", start=0, end=9)
    assert sleeps == [0.5, 1.0, 2.0, 2.0]  # exponential, capped


def test_retry_blocked_by_request_cap(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", client_error("ServiceUnavailable", 503))
    sleeps: list[float] = []
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=10, max_local_bytes=10),
        make_ledger(tmp_path),
        sleep=sleeps.append,
    )
    with pytest.raises(CapExceededError) as excinfo:
        transport.get_range("k", start=0, end=0)
    assert excinfo.value.cap == "requests"
    assert len(client.calls) == 1
    assert sleeps == [0.0]


# ---------------------------------------------------------------------------
# Never-retry classification
# ---------------------------------------------------------------------------


def test_auth_errors_never_retried(tmp_path: Path) -> None:
    factories = [
        lambda: client_error("AccessDenied", 403),
        lambda: client_error("SignatureDoesNotMatch", 403),
        lambda: NoCredentialsError(),
    ]
    for factory in factories:
        client = FakeS3Client()
        client.plan("head_object", factory())
        sleeps: list[float] = []
        transport = make_transport(
            client, base_caps(), make_ledger(tmp_path), sleep=sleeps.append
        )
        with pytest.raises(AuthenticationError):
            transport.head_object("k")
        assert sleeps == []
        assert len(client.calls) == 1
        assert transport.counters().requests == 1


def test_checksum_error_never_retried(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object",
        ChecksumError(
            checksum_type="CRC32",
            expected_checksum="abc",
            actual_checksum="def",
        ),
    )
    sleeps: list[float] = []
    transport = make_transport(
        client, base_caps(), make_ledger(tmp_path), sleep=sleeps.append
    )
    with pytest.raises(ChecksumMismatchError):
        transport.get_range("k", start=0, end=9)
    assert sleeps == []
    assert len(client.calls) == 1


def test_unexpected_client_error_never_retried(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", client_error("InvalidArgument", 400))
    sleeps: list[float] = []
    transport = make_transport(
        client, base_caps(), make_ledger(tmp_path), sleep=sleeps.append
    )
    with pytest.raises(MalformedResponseError):
        transport.get_range("k", start=0, end=9)
    assert sleeps == []
    assert len(client.calls) == 1


def test_range_mismatch_never_retried(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object", range_resp(b"0123456789", status=200, start=0, end=9, total=10)
    )
    sleeps: list[float] = []
    transport = make_transport(
        client, base_caps(), make_ledger(tmp_path), sleep=sleeps.append
    )
    with pytest.raises(RangeMismatchError):
        transport.get_range("k", start=0, end=9)
    assert sleeps == []
    assert len(client.calls) == 1
    assert transport.counters().transfer_bytes == 10


def test_not_found_never_retried(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("head_object", client_error("NoSuchKey", 404))
    sleeps: list[float] = []
    transport = make_transport(
        client, base_caps(), make_ledger(tmp_path), sleep=sleeps.append
    )
    with pytest.raises(NotFoundError):
        transport.head_object("missing")
    assert sleeps == []
    assert len(client.calls) == 1


def test_invalid_range_416_maps_to_range_error(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", client_error("InvalidRange", 416))
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(RangeMismatchError):
        transport.get_range("k", start=0, end=9)
    assert len(client.calls) == 1


def test_precondition_failed_412_maps_to_identity_error(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", client_error("PreconditionFailed", 412))
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(IdentityMismatchError):
        transport.get_range("k", start=0, end=9, expected_etag="e1")
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# Identity and range validation
# ---------------------------------------------------------------------------


def test_identity_mismatch_etag(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object",
        range_resp(b"0123456789", start=0, end=9, total=100, etag='"other"'),
    )
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(IdentityMismatchError):
        transport.get_range("k", start=0, end=9, expected_etag="e1")
    assert len(client.calls) == 1
    assert transport.counters().requests == 1


def test_identity_mismatch_version(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object",
        range_resp(b"0123456789", start=0, end=9, total=100, version_id="v2"),
    )
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(IdentityMismatchError):
        transport.get_range("k", start=0, end=9, expected_version_id="v1")
    assert len(client.calls) == 1
    # expected version is passed through to the request
    assert client.calls[0][1]["VersionId"] == "v1"
    # expected etag is passed as IfMatch
    client2 = FakeS3Client()
    client2.plan(
        "get_object", range_resp(b"0123456789", start=0, end=9, total=100, etag='"e1"')
    )
    transport2 = make_transport(client2, base_caps(), make_ledger(tmp_path))
    transport2.get_range("k", start=0, end=9, expected_etag="e1")
    assert client2.calls[0][1]["IfMatch"] == "e1"


def test_content_range_mismatch_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", range_resp(b"0123456789", start=5, end=14, total=100))
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(RangeMismatchError):
        transport.get_range("k", start=0, end=9)


def test_body_length_mismatch_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object", range_resp(b"01234", start=0, end=9, total=100)
    )  # 5 bytes claimed as 10
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(RangeMismatchError):
        transport.get_range("k", start=0, end=9)


def test_content_length_header_mismatch_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object",
        range_resp(b"0123456789", start=0, end=9, total=100, content_length=7),
    )
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(MalformedResponseError):
        transport.get_range("k", start=0, end=9)


def test_206_without_content_range_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "get_object", resp(206, Body=io.BytesIO(b"0123456789"), ContentLength=10)
    )
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(MalformedResponseError):
        transport.get_range("k", start=0, end=9)


# ---------------------------------------------------------------------------
# HEAD and LIST
# ---------------------------------------------------------------------------


def test_head_object_success(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "head_object", resp(200, ContentLength=1234, ETag='"e1"', VersionId="v9")
    )
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=100, max_local_bytes=100),
        make_ledger(tmp_path),
    )
    result = transport.head_object(
        "k.tif", expected_etag="e1", expected_version_id="v9"
    )
    assert result.key == "k.tif"
    assert result.size == 1234
    assert result.etag == "e1"
    assert result.version_id == "v9"
    counters = transport.counters()
    assert counters.requests == 1
    assert counters.transfer_bytes == 0
    assert counters.local_bytes == 0


def test_head_object_identity_mismatch(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("head_object", resp(200, ContentLength=5, ETag='"x"'))
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(IdentityMismatchError):
        transport.head_object("k", expected_etag="e1")
    assert len(client.calls) == 1


def test_list_objects_returns_meta_and_counts_bytes(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan(
        "list_objects_v2",
        list_resp(
            [("a.tif", 10, '"e1"'), ("b.tif", 20, None)],
            is_truncated=True,
            next_token="tok",
            content_length=321,
        ),
    )
    ledger = make_ledger(tmp_path)
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=10_000, max_local_bytes=10_000),
        ledger,
    )
    result = transport.list_objects("prefix/", max_keys=100, max_response_bytes=5000)
    assert isinstance(result, ListObjectsResult)
    assert [o.key for o in result.objects] == ["a.tif", "b.tif"]
    assert result.objects[0].size == 10
    assert result.objects[0].etag == "e1"
    assert result.is_truncated is True
    assert result.next_continuation_token == "tok"
    counters = transport.counters()
    assert counters.requests == 1
    assert counters.transfer_bytes == 321
    assert counters.local_bytes == 0
    records = ledger.records()
    assert records[0]["reserved_transfer_bytes"] == 5000
    assert records[1]["actual_transfer_bytes"] == 321


def test_list_objects_bytes_fallback_is_deterministic(tmp_path: Path) -> None:
    client = FakeS3Client()
    response = list_resp([("a.tif", 10, '"e1"')])  # no content-length header
    client.plan("list_objects_v2", response)
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=10_000, max_local_bytes=10_000),
        make_ledger(tmp_path),
    )
    transport.list_objects("prefix/", max_response_bytes=2000)
    expected = len(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    assert transport.counters().transfer_bytes == expected


def test_list_objects_continuation_passed_through(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("list_objects_v2", list_resp(content_length=10))
    transport = make_transport(
        client,
        NetworkCaps(max_requests=1, max_transfer_bytes=10_000, max_local_bytes=10_000),
        make_ledger(tmp_path),
    )
    transport.list_objects("p", continuation_token="tok123", max_response_bytes=2000)
    assert client.calls[0][1]["ContinuationToken"] == "tok123"


def test_list_objects_exceeding_declared_max_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("list_objects_v2", list_resp([("a", 5, None)], content_length=500))
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(MalformedResponseError):
        transport.list_objects("p", max_response_bytes=100)


def test_list_objects_too_many_keys_rejected(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("list_objects_v2", list_resp([("a", 5, None)] * 3, content_length=10))
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(MalformedResponseError):
        transport.list_objects("p", max_keys=2, max_response_bytes=1000)


def test_list_objects_requires_max_response_bytes(tmp_path: Path) -> None:
    client = FakeS3Client()
    transport = make_transport(client, base_caps(), make_ledger(tmp_path))
    with pytest.raises(ValueError, match="max_response_bytes"):
        transport.list_objects("prefix")
    assert client.calls == []


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_reservations_are_thread_safe(tmp_path: Path) -> None:
    client = FakeS3Client()
    n = 6
    for _ in range(n):
        client.plan("get_object", range_resp(b"x" * 8, start=0, end=7, total=8))
    caps = NetworkCaps(max_requests=n, max_transfer_bytes=8 * n, max_local_bytes=8 * n)
    ledger = make_ledger(tmp_path)
    transport = make_transport(client, caps, ledger)
    barrier = threading.Barrier(n)

    def worker(index: int) -> bytes:
        barrier.wait()
        return transport.get_range(f"k{index}", start=0, end=7).content

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(worker, range(n)))
    assert all(result == b"x" * 8 for result in results)
    counters = transport.counters()
    assert counters.requests == n
    assert counters.transfer_bytes == 8 * n
    assert counters.local_bytes == 8 * n
    records = ledger.records()
    assert len(records) == 2 * n
    assert sum(1 for r in records if r["event"] == "reserve") == n
    assert sum(1 for r in records if r["event"] == "settle") == n
    assert len(client.calls) == n


def test_concurrent_cap_never_overshot(tmp_path: Path) -> None:
    client = FakeS3Client()
    for _ in range(10):
        client.plan("get_object", range_resp(b"x" * 60, start=0, end=59, total=60))
    caps = NetworkCaps(max_requests=10, max_transfer_bytes=100, max_local_bytes=100)
    transport = make_transport(client, caps, make_ledger(tmp_path))
    barrier = threading.Barrier(10)

    def worker(index: int) -> str:
        barrier.wait()
        try:
            transport.get_range(f"k{index}", start=0, end=59)
            return "ok"
        except CapExceededError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(worker, range(10)))
    assert outcomes.count("ok") == 1
    assert outcomes.count("blocked") == 9
    counters = transport.counters()
    assert counters.requests == 1
    assert counters.transfer_bytes == 60
    assert counters.local_bytes == 60
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# Ledger determinism and secret-free records
# ---------------------------------------------------------------------------


def test_ledger_rejects_secrets(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger.append({"Authorization": "Bearer xyz"})
    with pytest.raises(ValueError):
        ledger.append({"data": "AKIAIOSFODNN7EXAMPLE"})
    with pytest.raises(ValueError):
        ledger.append({"signed": "https://bucket/key?X-Amz-Signature=abc123"})
    with pytest.raises(ValueError):
        ledger.append({"nested": {"aws_secret_access_key": "supersecret"}})
    assert ledger.records() == []


def test_transport_records_are_secret_free_after_run(tmp_path: Path) -> None:
    client = FakeS3Client()
    client.plan("get_object", client_error("ServiceUnavailable", 503))
    client.plan("get_object", range_resp(b"0123456789", start=0, end=9, total=100))
    ledger = make_ledger(tmp_path)
    transport = make_transport(client, base_caps(), ledger)
    transport.get_range("k", start=0, end=9)
    text = (tmp_path / "ledger.jsonl").read_text()
    for secret in (
        "AKIA",
        "ASIA",
        "Authorization",
        "X-Amz-Signature",
        "aws_secret",
        "Bearer ",
        "password",
        "credential",
        "token=",
    ):
        assert secret not in text


def test_deterministic_records(tmp_path: Path) -> None:
    def run_scenario(path: Path) -> str:
        client = FakeS3Client()
        client.plan("get_object", client_error("ServiceUnavailable", 503))
        client.plan("get_object", range_resp(b"0123456789", start=0, end=9, total=100))
        client.plan("head_object", resp(200, ContentLength=5, ETag='"e"'))
        client.plan("list_objects_v2", list_resp([("a", 5, None)], content_length=42))
        ledger = MeteredLedger(path, clock=FIXED_CLOCK)
        caps = base_caps()
        transport = make_transport(client, caps, ledger)
        transport.get_range("k", start=0, end=9)
        transport.head_object("k")
        transport.list_objects("p", max_response_bytes=10000)
        return path.read_text()

    first = run_scenario(tmp_path / "a.jsonl")
    second = run_scenario(tmp_path / "b.jsonl")
    assert first == second
    assert_canonical_jsonl(tmp_path / "a.jsonl")


def test_ledger_appends_across_sessions_without_losing_history(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = MeteredLedger(path, clock=FIXED_CLOCK)
    ledger.append({"event": "first"})
    resumed = MeteredLedger(path, clock=FIXED_CLOCK)
    resumed.append({"event": "second"})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["seq"] == 0
    assert json.loads(lines[1])["seq"] == 1


# ---------------------------------------------------------------------------
# SDK retry configuration and input validation
# ---------------------------------------------------------------------------


def test_sdk_implicit_retries_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import boto3

    captured: dict = {}

    def fake_client(service_name: str, **kwargs: object) -> object:
        captured["service"] = service_name
        captured["config"] = kwargs["config"]
        return object()

    monkeypatch.setattr(boto3, "client", fake_client)
    Boto3MeteredTransport(
        "bucket",
        caps=NetworkCaps(),
        ledger=MeteredLedger(tmp_path / "l.jsonl", clock=FIXED_CLOCK),
        rate_card=RATE,
    )
    assert captured["service"] == "s3"
    assert captured["config"].retries == {
        "total_max_attempts": 1,
        "mode": "standard",
    }


def test_invalid_configuration_rejected(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    caps = NetworkCaps(max_requests=1, max_transfer_bytes=10, max_local_bytes=10)
    with pytest.raises(ValueError):
        Boto3MeteredTransport(
            "b",
            client=FakeS3Client(),
            caps=caps,
            ledger=ledger,
            rate_card=RATE,
            max_attempts=0,
        )
    with pytest.raises(ValueError):
        Boto3MeteredTransport(
            "b",
            client=FakeS3Client(),
            caps=caps,
            ledger=ledger,
            rate_card=RATE,
            base_backoff_seconds=0,
        )
    with pytest.raises(ValueError):
        Boto3MeteredTransport(
            "b",
            client=FakeS3Client(),
            caps=caps,
            ledger=ledger,
            rate_card=RATE,
            max_backoff_seconds=0.1,
            base_backoff_seconds=0.5,
        )
    with pytest.raises(ValueError):
        NetworkCaps(max_requests=-1)
    with pytest.raises(ValueError):
        NetworkCaps(
            max_requests=1,
            max_transfer_bytes=1,
            max_local_bytes=1,
            max_requester_pays_usd=Decimal("-1"),
        )
    with pytest.raises(ValueError):
        RateCard(
            source="",
            date="2026-08-10",
            request_cost_usd=Decimal("0"),
            transfer_cost_per_gb_usd=Decimal("0"),
        )
