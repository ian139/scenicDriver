"""Source-specific NAIP + USGS 3DEP ingest workflow: planning, authorization, execution.

This module implements the open-data acquisition contract for Scenic Drive:

- imagery source: USDA NAIP ``naip-visualization`` 3-band RGB COGs (requester pays);
- terrain source: USGS 3DEP 1/3 arc-second seamless elevation (``usgs-3dep-13as``);
- target grid: exact Web Mercator EPSG:3857 z14, 512x512, deterministic RGB PNGs.

It is the planning/execution engine behind ``plan_active_learning_region.py`` and
is deliberately provider-specific: no Mapbox and no unknown billable source is
admissible.  All remote I/O must flow through a ``MeteredTransport``; all raster
reads are local files obtained by that transport; rasterio errors fail closed.

Fail-closed authorization boundaries (zero network by default):

1. Missing local catalog snapshot -> ``discovery_authorization_request.json``
   (proposed operations, null caps, USD, requester-pays policy false).
2. Complete catalogs but no execution authorization -> immutable
   ``execution_plan.json`` + ``execution_plan.sha256`` plus
   ``acquisition_authorization_request.json`` (null caps).
3. ``execute_plan()`` requires the plan path, its expected SHA-256, positive
   request/transfer/local/requester-pays-spend caps and ``allow_requester_pays``.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from src.data_pipeline.metered_transport import (
    Boto3MeteredTransport,
    CapExceededError,
    Counters,
    HeadObjectResult,
    MeteredLedger,
    MeteredTransport,
    NetworkCaps,
    RateCard,
    SharedMeteredBudget,
)
from src.data_pipeline.naip import (
    NaipAdapter,
    calculate_intersection_area,
)
from src.data_pipeline.raster_processing import (
    NoDataOnLandError,
    TargetGrid,
    TerrainRGBRangeError,
    process_dem,
    process_imagery,
    write_atomic_png,
)
from src.data_pipeline.region_planning import parse_geojson_geometry
from src.data_pipeline.source_contracts import (
    SourceAsset,
    compute_acquisition_tile_identity,
)
from src.data_pipeline.usgs_3dep import (
    VALID_ELEVATION_UNITS,
    VALID_VERTICAL_DATUMS,
    load_hash_pinned_catalog,
    validate_3dep_record,
)
from src.data_pipeline.web_mercator import tile_bounds_wgs84

SUPPORTED_IMAGERY_SOURCE = "naip-visualization"
SUPPORTED_TERRAIN_SOURCE = "usgs-3dep-13as"
SUPPORTED_STYLES = ("satellite", "terrain")

PLAN_SCHEMA_VERSION = 1
AUTHORIZATION_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 5

MAX_COORDINATES = 370_000
MAX_RASTERS = 740_000
PILOT_MAX_COORDINATES = 500
PILOT_SELECTION_SEED = 42

# Guard against pathological concurrency; bounded by design.
MAX_WORKERS = 16


def repository_source_tree_digest(root: Path | None = None) -> str:
    """Hash the current tracked and untracked source tree, failing on unreadable files."""
    import subprocess

    repository_root = (
        Path(__file__).resolve().parents[2] if root is None else Path(root).resolve()
    )
    relative_paths: list[str] = []
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        relative_paths = [
            line.strip().replace("\\", "/")
            for line in result.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceContractError(
            "repository source-tree inputs could not be enumerated by Git"
        ) from exc

    def included(relative_path: str) -> bool:
        return not (
            relative_path.startswith((".env", "data/", "logs/", "build/", "dist/"))
            or "/." in relative_path
            or relative_path.endswith((".pyc", ".pyo", ".png", ".jpg", ".bin"))
        )

    paths = sorted(set(filter(included, relative_paths)))
    if not paths:
        raise SourceContractError("repository source-tree digest has no input files")

    hasher = hashlib.sha256()
    for relative_path in paths:
        path = repository_root / relative_path
        hasher.update(f"{relative_path}\n".encode("utf-8"))
        if not path.exists():
            hasher.update(b"<deleted>\n")
            continue
        if not path.is_file():
            raise SourceContractError(
                f"repository source-tree input is not a file: {relative_path}"
            )
        try:
            hasher.update(path.read_bytes())
        except OSError as exc:
            raise SourceContractError(
                f"repository source-tree input is unreadable: {relative_path}"
            ) from exc
    return hasher.hexdigest()


# Declared identity of the NAIP manifest parser that produces normalized
# catalogs; a stable constant binding parser identity (not an observed value,
# and never a per-run fabrication).
NAIP_MANIFEST_PARSER_HASH = hashlib.sha256(
    json.dumps(
        {
            "parser": "naip_manifest_txt",
            "schema_version": 1,
            "derivation": "state/year object paths parsed from manifest.txt lines",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class SourceContractError(ValueError):
    """Invalid or unsupported source-contract configuration."""


class PlanDriftError(ValueError):
    """The recomputed execution plan does not match the immutable plan."""


class OwnershipError(RuntimeError):
    """An output or cache file is owned by another run and must not be touched."""


def _resolve_repository_source_digest(value: str | None) -> str:
    digest = repository_source_tree_digest() if value is None else value
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SourceContractError(
            "repository source-tree digest must be a lowercase SHA-256"
        )
    return digest


# ---------------------------------------------------------------------------
# Canonical serialization and atomic helpers
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    """Deterministic compact JSON bytes (sorted keys, no NaN, UTF-8)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def identity_digest(value: Any) -> str:
    """SHA-256 of the canonical JSON of *value*."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# Fields an authorization artifact carries that must not participate in its own
# digest: ``resume_command`` is a rendered shell command and
# ``authorization_digest`` is the digest itself.  Excluding both makes the
# digest stable, so an artifact can truthfully carry its own digest and a
# hash-bound resume command without a circular self-hash.
_DIGEST_EXCLUDED_AUTHORIZATION_FIELDS = frozenset(
    {"authorization_digest", "resume_command"}
)


def authorization_payload_digest(payload: dict[str, Any]) -> str:
    """Stable SHA-256 of an authorization artifact's payload content.

    Computed over the canonical JSON of *payload* with
    ``authorization_digest`` and ``resume_command`` excluded, so the value is
    stable across renderings of the resume command and is self-consistent when
    stored inside the artifact itself.  Callers MUST recompute and verify this
    digest before constructing a transport or touching the network.
    """
    if not isinstance(payload, dict):
        raise SourceContractError("authorization artifact must be a JSON object")
    content = {
        key: value
        for key, value in payload.items()
        if key not in _DIGEST_EXCLUDED_AUTHORIZATION_FIELDS
    }
    return identity_digest(content)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SourceContractError(f"{label} must be a positive int, got {value!r}")
    return value


def _verify_discovery_authorization_contents(
    *,
    request: dict[str, Any],
    run_name: str,
    contract: dict[str, Any],
    contract_sha: str,
    missing: Sequence[str],
    current_source_digest: str,
    caps: NetworkCaps,
    allow_requester_pays: bool,
) -> None:
    """Validate a discovery authorization artifact's contents before any network.

    Fail-closed checks: schema/type/run_name identity, the immutable
    source-contract hash, the declared missing collections, the operation
    whitelist (provider/bucket/prefix/key and their exact declared bounds),
    requester-pays acknowledgement, and supplied caps covering the declared
    maxima.
    """
    if request.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise SourceContractError(
            f"discovery authorization request has unsupported schema_version "
            f"{request.get('schema_version')!r}"
        )
    if request.get("authorization_type") != "discovery_authorization_request":
        raise SourceContractError(
            f"discovery authorization request has unexpected type "
            f"{request.get('authorization_type')!r}"
        )
    if request.get("run_name") != run_name:
        raise SourceContractError(
            f"discovery authorization request run_name {request.get('run_name')!r} "
            f"does not match run directory {run_name!r}"
        )
    identities = request.get("identities")
    if (
        not isinstance(identities, dict)
        or identities.get("source_contract_sha256") != contract_sha
        or identities.get("repository_source_tree_digest") != current_source_digest
    ):
        raise SourceContractError(
            "discovery authorization request source-contract or source-tree identity "
            "does not match the current immutable inputs"
        )
    if sorted(request.get("missing_catalogs", [])) != sorted(missing):
        raise SourceContractError(
            f"discovery authorization request missing_catalogs "
            f"{sorted(request.get('missing_catalogs', []))} do not match the "
            f"declared missing set {sorted(missing)}"
        )

    required = request.get("required_authorization")
    if not isinstance(required, dict):
        raise SourceContractError(
            "discovery authorization request missing required_authorization"
        )
    declared_requests = _verify_positive_int(
        required.get("max_requests"), "required_authorization.max_requests"
    )
    declared_transfer = _verify_positive_int(
        required.get("max_transfer_bytes"),
        "required_authorization.max_transfer_bytes",
    )
    declared_local = _verify_positive_int(
        required.get("max_local_bytes"), "required_authorization.max_local_bytes"
    )
    if (
        caps.max_requests < declared_requests
        or caps.max_transfer_bytes < declared_transfer
        or caps.max_local_bytes < declared_local
    ):
        raise SourceContractError(
            "supplied network caps are below the discovery authorization request's "
            f"declared maxima (requests {declared_requests}, transfer "
            f"{declared_transfer}, local {declared_local})"
        )

    ops = request.get("proposed_operations")
    if not isinstance(ops, list) or not ops:
        raise SourceContractError(
            "discovery authorization request proposes no operations"
        )
    imagery_bucket = contract["imagery"]["bucket"]
    terrain_bucket = contract["terrain"]["bucket"]
    manifest_key = contract["imagery"]["manifest_key"]
    manifest_bytes = _verify_positive_int(
        contract["imagery"]["manifest_bytes_upper_bound"],
        "imagery.manifest_bytes_upper_bound",
    )
    page_size = _verify_positive_int(
        contract["terrain"]["catalog_page_size"], "terrain.catalog_page_size"
    )
    if page_size > 1000:
        raise SourceContractError(
            "terrain.catalog_page_size exceeds the S3 cap of 1000"
        )
    max_list_pages = math.ceil(
        _verify_positive_int(
            contract["terrain"]["catalog_max_objects"],
            "terrain.catalog_max_objects",
        )
        / page_size
    )
    page_bytes = _verify_positive_int(
        contract["terrain"]["page_response_bytes_upper_bound"],
        "terrain.page_response_bytes_upper_bound",
    )

    saw_imagery = False
    seen_terrain_prefixes: set[str] = set()
    seen_operation_identities: set[tuple[str, str, str]] = set()
    for op in ops:
        if not isinstance(op, dict):
            raise SourceContractError("proposed_operations entries must be objects")
        provider = op.get("provider")
        operation = op.get("operation")
        if provider not in ("usda_naip", "usgs"):
            raise SourceContractError(
                f"discovery authorization request has unknown provider {provider!r}"
            )
        if operation not in ("s3:GetObject", "s3:ListBucket"):
            raise SourceContractError(
                f"discovery authorization request has unknown operation {operation!r}"
            )
        op_requests = _verify_positive_int(op.get("requests"), "op.requests")
        op_bytes = _verify_positive_int(op.get("reserved_bytes"), "op.reserved_bytes")
        operation_identity = (
            str(provider),
            str(operation),
            str(op.get("key") or op.get("prefix") or ""),
        )
        if operation_identity in seen_operation_identities:
            raise SourceContractError(
                f"discovery authorization request duplicates operation "
                f"{operation_identity!r}"
            )
        seen_operation_identities.add(operation_identity)
        if not isinstance(op.get("requester_pays"), bool):
            raise SourceContractError(
                "discovery authorization request operation requester_pays must be a bool"
            )
        if provider == "usda_naip":
            saw_imagery = True
            if operation != "s3:GetObject":
                raise SourceContractError(
                    "NAIP discovery requires exactly one manifest s3:GetObject operation"
                )
            if op.get("bucket") != imagery_bucket:
                raise SourceContractError(
                    f"NAIP discovery operation bucket {op.get('bucket')!r} != {imagery_bucket!r}"
                )
            if op.get("key") != manifest_key:
                raise SourceContractError(
                    f"NAIP discovery operation key {op.get('key')!r} != declared manifest_key {manifest_key!r}"
                )
            if op_requests != 1 or op_bytes != manifest_bytes:
                raise SourceContractError(
                    "NAIP manifest operation requests/bytes do not match declared "
                    "manifest bounds (1 request, "
                    f"{manifest_bytes} bytes)"
                )
            if op.get("requester_pays") is not True:
                raise SourceContractError(
                    "NAIP manifest operation must acknowledge requester_pays"
                )
        else:
            prefix = op.get("prefix")
            seen_terrain_prefixes.add(str(prefix))
            if operation != "s3:ListBucket":
                raise SourceContractError(
                    "3DEP discovery requires an s3:ListBucket listing operation"
                )
            if op.get("bucket") != terrain_bucket:
                raise SourceContractError(
                    f"3DEP discovery operation bucket {op.get('bucket')!r} != {terrain_bucket!r}"
                )
            if not isinstance(prefix, str) or prefix not in contract["terrain"].get(
                "discovery_prefixes", []
            ):
                raise SourceContractError(
                    f"3DEP discovery operation prefix {prefix!r} is not a declared "
                    "terrain.discovery_prefixes entry"
                )
            if op_requests != max_list_pages or op_bytes != page_bytes:
                raise SourceContractError(
                    "3DEP listing operation requests/bytes do not match the declared "
                    f"conservative page ceiling (max_pages={max_list_pages}, "
                    f"page bytes={page_bytes})"
                )
            if op.get("page_size") != page_size:
                raise SourceContractError(
                    f"3DEP listing operation page_size {op.get('page_size')!r} != declared {page_size}"
                )
            if op.get("requester_pays") is not False:
                raise SourceContractError(
                    "3DEP listing operation must not declare requester_pays"
                )

    expected_terrain_prefixes = (
        set(contract["terrain"]["discovery_prefixes"])
        if "terrain" in missing
        else set()
    )
    if seen_terrain_prefixes != expected_terrain_prefixes:
        raise SourceContractError(
            "discovery authorization request operations do not cover exactly the "
            f"declared terrain prefixes: {sorted(seen_terrain_prefixes)} != "
            f"{sorted(expected_terrain_prefixes)}"
        )
    if ("imagery" in missing) != saw_imagery:
        raise SourceContractError(
            "discovery authorization request operations do not cover exactly the "
            "declared missing catalogs"
        )
    if required.get("allow_requester_pays") is not saw_imagery:
        raise SourceContractError(
            "discovery authorization request requester-pays acknowledgement does not "
            "match its operations"
        )
    computed_requests = sum(int(op["requests"]) for op in ops)
    computed_transfer = sum(
        int(op["requests"]) * int(op["reserved_bytes"]) for op in ops
    )
    if (
        declared_requests != computed_requests
        or declared_transfer != computed_transfer
        or declared_local != computed_transfer
    ):
        raise SourceContractError(
            "discovery authorization request declared maxima do not match its "
            "enumerated operations"
        )
    rate_card = contract["rate_card"]
    computed_cost = Decimal(str(rate_card["request_cost_usd"])) * Decimal(
        computed_requests
    ) + Decimal(str(rate_card["transfer_cost_per_gb_usd"])) * (
        Decimal(computed_transfer) / Decimal(10**9)
    )
    expected_spend = str(
        computed_cost.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    )
    if required.get("max_spend_usd") != expected_spend:
        raise SourceContractError(
            "discovery authorization request max_spend_usd does not match its "
            "enumerated operations and pinned rate card"
        )
    if "imagery" in missing:
        if not allow_requester_pays:
            raise SourceContractError(
                "discovery of NAIP imagery requires explicit allow_requester_pays=True"
            )
        if not caps.allow_requester_pays or caps.max_requester_pays_usd < computed_cost:
            raise SourceContractError(
                "discovery of NAIP imagery requires requester-pays caps covering "
                f"the declared conservative maximum {computed_cost:.6f} USD"
            )


def discover_raw_catalogs(
    *,
    discovery_request_path: str | Path,
    expected_discovery_request_sha256: str,
    contract_path: str | Path,
    missing: Sequence[str],
    caps: NetworkCaps,
    allow_requester_pays: bool,
    run_dir: str | Path,
    transport_factory: Callable[..., MeteredTransport] | None = None,
) -> dict[str, Any]:
    """Boundary 1: Execute raw catalog discovery over metered transport and emit metadata authorization request.

    Verifies the discovery request payload digest BEFORE any network call,
    validates the authorization contents against the immutable source contract,
    fetches the raw NAIP manifest and the complete bounded-pagination 3DEP
    listing over the metered transport, preserves raw bytes atomically, and
    emits ``catalog_metadata_authorization_request.json`` (which carries its own
    stable digest and a hash-bound resume command).  Does NOT write strict
    normalized catalog files or mutate the source contract.
    """
    req_path = Path(discovery_request_path)
    if not req_path.is_file():
        raise SourceContractError(
            f"discovery authorization request file not found: {req_path}"
        )
    try:
        request = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceContractError(
            f"discovery authorization request is not valid JSON: {exc}"
        ) from exc

    computed_req_sha = authorization_payload_digest(request)
    if computed_req_sha.lower() != expected_discovery_request_sha256.lower():
        raise SourceContractError(
            f"discovery authorization request digest mismatch before network call: "
            f"computed {computed_req_sha}, expected {expected_discovery_request_sha256}"
        )

    if not missing:
        return {}

    contract, contract_sha = load_source_contract(contract_path)
    current_source_digest = repository_source_tree_digest()
    run_p = Path(run_dir)
    _verify_discovery_authorization_contents(
        request=request,
        run_name=run_p.name,
        contract=contract,
        contract_sha=contract_sha,
        missing=missing,
        current_source_digest=current_source_digest,
        caps=caps,
        allow_requester_pays=allow_requester_pays,
    )

    if transport_factory is None:
        transport_factory = _default_transport_factory

    rate_card = RateCard(
        source=contract["rate_card"]["source"],
        date=contract["rate_card"]["date"],
        request_cost_usd=Decimal(contract["rate_card"]["request_cost_usd"]),
        transfer_cost_per_gb_usd=Decimal(
            contract["rate_card"]["transfer_cost_per_gb_usd"]
        ),
    )
    ledger_path = run_p / "discovery_ledger.jsonl"
    ledger = MeteredLedger(ledger_path)
    shared_budget = SharedMeteredBudget(caps)

    candidate_naip_objects: list[dict[str, Any]] = []
    candidate_3dep_objects: list[dict[str, Any]] = []

    if "imagery" in missing:
        imagery = contract["imagery"]
        bucket = imagery["bucket"]
        manifest_key = imagery["manifest_key"]
        manifest_bound = _verify_positive_int(
            imagery["manifest_bytes_upper_bound"],
            "imagery.manifest_bytes_upper_bound",
        )
        imagery_transport = transport_factory(
            bucket,
            caps,
            ledger,
            rate_card,
            True,
            shared_budget,
        )
        raw_manifest_bytes = imagery_transport.get_object(
            manifest_key, max_response_bytes=manifest_bound
        )
        if not isinstance(raw_manifest_bytes, bytes) or not raw_manifest_bytes:
            raise SourceContractError(
                f"NAIP discovery failed: empty {manifest_key!r} response"
            )
        if len(raw_manifest_bytes) > manifest_bound:
            raise SourceContractError(
                f"NAIP manifest exceeds declared manifest_bytes_upper_bound "
                f"({len(raw_manifest_bytes)} > {manifest_bound})"
            )

        raw_manifest_path = run_p / "raw_naip_manifest.txt"
        atomic_write_bytes(raw_manifest_path, raw_manifest_bytes)

        try:
            manifest_text = raw_manifest_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceContractError(
                f"NAIP {manifest_key!r} is not valid UTF-8"
            ) from exc
        allowed_states = set(s.lower() for s in imagery.get("allowed_states", []))

        for line in manifest_text.splitlines():
            line = line.strip()
            if not line:
                continue
            subpath = line
            if subpath.startswith("s3://"):
                subpath = subpath.split("/", 3)[-1]
            parts = [p for p in subpath.split("/") if p]
            if len(parts) < 2:
                continue
            st = parts[0].lower()
            if allowed_states and st not in allowed_states:
                continue
            try:
                yr = int(parts[1])
            except ValueError as exc:
                raise SourceContractError(
                    f"NAIP object path missing integer year: '{line}'"
                ) from exc

            canonical_uri = (
                f"s3://{bucket}/{subpath}"
                if not subpath.startswith("s3://")
                else subpath
            )
            candidate_naip_objects.append(
                {
                    "provider": "usda_naip",
                    "collection": imagery["collection"],
                    "bucket": bucket,
                    "key": subpath,
                    "canonical_uri": canonical_uri,
                    "state": st,
                    "acquisition_year": yr,
                }
            )

    if "terrain" in missing:
        terrain = contract["terrain"]
        bucket = terrain["bucket"]
        terrain_transport = transport_factory(
            bucket,
            caps,
            ledger,
            rate_card,
            False,
            shared_budget,
        )
        prefix = terrain["discovery_prefixes"][0]
        page_size = _verify_positive_int(
            terrain["catalog_page_size"], "terrain.catalog_page_size"
        )
        page_response_bound = _verify_positive_int(
            terrain["page_response_bytes_upper_bound"],
            "terrain.page_response_bytes_upper_bound",
        )
        max_objects = _verify_positive_int(
            terrain["catalog_max_objects"], "terrain.catalog_max_objects"
        )
        max_pages = math.ceil(max_objects / page_size)

        all_objects: list[Any] = []
        token: str | None = None
        page_count = 0
        while True:
            page_count += 1
            if page_count > max_pages:
                raise SourceContractError(
                    "3DEP listing exceeds the declared catalog page bound: "
                    f"{page_count} pages needed beyond max_pages={max_pages} "
                    f"(catalog_max_objects={max_objects}, "
                    f"catalog_page_size={page_size}); truncated listing cannot be "
                    "authorized — raise terrain.catalog_max_objects and re-discover"
                )
            list_res = terrain_transport.list_objects(
                prefix=prefix,
                max_keys=page_size,
                continuation_token=token,
                max_response_bytes=page_response_bound,
            )
            if not hasattr(list_res, "objects"):
                raise SourceContractError(
                    "3DEP listing transport returned an unrecognized response shape"
                )
            page_objects = list_res.objects
            all_objects.extend(page_objects)
            if len(all_objects) > max_objects:
                raise SourceContractError(
                    "3DEP listing exceeds declared terrain.catalog_max_objects "
                    f"({len(all_objects)} > {max_objects}); truncated listing cannot "
                    "be authorized — raise the bound and re-discover"
                )
            if not list_res.is_truncated or not list_res.next_continuation_token:
                break
            token = list_res.next_continuation_token

        raw_listing_keys = [
            getattr(obj, "key", obj if isinstance(obj, str) else str(obj))
            for obj in all_objects
        ]
        raw_listing_bytes = canonical_json_bytes(raw_listing_keys) + b"\n"
        raw_listing_path = run_p / "raw_3dep_listing.json"
        atomic_write_bytes(raw_listing_path, raw_listing_bytes)
        listing_key_set = set(raw_listing_keys)

        for obj in all_objects:
            key = getattr(obj, "key", obj if isinstance(obj, str) else str(obj))
            if not isinstance(key, str) or not key or not key.endswith(".tif"):
                continue
            filename = Path(key).name
            asset_id = filename.rsplit(".", 1)[0]
            if not asset_id.startswith("USGS_13_"):
                continue
            match = re.search(r"([ns])(\d+)([ew])(\d+)", key, re.IGNORECASE)
            if not match:
                raise SourceContractError(
                    f"3DEP key '{key}' missing nXXwYYY tile coordinate pattern"
                )
            ns, lat_str, ew, lon_str = match.groups()
            lat_val = float(lat_str) * (1 if ns.lower() == "n" else -1)
            lon_val = float(lon_str) * (-1 if ew.lower() == "w" else 1)
            quad_bbox = [lon_val, lat_val - 1.0, lon_val + 1.0, lat_val]

            companion_key = None
            candidate_companion = f"{key.rsplit('.', 1)[0]}.xml"
            if candidate_companion in listing_key_set:
                companion_key = candidate_companion

            canonical_uri = (
                f"s3://{bucket}/{key}" if not key.startswith("s3://") else key
            )
            candidate_3dep_objects.append(
                {
                    "provider": "usgs",
                    "collection": terrain["collection"],
                    "bucket": bucket,
                    "key": key,
                    "canonical_uri": canonical_uri,
                    "asset_id": asset_id,
                    "quad_bbox": quad_bbox,
                    "companion_metadata_key": companion_key,
                }
            )

    meta_auth = build_catalog_metadata_authorization_request(
        run_name=run_p.name,
        contract=contract,
        contract_sha=contract_sha,
        naip_candidates=candidate_naip_objects,
        tnm_candidates=candidate_3dep_objects,
        raw_naip_manifest_sha256=(
            file_sha256(run_p / "raw_naip_manifest.txt")
            if candidate_naip_objects
            else None
        ),
        raw_3dep_listing_sha256=(
            file_sha256(run_p / "raw_3dep_listing.json")
            if candidate_3dep_objects
            else None
        ),
        resume_command="",
        repository_source_tree_digest=current_source_digest,
    )
    meta_auth_sha = authorization_payload_digest(meta_auth)
    meta_auth_path = run_p / "catalog_metadata_authorization_request.json"
    req_auth = meta_auth["required_authorization"]
    meta_resume_cmd = (
        "uv run python scripts/ingest/plan_active_learning_region.py "
        f"--run-name {run_p.name} "
        f"--discover-catalog-metadata "
        f"--metadata-request {meta_auth_path} "
        f"--expected-metadata-request-sha256 {meta_auth_sha} "
        f"--source-contract {contract_path} "
        " --allow-requester-pays "
        f"--max-source-requests {req_auth['max_requests']} "
        f"--max-transfer-bytes {req_auth['max_transfer_bytes']} "
        f"--max-local-bytes {req_auth['max_local_bytes']} "
        f"--max-requester-pays-usd {req_auth['max_spend_usd']}"
    )
    meta_auth["resume_command"] = meta_resume_cmd
    atomic_write_json(meta_auth_path, meta_auth)

    return {
        "state": "needs_metadata_discovery",
        "metadata_authorization_path": str(meta_auth_path),
        "metadata_authorization_sha256": meta_auth_sha,
    }


def build_catalog_metadata_authorization_request(
    *,
    run_name: str,
    contract: dict[str, Any],
    contract_sha: str,
    naip_candidates: list[dict[str, Any]],
    tnm_candidates: list[dict[str, Any]],
    raw_naip_manifest_sha256: str | None,
    raw_3dep_listing_sha256: str | None,
    resume_command: str,
    repository_source_tree_digest: str | None = None,
) -> dict[str, Any]:
    """Boundary 2 authorization request artifact.

    Enumerates every HEAD and whole-object GET needed to derive strict catalog
    metadata, with per-operation upper bounds: one HEAD (etag / size /
    last-modified / Accept-Ranges observation) plus one whole-object GET
    (etag-bound, size-capped) per tile candidate, and one explicit GET per
    companion metadata object (3DEP per-quad XML) that the raw listing
    preserved.  Also binds the exact immutable source-contract SHA and the raw
    artifact hashes the normalized catalogs will reference.  Producing this
    document makes no network call.
    """
    repository_source_tree_digest = _resolve_repository_source_digest(
        repository_source_tree_digest
    )
    rate_card = contract["rate_card"]
    req_cost_dec = Decimal(str(rate_card["request_cost_usd"]))
    trans_cost_dec = Decimal(str(rate_card["transfer_cost_per_gb_usd"]))

    imagery_asset_bytes = _verify_positive_int(
        contract["imagery"]["estimates"]["per_asset_bytes_upper_bound"],
        "imagery.estimates.per_asset_bytes_upper_bound",
    )
    terrain_asset_bytes = _verify_positive_int(
        contract["terrain"]["estimates"]["per_asset_bytes_upper_bound"],
        "terrain.estimates.per_asset_bytes_upper_bound",
    )
    companion_bound = _verify_positive_int(
        contract["terrain"]["companion_metadata_bytes_upper_bound"],
        "terrain.companion_metadata_bytes_upper_bound",
    )

    proposed_operations: list[dict[str, Any]] = []
    missing_catalogs: list[str] = []

    if naip_candidates:
        missing_catalogs.append("imagery")
        for cand in naip_candidates:
            key = cand["key"]
            proposed_operations.append(
                {
                    "provider": "usda_naip",
                    "collection": contract["imagery"]["collection"],
                    "bucket": contract["imagery"]["bucket"],
                    "operation": "s3:HeadObject",
                    "key": key,
                    "target": "object_metadata",
                    "requests": 1,
                    "reserved_bytes": 0,
                    "note": "HEAD responses carry no billed body bytes; the reservation covers the request count only",
                    "requester_pays": True,
                }
            )
            proposed_operations.append(
                {
                    "provider": "usda_naip",
                    "collection": contract["imagery"]["collection"],
                    "bucket": contract["imagery"]["bucket"],
                    "operation": "s3:GetObject",
                    "key": key,
                    "target": "object_bytes",
                    "requests": 1,
                    "reserved_bytes": imagery_asset_bytes,
                    "expected_etag_binding": True,
                    "requester_pays": True,
                }
            )

    if tnm_candidates:
        missing_catalogs.append("terrain")
        seen_companions: set[str] = set()
        for cand in tnm_candidates:
            key = cand["key"]
            proposed_operations.append(
                {
                    "provider": "usgs",
                    "collection": contract["terrain"]["collection"],
                    "bucket": contract["terrain"]["bucket"],
                    "operation": "s3:HeadObject",
                    "key": key,
                    "target": "object_metadata",
                    "requests": 1,
                    "reserved_bytes": 0,
                    "note": "HEAD responses carry no billed body bytes; the reservation covers the request count only",
                    "requester_pays": False,
                }
            )
            proposed_operations.append(
                {
                    "provider": "usgs",
                    "collection": contract["terrain"]["collection"],
                    "bucket": contract["terrain"]["bucket"],
                    "operation": "s3:GetObject",
                    "key": key,
                    "target": "object_bytes",
                    "requests": 1,
                    "reserved_bytes": terrain_asset_bytes,
                    "expected_etag_binding": True,
                    "requester_pays": False,
                }
            )
            companion_key = cand.get("companion_metadata_key")
            if companion_key and companion_key not in seen_companions:
                seen_companions.add(companion_key)
                proposed_operations.append(
                    {
                        "provider": "usgs",
                        "collection": contract["terrain"]["collection"],
                        "bucket": contract["terrain"]["bucket"],
                        "operation": "s3:GetObject",
                        "key": companion_key,
                        "target": "companion_metadata",
                        "requests": 1,
                        "reserved_bytes": companion_bound,
                        "requester_pays": False,
                    }
                )

    if not proposed_operations:
        raise SourceContractError(
            "catalog metadata authorization request proposes no operations"
        )

    total_requests = sum(int(op["requests"]) for op in proposed_operations)
    total_transfer_bytes = sum(
        int(op["reserved_bytes"]) * int(op["requests"]) for op in proposed_operations
    )
    total_local_bytes = total_transfer_bytes
    requires_requester_pays = any(
        bool(op.get("requester_pays")) for op in proposed_operations
    )

    req_cost = req_cost_dec * Decimal(total_requests)
    gb_transferred = Decimal(total_transfer_bytes) / Decimal(10**9)
    trans_cost = trans_cost_dec * gb_transferred
    total_cost = req_cost + trans_cost
    max_spend_usd = str(
        total_cost.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    )

    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_type": "catalog_metadata_authorization_request",
        "run_name": run_name,
        "missing_catalogs": sorted(missing_catalogs),
        "proposed_operations": proposed_operations,
        "per_operation_byte_reservation": max(
            int(op["reserved_bytes"]) for op in proposed_operations
        ),
        "required_authorization": {
            "currency": "USD",
            "allow_requester_pays": requires_requester_pays,
            "max_requests": total_requests,
            "max_transfer_bytes": total_transfer_bytes,
            "max_local_bytes": total_local_bytes,
            "max_spend_usd": max_spend_usd,
        },
        "policy": {
            "currency": "USD",
            "requester_pays": requires_requester_pays,
            "network_call_made": False,
            "paid_source_allowed": False,
            "rate_card_source": rate_card["source"],
            "rate_card_date": rate_card["date"],
            "note": "This document authorizes enumerated HEAD/whole-object GET observations; execution requires explicit positive caps and --allow-requester-pays for the NAIP bucket.",
        },
        "rate_card": {
            "source": rate_card["source"],
            "date": rate_card["date"],
            "request_cost_usd": rate_card["request_cost_usd"],
            "transfer_cost_per_gb_usd": rate_card["transfer_cost_per_gb_usd"],
        },
        "identities": {
            "source_contract_sha256": contract_sha,
            "raw_naip_manifest_sha256": raw_naip_manifest_sha256,
            "raw_3dep_listing_sha256": raw_3dep_listing_sha256,
            "repository_source_tree_digest": repository_source_tree_digest,
        },
        "has_secrets": False,
        "resume_command": resume_command,
    }
    payload["authorization_digest"] = authorization_payload_digest(payload)
    return payload


def _verify_metadata_authorization_contents(
    *,
    request: dict[str, Any],
    run_name: str,
    contract: dict[str, Any],
    contract_sha: str,
    current_source_digest: str,
    caps: NetworkCaps,
    allow_requester_pays: bool,
) -> None:
    """Validate a catalog-metadata authorization artifact before any network.

    Fail-closed checks: schema/type/run_name identity, the immutable
    source-contract hash, operation whitelist with exact HEAD/GET pairing and
    bucket/key shapes, requester-pays acknowledgement, raw artifact hash
    declarations, the declared missing collections, and supplied caps covering
    the declared maxima.
    """
    if request.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise SourceContractError(
            f"catalog metadata authorization request has unsupported schema_version "
            f"{request.get('schema_version')!r}"
        )
    if request.get("authorization_type") != "catalog_metadata_authorization_request":
        raise SourceContractError(
            f"catalog metadata authorization request has unexpected type "
            f"{request.get('authorization_type')!r}"
        )
    if request.get("run_name") != run_name:
        raise SourceContractError(
            f"catalog metadata authorization request run_name {request.get('run_name')!r} "
            f"does not match run directory {run_name!r}"
        )
    identities = request.get("identities")
    if (
        not isinstance(identities, dict)
        or identities.get("source_contract_sha256") != contract_sha
        or identities.get("repository_source_tree_digest") != current_source_digest
    ):
        raise SourceContractError(
            "catalog metadata authorization request source-contract or source-tree "
            "identity does not match the current immutable inputs"
        )

    required = request.get("required_authorization")
    if not isinstance(required, dict):
        raise SourceContractError(
            "catalog metadata authorization request missing required_authorization"
        )
    declared_requests = _verify_positive_int(
        required.get("max_requests"), "required_authorization.max_requests"
    )
    declared_transfer = _verify_positive_int(
        required.get("max_transfer_bytes"),
        "required_authorization.max_transfer_bytes",
    )
    declared_local = _verify_positive_int(
        required.get("max_local_bytes"), "required_authorization.max_local_bytes"
    )
    if (
        caps.max_requests < declared_requests
        or caps.max_transfer_bytes < declared_transfer
        or caps.max_local_bytes < declared_local
    ):
        raise SourceContractError(
            "supplied network caps are below the catalog metadata authorization "
            f"request's declared maxima (requests {declared_requests}, transfer "
            f"{declared_transfer}, local {declared_local})"
        )

    ops = request.get("proposed_operations")
    if not isinstance(ops, list) or not ops:
        raise SourceContractError(
            "catalog metadata authorization request proposes no operations"
        )
    imagery_bucket = contract["imagery"]["bucket"]
    terrain_bucket = contract["terrain"]["bucket"]
    allowed_states = set(s.lower() for s in contract["imagery"]["allowed_states"])
    terrain_prefix = contract["terrain"]["discovery_prefixes"][0]

    naip_keys: set[str] = set()
    tnm_keys: set[str] = set()
    head_keys: set[str] = set()
    get_keys: set[str] = set()
    xml_keys: set[str] = set()
    seen_operations: set[tuple[str, str, str]] = set()
    imagery_asset_bound = _verify_positive_int(
        contract["imagery"]["estimates"]["per_asset_bytes_upper_bound"],
        "imagery.estimates.per_asset_bytes_upper_bound",
    )
    terrain_asset_bound = _verify_positive_int(
        contract["terrain"]["estimates"]["per_asset_bytes_upper_bound"],
        "terrain.estimates.per_asset_bytes_upper_bound",
    )
    companion_bound = _verify_positive_int(
        contract["terrain"]["companion_metadata_bytes_upper_bound"],
        "terrain.companion_metadata_bytes_upper_bound",
    )
    for op in ops:
        if not isinstance(op, dict):
            raise SourceContractError("proposed_operations entries must be objects")
        provider = op.get("provider")
        operation = op.get("operation")
        if provider not in ("usda_naip", "usgs"):
            raise SourceContractError(
                f"catalog metadata authorization request has unknown provider {provider!r}"
            )
        if operation not in ("s3:HeadObject", "s3:GetObject"):
            raise SourceContractError(
                f"catalog metadata authorization request has unknown operation {operation!r}"
            )
        key = op.get("key")
        if not isinstance(key, str) or not key:
            raise SourceContractError(
                "catalog metadata authorization request operation missing key"
            )
        operation_identity = (str(provider), str(operation), key)
        if operation_identity in seen_operations:
            raise SourceContractError(
                f"catalog metadata authorization request duplicates operation "
                f"{operation_identity!r}"
            )
        seen_operations.add(operation_identity)
        if op.get("requests") != 1:
            raise SourceContractError(
                "catalog metadata authorization operations must declare exactly one request"
            )
        reserved = op.get("reserved_bytes")
        if not isinstance(reserved, int) or isinstance(reserved, bool) or reserved < 0:
            raise SourceContractError(
                "catalog metadata authorization operation reserved_bytes must be a "
                "non-negative int"
            )
        if not isinstance(op.get("requester_pays"), bool):
            raise SourceContractError(
                "catalog metadata authorization operation requester_pays must be a bool"
            )
        if provider == "usda_naip":
            if op.get("collection") != contract["imagery"]["collection"]:
                raise SourceContractError(
                    "NAIP metadata operation collection does not match the source contract"
                )
            if not key.lower().endswith((".tif", ".tiff")):
                raise SourceContractError(
                    f"NAIP metadata operation key '{key}' is not a TIFF object"
                )
            naip_keys.add(key)
            if op.get("bucket") != imagery_bucket:
                raise SourceContractError(
                    f"NAIP metadata operation bucket {op.get('bucket')!r} != {imagery_bucket!r}"
                )
            if op.get("requester_pays") is not True:
                raise SourceContractError(
                    "NAIP metadata operations must acknowledge requester_pays"
                )
        else:
            tnm_keys.add(key)
            if op.get("collection") != contract["terrain"]["collection"]:
                raise SourceContractError(
                    "3DEP metadata operation collection does not match the source contract"
                )
            if op.get("bucket") != terrain_bucket:
                raise SourceContractError(
                    f"3DEP metadata operation bucket {op.get('bucket')!r} != {terrain_bucket!r}"
                )
            if op.get("requester_pays") is not False:
                raise SourceContractError(
                    "3DEP metadata operations must not declare requester_pays"
                )
        if operation == "s3:HeadObject":
            if reserved != 0 or op.get("target") != "object_metadata":
                raise SourceContractError(
                    "catalog metadata HEAD operations must target object_metadata "
                    "with zero reserved bytes"
                )
            head_keys.add(key)
        else:
            get_keys.add(key)
            if key.endswith(".xml"):
                xml_keys.add(key)
                if (
                    provider != "usgs"
                    or reserved != companion_bound
                    or op.get("target") != "companion_metadata"
                ):
                    raise SourceContractError(
                        "3DEP companion metadata GET does not match the declared "
                        "operation contract"
                    )
            else:
                expected_bound = (
                    imagery_asset_bound
                    if provider == "usda_naip"
                    else terrain_asset_bound
                )
                if (
                    reserved != expected_bound
                    or op.get("target") != "object_bytes"
                    or op.get("expected_etag_binding") is not True
                ):
                    raise SourceContractError(
                        "catalog object GET does not match the declared byte bound, "
                        "target, and ETag-binding contract"
                    )
    # Every tile object must have exactly one HEAD and one whole-object GET;
    # companion metadata objects carry a GET only.
    if head_keys != get_keys - xml_keys:
        raise SourceContractError(
            "catalog metadata authorization request must pair exactly one HEAD with "
            "each whole-object GET (companion metadata objects are GET-only)"
        )
    for key in naip_keys:
        parts = [p for p in key.split("/") if p]
        if len(parts) < 2 or parts[0].lower() not in allowed_states:
            raise SourceContractError(
                f"NAIP metadata operation key '{key}' is not an allowed state/year object"
            )
        try:
            int(parts[1])
        except ValueError as exc:
            raise SourceContractError(
                f"NAIP metadata operation key '{key}' has a non-integer year"
            ) from exc
    for key in tnm_keys:
        if not key.startswith(terrain_prefix):
            raise SourceContractError(
                f"3DEP metadata operation key '{key}' is outside the declared discovery prefix"
            )
        if not key.rsplit("/", 1)[-1].startswith("USGS_13_"):
            raise SourceContractError(
                f"3DEP metadata operation key '{key}' is not a USGS_13_* object"
            )

    expected_missing = sorted(
        (["imagery"] if naip_keys else []) + (["terrain"] if tnm_keys else [])
    )
    if sorted(request.get("missing_catalogs", [])) != expected_missing:
        raise SourceContractError(
            "catalog metadata authorization request missing_catalogs do not match "
            "its operations"
        )
    if required.get("allow_requester_pays") is not (bool(naip_keys)):
        raise SourceContractError(
            "catalog metadata authorization request requester-pays acknowledgement "
            "does not match its operations"
        )
    computed_requests = len(ops)
    computed_transfer = sum(int(op["reserved_bytes"]) for op in ops)
    if (
        declared_requests != computed_requests
        or declared_transfer != computed_transfer
        or declared_local != computed_transfer
    ):
        raise SourceContractError(
            "catalog metadata authorization declared maxima do not match its "
            "enumerated operations"
        )
    rate_card = contract["rate_card"]
    computed_cost = Decimal(str(rate_card["request_cost_usd"])) * Decimal(
        computed_requests
    ) + Decimal(str(rate_card["transfer_cost_per_gb_usd"])) * (
        Decimal(computed_transfer) / Decimal(10**9)
    )
    expected_spend = str(
        computed_cost.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    )
    if required.get("max_spend_usd") != expected_spend:
        raise SourceContractError(
            "catalog metadata authorization max_spend_usd does not match its "
            "enumerated operations and pinned rate card"
        )
    if naip_keys:
        if not allow_requester_pays:
            raise SourceContractError(
                "NAIP metadata observation requires explicit allow_requester_pays=True"
            )
        if not caps.allow_requester_pays or caps.max_requester_pays_usd < computed_cost:
            raise SourceContractError(
                "NAIP metadata observation requires requester-pays caps covering "
                f"the declared conservative maximum {computed_cost:.6f} USD"
            )
    for name, present in (
        ("raw_naip_manifest_sha256", bool(naip_keys)),
        ("raw_3dep_listing_sha256", bool(tnm_keys)),
    ):
        value = identities.get(name)
        if present and (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
        ):
            raise SourceContractError(
                f"catalog metadata authorization request missing or invalid {name}"
            )


def _observed_wgs84_footprint(crs: Any, bounds: Any, label: str) -> dict[str, Any]:
    """Transform observed raster bounds into a WGS84 GeoJSON Polygon footprint."""
    from rasterio.warp import transform_bounds

    try:
        min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, "EPSG:4326", *bounds)
    except Exception as exc:
        raise SourceContractError(
            f"{label} bounds could not be transformed to WGS84: {exc}"
        ) from exc
    for value in (min_lon, min_lat, max_lon, max_lat):
        if not math.isfinite(float(value)):
            raise SourceContractError(
                f"{label} produced non-finite WGS84 footprint bounds"
            )
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


_NAIP_CAPTURE_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def _parse_naip_capture_date(key: str, acquisition_year: int) -> str:
    """Require an unambiguous NAIP capture date from the object filename.

    NAIP visualization filenames end with an 8-digit YYYYMMDD capture date.
    The LAST 8-digit run in the filename must parse as a real calendar date
    whose year equals the acquisition year from the object path; any other
    outcome is ambiguous and the record fails closed (no default dates).
    """
    filename = key.rsplit("/", 1)[-1]
    matches = _NAIP_CAPTURE_DATE_RE.findall(filename)
    if not matches:
        raise SourceContractError(
            f"NAIP key '{key}' has no 8-digit capture date in its filename; capture "
            "date must come from the object path, never a default"
        )
    date_str = matches[-1]
    try:
        parsed = datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError as exc:
        raise SourceContractError(
            f"NAIP key '{key}' capture date '{date_str}' is not a valid calendar date"
        ) from exc
    if parsed.year != acquisition_year:
        raise SourceContractError(
            f"NAIP key '{key}' capture date year {parsed.year} does not match the "
            f"path acquisition year {acquisition_year}"
        )
    return parsed.isoformat()


def _extract_3dep_vertical(
    tags: dict[str, Any], companion_xml_text: str | None
) -> tuple[str, str] | None:
    """Observe 3DEP vertical datum/units from actual metadata, or return None.

    Checks GeoTIFF tags first, then the companion per-quad XML (FGDC CSDGM
    elevation/altitude system elements with a narrow value-token fallback).
    Returns a normalized (datum, units) pair; ``None`` means the caller must
    fail closed rather than fabricate values.
    """
    datum: str | None = None
    units: str | None = None
    for tag_key, value in tags.items():
        lowered = tag_key.lower()
        text = str(value).strip()
        if not text:
            continue
        normalized = " ".join(text.split()).upper()
        if normalized in VALID_VERTICAL_DATUMS and (
            "datum" in lowered or "vert" in lowered or "elev" in lowered
        ):
            datum = normalized
    for tag_key, value in tags.items():
        lowered = tag_key.lower()
        text = str(value).strip()
        if not text:
            continue
        if "unit" in lowered and text.lower() in VALID_ELEVATION_UNITS:
            units = text.lower()
    if datum and units:
        return datum, units

    if companion_xml_text:
        try:
            root = ET.fromstring(companion_xml_text)
        except Exception as exc:
            raise SourceContractError(
                f"companion 3DEP metadata XML is malformed: {exc}"
            ) from exc
        for elem in root.iter():
            tag = elem.tag.rsplit("}", 1)[-1].lower()
            text = (elem.text or "").strip()
            if not text:
                continue
            if (
                "datum" in tag
                and re.search(r"elev|alt|vert", tag) is not None
                and datum is None
            ):
                normalized = " ".join(text.split()).upper()
                if normalized in VALID_VERTICAL_DATUMS:
                    datum = normalized
            elif (
                "unit" in tag
                and re.search(r"elev|alt|vert", tag) is not None
                and units is None
            ):
                lowered = text.lower()
                if lowered in VALID_ELEVATION_UNITS:
                    units = lowered
        if datum and units:
            return datum, units
        # Narrow value-token fallback over the actual XML text (still an
        # observation of the fetched metadata, never a default).
        if datum is None:
            match = re.search(
                r"NAVD\s*88|NAVD88_HEIGHT|EPSG:\s*5703",
                companion_xml_text,
                re.IGNORECASE,
            )
            if match:
                datum = " ".join(match.group(0).split()).upper()
        if units is None:
            match = re.search(r"\bmeters?\b", companion_xml_text, re.IGNORECASE)
            if match:
                units = match.group(0).lower()
        if datum and units:
            return datum, units
    return None


def _observe_naip_asset_record(
    *,
    contract: dict[str, Any],
    imagery: dict[str, Any],
    key: str,
    head: HeadObjectResult,
    content_bytes: bytes,
    content_sha256: str,
) -> dict[str, Any]:
    """Derive a strict NAIP catalog record purely from observed bytes/metadata."""
    from rasterio.io import MemoryFile

    parts = [p for p in key.split("/") if p]
    if len(parts) < 2:
        raise SourceContractError(
            f"NAIP key '{key}' does not follow state/year pattern"
        )
    st = parts[0].lower()
    try:
        yr = int(parts[1])
    except ValueError as exc:
        raise SourceContractError(
            f"NAIP key '{key}' has a non-integer acquisition year"
        ) from exc
    allowed_states = set(s.lower() for s in contract["imagery"]["allowed_states"])
    if st not in allowed_states:
        raise SourceContractError(
            f"NAIP key '{key}' state '{st}' is not in the contract allowed_states"
        )
    canonical_uri = f"s3://{imagery['bucket']}/{key}"

    if not head.etag:
        raise SourceContractError(
            f"NAIP object '{key}' HEAD returned no ETag; refusing fabricated identity"
        )
    if not head.last_modified:
        raise SourceContractError(
            f"NAIP object '{key}' HEAD returned no Last-Modified; refusing a default"
        )
    if not isinstance(head.size, int) or head.size <= 0:
        raise SourceContractError(f"NAIP object '{key}' HEAD returned no positive size")
    if len(content_bytes) != head.size:
        raise SourceContractError(
            f"NAIP object '{key}' GET body length {len(content_bytes)} != HEAD size {head.size}"
        )
    if head.accept_ranges is not True:
        raise SourceContractError(
            f"NAIP object '{key}' HEAD did not confirm byte-range support (Accept-Ranges)"
        )

    capture_date = _parse_naip_capture_date(key, yr)

    try:
        with MemoryFile(content_bytes) as memory_file:
            with memory_file.open() as dataset:
                crs = dataset.crs
                if crs is None:
                    raise SourceContractError(
                        f"NAIP object '{key}' GeoTIFF header has no CRS"
                    )
                bounds = dataset.bounds
                resolution_x = float(abs(dataset.res[0]))
                resolution_y = float(abs(dataset.res[1]))
                if (
                    not math.isfinite(resolution_x)
                    or not math.isfinite(resolution_y)
                    or resolution_x <= 0
                    or resolution_y <= 0
                    or not math.isclose(
                        resolution_x, resolution_y, rel_tol=1e-9, abs_tol=1e-12
                    )
                ):
                    raise SourceContractError(
                        f"NAIP object '{key}' GeoTIFF header has no positive square-pixel resolution"
                    )
                native_resolution = resolution_x
                if dataset.profile.get("tiled") is not True:
                    raise SourceContractError(
                        f"NAIP object '{key}' is not an observed tiled GeoTIFF/COG"
                    )
                colorinterp = [c.name.lower() for c in dataset.colorinterp]
                if dataset.count != 3 or colorinterp != ["red", "green", "blue"]:
                    raise SourceContractError(
                        f"NAIP object '{key}' is not an observed 3-band RGB GeoTIFF "
                        f"(count={dataset.count}, colorinterp={colorinterp})"
                    )
                if any(dtype != "uint8" for dtype in dataset.dtypes):
                    raise SourceContractError(
                        f"NAIP object '{key}' is not an observed 8-bit visualization COG"
                    )
                nodata = dataset.nodata
    except SourceContractError:
        raise
    except Exception as exc:
        raise SourceContractError(
            f"NAIP object '{key}' is not a parseable GeoTIFF: {exc}"
        ) from exc

    footprint_geo = _observed_wgs84_footprint(crs, bounds, f"NAIP object '{key}'")
    metadata_sha256 = identity_digest(
        {
            "canonical_uri": canonical_uri,
            "object_size_bytes": head.size,
            "etag": head.etag,
            "last_modified": head.last_modified,
            "capture_date": capture_date,
            "horizontal_crs": str(crs),
            "native_resolution": native_resolution,
            "band_contract": colorinterp,
            "accept_ranges": True,
            "footprint": footprint_geo,
        }
    )
    record = {
        "provider": "usda_naip",
        "collection": imagery["collection"],
        "asset_id": key,
        "canonical_uri": canonical_uri,
        "key": canonical_uri,
        "state": st,
        "state_or_region": st,
        "acquisition_year": yr,
        "capture_date": capture_date,
        "license": contract["imagery"]["license"],
        "attribution": contract["imagery"]["attribution"],
        "horizontal_crs": str(crs),
        "native_resolution": native_resolution,
        "band_contract": colorinterp,
        "metadata_sha256": metadata_sha256,
        "etag": head.etag,
        "checksum_sha256": content_sha256,
        "object_size_bytes": head.size,
        "last_modified": head.last_modified,
        "accept_ranges": True,
        "footprint": footprint_geo,
        "footprint_geojson": json.dumps(
            footprint_geo, sort_keys=True, separators=(",", ":")
        ),
    }
    if nodata is not None:
        record["nodata"] = nodata
    return record


def _observe_3dep_asset_record(
    *,
    contract: dict[str, Any],
    terrain: dict[str, Any],
    key: str,
    head: HeadObjectResult,
    content_bytes: bytes,
    content_sha256: str,
    companion_xml_text: str | None,
) -> dict[str, Any]:
    """Derive a strict 3DEP catalog record purely from observed bytes/metadata."""
    from rasterio.io import MemoryFile

    if re.search(r"([ns])(\d+)([ew])(\d+)", key, re.IGNORECASE) is None:
        raise SourceContractError(
            f"3DEP key '{key}' missing nXXwYYY tile coordinate pattern"
        )
    filename = Path(key).name
    asset_id = filename.rsplit(".", 1)[0]
    if not asset_id.startswith("USGS_13_"):
        raise SourceContractError(f"3DEP key '{key}' is not a USGS_13_* tile object")
    canonical_uri = f"s3://{terrain['bucket']}/{key}"

    if not head.etag:
        raise SourceContractError(
            f"3DEP object '{key}' HEAD returned no ETag; refusing fabricated identity"
        )
    if not head.last_modified:
        raise SourceContractError(
            f"3DEP object '{key}' HEAD returned no Last-Modified; refusing a default"
        )
    if not isinstance(head.size, int) or head.size <= 0:
        raise SourceContractError(f"3DEP object '{key}' HEAD returned no positive size")
    if len(content_bytes) != head.size:
        raise SourceContractError(
            f"3DEP object '{key}' GET body length {len(content_bytes)} != HEAD size {head.size}"
        )
    if head.accept_ranges is not True:
        raise SourceContractError(
            f"3DEP object '{key}' HEAD did not confirm byte-range support (Accept-Ranges)"
        )

    try:
        with MemoryFile(content_bytes) as memory_file:
            with memory_file.open() as dataset:
                crs = dataset.crs
                if crs is None:
                    raise SourceContractError(
                        f"3DEP object '{key}' GeoTIFF header has no CRS"
                    )
                bounds = dataset.bounds
                resolution_x = float(dataset.res[0])
                resolution_y = float(dataset.res[1])
                if not (
                    math.isfinite(resolution_x)
                    and math.isfinite(resolution_y)
                    and resolution_x != 0
                    and resolution_y != 0
                ):
                    raise SourceContractError(
                        f"3DEP object '{key}' GeoTIFF header has no finite resolution"
                    )
                if dataset.profile.get("tiled") is not True:
                    raise SourceContractError(
                        f"3DEP object '{key}' is not an observed tiled GeoTIFF/COG"
                    )
                nodata = dataset.nodata
                if nodata is None or not math.isfinite(float(nodata)):
                    raise SourceContractError(
                        f"3DEP object '{key}' GeoTIFF header has no finite nodata value"
                    )
                tags = dict(dataset.tags())
    except SourceContractError:
        raise
    except Exception as exc:
        raise SourceContractError(
            f"3DEP object '{key}' is not a parseable GeoTIFF: {exc}"
        ) from exc

    vertical = _extract_3dep_vertical(tags, companion_xml_text)
    if vertical is None:
        raise SourceContractError(
            f"3DEP object '{key}' vertical datum/units were not observed from the "
            "GeoTIFF header or its companion metadata; refusing fabricated values"
        )
    datum, units = vertical

    footprint_geo = _observed_wgs84_footprint(crs, bounds, f"3DEP object '{key}'")
    metadata_sha256 = identity_digest(
        {
            "canonical_uri": canonical_uri,
            "object_size_bytes": head.size,
            "etag": head.etag,
            "last_modified": head.last_modified,
            "horizontal_crs": str(crs),
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "nodata": float(nodata),
            "vertical_datum": datum,
            "elevation_unit": units,
            "accept_ranges": True,
            "footprint": footprint_geo,
        }
    )
    return {
        "provider": "USGS",
        "collection": terrain["collection"],
        "asset_id": asset_id,
        "canonical_uri": canonical_uri,
        "uri": canonical_uri,
        "license": contract["terrain"]["license"],
        "attribution": contract["terrain"]["attribution"],
        "horizontal_crs": str(crs),
        "crs": str(crs),
        "vertical_datum": datum,
        "vdatum": datum,
        "elevation_unit": units,
        "units": units,
        "nodata": float(nodata),
        "native_resolution": [resolution_x, resolution_y],
        "resolution_x": resolution_x,
        "resolution_y": resolution_y,
        "metadata_sha256": metadata_sha256,
        "cog_header_observed": True,
        "header_observed": True,
        "object_size_bytes": head.size,
        "last_modified": head.last_modified,
        "accept_ranges": True,
        "etag": head.etag,
        "checksum_sha256": content_sha256,
        "footprint_geojson": json.dumps(
            footprint_geo, sort_keys=True, separators=(",", ":")
        ),
        "footprint": footprint_geo,
    }


def _build_resolved_contract(
    *,
    contract: dict[str, Any],
    contract_sha: str,
    imagery_sha: str | None,
    terrain_sha: str | None,
) -> dict[str, Any]:
    """A publishable source-contract copy recording discovered catalog hashes.

    Never mutates the tracked input contract: the resolved copy is written
    under the data root, keeps the tracked catalog paths, records the actual
    catalog file hashes, and preserves the original immutable input hash.
    """
    resolved = json.loads(canonical_json_bytes(contract).decode("utf-8"))
    for section, sha in (("imagery", imagery_sha), ("terrain", terrain_sha)):
        if sha is not None:
            resolved[section]["catalog_sha256"] = sha
        else:
            prior_sha = contract[section].get("catalog_sha256")
            if (
                not isinstance(prior_sha, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", prior_sha) is None
            ):
                raise SourceContractError(
                    f"cannot resolve {section} catalog without an observed new "
                    "catalog or an existing hash-pinned contract"
                )
            resolved[section]["catalog_sha256"] = prior_sha.lower()
    resolved["resolved_from_source_contract_sha256"] = contract_sha
    return resolved


def _staged_temp_path(final: Path) -> Path:
    """Create an empty sibling temp path for atomic staging (caller writes it)."""
    final.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{final.name}.", suffix=".tmp", dir=final.parent
    )
    os.close(descriptor)
    return Path(temp_name)


def discover_catalog_metadata(
    *,
    metadata_request_path: str | Path,
    expected_metadata_request_sha256: str,
    contract_path: str | Path,
    run_dir: str | Path,
    caps: NetworkCaps,
    allow_requester_pays: bool,
    transport_factory: Callable[..., MeteredTransport] | None = None,
) -> dict[str, Any]:
    """Boundary 2: observe strict catalog metadata and publish normalized catalogs.

    Verifies the metadata request payload digest and its contents BEFORE any
    network call or transport construction, then performs, per object: a
    metered HEAD (etag / size / Last-Modified / Accept-Ranges), a whole-object
    GET bound to the observed ETag, a byte-length equality check, and a
    rasterio.MemoryFile parse of the local bytes for CRS/bounds/resolution/
    bands/nodata/tags.  The WGS84 footprint is derived from the observed
    source bounds/CRS.  NAIP capture dates must come from the object path;
    3DEP vertical datum/units/nodata must be observed (companion XML fetched
    through the metered transport when present) or the run fails closed.

    Normalized snapshots are built only after every candidate record
    validates, staged to temp paths, strict-loader validated
    (load_naip_catalog + NaipAdapter / load_3dep_catalog), and only then
    atomically published together with a resolved source-contract copy under
    the data root.  The tracked source contract is never mutated; any partial
    failure leaves prior strict catalog files and the tracked contract
    unchanged.
    """
    req_path = Path(metadata_request_path)
    if not req_path.is_file():
        raise SourceContractError(
            f"metadata authorization request file not found: {req_path}"
        )
    try:
        request = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceContractError(
            f"metadata authorization request is not valid JSON: {exc}"
        ) from exc

    computed_req_sha = authorization_payload_digest(request)
    if computed_req_sha.lower() != expected_metadata_request_sha256.lower():
        raise SourceContractError(
            f"metadata authorization request digest mismatch before network call: "
            f"computed {computed_req_sha}, expected {expected_metadata_request_sha256}"
        )

    contract, contract_sha = load_source_contract(contract_path)
    current_source_digest = repository_source_tree_digest()
    run_p = Path(run_dir)
    _verify_metadata_authorization_contents(
        request=request,
        run_name=run_p.name,
        contract=contract,
        contract_sha=contract_sha,
        current_source_digest=current_source_digest,
        caps=caps,
        allow_requester_pays=allow_requester_pays,
    )

    if transport_factory is None:
        transport_factory = _default_transport_factory

    rate_card = RateCard(
        source=contract["rate_card"]["source"],
        date=contract["rate_card"]["date"],
        request_cost_usd=Decimal(contract["rate_card"]["request_cost_usd"]),
        transfer_cost_per_gb_usd=Decimal(
            contract["rate_card"]["transfer_cost_per_gb_usd"]
        ),
    )
    ledger_path = run_p / "discovery_ledger.jsonl"
    ledger = MeteredLedger(ledger_path)
    shared_budget = SharedMeteredBudget(caps)

    ops = request["proposed_operations"]
    naip_get_ops = [
        op
        for op in ops
        if op["provider"] == "usda_naip" and op["operation"] == "s3:GetObject"
    ]
    tnm_get_ops = [
        op
        for op in ops
        if op["provider"] == "usgs" and op["operation"] == "s3:GetObject"
    ]
    naip_keys = sorted(op["key"] for op in naip_get_ops)
    tnm_tile_keys = sorted(
        op["key"] for op in tnm_get_ops if op["key"].endswith(".tif")
    )
    tnm_xml_ops = [op for op in tnm_get_ops if op["key"].endswith(".xml")]

    # Bind Boundary 2 to Boundary 1's raw bytes before any network call.
    if naip_keys:
        raw_manifest_path = run_p / "raw_naip_manifest.txt"
        if not raw_manifest_path.is_file():
            raise SourceContractError(
                f"raw NAIP manifest missing at {raw_manifest_path}; Boundary 1 must "
                "run first in the same run directory"
            )
        declared = request["identities"]["raw_naip_manifest_sha256"]
        if file_sha256(raw_manifest_path).lower() != declared.lower():
            raise SourceContractError(
                "raw NAIP manifest hash does not match the metadata authorization "
                "request identity"
            )
    if tnm_tile_keys:
        raw_listing_path = run_p / "raw_3dep_listing.json"
        if not raw_listing_path.is_file():
            raise SourceContractError(
                f"raw 3DEP listing missing at {raw_listing_path}; Boundary 1 must "
                "run first in the same run directory"
            )
        declared = request["identities"]["raw_3dep_listing_sha256"]
        if file_sha256(raw_listing_path).lower() != declared.lower():
            raise SourceContractError(
                "raw 3DEP listing hash does not match the metadata authorization "
                "request identity"
            )

    validated_naip_assets: list[dict[str, Any]] = []
    validated_3dep_records: list[dict[str, Any]] = []

    if naip_keys:
        imagery = contract["imagery"]
        imagery_transport = transport_factory(
            imagery["bucket"], caps, ledger, rate_card, True, shared_budget
        )
        get_op_by_key = {op["key"]: op for op in naip_get_ops}
        for key in naip_keys:
            head = imagery_transport.head_object(key)
            content = imagery_transport.get_object(
                key,
                max_response_bytes=get_op_by_key[key]["reserved_bytes"],
                expected_etag=head.etag,
                expected_version_id=head.version_id,
            )
            if len(content) != head.size:
                raise SourceContractError(
                    f"NAIP object '{key}' GET body length {len(content)} != HEAD size {head.size}"
                )
            content_sha = hashlib.sha256(content).hexdigest()
            validated_naip_assets.append(
                _observe_naip_asset_record(
                    contract=contract,
                    imagery=imagery,
                    key=key,
                    head=head,
                    content_bytes=content,
                    content_sha256=content_sha,
                )
            )
        validated_naip_assets.sort(key=lambda record: record["canonical_uri"])

    if tnm_tile_keys:
        terrain = contract["terrain"]
        terrain_transport = transport_factory(
            terrain["bucket"], caps, ledger, rate_card, False, shared_budget
        )
        xml_text_by_key: dict[str, str] = {}
        for op in tnm_xml_ops:
            content = terrain_transport.get_object(
                op["key"], max_response_bytes=op["reserved_bytes"]
            )
            try:
                xml_text_by_key[op["key"]] = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SourceContractError(
                    f"3DEP companion metadata {op['key']!r} is not valid UTF-8"
                ) from exc
        get_op_by_key = {op["key"]: op for op in tnm_get_ops}
        for key in tnm_tile_keys:
            head = terrain_transport.head_object(key)
            content = terrain_transport.get_object(
                key,
                max_response_bytes=get_op_by_key[key]["reserved_bytes"],
                expected_etag=head.etag,
                expected_version_id=head.version_id,
            )
            if len(content) != head.size:
                raise SourceContractError(
                    f"3DEP object '{key}' GET body length {len(content)} != HEAD size {head.size}"
                )
            content_sha = hashlib.sha256(content).hexdigest()
            companion_key = f"{key.rsplit('.', 1)[0]}.xml"
            validated_3dep_records.append(
                _observe_3dep_asset_record(
                    contract=contract,
                    terrain=terrain,
                    key=key,
                    head=head,
                    content_bytes=content,
                    content_sha256=content_sha,
                    companion_xml_text=xml_text_by_key.get(companion_key),
                )
            )
        validated_3dep_records.sort(
            key=lambda record: (record["asset_id"], record["canonical_uri"])
        )

    if naip_keys and not validated_naip_assets:
        raise SourceContractError(
            "NAIP metadata discovery returned zero valid asset records"
        )
    if tnm_tile_keys and not validated_3dep_records:
        raise SourceContractError(
            "3DEP metadata discovery returned zero valid asset records"
        )

    # ------------------------------------------------------------------
    # Publish: stage both normalized catalogs + resolved contract, strict-
    # loader validate every staged artifact, then atomically replace.  Any
    # failure before the replace loop leaves prior strict catalog files and
    # the tracked source contract byte-for-byte unchanged.
    # ------------------------------------------------------------------
    staged_pairs: list[tuple[Path, Path]] = []
    temps: list[Path] = []
    try:
        imagery_sha: str | None = None
        terrain_sha: str | None = None

        if naip_keys:
            imagery = contract["imagery"]
            allowed_states = sorted(s.lower() for s in imagery["allowed_states"])
            normalized_naip = {
                "collection": imagery["collection"],
                "bucket": imagery["bucket"],
                "region": imagery.get("region", "us-west-2"),
                "requester_pays": True,
                "raw_hash": request["identities"]["raw_naip_manifest_sha256"],
                "catalog_hash": identity_digest(validated_naip_assets),
                "parser_hash": NAIP_MANIFEST_PARSER_HASH,
                "region_hash": identity_digest(allowed_states),
                "assets": validated_naip_assets,
            }
            final_naip = Path(imagery["catalog_snapshot"])
            naip_bytes = canonical_json_bytes(normalized_naip) + b"\n"
            imagery_sha = hashlib.sha256(naip_bytes).hexdigest()
            temp_naip = _staged_temp_path(final_naip)
            temps.append(temp_naip)
            atomic_write_bytes(temp_naip, naip_bytes)
            if load_naip_catalog(temp_naip, expected_sha256=imagery_sha) is None:
                raise SourceContractError(
                    f"staged NAIP catalog snapshot {final_naip} failed strict validation"
                )
            try:
                NaipAdapter(json.loads(naip_bytes.decode("utf-8")))
            except Exception as exc:
                raise SourceContractError(
                    f"staged NAIP catalog snapshot {final_naip} failed adapter validation: {exc}"
                ) from exc
            staged_pairs.append((temp_naip, final_naip))

        if tnm_tile_keys:
            terrain = contract["terrain"]
            normalized_3dep = {
                "records": validated_3dep_records,
                "raw_hash": request["identities"]["raw_3dep_listing_sha256"],
            }
            final_tnm = Path(terrain["catalog_snapshot"])
            tnm_bytes = canonical_json_bytes(normalized_3dep) + b"\n"
            terrain_sha = hashlib.sha256(tnm_bytes).hexdigest()
            temp_tnm = _staged_temp_path(final_tnm)
            temps.append(temp_tnm)
            atomic_write_bytes(temp_tnm, tnm_bytes)
            if load_3dep_catalog(temp_tnm, expected_sha256=terrain_sha) is None:
                raise SourceContractError(
                    f"staged 3DEP catalog snapshot {final_tnm} failed strict validation"
                )
            staged_pairs.append((temp_tnm, final_tnm))

        resolved = _build_resolved_contract(
            contract=contract,
            contract_sha=contract_sha,
            imagery_sha=imagery_sha,
            terrain_sha=terrain_sha,
        )
        resolved_final = (
            Path(contract["imagery"]["catalog_snapshot"]).parent
            / f"{contract['contract_id']}.resolved.json"
        )
        resolved_bytes = canonical_json_bytes(resolved) + b"\n"
        temp_resolved = _staged_temp_path(resolved_final)
        temps.append(temp_resolved)
        atomic_write_bytes(temp_resolved, resolved_bytes)
        # The resolved copy must itself be a valid source contract.
        load_source_contract(temp_resolved)
        staged_pairs.append((temp_resolved, resolved_final))

        originals = {
            final: (final.read_bytes() if final.is_file() else None)
            for _, final in staged_pairs
        }
        committed: list[Path] = []
        try:
            # Publish the resolved contract last: it is the commit marker consumers
            # use to bind the complete normalized catalog set.
            for temp, final in staged_pairs:
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp, final)
                committed.append(final)
        except BaseException:
            # Restore every earlier destination if any later replace fails. This
            # keeps an ordinary partial publication failure from changing the
            # previously usable catalog set.
            for final in reversed(committed):
                previous = originals[final]
                if previous is None:
                    final.unlink(missing_ok=True)
                else:
                    atomic_write_bytes(final, previous)
            raise
    finally:
        for temp in temps:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass

    return {
        "imagery_catalog_sha256": imagery_sha,
        "terrain_catalog_sha256": terrain_sha,
        "resolved_contract_path": str(resolved_final),
        "resolved_contract_sha256": file_sha256(resolved_final),
        "source_contract_sha256": contract_sha,
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one deterministic compact JSON line; never rewrites history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json_bytes(record) + b"\n"
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# Tile manifest contract (scalar CSV columns; nested provenance in JSONL)
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "region",
    "z",
    "x",
    "y",
    "lat",
    "lon",
    "image_path",
    "satellite_path",
    "terrain_path",
    "satellite_present",
    "terrain_present",
    "satellite_water_fraction",
    "terrain_sea_level_fraction",
    "effective_water_fraction",
    "water_filter_status",
    "unusable_reason",
    "admission_reason",
    "land_fraction",
    "state",
    "source_contract_sha256",
    "preprocessing_contract_sha256",
    "boundary_geometry_sha256",
    "grid_sha256",
    "acquisition_tile_identity_sha256",
    "satellite_provider",
    "satellite_collection",
    "satellite_asset_ids",
    "satellite_acquisition_year",
    "satellite_capture_date",
    "satellite_license",
    "satellite_attribution",
    "satellite_source_checksums",
    "satellite_output_sha256",
    "satellite_valid_fraction",
    "terrain_provider",
    "terrain_collection",
    "terrain_asset_ids",
    "terrain_vertical_datum",
    "terrain_native_resolution",
    "terrain_license",
    "terrain_attribution",
    "terrain_source_checksums",
    "terrain_output_sha256",
    "terrain_valid_fraction",
    "mosaic_contributions",
    "processing_version",
]


def write_manifest_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Deterministically write the tile manifest with canonical column order."""
    import csv as _csv
    import io as _io

    output = _io.StringIO(newline="")
    writer = _csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {column: row.get(column, "") for column in MANIFEST_COLUMNS} for row in rows
    )
    atomic_write_text(path, output.getvalue())


# ---------------------------------------------------------------------------
# Source-contract configuration
# ---------------------------------------------------------------------------


def load_source_contract(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and validate the source-contract JSON; returns (contract, sha256)."""
    contract_path = Path(path)
    if not contract_path.is_file():
        raise SourceContractError(f"source contract not found: {contract_path}")
    raw = contract_path.read_bytes()
    try:
        contract = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise SourceContractError(
            f"invalid source contract JSON {contract_path}: {exc}"
        ) from exc
    if not isinstance(contract, dict):
        raise SourceContractError("source contract must be a JSON object")

    if contract.get("schema_version") != 1:
        raise SourceContractError(
            f"unsupported source-contract schema version: {contract.get('schema_version')}"
        )
    if not contract.get("contract_id"):
        raise SourceContractError("source contract missing contract_id")

    imagery = contract.get("imagery")
    terrain = contract.get("terrain")
    if not isinstance(imagery, dict) or not isinstance(terrain, dict):
        raise SourceContractError(
            "source contract requires 'imagery' and 'terrain' sections"
        )

    for section_name, section in (("imagery", imagery), ("terrain", terrain)):
        for field in ("provider", "collection", "bucket", "catalog_snapshot"):
            if not isinstance(section.get(field), str) or not section[field].strip():
                raise SourceContractError(
                    f"source contract {section_name} missing non-empty '{field}'"
                )
        if section.get("collection") == SUPPORTED_IMAGERY_SOURCE:
            if section.get("requester_pays") is not True:
                raise SourceContractError(
                    f"imagery collection must declare requester_pays=True, got "
                    f"{section.get('requester_pays')!r}"
                )
        elif section.get("collection") == SUPPORTED_TERRAIN_SOURCE:
            if section.get("requester_pays") is not False:
                raise SourceContractError(
                    f"terrain collection must declare requester_pays=False, got "
                    f"{section.get('requester_pays')!r}"
                )
        else:
            raise SourceContractError(
                f"unsupported collection {section.get('collection')!r} in source contract; "
                f"only {SUPPORTED_IMAGERY_SOURCE} and {SUPPORTED_TERRAIN_SOURCE} are admissible"
            )
        estimates = section.get("estimates")
        if not isinstance(estimates, dict):
            raise SourceContractError(
                f"source contract {section_name} missing 'estimates'"
            )
        for field in ("per_asset_requests", "per_asset_bytes_upper_bound"):
            value = estimates.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SourceContractError(
                    f"source contract {section_name} estimates.{field} must be a positive int"
                )

    # Explicit discovery bounds: the authorization arithmetic derives exact
    # conservative operation counts from these, so they must be declared and
    # validated here rather than guessed at execution time.
    imagery_manifest_key = imagery.get("manifest_key")
    if not isinstance(imagery_manifest_key, str) or not imagery_manifest_key.strip():
        raise SourceContractError(
            "source contract imagery.manifest_key must be a non-empty object key"
        )
    imagery_manifest_bytes = imagery.get("manifest_bytes_upper_bound")
    if (
        not isinstance(imagery_manifest_bytes, int)
        or isinstance(imagery_manifest_bytes, bool)
        or imagery_manifest_bytes <= 0
    ):
        raise SourceContractError(
            "source contract imagery.manifest_bytes_upper_bound must be a positive int"
        )

    terrain_catalog_max_objects = terrain.get("catalog_max_objects")
    if (
        not isinstance(terrain_catalog_max_objects, int)
        or isinstance(terrain_catalog_max_objects, bool)
        or terrain_catalog_max_objects <= 0
    ):
        raise SourceContractError(
            "source contract terrain.catalog_max_objects must be a positive int"
        )
    terrain_catalog_page_size = terrain.get("catalog_page_size")
    if (
        not isinstance(terrain_catalog_page_size, int)
        or isinstance(terrain_catalog_page_size, bool)
        or not 1 <= terrain_catalog_page_size <= 1000
    ):
        raise SourceContractError(
            "source contract terrain.catalog_page_size must be an int in [1, 1000] "
            "(S3 ListObjectsV2 MaxKeys cap)"
        )
    terrain_page_bytes = terrain.get("page_response_bytes_upper_bound")
    if (
        not isinstance(terrain_page_bytes, int)
        or isinstance(terrain_page_bytes, bool)
        or terrain_page_bytes <= 0
    ):
        raise SourceContractError(
            "source contract terrain.page_response_bytes_upper_bound must be a positive int"
        )
    terrain_companion_bytes = terrain.get("companion_metadata_bytes_upper_bound")
    if (
        not isinstance(terrain_companion_bytes, int)
        or isinstance(terrain_companion_bytes, bool)
        or terrain_companion_bytes <= 0
    ):
        raise SourceContractError(
            "source contract terrain.companion_metadata_bytes_upper_bound must be a positive int"
        )

    allowed_states = imagery.get("allowed_states")
    discovery_prefixes = imagery.get("discovery_prefixes")
    if (
        not isinstance(allowed_states, list)
        or not allowed_states
        or any(
            not isinstance(state, str) or re.fullmatch(r"[a-z]{2}", state) is None
            for state in allowed_states
        )
        or len(set(allowed_states)) != len(allowed_states)
    ):
        raise SourceContractError("imagery.allowed_states must be unique state codes")
    expected_prefixes = [f"{state}/" for state in allowed_states]
    if discovery_prefixes != expected_prefixes:
        raise SourceContractError(
            "imagery.discovery_prefixes must exactly match allowed_states"
        )
    preprocessing = contract.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise SourceContractError("source contract missing 'preprocessing'")
    if preprocessing.get("zoom") != 14:
        raise SourceContractError("preprocessing zoom must be exactly 14")
    if (preprocessing.get("tile_width"), preprocessing.get("tile_height")) != (
        512,
        512,
    ):
        raise SourceContractError("preprocessing tile size must be exactly 512x512")
    if preprocessing.get("target_crs") != "EPSG:3857":
        raise SourceContractError("preprocessing target_crs must be EPSG:3857")
    if preprocessing.get("target_vertical_datum") != "NAVD88":
        raise SourceContractError("target_vertical_datum must be NAVD88")
    if preprocessing.get("target_vertical_crs") != "EPSG:5703":
        raise SourceContractError("target_vertical_crs must be EPSG:5703")
    if preprocessing.get("vertical_transform_grid") is not None:
        raise SourceContractError(
            "vertical_transform_grid must be null when source and target are NAVD88"
        )
    satellite_qc = preprocessing.get("satellite_qc")
    if not isinstance(satellite_qc, dict):
        raise SourceContractError("preprocessing.satellite_qc is required")
    min_variance = satellite_qc.get("min_land_variance")
    if (
        not isinstance(min_variance, (int, float))
        or isinstance(min_variance, bool)
        or not math.isfinite(float(min_variance))
        or float(min_variance) <= 0
    ):
        raise SourceContractError(
            "preprocessing.satellite_qc.min_land_variance must be positive and finite"
        )
    if satellite_qc.get("reject_all_black") is not True:
        raise SourceContractError("satellite_qc.reject_all_black must be true")
    if satellite_qc.get("reject_all_white") is not True:
        raise SourceContractError("satellite_qc.reject_all_white must be true")

    terrain_discovery_prefixes = terrain.get("discovery_prefixes")
    if (
        not isinstance(terrain_discovery_prefixes, list)
        or not terrain_discovery_prefixes
        or any(
            not isinstance(p, str) or not p.strip() for p in terrain_discovery_prefixes
        )
    ):
        raise SourceContractError(
            "terrain.discovery_prefixes must be a non-empty list of non-empty prefix strings"
        )

    rate_card = contract.get("rate_card")
    if not isinstance(rate_card, dict):
        raise SourceContractError("source contract missing 'rate_card'")
    for field in ("source", "date", "request_cost_usd", "transfer_cost_per_gb_usd"):
        value = rate_card.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SourceContractError(f"rate_card.{field} must be a non-empty string")
    try:
        req_cost_dec = Decimal(rate_card["request_cost_usd"])
        if req_cost_dec <= Decimal("0") or not req_cost_dec.is_finite():
            raise ValueError()
    except Exception as exc:
        raise SourceContractError(
            f"rate_card.request_cost_usd must be a positive finite decimal string, got {rate_card.get('request_cost_usd')!r}"
        ) from exc
    try:
        trans_cost_dec = Decimal(rate_card["transfer_cost_per_gb_usd"])
        if trans_cost_dec < Decimal("0") or not trans_cost_dec.is_finite():
            raise ValueError()
    except Exception as exc:
        raise SourceContractError(
            f"rate_card.transfer_cost_per_gb_usd must be a non-negative finite decimal string, got {rate_card.get('transfer_cost_per_gb_usd')!r}"
        ) from exc
    limits = contract.get("limits", {})
    if limits.get("max_coordinates") != MAX_COORDINATES:
        raise SourceContractError(
            f"source contract limits.max_coordinates must be {MAX_COORDINATES}"
        )
    if limits.get("max_rasters") != MAX_RASTERS:
        raise SourceContractError(
            f"source contract limits.max_rasters must be {MAX_RASTERS}"
        )

    contract_sha = hashlib.sha256(raw).hexdigest()
    return contract, contract_sha


def build_discovery_authorization_request(
    *,
    run_name: str,
    contract: dict[str, Any],
    contract_sha: str,
    region_spec_sha: str,
    geometry_digest: str,
    coordinate_count: int,
    jurisdictions: Sequence[str],
    resume_command: str,
    missing: Sequence[str],
    target_bbox: tuple[float, float, float, float] | None,
    repository_source_tree_digest: str | None = None,
) -> dict[str, Any]:
    """Unified discovery-authorization artifact for missing catalog snapshots.

    Provides exact catalog discovery operations, per-operation request and byte caps,
    and aggregate caps sufficient for discovery based on the source contract.
    No network call is made by this document or by producing it.
    """
    repository_source_tree_digest = _resolve_repository_source_digest(
        repository_source_tree_digest
    )
    if not missing:
        raise SourceContractError(
            "missing catalog list must not be empty for discovery authorization"
        )

    rate_card = contract.get("rate_card")
    if not isinstance(rate_card, dict):
        raise SourceContractError("source contract missing 'rate_card'")
    try:
        req_cost_dec = Decimal(str(rate_card.get("request_cost_usd", "")))
        if req_cost_dec <= Decimal("0") or not req_cost_dec.is_finite():
            raise ValueError()
    except Exception as exc:
        raise SourceContractError(
            f"rate_card.request_cost_usd must be a positive finite decimal string, got {rate_card.get('request_cost_usd')!r}"
        ) from exc
    try:
        trans_cost_dec = Decimal(str(rate_card.get("transfer_cost_per_gb_usd", "")))
        if trans_cost_dec < Decimal("0") or not trans_cost_dec.is_finite():
            raise ValueError()
    except Exception as exc:
        raise SourceContractError(
            f"rate_card.transfer_cost_per_gb_usd must be a non-negative finite decimal string, got {rate_card.get('transfer_cost_per_gb_usd')!r}"
        ) from exc

    proposed_operations: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []
    per_op_reservations: list[int] = []

    if "imagery" in missing:
        imagery = contract.get("imagery")
        if not isinstance(imagery, dict):
            raise SourceContractError("source contract missing 'imagery' section")
        estimates = imagery.get("estimates")
        if not isinstance(estimates, dict):
            raise SourceContractError("imagery missing 'estimates'")
        per_asset_reqs = estimates.get("per_asset_requests")
        per_asset_bytes = estimates.get("per_asset_bytes_upper_bound")
        if (
            not isinstance(per_asset_reqs, int)
            or isinstance(per_asset_reqs, bool)
            or per_asset_reqs <= 0
        ):
            raise SourceContractError(
                "imagery.estimates.per_asset_requests must be a positive int"
            )
        if (
            not isinstance(per_asset_bytes, int)
            or isinstance(per_asset_bytes, bool)
            or per_asset_bytes <= 0
        ):
            raise SourceContractError(
                "imagery.estimates.per_asset_bytes_upper_bound must be a positive int"
            )

        discovery_prefixes = imagery.get("discovery_prefixes")
        if not isinstance(discovery_prefixes, list) or not discovery_prefixes:
            raise SourceContractError(
                "imagery.discovery_prefixes must be a non-empty list"
            )
        manifest_key = imagery.get("manifest_key")
        manifest_bytes = imagery.get("manifest_bytes_upper_bound")
        if not isinstance(manifest_key, str) or not manifest_key:
            raise SourceContractError("imagery.manifest_key must be a non-empty string")
        if (
            not isinstance(manifest_bytes, int)
            or isinstance(manifest_bytes, bool)
            or manifest_bytes <= 0
        ):
            raise SourceContractError(
                "imagery.manifest_bytes_upper_bound must be a positive int"
            )
        # The NAIP catalog manifest is a single object; exactly one GET is
        # proposed (one per state prefix would over-count by 16x).
        proposed_operations.append(
            {
                "provider": "usda_naip",
                "collection": imagery["collection"],
                "bucket": imagery["bucket"],
                "operation": "s3:GetObject",
                "key": manifest_key,
                "prefix": "",
                "target": "catalog_manifest",
                "requests": 1,
                "reserved_bytes": manifest_bytes,
                "requester_pays": True,
            }
        )
        per_op_reservations.append(manifest_bytes)
        providers.append(
            {
                "provider": "usda_naip",
                "collection": imagery["collection"],
                "bucket": imagery["bucket"],
                "region": imagery.get("region"),
                "source_requester_pays": bool(imagery.get("requester_pays")),
                "missing": True,
            }
        )

    if "terrain" in missing:
        terrain = contract.get("terrain")
        if not isinstance(terrain, dict):
            raise SourceContractError("source contract missing 'terrain' section")
        estimates = terrain.get("estimates")
        if not isinstance(estimates, dict):
            raise SourceContractError("terrain missing 'estimates'")
        per_asset_reqs = estimates.get("per_asset_requests")
        per_asset_bytes = estimates.get("per_asset_bytes_upper_bound")
        if (
            not isinstance(per_asset_reqs, int)
            or isinstance(per_asset_reqs, bool)
            or per_asset_reqs <= 0
        ):
            raise SourceContractError(
                "terrain.estimates.per_asset_requests must be a positive int"
            )
        if (
            not isinstance(per_asset_bytes, int)
            or isinstance(per_asset_bytes, bool)
            or per_asset_bytes <= 0
        ):
            raise SourceContractError(
                "terrain.estimates.per_asset_bytes_upper_bound must be a positive int"
            )

        discovery_prefixes = terrain.get("discovery_prefixes")
        if not isinstance(discovery_prefixes, list) or not discovery_prefixes:
            raise SourceContractError(
                "terrain.discovery_prefixes must be a non-empty list"
            )
        catalog_max_objects = terrain.get("catalog_max_objects")
        catalog_page_size = terrain.get("catalog_page_size")
        page_response_bytes = terrain.get("page_response_bytes_upper_bound")
        if (
            not isinstance(catalog_max_objects, int)
            or isinstance(catalog_max_objects, bool)
            or catalog_max_objects <= 0
        ):
            raise SourceContractError(
                "terrain.catalog_max_objects must be a positive int"
            )
        if (
            not isinstance(catalog_page_size, int)
            or isinstance(catalog_page_size, bool)
            or not 1 <= catalog_page_size <= 1000
        ):
            raise SourceContractError(
                "terrain.catalog_page_size must be an int in [1, 1000]"
            )
        if (
            not isinstance(page_response_bytes, int)
            or isinstance(page_response_bytes, bool)
            or page_response_bytes <= 0
        ):
            raise SourceContractError(
                "terrain.page_response_bytes_upper_bound must be a positive int"
            )
        # The 3DEP listing is paginated with continuation tokens; the exact
        # page count of the live bucket is unknown up front, so the declared
        # object bound yields a conservative page ceiling:
        #   max_pages = ceil(catalog_max_objects / catalog_page_size)
        # Each actual page is one billed ListBucket request and is counted per
        # page in the ledger; raw discovery must reject truncation beyond this
        # ceiling, so the ceiling is a true upper bound on requests.
        max_list_pages = math.ceil(catalog_max_objects / catalog_page_size)
        prefix = (
            discovery_prefixes[0]
            if discovery_prefixes
            else "StagedProducts/Elevation/13/TIFF/current/"
        )
        proposed_operations.append(
            {
                "provider": "usgs",
                "collection": terrain["collection"],
                "bucket": terrain["bucket"],
                "operation": "s3:ListBucket",
                "prefix": prefix,
                "target": "catalog_snapshot",
                "requests": max_list_pages,
                "reserved_bytes": page_response_bytes,
                "page_size": catalog_page_size,
                "max_objects": catalog_max_objects,
                "bound_kind": "conservative_upper_bound",
                "note": (
                    "max_pages = ceil(catalog_max_objects / catalog_page_size) is a "
                    "conservative ceiling over the unknown live page count; each "
                    "actual page is metered per page in the ledger"
                ),
                "requester_pays": False,
            }
        )
        per_op_reservations.append(page_response_bytes)
        providers.append(
            {
                "provider": "usgs",
                "collection": terrain["collection"],
                "bucket": terrain["bucket"],
                "region": terrain.get("region"),
                "source_requester_pays": bool(terrain.get("requester_pays")),
                "missing": True,
            }
        )

    total_requests = sum(int(op["requests"]) for op in proposed_operations)
    total_transfer_bytes = sum(
        int(op["reserved_bytes"]) * int(op["requests"]) for op in proposed_operations
    )
    total_local_bytes = total_transfer_bytes
    requires_requester_pays = any(
        bool(op.get("requester_pays")) for op in proposed_operations
    )

    req_cost = req_cost_dec * Decimal(total_requests)
    gb_transferred = Decimal(total_transfer_bytes) / Decimal(10**9)
    trans_cost = trans_cost_dec * gb_transferred
    total_cost = req_cost + trans_cost
    max_spend_usd = str(
        total_cost.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    )

    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_type": "discovery_authorization_request",
        "run_name": run_name,
        "target_geometry": {
            "coordinate_count": coordinate_count,
            "bbox": list(target_bbox) if target_bbox is not None else None,
            "jurisdictions": sorted(jurisdictions),
        },
        "missing_catalogs": sorted(missing),
        "proposed_operations": proposed_operations,
        "per_operation_byte_reservation": max(per_op_reservations)
        if per_op_reservations
        else None,
        "required_authorization": {
            "currency": "USD",
            "allow_requester_pays": requires_requester_pays,
            "max_requests": total_requests,
            "max_transfer_bytes": total_transfer_bytes,
            "max_local_bytes": total_local_bytes,
            "max_spend_usd": max_spend_usd,
        },
        "policy": {
            "currency": "USD",
            "requester_pays": requires_requester_pays,
            "network_call_made": False,
            "paid_source_allowed": False,
            "rate_card_source": rate_card["source"],
            "rate_card_date": rate_card["date"],
            "note": "This document authorizes nothing; positive caps and --allow-requester-pays must be supplied at execution time.",
        },
        "rate_card": {
            "source": rate_card["source"],
            "date": rate_card["date"],
            "request_cost_usd": rate_card["request_cost_usd"],
            "transfer_cost_per_gb_usd": rate_card["transfer_cost_per_gb_usd"],
        },
        "identities": {
            "source_contract_sha256": contract_sha,
            "region_spec_sha256": region_spec_sha,
            "geometry_digest": geometry_digest,
            "repository_source_tree_digest": repository_source_tree_digest,
        },
        "has_secrets": False,
        "resume_command": resume_command,
    }
    payload["authorization_digest"] = authorization_payload_digest(payload)
    return payload


def validate_imagery_source(value: str) -> None:
    if value != SUPPORTED_IMAGERY_SOURCE:
        raise SourceContractError(
            f"imagery source {value!r} is not supported; only {SUPPORTED_IMAGERY_SOURCE!r} "
            "is admissible in the active pipeline (unknown or paid sources are refused)"
        )


def validate_terrain_source(value: str) -> None:
    if value != SUPPORTED_TERRAIN_SOURCE:
        raise SourceContractError(
            f"terrain source {value!r} is not supported; only {SUPPORTED_TERRAIN_SOURCE!r} "
            "is admissible in the active pipeline"
        )


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_naip_catalog(
    catalog_path: str | Path | None, expected_sha256: str | None = None
) -> dict[str, Any] | None:
    """Load a normalized naip-visualization catalog dict, hash-pinned when pinned."""
    if catalog_path is None:
        return None
    path = Path(catalog_path)
    if not path.is_file():
        return None
    if expected_sha256 is not None:
        computed = file_sha256(path)
        if computed.lower() != expected_sha256.lower():
            raise SourceContractError(
                f"NAIP catalog SHA-256 mismatch for {path}: computed {computed}, "
                f"expected {expected_sha256}"
            )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceContractError(f"invalid NAIP catalog JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SourceContractError(f"NAIP catalog {path} must be a JSON object")
    # Adapter-level validation happens when NaipAdapter is constructed.
    return data


def load_3dep_catalog(
    catalog_path: str | Path | None, expected_sha256: str | None = None
) -> list[SourceAsset] | None:
    """Load and strictly validate the 3DEP 1/3 arc-second catalog records."""
    if catalog_path is None:
        return None
    path = Path(catalog_path)
    if not path.is_file():
        return None
    records, computed = load_hash_pinned_catalog(path, expected_sha256)
    assets: list[SourceAsset] = []
    for record in records:
        assets.append(validate_3dep_record(record))
    assets.sort(key=lambda asset: (asset.asset_id, asset.canonical_uri))
    return assets


def catalog_sha(catalog_path: str | Path | None) -> str | None:
    if catalog_path is None:
        return None
    path = Path(catalog_path)
    return file_sha256(path) if path.is_file() else None


# ---------------------------------------------------------------------------
# Per-tile asset assignment (deterministic)
# ---------------------------------------------------------------------------


def footprint_bounds(asset: SourceAsset) -> tuple[float, float, float, float]:
    """Return the finite WGS84 bounds of a required canonical asset footprint."""
    if not asset.footprint_geojson:
        raise SourceContractError(f"asset lacks a footprint: {asset.asset_id}")
    try:
        geometry = json.loads(asset.footprint_geojson)
    except Exception as exc:
        raise SourceContractError(
            f"asset footprint is invalid JSON: {asset.asset_id}"
        ) from exc
    if geometry.get("type") == "Feature":
        geometry = geometry.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") not in (
        "Polygon",
        "MultiPolygon",
    ):
        raise SourceContractError(
            f"asset footprint is not Polygon/MultiPolygon: {asset.asset_id}"
        )
    coordinates = geometry.get("coordinates")
    lons: list[float] = []
    lats: list[float] = []

    def _collect(value: Any) -> None:
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and all(
                isinstance(component, (int, float)) and not isinstance(component, bool)
                for component in value[:2]
            )
        ):
            lon, lat = float(value[0]), float(value[1])
            if not (math.isfinite(lon) and math.isfinite(lat)):
                raise SourceContractError(
                    f"asset footprint is non-finite: {asset.asset_id}"
                )
            lons.append(lon)
            lats.append(lat)
        elif isinstance(value, (list, tuple)):
            for child in value:
                _collect(child)

    _collect(coordinates)
    if not lons or not lats:
        raise SourceContractError(
            f"asset footprint has no coordinates: {asset.asset_id}"
        )
    return min(lons), min(lats), max(lons), max(lats)


def _bbox_overlap_area(
    tile_bbox: tuple[float, float, float, float],
    asset_bboxes: np.ndarray,
) -> np.ndarray:
    """Vectorized positive-area bbox overlap in degrees (deterministic)."""
    min_lon, min_lat, max_lon, max_lat = tile_bbox
    ox = np.minimum(asset_bboxes[:, 2], max_lon) - np.maximum(
        asset_bboxes[:, 0], min_lon
    )
    oy = np.minimum(asset_bboxes[:, 3], max_lat) - np.maximum(
        asset_bboxes[:, 1], min_lat
    )
    return np.maximum(ox, 0.0) * np.maximum(oy, 0.0)


@dataclass(frozen=True)
class StateIndex:
    """Per-state ordered NAIP asset lists plus vectorized bbox tables."""

    states: tuple[str, ...]
    ordered: dict[str, tuple[SourceAsset, ...]]
    bboxes: dict[str, np.ndarray]

    @classmethod
    def build(cls, adapter: NaipAdapter, allowed_states: Sequence[str]) -> "StateIndex":
        ordered: dict[str, tuple[SourceAsset, ...]] = {}
        bboxes: dict[str, np.ndarray] = {}
        for state in allowed_states:
            assets, _ = adapter.select_assets(desired_state=state, target_geometry=None)
            ordered[state] = tuple(assets)
            table = np.array(
                [footprint_bounds(asset) for asset in assets], dtype=np.float64
            )
            if table.size == 0:
                table = np.empty((0, 4), dtype=np.float64)
            bboxes[state] = table
        return cls(states=tuple(allowed_states), ordered=ordered, bboxes=bboxes)

    def assign_state(self, tile_bbox: tuple[float, float, float, float]) -> str | None:
        best_state: str | None = None
        best_area = 0.0
        for state in self.states:
            areas = _bbox_overlap_area(tile_bbox, self.bboxes[state])
            max_area = float(areas.max()) if areas.size else 0.0
            if max_area > best_area + 1e-12:
                best_area = max_area
                best_state = state
            elif abs(max_area - best_area) <= 1e-12 and best_state is not None:
                # Deterministic tie-break: lexicographically smaller state code.
                if state < best_state:
                    best_state = state
        return best_state

    def intersecting(
        self, state: str, tile_bbox: tuple[float, float, float, float]
    ) -> list[SourceAsset]:
        """Adapter-consistent positive-area footprint intersection, ordered."""
        assets = self.ordered[state]
        if not assets:
            return []
        areas = _bbox_overlap_area(tile_bbox, self.bboxes[state])
        candidates = [
            asset
            for asset, area in zip(assets, areas)
            if area > 0.0
            and calculate_intersection_area(asset.footprint_geojson, tile_bbox) > 0.0
        ]
        return candidates


def select_naip_assets_for_tile(
    state_index: StateIndex,
    tile_bbox: tuple[float, float, float, float],
) -> tuple[str | None, list[SourceAsset], list[str]]:
    """Return (state, assets, warnings) for one tile under the declared rule.

    Rule (declared in the source contract): state = argmax positive-area bbox
    intersection; assets = positive-area footprints of the chosen state limited
    to the maximum acquisition year present (single-vintage preference), in the
    adapter's deterministic order.
    """
    state = state_index.assign_state(tile_bbox)
    warnings: list[str] = []
    if state is None:
        return None, [], ["no_state_assignment"]

    intersecting = state_index.intersecting(state, tile_bbox)
    if not intersecting:
        return state, [], [f"state_assigned_no_naip_assets:{state}"]

    years = [
        asset.acquisition_year
        for asset in intersecting
        if asset.acquisition_year is not None
    ]
    if not years:
        selected = intersecting
    else:
        top_year = max(years)
        selected = [
            asset for asset in intersecting if asset.acquisition_year == top_year
        ]
        older = len(intersecting) - len(selected)
        if older:
            warnings.append(f"excluded_{older}_older_vintage_assets:{state}")
    if len(selected) > 1:
        warnings.append(
            "mosaic_candidates:" + ",".join(asset.asset_id for asset in selected)
        )
    return state, selected, warnings


def select_3dep_assets_for_tile(
    terrain_assets: Sequence[SourceAsset],
    terrain_bboxes: np.ndarray,
    tile_bbox: tuple[float, float, float, float],
) -> tuple[list[SourceAsset], list[str]]:
    """All intersecting 3DEP records ordered by (asset_id, canonical_uri)."""
    assets = list(terrain_assets)
    if not assets:
        return [], ["no_3dep_catalog_assets"]
    areas = _bbox_overlap_area(tile_bbox, terrain_bboxes)
    selected = [asset for asset, area in zip(assets, areas) if area > 0.0]
    if not selected:
        return [], ["no_terrain_assets_for_tile"]
    return selected, []


def _coord_key(x: int, y: int) -> str:
    return f"{x}_{y}"


def build_tile_plans(
    *,
    planned: dict[str, Any],
    contract: dict[str, Any],
    imagery_catalog: dict[str, Any] | None,
    terrain_assets: list[SourceAsset] | None,
    zoom: int,
) -> dict[str, Any]:
    """Compute deterministic per-coordinate asset assignments and estimates."""
    imagery = contract["imagery"]
    terrain = contract["terrain"]
    allowed_states = tuple(str(s).lower() for s in imagery.get("allowed_states", []))

    if imagery_catalog is None or terrain_assets is None:
        # Catalog missing: the caller stops at the discovery boundary. We still
        # produce per-coordinate geometry-level rows with no asset assignment.
        tile_plans: dict[str, dict[str, Any]] = {}
        for x, y in planned["ordered_coords"]:
            region = planned["coord_to_region"][(x, y)]
            tile_plans[_coord_key(x, y)] = {
                "region": region,
                "state": None,
                "satellite_asset_ids": [],
                "terrain_asset_ids": [],
                "estimated_satellite_bytes": 0,
                "estimated_terrain_bytes": 0,
                "satellite_acquisition_year": None,
                "satellite_capture_date": "",
                "satellite_source_checksums": {},
                "terrain_vertical_datum": None,
                "terrain_native_resolution": None,
                "terrain_source_checksums": {},
                "warnings": ["catalog_unavailable"],
            }
        return {
            "tile_plans": tile_plans,
            "unique_naip_asset_ids": [],
            "unique_3dep_asset_ids": [],
            "estimated_object_gets": 0,
            "estimated_object_gets_upper_bound": 0,
            "estimated_transfer_bytes": 0,
            "estimated_transfer_bytes_upper_bound": 0,
            "estimated_local_bytes": 0,
            "tiles_without_satellite_assets": 0,
            "tiles_without_terrain_assets": 0,
            "mosaic_tile_count": 0,
        }

    adapter = NaipAdapter(imagery_catalog)
    state_index = StateIndex.build(adapter, allowed_states)
    terrain_bboxes = np.array(
        [footprint_bounds(asset) for asset in terrain_assets], dtype=np.float64
    )
    if terrain_bboxes.size == 0:
        raise SourceContractError("3DEP catalog contains no validated assets")

    sat_upper = int(imagery["estimates"]["per_asset_bytes_upper_bound"])
    ter_upper = int(terrain["estimates"]["per_asset_bytes_upper_bound"])
    sat_requests = int(imagery["estimates"]["per_asset_requests"])
    ter_requests = int(terrain["estimates"]["per_asset_requests"])

    def _asset_bytes(asset: SourceAsset, upper: int) -> int:
        if not isinstance(asset.object_size_bytes, int) or isinstance(
            asset.object_size_bytes, bool
        ):
            raise SourceContractError(f"asset lacks object size: {asset.asset_id}")
        if asset.object_size_bytes > upper:
            raise SourceContractError(
                f"asset exceeds configured byte upper bound: {asset.asset_id}"
            )
        return asset.object_size_bytes

    tile_plans = {}
    unique_naip: dict[str, SourceAsset] = {}
    unique_3dep: dict[str, SourceAsset] = {}
    no_sat = 0
    no_ter = 0
    mosaic = 0

    for x, y in planned["ordered_coords"]:
        region = planned["coord_to_region"][(x, y)]
        tile_bbox = tile_bounds_wgs84(x, y, zoom)
        state, sat_assets, sat_warnings = select_naip_assets_for_tile(
            state_index, tile_bbox
        )
        ter_assets, ter_warnings = select_3dep_assets_for_tile(
            terrain_assets, terrain_bboxes, tile_bbox
        )
        warnings = sat_warnings + ter_warnings
        if not sat_assets:
            no_sat += 1
        if not ter_assets:
            no_ter += 1
        if len(sat_assets) > 1:
            mosaic += 1
        for asset in sat_assets:
            unique_naip[asset.canonical_uri] = asset
        for asset in ter_assets:
            unique_3dep[asset.canonical_uri] = asset
        sat_years = [
            a.acquisition_year for a in sat_assets if a.acquisition_year is not None
        ]
        tile_plans[_coord_key(x, y)] = {
            "region": region,
            "state": state,
            "satellite_asset_ids": [asset.asset_id for asset in sat_assets],
            "terrain_asset_ids": [asset.asset_id for asset in ter_assets],
            "estimated_satellite_bytes": sum(
                _asset_bytes(asset, sat_upper) for asset in sat_assets
            ),
            "estimated_terrain_bytes": sum(
                _asset_bytes(asset, ter_upper) for asset in ter_assets
            ),
            "satellite_acquisition_year": max(sat_years) if sat_years else None,
            "satellite_capture_date": max(
                (asset.capture_date or "" for asset in sat_assets), default=""
            ),
            "satellite_source_checksums": {
                asset.asset_id: (asset.checksum_sha256 or asset.etag or "")
                for asset in sat_assets
            },
            "terrain_vertical_datum": (
                ter_assets[0].vertical_datum if ter_assets else None
            ),
            "terrain_native_resolution": (
                ter_assets[0].native_resolution if ter_assets else None
            ),
            "terrain_source_checksums": {
                asset.asset_id: (asset.checksum_sha256 or asset.etag or "")
                for asset in ter_assets
            },
            "warnings": warnings,
        }

    naip_assets = sorted(
        unique_naip.values(), key=lambda a: (a.canonical_uri, a.asset_id)
    )
    tnm_assets = sorted(
        unique_3dep.values(), key=lambda a: (a.canonical_uri, a.asset_id)
    )
    unique_requests = (len(naip_assets) * sat_requests) + (
        len(tnm_assets) * ter_requests
    )
    unique_bytes_upper = sum(_asset_bytes(a, sat_upper) for a in naip_assets) + sum(
        _asset_bytes(a, ter_upper) for a in tnm_assets
    )
    per_tile_reads = sum(
        len(tile_plans[key]["satellite_asset_ids"])
        + len(tile_plans[key]["terrain_asset_ids"])
        for key in tile_plans
    )
    # Worst-case local storage: unique source bytes plus both output PNGs per coordinate.
    output_upper = int(imagery["estimates"]["per_tile_output_bytes_upper_bound"]) + int(
        terrain["estimates"]["per_tile_output_bytes_upper_bound"]
    )
    expected_local = unique_bytes_upper + output_upper * len(tile_plans)

    return {
        "tile_plans": tile_plans,
        "unique_naip_asset_ids": [asset.asset_id for asset in naip_assets],
        "unique_3dep_asset_ids": [asset.asset_id for asset in tnm_assets],
        "unique_naip_assets": [asset.to_dict() for asset in naip_assets],
        "unique_3dep_assets": [asset.to_dict() for asset in tnm_assets],
        "estimated_object_gets": unique_requests,
        "estimated_object_gets_upper_bound": unique_requests + per_tile_reads,
        "estimated_transfer_bytes": unique_bytes_upper,
        "estimated_transfer_bytes_upper_bound": unique_bytes_upper,
        "estimated_local_bytes": expected_local,
        "tiles_without_satellite_assets": no_sat,
        "tiles_without_terrain_assets": no_ter,
        "mosaic_tile_count": mosaic,
        "state_index": state_index,
        "terrain_asset_list": terrain_assets,
        "terrain_bbox_table": terrain_bboxes,
    }


# ---------------------------------------------------------------------------
# Pilot frame selection (deterministic, frozen)
# ---------------------------------------------------------------------------


def select_pilot_frame(
    ordered_coords: Sequence[tuple[int, int]],
    coord_to_region: dict[tuple[int, int], str],
    pilot_budget: int,
    seed: int = PILOT_SELECTION_SEED,
) -> tuple[list[tuple[int, int]], str]:
    """Deterministic capped pilot frame of at most ``pilot_budget`` coordinates.

    Macroblock-stratified: coordinates are grouped by (x // 16, y // 16)
    macroblock, groups are ordered by a stable seed hash, and coordinates
    within each group by their own stable seed hash.  The first
    ``pilot_budget`` coordinates of that ordering form the frozen frame.
    """
    if pilot_budget < 1 or pilot_budget > PILOT_MAX_COORDINATES:
        raise ValueError(
            f"pilot_budget must be between 1 and {PILOT_MAX_COORDINATES}, got {pilot_budget}"
        )

    def _stable(value: Any) -> bytes:
        return hashlib.sha256(canonical_json_bytes([seed, value])).digest()

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y in ordered_coords:
        groups.setdefault((x // 16, y // 16), []).append((x, y))
    group_order = sorted(
        groups,
        key=lambda block: (_stable(list(block)), block[0], block[1]),
    )
    frame: list[tuple[int, int]] = []
    for block in group_order:
        coords = sorted(
            groups[block],
            key=lambda coord: (
                _stable(list(coord)),
                coord_to_region[coord],
                coord[0],
                coord[1],
            ),
        )
        for coord in coords:
            if len(frame) >= pilot_budget:
                break
            frame.append(coord)
        if len(frame) >= pilot_budget:
            break
    frame_sha = identity_digest(
        {"seed": seed, "budget": pilot_budget, "coords": list(frame)}
    )
    return frame, frame_sha


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------


def _resume_command(
    *,
    run_name: str,
    spec_path: str | None,
    contract_path: str | Path,
    image_root: str | Path,
    output_root: str | Path,
    budget: int,
    pilot_budget: int | None,
    config_path: str | Path,
    imagery_source: str,
    terrain_source: str,
) -> str:
    parts = [
        "uv run python scripts/ingest/plan_active_learning_region.py",
        f"--run-name {run_name}",
        f"--imagery-source {imagery_source}",
        f"--terrain-source {terrain_source}",
        f"--source-contract {contract_path}",
        f"--image-root {image_root}",
        f"--output-root {output_root}",
        f"--budget {budget}",
    ]
    if pilot_budget is not None:
        parts.append(f"--pilot-budget {pilot_budget}")
    if spec_path:
        parts.append(f"--region-spec {spec_path}")
    if config_path != "config/app_regions.json":
        parts.append(f"--region-source {config_path}")
    return " \\\n  ".join(parts)


def build_execution_plan(
    *,
    run_name: str,
    zoom: int,
    contract: dict[str, Any],
    contract_sha: str,
    contract_path: str | Path,
    spec_path: str | None,
    spec_sha: str,
    geometry_digest: str,
    imagery_catalog_path: str | Path | None,
    imagery_catalog_sha: str | None,
    terrain_catalog_path: str | Path | None,
    terrain_catalog_sha: str | None,
    tile_plans_result: dict[str, Any],
    planned_rows: list[dict[str, Any]],
    tile_manifest_sha: str,
    pilot_frame_sha: str | None,
    budget: int,
    pilot_budget: int | None,
    image_root: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    imagery_source: str,
    terrain_source: str,
    max_runtime_minutes: int | None,
    land_geometry_path: str | Path | None,
    land_geometry_sha256: str | None,
    missing_satellite: int,
    missing_terrain: int,
    local_reuse_count: int,
    repository_source_tree_digest: str | None = None,
) -> dict[str, Any]:
    repository_source_tree_digest = _resolve_repository_source_digest(
        repository_source_tree_digest
    )
    coordinate_list_sha = identity_digest(
        [
            (row["region"], int(row["z"]), int(row["x"]), int(row["y"]))
            for row in planned_rows
        ]
    )
    preprocessing = contract["preprocessing"]
    rate_card = contract["rate_card"]
    estimated_cost = (
        Decimal(str(rate_card["request_cost_usd"]))
        * int(tile_plans_result["estimated_object_gets_upper_bound"])
        + Decimal(str(rate_card["transfer_cost_per_gb_usd"]))
        * int(tile_plans_result["estimated_transfer_bytes_upper_bound"])
        / Decimal(10**9)
    ).quantize(Decimal("0.000000000001"), rounding=ROUND_CEILING)

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_name": run_name,
        "state": "planned",
        "zoom": zoom,
        "grid_contract": {
            "crs": preprocessing["target_crs"],
            "tile_width": preprocessing["tile_width"],
            "tile_height": preprocessing["tile_height"],
            "semantics": "slippy-map Web Mercator; x east, y south; bounds computed from (z, x, y)",
        },
        "coordinate_count": len(planned_rows),
        "raster_count": len(planned_rows) * 2,
        "missing_satellite": missing_satellite,
        "missing_terrain": missing_terrain,
        "unique_naip_assets": tile_plans_result["unique_naip_asset_ids"],
        "unique_3dep_assets": tile_plans_result["unique_3dep_asset_ids"],
        "estimated_cog_range_requests": 0,
        "range_requests_note": "whole-object downloads through the metered transport; no COG byte ranges are issued",
        "estimated_object_gets": tile_plans_result["estimated_object_gets"],
        "estimated_object_gets_upper_bound": tile_plans_result[
            "estimated_object_gets_upper_bound"
        ],
        "estimated_transfer_bytes": tile_plans_result["estimated_transfer_bytes"],
        "estimated_transfer_bytes_upper_bound": tile_plans_result[
            "estimated_transfer_bytes_upper_bound"
        ],
        "expected_local_storage_bytes": tile_plans_result["estimated_local_bytes"],
        "local_reuse_count": local_reuse_count,
        "s3_reuse_count": 0,
        "requester_pays": {
            "imagery": bool(contract["imagery"].get("requester_pays")),
            "terrain": bool(contract["terrain"].get("requester_pays")),
        },
        "rate_card": {
            "source": rate_card["source"],
            "date": rate_card["date"],
            "request_cost_usd": rate_card["request_cost_usd"],
            "transfer_cost_per_gb_usd": rate_card["transfer_cost_per_gb_usd"],
        },
        "estimated_cost_usd": format(estimated_cost, "f"),
        "estimated_cost_note": "conservative request and upper-bound transfer estimate; execution still requires explicit caps",
        "caps": {
            "max_requests": None,
            "max_transfer_bytes": None,
            "max_local_bytes": None,
            "max_requester_pays_usd": None,
            "allow_requester_pays": False,
        },
        "hashes": {
            "source_contract_sha256": contract_sha,
            "preprocessing_contract_sha256": identity_digest(preprocessing),
            "region_spec_sha256": spec_sha,
            "geometry_digest": geometry_digest,
            "land_geometry_sha256": land_geometry_sha256,
            "pilot_frame_sha256": pilot_frame_sha,
            "tile_manifest_sha256": tile_manifest_sha,
            "coordinate_list_sha256": coordinate_list_sha,
            "imagery_catalog_sha256": imagery_catalog_sha,
            "terrain_catalog_sha256": terrain_catalog_sha,
            "repository_source_tree_digest": repository_source_tree_digest,
        },
        "inputs": {
            "source_contract_path": str(contract_path),
            "region_spec_path": spec_path,
            "land_geometry_path": (
                str(land_geometry_path) if land_geometry_path is not None else None
            ),
            "image_root": str(image_root),
            "output_root": str(output_root),
            "app_regions_config": str(config_path),
            "imagery_source": imagery_source,
            "terrain_source": terrain_source,
            "budget": budget,
            "pilot_budget": pilot_budget,
            "imagery_catalog_path": str(imagery_catalog_path)
            if imagery_catalog_path
            else None,
            "terrain_catalog_path": str(terrain_catalog_path)
            if terrain_catalog_path
            else None,
            "tiles_without_satellite_assets": tile_plans_result[
                "tiles_without_satellite_assets"
            ],
            "tiles_without_terrain_assets": tile_plans_result[
                "tiles_without_terrain_assets"
            ],
            "mosaic_tile_count": tile_plans_result["mosaic_tile_count"],
        },
        "deadline": {"max_runtime_minutes": max_runtime_minutes},
        "resume_command": _resume_command(
            run_name=run_name,
            spec_path=spec_path,
            contract_path=contract_path,
            image_root=image_root,
            output_root=output_root,
            budget=budget,
            pilot_budget=pilot_budget,
            config_path=config_path,
            imagery_source=imagery_source,
            terrain_source=terrain_source,
        ),
        "determinism_note": (
            "recomputed from identical inputs and compared byte-for-byte at execution; "
            "no wall-clock fields are part of this plan"
        ),
    }
    return plan


def plan_digest(plan: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON bytes of the plan (plan carries no digest field)."""
    return hashlib.sha256(canonical_json_bytes(plan)).hexdigest()


def load_and_verify_plan(plan_path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Load the immutable plan and verify its digest byte-for-byte."""
    path = Path(plan_path)
    if not path.is_file():
        raise PlanDriftError(f"execution plan not found: {path}")
    raw = path.read_bytes()
    computed = hashlib.sha256(raw).hexdigest()
    if computed.lower() != expected_sha256.lower():
        raise PlanDriftError(
            f"execution plan SHA-256 mismatch: computed {computed}, expected {expected_sha256}"
        )
    try:
        plan = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise PlanDriftError(f"execution plan is not valid JSON: {exc}") from exc
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanDriftError(
            f"unsupported execution plan schema version: {plan.get('schema_version')}"
        )
    return plan


# ---------------------------------------------------------------------------
# Authorization artifacts
# ---------------------------------------------------------------------------


def _authorization_policy(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "currency": "USD",
        "requester_pays": False,
        "max_requests": None,
        "max_transfer_bytes": None,
        "max_local_bytes": None,
        "max_spend_usd": None,
    }


def build_acquisition_authorization_request(
    *,
    run_name: str,
    plan: dict[str, Any],
    plan_sha: str,
    contract: dict[str, Any],
    repository_source_tree_digest: str | None = None,
) -> dict[str, Any]:
    """Unified acquisition-authorization artifact for a complete immutable plan.

    Lists the exact proposed operations (HEAD/GET per unique asset, per-tile
    local processing) with null byte/request/local/spend caps and a false
    requester-pays policy: nothing here authorizes network spend.
    """
    repository_source_tree_digest = _resolve_repository_source_digest(
        repository_source_tree_digest
        if repository_source_tree_digest is not None
        else plan.get("hashes", {}).get("repository_source_tree_digest")
    )
    if (
        plan.get("hashes", {}).get("repository_source_tree_digest")
        != repository_source_tree_digest
    ):
        raise SourceContractError(
            "acquisition authorization source-tree identity does not match its plan"
        )
    proposed_operations: list[dict[str, Any]] = []
    imagery = contract["imagery"]
    terrain = contract["terrain"]
    for asset_id in plan["unique_naip_assets"]:
        proposed_operations.append(
            {
                "provider": "usda_naip",
                "collection": imagery["collection"],
                "bucket": imagery["bucket"],
                "operation": "s3:HeadObject",
                "key": asset_id,
                "requester_pays": True,
                "reserved_bytes": None,
            }
        )
        proposed_operations.append(
            {
                "provider": "usda_naip",
                "collection": imagery["collection"],
                "bucket": imagery["bucket"],
                "operation": "s3:GetObject",
                "key": asset_id,
                "requester_pays": True,
                "reserved_bytes": None,
            }
        )
    for asset_id in plan["unique_3dep_assets"]:
        proposed_operations.append(
            {
                "provider": "usgs",
                "collection": terrain["collection"],
                "bucket": terrain["bucket"],
                "operation": "s3:HeadObject",
                "key": asset_id,
                "requester_pays": False,
                "reserved_bytes": None,
            }
        )
        proposed_operations.append(
            {
                "provider": "usgs",
                "collection": terrain["collection"],
                "bucket": terrain["bucket"],
                "operation": "s3:GetObject",
                "key": asset_id,
                "requester_pays": False,
                "reserved_bytes": None,
            }
        )
    proposed_operations.append(
        {
            "operation": "local_raster_processing",
            "tiles": plan["coordinate_count"],
            "rasters": plan["raster_count"],
            "reserved_bytes": None,
        }
    )

    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_type": "acquisition_authorization_request",
        "run_name": run_name,
        "execution_plan_path": "execution_plan.json",
        "execution_plan_sha256": plan_sha,
        "proposed_operations": proposed_operations,
        "per_operation_byte_reservation": None,
        "required_authorization": _authorization_policy(contract),
        "policy": {
            "currency": "USD",
            "requester_pays": False,
            "network_call_made": False,
            "paid_source_allowed": False,
            "note": "This document authorizes nothing; execution requires the plan, its SHA-256, positive caps, and --allow-requester-pays.",
        },
        "identities": {
            "source_contract_sha256": plan["hashes"]["source_contract_sha256"],
            "preprocessing_contract_sha256": plan["hashes"][
                "preprocessing_contract_sha256"
            ],
            "region_spec_sha256": plan["hashes"]["region_spec_sha256"],
            "geometry_digest": plan["hashes"]["geometry_digest"],
            "pilot_frame_sha256": plan["hashes"]["pilot_frame_sha256"],
            "repository_source_tree_digest": (
                repository_source_tree_digest
                if repository_source_tree_digest is not None
                else plan["hashes"].get("repository_source_tree_digest")
            ),
        },
        "has_secrets": False,
        "resume_command": plan["resume_command"],
    }


# ---------------------------------------------------------------------------
# Layered source cache (content-addressed, ownership-safe)
# ---------------------------------------------------------------------------


def _cache_object_path(cache_root: Path, content_sha: str) -> Path:
    return cache_root / "objects" / f"{content_sha}.bin"


def _cache_sidecar_path(cache_root: Path, content_sha: str) -> Path:
    return cache_root / "objects" / f"{content_sha}.json"


def _install_cached_object(
    cache_root: Path,
    *,
    bucket: str,
    key: str,
    etag: str | None,
    size: int,
    content: bytes,
) -> tuple[Path, str]:
    """Atomically install downloaded bytes into the content-addressed cache."""
    content_sha = hashlib.sha256(content).hexdigest()
    target = _cache_object_path(cache_root, content_sha)
    sidecar = _cache_sidecar_path(cache_root, content_sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        existing = file_sha256(target)
        if existing != content_sha:
            raise OwnershipError(
                f"cache object {target} content does not match its content-addressed "
                f"name (corrupt or foreign); refusing to overwrite"
            )
        # Identical bytes already cached; just ensure the sidecar is present.
        if not sidecar.is_file():
            atomic_write_json(
                sidecar,
                {
                    "bucket": bucket,
                    "key": key,
                    "etag": etag,
                    "size": size,
                    "content_sha256": content_sha,
                },
            )
        return target, content_sha

    descriptor, temporary = tempfile.mkstemp(
        prefix=".download.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_bytes(content)
        os.replace(temporary_path, target)
        atomic_write_json(
            sidecar,
            {
                "bucket": bucket,
                "key": key,
                "etag": etag,
                "size": size,
                "content_sha256": content_sha,
            },
        )
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
    return target, content_sha


def _cached_object_for(
    cache_root: Path,
    *,
    bucket: str,
    key: str,
    etag: str | None,
) -> tuple[Path | None, str | None]:
    """Return (path, content_sha) of a verified cache hit, or (None, None)."""
    # We do not know the content SHA until downloaded; search sidecars by key.
    objects_dir = cache_root / "objects"
    if not objects_dir.is_dir():
        return None, None
    for sidecar in objects_dir.glob("*.json"):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            meta.get("bucket") == bucket
            and meta.get("key") == key
            and (etag is None or meta.get("etag") is None or meta.get("etag") == etag)
        ):
            content_sha = meta.get("content_sha256")
            obj_path = _cache_object_path(cache_root, content_sha)
            if (
                isinstance(content_sha, str)
                and obj_path.is_file()
                and file_sha256(obj_path) == content_sha
            ):
                return obj_path, content_sha
    return None, None


class _SingleFlight:
    """Keyed single-flight: at most one in-flight acquisition per object.

    Concurrent callers for the same key share one execution: the leader runs
    the acquisition (cache check, download, atomic install) while followers
    block on a per-key event and then receive the leader's validated result
    (or re-raise the leader's exception).  Different keys never contend, and
    the entry is always removed on completion or failure, so a failed leader
    does not poison later retries.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._inflight: dict[Any, _SingleFlight._Flight] = {}

    class _Flight:
        __slots__ = ("event", "result", "error")

        def __init__(self) -> None:
            self.event = threading.Event()
            self.result: tuple[Path, str] | None = None
            self.error: BaseException | None = None

    def run(
        self, key: Any, fn: Callable[[], tuple[Path, str]]
    ) -> tuple[tuple[Path, str], bool]:
        """Run *fn* once for *key*; return ``(result, leader)``.

        ``leader`` is True for the caller that executed *fn* (and therefore
        already recorded its own ledger events); followers receive the
        leader's result or re-raise the leader's exception.
        """
        with self._guard:
            flight = self._inflight.get(key)
            if flight is None:
                flight = self._Flight()
                self._inflight[key] = flight
                leader = True
            else:
                leader = False
        if not leader:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            assert flight.result is not None
            return flight.result, False
        try:
            result = fn()
            flight.result = result
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._guard:
                if self._inflight.get(key) is flight:
                    del self._inflight[key]
            flight.event.set()
        return result, True


def _ensure_object(
    *,
    transport: MeteredTransport,
    cache_root: Path,
    bucket: str,
    key: str,
    etag: str | None,
    max_response_bytes: int,
    ledger: MeteredLedger,
    flights: _SingleFlight | None = None,
) -> tuple[Path, str]:
    """Return (local_path, content_sha256) of the object in the local cache.

    Reuses a verified cached copy when present; otherwise downloads exactly
    this object through the metered transport (never the whole bucket).
    Concurrent callers for the same (bucket, key, etag) share a single
    acquisition through *flights* (a fresh single-flight when omitted), so the
    object is fetched at most once per identity while different identities
    proceed concurrently.
    """
    if flights is None:
        flights = _SingleFlight()

    def acquire() -> tuple[Path, str]:
        cached_path, cached_sha = _cached_object_for(
            cache_root, bucket=bucket, key=key, etag=etag
        )
        if cached_path is not None:
            ledger.append(
                {
                    "event": "cache_hit",
                    "bucket": bucket,
                    "key": key,
                    "etag": etag,
                    "content_sha256": cached_sha,
                }
            )
            return cached_path, cached_sha  # type: ignore[return-value]

        size: int | None = None
        resolved_etag: str | None = etag
        if resolved_etag is None:
            head = transport.head_object(key)
            size = head.size
            resolved_etag = head.etag
            ledger.append(
                {
                    "event": "object_meta",
                    "bucket": bucket,
                    "key": key,
                    "size": size,
                    "etag": resolved_etag,
                }
            )
        if size is None:
            # Open-ended GET with a declared maximum response size.
            result = transport.get_range(
                key,
                start=0,
                max_response_bytes=max_response_bytes,
                expected_etag=resolved_etag,
            )
        else:
            result = transport.get_range(
                key,
                start=0,
                end=size - 1,
                max_response_bytes=max_response_bytes,
                expected_etag=resolved_etag,
            )
        return _install_cached_object(
            cache_root,
            bucket=bucket,
            key=key,
            etag=result.etag or resolved_etag,
            size=len(result.content),
            content=result.content,
        )

    result, leader = flights.run((bucket, key, etag), acquire)
    path, content_sha = result
    if not leader:
        # Followers consumed the leader's validated cached bytes without any
        # additional network attempt; record them as a cache hit.
        ledger.append(
            {
                "event": "cache_hit",
                "bucket": bucket,
                "key": key,
                "etag": etag,
                "content_sha256": content_sha,
            }
        )
    return path, content_sha


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------


def _grid_sha(z: int, x: int, y: int, width: int, height: int) -> str:
    return identity_digest(
        {
            "z": z,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "crs": "EPSG:3857",
        }
    )


def process_tile_pair(
    *,
    z: int,
    x: int,
    y: int,
    region: str,
    satellite_paths: Sequence[Path],
    terrain_paths: Sequence[Path],
    terrain_vertical_datums: Sequence[str | None],
    image_root: Path,
    contract: dict[str, Any],
    land_geometry: Any,
) -> dict[str, Any]:
    """Process one aligned satellite/terrain pair; returns full provenance.

    Satellite and terrain share one exact TargetGrid object.  Rasterio errors
    propagate (fail closed); no-data-on-land and out-of-range elevations raise
    before anything is written.
    """
    preprocessing = contract["preprocessing"]
    width = int(preprocessing["tile_width"])
    height = int(preprocessing["tile_height"])
    grid = TargetGrid.from_tile(
        z=z, x=x, y=y, width=width, height=height, crs="EPSG:3857"
    )

    satellite_fill = preprocessing.get("satellite_fill", [0, 0, 0])
    if not isinstance(satellite_fill, list) or len(satellite_fill) != 3:
        raise SourceContractError(
            "preprocessing.satellite_fill must contain three values"
        )
    fill_value = (
        int(satellite_fill[0]),
        int(satellite_fill[1]),
        int(satellite_fill[2]),
    )
    if any(
        not isinstance(value, str) or not value for value in terrain_vertical_datums
    ):
        raise SourceContractError("every terrain asset requires a vertical datum")
    source_vertical_datums = [str(value) for value in terrain_vertical_datums]
    satellite_qc_contract = preprocessing.get("satellite_qc", {})
    satellite_rgb, satellite_qc = process_imagery(
        list(satellite_paths),
        grid,
        land_geometry,
        resampling=str(preprocessing["satellite_resampling"]),
        fill_value=fill_value,
        mosaic_order=str(preprocessing.get("mosaic_order", "first")),
        min_land_variance=float(satellite_qc_contract.get("min_land_variance", 0.0)),
        reject_all_black=bool(satellite_qc_contract.get("reject_all_black", True)),
        reject_all_white=bool(satellite_qc_contract.get("reject_all_white", True)),
    )
    terrain_rgb, _dem, terrain_qc = process_dem(
        list(terrain_paths),
        grid,
        land_geometry,
        resampling=str(preprocessing["terrain_resampling"]),
        outside_land_elevation=float(
            preprocessing.get("outside_land_elevation", -10000.0)
        ),
        mosaic_order=str(preprocessing.get("mosaic_order", "first")),
        source_vertical_datum=source_vertical_datums,
        target_vertical_datum=str(preprocessing["target_vertical_datum"]),
    )

    satellite_path = image_root / "satellite" / f"z{z}" / region / f"{x}_{y}.png"
    terrain_path = image_root / "terrain" / f"z{z}" / region / f"{x}_{y}.png"
    _, satellite_sha = write_atomic_png(satellite_rgb, satellite_path)
    _, terrain_sha = write_atomic_png(terrain_rgb, terrain_path)

    return {
        "region": region,
        "z": z,
        "x": x,
        "y": y,
        "grid_sha256": _grid_sha(z, x, y, width, height),
        "satellite": {
            "path": str(satellite_path),
            "output_sha256": satellite_sha,
            "valid_fraction": float(satellite_qc["valid_fraction"]),
            "qc": satellite_qc,
        },
        "terrain": {
            "path": str(terrain_path),
            "output_sha256": terrain_sha,
            "valid_fraction": float(terrain_qc["valid_fraction"]),
            "qc": terrain_qc,
        },
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _build_caps(
    *,
    max_source_requests: int,
    max_transfer_bytes: int,
    max_local_bytes: int,
    max_requester_pays_usd: Decimal | str,
    allow_requester_pays: bool,
) -> NetworkCaps:
    for name, value in (
        ("max_source_requests", max_source_requests),
        ("max_transfer_bytes", max_transfer_bytes),
        ("max_local_bytes", max_local_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not isinstance(max_requester_pays_usd, Decimal):
        max_requester_pays_usd = Decimal(str(max_requester_pays_usd))
    if not max_requester_pays_usd.is_finite() or max_requester_pays_usd <= 0:
        raise ValueError(
            f"max_requester_pays_usd must be positive, got {max_requester_pays_usd}"
        )
    return NetworkCaps(
        max_requests=max_source_requests,
        max_transfer_bytes=max_transfer_bytes,
        max_local_bytes=max_local_bytes,
        max_requester_pays_usd=max_requester_pays_usd,
        allow_requester_pays=bool(allow_requester_pays),
    )


def _default_transport_factory(
    bucket: str,
    caps: NetworkCaps,
    ledger: MeteredLedger,
    rate_card: RateCard,
    requester_pays: bool,
    shared_budget: SharedMeteredBudget,
) -> MeteredTransport:
    return Boto3MeteredTransport(
        bucket=bucket,
        caps=caps,
        ledger=ledger,
        rate_card=rate_card,
        requester_pays=requester_pays,
        shared_budget=shared_budget,
    )


@dataclass
class _TileOutcome:
    coord_key: str
    provenance: dict[str, Any] | None
    failure: dict[str, Any] | None
    skipped: bool = False


def _record_path(run_dir: Path, region: str, z: int, x: int, y: int) -> Path:
    return run_dir / "tile_records" / f"{region}_z{z}_{x}_{y}.json"


def _validate_existing_record(
    *,
    sat_path: Path,
    ter_path: Path,
    record_path: Path,
    run_id: str,
    plan_sha: str,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Return (valid, reason, record).

    A tile is valid only when BOTH outputs exist, the adjacent run record
    carries this exact run id and execution-plan SHA, and every recorded
    output SHA-256 matches the bytes on disk.  Existence or non-zero size
    alone is never trusted.
    """
    if not sat_path.is_file() or not ter_path.is_file():
        return False, "missing_output", None
    if not record_path.is_file():
        return False, "foreign_or_unowned_output", None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "invalid_run_record", None
    if (
        record.get("run_id") != run_id
        or record.get("execution_plan_sha256") != plan_sha
    ):
        return False, "foreign_owner", record
    for style, path in (("satellite", sat_path), ("terrain", ter_path)):
        recorded = (record.get(style) or {}).get("sha256")
        if not isinstance(recorded, str) or not recorded:
            return False, f"{style}_record_missing_sha", record
        if file_sha256(path) != recorded:
            return False, f"{style}_content_mismatch", record
    return True, None, record


def _quarantine(path: Path) -> None:
    """Move run-owned corrupt bytes aside; never delete them."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_name(f"{path.name}.quarantine.{timestamp}")
    os.replace(path, quarantine)


def _read_checkpoint(run_dir: Path, plan_sha: str) -> dict[str, Any] | None:
    path = run_dir / "checkpoint.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("execution_plan_sha256") != plan_sha:
        # Checkpoint belongs to a different plan; treated as foreign.
        return None
    return data


def _write_checkpoint(
    run_dir: Path,
    *,
    run_name: str,
    plan_sha: str,
    completed: list[str],
    counters: Counters,
    state: str,
) -> None:
    atomic_write_json(
        run_dir / "checkpoint.json",
        {
            "run_id": run_name,
            "execution_plan_sha256": plan_sha,
            "completed_coords": sorted(completed),
            "state": state,
            "counters": {
                "requests": counters.requests,
                "transfer_bytes": counters.transfer_bytes,
                "local_bytes": counters.local_bytes,
                "cost_usd": str(counters.cost_usd),
            },
        },
    )


def execute_plan(
    plan_path: str | Path,
    *,
    expected_plan_sha256: str,
    allow_requester_pays: bool,
    max_source_requests: int,
    max_transfer_bytes: int,
    max_local_bytes: int,
    max_requester_pays_usd: Decimal | str,
    workers: int = 8,
    max_runtime_minutes: int | None = None,
    transport_factory: Callable[..., MeteredTransport] | None = None,
) -> dict[str, Any]:
    """Execute (or resume) an immutable execution plan under hard caps.

    ``transport_factory`` is dependency-injectable so tests run against local
    fixtures with no network; it receives
    ``(bucket, caps, ledger, rate_card, requester_pays, shared_budget)``.
    """
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}, got {workers}")
    if max_runtime_minutes is not None and (
        not isinstance(max_runtime_minutes, int)
        or isinstance(max_runtime_minutes, bool)
        or max_runtime_minutes < 1
    ):
        raise ValueError(
            f"max_runtime_minutes must be a positive int or None, got {max_runtime_minutes!r}"
        )
    caps = _build_caps(
        max_source_requests=max_source_requests,
        max_transfer_bytes=max_transfer_bytes,
        max_local_bytes=max_local_bytes,
        max_requester_pays_usd=max_requester_pays_usd,
        allow_requester_pays=allow_requester_pays,
    )
    if transport_factory is None:
        transport_factory = _default_transport_factory

    plan = load_and_verify_plan(plan_path, expected_plan_sha256)
    planned_source_digest = plan.get("hashes", {}).get("repository_source_tree_digest")
    if not isinstance(planned_source_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", planned_source_digest
    ):
        raise PlanDriftError(
            "execution plan lacks a valid repository source-tree digest"
        )
    current_source_digest = repository_source_tree_digest()
    if current_source_digest != planned_source_digest:
        raise PlanDriftError("repository source-tree digest drift")
    planned_runtime = plan.get("deadline", {}).get("max_runtime_minutes")
    if planned_runtime is not None:
        if (
            not isinstance(planned_runtime, int)
            or isinstance(planned_runtime, bool)
            or planned_runtime < 1
        ):
            raise PlanDriftError("plan contains an invalid runtime deadline")
        max_runtime_minutes = (
            planned_runtime
            if max_runtime_minutes is None
            else min(max_runtime_minutes, planned_runtime)
        )
    run_name = plan["run_name"]
    run_dir = Path(plan_path).resolve().parent
    output_root = Path(plan["inputs"]["output_root"])
    if run_dir != (output_root / run_name).resolve():
        raise PlanDriftError(
            f"plan run directory {run_dir} does not match output_root/run_name "
            f"{(output_root / run_name).resolve()}"
        )

    # --- inputs and hashes ------------------------------------------------
    contract_path = Path(plan["inputs"]["source_contract_path"])
    contract, contract_sha = load_source_contract(contract_path)
    if contract_sha != plan["hashes"]["source_contract_sha256"]:
        raise PlanDriftError("source contract SHA-256 drift")
    image_root = Path(plan["inputs"]["image_root"])
    cache_root = image_root.parent / "cache"
    preprocessing_sha = identity_digest(contract["preprocessing"])
    if preprocessing_sha != plan["hashes"]["preprocessing_contract_sha256"]:
        raise PlanDriftError("preprocessing contract SHA-256 drift")

    imagery_catalog_path = Path(plan["inputs"]["imagery_catalog_path"])
    terrain_catalog_path = Path(plan["inputs"]["terrain_catalog_path"])
    imagery_catalog = load_naip_catalog(
        imagery_catalog_path, plan["hashes"]["imagery_catalog_sha256"]
    )
    terrain_assets = load_3dep_catalog(
        terrain_catalog_path, plan["hashes"]["terrain_catalog_sha256"]
    )
    if imagery_catalog is None or terrain_assets is None:
        raise PlanDriftError(
            "execution requires both catalog snapshots; catalog missing -> "
            "re-run planning to the discovery authorization boundary"
        )
    if catalog_sha(imagery_catalog_path) != plan["hashes"]["imagery_catalog_sha256"]:
        raise PlanDriftError("imagery catalog SHA-256 drift")
    if catalog_sha(terrain_catalog_path) != plan["hashes"]["terrain_catalog_sha256"]:
        raise PlanDriftError("terrain catalog SHA-256 drift")

    # --- land geometry binding ---------------------------------------------
    planned_land_sha = plan["hashes"].get("land_geometry_sha256")
    land_geometry_path_value = plan["inputs"].get("land_geometry_path")
    if not planned_land_sha or not land_geometry_path_value:
        raise PlanDriftError("execution plan must bind a pinned land geometry file")
    land_geometry_path = Path(land_geometry_path_value)
    if not land_geometry_path.is_file():
        raise PlanDriftError(f"land geometry file not found: {land_geometry_path}")
    land_geometry_sha = file_sha256(land_geometry_path)
    if land_geometry_sha != planned_land_sha:
        raise PlanDriftError(
            f"land geometry SHA-256 mismatch: computed {land_geometry_sha}, "
            f"plan expects {planned_land_sha}"
        )
    try:
        land_geometry_data = json.loads(land_geometry_path.read_text(encoding="utf-8"))
        land_geometry = parse_geojson_geometry(land_geometry_data)
    except Exception as exc:
        raise PlanDriftError(f"land geometry is invalid GeoJSON: {exc}") from exc

    # --- recompute the plan from identical inputs (drift check) -----------
    planned: dict[str, Any] = {"ordered_coords": [], "coord_to_region": {}}
    manifest_path = run_dir / "tile_manifest.csv"
    import csv as _csv

    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows: list[dict[str, Any]] = list(_csv.DictReader(stream))
    for row in rows:
        # CSV round-trips booleans as strings; restore them so recomputation
        # matches the in-memory plan exactly.
        for key in ("satellite_present", "terrain_present"):
            value = row.get(key)
            row[key] = value in (True, "True", "true", "1")
        coord = (int(row["x"]), int(row["y"]))
        planned["ordered_coords"].append(coord)
        planned["coord_to_region"][coord] = row["region"]
    # The execution manifest is extended (output hashes) as tiles complete, so
    # the plan binds the coordinate list itself; execution verifies it from the
    # parsed rows rather than from the mutable file hash.
    current_coord_sha = identity_digest(
        [(row["region"], int(row["z"]), int(row["x"]), int(row["y"])) for row in rows]
    )
    if current_coord_sha != plan["hashes"]["coordinate_list_sha256"]:
        raise PlanDriftError(
            "tile manifest coordinate list SHA-256 drift: computed "
            f"{current_coord_sha}, plan expects "
            f"{plan['hashes']['coordinate_list_sha256']}"
        )
    tile_plans_result = build_tile_plans(
        planned=planned,
        contract=contract,
        imagery_catalog=imagery_catalog,
        terrain_assets=terrain_assets,
        zoom=int(plan["zoom"]),
    )
    recomputed = build_execution_plan(
        run_name=run_name,
        zoom=int(plan["zoom"]),
        contract=contract,
        contract_sha=contract_sha,
        contract_path=contract_path,
        spec_path=plan["inputs"]["region_spec_path"],
        spec_sha=plan["hashes"]["region_spec_sha256"],
        geometry_digest=plan["hashes"]["geometry_digest"],
        imagery_catalog_path=imagery_catalog_path,
        imagery_catalog_sha=plan["hashes"]["imagery_catalog_sha256"],
        terrain_catalog_path=terrain_catalog_path,
        terrain_catalog_sha=plan["hashes"]["terrain_catalog_sha256"],
        tile_plans_result=tile_plans_result,
        planned_rows=rows,
        tile_manifest_sha=plan["hashes"]["tile_manifest_sha256"],
        pilot_frame_sha=plan["hashes"]["pilot_frame_sha256"],
        budget=int(plan["inputs"]["budget"]),
        pilot_budget=plan["inputs"]["pilot_budget"],
        image_root=image_root,
        output_root=output_root,
        config_path=plan["inputs"]["app_regions_config"],
        imagery_source=plan["inputs"]["imagery_source"],
        terrain_source=plan["inputs"]["terrain_source"],
        max_runtime_minutes=plan["deadline"]["max_runtime_minutes"],
        land_geometry_path=land_geometry_path,
        land_geometry_sha256=land_geometry_sha,
        # Presence-dependent counts are immutable plan facts (the acquisition
        # target), not recomputed from the manifest which execution extends.
        missing_satellite=int(plan["missing_satellite"]),
        missing_terrain=int(plan["missing_terrain"]),
        local_reuse_count=int(plan["local_reuse_count"]),
        repository_source_tree_digest=current_source_digest,
    )
    if canonical_json_bytes(recomputed) != canonical_json_bytes(plan):
        raise PlanDriftError(
            "recomputed execution plan does not match the immutable plan"
        )

    # --- transports -------------------------------------------------------
    ledger = MeteredLedger(run_dir / "request_transfer_ledger.jsonl")
    rate_card = RateCard(
        source=contract["rate_card"]["source"],
        date=contract["rate_card"]["date"],
        request_cost_usd=Decimal(contract["rate_card"]["request_cost_usd"]),
        transfer_cost_per_gb_usd=Decimal(
            contract["rate_card"]["transfer_cost_per_gb_usd"]
        ),
    )
    shared_budget = SharedMeteredBudget(caps)
    transports: dict[str, MeteredTransport] = {}
    imagery_transport = transport_factory(
        contract["imagery"]["bucket"],
        caps,
        ledger,
        rate_card,
        bool(contract["imagery"].get("requester_pays")),
        shared_budget,
    )
    terrain_transport = transport_factory(
        contract["terrain"]["bucket"],
        caps,
        ledger,
        rate_card,
        bool(contract["terrain"].get("requester_pays")),
        shared_budget,
    )
    transports[contract["imagery"]["bucket"]] = imagery_transport
    transports[contract["terrain"]["bucket"]] = terrain_transport

    # Concurrent tile workers share one single-flight so the same object is
    # acquired at most once per (bucket, key, etag) identity.
    object_flights = _SingleFlight()

    # --- resume state -----------------------------------------------------
    failures_path = run_dir / "failures.jsonl"
    provenance_path = run_dir / "nested_provenance.jsonl"
    tile_plans_by_key = tile_plans_result["tile_plans"]
    state_index = tile_plans_result.get("state_index")
    terrain_asset_list = tile_plans_result.get("terrain_asset_list") or []
    terrain_bbox_table = tile_plans_result.get("terrain_bbox_table")
    if terrain_bbox_table is None:
        terrain_bbox_table = np.array(
            [footprint_bounds(a) for a in terrain_asset_list], dtype=np.float64
        )

    asset_by_id: dict[str, SourceAsset] = {}
    if state_index is not None:
        for state_assets in state_index.ordered.values():
            for asset in state_assets:
                asset_by_id[asset.asset_id] = asset
    for asset in terrain_asset_list:
        asset_by_id[asset.asset_id] = asset

    imagery_estimates = contract["imagery"]["estimates"]
    terrain_estimates = contract["terrain"]["estimates"]
    sat_upper = int(imagery_estimates["per_asset_bytes_upper_bound"])
    ter_upper = int(terrain_estimates["per_asset_bytes_upper_bound"])
    preprocessing = contract["preprocessing"]
    tile_width = int(preprocessing["tile_width"])
    tile_height = int(preprocessing["tile_height"])
    generated_output_reservation = (
        2 * (tile_width * tile_height * 4 + 1024 * 1024) + 256 * 1024
    )

    def reserve_generated_output(coord_key: str) -> None:
        shared_budget.reserve(
            requests=0,
            transfer_bytes=0,
            local_bytes=generated_output_reservation,
            cost_usd=Decimal("0"),
        )
        try:
            ledger.append(
                {
                    "event": "reserve",
                    "operation": "generated_tile_pair",
                    "key": coord_key,
                    "reserved_requests": 0,
                    "reserved_transfer_bytes": 0,
                    "reserved_local_bytes": generated_output_reservation,
                    "reserved_cost_usd": "0",
                }
            )
        except Exception:
            shared_budget.adjust(local_bytes=-generated_output_reservation)
            raise

    def settle_generated_output(coord_key: str, paths: Sequence[Path]) -> None:
        actual_local = sum(path.stat().st_size for path in paths if path.is_file())
        delta = actual_local - generated_output_reservation
        shared_budget.adjust(local_bytes=delta)
        try:
            ledger.append(
                {
                    "event": "settle",
                    "operation": "generated_tile_pair",
                    "key": coord_key,
                    "actual_transfer_bytes": 0,
                    "actual_local_bytes": actual_local,
                    "actual_cost_usd": "0",
                    "outcome": "ok",
                }
            )
        except Exception:
            shared_budget.adjust(local_bytes=-delta)
            raise

    row_by_key = {f"{int(r['x'])}_{int(r['y'])}": r for r in rows}
    start_wall = time.monotonic()
    deadline_seconds = (
        max_runtime_minutes * 60 if max_runtime_minutes is not None else None
    )
    abort_state: dict[str, Any] = {"set": False, "reason": None}
    results: dict[str, _TileOutcome] = {}
    _results_lock = threading.Lock()

    def process_one(coord_key: str) -> _TileOutcome:
        row = row_by_key[coord_key]
        x, y = int(row["x"]), int(row["y"])
        z = int(row["z"])
        region = str(row["region"])
        record_path = _record_path(run_dir, region, z, x, y)
        tile_plan = tile_plans_by_key[coord_key]

        # Ownership-safe resume: skip only exact valid matches of this run.
        sat_path = image_root / "satellite" / f"z{z}" / region / f"{x}_{y}.png"
        ter_path = image_root / "terrain" / f"z{z}" / region / f"{x}_{y}.png"
        record_ok, _reason, record = _validate_existing_record(
            sat_path=sat_path,
            ter_path=ter_path,
            record_path=record_path,
            run_id=run_name,
            plan_sha=expected_plan_sha256,
        )
        if record_ok:
            return _TileOutcome(
                coord_key=coord_key, provenance=None, failure=None, skipped=True
            )

        # Run-owned corrupt outputs are quarantined (never deleted, never foreign).
        for path in (sat_path, ter_path):
            if path.is_file():
                if record_path.is_file():
                    try:
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                    except Exception:
                        record = {}
                    if (
                        record.get("run_id") == run_name
                        and record.get("execution_plan_sha256") == expected_plan_sha256
                    ):
                        _quarantine(path)
                    else:
                        return _TileOutcome(
                            coord_key=coord_key,
                            provenance=None,
                            failure={
                                "region": region,
                                "z": z,
                                "x": x,
                                "y": y,
                                "style": "pair",
                                "reason": "foreign_owner_refused",
                                "category": "ownership",
                            },
                        )
                else:
                    # File exists with no run record: unowned -> immutable.
                    return _TileOutcome(
                        coord_key=coord_key,
                        provenance=None,
                        failure={
                            "region": region,
                            "z": z,
                            "x": x,
                            "y": y,
                            "style": "pair",
                            "reason": "unowned_output_refused",
                            "category": "ownership",
                        },
                    )

        if record_path.is_file():
            if not isinstance(record, dict):
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except Exception:
                    record = None
            if not (
                isinstance(record, dict)
                and record.get("run_id") == run_name
                and record.get("execution_plan_sha256") == expected_plan_sha256
            ):
                return _TileOutcome(
                    coord_key=coord_key,
                    provenance=None,
                    failure={
                        "region": region,
                        "z": z,
                        "x": x,
                        "y": y,
                        "style": "pair",
                        "reason": "foreign_owner_refused",
                        "category": "ownership",
                    },
                )

        satellite_asset_ids = tile_plan["satellite_asset_ids"]
        terrain_asset_ids = tile_plan["terrain_asset_ids"]
        if not satellite_asset_ids or not terrain_asset_ids:
            return _TileOutcome(
                coord_key=coord_key,
                provenance=None,
                failure={
                    "region": region,
                    "z": z,
                    "x": x,
                    "y": y,
                    "style": "pair",
                    "reason": "missing_asset_assignment",
                    "category": "plan",
                    "satellite_assets": len(satellite_asset_ids),
                    "terrain_assets": len(terrain_asset_ids),
                },
            )

        satellite_paths: list[Path] = []
        for asset_id in satellite_asset_ids:
            asset = asset_by_id[asset_id]
            path, _ = _ensure_object(
                transport=imagery_transport,
                cache_root=cache_root,
                bucket=contract["imagery"]["bucket"],
                key=asset.canonical_uri,
                etag=asset.etag,
                max_response_bytes=sat_upper,
                ledger=ledger,
                flights=object_flights,
            )
            satellite_paths.append(path)
        terrain_paths: list[Path] = []
        for asset_id in terrain_asset_ids:
            asset = asset_by_id[asset_id]
            path, _ = _ensure_object(
                transport=terrain_transport,
                cache_root=cache_root,
                bucket=contract["terrain"]["bucket"],
                key=asset.canonical_uri,
                etag=asset.etag,
                max_response_bytes=ter_upper,
                ledger=ledger,
                flights=object_flights,
            )
            terrain_paths.append(path)

        reserve_generated_output(coord_key)
        installing_record = {
            "state": "installing",
            "run_id": run_name,
            "execution_plan_sha256": expected_plan_sha256,
            "source_contract_sha256": contract_sha,
            "preprocessing_contract_sha256": preprocessing_sha,
            "satellite": {"path": str(sat_path)},
            "terrain": {"path": str(ter_path)},
        }
        try:
            atomic_write_json(record_path, installing_record)
        except Exception:
            settle_generated_output(coord_key, (sat_path, ter_path, record_path))
            raise

        try:
            provenance = process_tile_pair(
                z=z,
                x=x,
                y=y,
                region=region,
                satellite_paths=satellite_paths,
                terrain_paths=terrain_paths,
                image_root=image_root,
                terrain_vertical_datums=[
                    asset_by_id[asset_id].vertical_datum
                    for asset_id in terrain_asset_ids
                ],
                contract=contract,
                land_geometry=land_geometry,
            )
        except (NoDataOnLandError, TerrainRGBRangeError, ValueError) as exc:
            settle_generated_output(coord_key, (sat_path, ter_path, record_path))
            return _TileOutcome(
                coord_key=coord_key,
                provenance=None,
                failure={
                    "region": region,
                    "z": z,
                    "x": x,
                    "y": y,
                    "style": "pair",
                    "reason": type(exc).__name__,
                    "category": "processing",
                    "detail": str(exc)[:400],
                },
            )
        except Exception:
            settle_generated_output(coord_key, (sat_path, ter_path, record_path))
            raise

        provenance["satellite"]["asset_ids"] = satellite_asset_ids
        provenance["terrain"]["asset_ids"] = terrain_asset_ids
        provenance["state"] = tile_plan.get("state")
        provenance["warnings"] = tile_plan.get("warnings", [])
        record = {
            "state": "complete",
            "run_id": run_name,
            "execution_plan_sha256": expected_plan_sha256,
            "source_contract_sha256": contract_sha,
            "preprocessing_contract_sha256": preprocessing_sha,
            "grid_sha256": provenance["grid_sha256"],
            "satellite": {
                "path": provenance["satellite"]["path"],
                "sha256": provenance["satellite"]["output_sha256"],
                "valid_fraction": provenance["satellite"]["valid_fraction"],
            },
            "terrain": {
                "path": provenance["terrain"]["path"],
                "sha256": provenance["terrain"]["output_sha256"],
                "valid_fraction": provenance["terrain"]["valid_fraction"],
            },
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            atomic_write_json(record_path, record)
        except Exception:
            settle_generated_output(coord_key, (sat_path, ter_path, record_path))
            raise
        settle_generated_output(coord_key, (sat_path, ter_path, record_path))
        return _TileOutcome(coord_key=coord_key, provenance=provenance, failure=None)

    ordered_keys = [f"{int(r['x'])}_{int(r['y'])}" for r in rows]
    # The checkpoint is a progress record only: every tile (including tiles the
    # checkpoint lists as complete) is re-validated against its run record and
    # recorded output SHA-256s before being skipped, so corrupt or tampered
    # outputs can never be silently treated as complete.
    pending_keys = list(ordered_keys)

    if pending_keys:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="naip-3dep-execute"
        ) as executor:
            pending: dict[Future[_TileOutcome], str] = {}
            for key in pending_keys:
                if abort_state["set"]:
                    break
                if deadline_seconds is not None and (
                    time.monotonic() - start_wall >= deadline_seconds
                ):
                    abort_state["set"] = True
                    abort_state["reason"] = "deadline"
                    break
                pending[executor.submit(process_one, key)] = key
                if len(pending) >= workers * 2:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        key = pending.pop(future)
                        try:
                            outcome = future.result()
                        except CapExceededError as exc:
                            outcome = _TileOutcome(
                                coord_key=key,
                                provenance=None,
                                failure={
                                    "coord_key": key,
                                    "reason": "cap_exceeded",
                                    "category": "cap",
                                    "detail": str(exc)[:400],
                                },
                            )
                            abort_state["set"] = True
                            abort_state["reason"] = "cap"
                        except Exception as exc:  # fail closed
                            outcome = _TileOutcome(
                                coord_key=key,
                                provenance=None,
                                failure={
                                    "coord_key": key,
                                    "reason": type(exc).__name__,
                                    "category": "fatal",
                                    "detail": str(exc)[:400],
                                },
                            )
                        with _results_lock:
                            results[key] = outcome
                        if abort_state["set"]:
                            for fut in pending:
                                fut.cancel()
                            break
            for future, key in list(pending.items()):
                if future.cancelled():
                    continue
                try:
                    outcome = future.result()
                except CapExceededError as exc:
                    outcome = _TileOutcome(
                        coord_key=key,
                        provenance=None,
                        failure={
                            "coord_key": key,
                            "reason": "cap_exceeded",
                            "category": "cap",
                            "detail": str(exc)[:400],
                        },
                    )
                    abort_state["set"] = True
                    abort_state["reason"] = "cap"
                except Exception as exc:
                    outcome = _TileOutcome(
                        coord_key=key,
                        provenance=None,
                        failure={
                            "coord_key": key,
                            "reason": type(exc).__name__,
                            "category": "fatal",
                            "detail": str(exc)[:400],
                        },
                    )
                with _results_lock:
                    results[key] = outcome

    # --- record outcomes deterministically (main thread) ------------------
    failures: list[dict[str, Any]] = []
    provenance_lines: list[dict[str, Any]] = []
    for key in ordered_keys:
        outcome = results.get(key)
        if outcome is None:
            continue
        if outcome.skipped:
            continue
        if outcome.failure is not None:
            failures.append(outcome.failure)
            append_jsonl(failures_path, outcome.failure)
            continue
        assert outcome.provenance is not None
        provenance_lines.append(outcome.provenance)
        row = row_by_key[key]
        sat = outcome.provenance["satellite"]
        ter = outcome.provenance["terrain"]
        row["satellite_output_sha256"] = sat["output_sha256"]
        row["satellite_valid_fraction"] = f"{sat['valid_fraction']:.6f}"
        row["terrain_output_sha256"] = ter["output_sha256"]
        row["terrain_valid_fraction"] = f"{ter['valid_fraction']:.6f}"
        row["grid_sha256"] = outcome.provenance["grid_sha256"]
        tile_plan = tile_plans_by_key[key]
        row["satellite_asset_ids"] = json.dumps(
            tile_plan["satellite_asset_ids"], sort_keys=True, separators=(",", ":")
        )
        row["terrain_asset_ids"] = json.dumps(
            tile_plan["terrain_asset_ids"], sort_keys=True, separators=(",", ":")
        )

        def _asset_contributions(
            asset_ids: Sequence[str], qc: Mapping[str, Any]
        ) -> list[dict[str, Any]]:
            contributions = qc.get("source_contributions", [])
            if len(contributions) != len(asset_ids):
                raise ValueError("source contribution count does not match asset plan")
            return [
                {
                    "asset_id": asset_id,
                    "pixel_count": int(contribution["pixel_count"]),
                    "pixel_fraction": float(contribution["pixel_fraction"]),
                }
                for asset_id, contribution in zip(asset_ids, contributions, strict=True)
            ]

        row["mosaic_contributions"] = json.dumps(
            {
                "satellite": _asset_contributions(
                    tile_plan["satellite_asset_ids"], sat["qc"]
                ),
                "terrain": _asset_contributions(
                    tile_plan["terrain_asset_ids"], ter["qc"]
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    for outcome in results.values():
        if outcome.skipped:
            key = outcome.coord_key
            row = row_by_key[key]
            record_path = _record_path(
                run_dir, str(row["region"]), int(row["z"]), int(row["x"]), int(row["y"])
            )
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            row["satellite_output_sha256"] = record.get("satellite", {}).get(
                "sha256", ""
            )
            row["terrain_output_sha256"] = record.get("terrain", {}).get("sha256", "")
            row["satellite_valid_fraction"] = str(
                record.get("satellite", {}).get("valid_fraction", "")
            )
            row["terrain_valid_fraction"] = str(
                record.get("terrain", {}).get("valid_fraction", "")
            )
            row["grid_sha256"] = record.get("grid_sha256", "")

    for row in rows:
        if not all(
            row.get(name)
            for name in (
                "grid_sha256",
                "satellite_output_sha256",
                "terrain_output_sha256",
            )
        ):
            continue
        row["acquisition_tile_identity_sha256"] = compute_acquisition_tile_identity(
            z=int(row["z"]),
            x=int(row["x"]),
            y=int(row["y"]),
            region=str(row["region"]),
            target_grid_sha256=str(row["grid_sha256"]),
            satellite_output_sha256=str(row["satellite_output_sha256"]),
            terrain_output_sha256=str(row["terrain_output_sha256"]),
            source_contract_sha256=contract_sha,
            preprocessing_contract_sha256=preprocessing_sha,
        ).sha256()

    provenance_lines.sort(
        key=lambda p: (p["region"], int(p["z"]), int(p["x"]), int(p["y"]))
    )
    for line in provenance_lines:
        append_jsonl(provenance_path, line)

    # --- finalize manifests -------------------------------------------------
    new_completed = {
        key
        for key in ordered_keys
        if results.get(key) is not None
        and (results[key].skipped or results[key].provenance is not None)
    }
    if abort_state["set"]:
        final_state = (
            "aborted_cap" if abort_state["reason"] == "cap" else "paused_deadline"
        )
    elif failures or len(new_completed) != len(ordered_keys):
        final_state = "failed"
    else:
        final_state = "acquired"
    _write_checkpoint(
        run_dir,
        run_name=run_name,
        plan_sha=expected_plan_sha256,
        completed=sorted(new_completed),
        counters=shared_budget.counters(),
        state=final_state,
    )

    # Re-inventory and re-assess water so the manifest is truthful.
    from src.data_pipeline.tile_inventory import scan_tile_inventory

    rows, counts = scan_tile_inventory(
        rows, image_root=image_root, expected_dimensions=(512, 512)
    )
    rows.sort(key=lambda r: (r["region"], int(r["z"]), int(r["x"]), int(r["y"])))
    write_manifest_csv(run_dir / "tile_manifest.csv", rows)

    counters = shared_budget.counters()
    return {
        "run_dir": str(run_dir),
        "plan_sha256": expected_plan_sha256,
        "state": final_state,
        "completed_coordinates": len(new_completed),
        "failed_coordinates": len(failures),
        "pending_coordinates": len(ordered_keys) - len(new_completed),
        "counters": {
            "requests": counters.requests,
            "transfer_bytes": counters.transfer_bytes,
            "local_bytes": counters.local_bytes,
            "cost_usd": str(counters.cost_usd),
        },
        "checkpoint_path": str(run_dir / "checkpoint.json"),
        "abort_reason": abort_state["reason"],
    }
