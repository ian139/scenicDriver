"""Typed, immutable source contracts, preprocessing contracts, and identity models."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

FORBIDDEN_QUERY_PARAMS = {
    "signature",
    "sig",
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-date",
    "x-amz-algorithm",
    "awsaccesskeyid",
    "access_token",
    "token",
    "auth",
    "api_key",
    "apikey",
    "secret",
    "password",
    "sas",
    "se",
    "sp",
    "sv",
    "sr",
    "st",
}
VALID_URI_SCHEMES = {"http", "https", "s3", "file", "gs", "az", "s3a", "s3n"}


def validate_canonical_uri(uri: str) -> None:
    """Validate that URI is canonical and contains no credentials or secret parameters."""
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("URI must be a non-empty string")
    if re.search(r"\s", uri):
        raise ValueError(f"URI contains invalid whitespace: {uri!r}")

    parsed = urlparse(uri)
    if not parsed.scheme or parsed.scheme.lower() not in VALID_URI_SCHEMES:
        raise ValueError(
            f"URI scheme {parsed.scheme!r} is noncanonical or unsupported in {uri!r}"
        )

    if (
        "@" in parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"URI contains credentials or userinfo: {uri!r}")

    if parsed.query:
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        for key in query_params:
            if key.lower() in FORBIDDEN_QUERY_PARAMS:
                raise ValueError(
                    f"URI contains forbidden secret parameter {key!r}: {uri!r}"
                )


def validate_sha256(val: str | None, name: str) -> None:
    """Validate optional 64-character lowercase hex SHA-256 string."""
    if val is None:
        return
    if not isinstance(val, str) or isinstance(val, bool):
        raise ValueError(f"{name} must be a hex string, got {type(val).__name__}")
    if len(val) != 64 or not re.match(r"^[0-9a-f]{64}$", val):
        raise ValueError(
            f"{name} must be a 64-character lowercase hex SHA-256 string, got {val!r}"
        )


def validate_acquisition_year(year: int | None) -> None:
    """Validate optional acquisition year integer."""
    if year is None:
        return
    if not isinstance(year, int) or isinstance(year, bool):
        raise ValueError(
            f"acquisition_year must be an integer, got {type(year).__name__}"
        )
    if not (1800 <= year <= 2100):
        raise ValueError(f"acquisition_year must be between 1800 and 2100, got {year}")


def validate_resolution(
    res: float | tuple[float, ...] | list[float] | int | None, name: str
) -> None:
    """Validate optional positive numeric resolution value or tuple/list of resolutions."""
    if res is None:
        return
    if isinstance(res, (tuple, list)):
        if len(res) == 0:
            raise ValueError(f"{name} must be a non-empty tuple/list, got {res}")
        for val in res:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or val <= 0
            ):
                raise ValueError(
                    f"{name} elements must be positive finite numbers, got {val!r}"
                )
        return
    if not isinstance(res, (int, float)) or isinstance(res, bool):
        raise ValueError(
            f"{name} must be numeric or tuple/list, got {type(res).__name__}"
        )
    if not math.isfinite(res) or res <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {res}")


def validate_footprint_geojson(geojson_str: str | None) -> None:
    """Validate optional GeoJSON string for valid geometry structure and canonical JSON formatting."""
    if geojson_str is None:
        return
    if not isinstance(geojson_str, str):
        raise ValueError("footprint_geojson must be a string")
    try:
        data = json.loads(geojson_str)
    except Exception as exc:
        raise ValueError(f"footprint_geojson is invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("footprint_geojson JSON must be a GeoJSON object/dict")

    if "type" not in data or not isinstance(data["type"], str):
        raise ValueError("footprint_geojson must have a valid 'type' string field")

    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if geojson_str != canonical:
        raise ValueError(
            f"footprint_geojson is noncanonical. Expected compact sorted JSON:\n{canonical!r}\ngot:\n{geojson_str!r}"
        )


def _canonical_json_bytes(d: dict[str, Any]) -> bytes:
    return json.dumps(
        d, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_dict(d: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(d)).hexdigest()


@dataclass(frozen=True)
class SourceAsset:
    """Immutable representation of a single remote or local source raster asset."""

    provider: str
    collection: str
    asset_id: str
    canonical_uri: str
    state_or_region: str | None = None
    acquisition_year: int | None = None
    capture_date: str | None = None
    published_at: str | None = None
    license: str | None = None
    attribution: str | None = None
    horizontal_crs: str | None = None
    vertical_datum: str | None = None
    native_resolution: float | tuple[float, ...] | None = None
    band_contract: dict[str, Any] | list[Any] | str | None = None
    nodata: int | float | None = None
    etag: str | None = None
    version_id: str | None = None
    checksum_sha256: str | None = None
    metadata_sha256: str | None = None
    object_size_bytes: int | None = None
    last_modified: str | None = None
    accept_ranges: bool | None = None
    footprint_geojson: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.collection, str) or not self.collection.strip():
            raise ValueError("collection must be a non-empty string")
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")

        validate_canonical_uri(self.canonical_uri)
        validate_acquisition_year(self.acquisition_year)
        validate_resolution(self.native_resolution, "native_resolution")
        validate_sha256(self.checksum_sha256, "checksum_sha256")
        validate_sha256(self.metadata_sha256, "metadata_sha256")
        validate_footprint_geojson(self.footprint_geojson)
        if self.object_size_bytes is not None and (
            isinstance(self.object_size_bytes, bool)
            or not isinstance(self.object_size_bytes, int)
            or self.object_size_bytes < 0
        ):
            raise ValueError("object_size_bytes must be a non-negative integer")
        if self.accept_ranges is not None and not isinstance(self.accept_ranges, bool):
            raise ValueError("accept_ranges must be a boolean or None")

        if self.nodata is not None and (
            isinstance(self.nodata, bool) or not isinstance(self.nodata, (int, float))
        ):
            raise ValueError(
                f"nodata must be numeric or None, got {type(self.nodata).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "provider": self.provider,
            "collection": self.collection,
            "asset_id": self.asset_id,
            "canonical_uri": self.canonical_uri,
            "state_or_region": self.state_or_region,
            "acquisition_year": self.acquisition_year,
            "capture_date": self.capture_date,
            "published_at": self.published_at,
            "license": self.license,
            "attribution": self.attribution,
            "horizontal_crs": self.horizontal_crs,
            "vertical_datum": self.vertical_datum,
            "native_resolution": list(self.native_resolution)
            if isinstance(self.native_resolution, tuple)
            else self.native_resolution,
            "band_contract": self.band_contract,
            "nodata": self.nodata,
            "etag": self.etag,
            "version_id": self.version_id,
            "checksum_sha256": self.checksum_sha256,
            "metadata_sha256": self.metadata_sha256,
            "object_size_bytes": self.object_size_bytes,
            "last_modified": self.last_modified,
            "accept_ranges": self.accept_ranges,
            "footprint_geojson": self.footprint_geojson,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceAsset:
        if not isinstance(data, dict):
            raise ValueError("from_dict expects a dict")
        known_fields = {f.name for f in fields(cls)}
        extra = set(data.keys()) - known_fields
        if extra:
            raise ValueError(
                f"Unknown extra fields for {cls.__name__}: {sorted(extra)}"
            )
        data_copy = dict(data)
        if isinstance(data_copy.get("native_resolution"), list):
            data_copy["native_resolution"] = tuple(data_copy["native_resolution"])
        return cls(**data_copy)

    def sha256(self) -> str:
        return _sha256_dict(self.to_dict())


@dataclass(frozen=True)
class SourceContract:
    """Immutable collection contract representing provider catalog specifications."""

    provider: str
    collection: str
    assets: tuple[SourceAsset, ...] = ()
    horizontal_crs: str | None = None
    vertical_datum: str | None = None
    native_resolution: float | tuple[float, ...] | None = None
    band_contract: dict[str, Any] | None = None
    contract_version: str = "1.0"

    def __post_init__(self) -> None:
        if isinstance(self.assets, list):
            object.__setattr__(self, "assets", tuple(self.assets))
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if not isinstance(self.collection, str) or not self.collection.strip():
            raise ValueError("collection must be a non-empty string")
        if (
            not isinstance(self.contract_version, str)
            or not self.contract_version.strip()
        ):
            raise ValueError("contract_version must be a non-empty string")
        validate_resolution(self.native_resolution, "native_resolution")
        for idx, asset in enumerate(self.assets):
            if not isinstance(asset, SourceAsset):
                raise ValueError(
                    f"Asset at index {idx} must be a SourceAsset instance, got {type(asset).__name__}"
                )
            asset.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "provider": self.provider,
            "collection": self.collection,
            "assets": [asset.to_dict() for asset in self.assets],
            "horizontal_crs": self.horizontal_crs,
            "vertical_datum": self.vertical_datum,
            "native_resolution": list(self.native_resolution)
            if isinstance(self.native_resolution, tuple)
            else self.native_resolution,
            "band_contract": self.band_contract,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceContract:
        if not isinstance(data, dict):
            raise ValueError("from_dict expects a dict")
        known_fields = {f.name for f in fields(cls)}
        extra = set(data.keys()) - known_fields
        if extra:
            raise ValueError(
                f"Unknown extra fields for {cls.__name__}: {sorted(extra)}"
            )

        data_copy = dict(data)
        if "assets" in data_copy and data_copy["assets"] is not None:
            raw_assets = data_copy["assets"]
            if not isinstance(raw_assets, (list, tuple)):
                raise ValueError("assets must be a list or tuple")
            parsed_assets = []
            for item in raw_assets:
                if isinstance(item, dict):
                    parsed_assets.append(SourceAsset.from_dict(item))
                elif isinstance(item, SourceAsset):
                    parsed_assets.append(item)
                else:
                    raise ValueError(f"Invalid asset item: {item!r}")
            data_copy["assets"] = tuple(parsed_assets)
        if "native_resolution" in data_copy and isinstance(
            data_copy["native_resolution"], list
        ):
            data_copy["native_resolution"] = tuple(data_copy["native_resolution"])
        return cls(**data_copy)

    def sha256(self) -> str:
        return _sha256_dict(self.to_dict())


@dataclass(frozen=True)
class PreprocessingContract:
    """Immutable preprocessing specification contract."""

    contract_id: str
    target_crs: str
    target_resolution: float
    resample_alg: str = "bilinear"
    band_mapping: dict[str, Any] | None = None
    nodata_value: float | int | None = None
    normalization_params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.contract_id, str) or not self.contract_id.strip():
            raise ValueError("contract_id must be a non-empty string")
        if not isinstance(self.target_crs, str) or not self.target_crs.strip():
            raise ValueError("target_crs must be a non-empty string")
        validate_resolution(self.target_resolution, "target_resolution")
        if not isinstance(self.resample_alg, str) or not self.resample_alg.strip():
            raise ValueError("resample_alg must be a non-empty string")
        if self.nodata_value is not None and (
            isinstance(self.nodata_value, bool)
            or not isinstance(self.nodata_value, (int, float))
        ):
            raise ValueError(
                f"nodata_value must be numeric or None, got {type(self.nodata_value).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "contract_id": self.contract_id,
            "target_crs": self.target_crs,
            "target_resolution": self.target_resolution,
            "resample_alg": self.resample_alg,
            "band_mapping": self.band_mapping,
            "nodata_value": self.nodata_value,
            "normalization_params": self.normalization_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingContract:
        if not isinstance(data, dict):
            raise ValueError("from_dict expects a dict")
        known_fields = {f.name for f in fields(cls)}
        extra = set(data.keys()) - known_fields
        if extra:
            raise ValueError(
                f"Unknown extra fields for {cls.__name__}: {sorted(extra)}"
            )
        return cls(**data)

    def sha256(self) -> str:
        return _sha256_dict(self.to_dict())


@dataclass(frozen=True)
class AcquisitionTileIdentity:
    """Immutable model-independent identity for one processed acquisition tile."""

    z: int
    x: int
    y: int
    region: str
    target_grid_sha256: str
    satellite_output_sha256: str
    terrain_output_sha256: str
    source_contract_sha256: str
    preprocessing_contract_sha256: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("z", "x", "y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        if not isinstance(self.region, str) or not self.region.strip():
            raise ValueError("region must be a non-empty string")
        for name in (
            "target_grid_sha256",
            "satellite_output_sha256",
            "terrain_output_sha256",
            "source_contract_sha256",
            "preprocessing_contract_sha256",
        ):
            validate_sha256(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "z": self.z,
            "x": self.x,
            "y": self.y,
            "region": self.region,
            "target_grid_sha256": self.target_grid_sha256,
            "satellite_output_sha256": self.satellite_output_sha256,
            "terrain_output_sha256": self.terrain_output_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "preprocessing_contract_sha256": self.preprocessing_contract_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcquisitionTileIdentity:
        if not isinstance(data, dict):
            raise ValueError("from_dict expects a dict")
        known_fields = {f.name for f in fields(cls)}
        extra = set(data) - known_fields
        if extra:
            raise ValueError(
                f"Unknown extra fields for {cls.__name__}: {sorted(extra)}"
            )
        return cls(**data)

    def sha256(self) -> str:
        return _sha256_dict(self.to_dict())


@dataclass(frozen=True)
class EmbeddingFeatureIdentity:
    """Identity for classifier embeddings and terrain features."""

    acquisition_tile_sha256: str
    classifier_preprocessing_sha256: str
    classifier_checkpoint_sha256: str
    terrain_feature_schema: str
    terrain_feature_version: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_sha256(self.acquisition_tile_sha256, "acquisition_tile_sha256")
        validate_sha256(
            self.classifier_preprocessing_sha256,
            "classifier_preprocessing_sha256",
        )
        validate_sha256(
            self.classifier_checkpoint_sha256, "classifier_checkpoint_sha256"
        )
        for name in ("terrain_feature_schema", "terrain_feature_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "acquisition_tile_sha256": self.acquisition_tile_sha256,
            "classifier_preprocessing_sha256": self.classifier_preprocessing_sha256,
            "classifier_checkpoint_sha256": self.classifier_checkpoint_sha256,
            "terrain_feature_schema": self.terrain_feature_schema,
            "terrain_feature_version": self.terrain_feature_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingFeatureIdentity:
        if not isinstance(data, dict):
            raise ValueError("from_dict expects a dict")
        known_fields = {f.name for f in fields(cls)}
        extra = set(data) - known_fields
        if extra:
            raise ValueError(
                f"Unknown extra fields for {cls.__name__}: {sorted(extra)}"
            )
        return cls(**data)

    def sha256(self) -> str:
        return _sha256_dict(self.to_dict())


@dataclass(frozen=True)
class PredictionReportIdentity:
    """Identity for one calibrated, schema-versioned prediction report."""

    embedding_feature_sha256: str
    regression_checkpoint_sha256: str
    calibration_type: str
    calibration_config_sha256: str
    calibration_artifact_sha256: str
    score_schema_version: str
    label_schema_version: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "embedding_feature_sha256",
            "regression_checkpoint_sha256",
            "calibration_config_sha256",
            "calibration_artifact_sha256",
        ):
            validate_sha256(getattr(self, name), name)
        for name in (
            "calibration_type",
            "score_schema_version",
            "label_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "embedding_feature_sha256": self.embedding_feature_sha256,
            "regression_checkpoint_sha256": self.regression_checkpoint_sha256,
            "calibration_type": self.calibration_type,
            "calibration_config_sha256": self.calibration_config_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "score_schema_version": self.score_schema_version,
            "label_schema_version": self.label_schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionReportIdentity:
        if not isinstance(data, dict):
            raise ValueError("from_dict expects a dict")
        known_fields = {f.name for f in fields(cls)}
        extra = set(data) - known_fields
        if extra:
            raise ValueError(
                f"Unknown extra fields for {cls.__name__}: {sorted(extra)}"
            )
        return cls(**data)

    def sha256(self) -> str:
        return _sha256_dict(self.to_dict())


def compute_acquisition_tile_identity(
    *,
    z: int,
    x: int,
    y: int,
    region: str,
    target_grid_sha256: str,
    satellite_output_sha256: str,
    terrain_output_sha256: str,
    source_contract_sha256: str,
    preprocessing_contract_sha256: str,
) -> AcquisitionTileIdentity:
    """Build the identity of processed source bytes on an exact target grid."""
    return AcquisitionTileIdentity(
        z=z,
        x=x,
        y=y,
        region=region,
        target_grid_sha256=target_grid_sha256,
        satellite_output_sha256=satellite_output_sha256,
        terrain_output_sha256=terrain_output_sha256,
        source_contract_sha256=source_contract_sha256,
        preprocessing_contract_sha256=preprocessing_contract_sha256,
    )


def compute_embedding_feature_identity(
    acquisition_tile: AcquisitionTileIdentity | str,
    *,
    classifier_preprocessing_sha256: str,
    classifier_checkpoint_sha256: str,
    terrain_feature_schema: str,
    terrain_feature_version: str,
) -> EmbeddingFeatureIdentity:
    """Build an embedding identity from acquisition, classifier, and terrain inputs."""
    if isinstance(acquisition_tile, AcquisitionTileIdentity):
        acquisition_sha256 = acquisition_tile.sha256()
    elif isinstance(acquisition_tile, str):
        acquisition_sha256 = acquisition_tile
    else:
        raise ValueError(
            "acquisition_tile must be AcquisitionTileIdentity or sha256 str, "
            f"got {type(acquisition_tile).__name__}"
        )
    return EmbeddingFeatureIdentity(
        acquisition_tile_sha256=acquisition_sha256,
        classifier_preprocessing_sha256=classifier_preprocessing_sha256,
        classifier_checkpoint_sha256=classifier_checkpoint_sha256,
        terrain_feature_schema=terrain_feature_schema,
        terrain_feature_version=terrain_feature_version,
    )


def compute_prediction_report_identity(
    embedding_feature: EmbeddingFeatureIdentity | str,
    *,
    regression_checkpoint_sha256: str,
    calibration_type: str,
    calibration_config_sha256: str,
    calibration_artifact_sha256: str,
    score_schema_version: str,
    label_schema_version: str,
) -> PredictionReportIdentity:
    """Build a calibrated prediction-report identity."""
    if isinstance(embedding_feature, EmbeddingFeatureIdentity):
        embedding_sha256 = embedding_feature.sha256()
    elif isinstance(embedding_feature, str):
        embedding_sha256 = embedding_feature
    else:
        raise ValueError(
            "embedding_feature must be EmbeddingFeatureIdentity or sha256 str, "
            f"got {type(embedding_feature).__name__}"
        )
    return PredictionReportIdentity(
        embedding_feature_sha256=embedding_sha256,
        regression_checkpoint_sha256=regression_checkpoint_sha256,
        calibration_type=calibration_type,
        calibration_config_sha256=calibration_config_sha256,
        calibration_artifact_sha256=calibration_artifact_sha256,
        score_schema_version=score_schema_version,
        label_schema_version=label_schema_version,
    )
