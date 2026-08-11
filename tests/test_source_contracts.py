"""Tests for source contracts, validation, secret rejection, and layered identities."""

import json

import pytest

from src.data_pipeline.source_contracts import (
    AcquisitionTileIdentity,
    EmbeddingFeatureIdentity,
    PredictionReportIdentity,
    PreprocessingContract,
    SourceAsset,
    SourceContract,
    compute_acquisition_tile_identity,
    compute_embedding_feature_identity,
    compute_prediction_report_identity,
)


def sample_source_asset() -> SourceAsset:
    valid_footprint = json.dumps(
        {
            "coordinates": [
                [
                    [-78.0, 38.0],
                    [-77.0, 38.0],
                    [-77.0, 39.0],
                    [-78.0, 39.0],
                    [-78.0, 38.0],
                ]
            ],
            "type": "Polygon",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return SourceAsset(
        provider="USDA",
        collection="NAIP",
        asset_id="va_2021_1m_3707701_ne",
        canonical_uri="https://naip-analytic.s3.amazonaws.com/va/2021/1m/rgbir/37077/m_3707701_ne_18_060_20210522.tif",
        state_or_region="VA",
        acquisition_year=2021,
        capture_date="2021-05-22",
        published_at="2021-12-01T00:00:00Z",
        license="Public Domain",
        attribution="USDA Farm Service Agency",
        horizontal_crs="EPSG:26918",
        vertical_datum="NAVD88",
        native_resolution=1.0,
        band_contract={"bands": ["R", "G", "B", "NIR"]},
        nodata=0,
        etag='"123456789"',
        version_id="v1.0",
        checksum_sha256="a" * 64,
        metadata_sha256="b" * 64,
        footprint_geojson=valid_footprint,
    )


def sample_acquisition(**overrides) -> AcquisitionTileIdentity:
    values = {
        "z": 14,
        "x": 4800,
        "y": 6200,
        "region": "southern_new_england",
        "target_grid_sha256": "1" * 64,
        "satellite_output_sha256": "2" * 64,
        "terrain_output_sha256": "3" * 64,
        "source_contract_sha256": "4" * 64,
        "preprocessing_contract_sha256": "5" * 64,
    }
    values.update(overrides)
    return compute_acquisition_tile_identity(**values)


def sample_embedding(
    acquisition: AcquisitionTileIdentity | None = None, **overrides
) -> EmbeddingFeatureIdentity:
    values = {
        "classifier_preprocessing_sha256": "6" * 64,
        "classifier_checkpoint_sha256": "7" * 64,
        "terrain_feature_schema": "scenic-terrain-features",
        "terrain_feature_version": "3",
    }
    values.update(overrides)
    return compute_embedding_feature_identity(
        acquisition or sample_acquisition(), **values
    )


def sample_report(
    embedding: EmbeddingFeatureIdentity | None = None, **overrides
) -> PredictionReportIdentity:
    values = {
        "regression_checkpoint_sha256": "8" * 64,
        "calibration_type": "isotonic",
        "calibration_config_sha256": "9" * 64,
        "calibration_artifact_sha256": "a" * 64,
        "score_schema_version": "3",
        "label_schema_version": "2",
    }
    values.update(overrides)
    return compute_prediction_report_identity(embedding or sample_embedding(), **values)


def test_deterministic_round_trips():
    asset = sample_source_asset()
    asset_dict = asset.to_dict()
    asset_rt = SourceAsset.from_dict(asset_dict)
    assert asset == asset_rt
    assert asset.sha256() == asset_rt.sha256()

    contract = SourceContract(
        provider="USDA",
        collection="NAIP",
        assets=(asset,),
        horizontal_crs="EPSG:26918",
        native_resolution=1.0,
    )
    contract_dict = contract.to_dict()
    contract_rt = SourceContract.from_dict(contract_dict)
    assert contract == contract_rt
    assert contract.sha256() == contract_rt.sha256()

    preproc = PreprocessingContract(
        contract_id="naip_resample_v1",
        target_crs="EPSG:3857",
        target_resolution=1.0,
        resample_alg="bilinear",
    )
    preproc_dict = preproc.to_dict()
    preproc_rt = PreprocessingContract.from_dict(preproc_dict)
    assert preproc == preproc_rt
    assert preproc.sha256() == preproc_rt.sha256()

    acq_tile = sample_acquisition()
    acq_rt = AcquisitionTileIdentity.from_dict(acq_tile.to_dict())
    assert acq_tile == acq_rt
    assert acq_tile.sha256() == acq_rt.sha256()

    embed_feat = sample_embedding(acq_tile)
    embed_rt = EmbeddingFeatureIdentity.from_dict(embed_feat.to_dict())
    assert embed_feat == embed_rt
    assert embed_feat.sha256() == embed_rt.sha256()

    pred_report = sample_report(embed_feat)
    pred_rt = PredictionReportIdentity.from_dict(pred_report.to_dict())
    assert pred_report == pred_rt
    assert pred_report.sha256() == pred_rt.sha256()


def test_field_validation():
    # Test invalid SHA-256 strings (uppercase, wrong length, non-hex)
    with pytest.raises(ValueError, match="checksum_sha256"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            checksum_sha256="A" * 64,  # uppercase rejected
        )

    with pytest.raises(ValueError, match="checksum_sha256"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            checksum_sha256="a" * 63,  # wrong length
        )

    with pytest.raises(ValueError, match="checksum_sha256"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            checksum_sha256="g" * 64,  # non-hex
        )

    # Test invalid acquisition year
    with pytest.raises(ValueError, match="acquisition_year"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            acquisition_year=1750,
        )

    with pytest.raises(ValueError, match="acquisition_year"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            acquisition_year=True,
        )

    # Test invalid resolutions
    with pytest.raises(ValueError, match="native_resolution"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            native_resolution=-1.0,
        )

    with pytest.raises(ValueError, match="native_resolution"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            native_resolution=0.0,
        )

    # Test noncanonical / invalid footprint GeoJSON
    with pytest.raises(ValueError, match="footprint_geojson"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            footprint_geojson="invalid json string",
        )

    with pytest.raises(ValueError, match="footprint_geojson"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://example.com/a.tif",
            footprint_geojson=json.dumps(
                {"type": "Polygon"}, indent=2
            ),  # noncompact JSON rejected
        )

    # Test unknown extra fields in from_dict
    asset_dict = sample_source_asset().to_dict()
    asset_dict["unknown_field"] = "value"
    with pytest.raises(ValueError, match="Unknown extra fields"):
        SourceAsset.from_dict(asset_dict)


def test_secret_rejection():
    # Test rejection of basic auth user/pass credentials in URI
    with pytest.raises(ValueError, match="credentials or userinfo"):
        SourceAsset(
            provider="USDA",
            collection="NAIP",
            asset_id="a1",
            canonical_uri="https://user:pass@s3.amazonaws.com/bucket/file.tif",
        )

    # Test rejection of AWS signed query parameters
    secret_uris = [
        "https://s3.amazonaws.com/bucket/file.tif?AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE&Signature=vj22898",
        "https://s3.amazonaws.com/bucket/file.tif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=12345",
        "https://example.com/file.tif?access_token=secret_token_123",
        "s3://bucket/file.tif?token=abc",
        "https://example.com/file.tif?api_key=xyz",
        "https://example.com/file.tif?secret=topsecret",
    ]
    for uri in secret_uris:
        with pytest.raises(ValueError, match="forbidden secret"):
            SourceAsset(
                provider="USDA",
                collection="NAIP",
                asset_id="a1",
                canonical_uri=uri,
            )


def test_layered_identities_change_only_at_the_affected_layer():
    acquisition = sample_acquisition()
    changed_satellite = sample_acquisition(satellite_output_sha256="b" * 64)
    changed_terrain = sample_acquisition(terrain_output_sha256="c" * 64)
    changed_preprocessing = sample_acquisition(preprocessing_contract_sha256="d" * 64)
    assert acquisition.sha256() != changed_satellite.sha256()
    assert acquisition.sha256() != changed_terrain.sha256()
    assert acquisition.sha256() != changed_preprocessing.sha256()

    embedding = sample_embedding(acquisition)
    changed_classifier = sample_embedding(
        acquisition, classifier_checkpoint_sha256="e" * 64
    )
    changed_terrain_schema = sample_embedding(acquisition, terrain_feature_version="4")
    assert embedding.sha256() != changed_classifier.sha256()
    assert embedding.sha256() != changed_terrain_schema.sha256()

    report = sample_report(embedding)
    changed_regression = sample_report(embedding, regression_checkpoint_sha256="f" * 64)
    changed_calibration = sample_report(embedding, calibration_artifact_sha256="0" * 64)
    changed_score_schema = sample_report(embedding, score_schema_version="4")
    assert (
        report.embedding_feature_sha256 == changed_regression.embedding_feature_sha256
    )
    assert report.sha256() != changed_regression.sha256()
    assert report.sha256() != changed_calibration.sha256()
    assert report.sha256() != changed_score_schema.sha256()


def test_acquisition_identity_rejects_incomplete_or_non_hash_inputs():
    data = sample_acquisition().to_dict()
    data.pop("terrain_output_sha256")
    with pytest.raises(TypeError):
        AcquisitionTileIdentity.from_dict(data)

    with pytest.raises(ValueError, match="target_grid_sha256"):
        sample_acquisition(target_grid_sha256="not-a-hash")


def test_prediction_identity_distinguishes_calibration_config_and_artifact():
    embedding = sample_embedding()
    report = sample_report(embedding)
    changed_config = sample_report(embedding, calibration_config_sha256="b" * 64)
    changed_artifact = sample_report(embedding, calibration_artifact_sha256="c" * 64)
    assert report.sha256() != changed_config.sha256()
    assert report.sha256() != changed_artifact.sha256()
