from __future__ import annotations

from pathlib import Path
import pytest

from src.heuristics.report import (
    build_report,
    union_report_tiles,
    union_reports,
)


def test_additive_unequal_report_union() -> None:
    tiles_a = [
        {
            "image_path": "images/satellite/z14/fixture/100_200.png",
            "z": 14,
            "x": 100,
            "y": 200,
            "lat": 40.0,
            "lon": -70.0,
            "scenic_score": 5.0,
            "dataset_origin": "ds_a",
        },
        {
            "image_path": "images/satellite/z14/fixture/101_200.png",
            "z": 14,
            "x": 101,
            "y": 200,
            "lat": 40.1,
            "lon": -70.0,
            "scenic_score": 6.0,
            "dataset_origin": "ds_a",
        },
    ]
    tiles_b = [
        {
            "image_path": "images/satellite/z14/fixture/101_200.png",
            "z": 14,
            "x": 101,
            "y": 200,
            "lat": 40.1,
            "lon": -70.0,
            "scenic_score": 6.0,
            "dataset_origin": "ds_b",
        },
        {
            "image_path": "images/satellite/z14/fixture/102_200.png",
            "z": 14,
            "x": 102,
            "y": 200,
            "lat": 40.2,
            "lon": -70.0,
            "scenic_score": 7.0,
            "dataset_origin": "ds_b",
        },
    ]

    res = union_reports([{"tiles": tiles_a}, {"tiles": tiles_b}])
    assert res["summary"]["total_tiles"] == 3
    tiles = res["tiles"]
    assert [t["x"] for t in tiles] == [100, 101, 102]
    # Check dataset origin preservation
    assert tiles[1]["dataset_origin"] in ("ds_a,ds_b", "ds_b,ds_a", ["ds_a", "ds_b"])


def test_duplicate_disagreement_rejection() -> None:
    tiles_a = [
        {
            "image_path": "images/satellite/z14/fixture/100_200.png",
            "z": 14,
            "x": 100,
            "y": 200,
            "lat": 40.0,
            "lon": -70.0,
            "scenic_score": 5.0,
        }
    ]
    tiles_b_disagree = [
        {
            "image_path": "images/satellite/z14/fixture/100_200.png",
            "z": 14,
            "x": 100,
            "y": 200,
            "lat": 42.0,  # Coordinate disagreement
            "lon": -70.0,
            "scenic_score": 5.0,
        }
    ]

    with pytest.raises(ValueError, match="Coordinate disagreement"):
        union_report_tiles([tiles_a, tiles_b_disagree])


def test_deterministic_report_rerun(tmp_path: Path) -> None:
    tiles = [
        {
            "image_path": "images/satellite/z14/fixture/102_200.png",
            "z": 14,
            "x": 102,
            "y": 200,
            "lat": 40.2,
            "lon": -70.0,
            "scenic_score": 7.0,
        },
        {
            "image_path": "images/satellite/z14/fixture/100_200.png",
            "z": 14,
            "x": 100,
            "y": 200,
            "lat": 40.0,
            "lon": -70.0,
            "scenic_score": 5.0,
        },
    ]

    rep1 = build_report(
        tiles=tiles,
        report_dir=tmp_path / "rep1",
        raw_dir=tmp_path,
        run_info={"run": "test"},
        include_thumbs=False,
    )
    rep2 = build_report(
        tiles=tiles,
        report_dir=tmp_path / "rep2",
        raw_dir=tmp_path,
        run_info={"run": "test"},
        include_thumbs=False,
    )

    assert rep1["tiles"] == rep2["tiles"]
    assert rep1["summary"] == rep2["summary"]
    # Check deterministic ordering by (z, x, y, image_path)
    assert rep1["tiles"][0]["x"] == 100
    assert rep1["tiles"][1]["x"] == 102


def _complete_identity(overrides: dict[str, str] | None = None) -> dict[str, str]:
    base = {
        "source_contract_sha256": "c" * 64,
        "preprocessing_contract_sha256": "b" * 64,
        "grid_contract_sha256": "d" * 64,
        "classifier_checkpoint_sha256": "e" * 64,
        "regression_checkpoint_sha256": "a" * 64,
        "score_schema_version": "1.0",
        "label_schema_version": "1.0",
        "calibration_artifact_sha256": "f" * 64,
        "scoring_mode": "learned",
    }
    if overrides:
        base.update(overrides)
    return base


def test_compatible_report_union_identities() -> None:
    rep_a = {
        "tiles": [
            {
                "image_path": "img1.png",
                "z": 14,
                "x": 100,
                "y": 200,
                "scenic_score": 5.0,
            }
        ],
        "run_info": _complete_identity(),
    }
    rep_b = {
        "tiles": [
            {
                "image_path": "img2.png",
                "z": 14,
                "x": 101,
                "y": 200,
                "scenic_score": 6.0,
            }
        ],
        "run_info": _complete_identity(),
    }

    res = union_reports([rep_a, rep_b])
    assert res["summary"]["total_tiles"] == 2
    assert res["run_info"]["regression_checkpoint_sha256"] == "a" * 64
    assert res["run_info"]["scoring_mode"] == "learned"


def test_incompatible_regression_checkpoint_identity_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": _complete_identity({"regression_checkpoint_sha256": "a" * 64}),
    }
    rep_b = {
        "tiles": [],
        "run_info": _complete_identity({"regression_checkpoint_sha256": "d" * 64}),
    }

    with pytest.raises(ValueError, match="'regression_checkpoint'"):
        union_reports([rep_a, rep_b])


def test_incompatible_classifier_checkpoint_identity_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": _complete_identity(
            {"classifier_checkpoint_sha256": "cls1" + "0" * 60}
        ),
    }
    rep_b = {
        "tiles": [],
        "run_info": _complete_identity(
            {"classifier_checkpoint_sha256": "cls2" + "0" * 60}
        ),
    }

    with pytest.raises(ValueError, match="'classifier_checkpoint'"):
        union_reports([rep_a, rep_b])


def test_incompatible_grid_identity_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": _complete_identity({"grid_contract_sha256": "1" * 64}),
    }
    rep_b = {
        "tiles": [],
        "run_info": _complete_identity({"grid_contract_sha256": "2" * 64}),
    }

    with pytest.raises(ValueError, match="'grid'"):
        union_reports([rep_a, rep_b])


def test_incompatible_score_schema_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": _complete_identity({"score_schema_version": "1.0"}),
    }
    rep_b = {
        "tiles": [],
        "run_info": _complete_identity({"score_schema_version": "2.0"}),
    }

    with pytest.raises(ValueError, match="'score_schema'"):
        union_reports([rep_a, rep_b])


def test_incompatible_label_schema_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": _complete_identity({"label_schema_version": "1.0"}),
    }
    rep_b = {
        "tiles": [],
        "run_info": _complete_identity({"label_schema_version": "2.0"}),
    }

    with pytest.raises(ValueError, match="'label_schema'"):
        union_reports([rep_a, rep_b])


def test_incompatible_scoring_mode_rejection() -> None:
    rep_a = {"tiles": [], "run_info": _complete_identity()}
    rep_b = {"tiles": [], "run_info": {"scoring_mode": "heuristic"}}

    with pytest.raises(ValueError, match="non-learned/heuristic"):
        union_reports([rep_a, rep_b])


def test_incompatible_preprocessing_contract_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": _complete_identity({"preprocessing_contract_sha256": "b" * 64}),
    }
    rep_b = {
        "tiles": [],
        "run_info": _complete_identity({"preprocessing_contract_sha256": "e" * 64}),
    }

    with pytest.raises(ValueError, match="'preprocessing'"):
        union_reports([rep_a, rep_b])


def test_incompatible_source_contract_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": _complete_identity({"source_contract_sha256": "c" * 64}),
    }
    rep_b = {
        "tiles": [],
        "run_info": _complete_identity({"source_contract_sha256": "f" * 64}),
    }

    with pytest.raises(ValueError, match="'source_contract'"):
        union_reports([rep_a, rep_b])


def test_learned_report_missing_required_dimension_fails_closed() -> None:
    rep = {
        "tiles": [],
        "run_info": {"scoring_mode": "learned", "source_contract_sha256": "c" * 64},
    }
    with pytest.raises(ValueError, match="missing required identity dimension"):
        union_reports([rep])


def test_learned_report_rejects_legacy_identity_aliases() -> None:
    identity = _complete_identity()
    identity["grid_identity"] = identity.pop("grid_contract_sha256")
    identity["calibration_identity"] = identity.pop("calibration_artifact_sha256")
    with pytest.raises(ValueError, match="exact field 'grid_contract_sha256'"):
        union_reports([{"tiles": [], "run_info": identity}])


def test_conflicting_duplicate_identity_field_fails_closed() -> None:
    identity = _complete_identity()
    report = {
        "tiles": [],
        "run_info": {
            **identity,
            "identity": {
                **identity,
                "source_contract_sha256": "9" * 64,
            },
        },
    }
    with pytest.raises(ValueError, match="Conflicting identity field"):
        union_reports([report])


def test_onesided_missing_identity_rejection() -> None:
    rep_a = {
        "tiles": [],
        "run_info": {**_complete_identity(), "extra_contract_sha256": "x" * 64},
    }
    rep_b = {"tiles": [], "run_info": _complete_identity()}

    with pytest.raises(ValueError, match="One-sided missing identity"):
        union_reports([rep_a, rep_b])


def test_file_based_report_union_incompatibility(tmp_path: Path) -> None:
    import json

    dir_a = tmp_path / "rep_a"
    dir_b = tmp_path / "rep_b"
    dir_a.mkdir()
    dir_b.mkdir()

    (dir_a / "report.json").write_text(
        json.dumps(
            {
                "tiles": [],
                "run_info": _complete_identity(
                    {"regression_checkpoint_sha256": "a" * 64}
                ),
            }
        ),
        encoding="utf-8",
    )
    (dir_b / "report.json").write_text(
        json.dumps(
            {
                "tiles": [],
                "run_info": _complete_identity(
                    {"regression_checkpoint_sha256": "z" * 64}
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'regression_checkpoint'"):
        union_reports([dir_a, dir_b])
