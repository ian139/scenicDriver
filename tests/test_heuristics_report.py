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


@pytest.fixture
def learned_cli_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import argparse
    import hashlib
    import json
    import numpy as np
    import torch
    from PIL import Image
    from src.heuristics import labeler
    from src.scenic_scorer.regression import ScenicRegressionModel
    from src.terrain.features import compute_terrain_features, compute_terrain_valid_mask

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SCENIC_S3_BUCKET", raising=False)
    monkeypatch.setenv("SCENIC_S3_ONLY", "0")
    raw = tmp_path / "raw"
    satellite = raw / "images/satellite/z14"
    terrain = raw / "images/terrain/z14"
    satellite.mkdir(parents=True)
    terrain.mkdir(parents=True)
    image = Image.new("RGB", (32, 32), (35, 120, 60))
    image.save(satellite / "100_200.png")
    image.save(terrain / "100_200.png")

    class TinyClassifier(torch.nn.Module):
        def forward(self, batch):
            return torch.tensor([[1.0, 0.0, -1.0]]).expand(len(batch), -1)

        def get_features(self, batch):
            return batch.mean(dim=(2, 3))[:, :2]

    classifier = TinyClassifier()
    classifier_path = tmp_path / "classifier.pt"
    torch.save(classifier.state_dict(), classifier_path)
    terrain_dim = len(
        compute_terrain_features(
            image, image, valid_mask=compute_terrain_valid_mask(image)
        ).features.to_array()
    )
    model = ScenicRegressionModel(vit_dim=2, terrain_dim=terrain_dim, num_classes=3)
    regression_path = tmp_path / "regression.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "vit_dim": 2,
        "terrain_dim": terrain_dim, "num_classes": 3,
    }, regression_path)
    transform = lambda img: torch.from_numpy(
        np.array(img, dtype=np.float32).transpose(2, 0, 1).copy()
    ) / 255.0
    monkeypatch.setattr(labeler, "_load_classifier", lambda **kwargs: (
        classifier, transform, "cpu", ["forest", "water", "urban"]
    ))

    identity = _complete_identity()
    for field in (
        "source_contract_sha256", "preprocessing_contract_sha256",
        "grid_contract_sha256", "calibration_artifact_sha256",
    ):
        artifact = tmp_path / f"{field}.json"
        artifact.write_text(json.dumps({"contract": field}), encoding="utf-8")
        identity[field] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    identity["classifier_checkpoint_sha256"] = hashlib.sha256(classifier_path.read_bytes()).hexdigest()
    identity["regression_checkpoint_sha256"] = hashlib.sha256(regression_path.read_bytes()).hexdigest()
    metadata = tmp_path / "admitted-run.json"
    metadata.write_text(json.dumps({"run_info": identity}), encoding="utf-8")
    args = argparse.Namespace(
        run_name="learned-test", preview=False, all_tiles=False, max_tiles=1,
        write_labels=True, write_raw_labels=False, no_classifier=False,
        device="cpu", seed=42, open=False, satellite_dir=str(satellite),
        terrain_dir=str(terrain), raw_dir=str(raw), s3_only=False,
        scoring="learned", regression_ckpt=str(regression_path),
        classifier_ckpt=str(classifier_path), learned_run_json=str(metadata),
    )
    return args, identity


def test_learned_report_cli_preserves_admitted_identity(learned_cli_args) -> None:
    import json
    from scripts.reports.heuristic_report import run_report
    from src.heuristics.report import REQUIRED_LEARNED_FIELDS

    args, identity = learned_cli_args
    run_dir = run_report(args)
    record = json.loads((run_dir / "run.json").read_text())
    report = json.loads((run_dir / "report/report.json").read_text())
    for field in REQUIRED_LEARNED_FIELDS.values():
        assert record["run_info"][field] == identity[field]
        assert report["run_info"][field] == identity[field]
    assert record["run_info"]["scoring_mode"] == "learned"
    assert report["summary"]["total_tiles"] == 1
    score = report["tiles"][0]["scenic_score"]
    assert 0.0 <= score <= 10.0


def test_learned_report_cli_missing_identity_is_actionable(learned_cli_args) -> None:
    import json
    from scripts.reports.heuristic_report import run_report

    args, identity = learned_cli_args
    del identity["calibration_artifact_sha256"]
    Path(args.learned_run_json).write_text(json.dumps({"run_info": identity}))
    with pytest.raises(ValueError, match="calibration_artifact_sha256.*--learned-run-json"):
        run_report(args)


def test_learned_report_cli_rejects_wrong_checkpoint(learned_cli_args) -> None:
    import json
    from scripts.reports.heuristic_report import run_report

    args, identity = learned_cli_args
    identity["classifier_checkpoint_sha256"] = "0" * 64
    Path(args.learned_run_json).write_text(json.dumps({"run_info": identity}))
    with pytest.raises(ValueError, match="classifier_checkpoint_sha256.*does not match"):
        run_report(args)


def _rebuild_args(labels: Path, **overrides):
    import argparse
    return argparse.Namespace(
        labels_csv=labels, report_dir=overrides.get("report_dir"), raw_dir=str(labels.parent),
        run_name=None, s3_bucket=None, s3_only=False, skip_thumbs=True,
        benchmark=False, source_run_json=overrides.get("source_run_json"),
    )


def test_rebuild_preserves_learned_identity_and_source(tmp_path: Path, monkeypatch) -> None:
    import json
    from scripts.reports import rebuild_report_from_labels as rebuild

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,scenic_score\nimages/satellite/z14/100_200.png,6.5\n")
    identity = {**_complete_identity(), "dataset_sha256": "9" * 64}
    record = {"run_name": "source", "run_info": identity, "config": {"scoring": "learned"}}
    source = tmp_path / "run.json"
    source.write_text(json.dumps(record))
    source_bytes = source.read_bytes()
    monkeypatch.setattr(rebuild, "parse_args", lambda: _rebuild_args(labels))
    rebuild.main()
    assert source.read_bytes() == source_bytes
    report = json.loads((tmp_path / "report/report.json").read_text())
    rebuilt = json.loads((tmp_path / "rebuild.json").read_text())
    assert report["run_info"] == identity
    assert rebuilt["run_info"] == identity
    with pytest.raises(ValueError, match="non-learned/heuristic"):
        union_reports([report, {"tiles": [], "run_info": {"scoring_mode": "heuristic"}}])
    rebuild.main()
    assert source.read_bytes() == source_bytes
    assert json.loads((tmp_path / "rebuild.json").read_text())["run_info"] == identity
    rebuilt_path = tmp_path / "rebuild.json"
    rebuilt_bytes = rebuilt_path.read_bytes()
    monkeypatch.setattr(rebuild, "parse_args", lambda: _rebuild_args(
        labels, source_run_json=rebuilt_path
    ))
    rebuild.main()
    assert rebuilt_path.read_bytes() == rebuilt_bytes
    assert json.loads((tmp_path / "rebuild-record.json").read_text())["run_info"] == identity
    output = tmp_path / "output/report"
    monkeypatch.setattr(rebuild, "parse_args", lambda: _rebuild_args(labels, report_dir=output))
    rebuild.main()
    assert json.loads((output.parent / "run.json").read_text())["run_info"] == identity


@pytest.mark.parametrize("record", [
    None, {}, {"run_info": {"counts": {"paired": 1}}},
    {"run_info": {"scoring_mode": "learned"}},
])
def test_rebuild_rejects_unrecoverable_identity(tmp_path: Path, monkeypatch, record) -> None:
    import json
    from scripts.reports import rebuild_report_from_labels as rebuild

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,scenic_score\nimage.png,6.5\n")
    if record is not None:
        (tmp_path / "run.json").write_text(json.dumps(record))
    monkeypatch.setattr(rebuild, "parse_args", lambda: _rebuild_args(labels))
    with pytest.raises(ValueError, match="--source-run-json"):
        rebuild.main()
    assert not (tmp_path / "report").exists()


def test_fresh_learned_report_uses_checkpoint_metadata(learned_cli_args) -> None:
    import hashlib
    import json
    import torch
    from scripts.reports.heuristic_report import run_report
    from src.heuristics.report import REQUIRED_LEARNED_FIELDS

    args, identity = learned_cli_args
    checkpoint = Path(args.regression_ckpt)
    content = torch.load(checkpoint, weights_only=True)
    content["metadata"] = {
        field: value for field, value in identity.items()
        if field != "regression_checkpoint_sha256"
    }
    torch.save(content, checkpoint)
    args.learned_run_json = None
    Path("admitted-run.json").unlink()
    run_dir = run_report(args)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["run_info"]["regression_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for field in REQUIRED_LEARNED_FIELDS.values():
        if field != "regression_checkpoint_sha256":
            assert record["run_info"][field] == identity[field]
    assert record["run_info"]["scoring_mode"] == "learned"


def test_learned_report_rejects_checkpoint_metadata_override(learned_cli_args) -> None:
    import json
    import torch
    from scripts.reports.heuristic_report import run_report

    args, identity = learned_cli_args
    checkpoint = Path(args.regression_ckpt)
    content = torch.load(checkpoint, weights_only=True)
    content["metadata"] = {"grid_contract_sha256": identity["grid_contract_sha256"]}
    torch.save(content, checkpoint)
    identity["grid_contract_sha256"] = "0" * 64
    Path(args.learned_run_json).write_text(json.dumps({"run_info": identity}))
    with pytest.raises(ValueError, match="grid_contract_sha256.*does not match regression checkpoint metadata"):
        run_report(args)


def test_legacy_checkpoint_without_identity_fails_actionably(learned_cli_args) -> None:
    from scripts.reports.heuristic_report import run_report

    args, _ = learned_cli_args
    args.learned_run_json = None
    with pytest.raises(ValueError, match="source_contract_sha256.*checkpoint metadata.*--learned-run-json"):
        run_report(args)


def test_rebuild_labels_only_accepts_explicit_admitted_source(tmp_path: Path, monkeypatch) -> None:
    import json
    from scripts.reports import rebuild_report_from_labels as rebuild

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,scenic_score\nimage.png,6.5\n")
    source = tmp_path / "admitted.json"
    source.write_text(json.dumps({"run_info": {"scoring_mode": "heuristic"}}))
    source_bytes = source.read_bytes()
    monkeypatch.setattr(rebuild, "parse_args", lambda: _rebuild_args(labels, source_run_json=source))
    rebuild.main()
    assert source.read_bytes() == source_bytes
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert result["run_info"]["scoring_mode"] == "heuristic"
    assert result["tiles"][0]["scenic_score"] == 6.5
