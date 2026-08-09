from __future__ import annotations

import csv
import json
from pathlib import Path
import typing

import numpy as np
import pytest
import torch

from src.scenic_scorer.active_evaluation import (
    evaluate_active_baseline,
    evaluate_candidate_validation,
    evaluate_stage_two,
    file_sha256,
    promote_from_decision,
    rollback_registry,
)
from src.scenic_scorer.regression import ScenicRegressionModel


def _sha(path: Path) -> str:
    return file_sha256(path)


def create_real_checkpoint(
    path: Path,
    vit_dim: int = 8,
    terrain_dim: int = 2,
    num_classes: int = 3,
    bias_shift: float = 0.0,
) -> Path:
    model = ScenicRegressionModel(
        vit_dim=vit_dim, terrain_dim=terrain_dim, num_classes=num_classes
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # Route one deterministic feature through the three linear layers.
        model.network[0].weight[0, 0] = 1.0
        model.network[3].weight[0, 0] = 1.0
        model.network[6].weight[0, 0] = 0.1
        model.network[6].bias.fill_(bias_shift)
    payload = {
        "checkpoint_schema_version": 1,
        "checkpoint_state": "completed",
        "model_state_dict": model.state_dict(),
        "vit_dim": vit_dim,
        "terrain_dim": terrain_dim,
        "num_classes": num_classes,
    }
    torch.save(payload, path)
    return path


def create_legacy_checkpoint(
    path: Path,
    vit_dim: int = 8,
    terrain_dim: int = 2,
    num_classes: int = 3,
    bias_shift: float = 0.0,
) -> Path:
    model = ScenicRegressionModel(
        vit_dim=vit_dim, terrain_dim=terrain_dim, num_classes=num_classes
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.network[0].weight[0, 0] = 1.0
        model.network[3].weight[0, 0] = 1.0
        model.network[6].weight[0, 0] = 0.1
        model.network[6].bias.fill_(bias_shift)
    payload = {
        "model_state_dict": model.state_dict(),
        "vit_dim": vit_dim,
        "terrain_dim": terrain_dim,
        "num_classes": num_classes,
    }
    torch.save(payload, path)
    return path


def create_npz_dataset(
    path: Path,
    image_paths: list[str],
    vit_dim: int = 8,
    terrain_dim: int = 2,
    num_classes: int = 3,
    splits: list[str] | None = None,
) -> Path:
    n = len(image_paths)
    vit = np.linspace(-1.0, 1.0, n * vit_dim, dtype=np.float32).reshape(n, vit_dim)
    terr = np.linspace(0.1, 0.9, n * terrain_dim, dtype=np.float32).reshape(
        n, terrain_dim
    )
    cls_logits = np.linspace(-0.5, 0.5, n * num_classes, dtype=np.float32).reshape(
        n, num_classes
    )
    scenic_scores = np.linspace(2.0, 8.0, n, dtype=np.float32)
    sample_weights = np.ones(n, dtype=np.float32)
    label_sources = np.array(["human"] * n)
    if splits is None:
        splits = ["test"] * n
    np.savez_compressed(
        path,
        vit_embeddings=vit,
        terrain_features=terr,
        class_logits=cls_logits,
        scenic_scores=scenic_scores,
        sample_weights=sample_weights,
        image_paths=np.array(image_paths),
        label_sources=label_sources,
        splits=np.array(splits),
    )
    return path


def create_benchmark_csv(
    path: Path,
    rows: list[dict[str, typing.Any]],
) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def create_route_qa_json(
    path: Path,
    all_invariants_pass: bool = True,
    stability_confirmed: bool = True,
    complexity_accepted: bool = True,
) -> Path:
    data = {
        "routes": [{"route_id": "r1"}],
        "all_invariants_pass": all_invariants_pass,
        "stability_confirmed": stability_confirmed,
        "complexity_accepted": complexity_accepted,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def get_default_thresholds() -> dict[str, typing.Any]:
    return {
        "min_expanded_corr": 0.0,
        "max_expanded_mse": 100.0,
        "min_expanded_mse_improvement": 0.0,
        "max_control_mse_regression": 100.0,
        "min_control_corr": 0.0,
        "max_worst_slice_mse": 100.0,
        "max_calibration_error": 100.0,
        "min_supported_slice_samples": 2,
        "min_spread_ratio": 0.0,
        "max_mean_drift": 100.0,
        "max_saturation_ratio": 1.0,
        "min_unique_ratio": 0.0,
    }


def test_evaluate_stage_two_filters_only_split_test_rows(tmp_path: Path) -> None:
    dataset_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        splits=["test", "test", "train", "val"],
    )
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img5.jpg", "img6.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt", bias_shift=0.0)

    exp_csv = create_benchmark_csv(
        tmp_path / "exp.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human_mean": "6.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img3.jpg",
                "split": "train",
                "scenic_human_mean": "1.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img4.jpg",
                "split": "val",
                "scenic_human_mean": "9.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {
                "image_path": "img5.jpg",
                "split": "test",
                "scenic_human": "4.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img6.jpg",
                "split": "test",
                "scenic_human": "6.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"

    decision = evaluate_stage_two(
        dataset_path=dataset_path,
        control_dataset_path=control_dataset_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=base_ckpt,
        expanded_benchmark_csv=exp_csv,
        control_benchmark_csv=ctrl_csv,
        route_qa_json=route_qa,
        thresholds=get_default_thresholds(),
        output_path=out_json,
    )

    assert decision["expanded_human_benchmark"]["candidate_metrics"]["samples"] == 2
    assert decision["control_benchmark"]["candidate_metrics"]["samples"] == 2


def test_evaluate_stage_two_rejects_missing_split_or_no_test_rows(
    tmp_path: Path,
) -> None:
    images = ["img1.jpg", "img2.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img3.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"

    # Missing split column
    exp_no_split = create_benchmark_csv(
        tmp_path / "no_split.csv",
        [
            {
                "image_path": "img1.jpg",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {
                "image_path": "img3.jpg",
                "split": "test",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    with pytest.raises(ValueError, match="requires an explicit split column"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_no_split,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )

    # No test rows (only train)
    exp_train_only = create_benchmark_csv(
        tmp_path / "train_only.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "train",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    with pytest.raises(ValueError, match="contains no split=test rows"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_train_only,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )
    split_bound_dataset = create_npz_dataset(
        tmp_path / "split_bound.npz", images, splits=["train", "test"]
    )
    exp_relabels_train = create_benchmark_csv(
        tmp_path / "relabels_train.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            }
        ],
    )
    with pytest.raises(ValueError, match="relabels non-test prepared identity"):
        evaluate_stage_two(
            dataset_path=split_bound_dataset,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_relabels_train,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_rejects_weak_only_targets(tmp_path: Path) -> None:
    images = ["img1.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img2.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"

    exp_weak_only = create_benchmark_csv(
        tmp_path / "weak_only.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_score": "4.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    with pytest.raises(
        ValueError,
        match="must be scenic_human_mean or scenic_human, never weak/mixed scenic_score",
    ):
        evaluate_stage_two(
            dataset_path=dataset_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_weak_only,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )
    out_of_range = create_benchmark_csv(
        tmp_path / "out_of_range.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human_mean": "11",
                "region": "r1",
                "slice": "s1",
            }
        ],
    )
    with pytest.raises(ValueError, match=r"finite and in \[0, 10\]"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=out_of_range,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_denominator_and_hash_reporting(tmp_path: Path) -> None:
    images = ["img1.jpg", "img2.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img3.jpg", "img4.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt", bias_shift=0.0)

    exp_csv = create_benchmark_csv(
        tmp_path / "exp.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human_mean": "6.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {
                "image_path": "img3.jpg",
                "split": "test",
                "scenic_human": "4.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img4.jpg",
                "split": "test",
                "scenic_human": "6.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"

    decision = evaluate_stage_two(
        dataset_path=dataset_path,
        control_dataset_path=control_dataset_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=base_ckpt,
        expanded_benchmark_csv=exp_csv,
        control_benchmark_csv=ctrl_csv,
        route_qa_json=route_qa,
        thresholds=get_default_thresholds(),
        output_path=out_json,
    )

    assert decision["candidate"]["sha256"] == _sha(cand_ckpt)
    assert decision["baseline"]["sha256"] == _sha(base_ckpt)
    assert decision["expanded_human_benchmark"]["candidate_metrics"]["samples"] == 2

    # Path absent from prepared dataset
    exp_absent_path = create_benchmark_csv(
        tmp_path / "exp_absent.csv",
        [
            {
                "image_path": "img_missing.jpg",
                "split": "test",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    with pytest.raises(ValueError, match="absent from the prepared dataset"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_absent_path,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_passing_decision_schema_and_gates(tmp_path: Path) -> None:
    images = ["img3.jpg", "img4.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img1.jpg", "img2.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.0)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt", bias_shift=0.0)

    exp_csv = create_benchmark_csv(
        tmp_path / "exp.csv",
        [
            {
                "image_path": "img3.jpg",
                "split": "test",
                "scenic_human_mean": "4.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img4.jpg",
                "split": "test",
                "scenic_human_mean": "6.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human": "4.0",
                "region": "r1",
                "slice": "s1",
            },
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human": "6.0",
                "region": "r1",
                "slice": "s1",
            },
        ],
    )
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"

    decision = evaluate_stage_two(
        dataset_path=dataset_path,
        control_dataset_path=control_dataset_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=base_ckpt,
        expanded_benchmark_csv=exp_csv,
        control_benchmark_csv=ctrl_csv,
        route_qa_json=route_qa,
        thresholds=get_default_thresholds(),
        output_path=out_json,
    )

    expected_top_keys = {
        "timestamp",
        "all_gates_pass",
        "candidate",
        "baseline",
        "expanded_dataset",
        "control_dataset",
        "expanded_human_benchmark",
        "control_benchmark",
        "calibration_and_distribution",
        "route_qa_evidence",
        "thresholds_evaluated",
        "gates",
    }
    assert set(decision.keys()) == expected_top_keys
    assert decision["expanded_dataset"]["sha256"] == _sha(dataset_path)
    assert decision["control_dataset"]["sha256"] == _sha(control_dataset_path)
    assert decision["all_gates_pass"], decision["gates"]

    # Verify structured distribution, regional, route, stability, and complexity evidence.
    gates = decision["gates"]
    assert gates["integrity_pass"] is True
    assert gates["route_evidence_pass"] is True
    assert gates["distribution_pass"] is True
    assert "r1" in decision["expanded_human_benchmark"]["region_metrics"]
    route_evidence = decision["route_qa_evidence"]
    assert route_evidence["stability_confirmed"] is True
    assert route_evidence["complexity_accepted"] is True

    # Test gate failure when route invariants fail
    route_qa_failed = create_route_qa_json(
        tmp_path / "route_failed.json", all_invariants_pass=False
    )
    decision_failed = evaluate_stage_two(
        dataset_path=dataset_path,
        control_dataset_path=control_dataset_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=base_ckpt,
        expanded_benchmark_csv=exp_csv,
        control_benchmark_csv=ctrl_csv,
        route_qa_json=route_qa_failed,
        thresholds=get_default_thresholds(),
        output_path=out_json,
    )
    assert decision_failed["gates"]["route_evidence_pass"] is False
    assert decision_failed["all_gates_pass"] is False


def test_promote_from_decision_hash_guarded_atomic(tmp_path: Path) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")

    registry = tmp_path / "registry.json"
    initial_registry_data = {
        "active": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
        "history": [],
    }
    registry.write_text(json.dumps(initial_registry_data, indent=2), encoding="utf-8")
    reg_hash = _sha(registry)

    decision_file = tmp_path / "decision.json"
    decision_payload = {
        "all_gates_pass": True,
        "candidate": {"checkpoint": str(cand_ckpt), "sha256": _sha(cand_ckpt)},
        "baseline": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
        "expanded_human_benchmark": {
            "candidate_metrics": {
                "mse": 0.1,
                "mae": 0.05,
                "rmse": 0.1,
                "pearson_corr": 0.95,
                "samples": 20,
            }
        },
    }
    decision_file.write_text(json.dumps(decision_payload), encoding="utf-8")

    # Hash mismatch on registry expected sha256
    with pytest.raises(ValueError, match="Registry SHA256 mismatch"):
        promote_from_decision(
            decision_path=decision_file,
            candidate_checkpoint=cand_ckpt,
            registry_path=registry,
            expected_registry_sha256="wrong_hash" + "0" * 54,
            run_name="run_1",
        )

    # Hash mismatch on decision candidate sha256 vs actual candidate file
    bad_decision_file = tmp_path / "bad_decision.json"
    bad_decision_payload = dict(decision_payload)
    bad_decision_payload["candidate"] = {
        "checkpoint": str(cand_ckpt),
        "sha256": "0" * 64,
    }
    bad_decision_file.write_text(json.dumps(bad_decision_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Candidate SHA256 mismatch"):
        promote_from_decision(
            decision_path=bad_decision_file,
            candidate_checkpoint=cand_ckpt,
            registry_path=registry,
            expected_registry_sha256=reg_hash,
            run_name="run_1",
        )

    # Successful promotion
    result = promote_from_decision(
        decision_path=decision_file,
        candidate_checkpoint=cand_ckpt,
        registry_path=registry,
        expected_registry_sha256=reg_hash,
        run_name="run_1",
    )
    assert result["status"] == "promoted"

    updated_registry = json.loads(registry.read_text(encoding="utf-8"))
    assert updated_registry["active"]["sha256"] == _sha(cand_ckpt)
    assert len(updated_registry["history"]) == 1
    assert updated_registry["history"][0]["event"] == "promote"


def test_promotion_preserves_immutable_history(tmp_path: Path) -> None:
    cand_ckpt1 = create_real_checkpoint(tmp_path / "cand1.pt", bias_shift=0.1)
    cand_ckpt2 = create_real_checkpoint(tmp_path / "cand2.pt", bias_shift=0.2)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")

    registry = tmp_path / "registry.json"
    initial_registry_data = {
        "active": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
        "history": [],
    }
    registry.write_text(json.dumps(initial_registry_data, indent=2), encoding="utf-8")

    dec1 = tmp_path / "dec1.json"
    dec1.write_text(
        json.dumps(
            {
                "all_gates_pass": True,
                "candidate": {
                    "checkpoint": str(cand_ckpt1),
                    "sha256": _sha(cand_ckpt1),
                },
                "baseline": {
                    "checkpoint": str(base_ckpt),
                    "sha256": _sha(base_ckpt),
                },
                "expanded_human_benchmark": {
                    "candidate_metrics": {
                        "mse": 0.1,
                        "mae": 0.05,
                        "rmse": 0.1,
                        "pearson_corr": 0.95,
                        "samples": 20,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    res1 = promote_from_decision(
        decision_path=dec1,
        candidate_checkpoint=cand_ckpt1,
        registry_path=registry,
        expected_registry_sha256=_sha(registry),
        run_name="run_1",
    )

    dec2 = tmp_path / "dec2.json"
    dec2.write_text(
        json.dumps(
            {
                "all_gates_pass": True,
                "candidate": {
                    "checkpoint": str(cand_ckpt2),
                    "sha256": _sha(cand_ckpt2),
                },
                "baseline": {
                    "checkpoint": str(cand_ckpt1),
                    "sha256": _sha(cand_ckpt1),
                },
                "expanded_human_benchmark": {
                    "candidate_metrics": {
                        "mse": 0.1,
                        "mae": 0.05,
                        "rmse": 0.1,
                        "pearson_corr": 0.95,
                        "samples": 20,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    promote_from_decision(
        decision_path=dec2,
        candidate_checkpoint=cand_ckpt2,
        registry_path=registry,
        expected_registry_sha256=res1["new_registry_sha256"],
        run_name="run_2",
    )

    reg_data = json.loads(registry.read_text(encoding="utf-8"))
    assert len(reg_data["history"]) == 2
    assert reg_data["history"][0]["record"]["sha256"] == _sha(cand_ckpt1)
    assert reg_data["history"][1]["record"]["sha256"] == _sha(cand_ckpt2)


def test_rollback_to_exact_historical_checkpoint(tmp_path: Path) -> None:
    old_ckpt = create_real_checkpoint(tmp_path / "old.pt", bias_shift=0.0)
    new_ckpt = create_real_checkpoint(tmp_path / "new.pt", bias_shift=0.3)

    old_sha = _sha(old_ckpt)
    new_sha = _sha(new_ckpt)

    registry = tmp_path / "registry.json"
    registry_data = {
        "active": {
            "checkpoint": str(new_ckpt),
            "sha256": new_sha,
            "run_name": "run_2",
        },
        "history": [
            {
                "event": "promote",
                "record": {
                    "checkpoint": str(old_ckpt),
                    "sha256": old_sha,
                    "run_name": "run_1",
                },
            }
        ],
    }
    registry.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")

    # Rollback with incorrect registry hash fails
    with pytest.raises(ValueError, match="Registry SHA256 mismatch"):
        rollback_registry(
            registry,
            target_history_index=0,
            expected_registry_sha256="wrong_hash" + "0" * 54,
        )

    # Rollback with invalid target index fails
    with pytest.raises(ValueError, match="Invalid target history index"):
        rollback_registry(
            registry, target_history_index=99, expected_registry_sha256=_sha(registry)
        )

    # Successful rollback
    result = rollback_registry(
        registry_path=registry,
        target_history_index=0,
        expected_registry_sha256=_sha(registry),
    )
    assert result["status"] == "rolled_back"

    reg_after = json.loads(registry.read_text(encoding="utf-8"))
    assert reg_after["active"]["sha256"] == old_sha
    assert len(reg_after["history"]) == 2
    assert reg_after["history"][1]["event"] == "rollback"

    # Rollback fails if historical checkpoint file is missing or corrupted (hash mismatch)
    old_ckpt.write_bytes(b"corrupted contents")
    reg_corrupt_data = {
        "active": {
            "checkpoint": str(new_ckpt),
            "sha256": new_sha,
        },
        "history": [
            {
                "event": "promote",
                "record": {
                    "checkpoint": str(old_ckpt),
                    "sha256": old_sha,
                },
            }
        ],
    }
    registry.write_text(json.dumps(reg_corrupt_data, indent=2), encoding="utf-8")
    with pytest.raises(
        ValueError, match="Historical checkpoint identity validation failed"
    ):
        rollback_registry(
            registry, target_history_index=0, expected_registry_sha256=_sha(registry)
        )


def test_rejected_decision_preserves_registry_bytes(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"active": {"checkpoint": "old.pt"}, "history": []}),
        encoding="utf-8",
    )
    before = registry.read_bytes()
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"all_gates_pass": False}), encoding="utf-8")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        promote_from_decision(decision, candidate, registry, _sha(registry), "run")
    assert registry.read_bytes() == before


def test_promote_from_decision_rejects_string_false_and_non_dict(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "active": {"checkpoint": str(cand_ckpt), "sha256": _sha(cand_ckpt)},
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    reg_hash = _sha(registry)
    dec_file = tmp_path / "dec_string_false.json"
    dec_file.write_text(
        json.dumps(
            {
                "all_gates_pass": "false",
                "candidate": {"checkpoint": str(cand_ckpt), "sha256": _sha(cand_ckpt)},
                "expanded_human_benchmark": {
                    "candidate_metrics": {
                        "mse": 0.1,
                        "mae": 0.05,
                        "rmse": 0.1,
                        "pearson_corr": 0.95,
                        "samples": 20,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="all_gates_pass != True"):
        promote_from_decision(
            decision_path=dec_file,
            candidate_checkpoint=cand_ckpt,
            registry_path=registry,
            expected_registry_sha256=reg_hash,
            run_name="run_1",
        )

    dec_file_true_str = tmp_path / "dec_string_true.json"
    dec_file_true_str.write_text(
        json.dumps(
            {
                "all_gates_pass": "true",
                "candidate": {"checkpoint": str(cand_ckpt), "sha256": _sha(cand_ckpt)},
                "expanded_human_benchmark": {
                    "candidate_metrics": {
                        "mse": 0.1,
                        "mae": 0.05,
                        "rmse": 0.1,
                        "pearson_corr": 0.95,
                        "samples": 20,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="all_gates_pass != True"):
        promote_from_decision(
            decision_path=dec_file_true_str,
            candidate_checkpoint=cand_ckpt,
            registry_path=registry,
            expected_registry_sha256=reg_hash,
            run_name="run_1",
        )

    # Non-dict decision
    dec_non_dict = tmp_path / "dec_non_dict.json"
    dec_non_dict.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a dictionary"):
        promote_from_decision(
            decision_path=dec_non_dict,
            candidate_checkpoint=cand_ckpt,
            registry_path=registry,
            expected_registry_sha256=reg_hash,
            run_name="run_1",
        )


def test_evaluate_stage_two_rejects_paused_or_legacy_candidate(
    tmp_path: Path,
) -> None:
    images = ["img1.jpg", "img2.jpg"]
    ds_path = create_npz_dataset(tmp_path / "ds.npz", images)
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img3.jpg"]
    )
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")
    exp_csv = create_benchmark_csv(
        tmp_path / "exp.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human": 5.0,
                "region": "r1",
            },
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human": 6.0,
                "region": "r1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {"image_path": "img1.jpg", "split": "val", "scenic_human": 5.0},
            {"image_path": "img2.jpg", "split": "val", "scenic_human": 6.0},
        ],
    )
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "dec.json"

    # Paused state candidate
    paused_cand = tmp_path / "paused_cand.pt"
    model = ScenicRegressionModel(vit_dim=8, terrain_dim=2, num_classes=3)
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "checkpoint_state": "paused",
            "model_state_dict": model.state_dict(),
            "vit_dim": 8,
            "terrain_dim": 2,
            "num_classes": 3,
        },
        paused_cand,
    )

    with pytest.raises(ValueError, match="checkpoint_state must be 'completed'"):
        evaluate_stage_two(
            dataset_path=ds_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=paused_cand,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )

    # Legacy candidate (missing schema version / state)
    legacy_cand = create_legacy_checkpoint(tmp_path / "legacy_cand.pt")
    with pytest.raises(ValueError, match="missing required keys"):
        evaluate_stage_two(
            dataset_path=ds_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=legacy_cand,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_accepts_legacy_baseline(tmp_path: Path) -> None:
    images = ["img1.jpg", "img2.jpg"]
    ds_path = create_npz_dataset(tmp_path / "ds.npz", images)
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img3.jpg", "img4.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    legacy_base_ckpt = create_legacy_checkpoint(tmp_path / "legacy_base.pt")

    exp_csv = create_benchmark_csv(
        tmp_path / "exp.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human": 5.0,
                "region": "r1",
            },
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human": 6.0,
                "region": "r1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {"image_path": "img3.jpg", "split": "test", "scenic_human": 5.0},
            {"image_path": "img4.jpg", "split": "test", "scenic_human": 6.0},
        ],
    )
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "dec.json"

    decision = evaluate_stage_two(
        dataset_path=ds_path,
        control_dataset_path=control_dataset_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=legacy_base_ckpt,
        expanded_benchmark_csv=exp_csv,
        control_benchmark_csv=ctrl_csv,
        route_qa_json=route_qa,
        thresholds=get_default_thresholds(),
        output_path=out_json,
    )
    assert decision["gates"]["integrity_pass"] is True


def test_evaluate_stage_two_rejects_dataset_overlap(tmp_path: Path) -> None:
    images = ["img1.jpg", "img2.jpg"]
    ds_path = create_npz_dataset(tmp_path / "ds.npz", images)
    # Control dataset must be disjoint from the expanded prepared dataset: img1.jpg
    # appears in both, so evaluation must fail closed before any benchmark work.
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img1.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")

    exp_csv = create_benchmark_csv(
        tmp_path / "exp.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human": 5.0,
                "region": "r1",
            },
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human": 6.0,
                "region": "r1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human": 5.0,
                "region": "r1",
            },
        ],
    )
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "dec.json"

    with pytest.raises(
        ValueError,
        match="Overlap detected between expanded and control prepared datasets",
    ):
        evaluate_stage_two(
            dataset_path=ds_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_rejects_malformed_routes(tmp_path: Path) -> None:
    images = ["img1.jpg", "img2.jpg"]
    ds_path = create_npz_dataset(tmp_path / "ds.npz", images)
    control_dataset_path = create_npz_dataset(
        tmp_path / "control_dataset.npz", ["img3.jpg", "img4.jpg"]
    )
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")

    exp_csv = create_benchmark_csv(
        tmp_path / "exp.csv",
        [
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human": 5.0,
                "region": "r1",
            },
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human": 6.0,
                "region": "r1",
            },
        ],
    )
    ctrl_csv = create_benchmark_csv(
        tmp_path / "ctrl.csv",
        [
            {"image_path": "img3.jpg", "split": "test", "scenic_human": 5.0},
            {"image_path": "img4.jpg", "split": "test", "scenic_human": 6.0},
        ],
    )

    malformed_cases = [
        {
            "routes": {"scenic": {}},
            "all_invariants_pass": True,
            "stability_confirmed": True,
            "complexity_accepted": True,
        },  # dict instead of list
        {
            "routes": [],
            "all_invariants_pass": True,
            "stability_confirmed": True,
            "complexity_accepted": True,
        },  # empty list
        {
            "routes": ["not_a_dict"],
            "all_invariants_pass": True,
            "stability_confirmed": True,
            "complexity_accepted": True,
        },  # list of non-dicts
        {
            "routes": [{}],
            "all_invariants_pass": True,
            "stability_confirmed": True,
            "complexity_accepted": True,
        },  # list of empty dicts
    ]

    for i, data in enumerate(malformed_cases):
        rq_path = tmp_path / f"rq_{i}.json"
        rq_path.write_text(json.dumps(data), encoding="utf-8")
        out_json = tmp_path / f"dec_{i}.json"
        decision = evaluate_stage_two(
            dataset_path=ds_path,
            control_dataset_path=control_dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=rq_path,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )
        assert decision["gates"]["route_evidence_pass"] is False


def test_promotion_content_addressing_isolates_from_candidate_mutation(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")

    registry = tmp_path / "registry.json"
    initial_registry_data = {
        "active": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
        "history": [],
    }
    registry.write_text(json.dumps(initial_registry_data, indent=2), encoding="utf-8")
    reg_hash = _sha(registry)
    orig_cand_sha = _sha(cand_ckpt)

    decision_file = tmp_path / "decision.json"
    decision_payload = {
        "all_gates_pass": True,
        "candidate": {"checkpoint": str(cand_ckpt), "sha256": orig_cand_sha},
        "baseline": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
        "expanded_human_benchmark": {
            "candidate_metrics": {
                "mse": 0.1,
                "mae": 0.05,
                "rmse": 0.1,
                "pearson_corr": 0.95,
                "samples": 20,
            }
        },
    }
    decision_file.write_text(json.dumps(decision_payload), encoding="utf-8")

    result = promote_from_decision(
        decision_path=decision_file,
        candidate_checkpoint=cand_ckpt,
        registry_path=registry,
        expected_registry_sha256=reg_hash,
        run_name="run_1",
    )
    published_active = Path(result["active"]["checkpoint"])
    assert published_active.exists()
    assert published_active != cand_ckpt.resolve()
    assert _sha(published_active) == orig_cand_sha

    # Overwrite the original candidate file in training workspace
    cand_ckpt.write_bytes(b"overwritten candidate bytes")

    # Active checkpoint in registry and published file remain unchanged
    reg_data_after = json.loads(registry.read_text(encoding="utf-8"))
    active_after_path = Path(reg_data_after["active"]["checkpoint"])
    assert active_after_path.exists()
    assert _sha(active_after_path) == orig_cand_sha
    assert active_after_path.read_bytes() != cand_ckpt.read_bytes()
    assert "updated_at" in reg_data_after["active"]
    assert "promoted_at" in reg_data_after["active"]
    assert reg_data_after["active"]["metrics"]["corr"] == 0.95


def test_promote_from_decision_accepts_missing_sha_active_baseline(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")

    # Baseline active record lacks sha256 (canonical production baseline registry format)
    registry = tmp_path / "registry.json"
    initial_registry_data = {
        "active": {"checkpoint": str(base_ckpt)},
        "history": [],
    }
    registry.write_text(json.dumps(initial_registry_data, indent=2), encoding="utf-8")
    reg_hash = _sha(registry)

    decision_file = tmp_path / "decision.json"
    decision_payload = {
        "all_gates_pass": True,
        "candidate": {"checkpoint": str(cand_ckpt), "sha256": _sha(cand_ckpt)},
        "baseline": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
        "expanded_human_benchmark": {
            "candidate_metrics": {
                "mse": 0.1,
                "mae": 0.05,
                "rmse": 0.1,
                "pearson_corr": 0.95,
                "samples": 20,
            }
        },
    }
    decision_file.write_text(json.dumps(decision_payload), encoding="utf-8")

    result = promote_from_decision(
        decision_path=decision_file,
        candidate_checkpoint=cand_ckpt,
        registry_path=registry,
        expected_registry_sha256=reg_hash,
        run_name="run_1",
    )
    assert result["status"] == "promoted"
    active = result["active"]
    assert "updated_at" in active
    assert "promoted_at" in active
    assert active["updated_at"] == active["promoted_at"]
    assert active["sha256"] == _sha(cand_ckpt)
    assert active["metrics"]["corr"] == 0.95
    assert active["metrics"]["mae"] == 0.05
    assert active["metrics"]["rmse"] == 0.1
    assert active["metrics"]["samples"] == 20


def test_promote_from_decision_rejects_absent_or_malformed_metrics(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "active": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    reg_hash = _sha(registry)

    bad_metrics_cases = [
        {},  # missing expanded_human_benchmark
        {"expanded_human_benchmark": {}},  # missing candidate_metrics
        {
            "expanded_human_benchmark": {
                "candidate_metrics": {"mae": 0.1, "rmse": 0.1, "samples": 10}
            }
        },  # missing corr
        {
            "expanded_human_benchmark": {
                "candidate_metrics": {"pearson_corr": 0.9, "rmse": 0.1, "samples": 10}
            }
        },  # missing mae
        {
            "expanded_human_benchmark": {
                "candidate_metrics": {"pearson_corr": 0.9, "mae": 0.1, "samples": 10}
            }
        },  # missing rmse
        {
            "expanded_human_benchmark": {
                "candidate_metrics": {"pearson_corr": 0.9, "mae": 0.1, "rmse": 0.1}
            }
        },  # missing samples
        {
            "expanded_human_benchmark": {
                "candidate_metrics": {
                    "pearson_corr": True,
                    "mae": 0.1,
                    "rmse": 0.1,
                    "samples": 10,
                }
            }
        },  # bool corr
        {
            "expanded_human_benchmark": {
                "candidate_metrics": {
                    "pearson_corr": 0.9,
                    "mae": 0.1,
                    "rmse": 0.1,
                    "samples": True,
                }
            }
        },  # bool samples
        {
            "expanded_human_benchmark": {
                "candidate_metrics": {
                    "pearson_corr": 0.9,
                    "mae": 0.1,
                    "rmse": 0.1,
                    "samples": 0,
                }
            }
        },  # non-positive samples
    ]

    for i, extra in enumerate(bad_metrics_cases):
        dec_file = tmp_path / f"dec_bad_{i}.json"
        payload = {
            "all_gates_pass": True,
            "candidate": {"checkpoint": str(cand_ckpt), "sha256": _sha(cand_ckpt)},
            "baseline": {"checkpoint": str(base_ckpt), "sha256": _sha(base_ckpt)},
        }
        payload.update(extra)
        dec_file.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="candidate metric"):
            promote_from_decision(
                decision_path=dec_file,
                candidate_checkpoint=cand_ckpt,
                registry_path=registry,
                expected_registry_sha256=reg_hash,
                run_name="run_1",
            )


def test_evaluate_active_baseline_success(tmp_path: Path) -> None:
    ckpt = create_legacy_checkpoint(tmp_path / "baseline.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["img1.jpg", "img2.jpg"],
        splits=["test", "test"],
    )
    control_npz = create_npz_dataset(
        tmp_path / "control_dataset.npz",
        image_paths=["img3.jpg", "img4.jpg"],
        splits=["test", "test"],
    )
    exp_csv = tmp_path / "expanded.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "split", "scenic_human_mean", "region", "slice"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "img1.jpg",
                "split": "test",
                "scenic_human_mean": "5.0",
                "region": "us_west",
                "slice": "mountain",
            }
        )
        writer.writerow(
            {
                "image_path": "img2.jpg",
                "split": "test",
                "scenic_human_mean": "6.0",
                "region": "us_east",
                "slice": "coastal",
            }
        )

    ctrl_csv = tmp_path / "control.csv"
    with open(ctrl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "split", "scenic_human_mean", "region", "slice"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "img3.jpg",
                "split": "test",
                "scenic_human_mean": "4.5",
                "region": "us_central",
                "slice": "plains",
            }
        )
        writer.writerow(
            {
                "image_path": "img4.jpg",
                "split": "test",
                "scenic_human_mean": "7.2",
                "region": "us_south",
                "slice": "forest",
            }
        )

    output_json = tmp_path / "baseline_summary.json"
    summary = evaluate_active_baseline(
        dataset_path=npz_path,
        control_dataset_path=control_npz,
        checkpoint_path=ckpt,
        expanded_benchmark_csv=exp_csv,
        control_benchmark_csv=ctrl_csv,
        output_path=output_json,
    )

    assert output_json.exists()
    assert summary["hashes"]["dataset_sha256"] == file_sha256(npz_path)
    assert summary["hashes"]["checkpoint_sha256"] == file_sha256(ckpt)
    assert summary["hashes"]["expanded_benchmark_sha256"] == file_sha256(exp_csv)
    assert summary["hashes"]["control_benchmark_sha256"] == file_sha256(ctrl_csv)

    assert summary["deterministic_inference"]["tolerance"] == 1e-7
    assert summary["deterministic_inference"]["max_absolute_difference"] <= 1e-7

    assert summary["sample_counts"]["dataset_total"] == 2
    assert summary["sample_counts"]["dataset_test"] == 2
    assert summary["sample_counts"]["expanded_benchmark_test"] == 2
    assert summary["sample_counts"]["control_benchmark_test"] == 2

    assert "expanded_human_benchmark" in summary["benchmarks"]
    assert "control_benchmark" in summary["benchmarks"]
    exp_m = summary["benchmarks"]["expanded_human_benchmark"]["metrics"]
    assert exp_m["samples"] == 2
    assert "mae" in exp_m
    assert "rmse" in exp_m
    assert "pearson_corr" in exp_m
    assert "spearman_corr" in exp_m

    assert "expanded_human_benchmark" in summary["calibration_distribution_summary"]
    cal = summary["calibration_distribution_summary"]["expanded_human_benchmark"]
    assert "calibration_error" in cal
    assert "prediction_mean" in cal


def test_evaluate_active_baseline_missing_identity(tmp_path: Path) -> None:
    ckpt = create_legacy_checkpoint(tmp_path / "baseline.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["img1.jpg"],
        splits=["test"],
    )
    control_npz = create_npz_dataset(
        tmp_path / "control_dataset.npz",
        image_paths=["img3.jpg"],
        splits=["test"],
    )

    exp_csv = tmp_path / "expanded.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "5.0"}
        )
        writer.writerow(
            {
                "image_path": "img_missing.jpg",
                "split": "test",
                "scenic_human_mean": "6.0",
            }
        )

    ctrl_csv = tmp_path / "control.csv"
    with open(ctrl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img3.jpg", "split": "test", "scenic_human_mean": "4.5"}
        )

    with pytest.raises(ValueError, match="absent from the prepared dataset"):
        evaluate_active_baseline(
            dataset_path=npz_path,
            control_dataset_path=control_npz,
            checkpoint_path=ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
        )


def test_evaluate_active_baseline_split_mismatch(tmp_path: Path) -> None:
    ckpt = create_legacy_checkpoint(tmp_path / "baseline.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["img1.jpg"],
        splits=["train"],
    )
    control_npz = create_npz_dataset(
        tmp_path / "control_dataset.npz",
        image_paths=["img2.jpg"],
        splits=["test"],
    )

    exp_csv = tmp_path / "expanded.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "5.0"}
        )

    ctrl_csv = tmp_path / "control.csv"
    with open(ctrl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img2.jpg", "split": "test", "scenic_human_mean": "4.5"}
        )

    with pytest.raises(ValueError, match="relabels non-test prepared identity"):
        evaluate_active_baseline(
            dataset_path=npz_path,
            control_dataset_path=control_npz,
            checkpoint_path=ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
        )


def test_evaluate_active_baseline_nondeterminism_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = create_legacy_checkpoint(tmp_path / "baseline.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["img1.jpg"],
        splits=["test"],
    )
    control_npz = create_npz_dataset(
        tmp_path / "control_dataset.npz",
        image_paths=["img2.jpg"],
        splits=["test"],
    )

    exp_csv = tmp_path / "expanded.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "5.0"}
        )

    ctrl_csv = tmp_path / "control.csv"
    with open(ctrl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img2.jpg", "split": "test", "scenic_human_mean": "4.5"}
        )

    import src.scenic_scorer.active_evaluation

    original_predict = src.scenic_scorer.active_evaluation._predict_dataset
    call_count = 0

    def mock_predict(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        res = original_predict(*args, **kwargs)
        if call_count == 2:
            res = res + 1e-5
        return res

    monkeypatch.setattr(
        src.scenic_scorer.active_evaluation, "_predict_dataset", mock_predict
    )

    with pytest.raises(ValueError, match="Deterministic CPU inference failure"):
        evaluate_active_baseline(
            dataset_path=npz_path,
            control_dataset_path=control_npz,
            checkpoint_path=ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
        )


def test_evaluate_active_baseline_disjoint_benchmark_overlap(tmp_path: Path) -> None:
    ckpt = create_legacy_checkpoint(tmp_path / "baseline.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["img1.jpg", "img2.jpg"],
        splits=["test", "test"],
    )
    control_npz = create_npz_dataset(
        tmp_path / "control_dataset.npz",
        image_paths=["img1.jpg"],
        splits=["test"],
    )

    exp_csv = tmp_path / "expanded.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "5.0"}
        )

    ctrl_csv = tmp_path / "control.csv"
    with open(ctrl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "4.5"}
        )

    with pytest.raises(
        ValueError, match="Overlap detected between expanded and control"
    ):
        evaluate_active_baseline(
            dataset_path=npz_path,
            control_dataset_path=control_npz,
            checkpoint_path=ckpt,
            expanded_benchmark_csv=exp_csv,
            control_benchmark_csv=ctrl_csv,
        )


def test_evaluate_candidate_validation_filters_only_split_val_rows(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_legacy_checkpoint(tmp_path / "base.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["val1.jpg", "val2.jpg", "test1.jpg"],
        splits=["val", "val", "test"],
    )

    exp_csv = tmp_path / "benchmark.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean", "region"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "val1.jpg", "split": "val", "scenic_human_mean": "6.0", "region": "r1"}
        )
        writer.writerow(
            {"image_path": "val2.jpg", "split": "val", "scenic_human_mean": "8.0", "region": "r1"}
        )
        writer.writerow(
            {"image_path": "test1.jpg", "split": "test", "scenic_human_mean": "1.0", "region": "r2"}
        )

    out_json = tmp_path / "val_result.json"
    result = evaluate_candidate_validation(
        dataset_path=npz_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=base_ckpt,
        expanded_benchmark_csv=exp_csv,
        output_path=out_json,
    )

    assert result["candidate_metrics"]["samples"] == 2
    assert result["baseline_metrics"]["samples"] == 2
    assert out_json.exists()


def test_evaluate_candidate_validation_rejects_missing_val_or_no_val_rows(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_legacy_checkpoint(tmp_path / "base.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["img1.jpg"],
        splits=["test"],
    )

    # 1. No split column
    csv_no_split = tmp_path / "no_split.csv"
    with open(csv_no_split, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "scenic_human_mean"])
        writer.writeheader()
        writer.writerow({"image_path": "img1.jpg", "scenic_human_mean": "5.0"})

    with pytest.raises(ValueError, match="requires an explicit split column"):
        evaluate_candidate_validation(
            dataset_path=npz_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=csv_no_split,
            output_path=tmp_path / "out.json",
        )

    # 2. No split=val rows
    csv_no_val = tmp_path / "no_val.csv"
    with open(csv_no_val, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "split", "scenic_human_mean"])
        writer.writeheader()
        writer.writerow(
            {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "5.0"}
        )

    with pytest.raises(ValueError, match="contains no split=val rows"):
        evaluate_candidate_validation(
            dataset_path=npz_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=csv_no_val,
            output_path=tmp_path / "out.json",
        )


def test_evaluate_candidate_validation_rejects_mislabeled_or_non_val_npz_identity(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_legacy_checkpoint(tmp_path / "base.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["img1.jpg"],
        splits=["test"],
    )

    exp_csv = tmp_path / "mislabeled.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "split", "scenic_human_mean"])
        writer.writeheader()
        writer.writerow(
            {"image_path": "img1.jpg", "split": "val", "scenic_human_mean": "5.0"}
        )

    with pytest.raises(ValueError, match="relabels non-val prepared identity"):
        evaluate_candidate_validation(
            dataset_path=npz_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_csv,
            output_path=tmp_path / "out.json",
        )


def test_evaluate_candidate_validation_rejects_weak_targets_and_nonfinite(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_legacy_checkpoint(tmp_path / "base.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["val1.jpg"],
        splits=["val"],
    )

    # 1. Weak target field only (scenic_score)
    weak_csv = tmp_path / "weak.csv"
    with open(weak_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "split", "scenic_score"])
        writer.writeheader()
        writer.writerow({"image_path": "val1.jpg", "split": "val", "scenic_score": "5.0"})

    with pytest.raises(ValueError, match="val target must be scenic_human_mean or scenic_human"):
        evaluate_candidate_validation(
            dataset_path=npz_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=weak_csv,
            output_path=tmp_path / "out.json",
        )

    # 2. Out of range human target (> 10)
    range_csv = tmp_path / "range.csv"
    with open(range_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"image_path": "val1.jpg", "split": "val", "scenic_human_mean": "15.0"}
        )

    with pytest.raises(ValueError, match="human targets must be finite and in \\[0, 10\\]"):
        evaluate_candidate_validation(
            dataset_path=npz_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=range_csv,
            output_path=tmp_path / "out.json",
        )


def test_evaluate_candidate_validation_exact_denominator_and_metrics(
    tmp_path: Path,
) -> None:
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.0)
    base_ckpt = create_legacy_checkpoint(tmp_path / "base.pt")
    npz_path = create_npz_dataset(
        tmp_path / "dataset.npz",
        image_paths=["v1.jpg", "v2.jpg"],
        splits=["val", "val"],
    )

    exp_csv = tmp_path / "valid.csv"
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "split", "scenic_human_mean", "region", "slice"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "v1.jpg",
                "split": "val",
                "scenic_human_mean": "5.0",
                "region": "r1",
                "slice": "s1",
            }
        )
        writer.writerow(
            {
                "image_path": "v2.jpg",
                "split": "val",
                "scenic_human_mean": "7.0",
                "region": "r1",
                "slice": "s1",
            }
        )

    out_json = tmp_path / "decision_val.json"
    result = evaluate_candidate_validation(
        dataset_path=npz_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=base_ckpt,
        expanded_benchmark_csv=exp_csv,
        output_path=out_json,
    )

    assert "candidate" in result
    assert "baseline" in result
    assert "candidate_metrics" in result
    assert "baseline_metrics" in result
    assert "mse_improvement" in result
    assert "sliced_metrics" in result
    assert "region_metrics" in result
    assert result["candidate_metrics"]["samples"] == 2
    assert json.loads(out_json.read_text(encoding="utf-8")) == result
