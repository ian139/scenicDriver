from __future__ import annotations

import csv
import json
from pathlib import Path
import typing

import numpy as np
import pytest
import torch

from src.scenic_scorer.active_evaluation import (
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
) -> Path:
    n = len(image_paths)
    vit = np.linspace(-1.0, 1.0, n * vit_dim, dtype=np.float32).reshape(n, vit_dim)
    terr = np.linspace(0.1, 0.9, n * terrain_dim, dtype=np.float32).reshape(n, terrain_dim)
    cls_logits = np.linspace(-0.5, 0.5, n * num_classes, dtype=np.float32).reshape(n, num_classes)
    scenic_scores = np.linspace(2.0, 8.0, n, dtype=np.float32)
    sample_weights = np.ones(n, dtype=np.float32)
    label_sources = np.array(["human"] * n)
    np.savez_compressed(
        path,
        vit_embeddings=vit,
        terrain_features=terr,
        class_logits=cls_logits,
        scenic_scores=scenic_scores,
        sample_weights=sample_weights,
        image_paths=np.array(image_paths),
        label_sources=label_sources,
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
    images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt", bias_shift=0.0)
    
    exp_csv = create_benchmark_csv(tmp_path / "exp.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
        {"image_path": "img2.jpg", "split": "test", "scenic_human_mean": "6.0", "region": "r1", "slice": "s1"},
        {"image_path": "img3.jpg", "split": "train", "scenic_human_mean": "1.0", "region": "r1", "slice": "s1"},
        {"image_path": "img4.jpg", "split": "val", "scenic_human_mean": "9.0", "region": "r1", "slice": "s1"},
    ])
    ctrl_csv = create_benchmark_csv(tmp_path / "ctrl.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human": "4.0", "region": "r1", "slice": "s1"},
        {"image_path": "img2.jpg", "split": "test", "scenic_human": "6.0", "region": "r1", "slice": "s1"},
    ])
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"
    
    decision = evaluate_stage_two(
        dataset_path=dataset_path,
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


def test_evaluate_stage_two_rejects_missing_split_or_no_test_rows(tmp_path: Path) -> None:
    images = ["img1.jpg", "img2.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"
    
    # Missing split column
    exp_no_split = create_benchmark_csv(tmp_path / "no_split.csv", [
        {"image_path": "img1.jpg", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
    ])
    ctrl_csv = create_benchmark_csv(tmp_path / "ctrl.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
    ])
    with pytest.raises(ValueError, match="requires an explicit split column"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_no_split,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )

    # No test rows (only train)
    exp_train_only = create_benchmark_csv(tmp_path / "train_only.csv", [
        {"image_path": "img1.jpg", "split": "train", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
    ])
    with pytest.raises(ValueError, match="contains no split=test rows"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_train_only,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_rejects_weak_only_targets(tmp_path: Path) -> None:
    images = ["img1.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt")
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt")
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"
    
    exp_weak_only = create_benchmark_csv(tmp_path / "weak_only.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_score": "4.0", "region": "r1", "slice": "s1"},
    ])
    ctrl_csv = create_benchmark_csv(tmp_path / "ctrl.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
    ])
    with pytest.raises(ValueError, match="must be scenic_human_mean or scenic_human, never weak/mixed scenic_score"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_weak_only,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_denominator_and_hash_reporting(tmp_path: Path) -> None:
    images = ["img1.jpg", "img2.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.1)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt", bias_shift=0.0)
    
    exp_csv = create_benchmark_csv(tmp_path / "exp.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
        {"image_path": "img2.jpg", "split": "test", "scenic_human_mean": "6.0", "region": "r1", "slice": "s1"},
    ])
    ctrl_csv = create_benchmark_csv(tmp_path / "ctrl.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human": "4.0", "region": "r1", "slice": "s1"},
        {"image_path": "img2.jpg", "split": "test", "scenic_human": "6.0", "region": "r1", "slice": "s1"},
    ])
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"
    
    decision = evaluate_stage_two(
        dataset_path=dataset_path,
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
    exp_absent_path = create_benchmark_csv(tmp_path / "exp_absent.csv", [
        {"image_path": "img_missing.jpg", "split": "test", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
    ])
    with pytest.raises(ValueError, match="absent from the prepared dataset"):
        evaluate_stage_two(
            dataset_path=dataset_path,
            candidate_checkpoint=cand_ckpt,
            baseline_checkpoint=base_ckpt,
            expanded_benchmark_csv=exp_absent_path,
            control_benchmark_csv=ctrl_csv,
            route_qa_json=route_qa,
            thresholds=get_default_thresholds(),
            output_path=out_json,
        )


def test_evaluate_stage_two_passing_decision_schema_and_gates(tmp_path: Path) -> None:
    images = ["img1.jpg", "img2.jpg"]
    dataset_path = create_npz_dataset(tmp_path / "dataset.npz", images)
    cand_ckpt = create_real_checkpoint(tmp_path / "cand.pt", bias_shift=0.0)
    base_ckpt = create_real_checkpoint(tmp_path / "base.pt", bias_shift=0.0)
    
    exp_csv = create_benchmark_csv(tmp_path / "exp.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human_mean": "4.0", "region": "r1", "slice": "s1"},
        {"image_path": "img2.jpg", "split": "test", "scenic_human_mean": "6.0", "region": "r1", "slice": "s1"},
    ])
    ctrl_csv = create_benchmark_csv(tmp_path / "ctrl.csv", [
        {"image_path": "img1.jpg", "split": "test", "scenic_human": "4.0", "region": "r1", "slice": "s1"},
        {"image_path": "img2.jpg", "split": "test", "scenic_human": "6.0", "region": "r1", "slice": "s1"},
    ])
    route_qa = create_route_qa_json(tmp_path / "route_qa.json")
    out_json = tmp_path / "decision.json"
    
    decision = evaluate_stage_two(
        dataset_path=dataset_path,
        candidate_checkpoint=cand_ckpt,
        baseline_checkpoint=base_ckpt,
        expanded_benchmark_csv=exp_csv,
        control_benchmark_csv=ctrl_csv,
        route_qa_json=route_qa,
        thresholds=get_default_thresholds(),
        output_path=out_json,
    )
    
    expected_top_keys = {
        "timestamp", "all_gates_pass", "candidate", "baseline",
        "expanded_human_benchmark", "control_benchmark",
        "calibration_and_distribution", "route_qa_evidence",
        "thresholds_evaluated", "gates",
    }
    assert set(decision.keys()) == expected_top_keys
    assert decision["all_gates_pass"] is True
    
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
    route_qa_failed = create_route_qa_json(tmp_path / "route_failed.json", all_invariants_pass=False)
    decision_failed = evaluate_stage_two(
        dataset_path=dataset_path,
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
    
    registry = tmp_path / "registry.json"
    initial_registry_data = {
        "active": {"checkpoint": str(tmp_path / "base.pt"), "sha256": "0" * 64},
        "history": [],
    }
    registry.write_text(json.dumps(initial_registry_data, indent=2), encoding="utf-8")
    reg_hash = _sha(registry)
    
    decision_file = tmp_path / "decision.json"
    decision_payload = {
        "all_gates_pass": True,
        "candidate": {"checkpoint": str(cand_ckpt), "sha256": _sha(cand_ckpt)},
        "expanded_human_benchmark": {"candidate_metrics": {"mse": 0.1}},
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
    bad_decision_payload["candidate"] = {"checkpoint": str(cand_ckpt), "sha256": "0" * 64}
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
    
    registry = tmp_path / "registry.json"
    initial_registry_data = {
        "active": {"checkpoint": str(tmp_path / "base.pt"), "sha256": "0" * 64},
        "history": [],
    }
    registry.write_text(json.dumps(initial_registry_data, indent=2), encoding="utf-8")
    
    dec1 = tmp_path / "dec1.json"
    dec1.write_text(json.dumps({
        "all_gates_pass": True,
        "candidate": {"checkpoint": str(cand_ckpt1), "sha256": _sha(cand_ckpt1)},
    }), encoding="utf-8")
    
    res1 = promote_from_decision(
        decision_path=dec1,
        candidate_checkpoint=cand_ckpt1,
        registry_path=registry,
        expected_registry_sha256=_sha(registry),
        run_name="run_1",
    )
    
    dec2 = tmp_path / "dec2.json"
    dec2.write_text(json.dumps({
        "all_gates_pass": True,
        "candidate": {"checkpoint": str(cand_ckpt2), "sha256": _sha(cand_ckpt2)},
    }), encoding="utf-8")
    
    res2 = promote_from_decision(
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
        rollback_registry(registry, target_history_index=0, expected_registry_sha256="wrong_hash" + "0" * 54)

    # Rollback with invalid target index fails
    with pytest.raises(ValueError, match="Invalid target history index"):
        rollback_registry(registry, target_history_index=99, expected_registry_sha256=_sha(registry))

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
    with pytest.raises(ValueError, match="Historical checkpoint identity validation failed"):
        rollback_registry(registry, target_history_index=0, expected_registry_sha256=_sha(registry))


def test_rejected_decision_preserves_registry_bytes(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"active": {"checkpoint": "old.pt"}, "history": []}), encoding="utf-8")
    before = registry.read_bytes()
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"all_gates_pass": False}), encoding="utf-8")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        promote_from_decision(decision, candidate, registry, _sha(registry), "run")
    assert registry.read_bytes() == before
