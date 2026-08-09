"""Focused tests for Stage-Two autoresearch orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import pytest

from scripts.modeling.run_active_scenic_autoresearch import (
    EXPOSURE_MIN_FRESH_TEST_ROWS,
    EXPOSURE_REJECTION_REASON,
    FIXED_VALIDATION_SELECTION_ROWS,
    DeadlineExceededError,
    build_candidate_ladder,
    build_exposure_annotation_request,
    compute_experiment_digest,
    compute_sha256,
    deadline_guard,
    detect_heldout_test_exposure,
    determine_aggregate_run_state,
    has_validation_evidence,
    is_valid_paused_experiment,
    load_existing_experiments,
    parse_args,
    resolve_stage_one_handoff,
    sanitize_command,
    select_best_candidate,
    select_validation_finalist,
    validate_experiment_record,
    validate_handoff_content,
)
from src.scenic_scorer.active_training import ActiveTrainingConfig


@pytest.fixture(autouse=True)
def _mock_validate_handoff_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    def _mock_preflight(p: Path) -> dict[str, int]:
        return {
            "artifacts": 1,
            "candidate_rows": 10,
            "split_rows": 10,
            "expanded_rows": 5,
            "control_rows": 5,
            "leakage_violations": 0,
        }

    monkeypatch.setattr(mod, "validate_handoff", _mock_preflight)


def _handoff(path: Path, *, ready: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "ready_for_stage2": ready,
        "blockers": [] if ready else ["incomplete"],
        "artifacts": {},
    }
    result = path / "stage1_handoff.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def test_explicit_handoff_requires_ready_state(tmp_path: Path) -> None:
    handoff = _handoff(tmp_path / "run", ready=False)
    resolved, payload = resolve_stage_one_handoff(str(handoff))
    assert resolved == handoff
    with pytest.raises(ValueError, match="not marked ready"):
        validate_handoff_content(resolved, payload)


def test_implicit_handoff_rejects_ambiguous_ready_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data" / "processed" / "active_learning"
    _handoff(root / "a")
    _handoff(root / "b")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        resolve_stage_one_handoff()


def _create_valid_exp_files(
    tmp_path: Path,
    exp_id: str,
    *,
    all_gates_pass: bool = True,
    baseline_sha256: str = "base_sha",
    validation_metrics: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, str]:
    exp_dir = tmp_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / "candidate_model.pt"
    ckpt_path.write_bytes(f"model_data_{exp_id}".encode("utf-8"))
    ckpt_hash = compute_sha256(ckpt_path)

    eval_path = exp_dir / "eval_decision.json"
    eval_data = {
        "all_gates_pass": all_gates_pass,
        "candidate": {"checkpoint": str(ckpt_path), "sha256": ckpt_hash},
        "baseline": {"checkpoint": "base.pt", "sha256": baseline_sha256},
    }
    eval_path.write_text(json.dumps(eval_data), encoding="utf-8")

    metrics = dict(validation_metrics) if validation_metrics else {"mse": 0.05}

    validation_path = exp_dir / "validation_decision.json"
    validation_data = {
        "candidate": {"checkpoint": str(ckpt_path), "sha256": ckpt_hash},
        "baseline": {"checkpoint": "base.pt", "sha256": baseline_sha256},
        "candidate_metrics": metrics,
    }
    validation_path.write_text(json.dumps(validation_data), encoding="utf-8")
    return ckpt_path, eval_path, validation_path, ckpt_hash


def _build_run_env(tmp_path: Path) -> dict[str, Any]:
    """Create registry, handoff, benchmark inputs, and thresholds for main()."""
    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")

    val_rows = "".join(
        f"val,img_val_{i}.jpg,{5.0 + i * 0.5}\n" for i in range(5)
    )
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text(
        "split,image_path,scenic_human_mean\n" + val_rows, encoding="utf-8"
    )
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("split,image_path,scenic_human_mean\n", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")
    thresh_json = tmp_path / "thresholds.json"
    thresh_json.write_text(
        json.dumps({"min_expanded_validation_samples": 5}), encoding="utf-8"
    )

    return {
        "handoff": handoff,
        "exp_csv": exp_csv,
        "ctrl_csv": ctrl_csv,
        "route_json": route_json,
        "thresh_json": thresh_json,
        "reg_file": reg_file,
        "baseline_ckpt": ckpt,
    }


def _run_args(env: dict[str, Any], run_name: str, *, max_experiments: int = 2) -> list[str]:
    return [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(env["handoff"]),
        "--run-name",
        run_name,
        "--max-experiments",
        str(max_experiments),
        "--expanded-benchmark-csv",
        str(env["exp_csv"]),
        "--control-benchmark-csv",
        str(env["ctrl_csv"]),
        "--control-dataset",
        str(env["ctrl_csv"]),
        "--route-qa-json",
        str(env["route_json"]),
        "--thresholds-json",
        str(env["thresh_json"]),
    ]


def _mock_prepare(h_path: Path, out_path: Path) -> dict[str, Any]:
    """Realistic prepare mock: materialize NPZ plus split CSV sidecar."""
    out_path.write_bytes(b"dataset_npz_data")
    split_csv = out_path.parent / "prepared_split.csv"
    split_csv.write_text("split,image_path\n", encoding="utf-8")
    return {"dataset_path": out_path, "split_path": split_csv}


def _make_train_mock(train_calls: dict[str, Any]):
    """Train mock writing candidate.pt + completed training summary per exp dir."""

    def _mock_train(*, dataset_path, split_csv, output_dir, config, resume):
        exp_dir = Path(output_dir)
        exp_name = exp_dir.name
        train_calls[exp_name] = {"resume": resume}
        ckpt = exp_dir / "candidate.pt"
        ckpt.write_bytes(f"candidate_{exp_name}".encode("utf-8"))
        (exp_dir / "training_summary.json").write_text(
            json.dumps({"state": "completed"}), encoding="utf-8"
        )
        return {"state": "completed", "candidate_checkpoint": str(ckpt)}

    return _mock_train


def _make_validation_mock(mse_by_exp: dict[str, float]):
    """Validation mock writing a truthful validation_decision.json per candidate."""

    def _mock_evaluate_candidate_validation(**kwargs: Any) -> dict[str, Any]:
        cand = str(kwargs["candidate_checkpoint"])
        exp_name = Path(cand).parent.name
        mse = float(mse_by_exp.get(exp_name, 0.5))
        payload = {
            "timestamp": "2026-08-09T00:00:00Z",
            "candidate": {"checkpoint": cand, "sha256": compute_sha256(cand)},
            "baseline": {
                "checkpoint": str(kwargs["baseline_checkpoint"]),
                "sha256": compute_sha256(kwargs["baseline_checkpoint"]),
            },
            "candidate_metrics": {
                "samples": 64,
                "mse": mse,
                "mae": mse**0.5,
                "rmse": mse**0.5,
                "r2": 0.9,
                "pearson_corr": 0.95,
                "spearman_corr": 0.94,
            },
            "baseline_metrics": {"mse": mse + 0.1},
            "mse_improvement": 0.1,
        }
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    return _mock_evaluate_candidate_validation


def _make_stage_two_mock(
    stage_two_calls: list[dict[str, Any]], *, all_gates_pass: bool = True
):
    """Full stage-two mock writing a truthful eval_decision.json for the winner."""

    def _mock_stage_two(**kwargs: Any) -> dict[str, Any]:
        cand = str(kwargs["candidate_checkpoint"])
        stage_two_calls.append({"candidate_checkpoint": cand})
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "all_gates_pass": all_gates_pass,
            "candidate": {"checkpoint": cand, "sha256": compute_sha256(cand)},
            "baseline": {"sha256": compute_sha256(kwargs["baseline_checkpoint"])},
            "expanded_human_benchmark": {
                "candidate_metrics": {
                    "samples": 68,
                    "mse": 0.9,
                    "mae": 0.5,
                    "rmse": 0.95,
                    "pearson_corr": 0.9,
                }
            },
        }
        out.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    return _mock_stage_two


def test_resume_loads_only_completed_and_valid_records(tmp_path: Path) -> None:
    ckpt1, eval1, val1, _ = _create_valid_exp_files(
        tmp_path, "exp_01", validation_metrics={"mse": 0.05}
    )
    ckpt3, eval3, val3, _ = _create_valid_exp_files(
        tmp_path, "exp_03", validation_metrics={"mse": 0.07}
    )

    records = [
        {
            "exp_id": "exp_01",
            "status": "completed",
            "candidate_checkpoint": str(ckpt1),
            "eval_decision_path": str(eval1),
            "validation_decision_path": str(val1),
            "validation_metrics": {"mse": 0.05},
            "all_gates_pass": True,
        },
        {"exp_id": "exp_02", "status": "failed"},
        {
            "exp_id": "exp_03",
            "status": "retained",
            "candidate_checkpoint": str(ckpt3),
            "validation_decision_path": str(val3),
            "validation_metrics": {"mse": 0.07},
            "eval_decision_path": str(eval3),
            "all_gates_pass": True,
        },
        {
            "exp_id": "exp_04",
            "status": "completed",
            "candidate_checkpoint": str(tmp_path / "nonexistent.pt"),
            "eval_decision_path": str(eval1),
            "validation_decision_path": str(val1),
            "validation_metrics": {"mse": 0.09},
            "all_gates_pass": True,
        },
    ]
    path = tmp_path / "experiments.jsonl"
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    completed, all_records = load_existing_experiments(path)
    assert set(completed) == {"exp_01", "exp_03"}
    assert len(all_records) == 4


def test_parse_args_max_seconds_default_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_args = [
        "run_active_scenic_autoresearch.py",
        "--expanded-benchmark-csv",
        "exp.csv",
        "--control-benchmark-csv",
        "ctrl.csv",
        "--control-dataset",
        "ctrl.csv",
        "--route-qa-json",
        "route.json",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = parse_args()
    assert parsed.max_seconds == 1800.0

    monkeypatch.setattr(
        sys,
        "argv",
        test_args + ["--max-seconds", "-10"],
    )
    with pytest.raises(ValueError, match="must be positive"):
        parse_args()


def test_select_best_candidate_corr_descending_mae_ascending() -> None:
    records = [
        {
            "exp_id": "exp_01",
            "all_gates_pass": True,
            "metrics": {"pearson_corr": 0.80, "mae": 0.10},
        },
        {
            "exp_id": "exp_02",
            "all_gates_pass": True,
            "metrics": {"pearson_corr": 0.85, "mae": 0.20},
        },
        {
            "exp_id": "exp_03",
            "all_gates_pass": True,
            "metrics": {"pearson_corr": 0.85, "mae": 0.12},
        },
        {
            "exp_id": "exp_04",
            "all_gates_pass": False,
            "metrics": {"pearson_corr": 0.95, "mae": 0.01},
        },
    ]
    best = select_best_candidate(records)
    assert best is not None
    assert best["exp_id"] == "exp_03"


def test_validate_experiment_record_contract(tmp_path: Path) -> None:
    ckpt, eval_p, val_p, hash_val = _create_valid_exp_files(
        tmp_path, "exp_val", baseline_sha256="expected_base"
    )
    rec = {
        "exp_id": "exp_val",
        "status": "completed",
        "candidate_checkpoint": str(ckpt),
        "eval_decision_path": str(eval_p),
        "validation_decision_path": str(val_p),
        "validation_metrics": {"mse": 0.05},
        "all_gates_pass": True,
    }
    # Legacy 'completed' records lacking validation evidence never bypass validation
    legacy_no_metrics = {k: v for k, v in rec.items() if k != "validation_metrics"}
    assert validate_experiment_record(legacy_no_metrics) is False
    legacy_no_path = {k: v for k, v in rec.items() if k != "validation_decision_path"}
    assert validate_experiment_record(legacy_no_path) is False
    # 'validated' records are reusable with validation evidence alone
    validated_rec = dict(rec, status="validated")
    del validated_rec["eval_decision_path"]
    del validated_rec["all_gates_pass"]
    assert (
        validate_experiment_record(
            validated_rec, expected_baseline_sha256="expected_base"
        )
        is True
    )
    # Validation decision file candidate hash must match the checkpoint
    bad_val_path = tmp_path / "bad_val.json"
    bad_val_path.write_text(
        json.dumps(
            {
                "candidate": {"checkpoint": str(ckpt), "sha256": "wrong_hash"},
                "baseline": {"sha256": "expected_base"},
            }
        ),
        encoding="utf-8",
    )
    bad_val_rec = dict(rec, validation_decision_path=str(bad_val_path))
    assert validate_experiment_record(bad_val_rec) is False
    # Valid
    assert (
        validate_experiment_record(rec, expected_baseline_sha256="expected_base")
        is True
    )
    # Baseline mismatch
    assert (
        validate_experiment_record(rec, expected_baseline_sha256="other_base") is False
    )
    # Gate pass mismatch
    bad_gate_rec = dict(rec, all_gates_pass=False)
    assert (
        validate_experiment_record(
            bad_gate_rec, expected_baseline_sha256="expected_base"
        )
        is False
    )
    # Missing all_gates_pass in rec
    no_gate_rec = {k: v for k, v in rec.items() if k != "all_gates_pass"}
    assert (
        validate_experiment_record(
            no_gate_rec, expected_baseline_sha256="expected_base"
        )
        is False
    )
    # Missing checkpoint file
    bad_ckpt_rec = dict(rec, candidate_checkpoint=str(tmp_path / "missing.pt"))
    assert validate_experiment_record(bad_ckpt_rec) is False
    # Missing eval decision file
    bad_eval_rec = dict(rec, eval_decision_path=str(tmp_path / "missing.json"))
    assert validate_experiment_record(bad_eval_rec) is False

    # Missing all_gates_pass in eval_decision JSON
    no_eval_gate_path = tmp_path / "no_eval_gate.json"
    no_eval_gate_path.write_text(
        json.dumps({"candidate": {"checkpoint": str(ckpt), "sha256": hash_val}}),
        encoding="utf-8",
    )
    no_eval_gate_rec = dict(rec, eval_decision_path=str(no_eval_gate_path))
    assert validate_experiment_record(no_eval_gate_rec) is False

    # Missing baseline mapping in decision JSON
    no_base_path = tmp_path / "no_base.json"
    no_base_path.write_text(
        json.dumps(
            {
                "all_gates_pass": True,
                "candidate": {"checkpoint": str(ckpt), "sha256": hash_val},
            }
        ),
        encoding="utf-8",
    )
    no_base_rec = dict(rec, eval_decision_path=str(no_base_path))
    assert (
        validate_experiment_record(
            no_base_rec, expected_baseline_sha256="expected_base"
        )
        is False
    )

    # Empty baseline sha in decision JSON
    empty_base_path = tmp_path / "empty_base.json"
    empty_base_path.write_text(
        json.dumps(
            {
                "all_gates_pass": True,
                "candidate": {"checkpoint": str(ckpt), "sha256": hash_val},
                "baseline": {"sha256": ""},
            }
        ),
        encoding="utf-8",
    )
    empty_base_rec = dict(rec, eval_decision_path=str(empty_base_path))
    assert (
        validate_experiment_record(
            empty_base_rec, expected_baseline_sha256="expected_base"
        )
        is False
    )


def test_validate_experiment_record_rejects_tampered_validation_metrics(
    tmp_path: Path,
) -> None:
    metrics = {
        "samples": 64,
        "mse": 0.05,
        "mae": 0.22,
        "rmse": 0.22,
        "pearson_corr": 0.9,
        "spearman_corr": 0.88,
    }
    ckpt, eval_p, val_p, _ = _create_valid_exp_files(
        tmp_path, "exp_tamper", validation_metrics=metrics
    )
    rec = {
        "exp_id": "exp_tamper",
        "status": "completed",
        "candidate_checkpoint": str(ckpt),
        "eval_decision_path": str(eval_p),
        "validation_decision_path": str(val_p),
        "validation_metrics": dict(metrics),
        "all_gates_pass": True,
    }
    # Untampered record passes: JSONL metrics exactly match the artifact.
    assert validate_experiment_record(rec) is True

    # Tampered JSONL mse no longer equals the decision artifact.
    tampered_mse = dict(rec, validation_metrics=dict(metrics, mse=0.01))
    assert validate_experiment_record(tampered_mse) is False

    # Tampered JSONL pearson_corr no longer equals the decision artifact.
    tampered_corr = dict(
        rec, validation_metrics=dict(metrics, pearson_corr=0.99)
    )
    assert validate_experiment_record(tampered_corr) is False

    # Missing selection key in the record (artifact-backed samples stripped).
    missing_samples = dict(
        rec,
        validation_metrics={
            k: v for k, v in metrics.items() if k != "samples"
        },
    )
    assert validate_experiment_record(missing_samples) is False

    # Missing selection key in the decision artifact (stale artifact).
    stale_val_path = tmp_path / "stale_val.json"
    stale_val_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "checkpoint": str(ckpt),
                    "sha256": compute_sha256(ckpt),
                },
                "baseline": {"sha256": "base_sha"},
                "candidate_metrics": {
                    k: v for k, v in metrics.items() if k != "pearson_corr"
                },
            }
        ),
        encoding="utf-8",
    )
    stale_rec = dict(rec, validation_decision_path=str(stale_val_path))
    assert validate_experiment_record(stale_rec) is False

    # Decision artifact without candidate_metrics entirely (missing evidence).
    no_metrics_path = tmp_path / "no_metrics_val.json"
    no_metrics_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "checkpoint": str(ckpt),
                    "sha256": compute_sha256(ckpt),
                },
                "baseline": {"sha256": "base_sha"},
            }
        ),
        encoding="utf-8",
    )
    no_metrics_rec = dict(rec, validation_decision_path=str(no_metrics_path))
    assert validate_experiment_record(no_metrics_rec) is False

    # Tampered artifact metrics (stale artifact with rewritten mse).
    tampered_artifact_path = tmp_path / "tampered_artifact_val.json"
    tampered_artifact_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "checkpoint": str(ckpt),
                    "sha256": compute_sha256(ckpt),
                },
                "baseline": {"sha256": "base_sha"},
                "candidate_metrics": dict(metrics, mse=0.09),
            }
        ),
        encoding="utf-8",
    )
    tampered_artifact_rec = dict(
        rec, validation_decision_path=str(tampered_artifact_path)
    )
    assert validate_experiment_record(tampered_artifact_rec) is False


def test_validate_experiment_record_rejects_nonfinite_metrics(
    tmp_path: Path,
) -> None:
    base_metrics = {
        "samples": 64,
        "mse": 0.05,
        "mae": 0.22,
        "rmse": 0.22,
        "pearson_corr": 0.9,
        "spearman_corr": 0.88,
    }
    ckpt, eval_p, val_p, _ = _create_valid_exp_files(
        tmp_path, "exp_finite", validation_metrics=dict(base_metrics)
    )
    rec = {
        "exp_id": "exp_finite",
        "status": "completed",
        "candidate_checkpoint": str(ckpt),
        "eval_decision_path": str(eval_p),
        "validation_decision_path": str(val_p),
        "validation_metrics": dict(base_metrics),
        "all_gates_pass": True,
    }
    assert validate_experiment_record(rec) is True

    # NaN mse in the record is never usable selection evidence.
    nan_mse = dict(
        rec, validation_metrics=dict(base_metrics, mse=float("nan"))
    )
    assert validate_experiment_record(nan_mse) is False

    # Infinite mse in the record is rejected.
    inf_mse = dict(
        rec, validation_metrics=dict(base_metrics, mse=float("inf"))
    )
    assert validate_experiment_record(inf_mse) is False

    # Non-finite mae (present selection metric) is rejected.
    nan_mae = dict(rec, validation_metrics=dict(base_metrics, mae=float("nan")))
    assert validate_experiment_record(nan_mae) is False

    # Non-finite samples count is rejected.
    nan_samples = dict(
        rec, validation_metrics=dict(base_metrics, samples=float("nan"))
    )
    assert validate_experiment_record(nan_samples) is False

    # Non-finite pearson correlation is rejected.
    inf_corr = dict(
        rec,
        validation_metrics=dict(base_metrics, pearson_corr=float("inf")),
    )
    assert validate_experiment_record(inf_corr) is False

    # Non-finite artifact candidate_metrics value is rejected even when the
    # record itself is finite (NaN != NaN makes the equality check fail).
    nan_artifact_path = tmp_path / "nan_artifact_val.json"
    nan_artifact_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "checkpoint": str(ckpt),
                    "sha256": compute_sha256(ckpt),
                },
                "baseline": {"sha256": "base_sha"},
                "candidate_metrics": dict(base_metrics, mse=float("nan")),
            }
        ),
        encoding="utf-8",
    )
    nan_artifact_rec = dict(rec, validation_decision_path=str(nan_artifact_path))
    assert validate_experiment_record(nan_artifact_rec) is False

    # 'validated' records with finite matching metrics remain reusable.
    validated_rec = dict(rec, status="validated")
    del validated_rec["eval_decision_path"]
    del validated_rec["all_gates_pass"]
    assert validate_experiment_record(validated_rec) is True


def test_string_gates_pass_not_reusable_or_selected(tmp_path: Path) -> None:
    ckpt, eval_p, val_p, hash_val = _create_valid_exp_files(
        tmp_path, "exp_str_gate", baseline_sha256="expected_base"
    )
    rec = {
        "exp_id": "exp_str_gate",
        "status": "completed",
        "candidate_checkpoint": str(ckpt),
        "eval_decision_path": str(eval_p),
        "validation_decision_path": str(val_p),
        "validation_metrics": {"mse": 0.05},
        "all_gates_pass": True,
    }

    str_gate_rec = dict(rec, all_gates_pass="false")
    assert (
        validate_experiment_record(
            str_gate_rec, expected_baseline_sha256="expected_base"
        )
        is False
    )

    str_eval_path = tmp_path / "str_eval.json"
    str_eval_path.write_text(
        json.dumps(
            {
                "all_gates_pass": "false",
                "candidate": {"checkpoint": str(ckpt), "sha256": hash_val},
                "baseline": {"sha256": "expected_base"},
            }
        ),
        encoding="utf-8",
    )
    str_eval_rec = dict(rec, eval_decision_path=str(str_eval_path))
    assert (
        validate_experiment_record(
            str_eval_rec, expected_baseline_sha256="expected_base"
        )
        is False
    )

    records_with_str_gate = [
        {
            "exp_id": "exp_str1",
            "all_gates_pass": "false",
            "metrics": {"pearson_corr": 0.9, "mae": 0.1},
        },
        {
            "exp_id": "exp_str2",
            "all_gates_pass": "true",
            "metrics": {"pearson_corr": 0.95, "mae": 0.05},
        },
    ]
    assert select_best_candidate(records_with_str_gate) is None


def test_candidate_ladder_is_bounded_and_deterministic() -> None:
    from scripts.modeling.run_active_scenic_autoresearch import ActiveTrainingConfig

    config = ActiveTrainingConfig(seed=7)
    ladder = build_candidate_ladder(config, 3)
    assert [item["exp_id"] for item in ladder] == [
        "exp_01_baseline_control",
        "exp_02_region_balanced",
        "exp_03_robust_huber_loss",
    ]


def test_command_sanitization_redacts_secrets() -> None:
    rendered = sanitize_command("train --api_key=secret-value --device=cpu")
    assert "secret-value" not in rendered
    assert "api_key=[REDACTED]" in rendered


def test_deadline_guard_raises_on_timeout() -> None:
    with pytest.raises(DeadlineExceededError):
        with deadline_guard(0.05):
            time.sleep(0.2)


def test_deadline_guard_allows_fast_execution() -> None:
    with deadline_guard(1.0):
        pass


def test_deadline_guard_raises_immediately_if_expired() -> None:
    with pytest.raises(DeadlineExceededError):
        with deadline_guard(0.0):
            pass


def test_parse_args_validates_run_name(monkeypatch: pytest.MonkeyPatch) -> None:
    test_args = [
        "run_active_scenic_autoresearch.py",
        "--run-name",
        "valid_run_1",
        "--expanded-benchmark-csv",
        "exp.csv",
        "--control-benchmark-csv",
        "ctrl.csv",
        "--control-dataset",
        "ctrl.csv",
        "--route-qa-json",
        "route.json",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = parse_args()
    assert parsed.run_name == "valid_run_1"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_active_scenic_autoresearch.py",
            "--run-name",
            "../traversal_run",
            "--expanded-benchmark-csv",
            "exp.csv",
            "--control-benchmark-csv",
            "ctrl.csv",
            "--control-dataset",
            "ctrl.csv",
            "--route-qa-json",
            "route.json",
        ],
    )
    with pytest.raises(ValueError, match="traversal"):
        parse_args()


def test_experiment_digest_changes_on_config_or_input_change() -> None:
    d1 = compute_experiment_digest(
        exp_id="exp_01",
        config={"seed": 42, "lr": 1e-4},
        handoff_sha256="h1",
        dataset_sha256="d1",
        control_dataset_sha256="control-dataset",
        expanded_benchmark_sha256="eb1",
        control_benchmark_sha256="cb1",
        route_qa_sha256="rq1",
        baseline_checkpoint_sha256="base1",
    )
    d2 = compute_experiment_digest(
        exp_id="exp_01",
        config={"seed": 43, "lr": 1e-4},
        handoff_sha256="h1",
        dataset_sha256="d1",
        control_dataset_sha256="control-dataset",
        expanded_benchmark_sha256="eb1",
        control_benchmark_sha256="cb1",
        route_qa_sha256="rq1",
        baseline_checkpoint_sha256="base1",
    )
    assert d1 != d2


def test_resume_rejects_mismatched_input_digest(tmp_path: Path) -> None:
    ckpt, eval_p, val_p, _ = _create_valid_exp_files(tmp_path, "exp_dig")
    rec = {
        "exp_id": "exp_dig",
        "status": "completed",
        "candidate_checkpoint": str(ckpt),
        "eval_decision_path": str(eval_p),
        "validation_decision_path": str(val_p),
        "validation_metrics": {"mse": 0.05},
        "all_gates_pass": True,
        "input_digest": "correct_digest_123",
    }
    path = tmp_path / "experiments.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    completed_map, _ = load_existing_experiments(
        path, expected_input_digests={"exp_dig": "correct_digest_123"}
    )
    assert "exp_dig" in completed_map

    completed_map_mismatch, _ = load_existing_experiments(
        path, expected_input_digests={"exp_dig": "different_digest_456"}
    )
    assert "exp_dig" not in completed_map_mismatch


def test_run_dir_isolation_rejects_clobbering_without_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    run_dir = tmp_path / "data" / "processed" / "modeling_autoresearch" / "isolated_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    handoff = _handoff(tmp_path / "handoff")

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "isolated_run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(FileExistsError, match="already exists with run artifacts"):
        mod.main()


def test_dry_run_never_creates_prepared_dataset_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "dry_run_test",
        "--dry-run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        mod.main()

    assert exc_info.value.code == 0

    run_dir = tmp_path / "data" / "processed" / "modeling_autoresearch" / "dry_run_test"
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "experiments.jsonl").exists()
    assert not (run_dir / "prepared_dataset.npz").exists()


def test_determine_aggregate_run_state() -> None:
    recs_completed = [{"status": "completed"}, {"status": "completed"}]
    assert determine_aggregate_run_state(recs_completed, 2) == "completed"

    recs_stopped = [{"status": "completed"}, {"status": "stopped"}]
    assert determine_aggregate_run_state(recs_stopped, 2) == "timed_out"

    assert determine_aggregate_run_state(recs_completed, 5) == "timed_out"
    assert (
        determine_aggregate_run_state(recs_completed, 2, timed_out=True) == "timed_out"
    )

    recs_failed = [{"status": "completed"}, {"status": "failed"}]
    assert determine_aggregate_run_state(recs_failed, 2) == "failed"

    recs_paused = [{"status": "completed"}, {"status": "paused"}]
    assert determine_aggregate_run_state(recs_paused, 2) == "paused"

    recs_validated = [{"status": "validated"}, {"status": "completed"}]
    assert determine_aggregate_run_state(recs_validated, 2) == "completed"
    recs_validated_paused = [{"status": "validated"}, {"status": "paused"}]
    assert determine_aggregate_run_state(recs_validated_paused, 2) == "paused"


def test_is_valid_paused_experiment(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_01"
    assert not is_valid_paused_experiment(exp_dir)

    exp_dir.mkdir(parents=True, exist_ok=True)
    resume_pt = exp_dir / "resume.pt"
    summary_json = exp_dir / "training_summary.json"

    resume_pt.write_bytes(b"dummy")
    assert not is_valid_paused_experiment(exp_dir)

    summary_json.write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    assert not is_valid_paused_experiment(exp_dir)

    summary_json.write_text(json.dumps({"state": "paused"}), encoding="utf-8")
    assert is_valid_paused_experiment(exp_dir)


def test_resume_passes_resume_false_to_fresh_later_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")
    thresh_json = tmp_path / "thresholds.json"
    thresh_json.write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / "run_resume_test"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": "run_resume_test",
        "created_at": "2026-08-04T00:00:00Z",
        "stage1_handoff_path": str(handoff),
        "stage1_handoff_sha256": compute_sha256(handoff),
        "baseline_registry_path": str(reg_file),
        "baseline_registry_sha256": compute_sha256(reg_file),
        "baseline_checkpoint_path": str(ckpt),
        "baseline_checkpoint_sha256": compute_sha256(ckpt),
        "expanded_benchmark_sha256": compute_sha256(exp_csv),
        "control_benchmark_sha256": compute_sha256(ctrl_csv),
        "control_dataset_sha256": compute_sha256(ctrl_csv),
        "route_qa_sha256": compute_sha256(route_json),
        "thresholds_sha256": compute_sha256(thresh_json),
        "dry_run": False,
        "promote_requested": False,
        "config": {
            "seed": 42,
            "device": "cpu",
            "max_experiments": 2,
            "max_steps": None,
            "max_seconds": 1800.0,
            "expanded_benchmark_csv": str(exp_csv),
            "control_benchmark_csv": str(ctrl_csv),
            "control_dataset": str(ctrl_csv),
            "route_qa_json": str(route_json),
            "thresholds_json": str(thresh_json),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    exp01_dir = run_dir / "exp_01_baseline_control"
    exp01_dir.mkdir(parents=True, exist_ok=True)
    (exp01_dir / "resume.pt").write_bytes(b"checkpoint_state")
    (exp01_dir / "training_summary.json").write_text(
        json.dumps({"state": "paused"}), encoding="utf-8"
    )

    dataset_npz = run_dir / "prepared_dataset.npz"
    dataset_npz.write_bytes(b"npz_data")
    split_csv = run_dir / "prepared_split.csv"
    split_csv.write_text("header", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "prepare_active_dataset",
        lambda h, p: {"dataset_path": p, "split_path": split_csv},
    )

    train_resume_calls: dict[str, bool] = {}

    def mock_train_active_model(*, dataset_path, split_csv, output_dir, config, resume):
        exp_name = Path(output_dir).name
        train_resume_calls[exp_name] = resume
        return {
            "state": "completed",
            "candidate_checkpoint": str(Path(output_dir) / "candidate.pt"),
        }

    monkeypatch.setattr(mod, "train_active_model", mock_train_active_model)

    def mock_evaluate_stage_two(**kwargs):
        return {
            "all_gates_pass": True,
            "expanded_human_benchmark": {
                "candidate_metrics": {"mae": 0.1, "rmse": 0.2, "pearson_corr": 0.9}
            },
        }

    monkeypatch.setattr(mod, "evaluate_stage_two", mock_evaluate_stage_two)
    monkeypatch.setattr(mod, "evaluate_candidate_validation", _make_validation_mock({}))

    thresh_json = tmp_path / "thresholds.json"
    thresh_json.write_text("{}", encoding="utf-8")

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "run_resume_test",
        "--resume",
        "--max-experiments",
        "2",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
        "--thresholds-json",
        str(thresh_json),
    ]

    monkeypatch.setattr(sys, "argv", test_args)
    mod.main()

    assert train_resume_calls.get("exp_01_baseline_control") is True
    assert train_resume_calls.get("exp_02_region_balanced") is False


def test_resume_rejects_changed_material_config_or_input_without_manifest_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    init_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "test_reject_drift",
        "--dry-run",
        "--seed",
        "42",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", init_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / "test_reject_drift"
    )
    manifest_path = run_dir / "run_manifest.json"
    original_manifest_text = manifest_path.read_text(encoding="utf-8")

    resume_changed_seed_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "test_reject_drift",
        "--resume",
        "--seed",
        "999",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", resume_changed_seed_args)

    with pytest.raises(
        ValueError, match="manifest validation failed|config.seed mismatch"
    ):
        mod.main()

    assert manifest_path.read_text(encoding="utf-8") == original_manifest_text

    exp_csv.write_text("header_modified_data", encoding="utf-8")

    resume_changed_input_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "test_reject_drift",
        "--resume",
        "--seed",
        "42",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", resume_changed_input_args)

    with pytest.raises(
        ValueError,
        match="manifest validation failed|expanded_benchmark_sha256 mismatch",
    ):
        mod.main()

    assert manifest_path.read_text(encoding="utf-8") == original_manifest_text


def test_resume_allows_varying_max_seconds_timing_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    init_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "test_timing_budget",
        "--dry-run",
        "--max-seconds",
        "1000",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", init_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / "test_timing_budget"
    )
    manifest_path = run_dir / "run_manifest.json"
    original_manifest_text = manifest_path.read_text(encoding="utf-8")

    resume_timing_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "test_timing_budget",
        "--dry-run",
        "--resume",
        "--max-seconds",
        "3600",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", resume_timing_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    assert manifest_path.read_text(encoding="utf-8") == original_manifest_text


def test_resume_requires_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path
        / "data"
        / "processed"
        / "modeling_autoresearch"
        / "missing_manifest_run"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiments.jsonl").write_text("", encoding="utf-8")

    resume_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "missing_manifest_run",
        "--resume",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", resume_args)

    with pytest.raises(FileNotFoundError, match="run manifest missing"):
        mod.main()


def test_resume_rejects_deleted_manifest_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    init_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "test_del_keys",
        "--dry-run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", init_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / "test_del_keys"
    )
    manifest_path = run_dir / "run_manifest.json"

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest_data["config"]["seed"]
    corrupted_text = json.dumps(manifest_data)
    manifest_path.write_text(corrupted_text, encoding="utf-8")

    resume_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "test_del_keys",
        "--resume",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", resume_args)

    with pytest.raises(
        ValueError,
        match="manifest validation failed|missing config key in manifest: seed",
    ):
        mod.main()

    assert manifest_path.read_text(encoding="utf-8") == corrupted_text


def test_resume_uses_immutable_manifest_baseline_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    baseline_sha = compute_sha256(ckpt)
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / "test_manifest_base"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": "test_manifest_base",
        "created_at": "2026-08-04T00:00:00Z",
        "stage1_handoff_path": str(handoff),
        "stage1_handoff_sha256": compute_sha256(handoff),
        "baseline_registry_path": str(reg_file),
        "baseline_registry_sha256": compute_sha256(reg_file),
        "baseline_checkpoint_path": str(ckpt),
        "baseline_checkpoint_sha256": baseline_sha,
        "expanded_benchmark_sha256": compute_sha256(exp_csv),
        "control_benchmark_sha256": compute_sha256(ctrl_csv),
        "route_qa_sha256": compute_sha256(route_json),
        "thresholds_sha256": None,
        "dry_run": False,
        "promote_requested": False,
        "config": {
            "seed": 42,
            "device": "cpu",
            "max_experiments": 1,
            "max_steps": None,
            "max_seconds": 1800.0,
            "expanded_benchmark_csv": str(exp_csv),
            "control_benchmark_csv": str(ctrl_csv),
            "route_qa_json": str(route_json),
            "thresholds_json": None,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    exp_dir = run_dir / "exp_01_baseline_control"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_ckpt = exp_dir / "candidate.pt"
    exp_ckpt.write_bytes(b"cand_data")
    exp_ckpt_sha = compute_sha256(exp_ckpt)

    eval_dec = exp_dir / "eval_decision.json"
    eval_dec.write_text(
        json.dumps(
            {
                "all_gates_pass": True,
                "candidate": {"checkpoint": str(exp_ckpt), "sha256": exp_ckpt_sha},
                "baseline": {"sha256": baseline_sha},
            }
        ),
        encoding="utf-8",
    )

    val_dec = exp_dir / "validation_decision.json"
    val_dec.write_text(
        json.dumps(
            {
                "candidate": {"checkpoint": str(exp_ckpt), "sha256": exp_ckpt_sha},
                "baseline": {"sha256": baseline_sha},
                "candidate_metrics": {
                    "mse": 0.05,
                    "mae": 0.1,
                    "rmse": 0.2,
                    "pearson_corr": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )

    exp_digest = compute_experiment_digest(
        exp_id="exp_01_baseline_control",
        config=build_candidate_ladder(ActiveTrainingConfig(seed=42), 1)[0]["config"],
        handoff_sha256=compute_sha256(handoff),
        dataset_sha256=compute_sha256(run_dir / "prepared_dataset.npz")
        if (run_dir / "prepared_dataset.npz").exists()
        else "pending_preparation",
        control_dataset_sha256="control-dataset",
        expanded_benchmark_sha256=compute_sha256(exp_csv),
        control_benchmark_sha256=compute_sha256(ctrl_csv),
        route_qa_sha256=compute_sha256(route_json),
        baseline_checkpoint_sha256=baseline_sha,
        thresholds_sha256=None,
    )

    exp_record = {
        "exp_id": "exp_01_baseline_control",
        "status": "completed",
        "candidate_checkpoint": str(exp_ckpt),
        "eval_decision_path": str(eval_dec),
        "validation_decision_path": str(val_dec),
        "validation_metrics": {"mse": 0.05, "mae": 0.1, "rmse": 0.2, "pearson_corr": 0.9},
        "all_gates_pass": True,
        "metrics": {"mae": 0.1, "rmse": 0.2, "pearson_corr": 0.9},
        "input_digest": exp_digest,
    }
    (run_dir / "experiments.jsonl").write_text(
        json.dumps(exp_record) + "\n", encoding="utf-8"
    )

    completed_map, _ = load_existing_experiments(
        run_dir / "experiments.jsonl",
        expected_baseline_sha256=manifest["baseline_checkpoint_sha256"],
        expected_input_digests={"exp_01_baseline_control": exp_digest},
    )
    assert "exp_01_baseline_control" in completed_map


def test_parse_args_max_experiments_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    test_args = [
        "run_active_scenic_autoresearch.py",
        "--expanded-benchmark-csv",
        "exp.csv",
        "--control-benchmark-csv",
        "ctrl.csv",
        "--control-dataset",
        "ctrl.csv",
        "--route-qa-json",
        "route.json",
        "--max-experiments",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(ValueError, match="--max-experiments must be between 1 and 5"):
        parse_args()

    test_args[10] = "6"
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(ValueError, match="--max-experiments must be between 1 and 5"):
        parse_args()

    test_args[10] = "3"
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = parse_args()
    assert parsed.max_experiments == 3


def test_resume_rejects_dry_run_or_promote_intent_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    # Create dry_run manifest
    init_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "mode_drift_test",
        "--dry-run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", init_args)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    # Attempt to resume dry_run plan into real training without --dry-run
    resume_non_dry_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "mode_drift_test",
        "--resume",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", resume_non_dry_args)
    with pytest.raises(ValueError, match="manifest validation failed|dry_run mismatch"):
        mod.main()

    # Attempt to resume with --promote when manifest had promote_requested=False
    resume_promote_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "mode_drift_test",
        "--dry-run",
        "--resume",
        "--promote",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", resume_promote_args)
    with pytest.raises(
        ValueError, match="manifest validation failed|promote_requested mismatch"
    ):
        mod.main()


def test_resume_with_completed_summary_without_eval_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text(
        "image_path,split,scenic_human_mean\n"
        + "".join(f"img{i}.jpg,val,5.0\n" for i in range(5)),
        encoding="utf-8",
    )
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")
    thresh_json = tmp_path / "thresholds.json"
    thresh_json.write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path
        / "data"
        / "processed"
        / "modeling_autoresearch"
        / "run_completed_no_eval"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": "run_completed_no_eval",
        "created_at": "2026-08-04T00:00:00Z",
        "stage1_handoff_path": str(handoff),
        "stage1_handoff_sha256": compute_sha256(handoff),
        "baseline_registry_path": str(reg_file),
        "baseline_registry_sha256": compute_sha256(reg_file),
        "baseline_checkpoint_path": str(ckpt),
        "baseline_checkpoint_sha256": compute_sha256(ckpt),
        "expanded_benchmark_sha256": compute_sha256(exp_csv),
        "control_benchmark_sha256": compute_sha256(ctrl_csv),
        "control_dataset_sha256": compute_sha256(ctrl_csv),
        "route_qa_sha256": compute_sha256(route_json),
        "thresholds_sha256": compute_sha256(thresh_json),
        "dry_run": False,
        "promote_requested": False,
        "config": {
            "seed": 42,
            "device": "cpu",
            "max_experiments": 1,
            "max_steps": None,
            "max_seconds": 1800.0,
            "expanded_benchmark_csv": str(exp_csv),
            "control_benchmark_csv": str(ctrl_csv),
            "control_dataset": str(ctrl_csv),
            "route_qa_json": str(route_json),
            "thresholds_json": str(thresh_json),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    exp01_dir = run_dir / "exp_01_baseline_control"
    exp01_dir.mkdir(parents=True, exist_ok=True)
    (exp01_dir / "candidate.pt").write_bytes(b"candidate_state")
    (exp01_dir / "training_summary.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )

    dataset_npz = run_dir / "prepared_dataset.npz"
    dataset_npz.write_bytes(b"npz_data")
    split_csv = run_dir / "prepared_split.csv"
    split_csv.write_text("header", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "prepare_active_dataset",
        lambda h, p: {"dataset_path": p, "split_path": split_csv},
    )

    train_resume_calls: dict[str, bool] = {}

    def mock_train_active_model(*, dataset_path, split_csv, output_dir, config, resume):
        exp_name = Path(output_dir).name
        train_resume_calls[exp_name] = resume
        return {
            "state": "completed",
            "candidate_checkpoint": str(Path(output_dir) / "candidate.pt"),
        }

    monkeypatch.setattr(mod, "train_active_model", mock_train_active_model)

    def mock_evaluate_stage_two(**kwargs):
        return {
            "all_gates_pass": True,
            "expanded_human_benchmark": {
                "candidate_metrics": {"mae": 0.1, "rmse": 0.2, "pearson_corr": 0.9}
            },
        }

    monkeypatch.setattr(mod, "evaluate_stage_two", mock_evaluate_stage_two)
    monkeypatch.setattr(mod, "evaluate_candidate_validation", _make_validation_mock({}))

    thresh_json = tmp_path / "thresholds.json"
    thresh_json.write_text("{}", encoding="utf-8")

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "run_completed_no_eval",
        "--resume",
        "--max-experiments",
        "1",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
        "--thresholds-json",
        str(thresh_json),
    ]

    monkeypatch.setattr(sys, "argv", test_args)
    mod.main()

    # Completed training output is reused WITHOUT retraining; only validation runs.
    assert train_resume_calls == {}
    exp_records = [
        json.loads(line)
        for line in (run_dir / "experiments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [r["status"] for r in exp_records] == ["validated", "completed"]
    assert exp_records[0]["candidate_checkpoint"] == str(exp01_dir / "candidate.pt")
    assert exp_records[0]["validation_metrics"]["mse"] == 0.5
    assert (run_dir / "final_summary.json").is_file()


def test_status_mode_fails_closed_on_missing_or_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path
        / "data"
        / "processed"
        / "modeling_autoresearch"
        / "status_missing_manifest"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiments.jsonl").write_text("", encoding="utf-8")

    status_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "status_missing_manifest",
        "--status",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", status_args)

    with pytest.raises(FileNotFoundError, match="manifest missing"):
        mod.main()

    # Write malformed manifest
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        ValueError, match="malformed or lacks baseline_checkpoint_sha256"
    ):
        mod.main()


def test_status_mode_after_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    original_ckpt = reg_dir / "baseline_orig.pt"
    original_ckpt.write_bytes(b"original_baseline_data")
    original_base_sha = compute_sha256(original_ckpt)

    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(original_ckpt)}}), encoding="utf-8"
    )

    handoff = _handoff(tmp_path / "handoff")
    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("header", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("header", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / "status_post_promo"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": "status_post_promo",
        "created_at": "2026-08-04T00:00:00Z",
        "stage1_handoff_path": str(handoff),
        "stage1_handoff_sha256": compute_sha256(handoff),
        "baseline_registry_path": str(reg_file),
        "baseline_registry_sha256": compute_sha256(reg_file),
        "baseline_checkpoint_path": str(original_ckpt),
        "baseline_checkpoint_sha256": original_base_sha,
        "expanded_benchmark_sha256": compute_sha256(exp_csv),
        "control_benchmark_sha256": compute_sha256(ctrl_csv),
        "route_qa_sha256": compute_sha256(route_json),
        "thresholds_sha256": None,
        "dry_run": False,
        "promote_requested": False,
        "config": {
            "seed": 42,
            "device": "cpu",
            "max_experiments": 1,
            "max_steps": None,
            "max_seconds": 1800.0,
            "expanded_benchmark_csv": str(exp_csv),
            "control_benchmark_csv": str(ctrl_csv),
            "route_qa_json": str(route_json),
            "thresholds_json": None,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Completed experiment recorded under original_base_sha
    exp_dir = run_dir / "exp_01_baseline_control"
    exp_dir.mkdir(parents=True, exist_ok=True)
    cand_ckpt = exp_dir / "candidate.pt"
    cand_ckpt.write_bytes(b"candidate_data")
    cand_sha = compute_sha256(cand_ckpt)

    eval_dec = exp_dir / "eval_decision.json"
    eval_dec.write_text(
        json.dumps(
            {
                "all_gates_pass": True,
                "candidate": {"checkpoint": str(cand_ckpt), "sha256": cand_sha},
                "baseline": {"sha256": original_base_sha},
            }
        ),
        encoding="utf-8",
    )

    val_dec = exp_dir / "validation_decision.json"
    val_dec.write_text(
        json.dumps(
            {
                "candidate": {"checkpoint": str(cand_ckpt), "sha256": cand_sha},
                "baseline": {"sha256": original_base_sha},
                "candidate_metrics": {
                    "mse": 0.05,
                    "mae": 0.1,
                    "rmse": 0.2,
                    "pearson_corr": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )

    exp_digest = compute_experiment_digest(
        exp_id="exp_01_baseline_control",
        config=build_candidate_ladder(ActiveTrainingConfig(seed=42), 1)[0]["config"],
        handoff_sha256=compute_sha256(handoff),
        dataset_sha256="pending_preparation",
        control_dataset_sha256="control-dataset",
        expanded_benchmark_sha256=compute_sha256(exp_csv),
        control_benchmark_sha256=compute_sha256(ctrl_csv),
        route_qa_sha256=compute_sha256(route_json),
        baseline_checkpoint_sha256=original_base_sha,
        thresholds_sha256=None,
    )

    exp_record = {
        "exp_id": "exp_01_baseline_control",
        "status": "completed",
        "candidate_checkpoint": str(cand_ckpt),
        "eval_decision_path": str(eval_dec),
        "validation_decision_path": str(val_dec),
        "validation_metrics": {"mse": 0.05, "mae": 0.1, "rmse": 0.2, "pearson_corr": 0.9},
        "all_gates_pass": True,
        "metrics": {"mae": 0.1, "rmse": 0.2, "pearson_corr": 0.9},
        "input_digest": exp_digest,
    }
    (run_dir / "experiments.jsonl").write_text(
        json.dumps(exp_record) + "\n", encoding="utf-8"
    )

    # Now update registry to point to a NEW promoted active checkpoint (simulating post-promotion state)
    new_promoted_ckpt = reg_dir / "promoted_active.pt"
    new_promoted_ckpt.write_bytes(b"promoted_active_data")
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(new_promoted_ckpt)}}), encoding="utf-8"
    )

    status_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "status_post_promo",
        "--status",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", status_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    captured = capsys.readouterr().out
    assert "Completed Experiments: 1" in captured
    assert f"Recorded Run Baseline Checkpoint SHA256: {original_base_sha}" in captured


def test_validate_handoff_preflight_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    handoff = _handoff(tmp_path / "handoff")

    def _fail_preflight(p: Path) -> dict[str, int]:
        raise ValueError("Deep preflight validation failed")

    monkeypatch.setattr(mod, "validate_handoff", _fail_preflight)

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "preflight_fail_test",
        "--dry-run",
        "--expanded-benchmark-csv",
        "exp.csv",
        "--control-benchmark-csv",
        "ctrl.csv",
        "--control-dataset",
        "ctrl.csv",
        "--route-qa-json",
        "route.json",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_thresholds_missing_fails_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    handoff = _handoff(tmp_path / "handoff")

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("split,image_path,target\n", encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("split,image_path,target\n", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    prep_called = []

    def mock_prep(*args: Any, **kwargs: Any) -> dict[str, Any]:
        prep_called.append(True)
        return {
            "dataset_path": tmp_path / "dataset.npz",
            "split_path": tmp_path / "split.csv",
        }

    monkeypatch.setattr(mod, "prepare_active_dataset", mock_prep)

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "thresh_missing_test",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(
        ValueError, match="--thresholds-json is required for non-dry execution"
    ):
        mod.main()

    assert len(prep_called) == 0


def test_count_expanded_val_samples(tmp_path: Path) -> None:
    from scripts.modeling.run_active_scenic_autoresearch import (
        count_expanded_val_samples,
    )

    exp_csv = tmp_path / "expanded_benchmark.csv"
    exp_csv.write_text(
        "split,image_path,scenic_human_mean\n"
        "val,img1.jpg,7.5\n"
        "val,img2.jpg,4.0\n"
        "val,img1.jpg,8.0\n"
        "val,img3.jpg,15.0\n"
        "val,img4.jpg,nan\n"
        "train,img5.jpg,5.0\n",
        encoding="utf-8",
    )
    count = count_expanded_val_samples(exp_csv)
    assert count == 2


def test_data_limited_mode_skips_eval_and_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    handoff = _handoff(tmp_path / "handoff")

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt = reg_dir / "baseline.pt"
    ckpt.write_bytes(b"baseline_checkpoint_data")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(
        json.dumps({"active": {"checkpoint": str(ckpt)}}), encoding="utf-8"
    )

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text(
        "split,image_path,scenic_human_mean\nval,img_val1.jpg,6.5\n", encoding="utf-8"
    )
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("split,image_path,scenic_human_mean\n", encoding="utf-8")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")

    thresh_json = tmp_path / "thresholds.json"
    thresh_json.write_text(
        json.dumps({"min_expanded_validation_samples": 5}), encoding="utf-8"
    )

    def mock_prep(h_path: Path, out_path: Path) -> dict[str, Any]:
        out_path.write_bytes(b"dataset_npz_data")
        split_csv = out_path.parent / "dataset_splits.csv"
        split_csv.write_text("split,image_path\n", encoding="utf-8")
        return {"dataset_path": out_path, "split_path": split_csv}

    monkeypatch.setattr(mod, "prepare_active_dataset", mock_prep)

    cand_ckpt_path = tmp_path / "candidate_trained.pt"
    cand_ckpt_path.write_bytes(b"cand_ckpt_data")

    def mock_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "state": "completed",
            "candidate_checkpoint": str(cand_ckpt_path),
            "candidate_checkpoint_sha256": "mock_cand_sha256",
            "metrics": {"train_loss": 0.05, "val_loss": 0.10},
        }

    eval_called = []
    promo_called = []

    def mock_eval(*args: Any, **kwargs: Any) -> dict[str, Any]:
        eval_called.append(True)
        raise AssertionError(
            "evaluate_stage_two should NEVER be called in data-limited mode!"
        )

    def mock_promo(*args: Any, **kwargs: Any) -> dict[str, Any]:
        promo_called.append(True)
        raise AssertionError(
            "promote_from_decision should NEVER be called in data-limited mode!"
        )

    def mock_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(
            "evaluate_candidate_validation should NEVER be called in data-limited mode!"
        )

    monkeypatch.setattr(mod, "train_active_model", mock_train)
    monkeypatch.setattr(mod, "evaluate_stage_two", mock_eval)
    monkeypatch.setattr(mod, "evaluate_candidate_validation", mock_validation)
    monkeypatch.setattr(mod, "promote_from_decision", mock_promo)

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--handoff",
        str(handoff),
        "--run-name",
        "data_limited_run",
        "--max-experiments",
        "2",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_csv),
        "--route-qa-json",
        str(route_json),
        "--thresholds-json",
        str(thresh_json),
        "--promote",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    mod.main()

    assert len(eval_called) == 0
    assert len(promo_called) == 0

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / "data_limited_run"
    )
    assert (run_dir / "run_manifest.json").exists()
    manifest_data = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_data.get("data_limited") is True
    assert manifest_data.get("expanded_val_support") == 1
    assert manifest_data.get("min_expanded_validation_samples") == 5

    exp_records = [
        json.loads(line)
        for line in (run_dir / "experiments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(exp_records) == 2
    for r in exp_records:
        assert r["status"] == "rejected"
        assert r["all_gates_pass"] is False
        assert r["rejection_reason"] == "insufficient_expanded_human_validation_support"
        assert "metrics" in r

    decision_data = json.loads(
        (run_dir / "promotion_decision.json").read_text(encoding="utf-8")
    )
    assert decision_data["data_limited"] is True
    assert decision_data["all_gates_pass"] is False
    assert decision_data["retained_candidate"] is None
    assert decision_data["registry_status"] == "unchanged"
    assert decision_data["observed_expanded_val_samples"] == 1
    assert decision_data["min_expanded_validation_samples"] == 5
    req_batch = decision_data["requested_annotation_batch"]
    assert req_batch["target_validation_human_rows"] >= 5
    assert req_batch["target_test_human_rows"] >= 20

    summary_data = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary_data["data_limited"] is True
    assert summary_data["retained_exp_id"] is None
    assert summary_data["promoted"] is False
    assert summary_data["registry_status"] == "unchanged"
    assert summary_data["observed_expanded_val_samples"] == 1
    assert summary_data["min_expanded_validation_samples"] == 5


def test_select_best_candidate_ignores_rejected() -> None:
    from scripts.modeling.run_active_scenic_autoresearch import select_best_candidate

    records = [
        {
            "exp_id": "exp_01",
            "status": "rejected",
            "all_gates_pass": False,
            "rejection_reason": "insufficient_expanded_human_validation_support",
            "metrics": {"pearson_corr": 0.95, "mae": 0.1},
        },
        {
            "exp_id": "exp_02",
            "status": "completed",
            "all_gates_pass": False,
            "metrics": {"pearson_corr": 0.85, "mae": 0.2},
        },
    ]
    assert select_best_candidate(records) is None


def test_supplemental_args_all_or_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.modeling.run_active_scenic_autoresearch import parse_args

    base_args = [
        "run_active_scenic_autoresearch.py",
        "--expanded-benchmark-csv",
        "exp.csv",
        "--control-benchmark-csv",
        "ctrl.csv",
        "--control-dataset",
        "ctrl.npz",
        "--route-qa-json",
        "route.json",
    ]
    # 1 provided -> ValueError
    monkeypatch.setattr(
        sys,
        "argv",
        base_args + ["--supplemental-annotations", "supp.csv"],
    )
    with pytest.raises(ValueError, match="must be provided together or all omitted"):
        parse_args()

    # 2 provided -> ValueError
    monkeypatch.setattr(
        sys,
        "argv",
        base_args
        + [
            "--supplemental-annotations",
            "supp.csv",
            "--supplemental-annotations-sha256",
            "hash1",
        ],
    )
    with pytest.raises(ValueError, match="must be provided together or all omitted"):
        parse_args()

    # 3 provided -> Success
    monkeypatch.setattr(
        sys,
        "argv",
        base_args
        + [
            "--supplemental-annotations",
            "supp.csv",
            "--supplemental-annotations-sha256",
            "hash1",
            "--supplemental-benchmark-sha256",
            "hash2",
        ],
    )
    parsed = parse_args()
    assert str(parsed.supplemental_annotations) == "supp.csv"
    assert parsed.supplemental_annotations_sha256 == "hash1"
    assert parsed.supplemental_benchmark_sha256 == "hash2"

    # 0 provided -> Success
    monkeypatch.setattr(sys, "argv", base_args)
    parsed_zero = parse_args()
    assert parsed_zero.supplemental_annotations is None
    assert parsed_zero.supplemental_annotations_sha256 is None
    assert parsed_zero.supplemental_benchmark_sha256 is None


def test_supplemental_failure_exits_before_run_dir_or_prep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    root = tmp_path / "data" / "processed" / "active_learning"
    _handoff(root / "run_01")
    monkeypatch.chdir(tmp_path)

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("split,image_path,scenic_human_mean\nval,1.jpg,5.0\n" * 5, encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("split,image_path\n", encoding="utf-8")
    ctrl_npz = tmp_path / "ctrl.npz"
    ctrl_npz.write_bytes(b"data")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")
    supp_csv = tmp_path / "supp.csv"
    supp_csv.write_text("supp_data", encoding="utf-8")

    supp_sha = compute_sha256(supp_csv)
    bench_sha = compute_sha256(exp_csv)

    def _failing_validate_supplemental(**kwargs: Any) -> dict[str, int]:
        raise ValueError("Supplemental validation schema mismatch")

    monkeypatch.setattr(mod, "validate_supplemental", _failing_validate_supplemental)

    run_name = "run_supp_fail"
    run_dir = tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--run-name",
        run_name,
        "--dry-run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_npz),
        "--route-qa-json",
        str(route_json),
        "--supplemental-annotations",
        str(supp_csv),
        "--supplemental-annotations-sha256",
        supp_sha,
        "--supplemental-benchmark-sha256",
        bench_sha,
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(ValueError, match="Supplemental validation schema mismatch"):
        mod.main()

    # Verify run directory was NOT created before failure
    assert not run_dir.exists()


def test_supplemental_manifest_and_digest_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    # Test compute_experiment_digest changes when supplemental hashes change
    d1 = compute_experiment_digest(
        exp_id="exp_01",
        config={"seed": 42},
        handoff_sha256="h",
        dataset_sha256="d",
        control_dataset_sha256="cd",
        expanded_benchmark_sha256="eb",
        control_benchmark_sha256="cb",
        route_qa_sha256="rq",
        baseline_checkpoint_sha256="bc",
        supplemental_annotations_sha256="supp_hash1",
        supplemental_benchmark_sha256="bench_hash1",
    )
    d2 = compute_experiment_digest(
        exp_id="exp_01",
        config={"seed": 42},
        handoff_sha256="h",
        dataset_sha256="d",
        control_dataset_sha256="cd",
        expanded_benchmark_sha256="eb",
        control_benchmark_sha256="cb",
        route_qa_sha256="rq",
        baseline_checkpoint_sha256="bc",
        supplemental_annotations_sha256="supp_hash2",
        supplemental_benchmark_sha256="bench_hash1",
    )
    assert d1 != d2

    # Test manifest binding during dry-run
    root = tmp_path / "data" / "processed" / "active_learning"
    _handoff(root / "run_01")
    monkeypatch.chdir(tmp_path)

    # Setup dummy baseline model registry & checkpoint
    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = reg_dir / "base.pt"
    ckpt_file.write_bytes(b"baseline")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(json.dumps({"active": {"checkpoint": str(ckpt_file)}}), encoding="utf-8")

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("split,image_path,scenic_human_mean\nval,1.jpg,5.0\n" * 5, encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("split,image_path\n", encoding="utf-8")
    ctrl_npz = tmp_path / "ctrl.npz"
    ctrl_npz.write_bytes(b"data")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")
    supp_csv = tmp_path / "supp.csv"
    supp_csv.write_text("supp_data", encoding="utf-8")

    supp_sha = compute_sha256(supp_csv)
    bench_sha = compute_sha256(exp_csv)

    metrics_mock = {"row_count": 20, "val_count": 5, "test_count": 15, "skipped_count": 0}
    monkeypatch.setattr(mod, "validate_supplemental", lambda **kwargs: metrics_mock)

    run_name = "run_supp_manifest"
    run_dir = tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--run-name",
        run_name,
        "--dry-run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_npz),
        "--route-qa-json",
        str(route_json),
        "--supplemental-annotations",
        str(supp_csv),
        "--supplemental-annotations-sha256",
        supp_sha,
        "--supplemental-benchmark-sha256",
        bench_sha,
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    manifest_path = run_dir / "run_manifest.json"
    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["supplemental_annotations_path"] == str(supp_csv)
    assert manifest_data["supplemental_annotations_sha256"] == supp_sha
    assert manifest_data["supplemental_benchmark_sha256"] == bench_sha
    assert manifest_data["supplemental_metrics"] == metrics_mock
    assert manifest_data["config"]["supplemental_annotations"] == str(supp_csv)
    assert manifest_data["config"]["supplemental_annotations_sha256"] == supp_sha
    assert manifest_data["config"]["supplemental_benchmark_sha256"] == bench_sha
    assert manifest_data["config"]["supplemental_metrics"] == metrics_mock


def test_supplemental_stale_resume_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    root = tmp_path / "data" / "processed" / "active_learning"
    _handoff(root / "run_01")
    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = reg_dir / "base.pt"
    ckpt_file.write_bytes(b"baseline")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(json.dumps({"active": {"checkpoint": str(ckpt_file)}}), encoding="utf-8")

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("split,image_path,scenic_human_mean\nval,1.jpg,5.0\n" * 5, encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("split,image_path\n", encoding="utf-8")
    ctrl_npz = tmp_path / "ctrl.npz"
    ctrl_npz.write_bytes(b"data")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")
    supp_csv = tmp_path / "supp.csv"
    supp_csv.write_text("supp_data", encoding="utf-8")

    supp_sha = compute_sha256(supp_csv)
    bench_sha = compute_sha256(exp_csv)

    metrics_mock = {"row_count": 20, "val_count": 5, "test_count": 15, "skipped_count": 0}
    monkeypatch.setattr(mod, "validate_supplemental", lambda **kwargs: metrics_mock)

    run_name = "run_supp_resume"

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--run-name",
        run_name,
        "--dry-run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_npz),
        "--route-qa-json",
        str(route_json),
        "--supplemental-annotations",
        str(supp_csv),
        "--supplemental-annotations-sha256",
        supp_sha,
        "--supplemental-benchmark-sha256",
        bench_sha,
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Initial dry run creates valid manifest
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    # Resume with changed supplemental_annotations_sha256 -> ValueError
    resume_args = [
        "run_active_scenic_autoresearch.py",
        "--run-name",
        run_name,
        "--dry-run",
        "--resume",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_npz),
        "--route-qa-json",
        str(route_json),
        "--supplemental-annotations",
        str(supp_csv),
        "--supplemental-annotations-sha256",
        "changed_hash",
        "--supplemental-benchmark-sha256",
        bench_sha,
    ]
    monkeypatch.setattr(sys, "argv", resume_args)
    with pytest.raises(ValueError, match="Run manifest validation failed on resume"):
        mod.main()

    # Status mode with changed supplemental_annotations_sha256 -> ValueError
    status_args = [a for a in resume_args if a != "--resume"] + ["--status"]
    monkeypatch.setattr(sys, "argv", status_args)
    with pytest.raises(ValueError, match="Run manifest validation failed on resume"):
        mod.main()


def test_supplemental_success_metrics_emitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    root = tmp_path / "data" / "processed" / "active_learning"
    _handoff(root / "run_01")
    monkeypatch.chdir(tmp_path)

    reg_dir = tmp_path / "data" / "processed" / "regression"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = reg_dir / "base.pt"
    ckpt_file.write_bytes(b"baseline")
    reg_file = reg_dir / "model_registry.json"
    reg_file.write_text(json.dumps({"active": {"checkpoint": str(ckpt_file)}}), encoding="utf-8")

    exp_csv = tmp_path / "exp.csv"
    exp_csv.write_text("split,image_path,scenic_human_mean\nval,1.jpg,5.0\n" * 5, encoding="utf-8")
    ctrl_csv = tmp_path / "ctrl.csv"
    ctrl_csv.write_text("split,image_path\n", encoding="utf-8")
    ctrl_npz = tmp_path / "ctrl.npz"
    ctrl_npz.write_bytes(b"data")
    route_json = tmp_path / "route.json"
    route_json.write_text("{}", encoding="utf-8")
    supp_csv = tmp_path / "supp.csv"
    supp_csv.write_text("supp_data", encoding="utf-8")

    supp_sha = compute_sha256(supp_csv)
    bench_sha = compute_sha256(exp_csv)

    metrics_mock = {
        "supplemental_rows": 15,
        "supplemental_val_count": 4,
        "supplemental_test_count": 11,
        "supplemental_skipped_count": 0,
    }
    monkeypatch.setattr(mod, "validate_supplemental", lambda **kwargs: metrics_mock)

    run_name = "run_supp_metrics"

    test_args = [
        "run_active_scenic_autoresearch.py",
        "--run-name",
        run_name,
        "--dry-run",
        "--expanded-benchmark-csv",
        str(exp_csv),
        "--control-benchmark-csv",
        str(ctrl_csv),
        "--control-dataset",
        str(ctrl_npz),
        "--route-qa-json",
        str(route_json),
        "--supplemental-annotations",
        str(supp_csv),
        "--supplemental-annotations-sha256",
        supp_sha,
        "--supplemental-benchmark-sha256",
        bench_sha,
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "METRIC supplemental_rows=15" in out
    assert "METRIC supplemental_val_count=4" in out
    assert "METRIC supplemental_test_count=11" in out
    assert "METRIC supplemental_skipped_count=0" in out


def test_select_validation_finalist_lowest_mse_with_ladder_tie_break() -> None:
    ladder_ids = [
        "exp_01_baseline_control",
        "exp_02_region_balanced",
        "exp_03_robust_huber_loss",
    ]
    records = [
        {
            "exp_id": "exp_03_robust_huber_loss",
            "status": "validated",
            "validation_metrics": {"mse": 0.3},
        },
        {
            "exp_id": "exp_01_baseline_control",
            "status": "validated",
            "validation_metrics": {"mse": 0.5},
        },
        {
            "exp_id": "exp_02_region_balanced",
            "status": "validated",
            "validation_metrics": {"mse": 0.4},
        },
    ]
    best = select_validation_finalist(records, ladder_ids)
    assert best is not None
    assert best["exp_id"] == "exp_03_robust_huber_loss"

    # Deterministic ladder-order tie-break on equal MSE
    tie_records = [
        {
            "exp_id": "exp_02_region_balanced",
            "status": "validated",
            "validation_metrics": {"mse": 0.4},
        },
        {
            "exp_id": "exp_01_baseline_control",
            "status": "validated",
            "validation_metrics": {"mse": 0.4},
        },
        {
            "exp_id": "exp_03_robust_huber_loss",
            "status": "validated",
            "validation_metrics": {"mse": 0.5},
        },
    ]
    best_tie = select_validation_finalist(tie_records, ladder_ids)
    assert best_tie is not None
    assert best_tie["exp_id"] == "exp_01_baseline_control"

    # Any nonterminal candidate blocks selection entirely
    paused_records = tie_records + [
        {"exp_id": "exp_03_robust_huber_loss", "status": "paused"}
    ]
    assert select_validation_finalist(paused_records, ladder_ids) is None
    stopped_records = tie_records + [
        {"exp_id": "exp_03_robust_huber_loss", "status": "stopped"}
    ]
    assert select_validation_finalist(stopped_records, ladder_ids) is None

    # Missing candidate blocks selection
    two_ladder = ladder_ids[:2]
    two_records = [
        {
            "exp_id": "exp_02_region_balanced",
            "status": "validated",
            "validation_metrics": {"mse": 0.4},
        },
        {
            "exp_id": "exp_01_baseline_control",
            "status": "validated",
            "validation_metrics": {"mse": 0.4},
        },
    ]
    assert select_validation_finalist(two_records, ladder_ids) is None

    # Newest record wins for an exp_id (completed full-eval supersedes validated)
    supersede = [
        {
            "exp_id": "exp_01_baseline_control",
            "status": "validated",
            "validation_metrics": {"mse": 0.9},
        },
        {
            "exp_id": "exp_01_baseline_control",
            "status": "completed",
            "validation_metrics": {"mse": 0.1},
            "eval_decision_path": "x",
            "all_gates_pass": True,
        },
        {
            "exp_id": "exp_02_region_balanced",
            "status": "validated",
            "validation_metrics": {"mse": 0.2},
        },
    ]
    best_sup = select_validation_finalist(supersede, two_ladder)
    assert best_sup is not None
    assert best_sup["exp_id"] == "exp_01_baseline_control"


def test_fresh_run_validates_all_candidates_and_full_evaluates_winner_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    env = _build_run_env(tmp_path)
    run_name = "run_full_flow"
    monkeypatch.setattr(mod, "prepare_active_dataset", _mock_prepare)

    train_calls: dict[str, Any] = {}
    monkeypatch.setattr(mod, "train_active_model", _make_train_mock(train_calls))

    mse_by_exp = {
        "exp_01_baseline_control": 0.40,
        "exp_02_region_balanced": 0.30,
        "exp_03_robust_huber_loss": 0.20,
        "exp_04_fine_learning_rate": 0.35,
        "exp_05_extended_epochs": 0.25,
    }
    monkeypatch.setattr(
        mod, "evaluate_candidate_validation", _make_validation_mock(mse_by_exp)
    )
    stage_two_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod, "evaluate_stage_two", _make_stage_two_mock(stage_two_calls)
    )

    monkeypatch.setattr(sys, "argv", _run_args(env, run_name, max_experiments=5))
    mod.main()

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name
    )
    exp_records = [
        json.loads(line)
        for line in (run_dir / "experiments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(exp_records) == 6
    validated = [r for r in exp_records if r["status"] == "validated"]
    assert len(validated) == 5
    for r in validated:
        assert r["candidate_checkpoint_sha256"]
        assert isinstance(r["validation_metrics"]["mse"], float)
        assert r["validation_decision_path"]
        assert Path(r["validation_decision_path"]).is_file()
        assert r["input_digest"]
        assert "validated_at" in r
        assert "eval_decision_path" not in r

    # Exactly one full stage-two evaluation, for the lowest-validation-MSE winner.
    assert len(stage_two_calls) == 1
    assert (
        Path(stage_two_calls[0]["candidate_checkpoint"]).parent.name
        == "exp_03_robust_huber_loss"
    )
    completed = [r for r in exp_records if r["status"] == "completed"]
    assert len(completed) == 1
    assert completed[0]["exp_id"] == "exp_03_robust_huber_loss"
    assert completed[0]["eval_decision_path"]
    assert completed[0]["metrics"]
    assert completed[0]["all_gates_pass"] is True
    assert completed[0]["validation_decision_path"]

    summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["run_state"] == "completed"
    assert summary["selected_finalist"]["exp_id"] == "exp_03_robust_huber_loss"
    assert summary["selected_finalist"]["validation_mse"] == 0.20
    assert summary["selected_finalist"]["full_evaluation"]["all_gates_pass"] is True
    assert summary["retained_exp_id"] == "exp_03_robust_huber_loss"
    assert summary["all_gates_pass"] is True
    assert summary["pending_final_evaluation"] is False


def test_selection_based_only_on_validation_mse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    env = _build_run_env(tmp_path)
    run_name = "run_val_selection"
    monkeypatch.setattr(mod, "prepare_active_dataset", _mock_prepare)

    train_calls: dict[str, Any] = {}
    monkeypatch.setattr(mod, "train_active_model", _make_train_mock(train_calls))

    # exp_01 wins validation (lowest MSE); exp_02 is worse on validation.
    mse_by_exp = {
        "exp_01_baseline_control": 0.10,
        "exp_02_region_balanced": 0.60,
    }
    monkeypatch.setattr(
        mod, "evaluate_candidate_validation", _make_validation_mock(mse_by_exp)
    )
    # The winner's full evaluation FAILS all compound gates. Selection is purely
    # validation-based, so gates never re-rank: only the val-winner is tested.
    stage_two_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod,
        "evaluate_stage_two",
        _make_stage_two_mock(stage_two_calls, all_gates_pass=False),
    )

    monkeypatch.setattr(sys, "argv", _run_args(env, run_name, max_experiments=2))
    mod.main()

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name
    )
    assert len(stage_two_calls) == 1
    assert (
        Path(stage_two_calls[0]["candidate_checkpoint"]).parent.name
        == "exp_01_baseline_control"
    )
    summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["selected_finalist"]["exp_id"] == "exp_01_baseline_control"
    assert summary["selected_finalist"]["validation_mse"] == 0.10
    assert summary["all_gates_pass"] is False
    assert summary["retained_exp_id"] is None
    assert summary["promoted"] is False
    assert summary["total_experiments"] == 2
    assert summary["rejection_reason"] == "selected_finalist_failed_compound_gates"
    decision = json.loads(
        (run_dir / "promotion_decision.json").read_text(encoding="utf-8")
    )
    assert decision["rejection_reason"] == "selected_finalist_failed_compound_gates"


def test_no_full_evaluation_while_any_candidate_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    env = _build_run_env(tmp_path)
    run_name = "run_paused_candidate"
    monkeypatch.setattr(mod, "prepare_active_dataset", _mock_prepare)

    train_calls: dict[str, Any] = {}

    def mock_train(*, dataset_path, split_csv, output_dir, config, resume):
        exp_dir = Path(output_dir)
        exp_name = exp_dir.name
        train_calls[exp_name] = True
        if exp_name == "exp_01_baseline_control":
            (exp_dir / "training_summary.json").write_text(
                json.dumps({"state": "paused"}), encoding="utf-8"
            )
            (exp_dir / "resume.pt").write_bytes(b"resume_state")
            return {"state": "paused"}
        ckpt = exp_dir / "candidate.pt"
        ckpt.write_bytes(b"candidate_data")
        (exp_dir / "training_summary.json").write_text(
            json.dumps({"state": "completed"}), encoding="utf-8"
        )
        return {"state": "completed", "candidate_checkpoint": str(ckpt)}

    monkeypatch.setattr(mod, "train_active_model", mock_train)

    validation_calls: list[str] = []
    base_validation = _make_validation_mock({})

    def tracking_validation(**kwargs: Any) -> dict[str, Any]:
        validation_calls.append(str(kwargs["candidate_checkpoint"]))
        return base_validation(**kwargs)

    monkeypatch.setattr(mod, "evaluate_candidate_validation", tracking_validation)
    stage_two_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod, "evaluate_stage_two", _make_stage_two_mock(stage_two_calls)
    )

    monkeypatch.setattr(sys, "argv", _run_args(env, run_name, max_experiments=2))
    mod.main()

    # The paused candidate blocks selection: zero full evaluations.
    assert len(stage_two_calls) == 0
    assert len(validation_calls) == 1

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name
    )
    exp_records = [
        json.loads(line)
        for line in (run_dir / "experiments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_id = {r["exp_id"]: r for r in exp_records}
    assert by_id["exp_01_baseline_control"]["status"] == "paused"
    assert by_id["exp_02_region_balanced"]["status"] == "validated"

    summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["run_state"] == "paused"
    assert summary["selected_finalist"] is None
    assert summary["all_gates_pass"] is False
    assert summary["retained_exp_id"] is None


def test_legacy_full_eval_without_validation_quarantines_run_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    env = _build_run_env(tmp_path)
    run_name = "run_legacy_resume"
    monkeypatch.setattr(mod, "prepare_active_dataset", _mock_prepare)

    train_calls: dict[str, Any] = {}
    monkeypatch.setattr(mod, "train_active_model", _make_train_mock(train_calls))
    validation_calls: list[str] = []
    base_validation = _make_validation_mock(
        {
            "exp_01_baseline_control": 0.2,
            "exp_02_region_balanced": 0.4,
        }
    )

    def tracking_validation(**kwargs: Any) -> dict[str, Any]:
        validation_calls.append(str(kwargs["candidate_checkpoint"]))
        return base_validation(**kwargs)

    monkeypatch.setattr(mod, "evaluate_candidate_validation", tracking_validation)
    stage_two_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod, "evaluate_stage_two", _make_stage_two_mock(stage_two_calls)
    )

    run_args = _run_args(env, run_name, max_experiments=2)
    monkeypatch.setattr(sys, "argv", run_args)
    mod.main()

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name
    )
    lines = [
        json.loads(line)
        for line in (run_dir / "experiments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    validated_by_id = {r["exp_id"]: r for r in lines if r["status"] == "validated"}
    assert len(validation_calls) == 2
    assert len(stage_two_calls) == 1

    # Simulate the historical exposure artifact: completed records with full
    # stage-two evaluation (eval_decision_path) but no validation evidence
    # (old-style full-eval-only records that touched the held-out test set
    # before finalist selection). Both eval decision files exist on disk.
    for exp_id in ["exp_01_baseline_control", "exp_02_region_balanced"]:
        eval_path = run_dir / exp_id / "eval_decision.json"
        if not eval_path.exists():
            eval_path.write_text(
                json.dumps(
                    {
                        "all_gates_pass": True,
                        "candidate": {
                            "checkpoint": str(
                                run_dir / exp_id / "candidate.pt"
                            ),
                            "sha256": compute_sha256(
                                run_dir / exp_id / "candidate.pt"
                            ),
                        },
                        "baseline": {"sha256": "base_sha"},
                    }
                ),
                encoding="utf-8",
            )
    legacy_records = []
    for exp_id in ["exp_01_baseline_control", "exp_02_region_balanced"]:
        old = validated_by_id[exp_id]
        legacy_records.append(
            {
                "exp_id": exp_id,
                "hypothesis": old.get("hypothesis"),
                "status": "completed",
                "candidate_checkpoint": old["candidate_checkpoint"],
                "eval_decision_path": str(run_dir / exp_id / "eval_decision.json"),
                "all_gates_pass": True,
                "metrics": {"mae": 0.1, "rmse": 0.2, "pearson_corr": 0.9},
                "input_digest": old["input_digest"],
            }
        )
    (run_dir / "experiments.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in legacy_records), encoding="utf-8"
    )

    # Legacy records lacking validation evidence never pass the validator and
    # are detected as held-out test exposure.
    for rec in legacy_records:
        assert validate_experiment_record(rec) is False
    exposed = detect_heldout_test_exposure(legacy_records)
    assert len(exposed) == 2
    assert [r["exp_id"] for r in exposed] == [
        "exp_01_baseline_control",
        "exp_02_region_balanced",
    ]

    # Resume a quarantined run: nothing retrains, nothing re-validates,
    # nothing re-runs a full evaluation, and nothing is promoted.
    validation_calls.clear()
    stage_two_calls.clear()
    monkeypatch.setattr(sys, "argv", run_args + ["--resume"])
    mod.main()

    assert len(train_calls) == 2  # run-1 training only; no retraining on resume
    assert len(validation_calls) == 0  # no revalidation on quarantined resume
    assert len(stage_two_calls) == 0  # no full evaluation on quarantined resume

    # Historical exposure records are preserved, not deleted.
    final_lines = [
        json.loads(line)
        for line in (run_dir / "experiments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(final_lines) == len(legacy_records)
    assert detect_heldout_test_exposure(final_lines) == exposed

    summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["run_state"] == "rejected"
    assert summary["rejection_reason"] == EXPOSURE_REJECTION_REASON
    assert summary["rejection_reason"] == (
        "heldout_test_exposure_before_finalist_selection"
    )
    assert summary["all_gates_pass"] is False
    assert summary["promotion_evidence_valid"] is False
    assert summary["selected_finalist"] is None
    assert summary["retained_exp_id"] is None
    assert summary["retained_candidate"] is None
    assert summary["promoted"] is False
    assert summary["pending_final_evaluation"] is False
    assert summary["registry_status"] == "unchanged"
    exposure = summary["heldout_test_exposure"]
    assert exposure["detected"] is True
    assert exposure["count"] == 2
    assert exposure["exp_ids"] == [
        "exp_01_baseline_control",
        "exp_02_region_balanced",
    ]
    req_batch = summary["requested_annotation_batch"]
    assert req_batch["min_fresh_test_human_rows"] >= EXPOSURE_MIN_FRESH_TEST_ROWS
    assert req_batch["target_test_human_rows"] >= EXPOSURE_MIN_FRESH_TEST_ROWS
    assert (
        req_batch["retained_validation_rows"] == FIXED_VALIDATION_SELECTION_ROWS
    )
    assert req_batch["validation_selection_unchanged"] is True
    assert req_batch["overlap_constraints"] == {
        "current_run_test_rows": 0,
        "control_rows": 0,
    }
    assert "geographically" in req_batch["sampling_strategy"]
    assert "fresh" in req_batch["sampling_strategy"]

    decision = json.loads(
        (run_dir / "promotion_decision.json").read_text(encoding="utf-8")
    )
    assert decision["all_gates_pass"] is False
    assert decision["promotion_evidence_valid"] is False
    assert decision["rejection_reason"] == EXPOSURE_REJECTION_REASON
    assert decision["selected_finalist_exp_id"] is None
    assert decision["validation_mse"] is None
    assert decision["retained_candidate"] is None
    assert decision["registry_status"] == "unchanged"
    assert decision["heldout_test_exposure"]["detected"] is True
    assert decision["heldout_test_exposure"]["count"] == 2
    assert decision["heldout_test_exposure"]["exp_ids"] == [
        "exp_01_baseline_control",
        "exp_02_region_balanced",
    ]
    assert (
        decision["requested_annotation_batch"]["min_fresh_test_human_rows"]
        >= EXPOSURE_MIN_FRESH_TEST_ROWS
    )
    assert (
        decision["requested_annotation_batch"]["retained_validation_rows"]
        == FIXED_VALIDATION_SELECTION_ROWS
    )


def test_resume_reuses_full_evidence_without_rerunning_validation_or_full_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    env = _build_run_env(tmp_path)
    run_name = "run_resume_reuse"
    monkeypatch.setattr(mod, "prepare_active_dataset", _mock_prepare)

    train_calls: dict[str, Any] = {}
    monkeypatch.setattr(mod, "train_active_model", _make_train_mock(train_calls))
    validation_calls: list[str] = []
    base_validation = _make_validation_mock(
        {
            "exp_01_baseline_control": 0.2,
            "exp_02_region_balanced": 0.4,
        }
    )

    def tracking_validation(**kwargs: Any) -> dict[str, Any]:
        validation_calls.append(str(kwargs["candidate_checkpoint"]))
        return base_validation(**kwargs)

    monkeypatch.setattr(mod, "evaluate_candidate_validation", tracking_validation)
    stage_two_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod, "evaluate_stage_two", _make_stage_two_mock(stage_two_calls)
    )

    run_args = _run_args(env, run_name, max_experiments=2)
    monkeypatch.setattr(sys, "argv", run_args)
    mod.main()

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name
    )
    jsonl_path = run_dir / "experiments.jsonl"
    first_run_lines = [
        line
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(first_run_lines) == 3  # 2 validated + 1 completed finalist
    assert len(train_calls) == 2
    assert len(validation_calls) == 2
    assert len(stage_two_calls) == 1

    # Resume a fully-completed run: nothing retrains, nothing re-validates,
    # nothing re-runs the full evaluation.
    monkeypatch.setattr(sys, "argv", run_args + ["--resume"])
    mod.main()

    assert len(train_calls) == 2
    assert len(validation_calls) == 2
    assert len(stage_two_calls) == 1
    assert (
        jsonl_path.read_text(encoding="utf-8").splitlines() == first_run_lines
    )

    summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["run_state"] == "completed"
    assert summary["selected_finalist"]["exp_id"] == "exp_01_baseline_control"
    assert summary["selected_finalist"]["full_evaluation"]["all_gates_pass"] is True
    assert summary["all_gates_pass"] is True
    assert summary["retained_exp_id"] == "exp_01_baseline_control"
    assert summary["pending_final_evaluation"] is False


def test_exposure_annotation_request_contract() -> None:
    req = build_exposure_annotation_request(
        observed_test_samples=68, observed_val_samples=64
    )
    # At least 20 fresh, untouched test rows are requested.
    assert req["min_fresh_test_human_rows"] >= 20
    assert req["target_test_human_rows"] >= 20
    assert req["needed_fresh_test_human_rows"] >= 20
    # Exposed rows are never counted as satisfying the request.
    assert req["observed_expanded_test_samples"] == 68
    # Zero overlap against current-run test rows and control rows.
    assert req["overlap_constraints"] == {
        "current_run_test_rows": 0,
        "control_rows": 0,
    }
    assert "geographically" in req["sampling_strategy"]
    assert "fresh" in req["sampling_strategy"]
    assert "no_current_or_control_overlap" in req["sampling_strategy"]
    # The fixed 64-row validation selection is retained unchanged.
    assert req["retained_validation_rows"] == 64
    assert req["validation_selection_unchanged"] is True


def test_clean_run_resume_remains_promotion_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.modeling import run_active_scenic_autoresearch as mod

    monkeypatch.chdir(tmp_path)
    env = _build_run_env(tmp_path)
    run_name = "run_clean_promo"
    monkeypatch.setattr(mod, "prepare_active_dataset", _mock_prepare)

    train_calls: dict[str, Any] = {}
    monkeypatch.setattr(mod, "train_active_model", _make_train_mock(train_calls))
    validation_calls: list[str] = []
    base_validation = _make_validation_mock(
        {
            "exp_01_baseline_control": 0.2,
            "exp_02_region_balanced": 0.4,
        }
    )

    def tracking_validation(**kwargs: Any) -> dict[str, Any]:
        validation_calls.append(str(kwargs["candidate_checkpoint"]))
        return base_validation(**kwargs)

    monkeypatch.setattr(mod, "evaluate_candidate_validation", tracking_validation)
    stage_two_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod, "evaluate_stage_two", _make_stage_two_mock(stage_two_calls)
    )

    run_args = _run_args(env, run_name, max_experiments=2)
    monkeypatch.setattr(sys, "argv", run_args)
    mod.main()

    run_dir = (
        tmp_path / "data" / "processed" / "modeling_autoresearch" / run_name
    )
    jsonl_path = run_dir / "experiments.jsonl"
    first_run_lines = [
        line
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Clean run: every candidate has validation evidence and exactly one
    # validation-selected full record (the finalist) exists.
    assert len(first_run_lines) == 3
    assert len(validation_calls) == 2
    assert len(stage_two_calls) == 1
    all_records = [json.loads(line) for line in first_run_lines]
    assert detect_heldout_test_exposure(all_records) == []
    assert all(has_validation_evidence(r) for r in all_records)

    # Resume of the clean run stays promotion-eligible: evidence valid, no
    # exposure, retained candidate present, nothing retrains or re-evaluates.
    monkeypatch.setattr(sys, "argv", run_args + ["--resume"])
    mod.main()

    assert len(train_calls) == 2
    assert len(validation_calls) == 2
    assert len(stage_two_calls) == 1
    assert (
        jsonl_path.read_text(encoding="utf-8").splitlines() == first_run_lines
    )

    summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["run_state"] == "completed"
    assert summary["promotion_evidence_valid"] is True
    assert summary["heldout_test_exposure"] == {
        "detected": False,
        "count": 0,
        "exp_ids": [],
    }
    assert summary["all_gates_pass"] is True
    assert summary["retained_exp_id"] == "exp_01_baseline_control"
    assert summary["retained_candidate"] is not None
    assert summary["retained_candidate"]["exp_id"] == "exp_01_baseline_control"
    assert summary["rejection_reason"] is None
    assert summary["requested_annotation_batch"] is None

    decision = json.loads(
        (run_dir / "promotion_decision.json").read_text(encoding="utf-8")
    )
    assert decision["promotion_evidence_valid"] is True
    assert decision["heldout_test_exposure"]["detected"] is False
    assert decision["all_gates_pass"] is True
    assert decision["retained_candidate"] is not None
    assert decision["registry_status"] == "unchanged"  # --promote not requested
