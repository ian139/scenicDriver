"""Focused tests for Stage-Two autoresearch orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.modeling.run_active_scenic_autoresearch import (
    build_candidate_ladder,
    load_existing_experiments,
    resolve_stage_one_handoff,
    sanitize_command,
    validate_handoff_content,
)


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


def test_implicit_handoff_rejects_ambiguous_ready_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "data" / "processed" / "active_learning"
    _handoff(root / "a")
    _handoff(root / "b")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        resolve_stage_one_handoff()


def test_resume_loads_only_completed_records(tmp_path: Path) -> None:
    records = [
        {"exp_id": "exp_01", "status": "completed"},
        {"exp_id": "exp_02", "status": "failed"},
        {"exp_id": "exp_03", "status": "retained"},
    ]
    path = tmp_path / "experiments.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    completed, all_records = load_existing_experiments(path)
    assert set(completed) == {"exp_01", "exp_03"}
    assert len(all_records) == 3


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
