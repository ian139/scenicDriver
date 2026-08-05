from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.scenic_scorer.active_evaluation import (
    promote_from_decision,
    rollback_registry,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_rollback_is_hash_guarded(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "active": {"checkpoint": "new.pt"},
        "history": [{"checkpoint": "old.pt"}],
    }, indent=2), encoding="utf-8")
    before = registry.read_bytes()
    with pytest.raises(ValueError):
        rollback_registry(registry, 0, "0" * 64)
    assert registry.read_bytes() == before
