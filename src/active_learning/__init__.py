"""Deterministic active-learning selection and stage-one handoff helpers."""

from .finalize import finalize_stage1
from .selection import (
    SelectionArtifacts,
    SelectionConfig,
    audit_geographic_leakage,
    build_geographic_splits,
    load_candidate_table,
    run_selection,
    select_active_learning,
    select_candidates,
)

__all__ = [
    "SelectionArtifacts",
    "SelectionConfig",
    "audit_geographic_leakage",
    "build_geographic_splits",
    "finalize_stage1",
    "load_candidate_table",
    "run_selection",
    "select_active_learning",
    "select_candidates",
]
