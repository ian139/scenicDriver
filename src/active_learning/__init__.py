"""Deterministic active-learning selection and stage-one handoff helpers."""

from .finalize import finalize_stage1
from .scoring import (
    CANDIDATE_POOL_COLUMNS,
    CANONICAL_TILE_COLUMNS,
    SCORING_SCHEMA_VERSION,
    ScoringDependencies,
    LoadedScoringModels,
    load_scoring_models,
    normalized_class_entropy,
    resolve_active_regression_checkpoint,
    run_active_learning_scoring,
    score_active_learning_pool,
    score_manifest,
    score_tile_manifest,
)
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
    "CANDIDATE_POOL_COLUMNS",
    "CANONICAL_TILE_COLUMNS",
    "LoadedScoringModels",
    "SCORING_SCHEMA_VERSION",
    "ScoringDependencies",
    "SelectionArtifacts",
    "SelectionConfig",
    "audit_geographic_leakage",
    "build_geographic_splits",
    "finalize_stage1",
    "load_candidate_table",
    "load_scoring_models",
    "normalized_class_entropy",
    "resolve_active_regression_checkpoint",
    "run_active_learning_scoring",
    "run_selection",
    "score_active_learning_pool",
    "score_manifest",
    "score_tile_manifest",
    "select_active_learning",
    "select_candidates",
]
