#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"

export LANG=C
export LC_ALL=C
export TZ=UTC
export PYTHONHASHSEED=0
export UV_OFFLINE=1
export UV_FROZEN=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

HANDOFF="data/processed/active_learning/run_v1_expanded_20260805/stage1_handoff.json"
EXPECTED_HANDOFF_SHA256="52dfe63b6064a3906a3b5ddb64c7d0e2c3664513ef42ededef5c41d5e0c078c1"
SUPPLEMENTAL_ANNOTATIONS="data/processed/active_learning/run_v2_annotation_expansion_20260807/annotations.csv"
SUPPLEMENTAL_ANNOTATIONS_SHA256="adb25c1dc50c7b4bf1a6743a233ba3681511fed31da26f3c33334199d5cd0b1d"
SUPPLEMENTAL_BENCHMARK="data/processed/active_learning/run_v2_annotation_expansion_20260807/benchmark_split.csv"
SUPPLEMENTAL_BENCHMARK_SHA256="b157013d15a95c0d57c3baa60b5ab17a721ecbbdc05ed944610b74a66fed795b"
CONTROL_BENCHMARK="data/processed/regression/masswhites_human_benchmark_v4/benchmark_split.csv"
CONTROL_DATASET="data/processed/regression/features_masswhites_z14_mixed5000_v5_h4.npz"
ROUTE_QA="data/processed/modeling_autoresearch/prompt_two_20260807/route_qa.json"
THRESHOLDS="data/processed/modeling_autoresearch/prompt_two_post_annotation_20260809/thresholds.json"
RUN_NAME="prompt_two_supplemental_20260809"

actual_handoff_sha256="$(sha256sum "$HANDOFF" | cut -d ' ' -f 1)"
if [[ "$actual_handoff_sha256" != "$EXPECTED_HANDOFF_SHA256" ]]; then
  printf 'Handoff SHA-256 mismatch: expected %s, got %s\n' \
    "$EXPECTED_HANDOFF_SHA256" "$actual_handoff_sha256" >&2
  exit 1
fi

uv run --offline --frozen python scripts/modeling/validate_stage2_preflight.py \
  --handoff "$HANDOFF" \
  --supplemental-annotations "$SUPPLEMENTAL_ANNOTATIONS" \
  --supplemental-annotations-sha256 "$SUPPLEMENTAL_ANNOTATIONS_SHA256" \
  --supplemental-benchmark "$SUPPLEMENTAL_BENCHMARK" \
  --supplemental-benchmark-sha256 "$SUPPLEMENTAL_BENCHMARK_SHA256" \
  --control-benchmark "$CONTROL_BENCHMARK"

resume_args=()
if [[ -f "data/processed/modeling_autoresearch/$RUN_NAME/run_manifest.json" ]]; then
  resume_args=(--resume)
fi

uv run --offline --frozen python scripts/modeling/run_active_scenic_autoresearch.py \
  --handoff "$HANDOFF" \
  --run-name "$RUN_NAME" \
  --max-experiments 5 \
  --max-steps 2000 \
  --max-seconds 3600 \
  --seed 42 \
  --device mps \
  --expanded-benchmark-csv "$SUPPLEMENTAL_BENCHMARK" \
  --control-benchmark-csv "$CONTROL_BENCHMARK" \
  --control-dataset "$CONTROL_DATASET" \
  --route-qa-json "$ROUTE_QA" \
  --thresholds-json "$THRESHOLDS" \
  --supplemental-annotations "$SUPPLEMENTAL_ANNOTATIONS" \
  --supplemental-annotations-sha256 "$SUPPLEMENTAL_ANNOTATIONS_SHA256" \
  --supplemental-benchmark-sha256 "$SUPPLEMENTAL_BENCHMARK_SHA256" \
  "${resume_args[@]}"

printf 'METRIC stage2_preflight_valid=1\n'
printf 'METRIC immutable_handoff_valid=1\n'
