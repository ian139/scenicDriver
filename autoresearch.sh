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

actual_handoff_sha256="$(sha256sum "$HANDOFF" | cut -d ' ' -f 1)"
if [[ "$actual_handoff_sha256" != "$EXPECTED_HANDOFF_SHA256" ]]; then
  printf 'Handoff SHA-256 mismatch: expected %s, got %s\n' \
    "$EXPECTED_HANDOFF_SHA256" "$actual_handoff_sha256" >&2
  exit 1
fi

uv run --offline --frozen python scripts/modeling/validate_stage2_preflight.py \
  --handoff "$HANDOFF"

printf 'METRIC stage2_preflight_valid=1\n'
printf 'METRIC immutable_handoff_valid=1\n'
