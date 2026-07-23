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

uv run --offline --frozen python scripts/routing/autoresearch_profile.py
