#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export PYTHONHASHSEED=0

exec uv run python scripts/routing/northeast_expanded_autoresearch_benchmark.py --timed-runs 3
