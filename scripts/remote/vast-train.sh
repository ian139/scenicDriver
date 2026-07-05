#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/remote/vast_train.py "$@"
