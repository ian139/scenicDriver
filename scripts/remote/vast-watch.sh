#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/remote/cmux_vast_host.py watch "$@"
