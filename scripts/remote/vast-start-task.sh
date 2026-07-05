#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/remote/orca_vast_host.py start-task "$@"
