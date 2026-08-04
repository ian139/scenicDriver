#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision_vast.sh — fail-fast runtime provisioning for a Vast.ai container
#
# Image = how to run. S3 = what to run on. This script prepares one
# temporary Vast.ai GPU instance/container for one short validation run.
#
# Usage:
#   SCENIC_S3_BUCKET=<bucket> [SCENIC_S3_DATA_PREFIX=<prefix>] \
#   [SCENIC_S3_MODELS_PREFIX=<prefix>] [SCENIC_S3_OUTPUT_PREFIX=<prefix>] \
#   [SCENIC_CHECKPOINT_PATH=<path>] [SCENIC_DATASET_PATH=<path>] \
#   [SCENIC_TIMEOUT_MINUTES=<n>] bash scripts/remote/provision_vast.sh
#
# Environment variables (no secrets in this file):
#   SCENIC_S3_BUCKET                 S3 bucket name (required)
#   SCENIC_S3_DATA_PREFIX            Required S3 data prefix (default: processed/regression/)
#   SCENIC_S3_MODELS_PREFIX          Required S3 model prefix (default: models/)
#   SCENIC_S3_OUTPUT_PREFIX          S3 output prefix (default: outputs/vast/<run-id>/)
#   SCENIC_RUN_ID                    Output run id (default: UTC timestamp)
#   SCENIC_CHECKPOINT_PATH           Local checkpoint path override (optional)
#   SCENIC_DATASET_PATH              Local .npz dataset path override (optional)
#   SCENIC_TIMEOUT_MINUTES           Global wall-clock timeout (default: 30)
#   SCENIC_STEP_TIMEOUT_SECONDS      Per cloud/compute command timeout (default: 600)
#   SCENIC_ALLOW_MISSING_ARTIFACTS   Set to 1 for smoke-only fallback (default: 0)
#   SCENIC_DATA_ROOT                 Local data dir (default: <repo>/data/processed/regression)
#   SCENIC_MODELS_ROOT               Local models dir (default: <repo>/models)
#   SCENIC_OUTPUT_ROOT               Local outputs dir (default: <repo>/scenic_artifacts/vast/<run-id>)
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve project root early so Python commands work from any cwd.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── helpers ────────────────────────────────────────────────────────────────
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }
die() { log "FATAL: $*"; exit 1; }

# ── defaults ───────────────────────────────────────────────────────────────
: "${SCENIC_S3_DATA_PREFIX:=processed/regression/}"
: "${SCENIC_S3_MODELS_PREFIX:=models/}"
: "${SCENIC_RUN_ID:=$(date -u +"%Y%m%dT%H%M%SZ")}"
: "${SCENIC_S3_OUTPUT_PREFIX:=outputs/vast/${SCENIC_RUN_ID}/}"
: "${SCENIC_TIMEOUT_MINUTES:=30}"
: "${SCENIC_STEP_TIMEOUT_SECONDS:=600}"
: "${SCENIC_ALLOW_MISSING_ARTIFACTS:=0}"
: "${SCENIC_DATA_ROOT:=${PROJECT_ROOT}/data/processed/regression}"
: "${SCENIC_MODELS_ROOT:=${PROJECT_ROOT}/models}"
: "${SCENIC_OUTPUT_ROOT:=${PROJECT_ROOT}/scenic_artifacts/vast/${SCENIC_RUN_ID}}"
: "${SCENIC_CHECKPOINT_PATH:=}"
: "${SCENIC_DATASET_PATH:=}"

TIMEOUT_SEC=$(( SCENIC_TIMEOUT_MINUTES * 60 ))
START_EPOCH=$(date +%s)

elapsed_seconds() { echo $(( $(date +%s) - START_EPOCH )); }

timeout_check() {
    local elapsed
    elapsed=$(elapsed_seconds)
    if (( elapsed >= TIMEOUT_SEC )); then
        die "Hard timeout of ${SCENIC_TIMEOUT_MINUTES}m reached after ${elapsed}s"
    fi
}

run_with_timeout() {
    timeout_check
    log "+ $*"
    if declare -F "$1" >/dev/null 2>&1; then
        "$@"
    elif command -v timeout >/dev/null 2>&1; then
        timeout "${SCENIC_STEP_TIMEOUT_SECONDS}" "$@"
    else
        "$@"
    fi
}

verify_identity() {
    python - <<'PY'
import boto3

boto3.client("sts").get_caller_identity()
PY
}

verify_bucket() {
    python - <<'PY'
import os
import boto3

boto3.client("s3").head_bucket(Bucket=os.environ["SCENIC_S3_BUCKET"])
PY
}

download_prefix() {
    local prefix="$1"
    local dest="$2"
    run_with_timeout python -m src.data_pipeline.s3 download-prefix \
        --bucket "$SCENIC_S3_BUCKET" \
        --prefix "$prefix" \
        --dest "$dest" \
        --required
}

upload_prefix() {
    local src="$1"
    local prefix="$2"
    run_with_timeout python -m src.data_pipeline.s3 upload-prefix \
        --src "$src" \
        --bucket "$SCENIC_S3_BUCKET" \
        --prefix "$prefix" \
        --required
}

require_s3_prefix() {
    local prefix="$1"
    local label="$2"
    if [[ "$SCENIC_ALLOW_MISSING_ARTIFACTS" == "1" ]]; then
        log "allow-missing: skipping required-prefix check for ${label}: s3://${SCENIC_S3_BUCKET}/${prefix}"
        return 0
    fi
    if ! run_with_timeout python -m src.data_pipeline.s3 check-prefix \
        --bucket "$SCENIC_S3_BUCKET" \
        --prefix "$prefix" \
        --required
    then
        die "Required ${label} prefix is empty or inaccessible: s3://${SCENIC_S3_BUCKET}/${prefix}"
    fi
    log "required ${label} prefix has objects: s3://${SCENIC_S3_BUCKET}/${prefix}"
}

first_file() {
    local root="$1"
    local pattern="$2"
    find "$root" -type f -name "$pattern" -print -quit
}

# ── 1. S3 auth ────────────────────────────────────────────────────────────
log "=== STEP 1: S3 auth ==="
log "project root: $PROJECT_ROOT"
timeout_check

if [[ -z "${SCENIC_S3_BUCKET:-}" ]]; then
    die "SCENIC_S3_BUCKET is not set"
fi
if ! run_with_timeout verify_identity; then
    die "AWS identity check failed; verify runtime credentials"
fi
if ! run_with_timeout verify_bucket; then
    die "S3 bucket is not reachable or credentials lack access: s3://${SCENIC_S3_BUCKET}"
fi
log "AWS identity verified"
log "S3 bucket reachable: s3://${SCENIC_S3_BUCKET}"

# ── 2. Runtime folders ────────────────────────────────────────────────────
log "=== STEP 2: runtime folders ==="
for d in "$SCENIC_DATA_ROOT" "$SCENIC_MODELS_ROOT" "$SCENIC_OUTPUT_ROOT"; do
    mkdir -p "$d"
    log "created/verified $d"
done

# ── 3. Pull data and models from S3 ───────────────────────────────────────
log "=== STEP 3: required S3 input sync ==="
require_s3_prefix "$SCENIC_S3_DATA_PREFIX" "data"
require_s3_prefix "$SCENIC_S3_MODELS_PREFIX" "models"
download_prefix "$SCENIC_S3_DATA_PREFIX" "$SCENIC_DATA_ROOT"
download_prefix "$SCENIC_S3_MODELS_PREFIX" "$SCENIC_MODELS_ROOT"
log "S3 input sync complete"

# ── 4. GPU / CUDA smoke ───────────────────────────────────────────────────
log "=== STEP 4: GPU and CUDA smoke ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found; Vast instance is not exposing NVIDIA GPU tooling"
fi
run_with_timeout nvidia-smi
run_with_timeout python scripts/remote/container_smoke.py --device cuda --check-imports --json | tee "$SCENIC_OUTPUT_ROOT/container_smoke.json"
log "CUDA smoke passed"

# ── 5. Minimal real inference ─────────────────────────────────────────────
log "=== STEP 5: minimal inference ==="
if [[ -z "$SCENIC_CHECKPOINT_PATH" ]]; then
    SCENIC_CHECKPOINT_PATH="$(first_file "$SCENIC_MODELS_ROOT" "*.pt")"
fi
if [[ -z "$SCENIC_DATASET_PATH" ]]; then
    SCENIC_DATASET_PATH="$(first_file "$SCENIC_DATA_ROOT" "*.npz")"
fi

INFERENCE_ARGS=(--device cuda --output "$SCENIC_OUTPUT_ROOT/inference_result.json")
if [[ -n "$SCENIC_CHECKPOINT_PATH" && -f "$SCENIC_CHECKPOINT_PATH" ]]; then
    INFERENCE_ARGS+=(--checkpoint "$SCENIC_CHECKPOINT_PATH")
    log "using checkpoint: $SCENIC_CHECKPOINT_PATH"
elif [[ "$SCENIC_ALLOW_MISSING_ARTIFACTS" == "1" ]]; then
    log "allow-missing: no checkpoint found; running synthetic classifier fallback"
else
    die "No .pt checkpoint found under $SCENIC_MODELS_ROOT; set SCENIC_CHECKPOINT_PATH or upload model weights to S3"
fi
if [[ -n "$SCENIC_DATASET_PATH" && -f "$SCENIC_DATASET_PATH" ]]; then
    INFERENCE_ARGS+=(--dataset "$SCENIC_DATASET_PATH")
    log "using dataset sample: $SCENIC_DATASET_PATH"
elif [[ "$SCENIC_ALLOW_MISSING_ARTIFACTS" == "1" ]]; then
    log "allow-missing: no .npz dataset found; using synthetic regression features if checkpoint exists"
else
    die "No .npz dataset found under $SCENIC_DATA_ROOT; set SCENIC_DATASET_PATH or upload data to S3"
fi
run_with_timeout python scripts/remote/minimal_inference.py "${INFERENCE_ARGS[@]}" | tee "$SCENIC_OUTPUT_ROOT/minimal_inference.log"
log "minimal inference output: $SCENIC_OUTPUT_ROOT/inference_result.json"

# ── 6. Sync outputs to S3 ─────────────────────────────────────────────────
log "=== STEP 6: S3 output sync ==="
upload_prefix "$SCENIC_OUTPUT_ROOT" "$SCENIC_S3_OUTPUT_PREFIX"
log "outputs synced to s3://${SCENIC_S3_BUCKET}/${SCENIC_S3_OUTPUT_PREFIX}"

# ── 7. Manual stop point ──────────────────────────────────────────────────
ELAPSED=$(elapsed_seconds)
log "=== PROVISIONING COMPLETE (${ELAPSED}s) ==="
log "No long job has been started. Review smoke/inference outputs before spending more GPU time."
log "Local outputs: $SCENIC_OUTPUT_ROOT"
log "S3 outputs: s3://${SCENIC_S3_BUCKET}/${SCENIC_S3_OUTPUT_PREFIX}"
log "Stop/destroy from local orchestrator: scripts/remote/vast-down.sh <task-name> --copy-artifacts --destroy --yes"
