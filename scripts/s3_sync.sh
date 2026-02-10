#!/usr/bin/env bash
set -euo pipefail

# Sync local data to S3.
# Usage:
#   SCENIC_S3_BUCKET=scenicdriver-data ./scripts/s3_sync.sh
# Optional:
#   DRY_RUN=1 SCENIC_S3_BUCKET=... ./scripts/s3_sync.sh

if [[ -z "${SCENIC_S3_BUCKET:-}" ]]; then
  echo "SCENIC_S3_BUCKET is required (e.g., scenicdriver-data)."
  exit 1
fi

S3_URI="s3://${SCENIC_S3_BUCKET}"
DRY_FLAG=""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_FLAG="--dryrun"
fi

echo "Syncing raw tiles..."
aws s3 sync data/raw/ "${S3_URI}/raw/" ${DRY_FLAG} \
  --exclude "*.tif" --exclude "*.npy"

echo "Syncing processed artifacts..."
aws s3 sync data/processed/ "${S3_URI}/processed/" ${DRY_FLAG}

echo "Syncing models (optional)..."
aws s3 sync models/ "${S3_URI}/models/" ${DRY_FLAG}

echo "Done."
