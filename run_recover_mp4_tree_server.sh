#!/usr/bin/env bash
set -euo pipefail

# Server defaults.
TRBD_ROOT=/mnt/datalake/data/TRBD-53761
AA_ROOT=/mnt/datalake/data/AA-56119
LOG_DIR=/home/auto/CODE/utils/video-sync-nbu

exec python -m scripts.fix.recover_mp4_tree \
  --roots "$TRBD_ROOT" "$AA_ROOT" \
  --log-dir "$LOG_DIR" \
  "$@"
