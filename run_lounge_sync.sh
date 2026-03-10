#!/usr/bin/env bash
set -euo pipefail

AUDIO_DIR=~/mnt/datalake/TRBD-53761/TRBD001/NBU/2026-02-19/audio/lounge
VIDEO_DIR=~/mnt/datalake/TRBD-53761/TRBD001/NBU/2026-02-19/video/lounge
OUT_DIR=data/TRBD001_260219_lounge

python -m scripts.cli.cli_nbu \
  --audio-dir "$AUDIO_DIR" \
  --video-dir "$VIDEO_DIR" \
  --out-dir "$OUT_DIR" \
  --site nbu_lounge \
  --split \
  --log-level INFO \
  "$@"
