#!/usr/bin/env bash
# Resume TRBD001 2026-02-19 lounge sync from segment 101318 onward.
# Segments 091142–100302 (6 segments, all 8 cameras) were fully synced in run0001.
# Segment 101318 was interrupted (7/8 cameras done, missing 25290590).
# Uses --skip-decode to reuse the existing decoded serial CSV.
#
# Note: old output is in runs/run0001/. This run writes flat to out-dir
# since --run-id is not used (CLI mode).
#
# Run on elias in a tmux session:
#   cd /scratch/yewen/BCM/video-sync-nbu
#   tmux new -s sync
#   bash run_lounge_sync_resume.sh
set -euo pipefail

AUDIO_DIR=/mnt/datalake/data/TRBD-53761/TRBD001/NBU/2026-02-19/audio/lounge
VIDEO_DIR=/mnt/datalake/data/TRBD-53761/TRBD001/NBU/2026-02-19/video/lounge
OUT_DIR=/mnt/datalake/synced_videos/TRBD-53761/TRBD001/TRBD001_02192026/lounge/out

python -m scripts.cli.cli_nbu \
  --audio-dir "$AUDIO_DIR" \
  --video-dir "$VIDEO_DIR" \
  --out-dir "$OUT_DIR" \
  --site nbu_lounge \
  --skip-decode \
  --resume-from-segment TRBD001_20260219_101318 \
  --log-level INFO \
  "$@"
