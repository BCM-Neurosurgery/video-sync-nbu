#!/usr/bin/env bash
# Build manifest CSV for the NBU video FPS/filename remediation pipeline.
# Run on elias login node as yewen.
#
# First run (full scan, ~30-45 min):
#   bash scripts/datalake/resave_videos/build_manifest.sh
#
# Re-run with cache (skips ffprobe for already-probed files, much faster):
#   bash scripts/datalake/resave_videos/build_manifest.sh --cache

set -euo pipefail

cd /scratch/yewen/BCM/video-sync-nbu

/scratch/yewen/miniconda3/envs/videosyncnbu/bin/python \
  -m scripts.datalake.resave_videos.build_manifest \
  --roots /mnt/datalake/data/TRBD-53761 /mnt/datalake/data/AA-56119 \
  --out-root /mnt/new-datalake/NBU-video-recover \
  --output scripts/datalake/resave_videos/manifest.csv \
  "$@"
