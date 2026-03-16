#!/usr/bin/env bash
# Build manifest CSV for clinic (Jamail) FLIR video remediation.
# Run on elias login node as yewen.
#
# First run (full scan):
#   bash scripts/datalake/resave_videos/build_manifest_clinic.sh
#
# Re-run with cache:
#   bash scripts/datalake/resave_videos/build_manifest_clinic.sh --cache

set -euo pipefail

cd /scratch/yewen/BCM/video-sync-nbu

/scratch/yewen/miniconda3/envs/videosyncnbu/bin/python \
  -m scripts.datalake.resave_videos.build_manifest \
  --layout clinic \
  --roots /mnt/datalake/data/TRBD-53761 /mnt/datalake/data/AA-56119 \
  --out-root /mnt/new-datalake/NBU-video-recover \
  --output scripts/datalake/resave_videos/manifest_clinic.csv \
  "$@"
