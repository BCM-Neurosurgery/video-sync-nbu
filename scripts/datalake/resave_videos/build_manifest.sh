#!/usr/bin/env bash
# Build manifest CSV for the NBU video FPS/filename remediation pipeline.
# Run on elias login node as yewen. Takes ~30-45 min for full datalake scan.
#
# Usage: bash scripts/datalake/resave_videos/build_manifest.sh

set -euo pipefail

cd /scratch/yewen/BCM/video-sync-nbu

/scratch/yewen/miniconda3/envs/videosyncnbu/bin/python \
  -m scripts.datalake.resave_videos.build_manifest \
  --roots /mnt/datalake/data/TRBD-53761 /mnt/datalake/data/AA-56119 \
  --out-root /mnt/new-datalake/NBU-video-recover \
  --output scripts/datalake/resave_videos/manifest.csv \
  --workers 4
