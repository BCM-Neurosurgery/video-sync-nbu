#!/usr/bin/env bash
# Submit the NBU video fix SLURM job array.
# Run from elias login node as yewen.
#
# Usage: bash scripts/datalake/resave_videos/fix_videos.sh
#    or: bash scripts/datalake/resave_videos/fix_videos.sh --dry-run

set -euo pipefail

REPO=/scratch/yewen/BCM/video-sync-nbu
cd "${REPO}"

# Create logs directory
mkdir -p scripts/datalake/resave_videos/logs

# Verify manifest exists
MANIFEST=scripts/datalake/resave_videos/manifest.csv
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: ${MANIFEST} not found. Run build_manifest.sh first."
    exit 1
fi

TOTAL=$(tail -n +2 "${MANIFEST}" | wc -l)
CHUNKS=$(( (TOTAL + 99) / 100 ))
echo "Manifest: ${TOTAL} files → ${CHUNKS} array tasks (100 files/task, max 4 concurrent)"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Dry run — not submitting. Would run: sbatch fix_videos.sbatch"
    exit 0
fi

sbatch scripts/datalake/resave_videos/fix_videos.sbatch
echo "Submitted. Monitor with: squeue -u yewen"
