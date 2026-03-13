#!/usr/bin/env bash
# SLURM job array for NBU video FPS/filename remediation.
# Processes 24,601 actionable files from manifest.csv in chunks of 100.
#
# Submit:  sbatch scripts/datalake/resave_videos/fix_videos.sbatch
# Monitor: squeue -u yewen
# Logs:    scripts/datalake/resave_videos/logs/fix_video_<task_id>.out

#SBATCH --job-name=fix-nbu-video
#SBATCH --partition=guppy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-246%4
#SBATCH --output=scripts/datalake/resave_videos/logs/fix_video_%a.out
#SBATCH --error=scripts/datalake/resave_videos/logs/fix_video_%a.err

set -euo pipefail

REPO=/scratch/yewen/BCM/video-sync-nbu
MANIFEST=${REPO}/scripts/datalake/resave_videos/manifest.csv
PYTHON=/scratch/yewen/miniconda3/envs/videosyncnbu/bin/python
CHUNK_SIZE=100

cd "${REPO}"

echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} -m scripts.datalake.resave_videos.fix_video \
    --manifest "${MANIFEST}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --chunk-size "${CHUNK_SIZE}" \
    -v

echo "Task ${SLURM_ARRAY_TASK_ID} finished — $(date)"
