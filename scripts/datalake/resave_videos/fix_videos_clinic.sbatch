#!/usr/bin/env bash
# SLURM job array for clinic (Jamail) FLIR video FPS/filename remediation.
# Processes 3,030 actionable files from manifest_clinic.csv in chunks of 100.
#
# Submit:  sbatch scripts/datalake/resave_videos/fix_videos_clinic.sbatch
# Monitor: squeue -u yewen

#SBATCH --job-name=fix-clinic-video
#SBATCH --partition=guppy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-30%4
#SBATCH --output=scripts/datalake/resave_videos/logs/fix_clinic_%a.out
#SBATCH --error=scripts/datalake/resave_videos/logs/fix_clinic_%a.err

set -euo pipefail

REPO=/scratch/yewen/BCM/video-sync-nbu
MANIFEST=${REPO}/scripts/datalake/resave_videos/manifest_clinic.csv
CONDA_ENV=/scratch/yewen/miniconda3/envs/videosyncnbu
PYTHON=${CONDA_ENV}/bin/python
CHUNK_SIZE=100

export PATH="${CONDA_ENV}/bin:${PATH}"

cd "${REPO}"

echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

${PYTHON} -m scripts.datalake.resave_videos.fix_video \
    --manifest "${MANIFEST}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --chunk-size "${CHUNK_SIZE}" \
    --status-dir "${REPO}/scripts/datalake/resave_videos/status_clinic" \
    -v

echo "Task ${SLURM_ARRAY_TASK_ID} finished — $(date)"
