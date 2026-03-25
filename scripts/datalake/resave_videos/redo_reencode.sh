#!/usr/bin/env bash
# redo_reencode.sh — Re-do all re-encoded videos after the -r 30 fix
#
# The original _reencode() placed -r 30 AFTER -i, which dropped ~24% of
# frames.  The fix moves -r 30 BEFORE -i so ffmpeg reinterprets the
# container's ~39.58fps as 30fps, keeping all frames.
#
# Usage:
#   cd /scratch/yewen/BCM/video-sync-nbu
#   bash scripts/datalake/resave_videos/redo_reencode.sh pull
#   bash scripts/datalake/resave_videos/redo_reencode.sh delete
#   bash scripts/datalake/resave_videos/redo_reencode.sh submit
#   # ... wait for jobs to finish ...
#   bash scripts/datalake/resave_videos/redo_reencode.sh validate

set -euo pipefail
cd /scratch/yewen/BCM/video-sync-nbu

MANIFEST_NBU=scripts/datalake/resave_videos/manifest.csv
MANIFEST_CLINIC=scripts/datalake/resave_videos/manifest_clinic.csv

extract_reencode_paths() {
    awk -F',' '
        { gsub(/\r/, "") }
        NR==1 { for(i=1;i<=NF;i++) {if($i=="action") a=i; if($i=="dst_path") d=i} next }
        $a=="reencode" { print $d }
    ' "$1"
}

step="${1:-help}"

case "$step" in
    pull)
        echo "=== Pulling latest code ==="
        git pull
        echo "Done."
        ;;

    delete)
        echo "=== Deleting bad re-encodes ==="

        echo "--- NBU ---"
        extract_reencode_paths "$MANIFEST_NBU" > /tmp/reencode_nbu.txt
        echo "Files to delete: $(wc -l < /tmp/reencode_nbu.txt)"
        xargs rm -f < /tmp/reencode_nbu.txt

        echo "--- Clinic ---"
        extract_reencode_paths "$MANIFEST_CLINIC" > /tmp/reencode_clinic.txt
        echo "Files to delete: $(wc -l < /tmp/reencode_clinic.txt)"
        xargs rm -f < /tmp/reencode_clinic.txt

        rm -rf /mnt/new-datalake/NBU-video-recover/_test_reencode

        echo "Done. Deleted all bad re-encodes."
        ;;

    submit)
        echo "=== Submitting SLURM jobs ==="
        sbatch scripts/datalake/resave_videos/fix_videos.sbatch
        sbatch scripts/datalake/resave_videos/fix_videos_clinic.sbatch
        echo "Monitor with: squeue -u yewen"
        ;;

    validate)
        echo "=== Validating NBU ==="
        python -m scripts.datalake.resave_videos.validate \
            --manifest "$MANIFEST_NBU"

        echo "=== Validating Clinic ==="
        python -m scripts.datalake.resave_videos.validate \
            --manifest "$MANIFEST_CLINIC"
        ;;

    help|*)
        echo "Usage: bash $0 {pull|delete|submit|validate}"
        echo ""
        echo "  pull      — git pull the fix"
        echo "  delete    — delete bad re-encoded files (keeps remux/copy)"
        echo "  submit    — resubmit SLURM job arrays"
        echo "  validate  — run validation with frame count check"
        ;;
esac
