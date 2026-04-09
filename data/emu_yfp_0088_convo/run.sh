#!/usr/bin/env bash
# EMU time-sync: patient YFP, task EMU-0088_convo
# Cameras: 23512014, 18486634, 18486638
#
# Run from the repo root:
#   bash data/emu_yfp_0088_convo/run.sh
#
# Flags:
#   --patient-dir  stitched NSP files for this patient
#   --video-dir    multi-camera MP4s + JSON companions
#   --out-dir      synced output directory
#   --chunk-base   datalake root for resolving chunk NEV paths.
#                  The DB stores relative paths like
#                  "YFPDatafile/DATA/20250507-.../NSP1-....nev";
#                  chunk-base is prepended to form the full path.
#                  On Elias: /mnt/datalake/data/emu/
#                  On Mac NFS mount: ~/mnt/datalake/emu/
#   --keywords     filter to tasks matching this substring
#   --cam-serial   camera serials to sync (repeat per camera)

set -euo pipefail
cd "$(dirname "$0")/../.."

python -m scripts.cli.cli_emu_time \
  --patient-dir ~/mnt/stitched/EMU-18112/YFP \
  --video-dir ~/mnt/datalake/emu/YFPDatafile/VIDEO \
  --out-dir data/emu_yfp_0088_convo/out \
  --chunk-base ~/mnt/datalake/emu/ \
  --keywords EMU-0088 \
  --cam-serial 23512014 \
  --cam-serial 18486634 \
  --cam-serial 18486638 \
  --log-level DEBUG
