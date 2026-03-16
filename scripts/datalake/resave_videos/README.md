# NBU Video Resave Pipeline

One-off batch pipeline to fix two bugs in ~26,000 NBU video files recorded by FLIR
Blackfly S cameras via the MulticameraTracking software, then save corrected copies to a
new datalake location.

## The Two Bugs

**Bug 1 — Wrong FPS (~39.58 instead of 30)**

The recording software computed FPS from hardware timestamp deltas
(`1.0 / mean(diff(timestamps)) * 1e-9`), which gave ~39.58 due to initialization
artifacts. The cameras are actually Arduino-triggered at 30 FPS. Fixed in commit `a31a3a9`
(2025-12-05). All 2025 data is affected; 2026 data is correct (~29.995 FPS).

**Bug 2 — Wrong filename timestamps (cumulative drift per segment)**

`update_filename()` in `flir_recording_api.py` calculated each segment's filename timestamp
by adding `segment_length / hardware_fps` to the previous segment's timestamp. Because it
used ~39.5 FPS instead of ~30, each segment drifts ~2.5 minutes behind real time. The first
segment of each session is always correct (set by `datetime.now()`). Affected period:
2025-03-05 to 2025-12-05.

## The Fix

1. Correct filenames using `real_times[0]` from companion JSON files (authoritative UTC
   wall-clock time, converted to America/Chicago)
2. Re-encode wrong-FPS videos to 30 FPS using NVIDIA h264_nvenc on GPU cluster
3. Stream-copy (remux) videos that only need a filename fix
4. Copy already-correct videos to the new location

**Source:** `/mnt/datalake/data/TRBD-53761`, `/mnt/datalake/data/AA-56119`
**Destination:** `/mnt/new-datalake/NBU-video-recover`

## Files in This Directory

| File | Purpose |
|------|---------|
| `build_manifest.py` | Phase 0: scan datalake, probe each MP4, classify into reencode/remux/copy/skip |
| `build_manifest.sh` | Shell wrapper for NBU manifest |
| `build_manifest_clinic.sh` | Shell wrapper for clinic (Jamail) manifest |
| `fix_video.py` | Phase 1: per-chunk worker that re-encodes, remuxes, or copies based on manifest |
| `fix_videos.sbatch` | Phase 2: SLURM job array definition |
| `fix_videos.sh` | Shell wrapper to submit the SLURM job array |
| `manifest.csv` | NBU manifest — 24,630 actionable files (on server, not committed) |
| `manifest_clinic.csv` | Clinic manifest — 3,038 actionable files (on server, not committed) |
| `manifest_nbu_report.md` | NBU manifest statistics and verification |
| `manifest_clinic_report.md` | Clinic manifest statistics and verification |
| `status/` | Per-task status CSVs written by `fix_video.py` (on server) |

## How to Run

All steps run on the **elias** cluster as user **yewen**.
Repo location on server: `/scratch/yewen/BCM/video-sync-nbu`

### Phase 0 — Build Manifest

Scans the datalake, ffprobes every MP4, pairs with companion JSONs, and classifies each
file. Runs on the login node (no GPU needed). Takes ~30-45 minutes.

```bash
ssh yewen@elias
cd /scratch/yewen/BCM/video-sync-nbu
bash scripts/datalake/resave_videos/build_manifest.sh
```

Produces `manifest.csv` and `manifest_skipped.csv` under `scripts/datalake/resave_videos/`.

### Phase 1+2 — Fix Videos via SLURM

Submits a SLURM job array that processes all 24,601 files in parallel across 4 GPU nodes.
Each task handles 100 files. Re-encodes use h264_nvenc on NVIDIA L40 GPUs.

```bash
# From elias login node:
cd /scratch/yewen/BCM/video-sync-nbu
bash scripts/datalake/resave_videos/fix_videos.sh

# Monitor progress:
squeue -u yewen

# Check logs:
ls scripts/datalake/resave_videos/logs/
cat scripts/datalake/resave_videos/logs/fix_video_0.out

# Check per-task status:
ls scripts/datalake/resave_videos/status/
```

The job is **resume-safe** — if a destination file already exists, that row is skipped.
You can resubmit the same job array to retry any failures.

### Phase 3 — Validate (not yet built)

Will verify each output file exists, probe FPS = 30, and confirm the filename matches the
JSON timestamp. Will produce a `validation_report.csv`.

### Phase 4 — Reporting (not yet built)

Final stakeholder summary.

## Cluster Details

| Item | Value |
|------|-------|
| Login node | elias (10.28.0.202) |
| User | yewen |
| Repo path | `/scratch/yewen/BCM/video-sync-nbu` |
| Conda env | `/scratch/yewen/miniconda3/envs/videosyncnbu/` |
| ffmpeg | Conda 8.0.1 with h264_nvenc (available on all nodes via NFS) |
| SLURM partition | `guppy` (4 nodes x 1 NVIDIA L40 GPU) |
| Datalake (source) | `/mnt/datalake/data/` (NFS, read-only for yewen) |
| New datalake (dest) | `/mnt/new-datalake/NBU-video-recover/` (chmod 777) |

## Quick Reference

```bash
# How many files in the manifest?
wc -l scripts/datalake/resave_videos/manifest.csv

# How many done so far?
find /mnt/new-datalake/NBU-video-recover -name "*.mp4" | wc -l

# How many errors across all tasks?
grep -c '"error"' scripts/datalake/resave_videos/status/status_*.csv

# Re-submit failed tasks only (after fixing the issue):
# Just resubmit the same sbatch — already-done files are skipped automatically
sbatch scripts/datalake/resave_videos/fix_videos.sbatch
```
