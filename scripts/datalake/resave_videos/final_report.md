# Video FPS & Filename Remediation — Final Report

**Date:** 2026-03-25 (updated)
**Author:** Yewen Zhou

## Summary

Two bugs in the FLIR camera recording software caused ~27,000 video files to be saved with
incorrect frame rates and drifted filename timestamps. All affected files across both NBU
and clinic recordings have been re-encoded with correct FPS, renamed with correct timestamps,
and validated.

| Metric | NBU | Clinic | Total |
|--------|----:|-------:|------:|
| Files re-encoded (FPS fix) | 18,196 | 2,536 | 20,732 |
| Files remuxed (filename fix only) | 12 | 0 | 12 |
| Files copied (already correct) | 6,422 | 494 | 6,916 |
| **Total fixed** | **24,630** | **3,030** | **27,660** |
| Skipped (corrupt/no metadata) | 1,784 | 176 | 1,960 |

> **2026-03-25 — Re-encode redo required.** See [Re-encode Fix](#re-encode-fix-2026-03-25) below.
> The initial validation passed all 27,660 files but did not check frame counts, missing a
> frame-dropping bug in the re-encode step. All 20,732 re-encoded files must be redone.

## The Bugs

**Bug 1 — Wrong FPS (~39.58 instead of 30)**

The recording software computed FPS from hardware timestamp deltas instead of using the
actual camera trigger rate (30 FPS from Arduino). This caused all 2025 video files to be
encoded at ~39.58 FPS, making them play back ~32% too fast.

- Affected period: all 2025 recordings
- Fixed in recording software: 2025-12-05
- 2026 recordings are unaffected

**Bug 2 — Filename timestamp drift (~2.5 min per segment)**

The recording software calculated each segment's filename by adding
`segment_length / hardware_fps` to the previous segment's timestamp. Because it used ~39.5
instead of ~30 for FPS, each segment's filename drifted ~2.5 minutes behind real time. By
the 5th segment, filenames were ~13 minutes off.

- Affected period: 2025-03-05 to 2025-12-05
- First segment of each session was always correct
- Fix: replaced filename timestamps using `real_times[0]` from companion JSON metadata
  (authoritative UTC wall-clock time, converted to America/Chicago timezone)

## What Was Done

1. **Scanned** both datalake roots (`TRBD-53761`, `AA-56119`) across NBU and clinic layouts
2. **Probed** every MP4 via ffprobe for FPS, paired with companion JSON for correct timestamps
3. **Classified** each file: re-encode (wrong FPS), remux (wrong filename only), copy (correct), skip (corrupt/no metadata)
4. **Re-encoded** 20,732 files to 30 FPS using NVIDIA h264_nvenc on the GPU cluster (guppy partition, NVIDIA L40 GPUs)
5. **Renamed** 7,666 + 12 + 3 files using timestamps from companion JSON metadata
6. **Copied** 6,916 already-correct files to the new location
7. **Validated** all 27,660 output files via ffprobe (FPS check + filename check)

## Output Location

All fixed files are at `/mnt/new-datalake/NBU-video-recover/`, preserving the original
directory structure:

```
/mnt/new-datalake/NBU-video-recover/
  TRBD-53761/
    <patient>/NBU/<date>/video/<site>/*.mp4       (NBU recordings)
    <patient>/clinic/<date>/video/FLIR/*.mp4      (clinic recordings)
  AA-56119/
    <patient>/NBU/<date>/video/<site>/*.mp4
    <patient>/clinic/<date>/video/FLIR/*.mp4
```

Source files on the original datalake (`/mnt/datalake/data/`) are untouched.

## Skipped Files

| Reason | NBU | Clinic | Total | Notes |
|--------|----:|-------:|------:|-------|
| Corrupt (ffprobe failed) | 950 | 104 | 1,054 | Missing moov atom, same files from MP4 recovery project |
| No companion JSON | 834 | 72 | 906 | Cannot determine correct filename without metadata |
| **Total skipped** | **1,784** | **176** | **1,960** | |

These files remain on the original datalake only. The 1,054 corrupt files are candidates for
the video recovery pipeline if needed.

## Infrastructure Used

- **GPU cluster:** guppy partition (4 nodes x 1 NVIDIA L40 GPU each)
- **Encoding:** NVIDIA h264_nvenc hardware encoder via ffmpeg
- **Parallelization:** SLURM job arrays — 247 tasks for NBU, 31 tasks for clinic
- **Total processing time:** ~20 hours (NBU), ~2 hours (clinic)
- **Pipeline:** fully resume-safe (resubmit after interruption with no wasted work)

## Scripts

All pipeline code is in `scripts/datalake/resave_videos/`:

| Script | Purpose |
|--------|---------|
| `build_manifest.py` | Scan, probe, classify (supports `--layout nbu\|clinic`) |
| `fix_video.py` | Per-chunk re-encode/remux/copy worker |
| `validate.py` | Post-processing validation via ffprobe |
| `fix_videos.sbatch` | SLURM job array for NBU |
| `fix_videos_clinic.sbatch` | SLURM job array for clinic |

Detailed per-layout statistics in `manifest_nbu_report.md` and `manifest_clinic_report.md`.

## Re-encode Fix (2026-03-25)

### The Bug

The `_reencode()` function in `fix_video.py` placed `-r 30` **after** `-i` in the ffmpeg
command. As an output option, this told ffmpeg to conform the output to 30 FPS by **dropping
frames** to match the original (wrong) duration:

```
# OLD (wrong): -r 30 after -i = output option = drop frames
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -r 30 -fps_mode cfr output.mp4
```

Since the container metadata said ~39.58 FPS, ffmpeg dropped ~24% of frames to fit 30 FPS
into the same ~455-second duration.

Example (TRBD001_20250417_000543.24253448.mp4):

| | FPS | Frames | Duration |
|---|---|---|---|
| Source | 39.58 | 18,000 | 454.78s (wrong — container metadata) |
| Old re-encode (wrong) | 30.00 | 13,645 | 454.83s (same wrong duration, **4,355 frames dropped**) |
| Fixed re-encode | 30.00 | **18,000** | **600.00s** (correct real-time duration) |

### The Fix

Move `-r 30` **before** `-i` so it acts as an input option, telling ffmpeg to reinterpret
the container's ~39.58 FPS metadata as 30 FPS. All frames are kept and the duration stretches
to the correct real-time length:

```
# NEW (fixed): -r 30 before -i = input option = reinterpret timestamps
ffmpeg -r 30 -i input.mp4 -c:v h264_nvenc -preset p4 output.mp4
```

### Impact

| Category | NBU | Clinic | Total | Action needed |
|----------|----:|-------:|------:|---------------|
| reencode | 18,196 | 2,536 | 20,732 | **Must redo** — frames were dropped |
| remux | 12 | 0 | 12 | Fine — stream copy, no re-encoding |
| copy | 6,422 | 494 | 6,916 | Fine — literal file copy |
| **Total actionable** | **24,630** | **3,030** | **27,660** | |

**20,732 files must be re-encoded.** The 6,928 remux/copy files are unaffected.

### Why Validation Missed It

The original validation checked:
1. File exists
2. FPS = 30 (within tolerance)
3. Filename timestamp matches JSON

It did **not** check frame count. The re-encoded files had correct FPS (30) and correct
filenames, so they passed. Validation has now been updated to also compare source vs
destination frame count.

### Remediation Steps

```bash
cd /scratch/yewen/BCM/video-sync-nbu
bash scripts/datalake/resave_videos/redo_reencode.sh pull      # pull the fix
bash scripts/datalake/resave_videos/redo_reencode.sh delete    # delete 20,732 bad re-encodes
bash scripts/datalake/resave_videos/redo_reencode.sh submit    # resubmit SLURM jobs
# wait for jobs to finish...
bash scripts/datalake/resave_videos/redo_reencode.sh validate  # validate with frame count check
```
