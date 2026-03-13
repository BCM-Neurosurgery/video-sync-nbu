# NBU Video Manifest Report

**Generated:** 2026-03-13 (updated with `_fixed.mp4` support)
**Script:** `build_manifest.py`
**Source roots:** `/mnt/datalake/data/TRBD-53761`, `/mnt/datalake/data/AA-56119`
**Output root:** `/mnt/new-datalake/NBU-video-recover`

## Total Files

| Category | Files |
|----------|------:|
| Actionable (in manifest.csv) | 24,630 |
| Skipped (in manifest_skipped.csv) | 1,784 |
| **Grand Total** | **26,414** |

Verified: grand total matches `find ... -name "*.mp4" | wc -l` across both datalake roots.

## Actionable Files by Category

| Category | FPS wrong? | Filename wrong? | Files | Hours | Action |
|----------|:---:|:---:|------:|------:|--------|
| Both wrong | Yes | Yes | 7,666 | — | reencode + rename |
| FPS only wrong | Yes | No | 10,530 | — | reencode (keep name) |
| Filename only wrong | No | Yes | 12 | 2.0 | remux (stream copy + rename) |
| Both correct | No | No | 6,422 | 1,060.3 | copy to new location |
| **Total reencode** | | | **18,196** | **2,275.3** | |
| **Total remux** | | | **12** | **2.0** | |
| **Total copy** | | | **6,422** | **1,060.3** | |
| **All actionable** | | | **24,630** | **3,337.5** | |

## Skipped Files by Reason

| Reason | Files | Description |
|--------|------:|-------------|
| ffprobe_failed | 950 | Corrupted MP4s (missing moov atom), same files from recovery project |
| no_companion_json | 834 | MP4s with no matching `<segment_id>.json` metadata (includes 662 `*_fixed.mp4` from recovery) |
| **Total skipped** | **1,784** | |

No DST collisions detected (no `manifest_collisions.csv` generated).

## Per-Patient Breakdown

### Actionable

| Patient | reencode | remux | copy | Total | Hours |
|---------|------:|------:|------:|------:|------:|
| TRBD000 | 0 | 0 | 1,272 | 1,272 | 210.7 |
| TRBD001 | 6,279 | 6 | 1,420 | 7,705 | 1,021.7 |
| TRBD002 | 5,422 | 6 | 1,450 | 6,878 | 920.7 |
| TRBD003 | 0 | 0 | 1,368 | 1,368 | 226.2 |
| AA002 | 1,991 | 0 | 912 | 2,903 | 395.9 |
| AA006 | 4,504 | 0 | 0 | 4,504 | 562.4 |
| **Total** | **18,196** | **12** | **6,422** | **24,630** | **3,337.5** |

### Skipped

| Patient | ffprobe_failed | no_companion_json | Total |
|---------|------:|------:|------:|
| TRBD001 | 492 | 378 | 870 |
| TRBD002 | 215 | 157 | 372 |
| AA002 | 188 | 166 | 354 |
| AA006 | 55 | 127 | 182 |
| TRBD003 | 0 | 6 | 6 |
| TRBD000 | 0 | 0 | 0 |
| **Total** | **950** | **834** | **1,784** |

## The 12 Remux Files — 1-Second Rounding, Not Drift

These 12 files have correct FPS (29.995) but a 1-second timestamp mismatch between the
filename and the companion JSON. They are **not** caused by the multi-minute drift bug
(which only affects 2025 data). Instead, this is a sub-second rounding difference between
`datetime.now()` in the recording software and `real_times[0]` from the hardware timestamps.

All 12 come from 2 segments across 2 patients, each with 6 cameras:

| Source file | Filename TS | JSON TS |
|-------------|-------------|---------|
| `.../TRBD001/NBU/2026-02-20/video/sleep/TRBD001_20260220_061353.24253448.mp4` | `20260220_061353` | `20260220_061352` |
| `.../TRBD001/NBU/2026-02-20/video/sleep/TRBD001_20260220_061353.24253450.mp4` | `20260220_061353` | `20260220_061352` |
| `.../TRBD001/NBU/2026-02-20/video/sleep/TRBD001_20260220_061353.24253452.mp4` | `20260220_061353` | `20260220_061352` |
| `.../TRBD001/NBU/2026-02-20/video/sleep/TRBD001_20260220_061353.24253459.mp4` | `20260220_061353` | `20260220_061352` |
| `.../TRBD001/NBU/2026-02-20/video/sleep/TRBD001_20260220_061353.24253460.mp4` | `20260220_061353` | `20260220_061352` |
| `.../TRBD001/NBU/2026-02-20/video/sleep/TRBD001_20260220_061353.24253466.mp4` | `20260220_061353` | `20260220_061352` |
| `.../TRBD002/NBU/2026-01-28/video/sleep/TRBD002_20260128_041721.24253448.mp4` | `20260128_041721` | `20260128_041720` |
| `.../TRBD002/NBU/2026-01-28/video/sleep/TRBD002_20260128_041721.24253450.mp4` | `20260128_041721` | `20260128_041720` |
| `.../TRBD002/NBU/2026-01-28/video/sleep/TRBD002_20260128_041721.24253452.mp4` | `20260128_041721` | `20260128_041720` |
| `.../TRBD002/NBU/2026-01-28/video/sleep/TRBD002_20260128_041721.24253459.mp4` | `20260128_041721` | `20260128_041720` |
| `.../TRBD002/NBU/2026-01-28/video/sleep/TRBD002_20260128_041721.24253460.mp4` | `20260128_041721` | `20260128_041720` |
| `.../TRBD002/NBU/2026-01-28/video/sleep/TRBD002_20260128_041721.24253466.mp4` | `20260128_041721` | `20260128_041720` |

In both cases the filename is exactly 1 second ahead of the JSON timestamp. The JSON
`real_times[0]` (hardware clock, UTC-converted) is the authoritative source, so these files
are remuxed (stream-copied with the corrected filename) — no re-encoding needed.

## Notes

- TRBD000 and TRBD003 have all correct FPS (2026 recordings) — only need copying
- AA006 has zero correct files — all 4,504 need re-encoding
- The 950 ffprobe failures correspond to corrupted MP4s from the recovery project (2026-03-11)
- 692 `*_fixed.mp4` files from previous manual recovery were parsed in the second manifest run:
  29 became actionable, 1 failed ffprobe, 662 had no companion JSON (skipped)
