# NBU Video Manifest Report

**Generated:** 2026-03-13
**Script:** `build_manifest.py`
**Source roots:** `/mnt/datalake/data/TRBD-53761`, `/mnt/datalake/data/AA-56119`
**Output root:** `/mnt/new-datalake/NBU-video-recover`

## Total Files

| Category | Files |
|----------|------:|
| Actionable (in manifest.csv) | 24,601 |
| Skipped (in manifest_skipped.csv) | 1,813 |
| **Grand Total** | **26,414** |

Verified: grand total matches `find ... -name "*.mp4" | wc -l` across both datalake roots.

## Actionable Files by Category

| Category | FPS wrong? | Filename wrong? | Files | Hours | Action |
|----------|:---:|:---:|------:|------:|--------|
| Both wrong | Yes | Yes | 7,666 | — | reencode + rename |
| FPS only wrong | Yes | No | 10,502 | — | reencode (keep name) |
| Filename only wrong | No | Yes | 12 | 2.0 | remux (stream copy + rename) |
| Both correct | No | No | 6,421 | 1,060.3 | copy to new location |
| **Total reencode** | | | **18,168** | **2,274.0** | |
| **Total remux** | | | **12** | **2.0** | |
| **Total copy** | | | **6,421** | **1,060.3** | |
| **All actionable** | | | **24,601** | **3,336.3** | |

## Skipped Files by Reason

| Reason | Files | Description |
|--------|------:|-------------|
| ffprobe_failed | 949 | Corrupted MP4s (missing moov atom), same files from recovery project |
| unparseable_filename | 692 | `*_fixed.mp4` files from previous manual video recovery |
| no_companion_json | 172 | MP4s with no matching `<segment_id>.json` metadata |
| **Total skipped** | **1,813** | |

No DST collisions detected (no `manifest_collisions.csv` generated).

## Per-Patient Breakdown

### Actionable

| Patient | reencode | remux | copy | Total | Hours |
|---------|------:|------:|------:|------:|------:|
| TRBD000 | 0 | 0 | 1,272 | 1,272 | 210.7 |
| TRBD001 | 6,274 | 6 | 1,420 | 7,700 | 1,021.7 |
| TRBD002 | 5,400 | 6 | 1,450 | 6,856 | 919.7 |
| TRBD003 | 0 | 0 | 1,368 | 1,368 | 226.2 |
| AA002 | 1,991 | 0 | 911 | 2,902 | 395.8 |
| AA006 | 4,503 | 0 | 0 | 4,503 | 562.3 |
| **Total** | **18,168** | **12** | **6,421** | **24,601** | **3,336.3** |

### Skipped

| Patient | ffprobe_failed | unparseable_filename | no_companion_json | Total |
|---------|------:|------:|------:|------:|
| TRBD001 | 492 | 323 | 60 | 875 |
| TRBD002 | 214 | 156 | 24 | 394 |
| AA002 | 188 | 159 | 8 | 355 |
| AA006 | 55 | 54 | 74 | 183 |
| TRBD003 | 0 | 0 | 6 | 6 |
| TRBD000 | 0 | 0 | 0 | 0 |
| **Total** | **949** | **692** | **172** | **1,813** |

## Notes

- TRBD000 and TRBD003 have all correct FPS (2026 recordings) — only need copying
- AA006 has zero correct files — all 4,503 need re-encoding
- The 12 remux files are split evenly between TRBD001 (6) and TRBD002 (6) — these have correct FPS but drifted filenames
- The 949 ffprobe failures match the corrupted file count from the MP4 recovery project (2026-03-11)
- The 692 unparseable files are `*_fixed.mp4` outputs from previous manual recovery work
