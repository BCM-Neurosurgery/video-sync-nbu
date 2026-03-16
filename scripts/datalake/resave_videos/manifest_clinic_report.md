# Clinic (Jamail) Video Manifest Report

**Generated:** 2026-03-16
**Script:** `build_manifest.py --layout clinic`
**Source roots:** `/mnt/datalake/data/TRBD-53761`, `/mnt/datalake/data/AA-56119`
**Output root:** `/mnt/new-datalake/NBU-video-recover`

## Total Files

| Category | Files |
|----------|------:|
| Actionable (in manifest_clinic.csv) | 3,038 |
| Skipped (in manifest_clinic_skipped.csv) | 176 |
| **Grand Total** | **3,214** |

## Actionable Files by Category

| Category | FPS wrong? | Filename wrong? | Files | Action |
|----------|:---:|:---:|------:|--------|
| FPS only wrong | Yes | No | 2,533 | reencode (keep name) |
| Both wrong | Yes | Yes | 3 | reencode + rename |
| Both correct | No | No | 502 | copy to new location |
| **Total reencode** | | | **2,536** | |
| **Total copy** | | | **502** | |
| **All actionable** | | | **3,038** | **230.4 hours** |

Nearly all clinic files have correct filenames (only 3 with drift). Clinic visits are
shorter than NBU stays, so most segments are first segments where `datetime.now()` produces
the correct filename.

## Skipped Files by Reason

| Reason | Files | Description |
|--------|------:|-------------|
| ffprobe_failed | 104 | Corrupted MP4s (missing moov atom) |
| no_companion_json | 72 | MP4s with no matching `<segment_id>.json` metadata |
| **Total skipped** | **176** | |

## Per-Patient Breakdown

### Actionable

| Patient | reencode | copy | Total | Hours |
|---------|------:|------:|------:|------:|
| AA001 | 495 | 0 | 495 | 18.6 |
| AA002 | 662 | 18 | 680 | 33.8 |
| AA004 | 378 | 24 | 402 | 26.4 |
| AA005 | 171 | 76 | 247 | 32.1 |
| AA006 | 285 | 73 | 358 | 45.9 |
| AA007 | 0 | 159 | 159 | 24.1 |
| AA009 | 0 | 9 | 9 | 1.2 |
| AA010 | 0 | 24 | 24 | 3.1 |
| TRBD001 | 212 | 18 | 230 | 14.8 |
| TRBD002 | 333 | 60 | 393 | 24.7 |
| TRBD003 | 0 | 33 | 33 | 5.4 |
| TEST1208 | 0 | 2 | 2 | 0.0 |
| test | 0 | 6 | 6 | 0.3 |
| **Total** | **2,536** | **502** | **3,038** | **230.4** |

### Skipped

| Patient | ffprobe_failed | no_companion_json | Total |
|---------|------:|------:|------:|
| AA005 | 50 | 22 | 72 |
| AA007 | 3 | 48 | 51 |
| AA004 | 15 | 0 | 15 |
| TRBD002 | 15 | 0 | 15 |
| AA001 | 9 | 0 | 9 |
| AA006 | 6 | 0 | 6 |
| TRBD001 | 5 | 2 | 7 |
| AA002 | 1 | 0 | 1 |
| **Total** | **104** | **72** | **176** |

## Notes

- 63 clinic directories discovered across both datalake roots
- Directory layout: `<root>/<patient>/clinic/<date>/video/FLIR/`
- 3 cameras at Jamail (vs 6-8 at NBU), hence lower file counts
- AA001, AA007, AA009, AA010, TEST1208, test are patients not seen in the NBU manifest
- The 104 ffprobe failures are candidates for the video recovery pipeline
