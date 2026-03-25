#!/usr/bin/env python3
"""Validate output files from the video resave pipeline.

Reads a manifest CSV, probes each dst_path via ffprobe, and verifies:
  1. File exists
  2. FPS is within tolerance of 30 (for reencode/remux files)
  3. Filename timestamp matches json_timestamp (for renamed files)

Usage:
    python -m scripts.datalake.resave_videos.validate \
        --manifest manifest.csv \
        --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.parsers.videofileparser import VideoFileParser

log = logging.getLogger(__name__)

TARGET_FPS = 30.0
FPS_TOLERANCE = 1.0
_RE_TIMESTAMP_IN_FILENAME = re.compile(r"_(\d{8}_\d{6})\.")


# ── Per-file validation ──────────────────────────────────────────────


def _extract_filename_timestamp(path: str) -> str:
    """Extract YYYYMMDD_HHMMSS from a filename like ...prefix_20250603_133409.serial.mp4."""
    m = _RE_TIMESTAMP_IN_FILENAME.search(Path(path).name)
    return m.group(1) if m else ""


def validate_row(row: dict) -> dict:
    """Validate a single manifest row. Returns result dict."""
    dst = Path(row["dst_path"])
    action = row["action"]
    json_ts = row["json_timestamp"]
    was_reencoded = row["needs_reencode"] == "True"
    was_renamed = row["timestamp_mismatch"] == "True"
    src_frame_count = int(row.get("frame_count") or 0)

    result = {
        "dst_path": str(dst),
        "action": action,
        "exists": False,
        "fps_ok": "",
        "fps_actual": "",
        "frames_ok": "",
        "frames_actual": "",
        "filename_ok": "",
        "status": "pending",
        "errors": [],
    }

    # Check 1: file exists
    if not dst.exists():
        result["status"] = "FAIL"
        result["errors"].append("file missing")
        return result
    result["exists"] = True

    # Probe once, reuse for checks 2 and 3
    vfp = None
    if was_reencoded:
        try:
            vfp = VideoFileParser(dst)
        except Exception as e:
            result["fps_ok"] = "FAIL"
            result["frames_ok"] = "FAIL"
            result["errors"].append(f"ffprobe error: {str(e)[:200]}")

    # Check 2: FPS (only for re-encoded files)
    if was_reencoded and vfp is not None:
        fps = vfp.fps
        result["fps_actual"] = f"{fps:.3f}"
        if abs(fps - TARGET_FPS) <= FPS_TOLERANCE:
            result["fps_ok"] = "pass"
        else:
            result["fps_ok"] = "FAIL"
            result["errors"].append(f"fps={fps:.3f}")
    elif not was_reencoded:
        result["fps_ok"] = "skip"

    # Check 3: frame count (only for re-encoded files with known source count)
    if was_reencoded and vfp is not None and src_frame_count > 0:
        dst_frames = vfp.frame_count
        result["frames_actual"] = str(dst_frames)
        if dst_frames == src_frame_count:
            result["frames_ok"] = "pass"
        else:
            result["frames_ok"] = "FAIL"
            result["errors"].append(f"frames: src={src_frame_count}, dst={dst_frames}")
    elif not was_reencoded:
        result["frames_ok"] = "skip"

    # Check 4: filename timestamp (only for renamed files)
    if was_renamed and json_ts:
        actual_ts = _extract_filename_timestamp(str(dst))
        if actual_ts == json_ts:
            result["filename_ok"] = "pass"
        else:
            result["filename_ok"] = "FAIL"
            result["errors"].append(f"expected ts={json_ts}, got={actual_ts}")
    else:
        result["filename_ok"] = "skip"

    result["status"] = "FAIL" if result["errors"] else "pass"
    return result


# ── Orchestration ────────────────────────────────────────────────────

REPORT_FIELDS = [
    "dst_path",
    "action",
    "exists",
    "fps_ok",
    "fps_actual",
    "frames_ok",
    "frames_actual",
    "filename_ok",
    "status",
    "errors",
]


def validate_manifest(manifest: Path, workers: int) -> None:
    """Validate all files in a manifest CSV."""
    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))

    log.info("Validating %d files from %s", len(rows), manifest.name)

    results: list[dict] = []
    done = 0
    total = len(rows)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(validate_row, row): row for row in rows}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            results.append(r)
            if done % 500 == 0 or r["status"] == "FAIL":
                log.info(
                    "[%d/%d] %s %s",
                    done,
                    total,
                    r["status"],
                    ", ".join(r["errors"]) if r["errors"] else "",
                )

    # Write report CSV
    report_path = manifest.with_name(f"{manifest.stem}_validation{manifest.suffix}")
    with open(report_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        w.writeheader()
        for r in results:
            r["errors"] = "; ".join(r["errors"])
            w.writerow(r)
    log.info("Validation report → %s", report_path)

    # Summary
    pass_count = sum(1 for r in results if r["status"] == "pass")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    log.info("Results: %d pass, %d FAIL out of %d total", pass_count, fail_count, total)

    if fail_count:
        log.warning("FAILURES detected. Check %s for details.", report_path.name)
        # Show first few failures
        for r in results:
            if r["status"] == "FAIL":
                log.warning("  FAIL: %s — %s", Path(r["dst_path"]).name, r["errors"])
                fail_count -= 1
                if fail_count <= 0 or fail_count > len(results) - 10:
                    break
        sys.exit(1)
    else:
        log.info("All files passed validation.")


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate resaved video files")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel validation workers (default: 8)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    validate_manifest(args.manifest, args.workers)


if __name__ == "__main__":
    main()
