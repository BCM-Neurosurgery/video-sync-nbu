#!/usr/bin/env python3
"""Fix NBU videos: re-encode, remux, or copy based on manifest.

Designed to be called from SLURM job arrays. Each invocation processes
a chunk of rows from the manifest CSV.

Usage:
    python -m scripts.datalake.resave_videos.fix_video \
        --manifest manifest.csv \
        --task-id 0 \
        --chunk-size 100
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


# ── ffmpeg commands ──────────────────────────────────────────────────


def _reencode(src: Path, dst: Path) -> subprocess.CompletedProcess:
    """Re-encode video to 30 FPS using NVIDIA h264_nvenc."""
    return subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-hwaccel",
            "cuda",
            "-i",
            str(src),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-r",
            "30",
            "-fps_mode",
            "cfr",
            str(dst),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _remux(src: Path, dst: Path) -> subprocess.CompletedProcess:
    """Stream-copy into new container (rename only, no re-encoding)."""
    return subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-i",
            str(src),
            "-c",
            "copy",
            str(dst),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


# ── Per-file processing ─────────────────────────────────────────────


def process_row(row: dict, dry_run: bool = False) -> dict:
    """Process a single manifest row. Returns status dict."""
    src = Path(row["src_path"])
    dst = Path(row["dst_path"])
    action = row["action"]

    result = {
        "src_path": str(src),
        "dst_path": str(dst),
        "action": action,
        "status": "pending",
        "error": "",
        "elapsed_sec": 0.0,
    }

    if not src.exists():
        result["status"] = "error"
        result["error"] = "source not found"
        return result

    if dst.exists():
        result["status"] = "skipped"
        return result

    if dry_run:
        result["status"] = "dry_run"
        return result

    dst.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    try:
        if action == "reencode":
            _reencode(src, dst)
        elif action == "remux":
            _remux(src, dst)
        elif action == "copy":
            shutil.copy2(str(src), str(dst))
        else:
            result["status"] = "error"
            result["error"] = f"unknown action: {action}"
            return result

        result["status"] = "done"
    except subprocess.CalledProcessError as e:
        result["status"] = "error"
        result["error"] = (e.stderr or str(e))[:500]
        if dst.exists():
            dst.unlink()
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        if dst.exists():
            dst.unlink()

    result["elapsed_sec"] = round(time.monotonic() - t0, 2)
    return result


# ── Chunk processing ─────────────────────────────────────────────────


def process_chunk(
    manifest_path: Path, task_id: int, chunk_size: int, dry_run: bool = False
) -> list[dict]:
    """Process a chunk of manifest rows identified by task_id."""
    with open(manifest_path) as fh:
        rows = list(csv.DictReader(fh))

    start = task_id * chunk_size
    end = min(start + chunk_size, len(rows))
    chunk = rows[start:end]

    if not chunk:
        log.warning("Task %d: no rows (start=%d, total=%d)", task_id, start, len(rows))
        return []

    log.info(
        "Task %d: processing rows %d–%d (%d files)", task_id, start, end - 1, len(chunk)
    )

    results = []
    for i, row in enumerate(chunk):
        r = process_row(row, dry_run=dry_run)
        results.append(r)

        sym = {"done": ".", "skipped": "S", "error": "E", "dry_run": "D"}.get(
            r["status"], "?"
        )
        if (i + 1) % 10 == 0 or r["status"] == "error":
            log.info(
                "  [%d/%d] %s %s",
                i + 1,
                len(chunk),
                sym,
                r["error"] if r["status"] == "error" else "",
            )

    return results


# ── Status output ────────────────────────────────────────────────────

STATUS_FIELDS = ["src_path", "dst_path", "action", "status", "error", "elapsed_sec"]


def write_status(results: list[dict], output: Path) -> None:
    """Write per-task status CSV."""
    if not results:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STATUS_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(r)


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Fix NBU videos based on manifest")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest.csv")
    ap.add_argument(
        "--task-id", type=int, required=True, help="SLURM array task ID (0-based)"
    )
    ap.add_argument(
        "--chunk-size", type=int, default=100, help="Rows per task (default: 100)"
    )
    ap.add_argument(
        "--status-dir",
        type=Path,
        default=None,
        help="Directory for per-task status CSVs (default: <manifest_dir>/status/)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Log actions without executing"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = process_chunk(args.manifest, args.task_id, args.chunk_size, args.dry_run)

    # Summary
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    total_time = sum(r["elapsed_sec"] for r in results)
    log.info("Task %d done: %s (%.1fs total)", args.task_id, by_status, total_time)

    # Write status CSV
    status_dir = args.status_dir or args.manifest.parent / "status"
    write_status(results, status_dir / f"status_{args.task_id:04d}.csv")
    log.info("Status → %s", status_dir / f"status_{args.task_id:04d}.csv")


if __name__ == "__main__":
    main()
