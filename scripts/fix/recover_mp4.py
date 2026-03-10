#!/usr/bin/env python3
"""
recover_mp4.py — Recover corrupted MP4 files missing the moov atom.

Scans a directory of MP4 files, identifies which are corrupted (missing moov),
automatically selects good reference files per camera, extracts the MPEG-4 VOL
header, and recovers each corrupted file by remuxing with ffmpeg.

Works with MP4 files written by OpenCV VideoWriter / MulticameraTracking,
which only writes the moov atom on clean shutdown. If the recording PC
crashes, the raw video frames (mdat) are intact but the index (moov) is
missing.

Usage:
  python -m scripts.fix.recover_mp4 INPUT_DIR OUTPUT_DIR                # dry-run
  python -m scripts.fix.recover_mp4 INPUT_DIR OUTPUT_DIR --run          # recover all
  python -m scripts.fix.recover_mp4 INPUT_DIR OUTPUT_DIR --run --cam X  # one camera

Requires: ffmpeg and ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from scripts.errors import FFmpegNotFoundError, MP4RecoveryError
from scripts.index.filepatterns import FilePatterns
from scripts.log.logutils import configure_standalone_logging, log_context
from scripts.parsers.videofileparser import VideoFileParser

__all__ = ["scan_directory", "recover_one", "main"]

log = logging.getLogger(__name__)

# MP4 container overhead: ftyp(28) + free(8) + mdat_header(8) = 44 bytes
CONTAINER_HEADER_SIZE = 44

# Files smaller than this are considered empty stubs (no recoverable frames)
EMPTY_STUB_THRESHOLD = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_ffmpeg() -> str:
    """Return the ffmpeg path, or raise FFmpegNotFoundError."""
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError("ffmpeg not found on PATH. Install FFmpeg.")
    return path


def _cam_from_path(path: Path) -> str:
    """Extract camera serial from a video filename using FilePatterns.

    Falls back to dot-separated parsing if the filename doesn't match
    the standard pattern (e.g. already-recovered *_fixed.mp4 files).
    """
    parsed = FilePatterns.parse_video_filename(path)
    if parsed:
        return parsed[1]
    # Fallback: PREFIX.SERIAL.mp4
    parts = path.stem.rsplit(".", 1)
    return parts[-1] if len(parts) >= 2 else ""


def _segment_id_from_path(path: Path) -> str:
    """Extract segment ID (timestamp group) from a video filename."""
    parsed = FilePatterns.parse_video_filename(path)
    if parsed:
        return parsed[0]
    return path.name.split(".")[0]


# ---------------------------------------------------------------------------
# MP4 atom scanning
# ---------------------------------------------------------------------------


def parse_mp4_atoms(path: Path) -> list[str]:
    """Return list of top-level atom names in an MP4 file."""
    fsize = path.stat().st_size
    atoms = []
    with open(path, "rb") as f:
        pos = 0
        while pos < fsize:
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            sz, name = struct.unpack(">I4s", hdr)
            name = name.decode("ascii", errors="replace")
            if sz == 0:
                sz = fsize - pos
            elif sz == 1:
                ext = f.read(8)
                if len(ext) < 8:
                    break
                sz = struct.unpack(">Q", ext)[0]
            if sz < 8:
                break
            atoms.append(name)
            pos += sz
    return atoms


def scan_directory(input_dir: Path) -> tuple[list[Path], list[Path], dict[str, Path]]:
    """Scan input_dir for MP4 files.

    Returns:
        (corrupted, good, ref_by_cam)
        - corrupted: MP4s missing moov atom
        - good: MP4s with moov atom
        - ref_by_cam: first good file per camera serial
    """
    corrupted = []
    good = []
    ref_by_cam: dict[str, Path] = {}

    mp4s = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".mp4")
    for path in mp4s:
        atoms = parse_mp4_atoms(path)
        if "moov" in atoms:
            good.append(path)
            cam = _cam_from_path(path)
            if cam and cam not in ref_by_cam:
                ref_by_cam[cam] = path
        else:
            corrupted.append(path)

    return corrupted, good, ref_by_cam


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def extract_vol_header(ref_path: Path) -> bytes:
    """Extract the MPEG-4 VOL header from a reference file's esds atom."""
    with open(ref_path, "rb") as f:
        data = f.read()

    idx = data.find(b"esds")
    if idx < 0:
        raise MP4RecoveryError(f"No esds atom found in {ref_path}")

    sz = struct.unpack(">I", data[idx - 4 : idx])[0]
    esds_data = data[idx - 4 : idx - 4 + sz]

    for i in range(len(esds_data)):
        if esds_data[i] == 0x05:
            j = i + 1
            while j < len(esds_data) and esds_data[j] == 0x80:
                j += 1
            config_len = esds_data[j]
            return esds_data[j + 1 : j + 1 + config_len]

    raise MP4RecoveryError(f"No DecoderSpecificInfo in esds of {ref_path}")


def detect_reference_info(ref_path: Path) -> tuple[str, float, int]:
    """Detect framerate, fps float, and frame count from a reference file.

    Returns:
        (r_frame_rate_str, fps_float, frame_count)
    """
    # Use raw ffprobe for the exact r_frame_rate fraction string (e.g. "1979/50")
    # because VideoFileParser returns a float which loses the fraction.
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(ref_path),
        ],
        capture_output=True,
        text=True,
    )
    fps_str = result.stdout.strip()
    if not fps_str or result.returncode != 0:
        raise MP4RecoveryError(f"Could not detect framerate from {ref_path}")

    # Also probe via VideoFileParser for structured info
    vfp = VideoFileParser(str(ref_path))
    return fps_str, vfp.fps, vfp.frame_count


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def recover_one(
    src: Path,
    out: Path,
    vol_header: bytes,
    framerate: str,
    tmpdir: Path,
) -> tuple[bool, str]:
    """Recover a single corrupted MP4. Returns (success, message)."""
    _ensure_ffmpeg()

    tmp_m4v = tmpdir / f"{src.stem}.m4v"
    with open(src, "rb") as s, open(tmp_m4v, "wb") as dst:
        dst.write(vol_header)
        s.seek(CONTAINER_HEADER_SIZE)
        shutil.copyfileobj(s, dst)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "m4v",
        "-r",
        framerate,
        "-i",
        str(tmp_m4v),
        "-c:v",
        "copy",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_m4v.unlink(missing_ok=True)

    if result.returncode != 0:
        return False, result.stderr.strip().split("\n")[-1]

    try:
        vfp = VideoFileParser(str(out))
        return True, f"{vfp.frame_count} frames, {vfp.duration:.1f}s"
    except Exception as exc:
        return False, f"ffprobe verification failed: {exc}"


def _probe_fixed_file(path: Path) -> tuple[int, float]:
    """Probe an already-recovered file for frame count and duration."""
    try:
        vfp = VideoFileParser(str(path))
        return vfp.frame_count, round(vfp.duration, 2)
    except Exception:
        return 0, 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _write_report(
    records: list[dict],
    input_dir: Path,
    outdir: Path,
    good_count: int,
    total_mp4s: int,
) -> None:
    """Write recovery_report.json and recovery_report.csv to outdir."""
    recovered = [r for r in records if r["status"] == "recovered"]
    failed = [r for r in records if r["status"] == "failed"]
    empty = [r for r in records if r["status"] == "empty_stub"]
    pending = [r for r in records if r["status"] == "pending"]

    total_source_mb = sum(r["source_mb"] for r in records)
    total_recovered_frames = sum(r["recovered_frames"] for r in recovered)
    total_recovered_duration = sum(r["recovered_duration_s"] for r in recovered)

    groups = sorted(set(r["timestamp_group"] for r in records))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(input_dir),
        "output_dir": str(outdir),
        "cameras": sorted(set(r["camera"] for r in records)),
        "scan": {
            "total_mp4s_in_directory": total_mp4s,
            "good_files": good_count,
            "corrupted_files": len(records),
        },
        "recovery": {
            "recovered": len(recovered),
            "failed": len(failed),
            "empty_stubs": len(empty),
            "pending": len(pending),
            "total_source_mb": round(total_source_mb, 1),
            "total_recovered_frames": total_recovered_frames,
            "total_recovered_duration_s": round(total_recovered_duration, 2),
            "total_recovered_duration_min": round(total_recovered_duration / 60, 1),
            "corrupted_recording_groups": len(groups),
            "recording_groups": groups,
        },
        "files": records,
    }

    json_path = outdir / "recovery_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Report: %s", json_path)

    csv_path = outdir / "recovery_report.csv"
    fieldnames = [
        "file",
        "camera",
        "timestamp_group",
        "status",
        "source_bytes",
        "source_mb",
        "recovered_frames",
        "recovered_duration_s",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info("Report: %s", csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing MP4 files (good + corrupted)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write recovered *_fixed.mp4 files and reports",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually recover (default is dry-run)",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="",
        help="Only recover files from this camera serial",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    configure_standalone_logging(level=args.log_level)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        log.error("%s is not a directory", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Scan ----
    log.info("Scanning %s ...", input_dir)
    corrupted, good, ref_by_cam = scan_directory(input_dir)
    total_mp4s = len(corrupted) + len(good)

    log.info(
        "Total MP4s: %d  |  Good: %d  |  Corrupted: %d",
        total_mp4s,
        len(good),
        len(corrupted),
    )
    log.info("Cameras: %s", ", ".join(sorted(ref_by_cam.keys())) or "none detected")

    if not corrupted:
        log.info("No corrupted files found. Nothing to do.")
        sys.exit(0)

    if not ref_by_cam:
        log.error("No good reference files found. Cannot recover.")
        sys.exit(1)

    # ---- Extract VOL + detect framerate from first reference ----
    first_ref = next(iter(ref_by_cam.values()))
    vol_header = extract_vol_header(first_ref)
    framerate_str, fps_float, _ = detect_reference_info(first_ref)
    log.info("VOL header: %d bytes (from %s)", len(vol_header), first_ref.name)
    log.info("Framerate: %s (%.2f fps)", framerate_str, fps_float)

    # Check which cameras lack references
    corrupted_cams = set(_cam_from_path(p) for p in corrupted)
    missing_ref_cams = corrupted_cams - set(ref_by_cam.keys())
    if missing_ref_cams:
        log.warning(
            "No reference file for camera(s): %s — files will be skipped.",
            ", ".join(sorted(missing_ref_cams)),
        )

    log.info("Output: %s", output_dir)
    log.info("Mode: %s", "LIVE" if args.run else "DRY RUN (use --run to execute)")

    # ---- Process ----
    n_recovered = 0
    n_failed = 0
    n_empty = 0
    n_already = 0
    n_no_ref = 0
    file_records: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="mp4recover_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for src in corrupted:
            fname = src.name
            cam = _cam_from_path(src)
            seg_id = _segment_id_from_path(src)

            if args.cam and cam != args.cam:
                continue

            stem_no_ext = fname.rsplit(".", 1)[0]
            out = output_dir / f"{stem_no_ext}_fixed.mp4"
            fsize = src.stat().st_size
            size_mb = fsize / 1024 / 1024

            record = {
                "file": fname,
                "camera": cam,
                "timestamp_group": seg_id,
                "source_bytes": fsize,
                "source_mb": round(size_mb, 1),
            }

            with log_context(seg=seg_id, cam=cam):
                # No reference for this camera
                if cam not in ref_by_cam:
                    log.info("SKIP (no ref): %s", fname)
                    n_no_ref += 1
                    record.update(
                        status="no_reference",
                        recovered_frames=0,
                        recovered_duration_s=0,
                    )
                    file_records.append(record)
                    continue

                # Empty stub
                if fsize < EMPTY_STUB_THRESHOLD:
                    log.info("SKIP (empty %dB stub): %s", fsize, fname)
                    n_empty += 1
                    record.update(
                        status="empty_stub", recovered_frames=0, recovered_duration_s=0
                    )
                    file_records.append(record)
                    continue

                # Already recovered
                if out.exists():
                    log.info("SKIP (exists): %s", fname)
                    n_already += 1
                    frames, dur = _probe_fixed_file(out)
                    record.update(
                        status="recovered",
                        recovered_frames=frames,
                        recovered_duration_s=dur,
                    )
                    file_records.append(record)
                    continue

                # Dry run
                if not args.run:
                    log.info("WOULD RECOVER: %s  (%.1f MB)", fname, size_mb)
                    n_recovered += 1
                    record.update(
                        status="pending", recovered_frames=0, recovered_duration_s=0
                    )
                    file_records.append(record)
                    continue

                # Recover
                log.info("RECOVERING: %s  (%.1f MB) ...", fname, size_mb)
                ok, msg = recover_one(src, out, vol_header, framerate_str, tmpdir_path)
                if ok:
                    log.info("OK: %s — %s", fname, msg)
                    n_recovered += 1
                    parts = msg.split(", ")
                    frames = int(parts[0].split()[0]) if parts else 0
                    dur = float(parts[1].rstrip("s")) if len(parts) > 1 else 0
                    record.update(
                        status="recovered",
                        recovered_frames=frames,
                        recovered_duration_s=round(dur, 2),
                    )
                else:
                    log.error("FAILED: %s — %s", fname, msg)
                    out.unlink(missing_ok=True)
                    n_failed += 1
                    record.update(
                        status="failed",
                        recovered_frames=0,
                        recovered_duration_s=0,
                        error=msg,
                    )

            file_records.append(record)

    # ---- Report ----
    if file_records and (args.run or n_already > 0):
        _write_report(file_records, input_dir, output_dir, len(good), total_mp4s)

    log.info("=== Results ===")
    label = "Recovered" if args.run else "Would recover"
    log.info(
        "%s: %d  |  Failed: %d  |  Empty stubs: %d  |  No reference: %d  |  Already done: %d",
        label,
        n_recovered,
        n_failed,
        n_empty,
        n_no_ref,
        n_already,
    )


if __name__ == "__main__":
    main()
