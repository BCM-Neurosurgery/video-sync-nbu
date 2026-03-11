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
from typing import Any

from scripts.errors import FFmpegNotFoundError, MP4RecoveryError
from scripts.index.filepatterns import FilePatterns
from scripts.log.logutils import configure_standalone_logging, log_context
from scripts.parsers.videofileparser import VideoFileParser

__all__ = [
    "cam_from_path",
    "scan_directory",
    "plan_recovery",
    "recover_one",
    "run_recovery",
    "main",
]

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


def cam_from_path(path: Path) -> str:
    """Public camera parser used by other modules."""
    return _cam_from_path(path)


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
# Public API
# ---------------------------------------------------------------------------


def _parse_recover_message(msg: str) -> tuple[int, float]:
    parts = msg.split(", ")
    try:
        frames = int(parts[0].split()[0]) if parts else 0
    except (ValueError, IndexError):
        frames = 0
    try:
        dur = float(parts[1].rstrip("s")) if len(parts) > 1 else 0.0
    except (ValueError, IndexError):
        dur = 0.0
    return frames, round(dur, 2)


def _classify_corrupted_files(
    corrupted: list[Path],
    ref_by_cam: dict[str, Path],
    output_dir: Path,
    cam: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    pending_entries: list[dict[str, Any]] = []
    counts = {
        "already_done": 0,
        "pending": 0,
        "empty_stub": 0,
        "no_reference": 0,
    }
    targeted_total = 0

    for src in corrupted:
        camera = _cam_from_path(src)
        if cam and camera != cam:
            continue

        targeted_total += 1
        out = output_dir / f"{src.stem}_fixed.mp4"
        fsize = src.stat().st_size
        size_mb = fsize / 1024 / 1024
        seg_id = _segment_id_from_path(src)

        record: dict[str, Any] = {
            "file": src.name,
            "camera": camera,
            "timestamp_group": seg_id,
            "source_bytes": fsize,
            "source_mb": round(size_mb, 1),
            "recovered_frames": 0,
            "recovered_duration_s": 0,
        }

        if camera not in ref_by_cam:
            counts["no_reference"] += 1
            record["status"] = "no_reference"
        elif fsize < EMPTY_STUB_THRESHOLD:
            counts["empty_stub"] += 1
            record["status"] = "empty_stub"
        elif out.exists():
            counts["already_done"] += 1
            frames, dur = _probe_fixed_file(out)
            record["status"] = "recovered"
            record["recovered_frames"] = frames
            record["recovered_duration_s"] = dur
        else:
            counts["pending"] += 1
            record["status"] = "pending"
            pending_entries.append({"src": src, "out": out, "record": record})

        records.append(record)

    return {
        "records": records,
        "pending_entries": pending_entries,
        "counts": counts,
        "targeted_corrupted_files": targeted_total,
    }


def plan_recovery(
    input_dir: Path,
    output_dir: Path,
    *,
    cam: str = "",
) -> dict[str, Any]:
    """Build a reusable recovery plan without mutating files."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")

    corrupted, good, ref_by_cam = scan_directory(input_dir)
    total_mp4s = len(corrupted) + len(good)

    classified = _classify_corrupted_files(corrupted, ref_by_cam, output_dir, cam)
    counts = classified["counts"]
    targeted_corrupted = int(classified["targeted_corrupted_files"])

    status = "ok"
    error = ""
    if total_mp4s == 0:
        status = "no_mp4"
    elif not corrupted:
        status = "no_corrupted"
    elif not ref_by_cam:
        status = "no_reference_files"
        error = "No good reference files found. Cannot recover."
    elif targeted_corrupted == 0:
        status = "no_target_corruption"

    vol_header = b""
    framerate_str = ""
    fps_float = 0.0
    if status == "ok" and counts["pending"] > 0:
        first_ref = next(iter(ref_by_cam.values()))
        vol_header = extract_vol_header(first_ref)
        framerate_str, fps_float, _ = detect_reference_info(first_ref)

    return {
        "status": status,
        "error": error,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "cam": cam,
        "scan": {
            "total_mp4s": total_mp4s,
            "good_files": len(good),
            "corrupted_files": len(corrupted),
            "cameras": sorted(ref_by_cam.keys()),
        },
        "counts": counts,
        "targeted_corrupted_files": targeted_corrupted,
        "records": classified["records"],
        "pending_entries": classified["pending_entries"],
        "good_count": len(good),
        "vol_header": vol_header,
        "framerate_str": framerate_str,
        "fps_float": fps_float,
    }


def run_recovery(
    input_dir: Path,
    output_dir: Path,
    *,
    run: bool = False,
    cam: str = "",
    write_report: bool = True,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run MP4 recovery for one directory and return structured results."""
    if plan is None:
        input_dir = input_dir.resolve()
        if not input_dir.is_dir():
            raise NotADirectoryError(f"{input_dir} is not a directory")
        log.info("Scanning %s ...", input_dir)
        plan = plan_recovery(input_dir=input_dir, output_dir=output_dir, cam=cam)
        scan = plan["scan"]
        log.info(
            "Total MP4s: %d  |  Good: %d  |  Corrupted: %d",
            scan["total_mp4s"],
            scan["good_files"],
            scan["corrupted_files"],
        )
        log.info("Cameras: %s", ", ".join(scan["cameras"]) or "none detected")

    input_dir = Path(plan["input_dir"])
    output_dir = Path(plan["output_dir"])
    scan = plan["scan"]
    counts_plan = plan["counts"]
    file_records: list[dict[str, Any]] = list(plan["records"])
    pending_entries: list[dict[str, Any]] = list(plan["pending_entries"])
    status = str(plan["status"])
    error = str(plan.get("error", ""))
    targeted_corrupted = int(plan.get("targeted_corrupted_files", 0))

    result: dict[str, Any] = {
        "status": status,
        "error": error,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": "run" if run else "dry_run",
        "scan": scan,
        "targeted_corrupted_files": targeted_corrupted,
        "counts": {
            "recovered": 0,
            "would_recover": 0,
            "failed": 0,
            "empty_stub": 0,
            "no_reference": 0,
            "already_done": 0,
            "pending": 0,
        },
        "records": [],
        "report_written": False,
        "report_json": str(output_dir / "recovery_report.json"),
        "report_csv": str(output_dir / "recovery_report.csv"),
    }

    if status in {"no_mp4", "no_corrupted"}:
        log.info("No corrupted files found. Nothing to do.")
        result["records"] = file_records
        result["counts"] = {
            "recovered": 0,
            "would_recover": 0,
            "failed": 0,
            "empty_stub": counts_plan["empty_stub"],
            "no_reference": counts_plan["no_reference"],
            "already_done": counts_plan["already_done"],
            "pending": counts_plan["pending"],
        }
        return result

    if status == "no_target_corruption":
        result["records"] = file_records
        result["counts"] = {
            "recovered": 0,
            "would_recover": 0,
            "failed": 0,
            "empty_stub": counts_plan["empty_stub"],
            "no_reference": counts_plan["no_reference"],
            "already_done": counts_plan["already_done"],
            "pending": counts_plan["pending"],
        }
        return result

    if status == "no_reference_files":
        log.error(error)
        result["records"] = file_records
        result["counts"] = {
            "recovered": 0,
            "would_recover": 0,
            "failed": 0,
            "empty_stub": counts_plan["empty_stub"],
            "no_reference": counts_plan["no_reference"],
            "already_done": counts_plan["already_done"],
            "pending": counts_plan["pending"],
        }
        return result

    log.info("Output: %s", output_dir)
    log.info("Mode: %s", "LIVE" if run else "DRY RUN (use --run to execute)")

    n_recovered = 0
    n_failed = 0
    n_empty = int(counts_plan["empty_stub"])
    n_already = int(counts_plan["already_done"])
    n_no_ref = int(counts_plan["no_reference"])
    n_pending = int(counts_plan["pending"])

    if not run:
        result["counts"] = {
            "recovered": 0,
            "would_recover": n_pending,
            "failed": 0,
            "empty_stub": n_empty,
            "no_reference": n_no_ref,
            "already_done": n_already,
            "pending": n_pending,
        }
        result["records"] = file_records
        if file_records and n_already > 0 and write_report:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_report(
                file_records,
                input_dir,
                output_dir,
                scan["good_files"],
                scan["total_mp4s"],
            )
            result["report_written"] = True
        log.info("=== Results ===")
        log.info(
            "Would recover: %d  |  Failed: %d  |  Empty stubs: %d  |  No reference: %d  |  Already done: %d",
            n_pending,
            0,
            n_empty,
            n_no_ref,
            n_already,
        )
        return result

    if n_pending > 0:
        vol_header = bytes(plan.get("vol_header", b""))
        framerate_str = str(plan.get("framerate_str", ""))
        fps_float = float(plan.get("fps_float", 0.0))
        if not vol_header or not framerate_str:
            raise MP4RecoveryError(
                f"Recovery plan missing reference metadata for {input_dir}"
            )
        log.info("VOL header: %d bytes", len(vol_header))
        log.info("Framerate: %s (%.2f fps)", framerate_str, fps_float)

    with tempfile.TemporaryDirectory(prefix="mp4recover_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for entry in pending_entries:
            src = Path(entry["src"])
            out = Path(entry["out"])
            record: dict[str, Any] = entry["record"]
            fname = src.name
            seg_id = str(record["timestamp_group"])
            camera = str(record["camera"])
            size_mb = float(record["source_mb"])

            with log_context(seg=seg_id, cam=camera):
                log.info("RECOVERING: %s  (%.1f MB) ...", fname, size_mb)
                ok, msg = recover_one(src, out, vol_header, framerate_str, tmpdir_path)  # type: ignore[arg-type]
                if ok:
                    log.info("OK: %s — %s", fname, msg)
                    n_recovered += 1
                    frames, dur = _parse_recover_message(msg)
                    record.update(
                        status="recovered",
                        recovered_frames=frames,
                        recovered_duration_s=dur,
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
    n_pending_after = sum(1 for r in file_records if r.get("status") == "pending")

    if file_records and write_report:
        _write_report(
            file_records,
            input_dir,
            output_dir,
            scan["good_files"],
            scan["total_mp4s"],
        )
        result["report_written"] = True

    result["counts"] = {
        "recovered": n_recovered,
        "would_recover": 0,
        "failed": n_failed,
        "empty_stub": n_empty,
        "no_reference": n_no_ref,
        "already_done": n_already,
        "pending": n_pending_after,
    }
    result["records"] = file_records

    log.info("=== Results ===")
    log.info(
        "Recovered: %d  |  Failed: %d  |  Empty stubs: %d  |  No reference: %d  |  Already done: %d",
        n_recovered,
        n_failed,
        n_empty,
        n_no_ref,
        n_already,
    )

    return result


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

    try:
        res = run_recovery(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            run=args.run,
            cam=args.cam,
            write_report=True,
        )
    except NotADirectoryError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if res.get("status") == "no_reference_files":
        sys.exit(1)


if __name__ == "__main__":
    main()
