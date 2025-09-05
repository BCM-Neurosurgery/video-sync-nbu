#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mp3split.py — Split a long MP3 into fixed-length WAV chunks (default: 1 hour)

- Uses FFmpeg's segment muxer; decodes to WAV (pcm_s16le) per chunk.
- Public API: split_mp3_to_wav(...)
- CLI: python -m scripts.split.mp3split INPUT.mp3 [--outdir DIR] [--chunk-seconds 3600]
       [--overwrite | --clean] [-v] [--seg SEG] [--cam CAM] [--print-paths]
- Logs a readable line RIGHT AFTER each chunk is finished (size stabilized).
- Writes a manifest JSON alongside the chunks with absolute offsets.

Outputs: "<stem>-NNN.wav" and "<stem>_manifest.json" in the output folder (001, 002, …).

Manifest schema (summary):
{
  "input": "<abs path to original mp3>",
  "output_dir": "<abs outdir>",
  "sample_rate_hz": 44100,
  "channels": 1,
  "chunk_seconds": 3600,
  "start_number": 1,
  "overwrite": false,
  "clean": true,
  "segments": [
    {"file": "…-001.wav", "start_sample": 0, "num_samples": 158760000},
    {"file": "…-002.wav", "start_sample": 158760000, "num_samples": 158760000}
  ]
}
"""


from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import time
import re
import json
import wave
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.log.logutils import configure_standalone_logging, log_context
from scripts.errors import FFmpegNotFoundError, SplitFailureError

__all__ = [
    "split_mp3_to_wav",
    "FFmpegNotFoundError",
    "SplitFailureError",
    "main",
]

log = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------
def _ensure_ffmpeg_available(ffmpeg_bin: str = "ffmpeg") -> str:
    path = shutil.which(ffmpeg_bin)
    if not path:
        raise FFmpegNotFoundError(
            f"ffmpeg binary not found: {ffmpeg_bin!r}. Install FFmpeg or pass --ffmpeg-bin."
        )
    return path


def _build_ffmpeg_cmd(
    ffmpeg_bin: str,
    input_mp3: Path,
    out_pattern: Path,
    chunk_seconds: int,
    start_number: int,
    overwrite: bool,
    ffmpeg_loglevel: str,
) -> List[str]:
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        ffmpeg_loglevel,  # "error" | "warning" | "info" | "debug"
        "-y" if overwrite else "-n",
        "-i",
        str(input_mp3),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(int(chunk_seconds)),
        "-segment_start_number",
        str(int(start_number)),
        "-segment_format",
        "wav",
        "-reset_timestamps",
        "1",
        str(out_pattern),
    ]


def _verbosity_to_levels(v: int) -> Tuple[str, int]:
    if v >= 2:
        return "debug", logging.DEBUG
    if v == 1:
        return "info", logging.INFO
    return "info", logging.INFO


def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _detect_completed_chunks(
    outdir: Path,
    stem: str,
    known: set[Path],
    pending_sizes: Dict[Path, int],
) -> List[Tuple[Path, int, int]]:
    """
    Detect NEW chunks whose sizes have stabilized across two polls.
    Returns [(path, index, size_bytes), ...] sorted by index.
    """
    completed: List[Tuple[Path, int, int]] = []
    pattern = re.compile(rf"^{re.escape(stem)}-(\d{{3}})\.wav$")
    for p in outdir.glob(f"{stem}-[0-9][0-9][0-9].wav"):
        if p in known:
            continue
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            continue
        prev = pending_sizes.get(p)
        if prev is not None and size == prev and size > 0:
            m = pattern.match(p.name)
            idx = int(m.group(1)) if m else -1
            completed.append((p, idx, size))
            known.add(p)
            pending_sizes.pop(p, None)
        else:
            pending_sizes[p] = size
    return sorted(completed, key=lambda t: t[1])


def _delete_matching_chunks(outdir: Path, stem: str) -> int:
    """Delete all existing '<stem>-NNN.wav' files. Returns count removed."""
    count = 0
    for p in outdir.glob(f"{stem}-[0-9][0-9][0-9].wav"):
        try:
            p.unlink()
            count += 1
        except FileNotFoundError:
            pass
    return count


def _wav_frames_and_rate(path: Path) -> Tuple[int, int, int]:
    """Return (nframes, samplerate, nchannels) for a WAV file."""
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        return wf.getnframes(), wf.getframerate(), wf.getnchannels()


# -----------------------------
# Public API
# -----------------------------
def split_mp3_to_wav(
    input_mp3: str | Path,
    outdir: Optional[str | Path] = None,
    *,
    chunk_seconds: int = 3600,
    start_number: int = 1,
    overwrite: bool = False,
    clean: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    ffmpeg_loglevel: str = "info",
    poll_interval: float = 0.5,
) -> List[Path]:
    """
    Split an MP3 file into fixed-length WAV chunks using FFmpeg and log each chunk as it finishes.

    Policy
    ------
    - Default (safe): if any '<stem>-NNN.wav' already exist in outdir → FAIL with guidance.
    - '--clean': delete matching chunks before running, then proceed.
    - '--overwrite': allow ffmpeg to overwrite matching files; may leave stale extras.

    Parameters
    ----------
    input_mp3 : str | Path
        Input MP3 path.
    outdir : str | Path, optional
        Output directory (default: "<stem>_chunks" alongside input).
    chunk_seconds : int, default 3600
        Chunk size in seconds.
    start_number : int, default 1
        Starting index for "<stem>-NNN.wav".
    overwrite : bool, default False
        Pass '-y' to ffmpeg to overwrite matching files.
    clean : bool, default False
        Delete matching '<stem>-NNN.wav' before splitting.
    ffmpeg_bin : str, default "ffmpeg"
        ffmpeg executable.
    ffmpeg_loglevel : str, default "info"
        FFmpeg log level string.
    poll_interval : float, default 0.5
        How often to poll the output directory for completed chunks.

    Returns
    -------
    List[Path]
        List of all produced chunk paths (sorted).
    """
    input_path = Path(input_mp3)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if chunk_seconds < 1:
        raise ValueError("chunk_seconds must be >= 1")
    if start_number < 0:
        raise ValueError("start_number must be >= 0")

    ffmpeg_path = _ensure_ffmpeg_available(ffmpeg_bin)

    outdir_path = (
        input_path.parent / f"{input_path.stem}_chunks"
        if outdir is None
        else Path(outdir)
    )
    outdir_path.mkdir(parents=True, exist_ok=True)

    out_pattern = outdir_path / f"{input_path.stem}-%03d.wav"

    log.info("Input: %s", input_path)
    log.info("Output dir: %s", outdir_path)
    log.info("Chunk length: %s seconds", chunk_seconds)

    # ---- Existing-file policy gate ----
    existing = sorted(outdir_path.glob(f"{input_path.stem}-[0-9][0-9][0-9].wav"))
    if clean:
        removed = _delete_matching_chunks(outdir_path, input_path.stem)
        if removed:
            log.info("Clean mode: removed %d existing chunk(s).", removed)
        existing = []  # after clean, treat as empty
    elif existing and not overwrite:
        raise SplitFailureError(
            f"Found {len(existing)} existing chunk(s) matching pattern '{input_path.stem}-NNN.wav' in {outdir_path}.\n"
            f"Choose one: use --clean to delete them first, use --overwrite to replace, or pick a new --outdir."
        )

    # Logging dedupe policy:
    # - overwrite=True → start with known = empty to log overwritten chunks too
    # - otherwise → seed known with whatever existed pre-run
    known: set[Path] = set() if overwrite else set(existing)
    pending_sizes: Dict[Path, int] = {}

    cmd = _build_ffmpeg_cmd(
        ffmpeg_bin=ffmpeg_path,
        input_mp3=input_path,
        out_pattern=out_pattern,
        chunk_seconds=chunk_seconds,
        start_number=start_number,
        overwrite=overwrite,
        ffmpeg_loglevel=ffmpeg_loglevel,
    )

    log.debug("Running ffmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        while True:
            # Detect newly completed chunks and log immediately.
            for p, idx, size in _detect_completed_chunks(
                outdir_path, input_path.stem, known, pending_sizes
            ):
                log.info("Saved chunk %03d (%s): %s", idx, _fmt_bytes(size), p.name)

            if proc.poll() is not None:
                # Final sweep after ffmpeg exits.
                for p, idx, size in _detect_completed_chunks(
                    outdir_path, input_path.stem, known, pending_sizes
                ):
                    log.info("Saved chunk %03d (%s): %s", idx, _fmt_bytes(size), p.name)
                break

            time.sleep(poll_interval)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    if proc.returncode != 0:
        raise SplitFailureError(f"ffmpeg failed with exit code {proc.returncode}")

    produced = sorted(outdir_path.glob(f"{input_path.stem}-[0-9][0-9][0-9].wav"))
    if not produced:
        raise SplitFailureError("No chunks produced. Check the input and arguments.")
    log.info("Produced %d chunk(s).", len(produced))

    # ---- Write manifest with absolute start_sample per chunk ----
    segments: List[Dict[str, int | str]] = []
    offset = 0
    sr_seen: Optional[int] = None
    ch_seen: Optional[int] = None

    for p in produced:
        try:
            nframes, sr, nch = _wav_frames_and_rate(p)
        except Exception as e:
            log.warning("Failed to read WAV header for %s: %s", p.name, e)
            continue

        if sr_seen is None:
            sr_seen = sr
            ch_seen = nch
        else:
            if sr != sr_seen:
                log.warning(
                    "Sample rate mismatch: %s has %d Hz, expected %d Hz",
                    p.name,
                    sr,
                    sr_seen,
                )
            if nch != ch_seen:
                log.warning(
                    "Channel count mismatch: %s has %d ch, expected %d ch",
                    p.name,
                    nch,
                    ch_seen,
                )

        segments.append(
            {
                "file": p.name,
                "start_sample": int(offset),  # absolute offset from original start
                "num_samples": int(
                    nframes
                ),  # frames in this chunk (per channel frame count)
            }
        )
        offset += nframes

    manifest = {
        "input": str(input_path.resolve()),
        "output_dir": str(outdir_path.resolve()),
        "sample_rate_hz": sr_seen,
        "channels": ch_seen,
        "chunk_seconds": int(chunk_seconds),
        "start_number": int(start_number),
        "overwrite": bool(overwrite),
        "clean": bool(clean),
        "segments": segments,
    }
    manifest_path = outdir_path / f"{input_path.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote manifest: %s", manifest_path.name)

    return produced


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a long MP3 into fixed-length WAV chunks using FFmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Path to the input MP3 file.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help='Output dir (default: "<stem>_chunks").',
    )
    p.add_argument(
        "--chunk-seconds", type=int, default=3600, help="Chunk size in seconds."
    )
    p.add_argument(
        "--start-number", type=int, default=1, help="Starting index for numbering."
    )
    p.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="Path/name of the ffmpeg executable.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v / -vv).",
    )
    p.add_argument("--seg", default="-", help="Segment ID for log stamping.")
    p.add_argument("--cam", default="-", help="Camera serial for log stamping.")
    p.add_argument(
        "--print-paths",
        action="store_true",
        help="Also print each output path at the end (off by default).",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite matching existing chunk files if present.",
    )
    group.add_argument(
        "--clean",
        action="store_true",
        help="Delete matching existing chunk files before splitting.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ffmpeg_loglevel, py_level = _verbosity_to_levels(args.verbose)

    # Standalone console logger only if the root has no handlers (driver-safe).
    configure_standalone_logging(level=py_level, seg=args.seg, cam=args.cam)

    try:
        with log_context(seg=args.seg, cam=args.cam):
            produced = split_mp3_to_wav(
                input_mp3=args.input,
                outdir=args.outdir,
                chunk_seconds=args.chunk_seconds,
                start_number=args.start_number,
                overwrite=args.overwrite,
                clean=args.clean,
                ffmpeg_bin=args.ffmpeg_bin,
                ffmpeg_loglevel=ffmpeg_loglevel,
            )
            if args.print_paths:
                for p in produced:
                    print(p)
    except (FFmpegNotFoundError, SplitFailureError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
