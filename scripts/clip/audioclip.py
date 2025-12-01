#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audioclip.py — Clip channel audios to a common window derived from a CSV, and
               write a 'local' CSV whose sample indices start at 0.

What it does
------------
1) Reads a CSV with columns: serial,start_sample,end_sample
   → finds global s_min, s_max over all rows.
2) Applies a safety margin (default 5s) around [s_min, s_max] using the
   provided sample rate (default 44100 Hz).
3) Clips every audio in `audio_dir` that matches:
      <prefix>-<chan>.(wav|mp3)   where <chan> is two digits (01, 02, 03)
   to that window, writing WAV files named:
      <prefix>-clipped-<chan>.wav
   into `out_dir`.
4) Writes an updated CSV (beside the input CSV) named:
      <input_basename>-local.csv
   with start_sample/end_sample shifted so the clipped window starts at 0.

Logging
-------
- Uses scripts.log.logutils for consistent console/run stamping.
- As a standalone script, use --seg/--cam to stamp messages.
- When run under your driver, the driver configures logging; this module just
  uses a module-level logger.

Public API
----------
from audioclip import (
    ClipWindow,
    clip_from_csv,
    compute_window_from_csv,
    write_shifted_local_csv,
    clip_all_audios,
)

CLI
---
python audioclip.py \
  /path/to/serial_blocks.csv \
  /path/to/audio_dir \
  /path/to/out_dir \
  --sr 44100 \
  --margin-sec 5 \
  [--overwrite] \
  [--log-level INFO] \
  [--seg SEGID] [--cam CAMSERIAL]

Requirements
------------
- ffmpeg available on PATH.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from scripts.log.logutils import configure_standalone_logging, log_context

logger = logging.getLogger(__name__)


AUDIO_NAME_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<chan>\d{2})\.(?P<ext>wav|mp3)$", re.IGNORECASE
)


@dataclass
class ClipWindow:
    """Inclusive-exclusive sample window and CSV sample rate (Hz)."""

    s0: int
    s1: int
    sr_csv: int

    @property
    def start_sec(self) -> float:
        return self.s0 / float(self.sr_csv)

    @property
    def end_sec(self) -> float:
        return self.s1 / float(self.sr_csv)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found on PATH.")


def _list_audio_files(audio_dir: Path) -> List[Path]:
    return [
        p for p in audio_dir.iterdir() if p.is_file() and AUDIO_NAME_RE.match(p.name)
    ]


def _parse_csv_bounds(csv_path: Path) -> Tuple[int, int]:
    smin: Optional[int] = None
    smax: Optional[int] = None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = int((row.get("start_sample") or "").strip())
                b = int((row.get("end_sample") or "").strip())
            except Exception:
                continue
            if smin is None or a < smin:
                smin = a
            if smax is None or b > smax:
                smax = b
    if smin is None or smax is None or smax <= smin:
        raise ValueError(f"No valid start/end samples in {csv_path}")
    return smin, smax


def _build_out_name(in_name: str) -> str:
    m = AUDIO_NAME_RE.match(in_name)
    assert m, f"Bad audio name: {in_name}"
    prefix = m.group("prefix")
    chan = m.group("chan")
    return f"{prefix}-clipped-{chan}.wav"


def _ffmpeg_clip(
    in_path: Path, out_path: Path, start_sec: float, end_sec: float, overwrite: bool
) -> None:
    """
    Use FFmpeg to precisely trim [start_sec, end_sec) and write 16-bit PCM WAV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filt = f"atrim=start={start_sec:.9f}:end={end_sec:.9f},asetpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-vn",
        "-i",
        str(in_path),
        "-af",
        filt,
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    log_path = out_path.with_suffix(f"{out_path.suffix}.ffmpeg.log")
    with log_path.open("w") as ferr:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trim failed for '{in_path.name}'. See log: {log_path}"
        )


# -----------------------------------------------------------------------------
# Public helpers (importable)
# -----------------------------------------------------------------------------
def compute_window_from_csv(
    csv_path: Path, sr_csv: int, margin_sec: float
) -> ClipWindow:
    """
    Compute inclusive-exclusive sample window [s0, s1) from CSV min/max and margin.
    """
    smin, smax = _parse_csv_bounds(csv_path)
    margin_samples = int(round(margin_sec * sr_csv))
    s0 = max(0, smin - margin_samples)
    s1 = smax + margin_samples
    if s1 <= s0:
        raise ValueError(f"Invalid window after margin: [{s0}, {s1})")
    logger.info(
        "Window from CSV: [%d, %d) samples @ %d Hz (%.3f..%.3f s, margin=%.3fs)",
        s0,
        s1,
        sr_csv,
        s0 / sr_csv,
        s1 / sr_csv,
        margin_sec,
    )
    return ClipWindow(s0=s0, s1=s1, sr_csv=sr_csv)


def clip_all_audios(
    audio_dir: Path,
    out_dir: Path,
    window: ClipWindow,
    overwrite: bool = False,
) -> List[Tuple[Path, Path]]:
    """
    Clip every matching audio to `out_dir`. Returns list of (in_path, out_path).
    """
    pairs: List[Tuple[Path, Path]] = []
    files = sorted(_list_audio_files(audio_dir))
    if not files:
        logger.warning("No matching audio files found in %s", audio_dir)
        return pairs
    for p in files:
        out_name = _build_out_name(p.name)
        out_path = out_dir / out_name
        logger.info("Clipping %s → %s", p.name, out_path.name)
        _ffmpeg_clip(p, out_path, window.start_sec, window.end_sec, overwrite=overwrite)
        pairs.append((p, out_path))
    return pairs


def write_shifted_local_csv(csv_in: Path, csv_out: Path, shift_by_samples: int) -> int:
    """
    Write a '-local.csv' with start/end shifted by `shift_by_samples` (clipped timeline).
    Returns number of written rows (excluding header).
    """
    count = 0
    with (
        csv_in.open("r", newline="", encoding="utf-8") as fin,
        csv_out.open("w", newline="", encoding="utf-8") as fout,
    ):
        rin = csv.DictReader(fin)
        wout = csv.DictWriter(fout, fieldnames=["serial", "start_sample", "end_sample"])
        wout.writeheader()
        for row in rin:
            try:
                serial = int((row.get("serial") or "").strip())
                s = int((row.get("start_sample") or "").strip())
                e = int((row.get("end_sample") or "").strip())
            except Exception:
                continue
            s_local = max(0, s - shift_by_samples)
            e_local = max(0, e - shift_by_samples)
            wout.writerow(
                {"serial": serial, "start_sample": s_local, "end_sample": e_local}
            )
            count += 1
    logger.info("Wrote shifted CSV (%d rows) → %s", count, csv_out.name)
    return count


def clip_from_csv(
    csv_path: Path,
    audio_dir: Path,
    out_dir: Path,
    *,
    sr: int = 44100,
    margin_sec: float = 5.0,
    overwrite: bool = False,
) -> Tuple[ClipWindow, List[Tuple[Path, Path]], Path]:
    """
    High-level one-shot: compute window → clip audios → write -local.csv.

    Parameters
    ----------
    sr : int
        Sample rate (Hz) used by the CSV's indices. Default 44100.

    Returns
    -------
    (window, pairs, local_csv_path)
    """
    _require_tool("ffmpeg")

    window = compute_window_from_csv(csv_path, sr, margin_sec)

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = clip_all_audios(audio_dir, out_dir, window, overwrite=overwrite)

    local_csv = csv_path.with_name(csv_path.stem + "-local.csv")
    write_shifted_local_csv(csv_path, local_csv, window.s0)

    return window, pairs, local_csv


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Clip channel audios to a CSV-derived window and write a shifted '-local.csv'."
    )
    ap.add_argument(
        "csv", type=Path, help="CSV with columns serial,start_sample,end_sample"
    )
    ap.add_argument(
        "audio_dir", type=Path, help="Directory with <prefix>-<chan>.(wav|mp3)"
    )
    ap.add_argument("out_dir", type=Path, help="Directory to write clipped WAVs")
    ap.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate (Hz) of CSV indices (default: 44100)",
    )
    ap.add_argument(
        "--margin-sec",
        type=float,
        default=5.0,
        help="Safety margin in seconds (default: 5.0)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting existing outputs"
    )
    ap.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    # Standalone stamping (useful when not under the driver)
    ap.add_argument(
        "--seg", default="-", help="Segment ID stamp for logs (standalone mode)"
    )
    ap.add_argument(
        "--cam", default="-", help="Camera serial stamp for logs (standalone mode)"
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    # Standalone console logging (no-op if driver already configured root)
    configure_standalone_logging(args.log_level, seg=args.seg, cam=args.cam)

    csv_path: Path = args.csv.resolve()
    audio_dir: Path = args.audio_dir.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not audio_dir.exists():
        raise FileNotFoundError(audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with log_context(seg=args.seg, cam=args.cam):
        logger.info("Starting audio clip process.")
        window, pairs, local_csv = clip_from_csv(
            csv_path,
            audio_dir,
            out_dir,
            sr=args.sr,
            margin_sec=args.margin_sec,
            overwrite=args.overwrite,
        )
        logger.info(
            "Clipped %d file(s) to [%.3f, %.3f] s.",
            len(pairs),
            window.start_sec,
            window.end_sec,
        )
        logger.info("Shifted CSV written to: %s", local_csv)
        logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
