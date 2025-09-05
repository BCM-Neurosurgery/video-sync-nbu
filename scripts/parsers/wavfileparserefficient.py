#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wavfileparserefficient.py — Streaming, two-pass decoder for frame IDs embedded in audio,
with project logging integration + progress + memory usage.

- Streaming FFmpeg decode (handles arbitrarily long MP3/WAV)
- Pass 1: global min/max via streaming (with progress)
- Pass 2: normalize→binarize→fixed-window sampler (with progress)
- Outputs:
    raw.csv  (serial,start_sample,end_sample)
    raw_info.txt

Logging
-------
- If running under the pipeline driver that already configured logging: reuse it.
- If running standalone: we install a minimal console logger via
  `scripts.log.logutils.configure_standalone_logging(level, seg, cam)`.
- All records are stamped as "[seg/cam]" using `log_context(seg, cam)`.

Usage
-----
python wavfileparserefficient.py /path/to/audio.(mp3|wav) \
    --site jamail \
    --threshold 0.5 \
    --outdir /path/to/output_dir \
    [--log-level INFO] [--seg SEGID] [--cam CAMSERIAL] \
    [--progress auto|bar|log|none] [--progress-interval 5] [--mem-interval 10]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from collections import deque

from scripts.log.logutils import (
    configure_standalone_logging,
    log_context,
)


# Try tqdm lazily only if needed
def _try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


# ---------------------------- Site presets (same as original) ----------------------------

BLOCK_PRESETS: Dict[str, Dict[str, object]] = {
    "jamail": {
        "flip_signal": True,
        "flip_window": True,
        "window_samples": 231,
        "block_stride": 1100,
        "transition_points_1b": [6, 53, 100, 147, 194],
        "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],
    },
    "nbu_sleep": {
        "flip_signal": True,
        "flip_window": True,
        "window_samples": 231,
        "block_stride": 1100,
        "transition_points_1b": [6, 53, 100, 147, 194],
        "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],
    },
    "nbu_lounge": {
        "flip_signal": True,
        "flip_window": True,
        "window_samples": 231,
        "block_stride": 1100,
        "transition_points_1b": [6, 53, 100, 147, 194],
        "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],
    },
}

# ---------------------------- Logger ----------------------------
log = logging.getLogger("sync")

# ---------------------------- Data classes ----------------------------


@dataclass
class DecodeStats:
    bytes_total: int
    starts_total: int
    flips: bool
    best_offset: int
    monotonic_span: int


# ---------------------------- Utils: memory + human formatting ----------------------------


def _fmt_bytes(b: int) -> str:
    # human-friendly binary units
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(b)
    for u in units:
        if abs(n) < 1024.0 or u == units[-1]:
            return f"{n:.1f} {u}"
        n /= 1024.0
    return f"{b} B"


def _rss_and_peak() -> Tuple[int, Optional[int]]:
    """
    Return (rss_bytes, peak_bytes or None). Tries psutil, then resource, then /proc/self/status.
    """
    # 1) psutil
    try:
        import psutil  # type: ignore

        p = psutil.Process()
        mem = p.memory_info()
        rss = int(mem.rss)
        peak = getattr(mem, "peak_wset", None) or getattr(mem, "peak_pagefile", None)
        # psutil peak fields are platform-specific; often None on Linux
        return rss, int(peak) if peak is not None else None
    except Exception:
        pass

    # 2) resource (ru_maxrss: kilobytes on Linux, bytes on macOS)
    try:
        import resource  # type: ignore

        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss = None  # no current RSS via resource
        peak = ru.ru_maxrss
        # Heuristic: Linux returns KiB, macOS returns bytes
        if sys.platform != "darwin":
            peak = int(peak) * 1024
        else:
            peak = int(peak)
        # Try /proc for current
        cur = _rss_from_proc_status()
        return cur if cur is not None else 0, peak
    except Exception:
        pass

    # 3) /proc/self/status (Linux)
    cur = _rss_from_proc_status()
    return cur if cur is not None else 0, None


def _rss_from_proc_status() -> Optional[int]:
    try:
        with open("/proc/self/status", "r") as f:
            text = f.read()
        m = re.search(r"VmRSS:\s+(\d+)\s+kB", text)
        if m:
            return int(m.group(1)) * 1024
    except Exception:
        pass
    return None


# ---------------------------- FFmpeg helpers ----------------------------


def require_ffmpeg() -> Tuple[str, str]:
    """Locate ffmpeg and ffprobe on PATH; raise if missing."""
    ffmpeg = shutil_which("ffmpeg")
    ffprobe = shutil_which("ffprobe")
    if not ffmpeg:
        log.error("ffmpeg not found on PATH.")
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")
    if not ffprobe:
        log.error("ffprobe not found on PATH.")
        raise RuntimeError("ffprobe not found on PATH. Please install ffprobe.")
    return ffmpeg, ffprobe


def shutil_which(name: str) -> Optional[str]:
    from shutil import which

    return which(name)


def ffprobe_sample_rate(input_path: str | Path) -> int:
    """Probe the sample rate of the first audio stream using ffprobe."""
    _, ffprobe = require_ffmpeg()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=nw=1:nk=1",
        str(input_path),
    ]
    log.debug("ffprobe sample_rate: %s", " ".join(map(str, cmd)))
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
    try:
        sr = int(out)
        if sr <= 0:
            raise ValueError("non-positive sample rate")
        return sr
    except Exception as e:
        log.exception("ffprobe could not determine sample_rate (raw=%r)", out)
        raise RuntimeError(
            f"ffprobe could not determine sample_rate (got {out!r})"
        ) from e


def ffprobe_duration_seconds(input_path: str | Path) -> float:
    """Probe container-reported duration (seconds), best-effort."""
    _, ffprobe = require_ffmpeg()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(input_path),
    ]
    try:
        out = (
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
        dur = float(out)
        return dur if dur > 0 else 0.0
    except Exception:
        return 0.0


def spawn_ffmpeg_pcm_pipe(
    input_path: str | Path, force_sample_rate: Optional[int] = None
) -> subprocess.Popen:
    """
    Start ffmpeg to decode `input_path` to mono f32le PCM on stdout.
    If `force_sample_rate` is provided, resample to that rate.
    """
    ffmpeg, _ = require_ffmpeg()
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-vn",
        "-sn",
        "-dn",
        "-i",
        str(input_path),
        "-ac",
        "1",
    ]
    if force_sample_rate:
        args += ["-ar", str(force_sample_rate)]
    args += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    log.debug("spawn ffmpeg: %s", " ".join(map(str, args)))
    return subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=256 * 1024
    )


def iter_pcm_blocks(proc: subprocess.Popen, block_samples: int) -> Iterator[np.ndarray]:
    """
    Stream float32 PCM from ffmpeg stdout in fixed-size blocks of `block_samples`.
    Yields arrays of shape (block_samples,) except possibly the final (short) block.
    """
    assert proc.stdout is not None
    bytes_per_sample = 4  # float32
    chunk_bytes = block_samples * bytes_per_sample
    leftover = b""
    read = proc.stdout.read

    while True:
        buf = read(chunk_bytes - len(leftover))
        if not buf:
            if leftover:
                yield np.frombuffer(leftover, dtype=np.float32)
            break
        data = leftover + buf
        n_full = len(data) // bytes_per_sample
        n_emit = (n_full // block_samples) * block_samples
        if n_emit:
            emit = np.frombuffer(data[: n_emit * bytes_per_sample], dtype=np.float32)
            for i in range(0, emit.size, block_samples):
                yield emit[i : i + block_samples]
            leftover = data[n_emit * bytes_per_sample :]
        else:
            leftover = data


# ---------------------------- Progress helpers ----------------------------


def _should_bar(progress_mode: str) -> bool:
    if progress_mode == "bar":
        return True
    if progress_mode == "auto":
        try:
            return sys.stderr.isatty()
        except Exception:
            return False
    return False


class _LogProgress:
    """Throttled progress via logger with memory usage in messages."""

    def __init__(
        self,
        total: Optional[int],
        label: str,
        interval: float,
        mem_interval: float,
        sample_rate: Optional[int],
    ):
        self.total = total
        self.label = label
        self.interval = max(0.25, float(interval))
        self.mem_interval = max(0.5, float(mem_interval))
        self.sample_rate = sample_rate
        self.done = 0
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._last_mem = self._t0

    def update(self, n: int):
        self.done += int(n)
        t = time.perf_counter()
        if t - self._last >= self.interval:
            pct = f"{(100.0 * self.done / self.total):.1f}%" if self.total else "…"
            dur = t - self._t0
            rss, peak = _rss_and_peak()
            mem_txt = ""
            if t - self._last_mem >= self.mem_interval:
                mem_txt = f" | rss={_fmt_bytes(rss)}" + (
                    f", peak≈{_fmt_bytes(peak)}" if peak else ""
                )
                self._last_mem = t
            if self.sample_rate and self.total:
                secs_done = self.done / float(self.sample_rate)
                secs_total = self.total / float(self.sample_rate)
                log.info(
                    "%s %s (%0.1fs/%0.1fs)%s",
                    self.label,
                    pct,
                    secs_done,
                    secs_total,
                    mem_txt,
                )
            else:
                log.info("%s %s (processed %d)%s", self.label, pct, self.done, mem_txt)
            self._last = t

    def close(self):
        # final line
        if self.total:
            log.info("%s 100.0%% done.", self.label)


# ---------------------------- Core decoder (streaming two-pass) ----------------------------


class StreamingSerialDecoder:
    """
    Streaming two-pass decoder; logs key milestones/timings; supports progress + mem logs.
    """

    def __init__(
        self,
        filepath: str | Path,
        *,
        target_sample_rate: Optional[int] = None,
        block_seconds: float = 2.0,
        ring_seconds: float = 2.0,
        progress_mode: str = "log",  # auto|bar|log|none
        progress_interval: float = 5.0,  # seconds between log updates
        mem_interval: float = 10.0,  # seconds between mem stats in progress logs
    ) -> None:
        self.filepath = str(filepath)
        self._sr_probe = ffprobe_sample_rate(self.filepath)
        self.sample_rate = int(target_sample_rate or self._sr_probe)
        self._dur_probe = ffprobe_duration_seconds(self.filepath)
        self._expected_samples = (
            int(self.sample_rate * self._dur_probe) if self._dur_probe > 0 else None
        )

        # Derived sizes (in samples)
        self.block_samples = max(int(self.sample_rate * block_seconds), 1)
        self.ring_capacity = max(
            int(self.sample_rate * ring_seconds), self.block_samples
        )

        # Stats from pass 1
        self._gmin: float = 0.0
        self._gmax: float = 0.0
        self._n_total: int = 0

        # Outputs from pass 2
        self.frame_ranges: List[Tuple[int, int]] = []
        self.counts: List[int] = []
        self.starts_total: int = 0

        # progress config
        self.progress_mode = progress_mode
        self.progress_interval = progress_interval
        self.mem_interval = mem_interval

        log.info(
            "Init decoder: sr=%d Hz, block=%d samples (%.2fs), ring=%d samples (%.2fs)%s",
            self.sample_rate,
            self.block_samples,
            self.block_samples / self.sample_rate,
            self.ring_capacity,
            self.ring_capacity / self.sample_rate,
            f", dur≈{self._dur_probe:.2f}s" if self._dur_probe else "",
        )

    # -------------------- Pass 1: stats --------------------

    def _pass1_stats(self) -> None:
        log.info("Pass1: streaming min/max & total samples…")
        t0 = time.perf_counter()
        proc = spawn_ffmpeg_pcm_pipe(self.filepath, self.sample_rate)
        gmin = math.inf
        gmax = -math.inf
        n_total = 0

        total_est = self._expected_samples
        use_bar = _should_bar(self.progress_mode)
        tqdm = _try_import_tqdm() if use_bar else None
        pbar = None
        lp = None

        try:
            if use_bar and tqdm is not None and total_est:
                pbar = tqdm(
                    total=total_est,
                    unit="smp",
                    desc="Pass1 (min/max)",
                    leave=False,
                    dynamic_ncols=True,
                )
            elif self.progress_mode in ("auto", "log"):
                lp = _LogProgress(
                    total_est,
                    "Pass1",
                    self.progress_interval,
                    self.mem_interval,
                    self.sample_rate,
                )

            for block in iter_pcm_blocks(proc, self.block_samples):
                if block.size == 0:
                    continue
                n_total += block.size
                bmin = float(block.min())
                bmax = float(block.max())
                if bmin < gmin:
                    gmin = bmin
                if bmax > gmax:
                    gmax = bmax

                if pbar:
                    pbar.update(block.size)
                    # put mem in postfix occasionally
                    if (
                        int(pbar.n)
                        % (self.sample_rate * max(1, int(self.mem_interval)))
                        == 0
                    ):
                        rss, peak = _rss_and_peak()
                        pbar.set_postfix_str(f"rss {_fmt_bytes(rss)}")
                elif lp:
                    lp.update(block.size)

        finally:
            try:
                proc.stdout and proc.stdout.close()
                proc.stderr and proc.stderr.close()
            except Exception:
                pass
            proc.kill()
            if pbar:
                pbar.close()
            if lp:
                lp.close()

        if not math.isfinite(gmin) or not math.isfinite(gmax) or n_total == 0:
            log.warning("Pass1: empty/invalid input (n_total=%d).", n_total)
            self._gmin = 0.0
            self._gmax = 0.0
            self._n_total = 0
            return
        if gmax <= gmin:
            gmax = gmin + 1e-12
        self._gmin = gmin
        self._gmax = gmax
        self._n_total = n_total
        dt = time.perf_counter() - t0
        rss, peak = _rss_and_peak()
        log.info(
            "Pass1: done in %.2fs (min=%+.6f, max=%+.6f, samples=%d, dur=%.2fs, rss=%s%s)",
            dt,
            self._gmin,
            self._gmax,
            self._n_total,
            self._n_total / self.sample_rate if self.sample_rate else 0.0,
            _fmt_bytes(rss),
            f", peak≈{_fmt_bytes(peak)}" if peak else "",
        )

    # -------------------- Helpers for pass 2 --------------------

    @staticmethod
    def _normalize01(
        block: np.ndarray, gmin: float, gmax: float
    ) -> Optional[np.ndarray]:
        if not math.isfinite(gmin) or not math.isfinite(gmax) or gmax <= gmin:
            return None
        return ((block - gmin) / (gmax - gmin)).astype(np.float32)

    @staticmethod
    def _binarize(sig01: np.ndarray, threshold: float) -> np.ndarray:
        return (sig01 > threshold).astype(np.uint8)

    @staticmethod
    def _sample_window(
        win_bits: np.ndarray, trans: Sequence[int], offs7: Sequence[int]
    ) -> Optional[List[int]]:
        bytes5: List[int] = []
        W = win_bits.size
        for t in trans:
            idxs = [t + o for o in offs7]
            if idxs[-1] >= W or idxs[0] < 0:
                return None
            bits = [int(win_bits[j]) for j in idxs]
            bits = bits[::-1]
            val = 0
            for b in bits:
                val = (val << 1) | b
            bytes5.append(val)
        return bytes5

    @staticmethod
    def _concat_bytes(bytes5: Sequence[int]) -> int:
        out = 0
        for b in bytes5[::-1]:
            out = (out << 7) | (b & 0x7F)
        return out

    @staticmethod
    def _longest_plus_one_span(vals: Sequence[int]) -> int:
        if not vals:
            return 0
        best = cur = 1
        for k in range(1, len(vals)):
            if vals[k] - vals[k - 1] == 1:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 1
        return best

    # -------------------- Pass 2: decode --------------------

    def _pass2_decode(self, *, site: str, threshold: float) -> DecodeStats:
        cfg = self._get_cfg(site)
        W = cfg["W"]
        stride = cfg["stride"]
        trans = cfg["trans"]
        offs7 = cfg["offs7"]
        flip_window = cfg["flip_window"]

        ring: Deque[np.uint8] = deque(maxlen=max(self.ring_capacity, W + stride + 8))
        ring_start_global = 0
        starts_total = 0

        log.info(
            "Pass2: decoding (site=%s, threshold=%.3f, W=%d, stride=%d)…",
            site,
            threshold,
            W,
            stride,
        )
        t0 = time.perf_counter()
        proc = spawn_ffmpeg_pcm_pipe(self.filepath, self.sample_rate)

        total_known = self._n_total if self._n_total > 0 else None
        use_bar = _should_bar(self.progress_mode)
        tqdm = _try_import_tqdm() if use_bar else None
        pbar = None
        lp = None
        processed = 0

        try:
            if use_bar and tqdm is not None and total_known:
                pbar = tqdm(
                    total=total_known,
                    unit="smp",
                    desc="Pass2 (decode)",
                    leave=False,
                    dynamic_ncols=True,
                )
            elif self.progress_mode in ("auto", "log"):
                lp = _LogProgress(
                    total_known,
                    "Pass2",
                    self.progress_interval,
                    self.mem_interval,
                    self.sample_rate,
                )

            for pcm_block in iter_pcm_blocks(proc, self.block_samples):
                if pcm_block.size == 0:
                    continue
                processed += pcm_block.size

                if pbar:
                    pbar.update(pcm_block.size)
                    if (
                        int(pbar.n)
                        % (self.sample_rate * max(1, int(self.mem_interval)))
                        == 0
                    ):
                        rss, peak = _rss_and_peak()
                        pbar.set_postfix_str(f"rss {_fmt_bytes(rss)}")
                elif lp:
                    lp.update(pcm_block.size)

                sig01 = self._normalize01(pcm_block, self._gmin, self._gmax)
                if sig01 is None:
                    continue
                bits = self._binarize(sig01, threshold)
                for b in bits:
                    ring.append(np.uint8(b))

                i = 0
                usable = len(ring)
                while i + W <= usable:
                    if ring[i] == 1:
                        i += 1
                        continue
                    starts_total += 1
                    win = np.fromiter(
                        (ring[j] for j in range(i, i + W)), dtype=np.uint8, count=W
                    )
                    if flip_window:
                        win = win[::-1]
                    bytes5 = self._sample_window(win, trans, offs7)
                    if bytes5 is None:
                        break
                    serial_val = self._concat_bytes(bytes5)
                    start_global = ring_start_global + i
                    end_global = start_global + W
                    self.counts.append(int(serial_val))
                    self.frame_ranges.append((start_global, end_global))
                    i += stride

                keep = min(len(ring), W + stride)
                drop = len(ring) - keep
                if drop > 0:
                    ring_start_global += drop
                    for _ in range(drop):
                        ring.popleft()

        finally:
            try:
                proc.stdout and proc.stdout.close()
                proc.stderr and proc.stderr.close()
            except Exception:
                pass
            proc.kill()
            if pbar:
                pbar.close()
            if lp:
                lp.close()

        span = self._longest_plus_one_span(self.counts)
        dt = time.perf_counter() - t0
        rss, peak = _rss_and_peak()
        log.info(
            "Pass2: done in %.2fs (decoded=%d frames, longest +1 span=%d, starts_total=%d, rss=%s%s)",
            dt,
            len(self.counts),
            span,
            starts_total,
            _fmt_bytes(rss),
            f", peak≈{_fmt_bytes(peak)}" if peak else "",
        )
        return DecodeStats(
            bytes_total=len(self.counts) * 5,
            starts_total=starts_total,
            flips=bool(cfg["flip_signal"]),
            best_offset=0,
            monotonic_span=span,
        )

    # -------------------- Public API --------------------

    def decode(
        self, *, site: str = "jamail", threshold: float = 0.5
    ) -> Tuple[List[int], List[Tuple[int, int]], DecodeStats]:
        self._pass1_stats()
        if self._n_total <= 0:
            log.warning("No samples to decode; returning empty outputs.")
            return [], [], DecodeStats(0, 0, False, 0, 0)
        stats = self._pass2_decode(site=site, threshold=threshold)
        return self.counts, self.frame_ranges, stats

    # -------------------- Config helpers --------------------

    def _get_cfg(self, site: str) -> Dict[str, object]:
        base = BLOCK_PRESETS.get(site, BLOCK_PRESETS["jamail"])
        cfg = dict(base)
        cfg["trans"] = [int(x) - 1 for x in cfg["transition_points_1b"]]  # 0-based
        offs8 = [int(x) - 1 for x in cfg["bit_offsets_1b"]]
        cfg["offs7"] = offs8[:-1]
        cfg["W"] = int(cfg["window_samples"])
        cfg["stride"] = int(cfg["block_stride"])
        cfg["flip_window"] = bool(cfg["flip_window"])
        cfg["flip_signal"] = bool(cfg["flip_signal"])
        return cfg


# ---------------------------- File I/O helpers ----------------------------


def write_csv(
    out_path: str | Path, counts: Sequence[int], ranges: Sequence[Tuple[int, int]]
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["serial", "start_sample", "end_sample"])
        for i, val in enumerate(counts):
            if i < len(ranges):
                s, e = ranges[i]
            else:
                s, e = ("", "")
            w.writerow([int(val), s, e])
    log.info("Wrote CSV: %s (rows=%d)", p, len(counts))
    return p


def write_info_txt(
    out_path: str | Path,
    audio_input: str | Path,
    sample_rate: int,
    n_total_samples: int,
) -> Path:
    dur_s = (n_total_samples / sample_rate) if sample_rate else 0.0
    rss, peak = _rss_and_peak()
    lines = [
        f"audio_input: {audio_input}",
        f"sample_rate_hz: {sample_rate}",
        f"duration_s: {dur_s:.6f}",
        f"processed_at: {datetime.now().isoformat(timespec='seconds')}",
        f"rss_bytes: {rss}",
    ]
    if peak:
        lines.append(f"peak_bytes: {peak}")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote info: %s (duration=%.2fs)", p, dur_s)
    return p


# ---------------------------- CLI ----------------------------


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Efficient, streaming decoder for frame IDs (FFmpeg-backed) with project logging + progress."
    )
    ap.add_argument("audio", help="Path to input audio (.mp3 or .wav)")
    ap.add_argument(
        "--site", default="jamail", help="Site preset (jamail|nbu_sleep|nbu_lounge)"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold in [0,1] after global normalization",
    )
    ap.add_argument(
        "--outdir", help="Directory to write outputs (default: alongside input)"
    )
    ap.add_argument(
        "--block-seconds",
        type=float,
        default=2.0,
        help="Streaming PCM block size in seconds",
    )
    ap.add_argument(
        "--ring-seconds",
        type=float,
        default=2.0,
        help="Ring buffer capacity in seconds",
    )
    ap.add_argument(
        "--target-sample-rate",
        type=int,
        default=0,
        help="Optional resample rate (0 = keep source rate)",
    )

    # ---- Progress & Logging CLI ----
    ap.add_argument(
        "--progress",
        choices=["auto", "bar", "log", "none"],
        default="log",
        help="Progress display: bar=use tqdm when possible, log=periodic logs, auto=bar if TTY else log, none=off",
    )
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress log lines",
    )
    ap.add_argument(
        "--mem-interval",
        type=float,
        default=10.0,
        help="Seconds between memory stats in progress",
    )

    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for standalone mode",
    )
    ap.add_argument(
        "--seg",
        default=None,
        help="Segment id for log stamping (default: audio filename stem)",
    )
    ap.add_argument(
        "--cam", default="-", help="Camera serial for log stamping (default: '-')"
    )

    args = ap.parse_args()
    audio_path = Path(args.audio)
    out_dir = Path(args.outdir) if args.outdir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Configure logging (standalone-safe) ----
    seg_for_log = args.seg or audio_path.stem
    cam_for_log = args.cam or "-"
    configure_standalone_logging(args.log_level, seg=seg_for_log, cam=cam_for_log)
    global log
    log = logging.getLogger("sync")

    with log_context(seg=seg_for_log, cam=cam_for_log):
        log.info("Starting decode: audio=%s, outdir=%s", audio_path, out_dir)

        # Instantiate decoder
        decoder = StreamingSerialDecoder(
            filepath=audio_path,
            target_sample_rate=(args.target_sample_rate or None),
            block_seconds=max(0.25, float(args.block_seconds)),
            ring_seconds=max(float(args.ring_seconds), 2.0),
            progress_mode=args.progress,
            progress_interval=args.progress_interval,
            mem_interval=args.mem_interval,
        )

        # Run decode
        t0 = time.perf_counter()
        counts, ranges, stats = decoder.decode(site=args.site, threshold=args.threshold)
        t_decode = time.perf_counter() - t0
        log.info("Decode finished in %.2fs (frames=%d)", t_decode, len(counts))

        # Write outputs
        csv_path = out_dir / "raw.csv"
        info_path = out_dir / "raw_info.txt"
        write_csv(csv_path, counts, ranges)
        write_info_txt(
            info_path,
            audio_input=audio_path,
            sample_rate=decoder.sample_rate,
            n_total_samples=decoder._n_total,
        )

        # Brief console summary (still via logger)
        if counts:
            log.debug("first 10: %s", counts[:10])
            log.debug("last 10 : %s", counts[-10:])
        else:
            log.warning("No counts decoded.")

        log.info("Done. CSV: %s | INFO: %s", csv_path, info_path)


if __name__ == "__main__":
    _main()
