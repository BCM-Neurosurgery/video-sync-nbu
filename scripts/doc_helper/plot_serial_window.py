#!/usr/bin/env python3
"""
plot_serial_window.py — focus view of a decoded window with anchor/tap annotations.

Given a serial audio WAV, the target chunk serial identifier, and the window
sample bounds (in the ORIGINAL time axis), this helper plots:

1. Raw waveform with the window highlighted (plus optional cushion).
2. Binarized stream in original orientation with the window highlighted.
3. The sampling window (after per-window flip, if configured) annotated with
   anchor/tap markers and decoded byte values.

This mirrors the zoom+annotation portion of ``make_serial_plots.py`` but shows
the raw waveform (no normalization/binarization) for a chosen window so you can
see the exact amplitudes sampled by the decoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from scripts.doc_helper.make_serial_plots import (
    get_cfg,
    maybe_flip,
    read_wav_mono_float32,
    savefig,
    normalize01,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a serial window with anchors/taps annotated."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to WAV file.")
    parser.add_argument(
        "--site", default="jamail", help="Site preset (default: jamail)."
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Chunk serial identifier to display in titles (optional).",
    )
    parser.add_argument(
        "--start-sample",
        type=int,
        required=True,
        help="Start sample (inclusive) of the window in original orientation.",
    )
    parser.add_argument(
        "--end-sample",
        type=int,
        required=True,
        help="End sample (exclusive) of the window in original orientation.",
    )
    parser.add_argument(
        "--pad-ms",
        type=float,
        default=8.0,
        help="Cushion (milliseconds) on each side for the zoom plots (default: 8 ms).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to write PNG outputs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Horizontal threshold line for the window plot (default: 0.5).",
    )
    parser.add_argument(
        "--plot-normalized",
        action="store_true",
        help="Normalize audio to [0,1] before plotting.",
    )
    return parser.parse_args(argv)


def window_indices_for_scan(
    start_sample: int, end_sample: int, total_samples: int, flip_signal: bool
) -> tuple[int, int]:
    """Return the [start,end) indices in the scanning stream for a window."""
    if flip_signal:
        return total_samples - end_sample, total_samples - start_sample
    return start_sample, end_sample


def annotate_window(
    ax: plt.Axes,
    window: np.ndarray,
    trans: List[int],
    offs7: List[int],
    flip_window: bool,
    threshold: float,
    serial: Optional[str],
) -> None:
    """Draw anchor/tap guides on the supplied axes."""
    title_stub = "after per-window flip" if flip_window else "sampling window"
    serial_label = f" · serial {serial}" if serial else ""

    for gi, tpos in enumerate(trans):
        if not (0 <= tpos < len(window)):
            continue
        ax.axvline(tpos, linestyle="--", linewidth=1)
        taps = [tpos + o for o in offs7]
        taps = [p for p in taps if 0 <= p < len(window)]
        if not taps:
            continue
        ax.plot(taps, window[taps], marker="o", linestyle="None")

    ax.set_title(f"{title_stub}{serial_label} with anchors/taps (raw amplitude)")
    ax.axhline(threshold, linestyle=":", color="tab:red", linewidth=1.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    audio, sr = read_wav_mono_float32(args.audio)
    total_samples = audio.size

    start = int(args.start_sample)
    end = int(args.end_sample)
    if start < 0 or end < 0:
        raise ValueError("start/end samples must be non-negative.")
    if end <= start:
        raise ValueError("end_sample must be greater than start_sample.")
    if end > total_samples:
        raise ValueError("end_sample exceeds audio length.")

    cfg = get_cfg(args.site)
    window_expected = int(cfg["W"])
    window_len = end - start
    if window_len != window_expected:
        print(
            f"Warning: window length {window_len} differs from preset W={window_expected}."
        )

    flip_signal = bool(cfg["flip_signal"])
    flip_window = bool(cfg["flip_window"])
    trans: List[int] = cfg["trans"]
    offs7: List[int] = cfg["offs7"]
    threshold = float(args.threshold)

    if args.plot_normalized:
        audio01 = normalize01(audio)
        if audio01 is not None:
            audio = audio01
    raw_scan = maybe_flip(audio, flip_signal)
    scan_start, scan_end = window_indices_for_scan(
        start, end, total_samples, flip_signal
    )
    window_scan = raw_scan[scan_start:scan_end]
    if window_scan.size != window_len:
        raise RuntimeError("Window length mismatch after mapping to scanning stream.")

    window_for_taps = maybe_flip(window_scan, flip_window)

    pad_samples = int(round(sr * (float(args.pad_ms) / 1000.0)))
    zoom_start = max(0, start - pad_samples)
    zoom_end = min(total_samples, end + pad_samples)
    time_ms = (np.arange(zoom_start, zoom_end) - zoom_start) / sr * 1000.0
    highlight_start_ms = (start - zoom_start) / sr * 1000.0
    highlight_end_ms = (end - zoom_start) / sr * 1000.0

    serial_label = f" · serial {args.serial}" if args.serial else ""
    window_ms = window_len / sr * 1000.0

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot raw waveform around the window
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.plot(time_ms, audio[zoom_start:zoom_end])
    ax.axvspan(
        highlight_start_ms,
        highlight_end_ms,
        facecolor="tab:orange",
        alpha=0.25,
        edgecolor="black",
    )
    ax.set_title(
        f"Serial window (raw){serial_label} — {window_len} samples (~{window_ms:.2f} ms)"
    )
    ax.set_xlabel("Time in zoom (ms)")
    ax.set_ylabel("Amplitude")
    savefig(fig, out_dir, "window_zoom_raw.png")

    # Plot annotated sampling window
    x = np.arange(window_len)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(x, window_for_taps, linewidth=1.2)
    ymin = float(window_for_taps.min())
    ymax = float(window_for_taps.max())
    if np.isclose(ymin, ymax):
        delta = max(1e-3, abs(ymin) * 0.05)
        ymin -= delta
        ymax += delta
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_xlabel("Window sample index (0-based)")
    ax.set_ylabel("Amplitude")
    annotate_window(
        ax, window_for_taps, trans, offs7, flip_window, threshold, args.serial
    )
    savefig(fig, out_dir, f"window_taps_{args.serial}.png")
    print(f"Wrote window plots to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
