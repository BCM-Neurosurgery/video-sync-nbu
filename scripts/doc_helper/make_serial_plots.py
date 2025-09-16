#!/usr/bin/env python3
"""
make_serial_algo_plots.py — step-by-step visuals that follow the MATLAB-style block decoder

What this does
--------------
Given a serial track (WAV or MP3), this script reproduces the **exact** algorithmic steps
your decoder uses and emits clear, ordered figures:

  1) Raw waveform (original time)
  2) Normalized signal in [0,1] with threshold line
  3) Binarized stream (original, pre-flip)
  4) Binarized stream used for scanning (post global-flip if preset says so)
  5) Window scan on the **scanning stream** (shaded W-length windows, stride hops)
  6) Same windows mapped back to the **original** time axis (for intuition)
  7) Zoom (original time) around one chosen decoded block
  8) Inside that block: window after per-window flip (if any) with anchors/taps,
     bit order annotations by group (G1=MSB … G5=LSB), and the 7-bit values

The window search, per-window flip, anchors, and taps all match your production algorithm.

Usage
-----
python make_serial_algo_plots.py \
  --audio /path/to/serial.(wav|mp3) \
  --site nbu_lounge \
  --outdir docs/assets/serial_v2 \
  --threshold 0.5 \
  --zoom-index -1

Notes
-----
• The site preset controls: flip_signal, flip_window, W, stride, anchors/taps.
• --zoom-index is the index in **chronological** order (0..K-1, default -1 = last).
• MP3 support requires pydub + ffmpeg on PATH; WAV is read with stdlib wave.

Outputs
-------
step01_raw.png
step02_normalized.png
step03_binary_preflip.png
step04_binary_scanstream.png
step05_scanstream_windows.png
step06_original_windows.png
step07_zoom_raw.png
step08_zoom_binary.png
step09_window_taps.png
"""

import os
import argparse
import contextlib
import wave
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -------- Site presets (mirrors your decoder config) --------
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


# -------- I/O (WAV + optional MP3) --------
def read_wav_mono_float32(path: Path) -> Tuple[np.ndarray, int]:
    """Strict WAV reader; mono only; returns (audio, sr) with audio in [-1,1]."""
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if n_channels != 1:
        raise ValueError(f"WAV must be mono; got {n_channels} channels.")

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        norm = float(np.iinfo(np.int16).max)
    elif sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        data = data.astype(np.float32)
        norm = 127.0
    elif sampwidth == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        sign = (a[:, 2] & 0x80) != 0
        b = (
            a[:, 0].astype(np.int32)
            | (a[:, 1].astype(np.int32) << 8)
            | (a[:, 2].astype(np.int32) << 16)
        )
        b[sign] |= ~0 << 24  # sign-extend
        data = b.astype(np.float32)
        norm = float(2**23 - 1)
    else:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        norm = float(np.iinfo(np.int16).max)

    audio = np.clip(data / norm, -1.0, 1.0).astype(np.float32)
    return audio, sr


def read_mp3_float32(path: Path) -> Tuple[np.ndarray, int]:
    """MP3 via pydub + ffmpeg; downmix to mono; returns (audio, sr) in [-1,1]."""
    try:
        from pydub import AudioSegment
        from pydub.utils import which
    except Exception as e:
        raise ImportError("MP3 support requires 'pydub' (pip install pydub)") from e

    if not which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH (required for MP3).")

    seg = AudioSegment.from_file(str(path))
    sr = int(seg.frame_rate)
    ch = int(seg.channels)
    sw = int(seg.sample_width)  # bytes
    arr = np.array(seg.get_array_of_samples())

    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)  # downmix
    bits = 8 * sw
    denom = float((2 ** (bits - 1)) - 1) if bits > 8 else 127.0
    audio = (arr.astype(np.float32) / denom).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    ext = path.suffix.lower()
    if ext == ".wav":
        return read_wav_mono_float32(path)
    if ext == ".mp3":
        return read_mp3_float32(path)
    raise ValueError("Only .wav and .mp3 are supported.")


# -------- Helpers mirroring production decoder --------
def get_cfg(site: str) -> Dict[str, object]:
    cfg = BLOCK_PRESETS.get(site, BLOCK_PRESETS["jamail"]).copy()
    cfg["W"] = int(cfg["window_samples"])
    cfg["stride"] = int(cfg["block_stride"])
    cfg["trans"] = [x - 1 for x in cfg["transition_points_1b"]]
    offs8 = [x - 1 for x in cfg["bit_offsets_1b"]]
    cfg["offs7"] = offs8[:-1]
    return cfg


def normalize01(sig: np.ndarray) -> Optional[np.ndarray]:
    smin, smax = float(sig.min()), float(sig.max())
    if smax <= smin:
        return None
    return (sig - smin) / (smax - smin)


def binarize(sig01: np.ndarray, thr: float) -> np.ndarray:
    return (sig01 > thr).astype(np.uint8)


def maybe_flip(v: np.ndarray, do: bool) -> np.ndarray:
    return v[::-1] if do else v


def sample_window_bits(
    win: np.ndarray, trans: List[int], offs7: List[int]
) -> Optional[List[int]]:
    """Return five 7-bit integers (one per group). Follows MATLAB bit order (flip inner)."""
    vals: List[int] = []
    for t in trans:
        try:
            bits = [int(win[t + o]) for o in offs7]
        except IndexError:
            return None
        bits = bits[::-1]  # MSB..LSB
        bval = 0
        for b in bits:
            bval = (bval << 1) | b
        vals.append(bval)
    return vals


def concat_bytes_5x7(bytes5: List[int]) -> int:
    """b5‖b4‖b3‖b2‖b1, each 7 bits → 35-bit integer."""
    out = 0
    for b in bytes5[::-1]:
        out = (out << 7) | b
    return out


# -------- Plotting primitives --------
def savefig(fig, outdir: Path, name: str, dpi: int = 160):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=dpi)
    plt.close(fig)


def shade_windows_on_axis(
    ax, windows: List[Tuple[int, int]], *, face="tab:orange", alpha=0.35, lw=1.0
):
    for s, e in windows:
        ax.axvspan(
            s, e, facecolor=face, alpha=alpha, edgecolor="black", linewidth=lw, zorder=0
        )
        ax.axvline(s, color="black", linewidth=lw, alpha=0.9, zorder=2)
        ax.axvline(e, color="black", linewidth=lw, alpha=0.9, zorder=2)


# -------- Main visualization pipeline --------
def main():
    ap = argparse.ArgumentParser(
        description="Step-by-step visuals for the MATLAB-style serial decoder."
    )
    ap.add_argument(
        "--audio", required=True, help="Path to serial audio (.wav or .mp3)"
    )
    ap.add_argument(
        "--site", default="nbu_lounge", help="Preset: jamail | nbu_sleep | nbu_lounge"
    )
    ap.add_argument(
        "--outdir", default="docs/assets/serial_algo", help="Output folder for PNGs"
    )
    ap.add_argument(
        "--threshold", type=float, default=0.5, help="Binarization threshold in [0,1]"
    )
    ap.add_argument(
        "--zoom-index",
        type=int,
        default=-1,
        help="Which decoded block to zoom (chronological index)",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=200,
        help="Cap number of windows drawn/shaded",
    )
    args = ap.parse_args()

    audio_path = Path(args.audio)
    out = Path(args.outdir)

    # 1) Read audio
    audio, sr = read_audio(audio_path)
    N = audio.size
    t_ms = np.arange(N) / sr * 1000.0
    ds = max(1, int(sr // 2000))  # decimate for plotting

    # step01: raw waveform (original)
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.plot(t_ms[::ds], audio[::ds])
    ax.set_title("Step 1 — Raw serial audio (original time)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    savefig(fig, out, "step01_raw.png")

    # 2) Normalize → 3) Binarize (original)
    cfg = get_cfg(args.site)
    sig01 = normalize01(audio)
    if sig01 is None:
        raise RuntimeError("Flat signal; nothing to plot.")

    thr = float(args.threshold)
    binary_pre = binarize(sig01, thr)

    # step02: normalized with threshold
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.plot(t_ms[::ds], sig01[::ds])
    ax.axhline(thr, linestyle="--")
    ax.set_title("Step 2 — Normalized signal with threshold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    savefig(fig, out, "step02_normalized.png")

    # step03: binarized (original, pre-flip)
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.step(np.arange(N)[::ds], binary_pre[::ds], where="post")
    ax.set_title("Step 3 — Binarized stream (original, pre-flip)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("0/1")
    savefig(fig, out, "step03_binary_preflip.png")

    # 4) Global flip if preset says so → this is the SCANNING stream
    flip_signal = bool(cfg["flip_signal"])
    bin_scan = maybe_flip(binary_pre, flip_signal)

    # step04: binarized scan stream
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.step(np.arange(N)[::ds], bin_scan[::ds], where="post")
    ax.set_title(
        "Step 4 — Binarized scan stream "
        + ("(after global flip)" if flip_signal else "(no global flip)")
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("0/1")
    savefig(fig, out, "step04_binary_scanstream.png")

    # 5) Window scan on the SCANNING stream (exact algorithm)
    W = int(cfg["W"])
    stride = int(cfg["stride"])
    starts_scan: List[int] = []
    i = 0
    starts_total = 0
    while i < N and len(starts_scan) < args.max_windows:
        if bin_scan[i] == 1:
            i += 1
            continue
        starts_total += 1
        if i + W > N:
            break
        starts_scan.append(i)
        i += stride

    windows_scan = [(s, s + W) for s in starts_scan]

    # step05: shaded windows on the SCANNING stream axis
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.step(np.arange(N)[::ds], bin_scan[::ds], where="post", zorder=1)
    shade_windows_on_axis(ax, windows_scan)
    ax.set_title(f"Step 5 — Window scan on scan-stream (W={W}, stride={stride})")
    ax.set_xlabel("Sample index (scan-stream)")
    ax.set_ylabel("0/1")
    savefig(fig, out, "step05_scanstream_windows.png")

    # Map windows back to ORIGINAL orientation and reverse to chronological order,
    # matching the production decoder’s final frames order.
    if flip_signal:
        windows_orig = [(N - e, N - s) for (s, e) in windows_scan]
    else:
        windows_orig = windows_scan[:]
    windows_chrono = windows_orig[::-1]  # chronological

    # step06: same windows on the ORIGINAL axis for intuition
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.step(np.arange(N)[::ds], binary_pre[::ds], where="post", zorder=1)
    shade_windows_on_axis(ax, windows_orig)
    ax.set_title("Step 6 — Same windows mapped to original time")
    ax.set_xlabel("Sample index (original)")
    ax.set_ylabel("0/1")
    savefig(fig, out, "step06_original_windows.png")

    # Decode values (so we can choose an exact block to zoom)
    flip_window = bool(cfg["flip_window"])
    trans: List[int] = cfg["trans"]  # 0-based
    offs7: List[int] = cfg["offs7"]

    bytes_per_block: List[List[int]] = []
    ids35: List[int] = []
    for s, e in windows_scan:  # scan order
        win = bin_scan[s:e]
        win = maybe_flip(win, flip_window)
        vals = sample_window_bits(win, trans, offs7)
        if vals is None:
            break
        bytes_per_block.append(vals)
        ids35.append(concat_bytes_5x7(vals))

    # Choose which block to zoom by chronological index
    K = len(windows_chrono)
    if K == 0:
        raise RuntimeError("No complete windows found to zoom/annotate.")
    zoom_idx = args.zoom_index if args.zoom_index >= 0 else (K - 1)
    zoom_idx = max(0, min(zoom_idx, K - 1))
    # Chronological window in ORIGINAL coordinates:
    z_s_orig, z_e_orig = windows_chrono[zoom_idx]

    # step07: zoom raw (original time)
    pad_ms = 8.0
    pad = int(sr * (pad_ms / 1000.0))
    z0 = max(0, z_s_orig - pad)
    z1 = min(N, z_e_orig + pad)
    tz_ms = (np.arange(z0, z1) - z0) / sr * 1000.0

    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.plot(tz_ms, audio[z0:z1])
    ax.axvspan(
        (z_s_orig - z0) / sr * 1000.0,
        (z_e_orig - z0) / sr * 1000.0,
        facecolor="tab:orange",
        alpha=0.25,
        edgecolor="black",
    )
    ax.set_title(
        f"Step 7 — Zoom (raw, original time) — block #{zoom_idx} (chronological)"
    )
    ax.set_xlabel("Time in zoom (ms)")
    ax.set_ylabel("Amplitude")
    savefig(fig, out, "step07_zoom_raw.png")

    # step08: zoom binary (original time)
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.step(tz_ms, binary_pre[z0:z1], where="post")
    ax.axvspan(
        (z_s_orig - z0) / sr * 1000.0,
        (z_e_orig - z0) / sr * 1000.0,
        facecolor="tab:orange",
        alpha=0.25,
        edgecolor="black",
    )
    ax.set_title("Step 8 — Zoom (binary, original time) with window shaded")
    ax.set_xlabel("Time in zoom (ms)")
    ax.set_ylabel("0/1")
    savefig(fig, out, "step08_zoom_binary.png")

    # step09: inside the window used for sampling (post per-window flip if preset says so)
    # We need the window in the **sampling orientation**: start from scan-stream window that
    # corresponds to this chronological window.
    # windows_scan (scan order)  <->  windows_chrono (chronological) via reverse + map
    # Find matching scan-stream window index:
    # If flipped, windows_orig = map(windows_scan), then windows_chrono = windows_orig[::-1]
    # We can recover the scan index as: scan_k = (len(windows_scan) - 1 - zoom_idx)
    scan_k = len(windows_scan) - 1 - zoom_idx
    s_scan, e_scan = windows_scan[scan_k]
    win_scan = bin_scan[s_scan:e_scan]
    win_for_taps = maybe_flip(win_scan, flip_window)

    # Compute byte values for this block
    vals = sample_window_bits(win_for_taps, trans, offs7)
    if vals is None:
        vals = []
    val35 = concat_bytes_5x7(vals) if vals else None

    # Plot window with anchors/taps and annotations
    x = np.arange(W)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.step(x, win_for_taps, where="post")
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("Window sample index (0-based)")
    ax.set_ylabel("0/1")

    group_labels = ["G1 (MSB)", "G2", "G3", "G4", "G5 (LSB)"]
    for gi, tpos in enumerate(trans):
        if 0 <= tpos < W:
            ax.axvline(tpos, linestyle="--", linewidth=1)
            taps = [tpos + o for o in offs7]
            taps = [p for p in taps if 0 <= p < W]
            if taps:
                ax.plot(taps, win_for_taps[taps], marker="o", linestyle="None")
                # annotate b1..b7 in tap order (already MSB..LSB due to bits flip in sampling)
                for bi, p in enumerate(taps, start=1):
                    y = win_for_taps[p]
                    ytxt = 1.02 if y > 0.5 else -0.12
                    ax.text(
                        p,
                        ytxt,
                        f"{group_labels[gi]} b{bi}",
                        fontsize=7,
                        rotation=90,
                        ha="center",
                        va="bottom" if ytxt > 0 else "top",
                    )

    subtitle = "after per-window flip" if flip_window else "sampling window"
    if vals:
        ax.set_title(
            f"Step 9 — {subtitle} with anchors/taps · bytes7 = {vals} · id35 = {val35}"
        )
    else:
        ax.set_title(f"Step 9 — {subtitle} with anchors/taps (out-of-range taps)")

    savefig(fig, out, "step09_window_taps.png")

    # Quick console recap
    print(f"Wrote figures to: {out.resolve()}")
    print(
        f"site={args.site}  flip_signal={flip_signal}  flip_window={flip_window}  W={W}  stride={stride}"
    )
    print(
        f"windows found: {len(windows_scan)} (scan order)  → {len(windows_chrono)} (chronological)"
    )
    if vals:
        print(
            f"zoom block index (chronological) = {zoom_idx}  bytes7={vals}  id35={val35}"
        )


if __name__ == "__main__":
    main()
