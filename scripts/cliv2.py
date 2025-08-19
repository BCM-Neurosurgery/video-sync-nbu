#!/usr/bin/env python3
"""
cli.py — Streamlined orchestrator for A/V sync (uses your models.py & discover.py)

This CLI wires together your existing discovery layer with a practical
serial→audio mapping + per-segment sync. We intentionally **do not** implement
jitter/drift correction here—just a robust affine fit (RANSAC) and per-segment CFR.

Modules assumed (per your repo layout):
  - scripts.discover      → discover(audio_dir, video_dir, default_serial_channel=3) → AudioVideoSession
  - scripts.models        → dataclasses: AudioGroup, VideoGroup, AudioVideoSession, etc.
  - scripts.wavfileparser → WavSerialDecoder (for A3 serial decoding)

High-level flow
---------------
1) discover: find A1/A2 (program), A3 (serial), and segments (JSON+MP4s grouped by BASE)
2) index-serials: build A3 serial→sample index (midpoint per decoded block)
3) fit: collect anchors (NORMAL frames only) across all segments and RANSAC-fit n ≈ α·s + β
4) sync-segments: per segment & camera → compute audio window, clip A1/A2, mux with video (CFR)

Notes
-----
- This file leaves **TODOs** for: ffmpeg-based trims/mux, and persisting parquet/CSV if desired.
- We rely on **camera serial** (stable identity), not positional camera id.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import shutil
import subprocess
import csv

# --- Your modules (import paths per discover.py) ---
from scripts.discover import discover as run_discover
from scripts.wavfileparser import WavSerialDecoder

# --- Logging ---
logger = logging.getLogger("sync")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# --- Small data holders ---
@dataclass
class Anchor:
    serial: int
    audio_sample: int
    cam_serial: str
    segment_id: str


@dataclass
class FitResult:
    alpha: float
    beta: float
    inliers: int
    total: int
    rmse: float

    def predict(self, s: int) -> float:
        return self.alpha * s + self.beta


@dataclass
class ClipWindow:
    start: int
    end: int
    pad_head: int
    pad_tail: int


# ---------------------------------------------------------------------------
# Stage 0 — Discovery
# ---------------------------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> int:
    sess = run_discover(
        Path(args.audio_dir),
        Path(args.video_dir),
        default_serial_channel=args.serial_channel,
    )
    ag = sess.audiogroup
    vgs = sess.videogroups

    print("\nAudioGroup:")
    for ch in sorted(ag.audios.keys()):
        a = ag.audios[ch]
        print(
            f"  ch {ch:02d}: {a.path.name} (ext={a.extension}, sr={a.sample_rate}, dur={a.duration:.2f}s)"
        )
    if ag.serial_audio:
        print(
            f"  serial channel: ch {ag.serial_audio.channel:02d} ({ag.serial_audio.path.name})"
        )

    print("\nSegments:")
    for vg in vgs:
        ts = vg.timestamp.isoformat() if vg.timestamp else "None"
        cams = ", ".join(vg.cam_serials or [])
        print(f"  * {vg.group_id}  ts={ts}  cams=[{cams}]  json={vg.json.path.name}")
        if vg.videos:
            for v in vg.videos:
                print(f"      - cam {v.cam_serial}: {v.path.name}")
    return 0


# ---------------------------------------------------------------------------
# Stage 1 — Build serial index from A3
# ---------------------------------------------------------------------------


def build_serial_index(
    a3_path: Path, out_index: Path, *, site: str = "jamail", threshold: float = 0.5
) -> Path:
    """Decode A3 once and persist per-block rows using decoder ranges.

    CSV columns
    -----------
    serial,start_sample,end_sample
    (one row per decoded block; mirrors wavfileparser.save_counts_csv)
    """
    logger.info("Decoding serial audio (A3): %s", a3_path)
    dec = WavSerialDecoder(str(a3_path))
    frames, stats = dec.decode_by_block(site=site, threshold=threshold)

    logger.info(
        "Decoded %d serials (bytes_total=%d, longest_monotone=%d)",
        len(frames),
        getattr(stats, "bytes_total", 0),
        getattr(stats, "monotonic_span", 0),
    )

    # Prefer `frame_ranges` (as in wavfileparser); fall back to `ranges` if present.
    ranges = getattr(dec, "frame_ranges", None) or getattr(dec, "ranges", None) or []

    out_index.parent.mkdir(parents=True, exist_ok=True)
    with out_index.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["serial", "start_sample", "end_sample"])
        for i, val in enumerate(frames):
            s, e = ranges[i] if i < len(ranges) else ("", "")
            # Guard in case `val` can be None
            serial = "" if val is None else int(val)
            w.writerow([serial, s, e])

    logger.info("Wrote serial CSV with %d rows → %s", len(frames), out_index)
    return out_index


def cmd_index(args: argparse.Namespace) -> int:
    sess = run_discover(
        Path(args.audio_dir),
        Path(args.video_dir),
        default_serial_channel=args.serial_channel,
    )
    a3 = sess.audiogroup.serial_audio
    assert (
        a3 is not None
    ), "No serial channel found (expected channel == serial_channel)."
    build_serial_index(
        Path(a3.path), Path(args.out_index), site=args.site, threshold=args.threshold
    )
    return 0


def load_serial_index(path: Path) -> Dict[int, int]:
    """
    Load a CSV with columns: serial,start_sample,end_sample
    Returns a mapping {serial: start_sample}, first occurrence wins.
    """
    mapping: Dict[int, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                serial_str = (row.get("serial") or "").strip()
                start_str = (row.get("start_sample") or "").strip()
                if not serial_str or not start_str:
                    continue
                serial = int(serial_str)
                start_sample = int(start_str)
                # keep the first occurrence
                mapping.setdefault(serial, start_sample)
            except (ValueError, TypeError, KeyError):
                # skip malformed rows
                continue
    return mapping


# ---------------------------------------------------------------------------
# Stage 2 — Anchors & labeling
# ---------------------------------------------------------------------------


def label_frames(serials: Sequence[int], frame_ids: Sequence[int]) -> List[str]:
    """Mark frames for anchor selection & diagnostics.
    NORMAL:    Δfid=1 & Δserial=1
    DUPLICATE: Δfid=1 & Δserial=0
    DROP:      Δfid>1
    MISSING:   serial<=0 or None
    """
    labels: List[str] = []
    prev_fid: Optional[int] = None
    prev_s: Optional[int] = None
    for s, f in zip(serials, frame_ids):
        if s is None or s <= 0:
            labels.append("MISSING")
        elif prev_fid is None:
            labels.append("NORMAL")
        else:
            df = f - prev_fid
            ds = s - prev_s  # type: ignore[arg-type]
            if df == 1 and ds == 1:
                labels.append("NORMAL")
            elif df == 1 and ds == 0:
                labels.append("DUPLICATE")
            elif df > 1:
                labels.append("DROP")
            else:
                labels.append("MISSING")
        prev_fid, prev_s = f, s
    return labels


def collect_anchors(
    index_map: Dict[int, int], session, *, min_k: int = 3, min_span_ratio: float = 0.05
) -> List[Anchor]:
    """Build a list of audio/video alignment anchors from all segments/cameras.

    This traverses every VideoGroup in `session` and, for each Video in that group,
    looks up the corresponding CamJson (via `vg.json.cam_jsons[cam_serial]`). From
    that CamJson it expects:
      • `fixed_serials`: per-frame serial IDs after any fixing/cleanup
      • `raw_frame_ids`: per-frame frame IDs as recorded
    It labels each frame with `label_frames(serials, frame_ids)` and keeps only
    frames labeled "NORMAL". For each kept frame whose serial `s` exists in
    `index_map`, it emits an Anchor:
        Anchor(serial=s,
               audio_sample=index_map[s],
               cam_serial=str(v.cam_serial),
               segment_id=vg.group_id)

    Parameters
    ----------
    index_map : Dict[int, int]
        Mapping from decoded serial ID (from the serial audio channel) to the
        corresponding audio sample index (start index of each block).
        Keys and values must be integers.
    session : scripts.models.AudioVideoSession
        Result of `discover(...)`. Must contain `videogroups`, each with a `json`
        that has `cam_jsons: Dict[str, CamJson]`, and each `CamJson` provides
        `fixed_serials` and `raw_frame_ids`.
    min_k : int, default 3
        If a camera yields fewer than `min_k` candidate anchors in its segment,
        a warning is logged. This does not prevent anchors from being returned.
    min_span_ratio : float, default 0.05
        Heuristic span check. Let `s_vals` be the kept serials for a cam/segment
        and `span = max(s_vals) - min(s_vals)`. If `span` is smaller than
        `max(1, int(min_span_ratio * (max(s_vals) - min(s_vals) + 1)))`, a warning
        is logged to flag poor coverage (e.g., all anchors clumped together).

    Returns
    -------
    List[Anchor]
        One Anchor per kept frame (NORMAL + present in `index_map`), across all
        segments and cameras. The list may be empty if no valid anchors exist.

    Notes
    -----
    • This function does *not* deduplicate anchors across segments/cameras.
      Multiple segments containing the same serial will yield multiple anchors.
    • It assumes `fixed_serials` and `raw_frame_ids` are 1:1 aligned and of the
      same length for a given CamJson.
    • Logging:
        - Warns if CamJson is missing or lacks required arrays.
        - Warns if a cam/segment yields < `min_k` anchors.
        - Warns if the serial span heuristic indicates low coverage.
      Finally logs the total anchors collected and segment count.

    """
    anchors: List[Anchor] = []
    for vg in session.videogroups:
        if not vg.videos:
            continue
        for v in vg.videos:
            cam_serial = str(v.cam_serial)
            cj = vg.json.cam_jsons.get(cam_serial)
            if not cj or not cj.fixed_serials or not cj.raw_frame_ids:
                logger.warning(
                    "%s cam %s: missing fixed serials/frame_ids in JSON",
                    vg.group_id,
                    cam_serial,
                )
                continue
            serials = list(cj.fixed_serials)
            frame_ids = list(cj.raw_frame_ids)
            labels = label_frames(serials, frame_ids)

            # Build anchors: NORMAL frames with serial present in index_map
            cand = [
                (i, s)
                for i, (lab, s) in enumerate(zip(labels, serials))
                if lab == "NORMAL" and s in index_map
            ]
            if len(cand) < min_k:
                logger.warning(
                    "Few anchors for %s cam %s: %d", vg.group_id, cam_serial, len(cand)
                )
            if cand:
                s_vals = [s for _, s in cand]
                span = max(s_vals) - min(s_vals) if len(s_vals) > 1 else 0
                # Span check (relative to local range)
                if span < max(1, int(min_span_ratio * (max(s_vals) - min(s_vals) + 1))):
                    logger.warning(
                        "Low anchor span for %s cam %s: span=%d",
                        vg.group_id,
                        cam_serial,
                        span,
                    )

            for i, s in cand:
                anchors.append(
                    Anchor(
                        serial=int(s),
                        audio_sample=int(index_map[int(s)]),
                        cam_serial=cam_serial,
                        segment_id=vg.group_id,
                    )
                )

    logger.info(
        "Collected %d anchors across %d segments.",
        len(anchors),
        len(session.videogroups),
    )
    return anchors


# ---------------------------------------------------------------------------
# Stage 3 — Robust affine fit (RANSAC → LS on inliers)
# ---------------------------------------------------------------------------


def ransac_affine(
    anchors: List[Anchor],
    tau_samples: float = 3200.0,
    iters: int = 1000,
    min_inliers: int = 20,
) -> FitResult:
    """
    Robustly fit an affine map y ≈ α·x + β between serial/frame indices and audio
    sample indices using a simple RANSAC followed by least-squares on the inliers.

    This is typically used to align camera-derived indices (e.g., frame IDs or
    decoded chunk serials) to audio sample positions in the serial channel.

    Parameters
    ----------
    anchors : List[Anchor]
        Collection of correspondence points. Each `Anchor` must expose:
          - `serial` (x): camera-side index (e.g., frame ID or chunk-serial)
          - `audio_sample` (y): matching audio sample index (integer or float)
        At least two anchors are required.
    tau_samples : float, default=3200.0
        Inlier threshold in *audio samples*. A residual |y - (αx + β)| ≤ `tau_samples`
        is treated as an inlier during RANSAC. (Example: at 48 kHz, 3200 samples ≈ 66.7 ms.)
    iters : int, default=1000
        Number of RANSAC iterations. Each iteration samples two anchors to
        hypothesize (α, β), then counts inliers under `tau_samples`.
    min_inliers : int, default=20
        Minimum absolute number of inliers required for a "strong" model. The
        implementation also requires at least 50% of all anchors to be inliers.
        If this is not met, a warning is logged and the fit proceeds with the
        best model found.

    Returns
    -------
    FitResult
        Dataclass summarizing the fit with fields:
          - `alpha` : float
                Slope (samples per serial unit). If x is frame ID, then
                α ≈ sample_rate / fps.
          - `beta` : float
                Intercept at x = 0 (samples).
          - `inliers` : int
                Number of inliers used in the final least-squares refit.
          - `total` : int
                Total number of anchors provided.
          - `rmse` : float
                Root-mean-square error over the inlier set (in samples).

    Notes
    -----
    - The RANSAC hypothesis uses two random distinct anchors; vertical models
      (Δx = 0) are skipped.
    - After selecting the best inlier set, parameters (α, β) are recomputed
      via closed-form least squares on those inliers.
    - For reproducibility, set `random.seed(...)` in the caller before invoking.
    - A warning is emitted if the best consensus set is "weak" (too few inliers).

    Raises
    ------
    AssertionError
        If fewer than two anchors are provided.

    Examples
    --------
    >>> # anchors: serial -> audio_sample
    >>> anchors = [Anchor(serial=0, audio_sample=1000),
    ...            Anchor(serial=10, audio_sample=58000),
    ...            Anchor(serial=20, audio_sample=115000)]
    >>> fit = ransac_affine(anchors, tau_samples=2000, iters=500)
    >>> fit.alpha, fit.beta  # doctest: +SKIP
    (approx_sample_per_serial, approx_intercept)
    """
    import random

    assert len(anchors) >= 2, "Not enough anchors to fit."

    xs = [a.serial for a in anchors]
    ys = [a.audio_sample for a in anchors]

    best = None
    for _ in range(iters):
        i1, i2 = random.sample(range(len(anchors)), 2)
        x1, y1 = xs[i1], ys[i1]
        x2, y2 = xs[i2], ys[i2]
        if x2 == x1:
            continue
        alpha = (y2 - y1) / (x2 - x1)
        beta = y1 - alpha * x1
        resid = [abs(y - (alpha * x + beta)) for x, y in zip(xs, ys)]
        inl_idx = [i for i, r in enumerate(resid) if r <= tau_samples]
        if best is None or len(inl_idx) > best["ninl"]:
            best = {"alpha": alpha, "beta": beta, "idx": inl_idx, "ninl": len(inl_idx)}

    if best is None or best["ninl"] < max(min_inliers, int(0.5 * len(anchors))):
        logger.warning("Weak RANSAC fit; consider increasing anchors or tau.")

    inliers = best["idx"] if best else list(range(len(anchors)))
    X = [xs[i] for i in inliers]
    Y = [ys[i] for i in inliers]
    n = len(X)

    # Least squares on inliers
    xbar = sum(X) / n
    ybar = sum(Y) / n
    num = sum((x - xbar) * (y - ybar) for x, y in zip(X, Y))
    den = sum((x - xbar) ** 2 for x in X) or 1.0
    alpha = num / den
    beta = ybar - alpha * xbar

    rmse = math.sqrt(
        sum((y - (alpha * x + beta)) ** 2 for x, y in zip(X, Y)) / max(1, n)
    )
    return FitResult(
        alpha=alpha, beta=beta, inliers=len(inliers), total=len(anchors), rmse=rmse
    )


# ---------------------------------------------------------------------------
# Stage 4 — Windows, trims, mux
# ---------------------------------------------------------------------------


def first_last_valid_serial(serials: Sequence[int]) -> Optional[Tuple[int, int]]:
    vals = [s for s in serials if s is not None and s > 0]
    return (vals[0], vals[-1]) if vals else None


def compute_clip_window_for_segment(
    serials: Sequence[int],
    fit: FitResult,
    *,
    margin_samples: int,
    audio_len_samples: int,
) -> Optional[ClipWindow]:
    """
    Compute the **sample-accurate audio window** for a video segment using the
    first and last *valid* serials in that segment and an affine mapping
    (``sample ≈ alpha·serial + beta``).

    The function returns a **clamped** window `[start, end)` that lies inside the
    actual recorder file, along with **diagnostic** values `pad_head` and
    `pad_tail` indicating how many samples would be *missing* at the head/tail
    if we attempted to use the *unclamped* ideal window. **No padding is added
    here**—callers may choose to synthesize silence later if required.

    Parameters
    ----------
    serials : Sequence[int]
        Per-frame serial values for the segment (e.g., from JSON). Non-positive
        values (≤0) are treated as invalid/missing and ignored when locating
        the endpoints.
    fit : FitResult
        Affine map from serial → audio sample (``predict(s)`` returns a sample
        index in the recorder timeline).
    margin_samples : int
        Safety margin (in samples) applied to both sides of the raw window
        before clamping. Use roughly one serial block in samples.
    audio_len_samples : int
        Total length of the underlying recorder file in samples; used to clamp
        the window to `[0, audio_len_samples)` and compute diagnostic padding.

    Returns
    -------
    Optional[ClipWindow]
        ``ClipWindow(start, end, pad_head, pad_tail)`` if the segment contains
        at least one valid serial; otherwise ``None``.

    Notes
    -----
    Algorithm steps:
      1) Find the first/last **positive** serial in ``serials`` → ``s_first``, ``s_last``.
      2) Compute the **raw** window in samples using the fit and margin:
         ``start_raw = floor(predict(s_first) - margin)``
         ``end_raw   = ceil (predict(s_last)  + margin)``
      3) Derive diagnostic padding relative to the recorder bounds:
         ``pad_head = max(0, -start_raw)``,
         ``pad_tail = max(0, end_raw - audio_len_samples)``
      4) Clamp to the recorder timeline:
         ``start = max(0, start_raw)``,
         ``end   = min(audio_len_samples, end_raw)``
      5) Return the clamped window and diagnostics.
    """
    pair = first_last_valid_serial(serials)
    if not pair:
        return None
    s_first, s_last = pair
    start = math.floor(fit.predict(s_first) - margin_samples)
    end = math.ceil(fit.predict(s_last) + margin_samples)
    pad_head = max(0, -start)
    pad_tail = max(0, end - audio_len_samples)
    start = max(0, start)
    end = min(audio_len_samples, end)
    return ClipWindow(start, end, pad_head, pad_tail)


def clip_program_audio(
    a1: Path, a2: Path, window: ClipWindow, out_dir: Path, tag: str
) -> Tuple[Path, Path]:
    """
    Trim the two program-audio files to a sample-accurate window using ffmpeg's
    `atrim` filter and write lossless WAVs.
    """
    import shutil
    import subprocess

    def _run_atrim(in_path: Path, out_path: Path, start: int, end: int) -> None:
        """Run ffmpeg with a sample-accurate atrim and write PCM WAV."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        filter_expr = (
            f"atrim=start_sample={start}:end_sample={end},asetpts=PTS-STARTPTS"
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-vn",
            "-i",
            str(in_path),
            "-af",
            filter_expr,
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            err_text = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"ffmpeg atrim failed for '{in_path}' → '{out_path}' "
                f"(start={start}, end={end}).\n{err_text}"
            )

    # Sanity checks
    if window.end <= window.start:
        raise RuntimeError(
            f"Invalid clip window: start={window.start}, end={window.end}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_a1 = out_dir / f"{tag}.A1.wav"
    out_a2 = out_dir / f"{tag}.A2.wav"

    _run_atrim(a1, out_a1, window.start, window.end)
    _run_atrim(a2, out_a2, window.start, window.end)

    logger.info(
        "Clipped A1/A2 samples [%d:%d) → %s, %s",
        window.start,
        window.end,
        out_a1.name,
        out_a2.name,
    )
    return out_a1, out_a2


def mux_video_audio(
    mp4_in: Path, a1_clip: Path, a2_clip: Path, fps: Optional[float], out_path: Path
) -> Path:
    """
    Mux one MP4 video with two mono program-audio clips into an MP4.

    Behavior
    --------
    - If `fps` is provided, the video is **re-encoded** to a constant frame rate (CFR)
      using libx264 at that fps. This is the safest way to keep A/V in lock-step
      with your sample-accurate audio trims. We use `-vsync cfr`, output `-r`, and
      `-shortest` so the mux stops at the shortest input (typically the audio).
    - If `fps` is None, the video stream is **copied** (`-c:v copy`) and only audio
      is re-encoded to AAC. This preserves any source VFR timing; only use this if
      your upstream PTS are already correct.

    Inputs
    ------
    mp4_in   : Path to the source MP4 (video stream 0:v:0 is used).
    a1_clip  : Path to clipped program-audio for channel A1 (e.g., WAV).
    a2_clip  : Path to clipped program-audio for channel A2 (e.g., WAV).
               If you only have one program channel, pass the same file for both.
    fps      : Target CFR (e.g., 30.0). If None, video is copied (no CFR enforcement).
    out_path : Destination MP4 (parent dirs are created).

    Returns
    -------
    Path to the output file on success.

    Raises
    ------
    FileNotFoundError if ffmpeg is not found.
    RuntimeError if ffmpeg returns a non-zero exit code.
    """
    # Preconditions
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH. Please install ffmpeg.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Base command: inputs + stream mapping (video + two audio tracks)
    cmd = [
        "ffmpeg",
        "-y",  # overwrite out_path if it exists
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4_in),
        "-i",
        str(a1_clip),
        "-i",
        str(a2_clip),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
    ]

    if fps is not None:
        # Enforce CFR by re-encoding video. Keep this conservative & fast.
        cmd += [
            "-r",
            f"{fps:.6f}",  # output frame rate
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        # Preserve original video stream/timestamps
        cmd += ["-c:v", "copy"]

    # Encode audio to AAC (WAV/FLAC/etc. will be transcoded)
    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",  # stop when the shortest stream ends (usually audio)
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    logger.info(
        "Muxing %s with A1=%s, A2=%s %s → %s",
        mp4_in.name,
        a1_clip.name,
        a2_clip.name,
        f"(CFR {fps:.6f} fps)" if fps is not None else "(copy video)",
        out_path.name,
    )

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed during mux:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )

    return out_path


# ---------------------------------------------------------------------------
# Stage 5 — Orchestration commands: fit & sync
# ---------------------------------------------------------------------------


def cmd_fit(args: argparse.Namespace) -> int:
    sess = run_discover(
        Path(args.audio_dir),
        Path(args.video_dir),
        default_serial_channel=args.serial_channel,
    )
    index_map = load_serial_index(Path(args.index))

    anchors = collect_anchors(
        index_map, sess, min_k=args.min_k, min_span_ratio=args.min_span
    )
    fit = ransac_affine(
        anchors, tau_samples=args.tau, iters=args.iters, min_inliers=args.min_inliers
    )

    print(
        json.dumps(
            {
                "alpha": fit.alpha,
                "beta": fit.beta,
                "inliers": fit.inliers,
                "total": fit.total,
                "rmse": fit.rmse,
            },
            indent=2,
        )
    )

    Path(args.out_fit).write_text(
        json.dumps(
            {
                "alpha": fit.alpha,
                "beta": fit.beta,
                "inliers": fit.inliers,
                "total": fit.total,
                "rmse": fit.rmse,
            },
            indent=2,
        )
    )
    logger.info("Saved fit → %s", args.out_fit)
    return 0


def cmd_sync_segments(args: argparse.Namespace) -> int:
    sess = run_discover(
        Path(args.audio_dir),
        Path(args.video_dir),
        default_serial_channel=args.serial_channel,
    )
    ag = sess.audiogroup
    vgs = sess.videogroups

    # Select program audios (all channels except the serial channel)
    assert ag.serial_audio is not None, "No serial channel found."
    serial_ch = ag.serial_audio.channel
    prog_channels = [ch for ch in sorted(ag.audios.keys()) if ch != serial_ch]
    assert len(prog_channels) >= 1, "No program audio channels found."
    a1 = Path(ag.audios[prog_channels[0]].path)
    a2 = (
        Path(ag.audios[prog_channels[1]].path)
        if len(prog_channels) > 1
        else Path(ag.audios[prog_channels[0]].path)
    )

    # Audio sample rate & length from the serial channel (shared recorder clock)
    fs = int(ag.serial_audio.sample_rate)
    audio_len_samples = int(fs * float(ag.serial_audio.duration))

    # Load fit parameters
    params = json.loads(Path(args.fit).read_text())
    fit = FitResult(
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        inliers=int(params.get("inliers", 0)),
        total=int(params.get("total", 0)),
        rmse=float(params.get("rmse", 0.0)),
    )

    out_audio = Path(args.out_audio)
    out_video = Path(args.out_video)

    for vg in vgs:
        if not vg.videos:
            logger.warning("%s: no videos.", vg.group_id)
            continue
        for v in vg.videos:
            cam_serial = str(v.cam_serial)
            cj = vg.json.cam_jsons.get(cam_serial)
            if not cj or not cj.fixed_serials:
                logger.warning(
                    "%s cam %s: missing JSON serials.", vg.group_id, cam_serial
                )
                continue

            window = compute_clip_window_for_segment(
                cj.fixed_serials,
                fit,
                margin_samples=args.margin,
                audio_len_samples=audio_len_samples,
            )
            if not window:
                logger.warning("%s cam %s: no valid window.", vg.group_id, cam_serial)
                continue

            # CFR fps so that video duration == audio clip duration
            n_frames = v.frame_count
            fps = n_frames / max(1e-9, (window.end - window.start) / fs)

            tag = f"{vg.group_id}.serial{cam_serial}"
            a1_clip, a2_clip = clip_program_audio(a1, a2, window, out_audio, tag)
            out_path = out_video / f"{tag}_synced.mp4"
            mux_video_audio(Path(v.path), a1_clip, a2_clip, fps, out_path)

    logger.info("Sync complete (placeholders used for trims/mux; fill TODOs).")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sync",
        description="Audio/Video sync orchestrator (models.py + discover.py)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="List discovered audio/video segments")
    d.add_argument("--audio-dir", required=True)
    d.add_argument("--video-dir", required=True)
    d.add_argument("--serial-channel", type=int, default=3)
    d.set_defaults(func=cmd_discover)

    i = sub.add_parser(
        "index-serials", help="Build serial index JSON from A3 (serial channel)"
    )
    i.add_argument("--audio-dir", required=True)
    i.add_argument("--video-dir", required=True)
    i.add_argument("--serial-channel", type=int, default=3)
    i.add_argument("--out-index", required=True)
    i.add_argument("--site", default="jamail")
    i.add_argument("--threshold", type=float, default=0.5)
    i.set_defaults(func=cmd_index)

    f = sub.add_parser("fit", help="Collect anchors and fit affine map n ≈ α·s + β")
    f.add_argument("--audio-dir", required=True)
    f.add_argument("--video-dir", required=True)
    f.add_argument("--serial-channel", type=int, default=3)
    f.add_argument("--index", required=True)
    f.add_argument("--out-fit", required=True)
    f.add_argument("--min-k", type=int, default=3)
    f.add_argument("--min-span", type=float, default=0.05)
    f.add_argument("--tau", type=float, default=3200.0)
    f.add_argument("--iters", type=int, default=1000)
    f.add_argument("--min-inliers", type=int, default=20)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser(
        "sync-segments",
        help="Compute per-segment windows, clip A1/A2, and mux to synced MP4s (CFR)",
    )
    s.add_argument("--audio-dir", required=True)
    s.add_argument("--video-dir", required=True)
    s.add_argument("--serial-channel", type=int, default=3)
    s.add_argument("--fit", required=True)
    s.add_argument("--out-audio", required=True)
    s.add_argument("--out-video", required=True)
    s.add_argument(
        "--margin",
        type=int,
        default=1600,
        help="Samples of safety margin (~1 serial block)",
    )
    s.set_defaults(func=cmd_sync_segments)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        logger.exception(e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
