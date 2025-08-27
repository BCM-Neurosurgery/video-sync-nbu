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
4) sync-segments: per segment & camera → compute audio window, clip A1/A2, mux with video (CFR).

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
from typing import Dict, List, Optional, Sequence, Tuple
import shutil
import subprocess
import csv

from scripts.discover import discover as run_discover

logger = logging.getLogger("sync")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


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


@dataclass
class MatchedWindow:
    fid0: int
    fid1: int
    s0: int
    s1: int
    fps: float  # CFR computed from anchors (frames / audio duration)


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
    """Compute the sample-accurate audio window for a segment using an affine fit."""
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
    a1: Path,
    a2: Path,
    window: "ClipWindow",
    out_dir: Path,
    tag: str,
    out_fs: Optional[int] = None,
    serial_fs: int = 48000,
) -> Tuple[Path, Path]:
    """
    Trim A1/A2 to the serial-defined window. `window.start/end` are in SERIAL samples.
    We convert to seconds and trim by time, then (optionally) resample to `out_fs`.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH.")
    if window.end <= window.start:
        raise RuntimeError(
            f"Invalid clip window: start={window.start}, end={window.end}"
        )
    if out_fs is not None and (not isinstance(out_fs, int) or out_fs <= 0):
        raise ValueError(f"out_fs must be a positive int (Hz), got {out_fs!r}")

    # Convert serial samples → seconds
    start_sec = window.start / float(serial_fs)
    end_sec = window.end / float(serial_fs)

    def _run_trim(in_path: Path, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base = f"atrim=start={start_sec:.9f}:end={end_sec:.9f},asetpts=PTS-STARTPTS"
        filt = (
            base
            if out_fs is None
            else f"{base},aresample=sample_rate={out_fs}:resampler=soxr"
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
            filt,
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg trim failed for '{in_path}':\n{proc.stderr}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_a1 = out_dir / f"{tag}.A1.wav"
    out_a2 = out_dir / f"{tag}.A2.wav"
    _run_trim(a1, out_a1)
    _run_trim(a2, out_a2)

    logger.info(
        "Clipped A1/A2 to t=[%.6f, %.6f) s (serial_fs=%d)%s → %s, %s",
        start_sec,
        end_sec,
        serial_fs,
        f", out_fs={out_fs}" if out_fs else "",
        out_a1.name,
        out_a2.name,
    )
    return out_a1, out_a2


def mux_video_audio(
    mp4_in: Path, a1_clip: Path, a2_clip: Path, fps: Optional[float], out_path: Path
) -> Path:
    """
    Mux one MP4 video with two mono program-audio clips into an MP4.
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
            f"{fps:.6f}",
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
        "-shortest",
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


def compute_window_from_anchors(
    anchors_for_video: List[dict],
    fs: int,
    audio_len_samples: int,
    *,
    margin_samples: int = 0,
) -> MatchedWindow:
    if not anchors_for_video:
        raise RuntimeError("No anchors for this video")

    anchors_for_video = sorted(
        anchors_for_video, key=lambda a: int(a["frame_id_reidx"])
    )
    a_start, a_end = anchors_for_video[0], anchors_for_video[-1]

    fid0, fid1 = int(a_start["frame_id_reidx"]), int(a_end["frame_id_reidx"])
    if fid1 < fid0:
        fid0, fid1 = fid1, fid0

    s0, s1 = int(a_start["audio_sample"]), int(a_end["audio_sample"])
    if s1 < s0:
        s0, s1 = s1, s0

    if margin_samples:
        _s0, _s1 = s0, s1
        s0 = max(0, s0 - margin_samples)
        s1 = min(audio_len_samples, s1 + margin_samples)
        logger.debug(
            "Applied margins: samples [%d:%d) → [%d:%d) (+/-%d)",
            _s0,
            _s1,
            s0,
            s1,
            margin_samples,
        )

    n_frames = fid1 - fid0 + 1
    if n_frames <= 0:
        raise RuntimeError(f"Invalid frame span: [{fid0}, {fid1}]")

    audio_dur_sec = (s1 - s0) / float(fs)
    if audio_dur_sec <= 0:
        raise RuntimeError(f"Invalid audio sample span: [{s0}, {s1}] @ fs={fs}")

    fps = n_frames / audio_dur_sec

    logger.info(
        "Matched window: frames [%d..%d] (n=%d), samples [%d..%d) (%.3fs), CFR=%.6f fps",
        fid0,
        fid1,
        n_frames,
        s0,
        s1,
        audio_dur_sec,
        fps,
    )
    return MatchedWindow(fid0=fid0, fid1=fid1, s0=s0, s1=s1, fps=fps)


def clip_video_by_frames(
    mp4_in: Path,
    n0: int,
    n1: int,
    fps: float,
    out_path: Path,
) -> Path:
    """
    Extract frames in [n0, n1] inclusive by index and re-encode at true CFR `fps`
    without dropping/duplicating frames.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH. Please install ffmpeg.")
    if n1 < n0:
        raise RuntimeError(f"Invalid frame window: [{n0}, {n1}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # trim uses end_frame as EXCLUSIVE → add +1 to include n1
    end_frame_excl = n1 + 1
    vf = f"trim=start_frame={n0}:end_frame={end_frame_excl},setpts=(N/{fps:.9f})/TB"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4_in),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]

    dur_sec = (n1 - n0 + 1) / float(fps)
    logger.info(
        "Video trim %s → frames [%d..%d] @ %.6f fps (≈%.3fs) → %s",
        mp4_in.name,
        n0,
        n1,
        fps,
        dur_sec,
        out_path.name,
    )
    logger.debug("FFmpeg (video-trim) cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed during video frame-trim:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )
    return out_path


def cmd_sync_segments(args: argparse.Namespace) -> int:
    sess = run_discover(
        Path(args.audio_dir),
        Path(args.video_dir),
        default_serial_channel=args.serial_channel,
    )
    ag = sess.audiogroup
    vgs = sess.videogroups

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

    fs = int(ag.serial_audio.sample_rate)
    logger.info("Audio sample rate: %d Hz", fs)
    audio_len_samples = int(fs * float(ag.serial_audio.duration))

    if not getattr(args, "anchors", None):
        raise ValueError("--anchors is required for anchor-driven sync.")
    try:
        anchors_all = json.loads(Path(args.anchors).read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to load anchors JSON ({args.anchors}): {e}") from e

    out_audio = Path(args.out_audio)
    out_video = Path(args.out_video)

    total_videos = sum(len(vg.videos or []) for vg in vgs)
    logger.info(
        "Starting anchor-driven sync: %d segments, %d videos, fs=%d Hz",
        len(vgs),
        total_videos,
        fs,
    )
    logger.info("Loaded %d anchors (global).", len(anchors_all))

    produced = 0
    for vg in vgs:
        if not vg.videos:
            logger.warning("%s: no videos.", vg.group_id)
            continue

        logger.info("Segment %s: %d videos", vg.group_id, len(vg.videos))
        for v in vg.videos:
            cam_serial = str(v.cam_serial)
            tag = f"{vg.group_id}.serial{cam_serial}"

            cand = [
                a
                for a in anchors_all
                if a.get("segment_id") == vg.group_id
                and a.get("cam_serial") == cam_serial
            ]
            if not cand:
                logger.warning("No anchors for %s cam %s", vg.group_id, cam_serial)
                continue
            logger.info("%s: %d anchors", tag, len(cand))

            # Compute matched window (anchors only)
            mw = compute_window_from_anchors(
                anchors_for_video=cand,
                fs=fs,
                audio_len_samples=audio_len_samples,
                margin_samples=0,  # keep as-is
            )

            # 1) Clip program audio to [s0, s1)
            awindow = ClipWindow(start=mw.s0, end=mw.s1, pad_head=0, pad_tail=0)
            a1_clip, a2_clip = clip_program_audio(
                a1, a2, awindow, out_audio, tag, out_fs=fs, serial_fs=fs
            )

            # 2) Clip video frames [fid0..fid1] at CFR=mw.fps
            clip_mp4 = out_video / f"{tag}_clip.mp4"
            clip_video_by_frames(Path(v.path), mw.fid0, mw.fid1, mw.fps, clip_mp4)

            # 3) Mux: copy video, add program audio
            out_path = out_video / f"{tag}_synced.mp4"
            logger.info("Mux → %s", out_path.name)
            mux_video_audio(clip_mp4, a1_clip, a2_clip, fps=None, out_path=out_path)
            produced += 1

    logger.info("Anchor-driven sync complete. Wrote %d files.", produced)
    return 0


# ---------------------------------------------------------------------------
# Single segment/camera sync
# ---------------------------------------------------------------------------


def cmd_sync_one(args: argparse.Namespace) -> int:
    """Anchor-driven sync for exactly one (segment_id, cam_serial) pair."""
    sess = run_discover(
        Path(args.audio_dir),
        Path(args.video_dir),
        default_serial_channel=args.serial_channel,
    )
    ag = sess.audiogroup

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

    fs = int(ag.serial_audio.sample_rate)
    audio_len_samples = int(fs * float(ag.serial_audio.duration))

    if not getattr(args, "anchors", None):
        raise ValueError("--anchors is required for anchor-driven sync.")
    try:
        anchors_all = json.loads(Path(args.anchors).read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to load anchors JSON ({args.anchors}): {e}") from e

    seg_id = str(args.segment_id)
    cam_serial = str(args.cam_serial)

    # find the requested Video in the discovered session
    target_video = None
    for vg in sess.videogroups:
        if vg.group_id != seg_id or not vg.videos:
            continue
        for v in vg.videos:
            if str(v.cam_serial) == cam_serial:
                target_video = (vg, v)
                break
        if target_video:
            break
    if not target_video:
        raise RuntimeError(
            f"Could not find video for segment_id='{seg_id}' and cam_serial='{cam_serial}'."
        )

    vg, v = target_video
    tag = f"{vg.group_id}.serial{cam_serial}"
    logger.info("Syncing only: segment=%s, cam=%s", vg.group_id, cam_serial)

    # Restrict anchors to this exact (segment, camera)
    cand = [
        a
        for a in anchors_all
        if a.get("segment_id") == vg.group_id and a.get("cam_serial") == cam_serial
    ]
    if not cand:
        raise RuntimeError(
            f"No anchors found for segment '{seg_id}' cam '{cam_serial}'."
        )
    logger.info("%s: %d anchors", tag, len(cand))

    # Compute matched window (anchors only)
    mw = compute_window_from_anchors(
        anchors_for_video=cand,
        fs=fs,
        audio_len_samples=audio_len_samples,
        margin_samples=0,
    )

    # Outputs
    out_audio = Path(args.out_audio)
    out_video = Path(args.out_video)

    # 1) Clip program audio to [s0, s1)
    awindow = ClipWindow(start=mw.s0, end=mw.s1, pad_head=0, pad_tail=0)
    a1_clip, a2_clip = clip_program_audio(
        a1, a2, awindow, out_audio, tag, out_fs=fs, serial_fs=fs
    )

    # 2) Clip video frames [fid0..fid1] at CFR=mw.fps
    clip_mp4 = out_video / f"{tag}_clip.mp4"
    clip_video_by_frames(Path(v.path), mw.fid0, mw.fid1, mw.fps, clip_mp4)

    # 3) Mux: copy video, add program audio
    out_path = out_video / f"{tag}_synced.mp4"
    logger.info("Mux → %s", out_path.name)
    mux_video_audio(clip_mp4, a1_clip, a2_clip, fps=None, out_path=out_path)

    logger.info("Single-segment sync complete → %s", out_path)
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

    s = sub.add_parser(
        "sync-segments",
        help="Compute per-segment windows, clip A1/A2, and mux to synced MP4s (CFR)",
    )
    s.add_argument("--audio-dir", required=True)
    s.add_argument("--video-dir", required=True)
    s.add_argument("--serial-channel", type=int, default=3)
    s.add_argument("--anchors", help="Path to anchors JSON saved during 'fit'")
    s.add_argument("--out-audio", required=True)
    s.add_argument("--out-video", required=True)
    s.add_argument(
        "--margin",
        type=int,
        default=1600,
        help="Samples of safety margin (~1 serial block)",
    )
    s.set_defaults(func=cmd_sync_segments)

    one = sub.add_parser(
        "sync-one",
        help="Sync exactly one (segment_id, cam_serial) using anchors; trims A1/A2 and muxes",
    )
    one.add_argument("--audio-dir", required=True)
    one.add_argument("--video-dir", required=True)
    one.add_argument("--serial-channel", type=int, default=3)
    one.add_argument("--segment-id", required=True, help="Target segment/group id")
    one.add_argument("--cam-serial", required=True, help="Target camera serial")
    one.add_argument("--anchors", required=True, help="Path to anchors JSON")
    one.add_argument("--out-audio", required=True)
    one.add_argument("--out-video", required=True)
    one.set_defaults(func=cmd_sync_one)

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
