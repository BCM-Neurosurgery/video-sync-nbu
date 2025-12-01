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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import shutil
import subprocess

from scripts.index.audiodiscover import AudioDiscoverer
from scripts.models import AudioGroup, Video

logger = logging.getLogger(__name__)


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
        log_path = out_path.with_suffix(f"{out_path.suffix}.ffmpeg.log")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        base = f"atrim=start={start_sec:.9f}:end={end_sec:.9f},asetpts=PTS-STARTPTS"
        filt = (
            base
            if out_fs is None
            else f"{base},aresample=sample_rate={out_fs}:resampler=soxr"
        )
        logger.info("Trimming %s → %s", in_path.name, out_path.name)
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
        with log_path.open("w") as ferr:
            proc = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=ferr, text=True
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg trim failed for '{in_path}'. See log: {log_path}"
            )
        logger.info("Trim finished %s → %s", in_path.name, out_path.name)

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

    log_path = out_path.with_suffix(f"{out_path.suffix}.ffmpeg.log")
    with log_path.open("w") as ferr:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed during mux.\n"
            f"Command: {' '.join(cmd)}\n"
            f"See log: {log_path}"
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
    log_path = out_path.with_suffix(f"{out_path.suffix}.ffmpeg.log")
    with log_path.open("w") as ferr:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed during video frame-trim.\n"
            f"Command: {' '.join(cmd)}\n"
            f"See log: {log_path}"
        )
    return out_path


# ---------------------------------------------------------------------------
# Public API — single segment/camera sync
# ---------------------------------------------------------------------------


def sync_one_video(
    *,
    audio_dir: Union[str, Path],
    video: Video,
    anchors_json: Union[str, Path],
    out_audio_dir: Union[str, Path],
    out_video_dir: Union[str, Path],
) -> Path:
    """
    Programmatic API to sync exactly one (segment_id, cam_serial) pair using anchors,
    discovering audio from `audio_dir` and operating directly on the provided `video`.

    Steps
    -----
      1) Discover AudioGroup (A1/A2 program, A3 serial) from `audio_dir`.
      2) Load & filter anchors for (segment_id, video.cam_serial); compute matched window.
      3) Trim A1/A2 WAVs to audio window [s0, s1).
      4) Frame-accurate clip of `video` to [fid0..fid1] at CFR = mw.fps.
      5) Mux clipped video with the two program-audio tracks.

    Returns
    -------
    Path
        Absolute path to the final muxed MP4 file.

    Raises
    ------
    FileNotFoundError
        If `anchors_json` does not exist or ffmpeg is not available.
    RuntimeError, ValueError, AssertionError
        For discovery failures, missing media, malformed anchors, or processing errors.
    """
    vpath = Path(video.path)
    cam_serial = str(video.cam_serial)
    anchors_path = Path(anchors_json)
    if not anchors_path.exists():
        raise FileNotFoundError(f"Anchors JSON not found: {anchors_path}")

    ad = AudioDiscoverer(
        audio_dir=Path(audio_dir),
        log=logger,
    )
    ag: AudioGroup = ad.get_audio_group()

    assert ag.serial_audio is not None, "No serial channel found."
    serial_ch = ag.serial_audio.channel
    prog_channels = [ch for ch in sorted(ag.audios.keys()) if ch != serial_ch]
    assert len(prog_channels) >= 1, "No program audio channels found in AudioGroup."

    a1 = Path(ag.audios[prog_channels[0]].path)
    a2 = (
        Path(ag.audios[prog_channels[1]].path)
        if len(prog_channels) > 1
        else Path(ag.audios[prog_channels[0]].path)
    )

    fs = int(ag.serial_audio.sample_rate)
    audio_len_samples = int(fs * float(ag.serial_audio.duration))

    tag = f"{video.segment_id}.serial{cam_serial}"
    logger.info(
        "Syncing single video: segment=%s, cam=%s → %s",
        video.segment_id,
        cam_serial,
        vpath.name,
    )

    # ---- Load & filter anchors ----------------------------------------------
    try:
        anchors_all = json.loads(anchors_path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to load anchors JSON ({anchors_path}): {e}") from e

    cand = [
        a
        for a in anchors_all
        if a.get("segment_id") == video.segment_id and a.get("cam_serial") == cam_serial
    ]
    if not cand:
        raise RuntimeError(
            f"No anchors found for segment '{video.segment_id}' cam '{cam_serial}'."
        )
    logger.info("%s: %d anchors", tag, len(cand))

    # ---- Compute matched window ---------------------------------------------
    mw = compute_window_from_anchors(
        anchors_for_video=cand,
        fs=fs,
        audio_len_samples=audio_len_samples,
        margin_samples=0,
    )
    # Expect mw: s0, s1 (audio sample indices); fid0, fid1 (inclusive frame ids); fps (CFR)
    logger.info(
        "%s: matched window audio=[%d, %d) frames=[%d..%d] @ %.6f fps",
        tag,
        mw.s0,
        mw.s1,
        mw.fid0,
        mw.fid1,
        mw.fps,
    )

    # ---- Prepare outputs -----------------------------------------------------
    out_audio = Path(out_audio_dir)
    out_audio.mkdir(parents=True, exist_ok=True)
    out_video = Path(out_video_dir)
    out_video.mkdir(parents=True, exist_ok=True)

    # 1) Clip program audio to [s0, s1)
    awindow = ClipWindow(start=mw.s0, end=mw.s1, pad_head=0, pad_tail=0)
    a1_clip, a2_clip = clip_program_audio(
        a1, a2, awindow, out_audio, tag, out_fs=fs, serial_fs=fs
    )

    # 2) Clip video frames [fid0..fid1] at CFR=mw.fps
    clip_mp4 = out_video / f"{tag}_clip.mp4"
    logger.info(
        "%s: clipping video frames [%d..%d] @ %.6f fps → %s",
        tag,
        mw.fid0,
        mw.fid1,
        mw.fps,
        clip_mp4.name,
    )
    clip_video_by_frames(vpath, mw.fid0, mw.fid1, mw.fps, clip_mp4)

    # 3) Mux: copy (clipped) video timing, add program audio
    out_path = out_video / f"{tag}_synced.mp4"
    logger.info("%s: muxing video+audio → %s", tag, out_path.name)
    # If muxer supports it, letting fps=None preserves the clipped video timestamps.
    mux_video_audio(clip_mp4, a1_clip, a2_clip, fps=None, out_path=out_path)

    logger.info("Single-video sync complete → %s", out_path.name)
    return out_path.resolve()


# ---------------------------------------------------------------------------
# CLI — single segment/camera sync
# ---------------------------------------------------------------------------


def cmd_sync_one(args: argparse.Namespace) -> int:
    """Anchor-driven sync for exactly one (segment_id, cam_serial) pair."""
    try:
        out_path = sync_one_video(
            audio_dir=args.audio_dir,
            video_dir=args.video_dir,
            segment_id=str(args.segment_id),
            cam_serial=str(args.cam_serial),
            anchors_json=args.anchors,
            out_audio_dir=args.out_audio,
            out_video_dir=args.out_video,
            serial_channel=args.serial_channel,
        )
        logger.info("Wrote %s", out_path)
        return 0
    except Exception as e:
        logger.exception(e)
        return 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sync",
        description="Audio/Video sync orchestrator (models.py + discover.py)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

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
