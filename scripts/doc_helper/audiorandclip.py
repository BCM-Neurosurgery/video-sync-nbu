#!/usr/bin/env python3
"""
mp3_random_minclip.py
---------------------

Randomly clip a window from an MP3 and save it as WAV.

Usage examples
--------------
# 1-minute random clip (default)
python mp3_random_minclip.py /path/to/audio.mp3

# 30-second random clip (seconds override minutes)
python mp3_random_minclip.py /path/to/audio.mp3 --seconds 30

# 90-second clip to an output directory, reproducible seed
python mp3_random_minclip.py /path/to/audio.mp3 -s 90 --out-dir ./clips --seed 42

# If the MP3 is shorter than requested, allow clipping the full file
python mp3_random_minclip.py /path/to/audio.mp3 -s 75 --allow-shorter
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def hhmmss(ms: int) -> str:
    s = ms // 1000
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def pick_window(duration_ms: int, clip_ms: int, rng: random.Random) -> Tuple[int, int]:
    """Return (start_ms, end_ms) inclusive of start, exclusive of end."""
    if clip_ms > duration_ms:
        raise ValueError("Requested clip is longer than the MP3.")
    if clip_ms == duration_ms:
        return 0, duration_ms
    max_start = duration_ms - clip_ms
    start = rng.randrange(max_start + 1)
    return start, start + clip_ms


def clip_with_pydub(mp3_path: Path, start_ms: int, clip_ms: int, wav_out: Path) -> None:
    from pydub import AudioSegment  # lazy import

    audio = AudioSegment.from_file(mp3_path, format="mp3")
    slice_seg = audio[start_ms : start_ms + clip_ms]
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    slice_seg.export(wav_out, format="wav", parameters=["-acodec", "pcm_s16le"])


def clip_with_ffmpeg(
    mp3_path: Path, start_ms: int, clip_ms: int, wav_out: Path
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or use pydub with ffmpeg.")
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    start_sec = f"{start_ms/1000:.3f}"
    dur_sec = f"{clip_ms/1000:.3f}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        start_sec,
        "-t",
        dur_sec,
        "-i",
        str(mp3_path),
        "-acodec",
        "pcm_s16le",
        str(wav_out),
    ]
    log_path = wav_out.with_suffix(f"{wav_out.suffix}.ffmpeg.log")
    with log_path.open("w") as ferr:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=ferr, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed when clipping {mp3_path.name}. See log: {log_path}"
        )


def get_mp3_duration_ms(mp3_path: Path) -> int:
    """Try pydub/ffprobe to get duration (ms)."""
    try:
        from pydub.utils import mediainfo_json  # type: ignore

        info = mediainfo_json(str(mp3_path))
        dur_s = float(info.get("format", {}).get("duration", "0"))
        if dur_s > 0:
            return int(round(dur_s * 1000))
    except Exception:
        pass

    if shutil.which("ffprobe") is None:
        raise RuntimeError("Could not determine duration (need pydub or ffprobe).")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{proc.stderr}")
    dur_s = float(proc.stdout.strip())
    return int(round(dur_s * 1000))


def make_out_name(mp3_path: Path, label: str, out_dir: Optional[Path]) -> Path:
    base = mp3_path.stem
    out_dir = out_dir or mp3_path.parent
    return (out_dir / f"{base}-{label}.wav").resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Randomly clip a window from an MP3 and save it as WAV."
    )
    ap.add_argument("mp3", type=Path, help="Path to the source MP3 file")
    ap.add_argument(
        "--minutes",
        "-m",
        type=int,
        default=1,
        help="Clip length in minutes (default: 1).",
    )
    ap.add_argument(
        "--seconds",
        "-s",
        type=int,
        default=None,
        help="Clip length in seconds (overrides --minutes).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output WAV (default: alongside the MP3).",
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="Random seed (for reproducibility)."
    )
    ap.add_argument(
        "--allow-shorter",
        action="store_true",
        help="If the MP3 is shorter than requested, clip the entire file instead of erroring.",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists."
    )
    args = ap.parse_args()

    mp3_path: Path = args.mp3
    if not mp3_path.is_file():
        ap.error(f"MP3 not found: {mp3_path}")

    if args.seconds is not None and args.seconds <= 0:
        ap.error("--seconds must be a positive integer")
    if args.seconds is None and args.minutes <= 0:
        ap.error("--minutes must be a positive integer")

    rng = random.Random(args.seed)

    # Determine clip length and output label
    if args.seconds is not None:
        clip_ms = args.seconds * 1000
        label = f"{args.seconds}secclip"
    else:
        clip_ms = args.minutes * 60_000
        label = f"{args.minutes}minclip"

    out_wav = make_out_name(mp3_path, label, args.out_dir)
    if out_wav.exists() and not args.overwrite:
        ap.error(f"Output exists: {out_wav} (use --overwrite)")

    duration_ms = get_mp3_duration_ms(mp3_path)
    actual_clip_ms = clip_ms
    if duration_ms < clip_ms:
        if not args.allow_shorter:
            want = (
                f"{args.seconds}s" if args.seconds is not None else f"{args.minutes}m"
            )
            ap.error(
                f"MP3 is only {hhmmss(duration_ms)} long; requested {want}. "
                "Use --allow-shorter to clip the full file."
            )
        actual_clip_ms = duration_ms

    start_ms, end_ms = pick_window(duration_ms, actual_clip_ms, rng)
    print(
        f"Selected window: {hhmmss(start_ms)} → {hhmmss(end_ms)} "
        f"({(end_ms - start_ms)/1000:.3f}s) from {mp3_path.name}"
    )

    # Try pydub first; fall back to ffmpeg
    try:
        clip_with_pydub(mp3_path, start_ms, end_ms - start_ms, out_wav)
    except Exception as e_pydub:
        if shutil.which("ffmpeg"):
            print("pydub failed; falling back to ffmpeg…")
            clip_with_ffmpeg(mp3_path, start_ms, end_ms - start_ms, out_wav)
        else:
            raise RuntimeError(
                f"Clipping failed via pydub and ffmpeg is not available.\n{e_pydub}"
            ) from e_pydub

    print(f"Saved: {out_wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
