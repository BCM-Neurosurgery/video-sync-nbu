#!/usr/bin/env python3
"""
discover.py — discover audio (3 files) and chunked video segments (many files).

Audio examples (in AUDIO_DIR):
  VideoTest03062025-01.wav / .mp3
  VideoTest03062025-02.wav / .mp3
  VideoTest03062025-03.wav / .mp3   # default serial channel

Video/JSON examples (in VIDEO_DIR):
  TestVideo03062025_20250306_153829.24253445.mp4
  TestVideo03062025_20250306_153829.json

We group by BASE = '<anything>_YYYYMMDD_HHMMSS'.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "Segment",
    "AudioDiscovery",
    "DiscoveryResult",
    "discover_audio",
    "discover_segments",
    "discover",
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------- Data models ----------


@dataclass(frozen=True)
class Segment:
    """A chunked 10-min (or n-min) recording segment defined by its JSON."""

    segment_id: str  # BASE, e.g. "TestVideo03062025_20250306_153829"
    json_path: Path  # BASE.json
    camera_files: Dict[int, Path] = field(default_factory=dict)  # {camera_id: mp4 path}


@dataclass(frozen=True)
class AudioDiscovery:
    """Audio files discovered under AUDIO_DIR."""

    files_by_channel: Dict[int, Path]  # {1: Path(...-01.wav), 2: ..., 3: ...}
    all_files: List[Path]  # all audio files seen (sorted)
    serial_channel: Optional[int]  # e.g., 3 if present
    serial_path: Optional[Path]  # resolved path for serial channel (if present)


@dataclass(frozen=True)
class DiscoveryResult:
    """Full discovery result from AUDIO_DIR + VIDEO_DIR."""

    audio: AudioDiscovery
    segments: List[Segment]


# ---------- Filename patterns ----------

# Segment tail must end with "_YYYYMMDD_HHMMSS"
_RE_TAIL = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})$")
_RE_VIDEO = re.compile(r"^(?P<base>.+?_\d{8}_\d{6})\.(?P<cam>\d+)\.mp4$", re.IGNORECASE)
_RE_JSON = re.compile(r"^(?P<base>.+?_\d{8}_\d{6})\.json$", re.IGNORECASE)

# Audio: any prefix, then "-NN.(wav|mp3)"
_RE_AUDIO = re.compile(
    r"^(?P<prefix>.+)-(?P<chan>\d{2})\.(?P<ext>wav|mp3)$", re.IGNORECASE
)


def _parse_video_filename(p: Path) -> Optional[Tuple[str, int]]:
    m = _RE_VIDEO.match(p.name)
    if not m:
        return None
    base = m.group("base")
    try:
        cam_id = int(m.group("cam"))
    except ValueError:
        return None
    return base, cam_id


def _parse_json_filename(p: Path) -> Optional[str]:
    m = _RE_JSON.match(p.name)
    return m.group("base") if m else None


def _parse_audio_filename(p: Path) -> Optional[Tuple[int, str]]:
    m = _RE_AUDIO.match(p.name)
    if not m:
        return None
    try:
        ch = int(m.group("chan"))
    except ValueError:
        return None
    return ch, m.group("ext").lower()


def _segment_sort_key(seg_id: str) -> Tuple[int, int, str]:
    m = _RE_TAIL.search(seg_id)
    if m:
        return int(m.group("date")), int(m.group("time")), seg_id
    return (10**12, 10**8, seg_id)


# ---------- Discovery (public) ----------


def discover_audio(audio_dir: Path, default_serial_channel: int = 3) -> AudioDiscovery:
    """
    Discover audio in AUDIO_DIR:
      - Accept *.wav/*.mp3 named like '*-NN.ext'
      - Prefer WAV if both WAV/MP3 exist for the same channel NN
      - Expect ~3 channels; warn if fewer/more
    """
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    candidates = sorted([*audio_dir.glob("*.wav"), *audio_dir.glob("*.mp3")])

    files_by_channel: Dict[int, Path] = {}
    all_files: List[Path] = []

    for p in candidates:
        parsed = _parse_audio_filename(p)
        all_files.append(p)
        if not parsed:
            logger.warning("Skipping audio with unexpected name: %s", p.name)
            continue

        ch, ext = parsed
        existing = files_by_channel.get(ch)
        if existing is None:
            files_by_channel[ch] = p
        else:
            # Prefer WAV over MP3
            if existing.suffix.lower() == ".mp3" and ext == "wav":
                files_by_channel[ch] = p

    # Sort channels
    files_by_channel = dict(sorted(files_by_channel.items(), key=lambda kv: kv[0]))

    # Quick sanity check (3 channels typical)
    if len(files_by_channel) != 3:
        logger.warning(
            "Expected 3 audio files, found %d (channels: %s).",
            len(files_by_channel),
            ", ".join(str(k).zfill(2) for k in files_by_channel.keys()) or "none",
        )

    # Serial channel
    serial_channel = (
        default_serial_channel if default_serial_channel in files_by_channel else None
    )
    serial_path = (
        files_by_channel.get(default_serial_channel) if serial_channel else None
    )

    if serial_channel:
        logger.info(
            "Default serial channel %02d found: %s", serial_channel, serial_path.name
        )
    else:
        logger.warning(
            "Default serial channel %02d not found in AUDIO_DIR.",
            default_serial_channel,
        )

    return AudioDiscovery(
        files_by_channel=files_by_channel,
        all_files=sorted(all_files),
        serial_channel=serial_channel,
        serial_path=serial_path,
    )


def discover_segments(video_dir: Path) -> List[Segment]:
    """
    Discover chunked segments in VIDEO_DIR:
      - A valid segment must have exactly one JSON: <BASE>.json
      - Attach all MP4s that match <BASE>.<CAM>.mp4
      - Warn on orphans (MP4 with no JSON) and JSON with no MP4s
    """
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    # JSONs define segments
    seg_for_json: Dict[str, Path] = {}
    for jp in sorted(video_dir.glob("*.json")):
        if not jp.is_file():
            continue
        seg_id = _parse_json_filename(jp)
        if not seg_id:
            logger.warning("Skipping JSON with unexpected name: %s", jp.name)
            continue
        if seg_id in seg_for_json:
            logger.warning(
                "Duplicate JSON for %s; keeping first: %s (ignoring %s)",
                seg_id,
                seg_for_json[seg_id].name,
                jp.name,
            )
            continue
        seg_for_json[seg_id] = jp

    if not seg_for_json:
        logger.warning("No valid segment JSON files found under %s", video_dir)

    # MP4s grouped by (BASE, CAM)
    vids_by_segment: Dict[str, Dict[int, Path]] = {}
    for vp in sorted(video_dir.glob("*.mp4")):
        if not vp.is_file():
            continue
        parsed = _parse_video_filename(vp)
        if not parsed:
            logger.warning("Skipping MP4 with unexpected name: %s", vp.name)
            continue
        seg_id, cam_id = parsed
        vids_by_segment.setdefault(seg_id, {})
        if cam_id in vids_by_segment[seg_id]:
            logger.warning(
                "Duplicate MP4 for %s cam %d; keeping first: %s (ignoring %s)",
                seg_id,
                cam_id,
                vids_by_segment[seg_id][cam_id].name,
                vp.name,
            )
            continue
        vids_by_segment[seg_id][cam_id] = vp

    # Build segments for those with JSON
    segments: List[Segment] = []
    for seg_id, json_path in seg_for_json.items():
        cams = vids_by_segment.get(seg_id, {})
        if not cams:
            logger.warning(
                "No MP4s found for segment %s (JSON: %s)", seg_id, json_path.name
            )
        segments.append(
            Segment(
                segment_id=seg_id,
                json_path=json_path,
                camera_files=dict(sorted(cams.items(), key=lambda kv: kv[0])),
            )
        )

    # Orphan MP4s (no JSON)
    orphans = sorted(set(vids_by_segment.keys()) - set(seg_for_json.keys()))
    for seg_id in orphans:
        logger.warning(
            "Found MP4(s) for %s but no matching JSON; they will be ignored.", seg_id
        )

    # Sort chronologically by tail
    segments.sort(key=lambda s: _segment_sort_key(s.segment_id))
    logger.info("Discovered %d segment(s).", len(segments))
    return segments


def discover(
    audio_dir: Path, video_dir: Path, default_serial_channel: int = 3
) -> DiscoveryResult:
    """Convenience wrapper: discover audio from AUDIO_DIR + segments from VIDEO_DIR."""
    audio = discover_audio(audio_dir, default_serial_channel=default_serial_channel)
    segments = discover_segments(video_dir)
    return DiscoveryResult(audio=audio, segments=segments)


# ---------- CLI for a quick report ----------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Discover audio (AUDIO_DIR) and chunked segments (VIDEO_DIR)."
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory with ~3 audio files (wav/mp3).",
    )
    ap.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory with chunked JSON/MP4 files.",
    )
    ap.add_argument(
        "--serial-channel",
        type=int,
        default=3,
        help="Default serial channel (e.g., 3).",
    )
    args = ap.parse_args()

    res = discover(
        args.audio_dir, args.video_dir, default_serial_channel=args.serial_channel
    )

    print("\nAudio (by channel):")
    for ch, p in res.audio.files_by_channel.items():
        print(f"  - ch {ch:02d}: {p.name}")
    if res.audio.serial_channel:
        print(
            f"Serial channel: {res.audio.serial_channel:02d} -> {res.audio.serial_path.name}"
        )
    else:
        print("Serial channel not found.")

    print("\nSegments:")
    for seg in res.segments:
        print(f"  * {seg.segment_id}  JSON={seg.json_path.name}")
        for cam_id, mp4 in seg.camera_files.items():
            print(f"      cam {cam_id}: {mp4.name}")
