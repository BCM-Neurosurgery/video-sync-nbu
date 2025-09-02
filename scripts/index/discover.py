#!/usr/bin/env python3
"""
Object-oriented discoverer that returns an AudioVideoSession populated
with models from scripts.models.

- Audio: builds Audio (or SerialAudio for the serial channel), grouped in AudioGroup
- Video: groups MP4s by segment (BASE) with a Json wrapper and Video list
- Session: aggregates into AudioVideoSession and computes shared_cam_serials

Assumptions
-----------
- Audio files are named like "<PREFIX>-NN.(wav|mp3)" where NN ∈ {01,02,03}
- Video files are named like "<BASE>.<CAM_SERIAL>.mp4"
- JSON files are named  "<BASE>.json"
- BASE ends with "_YYYYMMDD_HHMMSS" (local America/Chicago)

This script is metadata-light: WAV sample_rate/duration are probed via the
standard library; MP3/video metadata fall back to placeholders.

Update
------
- If --segment-id is provided, the script returns exactly one VideoGroup
  for that segment, including all cameras found. No camera-serial filter.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from scripts.index.filepatterns import FilePatterns
from scripts.index.audiodiscover import AudioDiscoverer
from scripts.index.videodiscover import VideoDiscoverer

from scripts.models import (
    AudioGroup,
    VideoGroup,
    AudioVideoSession,
)

from scripts.log.logutils import configure_standalone_logging, log_context

__all__ = [
    "discover_audio",
    "discover_segments",
    "discover_segment",
    "discover",
    "FilePatterns",
    "AudioDiscoverer",
    "VideoDiscoverer",
    "Discoverer",
]

# ---------------------------------------------------------------------------
# Logging (library module: no handlers/levels; let the driver configure root)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Discoverer:
    """High-level orchestrator combining audio and video discovery."""

    def __init__(
        self,
        audio_dir: Path,
        video_dir: Path,
        *,
        default_serial_channel: int = 3,
        log: logging.Logger = logger,
    ) -> None:
        self.audio = AudioDiscoverer(audio_dir, default_serial_channel, log=log)
        self.video = VideoDiscoverer(video_dir, log=log)
        self.log = log

    def _compute_shared_cam_serials(
        self, groups: List[VideoGroup]
    ) -> Optional[List[str]]:
        if not groups:
            return None
        sets = [set(vg.cam_serials or []) for vg in groups if vg.cam_serials]
        if not sets:
            return None
        inter = set.intersection(*sets) if len(sets) > 1 else next(iter(sets))
        return sorted(inter) if inter else None

    def run(self) -> AudioVideoSession:
        audiogroup = self.audio.get_audio_group()
        videogroups = self.video.discover()
        shared = self._compute_shared_cam_serials(videogroups)
        return AudioVideoSession(
            audiogroup=audiogroup,
            videogroups=videogroups,
            shared_cam_serials=shared,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def discover_audio(
    audio_dir: Path, log: logging.Logger, default_serial_channel: int = 3
) -> AudioGroup:
    return AudioDiscoverer(audio_dir, default_serial_channel, log=log).get_audio_group()


def discover_segments(video_dir: Path, log: logging.Logger) -> List[VideoGroup]:
    return VideoDiscoverer(video_dir, log=log).discover()


def discover_segment(
    video_dir: Path, segment_id: str, log: logging.Logger
) -> Optional[VideoGroup]:
    return VideoDiscoverer(video_dir, log=log).discover_one(segment_id)


def discover(
    audio_dir: Path,
    video_dir: Path,
    log: logging.Logger,
    default_serial_channel: int = 3,
) -> AudioVideoSession:
    return Discoverer(
        audio_dir, video_dir, default_serial_channel=default_serial_channel, log=log
    ).run()


# ---------------------------------------------------------------------------
# CLI presentation helpers
# ---------------------------------------------------------------------------
def _print_audiogroup(ag: AudioGroup) -> None:
    print("\nAudioGroup:")
    for ch in sorted(ag.audios.keys()):
        a = ag.audios[ch]
        print(
            f"  ch {ch:02d}: {a.path.name} "
            f"(ext={a.extension}, sr={a.sample_rate}, dur={a.duration:.2f}s)"
        )
    if ag.serial_audio:
        print(
            f"  serial_audio: ch {ag.serial_audio.channel:02d} -> {ag.serial_audio.path.name}"
        )
    if ag.shared_extension:
        print(f"  shared_extension: {ag.shared_extension}")


def _print_videogroups(vgs: List[VideoGroup]) -> None:
    print("\nVideoGroups:")
    for vg in vgs:
        ts = vg.timestamp.isoformat() if vg.timestamp else "None"
        print(f"  * {vg.group_id}  ts={ts}  JSON={vg.json.path.name}")
        if vg.videos:
            for v in vg.videos:
                print(f"      cam {v.cam_serial}: {v.path.name}")
        if vg.cam_serials:
            print(f"      cam_serials: {', '.join(vg.cam_serials)}")


def _print_shared(shared: Optional[List[str]]) -> None:
    if shared:
        print(f"\nShared cam serials across groups: {', '.join(shared)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Discover audio (AUDIO_DIR) and chunked segments (VIDEO_DIR).",
    )
    # Make audio-dir optional (required only for full discovery)
    ap.add_argument("--audio-dir", type=Path, help="Directory containing audio files")
    ap.add_argument(
        "--video-dir", type=Path, required=True, help="Directory with JSON/MP4 files"
    )
    ap.add_argument("--serial-channel", type=int, default=3)

    # Fast single-segment option
    ap.add_argument(
        "--segment-id",
        type=str,
        help="Exact segment BASE like 'TRBD001_20250715_143011'",
    )

    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (standalone only; ignored when called from driver)",
    )

    args = ap.parse_args(argv)

    # Standalone: configure minimal console logging (no-op under driver),
    # and set a log context so messages carry [seg/-] when segment is known.
    configure_standalone_logging(args.log_level, seg=(args.segment_id or "-"), cam="-")

    # Fast path: just one VideoGroup (no audio scan, all cameras included)
    if args.segment_id:
        with log_context(seg=args.segment_id, cam="-"):
            vg = VideoDiscoverer(args.video_dir).discover_one(args.segment_id)
            if not vg:
                logger.error("Nothing found for segment %s", args.segment_id)
                return 1
            _print_videogroups([vg])
            return 0

    # Full discovery path requires audio-dir
    if not args.audio_dir:
        logger.error("--audio-dir is required when not using --segment-id")
        return 2

    with log_context(seg="-", cam="-"):
        session = discover(
            args.audio_dir, args.video_dir, default_serial_channel=args.serial_channel
        )

    _print_audiogroup(session.audiogroup)
    _print_videogroups(session.videogroups)
    _print_shared(session.shared_cam_serials)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
