#!/usr/bin/env python3
"""
Object-oriented redesign of the original discover.py script.

Goals
-----
- Keep the same core behavior: discover audio channels and chunked
  video segments based on filenames and directory contents.
- Provide clear, testable classes with small responsibilities.
- Preserve the CLI so the script can still be run directly.

Key classes
-----------
- FilePatterns: centralizes all filename regex patterns.
- Segment (dataclass): a recording segment defined by its JSON and MP4s.
- AudioDiscovery (dataclass): result of scanning the audio directory.
- DiscoveryResult (dataclass): aggregate result for audio + video.
- AudioDiscoverer: discovers channelized audio, prefers WAV over MP3.
- SegmentDiscoverer: discovers segments defined by JSON + attached MP4s.
- Discoverer: orchestrates both discoverers to produce a DiscoveryResult.

The parsing, sorting, and logging behavior mirror the original script.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "Segment",
    "AudioDiscovery",
    "DiscoveryResult",
    "FilePatterns",
    "AudioDiscoverer",
    "SegmentDiscoverer",
    "Discoverer",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    """A chunked n-minute recording segment defined by its JSON.

    Attributes
    ----------
    segment_id: The shared BASE identifier, e.g. "Test_20250101_120000".
    json_path: Path to <BASE>.json.
    camera_files: Mapping {camera_id: Path to <BASE>.<CAM>.mp4}.
    """

    segment_id: str
    json_path: Path
    camera_files: Dict[int, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioDiscovery:
    """Audio files discovered under the audio directory.

    Attributes
    ----------
    files_by_channel: Mapping {channel: Path}.
    all_files: All audio files seen (sorted), including non-matching.
    serial_channel: The default serial channel if present, else None.
    serial_path: Resolved path for the serial channel if present.
    """

    files_by_channel: Dict[int, Path]
    all_files: List[Path]
    serial_channel: Optional[int]
    serial_path: Optional[Path]


@dataclass(frozen=True)
class DiscoveryResult:
    """Full discovery result for audio + segments."""

    audio: AudioDiscovery
    segments: List[Segment]


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
class FilePatterns:
    """Centralizes filename parsing logic.

    Segment BASE ends with "_YYYYMMDD_HHMMSS".
    Video:  <BASE>.<CAM>.mp4
    JSON:   <BASE>.json
    Audio:  <prefix>-NN.(wav|mp3)
    """

    RE_TAIL = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})$")
    RE_VIDEO = re.compile(
        r"^(?P<base>.+?_\d{8}_\d{6})\.(?P<cam>\d+)\.mp4$", re.IGNORECASE
    )
    RE_JSON = re.compile(r"^(?P<base>.+?_\d{8}_\d{6})\.json$", re.IGNORECASE)
    RE_AUDIO = re.compile(
        r"^(?P<prefix>.+)-(?P<chan>\d{2})\.(?P<ext>wav|mp3)$", re.IGNORECASE
    )

    @classmethod
    def parse_video_filename(cls, p: Path) -> Optional[Tuple[str, int]]:
        m = cls.RE_VIDEO.match(p.name)
        if not m:
            return None
        base = m.group("base")
        try:
            cam_id = int(m.group("cam"))
        except ValueError:
            return None
        return base, cam_id

    @classmethod
    def parse_json_filename(cls, p: Path) -> Optional[str]:
        m = cls.RE_JSON.match(p.name)
        return m.group("base") if m else None

    @classmethod
    def parse_audio_filename(cls, p: Path) -> Optional[Tuple[int, str]]:
        m = cls.RE_AUDIO.match(p.name)
        if not m:
            return None
        try:
            ch = int(m.group("chan"))
        except ValueError:
            return None
        return ch, m.group("ext").lower()

    @classmethod
    def segment_sort_key(cls, seg_id: str) -> Tuple[int, int, str]:
        """Sort by tail date/time if present; else push to end deterministically."""
        m = cls.RE_TAIL.search(seg_id)
        if m:
            return int(m.group("date")), int(m.group("time")), seg_id
        return (10**12, 10**8, seg_id)


# ---------------------------------------------------------------------------
# Discovery engines
# ---------------------------------------------------------------------------
class _DirMixin:
    def _ensure_exists(self, directory: Path) -> None:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")


class AudioDiscoverer(_DirMixin):
    """Discovers channelized audio under a directory.

    Policy:
      - Accept *.wav/*.mp3 named like '*-NN.ext'.
      - Prefer WAV if both formats exist for the same channel.
      - Expect ~3 channels; warn if fewer/more.
    """

    def __init__(
        self,
        audio_dir: Path,
        default_serial_channel: int = 3,
        *,
        log: logging.Logger = logger,
    ):
        self.audio_dir = audio_dir
        self.default_serial_channel = default_serial_channel
        self.log = log

    def discover(self) -> AudioDiscovery:
        self._ensure_exists(self.audio_dir)

        candidates = sorted(
            [*self.audio_dir.glob("*.wav"), *self.audio_dir.glob("*.mp3")]
        )

        files_by_channel: Dict[int, Path] = {}
        all_files: List[Path] = []

        for p in candidates:
            parsed = FilePatterns.parse_audio_filename(p)
            all_files.append(p)
            if not parsed:
                self.log.warning("Skipping audio with unexpected name: %s", p.name)
                continue

            ch, ext = parsed
            existing = files_by_channel.get(ch)
            if existing is None:
                files_by_channel[ch] = p
            else:
                # Prefer WAV over MP3
                if existing.suffix.lower() == ".mp3" and ext == "wav":
                    files_by_channel[ch] = p

        files_by_channel = dict(sorted(files_by_channel.items(), key=lambda kv: kv[0]))

        if len(files_by_channel) != 3:
            ch_list = (
                _format_channels(files_by_channel.keys())
                if files_by_channel
                else "none"
            )
            self.log.warning(
                "Expected 3 audio files, found %d (channels: %s).",
                len(files_by_channel),
                ch_list,
            )

        serial_channel = (
            self.default_serial_channel
            if self.default_serial_channel in files_by_channel
            else None
        )
        serial_path = (
            files_by_channel.get(self.default_serial_channel)
            if serial_channel
            else None
        )

        if serial_channel:
            self.log.info(
                "Default serial channel %02d found: %s",
                serial_channel,
                serial_path.name if serial_path else "<missing>",
            )
        else:
            self.log.warning(
                "Default serial channel %02d not found in AUDIO_DIR.",
                self.default_serial_channel,
            )

        return AudioDiscovery(
            files_by_channel=files_by_channel,
            all_files=sorted(all_files),
            serial_channel=serial_channel,
            serial_path=serial_path,
        )


class SegmentDiscoverer(_DirMixin):
    """Discovers JSON-defined segments and attaches MP4s by camera id."""

    def __init__(self, video_dir: Path, *, log: logging.Logger = logger):
        self.video_dir = video_dir
        self.log = log

    def discover(self) -> List[Segment]:
        self._ensure_exists(self.video_dir)

        # JSONs define segments
        seg_for_json: Dict[str, Path] = {}
        for jp in sorted(self.video_dir.glob("*.json")):
            if not jp.is_file():
                continue
            seg_id = FilePatterns.parse_json_filename(jp)
            if not seg_id:
                self.log.warning("Skipping JSON with unexpected name: %s", jp.name)
                continue
            if seg_id in seg_for_json:
                self.log.warning(
                    "Duplicate JSON for %s; keeping first: %s (ignoring %s)",
                    seg_id,
                    seg_for_json[seg_id].name,
                    jp.name,
                )
                continue
            seg_for_json[seg_id] = jp

        if not seg_for_json:
            self.log.warning(
                "No valid segment JSON files found under %s", self.video_dir
            )

        # MP4s grouped by (BASE, CAM)
        vids_by_segment: Dict[str, Dict[int, Path]] = {}
        for vp in sorted(self.video_dir.glob("*.mp4")):
            if not vp.is_file():
                continue
            parsed = FilePatterns.parse_video_filename(vp)
            if not parsed:
                self.log.warning("Skipping MP4 with unexpected name: %s", vp.name)
                continue
            seg_id, cam_id = parsed
            vids_by_segment.setdefault(seg_id, {})
            if cam_id in vids_by_segment[seg_id]:
                self.log.warning(
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
                self.log.warning(
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
            self.log.warning(
                "Found MP4(s) for %s but no matching JSON; they will be ignored.",
                seg_id,
            )

        # Sort chronologically by tail
        segments.sort(key=lambda s: FilePatterns.segment_sort_key(s.segment_id))
        self.log.info("Discovered %d segment(s).", len(segments))
        return segments


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Discoverer:
    """High-level orchestrator combining audio and segment discovery."""

    def __init__(
        self,
        audio_dir: Path,
        video_dir: Path,
        *,
        default_serial_channel: int = 3,
        log: logging.Logger = logger,
    ) -> None:
        self.audio = AudioDiscoverer(audio_dir, default_serial_channel, log=log)
        self.segments = SegmentDiscoverer(video_dir, log=log)
        self.log = log

    def run(self) -> DiscoveryResult:
        audio = self.audio.discover()
        segments = self.segments.discover()
        return DiscoveryResult(audio=audio, segments=segments)


# ---------------------------------------------------------------------------
# Helpers (public API for library use, preserving function names)
# ---------------------------------------------------------------------------
def discover_audio(audio_dir: Path, default_serial_channel: int = 3) -> AudioDiscovery:
    return AudioDiscoverer(audio_dir, default_serial_channel).discover()


def discover_segments(video_dir: Path) -> List[Segment]:
    return SegmentDiscoverer(video_dir).discover()


def discover(
    audio_dir: Path, video_dir: Path, default_serial_channel: int = 3
) -> DiscoveryResult:
    return Discoverer(
        audio_dir, video_dir, default_serial_channel=default_serial_channel
    ).run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_channels(channels: Iterable[int]) -> str:
    return ", ".join(f"{ch:02d}" for ch in channels)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Discover audio (AUDIO_DIR) and chunked segments (VIDEO_DIR).",
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

    args = ap.parse_args(argv)

    res = discover(
        args.audio_dir, args.video_dir, default_serial_channel=args.serial_channel
    )

    print("\nAudio (by channel):")
    for ch, p in res.audio.files_by_channel.items():
        print(f"  - ch {ch:02d}: {p.name}")
    if res.audio.serial_channel and res.audio.serial_path is not None:
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
