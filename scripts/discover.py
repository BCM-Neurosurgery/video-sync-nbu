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
"""

from __future__ import annotations

import argparse
import logging
import re
import wave
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, List, Optional, Tuple

# Try optional MP3 probe (pydub)
try:
    from pydub import AudioSegment  # type: ignore

    _HAVE_PYDUB = True
except Exception:
    _HAVE_PYDUB = False

# Import your dataclasses
from scripts.models import (
    CamJson,  # not used yet, but kept for completeness
    Json,
    Audio,
    SerialAudio,
    Video,
    AudioGroup,
    VideoGroup,
    AudioVideoSession,
)

__all__ = [
    "discover_audio",
    "discover_segments",
    "discover",
    "FilePatterns",
    "AudioDiscoverer",
    "VideoDiscoverer",
    "Discoverer",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

DEFAULT_TZ = ZoneInfo("America/Chicago")


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
class FilePatterns:
    """Centralizes filename parsing logic."""

    RE_TAIL = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})$")
    RE_VIDEO = re.compile(
        r"^(?P<base>.+?_\d{8}_\d{6})\.(?P<cam>[0-9A-Za-z]+)\.mp4$", re.IGNORECASE
    )
    RE_JSON = re.compile(r"^(?P<base>.+?_\d{8}_\d{6})\.json$", re.IGNORECASE)
    RE_AUDIO = re.compile(
        r"^(?P<prefix>.+)-(?P<chan>\d{2})\.(?P<ext>wav|mp3)$", re.IGNORECASE
    )

    @classmethod
    def parse_video_filename(cls, p: Path) -> Optional[Tuple[str, str]]:
        m = cls.RE_VIDEO.match(p.name)
        if not m:
            return None
        return m.group("base"), m.group("cam")  # keep cam as string (serial)

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
    def videogroup_sort_key(cls, seg_id: str) -> Tuple[int, int, str]:
        m = cls.RE_TAIL.search(seg_id)
        if m:
            return int(m.group("date")), int(m.group("time")), seg_id
        return (10**12, 10**8, seg_id)

    @classmethod
    def parse_tail_datetime(
        cls, seg_id: str, tz: ZoneInfo = DEFAULT_TZ
    ) -> Optional[datetime]:
        m = cls.RE_TAIL.search(seg_id)
        if not m:
            return None
        dt = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=tz)


# ---------------------------------------------------------------------------
# Small probes
# ---------------------------------------------------------------------------
def _filesize_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def _probe_wav(p: Path) -> Tuple[float, int]:
    """Return (duration_sec, sample_rate) for WAV, else (0.0, 0)."""
    try:
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            dur = float(n) / sr if sr else 0.0
            return dur, sr
    except Exception:
        return 0.0, 0


def _probe_mp3(p: Path) -> Tuple[float, int]:
    """Return (duration_sec, sample_rate) for MP3 using pydub if available, else (0.0, 0)."""
    if not _HAVE_PYDUB:
        return 0.0, 0
    try:
        seg = AudioSegment.from_file(p)
        dur = len(seg) / 1000.0
        sr = seg.frame_rate
        return dur, sr
    except Exception:
        return 0.0, 0


# ---------------------------------------------------------------------------
# Discovery engines
# ---------------------------------------------------------------------------
class _DirMixin:
    def _ensure_exists(self, directory: Path) -> None:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")


class AudioDiscoverer(_DirMixin):
    """Discovers channelized audio and builds an AudioGroup."""

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

    def _build_audio_obj(self, ch: int, p: Path) -> Audio:
        ext = p.suffix.lower().lstrip(".")
        if ext == "wav":
            dur, sr = _probe_wav(p)
        elif ext == "mp3":
            dur, sr = _probe_mp3(p)
        else:
            dur, sr = 0.0, 0
        return Audio(
            path=p,
            duration=dur,
            file_size=_filesize_mb(p),
            sample_rate=sr,
            extension=ext,
            channel=ch,
        )

    def _build_serial_audio_obj(self, ch: int, p: Path) -> SerialAudio:
        base = self._build_audio_obj(ch, p)
        # SerialAudio subclasses Audio; dataclass copying is straightforward:
        return SerialAudio(
            path=base.path,
            duration=base.duration,
            file_size=base.file_size,
            sample_rate=base.sample_rate,
            extension=base.extension,
            channel=base.channel,
        )

    def discover(self) -> AudioGroup:
        self._ensure_exists(self.audio_dir)

        candidates = sorted(
            [*self.audio_dir.glob("*.wav"), *self.audio_dir.glob("*.mp3")]
        )

        chosen_by_channel: Dict[int, Path] = {}
        for p in candidates:
            parsed = FilePatterns.parse_audio_filename(p)
            if not parsed:
                self.log.warning("Skipping audio with unexpected name: %s", p.name)
                continue
            ch, ext = parsed
            existing = chosen_by_channel.get(ch)
            if existing is None:
                chosen_by_channel[ch] = p
            else:
                # Prefer WAV over MP3
                if existing.suffix.lower() == ".mp3" and ext == "wav":
                    chosen_by_channel[ch] = p

        if len(chosen_by_channel) != 3:
            ch_list = _format_channels(chosen_by_channel.keys()) or "none"
            self.log.warning(
                "Expected 3 audio files, found %d (channels: %s).",
                len(chosen_by_channel),
                ch_list,
            )

        # Shared extension (if uniform)
        exts = {p.suffix.lower().lstrip(".") for p in chosen_by_channel.values()}
        shared_ext: Optional[str] = None
        if len(exts) == 1 and exts <= {"wav", "mp3"}:
            shared_ext = next(iter(exts))
        elif exts:
            self.log.warning(
                "Mixed audio extensions detected across channels: %s.",
                ", ".join(sorted(exts)),
            )

        # Build Audio / SerialAudio objects
        audios: Dict[int, Audio] = {}
        serial_audio: Optional[SerialAudio] = None
        for ch, p in sorted(chosen_by_channel.items(), key=lambda kv: kv[0]):
            if ch == self.default_serial_channel:
                serial_audio = self._build_serial_audio_obj(ch, p)
                audios[ch] = serial_audio  # SerialAudio is an Audio
            else:
                audios[ch] = self._build_audio_obj(ch, p)

        if serial_audio is None:
            self.log.warning(
                "Default serial channel %02d not found in AUDIO_DIR.",
                self.default_serial_channel,
            )

        return AudioGroup(
            audios=audios,
            serial_audio=serial_audio,
            shared_extension=shared_ext,
        )


class VideoDiscoverer(_DirMixin):
    """Discovers JSON-defined segments and attaches MP4s by camera serial."""

    def __init__(self, video_dir: Path, *, log: logging.Logger = logger):
        self.video_dir = video_dir
        self.log = log

    def discover(self) -> List[VideoGroup]:
        self._ensure_exists(self.video_dir)

        # JSONs define segments
        json_by_seg: Dict[str, Path] = {}
        for jp in sorted(self.video_dir.glob("*.json")):
            if not jp.is_file():
                continue
            seg_id = FilePatterns.parse_json_filename(jp)
            if not seg_id:
                self.log.warning("Skipping JSON with unexpected name: %s", jp.name)
                continue
            if seg_id in json_by_seg:
                self.log.warning(
                    "Duplicate JSON for %s; keeping first: %s (ignoring %s)",
                    seg_id,
                    json_by_seg[seg_id].name,
                    jp.name,
                )
                continue
            json_by_seg[seg_id] = jp

        if not json_by_seg:
            self.log.warning(
                "No valid segment JSON files found under %s", self.video_dir
            )

        # MP4s grouped by (BASE -> {cam_serial -> Path})
        vids_by_seg: Dict[str, Dict[str, Path]] = {}
        for vp in sorted(self.video_dir.glob("*.mp4")):
            if not vp.is_file():
                continue
            parsed = FilePatterns.parse_video_filename(vp)
            if not parsed:
                self.log.warning("Skipping MP4 with unexpected name: %s", vp.name)
                continue
            seg_id, cam_serial = parsed
            vids_by_seg.setdefault(seg_id, {})
            if cam_serial in vids_by_seg[seg_id]:
                self.log.warning(
                    "Duplicate MP4 for %s cam %s; keeping first: %s (ignoring %s)",
                    seg_id,
                    cam_serial,
                    vids_by_seg[seg_id][cam_serial].name,
                    vp.name,
                )
                continue
            vids_by_seg[seg_id][cam_serial] = vp

        # Build VideoGroup list for segments that have JSONs
        videogroups: List[VideoGroup] = []
        for seg_id, json_path in json_by_seg.items():
            ts = FilePatterns.parse_tail_datetime(seg_id, DEFAULT_TZ)
            cams = vids_by_seg.get(seg_id, {})

            # Build Video objects (meta placeholders)
            videos: List[Video] = []
            for cam_serial, mp4_path in sorted(cams.items(), key=lambda kv: kv[0]):
                videos.append(
                    Video(
                        path=mp4_path,
                        cam_serial=str(cam_serial),
                        timestamp=ts,
                        duration=0.0,  # placeholder – probe via ffprobe if desired
                        resolution="",  # placeholder
                        frame_rate=0.0,  # placeholder
                    )
                )

            cam_serials = [v.cam_serial for v in videos] if videos else []

            # Build Json wrapper for the segment (cam_jsons left empty for now)
            json_wrap = Json(
                cam_serials=cam_serials or None,
                timestamp=ts,
                path=json_path,
                cam_jsons={},  # could be populated by a separate JSON parser later
            )

            if not videos:
                self.log.warning(
                    "No MP4s found for segment %s (JSON: %s)", seg_id, json_path.name
                )

            videogroups.append(
                VideoGroup(
                    group_id=seg_id,
                    timestamp=ts,
                    json=json_wrap,
                    videos=videos or None,
                    cam_serials=cam_serials or None,
                )
            )

        # Orphan MP4s (no JSON)
        orphans = sorted(set(vids_by_seg.keys()) - set(json_by_seg.keys()))
        for seg_id in orphans:
            self.log.warning(
                "Found MP4(s) for %s but no matching JSON; they will be ignored.",
                seg_id,
            )

        # Sort chronologically by tail
        videogroups.sort(key=lambda s: FilePatterns.videogroup_sort_key(s.group_id))
        self.log.info("Discovered %d segment(s).", len(videogroups))
        return videogroups


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

    def run(self) -> AudioVideoSession:
        audiogroup = self.audio.discover()
        videogroups = self.video.discover()

        # Compute shared cam serials across all groups (intersection)
        shared: Optional[List[str]] = None
        if videogroups:
            sets = [set(vg.cam_serials or []) for vg in videogroups if vg.cam_serials]
            if sets:
                inter = set.intersection(*sets) if len(sets) > 1 else next(iter(sets))
                shared = sorted(inter) if inter else None

        return AudioVideoSession(
            audiogroup=audiogroup,
            videogroups=videogroups,
            shared_cam_serials=shared,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def discover_audio(audio_dir: Path, default_serial_channel: int = 3) -> AudioGroup:
    return AudioDiscoverer(audio_dir, default_serial_channel).discover()


def discover_segments(video_dir: Path) -> List[VideoGroup]:
    return VideoDiscoverer(video_dir).discover()


def discover(
    audio_dir: Path, video_dir: Path, default_serial_channel: int = 3
) -> AudioVideoSession:
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
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--serial-channel", type=int, default=3)

    args = ap.parse_args(argv)

    session = discover(
        args.audio_dir, args.video_dir, default_serial_channel=args.serial_channel
    )

    # Pretty print
    ag = session.audiogroup
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

    print("\nVideoGroups:")
    for vg in session.videogroups:
        ts = vg.timestamp.isoformat() if vg.timestamp else "None"
        print(f"  * {vg.group_id}  ts={ts}  JSON={vg.json.path.name}")
        if vg.videos:
            for v in vg.videos:
                print(f"      cam {v.cam_serial}: {v.path.name}")
        if vg.cam_serials:
            print(f"      cam_serials: {', '.join(vg.cam_serials)}")

    if session.shared_cam_serials:
        print(
            f"\nShared cam serials across groups: {', '.join(session.shared_cam_serials)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
