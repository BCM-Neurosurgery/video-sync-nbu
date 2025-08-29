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
import re
import wave
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, List, Optional, Tuple

from scripts.parsers.videofileparser import VideoFileParser
from scripts.parsers.jsonfileparser import JsonParser

# Try optional MP3 probe (pydub)
try:
    from pydub import AudioSegment  # type: ignore

    _HAVE_PYDUB = True
except Exception:
    _HAVE_PYDUB = False

from scripts.models import (
    CamJson,
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

DEFAULT_TZ = ZoneInfo("America/Chicago")


# ---------------------------------------------------------------------------
# Filename Patterns & Utilities
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
        return (m.group("base"), m.group("cam")) if m else None

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
# Small probes & helpers
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
        return len(seg) / 1000.0, seg.frame_rate
    except Exception:
        return 0.0, 0


def _format_channels(channels: Iterable[int]) -> str:
    return ", ".join(f"{ch:02d}" for ch in channels)


def _safe_glob(directory: Path, patterns: Iterable[str]) -> List[Path]:
    """Glob multiple patterns and return a single sorted list."""
    results: List[Path] = []
    for pat in patterns:
        results.extend(directory.glob(pat))
    return sorted({p for p in results if p.is_file()})


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

    # ---- small helpers -----------------------------------------------------

    def _collect_candidates(self) -> List[Path]:
        self._ensure_exists(self.audio_dir)
        return _safe_glob(self.audio_dir, ("*.wav", "*.mp3"))

    def _choose_best_per_channel(self, candidates: List[Path]) -> Dict[int, Path]:
        """Pick one file per channel, preferring WAV over MP3 when both exist."""
        chosen: Dict[int, Path] = {}
        for p in candidates:
            parsed = FilePatterns.parse_audio_filename(p)
            if not parsed:
                self.log.warning("Skipping audio with unexpected name: %s", p.name)
                continue
            ch, ext = parsed
            existing = chosen.get(ch)
            if existing is None:
                chosen[ch] = p
            else:
                if existing.suffix.lower() == ".mp3" and ext == "wav":
                    chosen[ch] = p
        return chosen

    def _warn_if_channel_count_unexpected(self, chosen: Dict[int, Path]) -> None:
        if len(chosen) != 3:
            ch_list = _format_channels(chosen.keys()) or "none"
            self.log.warning(
                "Expected 3 audio files, found %d (channels: %s).",
                len(chosen),
                ch_list,
            )

    def _infer_shared_extension(self, paths: Iterable[Path]) -> Optional[str]:
        exts = {p.suffix.lower().lstrip(".") for p in paths}
        if len(exts) == 1 and exts <= {"wav", "mp3"}:
            return next(iter(exts))
        if exts:
            self.log.warning(
                "Mixed audio extensions detected across channels: %s.",
                ", ".join(sorted(exts)),
            )
        return None

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
        a = self._build_audio_obj(ch, p)
        return SerialAudio(
            path=a.path,
            duration=a.duration,
            file_size=a.file_size,
            sample_rate=a.sample_rate,
            extension=a.extension,
            channel=a.channel,
        )

    def _build_audio_map(
        self, chosen: Dict[int, Path]
    ) -> Tuple[Dict[int, Audio], Optional[SerialAudio]]:
        audios: Dict[int, Audio] = {}
        serial_audio: Optional[SerialAudio] = None
        for ch, p in sorted(chosen.items(), key=lambda kv: kv[0]):
            if ch == self.default_serial_channel:
                serial_audio = self._build_serial_audio_obj(ch, p)
                audios[ch] = serial_audio
            else:
                audios[ch] = self._build_audio_obj(ch, p)
        if serial_audio is None:
            self.log.warning(
                "Default serial channel %02d not found in AUDIO_DIR.",
                self.default_serial_channel,
            )
        return audios, serial_audio

    # ---- public ------------------------------------------------------------

    def discover(self) -> AudioGroup:
        candidates = self._collect_candidates()
        chosen_by_channel = self._choose_best_per_channel(candidates)
        self._warn_if_channel_count_unexpected(chosen_by_channel)

        shared_ext = self._infer_shared_extension(chosen_by_channel.values())
        audios, serial_audio = self._build_audio_map(chosen_by_channel)

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

    # ---- scanning helpers --------------------------------------------------

    def _index_jsons(self) -> Dict[str, Path]:
        self._ensure_exists(self.video_dir)
        json_by_seg: Dict[str, Path] = {}
        for jp in _safe_glob(self.video_dir, ("*.json",)):
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
        return json_by_seg

    def _index_mp4s(self) -> Dict[str, Dict[str, Path]]:
        vids_by_seg: Dict[str, Dict[str, Path]] = {}
        for vp in _safe_glob(self.video_dir, ("*.mp4",)):
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
        return vids_by_seg

    # ---- metadata helpers --------------------------------------------------

    def _extract_video_meta(self, mp4_path: Path) -> tuple[float, str, float, int]:
        """Return (duration_sec, 'WxH', fps, frame_count). On failure, return zeros/empty."""
        try:
            vp = VideoFileParser(str(mp4_path))
            w, h = vp.resolution
            return vp.duration, f"{w}x{h}", vp.fps, vp.frame_count
        except Exception as e:
            self.log.warning(
                "ffprobe failed for %s: %s; leaving meta blank.", mp4_path.name, e
            )
            return 0.0, "", 0.0, 0

    def _build_videos_for_seg(
        self, cams: Dict[str, Path], ts: Optional[datetime]
    ) -> List[Video]:
        videos: List[Video] = []
        for cam_serial, mp4_path in sorted(cams.items(), key=lambda kv: kv[0]):
            dur, res, fps, frame_count = self._extract_video_meta(mp4_path)
            videos.append(
                Video(
                    path=mp4_path,
                    cam_serial=str(cam_serial),
                    timestamp=ts,
                    duration=dur,
                    resolution=res,
                    frame_rate=fps,
                    frame_count=frame_count,
                )
            )
        return videos

    def _extract_cam_jsons(
        self, json_path: Path, ts: Optional[datetime]
    ) -> tuple[list[str], dict[str, CamJson]]:
        """
        Use JsonParser to populate per-camera CamJson objects.

        Returns: (cam_serials_as_strings_in_json, cam_jsons_map_by_serial)
        """
        cam_jsons: dict[str, CamJson] = {}
        try:
            jp = JsonParser(str(json_path))
            parser_serials = jp.get_camera_serials()  # e.g., [24253445, ...]
            cam_serials_all = [str(s) for s in parser_serials]

            for s in parser_serials:
                s_str = str(s)
                try:
                    raw_serials = jp.get_chunk_serial_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_chunk_serial_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    raw_serials = None

                try:
                    raw_frame_ids = jp.get_frame_ids_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_frame_ids_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    raw_frame_ids = None

                try:
                    fixed_serials = jp.get_fixed_chunk_serial_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_fixed_chunk_serial_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    fixed_serials = None

                try:
                    fixed_frame_ids = jp.get_fixed_frame_ids_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_fixed_frame_ids_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    fixed_frame_ids = None

                try:
                    fixed_reidx_frame_ids = (
                        [f - fixed_frame_ids[0] for f in fixed_frame_ids]
                        if fixed_frame_ids
                        else None
                    )
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_fixed_reindexed_frame_ids_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    fixed_reidx_frame_ids = None

                cam_jsons[s_str] = CamJson(
                    cam_serial=s_str,
                    timestamp=ts,
                    path=json_path,
                    raw_serials=raw_serials,
                    raw_frame_ids=raw_frame_ids,
                    fixed_serials=fixed_serials,
                    fixed_frame_ids=fixed_frame_ids,
                    fixed_reidx_frame_ids=fixed_reidx_frame_ids,
                )

            return cam_serials_all, cam_jsons

        except Exception as e:
            self.log.warning("Failed to parse JSON %s: %s", json_path.name, e)
            return [], {}

    def _build_json_wrapper(
        self, json_path: Path, ts: Optional[datetime]
    ) -> Tuple[Json, List[str], Dict[str, CamJson]]:
        cam_serials_from_json, cam_jsons = self._extract_cam_jsons(json_path, ts)
        json_wrap = Json(
            cam_serials=cam_serials_from_json or None,
            timestamp=ts,
            path=json_path,
            cam_jsons=cam_jsons,
        )
        return json_wrap, cam_serials_from_json, cam_jsons

    # ---- public: fast single-segment path (no cam filter) -----------------

    def discover_one(self, segment_id: str) -> Optional[VideoGroup]:
        """
        Build a single VideoGroup for a given segment_id, including all cameras.
        Avoids a full directory scan.
        """
        self._ensure_exists(self.video_dir)

        # JSON for this segment
        json_matches = list(self.video_dir.glob(f"{segment_id}.json"))
        if not json_matches:
            self.log.warning("No JSON found for segment %s", segment_id)
            return None
        json_path = json_matches[0]

        ts = FilePatterns.parse_tail_datetime(segment_id, DEFAULT_TZ)

        # All MP4s for this segment
        cams: Dict[str, Path] = {}
        for vp in self.video_dir.glob(f"{segment_id}.*.mp4"):
            parsed = FilePatterns.parse_video_filename(vp)
            if parsed:
                _, cam = parsed
                if cam not in cams:  # keep first if duplicates
                    cams[cam] = vp

        videos = self._build_videos_for_seg(cams, ts)

        json_wrap, _, _ = self._build_json_wrapper(json_path, ts)

        return VideoGroup(
            group_id=segment_id,
            timestamp=ts,
            json=json_wrap,
            videos=videos or None,
            cam_serials=(sorted({v.cam_serial for v in videos}) if videos else None),
        )

    # ---- public: full directory path --------------------------------------

    def discover(self) -> List[VideoGroup]:
        json_by_seg = self._index_jsons()
        vids_by_seg = self._index_mp4s()

        videogroups: List[VideoGroup] = []
        for seg_id, json_path in json_by_seg.items():
            ts = FilePatterns.parse_tail_datetime(seg_id, DEFAULT_TZ)
            cams = vids_by_seg.get(seg_id, {})

            videos = self._build_videos_for_seg(cams, ts)
            if not videos:
                self.log.warning(
                    "No MP4s found for segment %s (JSON: %s)", seg_id, json_path.name
                )

            json_wrap, _, _ = self._build_json_wrapper(json_path, ts)

            videogroups.append(
                VideoGroup(
                    group_id=seg_id,
                    timestamp=ts,
                    json=json_wrap,
                    videos=videos or None,
                    cam_serials=(
                        sorted({v.cam_serial for v in videos}) if videos else None
                    ),
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
        audiogroup = self.audio.discover()
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
def discover_audio(audio_dir: Path, default_serial_channel: int = 3) -> AudioGroup:
    return AudioDiscoverer(audio_dir, default_serial_channel).discover()


def discover_segments(video_dir: Path) -> List[VideoGroup]:
    return VideoDiscoverer(video_dir).discover()


def discover_segment(video_dir: Path, segment_id: str) -> Optional[VideoGroup]:
    return VideoDiscoverer(video_dir).discover_one(segment_id)


def discover(
    audio_dir: Path, video_dir: Path, default_serial_channel: int = 3
) -> AudioVideoSession:
    return Discoverer(
        audio_dir, video_dir, default_serial_channel=default_serial_channel
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

    args = ap.parse_args(argv)

    # Fast path: just one VideoGroup (no audio scan, all cameras included)
    if args.segment_id:
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

    session = discover(
        args.audio_dir, args.video_dir, default_serial_channel=args.serial_channel
    )

    _print_audiogroup(session.audiogroup)
    _print_videogroups(session.videogroups)
    _print_shared(session.shared_cam_serials)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
