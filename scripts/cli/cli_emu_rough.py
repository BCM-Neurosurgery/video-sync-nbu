"""
EMU Rough Sync CLI
==================

Self-contained command line utility that performs "rough" alignment of EMU
recordings when chunk serials are unavailable. The workflow:

1. Discover matching EMU tasks (patient directory + NS5 audio).
2. Export the full-room NS5 audio for each task after calibrating the NS5 UTC
   origin via the first NEV chunk.
3. Locate camera segments whose realtime metadata overlaps the NS5 window and
   build clip plans using fixed chunk/frame metadata from the JSON companions.
4. Clip, pad, merge, and mux the candidate videos with the extracted audio to
   produce per-camera MP4s ready for review.

The implementation used to live across ``cli_emu.py`` and
``cli_emu_rough.py``. This module consolidates the rough-sync functionality so
it stands alone and does not depend on the serial-sync CLI.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from scipy.io.wavfile import write as wav_write

from scripts.analysis.video_analysis import FrameIDAnalysisResult, analyze_video
from scripts.index.filepatterns import FilePatterns
from scripts.models import CamJson, RoomAudio, StitchedNS5, Video
from scripts.pad.videoplanapplier import apply_video_padding_plan
from scripts.parsers.jsonfileparser import JsonParser
from scripts.parsers.nevfileparser import Nev
from scripts.parsers.ns5fileparser import Nsx
from scripts.parsers.videofileparser import VideoFileParser

LOGGER = logging.getLogger("cli_emu_rough")

_REALTIME_STRICT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
DATE_DIR_RE = re.compile(r"\d{8}")


@dataclass(frozen=True)
class VideoSegmentClipPlan:
    """Represents the portion of a camera segment we need to keep."""

    video: Video
    serials: List[int]
    frame_ids: List[int]
    frame_ids_local: List[int]
    clip_start_index: int
    clip_end_index: int


@dataclass
class TaskContext:
    """Bundle task identity, NS5 media, and calibrated UTC bounds."""

    patient_id: str
    task_id: str
    ns5: StitchedNS5
    nsx_parser: Nsx
    stitched_nev_path: Path
    first_nev_path: Path
    ns5_start_utc: datetime
    ns5_end_utc: datetime


@dataclass
class SyncPlan:
    """Inputs required to run the rough-sync pipeline for one task."""

    mode: str
    audio_path: Path
    clip_plans_by_cam: Dict[str, List[VideoSegmentClipPlan]]
    audio_start: datetime
    audio_end: datetime
    audio_ready: bool = True


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _ensure_tool(name: str) -> None:
    """Ensure an external CLI dependency is available."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found on PATH.")


def _parse_realtime_string(value: str) -> Optional[datetime]:
    """Parse a realtime string into a datetime if possible."""
    try:
        return datetime.strptime(value, _REALTIME_STRICT_FORMAT)
    except Exception:
        try:
            return datetime.fromisoformat(value)
        except Exception:
            LOGGER.debug("Unable to parse realtime value '%s'", value)
            return None


def _coerce_real_times(values: Optional[Sequence[object]]) -> Optional[List[datetime]]:
    """Convert a sequence of mixed values into datetimes or return None on failure."""
    if not values:
        return None
    parsed: List[datetime] = []
    for item in values:
        if isinstance(item, datetime):
            parsed.append(item)
        elif isinstance(item, str):
            parsed_dt = _parse_realtime_string(item)
            if parsed_dt is None:
                return None
            parsed.append(parsed_dt)
        else:
            LOGGER.debug("Unexpected realtime value type: %s", type(item).__name__)
            return None
    return parsed


def _find_nsp1_file(task_dir: Path, ext: str) -> Optional[Path]:
    """Return the first NSP-1 file with the requested extension in a task."""
    pattern = f"*NSP-1.{ext}"
    matches = sorted(task_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        LOGGER.warning(
            "Multiple NSP-1 %s files in %s; using %s",
            ext,
            task_dir.name,
            matches[0].name,
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Task discovery and parser loading
# ---------------------------------------------------------------------------


def load_ns5(
    path: Path,
    *,
    reference_time_origin: datetime,
    reference_start_timestamp: int,
) -> tuple[StitchedNS5, Nsx, datetime, datetime]:
    """Load NS5 metadata using the first chunk NEV as the UTC reference."""
    LOGGER.debug("Loading NS5 file %s", path)
    parser = Nsx(str(path))
    sample_rate = float(parser.get_sample_resolution())
    if sample_rate <= 0:
        raise RuntimeError("NS5 sample resolution must be positive")
    start_ts = int(parser.get_start_timestamp())
    offset_seconds = (start_ts - reference_start_timestamp) / float(sample_rate)
    start_utc = reference_time_origin + timedelta(seconds=offset_seconds)
    recording_duration = max(0.0, float(parser.get_recording_duration_s()))
    end_utc = start_utc + timedelta(seconds=recording_duration)
    LOGGER.debug(
        "NS5 %s calibrated with start UTC %s and end UTC %s (duration %.2fs)",
        path.name,
        start_utc.isoformat(),
        end_utc.isoformat(),
        recording_duration,
    )

    def _load_channel(channel_name: str) -> RoomAudio:
        try:
            channel_array = parser.get_channel_array(channel_name)
            arr = np.asarray(channel_array)
            num_samples = int(arr.shape[0])
            channel_end_ts = start_ts + max(0, num_samples - 1)
            duration = num_samples / sample_rate if sample_rate else None
            return RoomAudio(
                raw_array=arr,
                start_timestamp=start_ts,
                end_timestamp=channel_end_ts,
                duration=duration,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Failed reading %s from %s: %s", channel_name, path.name, exc
            )
            return RoomAudio(
                raw_array=None,
                start_timestamp=None,
                end_timestamp=None,
                duration=None,
            )

    ns5_model = StitchedNS5(
        path=path,
        start_utc_time=start_utc,
        sample_resolution=sample_rate,
        duration=recording_duration,
        room_mic1=_load_channel("RoomMic1"),
        room_mic2=_load_channel("RoomMic2"),
    )
    return ns5_model, parser, start_utc, end_utc


def _resolve_first_chunk_nev(stitched_nev: Path) -> Path:
    """Return the path to the first chunk NEV for a stitched NEV."""
    script = (
        Path(__file__).resolve().parent.parent
        / "dj"
        / "find_first_nev_from_stitched.py"
    )
    if not script.exists():
        raise FileNotFoundError(
            f"find_first_nev_from_stitched.py not found at {script}"
        )
    proc = subprocess.run(
        [sys.executable, str(script), str(stitched_nev)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or "unknown error"
        raise RuntimeError(f"find_first_nev_from_stitched failed: {message}")
    output = proc.stdout.strip()
    if not output:
        raise RuntimeError("find_first_nev_from_stitched returned no path")
    return Path(output)


def discover_task_contexts(
    patient_dir: Path,
    keywords: Optional[Sequence[str]],
) -> List[TaskContext]:
    """Discover NS5 tasks and calibrate their UTC range via the first NEV chunk."""
    contexts: List[TaskContext] = []
    for task_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        task_id = task_dir.name
        if keywords and not any(kw.lower() in task_id.lower() for kw in keywords):
            LOGGER.debug("Skipping task %s (keyword filter)", task_id)
            continue
        LOGGER.debug("Inspecting task directory %s", task_dir)

        nev_path = _find_nsp1_file(task_dir, "nev")
        ns5_path = _find_nsp1_file(task_dir, "ns5")
        if not nev_path or not ns5_path:
            LOGGER.warning(
                "Skipping %s: missing stitched NEV (%s) or NS5 (%s)",
                task_id,
                bool(nev_path),
                bool(ns5_path),
            )
            continue

        try:
            first_nev_path = _resolve_first_chunk_nev(nev_path)
            first_nev_parser = Nev(str(first_nev_path))
            reference_origin = first_nev_parser.get_time_origin()
            reference_start_ts = int(first_nev_parser.get_start_timestamp())
            LOGGER.debug(
                "Resolved first chunk NEV for %s: %s (origin=%s start_ts=%d)",
                task_id,
                first_nev_path,
                reference_origin.isoformat(),
                reference_start_ts,
            )
        except Exception as exc:
            LOGGER.error("Failed to resolve first NEV for %s: %s", task_id, exc)
            continue

        try:
            ns5_model, nsx_parser, ns5_start_utc, ns5_end_utc = load_ns5(
                ns5_path,
                reference_time_origin=reference_origin,
                reference_start_timestamp=reference_start_ts,
            )
        except Exception as exc:
            LOGGER.error("Failed loading NS5 for %s: %s", task_id, exc)
            continue

        contexts.append(
            TaskContext(
                patient_id=patient_dir.name,
                task_id=task_id,
                ns5=ns5_model,
                nsx_parser=nsx_parser,
                stitched_nev_path=nev_path,
                first_nev_path=first_nev_path,
                ns5_start_utc=ns5_start_utc,
                ns5_end_utc=ns5_end_utc,
            )
        )
        LOGGER.debug(
            "Task %s ready (NS5 window %s -> %s)",
            task_id,
            ns5_start_utc.isoformat(),
            ns5_end_utc.isoformat(),
        )

    LOGGER.info("Built %d stitched task context(s) from %s", len(contexts), patient_dir)
    return contexts


# ---------------------------------------------------------------------------
# Video directory helpers
# ---------------------------------------------------------------------------


def _iter_date_dirs(video_dir: Path) -> List[Path]:
    """Return sorted YYYYMMDD subdirectories that contain MP4/JSON assets."""

    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    subdirs = [p for p in sorted(video_dir.iterdir()) if p.is_dir()]
    if not subdirs:
        raise RuntimeError(
            "Video directory must contain date subdirectories (YYYYMMDD) with MP4/JSON files."
        )

    valid_dirs: List[Path] = []
    for sub in subdirs:
        if not DATE_DIR_RE.fullmatch(sub.name or ""):
            raise RuntimeError(
                f"Unexpected subdirectory '{sub.name}' in video dir; expected YYYYMMDD folders."
            )
        jsons = list(sub.glob("*.json"))
        mp4s = list(sub.glob("*.mp4"))
        if not jsons or not mp4s:
            raise RuntimeError(
                f"Date folder '{sub.name}' must contain both MP4 and JSON files."
            )
        valid_dirs.append(sub)

    stray_files = [p for p in video_dir.iterdir() if p.is_file()]
    if stray_files:
        raise RuntimeError(
            "Video directory should not contain files directly; only date subdirectories."
        )

    return valid_dirs


def _iter_segment_entries(video_dir: Path):
    """Yield segment IDs with their JSON companions and per-camera MP4s."""
    for date_dir in _iter_date_dirs(video_dir):
        for json_path in sorted(date_dir.glob("*.json")):
            seg_id = json_path.stem
            mp4s: Dict[str, Path] = {}
            for mp4_path in date_dir.glob(f"{seg_id}.*.mp4"):
                parsed = FilePatterns.parse_video_filename(mp4_path)
                if parsed:
                    _, cam = parsed
                    mp4s[str(cam)] = mp4_path
            yield seg_id, date_dir, json_path, mp4s


def _as_int_list(seq: Optional[Iterable[object]]) -> Optional[List[int]]:
    """Attempt to convert an iterable to a list of integers."""
    if seq is None:
        return None
    result: List[int] = []
    for item in seq:
        try:
            result.append(int(item))
        except Exception:
            return None
    return result


def _build_cam_json_from_parser(
    jp: JsonParser,
    json_path: Path,
    ts: Optional[datetime],
    cam_value,
    cam_serial_str: str,
    *,
    real_times: Optional[List[datetime]] = None,
) -> CamJson:
    """Build a CamJson model from parser data and optional realtime overrides."""
    start_real = None
    try:
        start_real = jp.get_start_realtime()
    except Exception:
        start_real = None

    raw_serials = _as_int_list(jp.get_chunk_serial_list(cam_value))
    raw_frame_ids = _as_int_list(jp.get_frame_ids_list(cam_value))
    fixed_serials = _as_int_list(jp.get_fixed_chunk_serial_list(cam_value))
    fixed_frame_ids = _as_int_list(jp.get_fixed_frame_ids_list(cam_value))
    if fixed_frame_ids:
        fixed_reidx = [fid - fixed_frame_ids[0] for fid in fixed_frame_ids]
    else:
        fixed_reidx = None

    return CamJson(
        cam_serial=cam_serial_str,
        timestamp=ts,
        path=json_path,
        start_realtime=start_real,
        real_times=list(real_times) if real_times else None,
        raw_serials=raw_serials,
        raw_frame_ids=raw_frame_ids,
        fixed_serials=fixed_serials,
        fixed_frame_ids=fixed_frame_ids,
        fixed_reidx_frame_ids=_as_int_list(fixed_reidx) if fixed_reidx else None,
    )


def _build_video_from_json(
    seg_id: str,
    cam_serial_str: str,
    cam_value,
    jp: JsonParser,
    json_path: Path,
    ts: Optional[datetime],
    mp4_path: Path,
) -> Video:
    """Construct a Video model enriched with its companion JSON metadata."""
    parsed_real_times = getattr(jp, "_parsed_real_times", None)
    if parsed_real_times is None:
        parsed_real_times = _coerce_real_times(jp.dic.get("real_times"))
        setattr(jp, "_parsed_real_times", parsed_real_times)

    cam_json = _build_cam_json_from_parser(
        jp,
        json_path,
        ts,
        cam_value,
        cam_serial_str,
        real_times=parsed_real_times,
    )
    try:
        probe = VideoFileParser(str(mp4_path))
        duration = probe.duration
        res = f"{probe.resolution[0]}x{probe.resolution[1]}"
        fps = probe.fps
        frame_count = probe.frame_count
    except Exception as exc:
        raise RuntimeError(f"ffprobe failed for {mp4_path}: {exc}")

    start_rt = cam_json.start_realtime or ts
    return Video(
        path=mp4_path,
        segment_id=seg_id,
        cam_serial=cam_serial_str,
        timestamp=ts,
        start_realtime=start_rt,
        duration=duration,
        resolution=res,
        frame_rate=fps,
        frame_count=frame_count,
        companion_json=cam_json,
    )


# ---------------------------------------------------------------------------
# Clip plan helpers
# ---------------------------------------------------------------------------


def _choose_serials(cam_json: CamJson) -> List[int]:
    """Return the fixed chunk serials for a camera JSON."""
    if not cam_json.fixed_serials:
        raise ValueError("CamJson missing fixed_serials")
    return [int(s) for s in cam_json.fixed_serials]


def _choose_frame_ids(cam_json: CamJson) -> List[int]:
    """Return the normalized frame IDs for a camera JSON."""
    if not cam_json.fixed_reidx_frame_ids:
        raise ValueError("CamJson missing fixed_reidx_frame_ids")
    return [int(f) for f in cam_json.fixed_reidx_frame_ids]


def collect_videos_by_time(
    video_dir: Path,
    window_start: datetime,
    window_end: datetime,
    camera_filter: Optional[Set[str]] = None,
) -> Dict[str, List[Video]]:
    """Group videos by camera whose realtime span overlaps the target window."""

    allowed: Optional[Set[str]] = (
        {c.strip() for c in camera_filter} if camera_filter else None
    )
    matches: Dict[str, List[Video]] = defaultdict(list)
    LOGGER.debug(
        "Scanning videos between %s and %s (filter=%s)",
        window_start.isoformat(),
        window_end.isoformat(),
        sorted(allowed) if allowed else "ALL",
    )

    def classify_segment(
        seg_id: str,
        start_rt: Optional[datetime],
        end_rt: Optional[datetime],
    ) -> str:
        if start_rt is None or end_rt is None:
            LOGGER.debug(
                "Segment %s lacks realtime bounds in JSON; skipping rough lookup.",
                seg_id,
            )
            return "skip"
        if end_rt < window_start:
            LOGGER.debug("Segment %s ends before audio window; skipping.", seg_id)
            return "before"
        if start_rt > window_end:
            LOGGER.debug("Segment %s starts after audio window; stopping scan.", seg_id)
            return "after"
        return "include"

    for seg_id, _date_dir, json_path, mp4s in _iter_segment_entries(video_dir):
        ts = FilePatterns.parse_tail_datetime(seg_id)
        try:
            jp = JsonParser(str(json_path))
        except Exception as exc:
            LOGGER.error("Failed to parse JSON %s: %s", json_path, exc)
            continue

        segment_start_rt = jp.get_start_realtime()
        segment_end_rt = jp.get_end_realtime()

        decision = classify_segment(seg_id, segment_start_rt, segment_end_rt)
        if decision in {"skip", "before"}:
            continue
        if decision == "after":
            break

        serials_from_json = jp.get_camera_serials()
        serial_map = {str(s): s for s in serials_from_json}
        target_serials = (
            sorted(set(serial_map.keys()) & allowed)
            if allowed
            else sorted(serial_map.keys())
        )
        if not target_serials:
            continue

        for cam_serial in target_serials:
            if cam_serial not in mp4s:
                LOGGER.debug(
                    "Skipping segment %s cam %s (MP4 missing)", seg_id, cam_serial
                )
                continue
            try:
                video = _build_video_from_json(
                    seg_id,
                    cam_serial,
                    serial_map[cam_serial],
                    jp,
                    json_path,
                    ts,
                    mp4s[cam_serial],
                )
            except Exception as exc:
                LOGGER.error(
                    "Failed building video object for %s cam %s: %s",
                    seg_id,
                    cam_serial,
                    exc,
                )
                continue

            cam_json = video.companion_json
            start_rt = cam_json.start_realtime if cam_json else None
            start_rt = start_rt or segment_start_rt
            end_rt = segment_end_rt
            if start_rt is None or end_rt is None:
                LOGGER.debug(
                    "Skipping segment %s cam %s due to missing realtime bounds.",
                    seg_id,
                    cam_serial,
                )
                continue
            if end_rt < window_start or start_rt > window_end:
                continue
            matches[cam_serial].append(video)
            LOGGER.debug(
                "Queued segment %s cam %s (start=%s end=%s)",
                seg_id,
                cam_serial,
                start_rt.isoformat(),
                end_rt.isoformat(),
            )

    for cam_serial, videos in matches.items():
        videos.sort(
            key=lambda v: (
                v.start_realtime or v.timestamp or window_start,
                v.path.name,
            )
        )

    LOGGER.info(
        "Found videos for %d camera(s) overlapping %s-%s",
        len(matches),
        window_start,
        window_end,
    )
    return matches


def build_time_clip_plan(
    video: Video, audio_start: datetime, audio_end: datetime
) -> Optional[VideoSegmentClipPlan]:
    """Derive the frame slice that overlaps an audio window for one video."""
    cam_json = video.companion_json
    if cam_json is None:
        return None

    serials_full = _choose_serials(cam_json)
    frames_full = _choose_frame_ids(cam_json)
    if not serials_full or not frames_full or len(serials_full) != len(frames_full):
        return None

    real_times_seq = _coerce_real_times(getattr(cam_json, "real_times", None))
    if not real_times_seq:
        raise ValueError(
            f"Video {video.path} missing realtime data required for rough sync"
        )

    if len(serials_full) != len(real_times_seq) or len(frames_full) != len(
        real_times_seq
    ):
        raise ValueError(
            (
                "Rough sync expects matching lengths for serials, frame IDs, and realtime "
                "stamps"
            )
        )

    total_frames = len(real_times_seq)
    if total_frames == 0:
        return None

    video_start_time = real_times_seq[0]
    video_end_time = real_times_seq[-1]
    if audio_end <= video_start_time or audio_start >= video_end_time:
        return None

    start_idx = next(
        (i for i, ts in enumerate(real_times_seq) if ts >= audio_start),
        0,
    )
    end_idx = next(
        (i for i in range(total_frames - 1, -1, -1) if real_times_seq[i] <= audio_end),
        total_frames - 1,
    )
    if end_idx < start_idx:
        end_idx = start_idx

    serial_slice = serials_full[start_idx : end_idx + 1]
    frame_slice = frames_full[start_idx : end_idx + 1]
    if not serial_slice or not frame_slice:
        return None

    min_frame = frame_slice[0]
    local_frames = [f - min_frame for f in frame_slice]

    return VideoSegmentClipPlan(
        video=video,
        serials=[int(s) for s in serial_slice],
        frame_ids=[int(f) for f in frame_slice],
        frame_ids_local=local_frames,
        clip_start_index=start_idx,
        clip_end_index=end_idx,
    )


def _build_time_clip_plans_for_camera(
    task_id: str,
    cam_serial: str,
    videos: Sequence[Video],
    audio_start: datetime,
    audio_end: datetime,
) -> List[VideoSegmentClipPlan]:
    """Compile clip plans for a camera limited to the audio overlap window."""
    cam_plans: List[VideoSegmentClipPlan] = []
    for video in videos:
        try:
            plan = build_time_clip_plan(video, audio_start, audio_end)
        except Exception as exc:
            LOGGER.error(
                "Failed building clip plan for %s cam %s (rough mode): %s",
                video.path.name,
                cam_serial,
                exc,
            )
            continue
        if plan:
            cam_plans.append(plan)
    if not cam_plans:
        LOGGER.error(
            "No usable video segments for %s camera %s (rough mode)",
            task_id,
            cam_serial,
        )
    else:
        cam_plans.sort(key=lambda p: (p.video.segment_id, p.clip_start_index))
    return cam_plans


def _warn_missing_cameras(
    task_id: str,
    mode_label: str,
    requested: Optional[Set[str]],
    available: Set[str],
) -> None:
    """Warn if requested cameras are absent from the available set."""
    if not requested:
        return
    for cam_serial in sorted({cam for cam in requested if cam not in available}):
        LOGGER.warning(
            "Requested camera %s not found for %s (%s mode)",
            cam_serial,
            task_id,
            mode_label,
        )


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_full_ns5_audio(
    task: TaskContext,
    room_mic: str,
    out_path: Path,
    overwrite: bool,
) -> tuple[Path, datetime, datetime]:
    """Export the NS5 room-mic audio and metadata for the task."""
    channel_lookup = {"roommic1": "RoomMic1", "roommic2": "RoomMic2"}
    channel_name = channel_lookup[room_mic.lower()]
    audio = task.ns5.room_mic1 if channel_name == "RoomMic1" else task.ns5.room_mic2
    if audio.raw_array is None:
        raise RuntimeError(f"Room audio {channel_name} unavailable in NS5")
    LOGGER.debug(
        "Extracting NS5 audio for task %s mic %s -> %s",
        task.task_id,
        channel_name,
        out_path,
    )

    sample_rate = float(task.ns5.sample_resolution)
    if sample_rate <= 0:
        raise RuntimeError("Invalid NS5 sample rate")

    nsx = task.nsx_parser
    start_ts = (
        audio.start_timestamp
        if audio.start_timestamp is not None
        else nsx.get_start_timestamp()
    )
    arr = np.asarray(audio.raw_array, dtype=np.int16)
    if arr.size <= 0:
        raise RuntimeError("NS5 audio channel is empty")
    end_ts = (
        audio.end_timestamp
        if audio.end_timestamp is not None
        else start_ts + max(0, arr.size - 1)
    )

    ns5_start_ts = int(nsx.get_start_timestamp())
    start_offset = (start_ts - ns5_start_ts) / float(sample_rate)
    end_offset = (end_ts - ns5_start_ts) / float(sample_rate)
    start_dt = task.ns5_start_utc + timedelta(seconds=start_offset)
    end_dt = task.ns5_start_utc + timedelta(seconds=end_offset)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or overwrite:
        wav_write(str(out_path), int(sample_rate), arr)

    duration_seconds = float(arr.size) / float(sample_rate)
    metadata = {
        "patient_id": task.patient_id,
        "task_id": task.task_id,
        "room_mic": room_mic,
        "channel_name": channel_name,
        "sample_rate_hz": int(sample_rate),
        "sample_count": int(arr.size),
        "duration_seconds": duration_seconds,
        "start_timestamp_ticks": int(start_ts),
        "end_timestamp_ticks": int(end_ts),
        "start_time_utc": start_dt.isoformat(),
        "end_time_utc": end_dt.isoformat(),
        "audio_path": str(out_path),
        "ns5_path": str(task.ns5.path),
    }
    metadata_path = out_path.with_suffix(".json")
    if not metadata_path.exists() or overwrite:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.debug(
        "Audio export complete (%s) start=%s end=%s samples=%d",
        out_path,
        start_dt.isoformat(),
        end_dt.isoformat(),
        arr.size,
    )

    return out_path, start_dt, end_dt


# ---------------------------------------------------------------------------
# Video preparation
# ---------------------------------------------------------------------------


def build_padding_operations(plan: VideoSegmentClipPlan) -> List[dict]:
    """Describe how many frames to insert to bridge serial gaps."""
    ops: List[dict] = []
    serials = plan.serials
    frames_local = plan.frame_ids_local
    frames_abs = plan.frame_ids
    for idx in range(len(serials) - 1):
        curr = serials[idx]
        nxt = serials[idx + 1]
        if curr < 0 or nxt < 0:
            continue
        gap = nxt - curr
        if gap <= 1:
            continue
        ops.append(
            {
                "after_index": int(frames_local[idx]),
                "insert": int(gap - 1),
                "frame_id_before": int(frames_abs[idx]),
                "frame_id_after": int(frames_abs[idx + 1]),
            }
        )
    return ops


def pad_video_if_needed(
    plan: VideoSegmentClipPlan,
    clip_path: Path,
    work_dir: Path,
) -> Path:
    """Apply padding operations to a clip when frame gaps exist."""
    ops = build_padding_operations(plan)
    if not ops:
        return clip_path
    LOGGER.debug("Padding clip %s with %d operations", clip_path.name, len(ops))

    padded_dir = work_dir / "padded"
    padded_dir.mkdir(parents=True, exist_ok=True)

    plan_payload = {
        "version": 1,
        "segment_id": plan.video.segment_id,
        "cam_serial": plan.video.cam_serial,
        "source_video": str(clip_path),
        "source_fps": float(plan.video.frame_rate),
        "target_fps": float(plan.video.frame_rate),
        "policy": "dup-prev",
        "total_insertions": int(sum(op["insert"] for op in ops)),
        "operations": ops,
    }
    plan_json = padded_dir / f"{clip_path.stem}-videopad.json"
    plan_json.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    probe = VideoFileParser(str(clip_path))
    cam_json = CamJson(
        cam_serial=plan.video.cam_serial,
        timestamp=plan.video.timestamp,
        path=plan.video.companion_json.path,
        start_realtime=plan.video.companion_json.start_realtime,
        raw_serials=plan.serials,
        raw_frame_ids=plan.frame_ids,
        fixed_serials=plan.serials,
        fixed_frame_ids=plan.frame_ids,
        fixed_reidx_frame_ids=plan.frame_ids_local,
    )
    video_model = Video(
        path=clip_path,
        segment_id=plan.video.segment_id,
        cam_serial=plan.video.cam_serial,
        timestamp=plan.video.timestamp,
        start_realtime=plan.video.start_realtime,
        duration=float(probe.duration),
        resolution=plan.video.resolution,
        frame_rate=float(probe.fps),
        frame_count=int(probe.frame_count),
        companion_json=cam_json,
    )

    padded_path, _ = apply_video_padding_plan(plan_json, video_model, padded_dir)
    return padded_path


def clip_video(plan: VideoSegmentClipPlan, out_dir: Path, overwrite: bool) -> Path:
    """Trim the source video to the frames described by the clip plan."""
    _ensure_tool("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    video = plan.video
    LOGGER.debug(
        "Clipping %s from frame %d to %d",
        video.path.name,
        plan.frame_ids[0],
        plan.frame_ids[-1],
    )
    real_times: Optional[Sequence[datetime]] = None
    if video.companion_json and getattr(video.companion_json, "real_times", None):
        existing = video.companion_json.real_times
        if existing and isinstance(existing[0], datetime):
            real_times = existing
        else:
            real_times = _coerce_real_times(existing)

    fps = float(video.frame_rate or 0.0)
    if fps <= 0:
        raise ValueError(
            f"Video {video.path} missing frame rate and realtime metadata for clipping"
        )

    start_frame = int(plan.frame_ids[0])
    end_frame = int(plan.frame_ids[-1])
    if end_frame < start_frame:
        raise ValueError(
            f"Invalid clip frame window for {video.path.name}: {start_frame}>{end_frame}"
        )

    end_frame_exclusive = end_frame + 1
    vf_filter = (
        f"trim=start_frame={start_frame}:end_frame={end_frame_exclusive},"
        f"setpts=(N/{fps:.9f})/TB"
    )

    out_path = out_dir / f"{video.segment_id}.{video.cam_serial}.clip.mp4"
    ff_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video.path),
        "-vf",
        vf_filter,
        "-an",
        str(out_path),
    ]
    proc = subprocess.run(
        ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clip failed for {video.path.name}: {proc.stderr.strip()}"
        )
    return out_path


# ---------------------------------------------------------------------------
# Clip merging & muxing
# ---------------------------------------------------------------------------


def concat_videos(
    clip_paths: List[Path],
    work_dir: Path,
    overwrite: bool,
    *,
    target_fps: Optional[float] = None,
) -> Path:
    """Concatenate clip files into a single MP4, optionally re-encoding."""
    if not clip_paths:
        raise RuntimeError("No clips to merge")

    _ensure_tool("ffmpeg")
    work_dir.mkdir(parents=True, exist_ok=True)

    if len(clip_paths) == 1:
        source = clip_paths[0]
        if not target_fps:
            return source
        out_path = work_dir / f"{source.stem}.fps{target_fps:.3f}.mp4"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(source),
            "-vf",
            f"fps={target_fps}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-an",
            str(out_path),
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg fps convert failed: {result.stderr.strip()}")
        return out_path

    concat_file = work_dir / "concat.txt"

    def _quote(p: Path) -> str:
        return str(p).replace("'", "'\\''")

    concat_file.write_text(
        "\n".join(f"file '{_quote(path)}'" for path in clip_paths) + "\n",
        encoding="utf-8",
    )
    out_path = work_dir / "merged.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
    ]

    if target_fps:
        cmd.extend(
            [
                "-vf",
                f"fps={target_fps}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-an",
            ]
        )
    else:
        cmd.extend(["-c", "copy"])

    cmd.append(str(out_path))
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()}")
    return out_path


def mux_audio(
    video_path: Path, audio_path: Path, out_path: Path, overwrite: bool
) -> Path:
    """Combine merged video with audio, retiming video if durations differ."""
    _ensure_tool("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(audio_path), "rb") as wav_in:
        audio_frames = wav_in.getnframes()
        audio_rate = wav_in.getframerate()
    if audio_rate <= 0:
        raise RuntimeError("Invalid audio sample rate when muxing")
    audio_duration = audio_frames / float(audio_rate)
    if audio_duration <= 0:
        raise RuntimeError("Invalid audio duration when muxing")

    video_probe = VideoFileParser(str(video_path))
    frame_count = int(video_probe.frame_count)
    video_duration = float(video_probe.duration)
    if frame_count <= 0 or video_duration <= 0:
        raise RuntimeError("Invalid video probe data when muxing")

    original_fps = float(video_probe.fps)
    target_fps = frame_count / audio_duration
    if not math.isfinite(target_fps) or target_fps <= 0:
        target_fps = max(original_fps, 1.0)

    speed_scale = audio_duration / video_duration
    if not math.isfinite(speed_scale) or speed_scale <= 0:
        speed_scale = 1.0

    LOGGER.debug(
        "Muxing %s with audio %s (frame_count=%d, video_dur=%.6fs, audio_dur=%.6fs, target_fps=%.6f, orig_fps=%.6f, speed_scale=%.6f)",
        video_path.name,
        audio_path.name,
        frame_count,
        video_duration,
        audio_duration,
        target_fps,
        original_fps,
        speed_scale,
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]

    if not math.isclose(speed_scale, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        cmd.extend(
            [
                "-filter:v",
                f"setpts={speed_scale:.12f}*PTS",
            ]
        )

    cmd.extend(
        [
            "-vsync",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    progress_step = 0
    while True:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        line = line.strip()
        if line.startswith("frame="):
            try:
                current_frame = int(line.split("=", 1)[1])
            except ValueError:
                continue
            if frame_count > 0:
                percent = int(min(100, (current_frame * 100) // frame_count))
                if percent >= progress_step:
                    LOGGER.info(
                        "Muxing %s: %d%% (%d/%d frames)",
                        video_path.name,
                        percent,
                        current_frame,
                        frame_count,
                    )
                    progress_step = percent + 10
        elif line == "progress=end":
            LOGGER.info("Muxing %s: 100%% (%d frames)", video_path.name, frame_count)
            break

    _, stderr_output = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {stderr_output.strip()}")
    return out_path


# ---------------------------------------------------------------------------
# Execution pipeline
# ---------------------------------------------------------------------------


def _execute_sync_plan(
    sync_plan: SyncPlan,
    task: TaskContext,
    out_dir: Path,
    room_mic: str,
    overwrite: bool,
) -> bool:
    """Run clipping, padding, merging, and muxing for a prepared sync plan."""
    if not sync_plan.audio_ready:
        raise RuntimeError("Rough sync requires a prepared audio track")

    success = False
    for cam_serial, cam_plans in sorted(sync_plan.clip_plans_by_cam.items()):
        cam_dir = out_dir / cam_serial
        work_dir = cam_dir / "work"
        clips_dir = work_dir / "clips"
        clip_paths: List[Path] = []
        clip_metadata_entries: List[dict] = []
        analysis_dir = work_dir / "analysis"
        video_analysis_cache: Dict[Path, FrameIDAnalysisResult] = {}

        for clip_plan in cam_plans:
            try:
                video_path = Path(clip_plan.video.path).resolve()
                analysis_result = video_analysis_cache.get(video_path)
                if analysis_result is None:
                    try:
                        analysis_result = analyze_video(
                            clip_plan.video,
                            outdir=analysis_dir,
                        )
                        video_analysis_cache[video_path] = analysis_result
                    except Exception as analysis_exc:
                        LOGGER.error(
                            "Video analysis failed for %s cam %s: %s",
                            clip_plan.video.segment_id,
                            cam_serial,
                            analysis_exc,
                        )
                        analysis_result = None

                clip_path = clip_video(clip_plan, clips_dir, overwrite)
                padded_path = pad_video_if_needed(clip_plan, clip_path, clips_dir)
                clip_paths.append(padded_path)
                clip_entry = {
                    "segment_id": clip_plan.video.segment_id,
                    "cam_serial": cam_serial,
                    "source_video": str(clip_plan.video.path),
                    "clip_path": str(padded_path),
                    "clip_start_index": int(clip_plan.clip_start_index),
                    "clip_end_index": int(clip_plan.clip_end_index),
                    "frame_start": int(clip_plan.frame_ids[0]),
                    "frame_end": int(clip_plan.frame_ids[-1]),
                    "local_frame_start": int(clip_plan.frame_ids_local[0]),
                    "local_frame_end": int(clip_plan.frame_ids_local[-1]),
                    "frame_count": int(len(clip_plan.frame_ids)),
                    "serial_start": int(clip_plan.serials[0]),
                    "serial_end": int(clip_plan.serials[-1]),
                    "frame_rate": float(clip_plan.video.frame_rate or 0.0),
                }
                if analysis_result is not None:
                    clip_entry["analysis_path"] = str(analysis_result.out_json_path)
                    clip_entry["analysis_missing_frames"] = int(
                        analysis_result.missing_frames
                    )
                    clip_entry["analysis_counts"] = analysis_result.counts
                    clip_entry["analysis_strictly_monotonic"] = bool(
                        analysis_result.strictly_monotonic
                    )
                if clip_plan.video.start_realtime:
                    clip_entry["segment_start_realtime"] = (
                        clip_plan.video.start_realtime.isoformat()
                    )
                clip_metadata_entries.append(clip_entry)
            except Exception as exc:
                LOGGER.error(
                    "Clip/pad failed for %s cam %s (rough mode): %s",
                    clip_plan.video.segment_id,
                    cam_serial,
                    exc,
                )
                continue

        if not clip_paths:
            LOGGER.error(
                "No successfully prepared clips for %s camera %s (rough mode)",
                task.task_id,
                cam_serial,
            )
            continue

        work_dir.mkdir(parents=True, exist_ok=True)
        clip_metadata_doc = {
            "task_id": task.task_id,
            "mode": sync_plan.mode,
            "cam_serial": cam_serial,
            "audio_path": str(sync_plan.audio_path),
            "audio_window_start": sync_plan.audio_start.isoformat(),
            "audio_window_end": sync_plan.audio_end.isoformat(),
            "clip_entries": clip_metadata_entries,
        }
        metadata_path = work_dir / "clip_metadata.json"
        if not metadata_path.exists() or overwrite:
            metadata_path.write_text(
                json.dumps(clip_metadata_doc, indent=2), encoding="utf-8"
            )

        merged_dir = work_dir / "merged"
        total_frames = sum(len(plan.frame_ids) for plan in cam_plans)
        audio_span = (sync_plan.audio_end - sync_plan.audio_start).total_seconds()
        target_fps = (
            total_frames / audio_span if audio_span > 0 and total_frames > 0 else None
        )
        if target_fps:
            LOGGER.debug(
                "Target FPS for %s camera %s set to %.6f (frames=%d span=%.3fs)",
                task.task_id,
                cam_serial,
                target_fps,
                total_frames,
                audio_span,
            )
        try:
            merged_video = concat_videos(
                clip_paths, merged_dir, overwrite, target_fps=target_fps
            )
        except Exception as exc:
            LOGGER.error(
                "Video concat failed for %s camera %s (rough mode): %s",
                task.task_id,
                cam_serial,
                exc,
            )
            continue

        synced_dir = cam_dir / "synced_video"
        synced_dir.mkdir(parents=True, exist_ok=True)
        final_path = synced_dir / f"{task.task_id}_{cam_serial}.mp4"
        try:
            mux_audio(merged_video, sync_plan.audio_path, final_path, overwrite)
        except Exception as exc:
            LOGGER.error(
                "Mux failed for %s camera %s (rough mode): %s",
                task.task_id,
                cam_serial,
                exc,
            )
            continue

        LOGGER.info(
            "Rough-synced output written: %s (camera %s)",
            final_path,
            cam_serial,
        )
        success = True

    return success


def prepare_rough_sync_plan(
    task: TaskContext,
    video_dir: Path,
    out_dir: Path,
    room_mic: str,
    camera_serials: Optional[Set[str]],
    overwrite: bool,
) -> Optional[SyncPlan]:
    """Assemble the clip and audio plan required to run the rough sync pipeline."""
    LOGGER.info("Starting rough sync for %s", task.task_id)
    LOGGER.debug(
        "Task %s NS5 window %s -> %s",
        task.task_id,
        task.ns5_start_utc.isoformat(),
        task.ns5_end_utc.isoformat(),
    )

    audio_dir = out_dir / "audio"
    audio_path = audio_dir / f"{task.task_id}-{room_mic}.wav"
    try:
        audio_path, audio_start, audio_end = extract_full_ns5_audio(
            task, room_mic, audio_path, overwrite
        )
    except Exception as exc:
        LOGGER.error("Audio extraction (rough) failed for %s: %s", task.task_id, exc)
        return None

    videos_by_cam = collect_videos_by_time(
        video_dir, task.ns5_start_utc, task.ns5_end_utc, camera_serials
    )
    if not videos_by_cam:
        LOGGER.warning(
            "No videos overlap NS5 audio window for %s (rough mode)",
            task.task_id,
        )
        return None

    _warn_missing_cameras(
        task.task_id,
        "rough",
        camera_serials,
        set(videos_by_cam.keys()),
    )
    LOGGER.debug(
        "Task %s will process cameras: %s",
        task.task_id,
        ", ".join(sorted(videos_by_cam.keys())) if videos_by_cam else "none",
    )

    clip_plans_by_cam: Dict[str, List[VideoSegmentClipPlan]] = {}
    for cam_serial, videos in sorted(videos_by_cam.items()):
        if camera_serials and cam_serial not in camera_serials:
            continue
        if not videos:
            continue
        LOGGER.debug(
            "Building clip plans for task %s camera %s (%d videos)",
            task.task_id,
            cam_serial,
            len(videos),
        )
        cam_plans = _build_time_clip_plans_for_camera(
            task.task_id,
            cam_serial,
            videos,
            audio_start,
            audio_end,
        )
        if cam_plans:
            clip_plans_by_cam[cam_serial] = cam_plans

    if not clip_plans_by_cam:
        return None

    return SyncPlan(
        mode="rough",
        audio_path=audio_path,
        clip_plans_by_cam=clip_plans_by_cam,
        audio_start=audio_start,
        audio_end=audio_end,
        audio_ready=True,
    )


def process_task(
    task: TaskContext,
    video_dir: Path,
    out_dir: Path,
    room_mic: str,
    overwrite: bool,
    camera_serials: Optional[Set[str]] = None,
) -> bool:
    """Execute the rough sync workflow for a stitched task."""
    sync_plan = prepare_rough_sync_plan(
        task,
        video_dir,
        out_dir,
        room_mic,
        camera_serials,
        overwrite,
    )
    if not sync_plan:
        return False

    return _execute_sync_plan(sync_plan, task, out_dir, room_mic, overwrite)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the rough sync CLI."""
    parser = argparse.ArgumentParser(description="EMU rough-sync utility")
    parser.add_argument("--patient-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Optional task-name keywords to match (case-insensitive).",
    )
    parser.add_argument(
        "--cam-serial",
        dest="cam_serials",
        action="append",
        default=None,
        help=(
            "Camera serial to process (repeat for multiple). If omitted, all cameras are synced."
        ),
    )
    parser.add_argument(
        "--room-mic",
        choices=("roommic1", "roommic2"),
        default="roommic1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite intermediate and final outputs if they exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser.parse_args(argv)


def configure_logging(level: str, log_path: Optional[Path] = None) -> bool:
    """Configure logging outputs and report whether file logging is active."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    file_handler_error: Optional[str] = None
    file_handler_added = False
    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
            file_handler_added = True
        except Exception as exc:  # pragma: no cover - defensive fallback
            file_handler_error = str(exc)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    if log_path and not file_handler_added:
        LOGGER.warning(
            "Unable to create log file %s (%s); continuing with console logging only",
            log_path,
            file_handler_error or "unknown error",
        )
    return file_handler_added


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the EMU rough sync workflow."""
    args = parse_args(argv)

    patient_dir = args.patient_dir.resolve()
    video_dir = args.video_dir.resolve()
    out_dir = args.out_dir.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / "logs" / f"cli_emu_rough_{timestamp}.log"
    if configure_logging(args.log_level, log_path):
        LOGGER.info("Log file: %s", log_path)

    if not patient_dir.is_dir():
        LOGGER.error("Patient dir not found: %s", patient_dir)
        return 2
    if not video_dir.is_dir():
        LOGGER.error("Video dir not found: %s", video_dir)
        return 2
    try:
        _iter_date_dirs(video_dir)
    except Exception as exc:
        LOGGER.error("Invalid video directory structure: %s", exc)
        return 2
    LOGGER.info("Validated video directory structure under %s", video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contexts = discover_task_contexts(
        patient_dir,
        args.keywords,
    )
    if not contexts:
        LOGGER.error("No stitched tasks discovered.")
        return 3

    camera_serials: Optional[Set[str]] = None
    if args.cam_serials:
        camera_serials = {c.strip() for c in args.cam_serials if c and c.strip()}
        if not camera_serials:
            camera_serials = None

    exit_code = 0
    for context in contexts:
        task_out_dir = out_dir / context.patient_id / context.task_id
        task_out_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Processing task %s", context.task_id)
        result = process_task(
            context,
            video_dir=video_dir,
            out_dir=task_out_dir,
            room_mic=args.room_mic,
            overwrite=args.overwrite,
            camera_serials=camera_serials,
        )
        if not result:
            exit_code = 4
        else:
            LOGGER.debug("Task %s completed successfully", context.task_id)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
