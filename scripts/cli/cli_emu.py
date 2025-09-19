"""
EMU A/V stitching CLI
=====================

This script wires NEV/NS5 recordings with chunked video segments by:

1. Discovering EMU tasks under a patient directory and loading them into
   ``scripts.models.StitchedTask`` objects.
2. Extracting the chunk-serial window from the NEV digital events so we know the
   timeline to stitch.
3. Scanning the video directory for segments whose JSON chunk serials overlap
   the NEV window.
4. Aligning NS5 room audio to the NEV timestamps and exporting the slice as WAV.
5. Clipping each relevant video segment using the serial window and padding
   missing frames when gaps are detected.
6. Concatenating the clips and muxing the stitched audio to produce per-camera
   synced MP4s (optionally restricted via ``--cam-serial``).

Input layout (example)
----------------------
- video/
    - 20250505
        - <SEGMENT_ID>_<TIMESTAMP>.<CAM>.mp4
        - <SEGMENT_ID>_<TIMESTAMP>.json
        ...
    - 20250506
        ...
- YFP/                                (patient_dir)
    - EMU-0088_convo/                 (task_id)
        - EMU-0088_convo_NSP-1.nev
        - EMU-0088_convo_NSP-1.ns5

Output layout (for each task)
-----------------------------
- out/
    - YFP/
        - EMU-0088_convo/
            - audio/
                - EMU-0088_convo-roommic1.wav
            - 23512099/
                - work/
                    - clips/
                    - clips/padded/
                    - merged/
                - synced_video/
                    - EMU-0088_convo_23512099.mp4
            - 23512110/
                ...
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from scripts.index.videodiscover import VideoDiscoverer
from scripts.models import CamJson, DIGIEVTS, NEV, NS5, RoomAudio, StitchedTask, Video
from scripts.pad.videoplanapplier import apply_video_padding_plan
from scripts.parsers.nevfileparser import Nev
from scripts.parsers.ns5fileparser import Nsx
from scripts.parsers.videofileparser import VideoFileParser
from scripts.utility.utils import ts2unix

LOGGER = logging.getLogger("cli_emu")


@dataclass(frozen=True)
class ChunkSerialRange:
    """Range of chunk serials and their NEV timestamps."""

    start_serial: int
    end_serial: int
    start_timestamp: int
    end_timestamp: int


@dataclass
class VideoSegmentClipPlan:
    """Represents a clip window for a single camera segment."""

    video: Video
    serials: List[int]
    frame_ids: List[int]
    frame_ids_local: List[int]
    clip_start_index: int
    clip_end_index: int


@dataclass
class TaskContext:
    """Runtime bundle holding the stitched task and parsed IO handles."""

    stitched: StitchedTask
    nev_parser: Nev
    nsx_parser: Nsx


def _ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found on PATH.")


def _infer_sample_rate(sample_resolution_microseconds: float) -> int:
    if sample_resolution_microseconds <= 0:
        return 0
    return int(round(1_000_000 / float(sample_resolution_microseconds)))


def _find_nsp1_file(task_dir: Path, ext: str) -> Optional[Path]:
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


def load_nev(path: Path) -> tuple[NEV, Nev]:
    parser = Nev(str(path))
    data = parser.get_data()
    raw_events = data.get("digital_events") if isinstance(data, dict) else None
    raw_df = parser.get_digital_events_df()
    chunk_df = parser.get_chunk_serial_df() if parser.has_unparsed_data() else None

    start_serial = None
    end_serial = None
    start_ts = None
    end_ts = None
    if chunk_df is not None and not chunk_df.empty:
        start_serial = int(chunk_df["chunk_serial"].iloc[0])
        end_serial = int(chunk_df["chunk_serial"].iloc[-1])
        start_ts = int(chunk_df["TimeStamps"].iloc[0])
        end_ts = int(chunk_df["TimeStamps"].iloc[-1])

    digital_events = DIGIEVTS(
        raw=raw_events,
        raw_df=raw_df,
        chunk_serial_df=chunk_df,
        start_serial=start_serial,
        end_serial=end_serial,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
    )

    basic_header = parser.get_basic_header()
    sample_resolution = float(
        basic_header.get("SampleTimeResolution", parser.get_timestampResolution())
    )
    duration_sec = (parser.get_end_timestamp() - parser.get_start_timestamp()) / float(
        parser.get_timestampResolution()
    )

    nev_model = NEV(
        path=path,
        start_utc_time=parser.get_time_origin(),
        sample_resolution=sample_resolution,
        duration=duration_sec,
        digital_events=digital_events,
    )
    return nev_model, parser


def load_ns5(path: Path) -> tuple[NS5, Nsx]:
    parser = Nsx(str(path))
    sample_resolution = float(parser.get_sample_resolution())
    sample_rate = _infer_sample_rate(sample_resolution)
    start_ts = int(parser.get_start_timestamp())

    def _load_channel(channel_name: str) -> RoomAudio:
        try:
            channel_array = parser.get_channel_array(channel_name)
            arr = np.asarray(channel_array)
            num_samples = int(arr.shape[0])
            end_ts = start_ts + max(0, num_samples - 1)
            duration = num_samples / sample_rate if sample_rate else None
            return RoomAudio(
                raw_array=arr,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                duration=duration,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Failed reading %s from %s: %s", channel_name, path.name, exc
            )
            return RoomAudio(
                raw_array=None, start_timestamp=None, end_timestamp=None, duration=None
            )

    ns5_model = NS5(
        path=path,
        start_utc_time=parser.get_timeOrigin(),
        sample_resolution=sample_resolution,
        duration=float(parser.get_recording_duration_s()),
        room_mic1=_load_channel("RoomMic1"),
        room_mic2=_load_channel("RoomMic2"),
    )
    return ns5_model, parser


def discover_task_contexts(
    patient_dir: Path, keywords: Optional[Sequence[str]]
) -> List[TaskContext]:
    contexts: List[TaskContext] = []
    for task_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        name = task_dir.name
        if keywords and not any(kw.lower() in name.lower() for kw in keywords):
            LOGGER.debug("Skipping task %s (keyword filter)", name)
            continue

        nev_path = _find_nsp1_file(task_dir, "nev")
        ns5_path = _find_nsp1_file(task_dir, "ns5")
        if not nev_path or not ns5_path:
            LOGGER.warning(
                "Missing NEV/NS5 in %s (nev=%s, ns5=%s)",
                name,
                bool(nev_path),
                bool(ns5_path),
            )
            continue

        try:
            nev_model, nev_parser = load_nev(nev_path)
            ns5_model, nsx_parser = load_ns5(ns5_path)
        except Exception as exc:  # pragma: no cover - I/O heavy
            LOGGER.error("Failed loading task %s: %s", name, exc)
            continue

        stitched = StitchedTask(
            patient_id=patient_dir.name,
            task_id=name,
            nsp1_nev=nev_model,
            nsp1_ns5=ns5_model,
        )
        contexts.append(
            TaskContext(stitched=stitched, nev_parser=nev_parser, nsx_parser=nsx_parser)
        )
    return contexts


def compute_chunk_range(nev: NEV) -> Optional[ChunkSerialRange]:
    df = nev.digital_events.chunk_serial_df
    if df is None or df.empty:
        return None
    return ChunkSerialRange(
        start_serial=int(df["chunk_serial"].iloc[0]),
        end_serial=int(df["chunk_serial"].iloc[-1]),
        start_timestamp=int(df["TimeStamps"].iloc[0]),
        end_timestamp=int(df["TimeStamps"].iloc[-1]),
    )


def _serial_min_max(serials: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    values: List[int] = []
    for serial in serials:
        try:
            val = int(serial)
        except Exception:
            continue
        if val < 0:
            continue
        values.append(val)
    if not values:
        return None, None
    return min(values), max(values)


def _iter_video_roots(video_dir: Path) -> List[Path]:
    """Return candidate directories that may contain segment MP4/JSON files."""

    def _has_assets(path: Path) -> bool:
        try:
            next(path.glob("*.json"))
            return True
        except StopIteration:
            pass
        try:
            next(path.glob("*.mp4"))
            return True
        except StopIteration:
            return False

    roots: List[Path] = []
    if video_dir.exists() and video_dir.is_dir() and _has_assets(video_dir):
        roots.append(video_dir)

    if video_dir.exists() and video_dir.is_dir():
        for child in sorted(video_dir.iterdir()):
            if not child.is_dir():
                continue
            if _has_assets(child):
                roots.append(child)

    return roots or ([video_dir] if video_dir.exists() else [])


def _choose_serials(cam_json: CamJson) -> Optional[List[int]]:
    if cam_json.fixed_serials:
        return cam_json.fixed_serials
    raise ValueError("CamJson missing fixed_serials")


def _choose_frame_ids(cam_json: CamJson) -> Optional[List[int]]:
    if cam_json.fixed_reidx_frame_ids:
        return cam_json.fixed_reidx_frame_ids
    raise ValueError("CamJson missing fixed_reidx_frame_ids")


def build_clip_plan(
    video: Video, chunk_range: ChunkSerialRange
) -> Optional[VideoSegmentClipPlan]:
    cam_json = video.companion_json
    if cam_json is None:
        return None
    serials = _choose_serials(cam_json)
    frames = _choose_frame_ids(cam_json)
    if not serials or not frames or len(serials) != len(frames):
        return None

    indices_in_range: List[int] = []
    normalized_serials: List[int] = []
    for idx, serial in enumerate(serials):
        try:
            val = int(serial)
        except Exception:
            val = -1
        normalized_serials.append(val)
        if val >= chunk_range.start_serial and val <= chunk_range.end_serial:
            indices_in_range.append(idx)

    if not indices_in_range:
        return None

    start_idx = min(indices_in_range)
    end_idx = max(indices_in_range)
    serial_slice = normalized_serials[start_idx : end_idx + 1]
    frame_slice = [int(frames[i]) for i in range(start_idx, end_idx + 1)]
    min_frame = frame_slice[0]
    local_frames = [f - min_frame for f in frame_slice]

    return VideoSegmentClipPlan(
        video=video,
        serials=serial_slice,
        frame_ids=frame_slice,
        frame_ids_local=local_frames,
        clip_start_index=start_idx,
        clip_end_index=end_idx,
    )


def discover_clip_plans(
    video_dir: Path,
    serial_range: ChunkSerialRange,
    camera_filter: Optional[Set[str]] = None,
) -> List[VideoSegmentClipPlan]:
    plans: List[VideoSegmentClipPlan] = []
    allowed: Optional[Set[str]] = (
        {c.strip() for c in camera_filter} if camera_filter else None
    )

    stop_scan = False

    for root in _iter_video_roots(video_dir):
        discoverer = VideoDiscoverer(root, log=LOGGER)
        try:
            video_groups = discoverer.discover()
        except Exception as exc:  # pragma: no cover - discovery failure
            LOGGER.error("Video discovery failed under %s: %s", root, exc)
            continue

        for group in video_groups:
            cam_jsons = (
                group.json.cam_jsons if group.json and group.json.cam_jsons else {}
            )
            if not cam_jsons:
                continue

            group_min: Optional[int] = None
            group_max: Optional[int] = None
            for cam_serial, cam_json in cam_jsons.items():
                if allowed and cam_serial not in allowed:
                    continue
                serials = _choose_serials(cam_json)
                if not serials:
                    continue
                min_serial, max_serial = _serial_min_max(serials)
                if min_serial is None or max_serial is None:
                    continue
                group_min = (
                    min_serial if group_min is None else min(group_min, min_serial)
                )
                group_max = (
                    max_serial if group_max is None else max(group_max, max_serial)
                )

            if group_min is None and group_max is None:
                continue

            if group_min is not None and group_min > serial_range.end_serial:
                stop_scan = True
                break
            if group_max is not None and group_max < serial_range.start_serial:
                continue

            for cam_serial, cam_json in sorted(cam_jsons.items()):
                if allowed and cam_serial not in allowed:
                    continue
                serials = _choose_serials(cam_json)
                if not serials:
                    continue
                cam_min, cam_max = _serial_min_max(serials)
                if cam_min is None or cam_max is None:
                    continue
                if (
                    cam_max < serial_range.start_serial
                    or cam_min > serial_range.end_serial
                ):
                    continue

                video = discoverer.discover_video(group.group_id, cam_serial)
                if not video:
                    continue
                plan = build_clip_plan(video, serial_range)
                if plan:
                    plans.append(plan)
        if stop_scan:
            break
    plans.sort(key=lambda p: (p.video.segment_id, p.video.cam_serial))
    return plans


def clip_video(plan: VideoSegmentClipPlan, out_dir: Path, overwrite: bool) -> Path:
    _ensure_tool("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    video = plan.video
    if video.frame_rate is None or float(video.frame_rate) <= 0:
        raise ValueError(
            f"Video {video.path} is missing a valid frame rate for clipping"
        )
    fps = float(video.frame_rate)
    start_frame = plan.frame_ids[0]
    end_frame = plan.frame_ids[-1]
    start_sec = start_frame / fps
    end_sec = (end_frame + 1) / fps
    duration = max(0.0, end_sec - start_sec)

    out_path = out_dir / f"{video.segment_id}.{video.cam_serial}.clip.mp4"
    ff_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video.path),
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        str(out_path),
    ]
    result = subprocess.run(
        ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clip failed for {video.path.name}: {result.stderr.strip()}"
        )
    return out_path


def build_padding_operations(plan: VideoSegmentClipPlan) -> List[dict]:
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
    ops = build_padding_operations(plan)
    if not ops:
        return clip_path

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


def _timestamp_to_datetime(
    time_origin: datetime, resolution: float, timestamp: int
) -> datetime:
    return ts2unix(time_origin, int(resolution), int(timestamp))


def extract_audio_slice(
    task: TaskContext,
    chunk_range: ChunkSerialRange,
    room_mic: str,
    out_path: Path,
    overwrite: bool,
) -> Path:
    channel_lookup = {"roommic1": "RoomMic1", "roommic2": "RoomMic2"}
    channel_name = channel_lookup[room_mic.lower()]
    audio = (
        task.stitched.nsp1_ns5.room_mic1
        if channel_name == "RoomMic1"
        else task.stitched.nsp1_ns5.room_mic2
    )
    if audio.raw_array is None:
        raise RuntimeError(f"Room audio {channel_name} unavailable in NS5")

    nsx = task.nsx_parser
    sample_rate = _infer_sample_rate(task.stitched.nsp1_ns5.sample_resolution)
    if sample_rate <= 0:
        raise RuntimeError("Invalid NS5 sample rate")

    nev_parser = task.nev_parser
    start_dt = _timestamp_to_datetime(
        nev_parser.get_time_origin(),
        nev_parser.get_timestampResolution(),
        chunk_range.start_timestamp,
    )
    end_dt = _timestamp_to_datetime(
        nev_parser.get_time_origin(),
        nev_parser.get_timestampResolution(),
        chunk_range.end_timestamp,
    )

    nsx_start_dt = _timestamp_to_datetime(
        nsx.get_timeOrigin(),
        nsx.timestampResolution,
        nsx.get_start_timestamp(),
    )
    sample_period = nsx.sampleResolution / 1_000_000
    start_offset = (start_dt - nsx_start_dt).total_seconds()
    end_offset = (end_dt - nsx_start_dt).total_seconds()
    start_idx = int(max(0, math.floor(start_offset / sample_period)))
    end_idx = int(min(audio.raw_array.shape[0], math.ceil(end_offset / sample_period)))
    if end_idx <= start_idx:
        raise RuntimeError("Computed empty audio slice")

    slice_array = np.asarray(audio.raw_array[start_idx:end_idx], dtype=np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        LOGGER.info("Audio output exists, skipping overwrite: %s", out_path)
        return out_path
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(slice_array.tobytes())
    return out_path


def concat_videos(clip_paths: List[Path], work_dir: Path, overwrite: bool) -> Path:
    if not clip_paths:
        raise RuntimeError("No clips to merge")
    if len(clip_paths) == 1:
        return clip_paths[0]

    _ensure_tool("ffmpeg")
    work_dir.mkdir(parents=True, exist_ok=True)
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
        "-c",
        "copy",
        str(out_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr.strip()}")
    return out_path


def mux_audio(
    video_path: Path, audio_path: Path, out_path: Path, overwrite: bool
) -> Path:
    _ensure_tool("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr.strip()}")
    return out_path


def process_task(
    task: TaskContext,
    video_dir: Path,
    out_dir: Path,
    room_mic: str,
    overwrite: bool,
    camera_serials: Optional[Set[str]] = None,
) -> bool:
    chunk_range = compute_chunk_range(task.stitched.nsp1_nev)
    if not chunk_range:
        LOGGER.warning("No chunk serials for task %s; skipping", task.stitched.task_id)
        return False

    plans = discover_clip_plans(video_dir, chunk_range, camera_filter=camera_serials)
    if not plans:
        LOGGER.warning(
            "No video segments overlap serial range for %s", task.stitched.task_id
        )
        return False

    plans_by_cam: Dict[str, List[VideoSegmentClipPlan]] = defaultdict(list)
    for plan in plans:
        plans_by_cam[plan.video.cam_serial].append(plan)

    if camera_serials:
        missing = sorted({c for c in camera_serials if c not in plans_by_cam})
        for cam in missing:
            LOGGER.warning(
                "Requested camera %s not found or lacked serial overlap for %s",
                cam,
                task.stitched.task_id,
            )

    audio_path: Optional[Path] = None
    audio_dir = out_dir / "audio"
    success = False

    for cam_serial, cam_plans in sorted(plans_by_cam.items()):
        if camera_serials and cam_serial not in camera_serials:
            continue

        cam_dir = out_dir / cam_serial
        work_dir = cam_dir / "work"
        clips_dir = work_dir / "clips"
        clip_paths: List[Path] = []
        for plan in sorted(
            cam_plans, key=lambda p: (p.video.segment_id, p.clip_start_index)
        ):
            try:
                clip_path = clip_video(plan, clips_dir, overwrite)
                padded_path = pad_video_if_needed(plan, clip_path, clips_dir)
                clip_paths.append(padded_path)
            except Exception as exc:
                LOGGER.error(
                    "Clip/pad failed for %s cam %s: %s",
                    plan.video.segment_id,
                    plan.video.cam_serial,
                    exc,
                )
                continue

        if not clip_paths:
            LOGGER.error(
                "All clips failed for task %s camera %s",
                task.stitched.task_id,
                cam_serial,
            )
            continue

        if audio_path is None:
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{task.stitched.task_id}-{room_mic}.wav"
            try:
                extract_audio_slice(task, chunk_range, room_mic, audio_path, overwrite)
            except Exception as exc:
                LOGGER.error(
                    "Audio extraction failed for %s: %s", task.stitched.task_id, exc
                )
                return False

        merged_dir = work_dir / "merged"
        try:
            merged_video = concat_videos(clip_paths, merged_dir, overwrite)
        except Exception as exc:
            LOGGER.error(
                "Video concat failed for %s cam %s: %s",
                task.stitched.task_id,
                cam_serial,
                exc,
            )
            continue

        synced_dir = cam_dir / "synced_video"
        final_path = synced_dir / f"{task.stitched.task_id}_{cam_serial}.mp4"
        try:
            mux_audio(merged_video, audio_path, final_path, overwrite)
        except Exception as exc:
            LOGGER.error(
                "Mux failed for %s cam %s: %s",
                task.stitched.task_id,
                cam_serial,
                exc,
            )
            continue
        LOGGER.info("Synced output written: %s (camera %s)", final_path, cam_serial)
        success = True

    return success


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMU stitching utility")
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


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(levelname)s] %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    patient_dir = args.patient_dir.resolve()
    video_dir = args.video_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not patient_dir.is_dir():
        LOGGER.error("Patient dir not found: %s", patient_dir)
        return 2
    if not video_dir.is_dir():
        LOGGER.error("Video dir not found: %s", video_dir)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    contexts = discover_task_contexts(patient_dir, args.keywords)
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
        task_out_dir = out_dir / context.stitched.patient_id / context.stitched.task_id
        task_out_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Processing task %s", context.stitched.task_id)
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
