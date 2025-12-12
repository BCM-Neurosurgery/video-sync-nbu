"""Utilities to derive absolute audio start timestamps using anchors.

This module provides a reusable API that inspects the same artifacts created by
``cli_nbu`` (decoded serial CSVs, anchors, camera JSON companions) to emit the
absolute start timestamps (UTC + Chicago) for every audio file discovered under
a visit.  When the typical outputs already exist (e.g.,
``audio_decoded/raw-gapfilled-filtered.csv`` or
``gapfilled-filtered-anchors.json``) the helper reuses them; otherwise it
decodes + gap-fills + filters the serial audio and synthesizes anchors on the
fly, writing the intermediate CSVs alongside ``cli_nbu``'s layout.  The module
can be imported from other drivers or executed as a standalone CLI:

    python -m scripts.time.find_audio_abs_time \
        --audio-dir /path/to/visit/audio \
        --video-dir /path/to/video/jsons \
        --out-dir   /path/to/pipeline/output

The logic is intentionally stateless: we rely on the earliest anchor across
all processed segments, use the frame's realtime metadata to determine the
instant when that serial was observed, and back-compute the audio start by
subtracting the serial's sample offset.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from zoneinfo import ZoneInfo

from scripts.index.audiodiscover import AudioDiscoverer
from scripts.index.common import DEFAULT_TZ
from scripts.index.videodiscover import VideoDiscoverer, build_video_obj
from scripts.align.collect_anchors import (
    load_serial_index_csv,
    _collect_anchors_for_cam,
    _extract_cam_arrays_from_video,
)
from scripts.fix.audiofilter import filter_audio_file
from scripts.fix.audiogapfiller import gapfill_csv_file
from scripts.decode.wavfileparser import decode_to_raw

LOGGER = logging.getLogger(__name__)

REALTIME_STRICT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
ANCHOR_FILENAME = "gapfilled-filtered-anchors.json"
RAW_CSV_NAME = "raw.csv"
GAPFILLED_CSV_NAME = "raw-gapfilled.csv"
FILTERED_CSV_NAME = "raw-gapfilled-filtered.csv"
AUDIO_DECODED_SUBDIR = "audio_decoded"
AUDIO_METADATA_SUBDIR = "audio_metadata"
DEFAULT_METADATA_FILENAME = "audio_abs_start.json"


@dataclass(frozen=True)
class OutputLayout:
    """Convenience wrapper for standard ``cli_nbu`` directory layout."""

    root: Path

    @property
    def decoded_dir(self) -> Path:
        return self.root / AUDIO_DECODED_SUBDIR

    @property
    def metadata_dir(self) -> Path:
        return self.root / AUDIO_METADATA_SUBDIR

    @property
    def default_metadata_path(self) -> Path:
        return self.metadata_dir / DEFAULT_METADATA_FILENAME

    @property
    def raw_csv(self) -> Path:
        return self.decoded_dir / RAW_CSV_NAME

    @property
    def gapfilled_csv(self) -> Path:
        return self.decoded_dir / GAPFILLED_CSV_NAME

    @property
    def filtered_csv(self) -> Path:
        return self.decoded_dir / FILTERED_CSV_NAME

    def iter_anchor_files(self) -> Iterable[Path]:
        if not self.root.exists():
            return iter(())
        return self.root.rglob(ANCHOR_FILENAME)


@dataclass(frozen=True)
class AnchorEntry:
    """Minimal anchor information required to map serial samples to video."""

    serial: int
    audio_sample: int
    segment_id: str
    cam_serial: str
    frame_id: int
    frame_id_reidx: Optional[int]


@dataclass(frozen=True)
class AnchorMatch:
    """Resolved anchor enriched with realtime metadata."""

    anchor: AnchorEntry
    anchor_path: Path
    frame_time_utc: datetime


@dataclass(frozen=True)
class AudioStartRecord:
    """Represents the absolute start time for one audio file."""

    audio_path: Path
    channel: int
    sample_rate: int
    start_time_utc: datetime
    start_time_chicago: datetime
    reference_audio_sample: int
    reference_sample_rate: int
    anchor_segment: str
    anchor_cam_serial: str
    anchor_frame_id: int
    anchor_frame_id_reidx: Optional[int]
    anchor_frame_time_utc: datetime
    anchor_json_path: Path


def _parse_anchor_entry(payload: dict) -> AnchorEntry:
    """Convert a JSON dict into an :class:`AnchorEntry`."""

    return AnchorEntry(
        serial=int(payload["serial"]),
        audio_sample=int(payload["audio_sample"]),
        segment_id=str(payload["segment_id"]),
        cam_serial=str(payload["cam_serial"]),
        frame_id=int(payload.get("frame_id", 0)),
        frame_id_reidx=(
            int(payload["frame_id_reidx"])
            if payload.get("frame_id_reidx") is not None
            else None
        ),
    )


def _parse_realtime(value: str) -> Optional[datetime]:
    """Parse realtime strings into aware UTC datetimes."""

    for parser in (
        lambda v: datetime.strptime(v, REALTIME_STRICT_FORMAT),
        datetime.fromisoformat,
    ):
        try:
            dt = parser(value)
            break
        except Exception:
            continue
    else:
        LOGGER.debug("Unable to parse realtime value '%s'", value)
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_real_times(values: Optional[Sequence[object]]) -> Optional[List[datetime]]:
    """Convert a JSON ``real_times`` array into UTC-aware datetimes."""

    if not values:
        return None
    parsed: List[datetime] = []
    for item in values:
        if isinstance(item, datetime):
            dt = item
        elif isinstance(item, str):
            dt = _parse_realtime(item)
            if dt is None:
                return None
        else:
            LOGGER.debug("Unexpected realtime entry type: %s", type(item).__name__)
            return None
        parsed.append(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))
    return parsed


def _anchor_frame_time_from_video(video, anchor: AnchorEntry) -> datetime:
    """Resolve frame realtime using the ``Video`` companion metadata."""

    cam_json = getattr(video, "companion_json", None)
    if cam_json is None:
        raise RuntimeError(f"Video {video.path} missing companion metadata")

    real_times = _coerce_real_times(getattr(cam_json, "real_times", None))
    idx = (
        anchor.frame_id_reidx if anchor.frame_id_reidx is not None else anchor.frame_id
    )

    if real_times and 0 <= idx < len(real_times):
        return real_times[idx]

    start_rt = getattr(cam_json, "start_realtime", None)
    if start_rt is None:
        raise RuntimeError(
            f"Camera JSON {cam_json.path} lacks realtime data for segment {anchor.segment_id}"
        )

    fps = getattr(video, "frame_rate", None)
    if not fps or fps <= 0:
        LOGGER.warning(
            "Video %s missing valid frame_rate; defaulting to 30fps for anchor approximation",
            video.path.name,
        )
        fps = 30.0

    approx = start_rt + timedelta(seconds=float(idx) / float(fps))
    if not real_times:
        LOGGER.warning(
            "Video %s lacks per-frame realtime metadata; approximating using fps=%.3f",
            video.path.name,
            fps,
        )
    return approx if approx.tzinfo else approx.replace(tzinfo=timezone.utc)


def _iter_videos(video_dir: Path):
    """Yield ``Video`` objects discovered under ``video_dir``."""

    discoverer = VideoDiscoverer(video_dir, log=LOGGER)
    for group in discoverer.discover():
        if not group.videos:
            continue
        for video in group.videos:
            yield video


def _iter_existing_anchor_files(layout: OutputLayout) -> Iterable[Path]:
    """Yield anchor JSON files stored under ``layout``'s root."""

    anchors = layout.iter_anchor_files()
    return anchors if anchors is not None else iter(())


def _load_anchor_file(path: Path) -> List[AnchorEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):  # pragma: no cover - defensive guard
        raise ValueError(f"Anchor file {path} is not a list")
    return [_parse_anchor_entry(item) for item in raw]


def _anchor_from_existing_files(
    layout: OutputLayout,
    video_dir: Path,
) -> Optional[AnchorMatch]:
    """Return an anchor derived from on-disk anchor JSONs (if any exist)."""

    best: Optional[tuple[AnchorEntry, Path]] = None
    for anchor_path in _iter_existing_anchor_files(layout):
        try:
            entries = _load_anchor_file(anchor_path)
        except Exception as exc:
            LOGGER.warning("Skipping anchor file %s: %s", anchor_path, exc)
            continue
        for entry in entries:
            if best is None or entry.audio_sample < best[0].audio_sample:
                best = (entry, anchor_path)

    if best is None:
        return None

    entry, anchor_path = best
    video = build_video_obj(video_dir, entry.segment_id, entry.cam_serial, log=LOGGER)
    if video is None:
        LOGGER.warning(
            "Anchor references missing video for %s cam %s; regenerating anchors",
            entry.segment_id,
            entry.cam_serial,
        )
        return None

    frame_time = _anchor_frame_time_from_video(video, entry)
    return AnchorMatch(anchor=entry, anchor_path=anchor_path, frame_time_utc=frame_time)


def _derive_anchor_from_videos(
    serial_index_map: Dict[int, int], video_dir: Path
) -> AnchorMatch:
    """Generate anchors on the fly and return the earliest one."""

    if not serial_index_map:
        raise RuntimeError("Serial index map is empty; decode step may have failed.")

    best: Optional[tuple[AnchorEntry, object]] = None
    processed = 0
    LOGGER.info("Scanning videos in %s to derive anchors…", video_dir)

    for video in _iter_videos(video_dir):
        processed += 1
        if processed % 10 == 0:
            LOGGER.info(
                "Processed %d video(s) while searching for earliest anchor",
                processed,
            )
        try:
            serials, frame_ids, frame_ids_reidx = _extract_cam_arrays_from_video(video)
        except Exception as exc:
            LOGGER.debug("Skipping video %s: %s", video.path.name, exc)
            continue

        anchors = _collect_anchors_for_cam(
            serial_index_map,
            video.segment_id,
            video.cam_serial,
            serials,
            frame_ids,
            frame_ids_reidx,
            min_k=1,
            min_span_ratio=0.0,
        )
        for anchor in anchors:
            entry = AnchorEntry(
                serial=anchor.serial,
                audio_sample=anchor.audio_sample,
                segment_id=anchor.segment_id,
                cam_serial=anchor.cam_serial,
                frame_id=anchor.frame_id,
                frame_id_reidx=anchor.frame_id_reidx,
            )
            if best is None or entry.audio_sample < best[0].audio_sample:
                best = (entry, video)

    if best is None:
        raise RuntimeError(
            "Unable to derive anchors; ensure the video directory contains valid segments."
        )

    anchor_entry, video = best
    frame_time = _anchor_frame_time_from_video(video, anchor_entry)
    anchor_json_path = getattr(video.companion_json, "path", None)
    if anchor_json_path is None:
        anchor_json_path = Path(video.path).with_suffix(".json")
    return AnchorMatch(
        anchor=anchor_entry,
        anchor_path=anchor_json_path,
        frame_time_utc=frame_time,
    )


def resolve_reference_anchor(
    serial_index_map: Dict[int, int],
    video_dir: Path,
    *,
    layout: OutputLayout,
) -> AnchorMatch:
    """Use existing anchors when available; otherwise synthesize from videos."""

    existing = _anchor_from_existing_files(layout, video_dir)
    if existing is not None:
        LOGGER.info(
            "Reusing existing anchors from %s (segment %s cam %s)",
            existing.anchor_path,
            existing.anchor.segment_id,
            existing.anchor.cam_serial,
        )
        return existing
    LOGGER.info("No cached anchors found; deriving anchors from video metadata")
    return _derive_anchor_from_videos(serial_index_map, video_dir)


def _discover_audio(audio_dir: Path) -> AudioDiscoverer:
    """Instantiate ``AudioDiscoverer`` with a quiet logger."""

    logger = logging.getLogger("find_audio_abs_time.audio")
    logger.addHandler(logging.NullHandler())
    return AudioDiscoverer(audio_dir=audio_dir, log=logger)


def _ensure_serial_csvs(
    serial_audio_path: Path,
    layout: OutputLayout,
    *,
    site: str,
) -> Path:
    """Ensure raw/gapfilled/filtered CSVs exist and return the filtered path."""

    layout.decoded_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = layout.raw_csv
    if raw_csv.exists():
        source_raw = raw_csv
        LOGGER.info("Reusing decoded serial CSV: %s", source_raw)
    else:
        source_raw, _, _, _ = decode_to_raw(
            serial_audio_path, outdir=layout.decoded_dir, site=site
        )
        LOGGER.info("Decoded serial audio → %s", source_raw)

    gapfilled_csv = layout.gapfilled_csv
    if gapfilled_csv.exists():
        gapfilled_path = gapfilled_csv
        LOGGER.info("Reusing gapfilled CSV: %s", gapfilled_path)
    else:
        gapfilled_path = gapfill_csv_file(source_raw, out_path=gapfilled_csv)
        LOGGER.info("Gapfilled serial CSV → %s", gapfilled_path)

    filtered_csv = layout.filtered_csv
    if not filtered_csv.exists():
        filter_audio_file(gapfilled_path, out_path=filtered_csv)
        LOGGER.info("Filtered serial CSV → %s", filtered_csv)
    else:
        LOGGER.info("Reusing filtered CSV: %s", filtered_csv)

    return filtered_csv


def _build_serial_index_map(
    serial_audio_path: Path,
    layout: OutputLayout,
    *,
    site: str,
) -> Dict[int, int]:
    """Decode/gapfill/filter serial audio and return {serial -> start_sample}."""

    filtered_csv = _ensure_serial_csvs(serial_audio_path, layout, site=site)
    return load_serial_index_csv(filtered_csv)


def compute_audio_start_records(
    audio_dir: Path,
    video_dir: Path,
    *,
    out_dir: Path,
    local_tz: ZoneInfo = DEFAULT_TZ,
    site: str = "jamail",
) -> List[AudioStartRecord]:
    """Compute absolute start timestamps for every audio file in ``audio_dir``."""

    layout = OutputLayout(out_dir)
    discoverer = _discover_audio(audio_dir)
    group = discoverer.get_audio_group()
    serial_audio = group.serial_audio
    if serial_audio is None:
        raise RuntimeError("Audio directory does not contain a serial (-03) channel")

    if serial_audio.sample_rate <= 0:
        raise RuntimeError("Serial audio sample rate must be positive")

    serial_index_map = _build_serial_index_map(
        Path(serial_audio.path),
        layout,
        site=site,
    )

    anchor_match = resolve_reference_anchor(
        serial_index_map,
        video_dir,
        layout=layout,
    )
    anchor_time = anchor_match.frame_time_utc
    offset_seconds = anchor_match.anchor.audio_sample / float(serial_audio.sample_rate)
    start_utc = anchor_time - timedelta(seconds=offset_seconds)
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    start_local = start_utc.astimezone(local_tz)

    records: List[AudioStartRecord] = []
    for channel, audio in sorted(group.audios.items()):
        records.append(
            AudioStartRecord(
                audio_path=audio.path,
                channel=channel,
                sample_rate=int(audio.sample_rate),
                start_time_utc=start_utc,
                start_time_chicago=start_local,
                reference_audio_sample=anchor_match.anchor.audio_sample,
                reference_sample_rate=int(serial_audio.sample_rate),
                anchor_segment=anchor_match.anchor.segment_id,
                anchor_cam_serial=anchor_match.anchor.cam_serial,
                anchor_frame_id=anchor_match.anchor.frame_id,
                anchor_frame_id_reidx=anchor_match.anchor.frame_id_reidx,
                anchor_frame_time_utc=anchor_time,
                anchor_json_path=anchor_match.anchor_path,
            )
        )
    return records


def _record_to_payload(record: AudioStartRecord) -> dict:
    """Render an :class:`AudioStartRecord` into a JSON-friendly dict."""

    payload = asdict(record)
    payload["audio_path"] = str(record.audio_path)
    payload["start_time_utc"] = record.start_time_utc.isoformat()
    payload["start_time_chicago"] = record.start_time_chicago.isoformat()
    payload["anchor_frame_time_utc"] = record.anchor_frame_time_utc.isoformat()
    payload["anchor_json_path"] = str(record.anchor_json_path)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute absolute audio start timestamps using anchors."
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TZ.key,
        help="IANA timezone name for the local site (default: America/Chicago)",
    )
    parser.add_argument(
        "--site",
        default="jamail",
        help="Decoder site preset (jamail | nbu_lounge | nbu_sleep).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the computed records as JSON.",
    )

    args = parser.parse_args(argv)

    # Configure basic logging when invoked standalone (no handlers configured yet).
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(message)s",
        )

    try:
        local_tz = ZoneInfo(args.timezone)
    except Exception as exc:  # pragma: no cover - invalid TZ
        raise SystemExit(f"Invalid timezone '{args.timezone}': {exc}")

    records = compute_audio_start_records(
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
        out_dir=args.out_dir,
        local_tz=local_tz,
        site=args.site,
    )

    payload = [_record_to_payload(rec) for rec in records]

    if args.output_json:
        output_path = args.output_json
    else:
        output_path = OutputLayout(args.out_dir).default_metadata_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d record(s) to %s", len(payload), output_path)

    if args.output_json:
        print(output_path)
    else:
        print(json.dumps(payload, indent=2))
        print(f"\nSaved metadata to {output_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
