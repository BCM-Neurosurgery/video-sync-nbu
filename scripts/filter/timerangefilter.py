#!/usr/bin/env python3
"""
timerangefilter.py — Filter serial CSV and video segments by time range or audio sample range

What it does
------------
This module exposes two filtering modes:
1) Time-range mode:
   - Query video JSON `real_times` to find frames within a user-specified clock range
   - Derive corresponding serial bounds
   - Filter the serial CSV by those serials
2) Audio-sample mode:
   - Filter the serial CSV by user-specified sample bounds
   - Derive corresponding serial bounds from the filtered CSV
   - Find affected segment/camera pairs from JSON fixed serials

Time Reference
--------------
Uses time from video JSON `real_times[]` array, which contains actual
timestamps in format: "YYYY-MM-DD HH:MM:SS.ffffff"
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from scripts.parsers.jsonfileparser import JsonParser
from scripts.index.videodiscover import VideoDiscoverer
from scripts.utility.utils import _name

logger = logging.getLogger(__name__)


class TimeRangeFilterError(Exception):
    """Raised when time range filtering fails."""

    pass


class AudioSampleRangeFilterError(Exception):
    """Raised when audio-sample range filtering fails."""

    pass


class TimeRangeFilter:
    """
    Filter serial CSV to a specific time range using video JSON timestamps.

    Attributes
    ----------
    serial_csv : Path
        Path to the input serial CSV file
    video_dir : Path
        Directory containing video JSON files
    time_start : datetime
        Start time (inclusive)
    time_end : datetime
        End time (inclusive)
    """

    def __init__(
        self,
        serial_csv: Path,
        video_dir: Path,
        time_start: str,
        time_end: str,
        user_timezone: str = "UTC",
    ):
        self.serial_csv = Path(serial_csv)
        self.video_dir = Path(video_dir)
        self.user_timezone = user_timezone

        # Parse time strings (will convert from user timezone to UTC)
        self.time_start, self.time_end = self._parse_time_range(time_start, time_end)

        # Storage for serial range
        self.serial_min: Optional[int] = None
        self.serial_max: Optional[int] = None
        self.affected_segments: Dict[str, Set[str]] = (
            {}
        )  # {segment_id: {cam_serial, ...}}

    def _parse_time_string(self, time_str: str) -> datetime:
        """
        Parse a time string in format: "YYYY-MM-DD HH:MM:SS"
        Interprets the time in user's timezone and converts to UTC.
        """
        time_str = time_str.strip()

        try:
            # Parse the naive datetime
            dt_naive = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

            # Localize to user's timezone
            try:
                user_tz = ZoneInfo(self.user_timezone)
                dt_local = dt_naive.replace(tzinfo=user_tz)
            except Exception as e:
                raise TimeRangeFilterError(
                    f"Invalid timezone '{self.user_timezone}': {e}"
                )

            # Convert to UTC and return as naive datetime (for comparison with video timestamps)
            dt_utc = dt_local.astimezone(timezone.utc)
            return dt_utc.replace(tzinfo=None)

        except ValueError as e:
            raise TimeRangeFilterError(
                f"Unable to parse time string '{time_str}'. "
                f"Required format: 'YYYY-MM-DD HH:MM:SS' (e.g., '2025-11-09 14:30:00')"
            )

    def _parse_time_range(
        self, start_str: str, end_str: str
    ) -> Tuple[datetime, datetime]:
        """Parse start and end time strings and apply 1-second buffer for boundary frame capture"""
        start_time = self._parse_time_string(start_str)
        end_time = self._parse_time_string(end_str)

        if end_time <= start_time:
            raise TimeRangeFilterError(
                f"End time ({end_time}) must be after start time ({start_time})"
            )

        # Apply 1-second buffer on both ends to ensure boundary frames are captured
        start_time_buffered = start_time - timedelta(seconds=1)
        end_time_buffered = end_time + timedelta(seconds=1)

        logger.info(
            "Time range filter (%s): %s to %s UTC (duration: %s, with 1s buffer on each end)",
            self.user_timezone,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time - start_time,
        )

        return start_time_buffered, end_time_buffered

    def _find_serial_range_from_videos(self) -> Tuple[int, int]:
        """
        Query all video JSONs to find serials within the time range.

        Returns: (serial_min, serial_max)
        """
        all_serials: Set[int] = set()

        # Get list of JSON files
        json_files = sorted(self.video_dir.glob("*.json"))
        if not json_files:
            raise TimeRangeFilterError(f"No JSON files found in {self.video_dir}")

        # Process each JSON
        for json_path in json_files:
            segment_id = json_path.stem

            try:
                jp = JsonParser(str(json_path))

                # Get real_times array
                real_times_raw = jp.dic.get("real_times", [])
                if not real_times_raw:
                    logger.warning("No real_times in %s, skipping", json_path.name)
                    continue

                # Parse timestamps
                real_times = [
                    datetime.strptime(rt, "%Y-%m-%d %H:%M:%S.%f")
                    for rt in real_times_raw
                ]

                # Use VideoDiscoverer to get fixed CamJson data (with fallbacks already applied)
                vd = VideoDiscoverer(self.video_dir, log=logger)
                _, cam_serials_list, cam_jsons = vd._build_json_wrapper(json_path, None)

                if not cam_jsons:
                    logger.warning("No camera data in %s, skipping", json_path.name)
                    continue

                # Build frame-wise serial data from fixed_serials
                # cam_jsons maps cam_serial_str -> CamJson (with fixed_serials already applied)
                num_frames = len(real_times)
                chunk_serial_data = []

                for frame_idx in range(num_frames):
                    frame_serials = []
                    for cam_serial_str in cam_serials_list:
                        cam_json = cam_jsons.get(cam_serial_str)
                        if (
                            cam_json
                            and cam_json.fixed_serials
                            and frame_idx < len(cam_json.fixed_serials)
                        ):
                            frame_serials.append(cam_json.fixed_serials[frame_idx])
                        else:
                            frame_serials.append(-1)
                    chunk_serial_data.append(frame_serials)

                camera_serials = [int(cs) for cs in cam_serials_list]

                # Find frames within time range (buffer already applied in _parse_time_range)
                for idx, rt in enumerate(real_times):
                    if self.time_start <= rt <= self.time_end:
                        # This frame is in range; get serials for all cameras
                        serials_for_frame = chunk_serial_data[idx]

                        for cam_idx, cam_serial in enumerate(camera_serials):
                            if cam_idx < len(serials_for_frame):
                                serial = serials_for_frame[cam_idx]
                                if serial > 0:  # Ignore invalid serials (-1, 0)
                                    all_serials.add(serial)

                                    # Track affected segments/cameras
                                    if segment_id not in self.affected_segments:
                                        self.affected_segments[segment_id] = set()
                                    self.affected_segments[segment_id].add(
                                        str(cam_serial)
                                    )

                logger.debug(
                    "Processed %s: found %d frames in time range",
                    json_path.name,
                    sum(
                        1 for rt in real_times if self.time_start <= rt <= self.time_end
                    ),
                )

            except Exception as e:
                logger.error("Error processing %s: %s", json_path.name, e)
                continue

        if not all_serials:
            raise TimeRangeFilterError(
                f"No frames found in time range {self.time_start} to {self.time_end}. "
                f"Check that your time range overlaps with the video recordings."
            )

        serial_min = min(all_serials)
        serial_max = max(all_serials)

        logger.info(
            "Found serial range [%d, %d] covering %d unique serials across %d segment(s)",
            serial_min,
            serial_max,
            len(all_serials),
            len(self.affected_segments),
        )

        for seg_id, cams in sorted(self.affected_segments.items()):
            logger.info("  - %s: cameras %s", seg_id, ", ".join(sorted(cams)))

        return serial_min, serial_max

    def _filter_csv(self, serial_min: int, serial_max: int) -> Path:
        """
        Filter the serial CSV to only include rows within [serial_min, serial_max].

        Returns: Path to filtered CSV
        """
        if not self.serial_csv.exists():
            raise TimeRangeFilterError(f"Serial CSV not found: {self.serial_csv}")

        # Read input CSV
        rows = []
        with self.serial_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    serial = int(row.get("serial", "").strip())
                    if serial_min <= serial <= serial_max:
                        rows.append(row)
                except (ValueError, AttributeError):
                    continue

        if not rows:
            raise TimeRangeFilterError(
                f"No CSV rows found with serials in range [{serial_min}, {serial_max}]. "
                f"Input CSV: {self.serial_csv}"
            )

        # Write filtered CSV with time range in filename
        # Format times as YYYYMMDD_HHMMSS for filename-safe identifiers
        start_str = self.time_start.strftime("%Y%m%d_%H%M%S")
        end_str = self.time_end.strftime("%Y%m%d_%H%M%S")
        output_csv = (
            self.serial_csv.parent
            / f"{self.serial_csv.stem}-timerange-{start_str}-{end_str}.csv"
        )
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(
            "Filtered CSV: %d rows → %d rows (serials [%d, %d])",
            sum(1 for _ in self.serial_csv.open()),
            len(rows) + 1,  # +1 for header
            serial_min,
            serial_max,
        )
        logger.info("Filtered CSV saved: %s", _name(output_csv))

        return output_csv

    def run(self) -> Tuple[Path, Dict[str, List[str]]]:
        """
        Execute the time range filter.

        Returns
        -------
        filtered_csv : Path
            Path to the filtered serial CSV
        affected_segments : Dict[str, List[str]]
            Dictionary mapping segment_id → list of camera serials
        """
        # Find serial range from video JSONs
        serial_min, serial_max = self._find_serial_range_from_videos()
        self.serial_min = serial_min
        self.serial_max = serial_max

        # Filter CSV
        filtered_csv = self._filter_csv(serial_min, serial_max)

        # Convert affected_segments sets to lists
        affected = {
            seg: sorted(list(cams)) for seg, cams in self.affected_segments.items()
        }

        return filtered_csv, affected


# ---- Public API ----


def filter_by_time_range(
    serial_csv: Path,
    video_dir: Path,
    time_start: str,
    time_end: str,
    user_timezone: str = "UTC",
) -> Tuple[Path, Dict[str, List[str]]]:
    """
    Filter serial CSV to a specific wall-clock time range.

    Parameters
    ----------
    serial_csv : Path
        Path to the input serial CSV (with columns: serial, start_sample, end_sample)
    video_dir : Path
        Directory containing video JSON files with real_times data
    time_start : str
        Start time string in user's timezone (format: "YYYY-MM-DD HH:MM:SS")
    time_end : str
        End time string in user's timezone (format: "YYYY-MM-DD HH:MM:SS")
    user_timezone : str
        IANA timezone name (e.g., 'America/Chicago', 'US/Central', 'UTC'). Default: 'UTC'

    Returns
    -------
    filtered_csv : Path
        Path to the filtered CSV file
    affected_segments : Dict[str, List[str]]
        Dictionary mapping segment_id → list of camera serials that have data in the time range

    Raises
    ------
    TimeRangeFilterError
        If parsing fails, no data found in range, or other errors occur

    Examples
    --------
    >>> # Using local timezone (Central Time)
    >>> filtered_csv, segments = filter_by_time_range(
    ...     Path("output/audio_decoded/raw-gapfilled-filtered.csv"),
    ...     Path("input/video"),
    ...     "2025-11-09 14:30:00",  # 2:30 PM Central
    ...     "2025-11-09 15:00:00",  # 3:00 PM Central
    ...     user_timezone="America/Chicago"
    ... )
    >>> print(f"Filtered CSV: {filtered_csv}")
    >>> print(f"Affected segments: {segments}")
    """
    filter_obj = TimeRangeFilter(
        serial_csv, video_dir, time_start, time_end, user_timezone
    )
    return filter_obj.run()


def _discover_affected_segments_for_serial_range(
    video_dir: Path,
    *,
    serial_min: int,
    serial_max: int,
) -> Dict[str, Set[str]]:
    """
    Return {segment_id -> {cam_serial, ...}} for cameras whose fixed serials
    overlap [serial_min, serial_max] (inclusive).
    """
    affected: Dict[str, Set[str]] = {}
    json_files = sorted(video_dir.glob("*.json"))
    if not json_files:
        raise AudioSampleRangeFilterError(f"No JSON files found in {video_dir}")

    vd = VideoDiscoverer(video_dir, log=logger)
    for json_path in json_files:
        segment_id = json_path.stem
        try:
            _, _, cam_jsons = vd._build_json_wrapper(json_path, None)
            if not cam_jsons:
                logger.warning("No camera data in %s, skipping", json_path.name)
                continue

            for cam_serial, cam_json in cam_jsons.items():
                fixed_serials = cam_json.fixed_serials or []
                if not fixed_serials:
                    continue

                has_overlap = False
                for value in fixed_serials:
                    try:
                        serial = int(value)
                    except (TypeError, ValueError):
                        continue
                    if serial_min <= serial <= serial_max:
                        has_overlap = True
                        break

                if has_overlap:
                    affected.setdefault(segment_id, set()).add(str(cam_serial))
        except Exception as e:
            logger.error("Error processing %s: %s", json_path.name, e)
            continue

    return affected


class AudioSampleRangeFilter:
    """
    Filter serial CSV by audio sample bounds and discover affected segment/cameras.
    """

    def __init__(
        self,
        serial_csv: Path,
        video_dir: Path,
        audio_sample_start: int,
        audio_sample_end: int,
    ):
        self.serial_csv = Path(serial_csv)
        self.video_dir = Path(video_dir)
        self.audio_sample_start = int(audio_sample_start)
        self.audio_sample_end = int(audio_sample_end)

        if self.audio_sample_start < 0 or self.audio_sample_end < 0:
            raise AudioSampleRangeFilterError(
                "Audio sample range must be non-negative."
            )
        if self.audio_sample_end < self.audio_sample_start:
            raise AudioSampleRangeFilterError(
                f"audio_sample_end ({self.audio_sample_end}) must be >= "
                f"audio_sample_start ({self.audio_sample_start})."
            )

    def _filter_csv_by_sample_range(self) -> Tuple[Path, int, int]:
        """
        Keep rows whose [start_sample, end_sample] overlaps the query range.
        Returns (filtered_csv_path, serial_min, serial_max).
        """
        if not self.serial_csv.exists():
            raise AudioSampleRangeFilterError(
                f"Serial CSV not found: {self.serial_csv}"
            )

        rows: List[Dict[str, str]] = []
        serials: List[int] = []
        total_rows = 0

        with self.serial_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                try:
                    serial = int(str(row.get("serial", "")).strip())
                    start_sample = int(str(row.get("start_sample", "")).strip())
                    end_sample = int(str(row.get("end_sample", "")).strip())
                except (TypeError, ValueError):
                    continue

                if end_sample < self.audio_sample_start:
                    continue
                if start_sample > self.audio_sample_end:
                    continue

                rows.append(row)
                serials.append(serial)

        if not rows or not serials:
            raise AudioSampleRangeFilterError(
                f"No CSV rows overlap audio sample range "
                f"[{self.audio_sample_start}, {self.audio_sample_end}]. "
                f"Input CSV: {self.serial_csv}"
            )

        serial_min = min(serials)
        serial_max = max(serials)
        output_csv = self.serial_csv.parent / (
            f"{self.serial_csv.stem}-audiosamplerange-"
            f"{self.audio_sample_start}-{self.audio_sample_end}.csv"
        )

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "Audio-sample filter: %d rows → %d rows (samples [%d, %d], serials [%d, %d])",
            total_rows,
            len(rows),
            self.audio_sample_start,
            self.audio_sample_end,
            serial_min,
            serial_max,
        )
        logger.info("Filtered CSV saved: %s", _name(output_csv))

        return output_csv, serial_min, serial_max

    def run(self) -> Tuple[Path, Dict[str, List[str]]]:
        filtered_csv, serial_min, serial_max = self._filter_csv_by_sample_range()

        affected_sets = _discover_affected_segments_for_serial_range(
            self.video_dir,
            serial_min=serial_min,
            serial_max=serial_max,
        )
        if not affected_sets:
            raise AudioSampleRangeFilterError(
                "No segment/camera overlap found for filtered serial range "
                f"[{serial_min}, {serial_max}] in {self.video_dir}"
            )

        logger.info(
            "Affected serial range [%d, %d] overlaps %d segment(s)",
            serial_min,
            serial_max,
            len(affected_sets),
        )
        for seg_id, cams in sorted(affected_sets.items()):
            logger.info("  - %s: cameras %s", seg_id, ", ".join(sorted(cams)))

        affected = {seg: sorted(cams) for seg, cams in affected_sets.items()}
        return filtered_csv, affected


def filter_by_audio_sample_range(
    serial_csv: Path,
    video_dir: Path,
    audio_sample_start: int,
    audio_sample_end: int,
) -> Tuple[Path, Dict[str, List[str]]]:
    """
    Filter serial CSV by an audio sample range and return affected segment/camera map.
    """
    filter_obj = AudioSampleRangeFilter(
        serial_csv=serial_csv,
        video_dir=video_dir,
        audio_sample_start=audio_sample_start,
        audio_sample_end=audio_sample_end,
    )
    return filter_obj.run()
