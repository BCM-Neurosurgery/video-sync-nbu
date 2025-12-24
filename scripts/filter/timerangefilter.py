#!/usr/bin/env python3
"""
timerangefilter.py — Filter serial CSV and video segments to a user-specified time range

What it does
------------
Given a time range (e.g., "14:30:00" to "15:00:00"), this module:
1. Queries video JSON files to find frames within that time range
2. Extracts the serial numbers associated with those frames
3. Filters the serial CSV to only include those serials
4. Returns both the filtered CSV and the affected segment/camera combinations

Time Reference
--------------
Uses time from video JSON `real_times[]` array, which contains actual
timestamps in format: "YYYY-MM-DD HH:MM:SS.ffffff"

User must specify times in format: "YYYY-MM-DD HH:MM:SS"

----------
filter_by_time_range(
    serial_csv: Path,
    video_dir: Path,
    time_start: str,
    time_end: str
) -> Tuple[Path, Dict[str, List[str]]]

Returns: (filtered_csv_path, affected_segments_dict)
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, time, timedelta, timezone
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
        self.affected_segments: Dict[str, Set[str]] = {}  # {segment_id: {cam_serial, ...}}
        
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
    
    def _parse_time_range(self, start_str: str, end_str: str) -> Tuple[datetime, datetime]:
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
            end_time - start_time
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
                        if cam_json and cam_json.fixed_serials and frame_idx < len(cam_json.fixed_serials):
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
                                    self.affected_segments[segment_id].add(str(cam_serial))
                
                logger.debug(
                    "Processed %s: found %d frames in time range",
                    json_path.name,
                    sum(1 for rt in real_times if self.time_start <= rt <= self.time_end)
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
            serial_min, serial_max, len(all_serials), len(self.affected_segments)
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
        output_csv = self.serial_csv.parent / f"{self.serial_csv.stem}-timerange-{start_str}-{end_str}.csv"
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
            serial_max
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
        affected = {seg: sorted(list(cams)) for seg, cams in self.affected_segments.items()}
        
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
    filter_obj = TimeRangeFilter(serial_csv, video_dir, time_start, time_end, user_timezone)
    return filter_obj.run()
