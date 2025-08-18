# videofileparser.py
# Minimal OOP wrapper around ffprobe to get duration (s), FPS, resolution, and frame_count.
# Requires FFmpeg's ffprobe to be installed and available on PATH.

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional, Tuple


class VideoFileParser:
    """
    Parse a video file for duration (seconds), FPS, resolution, and total frame count using ffprobe.

    Usage:
        p = VideoFileParser("path/to/video.mp4")
        print(p.duration)      # float seconds
        print(p.fps)           # float frames per second
        print(p.resolution)    # (width, height)
        print(p.frame_count)   # int total frames
    """

    def __init__(self, path: str) -> None:
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"No such file: {self.path}")
        if shutil.which("ffprobe") is None:
            raise RuntimeError("ffprobe not found on PATH. Please install FFmpeg.")

        self._duration: Optional[float] = None
        self._fps: Optional[float] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._frame_count: Optional[int] = None

        self._probe()

    # --- Public API ---------------------------------------------------------
    @property
    def duration(self) -> float:
        assert self._duration is not None
        return self._duration

    @property
    def fps(self) -> float:
        assert self._fps is not None
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        assert self._width is not None and self._height is not None
        return (self._width, self._height)

    @property
    def frame_count(self) -> int:
        assert self._frame_count is not None
        return self._frame_count

    # --- Internal helpers ---------------------------------------------------
    def _probe(self) -> None:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
            "-print_format",
            "json",
            self.path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not res.stdout.strip():
            raise RuntimeError(f"ffprobe failed to read {self.path}")

        data = json.loads(res.stdout)
        streams = data.get("streams") or []
        if not streams:
            raise RuntimeError("No video stream found.")
        stream = streams[0]
        fmt = data.get("format", {})

        # Resolution
        self._width = int(stream.get("width") or 0)
        self._height = int(stream.get("height") or 0)
        if self._width <= 0 or self._height <= 0:
            raise RuntimeError("Invalid resolution returned by ffprobe.")

        # Duration (prefer container duration)
        duration_str = fmt.get("duration")
        self._duration = float(duration_str) if duration_str not in (None, "") else 0.0
        if self._duration <= 0:
            raise RuntimeError("Invalid duration returned by ffprobe.")

        # FPS: try avg_frame_rate, then r_frame_rate, then nb_frames/duration
        fps = self._parse_rate(stream.get("avg_frame_rate", "0/0"))
        if fps <= 0:
            fps = self._parse_rate(stream.get("r_frame_rate", "0/0"))
        if fps <= 0:
            nb = stream.get("nb_frames")
            try:
                if nb not in (None, "", "N/A"):
                    fps = float(nb) / self._duration
            except Exception:
                fps = 0.0
        if fps <= 0:
            raise RuntimeError("Could not determine FPS from ffprobe output.")
        self._fps = fps

        # Frame count: prefer nb_frames; otherwise compute round(duration * fps)
        nb_frames_raw = stream.get("nb_frames")
        frame_count = 0
        if nb_frames_raw not in (None, "", "N/A"):
            try:
                frame_count = int(nb_frames_raw)
            except Exception:
                frame_count = 0
        if frame_count <= 0:
            raise RuntimeError("Could not determine frame count from ffprobe output.")
        self._frame_count = frame_count

    @staticmethod
    def _parse_rate(rate: Optional[str]) -> float:
        """Convert a rate like '30000/1001' or '30' into a float FPS."""
        if not rate:
            return 0.0
        if "/" in rate:
            num, den = rate.split("/", 1)
            try:
                n = float(num)
                d = float(den)
                return n / d if d != 0 else 0.0
            except Exception:
                return 0.0
        try:
            return float(rate)
        except Exception:
            return 0.0


if __name__ == "__main__":
    # Quick manual test: python videofileparser.py path/to/file.mp4
    import sys

    if len(sys.argv) >= 2:
        p = VideoFileParser(sys.argv[1])
        print("Duration (s):", p.duration)
        print("FPS:", p.fps)
        print("Resolution:", p.resolution)
        print("Frame count:", p.frame_count)
    else:
        print("Usage: python videofileparser.py <video.mp4>")
