#!/usr/bin/env python3
"""
video_analysis.py — Generate a simple text report for a video.

Given an MP4 path and an output directory, this script parses the file with
`VideoFileParser` (ffprobe-backed) and writes a human-readable .txt report
containing: duration (s), FPS, resolution, and total frame count.

Usage
-----
python video_analysis.py /path/to/video.mp4 --outdir /path/to/output [--log-level INFO]

Output
------
<outdir>/<video-stem>.txt
    File        : video.mp4
    Duration(s) : 123.456
    FPS         : 29.970030
    Resolution  : 1920x1080
    Frames      : 37000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

from scripts.parsers.videofileparser import VideoFileParser
from scripts.log.logutils import configure_standalone_logging, log_context

logger = logging.getLogger(__name__)


def _format_report(
    video_path: Path,
    duration: float,
    fps: float,
    resolution: Tuple[int, int],
    frames: int,
) -> str:
    """Return a human-readable, line-based report string."""
    w, h = resolution
    return (
        f"File        : {video_path.name}\n"
        f"Duration(s) : {duration:.3f}\n"
        f"FPS         : {fps:.6f}\n"
        f"Resolution  : {w}x{h}\n"
        f"Frames      : {frames}\n"
    )


def analyze_and_write(video_path: Path, outdir: Path) -> Path:
    """Parse `video_path` and write a text report into `outdir`. Returns the report path."""
    if not video_path.is_file():
        raise FileNotFoundError(f"No such file: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an .mp4 file, got: {video_path.suffix}")

    logger.info("Probing video with ffprobe: %s", video_path.name)
    parser = VideoFileParser(str(video_path))

    report = _format_report(
        video_path=video_path,
        duration=parser.duration,
        fps=parser.fps,
        resolution=parser.resolution,
        frames=parser.frame_count,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / f"{video_path.stem}.txt"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Wrote report → %s", report_path.name)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze a video (MP4) and save a text report with FPS, duration, resolution, and frame count."
    )
    p.add_argument("video", type=Path, help="Path to input .mp4 video")
    p.add_argument(
        "--outdir", required=True, type=Path, help="Directory to write the .txt report"
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (standalone only; ignored when called from driver)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Standalone logging: only attaches a console handler if root has none.
    root = logging.getLogger()
    was_handlerless = not root.handlers
    configure_standalone_logging(args.log_level, seg="-", cam="-")

    try:
        # Stamp a helpful seg/cam only in standalone so we don't override driver context.
        if was_handlerless:
            with log_context(seg=args.video.stem, cam="-"):
                analyze_and_write(args.video, args.outdir)
        else:
            analyze_and_write(args.video, args.outdir)
        return 0
    except Exception as e:
        logger.error("%s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
