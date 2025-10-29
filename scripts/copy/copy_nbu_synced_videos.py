#!/usr/bin/env python3
"""
Copy NBU synced video clips that match the expected naming convention into a target directory.

Expected filename format:
    <patient>_<YYYYMMDD>_<HHMMSS>.serial<8 digits>_synced.mp4
Example:
    AA004_20250904_113323.serial24253445_synced.mp4
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Iterable, List


FILENAME_PATTERN = re.compile(
    r"""
    ^
    (?P<patient>[A-Za-z0-9]+)_
    (?P<date>\d{8})_
    (?P<time>\d{6})
    \.serial
    (?P<serial>\d{8})
    _synced\.mp4
    $
    """,
    re.VERBOSE,
)


def files_match(src: Path, dest: Path) -> bool:
    """Return True when `src` and `dest` share size and modification time metadata."""
    try:
        src_stat = src.stat()
        dest_stat = dest.stat()
    except FileNotFoundError:
        return False
    return (
        src_stat.st_size == dest_stat.st_size
        and src_stat.st_mtime_ns == dest_stat.st_mtime_ns
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy NBU synced video clips matching the naming convention to another directory."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory to search recursively for synced video clips.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory where matched clips will be copied.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be copied without performing the copy.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def discover_clips(root: Path) -> Iterable[Path]:
    """Yield files under `root` that match the naming convention."""
    for path in root.rglob("*.mp4"):
        if FILENAME_PATTERN.match(path.name):
            yield path


def copy_clips(
    clips: Iterable[Path], destination: Path, dry_run: bool = False
) -> List[Path]:
    """Copy provided clip paths into `destination`. Return list of copied file paths."""
    copied: List[Path] = []
    for clip in clips:
        target = destination / clip.name
        if target.exists() and files_match(clip, target):
            if dry_run:
                logging.info(
                    "[DRY RUN] Skipping %s (already present with matching size and mtime)",
                    target,
                )
            else:
                logging.info(
                    "Skipping copy for %s (already present with matching size and mtime)",
                    target,
                )
            continue
        if target.exists():
            logging.info("Overwriting existing file %s", target)
        if dry_run:
            logging.info("[DRY RUN] Would copy %s -> %s", clip, target)
            copied.append(target)
            continue
        logging.info("Copying %s -> %s", clip, target)
        shutil.copy2(clip, target)
        copied.append(target)
    return copied


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    clips = list(discover_clips(input_dir))
    if not clips:
        logging.warning("No matching clips found under %s", input_dir)
        return

    copy_clips(clips, output_dir, dry_run=args.dry_run)
    logging.info("Finished processing %d clips.", len(clips))


if __name__ == "__main__":
    main()
