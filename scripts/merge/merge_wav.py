from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


WAV_NAME_PATTERN = re.compile(
    r"^(?P<channel>\d+)[-_](?P<date>\d{6})[-_](?P<time>\d{4,6})\.wav$", re.IGNORECASE
)


@dataclass(frozen=True)
class WavFileInfo:
    channel: str
    timestamp: datetime
    path: Path


def parse_wav_filename(path: Path) -> WavFileInfo | None:
    match = WAV_NAME_PATTERN.match(path.name)
    if not match:
        return None

    time_token = match.group("time")
    try:
        timestamp = datetime.strptime(
            f"{match.group('date')}-{time_token}",
            "%y%m%d-%H%M" if len(time_token) == 4 else "%y%m%d-%H%M%S",
        )
    except ValueError as exc:
        raise ValueError(f"Failed to parse timestamp from {path.name}") from exc

    return WavFileInfo(match.group("channel"), timestamp, path)


def discover_wavs(directory: Path) -> list[WavFileInfo]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Input directory {directory} does not exist or is not a directory."
        )

    wavs: list[WavFileInfo] = []
    for candidate in directory.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() != ".wav":
            continue
        info = parse_wav_filename(candidate)
        if info:
            wavs.append(info)
        else:
            logging.debug("Skipping file with unexpected name format: %s", candidate)

    return wavs


def group_by_channel(wavs: Iterable[WavFileInfo]) -> dict[str, list[WavFileInfo]]:
    grouped: dict[str, list[WavFileInfo]] = defaultdict(list)
    for info in wavs:
        grouped[info.channel].append(info)
    for files in grouped.values():
        files.sort(key=lambda info: info.timestamp)
    return grouped


def _require_ffmpeg() -> str:
    """Return the ffmpeg binary path, or raise if not found."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise FileNotFoundError(
            "ffmpeg is required for merging WAV segments but was not found on PATH."
        )
    return path


def merge_channel_wavs(
    channel: str, files: Iterable[WavFileInfo], output_dir: Path
) -> Path:
    """Concatenate per-channel WAV segments into a single file using ffmpeg.

    Uses ffmpeg's concat demuxer with ``-rf64 auto`` so that files exceeding
    the 4 GiB RIFF limit are automatically promoted to RF64 format.  Files
    under the limit remain standard WAV.  The output keeps a ``.wav``
    extension in both cases — downstream tools (ffmpeg, soundfile, libsndfile)
    read RF64 transparently.
    """
    ordered_files = list(files)
    if not ordered_files:
        raise ValueError(f"No files provided for channel {channel}.")

    ffmpeg = _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"merged-{channel}.wav"

    # Build ffmpeg concat list in a temp file.
    concat_fd, concat_path = tempfile.mkstemp(suffix=".txt", prefix="ffconcat_")
    try:
        with open(concat_fd, "w", encoding="utf-8") as f:
            for info in ordered_files:
                # Escape single quotes in paths for ffmpeg concat format.
                escaped = str(info.path.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_path,
            "-c", "copy",
            "-rf64", "auto",
            str(output_path),
        ]
        logging.debug("ffmpeg merge cmd: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed for channel {channel} "
                f"(rc={proc.returncode}):\n{proc.stderr}"
            )
    finally:
        Path(concat_path).unlink(missing_ok=True)

    logging.info(
        "Merged %d file(s) for channel %s into %s",
        len(ordered_files),
        channel,
        output_path,
    )

    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "channel": channel,
        "merged_file": output_path.name,
        "source_files": [str(info.path) for info in ordered_files],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logging.debug("Wrote metadata for channel %s to %s", channel, metadata_path)

    return output_path


def run_merge(input_dir: Path, output_dir: Path) -> list[Path]:
    wav_infos = discover_wavs(input_dir)
    if not wav_infos:
        logging.warning(
            "No WAV files found in %s matching expected pattern.", input_dir
        )
        return []

    grouped = group_by_channel(wav_infos)
    merged_paths: list[Path] = []
    for channel, files in grouped.items():
        merged_paths.append(merge_channel_wavs(channel, files, output_dir))

    return merged_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge chronological WAV files per channel. "
            "Expects filenames like 01-YYMMDD-HHMM.wav."
        )
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing the source WAV files."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write merged WAV files."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (can be specified multiple times).",
    )
    return parser


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(args.verbose)
    merged = run_merge(args.input_dir, args.output_dir)
    if not merged:
        logging.warning("No merged files were produced.")
        return 1

    for path in merged:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
