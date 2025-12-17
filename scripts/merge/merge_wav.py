from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import wave


WAV_NAME_PATTERN = re.compile(
    r"^(?P<channel>\d+)[-_](?P<date>\d{6})[-_](?P<time>\d{4,6})\.wav$", re.IGNORECASE
)
CHUNK_FRAMES = 64 * 1024


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


def stream_frames(
    reader: wave.Wave_read, writer: wave.Wave_write, chunk_frames: int = CHUNK_FRAMES
) -> None:
    while True:
        data = reader.readframes(chunk_frames)
        if not data:
            break
        writer.writeframes(data)


def merge_channel_wavs(
    channel: str, files: Iterable[WavFileInfo], output_dir: Path
) -> Path:
    ordered_files = list(files)
    if not ordered_files:
        raise ValueError(f"No files provided for channel {channel}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"merged-{channel}.wav"

    params_reference = None
    with wave.open(str(output_path), "wb") as writer:
        for info in ordered_files:
            with wave.open(str(info.path), "rb") as reader:
                params = reader.getparams()
                if params_reference is None:
                    params_reference = params
                    writer.setparams(params)
                else:
                    if reader.getnchannels() != params_reference.nchannels:
                        raise ValueError(
                            f"Channel count mismatch for {info.path}: "
                            f"expected {params_reference.nchannels}, found {reader.getnchannels()}"
                        )
                    if reader.getsampwidth() != params_reference.sampwidth:
                        raise ValueError(
                            f"Sample width mismatch for {info.path}: "
                            f"expected {params_reference.sampwidth}, found {reader.getsampwidth()}"
                        )
                    if reader.getframerate() != params_reference.framerate:
                        raise ValueError(
                            f"Sample rate mismatch for {info.path}: "
                            f"expected {params_reference.framerate}, found {reader.getframerate()}"
                        )
                    if params.comptype != params_reference.comptype:
                        raise ValueError(
                            f"Compression type mismatch for {info.path}: "
                            f"expected {params_reference.comptype}, found {params.comptype}"
                        )

                logging.debug(
                    "Appending %s to channel %s output", info.path.name, channel
                )
                stream_frames(reader, writer)

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
