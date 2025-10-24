#!/usr/bin/env python3
"""
Generate an interactive Plotly waveform for a provided audio file.

The script reads the audio samples, optionally downsamples to limit trace size,
and saves an interactive HTML plot with one trace per channel.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import soundfile as sf


MAX_POINTS_DEFAULT = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an interactive Plotly waveform for the supplied audio file."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the audio file (any format readable by soundfile).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the interactive plot HTML file will be written.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=MAX_POINTS_DEFAULT,
        help=(
            "Maximum number of samples to plot per channel (default: "
            f"{MAX_POINTS_DEFAULT:,}). Larger files are stride-downsampled."
        ),
    )
    return parser.parse_args()


def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    """Load audio samples using soundfile, returning (samples, sample_rate)."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    data, sample_rate = sf.read(audio_path, always_2d=True)
    return data, sample_rate


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    """Return indices for stride-based downsampling capped at max_points."""
    if length <= max_points:
        return np.arange(length)
    stride = math.ceil(length / max_points)
    return np.arange(0, length, stride)


def create_waveform_figure(
    samples: np.ndarray,
    sample_rate: int,
    title: str,
    max_points: int,
) -> go.Figure:
    """Create a Plotly Figure containing one waveform trace per audio channel."""
    num_samples, num_channels = samples.shape
    indices = downsample_indices(num_samples, max_points)
    downsampled = samples[indices, :]
    times_seconds = indices / sample_rate

    fig = go.Figure()
    for channel_idx in range(num_channels):
        channel_data = downsampled[:, channel_idx]
        fig.add_trace(
            go.Scattergl(
                x=times_seconds,
                y=channel_data,
                mode="lines",
                name=f"Channel {channel_idx + 1}",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def build_output_path(audio_path: Path, output_dir: Path) -> Path:
    """Derive the output HTML path from the audio file name."""
    stem = audio_path.stem
    return output_dir / f"{stem}_plot.html"


def main() -> None:
    args = parse_args()
    audio_path: Path = args.audio_path.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, sample_rate = load_audio(audio_path)
    title = f"Waveform: {audio_path.name} ({sample_rate} Hz)"
    figure = create_waveform_figure(
        samples=samples,
        sample_rate=sample_rate,
        title=title,
        max_points=args.max_points,
    )
    output_path = build_output_path(audio_path, output_dir)
    figure.write_html(output_path)
    print(f"Saved interactive waveform to {output_path}")


if __name__ == "__main__":
    main()
