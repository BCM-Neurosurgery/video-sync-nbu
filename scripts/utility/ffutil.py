"""FFmpeg utility helpers used by sync and padding code."""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def detect_h264_encoder() -> str:
    """Detect the best available H.264 encoder.

    Priority: h264_nvenc (NVIDIA) > h264_videotoolbox (Apple) > libx264 (CPU).
    The 256x256 probe is intentional: NVIDIA L40 rejects smaller dimensions.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return "libx264"

    for encoder in ("h264_nvenc", "h264_videotoolbox"):
        if encoder not in out:
            continue
        probe = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "nullsrc=s=256x256:d=0.1",
                "-c:v",
                encoder,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0:
            log.info("Using H.264 encoder: %s", encoder)
            return encoder
        log.debug("Encoder %s compiled in but not usable: %s", encoder, probe.stderr)

    log.info("No hardware H.264 encoder found, using libx264")
    return "libx264"


def h264_encode_args(*, crf: int = 18, preset: str = "veryfast") -> list[str]:
    """Return ffmpeg H.264 encoding args for the best available encoder."""
    encoder = detect_h264_encoder()
    if encoder == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p6",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(crf),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "h264_videotoolbox":
        return [
            "-c:v",
            "h264_videotoolbox",
            "-q:v",
            "70",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
