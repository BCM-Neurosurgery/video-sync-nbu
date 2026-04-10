"""Shared FFmpeg hardware-acceleration helpers.

Provides cached detection of the best available H.264 encoder and
corresponding hardware-accelerated decode flags.  Used by sync.py,
videoplanapplier.py, and cli_emu_time.py so every ffmpeg call site
picks up GPU acceleration automatically.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_hw_encoder: Optional[str] = None  # cached after first probe


def detect_hw_encoder() -> str:
    """Detect the best available H.264 encoder. Cached after first call.

    Priority: h264_nvenc (NVIDIA) > h264_videotoolbox (Apple) > libx264 (CPU).
    """
    global _hw_encoder
    if _hw_encoder is not None:
        return _hw_encoder

    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        _hw_encoder = "libx264"
        return _hw_encoder

    for encoder in ("h264_nvenc", "h264_videotoolbox"):
        if encoder in out:
            _hw_encoder = encoder
            logger.info("Using hardware encoder: %s", encoder)
            return _hw_encoder

    _hw_encoder = "libx264"
    logger.info("No hardware encoder found, using libx264 (CPU)")
    return _hw_encoder


def h264_encode_args() -> list[str]:
    """Return ffmpeg args for H.264 encoding using the best available encoder."""
    encoder = detect_hw_encoder()
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
            "19",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    elif encoder == "h264_videotoolbox":
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
    else:
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]


def hwaccel_decode_args() -> list[str]:
    """Return ffmpeg args to enable hardware-accelerated H.264 decoding.

    Frames are automatically transferred to CPU memory so downstream
    video filters (trim, setpts, etc.) work without modification.
    """
    encoder = detect_hw_encoder()
    if encoder == "h264_nvenc":
        return ["-hwaccel", "cuda"]
    elif encoder == "h264_videotoolbox":
        return ["-hwaccel", "videotoolbox"]
    return []
