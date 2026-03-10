"""
Per-site configuration for the A/V sync pipeline.

Single source of truth for site-specific settings (serial channel,
decoder presets, etc.). All other modules should import from here
rather than hardcoding site-specific values.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Site definitions
# ---------------------------------------------------------------------------
# Each site maps to:
#   serial_channel : int — which audio channel carries serial data
#   block_preset   : dict — decoder parameters (from lab MATLAB calibration)

SITE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "jamail": {
        "serial_channel": 3,
        "block_preset": {
            "flip_signal": True,
            "flip_window": True,
            "window_samples": 231,
            "block_stride": 1100,
            "transition_points_1b": [6, 53, 100, 147, 194],
            "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],
        },
    },
    "nbu_sleep": {
        "serial_channel": 3,
        "block_preset": {
            "flip_signal": True,
            "flip_window": True,
            "window_samples": 231,
            "block_stride": 1100,
            "transition_points_1b": [6, 53, 100, 147, 194],
            "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],
        },
    },
    "nbu_lounge": {
        "serial_channel": 5,
        "block_preset": {
            "flip_signal": True,
            "flip_window": True,
            "window_samples": 231,
            "block_stride": 1100,
            "transition_points_1b": [6, 53, 100, 147, 194],
            "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],
        },
    },
}

SITE_CHOICES = tuple(SITE_CONFIGS.keys())


def get_serial_channel(site: str) -> int:
    """Return the serial channel number for the given site."""
    return SITE_CONFIGS[site]["serial_channel"]


def get_block_preset(site: str) -> Dict[str, Any]:
    """Return the block decoder preset for the given site."""
    return SITE_CONFIGS[site]["block_preset"]
