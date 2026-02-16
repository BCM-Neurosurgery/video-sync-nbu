from __future__ import annotations

from typing import Any, Dict, Iterable

from scripts.webui.wizard_state import clear_decode_state


_AUDIO_CHANGE_KEYS = (
    "video_dir",
    "segments_all",
    "cameras_all",
    "segments",
    "cameras",
    "target_pairs",
    "manual_target_pairs",
    "range_config",
    "out_dir",
    "out_ok",
    "out_result",
    "out_error",
)

_VIDEO_CHANGE_KEYS = (
    "segments_all",
    "cameras_all",
    "segments",
    "cameras",
    "all_segments",
    "all_cameras",
    "target_pairs",
    "manual_target_pairs",
    "range_config",
    "out_ok",
    "out_result",
    "out_error",
)

_OUT_CHANGE_KEYS = (
    "target_pairs",
    "manual_target_pairs",
    "range_config",
)


def _pop_many(data: Dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        data.pop(key, None)


def invalidate_after_audio_change(
    data: Dict[str, Any], *, include_timestamp_state: bool = False
) -> None:
    _pop_many(data, _AUDIO_CHANGE_KEYS)
    clear_decode_state(data)
    if include_timestamp_state:
        data.pop("timestamp_output_path", None)
        data.pop("timestamp_run_id", None)


def invalidate_after_video_change(
    data: Dict[str, Any], *, include_timestamp_state: bool = False
) -> None:
    _pop_many(data, _VIDEO_CHANGE_KEYS)
    clear_decode_state(data)
    if include_timestamp_state:
        data.pop("timestamp_output_path", None)
        data.pop("timestamp_run_id", None)


def invalidate_after_out_change(
    data: Dict[str, Any], *, include_timestamp_state: bool = False
) -> None:
    _pop_many(data, _OUT_CHANGE_KEYS)
    clear_decode_state(data)
    if include_timestamp_state:
        data.pop("timestamp_output_path", None)
        data.pop("timestamp_run_id", None)
