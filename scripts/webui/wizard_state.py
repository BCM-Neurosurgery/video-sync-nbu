from __future__ import annotations

from typing import Any, Dict


def clear_decode_state(data: Dict[str, Any]) -> None:
    data.pop("decode_choice", None)
    data.pop("decode_choice_out_dir", None)
    data.pop("decode_error", None)
    data.pop("decode_log_path", None)
    data["skip_decode"] = False


def mark_decode_ready(data: Dict[str, Any], *, choice: str) -> None:
    out_dir = str(data.get("out_dir", "")).strip()
    data["decode_choice"] = choice
    data["decode_choice_out_dir"] = out_dir
    data["decode_error"] = None
    data["skip_decode"] = True
    data["split"] = False
    data["split_overwrite"] = False
    data["split_clean"] = False


def is_decode_ready(data: Dict[str, Any]) -> bool:
    if str(data.get("mode") or "sync") == "audio_timestamp":
        return True
    out_dir = str(data.get("out_dir", "")).strip()
    if not out_dir:
        return False
    if str(data.get("decode_choice_out_dir") or "").strip() != out_dir:
        return False
    choice = str(data.get("decode_choice") or "").strip().lower()
    if choice not in {"reuse", "rebuild"}:
        return False
    out_result = data.get("out_result") or {}
    audio_decoded = out_result.get("audio_decoded") or {}
    return bool(audio_decoded.get("raw_csv") and audio_decoded.get("filtered_csv"))


def wizard_flow_context(data: Dict[str, Any], *, step: int) -> Dict[str, Any]:
    mode = str(data.get("mode") or "sync")
    if mode == "audio_timestamp":
        flow_steps = ["Audio", "Video", "Output", "Select", "Summary"]
        subtitles = {
            1: "Step 1/5 - Select audio folder",
            2: "Step 2/5 - Select video folder",
            3: "Step 3/5 - Select output folder",
            4: "Step 4/5 - Configure output",
            5: "Step 5/5 - Summary",
        }
        return {
            "mode": mode,
            "base_layout": "layout_timestamp.html",
            "flow_title": "Find audio timestamp",
            "flow_subtitle": subtitles.get(step),
            "flow_nav": "timestamp_new",
            "cancel_url": "/",
            "flow_steps": flow_steps,
            "select_hint": (
                "Select segment-camera pairs to use as reference anchors. "
                "-- = missing video. Click a segment or camera header to toggle its row/column."
            ),
        }

    subtitles = {
        1: "Step 1/6 - Select audio folder",
        2: "Step 2/6 - Select video folder",
        3: "Step 3/6 - Select output folder",
        4: "Step 4/6 - Decode audio artifacts",
        5: "Step 5/6 - Select sync targets",
        6: "Step 6/6 - Summary",
    }
    return {
        "mode": mode,
        "base_layout": "layout_nbu.html",
        "flow_title": "New run",
        "flow_subtitle": subtitles.get(step),
        "flow_nav": "new",
        "cancel_url": "/runs",
        "flow_steps": ["Audio", "Video", "Output", "Decode", "Select", "Summary"],
        "select_hint": None,
    }
