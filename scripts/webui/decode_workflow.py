from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from scripts.cli.cli_nbu import prepare_serial_audio
from scripts.errors import (
    AudioDecodingError,
    AudioGroupDiscoverError,
    FilteredError,
    GapFillError,
)
from scripts.index.discover import AudioDiscoverer
from scripts.validate.validate_out_dir import validate_out_dir

from .wizard_state import clear_decode_state, mark_decode_ready


def rebuild_decode_artifacts_for_draft(
    data: Dict[str, Any],
    *,
    site_value: str,
    on_phase: Optional[Callable[[str, str], None]] = None,
) -> Optional[str]:
    def emit(phase: str, message: str) -> None:
        if on_phase is None:
            return
        try:
            on_phase(phase, message)
        except Exception:
            pass

    out_dir = str(data.get("out_dir", "")).strip()
    if not out_dir:
        clear_decode_state(data)
        return "Output dir is required."

    emit("discover", "Discovering audio files...")
    try:
        audio_dir = Path(str(data.get("audio_dir", "")).strip()).expanduser()
        out_dir_path = Path(out_dir).expanduser()
        decode_log = logging.getLogger("webui.decode.audio")
        audiogroup = AudioDiscoverer(
            audio_dir=audio_dir, log=decode_log
        ).get_audio_group()
    except (AudioGroupDiscoverError, ValueError) as exc:
        clear_decode_state(data)
        return f"Audio decode failed: {exc}"
    except Exception as exc:
        clear_decode_state(data)
        return f"Audio decode failed: {exc}"

    emit(
        "decode",
        "Decoding serial audio (split + split-overwrite) and rebuilding artifacts...",
    )
    try:
        split_chunk_seconds = int(data.get("split_chunk_seconds", 3600) or 3600)
        prepare_serial_audio(
            audiogroup=audiogroup,
            artifact_root=out_dir_path,
            site=site_value,
            skip_decode=False,
            do_split=True,
            split_chunk_seconds=split_chunk_seconds,
            split_overwrite=True,
            split_clean=False,
        )
    except (
        AudioDecodingError,
        GapFillError,
        FilteredError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        clear_decode_state(data)
        return f"Audio decode failed: {exc}"
    except Exception as exc:
        clear_decode_state(data)
        return f"Audio decode failed: {exc}"

    emit("finalize", "Refreshing output artifact status...")
    segments = data.get("segments_all")
    if not isinstance(segments, list):
        segments = None
    refreshed = validate_out_dir(out_dir, segments=segments)
    data["out_ok"] = bool(refreshed.get("ok"))
    data["out_result"] = refreshed
    data["out_error"] = refreshed.get("error")
    if not bool(refreshed.get("can_reuse_audio_decoded")):
        clear_decode_state(data)
        return "Decode completed but expected audio artifacts were not found in output."

    mark_decode_ready(data, choice="rebuild")
    data["decode_error"] = None
    return None
