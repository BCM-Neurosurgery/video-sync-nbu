from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from scripts.validate.validate_audio_dir import (
    validate_audio_dir,
    validate_audio_dir_progress,
)
from scripts.validate.validate_out_dir import (
    validate_out_dir,
    validate_out_dir_progress,
)
from scripts.validate.validate_video_dir import (
    discover_from_video_dir,
    validate_video_dir,
    validate_video_dir_progress,
)
from scripts.webui.services.wizard_service import (
    invalidate_after_audio_change,
    invalidate_after_out_change,
    invalidate_after_video_change,
)

DraftGetter = Callable[[int], Dict[str, Any]]
DraftSetter = Callable[[int, Dict[str, Any]], None]


class ValidationService:
    _AUDIO_CHECK_NAMES = [
        "Find audio files",
        "Naming pattern",
        "Serial channel present",
        "Program channel present",
        "Sample rate detected",
        "Duration detected",
    ]

    _VIDEO_CHECK_NAMES = [
        "Segment JSON present",
        "Camera MP4 present",
        "Segments discovered",
        "Companion JSON per segment",
        "Cameras discovered",
    ]

    _OUT_CHECK_NAMES = [
        "Directory empty",
        "Audio metadata present",
    ]

    def __init__(self, *, draft_get: DraftGetter, draft_set: DraftSetter) -> None:
        self._draft_get = draft_get
        self._draft_set = draft_set

        self._audio_lock = threading.Lock()
        self._audio_state: Dict[int, Dict[str, Any]] = {}

        self._video_lock = threading.Lock()
        self._video_state: Dict[int, Dict[str, Any]] = {}

        self._out_lock = threading.Lock()
        self._out_state: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _copy(payload: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(payload))

    def validate_audio(
        self,
        draft_id: int,
        audio_dir: str,
        *,
        include_timestamp_state: bool = False,
    ) -> Dict[str, Any]:
        data = self._draft_get(draft_id)
        audio_dir = audio_dir.strip()
        if data.get("audio_dir") != audio_dir:
            invalidate_after_audio_change(
                data, include_timestamp_state=include_timestamp_state
            )
        data["audio_dir"] = audio_dir

        result = validate_audio_dir(
            audio_dir, logger=logging.getLogger("webui.validate.audio")
        )
        data["audio_ok"] = bool(result.get("ok"))
        data["audio_result"] = result
        data["audio_error"] = result.get("error")
        self._draft_set(draft_id, data)
        return result

    def start_audio_validation(
        self,
        draft_id: int,
        audio_dir: str,
        *,
        include_timestamp_state: bool = False,
    ) -> Dict[str, Any]:
        data = self._draft_get(draft_id)
        audio_dir = audio_dir.strip()
        if data.get("audio_dir") != audio_dir:
            invalidate_after_audio_change(
                data, include_timestamp_state=include_timestamp_state
            )
        data["audio_dir"] = audio_dir
        self._draft_set(draft_id, data)

        with self._audio_lock:
            st = self._audio_state.get(draft_id)
            if st and st.get("running") and st.get("audio_dir") == audio_dir:
                return st.get("result") or {
                    "audio_dir": audio_dir,
                    "ok": False,
                    "running": True,
                    "files": [],
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in self._AUDIO_CHECK_NAMES
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            self._audio_state[draft_id] = {
                "seq": seq,
                "audio_dir": audio_dir,
                "running": True,
                "result": {
                    "audio_dir": audio_dir,
                    "ok": False,
                    "running": True,
                    "files": [],
                    "nested_files": [],
                    "nested_count": 0,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in self._AUDIO_CHECK_NAMES
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_audio_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                snap = self._copy(payload)
                with self._audio_lock:
                    cur = self._audio_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            result = validate_audio_dir_progress(
                local_audio_dir,
                logger=logging.getLogger("webui.validate.audio"),
                on_progress=on_progress,
            )
            result["running"] = False
            on_progress(result)

            with self._audio_lock:
                cur = self._audio_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = self._copy(result)

            data2 = self._draft_get(draft_id)
            if data2.get("audio_dir") != local_audio_dir:
                return
            data2["audio_ok"] = bool(result.get("ok"))
            data2["audio_result"] = result
            data2["audio_error"] = result.get("error")
            self._draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, audio_dir),
            name=f"webui-audio-validate-{draft_id}",
            daemon=True,
        ).start()

        with self._audio_lock:
            return self._audio_state[draft_id]["result"]

    def audio_validation_status(self, draft_id: int) -> Dict[str, Any]:
        with self._audio_lock:
            st = self._audio_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]

        data = self._draft_get(draft_id)
        if data.get("audio_result"):
            return data["audio_result"]
        return {
            "audio_dir": data.get("audio_dir", ""),
            "ok": False,
            "running": False,
            "files": [],
            "checks": [
                {"name": n, "status": "pending", "message": ""}
                for n in self._AUDIO_CHECK_NAMES
            ],
            "error": None,
        }

    def validate_video(
        self,
        draft_id: int,
        video_dir: str,
        *,
        include_timestamp_state: bool = False,
    ) -> Dict[str, Any]:
        data = self._draft_get(draft_id)
        video_dir = video_dir.strip()
        if data.get("video_dir") != video_dir:
            invalidate_after_video_change(
                data, include_timestamp_state=include_timestamp_state
            )
        data["video_dir"] = video_dir

        result = validate_video_dir(video_dir)
        data["video_ok"] = bool(result.get("ok"))
        data["video_result"] = result
        data["video_error"] = result.get("error")
        if data["video_ok"]:
            try:
                idx = discover_from_video_dir(Path(video_dir).expanduser())
                data["segments_all"] = idx["segments"]
                data["cameras_all"] = idx["cameras"]
            except Exception as exc:
                data["video_ok"] = False
                data["video_error"] = str(exc)
                data["video_result"] = {**result, "ok": False, "error": str(exc)}
        self._draft_set(draft_id, data)
        return data.get("video_result") or result

    def start_video_validation(
        self,
        draft_id: int,
        video_dir: str,
        *,
        include_timestamp_state: bool = False,
    ) -> Dict[str, Any]:
        data = self._draft_get(draft_id)
        video_dir = video_dir.strip()
        if data.get("video_dir") != video_dir:
            invalidate_after_video_change(
                data, include_timestamp_state=include_timestamp_state
            )
        data["video_dir"] = video_dir
        self._draft_set(draft_id, data)

        with self._video_lock:
            st = self._video_state.get(draft_id)
            if st and st.get("running") and st.get("video_dir") == video_dir:
                return st.get("result") or {
                    "video_dir": video_dir,
                    "ok": False,
                    "running": True,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in self._VIDEO_CHECK_NAMES
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            self._video_state[draft_id] = {
                "seq": seq,
                "video_dir": video_dir,
                "running": True,
                "result": {
                    "video_dir": video_dir,
                    "ok": False,
                    "running": True,
                    "segments_count": 0,
                    "cameras_count": 0,
                    "segments_first": None,
                    "segments_last": None,
                    "segments_preview": [],
                    "cameras": [],
                    "cameras_preview": [],
                    "json_count": 0,
                    "mp4_count": 0,
                    "nested_files": [],
                    "nested_count": 0,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in self._VIDEO_CHECK_NAMES
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_video_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                snap = self._copy(payload)
                with self._video_lock:
                    cur = self._video_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            result = validate_video_dir_progress(
                local_video_dir, on_progress=on_progress
            )
            result["running"] = False
            on_progress(result)

            with self._video_lock:
                cur = self._video_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = self._copy(result)

            data2 = self._draft_get(draft_id)
            if data2.get("video_dir") != local_video_dir:
                return
            data2["video_ok"] = bool(result.get("ok"))
            data2["video_result"] = result
            data2["video_error"] = result.get("error")
            if data2["video_ok"]:
                try:
                    idx = discover_from_video_dir(Path(local_video_dir).expanduser())
                    data2["segments_all"] = idx["segments"]
                    data2["cameras_all"] = idx["cameras"]
                except Exception as exc:
                    data2["video_ok"] = False
                    data2["video_error"] = str(exc)
                    data2["video_result"] = {
                        **result,
                        "ok": False,
                        "error": str(exc),
                    }
            self._draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, video_dir),
            name=f"webui-video-validate-{draft_id}",
            daemon=True,
        ).start()

        with self._video_lock:
            return self._video_state[draft_id]["result"]

    def video_validation_status(self, draft_id: int) -> Dict[str, Any]:
        with self._video_lock:
            st = self._video_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]

        data = self._draft_get(draft_id)
        if data.get("video_result"):
            return data["video_result"]
        return {
            "video_dir": data.get("video_dir", ""),
            "ok": False,
            "running": False,
            "segments_count": 0,
            "cameras_count": 0,
            "segments_first": None,
            "segments_last": None,
            "segments_preview": [],
            "cameras": [],
            "cameras_preview": [],
            "json_count": 0,
            "mp4_count": 0,
            "nested_files": [],
            "nested_count": 0,
            "checks": [
                {"name": n, "status": "pending", "message": ""}
                for n in self._VIDEO_CHECK_NAMES
            ],
            "error": None,
        }

    def validate_out(
        self,
        draft_id: int,
        out_dir: str,
        *,
        include_timestamp_state: bool = False,
    ) -> Dict[str, Any]:
        data = self._draft_get(draft_id)
        out_dir = out_dir.strip()
        if data.get("out_dir") != out_dir:
            invalidate_after_out_change(
                data, include_timestamp_state=include_timestamp_state
            )
        data["out_dir"] = out_dir

        segments: Optional[List[str]] = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None

        result = validate_out_dir(out_dir, segments=segments)
        data["out_ok"] = bool(result.get("ok"))
        data["out_result"] = result
        data["out_error"] = result.get("error")
        self._draft_set(draft_id, data)
        return result

    def start_out_validation(
        self,
        draft_id: int,
        out_dir: str,
        *,
        include_timestamp_state: bool = False,
    ) -> Dict[str, Any]:
        data = self._draft_get(draft_id)
        out_dir = out_dir.strip()
        if data.get("out_dir") != out_dir:
            invalidate_after_out_change(
                data, include_timestamp_state=include_timestamp_state
            )
        data["out_dir"] = out_dir
        self._draft_set(draft_id, data)

        segments: Optional[List[str]] = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None

        with self._out_lock:
            st = self._out_state.get(draft_id)
            if st and st.get("running") and st.get("out_dir") == out_dir:
                return st.get("result") or {
                    "out_dir": out_dir,
                    "ok": False,
                    "running": True,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in self._OUT_CHECK_NAMES
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            self._out_state[draft_id] = {
                "seq": seq,
                "out_dir": out_dir,
                "running": True,
                "result": {
                    "out_dir": out_dir,
                    "ok": False,
                    "running": True,
                    "exists": False,
                    "will_create": False,
                    "empty": None,
                    "entries_count": 0,
                    "audio_decoded": {
                        "exists": False,
                        "raw_csv": False,
                        "filtered_csv": False,
                    },
                    "can_reuse_audio_decoded": False,
                    "split_decoded_count": 0,
                    "serial_chunks_count": 0,
                    "synced_segments_count": 0,
                    "synced_segments_preview": [],
                    "total_segments": None,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in self._OUT_CHECK_NAMES
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_out_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                snap = self._copy(payload)
                with self._out_lock:
                    cur = self._out_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            result = validate_out_dir_progress(
                local_out_dir, segments=segments, on_progress=on_progress
            )
            result["running"] = False
            on_progress(result)

            with self._out_lock:
                cur = self._out_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = self._copy(result)

            data2 = self._draft_get(draft_id)
            if data2.get("out_dir") != local_out_dir:
                return
            data2["out_ok"] = bool(result.get("ok"))
            data2["out_result"] = result
            data2["out_error"] = result.get("error")
            self._draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, out_dir),
            name=f"webui-out-validate-{draft_id}",
            daemon=True,
        ).start()

        with self._out_lock:
            return self._out_state[draft_id]["result"]

    def out_validation_status(self, draft_id: int) -> Dict[str, Any]:
        with self._out_lock:
            st = self._out_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]

        data = self._draft_get(draft_id)
        if data.get("out_result"):
            return data["out_result"]
        return {
            "out_dir": data.get("out_dir", ""),
            "ok": False,
            "running": False,
            "exists": False,
            "will_create": False,
            "empty": None,
            "entries_count": 0,
            "audio_decoded": {
                "exists": False,
                "raw_csv": False,
                "filtered_csv": False,
            },
            "can_reuse_audio_decoded": False,
            "split_decoded_count": 0,
            "serial_chunks_count": 0,
            "audio_metadata": {
                "exists": False,
                "abs_json": False,
            },
            "synced_segments_count": 0,
            "synced_segments_preview": [],
            "total_segments": None,
            "checks": [
                {"name": n, "status": "pending", "message": ""}
                for n in self._OUT_CHECK_NAMES
            ],
            "error": None,
        }
