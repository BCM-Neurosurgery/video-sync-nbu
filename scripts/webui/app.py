from __future__ import annotations

import json
import logging
import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.webui.repositories import run_repo, timestamp_run_repo
from scripts.webui.services.draft_service import (
    create_draft as _draft_create,
    delete_draft as _draft_delete,
    get_draft as _draft_get,
    set_draft as _draft_set,
)
from scripts.webui.services.timestamp_service import (
    create_timestamp_run as _create_timestamp_run,
    resolve_timestamp_output_path as _resolve_timestamp_output_path,
    start_audio_timestamp_job as _start_audio_timestamp_job,
)
from scripts.webui.services.run_service import enqueue_sync_runs, resolve_schedule
from scripts.webui.services.wizard_service import (
    invalidate_after_audio_change,
    invalidate_after_out_change,
    invalidate_after_video_change,
)
from scripts.webui.models import utc_now_iso
from scripts.webui.runner import Runner, RunnerConfig, build_cli_cmd, tail_log_sse
from scripts.webui.decode_workflow import (
    rebuild_decode_artifacts_for_draft as _rebuild_decode_artifacts_for_draft,
)
from scripts.validate.validate_audio_dir import validate_audio_dir
from scripts.validate.validate_audio_dir import validate_audio_dir_progress
from scripts.validate.validate_out_dir import discover_synced_pairs, validate_out_dir
from scripts.validate.validate_video_dir import (
    discover_from_video_dir,
    validate_video_dir,
    validate_video_dir_progress,
)
from scripts.index.common import DEFAULT_TZ
from scripts.webui.wizard_state import (
    clear_decode_state as _clear_decode_state,
    is_decode_ready as _is_decode_ready,
    mark_decode_ready as _mark_decode_ready,
    wizard_flow_context as _wizard_flow_context,
)
from scripts.webui.sync_selection import (
    WEBUI_DEFAULT_TIME_ZONE,
    _build_range_catalog,
    _build_range_config_from_form,
    _build_sync_run_groups,
    _coerce_range_config,
    _compute_effective_target_pairs,
    _extract_available_pair_map,
    _extract_missing_json_map,
    _extract_synced_pair_map,
    _normalize_target_pairs,
    _segment_range_title,
)


ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
try:
    templates.env.globals["static_v"] = int(
        (ROOT / "static" / "app.css").stat().st_mtime
    )
except Exception:
    templates.env.globals["static_v"] = 1


def create_app() -> FastAPI:
    app = FastAPI(title="video-sync-nbu Web UI")
    app.mount(
        "/static",
        StaticFiles(directory=str(ROOT / "static")),
        name="static",
    )

    max_parallel = int(os.environ.get("VSYNC_WEBUI_MAX_PARALLEL", "1"))
    runner = Runner(cfg=RunnerConfig(max_parallel=max_parallel))
    runner.start()

    audio_val_lock = threading.Lock()
    audio_val_state: Dict[int, Dict[str, Any]] = {}
    audio_check_names = [
        "Find audio files",
        "Naming pattern",
        "Serial channel present",
        "Program channel present",
        "Sample rate detected",
        "Duration detected",
    ]

    video_val_lock = threading.Lock()
    video_val_state: Dict[int, Dict[str, Any]] = {}
    video_check_names = [
        "Segment JSON present",
        "Camera MP4 present",
        "Segments discovered",
        "Companion JSON per segment",
        "Cameras discovered",
    ]

    out_val_lock = threading.Lock()
    out_val_state: Dict[int, Dict[str, Any]] = {}
    out_check_names = [
        "Directory empty",
        "Audio metadata present",
    ]
    decode_val_lock = threading.Lock()
    decode_val_state: Dict[int, Dict[str, Any]] = {}

    def _decode_payload(
        *,
        draft_id: int,
        out_dir: str,
        action: str,
        status: str,
        running: bool,
        phase: str,
        message: str,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        log_available: bool = False,
    ) -> Dict[str, Any]:
        return {
            "draft_id": draft_id,
            "out_dir": out_dir,
            "action": action,
            "status": status,
            "running": running,
            "phase": phase,
            "message": message,
            "error": error,
            "started_at": started_at,
            "finished_at": finished_at,
            "redirect_url": (
                f"/wizard/{draft_id}/select" if status == "succeeded" else ""
            ),
            "log_stream_url": f"/api/drafts/{draft_id}/decode/log/stream",
            "log_available": bool(log_available),
        }

    def _decode_status_from_draft(
        draft_id: int, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        out_dir = str(data.get("out_dir", "")).strip()
        log_available = bool(str(data.get("decode_log_path", "")).strip())
        decode_choice = str(data.get("decode_choice") or "").strip().lower()
        if decode_choice not in {"reuse", "rebuild"}:
            decode_choice = "rebuild"
        if _is_decode_ready(data):
            return _decode_payload(
                draft_id=draft_id,
                out_dir=out_dir,
                action=decode_choice,
                status="succeeded",
                running=False,
                phase="done",
                message="Audio artifacts are ready.",
                log_available=log_available,
            )
        decode_error = str(data.get("decode_error") or "").strip()
        if decode_error:
            return _decode_payload(
                draft_id=draft_id,
                out_dir=out_dir,
                action=decode_choice,
                status="failed",
                running=False,
                phase="failed",
                message=decode_error,
                error=decode_error,
                log_available=log_available,
            )
        can_reuse = bool((data.get("out_result") or {}).get("can_reuse_audio_decoded"))
        idle_msg = (
            "Audio artifacts found. Choose reuse or rebuild."
            if can_reuse
            else "No audio artifacts found. Decode is required."
        )
        return _decode_payload(
            draft_id=draft_id,
            out_dir=out_dir,
            action=decode_choice,
            status="idle",
            running=False,
            phase="idle",
            message=idle_msg,
            log_available=log_available,
        )

    def _resolve_decode_log_path_for_draft(
        draft_id: int, data: Dict[str, Any]
    ) -> Optional[Path]:
        out_dir = str(data.get("out_dir", "")).strip()
        log_path_raw = ""
        with decode_val_lock:
            st = decode_val_state.get(draft_id)
            if st and str(st.get("out_dir", "")).strip() == out_dir:
                log_path_raw = str(st.get("log_path") or "").strip()
        if not log_path_raw:
            log_path_raw = str(data.get("decode_log_path") or "").strip()
        if not log_path_raw:
            return None
        return Path(log_path_raw)

    @app.get("/api/video-index")
    def api_video_index(video_dir: str) -> Dict[str, Any]:
        """
        Return discovered segment IDs and camera serials for a given video directory.
        """
        p = Path(video_dir).expanduser()
        try:
            data = discover_from_video_dir(p)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to scan video_dir: {e}"
            )
        return {"video_dir": str(p), **data}

    @app.get("/api/audio-metadata")
    def api_audio_metadata(out_dir: str) -> Dict[str, Any]:
        out_dir = (out_dir or "").strip()
        if not out_dir:
            raise HTTPException(status_code=400, detail="out_dir is required")
        base = Path(out_dir).expanduser()
        path = base / "audio_metadata" / "audio_abs_start.json"
        if not path.exists():
            return {
                "ok": False,
                "path": str(path),
                "error": "audio_abs_start.json not found",
            }
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return {
                "ok": False,
                "path": str(path),
                "error": f"Failed to read JSON: {exc}",
            }
        max_chars = 200000
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
            truncated = True
        return {"ok": True, "path": str(path), "text": text, "truncated": truncated}

    @app.get("/api/pick-dir")
    def api_pick_dir(initial: str = "") -> Dict[str, Any]:
        """
        Open a native directory picker on the machine running the web server.

        Note: this cannot open a picker on a remote client's machine; it opens locally.
        """
        initial = (initial or "").strip()

        # Prefer platform-native pickers where possible (tkinter is flaky on some macOS setups).
        if sys.platform == "darwin":
            script_lines = [
                "try",
                'set promptText to "Select folder"',
            ]
            if initial:
                # If a file path is provided, use its parent directory as the initial folder.
                p = Path(initial).expanduser()
                if p.is_file():
                    p = p.parent
                initial_dir = str(p)
                # Escape quotes for AppleScript.
                initial_dir = initial_dir.replace('"', '\\"')
                script_lines.append(
                    f'set chosen to (choose folder with prompt promptText default location (POSIX file "{initial_dir}"))'
                )
            else:
                script_lines.append(
                    "set chosen to (choose folder with prompt promptText)"
                )
            script_lines += [
                "POSIX path of chosen",
                "on error number -128",
                '""',
                "end try",
            ]
            script = "\n".join(script_lines)
            try:
                r = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                chosen = (r.stdout or "").strip()
                if chosen:
                    return {"path": chosen, "canceled": False}
                return {"path": "", "canceled": True}
            except Exception:
                # Fall back to tkinter.
                pass

        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return {
                "path": "",
                "canceled": False,
                "error": "Directory picker requires tkinter (not available in this Python environment).",
            }

        try:
            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            chosen = filedialog.askdirectory(
                initialdir=initial or str(Path.cwd()),
                title="Select folder",
            )
            root.destroy()
        except Exception:
            try:
                root.destroy()  # type: ignore[name-defined]
            except Exception:
                pass
            return {
                "path": "",
                "canceled": False,
                "error": "Directory picker failed to open.",
            }

        if chosen:
            return {"path": str(chosen), "canceled": False}
        return {"path": "", "canceled": True}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("home.html", {"request": request})

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        items = run_repo.list_runs(limit=200)
        runs_view: List[Dict[str, Any]] = []
        for row in items:
            d = dict(row)
            try:
                args = json.loads(d.get("args_json") or "{}")
            except Exception:
                args = {}
            d["display_title"] = _segment_range_title(args)
            runs_view.append(d)
        return templates.TemplateResponse(
            "runs.html",
            {"request": request, "runs": runs_view},
        )

    @app.get("/audio-timestamp/runs", response_class=HTMLResponse)
    def audio_timestamp_runs(request: Request) -> HTMLResponse:
        rows = timestamp_run_repo.list_timestamp_runs(limit=200)
        runs_view: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            try:
                args = json.loads(d.get("args_json") or "{}")
            except Exception:
                args = {}
            d["site"] = args.get("site", "")
            runs_view.append(d)
        return templates.TemplateResponse(
            "audio_timestamp_runs.html",
            {"request": request, "runs": runs_view},
        )

    @app.get("/api/runs")
    def api_runs() -> List[Dict[str, Any]]:
        rows = run_repo.list_runs_summary(limit=200)
        return [dict(r) for r in rows]

    @app.get("/api/audio-timestamp/runs")
    def api_audio_timestamp_runs() -> List[Dict[str, Any]]:
        rows = timestamp_run_repo.list_timestamp_runs_summary(limit=200)
        return [dict(r) for r in rows]

    @app.post("/runs/clear")
    def runs_clear() -> RedirectResponse:
        """
        Clear run history (DB rows + log files) for all runs that are not running.
        """
        rows = run_repo.list_non_running_runs()
        for r in rows:
            try:
                p = Path(r["log_path"])
                if p.exists():
                    p.unlink()
            except Exception:
                # Best-effort deletion; DB clear still proceeds.
                pass
        run_repo.delete_non_running_runs()
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/new", response_class=HTMLResponse)
    def runs_new() -> RedirectResponse:
        draft_id = _draft_create(mode="sync")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/tools/audio-timestamp")
    def tools_audio_timestamp() -> RedirectResponse:
        draft_id = _draft_create(mode="audio_timestamp")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/tools/audio-timestamp/new")
    def tools_audio_timestamp_new() -> RedirectResponse:
        draft_id = _draft_create(mode="audio_timestamp")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/api/drafts/{draft_id}/validate-audio")
    def api_validate_audio(draft_id: int, audio_dir: str) -> Dict[str, Any]:
        """
        Validate audio_dir and persist the result into the run-creation draft.
        """
        data = _draft_get(draft_id)
        if data.get("audio_dir") != audio_dir.strip():
            # Changing audio dir invalidates downstream choices.
            invalidate_after_audio_change(data)
        data["audio_dir"] = audio_dir.strip()

        result = validate_audio_dir(
            audio_dir, logger=logging.getLogger("webui.validate.audio")
        )
        data["audio_ok"] = bool(result.get("ok"))
        data["audio_result"] = result
        data["audio_error"] = result.get("error")
        _draft_set(draft_id, data)
        return result

    @app.get("/api/drafts/{draft_id}/validate-audio/start")
    def api_validate_audio_start(draft_id: int, audio_dir: str) -> Dict[str, Any]:
        """
        Start async audio validation so the UI can show per-check progress.
        """
        data = _draft_get(draft_id)
        audio_dir = audio_dir.strip()
        if data.get("audio_dir") != audio_dir:
            # Changing audio dir invalidates downstream choices.
            invalidate_after_audio_change(data, include_timestamp_state=True)
        data["audio_dir"] = audio_dir
        _draft_set(draft_id, data)

        with audio_val_lock:
            st = audio_val_state.get(draft_id)
            if st and st.get("running") and st.get("audio_dir") == audio_dir:
                return st.get("result") or {
                    "audio_dir": audio_dir,
                    "ok": False,
                    "running": True,
                    "files": [],
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in audio_check_names
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            audio_val_state[draft_id] = {
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
                        for n in audio_check_names
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_audio_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                # Deep-copy into state to avoid sharing mutable dicts across threads.
                snap = json.loads(json.dumps(payload))
                with audio_val_lock:
                    cur = audio_val_state.get(draft_id)
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

            with audio_val_lock:
                cur = audio_val_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = json.loads(json.dumps(result))

            # Persist the final result to the draft.
            data2 = _draft_get(draft_id)
            if data2.get("audio_dir") != local_audio_dir:
                return
            data2["audio_ok"] = bool(result.get("ok"))
            data2["audio_result"] = result
            data2["audio_error"] = result.get("error")
            _draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, audio_dir),
            name=f"webui-audio-validate-{draft_id}",
            daemon=True,
        ).start()

        with audio_val_lock:
            return audio_val_state[draft_id]["result"]

    @app.get("/api/drafts/{draft_id}/validate-audio/status")
    def api_validate_audio_status(draft_id: int) -> Dict[str, Any]:
        with audio_val_lock:
            st = audio_val_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]
        data = _draft_get(draft_id)
        if data.get("audio_result"):
            return data["audio_result"]
        return {
            "audio_dir": data.get("audio_dir", ""),
            "ok": False,
            "running": False,
            "files": [],
            "checks": [
                {"name": n, "status": "pending", "message": ""}
                for n in audio_check_names
            ],
            "error": None,
        }

    @app.get("/api/drafts/{draft_id}/validate-video")
    def api_validate_video(draft_id: int, video_dir: str) -> Dict[str, Any]:
        """
        Validate video_dir and persist the result into the run-creation draft.
        """
        data = _draft_get(draft_id)
        if data.get("video_dir") != video_dir.strip():
            # Changing video dir invalidates downstream choices.
            invalidate_after_video_change(data)
        data["video_dir"] = video_dir.strip()

        result = validate_video_dir(video_dir)
        data["video_ok"] = bool(result.get("ok"))
        data["video_result"] = result
        data["video_error"] = result.get("error")
        if data["video_ok"]:
            try:
                idx = discover_from_video_dir(Path(video_dir).expanduser())
                data["segments_all"] = idx["segments"]
                data["cameras_all"] = idx["cameras"]
            except Exception as e:
                data["video_ok"] = False
                data["video_error"] = str(e)
                data["video_result"] = {**result, "ok": False, "error": str(e)}
        _draft_set(draft_id, data)
        return data.get("video_result") or result

    @app.get("/api/drafts/{draft_id}/validate-video/start")
    def api_validate_video_start(draft_id: int, video_dir: str) -> Dict[str, Any]:
        """
        Start async video validation so the UI can show per-check progress.
        """
        data = _draft_get(draft_id)
        video_dir = video_dir.strip()
        if data.get("video_dir") != video_dir:
            # Changing video dir invalidates downstream choices.
            invalidate_after_video_change(data)
        data["video_dir"] = video_dir
        _draft_set(draft_id, data)

        with video_val_lock:
            st = video_val_state.get(draft_id)
            if st and st.get("running") and st.get("video_dir") == video_dir:
                return st.get("result") or {
                    "video_dir": video_dir,
                    "ok": False,
                    "running": True,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in video_check_names
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            video_val_state[draft_id] = {
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
                        for n in video_check_names
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_video_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                snap = json.loads(json.dumps(payload))
                with video_val_lock:
                    cur = video_val_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            result = validate_video_dir_progress(
                local_video_dir, on_progress=on_progress
            )
            result["running"] = False
            on_progress(result)

            with video_val_lock:
                cur = video_val_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = json.loads(json.dumps(result))

            # Persist the final result to the draft.
            data2 = _draft_get(draft_id)
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
                except Exception as e:
                    data2["video_ok"] = False
                    data2["video_error"] = str(e)
                    data2["video_result"] = {**result, "ok": False, "error": str(e)}
            _draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, video_dir),
            name=f"webui-video-validate-{draft_id}",
            daemon=True,
        ).start()

        with video_val_lock:
            return video_val_state[draft_id]["result"]

    @app.get("/api/drafts/{draft_id}/validate-video/status")
    def api_validate_video_status(draft_id: int) -> Dict[str, Any]:
        with video_val_lock:
            st = video_val_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]
        data = _draft_get(draft_id)
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
                for n in video_check_names
            ],
            "error": None,
        }

    @app.get("/api/drafts/{draft_id}/validate-out")
    def api_validate_out(draft_id: int, out_dir: str) -> Dict[str, Any]:
        """
        Validate out_dir and persist the result into the run-creation draft.
        """
        data = _draft_get(draft_id)
        out_dir = out_dir.strip()
        if data.get("out_dir") != out_dir:
            invalidate_after_out_change(data)
        data["out_dir"] = out_dir

        segments = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None

        result = validate_out_dir(out_dir, segments=segments)
        data["out_ok"] = bool(result.get("ok"))
        data["out_result"] = result
        data["out_error"] = result.get("error")
        _draft_set(draft_id, data)
        return result

    @app.get("/api/drafts/{draft_id}/validate-out/start")
    def api_validate_out_start(draft_id: int, out_dir: str) -> Dict[str, Any]:
        """
        Start async out_dir validation so the UI can show per-check progress.
        """
        data = _draft_get(draft_id)
        out_dir = out_dir.strip()
        if data.get("out_dir") != out_dir:
            invalidate_after_out_change(data)
        data["out_dir"] = out_dir
        _draft_set(draft_id, data)

        segments = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None

        with out_val_lock:
            st = out_val_state.get(draft_id)
            if st and st.get("running") and st.get("out_dir") == out_dir:
                return st.get("result") or {
                    "out_dir": out_dir,
                    "ok": False,
                    "running": True,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in out_check_names
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            out_val_state[draft_id] = {
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
                        for n in out_check_names
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_out_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                snap = json.loads(json.dumps(payload))
                with out_val_lock:
                    cur = out_val_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            from scripts.validate.validate_out_dir import validate_out_dir_progress

            result = validate_out_dir_progress(
                local_out_dir, segments=segments, on_progress=on_progress
            )
            result["running"] = False
            on_progress(result)

            with out_val_lock:
                cur = out_val_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = json.loads(json.dumps(result))

            data2 = _draft_get(draft_id)
            if data2.get("out_dir") != local_out_dir:
                return
            data2["out_ok"] = bool(result.get("ok"))
            data2["out_result"] = result
            data2["out_error"] = result.get("error")
            _draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, out_dir),
            name=f"webui-out-validate-{draft_id}",
            daemon=True,
        ).start()

        with out_val_lock:
            return out_val_state[draft_id]["result"]

    @app.get("/api/drafts/{draft_id}/validate-out/status")
    def api_validate_out_status(draft_id: int) -> Dict[str, Any]:
        with out_val_lock:
            st = out_val_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]
        data = _draft_get(draft_id)
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
                {"name": n, "status": "pending", "message": ""} for n in out_check_names
            ],
            "error": None,
        }

    @app.post("/api/drafts/{draft_id}/decode/start")
    def api_decode_start(
        draft_id: int,
        decode_action: str = Form("rebuild"),
        site: str = Form("nbu_lounge"),
    ) -> Dict[str, Any]:
        data = _draft_get(draft_id)
        mode = str(data.get("mode") or "sync")
        out_dir = str(data.get("out_dir", "")).strip()
        site_value = (
            str(site or data.get("site") or "nbu_lounge").strip() or "nbu_lounge"
        )

        if mode == "audio_timestamp":
            return _decode_payload(
                draft_id=draft_id,
                out_dir=out_dir,
                action="rebuild",
                status="failed",
                running=False,
                phase="failed",
                message="Decode stage is not used in audio timestamp mode.",
                error="Decode stage is not used in audio timestamp mode.",
            )
        if not bool(data.get("audio_ok", False)):
            return _decode_payload(
                draft_id=draft_id,
                out_dir=out_dir,
                action="rebuild",
                status="failed",
                running=False,
                phase="failed",
                message="Audio input is not valid. Go back to Step 1.",
                error="Audio input is not valid.",
            )
        if not bool(data.get("video_ok", False)):
            return _decode_payload(
                draft_id=draft_id,
                out_dir=out_dir,
                action="rebuild",
                status="failed",
                running=False,
                phase="failed",
                message="Video input is not valid. Go back to Step 2.",
                error="Video input is not valid.",
            )
        if not bool(data.get("out_ok", False)) or not out_dir:
            return _decode_payload(
                draft_id=draft_id,
                out_dir=out_dir,
                action="rebuild",
                status="failed",
                running=False,
                phase="failed",
                message="Output directory is not ready. Go back to Step 3.",
                error="Output directory is not ready.",
            )

        data["site"] = site_value
        out_result = data.get("out_result") or {}
        can_reuse = bool(out_result.get("can_reuse_audio_decoded"))
        action = str(decode_action or "").strip().lower()
        if action not in {"reuse", "rebuild"}:
            action = "reuse" if can_reuse else "rebuild"

        if action == "reuse":
            if can_reuse:
                _mark_decode_ready(data, choice="reuse")
                data["decode_log_path"] = ""
            else:
                _clear_decode_state(data)
                data["decode_error"] = (
                    "No reusable audio artifacts found. Decode audio artifacts to continue."
                )
            _draft_set(draft_id, data)
            payload = _decode_status_from_draft(draft_id, data)
            with decode_val_lock:
                st = decode_val_state.get(draft_id)
                seq = int(st.get("seq", 0)) + 1 if st else 1
                decode_val_state[draft_id] = {
                    "seq": seq,
                    "out_dir": out_dir,
                    "running": False,
                    "result": payload,
                    "log_path": "",
                }
            return payload

        data["decode_error"] = None

        with decode_val_lock:
            st = decode_val_state.get(draft_id)
            if (
                st
                and bool(st.get("running"))
                and str(st.get("out_dir", "")).strip() == out_dir
            ):
                existing = st.get("result")
                if isinstance(existing, dict):
                    return existing

            seq = int(st.get("seq", 0)) + 1 if st else 1
            decode_log_path: Optional[Path] = None
            try:
                logs_dir = Path(".webui") / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                decode_log_path = logs_dir / f"decode-{draft_id}-{seq}.log"
                header = (
                    f"\n==== DECODE DRAFT {draft_id} @ {utc_now_iso()} ====\n"
                    f"OUT: {out_dir}\n"
                    f"SITE: {site_value}\n"
                    f"ACTION: rebuild\n\n"
                )
                decode_log_path.write_text(header, encoding="utf-8")
                data["decode_log_path"] = str(decode_log_path)
            except Exception:
                decode_log_path = None
                data["decode_log_path"] = ""
            _draft_set(draft_id, data)
            started_at = utc_now_iso()
            start_payload = _decode_payload(
                draft_id=draft_id,
                out_dir=out_dir,
                action="rebuild",
                status="running",
                running=True,
                phase="queued",
                message="Decode started...",
                started_at=started_at,
                log_available=decode_log_path is not None,
            )
            decode_val_state[draft_id] = {
                "seq": seq,
                "out_dir": out_dir,
                "running": True,
                "result": start_payload,
                "log_path": str(decode_log_path or ""),
            }

        def run_decode(local_seq: int, expected_out_dir: str, local_site: str) -> None:
            def update(
                *,
                status: Optional[str] = None,
                running: Optional[bool] = None,
                phase: Optional[str] = None,
                message: Optional[str] = None,
                error: Optional[str] = None,
                finished: bool = False,
            ) -> None:
                with decode_val_lock:
                    cur = decode_val_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    res = dict(cur.get("result") or {})
                    if status is not None:
                        res["status"] = status
                    if running is not None:
                        res["running"] = running
                        cur["running"] = running
                    if phase is not None:
                        res["phase"] = phase
                    if message is not None:
                        res["message"] = message
                    if error is not None:
                        res["error"] = error
                    if finished:
                        res["finished_at"] = utc_now_iso()
                    cur["result"] = res

            def persist_if_current(updated: Dict[str, Any]) -> bool:
                latest = _draft_get(draft_id)
                if str(latest.get("out_dir", "")).strip() != expected_out_dir:
                    return False
                keys = [
                    "site",
                    "out_ok",
                    "out_result",
                    "out_error",
                    "decode_choice",
                    "decode_choice_out_dir",
                    "decode_error",
                    "skip_decode",
                    "split",
                    "split_overwrite",
                    "split_clean",
                    "decode_log_path",
                ]
                for key in keys:
                    if key in updated:
                        latest[key] = updated[key]
                    else:
                        latest.pop(key, None)
                _draft_set(draft_id, latest)
                return True

            log_path: Optional[Path] = None
            with decode_val_lock:
                cur = decode_val_state.get(draft_id)
                if cur and int(cur.get("seq", 0)) == local_seq:
                    raw_path = str(cur.get("log_path") or "").strip()
                    if raw_path:
                        log_path = Path(raw_path)
            log_handler: Optional[logging.Handler] = None
            root_logger = logging.getLogger()
            patched_levels: List[Tuple[logging.Logger, int]] = []
            if log_path is not None:
                try:
                    this_thread_name = threading.current_thread().name

                    class _ThreadFilter(logging.Filter):
                        def filter(self, record: logging.LogRecord) -> bool:
                            return record.threadName == this_thread_name

                    log_handler = logging.FileHandler(
                        log_path, mode="a", encoding="utf-8"
                    )
                    log_handler.setLevel(logging.INFO)
                    log_handler.setFormatter(
                        logging.Formatter(
                            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                        )
                    )
                    log_handler.addFilter(_ThreadFilter())
                    for logger_name in ("scripts", "cli", "webui.decode.audio"):
                        lg = logging.getLogger(logger_name)
                        patched_levels.append((lg, lg.level))
                        lg.setLevel(logging.INFO)
                    root_logger.addHandler(log_handler)
                    logging.getLogger("webui.decode.audio").info(
                        "Decode worker started for draft %s", draft_id
                    )
                except Exception:
                    log_handler = None
                    patched_levels = []

            try:
                data2 = _draft_get(draft_id)
                if str(data2.get("out_dir", "")).strip() != expected_out_dir:
                    update(
                        status="failed",
                        running=False,
                        phase="failed",
                        message="Decode canceled because output directory changed.",
                        error="Output directory changed while decode was running.",
                        finished=True,
                    )
                    return

                data2["site"] = local_site
                data2["decode_error"] = None
                _draft_set(draft_id, data2)

                err = _rebuild_decode_artifacts_for_draft(
                    data2,
                    site_value=local_site,
                    on_phase=lambda p, m: (
                        update(
                            status="running",
                            running=True,
                            phase=p,
                            message=m,
                        ),
                        logging.getLogger("webui.decode.audio").info("[%s] %s", p, m),
                    ),
                )
                if err:
                    data2["decode_error"] = err
                    if not persist_if_current(data2):
                        update(
                            status="failed",
                            running=False,
                            phase="failed",
                            message="Decode canceled because output directory changed.",
                            error="Output directory changed while decode was running.",
                            finished=True,
                        )
                        return
                    update(
                        status="failed",
                        running=False,
                        phase="failed",
                        message=err,
                        error=err,
                        finished=True,
                    )
                    return

                logging.getLogger("webui.decode.audio").info(
                    "Decode finished successfully."
                )
                if not persist_if_current(data2):
                    update(
                        status="failed",
                        running=False,
                        phase="failed",
                        message="Decode canceled because output directory changed.",
                        error="Output directory changed while decode was running.",
                        finished=True,
                    )
                    return
                update(
                    status="succeeded",
                    running=False,
                    phase="done",
                    message="Decode finished. Audio artifacts are ready.",
                    error=None,
                    finished=True,
                )
            except Exception as exc:
                try:
                    data3 = _draft_get(draft_id)
                    _clear_decode_state(data3)
                    data3["decode_error"] = f"Audio decode failed: {exc}"
                    _draft_set(draft_id, data3)
                except Exception:
                    pass
                update(
                    status="failed",
                    running=False,
                    phase="failed",
                    message=f"Audio decode failed: {exc}",
                    error=f"Audio decode failed: {exc}",
                    finished=True,
                )
            finally:
                if log_handler is not None:
                    try:
                        root_logger.removeHandler(log_handler)
                    except Exception:
                        pass
                    try:
                        log_handler.close()
                    except Exception:
                        pass
                for lg, prev_level in patched_levels:
                    try:
                        lg.setLevel(prev_level)
                    except Exception:
                        pass

        threading.Thread(
            target=run_decode,
            args=(seq, out_dir, site_value),
            name=f"webui-decode-{draft_id}",
            daemon=True,
        ).start()
        with decode_val_lock:
            return dict(decode_val_state[draft_id].get("result") or start_payload)

    @app.get("/api/drafts/{draft_id}/decode/status")
    def api_decode_status(draft_id: int) -> Dict[str, Any]:
        data = _draft_get(draft_id)
        out_dir = str(data.get("out_dir", "")).strip()
        with decode_val_lock:
            st = decode_val_state.get(draft_id)
            if st:
                state_out_dir = str(st.get("out_dir", "")).strip()
                if state_out_dir == out_dir:
                    payload = st.get("result")
                    if isinstance(payload, dict):
                        return payload
                elif not bool(st.get("running")):
                    decode_val_state.pop(draft_id, None)
        return _decode_status_from_draft(draft_id, data)

    @app.get("/api/drafts/{draft_id}/decode/log")
    def api_decode_log(draft_id: int) -> Dict[str, Any]:
        data = _draft_get(draft_id)
        path = _resolve_decode_log_path_for_draft(draft_id, data)
        if path is None:
            return {"ok": False, "path": "", "error": "No decode log available."}
        if not path.exists():
            return {
                "ok": False,
                "path": str(path),
                "error": "Decode log file not found yet.",
            }
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return {
                "ok": False,
                "path": str(path),
                "error": f"Failed to read log: {exc}",
            }

        max_chars = 300000
        truncated = False
        if len(text) > max_chars:
            text = text[-max_chars:]
            truncated = True
        return {
            "ok": True,
            "path": str(path),
            "text": text,
            "truncated": truncated,
        }

    @app.get("/api/drafts/{draft_id}/decode/log/download")
    def api_decode_log_download(draft_id: int) -> StreamingResponse:
        data = _draft_get(draft_id)
        path = _resolve_decode_log_path_for_draft(draft_id, data)
        if path is None:
            raise HTTPException(status_code=404, detail="Decode log not found")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Decode log not found")

        def _iter() -> Iterable[bytes]:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 64)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _iter(),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
        )

    @app.get("/api/drafts/{draft_id}/decode/log/stream")
    async def api_decode_log_stream(
        draft_id: int, from_end: int = 0
    ) -> StreamingResponse:
        data = _draft_get(draft_id)
        path = _resolve_decode_log_path_for_draft(draft_id, data)
        if path is None:

            async def no_log() -> Iterable[str]:
                yield "event: status\ndata: no_log\n\n"

            return StreamingResponse(
                no_log(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        return StreamingResponse(
            tail_log_sse(path, start_at_end=bool(from_end)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/wizard/{draft_id}/audio", response_class=HTMLResponse)
    def wizard_audio(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        return templates.TemplateResponse(
            "wizard_audio.html",
            {
                "request": request,
                "draft_id": draft_id,
                "audio_dir": data.get("audio_dir", ""),
                "audio_ok": bool(data.get("audio_ok", False)),
                "audio_result": data.get("audio_result"),
                "error": data.get("audio_error"),
                **_wizard_flow_context(data, step=1),
            },
        )

    @app.post("/wizard/{draft_id}/audio")
    def wizard_audio_post(
        draft_id: int, audio_dir: str = Form(...)
    ) -> RedirectResponse:
        audio_dir = audio_dir.strip()
        data = _draft_get(draft_id)
        if data.get("audio_dir") != audio_dir:
            # Changing audio dir invalidates downstream choices.
            invalidate_after_audio_change(data)

        data["audio_dir"] = audio_dir
        # Validate on POST as a no-JS fallback (JS uses /api/drafts/<id>/validate-audio).
        result = validate_audio_dir(
            audio_dir, logger=logging.getLogger("webui.validate.audio")
        )
        data["audio_ok"] = bool(result.get("ok"))
        data["audio_result"] = result
        data["audio_error"] = result.get("error")
        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/wizard/{draft_id}/video", response_class=HTMLResponse)
    def wizard_video(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            # If a user jumps directly here, we still allow them to set video_dir and validate.
            pass
        return templates.TemplateResponse(
            "wizard_video.html",
            {
                "request": request,
                "draft_id": draft_id,
                "video_dir": data.get("video_dir", ""),
                "video_ok": bool(data.get("video_ok", False)),
                "video_result": data.get("video_result"),
                "error": data.get("video_error"),
                **_wizard_flow_context(data, step=2),
            },
        )

    @app.post("/wizard/{draft_id}/video")
    def wizard_video_post(
        draft_id: int, video_dir: str = Form(...)
    ) -> RedirectResponse:
        video_dir = video_dir.strip()
        data = _draft_get(draft_id)
        if data.get("video_dir") != video_dir:
            invalidate_after_video_change(data, include_timestamp_state=True)

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
            except Exception as e:
                data["video_ok"] = False
                data["video_error"] = str(e)
                data["video_result"] = {**result, "ok": False, "error": str(e)}

        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)

    @app.get("/wizard/{draft_id}/output", response_class=HTMLResponse)
    def wizard_output(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        mode = str(data.get("mode") or "sync")
        template_name = (
            "wizard_output_timestamp.html"
            if mode == "audio_timestamp"
            else "wizard_output_sync.html"
        )
        continue_url = (
            f"/wizard/{draft_id}/select"
            if mode == "audio_timestamp"
            else f"/wizard/{draft_id}/decode"
        )
        return templates.TemplateResponse(
            template_name,
            {
                "request": request,
                "draft_id": draft_id,
                "out_dir": data.get("out_dir", ""),
                "out_ok": bool(data.get("out_ok", False)),
                "out_result": data.get("out_result"),
                "error": data.get("out_error"),
                "continue_url": continue_url,
                **_wizard_flow_context(data, step=3),
            },
        )

    @app.post("/wizard/{draft_id}/output")
    def wizard_output_post(draft_id: int, out_dir: str = Form(...)) -> RedirectResponse:
        out_dir = out_dir.strip()
        data = _draft_get(draft_id)
        if data.get("out_dir") != out_dir:
            invalidate_after_out_change(data, include_timestamp_state=True)
        data["out_dir"] = out_dir
        segments = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None
        result = validate_out_dir(out_dir, segments=segments)
        data["out_ok"] = bool(result.get("ok"))
        data["out_result"] = result
        data["out_error"] = result.get("error")
        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

    @app.get("/wizard/{draft_id}/decode", response_class=HTMLResponse)
    def wizard_decode(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if str(data.get("mode") or "sync") == "audio_timestamp":
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        out_result = data.get("out_result") or {}
        can_reuse = bool(out_result.get("can_reuse_audio_decoded"))
        decode_choice = str(data.get("decode_choice") or "").strip().lower()
        if decode_choice not in {"reuse", "rebuild"}:
            decode_choice = "reuse" if can_reuse else "rebuild"
        decode_status = _decode_status_from_draft(draft_id, data)
        out_dir = str(data.get("out_dir", "")).strip()
        with decode_val_lock:
            st = decode_val_state.get(draft_id)
            if st and str(st.get("out_dir", "")).strip() == out_dir:
                payload = st.get("result")
                if isinstance(payload, dict):
                    decode_status = payload

        return templates.TemplateResponse(
            "wizard_decode.html",
            {
                "request": request,
                "draft_id": draft_id,
                "site": data.get("site", "nbu_lounge"),
                "out_result": out_result,
                "decode_choice": decode_choice,
                "decode_error": data.get("decode_error"),
                "can_reuse_audio": can_reuse,
                "decode_ready": _is_decode_ready(data),
                "decode_status": decode_status,
                **_wizard_flow_context(data, step=4),
            },
        )

    @app.post("/wizard/{draft_id}/decode")
    def wizard_decode_post(
        draft_id: int,
        decode_action: str = Form("reuse"),
        site: str = Form("nbu_lounge"),
    ) -> RedirectResponse:
        data = _draft_get(draft_id)
        if str(data.get("mode") or "sync") == "audio_timestamp":
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        out_dir = str(data.get("out_dir", "")).strip()
        if not out_dir:
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        site_value = str(site or data.get("site") or "nbu_lounge").strip()
        if not site_value:
            site_value = "nbu_lounge"
        data["site"] = site_value

        out_result = data.get("out_result") or {}
        can_reuse = bool(out_result.get("can_reuse_audio_decoded"))
        action = str(decode_action or "").strip().lower()
        if action not in {"reuse", "rebuild"}:
            action = "reuse" if can_reuse else "rebuild"

        if action == "reuse":
            if not can_reuse:
                _clear_decode_state(data)
                data["decode_error"] = (
                    "No reusable audio artifacts found. Decode audio artifacts to continue."
                )
                _draft_set(draft_id, data)
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/decode", status_code=303
                )
            _mark_decode_ready(data, choice="reuse")
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        err = _rebuild_decode_artifacts_for_draft(data, site_value=site_value)
        if err:
            data["decode_error"] = err
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/decode", status_code=303)
        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @app.get("/wizard/{draft_id}/select", response_class=HTMLResponse)
    def wizard_select(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if str(data.get("mode") or "sync") == "audio_timestamp":
            if not bool(data.get("audio_ok", False)):
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/audio", status_code=303
                )
            if not bool(data.get("video_ok", False)):
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/video", status_code=303
                )
            if not bool(data.get("out_ok", False)):
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/output", status_code=303
                )
            output_json = _resolve_timestamp_output_path(data)
            return templates.TemplateResponse(
                "wizard_timestamp_select.html",
                {
                    "request": request,
                    "draft_id": draft_id,
                    "site": data.get("site", "nbu_lounge"),
                    "output_json": output_json or "",
                    "error": data.get("select_error"),
                    **_wizard_flow_context(data, step=4),
                },
            )
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)
        if not _is_decode_ready(data):
            return RedirectResponse(url=f"/wizard/{draft_id}/decode", status_code=303)
        video_result = data.get("video_result") or {}
        out_result = data.get("out_result") or {}
        available_pair_map = _extract_available_pair_map(video_result)
        missing_json_map = _extract_missing_json_map(video_result)
        synced_pair_map = _extract_synced_pair_map(out_result)

        cameras = [str(c) for c in (data.get("cameras_all") or []) if str(c).strip()]
        if not cameras:
            cameras = sorted(
                {k.split("::", 1)[1] for k in available_pair_map.keys() if "::" in k}
            )
        range_config = _coerce_range_config(data.get("range_config"), cameras)
        selection_mode = str(data.get("selection_mode") or "segments").strip().lower()
        if selection_mode not in {"segments", "time", "sample"}:
            selection_mode = "segments"
        video_dir = Path(str(data.get("video_dir", "")).strip()).expanduser()
        out_dir = Path(str(data.get("out_dir", "")).strip()).expanduser()
        try:
            range_catalog = _build_range_catalog(
                video_dir=video_dir,
                out_dir=out_dir,
                available_pair_map=available_pair_map,
            )
        except Exception:
            range_catalog = {
                "pairs": {},
                "pairs_by_camera": {},
                "camera_time_bounds": {},
                "camera_sample_bounds": {},
                "global_time_bounds": None,
                "global_sample_bounds": None,
                "sample_ready": False,
            }
        return templates.TemplateResponse(
            "wizard_select.html",
            {
                "request": request,
                "draft_id": draft_id,
                "site": data.get("site", "nbu_lounge"),
                "segments": data.get("segments_all", []),
                "cameras": cameras,
                "selected_pairs": data.get(
                    "manual_target_pairs", data.get("target_pairs", [])
                ),
                "run_error": data.get("run_error"),
                "available_pair_map": available_pair_map,
                "missing_json_map": missing_json_map,
                "synced_pair_map": synced_pair_map,
                "range_config": range_config,
                "range_catalog": range_catalog,
                "selection_mode": selection_mode,
                "error": data.get("select_error"),
                **_wizard_flow_context(data, step=5),
            },
        )

    @app.post("/wizard/{draft_id}/select")
    async def wizard_select_post(draft_id: int, request: Request) -> RedirectResponse:
        form = await request.form()
        log_level = str(form.get("log_level") or "INFO").strip() or "INFO"
        output_json = str(form.get("output_json") or "").strip() or None
        selection_mode = str(form.get("selection_mode") or "segments").strip().lower()
        if selection_mode not in {"segments", "time", "sample"}:
            selection_mode = "segments"
        picked_pairs = _normalize_target_pairs(
            [str(v) for v in form.getlist("target_pairs")]
        )

        data = _draft_get(draft_id)
        mode = str(data.get("mode") or "sync")
        data["all_segments"] = False
        data["all_cameras"] = False
        data["segments"] = []
        data["cameras"] = []
        data["selection_mode"] = selection_mode

        data["select_error"] = None
        data["run_error"] = None
        if mode == "audio_timestamp":
            out_path = _resolve_timestamp_output_path(data, override=output_json)
            if not out_path:
                data["select_error"] = "Output JSON path is required."
                _draft_set(draft_id, data)
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/select", status_code=303
                )
            data["timestamp_output_path"] = out_path
            data["target_pairs"] = []
            data["log_level"] = log_level
            data["skip_decode"] = False
            data["overwrite_clips"] = False
            data["split"] = False
            data["split_overwrite"] = False
            data["split_clean"] = False
            data["split_chunk_seconds"] = 3600
            data["schedule_at"] = ""
            data["timestamp_run_id"] = None
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/summary", status_code=303)

        if not _is_decode_ready(data):
            return RedirectResponse(url=f"/wizard/{draft_id}/decode", status_code=303)

        video_result = data.get("video_result") or {}
        available_pair_map = _extract_available_pair_map(video_result)
        cameras = [str(c) for c in (data.get("cameras_all") or []) if str(c).strip()]
        if not cameras:
            cameras = sorted(
                {k.split("::", 1)[1] for k in available_pair_map.keys() if "::" in k}
            )
        range_config = _build_range_config_from_form(
            selection_mode=selection_mode,
            cameras=cameras,
            form=form,
            prev_range_config=data.get("range_config"),
        )
        data["range_config"] = range_config
        data["manual_target_pairs"] = picked_pairs

        video_dir = Path(str(data.get("video_dir", "")).strip()).expanduser()
        out_dir_path = Path(str(data.get("out_dir", "")).strip()).expanduser()
        range_catalog = _build_range_catalog(
            video_dir=video_dir,
            out_dir=out_dir_path,
            available_pair_map=available_pair_map,
        )
        if selection_mode == "segments" and not picked_pairs:
            data["select_error"] = "Select at least one segment/camera pair."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        effective_pairs, range_error = _compute_effective_target_pairs(
            manual_pairs=picked_pairs,
            cameras=cameras,
            available_pair_map=available_pair_map,
            range_config=range_config,
            range_catalog=range_catalog,
        )
        if range_error:
            data["select_error"] = range_error
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)
        data["target_pairs"] = effective_pairs

        data["title"] = ""
        data["log_level"] = log_level
        data["skip_decode"] = True
        overwrite_clips = False
        out_dir = str(data.get("out_dir", "")).strip()
        if out_dir and effective_pairs:
            segs = sorted({p.split("::", 1)[0] for p in effective_pairs if "::" in p})
            cams = sorted({p.split("::", 1)[1] for p in effective_pairs if "::" in p})
            synced_pairs = discover_synced_pairs(
                out_dir, segments=segs or None, cameras=cams or None
            ).get("synced_pairs", [])
            synced_set = {
                f"{p.get('segment','')}::{p.get('camera','')}"
                for p in synced_pairs
                if p.get("segment") and p.get("camera")
            }
            overwrite_clips = any(pair in synced_set for pair in effective_pairs)
        data["overwrite_clips"] = overwrite_clips
        data["split"] = False
        data["split_overwrite"] = False
        data["split_clean"] = False
        data["split_chunk_seconds"] = 3600
        data["schedule_at"] = ""

        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/summary", status_code=303)

    @app.get("/wizard/{draft_id}/run")
    def wizard_run(draft_id: int) -> RedirectResponse:
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @app.post("/wizard/{draft_id}/run")
    def wizard_run_post(draft_id: int) -> RedirectResponse:
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @app.get("/wizard/{draft_id}/summary", response_class=HTMLResponse)
    def wizard_draft_summary(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        mode = str(data.get("mode") or "sync")
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        if not str(data.get("out_dir", "")).strip():
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        args: Dict[str, Any] = {
            "audio_dir": data.get("audio_dir", ""),
            "video_dir": data.get("video_dir", ""),
            "out_dir": data.get("out_dir", ""),
            "site": data.get("site", "nbu_lounge"),
            "segments": data.get("segments", []),
            "cameras": data.get("cameras", []),
            "target_pairs": data.get("target_pairs", []),
            "log_level": data.get("log_level", "INFO"),
            "skip_decode": bool(data.get("skip_decode", False)),
            "decode_choice": str(data.get("decode_choice") or ""),
            "overwrite_clips": bool(data.get("overwrite_clips", False)),
            "split": bool(data.get("split", False)),
            "split_overwrite": bool(data.get("split_overwrite", False)),
            "split_clean": bool(data.get("split_clean", False)),
            "split_chunk_seconds": int(data.get("split_chunk_seconds", 3600)),
            "selection_mode": str(data.get("selection_mode") or "segments"),
            "range_config": _coerce_range_config(
                data.get("range_config"),
                [str(c) for c in (data.get("cameras_all") or []) if str(c).strip()],
            ),
        }
        if mode == "audio_timestamp":
            output_path = _resolve_timestamp_output_path(data)
            return templates.TemplateResponse(
                "wizard_timestamp_summary.html",
                {
                    "request": request,
                    "draft_id": draft_id,
                    "args": args,
                    "output_path": output_path,
                    **_wizard_flow_context(data, step=5),
                    "back_url": f"/wizard/{draft_id}/select",
                },
            )
        if not _is_decode_ready(data):
            return RedirectResponse(url=f"/wizard/{draft_id}/decode", status_code=303)
        run_groups = _build_sync_run_groups(data)
        return templates.TemplateResponse(
            "wizard_summary.html",
            {
                "request": request,
                "draft_id": draft_id,
                "args": args,
                "run_groups": run_groups,
                **_wizard_flow_context(data, step=6),
                "back_url": f"/wizard/{draft_id}/select",
            },
        )

    @app.post("/wizard/{draft_id}/start")
    def wizard_start(draft_id: int) -> RedirectResponse:
        data = _draft_get(draft_id)
        data["run_error"] = None
        mode = str(data.get("mode") or "sync")

        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)
        if not str(data.get("out_dir", "")).strip():
            data["run_error"] = "Output dir is required."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        if mode == "audio_timestamp":
            existing_run_id = data.get("timestamp_run_id")
            if isinstance(existing_run_id, int):
                return RedirectResponse(
                    url=f"/audio-timestamp/runs/{existing_run_id}", status_code=303
                )
            output_path = _resolve_timestamp_output_path(data)
            args: Dict[str, Any] = {
                "audio_dir": data.get("audio_dir", ""),
                "video_dir": data.get("video_dir", ""),
                "out_dir": data.get("out_dir", ""),
                "site": data.get("site", "nbu_lounge"),
                "timezone": DEFAULT_TZ.key,
                "output_json": output_path,
            }
            run_id = _create_timestamp_run(args, output_path=output_path or None)
            data["timestamp_run_id"] = run_id
            _draft_set(draft_id, data)
            _start_audio_timestamp_job(draft_id, run_id=run_id)
            return RedirectResponse(
                url=f"/audio-timestamp/runs/{run_id}", status_code=303
            )
        if not _is_decode_ready(data):
            data["run_error"] = "Decode audio artifacts before starting the run."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/decode", status_code=303)

        base_args: Dict[str, Any] = {
            "audio_dir": data.get("audio_dir", ""),
            "video_dir": data.get("video_dir", ""),
            "out_dir": data.get("out_dir", ""),
            "site": data.get("site", "nbu_lounge"),
            "segments": data.get("segments", []),
            "cameras": data.get("cameras", []),
            "log_level": data.get("log_level", "INFO"),
            "skip_decode": bool(data.get("skip_decode", False)),
            "overwrite_clips": bool(data.get("overwrite_clips", False)),
            "split": bool(data.get("split", False)),
            "split_overwrite": bool(data.get("split_overwrite", False)),
            "split_clean": bool(data.get("split_clean", False)),
            "split_chunk_seconds": int(data.get("split_chunk_seconds", 3600)),
        }
        run_groups = _build_sync_run_groups(data)
        if not run_groups:
            data["run_error"] = "No valid target pairs to run."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        schedule_at = str(data.get("schedule_at", "") or "").strip()
        now = utc_now_iso()
        try:
            status, scheduled_iso = resolve_schedule(schedule_at)
        except Exception as e:
            data["run_error"] = f"Bad schedule_at: {e}"
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        created_run_ids = enqueue_sync_runs(
            base_args=base_args,
            run_groups=run_groups,
            title=str(data.get("title") or "video-sync run"),
            created_at=now,
            status=status,
            scheduled_at=scheduled_iso,
            build_cmd=build_cli_cmd,
            default_time_zone=WEBUI_DEFAULT_TIME_ZONE,
        )

        _draft_delete(draft_id)
        if len(created_run_ids) == 1:
            return RedirectResponse(url=f"/runs/{created_run_ids[0]}", status_code=303)
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: int) -> HTMLResponse:
        row = run_repo.get_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            args = json.loads(row["args_json"])
        except Exception:
            args = {}
        return templates.TemplateResponse(
            "run_detail.html",
            {
                "request": request,
                "run": row,
                "args": args,
            },
        )

    @app.get("/audio-timestamp/runs/{run_id}", response_class=HTMLResponse)
    def audio_timestamp_run_detail(request: Request, run_id: int) -> HTMLResponse:
        row = timestamp_run_repo.get_timestamp_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            args = json.loads(row["args_json"])
        except Exception:
            args = {}
        json_text: Optional[str] = None
        output_path = row["output_path"]
        if output_path:
            try:
                json_text = Path(output_path).read_text(encoding="utf-8")
            except Exception:
                json_text = None
        return templates.TemplateResponse(
            "audio_timestamp_run_detail.html",
            {
                "request": request,
                "run": row,
                "args": args,
                "json_text": json_text,
            },
        )

    @app.post("/audio-timestamp/runs/{run_id}/delete")
    def audio_timestamp_run_delete(run_id: int) -> RedirectResponse:
        status = timestamp_run_repo.get_timestamp_run_status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if status == "running":
            raise HTTPException(status_code=400, detail="Cannot delete a running run")
        timestamp_run_repo.delete_timestamp_run(run_id)
        return RedirectResponse(url="/audio-timestamp/runs", status_code=303)

    # (Summary is shown before starting; see /wizard/{draft_id}/summary.)

    @app.post("/runs/{run_id}/cancel")
    def run_cancel(run_id: int) -> RedirectResponse:
        ok = runner.request_cancel(run_id)
        if not ok:
            raise HTTPException(status_code=400, detail="Cannot cancel this run")
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/delete")
    def run_delete(run_id: int) -> RedirectResponse:
        """
        Remove a run from history (DB row) and delete its log file (best-effort).
        """
        row = run_repo.get_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        if str(row["status"]) == "running":
            raise HTTPException(status_code=400, detail="Cannot delete a running run")

        try:
            p = Path(row["log_path"])
            if p.exists():
                p.unlink()
        except Exception:
            pass

        run_repo.delete_run(run_id)
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/{run_id}/logs")
    def run_logs(run_id: int) -> StreamingResponse:
        log_path = run_repo.get_run_log_path(run_id)
        if log_path is None:
            raise HTTPException(status_code=404, detail="Run not found")
        path = Path(log_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Log not found")

        def _iter() -> Any:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 64)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(_iter(), media_type="text/plain")

    @app.get("/runs/{run_id}/logs/stream")
    async def run_logs_stream(run_id: int) -> StreamingResponse:
        log_path = run_repo.get_run_log_path(run_id)
        if log_path is None:
            raise HTTPException(status_code=404, detail="Run not found")
        path = Path(log_path)
        return StreamingResponse(
            tail_log_sse(path),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: int) -> Dict[str, Any]:
        row = run_repo.get_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        d = dict(row)
        # don't return huge blobs
        return d

    @app.get("/api/audio-timestamp/runs/{run_id}")
    def api_audio_timestamp_run(run_id: int) -> Dict[str, Any]:
        row = timestamp_run_repo.get_timestamp_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "records_count": row["records_count"],
            "error": row["error"],
        }

    return app


app = create_app()
