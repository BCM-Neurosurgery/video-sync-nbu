from __future__ import annotations

import json
import logging
import os
import re
import sys
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.webui.db import get_conn, tx
from scripts.webui.models import utc_now_iso
from scripts.webui.runner import Runner, RunnerConfig, build_cli_cmd, tail_log_sse
from scripts.validate.validate_audio_dir import validate_audio_dir
from scripts.validate.validate_audio_dir import validate_audio_dir_progress
from scripts.validate.validate_out_dir import discover_synced_pairs, validate_out_dir
from scripts.validate.validate_video_dir import (
    discover_from_video_dir,
    validate_video_dir,
    validate_video_dir_progress,
)


ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
try:
    templates.env.globals["static_v"] = int(
        (ROOT / "static" / "app.css").stat().st_mtime
    )
except Exception:
    templates.env.globals["static_v"] = 1


def _default_args() -> Dict[str, Any]:
    return {
        "audio_dir": "",
        "video_dir": "",
        "out_dir": "",
        "site": "nbu_lounge",
        "segments": [],
        "cameras": [],
        "target_pairs": [],
        "log_level": "INFO",
        "skip_decode": False,
        "overwrite_clips": False,
        "split": False,
        "split_overwrite": False,
        "split_clean": False,
        "split_chunk_seconds": 3600,
    }


def _normalize_target_pairs(raw: Optional[List[str]]) -> List[str]:
    pairs: List[str] = []
    seen: set[str] = set()
    for item in raw or []:
        s = str(item).strip()
        if not s:
            continue
        if "::" in s:
            seg, cam = s.split("::", 1)
        elif ":" in s:
            seg, cam = s.split(":", 1)
        else:
            continue
        seg = seg.strip()
        cam = cam.strip()
        if not seg or not cam:
            continue
        key = f"{seg}::{cam}"
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _sort_segment_ids(segment_ids: List[str]) -> List[str]:
    def key(seg: str) -> tuple:
        match = re.search(r"(\d{8})_(\d{6})", seg)
        if match:
            try:
                return (0, int(match.group(1)), int(match.group(2)), seg)
            except ValueError:
                pass
        return (1, seg)

    return sorted(segment_ids, key=key)


def _segment_range_title(args: Dict[str, Any]) -> str:
    segs: List[str] = []
    raw_pairs = args.get("target_pairs")
    if isinstance(raw_pairs, list) and raw_pairs:
        seen: set[str] = set()
        for item in raw_pairs:
            s = str(item).strip()
            if not s:
                continue
            if "::" in s:
                seg = s.split("::", 1)[0]
            elif ":" in s:
                seg = s.split(":", 1)[0]
            else:
                continue
            seg = seg.strip()
            if seg and seg not in seen:
                seen.add(seg)
                segs.append(seg)
    else:
        raw = args.get("segments")
        if isinstance(raw, list):
            segs = [str(s).strip() for s in raw if str(s).strip()]

    segs = _sort_segment_ids(segs)
    if not segs:
        return "All segments"
    if len(segs) == 1:
        return segs[0]
    first = segs[0]
    last = segs[-1]
    prefix = first.rsplit("_", 1)[0] if "_" in first else ""
    if prefix and last.startswith(prefix + "_"):
        last = last[len(prefix) + 1 :]
    return f"{first} → {last}"


def _draft_create() -> int:
    conn = get_conn()
    now = utc_now_iso()
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO drafts(created_at, updated_at, data_json) VALUES(?, ?, ?)",
            (now, now, json.dumps({})),
        )
        return int(cur.lastrowid)


def _draft_get(draft_id: int) -> Dict[str, Any]:
    conn = get_conn()
    row = conn.execute(
        "SELECT data_json FROM drafts WHERE id=?", (draft_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Draft not found")
    try:
        return json.loads(row["data_json"])
    except Exception:
        return {}


def _draft_set(draft_id: int, data: Dict[str, Any]) -> None:
    conn = get_conn()
    now = utc_now_iso()
    with tx(conn):
        conn.execute(
            "UPDATE drafts SET updated_at=?, data_json=? WHERE id=?",
            (now, json.dumps(data), draft_id),
        )


def _draft_delete(draft_id: int) -> None:
    conn = get_conn()
    with tx(conn):
        conn.execute("DELETE FROM drafts WHERE id=?", (draft_id,))


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
    ]

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
    def home() -> RedirectResponse:
        return RedirectResponse(url="/runs")

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        conn = get_conn()
        items = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 200").fetchall()
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

    @app.get("/api/runs")
    def api_runs() -> List[Dict[str, Any]]:
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, status, exit_code FROM runs ORDER BY id DESC LIMIT 200"
        ).fetchall()
        return [dict(r) for r in rows]

    @app.post("/runs/clear")
    def runs_clear() -> RedirectResponse:
        """
        Clear run history (DB rows + log files) for all runs that are not running.
        """
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, status, log_path FROM runs WHERE status != 'running'"
        ).fetchall()
        for r in rows:
            try:
                p = Path(r["log_path"])
                if p.exists():
                    p.unlink()
            except Exception:
                # Best-effort deletion; DB clear still proceeds.
                pass
        with tx(conn):
            conn.execute("DELETE FROM runs WHERE status != 'running'")
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/new", response_class=HTMLResponse)
    def runs_new() -> RedirectResponse:
        draft_id = _draft_create()
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/api/drafts/{draft_id}/validate-audio")
    def api_validate_audio(draft_id: int, audio_dir: str) -> Dict[str, Any]:
        """
        Validate audio_dir and persist the result into the run-creation draft.
        """
        data = _draft_get(draft_id)
        if data.get("audio_dir") != audio_dir.strip():
            # Changing audio dir invalidates downstream choices.
            data.pop("video_dir", None)
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("target_pairs", None)
            data.pop("out_dir", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
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
            data.pop("video_dir", None)
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("target_pairs", None)
            data.pop("out_dir", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
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
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("all_segments", None)
            data.pop("all_cameras", None)
            data.pop("target_pairs", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
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
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("all_segments", None)
            data.pop("all_cameras", None)
            data.pop("target_pairs", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
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
            data.pop("target_pairs", None)
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
            data.pop("target_pairs", None)
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
            "synced_segments_count": 0,
            "synced_segments_preview": [],
            "total_segments": None,
            "checks": [
                {"name": n, "status": "pending", "message": ""} for n in out_check_names
            ],
            "error": None,
        }

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
            data.pop("video_dir", None)
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("target_pairs", None)
            data.pop("out_dir", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)

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
            },
        )

    @app.post("/wizard/{draft_id}/video")
    def wizard_video_post(
        draft_id: int, video_dir: str = Form(...)
    ) -> RedirectResponse:
        video_dir = video_dir.strip()
        data = _draft_get(draft_id)
        if data.get("video_dir") != video_dir:
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("all_segments", None)
            data.pop("all_cameras", None)
            data.pop("target_pairs", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)

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
        return templates.TemplateResponse(
            "wizard_output.html",
            {
                "request": request,
                "draft_id": draft_id,
                "out_dir": data.get("out_dir", ""),
                "out_ok": bool(data.get("out_ok", False)),
                "out_result": data.get("out_result"),
                "error": data.get("out_error"),
            },
        )

    @app.post("/wizard/{draft_id}/output")
    def wizard_output_post(draft_id: int, out_dir: str = Form(...)) -> RedirectResponse:
        out_dir = out_dir.strip()
        data = _draft_get(draft_id)
        if data.get("out_dir") != out_dir:
            data.pop("target_pairs", None)
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

    @app.get("/wizard/{draft_id}/select", response_class=HTMLResponse)
    def wizard_select(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)
        video_result = data.get("video_result") or {}
        out_result = data.get("out_result") or {}
        available_pair_map: Dict[str, bool] = {}
        missing_json_map: Dict[str, bool] = {}
        synced_pair_map: Dict[str, bool] = {}
        available_pairs = video_result.get("available_pairs")
        if isinstance(available_pairs, list):
            for item in available_pairs:
                if isinstance(item, dict):
                    seg = item.get("segment")
                    cam = item.get("camera")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    seg = item[0]
                    cam = item[1]
                else:
                    continue
                if seg and cam:
                    available_pair_map[f"{seg}::{cam}"] = True
        missing_json_segments = video_result.get("missing_json_segments")
        if isinstance(missing_json_segments, list):
            for seg in missing_json_segments:
                if seg:
                    missing_json_map[str(seg)] = True
        synced_pairs = out_result.get("synced_pairs")
        if isinstance(synced_pairs, list):
            for item in synced_pairs:
                if isinstance(item, dict):
                    seg = item.get("segment")
                    cam = item.get("camera")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    seg = item[0]
                    cam = item[1]
                else:
                    continue
                if seg and cam:
                    synced_pair_map[f"{seg}::{cam}"] = True
        return templates.TemplateResponse(
            "wizard_select.html",
            {
                "request": request,
                "draft_id": draft_id,
                "site": data.get("site", "nbu_lounge"),
                "segments": data.get("segments_all", []),
                "cameras": data.get("cameras_all", []),
                "selected_segments": data.get("segments", []),
                "selected_cameras": data.get("cameras", []),
                "selected_pairs": data.get("target_pairs", []),
                "all_segments": bool(data.get("all_segments", True)),
                "all_cameras": bool(data.get("all_cameras", True)),
                "title_value": data.get("title", ""),
                "log_level": data.get("log_level", "INFO"),
                "skip_decode": bool(data.get("skip_decode", False)),
                "split": bool(data.get("split", False)),
                "split_overwrite": bool(data.get("split_overwrite", False)),
                "split_clean": bool(data.get("split_clean", False)),
                "split_chunk_seconds": int(data.get("split_chunk_seconds", 3600)),
                "run_error": data.get("run_error"),
                "reuse_audio_available": bool(
                    out_result.get("audio_decoded", {}).get("filtered_csv")
                ),
                "available_pair_map": available_pair_map,
                "missing_json_map": missing_json_map,
                "synced_pair_map": synced_pair_map,
                "error": data.get("select_error"),
            },
        )

    @app.post("/wizard/{draft_id}/select")
    def wizard_select_post(
        draft_id: int,
        site: str = Form(default="nbu_lounge"),
        target_pairs: Optional[List[str]] = Form(default=None),
        log_level: str = Form(default="INFO"),
        reuse_audio: Optional[str] = Form(default=None),
    ) -> RedirectResponse:
        data = _draft_get(draft_id)
        data["site"] = site
        picked_pairs = _normalize_target_pairs(target_pairs)

        data["target_pairs"] = picked_pairs
        data["all_segments"] = False
        data["all_cameras"] = False
        data["segments"] = []
        data["cameras"] = []

        data["select_error"] = None
        data["run_error"] = None
        if not picked_pairs:
            data["select_error"] = "Select at least one segment/camera pair."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        data["title"] = ""
        data["log_level"] = log_level
        can_reuse = bool(
            (data.get("out_result") or {}).get("audio_decoded", {}).get("filtered_csv")
        )
        reuse_enabled = reuse_audio is not None and can_reuse
        data["skip_decode"] = reuse_enabled
        overwrite_clips = False
        out_dir = str(data.get("out_dir", "")).strip()
        if out_dir and picked_pairs:
            segs = sorted({p.split("::", 1)[0] for p in picked_pairs if "::" in p})
            cams = sorted({p.split("::", 1)[1] for p in picked_pairs if "::" in p})
            synced_pairs = discover_synced_pairs(
                out_dir, segments=segs or None, cameras=cams or None
            ).get("synced_pairs", [])
            synced_set = {
                f"{p.get('segment','')}::{p.get('camera','')}"
                for p in synced_pairs
                if p.get("segment") and p.get("camera")
            }
            overwrite_clips = any(pair in synced_set for pair in picked_pairs)
        data["overwrite_clips"] = overwrite_clips
        data["split"] = not reuse_enabled
        data["split_overwrite"] = not reuse_enabled
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
            "overwrite_clips": bool(data.get("overwrite_clips", False)),
            "split": bool(data.get("split", False)),
            "split_overwrite": bool(data.get("split_overwrite", False)),
            "split_clean": bool(data.get("split_clean", False)),
            "split_chunk_seconds": int(data.get("split_chunk_seconds", 3600)),
        }
        return templates.TemplateResponse(
            "wizard_summary.html",
            {
                "request": request,
                "draft_id": draft_id,
                "args": args,
                "title_value": data.get("title", ""),
                "schedule_at": data.get("schedule_at", ""),
            },
        )

    @app.post("/wizard/{draft_id}/start")
    def wizard_start(draft_id: int) -> RedirectResponse:
        data = _draft_get(draft_id)
        data["run_error"] = None

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
            "overwrite_clips": bool(data.get("overwrite_clips", False)),
            "split": bool(data.get("split", False)),
            "split_overwrite": bool(data.get("split_overwrite", False)),
            "split_clean": bool(data.get("split_clean", False)),
            "split_chunk_seconds": int(data.get("split_chunk_seconds", 3600)),
        }
        cmd = build_cli_cmd(args)

        schedule_at = str(data.get("schedule_at", "") or "").strip()
        now = utc_now_iso()
        scheduled_iso: Optional[str] = None
        status = "queued"
        if schedule_at:
            try:
                dt = datetime.fromisoformat(schedule_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
                scheduled_iso = dt.astimezone(timezone.utc).isoformat(
                    timespec="seconds"
                )
                status = "scheduled"
            except Exception as e:
                data["run_error"] = f"Bad schedule_at: {e}"
                _draft_set(draft_id, data)
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/select", status_code=303
                )

        logs_dir = Path(".webui") / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        conn = get_conn()
        with tx(conn):
            cur = conn.execute(
                """
                INSERT INTO runs(title, created_at, scheduled_at, status, cwd, cmd_json, args_json, log_path)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("title") or "video-sync run",
                    now,
                    scheduled_iso,
                    status,
                    str(Path.cwd()),
                    json.dumps(cmd),
                    json.dumps(args),
                    str(logs_dir / "pending.log"),
                ),
            )
            run_id = int(cur.lastrowid)
            log_path = logs_dir / f"run-{run_id}.log"
            conn.execute(
                "UPDATE runs SET log_path=? WHERE id=?", (str(log_path), run_id)
            )

        _draft_delete(draft_id)
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: int) -> HTMLResponse:
        conn = get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
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
        conn = get_conn()
        row = conn.execute(
            "SELECT status, log_path FROM runs WHERE id=?", (run_id,)
        ).fetchone()
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

        with tx(conn):
            conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/{run_id}/logs")
    def run_logs(run_id: int) -> StreamingResponse:
        conn = get_conn()
        row = conn.execute("SELECT log_path FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        path = Path(row["log_path"])
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
        conn = get_conn()
        row = conn.execute("SELECT log_path FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        path = Path(row["log_path"])
        return StreamingResponse(
            tail_log_sse(path),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: int) -> Dict[str, Any]:
        conn = get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        d = dict(row)
        # don't return huge blobs
        return d

    return app


app = create_app()
