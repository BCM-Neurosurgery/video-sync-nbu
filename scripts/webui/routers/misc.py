from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from scripts.validate.validate_video_dir import discover_from_video_dir


def create_misc_router(*, templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    @router.get("/api/video-index")
    def api_video_index(video_dir: str) -> Dict[str, Any]:
        p = Path(video_dir).expanduser()
        try:
            data = discover_from_video_dir(p)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to scan video_dir: {exc}"
            )
        return {"video_dir": str(p), **data}

    @router.get("/api/audio-metadata")
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

    @router.get("/api/pick-dir")
    def api_pick_dir(initial: str = "") -> Dict[str, Any]:
        initial = (initial or "").strip()

        if sys.platform == "darwin":
            script_lines = [
                "try",
                'set promptText to "Select folder"',
            ]
            if initial:
                p = Path(initial).expanduser()
                if p.is_file():
                    p = p.parent
                initial_dir = str(p).replace('"', '\\"')
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
                res = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                chosen = (res.stdout or "").strip()
                if chosen:
                    return {"path": chosen, "canceled": False}
                return {"path": "", "canceled": True}
            except Exception:
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

    @router.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("home.html", {"request": request})

    return router
