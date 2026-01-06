from __future__ import annotations

import json
import os
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


ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def _parse_csv_list(s: str) -> List[str]:
    items = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def _default_args() -> Dict[str, Any]:
    return {
        "audio_dir": "",
        "video_dir": "",
        "out_dir": "",
        "site": "nbu_lounge",
        "segments": [],
        "cameras": [],
        "log_level": "INFO",
        "skip_decode": False,
        "split": False,
        "split_overwrite": False,
        "split_clean": False,
        "split_chunk_seconds": 3600,
    }


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

    @app.get("/", response_class=HTMLResponse)
    def home() -> RedirectResponse:
        return RedirectResponse(url="/runs")

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        conn = get_conn()
        items = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 200").fetchall()
        return templates.TemplateResponse(
            "runs.html",
            {"request": request, "runs": items},
        )

    @app.get("/runs/new", response_class=HTMLResponse)
    def runs_new(request: Request) -> HTMLResponse:
        args = _default_args()
        return templates.TemplateResponse(
            "run_new.html",
            {"request": request, "args": args},
        )

    @app.post("/runs/new")
    def runs_create(
        request: Request,
        title: str = Form(default=""),
        audio_dir: str = Form(...),
        video_dir: str = Form(...),
        out_dir: str = Form(...),
        site: str = Form(default="nbu_lounge"),
        segments: str = Form(default=""),
        cameras: str = Form(default=""),
        log_level: str = Form(default="INFO"),
        skip_decode: Optional[str] = Form(default=None),
        split: Optional[str] = Form(default=None),
        split_overwrite: Optional[str] = Form(default=None),
        split_clean: Optional[str] = Form(default=None),
        split_chunk_seconds: int = Form(default=3600),
        schedule_at: str = Form(default=""),
    ) -> RedirectResponse:
        args: Dict[str, Any] = {
            "audio_dir": audio_dir,
            "video_dir": video_dir,
            "out_dir": out_dir,
            "site": site,
            "segments": _parse_csv_list(segments),
            "cameras": _parse_csv_list(cameras),
            "log_level": log_level,
            "skip_decode": skip_decode is not None,
            "split": split is not None,
            "split_overwrite": split_overwrite is not None,
            "split_clean": split_clean is not None,
            "split_chunk_seconds": int(split_chunk_seconds),
        }
        cmd = build_cli_cmd(args)

        now = utc_now_iso()
        scheduled_iso: Optional[str] = None
        status = "queued"
        if schedule_at.strip():
            try:
                # Expect local input like "2026-01-06T12:34"
                dt = datetime.fromisoformat(schedule_at.strip())
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                scheduled_iso = dt.astimezone(timezone.utc).isoformat(
                    timespec="seconds"
                )
                status = "scheduled"
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Bad schedule_at: {e}")

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
                    title.strip() or "video-sync run",
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
                "UPDATE runs SET log_path=? WHERE id=?",
                (str(log_path), run_id),
            )

        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: int) -> HTMLResponse:
        conn = get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return templates.TemplateResponse(
            "run_detail.html",
            {"request": request, "run": row},
        )

    @app.post("/runs/{run_id}/cancel")
    def run_cancel(run_id: int) -> RedirectResponse:
        ok = runner.request_cancel(run_id)
        if not ok:
            raise HTTPException(status_code=400, detail="Cannot cancel this run")
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

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
