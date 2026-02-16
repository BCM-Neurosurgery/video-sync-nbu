from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from scripts.webui.repositories import run_repo, timestamp_run_repo
from scripts.webui.runner import tail_log_sse


DraftCreateFn = Callable[..., int]
SegmentRangeTitleFn = Callable[[Dict[str, Any]], str]


def create_runs_router(
    *,
    templates: Jinja2Templates,
    draft_create: DraftCreateFn,
    segment_range_title: SegmentRangeTitleFn,
    runner: Any,
) -> APIRouter:
    router = APIRouter()

    @router.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        items = run_repo.list_runs(limit=200)
        runs_view: List[Dict[str, Any]] = []
        for row in items:
            d = dict(row)
            try:
                args = json.loads(d.get("args_json") or "{}")
            except Exception:
                args = {}
            d["display_title"] = segment_range_title(args)
            runs_view.append(d)
        return templates.TemplateResponse(
            "runs.html",
            {"request": request, "runs": runs_view},
        )

    @router.get("/audio-timestamp/runs", response_class=HTMLResponse)
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

    @router.get("/api/runs")
    def api_runs() -> List[Dict[str, Any]]:
        rows = run_repo.list_runs_summary(limit=200)
        return [dict(r) for r in rows]

    @router.get("/api/audio-timestamp/runs")
    def api_audio_timestamp_runs() -> List[Dict[str, Any]]:
        rows = timestamp_run_repo.list_timestamp_runs_summary(limit=200)
        return [dict(r) for r in rows]

    @router.post("/runs/clear")
    def runs_clear() -> RedirectResponse:
        rows = run_repo.list_non_running_runs()
        for r in rows:
            try:
                p = Path(r["log_path"])
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        run_repo.delete_non_running_runs()
        return RedirectResponse(url="/runs", status_code=303)

    @router.get("/runs/new", response_class=HTMLResponse)
    def runs_new() -> RedirectResponse:
        draft_id = draft_create(mode="sync")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @router.get("/tools/audio-timestamp")
    def tools_audio_timestamp() -> RedirectResponse:
        draft_id = draft_create(mode="audio_timestamp")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @router.get("/tools/audio-timestamp/new")
    def tools_audio_timestamp_new() -> RedirectResponse:
        draft_id = draft_create(mode="audio_timestamp")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @router.get("/runs/{run_id}", response_class=HTMLResponse)
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

    @router.get("/audio-timestamp/runs/{run_id}", response_class=HTMLResponse)
    def audio_timestamp_run_detail(request: Request, run_id: int) -> HTMLResponse:
        row = timestamp_run_repo.get_timestamp_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            args = json.loads(row["args_json"])
        except Exception:
            args = {}
        json_text = None
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

    @router.post("/audio-timestamp/runs/{run_id}/delete")
    def audio_timestamp_run_delete(run_id: int) -> RedirectResponse:
        status = timestamp_run_repo.get_timestamp_run_status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if status == "running":
            raise HTTPException(status_code=400, detail="Cannot delete a running run")
        timestamp_run_repo.delete_timestamp_run(run_id)
        return RedirectResponse(url="/audio-timestamp/runs", status_code=303)

    @router.post("/runs/{run_id}/cancel")
    def run_cancel(run_id: int) -> RedirectResponse:
        ok = runner.request_cancel(run_id)
        if not ok:
            raise HTTPException(status_code=400, detail="Cannot cancel this run")
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @router.post("/runs/{run_id}/delete")
    def run_delete(run_id: int) -> RedirectResponse:
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

    @router.get("/runs/{run_id}/logs")
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

    @router.get("/runs/{run_id}/logs/stream")
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

    @router.get("/api/runs/{run_id}")
    def api_run(run_id: int) -> Dict[str, Any]:
        row = run_repo.get_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return dict(row)

    @router.get("/api/audio-timestamp/runs/{run_id}")
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

    return router
