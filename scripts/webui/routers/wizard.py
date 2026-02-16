from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from scripts.index.common import DEFAULT_TZ
from scripts.validate.validate_out_dir import discover_synced_pairs
from scripts.webui.models import utc_now_iso
from scripts.webui.runner import build_cli_cmd, tail_log_sse
from scripts.webui.services.run_service import enqueue_sync_runs, resolve_schedule
from scripts.webui.services.validation_service import ValidationService
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
)
from scripts.webui.wizard_state import (
    clear_decode_state,
    is_decode_ready,
    mark_decode_ready,
    wizard_flow_context,
)

DraftGetFn = Callable[[int], Dict[str, Any]]
DraftSetFn = Callable[[int, Dict[str, Any]], None]
DraftDeleteFn = Callable[[int], None]
RebuildDecodeArtifactsFn = Callable[..., Optional[str]]
ResolveTimestampOutputPathFn = Callable[..., str]
CreateTimestampRunFn = Callable[..., int]
StartAudioTimestampJobFn = Callable[..., None]


def create_wizard_router(
    *,
    templates: Jinja2Templates,
    validation_service: ValidationService,
    draft_get: DraftGetFn,
    draft_set: DraftSetFn,
    draft_delete: DraftDeleteFn,
    rebuild_decode_artifacts_for_draft: RebuildDecodeArtifactsFn,
    resolve_timestamp_output_path: ResolveTimestampOutputPathFn,
    create_timestamp_run: CreateTimestampRunFn,
    start_audio_timestamp_job: StartAudioTimestampJobFn,
) -> APIRouter:
    router = APIRouter()

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
        if is_decode_ready(data):
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

    @router.post("/api/drafts/{draft_id}/decode/start")
    def api_decode_start(
        draft_id: int,
        decode_action: str = Form("rebuild"),
        site: str = Form("nbu_lounge"),
    ) -> Dict[str, Any]:
        data = draft_get(draft_id)
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
                mark_decode_ready(data, choice="reuse")
                data["decode_log_path"] = ""
            else:
                clear_decode_state(data)
                data["decode_error"] = (
                    "No reusable audio artifacts found. Decode audio artifacts to continue."
                )
            draft_set(draft_id, data)
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
            draft_set(draft_id, data)
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
                latest = draft_get(draft_id)
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
                draft_set(draft_id, latest)
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
                data2 = draft_get(draft_id)
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
                draft_set(draft_id, data2)

                err = rebuild_decode_artifacts_for_draft(
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
                    data3 = draft_get(draft_id)
                    clear_decode_state(data3)
                    data3["decode_error"] = f"Audio decode failed: {exc}"
                    draft_set(draft_id, data3)
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

    @router.get("/api/drafts/{draft_id}/decode/status")
    def api_decode_status(draft_id: int) -> Dict[str, Any]:
        data = draft_get(draft_id)
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

    @router.get("/api/drafts/{draft_id}/decode/log")
    def api_decode_log(draft_id: int) -> Dict[str, Any]:
        data = draft_get(draft_id)
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

    @router.get("/api/drafts/{draft_id}/decode/log/download")
    def api_decode_log_download(draft_id: int) -> StreamingResponse:
        data = draft_get(draft_id)
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

    @router.get("/api/drafts/{draft_id}/decode/log/stream")
    async def api_decode_log_stream(
        draft_id: int, from_end: int = 0
    ) -> StreamingResponse:
        data = draft_get(draft_id)
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

    @router.get("/wizard/{draft_id}/audio", response_class=HTMLResponse)
    def wizard_audio(request: Request, draft_id: int) -> HTMLResponse:
        data = draft_get(draft_id)
        return templates.TemplateResponse(
            "wizard_audio.html",
            {
                "request": request,
                "draft_id": draft_id,
                "audio_dir": data.get("audio_dir", ""),
                "audio_ok": bool(data.get("audio_ok", False)),
                "audio_result": data.get("audio_result"),
                "error": data.get("audio_error"),
                **wizard_flow_context(data, step=1),
            },
        )

    @router.post("/wizard/{draft_id}/audio")
    def wizard_audio_post(
        draft_id: int, audio_dir: str = Form(...)
    ) -> RedirectResponse:
        # Validate on POST as a no-JS fallback.
        validation_service.validate_audio(
            draft_id,
            audio_dir,
            include_timestamp_state=True,
        )
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @router.get("/wizard/{draft_id}/video", response_class=HTMLResponse)
    def wizard_video(request: Request, draft_id: int) -> HTMLResponse:
        data = draft_get(draft_id)
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
                **wizard_flow_context(data, step=2),
            },
        )

    @router.post("/wizard/{draft_id}/video")
    def wizard_video_post(
        draft_id: int, video_dir: str = Form(...)
    ) -> RedirectResponse:
        validation_service.validate_video(
            draft_id,
            video_dir,
            include_timestamp_state=True,
        )
        return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)

    @router.get("/wizard/{draft_id}/output", response_class=HTMLResponse)
    def wizard_output(request: Request, draft_id: int) -> HTMLResponse:
        data = draft_get(draft_id)
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
                **wizard_flow_context(data, step=3),
            },
        )

    @router.post("/wizard/{draft_id}/output")
    def wizard_output_post(draft_id: int, out_dir: str = Form(...)) -> RedirectResponse:
        validation_service.validate_out(
            draft_id,
            out_dir,
            include_timestamp_state=True,
        )
        return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

    @router.get("/wizard/{draft_id}/decode", response_class=HTMLResponse)
    def wizard_decode(request: Request, draft_id: int) -> HTMLResponse:
        data = draft_get(draft_id)
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
                "decode_ready": is_decode_ready(data),
                "decode_status": decode_status,
                **wizard_flow_context(data, step=4),
            },
        )

    @router.post("/wizard/{draft_id}/decode")
    def wizard_decode_post(
        draft_id: int,
        decode_action: str = Form("reuse"),
        site: str = Form("nbu_lounge"),
    ) -> RedirectResponse:
        data = draft_get(draft_id)
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
                clear_decode_state(data)
                data["decode_error"] = (
                    "No reusable audio artifacts found. Decode audio artifacts to continue."
                )
                draft_set(draft_id, data)
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/decode", status_code=303
                )
            mark_decode_ready(data, choice="reuse")
            draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        err = rebuild_decode_artifacts_for_draft(data, site_value=site_value)
        if err:
            data["decode_error"] = err
            draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/decode", status_code=303)
        draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @router.get("/wizard/{draft_id}/select", response_class=HTMLResponse)
    def wizard_select(request: Request, draft_id: int) -> HTMLResponse:
        data = draft_get(draft_id)
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
            output_json = resolve_timestamp_output_path(data)
            return templates.TemplateResponse(
                "wizard_timestamp_select.html",
                {
                    "request": request,
                    "draft_id": draft_id,
                    "site": data.get("site", "nbu_lounge"),
                    "output_json": output_json or "",
                    "error": data.get("select_error"),
                    **wizard_flow_context(data, step=4),
                },
            )
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)
        if not is_decode_ready(data):
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
                **wizard_flow_context(data, step=5),
            },
        )

    @router.post("/wizard/{draft_id}/select")
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

        data = draft_get(draft_id)
        mode = str(data.get("mode") or "sync")
        data["all_segments"] = False
        data["all_cameras"] = False
        data["segments"] = []
        data["cameras"] = []
        data["selection_mode"] = selection_mode

        data["select_error"] = None
        data["run_error"] = None
        if mode == "audio_timestamp":
            out_path = resolve_timestamp_output_path(data, override=output_json)
            if not out_path:
                data["select_error"] = "Output JSON path is required."
                draft_set(draft_id, data)
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
            draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/summary", status_code=303)

        if not is_decode_ready(data):
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
            draft_set(draft_id, data)
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
            draft_set(draft_id, data)
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

        draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/summary", status_code=303)

    @router.get("/wizard/{draft_id}/run")
    def wizard_run(draft_id: int) -> RedirectResponse:
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @router.post("/wizard/{draft_id}/run")
    def wizard_run_post(draft_id: int) -> RedirectResponse:
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @router.get("/wizard/{draft_id}/summary", response_class=HTMLResponse)
    def wizard_draft_summary(request: Request, draft_id: int) -> HTMLResponse:
        data = draft_get(draft_id)
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
            output_path = resolve_timestamp_output_path(data)
            return templates.TemplateResponse(
                "wizard_timestamp_summary.html",
                {
                    "request": request,
                    "draft_id": draft_id,
                    "args": args,
                    "output_path": output_path,
                    **wizard_flow_context(data, step=5),
                    "back_url": f"/wizard/{draft_id}/select",
                },
            )
        if not is_decode_ready(data):
            return RedirectResponse(url=f"/wizard/{draft_id}/decode", status_code=303)
        run_groups = _build_sync_run_groups(data)
        return templates.TemplateResponse(
            "wizard_summary.html",
            {
                "request": request,
                "draft_id": draft_id,
                "args": args,
                "run_groups": run_groups,
                **wizard_flow_context(data, step=6),
                "back_url": f"/wizard/{draft_id}/select",
            },
        )

    @router.post("/wizard/{draft_id}/start")
    def wizard_start(draft_id: int) -> RedirectResponse:
        data = draft_get(draft_id)
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
            draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        if mode == "audio_timestamp":
            existing_run_id = data.get("timestamp_run_id")
            if isinstance(existing_run_id, int):
                return RedirectResponse(
                    url=f"/audio-timestamp/runs/{existing_run_id}", status_code=303
                )
            output_path = resolve_timestamp_output_path(data)
            args: Dict[str, Any] = {
                "audio_dir": data.get("audio_dir", ""),
                "video_dir": data.get("video_dir", ""),
                "out_dir": data.get("out_dir", ""),
                "site": data.get("site", "nbu_lounge"),
                "timezone": DEFAULT_TZ.key,
                "output_json": output_path,
            }
            run_id = create_timestamp_run(args, output_path=output_path or None)
            data["timestamp_run_id"] = run_id
            draft_set(draft_id, data)
            start_audio_timestamp_job(draft_id, run_id=run_id)
            return RedirectResponse(
                url=f"/audio-timestamp/runs/{run_id}", status_code=303
            )
        if not is_decode_ready(data):
            data["run_error"] = "Decode audio artifacts before starting the run."
            draft_set(draft_id, data)
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
            draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        schedule_at = str(data.get("schedule_at", "") or "").strip()
        now = utc_now_iso()
        try:
            status, scheduled_iso = resolve_schedule(schedule_at)
        except Exception as e:
            data["run_error"] = f"Bad schedule_at: {e}"
            draft_set(draft_id, data)
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

        draft_delete(draft_id)
        if len(created_run_ids) == 1:
            return RedirectResponse(url=f"/runs/{created_run_ids[0]}", status_code=303)
        return RedirectResponse(url="/runs", status_code=303)

    return router
