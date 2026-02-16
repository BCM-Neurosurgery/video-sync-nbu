from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.index.common import DEFAULT_TZ
from scripts.time.find_audio_abs_time import (
    OutputLayout,
    _record_to_payload,
    compute_audio_start_records,
)
from scripts.webui.models import utc_now_iso
from scripts.webui.repositories import timestamp_run_repo
from scripts.webui.services import draft_service


def default_timestamp_output_path(data: Dict[str, Any]) -> str:
    out_dir = str(data.get("out_dir", "")).strip()
    if not out_dir:
        return ""
    try:
        return str(OutputLayout(Path(out_dir)).default_metadata_path)
    except Exception:
        return ""


def resolve_timestamp_output_path(
    data: Dict[str, Any], *, override: Optional[str] = None
) -> str:
    raw = (override or "").strip()
    if not raw:
        raw = str(data.get("timestamp_output_path") or "").strip()
    if raw:
        try:
            return str(Path(raw).expanduser())
        except Exception:
            return raw
    return default_timestamp_output_path(data)


def create_timestamp_run(
    args: Dict[str, Any], *, output_path: Optional[str] = None
) -> int:
    now = utc_now_iso()
    return timestamp_run_repo.create_timestamp_run(
        created_at=now,
        started_at=now,
        status="running",
        args_json=json.dumps(args),
        output_path=output_path,
    )


def update_timestamp_run(
    run_id: int,
    *,
    status: Optional[str] = None,
    finished_at: Optional[str] = None,
    output_path: Optional[str] = None,
    records_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    timestamp_run_repo.update_timestamp_run(
        run_id,
        status=status,
        finished_at=finished_at,
        output_path=output_path,
        records_count=records_count,
        error=error,
    )


def start_audio_timestamp_job(draft_id: int, *, run_id: int) -> None:
    def worker() -> None:
        data = draft_service.get_draft(draft_id)
        try:
            audio_dir = Path(str(data.get("audio_dir", "")).strip()).expanduser()
            video_dir = Path(str(data.get("video_dir", "")).strip()).expanduser()
            out_dir = Path(str(data.get("out_dir", "")).strip()).expanduser()
            site = str(data.get("site") or "nbu_lounge")

            records = compute_audio_start_records(
                audio_dir=audio_dir,
                video_dir=video_dir,
                out_dir=out_dir,
                local_tz=DEFAULT_TZ,
                site=site,
            )
            payload = [_record_to_payload(rec) for rec in records]
            output_path_raw = resolve_timestamp_output_path(data)
            if not output_path_raw:
                raise RuntimeError("Output JSON path is required.")
            output_path = Path(output_path_raw)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            latest = draft_service.get_draft(draft_id)
            latest["timestamp_output_path"] = str(output_path)
            draft_service.set_draft(draft_id, latest)
            update_timestamp_run(
                run_id,
                status="succeeded",
                finished_at=utc_now_iso(),
                output_path=str(output_path),
                records_count=len(payload),
            )
        except Exception as exc:
            update_timestamp_run(
                run_id,
                status="failed",
                finished_at=utc_now_iso(),
                error=str(exc),
            )

    threading.Thread(
        target=worker, name=f"audio-timestamp-{draft_id}", daemon=True
    ).start()
