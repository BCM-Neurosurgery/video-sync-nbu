from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from scripts.webui.db import get_conn, tx
from scripts.webui.repositories import run_repo


BuildCmd = Callable[[Dict[str, Any]], List[str]]


def resolve_schedule(schedule_at: str) -> Tuple[str, Optional[str]]:
    value = str(schedule_at or "").strip()
    if not value:
        return "queued", None

    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    scheduled_iso = dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    return "scheduled", scheduled_iso


def enqueue_sync_runs(
    *,
    base_args: Dict[str, Any],
    run_groups: List[Dict[str, Any]],
    title: str,
    created_at: str,
    status: str,
    scheduled_at: Optional[str],
    build_cmd: BuildCmd,
    default_time_zone: str,
) -> List[int]:
    logs_dir = Path(".webui") / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    created_run_ids: List[int] = []
    conn = get_conn()
    with tx(conn):
        total_groups = len(run_groups)
        for idx, group in enumerate(run_groups, start=1):
            args = dict(base_args)
            args["target_pairs"] = list(group.get("target_pairs") or [])
            if group.get("mode") == "time":
                args["time_start"] = str(group.get("time_start") or "")
                args["time_end"] = str(group.get("time_end") or "")
                args["time_zone"] = str(group.get("time_zone") or default_time_zone)
            elif group.get("mode") == "sample":
                args["audio_sample_start"] = group.get("audio_sample_start")
                args["audio_sample_end"] = group.get("audio_sample_end")

            run_title = title or "video-sync run"
            if total_groups > 1:
                mode_label = str(group.get("mode") or "manual")
                run_title = f"{run_title} ({mode_label} {idx}/{total_groups})"

            run_id = run_repo.insert_run(
                conn,
                title=run_title,
                created_at=created_at,
                scheduled_at=scheduled_at,
                status=status,
                cwd=str(Path.cwd()),
                cmd_json=json.dumps([]),
                args_json=json.dumps(args),
                log_path=str(logs_dir / "pending.log"),
            )
            args["run_id"] = run_id
            cmd = build_cmd(args)
            created_run_ids.append(run_id)
            log_path = logs_dir / f"run-{run_id}.log"
            run_repo.update_run_command_and_log(
                conn,
                run_id,
                cmd_json=json.dumps(cmd),
                args_json=json.dumps(args),
                log_path=str(log_path),
            )

    return created_run_ids
