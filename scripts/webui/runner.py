from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from scripts.webui.db import get_conn, tx
from scripts.webui.models import utc_now_iso


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def build_cli_cmd(args: Dict[str, object]) -> List[str]:
    py = os.environ.get("VSYNC_WEBUI_PYTHON") or sys.executable
    cmd: List[str] = [py, "-m", "scripts.cli.cli_nbu"]

    def add(flag: str, value: Optional[object]) -> None:
        if value is None:
            return
        s = str(value).strip()
        if s == "":
            return
        cmd.extend([flag, s])

    add("--audio-dir", args.get("audio_dir"))
    add("--video-dir", args.get("video_dir"))
    add("--out-dir", args.get("out_dir"))
    add("--site", args.get("site"))
    add("--log-level", args.get("log_level") or "INFO")

    for seg in args.get("segments") or []:
        cmd.extend(["--segment", str(seg)])
    for cam in args.get("cameras") or []:
        cmd.extend(["--camera", str(cam)])

    if args.get("skip_decode"):
        cmd.append("--skip-decode")
    if args.get("split"):
        cmd.append("--split")
    if args.get("split_overwrite"):
        cmd.append("--split-overwrite")
    if args.get("split_clean"):
        cmd.append("--split-clean")

    split_chunk_seconds = args.get("split_chunk_seconds")
    if split_chunk_seconds:
        cmd.extend(["--split-chunk-seconds", str(int(split_chunk_seconds))])

    return cmd


@dataclass
class RunnerConfig:
    max_parallel: int = 1
    poll_interval_sec: float = 1.0


class Runner:
    def __init__(self, *, cfg: RunnerConfig) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._running: Dict[int, subprocess.Popen] = {}

    def start(self) -> None:
        t = threading.Thread(target=self._loop, name="webui-runner", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()

    def request_cancel(self, run_id: int) -> bool:
        conn = get_conn()
        with tx(conn):
            row = conn.execute(
                "SELECT status FROM runs WHERE id=?",
                (run_id,),
            ).fetchone()
            if not row:
                return False
            status = str(row["status"])
            if status not in {"scheduled", "queued", "running"}:
                return False
            conn.execute(
                "UPDATE runs SET cancel_requested=1 WHERE id=?",
                (run_id,),
            )
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                # Keep the runner alive; errors are recorded per-run in _run_one.
                pass
            time.sleep(self.cfg.poll_interval_sec)

    def _tick(self) -> None:
        conn = get_conn()

        # Promote due scheduled runs to queued.
        with tx(conn):
            due = conn.execute(
                "SELECT id, scheduled_at FROM runs WHERE status='scheduled' AND scheduled_at IS NOT NULL"
            ).fetchall()
            now = _utc_now()
            for r in due:
                at = _parse_iso(r["scheduled_at"])
                if at is not None and at <= now:
                    conn.execute(
                        "UPDATE runs SET status='queued' WHERE id=? AND status='scheduled'",
                        (int(r["id"]),),
                    )

        # Start queued runs if we have capacity.
        with self._lock:
            active = len(self._running)
        if active >= self.cfg.max_parallel:
            return

        row = conn.execute(
            "SELECT * FROM runs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return

        run_id = int(row["id"])
        # Mark as running before starting process (best-effort).
        with tx(conn):
            conn.execute(
                "UPDATE runs SET status='running', started_at=?, pid=NULL WHERE id=? AND status='queued'",
                (utc_now_iso(), run_id),
            )

        threading.Thread(
            target=self._run_one,
            args=(run_id,),
            name=f"webui-run-{run_id}",
            daemon=True,
        ).start()

    def _run_one(self, run_id: int) -> None:
        conn = get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return

        args = json.loads(row["args_json"])
        cmd = json.loads(row["cmd_json"])
        log_path = Path(row["log_path"])
        cwd = Path(row["cwd"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        proc: Optional[subprocess.Popen] = None
        try:
            with log_path.open("ab") as logf:
                header = (
                    f"\n==== RUN {run_id} @ {utc_now_iso()} ====\n"
                    f"CWD: {cwd}\n"
                    f"CMD: {' '.join(cmd)}\n"
                ).encode("utf-8", errors="replace")
                logf.write(header)
                logf.flush()

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True,
                )
                with self._lock:
                    self._running[run_id] = proc

                with tx(conn):
                    conn.execute(
                        "UPDATE runs SET pid=? WHERE id=?",
                        (int(proc.pid), run_id),
                    )

                assert proc.stdout is not None
                for line in proc.stdout:
                    logf.write(line.encode("utf-8", errors="replace"))
                    logf.flush()
                    if self._cancel_requested(run_id):
                        self._terminate(proc)

                rc = proc.wait()
                finished = utc_now_iso()
                status = "succeeded" if rc == 0 else "failed"
                if self._cancel_requested(run_id):
                    status = "canceled"
                with tx(conn):
                    conn.execute(
                        "UPDATE runs SET status=?, finished_at=?, exit_code=? WHERE id=?",
                        (status, finished, int(rc), run_id),
                    )
        except Exception as e:
            with tx(conn):
                conn.execute(
                    "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
                    (utc_now_iso(), str(e), run_id),
                )
        finally:
            with self._lock:
                self._running.pop(run_id, None)

    def _cancel_requested(self, run_id: int) -> bool:
        conn = get_conn()
        row = conn.execute(
            "SELECT cancel_requested FROM runs WHERE id=?",
            (run_id,),
        ).fetchone()
        return bool(row and int(row["cancel_requested"]) == 1)

    def _terminate(self, proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is not None:
                return
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


async def tail_log_sse(path: Path, *, poll_interval: float = 0.25) -> Iterable[str]:
    path = Path(path)
    pos = 0
    last_emit = time.time()

    while True:
        if not path.exists():
            await asyncio.sleep(poll_interval)
            yield "event: status\ndata: waiting_for_log\n\n"
            continue

        try:
            with path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                if chunk:
                    pos = f.tell()
                    text = chunk.decode("utf-8", errors="replace")
                    # SSE data lines must not contain bare CR; normalize.
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    for line in text.split("\n"):
                        yield f"data: {line}\n"
                    yield "\n"
                    last_emit = time.time()
        except Exception:
            # ignore transient read errors
            pass

        # Keep-alive every ~10s to prevent proxies from timing out.
        if time.time() - last_emit > 10:
            yield ": keep-alive\n\n"
            last_emit = time.time()

        await asyncio.sleep(poll_interval)
