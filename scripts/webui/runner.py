from __future__ import annotations

import asyncio
import json
import os
import signal
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
    run_id = args.get("run_id")
    if run_id is not None and str(run_id).strip() != "":
        cmd.extend(["--run-id", str(int(run_id))])

    target_pairs = args.get("target_pairs") or []
    if target_pairs:
        for pair in target_pairs:
            cmd.extend(["--target", str(pair)])
    else:
        for seg in args.get("segments") or []:
            cmd.extend(["--segment", str(seg)])
        for cam in args.get("cameras") or []:
            cmd.extend(["--camera", str(cam)])

    if args.get("skip_decode"):
        cmd.append("--skip-decode")
    if args.get("overwrite_clips"):
        cmd.append("--overwrite-clips")
    if args.get("split"):
        cmd.append("--split")
    if args.get("split_overwrite"):
        cmd.append("--split-overwrite")
    if args.get("split_clean"):
        cmd.append("--split-clean")

    split_chunk_seconds = args.get("split_chunk_seconds")
    if split_chunk_seconds:
        cmd.extend(["--split-chunk-seconds", str(int(split_chunk_seconds))])

    time_start = args.get("time_start")
    time_end = args.get("time_end")
    if (time_start is not None and str(time_start).strip() != "") or (
        time_end is not None and str(time_end).strip() != ""
    ):
        add("--time-start", time_start)
        add("--time-end", time_end)
        add("--time-zone", args.get("time_zone") or "America/Chicago")

    audio_sample_start = args.get("audio_sample_start")
    if audio_sample_start is not None and str(audio_sample_start).strip() != "":
        cmd.extend(["--audio-sample-start", str(int(audio_sample_start))])
    audio_sample_end = args.get("audio_sample_end")
    if audio_sample_end is not None and str(audio_sample_end).strip() != "":
        cmd.extend(["--audio-sample-end", str(int(audio_sample_end))])

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
        status: Optional[str] = None
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

            # If the run has not started yet, cancel immediately so it won't be promoted/run.
            if status in {"scheduled", "queued"}:
                conn.execute(
                    "UPDATE runs SET status='canceled', finished_at=? WHERE id=?",
                    (utc_now_iso(), run_id),
                )

        # If currently running, try to terminate immediately (don't wait for next log line).
        if status == "running":
            with self._lock:
                proc = self._running.get(run_id)
            if proc is not None:
                self._terminate(proc)
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
            "SELECT * FROM runs WHERE status='queued' AND cancel_requested=0 ORDER BY created_at ASC LIMIT 1"
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

                popen_kwargs = {}
                if os.name == "posix":
                    # Put the run in its own process group so we can terminate children too.
                    popen_kwargs["start_new_session"] = True
                elif os.name == "nt":
                    # Best-effort: create a new process group on Windows.
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True,
                    **popen_kwargs,
                )
                with self._lock:
                    self._running[run_id] = proc

                with tx(conn):
                    conn.execute(
                        "UPDATE runs SET pid=? WHERE id=?",
                        (int(proc.pid), run_id),
                    )

                assert proc.stdout is not None

                def pump_output() -> None:
                    try:
                        for line in proc.stdout:  # type: ignore[union-attr]
                            logf.write(line.encode("utf-8", errors="replace"))
                            logf.flush()
                    except Exception:
                        # Best-effort log streaming; don't crash runner on read issues.
                        pass

                t = threading.Thread(
                    target=pump_output, name=f"webui-run-{run_id}-stdout", daemon=True
                )
                t.start()

                cancel_logged = False
                while proc.poll() is None:
                    if self._cancel_requested(run_id):
                        if not cancel_logged:
                            logf.write(
                                b"\n[webui] Cancel requested. Stopping process...\n"
                            )
                            logf.flush()
                            cancel_logged = True
                        self._terminate(proc)
                        # After termination attempt, break only when proc exits.
                    time.sleep(0.5)

                # Ensure we drained remaining output (best-effort).
                try:
                    t.join(timeout=2)
                except Exception:
                    pass

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
            if os.name == "posix":
                # Terminate the whole process group (parent + children).
                try:
                    os.killpg(int(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(int(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
            else:
                # Windows: best-effort terminate.
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception:
            pass


async def tail_log_sse(
    path: Path,
    *,
    poll_interval: float = 0.25,
    max_bytes: int = 1024 * 512,
    start_at_end: bool = False,
) -> Iterable[str]:
    path = Path(path)
    pos = 0
    last_emit = time.time()
    initialized = False
    drop_first_line = False
    notice_line: str | None = None

    while True:
        if not path.exists():
            await asyncio.sleep(poll_interval)
            yield "event: status\ndata: waiting_for_log\n\n"
            continue

        try:
            if not initialized:
                try:
                    size = path.stat().st_size
                    if start_at_end:
                        pos = max(0, size)
                        drop_first_line = False
                        notice_line = None
                    elif size > max_bytes:
                        pos = max(0, size - max_bytes)
                        drop_first_line = pos > 0
                        notice_line = (
                            f"... showing last {int(max_bytes / 1024)} KB of log ..."
                        )
                except Exception:
                    pass
                initialized = True
            with path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                if chunk:
                    pos = f.tell()
                    text = chunk.decode("utf-8", errors="replace")
                    # SSE data lines must not contain bare CR; normalize.
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    lines = text.splitlines()
                    if drop_first_line and lines:
                        lines = lines[1:]
                        drop_first_line = False
                    if notice_line:
                        yield f"data: {notice_line}\n\n"
                        notice_line = None
                    for line in lines:
                        # One SSE message per log line.
                        yield f"data: {line}\n\n"
                    last_emit = time.time()
        except Exception:
            # ignore transient read errors
            pass

        # Keep-alive every ~10s to prevent proxies from timing out.
        if time.time() - last_emit > 10:
            yield ": keep-alive\n\n"
            last_emit = time.time()

        await asyncio.sleep(poll_interval)
