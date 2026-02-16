from __future__ import annotations

from typing import Optional

from scripts.webui.db import get_conn, tx


def list_runs(limit: int = 200):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()


def list_runs_summary(limit: int = 200):
    conn = get_conn()
    return conn.execute(
        "SELECT id, status, exit_code FROM runs ORDER BY id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()


def list_non_running_runs():
    conn = get_conn()
    return conn.execute(
        "SELECT id, status, log_path FROM runs WHERE status != 'running'"
    ).fetchall()


def delete_non_running_runs() -> None:
    conn = get_conn()
    with tx(conn):
        conn.execute("DELETE FROM runs WHERE status != 'running'")


def insert_run(
    conn,
    *,
    title: str,
    created_at: str,
    scheduled_at: Optional[str],
    status: str,
    cwd: str,
    cmd_json: str,
    args_json: str,
    log_path: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs(title, created_at, scheduled_at, status, cwd, cmd_json, args_json, log_path)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            title,
            created_at,
            scheduled_at,
            status,
            cwd,
            cmd_json,
            args_json,
            log_path,
        ),
    )
    return int(cur.lastrowid)


def update_run_command_and_log(
    conn,
    run_id: int,
    *,
    cmd_json: str,
    args_json: str,
    log_path: str,
) -> None:
    conn.execute(
        "UPDATE runs SET cmd_json=?, args_json=?, log_path=? WHERE id=?",
        (cmd_json, args_json, log_path, run_id),
    )


def get_run(run_id: int):
    conn = get_conn()
    return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def get_run_log_path(run_id: int) -> Optional[str]:
    row = get_run(run_id)
    if not row:
        return None
    return str(row["log_path"])


def delete_run(run_id: int) -> None:
    conn = get_conn()
    with tx(conn):
        conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
