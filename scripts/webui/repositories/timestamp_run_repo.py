from __future__ import annotations

from typing import Any, List, Optional

from scripts.webui.db import get_conn, tx


def create_timestamp_run(
    *,
    created_at: str,
    started_at: str,
    status: str,
    args_json: str,
    output_path: Optional[str] = None,
) -> int:
    conn = get_conn()
    with tx(conn):
        cur = conn.execute(
            """
            INSERT INTO timestamp_runs(created_at, started_at, status, args_json, output_path)
            VALUES(?, ?, ?, ?, ?)
            """,
            (created_at, started_at, status, args_json, output_path),
        )
        return int(cur.lastrowid)


def update_timestamp_run(
    run_id: int,
    *,
    status: Optional[str] = None,
    finished_at: Optional[str] = None,
    output_path: Optional[str] = None,
    records_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    fields = []
    values: List[Any] = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if finished_at is not None:
        fields.append("finished_at=?")
        values.append(finished_at)
    if output_path is not None:
        fields.append("output_path=?")
        values.append(output_path)
    if records_count is not None:
        fields.append("records_count=?")
        values.append(records_count)
    if error is not None:
        fields.append("error=?")
        values.append(error)
    if not fields:
        return
    values.append(run_id)

    conn = get_conn()
    with tx(conn):
        conn.execute(
            f"UPDATE timestamp_runs SET {', '.join(fields)} WHERE id=?", values
        )


def list_timestamp_runs(limit: int = 200):
    conn = get_conn()
    return conn.execute(
        "SELECT * FROM timestamp_runs ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()


def list_timestamp_runs_summary(limit: int = 200):
    conn = get_conn()
    return conn.execute(
        """
        SELECT id, status, finished_at, records_count
        FROM timestamp_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()


def get_timestamp_run(run_id: int):
    conn = get_conn()
    return conn.execute("SELECT * FROM timestamp_runs WHERE id=?", (run_id,)).fetchone()


def get_timestamp_run_status(run_id: int) -> Optional[str]:
    row = get_timestamp_run(run_id)
    if not row:
        return None
    return str(row["status"])


def delete_timestamp_run(run_id: int) -> None:
    conn = get_conn()
    with tx(conn):
        conn.execute("DELETE FROM timestamp_runs WHERE id=?", (run_id,))
