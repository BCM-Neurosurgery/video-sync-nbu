from __future__ import annotations

from typing import Optional

from scripts.webui.db import get_conn, tx


def create_draft(*, created_at: str, updated_at: str, data_json: str) -> int:
    conn = get_conn()
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO drafts(created_at, updated_at, data_json) VALUES(?, ?, ?)",
            (created_at, updated_at, data_json),
        )
        return int(cur.lastrowid)


def get_draft_data_json(draft_id: int) -> Optional[str]:
    conn = get_conn()
    row = conn.execute(
        "SELECT data_json FROM drafts WHERE id=?", (draft_id,)
    ).fetchone()
    if not row:
        return None
    return str(row["data_json"])


def update_draft_data_json(*, draft_id: int, updated_at: str, data_json: str) -> None:
    conn = get_conn()
    with tx(conn):
        conn.execute(
            "UPDATE drafts SET updated_at=?, data_json=? WHERE id=?",
            (updated_at, data_json, draft_id),
        )


def delete_draft(draft_id: int) -> None:
    conn = get_conn()
    with tx(conn):
        conn.execute("DELETE FROM drafts WHERE id=?", (draft_id,))
