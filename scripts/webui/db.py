from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


def default_db_path() -> Path:
    env = os.environ.get("VSYNC_WEBUI_DB")
    if env:
        return Path(env).expanduser()
    return Path(".webui") / "webui.db"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


_CONN: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = _connect(default_db_path())
        init_db(_CONN)
    return _CONN


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT,
          created_at TEXT NOT NULL,
          scheduled_at TEXT,
          started_at TEXT,
          finished_at TEXT,
          status TEXT NOT NULL,
          exit_code INTEGER,
          pid INTEGER,
          cancel_requested INTEGER NOT NULL DEFAULT 0,
          cwd TEXT NOT NULL,
          cmd_json TEXT NOT NULL,
          args_json TEXT NOT NULL,
          log_path TEXT NOT NULL,
          error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_runs_status_scheduled
          ON runs(status, scheduled_at);
        CREATE INDEX IF NOT EXISTS idx_runs_created
          ON runs(created_at);

        CREATE TABLE IF NOT EXISTS drafts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          data_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_drafts_updated
          ON drafts(updated_at);

        CREATE TABLE IF NOT EXISTS timestamp_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          started_at TEXT,
          finished_at TEXT,
          status TEXT NOT NULL,
          args_json TEXT NOT NULL,
          output_path TEXT,
          records_count INTEGER,
          error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_timestamp_runs_created
          ON timestamp_runs(created_at);
        CREATE INDEX IF NOT EXISTS idx_timestamp_runs_status
          ON timestamp_runs(status);
        """
    )
    conn.commit()


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
