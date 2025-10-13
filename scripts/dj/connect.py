#!/usr/bin/env python3
"""
connect.py — Standalone DataJoint connector that reads credentials from .env.

Required Python deps:
  - datajoint
  - python-dotenv

Expected .env keys (string values unless noted):
  DJ_HOST        = db.example.org
  DJ_PORT        = 3306                  # int (optional)
  DJ_USER        = your_username
  DJ_PASSWORD    = your_password
  DJ_USE_TLS     = 1                     # 1/0, true/false/on/off (optional)
"""

from __future__ import annotations

import os
from typing import Optional

import datajoint as dj
from dotenv import load_dotenv, find_dotenv


def _parse_bool(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    v = val.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def connect(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: Optional[bool] = None,
):
    """
    Connect to a DataJoint/MySQL backend using values from .env (with optional overrides).

    Order of precedence for each field: explicit function arg → .env → default.

    Returns
    -------
    dj.Connection
        An active DataJoint connection object (reused if already open).

    Raises
    ------
    RuntimeError
        If a connection cannot be established.
    """
    # Load the nearest .env (searches upward from CWD and script dir)
    # NOTE: find_dotenv will return "" if nothing is found; load_dotenv("") is a no-op.
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        # Try from the file location context too
        env_path = find_dotenv(filename=".env", usecwd=False)
    load_dotenv(env_path, override=True)

    # Pull values from .env unless overridden by function args
    host = host or os.getenv("DJ_HOST") or os.getenv("DJ_HOSTNAME") or "localhost"
    port = port if port is not None else _parse_int(os.getenv("DJ_PORT"))
    username = username or os.getenv("DJ_USER")
    password = password or os.getenv("DJ_PASSWORD")
    use_tls = use_tls if use_tls is not None else _parse_bool(os.getenv("DJ_USE_TLS"))

    # Apply to DataJoint config
    cfg = {"database.host": host}
    if port is not None:
        cfg["database.port"] = port
    if username:
        cfg["database.user"] = username
    if password:
        cfg["database.password"] = password
    if use_tls is not None:
        cfg["database.use_tls"] = bool(use_tls)

    dj.config.update(cfg)

    # Establish (or reuse) connection
    conn = dj.conn()

    # Robust connected check across versions
    is_connected_attr = getattr(conn, "is_connected", None)
    if callable(is_connected_attr):
        ok = bool(is_connected_attr())
    elif is_connected_attr is None:
        ok = True
    else:
        ok = bool(is_connected_attr)

    if not conn or not ok:
        raise RuntimeError("Failed to connect to DataJoint; check .env and network.")

    return conn


# Optional quick test if invoked directly:
if __name__ == "__main__":
    c = connect()
    # Simple sanity query
    try:
        res = c.query("SELECT 1 AS ok")
        it = iter(res) if res is not None else iter(())
        next(it, None)
        print("DataJoint connection successful.")
    except Exception as e:
        raise RuntimeError("Connected, but a simple test query failed.") from e
