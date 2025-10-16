#!/usr/bin/env python3
"""Given a stitched NEV (absolute path), print the absolute path to its start chunk NEV.

Exit codes:
  0: success (path printed)
  1: stitched NEV not found in DataJoint
  2: start chunk path missing for this stitched NEV
  3: other runtime error
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import datajoint as dj
from scripts.dj.connect import connect
from scripts.dj import settings


def canonical_path(path: Path) -> str:
    """Return a canonical absolute path; if not found, resolve as far as possible."""
    expanded = path.expanduser()
    try:
        return str(expanded.resolve(strict=True))
    except FileNotFoundError:
        return str(expanded.resolve(strict=False))


def relative_path_from_exact(exact: Path, base: Path) -> str:
    """
    Return path string to match how it's stored:
    relative to `base` if possible, otherwise absolute string.
    """
    exact_path = Path(canonical_path(exact))
    base_resolved = base.resolve()
    try:
        return str(exact_path.relative_to(base_resolved))
    except ValueError:
        return str(exact_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="From a stitched NEV path, print the absolute path to the start chunk NEV."
    )
    p.add_argument("stitched_nev", type=Path, help="Absolute path to the stitched NEV.")
    p.add_argument("-u", "--username", help="DataJoint username override.")
    p.add_argument("-p", "--password", help="DataJoint password override.")
    return p


# ---------- core query ----------


def fetch_start_chunk_abs(
    conn: dj.connection.Connection,
    stitched_exact: Path,
    stitched_base: Path,
    chunk_base: Path,
) -> tuple[Path | None, bool]:
    """
    Return (absolute start NEV path, stitched_found_flag).

    stitched_found_flag=True if stitched NEV exists in the Ext_Stitch external table.
    """
    stitched_rel = relative_path_from_exact(stitched_exact, stitched_base)

    # Check stitched presence
    check_cur = conn.query(
        "SELECT 1 FROM emu24_stitch.`~external_Ext_Stitch` WHERE filepath = %s LIMIT 1",
        (stitched_rel,),
    )
    stitched_found = check_cur.fetchone() is not None
    if not stitched_found:
        return None, False

    # Fetch start chunk relative path
    query = [
        "SELECT start_ext.filepath AS start_rel_path",
        "FROM emu24_stitch.`__stitched_chunks` AS s",
        "JOIN emu24_stitch.`~external_Ext_Stitch` AS ext ON s.nev_file = ext.hash",
        "LEFT JOIN emu24_stitch.`__n_s_p_chunks` AS start_chunk ON start_chunk.file = s.start_filename",
        "LEFT JOIN emu24_stitch.`~external_Ext_Chunk` AS start_ext ON start_chunk.nev_file = start_ext.hash",
        "WHERE ext.filepath = %s",
        "LIMIT 1",
    ]
    cur = conn.query(" ".join(query), (stitched_rel,))
    row = cur.fetchone()
    if not row or not row[0]:
        return None, True

    start_rel = row[0]
    start_abs = (Path(chunk_base) / start_rel).resolve(strict=False)
    return start_abs, True


# ---------- main ----------


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    # Let CLI overrides flow through to connect(); we also mirror to env so any later imports see them.
    if args.username:
        os.environ["DJ_USER"] = args.username
    if args.password:
        os.environ["DJ_PASSWORD"] = args.password

    try:
        # This loads .env internally and applies dj.config; uses args/env precedence.
        connect(username=args.username, password=args.password)
        conn = dj.conn()

        stitched_base = Path(settings.STITCHED_PATH)
        stores = settings.DJ_CONFIG_STORES
        chunk_base = Path(stores["Ext_Chunk"]["location"])

        start_abs, stitched_found = fetch_start_chunk_abs(
            conn=conn,
            stitched_exact=args.stitched_nev,
            stitched_base=stitched_base,
            chunk_base=chunk_base,
        )

        if not stitched_found:
            print(
                "ERROR: Stitched NEV not found in DataJoint external store.",
                file=sys.stderr,
            )
            return 1
        if start_abs is None:
            print(
                "ERROR: Start chunk NEV path missing for this stitched NEV.",
                file=sys.stderr,
            )
            return 2

        print(str(start_abs))
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    dj.config["display.limit"] = 50
    sys.exit(main(sys.argv[1:]))
