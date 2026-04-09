#!/usr/bin/env python3
"""Dump all stitched -> first-chunk NEV mappings from DataJoint to JSON.

Run once on Elias (admin) to generate the lookup file that lets other users
run cli_emu_time without DataJoint/MySQL access.

Usage:
    python -m scripts.dj.dump_first_chunk_map [-o OUTPUT_PATH]

Output defaults to <STITCHED_PATH>/first_chunk_map.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import datajoint as dj

from scripts.dj import settings
from scripts.dj.connect import connect


def build_parser() -> argparse.ArgumentParser:
    default_output = Path(settings.STITCHED_PATH) / "first_chunk_map.json"
    p = argparse.ArgumentParser(
        description="Dump stitched -> first-chunk NEV mappings to JSON."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output path (default: {default_output}).",
    )
    p.add_argument("-u", "--username", help="DataJoint username override.")
    p.add_argument("-p", "--password", help="DataJoint password override.")
    return p


def fetch_all_mappings(conn: dj.connection.Connection) -> dict[str, str]:
    """Query all stitched NEV -> first-chunk NEV relative-path mappings.

    Uses the same table joins as ``fetch_start_chunk_abs`` in
    ``find_first_nev_from_stitched.py`` but fetches all rows.

    Returns
    -------
    dict[str, str]
        ``{stitched_rel_posix: chunk_rel_posix}``
    """
    query = " ".join(
        [
            "SELECT ext.filepath AS stitched_rel,",
            "       start_ext.filepath AS chunk_rel",
            "FROM emu24_stitch.`__stitched_chunks` AS s",
            "JOIN emu24_stitch.`~external_Ext_Stitch` AS ext",
            "     ON s.nev_file = ext.hash",
            "LEFT JOIN emu24_stitch.`__n_s_p_chunks` AS start_chunk",
            "     ON start_chunk.file = s.start_filename",
            "LEFT JOIN emu24_stitch.`~external_Ext_Chunk` AS start_ext",
            "     ON start_chunk.nev_file = start_ext.hash",
            "WHERE start_ext.filepath IS NOT NULL",
        ]
    )
    cur = conn.query(query)
    mapping: dict[str, str] = {}
    for stitched_rel, chunk_rel in cur:
        key = PurePosixPath(stitched_rel).as_posix()
        value = PurePosixPath(chunk_rel).as_posix()
        mapping[key] = value
    return mapping


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.username:
        os.environ["DJ_USER"] = args.username
    if args.password:
        os.environ["DJ_PASSWORD"] = args.password

    try:
        connect(username=args.username, password=args.password)
        conn = dj.conn()
    except Exception as exc:
        print(f"ERROR: Failed to connect to DataJoint: {exc}", file=sys.stderr)
        return 2

    try:
        mapping = fetch_all_mappings(conn)
    except Exception as exc:
        print(f"ERROR: Query failed: {exc}", file=sys.stderr)
        return 2

    if not mapping:
        print("WARNING: No mappings found in database.", file=sys.stderr)

    payload = {
        "_meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "stitched_base": settings.STITCHED_PATH,
            "chunk_base": settings.DATALAKE_PATH,
            "count": len(mapping),
        },
        "map": mapping,
    }

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(mapping)} mappings to {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
