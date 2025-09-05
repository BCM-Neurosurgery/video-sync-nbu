#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mergecsv.py — Stream-merge many per-chunk serial CSVs into one canonical CSV

What it does
------------
- Reads lots of small CSVs produced per split WAV (columns: serial,start_sample,end_sample).
- Concatenates them in **global timeline order** (by default using natural filename sort;
  or use --manifest to respect the splitter manifest order).
- Writes a single merged CSV (default: **gzip-compressed**) efficiently:
  - **Streaming**: constant memory; never loads all rows at once.
  - **De-duplication**: removes exact duplicates at chunk boundaries
    (same serial, start_sample, end_sample).
  - **Monotonicity check**: warns if an input file appears out of order.

Public API
----------
merge_split_csvs(split_dir, outdir, pattern, manifest, output_name, gzip_output,
                 dedupe, tolerance_samples, logger) -> Path

CLI
---
python -m scripts.merge.mergecsv /path/to/split_csvs \
  --outdir /path/to/output \
  [--pattern "*-03-[0-9][0-9][0-9].csv"] \
  [--manifest /path/to/<stem>_manifest.json] \
  [--output-name raw.csv.gz] \
  [--no-gzip] \
  [--no-dedupe] \
  [--tolerance-samples 0] \
  [-v|-vv] [--seg SEG] [--cam CAM]

Notes
-----
- If you pass --manifest (from mp3split), files are merged in manifest segment order.
- Output defaults to gzip (raw.csv.gz). Use --no-gzip to write plain CSV (raw.csv).
- The merged CSV is ready for existing pipeline (gapfill → filter → anchors → ...).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from scripts.log.logutils import configure_standalone_logging, log_context

__all__ = ["merge_split_csvs", "main"]

log = logging.getLogger("merge")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _natural_key(path: Path) -> Tuple:
    """
    Natural sort key: splits digits and text so '...-2.csv' < '...-10.csv'.
    """
    s = path.name
    parts = re.split(r"(\d+)", s)
    key: List[object] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def _iter_csv_rows(csv_path: Path) -> Iterator[Tuple[int, int, int]]:
    """
    Yield (serial, start_sample, end_sample) rows from a CSV (.csv or .csv.gz).
    Skips a header row if present (first column == 'serial', case-insensitive).
    """
    opener = gzip.open if csv_path.suffix == ".gz" else open
    mode = "rt"
    with opener(csv_path, mode, newline="") as fh:
        r = csv.reader(fh)
        first = True
        for row in r:
            if not row:
                continue
            if first and str(row[0]).strip().lower() == "serial":
                first = False
                continue
            first = False
            try:
                serial = int(row[0])
                start = int(row[1])
                end = int(row[2])
            except Exception as e:
                raise ValueError(f"Bad row in {csv_path}: {row!r} ({e})")
            yield (serial, start, end)


def _resolve_inputs(
    split_dir: Path, pattern: str, manifest: Optional[Path]
) -> List[Path]:
    """
    Build the ordered list of CSV input files.
    - If manifest given, follow its 'segments[].file' order (existing files only).
    - Otherwise, natural-sort by filename.
    """
    if manifest:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        segs = data.get("segments", []) or []
        ordered: List[Path] = []
        for seg in segs:
            fname = str(seg.get("file", "")).strip()
            if not fname:
                continue
            p = split_dir / fname
            if p.exists():
                ordered.append(p)
        if ordered:
            return ordered
        # Fall through to glob if manifest had no match (warn)
        log.warning(
            "Manifest provided but matched no CSVs in %s; falling back to glob.",
            split_dir,
        )

    files = sorted(split_dir.glob(pattern), key=_natural_key)
    return [p for p in files if p.is_file()]


def _rows_are_dupes(
    last: Tuple[int, int, int] | None,
    cur: Tuple[int, int, int],
    tol: int,
) -> bool:
    """
    Decide if 'cur' is a duplicate of 'last' within tolerance.
    Exact match if tol == 0; otherwise allow |start| and |end| diffs <= tol.
    """
    if last is None:
        return False
    ls, lstart, lend = last
    cs, cstart, cend = cur
    if ls != cs:
        return False
    if tol <= 0:
        return (lstart == cstart) and (lend == cend)
    return abs(lstart - cstart) <= tol and abs(lend - cend) <= tol


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def merge_split_csvs(
    split_dir: str | Path,
    outdir: str | Path,
    *,
    pattern: str = "*-03-[0-9][0-9][0-9].csv",
    manifest: str | Path | None = None,
    output_name: str = "raw.csv.gz",
    gzip_output: bool = True,
    dedupe: bool = True,
    tolerance_samples: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Merge many per-chunk serial CSVs into one canonical CSV (optionally gzip-compressed).

    Parameters
    ----------
    split_dir : str | Path
        Directory containing the per-chunk CSVs (one per split WAV).
    outdir : str | Path
        Directory to write the merged CSV.
    pattern : str, default "*-03-[0-9][0-9][0-9].csv"
        Glob used to discover input CSVs when no manifest is provided.
    manifest : str | Path | None
        Optional splitter manifest JSON. If provided, input files are ordered by
        segments[].file from the manifest (skipping any that do not exist).
    output_name : str, default "raw.csv.gz"
        Output filename. Use '.gz' extension when gzip_output=True.
    gzip_output : bool, default True
        Compress the merged CSV with gzip.
    dedupe : bool, default True
        Remove duplicate rows at chunk boundaries.
    tolerance_samples : int, default 0
        De-duplication tolerance for start/end samples. 0 means exact match only.
    logger : logging.Logger | None
        Optional logger. If None, uses module-level 'merge' logger.

    Returns
    -------
    Path
        Path to the merged CSV file.
    """
    lg = logger or log
    in_dir = Path(split_dir)
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(in_dir, pattern, Path(manifest) if manifest else None)
    if not inputs:
        raise FileNotFoundError(
            f"No input CSVs found in {in_dir} (pattern={pattern!r})."
        )

    if gzip_output:
        if not output_name.endswith(".gz"):
            output_name = output_name + ".gz"
    else:
        if output_name.endswith(".gz"):
            output_name = output_name[:-3]
    out_path = out_dir / output_name

    lg.info("Merging %d CSV file(s) → %s", len(inputs), out_path.name)
    if manifest:
        lg.info("Using manifest order: %s", Path(manifest).name)

    # Open writer (gzip or plain)
    opener = (
        (lambda p, m: gzip.open(p, m, newline=""))
        if gzip_output
        else (lambda p, m: open(p, m, newline=""))
    )
    with opener(out_path, "wt") as fh_out:
        w = csv.writer(fh_out)
        # Header
        w.writerow(["serial", "start_sample", "end_sample"])

        total_in = 0
        total_out = 0
        skipped_dupes = 0
        last_row: Tuple[int, int, int] | None = None
        last_start_global: int | None = None

        for idx, path in enumerate(inputs, start=1):
            lg.info("Reading %03d/%03d: %s", idx, len(inputs), path.name)

            file_first_start: Optional[int] = None
            file_last_start: Optional[int] = None

            for serial, start, end in _iter_csv_rows(path):
                total_in += 1

                if file_first_start is None:
                    file_first_start = start
                file_last_start = start

                # Monotonicity warnings (file-scope & global)
                if last_start_global is not None and start < last_start_global:
                    lg.warning(
                        "Non-monotonic start_sample detected (%s: %d < %d). "
                        "Are the inputs out of order?",
                        path.name,
                        start,
                        last_start_global,
                    )

                # De-duplication at boundaries
                row = (serial, start, end)
                if dedupe and _rows_are_dupes(last_row, row, tolerance_samples):
                    skipped_dupes += 1
                    continue

                w.writerow([serial, start, end])
                total_out += 1
                last_row = row
                last_start_global = start

            if file_first_start is None:
                lg.warning("File had no data rows: %s", path.name)
            else:
                lg.debug(
                    "File span: start=%d, end=%d (rows=%s)",
                    file_first_start,
                    file_last_start,
                    "unknown",  # per-file count omitted for speed
                )

        lg.info(
            "Merged %d rows, wrote %d rows (skipped %d duplicate%s).",
            total_in,
            total_out,
            skipped_dupes,
            "" if skipped_dupes == 1 else "s",
        )

    return out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream-merge many per-chunk serial CSVs into one canonical CSV (optionally gzip-compressed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("split_dir", type=Path, help="Directory containing per-chunk CSVs.")
    p.add_argument(
        "--outdir", required=True, type=Path, help="Directory for the merged CSV."
    )
    p.add_argument(
        "--pattern",
        default="*-03-[0-9][0-9][0-9].csv",
        help="Glob for input CSVs (used when --manifest is not provided).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        help="Optional splitter manifest JSON to define input order (segments[].file).",
    )
    p.add_argument(
        "--output-name",
        default="raw.csv",
        help="Output file name (add .gz automatically if --gzip is on and name lacks it).",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--no-gzip", dest="gzip_output", action="store_false", help="Write plain CSV."
    )
    g.add_argument(
        "--gzip",
        dest="gzip_output",
        action="store_true",
        help="Write gzip-compressed CSV.",
    )
    p.set_defaults(gzip_output=True)

    g2 = p.add_mutually_exclusive_group()
    g2.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Do not remove duplicate rows.",
    )
    g2.add_argument(
        "--dedupe",
        dest="dedupe",
        action="store_true",
        help="Remove duplicate rows at boundaries.",
    )
    p.set_defaults(dedupe=True)

    p.add_argument(
        "--tolerance-samples",
        type=int,
        default=0,
        help="De-dup tolerance for start/end (samples). 0 = exact duplicates only.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v / -vv).",
    )
    p.add_argument("--seg", default="-", help="Segment ID for log stamping.")
    p.add_argument("--cam", default="-", help="Camera serial for log stamping.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Verbosity mapping
    level = logging.INFO if args.verbose <= 1 else logging.DEBUG
    configure_standalone_logging(level=level, seg=args.seg, cam=args.cam)

    with log_context(seg=args.seg, cam=args.cam):
        try:
            out_path = merge_split_csvs(
                split_dir=args.split_dir,
                outdir=args.outdir,
                pattern=args.pattern,
                manifest=args.manifest,
                output_name=args.output_name,
                gzip_output=args.gzip_output,
                dedupe=args.dedupe,
                tolerance_samples=args.tolerance_samples,
                logger=log,
            )
            log.info("Wrote %s", out_path)
        except Exception as e:
            log.error("%s", e)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
