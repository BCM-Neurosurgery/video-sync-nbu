#!/usr/bin/env python3
"""
csv_clipper.py — Clip a serial→sample index CSV to the serial window defined by anchors.

Input
-----
1) CSV with columns:
   serial,start_sample,end_sample

2) Anchors JSON (list of dicts), each like:
   {
     "serial": 32948939,
     "audio_sample": 71082053,
     "cam_serial": "23512909",
     "segment_id": "TRBD002_20250806_104707",
     "frame_id": 54001,
     "frame_id_reidx": 0
   }

Behavior
--------
- Validates the anchors JSON contains **exactly one** unique cam_serial and **exactly one**
  unique segment_id.
- Finds the **starting** and **ending** serial by taking the min/max of `serial` over the
  anchors list (integers).
- Clips the input CSV to rows where `serial` ∈ [start_serial, end_serial] (inclusive).
- Writes `<input_stem>-clipped.csv` next to the input by default, or a user-provided output.

Public API
----------
- clip_with_anchors(input_csv: str | Path, anchors_json: str | Path, output_csv: str | Path | None) -> Path

CLI
---
Usage:
    python csv_clipper.py /path/to/in.csv /path/to/anchors.json [-o /path/to/out.csv] [--log-level INFO] [-q]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scripts.log.logutils import configure_standalone_logging, log_context

logger = logging.getLogger(__name__)


# ---- Core -----------------------------------------------------------------------
class CSVClipper:
    """
    Clip a serial index CSV to a serial range derived from an anchors JSON.

    Methods
    -------
    load_anchor_window(path) -> (s_min, s_max, cam_serial, segment_id)
        Read and validate anchors; return the serial window (inclusive) and IDs.
    clip_csv(input_csv, s_min, s_max) -> list[dict]
        Load CSV rows and return only those with serial in [s_min, s_max].
    save_csv(rows, out_path) -> Path
        Save rows with header serial,start_sample,end_sample to out_path.
    run(input_csv, anchors_json, output_csv=None) -> Path
        Orchestrate: read anchors, clip CSV, write output, return output path.
    """

    REQUIRED_CSV_COLS = ("serial", "start_sample", "end_sample")
    REQUIRED_ANCHOR_KEYS = ("serial", "cam_serial", "segment_id")

    # ---- Anchors ----
    def load_anchor_window(self, path: Path) -> Tuple[int, int, str, str]:
        """Return (s_min, s_max, cam_serial, segment_id) after validation."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to read anchors JSON '{path}': {e}") from e

        if not isinstance(data, list) or not data:
            raise ValueError("Anchors JSON must be a non-empty list of dicts.")

        cam_ids = set()
        seg_ids = set()
        serials: List[int] = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Anchor #{i} is not an object.")
            # Validate required keys exist (values may be strings; cast later)
            for k in self.REQUIRED_ANCHOR_KEYS:
                if k not in item:
                    raise ValueError(f"Anchor #{i} missing key '{k}'.")
            cam_ids.add(str(item["cam_serial"]))
            seg_ids.add(str(item["segment_id"]))

            s = item.get("serial")
            try:
                s_int = int(s)
            except Exception:
                raise ValueError(f"Anchor #{i} has non-integer 'serial': {s!r}")
            serials.append(s_int)

        if len(cam_ids) != 1 or len(seg_ids) != 1:
            raise ValueError(
                f"Anchors must contain exactly 1 unique cam_serial and 1 unique segment_id; "
                f"got cam_serials={sorted(cam_ids)}, segment_ids={sorted(seg_ids)}"
            )

        s_min, s_max = min(serials), max(serials)
        cam_serial = next(iter(cam_ids))
        segment_id = next(iter(seg_ids))

        # Stamp this summary with [seg/cam] so it matches the rest of the pipeline
        with log_context(seg=segment_id, cam=cam_serial):
            logger.info(
                "Anchors: cam=%s segment=%s  serial_window=[%d..%d] (%d anchors)",
                cam_serial,
                segment_id,
                s_min,
                s_max,
                len(serials),
            )
        return s_min, s_max, cam_serial, segment_id

    # ---- CSV ----
    def _iter_csv_rows(self, path: Path) -> Iterable[Dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = tuple((h or "").strip() for h in (reader.fieldnames or []))
            if headers != self.REQUIRED_CSV_COLS:
                # allow any order, but require the same set
                if set(headers) != set(self.REQUIRED_CSV_COLS):
                    raise ValueError(
                        f"CSV must have columns {self.REQUIRED_CSV_COLS}, found {headers or 'None'}"
                    )
            for row in reader:
                yield row

    def clip_csv(self, input_csv: Path, s_min: int, s_max: int) -> List[Dict[str, str]]:
        """Return only rows with serial ∈ [s_min, s_max] (inclusive)."""
        kept: List[Dict[str, str]] = []
        n_in = 0
        for row in self._iter_csv_rows(input_csv):
            n_in += 1
            try:
                s = int((row.get("serial") or "").strip())
            except Exception:
                # skip malformed serial rows
                continue
            if s_min <= s <= s_max:
                # keep only the required columns; preserve as strings
                kept.append(
                    {
                        "serial": str(s),
                        "start_sample": (row.get("start_sample") or "").strip(),
                        "end_sample": (row.get("end_sample") or "").strip(),
                    }
                )
        logger.info("Clipped %d/%d rows within serial window.", len(kept), n_in)
        return kept

    def save_csv(self, rows: List[Dict[str, str]], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.REQUIRED_CSV_COLS))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote clipped CSV → %s", out_path.name)
        return out_path

    # ---- Orchestrator ----
    def run(
        self, input_csv: Path, anchors_json: Path, output_csv: Optional[Path] = None
    ) -> Path:
        s_min, s_max, cam_serial, segment_id = self.load_anchor_window(anchors_json)

        # Stamp all subsequent logs with [seg/cam]
        with log_context(seg=segment_id, cam=cam_serial):
            rows = self.clip_csv(input_csv, s_min, s_max)
            if output_csv is None:
                output_csv = input_csv.with_name(f"{input_csv.stem}-clipped.csv")
            return self.save_csv(rows, output_csv)


# ---- Public API -----------------------------------------------------------------
def clip_with_anchors(
    input_csv: str | Path,
    anchors_json: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    """
    Programmatic API to clip a CSV using an anchors JSON.

    Parameters
    ----------
    input_csv : str | Path
        Path to CSV with columns (serial,start_sample,end_sample).
    anchors_json : str | Path
        Path to anchors JSON (see module docstring).
    output_csv : str | Path | None
        Destination CSV path. If None, writes next to input with '-clipped' suffix.

    Returns
    -------
    Path
        Output CSV path.
    """
    clipper = CSVClipper()
    in_p = Path(input_csv)
    an_p = Path(anchors_json)
    out_p = Path(output_csv) if output_csv is not None else None
    return clipper.run(in_p, an_p, out_p)


# ---- CLI ------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csvclipper",
        description="Clip a serial→sample CSV to the serial window defined by anchors.",
    )
    p.add_argument("csv", help="Input CSV (serial,start_sample,end_sample).")
    p.add_argument("anchors", help="Anchors JSON (single cam_serial & segment_id).")
    p.add_argument(
        "-o",
        "--out",
        dest="out",
        help="Output CSV path. Defaults to <input_stem>-clipped.csv next to input.",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="Only print warnings and errors."
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (standalone only; ignored when called from driver).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Standalone console logging (no-op under the driver).
    level = "WARNING" if args.quiet else args.log_level
    configure_standalone_logging(level, seg="-", cam="-")

    try:
        _ = clip_with_anchors(args.csv, args.anchors, args.out)
        return 0
    except Exception as e:
        logger.error("%s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
