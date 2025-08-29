#!/usr/bin/env python3
"""
anchor_analysis.py — Analyze anchors JSON (serial ↔ audio_sample ↔ frame_id)

Expectation
-----------
Anchors JSON must contain EXACTLY ONE unique segment_id and EXACTLY ONE unique cam_serial.
If not, the script raises an error.

Input schema (list of dicts)
----------------------------
{
  "serial": 32948939,
  "audio_sample": 79759336,
  "cam_serial": "18486634",
  "segment_id": "TRBD002_20250806_104707",
  "frame_id": 0
}

What it does
------------
1) GLOBAL serial analysis (expect_step=1) using unique, sorted serials across anchors.
2) FRAME-ID analysis (expect_step=1) using unique, sorted frame_ids (single segment & camera).

Output (default)
----------------
Writes a single formatted text report next to the JSON:
  <anchors_json_stem>.txt

CLI
---
python -m scripts.analysis.anchor_analysis /path/to/anchors.json
  [--out-text /path/to/report.txt]
  [--hist-cols 2]
  [--top 5]
  [--log-level INFO]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, List, Set, Optional, Union

import pandas as pd

from scripts.analysis.serial_analysis import analyze, summarize_text
from scripts.log.logutils import configure_standalone_logging, log_context

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"serial", "audio_sample", "cam_serial", "segment_id", "frame_id"}


def load_anchors(path: Path) -> List[dict]:
    """Load anchors JSON and validate minimal schema."""
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read anchors JSON: {path}\n{e}") from e
    if not isinstance(data, list):
        raise ValueError("Anchors JSON must be a list of objects.")

    ok: List[dict] = []
    bad_idx: List[int] = []
    for i, row in enumerate(data):
        if isinstance(row, dict) and REQUIRED_KEYS.issubset(row.keys()):
            ok.append(row)
        else:
            bad_idx.append(i)
    if bad_idx:
        logger.warning(
            "Skipped %d malformed anchor(s): indices=%s", len(bad_idx), bad_idx[:10]
        )
    if not ok:
        raise ValueError("No valid anchors found after validation.")
    return ok


def enforce_single_segment_and_cam(anchors: List[dict]) -> tuple[str, str]:
    """Require exactly one unique segment_id and one unique cam_serial."""
    segs: Set[str] = {str(a["segment_id"]) for a in anchors}
    cams: Set[str] = {str(a["cam_serial"]) for a in anchors}
    if len(segs) != 1 or len(cams) != 1:
        raise ValueError(
            f"Expected exactly 1 unique segment_id and 1 unique cam_serial; "
            f"found segment_id={sorted(segs)} (n={len(segs)}), cam_serial={sorted(cams)} (n={len(cams)})."
        )
    return next(iter(segs)), next(iter(cams))


def unique_sorted_ints(values: Iterable[Any]) -> List[int]:
    """Coerce to int, drop None/NaN, return sorted unique list."""
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().astype("int64")
    return sorted(set(int(v) for v in s.tolist()))


def build_serial_report(anchors: List[dict], *, hist_cols: int, top: int) -> str:
    """Global serial analysis using unique, sorted serials."""
    serial_values = unique_sorted_ints(a.get("serial") for a in anchors)
    if len(serial_values) < 2:
        raise ValueError("Need at least two unique serial values for analysis.")
    res = analyze(serial_values, expect_step=1, top_k=top)
    header = [
        "=== GLOBAL SERIAL ANALYSIS (unique sorted serials) ===",
        f"N unique serials: {len(serial_values)}",
        "",
    ]
    return "\n".join(header) + summarize_text(
        res, include_tops=True, hist_cols=max(1, hist_cols)
    )


def build_frame_report(anchors: List[dict], *, hist_cols: int, top: int) -> str:
    """Frame-ID analysis (single segment & cam) using unique, sorted frame_ids."""
    fids = unique_sorted_ints(a.get("frame_id") for a in anchors)
    if len(fids) < 2:
        raise ValueError("Need at least two unique frame IDs for analysis.")
    res = analyze(fids, expect_step=1, top_k=top)
    header = [
        "=== FRAME-ID ANALYSIS (unique sorted frame_ids) ===",
        f"N unique frame_ids: {len(fids)}",
        "",
    ]
    return "\n".join(header) + summarize_text(
        res, include_tops=True, hist_cols=max(1, hist_cols)
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyze anchors JSON with exactly one segment_id and one cam_serial. "
            "Writes a formatted text report next to the JSON by default."
        )
    )
    p.add_argument("path", help="Path to anchors JSON file")
    p.add_argument(
        "--out-text", default=None, help="Optional path to write formatted text report"
    )
    p.add_argument(
        "--hist-cols",
        type=int,
        default=2,
        help="Histogram entries per row for pretty print",
    )
    p.add_argument("--top", type=int, default=5, help="Top forward/drops to list")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (standalone only; ignored when called from driver)",
    )
    return p


# -----------------------------
# Public API
# -----------------------------
def analyze_anchors_file(
    anchors_json: Union[str, Path],
    *,
    out_text: Optional[Union[str, Path]] = None,
    hist_cols: int = 2,
    top: int = 5,
) -> Path:
    """
    Analyze an anchors JSON and write a formatted text report.

    Parameters
    ----------
    anchors_json : str | Path
        Path to the anchors JSON file.
    out_text : str | Path | None, default None
        Where to write the formatted text report. If None, uses "<json_stem>.txt"
        next to the input.
    hist_cols : int, default 2
        Histogram entries per row for pretty printing.
    top : int, default 5
        Number of top forward/drops to list.

    Returns
    -------
    Path
        Absolute path to the written text report.

    Raises
    ------
    FileNotFoundError
        If the anchors_json path does not exist.
    ValueError, RuntimeError
        If the JSON is malformed, schema is invalid, or analysis fails.
    """
    in_path = Path(anchors_json)
    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {in_path}")

    anchors = load_anchors(in_path)
    segment_id, cam_serial = enforce_single_segment_and_cam(anchors)

    with log_context(seg=segment_id, cam=cam_serial):
        serial_txt = build_serial_report(anchors, hist_cols=hist_cols, top=top)
        frame_txt = build_frame_report(anchors, hist_cols=hist_cols, top=top)

        # Compose text report
        report_lines = [
            f"Source → {in_path.name}",
            f"Segment  : {segment_id}",
            f"Camera   : {cam_serial}",
            "",
            serial_txt,
            "",
            frame_txt,
        ]
        full_text = "\n".join(report_lines)

        # Write text
        out_text_path = Path(out_text) if out_text else in_path.with_suffix(".txt")
        out_text_path.parent.mkdir(parents=True, exist_ok=True)
        out_text_path.write_text(full_text.rstrip() + "\n", encoding="utf-8")
        logger.info("Anchor analysis written → %s", out_text_path.name)

        return out_text_path.resolve()


def main(argv: list[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    # Standalone: minimal console logging (no-op under pipeline driver).
    configure_standalone_logging(args.log_level, seg="-", cam="-")

    in_path = Path(args.path)
    if not in_path.exists():
        logger.error("File not found: %s", in_path)
        return 2

    try:
        _ = analyze_anchors_file(
            in_path,
            out_text=args.out_text,
            hist_cols=args.hist_cols,
            top=args.top,
        )
    except Exception as e:
        logger.error(str(e))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
