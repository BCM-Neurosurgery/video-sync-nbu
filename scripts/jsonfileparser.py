#!/usr/bin/env python3
"""
jsonfileparser.py — minimal parser for the sync pipeline.

What it returns (programmatic use)
----------------------------------
parse_json(path, prefer_serial=None) -> ParsedJSON, with:
  - video_serial   : list[int]        # from chunk_serial_data (chosen camera column), leading -1s back-filled
  - timestamps_ns  : list[int]        # corresponding timestamps (ns) for that same camera column
  - measured_fps   : float            # robust FPS estimate from timestamp diffs
  - plus small bits of context (segment_id, chosen_cam_serial, etc.)

Assumptions
-----------
- JSON has keys: "serials", "timestamps", "chunk_serial_data".
- Shapes are (num_frames x num_cameras). We only keep ONE camera column,
  chosen by:
    1) prefer_serial if provided and present, else
    2) the column with the most non-negative serial IDs.

- Only *leading* -1 values in chunk_serial_data are back-filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import argparse
import json
import statistics


# ----------------- public container -----------------


@dataclass(frozen=True)
class ParsedJSON:
    path: Path
    segment_id: str  # e.g., "TestVideo03062025_20250306_153829"
    camera_serials: Tuple[str, ...]  # all serials in file
    chosen_cam_index: int  # 0-based column used
    chosen_cam_serial: str  # camera serial string (FLIR)
    video_serial: List[int]  # per-frame serial IDs (after leading backfill)
    timestamps_ns: List[int]  # per-frame timestamps (ns) for the same column
    measured_fps: float  # estimated fps from timestamps
    n_frames: int  # len(video_serial) == len(timestamps_ns)


# ----------------- public API -----------------


def parse_json(path: str | Path, prefer_serial: Optional[str] = None) -> ParsedJSON:
    """
    Load a FLIR sidecar JSON and extract the minimal fields needed by the sync pipeline.

    Parameters
    ----------
    path : str | Path
        JSON path.
    prefer_serial : Optional[str]
        If provided and present in 'serials', use that camera column. Otherwise the
        column with the most valid (>=0) serial IDs is used.

    Returns
    -------
    ParsedJSON
    """
    path = Path(path)
    raw = _load_json(path)
    _ensure_keys(raw, ["serials", "timestamps", "chunk_serial_data"], context=str(path))

    serials: List[str] = list(raw["serials"])
    num_cams = len(serials)
    if num_cams == 0:
        raise ValueError("No cameras listed in 'serials'.")

    # Normalize 2D arrays to (n_rows x num_cams)
    timestamps = _normalize_2d(raw["timestamps"], num_cams)
    chunks = _normalize_2d(raw["chunk_serial_data"], num_cams)

    n = min(len(timestamps), len(chunks))
    if n == 0:
        raise ValueError("No frames in JSON (timestamps/chunk_serial_data empty).")
    timestamps = timestamps[:n]
    chunks = chunks[:n]

    # Choose camera column
    col_idx = _choose_column(serials, chunks, prefer_serial)

    # Extract chosen column
    ts_col = _col(timestamps, col_idx)
    ch_col = _col(chunks, col_idx)

    # Convert to ints and backfill leading -1s
    ts_ns = [int(x) for x in ts_col if x is not None]
    if len(ts_ns) != n:
        # Keep it simple: require all timestamps for chosen column present
        raise ValueError("Chosen camera column has missing timestamps.")
    vid_serial = _backfill_leading_minus_ones([_to_int_or_minus1(x) for x in ch_col])

    # FPS estimate (robust median of diffs, fall back to mean if needed)
    diffs = [
        b - a
        for a, b in zip(ts_ns, ts_ns[1:])
        if b is not None and a is not None and b > a
    ]
    if not diffs:
        raise ValueError("Not enough timestamps to estimate FPS.")
    try:
        fps = 1e9 / statistics.median(diffs)
    except statistics.StatisticsError:
        fps = 1e9 / (sum(diffs) / len(diffs))

    seg_id = path.stem  # "TestVideo03062025_20250306_153829"

    return ParsedJSON(
        path=path,
        segment_id=seg_id,
        camera_serials=tuple(serials),
        chosen_cam_index=col_idx,
        chosen_cam_serial=serials[col_idx],
        video_serial=vid_serial,
        timestamps_ns=ts_ns,
        measured_fps=float(fps),
        n_frames=n,
    )


# ----------------- helpers -----------------


def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_keys(d: dict, keys: Sequence[str], *, context: str = ""):
    missing = [k for k in keys if k not in d]
    if missing:
        where = f" in {context}" if context else ""
        raise KeyError(f"Missing keys{where}: {', '.join(missing)}")


def _normalize_2d(
    matrix: Sequence[Sequence[object]], width: int
) -> List[Tuple[Optional[object], ...]]:
    out: List[Tuple[Optional[object], ...]] = []
    for row in matrix:
        r = list(row) if row is not None else []
        if len(r) < width:
            r += [None] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        out.append(tuple(r))
    return out


def _col(matrix: Sequence[Sequence[object]], j: int) -> List[Optional[object]]:
    return [row[j] for row in matrix]


def _choose_column(
    serials: List[str],
    chunks: List[Tuple[Optional[object], ...]],
    prefer: Optional[str],
) -> int:
    if prefer and prefer in serials:
        return serials.index(prefer)
    # Count valid (>=0 int) entries per column; choose the max
    num_cams = len(serials)
    counts = [0] * num_cams
    for col in range(num_cams):
        c = 0
        for row in chunks:
            v = row[col]
            if isinstance(v, int) and v >= 0:
                c += 1
        counts[col] = c
    # Argmax with stable tie-break on lower index
    best = max(range(num_cams), key=lambda i: counts[i])
    return best


def _to_int_or_minus1(x: Optional[object]) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    return -1


def _backfill_leading_minus_ones(col: List[int]) -> List[int]:
    """Back-fill only leading -1s using the first valid value stepping backwards by 1."""
    out = list(col)
    # find first valid index
    first_idx = next((i for i, v in enumerate(out) if v >= 0), None)
    if first_idx is None:
        return out
    first_val = out[first_idx]
    for i in range(first_idx - 1, -1, -1):
        if out[i] == -1:
            out[i] = first_val - (first_idx - i)
        else:
            break
    return out


# ----------------- tiny CLI (optional) -----------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Minimal JSON parser for the sync pipeline."
    )
    p.add_argument("json_path", type=str, help="Path to JSON file.")
    p.add_argument(
        "--prefer-serial",
        type=str,
        default=None,
        help="Prefer this camera serial if present.",
    )
    p.add_argument("--summary", action="store_true", help="Print a short summary.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = _build_argparser()
    args = ap.parse_args(argv)
    pj = parse_json(args.json_path, prefer_serial=args.prefer_serial)
    if args.summary:
        print(
            f"{pj.segment_id}: {pj.n_frames} frames | chosen cam {pj.chosen_cam_serial} "
            f"({pj.chosen_cam_index}) | fps≈{pj.measured_fps:.6f}"
        )


if __name__ == "__main__":
    main()
