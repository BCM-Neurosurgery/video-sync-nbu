#!/usr/bin/env python3
"""
jsonfileparser.py

Tiny, readable parser for Provenza Lab FLIR JSON sidecar files.

What it does
------------
- Loads one JSON file.
- Validates required fields and consistent lengths.
- Normalizes "real_times" into Python datetimes.
- Exposes per-frame rows (timestamps, frame IDs, serial IDs).
- Optionally back-fills leading -1 values in chunk_serial_data (per camera).
- Provides handy summaries and CSV export.

Usage
-----
$ python jsonfileparser.py /path/to/TestVideo03062025_20250306_154133.json --summary
$ python jsonfileparser.py /path/to/file.json --csv out.csv
$ python jsonfileparser.py /path/to/file.json --no-fix-missing

The module can also be imported and used programmatically.

Design notes
------------
Columns in the per-frame arrays are assumed to be ordered the same way as the
top-level "serials" array (one column per camera). This parser enforces shapes
and keeps the association between camera index (0..2) and serial string.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import csv
import json
import math
import statistics

# --------- Data containers ---------


@dataclass(frozen=True)
class CameraMeta:
    """Metadata for one camera."""

    serial: str
    exposure_time_ms: Optional[float] = None
    frame_rate_request_hz: Optional[float] = None
    frame_rate_binning_hz: Optional[float] = None
    info: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FrameRow:
    """One video frame across all cameras."""

    index: int
    real_time: Optional[datetime]  # wall clock time, if present
    frame_id: Optional[int]  # shared per-frame ID (if present)
    timestamps_ns: Tuple[Optional[int], ...]  # len == num_cams
    frame_id_abs: Tuple[Optional[int], ...]  # len == num_cams
    serial_ids: Tuple[
        Optional[int], ...
    ]  # decoded Arduino IDs (chunk_serial_data), len == num_cams
    serial_msg: Tuple[Optional[str], ...]  # raw serial msg strings, len == num_cams


@dataclass
class RecordingJSON:
    """Parsed JSON with per-camera metadata and per-frame rows."""

    path: Path
    serials: Tuple[str, ...]
    cameras: Tuple[CameraMeta, ...]
    rows: List[FrameRow]
    meta_info: Optional[Dict[str, Any]] = None

    # ---------- Constructors ----------
    @classmethod
    def from_file(
        cls, path: Path | str, *, fix_missing_serials: bool = True
    ) -> "RecordingJSON":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Required keys we rely on
        required: Tuple[str, ...] = (
            "serials",
            "real_times",
            "timestamps",
            "frame_id_abs",
            "chunk_serial_data",
            "serial_msg",
        )
        _ensure_keys(raw, required, context=str(path))

        serials: Tuple[str, ...] = tuple(raw["serials"])
        num_cams = len(serials)

        # Optional fields
        exposure_times = raw.get("exposure_times", [None] * num_cams)
        frame_rates_req = raw.get("frame_rates_requested", [None] * num_cams)
        frame_rates_bin = raw.get("frame_rates_binning", [None] * num_cams)
        camera_info_map = raw.get("camera_info", {}) or {}

        # Build camera meta in the exact serial order
        cameras: List[CameraMeta] = []
        for i, s in enumerate(serials):
            cameras.append(
                CameraMeta(
                    serial=s,
                    exposure_time_ms=_get_index_or_none(exposure_times, i),
                    frame_rate_request_hz=_get_index_or_none(frame_rates_req, i),
                    frame_rate_binning_hz=_get_index_or_none(frame_rates_bin, i),
                    info=camera_info_map.get(s),
                )
            )

        # Normalize arrays and shapes
        real_times: List[Optional[datetime]] = _parse_datetimes(
            raw.get("real_times", [])
        )
        timestamps: List[Tuple[Optional[int], ...]] = _normalize_2d(
            raw["timestamps"], num_cams
        )
        frame_id_abs: List[Tuple[Optional[int], ...]] = _normalize_2d(
            raw["frame_id_abs"], num_cams
        )
        serial_msg: List[Tuple[Optional[str], ...]] = _normalize_2d(
            raw["serial_msg"], num_cams
        )

        # Some files also carry a per-frame shared "frame_id"
        frame_id_shared: List[Optional[int]] = raw.get("frame_id") or [None] * len(
            real_times
        )
        # Arduino per-camera serial IDs
        chunk_serial_data: List[Tuple[Optional[int], ...]] = _normalize_2d(
            raw["chunk_serial_data"], num_cams
        )

        # Shapes must match
        n = _common_length(
            real_times,
            timestamps,
            frame_id_abs,
            serial_msg,
            chunk_serial_data,
            frame_id_shared,
        )
        if n == 0:
            raise ValueError("No frames detected in JSON.")

        # Align lengths  (truncate overly-long arrays to the shortest length to be robust)
        real_times = real_times[:n]
        timestamps = timestamps[:n]
        frame_id_abs = frame_id_abs[:n]
        serial_msg = serial_msg[:n]
        chunk_serial_data = chunk_serial_data[:n]
        frame_id_shared = frame_id_shared[:n]

        # Optionally back-fill leading -1s in chunk_serial_data per camera
        if fix_missing_serials:
            chunk_serial_data = _backfill_leading_minus_ones(chunk_serial_data)

        # Build row objects
        rows: List[FrameRow] = []
        for i in range(n):
            rows.append(
                FrameRow(
                    index=i,
                    real_time=real_times[i] if i < len(real_times) else None,
                    frame_id=frame_id_shared[i] if i < len(frame_id_shared) else None,
                    timestamps_ns=timestamps[i],
                    frame_id_abs=frame_id_abs[i],
                    serial_ids=chunk_serial_data[i],
                    serial_msg=serial_msg[i],
                )
            )

        return cls(
            path=path,
            serials=serials,
            cameras=tuple(cameras),
            rows=rows,
            meta_info=raw.get("meta_info"),
        )

    # ---------- Convenience ----------
    def fps_estimate(self) -> List[Optional[float]]:
        """Estimate FPS per camera from timestamp deltas (nanoseconds)."""
        num_cams = len(self.serials)
        per_cam: List[Optional[float]] = []
        for cam in range(num_cams):
            ns = [
                r.timestamps_ns[cam]
                for r in self.rows
                if r.timestamps_ns[cam] is not None
            ]
            if len(ns) < 2:
                per_cam.append(None)
                continue
            diffs = [
                b - a
                for a, b in zip(ns, ns[1:])
                if b is not None and a is not None and b > a
            ]
            if not diffs:
                per_cam.append(None)
                continue
            mean_ns = statistics.mean(diffs)
            per_cam.append(1e9 / mean_ns if mean_ns > 0 else None)
        return per_cam

    def to_csv(self, out_path: Path | str) -> Path:
        """Write a wide CSV with one row per frame."""
        out = Path(out_path)
        num_cams = len(self.serials)
        # Prepare headers
        headers = [
            "index",
            "real_time",
            "frame_id",
        ]
        for cam in range(num_cams):
            s = self.serials[cam]
            headers += [
                f"ts_ns_{s}",
                f"frame_id_abs_{s}",
                f"serial_id_{s}",
                f"serial_msg_{s}",
            ]

        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in self.rows:
                row = [r.index, _fmt_dt(r.real_time), r.frame_id]
                for cam in range(num_cams):
                    row += [
                        r.timestamps_ns[cam],
                        r.frame_id_abs[cam],
                        r.serial_ids[cam],
                        r.serial_msg[cam],
                    ]
                w.writerow(row)
        return out

    # ---------- Pretty printing ----------
    def summary(self) -> str:
        """Human-friendly one-paragraph summary."""
        n = len(self.rows)
        cams = ", ".join(self.serials)
        fps = self.fps_estimate()
        fps_str = ", ".join(
            (
                f"{self.serials[i]}: {fps[i]:.3f} Hz"
                if fps[i]
                else f"{self.serials[i]}: n/a"
            )
            for i in range(len(self.serials))
        )
        start = _fmt_dt(self.rows[0].real_time) if self.rows else "n/a"
        stop = _fmt_dt(self.rows[-1].real_time) if self.rows else "n/a"
        return (
            f"{n} frames across {len(self.serials)} cameras [{cams}]. "
            f"Real-time start: {start}, end: {stop}. "
            f"FPS (est.): {fps_str}."
        )


# --------- Helpers ---------


def _ensure_keys(d: Dict[str, Any], keys: Iterable[str], *, context: str = "") -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        where = f" in {context}" if context else ""
        raise KeyError(f"Missing keys{where}: {', '.join(missing)}")


def _get_index_or_none(seq: Sequence[Any], i: int) -> Optional[Any]:
    try:
        return seq[i]
    except Exception:
        return None


def _normalize_2d(
    matrix: Sequence[Sequence[Any]], width: int
) -> List[Tuple[Optional[Any], ...]]:
    """Ensure a list of rows, each a tuple of length == width, coercing missing entries to None."""
    out: List[Tuple[Optional[Any], ...]] = []
    for row in matrix:
        if row is None:
            out.append(tuple([None] * width))
        else:
            # Coerce to list, then pad/truncate
            r = list(row)
            if len(r) < width:
                r = r + [None] * (width - len(r))
            elif len(r) > width:
                r = r[:width]
            out.append(tuple(r))
    return out


def _parse_datetimes(strings: Sequence[str]) -> List[Optional[datetime]]:
    out: List[Optional[datetime]] = []
    for s in strings:
        if not s:
            out.append(None)
            continue
        # Accept "YYYY-MM-DD HH:MM:SS.mmm" and "YYYY-MM-DD HH:MM:SS" (fallback)
        try:
            out.append(datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f"))
        except ValueError:
            try:
                out.append(datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                out.append(None)
    return out


def _common_length(*arrays: Sequence[Any]) -> int:
    """Return the minimum length among sequences (ignoring Nones) to keep things robust."""
    lens = [len(a) for a in arrays if a is not None]
    return min(lens) if lens else 0


def _fmt_dt(dt: Optional[datetime]) -> str:
    return (
        dt.isoformat(sep=" ", timespec="milliseconds")
        if isinstance(dt, datetime)
        else ""
    )


def _backfill_leading_minus_ones(
    matrix: List[Tuple[Optional[int], ...]],
) -> List[Tuple[Optional[int], ...]]:
    """Fill leading -1s per camera using the first valid value and stepping backwards by 1 per frame.
    Non-leading -1s are left as-is.
    """
    if not matrix:
        return matrix
    num_cams = len(matrix[0])
    cols = [[row[c] for row in matrix] for c in range(num_cams)]

    for c in range(num_cams):
        col = cols[c]
        # Find first index with a valid (>= 0) integer
        first_idx = next(
            (i for i, v in enumerate(col) if isinstance(v, int) and v >= 0), None
        )
        if first_idx is None:
            # Nothing to do for this camera
            continue
        first_val = col[first_idx]
        # Back-fill only the leading -1s before the first_idx
        for i in range(first_idx - 1, -1, -1):
            if col[i] == -1:
                col[i] = first_val - (first_idx - i)
            else:
                break  # stop at the first non -1 we encounter before the valid run

        cols[c] = col

    # Re-assemble rows
    filled = [tuple(cols[c][r] for c in range(num_cams)) for r in range(len(matrix))]
    return filled


# --------- CLI ---------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse a FLIR JSON file."
    )
    p.add_argument("json_path", type=str, help="Path to JSON file.")
    p.add_argument(
        "--csv", type=str, default=None, help="Optional path to write a CSV export."
    )
    p.add_argument(
        "--summary", action="store_true", help="Print a short summary to stdout."
    )
    p.add_argument(
        "--no-fix-missing",
        action="store_true",
        help="Do NOT back-fill leading -1 values in chunk_serial_data.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    rec = RecordingJSON.from_file(
        args.json_path, fix_missing_serials=not args.no_fix_missing
    )

    if args.summary:
        print(rec.summary())

    if args.csv:
        out = rec.to_csv(args.csv)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
