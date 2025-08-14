#!/usr/bin/env python3
"""
serial_basic_diag_pandas_drop_fmt.py — Pandas-based CLI to analyze discontinuities
with **readable, columnized histograms**.

Differences vs previous version:
- Renamed "backward_step" -> **drop** (kept)
- Histograms show **raw diff** values (kept)
- NEW: Pretty, columnized histogram printing. Control columns via `--hist-cols` (default 2).

Categories (expect +1):
- ok        : diff == +1
- duplicate : diff == 0
- forward   : diff > +1 (we still compute total_missing_ids = sum(diff-1))
- drop      : diff < 0

Usage examples:
  python serial_basic_diag_pandas_drop_fmt.py data.csv                        # column 'serial'
  python serial_basic_diag_pandas_drop_fmt.py data.csv --column frame_id
  python serial_basic_diag_pandas_drop_fmt.py data.csv --hist-cols 1          # one entry per row
  python serial_basic_diag_pandas_drop_fmt.py data.csv --out-text rep.txt --out-json rep.json
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import json
import sys

import pandas as pd

# -----------------------------
# Labels
# -----------------------------
OK = "ok"
DUP = "duplicate"
FWD = "forward_jump"
DROP = "drop"


# -----------------------------
# Data containers
# -----------------------------
@dataclass(frozen=True)
class StepEvent:
    """One non-OK transition ids[i] -> ids[i+1]."""

    index: int
    kind: str
    prev: int
    curr: int
    diff: int


@dataclass
class Analysis:
    n_values: int
    total_steps: int
    expect_step: int
    ok_steps: int
    ok_ratio: float
    counts: Dict[str, int]
    total_missing_ids: int
    forward_diff_hist: Dict[int, int]  # raw positive diffs (> +1)
    drop_diff_hist: Dict[int, int]  # raw negative diffs (< 0)
    longest_ok_segment: Tuple[Optional[int], int]  # (start_index, length_in_steps)
    top_forward_jumps: List[StepEvent]
    top_drops: List[StepEvent]


# -----------------------------
# I/O helpers
# -----------------------------


def load_series_from_csv(path: str, column: str = "serial") -> pd.Series:
    """Load a single integer series from a CSV column using pandas.

    - Accepts any delimiter pandas can infer.
    - Coerces to numeric, drops non-numeric rows.
    - Casts to ints for exact diff behavior.
    """
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
    s = pd.to_numeric(df[column], errors="coerce").dropna().astype("int64")
    if s.size < 2:
        raise ValueError("Fewer than 2 numeric values in the specified column.")
    return s.reset_index(drop=True)


# -----------------------------
# Core analysis
# -----------------------------


def classify_discontinuities(
    ids: Sequence[int], expect_step: int
) -> Tuple[Counter, Counter, Counter, List[StepEvent], int, int]:
    """Walk once through the sequence and classify step types.

    Returns:
        counts: Counter of {OK, DUP, FWD, DROP}
        fwd_hist: histogram of raw positive diffs (> +1)
        drop_hist: histogram of raw negative diffs (< 0)
        events: list of non-OK StepEvent
        ok_steps: count of OK steps
        total_steps: total transitions (len(ids)-1)
    """
    counts: Counter = Counter()
    fwd_hist: Counter = Counter()
    drop_hist: Counter = Counter()
    events: List[StepEvent] = []

    diffs = [ids[i + 1] - ids[i] for i in range(len(ids) - 1)]
    ok_steps = 0

    for i, d in enumerate(diffs):
        prev, curr = ids[i], ids[i + 1]
        if d == expect_step:
            counts[OK] += 1
            ok_steps += 1
            continue
        if d == 0:
            counts[DUP] += 1
            events.append(StepEvent(i, DUP, prev, curr, d))
            continue
        if d > expect_step:
            counts[FWD] += 1
            fwd_hist[d] += 1  # raw diff
            events.append(StepEvent(i, FWD, prev, curr, d))
            continue
        # d < expect_step
        counts[DROP] += 1
        drop_hist[d] += 1  # raw negative diff
        events.append(StepEvent(i, DROP, prev, curr, d))

    return counts, fwd_hist, drop_hist, events, ok_steps, len(diffs)


def longest_ok_span(ids: Sequence[int], expect_step: int) -> Tuple[Optional[int], int]:
    """Return (start_index, length_in_steps) of the longest consecutive OK segment."""
    best_start: Optional[int] = None
    best_len = 0

    cur_start: Optional[int] = None
    cur_len = 0

    for i in range(len(ids) - 1):
        if ids[i + 1] - ids[i] == expect_step:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
            cur_start = None

    return best_start, best_len


def pick_top(events: List[StepEvent], kind: str, top_k: int) -> List[StepEvent]:
    """Pick top-K events of a given kind (largest absolute deviation)."""
    filtered = [e for e in events if e.kind == kind]
    if kind == FWD:
        return sorted(filtered, key=lambda e: e.diff, reverse=True)[:top_k]
    if kind == DROP:
        return sorted(filtered, key=lambda e: e.diff)[:top_k]  # most negative first
    return filtered[:top_k]


def analyze(ids: Sequence[int], expect_step: int = 1, top_k: int = 5) -> Analysis:
    """Run the basic discontinuity analysis and return a structured result."""
    if len(ids) < 2:
        raise ValueError("Need at least two values to analyze.")

    counts, fwd_hist, drop_hist, events, ok_steps, total_steps = (
        classify_discontinuities(ids, expect_step)
    )
    top_forward = pick_top(events, FWD, top_k)
    top_drops = pick_top(events, DROP, top_k)

    # total missing IDs only makes sense for forward jumps
    total_missing_ids = sum((e.diff - expect_step) for e in events if e.kind == FWD)
    longest_span = longest_ok_span(ids, expect_step)

    return Analysis(
        n_values=len(ids),
        total_steps=total_steps,
        expect_step=expect_step,
        ok_steps=ok_steps,
        ok_ratio=(ok_steps / total_steps) if total_steps else 1.0,
        counts=dict(counts),
        total_missing_ids=int(total_missing_ids),
        forward_diff_hist=dict(fwd_hist),
        drop_diff_hist=dict(drop_hist),
        longest_ok_segment=longest_span,
        top_forward_jumps=top_forward,
        top_drops=top_drops,
    )


# -----------------------------
# Pretty printing helpers
# -----------------------------


def chunk(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def format_histogram_grid(d: Dict[int, int], label: str, cols: int = 2) -> str:
    """Render a histogram dict as neat, fixed columns.

    Args:
        d: mapping {diff: count}
        label: section header line
        cols: entries per row (e.g., 1 or 2)
    """
    if not d:
        return ""
    items = [f"{k}:{v}" for k, v in sorted(d.items())]
    rows = chunk(items, max(1, cols))

    # compute width per column
    ncols = max(len(r) for r in rows)
    colw = [0] * ncols
    for r in rows:
        for j, cell in enumerate(r):
            colw[j] = max(colw[j], len(cell))

    lines = [f"{label}:"]
    for r in rows:
        padded = [s.ljust(colw[j]) for j, s in enumerate(r)]
        lines.append("  " + "  ".join(padded))
    return "\n".join(lines)


# -----------------------------
# Presentation
# -----------------------------


def summarize_text(
    res: Analysis, *, include_tops: bool = True, hist_cols: int = 2
) -> str:
    lines: List[str] = []
    add = lines.append

    add(
        f"Values={res.n_values}  Steps={res.total_steps}  "
        f"ok={res.ok_steps} ({res.ok_ratio:.2%})"
    )
    add("Counts: " + ", ".join(f"{k}={v}" for k, v in sorted(res.counts.items())))

    if res.forward_diff_hist:
        add(f"Total missing IDs (from forward jumps): {res.total_missing_ids}")
        add(
            format_histogram_grid(
                res.forward_diff_hist,
                "Forward diff histogram (diff > +1)",
                cols=hist_cols,
            )
        )

    if res.drop_diff_hist:
        add(
            format_histogram_grid(
                res.drop_diff_hist, "Drop diff histogram (diff < 0)", cols=hist_cols
            )
        )

    seg_start, seg_len = res.longest_ok_segment
    add(f"Longest OK segment: start={seg_start}  length={seg_len} steps")

    if include_tops and res.top_forward_jumps:
        add("\nTop forward jumps: (index, prev, curr, diff)")
        for e in res.top_forward_jumps:
            add(f"  @i={e.index:6d}  {e.prev} -> {e.curr}  Δ={e.diff:+d}")

    if include_tops and res.top_drops:
        add("\nTop drops: (index, prev, curr, diff)")
        for e in res.top_drops:
            add(f"  @i={e.index:6d}  {e.prev} -> {e.curr}  Δ={e.diff:+d}")

    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Analyze discontinuities (ok, duplicate, forward, drop) in a monotonically increasing "
            "integer sequence from a CSV column (default column: 'serial')."
        )
    )
    p.add_argument("csv_path", help="Path to the CSV file")
    p.add_argument(
        "--column", default="serial", help="Column name to analyze (default: serial)"
    )
    p.add_argument(
        "--expect-step",
        type=int,
        default=1,
        help="Expected increment per step (default: 1)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top forward/drops to list (default: 5)",
    )
    p.add_argument(
        "--hist-cols",
        type=int,
        default=2,
        help="Histogram entries per row (1 = one per line, 2 = two per line, etc.)",
    )
    p.add_argument(
        "--out-text", default=None, help="Write a text report to this path (optional)"
    )
    p.add_argument(
        "--out-json", default=None, help="Write a JSON report to this path (optional)"
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    try:
        series = load_series_from_csv(args.csv_path, args.column)
        ids = series.astype(int).tolist()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    try:
        result = analyze(ids, expect_step=args.expect_step, top_k=args.top)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 3

    report = summarize_text(
        result, include_tops=True, hist_cols=max(1, int(args.hist_cols))
    )
    print(report)

    if args.out_text:
        try:
            with open(args.out_text, "w", encoding="utf-8") as f:
                f.write(report + "\n")
        except Exception as e:
            print(f"[warn] failed to write text report: {e}", file=sys.stderr)

    if args.out_json:
        try:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2)
        except Exception as e:
            print(f"[warn] failed to write JSON report: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
