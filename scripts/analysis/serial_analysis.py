#!/usr/bin/env python3
"""
serial_analysis.py — Diagnose discontinuities in integer sequences from a CSV column.

Categories (expect +1):
- ok        : diff == +1
- duplicate : diff == 0                 (adjacent equal values)
- forward   : diff > +1  (we also sum total_missing_ids = sum(diff-1))
- drop      : diff < 0

Extras
- Pretty, columnized histograms (use --hist-cols 1 for one bin per line)
- Optional text/JSON reports
- Reports:
    1) Adjacent-duplicate events (value -> how many times it repeated immediately)
    2) Values that appear in ≥2 non-consecutive groups (a “group” is a maximal block of the same value)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter
from pathlib import Path

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

    # 1) Adjacent duplicates (value repeated immediately)
    duplicate_value_hist: Dict[int, int]  # value -> number of adjacent-duplicate events

    # 2) Non-consecutive group repeats
    # A group is a maximal block of identical values; if a value occurs in ≥2 groups, it's a non-consecutive repeat.
    group_count_by_value: Dict[int, int]  # value -> #groups it appears in
    crossgroup_repeat_values: Dict[int, int]  # value -> #groups (only >=2 groups)
    n_crossgroup_repeat_unique: int  # how many unique values appear in ≥2 groups


# -----------------------------
# I/O helpers
# -----------------------------
def load_series_from_csv(path: str, column: str = "serial") -> pd.Series:
    """Load a single integer series from a CSV column using pandas."""
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
    """Walk once through the sequence and classify step types."""
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


def group_blocks(ids: Sequence[int]) -> List[Tuple[int, int, int]]:
    """
    Return list of groups as (value, start_index, end_index), where each group
    is a maximal block of identical consecutive values.
    """
    if not ids:
        return []
    groups: List[Tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            groups.append((ids[i - 1], start, i - 1))
            start = i
    groups.append((ids[-1], start, len(ids) - 1))
    return groups


def analyze(ids: Sequence[int], expect_step: int = 1, top_k: int = 5) -> Analysis:
    """Run the discontinuity analysis + duplicate stats (adjacent and non-consecutive groups)."""
    if len(ids) < 2:
        raise ValueError("Need at least two values to analyze.")

    counts, fwd_hist, drop_hist, events, ok_steps, total_steps = (
        classify_discontinuities(ids, expect_step)
    )
    top_forward = pick_top(events, FWD, top_k)
    top_drops = pick_top(events, DROP, top_k)
    total_missing_ids = sum((e.diff - expect_step) for e in events if e.kind == FWD)
    longest_span = longest_ok_span(ids, expect_step)

    # 1) Adjacent-duplicate histogram (value -> number of times value repeated immediately)
    duplicate_value_hist = Counter(e.prev for e in events if e.kind == DUP)

    # 2) Non-consecutive group repeats
    groups = group_blocks(ids)
    group_count_by_value: Dict[int, int] = Counter(g[0] for g in groups)
    crossgroup_repeat_values = {v: c for v, c in group_count_by_value.items() if c >= 2}
    n_crossgroup_repeat_unique = len(crossgroup_repeat_values)

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
        duplicate_value_hist=dict(duplicate_value_hist),
        group_count_by_value=dict(group_count_by_value),
        crossgroup_repeat_values=crossgroup_repeat_values,
        n_crossgroup_repeat_unique=n_crossgroup_repeat_unique,
    )


# -----------------------------
# Pretty printing helpers
# -----------------------------
def _chunk(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def format_histogram_grid(d: Dict[int, int], label: str, cols: int = 2) -> str:
    if not d:
        return ""
    items = [f"{k}:{v}" for k, v in sorted(d.items())]
    rows = _chunk(items, max(1, cols))
    # width per column
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

    # --- Quick guide block (prepended doc) ---
    add("Quick guide")
    add("-----------")
    add(f"• Step = ids[i+1] - ids[i]; expected step E = {res.expect_step}.")
    add("• ok           : diff == E    → values increase as expected.")
    add("• duplicate    : diff == 0    → adjacent repeated value (no increase).")
    add("• forward_jump : diff >  E    → skipped/missing values;")
    add("                  Total missing IDs = sum(diff - E) over all forward jumps.")
    add("• drop         : diff <  E    → value decreased (e.g., reset/rollover).")
    add(
        "• Counts       : number of steps in each category (there are N-1 steps for N values)."
    )
    add("")

    # --- Summary numbers ---
    add(
        f"Values={res.n_values}  Steps={res.total_steps}  ok={res.ok_steps} ({res.ok_ratio:.2%})"
    )
    add("Counts: " + ", ".join(f"{k}={v}" for k, v in sorted(res.counts.items())))

    # Adjacent duplicates summary
    if res.counts.get(DUP, 0) > 0:
        add(f"Adjacent duplicate events: {res.counts[DUP]}")
        dup_block = format_histogram_grid(
            res.duplicate_value_hist,
            "Adjacent-duplicate values (value:count)",
            cols=hist_cols,
        )
        if dup_block:
            add(dup_block)

    # Non-consecutive group repeats
    if res.n_crossgroup_repeat_unique:
        add(
            f"Values repeating in non-consecutive groups: {res.n_crossgroup_repeat_unique} unique value(s)"
        )
        cross_block = format_histogram_grid(
            res.crossgroup_repeat_values,
            "Non-consecutive group repeats (value:#groups)",
            cols=hist_cols,
        )
        if cross_block:
            add(cross_block)

    # Forward jumps / missing IDs
    if res.forward_diff_hist:
        add(f"Total missing IDs (from forward jumps): {res.total_missing_ids}")
        add(
            format_histogram_grid(
                res.forward_diff_hist,
                "Forward diff histogram (diff > +1)",
                cols=hist_cols,
            )
        )

    # Drops
    if res.drop_diff_hist:
        add(
            format_histogram_grid(
                res.drop_diff_hist, "Drop diff histogram (diff < 0)", cols=hist_cols
            )
        )

    # Longest OK segment
    seg_start, seg_len = res.longest_ok_segment
    add(f"Longest OK segment: start={seg_start}  length={seg_len} steps")

    # Top events
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
            "integer sequence from a CSV column."
        )
    )
    p.add_argument("path", help="Path to the CSV file")
    p.add_argument(
        "--column", default="serial", help="CSV column name (default: serial)"
    )

    # Analysis/reporting
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
        "--out-text",
        default=None,
        help=(
            "Write a text report to this path. "
            "Default: save next to the CSV with the same name but .txt."
        ),
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Write a JSON report to this path (optional; single series only)",
    )
    return p


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")


def _analyze_and_report_series(
    series: pd.Series,
    *,
    src_desc: str,
    expect_step: int,
    top: int,
    hist_cols: int,
) -> str:
    ids = series.astype(int).tolist()
    result = analyze(ids, expect_step=expect_step, top_k=top)
    header = f"Source → {src_desc}"
    report = (
        header
        + "\n"
        + summarize_text(result, include_tops=True, hist_cols=max(1, int(hist_cols)))
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # CSV mode only
    try:
        series = load_series_from_csv(args.path, args.column)
        report = _analyze_and_report_series(
            series,
            src_desc=f"CSV:{args.column}",
            expect_step=args.expect_step,
            top=args.top,
            hist_cols=args.hist_cols,
        )

        # Default: same directory/name as CSV, but with .txt
        out_text_path = (
            Path(args.out_text)
            if args.out_text
            else Path(args.path).with_suffix(".txt")
        )
        _write_text(out_text_path, report)
        logging.info(f"Serial Analysis written to {out_text_path}")

        if args.out_json:
            ids = series.astype(int).tolist()
            result = analyze(ids, expect_step=args.expect_step, top_k=args.top)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2)
            logging.info(f"JSON report written to {args.out_json}")

        return 0
    except Exception as e:
        logging.error(str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
