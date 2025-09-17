#!/usr/bin/env python3
"""
audio_loss_analysis.py — Quantify audio-side loss (START-based JSON) with anomaly prefilter.

Pipeline
--------
1) Load CSV → sort by start/end (original timeline).
2) Prefilter using scripts.fix.audiofilter.AudioFilter (if present) or a faithful fallback:
     - keep first row; anchor = last_kept = first serial
     - drop if cur <= anchor
     - drop if MAX_FWD_DELTA is not None and (cur - last_kept) > MAX_FWD_DELTA
   (Default MAX_FWD_DELTA=200; override via --max-fwd-delta; use None to disable.)
3) Apply a strict monotone keep (left→right keep s if s > last_kept).
4) On the kept stream, each Δserial>1 is a forward gap.
   For each gap:
     P_local = median Δstart from up to W nearby +1 steps (left/right),
               falling back to a global median or 1470 samples.
     missing_samples = max(0, Δserial * P_local - observed_Δstart)
5) Write JSON to <csv_stem>.audio_loss.json.

CLI
---
python audio_loss_analysis.py serials.csv --out-dir OUT \
  [--fs 44100] [--local-window 3] [--top 12] \
  [--prefilter] [--max-fwd-delta 200]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------- CSV I/O ----------------------------- #
def _find_col(df: pd.DataFrame, target: str) -> str:
    t = target.lower()
    for c in df.columns:
        if str(c).strip().lower() == t:
            return c
    raise ValueError(
        f"Missing required column '{target}'. Available: {list(df.columns)}"
    )


def load_serial_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    c_serial = _find_col(df, "serial")
    c_start = _find_col(df, "start_sample")
    try:
        c_end = _find_col(df, "end_sample")
    except ValueError:
        c_end = None

    out = pd.DataFrame(
        {
            "serial": pd.to_numeric(df[c_serial], errors="coerce").astype("Int64"),
            "start_sample": pd.to_numeric(df[c_start], errors="coerce").astype("Int64"),
        }
    )
    out["end_sample"] = (
        pd.to_numeric(df[c_end], errors="coerce").astype("Int64")
        if c_end
        else out["start_sample"]
    )

    out = out.dropna(subset=["serial", "start_sample", "end_sample"]).astype(
        {"serial": "int64", "start_sample": "int64", "end_sample": "int64"}
    )
    # Original timeline order (stable)
    out = out.sort_values(["start_sample", "end_sample"], kind="mergesort").reset_index(
        drop=True
    )
    return out


# ----------------------------- Prefilter (AudioFilter or fallback) ----------------------------- #
def apply_prefilter(
    df: pd.DataFrame, max_fwd_delta: Optional[int]
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """
    Try to use scripts.fix.audiofilter.AudioFilter; if not importable,
    use a faithful fallback implementing the same rules.
    Returns (filtered_df, stats, impl_name).
    """
    # Keep only the three columns to satisfy AudioFilter's strict schema
    df3 = df[["serial", "start_sample", "end_sample"]].copy()

    # Attempt real AudioFilter
    try:
        from scripts.fix.audiofilter import AudioFilter  # type: ignore

        # Configure forward jump bound
        setattr(AudioFilter, "MAX_FWD_DELTA", max_fwd_delta)
        af = AudioFilter()
        filtered = af.filter_csv(df3)
        impl = "scripts.fix.audiofilter.AudioFilter"
    except Exception:
        # Fallback: replicate the class behavior
        filtered = _fallback_audiofilter(df3, max_fwd_delta)
        impl = "fallback_audiofilter"

    stats = {
        "input_rows": int(len(df3)),
        "filtered_rows": int(len(filtered)),
        "dropped_rows": int(len(df3) - len(filtered)),
    }
    return filtered.reset_index(drop=True), stats, impl


def _fallback_audiofilter(
    df3: pd.DataFrame, max_fwd_delta: Optional[int]
) -> pd.DataFrame:
    s = pd.to_numeric(df3["serial"], errors="raise").astype("int64").to_numpy()
    if s.size == 0:
        return df3.copy()

    keep_idx = [0]
    anchor = int(s[0])
    last_kept = int(s[0])
    for i in range(1, s.size):
        cur = int(s[i])
        if cur <= anchor:
            continue
        if max_fwd_delta is not None and (cur - last_kept) > max_fwd_delta:
            continue
        keep_idx.append(i)
        anchor = cur
        last_kept = cur
    return df3.iloc[keep_idx].copy()


# ----------------------------- Monotone keep (strictly increasing) ----------------------------- #
def keep_strictly_increasing(serials: np.ndarray) -> np.ndarray:
    """Greedy left→right keep: take first, then keep s if s > last_kept."""
    keep = []
    last = None
    for i, s in enumerate(serials):
        if last is None or s > last:
            keep.append(i)
            last = s
    return np.array(keep, dtype=int)


# ----------------------------- Period estimation (local + global fallbacks) ----------------------------- #
def global_period_from_starts(kept_dserial: np.ndarray, kept_dstart: np.ndarray) -> int:
    """
    Global period from *start* deltas:
      1) median(Δstart) for Δserial==1 and Δstart>0
      2) else median(Δstart/Δserial) for Δserial in {2,3,4} and Δstart>0
      3) else 1470 (≈ 44100/30)
    """
    mask1 = (kept_dserial == 1) & (kept_dstart > 0)
    if np.any(mask1):
        return int(round(np.median(kept_dstart[mask1])))
    mask_small = (kept_dserial >= 2) & (kept_dstart > 0) & (kept_dserial <= 4)
    if np.any(mask_small):
        per_step = kept_dstart[mask_small] / kept_dserial[mask_small].astype(float)
        return int(round(np.median(per_step)))
    return 1470


def local_period_around_gap_from_starts(
    kept_serials: np.ndarray,
    kept_starts: np.ndarray,
    i_gap: int,
    window: int,
    P_global: int,
) -> int:
    """
    Look up to `window` Δserial==1 pairs before/after the gap (using *start* deltas).
    Take medians per side; if both sides exist, length-weighted average; else fallback to P_global.
    """
    dserial = np.diff(kept_serials)
    dstart = np.diff(kept_starts)

    left = []
    k = i_gap - 1
    while k >= 0 and len(left) < window:
        if dserial[k] == 1 and dstart[k] > 0:
            left.append(int(dstart[k]))
        k -= 1

    right = []
    k = i_gap + 1
    while k < dserial.size and len(right) < window:
        if dserial[k] == 1 and dstart[k] > 0:
            right.append(int(dstart[k]))
        k += 1

    vals, wts = [], []
    if left:
        vals.append(np.median(left))
        wts.append(len(left))
    if right:
        vals.append(np.median(right))
        wts.append(len(right))
    if not vals:
        return int(P_global)
    if len(vals) == 1:
        return int(round(vals[0]))
    return int(round(float(np.average(vals, weights=wts))))


# ----------------------------- Core: compute gaps & JSON payload ----------------------------- #
def analyze_loss_json(
    serials: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    fs: int,
    local_window: int,
    top: int,
) -> Dict:
    # Monotone keep (even after prefilter; harmless redundancy for safety)
    keep_idx = keep_strictly_increasing(serials)
    if keep_idx.size < 2:
        raise ValueError("Not enough increasing values after monotone keep (need ≥ 2).")

    s = serials[keep_idx]
    st = starts[keep_idx]
    en = ends[keep_idx]

    dserial = np.diff(s)
    dstart = np.diff(st)

    ok = int(np.sum(dserial == 1))
    fwd = int(np.sum(dserial > 1))
    steps = int(dserial.size)

    # Forward diff histogram
    fwd_hist: Dict[int, int] = {}
    for d in dserial[dserial > 1]:
        fwd_hist[int(d)] = fwd_hist.get(int(d), 0) + 1
    fwd_hist_items = sorted(fwd_hist.items())

    # Period baselines (from *starts*)
    P_glob = global_period_from_starts(dserial, dstart)
    step_ms_ref = (
        (np.median(dstart[dserial == 1]) / float(fs) * 1000.0)
        if np.any(dserial == 1)
        else (P_glob / float(fs) * 1000.0)
    )

    # Per-gap details
    total_missing_samples = 0.0
    gaps: List[Dict] = []
    for i in range(steps):
        ds = int(dserial[i])
        if ds <= 1:
            continue
        D = int(max(0, dstart[i]))
        P_loc = local_period_around_gap_from_starts(
            s, st, i_gap=i, window=local_window, P_global=P_glob
        )
        ideal = int(round(ds * P_loc))
        miss = max(0, ideal - D)
        total_missing_samples += float(miss)
        gaps.append(
            {
                "index": i,
                "prev_serial": int(s[i]),
                "curr_serial": int(s[i + 1]),
                "prev_start_sample": int(st[i]),
                "prev_end_sample": int(en[i]),
                "curr_start_sample": int(st[i + 1]),
                "curr_end_sample": int(en[i + 1]),
                "diff": ds,
                "observed_ms": D * 1000.0 / float(fs),  # from start[i+1] - start[i]
                "ideal_ms": ideal
                * 1000.0
                / float(fs),  # ds * local period (starts-based)
                "missing_ms": miss * 1000.0 / float(fs),  # max(0, ideal - observed)
                "local_period_samples": int(P_loc),
            }
        )

    # Sort "top_gaps" by largest Δserial, then smallest observed_ms
    top_gaps = sorted(gaps, key=lambda g: (g["diff"], -g["observed_ms"]), reverse=True)[
        : max(0, top)
    ]

    # Window / totals (use observed span on the kept stream)
    observed_total_samples = int(en[-1] - st[0] + 1)
    analyzed_seconds = max(0.0, observed_total_samples / float(fs))
    total_missing_seconds = total_missing_samples / float(fs)
    loss_share = (
        (100.0 * total_missing_seconds / analyzed_seconds)
        if analyzed_seconds > 0
        else 0.0
    )

    payload = {
        "meta": {
            "method": "prefilter_then_monotone_keep_local_period_start_based",
            "fs_hz": int(fs),
            "local_window": int(local_window),
            "kept_range": {"first_serial": int(s[0]), "last_serial": int(s[-1])},
            "time_bounds_samples": {
                "start_sample": int(st[0]),
                "end_sample": int(en[-1]),
                "observed_total_samples": observed_total_samples,
            },
            "default_global_period_samples_from_starts": int(P_glob),
            "step_ms_reference": float(step_ms_ref),
        },
        "summary": {
            "values_kept": int(s.size),
            "steps": steps,
            "ok_steps": ok,
            "forward_jumps": fwd,
            "ok_ratio": (ok / steps) if steps else 1.0,
            "total_missing_seconds": float(total_missing_seconds),
            "analyzed_seconds": float(analyzed_seconds),
            "loss_share_pct": float(loss_share),
        },
        "histograms": {
            "forward_diff": {int(k): int(v) for k, v in fwd_hist_items},
        },
        "longest_ok_segment": _longest_ok_segment_dict(dserial),
        "gaps": gaps,
        "top_gaps": top_gaps,
    }
    return payload


def _longest_ok_segment_dict(dserial: np.ndarray) -> Dict[str, int]:
    best_len = cur = 0
    best_start = 0
    for i, ds in enumerate(dserial):
        if ds == 1:
            cur += 1
            if cur > best_len:
                best_len = cur
                best_start = i - cur + 1
        else:
            cur = 0
    return {"start_index": int(best_start), "length_steps": int(best_len)}


# ----------------------------- CLI ----------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="audio-loss-analysis",
        description="Quantify audio-side loss (JSON, START-based) with AudioFilter prefilter.",
    )
    ap.add_argument("csv_path", help="CSV with columns: serial,start_sample,end_sample")
    ap.add_argument(
        "--out-dir", required=True, help="Directory to write the JSON report"
    )
    ap.add_argument(
        "--fs", type=int, default=44100, help="Audio sample rate (Hz). Default: 44100"
    )
    ap.add_argument(
        "--local-window",
        type=int,
        default=3,
        help="Neighbors per side for local median (default: 3)",
    )
    ap.add_argument(
        "--top", type=int, default=12, help="How many top gaps to copy into 'top_gaps'"
    )
    ap.add_argument(
        "--prefilter",
        action="store_true",
        help="Enable anomaly prefilter (recommended)",
    )
    ap.add_argument(
        "--max-fwd-delta",
        type=int,
        default=200,
        help="Max allowed forward jump between kept rows in prefilter; set -1 for 'no limit'",
    )
    args = ap.parse_args(argv)

    try:
        csv_path = Path(args.csv_path)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = load_serial_csv(csv_path)
        prefilter_stats = None
        prefilter_impl = None

        if args.prefilter:
            mfd = None if int(args.max_fwd_delta) < 0 else int(args.max_fwd_delta)
            df, prefilter_stats, prefilter_impl = apply_prefilter(df, mfd)

        if len(df) < 2:
            raise ValueError("Not enough rows after prefilter (need ≥ 2).")

        payload = analyze_loss_json(
            serials=df["serial"].to_numpy(dtype=np.int64),
            starts=df["start_sample"].to_numpy(dtype=np.int64),
            ends=df["end_sample"].to_numpy(dtype=np.int64),
            fs=int(args.fs),
            local_window=int(args.local_window),
            top=int(args.top),
        )

        # Attach prefilter metadata (if used)
        if args.prefilter:
            payload.setdefault("meta", {})["prefilter"] = {
                "impl": prefilter_impl,
                "max_fwd_delta": (
                    None if int(args.max_fwd_delta) < 0 else int(args.max_fwd_delta)
                ),
                **(prefilter_stats or {}),
            }

        out_json = out_dir / f"{csv_path.stem}.audio_loss.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON → {out_json}")
        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
