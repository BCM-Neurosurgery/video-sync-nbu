#!/usr/bin/env python3
"""
csv_clipper.py

Read a CSV with columns: serial, start_sample, end_sample.
Keep only rows whose `serial` is within the inclusive range [--start, --end].
Write the result to --out-dir using basename + ".<start>-<end>.csv".
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ("serial", "start_sample", "end_sample")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clip rows by inclusive serial range using pandas."
    )
    p.add_argument("--csv", required=True, help="Path to input CSV.")
    p.add_argument("--start", type=int, required=True, help="Inclusive start serial.")
    p.add_argument("--end", type=int, required=True, help="Inclusive end serial.")
    p.add_argument("--out-dir", required=True, help="Directory to write output CSV.")
    return p.parse_args()


def derive_out_path(in_csv: Path, start: int, end: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{in_csv.stem}.{start}-{end}.csv"


def main() -> int:
    args = parse_args()
    in_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    if not in_path.is_file():
        print(f"[ERROR] Input CSV not found: {in_path}")
        return 2
    if args.start > args.end:
        print(f"[ERROR] start ({args.start}) must be <= end ({args.end}).")
        return 2

    try:
        df = pd.read_csv(in_path, dtype=str)  # read as strings first for safety
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return 2

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] CSV missing required column(s): {', '.join(missing)}")
        return 2

    # Coerce serial to numeric (drop non-numeric), then filter inclusively
    df["serial"] = pd.to_numeric(df["serial"], errors="coerce")
    df = df.dropna(subset=["serial"])
    df["serial"] = df["serial"].astype("int64")

    mask = df["serial"].between(args.start, args.end, inclusive="both")
    clipped = df.loc[mask, ["serial", "start_sample", "end_sample"]].copy()

    out_path = derive_out_path(in_path, args.start, args.end, out_dir)
    try:
        clipped.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")
        return 2

    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
