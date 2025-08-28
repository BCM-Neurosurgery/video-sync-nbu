#!/usr/bin/env python3
"""
csv_serial_analysis.py — CSV-only front end for serial discontinuity diagnostics.

Public API:
    analyze_csv_serials(path, column="serial", expect_step=1, top=5, hist_cols=2,
                        out_text=None) -> tuple[Analysis, pathlib.Path]

CLI:
    csv-serial-analysis <path.csv> [--column SERIAL] [--expect-step 1]
                        [--top 5] [--hist-cols 2] [--out-text OUT.txt]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
from typing import Tuple
from scripts.analysis.serial_analysis import (
    load_series_from_csv,
    analyze,
    summarize_text,
    Analysis,
)


# -----------------------------
# Public API
# -----------------------------
def analyze_csv_serials(
    path: str,
    column: str = "serial",
    *,
    expect_step: int = 1,
    top: int = 5,
    hist_cols: int = 2,
    out_text: Optional[str] = None,
) -> Tuple[Analysis, Path]:
    """Analyze a CSV integer sequence column and write a text report.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    column : str, default "serial"
        Column name in the CSV containing the integer sequence.
    expect_step : int, default 1
        Expected increment per step (E). Steps are diff = ids[i+1] - ids[i].
    top : int, default 5
        How many top forward/drops to list in the text report.
    hist_cols : int, default 2
        Histogram entries per row in the text report (1 = one per line).
    out_text : str or None, default None
        If provided, write the text report to this path.
        If None, defaults to "<csv_stem>.txt" next to the CSV.

    Returns
    -------
    (Analysis, Path)
        The structured analysis and the absolute path to the written text report.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If the CSV column is missing or contains fewer than 2 numeric values.
    OSError
        If writing out files fails due to I/O errors.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load & analyze
    series = load_series_from_csv(str(csv_path), column)
    ids = series.astype(int).tolist()
    result = analyze(ids, expect_step=expect_step, top_k=top)

    # Render text report
    report_text = summarize_text(
        result, include_tops=True, hist_cols=max(1, int(hist_cols))
    )
    default_text_path = csv_path.with_suffix(".txt")
    out_text_path = Path(out_text) if out_text is not None else default_text_path
    out_text_path = out_text_path.resolve()

    # Write text output
    try:
        out_text_path.parent.mkdir(parents=True, exist_ok=True)
        out_text_path.write_text(
            "Source → CSV:{col}\n{body}\n".format(col=column, body=report_text),
            encoding="utf-8",
        )
    except OSError as e:
        raise OSError(f"Failed to write text report: {out_text_path}") from e

    return result, out_text_path


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="csv-serial-analysis",
        description="Analyze discontinuities (ok, duplicate, forward, drop) in a CSV integer sequence.",
    )
    p.add_argument("path", help="Path to the CSV file")
    p.add_argument(
        "--column", default="serial", help="CSV column name (default: serial)"
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
        help="Histogram entries per row (1 = one per line, default: 2)",
    )
    p.add_argument(
        "--out-text",
        default=None,
        help="Write a text report to this path (default: <csv_stem>.txt next to the CSV)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    try:
        res, outp = analyze_csv_serials(
            path=args.path,
            column=args.column,
            expect_step=args.expect_step,
            top=args.top,
            hist_cols=args.hist_cols,
            out_text=args.out_text,  # may be None → default computed in API
        )
        logging.info(
            "Done. Values=%d Steps=%d ok=%d (%.2f%%) MissingIDs=%d → %s",
            res.n_values,
            res.total_steps,
            res.ok_steps,
            100.0 * res.ok_ratio,
            res.total_missing_ids,
            str(outp),
        )
        return 0
    except (FileNotFoundError, ValueError, OSError) as e:
        logging.error(str(e))
        return 2
    except Exception as e:
        logging.exception("Unexpected error: %s", e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
