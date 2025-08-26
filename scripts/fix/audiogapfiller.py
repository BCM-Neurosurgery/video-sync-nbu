#!/usr/bin/env python3
"""
audio_gap_filler.py — Apply midpoint gap-filling passes to 'serial' only.

Auto gap selection:
- If no gaps are provided, choose powers of 10: [10, 100, 1000, ...] < len(series).
- For short sequences (< 10), no passes are applied.

Public API
----------
- class AudioGapFiller:
    - fill_list(series) -> List[int]
    - fill_csv(csv_like) -> pd.DataFrame

CLI
---
$ python audio_gap_filler.py input.csv [-o OUT] [--gaps 10,100,1000]
"""
from __future__ import annotations

from typing import Sequence, Optional, Union, List
from pathlib import Path
import logging
import math

import pandas as pd
from scripts.fix.serialfixer import SerialFixer  # uses apply_gap_passes_fast

logger = logging.getLogger("audio_gap_filler")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

REQUIRED_COLS = {"serial", "start_sample", "end_sample"}


class AudioGapFiller:
    """
    Fill gaps in an integer 'serial' sequence using midpoint passes.

    Gap policy
    ----------
    - If `gaps` is provided, use it verbatim.
    - Otherwise, auto-compute powers of 10 less than the sequence length:
        [10, 100, 1000, ..., < len(series)]
      (Each gap g must satisfy 2 <= g < len(series).)

    Notes
    -----
    - Only modifies 'serial' values; never drops/reorders rows.
    - Coerces 'serial' to int64 (raises on non-integer-like values).
    """

    def __init__(self, gaps: Optional[Sequence[int]] = None) -> None:
        self._gaps: Optional[List[int]] = list(gaps) if gaps else None

    # -----------------------------
    # Public API
    # -----------------------------
    def fill_list(self, series: Sequence[int]) -> List[int]:
        """
        Apply fast midpoint gap passes to a plain integer sequence.
        """
        s = [int(x) for x in series]
        gaps = self._resolve_gaps(len(s))
        if gaps:
            logger.info("Using midpoint gaps: %s", gaps)
            return SerialFixer.apply_gap_passes_fast(s, gaps)
        logger.info(
            "No applicable gaps for length %s; returning input unchanged", len(s)
        )
        return s

    def fill_csv(self, csv_like: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Load CSV/DataFrame, validate schema, and gap-fill the 'serial' column.
        """
        df = self._load_csv_like(csv_like)
        self._validate_schema(df)

        try:
            serial = (
                pd.to_numeric(df["serial"], errors="raise").astype("int64").tolist()
            )
        except Exception as exc:
            raise ValueError("Column 'serial' must be integer-like.") from exc

        fixed = self.fill_list(serial)
        out = df.copy()
        out.loc[:, "serial"] = fixed
        return out

    # -----------------------------
    # Helpers
    # -----------------------------
    def _resolve_gaps(self, n: int) -> List[int]:
        """
        If user provided `gaps`, sanitize them; otherwise auto-generate powers of 10.
        """
        if n < 3:
            return []

        if self._gaps is not None:
            # Keep only valid window sizes (2 <= g < n), dedup + sort (stable enough here)
            valid = sorted({int(g) for g in self._gaps if 2 <= int(g) < n})
            return valid

        return self._pow10_gaps(n)

    @staticmethod
    def _pow10_gaps(n: int) -> List[int]:
        """
        Generate [10, 100, 1000, ...] strictly less than n.
        """
        if n <= 10:
            return []
        gaps: List[int] = []
        k = 1  # 10^1 = 10
        while True:
            g = 10**k
            if g >= n:
                break
            gaps.append(g)
            k += 1
        return gaps

    @staticmethod
    def _load_csv_like(obj: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(obj, (str, Path)):
            return pd.read_csv(obj)
        return obj.copy()

    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> None:
        cols = list(df.columns)
        if len(cols) != 3 or set(cols) != REQUIRED_COLS:
            raise ValueError(
                f"Input must have exactly these 3 columns: {sorted(REQUIRED_COLS)}; found: {cols}"
            )


def _parse_gaps(gaps_arg: Optional[str]) -> Optional[List[int]]:
    """
    Parse a comma-separated integer list like "10,100,1000".
    Returns None if gaps_arg is falsy or equals 'auto' (case-insensitive).
    """
    if not gaps_arg or gaps_arg.strip().lower() == "auto":
        return None
    items = [x.strip() for x in gaps_arg.split(",") if x.strip()]
    try:
        return [int(x) for x in items]
    except Exception as exc:
        raise ValueError(f"Invalid --gaps argument: {gaps_arg!r}") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Gap-fill only: apply midpoint passes to 'serial' (no row filtering)."
    )
    p.add_argument("csv", type=Path, help="Input CSV (serial,start_sample,end_sample)")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_stem>-gapfilled.csv next to input)",
    )
    p.add_argument(
        "--gaps",
        type=str,
        default=None,
        help="Comma-separated window sizes (e.g., '10,100,1000') or 'auto' (default).",
    )

    args = p.parse_args(argv)

    in_path: Path = args.csv
    if not in_path.exists():
        logger.error("Input CSV not found: %s", in_path)
        return 2

    out_path: Path = args.out or in_path.with_name(f"{in_path.stem}-gapfilled.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gaps = _parse_gaps(args.gaps)
        filler = AudioGapFiller(gaps=gaps)
        logger.info("Reading %s", in_path)
        df_out = filler.fill_csv(in_path)
        df_out.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(df_out))
        print(str(out_path))
        return 0
    except Exception as e:
        logger.exception("Failed to gap-fill: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
