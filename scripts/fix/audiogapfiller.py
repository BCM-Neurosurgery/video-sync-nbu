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
- gapfill_csv_file(input_csv, out_path=None, gaps=None) -> Path
    Load CSV, apply gap-filling to 'serial', save to disk (default suffix '-gapfilled.csv'),
    and return the absolute output path.

CLI
---
$ python audio_gap_filler.py input.csv [-o OUT] [--gaps 10,100,1000]
"""
from __future__ import annotations

from typing import Sequence, Optional, Union, List
from pathlib import Path
import logging

import pandas as pd
from scripts.fix.serialfixer import SerialFixer
from scripts.log.logutils import configure_standalone_logging
from scripts.utility.utils import _name

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"serial", "start_sample", "end_sample"}


class AudioGapFiller:
    """
    Fill gaps in an integer 'serial' sequence using midpoint passes.

    Gap policy
    ----------
    - If `gaps` is provided, use it verbatim after sanitization.
    - Otherwise, auto-compute powers of 10 strictly less than the sequence length:
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

        Parameters
        ----------
        series : Sequence[int]
            Input 'serial' sequence.

        Returns
        -------
        List[int]
            Gap-filled sequence (or the original if no passes apply).
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

        Parameters
        ----------
        csv_like : str | Path | pd.DataFrame
            Input CSV path or a DataFrame with required columns.

        Returns
        -------
        pd.DataFrame
            A copy of the input with 'serial' gap-filled.
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
            try:
                valid = sorted({int(g) for g in self._gaps if 2 <= int(g) < n})
            except Exception as exc:
                raise ValueError("All gap sizes must be integers >=2.") from exc
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
        g = 10
        while g < n:
            gaps.append(g)
            g *= 10
        return gaps

    @staticmethod
    def _load_csv_like(obj: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(obj, (str, Path)):
            try:
                return pd.read_csv(obj)
            except FileNotFoundError:
                raise FileNotFoundError(f"Input CSV not found: {obj}")
            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"CSV is empty: {obj}") from exc
            except Exception as exc:
                raise RuntimeError(f"Failed to read CSV: {obj}") from exc
        return obj.copy()

    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> None:
        cols = list(df.columns)
        if len(cols) != 3 or set(cols) != REQUIRED_COLS:
            raise ValueError(
                f"Input must have exactly these 3 columns: {sorted(REQUIRED_COLS)}; found: {cols}"
            )


# -----------------------------
# Public API function
# -----------------------------
def gapfill_csv_file(
    input_csv: Union[str, Path],
    *,
    out_path: Optional[Union[str, Path]] = None,
    gaps: Optional[Sequence[int]] = None,
) -> Path:
    """
    Gap-fill 'serial' from an input CSV and save to disk.

    By default, writes alongside the input as "<stem>-gapfilled.csv".

    Parameters
    ----------
    input_csv : str | Path
        Path to the input CSV with columns: serial,start_sample,end_sample.
    out_path : str | Path | None, default None
        Output CSV path. If None, uses "<input_stem>-gapfilled.csv" in the same directory.
    gaps : Sequence[int] | None, default None
        Optional gap sizes to apply. If None, powers of 10 are auto-selected.

    Returns
    -------
    Path
        Absolute path to the written output CSV.

    Raises
    ------
    FileNotFoundError
        If the input CSV does not exist.
    ValueError
        If the CSV schema is invalid or serial values are not integer-like.
    RuntimeError
        If reading, processing, or writing fails unexpectedly.
    """
    in_path = Path(input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    filler = AudioGapFiller(gaps=gaps)
    logger.info("Reading %s", _name(in_path))
    df_out = filler.fill_csv(in_path)

    out = (
        Path(out_path)
        if out_path is not None
        else in_path.with_name(f"{in_path.stem}-gapfilled.csv")
    )
    out = out.resolve()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to write output CSV: {out}") from exc

    logger.info("Wrote %s (%d rows)", _name(out), len(df_out))
    return out


# -----------------------------
# CLI
# -----------------------------
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

    # Standalone console logging; no-op if the driver already configured root.
    configure_standalone_logging("INFO", seg="-", cam="-")

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

    try:
        outp = gapfill_csv_file(
            input_csv=args.csv,
            out_path=args.out,
            gaps=_parse_gaps(args.gaps),
        )
        print(str(outp))
        return 0
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 2
    except Exception as e:
        logger.exception("Failed to gap-fill: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
