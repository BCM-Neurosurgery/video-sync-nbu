from scripts.fix.serialfixer import SerialFixer
from typing import Union, Optional
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger("audio_serial_fixer")
REQUIRED_COLS = {"serial", "start_sample", "end_sample"}


class AudioSerialFixer(SerialFixer):
    """
    Audio strategy:
      • Run midpoint gap passes (k = 2..19) on 'serial'.
      • Then keep rows by consecutive-sequence rule.
    """

    GAPS = range(2, 20)
    MAX_FWD_DELTA = 500  # allowed forward step between kept rows; tune as needed

    def fix_csv(
        self,
        csv: Union[str, Path, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Load CSV/DataFrame, verify schema == {serial,start_sample,end_sample},
        apply midpoint gap passes to 'serial', then filter rows via
        keep_consecutive_seq; return the fixed DataFrame.
        """
        df = self._load_csv_like(csv)
        self._validate_schema(df)

        # Apply midpoint gap passes to the serial column
        try:
            serial = (
                pd.to_numeric(df["serial"], errors="raise").astype("int64").tolist()
            )
        except Exception as exc:
            raise ValueError("Column 'serial' must be integer-like.") from exc

        fixed_serial = SerialFixer.apply_gap_passes_fast(serial, self.GAPS)
        df_fixed = df.copy()
        df_fixed.loc[:, "serial"] = fixed_serial

        # Filter rows by the consecutive-sequence rule
        return self.keep_consecutive_seq(df_fixed)

    # --- helpers & sequence filter ---

    @staticmethod
    def _load_csv_like(obj: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load a CSV path or copy a DataFrame."""
        if isinstance(obj, (str, Path)):
            return pd.read_csv(obj)
        return obj.copy()

    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> None:
        """Ensure the input has exactly the required columns (order may vary)."""
        cols = list(df.columns)
        if len(cols) != 3 or set(cols) != REQUIRED_COLS:
            raise ValueError(
                f"Input must have exactly these 3 columns: {sorted(REQUIRED_COLS)}; found: {cols}"
            )

    @staticmethod
    def keep_consecutive_seq(df: pd.DataFrame) -> pd.DataFrame:
        """
        Anchor-aware filter over the 'serial' column.

        Behavior
        --------
        • Keep the first row; set ANCHOR = serial[0], LAST_KEPT = serial[0].
        • For each subsequent value cur:
            - Drop if cur <= ANCHOR            # back-jumps and duplicates
            - Drop if (MAX_FWD_DELTA is not None) and (cur - LAST_KEPT) > MAX_FWD_DELTA
            - Otherwise keep; set ANCHOR = LAST_KEPT = cur

        Examples
        --------
        1) Strictly increasing by 1 (keep all)
           serial: 10, 11, 12
           keep:   10, 11, 12

        2) Back-jump after an increasing run (small numbers suppressed until surpassing anchor)
           serial: 1001, 1002, 1003, 1004, 1005, 98, 99, 1009, 1010
           keep:   1001, 1002, 1003, 1004, 1005,      1009, 1010
           # 98 and 99 are <= anchor (1005) → dropped

        3) Duplicate values (drop duplicates)
           serial: 42, 42, 43, 43
           keep:   42,     43

        4) Large forward jump (controlled by MAX_FWD_DELTA)
           MAX_FWD_DELTA = 500
           serial: 1000, 1100, 1700, 2201
           keep:   1000, 1100, 1700
           # 2201 - 1700 = 501 > 500 → drop
           # Set MAX_FWD_DELTA = None to allow 2201 (and any forward jump).

        5) Minor decrease (treated as back-jump → drop)
           serial: 100, 101, 100, 102
           keep:   100, 101,      102

        Returns
        -------
        pd.DataFrame
            Rows corresponding to kept serials; index is reset.
        """
        # Schema check (lightweight; fix_csv already validates)
        missing = REQUIRED_COLS.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Robust integer coercion
        s = pd.to_numeric(df["serial"], errors="raise").astype("int64").to_numpy()
        n = len(s)
        if n <= 1:
            return df.copy()

        keep_idx = [0]  # always keep the first row
        anchor = int(s[0])  # last trusted increasing value
        last_kept = int(s[0])  # last value we actually appended
        max_delta = getattr(AudioSerialFixer, "MAX_FWD_DELTA", None)

        for i in range(1, n):
            cur = int(s[i])

            # Reject values at or below the anchor (handles 1005 → 98, 99, ...)
            if cur <= anchor:
                continue

            # Optional guard against implausibly large forward jumps
            if max_delta is not None and (cur - last_kept) > max_delta:
                continue

            # Accept and advance anchor
            keep_idx.append(i)
            anchor = cur
            last_kept = cur

        out = df.iloc[keep_idx].copy()
        out.reset_index(drop=True, inplace=True)
        return out


def fix_audio_csv(argv: Optional[list[str]] = None) -> None:
    """
    Minimal public CLI: read CSV → AudioSerialFixer.fix_csv → write '<stem>-fixed.csv'
    (or --out).
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description=(
            "Fix 'serial' via midpoint gap passes (k=2..19) and a consecutive-sequence "
            "filter. Schema must be serial,start_sample,end_sample."
        )
    )
    p.add_argument("csv", type=Path, help="Input CSV path")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_stem>-fixed.csv in the same directory)",
    )
    args = p.parse_args(argv)

    in_path: Path = args.csv
    if not in_path.exists():
        raise SystemExit(f"ERROR: input CSV not found: {in_path}")

    out_path: Path = args.out or in_path.with_name(f"{in_path.stem}-fixed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", in_path)
    fixer = AudioSerialFixer()

    try:
        df_out = fixer.fix_csv(in_path)
    except Exception as exc:
        logger.error("Failed to process CSV: %s", exc)
        raise

    df_out.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df_out))
    print(str(out_path))


if __name__ == "__main__":
    fix_audio_csv()
