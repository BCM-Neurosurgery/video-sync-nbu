from typing import Union, Optional
from pathlib import Path
import logging
import pandas as pd

from scripts.log.logutils import configure_standalone_logging, log_context

logger = logging.getLogger(__name__)
REQUIRED_COLS = {"serial", "start_sample", "end_sample"}


class AudioFilter:
    """
    Filter-only processor for serial blocks.

    Strategy
    --------
    Anchor-aware monotone filter on the 'serial' column:
      • Keep the first row; set ANCHOR = LAST_KEPT = serial[0].
      • For each subsequent value 'cur':
          - Drop if cur <= ANCHOR            (handles back-jumps & duplicates)
          - Drop if MAX_FWD_DELTA is not None and (cur - LAST_KEPT) > MAX_FWD_DELTA
          - Otherwise keep; update ANCHOR = LAST_KEPT = cur.

    Notes
    -----
    - No gap filling is performed.
    - Input schema must be exactly: serial,start_sample,end_sample (order may vary).
    """

    # Allowed forward step between kept rows; set to None to allow any forward jump.
    # by observation, the forward jump is ~64
    MAX_FWD_DELTA: Optional[int] = 200

    def filter_csv(self, csv: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Load CSV/DataFrame, validate schema, and return rows kept by
        the consecutive-sequence rule.

        Parameters
        ----------
        csv : Union[str, Path, pd.DataFrame]
            Path to CSV with columns {serial,start_sample,end_sample}, or a DataFrame.

        Returns
        -------
        pd.DataFrame
            Filtered rows with index reset.
        """
        df = self._load_csv_like(csv)
        self._validate_schema(df)

        # Ensure 'serial' is integer-like up front
        try:
            _ = pd.to_numeric(df["serial"], errors="raise").astype("int64")
        except Exception as exc:
            raise ValueError("Column 'serial' must be integer-like.") from exc

        return self.keep_consecutive_seq(df)

    # --- helpers & sequence filter ---

    @staticmethod
    def _load_csv_like(obj: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load a CSV path or copy a DataFrame (no mutation of the original)."""
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
        Anchor-aware filter over the 'serial' column. See class docstring for details.

        Returns
        -------
        pd.DataFrame
            Rows corresponding to kept serials; index is reset.
        """
        missing = REQUIRED_COLS.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        s = pd.to_numeric(df["serial"], errors="raise").astype("int64").to_numpy()
        n = len(s)
        if n <= 1:
            return df.copy()

        keep_idx = [0]  # always keep the first row
        anchor = int(s[0])  # last trusted increasing value
        last_kept = int(s[0])  # last value we actually appended
        max_delta = getattr(AudioFilter, "MAX_FWD_DELTA", None)

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


def filter_audio_file(
    input_csv: Union[str, Path], out_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Programmatic API: load CSV, apply AudioFilter, save CSV, return output path.
    Library function — no logging configuration or stdout printing here.
    """
    in_path = Path(input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = AudioFilter().filter_csv(in_path)
    out = (
        Path(out_path)
        if out_path
        else in_path.with_name(f"{in_path.stem}-filtered.csv")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def filter_audio_csv(argv: Optional[list[str]] = None) -> None:
    """
    Minimal CLI: read CSV → AudioFilter.filter_csv → write '<stem>-filtered.csv' (or --out).
    Uses shared logutils for clean standalone logging & stamps logs with [<csv-stem>/-].
    """
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Filter rows based on a monotone-increasing rule over 'serial'. "
            "Schema must be serial,start_sample,end_sample. No gap filling."
        )
    )
    p.add_argument("csv", type=Path, help="Input CSV path")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_stem>-filtered.csv in the same directory)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (standalone only; ignored when called from driver)",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="Only print warnings and errors."
    )
    args = p.parse_args(argv)

    in_path: Path = args.csv
    if not in_path.exists():
        raise SystemExit(f"ERROR: input CSV not found: {in_path}")

    # Standalone console logging; a no-op if a driver already configured root.
    level = "WARNING" if args.quiet else args.log_level
    root = logging.getLogger()
    was_handlerless = not root.handlers
    configure_standalone_logging(level, seg=in_path.stem, cam="-")

    out_path: Path = args.out or in_path.with_name(f"{in_path.stem}-filtered.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Only stamp context here in true-standalone case so we don't override driver tags.
        if was_handlerless:
            with log_context(seg=in_path.stem, cam="-"):
                logger.info("Reading %s", in_path.name)
                df_out = AudioFilter().filter_csv(in_path)
                df_out.to_csv(out_path, index=False)
                logger.info("Wrote %s (%d rows)", out_path.name, len(df_out))
        else:
            logger.info("Reading %s", in_path.name)
            df_out = AudioFilter().filter_csv(in_path)
            df_out.to_csv(out_path, index=False)
            logger.info("Wrote %s (%d rows)", out_path.name, len(df_out))

        # Preserve existing behavior: print the path to stdout for piping.
        print(str(out_path))
    except Exception as exc:
        logger.error("Failed to process CSV: %s", exc)
        raise


if __name__ == "__main__":
    filter_audio_csv()
