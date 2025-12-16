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
      • Scan for strictly increasing runs that, when MAX_FWD_DELTA is set, never jump
        forward by more than that amount between consecutive rows.
      • Keep every run whose length reaches MIN_RUN_LENGTH; discard shorter runs so
        they cannot seed anchors.

    Notes
    -----
    - No gap filling is performed.
    - Input schema must be exactly: serial,start_sample,end_sample (order may vary).
    """

    # Allowed forward step between kept rows; set to None to allow any forward jump.
    # by observation, the forward jump is ~64
    MAX_FWD_DELTA: Optional[int] = 200
    # Minimum length of a strictly increasing, delta-bounded run to consider it a valid start.
    MIN_RUN_LENGTH: int = 3

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

        max_delta = getattr(AudioFilter, "MAX_FWD_DELTA", None)
        min_run = getattr(AudioFilter, "MIN_RUN_LENGTH", 3)
        values = [int(v) for v in s.tolist()]
        runs = AudioFilter._collect_valid_runs(values, max_delta, min_run)
        if not runs:
            raise ValueError(
                "No strictly increasing run reached MIN_RUN_LENGTH; check serial data."
            )

        keep_rows: list[int] = []
        for start, end in runs:
            keep_rows.extend(range(start, end))

        out = df.iloc[keep_rows].copy()
        out.reset_index(drop=True, inplace=True)
        return out

    @staticmethod
    def _collect_valid_runs(
        values: list[int], max_delta: Optional[int], min_run: int
    ) -> list[tuple[int, int]]:
        """Return (start, end) pairs for monotone runs meeting MIN_RUN_LENGTH."""

        runs: list[tuple[int, int]] = []
        n = len(values)
        if n <= 1:
            return runs

        run_start = 0
        for idx in range(1, n):
            diff = values[idx] - values[idx - 1]
            violates = diff <= 0 or (max_delta is not None and diff > max_delta)
            if violates:
                if idx - run_start >= min_run:
                    runs.append((run_start, idx))
                run_start = idx

        if n - run_start >= min_run:
            runs.append((run_start, n))
        return runs


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
