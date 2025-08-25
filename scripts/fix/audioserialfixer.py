from scripts.fix.serialfixer import SerialFixer
from typing import Union, Optional
from pathlib import Path
import logging
import pandas as pd


logger = logging.getLogger("audio_serial_fixer")


class AudioSerialFixer(SerialFixer):
    """
    Audio strategy (simplified):
      • Only run midpoint gap passes (k = 2..19) on the 'serial' column.
    """

    GAPS = range(2, 20)

    def fix_csv(
        self,
        csv: Union[str, Path, "pd.DataFrame"],
    ) -> "pd.DataFrame":
        """
        Load CSV/DataFrame, verify it has exactly the columns:
        ['serial', 'start_sample', 'end_sample'], apply midpoint gap passes to
        'serial', and return a DataFrame with the same rows and fixed serials.
        """
        # Normalize input → DataFrame
        if isinstance(csv, (str, Path)):
            df = pd.read_csv(csv)
        else:
            df = csv.copy()

        # Strict schema check: exactly 3 required columns (order can vary)
        required = {"serial", "start_sample", "end_sample"}
        cols = list(df.columns)
        if len(cols) != 3 or set(cols) != required:
            raise ValueError(
                f"Input must have exactly these 3 columns: {sorted(required)}; found: {cols}"
            )

        # Ensure integer-like input for the fixer, then apply gap passes
        try:
            series = (
                pd.to_numeric(df["serial"], errors="raise").astype("int64").tolist()
            )
        except Exception as exc:
            raise ValueError("Column 'serial' must be integer-like.") from exc

        fixed = SerialFixer.apply_gap_passes_fast(series, self.GAPS)

        df_out = df.copy()
        df_out.loc[:, "serial"] = fixed
        return df_out


def fix_audio_csv(argv: Optional[list[str]] = None) -> None:
    """
    Public CLI entry: read CSV, verify schema, run AudioSerialFixer (gap passes only),
    and write '<stem>-fixed.csv' unless --out is provided.
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Fix 'serial' via midpoint gap passes (k = 2..19); schema must be serial,start_sample,end_sample."
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
        df_out: "pd.DataFrame" = fixer.fix_csv(in_path)
    except Exception as exc:
        logger.error("Failed to fix CSV: %s", exc)
        raise

    df_out.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df_out))


if __name__ == "__main__":
    fix_audio_csv()
