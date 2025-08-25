from scripts.fix.serialfixer import SerialFixer
from typing import List, Tuple, Optional
from pathlib import Path


class AudioSerialFixer(SerialFixer):
    """
    Audio strategy:
      1) Midpoint gap passes (k = 2..19).
      2) Drop zeros.
      3) Drop consecutive duplicates.
      4) Drop min/max outliers.
    """

    GAPS = range(2, 20)

    def fix(self, series: List[int]) -> Tuple[List[int], List[int]]:
        """
        Pipeline for audio: gap passes → drop zeros → drop consecutive duplicates
                            → drop min outliers → drop max outliers.
        Returns (filtered_values, kept_original_indices_after_pipeline).
        """
        # 1) Midpoint gap passes
        s = SerialFixer.apply_gap_passes_fast(series, self.GAPS)
        if not s:
            return [], []

        # 2) Drop zeros
        s2, k1 = SerialFixer.drop_zeros(s)
        if not s2:
            return [], []

        # 3) Drop consecutive duplicates (keep first of each run)
        s3, k2 = SerialFixer.drop_consecutive_duplicates(s2)
        if not s3:
            return [], []
        idx3 = [k1[i] for i in k2]

        # 4) Drop midpoints outliers
        s4, k3 = SerialFixer.drop_midpoints_gap(s3, 2)
        if not s4:
            return [], []
        idx4 = [idx3[i] for i in k3]

        return s4, idx4


def fix_audio_csv(argv: Optional[List[str]] = None) -> None:
    """
    CLI: read CSV (serial,start_sample,end_sample), run AudioSerialFixer pipeline ONCE
    (gap passes 2..19 → drop zeros → drop duplicates → drop min/max outliers),
    filter rows accordingly, and write '<stem>-fixed.csv'.

    Usage
    -----
    python script.py input.csv
    python script.py input.csv -o /path/to/out.csv
    """
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(
        description="Fix 'serial' via gap passes (2..19), then drop zeros, duplicates, and min/max outliers."
    )
    p.add_argument(
        "csv", type=Path, help="Input CSV with columns: serial,start_sample,end_sample"
    )
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

    out_path: Path = (
        args.out if args.out else in_path.with_name(f"{in_path.stem}-fixed.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    required = {"serial", "start_sample", "end_sample"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(
            f"ERROR: missing required columns: {sorted(missing)}. Found: {list(df.columns)}"
        )

    # Ensure integer-like serials
    try:
        serials = df["serial"].astype("int64").to_numpy()
    except Exception as exc:
        raise SystemExit("ERROR: column 'serial' must contain integers.") from exc

    fixer = AudioSerialFixer()
    final_vals, kept_idx = fixer.fix(serials.tolist())

    # Filter DataFrame to match the kept positions and assign fixed values.
    df = df.iloc[kept_idx].copy()
    df.reset_index(drop=True, inplace=True)
    df.loc[:, "serial"] = final_vals

    df.to_csv(out_path, index=False)
    print(str(out_path))


if __name__ == "__main__":
    fix_audio_csv()
