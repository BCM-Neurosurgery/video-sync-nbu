from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Iterable
import numpy as np
from tqdm import tqdm


class SerialFixer(ABC):
    """
    Abstract base for repairing integer serial sequences via sliding-window rules.

    Sliding-window rule (generic, gap = k >= 2)
    -------------------------------------------
    For each window bounded by indices:
        L = i - 1, R = i + (k - 1)   with interior i..R-1
    If s[R] - s[L] == k, the interior must be a strict +1 run:
        s[i]   = s[L] + 1
        s[i+1] = s[L] + 2
        ...
        s[R-1] = s[L] + (k - 1)
    Endpoints L and R are never changed by this rule.
    """

    @abstractmethod
    def fix(self, series: List[int]) -> Tuple[List[int], List[int]]:
        """Return a new list with this strategy's fixes applied."""
        raise NotImplementedError

    @staticmethod
    def apply_gap_passes(series: List[int], gaps: Iterable[int]) -> List[int]:
        """Apply fix_midpoints_gap for each gap (left→right, one pass each)."""
        s = list(series)
        for gap in tqdm(gaps):
            s = SerialFixer.fix_midpoints_gap(s, gap)
        return s

    @staticmethod
    def drop_zeros(series: List[int]) -> Tuple[List[int], List[int]]:
        """Remove all elements equal to 0. Returns (filtered, kept_indices)."""
        if not series:
            return [], []
        arr = np.asarray(series)
        keep_mask = arr != 0
        kept_idx = np.nonzero(keep_mask)[0]
        return arr[keep_mask].tolist(), kept_idx.tolist()

    @staticmethod
    def drop_consecutive_duplicates(series: List[int]) -> Tuple[List[int], List[int]]:
        """
        Drop elements that are equal to their immediate predecessor (keep the first).
        Readable for-loop implementation.
        """
        kept_vals: List[int] = []
        kept_idx: List[int] = []
        prev = None
        for i, v in enumerate(series):
            if prev is None or v != prev:
                kept_vals.append(v)
                kept_idx.append(i)
                prev = v
        return kept_vals, kept_idx

    @staticmethod
    def drop_min_outlier(series: List[int]) -> Tuple[List[int], List[int]]:
        """Drop elements s[i] such that s[i] < s[0]/10. Returns (filtered, kept_indices)."""
        if not series:
            return [], []
        m = series[0]
        thr = m / 10.0
        kept_vals: List[int] = []
        kept_idx: List[int] = []
        for i, v in enumerate(series):
            if v >= thr:
                kept_vals.append(v)
                kept_idx.append(i)
        return kept_vals, kept_idx

    @staticmethod
    def drop_max_outlier(series: List[int]) -> Tuple[List[int], List[int]]:
        """Drop elements s[i] such that s[i] > s[-1] * 10. Returns (filtered, kept_indices)."""
        if not series:
            return [], []
        M = series[-1]
        thr = M * 10.0
        kept_vals: List[int] = []
        kept_idx: List[int] = []
        for i, v in enumerate(series):
            if v <= thr:
                kept_vals.append(v)
                kept_idx.append(i)
        return kept_vals, kept_idx

    @staticmethod
    def fix_midpoints_gap(series: List[int], gap: int) -> List[int]:
        """Apply one left→right pass of the sliding-window rule for a given gap."""
        if gap < 2:
            return list(series)

        n = len(series)
        if n < gap + 1:  # need at least L + interior(k-1) + R → k+1 points
            return list(series)

        s = list(series)  # work on a copy

        # i indexes the first interior element; L=i-1, R=i+(gap-1)
        for i in range(1, n - gap + 1):
            L = i - 1
            R = i + (gap - 1)

            if s[R] - s[L] != gap:
                continue  # endpoints don't meet the span requirement

            # Check interior; if any mismatch, rewrite the whole interior block
            needs_fix = False
            for j in range(i, R):
                expected = s[L] + (j - L)
                if s[j] != expected:
                    needs_fix = True
                    break

            if needs_fix:
                for j in range(i, R):
                    s[j] = s[L] + (j - L)

        return s


class CamJsonSerialFixer(SerialFixer):
    """Camera JSON strategy: apply gap fixes in this order: [2, 130]."""

    def fix(self, series: List[int]) -> List[int]:
        s = list(series)
        for gap in (2, 130):
            s = self.fix_midpoints_gap(s, gap)
        return s


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
        s = SerialFixer.apply_gap_passes(series, self.GAPS)
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

        # 4) Drop min outliers
        s4, k3 = SerialFixer.drop_min_outlier(s3)
        if not s4:
            return [], []
        idx4 = [idx3[i] for i in k3]

        # 5) Drop max outliers
        s5, k4 = SerialFixer.drop_max_outlier(s4)
        if not s5:
            return [], []
        idx5 = [idx4[i] for i in k4]

        return s5, idx5


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
