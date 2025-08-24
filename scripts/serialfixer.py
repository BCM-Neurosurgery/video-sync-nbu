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
    def keep_increase_seq(series: List[int]) -> Tuple[List[int], List[int]]:
        """
        Keep a strictly increasing subsequence by using the last kept value as an anchor.
        If s[i] < last_kept, skip until a future s[j] > last_kept; append only those > anchor.

        Returns
        -------
        (filtered, kept_indices)
        """
        if not series:
            return [], []
        kept_vals: List[int] = []
        kept_idx: List[int] = []
        # Always keep the first element as the initial anchor
        anchor = series[0]
        kept_vals.append(anchor)
        kept_idx.append(0)
        for i in range(1, len(series)):
            v = series[i]
            if v > anchor:
                kept_vals.append(v)
                kept_idx.append(i)
                anchor = v  # advance anchor
            # else: v <= anchor → skip
        return kept_vals, kept_idx

    @staticmethod
    def drop_min_outlier(series: List[int]) -> Tuple[List[int], List[int]]:
        """
        Relative-downside outlier filter:
        walk the list and skip s[i] if s[i] < s[i-1] / 5.
        The comparison uses the immediate predecessor in the ORIGINAL sequence.
        Keeps the first element by definition.

        Returns (filtered_values, kept_indices).
        """
        if not series:
            return [], []
        kept_vals: List[int] = [series[0]]
        kept_idx: List[int] = [0]
        prev = series[0]
        for i in range(1, len(series)):
            v = series[i]
            if v >= prev / 5.0:
                kept_vals.append(v)
                kept_idx.append(i)
            # else: drop v
            prev = v  # always advance to the immediate predecessor in the original sequence
        return kept_vals, kept_idx

    @staticmethod
    def drop_max_outlier(series: List[int]) -> Tuple[List[int], List[int]]:
        """
        Relative-upside outlier filter:
        walk the list and skip s[i] if s[i] > s[i-1] * 10.
        The comparison uses the immediate predecessor in the ORIGINAL sequence.
        Keeps the first element by definition.

        Returns (filtered_values, kept_indices).
        """
        if not series:
            return [], []
        kept_vals: List[int] = [series[0]]
        kept_idx: List[int] = [0]
        prev = series[0]
        for i in range(1, len(series)):
            v = series[i]
            if v <= prev * 5.0:
                kept_vals.append(v)
                kept_idx.append(i)
            # else: drop v
            prev = v  # always advance to the immediate predecessor in the original sequence
        return kept_vals, kept_idx

    @staticmethod
    def drop_midpoints_gap(series: List[int], gap: int) -> Tuple[List[int], List[int]]:
        """
        One left→right pass over windows of size `gap` (k >= 2), with the same
        endpoint condition as fix_midpoints_gap/_fast:
            L = i-1, R = i+(gap-1), require s[R] - s[L] == gap.
        For each such window, drop any INTERIOR element s[j] (i <= j <= R-1) that is
            - ≥ 5x both endpoints, or
            - ≤ (1/5)x both endpoints.
        Returns (filtered_values, kept_indices) w.r.t. the ORIGINAL series.
        """
        s = np.asarray(series, dtype=np.int64)
        n = len(s)
        if gap < 2 or n < gap + 1:
            return list(series), list(range(n))

        keep = np.ones(n, dtype=bool)

        # Slide i so that L=i-1 and R=i+(gap-1) are valid
        for i in range(1, n - gap + 1):
            L = i - 1
            R = i + (gap - 1)

            # Same endpoint-span gate as the "fix" rule
            if s[R] - s[L] != gap:
                continue

            lo_ep = int(min(s[L], s[R]))
            hi_ep = int(max(s[L], s[R]))

            # Consider only interior indices i..R-1
            for j in range(i, R):
                v = int(s[j])
                # Drop if >= 5× both endpoints or <= 1/5× both endpoints
                if (v >= 5 * hi_ep) or (v * 5 <= lo_ep):
                    keep[j] = False

        kept_idx = np.nonzero(keep)[0].tolist()
        kept_vals = s[keep].tolist()
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

    def fix_midpoints_gap_fast(series: List[int], gap: int) -> List[int]:
        """
        Same semantics as fix_midpoints_gap (one left→right pass for a single gap),
        but faster: operate on diff and use sliding window sums.
        """
        s = np.asarray(series, dtype=np.int64)
        n = len(s)
        if gap < 2 or n < gap + 1:
            return s.tolist()

        diff = np.diff(s)  # length n-1
        is_one = diff == 1  # boolean view for "already perfect"

        # Initialize sliding-window sums for the first [0:gap) block
        sum_diff = int(diff[:gap].sum())  # equals s[gap]-s[0]
        sum_one = int(is_one[:gap].sum())

        # L runs over left endpoints of diff windows
        # this ensures R = L + gap is always less than n
        for L in range(0, n - gap):
            # Endpoint condition: s[L+gap]-s[L] == gap  <=>  sum_diff == gap
            # Need a write only if interior not already all ones (sum_one != gap)
            if sum_diff == gap and sum_one != gap:
                # Rewrite the whole interior block to +1 in one shot
                diff[L : L + gap] = 1
                is_one[L : L + gap] = True
                # Current window now perfect
                sum_diff = gap
                sum_one = gap

            # Slide window: remove diff[L], add diff[L+gap]
            if L + gap < len(diff):
                sum_diff = sum_diff - int(diff[L]) + int(diff[L + gap])
                sum_one = sum_one - int(is_one[L]) + int(is_one[L + gap])

        # Reconstruct s from s[0] and the (possibly modified) diffs
        out = np.empty_like(s)
        out[0] = s[0]
        out[1:] = s[0] + np.cumsum(diff)
        return out.tolist()

    def apply_gap_passes_fast(series: List[int], gaps: Iterable[int]) -> List[int]:
        """
        Faster replacement for SerialFixer.apply_gap_passes(series, gaps) that preserves
        the exact left→right, gap-by-gap semantics.
        """
        s = np.asarray(series, dtype=np.int64)
        n = len(s)
        if n < 3:
            return s.tolist()

        # Work on 'diff' directly and rebuild once at the end
        diff = np.diff(s)

        for gap in tqdm(gaps):
            if gap < 2 or n < gap + 1:
                continue

            is_one = diff == 1
            sum_diff = int(diff[:gap].sum())
            sum_one = int(is_one[:gap].sum())

            for L in range(0, n - gap):
                if sum_diff == gap and sum_one != gap:
                    diff[L : L + gap] = 1
                    is_one[L : L + gap] = True
                    sum_diff = gap
                    sum_one = gap

                if L + gap < len(diff):
                    sum_diff = sum_diff - int(diff[L]) + int(diff[L + gap])
                    sum_one = sum_one - int(is_one[L]) + int(is_one[L + gap])

        out = np.empty(n, dtype=np.int64)
        out[0] = s[0]
        out[1:] = s[0] + np.cumsum(diff)
        return out.tolist()


class CamJsonSerialFixer(SerialFixer):
    """Camera JSON strategy: apply gap fixes in this order: [2, 130]."""

    def fix(self, series: List[int]) -> List[int]:
        s = list(series)
        for gap in (2, 130):
            s = self.fix_midpoints_gap(s, gap)
        return s


class FrameIDFixer(SerialFixer):

    def fix(self, series: List[int]) -> List[int]:
        """
        Unwrap frame_id-style 16-bit counters so they continue increasing
        after 65535 instead of rolling over.

        Logic:
        - Assume the only decreases are true rollovers.
        - Start counter at 0; whenever a drop is observed, counter += 1.
        - Add 65535 * counter to each element.
        """
        if not series:
            return []

        s = np.asarray(series, dtype=np.int64)
        counters = np.zeros(len(s), dtype=np.int64)

        counter = 0
        for i in range(1, len(s)):
            if s[i - 1] > s[i]:
                counter += 1
            counters[i] = counter

        fixed = s + 65535 * counters
        return fixed.tolist()


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
