from __future__ import annotations
from abc import ABC
from typing import List, Tuple, Iterable
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
