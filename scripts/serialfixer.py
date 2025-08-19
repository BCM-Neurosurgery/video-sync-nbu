from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


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

    Subclasses implement a `fix()` strategy by choosing which gaps to apply
    (and in what order), typically chaining multiple passes.
    """

    @abstractmethod
    def fix(self, series: List[int]) -> List[int]:
        """Return a new list with this strategy’s fixes applied."""
        raise NotImplementedError

    @staticmethod
    def fix_midpoints_gap(series: List[int], gap: int) -> List[int]:
        """
        Apply one left→right pass of the sliding-window rule for a given gap.

        Parameters
        ----------
        series : List[int]
            Input integer sequence.
        gap : int
            Neighbor span k (>= 2). Enforces a +1 interior when s[R]-s[L] == k.

        Returns
        -------
        List[int]
            A new list with corrections applied for this single pass.

        Examples
        --------
        >>> SerialFixer.fix_midpoints_gap([1, 99, 3, 4], 2)
        [1, 2, 3, 4]
        >>> SerialFixer.fix_midpoints_gap([5, 6, 6, 8], 3)
        [5, 6, 7, 8]
        """
        if gap < 2:
            return list(series)

        n = len(series)
        if n < gap + 1:  # need at least L + interior(k-1) + R → k+1 points
            return list(series)

        s = list(series)  # work on a copy

        # i indexes the first interior element; L=i-1, R=i+(gap-1)
        # Valid i: 1 .. n-gap
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
    """
    Camera JSON strategy: apply gap fixes in this order: [2, 129].

    Rationale
    ---------
    - gap=2   : classic midpoint correction.
    - gap=129 : enforces long interior runs bounded by endpoints spaced by 129.
                (Interior length is 128.) This matches use cases where camera
                metadata should be strictly +1 within larger spans.

    Examples
    --------
    >>> CamJsonSerialFixer().fix([5, 6, 6, 8])
    [5, 6, 7, 8]
    """

    def fix(self, series: List[int]) -> List[int]:
        s = list(series)
        for gap in (2, 129):
            s = self.fix_midpoints_gap(s, gap)
        return s


class AudioSerialFixer(SerialFixer):
    """
    Audio strategy: apply a cascade of gap fixes for k = 2..10 (inclusive).

    This progressively tightens the interior constraints from short to longer
    windows, each in a single pass.

    Examples
    --------
    >>> AudioSerialFixer().fix([5, 6, 6, 8, 9, 10, 11])
    [5, 6, 7, 8, 9, 10, 11]
    >>> AudioSerialFixer().fix([2, 3, 5, 5, 5, 7])
    [2, 3, 4, 5, 6, 7]
    """

    def fix(self, series: List[int]) -> List[int]:
        s = list(series)
        for gap in range(2, 11):
            s = self.fix_midpoints_gap(s, gap)
        return s


def fix_csv(argv: Optional[List[str]] = None) -> None:
    """
    CLI: read CSV (serial,start_sample,end_sample), apply sliding-window fixes
    sequentially for gaps 2, 3, 4, 5 (one pass each), and write '<stem>-fixed.csv'.

    Usage
    -----
    python script.py input.csv
    python script.py input.csv -o /path/to/out.csv

    Notes
    -----
    - This CLI currently applies a conservative default (gaps 2..5).
      If you want the AudioSerialFixer (2..10) or CamJsonSerialFixer ([2,129]),
      call those classes directly in your pipeline code.
    """
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(
        description="Fix 'serial' column using sliding-window rules"
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
        serial_list = [int(x) for x in df["serial"].tolist()]
    except Exception as exc:
        raise SystemExit("ERROR: column 'serial' must contain integers.") from exc

    s = serial_list
    for gap in range(2, 11):
        s = SerialFixer.fix_midpoints_gap(s, gap)

    df["serial"] = s
    df.to_csv(out_path, index=False)
    print(str(out_path))


if __name__ == "__main__":
    fix_csv()
