from __future__ import annotations
from pathlib import Path
from typing import List, Optional


class SerialFixer:
    """
    Utilities for repairing integer serial sequences via sliding-window rules.

    Overview
    --------
    For a given gap k (>=2), consider a window bounded by indices:
        L = i - 1, R = i + (k - 1)   with interior i..R-1
    If s[R] - s[L] == k, enforce a strict +1 run on the interior:
        s[i]   = s[L] + 1
        s[i+1] = s[L] + 2
        ...
        s[R-1] = s[L] + (k - 1)
    Endpoints L and R are not changed.
    """

    def __init__(self, series: List[int]) -> None:
        self._orig = list(series)

    def fix_midpoints_gap(self, gap: int) -> List[int]:
        """
        Apply the sliding-window interior correction for a given `gap` (k >= 2).

        Returns a new list; single left-to-right pass.
        """
        if gap < 2:
            return list(self._orig)

        n = len(self._orig)
        if n < gap + 1:
            return list(self._orig)

        s = self._orig[:]

        # i indexes the first interior element; L=i-1, R=i+(gap-1)
        for i in range(1, n - gap + 1):
            L = i - 1
            R = i + (gap - 1)
            if s[R] - s[L] != gap:
                continue

            # Check interior; repair if any mismatch
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

    # Convenience wrappers for common gaps
    def fix_midpoints_gap2(self) -> List[int]:
        return self.fix_midpoints_gap(2)

    def fix_midpoints_gap3(self) -> List[int]:
        return self.fix_midpoints_gap(3)

    def fix_midpoints_gap4(self) -> List[int]:
        return self.fix_midpoints_gap(4)

    def fix_midpoints_gap5(self) -> List[int]:
        return self.fix_midpoints_gap(5)


def fix_csv(argv: Optional[List[str]] = None) -> None:
    """
    CLI: read CSV (serial,start_sample,end_sample), apply sliding-window fixes
    sequentially for gaps 2, 3, 4, 5 (one pass each), and write '<stem>-fixed.csv'.

    Usage:
        python script.py input.csv
        python script.py input.csv -o /path/to/out.csv
    """
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(
        description=(
            "Fix 'serial' column using sliding-window rules (gap=2,3,4,5; one pass each)"
        )
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
        s = SerialFixer(s).fix_midpoints_gap(gap)

    df["serial"] = s
    df.to_csv(out_path, index=False)
    print(str(out_path))


if __name__ == "__main__":
    fix_csv()
