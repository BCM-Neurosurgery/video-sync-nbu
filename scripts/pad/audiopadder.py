#!/usr/bin/env python3
"""
AudioPadder — build an EditPlan and a fixed CSV (sample indices on the *new* timeline)
from a CSV that lists per-serial sample ranges.

Input CSV (required columns):
  - serial            : integer counter (usually +1 increments; jumps indicate losses)
  - start_sample      : start index of the serial block in the original audio timeline
  - end_sample        : end index of the serial block in the original audio timeline

Outputs (written next to the input CSV):
  - <name>-editplan.json : list of insert operations on the *original* timeline
        [{"insert_after_sample": <int>, "insert_len_samples": <int>, "note": "gap s=..."}, ...]
  - <name>-padded.csv    : fixed CSV on the *new* timeline including synthetic rows for
        missing serials; columns: serial, start_sample, end_sample, center_sample, is_synthetic

Notes
-----
- Period P (samples between serial centers) is estimated robustly from rows with Δserial==1.
  Fallback to round(44100/30)=1470 if estimation fails.
- Block length L defaults to the median of (end-start+1).
- For a gap where Δserial>1 between rows i and i+1:
    Missing M = Δserial - 1
    Observed center gap D = center[i+1] - center[i]
    Insert length S = max(0, round(Δserial*P - D)) samples inserted *after row i*.
    Synthetic rows are placed at centers: center'[i] + m*P (m=1..M) on the *new* timeline.
- Cumulative shift C is tracked so that all mapped indices in the output CSV are consistent
  with the post-insertion timeline.
- This tool does not edit audio. It only plans padding and produces the fixed index CSV.

Usage
-----
python audio_padder.py /path/to/serial_blocks.csv [--period 1470] [--block-len L]
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class EditOp:
    """A single insertion on the *original* timeline.

    Attributes
    ----------
    insert_after_sample : int
        Insert S samples immediately after this sample index (original timeline).
    insert_len_samples : int
        Number of samples to insert.
    note : str | None
        Optional note for diagnostics.
    """

    insert_after_sample: int
    insert_len_samples: int
    note: str | None = None


class AudioPadder:
    """Compute an insertion plan and a fixed CSV for serial-indexed audio blocks.

    Parameters
    ----------
    csv_path : Path
        Path to the input CSV (must contain columns: serial, start_sample, end_sample).
    period : int | None
        Optional period P in samples (center-to-center); if None, estimated from data.
    block_len : int | None
        Optional block length L; if None, estimated as median(end-start+1).
    """

    REQUIRED_COLS = ("serial", "start_sample", "end_sample")

    def __init__(
        self, csv_path: Path, period: int | None = None, block_len: int | None = None
    ):
        self.csv_path = Path(csv_path)
        self.period = period
        self.block_len = block_len

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

    # ------------------------------ Public API ------------------------------ #
    def run(self) -> Tuple[List[EditOp], pd.DataFrame, Path, Path]:
        """Execute the padding analysis and write outputs.

        Returns
        -------
        ops : List[EditOp]
            The planned insert operations (original timeline).
        fixed_df : pd.DataFrame
            The fixed CSV content (new timeline), including synthetic rows.
        out_csv : Path
            Path to the written fixed CSV (<name>-padded.csv).
        out_plan : Path
            Path to the written edit plan JSON (<name>-editplan.json).
        """
        df = self._load_csv()
        df = self._sort_by_time(df)
        centers = self._ensure_centers(df)

        P = (
            self.period
            if self.period is not None
            else self._estimate_period(df, centers)
        )
        L = (
            self.block_len
            if self.block_len is not None
            else self._estimate_block_len(df)
        )

        logging.info(f"Using period P={P} samples; block length L={L} samples")

        ops, fixed_df = self._build_plan_and_fixed(df, centers, P, L)

        out_csv, out_plan = self._emit_outputs(fixed_df, ops)

        logging.info(f"Wrote fixed CSV: {out_csv}")
        logging.info(f"Wrote edit plan: {out_plan}")
        logging.info(
            "Summary: %d input rows → %d output rows (incl. %d synthetic), %d ops, total inserted %d samples",
            len(df),
            len(fixed_df),
            int(fixed_df["is_synthetic"].sum()),
            len(ops),
            sum(op.insert_len_samples for op in ops),
        )
        return ops, fixed_df, out_csv, out_plan

    # ------------------------------ Internals ------------------------------ #
    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        for col in self.REQUIRED_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {self.csv_path}")
        return df.copy()

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sort by start_sample to enforce time order deterministically.
        return df.sort_values(by=["start_sample", "end_sample"]).reset_index(drop=True)

    def _ensure_centers(self, df: pd.DataFrame) -> np.ndarray:
        centers = np.rint(
            (df["start_sample"].to_numpy() + df["end_sample"].to_numpy()) / 2.0
        ).astype(np.int64)
        return centers

    def _estimate_period(self, df: pd.DataFrame, centers: np.ndarray) -> int:
        serials = df["serial"].to_numpy()
        d_serial = np.diff(serials)
        d_center = np.diff(centers)
        mask = (d_serial == 1) & (d_center > 0)
        candidates = d_center[mask]
        if candidates.size == 0:
            # Fallback to nominal 44100/30 ≈ 1470
            logging.warning(
                "No Δserial==1 pairs for period estimation; falling back to 1470"
            )
            return 1470
        # Robust center-to-center period: median of plausible candidates
        # Limit outliers: within 0.5x..2.0x of nominal 1470 to stabilize
        nominal = 1470
        plausible = candidates[
            (candidates >= nominal * 0.5) & (candidates <= nominal * 2.0)
        ]
        use = plausible if plausible.size > 0 else candidates
        P = int(np.rint(np.median(use)))
        if P <= 0:
            logging.warning("Estimated non-positive period; forcing to 1470")
            return 1470
        return P

    def _estimate_block_len(self, df: pd.DataFrame) -> int:
        lengths = (
            df["end_sample"].to_numpy() - df["start_sample"].to_numpy() + 1
        ).astype(np.int64)
        pos = lengths[lengths > 0]
        if pos.size == 0:
            # Conservative small block if input looks degenerate
            logging.warning("Cannot estimate block length; falling back to 64 samples")
            return 64
        return int(np.rint(np.median(pos)))

    def _build_plan_and_fixed(
        self, df: pd.DataFrame, centers: np.ndarray, P: int, L: int
    ) -> Tuple[List[EditOp], pd.DataFrame]:
        """
        Build the insertion plan (EditPlan) and the fixed CSV on the *new* timeline.


        Parameters
        ----------
        df : pd.DataFrame
        Input rows sorted by time. Must contain columns
        ``serial``, ``start_sample``, ``end_sample`` on the *original* timeline.
        centers : np.ndarray
        Per-row center sample, computed as round((start+end)/2) on the original timeline.
        P : int
        Target center-to-center period (in samples). Typically ~1470 for 44.1kHz/30Hz.
        L : int
        Typical block length in samples. Used for synthetic rows. Observed rows
        keep their own measured length (``end - start + 1``).


        Returns
        -------
        ops : List[EditOp]
        Insert operations, each anchored to the *original* timeline
        (``insert_after_sample`` is an original sample index). When applied in order
        to the audio, these ops reproduce the fixed/new timeline.
        fixed_df : pd.DataFrame
        Rows describing the *post-insertion* timeline. Columns:
        ``serial, start_sample, end_sample, center_sample, is_synthetic``.


        Algorithm (single left→right pass)
        ---------------------------------
        Let C be the cumulative number of samples inserted *before* the current point.
        1) Append the first observed row on the new timeline using ``center' = center + C``.
        2) For every adjacent pair i → i+1 on the original timeline:
        - Δs = serial[i+1] - serial[i]
        - D = center[i+1] - center[i] # observed spacing on the original timeline
        - If Δs > 1 (gap of M = Δs - 1 missing blocks):
        * Compute the insertion length
        S = max(0, round(Δs * P - D))
        (i.e., the difference between the ideal spacing Δs·P and the observed D).
        * If S > 0, record an EditOp anchored after ``end[i]`` on the original timeline.
        * Emit M synthetic rows on the *new* timeline at centers
        center[i]' + m·P, m = 1..M,
        where center[i]' = center[i] + C. Each synthetic row uses length L.
        * Update cumulative shift: C ← C + S.
        - Append the (i+1)-th observed row on the new timeline at ``center[i+1]' = center[i+1] + C``
        using that row's measured length (``end[i+1] - start[i+1] + 1``).
        3) Sort all output rows by (start_sample, end_sample, serial) and return.


        Timeline conventions & invariants
        ---------------------------------
        - EditOps are defined on the *original* timeline and must be applied in chronological order,
        maintaining a running cumulative shift to mirror C.
        - ``fixed_df`` indices are on the *new* (post-insertion) timeline. This makes the CSV directly
        consumable for downstream alignment.
        - If Δs ≤ 0, the pair is treated as no gap (a warning is logged). If S ≤ 0, no EditOp is added,
        but synthetic rows are still placed so the serial sequence is complete.


        Complexity
        ----------
        O(n) time and O(n) memory, where n is the number of input rows.
        """
        serials = df["serial"].to_numpy()
        starts = df["start_sample"].to_numpy()
        ends = df["end_sample"].to_numpy()

        C = 0  # cumulative inserted samples so far
        ops: List[EditOp] = []
        out_rows: List[Dict] = []

        n = len(df)
        if n == 0:
            # Empty input: emit empty outputs
            return ops, pd.DataFrame(columns=["serial", "start_sample", "end_sample", "center_sample", "is_synthetic"])  # type: ignore

        # Helper to append a row on NEW timeline
        def append_row(
            serial_val: int, center_new: int, is_synth: bool, L_local: int = L
        ):
            start_new = int(center_new - L_local // 2)
            end_new = int(start_new + L_local - 1)
            out_rows.append(
                {
                    "serial": int(serial_val),
                    "start_sample": int(start_new),
                    "end_sample": int(end_new),
                    "center_sample": int(center_new),
                    "is_synthetic": int(1 if is_synth else 0),
                }
            )

        # Process first row: map to new timeline with current C (0)
        center0_new = int(centers[0] + C)
        append_row(
            serials[0],
            center0_new,
            is_synth=False,
            L_local=int(ends[0] - starts[0] + 1),
        )

        for i in range(n - 1):
            s_i, s_j = int(serials[i]), int(serials[i + 1])
            c_i, c_j = int(centers[i]), int(centers[i + 1])
            start_i, end_i = int(starts[i]), int(ends[i])

            delta_s = s_j - s_i
            D = c_j - c_i  # observed center gap on original timeline

            if delta_s <= 0:
                logging.warning(
                    "Non-forward serial pair at rows %d→%d (Δs=%d). Treating as no-gap.",
                    i,
                    i + 1,
                    delta_s,
                )
                # No insertion; next observed row will be appended after we compute its new center
                pass
            elif delta_s == 1:
                # No insertion
                pass
            else:
                # Gap detected: plan insertion after row i
                M = delta_s - 1
                # Number of audio samples to insert as padding
                S = int(max(0, round(delta_s * P - D)))
                if S > 0:
                    ops.append(
                        EditOp(
                            insert_after_sample=end_i,  # original timeline anchor
                            insert_len_samples=S,
                            note=f"gap Δserial={delta_s} around serial {s_i}->{s_j}",
                        )
                    )
                else:
                    logging.debug(
                        "Computed non-positive insert length S=%d at rows %d→%d; skipping op.",
                        S,
                        i,
                        i + 1,
                    )
                # Emit synthetic rows on the NEW timeline, centered at c_i' + m*P
                c_i_new = int(c_i + C)
                for m in range(1, M + 1):
                    c_syn = int(round(c_i_new + m * P))
                    append_row(s_i + m, c_syn, is_synth=True)
                # After placing synthetics, future rows shift by S
                C += max(0, S)

            # Append the (i+1)-th observed row on the NEW timeline
            center_next_new = int(centers[i + 1] + C)
            L_next = int(ends[i + 1] - starts[i + 1] + 1)
            append_row(serials[i + 1], center_next_new, is_synth=False, L_local=L_next)

        fixed_df = pd.DataFrame(
            out_rows,
            columns=[
                "serial",
                "start_sample",
                "end_sample",
                "center_sample",
                "is_synthetic",
            ],
        )
        # Ensure strictly increasing by start on the new axis
        fixed_df = fixed_df.sort_values(
            by=["start_sample", "end_sample", "serial"]
        ).reset_index(drop=True)
        return ops, fixed_df

    def _emit_outputs(
        self, fixed_df: pd.DataFrame, ops: List[EditOp]
    ) -> Tuple[Path, Path]:
        stem = self.csv_path.with_suffix("")
        out_csv = stem.with_name(stem.name + "-padded.csv")
        out_plan = stem.with_name(stem.name + "-editplan.json")

        fixed_df.to_csv(out_csv, index=False)

        serializable_ops = [
            {
                "insert_after_sample": int(op.insert_after_sample),
                "insert_len_samples": int(op.insert_len_samples),
                **({"note": op.note} if op.note else {}),
            }
            for op in ops
        ]
        with open(out_plan, "w", encoding="utf-8") as f:
            json.dump(serializable_ops, f, indent=2)
        return out_csv, out_plan


# --------------------------------- CLI --------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build edit plan and fixed CSV for serial-indexed audio blocks."
    )
    p.add_argument(
        "csv", type=Path, help="Input CSV with columns: serial,start_sample,end_sample"
    )
    p.add_argument(
        "--period",
        type=int,
        default=None,
        help="Override period P (samples between serial centers)",
    )
    p.add_argument(
        "--block-len",
        type=int,
        default=None,
        help="Override block length L (samples per block)",
    )
    p.add_argument(
        "--log",
        dest="loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return p


def main():
    ap = _build_argparser()
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel), format="%(levelname)s: %(message)s"
    )

    padder = AudioPadder(
        csv_path=args.csv, period=args.period, block_len=args.block_len
    )
    try:
        padder.run()
    except Exception as e:
        logging.error("Failed: %s", e)
        raise


if __name__ == "__main__":
    main()
