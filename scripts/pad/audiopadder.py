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
  - <name>-padded.csv    : fixed CSV on the *new* timeline (optionally includes synthetic rows)
        columns: serial, start_sample, end_sample, center_sample, is_synthetic

Notes
-----
**Exact-gap principle (no drift)**
You can choose how to compute the *ideal* time to fill per gap (Δserial > 1):

- **video** (default): assume each serial corresponds to one video frame.
  Then the missing time is exactly `Δserial / fps` seconds. Given audio sample
  rate `sr`, the ideal span in samples is `round(Δserial * sr / fps)`.
- **local**: estimate a *local* period `P_local` from nearby Δserial==1 pairs
  and use `round(Δserial * P_local)`.
- **global**: use a global period hint `P_global` (fallback to an estimate if not provided).
- **budget**: *control the final total duration*. Compute observed total samples from the CSV 
  (`observed = last_end - first_start + 1`). With `--frames` and `--fps`, 
  target samples are `target = round((frames / fps) * sample_rate)`. 
  The total insertion budget is `B = max(0, target - observed)` and 
  is **distributed across gaps proportional to their missing frames** `M_i = (Δserial_i - 1)` 
  (remainder assigned to the largest fractional shares). 
  If there are **no gaps**, insert all `B` samples at the **tail**. 
  If `target <= observed`, no samples are inserted.

For a gap between rows *i* and *i+1* with observed center gap `D = center[i+1]-center[i]`,
we insert `S = max(0, ideal_span - D)` samples *after* row *i* on the original timeline.
This ensures the filled time equals the intended missing time with no cumulative drift.

- Block length L defaults to the median of (end-start+1) unless overridden.
- Synthetic rows (if enabled) are placed by **evenly partitioning the ideal span**:
  step = ideal_span / Δserial (not a fixed constant). This guarantees the last synthetic
  lands exactly at `center[i]' + ideal_span` on the new timeline.
- Cumulative shift C is tracked so indices in the output CSV are consistent with the
  post-insertion timeline.
- This tool does not edit audio. It only plans padding and produces the fixed index CSV.

Usage
-----
python audio_padder.py /path/to/serial_blocks.csv \
  [--gap-policy video|local|global|budget] [--fps 30.0] [--sample-rate 44100] \
  [--period 1470] [--block-len L] [--no-synth] [--frames N]
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

from scripts.log.logutils import configure_standalone_logging

logger = logging.getLogger(__name__)


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
        Optional *global* period hint in samples (center-to-center); used only as a
        fallback when local estimation around a gap is impossible.
    block_len : int | None
        Optional block length L; if None, estimated as median(end-start+1).
    include_synthetic : bool
        If True (default), the output CSV includes synthetic rows for missing serials
        whose centers are linearly spaced across the *ideal* span of each gap. If False,
        only observed rows are emitted on the new timeline.
    gap_policy : str
        One of {"video", "local", "global", "budget"}. See module notes.
    sample_rate : int
        Audio sample rate in Hz, used when gap_policy="video". Default 44100.
    fps : float
        Video frame rate, used when gap_policy="video". Default 30.0.
    target_frames : int | None
        Required when gap_policy="budget": total frame count that sets the target duration.
    """

    REQUIRED_COLS = ("serial", "start_sample", "end_sample")

    def __init__(
        self,
        csv_path: Path,
        period: int | None = None,
        block_len: int | None = None,
        include_synthetic: bool = True,
        gap_policy: str = "video",
        sample_rate: int = 44100,
        fps: float = 30.0,
        target_frames: int | None = None,
        local_window: int = 3,
    ):
        self.csv_path = Path(csv_path)
        self.period = period
        self.block_len = block_len
        self.include_synthetic = include_synthetic
        self.gap_policy = gap_policy.lower()
        if self.gap_policy not in {"video", "local", "global", "budget"}:
            raise ValueError("gap_policy must be one of: video, local, global, budget")
        self.sample_rate = int(sample_rate)
        self.fps = float(fps)
        self.target_frames = target_frames
        self.local_window = int(local_window)

        if self.gap_policy == "budget" and self.target_frames is None:
            raise ValueError("When gap_policy='budget', you must specify --frames.")

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
            The fixed CSV content (new timeline), including synthetic rows when enabled.
        out_csv : Path
            Path to the written fixed CSV (<name>-padded.csv).
        out_plan : Path
            Path to the written edit plan JSON (<name>-editplan.json).
        """
        df = self._load_csv()
        df = self._sort_by_time(df)
        centers = self._ensure_centers(df)

        P_global = (
            self.period
            if self.period is not None
            else self._estimate_period(df, centers)
        )
        L = (
            self.block_len
            if self.block_len is not None
            else self._estimate_block_len(df)
        )

        logger.info(
            "Using fallback global period P=%d samples; block length L=%d samples; gap_policy=%s (sr=%d, fps=%.3f)",
            P_global,
            L,
            self.gap_policy,
            self.sample_rate,
            self.fps,
        )

        ops, fixed_df = self._build_plan_and_fixed(df, centers, P_global, L)

        out_csv, out_plan = self._emit_outputs(fixed_df, ops)

        logger.info("Wrote fixed CSV: %s", Path(out_csv).name)
        logger.info("Wrote edit plan: %s", Path(out_plan).name)
        logger.info(
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
            logger.warning(
                "No Δserial==1 pairs for period estimation; falling back to 1470"
            )
            return 1470
        nominal = 1470
        plausible = candidates[
            (candidates >= nominal * 0.5) & (candidates <= nominal * 2.0)
        ]
        use = plausible if plausible.size > 0 else candidates
        P = int(np.rint(np.median(use)))
        if P <= 0:
            logger.warning("Estimated non-positive period; forcing to 1470")
            return 1470
        return P

    def _estimate_block_len(self, df: pd.DataFrame) -> int:
        lengths = (
            df["end_sample"].to_numpy() - df["start_sample"].to_numpy() + 1
        ).astype(np.int64)
        pos = lengths[lengths > 0]
        if pos.size == 0:
            logger.warning("Cannot estimate block length; falling back to 64 samples")
            return 64
        return int(np.rint(np.median(pos)))

    def _build_plan_and_fixed(
        self, df: pd.DataFrame, centers: np.ndarray, P_global: int, L: int
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
        P_global : int
            Fallback global period (samples). Used only when local estimation fails.
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
           - D  = center[i+1] - center[i]   # observed spacing on the original timeline
           - If Δs > 1 (gap of M = Δs - 1 missing blocks):
             * Compute `ideal_span` per `gap_policy`:
               - video  → `round(Δs * sample_rate / fps)`
               - local  → `round(Δs * P_local)` where `P_local` is estimated from nearby Δs==1 pairs
               - global → `round(Δs * P_global)`
             * Let `S = max(0, ideal_span - D)` and record an EditOp (if S>0) anchored after `end[i]`.
             * If synthetic rows are enabled, place them on the **new** timeline by slicing
               the ideal span evenly: centers at `center[i]' + m * (ideal_span/Δs)` for m=1..M.
             * Update cumulative shift: `C ← C + S`.
           - Append the (i+1)-th observed row on the new timeline at ``center[i+1]' = center[i+1] + C``
             using that row's measured length (``end[i+1] - start[i+1] + 1``).
        3) Sort all output rows by (start_sample, end_sample, serial) and return.

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
            return ops, pd.DataFrame(
                columns=[
                    "serial",
                    "start_sample",
                    "end_sample",
                    "center_sample",
                    "is_synthetic",
                ]
            )  # type: ignore

        # --------- budget policy precomputation ---------
        per_gap_budget: Dict[int, int] = {}
        tail_budget = 0
        if self.gap_policy == "budget":
            observed_total_samples = int(ends[-1] - starts[0] + 1)
            target_total_samples = int(
                round((self.target_frames * self.sample_rate) / self.fps)  # type: ignore[arg-type]
            )
            total_insert_budget = max(0, target_total_samples - observed_total_samples)

            # Find all gap indices (i where Δserial > 1) and their missing frames
            gap_indices: List[int] = []
            missing_frames: List[int] = []
            for i in range(n - 1):
                ds = int(serials[i + 1]) - int(serials[i])
                if ds > 1:
                    gap_indices.append(i)
                    missing_frames.append(ds - 1)

            if gap_indices:
                # Proportional allocation by missing frames (Δserial-1)
                total_missing = int(np.sum(missing_frames)) if missing_frames else 0
                if total_insert_budget > 0 and total_missing > 0:
                    weights = np.array(missing_frames, dtype=np.int64)
                    shares = (total_insert_budget * weights) / float(total_missing)
                    floors = np.floor(shares).astype(int)
                    rem = int(total_insert_budget - int(floors.sum()))
                    if rem > 0:
                        frac = shares - floors
                        order = np.argsort(-frac)  # largest fractional parts first
                        for k in range(rem):
                            floors[int(order[k])] += 1
                    for idx, gap_i in enumerate(gap_indices):
                        per_gap_budget[gap_i] = int(floors[idx])
                else:
                    for gap_i in gap_indices:
                        per_gap_budget[gap_i] = 0

                base = total_insert_budget // len(gap_indices)
                rem = total_insert_budget - base * len(gap_indices)
                logger.info(
                    "budget policy: observed=%d samples, target=%d samples, total_insert=%d distributed across %d gaps (base=%d, rem=%d)",
                    observed_total_samples,
                    target_total_samples,
                    total_insert_budget,
                    len(gap_indices),
                    base,
                    rem,
                )
            else:
                # No gaps → insert once at tail
                tail_budget = total_insert_budget
                logger.info(
                    "budget policy: no gaps → tail insertion of %d samples (observed=%d, target=%d)",
                    tail_budget,
                    observed_total_samples,
                    target_total_samples,
                )

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

        # First row → new timeline
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
                logger.warning(
                    "Non-forward serial pair at rows %d→%d (Δs=%d). Treating as no-gap.",
                    i,
                    i + 1,
                    delta_s,
                )
            elif delta_s == 1:
                pass
            else:
                # Gap detected
                M = delta_s - 1
                if self.gap_policy == "video":
                    ideal_span = int(round(delta_s * (self.sample_rate / self.fps)))
                    S = int(max(0, ideal_span - D))
                elif self.gap_policy == "local":
                    P_local = self._estimate_local_period(
                        df,
                        centers,
                        i_left=i,
                        i_right=i + 1,
                        default_P=P_global,
                        window=self.local_window,
                    )
                    ideal_span = int(round(delta_s * P_local))
                    S = int(max(0, ideal_span - D))
                elif self.gap_policy == "global":
                    ideal_span = int(round(delta_s * P_global))
                    S = int(max(0, ideal_span - D))
                else:  # budget
                    S = int(per_gap_budget.get(i, 0))
                    ideal_span = int(D + S)

                if S > 0:
                    ops.append(
                        EditOp(
                            insert_after_sample=end_i,
                            insert_len_samples=S,
                            note=f"gap Δserial={delta_s} (policy={self.gap_policy}) around serial {s_i}->{s_j}",
                        )
                    )
                else:
                    logger.debug(
                        "Computed non-positive insert length S=%d at rows %d→%d; skipping op.",
                        S,
                        i,
                        i + 1,
                    )

                # Synthetic rows (NEW timeline) by slicing the ideal span evenly
                if self.include_synthetic and M > 0:
                    c_i_new = int(c_i + C)
                    step = ideal_span / float(delta_s)
                    for m in range(1, M + 1):
                        c_syn = int(round(c_i_new + m * step))
                        append_row(s_i + m, c_syn, is_synth=True)

                C += max(0, S)

            # Append next observed row on the NEW timeline
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
        fixed_df = fixed_df.sort_values(
            by=["start_sample", "end_sample", "serial"]
        ).reset_index(drop=True)
        return ops, fixed_df

    def _estimate_local_period(
        self,
        df: pd.DataFrame,
        centers: np.ndarray,
        i_left: int,
        i_right: int,
        default_P: int,
        window: int = 3,
    ) -> int:
        """Estimate a *local* center-to-center period around a gap.

        Looks at up to `window` Δserial==1 pairs immediately before `i_left`
        and after `i_right` and takes the median of their center gaps. If both
        sides are available, returns a length-weighted average of the two
        medians. If neither side is available, falls back to `default_P`.
        """
        serials = df["serial"].to_numpy()
        left_d: List[int] = []
        k = i_left - 1
        while k >= 0 and len(left_d) < window:
            if serials[k + 1] - serials[k] == 1:
                left_d.append(int(centers[k + 1] - centers[k]))
            k -= 1
        right_d: List[int] = []
        k = i_right
        while k + 1 < len(df) and len(right_d) < window:
            if serials[k + 1] - serials[k] == 1:
                right_d.append(int(centers[k + 1] - centers[k]))
            k += 1
        vals = []
        weights = []
        if left_d:
            vals.append(np.median(left_d))
            weights.append(len(left_d))
        if right_d:
            vals.append(np.median(right_d))
            weights.append(len(right_d))
        if not vals:
            return int(default_P)
        if len(vals) == 1:
            return int(round(vals[0]))
        P_local = float(np.average(vals, weights=weights))
        return int(round(P_local))

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
        "--gap-policy",
        choices=["video", "local", "global", "budget"],
        default="video",
        help="How to compute ideal span per gap",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate when gap-policy=video/budget",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate when gap-policy=video/budget",
    )
    p.add_argument(
        "--period",
        type=int,
        default=None,
        help="Override *global* fallback period (samples between serial centers)",
    )
    p.add_argument(
        "--block-len",
        type=int,
        default=None,
        help="Override block length L (samples per block)",
    )
    p.add_argument(
        "--no-synth",
        action="store_true",
        help="Do not emit synthetic rows; only map observed rows to the new timeline",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Total frame count for the target duration (required when --gap-policy=budget)",
    )
    p.add_argument(
        "--local-window",
        type=int,
        default=3,
        help="Window size (per side) for local period estimation when --gap-policy=local",
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

    # Standalone console logging that won't interfere with the driver
    configure_standalone_logging(level=args.loglevel, seg="-", cam="-")

    padder = AudioPadder(
        csv_path=args.csv,
        period=args.period,
        block_len=args.block_len,
        include_synthetic=not args.no_synth,
        gap_policy=args.gap_policy,
        sample_rate=args.sample_rate,
        fps=args.fps,
        target_frames=args.frames,
        local_window=args.local_window,
    )
    try:
        padder.run()
    except Exception as e:
        logger.error("Failed: %s", e)
        raise


if __name__ == "__main__":
    main()
