"""Utilities for estimating effective video FPS from timestamp data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pandas.api.types import is_datetime64_any_dtype
from typing import Optional


@dataclass
class FPSEstimate:
    """Container for FPS and interval statistics."""

    frames: int
    duration_s: float
    fps_overall: float  # (N-1) / total_duration
    fps_median_inst: float  # median of instantaneous FPS = median(1/dt)
    fps_mean_inst: float  # mean of instantaneous FPS
    fps_harmonic: float  # 1 / mean(dt)
    dt_mean_ms: float
    dt_std_ms: float
    dt_p5_ms: float
    dt_p95_ms: float


def estimate_fps(df: pd.DataFrame, time_col: str = "UTCTimeStamp") -> FPSEstimate:
    """
    Estimate actual frames-per-second (FPS) from a timestamped sequence.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a datetime-like column (ns precision is fine).
    time_col : str, default "UTCTimeStamp"
        Column name with pandas datetime64[ns], Timestamp objects, or strings
        parseable by ``pd.to_datetime`` (e.g. "2025-04-23 20:00:45.026").

    Returns
    -------
    FPSEstimate
        A dataclass with overall FPS (recommended) and alternative estimates.

    Notes
    -----
    - `fps_overall` = (N-1) / (t_last - t_first) is usually the most stable.
    - Instantaneous FPS uses 1 / Δt between consecutive timestamps.
    - Non-positive or NaN intervals (duplicates/out-of-order) are ignored.
    """
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame.")

    # Clean & order the time series
    series = df[time_col]

    if not is_datetime64_any_dtype(series):
        # Allow strings such as "2025-04-23 20:00:45.026" to be parsed.
        series = pd.to_datetime(series, errors="coerce")

    s = (
        series.dropna()
        .astype("datetime64[ns]")
        .sort_values(kind="mergesort")  # stable sort
    )

    if len(s) < 2:
        raise ValueError("Need at least two valid timestamps to estimate FPS.")

    # Consecutive intervals in seconds
    dt = s.diff().dt.total_seconds()
    dt = dt[dt > 0]  # drop first NaN and any non-positive gaps

    if len(dt) == 0:
        raise ValueError("No positive time intervals found (check timestamps).")

    # Overall FPS over the entire span
    duration_s = (s.iloc[-1] - s.iloc[0]).total_seconds()
    fps_overall = (len(s) - 1) / duration_s if duration_s > 0 else np.nan

    # Instantaneous FPS (per-interval) based stats
    inst_fps = 1.0 / dt.to_numpy()
    fps_median_inst = float(np.median(inst_fps))
    fps_mean_inst = float(np.mean(inst_fps))
    fps_harmonic = float(len(dt) / dt.sum())

    # Jitter summary in milliseconds
    dt_seconds = dt.to_numpy()
    dt_mean_ms = float(dt_seconds.mean() * 1000.0)
    dt_std_ms = float(dt_seconds.std(ddof=1) * 1000.0) if len(dt_seconds) > 1 else 0.0
    dt_p5_ms = float(np.percentile(dt_seconds, 5) * 1000.0)
    dt_p95_ms = float(np.percentile(dt_seconds, 95) * 1000.0)

    return FPSEstimate(
        frames=len(s),
        duration_s=duration_s,
        fps_overall=float(fps_overall),
        fps_median_inst=fps_median_inst,
        fps_mean_inst=fps_mean_inst,
        fps_harmonic=fps_harmonic,
        dt_mean_ms=dt_mean_ms,
        dt_std_ms=dt_std_ms,
        dt_p5_ms=dt_p5_ms,
        dt_p95_ms=dt_p95_ms,
    )
