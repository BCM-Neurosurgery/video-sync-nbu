#!/usr/bin/env python3
"""Estimate camera JSON sampling frequency from randomly sampled files.

The script walks a patient data tree, picks a random subset of camera JSON
files, computes FPS from the ``real_times`` field, and publishes an aggregate
summary for reporting purposes (see docs/emu/nev_vs_camera_clock.md).

Example::

    python docs/assets/nev_vs_camera_clock/sample_camera_json_fps.py \\
        --root /mnt/datalake/data/emu \\
        --sample-size 25 \\
        --pattern "*/VIDEO/**/*.json" \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:  # pragma: no cover
    from scripts.time.estimate_fps import FPSEstimate

try:
    from scripts.utility.utils import _name
except ModuleNotFoundError:  # pragma: no cover - fallback for isolated runs

    def _name(p: str | Path) -> str:
        return Path(p).name


def find_camera_json_files(root: Path, pattern: str) -> List[Path]:
    """Return JSON files under ``root`` matching ``pattern`` (glob syntax)."""
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found or not a directory: {root}")
    return [p for p in root.glob(pattern) if p.is_file()]


def _load_real_times(path: Path) -> Optional[Sequence[str]]:
    """Load ``real_times`` from ``path`` if present and non-empty."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        logging.warning("Malformed JSON %s: %s", path, exc)
        return None
    except OSError as exc:
        logging.warning("Failed to read %s: %s", path, exc)
        return None

    real_times = payload.get("real_times")
    if not real_times:
        logging.debug("Skipping %s: missing or empty real_times.", path)
        return None
    return real_times


def _estimate_fps_from_real_times(real_times: Sequence[str]):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "FPS estimation requires pandas. Install project dependencies first."
        ) from exc

    try:
        from scripts.time.estimate_fps import estimate_fps
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "FPS estimation requires pandas/numpy. Install project dependencies."
        ) from exc

    frame_df = pd.DataFrame({"real_times": real_times})
    return estimate_fps(frame_df, time_col="real_times")


def sample_camera_json_fps(
    candidates: Sequence[Path], sample_size: int, rng: random.Random
) -> List[Tuple[Path, "FPSEstimate"]]:
    """Sample up to ``sample_size`` camera JSONs with usable real_times."""
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    results: List[Tuple[Path, "FPSEstimate"]] = []

    for path in shuffled:
        if len(results) >= sample_size:
            break

        logging.debug("Inspecting %s", path)

        real_times = _load_real_times(path)
        if not real_times:
            logging.debug("Skipping %s due to missing real_times", path)
            continue

        try:
            fps = _estimate_fps_from_real_times(real_times)
        except Exception as exc:
            logging.warning("FPS estimation failed for %s: %s", path, exc)
            continue

        results.append((path, fps))
        logging.info(
            "%-60s | frames=%6d | fps_overall=%.8f | dt_mean_ms=%.3f",
            _name(path),
            fps.frames,
            round(fps.fps_overall, 8),
            round(fps.dt_mean_ms, 3),
        )

    return results


def summarise(results: Sequence[Tuple[Path, "FPSEstimate"]]) -> None:
    """Log per-file results alongside aggregate FPS statistics."""
    if not results:
        logging.error("No camera JSON FPS results collected.")
        return

    fps_overall = [entry.fps_overall for _, entry in results]
    fps_median = [entry.fps_median_inst for _, entry in results]
    dt_mean = [entry.dt_mean_ms for _, entry in results]

    logging.info("\nPer-file summary (k=%d):", len(results))
    for path, entry in results:
        logging.info(
            "  %-60s -> fps_overall=%.8f fps_median=%.8f dt_mean_ms=%.3f",
            _name(path),
            round(entry.fps_overall, 8),
            round(entry.fps_median_inst, 8),
            round(entry.dt_mean_ms, 3),
        )

    overall_pairs = list(zip(results, fps_overall))
    max_pair = max(overall_pairs, key=lambda item: item[1])
    min_pair = min(overall_pairs, key=lambda item: item[1])

    logging.info("\nAggregate statistics across sampled JSONs:")
    logging.info("  mean_fps_overall   = %.8f", round(statistics.mean(fps_overall), 8))
    logging.info(
        "  median_fps_overall = %.8f", round(statistics.median(fps_overall), 8)
    )
    logging.info("  mean_fps_median    = %.8f", round(statistics.mean(fps_median), 8))
    logging.info("  mean_dt_mean_ms    = %.3f", round(statistics.mean(dt_mean), 3))
    logging.info(
        "  max_fps_overall     = %.8f (%s)",
        round(max_pair[1], 8),
        _name(max_pair[0][0]),
    )
    logging.info(
        "  min_fps_overall     = %.8f (%s)",
        round(min_pair[1], 8),
        _name(min_pair[0][0]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate camera JSON FPS from randomly sampled files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing patient data with camera JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="*/VIDEO/**/*.json",
        help=(
            "Glob pattern (relative to root) for discovering camera JSONs. "
            "Default assumes <patient>/VIDEO/YYYYMMDD/ layout."
        ),
    )
    parser.add_argument(
        "--sample-size",
        "-k",
        type=int,
        default=10,
        help="Number of JSON files to sample for FPS estimation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling order.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    try:
        candidates = find_camera_json_files(args.root, args.pattern)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return 1

    if not candidates:
        logging.error(
            "No JSON files matching pattern %s under %s", args.pattern, args.root
        )
        return 1

    logging.info(
        "Discovered %d candidate JSON files under %s", len(candidates), args.root
    )

    rng = random.Random(args.seed)

    try:
        results = sample_camera_json_fps(candidates, args.sample_size, rng)
    except RuntimeError as exc:
        logging.error(str(exc))
        return 1

    if len(results) < args.sample_size:
        logging.warning(
            "Only %d JSON files with usable real_times processed (requested %d).",
            len(results),
            args.sample_size,
        )

    summarise(results)
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
