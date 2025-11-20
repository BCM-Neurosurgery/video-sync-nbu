#!/usr/bin/env python3
"""Estimate NEV serial sampling frequency from randomly sampled files.

This script scans a patient root directory for NEV files (default pattern
"*NSP-1.nev"), randomly selects up to ``k`` files that contain serial data
(InsertionReason == 129), builds the serial dataframe using ``Nev`` and
summarises the effective FPS metrics via ``estimate_fps``.

Usage example::

    python docs/assets/nev_vs_camera_clock/sample_serial_fps.py \
        --root /mnt/stitched/EMU-18112 \
        --sample-size 10 \
        --seed 42

The output includes per-file FPS values and an averaged summary that can be
referenced from docs/emu/nev_vs_camera_clock.md.
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:  # pragma: no cover - for static checkers only
    from scripts.time.estimate_fps import FPSEstimate

try:
    from scripts.utility.utils import _name
except ModuleNotFoundError:  # pragma: no cover - defensive fallback

    def _name(p: Path | str) -> str:
        return Path(p).name


def find_nev_files(root: Path, pattern: str) -> List[Path]:
    """Return NEV files under ``root`` matching ``pattern``."""
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found or not a directory: {root}")
    return [p for p in root.rglob(pattern) if p.is_file()]


def sample_serial_fps(
    candidates: Sequence[Path],
    sample_size: int,
    rng: random.Random,
) -> List[Tuple[Path, "FPSEstimate"]]:
    """Sample up to ``sample_size`` NEV files that produce serial FPS stats."""
    try:
        from scripts.parsers.nevfileparser import Nev
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "NEV parsing requires 'brpylib'. Install project dependencies first."
        ) from exc

    try:
        from scripts.time.estimate_fps import estimate_fps
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "FPS estimation depends on numpy/pandas. Install project deps first."
        ) from exc

    shuffled = list(candidates)
    rng.shuffle(shuffled)

    results: List[Tuple[Path, "FPSEstimate"]] = []

    for path in shuffled:
        if len(results) >= sample_size:
            break

        try:
            nev = Nev(str(path))
        except Exception as exc:
            logging.warning("Failed to read %s: %s", path, exc)
            continue

        if not nev.has_unparsed_data():
            logging.debug("Skipping %s: no serial UnparsedData payload.", path)
            continue

        try:
            serial_df = nev.get_chunk_serial_df()
        except Exception as exc:
            logging.warning("Serial reconstruction failed for %s: %s", path, exc)
            continue

        if serial_df.empty:
            logging.debug(
                "Skipping %s: serial dataframe empty after reconstruction.", path
            )
            continue

        try:
            fps = estimate_fps(serial_df, time_col="UTCTimeStamp")
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
    """Log an aggregate summary for the collected FPS estimates."""
    if not results:
        logging.error("No NEV files with serial FPS measurements collected.")
        return

    fps_overall = [entry.fps_overall for _, entry in results]
    fps_median = [entry.fps_median_inst for _, entry in results]
    dt_mean = [entry.dt_mean_ms for _, entry in results]

    overall_pairs = list(zip(results, fps_overall))
    max_pair = max(overall_pairs, key=lambda item: item[1])
    min_pair = min(overall_pairs, key=lambda item: item[1])

    logging.info("\nPer-file summary (k=%d):", len(results))
    for path, entry in results:
        logging.info(
            "  %-60s -> fps_overall=%.8f fps_median=%.8f dt_mean_ms=%.3f",
            _name(path),
            round(entry.fps_overall, 8),
            round(entry.fps_median_inst, 8),
            round(entry.dt_mean_ms, 3),
        )

    logging.info("\nAggregate statistics across sampled NEVs:")
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
    ap = argparse.ArgumentParser(
        description="Estimate NEV serial sampling frequency from random samples."
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing patient folders with NEV files.",
    )
    ap.add_argument(
        "--sample-size",
        "-k",
        type=int,
        default=5,
        help="Number of NEV files with serials to sample.",
    )
    ap.add_argument(
        "--pattern",
        default="*NSP-1.nev",
        help="Glob pattern for NEV discovery (default: '*NSP-1.nev').",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling order.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    try:
        candidates = find_nev_files(args.root, args.pattern)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return 1

    if not candidates:
        logging.error("No NEV files matching %s under %s", args.pattern, args.root)
        return 1

    logging.info("Discovered %d NEV files under %s", len(candidates), args.root)

    rng = random.Random(args.seed)
    try:
        results = sample_serial_fps(candidates, args.sample_size, rng)
    except RuntimeError as exc:
        logging.error(str(exc))
        return 1

    if len(results) < args.sample_size:
        logging.warning(
            "Only %d NEV files with serials were processed (requested %d).",
            len(results),
            args.sample_size,
        )

    summarise(results)
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
