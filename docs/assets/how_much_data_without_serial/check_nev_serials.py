#!/usr/bin/env python3
"""
Check NEV digital events for serial presence (InsertionReason == 129).

For each patient directory under ROOT (e.g., /mnt/stitched/EMU-18112/YFG/),
randomly sample up to N (default 10) NEV files that end with 'NSP-1.nev'
from any task subfolders, read their digital events using videosync.nev.Nev,
and determine whether any sampled file contains InsertionReason == 129.

If none of the sampled files contain 129 for a patient, the patient ID
is logged as a warning and listed in the final summary.

Usage:
    python check_nev_serials.py \
        --root /mnt/stitched/EMU-18112 \
        --sample-size 10 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple
from scripts.parsers.nevfileparser import Nev


def find_patient_dirs(root: Path) -> List[Path]:
    """Return immediate subdirectories of `root` that look like patient folders."""
    # Keep it simple: treat all immediate subdirs as patients.
    return [p for p in root.iterdir() if p.is_dir()]


def find_nev_files(patient_dir: Path) -> List[Path]:
    """Find NEV files ending with 'NSP-1.nev' anywhere under a patient directory."""
    return [p for p in patient_dir.rglob("*NSP-1.nev") if p.is_file()]


def nev_has_serial_129(nev_path: Path) -> bool:
    """Return True if this NEV's digital events contain any InsertionReason == 129."""
    nev = Nev(str(nev_path))
    df = nev.get_digital_events_df()
    if "InsertionReason" not in df.columns:
        return False
    # Coerce to numeric just in case; compare to 129.
    try:
        vals = df["InsertionReason"]
        # Handle mixed types robustly
        vals_num = vals.apply(lambda x: int(x) if str(x).strip().isdigit() else None)
        return (vals_num == 129).any()
    except Exception:
        return False


def sample_nevs(nevs: List[Path], k: int, rng: random.Random) -> List[Path]:
    """Sample up to k distinct NEV paths (or all if fewer)."""
    if not nevs:
        return []
    if len(nevs) <= k:
        return nevs
    return rng.sample(nevs, k)


def check_patient(
    patient_dir: Path, k: int, rng: random.Random
) -> Tuple[str, int, int]:
    """
    Check up to k NEV files for a single patient, but break early if any file
    shows InsertionReason == 129.

    Returns:
        (patient_id, n_checked, n_with_129)   # n_with_129 ∈ {0,1} due to early exit
    """
    patient_id = patient_dir.name
    nevs = find_nev_files(patient_dir)
    sampled = sample_nevs(nevs, k, rng)

    n_checked = 0
    found_129 = False

    for nev_path in sampled:
        try:
            n_checked += 1
            if nev_has_serial_129(nev_path):
                found_129 = True
                break  # early exit: we’re confident serials exist
        except Exception as e:
            logging.warning("Failed to read %s (%s)", nev_path, e)

    if sampled and not found_129:
        logging.warning(
            "No serial (InsertionReason=129) found for patient %s in %d checked NEVs.",
            patient_id,
            n_checked,
        )
    elif not sampled:
        logging.warning("No NEV files found for patient %s.", patient_id)

    return patient_id, n_checked, int(found_129)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scan NEVs per patient and flag those lacking InsertionReason=129."
    )
    ap.add_argument(
        "--root", required=True, help="Root folder (e.g., /mnt/stitched/EMU-18112)"
    )
    ap.add_argument(
        "--sample-size", type=int, default=10, help="NEVs to sample per patient (max)."
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    ap.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    rng = random.Random(args.seed)
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root not found or not a directory: {root}")

    patients = find_patient_dirs(root)
    logging.info("Found %d patient folders under %s", len(patients), root)

    flagged = []  # patients with zero 129 in all sampled NEVs
    summary_rows = []

    for pdir in sorted(patients):
        pid, n_sampled, n_with_129 = check_patient(pdir, args.sample_size, rng)
        summary_rows.append((pid, n_sampled, n_with_129))
        if n_sampled > 0 and n_with_129 == 0:
            flagged.append(pid)

    # Pretty summary
    print("\nSummary (patient, checked, with_129):")
    for pid, n_checked, n_with_129 in summary_rows:
        print(f"  {pid:>8}  {n_checked:3d}  {n_with_129:3d}")

    flagged = [
        pid
        for pid, n_checked, n_with_129 in summary_rows
        if n_checked > 0 and n_with_129 == 0
    ]

    if flagged:
        print("\nPatients with NO InsertionReason=129 in sampled NEVs:")
        for pid in flagged:
            print(f"  - {pid}")
    else:
        print(
            "\nNo patients flagged (every sampled set had at least one file with 129)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
