"""
scan.find_camera_serials
========================

Utility that samples EMU camera JSON companions to report which camera serials
are present per patient. Useful for quickly auditing large recording trees.

The script expects a directory structure that looks like::

    /mnt/datalake/data/emu/
        YFKDatafile/
            VIDEO/
                20240201/
                    segment_*.json
                20240202/
                    segment_*.json

For each patient directory matching ``Y*Datafile`` the tool gathers every JSON
within ``VIDEO/YYYYMMDD`` folders, samples ``k`` of them (default 5), and
reports the camera serials observed. Results are written to a single JSON file
containing per-patient summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from scripts.parsers.jsonfileparser import JsonParser

LOGGER = logging.getLogger(__name__)

DATE_DIR_RE = re.compile(r"\d{8}")
PATIENT_DIR_RE = re.compile(r"^Y.*Datafile$")


@dataclass
class SampledJson:
    """Snapshot of a sampled JSON file and its discovered camera serials."""

    path: Path
    serials: List[str]
    error: Optional[str] = None


@dataclass
class PatientSummary:
    """Aggregate results collected for one patient directory."""

    patient_id: str
    total_json_count: int
    sampled_jsons: List[SampledJson]

    @property
    def sample_size(self) -> int:
        return len(self.sampled_jsons)

    @property
    def union_serials(self) -> List[str]:
        serials: Set[str] = set()
        for entry in self.sampled_jsons:
            serials.update(entry.serials)
        return sorted(serials)

    @property
    def shared_serials(self) -> List[str]:
        shared: Optional[Set[str]] = None
        for entry in self.sampled_jsons:
            if shared is None:
                shared = set(entry.serials)
            else:
                shared &= set(entry.serials)
        return sorted(shared) if shared else []

    def to_dict(self) -> Dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "total_json_count": self.total_json_count,
            "sample_size": self.sample_size,
            "union_serials": self.union_serials,
            "shared_serials": self.shared_serials,
            "samples": [
                {
                    "path": str(entry.path),
                    "serials": entry.serials,
                    **({"error": entry.error} if entry.error else {}),
                }
                for entry in self.sampled_jsons
            ],
        }


def iter_patient_dirs(root_dir: Path) -> Iterable[Path]:
    """Yield patient directories that follow the ``Y*Datafile`` pattern."""
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and PATIENT_DIR_RE.fullmatch(child.name):
            LOGGER.debug("Discovered patient directory %s", child)
            yield child


def sample_json_paths(video_dir: Path, sample_size: int) -> tuple[int, List[Path]]:
    """
    Reservoir-sample JSON paths beneath VIDEO/YYYYMMDD folders.

    Returns the total number of JSON files discovered along with the sampled
    subset (at most ``sample_size`` entries). This avoids the memory hit of
    materialising every path when directories contain thousands of files.
    """
    reservoir: List[Path] = []
    total = 0
    for date_dir in sorted(video_dir.iterdir()):
        if not date_dir.is_dir() or not DATE_DIR_RE.fullmatch(date_dir.name):
            continue
        LOGGER.debug("Scanning date folder %s", date_dir)
        for json_path in sorted(date_dir.glob("*.json")):
            total += 1
            if len(reservoir) < sample_size:
                reservoir.append(json_path)
            else:
                idx = random.randint(0, total - 1)
                if idx < sample_size:
                    reservoir[idx] = json_path
    return total, reservoir


def parse_camera_serials(json_path: Path) -> SampledJson:
    """Extract camera serials from a JSON companion file."""
    try:
        parser = JsonParser(str(json_path))
        serials = parser.get_camera_serials() or []
        serial_strs = sorted({str(serial) for serial in serials})
        return SampledJson(path=json_path, serials=serial_strs)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed parsing %s: %s", json_path, exc)
        return SampledJson(path=json_path, serials=[], error=str(exc))


def summarize_patient_jsons(
    patient_dir: Path, sample_size: int
) -> Optional[PatientSummary]:
    """Sample JSON companions for a patient and build a summary."""
    video_dir = patient_dir / "VIDEO"
    if not video_dir.is_dir():
        LOGGER.info("Skipping %s (VIDEO directory missing)", patient_dir.name)
        return None

    total_jsons, sampled_paths = sample_json_paths(video_dir, sample_size)
    LOGGER.info(
        "Patient %s: discovered %d JSON(s); sampling %d",
        patient_dir.name,
        total_jsons,
        len(sampled_paths),
    )
    if total_jsons == 0:
        LOGGER.info("No JSON files found under %s", video_dir)
        return PatientSummary(
            patient_id=patient_dir.name,
            total_json_count=0,
            sampled_jsons=[],
        )

    sampled_entries = [parse_camera_serials(path) for path in sampled_paths]
    LOGGER.debug(
        "Patient %s: completed parsing %d sampled JSON(s)",
        patient_dir.name,
        len(sampled_entries),
    )

    return PatientSummary(
        patient_id=patient_dir.name,
        total_json_count=total_jsons,
        sampled_jsons=sampled_entries,
    )


def write_master_summary(
    summaries: Sequence[PatientSummary], output_path: Path
) -> None:
    """Write a roll-up summary that includes every patient in one file."""
    LOGGER.info("Writing master summary for %d patient(s)", len(summaries))
    payload = {summary.patient_id: summary.to_dict() for summary in summaries}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample EMU camera JSONs to report available camera serials."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/datalake/data/emu"),
        help="Recording root directory that contains patient folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("camera_serial_reports/camera_serials_summary.json"),
        help="Destination JSON path for the summary report.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of JSON companions to sample per patient (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.seed is not None:
        random.seed(args.seed)

    root_dir: Path = args.root.expanduser().resolve()
    if not root_dir.is_dir():
        LOGGER.error("Root directory not found: %s", root_dir)
        return 2

    output_path: Path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing summary to %s", output_path)

    sample_size = max(1, int(args.sample_size))

    summaries: List[PatientSummary] = []
    for patient_dir in iter_patient_dirs(root_dir):
        summary = summarize_patient_jsons(patient_dir, sample_size)
        if summary is None:
            continue
        summaries.append(summary)

    if not summaries:
        LOGGER.warning("No patient summaries generated.")
        return 0

    write_master_summary(summaries, output_path)
    LOGGER.info("Completed serial scan for %d patient(s).", len(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
