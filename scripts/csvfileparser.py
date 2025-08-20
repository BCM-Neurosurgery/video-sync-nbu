#!/usr/bin/env python3
"""
csvfileparser.py — Split a CSV into evenly sized parts.

Usage:
  python csvfileparser.py path/to/file.csv --parts 4

Writes next to the input:
  file-01.csv, file-02.csv, ...

Notes:
  • The header row (first row) is preserved in every output file.
  • Splits are as even as possible; earlier parts may have +1 row if needed.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class CSVFileParser:
    """Lightweight CSV splitter (streaming; no pandas dependency)."""

    def __init__(self, path: str | Path, *, encoding: str = "utf-8") -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")
        self.path: Path = p
        self.encoding = encoding

    def _count_rows(self) -> int:
        """Count *data* rows (excludes the header)."""
        with self.path.open("r", encoding=self.encoding, newline="") as f:
            reader = csv.reader(f)
            try:
                next(reader)  # header
            except StopIteration:
                return 0  # empty file
            total = sum(1 for _ in reader)
            logger.debug("Counted %d data rows in %s", total, self.path)
            return total

    def _out_path(self, idx1: int, n: int) -> Path:
        """Build output path like 'name-01.csv' in the same folder (min 2 digits)."""
        stem = self.path.stem
        ext = self.path.suffix or ".csv"
        width = max(2, len(str(n)))
        suffix = f"-{idx1:0{width}d}{ext}"
        return self.path.with_name(f"{stem}{suffix}")

    def split_evenly(self, parts: int) -> List[Path]:
        """Split into `parts` files as evenly as possible. Returns list of output paths."""
        if parts < 1:
            raise ValueError("parts must be >= 1")
        logger.info("Splitting %s into %d part(s)", self.path.name, parts)

        total_rows = self._count_rows()

        # Compute target sizes per part (sum == total_rows)
        base = total_rows // parts
        remainder = total_rows % parts
        targets = [base + (1 if i < remainder else 0) for i in range(parts)]
        logger.debug("Split plan (data rows per part): %s", targets)
        if total_rows < parts:
            logger.warning(
                "Fewer data rows (%d) than parts (%d); some outputs will contain only headers.",
                total_rows,
                parts,
            )

        outputs: List[Path] = []
        with self.path.open("r", encoding=self.encoding, newline="") as f_in:
            reader = csv.reader(f_in)
            try:
                header = next(reader)
            except StopIteration:
                header = []  # empty file; emit empty parts with no header

            for i, need in enumerate(targets, start=1):
                out_path = self._out_path(i, parts)
                logger.info(
                    "[%d/%d] writing %s (%d rows)", i, parts, out_path.name, need
                )
                with out_path.open("w", encoding=self.encoding, newline="") as f_out:
                    writer = csv.writer(f_out)
                    if header:
                        writer.writerow(header)
                    for _ in range(need):
                        try:
                            row = next(reader)
                        except StopIteration:
                            break
                        writer.writerow(row)
                outputs.append(out_path)

        return outputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Split a CSV into evenly sized parts.")
    ap.add_argument("csv_file", type=Path, help="Path to the input .csv")
    ap.add_argument(
        "--parts",
        "-p",
        type=int,
        default=2,
        help="Number of parts to split into (default: 2)",
    )
    ap.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Input/output text encoding (default: utf-8)",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        parser = CSVFileParser(args.csv_file, encoding=args.encoding)
        outputs = parser.split_evenly(args.parts)
    except Exception:
        logger.exception("Failed to split CSV")
        return 1

    # Single concise summary (no duplicate per-file listing)
    logger.info("Done. Created %d file(s) in %s", len(outputs), parser.path.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
