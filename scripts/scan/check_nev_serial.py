"""Quick check to ensure NSP1 NEV files carry serial data in digital events.

Usage
-----
CLI example to scan up to five NEV files inside a directory tree::

    python -m scripts.scan.check_nev_serial --dir /path/to/task --limit 5

The command returns exit code 0 when every sampled file contains serial data
(i.e., ``Nev.has_unparsed_data()`` is true) and non-zero otherwise. The module
also exposes :func:`check_nev_serial` for programmatic use.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
from typing import Iterable, Iterator, List, Sequence, Tuple
from itertools import islice

from scripts.parsers.nevfileparser import Nev

LOGGER = logging.getLogger(__name__)


def _iter_nev_files(root: Path) -> Iterator[Path]:
    """Yield NEV files under ``root`` lazily (case-insensitive NSP1 prefix)."""

    def matches(path: Path) -> bool:
        return path.name.lower().startswith("nsp1-") and path.suffix.lower() == ".nev"

    if root.is_file():
        if matches(root):
            yield root
        return

    for candidate in root.rglob("*.nev"):
        if matches(candidate):
            yield candidate


def _check_file(path: Path, *, preview_count: int = 3) -> Tuple[bool, str]:
    """Return (has_serial, message) for a NEV file."""

    try:
        nev = Nev(str(path))
        has_serial = bool(nev.has_unparsed_data())
        if not has_serial:
            return False, "digital_events lacks serial data"

        try:
            chunk_df = nev.get_chunk_serial_df()
            previews = chunk_df["chunk_serial"].tolist()[:preview_count]
        except Exception:
            previews = []

        if previews:
            preview_str = ", ".join(str(val) for val in previews)
            LOGGER.info("%s: first serial values %s", path.name, preview_str)
        else:
            LOGGER.info("%s: serial data present (no preview available)", path.name)

        return True, "serial data present"
    except Exception as exc:  # pragma: no cover - defensive guard
        return False, f"error parsing NEV: {exc}"


def check_nev_serial(
    directory: Path, *, limit: int = 5
) -> Tuple[bool, List[Tuple[Path, bool, str]]]:
    """Scan up to ``limit`` NEV files and verify they contain serial data."""

    files_iter = _iter_nev_files(directory)
    selected = list(islice(files_iter, max(1, limit)))
    if not selected:
        raise FileNotFoundError(f"No NSP1-*.nev files found under {directory}")

    results: List[Tuple[Path, bool, str]] = []
    overall = True
    for nev_path in selected:
        ok, message = _check_file(nev_path)
        results.append((nev_path, ok, message))
        if not ok:
            overall = False
    return overall, results


def _format_result(result: Tuple[Path, bool, str]) -> str:
    path, ok, message = result
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {path} — {message}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check NEV files for serial data")
    parser.add_argument(
        "--dir", type=Path, required=True, help="Directory containing NEV files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of files to inspect (default: 5)",
    )
    args = parser.parse_args(argv)

    if not LOGGER.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        passed, results = check_nev_serial(args.dir, limit=args.limit)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 2

    for entry in results:
        print(_format_result(entry))

    if passed:
        LOGGER.info("All %d NEV files contain serial data", len(results))
        return 0
    LOGGER.error("Detected NEV files missing serial data")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
