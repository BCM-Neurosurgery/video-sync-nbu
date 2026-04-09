"""Load and query the stitched -> first-chunk NEV mapping from a JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)

MAP_FILENAME = "first_chunk_map.json"


def load_first_chunk_map(map_path: Path) -> Dict[str, str]:
    """Load the ``"map"`` dict from a first_chunk_map.json file.

    Returns
    -------
    dict[str, str]
        Mapping of stitched NEV relative POSIX paths to chunk NEV relative
        POSIX paths.

    Raises
    ------
    FileNotFoundError
        If *map_path* does not exist.
    RuntimeError
        If the file is malformed or missing the ``"map"`` key.
    """
    if not map_path.is_file():
        raise FileNotFoundError(f"First-chunk map not found: {map_path}")
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load JSON {map_path}: {exc}") from exc
    if not isinstance(data, dict) or "map" not in data:
        raise RuntimeError(
            f"Malformed first-chunk map {map_path}: "
            "expected a JSON object with a 'map' key."
        )
    mapping = data["map"]
    if not isinstance(mapping, dict):
        raise RuntimeError(
            f"Malformed first-chunk map {map_path}: 'map' must be a JSON object."
        )
    LOGGER.debug(
        "Loaded first-chunk map with %d entries from %s", len(mapping), map_path
    )
    return mapping


def resolve_from_map(
    mapping: Dict[str, str],
    stitched_nev: Path,
    stitched_base: Path,
    chunk_base: Path,
) -> Optional[Path]:
    """Look up the first-chunk NEV for a stitched NEV in a pre-loaded map.

    Parameters
    ----------
    mapping
        The dict returned by :func:`load_first_chunk_map`.
    stitched_nev
        Absolute path to the stitched NEV file.
    stitched_base
        Root directory of the stitched files (typically the map file's parent).
    chunk_base
        Root directory of the chunk/datalake files.

    Returns
    -------
    Path or None
        Absolute path to the first chunk NEV, or ``None`` if not in the map.
    """
    try:
        rel = stitched_nev.resolve().relative_to(stitched_base.resolve())
    except ValueError:
        LOGGER.debug(
            "Stitched NEV %s is not under stitched_base %s",
            stitched_nev,
            stitched_base,
        )
        return None
    key = PurePosixPath(rel).as_posix()
    chunk_rel = mapping.get(key)
    if chunk_rel is None:
        return None
    return (chunk_base / chunk_rel).resolve(strict=False)


def auto_detect_map(patient_dir: Path) -> Optional[Path]:
    """Attempt to find first_chunk_map.json in *patient_dir*'s parent.

    Convention: ``patient_dir`` is ``<stitched_base>/<patient>``, so the map
    lives at ``<stitched_base>/first_chunk_map.json``.

    Returns ``None`` if the file does not exist.
    """
    candidate = patient_dir.resolve().parent / MAP_FILENAME
    if candidate.is_file():
        LOGGER.debug("Auto-detected first-chunk map: %s", candidate)
        return candidate
    return None
