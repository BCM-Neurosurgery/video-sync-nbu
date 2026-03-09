"""Shared serial-audio preparation pipeline.

Consolidates the split → decode → merge → gapfill → filter workflow used by
``cli_nbu``, ``find_audio_abs_time``, and the WebUI decode workflow into a
single reusable function.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from scripts.decode.wavfileparser import (
    decode_to_raw,
    decode_split_dir_to_csvs,
    MP3_PYDUB_SIZE_HARD_LIMIT,
    MAX_DECODE_BYTES_DEFAULT,
)
from scripts.fix.audiogapfiller import gapfill_csv_file
from scripts.fix.audiofilter import filter_audio_file
from scripts.merge.mergecsv import merge_split_csvs
from scripts.split.mp3split import split_mp3_to_wav

_LOG = logging.getLogger(__name__)


def _needs_split(path: Path) -> bool:
    """Return True if *path* is an MP3 that exceeds the safe direct-decode cap."""
    if path.suffix.lower() != ".mp3":
        return False
    try:
        fbytes = os.path.getsize(path)
    except OSError:
        return False
    env = os.getenv("VIDEOSYNC_DECODE_MAX_BYTES")
    try:
        soft_cap = int(env) if env else MAX_DECODE_BYTES_DEFAULT
    except Exception:
        soft_cap = MAX_DECODE_BYTES_DEFAULT
    return fbytes >= soft_cap or fbytes >= MP3_PYDUB_SIZE_HARD_LIMIT


def prepare_serial_csv(
    serial_audio_path: Path,
    artifact_root: Path,
    site: str,
    *,
    skip_decode: bool = False,
    do_split: bool | None = None,
    split_chunk_seconds: int = 3600,
    split_overwrite: bool = False,
    split_clean: bool = False,
    split_outdir: Path | None = None,
    run_analysis: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    """Decode/gapfill/filter serial audio and return the filtered CSV path.

    Parameters
    ----------
    serial_audio_path:
        Path to the serial audio file (MP3 or WAV, typically channel -03).
    artifact_root:
        Root output directory.  Intermediate files are written to standard
        subdirectories (``audio_decoded/``, ``serial_audio_splitted/``,
        ``split_decoded/``).
    site:
        Decoder site preset (``jamail``, ``nbu_lounge``, ``nbu_sleep``).
    skip_decode:
        If *True*, assume decoded artifacts already exist and skip all
        decoding.  Raises :class:`FileNotFoundError` if they are missing.
    do_split:
        ``True``  – force the split→decode→merge path (MP3 only).
        ``False`` – force direct single-file decode.
        ``None``  – auto-detect based on file size and extension.
    split_chunk_seconds:
        Chunk duration for MP3 splitting (default 3600 s = 1 hour).
    split_overwrite:
        Allow ffmpeg to overwrite existing chunk files.
    split_clean:
        Delete existing chunks before splitting.
    split_outdir:
        Custom directory for WAV chunks; defaults to
        ``<artifact_root>/serial_audio_splitted``.
    run_analysis:
        If *True*, run ``analyze_csv_serials`` after each stage (raw,
        gapfilled, filtered).  Only needed by the CLI driver.
    logger:
        Optional caller-specific logger; falls back to module logger.

    Returns
    -------
    Path
        Path to the filtered CSV (``raw-gapfilled-filtered.csv``).
    """
    log = logger or _LOG

    audio_decoded_dir = artifact_root / "audio_decoded"
    audio_decoded_dir.mkdir(parents=True, exist_ok=True)

    # -- Auto-detect split need ------------------------------------------------
    if do_split is None:
        do_split = _needs_split(serial_audio_path)
        if do_split:
            log.info(
                "Auto-detected large MP3 (%s); using split→decode→merge path.",
                serial_audio_path.name,
            )

    if do_split and serial_audio_path.suffix.lower() != ".mp3":
        log.warning(
            "--split requested but serial audio is %s (%s); "
            "falling back to direct decode.",
            serial_audio_path.suffix.lower(),
            serial_audio_path.name,
        )
        do_split = False

    # -- Skip-decode path ------------------------------------------------------
    if skip_decode:
        if do_split:
            log.info("--skip-decode is set; ignoring --split.")
        decoded_raw_csv = audio_decoded_dir / "raw.csv"
        prefiltered_csv = audio_decoded_dir / "raw-gapfilled-filtered.csv"

        missing = [p for p in (decoded_raw_csv, prefiltered_csv) if not p.exists()]
        if missing:
            for path in missing:
                log.error("--skip-decode set but missing %s", path.name)
            raise FileNotFoundError(
                "Required decoded audio artifacts missing while --skip-decode."
            )

        log.info(
            "Skip decode: using %s and %s",
            decoded_raw_csv.name,
            prefiltered_csv.name,
        )
        return prefiltered_csv

    # -- Reuse existing filtered CSV -------------------------------------------
    filtered_csv = audio_decoded_dir / "raw-gapfilled-filtered.csv"
    if filtered_csv.exists() and not split_overwrite:
        log.info("Reusing existing filtered CSV: %s", filtered_csv.name)
        return filtered_csv

    # -- Split → decode → merge ------------------------------------------------
    if do_split:
        chunks_dir = split_outdir or (artifact_root / "serial_audio_splitted")
        chunks_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Splitting serial MP3 into %ds chunks at %s",
            split_chunk_seconds,
            chunks_dir.name,
        )
        split_mp3_to_wav(
            input_mp3=serial_audio_path,
            outdir=chunks_dir,
            chunk_seconds=split_chunk_seconds,
            start_number=1,
            overwrite=split_overwrite,
            clean=split_clean,
            ffmpeg_bin="ffmpeg",
            ffmpeg_loglevel="info",
        )

        manifest_path = chunks_dir / f"{serial_audio_path.stem}_manifest.json"
        if not manifest_path.exists():
            log.warning("Manifest not found: %s", manifest_path.name)
            manifest_path = None

        split_csv_dir = artifact_root / "split_decoded"
        split_csv_dir.mkdir(parents=True, exist_ok=True)

        decode_split_dir_to_csvs(
            split_dir=chunks_dir,
            outdir=split_csv_dir,
            site=site,
            threshold=0.5,
            pattern=f"{serial_audio_path.stem}-[0-9][0-9][0-9].wav",
            manifest=manifest_path,
        )

        decoded_raw_csv = merge_split_csvs(
            split_dir=split_csv_dir,
            outdir=audio_decoded_dir,
            pattern="*.csv",
            manifest=manifest_path,
            output_name="raw.csv",
            gzip_output=False,
            dedupe=True,
            tolerance_samples=0,
            logger=log,
        )
        log.info("Merged per-chunk CSVs → %s", decoded_raw_csv.name)

    # -- Direct decode ---------------------------------------------------------
    else:
        log.info("Decoding serial…")
        decoded_raw_csv, _, _, _ = decode_to_raw(
            serial_audio_path, audio_decoded_dir, site=site
        )
        log.info("Decoded audio serial → %s", decoded_raw_csv.name)

    # -- Optional analysis -----------------------------------------------------
    if run_analysis:
        _run_analysis(decoded_raw_csv, log, stage="Raw")

    # -- Gapfill ---------------------------------------------------------------
    gapfilled_csv = gapfill_csv_file(input_csv=decoded_raw_csv)
    log.info("Gap-filled %s → %s", decoded_raw_csv.name, gapfilled_csv.name)

    if run_analysis:
        _run_analysis(gapfilled_csv, log, stage="Gap-filled")

    # -- Filter ----------------------------------------------------------------
    filtered_csv = filter_audio_file(input_csv=gapfilled_csv)
    log.info("Filtered %s → %s", gapfilled_csv.name, filtered_csv.name)

    if run_analysis:
        _run_analysis(filtered_csv, log, stage="Filtered")

    return filtered_csv


def _run_analysis(csv_path: Path, log: logging.Logger, *, stage: str) -> None:
    """Run serial analysis on *csv_path*, logging warnings on failure."""
    try:
        from scripts.analysis.csv_serial_analysis import analyze_csv_serials

        _, txt = analyze_csv_serials(path=csv_path)
        log.info("%s analysis → %s", stage, txt.name)
    except Exception as exc:
        log.error("%s analysis failed: %s", stage, exc)
